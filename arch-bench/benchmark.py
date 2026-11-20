"""Empirical benchmark of 5 neural architectures for 15x15 Renju Gomoku using rustAutograd (rg)."""

from __future__ import annotations

import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))  # arch-bench/
sys.path.insert(0, _HERE)
import rg
from rg import nn, optim, random as rg_random

from renju_env import (
    BLACK,
    BOARD_SIZE,
    NUM_CELLS,
    WHITE,
    RenjuBoard,
    generate_dataset,
    generate_seeded_board,
    opponent_of,
    rule_based_move,
)
from rg_models import (
    AxialOmokNet,
    BlurConvPatternNet,
    DualPathOmokNet,
    FastLine1x5Net,
    FlatMLPNet,
    LineAxialHybridNet,
    LineConvNet,
    OverlapPatternNet,
    PatternTableNet,
    ResNet3x3Net,
    count_params,
)


def dihedral_transform_batch(
    states_nchw: np.ndarray, policies: np.ndarray, sym: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies dihedral symmetry (0..7) to (B, 6, 15, 15) and (B, 225). Returns (B, 225, 6), (B, 225)."""
    k = sym % 4
    flip = sym >= 4
    s = np.rot90(states_nchw, k=k, axes=(2, 3))
    p = np.rot90(policies.reshape(-1, BOARD_SIZE, BOARD_SIZE), k=k, axes=(1, 2))
    if flip:
        s = np.flip(s, axis=3)
        p = np.flip(p, axis=2)
    s = np.ascontiguousarray(np.transpose(s, (0, 2, 3, 1)).reshape(-1, NUM_CELLS, 6))
    p = np.ascontiguousarray(p.reshape(-1, NUM_CELLS))
    return s, p


def to_nhwc_flat(states_nchw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(states_nchw, (0, 2, 3, 1)).reshape(-1, NUM_CELLS, 6))


def play_match_game(
    seed_board: RenjuBoard,
    start_player: int,
    agent_black,
    agent_white,
    max_plies: int = 120,
) -> int:
    """Plays from a seeded Renju board. Returns BLACK, WHITE, or 0 (draw)."""
    b = seed_board.clone()
    player = start_player
    for _ in range(max_plies):
        agent = agent_black if player == BLACK else agent_white
        mv = agent(b, player)
        if mv is None or not b.is_legal_move(mv, player):
            # Illegal / no move loses
            return opponent_of(player)
        b.place_move(mv, player)
        if b.is_win_at(mv, player):
            return player
        player = opponent_of(player)
    return 0


def make_nn_agent(infer_1, infer_k, params, lookahead: bool = False, top_k: int = 6):
    def select_move(board: RenjuBoard, player: int):
        mask = board.legal_mask(player)
        legal_idx = np.flatnonzero(mask)
        if len(legal_idx) == 0:
            return None

        # Always take immediate 5-win if present (keeps comparison focused on strategy & tactical reading)
        for idx in legal_idx:
            board.cells[idx] = player
            won = board.is_win_at(idx, player)
            board.cells[idx] = 0
            if won:
                return int(idx)

        t = board.encode_tensor(player, mask)
        x1 = to_nhwc_flat(t[None, ...])
        logits_t, _ = infer_1(params, x1)
        logits = np.asarray(logits_t.numpy(), dtype=np.float32)[0]
        logits[~mask] = -1e9

        if not lookahead:
            return int(np.argmax(logits))

        # 1-ply lookahead over top_k policy candidates using Value head
        k = min(top_k, len(legal_idx))
        cand_indices = np.argpartition(logits, -k)[-k:]
        batch_x = np.zeros((top_k, NUM_CELLS, 6), dtype=np.float32)
        opp = opponent_of(player)
        valid_cands: List[int] = []
        for i, mv in enumerate(cand_indices):
            mv = int(mv)
            if not mask[mv]:
                continue
            valid_cands.append(mv)
            cb = board.clone()
            cb.place_move(mv, player)
            ct = cb.encode_tensor(opp)
            batch_x[len(valid_cands) - 1] = to_nhwc_flat(ct[None, ...])[0]
        if not valid_cands:
            return int(np.argmax(logits))

        _, vals_t = infer_k(params, batch_x)
        opp_vals = np.asarray(vals_t.numpy(), dtype=np.float32)[: len(valid_cands)]
        # Combine value from opponent's loss (-opp_val) with slight policy prior tie-break
        best_pos = 0
        best_score = -1e9
        for i, mv in enumerate(valid_cands):
            score = float(-opp_vals[i]) + 0.05 * float(logits[mv])
            if score > best_score:
                best_score = score
                best_pos = i
        return valid_cands[best_pos]

    return select_move


def main() -> None:
    print("=== 15x15 Renju Neural Architecture Benchmark (rustAutograd) ===")
    t0 = time.time()
    n_train = 1200
    n_val = 384  # multiple of 32 for clean batched eval
    cache_path = os.path.join(_HERE, "dataset_2026.npz")
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        all_x, all_pi, all_z, all_best, all_crit = d["x"], d["pi"], d["z"], d["best"], d["crit"]
    else:
        all_x, all_pi, all_z, all_best, all_crit = generate_dataset(n_train + n_val, seed=2026)
        np.savez_compressed(cache_path, x=all_x, pi=all_pi, z=all_z, best=all_best, crit=all_crit)
    tr_x, tr_pi, tr_z = all_x[:n_train], all_pi[:n_train], all_z[:n_train]
    va_x, va_pi, va_z, va_best, va_crit = (
        all_x[n_train:],
        all_pi[n_train:],
        all_z[n_train:],
        all_best[n_train:],
        all_crit[n_train:],
    )
    va_x_nhwc = to_nhwc_flat(va_x)
    print(
        f"Dataset ready in {time.time() - t0:.2f}s | Train={n_train} (8x dihedral) | Val={n_val} (critical tactical={int(va_crit.sum())})"
    )

    # Pre-build training mini-batches (320 steps of B=32 with random 8x dihedral symmetry)
    B = 32
    steps = 320
    py_rng = random.Random(777)
    batches: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _ in range(steps):
        idxs = [py_rng.randrange(n_train) for _ in range(B)]
        sym = py_rng.randrange(8)
        xb, pib = dihedral_transform_batch(tr_x[idxs], tr_pi[idxs], sym)
        zb = tr_z[idxs].astype(np.float32)
        batches.append((xb, pib, zb))

    specs = [
        ("LineAxialHybrid", lambda: LineAxialHybridNet(width=48, stem_ch=16, line_blocks=2, axial_blocks=1, heads=4)),
        ("PatternTableNNUE", lambda: PatternTableNet(d_ray=24, width=64, blocks=3)),
        ("BlurPattern", lambda: BlurConvPatternNet(d_ray=24, width=64, blocks=3)),
        ("OverlapPattern", lambda: OverlapPatternNet(d_ray=24, d_outer=8, width=64, blocks=3, use_blur=False)),
        ("FoveaOverlap", lambda: OverlapPatternNet(d_ray=24, d_outer=8, width=64, blocks=3, use_blur=True)),
    ]

    trained: Dict[str, dict] = {}

    print("\n--- Phase 1: Supervised Tactical & Positional Training (320 steps x B=32) ---")
    print(
        f"{'Arch':16s} | {'Params':>7s} | {'Train(s)':>8s} | {'ValPLoss':>8s} | {'Top1':>6s} | {'Top3':>6s} | {'CritT1':>6s} | {'ValMSE':>7s} | {'Inf(b/s)':>8s}"
    )
    print("-" * 94)

    for label, ctor in specs:
        model = ctor()
        p0 = model.init(rg_random.key(42))
        n_params = count_params(p0)
        schedule = optim.cosine_decay(2.8e-3, total_steps=steps, warmup_steps=20, min_lr=2e-4)
        opt = optim.adamw(schedule, weight_decay=1e-4, max_grad_norm=1.0)
        st0 = opt.init(p0)

        def train_step(state, batch):
            pr_in, st_in = state
            xb, pib, zb = batch

            def loss_fn(pr):
                logits, val = model(pr, xb)
                lp = nn.log_softmax(logits, axis=-1)
                pl = -(pib * lp).sum(axis=-1).mean()
                vl = ((val - zb) ** 2).mean()
                return pl + 0.6 * vl

            loss, g = rg.value_and_grad(loss_fn)(pr_in)
            pr_out, st_out = opt.update(g, st_in, pr_in)
            return (pr_out, st_out), loss

        def infer_fn(pr, xb):
            return model(pr, xb)

        c_train = rg.compile(train_step, ((p0, st0), batches[0]), device="cpu", validate="basic")
        c_eval_b = rg.compile(infer_fn, (p0, va_x_nhwc[:B]), device="cpu", validate="basic")
        c_eval_1 = rg.compile(infer_fn, (p0, va_x_nhwc[:1]), device="cpu", validate="basic")
        c_eval_6 = rg.compile(infer_fn, (p0, va_x_nhwc[:6]), device="cpu", validate="basic")

        t_tr = time.time()
        res = rg.run(c_train, (p0, st0), batches, steps=steps)
        train_sec = time.time() - t_tr
        p_trained, _ = res.state

        # Evaluate on unseen validation boards
        t_inf = time.time()
        all_logits = []
        all_vals = []
        for off in range(0, n_val, B):
            lg, vl = c_eval_b(p_trained, va_x_nhwc[off : off + B])
            all_logits.append(np.asarray(lg.numpy(), dtype=np.float32))
            all_vals.append(np.asarray(vl.numpy(), dtype=np.float32))
        inf_sec = max(1e-6, time.time() - t_inf)
        boards_per_sec = n_val / inf_sec

        logits_np = np.concatenate(all_logits, axis=0)
        vals_np = np.concatenate(all_vals, axis=0)

        # Metrics
        max_l = np.max(logits_np, axis=1, keepdims=True)
        log_probs = logits_np - max_l - np.log(np.sum(np.exp(logits_np - max_l), axis=1, keepdims=True))
        val_ploss = float(-np.mean(np.sum(va_pi * log_probs, axis=1)))
        val_mse = float(np.mean((vals_np - va_z) ** 2))

        pred1 = np.argmax(logits_np, axis=1)
        top1_acc = float(np.mean(pred1 == va_best))
        top3_idx = np.argpartition(logits_np, -3, axis=1)[:, -3:]
        top3_acc = float(np.mean([va_best[i] in top3_idx[i] for i in range(n_val)]))
        crit_acc = float(np.mean(pred1[va_crit] == va_best[va_crit])) if va_crit.any() else 0.0

        print(
            f"{label:14s} | {n_params:7d} | {train_sec:7.2f}s | {val_ploss:8.3f} | {top1_acc*100:5.1f}% | {top3_acc*100:5.1f}% | {crit_acc*100:5.1f}% | {val_mse:7.3f} | {boards_per_sec:8.0f}"
        )

        trained[label] = {
            "model": model,
            "params": p_trained,
            "eval_1": c_eval_1,
            "eval_6": c_eval_6,
        }

    # Seeded opening boards for Arena
    arena_rng = random.Random(999)
    openings: List[Tuple[RenjuBoard, int]] = []
    while len(openings) < 10:
        seeded = generate_seeded_board(arena_rng, min_stones=4, max_stones=10)
        if seeded is not None:
            openings.append(seeded)

    print("\n--- Phase 2: Arena vs Rule-Based Bot (20 games each: 10 seeded boards x B/W) ---")
    rb_weak_rng = random.Random(101)
    rb_weak = lambda b, pl: rule_based_move(b, pl, noise_eps=0.30, rng=rb_weak_rng)
    rb_strong = lambda b, pl: rule_based_move(b, pl, noise_eps=0.0, rng=None)

    for mode_name, lookahead in [("0-ply Pure NN", False), ("1-ply Value Lookahead", True)]:
        print(f"\n[{mode_name}]")
        for label in [s[0] for s in specs]:
            info = trained[label]
            nn_agent = make_nn_agent(info["eval_1"], info["eval_6"], info["params"], lookahead=lookahead)
            w_weak, d_weak, l_weak = 0, 0, 0
            w_str, d_str, l_str = 0, 0, 0
            for sb, sp in openings:
                # NN as starter side
                res1 = play_match_game(sb, sp, nn_agent if sp == BLACK else rb_weak, rb_weak if sp == BLACK else nn_agent)
                if res1 == sp:
                    w_weak += 1
                elif res1 == 0:
                    d_weak += 1
                else:
                    l_weak += 1
                # NN as other side
                res2 = play_match_game(sb, sp, rb_weak if sp == BLACK else nn_agent, nn_agent if sp == BLACK else rb_weak)
                if res2 == opponent_of(sp):
                    w_weak += 1
                elif res2 == 0:
                    d_weak += 1
                else:
                    l_weak += 1

                # vs Strong Rule-Based
                res3 = play_match_game(sb, sp, nn_agent if sp == BLACK else rb_strong, rb_strong if sp == BLACK else nn_agent)
                if res3 == sp:
                    w_str += 1
                elif res3 == 0:
                    d_str += 1
                else:
                    l_str += 1
                res4 = play_match_game(sb, sp, rb_strong if sp == BLACK else nn_agent, nn_agent if sp == BLACK else rb_strong)
                if res4 == opponent_of(sp):
                    w_str += 1
                elif res4 == 0:
                    d_str += 1
                else:
                    l_str += 1

            wr_w = (w_weak + 0.5 * d_weak) / 20.0 * 100.0
            wr_s = (w_str + 0.5 * d_str) / 20.0 * 100.0
            print(
                f"  {label:14s} | vs Noisy-RB (W/D/L): {w_weak:2d}/{d_weak:2d}/{l_weak:2d} ({wr_w:5.1f}%) | vs Full-RB: {w_str:2d}/{d_str:2d}/{l_str:2d} ({wr_s:5.1f}%)"
            )

    print("\n--- Phase 3: Heterogeneous Neural League (Round-Robin, 0-ply Pure NN, 12 games/pair) ---")
    labels = [s[0] for s in specs]
    league_openings = openings[:6]
    scores = {k: 0.0 for k in labels}
    wins_vs = {k: {o: 0.0 for o in labels} for k in labels}

    agents_0ply = {
        k: make_nn_agent(trained[k]["eval_1"], trained[k]["eval_6"], trained[k]["params"], lookahead=False)
        for k in labels
    }

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b_name = labels[i], labels[j]
            ag_a, ag_b = agents_0ply[a], agents_0ply[b_name]
            pts_a, pts_b = 0.0, 0.0
            for sb, sp in league_openings:
                # A starts
                r1 = play_match_game(sb, sp, ag_a if sp == BLACK else ag_b, ag_b if sp == BLACK else ag_a)
                if r1 == sp:
                    pts_a += 1.0
                elif r1 == 0:
                    pts_a += 0.5
                    pts_b += 0.5
                else:
                    pts_b += 1.0
                # B starts
                r2 = play_match_game(sb, sp, ag_b if sp == BLACK else ag_a, ag_a if sp == BLACK else ag_b)
                if r2 == sp:
                    pts_b += 1.0
                elif r2 == 0:
                    pts_a += 0.5
                    pts_b += 0.5
                else:
                    pts_a += 1.0
            scores[a] += pts_a
            scores[b_name] += pts_b
            wins_vs[a][b_name] = pts_a
            wins_vs[b_name][a] = pts_b

    ranking = sorted(labels, key=lambda k: scores[k], reverse=True)
    for rank, k in enumerate(ranking, 1):
        max_pts = 12.0 * (len(labels) - 1)
        vs_str = " | ".join(f"{o[:7]}:{wins_vs[k][o]:4.1f}" for o in labels if o != k)
        print(f"  #{rank} {k:16s} | {scores[k]:4.1f}/{max_pts:.0f} ({scores[k]/max_pts*100:5.1f}%) | {vs_str}")


if __name__ == "__main__":
    main()
