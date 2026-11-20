"""Trains D4OverlapPatternNet with Metal GPU MCTS self-play and exports rich replays for the web UI."""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))  # arch-bench/
sys.path.insert(0, _HERE)
import rg
from rg import nn, optim, random as rg_random

from alphazero_mcts import (
    MCTSNode,
    encode_board_row,
    eval_validation_metal,
    exact_legal_mask,
    run_mcts_batch,
    sample_batch_from_buffer,
    to_nhwc_flat,
)
from renju_env import (
    BLACK,
    BOARD_SIZE,
    EMPTY,
    NUM_CELLS,
    WHITE,
    RenjuBoard,
    generate_dataset,
    opponent_of,
    rule_based_move,
)
from rg_models import D4OverlapPatternNet, count_params


def format_top_policies(probs: np.ndarray, k: int = 12) -> List[dict]:
    ranked = np.argsort(-probs)
    out = []
    for idx in ranked[:k]:
        p = float(probs[int(idx)])
        if p <= 0.005:
            break
        r, c = divmod(int(idx), BOARD_SIZE)
        out.append({"row": r, "col": c, "prob": round(p, 3)})
    return out


def masked_softmax_logits(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m_logits = np.where(mask, logits, -1e9)
    max_v = float(np.max(m_logits))
    exps = np.where(mask, np.exp(np.clip(m_logits - max_v, -30.0, 0.0)), 0.0)
    s = float(exps.sum())
    if s <= 1e-12:
        cnt = max(1, int(mask.sum()))
        return (mask.astype(np.float32) / float(cnt))
    return (exps / s).astype(np.float32)


def mcts_root_value(root: MCTSNode, fallback: float = 0.0) -> float:
    if root.win_move is not None:
        return 0.99
    if root.N_sum > 0.0:
        return float(np.clip(float(root.W.sum()) / root.N_sum, -0.99, 0.99))
    return float(fallback)


def run_recorded_selfplay_gen(
    gen: int,
    n_games: int,
    params,
    c_eval_32,
    n_sims: int,
    rng: random.Random,
    max_plies: int = 85,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], List[dict], Tuple[int, int, int, float]]:
    games = []
    for _ in range(n_games):
        b = RenjuBoard()
        n_open = rng.randint(0, 2)
        games.append({
            "board": b,
            "root": MCTSNode(BLACK),
            "player": BLACK,
            "step": 0,
            "n_open": n_open,
            "traj": [],
            "replay": [],
            "done": False,
            "winner": 0,
        })

    buf_x = np.zeros((32, NUM_CELLS, 6), dtype=np.float32)

    while True:
        active = [g for g in games if not g["done"]]
        if not active:
            break

        # Evaluate raw root value/logits once for opening display; MCTS runs on all active games
        for idx, g in enumerate(active):
            encode_board_row(g["board"], g["player"], exact_legal_mask(g["board"], g["player"]), buf_x[idx])
        lg_t, val_t = c_eval_32(params, buf_x)
        raw_logits = np.asarray(lg_t.numpy(), dtype=np.float32)
        raw_vals = np.asarray(val_t.numpy(), dtype=np.float32)

        mcts_games = [g for g in active if g["step"] >= g["n_open"]]
        if mcts_games:
            run_mcts_batch(
                mcts_games,
                params,
                c_eval_32,
                batch_size=32,
                n_sims=n_sims,
                c_puct=1.5,
                add_noise=True,
                temp_moves=8,
            )

        for idx, g in enumerate(active):
            b: RenjuBoard = g["board"]
            pl = g["player"]
            step = g["step"]
            if step < g["n_open"]:
                mask = exact_legal_mask(b, pl)
                if step == 0 and rng.random() < 0.65:
                    mv = 7 * BOARD_SIZE + 7
                else:
                    cands = [
                        r * BOARD_SIZE + c
                        for r in range(5, 10)
                        for c in range(5, 10)
                        if mask[r * BOARD_SIZE + c]
                    ]
                    mv = rng.choice(cands) if cands else int(np.flatnonzero(mask)[0])
                nn_probs = masked_softmax_logits(raw_logits[idx], mask)
                r, c = divmod(mv, BOARD_SIZE)
                g["replay"].append({
                    "row": r,
                    "col": c,
                    "player": pl,
                    "moverType": "opening",
                    "value": round(float(raw_vals[idx]), 3),
                    "topPolicies": format_top_policies(nn_probs),
                })
            else:
                mv = g["chosen"]
                if mv < 0 or not b.is_legal_move(mv, pl):
                    g["done"] = True
                    g["winner"] = opponent_of(pl)
                    continue
                s_nhwc = np.empty((NUM_CELLS, 6), dtype=np.float32)
                encode_board_row(b, pl, None, s_nhwc)
                g["traj"].append((s_nhwc, g["pi"], pl))
                mv_val = mcts_root_value(g["root"], float(raw_vals[idx]))
                r, c = divmod(mv, BOARD_SIZE)
                g["replay"].append({
                    "row": r,
                    "col": c,
                    "player": pl,
                    "moverType": "nn",
                    "value": round(mv_val, 3),
                    "topPolicies": format_top_policies(g["pi"]),
                })

            b.place_move(mv, pl)
            g["step"] += 1
            if b.is_win_at(mv, pl):
                g["done"] = True
                g["winner"] = pl
            elif g["step"] >= max_plies:
                g["done"] = True
                g["winner"] = 0
            else:
                nxt = opponent_of(pl)
                g["player"] = nxt
                g["root"] = g["root"].children.get(mv, MCTSNode(nxt))

    samples: List[Tuple[np.ndarray, np.ndarray, float]] = []
    recorded_games: List[dict] = []
    bw, ww, dr, tot_m = 0, 0, 0, 0
    for idx, g in enumerate(games):
        w = g["winner"]
        tot_m += g["step"]
        if w == BLACK:
            bw += 1
        elif w == WHITE:
            ww += 1
        else:
            dr += 1
        for s_nhwc, pi, mover in g["traj"]:
            z = 0.0 if w == 0 else (1.0 if w == mover else -1.0)
            samples.append((s_nhwc, pi, z))
        recorded_games.append({
            "winner": w,
            "moves": g["replay"],
            "generation": gen,
        })
    return samples, recorded_games, (bw, ww, dr, tot_m / float(max(1, n_games)))


def run_recorded_mcts128_vs_full_rb(
    params,
    c_eval_32,
    n_sims: int = 128,
    max_plies: int = 110,
) -> List[dict]:
    """Plays 8 full games (4 standard openings x Black/White) of MCTS-128 vs Full-RB with complete move metadata."""
    # 4 genuinely D4-distinct 3-move openings from center
    opening_pairs = [
        ("직선-꺾임", [(7, 7), (6, 7), (6, 8)]),
        ("대각-교차", [(7, 7), (6, 8), (6, 6)]),
        ("대각-확장", [(7, 7), (6, 8), (8, 8)]),
        ("직선-전면", [(7, 7), (6, 7), (7, 9)]),
    ]
    slots = []
    for op_name, seq in opening_pairs:
        for nn_side in (BLACK, WHITE):
            slots.append({
                "op": op_name,
                "seq": [r * BOARD_SIZE + c for r, c in seq],
                "board": RenjuBoard(),
                "root": MCTSNode(BLACK),
                "player": BLACK,
                "nn_side": nn_side,
                "step": 0,
                "replay": [],
                "done": False,
                "winner": 0,
            })

    buf_x = np.zeros((32, NUM_CELLS, 6), dtype=np.float32)

    while True:
        active = [g for g in slots if not g["done"]]
        if not active:
            break

        for idx, g in enumerate(active):
            encode_board_row(g["board"], g["player"], exact_legal_mask(g["board"], g["player"]), buf_x[idx])
        lg_t, val_t = c_eval_32(params, buf_x)
        raw_logits = np.asarray(lg_t.numpy(), dtype=np.float32)
        raw_vals = np.asarray(val_t.numpy(), dtype=np.float32)

        nn_active = [
            g for g in active
            if g["step"] >= len(g["seq"]) and g["player"] == g["nn_side"]
        ]
        if nn_active:
            for g in nn_active:
                if g["root"].player != g["player"]:
                    g["root"] = MCTSNode(g["player"])
            run_mcts_batch(
                nn_active,
                params,
                c_eval_32,
                batch_size=32,
                n_sims=n_sims,
                c_puct=1.5,
                add_noise=False,
                temp_moves=0,
            )

        for idx, g in enumerate(active):
            b: RenjuBoard = g["board"]
            pl = g["player"]
            step = g["step"]

            if step < len(g["seq"]):
                mv = g["seq"][step]
                mask = exact_legal_mask(b, pl)
                nn_probs = masked_softmax_logits(raw_logits[idx], mask)
                r, c = divmod(mv, BOARD_SIZE)
                g["replay"].append({
                    "row": r,
                    "col": c,
                    "player": pl,
                    "moverType": "opening",
                    "value": round(float(raw_vals[idx]), 3),
                    "topPolicies": format_top_policies(nn_probs),
                })
            elif pl == g["nn_side"]:
                mv = g["chosen"]
                if mv < 0 or not b.is_legal_move(mv, pl):
                    g["done"] = True
                    g["winner"] = opponent_of(pl)
                    continue
                mv_val = mcts_root_value(g["root"], float(raw_vals[idx]))
                r, c = divmod(mv, BOARD_SIZE)
                g["replay"].append({
                    "row": r,
                    "col": c,
                    "player": pl,
                    "moverType": "nn",
                    "value": round(mv_val, 3),
                    "topPolicies": format_top_policies(g["pi"]),
                })
            else:
                mv = rule_based_move(b, pl, noise_eps=0.0, rng=None)
                if mv is None or not b.is_legal_move(mv, pl):
                    g["done"] = True
                    g["winner"] = opponent_of(pl)
                    continue
                mask = exact_legal_mask(b, pl)
                nn_probs = masked_softmax_logits(raw_logits[idx], mask)
                r, c = divmod(mv, BOARD_SIZE)
                g["replay"].append({
                    "row": r,
                    "col": c,
                    "player": pl,
                    "moverType": "rule",
                    "value": round(float(raw_vals[idx]), 3),
                    "topPolicies": format_top_policies(nn_probs),
                })

            b.place_move(mv, pl)
            g["step"] += 1
            if b.is_win_at(mv, pl):
                g["done"] = True
                g["winner"] = pl
            elif g["step"] >= max_plies:
                g["done"] = True
                g["winner"] = 0
            else:
                nxt = opponent_of(pl)
                g["player"] = nxt
                g["root"] = g["root"].children.get(mv, MCTSNode(nxt))

    return slots


def main() -> None:
    refresh_only = "--refresh-showcase" in sys.argv
    print("=== Export Metal GPU D4OverlapPattern + MCTS Web Showcase Replays ===")
    rng = random.Random(20261119)
    np.random.seed(20261119)

    model = D4OverlapPatternNet(d_ray=24, d_outer=8, width=64, blocks=3)
    p0 = model.init(rg_random.key(42))
    n_params = count_params(p0)

    B = 64
    dummy_x32 = np.zeros((32, NUM_CELLS, 6), dtype=np.float32); dummy_x32[:, :, 4] = 1.0
    def infer_fn(pr, xb):
        return model(pr, xb)
    c_eval_32 = rg.compile(infer_fn, (p0, dummy_x32), device="metal", validate="basic")

    ckpt_path = os.path.join(_HERE, "d4_overlap_az.npz")
    out_path = os.path.join(_HERE, "web_data", "rl_replays.json")

    if refresh_only and os.path.exists(ckpt_path) and os.path.exists(out_path):
        ckpt = np.load(ckpt_path)
        leaves = [rg.tensor(ckpt[f"p_{i}"]) for i in range(len(rg.tree.leaves(p0)))]
        p_rl = rg.tree.unflatten(rg.tree.structure(p0), leaves)
        with open(out_path, "r", encoding="utf-8") as f:
            prev_archive = json.load(f)
        metrics = prev_archive.get("metrics", [])
        archived_games = [g for g in prev_archive.get("games", []) if g.get("id", "").startswith("az-gen-")]
        print(f"Loaded checkpoint {ckpt_path} ({len(leaves)} tensors) and kept {len(archived_games)} AZ self-play games.")
    else:
        cache_path = os.path.join(_HERE, "dataset_2026.npz")
        if os.path.exists(cache_path):
            d = np.load(cache_path)
            all_x, all_pi, all_z, all_best, all_crit = d["x"], d["pi"], d["z"], d["best"], d["crit"]
        else:
            all_x, all_pi, all_z, all_best, all_crit = generate_dataset(1584, seed=2026)
            np.savez_compressed(cache_path, x=all_x, pi=all_pi, z=all_z, best=all_best, crit=all_crit)

        n_train = 1200
        tr_x, tr_pi, tr_z = all_x[:n_train], all_pi[:n_train], all_z[:n_train]
        va_x, va_pi, va_z, va_best, va_crit = all_x[n_train:], all_pi[n_train:], all_z[n_train:], all_best[n_train:], all_crit[n_train:]
        va_x_nhwc = to_nhwc_flat(va_x)

        dummy_x64 = np.zeros((B, NUM_CELLS, 6), dtype=np.float32); dummy_x64[:, :, 4] = 1.0
        dummy_pi64 = np.full((B, NUM_CELLS), 1.0 / NUM_CELLS, dtype=np.float32)
        dummy_z64 = np.zeros((B,), dtype=np.float32)

        opt = optim.adamw(optim.cosine_decay(2.2e-3, total_steps=600, warmup_steps=20, min_lr=2.5e-4), weight_decay=1e-4, max_grad_norm=1.0)
        st0 = opt.init(p0)

        def train_step(state, batch):
            pr_in, st_in = state
            xb, pib, zb = batch

            def loss_fn(pr):
                logits, val = model(pr, xb)
                lp = nn.log_softmax(logits, axis=-1)
                pl = -(pib * lp).sum(axis=-1).mean()
                vl = ((val - zb) ** 2).mean()
                return pl + 0.6 * vl, (pl, vl)

            (tot_l, (pl, vl)), g = rg.value_and_grad(loss_fn, has_aux=True)(pr_in)
            pr_out, st_out = opt.update(g, st_in, pr_in)
            return (pr_out, st_out), {"p": pl, "v": vl}

        c_train_64 = rg.compile(train_step, ((p0, st0), (dummy_x64, dummy_pi64, dummy_z64)), device="metal", validate="basic")
        c_eval_64 = rg.compile(infer_fn, (p0, dummy_x64), device="metal", validate="basic")

        # 1. Warm-Start Bootstrap (200 steps B=64)
        boot_buf = [
            (to_nhwc_flat(tr_x[i : i + 1])[0], tr_pi[i].astype(np.float32), float(tr_z[i]))
            for i in range(n_train)
        ]
        boot_batches = [sample_batch_from_buffer(boot_buf, B, rng) for _ in range(200)]
        res_b = rg.run(c_train_64, (p0, st0), boot_batches, steps=200)
        (p_rl, st_rl) = res_b.state
        pl, t1, ct1, vm = eval_validation_metal(c_eval_64, p_rl, va_x_nhwc, va_pi, va_z, va_best, va_crit)
        print(f"Bootstrap 200st | ValPL={pl:.3f} Top1={t1*100:.1f}% Crit={ct1*100:.1f}%")

        # 2. Pure AlphaZero MCTS-48 Self-Play (5 Gens x 32 Parallel Games) with anti-forgetting anchor
        rl_buf: Deque[Tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=100_000)
        rl_buf.extend(boot_buf)
        rl_buf.extend(boot_buf)

        metrics = []
        archived_games = []

        for gen in range(1, 6):
            t0 = time.time()
            new_samples, rec_games, (bw, ww, dr, avg_m) = run_recorded_selfplay_gen(
                gen=gen, n_games=32, params=p_rl, c_eval_32=c_eval_32, n_sims=48, rng=rng, max_plies=85
            )
            sp_sec = time.time() - t0
            rl_buf.extend(new_samples)

            t_tr = time.time()
            buf_list = list(rl_buf)
            train_batches = [sample_batch_from_buffer(buf_list, B, rng) for _ in range(70)]
            res_r = rg.run(c_train_64, (p_rl, st_rl), train_batches, steps=70)
            (p_rl, st_rl) = res_r.state
            tr_sec = time.time() - t_tr

            pl_hist = [float(x) for x in res_r.history["aux.p"]]
            vl_hist = [float(x) for x in res_r.history["aux.v"]]
            pl_v, t1_v, ct1_v, vm_v = eval_validation_metal(c_eval_64, p_rl, va_x_nhwc, va_pi, va_z, va_best, va_crit)
            metrics.append({
                "generation": gen,
                "pLoss": round(sum(pl_hist) / max(1, len(pl_hist)), 3),
                "vLoss": round(sum(vl_hist) / max(1, len(vl_hist)), 3),
                "avgMoves": round(avg_m, 1),
                "blackWins": bw,
                "whiteWins": ww,
                "draws": dr,
                "spSec": round(sp_sec, 2),
                "trainSec": round(tr_sec, 2),
                "bufferSize": len(rl_buf),
            })
            print(
                f"AZ Gen {gen}/5 | SP {sp_sec:5.2f}s (avg {avg_m:4.1f}m, B/W/D={bw}/{ww}/{dr}) | Train {tr_sec:4.2f}s | ValPL={pl_v:.3f} Top1={t1_v*100:.1f}% Crit={ct1_v*100:.1f}%"
            )

            decisive = [g for g in rec_games if g["winner"] != 0]
            sweet = [g for g in decisive if 20 <= len(g["moves"]) <= 52]
            pick = min(sweet, key=lambda g: abs(len(g["moves"]) - 34)) if sweet else (
                min(decisive, key=lambda g: len(g["moves"])) if decisive else rec_games[0]
            )
            outcome = "흑승" if pick["winner"] == BLACK else "백승" if pick["winner"] == WHITE else "무승부"
            sp_name = f"D4OverlapPattern + MCTS-48 (Gen {gen})"
            archived_games.append({
                "id": f"az-gen-{gen}",
                "label": f"AZ Gen {gen} (MCTS 자가대국) · {len(pick['moves'])}수 {outcome}",
                "generation": gen,
                "winner": pick["winner"],
                "blackPlayer": sp_name,
                "whitePlayer": sp_name,
                "matchup": "AlphaZero MCTS 자가대국",
                "moves": pick["moves"],
            })

        # Save trained parameter leaves
        leaf_arrays = {f"p_{i}": np.asarray(leaf.numpy(), dtype=np.float32) for i, leaf in enumerate(rg.tree.leaves(p_rl))}
        np.savez_compressed(ckpt_path, **leaf_arrays)
        print(f"Saved weights to {ckpt_path} ({len(leaf_arrays)} tensors)")

    # 3. Record Final Showcase: MCTS-128 vs 룰베이스(Full-RB) across 8 standard openings (4B + 4W)
    print("\nRunning Final Showcase: D4OverlapPattern + MCTS-128 vs Full-RB Teacher (8 games)...")
    t_show = time.time()
    show_slots = run_recorded_mcts128_vs_full_rb(p_rl, c_eval_32, n_sims=128, max_plies=110)
    print(f"Finished 8 MCTS-128 vs Full-RB games in {time.time() - t_show:.2f}s")

    nn_wins = [g for g in show_slots if g["winner"] == g["nn_side"]]
    nn_draws = [g for g in show_slots if g["winner"] == 0]
    nn_losses = [g for g in show_slots if g["winner"] not in (0, g["nn_side"])]
    print(f"Showcase Record vs Full-RB Teacher: {len(nn_wins)}W / {len(nn_draws)}D / {len(nn_losses)}L")

    # Put draws/losses first and sharpest MCTS-128 wins LAST so UI (.reverse()) displays wins at the very top
    chosen_show = list(reversed((nn_wins + nn_draws + nn_losses)[:6]))
    for s_idx, g in enumerate(chosen_show, 1):
        w = g["winner"]
        outcome = "흑승" if w == BLACK else "백승" if w == WHITE else "무승부"
        b_name = "D4OverlapPattern + MCTS-128" if g["nn_side"] == BLACK else "룰베이스(Full-RB)"
        w_name = "D4OverlapPattern + MCTS-128" if g["nn_side"] == WHITE else "룰베이스(Full-RB)"
        archived_games.append({
            "id": f"az-mcts128-rb-{s_idx}",
            "label": f"AZ Final (MCTS-128 vs Full-RB · {g['op']}) · {len(g['replay'])}수 {outcome}",
            "generation": 6,
            "winner": w,
            "blackPlayer": b_name,
            "whitePlayer": w_name,
            "matchup": "MCTS-128 vs 룰베이스",
            "moves": g["replay"],
        })

    archive = {
        "rules": "renju",
        "modelInfo": {
            "name": "D4OverlapPattern + MCTS",
            "channels": 64,
            "blocks": 3,
            "params": int(n_params),
        },
        "metrics": metrics,
        "games": archived_games,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(archive, f, ensure_ascii=False, indent=2)
    print(f"Exported {len(archived_games)} AlphaZero MCTS replays to {out_path}")


if __name__ == "__main__":
    main()
