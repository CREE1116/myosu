"""Empirical tests for D4 Inductive Bias / Sample Efficiency and Teacher-Free 2-Ply Self-Play."""

from __future__ import annotations

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
    D4OverlapPatternNet,
    FlatMLPNet,
    LineConvNet,
    OverlapPatternNet,
    PatternTableNet,
    ResNet3x3Net,
    count_params,
)


def dihedral_transform_batch(
    states_nchw: np.ndarray, policies: np.ndarray, sym: int
) -> Tuple[np.ndarray, np.ndarray]:
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


def dihedral_single(
    state_nchw: np.ndarray, policy: np.ndarray, val: float
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    out = []
    for sym in range(8):
        s, p = dihedral_transform_batch(state_nchw[None, ...], policy[None, ...], sym)
        out.append((s[0], p[0], val))
    return out


def to_nhwc_flat(states_nchw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(states_nchw, (0, 2, 3, 1)).reshape(-1, NUM_CELLS, 6))


def eval_validation(c_eval_b, params, va_x_nhwc, va_pi, va_z, va_best, va_crit, B: int = 32):
    n_val = len(va_x_nhwc)
    all_logits = []
    all_vals = []
    for off in range(0, n_val, B):
        lg, vl = c_eval_b(params, va_x_nhwc[off : off + B])
        all_logits.append(np.asarray(lg.numpy(), dtype=np.float32))
        all_vals.append(np.asarray(vl.numpy(), dtype=np.float32))
    logits_np = np.concatenate(all_logits, axis=0)
    vals_np = np.concatenate(all_vals, axis=0)
    max_l = np.max(logits_np, axis=1, keepdims=True)
    log_probs = logits_np - max_l - np.log(np.sum(np.exp(logits_np - max_l), axis=1, keepdims=True))
    val_ploss = float(-np.mean(np.sum(va_pi * log_probs, axis=1)))
    val_mse = float(np.mean((vals_np - va_z) ** 2))
    pred1 = np.argmax(logits_np, axis=1)
    top1 = float(np.mean(pred1 == va_best))
    crit1 = float(np.mean(pred1[va_crit] == va_best[va_crit])) if va_crit.any() else 0.0
    return val_ploss, top1, crit1, val_mse


def play_match_game(seed_board: RenjuBoard, start_player: int, agent_black, agent_white, max_plies: int = 120) -> int:
    b = seed_board.clone()
    player = start_player
    for _ in range(max_plies):
        agent = agent_black if player == BLACK else agent_white
        mv = agent(b, player)
        if mv is None or not b.is_legal_move(mv, player):
            return opponent_of(player)
        b.place_move(mv, player)
        if b.is_win_at(mv, player):
            return player
        player = opponent_of(player)
    return 0


def arena_match(openings: List[Tuple[RenjuBoard, int]], agent_a, agent_b) -> Tuple[int, int, int, float]:
    wins, draws, losses = 0, 0, 0
    for sb, sp in openings:
        r1 = play_match_game(sb, sp, agent_a if sp == BLACK else agent_b, agent_b if sp == BLACK else agent_a)
        if r1 == sp:
            wins += 1
        elif r1 == 0:
            draws += 1
        else:
            losses += 1
        r2 = play_match_game(sb, sp, agent_b if sp == BLACK else agent_a, agent_a if sp == BLACK else agent_b)
        if r2 == opponent_of(sp):
            wins += 1
        elif r2 == 0:
            draws += 1
        else:
            losses += 1
    total = 2 * len(openings)
    wr = (wins + 0.5 * draws) / float(total) * 100.0
    return wins, draws, losses, wr


def masked_softmax_np(logits: np.ndarray, mask: np.ndarray, temp: float = 1.0) -> np.ndarray:
    m_logits = np.where(mask, logits, -1e9)
    max_v = float(np.max(m_logits))
    exps = np.where(mask, np.exp(np.clip((m_logits - max_v) / max(1e-4, temp), -30.0, 0.0)), 0.0)
    s = float(exps.sum())
    if s <= 1e-12:
        cnt = max(1, int(mask.sum()))
        return (mask.astype(np.float32) / float(cnt))
    return (exps / s).astype(np.float32)


def fast_legal_mask(board: RenjuBoard, player: int) -> np.ndarray:
    mask = np.asarray(board.cells, dtype=np.int32) == 0
    if player == BLACK:
        for idx in board.candidate_indices(radius=3):
            r, c = divmod(idx, BOARD_SIZE)
            if board.is_black_forbidden(r, c):
                mask[idx] = False
    return mask


def find_immediate_win(board: RenjuBoard, player: int, mask: np.ndarray) -> Optional[int]:
    for idx in board.candidate_indices(radius=1):
        if not mask[idx]:
            continue
        board.cells[idx] = player
        won = board.is_win_at(int(idx), player)
        board.cells[idx] = 0
        if won:
            return int(idx)
    return None


def search_move(
    board: RenjuBoard,
    player: int,
    eval_1,
    eval_6,
    eval_12,
    params,
    depth: int,
    rng: Optional[random.Random] = None,
    sample_temp: float = 0.0,
) -> Tuple[Optional[int], np.ndarray, float]:
    """Depth-0 (pure policy), Depth-1 (-V(next)), or Depth-2 (K1=4, K2=3 minimax via B=12) search."""
    mask = fast_legal_mask(board, player)
    legal_idx = np.flatnonzero(mask)
    if len(legal_idx) == 0:
        return None, np.zeros(NUM_CELLS, dtype=np.float32), 0.0

    win_idx = find_immediate_win(board, player, mask)
    t1 = board.encode_tensor(player, mask)
    x1 = to_nhwc_flat(t1[None, ...])
    logits_t, val_t = eval_1(params, x1)
    logits = np.asarray(logits_t.numpy(), dtype=np.float32)[0]
    root_val = float(np.asarray(val_t.numpy(), dtype=np.float32)[0])
    nn_probs = masked_softmax_np(logits, mask, temp=1.0)

    if win_idx is not None:
        pi_win = np.zeros(NUM_CELLS, dtype=np.float32)
        pi_win[win_idx] = 1.0
        return win_idx, pi_win, 1.0

    if depth == 0 or len(legal_idx) == 1:
        if sample_temp > 0.05 and rng is not None:
            sharp = masked_softmax_np(logits, mask, temp=sample_temp)
            mv = int(rng.choices(range(NUM_CELLS), weights=sharp.tolist(), k=1)[0])
        else:
            mv = int(np.argmax(np.where(mask, logits, -1e9)))
        return mv, nn_probs, root_val

    k1 = min(6 if depth == 1 else 4, len(legal_idx))
    masked_logits = np.where(mask, logits, -1e9)
    cand1 = [int(i) for i in np.argpartition(masked_logits, -k1)[-k1:] if mask[int(i)]]
    if not cand1:
        cand1 = [int(legal_idx[0])]

    opp = opponent_of(player)
    b1_boards: List[RenjuBoard] = []
    b1_masks: List[np.ndarray] = []
    batch6 = np.zeros((6, NUM_CELLS, 6), dtype=np.float32)
    for i, mv1 in enumerate(cand1):
        cb = board.clone()
        cb.place_move(mv1, player)
        omask = fast_legal_mask(cb, opp)
        b1_boards.append(cb)
        b1_masks.append(omask)
        batch6[i] = to_nhwc_flat(cb.encode_tensor(opp, omask)[None, ...])[0]

    opp_logits_t, opp_vals_t = eval_6(params, batch6)
    opp_logits_mat = np.asarray(opp_logits_t.numpy(), dtype=np.float32)[: len(cand1)]
    opp_vals = np.asarray(opp_vals_t.numpy(), dtype=np.float32)[: len(cand1)]

    if depth == 1:
        q_scores = np.full(NUM_CELLS, -1e9, dtype=np.float32)
        best_q = -1e9
        for i, mv1 in enumerate(cand1):
            q = float(-opp_vals[i]) * 2.5 + 0.5 * float(logits[mv1])
            q_scores[mv1] = q
            best_q = max(best_q, float(-opp_vals[i]))
        soft_q = masked_softmax_np(q_scores, mask, temp=0.5)
        pi = (0.80 * soft_q + 0.20 * nn_probs).astype(np.float32)
        pi /= max(1e-9, float(pi.sum()))
        if sample_temp > 0.05 and rng is not None:
            sharp = masked_softmax_np(np.log(np.maximum(pi, 1e-8)), mask, temp=sample_temp)
            chosen = int(rng.choices(range(NUM_CELLS), weights=sharp.tolist(), k=1)[0])
        else:
            chosen = int(np.argmax(np.where(mask, pi, -1e9)))
        return chosen, pi, float(np.clip(best_q, -1.0, 1.0))

    # Depth == 2: explicit opponent top-K2=3 reply minimax via single B=12 batched eval
    k2 = 3
    batch12 = np.zeros((12, NUM_CELLS, 6), dtype=np.float32)
    leaf_counts = [0] * len(cand1)
    opp_immed_lose = [False] * len(cand1)
    slot = 0
    for i, cb1 in enumerate(b1_boards):
        opp_mask = b1_masks[i]
        opp_legal = np.flatnonzero(opp_mask)
        if len(opp_legal) == 0:
            continue
        if find_immediate_win(cb1, opp, opp_mask) is not None:
            opp_immed_lose[i] = True
            continue
        kk = min(k2, len(opp_legal))
        ol = np.where(opp_mask, opp_logits_mat[i], -1e9)
        replies = [int(r) for r in np.argpartition(ol, -kk)[-kk:] if opp_mask[int(r)]]
        for rep in replies:
            cb2 = cb1.clone()
            cb2.place_move(rep, opp)
            pmask2 = fast_legal_mask(cb2, player)
            batch12[slot] = to_nhwc_flat(cb2.encode_tensor(player, pmask2)[None, ...])[0]
            slot += 1
            leaf_counts[i] += 1

    if slot > 0:
        _, leaf_vals_t = eval_12(params, batch12)
        leaf_vals = np.asarray(leaf_vals_t.numpy(), dtype=np.float32)[:slot]
    else:
        leaf_vals = np.zeros((0,), dtype=np.float32)

    q2_vals: List[float] = []
    q_scores = np.full(NUM_CELLS, -1e9, dtype=np.float32)
    cur = 0
    for i, mv1 in enumerate(cand1):
        cnt = leaf_counts[i]
        if opp_immed_lose[i]:
            q2 = -1.0
        elif cnt > 0:
            min_child = float(np.min(leaf_vals[cur : cur + cnt]))
            cur += cnt
            q2 = 0.75 * min_child + 0.25 * float(-opp_vals[i])
        else:
            q2 = float(-opp_vals[i])
        q2_vals.append(q2)
        q_scores[mv1] = 3.2 * q2 + 0.35 * float(logits[mv1])

    soft_q = masked_softmax_np(q_scores, mask, temp=0.45)
    pi = (0.85 * soft_q + 0.15 * nn_probs).astype(np.float32)
    pi /= max(1e-9, float(pi.sum()))
    best_q2 = float(max(q2_vals)) if q2_vals else root_val
    if sample_temp > 0.05 and rng is not None:
        sharp = masked_softmax_np(np.log(np.maximum(pi, 1e-8)), mask, temp=sample_temp)
        chosen = int(rng.choices(range(NUM_CELLS), weights=sharp.tolist(), k=1)[0])
    else:
        chosen = int(np.argmax(np.where(mask, pi, -1e9)))
    return chosen, pi, float(np.clip(best_q2, -1.0, 1.0))


def check_d4_equivariance(models: Dict[str, object], sample_x_nchw: np.ndarray) -> None:
    print("\n--- Check 0: Exact D4 Equivariance Verification (Max Error across 8 Symmetries) ---")
    x0 = sample_x_nchw[:1]
    dummy_p = np.zeros((1, NUM_CELLS), dtype=np.float32)
    for name, model in models.items():
        pr = model.init(rg_random.key(42))
        logits_ref, val_ref = model(pr, rg.tensor(to_nhwc_flat(x0)))
        l_ref = np.asarray(logits_ref.numpy(), dtype=np.float32)[0].reshape(BOARD_SIZE, BOARD_SIZE)
        v_ref = float(np.asarray(val_ref.numpy(), dtype=np.float32)[0])
        max_l_err = 0.0
        max_v_err = 0.0
        for sym in range(1, 8):
            xs, _ = dihedral_transform_batch(x0, dummy_p, sym)
            ls_t, vs_t = model(pr, rg.tensor(xs))
            ls = np.asarray(ls_t.numpy(), dtype=np.float32)[0].reshape(BOARD_SIZE, BOARD_SIZE)
            vs = float(np.asarray(vs_t.numpy(), dtype=np.float32)[0])
            # Apply same symmetry to l_ref (only on legal cells where mask == 1)
            mask0 = x0[0, 4] > 0.5
            k = sym % 4
            flip = sym >= 4
            l_rot = np.rot90(l_ref, k=k)
            m_rot = np.rot90(mask0, k=k)
            if flip:
                l_rot = np.flip(l_rot, axis=1)
                m_rot = np.flip(m_rot, axis=1)
            max_l_err = max(max_l_err, float(np.max(np.abs(ls[m_rot] - l_rot[m_rot]))))
            max_v_err = max(max_v_err, abs(vs - v_ref))
        print(f"  {name:18s} | max_policy_err = {max_l_err:.6e} | max_value_err = {max_v_err:.6e}")


def run_selfplay_trial(
    title: str,
    model,
    init_params,
    eval_va_tuple,
    openings: List[Tuple[RenjuBoard, int]],
    rb_weak,
    rb_strong,
    gens: int,
    games_per_gen: int,
    steps_per_gen: int,
    search_depth: int,
    teacher_weight: float,
    heur_weight: float,
    seed: int,
):
    print(f"\n=== {title} ===")
    va_x_nhwc, va_pi, va_z, va_best, va_crit = eval_va_tuple
    rng = random.Random(seed)
    B = 32
    total_steps = gens * steps_per_gen
    sched = optim.cosine_decay(1.8e-3, total_steps=total_steps, warmup_steps=min(15, total_steps // 6), min_lr=2e-4)
    opt = optim.adamw(sched, weight_decay=1e-4, max_grad_norm=1.0)
    params = init_params
    opt_state = opt.init(params)

    dummy_x = np.zeros((B, NUM_CELLS, 6), dtype=np.float32)
    dummy_x[:, :, 4] = 1.0
    dummy_pi = np.full((B, NUM_CELLS), 1.0 / NUM_CELLS, dtype=np.float32)
    dummy_z = np.zeros((B,), dtype=np.float32)

    def train_step(state, batch):
        pr_in, st_in = state
        xb, pib, zb = batch

        def loss_fn(pr):
            logits, val = model(pr, xb)
            lp = nn.log_softmax(logits, axis=-1)
            pl = -(pib * lp).sum(axis=-1).mean()
            vl = ((val - zb) ** 2).mean()
            return pl + 0.6 * vl

        l_val, g = rg.value_and_grad(loss_fn)(pr_in)
        pr_out, st_out = opt.update(g, st_in, pr_in)
        return (pr_out, st_out), l_val

    def infer_fn(pr, xb):
        return model(pr, xb)

    c_train = rg.compile(train_step, ((params, opt_state), (dummy_x, dummy_pi, dummy_z)), device="cpu", validate="basic")
    c_eval_1 = rg.compile(infer_fn, (params, dummy_x[:1]), device="cpu", validate="basic")
    c_eval_6 = rg.compile(infer_fn, (params, dummy_x[:6]), device="cpu", validate="basic")
    c_eval_12 = rg.compile(infer_fn, (params, dummy_x[:12]), device="cpu", validate="basic")
    c_eval_b = rg.compile(infer_fn, (params, va_x_nhwc[:B]), device="cpu", validate="basic")

    buf: Deque[Tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=12_000)

    pl0, t1_0, c1_0, vm0 = eval_validation(c_eval_b, params, va_x_nhwc, va_pi, va_z, va_best, va_crit)
    print(f"  [Before] ValPLoss={pl0:.3f} | Top1={t1_0*100:5.1f}% | CritT1={c1_0*100:5.1f}% | ValMSE={vm0:.3f}")

    for gen in range(1, gens + 1):
        t0 = time.time()
        b_wins, w_wins, draws, total_mv = 0, 0, 0, 0
        for _ in range(games_per_gen):
            b = RenjuBoard()
            player = BLACK
            rand_open = rng.randint(0, 3)
            traj: List[Tuple[np.ndarray, np.ndarray, float, int]] = []
            winner = 0
            for step in range(65):
                mask = fast_legal_mask(b, player)
                legal_idx = np.flatnonzero(mask)
                if len(legal_idx) == 0:
                    winner = opponent_of(player)
                    break
                st = b.encode_tensor(player, mask)
                temp = 0.70 if step < 8 else 0.12
                mv_search, pi_search, q_search = search_move(
                    b, player, c_eval_1, c_eval_6, c_eval_12, params, depth=search_depth, rng=rng, sample_temp=temp
                )
                if step < rand_open:
                    pool = [r * BOARD_SIZE + c for r in range(5, 10) for c in range(5, 10) if mask[r * BOARD_SIZE + c]]
                    chosen = 7 * BOARD_SIZE + 7 if (step == 0 and rng.random() < 0.6) else (rng.choice(pool) if pool else int(legal_idx[0]))
                else:
                    chosen = mv_search if mv_search is not None else int(legal_idx[0])

                if teacher_weight > 0.0:
                    from renju_env import score_candidates, board_heuristic_value
                    scored, _ = score_candidates(b, player, mask)
                    t_pi = np.zeros(NUM_CELLS, dtype=np.float32)
                    if scored:
                        top = scored[: min(16, len(scored))]
                        max_s = top[0][1]
                        tt = 350.0 if max_s < 50_000 else max_s * 0.08
                        exps = [math.exp(max(-30.0, (s - max_s) / tt)) for _, s in top]
                        zs = sum(exps)
                        for (ic, _), e in zip(top, exps):
                            t_pi[ic] = e / zs
                    else:
                        t_pi[mask] = 1.0 / float(mask.sum())
                    target_pi = (teacher_weight * t_pi + (1.0 - teacher_weight) * pi_search).astype(np.float32)
                    hv = board_heuristic_value(b, player)
                else:
                    target_pi = pi_search
                    hv = q_search

                ps = float(target_pi.sum())
                if ps > 0:
                    target_pi /= ps
                traj.append((st, target_pi, hv, player))
                b.place_move(chosen, player)
                total_mv += 1
                if b.is_win_at(chosen, player):
                    winner = player
                    break
                player = opponent_of(player)

            if winner == BLACK:
                b_wins += 1
            elif winner == WHITE:
                w_wins += 1
            else:
                draws += 1

            for st_item, pi_item, q_item, mover in traj:
                out_z = 0.0 if winner == 0 else (1.0 if winner == mover else -1.0)
                if heur_weight > 0.0:
                    zv = heur_weight * q_item + (1.0 - heur_weight) * out_z
                else:
                    zv = out_z
                buf.extend(dihedral_single(st_item, pi_item, zv))

        buf_list = list(buf)
        bn = len(buf_list)
        batches = []
        for _ in range(steps_per_gen):
            idxs = [rng.randrange(bn) for _ in range(B)]
            xb = np.stack([buf_list[k][0] for k in idxs], axis=0).astype(np.float32)
            pib = np.stack([buf_list[k][1] for k in idxs], axis=0)
            zb = np.array([buf_list[k][2] for k in idxs], dtype=np.float32)
            batches.append((xb, pib, zb))

        res = rg.run(c_train, (params, opt_state), batches, steps=steps_per_gen)
        (params, opt_state) = res.state
        pl_g, t1_g, c1_g, vm_g = eval_validation(c_eval_b, params, va_x_nhwc, va_pi, va_z, va_best, va_crit)
        print(
            f"  Gen {gen}/{gens} ({time.time()-t0:4.1f}s) | avg {total_mv/games_per_gen:4.1f}m B/W/D={b_wins}/{w_wins}/{draws} | ValPLoss={pl_g:.3f} Top1={t1_g*100:4.1f}% Crit={c1_g*100:4.1f}%"
        )

    eval_openings = openings[:5]
    for ev_depth in ([0, 1] if search_depth < 2 else [0, 1, 2]):
        ag = lambda brd, pl, d=ev_depth: search_move(brd, pl, c_eval_1, c_eval_6, c_eval_12, params, depth=d)[0]
        w_w, d_w, l_w, wr_w = arena_match(eval_openings, ag, rb_weak)
        w_s, d_s, l_s, wr_s = arena_match(eval_openings, ag, rb_strong)
        print(
            f"  [Eval {ev_depth}-ply, 10g] vs Noisy-RB: {w_w:2d}/{d_w:2d}/{l_w:2d} ({wr_w:5.1f}%) | vs Full-RB: {w_s:2d}/{d_s:2d}/{l_s:2d} ({wr_s:5.1f}%)"
        )
    return params, c_eval_1, c_eval_6, c_eval_12


def main() -> None:
    exp2_only = "--exp2-only" in sys.argv
    print("=== Empirical Experiment: D4 Inductive Bias & Teacher-Free 2-Ply Self-Play ===")
    cache_path = os.path.join(_HERE, "dataset_2026.npz")
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        all_x, all_pi, all_z, all_best, all_crit = d["x"], d["pi"], d["z"], d["best"], d["crit"]
    else:
        all_x, all_pi, all_z, all_best, all_crit = generate_dataset(1584, seed=2026)
        np.savez_compressed(cache_path, x=all_x, pi=all_pi, z=all_z, best=all_best, crit=all_crit)

    n_train = 1200
    n_val = 384
    tr_x, tr_pi, tr_z = all_x[:n_train], all_pi[:n_train], all_z[:n_train]
    va_x, va_pi, va_z, va_best, va_crit = all_x[n_train:], all_pi[n_train:], all_z[n_train:], all_best[n_train:], all_crit[n_train:]
    va_x_nhwc = to_nhwc_flat(va_x)
    eval_va_tuple = (va_x_nhwc, va_pi, va_z, va_best, va_crit)

    if not exp2_only:
        check_d4_equivariance(
            {
                "OverlapPattern": OverlapPatternNet(d_ray=24, d_outer=8, width=64, blocks=3),
                "D4OverlapPattern": D4OverlapPatternNet(d_ray=24, d_outer=8, width=64, blocks=3),
            },
            tr_x[:1],
        )

    # Identical training batch sequence (320 steps x B=32) across all models
    B = 32
    checkpoints = [320] if exp2_only else [20, 40, 80, 160, 320]
    py_rng = random.Random(777)
    batches = []
    for _ in range(320):
        idxs = [py_rng.randrange(n_train) for _ in range(B)]
        sym = py_rng.randrange(8)
        xb, pib = dihedral_transform_batch(tr_x[idxs], tr_pi[idxs], sym)
        zb = tr_z[idxs].astype(np.float32)
        batches.append((xb, pib, zb))

    all_specs = [
        ("FlatMLP", lambda: FlatMLPNet(width=128, blocks=3)),
        ("ResNet3x3", lambda: ResNet3x3Net(width=32, blocks=4)),
        ("LineConv4Axis", lambda: LineConvNet(width=36, line_ch=12, blocks=4)),
        ("PatternTableNNUE", lambda: PatternTableNet(d_ray=24, width=64, blocks=3)),
        ("OverlapPattern", lambda: OverlapPatternNet(d_ray=24, d_outer=8, width=64, blocks=3)),
        ("D4OverlapPattern", lambda: D4OverlapPatternNet(d_ray=24, d_outer=8, width=64, blocks=3)),
    ]
    specs = [all_specs[-1]] if exp2_only else all_specs

    if not exp2_only:
        print("\n--- Experiment 1: Sample-Efficiency Learning Curve (ValPLoss / Top1 / CritT1 by Step) ---")
        print("Arch               | Params | Step  20 (PLoss/T1/Crit) | Step  40              | Step  80              | Step 160              | Step 320              | Time")
        print("-" * 155)

    snapshots: Dict[Tuple[str, int], dict] = {}
    trained_final: Dict[str, dict] = {}

    for label, ctor in specs:
        model = ctor()
        p = model.init(rg_random.key(42))
        n_params = count_params(p)
        sched = optim.cosine_decay(2.8e-3, total_steps=320, warmup_steps=20, min_lr=2e-4)
        opt = optim.adamw(sched, weight_decay=1e-4, max_grad_norm=1.0)
        st = opt.init(p)

        def train_step(state, batch):
            pr_in, st_in = state
            xb, pib, zb = batch

            def loss_fn(pr):
                logits, val = model(pr, xb)
                lp = nn.log_softmax(logits, axis=-1)
                pl = -(pib * lp).sum(axis=-1).mean()
                vl = ((val - zb) ** 2).mean()
                return pl + 0.6 * vl

            l_val, g = rg.value_and_grad(loss_fn)(pr_in)
            pr_out, st_out = opt.update(g, st_in, pr_in)
            return (pr_out, st_out), l_val

        def infer_fn(pr, xb):
            return model(pr, xb)

        c_train = rg.compile(train_step, ((p, st), batches[0]), device="cpu", validate="basic")
        c_eval_b = rg.compile(infer_fn, (p, va_x_nhwc[:B]), device="cpu", validate="basic")
        c_eval_1 = rg.compile(infer_fn, (p, va_x_nhwc[:1]), device="cpu", validate="basic")
        c_eval_6 = rg.compile(infer_fn, (p, va_x_nhwc[:6]), device="cpu", validate="basic")
        c_eval_12 = rg.compile(infer_fn, (p, va_x_nhwc[:12]), device="cpu", validate="basic")

        prev_step = 0
        t_tot = 0.0
        cells = []
        for ckpt in checkpoints:
            seg = batches[prev_step:ckpt]
            t_seg = time.time()
            res = rg.run(c_train, (p, st), seg, steps=len(seg))
            t_tot += time.time() - t_seg
            (p, st) = res.state
            prev_step = ckpt
            pl, t1, ct1, _ = eval_validation(c_eval_b, p, va_x_nhwc, va_pi, va_z, va_best, va_crit)
            cells.append(f"{pl:5.2f}/{t1*100:4.1f}%/{ct1*100:4.1f}%")
            if ckpt in (40, 320):
                snapshots[(label, ckpt)] = {
                    "model": model,
                    "params": p,
                    "eval_1": c_eval_1,
                    "eval_6": c_eval_6,
                    "eval_12": c_eval_12,
                }
        trained_final[label] = snapshots[(label, 320)]
        print(f"{label:18s} | {n_params:6d} | " + " | ".join(f"{c:21s}" for c in cells) + f" | {t_tot:4.2f}s")

    # Seeded Arena openings
    arena_rng = random.Random(999)
    openings: List[Tuple[RenjuBoard, int]] = []
    while len(openings) < 10:
        seeded = generate_seeded_board(arena_rng, min_stones=4, max_stones=10)
        if seeded is not None:
            openings.append(seeded)

    def make_ag(info_dict, depth=0):
        return lambda brd, pl: search_move(
            brd, pl, info_dict["eval_1"], info_dict["eval_6"], info_dict["eval_12"], info_dict["params"], depth=depth
        )[0]

    if not exp2_only:
        print("\n--- Head-to-Head Arena: Inductive Bias Speedup (0-ply Pure NN, 20 games) ---")
        for (n1, s1), (n2, s2) in [
            (("D4OverlapPattern", 40), ("ResNet3x3", 40)),
            (("D4OverlapPattern", 40), ("ResNet3x3", 320)),
            (("D4OverlapPattern", 320), ("ResNet3x3", 320)),
            (("D4OverlapPattern", 320), ("OverlapPattern", 320)),
        ]:
            w, d_g, l, wr = arena_match(openings, make_ag(snapshots[(n1, s1)]), make_ag(snapshots[(n2, s2)]))
            print(f"  {n1}(step {s1}) vs {n2}(step {s2}) -> W/D/L: {w:2d}/{d_g:2d}/{l:2d} ({wr:5.1f}%)")

    # Experiment 2 Setup
    rb_weak_rng = random.Random(101)
    rb_weak = lambda b, pl: rule_based_move(b, pl, noise_eps=0.30, rng=rb_weak_rng)
    rb_strong = lambda b, pl: rule_based_move(b, pl, noise_eps=0.0, rng=None)

    d4_boot = snapshots[("D4OverlapPattern", 320)]
    if not exp2_only:
        print("\n--- Experiment 2B: Search Depth Scaling on Bootstrapped D4OverlapPattern (step 320) vs Full-RB Teacher ---")
        for depth in (0, 1, 2):
            t_dep = time.time()
            ag = make_ag(d4_boot, depth=depth)
            w_w, d_w, l_w, wr_w = arena_match(openings, ag, rb_weak)
            w_s, d_s, l_s, wr_s = arena_match(openings, ag, rb_strong)
            print(
                f"  Bootstrapped {depth}-ply ({time.time()-t_dep:4.1f}s) | vs Noisy-RB: {w_w:2d}/{d_w:2d}/{l_w:2d} ({wr_w:5.1f}%) | vs Full-RB Teacher: {w_s:2d}/{d_s:2d}/{l_s:2d} ({wr_s:5.1f}%)"
            )

    # Experiment 2A: Tabula-Rasa Zero-Teacher Self-Play from Random Init
    d4_scratch_model = D4OverlapPatternNet(d_ray=24, d_outer=8, width=64, blocks=3)
    d4_scratch_p0 = d4_scratch_model.init(rg_random.key(123))
    run_selfplay_trial(
        title="Experiment 2A: Tabula-Rasa Pure Self-Play (Zero Teacher alpha=0, From Random Init, 6 Gens x 8 Games)",
        model=d4_scratch_model,
        init_params=d4_scratch_p0,
        eval_va_tuple=eval_va_tuple,
        openings=openings,
        rb_weak=rb_weak,
        rb_strong=rb_strong,
        gens=6,
        games_per_gen=8,
        steps_per_gen=30,
        search_depth=1,
        teacher_weight=0.0,
        heur_weight=0.0,
        seed=501,
    )

    # Experiment 2C: Teacher-Free 2-Ply Self-Play Policy Iteration (Starting from Step 320 Bootstrap, ZERO Teacher!)
    p_sp2, e1, e6, e12 = run_selfplay_trial(
        title="Experiment 2C: Teacher-Free 2-Ply Minimax Self-Play (alpha_teacher=0, 4 Gens x 6 Games, B=12 Minimax)",
        model=d4_boot["model"],
        init_params=d4_boot["params"],
        eval_va_tuple=eval_va_tuple,
        openings=openings,
        rb_weak=rb_weak,
        rb_strong=rb_strong,
        gens=4,
        games_per_gen=6,
        steps_per_gen=40,
        search_depth=2,
        teacher_weight=0.0,
        heur_weight=0.5,  # 0.5 * q_2ply_minimax + 0.5 * game_outcome (NO rule-based heur_v!)
        seed=707,
    )

    print("\n--- Direct Comparison: After Teacher-Free 2-Ply Self-Play vs Before Bootstrap (Step 320) [10g] ---")
    for d_ev in (0, 2):
        ag_after = lambda brd, pl, dep=d_ev: search_move(brd, pl, e1, e6, e12, p_sp2, depth=dep)[0]
        ag_before = make_ag(d4_boot, depth=d_ev)
        w, d_g, l, wr = arena_match(openings[:5], ag_after, ag_before)
        print(f"  Post-2Ply-SP vs Pre-SP ({d_ev}-ply) -> W/D/L: {w:2d}/{d_g:2d}/{l:2d} ({wr:5.1f}%)")


if __name__ == "__main__":
    main()
