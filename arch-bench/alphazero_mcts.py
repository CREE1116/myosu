"""Metal GPU Batched Parallel AlphaGo Zero (PUCT MCTS + Self-Play) on D4OverlapPatternNet."""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))  # arch-bench/
sys.path.insert(0, _HERE)
import rg
from rg import nn, optim, random as rg_random

from renju_env import (
    BLACK,
    BOARD_SIZE,
    EMPTY,
    NUM_CELLS,
    WHITE,
    RenjuBoard,
    generate_dataset,
    generate_seeded_board,
    opponent_of,
    rule_based_move,
)
from rg_models import D4OverlapPatternNet, count_params

# Precomputed neighbor lookup tables for fast Renju mask & local move expansion
AX_NB: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(
        (r + s * dr) * BOARD_SIZE + (c + s * dc)
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
        for s in (-3, -2, -1, 1, 2, 3)
        if 0 <= r + s * dr < BOARD_SIZE and 0 <= c + s * dc < BOARD_SIZE
    )
    for r in range(BOARD_SIZE)
    for c in range(BOARD_SIZE)
)


def _build_radius_nb(radius: int) -> Tuple[Tuple[int, ...], ...]:
    out = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            nbs = []
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        nbs.append(nr * BOARD_SIZE + nc)
            out.append(tuple(nbs))
    return tuple(out)


NB_R1 = _build_radius_nb(1)
NB_R2 = _build_radius_nb(2)
NB_R3 = _build_radius_nb(3)


def exact_legal_mask(board: RenjuBoard, player: int) -> np.ndarray:
    """Exact Renju legal mask checking Black forbidden cells only near existing Black stones."""
    cells = board.cells
    mask = np.fromiter((v == EMPTY for v in cells), dtype=np.bool_, count=NUM_CELLS)
    if player == BLACK:
        cnt = [0] * NUM_CELLS
        for idx, v in enumerate(cells):
            if v == BLACK:
                for nb in AX_NB[idx]:
                    cnt[nb] += 1
        for idx in range(NUM_CELLS):
            if cells[idx] == EMPTY and cnt[idx] >= 2:
                r, c = divmod(idx, BOARD_SIZE)
                if board.is_black_forbidden(r, c):
                    mask[idx] = False
    return mask


def local_empty_indices(cells: List[int], nb_table: Tuple[Tuple[int, ...], ...]) -> List[int]:
    marked = [False] * NUM_CELLS
    has_stone = False
    for idx, v in enumerate(cells):
        if v != EMPTY:
            has_stone = True
            for nb in nb_table[idx]:
                if cells[nb] == EMPTY:
                    marked[nb] = True
    if not has_stone:
        return [7 * BOARD_SIZE + 7]
    return [i for i, m in enumerate(marked) if m]


def encode_board_row(
    board: RenjuBoard,
    player: int,
    mask: Optional[np.ndarray],
    out_row: np.ndarray,
) -> None:
    """Writes [225, 6] NHWC float32 state directly into pre-allocated buffer row."""
    cells_np = np.asarray(board.cells, dtype=np.int8)
    opp = opponent_of(player)
    out_row.fill(0.0)
    out_row[:, 0] = cells_np == player
    out_row[:, 1] = cells_np == opp
    if board.last_move is not None:
        out_row[board.last_move, 2] = 1.0
    if player == BLACK:
        out_row[:, 3] = 1.0
    if mask is None:
        out_row[:, 4] = cells_np == EMPTY
    else:
        out_row[:, 4] = mask
    out_row[:, 5] = 1.0


def find_immediate_wins_fast(board: RenjuBoard, player: int) -> List[int]:
    cells = board.cells
    wins: List[int] = []
    for idx in local_empty_indices(cells, NB_R1):
        cells[idx] = player
        won = board.is_win_at(idx, player)
        cells[idx] = EMPTY
        if won:
            wins.append(idx)
    return wins


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


def dihedral_transform_nhwc_row(
    state_nhwc: np.ndarray, policy: np.ndarray, sym: int
) -> Tuple[np.ndarray, np.ndarray]:
    k = sym % 4
    flip = sym >= 4
    s2 = state_nhwc.reshape(BOARD_SIZE, BOARD_SIZE, 6)
    p2 = policy.reshape(BOARD_SIZE, BOARD_SIZE)
    s2 = np.rot90(s2, k=k, axes=(0, 1))
    p2 = np.rot90(p2, k=k, axes=(0, 1))
    if flip:
        s2 = np.flip(s2, axis=1)
        p2 = np.flip(p2, axis=1)
    return (
        np.ascontiguousarray(s2.reshape(NUM_CELLS, 6), dtype=np.float32),
        np.ascontiguousarray(p2.reshape(NUM_CELLS), dtype=np.float32),
    )


def to_nhwc_flat(states_nchw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(states_nchw, (0, 2, 3, 1)).reshape(-1, NUM_CELLS, 6))


class MCTSNode:
    __slots__ = (
        "player",
        "expanded",
        "terminal",
        "terminal_val",
        "actions",
        "priors",
        "N",
        "W",
        "N_sum",
        "children",
        "win_move",
    )

    def __init__(self, player: int):
        self.player = player
        self.expanded = False
        self.terminal = False
        self.terminal_val = 0.0
        self.actions = np.empty((0,), dtype=np.int32)
        self.priors = np.empty((0,), dtype=np.float32)
        self.N = np.empty((0,), dtype=np.float32)
        self.W = np.empty((0,), dtype=np.float32)
        self.N_sum = 0.0
        self.children: Dict[int, MCTSNode] = {}
        self.win_move: Optional[int] = None


def expand_node_from_logits(
    node: MCTSNode,
    board: RenjuBoard,
    logits: np.ndarray,
    val: float,
    max_children: int,
    nb_table: Tuple[Tuple[int, ...], ...],
) -> float:
    """Expands MCTSNode lazily, checking immediate wins and Renju forbidden cells on candidates only."""
    cells = board.cells
    player = node.player
    wins = find_immediate_wins_fast(board, player)
    if wins:
        w_mv = wins[0]
        node.actions = np.array([w_mv], dtype=np.int32)
        node.priors = np.array([1.0], dtype=np.float32)
        node.N = np.zeros((1,), dtype=np.float32)
        node.W = np.zeros((1,), dtype=np.float32)
        node.N_sum = 0.0
        node.win_move = w_mv
        node.expanded = True
        return 1.0

    cands = local_empty_indices(cells, nb_table)
    cand_arr = np.asarray(cands, dtype=np.int32)
    cand_logits = logits[cand_arr]

    if player == BLACK and len(cand_arr) > 0:
        keep_n = min(len(cand_arr), max_children + 8)
        if len(cand_arr) > keep_n:
            top_rel = np.argpartition(cand_logits, -keep_n)[-keep_n:]
            cand_arr = cand_arr[top_rel]
            cand_logits = cand_logits[top_rel]
        # Filter Renju forbidden cells (only cells with >=2 Black stones within dist<=3 on 4 axes can be forbidden)
        legal_idx = []
        for idx in cand_arr.tolist():
            nb_b = 0
            for nb in AX_NB[idx]:
                if cells[nb] == BLACK:
                    nb_b += 1
                    if nb_b >= 2:
                        break
            if nb_b >= 2:
                r, c = divmod(idx, BOARD_SIZE)
                if board.is_black_forbidden(r, c):
                    continue
            legal_idx.append(idx)
        cand_arr = np.asarray(legal_idx, dtype=np.int32)
        cand_logits = logits[cand_arr] if len(cand_arr) > 0 else np.empty((0,), dtype=np.float32)

    # Fallback across full board if local zone had no legal Black move
    if len(cand_arr) == 0:
        mask = exact_legal_mask(board, player)
        full_legal = np.flatnonzero(mask).astype(np.int32)
        if len(full_legal) == 0:
            node.terminal = True
            node.terminal_val = -1.0
            node.expanded = True
            return -1.0
        cand_arr = full_legal
        cand_logits = logits[cand_arr]

    if len(cand_arr) > max_children:
        top_k_rel = np.argpartition(cand_logits, -max_children)[-max_children:]
        cand_arr = cand_arr[top_k_rel]
        cand_logits = cand_logits[top_k_rel]

    m_l = cand_logits - float(np.max(cand_logits))
    exps = np.exp(np.clip(m_l, -25.0, 0.0))
    priors = (exps / max(1e-9, float(exps.sum()))).astype(np.float32)

    node.actions = cand_arr
    node.priors = priors
    node.N = np.zeros((len(cand_arr),), dtype=np.float32)
    node.W = np.zeros((len(cand_arr),), dtype=np.float32)
    node.N_sum = 0.0
    node.expanded = True
    return float(val)


def backprop_and_undo(
    board: RenjuBoard,
    path: List[Tuple[MCTSNode, int, int]],
    leaf_val: float,
    saved_last_mv: Optional[int],
) -> None:
    v = leaf_val
    for idx in range(len(path) - 1, -1, -1):
        parent, a_idx, mv = path[idx]
        v = -v
        parent.N[a_idx] += 1.0
        parent.W[a_idx] += v
        parent.N_sum += 1.0
        board.cells[mv] = EMPTY
    board.last_move = saved_last_mv


def run_mcts_batch(
    games: List[dict],
    params,
    c_eval_batch,
    batch_size: int,
    n_sims: int,
    c_puct: float = 1.5,
    add_noise: bool = False,
    temp_moves: int = 14,
) -> None:
    """Runs `n_sims` batched PUCT MCTS waves concurrently across all active games in `games`."""
    n_active = len(games)
    if n_active == 0:
        return

    buf_x = np.zeros((batch_size, NUM_CELLS, 6), dtype=np.float32)
    root_noise: List[Optional[np.ndarray]] = [None] * n_active

    # Wave 0: expand any unexpanded root nodes in a single GPU batch
    root_slots: List[int] = []
    for g_idx, g in enumerate(games):
        root: MCTSNode = g["root"]
        if not root.expanded:
            root_slots.append(g_idx)
            encode_board_row(g["board"], root.player, None, buf_x[g_idx])

    if root_slots:
        logits_t, vals_t = c_eval_batch(params, buf_x)
        logits_np = np.asarray(logits_t.numpy(), dtype=np.float32)
        vals_np = np.asarray(vals_t.numpy(), dtype=np.float32)
        for g_idx in root_slots:
            g = games[g_idx]
            expand_node_from_logits(
                g["root"],
                g["board"],
                logits_np[g_idx],
                float(vals_np[g_idx]),
                max_children=36,
                nb_table=NB_R3,
            )

    if add_noise:
        for g_idx, g in enumerate(games):
            root = g["root"]
            if len(root.priors) > 1 and root.win_move is None:
                alpha = min(0.5, max(0.15, 10.0 / float(len(root.priors))))
                noise = np.random.dirichlet(np.full(len(root.priors), alpha, dtype=np.float32)).astype(np.float32)
                root_noise[g_idx] = 0.75 * root.priors + 0.25 * noise

    # Batched simulation waves 1..n_sims
    for _ in range(n_sims):
        eval_requests: List[Tuple[int, MCTSNode, List[Tuple[MCTSNode, int, int]], Optional[int]]] = []
        slot = 0

        for g_idx, g in enumerate(games):
            root = g["root"]
            if root.terminal:
                continue
            board: RenjuBoard = g["board"]
            saved_last = board.last_move
            node = root
            path: List[Tuple[MCTSNode, int, int]] = []

            # Descend via PUCT
            while node.expanded and not node.terminal and len(node.actions) > 0:
                n_arr = node.N
                q_arr = np.divide(node.W, n_arr, out=np.zeros_like(n_arr), where=n_arr > 0.0)
                p_arr = root_noise[g_idx] if (node is root and root_noise[g_idx] is not None) else node.priors
                sqrt_tot = math.sqrt(max(1.0, node.N_sum))
                u_arr = c_puct * p_arr * (sqrt_tot / (1.0 + n_arr))
                a_idx = int(np.argmax(q_arr + u_arr))
                mv = int(node.actions[a_idx])

                mover = node.player
                board.cells[mv] = mover
                board.last_move = mv
                path.append((node, a_idx, mv))

                child = node.children.get(mv)
                if child is None:
                    child = MCTSNode(opponent_of(mover))
                    node.children[mv] = child
                    if board.is_win_at(mv, mover):
                        child.terminal = True
                        child.terminal_val = -1.0
                        child.expanded = True
                        break
                node = child

            if node.terminal:
                backprop_and_undo(board, path, node.terminal_val, saved_last)
            elif not node.expanded:
                encode_board_row(board, node.player, None, buf_x[slot])
                eval_requests.append((g_idx, node, path, saved_last))
                slot += 1
            else:
                backprop_and_undo(board, path, 0.0, saved_last)

        if slot > 0:
            logits_t, vals_t = c_eval_batch(params, buf_x)
            logits_np = np.asarray(logits_t.numpy(), dtype=np.float32)
            vals_np = np.asarray(vals_t.numpy(), dtype=np.float32)
            for s_idx, (g_idx, leaf_node, path, saved_last) in enumerate(eval_requests):
                board = games[g_idx]["board"]
                val = expand_node_from_logits(
                    leaf_node,
                    board,
                    logits_np[s_idx],
                    float(vals_np[s_idx]),
                    max_children=24,
                    nb_table=NB_R2,
                )
                backprop_and_undo(board, path, val, saved_last)

    # Select move & record (state, pi_mcts, player) for every active game
    for g in games:
        root = g["root"]
        board = g["board"]
        step = g["step"]

        pi_full = np.zeros(NUM_CELLS, dtype=np.float32)
        if root.win_move is not None:
            chosen = root.win_move
            pi_full[chosen] = 1.0
        elif len(root.actions) == 0:
            chosen = -1
        else:
            visits = root.N.copy()
            v_sum = float(visits.sum())
            if v_sum > 0.0:
                pi_full[root.actions] = visits / v_sum
                if add_noise and step < temp_moves:
                    rel = int(np.random.choice(len(root.actions), p=(visits / v_sum)))
                else:
                    rel = int(np.argmax(visits))
                chosen = int(root.actions[rel])
            else:
                pi_full[root.actions] = root.priors
                chosen = int(root.actions[int(np.argmax(root.priors))])

        g["chosen"] = chosen
        g["pi"] = pi_full


def eval_validation_metal(c_eval_64, params, va_x_nhwc, va_pi, va_z, va_best, va_crit):
    n_val = len(va_x_nhwc)
    all_logits = []
    all_vals = []
    for off in range(0, n_val, 64):
        lg, vl = c_eval_64(params, va_x_nhwc[off : off + 64])
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


def arena_mcts_vs_opponent_parallel(
    openings: List[Tuple[RenjuBoard, int]],
    params,
    c_eval_32,
    opp_fn: Callable[[RenjuBoard, int], Optional[int]],
    n_sims: int,
    max_plies: int = 120,
) -> Tuple[int, int, int, float]:
    """Plays 2 * len(openings) games concurrently (both sides per opening) using batched GPU inference."""
    slots = []
    for sb, sp in openings:
        for nn_side in (sp, opponent_of(sp)):
            slots.append({
                "board": sb.clone(),
                "player": sp,
                "nn_side": nn_side,
                "done": False,
                "winner": 0,
                "plies": 0,
                "root": MCTSNode(sp),
                "step": 20,
            })

    buf_x = np.zeros((32, NUM_CELLS, 6), dtype=np.float32)

    while True:
        # Advance opponent turns immediately
        nn_pending = []
        any_alive = False
        for g in slots:
            while not g["done"]:
                any_alive = True
                if g["plies"] >= max_plies:
                    g["done"] = True
                    g["winner"] = 0
                    break
                b: RenjuBoard = g["board"]
                pl = g["player"]
                if pl == g["nn_side"]:
                    nn_pending.append(g)
                    break
                mv = opp_fn(b, pl)
                if mv is None or not b.is_legal_move(mv, pl):
                    g["done"] = True
                    g["winner"] = opponent_of(pl)
                    break
                b.place_move(mv, pl)
                g["plies"] += 1
                if b.is_win_at(mv, pl):
                    g["done"] = True
                    g["winner"] = pl
                    break
                nxt = opponent_of(pl)
                g["player"] = nxt
                cur_root = g["root"]
                g["root"] = cur_root.children.get(mv, MCTSNode(nxt))

        if not nn_pending:
            if not any_alive or all(g["done"] for g in slots):
                break
            continue

        if n_sims == 0:
            masks = []
            for idx, g in enumerate(nn_pending):
                m = exact_legal_mask(g["board"], g["player"])
                masks.append(m)
                encode_board_row(g["board"], g["player"], m, buf_x[idx])
            logits_t, _ = c_eval_32(params, buf_x)
            logits_np = np.asarray(logits_t.numpy(), dtype=np.float32)
            for idx, g in enumerate(nn_pending):
                b = g["board"]
                pl = g["player"]
                m = masks[idx]
                legal_idx = np.flatnonzero(m)
                if len(legal_idx) == 0:
                    g["done"] = True
                    g["winner"] = opponent_of(pl)
                    continue
                wins = find_immediate_wins_fast(b, pl)
                if wins:
                    mv = wins[0]
                else:
                    mv = int(np.argmax(np.where(m, logits_np[idx], -1e9)))
                b.place_move(mv, pl)
                g["plies"] += 1
                if b.is_win_at(mv, pl):
                    g["done"] = True
                    g["winner"] = pl
                else:
                    g["player"] = opponent_of(pl)
        else:
            for g in nn_pending:
                if g["root"].player != g["player"]:
                    g["root"] = MCTSNode(g["player"])
            run_mcts_batch(nn_pending, params, c_eval_32, batch_size=32, n_sims=n_sims, add_noise=False)
            for g in nn_pending:
                b = g["board"]
                pl = g["player"]
                mv = g["chosen"]
                if mv < 0 or not b.is_legal_move(mv, pl):
                    g["done"] = True
                    g["winner"] = opponent_of(pl)
                    continue
                nxt = opponent_of(pl)
                g["root"] = g["root"].children.get(mv, MCTSNode(nxt))
                b.place_move(mv, pl)
                g["plies"] += 1
                if b.is_win_at(mv, pl):
                    g["done"] = True
                    g["winner"] = pl
                else:
                    g["player"] = nxt

    wins, draws, losses = 0, 0, 0
    for g in slots:
        if g["winner"] == g["nn_side"]:
            wins += 1
        elif g["winner"] == 0:
            draws += 1
        else:
            losses += 1
    wr = (wins + 0.5 * draws) / float(len(slots)) * 100.0
    return wins, draws, losses, wr


def arena_nn_vs_nn_parallel(
    openings: List[Tuple[RenjuBoard, int]],
    params_a,
    params_b,
    c_eval_32,
    n_sims: int,
    max_plies: int = 120,
) -> Tuple[int, int, int, float]:
    """Plays 2 * len(openings) games concurrently between params_a and params_b."""
    slots = []
    for sb, sp in openings:
        for side_a in (sp, opponent_of(sp)):
            slots.append({
                "board": sb.clone(),
                "player": sp,
                "side_a": side_a,
                "done": False,
                "winner": 0,
                "plies": 0,
                "root_a": MCTSNode(sp),
                "root_b": MCTSNode(sp),
                "step": 20,
            })

    buf_x = np.zeros((32, NUM_CELLS, 6), dtype=np.float32)

    while True:
        turn_a = []
        turn_b = []
        for g in slots:
            if g["done"]:
                continue
            if g["plies"] >= max_plies:
                g["done"] = True
                g["winner"] = 0
                continue
            if g["player"] == g["side_a"]:
                turn_a.append(g)
            else:
                turn_b.append(g)
        if not turn_a and not turn_b:
            break

        for batch_list, pr_cur, root_key, other_key in (
            (turn_a, params_a, "root_a", "root_b"),
            (turn_b, params_b, "root_b", "root_a"),
        ):
            if not batch_list:
                continue
            if n_sims == 0:
                masks = []
                for idx, g in enumerate(batch_list):
                    m = exact_legal_mask(g["board"], g["player"])
                    masks.append(m)
                    encode_board_row(g["board"], g["player"], m, buf_x[idx])
                logits_t, _ = c_eval_32(pr_cur, buf_x)
                logits_np = np.asarray(logits_t.numpy(), dtype=np.float32)
                for idx, g in enumerate(batch_list):
                    b = g["board"]
                    pl = g["player"]
                    m = masks[idx]
                    if not np.any(m):
                        g["done"] = True
                        g["winner"] = opponent_of(pl)
                        continue
                    wins = find_immediate_wins_fast(b, pl)
                    mv = wins[0] if wins else int(np.argmax(np.where(m, logits_np[idx], -1e9)))
                    b.place_move(mv, pl)
                    g["plies"] += 1
                    if b.is_win_at(mv, pl):
                        g["done"] = True
                        g["winner"] = pl
                    else:
                        g["player"] = opponent_of(pl)
            else:
                sub = []
                for g in batch_list:
                    r_cur: MCTSNode = g[root_key]
                    if r_cur.player != g["player"]:
                        r_cur = MCTSNode(g["player"])
                        g[root_key] = r_cur
                    sub.append({"board": g["board"], "root": r_cur, "step": g["step"]})
                run_mcts_batch(sub, pr_cur, c_eval_32, batch_size=32, n_sims=n_sims, add_noise=False)
                for g, item in zip(batch_list, sub):
                    b = g["board"]
                    pl = g["player"]
                    mv = item["chosen"]
                    if mv < 0 or not b.is_legal_move(mv, pl):
                        g["done"] = True
                        g["winner"] = opponent_of(pl)
                        continue
                    nxt = opponent_of(pl)
                    g[root_key] = item["root"].children.get(mv, MCTSNode(nxt))
                    g[other_key] = g[other_key].children.get(mv, MCTSNode(nxt))
                    b.place_move(mv, pl)
                    g["plies"] += 1
                    if b.is_win_at(mv, pl):
                        g["done"] = True
                        g["winner"] = pl
                    else:
                        g["player"] = nxt

    wins, draws, losses = 0, 0, 0
    for g in slots:
        if g["winner"] == g["side_a"]:
            wins += 1
        elif g["winner"] == 0:
            draws += 1
        else:
            losses += 1
    wr = (wins + 0.5 * draws) / float(len(slots)) * 100.0
    return wins, draws, losses, wr


def generate_parallel_selfplay_games(
    n_games: int,
    params,
    c_eval_32,
    n_sims: int,
    rng: random.Random,
    max_plies: int = 90,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], Tuple[int, int, int, float]]:
    """Runs `n_games` (=32) full self-play games in parallel on Metal GPU via Batched PUCT MCTS."""
    games = []
    for _ in range(n_games):
        b = RenjuBoard()
        # 0..2 random center-ish opening plies for opening diversity; rest pure MCTS
        n_open = rng.randint(0, 2)
        player = BLACK
        for s_open in range(n_open):
            if s_open == 0:
                mv = 7 * BOARD_SIZE + 7 if rng.random() < 0.65 else (rng.randint(6, 8) * BOARD_SIZE + rng.randint(6, 8))
            else:
                cands = [r * BOARD_SIZE + c for r in range(5, 10) for c in range(5, 10) if b.cells[r * BOARD_SIZE + c] == EMPTY]
                mv = rng.choice(cands)
            b.place_move(mv, player)
            player = opponent_of(player)
        games.append({
            "board": b,
            "root": MCTSNode(player),
            "player": player,
            "step": n_open,
            "traj": [],
            "done": False,
            "winner": 0,
        })

    while True:
        active = [g for g in games if not g["done"]]
        if not active:
            break

        run_mcts_batch(active, params, c_eval_32, batch_size=32, n_sims=n_sims, c_puct=1.5, add_noise=True, temp_moves=14)

        for g in active:
            b: RenjuBoard = g["board"]
            pl = g["player"]
            mv = g["chosen"]
            if mv < 0 or not b.is_legal_move(mv, pl):
                g["done"] = True
                g["winner"] = opponent_of(pl)
                continue

            # Record (NHWC [225, 6], pi_mcts [225], mover)
            s_nhwc = np.empty((NUM_CELLS, 6), dtype=np.float32)
            encode_board_row(b, pl, None, s_nhwc)
            g["traj"].append((s_nhwc, g["pi"], pl))

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
                # Tree reuse (noise is regenerated cleanly on root_noise per move)
                g["root"] = g["root"].children.get(mv, MCTSNode(nxt))

    samples: List[Tuple[np.ndarray, np.ndarray, float]] = []
    bw, ww, dr, tot_m = 0, 0, 0, 0
    for g in games:
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
    return samples, (bw, ww, dr, tot_m / float(max(1, n_games)))


def sample_batch_from_buffer(
    buf_list: List[Tuple[np.ndarray, np.ndarray, float]],
    batch_size: int,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(buf_list)
    xb = np.empty((batch_size, NUM_CELLS, 6), dtype=np.float32)
    pib = np.empty((batch_size, NUM_CELLS), dtype=np.float32)
    zb = np.empty((batch_size,), dtype=np.float32)
    for b_i in range(batch_size):
        s_nhwc, pi, z = buf_list[rng.randrange(n)]
        sym = rng.randrange(8)
        if sym == 0:
            xb[b_i] = s_nhwc
            pib[b_i] = pi
        else:
            sx, px = dihedral_transform_nhwc_row(s_nhwc, pi, sym)
            xb[b_i] = sx
            pib[b_i] = px
        zb[b_i] = z
    return xb, pib, zb


def main() -> None:
    print("=== AlphaGo Zero Batched PUCT MCTS + Metal GPU on D4OverlapPatternNet ===")
    rng = random.Random(20261119)
    np.random.seed(20261119)

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

    arena_rng = random.Random(999)
    openings: List[Tuple[RenjuBoard, int]] = []
    while len(openings) < 10:
        seeded = generate_seeded_board(arena_rng, min_stones=4, max_stones=10)
        if seeded is not None:
            openings.append(seeded)

    rb_weak_rng = random.Random(101)
    rb_weak = lambda b, pl: rule_based_move(b, pl, noise_eps=0.30, rng=rb_weak_rng)
    rb_strong = lambda b, pl: rule_based_move(b, pl, noise_eps=0.0, rng=None)
    rand_rng = random.Random(303)
    def rand_legal_agent(b: RenjuBoard, pl: int) -> Optional[int]:
        cands = [i for i in local_empty_indices(b.cells, NB_R2) if b.is_legal_move(i, pl)]
        if not cands:
            mask = exact_legal_mask(b, pl)
            cands = [int(i) for i in np.flatnonzero(mask)]
        return rand_rng.choice(cands) if cands else None

    model = D4OverlapPatternNet(d_ray=24, d_outer=8, width=64, blocks=3)
    p0 = model.init(rg_random.key(42))
    print(f"Model: {model.name} | Params={count_params(p0)} | Device=metal")

    B = 64
    dummy_x32 = np.zeros((32, NUM_CELLS, 6), dtype=np.float32); dummy_x32[:, :, 4] = 1.0
    dummy_x64 = np.zeros((B, NUM_CELLS, 6), dtype=np.float32); dummy_x64[:, :, 4] = 1.0
    dummy_pi64 = np.full((B, NUM_CELLS), 1.0 / NUM_CELLS, dtype=np.float32)
    dummy_z64 = np.zeros((B,), dtype=np.float32)

    sched_total = 1000
    opt = optim.adamw(optim.cosine_decay(2.2e-3, total_steps=sched_total, warmup_steps=20, min_lr=2.5e-4), weight_decay=1e-4, max_grad_norm=1.0)
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

    def infer_fn(pr, xb):
        return model(pr, xb)

    c_train_64 = rg.compile(train_step, ((p0, st0), (dummy_x64, dummy_pi64, dummy_z64)), device="metal", validate="basic")
    c_eval_32 = rg.compile(infer_fn, (p0, dummy_x32), device="metal", validate="basic")
    c_eval_64 = rg.compile(infer_fn, (p0, dummy_x64), device="metal", validate="basic")

    # =========================================================================
    # PHASE 1: Pure Tabula-Rasa AlphaGo Zero (Random Weights -> MCTS-48 Self-Play)
    # =========================================================================
    print("\n=== Phase 1: Pure Tabula-Rasa AlphaGo Zero (Zero Teacher, Random Init, 10 Gens x 32 Parallel Games, MCTS-48) ===")
    p_az = p0
    st_az = opt.init(p0)
    az_buf: Deque[Tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=80_000)
    p1_history = []

    pl, t1, ct1, vm = eval_validation_metal(c_eval_64, p_az, va_x_nhwc, va_pi, va_z, va_best, va_crit)
    print(f"  Gen  0/10 (Scratch) | ValPLoss={pl:.3f} Top1={t1*100:4.1f}% Crit={ct1*100:4.1f}% ValMSE={vm:.3f}")

    for gen in range(1, 11):
        t_g = time.time()
        new_samples, (bw, ww, dr, avg_m) = generate_parallel_selfplay_games(
            n_games=32, params=p_az, c_eval_32=c_eval_32, n_sims=48, rng=rng, max_plies=85
        )
        sp_sec = time.time() - t_g
        az_buf.extend(new_samples)

        t_tr = time.time()
        buf_list = list(az_buf)
        train_batches = [sample_batch_from_buffer(buf_list, B, rng) for _ in range(60)]
        res = rg.run(c_train_64, (p_az, st_az), train_batches, steps=60)
        (p_az, st_az) = res.state
        tr_sec = time.time() - t_tr

        pl, t1, ct1, vm = eval_validation_metal(c_eval_64, p_az, va_x_nhwc, va_pi, va_z, va_best, va_crit)
        p1_history.append({
            "gen": gen, "spSec": round(sp_sec, 2), "trSec": round(tr_sec, 2),
            "avgMoves": round(avg_m, 1), "bw": bw, "ww": ww, "dr": dr,
            "buf": len(az_buf), "pLoss": round(pl, 3), "top1": round(t1 * 100, 1), "crit": round(ct1 * 100, 1), "vMse": round(vm, 3),
        })
        print(
            f"  Gen {gen:2d}/10 | SP {sp_sec:5.2f}s ({32.0/sp_sec:4.2f}g/s, avg {avg_m:4.1f}m, B/W/D={bw}/{ww}/{dr}) | Train {tr_sec:4.2f}s (buf={len(az_buf):5d}) | ValPL={pl:.3f} Top1={t1*100:4.1f}% Crit={ct1*100:4.1f}% VMSE={vm:.3f}"
        )

    print("\n--- Phase 1 Final Eval (Zero-Teacher Tabula-Rasa AlphaGo Zero, 20 games/matchup) ---")
    p1_eval = {}
    for sims in (0, 64):
        w_r, d_r, l_r, wr_r = arena_mcts_vs_opponent_parallel(openings, p_az, c_eval_32, rand_legal_agent, n_sims=sims)
        w_w, d_w, l_w, wr_w = arena_mcts_vs_opponent_parallel(openings, p_az, c_eval_32, rb_weak, n_sims=sims)
        w_s, d_s, l_s, wr_s = arena_mcts_vs_opponent_parallel(openings, p_az, c_eval_32, rb_strong, n_sims=sims)
        p1_eval[f"sims_{sims}"] = {
            "vsRandom": [w_r, d_r, l_r, round(wr_r, 1)],
            "vsNoisyRB": [w_w, d_w, l_w, round(wr_w, 1)],
            "vsFullRB": [w_s, d_s, l_s, round(wr_s, 1)],
        }
        print(
            f"  [Phase1 sims={sims:2d}] vs Random: {w_r:2d}/{d_r:2d}/{l_r:2d} ({wr_r:5.1f}%) | vs Noisy-RB: {w_w:2d}/{d_w:2d}/{l_w:2d} ({wr_w:5.1f}%) | vs Full-RB: {w_s:2d}/{d_s:2d}/{l_s:2d} ({wr_s:5.1f}%)"
        )

    # =========================================================================
    # PHASE 2: Warm-Start Bootstrap -> Zero-Teacher AlphaZero MCTS Self-Play
    # =========================================================================
    print("\n=== Phase 2: Warm-Start (200 steps B=64 on Metal) -> Pure MCTS-64 Self-Play (8 Gens x 32 Games) ===")
    p_boot = model.init(rg_random.key(42))
    st_boot = opt.init(p_boot)
    boot_buf: List[Tuple[np.ndarray, np.ndarray, float]] = [
        (to_nhwc_flat(tr_x[i : i + 1])[0], tr_pi[i].astype(np.float32), float(tr_z[i]))
        for i in range(n_train)
    ]
    t_b0 = time.time()
    boot_batches = [sample_batch_from_buffer(boot_buf, B, rng) for _ in range(200)]
    res_b = rg.run(c_train_64, (p_boot, st_boot), boot_batches, steps=200)
    (p_boot, st_boot) = res_b.state
    pl, t1, ct1, vm = eval_validation_metal(c_eval_64, p_boot, va_x_nhwc, va_pi, va_z, va_best, va_crit)
    print(f"  [Bootstrap 200st in {time.time()-t_b0:.2f}s] ValPL={pl:.3f} Top1={t1*100:4.1f}% Crit={ct1*100:4.1f}% VMSE={vm:.3f}")

    boot_eval = {}
    for sims in (0, 64, 128):
        w_w, d_w, l_w, wr_w = arena_mcts_vs_opponent_parallel(openings, p_boot, c_eval_32, rb_weak, n_sims=sims)
        w_s, d_s, l_s, wr_s = arena_mcts_vs_opponent_parallel(openings, p_boot, c_eval_32, rb_strong, n_sims=sims)
        boot_eval[f"sims_{sims}"] = {
            "vsNoisyRB": [w_w, d_w, l_w, round(wr_w, 1)],
            "vsFullRB": [w_s, d_s, l_s, round(wr_s, 1)],
        }
        print(
            f"  [Before SP sims={sims:3d}] vs Noisy-RB: {w_w:2d}/{d_w:2d}/{l_w:2d} ({wr_w:5.1f}%) | vs Full-RB Teacher: {w_s:2d}/{d_s:2d}/{l_s:2d} ({wr_s:5.1f}%)"
        )

    # Pure AlphaZero Self-Play from p_boot (anchor a smaller 2,400 bootstrap slice in sliding 100k buffer against forgetting, ZERO RB teacher during SP!)
    p_rl = p_boot
    st_rl = st_boot
    rl_buf: Deque[Tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=100_000)
    # Keep 2,400 initial positions from bootstrap (mixed naturally into large FIFO)
    rl_buf.extend(boot_buf)
    rl_buf.extend(boot_buf)
    p2_history = []

    for gen in range(1, 9):
        t_g = time.time()
        new_samples, (bw, ww, dr, avg_m) = generate_parallel_selfplay_games(
            n_games=32, params=p_rl, c_eval_32=c_eval_32, n_sims=64, rng=rng, max_plies=90
        )
        sp_sec = time.time() - t_g
        rl_buf.extend(new_samples)

        t_tr = time.time()
        buf_list = list(rl_buf)
        train_batches = [sample_batch_from_buffer(buf_list, B, rng) for _ in range(70)]
        res_r = rg.run(c_train_64, (p_rl, st_rl), train_batches, steps=70)
        (p_rl, st_rl) = res_r.state
        tr_sec = time.time() - t_tr

        pl, t1, ct1, vm = eval_validation_metal(c_eval_64, p_rl, va_x_nhwc, va_pi, va_z, va_best, va_crit)
        p2_history.append({
            "gen": gen, "spSec": round(sp_sec, 2), "trSec": round(tr_sec, 2),
            "avgMoves": round(avg_m, 1), "bw": bw, "ww": ww, "dr": dr,
            "buf": len(rl_buf), "pLoss": round(pl, 3), "top1": round(t1 * 100, 1), "crit": round(ct1 * 100, 1), "vMse": round(vm, 3),
        })
        print(
            f"  AZ Gen {gen}/8 | SP {sp_sec:5.2f}s (avg {avg_m:4.1f}m, B/W/D={bw}/{ww}/{dr}) | Train {tr_sec:4.2f}s (buf={len(rl_buf):5d}) | ValPL={pl:.3f} Top1={t1*100:4.1f}% Crit={ct1*100:4.1f}% VMSE={vm:.3f}"
        )

    print("\n--- Phase 2 Post-Self-Play Final Eval vs Rule-Based Teacher (20 games each) ---")
    post_eval = {}
    for sims in (0, 64, 128):
        w_w, d_w, l_w, wr_w = arena_mcts_vs_opponent_parallel(openings, p_rl, c_eval_32, rb_weak, n_sims=sims)
        w_s, d_s, l_s, wr_s = arena_mcts_vs_opponent_parallel(openings, p_rl, c_eval_32, rb_strong, n_sims=sims)
        post_eval[f"sims_{sims}"] = {
            "vsNoisyRB": [w_w, d_w, l_w, round(wr_w, 1)],
            "vsFullRB": [w_s, d_s, l_s, round(wr_s, 1)],
        }
        print(
            f"  [After SP  sims={sims:3d}] vs Noisy-RB: {w_w:2d}/{d_w:2d}/{l_w:2d} ({wr_w:5.1f}%) | vs Full-RB Teacher: {w_s:2d}/{d_s:2d}/{l_s:2d} ({wr_s:5.1f}%)"
        )

    print("\n--- Direct Head-to-Head: Post-AlphaZero Self-Play vs Pre-Self-Play Bootstrap (20 games) ---")
    h2h = {}
    for sims in (0, 64):
        w_h, d_h, l_h, wr_h = arena_nn_vs_nn_parallel(openings, p_rl, p_boot, c_eval_32, n_sims=sims)
        h2h[f"sims_{sims}"] = [w_h, d_h, l_h, round(wr_h, 1)]
        print(f"  [Post-AZ vs Bootstrap | sims={sims:2d}] W/D/L: {w_h:2d}/{d_h:2d}/{l_h:2d} ({wr_h:5.1f}%)")

    out_path = os.path.join(_HERE, "az_mcts_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "phase1_history": p1_history,
            "phase1_eval": p1_eval,
            "phase2_boot_eval": boot_eval,
            "phase2_history": p2_history,
            "phase2_post_eval": post_eval,
            "head_to_head_post_vs_boot": h2h,
        }, f, indent=2)
    print(f"\nSaved AlphaZero MCTS summary to {out_path}")


if __name__ == "__main__":
    main()
