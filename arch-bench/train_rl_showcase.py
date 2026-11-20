"""12-generation Curriculum + Rule-Based Sparring + Self-Play RL showcase for LineAxialHybridNet."""

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

from renju_env import (
    BLACK,
    BOARD_SIZE,
    NUM_CELLS,
    WHITE,
    RenjuBoard,
    board_heuristic_value,
    opponent_of,
    rule_based_move,
    score_candidates,
)
from rg_models import OverlapPatternNet, count_params


def to_nhwc_flat(states_nchw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(states_nchw, (0, 2, 3, 1)).reshape(-1, NUM_CELLS, 6))


def dihedral_samples(
    state_nchw: np.ndarray, policy: np.ndarray, value: float
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Applies all 8 dihedral symmetries to one (6, 15, 15) state and (225,) policy."""
    out: List[Tuple[np.ndarray, np.ndarray, float]] = []
    p2d = policy.reshape(BOARD_SIZE, BOARD_SIZE)
    for sym in range(8):
        k = sym % 4
        flip = sym >= 4
        s = np.rot90(state_nchw, k=k, axes=(1, 2))
        p = np.rot90(p2d, k=k, axes=(0, 1))
        if flip:
            s = np.flip(s, axis=2)
            p = np.flip(p, axis=1)
        s_flat = np.ascontiguousarray(np.transpose(s, (1, 2, 0)).reshape(NUM_CELLS, 6))
        p_flat = np.ascontiguousarray(p.reshape(NUM_CELLS))
        out.append((s_flat, p_flat, float(value)))
    return out


def masked_softmax_np(logits: np.ndarray, mask: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = logits.astype(np.float64).copy()
    x[~mask] = -1e9
    if temp != 1.0:
        x = x / max(1e-4, temp)
    max_v = np.max(x[mask])
    exps = np.zeros_like(x)
    exps[mask] = np.exp(np.clip(x[mask] - max_v, -40.0, 0.0))
    s = exps.sum()
    if s <= 0.0:
        exps[mask] = 1.0 / float(mask.sum())
    else:
        exps /= s
    return exps.astype(np.float32)


def nn_policy_and_lookahead(
    board: RenjuBoard,
    player: int,
    eval_1,
    eval_6,
    params,
    use_lookahead: bool,
    rng: random.Random,
    sample_temp: float,
) -> Tuple[Optional[int], np.ndarray, np.ndarray, float]:
    """Evaluates NN + optional 1-ply Value lookahead.
    Returns (chosen_move, nn_probs [225], lookahead_pi [225], root_value).
    """
    mask = board.legal_mask(player)
    legal_idx = np.flatnonzero(mask)
    if len(legal_idx) == 0:
        return None, np.zeros(NUM_CELLS, dtype=np.float32), np.zeros(NUM_CELLS, dtype=np.float32), 0.0

    state_t = board.encode_tensor(player, mask)
    x1 = to_nhwc_flat(state_t[None, ...])
    logits_t, val_t = eval_1(params, x1)
    logits = np.asarray(logits_t.numpy(), dtype=np.float32)[0]
    root_val = float(np.asarray(val_t.numpy(), dtype=np.float32)[0])
    nn_probs = masked_softmax_np(logits, mask, temp=1.0)

    # Check immediate 5-in-a-row win
    for idx in legal_idx:
        board.cells[idx] = player
        won = board.is_win_at(idx, player)
        board.cells[idx] = 0
        if won:
            win_pi = np.zeros(NUM_CELLS, dtype=np.float32)
            win_pi[idx] = 1.0
            return int(idx), np.maximum(nn_probs * 0.2, win_pi * 0.8), win_pi, 0.99

    if not use_lookahead or len(legal_idx) == 1:
        pi_la = nn_probs.copy()
    else:
        k = min(6, len(legal_idx))
        masked_logits = np.where(mask, logits, -1e9)
        cand_indices = np.argpartition(masked_logits, -k)[-k:]
        batch_x = np.zeros((6, NUM_CELLS, 6), dtype=np.float32)
        opp = opponent_of(player)
        valid_cands: List[int] = []
        for mv in cand_indices:
            mv_i = int(mv)
            if not mask[mv_i]:
                continue
            valid_cands.append(mv_i)
            cb = board.clone()
            cb.place_move(mv_i, player)
            ct = cb.encode_tensor(opp)
            batch_x[len(valid_cands) - 1] = to_nhwc_flat(ct[None, ...])[0]

        _, opp_vals_t = eval_6(params, batch_x)
        opp_vals = np.asarray(opp_vals_t.numpy(), dtype=np.float32)[: len(valid_cands)]

        la_logits = np.full(NUM_CELLS, -1e9, dtype=np.float32)
        for i, mv_i in enumerate(valid_cands):
            # Combine negated opponent value + prior score
            la_logits[mv_i] = 3.0 * float(-opp_vals[i]) + 0.6 * float(logits[mv_i])
        # Soft blending of 1-ply lookahead distribution and NN prior
        la_soft = masked_softmax_np(la_logits, mask, temp=0.6)
        pi_la = (0.75 * la_soft + 0.25 * nn_probs).astype(np.float32)
        s = float(pi_la.sum())
        if s > 0:
            pi_la /= s

    # Sample or pick argmax from pi_la
    if sample_temp <= 0.05:
        chosen = int(np.argmax(np.where(mask, pi_la, -1.0)))
    else:
        sharp = masked_softmax_np(np.log(np.maximum(pi_la, 1e-8)), mask, temp=sample_temp)
        chosen = int(rng.choices(range(NUM_CELLS), weights=sharp.tolist(), k=1)[0])
    return chosen, nn_probs, pi_la, root_val


def play_curriculum_game(
    mode: str,
    rb_noise: float,
    nn_is_black: bool,
    use_lookahead: bool,
    eval_1,
    eval_6,
    params,
    rng: random.Random,
    max_moves: int = 110,
) -> Tuple[int, List[dict], List[Tuple[np.ndarray, np.ndarray, float]], Optional[bool]]:
    """Plays 1 full game starting from an empty board with 0..4 random seeded opening moves.
    Returns (winner, replay_moves, train_samples_list, nn_won_if_sparring).
    """
    b = RenjuBoard()
    player = BLACK
    rand_open_len = rng.randint(0, 4)

    trajectory: List[Tuple[np.ndarray, np.ndarray, float, int]] = []
    replay_moves: List[dict] = []
    winner = 0

    for step in range(max_moves):
        mask = b.legal_mask(player)
        legal_idx = np.flatnonzero(mask)
        if len(legal_idx) == 0:
            winner = opponent_of(player)
            break

        state_t = b.encode_tensor(player, mask)
        is_nn_turn = (mode == "selfplay") or ((player == BLACK) == nn_is_black)

        # Always compute NN display probs + tactical teacher target for buffer quality
        sample_temp = 0.65 if step < 6 else 0.04
        nn_mv, nn_probs, pi_la, root_val = nn_policy_and_lookahead(
            b, player, eval_1, eval_6, params, use_lookahead, rng, sample_temp
        )

        scored, _ = score_candidates(b, player, mask)
        teacher_pi = np.zeros(NUM_CELLS, dtype=np.float32)
        if scored:
            top = scored[: min(16, len(scored))]
            max_s = top[0][1]
            temp = 350.0 if max_s < 50_000 else max_s * 0.08
            exps = [math.exp(max(-30.0, (s - max_s) / temp)) for _, s in top]
            z_sum = sum(exps)
            for (idx_c, _), e in zip(top, exps):
                teacher_pi[idx_c] = e / z_sum
        else:
            teacher_pi[mask] = 1.0 / float(mask.sum())

        if step < rand_open_len:
            if step == 0 and rng.random() < 0.6:
                chosen = 7 * BOARD_SIZE + 7
            else:
                pool = [
                    r * BOARD_SIZE + c
                    for r in range(5, 10)
                    for c in range(5, 10)
                    if mask[r * BOARD_SIZE + c]
                ]
                chosen = rng.choice(pool) if pool else int(legal_idx[0])
        elif is_nn_turn:
            chosen = nn_mv if nn_mv is not None else int(legal_idx[0])
        else:
            rb_mv = rule_based_move(b, player, noise_eps=rb_noise, rng=rng)
            chosen = rb_mv if rb_mv is not None else int(legal_idx[0])

        # Hybrid DAgger / MCTS-lite target
        target_pi = (0.65 * teacher_pi + 0.35 * pi_la).astype(np.float32)
        pi_sum = float(target_pi.sum())
        if pi_sum > 0:
            target_pi /= pi_sum

        heur_v = board_heuristic_value(b, player)
        trajectory.append((state_t, target_pi, heur_v, player))

        mover_type = (
            "opening"
            if step < rand_open_len
            else ("nn" if is_nn_turn else "rule")
        )

        # Top-12 post-lookahead policy probabilities (> 0.005) for replay UI
        top_cands = []
        ranked_idx = np.argsort(-pi_la)
        for idx_p in ranked_idx[:12]:
            prob = float(pi_la[idx_p])
            if prob <= 0.005:
                break
            pr, pc = divmod(int(idx_p), BOARD_SIZE)
            top_cands.append({
                "row": pr,
                "col": pc,
                "prob": round(prob, 3),
            })

        r, c = divmod(chosen, BOARD_SIZE)
        replay_moves.append({
            "row": r,
            "col": c,
            "player": player,
            "moverType": mover_type,
            "value": round(root_val, 3),
            "topPolicies": top_cands,
        })

        b.place_move(chosen, player)
        if b.is_win_at(chosen, player):
            winner = player
            break
        player = opponent_of(player)

    samples: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for state_t, target_pi, heur_v, mover in trajectory:
        if winner == 0:
            outcome_z = 0.0
        elif winner == mover:
            outcome_z = 1.0
        else:
            outcome_z = -1.0
        target_v = 0.5 * outcome_z + 0.5 * heur_v
        samples.extend(dihedral_samples(state_t, target_pi, target_v))

    nn_won: Optional[bool] = None
    if mode == "sparring":
        nn_player = BLACK if nn_is_black else WHITE
        nn_won = winner == nn_player

    return winner, replay_moves, samples, nn_won


def format_game_label(gen: int, winner: int, num_moves: int) -> str:
    outcome = "흑승" if winner == BLACK else "백승" if winner == WHITE else "무승부"
    return f"Gen {gen} · {num_moves}수 {outcome}"


def main() -> None:
    print("=== OverlapPattern 12-Gen Curriculum + Sparring + Self-Play RL Showcase ===")
    rng = random.Random(20261119)

    model = OverlapPatternNet(d_ray=24, d_outer=8, width=64, blocks=3, use_blur=False)
    params = model.init(rg_random.key(42))
    n_params = count_params(params)
    print(f"Model: {model.name} | d_ray=24 d_outer=8 width=64 blocks=3 | params={n_params}")

    iters = 12
    games_per_iter = 10
    steps_per_iter = 40
    total_steps = iters * steps_per_iter
    batch_size = 32
    max_buffer = 24_000

    schedule = optim.cosine_decay(2.4e-3, total_steps=total_steps, warmup_steps=20, min_lr=3e-4)
    opt = optim.adamw(schedule, weight_decay=1e-4, max_grad_norm=1.0)
    opt_state = opt.init(params)

    dummy_x = np.zeros((batch_size, NUM_CELLS, 6), dtype=np.float32)
    dummy_x[:, :, 4] = 1.0
    dummy_pi = np.full((batch_size, NUM_CELLS), 1.0 / NUM_CELLS, dtype=np.float32)
    dummy_z = np.zeros((batch_size,), dtype=np.float32)

    def train_step(state, batch):
        pr_in, st_in = state
        xb, pib, zb = batch

        def loss_fn(pr):
            logits, val = model(pr, xb)
            lp = nn.log_softmax(logits, axis=-1)
            pl = -(pib * lp).sum(axis=-1).mean()
            vl = ((val - zb) ** 2).mean()
            return pl + 0.6 * vl, (pl, vl)

        (total_l, (pl, vl)), g = rg.value_and_grad(loss_fn, has_aux=True)(pr_in)
        pr_out, st_out = opt.update(g, st_in, pr_in)
        return (pr_out, st_out), {"p": pl, "v": vl}

    def infer_fn(pr, xb):
        return model(pr, xb)

    c_train = rg.compile(train_step, ((params, opt_state), (dummy_x, dummy_pi, dummy_z)), device="cpu", validate="basic")
    c_eval_1 = rg.compile(infer_fn, (params, dummy_x[:1]), device="cpu", validate="basic")
    c_eval_6 = rg.compile(infer_fn, (params, dummy_x[:6]), device="cpu", validate="basic")

    # Pre-seed replay buffer with dataset_2026.npz (augmented 8x)
    replay_buffer: Deque[Tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=max_buffer)
    cache_path = os.path.join(_HERE, "dataset_2026.npz")
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        seed_count = min(600, len(d["x"]))
        for idx in range(seed_count):
            replay_buffer.extend(dihedral_samples(d["x"][idx], d["pi"][idx], float(d["z"][idx])))
        print(f"Pre-seeded replay buffer with {len(replay_buffer)} curriculum samples from dataset_2026.npz")

    metrics: List[dict] = []
    archived_games: List[dict] = []

    for gen in range(1, iters + 1):
        t0 = time.time()
        if gen <= 4:
            rb_noise = 0.35
            rb_label = "룰베이스(초급)"
            n_spar = 6
            use_la = False
        elif gen <= 8:
            rb_noise = 0.15
            rb_label = "룰베이스(중급)"
            n_spar = 5
            use_la = True
        else:
            rb_noise = 0.0
            rb_label = "룰베이스(Full-RB)"
            n_spar = 4
            use_la = True

        black_wins = 0
        white_wins = 0
        draws = 0
        total_moves = 0
        spar_wins = 0
        spar_games = 0
        gen_games: List[Tuple[int, List[dict], Optional[bool], str, bool, str]] = []

        for g_idx in range(games_per_iter):
            mode = "sparring" if g_idx < n_spar else "selfplay"
            nn_is_black = (g_idx % 2 == 0)
            winner, mv_list, samples, nn_won = play_curriculum_game(
                mode=mode,
                rb_noise=rb_noise,
                nn_is_black=nn_is_black,
                use_lookahead=use_la,
                eval_1=c_eval_1,
                eval_6=c_eval_6,
                params=params,
                rng=rng,
            )
            replay_buffer.extend(samples)
            total_moves += len(mv_list)
            if winner == BLACK:
                black_wins += 1
            elif winner == WHITE:
                white_wins += 1
            else:
                draws += 1
            if nn_won is not None:
                spar_games += 1
                if nn_won:
                    spar_wins += 1
            gen_games.append((winner, mv_list, nn_won, mode, nn_is_black, rb_label))

        sp_sec = time.time() - t0

        # Pick best showcase game: prefer an NN win in sparring or decisive self-play game
        nn_win_games = [g for g in gen_games if g[2] is True]
        decisive_games = [g for g in gen_games if g[0] != 0]
        if nn_win_games:
            showcase = min(nn_win_games, key=lambda g: len(g[1]))
        elif decisive_games:
            showcase = min(decisive_games, key=lambda g: len(g[1]))
        else:
            showcase = gen_games[0]

        s_win, s_moves, _, s_mode, s_nn_black, s_rb_label = showcase
        nn_name = f"OverlapPattern (Gen {gen})"
        if s_mode == "selfplay":
            black_player = nn_name
            white_player = nn_name
            matchup = "자가대국"
        else:
            black_player = nn_name if s_nn_black else s_rb_label
            white_player = s_rb_label if s_nn_black else nn_name
            matchup = "룰베이스 스파링"

        outcome = "흑승" if s_win == BLACK else "백승" if s_win == WHITE else "무승부"
        archived_games.append({
            "id": f"gen-{gen}",
            "label": f"Gen {gen} ({matchup}) · {len(s_moves)}수 {outcome}",
            "generation": gen,
            "winner": s_win,
            "blackPlayer": black_player,
            "whitePlayer": white_player,
            "matchup": matchup,
            "moves": s_moves,
        })

        # Train 40 batches of B=32 on replay buffer
        t_tr = time.time()
        buf_list = list(replay_buffer)
        buf_n = len(buf_list)
        train_batches: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for _ in range(steps_per_iter):
            idxs = [rng.randrange(buf_n) for _ in range(batch_size)]
            xb = np.stack([buf_list[k][0] for k in idxs], axis=0).astype(np.float32)
            pib = np.stack([buf_list[k][1] for k in idxs], axis=0).astype(np.float32)
            zb = np.array([buf_list[k][2] for k in idxs], dtype=np.float32)
            train_batches.append((xb, pib, zb))

        res = rg.run(c_train, (params, opt_state), train_batches, steps=steps_per_iter)
        (params, opt_state) = res.state
        train_sec = time.time() - t_tr

        pl_hist = [float(x) for x in res.history["aux.p"]]
        vl_hist = [float(x) for x in res.history["aux.v"]]
        avg_p = sum(pl_hist) / max(1, len(pl_hist))
        avg_v = sum(vl_hist) / max(1, len(vl_hist))
        avg_moves = total_moves / float(games_per_iter)

        metrics.append({
            "generation": gen,
            "pLoss": round(avg_p, 3),
            "vLoss": round(avg_v, 3),
            "avgMoves": round(avg_moves, 1),
            "blackWins": black_wins,
            "whiteWins": white_wins,
            "draws": draws,
            "spSec": round(sp_sec, 2),
            "trainSec": round(train_sec, 2),
            "bufferSize": buf_n,
        })

        print(
            f"Gen {gen:2d}/{iters} | SP: {sp_sec:5.2f}s (avg {avg_moves:4.1f}m, B/W/D={black_wins}/{white_wins}/{draws}, vsRB={spar_wins}/{spar_games}) | Train: {train_sec:4.2f}s (buf={buf_n:5d}, pLoss={avg_p:.3f}, vLoss={avg_v:.3f})"
        )

    out_path = os.path.join(_HERE, "web_data", "rl_replays.json")
    archive = {
        "rules": "renju",
        "modelInfo": {
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
    print(f"Saved 12-gen OverlapPattern showcase to {out_path} ({len(archived_games)} games)")


if __name__ == "__main__":
    main()
