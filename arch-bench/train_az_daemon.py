"""Rust MCTS + Metal GPU AlphaZero League Trainer with Virtual Loss, Champion Guard & Live Web Stream."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))  # arch-bench/
sys.path.insert(0, _HERE)
import rg
from rg import nn, optim, random as rg_random

from alphazero_mcts import (
    eval_validation_metal,
    to_nhwc_flat,
)
from export_az_web_replays import (
    format_top_policies,
    masked_softmax_logits,
    mcts_root_value,
)
from renju_env import (
    BLACK,
    BOARD_SIZE,
    NUM_CELLS,
    WHITE,
    RenjuBoard,
    generate_dataset,
    generate_seeded_board,
    opponent_of,
)
from rg_models import D4OverlapPatternV2Net, count_params
from rust_mcts_ext import (
    RustMctsPool,
    arena_mcts_vs_rb_rust,
    arena_nn_vs_nn_rust,
    encode_board_nhwc_rust,
    exact_legal_mask_rust,
    rule_based_move_rust,
    sample_d4_batch_rust,
)

CKPT_DIR = os.path.join(_HERE, "checkpoints")
MANIFEST_PATH = os.path.join(_HERE, "checkpoints", "manifest.json")
BUFFER_PATH = os.path.join(_HERE, "checkpoints", "rl_buffer.npz")
LIVE_JSON_PATH = os.path.join(_HERE, "web_data", "live_training.json")
REPLAYS_JSON_PATH = os.path.join(_HERE, "web_data", "rl_replays.json")
EVAL_MATCHES_PATH = os.path.join(_HERE, "web_data", "eval_matches.json")
LEGACY_LIVE_JSON = os.path.join(_HERE, "..", "public", "live_training.json")
LEGACY_REPLAYS_JSON = os.path.join(_HERE, "..", "public", "rl_replays.json")
SEED_CKPT_PATH = os.path.join(_HERE, "d4_overlap_az.npz")

MAX_SP_BUFFER = 30_000
OPT_STATE_PATH = os.path.join(_HERE, "checkpoints", "opt_state.npz")
# Continuing RL wants a steady step size: a cosine that hits its floor ~40 gens after every
# resume froze learning (gens 277-307 never beat Gen 276).
RL_LR = 1.5e-4
# Without a new champion for this many gens, restart from the champion's weights.
STALL_GENS = 8
# Fraction of each batch taken from the rule-based teacher set; it caps a net that outgrew it.
TEACHER_SHARE = 1 / 64
# Weight of the root search value inside the value target (the rest is the game result).
Q_MIX = 0.15

# 12 genuinely distinct classic center 3-move shapes around H8=(7,7)
CLASSIC_3MOVE_OPENINGS: List[Tuple[str, List[Tuple[int, int]]]] = [
    ("한성형", [(7, 7), (6, 7), (5, 7)]),
    ("계성형", [(7, 7), (6, 7), (5, 8)]),
    ("협성형", [(7, 7), (6, 7), (6, 8)]),
    ("서성형", [(7, 7), (6, 7), (7, 8)]),
    ("명성형", [(7, 7), (6, 7), (8, 8)]),
    ("수성형", [(7, 7), (6, 7), (9, 7)]),
    ("장성형", [(7, 7), (6, 8), (5, 9)]),
    ("운성형", [(7, 7), (6, 8), (6, 9)]),
    ("우성형", [(7, 7), (6, 8), (7, 9)]),
    ("금성형", [(7, 7), (6, 8), (8, 8)]),
    ("유성형", [(7, 7), (6, 8), (6, 6)]),
    ("산성형", [(7, 7), (6, 8), (9, 9)]),
]


def save_checkpoint(params, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw_leaves = [np.asarray(leaf.numpy(), dtype=np.float32) for leaf in rg.tree.leaves(params)]
    leaves = {f"p_{i}": arr for i, arr in enumerate(raw_leaves)}
    np.savez_compressed(path, **leaves)

    bin_path = path[:-4] + ".bin" if path.endswith(".npz") else f"{path}.bin"
    flat = np.concatenate([arr.reshape(-1) for arr in raw_leaves]).astype(np.float32)
    tmp_bin = f"{bin_path}.tmp"
    with open(tmp_bin, "wb") as fb:
        fb.write(flat.tobytes())
    os.replace(tmp_bin, bin_path)


def wipe_training_state() -> None:
    if os.path.isdir(CKPT_DIR):
        for fname in os.listdir(CKPT_DIR):
            if fname.endswith((".npz", ".bin", ".json", ".tmp")):
                try:
                    os.remove(os.path.join(CKPT_DIR, fname))
                except OSError:
                    pass
    for p in (LIVE_JSON_PATH, REPLAYS_JSON_PATH, EVAL_MATCHES_PATH, LEGACY_LIVE_JSON, LEGACY_REPLAYS_JSON):
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def write_checkpoint_manifest(ckpt_rows: List[dict], champ_gen: int, n_params: int) -> None:
    entries = []
    for c in ckpt_rows:
        g = int(c["gen"])
        fname = str(c.get("file", f"gen_{g:03d}.npz"))
        bin_name = fname[:-4] + ".bin" if fname.endswith(".npz") else f"{fname}.bin"
        entries.append({
            "id": str(c.get("id", f"ckpt-{g}")),
            "gen": g,
            "file": fname,
            "binFile": bin_name,
            "totalGames": int(c.get("totalGames", 0)),
            "elo": int(c.get("elo", 1000)),
            "crit": float(c.get("crit", 0.0)),
            "vsRbRecord": str(c.get("vsRbRecord", "-")),
            "vsRbWr": float(c.get("vsRbWr", 0.0)),
            "isChampion": g == int(champ_gen),
        })
    write_json_atomic(
        MANIFEST_PATH,
        {
            "modelName": "D4OverlapPatternV2 (3-Head) + Adaptive MCTS",
            "params": int(n_params),
            "championGen": int(champ_gen),
            "updatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "checkpoints": entries,
        },
    )


def save_tree(tree_obj, path: str) -> None:
    """Optimizer state (or any pytree) as plain leaves; atomic so a kill mid-write can't corrupt it."""
    leaves = {f"p_{i}": np.asarray(leaf.numpy(), dtype=np.float32) for i, leaf in enumerate(rg.tree.leaves(tree_obj))}
    tmp = f"{path}.tmp.npz"
    np.savez_compressed(tmp, **leaves)
    os.replace(tmp, path)


def load_tree(template, path: str):
    """Load a pytree saved by save_tree; None if missing or its shape no longer matches."""
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path)
        tmpl = rg.tree.leaves(template)
        if len(data.files) != len(tmpl):
            return None
        leaves = []
        for i, t in enumerate(tmpl):
            arr = data[f"p_{i}"]
            if tuple(arr.shape) != tuple(np.asarray(t.numpy()).shape):
                return None
            leaves.append(rg.tensor(arr))
        return rg.tree.unflatten(rg.tree.structure(template), leaves)
    except Exception:
        return None


def load_checkpoint(p_template, path: str):
    data = np.load(path)
    n_leaves = len(rg.tree.leaves(p_template))
    leaves = [rg.tensor(data[f"p_{i}"]) for i in range(n_leaves)]
    return rg.tree.unflatten(rg.tree.structure(p_template), leaves)


def write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_json_safe(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def pick_diverse_opening(rng: random.Random) -> Tuple[str, List[int]]:
    """50% Rule-compliant Random Cluster (4-12 stones) + 30% Classic 12 Shapes + 20% Tengen."""
    u = rng.random()
    if u < 0.30:
        name, coords = rng.choice(CLASSIC_3MOVE_OPENINGS)
        return name, [r * BOARD_SIZE + c for r, c in coords]
    if u < 0.50:
        return "천원 개시", [7 * BOARD_SIZE + 7]

    # Rule-compliant local random cluster: keeps stones connected while creating unseen shapes
    b = RenjuBoard()
    n_open = rng.randint(4, 12)
    seq: List[int] = []
    pl = BLACK
    placed_coords: List[Tuple[int, int]] = []
    for k in range(n_open):
        if k == 0:
            r0 = rng.randint(5, 9)
            c0 = rng.randint(5, 9)
            mv = r0 * BOARD_SIZE + c0
        else:
            cand_set = set()
            for pr, pc in placed_coords:
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        nr, nc = pr + dr, pc + dc
                        if 2 <= nr <= BOARD_SIZE - 3 and 2 <= nc <= BOARD_SIZE - 3:
                            idx = nr * BOARD_SIZE + nc
                            if b.is_legal_move(idx, pl):
                                cand_set.add(idx)
            cands = list(cand_set)
            rng.shuffle(cands)
            mv = -1
            for cand in cands[:24]:
                bt = b.clone()
                bt.place_move(cand, pl)
                if not bt.is_win_at(cand, pl):
                    mv = cand
                    break
            if mv < 0:
                break
        b.place_move(mv, pl)
        placed_coords.append(divmod(mv, BOARD_SIZE))
        seq.append(mv)
        pl = opponent_of(pl)
    if not seq:
        return "천원 개시", [7 * BOARD_SIZE + 7]
    return f"랜덤 {len(seq)}수", seq


def load_eval_matches() -> Dict[str, dict]:
    doc = read_json_safe(EVAL_MATCHES_PATH)
    if not doc or not isinstance(doc.get("matches"), dict):
        return {}
    return dict(doc["matches"])


def save_eval_matches(matches_db: Dict[str, dict]) -> None:
    write_json_atomic(EVAL_MATCHES_PATH, {"matches": matches_db})


def record_eval_match(matches_db: Dict[str, dict], a: str, b: str, wa: int, d: int, wb: int) -> None:
    if a == b or (wa + d + wb) <= 0:
        return
    if a > b:
        a, b = b, a
        wa, wb = wb, wa
    key = f"{a}__vs__{b}"
    cur = matches_db.get(key)
    if cur is None:
        matches_db[key] = {"a": a, "b": b, "wa": int(wa), "d": int(d), "wb": int(wb)}
    else:
        cur["wa"] = int(cur.get("wa", 0)) + int(wa)
        cur["d"] = int(cur.get("d", 0)) + int(d)
        cur["wb"] = int(cur.get("wb", 0)) + int(wb)


def solve_bradley_terry_elo(
    matches_db: Dict[str, dict], anchor: str = "gen_000", anchor_elo: float = 1000.0
) -> Dict[str, int]:
    """True AlphaZero-style Bradley-Terry MLE ratings from accumulated head-to-head match records."""
    players = set()
    for m in matches_db.values():
        players.add(str(m["a"]))
        players.add(str(m["b"]))
    if not players:
        return {anchor: int(round(anchor_elo))}
    players.add(anchor)
    plist = sorted(players)
    p = {name: 1.0 for name in plist}

    # Tiny virtual draw prior (0.25 each way vs anchor) prevents divergence on 100%/0% records
    prior_eps = 0.25
    for _ in range(250):
        max_delta = 0.0
        for name in plist:
            wins = 0.5 * prior_eps if name != anchor else 0.0
            denom = (prior_eps / (p[name] + p[anchor])) if name != anchor else 0.0
            for m in matches_db.values():
                a, b = str(m["a"]), str(m["b"])
                if a != name and b != name:
                    continue
                wa, d, wb = float(m["wa"]), float(m["d"]), float(m["wb"])
                n_games = wa + d + wb
                if n_games <= 0.0:
                    continue
                other = b if a == name else a
                w_self = (wa if a == name else wb) + 0.5 * d
                wins += w_self
                denom += n_games / (p[name] + p[other])
            if denom > 0.0 and wins > 0.0:
                new_val = wins / denom
                max_delta = max(max_delta, abs(new_val - p[name]) / max(1e-9, p[name]))
                p[name] = new_val
        scale = p[anchor]
        if scale > 0.0:
            for name in plist:
                p[name] /= scale
        if max_delta < 1e-7:
            break

    return {
        name: int(round(anchor_elo + 400.0 * math.log10(max(1e-6, p[name]))))
        for name in plist
    }


class SelfPlayBuffer:
    """Persistent circular replay buffer storing (NHWC [u8], policy [f32], z_mix [f32], surprise [f32], turns [f32])."""

    def __init__(self, max_size: int = MAX_SP_BUFFER):
        self.max_size = max_size
        self.s = np.empty((0, NUM_CELLS, 6), dtype=np.uint8)
        self.pi = np.empty((0, NUM_CELLS), dtype=np.float32)
        self.z = np.empty((0,), dtype=np.float32)
        self.surprise = np.empty((0,), dtype=np.float32)
        self.turns = np.empty((0,), dtype=np.float32)

    def __len__(self) -> int:
        return int(self.z.shape[0])

    def extend(self, samples: List[Tuple[np.ndarray, np.ndarray, float, float, float]]) -> None:
        if not samples:
            return
        new_s = np.stack([np.clip(x[0], 0.0, 1.0).astype(np.uint8) for x in samples], axis=0)
        new_pi = np.stack([x[1].astype(np.float32) for x in samples], axis=0)
        new_z = np.asarray([x[2] for x in samples], dtype=np.float32)
        new_surp = np.asarray([x[3] for x in samples], dtype=np.float32)
        new_turns = np.asarray([x[4] for x in samples], dtype=np.float32)
        if len(self) == 0:
            self.s, self.pi, self.z, self.surprise, self.turns = new_s, new_pi, new_z, new_surp, new_turns
        else:
            self.s = np.concatenate([self.s, new_s], axis=0)
            self.pi = np.concatenate([self.pi, new_pi], axis=0)
            self.z = np.concatenate([self.z, new_z], axis=0)
            self.surprise = np.concatenate([self.surprise, new_surp], axis=0)
            self.turns = np.concatenate([self.turns, new_turns], axis=0)
        if len(self) > self.max_size:
            drop = len(self) - self.max_size
            self.s = np.ascontiguousarray(self.s[drop:])
            self.pi = np.ascontiguousarray(self.pi[drop:])
            self.z = np.ascontiguousarray(self.z[drop:])
            self.surprise = np.ascontiguousarray(self.surprise[drop:])
            self.turns = np.ascontiguousarray(self.turns[drop:])

    def save(self, path: str) -> None:
        if len(self) == 0:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp.npz"
        np.savez_compressed(tmp, s=self.s, pi=self.pi, z=self.z, surprise=self.surprise, turns=self.turns)
        os.replace(tmp, path)

    def load(self, path: str) -> int:
        if not os.path.exists(path):
            return 0
        try:
            d = np.load(path)
            self.s = np.ascontiguousarray(d["s"][-self.max_size :], dtype=np.uint8)
            self.pi = np.ascontiguousarray(d["pi"][-self.max_size :], dtype=np.float32)
            self.z = np.ascontiguousarray(d["z"][-self.max_size :], dtype=np.float32)
            if "surprise" in d:
                self.surprise = np.ascontiguousarray(d["surprise"][-self.max_size :], dtype=np.float32)
            else:
                self.surprise = np.full((len(self.z),), 0.25, dtype=np.float32)
            if "turns" in d:
                self.turns = np.ascontiguousarray(d["turns"][-self.max_size :], dtype=np.float32)
            else:
                self.turns = np.full((len(self.z),), 0.5, dtype=np.float32)
            return len(self)
        except Exception:
            return 0


def sample_anchored_batch(
    t_s: np.ndarray,
    t_pi: np.ndarray,
    t_z: np.ndarray,
    sp_buf: SelfPlayBuffer,
    batch_size: int,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """25% tactical teacher + 75% MCTS (50% recent 4096, 25% uniform, 25% high-surprise) with Rust D4 aug."""
    n_sp = len(sp_buf)
    n_t = int(t_z.shape[0])
    if n_sp < 48:
        t_idx = np.asarray([rng.randrange(n_t) for _ in range(batch_size)], dtype=np.int32)
        t_sym = np.asarray([rng.randrange(8) for _ in range(batch_size)], dtype=np.uint8)
        x0, pi0, z0 = sample_d4_batch_rust(t_s, t_pi, t_z, t_idx, t_sym)
        return (
            x0,
            pi0,
            z0,
            np.full((batch_size,), 0.5, dtype=np.float32),
            np.zeros((batch_size,), dtype=np.float32),
        )

    n_teacher_b = max(1, int(round(batch_size * TEACHER_SHARE)))  # 1 of 64: the net outgrew the teacher
    n_sp_b = batch_size - n_teacher_b       # 58 when B=64
    n_recent = n_sp_b // 2                  # 29
    n_surp = n_sp_b // 4                    # 14 (high surprise / blunder replay)
    n_old = n_sp_b - n_recent - n_surp      # 15

    t_idx = np.asarray([rng.randrange(n_t) for _ in range(n_teacher_b)], dtype=np.int32)
    t_sym = np.asarray([rng.randrange(8) for _ in range(n_teacher_b)], dtype=np.uint8)
    xt, pit, zt = sample_d4_batch_rust(t_s, t_pi, t_z, t_idx, t_sym)
    tt = np.full((n_teacher_b,), 0.5, dtype=np.float32)
    twt = np.zeros((n_teacher_b,), dtype=np.float32)

    recent_start = max(0, n_sp - 4096)
    rec_indices = [rng.randrange(recent_start, n_sp) for _ in range(n_recent)]
    old_indices = [rng.randrange(n_sp) for _ in range(n_old)]

    # Surprise-weighted sampling: pick top candidate among 4 random draws
    surp_indices: List[int] = []
    surp_arr = sp_buf.surprise
    for _ in range(n_surp):
        best_i = rng.randrange(n_sp)
        best_s = float(surp_arr[best_i])
        for _ in range(3):
            cand_i = rng.randrange(n_sp)
            cand_s = float(surp_arr[cand_i])
            if cand_s > best_s:
                best_i, best_s = cand_i, cand_s
        surp_indices.append(best_i)

    sp_idx = np.asarray(rec_indices + old_indices + surp_indices, dtype=np.int32)
    sp_sym = np.asarray([rng.randrange(8) for _ in range(n_sp_b)], dtype=np.uint8)

    sel_s = np.ascontiguousarray(sp_buf.s[sp_idx], dtype=np.float32)
    sel_pi = np.ascontiguousarray(sp_buf.pi[sp_idx], dtype=np.float32)
    sel_z = np.ascontiguousarray(sp_buf.z[sp_idx], dtype=np.float32)
    ts = np.ascontiguousarray(sp_buf.turns[sp_idx], dtype=np.float32)
    tws = np.ones((n_sp_b,), dtype=np.float32)
    local_idx = np.arange(n_sp_b, dtype=np.int32)
    xs, pis, zs = sample_d4_batch_rust(sel_s, sel_pi, sel_z, local_idx, sp_sym)

    return (
        np.ascontiguousarray(np.concatenate([xt, xs], axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate([pit, pis], axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate([zt, zs], axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate([tt, ts], axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate([twt, tws], axis=0), dtype=np.float32),
    )


def evaluate_checkpoint_tuple(
    params,
    c_eval_32,
    c_eval_64,
    va_x_nhwc: np.ndarray,
    va_pi: np.ndarray,
    va_z: np.ndarray,
    va_best: np.ndarray,
    va_crit: np.ndarray,
    eval_openings: List[Tuple[RenjuBoard, int]],
    rb_sims: int = 32,
) -> Tuple[float, float, int, int, int, float]:
    """Returns (p_loss, crit_pct, w_rb, d_rb, l_rb, wr_rb)."""
    pl_v, _, crit_v, _ = eval_validation_metal(c_eval_64, params, va_x_nhwc, va_pi, va_z, va_best, va_crit)
    w_rb, d_rb, l_rb, wr_rb = arena_mcts_vs_rb_rust(eval_openings, params, c_eval_32, n_sims=rb_sims)
    return pl_v, round(crit_v * 100.0, 1), w_rb, d_rb, l_rb, round(wr_rb, 1)


def reconcile_history_and_replays(
    n_parallel: int,
    n_params: int,
    p0,
    p_base,
    c_eval_32,
    c_eval_64,
    va_x_nhwc: np.ndarray,
    va_pi: np.ndarray,
    va_z: np.ndarray,
    va_best: np.ndarray,
    va_crit: np.ndarray,
    eval_openings: List[Tuple[RenjuBoard, int]],
    fresh: bool = False,
) -> Tuple[int, List[dict], List[dict], dict, dict]:
    """Merges web_data + legacy public JSONs and fills any missing checkpoint rows on disk."""
    live_docs = [] if fresh else [d for d in (read_json_safe(LEGACY_LIVE_JSON), read_json_safe(LIVE_JSON_PATH)) if d]
    rep_docs = [] if fresh else [d for d in (read_json_safe(LEGACY_REPLAYS_JSON), read_json_safe(REPLAYS_JSON_PATH)) if d]

    hist_by_gen: Dict[int, dict] = {}
    ckpt_by_gen: Dict[int, dict] = {}
    games_by_id: Dict[str, dict] = {}
    latest_live_game = {
        "matchup": "초기화 중",
        "opening": "천원",
        "blackPlayer": "Gen 0",
        "whitePlayer": "Gen 0",
        "step": 0,
        "evalBlack": 0.0,
        "moves": [],
        "topPolicies": [],
    }

    for doc in live_docs:
        for h in doc.get("history", []):
            g = int(h["gen"])
            hist_by_gen[g] = dict(h)
        for c in doc.get("checkpoints", []):
            g = int(c["gen"])
            ckpt_by_gen[g] = dict(c)
        if isinstance(doc.get("liveGame"), dict) and doc["liveGame"].get("moves"):
            latest_live_game = doc["liveGame"]

    for doc in rep_docs:
        for gm in doc.get("games", []):
            gid = str(gm.get("id", ""))
            if gid:
                games_by_id[gid] = gm

    existing_ckpts = sorted(
        f for f in os.listdir(CKPT_DIR) if f.startswith("gen_") and f.endswith(".npz")
    )
    disk_gens = [int(f.replace("gen_", "").replace(".npz", "")) for f in existing_ckpts]

    for gen in disk_gens:
        ckpt_file = f"gen_{gen:03d}.npz"
        replay_id = f"league-gen-{gen}"
        has_replay = replay_id in games_by_id
        if gen not in ckpt_by_gen or gen not in hist_by_gen:
            p_g = load_checkpoint(p0, os.path.join(CKPT_DIR, ckpt_file))
            pl_v, crit_pct, w_rb, d_rb, l_rb, wr_rb_r = evaluate_checkpoint_tuple(
                p_g, c_eval_32, c_eval_64, va_x_nhwc, va_pi, va_z, va_best, va_crit, eval_openings, rb_sims=32
            )
            elo_val = 1000 if gen == 0 else int(ckpt_by_gen.get(gen, {}).get("elo", 1000))
            ckpt_entry = {
                "id": f"ckpt-{gen}",
                "gen": gen,
                "file": ckpt_file,
                "totalGames": gen * n_parallel,
                "elo": elo_val,
                "crit": crit_pct,
                "vsRbRecord": f"{w_rb}/{d_rb}/{l_rb}",
                "vsRbWr": wr_rb_r,
            }
            if has_replay:
                ckpt_entry["replayId"] = replay_id
            ckpt_by_gen[gen] = ckpt_entry
            hist_by_gen[gen] = {
                "gen": gen,
                "totalGames": gen * n_parallel,
                "elo": elo_val,
                "crit": crit_pct,
                "vsRbWr": wr_rb_r,
                "pLoss": round(pl_v, 3),
                "spSec": 0.0,
            }
        else:
            ckpt_by_gen[gen]["totalGames"] = gen * n_parallel
            hist_by_gen[gen]["totalGames"] = gen * n_parallel
            if has_replay and not ckpt_by_gen[gen].get("replayId"):
                ckpt_by_gen[gen]["replayId"] = replay_id
            if not has_replay and "replayId" in ckpt_by_gen[gen]:
                ckpt_by_gen[gen].pop("replayId", None)

    sorted_gens = sorted(ckpt_by_gen.keys())
    history_points = [hist_by_gen[g] for g in sorted_gens if g in hist_by_gen]
    ckpt_rows = [ckpt_by_gen[g] for g in sorted_gens]

    az_final_games = sorted(
        (gm for gid, gm in games_by_id.items() if gid.startswith("az-mcts128-rb")),
        key=lambda g: str(g.get("id", "")),
    )
    league_games = sorted(
        (gm for gid, gm in games_by_id.items() if gid.startswith("league-gen-")),
        key=lambda g: int(g.get("generation", 0)),
    )[-36:]
    other_games = [
        gm for gid, gm in games_by_id.items()
        if not gid.startswith("az-mcts128-rb") and not gid.startswith("league-gen-")
    ]

    rep_doc = {
        "rules": "renju",
        "modelInfo": {
            "name": "D4OverlapPattern + Rust MCTS",
            "channels": 64,
            "blocks": 3,
            "params": int(n_params),
        },
        "metrics": rep_docs[-1].get("metrics", []) if rep_docs else [],
        "games": other_games + league_games + az_final_games,
    }
    write_json_atomic(REPLAYS_JSON_PATH, rep_doc)

    # Clean up legacy public JSONs once merged safely into arch-bench/web_data
    for stale in (LEGACY_LIVE_JSON, LEGACY_REPLAYS_JSON):
        if os.path.exists(stale):
            try:
                os.remove(stale)
            except Exception:
                pass

    max_gen = sorted_gens[-1] if sorted_gens else 0
    return max_gen, history_points, ckpt_rows, latest_live_game, rep_doc


def pick_pfsp_opponent(
    league_pool: List[Tuple[int, object]],
    league_stats: Dict[int, List[float]],
    fallback_gen: int,
    fallback_params,
    rng: random.Random,
) -> Tuple[int, object]:
    """Prioritized Fictitious Self-Play: favor opponents against which current agent has lower win rate."""
    if not league_pool:
        return fallback_gen, fallback_params
    weights: List[float] = []
    for g_num, _ in league_pool:
        st = league_stats.get(g_num)
        if not st or st[1] <= 0:
            weights.append(0.65)
        else:
            wr = st[0] / st[1]
            weights.append(max(0.15, 1.0 - wr + 0.15))
    return rng.choices(league_pool, weights=weights, k=1)[0]


def run_league_selfplay_gen(
    gen: int,
    n_games: int,
    params_cur,
    params_champ,
    champ_gen: int,
    league_pool: List[Tuple[int, object]],
    league_stats: Dict[int, List[float]],
    pool_cur: RustMctsPool,
    pool_opp: RustMctsPool,
    c_eval_32,
    fast_sims: int,
    deep_sims: int,
    deep_prob: float,
    vloss_leaves: int,
    contempt: float,
    rng: random.Random,
    live_callback,
    max_plies: int = 90,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float, float, float]], List[dict], Tuple[int, int, int, float, int]]:
    """~50% Champion + ~47% PFSP Past Checkpoints + ~3% (1 Twin Pair) Rust Tactical Teacher."""
    past_gen, params_past = pick_pfsp_opponent(
        league_pool, league_stats, champ_gen, params_champ, rng
    )

    n_teacher = 2 if n_games >= 16 else 0
    n_champ = int(round((n_games - n_teacher) * 0.52))
    if n_champ % 2 != 0:
        n_champ = max(0, n_champ - 1)
    n_past = max(0, n_games - n_teacher - n_champ)

    games = []
    cached_pair_op: Tuple[str, List[int]] = ("", [])
    cached_pair_opp: Tuple[str, int] = ("champ", champ_gen)
    for g_idx in range(n_games):
        # Twin Games: pair (0,1), (2,3), ... share the exact same opening & opponent with swapped Black/White
        if g_idx % 2 == 0:
            cached_pair_op = pick_diverse_opening(rng)
            if g_idx < n_champ:
                cached_pair_opp = ("champ", champ_gen)
            elif g_idx < n_champ + n_past:
                cached_pair_opp = ("past", past_gen)
            else:
                cached_pair_opp = ("teacher", -1)

        op_name, op_seq = cached_pair_op
        opp_kind, opp_label_gen = cached_pair_opp

        cur_side = BLACK if (g_idx % 2 == 0) else WHITE
        pool_cur.reset_game(g_idx, BLACK)
        if opp_kind != "teacher":
            pool_opp.reset_game(g_idx, BLACK)

        games.append({
            "gid": g_idx,
            "board": RenjuBoard(),
            "player": BLACK,
            "step": 0,
            "op_name": op_name,
            "op_seq": list(op_seq),
            "opp_kind": opp_kind,
            "opp_gen": opp_label_gen,
            "cur_side": cur_side,
            "traj": [],      # (nhwc, pi, mover, q_root, is_teacher, step_idx)
            "replay": [],
            "live_moves": [],
            "last_top": [],
            "last_eval_b": 0.0,
            "done": False,
            "winner": 0,
        })

    buf_x = np.zeros((pool_cur.eval_batch_size, NUM_CELLS, 6), dtype=np.float32)
    wave_idx = 0

    while True:
        active = [g for g in games if not g["done"]]
        if not active:
            break
        wave_idx += 1

        # Partition active games into opening / cur_nn / champ_opp / past_opp / teacher
        cur_gids: List[int] = []
        champ_gids: List[int] = []
        past_gids: List[int] = []
        opening_slots: List[int] = []
        steps_map: Dict[int, int] = {}

        for g in active:
            gid = g["gid"]
            steps_map[gid] = g["step"]
            if g["step"] < len(g["op_seq"]):
                opening_slots.append(gid)
                continue
            if g["opp_kind"] == "teacher" and g["player"] != g["cur_side"]:
                continue
            if g["player"] == g["cur_side"]:
                cur_gids.append(gid)
            elif g["opp_kind"] == "champ":
                champ_gids.append(gid)
            else:
                past_gids.append(gid)

        # Single GPU call for opening visualization on Game #0 if needed
        raw_open_logits: Dict[int, np.ndarray] = {}
        raw_open_vals: Dict[int, float] = {}
        if opening_slots:
            n_op = min(pool_cur.eval_batch_size, len(opening_slots))
            for idx in range(n_op):
                og = games[opening_slots[idx]]
                encode_board_nhwc_rust(og["board"], og["player"], None, buf_x[idx])
            lg_t, val_t = c_eval_32(params_cur, buf_x)
            lg_np = np.asarray(lg_t.numpy(), dtype=np.float32)
            vl_np = np.asarray(val_t.numpy(), dtype=np.float32)
            for idx in range(n_op):
                ogid = opening_slots[idx]
                raw_open_logits[ogid] = lg_np[idx]
                raw_open_vals[ogid] = float(vl_np[idx])

        avg_step = sum(g["step"] for g in active) / max(1, len(active))
        # Adaptive playout cap: invest deep sims more often in critical midgame (steps 8..44)
        eff_deep_prob = min(0.55, deep_prob + 0.14) if 8 <= avg_step <= 44 else max(0.15, deep_prob - 0.10)
        is_deep = rng.random() < eff_deep_prob
        sims_cur = deep_sims if is_deep else fast_sims

        res_cur = (
            pool_cur.run_sims(
                cur_gids,
                params_cur,
                c_eval_32,
                n_sims=sims_cur,
                c_puct=1.5,
                add_noise=True,
                leaves_per_game=vloss_leaves,
                temp_moves=10,
                steps=steps_map,
                rng=rng,
            )
            if cur_gids
            else {}
        )
        res_champ = (
            pool_opp.run_sims(
                champ_gids,
                params_champ,
                c_eval_32,
                n_sims=fast_sims,
                c_puct=1.5,
                add_noise=True,
                leaves_per_game=vloss_leaves,
                temp_moves=10,
                steps=steps_map,
                rng=rng,
            )
            if champ_gids
            else {}
        )
        res_past = (
            pool_opp.run_sims(
                past_gids,
                params_past,
                c_eval_32,
                n_sims=fast_sims,
                c_puct=1.5,
                add_noise=True,
                leaves_per_game=vloss_leaves,
                temp_moves=10,
                steps=steps_map,
                rng=rng,
            )
            if past_gids
            else {}
        )

        for g in active:
            gid = g["gid"]
            b: RenjuBoard = g["board"]
            pl = g["player"]
            step = g["step"]

            if step < len(g["op_seq"]):
                mv = g["op_seq"][step]
                mask = exact_legal_mask_rust(b, pl)
                if gid in raw_open_logits:
                    nn_probs = masked_softmax_logits(raw_open_logits[gid], mask)
                    top_p = format_top_policies(nn_probs)
                    mv_val = raw_open_vals[gid]
                else:
                    top_p = []
                    mv_val = 0.0
                r, c = divmod(mv, BOARD_SIZE)
                g["replay"].append({
                    "row": r,
                    "col": c,
                    "player": pl,
                    "moverType": "opening",
                    "value": round(mv_val, 3),
                    "topPolicies": top_p,
                })
                g["last_top"] = top_p
                g["last_eval_b"] = mv_val if pl == BLACK else -mv_val
            elif g["opp_kind"] == "teacher" and pl != g["cur_side"]:
                mv = rule_based_move_rust(b, pl, noise_eps=0.08, seed=rng.randrange(1, 2**63))
                if mv is None or not b.is_legal_move(mv, pl):
                    g["done"] = True
                    g["winner"] = opponent_of(pl)
                    continue
                pi_teacher = np.zeros((NUM_CELLS,), dtype=np.float32)
                pi_teacher[mv] = 1.0
                s_nhwc = encode_board_nhwc_rust(b, pl)
                g["traj"].append((s_nhwc, pi_teacher, pl, 0.0, True, step))
                r, c = divmod(mv, BOARD_SIZE)
                top_p = [{"row": r, "col": c, "prob": 1.0}]
                g["replay"].append({
                    "row": r,
                    "col": c,
                    "player": pl,
                    "moverType": "rule",
                    "value": 0.0,
                    "topPolicies": top_p,
                })
                g["last_top"] = top_p
            else:
                if pl == g["cur_side"]:
                    mv, pi, q_root, is_forced_win = res_cur[gid]
                elif g["opp_kind"] == "champ":
                    mv, pi, q_root, is_forced_win = res_champ[gid]
                else:
                    mv, pi, q_root, is_forced_win = res_past[gid]

                if mv < 0 or not b.is_legal_move(mv, pl):
                    g["done"] = True
                    g["winner"] = opponent_of(pl)
                    continue

                if pl == g["cur_side"] and (is_deep or is_forced_win):
                    s_nhwc = encode_board_nhwc_rust(b, pl)
                    g["traj"].append((s_nhwc, pi, pl, float(q_root), False, step))

                r, c = divmod(mv, BOARD_SIZE)
                top_p = format_top_policies(pi)
                g["replay"].append({
                    "row": r,
                    "col": c,
                    "player": pl,
                    "moverType": "nn",
                    "value": round(float(q_root), 3),
                    "topPolicies": top_p,
                })
                g["last_top"] = top_p
                g["last_eval_b"] = float(q_root) if pl == BLACK else -float(q_root)

            r, c = divmod(mv, BOARD_SIZE)
            g["live_moves"].append({"row": r, "col": c, "player": pl})
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
                pool_cur.place_and_advance(gid, mv, pl, nxt)
                if g["opp_kind"] != "teacher":
                    pool_opp.place_and_advance(gid, mv, pl, nxt)

        if wave_idx % 2 == 0 or games[0]["done"]:
            live_g = games[0] if not games[0]["done"] else active[0]
            if live_g["opp_kind"] == "teacher":
                opp_lbl = "전술 교사 (Rust)"
                m_str = f"교사 스파링 (Gen {gen})"
            elif live_g["opp_kind"] == "champ":
                opp_lbl = f"Champion (Gen {live_g['opp_gen']})"
                m_str = f"Champion 자가대국 ({live_g['opp_gen']} vs {gen})"
            else:
                opp_lbl = f"Gen {live_g['opp_gen']} (과거)"
                m_str = f"League 자가대국 ({live_g['opp_gen']} vs {gen})"

            b_name = f"Gen {gen}" if live_g["cur_side"] == BLACK else opp_lbl
            w_name = f"Gen {gen}" if live_g["cur_side"] == WHITE else opp_lbl
            live_callback({
                "matchup": m_str,
                "opening": live_g["op_name"],
                "blackPlayer": b_name,
                "whitePlayer": w_name,
                "step": live_g["step"],
                "evalBlack": round(float(live_g["last_eval_b"]), 3),
                "moves": live_g["live_moves"],
                "topPolicies": live_g["last_top"],
            })

    samples: List[Tuple[np.ndarray, np.ndarray, float, float, float]] = []
    recorded_games: List[dict] = []
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

        if g["opp_kind"] == "past" and g["opp_gen"] >= 0:
            st = league_stats.setdefault(g["opp_gen"], [0.0, 0.0])
            st[1] += 1.0
            if w == g["cur_side"]:
                st[0] += 1.0
            elif w == 0:
                st[0] += 0.5

        for s_nhwc, pi, mover, q_root, is_teacher_mv, mv_step in g["traj"]:
            if is_teacher_mv and w != mover:
                # Only learn winning teacher tactical sequences to avoid diluting MCTS priors on losses
                continue
            plies_left = max(0, g["step"] - mv_step)
            raw_z = contempt if w == 0 else (1.0 if w == mover else -1.0)
            z_out = float(raw_z * (0.992 ** plies_left)) if w != 0 else float(contempt)
            z_mix = float(np.clip((1.0 - Q_MIX) * z_out + Q_MIX * q_root, -1.0, 1.0)) if not is_teacher_mv else z_out
            surp = float(abs(z_mix - q_root)) if not is_teacher_mv else 0.35
            t_norm = float(min(1.0, max(0.0, plies_left / 60.0)))
            samples.append((s_nhwc, pi, z_mix, surp, t_norm))
        recorded_games.append({
            "winner": w,
            "moves": g["replay"],
            "generation": gen,
            "opening": g["op_name"],
            "oppKind": g["opp_kind"],
            "oppGen": g["opp_gen"],
            "curSide": g["cur_side"],
        })
    return samples, recorded_games, (bw, ww, dr, tot_m / float(max(1, n_games)), past_gen)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens", type=int, default=4)
    parser.add_argument("--parallel", type=int, default=64)
    parser.add_argument("--games-per-gen", type=int, default=192)
    parser.add_argument("--replay-ratio", type=float, default=2.5)
    parser.add_argument("--fast-sims", type=int, default=24)
    parser.add_argument("--deep-sims", type=int, default=120)
    parser.add_argument("--deep-prob", type=float, default=0.35)
    parser.add_argument("--vloss-leaves", type=int, default=2)
    parser.add_argument("--contempt", type=float, default=-0.10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.reset or not args.resume:
        args.resume = False
        wipe_training_state()
        print("Cleared all previous checkpoints, buffers, replays, and live snapshots for fresh Gen 0 start.")

    n_parallel = max(4, min(128, args.parallel))
    n_waves = max(1, int(round(max(n_parallel, args.games_per_gen) / float(n_parallel))))
    games_per_gen = n_waves * n_parallel
    print(
        f"=== Rust MCTS + Metal GPU AlphaZero League | parallel={n_parallel} games/gen={games_per_gen} ({n_waves}w) "
        f"replay_ratio={args.replay_ratio:.1f} fast={args.fast_sims} "
        f"deep={args.deep_sims} ({args.deep_prob*100:.0f}%) vloss={args.vloss_leaves} contempt={args.contempt} ==="
    )

    rng = random.Random(int(time.time()) & 0xFFFFFFFF)
    np.random.seed(rng.randrange(1, 10_000_000))

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
    t_s = np.ascontiguousarray(to_nhwc_flat(tr_x), dtype=np.float32)
    t_pi = np.ascontiguousarray(tr_pi, dtype=np.float32)
    t_z = np.ascontiguousarray(tr_z, dtype=np.float32)

    arena_rng = random.Random(999)
    eval_openings: List[Tuple[RenjuBoard, int]] = []
    while len(eval_openings) < 20:  # 40 champion games: 24 was too noisy to promote reliably
        seeded = generate_seeded_board(arena_rng, min_stones=4, max_stones=10)
        if seeded is not None:
            eval_openings.append(seeded)

    model = D4OverlapPatternV2Net(d_ray=40, d_outer=16, width=128, blocks=6)
    p0 = model.init(rg_random.key(42))
    n_params = count_params(p0)

    B = 64
    dummy_x32 = np.zeros((32, NUM_CELLS, 6), dtype=np.float32); dummy_x32[:, :, 4] = 1.0
    dummy_x64 = np.zeros((B, NUM_CELLS, 6), dtype=np.float32); dummy_x64[:, :, 4] = 1.0
    dummy_pi64 = np.full((B, NUM_CELLS), 1.0 / NUM_CELLS, dtype=np.float32)
    dummy_z64 = np.zeros((B,), dtype=np.float32)
    dummy_t64 = np.full((B,), 0.5, dtype=np.float32)
    dummy_tw64 = np.ones((B,), dtype=np.float32)

    existing_pre = [f for f in os.listdir(CKPT_DIR) if f.startswith("gen_") and f.endswith(".npz")] if os.path.exists(CKPT_DIR) else []
    is_late_resume = args.resume and len(existing_pre) >= 30
    peak_lr = RL_LR if is_late_resume else 4.5e-4
    min_lr = RL_LR if is_late_resume else 1.2e-4
    # Late RL: warmup then a constant rate (min_lr == lr). The step count lives in the saved
    # optimizer state, so a resume continues instead of re-warming from scratch.
    opt = optim.adamw(
        optim.cosine_decay(peak_lr, total_steps=4000, warmup_steps=30 if is_late_resume else 25, min_lr=min_lr),
        weight_decay=1e-4,
        max_grad_norm=0.5,
    )
    st_rl = opt.init(p0)

    def train_step(state, batch):
        pr_in, st_in = state
        xb, pib, zb, tb, twb = batch

        def loss_fn(pr):
            logits, val, turns = model(pr, xb)
            legal = xb[:, :, 4]
            pib_safe = pib * legal
            pib_norm = pib_safe / (pib_safe.sum(axis=-1, keepdims=True) + 1e-8)
            masked_logits = logits * legal + (1.0 - legal) * (-1e4)
            lp = nn.log_softmax(masked_logits, axis=-1)
            pl = -(pib_norm * lp).sum(axis=-1).mean()
            vl = ((val - zb) ** 2).mean()
            tl = (((turns - tb) ** 2) * twb).sum() / (twb.sum() + 1e-6)
            return pl + 0.6 * vl + 0.25 * tl, (pl, vl)

        (_, (pl, vl)), g = rg.value_and_grad(loss_fn, has_aux=True)(pr_in)
        pr_out, st_out = opt.update(g, st_in, pr_in)
        return (pr_out, st_out), {"p": pl, "v": vl}

    def infer_fn(pr, xb):
        lg, vl, _ = model(pr, xb)
        return lg, vl

    c_train_64 = rg.compile(
        train_step,
        ((p0, st_rl), (dummy_x64, dummy_pi64, dummy_z64, dummy_t64, dummy_tw64)),
        device="metal",
        validate="basic",
    )
    c_eval_32 = rg.compile(infer_fn, (p0, dummy_x32), device="metal", validate="basic")
    c_eval_64 = rg.compile(infer_fn, (p0, dummy_x64), device="metal", validate="basic")

    os.makedirs(CKPT_DIR, exist_ok=True)
    base_ckpt_path = os.path.join(CKPT_DIR, "gen_000.npz")

    existing_ckpts = sorted(f for f in os.listdir(CKPT_DIR) if f.startswith("gen_") and f.endswith(".npz"))
    if args.resume and existing_ckpts:
        last_ckpt = existing_ckpts[-1]
        p_rl = load_checkpoint(p0, os.path.join(CKPT_DIR, last_ckpt))
        print(f"Resumed weights from {last_ckpt}")
    elif args.resume and os.path.exists(SEED_CKPT_PATH):
        p_rl = load_checkpoint(p0, SEED_CKPT_PATH)
        save_checkpoint(p_rl, base_ckpt_path)
        print(f"Seeded baseline gen_000 from {SEED_CKPT_PATH}")
    else:
        print("Running initial 200-step Metal bootstrap for baseline gen_000...")
        boot_batches = []
        boot_t = np.full((B,), 0.5, dtype=np.float32)
        boot_tw = np.zeros((B,), dtype=np.float32)
        for _ in range(200):
            bi = np.asarray([rng.randrange(n_train) for _ in range(B)], dtype=np.int32)
            bs = np.asarray([rng.randrange(8) for _ in range(B)], dtype=np.uint8)
            x0, pi0, z0 = sample_d4_batch_rust(t_s, t_pi, t_z, bi, bs)
            boot_batches.append((x0, pi0, z0, boot_t, boot_tw))
        res_b = rg.run(c_train_64, (p0, st_rl), boot_batches, steps=200)
        (p_rl, st_rl) = res_b.state
        save_checkpoint(p_rl, base_ckpt_path)

    p_base = load_checkpoint(p0, base_ckpt_path) if os.path.exists(base_ckpt_path) else p_rl

    start_gen, history_points, ckpt_rows, latest_live_game, rep_doc = reconcile_history_and_replays(
        n_parallel=n_parallel,
        n_params=n_params,
        p0=p0,
        p_base=p_base,
        c_eval_32=c_eval_32,
        c_eval_64=c_eval_64,
        va_x_nhwc=va_x_nhwc,
        va_pi=va_pi,
        va_z=va_z,
        va_best=va_best,
        va_crit=va_crit,
        eval_openings=eval_openings,
        fresh=not args.resume,
    )
    total_games = int(ckpt_rows[-1].get("totalGames", 0)) if ckpt_rows else 0
    print(f"Reconciled history: gens 0..{start_gen} ({len(ckpt_rows)} checkpoints, total_games={total_games})")

    # Load champion checkpoint by best composite score = 0.55 * vsRbWr + 0.45 * Crit
    champ_gen = start_gen
    best_composite = -1e9
    for c in ckpt_rows:
        comp = 0.55 * float(c.get("vsRbWr", 0.0)) + 0.45 * float(c.get("crit", 0.0))
        c_file = os.path.join(CKPT_DIR, str(c.get("file", f"gen_{int(c['gen']):03d}.npz")))
        if comp > best_composite and os.path.exists(c_file):
            best_composite = comp
            champ_gen = int(c["gen"])
    # The composite above comes from rule-based metrics the net has outgrown; the champion that
    # won its head-to-heads is in the manifest, so prefer it.
    man = read_json_safe(MANIFEST_PATH) if args.resume else None
    if man and os.path.exists(os.path.join(CKPT_DIR, f"gen_{int(man.get('championGen', -1)):03d}.npz")):
        champ_gen = int(man["championGen"])
    champ_path = os.path.join(CKPT_DIR, f"gen_{champ_gen:03d}.npz")
    p_champion = load_checkpoint(p0, champ_path) if os.path.exists(champ_path) else p_rl
    if best_composite < 0:
        best_composite = 70.0
    print(f"Champion initialized: Gen {champ_gen} (composite={best_composite:.1f})")
    champ_set_gen = start_gen  # gens since the last promotion count from here
    if args.resume and os.path.exists(champ_path) and champ_gen != start_gen:
        # Later gens all lost to the champion: train on from its weights, with fresh optimizer state.
        p_rl = load_checkpoint(p0, champ_path)
        st_rl = opt.init(p_rl)
        print(f"Training resumes from champion Gen {champ_gen} weights (not the latest gen)")
    elif args.resume:
        st_saved = load_tree(st_rl, OPT_STATE_PATH)
        if st_saved is not None:
            st_rl = st_saved
            print("Restored optimizer state (Adam moments + step count)")

    # Populate league pool from recent saved checkpoints
    existing_ckpts = sorted(f for f in os.listdir(CKPT_DIR) if f.startswith("gen_") and f.endswith(".npz"))
    league_pool: List[Tuple[int, object]] = []
    league_stats: Dict[int, List[float]] = {}
    for fname in existing_ckpts[-8:]:
        g_num = int(fname.replace("gen_", "").replace(".npz", ""))
        league_pool.append((g_num, load_checkpoint(p0, os.path.join(CKPT_DIR, fname))))
    if not league_pool:
        league_pool.append((0, p_base))

    sp_buf = SelfPlayBuffer(max_size=MAX_SP_BUFFER)
    if args.resume:
        loaded_n = sp_buf.load(BUFFER_PATH)
        if loaded_n > 0:
            print(f"Loaded persistent RL replay buffer: {loaded_n} MCTS positions")

    matches_db = {} if not args.resume else load_eval_matches()

    def apply_bt_elos() -> Dict[str, int]:
        bt = solve_bradley_terry_elo(matches_db)
        for row in ckpt_rows:
            k = f"gen_{int(row['gen']):03d}"
            if k in bt:
                row["elo"] = bt[k]
        for hp in history_points:
            k = f"gen_{int(hp['gen']):03d}"
            if k in bt:
                hp["elo"] = bt[k]
        return bt

    if args.resume and not matches_db and start_gen > 0:
        # Initial Bradley-Terry calibration bracket so Elo ladder is grounded from Gen 0
        calib_gens = sorted({0, max(0, start_gen // 2), champ_gen, start_gen})
        calib_params = {}
        for cg in calib_gens:
            cf = os.path.join(CKPT_DIR, f"gen_{cg:03d}.npz")
            if os.path.exists(cf):
                calib_params[cg] = load_checkpoint(p0, cf)
        for cg, cp in calib_params.items():
            ck = f"gen_{cg:03d}"
            w_r, d_r, l_r, _ = arena_mcts_vs_rb_rust(eval_openings[:6], cp, c_eval_32, n_sims=24)
            record_eval_match(matches_db, ck, "rb_tactical", w_r, d_r, l_r)
        c_list = sorted(calib_params.keys())
        for idx_i in range(len(c_list) - 1):
            ga, gb = c_list[idx_i], c_list[idx_i + 1]
            wa, da, la, _ = arena_nn_vs_nn_rust(
                eval_openings[:6], calib_params[ga], calib_params[gb], c_eval_32, n_sims=16
            )
            record_eval_match(matches_db, f"gen_{ga:03d}", f"gen_{gb:03d}", wa, da, la)
        save_eval_matches(matches_db)
        bt0 = apply_bt_elos()
        print(
            f"Calibrated Bradley-Terry Elo ladder: "
            + ", ".join(f"Gen{g}={bt0.get(f'gen_{g:03d}', 1000)}" for g in c_list)
            + f", TacticalRB={bt0.get('rb_tactical', 1000)}"
        )
    elif matches_db:
        apply_bt_elos()

    pool_cur = RustMctsPool(max_games=128, eval_batch_size=64)
    pool_opp = RustMctsPool(max_games=128, eval_batch_size=64)

    target_gens = start_gen + args.gens
    t_session_start = time.time()
    session_games = 0
    write_checkpoint_manifest(ckpt_rows, champ_gen, n_params)

    def publish_live(status: str, live_game: dict, cur_gen: int) -> None:
        nonlocal latest_live_game
        latest_live_game = live_game
        cur_elo = ckpt_rows[-1]["elo"] if ckpt_rows else 1000
        best_elo = max((c["elo"] for c in ckpt_rows), default=1000)
        best_crit = max((c["crit"] for c in ckpt_rows), default=0.0)
        cur_rb = ckpt_rows[-1]["vsRbWr"] if ckpt_rows else 0.0
        elapsed = max(0.1, time.time() - t_session_start)
        payload = {
            "status": status,
            "updatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "modelName": "D4OverlapPatternV2 (3-Head) + Adaptive MCTS",
            "params": int(n_params),
            "device": "Rust MCTS + Metal GPU",
            "config": {
                "parallel": n_parallel,
                "fastSims": args.fast_sims,
                "deepSims": args.deep_sims,
                "deepProb": args.deep_prob,
                "contempt": args.contempt,
                "openingMix": "랜덤 클러스터 50% · 12주형 30% · 천원 20% (Twin)",
                "leagueRatio": "챔피언 50% · PFSP 과거 풀 45% · 교사 5%",
            },
            "generation": cur_gen,
            "targetGens": target_gens,
            "totalGames": total_games,
            "gamesPerSec": round(session_games / elapsed, 2),
            "bufferSize": len(sp_buf) + n_train,
            "currentElo": cur_elo,
            "bestElo": best_elo,
            "bestCrit": round(best_crit, 1),
            "vsFullRbWinRate": cur_rb,
            "liveGame": live_game,
            "history": history_points,
            "checkpoints": ckpt_rows,
        }
        write_json_atomic(LIVE_JSON_PATH, payload)

    publish_live("running", latest_live_game, start_gen)

    for offset in range(1, args.gens + 1):
        gen = start_gen + offset
        t_gen = time.time()

        samples: List[Tuple[np.ndarray, np.ndarray, float, float, float]] = []
        rec_games: List[dict] = []
        bw = ww = dr = 0
        tot_m_sum = 0.0
        past_g = champ_gen
        for _wave in range(n_waves):
            w_samples, w_rec, (w_bw, w_ww, w_dr, w_avg_m, past_g) = run_league_selfplay_gen(
                gen=gen,
                n_games=n_parallel,
                params_cur=p_rl,
                params_champ=p_champion,
                champ_gen=champ_gen,
                league_pool=league_pool,
                league_stats=league_stats,
                pool_cur=pool_cur,
                pool_opp=pool_opp,
                c_eval_32=c_eval_64,
                fast_sims=args.fast_sims,
                deep_sims=args.deep_sims,
                deep_prob=args.deep_prob,
                vloss_leaves=args.vloss_leaves,
                contempt=args.contempt,
                rng=rng,
                live_callback=lambda lg: publish_live("running", lg, gen),
                max_plies=90,
            )
            samples.extend(w_samples)
            rec_games.extend(w_rec)
            bw += w_bw
            ww += w_ww
            dr += w_dr
            tot_m_sum += w_avg_m * n_parallel
        avg_m = tot_m_sum / float(max(1, games_per_gen))
        sp_sec = time.time() - t_gen
        total_games += games_per_gen
        session_games += games_per_gen
        sp_buf.extend(samples)
        sp_buf.save(BUFFER_PATH)

        t_tr = time.time()
        n_train_steps = max(80, min(400, int(round(len(samples) * args.replay_ratio / float(B)))))
        train_batches = [
            sample_anchored_batch(t_s, t_pi, t_z, sp_buf, B, rng) for _ in range(n_train_steps)
        ]
        res = rg.run(c_train_64, (p_rl, st_rl), train_batches, steps=n_train_steps)
        (p_rl, st_rl) = res.state
        tr_sec = time.time() - t_tr

        ckpt_file = f"gen_{gen:03d}.npz"
        ckpt_path = os.path.join(CKPT_DIR, ckpt_file)
        gen_key = f"gen_{gen:03d}"
        champ_key = f"gen_{champ_gen:03d}"

        # True AlphaZero evaluation: validation Crit% + Tactical Rust Teacher (12g) + Champion H2H (24g) + Ladder Pool
        pl_v, crit_pct, w_rb, d_rb, l_rb, wr_rb_r = evaluate_checkpoint_tuple(
            p_rl, c_eval_32, c_eval_64, va_x_nhwc, va_pi, va_z, va_best, va_crit, eval_openings[:6], rb_sims=24
        )
        wc, dc, lc, wr_vs_champ = arena_nn_vs_nn_rust(eval_openings, p_rl, p_champion, c_eval_32, n_sims=48)

        guard_tag = ""
        if wr_vs_champ >= 55.0:
            p_champion = p_rl
            champ_gen = gen
            champ_set_gen = gen
            guard_tag = f" [Champ+ vsC={wr_vs_champ:.0f}%]"
        elif wr_vs_champ < 30.0 and crit_pct < 75.0:
            # Regression guard: only blend on severe collapse in both H2H and tactical Crit
            p_rl = rg.tree.map(lambda a, b: 0.70 * a + 0.30 * b, p_rl, p_champion)
            pl_v, crit_pct, w_rb, d_rb, l_rb, wr_rb_r = evaluate_checkpoint_tuple(
                p_rl, c_eval_32, c_eval_64, va_x_nhwc, va_pi, va_z, va_best, va_crit, eval_openings[:6], rb_sims=24
            )
            wc, dc, lc, wr_vs_champ = arena_nn_vs_nn_rust(eval_openings, p_rl, p_champion, c_eval_32, n_sims=48)
            guard_tag = f" [Guard 70/30 vsC={wr_vs_champ:.0f}%]"

        record_eval_match(matches_db, gen_key, "rb_tactical", w_rb, d_rb, l_rb)
        record_eval_match(matches_db, gen_key, champ_key, wc, dc, lc)

        # Ladder anchor match against a rotating past milestone checkpoint to keep BT graph connected
        ladder_candidates = [0, max(0, champ_gen // 2), max(0, gen - 10), max(0, gen - 5)]
        opp_g = ladder_candidates[gen % len(ladder_candidates)]
        if opp_g != gen and opp_g != champ_gen:
            opp_path = os.path.join(CKPT_DIR, f"gen_{opp_g:03d}.npz")
            if os.path.exists(opp_path):
                p_ladder = load_checkpoint(p0, opp_path)
                wl, dl, ll, _ = arena_nn_vs_nn_rust(eval_openings[:6], p_rl, p_ladder, c_eval_32, n_sims=16)
                record_eval_match(matches_db, gen_key, f"gen_{opp_g:03d}", wl, dl, ll)

        save_eval_matches(matches_db)
        bt_elos = apply_bt_elos()
        elo_val = bt_elos.get(gen_key, 1000)

        save_checkpoint(p_rl, ckpt_path)
        league_pool.append((gen, p_rl))
        if gen - champ_set_gen >= STALL_GENS:
            # Stuck below the champion: restart from its weights; fresh Adam state re-warms the rate.
            p_rl = p_champion
            st_rl = opt.init(p_rl)
            champ_set_gen = gen
            guard_tag += f" [Reset->Champ{champ_gen}]"
        save_tree(st_rl, OPT_STATE_PATH)
        if len(league_pool) > 8:
            league_pool.pop(0)

        # Save highest learning-gain decisive game (max value swing + strong-opponent win bonus)
        replay_id = f"league-gen-{gen}"
        decisive = [g for g in rec_games if g["winner"] != 0 and len(g["moves"]) >= 16]
        pool_rep = decisive if decisive else [g for g in rec_games if g["winner"] != 0] or rec_games

        def _game_gain_score(g: dict) -> float:
            mvs = g["moves"]
            swing = 0.0
            for k in range(1, len(mvs)):
                swing += abs(float(mvs[k].get("value", 0.0)) - float(mvs[k - 1].get("value", 0.0)))
            cur_won = g["winner"] == g["curSide"]
            opp_bonus = 2.0 if (cur_won and g["oppKind"] == "champ") else (1.0 if cur_won else 0.0)
            return swing + opp_bonus

        pick = max(pool_rep, key=_game_gain_score)
        outcome = "흑승" if pick["winner"] == BLACK else "백승" if pick["winner"] == WHITE else "무승부"
        if pick["oppKind"] == "teacher":
            opp_name = "전술 교사 (Rust)"
            matchup_str = "교사 스파링"
        elif pick["oppKind"] == "champ":
            opp_name = f"Champion (Gen {pick['oppGen']})"
            matchup_str = "Champion 리그"
        else:
            opp_name = f"Gen {pick['oppGen']} (과거)"
            matchup_str = "과거 체크포인트 리그"
        b_nm = f"D4OverlapPattern + MCTS (Gen {gen})" if pick["curSide"] == BLACK else opp_name
        w_nm = f"D4OverlapPattern + MCTS (Gen {gen})" if pick["curSide"] == WHITE else opp_name

        latest_rep = read_json_safe(REPLAYS_JSON_PATH) or rep_doc
        az_final_games = [g for g in latest_rep.get("games", []) if str(g.get("id", "")).startswith("az-mcts128-rb")]
        other_games = [
            g for g in latest_rep.get("games", [])
            if not str(g.get("id", "")).startswith("az-mcts128-rb") and g.get("id") != replay_id
        ]
        other_games.append({
            "id": replay_id,
            "label": f"League Gen {gen} ({pick['opening']}) · {len(pick['moves'])}수 {outcome}",
            "generation": gen,
            "winner": pick["winner"],
            "blackPlayer": b_nm,
            "whitePlayer": w_nm,
            "matchup": matchup_str,
            "moves": pick["moves"],
        })
        latest_rep["games"] = other_games[-36:] + az_final_games
        write_json_atomic(REPLAYS_JSON_PATH, latest_rep)

        history_points.append({
            "gen": gen,
            "totalGames": total_games,
            "elo": elo_val,
            "crit": crit_pct,
            "vsRbWr": wr_rb_r,
            "pLoss": round(pl_v, 3),
            "spSec": round(sp_sec, 2),
        })
        ckpt_rows.append({
            "id": f"ckpt-{gen}",
            "gen": gen,
            "file": ckpt_file,
            "totalGames": total_games,
            "elo": elo_val,
            "crit": crit_pct,
            "vsRbRecord": f"{w_rb}/{d_rb}/{l_rb}",
            "vsRbWr": wr_rb_r,
            "replayId": replay_id,
        })
        write_checkpoint_manifest(ckpt_rows, champ_gen, n_params)

        print(
            f"Gen {gen:3d}/{target_gens} | SP {sp_sec:4.1f}s ({games_per_gen / max(0.01, sp_sec):.2f}g/s, "
            f"avg {avg_m:4.1f}m, B/W/D={bw}/{ww}/{dr}, opp=Gen{past_g}) | "
            f"Tr {tr_sec:3.1f}s ({n_train_steps}st) | Crit={crit_pct:4.1f}% | vsFullRB={w_rb}/{d_rb}/{l_rb} ({wr_rb_r:4.1f}%) | "
            f"Elo={elo_val}{guard_tag} -> {ckpt_file}"
        )
        publish_live("running", latest_live_game, gen)

    publish_live("idle", latest_live_game, target_gens)
    print(f"\nCompleted {args.gens} generations! Live snapshot saved to {LIVE_JSON_PATH}")


if __name__ == "__main__":
    main()
