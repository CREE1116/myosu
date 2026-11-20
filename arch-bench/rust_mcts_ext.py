"""ctypes wrapper for native Rust batched PUCT MCTS + Renju C-ABI library (libgomoku_mcts.dylib)."""

from __future__ import annotations

import ctypes
from ctypes import POINTER, byref, c_float, c_int32, c_uint8, c_uint64, c_void_p
import os
import random
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))  # arch-bench/
sys.path.insert(0, _HERE)
from renju_env import BLACK, BOARD_SIZE, EMPTY, NUM_CELLS, WHITE, RenjuBoard, opponent_of

LIB_PATH = os.path.join(_HERE, "..", "engine-rs", "target", "release", "libgomoku_mcts.dylib")
if not os.path.exists(LIB_PATH):
    raise RuntimeError(
        f"Native Rust library not found at {LIB_PATH}. Run `cd engine-rs && cargo build --release --lib` first."
    )

_lib = ctypes.CDLL(LIB_PATH)

_lib.rs_mcts_create.argtypes = [c_int32]
_lib.rs_mcts_create.restype = c_void_p

_lib.rs_mcts_free.argtypes = [c_void_p]
_lib.rs_mcts_free.restype = None

_lib.rs_mcts_reset_game.argtypes = [c_void_p, c_int32, c_uint8]
_lib.rs_mcts_reset_game.restype = None

_lib.rs_mcts_set_board.argtypes = [c_void_p, c_int32, POINTER(c_uint8), c_int32, c_uint8]
_lib.rs_mcts_set_board.restype = None

_lib.rs_mcts_place_and_advance.argtypes = [c_void_p, c_int32, c_int32, c_uint8, c_uint8]
_lib.rs_mcts_place_and_advance.restype = None

_lib.rs_mcts_prepare_roots.argtypes = [
    c_void_p,
    POINTER(c_int32),
    c_int32,
    POINTER(c_int32),
    POINTER(c_float),
]
_lib.rs_mcts_prepare_roots.restype = c_int32

_lib.rs_mcts_expand_roots.argtypes = [
    c_void_p,
    POINTER(c_int32),
    c_int32,
    POINTER(c_float),
    POINTER(c_float),
    c_uint8,
    c_uint64,
]
_lib.rs_mcts_expand_roots.restype = None

_lib.rs_mcts_apply_root_noise.argtypes = [c_void_p, POINTER(c_int32), c_int32, c_uint64]
_lib.rs_mcts_apply_root_noise.restype = None

_lib.rs_mcts_select_leaves.argtypes = [
    c_void_p,
    POINTER(c_int32),
    c_int32,
    c_int32,
    c_int32,
    c_float,
    POINTER(c_float),
]
_lib.rs_mcts_select_leaves.restype = c_int32

_lib.rs_mcts_expand_backprop.argtypes = [c_void_p, POINTER(c_float), POINTER(c_float)]
_lib.rs_mcts_expand_backprop.restype = None

_lib.rs_mcts_finalize_move.argtypes = [
    c_void_p,
    c_int32,
    c_uint8,
    c_uint64,
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_uint8),
]
_lib.rs_mcts_finalize_move.restype = c_int32

_lib.rs_exact_legal_mask.argtypes = [POINTER(c_uint8), c_uint8, POINTER(c_uint8)]
_lib.rs_exact_legal_mask.restype = None

_lib.rs_encode_board_nhwc.argtypes = [POINTER(c_uint8), c_int32, c_uint8, POINTER(c_float)]
_lib.rs_encode_board_nhwc.restype = None

_lib.rs_rule_based_move.argtypes = [POINTER(c_uint8), c_uint8, c_float, c_uint64]
_lib.rs_rule_based_move.restype = c_int32

_lib.rs_sample_d4_batch.argtypes = [
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    c_int32,
    POINTER(c_int32),
    POINTER(c_uint8),
    c_int32,
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
]
_lib.rs_sample_d4_batch.restype = None


def _as_cells_u8(board_or_cells) -> Tuple[np.ndarray, int]:
    if isinstance(board_or_cells, RenjuBoard):
        arr = np.asarray(board_or_cells.cells, dtype=np.uint8)
        lm = -1 if board_or_cells.last_move is None else int(board_or_cells.last_move)
        return np.ascontiguousarray(arr, dtype=np.uint8), lm
    arr = np.ascontiguousarray(board_or_cells, dtype=np.uint8)
    return arr, -1


def exact_legal_mask_rust(board_or_cells, player: int) -> np.ndarray:
    cells_u8, _ = _as_cells_u8(board_or_cells)
    out = np.empty((NUM_CELLS,), dtype=np.uint8)
    _lib.rs_exact_legal_mask(
        cells_u8.ctypes.data_as(POINTER(c_uint8)),
        c_uint8(player),
        out.ctypes.data_as(POINTER(c_uint8)),
    )
    return out.astype(np.bool_)


def encode_board_nhwc_rust(
    board_or_cells, player: int, last_move: Optional[int] = None, out_row: Optional[np.ndarray] = None
) -> np.ndarray:
    cells_u8, default_lm = _as_cells_u8(board_or_cells)
    lm = default_lm if last_move is None else int(last_move)
    if out_row is None:
        out_row = np.empty((NUM_CELLS, 6), dtype=np.float32)
    else:
        assert out_row.flags["C_CONTIGUOUS"] and out_row.dtype == np.float32
    _lib.rs_encode_board_nhwc(
        cells_u8.ctypes.data_as(POINTER(c_uint8)),
        c_int32(lm),
        c_uint8(player),
        out_row.ctypes.data_as(POINTER(c_float)),
    )
    return out_row


def rule_based_move_rust(
    board_or_cells, player: int, noise_eps: float = 0.0, seed: Optional[int] = None
) -> Optional[int]:
    cells_u8, _ = _as_cells_u8(board_or_cells)
    s = int(seed) if seed is not None else random.randrange(1, 2**63)
    mv = int(
        _lib.rs_rule_based_move(
            cells_u8.ctypes.data_as(POINTER(c_uint8)),
            c_uint8(player),
            c_float(noise_eps),
            c_uint64(s),
        )
    )
    return None if mv < 0 else mv


def sample_d4_batch_rust(
    buf_s: np.ndarray,
    buf_pi: np.ndarray,
    buf_z: np.ndarray,
    indices: np.ndarray,
    syms: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_buf = int(buf_s.shape[0])
    b = int(indices.shape[0])
    idx_c = np.ascontiguousarray(indices, dtype=np.int32)
    sym_c = np.ascontiguousarray(syms, dtype=np.uint8)
    xb = np.empty((b, NUM_CELLS, 6), dtype=np.float32)
    pib = np.empty((b, NUM_CELLS), dtype=np.float32)
    zb = np.empty((b,), dtype=np.float32)
    _lib.rs_sample_d4_batch(
        buf_s.ctypes.data_as(POINTER(c_float)),
        buf_pi.ctypes.data_as(POINTER(c_float)),
        buf_z.ctypes.data_as(POINTER(c_float)),
        c_int32(n_buf),
        idx_c.ctypes.data_as(POINTER(c_int32)),
        sym_c.ctypes.data_as(POINTER(c_uint8)),
        c_int32(b),
        xb.ctypes.data_as(POINTER(c_float)),
        pib.ctypes.data_as(POINTER(c_float)),
        zb.ctypes.data_as(POINTER(c_float)),
    )
    return xb, pib, zb


class RustMctsPool:
    """Zero-overhead native Rust MCTS arena driving Metal GPU batched evaluations."""

    def __init__(self, max_games: int = 32, eval_batch_size: int = 32):
        self.max_games = max(1, min(256, int(max_games)))
        self.eval_batch_size = max(1, int(eval_batch_size))
        self._ptr = _lib.rs_mcts_create(c_int32(self.max_games))
        self._buf_x = np.zeros((self.eval_batch_size, NUM_CELLS, 6), dtype=np.float32)
        self._gids = np.zeros((self.max_games,), dtype=np.int32)
        self._eval_ids = np.zeros((self.max_games,), dtype=np.int32)
        self._out_pi = np.zeros((NUM_CELLS,), dtype=np.float32)
        self._out_q = c_float(0.0)
        self._out_win = c_uint8(0)

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _lib.rs_mcts_free(ptr)
            self._ptr = None

    def reset_game(self, g_idx: int, player: int = BLACK) -> None:
        _lib.rs_mcts_reset_game(self._ptr, c_int32(g_idx), c_uint8(player))

    def set_board(self, g_idx: int, board_or_cells, player: int, last_move: Optional[int] = None) -> None:
        cells_u8, default_lm = _as_cells_u8(board_or_cells)
        lm = default_lm if last_move is None else int(last_move)
        _lib.rs_mcts_set_board(
            self._ptr,
            c_int32(g_idx),
            cells_u8.ctypes.data_as(POINTER(c_uint8)),
            c_int32(lm),
            c_uint8(player),
        )

    def place_and_advance(self, g_idx: int, mv: int, mover: int, next_player: int) -> None:
        _lib.rs_mcts_place_and_advance(
            self._ptr,
            c_int32(g_idx),
            c_int32(mv),
            c_uint8(mover),
            c_uint8(next_player),
        )

    def run_sims(
        self,
        game_indices: List[int],
        params,
        c_eval_32,
        n_sims: int,
        c_puct: float = 1.5,
        add_noise: bool = True,
        leaves_per_game: int = 2,
        temp_moves: int = 10,
        steps: Optional[Dict[int, int]] = None,
        rng: Optional[random.Random] = None,
    ) -> Dict[int, Tuple[int, np.ndarray, float, bool]]:
        n_active = len(game_indices)
        if n_active == 0:
            return {}

        seed_gen = rng.randrange if rng is not None else random.randrange
        self._gids[:n_active] = game_indices
        gids_ptr = self._gids.ctypes.data_as(POINTER(c_int32))
        eval_ids_ptr = self._eval_ids.ctypes.data_as(POINTER(c_int32))
        buf_x_ptr = self._buf_x.ctypes.data_as(POINTER(c_float))

        # Step 1: Prepare & expand unexpanded roots in chunks of <= eval_batch_size
        bsz = self.eval_batch_size
        off = 0
        while off < n_active:
            chunk = min(bsz, n_active - off)
            sub_gids = self._gids[off : off + chunk].ctypes.data_as(POINTER(c_int32))
            n_eval = int(_lib.rs_mcts_prepare_roots(self._ptr, sub_gids, c_int32(chunk), eval_ids_ptr, buf_x_ptr))
            if n_eval > 0:
                lg_t, val_t = c_eval_32(params, self._buf_x)
                lg_np = np.ascontiguousarray(lg_t.numpy()[:n_eval], dtype=np.float32)
                vl_np = np.ascontiguousarray(val_t.numpy()[:n_eval], dtype=np.float32)
                _lib.rs_mcts_expand_roots(
                    self._ptr,
                    eval_ids_ptr,
                    c_int32(n_eval),
                    lg_np.ctypes.data_as(POINTER(c_float)),
                    vl_np.ctypes.data_as(POINTER(c_float)),
                    c_uint8(0),
                    c_uint64(0),
                )
            off += chunk

        if add_noise:
            _lib.rs_mcts_apply_root_noise(self._ptr, gids_ptr, c_int32(n_active), c_uint64(seed_gen(1, 2**63)))

        # Step 2: Run batched virtual-loss simulation waves
        effective_lpg = max(1, min(int(leaves_per_game), max(1, bsz // n_active)))
        n_waves = max(1, (int(n_sims) + effective_lpg - 1) // effective_lpg)
        for _ in range(n_waves):
            queued = int(
                _lib.rs_mcts_select_leaves(
                    self._ptr,
                    gids_ptr,
                    c_int32(n_active),
                    c_int32(effective_lpg),
                    c_int32(bsz),
                    c_float(c_puct),
                    buf_x_ptr,
                )
            )
            if queued <= 0:
                break
            lg_t, val_t = c_eval_32(params, self._buf_x)
            lg_np = np.ascontiguousarray(lg_t.numpy()[:queued], dtype=np.float32)
            vl_np = np.ascontiguousarray(val_t.numpy()[:queued], dtype=np.float32)
            _lib.rs_mcts_expand_backprop(
                self._ptr,
                lg_np.ctypes.data_as(POINTER(c_float)),
                vl_np.ctypes.data_as(POINTER(c_float)),
            )

        # Step 3: Extract move, visit policy pi, root Q value, and forced win flag
        results: Dict[int, Tuple[int, np.ndarray, float, bool]] = {}
        out_pi_ptr = self._out_pi.ctypes.data_as(POINTER(c_float))
        for gid in game_indices:
            st = steps.get(gid, 999) if steps is not None else 999
            use_temp = 1 if (add_noise and st < temp_moves) else 0
            mv = int(
                _lib.rs_mcts_finalize_move(
                    self._ptr,
                    c_int32(gid),
                    c_uint8(use_temp),
                    c_uint64(seed_gen(1, 2**63)),
                    out_pi_ptr,
                    byref(self._out_q),
                    byref(self._out_win),
                )
            )
            results[gid] = (mv, self._out_pi.copy(), float(self._out_q.value), bool(self._out_win.value))
        return results


def arena_mcts_vs_rb_rust(
    openings: List[Tuple[RenjuBoard, int]],
    params,
    c_eval_32,
    n_sims: int = 64,
    max_plies: int = 120,
) -> Tuple[int, int, int, float]:
    """Plays 2 * len(openings) games concurrently vs Rust tactical teacher using Rust MCTS."""
    slots = []
    pool = RustMctsPool(max_games=max(4, len(openings) * 2))
    gid = 0
    for sb, sp in openings:
        for nn_side in (sp, opponent_of(sp)):
            b_copy = sb.clone()
            pool.set_board(gid, b_copy, sp)
            slots.append({
                "gid": gid,
                "board": b_copy,
                "player": sp,
                "nn_side": nn_side,
                "done": False,
                "winner": 0,
                "plies": 0,
            })
            gid += 1

    while True:
        nn_gids: List[int] = []
        nn_slots = []
        for g in slots:
            while not g["done"]:
                if g["plies"] >= max_plies:
                    g["done"] = True
                    g["winner"] = 0
                    break
                b: RenjuBoard = g["board"]
                pl = g["player"]
                if pl == g["nn_side"]:
                    nn_gids.append(g["gid"])
                    nn_slots.append(g)
                    break
                mv = rule_based_move_rust(b, pl, noise_eps=0.0)
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
                pool.place_and_advance(g["gid"], mv, pl, nxt)

        if not nn_gids:
            if all(g["done"] for g in slots):
                break
            continue

        res = pool.run_sims(nn_gids, params, c_eval_32, n_sims=n_sims, add_noise=False, leaves_per_game=2)
        for g in nn_slots:
            b = g["board"]
            pl = g["player"]
            mv, _, _, _ = res[g["gid"]]
            if mv < 0 or not b.is_legal_move(mv, pl):
                g["done"] = True
                g["winner"] = opponent_of(pl)
                continue
            b.place_move(mv, pl)
            g["plies"] += 1
            if b.is_win_at(mv, pl):
                g["done"] = True
                g["winner"] = pl
            else:
                nxt = opponent_of(pl)
                g["player"] = nxt
                pool.place_and_advance(g["gid"], mv, pl, nxt)

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


def arena_nn_vs_nn_rust(
    openings: List[Tuple[RenjuBoard, int]],
    params_a,
    params_b,
    c_eval_32,
    n_sims: int = 16,
    max_plies: int = 120,
) -> Tuple[int, int, int, float]:
    """Plays 2 * len(openings) games concurrently between params_a and params_b via Rust MCTS."""
    n_slots = len(openings) * 2
    pool_a = RustMctsPool(max_games=max(4, n_slots))
    pool_b = RustMctsPool(max_games=max(4, n_slots))
    slots = []
    gid = 0
    for sb, sp in openings:
        for side_a in (sp, opponent_of(sp)):
            b_copy = sb.clone()
            pool_a.set_board(gid, b_copy, sp)
            pool_b.set_board(gid, b_copy, sp)
            slots.append({
                "gid": gid,
                "board": b_copy,
                "player": sp,
                "side_a": side_a,
                "done": False,
                "winner": 0,
                "plies": 0,
            })
            gid += 1

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

        for batch_list, pool_cur, pr_cur in ((turn_a, pool_a, params_a), (turn_b, pool_b, params_b)):
            if not batch_list:
                continue
            gids = [g["gid"] for g in batch_list]
            res = pool_cur.run_sims(gids, pr_cur, c_eval_32, n_sims=n_sims, add_noise=False, leaves_per_game=2)
            for g in batch_list:
                b: RenjuBoard = g["board"]
                pl = g["player"]
                mv, _, _, _ = res[g["gid"]]
                if mv < 0 or not b.is_legal_move(mv, pl):
                    g["done"] = True
                    g["winner"] = opponent_of(pl)
                    continue
                b.place_move(mv, pl)
                g["plies"] += 1
                if b.is_win_at(mv, pl):
                    g["done"] = True
                    g["winner"] = pl
                else:
                    nxt = opponent_of(pl)
                    g["player"] = nxt
                    pool_a.place_and_advance(g["gid"], mv, pl, nxt)
                    pool_b.place_and_advance(g["gid"], mv, pl, nxt)

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


def _benchmark() -> None:
    import rg
    from rg import random as rg_random
    from rg_models import D4OverlapPatternNet, count_params

    rng = random.Random(20261119)
    model = D4OverlapPatternNet(d_ray=24, d_outer=8, width=64, blocks=3)
    p0 = model.init(rg_random.key(42))
    ckpt_path = os.path.join(_HERE, "d4_overlap_az.npz")
    if os.path.exists(ckpt_path):
        d = np.load(ckpt_path)
        leaves = [rg.tensor(d[f"p_{i}"]) for i in range(len(rg.tree.leaves(p0)))]
        p0 = rg.tree.unflatten(rg.tree.structure(p0), leaves)

    dummy_x32 = np.zeros((32, NUM_CELLS, 6), dtype=np.float32)
    dummy_x32[:, :, 4] = 1.0
    c_eval_32 = rg.compile(lambda pr, xb: model(pr, xb), (p0, dummy_x32), device="metal", validate="basic")
    # Warm up Metal shader compilation
    _ = c_eval_32(p0, dummy_x32)

    pool = RustMctsPool(max_games=32)
    n_games = 16
    boards = [RenjuBoard() for _ in range(n_games)]
    players = [BLACK] * n_games
    steps = [0] * n_games
    done = [False] * n_games
    winners = [0] * n_games
    for i in range(n_games):
        pool.reset_game(i, BLACK)
        mv = 7 * BOARD_SIZE + 7
        boards[i].place_move(mv, BLACK)
        pool.place_and_advance(i, mv, BLACK, WHITE)
        players[i] = WHITE
        steps[i] = 1

    t0 = time.time()
    while not all(done):
        active_ids = [i for i in range(n_games) if not done[i]]
        n_sims = 80 if rng.random() < 0.30 else 16
        res = pool.run_sims(
            active_ids,
            p0,
            c_eval_32,
            n_sims=n_sims,
            add_noise=True,
            leaves_per_game=2,
            temp_moves=10,
            steps={i: steps[i] for i in active_ids},
            rng=rng,
        )
        for i in active_ids:
            mv, _, _, _ = res[i]
            pl = players[i]
            b = boards[i]
            if mv < 0 or not b.is_legal_move(mv, pl):
                done[i] = True
                winners[i] = opponent_of(pl)
                continue
            b.place_move(mv, pl)
            steps[i] += 1
            if b.is_win_at(mv, pl):
                done[i] = True
                winners[i] = pl
            elif steps[i] >= 90:
                done[i] = True
                winners[i] = 0
            else:
                nxt = opponent_of(pl)
                players[i] = nxt
                pool.place_and_advance(i, mv, pl, nxt)

    dt = time.time() - t0
    avg_m = sum(steps) / float(n_games)
    bw = sum(1 for w in winners if w == BLACK)
    ww = sum(1 for w in winners if w == WHITE)
    dr = sum(1 for w in winners if w == 0)
    print(
        f"Rust MCTS + Metal Benchmark | {count_params(p0)}p | 16 games in {dt:.2f}s "
        f"({n_games / max(0.01, dt):.2f} games/s) | avg {avg_m:.1f} moves | B/W/D={bw}/{ww}/{dr}"
    )


if __name__ == "__main__":
    _benchmark()
