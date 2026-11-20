"""15x15 Renju Gomoku environment, curriculum generator, and rule-based opponent."""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import numpy as np

BOARD_SIZE = 15
NUM_CELLS = BOARD_SIZE * BOARD_SIZE
EMPTY = 0
BLACK = 1
WHITE = 2
AXES = ((0, 1), (1, 0), (1, 1), (1, -1))


def opponent_of(player: int) -> int:
    return WHITE if player == BLACK else BLACK


class RenjuBoard:
    """Fast 15x15 Renju board with forbidden-move detection and tensor encoding."""

    def __init__(self) -> None:
        self.cells = [EMPTY] * NUM_CELLS
        self.last_move: Optional[int] = None
        self.move_count: int = 0

    def clone(self) -> "RenjuBoard":
        b = RenjuBoard()
        b.cells = self.cells.copy()
        b.last_move = self.last_move
        b.move_count = self.move_count
        return b

    @staticmethod
    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    def _contiguous_indices(
        self, r: int, c: int, dr: int, dc: int, player: int
    ) -> List[int]:
        neg: List[int] = []
        for step in range(1, BOARD_SIZE):
            nr, nc = r - step * dr, c - step * dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and self.cells[nr * BOARD_SIZE + nc] == player:
                neg.append(nr * BOARD_SIZE + nc)
            else:
                break
        line = neg[::-1]
        line.append(r * BOARD_SIZE + c)
        for step in range(1, BOARD_SIZE):
            nr, nc = r + step * dr, c + step * dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and self.cells[nr * BOARD_SIZE + nc] == player:
                line.append(nr * BOARD_SIZE + nc)
            else:
                break
        return line

    def _contiguous_len(self, r: int, c: int, dr: int, dc: int, player: int) -> int:
        cnt = 1
        for sign in (-1, 1):
            for step in range(1, BOARD_SIZE):
                nr, nc = r + sign * step * dr, c + sign * step * dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and self.cells[nr * BOARD_SIZE + nc] == player:
                    cnt += 1
                else:
                    break
        return cnt

    def is_win_at(self, idx: int, player: int) -> bool:
        if self.cells[idx] != player:
            return False
        r, c = divmod(idx, BOARD_SIZE)
        exact_only = player == BLACK
        for dr, dc in AXES:
            ln = self._contiguous_len(r, c, dr, dc, player)
            if (ln == 5) if exact_only else (ln >= 5):
                return True
        return False

    def _has_overline_at(self, r: int, c: int, player: int) -> bool:
        if player != BLACK:
            return False
        for dr, dc in AXES:
            if self._contiguous_len(r, c, dr, dc, player) >= 6:
                return True
        return False

    def _has_nearby_stone(self, r: int, c: int, player: int, radius: int = 2) -> bool:
        cells = self.cells
        for dr, dc in AXES:
            for d in range(-radius, radius + 1):
                if d == 0:
                    continue
                nr, nc = r + d * dr, c + d * dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if cells[nr * BOARD_SIZE + nc] == player:
                        return True
        return False

    def _fours_on_axis(
        self, r: int, c: int, dr: int, dc: int, player: int
    ) -> List[Tuple[Tuple[int, ...], List[int]]]:
        exact_only = player == BLACK
        origin = r * BOARD_SIZE + c
        groups: List[Tuple[Tuple[int, ...], List[int]]] = []
        cells = self.cells

        for d in range(-4, 5):
            if d == 0:
                continue
            nr, nc = r + d * dr, c + d * dc
            if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                continue
            nidx = nr * BOARD_SIZE + nc
            if cells[nidx] != EMPTY:
                continue

            cells[nidx] = player
            line = self._contiguous_indices(nr, nc, dr, dc, player)
            cells[nidx] = EMPTY

            if origin not in line:
                continue
            valid = (len(line) == 5) if exact_only else (len(line) >= 5)
            if not valid:
                continue

            base = tuple(sorted(x for x in line if x != nidx))
            found = False
            for i, (k, comp) in enumerate(groups):
                if k == base:
                    comp.append(nidx)
                    found = True
                    break
            if not found:
                groups.append((base, [nidx]))
        return groups

    def _open_threes_on_axis(
        self, r: int, c: int, dr: int, dc: int, player: int
    ) -> int:
        origin = r * BOARD_SIZE + c
        seen: set[Tuple[int, ...]] = set()
        cells = self.cells

        for d in range(-3, 4):
            if d == 0:
                continue
            qr, qc = r + d * dr, c + d * dc
            if not (0 <= qr < BOARD_SIZE and 0 <= qc < BOARD_SIZE):
                continue
            qidx = qr * BOARD_SIZE + qc
            if cells[qidx] != EMPTY:
                continue

            cells[qidx] = player
            fours = self._fours_on_axis(qr, qc, dr, dc, player)
            cells[qidx] = EMPTY

            matching = None
            for stones, comp in fours:
                if len(comp) >= 2 and origin in stones:
                    matching = stones
                    break
            if matching is None:
                continue

            if player == BLACK and self.is_black_forbidden(qr, qc):
                continue

            three_key = tuple(sorted(x for x in matching if x != qidx))
            if len(three_key) == 3:
                seen.add(three_key)
        return len(seen)

    def is_black_forbidden(self, r: int, c: int) -> bool:
        idx = r * BOARD_SIZE + c
        if self.cells[idx] != EMPTY:
            return False
        if not self._has_nearby_stone(r, c, BLACK, 2):
            return False

        self.cells[idx] = BLACK
        forbidden = False
        try:
            if self.is_win_at(idx, BLACK):
                forbidden = False
            elif self._has_overline_at(r, c, BLACK):
                forbidden = True
            else:
                four_count = 0
                for dr, dc in AXES:
                    four_count += len(self._fours_on_axis(r, c, dr, dc, BLACK))
                    if four_count >= 2:
                        forbidden = True
                        break
                if not forbidden:
                    three_count = 0
                    for dr, dc in AXES:
                        if self._fours_on_axis(r, c, dr, dc, BLACK):
                            continue
                        three_count += self._open_threes_on_axis(r, c, dr, dc, BLACK)
                        if three_count >= 2:
                            forbidden = True
                            break
        finally:
            self.cells[idx] = EMPTY
        return forbidden

    def is_legal_move(self, idx: int, player: int) -> bool:
        if idx < 0 or idx >= NUM_CELLS or self.cells[idx] != EMPTY:
            return False
        if player == BLACK:
            r, c = divmod(idx, BOARD_SIZE)
            return not self.is_black_forbidden(r, c)
        return True

    def legal_mask(self, player: int) -> np.ndarray:
        mask = np.zeros(NUM_CELLS, dtype=bool)
        cells = self.cells
        for i in range(NUM_CELLS):
            if cells[i] == EMPTY:
                if player == WHITE:
                    mask[i] = True
                else:
                    r, c = divmod(i, BOARD_SIZE)
                    mask[i] = not self.is_black_forbidden(r, c)
        return mask

    def place_move(self, idx: int, player: int) -> None:
        self.cells[idx] = player
        self.last_move = idx
        self.move_count += 1

    def candidate_indices(self, radius: int = 2) -> List[int]:
        cells = self.cells
        marked = [False] * NUM_CELLS
        has_stone = False
        for i in range(NUM_CELLS):
            if cells[i] == EMPTY:
                continue
            has_stone = True
            r, c = divmod(i, BOARD_SIZE)
            r0 = max(0, r - radius)
            r1 = min(BOARD_SIZE, r + radius + 1)
            c0 = max(0, c - radius)
            c1 = min(BOARD_SIZE, c + radius + 1)
            for nr in range(r0, r1):
                base = nr * BOARD_SIZE
                for nc in range(c0, c1):
                    idx = base + nc
                    if cells[idx] == EMPTY:
                        marked[idx] = True
        if not has_stone:
            return [7 * BOARD_SIZE + 7]
        return [i for i, m in enumerate(marked) if m]

    def encode_tensor(self, player: int, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Encodes [6, 15, 15]:
        0: current player stones
        1: opponent stones
        2: last move one-hot
        3: turn color (1.0 if BLACK else 0.0)
        4: legal mask
        5: ones plane (board boundary anchor)
        """
        if mask is None:
            mask = self.legal_mask(player)
        opp = opponent_of(player)
        arr = np.array(self.cells, dtype=np.float32).reshape(BOARD_SIZE, BOARD_SIZE)
        t = np.zeros((6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        t[0] = (arr == player).astype(np.float32)
        t[1] = (arr == opp).astype(np.float32)
        if self.last_move is not None:
            lr, lc = divmod(self.last_move, BOARD_SIZE)
            t[2, lr, lc] = 1.0
        if player == BLACK:
            t[3].fill_(1.0) if hasattr(t[3], "fill_") else t[3].fill(1.0)
        t[4] = mask.astype(np.float32).reshape(BOARD_SIZE, BOARD_SIZE)
        t[5].fill(1.0)
        return t


def _line_shape_score(board: RenjuBoard, idx: int, player: int) -> Tuple[int, int, int, int]:
    """Fast local tactical signature on placing `player` at `idx`:
    Returns (max_contig_or_win, open_fours, blocked_fours, open_threes).
    """
    r, c = divmod(idx, BOARD_SIZE)
    cells = board.cells
    opp = opponent_of(player)
    exact_only = player == BLACK

    best_score = 0
    open_fours = 0
    blocked_fours = 0
    open_threes = 0
    open_twos = 0

    cells[idx] = player
    try:
        for dr, dc in AXES:
            # Sample window [-5..+5] around (r, c)
            cnt_neg = 0
            for s in range(1, 6):
                nr, nc = r - s * dr, c - s * dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and cells[nr * BOARD_SIZE + nc] == player:
                    cnt_neg += 1
                else:
                    break
            cnt_pos = 0
            for s in range(1, 6):
                nr, nc = r + s * dr, c + s * dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and cells[nr * BOARD_SIZE + nc] == player:
                    cnt_pos += 1
                else:
                    break
            run = cnt_neg + 1 + cnt_pos

            lr, lc = r - (cnt_neg + 1) * dr, c - (cnt_neg + 1) * dc
            rr, rc = r + (cnt_pos + 1) * dr, c + (cnt_pos + 1) * dc
            left_empty = 0 <= lr < BOARD_SIZE and 0 <= lc < BOARD_SIZE and cells[lr * BOARD_SIZE + lc] == EMPTY
            right_empty = 0 <= rr < BOARD_SIZE and 0 <= rc < BOARD_SIZE and cells[rr * BOARD_SIZE + rc] == EMPTY

            # Check immediate overline prohibition edge
            left_same_after = False
            right_same_after = False
            if exact_only:
                if left_empty:
                    l2r, l2c = lr - dr, lc - dc
                    if 0 <= l2r < BOARD_SIZE and 0 <= l2c < BOARD_SIZE and cells[l2r * BOARD_SIZE + l2c] == player:
                        left_same_after = True
                if right_empty:
                    r2r, r2c = rr + dr, rc + dc
                    if 0 <= r2r < BOARD_SIZE and 0 <= r2c < BOARD_SIZE and cells[r2r * BOARD_SIZE + r2c] == player:
                        right_same_after = True

            if (run == 5) if exact_only else (run >= 5):
                return 1_000_000, 1, 0, 0

            # Check 1-gap extensions on both sides
            ext_left = 0
            if left_empty:
                for s in range(1, 5):
                    nr, nc = lr - s * dr, lc - s * dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and cells[nr * BOARD_SIZE + nc] == player:
                        ext_left += 1
                    else:
                        break
            ext_right = 0
            if right_empty:
                for s in range(1, 5):
                    nr, nc = rr + s * dr, rc + s * dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and cells[nr * BOARD_SIZE + nc] == player:
                        ext_right += 1
                    else:
                        break

            # Jump-4 check
            jump4 = False
            if left_empty and run + ext_left == 4 and not (exact_only and left_same_after and run + ext_left > 4):
                jump4 = True
            if right_empty and run + ext_right == 4 and not (exact_only and right_same_after and run + ext_right > 4):
                jump4 = True

            if run == 4:
                open_ends = int(left_empty and not left_same_after) + int(right_empty and not right_same_after)
                if open_ends == 2:
                    open_fours += 1
                elif open_ends == 1:
                    blocked_fours += 1
            elif jump4:
                blocked_fours += 1
            elif run == 3:
                if left_empty and right_empty:
                    # Ensure ends themselves aren't blocked right outside for open-3
                    l2r, l2c = lr - dr, lc - dc
                    r2r, r2c = rr + dr, rc + dc
                    l2_open = 0 <= l2r < BOARD_SIZE and 0 <= l2c < BOARD_SIZE and cells[l2r * BOARD_SIZE + l2c] == EMPTY
                    r2_open = 0 <= r2r < BOARD_SIZE and 0 <= r2c < BOARD_SIZE and cells[r2r * BOARD_SIZE + r2c] == EMPTY
                    if l2_open or r2_open:
                        open_threes += 1
                    else:
                        best_score += 200
                elif left_empty or right_empty:
                    best_score += 250
            elif run + max(ext_left, ext_right) == 3 and (ext_left > 0 or ext_right > 0):
                # Jump-3 (e.g. X _ X X)
                if left_empty and ext_left > 0:
                    outer_r, outer_c = lr - (ext_left + 1) * dr, lc - (ext_left + 1) * dc
                    if right_empty and 0 <= outer_r < BOARD_SIZE and 0 <= outer_c < BOARD_SIZE and cells[outer_r * BOARD_SIZE + outer_c] == EMPTY:
                        open_threes += 1
                    else:
                        best_score += 180
                if right_empty and ext_right > 0:
                    outer_r, outer_c = rr + (ext_right + 1) * dr, rc + (ext_right + 1) * dc
                    if left_empty and 0 <= outer_r < BOARD_SIZE and 0 <= outer_c < BOARD_SIZE and cells[outer_r * BOARD_SIZE + outer_c] == EMPTY:
                        open_threes += 1
                    else:
                        best_score += 180
            elif run == 2:
                if left_empty and right_empty:
                    open_twos += 1
                elif left_empty or right_empty:
                    best_score += 35
            elif run == 1:
                if left_empty and right_empty:
                    best_score += 8
    finally:
        cells[idx] = EMPTY

    # Center proximity tie-breaker
    center_bonus = max(0, 14 - (abs(r - 7) + abs(c - 7))) * 3
    score = (
        best_score
        + open_fours * 80_000
        + blocked_fours * 7_500
        + open_threes * 4_500
        + open_twos * 280
        + center_bonus
    )
    if blocked_fours >= 1 and open_threes >= 1:
        score += 45_000  # 4-3 winning fork!
    elif open_threes >= 2:
        score += 22_000  # Double-3 (legal for White)
    elif blocked_fours >= 2:
        score += 35_000  # Double-4 (legal for White)
    return score, open_fours, blocked_fours, open_threes


def score_candidates(
    board: RenjuBoard, player: int, mask: Optional[np.ndarray] = None
) -> Tuple[List[Tuple[int, float]], bool]:
    """Evaluates legal candidate moves with tactical urgency.
    Returns (sorted [(idx, score), ...], is_critical_tactical_position).
    """
    if mask is None:
        mask = board.legal_mask(player)
    opp = opponent_of(player)
    cands = [idx for idx in board.candidate_indices(radius=2) if mask[idx]]
    if not cands:
        legal_all = np.flatnonzero(mask)
        if len(legal_all) == 0:
            return [], False
        cands = legal_all.tolist()

    scored: List[Tuple[int, float]] = []
    has_critical = False
    for idx in cands:
        my_s, my_of, my_bf, my_ot = _line_shape_score(board, idx, player)
        if my_s >= 1_000_000:
            has_critical = True
            scored.append((idx, 2_000_000.0))
            continue

        # Check what opponent gets if opponent places here
        opp_s, opp_of, opp_bf, opp_ot = _line_shape_score(board, idx, opp)
        if opp_s >= 1_000_000:
            has_critical = True
            scored.append((idx, 1_200_000.0 + my_s * 0.1))
            continue

        if my_of >= 1 or (my_bf >= 1 and my_ot >= 1) or opp_of >= 1 or (opp_bf >= 1 and opp_ot >= 1):
            has_critical = True

        total = my_s + opp_s * 0.92
        scored.append((idx, float(total)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored, has_critical


def rule_based_move(
    board: RenjuBoard, player: int, noise_eps: float = 0.0, rng: Optional[random.Random] = None
) -> Optional[int]:
    scored, is_crit = score_candidates(board, player)
    if not scored:
        return None
    if noise_eps > 0.0 and rng is not None and not is_crit and len(scored) > 1:
        if rng.random() < noise_eps:
            top_k = scored[: min(5, len(scored))]
            max_s = top_k[0][1]
            weights = [math.exp((s - max_s) / max(150.0, abs(max_s) * 0.25)) for _, s in top_k]
            return rng.choices([m for m, _ in top_k], weights=weights, k=1)[0]
    return scored[0][0]


def board_heuristic_value(board: RenjuBoard, perspective: int) -> float:
    """Quick [-1, 1] heuristic board evaluation from `perspective` viewpoint."""
    opp = opponent_of(perspective)
    cands = board.candidate_indices(radius=2)
    my_best = 0.0
    opp_best = 0.0
    my_sum = 0.0
    opp_sum = 0.0
    for idx in cands:
        ms, _, _, _ = _line_shape_score(board, idx, perspective)
        os, _, _, _ = _line_shape_score(board, idx, opp)
        if ms > my_best:
            my_best = ms
        if os > opp_best:
            opp_best = os
        my_sum += min(ms, 20_000)
        opp_sum += min(os, 20_000)

    if my_best >= 1_000_000:
        return 0.95
    if opp_best >= 1_000_000:
        return -0.85
    diff = (my_best + 0.15 * my_sum) - 1.05 * (opp_best + 0.15 * opp_sum)
    return math.tanh(diff / 18_000.0)


def generate_seeded_board(
    rng: random.Random, min_stones: int = 2, max_stones: int = 24
) -> Optional[Tuple[RenjuBoard, int]]:
    """Generates a valid, non-terminal Renju board by seeding random opening stones + noisy play."""
    b = RenjuBoard()
    player = BLACK
    target_moves = rng.randint(min_stones, max_stones)

    # First 2-4 stones: semi-random near center (radius 3..5) to diversify openings
    rand_prefix = min(target_moves, rng.randint(2, 5))
    for step in range(rand_prefix):
        if step == 0 and rng.random() < 0.5:
            idx = 7 * BOARD_SIZE + 7
        else:
            r = rng.randint(4, 10)
            c = rng.randint(4, 10)
            idx = r * BOARD_SIZE + c
            tries = 0
            while not b.is_legal_move(idx, player) and tries < 20:
                idx = rng.randint(3, 11) * BOARD_SIZE + rng.randint(3, 11)
                tries += 1
        if not b.is_legal_move(idx, player):
            return None
        b.place_move(idx, player)
        if b.is_win_at(idx, player):
            return None
        player = opponent_of(player)

    for _ in range(rand_prefix, target_moves):
        mv = rule_based_move(b, player, noise_eps=0.35, rng=rng)
        if mv is None:
            return None
        b.place_move(mv, player)
        if b.is_win_at(mv, player):
            return None
        player = opponent_of(player)

    return b, player


def generate_dataset(
    num_samples: int, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates (states [N,6,15,15], policy_probs [N,225], values [N], best_moves [N], is_critical [N])."""
    rng = random.Random(seed)
    states: List[np.ndarray] = []
    policies: List[np.ndarray] = []
    values: List[float] = []
    best_moves: List[int] = []
    criticals: List[bool] = []

    while len(states) < num_samples:
        seeded = generate_seeded_board(rng, min_stones=2, max_stones=26)
        if seeded is None:
            continue
        b, player = seeded
        mask = b.legal_mask(player)
        scored, is_crit = score_candidates(b, player, mask)
        if not scored:
            continue

        # Target policy: focused softmax over top candidates
        pi = np.zeros(NUM_CELLS, dtype=np.float32)
        top = scored[: min(16, len(scored))]
        max_s = top[0][1]
        temp = 350.0 if max_s < 50_000 else max_s * 0.08
        exps = [math.exp(max(-30.0, (s - max_s) / temp)) for _, s in top]
        z = sum(exps)
        for (idx, _), e in zip(top, exps):
            pi[idx] = e / z

        val = board_heuristic_value(b, player)
        states.append(b.encode_tensor(player, mask))
        policies.append(pi)
        values.append(val)
        best_moves.append(scored[0][0])
        criticals.append(is_crit)

    return (
        np.stack(states, axis=0),
        np.stack(policies, axis=0),
        np.array(values, dtype=np.float32),
        np.array(best_moves, dtype=np.int64),
        np.array(criticals, dtype=bool),
    )
