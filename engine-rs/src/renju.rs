use serde::{Deserialize, Serialize};

pub const BOARD_SIZE: usize = 15;
pub const NUM_CELLS: usize = BOARD_SIZE * BOARD_SIZE;

pub const EMPTY: u8 = 0;
pub const BLACK: u8 = 1;
pub const WHITE: u8 = 2;

const AXES: [(isize, isize); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuleSet {
    Renju,
    Freestyle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForbiddenReason {
    ThreeThree,
    FourFour,
    Overline,
}

#[derive(Clone)]
pub struct Board {
    pub cells: [u8; NUM_CELLS],
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Board {
    pub fn new() -> Self {
        Self {
            cells: [EMPTY; NUM_CELLS],
        }
    }

    #[inline]
    pub fn idx(row: usize, col: usize) -> usize {
        row * BOARD_SIZE + col
    }

    #[inline]
    pub fn rc(idx: usize) -> (usize, usize) {
        (idx / BOARD_SIZE, idx % BOARD_SIZE)
    }

    #[inline]
    pub fn in_bounds(r: isize, c: isize) -> bool {
        r >= 0 && r < BOARD_SIZE as isize && c >= 0 && c < BOARD_SIZE as isize
    }

    #[inline]
    pub fn get(&self, r: isize, c: isize) -> Option<u8> {
        if Self::in_bounds(r, c) {
            Some(self.cells[Self::idx(r as usize, c as usize)])
        } else {
            None
        }
    }

    #[inline]
    pub fn opponent(player: u8) -> u8 {
        if player == BLACK {
            WHITE
        } else {
            BLACK
        }
    }

    fn contiguous_line_indices(
        &self,
        row: usize,
        col: usize,
        dr: isize,
        dc: isize,
        player: u8,
    ) -> Vec<usize> {
        let mut line = Vec::with_capacity(9);
        let r0 = row as isize;
        let c0 = col as isize;

        let mut neg = Vec::new();
        for step in 1..BOARD_SIZE as isize {
            let nr = r0 - step * dr;
            let nc = c0 - step * dc;
            if self.get(nr, nc) == Some(player) {
                neg.push(Self::idx(nr as usize, nc as usize));
            } else {
                break;
            }
        }
        for &idx in neg.iter().rev() {
            line.push(idx);
        }
        line.push(Self::idx(row, col));
        for step in 1..BOARD_SIZE as isize {
            let nr = r0 + step * dr;
            let nc = c0 + step * dc;
            if self.get(nr, nc) == Some(player) {
                line.push(Self::idx(nr as usize, nc as usize));
            } else {
                break;
            }
        }
        line
    }

    pub fn is_win_at(&self, row: usize, col: usize, player: u8, rules: RuleSet) -> bool {
        if self.cells[Self::idx(row, col)] != player {
            return false;
        }
        let exact_only = rules == RuleSet::Renju && player == BLACK;
        for &(dr, dc) in &AXES {
            let len = self.contiguous_line_indices(row, col, dr, dc, player).len();
            if if exact_only { len == 5 } else { len >= 5 } {
                return true;
            }
        }
        false
    }

    fn has_overline_at(&self, row: usize, col: usize, player: u8, rules: RuleSet) -> bool {
        if rules != RuleSet::Renju || player != BLACK {
            return false;
        }
        for &(dr, dc) in &AXES {
            if self.contiguous_line_indices(row, col, dr, dc, player).len() >= 6 {
                return true;
            }
        }
        false
    }

    fn has_nearby_stone(&self, row: usize, col: usize, player: u8, radius: isize) -> bool {
        let r0 = row as isize;
        let c0 = col as isize;
        for &(dr, dc) in &AXES {
            for d in -radius..=radius {
                if d == 0 {
                    continue;
                }
                if self.get(r0 + d * dr, c0 + d * dc) == Some(player) {
                    return true;
                }
            }
        }
        false
    }

    /// Returns `(stone_key, completing_cells)` pairs for distinct Fours containing `(row, col)` on `(dr, dc)`.
    fn fours_on_axis(
        &mut self,
        row: usize,
        col: usize,
        dr: isize,
        dc: isize,
        player: u8,
        rules: RuleSet,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        let exact_only = rules == RuleSet::Renju && player == BLACK;
        let origin = Self::idx(row, col);
        let mut groups: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();

        for d in -4isize..=4 {
            if d == 0 {
                continue;
            }
            let nr = row as isize + d * dr;
            let nc = col as isize + d * dc;
            if !Self::in_bounds(nr, nc) {
                continue;
            }
            let nidx = Self::idx(nr as usize, nc as usize);
            if self.cells[nidx] != EMPTY {
                continue;
            }

            self.cells[nidx] = player;
            let line = self.contiguous_line_indices(nr as usize, nc as usize, dr, dc, player);
            self.cells[nidx] = EMPTY;

            if !line.contains(&origin) {
                continue;
            }
            let valid = if exact_only {
                line.len() == 5
            } else {
                line.len() >= 5
            };
            if !valid {
                continue;
            }

            let mut base: Vec<usize> = line.into_iter().filter(|&x| x != nidx).collect();
            base.sort_unstable();

            if let Some(entry) = groups.iter_mut().find(|(k, _)| *k == base) {
                entry.1.push(nidx);
            } else {
                groups.push((base, vec![nidx]));
            }
        }

        groups
    }

    fn open_threes_on_axis(
        &mut self,
        row: usize,
        col: usize,
        dr: isize,
        dc: isize,
        player: u8,
        rules: RuleSet,
    ) -> usize {
        let origin = Self::idx(row, col);
        let mut seen_keys: Vec<Vec<usize>> = Vec::new();

        for d in -3isize..=3 {
            if d == 0 {
                continue;
            }
            let qr = row as isize + d * dr;
            let qc = col as isize + d * dc;
            if !Self::in_bounds(qr, qc) {
                continue;
            }
            let qidx = Self::idx(qr as usize, qc as usize);
            if self.cells[qidx] != EMPTY {
                continue;
            }

            self.cells[qidx] = player;
            let fours = self.fours_on_axis(qr as usize, qc as usize, dr, dc, player, rules);
            self.cells[qidx] = EMPTY;

            let matching = fours
                .into_iter()
                .find(|(stones, comp)| comp.len() >= 2 && stones.contains(&origin));

            let Some((four_stones, _)) = matching else {
                continue;
            };

            if rules == RuleSet::Renju
                && player == BLACK
                && self.black_forbidden_reason(qr as usize, qc as usize, rules).is_some()
            {
                continue;
            }

            let mut three_key: Vec<usize> =
                four_stones.into_iter().filter(|&x| x != qidx).collect();
            three_key.sort_unstable();
            if three_key.len() == 3 && !seen_keys.contains(&three_key) {
                seen_keys.push(three_key);
            }
        }

        seen_keys.len()
    }

    pub fn black_forbidden_reason(
        &mut self,
        row: usize,
        col: usize,
        rules: RuleSet,
    ) -> Option<ForbiddenReason> {
        if rules != RuleSet::Renju {
            return None;
        }
        let idx = Self::idx(row, col);
        if self.cells[idx] != EMPTY {
            return None;
        }
        if !self.has_nearby_stone(row, col, BLACK, 2) {
            return None;
        }

        self.cells[idx] = BLACK;
        let reason = (|| {
            if self.is_win_at(row, col, BLACK, rules) {
                return None;
            }
            if self.has_overline_at(row, col, BLACK, rules) {
                return Some(ForbiddenReason::Overline);
            }

            let mut four_count = 0;
            for &(dr, dc) in &AXES {
                four_count += self.fours_on_axis(row, col, dr, dc, BLACK, rules).len();
                if four_count >= 2 {
                    return Some(ForbiddenReason::FourFour);
                }
            }

            let mut open_three_count = 0;
            for &(dr, dc) in &AXES {
                if !self.fours_on_axis(row, col, dr, dc, BLACK, rules).is_empty() {
                    continue;
                }
                open_three_count += self.open_threes_on_axis(row, col, dr, dc, BLACK, rules);
                if open_three_count >= 2 {
                    return Some(ForbiddenReason::ThreeThree);
                }
            }

            None
        })();
        self.cells[idx] = EMPTY;
        reason
    }

    pub fn is_legal_move(&mut self, idx: usize, player: u8, rules: RuleSet) -> bool {
        if idx >= NUM_CELLS || self.cells[idx] != EMPTY {
            return false;
        }
        if rules == RuleSet::Renju && player == BLACK {
            let (r, c) = Self::rc(idx);
            return self.black_forbidden_reason(r, c, rules).is_none();
        }
        true
    }

    pub fn legal_mask(&mut self, player: u8, rules: RuleSet) -> [bool; NUM_CELLS] {
        let mut mask = [false; NUM_CELLS];
        for i in 0..NUM_CELLS {
            if self.cells[i] == EMPTY {
                mask[i] = self.is_legal_move(i, player, rules);
            }
        }
        mask
    }

    /// Encodes `[4, 15, 15]` tensor:
    /// - plane 0: current player stones
    /// - plane 1: opponent stones
    /// - plane 2: 1.0 if player == BLACK else 0.0
    /// - plane 3: legal move mask
    pub fn encode_tensor(&mut self, player: u8, rules: RuleSet) -> [f32; 4 * NUM_CELLS] {
        let mask = self.legal_mask(player, rules);
        self.encode_tensor_with_mask(player, &mask)
    }

    pub fn encode_tensor_with_mask(
        &self,
        player: u8,
        mask: &[bool; NUM_CELLS],
    ) -> [f32; 4 * NUM_CELLS] {
        let opp = Self::opponent(player);
        let color_val = if player == BLACK { 1.0f32 } else { 0.0f32 };
        let mut out = [0.0f32; 4 * NUM_CELLS];

        for i in 0..NUM_CELLS {
            let v = self.cells[i];
            if v == player {
                out[i] = 1.0;
            } else if v == opp {
                out[NUM_CELLS + i] = 1.0;
            }
            out[2 * NUM_CELLS + i] = color_val;
            if mask[i] {
                out[3 * NUM_CELLS + i] = 1.0;
            }
        }
        out
    }
}

/// Applies dihedral symmetry `sym` (0..8) to a `(4, 15, 15)` state tensor and `225`-length policy.
pub fn augment_dihedral(
    state: &[f32; 4 * NUM_CELLS],
    policy: &[f32; NUM_CELLS],
    sym: usize,
) -> ([f32; 4 * NUM_CELLS], [f32; NUM_CELLS]) {
    let mut s_out = [0.0f32; 4 * NUM_CELLS];
    let mut p_out = [0.0f32; NUM_CELLS];

    for r in 0..BOARD_SIZE {
        for c in 0..BOARD_SIZE {
            let (nr, nc) = transform_rc(r, c, sym);
            let src = r * BOARD_SIZE + c;
            let dst = nr * BOARD_SIZE + nc;
            p_out[dst] = policy[src];
            for ch in 0..4 {
                s_out[ch * NUM_CELLS + dst] = state[ch * NUM_CELLS + src];
            }
        }
    }
    (s_out, p_out)
}

#[inline]
fn transform_rc(r: usize, c: usize, sym: usize) -> (usize, usize) {
    let n = BOARD_SIZE - 1;
    let (rr, cc) = match sym & 3 {
        0 => (r, c),
        1 => (c, n - r),
        2 => (n - r, n - c),
        _ => (n - c, r),
    };
    if sym & 4 != 0 {
        (rr, n - cc)
    } else {
        (rr, cc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_renju_forbidden_and_overrides() {
        let mut b = Board::new();
        // 3-3 at (7, 7)
        b.cells[Board::idx(7, 5)] = BLACK;
        b.cells[Board::idx(7, 6)] = BLACK;
        b.cells[Board::idx(5, 7)] = BLACK;
        b.cells[Board::idx(8, 7)] = BLACK;
        assert_eq!(
            b.black_forbidden_reason(7, 7, RuleSet::Renju),
            Some(ForbiddenReason::ThreeThree)
        );
        assert!(!b.is_legal_move(Board::idx(7, 7), BLACK, RuleSet::Renju));
        assert!(b.is_legal_move(Board::idx(7, 7), WHITE, RuleSet::Renju));

        // Same-line 4-4 at (7, 7): ● ● ● _ [X] _ ● ● ●
        let mut b44 = Board::new();
        for c in [3, 4, 5, 9, 10, 11] {
            b44.cells[Board::idx(7, c)] = BLACK;
        }
        assert_eq!(
            b44.black_forbidden_reason(7, 7, RuleSet::Renju),
            Some(ForbiddenReason::FourFour)
        );

        // Exact 5 overrides overline
        let mut b5 = Board::new();
        for c in [3, 4, 5, 6] {
            b5.cells[Board::idx(7, c)] = BLACK;
        }
        for r in [4, 5, 6, 8, 9] {
            b5.cells[Board::idx(r, 7)] = BLACK;
        }
        assert_eq!(b5.black_forbidden_reason(7, 7, RuleSet::Renju), None);
        b5.cells[Board::idx(7, 7)] = BLACK;
        assert!(b5.is_win_at(7, 7, BLACK, RuleSet::Renju));
    }
}
