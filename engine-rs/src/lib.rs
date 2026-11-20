pub mod renju;

use rand::distributions::WeightedIndex;
use rand::prelude::*;
use renju::{Board, RuleSet, BLACK, BOARD_SIZE, EMPTY, NUM_CELLS};
use std::slice;

const AXES: [(isize, isize); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
const MAX_ARENA_REUSE: usize = 8192;

struct Tables {
    nb_r1: [Vec<u16>; NUM_CELLS],
    nb_r2: [Vec<u16>; NUM_CELLS],
    nb_r3: [Vec<u16>; NUM_CELLS],
    ax_nb4: [Vec<u16>; NUM_CELLS],
    d4_perm: [[u16; NUM_CELLS]; 8],
}

fn build_radius_table(radius: isize) -> [Vec<u16>; NUM_CELLS] {
    let mut table: [Vec<u16>; NUM_CELLS] = std::array::from_fn(|_| Vec::new());
    for r in 0..BOARD_SIZE as isize {
        for c in 0..BOARD_SIZE as isize {
            let idx = (r as usize) * BOARD_SIZE + (c as usize);
            let mut v = Vec::new();
            for dr in -radius..=radius {
                for dc in -radius..=radius {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let nr = r + dr;
                    let nc = c + dc;
                    if nr >= 0 && nr < BOARD_SIZE as isize && nc >= 0 && nc < BOARD_SIZE as isize {
                        v.push(((nr as usize) * BOARD_SIZE + (nc as usize)) as u16);
                    }
                }
            }
            table[idx] = v;
        }
    }
    table
}

fn build_tables() -> Tables {
    let nb_r1 = build_radius_table(1);
    let nb_r2 = build_radius_table(2);
    let nb_r3 = build_radius_table(3);

    let mut ax_nb4: [Vec<u16>; NUM_CELLS] = std::array::from_fn(|_| Vec::with_capacity(32));
    for r in 0..BOARD_SIZE as isize {
        for c in 0..BOARD_SIZE as isize {
            let idx = (r as usize) * BOARD_SIZE + (c as usize);
            let mut v = Vec::with_capacity(32);
            for &(dr, dc) in &AXES {
                for s in -4isize..=4isize {
                    if s == 0 {
                        continue;
                    }
                    let nr = r + s * dr;
                    let nc = c + s * dc;
                    if nr >= 0 && nr < BOARD_SIZE as isize && nc >= 0 && nc < BOARD_SIZE as isize {
                        v.push(((nr as usize) * BOARD_SIZE + (nc as usize)) as u16);
                    }
                }
            }
            ax_nb4[idx] = v;
        }
    }

    let mut d4_perm = [[0u16; NUM_CELLS]; 8];
    let n = BOARD_SIZE - 1;
    for sym in 0..8 {
        let k = sym & 3;
        let flip = (sym & 4) != 0;
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                let (rr, mut cc) = match k {
                    0 => (r, c),
                    1 => (c, n - r),
                    2 => (n - r, n - c),
                    _ => (n - c, r),
                };
                if flip {
                    cc = n - cc;
                }
                let src = r * BOARD_SIZE + c;
                let dst = rr * BOARD_SIZE + cc;
                d4_perm[sym][dst] = src as u16;
            }
        }
    }

    Tables {
        nb_r1,
        nb_r2,
        nb_r3,
        ax_nb4,
        d4_perm,
    }
}

fn tables() -> &'static Tables {
    use std::sync::OnceLock;
    static INSTANCE: OnceLock<Tables> = OnceLock::new();
    INSTANCE.get_or_init(build_tables)
}

fn local_empty_indices(cells: &[u8; NUM_CELLS], nb_table: &[Vec<u16>; NUM_CELLS]) -> Vec<usize> {
    let mut marked = [false; NUM_CELLS];
    let mut has_stone = false;
    for (idx, &v) in cells.iter().enumerate() {
        if v != EMPTY {
            has_stone = true;
            for &nb in &nb_table[idx] {
                let u = nb as usize;
                if cells[u] == EMPTY {
                    marked[u] = true;
                }
            }
        }
    }
    if !has_stone {
        return vec![7 * BOARD_SIZE + 7];
    }
    let mut out = Vec::with_capacity(64);
    for (i, &m) in marked.iter().enumerate() {
        if m {
            out.push(i);
        }
    }
    out
}

pub fn exact_legal_mask_fast(board: &mut Board, player: u8) -> [bool; NUM_CELLS] {
    let tbl = tables();
    let mut mask = [false; NUM_CELLS];
    for i in 0..NUM_CELLS {
        if board.cells[i] == EMPTY {
            mask[i] = true;
        }
    }
    if player == BLACK {
        let mut cnt = [0u8; NUM_CELLS];
        for i in 0..NUM_CELLS {
            if board.cells[i] == BLACK {
                for &nb in &tbl.ax_nb4[i] {
                    cnt[nb as usize] = cnt[nb as usize].saturating_add(1);
                }
            }
        }
        for i in 0..NUM_CELLS {
            if mask[i] && cnt[i] >= 2 {
                let (r, c) = Board::rc(i);
                if board.black_forbidden_reason(r, c, RuleSet::Renju).is_some() {
                    mask[i] = false;
                }
            }
        }
    }
    mask
}

fn encode_nhwc_one(
    board: &mut Board,
    last_move: i32,
    player: u8,
    mask_opt: Option<&[bool; NUM_CELLS]>,
    out_nhwc: &mut [f32],
) {
    let computed_mask;
    let mask = match mask_opt {
        Some(m) => m,
        None => {
            computed_mask = exact_legal_mask_fast(board, player);
            &computed_mask
        }
    };
    let opp = Board::opponent(player);
    let color_val = if player == BLACK { 1.0f32 } else { 0.0f32 };
    let lm = if last_move >= 0 && (last_move as usize) < NUM_CELLS {
        Some(last_move as usize)
    } else {
        None
    };

    for i in 0..NUM_CELLS {
        let base = i * 6;
        let v = board.cells[i];
        out_nhwc[base] = if v == player { 1.0 } else { 0.0 };
        out_nhwc[base + 1] = if v == opp { 1.0 } else { 0.0 };
        out_nhwc[base + 2] = if lm == Some(i) { 1.0 } else { 0.0 };
        out_nhwc[base + 3] = color_val;
        out_nhwc[base + 4] = if mask[i] { 1.0 } else { 0.0 };
        out_nhwc[base + 5] = 1.0;
    }
}

enum TacticalScan {
    WinMove(usize),
    ForcedBlocks(Vec<usize>),
    TerminalLoss,
    Normal,
}

fn scan_tactical_actions(board: &mut Board, player: u8) -> TacticalScan {
    let tbl = tables();
    let r1_cands = local_empty_indices(&board.cells, &tbl.nb_r1);

    // 1. Immediate 1-ply win check (in Renju, exact-5 is legal even for Black)
    for &idx in &r1_cands {
        let (r, c) = Board::rc(idx);
        board.cells[idx] = player;
        let won = board.is_win_at(r, c, player, RuleSet::Renju);
        board.cells[idx] = EMPTY;
        if won {
            return TacticalScan::WinMove(idx);
        }
    }

    // 2. Immediate 1-ply threat check for opponent (forced blocks)
    let opp = Board::opponent(player);
    let mut opp_wins = Vec::new();
    for &idx in &r1_cands {
        let (r, c) = Board::rc(idx);
        board.cells[idx] = opp;
        let opp_won = board.is_win_at(r, c, opp, RuleSet::Renju);
        board.cells[idx] = EMPTY;
        if opp_won {
            opp_wins.push(idx);
        }
    }

    if !opp_wins.is_empty() {
        let mut legal_blocks = Vec::with_capacity(opp_wins.len());
        for idx in opp_wins {
            if board.is_legal_move(idx, player, RuleSet::Renju) {
                legal_blocks.push(idx);
            }
        }
        if !legal_blocks.is_empty() {
            return TacticalScan::ForcedBlocks(legal_blocks);
        }
        return TacticalScan::TerminalLoss;
    }

    TacticalScan::Normal
}

fn line_shape_score(board: &mut Board, idx: usize, player: u8) -> (i64, i32, i32, i32) {
    let (r_u, c_u) = Board::rc(idx);
    let r = r_u as isize;
    let c = c_u as isize;
    let exact_only = player == BLACK;

    let mut best_score: i64 = 0;
    let mut open_fours: i32 = 0;
    let mut blocked_fours: i32 = 0;
    let mut open_threes: i32 = 0;
    let mut open_twos: i32 = 0;

    board.cells[idx] = player;

    for &(dr, dc) in &AXES {
        let mut cnt_neg = 0isize;
        for s in 1..6isize {
            let nr = r - s * dr;
            let nc = c - s * dc;
            if board.get(nr, nc) == Some(player) {
                cnt_neg += 1;
            } else {
                break;
            }
        }
        let mut cnt_pos = 0isize;
        for s in 1..6isize {
            let nr = r + s * dr;
            let nc = c + s * dc;
            if board.get(nr, nc) == Some(player) {
                cnt_pos += 1;
            } else {
                break;
            }
        }
        let run = cnt_neg + 1 + cnt_pos;

        if if exact_only { run == 5 } else { run >= 5 } {
            board.cells[idx] = EMPTY;
            return (1_000_000, 1, 0, 0);
        }

        let lr = r - (cnt_neg + 1) * dr;
        let lc = c - (cnt_neg + 1) * dc;
        let rr = r + (cnt_pos + 1) * dr;
        let rc = c + (cnt_pos + 1) * dc;
        let left_empty = board.get(lr, lc) == Some(EMPTY);
        let right_empty = board.get(rr, rc) == Some(EMPTY);

        let left_same_after = exact_only && left_empty && board.get(lr - dr, lc - dc) == Some(player);
        let right_same_after = exact_only && right_empty && board.get(rr + dr, rc + dc) == Some(player);

        let mut ext_left = 0isize;
        if left_empty {
            for s in 1..5isize {
                if board.get(lr - s * dr, lc - s * dc) == Some(player) {
                    ext_left += 1;
                } else {
                    break;
                }
            }
        }
        let mut ext_right = 0isize;
        if right_empty {
            for s in 1..5isize {
                if board.get(rr + s * dr, rc + s * dc) == Some(player) {
                    ext_right += 1;
                } else {
                    break;
                }
            }
        }

        let mut jump4 = false;
        if left_empty && run + ext_left == 4 && !(exact_only && left_same_after && run + ext_left > 4) {
            jump4 = true;
        }
        if right_empty && run + ext_right == 4 && !(exact_only && right_same_after && run + ext_right > 4) {
            jump4 = true;
        }

        if run == 4 {
            let open_ends = i32::from(left_empty && !left_same_after) + i32::from(right_empty && !right_same_after);
            if open_ends == 2 {
                open_fours += 1;
            } else if open_ends == 1 {
                blocked_fours += 1;
            }
        } else if jump4 {
            blocked_fours += 1;
        } else if run == 3 {
            if left_empty && right_empty {
                let l2_open = board.get(lr - dr, lc - dc) == Some(EMPTY);
                let r2_open = board.get(rr + dr, rc + dc) == Some(EMPTY);
                if l2_open || r2_open {
                    open_threes += 1;
                } else {
                    best_score += 200;
                }
            } else if left_empty || right_empty {
                best_score += 250;
            }
        } else if run + ext_left.max(ext_right) == 3 && (ext_left > 0 || ext_right > 0) {
            if left_empty && ext_left > 0 {
                let outer_r = lr - (ext_left + 1) * dr;
                let outer_c = lc - (ext_left + 1) * dc;
                if right_empty && board.get(outer_r, outer_c) == Some(EMPTY) {
                    open_threes += 1;
                } else {
                    best_score += 180;
                }
            }
            if right_empty && ext_right > 0 {
                let outer_r = rr + (ext_right + 1) * dr;
                let outer_c = rc + (ext_right + 1) * dc;
                if left_empty && board.get(outer_r, outer_c) == Some(EMPTY) {
                    open_threes += 1;
                } else {
                    best_score += 180;
                }
            }
        } else if run == 2 {
            if left_empty && right_empty {
                open_twos += 1;
            } else if left_empty || right_empty {
                best_score += 35;
            }
        } else if run == 1 && left_empty && right_empty {
            best_score += 8;
        }
    }

    board.cells[idx] = EMPTY;

    let dist = (r_u as isize - 7).abs() + (c_u as isize - 7).abs();
    let center_bonus = (14 - dist).max(0) as i64 * 3;
    let mut score = best_score
        + (open_fours as i64) * 80_000
        + (blocked_fours as i64) * 7_500
        + (open_threes as i64) * 4_500
        + (open_twos as i64) * 280
        + center_bonus;
    if blocked_fours >= 1 && open_threes >= 1 {
        score += 45_000;
    } else if open_threes >= 2 {
        score += 22_000;
    } else if blocked_fours >= 2 {
        score += 35_000;
    }
    (score, open_fours, blocked_fours, open_threes)
}

fn find_five_threats(board: &mut Board, attacker: u8, mask: &[bool; NUM_CELLS]) -> Vec<usize> {
    let mut threats = Vec::new();
    for i in 0..NUM_CELLS {
        if !mask[i] {
            continue;
        }
        let (s, _, _, _) = line_shape_score(board, i, attacker);
        if s >= 1_000_000 {
            threats.push(i);
            if threats.len() >= 2 {
                break;
            }
        }
    }
    threats
}

fn vcf_dfs(board: &mut Board, attacker: u8, depth: i32, nodes: &mut i32) -> Option<usize> {
    if depth <= 0 || *nodes <= 0 {
        return None;
    }
    *nodes -= 1;
    let defender = Board::opponent(attacker);
    let att_mask = exact_legal_mask_fast(board, attacker);
    let tbl = tables();
    let cands = local_empty_indices(&board.cells, &tbl.nb_r2);

    let mut forcing: Vec<(usize, i64)> = Vec::new();
    for &idx in &cands {
        if !att_mask[idx] {
            continue;
        }
        let (s, of, bf, _) = line_shape_score(board, idx, attacker);
        if s >= 1_000_000 {
            return Some(idx);
        }
        if of >= 1 {
            // Open-4 is an immediate win unless defender already has a 5 on board next turn,
            // which we guard at root / parent. Still verify defender has no immediate 5!
            return Some(idx);
        }
        if bf >= 1 {
            forcing.push((idx, s));
        }
    }
    forcing.sort_by(|a, b| b.1.cmp(&a.1));
    forcing.truncate(6);

    for (mv, _) in forcing {
        board.cells[mv] = attacker;
        let att_after_mask = exact_legal_mask_fast(board, attacker);
        let threats = find_five_threats(board, attacker, &att_after_mask);
        if threats.len() >= 2 {
            board.cells[mv] = EMPTY;
            return Some(mv);
        }
        if threats.len() == 1 {
            let block_mv = threats[0];
            let def_mask = exact_legal_mask_fast(board, defender);
            if !def_mask[block_mv] {
                // Defender cannot legally block (e.g. Black forbidden move) -> attacker wins!
                board.cells[mv] = EMPTY;
                return Some(mv);
            }
            let (def_s, _, _, _) = line_shape_score(board, block_mv, defender);
            if def_s < 1_000_000 {
                board.cells[block_mv] = defender;
                let res = vcf_dfs(board, attacker, depth - 1, nodes);
                board.cells[block_mv] = EMPTY;
                if res.is_some() {
                    board.cells[mv] = EMPTY;
                    return Some(mv);
                }
            }
        }
        board.cells[mv] = EMPTY;
    }
    None
}

fn find_vcf_move(board: &mut Board, attacker: u8, max_depth: i32, node_budget: i32) -> Option<usize> {
    let mut nodes = node_budget;
    vcf_dfs(board, attacker, max_depth, &mut nodes)
}

fn board_tactical_eval(board: &mut Board, side: u8) -> i64 {
    let tbl = tables();
    let opp = Board::opponent(side);
    let my_mask = exact_legal_mask_fast(board, side);
    let opp_mask = exact_legal_mask_fast(board, opp);
    let cands = local_empty_indices(&board.cells, &tbl.nb_r2);

    let mut my_top1: i64 = 0;
    let mut my_top2: i64 = 0;
    let mut opp_top1: i64 = 0;
    let mut opp_top2: i64 = 0;
    let mut my_of_cnt = 0;
    let mut opp_of_cnt = 0;

    for &idx in &cands {
        if my_mask[idx] {
            let (s, of, _, _) = line_shape_score(board, idx, side);
            if s >= 1_000_000 {
                return 900_000;
            }
            if of >= 1 {
                my_of_cnt += 1;
            }
            if s > my_top1 {
                my_top2 = my_top1;
                my_top1 = s;
            } else if s > my_top2 {
                my_top2 = s;
            }
        }
        if opp_mask[idx] {
            let (s, of, _, _) = line_shape_score(board, idx, opp);
            if s >= 1_000_000 {
                opp_top1 = opp_top1.max(600_000);
            }
            if of >= 1 {
                opp_of_cnt += 1;
            }
            if s > opp_top1 {
                opp_top2 = opp_top1;
                opp_top1 = s;
            } else if s > opp_top2 {
                opp_top2 = s;
            }
        }
    }
    let my_val = my_top1 + my_top2 / 3 + (my_of_cnt as i64) * 50_000;
    let opp_val = opp_top1 + opp_top2 / 3 + (opp_of_cnt as i64) * 55_000;
    my_val - (opp_val * 92) / 100
}

fn alphabeta_negamax(
    board: &mut Board,
    side: u8,
    depth: i32,
    mut alpha: i64,
    beta: i64,
    nodes: &mut i32,
) -> i64 {
    *nodes -= 1;
    if depth <= 0 || *nodes <= 0 {
        return board_tactical_eval(board, side);
    }
    let tbl = tables();
    let opp = Board::opponent(side);
    let mask = exact_legal_mask_fast(board, side);
    let cands = local_empty_indices(&board.cells, &tbl.nb_r2);

    let mut ordered: Vec<(usize, i64)> = Vec::with_capacity(cands.len());
    for &idx in &cands {
        if !mask[idx] {
            continue;
        }
        let (ms, _, _, _) = line_shape_score(board, idx, side);
        if ms >= 1_000_000 {
            return 1_000_000 + (depth as i64) * 10_000;
        }
        let (os, _, _, _) = line_shape_score(board, idx, opp);
        let key = ms + (os * 92) / 100;
        ordered.push((idx, key));
    }
    if ordered.is_empty() {
        return 0;
    }
    ordered.sort_by(|a, b| b.1.cmp(&a.1));
    let branch = if depth >= 3 { 8 } else { 6 };
    ordered.truncate(branch);

    let mut best = -2_000_000_i64;
    for (mv, _) in ordered {
        board.cells[mv] = side;
        let val = -alphabeta_negamax(board, opp, depth - 1, -beta, -alpha, nodes);
        board.cells[mv] = EMPTY;
        if val > best {
            best = val;
        }
        if val > alpha {
            alpha = val;
        }
        if alpha >= beta {
            break;
        }
    }
    best
}

pub fn rule_based_move_rust(board: &mut Board, player: u8, noise_eps: f32, seed: u64) -> Option<usize> {
    let tbl = tables();
    let mask = exact_legal_mask_fast(board, player);
    let opp = Board::opponent(player);

    let mut cands: Vec<usize> = local_empty_indices(&board.cells, &tbl.nb_r2)
        .into_iter()
        .filter(|&i| mask[i])
        .collect();
    if cands.is_empty() {
        for i in 0..NUM_CELLS {
            if mask[i] {
                cands.push(i);
            }
        }
        if cands.is_empty() {
            return None;
        }
    }

    let mut scored: Vec<(usize, f64)> = Vec::with_capacity(cands.len());
    let mut win_moves: Vec<usize> = Vec::new();
    let mut block_five_moves: Vec<(usize, i64)> = Vec::new();
    let mut is_crit = false;
    for &idx in &cands {
        let (my_s, my_of, my_bf, my_ot) = line_shape_score(board, idx, player);
        if my_s >= 1_000_000 {
            win_moves.push(idx);
        }
        let (opp_s, opp_of, opp_bf, opp_ot) = line_shape_score(board, idx, opp);
        if opp_s >= 1_000_000 {
            block_five_moves.push((idx, my_s));
        }
        if my_of >= 1 || (my_bf >= 1 && my_ot >= 1) || opp_of >= 1 || (opp_bf >= 1 && opp_ot >= 1) {
            is_crit = true;
        }
        let total = (my_s as f64) + (opp_s as f64) * 0.92;
        scored.push((idx, total));
    }

    if !win_moves.is_empty() {
        return Some(win_moves[0]);
    }
    if !block_five_moves.is_empty() {
        block_five_moves.sort_by(|a, b| b.1.cmp(&a.1));
        return Some(block_five_moves[0].0);
    }

    // 11-ply deep VCF attack search
    if let Some(vcf_mv) = find_vcf_move(board, player, 11, 500) {
        if mask[vcf_mv] {
            return Some(vcf_mv);
        }
    }

    scored.sort_by(|a, b| b.1.total_cmp(&a.1));

    // 9-ply opponent VCF threat check
    let opp_has_vcf = find_vcf_move(board, opp, 9, 300).is_some();

    // 4-ply Alpha-Beta tactical search on top-8 candidates
    let k_look = scored.len().min(8);
    let mut lookahead: Vec<(usize, f64)> = Vec::with_capacity(k_look);
    for &(idx, s1) in &scored[..k_look] {
        let mut ab_nodes = 260;
        board.cells[idx] = player;
        let ab_score = -alphabeta_negamax(board, opp, 3, -2_000_000, 2_000_000, &mut ab_nodes) as f64;
        let vcf_penalty = if opp_has_vcf {
            if find_vcf_move(board, opp, 7, 140).is_some() {
                600_000.0
            } else {
                0.0
            }
        } else {
            0.0
        };
        board.cells[idx] = EMPTY;
        let total_score = ab_score + s1 * 0.02 - vcf_penalty;
        lookahead.push((idx, total_score));
    }
    lookahead.sort_by(|a, b| b.1.total_cmp(&a.1));

    if noise_eps > 0.0 && !is_crit && !opp_has_vcf && lookahead.len() > 1 {
        let mut rng = StdRng::seed_from_u64(seed);
        if rng.gen::<f32>() < noise_eps {
            let k = lookahead.len().min(3);
            let max_s = lookahead[0].1;
            let scale = (max_s.abs() * 0.20).max(200.0);
            let weights: Vec<f64> = lookahead[..k]
                .iter()
                .map(|(_, s)| ((s - max_s) / scale).max(-20.0).exp())
                .collect();
            if let Ok(dist) = WeightedIndex::new(&weights) {
                return Some(lookahead[dist.sample(&mut rng)].0);
            }
        }
    }
    Some(lookahead[0].0)
}

// ── Arena Batched PUCT MCTS Engine ──

#[derive(Clone)]
struct ArenaNode {
    player: u8,
    expanded: bool,
    terminal: bool,
    terminal_val: f32,
    win_move: i32,
    n_sum: f32,
    actions: Vec<u16>,
    priors: Vec<f32>,
    n: Vec<f32>,
    w: Vec<f32>,
    children: Vec<u32>,
}

impl ArenaNode {
    fn new(player: u8) -> Self {
        Self {
            player,
            expanded: false,
            terminal: false,
            terminal_val: 0.0,
            win_move: -1,
            n_sum: 0.0,
            actions: Vec::new(),
            priors: Vec::new(),
            n: Vec::new(),
            w: Vec::new(),
            children: Vec::new(),
        }
    }
}

struct InFlightLeaf {
    game_idx: usize,
    leaf_idx: u32,
    cells: [u8; NUM_CELLS],
    path: Vec<(u32, usize)>,
}

struct GameSlot {
    board: Board,
    last_move: i32,
    player: u8,
    root_idx: u32,
    nodes: Vec<ArenaNode>,
    root_noise: Vec<f32>,
}

impl GameSlot {
    fn new() -> Self {
        Self {
            board: Board::new(),
            last_move: -1,
            player: BLACK,
            root_idx: 0,
            nodes: vec![ArenaNode::new(BLACK)],
            root_noise: Vec::new(),
        }
    }

    fn reset(&mut self, player: u8) {
        self.board = Board::new();
        self.last_move = -1;
        self.player = player;
        self.root_idx = 0;
        self.nodes.clear();
        self.nodes.push(ArenaNode::new(player));
        self.root_noise.clear();
    }

    fn set_board(&mut self, cells: &[u8; NUM_CELLS], last_move: i32, player: u8) {
        self.board.cells = *cells;
        self.last_move = last_move;
        self.player = player;
        self.root_idx = 0;
        self.nodes.clear();
        self.nodes.push(ArenaNode::new(player));
        self.root_noise.clear();
    }

    fn advance_root(&mut self, mv: usize, next_player: u8) {
        self.root_noise.clear();
        if mv >= NUM_CELLS || self.nodes.len() > MAX_ARENA_REUSE {
            self.nodes.clear();
            self.nodes.push(ArenaNode::new(next_player));
            self.root_idx = 0;
            self.player = next_player;
            return;
        }
        let r_idx = self.root_idx as usize;
        if r_idx < self.nodes.len() {
            let root = &self.nodes[r_idx];
            if let Some(pos) = root.actions.iter().position(|&a| a as usize == mv) {
                let c_idx = root.children[pos];
                if c_idx != u32::MAX && (c_idx as usize) < self.nodes.len() {
                    if self.nodes[c_idx as usize].player == next_player {
                        self.root_idx = c_idx;
                        self.player = next_player;
                        return;
                    }
                }
            }
        }
        self.nodes.clear();
        self.nodes.push(ArenaNode::new(next_player));
        self.root_idx = 0;
        self.player = next_player;
    }
}

pub struct BatchMctsEngine {
    games: Vec<GameSlot>,
    in_flight: Vec<InFlightLeaf>,
}

impl BatchMctsEngine {
    fn new(max_games: usize) -> Self {
        let games = (0..max_games).map(|_| GameSlot::new()).collect();
        Self {
            games,
            in_flight: Vec::with_capacity(64),
        }
    }
}

fn expand_arena_node(
    node: &mut ArenaNode,
    board: &mut Board,
    logits: &[f32],
    val: f32,
    max_children: usize,
    use_r3: bool,
) -> f32 {
    let player = node.player;
    match scan_tactical_actions(board, player) {
        TacticalScan::WinMove(w_mv) => {
            node.actions = vec![w_mv as u16];
            node.priors = vec![1.0];
            node.n = vec![0.0];
            node.w = vec![0.0];
            node.children = vec![u32::MAX];
            node.n_sum = 0.0;
            node.win_move = w_mv as i32;
            node.expanded = true;
            return 1.0;
        }
        TacticalScan::TerminalLoss => {
            node.terminal = true;
            node.terminal_val = -1.0;
            node.expanded = true;
            return -1.0;
        }
        TacticalScan::ForcedBlocks(blocks) => {
            let mut ranked: Vec<(u16, f32)> = blocks.into_iter().map(|i| (i as u16, logits[i])).collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            let k = ranked.len();
            let max_l = ranked[0].1;
            let mut exps = Vec::with_capacity(k);
            let mut sum = 0.0f32;
            for &(_, l) in &ranked {
                let e = (l - max_l).clamp(-25.0, 0.0).exp();
                exps.push(e);
                sum += e;
            }
            let inv = 1.0 / sum.max(1e-9);
            node.actions = ranked.iter().map(|&(a, _)| a).collect();
            node.priors = exps.into_iter().map(|e| e * inv).collect();
            node.n = vec![0.0; k];
            node.w = vec![0.0; k];
            node.children = vec![u32::MAX; k];
            node.n_sum = 0.0;
            node.expanded = true;
            return val.clamp(-1.0, 1.0);
        }
        TacticalScan::Normal => {}
    }

    let tbl = tables();
    let nb_table = if use_r3 { &tbl.nb_r3 } else { &tbl.nb_r2 };
    let local_cands = local_empty_indices(&board.cells, nb_table);

    let mut filtered: Vec<(u16, f32)> = Vec::with_capacity(local_cands.len());
    if player == BLACK {
        let mut ranked: Vec<(usize, f32)> = local_cands.into_iter().map(|i| (i, logits[i])).collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        for &(idx, l) in &ranked {
            if board.is_legal_move(idx, BLACK, RuleSet::Renju) {
                filtered.push((idx as u16, l));
                if filtered.len() >= max_children {
                    break;
                }
            }
        }
        if filtered.is_empty() {
            let full_mask = exact_legal_mask_fast(board, BLACK);
            for (idx, &ok) in full_mask.iter().enumerate() {
                if ok {
                    filtered.push((idx as u16, logits[idx]));
                }
            }
            filtered.sort_by(|a, b| b.1.total_cmp(&a.1));
            filtered.truncate(max_children);
        }
    } else {
        for i in local_cands {
            filtered.push((i as u16, logits[i]));
        }
        filtered.sort_by(|a, b| b.1.total_cmp(&a.1));
        filtered.truncate(max_children);
    }

    if filtered.is_empty() {
        node.terminal = true;
        node.terminal_val = -1.0;
        node.expanded = true;
        return -1.0;
    }

    let k = filtered.len();
    let max_l = filtered[0].1;
    let mut exps = Vec::with_capacity(k);
    let mut sum = 0.0f32;
    for &(_, l) in &filtered {
        let e = (l - max_l).clamp(-25.0, 0.0).exp();
        exps.push(e);
        sum += e;
    }
    let inv = 1.0 / sum.max(1e-9);

    node.actions = filtered.iter().map(|&(a, _)| a).collect();
    node.priors = exps.into_iter().map(|e| e * inv).collect();
    node.n = vec![0.0; k];
    node.w = vec![0.0; k];
    node.children = vec![u32::MAX; k];
    node.n_sum = 0.0;
    node.expanded = true;
    val.clamp(-1.0, 1.0)
}

fn backprop_path(nodes: &mut [ArenaNode], path: &[(u32, usize)], leaf_val: f32, revert_vloss: bool) {
    let mut v = leaf_val;
    for &(n_idx, a_idx) in path.iter().rev() {
        let nd = &mut nodes[n_idx as usize];
        if revert_vloss {
            nd.n[a_idx] = (nd.n[a_idx] - 1.0).max(0.0);
            nd.w[a_idx] += 1.0;
            nd.n_sum = (nd.n_sum - 1.0).max(0.0);
        }
        v = -v;
        nd.n[a_idx] += 1.0;
        nd.w[a_idx] += v;
        nd.n_sum += 1.0;
    }
}

fn revert_vloss_only(nodes: &mut [ArenaNode], path: &[(u32, usize)]) {
    for &(n_idx, a_idx) in path {
        let nd = &mut nodes[n_idx as usize];
        nd.n[a_idx] = (nd.n[a_idx] - 1.0).max(0.0);
        nd.w[a_idx] += 1.0;
        nd.n_sum = (nd.n_sum - 1.0).max(0.0);
    }
}

// ── C-ABI Exports ──

#[no_mangle]
pub extern "C" fn rs_mcts_create(max_games: i32) -> *mut BatchMctsEngine {
    let cap = max_games.max(1).min(64) as usize;
    Box::into_raw(Box::new(BatchMctsEngine::new(cap)))
}

#[no_mangle]
pub unsafe extern "C" fn rs_mcts_free(engine: *mut BatchMctsEngine) {
    if !engine.is_null() {
        drop(Box::from_raw(engine));
    }
}

#[no_mangle]
pub unsafe extern "C" fn rs_mcts_reset_game(engine: *mut BatchMctsEngine, game_idx: i32, player: u8) {
    let eng = &mut *engine;
    if let Some(slot) = eng.games.get_mut(game_idx as usize) {
        slot.reset(player);
    }
}

#[no_mangle]
pub unsafe extern "C" fn rs_mcts_set_board(
    engine: *mut BatchMctsEngine,
    game_idx: i32,
    cells_ptr: *const u8,
    last_move: i32,
    player: u8,
) {
    let eng = &mut *engine;
    let cells_slice = slice::from_raw_parts(cells_ptr, NUM_CELLS);
    if let Some(slot) = eng.games.get_mut(game_idx as usize) {
        let mut arr = [EMPTY; NUM_CELLS];
        arr.copy_from_slice(cells_slice);
        slot.set_board(&arr, last_move, player);
    }
}

#[no_mangle]
pub unsafe extern "C" fn rs_mcts_place_and_advance(
    engine: *mut BatchMctsEngine,
    game_idx: i32,
    mv: i32,
    mover: u8,
    next_player: u8,
) {
    let eng = &mut *engine;
    if let Some(slot) = eng.games.get_mut(game_idx as usize) {
        if mv >= 0 && (mv as usize) < NUM_CELLS {
            let m = mv as usize;
            slot.board.cells[m] = mover;
            slot.last_move = mv;
            slot.advance_root(m, next_player);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn rs_mcts_prepare_roots(
    engine: *mut BatchMctsEngine,
    game_ids_ptr: *const i32,
    n_games: i32,
    out_game_ids_ptr: *mut i32,
    out_nhwc_ptr: *mut f32,
) -> i32 {
    let eng = &mut *engine;
    let ids = slice::from_raw_parts(game_ids_ptr, n_games.max(0) as usize);
    let out_ids = slice::from_raw_parts_mut(out_game_ids_ptr, n_games.max(0) as usize);
    let out_nhwc = slice::from_raw_parts_mut(out_nhwc_ptr, (n_games.max(0) as usize) * NUM_CELLS * 6);

    let mut count = 0usize;
    for &gid_i32 in ids {
        let gid = gid_i32 as usize;
        if gid >= eng.games.len() {
            continue;
        }
        let slot = &mut eng.games[gid];
        let r_idx = slot.root_idx as usize;
        if !slot.nodes[r_idx].expanded {
            out_ids[count] = gid_i32;
            let row_slice = &mut out_nhwc[count * NUM_CELLS * 6..(count + 1) * NUM_CELLS * 6];
            let lm = slot.last_move;
            let pl = slot.nodes[r_idx].player;
            encode_nhwc_one(&mut slot.board, lm, pl, None, row_slice);
            count += 1;
        }
    }
    count as i32
}

#[no_mangle]
pub unsafe extern "C" fn rs_mcts_expand_roots(
    engine: *mut BatchMctsEngine,
    eval_ids_ptr: *const i32,
    n_eval: i32,
    logits_ptr: *const f32,
    vals_ptr: *const f32,
    add_noise: u8,
    seed: u64,
) {
    let eng = &mut *engine;
    let n = n_eval.max(0) as usize;
    let ids = slice::from_raw_parts(eval_ids_ptr, n);
    let logits = slice::from_raw_parts(logits_ptr, n * NUM_CELLS);
    let vals = slice::from_raw_parts(vals_ptr, n);

    for (i, &gid_i32) in ids.iter().enumerate() {
        let gid = gid_i32 as usize;
        if gid >= eng.games.len() {
            continue;
        }
        let slot = &mut eng.games[gid];
        let r_idx = slot.root_idx as usize;
        let row_logits = &logits[i * NUM_CELLS..(i + 1) * NUM_CELLS];
        expand_arena_node(
            &mut slot.nodes[r_idx],
            &mut slot.board,
            row_logits,
            vals[i],
            36,
            true,
        );
    }

    if add_noise != 0 {
        // Note: Noise applied cleanly across ALL passed root slots, not just newly expanded ones,
        // wait: callers can also call rs_mcts_apply_root_noise separately. Do nothing extra here.
        let _ = seed;
    }
}

#[no_mangle]
pub unsafe extern "C" fn rs_mcts_apply_root_noise(
    engine: *mut BatchMctsEngine,
    game_ids_ptr: *const i32,
    n_games: i32,
    seed: u64,
) {
    let eng = &mut *engine;
    let ids = slice::from_raw_parts(game_ids_ptr, n_games.max(0) as usize);
    let mut rng = StdRng::seed_from_u64(seed);

    for &gid_i32 in ids {
        let gid = gid_i32 as usize;
        if gid >= eng.games.len() {
            continue;
        }
        let slot = &mut eng.games[gid];
        let root = &slot.nodes[slot.root_idx as usize];
        let k = root.priors.len();
        if k > 1 && root.win_move < 0 && !root.terminal {
            let alpha = (10.0f32 / (k as f32)).clamp(0.15, 0.5);
            let mut noise = vec![0.0f32; k];
            let mut sum = 0.0f32;
            // Marsaglia-Tsang for Gamma(alpha < 1) = Gamma(alpha + 1) * U^(1/alpha)
            let ap1 = alpha + 1.0;
            let d = ap1 - 1.0 / 3.0;
            let c = 1.0 / (9.0 * d).sqrt();
            for item in &mut noise {
                let g1 = loop {
                    let u1 = rng.gen::<f32>().max(1e-7);
                    let u2 = rng.gen::<f32>();
                    let z = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
                    let v = (1.0 + c * z).powi(3);
                    if v > 0.0 {
                        let u = rng.gen::<f32>().max(1e-7);
                        if u < 1.0 - 0.0331 * z.powi(4) || u.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) {
                            break d * v;
                        }
                    }
                };
                let u_pow = rng.gen::<f32>().max(1e-7).powf(1.0 / alpha);
                let g_val = g1 * u_pow;
                *item = g_val;
                sum += g_val;
            }
            slot.root_noise.clear();
            let inv = 1.0 / sum.max(1e-9);
            for idx in 0..k {
                slot.root_noise.push(0.75 * root.priors[idx] + 0.25 * noise[idx] * inv);
            }
        } else {
            slot.root_noise.clear();
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn rs_mcts_select_leaves(
    engine: *mut BatchMctsEngine,
    game_ids_ptr: *const i32,
    n_games: i32,
    leaves_per_game: i32,
    max_batch: i32,
    c_puct: f32,
    out_nhwc_ptr: *mut f32,
) -> i32 {
    let eng = &mut *engine;
    eng.in_flight.clear();
    let ids = slice::from_raw_parts(game_ids_ptr, n_games.max(0) as usize);
    let max_b = max_batch.max(1) as usize;
    let lpg = leaves_per_game.max(1) as usize;
    let out_nhwc = slice::from_raw_parts_mut(out_nhwc_ptr, max_b * NUM_CELLS * 6);

    let mut queued = 0usize;
    let mut queued_leaves: Vec<(usize, u32)> = Vec::with_capacity(max_b);

    for _leaf_round in 0..lpg {
        if queued >= max_b {
            break;
        }
        for &gid_i32 in ids {
            if queued >= max_b {
                break;
            }
            let gid = gid_i32 as usize;
            if gid >= eng.games.len() {
                continue;
            }
            let slot = &mut eng.games[gid];
            let r_idx = slot.root_idx;
            let r_ref = &slot.nodes[r_idx as usize];
            if r_ref.terminal || r_ref.win_move >= 0 {
                continue;
            }
            if r_ref.expanded && r_ref.actions.len() <= 1 && r_ref.n_sum >= 1.0 {
                continue;
            }
            if r_ref.expanded && r_ref.actions.len() >= 2 && r_ref.n_sum >= 12.0 {
                let mut max1 = 0.0f32;
                let mut max2 = 0.0f32;
                for &ni in &r_ref.n {
                    if ni > max1 {
                        max2 = max1;
                        max1 = ni;
                    } else if ni > max2 {
                        max2 = ni;
                    }
                }
                if max1 - max2 > 14.0 && max1 >= 0.68 * r_ref.n_sum {
                    continue;
                }
            }

            let mut scratch_board = slot.board.clone();
            let mut cur_last_mv = slot.last_move;
            let mut node_idx = r_idx;
            let mut path: Vec<(u32, usize)> = Vec::with_capacity(16);

            loop {
                let nd = &slot.nodes[node_idx as usize];
                if !nd.expanded || nd.terminal || nd.actions.is_empty() {
                    break;
                }
                let k = nd.actions.len();
                let sqrt_tot = nd.n_sum.max(1.0).sqrt();
                let use_noise = node_idx == r_idx && slot.root_noise.len() == k;

                let fpu = if node_idx == r_idx { -0.05 } else { -0.25 };
                let mut best_a = 0usize;
                let mut best_score = f32::NEG_INFINITY;
                for a_i in 0..k {
                    let ni = nd.n[a_i];
                    let q = if ni > 0.0 { nd.w[a_i] / ni } else { fpu };
                    let p = if use_noise { slot.root_noise[a_i] } else { nd.priors[a_i] };
                    let u = c_puct * p * (sqrt_tot / (1.0 + ni));
                    let s = q + u;
                    if s > best_score {
                        best_score = s;
                        best_a = a_i;
                    }
                }

                let mv = nd.actions[best_a] as usize;
                let mover = nd.player;
                scratch_board.cells[mv] = mover;
                cur_last_mv = mv as i32;

                // Apply virtual loss on edge
                let nd_mut = &mut slot.nodes[node_idx as usize];
                nd_mut.n[best_a] += 1.0;
                nd_mut.w[best_a] -= 1.0;
                nd_mut.n_sum += 1.0;
                path.push((node_idx, best_a));

                let mut child_idx = slot.nodes[node_idx as usize].children[best_a];
                if child_idx == u32::MAX {
                    let (mr, mc) = Board::rc(mv);
                    let opp = Board::opponent(mover);
                    let mut child = ArenaNode::new(opp);
                    if scratch_board.is_win_at(mr, mc, mover, RuleSet::Renju) {
                        child.terminal = true;
                        child.terminal_val = -1.0;
                        child.expanded = true;
                    }
                    child_idx = slot.nodes.len() as u32;
                    slot.nodes.push(child);
                    slot.nodes[node_idx as usize].children[best_a] = child_idx;
                }
                node_idx = child_idx;
            }

            let leaf_nd = &slot.nodes[node_idx as usize];
            if leaf_nd.terminal {
                let tv = leaf_nd.terminal_val;
                backprop_path(&mut slot.nodes, &path, tv, true);
                continue;
            }
            if leaf_nd.expanded {
                backprop_path(&mut slot.nodes, &path, 0.0, true);
                continue;
            }
            if queued_leaves.contains(&(gid, node_idx)) {
                revert_vloss_only(&mut slot.nodes, &path);
                continue;
            }

            let row_slice = &mut out_nhwc[queued * NUM_CELLS * 6..(queued + 1) * NUM_CELLS * 6];
            encode_nhwc_one(&mut scratch_board, cur_last_mv, leaf_nd.player, None, row_slice);

            queued_leaves.push((gid, node_idx));
            eng.in_flight.push(InFlightLeaf {
                game_idx: gid,
                leaf_idx: node_idx,
                cells: scratch_board.cells,
                path,
            });
            queued += 1;
        }
    }

    queued as i32
}

#[no_mangle]
pub unsafe extern "C" fn rs_mcts_expand_backprop(
    engine: *mut BatchMctsEngine,
    logits_ptr: *const f32,
    vals_ptr: *const f32,
) {
    let eng = &mut *engine;
    let n = eng.in_flight.len();
    if n == 0 {
        return;
    }
    let logits = slice::from_raw_parts(logits_ptr, n * NUM_CELLS);
    let vals = slice::from_raw_parts(vals_ptr, n);

    for (i, item) in eng.in_flight.drain(..).enumerate() {
        let slot = &mut eng.games[item.game_idx];
        let mut leaf_board = Board { cells: item.cells };
        let l_idx = item.leaf_idx as usize;
        let row_logits = &logits[i * NUM_CELLS..(i + 1) * NUM_CELLS];

        // Temporarily swap out leaf_node? Not needed: borrow field directly
        let mut node_tmp = ArenaNode::new(slot.nodes[l_idx].player);
        let true_val = expand_arena_node(
            &mut node_tmp,
            &mut leaf_board,
            row_logits,
            vals[i],
            24,
            false,
        );
        slot.nodes[l_idx] = node_tmp;
        backprop_path(&mut slot.nodes, &item.path, true_val, true);
    }
}

#[no_mangle]
pub unsafe extern "C" fn rs_mcts_finalize_move(
    engine: *mut BatchMctsEngine,
    game_idx: i32,
    use_temp: u8,
    seed: u64,
    out_pi_ptr: *mut f32,
    out_root_q_ptr: *mut f32,
    out_is_forced_win_ptr: *mut u8,
) -> i32 {
    let eng = &*engine;
    let out_pi = slice::from_raw_parts_mut(out_pi_ptr, NUM_CELLS);
    out_pi.fill(0.0);

    let Some(slot) = eng.games.get(game_idx as usize) else {
        *out_root_q_ptr = 0.0;
        *out_is_forced_win_ptr = 0;
        return -1;
    };
    let root = &slot.nodes[slot.root_idx as usize];

    if root.win_move >= 0 {
        let wm = root.win_move as usize;
        if wm < NUM_CELLS {
            out_pi[wm] = 1.0;
        }
        *out_root_q_ptr = 1.0;
        *out_is_forced_win_ptr = 1;
        return root.win_move;
    }

    *out_is_forced_win_ptr = 0;
    let k = root.actions.len();
    if k == 0 {
        *out_root_q_ptr = -1.0;
        return -1;
    }

    let v_sum: f32 = root.n.iter().sum();
    if v_sum > 0.0 {
        let inv = 1.0 / v_sum;
        let mut w_sum = 0.0f32;
        for i in 0..k {
            out_pi[root.actions[i] as usize] = root.n[i] * inv;
            w_sum += root.w[i];
        }
        *out_root_q_ptr = (w_sum * inv).clamp(-1.0, 1.0);

        if use_temp != 0 {
            let mut rng = StdRng::seed_from_u64(seed);
            let weights: Vec<f64> = root.n.iter().map(|&v| v.max(0.0) as f64).collect();
            if let Ok(dist) = WeightedIndex::new(&weights) {
                return root.actions[dist.sample(&mut rng)] as i32;
            }
        }
        let mut best_i = 0usize;
        let mut best_v = -1.0f32;
        for i in 0..k {
            if root.n[i] > best_v {
                best_v = root.n[i];
                best_i = i;
            }
        }
        root.actions[best_i] as i32
    } else {
        for i in 0..k {
            out_pi[root.actions[i] as usize] = root.priors[i];
        }
        *out_root_q_ptr = 0.0;
        let mut best_i = 0usize;
        let mut best_p = -1.0f32;
        for i in 0..k {
            if root.priors[i] > best_p {
                best_p = root.priors[i];
                best_i = i;
            }
        }
        root.actions[best_i] as i32
    }
}

#[no_mangle]
pub unsafe extern "C" fn rs_exact_legal_mask(
    cells_ptr: *const u8,
    player: u8,
    out_mask_ptr: *mut u8,
) {
    let cells_slice = slice::from_raw_parts(cells_ptr, NUM_CELLS);
    let out_mask = slice::from_raw_parts_mut(out_mask_ptr, NUM_CELLS);
    let mut board = Board::new();
    board.cells.copy_from_slice(cells_slice);
    let mask = exact_legal_mask_fast(&mut board, player);
    for i in 0..NUM_CELLS {
        out_mask[i] = u8::from(mask[i]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn rs_encode_board_nhwc(
    cells_ptr: *const u8,
    last_move: i32,
    player: u8,
    out_nhwc_ptr: *mut f32,
) {
    let cells_slice = slice::from_raw_parts(cells_ptr, NUM_CELLS);
    let out_nhwc = slice::from_raw_parts_mut(out_nhwc_ptr, NUM_CELLS * 6);
    let mut board = Board::new();
    board.cells.copy_from_slice(cells_slice);
    encode_nhwc_one(&mut board, last_move, player, None, out_nhwc);
}

#[no_mangle]
pub unsafe extern "C" fn rs_rule_based_move(
    cells_ptr: *const u8,
    player: u8,
    noise_eps: f32,
    seed: u64,
) -> i32 {
    let cells_slice = slice::from_raw_parts(cells_ptr, NUM_CELLS);
    let mut board = Board::new();
    board.cells.copy_from_slice(cells_slice);
    rule_based_move_rust(&mut board, player, noise_eps, seed)
        .map(|i| i as i32)
        .unwrap_or(-1)
}

#[no_mangle]
pub unsafe extern "C" fn rs_sample_d4_batch(
    buf_s_ptr: *const f32,
    buf_pi_ptr: *const f32,
    buf_z_ptr: *const f32,
    n_buf: i32,
    indices_ptr: *const i32,
    syms_ptr: *const u8,
    batch_size: i32,
    out_x_ptr: *mut f32,
    out_pi_ptr: *mut f32,
    out_z_ptr: *mut f32,
) {
    let tbl = tables();
    let n = n_buf.max(1) as usize;
    let b = batch_size.max(0) as usize;
    let buf_s = slice::from_raw_parts(buf_s_ptr, n * NUM_CELLS * 6);
    let buf_pi = slice::from_raw_parts(buf_pi_ptr, n * NUM_CELLS);
    let buf_z = slice::from_raw_parts(buf_z_ptr, n);
    let indices = slice::from_raw_parts(indices_ptr, b);
    let syms = slice::from_raw_parts(syms_ptr, b);

    let out_x = slice::from_raw_parts_mut(out_x_ptr, b * NUM_CELLS * 6);
    let out_pi = slice::from_raw_parts_mut(out_pi_ptr, b * NUM_CELLS);
    let out_z = slice::from_raw_parts_mut(out_z_ptr, b);

    for bi in 0..b {
        let src_i = (indices[bi] as usize) % n;
        let sym = (syms[bi] & 7) as usize;
        out_z[bi] = buf_z[src_i];

        let s_off = src_i * NUM_CELLS * 6;
        let p_off = src_i * NUM_CELLS;
        let x_dst = bi * NUM_CELLS * 6;
        let p_dst = bi * NUM_CELLS;

        if sym == 0 {
            out_x[x_dst..x_dst + NUM_CELLS * 6].copy_from_slice(&buf_s[s_off..s_off + NUM_CELLS * 6]);
            out_pi[p_dst..p_dst + NUM_CELLS].copy_from_slice(&buf_pi[p_off..p_off + NUM_CELLS]);
        } else {
            let perm = &tbl.d4_perm[sym];
            for dst in 0..NUM_CELLS {
                let src = perm[dst] as usize;
                out_pi[p_dst + dst] = buf_pi[p_off + src];
                let d6 = x_dst + dst * 6;
                let s6 = s_off + src * 6;
                out_x[d6..d6 + 6].copy_from_slice(&buf_s[s6..s6 + 6]);
            }
        }
    }
}

#[cfg(test)]
mod lib_tests {
    use super::*;
    use renju::WHITE;

    #[test]
    fn test_rust_mcts_tactical_and_legal() {
        let mut board = Board::new();
        for c in 4..8 {
            board.cells[Board::idx(7, c)] = BLACK;
        }
        // White has urgent block at (7,3) or (7,8) since Black has open four!
        match scan_tactical_actions(&mut board, WHITE) {
            TacticalScan::ForcedBlocks(v) => {
                assert!(v.contains(&Board::idx(7, 3)));
                assert!(v.contains(&Board::idx(7, 8)));
                assert_eq!(v.len(), 2);
            }
            _ => panic!("Expected forced blocks for White against Black open 4"),
        }

        // Black immediate win overrides 3-3 or overline? Exact-5 wins!
        let mut b5 = Board::new();
        for c in 3..7 {
            b5.cells[Board::idx(7, c)] = BLACK;
        }
        match scan_tactical_actions(&mut b5, BLACK) {
            TacticalScan::WinMove(m) => assert!(m == Board::idx(7, 2) || m == Board::idx(7, 7)),
            _ => panic!("Expected immediate win move for Black"),
        }

        // Exact legal mask respects White having no 3-3 restriction
        let mut b33 = Board::new();
        b33.cells[Board::idx(7, 5)] = WHITE;
        b33.cells[Board::idx(7, 6)] = WHITE;
        b33.cells[Board::idx(5, 7)] = WHITE;
        b33.cells[Board::idx(6, 7)] = WHITE;
        let w_mask = exact_legal_mask_fast(&mut b33, WHITE);
        assert!(w_mask[Board::idx(7, 7)]);

        // D4 permutation inv-consistency check on sample_d4_batch
        let tbl = tables();
        for sym in 0..8 {
            let mut seen = [false; NUM_CELLS];
            for &src in &tbl.d4_perm[sym] {
                assert!(!seen[src as usize]);
                seen[src as usize] = true;
            }
        }
    }
}
