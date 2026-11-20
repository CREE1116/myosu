use crate::nn::PolicyValueNet;
use crate::renju::{Board, EMPTY, NUM_CELLS, RuleSet};
use rand::distributions::WeightedIndex;
use rand::prelude::*;

struct MctsNode {
    move_idx: usize,
    prior: f32,
    visits: u32,
    value_sum: f32,
    terminal: Option<f32>, // Value from the perspective of the player who played `move_idx`
    children: Vec<usize>,
}

impl MctsNode {
    fn new(move_idx: usize, prior: f32) -> Self {
        Self {
            move_idx,
            prior,
            visits: 0,
            value_sum: 0.0,
            terminal: None,
            children: Vec::new(),
        }
    }

    #[inline]
    fn q(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

fn masked_softmax(logits: &[f32; NUM_CELLS], mask: &[bool; NUM_CELLS]) -> [f32; NUM_CELLS] {
    let mut max_l = f32::NEG_INFINITY;
    for i in 0..NUM_CELLS {
        if mask[i] && logits[i] > max_l {
            max_l = logits[i];
        }
    }
    let mut probs = [0.0f32; NUM_CELLS];
    let mut sum = 0.0f32;
    for i in 0..NUM_CELLS {
        if mask[i] {
            let e = (logits[i] - max_l).exp();
            probs[i] = e;
            sum += e;
        }
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for i in 0..NUM_CELLS {
            probs[i] *= inv;
        }
    }
    probs
}

/// Generates simple Gamma-distributed Dirichlet noise without external `rand_distr` crate.
fn add_dirichlet_noise(priors: &mut [f32], legal_indices: &[usize], alpha: f32, eps: f32, rng: &mut impl Rng) {
    if legal_indices.is_empty() {
        return;
    }
    // Marsaglia-Tsang or simple inverse-log approximation for alpha < 1 (alpha = 0.3)
    let mut noise = Vec::with_capacity(legal_indices.len());
    let mut sum = 0.0f32;
    for _ in 0..legal_indices.len() {
        // Ahrens-Dieter / rejection for Gamma(alpha < 1) via Gamma(1+alpha) * U^(1/alpha)
        let d = (1.0 + alpha) - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        let g1 = loop {
            let u1: f32 = rng.gen::<f32>().max(1e-7);
            let u2: f32 = rng.gen::<f32>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            let v = (1.0 + c * z).powi(3);
            if v > 0.0 {
                let u: f32 = rng.gen::<f32>().max(1e-7);
                if u.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) {
                    break d * v;
                }
            }
        };
        let u_pow: f32 = rng.gen::<f32>().max(1e-7).powf(1.0 / alpha);
        let sample = g1 * u_pow;
        noise.push(sample);
        sum += sample;
    }

    if sum > 0.0 {
        for (k, &idx) in legal_indices.iter().enumerate() {
            let n = noise[k] / sum;
            priors[idx] = (1.0 - eps) * priors[idx] + eps * n;
        }
    }
}

pub struct MctsResult {
    pub policy: [f32; NUM_CELLS],
    pub chosen_move: usize,
    pub root_value: f32,
}

pub fn run_mcts(
    root_board: &Board,
    root_player: u8,
    rules: RuleSet,
    net: &PolicyValueNet,
    sims: usize,
    temperature: f32,
    dirichlet_noise: bool,
    rng: &mut impl Rng,
) -> Option<MctsResult> {
    let mut work_board = root_board.clone();
    let root_mask = work_board.legal_mask(root_player, rules);
    let legal_indices: Vec<usize> = (0..NUM_CELLS).filter(|&i| root_mask[i]).collect();
    if legal_indices.is_empty() {
        return None;
    }

    // Immediate 5-in-a-row win shortcut for clean terminal detection
    for &idx in &legal_indices {
        work_board.cells[idx] = root_player;
        let (r, c) = Board::rc(idx);
        let win = work_board.is_win_at(r, c, root_player, rules);
        work_board.cells[idx] = EMPTY;
        if win {
            let mut policy = [0.0f32; NUM_CELLS];
            policy[idx] = 1.0;
            return Some(MctsResult {
                policy,
                chosen_move: idx,
                root_value: 1.0,
            });
        }
    }

    let c_puct = 1.5f32;
    let mut nodes: Vec<MctsNode> = Vec::with_capacity(sims * 4);
    nodes.push(MctsNode::new(usize::MAX, 1.0));

    // Expand root
    let root_tensor = work_board.encode_tensor_with_mask(root_player, &root_mask);
    let (logits, _) = net.forward(&root_tensor);
    let mut priors = masked_softmax(&logits, &root_mask);
    if dirichlet_noise {
        add_dirichlet_noise(&mut priors, &legal_indices, 0.3, 0.25, rng);
    }

    for &idx in &legal_indices {
        let child_id = nodes.len();
        nodes.push(MctsNode::new(idx, priors[idx]));
        nodes[0].children.push(child_id);
    }
    nodes[0].visits = 1;

    let root_move_count = root_board.cells.iter().filter(|&&v| v != EMPTY).count();

    for _ in 0..sims {
        let mut path: Vec<usize> = vec![0];
        let mut cur = 0usize;
        let mut cur_board = root_board.clone();
        let mut cur_player = root_player;
        let mut move_count = root_move_count;

        // Selection
        while !nodes[cur].children.is_empty() {
            let parent_sqrt = (nodes[cur].visits as f32).sqrt();
            let mut best_child = nodes[cur].children[0];
            let mut best_score = f32::NEG_INFINITY;

            for &cid in &nodes[cur].children {
                let child = &nodes[cid];
                let u = c_puct * child.prior * parent_sqrt / (1.0 + child.visits as f32);
                let score = child.q() + u;
                if score > best_score {
                    best_score = score;
                    best_child = cid;
                }
            }

            let mv = nodes[best_child].move_idx;
            cur_board.cells[mv] = cur_player;
            move_count += 1;
            path.push(best_child);
            cur = best_child;
            cur_player = Board::opponent(cur_player);
        }

        // Evaluate / Expand leaf `cur`
        let leaf_value_from_mover = if let Some(tv) = nodes[cur].terminal {
            tv
        } else {
            let prev_player = Board::opponent(cur_player);
            let mv = nodes[cur].move_idx;
            let (mr, mc) = Board::rc(mv);
            if cur_board.is_win_at(mr, mc, prev_player, rules) {
                nodes[cur].terminal = Some(1.0);
                1.0
            } else if move_count >= NUM_CELLS {
                nodes[cur].terminal = Some(0.0);
                0.0
            } else {
                let mask = cur_board.legal_mask(cur_player, rules);
                let legal: Vec<usize> = (0..NUM_CELLS).filter(|&i| mask[i]).collect();
                if legal.is_empty() {
                    nodes[cur].terminal = Some(0.0);
                    0.0
                } else {
                    let tensor = cur_board.encode_tensor_with_mask(cur_player, &mask);
                    let (logits, val_cur) = net.forward(&tensor);
                    let leaf_priors = masked_softmax(&logits, &mask);
                    let mut child_ids = Vec::with_capacity(legal.len());
                    for &idx in &legal {
                        let cid = nodes.len();
                        nodes.push(MctsNode::new(idx, leaf_priors[idx]));
                        child_ids.push(cid);
                    }
                    nodes[cur].children = child_ids;
                    // `val_cur` is from `cur_player`'s perspective; parent edge was played by `prev_player`
                    -val_cur
                }
            }
        };

        // Backpropagate alternating sign along path from leaf up to root
        let mut v = leaf_value_from_mover;
        for &nid in path.iter().rev() {
            nodes[nid].visits += 1;
            nodes[nid].value_sum += v;
            v = -v;
        }
    }

    let mut policy = [0.0f32; NUM_CELLS];
    let mut total_visits = 0.0f32;
    for &cid in &nodes[0].children {
        let vis = nodes[cid].visits as f32;
        policy[nodes[cid].move_idx] = vis;
        total_visits += vis;
    }
    if total_visits > 0.0 {
        for p in &mut policy {
            *p /= total_visits;
        }
    }

    let chosen_move = if temperature <= 0.05 {
        *nodes[0]
            .children
            .iter()
            .max_by_key(|&&cid| nodes[cid].visits)
            .map(|&cid| &nodes[cid].move_idx)
            .unwrap()
    } else {
        let inv_t = 1.0 / temperature;
        let weights: Vec<f32> = nodes[0]
            .children
            .iter()
            .map(|&cid| (nodes[cid].visits as f32).powf(inv_t).max(1e-6))
            .collect();
        let dist = WeightedIndex::new(&weights).unwrap();
        let pick = dist.sample(rng);
        nodes[nodes[0].children[pick]].move_idx
    };

    Some(MctsResult {
        policy,
        chosen_move,
        root_value: -nodes[0].q(),
    })
}
