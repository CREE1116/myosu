mod mcts;
mod nn;
mod renju;

use crate::mcts::run_mcts;
use crate::nn::{AdamW, PolicyValueNet};
use crate::renju::{augment_dihedral, Board, RuleSet, BLACK, NUM_CELLS, WHITE};
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::time::Instant;

#[derive(Clone)]
struct TrainSample {
    state: [f32; 4 * NUM_CELLS],
    policy: [f32; NUM_CELLS],
    value: f32,
}

#[derive(Clone, Serialize, Deserialize)]
struct PolicyCandidate {
    row: usize,
    col: usize,
    prob: f32,
}

#[derive(Clone, Serialize, Deserialize)]
struct ReplayMove {
    row: usize,
    col: usize,
    player: u8,
    value: f32,
    #[serde(rename = "topPolicies")]
    top_policies: Vec<PolicyCandidate>,
}

#[derive(Clone, Serialize, Deserialize)]
struct GenMetric {
    generation: usize,
    #[serde(rename = "pLoss")]
    p_loss: f32,
    #[serde(rename = "vLoss")]
    v_loss: f32,
    #[serde(rename = "avgMoves")]
    avg_moves: f32,
    #[serde(rename = "blackWins")]
    black_wins: usize,
    #[serde(rename = "whiteWins")]
    white_wins: usize,
    draws: usize,
    #[serde(rename = "spSec")]
    sp_sec: f32,
    #[serde(rename = "trainSec")]
    train_sec: f32,
    #[serde(rename = "bufferSize")]
    buffer_size: usize,
}

#[derive(Clone, Serialize, Deserialize)]
struct ModelInfo {
    channels: usize,
    blocks: usize,
    params: usize,
}

#[derive(Clone, Serialize, Deserialize)]
struct ReplayGame {
    id: String,
    label: String,
    generation: usize,
    winner: u8, // 0=draw, 1=black, 2=white
    moves: Vec<ReplayMove>,
}

#[derive(Serialize, Deserialize)]
struct ReplayArchive {
    rules: String,
    #[serde(rename = "modelInfo")]
    model_info: ModelInfo,
    metrics: Vec<GenMetric>,
    games: Vec<ReplayGame>,
}

struct SelfPlayGameResult {
    samples: Vec<TrainSample>,
    winner: u8,
    moves: Vec<ReplayMove>,
}

fn play_self_play_game(
    net: &PolicyValueNet,
    rules: RuleSet,
    sims: usize,
    seed: u64,
    greedy: bool,
) -> SelfPlayGameResult {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut board = Board::new();
    let mut player = BLACK;
    let mut trajectory: Vec<([f32; 4 * NUM_CELLS], [f32; NUM_CELLS], u8)> = Vec::new();
    let mut moves: Vec<ReplayMove> = Vec::new();
    let mut winner = 0u8;

    for step in 0..NUM_CELLS {
        let temp = if greedy {
            0.0
        } else if step < 12 {
            1.0
        } else {
            0.15
        };

        let Some(res) = run_mcts(
            &board,
            player,
            rules,
            net,
            sims,
            temp,
            !greedy,
            &mut rng,
        ) else {
            break;
        };

        let state = board.encode_tensor(player, rules);
        trajectory.push((state, res.policy, player));

        // Capture top-12 policy distribution for NN visualization
        let mut ranked: Vec<(usize, f32)> = res
            .policy
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.005)
            .map(|(i, &p)| (i, p))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(12);
        let top_policies = ranked
            .into_iter()
            .map(|(idx, prob)| {
                let (pr, pc) = Board::rc(idx);
                PolicyCandidate {
                    row: pr,
                    col: pc,
                    prob: (prob * 1000.0).round() / 1000.0,
                }
            })
            .collect();

        let (r, c) = Board::rc(res.chosen_move);
        board.cells[res.chosen_move] = player;
        moves.push(ReplayMove {
            row: r,
            col: c,
            player,
            value: (res.root_value * 1000.0).round() / 1000.0,
            top_policies,
        });

        if board.is_win_at(r, c, player, rules) {
            winner = player;
            break;
        }
        player = Board::opponent(player);
    }

    let mut samples = Vec::with_capacity(trajectory.len() * 8);
    for (state, policy, mover) in trajectory {
        let z = if winner == 0 {
            0.0
        } else if winner == mover {
            1.0
        } else {
            -1.0
        };
        for sym in 0..8 {
            let (s_aug, p_aug) = augment_dihedral(&state, &policy, sym);
            samples.push(TrainSample {
                state: s_aug,
                policy: p_aug,
                value: z,
            });
        }
    }

    SelfPlayGameResult {
        samples,
        winner,
        moves,
    }
}

fn format_game_label(gen: usize, winner: u8, num_moves: usize) -> String {
    let outcome = match winner {
        BLACK => "흑승",
        WHITE => "백승",
        _ => "무승부",
    };
    format!("Gen {gen} · {num_moves}수 {outcome}")
}

struct Config {
    iters: usize,
    games_per_iter: usize,
    sims: usize,
    channels: usize,
    blocks: usize,
    batch_size: usize,
    epochs: usize,
    lr: f32,
    out_path: String,
    checkpoint_path: String,
    resume: bool,
    rules: RuleSet,
}

fn parse_args() -> Config {
    let mut cfg = Config {
        iters: 8,
        games_per_iter: 12,
        sims: 48,
        channels: 32,
        blocks: 3,
        batch_size: 64,
        epochs: 2,
        lr: 1.5e-3,
        out_path: "../public/rl_replays.json".to_string(),
        checkpoint_path: "checkpoints/latest.json".to_string(),
        resume: false,
        rules: RuleSet::Renju,
    };

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "train" => {}
            "--iters" => {
                i += 1;
                cfg.iters = args[i].parse().expect("invalid --iters");
            }
            "--games" => {
                i += 1;
                cfg.games_per_iter = args[i].parse().expect("invalid --games");
            }
            "--sims" => {
                i += 1;
                cfg.sims = args[i].parse().expect("invalid --sims");
            }
            "--channels" => {
                i += 1;
                cfg.channels = args[i].parse().expect("invalid --channels");
            }
            "--blocks" => {
                i += 1;
                cfg.blocks = args[i].parse().expect("invalid --blocks");
            }
            "--batch" => {
                i += 1;
                cfg.batch_size = args[i].parse().expect("invalid --batch");
            }
            "--epochs" => {
                i += 1;
                cfg.epochs = args[i].parse().expect("invalid --epochs");
            }
            "--lr" => {
                i += 1;
                cfg.lr = args[i].parse().expect("invalid --lr");
            }
            "--out" => {
                i += 1;
                cfg.out_path = args[i].clone();
            }
            "--checkpoint" => {
                i += 1;
                cfg.checkpoint_path = args[i].clone();
            }
            "--resume" => {
                cfg.resume = true;
            }
            "--freestyle" => {
                cfg.rules = RuleSet::Freestyle;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: gomoku-rl train [--iters N] [--games N] [--sims N] [--channels N] [--blocks N] [--out PATH] [--resume] [--freestyle]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }
    cfg
}

fn main() {
    let cfg = parse_args();
    let mut rng = StdRng::seed_from_u64(20261119);

    let mut net = if cfg.resume && Path::new(&cfg.checkpoint_path).exists() {
        let json = fs::read_to_string(&cfg.checkpoint_path).expect("failed to read checkpoint");
        println!("Resuming checkpoint from {}", cfg.checkpoint_path);
        serde_json::from_str(&json).expect("invalid checkpoint json")
    } else {
        PolicyValueNet::new(cfg.channels, cfg.blocks, &mut rng)
    };

    println!(
        "Gomoku RL (Rust) | rules={:?} | params={} | iters={} games/iter={} sims={}",
        cfg.rules,
        net.num_params(),
        cfg.iters,
        cfg.games_per_iter,
        cfg.sims
    );

    let mut optimizer = AdamW::new(&net, cfg.lr, 1e-4);
    let mut replay_buffer: VecDeque<TrainSample> = VecDeque::with_capacity(24_000);
    let max_buffer = 24_000usize;
    let mut archived_games: Vec<ReplayGame> = Vec::new();
    let mut metrics: Vec<GenMetric> = Vec::new();

    for iter in 1..=cfg.iters {
        let t0 = Instant::now();

        // 1. Parallel Self-Play across all CPU cores
        let seeds: Vec<u64> = (0..cfg.games_per_iter).map(|_| rng.gen()).collect();
        let results: Vec<SelfPlayGameResult> = seeds
            .par_iter()
            .map(|&s| play_self_play_game(&net, cfg.rules, cfg.sims, s, false))
            .collect();

        let sp_elapsed = t0.elapsed().as_secs_f32();
        let mut black_wins = 0usize;
        let mut white_wins = 0usize;
        let mut draws = 0usize;
        let mut total_moves = 0usize;

        for r in &results {
            total_moves += r.moves.len();
            match r.winner {
                BLACK => black_wins += 1,
                WHITE => white_wins += 1,
                _ => draws += 1,
            }
        }

        // Keep shortest decisive game (or first game) from this generation for replay
        let showcase = results
            .iter()
            .filter(|r| r.winner != 0)
            .min_by_key(|r| r.moves.len())
            .unwrap_or(&results[0]);

        archived_games.push(ReplayGame {
            id: format!("gen-{iter}"),
            label: format_game_label(iter, showcase.winner, showcase.moves.len()),
            generation: iter,
            winner: showcase.winner,
            moves: showcase.moves.clone(),
        });

        for r in results {
            for sample in r.samples {
                if replay_buffer.len() >= max_buffer {
                    replay_buffer.pop_front();
                }
                replay_buffer.push_back(sample);
            }
        }

        // 2. Train on random mini-batches from replay buffer (capped so step time stays constant)
        let t_train = Instant::now();
        let mut indices: Vec<usize> = (0..replay_buffer.len()).collect();
        let mut p_loss_sum = 0.0f32;
        let mut v_loss_sum = 0.0f32;
        let mut step_count = 0usize;
        let max_steps = 64usize;

        let buf_slice = replay_buffer.make_contiguous();
        'train_loop: for _ in 0..cfg.epochs {
            indices.shuffle(&mut rng);
            for chunk in indices.chunks(cfg.batch_size) {
                let (grads, pl, vl) = chunk
                    .par_iter()
                    .fold(
                        || (net.zero_grads(), 0.0f32, 0.0f32),
                        |(mut acc_g, acc_p, acc_v), &idx| {
                            let s = &buf_slice[idx];
                            let (pl, vl) = net.accumulate_sample_grads(
                                &s.state,
                                &s.policy,
                                s.value,
                                &mut acc_g,
                            );
                            (acc_g, acc_p + pl, acc_v + vl)
                        },
                    )
                    .reduce(
                        || (net.zero_grads(), 0.0f32, 0.0f32),
                        |(mut g1, p1, v1), (g2, p2, v2)| {
                            g1.add_assign(&g2);
                            (g1, p1 + p2, v1 + v2)
                        },
                    );

                optimizer.step(&mut net, &grads, chunk.len());
                p_loss_sum += pl / chunk.len() as f32;
                v_loss_sum += vl / chunk.len() as f32;
                step_count += 1;
                if step_count >= max_steps {
                    break 'train_loop;
                }
            }
        }

        let avg_p = p_loss_sum / step_count.max(1) as f32;
        let avg_v = v_loss_sum / step_count.max(1) as f32;
        let train_elapsed = t_train.elapsed().as_secs_f32();
        let avg_moves = total_moves as f32 / cfg.games_per_iter as f32;

        metrics.push(GenMetric {
            generation: iter,
            p_loss: (avg_p * 1000.0).round() / 1000.0,
            v_loss: (avg_v * 1000.0).round() / 1000.0,
            avg_moves: (avg_moves * 10.0).round() / 10.0,
            black_wins,
            white_wins,
            draws,
            sp_sec: (sp_elapsed * 100.0).round() / 100.0,
            train_sec: (train_elapsed * 100.0).round() / 100.0,
            buffer_size: buf_slice.len(),
        });

        println!(
            "Gen {:2}/{:2} | SP: {:.2}s ({} games, avg {:.1} moves, B/W/D={}/{}/{}) | Train: {:.2}s (buf={}, p_loss={:.3}, v_loss={:.3})",
            iter,
            cfg.iters,
            sp_elapsed,
            cfg.games_per_iter,
            avg_moves,
            black_wins,
            white_wins,
            draws,
            train_elapsed,
            buf_slice.len(),
            avg_p,
            avg_v
        );

        // Save incremental checkpoint & replay archive every generation
        if let Some(parent) = Path::new(&cfg.checkpoint_path).parent() {
            fs::create_dir_all(parent).ok();
        }
        let ckpt_json = serde_json::to_string(&net).expect("serialize model");
        fs::write(&cfg.checkpoint_path, ckpt_json).expect("write checkpoint");

        if let Some(parent) = Path::new(&cfg.out_path).parent() {
            fs::create_dir_all(parent).ok();
        }
        let archive = ReplayArchive {
            rules: match cfg.rules {
                RuleSet::Renju => "renju".to_string(),
                RuleSet::Freestyle => "freestyle".to_string(),
            },
            model_info: ModelInfo {
                channels: net.channels,
                blocks: net.blocks.len(),
                params: net.num_params(),
            },
            metrics: metrics.clone(),
            games: archived_games.clone(),
        };
        let archive_json = serde_json::to_string_pretty(&archive).expect("serialize archive");
        fs::write(&cfg.out_path, archive_json).expect("write replay archive");
    }

    println!(
        "Done! Checkpoint: {} | Replay archive ({} games): {}",
        cfg.checkpoint_path,
        archived_games.len(),
        cfg.out_path
    );
}
