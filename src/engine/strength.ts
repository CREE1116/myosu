// How hard each cat plays. Both use the same neural net; 하양이 reads far less and picks
// among its top candidates instead of always the best, but never misses a forced move.

import type { CatId } from '../ui/cat';
import { analyzeMove, opponentOf, scanBoardThreats } from './renju';
import { Board, Coord, Player, RuleSet } from './types';

export interface Strength {
  /** MCTS simulations and time budget per move. */
  sims: number;
  budgetMs: number;
  /** Sample among this many top candidates (1 = always the best). */
  topK: number;
  /** Sampling temperature over visit shares; higher = more varied. */
  temperature: number;
  /** Always finish its own five and block the opponent's four. */
  guard: boolean;
}

// shortcut: tuned by feel; measure 하양이 vs 까망이 win rate in self-play before release.
export const STRENGTH: Record<CatId, Strength> = {
  kkamang: { sims: 160, budgetMs: 950, topK: 1, temperature: 1, guard: false },
  hayang: { sims: 24, budgetMs: 450, topK: 4, temperature: 1.2, guard: true },
};

/** A move that must be played now: own five first, otherwise blocking the opponent's four. */
export function forcedMove(board: Board, player: Player, rules: RuleSet): Coord | null {
  const fours = scanBoardThreats(board, rules).filter((t) => t.type === 'OPEN_FOUR' || t.type === 'CLOSED_FOUR');
  for (const t of fours.filter((x) => x.player === player)) {
    for (const c of t.targetCells) {
      if (analyzeMove(board, c.row, c.col, player, rules).isWin) return c;
    }
  }
  const opp = opponentOf(player);
  for (const t of fours.filter((x) => x.player === opp)) {
    for (const c of t.targetCells) {
      if (analyzeMove(board, c.row, c.col, player, rules).legal) return c;
    }
  }
  return null;
}

/** Pick one of the top-k candidates with probability ∝ share^(1/temperature). */
export function sampleCandidate<T extends { prob: number }>(
  candidates: T[],
  topK: number,
  temperature: number,
  rand: () => number = Math.random,
): T | null {
  const pool = candidates.slice(0, Math.max(1, topK)).filter((c) => c.prob > 0);
  if (pool.length === 0) return candidates[0] ?? null;
  const weights = pool.map((c) => Math.pow(c.prob, 1 / Math.max(0.05, temperature)));
  let r = rand() * weights.reduce((a, b) => a + b, 0);
  for (let i = 0; i < pool.length; i++) {
    r -= weights[i];
    if (r <= 0) return pool[i];
  }
  return pool[pool.length - 1];
}
