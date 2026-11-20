// Finds where a finished game turned, from the human's side, so the cat can point at it.
// Only verified points: a winning four-sequence the human let go (exact search), and moves the
// engine's own search says lost the game while a better move kept it alive. Nothing → nothing.

import { analyzeMove, createEmptyBoard, opponentOf } from './renju';
import { proveVcf } from './vcf';
import { BLACK, Board, Coord, MoveRecord, Player, RuleSet } from './types';

export type TurningKind = 'missedWin' | 'turn';

export interface TurningPoint {
  /** Index of the human's move in `moves`; the position to show is moves.slice(0, index). */
  index: number;
  kind: TurningKind;
  /** missedWin: the winning four-sequence the human had. */
  win?: Coord[];
  /** turn: the engine's move that kept the game alive. */
  save?: Coord;
  /** turn: outlook with the best move, and after the move played (human side, -1..1). */
  kept?: number;
  lost?: number;
}

/** Engine search from `player`'s side: its best move and the position value for `player`. */
export type Analyzer = (board: Board, player: Player, last: Coord | null) => Promise<{ best: Coord; value: number } | null>;

/** A turn needs the best move to keep it level-ish and the played move to leave it lost. */
const KEPT_MIN = -0.2;
const LOST_MAX = -0.35;
const GAP_MIN = 0.4;

export function boardAt(moves: MoveRecord[], n: number): Board {
  const b = createEmptyBoard();
  for (const m of moves.slice(0, n)) b[m.row][m.col] = m.player;
  return b;
}

const nextTick = () => new Promise<void>((r) => setTimeout(r, 0));
const same = (a: Coord, b: Coord) => a.row === b.row && a.col === b.col;

function outlook(moves: MoveRecord[], i: number, human: Player): number | null {
  const v = moves[i]?.rawEval ?? moves[i]?.evalScore;
  if (v === undefined) return null;
  return human === BLACK ? v : -v; // stored evals are from Black's side
}

export async function findTurningPoints(
  moves: MoveRecord[],
  human: Player,
  winner: Player | null,
  rules: RuleSet,
  analyze?: Analyzer,
  max = 3,
): Promise<TurningPoint[]> {
  const opp = opponentOf(human);
  const found: TurningPoint[] = [];

  // 1. Missed wins: the human had a forced win and, by a complete search, no longer had it.
  if (winner !== human) {
    for (let i = 0; i < moves.length; i++) {
      if (moves[i].player !== human) continue;
      const had = proveVcf(boardAt(moves, i), human, rules);
      if (!had.line || same(had.line[0], moves[i])) continue;
      const after = proveVcf(boardAt(moves, Math.min(moves.length, i + 2)), human, rules);
      if (after.line || !after.complete) continue;
      // One missed attack often spans several moves; point at its first miss.
      if (found.some((f) => f.kind === 'missedWin' && i - f.index <= 4)) continue;
      found.push({ index: i, kind: 'missedWin', win: had.line });
      if (i % 6 === 0) await nextTick();
    }
  }

  // 2. Turns, checked by the engine. Shortlist: the biggest stored drops and the start of the
  //    final collapse (the opponent's four-sequence), since searching every move is too slow.
  if (analyze && winner !== human) {
    const short = new Set<number>();
    const drops: Array<[number, number]> = [];
    for (let i = 0; i < moves.length; i++) {
      if (moves[i].player !== human) continue;
      const was = i > 0 ? outlook(moves, i - 1, human) : 0;
      const now = outlook(moves, i + 1, human);
      if (was !== null && now !== null && was - now >= 0.15) drops.push([i, was - now]);
    }
    drops.sort((a, b) => b[1] - a[1]).slice(0, 3).forEach(([i]) => short.add(i));
    if (winner === opp) {
      let last = -1;
      for (let i = moves.length - 1; i >= 0; i--) {
        if (moves[i].player !== human) continue;
        if (!proveVcf(boardAt(moves, i + 1), opp, rules).line) break;
        last = i;
      }
      if (last >= 0) short.add(last);
      if (last >= 2) short.add(last - 2);
    }

    const turns: TurningPoint[] = [];
    for (const i of [...short].sort((a, b) => a - b)) {
      if (found.some((f) => f.index === i) || i + 1 >= moves.length) continue;
      const played = moves[i];
      const prev = moves[i - 1] ?? null;
      const before = boardAt(moves, i);
      const best = await analyze(before, human, prev);
      if (!best || same(best.best, played) || best.value < KEPT_MIN) continue; // already lost: not the turn
      if (!analyzeMove(before, best.best.row, best.best.col, human, rules).legal) continue;
      const reply = await analyze(boardAt(moves, i + 1), opp, played);
      if (!reply) continue;
      const kept = best.value;
      const lost = -reply.value;
      if (kept < KEPT_MIN || lost > LOST_MAX || kept - lost < GAP_MIN) continue;
      // The better move must not simply hand the opponent a four-sequence.
      before[best.best.row][best.best.col] = human;
      const hole = proveVcf(before, opp, rules).line;
      if (hole) continue;
      turns.push({ index: i, kind: 'turn', save: best.best, kept, lost });
    }
    // The clearest swings only.
    turns.sort((a, b) => b.kept! - b.lost! - (a.kept! - a.lost!)).slice(0, 2).forEach((t) => found.push(t));
  }

  return found.sort((a, b) => a.index - b.index).slice(-max);
}
