import { describe, expect, it } from 'vitest';
import { checkAttack, puzzleBoard, stillWinning } from '../puzzle';
import { PUZZLES } from '../puzzleSet';
import { opponentOf } from '../renju';
import { Analyzer, findTurningPoints } from '../review';
import { forcedMove } from '../strength';
import { BLACK, MoveRecord, Player, WHITE } from '../types';

const RULES = 'renju' as const;
const at = (k: number) => ({ row: Math.floor(k / 15), col: k % 15 });

describe('puzzle set', () => {
  it.each(PUZZLES.map((p) => [p.id, p] as const))('%s is solvable by its line', (_id, p) => {
    const board = puzzleBoard(p);
    const def = opponentOf(p.attacker);
    expect(stillWinning(board, p.attacker, RULES)).toBe(true);
    // Not a one-shot open four: at least two forcing fours before it's over.
    expect(p.line.length).toBeGreaterThanOrEqual(p.theme === 'trap' ? 3 : 5);
    p.line.forEach((k, i) => {
      const c = at(k);
      if (i % 2 === 0) {
        const last = i === p.line.length - 1;
        const res = checkAttack(board, c, p.attacker, RULES);
        expect(res).toBe(last && p.theme !== 'trap' ? 'win' : 'four');
        board[c.row][c.col] = p.attacker;
        // A trap ends on a four whose only block is forbidden for the defender.
        if (last && p.theme === 'trap') expect(forcedMove(board, def, RULES)).toBeNull();
      } else {
        board[c.row][c.col] = def;
        expect(stillWinning(board, p.attacker, RULES)).toBe(true);
      }
    });
  });
});

describe('turning points', () => {
  const mv = (i: number, k: number, player: Player, rawEval?: number): MoveRecord => ({
    ...at(k), player, moveNumber: i + 1, rawEval, evalScore: rawEval,
  });

  it('finds a five the human walked past', async () => {
    // Black (human) has four on row 7 (cols 3-6) after move 7, then plays elsewhere.
    const moves = [
      mv(0, 7 * 15 + 3, BLACK), mv(1, 0, WHITE),
      mv(2, 7 * 15 + 4, BLACK), mv(3, 2, WHITE),
      mv(4, 7 * 15 + 5, BLACK), mv(5, 4, WHITE),
      mv(6, 7 * 15 + 6, BLACK), mv(7, 7 * 15 + 2, WHITE),
      mv(8, 12 * 15 + 12, BLACK), mv(9, 7 * 15 + 7, WHITE),
    ];
    const tps = await findTurningPoints(moves, BLACK, null, RULES);
    expect(tps[0]).toMatchObject({ index: 8, kind: 'missedWin' });
    expect(tps[0].win?.[0]).toEqual({ row: 7, col: 7 });
  });

  it('keeps an engine-checked turn and drops a swing the engine does not confirm', async () => {
    // Stored evals: big drop at move 5 (index 4) and at move 7 (index 6).
    const moves = [
      mv(0, 112, BLACK, 0.1), mv(1, 113, WHITE, 0.1),
      mv(2, 100, BLACK, 0.1), mv(3, 101, WHITE, 0.0),
      mv(4, 140, BLACK, 0.0), mv(5, 102, WHITE, -0.4),
      mv(6, 30, BLACK, -0.4), mv(7, 103, WHITE, -0.9),
      mv(8, 31, BLACK, -0.9), mv(9, 104, WHITE, -0.95),
    ];
    // Fake engine: before move 7 Black could hold with C3 (value 0); after it Black is lost.
    // Before move 5 the engine says it was already bad, so that swing isn't a turn.
    const analyze: Analyzer = async (b, p) => {
      const n = b.flat().filter(Boolean).length;
      if (p === BLACK) return { best: { row: 2, col: 2 }, value: n === 6 ? 0 : -0.5 };
      return { best: { row: 0, col: 0 }, value: n === 7 ? 0.8 : 0.5 };
    };
    const tps = await findTurningPoints(moves, BLACK, WHITE, RULES, analyze);
    expect(tps).toEqual([expect.objectContaining({ index: 6, kind: 'turn', save: { row: 2, col: 2 } })]);
  });

  it('says nothing when nothing is verified', async () => {
    const moves = [mv(0, 112, BLACK), mv(1, 113, WHITE), mv(2, 100, BLACK), mv(3, 101, WHITE)];
    expect(await findTurningPoints(moves, BLACK, null, RULES)).toEqual([]);
  });
});
