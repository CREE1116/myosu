import { describe, expect, it } from 'vitest';
import { createEmptyBoard } from '../renju';
import { forcedMove, sampleCandidate } from '../strength';
import { BLACK, WHITE } from '../types';

describe('하양이 strength helpers', () => {
  it('finishes its own five before blocking', () => {
    const board = createEmptyBoard();
    for (let c = 3; c <= 6; c++) board[5][c] = WHITE; // white four on row 5
    for (let c = 3; c <= 6; c++) board[9][c] = BLACK; // black four on row 9
    const m = forcedMove(board, WHITE, 'renju');
    expect(m?.row).toBe(5);
  });

  it('blocks the opponent four when it has no five', () => {
    const board = createEmptyBoard();
    board[9][2] = WHITE;
    for (let c = 3; c <= 6; c++) board[9][c] = BLACK; // closed black four, only H10 completes it
    expect(forcedMove(board, WHITE, 'renju')).toEqual({ row: 9, col: 7 });
  });

  it('returns null in a quiet position', () => {
    const board = createEmptyBoard();
    board[7][7] = BLACK;
    expect(forcedMove(board, WHITE, 'renju')).toBeNull();
  });

  it('samples only inside the top-k by weight', () => {
    const cands = [{ prob: 0.6 }, { prob: 0.3 }, { prob: 0.1 }];
    expect(sampleCandidate(cands, 1, 1, () => 0.99)).toBe(cands[0]);
    expect(sampleCandidate(cands, 2, 1, () => 0.0)).toBe(cands[0]);
    expect(sampleCandidate(cands, 2, 1, () => 0.99)).toBe(cands[1]);
  });
});
