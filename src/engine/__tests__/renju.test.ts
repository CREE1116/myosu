// @ts-expect-error Vitest runs in Node environment without @types/node installed
import fs from 'node:fs';
import { describe, expect, it } from 'vitest';
import { chooseAIMove, evaluatePositionAdvantage } from '../ai';
import { catLines } from '../catDialogue';
import {
  TOTAL_WEIGHT_FLOATS,
  TOTAL_WEIGHT_FLOATS_V2,
  TOTAL_WEIGHT_FLOATS_V3,
  chooseCheckpointMove,
  evaluateD4OverlapPattern,
  parseD4Weights,
} from '../nnModel';
import {
  analyzeMove,
  checkWinner,
  createEmptyBoard,
  getForbiddenMoves,
  hasAnyLegalMove,
  scanBoardThreats,
} from '../renju';
import { findVcfSequence } from '../vcf';
import { BLACK, WHITE } from '../types';

describe('Renju Rules Engine', () => {
  it('detects exact 5 win for Black and 5+ win for White', () => {
    const board = createEmptyBoard();
    for (let c = 3; c <= 6; c++) board[7][c] = BLACK;

    const winMove = analyzeMove(board, 7, 7, BLACK, 'renju');
    expect(winMove.legal).toBe(true);
    expect(winMove.isWin).toBe(true);
    expect(winMove.winLine).toHaveLength(5);

    board[7][7] = BLACK;
    expect(checkWinner(board, { row: 7, col: 7 }, 'renju')?.winner).toBe(BLACK);
  });

  it('forbids Black overline (6+ in a row) in Renju, but allows White 6+', () => {
    const board = createEmptyBoard();
    // 3 black stones on left, gap at (7, 6), 2 black stones on right -> 6 in a row
    board[7][3] = BLACK;
    board[7][4] = BLACK;
    board[7][5] = BLACK;
    board[7][7] = BLACK;
    board[7][8] = BLACK;

    const blackOverline = analyzeMove(board, 7, 6, BLACK, 'renju');
    expect(blackOverline.legal).toBe(false);
    expect(blackOverline.forbiddenReason).toBe('OVERLINE');

    // In freestyle, 6 in a row is legal and wins
    const freestyleMove = analyzeMove(board, 7, 6, BLACK, 'freestyle');
    expect(freestyleMove.legal).toBe(true);
    expect(freestyleMove.isWin).toBe(true);

    // White 6 in a row in Renju is legal and wins
    const whiteBoard = createEmptyBoard();
    whiteBoard[7][3] = WHITE;
    whiteBoard[7][4] = WHITE;
    whiteBoard[7][5] = WHITE;
    whiteBoard[7][7] = WHITE;
    whiteBoard[7][8] = WHITE;
    const whiteOverline = analyzeMove(whiteBoard, 7, 6, WHITE, 'renju');
    expect(whiteOverline.legal).toBe(true);
    expect(whiteOverline.isWin).toBe(true);
  });

  it('forbids Black 3-3 (double open three), including jump-three', () => {
    const board = createEmptyBoard();
    // Horizontal open 3 if placed at (7,7): (7,5), (7,6)
    board[7][5] = BLACK;
    board[7][6] = BLACK;
    // Vertical jump-3 if placed at (7,7): (5,7), (8,7) -> wait: (5,7), _, (7,7), (8,7)
    board[5][7] = BLACK;
    board[8][7] = BLACK;

    const analysis = analyzeMove(board, 7, 7, BLACK, 'renju');
    expect(analysis.legal).toBe(false);
    expect(analysis.forbiddenReason).toBe('THREE_THREE');
    expect(analysis.openThrees).toHaveLength(2);

    // White at same position is legal
    const whiteBoard = createEmptyBoard();
    whiteBoard[7][5] = WHITE;
    whiteBoard[7][6] = WHITE;
    whiteBoard[5][7] = WHITE;
    whiteBoard[8][7] = WHITE;
    expect(analyzeMove(whiteBoard, 7, 7, WHITE, 'renju').legal).toBe(true);
  });

  it('forbids Black 4-4 (cross 4-4 and same-line 4-4)', () => {
    // Cross 4-4
    const crossBoard = createEmptyBoard();
    crossBoard[7][4] = BLACK;
    crossBoard[7][5] = BLACK;
    crossBoard[7][6] = BLACK;
    crossBoard[4][7] = BLACK;
    crossBoard[5][7] = BLACK;
    crossBoard[6][7] = BLACK;

    const cross44 = analyzeMove(crossBoard, 7, 7, BLACK, 'renju');
    expect(cross44.legal).toBe(false);
    expect(cross44.forbiddenReason).toBe('FOUR_FOUR');

    // Same-line 4-4: ● ● ● _ [X] _ ● ● ●
    const lineBoard = createEmptyBoard();
    lineBoard[7][3] = BLACK;
    lineBoard[7][4] = BLACK;
    lineBoard[7][5] = BLACK;
    // gap at (7,6), target at (7,7), gap at (7,8)
    lineBoard[7][9] = BLACK;
    lineBoard[7][10] = BLACK;
    lineBoard[7][11] = BLACK;

    const line44 = analyzeMove(lineBoard, 7, 7, BLACK, 'renju');
    expect(line44.legal).toBe(false);
    expect(line44.forbiddenReason).toBe('FOUR_FOUR');
  });

  it('allows Black 4-3 (one Four + one Open Three)', () => {
    const board = createEmptyBoard();
    // Horizontal 4 at (7,7)
    board[7][4] = BLACK;
    board[7][5] = BLACK;
    board[7][6] = BLACK;
    // Vertical open 3 at (7,7)
    board[5][7] = BLACK;
    board[6][7] = BLACK;

    const move43 = analyzeMove(board, 7, 7, BLACK, 'renju');
    expect(move43.legal).toBe(true);
    expect(move43.forbiddenReason).toBeNull();
    expect(move43.openFours).toHaveLength(1);
    expect(move43.openThrees).toHaveLength(1);
  });

  it('allows exact 5 even if it also creates 3-3, 4-4, or overline on other axes', () => {
    const board = createEmptyBoard();
    // Exact 5 horizontally at (7,7)
    board[7][3] = BLACK;
    board[7][4] = BLACK;
    board[7][5] = BLACK;
    board[7][6] = BLACK;
    // Overline vertically at (7,7)
    board[4][7] = BLACK;
    board[5][7] = BLACK;
    board[6][7] = BLACK;
    board[8][7] = BLACK;
    board[9][7] = BLACK;

    const res = analyzeMove(board, 7, 7, BLACK, 'renju');
    expect(res.legal).toBe(true);
    expect(res.isWin).toBe(true);
    expect(res.forbiddenReason).toBeNull();
  });

  it('allows false 3-3 when one open three cannot legally extend to an open four', () => {
    const board = createEmptyBoard();
    // Vertical open 3 through (7,7): (5,7), (6,7), [7,7]
    board[5][7] = BLACK;
    board[6][7] = BLACK;

    // Horizontal 3 through (7,7): (7,5), (7,6), [7,7]
    // Block BOTH open-4 completion points (7,4) and (7,8) with overline guards:
    // Put black stones at (7,2), (7,3) -> wait, (7,2) (7,3) _ (7,5) (7,6) [7,7]:
    // Placing at (7,4) creates 6 in a row! (7,2..7,7) -> overline!
    // Put black stone at (7,10) and white at (7,9)? No, white at (7,9) still leaves (7,8) empty,
    // wait: if (7,9) is WHITE, then placing at (7,8) makes 4 stones (7,5..7,8), blocked on right by WHITE(7,9) -> NOT an Open Four!
    board[7][2] = BLACK;
    board[7][3] = BLACK;
    board[7][5] = BLACK;
    board[7][6] = BLACK;
    board[7][9] = WHITE;

    // Now at (7,7):
    // - Extending horizontal left at (7,4) makes 6 stones -> OVERLINE (forbidden)!
    // - Extending horizontal right at (7,8) is blocked by WHITE(7,9) -> Closed Four, not Open Four!
    // So horizontal is NOT an Open Three -> NOT 3-3!
    const false33 = analyzeMove(board, 7, 7, BLACK, 'renju');
    expect(false33.legal).toBe(true);
    expect(false33.openThrees).toHaveLength(1);
  });

  it('lists all forbidden cells for Black and scans active 3 and 4 threats', () => {
    const board = createEmptyBoard();
    board[7][5] = BLACK;
    board[7][6] = BLACK;
    board[5][7] = BLACK;
    board[8][7] = BLACK;

    const forbidden = getForbiddenMoves(board, 'renju');
    expect(forbidden.some((f) => f.row === 7 && f.col === 7 && f.reason === 'THREE_THREE')).toBe(
      true,
    );

    // Add a White Open Three and Black Closed Four
    board[3][3] = WHITE;
    board[3][4] = WHITE;
    board[3][5] = WHITE;

    board[10][2] = WHITE; // blocker
    board[10][3] = BLACK;
    board[10][4] = BLACK;
    board[10][5] = BLACK;
    board[10][6] = BLACK;

    const threats = scanBoardThreats(board, 'renju');
    expect(
      threats.some((t) => t.player === WHITE && t.type === 'OPEN_THREE'),
    ).toBe(true);
    expect(
      threats.some(
        (t) =>
          t.player === BLACK &&
          t.type === 'CLOSED_FOUR' &&
          t.targetCells.some((c) => c.row === 10 && c.col === 7),
      ),
    ).toBe(true);
  });
});

describe('AI Engine', () => {
  it('takes immediate winning move and blocks opponent 4', () => {
    const winBoard = createEmptyBoard();
    for (let c = 4; c <= 7; c++) winBoard[7][c] = BLACK;
    winBoard[6][4] = WHITE;
    winBoard[6][5] = WHITE;
    winBoard[6][6] = WHITE;
    winBoard[6][7] = WHITE;

    const blackWin = chooseAIMove(winBoard, BLACK, 'renju', 'normal');
    expect(blackWin).not.toBeNull();
    expect(
      (blackWin!.row === 7 && blackWin!.col === 3) ||
        (blackWin!.row === 7 && blackWin!.col === 8),
    ).toBe(true);

    // Block opponent closed 4 when AI has no immediate win
    const defBoard = createEmptyBoard();
    defBoard[7][2] = WHITE; // blocks left
    defBoard[7][3] = BLACK;
    defBoard[7][4] = BLACK;
    defBoard[7][5] = BLACK;
    defBoard[7][6] = BLACK;
    defBoard[5][5] = WHITE;

    const whiteBlock = chooseAIMove(defBoard, WHITE, 'renju', 'normal');
    expect(whiteBlock).toEqual({ row: 7, col: 7 });
  });

  it('never chooses a forbidden move when playing Black in Renju', () => {
    const board = createEmptyBoard();
    board[7][5] = BLACK;
    board[7][6] = BLACK;
    board[5][7] = BLACK;
    board[8][7] = BLACK;
    board[1][1] = WHITE;

    const move = chooseAIMove(board, BLACK, 'renju', 'hard');
    expect(move).not.toBeNull();
    const check = analyzeMove(board, move!.row, move!.col, BLACK, 'renju');
    expect(check.legal).toBe(true);
  });
});

describe('VCF (Continuous Four) Solver', () => {
  it('finds a 5-ply Continuous Four winning sequence', () => {
    const board = createEmptyBoard();
    // Black threatens Closed Four at (7,7) -> forces White block at (7,8) ->
    // creating an Open Four vertically through (7,7) if Black has (5,7), (6,7), (8,7)
    //Wait: let's build a clean 5-ply VCF:
    // Horizontal closed 4 when Black plays (7,7): (7,3)=WHITE, (7,4..7,6)=BLACK, (7,7)=[move1], (7,8)=[forced block]
    // Wait: if (7,8) is blocked, we need another closed 4 / open 4 using (7,7)!
    // Put (5,7)=BLACK, (6,7)=BLACK, (4,7)=EMPTY, (8,7)=EMPTY, (3,7)=EMPTY, (9,7)=EMPTY.
    // Then playing (7,7) creates a horizontal Closed 4 (7,4..7,7) forcing White to (7,8),
    // AND vertically (5,7),(6,7),(7,7) is an Open 3! Wait: on move 3, Black plays (8,7) to make an Open Four (5..8,7)!
    board[7][3] = WHITE;
    board[7][4] = BLACK;
    board[7][5] = BLACK;
    board[7][6] = BLACK;
    board[5][7] = BLACK;
    board[6][7] = BLACK;

    const seq = findVcfSequence(board, BLACK, 'renju', 7);
    expect(seq).not.toBeNull();
    expect(seq![0]).toEqual({ row: 7, col: 7 });
    expect(seq![1]).toEqual({ row: 7, col: 8 });
    expect(seq!.length).toBe(5);
  });

  it('detects a Forbidden Move Trap where Black cannot legally block White four', () => {
    const board = createEmptyBoard();
    // Black has a 3-3 forbidden point at (7,7)
    board[7][5] = BLACK;
    board[7][6] = BLACK;
    board[5][7] = BLACK;
    board[8][7] = BLACK;
    // White has stones on diagonal or column ready to make a closed 4 whose ONLY block is (7,7):
    // Put White blocker? Wait: White at (3,11), (4,10), (5,9) and Black at (2,12) ->
    // Playing White at (6,8) makes 4 stones (3,11)..(6,8) blocked at (2,12), so ONLY completing cell is (7,7)!
    board[2][12] = BLACK;
    board[3][11] = WHITE;
    board[4][10] = WHITE;
    board[5][9] = WHITE;

    const seq = findVcfSequence(board, WHITE, 'renju', 5);
    expect(seq).not.toBeNull();
    expect(seq![0]).toEqual({ row: 6, col: 8 });
    expect(seq!.length).toBe(1);
  });

  it('returns legal moves across all 4 AI character tiers (lv1..lv4)', () => {
    const board = createEmptyBoard();
    board[7][7] = BLACK;
    board[7][8] = WHITE;
    board[6][7] = BLACK;
    for (const tier of ['lv1', 'lv2', 'lv3', 'lv4'] as const) {
      const mv = chooseAIMove(board, BLACK, 'renju', tier);
      expect(mv).not.toBeNull();
      const check = analyzeMove(board, mv!.row, mv!.col, BLACK, 'renju');
      expect(check.legal).toBe(true);
    }
  });

  it('cat warns about its own four and its pending VCF, once each', () => {
    const board = createEmptyBoard();
    board[7][4] = WHITE;
    board[7][5] = WHITE;
    board[7][6] = WHITE;
    board[7][7] = WHITE;
    board[7][3] = BLACK;
    const threats = scanBoardThreats(board, 'renju');
    const base = {
      aiPlayer: WHITE,
      turnPlayer: BLACK,
      moveCount: 5,
      evalScore: 0,
      lastMove: null,
      winner: null,
      isDraw: false,
      activeVcf: null,
      aiPendingVcf: null,
      nnPredictedPliesLeft: null,
    } as const;

    const four = catLines({ ...base, threats })[0];
    expect(four.tone).toBe('warn');
    expect(four.key).toBe('ai-four:I8');
    expect(four.text).toContain('I8');

    const vcf = catLines({
      ...base,
      threats: [],
      aiPendingVcf: [
        { row: 7, col: 8 },
        { row: 7, col: 9 },
        { row: 6, col: 8 },
      ],
    })[0];
    expect(vcf.mood).toBe('warn');
    expect(vcf.text).toContain('I8');

    // Quiet, even position: nothing worth saying.
    expect(catLines({ ...base, threats: [] })).toHaveLength(0);
  });

  it('detects draw via hasAnyLegalMove when board is full or only forbidden Black spots remain', () => {
    const board = createEmptyBoard();
    for (let r = 0; r < 15; r++) {
      for (let c = 0; c < 15; c++) {
        board[r][c] = (r + c) % 2 === 0 ? BLACK : WHITE;
      }
    }
    expect(hasAnyLegalMove(board, BLACK, 'renju')).toBe(false);
    expect(hasAnyLegalMove(board, WHITE, 'renju')).toBe(false);

    // Leave a single empty intersection at (7,7) that is an overline (6+) for Black
    board[7][7] = 0;
    board[7][4] = BLACK;
    board[7][5] = BLACK;
    board[7][6] = BLACK;
    board[7][8] = BLACK;
    board[7][9] = BLACK;
    board[7][3] = WHITE;
    board[7][10] = WHITE;
    expect(analyzeMove(board, 7, 7, BLACK, 'renju').legal).toBe(false);
    expect(hasAnyLegalMove(board, BLACK, 'renju')).toBe(false);
    expect(hasAnyLegalMove(board, WHITE, 'renju')).toBe(true);
  });

  it('keeps position advantage smooth across balanced openings and spikes on unblockable open four', () => {
    const board = createEmptyBoard();
    board[7][7] = BLACK;
    const afterB1 = evaluatePositionAdvantage(board, WHITE, 'renju');
    board[6][7] = WHITE;
    const afterW2 = evaluatePositionAdvantage(board, BLACK, 'renju');
    board[6][8] = BLACK;
    const afterB3 = evaluatePositionAdvantage(board, WHITE, 'renju');
    board[8][6] = WHITE;
    const afterW4 = evaluatePositionAdvantage(board, BLACK, 'renju');

    expect(Math.abs(afterB1)).toBeLessThan(0.2);
    expect(Math.abs(afterW2)).toBeLessThan(0.2);
    expect(Math.abs(afterB3)).toBeLessThan(0.2);
    expect(Math.abs(afterW4)).toBeLessThan(0.2);

    const winBoard = createEmptyBoard();
    winBoard[7][5] = BLACK;
    winBoard[7][6] = BLACK;
    winBoard[7][7] = BLACK;
    winBoard[7][8] = BLACK;
    winBoard[8][5] = WHITE;
    winBoard[8][6] = WHITE;
    expect(evaluatePositionAdvantage(winBoard, WHITE, 'renju')).toBeGreaterThan(0.8);
  });

  it('loads exported D4OverlapPatternNet .bin checkpoint and produces legal MCTS move', () => {
    // The released model is committed in models/, so this runs on a fresh clone too.
    const modelsDir = new URL('../../../models/', import.meta.url);
    const { binFile } = JSON.parse(fs.readFileSync(new URL('manifest.json', modelsDir), 'utf-8')).checkpoints[0];
    const binPath = new URL(binFile, modelsDir);
    const raw = fs.readFileSync(binPath);
    expect([TOTAL_WEIGHT_FLOATS * 4, TOTAL_WEIGHT_FLOATS_V2 * 4, TOTAL_WEIGHT_FLOATS_V3 * 4]).toContain(raw.byteLength);

    const ab = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
    const weights = parseD4Weights(ab);

    const board = createEmptyBoard();
    board[7][7] = BLACK;
    board[6][7] = WHITE;
    board[5][8] = BLACK;

    const evalOut = evaluateD4OverlapPattern(weights, board, WHITE, { row: 5, col: 8 });
    expect(Number.isFinite(evalOut.value)).toBe(true);
    expect(evalOut.value).toBeGreaterThanOrEqual(-1);
    expect(evalOut.value).toBeLessThanOrEqual(1);

    let probSum = 0;
    for (let i = 0; i < 225; i++) probSum += evalOut.probs[i];
    expect(probSum).toBeCloseTo(1, 3);

    const move = chooseCheckpointMove(board, WHITE, weights, { row: 5, col: 8 }, 16);
    expect(move).not.toBeNull();
    expect(board[move!.row][move!.col]).toBe(0);
    expect(move!.topCandidates.length).toBeGreaterThan(0);
  });
});
