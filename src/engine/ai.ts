import {
  AXES,
  analyzeMove,
  cloneBoard,
  inBounds,
  opponentOf,
} from './renju';
import { findVcfSequence } from './vcf';
import {
  AiCharacter,
  BLACK,
  BOARD_SIZE,
  Board,
  Coord,
  Difficulty,
  EMPTY,
  Player,
  RuleSet,
  WHITE,
} from './types';

export const AI_CHARACTERS: ReadonlyArray<AiCharacter> = [
  {
    id: 'lv1',
    name: '모코',
    tier: '입문',
    badge: '🐣',
    desc: '당장 5목만 막고 자유롭게 두는 입문자',
    greeting: '모코: 잘 부탁해요! 천천히 둬요.',
  },
  {
    id: 'lv2',
    name: '소라',
    tier: '초중급',
    badge: '🌱',
    desc: '4목과 열린 3을 꼼꼼히 챙기는 기원 수습생',
    greeting: '소라: 빈틈이 보이면 바로 두겠습니다.',
  },
  {
    id: 'lv3',
    name: '현우',
    tier: '중고수',
    badge: '🎋',
    desc: '4·3 공격과 7수 연속 사를 노리는 실전 사범',
    greeting: '현우: 좋은 수순으로 한 판 배워봅시다.',
  },
  {
    id: 'lv4',
    name: '윤',
    tier: '최강',
    badge: '⚡',
    desc: '깊은 수읽기와 금수 덫까지 읽는 천원 마스터',
    greeting: '윤: 빈틈없는 수읽기로 상대하겠습니다.',
  },
];

const WIN_SCORE = 1_000_000;

interface ScoredMove {
  row: number;
  col: number;
  score: number;
}

function getAllLines(board: Board): number[][] {
  const lines: number[][] = [];

  // Rows
  for (let r = 0; r < BOARD_SIZE; r++) {
    lines.push(board[r]);
  }
  // Cols
  for (let c = 0; c < BOARD_SIZE; c++) {
    const col: number[] = [];
    for (let r = 0; r < BOARD_SIZE; r++) col.push(board[r][c]);
    lines.push(col);
  }
  // Main diagonals (r - c = k)
  for (let k = -(BOARD_SIZE - 5); k <= BOARD_SIZE - 5; k++) {
    const diag: number[] = [];
    for (let r = 0; r < BOARD_SIZE; r++) {
      const c = r - k;
      if (c >= 0 && c < BOARD_SIZE) diag.push(board[r][c]);
    }
    if (diag.length >= 5) lines.push(diag);
  }
  // Anti-diagonals (r + c = k)
  for (let k = 4; k <= 2 * BOARD_SIZE - 6; k++) {
    const diag: number[] = [];
    for (let r = 0; r < BOARD_SIZE; r++) {
      const c = k - r;
      if (c >= 0 && c < BOARD_SIZE) diag.push(board[r][c]);
    }
    if (diag.length >= 5) lines.push(diag);
  }

  return lines;
}

function scoreContiguousRun(
  len: number,
  openEnds: number,
  isExactFiveOnly: boolean,
): number {
  if (len >= 5) {
    if (isExactFiveOnly && len >= 6) return -200; // Overline penalty for Renju Black
    return WIN_SCORE;
  }
  if (openEnds === 0) return 0;

  switch (len) {
    case 4:
      return openEnds === 2 ? 60_000 : 8_000;
    case 3:
      return openEnds === 2 ? 5_000 : 400;
    case 2:
      return openEnds === 2 ? 300 : 40;
    case 1:
      return openEnds === 2 ? 12 : 2;
    default:
      return 0;
  }
}

function evaluatePlayerOnLine(
  line: number[],
  player: Player,
  rules: RuleSet,
): number {
  let score = 0;
  const n = line.length;
  const isExactFiveOnly = rules === 'renju' && player === BLACK;

  let i = 0;
  while (i < n) {
    if (line[i] !== player) {
      i++;
      continue;
    }

    let j = i;
    while (j < n && line[j] === player) j++;
    const len = j - i;

    const leftOpen = i > 0 && line[i - 1] === EMPTY ? 1 : 0;
    const rightOpen = j < n && line[j] === EMPTY ? 1 : 0;
    score += scoreContiguousRun(len, leftOpen + rightOpen, isExactFiveOnly);

    // Check 1-gap jump patterns (e.g. X X _ X, X _ X X, X _ X)
    if (j + 1 < n && line[j] === EMPTY && line[j + 1] === player) {
      let k = j + 1;
      while (k < n && line[k] === player) k++;
      const rightLen = k - (j + 1);
      const total = len + rightLen;
      if (total === 4 && (!isExactFiveOnly || total === 4)) {
        score += 7_000; // Jump-4 threat
      } else if (total === 3) {
        const outerLeft = i > 0 && line[i - 1] === EMPTY;
        const outerRight = k < n && line[k] === EMPTY;
        if (outerLeft && outerRight) score += 3_600; // Jump open-3
        else if (outerLeft || outerRight) score += 250;
      }
    }

    i = j;
  }

  return score;
}

function evaluateSymmetricLines(board: Board, rules: RuleSet): number {
  const lines = getAllLines(board);
  let blackScore = 0;
  let whiteScore = 0;
  for (const line of lines) {
    blackScore += evaluatePlayerOnLine(line, BLACK, rules);
    whiteScore += evaluatePlayerOnLine(line, WHITE, rules);
  }
  return blackScore - whiteScore;
}

/**
 * Evaluates true Black-perspective position advantage in [-1, 1] considering whose turn it is next,
 * preventing single-ply stone-count / unblockable-threat sawtooth oscillations.
 */
export function evaluatePositionAdvantage(
  board: Board,
  turnToMove: Player,
  rules: RuleSet,
): number {
  const candidates = getCandidateCoords(board);
  const justMoved = opponentOf(turnToMove);

  let justMovedWins = 0;
  for (const { row, col } of candidates) {
    if (board[row][col] !== EMPTY) continue;
    const myA = analyzeMove(board, row, col, turnToMove, rules);
    if (myA.isWin) {
      return turnToMove === BLACK ? 0.96 : -0.96;
    }
    const oppA = analyzeMove(board, row, col, justMoved, rules);
    if (oppA.isWin) {
      justMovedWins++;
    }
  }

  // Two simultaneous immediate winning points (Open-4 or Double-4) cannot both be blocked
  if (justMovedWins >= 2) {
    return justMoved === BLACK ? 0.92 : -0.92;
  }

  const ordered = getOrderedMoves(board, turnToMove, rules, 1);
  if (ordered.length === 0) return 0;

  const sNow = evaluateSymmetricLines(board, rules);
  const reply = ordered[0];
  let sAfter = sNow;
  let decisiveReply = false;

  board[reply.row][reply.col] = turnToMove;
  try {
    if (justMovedWins === 0) {
      const rA = analyzeMove(board, reply.row, reply.col, turnToMove, rules);
      if (
        rA.openFours.length > 0 ||
        (rA.closedFours.length >= 1 && rA.openThrees.length >= 1) ||
        rA.closedFours.length >= 2
      ) {
        decisiveReply = true;
      }
    }
    if (!decisiveReply) {
      sAfter = evaluateSymmetricLines(board, rules);
    }
  } finally {
    board[reply.row][reply.col] = EMPTY;
  }

  if (decisiveReply) {
    return turnToMove === BLACK ? 0.84 : -0.84;
  }

  const blended = 0.25 * sNow + 0.75 * sAfter;
  let stoneCount = 0;
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      if (board[r][c] !== EMPTY) stoneCount++;
    }
  }
  const divisor = stoneCount <= 4 ? 9000 : 4200;
  return Math.max(-0.9, Math.min(0.9, Math.tanh(blended / divisor)));
}

export function evaluateBoard(
  board: Board,
  perspective: Player,
  rules: RuleSet,
): number {
  const opp = opponentOf(perspective);
  const lines = getAllLines(board);

  let myScore = 0;
  let oppScore = 0;
  for (const line of lines) {
    myScore += evaluatePlayerOnLine(line, perspective, rules);
    oppScore += evaluatePlayerOnLine(line, opp, rules);
  }

  return myScore - oppScore * 1.1;
}

function getCandidateCoords(board: Board): Coord[] {
  const marked = Array.from({ length: BOARD_SIZE }, () =>
    Array<boolean>(BOARD_SIZE).fill(false),
  );
  let hasStone = false;

  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      if (board[r][c] === EMPTY) continue;
      hasStone = true;
      for (let dr = -2; dr <= 2; dr++) {
        for (let dc = -2; dc <= 2; dc++) {
          const nr = r + dr;
          const nc = c + dc;
          if (inBounds(nr, nc) && board[nr][nc] === EMPTY) {
            marked[nr][nc] = true;
          }
        }
      }
    }
  }

  if (!hasStone) {
    return [{ row: 7, col: 7 }];
  }

  const coords: Coord[] = [];
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      if (marked[r][c]) coords.push({ row: r, col: c });
    }
  }
  return coords;
}

function scoreLocalMove(
  board: Board,
  row: number,
  col: number,
  player: Player,
  rules: RuleSet,
): { legal: boolean; isWin: boolean; score: number } {
  const opp = opponentOf(player);
  const myAnalysis = analyzeMove(board, row, col, player, rules);
  if (!myAnalysis.legal) {
    return { legal: false, isWin: false, score: -Infinity };
  }
  if (myAnalysis.isWin) {
    return { legal: true, isWin: true, score: WIN_SCORE };
  }

  const oppAnalysis = analyzeMove(board, row, col, opp, rules);

  let score = 0;
  if (oppAnalysis.isWin) score += 400_000;
  if (myAnalysis.openFours.length > 0) score += 120_000;
  if (
    myAnalysis.closedFours.length > 0 &&
    myAnalysis.openThrees.length > 0
  ) {
    score += 90_000; // 4-3 fork
  }
  if (oppAnalysis.openFours.length > 0) score += 70_000;
  if (
    oppAnalysis.closedFours.length > 0 &&
    oppAnalysis.openThrees.length > 0
  ) {
    score += 50_000;
  }

  score += myAnalysis.closedFours.length * 4_500;
  score += myAnalysis.openThrees.length * 4_000;
  score += oppAnalysis.closedFours.length * 2_800;
  score += oppAnalysis.openThrees.length * 3_500;

  // Adjacent stone / center bias
  for (const [dr, dc] of AXES) {
    for (const sign of [-1, 1]) {
      const r1 = row + sign * dr;
      const c1 = col + sign * dc;
      if (inBounds(r1, c1)) {
        if (board[r1][c1] === player) score += 80;
        else if (board[r1][c1] === opp) score += 50;
      }
    }
  }
  const distFromCenter = Math.abs(row - 7) + Math.abs(col - 7);
  score += Math.max(0, 14 - distFromCenter) * 4;

  return { legal: true, isWin: false, score };
}

function getOrderedMoves(
  board: Board,
  player: Player,
  rules: RuleSet,
  limit: number,
): ScoredMove[] {
  const candidates = getCandidateCoords(board);
  const scored: ScoredMove[] = [];
  const opp = opponentOf(player);
  const oppWinCells: ScoredMove[] = [];

  for (const { row, col } of candidates) {
    const res = scoreLocalMove(board, row, col, player, rules);
    if (!res.legal) continue;
    if (res.isWin) {
      return [{ row, col, score: WIN_SCORE }];
    }

    const oppWin = analyzeMove(board, row, col, opp, rules).isWin;
    if (oppWin) {
      oppWinCells.push({ row, col, score: res.score });
    }
    scored.push({ row, col, score: res.score });
  }

  // If opponent has immediate winning cells and we have no immediate win,
  // we must play on one of those blocking cells.
  if (oppWinCells.length > 0) {
    oppWinCells.sort((a, b) => b.score - a.score);
    return oppWinCells;
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, limit);
}

function negamax(
  board: Board,
  depth: number,
  alpha: number,
  beta: number,
  player: Player,
  rules: RuleSet,
  deadline: number,
): { score: number; move: Coord | null } {
  if (depth === 0 || Date.now() >= deadline) {
    return { score: evaluateBoard(board, player, rules), move: null };
  }

  const moves = getOrderedMoves(board, player, rules, depth >= 3 ? 10 : 12);
  if (moves.length === 0) {
    return { score: 0, move: null };
  }

  // Immediate win
  if (moves[0].score >= WIN_SCORE) {
    return {
      score: WIN_SCORE + depth,
      move: { row: moves[0].row, col: moves[0].col },
    };
  }

  let bestScore = -Infinity;
  let bestMove: Coord = { row: moves[0].row, col: moves[0].col };

  for (const m of moves) {
    board[m.row][m.col] = player;
    const child = negamax(
      board,
      depth - 1,
      -beta,
      -alpha,
      opponentOf(player),
      rules,
      deadline,
    );
    board[m.row][m.col] = EMPTY;

    const score = -child.score;
    if (score > bestScore) {
      bestScore = score;
      bestMove = { row: m.row, col: m.col };
    }
    if (score > alpha) alpha = score;
    if (alpha >= beta) break;
    if (Date.now() >= deadline) break;
  }

  return { score: bestScore, move: bestMove };
}

export function chooseAIMove(
  board: Board,
  player: Player,
  rules: RuleSet,
  difficulty: Difficulty,
): Coord | null {
  const workBoard = cloneBoard(board);
  const ordered = getOrderedMoves(workBoard, player, rules, 16);

  if (ordered.length === 0) {
    // Fallback scan entire board for any legal move
    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        if (
          workBoard[r][c] === EMPTY &&
          analyzeMove(workBoard, r, c, player, rules).legal
        ) {
          return { row: r, col: c };
        }
      }
    }
    return null;
  }

  // Immediate win or forced single defense
  if (ordered.length === 1 || ordered[0].score >= WIN_SCORE) {
    return { row: ordered[0].row, col: ordered[0].col };
  }

  const level: 'lv1' | 'lv2' | 'lv3' | 'lv4' =
    difficulty === 'easy'
      ? 'lv1'
      : difficulty === 'normal'
        ? 'lv2'
        : difficulty === 'hard'
          ? 'lv4'
          : difficulty;

  // Check Continuous Four (VCF) forced win for lv2+
  if (level !== 'lv1') {
    const vcfPlies = level === 'lv2' ? 3 : level === 'lv3' ? 7 : 9;
    const vcfMs = level === 'lv2' ? 15 : level === 'lv3' ? 30 : 45;
    const vcf = findVcfSequence(workBoard, player, rules, vcfPlies, vcfMs);
    if (vcf && vcf.length > 0) {
      return vcf[0];
    }
  }

  if (level === 'lv1') {
    if (ordered[0].score >= 120_000) {
      return { row: ordered[0].row, col: ordered[0].col };
    }
    const pool = ordered.slice(0, Math.min(4, ordered.length));
    const pick = pool[Math.floor(Math.random() * pool.length)];
    return { row: pick.row, col: pick.col };
  }

  const maxDepth = level === 'lv2' ? 2 : level === 'lv3' ? 3 : 4;
  const timeBudgetMs = level === 'lv2' ? 140 : level === 'lv3' ? 250 : 400;
  const deadline = Date.now() + timeBudgetMs;

  let bestMove: Coord = { row: ordered[0].row, col: ordered[0].col };
  const startDepth = maxDepth === 3 ? 3 : 2;
  for (let d = startDepth; d <= maxDepth; d += 2) {
    if (Date.now() >= deadline) break;
    const res = negamax(workBoard, d, -Infinity, Infinity, player, rules, deadline);
    if (res.move) {
      bestMove = res.move;
    }
    if (res.score >= WIN_SCORE) break;
  }

  return bestMove;
}
