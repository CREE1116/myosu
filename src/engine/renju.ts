import {
  BLACK,
  BOARD_SIZE,
  Board,
  BoardThreat,
  Coord,
  EMPTY,
  ForbiddenCell,
  ForbiddenReason,
  FourPattern,
  MoveAnalysis,
  OpenThreePattern,
  Player,
  RuleSet,
  WHITE,
  WinResult,
} from './types';

export const AXES: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [1, 0],
  [1, 1],
  [1, -1],
];

export function createEmptyBoard(): Board {
  return Array.from({ length: BOARD_SIZE }, () =>
    Array<typeof EMPTY>(BOARD_SIZE).fill(EMPTY),
  );
}

export function cloneBoard(board: Board): Board {
  return board.map((row) => [...row]);
}

export function inBounds(row: number, col: number): boolean {
  return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE;
}

export function opponentOf(player: Player): Player {
  return player === BLACK ? WHITE : BLACK;
}

function coordKey(coords: Coord[]): string {
  return [...coords]
    .sort((a, b) => (a.row !== b.row ? a.row - b.row : a.col - b.col))
    .map((c) => `${c.row},${c.col}`)
    .join('|');
}

function getContiguousLine(
  board: Board,
  row: number,
  col: number,
  dr: number,
  dc: number,
  player: Player,
): Coord[] {
  const cells: Coord[] = [{ row, col }];

  for (let step = 1; step < BOARD_SIZE; step++) {
    const r = row - step * dr;
    const c = col - step * dc;
    if (!inBounds(r, c) || board[r][c] !== player) break;
    cells.unshift({ row: r, col: c });
  }

  for (let step = 1; step < BOARD_SIZE; step++) {
    const r = row + step * dr;
    const c = col + step * dc;
    if (!inBounds(r, c) || board[r][c] !== player) break;
    cells.push({ row: r, col: c });
  }

  return cells;
}

function findWinLineOnPlaced(
  board: Board,
  row: number,
  col: number,
  player: Player,
  rules: RuleSet,
): Coord[] | null {
  const exactOnly = rules === 'renju' && player === BLACK;

  for (const [dr, dc] of AXES) {
    const cells = getContiguousLine(board, row, col, dr, dc, player);
    if (exactOnly ? cells.length === 5 : cells.length >= 5) {
      return cells;
    }
  }
  return null;
}

function hasOverlineOnPlaced(
  board: Board,
  row: number,
  col: number,
  player: Player,
  rules: RuleSet,
): boolean {
  if (rules !== 'renju' || player !== BLACK) return false;

  for (const [dr, dc] of AXES) {
    const cells = getContiguousLine(board, row, col, dr, dc, player);
    if (cells.length >= 6) return true;
  }
  return false;
}

/**
 * Finds all distinct Four patterns containing `(row, col)` along `(dr, dc)`.
 * Assumes `board[row][col] === player` already.
 */
function analyzeFoursOnAxis(
  board: Board,
  row: number,
  col: number,
  dr: number,
  dc: number,
  player: Player,
  rules: RuleSet,
): FourPattern[] {
  const exactOnly = rules === 'renju' && player === BLACK;
  const groups = new Map<
    string,
    { stones: Coord[]; completingCells: Coord[] }
  >();

  for (let d = -4; d <= 4; d++) {
    if (d === 0) continue;
    const nr = row + d * dr;
    const nc = col + d * dc;
    if (!inBounds(nr, nc) || board[nr][nc] !== EMPTY) continue;

    board[nr][nc] = player;
    const line = getContiguousLine(board, nr, nc, dr, dc, player);
    board[nr][nc] = EMPTY;

    const includesOrigin = line.some((c) => c.row === row && c.col === col);
    if (!includesOrigin) continue;

    const validLength = exactOnly ? line.length === 5 : line.length >= 5;
    if (!validLength) continue;

    const baseStones = line.filter((c) => !(c.row === nr && c.col === nc));
    const key = coordKey(baseStones);
    const existing = groups.get(key);
    if (existing) {
      existing.completingCells.push({ row: nr, col: nc });
    } else {
      groups.set(key, {
        stones: baseStones,
        completingCells: [{ row: nr, col: nc }],
      });
    }
  }

  return Array.from(groups.values()).map((g) => ({
    stones: g.stones,
    completingCells: g.completingCells,
    isOpen: g.completingCells.length >= 2,
  }));
}

function hasNearbyStone(
  board: Board,
  row: number,
  col: number,
  player: Player,
  radius: number,
): boolean {
  for (const [dr, dc] of AXES) {
    for (let d = -radius; d <= radius; d++) {
      if (d === 0) continue;
      const r = row + d * dr;
      const c = col + d * dc;
      if (inBounds(r, c) && board[r][c] === player) return true;
    }
  }
  return false;
}

/**
 * Checks whether `(row, col)` is forbidden for Black under Renju rules,
 * assuming `board[row][col]` is currently EMPTY.
 */
function getBlackForbiddenReason(
  board: Board,
  row: number,
  col: number,
  rules: RuleSet,
): ForbiddenReason | null {
  if (rules !== 'renju') return null;
  if (!hasNearbyStone(board, row, col, BLACK, 2)) return null;

  board[row][col] = BLACK;
  try {
    if (findWinLineOnPlaced(board, row, col, BLACK, rules)) {
      return null; // Exact 5 overrides all forbidden moves in Renju
    }

    if (hasOverlineOnPlaced(board, row, col, BLACK, rules)) {
      return 'OVERLINE';
    }

    let fourCount = 0;
    for (const [dr, dc] of AXES) {
      fourCount += analyzeFoursOnAxis(board, row, col, dr, dc, BLACK, rules).length;
      if (fourCount >= 2) return 'FOUR_FOUR';
    }

    let openThreeCount = 0;
    for (const [dr, dc] of AXES) {
      const axisFours = analyzeFoursOnAxis(board, row, col, dr, dc, BLACK, rules);
      if (axisFours.length > 0) continue;

      openThreeCount += analyzeOpenThreesOnAxis(
        board,
        row,
        col,
        dr,
        dc,
        BLACK,
        rules,
      ).length;
      if (openThreeCount >= 2) return 'THREE_THREE';
    }

    return null;
  } finally {
    board[row][col] = EMPTY;
  }
}

/**
 * Finds all distinct Open Three patterns containing `(row, col)` along `(dr, dc)`.
 * Assumes `board[row][col] === player` already.
 */
function analyzeOpenThreesOnAxis(
  board: Board,
  row: number,
  col: number,
  dr: number,
  dc: number,
  player: Player,
  rules: RuleSet,
): OpenThreePattern[] {
  const groups = new Map<
    string,
    { stones: Coord[]; extendingCells: Coord[] }
  >();

  for (let d = -3; d <= 3; d++) {
    if (d === 0) continue;
    const qr = row + d * dr;
    const qc = col + d * dc;
    if (!inBounds(qr, qc) || board[qr][qc] !== EMPTY) continue;

    board[qr][qc] = player;
    const foursAfterQ = analyzeFoursOnAxis(board, qr, qc, dr, dc, player, rules);
    board[qr][qc] = EMPTY;

    let matchingOpenFour: FourPattern | null = null;
    for (const four of foursAfterQ) {
      if (!four.isOpen) continue;
      if (four.stones.some((c) => c.row === row && c.col === col)) {
        matchingOpenFour = four;
        break;
      }
    }
    if (!matchingOpenFour) continue;

    // For Renju Black, the move q that turns the 3 into an Open Four must itself be legal
    if (rules === 'renju' && player === BLACK) {
      if (getBlackForbiddenReason(board, qr, qc, rules) !== null) {
        continue;
      }
    }

    const threeStones = matchingOpenFour.stones.filter(
      (c) => !(c.row === qr && c.col === qc),
    );
    if (threeStones.length !== 3) continue;

    const key = coordKey(threeStones);
    const existing = groups.get(key);
    if (existing) {
      existing.extendingCells.push({ row: qr, col: qc });
    } else {
      groups.set(key, {
        stones: threeStones,
        extendingCells: [{ row: qr, col: qc }],
      });
    }
  }

  return Array.from(groups.values());
}

/**
 * Full analysis of placing `player` at `(row, col)` on `board`.
 * Does not mutate `board`.
 */
export function analyzeMove(
  board: Board,
  row: number,
  col: number,
  player: Player,
  rules: RuleSet,
): MoveAnalysis {
  if (!inBounds(row, col) || board[row][col] !== EMPTY) {
    return {
      row,
      col,
      player,
      legal: false,
      forbiddenReason: null,
      isWin: false,
      winLine: null,
      openFours: [],
      closedFours: [],
      openThrees: [],
    };
  }

  board[row][col] = player;
  try {
    const winLine = findWinLineOnPlaced(board, row, col, player, rules);
    if (winLine) {
      return {
        row,
        col,
        player,
        legal: true,
        forbiddenReason: null,
        isWin: true,
        winLine,
        openFours: [],
        closedFours: [],
        openThrees: [],
      };
    }

    if (hasOverlineOnPlaced(board, row, col, player, rules)) {
      return {
        row,
        col,
        player,
        legal: false,
        forbiddenReason: 'OVERLINE',
        isWin: false,
        winLine: null,
        openFours: [],
        closedFours: [],
        openThrees: [],
      };
    }

    const openFours: FourPattern[] = [];
    const closedFours: FourPattern[] = [];
    const openThrees: OpenThreePattern[] = [];

    for (const [dr, dc] of AXES) {
      const axisFours = analyzeFoursOnAxis(board, row, col, dr, dc, player, rules);
      for (const f of axisFours) {
        if (f.isOpen) openFours.push(f);
        else closedFours.push(f);
      }

      if (axisFours.length === 0) {
        const axisThrees = analyzeOpenThreesOnAxis(
          board,
          row,
          col,
          dr,
          dc,
          player,
          rules,
        );
        openThrees.push(...axisThrees);
      }
    }

    let forbiddenReason: ForbiddenReason | null = null;
    if (rules === 'renju' && player === BLACK) {
      if (openFours.length + closedFours.length >= 2) {
        forbiddenReason = 'FOUR_FOUR';
      } else if (openThrees.length >= 2) {
        forbiddenReason = 'THREE_THREE';
      }
    }

    return {
      row,
      col,
      player,
      legal: forbiddenReason === null,
      forbiddenReason,
      isWin: false,
      winLine: null,
      openFours,
      closedFours,
      openThrees,
    };
  } finally {
    board[row][col] = EMPTY;
  }
}

export function getForbiddenMoves(
  board: Board,
  rules: RuleSet,
): ForbiddenCell[] {
  if (rules !== 'renju') return [];

  const result: ForbiddenCell[] = [];
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      if (board[r][c] !== EMPTY) continue;
      const reason = getBlackForbiddenReason(board, r, c, rules);
      if (reason !== null) {
        result.push({ row: r, col: c, reason });
      }
    }
  }
  return result;
}

export function checkWinner(
  board: Board,
  lastMove: Coord | null,
  rules: RuleSet,
): WinResult | null {
  if (!lastMove) return null;
  const { row, col } = lastMove;
  if (!inBounds(row, col)) return null;
  const stone = board[row][col];
  if (stone === EMPTY) return null;

  const winLine = findWinLineOnPlaced(board, row, col, stone, rules);
  if (!winLine) return null;
  return { winner: stone, line: winLine };
}

/**
 * Scans the current board for all active Open Fours, Closed Fours, and Open Threes
 * for both Black and White.
 */
export function scanBoardThreats(board: Board, rules: RuleSet): BoardThreat[] {
  const seen = new Set<string>();
  const threats: BoardThreat[] = [];

  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      const stone = board[r][c];
      if (stone === EMPTY) continue;

      for (const [dr, dc] of AXES) {
        const fours = analyzeFoursOnAxis(board, r, c, dr, dc, stone, rules);
        for (const f of fours) {
          const type = f.isOpen ? 'OPEN_FOUR' : 'CLOSED_FOUR';
          const id = `${stone}:${type}:${coordKey(f.stones)}`;
          if (seen.has(id)) continue;
          seen.add(id);
          threats.push({
            id,
            player: stone,
            type,
            stones: f.stones,
            targetCells: f.completingCells,
          });
        }

        if (fours.length === 0) {
          const threes = analyzeOpenThreesOnAxis(
            board,
            r,
            c,
            dr,
            dc,
            stone,
            rules,
          );
          for (const t of threes) {
            const id = `${stone}:OPEN_THREE:${coordKey(t.stones)}`;
            if (seen.has(id)) continue;
            seen.add(id);
            threats.push({
              id,
              player: stone,
              type: 'OPEN_THREE',
              stones: t.stones,
              targetCells: t.extendingCells,
            });
          }
        }
      }
    }
  }

  return threats;
}

/**
 * Returns true if `player` has at least one legal empty intersection on `board`.
 */
export function hasAnyLegalMove(
  board: Board,
  player: Player,
  rules: RuleSet,
): boolean {
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      if (board[r][c] !== EMPTY) continue;
      if (player === WHITE || !getBlackForbiddenReason(board, r, c, rules)) {
        return true;
      }
    }
  }
  return false;
}

export function formatCoord(row: number, col: number): string {
  const colLabel = String.fromCharCode(65 + col);
  const rowLabel = BOARD_SIZE - row;
  return `${colLabel}${rowLabel}`;
}
