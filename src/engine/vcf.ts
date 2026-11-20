import { analyzeMove, cloneBoard, inBounds, opponentOf } from './renju';
import { BOARD_SIZE, Board, Coord, EMPTY, Player, RuleSet } from './types';

interface VcfBudget {
  nodes: number;
  maxNodes: number;
  deadline: number;
}

function getImmediateWinCells(
  board: Board,
  player: Player,
  rules: RuleSet,
): Coord[] {
  const wins: Coord[] = [];
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      if (board[r][c] !== EMPTY) continue;
      if (analyzeMove(board, r, c, player, rules).isWin) {
        wins.push({ row: r, col: c });
      }
    }
  }
  return wins;
}

function getNearbyEmptyCells(board: Board): Coord[] {
  const marked = Array.from({ length: BOARD_SIZE }, () =>
    Array<boolean>(BOARD_SIZE).fill(false),
  );
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      if (board[r][c] === EMPTY) continue;
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

  const cells: Coord[] = [];
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      if (marked[r][c]) cells.push({ row: r, col: c });
    }
  }
  return cells;
}

function searchVcf(
  board: Board,
  attacker: Player,
  defender: Player,
  rules: RuleSet,
  maxPlies: number,
  budget: VcfBudget,
): Coord[] | null {
  if (maxPlies <= 0 || budget.nodes >= budget.maxNodes || Date.now() >= budget.deadline) {
    return null;
  }
  budget.nodes++;

  const immediateWins = getImmediateWinCells(board, attacker, rules);
  if (immediateWins.length > 0) {
    return [immediateWins[0]];
  }

  if (maxPlies < 3) return null;

  // If defender already has an immediate 5, attacker cannot force a multi-move VCF
  const oppWins = getImmediateWinCells(board, defender, rules);
  if (oppWins.length > 0) {
    return null;
  }

  const candidates = getNearbyEmptyCells(board);
  const openFourMoves: Array<{
    move: Coord;
    completing: Coord[];
  }> = [];
  const closedFourMoves: Array<{
    move: Coord;
    block: Coord;
  }> = [];

  for (const cell of candidates) {
    const analysis = analyzeMove(board, cell.row, cell.col, attacker, rules);
    if (!analysis.legal) continue;

    if (analysis.openFours.length > 0) {
      openFourMoves.push({
        move: cell,
        completing: analysis.openFours[0].completingCells,
      });
    } else if (analysis.closedFours.length >= 2) {
      // Legal double four (White or Freestyle)
      openFourMoves.push({
        move: cell,
        completing: [
          analysis.closedFours[0].completingCells[0],
          analysis.closedFours[1].completingCells[0],
        ],
      });
    } else if (analysis.closedFours.length === 1) {
      closedFourMoves.push({
        move: cell,
        block: analysis.closedFours[0].completingCells[0],
      });
    }
  }

  // Check winning Open Fours / Double Fours first (verified 3-ply or 1-ply trap)
  for (const item of openFourMoves) {
    if (item.completing.length < 2) continue;
    const [c0, c1] = item.completing;
    if (c0.row === c1.row && c0.col === c1.col) continue;

    board[item.move.row][item.move.col] = attacker;
    const def0 = analyzeMove(board, c0.row, c0.col, defender, rules);
    const def1 = analyzeMove(board, c1.row, c1.col, defender, rules);

    if (def0.isWin || def1.isWin) {
      board[item.move.row][item.move.col] = EMPTY;
      continue;
    }

    if (!def0.legal && !def1.legal) {
      board[item.move.row][item.move.col] = EMPTY;
      return [item.move];
    }

    let ok0 = !def0.legal;
    if (def0.legal) {
      board[c0.row][c0.col] = defender;
      ok0 = analyzeMove(board, c1.row, c1.col, attacker, rules).isWin;
      board[c0.row][c0.col] = EMPTY;
    }

    let ok1 = !def1.legal;
    if (def1.legal) {
      board[c1.row][c1.col] = defender;
      ok1 = analyzeMove(board, c0.row, c0.col, attacker, rules).isWin;
      board[c1.row][c1.col] = EMPTY;
    }

    board[item.move.row][item.move.col] = EMPTY;

    if (ok0 && ok1) {
      return !def0.legal ? [item.move, c1, c0] : [item.move, c0, c1];
    }
  }

  // Recurse on Closed Fours
  for (const item of closedFourMoves) {
    if (budget.nodes >= budget.maxNodes || Date.now() >= budget.deadline) {
      return null;
    }

    board[item.move.row][item.move.col] = attacker;
    const defAnalysis = analyzeMove(
      board,
      item.block.row,
      item.block.col,
      defender,
      rules,
    );

    // Forbidden move trap: defender cannot legally block!
    if (!defAnalysis.legal) {
      board[item.move.row][item.move.col] = EMPTY;
      return [item.move];
    }

    // Defender counter-wins on the block square
    if (defAnalysis.isWin) {
      board[item.move.row][item.move.col] = EMPTY;
      continue;
    }

    board[item.block.row][item.block.col] = defender;
    const tail = searchVcf(
      board,
      attacker,
      defender,
      rules,
      maxPlies - 2,
      budget,
    );
    board[item.block.row][item.block.col] = EMPTY;
    board[item.move.row][item.move.col] = EMPTY;

    if (tail) {
      return [item.move, item.block, ...tail];
    }
  }

  return null;
}

/**
 * Finds a forced Victory by Continuous Fours (연속 사) sequence for `attacker`,
 * returning alternating [attacker, defender, attacker, ...] coordinates ending in a win.
 */
export function findVcfSequence(
  board: Board,
  attacker: Player,
  rules: RuleSet,
  maxPlies = 9,
  timeBudgetMs = 35,
): Coord[] | null {
  const work = cloneBoard(board);
  const defender = opponentOf(attacker);
  const budget: VcfBudget = {
    nodes: 0,
    maxNodes: 2000,
    deadline: Date.now() + timeBudgetMs,
  };
  return searchVcf(work, attacker, defender, rules, maxPlies, budget);
}

/**
 * Like findVcfSequence, but also says whether the search finished: `complete` with a null line
 * means no four-sequence exists within `maxPlies`; an unfinished search proves nothing.
 */
export function proveVcf(
  board: Board,
  attacker: Player,
  rules: RuleSet,
  maxPlies = 13,
  maxNodes = 6000,
  timeBudgetMs = 250,
): { line: Coord[] | null; complete: boolean } {
  const budget: VcfBudget = { nodes: 0, maxNodes, deadline: Date.now() + timeBudgetMs };
  const line = searchVcf(cloneBoard(board), attacker, opponentOf(attacker), rules, maxPlies, budget);
  return { line, complete: line !== null || (budget.nodes < maxNodes && Date.now() < budget.deadline) };
}

/** Empty cells within two of any stone: where a sensible move can be. */
export function nearbyEmptyCells(board: Board): Coord[] {
  return getNearbyEmptyCells(board);
}
