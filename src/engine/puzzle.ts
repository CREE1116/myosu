// 묘수풀이: win by making a four on every move (VCF). The cat defends by the only reply it has.

import { analyzeMove, createEmptyBoard } from './renju';
import { findVcfSequence, proveVcf } from './vcf';
import { BLACK, Board, Coord, Player, RuleSet, WHITE } from './types';
import { boardAt } from './review';
import type { MoveRecord } from './types';

export type PuzzleTheme = 'chain' | 'trap' | 'long' | 'mine';

export const THEMES: Array<{ id: PuzzleTheme; title: string; desc: string }> = [
  { id: 'chain', title: '넷 이어 가기', desc: '넷을 세 번 이어서 끝내요. 첫 넷이 어디인지가 묘수예요.' },
  { id: 'trap', title: '금수 함정', desc: '흑이 막아야 할 자리를 금수로 만들어 버려요.' },
  { id: 'long', title: '긴 수순', desc: '넷을 네 번 넘게 이어 가는 어려운 문제예요.' },
  { id: 'mine', title: '내 기보에서', desc: '복기에서 찾은, 내가 놓친 필승이에요.' },
];

export interface Puzzle {
  id: string;
  theme: PuzzleTheme;
  /** Stones as row*15+col. */
  black: number[];
  white: number[];
  attacker: Player;
  /** Attacker moves in the shortest known solution (including the five). */
  moves: number;
  /** A known solution, attacker and defender alternating, as row*15+col. */
  line: number[];
}

export function puzzleBoard(p: Puzzle): Board {
  const b = createEmptyBoard();
  for (const k of p.black) b[Math.floor(k / 15)][k % 15] = BLACK;
  for (const k of p.white) b[Math.floor(k / 15)][k % 15] = WHITE;
  return b;
}

export type AttackCheck = 'win' | 'four' | 'notFour' | 'illegal';

/** Is this attacker move allowed in a four-sequence puzzle? */
export function checkAttack(board: Board, at: Coord, attacker: Player, rules: RuleSet): AttackCheck {
  const a = analyzeMove(board, at.row, at.col, attacker, rules);
  if (!a.legal) return 'illegal';
  if (a.isWin) return 'win';
  return a.openFours.length + a.closedFours.length > 0 ? 'four' : 'notFour';
}

/** After the defender's reply, can the attacker still force a win? */
export function stillWinning(board: Board, attacker: Player, rules: RuleSet): boolean {
  // Generous budget: a correct but long line must never be judged lost.
  return proveVcf(board, attacker, rules, 17, 30000, 1500).line !== null;
}

/** The position just before a missed win in one of the player's own games, as a puzzle. */
export function puzzleFromGame(moves: MoveRecord[], index: number, attacker: Player, rules: RuleSet): Puzzle | null {
  const board = boardAt(moves, index);
  const win = findVcfSequence(board, attacker, rules, 15, 250);
  if (!win || win.length < 3) return null; // a lone five is a slip, not a puzzle
  const black: number[] = [];
  const white: number[] = [];
  for (const m of moves.slice(0, index)) (m.player === BLACK ? black : white).push(m.row * 15 + m.col);
  // Same position, same id: reviewing a game twice doesn't add it twice.
  let h = 5381;
  for (const k of [...black.sort((x, y) => x - y), -1, ...white.sort((x, y) => x - y)]) h = (h * 33 + k) | 0;
  const id = `mine-${(h >>> 0).toString(36)}`;
  return { id, theme: 'mine', black, white, attacker, moves: Math.ceil(win.length / 2), line: win.map((c) => c.row * 15 + c.col) };
}
