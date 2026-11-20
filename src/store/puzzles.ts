import type { Puzzle } from '../engine/puzzle';
import { PUZZLES } from '../engine/puzzleSet';

const SOLVED_KEY = 'myosu_puzzles_solved';
const MINE_KEY = 'myosu_puzzles_mine';
const MAX_MINE = 20;

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function write(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage full or blocked: progress just isn't kept.
  }
}

/** The built-in set (easy to hard), then positions from the player's own games, newest first. */
export function allPuzzles(): Puzzle[] {
  return [...PUZZLES, ...read<Puzzle[]>(MINE_KEY, [])];
}

export function solvedIds(): Set<string> {
  return new Set(read<string[]>(SOLVED_KEY, []));
}

export function markSolved(id: string): void {
  const s = solvedIds();
  s.add(id);
  write(SOLVED_KEY, [...s]);
}

/** Keep a missed win from the player's own game as a puzzle. Returns true if it was new. */
export function addMyPuzzle(p: Puzzle): boolean {
  const mine = read<Puzzle[]>(MINE_KEY, []);
  if (mine.some((m) => m.id === p.id)) return false;
  write(MINE_KEY, [p, ...mine].slice(0, MAX_MINE));
  return true;
}
