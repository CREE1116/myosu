import { GameMode, MoveRecord, Player } from '../engine/types';
import type { CharId } from './profile';

export interface SavedMatch {
  id: string;
  playedAt: number;
  playerNickname: string;
  mode: GameMode;
  charId?: CharId;
  opponentName: string;
  opponentBadge: string;
  opponentTier?: string;
  humanPlayer: Player;
  winner: Player | null;
  isDraw: boolean;
  moves: MoveRecord[];
}

const MATCH_STORAGE_KEY = 'gomoku_renju_matches_v1';
const MAX_SAVED_MATCHES = 50;

export function getSavedMatches(): SavedMatch[] {
  try {
    const raw = localStorage.getItem(MATCH_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as SavedMatch[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveMatchRecord(
  entry: Omit<SavedMatch, 'id' | 'playedAt'>,
): SavedMatch {
  const full: SavedMatch = {
    ...entry,
    id: `${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
    playedAt: Date.now(),
  };
  try {
    const list = getSavedMatches();
    list.unshift(full);
    localStorage.setItem(
      MATCH_STORAGE_KEY,
      JSON.stringify(list.slice(0, MAX_SAVED_MATCHES)),
    );
  } catch {
    // Ignore storage quota errors
  }
  return full;
}

export function deleteSavedMatch(id: string): void {
  try {
    const filtered = getSavedMatches().filter((m) => m.id !== id);
    localStorage.setItem(MATCH_STORAGE_KEY, JSON.stringify(filtered));
  } catch {
    // Ignore storage errors
  }
}
