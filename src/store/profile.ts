import { AI_CHARACTERS } from '../engine/ai';
import { AiCharacter } from '../engine/types';

export type CharId = AiCharacter['id'];

export interface CharRecord {
  wins: number;
  losses: number;
  draws: number;
  bestWinMoves?: number;
}

export interface PlayerProfile {
  nickname: string;
  records: Record<CharId, CharRecord>;
  streak: number;
  bestStreak: number;
  updatedAt: number;
}

interface StorageState {
  currentNickname: string | null;
  profiles: Record<string, PlayerProfile>;
}

const STORAGE_KEY = 'gomoku_renju_profiles_v1';

function emptyRecords(): Record<CharId, CharRecord> {
  return {
    lv1: { wins: 0, losses: 0, draws: 0 },
    lv2: { wins: 0, losses: 0, draws: 0 },
    lv3: { wins: 0, losses: 0, draws: 0 },
    lv4: { wins: 0, losses: 0, draws: 0 },
  };
}

function loadState(): StorageState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { currentNickname: null, profiles: {} };
    const parsed = JSON.parse(raw) as StorageState;
    return {
      currentNickname: parsed.currentNickname ?? null,
      profiles: parsed.profiles ?? {},
    };
  } catch {
    return { currentNickname: null, profiles: {} };
  }
}

function saveState(state: StorageState): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // Ignore storage quota errors
  }
}

export function getCurrentUser(): PlayerProfile | null {
  const state = loadState();
  if (!state.currentNickname) return null;
  return state.profiles[state.currentNickname] ?? null;
}

export function loginUser(nickname: string): PlayerProfile {
  const clean = nickname.trim().slice(0, 16);
  const state = loadState();
  if (!state.profiles[clean]) {
    state.profiles[clean] = {
      nickname: clean,
      records: emptyRecords(),
      streak: 0,
      bestStreak: 0,
      updatedAt: Date.now(),
    };
  }
  state.currentNickname = clean;
  saveState(state);
  return state.profiles[clean];
}

export function logoutUser(): void {
  const state = loadState();
  state.currentNickname = null;
  saveState(state);
}

export function recordAiMatchResult(
  charId: CharId,
  outcome: 'win' | 'loss' | 'draw',
  moveCount: number,
): PlayerProfile | null {
  const state = loadState();
  if (!state.currentNickname) return null;
  const profile = state.profiles[state.currentNickname];
  if (!profile) return null;

  const rec = profile.records[charId] ?? { wins: 0, losses: 0, draws: 0 };
  if (outcome === 'win') {
    rec.wins++;
    if (!rec.bestWinMoves || moveCount < rec.bestWinMoves) {
      rec.bestWinMoves = moveCount;
    }
    profile.streak++;
    if (profile.streak > profile.bestStreak) {
      profile.bestStreak = profile.streak;
    }
  } else if (outcome === 'loss') {
    rec.losses++;
    profile.streak = 0;
  } else {
    rec.draws++;
  }

  profile.records[charId] = rec;
  profile.updatedAt = Date.now();
  saveState(state);
  return profile;
}

export function summarizeProfile(profile: PlayerProfile): {
  wins: number;
  losses: number;
  draws: number;
  total: number;
  winRate: number;
  clearedCount: number;
} {
  let wins = 0;
  let losses = 0;
  let draws = 0;
  let clearedCount = 0;
  for (const ch of AI_CHARACTERS) {
    const r = profile.records[ch.id] ?? { wins: 0, losses: 0, draws: 0 };
    wins += r.wins;
    losses += r.losses;
    draws += r.draws;
    if (r.wins > 0) clearedCount++;
  }
  const total = wins + losses + draws;
  const winRate = total > 0 ? Math.round((wins / total) * 100) : 0;
  return { wins, losses, draws, total, winRate, clearedCount };
}

export function getLeaderboard(): Array<
  PlayerProfile & ReturnType<typeof summarizeProfile>
> {
  const state = loadState();
  return Object.values(state.profiles)
    .map((p) => ({ ...p, ...summarizeProfile(p) }))
    .sort((a, b) => {
      if (b.clearedCount !== a.clearedCount) return b.clearedCount - a.clearedCount;
      if (b.wins !== a.wins) return b.wins - a.wins;
      if (b.winRate !== a.winRate) return b.winRate - a.winRate;
      return b.bestStreak - a.bestStreak;
    });
}
