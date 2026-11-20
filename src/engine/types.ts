export const BOARD_SIZE = 15;

export const EMPTY = 0 as const;
export const BLACK = 1 as const;
export const WHITE = 2 as const;

export type Stone = typeof EMPTY | typeof BLACK | typeof WHITE;
export type Player = typeof BLACK | typeof WHITE;
export type Board = Stone[][];

export type RuleSet = 'renju' | 'freestyle';
export type GameMode = 'pvp' | 'ai';
export type Difficulty = 'easy' | 'normal' | 'hard' | 'lv1' | 'lv2' | 'lv3' | 'lv4';

export interface AiCharacter {
  id: 'lv1' | 'lv2' | 'lv3' | 'lv4';
  name: string;
  tier: string;
  badge: string;
  desc: string;
  greeting: string;
}

export interface Coord {
  row: number;
  col: number;
}

export type ForbiddenReason = 'THREE_THREE' | 'FOUR_FOUR' | 'OVERLINE';

export interface FourPattern {
  stones: Coord[];
  completingCells: Coord[];
  isOpen: boolean;
}

export interface OpenThreePattern {
  stones: Coord[];
  extendingCells: Coord[];
}

export interface MoveAnalysis {
  row: number;
  col: number;
  player: Player;
  legal: boolean;
  forbiddenReason: ForbiddenReason | null;
  isWin: boolean;
  winLine: Coord[] | null;
  openFours: FourPattern[];
  closedFours: FourPattern[];
  openThrees: OpenThreePattern[];
}

export interface ForbiddenCell {
  row: number;
  col: number;
  reason: ForbiddenReason;
}

export type ThreatType = 'OPEN_FOUR' | 'CLOSED_FOUR' | 'OPEN_THREE';

export interface BoardThreat {
  id: string;
  player: Player;
  type: ThreatType;
  stones: Coord[];
  targetCells: Coord[];
}

export interface MoveRecord {
  row: number;
  col: number;
  player: Player;
  moveNumber: number;
  tag?: string;
  evalScore?: number; // normalized in [-1, 1] from Black's perspective, averaged with the previous ply
  rawEval?: number; // single-position eval; swings with whose turn it is
}

export interface WinResult {
  winner: Player;
  line: Coord[];
}
