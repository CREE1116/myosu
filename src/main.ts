import './style.css';
import { playForbiddenSound, playMeowSound, playStoneSound, playWinSound } from './audio/sound';
import { AI_CHARACTERS, chooseAIMove, evaluatePositionAdvantage } from './engine/ai';
import {
  CheckpointMoveResult,
  D4Weights,
  chooseCheckpointMoveAsync,
  evaluateD4OverlapPattern,
  parseD4Weights,
} from './engine/nnModel';
import { CatContext, CatVoice, VOICES, catForbiddenLine, catLines, catQuietMood, pickLine } from './engine/catDialogue';
import { STRENGTH, forcedMove, sampleCandidate } from './engine/strength';
import {
  analyzeMove,
  checkWinner,
  createEmptyBoard,
  formatCoord,
  getForbiddenMoves,
  hasAnyLegalMove,
  opponentOf,
  scanBoardThreats,
} from './engine/renju';
import { findVcfSequence } from './engine/vcf';
import { Analyzer, TurningPoint, boardAt, findTurningPoints } from './engine/review';
import { Puzzle, THEMES, checkAttack, puzzleBoard, puzzleFromGame, stillWinning } from './engine/puzzle';
import { addMyPuzzle, allPuzzles, markSolved, solvedIds } from './store/puzzles';
import { PATCH_NOTES, hasUnseenNews, markNewsSeen } from './data/patchNotes';
import {
  BLACK,
  BOARD_SIZE,
  Board,
  BoardThreat,
  Coord,
  EMPTY,
  ForbiddenCell,
  ForbiddenReason,
  GameMode,
  MoveRecord,
  Player,
  WHITE,
  WinResult,
} from './engine/types';
import {
  CharId,
  getCurrentUser,
  loginUser,
  logoutUser,
  recordAiMatchResult,
  summarizeProfile,
} from './store/profile';
import { getSavedMatches, SavedMatch, saveMatchRecord } from './store/replays';
import { CatId, CatMood, CatView, catSpriteUrl } from './ui/cat';
import {
  RULES,
  buildEvalChartSvg,
  buildStaticGridSvg,
  coordKey,
  createSvgElement,
  forbiddenLabel,
  getMoveTagOnBoard,
  playerLabel,
} from './ui/replayUtils';

// Records are kept per character id in the profile store.
const CHAR_ID: Record<CatId, CharId> = { kkamang: 'lv4', hayang: 'lv2' };
const CAT_NAME: Record<CatId, string> = { kkamang: '까망이', hayang: '하양이' };
const OPPONENT_KEY = 'myosu_opponent';
/** Takebacks per game: the easy cat is more forgiving. */
const UNDO_LIMIT: Record<CatId, number> = { kkamang: 1, hayang: 3 };
/** 한 수 추천 per game: the other cat drops by with its best move. */
const HINTS_PER_GAME = 3;
const otherCat = (c: CatId): CatId => (c === 'kkamang' ? 'hayang' : 'kkamang');

/** A tiny picture of a puzzle position for the list. */
function miniBoardSvg(p: Puzzle): string {
  const dot = (k: number, fill: string) =>
    `<circle cx="${(k % 15) + 0.5}" cy="${Math.floor(k / 15) + 0.5}" r="0.45" fill="${fill}"/>`;
  return `<svg class="puzzle-mini" viewBox="0 0 15 15" aria-hidden="true"><rect width="15" height="15" rx="0.6" fill="var(--wood)"/>${p.black.map((k) => dot(k, '#1d1719')).join('')}${p.white.map((k) => dot(k, '#fbf7ef')).join('')}</svg>`;
}

function loadOpponent(): CatId {
  try {
    return localStorage.getItem(OPPONENT_KEY) === 'hayang' ? 'hayang' : 'kkamang';
  } catch {
    return 'kkamang';
  }
}

function formatRecord(wins: number, draws: number, losses: number): string {
  return draws > 0
    ? `${wins}승 ${draws}무 ${losses}패`
    : `${wins}승 ${losses}패`;
}

interface RuleSlide {
  title: string;
  desc: string;
  bullets: string[];
  diagram?: string[];
}

const RULE_SLIDES: RuleSlide[] = [
  {
    title: '렌주 룰 기본 승패',
    desc: '15×15 바둑판 교차점에 <strong>흑(선공)</strong>과 <strong>백(후공)</strong>이 번갈아 두어 먼저 5목을 완성하면 승리합니다.',
    bullets: [
      '<strong>흑 (선공)</strong>: 먼저 두는 유리함을 보완하기 위해 금수(3·3, 4·4, 장목)가 적용됩니다.',
      '<strong>백 (후공)</strong>: 금수 제한 없이 5목 이상을 완성하면 승리합니다.',
    ],
  },
  {
    title: '흑 금수 ① — 쌍삼 (3·3)',
    desc: '흑이 한 수로 <strong>열린 3</strong>을 동시에 2개 이상 만들면 금수(✕ 33)로 착수할 수 없습니다.',
    bullets: [
      '<strong>열린 3</strong>: 양쪽이 막히지 않아 다음 수에 <strong>열린 4</strong>로 이어질 수 있는 3목(띄엄삼 포함)입니다.',
      '<strong>거짓 삼</strong>: 연장할 자리가 막히거나 금수여서 열린 4를 못 만들면 열린 3으로 치지 않습니다.',
    ],
    diagram: ['...b...', '...b...', '.bbX...', '.......', '.......'],
  },
  {
    title: '흑 금수 ② — 쌍사 (4·4) & 장목 (6+)',
    desc: '흑이 한 수로 <strong>4목 2개 이상</strong>을 만들거나 <strong>6목 이상</strong>을 이으면 금수입니다.',
    bullets: [
      '<strong>쌍사 (✕ 44)</strong>: 열린 4, 닫힌 4, 한 줄 양쪽 4목 모두 2개 이상이면 금수입니다.',
      '<strong>장목 (✕ 6+)</strong>: 흑은 6목 이상 착수가 금지되며, 백은 6목 이상도 승리합니다.',
    ],
    diagram: ['...b...', '...b...', '...b...', 'bbbX...', '.......'],
  },
  {
    title: '허용 예외 — 4·3 & 5목 우선',
    desc: '아래 두 경우는 흑에게도 <strong>합법 착수</strong>로 인정됩니다.',
    bullets: [
      '<strong>4·3 공격</strong>: 4목 1개와 열린 3목 1개를 동시에 만드는 수는 허용됩니다.',
      '<strong>5목 우선</strong>: 금수 자리라도 정확히 <strong>5목</strong>을 완성하면 즉시 흑이 승리합니다.',
    ],
    diagram: ['...b...', '...b...', 'bbbO...', '.......', '.......'],
  },
  {
    title: '화면 표시 안내',
    desc: '바둑판 위의 표시로 3목·4목 위협과 금수를 바로 확인할 수 있습니다.',
    bullets: [
      '<strong>✕ 33 / 44 / 6+</strong>: 흑 차례에 둘 수 없는 금수 위치입니다.',
      '<strong>주황/붉은 원(◎)</strong>: 다음 수로 5목이 완성되는 4목 공격·방어 급소입니다.',
      '<strong>청록 점선(◦)</strong>: 열린 4로 이어지는 열린 3 확장점입니다.',
    ],
  },
];

interface CheckpointManifestItem {
  id: string;
  gen: number;
  file?: string;
  binFile: string;
  totalGames?: number;
  elo?: number;
  crit?: number;
  vsRbRecord?: string;
  vsRbWr?: number;
  isChampion?: boolean;
}

interface CheckpointManifest {
  modelName?: string;
  params?: number;
  championGen?: number | null;
  updatedAt?: string;
  checkpoints: CheckpointManifestItem[];
}

interface LoadedCheckpoint {
  id: string;
  label: string;
  subLabel: string;
  gen: number | null;
  elo?: number;
  weights: D4Weights;
}

type ActivePage = 'lobby' | 'play' | 'history' | 'puzzles';

class GomokuApp {
  private root: HTMLElement;
  private appEl!: HTMLElement;
  private activePage: ActivePage = 'lobby';

  // Checkpoint / neural model selection
  private checkpointManifest: CheckpointManifest | null = null;
  private selectedEngineId = 'rule-master';
  private loadedCheckpoints = new Map<string, LoadedCheckpoint>();
  private loadingCheckpointId: string | null = null;
  private lastNnInsight: CheckpointMoveResult | null = null;

  // Live match state (never mutated by replay viewer)
  private board: Board = createEmptyBoard();
  private history: MoveRecord[] = [];
  private currentPlayer: Player = BLACK;
  private winner: WinResult | null = null;
  private isDraw = false;
  private matchRecorded = false;

  // Isolated replay state
  private historyMode: 'list' | 'viewer' = 'list';
  private replayTitle = '';
  private replayBlackName = '흑';
  private replayWhiteName = '백';
  private replayBadge: { label: string; tone: string } | null = null;
  private replayMoves: MoveRecord[] = [];
  /** Who the human was in the replayed game (null for 2-player games), and how it ended. */
  private replayHuman: Player | null = null;
  private replayCat: CatId = 'kkamang';
  /** Turning points of the replayed game; null while the cat is still looking. */
  private turningPoints: TurningPoint[] | null = null;
  /** The turning point on the board now: better move(s) and an optional winning line. */
  private reviewFocus: { index: number; cells: Coord[]; seq: Coord[] | null } | null = null;
  private reviewToken = 0;
  /** Missed wins from this replay that became puzzles. */
  private reviewPuzzles = 0;
  private reviewCat = new CatView(() => {});

  /** 묘수풀이 in progress on the play page; the lobby side choice is put back when it ends. */
  private puzzle: Puzzle | null = null;
  private puzzleState: 'solving' | 'solved' | 'failed' = 'solving';
  private puzzleHintUsed = false;
  private sideBeforePuzzle: Player | null = null;
  /** Playing on from a replayed position: not saved or counted, and can be restarted. */
  private practice: { moves: MoveRecord[]; human: Player; cat: CatId } | null = null;

  private mode: GameMode = 'ai';
  private humanPlayer: Player = BLACK;
  /** Cat picked in the lobby, and the cat of the game in progress (they differ until a new game starts). */
  private opponent: CatId = loadOpponent();
  private gameOpponent: CatId = this.opponent;
  private get masterCharId(): CharId {
    return CHAR_ID[this.gameOpponent];
  }
  private voice(cat: CatId = this.gameOpponent): CatVoice {
    return VOICES[cat];
  }

  private showRuleOverlay = true;
  private showMoveNumbers = false;
  private soundEnabled = true;
  /** Two-tap placing. Off by default: a single tap places, which is what people expect. */
  private confirmAimMode = false;
  private pendingAim: Coord | null = null;
  private activeTouchId: number | null = null;
  private touchDragMoved = false;
  private touchStartAim: Coord | null = null;
  private lastTouchTime = 0;
  private showRulesModal = false;
  private ruleSlideIndex = 0;
  private aiThinking = false;
  private aiSearchToken = 0;
  private aiTimer: number | null = null;

  private viewMoveIndex = 0;
  private autoPlayTimer: number | null = null;
  private activeVcf: Coord[] | null = null;
  private aiPendingVcf: Coord[] | null = null;
  private showVcfPreview = false;

  private forbiddenMap = new Map<string, ForbiddenCell>();
  private threats: BoardThreat[] = [];
  private lastPlacedKey: string | null = null;
  private historyFocusKey: string | null = null;
  private hoverWarningTimer: number | null = null;

  // DOM references
  private backLobbyBtn!: HTMLButtonElement;
  private backArchiveBtn!: HTMLButtonElement;
  private pageBadgeEl!: HTMLElement;
  private lobbyNavEl!: HTMLElement;
  private lobbyHistoryBtn!: HTMLButtonElement;
  private lobbyEl!: HTMLElement;
  private historyArchiveEl!: HTMLElement;
  private puzzleListEl!: HTMLElement;
  private workspaceEl!: HTMLElement;
  private seatTopRowEl!: HTMLElement;
  private playRightRailEl!: HTMLElement;
  private opponentSeatEl!: HTMLElement;
  private opponentSpeechEl!: HTMLElement;
  private leftMemoCardEl!: HTMLElement;
  private liveMovesCardEl!: HTMLElement;
  private liveMovesListEl!: HTMLElement;
  private liveMovesCountEl!: HTMLElement;
  private seatBottomRowEl!: HTMLElement;
  private selfSeatEl!: HTMLElement;
  private replayLeftPanelEl!: HTMLElement;
  private historyPanelEl!: HTMLElement;
  private nnSubEl!: HTMLElement;
  private evalTugBarEl!: HTMLElement;
  private cellsGrid!: HTMLElement;
  private overlaySvg!: SVGSVGElement;
  private aimFloatBadgeEl!: HTMLElement;
  private aimConfirmBarEl!: HTMLElement;
  private aimConfirmTextEl!: HTMLElement;
  private aimConfirmBtn!: HTMLButtonElement;
  private aimToggleBtn!: HTMLButtonElement;
  private cellButtons: HTMLButtonElement[][] = [];
  private turnDotEl!: HTMLElement;
  private turnTextEl!: HTMLElement;
  private moveCountEl!: HTMLElement;
  private statusHintEl!: HTMLElement;
  private statusRightEl!: HTMLElement;
  private evalMiniEl!: HTMLElement;
  private evalSparkBoxEl!: HTMLElement;
  private evalLabelEl!: HTMLElement;
  private historyListEl!: HTMLElement;
  private replayTitleBadgeEl!: HTMLElement;
  private replayTitleTextEl!: HTMLElement;
  private replayPlayersBarEl!: HTMLElement;
  private historyNumBtn!: HTMLButtonElement;
  private bottomNumBtn!: HTMLButtonElement;
  private evalChartBoxEl!: HTMLElement;
  private evalChartLabelEl!: HTMLElement;
  private replaySliderEl!: HTMLInputElement;
  private replayStepLabelEl!: HTMLElement;
  private copyHistoryBtn!: HTMLButtonElement;
  private replayFirstBtn!: HTMLButtonElement;
  private replayPrevBtn!: HTMLButtonElement;
  private replayPlayBtn!: HTMLButtonElement;
  private replayNextBtn!: HTMLButtonElement;
  private replayLastBtn!: HTMLButtonElement;
  private undoBtn!: HTMLButtonElement;
  private topRightEl!: HTMLElement;
  private playModeBtn!: HTMLButtonElement;
  private playCkptSelectEl!: HTMLSelectElement;
  private playSideBtn!: HTMLButtonElement;
  private postGameActionsEl!: HTMLElement;
  private endSheetEl!: HTMLElement;
  private endSheetTitleEl!: HTMLElement;
  private endSheetSubEl!: HTMLElement;

  // Modals
  private rulesBackdropEl!: HTMLElement;
  private modalStepEl!: HTMLElement;
  private modalTitleEl!: HTMLElement;
  private modalBodyEl!: HTMLElement;
  private modalDotsEl!: HTMLElement;
  private modalPrevBtn!: HTMLButtonElement;
  private modalNextBtn!: HTMLButtonElement;

  constructor(root: HTMLElement) {
    this.root = root;
    this.buildLayout();
    this.bindKeyboard();
    this.bindHistoryPopState();
    this.bindInstall();
    document.documentElement.classList.add('no-scroll'); // starts on the lobby
    this.renderLobby();
    this.recomputeAndRender();
    void this.fetchCheckpointManifest(true);
  }

  /** Chrome/Edge/Android hand us an install prompt; iOS Safari needs the share-sheet steps instead. */
  private installPrompt: (Event & { prompt(): Promise<void> }) | null = null;

  private isInstalled(): boolean {
    return (
      matchMedia('(display-mode: standalone)').matches ||
      (navigator as Navigator & { standalone?: boolean }).standalone === true
    );
  }

  private isIosSafari(): boolean {
    const ua = navigator.userAgent;
    const ios = /iphone|ipad|ipod/i.test(ua) || (ua.includes('Mac') && navigator.maxTouchPoints > 1);
    return ios && /safari/i.test(ua) && !/crios|fxios|edgios/i.test(ua);
  }

  private bindInstall(): void {
    window.addEventListener('beforeinstallprompt', (e) => {
      e.preventDefault();
      this.installPrompt = e as Event & { prompt(): Promise<void> };
      if (this.activePage === 'lobby') this.renderLobby();
    });
    window.addEventListener('appinstalled', () => {
      this.installPrompt = null;
      if (this.activePage === 'lobby') this.renderLobby();
      this.cat.setMood('happy');
      this.cat.say(this.voice(this.opponent).installed);
    });
  }

  private installButton(): HTMLButtonElement | null {
    if (this.isInstalled() || (!this.installPrompt && !this.isIosSafari())) return null;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn install-btn';
    btn.textContent = '앱으로 설치';
    btn.addEventListener('click', () => {
      if (this.installPrompt) {
        void this.installPrompt.prompt();
        return;
      }
      // iOS has no install prompt: the cat walks you through the share sheet.
      this.cat.setMood('happy');
      this.cat.say(this.voice(this.opponent).installIos);
    });
    return btn;
  }

  private bindHistoryPopState(): void {
    try {
      window.history.replaceState(
        { page: this.activePage, historyMode: this.historyMode },
        '',
      );
    } catch {
      // Ignore sandboxed history restrictions
    }
    window.addEventListener('popstate', (e) => {
      if (this.showRulesModal) this.closeRulesModal();
      const st = (e.state ?? {}) as {
        page?: ActivePage;
        historyMode?: 'list' | 'viewer';
      };
      const targetPage: ActivePage =
        st.page === 'play' || st.page === 'history' || st.page === 'lobby' || st.page === 'puzzles'
          ? st.page
          : 'lobby';
      this.historyMode = st.historyMode === 'viewer' ? 'viewer' : 'list';
      this.switchPage(targetPage, true);
    });
  }

  private buildLayout(): void {
    this.root.innerHTML = '';

    this.appEl = document.createElement('main');
    this.appEl.className = 'app';

    // ── Top Bar ──
    const topbar = document.createElement('header');
    topbar.className = 'topbar';

    const topLeft = document.createElement('div');
    topLeft.className = 'topbar-left';

    this.backLobbyBtn = document.createElement('button');
    this.backLobbyBtn.type = 'button';
    this.backLobbyBtn.className = 'btn ghost';
    this.backLobbyBtn.textContent = '← 로비';
    this.backLobbyBtn.hidden = true;
    this.backLobbyBtn.addEventListener('click', () => this.switchPage('lobby'));

    this.backArchiveBtn = document.createElement('button');
    this.backArchiveBtn.type = 'button';
    this.backArchiveBtn.className = 'btn ghost';
    this.backArchiveBtn.textContent = '← 기보 목록';
    this.backArchiveBtn.hidden = true;
    this.backArchiveBtn.addEventListener('click', () => {
      this.stopAutoPlay();
      this.historyMode = 'list';
      this.switchPage('history');
    });

    const brand = document.createElement('div');
    brand.className = 'brand';
    const mark = document.createElement('span');
    mark.className = 'brand-mark';
    const title = document.createElement('h1');
    title.textContent = '묘수';
    brand.append(mark, title);

    this.pageBadgeEl = document.createElement('span');
    this.pageBadgeEl.className = 'profile-pill';
    this.pageBadgeEl.textContent = '기보실';
    this.pageBadgeEl.hidden = this.activePage !== 'history';

    topLeft.append(this.backLobbyBtn, this.backArchiveBtn, brand, this.pageBadgeEl);

    this.lobbyNavEl = document.createElement('div');
    this.lobbyNavEl.className = 'topbar-nav';
    this.lobbyNavEl.hidden = true;

    this.lobbyHistoryBtn = document.createElement('button');
    this.lobbyHistoryBtn.type = 'button';
    this.lobbyHistoryBtn.className = 'btn';
    this.lobbyHistoryBtn.textContent = '기보실';
    this.lobbyHistoryBtn.addEventListener('click', () => {
      this.historyMode = 'list';
      this.switchPage('history');
    });

    this.lobbyNavEl.append(this.lobbyHistoryBtn);

    this.topRightEl = document.createElement('div');
    this.topRightEl.className = 'topbar-right';
    this.topRightEl.hidden = this.activePage !== 'play';

    this.playModeBtn = document.createElement('button');
    this.playModeBtn.type = 'button';
    this.playModeBtn.className = 'btn ghost';
    this.playModeBtn.addEventListener('click', () => {
      this.mode = this.mode === 'ai' ? 'pvp' : 'ai';
      this.syncModeControls();
      this.resetGame();
    });

    this.playSideBtn = document.createElement('button');
    this.playSideBtn.type = 'button';
    this.playSideBtn.className = 'btn ghost play-side-btn';
    this.playSideBtn.addEventListener('click', () => {
      this.mode = 'ai';
      this.humanPlayer = this.humanPlayer === BLACK ? WHITE : BLACK;
      this.syncModeControls();
      this.resetGame();
    });

    this.undoBtn = document.createElement('button');
    this.undoBtn.type = 'button';
    this.undoBtn.className = 'btn';
    this.undoBtn.textContent = '무르기';
    this.undoBtn.hidden = true;
    this.undoBtn.addEventListener('click', () => this.undo());

    const newMatchBtn = document.createElement('button');
    newMatchBtn.type = 'button';
    newMatchBtn.className = 'btn ghost';
    newMatchBtn.textContent = '새 대국';
    newMatchBtn.addEventListener('click', () => this.resetGame());

    this.syncModeControls();

    this.playCkptSelectEl = document.createElement('select');
    this.playCkptSelectEl.className = 'ckpt-select compact';
    this.playCkptSelectEl.setAttribute('aria-label', '대국 엔진 및 체크포인트 선택');
    this.populateEngineSelect(this.playCkptSelectEl, true);
    this.playCkptSelectEl.addEventListener('change', () => {
      void this.selectCheckpointEngine(this.playCkptSelectEl.value);
    });

    // Only one model is served, so the mode toggle and checkpoint picker stay off-screen.
    this.topRightEl.append(
      this.playSideBtn,
      newMatchBtn,
    );
    topbar.append(topLeft, this.lobbyNavEl, this.topRightEl);

    // ── Lobby View (Login + Single Omok Master + Leaderboard) ──
    this.lobbyEl = document.createElement('section');
    this.lobbyEl.className = 'lobby';
    this.lobbyEl.hidden = this.activePage !== 'lobby';

    // ── History Archive List View (Saved Win/Loss Matches) ──
    this.historyArchiveEl = document.createElement('section');
    this.historyArchiveEl.className = 'archive-view';
    this.historyArchiveEl.hidden = true;

    this.puzzleListEl = document.createElement('section');
    this.puzzleListEl.className = 'archive-view puzzle-view';
    this.puzzleListEl.hidden = true;

    // ── Workspace: 3-Zone Widescreen Play Stage (Left Seat / Center Board / Right Eval+Self) or History Viewer ──
    this.workspaceEl = document.createElement('div');
    this.workspaceEl.className = 'workspace';
    this.workspaceEl.hidden = this.activePage === 'lobby';

    // Left Rail: Opponent Character Card + Speech Bubble (Top-Left) & Match Memo (Bottom-Left)
    this.seatTopRowEl = document.createElement('aside');
    this.seatTopRowEl.className = 'play-left-rail';
    const oppHero = document.createElement('div');
    oppHero.className = 'play-left-hero';
    this.opponentSeatEl = document.createElement('div');
    this.opponentSeatEl.className = 'player-seat opponent';
    this.opponentSpeechEl = document.createElement('div');
    this.opponentSpeechEl.className = 'seat-bubble';
    oppHero.append(this.opponentSeatEl, this.opponentSpeechEl);

    this.leftMemoCardEl = document.createElement('div');
    this.leftMemoCardEl.className = 'side-memo-card';
    this.seatTopRowEl.append(oppHero, this.leftMemoCardEl);

    const playCol = document.createElement('section');
    playCol.className = 'play-col';

    // Fixed-height status strip (zero layout shift)
    const statusStrip = document.createElement('div');
    statusStrip.className = 'status-strip';

    const statusLeft = document.createElement('div');
    statusLeft.className = 'status-left';
    this.turnDotEl = document.createElement('span');
    this.turnDotEl.className = 'turn-dot black';
    this.turnTextEl = document.createElement('span');
    this.turnTextEl.className = 'turn-text';
    this.moveCountEl = document.createElement('span');
    this.moveCountEl.className = 'move-count';
    this.statusHintEl = document.createElement('span');
    this.statusHintEl.className = 'status-hint';
    statusLeft.append(
      this.turnDotEl,
      this.turnTextEl,
      this.moveCountEl,
      this.statusHintEl,
    );

    this.statusRightEl = document.createElement('div');
    this.statusRightEl.className = 'status-right';
    statusStrip.append(statusLeft, this.statusRightEl);

    // Board
    const boardWrap = document.createElement('div');
    boardWrap.className = 'board-wrap';
    this.helper.el.classList.add('helper-cat');
    boardWrap.appendChild(this.helper.el);
    const boardFrame = document.createElement('div');
    boardFrame.className = 'board-frame';

    const coordCols = document.createElement('div');
    coordCols.className = 'coord-cols';
    for (let c = 0; c < BOARD_SIZE; c++) {
      const span = document.createElement('span');
      span.textContent = String.fromCharCode(65 + c);
      coordCols.appendChild(span);
    }

    const coordRows = document.createElement('div');
    coordRows.className = 'coord-rows';
    for (let r = 0; r < BOARD_SIZE; r++) {
      const span = document.createElement('span');
      span.textContent = String(BOARD_SIZE - r);
      coordRows.appendChild(span);
    }

    const boardInner = document.createElement('div');
    boardInner.className = 'board-inner';
    const gridSvg = buildStaticGridSvg();
    this.overlaySvg = createSvgElement('svg', {
      class: 'board-svg',
      viewBox: '0 0 1500 1500',
    }) as SVGSVGElement;

    this.cellsGrid = document.createElement('div');
    this.cellsGrid.className = 'cells-grid';
    this.buildCellsGrid();
    this.bindTouchAiming();

    this.aimFloatBadgeEl = document.createElement('div');
    this.aimFloatBadgeEl.className = 'aim-float-badge';
    this.aimFloatBadgeEl.hidden = true;

    boardInner.append(gridSvg, this.overlaySvg, this.cellsGrid, this.aimFloatBadgeEl);
    boardFrame.append(coordRows, coordCols, boardInner);
    boardWrap.appendChild(boardFrame);

    // Aim confirmation bar (shown when a cell is aimed in Confirm/Touch mode)
    this.aimConfirmBarEl = document.createElement('div');
    this.aimConfirmBarEl.className = 'aim-confirm-bar';
    this.aimConfirmBarEl.hidden = true;

    this.aimConfirmTextEl = document.createElement('span');
    this.aimConfirmTextEl.className = 'aim-confirm-text';

    const aimActions = document.createElement('div');
    aimActions.className = 'aim-confirm-actions';

    const aimCancelBtn = document.createElement('button');
    aimCancelBtn.type = 'button';
    aimCancelBtn.className = 'btn ghost';
    aimCancelBtn.textContent = '취소';
    aimCancelBtn.addEventListener('click', () => {
      this.setPendingAim(null);
      this.resetHintMessage();
    });

    this.aimConfirmBtn = document.createElement('button');
    this.aimConfirmBtn.type = 'button';
    this.aimConfirmBtn.className = 'btn primary';
    this.aimConfirmBtn.textContent = '여기에 착수';
    this.aimConfirmBtn.addEventListener('click', () => {
      if (!this.pendingAim) return;
      const { row, col } = this.pendingAim;
      this.setPendingAim(null);
      this.tryPlaceAt(row, col);
    });

    aimActions.append(aimCancelBtn, this.aimConfirmBtn);
    this.aimConfirmBarEl.append(this.aimConfirmTextEl, aimActions);

    // Bottom tools strip
    const bottomStrip = document.createElement('div');
    bottomStrip.className = 'bottom-strip';

    const quietPills = document.createElement('div');
    quietPills.className = 'quiet-pills';

    const overlayBtn = this.createQuietBtn(
      '금수·위협',
      this.showRuleOverlay,
      (btn) => {
        this.showRuleOverlay = !this.showRuleOverlay;
        btn.classList.toggle('active', this.showRuleOverlay);
        this.renderBoard();
      },
    );

    this.bottomNumBtn = this.createQuietBtn(
      '수 번호',
      this.showMoveNumbers,
      () => this.toggleMoveNumbers(),
    );

    const soundBtn = this.createQuietBtn('소리', this.soundEnabled, (btn) => {
      this.soundEnabled = !this.soundEnabled;
      btn.classList.toggle('active', this.soundEnabled);
    });

    this.aimToggleBtn = this.createQuietBtn(
      '조준 착수',
      this.confirmAimMode,
      (btn) => {
        this.confirmAimMode = !this.confirmAimMode;
        btn.classList.toggle('active', this.confirmAimMode);
        if (!this.confirmAimMode) this.setPendingAim(null);
      },
    );

    const rulesBtn = this.createQuietBtn('규칙', false, () => {
      this.openRulesModal();
    });

    quietPills.append(
      overlayBtn,
      this.bottomNumBtn,
      this.aimToggleBtn,
      soundBtn,
      rulesBtn,
    );

    bottomStrip.append(quietPills);
    playCol.append(statusStrip, boardWrap, this.aimConfirmBarEl, bottomStrip);

    // Right Rail during Play: Live Advantage Graph (Top) + Self Seat & Actions (Bottom-Right)
    this.playRightRailEl = document.createElement('aside');
    this.playRightRailEl.className = 'play-right-rail';

    this.evalMiniEl = document.createElement('div');
    this.evalMiniEl.className = 'eval-card';
    const evalHead = document.createElement('div');
    evalHead.className = 'eval-card-head';
    const evalTitle = document.createElement('span');
    evalTitle.textContent = '실시간 형세';
    this.evalLabelEl = document.createElement('span');
    this.evalLabelEl.className = 'eval-card-pct';
    this.evalLabelEl.textContent = '호각 50%';
    evalHead.append(evalTitle, this.evalLabelEl);
    this.evalSparkBoxEl = document.createElement('div');
    this.evalSparkBoxEl.className = 'eval-spark-box';
    this.evalMiniEl.append(evalHead, this.evalSparkBoxEl);

    this.liveMovesCardEl = document.createElement('div');
    this.liveMovesCardEl.className = 'live-moves-card';
    const liveHead = document.createElement('div');
    liveHead.className = 'eval-card-head';
    const liveTitle = document.createElement('span');
    liveTitle.textContent = '착수 기록';
    this.liveMovesCountEl = document.createElement('span');
    this.liveMovesCountEl.className = 'eval-card-pct';
    this.liveMovesCountEl.textContent = '0수';
    liveHead.append(liveTitle, this.liveMovesCountEl);
    this.liveMovesListEl = document.createElement('div');
    this.liveMovesListEl.className = 'history-list live-moves-list';
    this.liveMovesCardEl.append(liveHead, this.liveMovesListEl);

    this.seatBottomRowEl = document.createElement('div');
    this.seatBottomRowEl.className = 'seat-bottom-row';
    this.selfSeatEl = document.createElement('div');
    this.selfSeatEl.className = 'player-seat self';
    this.postGameActionsEl = document.createElement('div');
    this.postGameActionsEl.className = 'postgame-actions';
    this.seatBottomRowEl.append(this.selfSeatEl);

    // End-of-game sheet: slides up from the bottom so it's obvious the game is over.
    this.endSheetEl = document.createElement('section');
    this.endSheetEl.className = 'end-sheet';
    this.endSheetEl.setAttribute('role', 'dialog');
    this.endSheetEl.setAttribute('aria-labelledby', 'end-sheet-title');
    this.endSheetTitleEl = document.createElement('h2');
    this.endSheetTitleEl.id = 'end-sheet-title';
    this.endSheetTitleEl.className = 'end-sheet-title';
    this.endSheetSubEl = document.createElement('p');
    this.endSheetSubEl.className = 'end-sheet-sub';
    const sheetHead = document.createElement('div');
    sheetHead.className = 'end-sheet-head';
    sheetHead.append(this.endSheetTitleEl, this.endSheetSubEl);
    this.endSheetEl.append(sheetHead, this.postGameActionsEl);

    this.hintBarEl = document.createElement('section');
    this.hintBarEl.className = 'hint-bar';
    this.playRightRailEl.append(
      this.evalMiniEl,
      this.hintBarEl,
      this.liveMovesCardEl,
      this.seatBottomRowEl,
    );

    // ── Separate 3-Column Replay Panels (Left: Matchup/Advantage/Moves, Right: Candidate Analysis/Loss) ──
    this.replayLeftPanelEl = document.createElement('aside');
    this.replayLeftPanelEl.className = 'history-page-panel replay-left-panel';
    this.replayLeftPanelEl.hidden = this.activePage !== 'history';

    this.historyPanelEl = document.createElement('aside');
    this.historyPanelEl.className = 'history-page-panel replay-right-panel';
    this.historyPanelEl.hidden = this.activePage !== 'history';

    const replayHead = document.createElement('div');
    replayHead.className = 'replay-match-head';

    const titleWrap = document.createElement('div');
    titleWrap.className = 'replay-match-title';
    this.replayTitleBadgeEl = document.createElement('span');
    this.replayTitleBadgeEl.className = 'archive-badge';
    this.replayTitleTextEl = document.createElement('span');
    this.replayTitleTextEl.textContent = '기보 복기';
    titleWrap.append(this.replayTitleBadgeEl, this.replayTitleTextEl);

    const historyActions = document.createElement('div');
    historyActions.className = 'history-actions';

    this.historyNumBtn = document.createElement('button');
    this.historyNumBtn.type = 'button';
    this.historyNumBtn.className = 'mini-btn';
    this.historyNumBtn.textContent = '수 번호';
    this.historyNumBtn.classList.toggle('active', this.showMoveNumbers);
    this.historyNumBtn.addEventListener('click', () => this.toggleMoveNumbers());

    this.copyHistoryBtn = document.createElement('button');
    this.copyHistoryBtn.type = 'button';
    this.copyHistoryBtn.className = 'mini-btn';
    this.copyHistoryBtn.textContent = '복사';
    this.copyHistoryBtn.addEventListener('click', () => this.copyHistory());

    historyActions.append(this.historyNumBtn, this.copyHistoryBtn);
    replayHead.append(titleWrap, historyActions);

    this.replayPlayersBarEl = document.createElement('div');
    this.replayPlayersBarEl.className = 'replay-players-bar';

    // Move-by-move Advantage Tug-of-War & Flow Graph
    this.evalChartBoxEl = document.createElement('div');
    this.evalChartBoxEl.className = 'eval-chart-box';
    const evalChartHead = document.createElement('div');
    evalChartHead.className = 'eval-chart-head';
    const evalChartTitle = document.createElement('span');
    evalChartTitle.textContent = '형세 분석';
    this.evalChartLabelEl = document.createElement('span');
    this.evalChartLabelEl.className = 'eval-judge-pill';
    this.evalChartLabelEl.textContent = '호각 50:50';
    evalChartHead.append(evalChartTitle, this.evalChartLabelEl);

    this.evalTugBarEl = document.createElement('div');
    this.evalTugBarEl.className = 'eval-tug-bar';

    this.evalChartBoxEl.append(evalChartHead, this.evalTugBarEl);

    // Scrub slider
    const sliderRow = document.createElement('div');
    sliderRow.className = 'replay-slider-row';
    this.replaySliderEl = document.createElement('input');
    this.replaySliderEl.type = 'range';
    this.replaySliderEl.className = 'replay-slider';
    this.replaySliderEl.min = '0';
    this.replaySliderEl.max = '0';
    this.replaySliderEl.value = '0';
    this.replaySliderEl.setAttribute('aria-label', '기보 탐색 슬라이더');
    this.replaySliderEl.addEventListener('input', () => {
      this.jumpToMove(Number(this.replaySliderEl.value));
    });
    this.replayStepLabelEl = document.createElement('span');
    this.replayStepLabelEl.className = 'replay-step-label';
    this.replayStepLabelEl.textContent = '0 / 0';
    sliderRow.append(this.replaySliderEl, this.replayStepLabelEl);

    const replayNav = document.createElement('div');
    replayNav.className = 'replay-nav';
    this.replayFirstBtn = this.createReplayBtn('|<', '처음', () =>
      this.jumpToMove(0),
    );
    this.replayPrevBtn = this.createReplayBtn('< 이전', '한 수 뒤로 (←)', () =>
      this.jumpToMove(this.viewMoveIndex - 1),
    );
    this.replayPlayBtn = this.createReplayBtn('▶ 재생', '자동 재생 (Space)', () =>
      this.toggleAutoPlay(),
    );
    this.replayNextBtn = this.createReplayBtn('다음 >', '한 수 앞으로 (→)', () =>
      this.jumpToMove(this.viewMoveIndex + 1),
    );
    this.replayLastBtn = this.createReplayBtn('>|', '마지막 수', () =>
      this.jumpToMove(this.getActiveReplayMoves().length),
    );
    replayNav.append(
      this.replayFirstBtn,
      this.replayPrevBtn,
      this.replayPlayBtn,
      this.replayNextBtn,
      this.replayLastBtn,
    );

    this.historyListEl = document.createElement('div');
    this.historyListEl.className = 'history-list';

    this.replayLeftPanelEl.append(
      replayHead,
      this.replayPlayersBarEl,
      sliderRow,
      replayNav,
      this.evalChartBoxEl,
      this.historyListEl,
    );

    this.nnSubEl = document.createElement('div');
    this.nnSubEl.className = 'history-subview';
    this.historyPanelEl.appendChild(this.nnSubEl);

    this.workspaceEl.append(
      this.seatTopRowEl,
      this.replayLeftPanelEl,
      playCol,
      this.playRightRailEl,
      this.historyPanelEl,
    );
    this.appEl.append(
      topbar,
      this.lobbyEl,
      this.historyArchiveEl,
      this.puzzleListEl,
      this.workspaceEl,
    );
    this.root.appendChild(this.appEl);
    this.root.appendChild(this.endSheetEl);

    this.buildRulesModal();
  }

  private getActiveCheckpoint(): LoadedCheckpoint | null {
    if (this.selectedEngineId === 'rule-master') return null;
    return this.loadedCheckpoints.get(this.selectedEngineId) ?? null;
  }

  private populateEngineSelect(selectEl: HTMLSelectElement, compact = false): void {
    const prevValue = this.selectedEngineId;
    selectEl.innerHTML = '';

    const ruleOpt = document.createElement('option');
    ruleOpt.value = 'rule-master';
    ruleOpt.textContent = compact
      ? '규칙 마스터 · 윤'
      : '규칙 마스터 · 윤 (VCF · 깊이4)';
    selectEl.appendChild(ruleOpt);

    if (this.checkpointManifest?.checkpoints?.length) {
      const sorted = [...this.checkpointManifest.checkpoints].sort(
        (a, b) => b.gen - a.gen,
      );
      for (const c of sorted) {
        const id = c.id || `ckpt-${c.gen}`;
        const isChamp =
          Boolean(c.isChampion) ||
          (this.checkpointManifest.championGen != null &&
            c.gen === this.checkpointManifest.championGen);
        const stats: string[] = [];
        if (c.elo != null) stats.push(`Elo ${c.elo}`);
        if (!compact && c.crit != null) stats.push(`Crit ${c.crit.toFixed(1)}%`);
        if (!compact && c.vsRbWr != null) stats.push(`vs봇 ${c.vsRbWr.toFixed(0)}%`);
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = `${isChamp ? '★ 챔피언 ' : ''}Gen ${c.gen}${
          stats.length ? ` (${stats.join(' · ')})` : ''
        }`;
        selectEl.appendChild(opt);
      }
    }

    for (const [id, ckpt] of this.loadedCheckpoints.entries()) {
      if (!id.startsWith('local-')) continue;
      const opt = document.createElement('option');
      opt.value = id;
      opt.textContent = compact
        ? `📁 ${ckpt.label}`
        : `📁 ${ckpt.label} (${ckpt.subLabel})`;
      selectEl.appendChild(opt);
    }

    const hasTarget = Array.from(selectEl.options).some((o) => o.value === prevValue);
    selectEl.value = hasTarget ? prevValue : 'rule-master';
  }

  private syncCheckpointControls(): void {
    if (!this.playCkptSelectEl) return;
    this.populateEngineSelect(this.playCkptSelectEl, true);
    this.playCkptSelectEl.hidden = this.mode !== 'ai';
    this.playCkptSelectEl.disabled =
      this.aiThinking || Boolean(this.loadingCheckpointId);
  }

  private async fetchCheckpointManifest(autoSelectChampion = false): Promise<void> {
    try {
      const res = await fetch(`./checkpoints/manifest.json?t=${Date.now()}`, {
        cache: 'no-store',
      });
      if (!res.ok) {
        this.checkpointManifest = null;
        this.syncCheckpointControls();
        if (this.activePage === 'lobby') this.renderLobby();
        return;
      }
      const ct = res.headers.get('content-type') ?? '';
      if (!ct.includes('application/json') && !ct.includes('text/json')) {
        this.checkpointManifest = null;
        this.syncCheckpointControls();
        if (this.activePage === 'lobby') this.renderLobby();
        return;
      }
      const data = (await res.json()) as CheckpointManifest;
      if (!data || !Array.isArray(data.checkpoints)) {
        this.checkpointManifest = null;
        this.syncCheckpointControls();
        if (this.activePage === 'lobby') this.renderLobby();
        return;
      }
      this.checkpointManifest = data;
      this.syncCheckpointControls();
      if (this.activePage === 'lobby') this.renderLobby();

      if (
        autoSelectChampion &&
        this.selectedEngineId === 'rule-master' &&
        this.history.length === 0 &&
        data.checkpoints.length > 0
      ) {
        const champ =
          data.checkpoints.find(
            (c) =>
              Boolean(c.isChampion) ||
              (data.championGen != null && c.gen === data.championGen),
          ) ?? [...data.checkpoints].sort((a, b) => b.gen - a.gen)[0];
        if (champ) {
          await this.selectCheckpointEngine(champ.id || `ckpt-${champ.gen}`, true);
        }
      }
    } catch {
      this.checkpointManifest = null;
      this.syncCheckpointControls();
      if (this.activePage === 'lobby') this.renderLobby();
    }
  }

  private async selectCheckpointEngine(
    id: string,
    onlyIfIdleAndUnset = false,
  ): Promise<void> {
    if (onlyIfIdleAndUnset && (this.selectedEngineId !== 'rule-master' || this.history.length > 0)) {
      return;
    }
    if (id === 'rule-master') {
      this.selectedEngineId = 'rule-master';
      this.loadingCheckpointId = null;
      this.lastNnInsight = null;
      this.syncModeControls();
      this.renderSeats();
      if (this.activePage === 'lobby') this.renderLobby();
      return;
    }

    if (this.loadedCheckpoints.has(id)) {
      this.selectedEngineId = id;
      this.loadingCheckpointId = null;
      this.lastNnInsight = null;
      this.syncModeControls();
      this.renderSeats();
      if (this.activePage === 'lobby') this.renderLobby();
      return;
    }

    const item = this.checkpointManifest?.checkpoints.find(
      (c) => (c.id || `ckpt-${c.gen}`) === id,
    );
    if (!item?.binFile) {
      this.syncCheckpointControls();
      return;
    }

    this.loadingCheckpointId = id;
    this.syncCheckpointControls();
    if (this.activePage === 'lobby') this.renderLobby();

    try {
      const res = await fetch(`./checkpoints/${item.binFile}?t=${Date.now()}`, {
        cache: 'no-store',
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const buf = await res.arrayBuffer();
      const weights = parseD4Weights(buf);
      const isChamp =
        Boolean(item.isChampion) ||
        (this.checkpointManifest?.championGen != null &&
          item.gen === this.checkpointManifest.championGen);
      const label = isChamp ? `★ Gen ${item.gen}` : `Gen ${item.gen}`;
      const parts: string[] = [];
      if (item.elo != null) parts.push(`Elo ${item.elo}`);
      if (item.crit != null) parts.push(`Crit ${item.crit.toFixed(1)}%`);
      if (item.vsRbWr != null) parts.push(`vs봇 ${item.vsRbWr.toFixed(0)}%`);
      const subLabel = parts.join(' · ') || '61,472 가중치';

      this.loadedCheckpoints.set(id, {
        id,
        label,
        subLabel,
        gen: item.gen,
        elo: item.elo,
        weights,
      });

      if (!onlyIfIdleAndUnset || (this.selectedEngineId === 'rule-master' && this.history.length === 0)) {
        this.selectedEngineId = id;
        this.lastNnInsight = null;
      }
    } catch {
      // Keep previous selection on fetch/parse error
    } finally {
      this.loadingCheckpointId = null;
      this.syncModeControls();
      this.renderSeats();
      if (this.activePage === 'lobby') this.renderLobby();
    }
  }

  private getMasterChar() {
    const base =
      AI_CHARACTERS.find((c) => c.id === this.masterCharId) ??
      AI_CHARACTERS[AI_CHARACTERS.length - 1];
    // No rank on screen; the cat (까망이) only speaks.
    return { ...base, name: CAT_NAME[this.gameOpponent], tier: '' };
  }

  private toggleMoveNumbers(): void {
    this.showMoveNumbers = !this.showMoveNumbers;
    this.bottomNumBtn.classList.toggle('active', this.showMoveNumbers);
    this.historyNumBtn.classList.toggle('active', this.showMoveNumbers);
    this.renderBoard();
  }

  private syncModeControls(): void {
    const ch = this.getMasterChar();
    this.playModeBtn.textContent =
      this.mode === 'ai' ? `vs ${ch.name} (${ch.tier})` : '2인 대국';
    this.playSideBtn.hidden = this.mode !== 'ai';
    this.playSideBtn.textContent =
      this.humanPlayer === BLACK ? '내 돌: 흑 (선공)' : '내 돌: 백 (후공)';
    this.syncCheckpointControls();
  }

  private syncPlayActions(): void {
    const finished = Boolean(this.winner || this.isDraw);
    const inHistoryViewer =
      this.activePage === 'history' && this.historyMode === 'viewer';
    const inHistoryList =
      this.activePage === 'history' && this.historyMode === 'list';

    this.backLobbyBtn.hidden = this.activePage === 'lobby' || inHistoryViewer;
    this.backArchiveBtn.hidden = !inHistoryViewer;
    this.pageBadgeEl.hidden = !inHistoryViewer && !inHistoryList && this.activePage !== 'puzzles';
    this.pageBadgeEl.textContent = this.activePage === 'puzzles' ? '묘수풀이' : '기보실';

    this.lobbyNavEl.hidden = true; // archive button moved into the lobby footer
    this.lobbyHistoryBtn.hidden = this.activePage === 'history';

    this.workspaceEl.classList.toggle('is-play', this.activePage === 'play');
    this.workspaceEl.classList.toggle('is-replay', inHistoryViewer);
    this.playRightRailEl.hidden = this.activePage !== 'play';
    this.replayLeftPanelEl.hidden = !inHistoryViewer;
    this.historyPanelEl.hidden = !inHistoryViewer;
    this.topRightEl.hidden = this.activePage !== 'play' || finished || this.puzzle !== null;
    this.workspaceEl.classList.toggle('is-puzzle', this.puzzle !== null);
    // Always visible in the help panel (no layout jump); just disabled when there's nothing to take back.
    this.undoBtn.hidden = finished;
    this.undoBtn.disabled =
      this.aiThinking || this.history.length === 0 || (this.mode === 'ai' && this.undoLeft <= 0);
    this.undoBtn.textContent = this.mode === 'ai' ? `무르기 (남은 ${this.undoLeft}번)` : '무르기';
    this.playCkptSelectEl.hidden = this.mode !== 'ai';
    this.playCkptSelectEl.disabled =
      this.aiThinking || Boolean(this.loadingCheckpointId);
  }

  private switchPage(page: ActivePage, fromPopState = false): void {
    // The lobby and the board fit one screen; letting them scroll made taps land on the wrong point.
    document.documentElement.classList.toggle('no-scroll', page === 'lobby' || page === 'play');
    if (page === 'lobby' || page === 'puzzles') this.leavePuzzle(); // a puzzle isn't a game to resume
    this.activePage = page;
    if (page !== 'play') this.endSheetEl.classList.remove('open');
    this.lobbyEl.hidden = page !== 'lobby';

    const showArchiveList = page === 'history' && this.historyMode === 'list';
    const showViewer = page === 'history' && this.historyMode === 'viewer';
    this.historyArchiveEl.hidden = !showArchiveList;
    this.puzzleListEl.hidden = page !== 'puzzles';
    this.workspaceEl.hidden = page === 'lobby' || showArchiveList || page === 'puzzles';
    this.appEl.classList.toggle('is-side', showViewer);
    this.replayLeftPanelEl.hidden = !showViewer;
    this.historyPanelEl.hidden = !showViewer;

    if (!fromPopState) {
      try {
        const cur = (window.history.state ?? {}) as {
          page?: ActivePage;
          historyMode?: 'list' | 'viewer';
        };
        const changed =
          cur.page !== page ||
          (page === 'history' && cur.historyMode !== this.historyMode);
        if (changed) {
          window.history.pushState(
            { page, historyMode: this.historyMode },
            '',
          );
        }
      } catch {
        // Ignore sandboxed history restrictions
      }
    }

    if (page === 'lobby') {
      this.stopAutoPlay();
      this.renderLobby();
      this.syncPlayActions();
      return;
    }

    if (page === 'puzzles') {
      this.stopAutoPlay();
      this.renderPuzzleList();
      this.syncPlayActions();
      return;
    }

    if (showArchiveList) {
      this.stopAutoPlay();
      this.renderHistoryArchive();
      this.syncPlayActions();
      return;
    }

    if (showViewer) {
      if (!this.showMoveNumbers) {
        this.showMoveNumbers = true;
        this.bottomNumBtn.classList.add('active');
        this.historyNumBtn.classList.add('active');
      }
    } else {
      this.stopAutoPlay();
      this.viewMoveIndex = this.history.length;
    }
    this.recomputeAndRender();
  }

  private formatPlayedAt(ts: number): string {
    const diffMin = Math.floor((Date.now() - ts) / 60000);
    if (diffMin < 1) return '방금 전';
    if (diffMin < 60) return `${diffMin}분 전`;
    const d = new Date(ts);
    return `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
  }

  private renderHistoryArchive(): void {
    this.historyArchiveEl.innerHTML = '';

    const wrap = document.createElement('div');
    wrap.className = 'archive-single';

    const matches = getSavedMatches();

    // My Completed Match History (Win / Loss / Draw)
    const matchCard = document.createElement('section');
    matchCard.className = 'lobby-card';
    const matchHead = document.createElement('div');
    matchHead.className = 'lobby-card-head';
    matchHead.innerHTML = `<span class="lobby-card-title">내 대국 기록 (${matches.length})</span><span class="lobby-card-sub">누르면 복기로 이동합니다</span>`;
    matchCard.appendChild(matchHead);

    const matchList = document.createElement('div');
    matchList.className = 'archive-list';

    if (matches.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'lb-empty';
      empty.textContent =
        '아직 완료된 대국 기보가 없습니다. 대국을 마치면 승패와 수순이 이곳에 자동 저장됩니다.';
      matchList.appendChild(empty);
    } else {
      for (const m of matches) {
        const row = document.createElement('button');
        row.type = 'button';
        row.className = 'archive-row';

        let badgeLabel = '무승부';
        let badgeCls = 'draw';
        if (m.winner) {
          if (m.mode === 'ai') {
            const won = m.winner === m.humanPlayer;
            badgeLabel = won ? '승리' : '패배';
            badgeCls = won ? 'win' : 'loss';
          } else {
            badgeLabel = m.winner === BLACK ? '흑 승' : '백 승';
            badgeCls = m.winner === BLACK ? 'black' : 'white';
          }
        }

        const blackName =
          m.mode === 'ai'
            ? m.humanPlayer === BLACK
              ? m.playerNickname
              : m.opponentName
            : 'P1';
        const whiteName =
          m.mode === 'ai'
            ? m.humanPlayer === WHITE
              ? m.playerNickname
              : m.opponentName
            : 'P2';
        const avatarHtml =
          m.mode === 'ai' && m.charId
            ? `<span class="char-avatar char-avatar-sm"><img src="${catSpriteUrl('idle', m.charId === CHAR_ID.hayang ? 'hayang' : 'kkamang')}" alt="" /></span>`
            : '';
        const titleText =
          m.mode === 'ai'
            ? `${m.opponentName}${m.opponentTier ? ` (${m.opponentTier})` : ''}`
            : '2인 대국';
        const subText = `⚫ 흑: ${blackName} · ⚪ 백: ${whiteName} · ${this.formatPlayedAt(m.playedAt)}`;

        row.innerHTML = `
          <div class="archive-row-left">
            <span class="archive-badge ${badgeCls}">${badgeLabel}</span>
            ${avatarHtml}
            <div class="archive-info">
              <span class="archive-title">${titleText}</span>
              <span class="archive-sub">${subText}</span>
            </div>
          </div>
          <div class="archive-row-right">
            <span>${m.moves.length}수</span>
            <span class="archive-cta">복기 ›</span>
          </div>
        `;
        row.addEventListener('click', () => this.openSavedMatchReplay(m));
        matchList.appendChild(row);
      }
    }
    matchCard.appendChild(matchList);
    wrap.appendChild(matchCard);
    this.historyArchiveEl.appendChild(wrap);
  }

  private openSavedMatchReplay(m: SavedMatch): void {
    this.stopAutoPlay();
    this.replayMoves = m.moves.map((mv) => ({ ...mv }));

    if (m.winner) {
      if (m.mode === 'ai') {
        const won = m.winner === m.humanPlayer;
        this.replayBadge = { label: won ? '승리' : '패배', tone: won ? 'win' : 'loss' };
      } else {
        this.replayBadge = {
          label: m.winner === BLACK ? '흑 승' : '백 승',
          tone: m.winner === BLACK ? 'black' : 'white',
        };
      }
    } else {
      this.replayBadge = { label: '무승부', tone: 'draw' };
    }

    this.replayTitle =
      m.mode === 'ai'
        ? `${m.opponentName} vs ${m.playerNickname}`
        : '2인 대국 기보';
    if (m.mode === 'ai') {
      this.replayBlackName = m.humanPlayer === BLACK ? m.playerNickname : m.opponentName;
      this.replayWhiteName = m.humanPlayer === WHITE ? m.playerNickname : m.opponentName;
    } else {
      this.replayBlackName = '흑 (P1)';
      this.replayWhiteName = '백 (P2)';
    }
    this.viewMoveIndex = this.replayMoves.length;
    this.showVcfPreview = false;
    this.historyMode = 'viewer';
    this.startReview(m.mode === 'ai' ? m.humanPlayer : null, m.winner, m.charId === 'lv2' ? 'hayang' : 'kkamang');
    this.switchPage('history');
  }

  private openLiveMatchInViewer(): void {
    this.stopAutoPlay();
    this.replayMoves = this.history.map((mv) => ({ ...mv }));
    const ch = this.getMasterChar();
    const myName = getCurrentUser()?.nickname ?? '나';

    if (this.winner) {
      if (this.mode === 'ai') {
        const won = this.winner.winner === this.humanPlayer;
        this.replayBadge = { label: won ? '승리' : '패배', tone: won ? 'win' : 'loss' };
      } else {
        this.replayBadge = {
          label: this.winner.winner === BLACK ? '흑 승' : '백 승',
          tone: this.winner.winner === BLACK ? 'black' : 'white',
        };
      }
    } else if (this.isDraw) {
      this.replayBadge = { label: '무승부', tone: 'draw' };
    } else {
      this.replayBadge = { label: '진행 중', tone: 'draw' };
    }

    this.replayTitle =
      this.mode === 'ai' ? `${ch.name} vs ${myName}` : '2인 대국 기보';
    if (this.mode === 'ai') {
      this.replayBlackName = this.humanPlayer === BLACK ? myName : ch.name;
      this.replayWhiteName = this.humanPlayer === WHITE ? myName : ch.name;
    } else {
      this.replayBlackName = '흑 (P1)';
      this.replayWhiteName = '백 (P2)';
    }
    this.viewMoveIndex = this.replayMoves.length;
    this.showVcfPreview = false;
    this.historyMode = 'viewer';
    this.startReview(this.mode === 'ai' ? this.humanPlayer : null, this.winner?.winner ?? null, this.gameOpponent);
    this.switchPage('history');
  }

  private stageTalkIdx = 0;
  private stagePoke: () => void = () => {};
  private cat = new CatView(() => this.stagePoke());
  /** Line keys already said this game, and the move count last reacted to. */
  private catSpoken = new Set<string>();
  private catSeenMoves = -1;
  private undoLeft = 0;
  /** Each hint kind can be used once per game; the other cat drops by to give it. */
  private hintsLeft = HINTS_PER_GAME;
  private hintCells: Coord[] = [];
  private hintBusy = false;
  private helper = new CatView(() => {});
  private helperTimer: number | null = null;
  private hintBarEl!: HTMLElement;

  private renderLobby(): void {
    this.lobbyEl.innerHTML = '';
    const user = getCurrentUser();

    const stage = document.createElement('section');
    stage.className = 'stage';

    const voice = this.voice(this.opponent);
    this.cat.setCharacter(this.opponent, CAT_NAME[this.opponent]);
    const say = (text: string, mood: CatMood) => {
      this.cat.setMood(mood);
      this.cat.say(text);
    };
    const sayLine = ([mood, options]: [CatMood, string[]]) => say(pickLine(options, Date.now()), mood);

    const rec = user?.records[CHAR_ID[this.opponent]];
    const played = rec && rec.wins + rec.losses + rec.draws > 0;
    // Unopened patch notes come first: the cat tells you something changed.
    sayLine(
      hasUnseenNews()
        ? voice.news
        : user
          ? voice.greetBack(user.nickname, played ? formatRecord(rec.wins, rec.draws, rec.losses) : null)
          : voice.greetFirst,
    );

    this.stagePoke = () => {
      playMeowSound(this.soundEnabled, 'curious');
      const pokes = this.voice(this.activePage === 'play' ? this.gameOpponent : this.opponent).pokes;
      const [text, mood] = pokes[this.stageTalkIdx++ % pokes.length];
      say(text, mood);
    };

    // Opponent picker: 까망이 (normal) or 하양이 (easy).
    const picker = this.createSegmented(
      [
        { value: 'kkamang', label: '까망이 · 보통' },
        { value: 'hayang', label: '하양이 · 쉬움' },
      ],
      this.opponent,
      (val) => {
        this.opponent = val as CatId;
        try {
          localStorage.setItem(OPPONENT_KEY, this.opponent);
        } catch {
          // private mode: the choice just isn't remembered
        }
        this.renderLobby();
      },
    );
    picker.classList.add('opponent-pick');


    const chips = document.createElement('div');
    chips.className = 'stage-chips';
    for (const [label, text, mood] of voice.topics) {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'stage-chip';
      chip.textContent = label;
      chip.addEventListener('click', () => say(text, mood));
      chips.appendChild(chip);
    }

    const actions = document.createElement('div');
    actions.className = 'stage-actions';
    const sidePick = this.createSegmented(
      [
        { value: String(BLACK), label: '흑 (선공)' },
        { value: String(WHITE), label: '백 (후공)' },
      ],
      String(this.humanPlayer),
      (val) => {
        this.humanPlayer = Number(val) as Player;
        sayLine(this.humanPlayer === BLACK ? voice.pickBlack : voice.pickWhite);
      },
    );
    actions.appendChild(sidePick);

    // The 3D board is the play board: switching page only swaps the panels around it.
    const enterPlay = (fresh: boolean) => {
      playMeowSound(this.soundEnabled);
      const go = () => {
        this.mode = 'ai';
        this.syncModeControls();
        this.switchPage('play');
        if (fresh) this.resetGame();
      };
      // The transition only runs once the page paints; if painting is paused (hidden tab),
      // start the game anyway instead of waiting forever.
      let started = false;
      const once = () => {
        if (started) return;
        started = true;
        go();
      };
      const doc = document as Document & { startViewTransition?: (cb: () => void) => unknown };
      if (doc.startViewTransition && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
        doc.startViewTransition(once);
        window.setTimeout(once, 400);
      } else {
        once();
      }
    };

    if (!this.puzzle && this.mode === 'ai' && this.history.length > 0 && !this.winner && !this.isDraw && this.gameOpponent === this.opponent) {
      const resumeBtn = document.createElement('button');
      resumeBtn.type = 'button';
      resumeBtn.className = 'btn';
      resumeBtn.textContent = `이어하기 (${this.history.length}수)`;
      resumeBtn.addEventListener('click', () => enterPlay(false));
      actions.appendChild(resumeBtn);
    }

    const startBtn = document.createElement('button');
    startBtn.type = 'button';
    startBtn.className = 'btn primary stage-start';
    startBtn.textContent = '한 판 두기';
    startBtn.addEventListener('click', () => enterPlay(true));
    actions.appendChild(startBtn);

    const puzzles = allPuzzles();
    const solved = solvedIds();
    const puzzleBtn = document.createElement('button');
    puzzleBtn.type = 'button';
    puzzleBtn.className = 'btn';
    puzzleBtn.textContent = `묘수풀이 ${puzzles.filter((p) => solved.has(p.id)).length}/${puzzles.length}`;
    puzzleBtn.addEventListener('click', () => this.switchPage('puzzles'));
    actions.appendChild(puzzleBtn);

    const foot = document.createElement('div');
    foot.className = 'stage-foot';
    if (!user) {
      const form = document.createElement('form');
      form.className = 'login-form';
      const input = document.createElement('input');
      input.type = 'text';
      input.className = 'text-input';
      input.maxLength = 16;
      input.placeholder = '닉네임 (전적이 이 브라우저에 저장돼요)';
      input.setAttribute('aria-label', '닉네임');
      const submitBtn = document.createElement('button');
      submitBtn.type = 'submit';
      submitBtn.className = 'btn';
      submitBtn.textContent = '이름 알려주기';
      form.addEventListener('submit', (e) => {
        e.preventDefault();
        const nick = input.value.trim();
        if (!nick) return;
        loginUser(nick);
        this.renderLobby();
      });
      form.append(input, submitBtn);
      foot.appendChild(form);
    } else {
      const s = summarizeProfile(user);
      const who = document.createElement('span');
      who.className = 'stage-who';
      who.textContent = `${user.nickname} · ${formatRecord(s.wins, s.draws, s.losses)}${
        user.streak >= 2 ? ` · ${user.streak}연승 중` : ''
      }`;
      const outBtn = document.createElement('button');
      outBtn.type = 'button';
      outBtn.className = 'btn ghost';
      outBtn.textContent = '로그아웃';
      outBtn.addEventListener('click', () => {
        logoutUser();
        this.renderLobby();
      });
      foot.append(who, outBtn);
    }
    // The archive lives with the lobby actions, not in the top bar.
    const archBtn = document.createElement('button');
    archBtn.type = 'button';
    archBtn.className = 'btn ghost';
    archBtn.textContent = `기보실 (${getSavedMatches().length})`;
    archBtn.addEventListener('click', () => {
      this.historyMode = 'list';
      this.switchPage('history');
    });
    foot.appendChild(archBtn);
    const newsBtn = document.createElement('button');
    newsBtn.type = 'button';
    newsBtn.className = 'btn ghost news-btn';
    newsBtn.textContent = '새 소식';
    if (hasUnseenNews()) {
      newsBtn.classList.add('has-new');
      newsBtn.setAttribute('aria-label', '새 소식 (읽지 않음)');
    }
    newsBtn.addEventListener('click', () => this.openNews());
    foot.appendChild(newsBtn);
    const install = this.installButton();
    if (install) foot.appendChild(install);

    const panel = document.createElement('div');
    panel.className = 'stage-panel';
    panel.append(picker, chips, actions, foot);
    stage.append(this.cat.el, panel);
    this.lobbyEl.appendChild(stage);
  }

  private createSegmented(
    options: Array<{ value: string; label: string }>,
    initial: string,
    onChange: (value: string) => void,
  ): HTMLElement {
    const wrap = document.createElement('div');
    wrap.className = 'segmented';
    for (const opt of options) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = opt.label;
      if (opt.value === initial) btn.classList.add('active');
      btn.addEventListener('click', () => {
        if (btn.classList.contains('active')) return;
        for (const child of wrap.children) child.classList.remove('active');
        btn.classList.add('active');
        onChange(opt.value);
      });
      wrap.appendChild(btn);
    }
    return wrap;
  }

  private createQuietBtn(
    label: string,
    initial: boolean,
    onClick: (btn: HTMLButtonElement) => void,
  ): HTMLButtonElement {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'quiet-pill';
    if (initial) btn.classList.add('active');
    btn.textContent = label;
    btn.addEventListener('click', () => onClick(btn));
    return btn;
  }

  private createReplayBtn(
    text: string,
    title: string,
    onClick: () => void,
  ): HTMLButtonElement {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'mini-btn';
    btn.textContent = text;
    btn.title = title;
    btn.setAttribute('aria-label', title);
    btn.addEventListener('click', onClick);
    return btn;
  }

  private buildCellsGrid(): void {
    this.cellButtons = [];
    for (let r = 0; r < BOARD_SIZE; r++) {
      const rowBtns: HTMLButtonElement[] = [];
      for (let c = 0; c < BOARD_SIZE; c++) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'cell';
        btn.setAttribute('aria-label', formatCoord(r, c));

        btn.addEventListener('click', () => this.handleCellClick(r, c));
        btn.addEventListener('mouseenter', () => this.handleCellHover(r, c));
        btn.addEventListener('mouseleave', () => this.resetHintMessage());

        this.cellsGrid.appendChild(btn);
        rowBtns.push(btn);
      }
      this.cellButtons.push(rowBtns);
    }
  }

  // ── Rules Carousel Modal ──
  /** 새 소식: patch notes in a dialog; opening it clears the lobby badge. */
  private openNews(): void {
    const backdrop = document.createElement('div');
    backdrop.className = 'modal-backdrop';
    const dialog = document.createElement('div');
    dialog.className = 'modal-dialog news-dialog';
    dialog.setAttribute('role', 'dialog');
    dialog.setAttribute('aria-modal', 'true');
    dialog.setAttribute('aria-labelledby', 'news-title');
    const header = document.createElement('div');
    header.className = 'modal-header';
    header.innerHTML = '<h2 class="modal-title" id="news-title">새 소식</h2>';
    const closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'modal-close';
    closeBtn.textContent = '✕';
    closeBtn.setAttribute('aria-label', '닫기');
    header.appendChild(closeBtn);
    const body = document.createElement('div');
    body.className = 'news-body';
    const wasUnseen = hasUnseenNews();
    PATCH_NOTES.forEach((note, i) => {
      const sec = document.createElement('section');
      sec.className = 'news-entry';
      const head = document.createElement('div');
      head.className = 'news-entry-head';
      const title = document.createElement('h3');
      title.textContent = note.title;
      const date = document.createElement('span');
      date.className = 'news-date';
      date.textContent = note.date + (i === 0 && wasUnseen ? ' · NEW' : '');
      head.append(title, date);
      const list = document.createElement('ul');
      for (const it of note.items) {
        const li = document.createElement('li');
        const tag = document.createElement('span');
        tag.className = `news-tag tag-${it.tag}`;
        tag.textContent = it.tag;
        const text = document.createElement('span');
        text.textContent = it.text;
        li.append(tag, text);
        list.appendChild(li);
      }
      sec.append(head, list);
      body.appendChild(sec);
    });
    dialog.append(header, body);
    backdrop.appendChild(dialog);
    const close = () => {
      backdrop.remove();
      window.removeEventListener('keydown', onKey);
      if (this.activePage === 'lobby') this.renderLobby();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close();
    };
    closeBtn.addEventListener('click', close);
    backdrop.addEventListener('click', (e) => {
      if (e.target === backdrop) close();
    });
    window.addEventListener('keydown', onKey);
    this.root.appendChild(backdrop);
    markNewsSeen();
    closeBtn.focus();
  }

  private buildRulesModal(): void {
    this.rulesBackdropEl = document.createElement('div');
    this.rulesBackdropEl.className = 'modal-backdrop';
    this.rulesBackdropEl.hidden = true;
    this.rulesBackdropEl.addEventListener('click', (e) => {
      if (e.target === this.rulesBackdropEl) this.closeRulesModal();
    });

    const dialog = document.createElement('div');
    dialog.className = 'modal-dialog';
    dialog.setAttribute('role', 'dialog');
    dialog.setAttribute('aria-modal', 'true');

    const header = document.createElement('div');
    header.className = 'modal-header';

    const titleWrap = document.createElement('div');
    titleWrap.className = 'modal-title-wrap';
    this.modalStepEl = document.createElement('span');
    this.modalStepEl.className = 'modal-step';
    this.modalTitleEl = document.createElement('h2');
    this.modalTitleEl.className = 'modal-title';
    titleWrap.append(this.modalStepEl, this.modalTitleEl);

    const closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'modal-close';
    closeBtn.textContent = '✕';
    closeBtn.addEventListener('click', () => this.closeRulesModal());
    header.append(titleWrap, closeBtn);

    this.modalBodyEl = document.createElement('div');
    this.modalBodyEl.className = 'modal-body';

    const footer = document.createElement('div');
    footer.className = 'modal-footer';

    this.modalDotsEl = document.createElement('div');
    this.modalDotsEl.className = 'modal-dots';
    RULE_SLIDES.forEach((_, idx) => {
      const dot = document.createElement('button');
      dot.type = 'button';
      dot.className = 'modal-dot';
      dot.addEventListener('click', () => {
        this.ruleSlideIndex = idx;
        this.renderRulesModal();
      });
      this.modalDotsEl.appendChild(dot);
    });

    const nav = document.createElement('div');
    nav.className = 'modal-nav';
    this.modalPrevBtn = document.createElement('button');
    this.modalPrevBtn.type = 'button';
    this.modalPrevBtn.className = 'btn';
    this.modalPrevBtn.textContent = '이전';
    this.modalPrevBtn.addEventListener('click', () => this.stepRulesSlide(-1));

    this.modalNextBtn = document.createElement('button');
    this.modalNextBtn.type = 'button';
    this.modalNextBtn.className = 'btn primary';
    this.modalNextBtn.textContent = '다음';
    this.modalNextBtn.addEventListener('click', () => this.stepRulesSlide(1));

    nav.append(this.modalPrevBtn, this.modalNextBtn);
    footer.append(this.modalDotsEl, nav);
    dialog.append(header, this.modalBodyEl, footer);
    this.rulesBackdropEl.appendChild(dialog);
    this.root.appendChild(this.rulesBackdropEl);
  }

  private openRulesModal(): void {
    this.showRulesModal = true;
    this.rulesBackdropEl.hidden = false;
    this.renderRulesModal();
  }

  private closeRulesModal(): void {
    this.showRulesModal = false;
    this.rulesBackdropEl.hidden = true;
  }

  private stepRulesSlide(delta: number): void {
    const next = this.ruleSlideIndex + delta;
    if (next < 0 || next >= RULE_SLIDES.length) return;
    this.ruleSlideIndex = next;
    this.renderRulesModal();
  }

  private renderRulesModal(): void {
    const slide = RULE_SLIDES[this.ruleSlideIndex];
    this.modalStepEl.textContent = `${this.ruleSlideIndex + 1} / ${RULE_SLIDES.length}`;
    this.modalTitleEl.textContent = slide.title;
    this.modalBodyEl.innerHTML = '';

    const desc = document.createElement('p');
    desc.className = 'modal-desc';
    desc.innerHTML = slide.desc;
    this.modalBodyEl.appendChild(desc);

    if (slide.diagram) {
      const diag = document.createElement('div');
      diag.className = 'mini-diagram';
      for (const rowStr of slide.diagram) {
        for (const ch of rowStr) {
          const cell = document.createElement('span');
          cell.className = 'mini-cell';
          if (ch === 'b' || ch === 'w') {
            const st = document.createElement('span');
            st.className = `mini-stone ${ch}`;
            cell.appendChild(st);
          } else if (ch === 'X') {
            const mk = document.createElement('span');
            mk.className = 'mini-mark x';
            mk.textContent = '✕';
            cell.appendChild(mk);
          } else if (ch === 'O') {
            const mk = document.createElement('span');
            mk.className = 'mini-mark ok';
            mk.textContent = '◎';
            cell.appendChild(mk);
          }
          diag.appendChild(cell);
        }
      }
      this.modalBodyEl.appendChild(diag);
    }

    const ul = document.createElement('ul');
    ul.className = 'modal-bullets';
    for (const item of slide.bullets) {
      const li = document.createElement('li');
      li.innerHTML = item;
      ul.appendChild(li);
    }
    this.modalBodyEl.appendChild(ul);

    Array.from(this.modalDotsEl.children).forEach((dot, idx) => {
      dot.classList.toggle('active', idx === this.ruleSlideIndex);
    });

    this.modalPrevBtn.disabled = this.ruleSlideIndex === 0;
    const isLast = this.ruleSlideIndex === RULE_SLIDES.length - 1;
    this.modalNextBtn.textContent = isLast ? '완료' : '다음';
    this.modalNextBtn.onclick = isLast
      ? () => this.closeRulesModal()
      : () => this.stepRulesSlide(1);
  }

  // ── 묘수풀이: four after four on the play board; the cat blocks ──
  private catSay([mood, lines]: [CatMood, string[]], seed = Date.now()): void {
    this.cat.setMood(mood);
    this.cat.say(pickLine(lines, seed));
  }

  private nextPuzzle(afterId: string): Puzzle | null {
    const theme = allPuzzles().find((p) => p.id === afterId)?.theme;
    const list = allPuzzles().filter((p) => p.theme === theme);
    const solved = solvedIds();
    const i = list.findIndex((p) => p.id === afterId);
    const rest = [...list.slice(i + 1), ...list.slice(0, Math.max(0, i))];
    return rest.find((p) => !solved.has(p.id)) ?? rest[0] ?? null;
  }

  private startPractice(setup: { moves: MoveRecord[]; human: Player; cat: CatId }): void {
    if (this.activePage !== 'play') {
      this.mode = 'ai';
      this.syncModeControls();
      this.switchPage('play');
    }
    this.resetGame();
    this.aiSearchToken++; // drop the fresh game's first AI move, if any
    if (this.aiTimer) window.clearTimeout(this.aiTimer);
    this.aiTimer = null;
    this.aiThinking = false;
    this.sideBeforePuzzle = this.humanPlayer;
    this.practice = setup;
    this.humanPlayer = setup.human;
    this.gameOpponent = setup.cat;
    this.undoLeft = UNDO_LIMIT[setup.cat];
    this.history = setup.moves.map((m) => ({ ...m }));
    this.board = boardAt(this.history, this.history.length);
    const last = this.history[this.history.length - 1];
    this.currentPlayer = last ? opponentOf(last.player) : BLACK;
    this.viewMoveIndex = this.history.length;
    this.catSeenMoves = this.history.length; // don't comment on moves already played
    this.cat.setCharacter(setup.cat, CAT_NAME[setup.cat]);
    this.catSay(this.voice().rematch);
    this.recomputeAndRender();
    this.maybeTriggerAI();
  }

  private renderPuzzleList(): void {
    this.puzzleListEl.innerHTML = '';
    const wrap = document.createElement('div');
    wrap.className = 'archive-single puzzle-list';
    const solved = solvedIds();
    const all = allPuzzles();
    for (const theme of THEMES) {
      const items = all.filter((p) => p.theme === theme.id);
      if (items.length === 0) continue;
      const card = document.createElement('section');
      card.className = 'lobby-card puzzle-theme';
      const done = items.filter((p) => solved.has(p.id)).length;
      card.innerHTML = `<div class="lobby-card-head"><span class="lobby-card-title">${theme.title}</span><span class="lobby-card-sub">${done}/${items.length}</span></div><p class="puzzle-theme-desc">${theme.desc}</p>`;
      const grid = document.createElement('div');
      grid.className = 'puzzle-grid';
      items.forEach((p, i) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = `puzzle-card${solved.has(p.id) ? ' solved' : ''}`;
        b.setAttribute('aria-label', `${theme.title} ${i + 1}번, ${p.moves}수${solved.has(p.id) ? ', 풀었음' : ''}`);
        b.innerHTML = `${miniBoardSvg(p)}<span class="puzzle-card-label"><span>${i + 1}</span><span class="puzzle-card-meta">${p.attacker === BLACK ? '흑' : '백'} · ${p.moves}수</span></span>`;
        b.addEventListener('click', () => this.startPuzzle(p));
        grid.appendChild(b);
      });
      card.appendChild(grid);
      wrap.appendChild(card);
    }
    this.puzzleListEl.appendChild(wrap);
  }

  private startPuzzle(p: Puzzle): void {
    if (this.activePage !== 'play') {
      this.mode = 'ai';
      this.syncModeControls();
      this.switchPage('play');
    }
    this.resetGame(); // clears the live game and any previous puzzle
    if (this.aiTimer) window.clearTimeout(this.aiTimer);
    this.aiTimer = null;
    this.sideBeforePuzzle = this.humanPlayer;
    this.puzzle = p;
    this.puzzleState = 'solving';
    this.puzzleHintUsed = false;
    this.aiThinking = false;
    this.humanPlayer = p.attacker;
    this.currentPlayer = p.attacker;
    this.board = puzzleBoard(p); // the starting stones are not moves
    this.history = [];
    this.catSay(this.voice().puzzle.intro(p.moves), p.moves);
    this.recomputeAndRender();
  }

  /** Leave a puzzle or a practice position: clear its board and give back the lobby side. */
  private leavePuzzle(): void {
    if (!this.puzzle && !this.practice) return;
    this.practice = null;
    this.aiSearchToken++; // drop a pending block
    if (this.aiTimer) window.clearTimeout(this.aiTimer);
    this.aiTimer = null;
    this.aiThinking = false;
    this.puzzle = null;
    if (this.sideBeforePuzzle !== null) this.humanPlayer = this.sideBeforePuzzle;
    this.sideBeforePuzzle = null;
    this.board = createEmptyBoard();
    this.history = [];
    this.currentPlayer = BLACK;
    this.winner = null;
    this.isDraw = false;
    this.hintCells = [];
  }

  private puzzleDefend(): void {
    const p = this.puzzle;
    if (!p) return;
    this.aiThinking = true;
    this.cat.setMood('think');
    this.syncPlayActions();
    const token = ++this.aiSearchToken;
    this.aiTimer = window.setTimeout(() => {
      if (token !== this.aiSearchToken || this.puzzle !== p) return;
      this.aiTimer = null;
      this.aiThinking = false;
      const defender = this.currentPlayer;
      // Its own five, else the block; a forbidden block leaves it any move at all.
      const block = forcedMove(this.board, defender, RULES);
      const move = block ?? chooseAIMove(this.board, defender, RULES, 'lv4');
      if (!move) return;
      this.placeStone(move.row, move.col);
      if (this.winner || this.puzzle !== p) return; // endPuzzle already spoke
      if (!block) {
        this.catSay(this.voice().puzzle.trapped);
      } else if (!stillWinning(this.board, p.attacker, RULES)) {
        this.endPuzzle(false);
      } else {
        this.catSay(this.voice().puzzle.blocked(formatCoord(move.row, move.col)), this.history.length);
      }
    }, 520);
  }

  private endPuzzle(solved: boolean): void {
    const p = this.puzzle;
    if (!p) return;
    this.puzzleState = solved ? 'solved' : 'failed';
    if (solved) markSolved(p.id);
    this.catSay(solved ? this.voice().puzzle.solved : this.voice().puzzle.failed);
    this.hintCells = [];
    this.recomputeAndRender();
  }

  private renderPuzzleBar(): void {
    const p = this.puzzle;
    if (!p) return;
    const list = allPuzzles().filter((x) => x.theme === p.theme);
    const head = document.createElement('div');
    head.className = 'hint-bar-head';
    const n = list.findIndex((x) => x.id === p.id) + 1;
    head.textContent = `${THEMES.find((t) => t.id === p.theme)?.title ?? '묘수풀이'} ${n}/${list.length}`;
    const row = document.createElement('div');
    row.className = 'hint-bar-row';
    const solving = this.puzzleState === 'solving' && !this.aiThinking;
    const make = (label: string, disabled: boolean, onClick: () => void) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'hint-btn';
      b.textContent = label;
      b.disabled = disabled;
      b.addEventListener('click', onClick);
      row.appendChild(b);
    };
    make('첫 수 보기', !solving || this.puzzleHintUsed || this.history.length > 0, () => {
      const k = p.line[0];
      this.puzzleHintUsed = true;
      this.hintCells = [{ row: Math.floor(k / 15), col: k % 15 }];
      this.catSay(this.voice().puzzle.hint(formatCoord(Math.floor(k / 15), k % 15)));
      this.recomputeAndRender();
    });
    make('처음부터', this.aiThinking || this.history.length === 0, () => this.startPuzzle(p));
    make('목록', false, () => this.switchPage('puzzles'));
    this.hintBarEl.append(head, row);
  }

  // ── Review: the game's cat points at where it turned ──
  private startReview(human: Player | null, winner: Player | null, cat: CatId): void {
    const token = ++this.reviewToken;
    this.replayHuman = human;
    this.replayCat = cat;
    this.reviewFocus = null;
    this.turningPoints = human === null ? [] : null;
    this.reviewCat.setCharacter(cat, CAT_NAME[cat]);
    const voice = this.voice(cat).review;
    const say = ([mood, lines]: [CatMood, string[]]) => {
      this.reviewCat.setMood(mood);
      this.reviewCat.say(pickLine(lines, Date.now()));
    };
    if (human === null) {
      this.reviewCat.setMood('idle');
      this.reviewCat.say(null);
      return;
    }
    say(voice.scanning);
    // Let the scanning line paint before the (synchronous) search.
    window.setTimeout(async () => {
      if (token !== this.reviewToken) return;
      // The cat's own engine checks each candidate, so a "better move" is one it would play.
      const ckpt = this.getActiveCheckpoint();
      const analyze: Analyzer | undefined = ckpt
        ? async (b, p, last) => {
            const r = await chooseCheckpointMoveAsync(b, p, ckpt.weights, last, 280, 1200);
            return r ? { best: { row: r.row, col: r.col }, value: r.value } : null;
          }
        : undefined;
      const tps = await findTurningPoints(this.replayMoves, human, winner, RULES, analyze);
      if (token !== this.reviewToken) return;
      this.turningPoints = tps;
      this.reviewPuzzles = 0;
      for (const tp of tps) {
        if (tp.kind !== 'missedWin') continue;
        const p = puzzleFromGame(this.replayMoves, tp.index, human, RULES);
        if (p) {
          addMyPuzzle(p);
          this.reviewPuzzles++;
        }
      }
      say(tps.length ? voice.summary(tps.length) : winner === human ? voice.clean : voice.none);
      if (this.activePage === 'history') this.renderReviewPanel();
    }, 60);
  }

  private focusTurningPoint(tp: TurningPoint): void {
    if (this.replayHuman === null) return;
    this.reviewToken++; // a scan still running for this replay no longer speaks
    const voice = this.voice(this.replayCat).review;
    const cell = tp.kind === 'missedWin' ? tp.win?.[0] : tp.save;
    if (!cell) return;
    this.stopAutoPlay();
    this.reviewFocus = { index: tp.index, cells: [cell], seq: tp.kind === 'missedWin' ? tp.win ?? null : null };
    const c = formatCoord(cell.row, cell.col);
    const [mood, lines] =
      tp.kind === 'missedWin' ? voice.missedWin(c, Math.ceil((tp.win?.length ?? 1) / 2)) : voice.turn(c);
    this.reviewCat.setMood(mood);
    this.reviewCat.say(pickLine(lines, tp.index));
    this.viewMoveIndex = -1; // force jumpToMove to redraw even when already there
    this.jumpToMove(tp.index);
  }

  private renderReviewPanel(): void {
    this.nnSubEl.innerHTML = '';
    const panel = document.createElement('div');
    panel.className = 'review-panel';

    const head = document.createElement('div');
    head.className = 'review-head';
    head.textContent = '승부처';
    panel.appendChild(head);

    if (this.replayHuman === null) {
      const note = document.createElement('p');
      note.className = 'review-note';
      note.textContent = '2인 대국은 고양이가 따로 짚어 주지 않아요. 한 수씩 넘겨 보세요.';
      panel.appendChild(note);
      this.nnSubEl.appendChild(panel);
      return;
    }

    this.reviewCat.el.classList.add('review-cat');
    panel.appendChild(this.reviewCat.el);

    const chips = document.createElement('div');
    chips.className = 'review-chips';
    const LABEL = { missedWin: '놓친 필승', turn: '판이 넘어간 수' } as const;
    for (const tp of this.turningPoints ?? []) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = `review-chip ${tp.kind}`;
      b.classList.toggle('active', this.reviewFocus?.index === tp.index && this.viewMoveIndex === tp.index);
      b.innerHTML = `<span class="review-chip-num">${tp.index + 1}수</span><span>${LABEL[tp.kind]}</span>`;
      b.addEventListener('click', () => void this.focusTurningPoint(tp));
      chips.appendChild(b);
    }
    panel.appendChild(chips);

    // Play on against the same cat from the position on the board.
    const view = this.getViewState();
    const human = this.replayHuman;
    const canPlay = !view.winner && !view.isDraw && view.turnPlayer === human;
    const playBtn = document.createElement('button');
    playBtn.type = 'button';
    playBtn.className = 'btn primary review-play';
    playBtn.textContent = canPlay ? `${this.viewMoveIndex + 1}수부터 다시 두기` : '내 차례인 수에서 다시 둘 수 있어요';
    playBtn.disabled = !canPlay;
    playBtn.addEventListener('click', () =>
      this.startPractice({ moves: this.replayMoves.slice(0, this.viewMoveIndex), human, cat: this.replayCat }),
    );
    panel.appendChild(playBtn);

    if (this.turningPoints && this.reviewPuzzles > 0) {
      const note = document.createElement('p');
      note.className = 'review-note';
      note.textContent = '놓친 필승은 묘수풀이 문제로도 저장돼요.';
      panel.appendChild(note);
    }
    this.nnSubEl.appendChild(panel);
  }

  private bindKeyboard(): void {
    window.addEventListener('keydown', (e) => {
      if (this.showRulesModal) {
        if (e.key === 'Escape') this.closeRulesModal();
        else if (e.key === 'ArrowLeft') this.stepRulesSlide(-1);
        else if (e.key === 'ArrowRight') this.stepRulesSlide(1);
        return;
      }
      if (
        this.activePage === 'history' &&
        this.historyMode === 'viewer' &&
        this.getActiveReplayMoves().length > 0
      ) {
        if (e.key === 'ArrowLeft') {
          e.preventDefault();
          this.jumpToMove(this.viewMoveIndex - 1);
        } else if (e.key === 'ArrowRight') {
          e.preventDefault();
          this.jumpToMove(this.viewMoveIndex + 1);
        } else if (e.key === ' ') {
          e.preventDefault();
          this.toggleAutoPlay();
        }
      }
    });
  }

  // ── Replay & RL Archive (Strictly isolated from live match state) ──
  private getActiveReplayMoves(): MoveRecord[] {
    return this.replayMoves;
  }

  private isReplaying(): boolean {
    if (this.activePage !== 'history') return false;
    return this.viewMoveIndex < this.getActiveReplayMoves().length;
  }

  private jumpToMove(index: number): void {
    const total = this.getActiveReplayMoves().length;
    const clamped = Math.max(0, Math.min(total, index));
    if (clamped === this.viewMoveIndex) return;
    if (this.autoPlayTimer !== null && clamped >= total) {
      this.stopAutoPlay();
    }
    this.viewMoveIndex = clamped;
    this.showVcfPreview = false;
    this.lastPlacedKey = null;
    this.recomputeAndRender();
  }

  private toggleAutoPlay(): void {
    if (this.autoPlayTimer !== null) {
      this.stopAutoPlay();
      this.renderHistory();
      return;
    }
    const total = this.getActiveReplayMoves().length;
    if (total === 0) return;
    if (this.viewMoveIndex >= total) {
      this.viewMoveIndex = 0;
      this.showVcfPreview = false;
      this.recomputeAndRender();
    }
    this.autoPlayTimer = window.setInterval(() => {
      const len = this.getActiveReplayMoves().length;
      if (this.viewMoveIndex >= len) {
        this.stopAutoPlay();
        this.renderHistory();
        return;
      }
      this.viewMoveIndex++;
      this.showVcfPreview = false;
      this.recomputeAndRender();
    }, 620);
    this.renderHistory();
  }

  private stopAutoPlay(): void {
    if (this.autoPlayTimer !== null) {
      window.clearInterval(this.autoPlayTimer);
      this.autoPlayTimer = null;
    }
  }

  private copyHistory(): void {
    const moves = this.getActiveReplayMoves();
    if (moves.length === 0) return;
    const text = moves
      .map(
        (m) =>
          `${m.moveNumber}. ${m.player === BLACK ? '흑' : '백'} ${formatCoord(m.row, m.col)}`,
      )
      .join('  ');
    void navigator.clipboard?.writeText(text).then(() => {
      this.copyHistoryBtn.textContent = '복사됨';
      window.setTimeout(() => {
        this.copyHistoryBtn.textContent = '복사';
      }, 1400);
    });
  }

  private getViewState(): {
    board: Board;
    moves: MoveRecord[];
    turnPlayer: Player;
    winner: WinResult | null;
    isDraw: boolean;
  } {
    if (this.activePage !== 'history') {
      return {
        board: this.board,
        moves: this.history,
        turnPlayer: this.currentPlayer,
        winner: this.winner,
        isDraw: this.isDraw,
      };
    }

    const allMoves = this.getActiveReplayMoves();
    const clampedIndex = Math.max(0, Math.min(allMoves.length, this.viewMoveIndex));
    const viewBoard = createEmptyBoard();
    const moves = allMoves.slice(0, clampedIndex);
    for (const m of moves) {
      viewBoard[m.row][m.col] = m.player;
    }
    const last = moves.length > 0 ? moves[moves.length - 1] : null;
    const winner = checkWinner(viewBoard, last, RULES);
    const turnPlayer = last ? opponentOf(last.player) : BLACK;
    const isDraw = !winner && !hasAnyLegalMove(viewBoard, turnPlayer, RULES);

    return { board: viewBoard, moves, turnPlayer, winner, isDraw };
  }

  // ── Gameplay ──
  private canHumanInteract(): boolean {
    if (this.activePage !== 'play') return false;
    if (this.winner || this.isDraw || this.aiThinking) return false;
    if (this.puzzle && this.puzzleState !== 'solving') return false;
    if (this.mode === 'ai' && this.currentPlayer !== this.humanPlayer) return false;
    return true;
  }

  private coordFromClientPoint(clientX: number, clientY: number): Coord | null {
    const rect = this.cellsGrid.getBoundingClientRect();
    if (
      rect.width <= 0 ||
      rect.height <= 0 ||
      clientX < rect.left ||
      clientX > rect.right ||
      clientY < rect.top ||
      clientY > rect.bottom
    ) {
      return null;
    }
    const col = Math.min(
      BOARD_SIZE - 1,
      Math.max(0, Math.floor(((clientX - rect.left) / rect.width) * BOARD_SIZE)),
    );
    const row = Math.min(
      BOARD_SIZE - 1,
      Math.max(0, Math.floor(((clientY - rect.top) / rect.height) * BOARD_SIZE)),
    );
    return { row, col };
  }

  private bindTouchAiming(): void {
    let startX = 0;
    let startY = 0;

    this.cellsGrid.addEventListener('pointerdown', (e) => {
      if (e.pointerType === 'mouse') return;
      this.lastTouchTime = performance.now();
      if (!this.canHumanInteract()) return;
      if (this.activeTouchId !== null) return;

      this.activeTouchId = e.pointerId;
      this.touchDragMoved = false;
      this.touchStartAim = this.pendingAim ? { ...this.pendingAim } : null;
      startX = e.clientX;
      startY = e.clientY;
      try {
        this.cellsGrid.setPointerCapture(e.pointerId);
      } catch {
        // Ignore capture failure on synthetic test events
      }

      const coord = this.coordFromClientPoint(e.clientX, e.clientY);
      if (coord && this.board[coord.row][coord.col] === EMPTY) {
        this.setPendingAim(coord);
      }
    });

    this.cellsGrid.addEventListener('pointermove', (e) => {
      if (e.pointerType === 'mouse' || e.pointerId !== this.activeTouchId) return;
      this.lastTouchTime = performance.now();
      if (!this.canHumanInteract()) return;

      if (Math.hypot(e.clientX - startX, e.clientY - startY) > 6) {
        this.touchDragMoved = true;
      }
      const coord = this.coordFromClientPoint(e.clientX, e.clientY);
      if (!coord) {
        this.setPendingAim(null);
      } else if (this.board[coord.row][coord.col] === EMPTY) {
        this.setPendingAim(coord);
      }
    });

    const finishTouch = (e: PointerEvent, cancelled: boolean) => {
      if (e.pointerType === 'mouse' || e.pointerId !== this.activeTouchId) return;
      this.activeTouchId = null;
      this.lastTouchTime = performance.now();

      if (cancelled || !this.canHumanInteract()) {
        this.setPendingAim(null);
        return;
      }

      const coord = this.coordFromClientPoint(e.clientX, e.clientY);
      if (!coord) {
        this.setPendingAim(null);
        return;
      }
      if (this.board[coord.row][coord.col] !== EMPTY) {
        return;
      }

      if (!this.confirmAimMode) {
        this.setPendingAim(null);
        this.tryPlaceAt(coord.row, coord.col);
        return;
      }

      const sameAsStart =
        !this.touchDragMoved &&
        Boolean(this.touchStartAim) &&
        this.touchStartAim?.row === coord.row &&
        this.touchStartAim?.col === coord.col;

      if (sameAsStart) {
        this.setPendingAim(null);
        this.tryPlaceAt(coord.row, coord.col);
      } else {
        this.setPendingAim(coord);
      }
    };

    this.cellsGrid.addEventListener('pointerup', (e) => finishTouch(e, false));
    this.cellsGrid.addEventListener('pointercancel', (e) => finishTouch(e, true));
  }

  private setPendingAim(coord: Coord | null): void {
    this.pendingAim = coord;
    this.updateAimVisuals();
  }

  private updateAimVisuals(skipOverlay = false): void {
    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        this.cellButtons[r][c].classList.remove('aimed');
      }
    }

    const active = this.pendingAim && this.canHumanInteract() ? this.pendingAim : null;
    if (!active || this.board[active.row][active.col] !== EMPTY) {
      if (this.aimFloatBadgeEl) this.aimFloatBadgeEl.hidden = true;
      if (this.aimConfirmBarEl) this.aimConfirmBarEl.hidden = true;
      if (!skipOverlay) this.renderSvgOverlays();
      return;
    }

    this.cellButtons[active.row][active.col].classList.add('aimed');
    const coordStr = formatCoord(active.row, active.col);
    const forbidden = this.forbiddenMap.get(coordKey(active.row, active.col));

    if (this.aimFloatBadgeEl) {
      this.aimFloatBadgeEl.hidden = false;
      this.aimFloatBadgeEl.classList.toggle('is-forbidden', Boolean(forbidden));
      this.aimFloatBadgeEl.classList.toggle('at-bottom', active.row < 4);
      this.aimFloatBadgeEl.textContent = forbidden
        ? `${coordStr} · ${forbiddenLabel(forbidden.reason).short} 금수`
        : coordStr;
    }

    if (this.aimConfirmBarEl && this.aimConfirmTextEl && this.aimConfirmBtn) {
      // Only two-tap mode needs the confirm bar; in one-tap mode it just flashed under the finger.
      this.aimConfirmBarEl.hidden = !this.confirmAimMode;
      this.aimConfirmTextEl.textContent = forbidden
        ? `${coordStr} — 금수 자리입니다`
        : `${coordStr} 조준 중 (드래그/탭으로 이동)`;
      this.aimConfirmBtn.disabled = Boolean(forbidden);
      this.aimConfirmBtn.textContent = `${coordStr} 착수`;
    }

    if (!skipOverlay) this.renderSvgOverlays();
  }

  private handleCellClick(row: number, col: number): void {
    if (performance.now() - this.lastTouchTime < 700) {
      return;
    }
    if (!this.canHumanInteract()) return;
    if (this.board[row][col] !== EMPTY) return;

    if (this.confirmAimMode) {
      if (
        this.pendingAim &&
        this.pendingAim.row === row &&
        this.pendingAim.col === col
      ) {
        this.setPendingAim(null);
        this.tryPlaceAt(row, col);
      } else {
        this.setPendingAim({ row, col });
      }
      return;
    }

    this.tryPlaceAt(row, col);
  }

  private tryPlaceAt(row: number, col: number): void {
    if (!this.canHumanInteract()) return;
    if (this.board[row][col] !== EMPTY) return;

    const analysis = analyzeMove(
      this.board,
      row,
      col,
      this.currentPlayer,
      RULES,
    );

    if (!analysis.legal && analysis.forbiddenReason) {
      this.triggerForbiddenFeedback(row, col, analysis.forbiddenReason);
      return;
    }

    if (this.puzzle && checkAttack(this.board, { row, col }, this.currentPlayer, RULES) === 'notFour') {
      playForbiddenSound(this.soundEnabled);
      const btn = this.cellButtons[row][col];
      btn.classList.remove('shake');
      void btn.offsetWidth;
      btn.classList.add('shake');
      this.catSay(this.voice().puzzle.notFour, row * 15 + col);
      return;
    }

    this.placeStone(row, col);
  }

  private recordFinishedMatch(): void {
    if (this.matchRecorded) return;
    if (!this.winner && !this.isDraw) return;
    this.matchRecorded = true;
    if (this.practice) return; // practice from a replay isn't a real game
    if (this.puzzle) {
      this.endPuzzle(this.winner?.winner === this.puzzle.attacker);
      return;
    }

    const user = getCurrentUser();
    const ch = this.getMasterChar();

    saveMatchRecord({
      playerNickname: user?.nickname ?? '나',
      mode: this.mode,
      charId: this.mode === 'ai' ? ch.id : undefined,
      opponentName: this.mode === 'ai' ? ch.name : '2인 대국',
      opponentBadge: this.mode === 'ai' ? ch.badge : '👥',
      opponentTier: this.mode === 'ai' ? ch.tier : undefined,
      humanPlayer: this.mode === 'ai' ? this.humanPlayer : BLACK,
      winner: this.winner ? this.winner.winner : null,
      isDraw: this.isDraw,
      moves: this.history.map((m) => ({ ...m })),
    });

    if (this.mode === 'ai') {
      const outcome = this.winner
        ? this.winner.winner === this.humanPlayer
          ? 'win'
          : 'loss'
        : 'draw';
      recordAiMatchResult(this.masterCharId, outcome, this.history.length);
    }
  }

  private placeStone(row: number, col: number): void {
    this.stopAutoPlay();
    this.pendingAim = null;
    const player = this.currentPlayer;
    const tag = getMoveTagOnBoard(this.board, row, col, player);
    this.board[row][col] = player;

    const moveNumber = this.history.length + 1;
    const win = checkWinner(this.board, { row, col }, RULES);
    const nextPlayer = opponentOf(player);
    let evalScore: number;
    if (win) {
      evalScore = win.winner === BLACK ? 1 : -1;
    } else {
      const activeCkpt = this.mode === 'ai' ? this.getActiveCheckpoint() : null;
      if (activeCkpt) {
        // Same post-move NN eval for both sides; the MCTS root mean describes the
        // pre-move position, so mixing the two produced fake "AI blunder" swings.
        const ev = evaluateD4OverlapPattern(
          activeCkpt.weights,
          this.board,
          nextPlayer,
          { row, col },
        );
        const nnBlack = nextPlayer === BLACK ? ev.value : -ev.value;
        evalScore = Math.max(-0.98, Math.min(0.98, nnBlack));
      } else {
        evalScore = evaluatePositionAdvantage(this.board, nextPlayer, RULES);
      }
    }

    // A lone eval flips with the side to move (tempo); averaging with the previous ply cancels it.
    const rawEval = evalScore;
    const prevRaw = this.history[this.history.length - 1]?.rawEval;
    if (!win && prevRaw !== undefined) evalScore = (evalScore + prevRaw) / 2;
    this.history.push({ row, col, player, moveNumber, tag, evalScore, rawEval });
    this.viewMoveIndex = this.history.length;
    this.showVcfPreview = false;
    this.lastPlacedKey = coordKey(row, col);
    if (player === this.humanPlayer) {
      this.hintCells = [];
      this.hideHelper();
    }

    if (win) {
      this.winner = win;
      playWinSound(this.soundEnabled);
      this.recordFinishedMatch();
    } else if (!hasAnyLegalMove(this.board, nextPlayer, RULES)) {
      this.isDraw = true;
      const last = this.history[this.history.length - 1];
      if (last) last.evalScore = 0;
      playStoneSound(this.soundEnabled);
      this.recordFinishedMatch();
    } else {
      this.currentPlayer = nextPlayer;
      playStoneSound(this.soundEnabled);
    }

    this.recomputeAndRender();
    this.maybeTriggerAI();
  }

  private triggerForbiddenFeedback(
    row: number,
    col: number,
    reason: ForbiddenReason,
  ): void {
    playForbiddenSound(this.soundEnabled);
    const btn = this.cellButtons[row][col];
    btn.classList.remove('shake');
    void btn.offsetWidth;
    btn.classList.add('shake');

    const { title } = forbiddenLabel(reason);
    const coord = formatCoord(row, col);
    this.setHint(`${coord} · 흑 금수 (${title})`, 'danger');
    if (this.mode === 'ai') {
      this.cat.setMood('warn');
      this.cat.say(catForbiddenLine(coord, reason, row * 15 + col, this.voice()), 'warn');
    }
    this.renderSeats();

    if (this.hoverWarningTimer) window.clearTimeout(this.hoverWarningTimer);
    this.hoverWarningTimer = window.setTimeout(() => {
      this.hoverWarningTimer = null;
      this.resetHintMessage();
      this.renderSeats();
    }, 1800);
  }

  private handleCellHover(row: number, col: number): void {
    if (
      this.activePage !== 'play' ||
      this.isReplaying() ||
      this.winner ||
      this.isDraw ||
      this.aiThinking
    ) {
      return;
    }
    if (this.board[row][col] !== EMPTY) return;
    if (this.mode === 'ai' && this.currentPlayer !== this.humanPlayer) return;

    const coord = formatCoord(row, col);
    const forbidden = this.forbiddenMap.get(coordKey(row, col));
    if (forbidden) {
      const { title } = forbiddenLabel(forbidden.reason);
      this.setHint(`${coord} · 금수 (${title})`, 'danger');
      return;
    }

    const analysis = analyzeMove(
      this.board,
      row,
      col,
      this.currentPlayer,
      RULES,
    );

    if (analysis.isWin) {
      this.setHint(`${coord} · 5목 승리!`, 'win');
      return;
    }

    const totalFours = analysis.openFours.length + analysis.closedFours.length;
    if (totalFours >= 1 && analysis.openThrees.length >= 1) {
      this.setHint(`${coord} · 4·3 공격`, 'four');
      return;
    }
    if (analysis.openFours.length >= 1) {
      this.setHint(`${coord} · 열린 4`, 'four');
      return;
    }
    if (analysis.closedFours.length >= 1) {
      this.setHint(`${coord} · 4목`, 'four');
      return;
    }
    if (analysis.openThrees.length >= 1) {
      this.setHint(
        `${coord} · 열린 3${analysis.openThrees.length > 1 ? ` (${analysis.openThrees.length})` : ''}`,
        'three',
      );
      return;
    }

    const opp = opponentOf(this.currentPlayer);
    const blocksOppFour = this.threats.some(
      (t) =>
        t.player === opp &&
        (t.type === 'OPEN_FOUR' || t.type === 'CLOSED_FOUR') &&
        t.targetCells.some((c) => c.row === row && c.col === col),
    );
    if (blocksOppFour) {
      this.setHint(`${coord} · 상대 4목 방어`, 'four');
      return;
    }

    this.setHint(`${coord}`, 'default');
  }

  private setHint(
    text: string,
    tone: 'default' | 'danger' | 'four' | 'three' | 'win',
  ): void {
    this.statusHintEl.textContent = text;
    this.statusHintEl.className =
      tone === 'default' ? 'status-hint' : `status-hint ${tone}`;
  }

  private resetHintMessage(): void {
    if (this.hoverWarningTimer) return;
    if (this.activePage === 'history') {
      const total = this.getActiveReplayMoves().length;
      if (this.viewMoveIndex < total) {
        this.setHint(`복기 중 (${this.viewMoveIndex}/${total})`, 'default');
      } else {
        this.setHint('내 대국 기보', 'default');
      }
      return;
    }
    if (this.winner) {
      this.setHint(`${this.winner.line.length}목 완성`, 'win');
      return;
    }
    if (this.isDraw) {
      const noLegalMovesLeft = this.history.length < BOARD_SIZE * BOARD_SIZE;
      this.setHint(
        noLegalMovesLeft ? '둘 수 있는 자리 없음 · 무승부' : '바둑판 만석 · 무승부',
        'default',
      );
      return;
    }
    if (this.aiThinking) return;

    const fours = this.threats.filter(
      (t) => t.type === 'OPEN_FOUR' || t.type === 'CLOSED_FOUR',
    );
    const openThrees = this.threats.filter((t) => t.type === 'OPEN_THREE');
    if (fours.length > 0) {
      const f = fours[0];
      this.setHint(
        `${playerLabel(f.player)} ${f.type === 'OPEN_FOUR' ? '열린 4' : '4목'} 위협`,
        'four',
      );
      return;
    }
    if (openThrees.length > 0) {
      const t = openThrees[0];
      this.setHint(`${playerLabel(t.player)} 열린 3`, 'three');
      return;
    }
    if (this.currentPlayer === BLACK && this.forbiddenMap.size > 0) {
      this.setHint(`금수 ${this.forbiddenMap.size}곳`, 'danger');
      return;
    }
    this.setHint('', 'default');
  }

  private maybeTriggerAI(): void {
    if (this.mode === 'pvp') return;
    if (this.winner || this.isDraw) return;
    if (this.mode === 'ai' && this.currentPlayer === this.humanPlayer) return;
    if (this.puzzle) {
      this.puzzleDefend();
      return;
    }

    this.aiThinking = true;
    const searchToken = ++this.aiSearchToken;
    this.renderStatus();
    this.renderSeats();
    this.resetHintMessage();
    this.syncPlayActions();

    const targetPaceMs = 760;

    if (this.aiTimer) window.clearTimeout(this.aiTimer);
    this.aiTimer = window.setTimeout(async () => {
      if (searchToken !== this.aiSearchToken || !this.aiThinking) return;
      const started = performance.now();
      const activeCkpt = this.getActiveCheckpoint();
      const strength = STRENGTH[this.gameOpponent];
      let move: Coord | null = null;
      // An easy cat still never misses its own five or a four it must block.
      const forced = strength.guard ? forcedMove(this.board, this.currentPlayer, RULES) : null;
      if (forced) {
        move = forced;
      } else if (activeCkpt) {
        const lastMove =
          this.history.length > 0
            ? {
                row: this.history[this.history.length - 1].row,
                col: this.history[this.history.length - 1].col,
              }
            : null;
        const nnRes = await chooseCheckpointMoveAsync(
          this.board,
          this.currentPlayer,
          activeCkpt.weights,
          lastMove,
          strength.sims,
          strength.budgetMs,
          undefined, // the cat's face shows it is thinking; no progress readout
          () => searchToken !== this.aiSearchToken || !this.aiThinking,
        );
        if (searchToken !== this.aiSearchToken || !this.aiThinking) return;
        if (nnRes) {
          this.lastNnInsight = nnRes;
          const choice =
            strength.topK > 1
              ? sampleCandidate(nnRes.topCandidates, strength.topK, strength.temperature)
              : null;
          move = choice ? { row: choice.row, col: choice.col } : { row: nnRes.row, col: nnRes.col };
        } else {
          this.lastNnInsight = null;
        }
      } else {
        this.lastNnInsight = null;
        move = chooseAIMove(
          this.board,
          this.currentPlayer,
          RULES,
          this.masterCharId,
        );
      }
      const elapsed = performance.now() - started;
      const remain = Math.max(80, targetPaceMs - elapsed);

      this.aiTimer = window.setTimeout(() => {
        if (searchToken !== this.aiSearchToken) return;
        this.aiTimer = null;
        this.aiThinking = false;
        if (move) {
          this.placeStone(move.row, move.col);
        } else {
          this.isDraw = true;
          this.recordFinishedMatch();
          this.recomputeAndRender();
        }
      }, remain);
    }, 45);
  }

  private undo(): void {
    if (this.aiThinking || this.history.length === 0) return;
    if (this.mode === 'ai') {
      if (this.undoLeft <= 0) return;
      this.undoLeft -= 1;
    }
    this.hintCells = [];
    this.hideHelper();
    this.stopAutoPlay();

    const stepsToUndo =
      this.mode === 'ai' &&
      this.history.length >= 2 &&
      this.history[this.history.length - 1].player !== this.humanPlayer
        ? 2
        : 1;

    for (let i = 0; i < stepsToUndo; i++) {
      const last = this.history.pop();
      if (!last) break;
      this.board[last.row][last.col] = EMPTY;
    }

    this.viewMoveIndex = this.history.length;
    this.showVcfPreview = false;
    this.winner = null;
    this.isDraw = false;
    this.matchRecorded = false;
    this.lastPlacedKey = null;
    this.lastNnInsight = null;
    this.pendingAim = null;

    if (this.history.length === 0) {
      this.currentPlayer = BLACK;
    } else {
      const prev = this.history[this.history.length - 1];
      this.currentPlayer = opponentOf(prev.player);
    }

    if (this.mode === 'ai') {
      this.catSeenMoves = this.history.length;
      const [mood, lines] = this.voice().undo;
      this.cat.setMood(mood);
      this.cat.say(pickLine(lines, this.history.length));
    }
    this.recomputeAndRender();
    this.maybeTriggerAI();
  }

  private resetGame(): void {
    this.leavePuzzle();
    this.stopAutoPlay();
    if (this.aiTimer) {
      window.clearTimeout(this.aiTimer);
      this.aiTimer = null;
    }
    this.aiThinking = false;
    this.board = createEmptyBoard();
    this.history = [];
    this.viewMoveIndex = 0;
    this.showVcfPreview = false;
    this.activeVcf = null;
    this.currentPlayer = BLACK;
    this.winner = null;
    this.isDraw = false;
    this.matchRecorded = false;
    this.lastPlacedKey = null;
    this.lastNnInsight = null;
    this.pendingAim = null;
    this.catSpoken.clear();
    this.catSeenMoves = 0;
    this.gameOpponent = this.opponent;
    this.undoLeft = UNDO_LIMIT[this.gameOpponent];
    this.hintsLeft = HINTS_PER_GAME;
    this.hintCells = [];
    this.hideHelper();
    this.cat.setCharacter(this.gameOpponent, CAT_NAME[this.gameOpponent]);
    this.cat.setMood('happy');
    this.cat.say(pickLine(this.voice().opening, Date.now()));
    this.recomputeAndRender();
    this.maybeTriggerAI();
  }

  // ── Render Pipeline ──
  private recomputeAndRender(): void {
    const view = this.getViewState();

    this.forbiddenMap.clear();
    if (!view.winner && !view.isDraw && view.turnPlayer === BLACK) {
      for (const cell of getForbiddenMoves(view.board, RULES)) {
        this.forbiddenMap.set(coordKey(cell.row, cell.col), cell);
      }
    }

    this.threats = view.winner ? [] : scanBoardThreats(view.board, RULES);

    this.activeVcf = null;
    this.aiPendingVcf = null;
    const isNeuralPlay =
      this.activePage === 'play' &&
      this.mode === 'ai' &&
      Boolean(this.getActiveCheckpoint());
    if (!isNeuralPlay && !view.winner && !view.isDraw && view.moves.length >= 4) {
      const vcf = findVcfSequence(view.board, view.turnPlayer, RULES, 7, 25);
      if (vcf && vcf.length >= 1) {
        this.activeVcf = vcf;
      } else if (
        this.activePage === 'play' &&
        this.mode === 'ai' &&
        view.turnPlayer === this.humanPlayer
      ) {
        const lurking = findVcfSequence(
          view.board,
          opponentOf(this.humanPlayer),
          RULES,
          5,
          18,
        );
        if (lurking && lurking.length >= 1) {
          this.aiPendingVcf = lurking;
        }
      }
    }
    if (!this.activeVcf) {
      this.showVcfPreview = false;
    }

    this.renderSeats(view);
    this.renderStatus(view);
    if (this.activePage === 'history' && this.historyMode === 'viewer') {
      this.renderHistory();
      this.renderReviewPanel();
    }
    this.renderBoard(view);
    this.resetHintMessage();
    this.syncPlayActions();
  }

  private renderHintBar(finished: boolean): void {
    this.hintBarEl.innerHTML = '';
    if (this.puzzle) {
      this.renderPuzzleBar();
      return;
    }
    const head = document.createElement('div');
    head.className = 'hint-bar-head';
    head.textContent = '도움';
    const row = document.createElement('div');
    row.className = 'hint-bar-row';
    const myTurn = !finished && !this.aiThinking && this.currentPlayer === this.humanPlayer;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'hint-btn';
    btn.textContent = `한 수 추천 (남은 ${this.hintsLeft}번)`;
    btn.classList.toggle('used', this.hintsLeft <= 0);
    btn.disabled = this.hintsLeft <= 0 || !myTurn || this.hintBusy;
    btn.addEventListener('click', () => void this.useHint());
    row.appendChild(btn);
    // Takebacks live here too, next to the other kinds of help.
    this.undoBtn.classList.add('hint-undo');
    this.hintBarEl.append(head, row, this.undoBtn);
  }

  private showHelper(text: string, mood: CatMood): void {
    if (this.helperTimer !== null) window.clearTimeout(this.helperTimer);
    this.helperTimer = null;
    const id = otherCat(this.gameOpponent);
    this.helper.setCharacter(id, CAT_NAME[id]);
    this.helper.setMood(mood);
    this.helper.say(text);
    this.helper.el.classList.add('show');
  }

  private hideHelper(): void {
    if (this.helperTimer !== null) window.clearTimeout(this.helperTimer);
    this.helperTimer = null;
    this.helper.el.classList.remove('show');
    this.helper.say(null);
  }

  /** The other cat drops by with its best move. A hint with nothing to show is not used up. */
  private async useHint(): Promise<void> {
    if (this.hintsLeft <= 0 || this.hintBusy || this.aiThinking) return;
    if (this.currentPlayer !== this.humanPlayer || this.winner || this.isDraw) return;
    const help = this.voice(otherCat(this.gameOpponent)).help;
    const me = this.humanPlayer;
    const legal = (c: Coord) => this.board[c.row][c.col] === EMPTY && !this.forbiddenMap.has(coordKey(c.row, c.col));
    let cells: Coord[] = [];
    let text = '';

    this.hintBusy = true;
    this.showHelper(pickLine(help.thinking, Date.now()), 'think');
    this.renderSeats();
    const ckpt = this.getActiveCheckpoint();
    const last = this.history[this.history.length - 1];
    const best = ckpt
      ? await chooseCheckpointMoveAsync(this.board, me, ckpt.weights, last ? { row: last.row, col: last.col } : null, 160, 950)
      : chooseAIMove(this.board, me, RULES, 'lv4');
    this.hintBusy = false;
    if (best && legal(best)) cells = [{ row: best.row, col: best.col }];
    text = pickLine(cells.length ? help.move(formatCoord(cells[0].row, cells[0].col)) : help.none, Date.now());

    if (cells.length) {
      this.hintsLeft -= 1;
      this.hintCells = cells; // an empty answer keeps any earlier hint on the board
    }
    this.showHelper(text, cells.length ? 'happy' : 'idle');
    // Say it and go: the helper shouldn't sit on the board corner. A shown hint stays marked on the board.
    this.helperTimer = window.setTimeout(() => this.hideHelper(), cells.length ? 5000 : 3000);
    if (cells.length) {
      // The cat you're playing overheard, and doesn't like it.
      const [mood, lines] = this.voice().hinted;
      this.cat.setMood(mood);
      this.cat.say(pickLine(lines, Date.now()));
    }
    this.renderBoard();
    this.renderSeats();
  }

  /** React once per new position: say the most important unsaid line, or just change face. */
  private updateCat(view = this.getViewState()): void {
    if (this.aiThinking) {
      this.cat.setMood('think');
      return;
    }
    if (this.puzzle) return;
    const len = view.moves.length;
    if (len === this.catSeenMoves && !view.winner && !view.isDraw) return;
    this.catSeenMoves = len;
    const aiPlayer = opponentOf(this.humanPlayer);
    const ctx: CatContext = {
      aiPlayer,
      turnPlayer: view.turnPlayer,
      moveCount: len,
      evalScore: view.moves[len - 1]?.evalScore ?? 0,
      prevEvalScore: len > 2 ? view.moves[len - 3]?.evalScore : undefined,
      lastMove: view.moves[len - 1] ?? null,
      winner: view.winner,
      isDraw: view.isDraw,
      threats: this.threats,
      activeVcf: this.activeVcf,
      aiPendingVcf: this.aiPendingVcf,
      nnPredictedPliesLeft: this.lastNnInsight?.predictedPliesLeft ?? null,
    };
    const line = catLines(ctx, this.voice()).find((l) => !this.catSpoken.has(l.key));
    if (line) {
      this.catSpoken.add(line.key);
      this.cat.setMood(line.mood);
      this.cat.say(line.text, line.tone);
    } else if (len > 0) {
      this.cat.setMood(catQuietMood(ctx));
      this.cat.say(null);
    }
  }

  private renderSeats(view = this.getViewState()): void {
    const showPlaySeats = this.activePage === 'play';
    this.seatTopRowEl.hidden = !showPlaySeats;
    this.seatBottomRowEl.hidden = !showPlaySeats;
    if (!showPlaySeats) return;

    const finished = Boolean(view.winner || view.isDraw) || (this.puzzle !== null && this.puzzleState === 'failed');
    const myPlayer: Player = this.mode === 'ai' ? this.humanPlayer : BLACK;
    const oppPlayer: Player = opponentOf(myPlayer);
    const isMyTurn = !finished && view.turnPlayer === myPlayer;
    const isOppTurn = !finished && view.turnPlayer === oppPlayer;

    // 1. Top-Left Opponent Seat (+ AI Speech Bubble)
    this.opponentSeatEl.innerHTML = '';
    this.opponentSeatEl.classList.toggle('is-turn', isOppTurn);

    const oppDot = document.createElement('span');
    oppDot.className = `turn-dot ${oppPlayer === BLACK ? 'black' : 'white'}`;

    if (this.mode === 'ai') {
      // The cat replaces the seat card, the memo card and the separate speech box.
      if (this.cat.el.parentElement !== this.seatTopRowEl) this.seatTopRowEl.prepend(this.cat.el);
      this.cat.setCharacter(this.gameOpponent, CAT_NAME[this.gameOpponent]);
      this.opponentSpeechEl.hidden = true;
      this.leftMemoCardEl.innerHTML = '';
      this.updateCat(view);
      this.renderHintBar(finished);
    } else {
      const avatar = document.createElement('span');
      avatar.className = 'char-avatar char-avatar-md seat-pvp-icon';
      avatar.innerHTML =
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="8" r="4"/><path d="M5 20c0-3.5 3.1-6 7-6s7 2.5 7 6"/></svg>';
      const name = document.createElement('span');
      name.className = 'seat-name';
      name.textContent = '상대 (백)';
      this.opponentSeatEl.append(avatar, name, oppDot);
      this.opponentSpeechEl.hidden = true;
      this.leftMemoCardEl.innerHTML = `
        <div class="eval-card-head"><span>2인 로컬 대국</span><span class="eval-card-pct">렌주 룰</span></div>
        <p class="side-memo-desc">한 화면에서 번갈아 착수합니다. 흑은 3·3, 4·4, 장목이 금지됩니다.</p>
      `;
    }

    // Live moves list on Right Rail
    this.liveMovesCountEl.textContent = `${view.moves.length}수`;
    this.liveMovesListEl.innerHTML = '';
    if (view.moves.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'history-empty';
      empty.textContent = '바둑판의 교차점을 눌러 첫 수를 두세요.';
      this.liveMovesListEl.appendChild(empty);
    } else {
      for (const m of view.moves) {
        const item = document.createElement('button');
        item.type = 'button';
        item.className = `history-item${m.moveNumber === view.moves.length ? ' latest' : ''}`;
        const left = document.createElement('div');
        left.className = 'history-item-left';
        left.innerHTML = `
          <span class="history-num">${m.moveNumber}</span>
          <span class="history-stone ${m.player === BLACK ? 'black' : 'white'}"></span>
          <span class="history-coord">${formatCoord(m.row, m.col)}</span>
        `;
        item.appendChild(left);
        if (m.tag) {
          const tag = document.createElement('span');
          tag.className = `history-tag ${m.tag === '승리' ? 'win' : m.tag === '열린3' ? 'three' : 'four'}`;
          tag.textContent = m.tag;
          item.appendChild(tag);
        }
        const focusKey = coordKey(m.row, m.col);
        item.addEventListener('mouseenter', () => {
          this.historyFocusKey = focusKey;
          this.updateHistoryFocus();
        });
        item.addEventListener('mouseleave', () => {
          if (this.historyFocusKey === focusKey) {
            this.historyFocusKey = null;
            this.updateHistoryFocus();
          }
        });
        this.liveMovesListEl.appendChild(item);
      }
      this.liveMovesListEl.scrollTop = this.liveMovesListEl.scrollHeight;
    }

    // 2. Bottom-Right Self Seat
    this.selfSeatEl.innerHTML = '';
    this.selfSeatEl.classList.toggle('is-turn', isMyTurn);

    const myDot = document.createElement('span');
    myDot.className = `turn-dot ${myPlayer === BLACK ? 'black' : 'white'}`;

    const user = getCurrentUser();
    const myName = document.createElement('span');
    myName.className = 'seat-name';
    myName.textContent =
      this.mode === 'ai' ? (user ? user.nickname : '나') : '나 (흑)';

    const mySub = document.createElement('span');
    mySub.className = 'seat-sub';
    if (this.mode === 'ai' && user) {
      const rec = user.records[this.masterCharId];
      mySub.textContent = formatRecord(rec.wins, rec.draws, rec.losses);
    } else {
      mySub.textContent = myPlayer === BLACK ? '흑 · 선공' : '백 · 후공';
    }

    const turnTag = document.createElement('span');
    turnTag.className = 'seat-turn-tag';
    if (view.winner) {
      turnTag.textContent = view.winner.winner === myPlayer ? '승리!' : '패배';
    } else if (view.isDraw) {
      turnTag.textContent = '무승부';
    } else if (isMyTurn) {
      turnTag.textContent = '내 차례';
    } else {
      turnTag.textContent = '대기 중';
    }

    this.selfSeatEl.append(myDot, myName, mySub, turnTag);

    // 3. Post-Game Decision Bar (Rematch / Next Character / New Opponent / Review)
    this.seatBottomRowEl.classList.toggle('is-finished', finished);
    this.workspaceEl.classList.toggle('is-over', finished);
    this.endSheetEl.classList.toggle('open', finished);
    if (finished && this.puzzle) {
      const solved = this.puzzleState === 'solved';
      this.endSheetTitleEl.textContent = solved ? '풀었다!' : '아깝다…';
      this.endSheetEl.dataset.result = solved ? 'win' : 'lose';
      this.endSheetSubEl.textContent = solved
        ? `${view.moves.length > 0 ? Math.ceil(view.moves.length / 2) : 0}수 만에 풀었어요.`
        : '이길 길이 끊겼어요. 처음부터 다시 해 볼까요?';
    } else if (finished) {
      const won = view.winner?.winner === myPlayer;
      this.endSheetTitleEl.textContent = view.winner ? (won ? '이겼다!' : '졌다…') : '무승부';
      this.endSheetEl.dataset.result = view.winner ? (won ? 'win' : 'lose') : 'draw';
      this.endSheetSubEl.textContent = `${view.moves.length}수 만에 끝났어요. 더는 돌을 둘 수 없어요.`;
    }
    this.postGameActionsEl.innerHTML = '';

    if (finished) {
      const makeBtn = (
        label: string,
        variant: 'primary' | 'default' | 'ghost',
        onClick: () => void,
      ) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = variant === 'default' ? 'btn' : `btn ${variant}`;
        b.textContent = label;
        b.addEventListener('click', onClick);
        return b;
      };

      if (this.practice) {
        const setup = this.practice;
        this.postGameActionsEl.appendChild(makeBtn('그 자리에서 다시', 'primary', () => this.startPractice(setup)));
        this.postGameActionsEl.appendChild(makeBtn('새 판', 'default', () => this.resetGame()));
        this.postGameActionsEl.appendChild(makeBtn('복기하기', 'ghost', () => this.openLiveMatchInViewer()));
        return;
      }

      if (this.puzzle) {
        const p = this.puzzle;
        const next = this.nextPuzzle(p.id);
        if (this.puzzleState === 'solved' && next) {
          this.postGameActionsEl.appendChild(makeBtn('다음 문제', 'primary', () => this.startPuzzle(next)));
        }
        this.postGameActionsEl.appendChild(
          makeBtn('다시 풀기', this.puzzleState === 'solved' ? 'default' : 'primary', () => this.startPuzzle(p)),
        );
        this.postGameActionsEl.appendChild(makeBtn('목록으로', 'ghost', () => this.switchPage('puzzles')));
        return;
      }

      this.postGameActionsEl.appendChild(
        makeBtn('한 판 더', 'primary', () => {
          this.resetGame();
        }),
      );

      this.postGameActionsEl.appendChild(
        makeBtn(
          this.mode === 'ai'
            ? this.humanPlayer === BLACK
              ? '백(후공)으로 변경'
              : '흑(선공)으로 변경'
            : '오목 마스터 대국',
          'default',
          () => {
            if (this.mode === 'ai') {
              this.humanPlayer = this.humanPlayer === BLACK ? WHITE : BLACK;
            } else {
              this.mode = 'ai';
            }
            this.syncModeControls();
            this.resetGame();
          },
        ),
      );

      if (this.history.length > 0) {
        this.postGameActionsEl.appendChild(
          makeBtn('복기하기', 'ghost', () => this.openLiveMatchInViewer()),
        );
      }
    }
  }

  private renderHistory(): void {
    const prevScrollTop = this.historyListEl.scrollTop;
    this.historyListEl.innerHTML = '';

    const moves = this.getActiveReplayMoves();
    const total = moves.length;
    this.replayTitleTextEl.textContent = this.replayTitle || `수 기록 (${total}수)`;
    const view = this.getViewState();
    const bActive = !view.winner && !view.isDraw && view.turnPlayer === BLACK;
    const wActive = !view.winner && !view.isDraw && view.turnPlayer === WHITE;
    const bWin = view.winner?.winner === BLACK;
    const wWin = view.winner?.winner === WHITE;
    this.replayPlayersBarEl.innerHTML = `
      <span class="replay-player-chip black${bActive ? ' is-turn' : ''}${bWin ? ' is-winner' : ''}">
        <span class="history-stone black"></span>
        <span>흑: ${this.replayBlackName}</span>
      </span>
      <span class="replay-vs">vs</span>
      <span class="replay-player-chip white${wActive ? ' is-turn' : ''}${wWin ? ' is-winner' : ''}">
        <span class="history-stone white"></span>
        <span>백: ${this.replayWhiteName}</span>
      </span>
    `;
    if (this.replayBadge) {
      this.replayTitleBadgeEl.hidden = false;
      this.replayTitleBadgeEl.className = `archive-badge ${this.replayBadge.tone}`;
      this.replayTitleBadgeEl.textContent = this.replayBadge.label;
    } else {
      this.replayTitleBadgeEl.hidden = true;
    }

    this.historyNumBtn.classList.toggle('active', this.showMoveNumbers);
    this.copyHistoryBtn.disabled = total === 0;
    this.replaySliderEl.max = String(total);
    this.replaySliderEl.value = String(this.viewMoveIndex);
    this.replaySliderEl.disabled = total === 0;
    this.replayStepLabelEl.textContent = `${this.viewMoveIndex} / ${total}`;

    this.replayFirstBtn.disabled = total === 0 || this.viewMoveIndex === 0;
    this.replayPrevBtn.disabled = total === 0 || this.viewMoveIndex === 0;
    this.replayPlayBtn.disabled = total === 0;
    this.replayPlayBtn.textContent = this.autoPlayTimer !== null ? '⏸ 정지' : '▶ 재생';
    this.replayPlayBtn.classList.toggle('active', this.autoPlayTimer !== null);
    this.replayNextBtn.disabled = total === 0 || this.viewMoveIndex >= total;
    this.replayLastBtn.disabled = total === 0 || this.viewMoveIndex >= total;

    this.renderEvalChart();

    if (total === 0) {
      const empty = document.createElement('div');
      empty.className = 'history-empty';
      empty.textContent = '아직 착수 기록이 없습니다.';
      this.historyListEl.appendChild(empty);
      return;
    }

    let activeRowEl: HTMLElement | null = null;
    for (let i = 0; i < total; i++) {
      const m = moves[i];
      const item = document.createElement('button');
      item.type = 'button';
      item.className = 'history-item';
      if (m.moveNumber === this.viewMoveIndex) {
        item.classList.add('latest');
        activeRowEl = item;
      } else if (m.moveNumber > this.viewMoveIndex) {
        item.classList.add('future');
      }

      const left = document.createElement('div');
      left.className = 'history-item-left';

      const num = document.createElement('span');
      num.className = 'history-num';
      num.textContent = `${m.moveNumber}`;

      const dot = document.createElement('span');
      dot.className = `history-stone ${m.player === BLACK ? 'black' : 'white'}`;

      const coord = document.createElement('span');
      coord.className = 'history-coord';
      coord.textContent = formatCoord(m.row, m.col);

      const mover = document.createElement('span');
      mover.className = 'history-mover';
      mover.textContent = m.player === BLACK ? this.replayBlackName : this.replayWhiteName;

      left.append(num, dot, coord, mover);
      item.appendChild(left);

      if (m.tag) {
        const tag = document.createElement('span');
        tag.className = 'history-tag';
        if (m.tag === '승리') tag.classList.add('win');
        else if (m.tag === '열린3') tag.classList.add('three');
        else tag.classList.add('four');
        tag.textContent = m.tag;
        item.appendChild(tag);
      }

      const focusKey = coordKey(m.row, m.col);
      item.addEventListener('click', () => this.jumpToMove(m.moveNumber));
      item.addEventListener('mouseenter', () => {
        this.historyFocusKey = focusKey;
        this.updateHistoryFocus();
      });
      item.addEventListener('mouseleave', () => {
        if (this.historyFocusKey === focusKey) {
          this.historyFocusKey = null;
          this.updateHistoryFocus();
        }
      });

      this.historyListEl.appendChild(item);
    }

    this.historyListEl.scrollTop = prevScrollTop;
    if (
      activeRowEl &&
      this.historyListEl.clientHeight > 0 &&
      this.historyListEl.scrollHeight > this.historyListEl.clientHeight + 2
    ) {
      const listEl = this.historyListEl;
      const listRect = listEl.getBoundingClientRect();
      const rowRect = activeRowEl.getBoundingClientRect();
      const relTop = rowRect.top - listRect.top + listEl.scrollTop;
      const relBottom = rowRect.bottom - listRect.top + listEl.scrollTop;
      if (relTop < listEl.scrollTop) {
        listEl.scrollTop = relTop;
      } else if (relBottom > listEl.scrollTop + listEl.clientHeight) {
        listEl.scrollTop = relBottom - listEl.clientHeight;
      }
    }
  }

  private renderEvalChart(): void {
    const oldSvg = this.evalChartBoxEl.querySelector('.eval-chart-svg');
    if (oldSvg) oldSvg.remove();

    const moves = this.getActiveReplayMoves();
    const curScore =
      this.viewMoveIndex > 0
        ? (moves[this.viewMoveIndex - 1]?.evalScore ?? 0)
        : 0;
    const view = this.getViewState();

    let blackPct = Math.max(1, Math.min(99, Math.round(50 + curScore * 50)));
    if (view.winner) {
      blackPct = view.winner.winner === BLACK ? 100 : 0;
    } else if (view.isDraw) {
      blackPct = 50;
    }
    const whitePct = 100 - blackPct;

    let judge = '호각';
    let judgeTone = 'even';
    if (blackPct === 100) {
      judge = '흑 승리';
      judgeTone = 'black';
    } else if (whitePct === 100) {
      judge = '백 승리';
      judgeTone = 'white';
    } else if (blackPct >= 83) {
      judge = '흑 승세';
      judgeTone = 'black';
    } else if (blackPct >= 68) {
      judge = '흑 우세';
      judgeTone = 'black';
    } else if (blackPct >= 56) {
      judge = '흑 약간 우세';
      judgeTone = 'black';
    } else if (whitePct >= 83) {
      judge = '백 승세';
      judgeTone = 'white';
    } else if (whitePct >= 68) {
      judge = '백 우세';
      judgeTone = 'white';
    } else if (whitePct >= 56) {
      judge = '백 약간 우세';
      judgeTone = 'white';
    }

    this.evalChartLabelEl.className = `eval-judge-pill ${judgeTone}`;
    this.evalChartLabelEl.textContent = judge;

    this.evalTugBarEl.innerHTML = `
      <div class="eval-tug-labels">
        <span class="eval-tug-b">⚫ 흑 ${blackPct}%</span>
        <span class="eval-tug-w">백 ${whitePct}% ⚪</span>
      </div>
      <div class="eval-tug-track">
        <span class="eval-tug-black" style="width:${blackPct}%"></span>
      </div>
    `;

    this.evalChartBoxEl.appendChild(
      buildEvalChartSvg(
        moves,
        this.viewMoveIndex,
        280,
        54,
        'eval-chart-svg',
        true,
        (idx) => this.jumpToMove(idx),
      ),
    );
  }

  private updateHistoryFocus(): void {
    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        const stone = this.cellButtons[r][c].querySelector('.stone');
        if (!stone) continue;
        stone.classList.toggle(
          'history-focus',
          this.historyFocusKey === coordKey(r, c),
        );
      }
    }
  }

  private renderEvalBar(view: {
    board: Board;
    moves: MoveRecord[];
    winner: WinResult | null;
    isDraw: boolean;
    turnPlayer: Player;
  }): void {
    this.evalSparkBoxEl.innerHTML = '';
    this.evalSparkBoxEl.appendChild(
      buildEvalChartSvg(
        view.moves,
        view.moves.length,
        240,
        74,
        'eval-spark-svg',
        false,
      ),
    );

    if (view.winner) {
      const isBlackWin = view.winner.winner === BLACK;
      this.evalLabelEl.textContent = isBlackWin ? '흑 승' : '백 승';
      return;
    }

    if (view.isDraw) {
      this.evalLabelEl.textContent = '무승부';
      return;
    }

    if (this.activeVcf) {
      this.evalLabelEl.textContent = `${playerLabel(view.turnPlayer)} 필승`;
      return;
    }

    const adv =
      view.moves.length > 0
        ? (view.moves[view.moves.length - 1].evalScore ?? 0)
        : 0;
    const blackPct = Math.max(1, Math.min(99, Math.round((adv + 1) * 50)));

    if (Math.abs(adv) < 0.04) {
      this.evalLabelEl.textContent = '호각 50%';
    } else if (adv > 0) {
      this.evalLabelEl.textContent = `흑 ${blackPct}%`;
    } else {
      this.evalLabelEl.textContent = `백 ${100 - blackPct}%`;
    }
  }

  private renderStatus(view = this.getViewState()): void {
    const activePlayer = view.winner ? view.winner.winner : view.turnPlayer;
    this.turnDotEl.className = `turn-dot ${activePlayer === BLACK ? 'black' : 'white'}`;

    if (view.winner) {
      if (this.activePage === 'history') {
        const wName =
          view.winner.winner === BLACK ? this.replayBlackName : this.replayWhiteName;
        this.turnTextEl.textContent = `${playerLabel(view.winner.winner)} (${wName}) 승리!`;
      } else {
        this.turnTextEl.textContent = `${playerLabel(view.winner.winner)} 승리!`;
      }
    } else if (view.isDraw) {
      this.turnTextEl.textContent = '무승부';
    } else {
      if (this.activePage === 'history') {
        const tName =
          view.turnPlayer === BLACK ? this.replayBlackName : this.replayWhiteName;
        this.turnTextEl.textContent = `${playerLabel(view.turnPlayer)} (${tName}) 차례`;
      } else {
        this.turnTextEl.textContent = `${playerLabel(view.turnPlayer)} 차례`;
      }
    }

    const totalForPage =
      this.activePage === 'history'
        ? this.getActiveReplayMoves().length
        : this.history.length;
    this.moveCountEl.textContent =
      this.activePage === 'history'
        ? `${this.viewMoveIndex}/${totalForPage}수`
        : `${totalForPage}수`;

    this.renderEvalBar(view);
    this.statusRightEl.innerHTML = '';

    if (this.activeVcf && !view.winner) {
      const vcfBtn = document.createElement('button');
      vcfBtn.type = 'button';
      vcfBtn.className = `vcf-btn${this.showVcfPreview ? ' active' : ''}`;
      vcfBtn.textContent = this.showVcfPreview
        ? '연속 사 숨기기'
        : `연속 사 ${this.activeVcf.length}수`;
      vcfBtn.addEventListener('click', () => {
        this.showVcfPreview = !this.showVcfPreview;
        this.renderStatus(view);
        this.renderBoard(view);
      });
      this.statusRightEl.appendChild(vcfBtn);
    }
  }

  private renderBoard(view = this.getViewState()): void {
    const moveNumberByKey = new Map<string, number>();
    for (const m of view.moves) {
      moveNumberByKey.set(coordKey(m.row, m.col), m.moveNumber);
    }

    const lastMove =
      view.moves.length > 0 ? view.moves[view.moves.length - 1] : null;
    const lastKey = lastMove ? coordKey(lastMove.row, lastMove.col) : null;

    const winSet = new Set<string>();
    if (view.winner) {
      for (const c of view.winner.line) winSet.add(coordKey(c.row, c.col));
    }

    const oppPlayer = opponentOf(view.turnPlayer);
    const blockTargetSet = new Set<string>();
    const fourTargetSet = new Set<string>();
    const threeTargetSet = new Set<string>();
    // In a puzzle the threat markers would give the answer away.
    if (this.showRuleOverlay && !view.winner && !this.puzzle) {
      for (const t of this.threats) {
        for (const target of t.targetCells) {
          const k = coordKey(target.row, target.col);
          if (t.type === 'OPEN_FOUR' || t.type === 'CLOSED_FOUR') {
            if (t.player === oppPlayer) blockTargetSet.add(k);
            else fourTargetSet.add(k);
          } else {
            threeTargetSet.add(k);
          }
        }
      }
    }

    // In a replay, the turning point being shown brings its better move and winning line.
    const focus =
      this.activePage === 'history' && this.reviewFocus?.index === this.viewMoveIndex ? this.reviewFocus : null;
    const markCells = focus ? focus.cells : this.activePage === 'play' ? this.hintCells : [];
    const playedThere = focus ? this.replayMoves[focus.index] : null;
    const seq = focus?.seq ?? (this.showVcfPreview ? this.activeVcf : null);
    const vcfMap = new Map<string, { step: number; player: Player }>();
    if (seq) {
      seq.forEach((c, idx) => {
        const k = coordKey(c.row, c.col);
        if (!vcfMap.has(k)) {
          const p = idx % 2 === 0 ? view.turnPlayer : oppPlayer;
          vcfMap.set(k, { step: idx + 1, player: p });
        }
      });
    }

    const replaying = this.isReplaying();
    const readOnlyBoard = this.activePage !== 'play' || replaying;

    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        const btn = this.cellButtons[r][c];
        const stone = view.board[r][c];
        const key = coordKey(r, c);
        const forbidden = this.forbiddenMap.get(key);

        btn.className = 'cell';
        if (view.winner || view.isDraw || readOnlyBoard) {
          btn.classList.add('game-over');
        }
        btn.innerHTML = '';

        if (stone !== EMPTY) {
          btn.classList.add('occupied');
          const stoneEl = document.createElement('span');
          stoneEl.className = `stone ${stone === BLACK ? 'black' : 'white'}`;

          if (!replaying && key === this.lastPlacedKey) {
            stoneEl.classList.add('new-stone');
            const ripple = document.createElement('span');
            ripple.className = 'place-ripple';
            btn.appendChild(ripple);
          }

          if (view.winner) {
            if (winSet.has(key)) stoneEl.classList.add('win-stone');
            else stoneEl.classList.add('dimmed');
          }

          if (this.showMoveNumbers) {
            const num = document.createElement('span');
            num.className = 'move-num';
            num.textContent = String(moveNumberByKey.get(key) ?? '');
            stoneEl.appendChild(num);
          } else if (key === lastKey) {
            const dot = document.createElement('span');
            dot.className = 'last-dot';
            stoneEl.appendChild(dot);
          }

          btn.appendChild(stoneEl);
          continue;
        }

        if (markCells.some((h) => h.row === r && h.col === c)) {
          const hint = document.createElement('span');
          hint.className = 'hint-mark';
          btn.appendChild(hint);
        }
        if (playedThere && playedThere.row === r && playedThere.col === c) {
          const was = document.createElement('span');
          was.className = 'review-played';
          was.title = '실전에서 둔 자리';
          btn.appendChild(was);
        }

        const vcfItem = vcfMap.get(key);
        if (vcfItem) {
          const vcfEl = document.createElement('span');
          vcfEl.className = `vcf-ghost ${vcfItem.player === BLACK ? 'black' : 'white'}`;
          vcfEl.textContent = String(vcfItem.step);
          btn.appendChild(vcfEl);
          continue;
        }

        if (this.showRuleOverlay && !view.winner && !view.isDraw) {
          if (blockTargetSet.has(key)) {
            const marker = document.createElement('span');
            marker.className = 'target-marker block';
            btn.appendChild(marker);
          } else if (fourTargetSet.has(key)) {
            const marker = document.createElement('span');
            marker.className = 'target-marker four';
            btn.appendChild(marker);
          } else if (threeTargetSet.has(key)) {
            const marker = document.createElement('span');
            marker.className = 'target-marker three';
            btn.appendChild(marker);
          }
        }

        if (forbidden) {
          btn.classList.add('forbidden');
          if (this.showRuleOverlay) {
            const mark = document.createElement('span');
            mark.className = 'forbidden-mark';
            const { short } = forbiddenLabel(forbidden.reason);
            mark.innerHTML = `
              <svg class="forbidden-x" viewBox="0 0 24 24" fill="none">
                <path d="M6 6L18 18M18 6L6 18" />
              </svg>
              <span class="forbidden-tag">${short}</span>
            `;
            btn.appendChild(mark);
          }
        } else if (!view.winner && !view.isDraw && !readOnlyBoard) {
          const ghost = document.createElement('span');
          ghost.className = `ghost-stone ${view.turnPlayer === BLACK ? 'black' : 'white'}`;
          btn.appendChild(ghost);
        }
      }
    }

    this.lastPlacedKey = null;
    this.renderSvgOverlays(view);
    this.updateAimVisuals(true);
  }

  private renderSvgOverlays(view = this.getViewState()): void {
    this.overlaySvg.innerHTML = '';

    const activeAim = this.pendingAim && this.canHumanInteract() ? this.pendingAim : null;
    if (activeAim) {
      const cx = activeAim.col * 100 + 50;
      const cy = activeAim.row * 100 + 50;
      const forbidden = this.forbiddenMap.has(coordKey(activeAim.row, activeAim.col));
      const stroke = forbidden ? 'rgba(217, 56, 58, 0.55)' : 'rgba(15, 139, 141, 0.55)';
      const hLine = createSvgElement('line', {
        x1: 20,
        y1: cy,
        x2: 1480,
        y2: cy,
        stroke,
        'stroke-width': 4,
        'stroke-dasharray': '14 10',
      });
      const vLine = createSvgElement('line', {
        x1: cx,
        y1: 20,
        x2: cx,
        y2: 1480,
        stroke,
        'stroke-width': 4,
        'stroke-dasharray': '14 10',
      });
      const ring = createSvgElement('circle', {
        cx,
        cy,
        r: 44,
        fill: 'none',
        stroke,
        'stroke-width': 5,
      });
      this.overlaySvg.append(hLine, vLine, ring);
    }

    if (view.winner && view.winner.line.length >= 2) {
      const first = view.winner.line[0];
      const last = view.winner.line[view.winner.line.length - 1];
      const x1 = first.col * 100 + 50;
      const y1 = first.row * 100 + 50;
      const x2 = last.col * 100 + 50;
      const y2 = last.row * 100 + 50;
      const length = Math.hypot(x2 - x1, y2 - y1);

      this.overlaySvg.appendChild(
        createSvgElement('line', {
          x1,
          y1,
          x2,
          y2,
          class: 'win-line-glow',
        }),
      );

      const core = createSvgElement('line', {
        x1,
        y1,
        x2,
        y2,
        class: 'win-line-core',
        'stroke-dasharray': length,
        'stroke-dashoffset': length,
      });
      core.style.setProperty('--line-len', String(length));
      this.overlaySvg.appendChild(core);
      return;
    }

    if (!this.showRuleOverlay || this.puzzle) return;

    for (const threat of this.threats) {
      if (threat.stones.length < 2) continue;
      const sorted = [...threat.stones].sort((a, b) =>
        a.row !== b.row ? a.row - b.row : a.col - b.col,
      );
      const first = sorted[0];
      const last = sorted[sorted.length - 1];
      const cls =
        threat.type === 'OPEN_THREE' ? 'threat-line-three' : 'threat-line-four';

      this.overlaySvg.appendChild(
        createSvgElement('line', {
          x1: first.col * 100 + 50,
          y1: first.row * 100 + 50,
          x2: last.col * 100 + 50,
          y2: last.row * 100 + 50,
          class: cls,
        }),
      );
    }
  }
}

const mount = document.getElementById('app');
if (mount) {
  new GomokuApp(mount);
}

// Offline/installable app; dev keeps the network so HMR is never served stale.
if (import.meta.env.PROD && 'serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./sw.js').catch(() => {});
  });
}
