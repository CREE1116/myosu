import './style.css';
import './monitor.css';
import { evaluatePositionAdvantage } from './engine/ai';
import {
  checkWinner,
  createEmptyBoard,
  formatCoord,
  getForbiddenMoves,
  hasAnyLegalMove,
  opponentOf,
  scanBoardThreats,
} from './engine/renju';
import { findVcfSequence } from './engine/vcf';
import {
  BLACK,
  BOARD_SIZE,
  Board,
  BoardThreat,
  Coord,
  EMPTY,
  ForbiddenCell,
  MoveRecord,
  Player,
  RuleSet,
  WHITE,
  WinResult,
} from './engine/types';
import {
  RULES,
  STAR_POINTS,
  buildEvalChartSvg,
  buildStaticGridSvg,
  computeTacticalCandidates,
  coordKey,
  createSvgElement,
  forbiddenLabel,
  formatPlayerThreats,
  getMoveTagOnBoard,
  playerLabel,
} from './ui/replayUtils';

interface PolicyCandidate {
  row: number;
  col: number;
  prob: number;
}

interface RlReplayMove {
  row: number;
  col: number;
  player: Player;
  moverType?: 'opening' | 'nn' | 'rule';
  value?: number;
  topPolicies?: PolicyCandidate[];
}

interface GenMetric {
  generation: number;
  pLoss: number;
  vLoss: number;
  avgMoves: number;
  blackWins: number;
  whiteWins: number;
  draws: number;
  spSec: number;
  trainSec: number;
  bufferSize: number;
}

interface ModelInfo {
  name?: string;
  channels: number;
  blocks: number;
  params: number;
}

interface RlReplayGame {
  id: string;
  label: string;
  generation: number;
  winner: number;
  blackPlayer?: string;
  whitePlayer?: string;
  matchup?: string;
  moves: RlReplayMove[];
}

interface RlReplayArchive {
  rules?: RuleSet;
  modelInfo?: ModelInfo;
  metrics?: GenMetric[];
  games: RlReplayGame[];
}

interface LiveTrainingMove {
  row: number;
  col: number;
  player: Player;
}

interface LiveTrainingPoint {
  gen: number;
  totalGames: number;
  elo: number;
  crit: number;
  vsRbWr: number;
  pLoss: number;
  spSec: number;
}

interface LiveCheckpointEntry {
  id: string;
  gen: number;
  file: string;
  totalGames: number;
  elo: number;
  crit: number;
  vsRbRecord: string;
  vsRbWr: number;
  replayId?: string;
}

interface LiveTrainingSnapshot {
  status: 'running' | 'idle';
  updatedAt: string;
  modelName: string;
  params: number;
  device: string;
  config: {
    parallel: number;
    fastSims: number;
    deepSims: number;
    deepProb: number;
    contempt: number;
    openingMix: string;
    leagueRatio: string;
  };
  generation: number;
  targetGens: number;
  totalGames: number;
  gamesPerSec: number;
  bufferSize: number;
  currentElo: number;
  bestElo: number;
  bestCrit: number;
  vsFullRbWinRate: number;
  liveGame: {
    matchup: string;
    opening: string;
    blackPlayer: string;
    whitePlayer: string;
    step: number;
    evalBlack: number;
    moves: LiveTrainingMove[];
    topPolicies: PolicyCandidate[];
  };
  history: LiveTrainingPoint[];
  checkpoints: LiveCheckpointEntry[];
}

type MonitorPage = 'live' | 'replays' | 'viewer';
type NnTab = 'policy' | 'input';

class TrainingMonitorApp {
  private root: HTMLElement;
  private appEl!: HTMLElement;
  private activePage: MonitorPage = 'live';
  private viewerReturnPage: 'live' | 'replays' = 'live';

  private liveData: LiveTrainingSnapshot | null = null;
  private pollTimer: number | null = null;
  private lastLiveKey: string | null = null;
  private lastConfigSig = '';
  private lastCkptSig = '';
  private lastChartSig = '';

  private rlArchive: RlReplayArchive | null = null;

  // Replay viewer state
  private replayTitle = '';
  private replayBlackName = '흑';
  private replayWhiteName = '백';
  private replayGeneration: number | null = null;
  private replayBadge: { label: string; tone: string } | null = null;
  private replayMoves: MoveRecord[] = [];
  private rlMoveMeta: RlReplayMove[] = [];
  private viewMoveIndex = 0;
  private autoPlayTimer: number | null = null;
  private showRuleOverlay = true;
  private showMoveNumbers = true;
  private showVcfPreview = false;
  private nnTab: NnTab = 'policy';
  private activeVcf: Coord[] | null = null;
  private forbiddenMap = new Map<string, ForbiddenCell>();
  private threats: BoardThreat[] = [];
  private historyFocusKey: string | null = null;

  // Topbar DOM
  private viewerBackBtn!: HTMLButtonElement;
  private tabLiveBtn!: HTMLButtonElement;
  private tabReplaysBtn!: HTMLButtonElement;

  // Live Monitor DOM (stable skeleton refs for zero scroll jitter)
  private livePageEl!: HTMLElement;
  private liveModelTitleEl!: HTMLElement;
  private liveStatusPillEl!: HTMLElement;
  private liveConfigRowEl!: HTMLElement;
  private liveEmptyCardEl!: HTMLElement;
  private liveContentWrapEl!: HTMLElement;
  private kpiValEls: HTMLElement[] = [];
  private kpiSubEls: HTMLElement[] = [];
  private liveGameSubEl!: HTMLElement;
  private liveStepPillEl!: HTMLElement;
  private liveBlackNameEl!: HTMLElement;
  private liveWhiteNameEl!: HTMLElement;
  private liveTugBlackPctEl!: HTMLElement;
  private liveTugWhitePctEl!: HTMLElement;
  private liveTugFillEl!: HTMLElement;
  private liveBoardSlotEl!: HTMLElement;
  private liveTopPoliciesEl!: HTMLElement;
  private liveChartSlotEl!: HTMLElement;
  private liveCkptTitleEl!: HTMLElement;
  private liveCkptTbodyEl!: HTMLElement;

  // Archive DOM
  private replaysPageEl!: HTMLElement;

  // Replay Viewer DOM
  private viewerWorkspaceEl!: HTMLElement;
  private replayLeftPanelEl!: HTMLElement;
  private historyPanelEl!: HTMLElement;
  private nnSubEl!: HTMLElement;
  private evalTugBarEl!: HTMLElement;
  private cellsGrid!: HTMLElement;
  private overlaySvg!: SVGSVGElement;
  private cellButtons: HTMLButtonElement[][] = [];
  private turnDotEl!: HTMLElement;
  private turnTextEl!: HTMLElement;
  private moveCountEl!: HTMLElement;
  private statusHintEl!: HTMLElement;
  private statusRightEl!: HTMLElement;
  private evalChartBoxEl!: HTMLElement;
  private evalChartLabelEl!: HTMLElement;
  private historyListEl!: HTMLElement;
  private replayTitleBadgeEl!: HTMLElement;
  private replayTitleTextEl!: HTMLElement;
  private replayPlayersBarEl!: HTMLElement;
  private historyNumBtn!: HTMLButtonElement;
  private bottomNumBtn!: HTMLButtonElement;
  private replaySliderEl!: HTMLInputElement;
  private replayStepLabelEl!: HTMLElement;
  private copyHistoryBtn!: HTMLButtonElement;
  private replayFirstBtn!: HTMLButtonElement;
  private replayPrevBtn!: HTMLButtonElement;
  private replayPlayBtn!: HTMLButtonElement;
  private replayNextBtn!: HTMLButtonElement;
  private replayLastBtn!: HTMLButtonElement;

  constructor(root: HTMLElement) {
    this.root = root;
    this.buildLayout();
    this.bindKeyboard();
    this.bindHistoryPopState();
    this.switchPage('live', true);
    void this.fetchLiveData(true);
    this.startPolling();
  }

  private bindHistoryPopState(): void {
    try {
      window.history.replaceState(
        { page: this.activePage, returnPage: this.viewerReturnPage },
        '',
      );
    } catch {
      // Ignore sandboxed history restrictions
    }
    window.addEventListener('popstate', (e) => {
      const st = (e.state ?? {}) as {
        page?: MonitorPage;
        returnPage?: 'live' | 'replays';
      };
      const targetPage: MonitorPage =
        st.page === 'replays' || st.page === 'viewer' || st.page === 'live'
          ? st.page
          : 'live';
      if (st.returnPage === 'live' || st.returnPage === 'replays') {
        this.viewerReturnPage = st.returnPage;
      }
      this.switchPage(targetPage, true);
    });
  }

  private buildLayout(): void {
    this.root.innerHTML = '';

    this.appEl = document.createElement('main');
    this.appEl.className = 'app';

    // Topbar
    const topbar = document.createElement('header');
    topbar.className = 'topbar';

    const topLeft = document.createElement('div');
    topLeft.className = 'topbar-left';

    this.viewerBackBtn = document.createElement('button');
    this.viewerBackBtn.type = 'button';
    this.viewerBackBtn.className = 'btn ghost';
    this.viewerBackBtn.hidden = true;
    this.viewerBackBtn.addEventListener('click', () => {
      this.stopAutoPlay();
      this.switchPage(this.viewerReturnPage);
    });

    const brand = document.createElement('div');
    brand.className = 'brand';
    const mark = document.createElement('span');
    mark.className = 'brand-mark';
    const title = document.createElement('h1');
    title.textContent = '오목 AI';
    brand.append(mark, title);

    const badge = document.createElement('span');
    badge.className = 'profile-pill';
    badge.textContent = '학습 모니터';

    topLeft.append(this.viewerBackBtn, brand, badge);

    const topRight = document.createElement('div');
    topRight.className = 'topbar-right';

    const tabs = document.createElement('nav');
    tabs.className = 'monitor-tabs';

    this.tabLiveBtn = document.createElement('button');
    this.tabLiveBtn.type = 'button';
    this.tabLiveBtn.className = 'monitor-tab active';
    this.tabLiveBtn.textContent = '실시간 모니터';
    this.tabLiveBtn.addEventListener('click', () => {
      this.stopAutoPlay();
      this.switchPage('live');
    });

    this.tabReplaysBtn = document.createElement('button');
    this.tabReplaysBtn.type = 'button';
    this.tabReplaysBtn.className = 'monitor-tab';
    this.tabReplaysBtn.textContent = '학습 기보실';
    this.tabReplaysBtn.addEventListener('click', () => {
      this.stopAutoPlay();
      this.switchPage('replays');
    });

    tabs.append(this.tabLiveBtn, this.tabReplaysBtn);

    const playAppLink = document.createElement('a');
    playAppLink.className = 'btn ghost monitor-brand-link';
    playAppLink.href = '/';
    playAppLink.target = '_blank';
    playAppLink.rel = 'noopener noreferrer';
    playAppLink.textContent = '대국 앱 ↗';

    topRight.append(tabs, playAppLink);
    topbar.append(topLeft, topRight);

    // View 1: Zero-Jitter Live Training Monitor
    this.livePageEl = document.createElement('section');
    this.livePageEl.className = 'train-page';
    this.buildLiveMonitorSkeleton();

    // View 2: RL Replay Archive
    this.replaysPageEl = document.createElement('section');
    this.replaysPageEl.className = 'archive-view';
    this.replaysPageEl.hidden = true;

    // View 3: 3-Column RL Replay Viewer
    this.buildViewerWorkspace();

    this.appEl.append(
      topbar,
      this.livePageEl,
      this.replaysPageEl,
      this.viewerWorkspaceEl,
    );
    this.root.appendChild(this.appEl);
  }

  private buildLiveMonitorSkeleton(): void {
    this.livePageEl.innerHTML = '';

    // Header Card
    const header = document.createElement('div');
    header.className = 'lobby-card train-header-bar';

    const titleSide = document.createElement('div');
    titleSide.className = 'train-header-left';

    const titleRow = document.createElement('div');
    titleRow.className = 'train-title-row';
    this.liveModelTitleEl = document.createElement('span');
    this.liveModelTitleEl.className = 'lobby-card-title';
    this.liveModelTitleEl.textContent = 'D4OverlapPattern + AlphaZero MCTS';
    this.liveStatusPillEl = document.createElement('span');
    this.liveStatusPillEl.className = 'train-status-pill idle';
    this.liveStatusPillEl.textContent = '● 불러오는 중';
    titleRow.append(this.liveModelTitleEl, this.liveStatusPillEl);

    this.liveConfigRowEl = document.createElement('div');
    this.liveConfigRowEl.className = 'train-config-row';
    titleSide.append(titleRow, this.liveConfigRowEl);

    const actionSide = document.createElement('div');
    actionSide.className = 'rl-head-actions';
    const pollBtn = document.createElement('button');
    pollBtn.type = 'button';
    pollBtn.className = 'btn';
    pollBtn.textContent = '↻ 새로고침';
    pollBtn.addEventListener('click', () => {
      void this.fetchLiveData(true);
    });
    actionSide.appendChild(pollBtn);
    header.append(titleSide, actionSide);

    // Empty Card (shown when no snapshot yet)
    this.liveEmptyCardEl = document.createElement('div');
    this.liveEmptyCardEl.className = 'lobby-card';
    this.liveEmptyCardEl.hidden = true;
    this.liveEmptyCardEl.innerHTML = `
      <div class="lobby-card-title">장기 자가대국 실행 명령어</div>
      <p class="modal-desc">아래 명령어를 실행하면 이 페이지에 1.5초마다 실시간 보드, Elo 곡선, 체크포인트 목록이 자동 반영됩니다.</p>
      <pre class="train-cli-box">python arch-bench/train_az_daemon.py --resume --gens 10 --parallel 16</pre>
    `;

    // Main Content Wrapper
    this.liveContentWrapEl = document.createElement('div');
    this.liveContentWrapEl.className = 'train-page';

    // 4 KPI Cards
    const kpiGrid = document.createElement('div');
    kpiGrid.className = 'train-kpi-grid';
    this.kpiValEls = [];
    this.kpiSubEls = [];
    const kpiTitles = [
      '세대 / 누적 대국',
      '현재 / 최고 Elo',
      'vs Full-RB 승률',
      '사활 정확도 / 버퍼',
    ];
    for (const label of kpiTitles) {
      const card = document.createElement('div');
      card.className = 'lobby-card train-kpi-card';
      const lblEl = document.createElement('span');
      lblEl.className = 'train-kpi-label';
      lblEl.textContent = label;
      const valEl = document.createElement('strong');
      valEl.className = 'train-kpi-val';
      valEl.textContent = '-';
      const subEl = document.createElement('span');
      subEl.className = 'train-kpi-sub';
      subEl.textContent = ' ';
      card.append(lblEl, valEl, subEl);
      kpiGrid.appendChild(card);
      this.kpiValEls.push(valEl);
      this.kpiSubEls.push(subEl);
    }

    // 2-Column Main Stage
    const stage = document.createElement('div');
    stage.className = 'train-stage';

    // Left: Live Self-Play Board Card
    const liveCard = document.createElement('section');
    liveCard.className = 'lobby-card train-live-card';

    const liveHead = document.createElement('div');
    liveHead.className = 'lobby-card-head';
    const liveHeadLeft = document.createElement('div');
    const liveHeadTitle = document.createElement('div');
    liveHeadTitle.className = 'lobby-card-title';
    liveHeadTitle.textContent = '실시간 자가대국 관전 (Game #1)';
    this.liveGameSubEl = document.createElement('div');
    this.liveGameSubEl.className = 'lobby-card-sub';
    this.liveGameSubEl.textContent = '대기 중';
    liveHeadLeft.append(liveHeadTitle, this.liveGameSubEl);

    this.liveStepPillEl = document.createElement('span');
    this.liveStepPillEl.className = 'profile-pill';
    this.liveStepPillEl.textContent = '0수';
    liveHead.append(liveHeadLeft, this.liveStepPillEl);

    const playersRow = document.createElement('div');
    playersRow.className = 'train-live-players';

    const bChip = document.createElement('span');
    bChip.className = 'replay-player-chip black';
    const bDot = document.createElement('span');
    bDot.className = 'history-stone black';
    this.liveBlackNameEl = document.createElement('span');
    this.liveBlackNameEl.textContent = '흑: -';
    bChip.append(bDot, this.liveBlackNameEl);

    const wChip = document.createElement('span');
    wChip.className = 'replay-player-chip white';
    const wDot = document.createElement('span');
    wDot.className = 'history-stone white';
    this.liveWhiteNameEl = document.createElement('span');
    this.liveWhiteNameEl.textContent = '백: -';
    wChip.append(wDot, this.liveWhiteNameEl);
    playersRow.append(bChip, wChip);

    const tug = document.createElement('div');
    tug.className = 'eval-tug-bar';
    const tugLabels = document.createElement('div');
    tugLabels.className = 'eval-tug-labels';
    this.liveTugBlackPctEl = document.createElement('span');
    this.liveTugBlackPctEl.textContent = '⚫ 흑 50%';
    this.liveTugWhitePctEl = document.createElement('span');
    this.liveTugWhitePctEl.textContent = '백 50% ⚪';
    tugLabels.append(this.liveTugBlackPctEl, this.liveTugWhitePctEl);
    const tugTrack = document.createElement('div');
    tugTrack.className = 'eval-tug-track';
    this.liveTugFillEl = document.createElement('div');
    this.liveTugFillEl.className = 'eval-tug-black';
    this.liveTugFillEl.style.width = '50%';
    tugTrack.appendChild(this.liveTugFillEl);
    tug.append(tugLabels, tugTrack);

    this.liveBoardSlotEl = document.createElement('div');
    this.liveBoardSlotEl.className = 'train-live-board-wrap';

    this.liveTopPoliciesEl = document.createElement('div');
    this.liveTopPoliciesEl.className = 'nn-prob-list train-top-policies';

    liveCard.append(
      liveHead,
      playersRow,
      tug,
      this.liveBoardSlotEl,
      this.liveTopPoliciesEl,
    );

    // Right Column: Chart + Checkpoint League + CLI Card
    const rightCol = document.createElement('div');
    rightCol.className = 'train-metrics-col';

    const chartCard = document.createElement('section');
    chartCard.className = 'lobby-card train-chart-card';
    const chartHead = document.createElement('div');
    chartHead.className = 'lobby-card-head';
    chartHead.innerHTML = `
      <span class="lobby-card-title">세대별 Elo & 승률 추이</span>
      <span class="nn-chart-legend"><span><span class="nn-dot-p">●</span> Elo</span><span><span class="nn-dot-v">●</span> vs Full-RB (%)</span></span>
    `;
    this.liveChartSlotEl = document.createElement('div');
    this.liveChartSlotEl.className = 'train-chart-wrap';
    chartCard.append(chartHead, this.liveChartSlotEl);

    const ckptCard = document.createElement('section');
    ckptCard.className = 'lobby-card train-ckpt-card';
    const ckptHead = document.createElement('div');
    ckptHead.className = 'lobby-card-head';
    this.liveCkptTitleEl = document.createElement('span');
    this.liveCkptTitleEl.className = 'lobby-card-title';
    this.liveCkptTitleEl.textContent = '체크포인트 리그 (0)';
    const ckptSub = document.createElement('span');
    ckptSub.className = 'lobby-card-sub';
    ckptSub.textContent = 'arch-bench/checkpoints';
    ckptHead.append(this.liveCkptTitleEl, ckptSub);

    const ckptScroll = document.createElement('div');
    ckptScroll.className = 'train-ckpt-scroll';
    const tbl = document.createElement('table');
    tbl.className = 'lb-table train-ckpt-table';
    tbl.innerHTML = `
      <thead>
        <tr><th>체크포인트</th><th>대국 수</th><th>Elo</th><th>Crit</th><th>vs Full-RB</th><th>기보</th></tr>
      </thead>
    `;
    this.liveCkptTbodyEl = document.createElement('tbody');
    tbl.appendChild(this.liveCkptTbodyEl);
    ckptScroll.appendChild(tbl);
    ckptCard.append(ckptHead, ckptScroll);

    const cliCard = document.createElement('section');
    cliCard.className = 'lobby-card';
    cliCard.innerHTML = `
      <div class="lobby-card-head"><span class="lobby-card-title">추가 세대 학습 실행</span><span class="lobby-card-sub">터미널에서 계속 이어서 학습 가능</span></div>
      <pre class="train-cli-box">python arch-bench/train_az_daemon.py --resume --gens 10 --parallel 16</pre>
    `;

    rightCol.append(chartCard, ckptCard, cliCard);
    stage.append(liveCard, rightCol);
    this.liveContentWrapEl.append(kpiGrid, stage);
    this.livePageEl.append(header, this.liveEmptyCardEl, this.liveContentWrapEl);
  }

  private buildViewerWorkspace(): void {
    this.viewerWorkspaceEl = document.createElement('div');
    this.viewerWorkspaceEl.className = 'workspace is-replay';
    this.viewerWorkspaceEl.hidden = true;

    // Left Replay Control Panel
    this.replayLeftPanelEl = document.createElement('aside');
    this.replayLeftPanelEl.className = 'history-page-panel replay-left-panel';

    const replayHead = document.createElement('div');
    replayHead.className = 'replay-match-head';

    const titleWrap = document.createElement('div');
    titleWrap.className = 'replay-match-title';
    this.replayTitleBadgeEl = document.createElement('span');
    this.replayTitleBadgeEl.className = 'archive-badge';
    this.replayTitleTextEl = document.createElement('span');
    this.replayTitleTextEl.textContent = '학습 기보 분석';
    titleWrap.append(this.replayTitleBadgeEl, this.replayTitleTextEl);

    const historyActions = document.createElement('div');
    historyActions.className = 'history-actions';

    this.historyNumBtn = document.createElement('button');
    this.historyNumBtn.type = 'button';
    this.historyNumBtn.className = 'mini-btn active';
    this.historyNumBtn.textContent = '수 번호';
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
      this.jumpToMove(this.replayMoves.length),
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

    // Center Board Column
    const playCol = document.createElement('section');
    playCol.className = 'play-col';

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

    const boardWrap = document.createElement('div');
    boardWrap.className = 'board-wrap';
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

    boardInner.append(gridSvg, this.overlaySvg, this.cellsGrid);
    boardFrame.append(coordRows, coordCols, boardInner);
    boardWrap.appendChild(boardFrame);

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
        this.renderViewerBoard();
      },
    );

    this.bottomNumBtn = this.createQuietBtn(
      '수 번호',
      this.showMoveNumbers,
      () => this.toggleMoveNumbers(),
    );

    quietPills.append(overlayBtn, this.bottomNumBtn);
    bottomStrip.append(quietPills);
    playCol.append(statusStrip, boardWrap, bottomStrip);

    // Right MCTS Policy Analysis Panel
    this.historyPanelEl = document.createElement('aside');
    this.historyPanelEl.className = 'history-page-panel replay-right-panel';
    this.nnSubEl = document.createElement('div');
    this.nnSubEl.className = 'history-subview';
    this.historyPanelEl.appendChild(this.nnSubEl);

    this.viewerWorkspaceEl.append(
      this.replayLeftPanelEl,
      playCol,
      this.historyPanelEl,
    );
  }

  private createQuietBtn(
    label: string,
    active: boolean,
    onClick: (btn: HTMLButtonElement) => void,
  ): HTMLButtonElement {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `quiet-pill${active ? ' active' : ''}`;
    btn.textContent = label;
    btn.addEventListener('click', () => onClick(btn));
    return btn;
  }

  private createReplayBtn(
    label: string,
    title: string,
    onClick: () => void,
  ): HTMLButtonElement {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'mini-btn';
    btn.textContent = label;
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
        btn.className = 'cell game-over';
        btn.setAttribute('aria-label', formatCoord(r, c));
        this.cellsGrid.appendChild(btn);
        rowBtns.push(btn);
      }
      this.cellButtons.push(rowBtns);
    }
  }

  private toggleMoveNumbers(): void {
    this.showMoveNumbers = !this.showMoveNumbers;
    this.bottomNumBtn.classList.toggle('active', this.showMoveNumbers);
    this.historyNumBtn.classList.toggle('active', this.showMoveNumbers);
    this.renderViewerBoard();
  }

  private switchPage(page: MonitorPage, fromPopState = false): void {
    this.activePage = page;

    const showLive = page === 'live';
    const showReplays = page === 'replays';
    const showViewer = page === 'viewer';

    this.livePageEl.hidden = !showLive;
    this.replaysPageEl.hidden = !showReplays;
    this.viewerWorkspaceEl.hidden = !showViewer;
    this.appEl.classList.toggle('is-side', showViewer);

    this.viewerBackBtn.hidden = !showViewer;
    this.viewerBackBtn.textContent =
      this.viewerReturnPage === 'live' ? '← 모니터' : '← 기보 목록';

    this.tabLiveBtn.classList.toggle(
      'active',
      showLive || (showViewer && this.viewerReturnPage === 'live'),
    );
    this.tabReplaysBtn.classList.toggle(
      'active',
      showReplays || (showViewer && this.viewerReturnPage === 'replays'),
    );

    if (showLive) {
      this.startPolling();
      void this.fetchLiveData(false);
    } else {
      this.stopPolling();
    }

    if (!fromPopState) {
      try {
        const cur = (window.history.state ?? {}) as {
          page?: MonitorPage;
          returnPage?: 'live' | 'replays';
        };
        if (cur.page !== page || cur.returnPage !== this.viewerReturnPage) {
          window.history.pushState(
            { page, returnPage: this.viewerReturnPage },
            '',
          );
        }
      } catch {
        // Ignore sandboxed history restrictions
      }
    }

    if (showReplays) {
      this.renderReplayArchive();
      void this.fetchRlReplays();
    } else if (showViewer) {
      this.recomputeViewer();
    }
  }

  private startPolling(): void {
    this.stopPolling();
    this.pollTimer = window.setInterval(() => {
      if (this.activePage === 'live') {
        void this.fetchLiveData(false);
      }
    }, 1500);
  }

  private stopPolling(): void {
    if (this.pollTimer !== null) {
      window.clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
  }

  private async fetchLiveData(forceRender: boolean): Promise<void> {
    try {
      const res = await fetch(`/monitor-data/live_training.json?t=${Date.now()}`);
      if (!res.ok) {
        if (forceRender) this.patchLiveMonitor();
        return;
      }
      const parsed = (await res.json()) as LiveTrainingSnapshot;
      const isFresh =
        parsed.status === 'running' &&
        Date.now() - Date.parse(parsed.updatedAt) < 90_000;
      const nextKey = [
        isFresh ? '1' : '0',
        parsed.status,
        parsed.updatedAt,
        parsed.generation,
        parsed.totalGames,
        parsed.liveGame.step,
        parsed.checkpoints.length,
      ].join('|');

      this.liveData = parsed;
      if (!forceRender && nextKey === this.lastLiveKey) {
        return;
      }
      this.lastLiveKey = nextKey;
      if (this.activePage === 'live') {
        this.patchLiveMonitor();
      }
    } catch {
      // Ignore transient atomic rename timing errors
    }
  }

  private patchLiveMonitor(): void {
    const data = this.liveData;
    const isFreshRunning = Boolean(
      data &&
        data.status === 'running' &&
        Date.now() - Date.parse(data.updatedAt) < 90_000,
    );

    this.liveModelTitleEl.textContent = data
      ? data.modelName
      : 'D4OverlapPattern + AlphaZero MCTS';
    this.liveStatusPillEl.className = `train-status-pill ${isFreshRunning ? 'live' : 'idle'}`;
    this.liveStatusPillEl.textContent = isFreshRunning
      ? '● 실시간 학습 중'
      : data
        ? '● 체크포인트 대기 / 완료'
        : '● 학습 스냅샷 없음';

    if (!data) {
      this.liveConfigRowEl.innerHTML =
        '<span class="lobby-card-sub">아직 live_training.json 데이터가 없습니다. 아래 명령으로 학습을 시작하세요.</span>';
      this.lastConfigSig = '';
      this.liveEmptyCardEl.hidden = false;
      this.liveContentWrapEl.hidden = true;
      return;
    }

    this.liveEmptyCardEl.hidden = true;
    this.liveContentWrapEl.hidden = false;

    const configSig = [
      data.params,
      data.device,
      data.config.parallel,
      data.config.fastSims,
      data.config.deepSims,
      data.config.openingMix,
      data.config.leagueRatio,
      data.config.contempt,
    ].join('|');
    if (configSig !== this.lastConfigSig) {
      this.lastConfigSig = configSig;
      this.liveConfigRowEl.innerHTML = `
        <span class="profile-pill">${data.params.toLocaleString()}p · ${data.device}</span>
        <span class="profile-pill">병렬 ${data.config.parallel}대국</span>
        <span class="profile-pill">Playout Cap ${data.config.fastSims}/${data.config.deepSims}</span>
        <span class="profile-pill">${data.config.openingMix}</span>
        <span class="profile-pill">League ${data.config.leagueRatio}</span>
        <span class="profile-pill">무승부 감점 ${data.config.contempt}</span>
      `;
    }

    // Patch 4 KPI cards in-place (zero reflow)
    const kpis: Array<[string, string]> = [
      [
        `Gen ${data.generation} / ${data.targetGens}`,
        `총 ${data.totalGames.toLocaleString()}판 · ${data.gamesPerSec.toFixed(2)}판/초`,
      ],
      [
        `${data.currentElo}`,
        `최고 ${data.bestElo} · Gen 0 기준 1000`,
      ],
      [
        `${data.vsFullRbWinRate.toFixed(1)}%`,
        '규칙 기반 교사 상대 MCTS 평가',
      ],
      [
        `${data.bestCrit.toFixed(1)}%`,
        `리플레이 ${data.bufferSize.toLocaleString()}국면`,
      ],
    ];
    kpis.forEach(([val, sub], idx) => {
      if (this.kpiValEls[idx]) this.kpiValEls[idx].textContent = val;
      if (this.kpiSubEls[idx]) this.kpiSubEls[idx].textContent = sub;
    });

    // Patch live board card in-place
    const lg = data.liveGame;
    this.liveGameSubEl.textContent = `${lg.matchup} · 오프닝: ${lg.opening} · ${lg.step}수 진행`;
    this.liveStepPillEl.textContent = `${lg.step}수`;
    this.liveBlackNameEl.textContent = `흑: ${lg.blackPlayer}`;
    this.liveWhiteNameEl.textContent = `백: ${lg.whitePlayer}`;

    const blackPct = Math.max(4, Math.min(96, Math.round(50 + lg.evalBlack * 46)));
    this.liveTugBlackPctEl.textContent = `⚫ 흑 ${blackPct}%`;
    this.liveTugWhitePctEl.textContent = `백 ${100 - blackPct}% ⚪`;
    this.liveTugFillEl.style.width = `${blackPct}%`;

    this.liveBoardSlotEl.replaceChildren(this.buildLiveTrainBoardSvg(lg));

    // Fixed-height 5-row Top-Policy list
    this.liveTopPoliciesEl.innerHTML = '';
    if (lg.topPolicies.length > 0) {
      const maxP = Math.max(0.01, lg.topPolicies[0].prob);
      for (let i = 0; i < Math.min(5, lg.topPolicies.length); i++) {
        const c = lg.topPolicies[i];
        const row = document.createElement('div');
        row.className = 'nn-prob-row';
        const pct = (c.prob * 100).toFixed(c.prob >= 0.1 ? 0 : 1);
        const wPct = Math.max(6, Math.round((c.prob / maxP) * 100));
        row.innerHTML = `
          <span class="nn-prob-rank">#${i + 1}</span>
          <span class="nn-prob-coord">${formatCoord(c.row, c.col)}</span>
          <span class="nn-prob-bar-wrap"><span class="nn-prob-bar-fill" style="width:${wPct}%"></span></span>
          <span class="nn-prob-pct">${pct}%</span>
          <span class="nn-played-slot"></span>
        `;
        this.liveTopPoliciesEl.appendChild(row);
      }
    } else {
      const emptyPol = document.createElement('div');
      emptyPol.className = 'nn-insight-box';
      emptyPol.textContent = 'MCTS 후보 정책을 집계 중입니다.';
      this.liveTopPoliciesEl.appendChild(emptyPol);
    }

    // Elo & Win-Rate chart (rebuild only when history points change)
    const chartSig = data.history
      .map((p) => `${p.gen}:${p.elo}:${p.vsRbWr}`)
      .join(',');
    if (chartSig !== this.lastChartSig) {
      this.lastChartSig = chartSig;
      if (data.history.length > 0) {
        this.liveChartSlotEl.replaceChildren(this.buildTrainEloSvg(data.history));
      } else {
        const empty = document.createElement('div');
        empty.className = 'lb-empty';
        empty.textContent = '1세대 완료 후 추이 그래프가 표시됩니다.';
        this.liveChartSlotEl.replaceChildren(empty);
      }
    }

    // Checkpoint League table (rebuild tbody only when rows change)
    const ckptSig = data.checkpoints
      .map((c) => `${c.id}:${c.totalGames}:${c.elo}:${c.crit}:${c.vsRbRecord}`)
      .join(',');
    if (ckptSig !== this.lastCkptSig) {
      this.lastCkptSig = ckptSig;
      this.liveCkptTitleEl.textContent = `체크포인트 리그 (${data.checkpoints.length})`;
      const prevScrollTop =
        this.liveCkptTbodyEl.closest('.train-ckpt-scroll')?.scrollTop ?? 0;
      this.liveCkptTbodyEl.innerHTML = '';
      for (const ckpt of [...data.checkpoints].reverse()) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>Gen ${ckpt.gen}</strong></td>
          <td>${ckpt.totalGames}</td>
          <td>${ckpt.elo}</td>
          <td>${ckpt.crit.toFixed(1)}%</td>
          <td>${ckpt.vsRbRecord} (${ckpt.vsRbWr.toFixed(0)}%)</td>
          <td></td>
        `;
        const tdAction = tr.lastElementChild as HTMLElement;
        if (ckpt.replayId) {
          const b = document.createElement('button');
          b.type = 'button';
          b.className = 'mini-btn';
          b.textContent = '복기 ›';
          b.addEventListener('click', () => {
            void this.openCheckpointReplay(ckpt.replayId!);
          });
          tdAction.appendChild(b);
        } else {
          tdAction.textContent = '-';
        }
        this.liveCkptTbodyEl.appendChild(tr);
      }
      const scrollEl = this.liveCkptTbodyEl.closest('.train-ckpt-scroll');
      if (scrollEl) scrollEl.scrollTop = prevScrollTop;
    }
  }

  private buildLiveTrainBoardSvg(
    liveGame: LiveTrainingSnapshot['liveGame'],
  ): SVGSVGElement {
    const size = 450;
    const pad = 26;
    const cell = (size - pad * 2) / (BOARD_SIZE - 1);
    const svg = createSvgElement('svg', {
      class: 'train-live-svg',
      viewBox: `0 0 ${size} ${size}`,
    }) as SVGSVGElement;

    svg.appendChild(
      createSvgElement('rect', {
        x: 0,
        y: 0,
        width: size,
        height: size,
        rx: 12,
        fill: '#dfb97c',
        stroke: '#be9757',
        'stroke-width': 1.5,
      }),
    );

    const gx0 = pad;
    const gx1 = pad + (BOARD_SIZE - 1) * cell;
    for (let i = 0; i < BOARD_SIZE; i++) {
      const p = pad + i * cell;
      svg.appendChild(
        createSvgElement('line', {
          x1: gx0,
          y1: p,
          x2: gx1,
          y2: p,
          stroke: 'rgba(65, 45, 18, 0.45)',
          'stroke-width': 1,
        }),
      );
      svg.appendChild(
        createSvgElement('line', {
          x1: p,
          y1: gx0,
          x2: p,
          y2: gx1,
          stroke: 'rgba(65, 45, 18, 0.45)',
          'stroke-width': 1,
        }),
      );

      const colLbl = createSvgElement('text', {
        x: p,
        y: size - 6,
        'text-anchor': 'middle',
        'font-size': 9,
        'font-weight': 700,
        fill: 'rgba(65, 45, 18, 0.72)',
      });
      colLbl.textContent = String.fromCharCode(65 + i);
      svg.appendChild(colLbl);

      const rowLbl = createSvgElement('text', {
        x: 10,
        y: p + 3,
        'text-anchor': 'middle',
        'font-size': 9,
        'font-weight': 700,
        fill: 'rgba(65, 45, 18, 0.72)',
      });
      rowLbl.textContent = String(BOARD_SIZE - i);
      svg.appendChild(rowLbl);
    }

    for (const st of STAR_POINTS) {
      svg.appendChild(
        createSvgElement('circle', {
          cx: pad + st.col * cell,
          cy: pad + st.row * cell,
          r: 2.8,
          fill: 'rgba(55, 38, 15, 0.72)',
        }),
      );
    }

    const maxProb =
      liveGame.topPolicies.length > 0
        ? Math.max(0.05, liveGame.topPolicies[0].prob)
        : 1;
    for (const c of liveGame.topPolicies.slice(0, 8)) {
      const alpha = Math.min(0.62, 0.14 + (c.prob / maxProb) * 0.48);
      const cx = pad + c.col * cell;
      const cy = pad + c.row * cell;
      const box = cell * 0.76;
      svg.appendChild(
        createSvgElement('rect', {
          x: cx - box / 2,
          y: cy - box / 2,
          width: box,
          height: box,
          rx: 4,
          fill: `rgba(232, 93, 42, ${alpha.toFixed(2)})`,
        }),
      );
      if (c.prob >= 0.04) {
        const pt = createSvgElement('text', {
          x: cx,
          y: cy + 3,
          'text-anchor': 'middle',
          'font-size': 8.5,
          'font-weight': 800,
          fill: '#1e1b18',
        });
        pt.textContent = `${Math.round(c.prob * 100)}%`;
        svg.appendChild(pt);
      }
    }

    const stoneR = cell * 0.43;
    liveGame.moves.forEach((m, idx) => {
      const cx = pad + m.col * cell;
      const cy = pad + m.row * cell;
      const isBlack = m.player === BLACK;
      const isLast = idx === liveGame.moves.length - 1;
      svg.appendChild(
        createSvgElement('circle', {
          cx,
          cy,
          r: stoneR,
          fill: isBlack ? '#1c1b19' : '#fbfaf7',
          stroke: isLast ? '#e85d2a' : isBlack ? '#000' : 'rgba(60, 50, 35, 0.45)',
          'stroke-width': isLast ? 2.2 : 1,
        }),
      );
      const numEl = createSvgElement('text', {
        x: cx,
        y: cy + 3.2,
        'text-anchor': 'middle',
        'font-size': 8.5,
        'font-weight': 700,
        fill: isBlack ? '#f4f2ed' : '#262420',
      });
      numEl.textContent = String(idx + 1);
      svg.appendChild(numEl);
    });

    return svg;
  }

  private buildTrainEloSvg(points: LiveTrainingPoint[]): SVGSVGElement {
    const w = 420;
    const h = 135;
    const svg = createSvgElement('svg', {
      class: 'train-chart-svg',
      viewBox: `0 0 ${w} ${h}`,
    }) as SVGSVGElement;

    if (points.length === 0) return svg;

    const minElo = Math.min(950, ...points.map((p) => p.elo)) - 25;
    const maxElo = Math.max(1250, ...points.map((p) => p.elo)) + 25;
    const xOf = (i: number) =>
      points.length === 1 ? w / 2 : 16 + (i / (points.length - 1)) * (w - 32);
    const yElo = (v: number) =>
      h - 18 - ((v - minElo) / Math.max(50, maxElo - minElo)) * (h - 32);
    const yWr = (v: number) => h - 18 - (Math.max(0, Math.min(100, v)) / 100) * (h - 32);

    svg.appendChild(
      createSvgElement('line', {
        x1: 12,
        y1: yWr(50),
        x2: w - 12,
        y2: yWr(50),
        stroke: '#ded8cc',
        'stroke-dasharray': '4 3',
        'stroke-width': 1,
      }),
    );

    const eloPts = points.map((p, i) => `${xOf(i).toFixed(1)},${yElo(p.elo).toFixed(1)}`);
    const wrPts = points.map((p, i) => `${xOf(i).toFixed(1)},${yWr(p.vsRbWr).toFixed(1)}`);

    svg.appendChild(
      createSvgElement('polyline', {
        fill: 'none',
        stroke: '#e85d2a',
        'stroke-width': 2.2,
        'stroke-linecap': 'round',
        'stroke-linejoin': 'round',
        points: eloPts.join(' '),
      }),
    );
    svg.appendChild(
      createSvgElement('polyline', {
        fill: 'none',
        stroke: '#0f8b8d',
        'stroke-width': 2,
        'stroke-dasharray': '5 3',
        'stroke-linecap': 'round',
        'stroke-linejoin': 'round',
        points: wrPts.join(' '),
      }),
    );

    points.forEach((p, i) => {
      svg.appendChild(
        createSvgElement('circle', {
          cx: xOf(i),
          cy: yElo(p.elo),
          r: 3.2,
          fill: '#e85d2a',
        }),
      );
      svg.appendChild(
        createSvgElement('circle', {
          cx: xOf(i),
          cy: yWr(p.vsRbWr),
          r: 2.8,
          fill: '#0f8b8d',
        }),
      );
    });

    return svg;
  }

  private async openCheckpointReplay(replayId: string): Promise<void> {
    await this.fetchRlReplays();
    const target = this.rlArchive?.games.find((g) => g.id === replayId);
    if (target) {
      this.viewerReturnPage = 'live';
      this.loadReplayGame(target);
      this.switchPage('viewer');
    } else {
      this.switchPage('replays');
    }
  }

  private async fetchRlReplays(): Promise<void> {
    try {
      const res = await fetch(`/monitor-data/rl_replays.json?t=${Date.now()}`);
      if (!res.ok) return;
      const txt = await res.text();
      this.applyRlArchiveText(txt, false);
    } catch {
      // Ignore transient errors
    }
  }

  private applyRlArchiveText(txt: string, autoLoadLatest: boolean): void {
    try {
      const parsed = JSON.parse(txt) as RlReplayArchive;
      if (!parsed.games || parsed.games.length === 0) return;

      this.rlArchive = parsed;
      if (autoLoadLatest) {
        this.viewerReturnPage = 'replays';
        this.loadReplayGame(parsed.games[parsed.games.length - 1]);
        this.switchPage('viewer');
      } else if (this.activePage === 'replays') {
        this.renderReplayArchive();
      } else if (this.activePage === 'viewer') {
        this.renderNnPanel();
      }
    } catch {
      // Ignore malformed JSON
    }
  }

  private renderReplayArchive(): void {
    this.replaysPageEl.innerHTML = '';

    const wrap = document.createElement('div');
    wrap.className = 'monitor-archive-wrap';

    const games = this.rlArchive?.games ?? [];
    const info = this.rlArchive?.modelInfo;
    const modelLabel = info?.name ?? 'OverlapPattern';

    const rlCard = document.createElement('section');
    rlCard.className = 'lobby-card';

    const rlHead = document.createElement('div');
    rlHead.className = 'lobby-card-head';
    const rlTitleWrap = document.createElement('div');
    rlTitleWrap.innerHTML = `<span class="lobby-card-title">AI 학습 기보${games.length > 0 ? ` (${games.length})` : ''}</span>${info ? ` <span class="lobby-card-sub">· ${modelLabel} ${info.params.toLocaleString()}p</span>` : ''}`;

    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.json,application/json';
    fileInput.hidden = true;
    fileInput.addEventListener('change', () => {
      const f = fileInput.files?.[0];
      if (!f) return;
      void f.text().then((txt) => {
        this.applyRlArchiveText(txt, true);
        fileInput.value = '';
      });
    });

    const rlActions = document.createElement('div');
    rlActions.className = 'rl-head-actions';

    const refreshBtn = document.createElement('button');
    refreshBtn.type = 'button';
    refreshBtn.className = 'btn';
    refreshBtn.textContent = '↻ 새로고침';
    refreshBtn.addEventListener('click', () => {
      void this.fetchRlReplays();
    });

    const fileBtn = document.createElement('button');
    fileBtn.type = 'button';
    fileBtn.className = 'btn';
    fileBtn.textContent = '+ JSON 열기';
    fileBtn.addEventListener('click', () => fileInput.click());

    rlActions.append(refreshBtn, fileBtn, fileInput);
    rlHead.append(rlTitleWrap, rlActions);
    rlCard.appendChild(rlHead);

    const rlList = document.createElement('div');
    rlList.className = 'archive-list';
    if (games.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'lb-empty';
      empty.textContent = 'AI 학습 기보(rl_replays.json)를 불러오는 중이거나 기보가 없습니다.';
      rlList.appendChild(empty);
    } else {
      for (const g of [...games].reverse()) {
        const row = document.createElement('button');
        row.type = 'button';
        row.className = 'archive-row';

        const badgeLabel =
          g.winner === BLACK ? '흑 승' : g.winner === WHITE ? '백 승' : '무승부';
        const badgeCls =
          g.winner === BLACK ? 'black' : g.winner === WHITE ? 'white' : 'draw';
        const bName = g.blackPlayer ?? `OverlapPattern (Gen ${g.generation})`;
        const wName = g.whitePlayer ?? `OverlapPattern (Gen ${g.generation})`;

        row.innerHTML = `
          <div class="archive-row-left">
            <span class="archive-badge ${badgeCls}">${badgeLabel}</span>
            <div class="archive-info">
              <span class="archive-title">${g.label}</span>
              <span class="archive-sub">⚫ 흑: ${bName} · ⚪ 백: ${wName}</span>
            </div>
          </div>
          <div class="archive-row-right">
            <span>${g.moves.length}수</span>
            <span class="archive-cta">분석 ›</span>
          </div>
        `;
        row.addEventListener('click', () => {
          this.viewerReturnPage = 'replays';
          this.loadReplayGame(g);
          this.switchPage('viewer');
        });
        rlList.appendChild(row);
      }
    }
    rlCard.appendChild(rlList);
    wrap.appendChild(rlCard);
    this.replaysPageEl.appendChild(wrap);
  }

  private loadReplayGame(game: RlReplayGame): void {
    this.stopAutoPlay();
    const scratchBoard = createEmptyBoard();
    const moves: MoveRecord[] = [];
    this.rlMoveMeta = game.moves;
    const nnBlackRaw: Array<number | null> = game.moves.map((mv) =>
      typeof mv.value === 'number' ? (mv.player === BLACK ? mv.value : -mv.value) : null,
    );

    for (let i = 0; i < game.moves.length; i++) {
      const mv = game.moves[i];
      const tag = getMoveTagOnBoard(scratchBoard, mv.row, mv.col, mv.player);
      scratchBoard[mv.row][mv.col] = mv.player;

      const tacScore = evaluatePositionAdvantage(
        scratchBoard,
        opponentOf(mv.player),
        RULES,
      );
      const vCur = nnBlackRaw[i];
      const vNext = nnBlackRaw[Math.min(game.moves.length - 1, i + 1)];
      const nnBlack =
        vCur !== null && vNext !== null
          ? 0.5 * vCur + 0.5 * vNext
          : (vNext ?? vCur ?? tacScore);
      const rawScore =
        Math.abs(tacScore) >= 0.85
          ? 0.25 * nnBlack + 0.75 * tacScore
          : 0.5 * nnBlack + 0.5 * tacScore;
      const clamped = Math.max(-0.96, Math.min(0.96, rawScore));
      const prevScore =
        moves.length > 0 ? (moves[moves.length - 1].evalScore ?? 0) : null;
      const evalScore =
        prevScore !== null && Math.abs(clamped - prevScore) < 0.25
          ? 0.3 * prevScore + 0.7 * clamped
          : clamped;

      moves.push({
        row: mv.row,
        col: mv.col,
        player: mv.player,
        moveNumber: i + 1,
        tag,
        evalScore,
      });
    }

    const last = moves[moves.length - 1];
    const win = last ? checkWinner(scratchBoard, last, RULES) : null;
    const nextTurn = last ? opponentOf(last.player) : BLACK;
    const draw = !win && !hasAnyLegalMove(scratchBoard, nextTurn, RULES);
    if (win && last) {
      last.evalScore = win.winner === BLACK ? 1 : -1;
    } else if (draw && last) {
      last.evalScore = 0;
    }

    this.replayMoves = moves;
    this.replayGeneration = game.generation;
    this.replayTitle = game.label;
    const fallbackModel = this.rlArchive?.modelInfo?.name ?? 'OverlapPattern';
    this.replayBlackName =
      game.blackPlayer ?? `${fallbackModel} (Gen ${game.generation})`;
    this.replayWhiteName =
      game.whitePlayer ?? `${fallbackModel} (Gen ${game.generation})`;
    this.replayBadge = {
      label:
        game.winner === BLACK ? '흑 승' : game.winner === WHITE ? '백 승' : '무승부',
      tone:
        game.winner === BLACK ? 'black' : game.winner === WHITE ? 'white' : 'draw',
    };
    this.viewMoveIndex = this.findInitialPolicyIndex(game.moves);
    this.showVcfPreview = false;
    this.nnTab = 'policy';
  }

  private findInitialPolicyIndex(moves: RlReplayMove[]): number {
    const limit = Math.max(0, moves.length - 1);
    for (let i = Math.min(4, limit); i < limit; i++) {
      if (moves[i]?.topPolicies?.length) return i;
    }
    for (let i = 0; i < limit; i++) {
      if (moves[i]?.topPolicies?.length) return i;
    }
    return Math.min(4, limit);
  }

  private getViewerState(): {
    board: Board;
    moves: MoveRecord[];
    turnPlayer: Player;
    winner: WinResult | null;
    isDraw: boolean;
  } {
    const visibleMoves = this.replayMoves.slice(0, this.viewMoveIndex);
    const board = createEmptyBoard();
    for (const m of visibleMoves) {
      board[m.row][m.col] = m.player;
    }
    const last = visibleMoves[visibleMoves.length - 1];
    const winner = last ? checkWinner(board, last, RULES) : null;
    const turnPlayer = last ? opponentOf(last.player) : BLACK;
    const isDraw = !winner && !hasAnyLegalMove(board, turnPlayer, RULES);
    return { board, moves: visibleMoves, turnPlayer, winner, isDraw };
  }

  private recomputeViewer(): void {
    const view = this.getViewerState();

    this.forbiddenMap.clear();
    if (!view.winner && !view.isDraw && view.turnPlayer === BLACK) {
      for (const cell of getForbiddenMoves(view.board, RULES)) {
        this.forbiddenMap.set(coordKey(cell.row, cell.col), cell);
      }
    }

    this.threats = view.winner ? [] : scanBoardThreats(view.board, RULES);
    this.activeVcf = null;
    if (!view.winner && !view.isDraw && view.moves.length >= 4) {
      const vcf = findVcfSequence(view.board, view.turnPlayer, RULES, 7, 25);
      if (vcf && vcf.length >= 1) this.activeVcf = vcf;
    }
    if (!this.activeVcf) this.showVcfPreview = false;

    this.renderViewerStatus(view);
    this.renderViewerHistory(view);
    this.renderNnPanel(view);
    this.renderViewerBoard(view);
  }

  private renderViewerStatus(view: ReturnType<TrainingMonitorApp['getViewerState']>): void {
    const activePlayer = view.winner ? view.winner.winner : view.turnPlayer;
    this.turnDotEl.className = `turn-dot ${activePlayer === BLACK ? 'black' : 'white'}`;

    if (view.winner) {
      const wName =
        view.winner.winner === BLACK ? this.replayBlackName : this.replayWhiteName;
      this.turnTextEl.textContent = `${playerLabel(view.winner.winner)} (${wName}) 승리!`;
    } else if (view.isDraw) {
      this.turnTextEl.textContent = '무승부';
    } else {
      const tName =
        view.turnPlayer === BLACK ? this.replayBlackName : this.replayWhiteName;
      this.turnTextEl.textContent = `${playerLabel(view.turnPlayer)} (${tName}) 차례`;
    }

    const total = this.replayMoves.length;
    this.moveCountEl.textContent = `${this.viewMoveIndex}/${total}수`;
    this.statusHintEl.textContent =
      this.viewMoveIndex < total
        ? `복기 중 (${this.viewMoveIndex}/${total})`
        : 'RL 학습 기보';
    this.statusHintEl.className = view.winner ? 'status-hint win' : 'status-hint';

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
        this.renderViewerStatus(view);
        this.renderViewerBoard(view);
      });
      this.statusRightEl.appendChild(vcfBtn);
    }
  }

  private renderViewerHistory(view: ReturnType<TrainingMonitorApp['getViewerState']>): void {
    const prevScrollTop = this.historyListEl.scrollTop;
    this.historyListEl.innerHTML = '';

    const moves = this.replayMoves;
    const total = moves.length;
    this.replayTitleTextEl.textContent = this.replayTitle || `수 기록 (${total}수)`;

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

    this.renderEvalChart(view);

    if (total === 0) {
      const empty = document.createElement('div');
      empty.className = 'history-empty';
      empty.textContent = '아직 착수 기록이 없습니다.';
      this.historyListEl.appendChild(empty);
      return;
    }

    let activeRowEl: HTMLElement | null = null;
    for (const m of moves) {
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
        tag.className = `history-tag ${m.tag === '승리' ? 'win' : m.tag === '열린3' ? 'three' : 'four'}`;
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

  private renderEvalChart(view: ReturnType<TrainingMonitorApp['getViewerState']>): void {
    const oldSvg = this.evalChartBoxEl.querySelector('.eval-chart-svg');
    if (oldSvg) oldSvg.remove();

    const moves = this.replayMoves;
    const curScore =
      this.viewMoveIndex > 0
        ? (moves[this.viewMoveIndex - 1]?.evalScore ?? 0)
        : 0;

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

  private renderNnPanel(view = this.getViewerState()): void {
    this.nnSubEl.innerHTML = '';

    const nextMeta = this.rlMoveMeta[this.viewMoveIndex];
    const actualNext = this.replayMoves[this.viewMoveIndex];
    const hasRlPolicy = Boolean(nextMeta?.topPolicies?.length);
    const finished = Boolean(view.winner || view.isDraw);

    const colHead = document.createElement('div');
    colHead.className = 'nn-col-head';
    const modeTitle = document.createElement('span');
    if (finished) {
      modeTitle.textContent = '대국 종료 · 최종 복기';
    } else if (this.nnTab === 'policy') {
      modeTitle.textContent = `수읽기 후보 분석 (${this.viewMoveIndex + 1}수)`;
    } else {
      modeTitle.textContent = '합법 착수 영역';
    }

    const tabs = document.createElement('div');
    tabs.className = 'nn-mini-tabs';
    for (const [id, label] of [
      ['policy', '후보 수'],
      ['input', '합법 영역'],
    ] as const) {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = label;
      if (this.nnTab === id) b.classList.add('active');
      b.addEventListener('click', () => {
        this.nnTab = id;
        this.renderNnPanel();
        this.renderViewerBoard();
      });
      tabs.appendChild(b);
    }
    colHead.append(modeTitle, tabs);
    this.nnSubEl.appendChild(colHead);

    const probBox = document.createElement('div');
    probBox.className = 'nn-prob-list';

    if (!finished) {
      const candidates: Array<{ row: number; col: number; prob: number }> =
        hasRlPolicy && nextMeta?.topPolicies
          ? nextMeta.topPolicies.slice(0, 8)
          : computeTacticalCandidates(view.board, view.turnPlayer);

      if (candidates.length > 0) {
        const maxProb = Math.max(0.01, candidates[0].prob);
        candidates.forEach((cand, idx) => {
          const isPlayed =
            Boolean(actualNext) &&
            actualNext.row === cand.row &&
            actualNext.col === cand.col;
          const row = document.createElement('div');
          row.className = `nn-prob-row${isPlayed ? ' is-played' : ''}`;

          const coord = formatCoord(cand.row, cand.col);
          const pct = (cand.prob * 100).toFixed(cand.prob >= 0.1 ? 0 : 1);
          const barW = Math.max(6, Math.round((cand.prob / maxProb) * 100));

          row.innerHTML = `
            <span class="nn-prob-rank">#${idx + 1}</span>
            <span class="nn-prob-coord">${coord}</span>
            <span class="nn-prob-bar-wrap"><span class="nn-prob-bar-fill" style="width:${barW}%"></span></span>
            <span class="nn-prob-pct">${pct}%</span>
            ${isPlayed ? '<span class="nn-played-badge">실전 착수</span>' : '<span class="nn-played-slot"></span>'}
          `;
          row.addEventListener('mouseenter', () => {
            this.historyFocusKey = coordKey(cand.row, cand.col);
            this.updateHistoryFocus();
          });
          row.addEventListener('mouseleave', () => {
            this.historyFocusKey = null;
            this.updateHistoryFocus();
          });
          probBox.appendChild(row);
        });
      } else {
        const empty = document.createElement('div');
        empty.className = 'nn-insight-box';
        empty.textContent = '초반 랜덤 착수 또는 분포 평탄 구간입니다.';
        probBox.appendChild(empty);
      }

      if (actualNext) {
        const insight = document.createElement('div');
        insight.className = 'nn-insight-box';
        const moverName =
          actualNext.player === BLACK ? this.replayBlackName : this.replayWhiteName;
        const playedCoord = formatCoord(actualNext.row, actualNext.col);
        const rankIdx = candidates.findIndex(
          (c) => c.row === actualNext.row && c.col === actualNext.col,
        );
        const tagText = actualNext.tag ? ` [${actualNext.tag}]` : '';
        const topHint =
          candidates.length > 0
            ? ` (추천 1순위 ${formatCoord(candidates[0].row, candidates[0].col)})`
            : '';

        if (nextMeta?.moverType === 'opening') {
          insight.textContent = `초반 오프닝 시드 (${moverName}): ${playedCoord} 착수${topHint}${tagText}`;
        } else if (nextMeta?.moverType === 'rule') {
          insight.textContent = `룰베이스 스파링 (${moverName}): ${playedCoord} 착수${topHint}${tagText}`;
        } else if (rankIdx === 0) {
          insight.textContent = `실전 ${moverName}: 1순위 추천 ${playedCoord} 선택${tagText}`;
        } else if (rankIdx > 0) {
          const topCoord = formatCoord(candidates[0].row, candidates[0].col);
          insight.textContent = `실전 ${moverName}: ${rankIdx + 1}순위 탐색 샘플링 ${playedCoord} (1순위 ${topCoord})${tagText}`;
        } else if (candidates.length > 0) {
          const topCoord = formatCoord(candidates[0].row, candidates[0].col);
          insight.textContent = `실전 ${moverName}: ${playedCoord} 착수 (추천 1순위 ${topCoord})${tagText}`;
        } else {
          insight.textContent = `실전 ${moverName}: ${playedCoord} 착수${tagText}`;
        }
        probBox.appendChild(insight);
      }
    } else {
      const doneBox = document.createElement('div');
      doneBox.className = 'nn-insight-box';
      doneBox.textContent = view.winner
        ? `${view.winner.winner === BLACK ? this.replayBlackName : this.replayWhiteName}의 승리로 종료되었습니다. 이전 수로 이동해 후보 수를 확인하세요.`
        : '무승부로 종료되었습니다. 이전 수로 이동해 후보 수를 확인하세요.';
      probBox.appendChild(doneBox);
    }

    this.nnSubEl.appendChild(probBox);

    const tacCard = document.createElement('div');
    tacCard.className = 'nn-tactical-card';

    const tacHead = document.createElement('div');
    tacHead.className = 'nn-tactical-head';
    const turnPill = finished
      ? view.winner
        ? `${playerLabel(view.winner.winner)} 승리`
        : '무승부'
      : `${playerLabel(view.turnPlayer)} 차례 (${this.viewMoveIndex + 1}수)`;
    tacHead.innerHTML = `<span>국면 전술 요약</span><span class="nn-tactical-pill">${turnPill}</span>`;

    let moverMode = '전술 수읽기';
    if (finished) {
      moverMode = `총 ${view.moves.length}수 완국`;
    } else if (nextMeta?.moverType === 'opening') {
      moverMode = '초반 오프닝 시드';
    } else if (nextMeta?.moverType === 'rule') {
      const ruleName =
        view.turnPlayer === BLACK ? this.replayBlackName : this.replayWhiteName;
      moverMode = `룰베이스 (${ruleName})`;
    } else {
      const turnName =
        view.turnPlayer === BLACK ? this.replayBlackName : this.replayWhiteName;
      const genTag =
        this.replayGeneration !== null ? `Gen ${this.replayGeneration}` : 'MCTS';
      moverMode = turnName.includes('MCTS') ? turnName : `${turnName} (${genTag})`;
    }

    const vcfVal =
      !finished && this.activeVcf && this.activeVcf.length > 0
        ? `연속 사 ${this.activeVcf.length}수 (${formatCoord(this.activeVcf[0].row, this.activeVcf[0].col)})`
        : '연속 사 없음';
    const forbidVal = finished
      ? '대국 종료'
      : view.turnPlayer === BLACK
        ? this.forbiddenMap.size > 0
          ? `흑 금수 ${this.forbiddenMap.size}곳`
          : '흑 금수 없음'
        : '백 차례 (금수 무관)';

    const grid = document.createElement('div');
    grid.className = 'nn-tactical-grid';
    const items: Array<[string, string, string?]> = [
      ['착수 주체', moverMode],
      ['VCF · 금수', `${vcfVal} · ${forbidVal}`],
      ['흑 위협', formatPlayerThreats(this.threats, BLACK), this.threats.some((t) => t.player === BLACK) ? 'has-threat' : ''],
      ['백 위협', formatPlayerThreats(this.threats, WHITE), this.threats.some((t) => t.player === WHITE) ? 'has-threat' : ''],
    ];
    for (const [label, val, cls] of items) {
      const item = document.createElement('div');
      item.className = `nn-tactical-item${cls ? ` ${cls}` : ''}`;
      item.innerHTML = `<span class="nn-tactical-label">${label}</span><span class="nn-tactical-val">${val}</span>`;
      grid.appendChild(item);
    }

    tacCard.append(tacHead, grid);
    this.nnSubEl.appendChild(tacCard);
  }

  private renderViewerBoard(view = this.getViewerState()): void {
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
    if (this.showRuleOverlay && !view.winner) {
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

    const vcfMap = new Map<string, { step: number; player: Player }>();
    if (this.showVcfPreview && this.activeVcf) {
      this.activeVcf.forEach((c, idx) => {
        const k = coordKey(c.row, c.col);
        if (!vcfMap.has(k)) {
          const p = idx % 2 === 0 ? view.turnPlayer : oppPlayer;
          vcfMap.set(k, { step: idx + 1, player: p });
        }
      });
    }

    const nnPolicyMap = new Map<string, number>();
    if (this.nnTab === 'policy') {
      const nextMeta = this.rlMoveMeta[this.viewMoveIndex];
      if (nextMeta?.topPolicies) {
        for (const p of nextMeta.topPolicies) {
          nnPolicyMap.set(coordKey(p.row, p.col), p.prob);
        }
      }
    }
    const maxNnProb = Math.max(0.01, ...nnPolicyMap.values());

    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        const btn = this.cellButtons[r][c];
        const stone = view.board[r][c];
        const key = coordKey(r, c);
        const forbidden = this.forbiddenMap.get(key);

        btn.className = 'cell game-over';
        btn.innerHTML = '';

        if (stone !== EMPTY) {
          btn.classList.add('occupied');
          const stoneEl = document.createElement('span');
          stoneEl.className = `stone ${stone === BLACK ? 'black' : 'white'}`;

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

        const vcfItem = vcfMap.get(key);
        if (vcfItem) {
          const vcfEl = document.createElement('span');
          vcfEl.className = `vcf-ghost ${vcfItem.player === BLACK ? 'black' : 'white'}`;
          vcfEl.textContent = String(vcfItem.step);
          btn.appendChild(vcfEl);
          continue;
        }

        if (!view.winner && !view.isDraw) {
          if (this.nnTab === 'input' && !forbidden) {
            const legalEl = document.createElement('span');
            legalEl.className = 'nn-input-legal';
            btn.appendChild(legalEl);
          } else if (this.nnTab === 'policy') {
            const prob = nnPolicyMap.get(key);
            if (prob && prob > 0) {
              const heat = document.createElement('span');
              heat.className = 'nn-board-heat';
              const alpha = 0.22 + 0.68 * (prob / maxNnProb);
              heat.style.background = `rgba(232, 93, 42, ${alpha.toFixed(2)})`;
              if (prob >= 0.03) {
                const pctEl = document.createElement('span');
                pctEl.className = 'nn-board-pct';
                pctEl.textContent = `${Math.round(prob * 100)}%`;
                heat.appendChild(pctEl);
              }
              btn.appendChild(heat);
            }
          }
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
        }
      }
    }

    this.renderSvgOverlays(view);
  }

  private renderSvgOverlays(view = this.getViewerState()): void {
    this.overlaySvg.innerHTML = '';

    if (view.winner && view.winner.line.length >= 2) {
      const first = view.winner.line[0];
      const last = view.winner.line[view.winner.line.length - 1];
      const x1 = first.col * 100 + 50;
      const y1 = first.row * 100 + 50;
      const x2 = last.col * 100 + 50;
      const y2 = last.row * 100 + 50;
      const length = Math.hypot(x2 - x1, y2 - y1);

      this.overlaySvg.appendChild(
        createSvgElement('line', { x1, y1, x2, y2, class: 'win-line-glow' }),
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

    if (!this.showRuleOverlay) return;

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

  private updateHistoryFocus(): void {
    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        const key = coordKey(r, c);
        const focused = this.historyFocusKey === key;
        const target =
          this.cellButtons[r][c].querySelector('.stone') ??
          this.cellButtons[r][c].querySelector('.nn-board-heat');
        if (target) {
          target.classList.toggle('history-focus', focused);
        }
      }
    }
  }

  private jumpToMove(index: number): void {
    const total = this.replayMoves.length;
    const clamped = Math.max(0, Math.min(total, index));
    if (clamped === this.viewMoveIndex) return;
    if (this.autoPlayTimer !== null && clamped >= total) {
      this.stopAutoPlay();
    }
    this.viewMoveIndex = clamped;
    this.showVcfPreview = false;
    this.recomputeViewer();
  }

  private toggleAutoPlay(): void {
    if (this.autoPlayTimer !== null) {
      this.stopAutoPlay();
      this.renderViewerHistory(this.getViewerState());
      return;
    }
    const total = this.replayMoves.length;
    if (total === 0) return;
    if (this.viewMoveIndex >= total) {
      this.viewMoveIndex = 0;
      this.showVcfPreview = false;
      this.recomputeViewer();
    }
    this.autoPlayTimer = window.setInterval(() => {
      if (this.viewMoveIndex >= this.replayMoves.length) {
        this.stopAutoPlay();
        this.renderViewerHistory(this.getViewerState());
        return;
      }
      this.viewMoveIndex++;
      this.showVcfPreview = false;
      this.recomputeViewer();
    }, 620);
    this.renderViewerHistory(this.getViewerState());
  }

  private stopAutoPlay(): void {
    if (this.autoPlayTimer !== null) {
      window.clearInterval(this.autoPlayTimer);
      this.autoPlayTimer = null;
    }
  }

  private copyHistory(): void {
    const moves = this.replayMoves;
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

  private bindKeyboard(): void {
    window.addEventListener('keydown', (e) => {
      if (this.activePage === 'viewer' && this.replayMoves.length > 0) {
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
}

const mount = document.getElementById('monitor-app');
if (mount) {
  new TrainingMonitorApp(mount);
}
