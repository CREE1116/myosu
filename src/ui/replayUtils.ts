import { analyzeMove, formatCoord, opponentOf } from '../engine/renju';
import {
  BLACK,
  BOARD_SIZE,
  Board,
  BoardThreat,
  Coord,
  EMPTY,
  ForbiddenReason,
  MoveRecord,
  Player,
  RuleSet,
} from '../engine/types';

export const RULES: RuleSet = 'renju';

export const STAR_POINTS: ReadonlyArray<Coord> = [
  { row: 3, col: 3 },
  { row: 3, col: 11 },
  { row: 7, col: 7 },
  { row: 11, col: 3 },
  { row: 11, col: 11 },
];

export function coordKey(row: number, col: number): string {
  return `${row},${col}`;
}

export function playerLabel(player: Player): string {
  return player === BLACK ? '흑' : '백';
}

export function forbiddenLabel(reason: ForbiddenReason): {
  short: string;
  title: string;
} {
  switch (reason) {
    case 'THREE_THREE':
      return { short: '33', title: '쌍삼 (3·3)' };
    case 'FOUR_FOUR':
      return { short: '44', title: '쌍사 (4·4)' };
    case 'OVERLINE':
      return { short: '6+', title: '장목 (6목 이상)' };
  }
}

export function createSvgElement(
  tag: string,
  attrs: Record<string, string | number>,
): SVGElement {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) {
    el.setAttribute(k, String(v));
  }
  return el;
}

export function buildStaticGridSvg(): SVGSVGElement {
  const svg = createSvgElement('svg', {
    class: 'board-svg',
    viewBox: '0 0 1500 1500',
  }) as SVGSVGElement;

  for (let i = 0; i < BOARD_SIZE; i++) {
    const pos = i * 100 + 50;
    const isEdge = i === 0 || i === BOARD_SIZE - 1;
    const strokeWidth = isEdge ? 4 : 2.2;

    svg.appendChild(
      createSvgElement('line', {
        x1: 50,
        y1: pos,
        x2: 1450,
        y2: pos,
        stroke: 'rgba(68, 45, 15, 0.56)',
        'stroke-width': strokeWidth,
      }),
    );
    svg.appendChild(
      createSvgElement('line', {
        x1: pos,
        y1: 50,
        x2: pos,
        y2: 1450,
        stroke: 'rgba(68, 45, 15, 0.56)',
        'stroke-width': strokeWidth,
      }),
    );
  }

  for (const star of STAR_POINTS) {
    svg.appendChild(
      createSvgElement('circle', {
        cx: star.col * 100 + 50,
        cy: star.row * 100 + 50,
        r: 11,
        fill: 'rgba(54, 35, 10, 0.78)',
      }),
    );
  }

  return svg;
}

export function buildEvalChartSvg(
  moves: MoveRecord[],
  cursorIndex: number,
  w: number,
  h: number,
  svgClass: string,
  interactive: boolean,
  onJump?: (index: number) => void,
): SVGSVGElement {
  const total = moves.length;
  const mid = h / 2;
  const svg = createSvgElement('svg', {
    class: svgClass,
    viewBox: `0 0 ${w} ${h}`,
  }) as SVGSVGElement;
  // The small live graph stretches to whatever box it sits in; strokes keep their width.
  if (!interactive) svg.setAttribute('preserveAspectRatio', 'none');

  svg.appendChild(
    createSvgElement('line', {
      x1: 0,
      y1: mid,
      x2: w,
      y2: mid,
      stroke: 'currentColor', // follows the theme text colour (set in CSS)
      'stroke-opacity': 0.25,
      'vector-effect': 'non-scaling-stroke',
      'stroke-width': 1,
      'stroke-dasharray': '3 3',
    }),
  );

  if (total === 0) return svg;

  const clamped = Math.max(0, Math.min(total, cursorIndex));
  const curScore = clamped > 0 ? (moves[clamped - 1]?.evalScore ?? 0) : 0;
  const padX = 4;
  const padY = Math.max(3, Math.round(h * 0.1));
  const xOf = (idx: number) => (idx / Math.max(1, total)) * (w - padX * 2) + padX;
  // Zoom the y-axis to the game's biggest swing (at least ±30%) so an even game isn't a flat line.
  let peak = 0.3;
  for (const m of moves.slice(0, total)) peak = Math.max(peak, Math.abs(m.evalScore ?? 0));
  const yOf = (score: number) => mid - (score / peak) * (mid - padY);

  const pts: Array<[number, number]> = [[xOf(0), yOf(0)]];
  for (let i = 0; i < total; i++) {
    pts.push([xOf(i + 1), yOf(moves[i].evalScore ?? 0)]);
  }

  const polyPoints = pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
  const areaPoints = `${xOf(0).toFixed(1)},${mid} ${polyPoints} ${xOf(total).toFixed(1)},${mid}`;

  svg.appendChild(
    createSvgElement('polygon', {
      points: areaPoints,
      fill: 'currentColor',
      'fill-opacity': 0.12,
    }),
  );
  svg.appendChild(
    createSvgElement('polyline', {
      points: polyPoints,
      fill: 'none',
      stroke: 'currentColor',
      'vector-effect': 'non-scaling-stroke',
      'stroke-width': interactive ? 1.8 : 1.5,
      'stroke-linecap': 'round',
      'stroke-linejoin': 'round',
    }),
  );

  const cx = xOf(clamped);
  const cy = yOf(curScore);
  if (interactive) {
    svg.appendChild(
      createSvgElement('line', {
        x1: cx,
        y1: 2,
        x2: cx,
        y2: h - 2,
        stroke: 'rgba(200, 66, 58, 0.6)', // scarf red
        'stroke-width': 1.2,
      }),
    );
  }
  if (interactive) {
    svg.appendChild(createSvgElement('circle', { cx, cy, r: 3.2, fill: '#c8423a' }));
  }

  if (interactive && onJump) {
    svg.addEventListener('click', (e) => {
      const rect = svg.getBoundingClientRect();
      const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      onJump(Math.round(ratio * total));
    });
  }

  return svg;
}

export function getMoveTagOnBoard(
  board: Board,
  row: number,
  col: number,
  player: Player,
): string | undefined {
  const analysis = analyzeMove(board, row, col, player, RULES);
  if (analysis.isWin) return '승리';
  const totalFours = analysis.openFours.length + analysis.closedFours.length;
  if (totalFours >= 1 && analysis.openThrees.length >= 1) return '4·3';
  if (analysis.openFours.length >= 1) return '열린4';
  if (analysis.closedFours.length >= 1) return '4목';
  if (analysis.openThrees.length >= 1) return '열린3';
  return undefined;
}

export function computeTacticalCandidates(
  board: Board,
  player: Player,
): Array<{ row: number; col: number; prob: number; score: number }> {
  const opp = opponentOf(player);
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
          if (
            nr >= 0 &&
            nr < BOARD_SIZE &&
            nc >= 0 &&
            nc < BOARD_SIZE &&
            board[nr][nc] === EMPTY
          ) {
            marked[nr][nc] = true;
          }
        }
      }
    }
  }
  if (!hasStone) {
    return [{ row: 7, col: 7, prob: 1.0, score: 100 }];
  }

  const raw: Array<{ row: number; col: number; score: number }> = [];
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      if (!marked[r][c]) continue;
      const my = analyzeMove(board, r, c, player, RULES);
      if (!my.legal) continue;
      const op = analyzeMove(board, r, c, opp, RULES);
      const centerBonus = Math.max(0, 14 - (Math.abs(r - 7) + Math.abs(c - 7))) * 8;
      const score =
        (my.isWin ? 1_000_000 : 0) +
        (op.isWin ? 550_000 : 0) +
        my.openFours.length * 100_000 +
        my.closedFours.length * 14_000 +
        my.openThrees.length * 7_500 +
        op.openFours.length * 80_000 +
        op.closedFours.length * 9_000 +
        op.openThrees.length * 5_500 +
        centerBonus;
      raw.push({ row: r, col: c, score });
    }
  }
  raw.sort((a, b) => b.score - a.score);
  const top = raw.slice(0, 8);
  if (top.length === 0) return [];
  const maxS = top[0].score;
  const temp = maxS >= 50_000 ? maxS * 0.2 : Math.max(250, maxS * 0.35);
  const exps = top.map((it) => Math.exp(Math.max(-20, (it.score - maxS) / temp)));
  const z = exps.reduce((a, b) => a + b, 0) || 1;
  return top.map((it, idx) => ({
    row: it.row,
    col: it.col,
    prob: exps[idx] / z,
    score: it.score,
  }));
}

export function formatPlayerThreats(threats: BoardThreat[], player: Player): string {
  const mine = threats.filter((t) => t.player === player);
  if (mine.length === 0) return '없음';
  const openFours = mine.filter((t) => t.type === 'OPEN_FOUR').length;
  const closedFours = mine.filter((t) => t.type === 'CLOSED_FOUR').length;
  const openThrees = mine.filter((t) => t.type === 'OPEN_THREE').length;
  const parts: string[] = [];
  if (openFours > 0) parts.push(`열린4 ${openFours}`);
  if (closedFours > 0) parts.push(`4목 ${closedFours}`);
  if (openThrees > 0) parts.push(`열린3 ${openThrees}`);
  const head =
    mine.find((t) => t.type === 'OPEN_FOUR' || t.type === 'CLOSED_FOUR') ??
    mine[0];
  const target = head.targetCells[0];
  const coord = target ? ` (${formatCoord(target.row, target.col)})` : '';
  return `${parts.join(' · ')}${coord}`;
}
