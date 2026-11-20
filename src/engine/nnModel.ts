import {
  analyzeMove,
  checkWinner,
  formatCoord,
  getForbiddenMoves,
  opponentOf,
} from './renju';
import { BLACK, BOARD_SIZE, Board, Coord, EMPTY, Player } from './types';

export const NUM_CELLS = BOARD_SIZE * BOARD_SIZE;
export const TOTAL_WEIGHT_FLOATS = 61472;
export const TOTAL_WEIGHT_FLOATS_V2 = 166235;
export const TOTAL_WEIGHT_FLOATS_V3 = 373395;

export interface D4ModelConfig {
  version: 1 | 2;
  dRay: number;
  dOuter: number;
  width: number;
  numBlocks: number;
  ovCompDim: number;
  pProjDim: number;
  vHiddenDim: number;
  tHiddenDim: number;
}

const V1_CONFIG: D4ModelConfig = {
  version: 1,
  dRay: 24,
  dOuter: 8,
  width: 64,
  numBlocks: 3,
  ovCompDim: 6,
  pProjDim: 16,
  vHiddenDim: 64,
  tHiddenDim: 0,
};

const V2_CONFIG: D4ModelConfig = {
  version: 2,
  dRay: 32,
  dOuter: 12,
  width: 96,
  numBlocks: 4,
  ovCompDim: 8,
  pProjDim: 24,
  vHiddenDim: 96,
  tHiddenDim: 48,
};

const V3_CONFIG: D4ModelConfig = {
  version: 2,
  dRay: 40,
  dOuter: 16,
  width: 128,
  numBlocks: 6,
  ovCompDim: 8,
  pProjDim: 24,
  vHiddenDim: 96,
  tHiddenDim: 48,
};

const AXES: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [1, 0],
  [1, 1],
  [1, -1],
];

// Precomputed gather indices matching arch-bench/rg_models.py D4OverlapPatternNet
const RAY_STENCIL = buildRayStencil();
const OUTER_STENCIL = buildOuterRayStencil();
const AXIS_OVERLAP_4 = buildAxisOverlap4();
const NB_R2 = buildRadiusTable(2);

function buildRayStencil(): Int32Array {
  // 225 * 8 * 4 (out of bounds -> 225)
  const out = new Int32Array(NUM_CELLS * 8 * 4);
  out.fill(NUM_CELLS);
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      const cell = r * BOARD_SIZE + c;
      for (let a = 0; a < 4; a++) {
        const [dr, dc] = AXES[a];
        for (let sgnI = 0; sgnI < 2; sgnI++) {
          const sgn = sgnI === 0 ? -1 : 1;
          const rayIdx = a * 2 + sgnI;
          for (let step = 1; step <= 4; step++) {
            const nr = r + sgn * step * dr;
            const nc = c + sgn * step * dc;
            if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
              out[(cell * 8 + rayIdx) * 4 + (step - 1)] = nr * BOARD_SIZE + nc;
            }
          }
        }
      }
    }
  }
  return out;
}

function buildOuterRayStencil(): Int32Array {
  // 225 * 8 * 3 for steps 5, 6, 7 (out of bounds -> 225)
  const out = new Int32Array(NUM_CELLS * 8 * 3);
  out.fill(NUM_CELLS);
  const steps = [5, 6, 7];
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      const cell = r * BOARD_SIZE + c;
      for (let a = 0; a < 4; a++) {
        const [dr, dc] = AXES[a];
        for (let sgnI = 0; sgnI < 2; sgnI++) {
          const sgn = sgnI === 0 ? -1 : 1;
          const rayIdx = a * 2 + sgnI;
          for (let si = 0; si < 3; si++) {
            const nr = r + sgn * steps[si] * dr;
            const nc = c + sgn * steps[si] * dc;
            if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
              out[(cell * 8 + rayIdx) * 3 + si] = nr * BOARD_SIZE + nc;
            }
          }
        }
      }
    }
  }
  return out;
}

function buildAxisOverlap4(): Int32Array[] {
  const ds = [-2, -1, 1, 2];
  return AXES.map(([dr, dc]) => {
    const out = new Int32Array(NUM_CELLS * 4);
    out.fill(NUM_CELLS);
    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        const cell = r * BOARD_SIZE + c;
        for (let j = 0; j < 4; j++) {
          const nr = r + ds[j] * dr;
          const nc = c + ds[j] * dc;
          if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
            out[cell * 4 + j] = nr * BOARD_SIZE + nc;
          }
        }
      }
    }
    return out;
  });
}

function buildRadiusTable(radius: number): number[][] {
  const out: number[][] = Array.from({ length: NUM_CELLS }, () => []);
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      const list = out[r * BOARD_SIZE + c];
      for (let dr = -radius; dr <= radius; dr++) {
        for (let dc = -radius; dc <= radius; dc++) {
          if (dr === 0 && dc === 0) continue;
          const nr = r + dr;
          const nc = c + dc;
          if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
            list.push(nr * BOARD_SIZE + nc);
          }
        }
      }
    }
  }
  return out;
}

interface ResBlockWeights {
  fc1B: Float32Array;
  fc1W: Float32Array;
  fc2B: Float32Array;
  fc2W: Float32Array;
  ln1Scale: Float32Array;
  ln1Shift: Float32Array;
  se1B: Float32Array;
  se1W: Float32Array;
  se2B: Float32Array;
  se2W: Float32Array;
}

export interface D4Weights {
  config: D4ModelConfig;
  blocks: ResBlockWeights[];
  headPOutB: Float32Array;
  headPOutW: Float32Array;
  headPProjB: Float32Array;
  headPProjW: Float32Array;
  headTFc1B: Float32Array | null;
  headTFc1W: Float32Array | null;
  headTFc2B: Float32Array | null;
  headTFc2W: Float32Array | null;
  headVFc1B: Float32Array;
  headVFc1W: Float32Array;
  headVFc2B: Float32Array;
  headVFc2W: Float32Array;
  inLnScale: Float32Array;
  inLnShift: Float32Array;
  inProjB: Float32Array;
  inProjW: Float32Array;
  ovCompB: Float32Array;
  ovCompW: Float32Array;
  ovSharedB: Float32Array;
  ovSharedW: Float32Array;
  tableIn: Float32Array;
  tableOut: Float32Array;
  wProdB: Float32Array;
  wProdW: Float32Array;
  wSumB: Float32Array;
  wSumW: Float32Array;
}

export function parseD4Weights(buffer: ArrayBuffer): D4Weights {
  let cfg: D4ModelConfig;
  if (buffer.byteLength === TOTAL_WEIGHT_FLOATS_V3 * 4) {
    cfg = V3_CONFIG;
  } else if (buffer.byteLength === TOTAL_WEIGHT_FLOATS_V2 * 4) {
    cfg = V2_CONFIG;
  } else if (buffer.byteLength === TOTAL_WEIGHT_FLOATS * 4) {
    cfg = V1_CONFIG;
  } else {
    throw new Error(
      `Invalid D4OverlapPatternNet .bin size: expected ${TOTAL_WEIGHT_FLOATS * 4}, ${TOTAL_WEIGHT_FLOATS_V2 * 4}, or ${TOTAL_WEIGHT_FLOATS_V3 * 4} bytes, got ${buffer.byteLength}`,
    );
  }
  const all = new Float32Array(buffer);
  let off = 0;
  const take = (len: number): Float32Array => {
    const view = all.subarray(off, off + len);
    off += len;
    return view;
  };

  const w = cfg.width;
  const halfW = w >> 1;
  const blocks: ResBlockWeights[] = [];
  for (let i = 0; i < cfg.numBlocks; i++) {
    blocks.push({
      fc1B: take(w),
      fc1W: take(w * w),
      fc2B: take(w),
      fc2W: take(w * w),
      ln1Scale: take(w),
      ln1Shift: take(w),
      se1B: take(halfW),
      se1W: take(w * halfW),
      se2B: take(w),
      se2W: take(halfW * w),
    });
  }

  const headPOutB = take(1);
  const headPOutW = take(cfg.pProjDim);
  const headPProjB = take(cfg.pProjDim);
  const headPProjW = take(w * cfg.pProjDim);

  // Alphabetical leaf order: t_fc1, t_fc2 come before v_fc1, v_fc2
  const headTFc1B = cfg.tHiddenDim > 0 ? take(cfg.tHiddenDim) : null;
  const headTFc1W = cfg.tHiddenDim > 0 ? take(2 * w * cfg.tHiddenDim) : null;
  const headTFc2B = cfg.tHiddenDim > 0 ? take(1) : null;
  const headTFc2W = cfg.tHiddenDim > 0 ? take(cfg.tHiddenDim) : null;

  const headVFc1B = take(cfg.vHiddenDim);
  const headVFc1W = take(2 * w * cfg.vHiddenDim);
  const headVFc2B = take(1);
  const headVFc2W = take(cfg.vHiddenDim);

  const dTot = cfg.dRay + cfg.dOuter;
  const symDim = 4 * cfg.ovCompDim;
  const inDim = 3 * cfg.dRay + 6;

  return {
    config: cfg,
    blocks,
    headPOutB,
    headPOutW,
    headPProjB,
    headPProjW,
    headTFc1B,
    headTFc1W,
    headTFc2B,
    headTFc2W,
    headVFc1B,
    headVFc1W,
    headVFc2B,
    headVFc2W,
    inLnScale: take(w),
    inLnShift: take(w),
    inProjB: take(w),
    inProjW: take(inDim * w),
    ovCompB: take(cfg.ovCompDim),
    ovCompW: take(cfg.dRay * cfg.ovCompDim),
    ovSharedB: take(cfg.dRay),
    ovSharedW: take(symDim * cfg.dRay),
    tableIn: take(256 * cfg.dRay),
    tableOut: take(64 * cfg.dOuter),
    wProdB: take(cfg.dRay),
    wProdW: take(dTot * cfg.dRay),
    wSumB: take(cfg.dRay),
    wSumW: take(dTot * cfg.dRay),
  };
}

const SQRT_2_OVER_PI = Math.sqrt(2.0 / Math.PI);

function gelu(x: number): number {
  const x3 = x * x * x;
  return 0.5 * x * (1.0 + Math.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x3)));
}

function sigmoid(x: number): number {
  return 1.0 / (1.0 + Math.exp(-x));
}

function computeLegalMask(board: Board, player: Player): Uint8Array {
  const mask = new Uint8Array(NUM_CELLS);
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      if (board[r][c] === EMPTY) {
        mask[r * BOARD_SIZE + c] = 1;
      }
    }
  }
  if (player === BLACK) {
    for (const f of getForbiddenMoves(board, 'renju')) {
      mask[f.row * BOARD_SIZE + f.col] = 0;
    }
  }
  return mask;
}

export interface D4EvalOutput {
  logits: Float32Array;
  probs: Float32Array;
  value: number;
  predictedPliesLeft: number | null;
  legalMask: Uint8Array;
}

// Reusable scratch buffers sized for V3 max dims (zero-GC single-board evaluation)
const scratchSp = new Int32Array(NUM_CELLS + 1);
const scratchSumIn = new Float32Array(56);
const scratchProdIn = new Float32Array(56);
const scratchHAx = new Float32Array(NUM_CELLS * 4 * 40);
const scratchHComp = new Float32Array(NUM_CELLS * 4 * 8);
const scratchHHat = new Float32Array(NUM_CELLS * 4 * 40);
const scratchSymNb = new Float32Array(32);
const scratchFeats = new Float32Array(126);
const scratchH = new Float32Array(NUM_CELLS * 128);
const scratchLn = new Float32Array(128);
const scratchHid = new Float32Array(128);
const scratchU = new Float32Array(NUM_CELLS * 128);
const scratchGMean = new Float32Array(128);
const scratchGMax = new Float32Array(128);
const scratchSe1 = new Float32Array(64);
const scratchSe2 = new Float32Array(128);
const scratchPh = new Float32Array(24);
const scratchVg = new Float32Array(256);
const scratchVh = new Float32Array(96);
const scratchTh = new Float32Array(48);

function layerNormRow(
  src: Float32Array,
  srcOff: number,
  width: number,
  scale: Float32Array,
  shift: Float32Array,
  out: Float32Array,
): void {
  let mean = 0;
  for (let j = 0; j < width; j++) mean += src[srcOff + j];
  mean /= width;

  let variance = 0;
  for (let j = 0; j < width; j++) {
    const d = src[srcOff + j] - mean;
    variance += d * d;
  }
  const invStd = 1.0 / Math.sqrt(variance / width + 1e-5);
  for (let j = 0; j < width; j++) {
    out[j] = (src[srcOff + j] - mean) * invStd * scale[j] + shift[j];
  }
}

export function evaluateD4OverlapPattern(
  weights: D4Weights,
  board: Board,
  player: Player,
  lastMove: Coord | null,
  legalMaskOverride?: Uint8Array,
): D4EvalOutput {
  const opp = opponentOf(player);
  const legalMask = legalMaskOverride ?? computeLegalMask(board, player);
  const cfg = weights.config;
  const { dRay, dOuter, width: w, ovCompDim, pProjDim, vHiddenDim, tHiddenDim } = cfg;
  const dTot = dRay + dOuter;
  const symDim = 4 * ovCompDim;
  const inDim = 3 * dRay + 6;
  const halfW = w >> 1;

  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      const v = board[r][c];
      scratchSp[r * BOARD_SIZE + c] = v === player ? 1 : v === opp ? 2 : 0;
    }
  }
  scratchSp[NUM_CELLS] = 3;

  const { tableIn, tableOut, wSumW, wSumB, wProdW, wProdB, ovCompW, ovCompB, ovSharedW, ovSharedB } = weights;

  // 1. Symmetric ray embeddings + compressed 4-axis representations
  for (let i = 0; i < NUM_CELLS; i++) {
    for (let a = 0; a < 4; a++) {
      const rL = a * 2;
      const rR = rL + 1;
      const rsL = (i * 8 + rL) * 4;
      const rsR = (i * 8 + rR) * 4;
      const osL = (i * 8 + rL) * 3;
      const osR = (i * 8 + rR) * 3;

      const idInL =
        scratchSp[RAY_STENCIL[rsL]] +
        4 * scratchSp[RAY_STENCIL[rsL + 1]] +
        16 * scratchSp[RAY_STENCIL[rsL + 2]] +
        64 * scratchSp[RAY_STENCIL[rsL + 3]];
      const idInR =
        scratchSp[RAY_STENCIL[rsR]] +
        4 * scratchSp[RAY_STENCIL[rsR + 1]] +
        16 * scratchSp[RAY_STENCIL[rsR + 2]] +
        64 * scratchSp[RAY_STENCIL[rsR + 3]];

      const idOutL =
        scratchSp[OUTER_STENCIL[osL]] +
        4 * scratchSp[OUTER_STENCIL[osL + 1]] +
        16 * scratchSp[OUTER_STENCIL[osL + 2]];
      const idOutR =
        scratchSp[OUTER_STENCIL[osR]] +
        4 * scratchSp[OUTER_STENCIL[osR + 1]] +
        16 * scratchSp[OUTER_STENCIL[osR + 2]];

      const tInLOff = idInL * dRay;
      const tInROff = idInR * dRay;
      for (let d = 0; d < dRay; d++) {
        const l = tableIn[tInLOff + d];
        const r = tableIn[tInROff + d];
        scratchSumIn[d] = l + r;
        scratchProdIn[d] = l * r;
      }

      const tOutLOff = idOutL * dOuter;
      const tOutROff = idOutR * dOuter;
      for (let d = 0; d < dOuter; d++) {
        const l = tableOut[tOutLOff + d];
        const r = tableOut[tOutROff + d];
        scratchSumIn[dRay + d] = l + r;
        scratchProdIn[dRay + d] = l * r;
      }

      const hAxOff = (i * 4 + a) * dRay;
      for (let j = 0; j < dRay; j++) {
        let acc = wSumB[j] + wProdB[j];
        for (let k = 0; k < dTot; k++) {
          const wIdx = k * dRay + j;
          acc += scratchSumIn[k] * wSumW[wIdx] + scratchProdIn[k] * wProdW[wIdx];
        }
        scratchHAx[hAxOff + j] = gelu(acc);
      }

      const hCompOff = (i * 4 + a) * ovCompDim;
      for (let j = 0; j < ovCompDim; j++) {
        let acc = ovCompB[j];
        for (let k = 0; k < dRay; k++) {
          acc += scratchHAx[hAxOff + k] * ovCompW[k * ovCompDim + j];
        }
        scratchHComp[hCompOff + j] = gelu(acc);
      }
    }
  }

  // 2. Reflection-symmetric 1D axis overlap along each axis
  for (let a = 0; a < 4; a++) {
    const axTbl = AXIS_OVERLAP_4[a];
    for (let i = 0; i < NUM_CELLS; i++) {
      const nb0 = axTbl[i * 4];
      const nb1 = axTbl[i * 4 + 1];
      const nb2 = axTbl[i * 4 + 2];
      const nb3 = axTbl[i * 4 + 3];

      const off0 = nb0 < NUM_CELLS ? (nb0 * 4 + a) * ovCompDim : -1;
      const off1 = nb1 < NUM_CELLS ? (nb1 * 4 + a) * ovCompDim : -1;
      const off2 = nb2 < NUM_CELLS ? (nb2 * 4 + a) * ovCompDim : -1;
      const off3 = nb3 < NUM_CELLS ? (nb3 * 4 + a) * ovCompDim : -1;

      for (let k = 0; k < ovCompDim; k++) {
        const v0 = off0 >= 0 ? scratchHComp[off0 + k] : 0;
        const v1 = off1 >= 0 ? scratchHComp[off1 + k] : 0;
        const v2 = off2 >= 0 ? scratchHComp[off2 + k] : 0;
        const v3 = off3 >= 0 ? scratchHComp[off3 + k] : 0;
        scratchSymNb[k] = v1 + v2;
        scratchSymNb[ovCompDim + k] = v1 * v2;
        scratchSymNb[2 * ovCompDim + k] = v0 + v3;
        scratchSymNb[3 * ovCompDim + k] = v0 * v3;
      }

      const hAxOff = (i * 4 + a) * dRay;
      for (let j = 0; j < dRay; j++) {
        let acc = ovSharedB[j];
        for (let k = 0; k < symDim; k++) {
          acc += scratchSymNb[k] * ovSharedW[k * dRay + j];
        }
        scratchHHat[hAxOff + j] = scratchHAx[hAxOff + j] + gelu(acc);
      }
    }
  }

  // 3. Cross-axis D4 features + input projection + LayerNorm + GELU
  const colorPlane = player === BLACK ? 1.0 : 0.0;
  const lastIdx = lastMove ? lastMove.row * BOARD_SIZE + lastMove.col : -1;
  const { inProjW, inProjB, inLnScale, inLnShift } = weights;
  const cellStride = 4 * dRay;

  for (let i = 0; i < NUM_CELLS; i++) {
    const o0 = i * cellStride;
    const o1 = o0 + dRay;
    const o2 = o0 + 2 * dRay;
    const o3 = o0 + 3 * dRay;
    for (let d = 0; d < dRay; d++) {
      const a0 = scratchHHat[o0 + d];
      const a1 = scratchHHat[o1 + d];
      const a2 = scratchHHat[o2 + d];
      const a3 = scratchHHat[o3 + d];
      const hOrtho = a0 + a1;
      const hDiag = a2 + a3;
      scratchFeats[d] = hOrtho + hDiag;
      scratchFeats[dRay + d] = a0 * a1 + a2 * a3;
      scratchFeats[2 * dRay + d] = hOrtho * hDiag;
    }

    const spVal = scratchSp[i];
    const base = 3 * dRay;
    scratchFeats[base] = spVal === 1 ? 1.0 : 0.0;
    scratchFeats[base + 1] = spVal === 2 ? 1.0 : 0.0;
    scratchFeats[base + 2] = i === lastIdx ? 1.0 : 0.0;
    scratchFeats[base + 3] = colorPlane;
    scratchFeats[base + 4] = legalMask[i] ? 1.0 : 0.0;
    scratchFeats[base + 5] = 1.0;

    const hOff = i * w;
    for (let j = 0; j < w; j++) {
      let acc = inProjB[j];
      for (let k = 0; k < inDim; k++) {
        acc += scratchFeats[k] * inProjW[k * w + j];
      }
      scratchH[hOff + j] = acc;
    }
    layerNormRow(scratchH, hOff, w, inLnScale, inLnShift, scratchLn);
    for (let j = 0; j < w; j++) {
      scratchH[hOff + j] = gelu(scratchLn[j]);
    }
  }

  // 4. Residual SE-MLP Blocks
  for (const blk of weights.blocks) {
    scratchGMean.fill(0, 0, w);
    for (let i = 0; i < NUM_CELLS; i++) {
      const hOff = i * w;
      layerNormRow(scratchH, hOff, w, blk.ln1Scale, blk.ln1Shift, scratchLn);

      for (let j = 0; j < w; j++) {
        let acc = blk.fc1B[j];
        for (let k = 0; k < w; k++) {
          acc += scratchLn[k] * blk.fc1W[k * w + j];
        }
        scratchHid[j] = gelu(acc);
      }

      for (let j = 0; j < w; j++) {
        let acc = blk.fc2B[j];
        for (let k = 0; k < w; k++) {
          acc += scratchHid[k] * blk.fc2W[k * w + j];
        }
        scratchU[hOff + j] = acc;
        scratchGMean[j] += acc;
      }
    }

    for (let j = 0; j < w; j++) {
      scratchGMean[j] /= NUM_CELLS;
    }

    for (let j = 0; j < halfW; j++) {
      let acc = blk.se1B[j];
      for (let k = 0; k < w; k++) {
        acc += scratchGMean[k] * blk.se1W[k * halfW + j];
      }
      scratchSe1[j] = gelu(acc);
    }
    for (let j = 0; j < w; j++) {
      let acc = blk.se2B[j];
      for (let k = 0; k < halfW; k++) {
        acc += scratchSe1[k] * blk.se2W[k * w + j];
      }
      scratchSe2[j] = sigmoid(acc);
    }

    for (let i = 0; i < NUM_CELLS; i++) {
      const hOff = i * w;
      for (let j = 0; j < w; j++) {
        scratchH[hOff + j] += scratchU[hOff + j] * scratchSe2[j];
      }
    }
  }

  // 5. Spatial Policy, Value & Turns-to-End Heads
  const logits = new Float32Array(NUM_CELLS);
  const probs = new Float32Array(NUM_CELLS);
  scratchGMean.fill(0, 0, w);
  scratchGMax.fill(-1e9, 0, w);

  const {
    headPProjW,
    headPProjB,
    headPOutW,
    headPOutB,
    headVFc1W,
    headVFc1B,
    headVFc2W,
    headVFc2B,
    headTFc1W,
    headTFc1B,
    headTFc2W,
    headTFc2B,
  } = weights;
  let maxLogit = -Infinity;

  for (let i = 0; i < NUM_CELLS; i++) {
    const hOff = i * w;
    for (let j = 0; j < w; j++) {
      const hv = scratchH[hOff + j];
      scratchGMean[j] += hv;
      if (hv > scratchGMax[j]) scratchGMax[j] = hv;
    }

    if (!legalMask[i]) {
      logits[i] = -1e4;
      continue;
    }

    for (let j = 0; j < pProjDim; j++) {
      let acc = headPProjB[j];
      for (let k = 0; k < w; k++) {
        acc += scratchH[hOff + k] * headPProjW[k * pProjDim + j];
      }
      scratchPh[j] = gelu(acc);
    }
    let l = headPOutB[0];
    for (let k = 0; k < pProjDim; k++) {
      l += scratchPh[k] * headPOutW[k];
    }
    logits[i] = l;
    if (l > maxLogit) maxLogit = l;
  }

  let sumExp = 0;
  for (let i = 0; i < NUM_CELLS; i++) {
    if (legalMask[i]) {
      const e = Math.exp(Math.max(-25, logits[i] - maxLogit));
      probs[i] = e;
      sumExp += e;
    }
  }
  if (sumExp > 0) {
    for (let i = 0; i < NUM_CELLS; i++) {
      probs[i] /= sumExp;
    }
  }

  for (let j = 0; j < w; j++) {
    scratchVg[j] = scratchGMean[j] / NUM_CELLS;
    scratchVg[w + j] = scratchGMax[j];
  }
  const vgDim = 2 * w;
  for (let j = 0; j < vHiddenDim; j++) {
    let acc = headVFc1B[j];
    for (let k = 0; k < vgDim; k++) {
      acc += scratchVg[k] * headVFc1W[k * vHiddenDim + j];
    }
    scratchVh[j] = gelu(acc);
  }
  let vAcc = headVFc2B[0];
  for (let k = 0; k < vHiddenDim; k++) {
    vAcc += scratchVh[k] * headVFc2W[k];
  }

  let predictedPliesLeft: number | null = null;
  if (tHiddenDim > 0 && headTFc1B && headTFc1W && headTFc2B && headTFc2W) {
    for (let j = 0; j < tHiddenDim; j++) {
      let acc = headTFc1B[j];
      for (let k = 0; k < vgDim; k++) {
        acc += scratchVg[k] * headTFc1W[k * tHiddenDim + j];
      }
      scratchTh[j] = gelu(acc);
    }
    let tAcc = headTFc2B[0];
    for (let k = 0; k < tHiddenDim; k++) {
      tAcc += scratchTh[k] * headTFc2W[k];
    }
    predictedPliesLeft = Math.max(1, Math.round(sigmoid(tAcc) * 60));
  }

  return { logits, probs, value: Math.tanh(vAcc), predictedPliesLeft, legalMask };
}

// ── Browser PUCT MCTS + Tactical Solver ──

interface BrowserMctsNode {
  player: Player;
  expanded: boolean;
  terminal: boolean;
  terminalVal: number;
  winMove: number;
  nSum: number;
  actions: number[];
  priors: number[];
  n: number[];
  w: number[];
  children: Array<BrowserMctsNode | null>;
}

function createNode(player: Player): BrowserMctsNode {
  return {
    player,
    expanded: false,
    terminal: false,
    terminalVal: 0,
    winMove: -1,
    nSum: 0,
    actions: [],
    priors: [],
    n: [],
    w: [],
    children: [],
  };
}

function getLocalEmptyIndices(board: Board): number[] {
  const marked = new Uint8Array(NUM_CELLS);
  let hasStone = false;
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      if (board[r][c] === EMPTY) continue;
      hasStone = true;
      for (const nb of NB_R2[r * BOARD_SIZE + c]) {
        const nr = (nb / BOARD_SIZE) | 0;
        const nc = nb % BOARD_SIZE;
        if (board[nr][nc] === EMPTY) marked[nb] = 1;
      }
    }
  }
  if (!hasStone) return [7 * BOARD_SIZE + 7];
  const out: number[] = [];
  for (let i = 0; i < NUM_CELLS; i++) {
    if (marked[i]) out.push(i);
  }
  return out;
}

function expandBrowserNode(
  node: BrowserMctsNode,
  board: Board,
  lastMove: Coord | null,
  weights: D4Weights,
  maxChildren: number,
): { value: number; predictedPliesLeft: number | null } {
  const pl = node.player;
  const opp = opponentOf(pl);
  const local = getLocalEmptyIndices(board);

  // Immediate win check
  for (const idx of local) {
    const r = (idx / BOARD_SIZE) | 0;
    const c = idx % BOARD_SIZE;
    board[r][c] = pl;
    const won = checkWinner(board, { row: r, col: c }, 'renju');
    board[r][c] = EMPTY;
    if (won) {
      node.winMove = idx;
      node.actions = [idx];
      node.priors = [1.0];
      node.n = [0];
      node.w = [0];
      node.children = [null];
      node.expanded = true;
      return { value: 1.0, predictedPliesLeft: 1 };
    }
  }

  const evalOut = evaluateD4OverlapPattern(weights, board, pl, lastMove);

  // Check opponent immediate wins (forced blocks)
  let hasOppWin = false;
  const oppWins: number[] = [];
  for (const idx of local) {
    const r = (idx / BOARD_SIZE) | 0;
    const c = idx % BOARD_SIZE;
    board[r][c] = opp;
    const won = checkWinner(board, { row: r, col: c }, 'renju');
    board[r][c] = EMPTY;
    if (won) {
      hasOppWin = true;
      if (evalOut.legalMask[idx]) oppWins.push(idx);
    }
  }

  let candList: number[];
  if (hasOppWin) {
    if (oppWins.length === 0) {
      node.terminal = true;
      node.terminalVal = -1.0;
      node.expanded = true;
      return { value: -1.0, predictedPliesLeft: 1 };
    }
    candList = oppWins;
  } else {
    candList = local.filter((idx) => evalOut.legalMask[idx] === 1);
    if (candList.length === 0) {
      for (let i = 0; i < NUM_CELLS; i++) {
        if (evalOut.legalMask[i]) candList.push(i);
      }
    }
  }

  if (candList.length === 0) {
    node.terminal = true;
    node.terminalVal = -1.0;
    node.expanded = true;
    return { value: -1.0, predictedPliesLeft: 1 };
  }

  candList.sort((a, b) => evalOut.logits[b] - evalOut.logits[a]);
  const kept = candList.slice(0, maxChildren);
  const maxL = evalOut.logits[kept[0]];
  const exps = kept.map((idx) => Math.exp(Math.max(-25, evalOut.logits[idx] - maxL)));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;

  node.actions = kept;
  node.priors = exps.map((e) => e / sum);
  node.n = new Array<number>(kept.length).fill(0);
  node.w = new Array<number>(kept.length).fill(0);
  node.children = new Array<BrowserMctsNode | null>(kept.length).fill(null);
  node.expanded = true;
  return { value: evalOut.value, predictedPliesLeft: evalOut.predictedPliesLeft };
}

export interface CheckpointMoveResult {
  row: number;
  col: number;
  value: number;
  topCandidates: Array<{ row: number; col: number; prob: number; coord: string }>;
  source: 'tactical' | 'mcts';
  provenWinPlies?: number | null;
  predictedPliesLeft?: number | null;
  simsUsed?: number;
}

interface PreparedMcts {
  immediate?: CheckpointMoveResult | null;
  clone: Board;
  root: BrowserMctsNode;
  rootOut: ReturnType<typeof expandBrowserNode>;
  targetSims: number;
  targetBudgetMs: number;
}

function prepareMctsContext(
  board: Board,
  player: Player,
  weights: D4Weights,
  lastMove: Coord | null,
  nSims: number,
  timeBudgetMs: number,
): PreparedMcts | null {
  const clone: Board = board.map((row) => [...row]);
  const opp = opponentOf(player);
  const legalMask = computeLegalMask(clone, player);
  const legalIndices: number[] = [];
  for (let i = 0; i < NUM_CELLS; i++) {
    if (legalMask[i]) legalIndices.push(i);
  }
  if (legalIndices.length === 0) return null;

  for (const idx of legalIndices) {
    const r = (idx / BOARD_SIZE) | 0;
    const c = idx % BOARD_SIZE;
    if (analyzeMove(clone, r, c, player, 'renju').isWin) {
      return {
        immediate: {
          row: r,
          col: c,
          value: 1.0,
          topCandidates: [{ row: r, col: c, prob: 1.0, coord: formatCoord(r, c) }],
          source: 'tactical',
          provenWinPlies: 1,
        },
        clone,
        root: createNode(player),
        rootOut: { value: 1, predictedPliesLeft: 1 },
        targetSims: 0,
        targetBudgetMs: 0,
      };
    }
  }

  for (const idx of legalIndices) {
    const r = (idx / BOARD_SIZE) | 0;
    const c = idx % BOARD_SIZE;
    clone[r][c] = opp;
    const oppWon = checkWinner(clone, { row: r, col: c }, 'renju');
    clone[r][c] = EMPTY;
    if (oppWon) {
      const ev = evaluateD4OverlapPattern(weights, clone, player, lastMove, legalMask);
      return {
        immediate: {
          row: r,
          col: c,
          value: ev.value,
          topCandidates: [{ row: r, col: c, prob: 1.0, coord: formatCoord(r, c) }],
          source: 'tactical',
          provenWinPlies: null,
        },
        clone,
        root: createNode(player),
        rootOut: ev,
        targetSims: 0,
        targetBudgetMs: 0,
      };
    }
  }

  const root = createNode(player);
  const rootOut = expandBrowserNode(root, clone, lastMove, weights, 28);
  if (root.actions.length === 0) return null;

  const rootTop1 = root.priors[0] ?? 0;
  const rootUncertain = 1 - Math.abs(rootOut.value);
  let targetSims = nSims;
  let targetBudgetMs = timeBudgetMs;
  if (nSims >= 120) {
    if (rootTop1 >= 0.72 && Math.abs(rootOut.value) >= 0.70) {
      targetSims = 48;
    } else if (rootUncertain >= 0.55 || rootTop1 <= 0.28) {
      targetSims = Math.max(nSims, 220);
      targetBudgetMs = Math.max(timeBudgetMs, 950);
    }
  }
  return { clone, root, rootOut, targetSims, targetBudgetMs };
}

function canEarlyExitMcts(root: BrowserMctsNode, sim: number, targetSims: number): boolean {
  if (sim < 32 || root.actions.length < 2 || sim % 8 !== 0) return false;
  let max1 = -1;
  let max2 = -1;
  for (let i = 0; i < root.actions.length; i++) {
    const ni = root.n[i];
    if (ni > max1) {
      max2 = max1;
      max1 = ni;
    } else if (ni > max2) {
      max2 = ni;
    }
  }
  return max1 - max2 > targetSims - sim;
}

function runOneMctsSim(
  root: BrowserMctsNode,
  clone: Board,
  lastMove: Coord | null,
  weights: D4Weights,
  cPuct = 1.5,
): void {
  let node = root;
  let curLast = lastMove;
  const path: Array<{ parent: BrowserMctsNode; aIdx: number; r: number; c: number }> = [];

  while (node.expanded && !node.terminal && node.actions.length > 0) {
    const sqrtTot = Math.sqrt(Math.max(1, node.nSum));
    const fpu = node === root ? -0.05 : -0.25;
    let bestIdx = 0;
    let bestScore = -Infinity;
    for (let i = 0; i < node.actions.length; i++) {
      const ni = node.n[i];
      const q = ni > 0 ? node.w[i] / ni : fpu;
      const u = cPuct * node.priors[i] * (sqrtTot / (1 + ni));
      if (q + u > bestScore) {
        bestScore = q + u;
        bestIdx = i;
      }
    }

    const mv = node.actions[bestIdx];
    const mr = (mv / BOARD_SIZE) | 0;
    const mc = mv % BOARD_SIZE;
    const mover = node.player;
    clone[mr][mc] = mover;
    curLast = { row: mr, col: mc };
    path.push({ parent: node, aIdx: bestIdx, r: mr, c: mc });

    let child = node.children[bestIdx];
    if (!child) {
      child = createNode(opponentOf(mover));
      if (checkWinner(clone, curLast, 'renju')) {
        child.terminal = true;
        child.terminalVal = -1.0;
        child.expanded = true;
      }
      node.children[bestIdx] = child;
    }
    node = child;
  }

  let leafVal: number;
  if (node.terminal) {
    leafVal = node.terminalVal;
  } else if (!node.expanded) {
    leafVal = expandBrowserNode(node, clone, curLast, weights, 20).value;
  } else {
    leafVal = 0;
  }

  let v = leafVal;
  for (let i = path.length - 1; i >= 0; i--) {
    const step = path[i];
    v = -v;
    step.parent.n[step.aIdx] += 1;
    step.parent.w[step.aIdx] += v;
    step.parent.nSum += 1;
    clone[step.r][step.c] = EMPTY;
  }
}

function finalizeMctsResult(
  root: BrowserMctsNode,
  rootOut: ReturnType<typeof expandBrowserNode>,
  player: Player,
  simsExecuted: number,
): CheckpointMoveResult {
  let bestA = 0;
  let bestVisits = -1;
  let wSum = 0;
  for (let i = 0; i < root.actions.length; i++) {
    wSum += root.w[i];
    if (
      root.n[i] > bestVisits ||
      (root.n[i] === bestVisits && root.priors[i] > root.priors[bestA])
    ) {
      bestVisits = root.n[i];
      bestA = i;
    }
  }

  const totalN = Math.max(1, root.nSum);
  const ranked = root.actions
    .map((idx, i) => {
      const r = (idx / BOARD_SIZE) | 0;
      const c = idx % BOARD_SIZE;
      return {
        row: r,
        col: c,
        prob: root.nSum > 0 ? root.n[i] / totalN : root.priors[i],
        coord: formatCoord(r, c),
      };
    })
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 5);

  const chosen = root.actions[bestA];
  const provenWinPlies = findProvenWinPlies(root.children[bestA], player, 0, 12);
  return {
    row: (chosen / BOARD_SIZE) | 0,
    col: chosen % BOARD_SIZE,
    value: root.nSum > 0 ? Math.max(-1, Math.min(1, wSum / totalN)) : 0,
    topCandidates: ranked,
    source: 'mcts',
    provenWinPlies: provenWinPlies !== null ? provenWinPlies + 1 : null,
    predictedPliesLeft: rootOut.predictedPliesLeft,
    simsUsed: simsExecuted,
  };
}

export function chooseCheckpointMove(
  board: Board,
  player: Player,
  weights: D4Weights,
  lastMove: Coord | null,
  nSims = 160,
  timeBudgetMs = 800,
): CheckpointMoveResult | null {
  const prep = prepareMctsContext(board, player, weights, lastMove, nSims, timeBudgetMs);
  if (!prep) return null;
  if (prep.immediate !== undefined) return prep.immediate;

  const deadline = performance.now() + prep.targetBudgetMs;
  let simsExecuted = 0;
  for (let sim = 0; sim < prep.targetSims; sim++) {
    if (sim >= 24 && performance.now() >= deadline) break;
    if (canEarlyExitMcts(prep.root, sim, prep.targetSims)) break;
    simsExecuted++;
    runOneMctsSim(prep.root, prep.clone, lastMove, weights);
  }
  return finalizeMctsResult(prep.root, prep.rootOut, player, simsExecuted);
}

export async function chooseCheckpointMoveAsync(
  board: Board,
  player: Player,
  weights: D4Weights,
  lastMove: Coord | null,
  nSims = 160,
  timeBudgetMs = 950,
  onProgress?: (pct: number, simsDone: number, simsTarget: number) => void,
  isCancelled?: () => boolean,
): Promise<CheckpointMoveResult | null> {
  const prep = prepareMctsContext(board, player, weights, lastMove, nSims, timeBudgetMs);
  if (!prep) return null;
  if (prep.immediate !== undefined) {
    onProgress?.(100, 1, 1);
    return prep.immediate;
  }

  onProgress?.(5, 0, prep.targetSims);
  const deadline = performance.now() + prep.targetBudgetMs;
  let simsExecuted = 0;
  let sim = 0;
  while (sim < prep.targetSims) {
    if (isCancelled?.()) return null;
    const chunkStart = performance.now();
    let chunkCount = 0;
    while (sim < prep.targetSims && chunkCount < 4) {
      if (sim >= 24 && performance.now() >= deadline) break;
      if (canEarlyExitMcts(prep.root, sim, prep.targetSims)) {
        sim = prep.targetSims;
        break;
      }
      simsExecuted++;
      runOneMctsSim(prep.root, prep.clone, lastMove, weights);
      sim++;
      chunkCount++;
      if (performance.now() - chunkStart >= 14) break;
    }
    if (sim >= 24 && performance.now() >= deadline) break;
    const pct = Math.min(99, Math.round((sim / Math.max(1, prep.targetSims)) * 100));
    onProgress?.(pct, simsExecuted, prep.targetSims);
    if (sim < prep.targetSims) {
      await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
    }
  }
  onProgress?.(100, simsExecuted, prep.targetSims);
  return finalizeMctsResult(prep.root, prep.rootOut, player, simsExecuted);
}

function findProvenWinPlies(
  node: BrowserMctsNode | null | undefined,
  attacker: Player,
  depth: number,
  maxDepth: number,
): number | null {
  if (!node || depth > maxDepth) return null;
  if (node.terminal) {
    // terminalVal < 0 means previous mover (opponent of node.player) just completed 5
    return node.terminalVal < 0 && node.player !== attacker ? 0 : null;
  }
  if (!node.expanded || node.actions.length === 0) return null;

  if (node.player === attacker) {
    let best: number | null = null;
    for (const ch of node.children) {
      const p = findProvenWinPlies(ch, attacker, depth + 1, maxDepth);
      if (p !== null && (best === null || p + 1 < best)) {
        best = p + 1;
      }
    }
    return best;
  }

  // Defender turn: must be proven winning across ALL expanded legal replies (and all actions expanded)
  let worst = 0;
  for (let i = 0; i < node.actions.length; i++) {
    const ch = node.children[i];
    const p = findProvenWinPlies(ch, attacker, depth + 1, maxDepth);
    if (p === null) return null;
    if (p + 1 > worst) worst = p + 1;
  }
  return worst;
}
