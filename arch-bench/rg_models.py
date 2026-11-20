"""Five neural network architectures for 15x15 Renju Gomoku built on rustAutograd (rg)."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import rg
from rg import nn, random, tree

BOARD_SIZE = 15
NUM_CELLS = BOARD_SIZE * BOARD_SIZE
IN_CHANNELS = 6


def build_stencil_indices(offsets: List[Tuple[int, int]]) -> np.ndarray:
    """Builds flat gather index array of shape (225 * len(offsets),) into a (B, 226, C) tensor
    where index 225 is the out-of-bounds zero padding row.
    """
    k = len(offsets)
    idx = np.full((NUM_CELLS, k), NUM_CELLS, dtype=np.int32)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            cell = r * BOARD_SIZE + c
            for j, (dr, dc) in enumerate(offsets):
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    idx[cell, j] = nr * BOARD_SIZE + nc
    return idx.reshape(-1)


IDX_3X3 = build_stencil_indices(
    [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
)

AXIS_INDICES = [
    build_stencil_indices([(d * dr, d * dc) for d in range(-3, 4)])
    for (dr, dc) in ((0, 1), (1, 0), (1, 1), (1, -1))
]


def build_axial_line_features() -> Tuple[np.ndarray, np.ndarray]:
    """Builds (line_mask [225, 225] float32, pair_feats [225, 225, 7] float32)
    encoding whether two cells share one of the 4 Gomoku axes and their distance.
    """
    mask = np.zeros((NUM_CELLS, NUM_CELLS), dtype=np.float32)
    feats = np.zeros((NUM_CELLS, NUM_CELLS, 7), dtype=np.float32)
    for i in range(NUM_CELLS):
        r1, c1 = divmod(i, BOARD_SIZE)
        for j in range(NUM_CELLS):
            r2, c2 = divmod(j, BOARD_SIZE)
            dr, dc = r2 - r1, c2 - c1
            on_row = (dr == 0)
            on_col = (dc == 0)
            on_d1 = (dr == dc)
            on_d2 = (dr == -dc)
            if on_row or on_col or on_d1 or on_d2:
                mask[i, j] = 1.0
                dist = max(abs(dr), abs(dc))
                feats[i, j, 0] = 1.0 if on_row else 0.0
                feats[i, j, 1] = 1.0 if on_col else 0.0
                feats[i, j, 2] = 1.0 if on_d1 else 0.0
                feats[i, j, 3] = 1.0 if on_d2 else 0.0
                feats[i, j, 4] = 1.0 if dist == 0 else 0.0
                feats[i, j, 5] = 1.0 / (1.0 + dist)
                feats[i, j, 6] = 1.0 if 1 <= dist <= 4 else 0.0
    return mask, feats


AXIAL_MASK_NP, AXIAL_FEATS_NP = build_axial_line_features()


def gather_stencil(x, idx_tensor, k: int):
    """x: (B, 225, C) -> (B, 225, k * C) using zero padding at index 225."""
    b, _, c = x.shape
    xz = rg.concatenate([x, rg.zeros((b, 1, c), x.dtype)], axis=1)
    return rg.take(xz, idx_tensor, axis=1).reshape(b, NUM_CELLS, k * c)


def policy_value_heads_spatial(p_head, x, legal_mask):
    """Spatial 1x1 policy head + global pool value head.
    x: (B, 225, C), legal_mask: (B, 225) float32 0/1.
    """
    b = x.shape[0]
    ph = nn.gelu(p_head["p_proj"](p_head["p_proj_p"], x))
    logits = p_head["p_out"](p_head["p_out_p"], ph).squeeze(-1)  # (B, 225)
    logits = rg.where(legal_mask > 0.5, logits, -1e4)

    g_mean = x.mean(axis=1)
    g_max = x.max(axis=1)
    vg = rg.concatenate([g_mean, g_max], axis=-1)
    vh = nn.gelu(p_head["v_fc1"](p_head["v_fc1_p"], vg))
    val = rg.tanh(p_head["v_fc2"](p_head["v_fc2_p"], vh)).squeeze(-1)  # (B,)
    return logits, val


class SpatialHead:
    def __init__(self, d: int):
        self.p_proj = nn.Linear(d, 16)
        self.p_out = nn.Linear(16, 1)
        self.v_fc1 = nn.Linear(2 * d, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def init(self, key):
        k1, k2, k3, k4 = random.split(key, 4)
        return {
            "p_proj": self.p_proj.init(k1),
            "p_out": self.p_out.init(k2),
            "v_fc1": self.v_fc1.init(k3),
            "v_fc2": self.v_fc2.init(k4),
        }

    def __call__(self, p, x, legal_mask):
        ph = nn.gelu(self.p_proj(p["p_proj"], x))
        logits = self.p_out(p["p_out"], ph).squeeze(-1)
        logits = rg.where(legal_mask > 0.5, logits, -1e4)

        vg = rg.concatenate([x.mean(axis=1), x.max(axis=1)], axis=-1)
        vh = nn.gelu(self.v_fc1(p["v_fc1"], vg))
        val = rg.tanh(self.v_fc2(p["v_fc2"], vh)).squeeze(-1)
        return logits, val


# ---------------------------------------------------------------------------
# 1. FlatMLP (Baseline: flattened board tensor into Residual MLP)
# ---------------------------------------------------------------------------
class FlatMLPNet:
    name = "FlatMLP"

    def __init__(self, width: int = 160, blocks: int = 3):
        self.width = width
        self.blocks = blocks
        self.in_proj = nn.Linear(NUM_CELLS * IN_CHANNELS, width)
        self.in_norm = nn.LayerNorm(width)
        self.fcs = [(nn.Linear(width, width), nn.LayerNorm(width)) for _ in range(blocks)]
        self.p_head = nn.Linear(width, NUM_CELLS)
        self.v_fc1 = nn.Linear(width, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def init(self, key):
        ks = random.split(key, 4 + self.blocks)
        return {
            "in_proj": self.in_proj.init(ks[0]),
            "in_norm": self.in_norm.init(),
            "blocks": [
                {"fc": fc.init(ks[1 + i]), "ln": ln.init()}
                for i, (fc, ln) in enumerate(self.fcs)
            ],
            "p_head": self.p_head.init(ks[1 + self.blocks]),
            "v_fc1": self.v_fc1.init(ks[2 + self.blocks]),
            "v_fc2": self.v_fc2.init(ks[3 + self.blocks]),
        }

    def __call__(self, p, x):
        # x: (B, 225, 6)
        b = x.shape[0]
        legal_mask = x[:, :, 4]
        h = nn.gelu(self.in_norm(p["in_norm"], self.in_proj(p["in_proj"], x.reshape(b, NUM_CELLS * IN_CHANNELS))))
        for i, (fc, ln) in enumerate(self.fcs):
            bp = p["blocks"][i]
            h = h + nn.gelu(ln(bp["ln"], fc(bp["fc"], h)))
        logits = self.p_head(p["p_head"], h)
        logits = rg.where(legal_mask > 0.5, logits, -1e4)
        vh = nn.gelu(self.v_fc1(p["v_fc1"], h))
        val = rg.tanh(self.v_fc2(p["v_fc2"], vh)).squeeze(-1)
        return logits, val


# ---------------------------------------------------------------------------
# 2. ResNet3x3 (Current engine-rs style 3x3 Conv ResNet)
# ---------------------------------------------------------------------------
class ResNet3x3Net:
    name = "ResNet3x3"

    def __init__(self, width: int = 32, blocks: int = 4, use_se: bool = False):
        self.width = width
        self.blocks = blocks
        self.use_se = use_se
        if use_se:
            self.name = "SEResNet3x3"
        self.idx_3x3 = rg.tensor(IDX_3X3)
        self.stem = nn.Linear(9 * IN_CHANNELS, width)
        self.stem_ln = nn.LayerNorm(width)
        self.res = []
        for _ in range(blocks):
            c1 = nn.Linear(9 * width, width)
            ln1 = nn.LayerNorm(width)
            c2 = nn.Linear(9 * width, width)
            ln2 = nn.LayerNorm(width)
            se1 = nn.Linear(width, width // 2) if use_se else None
            se2 = nn.Linear(width // 2, width) if use_se else None
            self.res.append((c1, ln1, c2, ln2, se1, se2))
        self.head = SpatialHead(width)

    def init(self, key):
        ks = random.split(key, 2 + self.blocks * 4)
        ki = 0
        bp_list = []
        for c1, ln1, c2, ln2, se1, se2 in self.res:
            d = {
                "c1": c1.init(ks[ki]),
                "ln1": ln1.init(),
                "c2": c2.init(ks[ki + 1]),
                "ln2": ln2.init(),
            }
            if self.use_se:
                d["se1"] = se1.init(ks[ki + 2])
                d["se2"] = se2.init(ks[ki + 3])
            ki += 4
            bp_list.append(d)
        return {
            "stem": self.stem.init(ks[ki]),
            "stem_ln": self.stem_ln.init(),
            "blocks": bp_list,
            "head": self.head.init(ks[ki + 1]),
        }

    def __call__(self, p, x):
        legal_mask = x[:, :, 4]
        h = nn.gelu(self.stem_ln(p["stem_ln"], self.stem(p["stem"], gather_stencil(x, self.idx_3x3, 9))))
        for i, (c1, ln1, c2, ln2, se1, se2) in enumerate(self.res):
            bp = p["blocks"][i]
            u = nn.gelu(ln1(bp["ln1"], c1(bp["c1"], gather_stencil(h, self.idx_3x3, 9))))
            u = ln2(bp["ln2"], c2(bp["c2"], gather_stencil(u, self.idx_3x3, 9)))
            if self.use_se:
                g = h.mean(axis=1, keepdims=True) + u.mean(axis=1, keepdims=True)
                s = nn.sigmoid(se2(bp["se2"], nn.gelu(se1(bp["se1"], g))))
                u = u * s
            h = nn.gelu(h + u)
        return self.head(p["head"], h, legal_mask)


# ---------------------------------------------------------------------------
# 3. LineConvNet (Omok-specific 4-Axis 7-tap 1D Line Convolutions + SE)
# ---------------------------------------------------------------------------
class LineConvNet:
    name = "LineConv4Axis"

    def __init__(self, width: int = 36, line_ch: int = 12, blocks: int = 4):
        self.width = width
        self.line_ch = line_ch
        self.blocks = blocks
        self.axis_idx = [rg.tensor(arr) for arr in AXIS_INDICES]

        # Stem: 4-axis 7-tap line conv right on the raw board!
        self.stem_axes = [nn.Linear(7 * IN_CHANNELS, line_ch) for _ in range(4)]
        self.stem_mix = nn.Linear(4 * line_ch, width)
        self.stem_ln = nn.LayerNorm(width)

        self.res = []
        for _ in range(blocks):
            in_proj = nn.Linear(width, line_ch)
            axes_conv = [nn.Linear(7 * line_ch, line_ch) for _ in range(4)]
            mix = nn.Linear(4 * line_ch, width)
            ln = nn.LayerNorm(width)
            se1 = nn.Linear(width, width // 2)
            se2 = nn.Linear(width // 2, width)
            self.res.append((in_proj, axes_conv, mix, ln, se1, se2))
        self.head = SpatialHead(width)

    def init(self, key):
        ks = random.split(key, 6 + self.blocks * 8)
        ki = 0
        stem_ax_p = []
        for ax in self.stem_axes:
            stem_ax_p.append(ax.init(ks[ki]))
            ki += 1
        stem_mix_p = self.stem_mix.init(ks[ki])
        ki += 1

        bp_list = []
        for in_proj, axes_conv, mix, ln, se1, se2 in self.res:
            ax_p = []
            for ax in axes_conv:
                ax_p.append(ax.init(ks[ki]))
                ki += 1
            bp_list.append({
                "in_proj": in_proj.init(ks[ki]),
                "axes": ax_p,
                "mix": mix.init(ks[ki + 1]),
                "ln": ln.init(),
                "se1": se1.init(ks[ki + 2]),
                "se2": se2.init(ks[ki + 3]),
            })
            ki += 4

        return {
            "stem_axes": stem_ax_p,
            "stem_mix": stem_mix_p,
            "stem_ln": self.stem_ln.init(),
            "blocks": bp_list,
            "head": self.head.init(ks[ki]),
        }

    def __call__(self, p, x):
        legal_mask = x[:, :, 4]
        ax_feats = [
            nn.gelu(ax(p["stem_axes"][a], gather_stencil(x, self.axis_idx[a], 7)))
            for a, ax in enumerate(self.stem_axes)
        ]
        h = nn.gelu(self.stem_ln(p["stem_ln"], self.stem_mix(p["stem_mix"], rg.concatenate(ax_feats, axis=-1))))

        for i, (in_proj, axes_conv, mix, ln, se1, se2) in enumerate(self.res):
            bp = p["blocks"][i]
            z = nn.gelu(in_proj(bp["in_proj"], ln(bp["ln"], h)))
            b_ax = [
                nn.gelu(ax(bp["axes"][a], gather_stencil(z, self.axis_idx[a], 7)))
                for a, ax in enumerate(axes_conv)
            ]
            u = mix(bp["mix"], rg.concatenate(b_ax, axis=-1))
            g = u.mean(axis=1, keepdims=True)
            s = nn.sigmoid(se2(bp["se2"], nn.gelu(se1(bp["se1"], g))))
            h = h + u * s
        return self.head(p["head"], h, legal_mask)


# ---------------------------------------------------------------------------
# 4. AxialOmokNet (Omok-specific 4-Axis Line Attention + Local LineConv Stem)
# ---------------------------------------------------------------------------
class AxialOmokNet:
    name = "AxialOmokNet"

    def __init__(self, width: int = 32, heads: int = 4, blocks: int = 3):
        self.width = width
        self.heads = heads
        self.blocks = blocks
        self.hd = width // heads
        self.axis_idx = [rg.tensor(arr) for arr in AXIS_INDICES]
        self.axial_mask = rg.tensor(AXIAL_MASK_NP)   # (225, 225)
        self.axial_feats = rg.tensor(AXIAL_FEATS_NP) # (225, 225, 7)

        # Stem: 4-axis local line pattern encoder
        self.stem_axes = [nn.Linear(7 * IN_CHANNELS, 8) for _ in range(4)]
        self.stem_mix = nn.Linear(32, width)
        self.stem_ln = nn.LayerNorm(width)

        self.attn_blocks = []
        for _ in range(blocks):
            ln1 = nn.LayerNorm(width)
            qkv = nn.Linear(width, 3 * width)
            rel_bias = nn.Linear(7, heads)
            proj = nn.Linear(width, width)
            ln2 = nn.LayerNorm(width)
            fc1 = nn.Linear(width, width * 2)
            fc2 = nn.Linear(width * 2, width)
            self.attn_blocks.append((ln1, qkv, rel_bias, proj, ln2, fc1, fc2))

        self.head = SpatialHead(width)

    def init(self, key):
        ks = random.split(key, 6 + self.blocks * 6)
        ki = 0
        stem_ax_p = []
        for ax in self.stem_axes:
            stem_ax_p.append(ax.init(ks[ki]))
            ki += 1
        stem_mix_p = self.stem_mix.init(ks[ki])
        ki += 1

        bp_list = []
        for ln1, qkv, rel_bias, proj, ln2, fc1, fc2 in self.attn_blocks:
            bp_list.append({
                "ln1": ln1.init(),
                "qkv": qkv.init(ks[ki]),
                "rel_bias": rel_bias.init(ks[ki + 1]),
                "proj": proj.init(ks[ki + 2]),
                "ln2": ln2.init(),
                "fc1": fc1.init(ks[ki + 3]),
                "fc2": fc2.init(ks[ki + 4]),
            })
            ki += 5

        return {
            "stem_axes": stem_ax_p,
            "stem_mix": stem_mix_p,
            "stem_ln": self.stem_ln.init(),
            "blocks": bp_list,
            "head": self.head.init(ks[ki]),
        }

    def __call__(self, p, x):
        b = x.shape[0]
        legal_mask = x[:, :, 4]
        ax_feats = [
            nn.gelu(ax(p["stem_axes"][a], gather_stencil(x, self.axis_idx[a], 7)))
            for a, ax in enumerate(self.stem_axes)
        ]
        h = nn.gelu(self.stem_ln(p["stem_ln"], self.stem_mix(p["stem_mix"], rg.concatenate(ax_feats, axis=-1))))

        mask_bt = self.axial_mask.reshape(1, 1, NUM_CELLS, NUM_CELLS) > 0.5
        for i, (ln1, qkv, rel_bias, proj, ln2, fc1, fc2) in enumerate(self.attn_blocks):
            bp = p["blocks"][i]
            hn = ln1(bp["ln1"], h)
            qkv_t = qkv(bp["qkv"], hn).reshape(b, NUM_CELLS, 3, self.heads, self.hd).transpose(2, 0, 3, 1, 4)
            q, k, v = qkv_t[0], qkv_t[1], qkv_t[2]
            # Pairwise 4-axis structural bias: (225, 225, heads) -> (1, heads, 225, 225)
            rb = rel_bias(bp["rel_bias"], self.axial_feats).transpose(2, 0, 1).reshape(1, self.heads, NUM_CELLS, NUM_CELLS)
            scores = (q @ rg.swapaxes(k, -1, -2)) / math.sqrt(self.hd) + rb
            scores = rg.where(mask_bt, scores, -1e4)
            attn = nn.softmax(scores)
            ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(b, NUM_CELLS, self.width)
            h = h + proj(bp["proj"], ctx)
            h = h + fc2(bp["fc2"], nn.gelu(fc1(bp["fc1"], ln2(bp["ln2"], h))))

        return self.head(p["head"], h, legal_mask)


# ---------------------------------------------------------------------------
# 5. FastLine1x5Net (Speed-optimized 4-Axis 1x5 Depthwise-Basis + SE/CrossGate)
# ---------------------------------------------------------------------------
def build_1x5_basis() -> np.ndarray:
    """(4, 5, 225 * 225) shift-basis matrices for 4 directions x 5 taps (-2..+2)."""
    basis = np.zeros((4, 5, NUM_CELLS * NUM_CELLS), dtype=np.float32)
    axes = ((0, 1), (1, 0), (1, 1), (1, -1))
    for a, (dr, dc) in enumerate(axes):
        for oi, d in enumerate(range(-2, 3)):
            mat = np.zeros((NUM_CELLS, NUM_CELLS), dtype=np.float32)
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    nr, nc = r + d * dr, c + d * dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        mat[nr * BOARD_SIZE + nc, r * BOARD_SIZE + c] = 1.0
            basis[a, oi] = mat.reshape(-1)
    return basis


BASIS_1X5_NP = build_1x5_basis()


class FastLine1x5Net:
    name = "FastLine1x5"

    def __init__(self, width: int = 48, groups_per_axis: int = 3, blocks: int = 4, axial_top: int = 0):
        assert width % (4 * groups_per_axis) == 0
        self.width = width
        self.gpa = groups_per_axis
        self.ngroups = 4 * groups_per_axis
        self.gch = width // self.ngroups
        self.blocks = blocks
        self.axial_top = axial_top
        if axial_top > 0:
            self.name = f"Line1x5_Ax{axial_top}"

        self.basis = rg.tensor(BASIS_1X5_NP)  # (4, 5, 50625)
        self.stem = nn.Linear(IN_CHANNELS, width)
        self.stem_ln = nn.LayerNorm(width)

        self.blocks_spec = list(range(blocks))
        self.head = SpatialHead(width)

        if axial_top > 0:
            self.heads = 4
            self.hd = width // 4
            self.axial_mask = rg.tensor(AXIAL_MASK_NP)
            self.axial_feats = rg.tensor(AXIAL_FEATS_NP)
            self.ax_ln1 = [nn.LayerNorm(width) for _ in range(axial_top)]
            self.ax_qkv = [nn.Linear(width, 3 * width) for _ in range(axial_top)]
            self.ax_rb = [nn.Linear(7, 4) for _ in range(axial_top)]
            self.ax_proj = [nn.Linear(width, width) for _ in range(axial_top)]
            self.ax_ln2 = [nn.LayerNorm(width) for _ in range(axial_top)]
            self.ax_fc1 = [nn.Linear(width, width * 2) for _ in range(axial_top)]
            self.ax_fc2 = [nn.Linear(width * 2, width) for _ in range(axial_top)]

    def init(self, key):
        ks = random.split(key, 4 + self.blocks * 5 + self.axial_top * 6)
        ki = 0
        s = math.sqrt(2.0 / self.width)
        blks = []
        for _ in range(self.blocks):
            blks.append({
                "kw": random.normal(ks[ki], (4, self.gpa, 5)) * 0.25,
                "pw1": random.normal(ks[ki + 1], (self.width, self.width)) * s,
                "b1": rg.zeros((self.width, 1)),
                "pw2": random.normal(ks[ki + 2], (self.width, self.width)) * s,
                "b2": rg.zeros((self.width, 1)),
                "se1": random.normal(ks[ki + 3], (self.width // 2, self.width)) * s,
                "se2": random.normal(ks[ki + 4], (self.width, self.width // 2)) * s,
            })
            ki += 5

        ax_blks = []
        for t in range(self.axial_top):
            ax_blks.append({
                "ln1": self.ax_ln1[t].init(),
                "qkv": self.ax_qkv[t].init(ks[ki]),
                "rb": self.ax_rb[t].init(ks[ki + 1]),
                "proj": self.ax_proj[t].init(ks[ki + 2]),
                "ln2": self.ax_ln2[t].init(),
                "fc1": self.ax_fc1[t].init(ks[ki + 3]),
                "fc2": self.ax_fc2[t].init(ks[ki + 4]),
            })
            ki += 5

        return {
            "stem": self.stem.init(ks[ki]),
            "stem_ln": self.stem_ln.init(),
            "stem_kw": random.normal(ks[ki + 1], (4, self.gpa, 5)) * 0.25,
            "blocks": blks,
            "ax": ax_blks,
            "head": self.head.init(ks[ki + 2]),
        }

    def _apply_1x5(self, h_bc225, kw):
        # h_bc225: (B, W, 225)
        b = h_bc225.shape[0]
        k_all = (kw @ self.basis).reshape(self.ngroups, NUM_CELLS, NUM_CELLS)
        ug = h_bc225.reshape(b, self.ngroups, self.gch, NUM_CELLS).transpose(1, 0, 2, 3).reshape(
            self.ngroups, b * self.gch, NUM_CELLS
        )
        yg = (ug @ k_all).reshape(self.ngroups, b, self.gch, NUM_CELLS).transpose(1, 0, 2, 3).reshape(
            b, self.width, NUM_CELLS
        )
        return yg

    def __call__(self, p, x):
        b = x.shape[0]
        legal_mask = x[:, :, 4]
        # Stem: project (B, 225, 6) -> (B, 225, W) -> transpose (B, W, 225) + 1x5 conv
        h0 = nn.gelu(self.stem_ln(p["stem_ln"], self.stem(p["stem"], x))).transpose(0, 2, 1)
        h = self._apply_1x5(h0, p["stem_kw"])

        for bp in p["blocks"]:
            u = nn.gelu(bp["pw1"] @ h + bp["b1"])
            y = self._apply_1x5(u, bp["kw"])
            out = nn.gelu(bp["pw2"] @ y + bp["b2"])
            # Channel SE gate (global mean over 225 cells)
            g = out.mean(axis=2, keepdims=True)  # (B, W, 1)
            gate = nn.sigmoid(bp["se2"] @ nn.gelu(bp["se1"] @ g))
            h = h + out * gate

        h_tok = h.transpose(0, 2, 1)  # (B, 225, W)

        if self.axial_top > 0:
            mask_bt = self.axial_mask.reshape(1, 1, NUM_CELLS, NUM_CELLS) > 0.5
            for t in range(self.axial_top):
                ab = p["ax"][t]
                hn = self.ax_ln1[t](ab["ln1"], h_tok)
                qkv_t = self.ax_qkv[t](ab["qkv"], hn).reshape(b, NUM_CELLS, 3, self.heads, self.hd).transpose(2, 0, 3, 1, 4)
                q, k, v = qkv_t[0], qkv_t[1], qkv_t[2]
                rb = self.ax_rb[t](ab["rb"], self.axial_feats).transpose(2, 0, 1).reshape(1, self.heads, NUM_CELLS, NUM_CELLS)
                scores = (q @ rg.swapaxes(k, -1, -2)) / math.sqrt(self.hd) + rb
                scores = rg.where(mask_bt, scores, -1e4)
                attn = nn.softmax(scores)
                ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(b, NUM_CELLS, self.width)
                h_tok = h_tok + self.ax_proj[t](ab["proj"], ctx)
                h_tok = h_tok + self.ax_fc2[t](ab["fc2"], nn.gelu(self.ax_fc1[t](ab["fc1"], self.ax_ln2[t](ab["ln2"], h_tok))))

        return self.head(p["head"], h_tok, legal_mask)


# ---------------------------------------------------------------------------
# 6. LineAxialHybridNet (4-Axis 16ch Stem + LineConv SE Blocks + Signed-Rich Axial)
# ---------------------------------------------------------------------------
def build_rich_axial_feats() -> np.ndarray:
    """Builds (225, 225, 14) structural relationship features along the 4 Gomoku axes:
    [on_row, on_col, on_d1, on_d2, sgn_pos, sgn_neg, d==0..5, d>=6, 1/(1+d)].
    """
    feats = np.zeros((NUM_CELLS, NUM_CELLS, 14), dtype=np.float32)
    for i in range(NUM_CELLS):
        r1, c1 = divmod(i, BOARD_SIZE)
        for j in range(NUM_CELLS):
            r2, c2 = divmod(j, BOARD_SIZE)
            dr, dc = r2 - r1, c2 - c1
            on_row = (dr == 0)
            on_col = (dc == 0)
            on_d1 = (dr == dc)
            on_d2 = (dr == -dc)
            if on_row or on_col or on_d1 or on_d2:
                dist = max(abs(dr), abs(dc))
                sgn = dc if on_row else dr
                feats[i, j, 0] = 1.0 if on_row else 0.0
                feats[i, j, 1] = 1.0 if on_col else 0.0
                feats[i, j, 2] = 1.0 if on_d1 else 0.0
                feats[i, j, 3] = 1.0 if on_d2 else 0.0
                feats[i, j, 4] = 1.0 if sgn > 0 else 0.0
                feats[i, j, 5] = 1.0 if sgn < 0 else 0.0
                for dk in range(6):
                    if dist == dk:
                        feats[i, j, 6 + dk] = 1.0
                if dist >= 6:
                    feats[i, j, 12] = 1.0
                feats[i, j, 13] = 1.0 / (1.0 + dist)
    return feats


RICH_AXIAL_FEATS_NP = build_rich_axial_feats()


class LineAxialHybridNet:
    name = "LineAxialHybrid"

    def __init__(
        self,
        width: int = 48,
        stem_ch: int = 16,
        line_blocks: int = 2,
        axial_blocks: int = 1,
        heads: int = 4,
    ):
        self.width = width
        self.stem_ch = stem_ch
        self.line_blocks = line_blocks
        self.axial_blocks = axial_blocks
        self.heads = heads
        self.hd = width // heads
        self.axis_idx = [rg.tensor(arr) for arr in AXIS_INDICES]
        self.axial_mask = rg.tensor(AXIAL_MASK_NP)
        self.axial_feats = rg.tensor(RICH_AXIAL_FEATS_NP)

        self.stem_axes = [nn.Linear(7 * IN_CHANNELS, stem_ch) for _ in range(4)]
        self.stem_mix = nn.Linear(4 * stem_ch, width)
        self.stem_ln = nn.LayerNorm(width)

        self.l_blocks = []
        for _ in range(line_blocks):
            in_p = nn.Linear(width, stem_ch)
            axes_c = [nn.Linear(7 * stem_ch, stem_ch) for _ in range(4)]
            mix = nn.Linear(4 * stem_ch, width)
            ln = nn.LayerNorm(width)
            se1 = nn.Linear(width, width // 2)
            se2 = nn.Linear(width // 2, width)
            self.l_blocks.append((in_p, axes_c, mix, ln, se1, se2))

        self.a_blocks = []
        for _ in range(axial_blocks):
            ln1 = nn.LayerNorm(width)
            qkv = nn.Linear(width, 3 * width)
            rel_bias = nn.Linear(14, heads)
            proj = nn.Linear(width, width)
            ln2 = nn.LayerNorm(width)
            fc1 = nn.Linear(width, width * 2)
            fc2 = nn.Linear(width * 2, width)
            self.a_blocks.append((ln1, qkv, rel_bias, proj, ln2, fc1, fc2))

        self.head = SpatialHead(width)

    def init(self, key):
        ks = random.split(key, 6 + self.line_blocks * 8 + self.axial_blocks * 6)
        ki = 0
        stem_ax_p = [ax.init(ks[ki + a]) for a, ax in enumerate(self.stem_axes)]
        ki += 4
        stem_mix_p = self.stem_mix.init(ks[ki])
        ki += 1

        lb_list = []
        for in_p, axes_c, mix, ln, se1, se2 in self.l_blocks:
            ax_p = [ax.init(ks[ki + a]) for a, ax in enumerate(axes_c)]
            ki += 4
            lb_list.append({
                "in_proj": in_p.init(ks[ki]),
                "axes": ax_p,
                "mix": mix.init(ks[ki + 1]),
                "ln": ln.init(),
                "se1": se1.init(ks[ki + 2]),
                "se2": se2.init(ks[ki + 3]),
            })
            ki += 4

        ab_list = []
        for ln1, qkv, rel_bias, proj, ln2, fc1, fc2 in self.a_blocks:
            ab_list.append({
                "ln1": ln1.init(),
                "qkv": qkv.init(ks[ki]),
                "rel_bias": rel_bias.init(ks[ki + 1]),
                "proj": proj.init(ks[ki + 2]),
                "ln2": ln2.init(),
                "fc1": fc1.init(ks[ki + 3]),
                "fc2": fc2.init(ks[ki + 4]),
            })
            ki += 5

        return {
            "stem_axes": stem_ax_p,
            "stem_mix": stem_mix_p,
            "stem_ln": self.stem_ln.init(),
            "l_blocks": lb_list,
            "a_blocks": ab_list,
            "head": self.head.init(ks[ki]),
        }

    def __call__(self, p, x):
        b = x.shape[0]
        legal_mask = x[:, :, 4]
        ax_feats = [
            nn.gelu(ax(p["stem_axes"][a], gather_stencil(x, self.axis_idx[a], 7)))
            for a, ax in enumerate(self.stem_axes)
        ]
        h = nn.gelu(self.stem_ln(p["stem_ln"], self.stem_mix(p["stem_mix"], rg.concatenate(ax_feats, axis=-1))))

        for i, (in_p, axes_c, mix, ln, se1, se2) in enumerate(self.l_blocks):
            bp = p["l_blocks"][i]
            z = nn.gelu(in_p(bp["in_proj"], ln(bp["ln"], h)))
            b_ax = [
                nn.gelu(ax(bp["axes"][a], gather_stencil(z, self.axis_idx[a], 7)))
                for a, ax in enumerate(axes_c)
            ]
            u = mix(bp["mix"], rg.concatenate(b_ax, axis=-1))
            g = u.mean(axis=1, keepdims=True)
            s = nn.sigmoid(se2(bp["se2"], nn.gelu(se1(bp["se1"], g))))
            h = h + u * s

        mask_bt = self.axial_mask.reshape(1, 1, NUM_CELLS, NUM_CELLS) > 0.5
        for i, (ln1, qkv, rel_bias, proj, ln2, fc1, fc2) in enumerate(self.a_blocks):
            bp = p["a_blocks"][i]
            hn = ln1(bp["ln1"], h)
            qkv_t = qkv(bp["qkv"], hn).reshape(b, NUM_CELLS, 3, self.heads, self.hd).transpose(2, 0, 3, 1, 4)
            q, k, v = qkv_t[0], qkv_t[1], qkv_t[2]
            rb = rel_bias(bp["rel_bias"], self.axial_feats).transpose(2, 0, 1).reshape(1, self.heads, NUM_CELLS, NUM_CELLS)
            scores = (q @ rg.swapaxes(k, -1, -2)) / math.sqrt(self.hd) + rb
            scores = rg.where(mask_bt, scores, -1e4)
            attn = nn.softmax(scores)
            ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(b, NUM_CELLS, self.width)
            h = h + proj(bp["proj"], ctx)
            h = h + fc2(bp["fc2"], nn.gelu(fc1(bp["fc1"], ln2(bp["ln2"], h))))

        return self.head(p["head"], h, legal_mask)


# ---------------------------------------------------------------------------
# 7. PatternTableNet (4-Axis Symmetric Ray Pattern Table + Cross-Pair Fork NNUE)
# ---------------------------------------------------------------------------
def build_ray_stencil() -> np.ndarray:
    """Builds flat gather indices (225 * 8 * 4,) for 8 rays (+/- along 4 axes), steps 1..4."""
    stencil = np.full((NUM_CELLS, 8, 4), NUM_CELLS, dtype=np.int32)
    axes = ((0, 1), (1, 0), (1, 1), (1, -1))
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            cell = r * BOARD_SIZE + c
            for a, (dr, dc) in enumerate(axes):
                for sgn_i, sgn in enumerate((-1, 1)):
                    ray_idx = a * 2 + sgn_i
                    for step in range(1, 5):
                        nr, nc = r + sgn * step * dr, c + sgn * step * dc
                        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                            stencil[cell, ray_idx, step - 1] = nr * BOARD_SIZE + nc
    return stencil.reshape(-1)


RAY_STENCIL_FLAT_NP = build_ray_stencil()
POW4_NP = np.array([1.0, 4.0, 16.0, 64.0], dtype=np.float32)


class PatternTableNet:
    name = "PatternTableNNUE"

    def __init__(self, d_ray: int = 24, width: int = 64, blocks: int = 3):
        self.d_ray = d_ray
        self.width = width
        self.blocks = blocks
        self.ray_stencil = rg.tensor(RAY_STENCIL_FLAT_NP)
        self.pow4 = rg.tensor(POW4_NP)

        self.w_sum = nn.Linear(d_ray, d_ray)
        self.w_prod = nn.Linear(d_ray, d_ray)
        self.in_proj = nn.Linear(2 * d_ray + IN_CHANNELS, width)
        self.in_ln = nn.LayerNorm(width)

        self.mlp_blocks = []
        for _ in range(blocks):
            fc1 = nn.Linear(width, width)
            ln1 = nn.LayerNorm(width)
            fc2 = nn.Linear(width, width)
            se1 = nn.Linear(width, width // 2)
            se2 = nn.Linear(width // 2, width)
            self.mlp_blocks.append((fc1, ln1, fc2, se1, se2))

        self.head = SpatialHead(width)

    def init(self, key):
        ks = random.split(key, 6 + self.blocks * 4)
        ki = 0
        table = random.normal(ks[ki], (256, self.d_ray)) * 0.15
        ki += 1
        ws = self.w_sum.init(ks[ki])
        wp = self.w_prod.init(ks[ki + 1])
        ip = self.in_proj.init(ks[ki + 2])
        ki += 3

        blks = []
        for fc1, ln1, fc2, se1, se2 in self.mlp_blocks:
            blks.append({
                "fc1": fc1.init(ks[ki]),
                "ln1": ln1.init(),
                "fc2": fc2.init(ks[ki + 1]),
                "se1": se1.init(ks[ki + 2]),
                "se2": se2.init(ks[ki + 3]),
            })
            ki += 4

        return {
            "table": table,
            "w_sum": ws,
            "w_prod": wp,
            "in_proj": ip,
            "in_ln": self.in_ln.init(),
            "blocks": blks,
            "head": self.head.init(ks[ki]),
        }

    def __call__(self, p, x):
        b = x.shape[0]
        legal_mask = x[:, :, 4]

        # Cell state in {0=empty, 1=mine, 2=opp, 3=wall(at idx 225)}
        s0 = x[:, :, 0] + 2.0 * x[:, :, 1]
        sp = rg.concatenate([s0, rg.full((b, 1), 3.0, x.dtype)], axis=1)
        g_ray = rg.take(sp, self.ray_stencil, axis=1).reshape(b, NUM_CELLS * 8, 4)
        ray_ids = rg.stop_gradient((g_ray @ self.pow4).cast("int32"))

        emb = rg.take(p["table"], ray_ids, axis=0).reshape(b, NUM_CELLS, 4, 2, self.d_ray)
        left_e = emb[:, :, :, 0, :]
        right_e = emb[:, :, :, 1, :]

        # Reflection-symmetric line feature along each axis
        h_ax = nn.gelu(self.w_sum(p["w_sum"], left_e + right_e) + self.w_prod(p["w_prod"], left_e * right_e))

        # Single-axis threat sum & exact 6-pair cross-axis fork interaction
        s1 = h_ax.sum(axis=2)
        s2 = (h_ax * h_ax).sum(axis=2)
        cross_pairs = 0.5 * (s1 * s1 - s2)

        cat = rg.concatenate([s1, cross_pairs, x], axis=-1)
        h = nn.gelu(self.in_ln(p["in_ln"], self.in_proj(p["in_proj"], cat)))

        for i, (fc1, ln1, fc2, se1, se2) in enumerate(self.mlp_blocks):
            bp = p["blocks"][i]
            u = fc2(bp["fc2"], nn.gelu(fc1(bp["fc1"], ln1(bp["ln1"], h))))
            g = u.mean(axis=1, keepdims=True)
            s = nn.sigmoid(se2(bp["se2"], nn.gelu(se1(bp["se1"], g))))
            h = h + u * s

        return self.head(p["head"], h, legal_mask)


# ---------------------------------------------------------------------------
# 8. DualPathOmokNet (Stream A: PatternTableNNUE | Stream B: 3x3 + Open 2D Attn)
# ---------------------------------------------------------------------------
def build_open_2d_feats() -> np.ndarray:
    """Builds (225, 225, 16) unmasked 2D geometric relationship features:
    [on_row, on_col, on_d1, on_d2, sgn_pos, sgn_neg, is_knight, off_axis_near,
     cheb==0..5, cheb>=6, 1/(1+euclid)].
    """
    feats = np.zeros((NUM_CELLS, NUM_CELLS, 16), dtype=np.float32)
    for i in range(NUM_CELLS):
        r1, c1 = divmod(i, BOARD_SIZE)
        for j in range(NUM_CELLS):
            r2, c2 = divmod(j, BOARD_SIZE)
            dr, dc = r2 - r1, c2 - c1
            adr, adc = abs(dr), abs(dc)
            cheb = max(adr, adc)
            euclid = math.sqrt(dr * dr + dc * dc)
            on_row = (dr == 0)
            on_col = (dc == 0)
            on_d1 = (dr == dc)
            on_d2 = (dr == -dc)
            on_any = on_row or on_col or on_d1 or on_d2
            sgn = (dc if on_row else dr) if on_any else (dr + dc)
            is_knight = (adr == 1 and adc == 2) or (adr == 2 and adc == 1)
            feats[i, j, 0] = 1.0 if on_row else 0.0
            feats[i, j, 1] = 1.0 if on_col else 0.0
            feats[i, j, 2] = 1.0 if on_d1 else 0.0
            feats[i, j, 3] = 1.0 if on_d2 else 0.0
            feats[i, j, 4] = 1.0 if sgn > 0 else 0.0
            feats[i, j, 5] = 1.0 if sgn < 0 else 0.0
            feats[i, j, 6] = 1.0 if is_knight else 0.0
            feats[i, j, 7] = 1.0 if (not on_any and cheb <= 3) else 0.0
            for dk in range(6):
                if cheb == dk:
                    feats[i, j, 8 + dk] = 1.0
            if cheb >= 6:
                feats[i, j, 14] = 1.0
            feats[i, j, 15] = 1.0 / (1.0 + euclid)
    return feats


OPEN_2D_FEATS_NP = build_open_2d_feats()


class DualPathOmokNet:
    name = "DualPathOmok"

    def __init__(
        self,
        d_ray: int = 24,
        width: int = 48,
        mlp_blocks: int = 2,
        heads: int = 4,
    ):
        self.d_ray = d_ray
        self.width = width
        self.mlp_blocks = mlp_blocks
        self.heads = heads
        self.hd = width // heads

        self.ray_stencil = rg.tensor(RAY_STENCIL_FLAT_NP)
        self.pow4 = rg.tensor(POW4_NP)
        self.idx_3x3 = rg.tensor(IDX_3X3)
        self.open_feats = rg.tensor(OPEN_2D_FEATS_NP)

        # Stream A: Fast PatternTableNNUE Tactical Highway
        self.w_sum = nn.Linear(d_ray, d_ray)
        self.w_prod = nn.Linear(d_ray, d_ray)
        self.in_proj = nn.Linear(2 * d_ray + IN_CHANNELS, width)
        self.in_ln = nn.LayerNorm(width)
        self.tac_blocks = [
            (
                nn.Linear(width, width),
                nn.LayerNorm(width),
                nn.Linear(width, width),
                nn.Linear(width, width // 2),
                nn.Linear(width // 2, width),
            )
            for _ in range(mlp_blocks)
        ]

        # Stream B: Off-Axis 3x3 Conv + Unmasked Full-Board Open 2D Attention
        self.conv_reduce = nn.Linear(width, width // 2)
        self.conv3x3 = nn.Linear(9 * (width // 2 + IN_CHANNELS), width)
        self.conv_ln = nn.LayerNorm(width)

        self.attn_ln1 = nn.LayerNorm(width)
        self.attn_qkv = nn.Linear(width, 3 * width)
        self.attn_rb = nn.Linear(16, heads)
        self.attn_proj = nn.Linear(width, width)
        self.attn_ln2 = nn.LayerNorm(width)
        self.attn_fc1 = nn.Linear(width, width * 2)
        self.attn_fc2 = nn.Linear(width * 2, width)

        # Dual-Path Pre-Norm Gated Residual Fusion
        self.fuse_ln = nn.LayerNorm(width)
        self.fuse_gate = nn.Linear(2 * width, width)
        self.fuse_proj = nn.Linear(width, width)

        self.head = SpatialHead(width)

    def init(self, key):
        ks = random.split(key, 18 + self.mlp_blocks * 4)
        ki = 0
        table = random.normal(ks[ki], (256, self.d_ray)) * 0.15
        ki += 1
        ws = self.w_sum.init(ks[ki])
        wp = self.w_prod.init(ks[ki + 1])
        ip = self.in_proj.init(ks[ki + 2])
        ki += 3

        t_blks = []
        for fc1, ln1, fc2, se1, se2 in self.tac_blocks:
            t_blks.append({
                "fc1": fc1.init(ks[ki]),
                "ln1": ln1.init(),
                "fc2": fc2.init(ks[ki + 1]),
                "se1": se1.init(ks[ki + 2]),
                "se2": se2.init(ks[ki + 3]),
            })
            ki += 4

        return {
            "table": table,
            "w_sum": ws,
            "w_prod": wp,
            "in_proj": ip,
            "in_ln": self.in_ln.init(),
            "tac_blocks": t_blks,
            "conv_reduce": self.conv_reduce.init(ks[ki]),
            "conv3x3": self.conv3x3.init(ks[ki + 1]),
            "conv_ln": self.conv_ln.init(),
            "attn_ln1": self.attn_ln1.init(),
            "attn_qkv": self.attn_qkv.init(ks[ki + 2]),
            "attn_rb": self.attn_rb.init(ks[ki + 3]),
            "attn_proj": self.attn_proj.init(ks[ki + 4]),
            "attn_ln2": self.attn_ln2.init(),
            "attn_fc1": self.attn_fc1.init(ks[ki + 5]),
            "attn_fc2": self.attn_fc2.init(ks[ki + 6]),
            "fuse_ln": self.fuse_ln.init(),
            "fuse_gate": self.fuse_gate.init(ks[ki + 7]),
            "fuse_proj": self.fuse_proj.init(ks[ki + 8]),
            "head": self.head.init(ks[ki + 9]),
        }

    def __call__(self, p, x):
        b = x.shape[0]
        legal_mask = x[:, :, 4]

        # Stream A: PatternTable + Symmetric Ray Fusion + Cross-Axis Fork
        s0 = x[:, :, 0] + 2.0 * x[:, :, 1]
        sp = rg.concatenate([s0, rg.full((b, 1), 3.0, x.dtype)], axis=1)
        g_ray = rg.take(sp, self.ray_stencil, axis=1).reshape(b, NUM_CELLS * 8, 4)
        ray_ids = rg.stop_gradient((g_ray @ self.pow4).cast("int32"))

        emb = rg.take(p["table"], ray_ids, axis=0).reshape(b, NUM_CELLS, 4, 2, self.d_ray)
        left_e = emb[:, :, :, 0, :]
        right_e = emb[:, :, :, 1, :]
        h_ax = nn.gelu(self.w_sum(p["w_sum"], left_e + right_e) + self.w_prod(p["w_prod"], left_e * right_e))
        s1 = h_ax.sum(axis=2)
        s2 = (h_ax * h_ax).sum(axis=2)
        cross_pairs = 0.5 * (s1 * s1 - s2)

        h0 = nn.gelu(self.in_ln(p["in_ln"], self.in_proj(p["in_proj"], rg.concatenate([s1, cross_pairs, x], axis=-1))))

        h_tac = h0
        for i, (fc1, ln1, fc2, se1, se2) in enumerate(self.tac_blocks):
            bp = p["tac_blocks"][i]
            u = fc2(bp["fc2"], nn.gelu(fc1(bp["fc1"], ln1(bp["ln1"], h_tac))))
            g = u.mean(axis=1, keepdims=True)
            s = nn.sigmoid(se2(bp["se2"], nn.gelu(se1(bp["se1"], g))))
            h_tac = h_tac + u * s

        # Stream B: Off-Axis 3x3 Conv + Unmasked Full-Board Open 2D Attention
        h_sub = nn.gelu(self.conv_reduce(p["conv_reduce"], h0))
        c_in = rg.concatenate([h_sub, x], axis=-1)
        h_glob = nn.gelu(self.conv_ln(p["conv_ln"], self.conv3x3(p["conv3x3"], gather_stencil(c_in, self.idx_3x3, 9))))

        hn = self.attn_ln1(p["attn_ln1"], h_glob)
        qkv_t = self.attn_qkv(p["attn_qkv"], hn).reshape(b, NUM_CELLS, 3, self.heads, self.hd).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv_t[0], qkv_t[1], qkv_t[2]
        rb = self.attn_rb(p["attn_rb"], self.open_feats).transpose(2, 0, 1).reshape(1, self.heads, NUM_CELLS, NUM_CELLS)
        scores = (q @ rg.swapaxes(k, -1, -2)) / math.sqrt(self.hd) + rb
        attn = nn.softmax(scores)
        ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(b, NUM_CELLS, self.width)
        h_glob = h_glob + self.attn_proj(p["attn_proj"], ctx)
        h_glob = h_glob + self.attn_fc2(p["attn_fc2"], nn.gelu(self.attn_fc1(p["attn_fc1"], self.attn_ln2(p["attn_ln2"], h_glob))))

        # Dual-Path Pre-Norm Gated Residual Fusion
        h_glob_n = self.fuse_ln(p["fuse_ln"], h_glob)
        gate = nn.sigmoid(self.fuse_gate(p["fuse_gate"], rg.concatenate([h_tac, h_glob_n], axis=-1)))
        h_fused = h_tac + gate * self.fuse_proj(p["fuse_proj"], h_glob_n)

        return self.head(p["head"], h_fused, legal_mask)


# ---------------------------------------------------------------------------
# 9. BlurConvPatternNet & OverlapPatternNet (Peripheral + Line-Overlap NNUE)
# ---------------------------------------------------------------------------
BLUR_OFFSETS = [
    (dr * 2, dc * 2) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)
] + [
    (dr * 4, dc * 4) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)
]
IDX_BLUR16 = build_stencil_indices(BLUR_OFFSETS)

AXIS_OVERLAP_4 = [
    build_stencil_indices([(d * dr, d * dc) for d in (-2, -1, 1, 2)])
    for (dr, dc) in ((0, 1), (1, 0), (1, 1), (1, -1))
]


def build_outer_ray_stencil() -> np.ndarray:
    """Builds flat gather indices (225 * 8 * 3,) for steps 5..7 along 8 rays."""
    stencil = np.full((NUM_CELLS, 8, 3), NUM_CELLS, dtype=np.int32)
    axes = ((0, 1), (1, 0), (1, 1), (1, -1))
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            cell = r * BOARD_SIZE + c
            for a, (dr, dc) in enumerate(axes):
                for sgn_i, sgn in enumerate((-1, 1)):
                    ray_idx = a * 2 + sgn_i
                    for step_i, step in enumerate((5, 6, 7)):
                        nr, nc = r + sgn * step * dr, c + sgn * step * dc
                        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                            stencil[cell, ray_idx, step_i] = nr * BOARD_SIZE + nc
    return stencil.reshape(-1)


OUTER_STENCIL_FLAT_NP = build_outer_ray_stencil()
POW4_3_NP = np.array([1.0, 4.0, 16.0], dtype=np.float32)


class BlurConvPatternNet:
    """Sharp 4-step PatternTable center + 16-point coarse peripheral Conv (radius 2 & 4)."""
    name = "BlurPattern"

    def __init__(self, d_ray: int = 24, width: int = 64, blocks: int = 3):
        self.d_ray = d_ray
        self.width = width
        self.blocks = blocks
        self.ray_stencil = rg.tensor(RAY_STENCIL_FLAT_NP)
        self.pow4 = rg.tensor(POW4_NP)
        self.idx_blur = rg.tensor(IDX_BLUR16)

        self.w_sum = nn.Linear(d_ray, d_ray)
        self.w_prod = nn.Linear(d_ray, d_ray)
        self.blur_conv = nn.Linear(16 * 3, 16)
        self.in_proj = nn.Linear(2 * d_ray + 16 + IN_CHANNELS, width)
        self.in_ln = nn.LayerNorm(width)
        self.mlp_blocks = [
            (
                nn.Linear(width, width),
                nn.LayerNorm(width),
                nn.Linear(width, width),
                nn.Linear(width, width // 2),
                nn.Linear(width // 2, width),
            )
            for _ in range(blocks)
        ]
        self.head = SpatialHead(width)

    def init(self, key):
        ks = random.split(key, 8 + self.blocks * 4)
        ki = 0
        table = random.normal(ks[ki], (256, self.d_ray)) * 0.15
        ki += 1
        ws = self.w_sum.init(ks[ki])
        wp = self.w_prod.init(ks[ki + 1])
        bc = self.blur_conv.init(ks[ki + 2])
        ip = self.in_proj.init(ks[ki + 3])
        ki += 4

        blks = []
        for fc1, ln1, fc2, se1, se2 in self.mlp_blocks:
            blks.append({
                "fc1": fc1.init(ks[ki]),
                "ln1": ln1.init(),
                "fc2": fc2.init(ks[ki + 1]),
                "se1": se1.init(ks[ki + 2]),
                "se2": se2.init(ks[ki + 3]),
            })
            ki += 4

        return {
            "table": table,
            "w_sum": ws,
            "w_prod": wp,
            "blur": bc,
            "in_proj": ip,
            "in_ln": self.in_ln.init(),
            "blocks": blks,
            "head": self.head.init(ks[ki]),
        }

    def __call__(self, p, x):
        b = x.shape[0]
        legal_mask = x[:, :, 4]
        s0 = x[:, :, 0] + 2.0 * x[:, :, 1]
        sp = rg.concatenate([s0, rg.full((b, 1), 3.0, x.dtype)], axis=1)
        g_ray = rg.take(sp, self.ray_stencil, axis=1).reshape(b, NUM_CELLS * 8, 4)
        ray_ids = rg.stop_gradient((g_ray @ self.pow4).cast("int32"))

        emb = rg.take(p["table"], ray_ids, axis=0).reshape(b, NUM_CELLS, 4, 2, self.d_ray)
        left_e, right_e = emb[:, :, :, 0, :], emb[:, :, :, 1, :]
        h_ax = nn.gelu(self.w_sum(p["w_sum"], left_e + right_e) + self.w_prod(p["w_prod"], left_e * right_e))
        s1 = h_ax.sum(axis=2)
        s2 = (h_ax * h_ax).sum(axis=2)
        cross_pairs = 0.5 * (s1 * s1 - s2)

        x_core = rg.concatenate([x[:, :, 0:2], x[:, :, 4:5]], axis=-1)
        blur_feat = nn.gelu(self.blur_conv(p["blur"], gather_stencil(x_core, self.idx_blur, 16)))

        cat = rg.concatenate([s1, cross_pairs, blur_feat, x], axis=-1)
        h = nn.gelu(self.in_ln(p["in_ln"], self.in_proj(p["in_proj"], cat)))
        for i, (fc1, ln1, fc2, se1, se2) in enumerate(self.mlp_blocks):
            bp = p["blocks"][i]
            u = fc2(bp["fc2"], nn.gelu(fc1(bp["fc1"], ln1(bp["ln1"], h))))
            g = u.mean(axis=1, keepdims=True)
            s = nn.sigmoid(se2(bp["se2"], nn.gelu(se1(bp["se1"], g))))
            h = h + u * s
        return self.head(p["head"], h, legal_mask)


class OverlapPatternNet:
    """Inner 1..4 sharp PatternTable + Outer 5..7 PatternTable + 4-axis line overlap (+ optional blur conv)."""
    name = "OverlapPattern"

    def __init__(
        self,
        d_ray: int = 24,
        d_outer: int = 8,
        width: int = 64,
        blocks: int = 3,
        use_blur: bool = False,
    ):
        self.d_ray = d_ray
        self.d_outer = d_outer
        self.width = width
        self.blocks = blocks
        self.use_blur = use_blur
        if use_blur:
            self.name = "FoveaOverlap"

        self.ray_stencil = rg.tensor(RAY_STENCIL_FLAT_NP)
        self.pow4 = rg.tensor(POW4_NP)
        self.outer_stencil = rg.tensor(OUTER_STENCIL_FLAT_NP)
        self.pow4_3 = rg.tensor(POW4_3_NP)
        self.ax_ov_idx = [rg.tensor(arr) for arr in AXIS_OVERLAP_4]
        self.idx_blur = rg.tensor(IDX_BLUR16)

        d_tot = d_ray + d_outer
        self.w_sum = nn.Linear(d_tot, d_ray)
        self.w_prod = nn.Linear(d_tot, d_ray)

        self.ov_comp = nn.Linear(d_ray, 6)
        self.ov_axes = [nn.Linear(4 * 6, 6) for _ in range(4)]

        self.blur_conv = nn.Linear(16 * 3, 12) if use_blur else None
        in_dim = 2 * d_ray + 24 + (12 if use_blur else 0) + IN_CHANNELS
        self.in_proj = nn.Linear(in_dim, width)
        self.in_ln = nn.LayerNorm(width)

        self.mlp_blocks = [
            (
                nn.Linear(width, width),
                nn.LayerNorm(width),
                nn.Linear(width, width),
                nn.Linear(width, width // 2),
                nn.Linear(width // 2, width),
            )
            for _ in range(blocks)
        ]
        self.head = SpatialHead(width)

    def init(self, key):
        ks = random.split(key, 15 + self.blocks * 4)
        ki = 0
        table_in = random.normal(ks[ki], (256, self.d_ray)) * 0.15
        ki += 1
        table_out = random.normal(ks[ki], (64, self.d_outer)) * 0.10
        ki += 1
        ws = self.w_sum.init(ks[ki])
        wp = self.w_prod.init(ks[ki + 1])
        ki += 2
        oc = self.ov_comp.init(ks[ki])
        ki += 1
        oa = [ax.init(ks[ki + a]) for a, ax in enumerate(self.ov_axes)]
        ki += 4
        bc = self.blur_conv.init(ks[ki]) if self.use_blur else None
        ki += 1
        ip = self.in_proj.init(ks[ki])
        ki += 1

        blks = []
        for fc1, ln1, fc2, se1, se2 in self.mlp_blocks:
            blks.append({
                "fc1": fc1.init(ks[ki]),
                "ln1": ln1.init(),
                "fc2": fc2.init(ks[ki + 1]),
                "se1": se1.init(ks[ki + 2]),
                "se2": se2.init(ks[ki + 3]),
            })
            ki += 4

        d = {
            "table_in": table_in,
            "table_out": table_out,
            "w_sum": ws,
            "w_prod": wp,
            "ov_comp": oc,
            "ov_axes": oa,
            "in_proj": ip,
            "in_ln": self.in_ln.init(),
            "blocks": blks,
            "head": self.head.init(ks[ki]),
        }
        if self.use_blur:
            d["blur"] = bc
        return d

    def __call__(self, p, x):
        b = x.shape[0]
        legal_mask = x[:, :, 4]
        s0 = x[:, :, 0] + 2.0 * x[:, :, 1]
        sp = rg.concatenate([s0, rg.full((b, 1), 3.0, x.dtype)], axis=1)

        g_in = rg.take(sp, self.ray_stencil, axis=1).reshape(b, NUM_CELLS * 8, 4)
        ids_in = rg.stop_gradient((g_in @ self.pow4).cast("int32"))
        emb_in = rg.take(p["table_in"], ids_in, axis=0).reshape(b, NUM_CELLS, 4, 2, self.d_ray)

        g_out = rg.take(sp, self.outer_stencil, axis=1).reshape(b, NUM_CELLS * 8, 3)
        ids_out = rg.stop_gradient((g_out @ self.pow4_3).cast("int32"))
        emb_out = rg.take(p["table_out"], ids_out, axis=0).reshape(b, NUM_CELLS, 4, 2, self.d_outer)

        emb = rg.concatenate([emb_in, emb_out], axis=-1)
        left_e, right_e = emb[:, :, :, 0, :], emb[:, :, :, 1, :]
        h_ax = nn.gelu(self.w_sum(p["w_sum"], left_e + right_e) + self.w_prod(p["w_prod"], left_e * right_e))

        s1 = h_ax.sum(axis=2)
        s2 = (h_ax * h_ax).sum(axis=2)
        cross_pairs = 0.5 * (s1 * s1 - s2)

        h_comp = nn.gelu(self.ov_comp(p["ov_comp"], h_ax))
        ov_parts = [
            nn.gelu(self.ov_axes[a](p["ov_axes"][a], gather_stencil(h_comp[:, :, a, :], self.ax_ov_idx[a], 4)))
            for a in range(4)
        ]
        ov_feat = rg.concatenate(ov_parts, axis=-1)

        feats = [s1, cross_pairs, ov_feat]
        if self.use_blur:
            x_core = rg.concatenate([x[:, :, 0:2], x[:, :, 4:5]], axis=-1)
            blur_feat = nn.gelu(self.blur_conv(p["blur"], gather_stencil(x_core, self.idx_blur, 16)))
            feats.append(blur_feat)
        feats.append(x)

        h = nn.gelu(self.in_ln(p["in_ln"], self.in_proj(p["in_proj"], rg.concatenate(feats, axis=-1))))
        for i, (fc1, ln1, fc2, se1, se2) in enumerate(self.mlp_blocks):
            bp = p["blocks"][i]
            u = fc2(bp["fc2"], nn.gelu(fc1(bp["fc1"], ln1(bp["ln1"], h))))
            g = u.mean(axis=1, keepdims=True)
            s = nn.sigmoid(se2(bp["se2"], nn.gelu(se1(bp["se1"], g))))
            h = h + u * s
        return self.head(p["head"], h, legal_mask)


class D4OverlapPatternNet:
    """Strictly D4-equivariant OverlapPatternNet: shared axis-symmetric overlap before angle-split Cross-Fork (90 vs 45)."""
    name = "D4OverlapPattern"

    def __init__(self, d_ray: int = 24, d_outer: int = 8, width: int = 64, blocks: int = 3):
        self.d_ray = d_ray
        self.d_outer = d_outer
        self.width = width
        self.blocks = blocks

        self.ray_stencil = rg.tensor(RAY_STENCIL_FLAT_NP)
        self.pow4 = rg.tensor(POW4_NP)
        self.outer_stencil = rg.tensor(OUTER_STENCIL_FLAT_NP)
        self.pow4_3 = rg.tensor(POW4_3_NP)
        self.ax_ov_idx = [rg.tensor(arr) for arr in AXIS_OVERLAP_4]

        d_tot = d_ray + d_outer
        self.w_sum = nn.Linear(d_tot, d_ray)
        self.w_prod = nn.Linear(d_tot, d_ray)

        self.ov_comp = nn.Linear(d_ray, 6)
        # Single shared overlap kernel over reflection-symmetric (d=1 sum, d=1 prod, d=2 sum, d=2 prod)
        self.ov_shared = nn.Linear(4 * 6, d_ray)

        in_dim = 3 * d_ray + IN_CHANNELS
        self.in_proj = nn.Linear(in_dim, width)
        self.in_ln = nn.LayerNorm(width)

        self.mlp_blocks = [
            (
                nn.Linear(width, width),
                nn.LayerNorm(width),
                nn.Linear(width, width),
                nn.Linear(width, width // 2),
                nn.Linear(width // 2, width),
            )
            for _ in range(blocks)
        ]
        self.head = SpatialHead(width)

    def init(self, key):
        ks = random.split(key, 12 + self.blocks * 4)
        ki = 0
        table_in = random.normal(ks[ki], (256, self.d_ray)) * 0.15
        ki += 1
        table_out = random.normal(ks[ki], (64, self.d_outer)) * 0.10
        ki += 1
        ws = self.w_sum.init(ks[ki])
        wp = self.w_prod.init(ks[ki + 1])
        oc = self.ov_comp.init(ks[ki + 2])
        os_p = self.ov_shared.init(ks[ki + 3])
        ip = self.in_proj.init(ks[ki + 4])
        ki += 5

        blks = []
        for fc1, ln1, fc2, se1, se2 in self.mlp_blocks:
            blks.append({
                "fc1": fc1.init(ks[ki]),
                "ln1": ln1.init(),
                "fc2": fc2.init(ks[ki + 1]),
                "se1": se1.init(ks[ki + 2]),
                "se2": se2.init(ks[ki + 3]),
            })
            ki += 4

        return {
            "table_in": table_in,
            "table_out": table_out,
            "w_sum": ws,
            "w_prod": wp,
            "ov_comp": oc,
            "ov_shared": os_p,
            "in_proj": ip,
            "in_ln": self.in_ln.init(),
            "blocks": blks,
            "head": self.head.init(ks[ki]),
        }

    def __call__(self, p, x):
        b = x.shape[0]
        legal_mask = x[:, :, 4]
        s0 = x[:, :, 0] + 2.0 * x[:, :, 1]
        sp = rg.concatenate([s0, rg.full((b, 1), 3.0, x.dtype)], axis=1)

        g_in = rg.take(sp, self.ray_stencil, axis=1).reshape(b, NUM_CELLS * 8, 4)
        ids_in = rg.stop_gradient((g_in @ self.pow4).cast("int32"))
        emb_in = rg.take(p["table_in"], ids_in, axis=0).reshape(b, NUM_CELLS, 4, 2, self.d_ray)

        g_out = rg.take(sp, self.outer_stencil, axis=1).reshape(b, NUM_CELLS * 8, 3)
        ids_out = rg.stop_gradient((g_out @ self.pow4_3).cast("int32"))
        emb_out = rg.take(p["table_out"], ids_out, axis=0).reshape(b, NUM_CELLS, 4, 2, self.d_outer)

        emb = rg.concatenate([emb_in, emb_out], axis=-1)
        left_e, right_e = emb[:, :, :, 0, :], emb[:, :, :, 1, :]
        h_ax = nn.gelu(self.w_sum(p["w_sum"], left_e + right_e) + self.w_prod(p["w_prod"], left_e * right_e))

        # Shared reflection-symmetric axis overlap BEFORE cross-axis aggregation
        h_comp = nn.gelu(self.ov_comp(p["ov_comp"], h_ax))
        h_hat_axes = []
        for a in range(4):
            nb = gather_stencil(h_comp[:, :, a, :], self.ax_ov_idx[a], 4).reshape(b, NUM_CELLS, 4, 6)
            s_d1 = nb[:, :, 1, :] + nb[:, :, 2, :]
            p_d1 = nb[:, :, 1, :] * nb[:, :, 2, :]
            s_d2 = nb[:, :, 0, :] + nb[:, :, 3, :]
            p_d2 = nb[:, :, 0, :] * nb[:, :, 3, :]
            sym_nb = rg.concatenate([s_d1, p_d1, s_d2, p_d2], axis=-1)
            h_hat_axes.append(h_ax[:, :, a, :] + nn.gelu(self.ov_shared(p["ov_shared"], sym_nb)))

        h0, h1, h2, h3 = h_hat_axes[0], h_hat_axes[1], h_hat_axes[2], h_hat_axes[3]
        h_ortho = h0 + h1
        h_diag = h2 + h3
        s1 = h_ortho + h_diag
        cross_90 = h0 * h1 + h2 * h3
        cross_45 = h_ortho * h_diag

        feats = rg.concatenate([s1, cross_90, cross_45, x], axis=-1)
        h = nn.gelu(self.in_ln(p["in_ln"], self.in_proj(p["in_proj"], feats)))
        for i, (fc1, ln1, fc2, se1, se2) in enumerate(self.mlp_blocks):
            bp = p["blocks"][i]
            u = fc2(bp["fc2"], nn.gelu(fc1(bp["fc1"], ln1(bp["ln1"], h))))
            g = u.mean(axis=1, keepdims=True)
            s = nn.sigmoid(se2(bp["se2"], nn.gelu(se1(bp["se1"], g))))
            h = h + u * s
        return self.head(p["head"], h, legal_mask)


class SpatialHeadV2:
    """3-Head output: Policy logits, Value in [-1, 1], and normalized Turns-to-End in [0, 1]."""

    def __init__(self, d: int):
        self.p_proj = nn.Linear(d, 24)
        self.p_out = nn.Linear(24, 1)
        self.v_fc1 = nn.Linear(2 * d, 96)
        self.v_fc2 = nn.Linear(96, 1)
        self.t_fc1 = nn.Linear(2 * d, 48)
        self.t_fc2 = nn.Linear(48, 1)

    def init(self, key):
        k1, k2, k3, k4, k5, k6 = random.split(key, 6)
        return {
            "p_proj": self.p_proj.init(k1),
            "p_out": self.p_out.init(k2),
            "v_fc1": self.v_fc1.init(k3),
            "v_fc2": self.v_fc2.init(k4),
            "t_fc1": self.t_fc1.init(k5),
            "t_fc2": self.t_fc2.init(k6),
        }

    def __call__(self, p, x, legal_mask):
        ph = nn.gelu(self.p_proj(p["p_proj"], x))
        logits = self.p_out(p["p_out"], ph).squeeze(-1)
        logits = rg.where(legal_mask > 0.5, logits, -1e4)

        vg = rg.concatenate([x.mean(axis=1), x.max(axis=1)], axis=-1)
        vh = nn.gelu(self.v_fc1(p["v_fc1"], vg))
        val = rg.tanh(self.v_fc2(p["v_fc2"], vh)).squeeze(-1)

        th = nn.gelu(self.t_fc1(p["t_fc1"], vg))
        turns = nn.sigmoid(self.t_fc2(p["t_fc2"], th)).squeeze(-1)
        return logits, val, turns


class D4OverlapPatternV2Net:
    """Upgraded D4-equivariant OverlapPatternNet (width=96, blocks=4) with 3-Head (Policy/Value/Turns)."""
    name = "D4OverlapPatternV2"

    def __init__(self, d_ray: int = 32, d_outer: int = 12, width: int = 96, blocks: int = 4):
        self.d_ray = d_ray
        self.d_outer = d_outer
        self.width = width
        self.blocks = blocks

        self.ray_stencil = rg.tensor(RAY_STENCIL_FLAT_NP)
        self.pow4 = rg.tensor(POW4_NP)
        self.outer_stencil = rg.tensor(OUTER_STENCIL_FLAT_NP)
        self.pow4_3 = rg.tensor(POW4_3_NP)
        self.ax_ov_idx = [rg.tensor(arr) for arr in AXIS_OVERLAP_4]

        d_tot = d_ray + d_outer
        self.w_sum = nn.Linear(d_tot, d_ray)
        self.w_prod = nn.Linear(d_tot, d_ray)

        self.ov_comp = nn.Linear(d_ray, 8)
        self.ov_shared = nn.Linear(4 * 8, d_ray)

        in_dim = 3 * d_ray + IN_CHANNELS
        self.in_proj = nn.Linear(in_dim, width)
        self.in_ln = nn.LayerNorm(width)

        self.mlp_blocks = [
            (
                nn.Linear(width, width),
                nn.LayerNorm(width),
                nn.Linear(width, width),
                nn.Linear(width, width // 2),
                nn.Linear(width // 2, width),
            )
            for _ in range(blocks)
        ]
        self.head = SpatialHeadV2(width)

    def init(self, key):
        ks = random.split(key, 12 + self.blocks * 4)
        ki = 0
        table_in = random.normal(ks[ki], (256, self.d_ray)) * 0.15
        ki += 1
        table_out = random.normal(ks[ki], (64, self.d_outer)) * 0.10
        ki += 1
        ws = self.w_sum.init(ks[ki])
        wp = self.w_prod.init(ks[ki + 1])
        oc = self.ov_comp.init(ks[ki + 2])
        os_p = self.ov_shared.init(ks[ki + 3])
        ip = self.in_proj.init(ks[ki + 4])
        ki += 5

        blks = []
        for fc1, ln1, fc2, se1, se2 in self.mlp_blocks:
            blks.append({
                "fc1": fc1.init(ks[ki]),
                "ln1": ln1.init(),
                "fc2": fc2.init(ks[ki + 1]),
                "se1": se1.init(ks[ki + 2]),
                "se2": se2.init(ks[ki + 3]),
            })
            ki += 4

        return {
            "table_in": table_in,
            "table_out": table_out,
            "w_sum": ws,
            "w_prod": wp,
            "ov_comp": oc,
            "ov_shared": os_p,
            "in_proj": ip,
            "in_ln": self.in_ln.init(),
            "blocks": blks,
            "head": self.head.init(ks[ki]),
        }

    def __call__(self, p, x):
        b = x.shape[0]
        legal_mask = x[:, :, 4]
        s0 = x[:, :, 0] + 2.0 * x[:, :, 1]
        sp = rg.concatenate([s0, rg.full((b, 1), 3.0, x.dtype)], axis=1)

        g_in = rg.take(sp, self.ray_stencil, axis=1).reshape(b, NUM_CELLS * 8, 4)
        ids_in = rg.stop_gradient((g_in @ self.pow4).cast("int32"))
        emb_in = rg.take(p["table_in"], ids_in, axis=0).reshape(b, NUM_CELLS, 4, 2, self.d_ray)

        g_out = rg.take(sp, self.outer_stencil, axis=1).reshape(b, NUM_CELLS * 8, 3)
        ids_out = rg.stop_gradient((g_out @ self.pow4_3).cast("int32"))
        emb_out = rg.take(p["table_out"], ids_out, axis=0).reshape(b, NUM_CELLS, 4, 2, self.d_outer)

        emb = rg.concatenate([emb_in, emb_out], axis=-1)
        left_e, right_e = emb[:, :, :, 0, :], emb[:, :, :, 1, :]
        h_ax = nn.gelu(self.w_sum(p["w_sum"], left_e + right_e) + self.w_prod(p["w_prod"], left_e * right_e))

        h_comp = nn.gelu(self.ov_comp(p["ov_comp"], h_ax))
        h_hat_axes = []
        for a in range(4):
            nb = gather_stencil(h_comp[:, :, a, :], self.ax_ov_idx[a], 4).reshape(b, NUM_CELLS, 4, 8)
            s_d1 = nb[:, :, 1, :] + nb[:, :, 2, :]
            p_d1 = nb[:, :, 1, :] * nb[:, :, 2, :]
            s_d2 = nb[:, :, 0, :] + nb[:, :, 3, :]
            p_d2 = nb[:, :, 0, :] * nb[:, :, 3, :]
            sym_nb = rg.concatenate([s_d1, p_d1, s_d2, p_d2], axis=-1)
            h_hat_axes.append(h_ax[:, :, a, :] + nn.gelu(self.ov_shared(p["ov_shared"], sym_nb)))

        h0, h1, h2, h3 = h_hat_axes[0], h_hat_axes[1], h_hat_axes[2], h_hat_axes[3]
        h_ortho = h0 + h1
        h_diag = h2 + h3
        s1 = h_ortho + h_diag
        cross_90 = h0 * h1 + h2 * h3
        cross_45 = h_ortho * h_diag

        feats = rg.concatenate([s1, cross_90, cross_45, x], axis=-1)
        h = nn.gelu(self.in_ln(p["in_ln"], self.in_proj(p["in_proj"], feats)))
        for i, (fc1, ln1, fc2, se1, se2) in enumerate(self.mlp_blocks):
            bp = p["blocks"][i]
            u = fc2(bp["fc2"], nn.gelu(fc1(bp["fc1"], ln1(bp["ln1"], h))))
            g = u.mean(axis=1, keepdims=True)
            s = nn.sigmoid(se2(bp["se2"], nn.gelu(se1(bp["se1"], g))))
            h = h + u * s
        return self.head(p["head"], h, legal_mask)


def count_params(p) -> int:
    return sum(int(np.prod(leaf.shape)) for leaf in tree.leaves(p))
