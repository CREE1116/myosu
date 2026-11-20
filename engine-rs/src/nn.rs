use crate::renju::NUM_CELLS;
use rand::Rng;
use serde::{Deserialize, Serialize};

const H: usize = 15;
const W: usize = 15;

#[cfg(target_os = "macos")]
mod blas {
    pub const ROW_MAJOR: i32 = 101;
    pub const NO_TRANS: i32 = 111;
    pub const TRANS: i32 = 112;

    extern "C" {
        pub fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn sgemm(
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    #[cfg(target_os = "macos")]
    unsafe {
        blas::cblas_sgemm(
            blas::ROW_MAJOR,
            if trans_a { blas::TRANS } else { blas::NO_TRANS },
            if trans_b { blas::TRANS } else { blas::NO_TRANS },
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a.as_ptr(),
            lda as i32,
            b.as_ptr(),
            ldb as i32,
            beta,
            c.as_mut_ptr(),
            ldc as i32,
        );
    }

    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    let av = if trans_a { a[p * lda + i] } else { a[i * lda + p] };
                    let bv = if trans_b { b[j * ldb + p] } else { b[p * ldb + j] };
                    acc += av * bv;
                }
                let idx = i * ldc + j;
                c[idx] = alpha * acc + if beta == 0.0 { 0.0 } else { beta * c[idx] };
            }
        }
    }
}

fn im2col_3x3(x: &[f32], in_c: usize) -> Vec<f32> {
    let mut col = vec![0.0f32; in_c * 9 * NUM_CELLS];
    for ci in 0..in_c {
        let x_plane = &x[ci * NUM_CELLS..(ci + 1) * NUM_CELLS];
        for ky in 0..3 {
            let dy_off = ky as isize - 1;
            let r_start = 0isize.max(-dy_off) as usize;
            let r_end = (H as isize).min(H as isize - dy_off) as usize;
            for kx in 0..3 {
                let dx_off = kx as isize - 1;
                let c_start = 0isize.max(-dx_off) as usize;
                let c_end = (W as isize).min(W as isize - dx_off) as usize;
                let col_plane = &mut col[(ci * 9 + ky * 3 + kx) * NUM_CELLS..];
                for r in r_start..r_end {
                    let xr = (r as isize + dy_off) as usize;
                    let dst_row = &mut col_plane[r * W..r * W + W];
                    let src_row = &x_plane[xr * W..xr * W + W];
                    for c in c_start..c_end {
                        let xc = (c as isize + dx_off) as usize;
                        dst_row[c] = src_row[xc];
                    }
                }
            }
        }
    }
    col
}

fn col2im_3x3(col: &[f32], in_c: usize) -> Vec<f32> {
    let mut dx = vec![0.0f32; in_c * NUM_CELLS];
    for ci in 0..in_c {
        let dx_plane = &mut dx[ci * NUM_CELLS..(ci + 1) * NUM_CELLS];
        for ky in 0..3 {
            let dy_off = ky as isize - 1;
            let r_start = 0isize.max(-dy_off) as usize;
            let r_end = (H as isize).min(H as isize - dy_off) as usize;
            for kx in 0..3 {
                let dx_off = kx as isize - 1;
                let c_start = 0isize.max(-dx_off) as usize;
                let c_end = (W as isize).min(W as isize - dx_off) as usize;
                let col_plane = &col[(ci * 9 + ky * 3 + kx) * NUM_CELLS..];
                for r in r_start..r_end {
                    let xr = (r as isize + dy_off) as usize;
                    let src_row = &col_plane[r * W..r * W + W];
                    let dst_row = &mut dx_plane[xr * W..xr * W + W];
                    for c in c_start..c_end {
                        let xc = (c as isize + dx_off) as usize;
                        dst_row[xc] += src_row[c];
                    }
                }
            }
        }
    }
    dx
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Conv2d {
    pub in_c: usize,
    pub out_c: usize,
    pub k: usize, // 1 or 3
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl Conv2d {
    pub fn new(in_c: usize, out_c: usize, k: usize, rng: &mut impl Rng) -> Self {
        let fan_in = in_c * k * k;
        let std = (2.0 / fan_in as f32).sqrt();
        let weight = (0..out_c * in_c * k * k)
            .map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * std * 0.85)
            .collect();
        let bias = vec![0.0; out_c];
        Self {
            in_c,
            out_c,
            k,
            weight,
            bias,
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0f32; self.out_c * NUM_CELLS];
        for co in 0..self.out_c {
            y[co * NUM_CELLS..(co + 1) * NUM_CELLS].fill(self.bias[co]);
        }

        if self.k == 1 {
            sgemm(
                false,
                false,
                self.out_c,
                NUM_CELLS,
                self.in_c,
                1.0,
                &self.weight,
                self.in_c,
                x,
                NUM_CELLS,
                1.0,
                &mut y,
                NUM_CELLS,
            );
        } else {
            let col = im2col_3x3(x, self.in_c);
            let k_dim = self.in_c * 9;
            sgemm(
                false,
                false,
                self.out_c,
                NUM_CELLS,
                k_dim,
                1.0,
                &self.weight,
                k_dim,
                &col,
                NUM_CELLS,
                1.0,
                &mut y,
                NUM_CELLS,
            );
        }
        y
    }

    pub fn backward(
        &self,
        x: &[f32],
        dy: &[f32],
        dw: &mut [f32],
        db: &mut [f32],
    ) -> Vec<f32> {
        for co in 0..self.out_c {
            let dy_plane = &dy[co * NUM_CELLS..(co + 1) * NUM_CELLS];
            let mut b_acc = 0.0f32;
            for &v in dy_plane {
                b_acc += v;
            }
            db[co] += b_acc;
        }

        if self.k == 1 {
            sgemm(
                false,
                true,
                self.out_c,
                self.in_c,
                NUM_CELLS,
                1.0,
                dy,
                NUM_CELLS,
                x,
                NUM_CELLS,
                1.0,
                dw,
                self.in_c,
            );
            let mut dx = vec![0.0f32; self.in_c * NUM_CELLS];
            sgemm(
                true,
                false,
                self.in_c,
                NUM_CELLS,
                self.out_c,
                1.0,
                &self.weight,
                self.in_c,
                dy,
                NUM_CELLS,
                0.0,
                &mut dx,
                NUM_CELLS,
            );
            dx
        } else {
            let col = im2col_3x3(x, self.in_c);
            let k_dim = self.in_c * 9;
            sgemm(
                false,
                true,
                self.out_c,
                k_dim,
                NUM_CELLS,
                1.0,
                dy,
                NUM_CELLS,
                &col,
                NUM_CELLS,
                1.0,
                dw,
                k_dim,
            );
            let mut d_col = vec![0.0f32; k_dim * NUM_CELLS];
            sgemm(
                true,
                false,
                k_dim,
                NUM_CELLS,
                self.out_c,
                1.0,
                &self.weight,
                k_dim,
                dy,
                NUM_CELLS,
                0.0,
                &mut d_col,
                NUM_CELLS,
            );
            col2im_3x3(&d_col, self.in_c)
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GroupNorm {
    pub channels: usize,
    pub groups: usize,
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
}

struct GnCache {
    x_hat: Vec<f32>,
    inv_std: Vec<f32>,
}

impl GroupNorm {
    pub fn new(channels: usize) -> Self {
        let groups = if channels >= 4 && channels % 4 == 0 {
            4
        } else {
            1
        };
        Self {
            channels,
            groups,
            gamma: vec![1.0; channels],
            beta: vec![0.0; channels],
        }
    }

    fn forward_with_cache(&self, x: &[f32]) -> (Vec<f32>, GnCache) {
        let c_per_g = self.channels / self.groups;
        let m = (c_per_g * NUM_CELLS) as f32;
        let mut y = vec![0.0f32; self.channels * NUM_CELLS];
        let mut x_hat = vec![0.0f32; self.channels * NUM_CELLS];
        let mut inv_std = vec![0.0f32; self.groups];

        for g in 0..self.groups {
            let c0 = g * c_per_g;
            let c1 = c0 + c_per_g;
            let mut mean = 0.0f32;
            for c in c0..c1 {
                for hw in 0..NUM_CELLS {
                    mean += x[c * NUM_CELLS + hw];
                }
            }
            mean /= m;

            let mut var = 0.0f32;
            for c in c0..c1 {
                for hw in 0..NUM_CELLS {
                    let diff = x[c * NUM_CELLS + hw] - mean;
                    var += diff * diff;
                }
            }
            var /= m;
            let istd = 1.0 / (var + 1e-5).sqrt();
            inv_std[g] = istd;

            for c in c0..c1 {
                let gam = self.gamma[c];
                let bet = self.beta[c];
                for hw in 0..NUM_CELLS {
                    let idx = c * NUM_CELLS + hw;
                    let xh = (x[idx] - mean) * istd;
                    x_hat[idx] = xh;
                    y[idx] = xh * gam + bet;
                }
            }
        }

        (y, GnCache { x_hat, inv_std })
    }

    fn backward(
        &self,
        dy: &[f32],
        cache: &GnCache,
        d_gamma: &mut [f32],
        d_beta: &mut [f32],
    ) -> Vec<f32> {
        let c_per_g = self.channels / self.groups;
        let m = (c_per_g * NUM_CELLS) as f32;
        let mut dx = vec![0.0f32; self.channels * NUM_CELLS];

        for g in 0..self.groups {
            let c0 = g * c_per_g;
            let c1 = c0 + c_per_g;
            let istd = cache.inv_std[g];

            let mut sum_dxhat = 0.0f32;
            let mut sum_dxhat_xhat = 0.0f32;

            for c in c0..c1 {
                let gam = self.gamma[c];
                let mut g_acc = 0.0f32;
                let mut b_acc = 0.0f32;
                for hw in 0..NUM_CELLS {
                    let idx = c * NUM_CELLS + hw;
                    let d = dy[idx];
                    let xh = cache.x_hat[idx];
                    g_acc += d * xh;
                    b_acc += d;
                    let dxh = d * gam;
                    sum_dxhat += dxh;
                    sum_dxhat_xhat += dxh * xh;
                }
                d_gamma[c] += g_acc;
                d_beta[c] += b_acc;
            }

            let scale = istd / m;
            for c in c0..c1 {
                let gam = self.gamma[c];
                for hw in 0..NUM_CELLS {
                    let idx = c * NUM_CELLS + hw;
                    let dxh = dy[idx] * gam;
                    dx[idx] = scale * (m * dxh - sum_dxhat - cache.x_hat[idx] * sum_dxhat_xhat);
                }
            }
        }

        dx
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Linear {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weight: Vec<f32>, // [out_dim, in_dim]
    pub bias: Vec<f32>,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize, rng: &mut impl Rng) -> Self {
        let std = (2.0 / in_dim as f32).sqrt();
        let weight = (0..out_dim * in_dim)
            .map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * std * 0.85)
            .collect();
        let bias = vec![0.0; out_dim];
        Self {
            in_dim,
            out_dim,
            weight,
            bias,
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0f32; self.out_dim];
        for o in 0..self.out_dim {
            let row = &self.weight[o * self.in_dim..(o + 1) * self.in_dim];
            let mut sum = self.bias[o];
            for i in 0..self.in_dim {
                sum += row[i] * x[i];
            }
            y[o] = sum;
        }
        y
    }

    pub fn backward(&self, x: &[f32], dy: &[f32], dw: &mut [f32], db: &mut [f32]) -> Vec<f32> {
        let mut dx = vec![0.0f32; self.in_dim];
        for o in 0..self.out_dim {
            let d = dy[o];
            db[o] += d;
            let row = &self.weight[o * self.in_dim..(o + 1) * self.in_dim];
            let dw_row = &mut dw[o * self.in_dim..(o + 1) * self.in_dim];
            for i in 0..self.in_dim {
                dw_row[i] += d * x[i];
                dx[i] += row[i] * d;
            }
        }
        dx
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ResBlock {
    pub conv1: Conv2d,
    pub gn1: GroupNorm,
    pub conv2: Conv2d,
    pub gn2: GroupNorm,
}

impl ResBlock {
    pub fn new(channels: usize, rng: &mut impl Rng) -> Self {
        Self {
            conv1: Conv2d::new(channels, channels, 3, rng),
            gn1: GroupNorm::new(channels),
            conv2: Conv2d::new(channels, channels, 3, rng),
            gn2: GroupNorm::new(channels),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PolicyValueNet {
    pub channels: usize,
    pub stem_conv: Conv2d,
    pub stem_gn: GroupNorm,
    pub blocks: Vec<ResBlock>,
    pub p_conv: Conv2d,
    pub p_gn: GroupNorm,
    pub p_fc: Linear,
    pub v_conv: Conv2d,
    pub v_gn: GroupNorm,
    pub v_fc1: Linear,
    pub v_fc2: Linear,
}

#[inline]
fn relu_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| if v > 0.0 { v } else { 0.0 }).collect()
}

#[inline]
fn relu_backward(dy: &[f32], pre_act: &[f32]) -> Vec<f32> {
    dy.iter()
        .zip(pre_act.iter())
        .map(|(&d, &a)| if a > 0.0 { d } else { 0.0 })
        .collect()
}

impl PolicyValueNet {
    pub fn new(channels: usize, num_blocks: usize, rng: &mut impl Rng) -> Self {
        let blocks = (0..num_blocks)
            .map(|_| ResBlock::new(channels, rng))
            .collect();
        Self {
            channels,
            stem_conv: Conv2d::new(4, channels, 3, rng),
            stem_gn: GroupNorm::new(channels),
            blocks,
            p_conv: Conv2d::new(channels, 2, 1, rng),
            p_gn: GroupNorm::new(2),
            p_fc: Linear::new(2 * NUM_CELLS, NUM_CELLS, rng),
            v_conv: Conv2d::new(channels, 1, 1, rng),
            v_gn: GroupNorm::new(1),
            v_fc1: Linear::new(NUM_CELLS, 64, rng),
            v_fc2: Linear::new(64, 1, rng),
        }
    }

    /// Inference only: returns `(logits[225], value in [-1, 1])`.
    pub fn forward(&self, state: &[f32; 4 * NUM_CELLS]) -> ([f32; NUM_CELLS], f32) {
        let mut h = self.stem_conv.forward(state);
        let (gn, _) = self.stem_gn.forward_with_cache(&h);
        h = relu_vec(&gn);

        for blk in &self.blocks {
            let c1 = blk.conv1.forward(&h);
            let (g1, _) = blk.gn1.forward_with_cache(&c1);
            let r1 = relu_vec(&g1);
            let c2 = blk.conv2.forward(&r1);
            let (g2, _) = blk.gn2.forward_with_cache(&c2);
            let mut sum = vec![0.0f32; h.len()];
            for i in 0..h.len() {
                let v = h[i] + g2[i];
                sum[i] = if v > 0.0 { v } else { 0.0 };
            }
            h = sum;
        }

        // Policy head
        let pc = self.p_conv.forward(&h);
        let (pg, _) = self.p_gn.forward_with_cache(&pc);
        let pr = relu_vec(&pg);
        let p_out = self.p_fc.forward(&pr);
        let mut logits = [0.0f32; NUM_CELLS];
        logits.copy_from_slice(&p_out);

        // Value head
        let vc = self.v_conv.forward(&h);
        let (vg, _) = self.v_gn.forward_with_cache(&vc);
        let vr = relu_vec(&vg);
        let vf1 = self.v_fc1.forward(&vr);
        let vf1_r = relu_vec(&vf1);
        let vf2 = self.v_fc2.forward(&vf1_r);
        let val = vf2[0].tanh();

        (logits, val)
    }

    pub fn num_params(&self) -> usize {
        let conv_n = |c: &Conv2d| c.weight.len() + c.bias.len();
        let gn_n = |g: &GroupNorm| g.gamma.len() + g.beta.len();
        let lin_n = |l: &Linear| l.weight.len() + l.bias.len();

        let mut n = conv_n(&self.stem_conv) + gn_n(&self.stem_gn);
        for b in &self.blocks {
            n += conv_n(&b.conv1) + gn_n(&b.gn1) + conv_n(&b.conv2) + gn_n(&b.gn2);
        }
        n += conv_n(&self.p_conv) + gn_n(&self.p_gn) + lin_n(&self.p_fc);
        n += conv_n(&self.v_conv) + gn_n(&self.v_gn) + lin_n(&self.v_fc1) + lin_n(&self.v_fc2);
        n
    }

    fn visit_params_mut<'a>(&'a mut self, mut f: impl FnMut(&'a mut [f32], bool)) {
        f(&mut self.stem_conv.weight, true);
        f(&mut self.stem_conv.bias, false);
        f(&mut self.stem_gn.gamma, false);
        f(&mut self.stem_gn.beta, false);
        for b in &mut self.blocks {
            f(&mut b.conv1.weight, true);
            f(&mut b.conv1.bias, false);
            f(&mut b.gn1.gamma, false);
            f(&mut b.gn1.beta, false);
            f(&mut b.conv2.weight, true);
            f(&mut b.conv2.bias, false);
            f(&mut b.gn2.gamma, false);
            f(&mut b.gn2.beta, false);
        }
        f(&mut self.p_conv.weight, true);
        f(&mut self.p_conv.bias, false);
        f(&mut self.p_gn.gamma, false);
        f(&mut self.p_gn.beta, false);
        f(&mut self.p_fc.weight, true);
        f(&mut self.p_fc.bias, false);
        f(&mut self.v_conv.weight, true);
        f(&mut self.v_conv.bias, false);
        f(&mut self.v_gn.gamma, false);
        f(&mut self.v_gn.beta, false);
        f(&mut self.v_fc1.weight, true);
        f(&mut self.v_fc1.bias, false);
        f(&mut self.v_fc2.weight, true);
        f(&mut self.v_fc2.bias, false);
    }

    pub fn zero_grads(&self) -> NetGrads {
        NetGrads {
            data: vec![0.0; self.num_params()],
        }
    }

    /// Computes forward + backward for a single training sample `(state, target_pi, target_z)`
    /// accumulating gradients into `grads` and returning `(policy_loss, value_loss)`.
    pub fn accumulate_sample_grads(
        &self,
        state: &[f32; 4 * NUM_CELLS],
        target_pi: &[f32; NUM_CELLS],
        target_z: f32,
        grads: &mut NetGrads,
    ) -> (f32, f32) {
        // Forward with cached intermediates
        let stem_c = self.stem_conv.forward(state);
        let (stem_g, stem_gn_cache) = self.stem_gn.forward_with_cache(&stem_c);
        let mut h = relu_vec(&stem_g);

        struct BlkCache {
            h_in: Vec<f32>,
            gn1_cache: GnCache,
            g1: Vec<f32>,
            r1: Vec<f32>,
            gn2_cache: GnCache,
            pre_relu2: Vec<f32>,
        }
        let mut blk_caches = Vec::with_capacity(self.blocks.len());
        for blk in &self.blocks {
            let h_in = h;
            let c1 = blk.conv1.forward(&h_in);
            let (g1, gn1_cache) = blk.gn1.forward_with_cache(&c1);
            let r1 = relu_vec(&g1);
            let c2 = blk.conv2.forward(&r1);
            let (g2, gn2_cache) = blk.gn2.forward_with_cache(&c2);
            let mut pre_relu2 = vec![0.0f32; h_in.len()];
            for i in 0..h_in.len() {
                pre_relu2[i] = h_in[i] + g2[i];
            }
            h = relu_vec(&pre_relu2);
            blk_caches.push(BlkCache {
                h_in,
                gn1_cache,
                g1,
                r1,
                gn2_cache,
                pre_relu2,
            });
        }

        // Policy head forward
        let pc = self.p_conv.forward(&h);
        let (pg, pg_cache) = self.p_gn.forward_with_cache(&pc);
        let pr = relu_vec(&pg);
        let p_logits = self.p_fc.forward(&pr);

        // Masked softmax over legal cells (plane 3 == 1.0)
        let legal_plane = &state[3 * NUM_CELLS..4 * NUM_CELLS];
        let mut max_l = f32::NEG_INFINITY;
        for i in 0..NUM_CELLS {
            if legal_plane[i] > 0.5 && p_logits[i] > max_l {
                max_l = p_logits[i];
            }
        }
        let mut exp_sum = 0.0f32;
        let mut probs = [0.0f32; NUM_CELLS];
        for i in 0..NUM_CELLS {
            if legal_plane[i] > 0.5 {
                let e = (p_logits[i] - max_l).exp();
                probs[i] = e;
                exp_sum += e;
            }
        }
        let inv_sum = if exp_sum > 0.0 { 1.0 / exp_sum } else { 0.0 };
        let mut p_loss = 0.0f32;
        let mut d_logits = vec![0.0f32; NUM_CELLS];
        for i in 0..NUM_CELLS {
            if legal_plane[i] > 0.5 {
                let p = (probs[i] * inv_sum).max(1e-8);
                p_loss -= target_pi[i] * p.ln();
                d_logits[i] = probs[i] * inv_sum - target_pi[i];
            }
        }

        // Value head forward
        let vc = self.v_conv.forward(&h);
        let (vg, vg_cache) = self.v_gn.forward_with_cache(&vc);
        let vr = relu_vec(&vg);
        let vf1 = self.v_fc1.forward(&vr);
        let vf1_r = relu_vec(&vf1);
        let vf2 = self.v_fc2.forward(&vf1_r);
        let v_pred = vf2[0].tanh();
        let v_diff = v_pred - target_z;
        let v_loss = v_diff * v_diff;

        // Partition grads.data cleanly into mutable slices matching parameter order
        let gdata = &mut grads.data[..];
        let (g_stem_cw, rest) = gdata.split_at_mut(self.stem_conv.weight.len());
        let (g_stem_cb, rest) = rest.split_at_mut(self.stem_conv.bias.len());
        let (g_stem_gg, rest) = rest.split_at_mut(self.stem_gn.gamma.len());
        let (g_stem_gb, mut rest) = rest.split_at_mut(self.stem_gn.beta.len());

        let mut blk_grads = Vec::with_capacity(self.blocks.len());
        for b in &self.blocks {
            let (c1w, r1) = rest.split_at_mut(b.conv1.weight.len());
            let (c1b, r2) = r1.split_at_mut(b.conv1.bias.len());
            let (g1g, r3) = r2.split_at_mut(b.gn1.gamma.len());
            let (g1b, r4) = r3.split_at_mut(b.gn1.beta.len());
            let (c2w, r5) = r4.split_at_mut(b.conv2.weight.len());
            let (c2b, r6) = r5.split_at_mut(b.conv2.bias.len());
            let (g2g, r7) = r6.split_at_mut(b.gn2.gamma.len());
            let (g2b, r8) = r7.split_at_mut(b.gn2.beta.len());
            rest = r8;
            blk_grads.push((c1w, c1b, g1g, g1b, c2w, c2b, g2g, g2b));
        }

        let (g_pcw, rest) = rest.split_at_mut(self.p_conv.weight.len());
        let (g_pcb, rest) = rest.split_at_mut(self.p_conv.bias.len());
        let (g_pgg, rest) = rest.split_at_mut(self.p_gn.gamma.len());
        let (g_pgb, rest) = rest.split_at_mut(self.p_gn.beta.len());
        let (g_pfw, rest) = rest.split_at_mut(self.p_fc.weight.len());
        let (g_pfb, rest) = rest.split_at_mut(self.p_fc.bias.len());

        let (g_vcw, rest) = rest.split_at_mut(self.v_conv.weight.len());
        let (g_vcb, rest) = rest.split_at_mut(self.v_conv.bias.len());
        let (g_vgg, rest) = rest.split_at_mut(self.v_gn.gamma.len());
        let (g_vgb, rest) = rest.split_at_mut(self.v_gn.beta.len());
        let (g_vf1w, rest) = rest.split_at_mut(self.v_fc1.weight.len());
        let (g_vf1b, rest) = rest.split_at_mut(self.v_fc1.bias.len());
        let (g_vf2w, rest) = rest.split_at_mut(self.v_fc2.weight.len());
        let (g_vf2b, _) = rest.split_at_mut(self.v_fc2.bias.len());

        // Policy head backward
        let d_pr = self.p_fc.backward(&pr, &d_logits, g_pfw, g_pfb);
        let d_pg = relu_backward(&d_pr, &pg);
        let d_pc = self.p_gn.backward(&d_pg, &pg_cache, g_pgg, g_pgb);
        let d_h_p = self.p_conv.backward(&h, &d_pc, g_pcw, g_pcb);

        // Value head backward: dL/dv = 2*(v - z), dv/d(vf2) = 1 - v^2
        let d_vf2 = [2.0 * v_diff * (1.0 - v_pred * v_pred)];
        let d_vf1_r = self.v_fc2.backward(&vf1_r, &d_vf2, g_vf2w, g_vf2b);
        let d_vf1 = relu_backward(&d_vf1_r, &vf1);
        let d_vr = self.v_fc1.backward(&vr, &d_vf1, g_vf1w, g_vf1b);
        let d_vg = relu_backward(&d_vr, &vg);
        let d_vc = self.v_gn.backward(&d_vg, &vg_cache, g_vgg, g_vgb);
        let d_h_v = self.v_conv.backward(&h, &d_vc, g_vcw, g_vcb);

        let mut d_h = vec![0.0f32; h.len()];
        for i in 0..d_h.len() {
            d_h[i] = d_h_p[i] + d_h_v[i];
        }

        // Residual blocks backward
        for idx in (0..self.blocks.len()).rev() {
            let blk = &self.blocks[idx];
            let bc = &blk_caches[idx];
            let (c1w, c1b, g1g, g1b, c2w, c2b, g2g, g2b) = &mut blk_grads[idx];

            let d_pre2 = relu_backward(&d_h, &bc.pre_relu2);
            let d_c2 = blk.gn2.backward(&d_pre2, &bc.gn2_cache, g2g, g2b);
            let d_r1 = blk.conv2.backward(&bc.r1, &d_c2, c2w, c2b);
            let d_g1 = relu_backward(&d_r1, &bc.g1);
            let d_c1 = blk.gn1.backward(&d_g1, &bc.gn1_cache, g1g, g1b);
            let d_h_branch = blk.conv1.backward(&bc.h_in, &d_c1, c1w, c1b);

            for i in 0..d_h.len() {
                d_h[i] = d_pre2[i] + d_h_branch[i];
            }
        }

        // Stem backward
        let d_stem_g = relu_backward(&d_h, &stem_g);
        let d_stem_c = self
            .stem_gn
            .backward(&d_stem_g, &stem_gn_cache, g_stem_gg, g_stem_gb);
        let _ = self
            .stem_conv
            .backward(state, &d_stem_c, g_stem_cw, g_stem_cb);

        (p_loss, v_loss)
    }
}

pub struct NetGrads {
    pub data: Vec<f32>,
}

impl NetGrads {
    pub fn add_assign(&mut self, other: &NetGrads) {
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += *b;
        }
    }
}

pub struct AdamW {
    pub lr: f32,
    pub weight_decay: f32,
    pub step: u32,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl AdamW {
    pub fn new(net: &PolicyValueNet, lr: f32, weight_decay: f32) -> Self {
        let n = net.num_params();
        Self {
            lr,
            weight_decay,
            step: 0,
            m: vec![0.0; n],
            v: vec![0.0; n],
        }
    }

    pub fn step(&mut self, net: &mut PolicyValueNet, grads: &NetGrads, batch_size: usize) {
        self.step += 1;
        let inv_b = 1.0 / batch_size.max(1) as f32;
        let b1 = 0.9f32;
        let b2 = 0.999f32;
        let eps = 1e-8f32;
        let bc1 = 1.0 - b1.powi(self.step as i32);
        let bc2 = 1.0 - b2.powi(self.step as i32);

        let mut offset = 0usize;
        let m_all = &mut self.m;
        let v_all = &mut self.v;
        let lr = self.lr;
        let wd = self.weight_decay;

        net.visit_params_mut(|params, decay| {
            let len = params.len();
            for i in 0..len {
                let gi = grads.data[offset + i] * inv_b;
                let m = b1 * m_all[offset + i] + (1.0 - b1) * gi;
                let v = b2 * v_all[offset + i] + (1.0 - b2) * gi * gi;
                m_all[offset + i] = m;
                v_all[offset + i] = v;

                let m_hat = m / bc1;
                let v_hat = v / bc2;
                let mut p = params[i];
                if decay && wd > 0.0 {
                    p -= lr * wd * p;
                }
                p -= lr * m_hat / (v_hat.sqrt() + eps);
                params[i] = p;
            }
            offset += len;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_gradient_check() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut net = PolicyValueNet::new(8, 1, &mut rng);

        let mut state = [0.0f32; 4 * NUM_CELLS];
        for i in 0..NUM_CELLS {
            state[3 * NUM_CELLS + i] = 1.0; // all legal
        }
        state[7 * 15 + 7] = 1.0;
        state[3 * NUM_CELLS + 7 * 15 + 7] = 0.0;

        let mut target_pi = [0.0f32; NUM_CELLS];
        target_pi[7 * 15 + 8] = 0.7;
        target_pi[8 * 15 + 7] = 0.3;
        let target_z = 0.5f32;

        let mut grads = net.zero_grads();
        net.accumulate_sample_grads(&state, &target_pi, target_z, &mut grads);

        let loss_fn = |n: &PolicyValueNet| -> f32 {
            let (logits, val) = n.forward(&state);
            let legal = &state[3 * NUM_CELLS..4 * NUM_CELLS];
            let mut max_l = f32::NEG_INFINITY;
            for i in 0..NUM_CELLS {
                if legal[i] > 0.5 && logits[i] > max_l {
                    max_l = logits[i];
                }
            }
            let mut sum = 0.0f32;
            for i in 0..NUM_CELLS {
                if legal[i] > 0.5 {
                    sum += (logits[i] - max_l).exp();
                }
            }
            let mut pl = 0.0f32;
            for i in 0..NUM_CELLS {
                if legal[i] > 0.5 && target_pi[i] > 0.0 {
                    let p = (logits[i] - max_l).exp() / sum;
                    pl -= target_pi[i] * p.ln();
                }
            }
            pl + (val - target_z).powi(2)
        };

        // Check representative parameters across stem, resblock, policy head, and value head
        let eps = 1e-3f32;
        let check_idx = [
            0usize,
            net.stem_conv.weight.len() + 1,
            net.stem_conv.weight.len() + net.stem_conv.bias.len() + net.stem_gn.gamma.len() + 5,
            net.num_params() - 10,
        ];
        for &target_offset in &check_idx {
            let mut cur = 0usize;
            let mut orig = 0.0f32;
            net.visit_params_mut(|p, _| {
                if target_offset >= cur && target_offset < cur + p.len() {
                    orig = p[target_offset - cur];
                    p[target_offset - cur] = orig + eps;
                }
                cur += p.len();
            });
            let lp = loss_fn(&net);

            cur = 0;
            net.visit_params_mut(|p, _| {
                if target_offset >= cur && target_offset < cur + p.len() {
                    p[target_offset - cur] = orig - eps;
                }
                cur += p.len();
            });
            let lm = loss_fn(&net);

            cur = 0;
            net.visit_params_mut(|p, _| {
                if target_offset >= cur && target_offset < cur + p.len() {
                    p[target_offset - cur] = orig;
                }
                cur += p.len();
            });

            let num_grad = (lp - lm) / (2.0 * eps);
            let ana_grad = grads.data[target_offset];
            let diff = (num_grad - ana_grad).abs();
            assert!(
                diff < 2e-2,
                "Gradient mismatch at {target_offset}: num={num_grad}, ana={ana_grad}, diff={diff}"
            );
        }
    }
}
