//! Weight management for ANE dynamic kernels.
//!
//! Provides weight packing into the IOSurface layout that dynamic kernels expect:
//! `[1, channels, 1, spatial]` where spatial = seq + weight_cols.
//! Also handles model loading (llama2.c format) and delta adapters for fine-tuning.

use crate::agent::ane_mil::MilConfig;
use std::io::{self, Read, Write};
use std::path::Path;

// ---------------------------------------------------------------------------
// Per-layer and full-model weight storage
// ---------------------------------------------------------------------------

/// Per-layer weight storage for a transformer layer.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub wq: Vec<f32>,      // [dim, dim]
    pub wk: Vec<f32>,      // [kv_dim, dim] (= [dim, dim] for MHA)
    pub wv: Vec<f32>,      // [kv_dim, dim] (= [dim, dim] for MHA)
    pub wo: Vec<f32>,      // [dim, dim]
    pub w1: Vec<f32>,      // [hidden, dim]  (gate proj, stored as [dim, hidden] row-major)
    pub w2: Vec<f32>,      // [dim, hidden]  (down proj, stored as [hidden, dim] row-major)
    pub w3: Vec<f32>,      // [hidden, dim]  (up proj, stored as [dim, hidden] row-major)
    pub rms_att: Vec<f32>, // [dim]
    pub rms_ffn: Vec<f32>, // [dim]
    pub q_norm: Option<Vec<f32>>,  // [head_dim] per-head Q RMSNorm (Qwen)
    pub k_norm: Option<Vec<f32>>,  // [head_dim] per-head K RMSNorm (Qwen)
}

/// Full model weights.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    pub cfg: MilConfig,
    pub layers: Vec<LayerWeights>,
    pub rms_final: Vec<f32>,   // [dim]
    pub embed: Vec<f32>,       // [vocab * dim]
    pub vocab_size: usize,
    pub lm_head: Option<Vec<f32>>,  // [vocab * dim] untied classifier (Qwen)
}

// ---------------------------------------------------------------------------
// Weight transpose
// ---------------------------------------------------------------------------

/// Transpose a row-major matrix: src[rows, cols] -> dst[cols, rows].
pub fn transpose_weight(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(src.len(), rows * cols, "transpose_weight: dimension mismatch");
    let mut dst = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    dst
}

// ---------------------------------------------------------------------------
// Generic dynamic matmul packing
// ---------------------------------------------------------------------------

/// Pack activations + weight matrix into `[1, ic, 1, seq+oc]` fp32 layout.
///
/// For each channel d in 0..ic:
///   spatial[0..seq]       = act[d*seq .. d*seq+seq]
///   spatial[seq..seq+oc]  = w[d*oc .. d*oc+oc]
pub fn pack_dyn_matmul(act: &[f32], w: &[f32], ic: usize, oc: usize, seq: usize) -> Vec<u8> {
    assert_eq!(act.len(), ic * seq, "pack_dyn_matmul: act size mismatch");
    assert_eq!(w.len(), ic * oc, "pack_dyn_matmul: weight size mismatch");
    let sp = seq + oc;
    let mut buf = vec![0.0f32; ic * sp];
    for d in 0..ic {
        buf[d * sp..d * sp + seq].copy_from_slice(&act[d * seq..d * seq + seq]);
        buf[d * sp + seq..d * sp + seq + oc].copy_from_slice(&w[d * oc..d * oc + oc]);
    }
    f32_slice_to_bytes(&buf)
}

// ---------------------------------------------------------------------------
// SDPA forward packing / unpacking
// ---------------------------------------------------------------------------

/// Pack xnorm + Wq/Wk/Wv/Wo into `[1, dim, 1, seq+4*dim]` fp32 layout.
pub fn pack_sdpa_fwd(
    xnorm: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    cfg: &MilConfig,
) -> Vec<u8> {
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let sp = seq + 4 * dim;
    assert_eq!(xnorm.len(), dim * seq);
    assert_eq!(wq.len(), dim * dim);
    assert_eq!(wk.len(), dim * dim);
    assert_eq!(wv.len(), dim * dim);
    assert_eq!(wo.len(), dim * dim);

    let mut buf = vec![0.0f32; dim * sp];
    for d in 0..dim {
        buf[d * sp..d * sp + seq].copy_from_slice(&xnorm[d * seq..d * seq + seq]);
        buf[d * sp + seq..d * sp + seq + dim].copy_from_slice(&wq[d * dim..d * dim + dim]);
        buf[d * sp + seq + dim..d * sp + seq + 2 * dim]
            .copy_from_slice(&wk[d * dim..d * dim + dim]);
        buf[d * sp + seq + 2 * dim..d * sp + seq + 3 * dim]
            .copy_from_slice(&wv[d * dim..d * dim + dim]);
        buf[d * sp + seq + 3 * dim..d * sp + seq + 4 * dim]
            .copy_from_slice(&wo[d * dim..d * dim + dim]);
    }
    f32_slice_to_bytes(&buf)
}

/// Unpack SDPA forward output: `[1, 6*dim, 1, seq]` fp32 -> 6 slices of [dim, seq].
/// Returns (o_out, Q, K, V, attn_out, xnorm_pass).
pub fn unpack_sdpa_fwd(output: &[u8], cfg: &MilConfig) -> [Vec<f32>; 6] {
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let total = 6 * dim * seq;
    let floats = bytes_to_f32_vec(output);
    assert_eq!(floats.len(), total, "unpack_sdpa_fwd: output size mismatch");

    let slice = |i: usize| floats[i * dim * seq..(i + 1) * dim * seq].to_vec();
    [slice(0), slice(1), slice(2), slice(3), slice(4), slice(5)]
}

// ---------------------------------------------------------------------------
// FFN packing / unpacking
// ---------------------------------------------------------------------------

/// Pack xnorm + W1 + W3 into `[1, dim, 1, seq+2*hidden]` fp32 layout.
pub fn pack_ffn_w13(
    xnorm: &[f32],
    w1: &[f32],
    w3: &[f32],
    cfg: &MilConfig,
) -> Vec<u8> {
    let dim = cfg.dim;
    let hidden = cfg.hidden_dim;
    let seq = cfg.seq_len;
    let sp = seq + 2 * hidden;
    assert_eq!(xnorm.len(), dim * seq);
    assert_eq!(w1.len(), dim * hidden);
    assert_eq!(w3.len(), dim * hidden);

    let mut buf = vec![0.0f32; dim * sp];
    for d in 0..dim {
        buf[d * sp..d * sp + seq].copy_from_slice(&xnorm[d * seq..d * seq + seq]);
        buf[d * sp + seq..d * sp + seq + hidden]
            .copy_from_slice(&w1[d * hidden..d * hidden + hidden]);
        buf[d * sp + seq + hidden..d * sp + seq + 2 * hidden]
            .copy_from_slice(&w3[d * hidden..d * hidden + hidden]);
    }
    f32_slice_to_bytes(&buf)
}

/// Unpack FFN W13 output: `[1, 3*hidden, 1, seq]` fp32 -> (h1, h3, gate).
pub fn unpack_ffn_w13(output: &[u8], cfg: &MilConfig) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hidden = cfg.hidden_dim;
    let seq = cfg.seq_len;
    let floats = bytes_to_f32_vec(output);
    assert_eq!(floats.len(), 3 * hidden * seq);

    let h1 = floats[0..hidden * seq].to_vec();
    let h3 = floats[hidden * seq..2 * hidden * seq].to_vec();
    let gate = floats[2 * hidden * seq..3 * hidden * seq].to_vec();
    (h1, h3, gate)
}

/// Pack gate + W2 into `[1, hidden, 1, seq+dim]` fp32 layout.
pub fn pack_ffn_w2(act: &[f32], w2: &[f32], cfg: &MilConfig) -> Vec<u8> {
    pack_dyn_matmul(act, w2, cfg.hidden_dim, cfg.dim, cfg.seq_len)
}

// ---------------------------------------------------------------------------
// Backward packing
// ---------------------------------------------------------------------------

/// Pack dh1 + dh3 + W1^T + W3^T into `[1, hidden, 1, 2*seq+2*dim]` fp32 layout.
///
/// Per channel d (0..hidden):
///   sp[0..seq]              = dh1[d*seq..d*seq+seq]
///   sp[seq..2*seq]          = dh3[d*seq..d*seq+seq]
///   sp[2*seq..2*seq+dim]    = w1t[d*dim..d*dim+dim]
///   sp[2*seq+dim..2*seq+2*dim] = w3t[d*dim..d*dim+dim]
pub fn pack_ffn_bwd_w13t(
    dh1: &[f32],
    dh3: &[f32],
    w1t: &[f32],
    w3t: &[f32],
    cfg: &MilConfig,
) -> Vec<u8> {
    let hidden = cfg.hidden_dim;
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let sp = 2 * seq + 2 * dim;
    assert_eq!(dh1.len(), hidden * seq);
    assert_eq!(dh3.len(), hidden * seq);
    assert_eq!(w1t.len(), hidden * dim);
    assert_eq!(w3t.len(), hidden * dim);

    let mut buf = vec![0.0f32; hidden * sp];
    for d in 0..hidden {
        buf[d * sp..d * sp + seq].copy_from_slice(&dh1[d * seq..d * seq + seq]);
        buf[d * sp + seq..d * sp + 2 * seq].copy_from_slice(&dh3[d * seq..d * seq + seq]);
        buf[d * sp + 2 * seq..d * sp + 2 * seq + dim]
            .copy_from_slice(&w1t[d * dim..d * dim + dim]);
        buf[d * sp + 2 * seq + dim..d * sp + 2 * seq + 2 * dim]
            .copy_from_slice(&w3t[d * dim..d * dim + dim]);
    }
    f32_slice_to_bytes(&buf)
}

/// Pack dq + dk + dv + Wq^T + Wk^T + Wv^T into `[1, dim, 1, 3*seq+3*dim]` fp32 layout.
///
/// Per channel d (0..dim):
///   sp[0..seq]                    = dq[d*seq..d*seq+seq]
///   sp[seq..2*seq]                = dk[d*seq..d*seq+seq]
///   sp[2*seq..3*seq]              = dv[d*seq..d*seq+seq]
///   sp[3*seq..3*seq+dim]          = wqt[d*dim..d*dim+dim]
///   sp[3*seq+dim..3*seq+2*dim]    = wkt[d*dim..d*dim+dim]
///   sp[3*seq+2*dim..3*seq+3*dim]  = wvt[d*dim..d*dim+dim]
pub fn pack_qkvb(
    dq: &[f32],
    dk: &[f32],
    dv: &[f32],
    wqt: &[f32],
    wkt: &[f32],
    wvt: &[f32],
    cfg: &MilConfig,
) -> Vec<u8> {
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let sp = 3 * seq + 3 * dim;
    assert_eq!(dq.len(), dim * seq);
    assert_eq!(dk.len(), dim * seq);
    assert_eq!(dv.len(), dim * seq);
    assert_eq!(wqt.len(), dim * dim);
    assert_eq!(wkt.len(), dim * dim);
    assert_eq!(wvt.len(), dim * dim);

    let mut buf = vec![0.0f32; dim * sp];
    for d in 0..dim {
        buf[d * sp..d * sp + seq].copy_from_slice(&dq[d * seq..d * seq + seq]);
        buf[d * sp + seq..d * sp + 2 * seq].copy_from_slice(&dk[d * seq..d * seq + seq]);
        buf[d * sp + 2 * seq..d * sp + 3 * seq].copy_from_slice(&dv[d * seq..d * seq + seq]);
        buf[d * sp + 3 * seq..d * sp + 3 * seq + dim]
            .copy_from_slice(&wqt[d * dim..d * dim + dim]);
        buf[d * sp + 3 * seq + dim..d * sp + 3 * seq + 2 * dim]
            .copy_from_slice(&wkt[d * dim..d * dim + dim]);
        buf[d * sp + 3 * seq + 2 * dim..d * sp + 3 * seq + 3 * dim]
            .copy_from_slice(&wvt[d * dim..d * dim + dim]);
    }
    f32_slice_to_bytes(&buf)
}

/// Pack Q, K, V, da into `[1, 4*dim, 1, seq]` fp16 layout (channel stacking).
///
/// Each slice is [dim, seq] fp32, converted to fp16 and stacked channel-wise.
pub fn pack_sdpa_bwd1(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    da: &[f32],
    cfg: &MilConfig,
) -> Vec<u8> {
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    assert_eq!(q.len(), dim * seq);
    assert_eq!(k.len(), dim * seq);
    assert_eq!(v.len(), dim * seq);
    assert_eq!(da.len(), dim * seq);

    let in_ch = 4 * dim;
    let mut buf = vec![0u8; in_ch * seq * 2]; // fp16
    let write_block = |buf: &mut Vec<u8>, ch_off: usize, data: &[f32]| {
        for i in 0..data.len() {
            let fp16 = half::f16::from_f32(data[i]);
            let off = (ch_off * seq + i) * 2;
            buf[off..off + 2].copy_from_slice(&fp16.to_le_bytes());
        }
    };
    write_block(&mut buf, 0, q);
    write_block(&mut buf, dim, k);
    write_block(&mut buf, 2 * dim, v);
    write_block(&mut buf, 3 * dim, da);
    buf
}

// ---------------------------------------------------------------------------
// Output unpacking for backward kernels
// ---------------------------------------------------------------------------

/// Unpack SDPA bwd1 output: `[1, dim+2*score_ch, 1, seq]` fp16 -> (dV, probs, dp).
pub fn unpack_sdpa_bwd1(output: &[u8], cfg: &MilConfig) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let score_ch = cfg.score_ch();
    let floats = fp16_bytes_to_f32(output);
    assert_eq!(floats.len(), (dim + 2 * score_ch) * seq);

    let dv = floats[0..dim * seq].to_vec();
    let probs = floats[dim * seq..(dim + score_ch) * seq].to_vec();
    let dp = floats[(dim + score_ch) * seq..(dim + 2 * score_ch) * seq].to_vec();
    (dv, probs, dp)
}

/// Unpack SDPA bwd2 output: `[1, 2*dim, 1, seq]` fp16 -> (dQ, dK).
pub fn unpack_sdpa_bwd2(output: &[u8], cfg: &MilConfig) -> (Vec<f32>, Vec<f32>) {
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let floats = fp16_bytes_to_f32(output);
    assert_eq!(floats.len(), 2 * dim * seq);

    let dq = floats[0..dim * seq].to_vec();
    let dk = floats[dim * seq..2 * dim * seq].to_vec();
    (dq, dk)
}

// ---------------------------------------------------------------------------
// Model loading (llama2.c binary format)
// ---------------------------------------------------------------------------

/// llama2.c binary file header.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Llama2Config {
    dim: i32,
    hidden_dim: i32,
    n_layers: i32,
    n_heads: i32,
    n_kv_heads: i32,
    vocab_size: i32,
    seq_len: i32,
}

impl ModelWeights {
    /// Load weights from a llama2.c binary model file.
    ///
    /// The binary format: 7 i32 config fields, then weight tensors in a specific order.
    pub fn from_llama2c(path: &Path, cfg: &MilConfig) -> io::Result<Self> {
        let mut f = std::fs::File::open(path)?;

        // Read config header (7 x i32 = 28 bytes)
        let mut hdr_buf = [0u8; 28];
        f.read_exact(&mut hdr_buf)?;
        let hdr = unsafe { std::ptr::read_unaligned(hdr_buf.as_ptr() as *const Llama2Config) };

        if hdr.dim as usize != cfg.dim || hdr.hidden_dim as usize != cfg.hidden_dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Config mismatch: file dim={} hidden={}, expected dim={} hidden={}",
                    hdr.dim, hdr.hidden_dim, cfg.dim, cfg.hidden_dim
                ),
            ));
        }

        let n_layers = hdr.n_layers as usize;
        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;
        let vocab = hdr.vocab_size.unsigned_abs() as usize;
        let wq_sz = dim * dim;
        let w1_sz = hidden * dim;
        let w2_sz = dim * hidden;

        // Helper: read n f32 values
        let read_f32 = |f: &mut std::fs::File, n: usize| -> io::Result<Vec<f32>> {
            let mut buf = vec![0u8; n * 4];
            f.read_exact(&mut buf)?;
            Ok(buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };

        // Read embedding
        let embed = read_f32(&mut f, vocab * dim)?;

        // Allocate layers
        let mut layers: Vec<LayerWeights> = (0..n_layers)
            .map(|_| LayerWeights {
                wq: vec![],
                wk: vec![],
                wv: vec![],
                wo: vec![],
                w1: vec![],
                w2: vec![],
                w3: vec![],
                rms_att: vec![],
                rms_ffn: vec![],
                q_norm: None,
                k_norm: None,
            })
            .collect();

        // Read in llama2.c order: all rms_att, then all Wq, all Wk, all Wv, all Wo,
        // all rms_ffn, all W1, all W2, all W3, then rms_final
        for l in 0..n_layers {
            layers[l].rms_att = read_f32(&mut f, dim)?;
        }
        for l in 0..n_layers {
            layers[l].wq = read_f32(&mut f, wq_sz)?;
        }
        for l in 0..n_layers {
            layers[l].wk = read_f32(&mut f, wq_sz)?;
        }
        for l in 0..n_layers {
            layers[l].wv = read_f32(&mut f, wq_sz)?;
        }
        for l in 0..n_layers {
            layers[l].wo = read_f32(&mut f, wq_sz)?;
        }
        for l in 0..n_layers {
            layers[l].rms_ffn = read_f32(&mut f, dim)?;
        }
        for l in 0..n_layers {
            layers[l].w1 = read_f32(&mut f, w1_sz)?;
        }
        for l in 0..n_layers {
            layers[l].w2 = read_f32(&mut f, w2_sz)?;
        }
        for l in 0..n_layers {
            layers[l].w3 = read_f32(&mut f, w1_sz)?;
        }

        let rms_final = read_f32(&mut f, dim)?;

        Ok(ModelWeights {
            cfg: cfg.clone(),
            layers,
            rms_final,
            embed,
            vocab_size: vocab,
            lm_head: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Delta adapter I/O
// ---------------------------------------------------------------------------

impl LayerWeights {
    /// Compute weight delta: current - base.
    pub fn delta_from(base: &LayerWeights, current: &LayerWeights) -> LayerWeights {
        LayerWeights {
            wq: vec_sub(&current.wq, &base.wq),
            wk: vec_sub(&current.wk, &base.wk),
            wv: vec_sub(&current.wv, &base.wv),
            wo: vec_sub(&current.wo, &base.wo),
            w1: vec_sub(&current.w1, &base.w1),
            w2: vec_sub(&current.w2, &base.w2),
            w3: vec_sub(&current.w3, &base.w3),
            rms_att: vec_sub(&current.rms_att, &base.rms_att),
            rms_ffn: vec_sub(&current.rms_ffn, &base.rms_ffn),
            q_norm: None,
            k_norm: None,
        }
    }

    /// Apply delta to base weights: base + delta.
    pub fn apply_delta(base: &LayerWeights, delta: &LayerWeights) -> LayerWeights {
        LayerWeights {
            wq: vec_add(&base.wq, &delta.wq),
            wk: vec_add(&base.wk, &delta.wk),
            wv: vec_add(&base.wv, &delta.wv),
            wo: vec_add(&base.wo, &delta.wo),
            w1: vec_add(&base.w1, &delta.w1),
            w2: vec_add(&base.w2, &delta.w2),
            w3: vec_add(&base.w3, &delta.w3),
            rms_att: vec_add(&base.rms_att, &delta.rms_att),
            rms_ffn: vec_add(&base.rms_ffn, &delta.rms_ffn),
            q_norm: base.q_norm.clone(),
            k_norm: base.k_norm.clone(),
        }
    }
}

impl ModelWeights {
    /// Save only the weight deltas (current - base) to a binary file.
    ///
    /// Format: magic(u32) + n_layers(u32) + dim(u32) + hidden(u32) + vocab(u32)
    /// + per-layer deltas + rms_final delta + embed delta.
    pub fn save_delta(&self, path: &Path, base: &ModelWeights) -> io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        let magic: u32 = 0x444C5441; // "DLTA"
        f.write_all(&magic.to_le_bytes())?;
        f.write_all(&(self.layers.len() as u32).to_le_bytes())?;
        f.write_all(&(self.cfg.dim as u32).to_le_bytes())?;
        f.write_all(&(self.cfg.hidden_dim as u32).to_le_bytes())?;
        f.write_all(&(self.vocab_size as u32).to_le_bytes())?;

        for (cur, bas) in self.layers.iter().zip(base.layers.iter()) {
            let delta = LayerWeights::delta_from(bas, cur);
            write_f32_vec(&mut f, &delta.wq)?;
            write_f32_vec(&mut f, &delta.wk)?;
            write_f32_vec(&mut f, &delta.wv)?;
            write_f32_vec(&mut f, &delta.wo)?;
            write_f32_vec(&mut f, &delta.w1)?;
            write_f32_vec(&mut f, &delta.w2)?;
            write_f32_vec(&mut f, &delta.w3)?;
            write_f32_vec(&mut f, &delta.rms_att)?;
            write_f32_vec(&mut f, &delta.rms_ffn)?;
        }

        let rms_delta = vec_sub(&self.rms_final, &base.rms_final);
        write_f32_vec(&mut f, &rms_delta)?;
        let embed_delta = vec_sub(&self.embed, &base.embed);
        write_f32_vec(&mut f, &embed_delta)?;
        Ok(())
    }

    /// Load base weights + delta file to reconstruct fine-tuned weights.
    pub fn load_delta(path: &Path, base: &ModelWeights) -> io::Result<Self> {
        let mut f = std::fs::File::open(path)?;
        let mut hdr = [0u8; 20];
        f.read_exact(&mut hdr)?;
        let magic = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
        if magic != 0x444C5441 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad delta magic"));
        }
        let n_layers = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;
        let dim = u32::from_le_bytes([hdr[8], hdr[9], hdr[10], hdr[11]]) as usize;
        let hidden = u32::from_le_bytes([hdr[12], hdr[13], hdr[14], hdr[15]]) as usize;
        let vocab = u32::from_le_bytes([hdr[16], hdr[17], hdr[18], hdr[19]]) as usize;

        if n_layers != base.layers.len() || dim != base.cfg.dim || hidden != base.cfg.hidden_dim {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "delta config mismatch"));
        }

        let read_f32 = |f: &mut std::fs::File, n: usize| -> io::Result<Vec<f32>> {
            let mut buf = vec![0u8; n * 4];
            f.read_exact(&mut buf)?;
            Ok(buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };

        let wq_sz = dim * dim;
        let w1_sz = hidden * dim;
        let w2_sz = dim * hidden;

        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let delta = LayerWeights {
                wq: read_f32(&mut f, wq_sz)?,
                wk: read_f32(&mut f, wq_sz)?,
                wv: read_f32(&mut f, wq_sz)?,
                wo: read_f32(&mut f, wq_sz)?,
                w1: read_f32(&mut f, w1_sz)?,
                w2: read_f32(&mut f, w2_sz)?,
                w3: read_f32(&mut f, w1_sz)?,
                rms_att: read_f32(&mut f, dim)?,
                rms_ffn: read_f32(&mut f, dim)?,
                q_norm: None,
                k_norm: None,
            };
            layers.push(LayerWeights::apply_delta(&base.layers[l], &delta));
        }

        let rms_delta = read_f32(&mut f, dim)?;
        let rms_final = vec_add(&base.rms_final, &rms_delta);

        let embed_delta = read_f32(&mut f, vocab * dim)?;
        let embed = vec_add(&base.embed, &embed_delta);

        Ok(ModelWeights {
            cfg: base.cfg.clone(),
            layers,
            rms_final,
            embed,
            vocab_size: vocab,
            lm_head: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_f32_vec(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn fp16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

fn vec_sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn vec_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn write_f32_vec(f: &mut std::fs::File, data: &[f32]) -> io::Result<()> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    f.write_all(&bytes)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::ane_bridge::{self, AneKernel};
    use crate::agent::ane_mil::*;

    fn init_ane() {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            ane_bridge::ane_init().expect("ane_init failed — is this Apple Silicon?");
        });
    }

    fn test_cfg() -> MilConfig {
        MilConfig::mha(64, 128, 4, 64)
    }

    // ---- Round 1: transpose_weight + pack_dyn_matmul ----

    #[test]
    fn test_transpose_weight_4x4() {
        let src = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let dst = transpose_weight(&src, 4, 4);
        // Column 0 of src becomes row 0 of dst
        assert_eq!(dst[0], 1.0);
        assert_eq!(dst[1], 5.0);
        assert_eq!(dst[2], 9.0);
        assert_eq!(dst[3], 13.0);
        // Column 1 of src becomes row 1 of dst
        assert_eq!(dst[4], 2.0);
        assert_eq!(dst[5], 6.0);
        assert_eq!(dst[6], 10.0);
        assert_eq!(dst[7], 14.0);
    }

    #[test]
    fn test_transpose_weight_nonsquare() {
        // 2x3 matrix
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dst = transpose_weight(&src, 2, 3);
        // dst is 3x2
        assert_eq!(dst, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_pack_dyn_matmul_identity_on_ane() {
        init_ane();

        let cfg = test_cfg();
        let ic = cfg.dim;
        let oc = cfg.dim;
        let seq = cfg.seq_len;

        // Build activation: small values
        let mut act = vec![0.0f32; ic * seq];
        for c in 0..ic {
            for s in 0..seq {
                act[c * seq + s] = ((c * seq + s) % 100) as f32 * 0.01;
            }
        }

        // Build identity weight
        let mut w = vec![0.0f32; ic * oc];
        for i in 0..ic.min(oc) {
            w[i * oc + i] = 1.0;
        }

        let input_buf = pack_dyn_matmul(&act, &w, ic, oc, seq);

        let spec = KernelSpec::for_kernel(&cfg, KernelType::DynMatmul { ic, oc });
        let kernel = AneKernel::compile(&spec.mil_text, None, &[spec.input_bytes], &[spec.output_bytes])
            .expect("compile failed");

        kernel.write_input(0, &input_buf);
        kernel.eval().expect("eval failed");

        let mut out_buf = vec![0u8; spec.output_bytes];
        kernel.read_output(0, &mut out_buf);
        let output = bytes_to_f32_vec(&out_buf);

        let mut max_err: f32 = 0.0;
        for c in 0..oc {
            for s in 0..seq {
                let expected = act[c * seq + s];
                let got = output[c * seq + s];
                let err = (expected - got).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        assert!(max_err < 0.05, "pack_dyn_matmul identity max error {max_err}");
    }

    // ---- Round 2: pack_sdpa_fwd + unpack_sdpa_fwd ----

    #[test]
    fn test_sdpa_fwd_pack_unpack() {
        init_ane();

        let cfg = test_cfg();
        let dim = cfg.dim;
        let seq = cfg.seq_len;

        // Synthetic xnorm
        let xnorm: Vec<f32> = (0..dim * seq)
            .map(|i| ((i % 200) as f32 - 100.0) * 0.001)
            .collect();

        // Identity Wq,Wk,Wv,Wo
        let mut w_id = vec![0.0f32; dim * dim];
        for i in 0..dim {
            w_id[i * dim + i] = 1.0;
        }

        let input_buf = pack_sdpa_fwd(&xnorm, &w_id, &w_id, &w_id, &w_id, &cfg);
        let mask_blob = build_causal_mask_blob(seq);

        let spec = KernelSpec::for_kernel(&cfg, KernelType::SdpaFwd);
        let kernel = AneKernel::compile_multi_weights(
            &spec.mil_text,
            &["@model_path/weights/mask.bin"],
            &[&mask_blob],
            &[spec.input_bytes],
            &[spec.output_bytes],
        )
        .expect("sdpa_fwd compile failed");

        kernel.write_input(0, &input_buf);
        kernel.eval().expect("sdpa_fwd eval failed");

        let mut out_buf = vec![0u8; spec.output_bytes];
        kernel.read_output(0, &mut out_buf);
        let slices = unpack_sdpa_fwd(&out_buf, &cfg);

        // Verify 6 slices each have [dim, seq] elements
        for s in &slices {
            assert_eq!(s.len(), dim * seq);
        }

        // With identity weights, xnorm passthrough (slice 5) should match input
        let xnorm_pass = &slices[5];
        let mut max_err: f32 = 0.0;
        for i in 0..xnorm.len() {
            let err = (xnorm[i] - xnorm_pass[i]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        // fp32→fp16→fp32 roundtrip for small values
        assert!(max_err < 0.1, "sdpa_fwd xnorm passthrough max error {max_err}");
    }

    // ---- Round 3: FFN packing ----

    #[test]
    fn test_ffn_w13_pack_eval_unpack() {
        init_ane();

        let cfg = test_cfg();
        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;
        let seq = cfg.seq_len;

        let xnorm: Vec<f32> = (0..dim * seq)
            .map(|i| ((i % 200) as f32 - 100.0) * 0.001)
            .collect();

        // Identity-like W1, W3 (dim x hidden, pad with zeros)
        let mut w1 = vec![0.0f32; dim * hidden];
        let mut w3 = vec![0.0f32; dim * hidden];
        for i in 0..dim.min(hidden) {
            w1[i * hidden + i] = 1.0;
            w3[i * hidden + i] = 1.0;
        }

        let input_buf = pack_ffn_w13(&xnorm, &w1, &w3, &cfg);
        let spec = KernelSpec::for_kernel(&cfg, KernelType::FfnW13);
        let kernel = AneKernel::compile(&spec.mil_text, None, &[spec.input_bytes], &[spec.output_bytes])
            .expect("ffn_w13 compile failed");

        kernel.write_input(0, &input_buf);
        kernel.eval().expect("ffn_w13 eval failed");

        let mut out_buf = vec![0u8; spec.output_bytes];
        kernel.read_output(0, &mut out_buf);
        let (h1, h3, gate) = unpack_ffn_w13(&out_buf, &cfg);

        assert_eq!(h1.len(), hidden * seq);
        assert_eq!(h3.len(), hidden * seq);
        assert_eq!(gate.len(), hidden * seq);

        // gate should be non-zero (it's silu(h1)*h3)
        let nonzero = gate.iter().filter(|v| v.abs() > 1e-10).count();
        assert!(nonzero > 0, "ffn_w13 gate is all zeros");
    }

    #[test]
    fn test_ffn_w2_pack_eval() {
        init_ane();

        let cfg = test_cfg();
        let hidden = cfg.hidden_dim;
        let dim = cfg.dim;
        let seq = cfg.seq_len;

        // Synthetic activation
        let act: Vec<f32> = (0..hidden * seq)
            .map(|i| ((i % 200) as f32 - 100.0) * 0.001)
            .collect();

        // Identity-like W2 (hidden x dim)
        let mut w2 = vec![0.0f32; hidden * dim];
        for i in 0..hidden.min(dim) {
            w2[i * dim + i] = 1.0;
        }

        let input_buf = pack_ffn_w2(&act, &w2, &cfg);
        let spec = KernelSpec::for_kernel(&cfg, KernelType::FfnW2);
        let kernel = AneKernel::compile(&spec.mil_text, None, &[spec.input_bytes], &[spec.output_bytes])
            .expect("ffn_w2 compile failed");

        kernel.write_input(0, &input_buf);
        kernel.eval().expect("ffn_w2 eval failed");

        let mut out_buf = vec![0u8; spec.output_bytes];
        kernel.read_output(0, &mut out_buf);
        let output = bytes_to_f32_vec(&out_buf);

        assert_eq!(output.len(), dim * seq);
        let nonzero = output.iter().filter(|v| v.abs() > 1e-10).count();
        assert!(nonzero > 0, "ffn_w2 output is all zeros");
    }

    // ---- Round 4: Backward packing ----

    #[test]
    fn test_ffn_bwd_w13t_pack_eval() {
        init_ane();

        let cfg = test_cfg();
        let hidden = cfg.hidden_dim;
        let dim = cfg.dim;
        let seq = cfg.seq_len;

        let dh1: Vec<f32> = (0..hidden * seq).map(|i| (i % 50) as f32 * 0.001).collect();
        let dh3: Vec<f32> = (0..hidden * seq).map(|i| (i % 30) as f32 * 0.001).collect();

        // W1^T and W3^T: identity-like (hidden x dim)
        let mut w1t = vec![0.0f32; hidden * dim];
        let mut w3t = vec![0.0f32; hidden * dim];
        for i in 0..hidden.min(dim) {
            w1t[i * dim + i] = 1.0;
            w3t[i * dim + i] = 1.0;
        }

        let input_buf = pack_ffn_bwd_w13t(&dh1, &dh3, &w1t, &w3t, &cfg);
        let spec = KernelSpec::for_kernel(&cfg, KernelType::FfnBwdW13t);
        let kernel = AneKernel::compile(&spec.mil_text, None, &[spec.input_bytes], &[spec.output_bytes])
            .expect("ffn_bwd_w13t compile failed");

        kernel.write_input(0, &input_buf);
        kernel.eval().expect("ffn_bwd_w13t eval failed");

        let mut out_buf = vec![0u8; spec.output_bytes];
        kernel.read_output(0, &mut out_buf);
        let output = bytes_to_f32_vec(&out_buf);

        assert_eq!(output.len(), dim * seq);
        let nonzero = output.iter().filter(|v| v.abs() > 1e-10).count();
        assert!(nonzero > 0, "ffn_bwd_w13t output is all zeros");
    }

    #[test]
    fn test_qkvb_pack_eval() {
        init_ane();

        let cfg = test_cfg();
        let dim = cfg.dim;
        let seq = cfg.seq_len;

        let dq: Vec<f32> = (0..dim * seq).map(|i| (i % 50) as f32 * 0.001).collect();
        let dk: Vec<f32> = (0..dim * seq).map(|i| (i % 30) as f32 * 0.001).collect();
        let dv: Vec<f32> = (0..dim * seq).map(|i| (i % 40) as f32 * 0.001).collect();

        // Identity transposed weights
        let mut wt = vec![0.0f32; dim * dim];
        for i in 0..dim {
            wt[i * dim + i] = 1.0;
        }

        let input_buf = pack_qkvb(&dq, &dk, &dv, &wt, &wt, &wt, &cfg);
        let spec = KernelSpec::for_kernel(&cfg, KernelType::Qkvb);
        let kernel = AneKernel::compile(&spec.mil_text, None, &[spec.input_bytes], &[spec.output_bytes])
            .expect("qkvb compile failed");

        kernel.write_input(0, &input_buf);
        kernel.eval().expect("qkvb eval failed");

        let mut out_buf = vec![0u8; spec.output_bytes];
        kernel.read_output(0, &mut out_buf);
        let output = bytes_to_f32_vec(&out_buf);

        assert_eq!(output.len(), dim * seq);
        let nonzero = output.iter().filter(|v| v.abs() > 1e-10).count();
        assert!(nonzero > 0, "qkvb output is all zeros");
    }

    #[test]
    fn test_sdpa_bwd1_pack_eval_unpack() {
        init_ane();

        let cfg = test_cfg();
        let dim = cfg.dim;
        let seq = cfg.seq_len;

        let q: Vec<f32> = (0..dim * seq).map(|i| ((i % 100) as f32 - 50.0) * 0.001).collect();
        let k: Vec<f32> = (0..dim * seq).map(|i| ((i % 80) as f32 - 40.0) * 0.001).collect();
        let v: Vec<f32> = (0..dim * seq).map(|i| ((i % 60) as f32 - 30.0) * 0.001).collect();
        let da: Vec<f32> = (0..dim * seq).map(|i| ((i % 50) as f32 - 25.0) * 0.001).collect();

        let input_buf = pack_sdpa_bwd1(&q, &k, &v, &da, &cfg);
        let mask_blob = build_causal_mask_blob(seq);

        let spec = KernelSpec::for_kernel(&cfg, KernelType::SdpaBwd1);
        let kernel = AneKernel::compile_multi_weights(
            &spec.mil_text,
            &["@model_path/weights/mask.bin"],
            &[&mask_blob],
            &[spec.input_bytes],
            &[spec.output_bytes],
        )
        .expect("sdpa_bwd1 compile failed");

        kernel.write_input(0, &input_buf);
        kernel.eval().expect("sdpa_bwd1 eval failed");

        let mut out_buf = vec![0u8; spec.output_bytes];
        kernel.read_output(0, &mut out_buf);
        let (dv_out, probs, dp) = unpack_sdpa_bwd1(&out_buf, &cfg);

        assert_eq!(dv_out.len(), dim * seq);
        assert_eq!(probs.len(), cfg.score_ch() * seq);
        assert_eq!(dp.len(), cfg.score_ch() * seq);
    }

    // ---- Round 5: ModelWeights::from_llama2c ----

    #[test]
    fn test_from_llama2c_if_present() {
        let model_path = std::path::Path::new("/Users/peppi/Dev/ANE/assets/models/stories110M.bin");
        if !model_path.exists() {
            eprintln!("stories110M.bin not found, skipping llama2c loader test");
            return;
        }

        let cfg = MilConfig::mha(768, 2048, 12, 256);
        let model = ModelWeights::from_llama2c(model_path, &cfg)
            .expect("from_llama2c failed");

        assert_eq!(model.layers.len(), 12);
        assert_eq!(model.layers[0].wq.len(), 768 * 768);
        assert_eq!(model.layers[0].w1.len(), 2048 * 768);
        assert_eq!(model.layers[0].w2.len(), 768 * 2048);
        assert_eq!(model.rms_final.len(), 768);
        assert!(model.vocab_size > 0);
        assert_eq!(model.embed.len(), model.vocab_size * 768);
    }

    // ---- Round 6: Delta adapter round-trip ----

    #[test]
    fn test_delta_roundtrip() {
        let dim = 8;
        let hidden = 16;

        let make_layer = |seed: f32| LayerWeights {
            wq: (0..dim * dim).map(|i| seed + i as f32 * 0.01).collect(),
            wk: (0..dim * dim).map(|i| seed + i as f32 * 0.02).collect(),
            wv: (0..dim * dim).map(|i| seed + i as f32 * 0.03).collect(),
            wo: (0..dim * dim).map(|i| seed + i as f32 * 0.04).collect(),
            w1: (0..hidden * dim).map(|i| seed + i as f32 * 0.05).collect(),
            w2: (0..dim * hidden).map(|i| seed + i as f32 * 0.06).collect(),
            w3: (0..hidden * dim).map(|i| seed + i as f32 * 0.07).collect(),
            rms_att: (0..dim).map(|i| 1.0 + seed + i as f32 * 0.001).collect(),
            rms_ffn: (0..dim).map(|i| 1.0 + seed + i as f32 * 0.002).collect(),
            q_norm: None,
            k_norm: None,
        };

        let base_layer = make_layer(0.0);
        let current_layer = make_layer(0.1);

        // delta_from + apply_delta round-trip
        let delta = LayerWeights::delta_from(&base_layer, &current_layer);
        let reconstructed = LayerWeights::apply_delta(&base_layer, &delta);

        for (a, b) in current_layer.wq.iter().zip(reconstructed.wq.iter()) {
            assert!((a - b).abs() < 1e-6, "wq mismatch: {a} vs {b}");
        }
        for (a, b) in current_layer.w1.iter().zip(reconstructed.w1.iter()) {
            assert!((a - b).abs() < 1e-6, "w1 mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_delta_save_load_roundtrip() {
        let dim = 8;
        let hidden = 16;
        let vocab = 32;
        let cfg = MilConfig::mha(dim, hidden, 2, 4);

        let make_layer = |seed: f32| LayerWeights {
            wq: (0..dim * dim).map(|i| seed + i as f32 * 0.01).collect(),
            wk: (0..dim * dim).map(|i| seed + i as f32 * 0.02).collect(),
            wv: (0..dim * dim).map(|i| seed + i as f32 * 0.03).collect(),
            wo: (0..dim * dim).map(|i| seed + i as f32 * 0.04).collect(),
            w1: (0..hidden * dim).map(|i| seed + i as f32 * 0.05).collect(),
            w2: (0..dim * hidden).map(|i| seed + i as f32 * 0.06).collect(),
            w3: (0..hidden * dim).map(|i| seed + i as f32 * 0.07).collect(),
            rms_att: (0..dim).map(|i| 1.0 + seed + i as f32 * 0.001).collect(),
            rms_ffn: (0..dim).map(|i| 1.0 + seed + i as f32 * 0.002).collect(),
            q_norm: None,
            k_norm: None,
        };

        let base = ModelWeights {
            cfg: cfg.clone(),
            layers: vec![make_layer(0.0), make_layer(0.5)],
            rms_final: (0..dim).map(|i| 1.0 + i as f32 * 0.01).collect(),
            embed: (0..vocab * dim).map(|i| i as f32 * 0.001).collect(),
            vocab_size: vocab,
            lm_head: None,
        };

        let current = ModelWeights {
            cfg: cfg.clone(),
            layers: vec![make_layer(0.1), make_layer(0.6)],
            rms_final: (0..dim).map(|i| 1.1 + i as f32 * 0.01).collect(),
            embed: (0..vocab * dim).map(|i| 0.01 + i as f32 * 0.001).collect(),
            vocab_size: vocab,
            lm_head: None,
        };

        let tmp = tempfile::NamedTempFile::new().expect("tmpfile");
        let path = tmp.path();

        current.save_delta(path, &base).expect("save_delta failed");
        let loaded = ModelWeights::load_delta(path, &base).expect("load_delta failed");

        // Verify all weights match
        for l in 0..2 {
            for (a, b) in current.layers[l].wq.iter().zip(loaded.layers[l].wq.iter()) {
                assert!((a - b).abs() < 1e-5, "layer {l} wq mismatch");
            }
            for (a, b) in current.layers[l].w2.iter().zip(loaded.layers[l].w2.iter()) {
                assert!((a - b).abs() < 1e-5, "layer {l} w2 mismatch");
            }
        }
        for (a, b) in current.rms_final.iter().zip(loaded.rms_final.iter()) {
            assert!((a - b).abs() < 1e-5, "rms_final mismatch");
        }
        for (a, b) in current.embed.iter().zip(loaded.embed.iter()) {
            assert!((a - b).abs() < 1e-5, "embed mismatch");
        }
    }
}
