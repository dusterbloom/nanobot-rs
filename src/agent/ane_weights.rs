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

/// GDN (Gated Delta Network) linear attention weights for a single layer.
///
/// Only present on layers that use linear attention (Qwen3.5 hybrid).
/// Field names match `MlxLinearAttention` in `mlx_lora.rs`.
#[derive(Debug, Clone)]
pub struct GdnLayerWeights {
    pub qkv_proj: Vec<f32>,    // [2*key_dim + value_dim, dim] combined QKV
    pub a_proj: Vec<f32>,      // [Hv, dim] decay parameter projection
    pub b_proj: Vec<f32>,      // [Hv, dim] write gate projection
    pub z_proj: Vec<f32>,      // [value_dim, dim] output gate projection
    pub o_proj: Vec<f32>,      // [dim, value_dim] output projection
    pub a_log: Vec<f32>,       // [Hv] learnable log decay
    pub dt_bias: Vec<f32>,     // [Hv] learnable time bias
    pub norm_weight: Vec<f32>, // [value_head_dim] shared per head or expanded [value_dim]
    pub conv_weight: Vec<f32>, // [qkv_dim, kernel_size] causal depthwise conv
    pub conv_bias: Vec<f32>,   // [qkv_dim] causal conv bias
}

/// Per-layer weight storage for a transformer layer.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub wq: Vec<f32>,             // [dim, dim]
    pub wk: Vec<f32>,             // [kv_dim, dim] (= [dim, dim] for MHA)
    pub wv: Vec<f32>,             // [kv_dim, dim] (= [dim, dim] for MHA)
    pub wo: Vec<f32>,             // [dim, dim]
    pub w1: Vec<f32>,             // [hidden, dim]  (gate proj, stored as [dim, hidden] row-major)
    pub w2: Vec<f32>,             // [dim, hidden]  (down proj, stored as [hidden, dim] row-major)
    pub w3: Vec<f32>,             // [hidden, dim]  (up proj, stored as [dim, hidden] row-major)
    pub rms_att: Vec<f32>,        // [dim]
    pub rms_ffn: Vec<f32>,        // [dim]
    pub q_norm: Option<Vec<f32>>, // [head_dim] per-head Q RMSNorm (Qwen)
    pub k_norm: Option<Vec<f32>>, // [head_dim] per-head K RMSNorm (Qwen)
    /// GDN weights — `Some` for linear attention layers, `None` for MHA layers.
    pub gdn: Option<GdnLayerWeights>,
}

/// Full model weights.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    pub cfg: MilConfig,
    pub layers: Vec<LayerWeights>,
    pub rms_final: Vec<f32>, // [dim]
    pub embed: Vec<f32>,     // [vocab * dim]
    pub vocab_size: usize,
    pub lm_head: Option<Vec<f32>>, // [vocab * dim] untied classifier (Qwen)
}

#[derive(Debug, Clone, Copy)]
struct MlxCheckpointMeta {
    group_size: usize,
    bits: usize,
    n_layers: usize,
    vocab_size: usize,
}

impl MlxCheckpointMeta {
    /// Number of quantized values packed per u32 word (8 for 4-bit, 4 for 8-bit).
    fn elems_per_u32(&self) -> usize {
        32 / self.bits
    }
}

fn parse_mlx_checkpoint_meta(root: &serde_json::Value) -> io::Result<MlxCheckpointMeta> {
    let text_config = root.get("text_config").unwrap_or(root);
    let quant = root
        .get("quantization")
        .or_else(|| root.get("quantization_config"));

    let read_usize = |field: &str| -> io::Result<usize> {
        text_config
            .get(field)
            .or_else(|| root.get(field))
            .and_then(|value| value.as_u64())
            .map(|value| value as usize)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("config.json missing integer field {field}"),
                )
            })
    };

    Ok(MlxCheckpointMeta {
        group_size: quant
            .and_then(|q| q.get("group_size"))
            .and_then(|value| value.as_u64())
            .unwrap_or(64) as usize,
        bits: quant
            .and_then(|q| q.get("bits"))
            .and_then(|value| value.as_u64())
            .unwrap_or(8) as usize,
        n_layers: read_usize("num_hidden_layers")?,
        vocab_size: read_usize("vocab_size")?,
    })
}

fn resolve_tensor_name<V>(tensors: &std::collections::HashMap<String, V>, name: &str) -> String {
    let prefixed = format!("language_model.{name}");
    if tensors.contains_key(name) {
        name.to_string()
    } else if tensors.contains_key(&prefixed) {
        prefixed
    } else {
        name.to_string()
    }
}

fn resolve_weight_base<V>(tensors: &std::collections::HashMap<String, V>, base: &str) -> String {
    let direct_weight = format!("{base}.weight");
    if tensors.contains_key(&direct_weight) {
        return base.to_string();
    }

    let prefixed = format!("language_model.{base}");
    let prefixed_weight = format!("{prefixed}.weight");
    if tensors.contains_key(&prefixed_weight) {
        prefixed
    } else {
        base.to_string()
    }
}

// ---------------------------------------------------------------------------
// RoPE blob generation
// ---------------------------------------------------------------------------

/// Generate precomputed RoPE cos/sin blobs as ANE BLOBFILE format.
///
/// Shape: [1, 1, seq, hd/2] fp16, packed with ANE blob header.
/// Uses half-convention (split, not interleaved): standard for LLaMA/Qwen.
pub fn generate_rope_blobs(seq: usize, head_dim: usize, theta: f64) -> (Vec<u8>, Vec<u8>) {
    let half_hd = head_dim / 2;
    let n = seq * half_hd;

    let mut cos_data = vec![0.0f32; n];
    let mut sin_data = vec![0.0f32; n];

    for t in 0..seq {
        for i in 0..half_hd {
            let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
            let angle = t as f64 * freq;
            cos_data[t * half_hd + i] = angle.cos() as f32;
            sin_data[t * half_hd + i] = angle.sin() as f32;
        }
    }

    (build_fp16_blob(&cos_data), build_fp16_blob(&sin_data))
}

/// Build an ANE blob with 128-byte header + fp16 data (same format as causal mask).
fn build_fp16_blob(data: &[f32]) -> Vec<u8> {
    let data_bytes = data.len() * 2;
    let header_bytes = 128;
    let mut blob = vec![0u8; header_bytes + data_bytes];

    blob[0] = 1;
    blob[4] = 2;
    blob[64] = 0xEF;
    blob[65] = 0xBE;
    blob[66] = 0xAD;
    blob[67] = 0xDE;
    blob[68] = 1;
    blob[72..76].copy_from_slice(&(data_bytes as u32).to_le_bytes());
    blob[80..84].copy_from_slice(&(header_bytes as u32).to_le_bytes());

    for (i, &v) in data.iter().enumerate() {
        let fp16 = half::f16::from_f32(v);
        let offset = header_bytes + i * 2;
        blob[offset..offset + 2].copy_from_slice(&fp16.to_le_bytes());
    }
    blob
}

// ---------------------------------------------------------------------------
// Weight transpose
// ---------------------------------------------------------------------------

/// Transpose a row-major matrix: src[rows, cols] -> dst[cols, rows].
pub fn transpose_weight(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(
        src.len(),
        rows * cols,
        "transpose_weight: dimension mismatch"
    );
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
pub fn pack_ffn_w13(xnorm: &[f32], w1: &[f32], w3: &[f32], cfg: &MilConfig) -> Vec<u8> {
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
        buf[d * sp + 2 * seq..d * sp + 2 * seq + dim].copy_from_slice(&w1t[d * dim..d * dim + dim]);
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
        buf[d * sp + 3 * seq..d * sp + 3 * seq + dim].copy_from_slice(&wqt[d * dim..d * dim + dim]);
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
pub fn pack_sdpa_bwd1(q: &[f32], k: &[f32], v: &[f32], da: &[f32], cfg: &MilConfig) -> Vec<u8> {
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
// Tiled DynMatmul packing / unpacking
// ---------------------------------------------------------------------------

/// Pack one OC-tile of a DynMatmul: act `[ic, seq]` + weight columns `[tile_start..tile_end]`.
///
/// Extracts weight columns from `w[ic, full_oc]` for the given tile, zero-pads if last tile
/// is smaller than `tile_oc`. Returns bytes for IOSurface `[1, ic, 1, seq+tile_oc]` fp32.
pub fn pack_dyn_matmul_oc_tile(
    act: &[f32],
    w: &[f32],
    ic: usize,
    full_oc: usize,
    tile_oc: usize,
    tile_start: usize,
    seq: usize,
) -> Vec<u8> {
    let actual_oc = (full_oc - tile_start).min(tile_oc);
    let sp = seq + tile_oc;
    let mut buf = vec![0.0f32; ic * sp];
    for d in 0..ic {
        buf[d * sp..d * sp + seq].copy_from_slice(&act[d * seq..d * seq + seq]);
        buf[d * sp + seq..d * sp + seq + actual_oc]
            .copy_from_slice(&w[d * full_oc + tile_start..d * full_oc + tile_start + actual_oc]);
        // remaining positions stay zero (padding)
    }
    f32_slice_to_bytes(&buf)
}

/// Unpack one OC-tile output and write into result buffer (concat along OC).
///
/// Copies the first `actual_oc` channels from tile output `[1, tile_oc, 1, seq]` into
/// `result[tile_start*seq..]`.
pub fn unpack_oc_tile(
    out_bytes: &[u8],
    result: &mut [f32],
    tile_oc: usize,
    tile_start: usize,
    actual_oc: usize,
    seq: usize,
) {
    let floats = bytes_to_f32_vec(out_bytes);
    for ch in 0..actual_oc {
        let src_off = ch * seq;
        let dst_off = (tile_start + ch) * seq;
        result[dst_off..dst_off + seq].copy_from_slice(&floats[src_off..src_off + seq]);
    }
}

/// Pack one IC-tile of a DynMatmul: act channels `[tile_start..tile_end]` + weight rows.
///
/// Extracts activation channels and weight rows for the tile, zero-pads last tile.
/// Returns bytes for IOSurface `[1, tile_ic, 1, seq+oc]` fp32.
pub fn pack_dyn_matmul_ic_tile(
    act: &[f32],
    w: &[f32],
    full_ic: usize,
    oc: usize,
    tile_ic: usize,
    tile_start: usize,
    seq: usize,
) -> Vec<u8> {
    let actual_ic = (full_ic - tile_start).min(tile_ic);
    let sp = seq + oc;
    let mut buf = vec![0.0f32; tile_ic * sp];
    for d in 0..actual_ic {
        let src_d = tile_start + d;
        buf[d * sp..d * sp + seq].copy_from_slice(&act[src_d * seq..src_d * seq + seq]);
        buf[d * sp + seq..d * sp + seq + oc]
            .copy_from_slice(&w[src_d * oc..src_d * oc + oc]);
    }
    // remaining channels stay zero (padding)
    f32_slice_to_bytes(&buf)
}

/// Unpack one IC-tile output and accumulate into result (reduction sum).
///
/// Adds tile output `[1, oc, 1, seq]` into `result[oc * seq]`.
pub fn unpack_ic_tile_accum(out_bytes: &[u8], result: &mut [f32], oc: usize, seq: usize) {
    let floats = bytes_to_f32_vec(out_bytes);
    for i in 0..oc * seq {
        result[i] += floats[i];
    }
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
                gdn: None,
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

    /// Load weights from an MLX safetensors directory (Qwen3 architecture).
    ///
    /// Handles 8-bit quantized weights (U32-packed + BF16 scales/biases, group_size=64).
    /// Expands KV weights for GQA (replicates each KV head `heads_per_group` times).
    pub fn from_mlx_safetensors(dir: &Path, cfg: &MilConfig) -> io::Result<Self> {
        use std::collections::HashMap;

        // Find safetensors files
        let mut st_files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
            .collect();
        st_files.sort();

        if st_files.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "no safetensors files",
            ));
        }

        // Parse all safetensors files into a name→data map
        let mut tensors: HashMap<String, Vec<u8>> = HashMap::new();
        let mut tensor_meta: HashMap<String, (String, Vec<usize>)> = HashMap::new(); // name → (dtype, shape)

        for st_path in &st_files {
            let data = std::fs::read(st_path)?;
            if data.len() < 8 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small"));
            }
            let hdr_size = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
            let hdr_json: serde_json::Value = serde_json::from_slice(&data[8..8 + hdr_size])
                .map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("bad header: {e}"))
                })?;
            let data_start = 8 + hdr_size;

            if let serde_json::Value::Object(map) = hdr_json {
                for (name, meta) in &map {
                    if name == "__metadata__" {
                        continue;
                    }
                    let dtype = meta["dtype"].as_str().unwrap_or("").to_string();
                    let shape: Vec<usize> = meta["shape"]
                        .as_array()
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_u64().map(|n| n as usize))
                                .collect()
                        })
                        .unwrap_or_default();
                    let offsets = meta["data_offsets"].as_array().unwrap();
                    let start = offsets[0].as_u64().unwrap() as usize;
                    let end = offsets[1].as_u64().unwrap() as usize;
                    tensors.insert(
                        name.clone(),
                        data[data_start + start..data_start + end].to_vec(),
                    );
                    tensor_meta.insert(name.clone(), (dtype, shape));
                }
            }
        }

        /// BF16 bytes → f32 vec
        fn bf16_to_f32(data: &[u8]) -> Vec<f32> {
            data.chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect()
        }

        /// Get a dequantized weight tensor by base name (e.g. "model.layers.0.self_attn.q_proj")
        ///
        /// `bits` — quantization width (4 or 8). Determines packing: 8 or 4 values per u32.
        let get_weight = |tensors: &HashMap<String, Vec<u8>>,
                          tensor_meta: &HashMap<String, (String, Vec<usize>)>,
                          base: &str,
                          group_size: usize,
                          bits: usize|
         -> io::Result<Vec<f32>> {
            let base = resolve_weight_base(tensors, base);
            let w_key = format!("{base}.weight");
            let s_key = format!("{base}.scales");
            let b_key = format!("{base}.biases");

            if let (Some(w), Some(s), Some(b)) = (
                tensors.get(&w_key),
                tensors.get(&s_key),
                tensors.get(&b_key),
            ) {
                let (_, shape) = tensor_meta.get(&w_key).unwrap();
                let rows = shape[0];
                let packed_cols = shape[1];
                let elems_per_u32 = 32 / bits;
                let cols = packed_cols * elems_per_u32;
                let sc = bf16_to_f32(s);
                let bi = bf16_to_f32(b);
                Ok(dequant_nbit(w, &sc, &bi, rows, cols, group_size, bits))
            } else {
                // Try non-quantized (BF16)
                let key = base.to_string();
                if let Some(data) = tensors.get(&format!("{base}.weight")).or(tensors.get(&key)) {
                    let (dtype, _) = tensor_meta
                        .get(&format!("{base}.weight"))
                        .or(tensor_meta.get(&key))
                        .unwrap();
                    if dtype == "BF16" {
                        Ok(bf16_to_f32(data))
                    } else {
                        Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("unsupported dtype {dtype} for {base}"),
                        ))
                    }
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("missing tensor: {base}"),
                    ))
                }
            }
        };

        /// Get a BF16 tensor directly (for layernorm weights, QK-norm)
        let get_bf16 = |tensors: &HashMap<String, Vec<u8>>, name: &str| -> io::Result<Vec<f32>> {
            let name = resolve_tensor_name(tensors, name);
            let data = tensors.get(&name).ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, format!("missing: {name}"))
            })?;
            Ok(bf16_to_f32(data))
        };

        // Read config.json for group_size
        let config_path = dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: serde_json::Value = serde_json::from_str(&config_str).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("bad config.json: {e}"))
        })?;
        let meta = parse_mlx_checkpoint_meta(&config)?;

        let dim = cfg.dim;
        let attn_dim = cfg.attn_dim();
        let kv_dim = cfg.kv_dim();
        let n_layers = meta.n_layers;
        let vocab_size = meta.vocab_size;
        let group_size = meta.group_size;
        let bits = meta.bits;
        let hpg = cfg.heads_per_group();

        // Embedding (quantized or bf16)
        let embed_raw = get_weight(
            &tensors,
            &tensor_meta,
            "model.embed_tokens",
            group_size,
            bits,
        )?;
        // embed_raw is [vocab, dim] row-major → we need [dim, vocab] col-major (our format: embed[d * vocab + v])
        // Actually our format: embed[v * dim + d] for embed_lookup which does: out[d * seq + t] = embed[token * dim + d]
        // So we need [vocab, dim] row-major — that's what we already have.
        // Wait, our embed_lookup: out[d*seq+t] = embed[tok*dim + d]. So embed is [vocab, dim] where embed[tok*dim+d].
        // The safetensors gives us [vocab, dim] row-major. That matches!

        // Expand KV: replicate each KV head hpg times to get [dim, dim] from [kv_dim, dim]
        let expand_kv = |kv: &[f32], kv_dim: usize, dim: usize, hpg: usize| -> Vec<f32> {
            if hpg == 1 {
                return kv.to_vec();
            }
            let head_dim = kv_dim / (dim / (hpg * (kv_dim / (dim / hpg / hpg))));
            // Simpler: kv is [kv_dim, dim_in] where kv_dim = n_kv_heads * head_dim
            // We want [dim, dim_in] by repeating each head_dim-sized block hpg times
            let hd = cfg.head_dim();
            let n_kv = kv_dim / hd;
            let dim_in = kv.len() / kv_dim;
            let mut expanded = vec![0.0f32; dim * dim_in];
            for kv_h in 0..n_kv {
                for rep in 0..hpg {
                    let dst_h = kv_h * hpg + rep;
                    for d in 0..hd {
                        let src_row = kv_h * hd + d;
                        let dst_row = dst_h * hd + d;
                        expanded[dst_row * dim_in..dst_row * dim_in + dim_in]
                            .copy_from_slice(&kv[src_row * dim_in..src_row * dim_in + dim_in]);
                    }
                }
            }
            expanded
        };

        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let prefix = format!("model.layers.{l}");

            let is_gdn = cfg.is_linear_attn_layer(l);

            // Attention weights — MHA and GDN layers use different projections
            let (wq, wk, wv, wo, q_norm, k_norm, gdn) = if is_gdn {
                // GDN layers: load from linear_attn.* prefix, MHA projections are empty
                let la = format!("{prefix}.linear_attn");
                let gdn_w = GdnLayerWeights {
                    qkv_proj: get_weight(
                        &tensors,
                        &tensor_meta,
                        &format!("{la}.in_proj_qkv"),
                        group_size,
                        bits,
                    )?,
                    a_proj: get_weight(
                        &tensors,
                        &tensor_meta,
                        &format!("{la}.in_proj_a"),
                        group_size,
                        bits,
                    )?,
                    b_proj: get_weight(
                        &tensors,
                        &tensor_meta,
                        &format!("{la}.in_proj_b"),
                        group_size,
                        bits,
                    )?,
                    z_proj: get_weight(
                        &tensors,
                        &tensor_meta,
                        &format!("{la}.in_proj_z"),
                        group_size,
                        bits,
                    )?,
                    o_proj: get_weight(
                        &tensors,
                        &tensor_meta,
                        &format!("{la}.out_proj"),
                        group_size,
                        bits,
                    )?,
                    a_log: get_bf16(&tensors, &format!("{la}.A_log"))?,
                    dt_bias: get_bf16(&tensors, &format!("{la}.dt_bias"))?,
                    norm_weight: get_bf16(&tensors, &format!("{la}.norm.weight"))?,
                    conv_weight: get_bf16(&tensors, &format!("{la}.conv1d.weight"))?,
                    conv_bias: get_bf16(&tensors, &format!("{la}.conv1d.bias")).unwrap_or_default(),
                };
                // GDN layers have no separate q/k/v/o projections
                (vec![], vec![], vec![], vec![], None, None, Some(gdn_w))
            } else {
                // MHA layers: standard q_proj/k_proj/v_proj/o_proj
                let wq = get_weight(
                    &tensors,
                    &tensor_meta,
                    &format!("{prefix}.self_attn.q_proj"),
                    group_size,
                    bits,
                )?;
                let wk_raw = get_weight(
                    &tensors,
                    &tensor_meta,
                    &format!("{prefix}.self_attn.k_proj"),
                    group_size,
                    bits,
                )?;
                let wv_raw = get_weight(
                    &tensors,
                    &tensor_meta,
                    &format!("{prefix}.self_attn.v_proj"),
                    group_size,
                    bits,
                )?;
                let wo = get_weight(
                    &tensors,
                    &tensor_meta,
                    &format!("{prefix}.self_attn.o_proj"),
                    group_size,
                    bits,
                )?;
                let wk = expand_kv(&wk_raw, kv_dim, attn_dim, hpg);
                let wv = expand_kv(&wv_raw, kv_dim, attn_dim, hpg);
                let q_norm = get_bf16(&tensors, &format!("{prefix}.self_attn.q_norm.weight")).ok();
                let k_norm = get_bf16(&tensors, &format!("{prefix}.self_attn.k_norm.weight")).ok();
                (wq, wk, wv, wo, q_norm, k_norm, None)
            };

            // FFN weights (shared by both MHA and GDN layers)
            // Fallback chain for MLP prefix:
            //   1. mlp.gate_proj (dense models)
            //   2. mlp.shared_expert.gate_proj (MoE — train shared expert only)
            //   3. mlp.gate_up_proj (fused gate+up, split in half)
            let try_load_ffn = |pfx: &str| -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
                let g = get_weight(
                    &tensors,
                    &tensor_meta,
                    &format!("{pfx}.gate_proj"),
                    group_size,
                    bits,
                )
                .ok()?;
                let u = get_weight(
                    &tensors,
                    &tensor_meta,
                    &format!("{pfx}.up_proj"),
                    group_size,
                    bits,
                )
                .ok()?;
                let d = get_weight(
                    &tensors,
                    &tensor_meta,
                    &format!("{pfx}.down_proj"),
                    group_size,
                    bits,
                )
                .ok()?;
                Some((g, u, d))
            };
            let (w1, w3, w2) = if let Some(ffn) = try_load_ffn(&format!("{prefix}.mlp")) {
                ffn
            } else if let Some(ffn) = try_load_ffn(&format!("{prefix}.mlp.shared_expert")) {
                ffn
            } else {
                // Fused gate_up_proj: [2*hidden_dim, dim] → split in half by rows
                let fused = get_weight(
                    &tensors,
                    &tensor_meta,
                    &format!("{prefix}.mlp.gate_up_proj"),
                    group_size,
                    bits,
                )?;
                let mid = fused.len() / 2;
                let d = get_weight(
                    &tensors,
                    &tensor_meta,
                    &format!("{prefix}.mlp.down_proj"),
                    group_size,
                    bits,
                )?;
                (fused[..mid].to_vec(), fused[mid..].to_vec(), d)
            };

            // RMSNorm weights (BF16, not quantized)
            let rms_att = get_bf16(&tensors, &format!("{prefix}.input_layernorm.weight"))?;
            let rms_ffn = get_bf16(
                &tensors,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )?;

            layers.push(LayerWeights {
                wq,
                wk,
                wv,
                wo,
                w1,
                w2,
                w3,
                rms_att,
                rms_ffn,
                q_norm,
                k_norm,
                gdn,
            });

            if l == 0 {
                tracing::debug!(
                    wq = layers[0].wq.len(),
                    wk = layers[0].wk.len(),
                    wv = layers[0].wv.len(),
                    wo = layers[0].wo.len(),
                    w1 = layers[0].w1.len(),
                    w2 = layers[0].w2.len(),
                    "loaded f32 layer 0"
                );
            }
        }

        let rms_final = get_bf16(&tensors, "model.norm.weight")?;

        Ok(ModelWeights {
            cfg: cfg.clone(),
            layers,
            rms_final,
            embed: embed_raw,
            vocab_size,
            lm_head: None, // tied embeddings
        })
    }
}

impl QuantizedModelWeights {
    /// Load weights from MLX safetensors WITHOUT dequantizing layer weights.
    ///
    /// Layer weight matrices (wq/wk/wv/wo/w1/w2/w3) stay in their quantized
    /// 8-bit representation. Only embedding, final RMSNorm, and per-layer norms
    /// are stored as f32.
    ///
    /// Memory savings vs `ModelWeights::from_mlx_safetensors`:
    /// - 1.7B model: ~7.6 GB → ~1.9 GB (quantized + norms + embed)
    /// - 9B model: doesn't fit → ~3 GB
    pub fn from_mlx_safetensors(dir: &Path, cfg: &MilConfig) -> io::Result<Self> {
        use std::collections::HashMap;

        let mut st_files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
            .collect();
        st_files.sort();

        if st_files.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "no safetensors files",
            ));
        }

        let mut tensors: HashMap<String, Vec<u8>> = HashMap::new();
        let mut tensor_meta: HashMap<String, (String, Vec<usize>)> = HashMap::new();

        for st_path in &st_files {
            let data = std::fs::read(st_path)?;
            if data.len() < 8 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small"));
            }
            let hdr_size = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
            let hdr_json: serde_json::Value = serde_json::from_slice(&data[8..8 + hdr_size])
                .map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("bad header: {e}"))
                })?;
            let data_start = 8 + hdr_size;

            if let serde_json::Value::Object(map) = hdr_json {
                for (name, meta) in &map {
                    if name == "__metadata__" {
                        continue;
                    }
                    let dtype = meta["dtype"].as_str().unwrap_or("").to_string();
                    let shape: Vec<usize> = meta["shape"]
                        .as_array()
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_u64().map(|n| n as usize))
                                .collect()
                        })
                        .unwrap_or_default();
                    let offsets = meta["data_offsets"].as_array().unwrap();
                    let start = offsets[0].as_u64().unwrap() as usize;
                    let end = offsets[1].as_u64().unwrap() as usize;
                    tensors.insert(
                        name.clone(),
                        data[data_start + start..data_start + end].to_vec(),
                    );
                    tensor_meta.insert(name.clone(), (dtype, shape));
                }
            }
        }

        fn bf16_to_f32(data: &[u8]) -> Vec<f32> {
            data.chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect()
        }

        /// Get a quantized tensor WITHOUT dequantizing.
        /// Returns QuantizedTensor with raw bytes + scales + biases.
        let get_quantized = |tensors: &HashMap<String, Vec<u8>>,
                             tensor_meta: &HashMap<String, (String, Vec<usize>)>,
                             base: &str,
                             group_size: usize,
                             bits: usize|
         -> io::Result<QuantizedTensor> {
            let base = resolve_weight_base(tensors, base);
            let w_key = format!("{base}.weight");
            let s_key = format!("{base}.scales");
            let b_key = format!("{base}.biases");

            if let (Some(w), Some(s), Some(b)) = (
                tensors.get(&w_key),
                tensors.get(&s_key),
                tensors.get(&b_key),
            ) {
                let (_, shape) = tensor_meta.get(&w_key).unwrap();
                let rows = shape[0];
                let packed_cols = shape[1];
                let elems_per_u32 = 32 / bits;
                let cols = packed_cols * elems_per_u32;

                Ok(QuantizedTensor {
                    data: w.clone(),
                    scales: bf16_to_f32(s),
                    biases: bf16_to_f32(b),
                    rows,
                    cols,
                    group_size,
                    bits,
                })
            } else {
                Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("missing quantized tensor: {base} (need .weight/.scales/.biases)"),
                ))
            }
        };

        let get_bf16 = |tensors: &HashMap<String, Vec<u8>>, name: &str| -> io::Result<Vec<f32>> {
            let name = resolve_tensor_name(tensors, name);
            let data = tensors.get(&name).ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, format!("missing: {name}"))
            })?;
            Ok(bf16_to_f32(data))
        };

        /// Dequantize a tensor to f32 (for embeddings that need random access as f32).
        let get_weight_f32 = |tensors: &HashMap<String, Vec<u8>>,
                              tensor_meta: &HashMap<String, (String, Vec<usize>)>,
                              base: &str,
                              group_size: usize,
                              bits: usize|
         -> io::Result<Vec<f32>> {
            let base = resolve_weight_base(tensors, base);
            let w_key = format!("{base}.weight");
            let s_key = format!("{base}.scales");
            let b_key = format!("{base}.biases");
            if let (Some(w), Some(s), Some(b)) = (
                tensors.get(&w_key),
                tensors.get(&s_key),
                tensors.get(&b_key),
            ) {
                let (_, shape) = tensor_meta.get(&w_key).unwrap();
                let rows = shape[0];
                let elems_per_u32 = 32 / bits;
                let cols = shape[1] * elems_per_u32;
                let sc = bf16_to_f32(s);
                let bi = bf16_to_f32(b);
                Ok(dequant_nbit(w, &sc, &bi, rows, cols, group_size, bits))
            } else if let Some(data) = tensors.get(&format!("{base}.weight")) {
                Ok(bf16_to_f32(data))
            } else {
                Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("missing tensor: {base}"),
                ))
            }
        };

        let config_path = dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: serde_json::Value = serde_json::from_str(&config_str).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("bad config.json: {e}"))
        })?;
        let meta = parse_mlx_checkpoint_meta(&config)?;
        let group_size = meta.group_size;
        let bits = meta.bits;
        let n_layers = meta.n_layers;
        let vocab_size = meta.vocab_size;
        let hpg = cfg.heads_per_group();

        // Embedding — must be f32 (accessed every step, random access pattern)
        let embed_raw = get_weight_f32(
            &tensors,
            &tensor_meta,
            "model.embed_tokens",
            group_size,
            bits,
        )?;

        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let prefix = format!("model.layers.{l}");
            let is_gdn = cfg.is_linear_attn_layer(l);

            // Attention weights — MHA and GDN layers use different projections
            let (wq, wk, wv, wo, q_norm, k_norm, gdn) = if is_gdn {
                // GDN layers: load from linear_attn.* prefix
                let la = format!("{prefix}.linear_attn");
                let gdn_w = QuantizedGdnLayerWeights {
                    qkv_proj: get_quantized(
                        &tensors,
                        &tensor_meta,
                        &format!("{la}.in_proj_qkv"),
                        group_size,
                        bits,
                    )?,
                    a_proj: get_quantized(
                        &tensors,
                        &tensor_meta,
                        &format!("{la}.in_proj_a"),
                        group_size,
                        bits,
                    )?,
                    b_proj: get_quantized(
                        &tensors,
                        &tensor_meta,
                        &format!("{la}.in_proj_b"),
                        group_size,
                        bits,
                    )?,
                    z_proj: get_quantized(
                        &tensors,
                        &tensor_meta,
                        &format!("{la}.in_proj_z"),
                        group_size,
                        bits,
                    )?,
                    o_proj: get_quantized(
                        &tensors,
                        &tensor_meta,
                        &format!("{la}.out_proj"),
                        group_size,
                        bits,
                    )?,
                    a_log: get_bf16(&tensors, &format!("{la}.A_log"))?,
                    dt_bias: get_bf16(&tensors, &format!("{la}.dt_bias"))?,
                    norm_weight: get_bf16(&tensors, &format!("{la}.norm.weight"))?,
                    conv_weight: get_bf16(&tensors, &format!("{la}.conv1d.weight"))?,
                    conv_bias: get_bf16(&tensors, &format!("{la}.conv1d.bias")).unwrap_or_default(),
                };
                // Dummy empty tensors for MHA fields (not used for GDN layers)
                let empty = QuantizedTensor {
                    data: vec![],
                    scales: vec![],
                    biases: vec![],
                    rows: 0,
                    cols: 0,
                    group_size: 1,
                    bits,
                };
                (
                    empty.clone(),
                    empty.clone(),
                    empty.clone(),
                    empty,
                    None,
                    None,
                    Some(gdn_w),
                )
            } else {
                // MHA layers: standard q_proj/k_proj/v_proj/o_proj
                let wq = get_quantized(
                    &tensors,
                    &tensor_meta,
                    &format!("{prefix}.self_attn.q_proj"),
                    group_size,
                    bits,
                )?;
                let wk = get_quantized(
                    &tensors,
                    &tensor_meta,
                    &format!("{prefix}.self_attn.k_proj"),
                    group_size,
                    bits,
                )?;
                let wv = get_quantized(
                    &tensors,
                    &tensor_meta,
                    &format!("{prefix}.self_attn.v_proj"),
                    group_size,
                    bits,
                )?;
                let wo = get_quantized(
                    &tensors,
                    &tensor_meta,
                    &format!("{prefix}.self_attn.o_proj"),
                    group_size,
                    bits,
                )?;
                let q_norm = get_bf16(&tensors, &format!("{prefix}.self_attn.q_norm.weight")).ok();
                let k_norm = get_bf16(&tensors, &format!("{prefix}.self_attn.k_norm.weight")).ok();
                (wq, wk, wv, wo, q_norm, k_norm, None)
            };

            // FFN weights (shared by both MHA and GDN layers)
            // Some models (distilled) use fused gate_up_proj instead of separate gate_proj/up_proj.
            // Same fallback chain as f32 path: dense → shared_expert (MoE) → fused
            let try_load_ffn_q =
                |pfx: &str| -> Option<(QuantizedTensor, QuantizedTensor, QuantizedTensor)> {
                    let g = get_quantized(
                        &tensors,
                        &tensor_meta,
                        &format!("{pfx}.gate_proj"),
                        group_size,
                        bits,
                    )
                    .ok()?;
                    let u = get_quantized(
                        &tensors,
                        &tensor_meta,
                        &format!("{pfx}.up_proj"),
                        group_size,
                        bits,
                    )
                    .ok()?;
                    let d = get_quantized(
                        &tensors,
                        &tensor_meta,
                        &format!("{pfx}.down_proj"),
                        group_size,
                        bits,
                    )
                    .ok()?;
                    Some((g, u, d))
                };
            let (w1, w3, w2) = if let Some(ffn) = try_load_ffn_q(&format!("{prefix}.mlp")) {
                ffn
            } else if let Some(ffn) = try_load_ffn_q(&format!("{prefix}.mlp.shared_expert")) {
                ffn
            } else {
                let fused = get_quantized(
                    &tensors,
                    &tensor_meta,
                    &format!("{prefix}.mlp.gate_up_proj"),
                    group_size,
                    bits,
                )?;
                let (g, u) = fused.split_rows_half();
                let d = get_quantized(
                    &tensors,
                    &tensor_meta,
                    &format!("{prefix}.mlp.down_proj"),
                    group_size,
                    bits,
                )?;
                (g, u, d)
            };

            let rms_att = get_bf16(&tensors, &format!("{prefix}.input_layernorm.weight"))?;
            let rms_ffn = get_bf16(
                &tensors,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )?;

            layers.push(QuantizedLayerWeights {
                wq,
                wk,
                wv,
                wo,
                w1,
                w2,
                w3,
                rms_att,
                rms_ffn,
                q_norm,
                k_norm,
                gdn,
            });

            if l == 0 {
                let ql = &layers[0];
                if ql.gdn.is_some() {
                    tracing::debug!(
                        qkv_bytes = ql.gdn.as_ref().unwrap().qkv_proj.quantized_bytes(),
                        "loaded quantized GDN layer 0",
                    );
                } else {
                    tracing::debug!(
                        wq_bytes = ql.wq.quantized_bytes(),
                        wk_bytes = ql.wk.quantized_bytes(),
                        "loaded quantized MHA layer 0",
                    );
                }
            }
        }

        let rms_final = get_bf16(&tensors, "model.norm.weight")?;

        Ok(QuantizedModelWeights {
            cfg: cfg.clone(),
            layers,
            rms_final,
            embed: embed_raw,
            vocab_size,
            lm_head: None,
            heads_per_group: hpg,
        })
    }
}

// ---------------------------------------------------------------------------
// QLoRA: quantized weight storage for low-memory training
// ---------------------------------------------------------------------------

/// Dequantize a weight matrix from N-bit packed u32 format to f32.
///
/// Handles both 4-bit (8 values per u32) and 8-bit (4 values per u32).
/// MLX stores quantized values as u32 words in little-endian byte order,
/// with values packed LSB-first within each word.
fn dequant_nbit(
    weight: &[u8],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
    bits: usize,
) -> Vec<f32> {
    let n_groups = cols / group_size;
    let elems_per_u32 = 32 / bits;
    let mask = (1u32 << bits) - 1;
    let packed_cols = cols / elems_per_u32; // u32 words per row
    let mut out = vec![0.0f32; rows * cols];

    for r in 0..rows {
        let row_byte_offset = r * packed_cols * 4; // 4 bytes per u32
        for c in 0..cols {
            let word_idx = c / elems_per_u32;
            let elem_idx = c % elems_per_u32;
            let byte_off = row_byte_offset + word_idx * 4;
            let u32_val = u32::from_le_bytes([
                weight[byte_off],
                weight[byte_off + 1],
                weight[byte_off + 2],
                weight[byte_off + 3],
            ]);
            let qval = ((u32_val >> (elem_idx * bits)) & mask) as f32;
            let g = c / group_size;
            let s = scales[r * n_groups + g];
            let b = biases[r * n_groups + g];
            out[r * cols + c] = s * qval + b;
        }
    }
    out
}

/// A single quantized weight matrix (8-bit or 4-bit with group scales/biases).
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub data: Vec<u8>,    // Raw quantized bytes (u32 words, little-endian)
    pub scales: Vec<f32>, // Per-group scales [rows * n_groups]
    pub biases: Vec<f32>, // Per-group biases [rows * n_groups]
    pub rows: usize,
    pub cols: usize, // Logical (unpacked) columns
    pub group_size: usize,
    pub bits: usize, // Quantization bits (4 or 8)
}

impl QuantizedTensor {
    /// Dequantize to full f32. Handles both 4-bit and 8-bit quantization.
    pub fn dequantize(&self) -> Vec<f32> {
        dequant_nbit(
            &self.data,
            &self.scales,
            &self.biases,
            self.rows,
            self.cols,
            self.group_size,
            self.bits,
        )
    }

    /// Memory footprint in bytes (quantized storage only).
    pub fn quantized_bytes(&self) -> usize {
        self.data.len() + (self.scales.len() + self.biases.len()) * 4
    }

    /// Split a fused tensor in half along the row dimension.
    ///
    /// Used for fused `gate_up_proj` [2*hidden_dim, dim] → gate [hidden_dim, dim] + up [hidden_dim, dim].
    pub fn split_rows_half(&self) -> (QuantizedTensor, QuantizedTensor) {
        let half_rows = self.rows / 2;
        let elems_per_u32 = 32 / self.bits;
        let packed_cols = self.cols / elems_per_u32;
        let bytes_per_row = packed_cols * 4; // 4 bytes per u32
        let data_mid = half_rows * bytes_per_row;

        let n_groups_per_row = self.cols / self.group_size;
        let scales_mid = half_rows * n_groups_per_row;

        (
            QuantizedTensor {
                data: self.data[..data_mid].to_vec(),
                scales: self.scales[..scales_mid].to_vec(),
                biases: self.biases[..scales_mid].to_vec(),
                rows: half_rows,
                cols: self.cols,
                group_size: self.group_size,
                bits: self.bits,
            },
            QuantizedTensor {
                data: self.data[data_mid..].to_vec(),
                scales: self.scales[scales_mid..].to_vec(),
                biases: self.biases[scales_mid..].to_vec(),
                rows: half_rows,
                cols: self.cols,
                group_size: self.group_size,
                bits: self.bits,
            },
        )
    }
}

/// GDN (linear attention) weights stored in quantized form.
///
/// Large projections are quantized; small parameters (A_log, dt_bias, norms,
/// conv) stay as f32 since they're tiny relative to the projections.
#[derive(Debug, Clone)]
pub struct QuantizedGdnLayerWeights {
    pub qkv_proj: QuantizedTensor, // [2*key_dim + value_dim, dim]
    pub a_proj: QuantizedTensor,   // [Hv, dim]
    pub b_proj: QuantizedTensor,   // [Hv, dim]
    pub z_proj: QuantizedTensor,   // [value_dim, dim]
    pub o_proj: QuantizedTensor,   // [dim, value_dim]
    pub a_log: Vec<f32>,           // [Hv]
    pub dt_bias: Vec<f32>,         // [Hv]
    pub norm_weight: Vec<f32>,     // [value_head_dim] shared per head or expanded [value_dim]
    pub conv_weight: Vec<f32>,     // [qkv_dim, kernel_size]
    pub conv_bias: Vec<f32>,       // [qkv_dim]
}

/// Per-layer weights stored in quantized form.
#[derive(Debug, Clone)]
pub struct QuantizedLayerWeights {
    pub wq: QuantizedTensor,
    pub wk: QuantizedTensor, // Pre-GQA-expansion dimensions
    pub wv: QuantizedTensor, // Pre-GQA-expansion dimensions
    pub wo: QuantizedTensor,
    pub w1: QuantizedTensor,
    pub w2: QuantizedTensor,
    pub w3: QuantizedTensor,
    pub rms_att: Vec<f32>,        // [dim] — always f32 (BF16 source, small)
    pub rms_ffn: Vec<f32>,        // [dim]
    pub q_norm: Option<Vec<f32>>, // [head_dim]
    pub k_norm: Option<Vec<f32>>, // [head_dim]
    /// GDN weights — `Some` for linear attention layers, `None` for MHA layers.
    pub gdn: Option<QuantizedGdnLayerWeights>,
}

impl QuantizedLayerWeights {
    /// Returns the model hidden dimension (dim) derived from actual weight tensors.
    /// Uses w2.rows since w2 is the FFN down projection: [dim, hidden_dim].
    pub fn dim(&self) -> usize {
        self.w2.rows
    }

    /// Returns the FFN intermediate dimension (hidden_dim) derived from actual weight tensors.
    /// Uses w2.cols since w2 is the FFN down projection: [dim, hidden_dim].
    pub fn hidden_dim(&self) -> usize {
        self.w2.cols
    }
}

/// Full model with quantized layer weights.
///
/// Embedding and final RMSNorm are kept in f32 (they're accessed every step
/// and are relatively small). Per-layer weights are quantized and dequantized
/// on demand during forward/backward to keep only one layer's f32 weights
/// in memory at a time.
#[derive(Debug, Clone)]
pub struct QuantizedModelWeights {
    pub cfg: MilConfig,
    pub layers: Vec<QuantizedLayerWeights>,
    pub rms_final: Vec<f32>,
    pub embed: Vec<f32>,
    pub vocab_size: usize,
    pub lm_head: Option<Vec<f32>>,
    /// GQA expansion factor: n_heads / n_kv_heads
    pub heads_per_group: usize,
}

impl QuantizedModelWeights {
    /// Dequantize a single layer's weights to f32, expanding KV for GQA.
    pub fn dequantize_layer(&self, l: usize) -> LayerWeights {
        let ql = &self.layers[l];

        // FFN weights (shared by MHA and GDN layers)
        let w1 = ql.w1.dequantize();
        let w2 = ql.w2.dequantize();
        let w3 = ql.w3.dequantize();

        // GDN layer: dequantize GDN projections, leave MHA weights empty
        if let Some(gdn_q) = &ql.gdn {
            return LayerWeights {
                wq: vec![],
                wk: vec![],
                wv: vec![],
                wo: vec![],
                w1,
                w2,
                w3,
                rms_att: ql.rms_att.clone(),
                rms_ffn: ql.rms_ffn.clone(),
                q_norm: None,
                k_norm: None,
                gdn: Some(GdnLayerWeights {
                    qkv_proj: gdn_q.qkv_proj.dequantize(),
                    a_proj: gdn_q.a_proj.dequantize(),
                    b_proj: gdn_q.b_proj.dequantize(),
                    z_proj: gdn_q.z_proj.dequantize(),
                    o_proj: gdn_q.o_proj.dequantize(),
                    a_log: gdn_q.a_log.clone(),
                    dt_bias: gdn_q.dt_bias.clone(),
                    norm_weight: gdn_q.norm_weight.clone(),
                    conv_weight: gdn_q.conv_weight.clone(),
                    conv_bias: gdn_q.conv_bias.clone(),
                }),
            };
        }

        // MHA layer: dequantize attention projections + expand KV for GQA
        let hpg = self.heads_per_group;
        let hd = self.cfg.head_dim();
        let attn_dim = self.cfg.attn_dim();

        let wq = ql.wq.dequantize();
        let wk_raw = ql.wk.dequantize();
        let wv_raw = ql.wv.dequantize();
        let wo = ql.wo.dequantize();

        // Expand KV for GQA (target is attn_dim = n_heads * head_dim, not dim)
        let wk = expand_kv_static(&wk_raw, ql.wk.rows, hd, hpg, attn_dim);
        let wv = expand_kv_static(&wv_raw, ql.wv.rows, hd, hpg, attn_dim);

        LayerWeights {
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            w3,
            rms_att: ql.rms_att.clone(),
            rms_ffn: ql.rms_ffn.clone(),
            q_norm: ql.q_norm.clone(),
            k_norm: ql.k_norm.clone(),
            gdn: None,
        }
    }

    /// Total memory footprint for quantized storage (excludes per-layer dequant buffers).
    pub fn quantized_memory_bytes(&self) -> usize {
        let layer_bytes: usize = self
            .layers
            .iter()
            .map(|l| {
                l.w1.quantized_bytes()
                    + l.w2.quantized_bytes()
                    + l.w3.quantized_bytes()
                    + (l.rms_att.len() + l.rms_ffn.len()) * 4
                    + if let Some(g) = &l.gdn {
                        g.qkv_proj.quantized_bytes()
                            + g.a_proj.quantized_bytes()
                            + g.b_proj.quantized_bytes()
                            + g.z_proj.quantized_bytes()
                            + g.o_proj.quantized_bytes()
                            + (g.a_log.len()
                                + g.dt_bias.len()
                                + g.norm_weight.len()
                                + g.conv_weight.len()
                                + g.conv_bias.len())
                                * 4
                    } else {
                        l.wq.quantized_bytes()
                            + l.wk.quantized_bytes()
                            + l.wv.quantized_bytes()
                            + l.wo.quantized_bytes()
                    }
            })
            .sum();
        let embed_bytes = self.embed.len() * 4;
        let rms_bytes = self.rms_final.len() * 4;
        layer_bytes + embed_bytes + rms_bytes
    }
}

/// Expand KV weights/activations from [kv_dim, in_dim] to [target_dim, in_dim]
/// by repeating each head_dim-sized block `hpg` times (GQA expansion).
///
/// `target_dim` should be `n_heads * head_dim` (= `MilConfig::attn_dim()`), which
/// equals `dim` for standard transformers but can be larger for models like
/// Qwen3.5 where n_heads * head_dim > dim (over-parameterised attention).
pub(crate) fn expand_kv_static(
    kv: &[f32],
    kv_dim: usize,
    head_dim: usize,
    hpg: usize,
    target_dim: usize,
) -> Vec<f32> {
    if hpg <= 1 {
        return kv.to_vec();
    }
    let n_kv = kv_dim / head_dim;
    let dim_in = kv.len() / kv_dim;
    let mut expanded = vec![0.0f32; target_dim * dim_in];
    for kv_h in 0..n_kv {
        for rep in 0..hpg {
            let dst_h = kv_h * hpg + rep;
            for d in 0..head_dim {
                let src_row = kv_h * head_dim + d;
                let dst_row = dst_h * head_dim + d;
                expanded[dst_row * dim_in..dst_row * dim_in + dim_in]
                    .copy_from_slice(&kv[src_row * dim_in..src_row * dim_in + dim_in]);
            }
        }
    }
    expanded
}

/// Trait for providing layer weights to forward/backward passes.
///
/// `ModelWeights` returns borrowed layer data (zero-copy).
/// `QuantizedModelWeights` dequantizes on demand (one layer at a time in memory).
pub trait WeightSource {
    fn cfg(&self) -> &MilConfig;
    fn cfg_mut(&mut self) -> &mut MilConfig;
    fn n_layers(&self) -> usize;
    fn layer(&self, l: usize) -> std::borrow::Cow<'_, LayerWeights>;
    fn quantized_layer(&self, _l: usize) -> Option<&QuantizedLayerWeights> {
        None
    }
    fn embed(&self) -> &[f32];
    fn rms_final(&self) -> &[f32];
    fn vocab_size(&self) -> usize;
    fn lm_head(&self) -> Option<&[f32]>;

    /// Returns the actual model hidden dimension from loaded weights.
    /// For QuantizedModelWeights, this is derived from w2.rows.
    /// For ModelWeights, this should match cfg.dim.
    fn actual_dim(&self) -> usize;

    /// Returns the actual FFN hidden dimension from loaded weights.
    /// For QuantizedModelWeights, this is derived from w2.cols.
    /// For ModelWeights, this should match cfg.hidden_dim.
    fn actual_hidden_dim(&self) -> usize;
}

impl WeightSource for ModelWeights {
    fn cfg(&self) -> &MilConfig {
        &self.cfg
    }
    fn cfg_mut(&mut self) -> &mut MilConfig {
        &mut self.cfg
    }
    fn n_layers(&self) -> usize {
        self.layers.len()
    }
    fn layer(&self, l: usize) -> std::borrow::Cow<'_, LayerWeights> {
        std::borrow::Cow::Borrowed(&self.layers[l])
    }
    fn embed(&self) -> &[f32] {
        &self.embed
    }
    fn rms_final(&self) -> &[f32] {
        &self.rms_final
    }
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn lm_head(&self) -> Option<&[f32]> {
        self.lm_head.as_deref()
    }
    fn actual_dim(&self) -> usize {
        if self.layers.is_empty() {
            return self.cfg.dim;
        }
        // Find first non-GDN layer (GDN layers have empty wo dummy tensors).
        for layer in &self.layers {
            if layer.gdn.is_none() {
                let wo_len = layer.wo.len();
                let dim = (wo_len as f64).sqrt() as usize;
                if dim * dim == wo_len {
                    return dim;
                }
            }
        }
        self.cfg.dim
    }
    fn actual_hidden_dim(&self) -> usize {
        if self.layers.is_empty() {
            return self.cfg.hidden_dim;
        }
        let dim = self.actual_dim();
        // Find first non-GDN layer for w2 dimensions.
        for layer in &self.layers {
            if layer.gdn.is_none() && !layer.w2.is_empty() {
                return layer.w2.len() / dim;
            }
        }
        self.cfg.hidden_dim
    }
}

impl WeightSource for QuantizedModelWeights {
    fn cfg(&self) -> &MilConfig {
        &self.cfg
    }
    fn cfg_mut(&mut self) -> &mut MilConfig {
        &mut self.cfg
    }
    fn n_layers(&self) -> usize {
        self.layers.len()
    }
    fn layer(&self, l: usize) -> std::borrow::Cow<'_, LayerWeights> {
        std::borrow::Cow::Owned(self.dequantize_layer(l))
    }
    fn quantized_layer(&self, l: usize) -> Option<&QuantizedLayerWeights> {
        Some(&self.layers[l])
    }
    fn embed(&self) -> &[f32] {
        &self.embed
    }
    fn rms_final(&self) -> &[f32] {
        &self.rms_final
    }
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn lm_head(&self) -> Option<&[f32]> {
        self.lm_head.as_deref()
    }
    fn actual_dim(&self) -> usize {
        if self.layers.is_empty() {
            return self.cfg.dim;
        }
        // Skip GDN layers (have empty dummy wq/wo tensors).
        for layer in &self.layers {
            if layer.gdn.is_none() {
                return layer.dim();
            }
        }
        self.cfg.dim
    }
    fn actual_hidden_dim(&self) -> usize {
        if self.layers.is_empty() {
            return self.cfg.hidden_dim;
        }
        for layer in &self.layers {
            if layer.gdn.is_none() {
                return layer.hidden_dim();
            }
        }
        self.cfg.hidden_dim
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
            gdn: None, // GDN weights are frozen, not fine-tuned via delta
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
            gdn: base.gdn.clone(), // GDN weights are frozen, preserved from base
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
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bad delta magic",
            ));
        }
        let n_layers = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;
        let dim = u32::from_le_bytes([hdr[8], hdr[9], hdr[10], hdr[11]]) as usize;
        let hidden = u32::from_le_bytes([hdr[12], hdr[13], hdr[14], hdr[15]]) as usize;
        let vocab = u32::from_le_bytes([hdr[16], hdr[17], hdr[18], hdr[19]]) as usize;

        if n_layers != base.layers.len() || dim != base.cfg.dim || hidden != base.cfg.hidden_dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "delta config mismatch",
            ));
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
                gdn: None,
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
    fp16_bytes_to_f32_neon(data)
}

/// Batch convert f32 slice to fp16 bytes.
///
/// Uses `half::slice::to_le_bytes` for efficient batch conversion.
/// Returns `Vec<u8>` of length `src.len() * 2`.
pub fn f32_to_fp16_bytes_neon(src: &[f32]) -> Vec<u8> {
    let fp16s: Vec<half::f16> = src.iter().map(|&v| half::f16::from_f32(v)).collect();
    let mut dst = vec![0u8; src.len() * 2];
    for (i, h) in fp16s.iter().enumerate() {
        dst[i * 2..i * 2 + 2].copy_from_slice(&h.to_le_bytes());
    }
    dst
}

/// Batch convert fp16 bytes to f32 vec.
///
/// Uses `half::f16` for conversion. On aarch64, the `half` crate
/// leverages hardware fp16 support for each conversion.
pub fn fp16_bytes_to_f32_neon(data: &[u8]) -> Vec<f32> {
    let n = data.len() / 2;
    let mut dst = vec![0.0f32; n];
    for j in 0..n {
        dst[j] = half::f16::from_le_bytes([data[j * 2], data[j * 2 + 1]]).to_f32();
    }
    dst
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
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
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
        let kernel = AneKernel::compile(
            &spec.mil_text,
            None,
            &[spec.input_bytes],
            &[spec.output_bytes],
        )
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
        assert!(
            max_err < 0.05,
            "pack_dyn_matmul identity max error {max_err}"
        );
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
        let (rope_cos_blob, rope_sin_blob) =
            generate_rope_blobs(seq, cfg.head_dim(), cfg.rope_theta);

        let spec = KernelSpec::for_kernel(&cfg, KernelType::SdpaFwd);
        let kernel = AneKernel::compile_multi_weights(
            &spec.mil_text,
            &[
                "@model_path/weights/mask.bin",
                "@model_path/weights/rope_cos.bin",
                "@model_path/weights/rope_sin.bin",
            ],
            &[&mask_blob, &rope_cos_blob, &rope_sin_blob],
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
        assert!(
            max_err < 0.1,
            "sdpa_fwd xnorm passthrough max error {max_err}"
        );
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
        let kernel = AneKernel::compile(
            &spec.mil_text,
            None,
            &[spec.input_bytes],
            &[spec.output_bytes],
        )
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
        let kernel = AneKernel::compile(
            &spec.mil_text,
            None,
            &[spec.input_bytes],
            &[spec.output_bytes],
        )
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
        let kernel = AneKernel::compile(
            &spec.mil_text,
            None,
            &[spec.input_bytes],
            &[spec.output_bytes],
        )
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
        let kernel = AneKernel::compile(
            &spec.mil_text,
            None,
            &[spec.input_bytes],
            &[spec.output_bytes],
        )
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

        let q: Vec<f32> = (0..dim * seq)
            .map(|i| ((i % 100) as f32 - 50.0) * 0.001)
            .collect();
        let k: Vec<f32> = (0..dim * seq)
            .map(|i| ((i % 80) as f32 - 40.0) * 0.001)
            .collect();
        let v: Vec<f32> = (0..dim * seq)
            .map(|i| ((i % 60) as f32 - 30.0) * 0.001)
            .collect();
        let da: Vec<f32> = (0..dim * seq)
            .map(|i| ((i % 50) as f32 - 25.0) * 0.001)
            .collect();

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
        let model = ModelWeights::from_llama2c(model_path, &cfg).expect("from_llama2c failed");

        assert_eq!(model.layers.len(), 12);
        assert_eq!(model.layers[0].wq.len(), 768 * 768);
        assert_eq!(model.layers[0].w1.len(), 2048 * 768);
        assert_eq!(model.layers[0].w2.len(), 768 * 2048);
        assert_eq!(model.rms_final.len(), 768);
        assert!(model.vocab_size > 0);
        assert_eq!(model.embed.len(), model.vocab_size * 768);
    }

    #[test]
    fn test_parse_mlx_checkpoint_meta_qwen3_root_layout() {
        let config = serde_json::json!({
            "num_hidden_layers": 28,
            "vocab_size": 151936,
            "quantization": {
                "group_size": 64
            }
        });

        let meta = parse_mlx_checkpoint_meta(&config).expect("should parse root-layout config");
        assert_eq!(meta.group_size, 64);
        assert_eq!(meta.n_layers, 28);
        assert_eq!(meta.vocab_size, 151936);
    }

    #[test]
    fn test_parse_mlx_checkpoint_meta_qwen3_5_text_config_layout() {
        let config = serde_json::json!({
            "model_type": "qwen3_5",
            "text_config": {
                "num_hidden_layers": 24,
                "vocab_size": 248320
            },
            "quantization_config": {
                "group_size": 128
            }
        });

        let meta = parse_mlx_checkpoint_meta(&config).expect("should parse text_config layout");
        assert_eq!(meta.group_size, 128);
        assert_eq!(meta.n_layers, 24);
        assert_eq!(meta.vocab_size, 248320);
    }

    #[test]
    fn test_resolve_weight_base_prefers_language_model_prefix_when_needed() {
        let tensors = std::collections::HashMap::from([(
            "language_model.model.embed_tokens.weight".to_string(),
            vec![1u8],
        )]);

        assert_eq!(
            resolve_weight_base(&tensors, "model.embed_tokens"),
            "language_model.model.embed_tokens"
        );
    }

    #[test]
    fn test_resolve_tensor_name_prefers_language_model_prefix_when_needed() {
        let tensors = std::collections::HashMap::from([(
            "language_model.model.norm.weight".to_string(),
            vec![1u8],
        )]);

        assert_eq!(
            resolve_tensor_name(&tensors, "model.norm.weight"),
            "language_model.model.norm.weight"
        );
    }

    #[test]
    fn test_from_mlx_safetensors_qwen3() {
        let model_dir = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/mlx-community/Qwen3-0.6B-8bit");
        if !model_dir.exists() {
            eprintln!("Qwen3-0.6B-8bit not found, skipping MLX loader test");
            return;
        }

        // Qwen3-0.6B: dim=1024, hidden=3072, 16 heads (8 KV), head_dim=128
        // BUT: 16*128=2048 ≠ 1024=dim. We can't use this model with current MIL kernels.
        // Use head_dim=64 (dim/n_heads) to match our MilConfig assumption.
        // This means KV expansion will produce wrong shapes for the real model,
        // but we can at least verify the loader parses correctly.

        // For a true E2E test, use Qwen3-1.7B (dim=2048, 16*128=2048=dim).
        let model_dir_17b = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/lmstudio-community/Qwen3-1.7B-MLX-8bit");
        if !model_dir_17b.exists() {
            eprintln!("Qwen3-1.7B-MLX-8bit not found, skipping MLX loader test");
            return;
        }

        // Qwen3-1.7B: dim=2048, hidden=6144, 16 heads, 8 KV heads, head_dim=128
        let cfg = MilConfig {
            dim: 2048,
            hidden_dim: 6144,
            n_heads: 16,
            seq_len: 32, // small seq for testing
            n_kv_heads: 8,
            rope_theta: 1_000_000.0,
            rms_eps: 1e-6,
            has_lm_head: false,
            head_dim_explicit: 2048 / 16,
            linear_attn_indices: vec![],
            linear_n_heads: 0,
            linear_head_dim: 0,
            linear_n_value_heads: 0,
            linear_value_head_dim: 0,
            conv_kernel_size: 0,
            attn_output_gate: false,
        };

        let t0 = std::time::Instant::now();
        let model = ModelWeights::from_mlx_safetensors(&model_dir_17b, &cfg)
            .expect("from_mlx_safetensors failed");
        let load_ms = t0.elapsed().as_millis();

        assert_eq!(model.layers.len(), 28);
        // After KV expansion, wk/wv should be [dim, dim]
        assert_eq!(model.layers[0].wq.len(), 2048 * 2048, "wq size");
        assert_eq!(model.layers[0].wk.len(), 2048 * 2048, "wk size (expanded)");
        assert_eq!(model.layers[0].wv.len(), 2048 * 2048, "wv size (expanded)");
        assert_eq!(model.layers[0].wo.len(), 2048 * 2048, "wo size");
        assert_eq!(model.layers[0].w1.len(), 6144 * 2048, "w1 size");
        assert_eq!(model.layers[0].w2.len(), 2048 * 6144, "w2 size");
        assert_eq!(model.rms_final.len(), 2048);

        // QK-norm should be present for Qwen3
        assert!(model.layers[0].q_norm.is_some(), "q_norm should be present");
        assert!(model.layers[0].k_norm.is_some(), "k_norm should be present");

        // Weights should be nonzero
        let sum: f32 = model.layers[0].wq.iter().take(100).map(|v| v.abs()).sum();
        assert!(sum > 0.01, "dequantized weights should be nonzero");

        eprintln!(
            "loaded Qwen3-1.7B in {load_ms}ms, {} layers, vocab={}",
            model.layers.len(),
            model.vocab_size
        );
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
            gdn: None,
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
            gdn: None,
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

    /// Verify expand_kv_static works for Qwen3.5-4B geometry where
    /// n_heads * head_dim (4096) > dim (2560).
    #[test]
    fn test_expand_kv_static_over_parameterised_attn() {
        // Qwen3.5-4B: dim=2560, n_heads=16, head_dim=256, n_kv_heads=4, hpg=4
        let head_dim = 256;
        let n_kv_heads = 4;
        let hpg = 4;
        let n_heads = n_kv_heads * hpg; // 16
        let kv_dim = n_kv_heads * head_dim; // 1024
        let attn_dim = n_heads * head_dim; // 4096 (> dim=2560)
        let seq = 3;

        // Create KV activation [kv_dim, seq] with identifiable values
        let kv: Vec<f32> = (0..kv_dim * seq).map(|i| i as f32).collect();

        let expanded = expand_kv_static(&kv, kv_dim, head_dim, hpg, attn_dim);
        assert_eq!(expanded.len(), attn_dim * seq);

        // Verify each KV head is replicated hpg times
        for kv_h in 0..n_kv_heads {
            for rep in 0..hpg {
                let dst_h = kv_h * hpg + rep;
                for d in 0..head_dim {
                    let src_row = kv_h * head_dim + d;
                    let dst_row = dst_h * head_dim + d;
                    for s in 0..seq {
                        assert_eq!(
                            expanded[dst_row * seq + s],
                            kv[src_row * seq + s],
                            "mismatch at kv_h={kv_h} rep={rep} d={d} s={s}"
                        );
                    }
                }
            }
        }
    }

    /// MilConfig::attn_dim() vs dim for standard vs over-parameterised.
    #[test]
    fn test_milconfig_attn_dim() {
        // Standard: attn_dim == dim
        let cfg = MilConfig::mha(512, 1024, 8, 64);
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.attn_dim(), 512);
        assert_eq!(cfg.attn_dim(), cfg.dim);

        // Over-parameterised (Qwen3.5-4B): attn_dim > dim
        let cfg2 = MilConfig {
            dim: 2560,
            hidden_dim: 9216,
            n_heads: 16,
            seq_len: 64,
            n_kv_heads: 4,
            rope_theta: 1e7,
            rms_eps: 1e-6,
            has_lm_head: false,
            head_dim_explicit: 256,
            linear_attn_indices: vec![],
            linear_n_heads: 0,
            linear_head_dim: 0,
            linear_n_value_heads: 0,
            linear_value_head_dim: 0,
            conv_kernel_size: 0,
            attn_output_gate: false,
        };
        assert_eq!(cfg2.head_dim(), 256);
        assert_eq!(cfg2.attn_dim(), 4096);
        assert!(cfg2.attn_dim() > cfg2.dim);
        assert_eq!(cfg2.kv_dim(), 4 * 256);
        assert_eq!(cfg2.heads_per_group(), 4);
    }
}
