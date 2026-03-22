//! Weight management for ANE dynamic kernels.
//!
//! Provides weight packing into the IOSurface layout that dynamic kernels expect:
//! `[1, channels, 1, spatial]` where spatial = seq + weight_cols.
//! Also handles model loading (llama2.c format) and delta adapters for fine-tuning.

use crate::agent::ane_mil::MilConfig;
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::Arc;

fn bench_trace_enabled() -> bool {
    std::env::var("NANOBOT_ANE_BENCH_TRACE_PHASES")
        .map(|v| v == "1")
        .unwrap_or(false)
}

fn bench_trace_weights(message: impl AsRef<str>) {
    if bench_trace_enabled() {
        eprintln!("[ANE_BENCH_TRACE] {}", message.as_ref());
    }
}

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

/// Single MoE expert FFN weights (quantized to minimize memory).
#[derive(Debug, Clone)]
pub struct MoeExpert {
    pub gate_proj: QuantizedTensor, // [moe_hidden, dim]
    pub up_proj: QuantizedTensor,   // [moe_hidden, dim]
    pub down_proj: QuantizedTensor, // [dim, moe_hidden]
}

/// Packed MoE expert weights — stores all experts as contiguous 3D tensors.
///
/// Instead of 256 individual `MoeExpert` structs (which caused 52 GB OOM when loading
/// the 35B model), this stores the raw packed data for ALL experts per projection type.
/// Individual experts are sliced on-demand during `moe_forward()`.
///
/// Memory: 3 contiguous `Vec<u8>` per layer (~400 MB total for 35B) instead of
/// 256×3 individual QuantizedTensors (~16 GB + allocation overhead).
#[derive(Debug, Clone)]
pub struct PackedMoeExperts {
    // Raw quantized data: [n_experts * rows * packed_cols] contiguous
    pub gate_data: Vec<u8>,
    pub gate_scales: Vec<f32>,
    pub gate_biases: Vec<f32>,
    pub up_data: Vec<u8>,
    pub up_scales: Vec<f32>,
    pub up_biases: Vec<f32>,
    pub down_data: Vec<u8>,
    pub down_scales: Vec<f32>,
    pub down_biases: Vec<f32>,
    // Dimensions (per expert)
    pub n_experts: usize,
    pub gate_rows: usize,  // moe_hidden (512 for 35B)
    pub gate_cols: usize,  // dim (2048 for 35B) — logical (unpacked)
    pub down_rows: usize,  // dim (2048 for 35B)
    pub down_cols: usize,  // moe_hidden (512 for 35B) — logical
    pub group_size: usize,
    pub bits: usize,
}

impl PackedMoeExperts {
    /// Compute byte/element offsets for expert `idx` into a packed projection.
    fn expert_offsets(
        expert_idx: usize,
        rows: usize,
        cols: usize,
        group_size: usize,
        bits: usize,
    ) -> (usize, usize, usize, usize) {
        let packed_cols = cols * bits / 32;
        let n_groups_per_row = (cols + group_size - 1) / group_size;
        let w_per_expert = rows * packed_cols * 4;
        let sb_per_expert = rows * n_groups_per_row;
        (
            expert_idx * w_per_expert,  // w_start (bytes)
            w_per_expert,                // w_len (bytes)
            expert_idx * sb_per_expert, // sb_start (elements)
            sb_per_expert,               // sb_len (elements)
        )
    }

    /// Dispatch to the fastest fused dequant-dot path for the quantization width.
    fn dispatch_matvec(
        data: &[u8], scales: &[f32], biases: &[f32],
        rows: usize, cols: usize, group_size: usize, bits: usize, x: &[f32],
    ) -> Vec<f32> {
        match bits {
            4 => super::ane_forward::cpu_quantized_matvec_4bit(
                data, scales, biases, rows, cols, group_size, x),
            3 => super::ane_forward::cpu_quantized_matvec_3bit(
                data, scales, biases, rows, cols, group_size, x),
            _ => super::ane_forward::cpu_quantized_matmul_raw(
                data, scales, biases, rows, cols, group_size, bits, x, 1),
        }
    }

    /// Compute gate_proj @ x for expert `idx`. Fused dequant-dot, zero allocation.
    pub fn gate_matmul(&self, expert_idx: usize, x: &[f32]) -> Vec<f32> {
        let (w_start, w_len, sb_start, sb_len) =
            Self::expert_offsets(expert_idx, self.gate_rows, self.gate_cols, self.group_size, self.bits);
        Self::dispatch_matvec(
            &self.gate_data[w_start..w_start + w_len],
            &self.gate_scales[sb_start..sb_start + sb_len],
            &self.gate_biases[sb_start..sb_start + sb_len],
            self.gate_rows, self.gate_cols, self.group_size, self.bits, x)
    }

    /// Compute up_proj @ x for expert `idx`. Fused dequant-dot, zero allocation.
    pub fn up_matmul(&self, expert_idx: usize, x: &[f32]) -> Vec<f32> {
        let (w_start, w_len, sb_start, sb_len) =
            Self::expert_offsets(expert_idx, self.gate_rows, self.gate_cols, self.group_size, self.bits);
        Self::dispatch_matvec(
            &self.up_data[w_start..w_start + w_len],
            &self.up_scales[sb_start..sb_start + sb_len],
            &self.up_biases[sb_start..sb_start + sb_len],
            self.gate_rows, self.gate_cols, self.group_size, self.bits, x)
    }

    /// Compute down_proj @ x for expert `idx`. Fused dequant-dot, zero allocation.
    pub fn down_matmul(&self, expert_idx: usize, x: &[f32]) -> Vec<f32> {
        let (w_start, w_len, sb_start, sb_len) =
            Self::expert_offsets(expert_idx, self.down_rows, self.down_cols, self.group_size, self.bits);
        Self::dispatch_matvec(
            &self.down_data[w_start..w_start + w_len],
            &self.down_scales[sb_start..sb_start + sb_len],
            &self.down_biases[sb_start..sb_start + sb_len],
            self.down_rows, self.down_cols, self.group_size, self.bits, x)
    }

    /// Dequantize gate+up projections for one expert into pre-allocated f32 buffers.
    /// Returns (gate_f32, up_f32) each of length [gate_rows * gate_cols].
    /// The cost is O(expert_size) but done once, then cblas_sgemv gets AMX acceleration.
    pub fn dequant_expert_gate_up(&self, expert_idx: usize) -> (Vec<f32>, Vec<f32>) {
        let (w_start, w_len, sb_start, sb_len) =
            Self::expert_offsets(expert_idx, self.gate_rows, self.gate_cols, self.group_size, self.bits);

        let gate_qt = QuantizedTensor {
            data: self.gate_data[w_start..w_start + w_len].to_vec(),
            scales: self.gate_scales[sb_start..sb_start + sb_len].to_vec(),
            biases: self.gate_biases[sb_start..sb_start + sb_len].to_vec(),
            rows: self.gate_rows,
            cols: self.gate_cols,
            group_size: self.group_size,
            bits: self.bits,
        };

        let up_qt = QuantizedTensor {
            data: self.up_data[w_start..w_start + w_len].to_vec(),
            scales: self.up_scales[sb_start..sb_start + sb_len].to_vec(),
            biases: self.up_biases[sb_start..sb_start + sb_len].to_vec(),
            rows: self.gate_rows,
            cols: self.gate_cols,
            group_size: self.group_size,
            bits: self.bits,
        };

        (gate_qt.dequantize(), up_qt.dequantize())
    }

    /// Dequantize down_proj for one expert.
    pub fn dequant_expert_down(&self, expert_idx: usize) -> Vec<f32> {
        let (w_start, w_len, sb_start, sb_len) =
            Self::expert_offsets(expert_idx, self.down_rows, self.down_cols, self.group_size, self.bits);

        let down_qt = QuantizedTensor {
            data: self.down_data[w_start..w_start + w_len].to_vec(),
            scales: self.down_scales[sb_start..sb_start + sb_len].to_vec(),
            biases: self.down_biases[sb_start..sb_start + sb_len].to_vec(),
            rows: self.down_rows,
            cols: self.down_cols,
            group_size: self.group_size,
            bits: self.bits,
        };

        down_qt.dequantize()
    }

    /// Memory footprint in bytes (quantized storage).
    pub fn quantized_bytes(&self) -> usize {
        self.gate_data.len() + self.up_data.len() + self.down_data.len()
            + (self.gate_scales.len() + self.gate_biases.len()
               + self.up_scales.len() + self.up_biases.len()
               + self.down_scales.len() + self.down_biases.len()) * 4
    }
}

/// MoE layer: router gate + packed experts + optional shared expert.
#[derive(Debug, Clone)]
pub struct MoeLayerWeights {
    /// Router gate: [num_experts, dim] — produces expert logits (dequantized).
    pub router: Vec<f32>,
    /// All routed experts packed as contiguous 3D tensors (zero-copy slicing).
    pub packed_experts: PackedMoeExperts,
    /// Shared expert (always active, not gated). Single expert, always in memory.
    pub shared_expert: Option<MoeExpert>,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_hidden: usize,
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
    /// MoE weights — `Some` for MoE layers, `None` for dense FFN layers.
    /// Wrapped in Arc to avoid cloning large packed expert data on dequantize.
    pub moe: Option<Arc<MoeLayerWeights>>,
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
    /// Factored vocabulary clusters for fast lm_head projection.
    /// Loaded from sidecar file `vocab_clusters.bin` next to model weights.
    pub vocab_clusters: Option<super::factored_vocab::VocabClusters>,
}

impl ModelWeights {
    /// Total f32 memory footprint in bytes.
    pub fn dense_memory_bytes(&self) -> usize {
        let layer_bytes: usize = self
            .layers
            .iter()
            .map(|l| {
                let attn = (l.wq.len() + l.wk.len() + l.wv.len() + l.wo.len()) * 4;
                let ffn = (l.w1.len() + l.w2.len() + l.w3.len()) * 4;
                let norm = (l.rms_att.len() + l.rms_ffn.len()) * 4;
                let qk = l.q_norm.as_ref().map_or(0, |v| v.len() * 4)
                    + l.k_norm.as_ref().map_or(0, |v| v.len() * 4);
                let gdn = l.gdn.as_ref().map_or(0, |g| {
                    (g.qkv_proj.len()
                        + g.a_proj.len()
                        + g.b_proj.len()
                        + g.z_proj.len()
                        + g.o_proj.len()
                        + g.a_log.len()
                        + g.dt_bias.len()
                        + g.norm_weight.len()
                        + g.conv_weight.len()
                        + g.conv_bias.len())
                        * 4
                });
                attn + ffn + norm + qk + gdn
            })
            .sum();
        let embed_bytes = self.embed.len() * 4;
        let rms_bytes = self.rms_final.len() * 4;
        let lm_head_bytes = self.lm_head.as_ref().map_or(0, |v| v.len() * 4);
        layer_bytes + embed_bytes + rms_bytes + lm_head_bytes
    }
}

#[derive(Debug, Clone, Copy)]
struct MlxCheckpointMeta {
    group_size: usize,
    bits: usize,
    n_layers: usize,
    vocab_size: usize,
}

impl MlxCheckpointMeta {
    /// Logical columns from packed u32 columns: `packed_cols * 32 / bits`.
    ///
    /// For power-of-2 bits this equals `packed_cols * (32/bits)`, but for
    /// non-power-of-2 (e.g. 3-bit) the multiply-first order avoids truncation.
    fn unpack_cols(&self, packed_cols: usize) -> usize {
        packed_cols * 32 / self.bits
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
// Memory-mapped safetensors store
// ---------------------------------------------------------------------------

/// Memory-mapped safetensors store for on-demand tensor access.
///
/// Instead of reading entire safetensors files into heap memory (via `std::fs::read`),
/// this mmaps each file and provides zero-copy access to individual tensors. MoE expert
/// weights can be skipped entirely, reducing peak memory from ~19 GB to ~4 GB for models
/// like Qwen3.5-35B where experts dominate file size.
struct MmapTensorStore {
    _mmaps: Vec<memmap2::Mmap>,
    /// tensor_name → (mmap_index, byte_start, byte_end)
    offsets: std::collections::HashMap<String, (usize, usize, usize)>,
    /// tensor_name → (dtype, shape)
    meta: std::collections::HashMap<String, (String, Vec<usize>)>,
}

impl MmapTensorStore {
    /// Open all safetensors files in `dir`, building an mmap-backed index.
    ///
    /// When `skip_experts` is true, tensors whose names contain `.experts.` are
    /// excluded from the index (their pages are never faulted in by the OS).
    fn open(dir: &Path, skip_experts: bool) -> io::Result<Self> {
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

        let mut mmaps = Vec::with_capacity(st_files.len());
        let mut offsets = HashMap::new();
        let mut meta = HashMap::new();
        let mut skipped = 0usize;

        for st_path in &st_files {
            let file = std::fs::File::open(st_path)?;
            let mmap = unsafe { memmap2::Mmap::map(&file)? };
            let mmap_idx = mmaps.len();

            if mmap.len() < 8 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small"));
            }
            let hdr_size = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
            let hdr_json: serde_json::Value = serde_json::from_slice(&mmap[8..8 + hdr_size])
                .map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("bad header: {e}"))
                })?;
            let data_start = 8 + hdr_size;

            if let serde_json::Value::Object(map) = hdr_json {
                for (name, m) in &map {
                    if name == "__metadata__" {
                        continue;
                    }
                    if skip_experts && name.contains(".experts.") {
                        skipped += 1;
                        continue;
                    }
                    let dtype = m["dtype"].as_str().unwrap_or("").to_string();
                    let shape: Vec<usize> = m["shape"]
                        .as_array()
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_u64().map(|n| n as usize))
                                .collect()
                        })
                        .unwrap_or_default();
                    let data_offsets = m["data_offsets"].as_array().unwrap();
                    let start = data_offsets[0].as_u64().unwrap() as usize;
                    let end = data_offsets[1].as_u64().unwrap() as usize;
                    offsets.insert(
                        name.clone(),
                        (mmap_idx, data_start + start, data_start + end),
                    );
                    meta.insert(name.clone(), (dtype, shape));
                }
            }

            mmaps.push(mmap);
        }

        if skipped > 0 {
            tracing::debug!("mmap tensor store: skipped {skipped} expert tensors");
        }
        tracing::debug!(
            "mmap tensor store: indexed {} tensors from {} files",
            offsets.len(),
            mmaps.len()
        );

        Ok(Self {
            _mmaps: mmaps,
            offsets,
            meta,
        })
    }

    /// Get raw bytes for a tensor (zero-copy slice into the mmap).
    fn get(&self, name: &str) -> Option<&[u8]> {
        let &(idx, start, end) = self.offsets.get(name)?;
        Some(&self._mmaps[idx][start..end])
    }

    fn contains_key(&self, name: &str) -> bool {
        self.offsets.contains_key(name)
    }

    fn meta(&self, name: &str) -> Option<&(String, Vec<usize>)> {
        self.meta.get(name)
    }

    /// Resolve weight base name, trying `language_model.` prefix fallback.
    fn resolve_weight_base(&self, base: &str) -> String {
        let direct_weight = format!("{base}.weight");
        if self.contains_key(&direct_weight) {
            return base.to_string();
        }
        let prefixed = format!("language_model.{base}");
        let prefixed_weight = format!("{prefixed}.weight");
        if self.contains_key(&prefixed_weight) {
            prefixed
        } else {
            base.to_string()
        }
    }

    /// Resolve tensor name, trying `language_model.` prefix fallback.
    fn resolve_tensor_name(&self, name: &str) -> String {
        let prefixed = format!("language_model.{name}");
        if self.contains_key(name) {
            name.to_string()
        } else if self.contains_key(&prefixed) {
            prefixed
        } else {
            name.to_string()
        }
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
pub(crate) fn build_fp16_blob(data: &[f32]) -> Vec<u8> {
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

/// Pack xnorm + W1_t + W3_t + W2_orig into `[1, dim, 1, seq + 3*hidden]` fp32 layout
/// for the fully-fused FFN kernel.
///
/// - `w1_t` and `w3_t` are ic-major `[dim, hidden]` (already transposed by caller).
/// - `w2_orig` is the ORIGINAL weight `[dim, hidden]` (out_features=dim, in_features=hidden).
pub fn pack_fused_ffn(
    xnorm: &[f32],
    w1_t: &[f32],
    w3_t: &[f32],
    w2_orig: &[f32],
    cfg: &MilConfig,
) -> Vec<u8> {
    let dim = cfg.dim;
    let hidden = cfg.hidden_dim;
    let seq = cfg.seq_len;
    let sp = seq + 3 * hidden;
    assert_eq!(xnorm.len(), dim * seq);
    assert_eq!(w1_t.len(), dim * hidden);
    assert_eq!(w3_t.len(), dim * hidden);
    assert_eq!(w2_orig.len(), dim * hidden);

    let mut buf = vec![0.0f32; dim * sp];
    for d in 0..dim {
        let row = d * sp;
        buf[row..row + seq].copy_from_slice(&xnorm[d * seq..(d + 1) * seq]);
        buf[row + seq..row + seq + hidden].copy_from_slice(&w1_t[d * hidden..(d + 1) * hidden]);
        buf[row + seq + hidden..row + seq + 2 * hidden]
            .copy_from_slice(&w3_t[d * hidden..(d + 1) * hidden]);
        buf[row + seq + 2 * hidden..row + seq + 3 * hidden]
            .copy_from_slice(&w2_orig[d * hidden..(d + 1) * hidden]);
    }
    f32_slice_to_bytes(&buf)
}

/// Unpack fused FFN output: `[1, 3*hidden+dim, 1, seq]` fp32 -> (h1, h3, gate, ffn_out).
pub fn unpack_fused_ffn(
    output: &[u8],
    cfg: &MilConfig,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let hidden = cfg.hidden_dim;
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let floats = bytes_to_f32_vec(output);
    assert_eq!(floats.len(), (3 * hidden + dim) * seq);

    let h1 = floats[0..hidden * seq].to_vec();
    let h3 = floats[hidden * seq..2 * hidden * seq].to_vec();
    let gate = floats[2 * hidden * seq..3 * hidden * seq].to_vec();
    let ffn_out = floats[3 * hidden * seq..(3 * hidden + dim) * seq].to_vec();
    (h1, h3, gate, ffn_out)
}

/// Unpacked output from fused attention GQA kernel.
pub struct FusedAttnGqaOutput {
    pub o_out: Vec<f32>,                 // [dim, seq]
    pub q: Vec<f32>,                     // [attn_dim, seq] post-RoPE
    pub k: Vec<f32>,                     // [kv_dim, seq] post-RoPE
    pub v: Vec<f32>,                     // [kv_dim, seq]
    pub attn_out: Vec<f32>,              // [attn_dim, seq] post-gate
    pub attn_pre_gate: Option<Vec<f32>>, // [attn_dim, seq] (if has_gate)
    pub attn_gate: Option<Vec<f32>>,     // [attn_dim, seq] raw gate (if has_gate)
    pub q_pre_norm: Option<Vec<f32>>,    // [attn_dim, seq] (if has_qk_norm)
    pub k_pre_norm: Option<Vec<f32>>,    // [kv_dim, seq] (if has_qk_norm)
}

/// Unpack the concatenated output of `gen_fused_attn_gqa_fwd`.
///
/// Channel layout: oo | qr_f | kr_f | v_f | ao_f [| pg_f | gr_f] [| qp_f | kp_f]
pub fn unpack_fused_attn_gqa(
    output: &[u8],
    cfg: &super::ane_mil::MilConfig,
    has_gate: bool,
    has_qk_norm: bool,
) -> FusedAttnGqaOutput {
    let dim = cfg.dim;
    let ad = cfg.attn_dim();
    let kvd = cfg.kv_dim();
    let seq = cfg.seq_len;
    let floats = bytes_to_f32_vec(output);

    let mut off = 0;
    let slice = |start: usize, ch: usize| -> Vec<f32> { floats[start..start + ch * seq].to_vec() };

    let o_out = slice(off, dim);
    off += dim * seq;
    let q = slice(off, ad);
    off += ad * seq;
    let k = slice(off, kvd);
    off += kvd * seq;
    let v = slice(off, kvd);
    off += kvd * seq;
    let attn_out = slice(off, ad);
    off += ad * seq;

    let (attn_pre_gate, attn_gate) = if has_gate {
        let pg = slice(off, ad);
        off += ad * seq;
        let gr = slice(off, ad);
        off += ad * seq;
        (Some(pg), Some(gr))
    } else {
        (None, None)
    };

    let (q_pre_norm, k_pre_norm) = if has_qk_norm {
        let qp = slice(off, ad);
        off += ad * seq;
        let kp = slice(off, kvd);
        off += kvd * seq;
        (Some(qp), Some(kp))
    } else {
        (None, None)
    };

    let _ = off; // suppress unused warning
    FusedAttnGqaOutput {
        o_out,
        q,
        k,
        v,
        attn_out,
        attn_pre_gate,
        attn_gate,
        q_pre_norm,
        k_pre_norm,
    }
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

/// Pack dq + dk + dv + Wq^T + Wk^T + Wv^T into `[1, q_proj_dim, 1, 3*seq+3*dim]` fp32 layout.
///
/// Per channel d (0..q_proj_dim):
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
    let qpd = cfg.q_proj_dim();
    let ad = cfg.attn_dim();
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let sp = 3 * seq + 3 * dim;
    assert_eq!(dq.len(), qpd * seq);
    assert_eq!(dk.len(), ad * seq);
    assert_eq!(dv.len(), ad * seq);
    assert_eq!(wqt.len(), qpd * dim);
    assert_eq!(wkt.len(), ad * dim);
    assert_eq!(wvt.len(), ad * dim);

    let mut buf = vec![0.0f32; qpd * sp];
    for d in 0..qpd {
        buf[d * sp..d * sp + seq].copy_from_slice(&dq[d * seq..d * seq + seq]);
        if d < ad {
            buf[d * sp + seq..d * sp + 2 * seq].copy_from_slice(&dk[d * seq..d * seq + seq]);
            buf[d * sp + 2 * seq..d * sp + 3 * seq].copy_from_slice(&dv[d * seq..d * seq + seq]);
        }
        buf[d * sp + 3 * seq..d * sp + 3 * seq + dim].copy_from_slice(&wqt[d * dim..d * dim + dim]);
        if d < ad {
            buf[d * sp + 3 * seq + dim..d * sp + 3 * seq + 2 * dim]
                .copy_from_slice(&wkt[d * dim..d * dim + dim]);
            buf[d * sp + 3 * seq + 2 * dim..d * sp + 3 * seq + 3 * dim]
                .copy_from_slice(&wvt[d * dim..d * dim + dim]);
        }
    }
    f32_slice_to_bytes(&buf)
}

/// Pack Q, K, V, da into `[1, 4*attn_dim, 1, seq]` fp16 layout (channel stacking).
///
/// Each slice is [attn_dim, seq] fp32, converted to fp16 and stacked channel-wise.
pub fn pack_sdpa_bwd1(q: &[f32], k: &[f32], v: &[f32], da: &[f32], cfg: &MilConfig) -> Vec<u8> {
    let attn_dim = cfg.attn_dim();
    let seq = cfg.seq_len;
    assert_eq!(q.len(), attn_dim * seq);
    assert_eq!(k.len(), attn_dim * seq);
    assert_eq!(v.len(), attn_dim * seq);
    assert_eq!(da.len(), attn_dim * seq);

    let in_ch = 4 * attn_dim;
    let mut buf = vec![0u8; in_ch * seq * 2]; // fp16
    let write_block = |buf: &mut Vec<u8>, ch_off: usize, data: &[f32]| {
        for i in 0..data.len() {
            let fp16 = half::f16::from_f32(data[i]);
            let off = (ch_off * seq + i) * 2;
            buf[off..off + 2].copy_from_slice(&fp16.to_le_bytes());
        }
    };
    write_block(&mut buf, 0, q);
    write_block(&mut buf, attn_dim, k);
    write_block(&mut buf, 2 * attn_dim, v);
    write_block(&mut buf, 3 * attn_dim, da);
    buf
}

// ---------------------------------------------------------------------------
// Output unpacking for backward kernels
// ---------------------------------------------------------------------------

/// Unpack SDPA bwd1 output: `[1, attn_dim+2*score_ch, 1, seq]` fp16 -> (dV, probs, dp).
pub fn unpack_sdpa_bwd1(output: &[u8], cfg: &MilConfig) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let attn_dim = cfg.attn_dim();
    let seq = cfg.seq_len;
    let score_ch = cfg.score_ch();
    let floats = fp16_bytes_to_f32(output);
    assert_eq!(floats.len(), (attn_dim + 2 * score_ch) * seq);

    let dv = floats[0..attn_dim * seq].to_vec();
    let probs = floats[attn_dim * seq..(attn_dim + score_ch) * seq].to_vec();
    let dp = floats[(attn_dim + score_ch) * seq..(attn_dim + 2 * score_ch) * seq].to_vec();
    (dv, probs, dp)
}

/// Unpack SDPA bwd2 output: `[1, 2*attn_dim, 1, seq]` fp16 -> (dQ, dK).
pub fn unpack_sdpa_bwd2(output: &[u8], cfg: &MilConfig) -> (Vec<f32>, Vec<f32>) {
    let attn_dim = cfg.attn_dim();
    let seq = cfg.seq_len;
    let floats = fp16_bytes_to_f32(output);
    assert_eq!(floats.len(), 2 * attn_dim * seq);

    let dq = floats[0..attn_dim * seq].to_vec();
    let dk = floats[attn_dim * seq..2 * attn_dim * seq].to_vec();
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

/// Pack one OC-tile of a DynMatmul directly from row-major `[oc, ic]` weights.
///
/// This avoids materializing a full transpose when the source tensor is stored
/// in the standard PyTorch/Safetensors `[out_features, in_features]` layout.
pub fn pack_dyn_matmul_oc_tile_row_major(
    act: &[f32],
    w_row_major: &[f32],
    ic: usize,
    full_oc: usize,
    tile_oc: usize,
    tile_start: usize,
    seq: usize,
) -> Vec<u8> {
    assert_eq!(
        w_row_major.len(),
        full_oc * ic,
        "pack_dyn_matmul_oc_tile_row_major: weight size mismatch"
    );
    let actual_oc = (full_oc - tile_start).min(tile_oc);
    let sp = seq + tile_oc;
    let mut buf = vec![0.0f32; ic * sp];
    for d in 0..ic {
        buf[d * sp..d * sp + seq].copy_from_slice(&act[d * seq..d * seq + seq]);
        for oc in 0..actual_oc {
            buf[d * sp + seq + oc] = w_row_major[(tile_start + oc) * ic + d];
        }
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
        buf[d * sp + seq..d * sp + seq + oc].copy_from_slice(&w[src_d * oc..src_d * oc + oc]);
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
                moe: None,
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
            vocab_clusters: None,
        })
    }

    /// Load weights from an MLX safetensors directory (Qwen3 architecture).
    ///
    /// Handles 8-bit quantized weights (U32-packed + BF16 scales/biases, group_size=64).
    /// Expands KV weights for GQA (replicates each KV head `heads_per_group` times).
    pub fn from_mlx_safetensors(dir: &Path, cfg: &MilConfig) -> io::Result<Self> {
        // Mmap all safetensors files, skipping MoE expert tensors to avoid OOM on
        // large models like Qwen3.5-35B (19 GB on disk, but only shared_expert used).
        let store = MmapTensorStore::open(dir, true)?;

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
        let get_weight = |base: &str, group_size: usize, bits: usize| -> io::Result<Vec<f32>> {
            let base = store.resolve_weight_base(base);
            let w_key = format!("{base}.weight");
            let s_key = format!("{base}.scales");
            let b_key = format!("{base}.biases");

            if let (Some(w), Some(s), Some(b)) =
                (store.get(&w_key), store.get(&s_key), store.get(&b_key))
            {
                let (_, shape) = store.meta(&w_key).unwrap();
                let rows = shape[0];
                // Correct for non-power-of-2 bits (3, 5, 6): multiply before dividing.
                let cols = shape[1] * 32 / bits;
                let sc = bf16_to_f32(s);
                let bi = bf16_to_f32(b);
                Ok(dequant_nbit(w, &sc, &bi, rows, cols, group_size, bits))
            } else {
                // Try non-quantized (BF16)
                if let Some(data) = store.get(&format!("{base}.weight")) {
                    let (dtype, _) = store.meta(&format!("{base}.weight")).unwrap();
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

        /// Get a small tensor directly (BF16 or F32) → Vec<f32>
        let get_bf16 = |name: &str| -> io::Result<Vec<f32>> {
            let name = store.resolve_tensor_name(name);
            let data = store.get(&name).ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, format!("missing: {name}"))
            })?;
            let dtype = store.meta(&name).map(|(d, _)| d.as_str()).unwrap_or("BF16");
            Ok(match dtype {
                "F32" => data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                _ => bf16_to_f32(data),
            })
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
        let embed_raw = get_weight("model.embed_tokens", group_size, bits)?;
        // embed_raw is [vocab, dim] row-major. embed_lookup: out[d*seq+t] = embed[tok*dim+d].
        let expected_embed = vocab_size * dim;
        if embed_raw.len() != expected_embed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "embed size mismatch: got {} f32 values, expected {} (vocab={} × dim={}). \
                     Possible {bits}-bit dequantization error.",
                    embed_raw.len(),
                    expected_embed,
                    vocab_size,
                    dim,
                ),
            ));
        }

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
                    qkv_proj: get_weight(&format!("{la}.in_proj_qkv"), group_size, bits)?,
                    a_proj: get_weight(&format!("{la}.in_proj_a"), group_size, bits)?,
                    b_proj: get_weight(&format!("{la}.in_proj_b"), group_size, bits)?,
                    z_proj: get_weight(&format!("{la}.in_proj_z"), group_size, bits)?,
                    o_proj: get_weight(&format!("{la}.out_proj"), group_size, bits)?,
                    a_log: get_bf16(&format!("{la}.A_log"))?,
                    dt_bias: get_bf16(&format!("{la}.dt_bias"))?,
                    norm_weight: get_bf16(&format!("{la}.norm.weight"))?,
                    conv_weight: get_bf16(&format!("{la}.conv1d.weight"))?,
                    conv_bias: get_bf16(&format!("{la}.conv1d.bias")).unwrap_or_default(),
                };
                // GDN layers have no separate q/k/v/o projections
                (vec![], vec![], vec![], vec![], None, None, Some(gdn_w))
            } else {
                // MHA layers: standard q_proj/k_proj/v_proj/o_proj
                let wq = get_weight(&format!("{prefix}.self_attn.q_proj"), group_size, bits)?;
                let wk_raw = get_weight(&format!("{prefix}.self_attn.k_proj"), group_size, bits)?;
                let wv_raw = get_weight(&format!("{prefix}.self_attn.v_proj"), group_size, bits)?;
                let wo = get_weight(&format!("{prefix}.self_attn.o_proj"), group_size, bits)?;
                let wk = expand_kv(&wk_raw, kv_dim, attn_dim, hpg);
                let wv = expand_kv(&wv_raw, kv_dim, attn_dim, hpg);
                let q_norm = get_bf16(&format!("{prefix}.self_attn.q_norm.weight")).ok();
                let k_norm = get_bf16(&format!("{prefix}.self_attn.k_norm.weight")).ok();
                (wq, wk, wv, wo, q_norm, k_norm, None)
            };

            // FFN weights (shared by both MHA and GDN layers)
            // Fallback chain for MLP prefix:
            //   1. mlp.gate_proj (dense models)
            //   2. mlp.shared_expert.gate_proj (MoE — train shared expert only)
            //   3. mlp.gate_up_proj (fused gate+up, split in half)
            let try_load_ffn = |pfx: &str| -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
                let g = get_weight(&format!("{pfx}.gate_proj"), group_size, bits).ok()?;
                let u = get_weight(&format!("{pfx}.up_proj"), group_size, bits).ok()?;
                let d = get_weight(&format!("{pfx}.down_proj"), group_size, bits).ok()?;
                Some((g, u, d))
            };
            let (w1, w3, w2) = if let Some(ffn) = try_load_ffn(&format!("{prefix}.mlp")) {
                ffn
            } else if let Some(ffn) = try_load_ffn(&format!("{prefix}.mlp.shared_expert")) {
                ffn
            } else {
                // Fused gate_up_proj: [2*hidden_dim, dim] → split in half by rows
                let fused = get_weight(&format!("{prefix}.mlp.gate_up_proj"), group_size, bits)?;
                let mid = fused.len() / 2;
                let d = get_weight(&format!("{prefix}.mlp.down_proj"), group_size, bits)?;
                (fused[..mid].to_vec(), fused[mid..].to_vec(), d)
            };

            // RMSNorm weights (BF16, not quantized)
            let rms_att = get_bf16(&format!("{prefix}.input_layernorm.weight"))?;
            let rms_ffn = get_bf16(&format!("{prefix}.post_attention_layernorm.weight"))?;

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
                moe: None, // MoE loaded separately
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

        let rms_final = get_bf16("model.norm.weight")?;

        Ok(ModelWeights {
            cfg: cfg.clone(),
            layers,
            rms_final,
            embed: embed_raw,
            vocab_size,
            lm_head: None, // tied embeddings
            vocab_clusters: None,
        })
    }

    /// Load MoE expert weights from safetensors into existing model.
    ///
    /// Opens the safetensors WITH expert weights (skip_experts=false) and
    /// populates each layer's `moe` field with the router gate, all experts
    /// (as QuantizedTensor — NOT dequantized), and shared expert.
    ///
    /// # Arguments
    /// - `dir`: model directory with safetensors files
    /// - `num_experts`: total experts per layer (e.g. 64)
    /// - `num_experts_per_tok`: active experts per token (e.g. 8)
    /// - `moe_hidden`: MoE intermediate dimension (from moe_intermediate_size)
    pub fn load_moe_experts(
        &mut self,
        dir: &Path,
        num_experts: usize,
        num_experts_per_tok: usize,
        moe_hidden: usize,
    ) -> io::Result<()> {
        let store = MmapTensorStore::open(dir, false)?; // DO NOT skip experts

        let config_path = dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{e}")))?;
        let meta = parse_mlx_checkpoint_meta(&config)?;
        let default_group_size = meta.group_size;
        let default_bits = meta.bits;

        // Parse per-tensor quantization overrides from config.json.
        // E.g. "language_model.model.layers.0.mlp.gate": {"bits": 8, "group_size": 64}
        let tc = config.get("text_config").unwrap_or(&config);
        let quant_section = tc.get("quantization").or_else(|| config.get("quantization"));
        let get_tensor_bits = |tensor_key: &str| -> (usize, usize) {
            if let Some(qs) = quant_section {
                // Try exact key, then with language_model. prefix
                for key in &[tensor_key.to_string(), format!("language_model.{tensor_key}")] {
                    if let Some(override_obj) = qs.get(key) {
                        let b = override_obj.get("bits").and_then(|v| v.as_u64())
                            .unwrap_or(default_bits as u64) as usize;
                        let g = override_obj.get("group_size").and_then(|v| v.as_u64())
                            .unwrap_or(default_group_size as u64) as usize;
                        return (b, g);
                    }
                }
            }
            (default_bits, default_group_size)
        };

        fn bf16_to_f32(data: &[u8]) -> Vec<f32> {
            data.chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect()
        }

        let load_quantized = |base: &str, bits: usize, group_size: usize| -> io::Result<QuantizedTensor> {
            let base = store.resolve_weight_base(base);
            let w_key = format!("{base}.weight");
            let s_key = format!("{base}.scales");
            let b_key = format!("{base}.biases");
            if let (Some(w), Some(s), Some(b)) =
                (store.get(&w_key), store.get(&s_key), store.get(&b_key))
            {
                let (_, shape) = store.meta(&w_key).unwrap();
                let rows = shape[0];
                let cols = shape[1] * 32 / bits;
                Ok(QuantizedTensor {
                    data: w.to_vec(),
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
                    format!("missing quantized: {base}"),
                ))
            }
        };

        /// Load a packed 3D tensor as contiguous data + dequantized scales/biases.
        /// Returns (data_bytes, scales_f32, biases_f32, rows_per_expert, cols_logical).
        fn load_packed_projection(
            store: &MmapTensorStore,
            key_base: &str,
            group_size: usize,
            bits: usize,
        ) -> io::Result<(Vec<u8>, Vec<f32>, Vec<f32>, usize, usize)> {
            let base = store.resolve_weight_base(key_base);
            let w_key = format!("{base}.weight");
            let s_key = format!("{base}.scales");
            let b_key = format!("{base}.biases");

            let (w, s, b) = match (store.get(&w_key), store.get(&s_key), store.get(&b_key)) {
                (Some(w), Some(s), Some(b)) => (w, s, b),
                _ => return Err(io::Error::new(io::ErrorKind::NotFound, format!("missing: {base}"))),
            };

            let (_, shape) = store.meta(&w_key).unwrap();
            // shape = [n_experts, rows_per_expert, packed_cols]
            let rows = shape[1];
            let packed_cols = shape[2];
            let cols = packed_cols * 32 / bits;

            Ok((w.to_vec(), bf16_to_f32(s), bf16_to_f32(b), rows, cols))
        }

        let load_expert_individual = |prefix: &str| -> io::Result<MoeExpert> {
            Ok(MoeExpert {
                gate_proj: load_quantized(&format!("{prefix}.gate_proj"), default_bits, default_group_size)?,
                up_proj: load_quantized(&format!("{prefix}.up_proj"), default_bits, default_group_size)?,
                down_proj: load_quantized(&format!("{prefix}.down_proj"), default_bits, default_group_size)?,
            })
        };

        let dim = self.cfg.dim;
        let mut total_expert_bytes = 0usize;

        for l in 0..self.layers.len() {
            let prefix = format!("model.layers.{l}");

            // Router gate: quantized "mlp.gate" (Qwen3.5 MLX format)
            // Router often has per-tensor override (8-bit in 3-bit models)
            let gate_key = format!("{prefix}.mlp.gate");
            let (gate_bits, gate_gs) = get_tensor_bits(&gate_key);
            let router = if let Ok(qt) = load_quantized(&gate_key, gate_bits, gate_gs) {
                qt.dequantize()
            } else {
                continue; // No router → not an MoE layer
            };

            if router.len() != num_experts * dim {
                tracing::warn!(
                    "L{l}: router size {} != expected {} — skipping MoE",
                    router.len(), num_experts * dim
                );
                continue;
            }

            // Load packed experts: ONE copy per projection (not 256 individual copies)
            // Expert weights use default bits (3-bit for 3-bit model), NOT router bits
            let switch_prefix = format!("{prefix}.mlp.switch_mlp");
            let expert_bits = default_bits;
            let expert_gs = default_group_size;

            let packed = if let Ok((gate_data, gate_scales, gate_biases, gate_rows, gate_cols)) =
                load_packed_projection(&store, &format!("{switch_prefix}.gate_proj"), expert_gs, expert_bits)
            {
                let (up_data, up_scales, up_biases, _, _) =
                    load_packed_projection(&store, &format!("{switch_prefix}.up_proj"), expert_gs, expert_bits)?;
                let (down_data, down_scales, down_biases, down_rows, down_cols) =
                    load_packed_projection(&store, &format!("{switch_prefix}.down_proj"), expert_gs, expert_bits)?;

                let bytes = gate_data.len() + up_data.len() + down_data.len();
                total_expert_bytes += bytes;

                PackedMoeExperts {
                    gate_data, gate_scales, gate_biases,
                    up_data, up_scales, up_biases,
                    down_data, down_scales, down_biases,
                    n_experts: num_experts,
                    gate_rows, gate_cols,
                    down_rows, down_cols,
                    group_size: expert_gs, bits: expert_bits,
                }
            } else {
                // Fallback: individual experts → pack them
                tracing::warn!("L{l}: no switch_mlp, falling back to individual experts");
                continue; // TODO: implement individual-to-packed conversion
            };

            // Shared expert (optional, always in memory — small)
            let shared = load_expert_individual(&format!("{prefix}.mlp.shared_expert")).ok();

            self.layers[l].moe = Some(Arc::new(MoeLayerWeights {
                router,
                packed_experts: packed,
                shared_expert: shared,
                num_experts,
                num_experts_per_tok,
                moe_hidden,
            }));

            if l == 0 {
                tracing::info!(
                    "MoE L0: {num_experts} experts loaded, {num_experts_per_tok} active/token, \
                     moe_hidden={moe_hidden}"
                );
            }
        }

        let moe_layers = self.layers.iter().filter(|l| l.moe.is_some()).count();
        tracing::info!(
            "MoE loaded: {moe_layers}/{} layers, {:.1} MB quantized expert storage",
            self.layers.len(),
            total_expert_bytes as f64 / 1e6
        );

        Ok(())
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
        // Mmap all safetensors files, skipping MoE expert tensors to avoid OOM on
        // large models like Qwen3.5-35B (19 GB on disk, but only shared_expert used).
        let store = MmapTensorStore::open(dir, true)?;

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
        let get_quantized =
            |base: &str, group_size: usize, bits: usize| -> io::Result<QuantizedTensor> {
                let base = store.resolve_weight_base(base);
                let w_key = format!("{base}.weight");
                let s_key = format!("{base}.scales");
                let b_key = format!("{base}.biases");

                if let (Some(w), Some(s), Some(b)) =
                    (store.get(&w_key), store.get(&s_key), store.get(&b_key))
                {
                    let (_, shape) = store.meta(&w_key).unwrap();
                    let rows = shape[0];
                    // Correct for non-power-of-2 bits (3, 5, 6): multiply before dividing.
                    let cols = shape[1] * 32 / bits;

                    Ok(QuantizedTensor {
                        data: w.to_vec(),
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

        let get_bf16 = |name: &str| -> io::Result<Vec<f32>> {
            let name = store.resolve_tensor_name(name);
            let data = store.get(&name).ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, format!("missing: {name}"))
            })?;
            let dtype = store.meta(&name).map(|(d, _)| d.as_str()).unwrap_or("BF16");
            Ok(match dtype {
                "F32" => data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                _ => bf16_to_f32(data),
            })
        };

        /// Dequantize a tensor to f32 (for embeddings that need random access as f32).
        let get_weight_f32 = |base: &str, group_size: usize, bits: usize| -> io::Result<Vec<f32>> {
            let base = store.resolve_weight_base(base);
            let w_key = format!("{base}.weight");
            let s_key = format!("{base}.scales");
            let b_key = format!("{base}.biases");
            if let (Some(w), Some(s), Some(b)) =
                (store.get(&w_key), store.get(&s_key), store.get(&b_key))
            {
                let (_, shape) = store.meta(&w_key).unwrap();
                let rows = shape[0];
                // Correct for non-power-of-2 bits: multiply before dividing.
                // e.g. 3-bit: packed_cols=192, 192*32/3 = 2048 (not 192*(32/3)=1920).
                let cols = shape[1] * 32 / bits;
                let sc = bf16_to_f32(s);
                let bi = bf16_to_f32(b);
                Ok(dequant_nbit(w, &sc, &bi, rows, cols, group_size, bits))
            } else if let Some(data) = store.get(&format!("{base}.weight")) {
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
        tracing::debug!(
            bits,
            group_size,
            vocab_size,
            dim = cfg.dim,
            "loading {bits}-bit embed table"
        );
        let embed_raw = get_weight_f32("model.embed_tokens", group_size, bits)?;
        let expected_embed = vocab_size * cfg.dim;
        if embed_raw.len() != expected_embed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "embed size mismatch: got {} f32 values, expected {} (vocab={} × dim={}). \
                     Possible {bits}-bit dequantization error.",
                    embed_raw.len(),
                    expected_embed,
                    vocab_size,
                    cfg.dim,
                ),
            ));
        }

        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let prefix = format!("model.layers.{l}");
            let is_gdn = cfg.is_linear_attn_layer(l);

            // Attention weights — MHA and GDN layers use different projections
            let (wq, wk, wv, wo, q_norm, k_norm, gdn) = if is_gdn {
                // GDN layers: load from linear_attn.* prefix
                let la = format!("{prefix}.linear_attn");
                let gdn_w = QuantizedGdnLayerWeights {
                    qkv_proj: get_quantized(&format!("{la}.in_proj_qkv"), group_size, bits)?,
                    a_proj: get_quantized(&format!("{la}.in_proj_a"), group_size, bits)?,
                    b_proj: get_quantized(&format!("{la}.in_proj_b"), group_size, bits)?,
                    z_proj: get_quantized(&format!("{la}.in_proj_z"), group_size, bits)?,
                    o_proj: get_quantized(&format!("{la}.out_proj"), group_size, bits)?,
                    a_log: get_bf16(&format!("{la}.A_log"))?,
                    dt_bias: get_bf16(&format!("{la}.dt_bias"))?,
                    norm_weight: get_bf16(&format!("{la}.norm.weight"))?,
                    conv_weight: get_bf16(&format!("{la}.conv1d.weight"))?,
                    conv_bias: get_bf16(&format!("{la}.conv1d.bias")).unwrap_or_default(),
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
                let wq = get_quantized(&format!("{prefix}.self_attn.q_proj"), group_size, bits)?;
                let wk = get_quantized(&format!("{prefix}.self_attn.k_proj"), group_size, bits)?;
                let wv = get_quantized(&format!("{prefix}.self_attn.v_proj"), group_size, bits)?;
                let wo = get_quantized(&format!("{prefix}.self_attn.o_proj"), group_size, bits)?;
                let q_norm = get_bf16(&format!("{prefix}.self_attn.q_norm.weight")).ok();
                let k_norm = get_bf16(&format!("{prefix}.self_attn.k_norm.weight")).ok();
                (wq, wk, wv, wo, q_norm, k_norm, None)
            };

            // FFN weights (shared by both MHA and GDN layers)
            // Some models (distilled) use fused gate_up_proj instead of separate gate_proj/up_proj.
            // Same fallback chain as f32 path: dense → shared_expert (MoE) → fused
            let try_load_ffn_q =
                |pfx: &str| -> Option<(QuantizedTensor, QuantizedTensor, QuantizedTensor)> {
                    let g = get_quantized(&format!("{pfx}.gate_proj"), group_size, bits).ok()?;
                    let u = get_quantized(&format!("{pfx}.up_proj"), group_size, bits).ok()?;
                    let d = get_quantized(&format!("{pfx}.down_proj"), group_size, bits).ok()?;
                    Some((g, u, d))
                };
            let (w1, w3, w2) = if let Some(ffn) = try_load_ffn_q(&format!("{prefix}.mlp")) {
                ffn
            } else if let Some(ffn) = try_load_ffn_q(&format!("{prefix}.mlp.shared_expert")) {
                ffn
            } else {
                let fused = get_quantized(&format!("{prefix}.mlp.gate_up_proj"), group_size, bits)?;
                let (g, u) = fused.split_rows_half();
                let d = get_quantized(&format!("{prefix}.mlp.down_proj"), group_size, bits)?;
                (g, u, d)
            };

            let rms_att = get_bf16(&format!("{prefix}.input_layernorm.weight"))?;
            let rms_ffn = get_bf16(&format!("{prefix}.post_attention_layernorm.weight"))?;

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

        let rms_final = get_bf16("model.norm.weight")?;

        let n = layers.len();
        Ok(QuantizedModelWeights {
            cfg: cfg.clone(),
            layers,
            rms_final,
            embed: embed_raw,
            vocab_size,
            lm_head: None,
            heads_per_group: hpg,
            moe: vec![None; n],
        })
    }
}

impl QuantizedModelWeights {
    /// Load MoE expert weights into the parallel `moe` vec.
    ///
    /// Same logic as `ModelWeights::load_moe_experts` but stores `Arc<MoeLayerWeights>`
    /// alongside quantized layers so `dequantize_layer` can pass them through cheaply.
    pub fn load_moe_experts(
        &mut self,
        dir: &Path,
        num_experts: usize,
        num_experts_per_tok: usize,
        moe_hidden: usize,
    ) -> io::Result<()> {
        let store = MmapTensorStore::open(dir, false)?;

        let config_path = dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{e}")))?;
        let meta = parse_mlx_checkpoint_meta(&config)?;
        let default_group_size = meta.group_size;
        let default_bits = meta.bits;

        let tc = config.get("text_config").unwrap_or(&config);
        let quant_section = tc.get("quantization").or_else(|| config.get("quantization"));
        let get_tensor_bits = |tensor_key: &str| -> (usize, usize) {
            if let Some(qs) = quant_section {
                for key in &[tensor_key.to_string(), format!("language_model.{tensor_key}")] {
                    if let Some(override_obj) = qs.get(key) {
                        let b = override_obj.get("bits").and_then(|v| v.as_u64())
                            .unwrap_or(default_bits as u64) as usize;
                        let g = override_obj.get("group_size").and_then(|v| v.as_u64())
                            .unwrap_or(default_group_size as u64) as usize;
                        return (b, g);
                    }
                }
            }
            (default_bits, default_group_size)
        };

        fn bf16_to_f32(data: &[u8]) -> Vec<f32> {
            data.chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect()
        }

        let load_quantized = |base: &str, bits: usize, group_size: usize| -> io::Result<QuantizedTensor> {
            let base = store.resolve_weight_base(base);
            let w_key = format!("{base}.weight");
            let s_key = format!("{base}.scales");
            let b_key = format!("{base}.biases");
            if let (Some(w), Some(s), Some(b)) =
                (store.get(&w_key), store.get(&s_key), store.get(&b_key))
            {
                let (_, shape) = store.meta(&w_key).unwrap();
                let rows = shape[0];
                let cols = shape[1] * 32 / bits;
                Ok(QuantizedTensor {
                    data: w.to_vec(),
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
                    format!("missing quantized: {base}"),
                ))
            }
        };

        fn load_packed_projection(
            store: &MmapTensorStore,
            key_base: &str,
            group_size: usize,
            bits: usize,
        ) -> io::Result<(Vec<u8>, Vec<f32>, Vec<f32>, usize, usize)> {
            let base = store.resolve_weight_base(key_base);
            let w_key = format!("{base}.weight");
            let s_key = format!("{base}.scales");
            let b_key = format!("{base}.biases");
            let (w, s, b) = match (store.get(&w_key), store.get(&s_key), store.get(&b_key)) {
                (Some(w), Some(s), Some(b)) => (w, s, b),
                _ => return Err(io::Error::new(io::ErrorKind::NotFound, format!("missing: {base}"))),
            };
            let (_, shape) = store.meta(&w_key).unwrap();
            let rows = shape[1];
            let packed_cols = shape[2];
            let cols = packed_cols * 32 / bits;
            Ok((w.to_vec(), bf16_to_f32(s), bf16_to_f32(b), rows, cols))
        }

        let load_expert_individual = |prefix: &str| -> io::Result<MoeExpert> {
            Ok(MoeExpert {
                gate_proj: load_quantized(&format!("{prefix}.gate_proj"), default_bits, default_group_size)?,
                up_proj: load_quantized(&format!("{prefix}.up_proj"), default_bits, default_group_size)?,
                down_proj: load_quantized(&format!("{prefix}.down_proj"), default_bits, default_group_size)?,
            })
        };

        let dim = self.cfg.dim;
        let mut total_expert_bytes = 0usize;

        for l in 0..self.layers.len() {
            let prefix = format!("model.layers.{l}");
            let gate_key = format!("{prefix}.mlp.gate");
            let (gate_bits, gate_gs) = get_tensor_bits(&gate_key);
            let router = if let Ok(qt) = load_quantized(&gate_key, gate_bits, gate_gs) {
                qt.dequantize()
            } else {
                continue;
            };
            if router.len() != num_experts * dim {
                tracing::warn!(
                    "L{l}: router size {} != expected {} — skipping MoE",
                    router.len(), num_experts * dim
                );
                continue;
            }

            let switch_prefix = format!("{prefix}.mlp.switch_mlp");
            let expert_bits = default_bits;
            let expert_gs = default_group_size;

            let packed = if let Ok((gate_data, gate_scales, gate_biases, gate_rows, gate_cols)) =
                load_packed_projection(&store, &format!("{switch_prefix}.gate_proj"), expert_gs, expert_bits)
            {
                let (up_data, up_scales, up_biases, _, _) =
                    load_packed_projection(&store, &format!("{switch_prefix}.up_proj"), expert_gs, expert_bits)?;
                let (down_data, down_scales, down_biases, down_rows, down_cols) =
                    load_packed_projection(&store, &format!("{switch_prefix}.down_proj"), expert_gs, expert_bits)?;
                total_expert_bytes += gate_data.len() + up_data.len() + down_data.len();
                PackedMoeExperts {
                    gate_data, gate_scales, gate_biases,
                    up_data, up_scales, up_biases,
                    down_data, down_scales, down_biases,
                    n_experts: num_experts,
                    gate_rows, gate_cols,
                    down_rows, down_cols,
                    group_size: expert_gs, bits: expert_bits,
                }
            } else {
                tracing::warn!("L{l}: no switch_mlp, falling back to individual experts");
                continue;
            };

            let shared = load_expert_individual(&format!("{prefix}.mlp.shared_expert")).ok();

            self.moe[l] = Some(Arc::new(MoeLayerWeights {
                router,
                packed_experts: packed,
                shared_expert: shared,
                num_experts,
                num_experts_per_tok,
                moe_hidden,
            }));

            if l == 0 {
                tracing::info!(
                    "MoE L0: {num_experts} experts loaded, {num_experts_per_tok} active/token, \
                     moe_hidden={moe_hidden}"
                );
            }
        }

        let moe_layers = self.moe.iter().filter(|m| m.is_some()).count();
        tracing::info!(
            "MoE loaded: {moe_layers}/{} layers, {:.1} MB quantized expert storage",
            self.layers.len(),
            total_expert_bytes as f64 / 1e6
        );

        Ok(())
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
pub(crate) fn dequant_nbit(
    weight: &[u8],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
    bits: usize,
) -> Vec<f32> {
    let n_groups = cols / group_size;
    let mask = (1u32 << bits) - 1;
    // Packed u32 words per row: cols * bits / 32 (exact for MLX quantization).
    let packed_cols = (cols * bits + 31) / 32;
    let mut out = vec![0.0f32; rows * cols];

    // For power-of-2 bit widths (4, 8), values never span u32 boundaries.
    // For non-power-of-2 (3, 5, 6), MLX packs bits contiguously across words.
    let spans_words = (32 % bits) != 0;

    for r in 0..rows {
        let row_byte_offset = r * packed_cols * 4; // 4 bytes per u32
        for c in 0..cols {
            let qval = if spans_words {
                // Contiguous bit packing: value c starts at absolute bit c*bits
                let bit_offset = c * bits;
                let word_idx = bit_offset / 32;
                let bit_within_word = bit_offset % 32;
                let byte_off = row_byte_offset + word_idx * 4;
                let lo_word = u32::from_le_bytes([
                    weight[byte_off],
                    weight[byte_off + 1],
                    weight[byte_off + 2],
                    weight[byte_off + 3],
                ]);
                if bit_within_word + bits <= 32 {
                    ((lo_word >> bit_within_word) & mask) as f32
                } else {
                    // Value spans two u32 words
                    let lo_bits = 32 - bit_within_word;
                    let hi_byte_off = byte_off + 4;
                    let hi_word = u32::from_le_bytes([
                        weight[hi_byte_off],
                        weight[hi_byte_off + 1],
                        weight[hi_byte_off + 2],
                        weight[hi_byte_off + 3],
                    ]);
                    let lo = lo_word >> bit_within_word;
                    let hi = hi_word & ((1u32 << (bits - lo_bits)) - 1);
                    (lo | (hi << lo_bits)) as f32
                }
            } else {
                // Non-spanning: each u32 holds exactly 32/bits values
                let elems_per_u32 = 32 / bits;
                let word_idx = c / elems_per_u32;
                let elem_idx = c % elems_per_u32;
                let byte_off = row_byte_offset + word_idx * 4;
                let u32_val = u32::from_le_bytes([
                    weight[byte_off],
                    weight[byte_off + 1],
                    weight[byte_off + 2],
                    weight[byte_off + 3],
                ]);
                ((u32_val >> (elem_idx * bits)) & mask) as f32
            };
            let g = c / group_size;
            let s = scales[r * n_groups + g];
            let b = biases[r * n_groups + g];
            out[r * cols + c] = s * qval + b;
        }
    }
    out
}

/// Quantize f32 values to N-bit packed format (inverse of `dequant_nbit`).
///
/// Returns `(packed_data, scales, biases)` matching MLX quantization format.
/// Round-trip: `dequant_nbit(quantize_nbit(v)) ≈ v` within quantization error.
pub fn quantize_nbit(
    values: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
    bits: usize,
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let max_qval = (1u32 << bits) - 1;
    let n_groups = cols / group_size;
    let packed_cols = (cols * bits + 31) / 32;

    let mut data = vec![0u8; rows * packed_cols * 4];
    let mut scales = vec![0.0f32; rows * n_groups];
    let mut biases = vec![0.0f32; rows * n_groups];

    let spans_words = (32 % bits) != 0;

    for r in 0..rows {
        for g in 0..n_groups {
            let start = r * cols + g * group_size;
            let group = &values[start..start + group_size];

            let min = group.iter().copied().fold(f32::INFINITY, f32::min);
            let max = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            let scale = if (max - min).abs() < f32::EPSILON {
                0.0
            } else {
                (max - min) / max_qval as f32
            };
            scales[r * n_groups + g] = scale;
            biases[r * n_groups + g] = min;
        }

        let row_byte_offset = r * packed_cols * 4;
        for c in 0..cols {
            let g = c / group_size;
            let scale = scales[r * n_groups + g];
            let bias = biases[r * n_groups + g];

            let qval = if scale == 0.0 {
                0u32
            } else {
                ((values[r * cols + c] - bias) / scale)
                    .round()
                    .clamp(0.0, max_qval as f32) as u32
            };

            if spans_words {
                let bit_offset = c * bits;
                let word_idx = bit_offset / 32;
                let bit_within_word = bit_offset % 32;
                let byte_off = row_byte_offset + word_idx * 4;

                let existing = u32::from_le_bytes([
                    data[byte_off],
                    data[byte_off + 1],
                    data[byte_off + 2],
                    data[byte_off + 3],
                ]);
                data[byte_off..byte_off + 4]
                    .copy_from_slice(&(existing | (qval << bit_within_word)).to_le_bytes());

                if bit_within_word + bits > 32 {
                    let hi_byte_off = byte_off + 4;
                    let hi_existing = u32::from_le_bytes([
                        data[hi_byte_off],
                        data[hi_byte_off + 1],
                        data[hi_byte_off + 2],
                        data[hi_byte_off + 3],
                    ]);
                    let hi_val = qval >> (32 - bit_within_word);
                    data[hi_byte_off..hi_byte_off + 4]
                        .copy_from_slice(&(hi_existing | hi_val).to_le_bytes());
                }
            } else {
                let elems_per_u32 = 32 / bits;
                let word_idx = c / elems_per_u32;
                let elem_idx = c % elems_per_u32;
                let byte_off = row_byte_offset + word_idx * 4;

                let existing = u32::from_le_bytes([
                    data[byte_off],
                    data[byte_off + 1],
                    data[byte_off + 2],
                    data[byte_off + 3],
                ]);
                data[byte_off..byte_off + 4]
                    .copy_from_slice(&(existing | (qval << (elem_idx * bits))).to_le_bytes());
            }
        }
    }

    (data, scales, biases)
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
    pub bits: usize, // Quantization bits (3, 4, or 8)
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
        let packed_cols = (self.cols * self.bits + 31) / 32;
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
    /// MoE weights per layer — loaded separately via `load_moe_experts`.
    pub moe: Vec<Option<Arc<MoeLayerWeights>>>,
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
                moe: self.moe[l].clone(),
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
            moe: self.moe[l].clone(),
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

    /// Dequantize all layers into a dense `ModelWeights` for fast training.
    ///
    /// Trades memory (~4× more than quantized) for speed: eliminates per-row
    /// dequantization during forward/backward, letting the training loop use
    /// dense SGEMM (with Accelerate + rayon) for ~3-5× speedup.
    pub fn to_dense(&self) -> ModelWeights {
        let layers: Vec<LayerWeights> = (0..self.layers.len())
            .map(|l| self.dequantize_layer(l))
            .collect();
        ModelWeights {
            cfg: self.cfg.clone(),
            layers,
            rms_final: self.rms_final.clone(),
            embed: self.embed.clone(),
            vocab_size: self.vocab_size,
            lm_head: self.lm_head.clone(),
            vocab_clusters: None, // not cloned — reclustered if needed
        }
    }

    /// Estimate dense f32 size of a single layer (used for memory budgeting).
    pub fn dense_layer_bytes(&self, l: usize) -> usize {
        let ql = &self.layers[l];
        let hpg = self.heads_per_group;
        let attn = if ql.gdn.is_some() {
            0 // GDN: wq/wk/wv/wo are empty in dense form
        } else {
            // wq: rows*cols (q_proj_dim * dim), wk/wv expanded for GQA
            let wq = ql.wq.rows * ql.wq.cols;
            let wk = ql.wk.rows * hpg * ql.wk.cols; // expanded
            let wv = ql.wv.rows * hpg * ql.wv.cols; // expanded
            let wo = ql.wo.rows * ql.wo.cols;
            (wq + wk + wv + wo) * 4
        };
        let ffn = (ql.w1.rows * ql.w1.cols + ql.w2.rows * ql.w2.cols + ql.w3.rows * ql.w3.cols) * 4;
        let gdn = if let Some(g) = &ql.gdn {
            (g.qkv_proj.rows * g.qkv_proj.cols
                + g.a_proj.rows * g.a_proj.cols
                + g.b_proj.rows * g.b_proj.cols
                + g.z_proj.rows * g.z_proj.cols
                + g.o_proj.rows * g.o_proj.cols
                + g.a_log.len()
                + g.dt_bias.len()
                + g.norm_weight.len()
                + g.conv_weight.len()
                + g.conv_bias.len())
                * 4
        } else {
            0
        };
        let norms = (ql.rms_att.len()
            + ql.rms_ffn.len()
            + ql.q_norm.as_ref().map_or(0, |v| v.len())
            + ql.k_norm.as_ref().map_or(0, |v| v.len()))
            * 4;
        attn + ffn + gdn + norms
    }

    pub fn total_dense_layer_bytes(&self) -> usize {
        (0..self.layers.len())
            .map(|l| self.dense_layer_bytes(l))
            .sum()
    }
}

/// Dense layer cache wrapping a quantized model for fast training.
///
/// Dequantizes layers into f32 up to a memory budget, then serves them via
/// `Cow::Borrowed` (zero-copy) to the dense forward/backward path. Layers
/// that don't fit in the budget fall back to per-row-block dequantization.
///
/// For small models (0.8B): caches all layers → ~4× speedup, ~2.9 GB.
/// For large models (35B MoE): caches what fits within the budget.
const MIB_BYTES: usize = 1024 * 1024;
const GIB_BYTES: usize = 1024 * MIB_BYTES;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseCacheBudgetPolicy {
    pub fixed_headroom_bytes: usize,
    pub inference_reserve_bytes: usize,
    pub cache_fraction: f64,
    pub explicit_budget_bytes: Option<usize>,
}

impl DenseCacheBudgetPolicy {
    pub fn split_silicon_default(quantized_bytes: usize) -> Self {
        let fixed_headroom_bytes =
            parse_env_mebibytes("NANOBOT_ANE_FIXED_HEADROOM_MB").unwrap_or(2 * GIB_BYTES);
        let inference_reserve_bytes = parse_env_mebibytes("NANOBOT_ANE_INFERENCE_RESERVE_MB")
            // Reserve enough unified memory for the foreground MLX model plus
            // some decode/KV headroom while ANE training runs in the background.
            .unwrap_or(quantized_bytes.saturating_add(GIB_BYTES));
        let cache_fraction = parse_env_fraction("NANOBOT_ANE_DENSE_CACHE_FRACTION")
            .unwrap_or(0.5)
            .clamp(0.0, 1.0);
        let explicit_budget_bytes = parse_env_mebibytes("NANOBOT_ANE_DENSE_CACHE_BUDGET_MB");

        Self {
            fixed_headroom_bytes,
            inference_reserve_bytes,
            cache_fraction,
            explicit_budget_bytes,
        }
    }

    pub fn explicit(dense_cache_budget_bytes: usize) -> Self {
        Self {
            fixed_headroom_bytes: 0,
            inference_reserve_bytes: 0,
            cache_fraction: 1.0,
            explicit_budget_bytes: Some(dense_cache_budget_bytes),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseCacheBudgetPlan {
    pub physical_mem_bytes: usize,
    pub quantized_bytes: usize,
    pub fixed_headroom_bytes: usize,
    pub inference_reserve_bytes: usize,
    pub reserved_bytes: usize,
    pub available_after_reserve_bytes: usize,
    pub total_dense_layer_bytes: usize,
    pub dense_cache_budget_bytes: usize,
    pub cache_fraction: f64,
}

impl DenseCacheBudgetPlan {
    pub fn split_silicon_default(
        physical_mem_bytes: usize,
        quantized_bytes: usize,
        total_dense_layer_bytes: usize,
    ) -> Self {
        let policy = DenseCacheBudgetPolicy::split_silicon_default(quantized_bytes);
        Self::from_policy(
            physical_mem_bytes,
            quantized_bytes,
            total_dense_layer_bytes,
            policy,
        )
    }

    pub fn explicit(
        physical_mem_bytes: usize,
        quantized_bytes: usize,
        total_dense_layer_bytes: usize,
        dense_cache_budget_bytes: usize,
    ) -> Self {
        Self::from_policy(
            physical_mem_bytes,
            quantized_bytes,
            total_dense_layer_bytes,
            DenseCacheBudgetPolicy::explicit(dense_cache_budget_bytes),
        )
    }

    fn from_policy(
        physical_mem_bytes: usize,
        quantized_bytes: usize,
        total_dense_layer_bytes: usize,
        policy: DenseCacheBudgetPolicy,
    ) -> Self {
        let reserved_bytes = quantized_bytes
            .saturating_add(policy.fixed_headroom_bytes)
            .saturating_add(policy.inference_reserve_bytes);
        let available_after_reserve_bytes = physical_mem_bytes.saturating_sub(reserved_bytes);
        let fraction_budget_bytes =
            ((available_after_reserve_bytes as f64) * policy.cache_fraction).round() as usize;
        let dense_cache_budget_bytes = policy
            .explicit_budget_bytes
            .unwrap_or(fraction_budget_bytes)
            .min(total_dense_layer_bytes);

        Self {
            physical_mem_bytes,
            quantized_bytes,
            fixed_headroom_bytes: policy.fixed_headroom_bytes,
            inference_reserve_bytes: policy.inference_reserve_bytes,
            reserved_bytes,
            available_after_reserve_bytes,
            total_dense_layer_bytes,
            dense_cache_budget_bytes,
            cache_fraction: policy.cache_fraction,
        }
    }
}

pub struct DenseCachedModel {
    quantized: QuantizedModelWeights,
    cache: Vec<Option<LayerWeights>>,
    cached_count: usize,
    cached_bytes: usize,
    cache_budget: DenseCacheBudgetPlan,
}

impl DenseCachedModel {
    /// Create a dense cache with automatic memory budgeting.
    ///
    /// Uses an explicit unified-memory budget plan:
    /// - reserve the quantized training weights
    /// - reserve foreground inference memory
    /// - reserve fixed system headroom
    /// - spend only a fraction of what remains on dense layer caching
    pub fn auto(quantized: QuantizedModelWeights) -> Self {
        let phys_mem = Self::physical_memory_bytes();
        let quantized_bytes = quantized.quantized_memory_bytes();
        let total_dense_layer_bytes = quantized.total_dense_layer_bytes();
        let budget = DenseCacheBudgetPlan::split_silicon_default(
            phys_mem,
            quantized_bytes,
            total_dense_layer_bytes,
        );
        Self::with_budget_plan(quantized, budget)
    }

    /// Create a dense cache with an explicit byte budget for layer storage.
    pub fn with_budget(quantized: QuantizedModelWeights, budget: usize) -> Self {
        let phys_mem = Self::physical_memory_bytes();
        let quantized_bytes = quantized.quantized_memory_bytes();
        let total_dense_layer_bytes = quantized.total_dense_layer_bytes();
        let budget = DenseCacheBudgetPlan::explicit(
            phys_mem,
            quantized_bytes,
            total_dense_layer_bytes,
            budget,
        );
        Self::with_budget_plan(quantized, budget)
    }

    pub fn with_budget_plan(
        quantized: QuantizedModelWeights,
        cache_budget: DenseCacheBudgetPlan,
    ) -> Self {
        let n = quantized.layers.len();
        let mut cache: Vec<Option<LayerWeights>> = (0..n).map(|_| None).collect();
        let mut used = 0usize;
        let mut cached_count = 0;

        for l in 0..n {
            let layer_bytes = quantized.dense_layer_bytes(l);
            if used + layer_bytes > cache_budget.dense_cache_budget_bytes {
                break;
            }
            cache[l] = Some(quantized.dequantize_layer(l));
            used += layer_bytes;
            cached_count += 1;
        }

        tracing::info!(
            "dense cache: {cached_count}/{n} layers cached ({:.1} MB / {:.1} MB budget, reserves {:.1} MB incl. inference {:.1} MB + headroom {:.1} MB)",
            used as f64 / 1_048_576.0,
            cache_budget.dense_cache_budget_bytes as f64 / 1_048_576.0,
            cache_budget.reserved_bytes as f64 / 1_048_576.0,
            cache_budget.inference_reserve_bytes as f64 / 1_048_576.0,
            cache_budget.fixed_headroom_bytes as f64 / 1_048_576.0,
        );

        Self {
            quantized,
            cache,
            cached_count,
            cached_bytes: used,
            cache_budget,
        }
    }

    /// Number of layers currently cached in dense form.
    pub fn cached_layer_count(&self) -> usize {
        self.cached_count
    }

    pub fn cached_bytes(&self) -> usize {
        self.cached_bytes
    }

    pub fn cache_budget(&self) -> DenseCacheBudgetPlan {
        self.cache_budget
    }

    #[cfg(target_os = "macos")]
    fn physical_memory_bytes() -> usize {
        unsafe {
            let mut size: u64 = 0;
            let mut len = std::mem::size_of::<u64>();
            let name = c"hw.memsize";
            libc::sysctlbyname(
                name.as_ptr(),
                &mut size as *mut u64 as *mut _,
                &mut len,
                std::ptr::null_mut(),
                0,
            );
            size as usize
        }
    }

    #[cfg(not(target_os = "macos"))]
    fn physical_memory_bytes() -> usize {
        // Fallback: assume 16 GB
        16 * 1024 * 1024 * 1024
    }

    /// Load MoE experts into the quantized model and refresh cached layers.
    pub fn load_moe_experts(
        &mut self,
        dir: &Path,
        num_experts: usize,
        num_experts_per_tok: usize,
        moe_hidden: usize,
    ) -> io::Result<()> {
        self.quantized.load_moe_experts(dir, num_experts, num_experts_per_tok, moe_hidden)?;
        // Patch cached layers with the newly loaded MoE data
        for l in 0..self.cache.len() {
            if let (Some(ref mut lw), Some(ref moe)) = (&mut self.cache[l], &self.quantized.moe[l]) {
                lw.moe = Some(Arc::clone(moe));
            }
        }
        Ok(())
    }
}

fn parse_env_mebibytes(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|mb| mb.saturating_mul(MIB_BYTES))
}

fn parse_env_fraction(name: &str) -> Option<f64> {
    std::env::var(name).ok().and_then(|v| v.parse::<f64>().ok())
}

impl WeightSource for DenseCachedModel {
    fn cfg(&self) -> &MilConfig {
        &self.quantized.cfg
    }
    fn cfg_mut(&mut self) -> &mut MilConfig {
        &mut self.quantized.cfg
    }
    fn n_layers(&self) -> usize {
        self.quantized.layers.len()
    }
    fn layer(&self, l: usize) -> std::borrow::Cow<'_, LayerWeights> {
        if let Some(ref lw) = self.cache[l] {
            std::borrow::Cow::Borrowed(lw)
        } else {
            std::borrow::Cow::Owned(self.quantized.dequantize_layer(l))
        }
    }
    fn quantized_layer(&self, l: usize) -> Option<&QuantizedLayerWeights> {
        // Return quantized ref only for uncached layers (forces quantized matmul path)
        if self.cache[l].is_some() {
            None
        } else {
            Some(&self.quantized.layers[l])
        }
    }
    fn embed(&self) -> &[f32] {
        &self.quantized.embed
    }
    fn rms_final(&self) -> &[f32] {
        &self.quantized.rms_final
    }
    fn vocab_size(&self) -> usize {
        self.quantized.vocab_size
    }
    fn lm_head(&self) -> Option<&[f32]> {
        self.quantized.lm_head.as_deref()
    }
    fn actual_dim(&self) -> usize {
        self.quantized.actual_dim()
    }
    fn actual_hidden_dim(&self) -> usize {
        self.quantized.actual_hidden_dim()
    }
}

// ---------------------------------------------------------------------------
// Pre-packed weight cache (Orion-style delta patching)
// ---------------------------------------------------------------------------

/// Per-layer pre-packed weight buffers for ANE IOSurface layout.
///
/// Weights are static across training steps, so we pre-transpose and pre-pack
/// them once into the IOSurface layout. Per step, only the activation slice
/// (the first `seq` columns per channel) is patched in — eliminating redundant
/// transpose + alloc + copy for the weight portion.
///
/// Saves ~2.9 GB of memory bandwidth per step on the 35B model (40 layers × fwd+bwd).
pub struct PrePackedLayerWeights {
    /// Forward fused FFN: f32 buffer [dim × (seq + 3*hidden)] with W1_t, W3_t, W2 pre-placed.
    /// Activation `xnorm` is patched into columns [0..seq] per channel.
    pub fwd_fused_ffn: Option<Vec<f32>>,

    /// Backward W2^T: f32 buffer [dim × (seq + hidden)] with W2 pre-placed.
    /// `dffn` activation is patched into columns [0..seq] per channel.
    pub bwd_w2t: Option<Vec<f32>>,

    /// Backward W1^T + W3^T: f32 buffer [hidden × (2*seq + 2*dim)] with W1, W3 pre-placed.
    /// `dh1`, `dh3` are patched into columns [0..seq] and [seq..2*seq] per channel.
    pub bwd_w13t: Option<Vec<f32>>,
}

/// QKV backward kernel variant: single (3 BLOBFILEs) or split (2 × 2 BLOBFILEs).
///
/// At 35B, WQ^T is 32MB which exceeds ANE SRAM per-BLOBFILE limit (16MB).
/// Split halves WQ^T along the reduction axis → 2 kernels of ~16MB max BLOBFILE each.
/// `dx = dx_a + dx_b` (exact math, no approximation).
pub enum QkvbKernel {
    /// Single kernel with 3 BLOBFILEs: WQ^T + WK^T + WV^T (fits in SRAM).
    Single(super::ane_bridge::AneKernel),
    /// Split into 2 kernels when WQ^T exceeds SRAM:
    ///  - half_a: WQ^T_first_half + WK^T
    ///  - half_b: WQ^T_second_half + WV^T
    Split {
        half_a: super::ane_bridge::AneKernel,
        half_b: super::ane_bridge::AneKernel,
    },
}

/// Complete set of pre-packed weights for all layers.
pub struct PrePackedWeights {
    pub layers: Vec<PrePackedLayerWeights>,
    pub seq_len: usize,
    /// Per-layer kernel clones with weights baked into IOSurface.
    /// When present, forward/backward evals use these instead of the shared
    /// bucket kernel — only activation data is patched per step via strided write,
    /// eliminating ~12MB/layer of weight copies through CPU caches.
    fwd_fused_kernels: Option<Vec<super::ane_bridge::AneKernel>>,
    bwd_w2t_kernels: Option<Vec<super::ane_bridge::AneKernel>>,
    bwd_w13t_kernels: Option<Vec<super::ane_bridge::AneKernel>>,
    /// Per-layer fused attention GQA kernels with real weights baked in.
    /// Created via `compile_multi_weights` with delta cache hits (same MIL, different weights).
    fwd_fused_attn_gqa_kernels: Option<Vec<Option<super::ane_bridge::AneKernel>>>,
    /// Per-layer fused backward attention GQA kernels with real weights baked in.
    /// Same delta-cache pattern as forward: same MIL text → cached net.plist → load-only.
    bwd_fused_attn_gqa_kernels: Option<Vec<Option<super::ane_bridge::AneKernel>>>,
    /// Per-layer Wo^T backward kernels with BLOBFILE weights (replaces DynMatmul).
    bwd_wot_kernels: Option<Vec<Option<super::ane_bridge::AneKernel>>>,
    /// Per-layer fused Wot+SDPA backward kernels (2 BLOBFILEs: Wo + mask).
    /// Merges Wot dispatch into SDPA → 2-dispatch attn backward (Wot+SDPA | QKV).
    bwd_wot_sdpa_kernels: Option<Vec<Option<super::ane_bridge::AneKernel>>>,
    /// Per-layer QKV backward kernels with BLOBFILE weights (replaces DynMatmul).
    /// Uses `QkvbKernel::Split` when WQ^T exceeds ANE SRAM per-BLOBFILE limit.
    bwd_qkvb_kernels: Option<Vec<Option<QkvbKernel>>>,
    /// Per-layer RMSNorm (attention) kernels with baked weights.
    rmsnorm_att_kernels: Option<Vec<super::ane_bridge::AneKernel>>,
    /// Per-layer RMSNorm (FFN) kernels with baked weights.
    rmsnorm_ffn_kernels: Option<Vec<super::ane_bridge::AneKernel>>,
    /// Per-layer RMSNorm backward (attention) kernels — dx only, no dw.
    rmsnorm_bwd_att_kernels: Option<Vec<super::ane_bridge::AneKernel>>,
    /// Per-layer RMSNorm backward (FFN) kernels — dx only, no dw.
    rmsnorm_bwd_ffn_kernels: Option<Vec<super::ane_bridge::AneKernel>>,
    /// Per-layer fused FFN backward kernels: W2^T + SiLU bwd + W13^T in 1 dispatch.
    fused_ffn_bwd_kernels: Option<Vec<super::ane_bridge::AneKernel>>,
    /// Shared fused FFN backward kernel with per-layer weight hotswap.
    /// Uses 1 ANE program slot for all layers (vs 40 slots for per-layer).
    fused_ffn_bwd_shared: Option<super::ane_bridge::AneKernel>,
    /// Pre-built weight blobs for FFN backward hotswap (per-layer, fp16).
    fused_ffn_bwd_weight_blobs: Option<Vec<[Vec<u8>; 3]>>,
    /// Shared split FFN backward kernel A: W2^T + SiLU bwd (1 slot, hotswap).
    split_ffn_bwd_a_shared: Option<super::ane_bridge::AneKernel>,
    /// Shared split FFN backward kernel B: W1^T + W3^T → dx (1 slot, hotswap).
    split_ffn_bwd_b_shared: Option<super::ane_bridge::AneKernel>,
    /// Pre-built weight blobs for split FFN backward A (per-layer, 1 blob: W2^T).
    split_ffn_bwd_a_blobs: Option<Vec<Vec<u8>>>,
    /// Pre-built weight blobs for split FFN backward B (per-layer, 2 blobs: W1^T, W3^T).
    split_ffn_bwd_b_blobs: Option<Vec<[Vec<u8>; 2]>>,
    /// Per-layer fused full-layer forward kernels (MHA only): entire transformer
    /// layer in 1 dispatch. Output includes packed activations for training backward.
    fused_layer_fwd_kernels: Option<Vec<super::ane_bridge::AneKernel>>,
    /// Output size per fused layer dispatch (varies: inference vs training).
    fused_layer_fwd_output_bytes: usize,
    /// Per-layer fused FFN forward kernels: RMSNorm + W1×SiLU×W3 + W2 + residual.
    fused_ffn_fwd_kernels: Option<Vec<super::ane_bridge::AneKernel>>,
    /// Output size per fused FFN dispatch.
    fused_ffn_fwd_output_bytes: usize,
    /// Per-layer fused GDN projection kernels: QKV+A+B+Z in 1 dispatch.
    fused_gdn_proj_kernels: Option<Vec<Option<super::ane_bridge::AneKernel>>>,
    /// Per-layer GDN O projection kernels with baked weights.
    gdn_o_proj_kernels: Option<Vec<Option<super::ane_bridge::AneKernel>>>,
    /// Per-layer GDN pre-recurrence kernels (fused: conv+SiLU+RMSNorm+GQA+decay+gate).
    gdn_pre_recur_per_layer: Option<Vec<Option<super::ane_bridge::AneKernel>>>,
    gdn_pre_recur_output_bytes: usize,
    gdn_pre_recur_input_bytes: usize,
    /// Chunk size for GDN pre-recurrence (0 = no chunking, kernel seq_len = full seq).
    gdn_pre_recur_chunk: usize,
    /// Conv overlap for chunked mode (kernel_size - 1).
    gdn_pre_recur_overlap: usize,
    /// Per-layer conv weights for CPU conv in chunked mode.
    /// Only populated when gdn_pre_recur_chunk > 0.
    gdn_pre_recur_conv_weights: Option<Vec<(Vec<f32>, Vec<f32>)>>, // (conv_weight, conv_bias)
}

impl PrePackedWeights {
    /// Create an empty PrePackedWeights (no FFN buffers, just attention kernel slots).
    ///
    /// Use when only attention priming is needed (e.g. tests, inference-only paths).
    pub fn build_empty(seq_len: usize, n_layers: usize) -> Self {
        let layers = (0..n_layers)
            .map(|_| PrePackedLayerWeights {
                fwd_fused_ffn: None,
                bwd_w2t: None,
                bwd_w13t: None,
            })
            .collect();
        Self {
            layers,
            seq_len,
            fwd_fused_kernels: None,
            bwd_w2t_kernels: None,
            bwd_w13t_kernels: None,
            fwd_fused_attn_gqa_kernels: None,
            bwd_fused_attn_gqa_kernels: None,
            bwd_wot_kernels: None,
            bwd_wot_sdpa_kernels: None,
            bwd_qkvb_kernels: None,
            rmsnorm_att_kernels: None,
            rmsnorm_ffn_kernels: None,
            rmsnorm_bwd_att_kernels: None,
            rmsnorm_bwd_ffn_kernels: None,
            fused_ffn_bwd_kernels: None,
            fused_ffn_bwd_shared: None,
            fused_ffn_bwd_weight_blobs: None,
            split_ffn_bwd_a_shared: None,
            split_ffn_bwd_b_shared: None,
            split_ffn_bwd_a_blobs: None,
            split_ffn_bwd_b_blobs: None,
            fused_layer_fwd_kernels: None,
            fused_layer_fwd_output_bytes: 0,
            fused_ffn_fwd_kernels: None,
            fused_ffn_fwd_output_bytes: 0,
            fused_gdn_proj_kernels: None,
            gdn_o_proj_kernels: None,
            gdn_pre_recur_per_layer: None,
            gdn_pre_recur_output_bytes: 0,
            gdn_pre_recur_input_bytes: 0,
            gdn_pre_recur_chunk: 0,
            gdn_pre_recur_overlap: 0,
            gdn_pre_recur_conv_weights: None,
        }
    }

    /// Build pre-packed weight cache from a DenseCachedModel.
    ///
    /// `seq_len` determines the activation slot size. This must match the
    /// bucket kernel's seq_len — build one PrePackedWeights per bucket.
    pub fn build(model: &DenseCachedModel, seq_len: usize, fused_ffn: bool) -> Self {
        let cfg = model.cfg();
        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;
        let n_layers = model.n_layers();
        let mut layers = Vec::with_capacity(n_layers);

        for l in 0..n_layers {
            let layer_t0 = std::time::Instant::now();
            let lw_cow = model.layer(l);
            let fetch_ms = layer_t0.elapsed().as_millis();
            let lw = &*lw_cow;
            let pack_t0 = std::time::Instant::now();

            let fwd_fused = if fused_ffn {
                // Pre-transpose W1, W3 and pack weights into fused FFN layout.
                // Layout: [dim, seq + 3*hidden] per channel d:
                //   [0..seq] = activation (patched per step)
                //   [seq..seq+hidden] = W1_t row d
                //   [seq+hidden..seq+2*hidden] = W3_t row d
                //   [seq+2*hidden..seq+3*hidden] = W2 row d
                let sp = seq_len + 3 * hidden;
                let mut buf = vec![0.0f32; dim * sp];
                for d in 0..dim {
                    let row = d * sp;
                    // W1_t: transpose from [hidden, dim] to [dim, hidden] inline
                    for h in 0..hidden {
                        buf[row + seq_len + h] = lw.w1[h * dim + d];
                    }
                    // W3_t: transpose from [hidden, dim] to [dim, hidden] inline
                    for h in 0..hidden {
                        buf[row + seq_len + hidden + h] = lw.w3[h * dim + d];
                    }
                    // W2: [dim, hidden] — direct copy (transposed inside kernel)
                    buf[row + seq_len + 2 * hidden..row + seq_len + 3 * hidden]
                        .copy_from_slice(&lw.w2[d * hidden..(d + 1) * hidden]);
                }
                Some(buf)
            } else {
                None
            };

            // Backward W2^T: layout [dim, seq + hidden]
            // W2 is [dim, hidden] — pack as weight columns
            let bwd_w2t = {
                let sp = seq_len + hidden;
                let mut buf = vec![0.0f32; dim * sp];
                for d in 0..dim {
                    let row = d * sp;
                    // W2 row d: [dim, hidden] direct
                    buf[row + seq_len..row + seq_len + hidden]
                        .copy_from_slice(&lw.w2[d * hidden..(d + 1) * hidden]);
                }
                Some(buf)
            };

            // Backward W1^T + W3^T: layout [hidden, 2*seq + 2*dim]
            // W1 is [hidden, dim], W3 is [hidden, dim] — these ARE the transposed forms
            let bwd_w13t = {
                let sp = 2 * seq_len + 2 * dim;
                let mut buf = vec![0.0f32; hidden * sp];
                for d in 0..hidden {
                    let row = d * sp;
                    // W1 row d = W1^T column d: [hidden, dim] direct
                    buf[row + 2 * seq_len..row + 2 * seq_len + dim]
                        .copy_from_slice(&lw.w1[d * dim..(d + 1) * dim]);
                    // W3 row d = W3^T column d
                    buf[row + 2 * seq_len + dim..row + 2 * seq_len + 2 * dim]
                        .copy_from_slice(&lw.w3[d * dim..(d + 1) * dim]);
                }
                Some(buf)
            };

            layers.push(PrePackedLayerWeights {
                fwd_fused_ffn: fwd_fused,
                bwd_w2t,
                bwd_w13t,
            });
            bench_trace_weights(format!(
                "prepacked_build:layer layer={} cached={} fetch_ms={} pack_ms={} total_ms={}",
                l,
                model.cache[l].is_some(),
                fetch_ms,
                pack_t0.elapsed().as_millis(),
                layer_t0.elapsed().as_millis()
            ));
        }

        tracing::info!(
            "pre-packed weights: {} layers, seq_len={}, fused_ffn={}",
            n_layers,
            seq_len,
            fused_ffn,
        );

        Self {
            layers,
            seq_len,
            fwd_fused_kernels: None,
            bwd_w2t_kernels: None,
            bwd_w13t_kernels: None,
            fwd_fused_attn_gqa_kernels: None,
            bwd_fused_attn_gqa_kernels: None,
            bwd_wot_kernels: None,
            bwd_wot_sdpa_kernels: None,
            bwd_qkvb_kernels: None,
            rmsnorm_att_kernels: None,
            rmsnorm_ffn_kernels: None,
            rmsnorm_bwd_att_kernels: None,
            rmsnorm_bwd_ffn_kernels: None,
            fused_ffn_bwd_kernels: None,
            fused_ffn_bwd_shared: None,
            fused_ffn_bwd_weight_blobs: None,
            split_ffn_bwd_a_shared: None,
            split_ffn_bwd_b_shared: None,
            split_ffn_bwd_a_blobs: None,
            split_ffn_bwd_b_blobs: None,
            fused_layer_fwd_kernels: None,
            fused_layer_fwd_output_bytes: 0,
            fused_ffn_fwd_kernels: None,
            fused_ffn_fwd_output_bytes: 0,
            fused_gdn_proj_kernels: None,
            gdn_o_proj_kernels: None,
            gdn_pre_recur_per_layer: None,
            gdn_pre_recur_output_bytes: 0,
            gdn_pre_recur_input_bytes: 0,
            gdn_pre_recur_chunk: 0,
            gdn_pre_recur_overlap: 0,
            gdn_pre_recur_conv_weights: None,
        }
    }

    /// Patch activation into a pre-packed fused FFN buffer and return bytes for kernel input.
    ///
    /// Only writes the activation slice (columns 0..seq per channel), leaving
    /// pre-packed weights untouched. Returns the buffer as bytes for `write_input`.
    #[inline]
    pub fn patch_fwd_fused_ffn(&mut self, layer: usize, xnorm: &[f32], dim: usize) -> &[u8] {
        let buf = self.layers[layer]
            .fwd_fused_ffn
            .as_mut()
            .expect("fwd_fused_ffn not pre-packed");
        let seq = self.seq_len;
        let sp = buf.len() / dim;
        for d in 0..dim {
            buf[d * sp..d * sp + seq].copy_from_slice(&xnorm[d * seq..d * seq + seq]);
        }
        // Safety: f32 slice → u8 slice (same backing memory, no alloc)
        unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 4) }
    }

    /// Patch activation into a pre-packed backward W2^T buffer.
    #[inline]
    pub fn patch_bwd_w2t(&mut self, layer: usize, dffn: &[f32], dim: usize) -> &[u8] {
        let buf = self.layers[layer]
            .bwd_w2t
            .as_mut()
            .expect("bwd_w2t not pre-packed");
        let seq = self.seq_len;
        let sp = buf.len() / dim;
        for d in 0..dim {
            buf[d * sp..d * sp + seq].copy_from_slice(&dffn[d * seq..d * seq + seq]);
        }
        unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 4) }
    }

    /// Patch activations into a pre-packed backward W13^T buffer.
    #[inline]
    pub fn patch_bwd_w13t(
        &mut self,
        layer: usize,
        dh1: &[f32],
        dh3: &[f32],
        hidden: usize,
    ) -> &[u8] {
        let buf = self.layers[layer]
            .bwd_w13t
            .as_mut()
            .expect("bwd_w13t not pre-packed");
        let seq = self.seq_len;
        let sp = buf.len() / hidden;
        for d in 0..hidden {
            buf[d * sp..d * sp + seq].copy_from_slice(&dh1[d * seq..d * seq + seq]);
            buf[d * sp + seq..d * sp + 2 * seq].copy_from_slice(&dh3[d * seq..d * seq + seq]);
        }
        unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 4) }
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| {
                l.fwd_fused_ffn.as_ref().map_or(0, |b| b.len() * 4)
                    + l.bwd_w2t.as_ref().map_or(0, |b| b.len() * 4)
                    + l.bwd_w13t.as_ref().map_or(0, |b| b.len() * 4)
            })
            .sum()
    }

    /// Returns true if fused full-layer forward kernels have been primed.
    pub fn has_fused_layer_fwd(&self) -> bool {
        self.fused_layer_fwd_kernels.is_some()
    }

    /// Returns true if per-layer kernels have been primed.
    pub fn has_per_layer_kernels(&self) -> bool {
        self.fwd_fused_kernels.is_some()
            || self.fwd_fused_attn_gqa_kernels.is_some()
            || self.bwd_fused_attn_gqa_kernels.is_some()
    }

    /// Drop per-layer IOSurface kernels to free memory bandwidth for inference.
    /// The heap-resident pre-packed layer buffers (`fwd_fused_ffn`, `bwd_w2t`,
    /// `bwd_w13t`) are retained — they are cheap to keep and avoid re-transposing
    /// weights from the dense cache. Call `prime_kernels()` again to rebuild.
    pub fn evict_kernels(&mut self) {
        self.fwd_fused_kernels = None;
        self.bwd_w2t_kernels = None;
        self.bwd_w13t_kernels = None;
        self.fwd_fused_attn_gqa_kernels = None;
        self.bwd_fused_attn_gqa_kernels = None;
        self.bwd_wot_kernels = None;
        self.bwd_wot_sdpa_kernels = None;
        self.bwd_qkvb_kernels = None;
        self.rmsnorm_att_kernels = None;
        self.rmsnorm_ffn_kernels = None;
        self.rmsnorm_bwd_att_kernels = None;
        self.rmsnorm_bwd_ffn_kernels = None;
        self.fused_ffn_bwd_kernels = None;
        self.fused_ffn_bwd_shared = None;
        self.fused_ffn_bwd_weight_blobs = None;
        self.split_ffn_bwd_a_shared = None;
        self.split_ffn_bwd_b_shared = None;
        self.split_ffn_bwd_a_blobs = None;
        self.split_ffn_bwd_b_blobs = None;
        self.fused_layer_fwd_kernels = None;
        self.fused_ffn_fwd_kernels = None;
        self.fused_gdn_proj_kernels = None;
        self.gdn_o_proj_kernels = None;
        self.gdn_pre_recur_per_layer = None;
    }

    /// Prime per-layer kernel clones with IOSurface-resident weights.
    ///
    /// Clones the template kernels once per layer and writes each layer's
    /// pre-packed weights into the clone's IOSurface. After priming, only
    /// activation data needs to be patched per step via strided write.
    ///
    /// `fwd_template`: the FullyFused FFN kernel (or None if not fused)
    /// `bwd_w2t_template`: backward W2^T kernel (or None)
    /// `bwd_w13t_template`: backward W1^T+W3^T kernel (or None)
    pub fn prime_kernels(
        &mut self,
        fwd_template: Option<&super::ane_bridge::AneKernel>,
        bwd_w2t_template: Option<&super::ane_bridge::AneKernel>,
        bwd_w13t_template: Option<&super::ane_bridge::AneKernel>,
    ) -> Result<(), String> {
        let n_layers = self.layers.len();

        if let Some(tmpl) = fwd_template {
            let mut kernels = Vec::with_capacity(n_layers);
            for (l, lw) in self.layers.iter().enumerate() {
                let Some(ref buf) = lw.fwd_fused_ffn else {
                    return Err(format!("layer {l}: fwd_fused_ffn not pre-packed"));
                };
                let k = tmpl
                    .clone_kernel()
                    .map_err(|e| format!("layer {l} fwd clone: {e}"))?;
                let bytes =
                    unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 4) };
                k.write_input(0, bytes);
                kernels.push(k);
            }
            tracing::info!(
                "primed {} per-layer fwd fused FFN kernels ({:.1} MB IOSurface)",
                n_layers,
                self.layers
                    .iter()
                    .filter_map(|l| l.fwd_fused_ffn.as_ref())
                    .map(|b| b.len() * 4)
                    .sum::<usize>() as f64
                    / 1_048_576.0,
            );
            self.fwd_fused_kernels = Some(kernels);
        }

        if let Some(tmpl) = bwd_w2t_template {
            let mut kernels = Vec::with_capacity(n_layers);
            for (l, lw) in self.layers.iter().enumerate() {
                let Some(ref buf) = lw.bwd_w2t else {
                    return Err(format!("layer {l}: bwd_w2t not pre-packed"));
                };
                let k = tmpl
                    .clone_kernel()
                    .map_err(|e| format!("layer {l} bwd_w2t clone: {e}"))?;
                let bytes =
                    unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 4) };
                k.write_input(0, bytes);
                kernels.push(k);
            }
            tracing::info!("primed {} per-layer bwd W2^T kernels", n_layers);
            self.bwd_w2t_kernels = Some(kernels);
        }

        if let Some(tmpl) = bwd_w13t_template {
            let mut kernels = Vec::with_capacity(n_layers);
            for (l, lw) in self.layers.iter().enumerate() {
                let Some(ref buf) = lw.bwd_w13t else {
                    return Err(format!("layer {l}: bwd_w13t not pre-packed"));
                };
                let k = tmpl
                    .clone_kernel()
                    .map_err(|e| format!("layer {l} bwd_w13t clone: {e}"))?;
                let bytes =
                    unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 4) };
                k.write_input(0, bytes);
                kernels.push(k);
            }
            tracing::info!("primed {} per-layer bwd W1^T+W3^T kernels", n_layers);
            self.bwd_w13t_kernels = Some(kernels);
        }

        Ok(())
    }

    /// Prime per-layer fused attention GQA kernels with real weights.
    ///
    /// Unlike FFN (weights packed into input IOSurface), the fused attention kernel
    /// stores weights as BLOBFILE constants compiled into the model. Per-layer kernels
    /// are created via `compile_multi_weights` — each call hits the delta compilation
    /// cache (same MIL text → same hexId → cached net.plist), so only loadWithQoS
    /// happens. Cost: ~17ms × N layers one-time.
    pub fn prime_attn_kernels<W: WeightSource>(
        &mut self,
        cfg: &super::ane_mil::MilConfig,
        model: &W,
        rope_cos_blob: &[u8],
        rope_sin_blob: &[u8],
        mask_blob: &[u8],
    ) -> Result<(), String> {
        let n_layers = model.n_layers();
        let dim = cfg.dim;
        let qpd = cfg.q_proj_dim();
        let kv_dim = cfg.kv_dim();
        let attn_dim = cfg.attn_dim();

        // Fused attn fwd has WQ [dim,qpd], WK [dim,kv], WV [dim,kv], Wo [ad,dim].
        // If the largest blob exceeds ANE SRAM, skip silently.
        let max_blob_bytes = (dim * qpd).max(dim * attn_dim) * 2;
        if max_blob_bytes > Self::ANE_MAX_BLOBFILE_BYTES {
            tracing::debug!(
                "skipping fused attn fwd: max blob {:.1}MB exceeds SRAM limit",
                max_blob_bytes as f64 / (1024.0 * 1024.0),
            );
            self.fwd_fused_attn_gqa_kernels = Some((0..n_layers).map(|_| None).collect());
            return Ok(());
        }

        let has_qk_norm = true; // QK-norm via reduce_mean(axis=-1) in [1,H,S,hd]
        let result = super::ane_mil::gen_fused_attn_gqa_fwd(cfg, has_qk_norm);

        let mut kernels = Vec::with_capacity(n_layers);
        let t0 = std::time::Instant::now();

        for l in 0..n_layers {
            let lw_cow = model.layer(l);
            let lw = &*lw_cow;

            // GDN (linear attention) layers have no Q/K/V/O — skip
            if lw.gdn.is_some() || lw.wq.is_empty() {
                kernels.push(None);
                continue;
            }

            // Un-expand GQA: wk/wv are stored expanded [attn_dim, dim] but MIL
            // wants compact [kv_dim, dim]. Extract head 0 from each group.
            let hd = cfg.head_dim();
            let hpg = cfg.heads_per_group();
            let unexpand_kv = |expanded: &[f32]| -> Vec<f32> {
                if hpg <= 1 {
                    return expanded.to_vec();
                }
                let n_kv = cfg.n_kv_heads;
                let mut compact = Vec::with_capacity(kv_dim * dim);
                for kv_h in 0..n_kv {
                    let src_h = kv_h * hpg; // first head in group
                    for d in 0..hd {
                        let row = src_h * hd + d;
                        compact.extend_from_slice(&expanded[row * dim..(row + 1) * dim]);
                    }
                }
                compact
            };

            // Build per-layer weight blobs: transpose to MIL layout then fp16
            let blobs: Vec<Vec<u8>> = result
                .weight_names
                .iter()
                .map(|name| match *name {
                    "@model_path/weights/rope_cos.bin" => rope_cos_blob.to_vec(),
                    "@model_path/weights/rope_sin.bin" => rope_sin_blob.to_vec(),
                    "@model_path/weights/mask.bin" => mask_blob.to_vec(),
                    "@model_path/weights/wq.bin" => {
                        // wq stored as [qpd, dim], MIL needs [dim, qpd]
                        let wq_t = transpose_weight(&lw.wq, qpd, dim);
                        build_fp16_blob(&wq_t)
                    }
                    "@model_path/weights/wk.bin" => {
                        // wk stored expanded [attn_dim, dim], un-expand → [kv_dim, dim], transpose → [dim, kv_dim]
                        let wk_compact = unexpand_kv(&lw.wk);
                        let wk_t = transpose_weight(&wk_compact, kv_dim, dim);
                        build_fp16_blob(&wk_t)
                    }
                    "@model_path/weights/wv.bin" => {
                        // wv stored expanded [attn_dim, dim], un-expand → [kv_dim, dim], transpose → [dim, kv_dim]
                        let wv_compact = unexpand_kv(&lw.wv);
                        let wv_t = transpose_weight(&wv_compact, kv_dim, dim);
                        build_fp16_blob(&wv_t)
                    }
                    "@model_path/weights/wo.bin" => {
                        // wo stored as [dim, attn_dim], MIL needs [attn_dim, dim]
                        let wo_t = transpose_weight(&lw.wo, dim, attn_dim);
                        build_fp16_blob(&wo_t)
                    }
                    "@model_path/weights/q_norm.bin" => {
                        build_fp16_blob(lw.q_norm.as_deref().unwrap_or(&vec![1.0f32; hd]))
                    }
                    "@model_path/weights/k_norm.bin" => {
                        build_fp16_blob(lw.k_norm.as_deref().unwrap_or(&vec![1.0f32; hd]))
                    }
                    _ => build_fp16_blob(&vec![0.0f32; 16]),
                })
                .collect();

            let names: Vec<&str> = result.weight_names.iter().copied().collect();
            let datas: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();

            let k = super::ane_bridge::AneKernel::compile_multi_weights(
                &result.mil_text,
                &names,
                &datas,
                &[result.input_bytes],
                &[result.output_bytes],
            )
            .map_err(|e| format!("layer {l} fused_attn_gqa compile: {e}"))?;

            kernels.push(Some(k));
        }

        let elapsed = t0.elapsed();
        let attn_layers = kernels.iter().filter(|k| k.is_some()).count();
        tracing::info!(
            "primed {attn_layers}/{n_layers} per-layer fused attention GQA kernels in {:.1}ms ({:.1}ms/layer)",
            elapsed.as_secs_f64() * 1000.0,
            if attn_layers > 0 { elapsed.as_secs_f64() * 1000.0 / attn_layers as f64 } else { 0.0 },
        );
        self.fwd_fused_attn_gqa_kernels = Some(kernels);
        Ok(())
    }

    /// Evaluate fused attention GQA on per-layer kernel (weights baked into compiled model).
    ///
    /// Only writes the input activation `xnorm` [dim, seq]. Returns None if
    /// per-layer attention kernels aren't primed.
    pub fn eval_fwd_fused_attn_gqa(
        &self,
        layer: usize,
        xnorm: &[f32],
        cfg: &super::ane_mil::MilConfig,
    ) -> Option<Result<FusedAttnGqaOutput, String>> {
        let kernels = self.fwd_fused_attn_gqa_kernels.as_ref()?;
        let kernel = kernels[layer].as_ref()?;

        let xnorm_bytes =
            unsafe { std::slice::from_raw_parts(xnorm.as_ptr() as *const u8, xnorm.len() * 4) };
        kernel.write_input(0, xnorm_bytes);

        if let Err(e) = kernel.eval() {
            return Some(Err(format!(
                "per-layer fused_attn_gqa eval layer {layer}: {e}"
            )));
        }

        let ad = cfg.attn_dim();
        let kvd = cfg.kv_dim();
        let has_gate = cfg.attn_output_gate;
        let out_ch = if has_gate {
            cfg.dim + 4 * ad + 2 * kvd
        } else {
            cfg.dim + 2 * ad + 2 * kvd
        };
        let mut buf = vec![0u8; out_ch * cfg.seq_len * 4];
        kernel.read_output(0, &mut buf);
        Some(Ok(unpack_fused_attn_gqa(&buf, cfg, has_gate, false)))
    }

    /// Prime per-layer fused backward attention GQA kernels with real weights.
    ///
    /// Same pattern as `prime_attn_kernels()`: per-layer weight blobs compiled via
    /// `compile_multi_weights` hitting delta cache. Cost: ~17ms × N layers one-time.
    pub fn prime_bwd_attn_kernels<W: WeightSource>(
        &mut self,
        cfg: &super::ane_mil::MilConfig,
        model: &W,
        rope_cos_blob: &[u8],
        rope_sin_blob: &[u8],
        mask_blob: &[u8],
    ) -> Result<(), String> {
        let n_layers = model.n_layers();
        let dim = cfg.dim;
        let qpd = cfg.q_proj_dim();
        let kv_dim = cfg.kv_dim();
        let attn_dim = cfg.attn_dim();

        // Fused attn bwd has WQ [qpd,dim], WK [kv,dim], WV [kv,dim], Wo [dim,ad].
        // If the largest blob exceeds ANE SRAM, skip silently.
        let max_blob_bytes = (dim * qpd).max(dim * attn_dim) * 2;
        if max_blob_bytes > Self::ANE_MAX_BLOBFILE_BYTES {
            tracing::debug!(
                "skipping fused attn bwd: max blob {:.1}MB exceeds SRAM limit",
                max_blob_bytes as f64 / (1024.0 * 1024.0),
            );
            self.bwd_fused_attn_gqa_kernels = Some((0..n_layers).map(|_| None).collect());
            return Ok(());
        }

        let has_qk_norm = false; // backward QK-norm not yet ported to axis=-1 approach
        let result = super::ane_mil::gen_fused_attn_gqa_bwd(cfg, has_qk_norm);

        let mut kernels = Vec::with_capacity(n_layers);
        let t0 = std::time::Instant::now();

        for l in 0..n_layers {
            let lw_cow = model.layer(l);
            let lw = &*lw_cow;

            // GDN (linear attention) layers have no Q/K/V/O — skip
            if lw.gdn.is_some() || lw.wq.is_empty() {
                kernels.push(None);
                continue;
            }

            // Un-expand GQA K/V (same logic as forward prime)
            let hd = cfg.head_dim();
            let hpg = cfg.heads_per_group();
            let unexpand_kv = |expanded: &[f32]| -> Vec<f32> {
                if hpg <= 1 {
                    return expanded.to_vec();
                }
                let n_kv = cfg.n_kv_heads;
                let mut compact = Vec::with_capacity(kv_dim * dim);
                for kv_h in 0..n_kv {
                    let src_h = kv_h * hpg;
                    for d in 0..hd {
                        let row = src_h * hd + d;
                        compact.extend_from_slice(&expanded[row * dim..(row + 1) * dim]);
                    }
                }
                compact
            };

            // Build per-layer weight blobs — same transposes as forward
            let blobs: Vec<Vec<u8>> = result
                .weight_names
                .iter()
                .map(|name| match *name {
                    "@model_path/weights/rope_cos.bin" => rope_cos_blob.to_vec(),
                    "@model_path/weights/rope_sin.bin" => rope_sin_blob.to_vec(),
                    "@model_path/weights/mask.bin" => mask_blob.to_vec(),
                    "@model_path/weights/wq.bin" => {
                        let wq_t = transpose_weight(&lw.wq, qpd, dim);
                        build_fp16_blob(&wq_t)
                    }
                    "@model_path/weights/wk.bin" => {
                        let wk_compact = unexpand_kv(&lw.wk);
                        let wk_t = transpose_weight(&wk_compact, kv_dim, dim);
                        build_fp16_blob(&wk_t)
                    }
                    "@model_path/weights/wv.bin" => {
                        let wv_compact = unexpand_kv(&lw.wv);
                        let wv_t = transpose_weight(&wv_compact, kv_dim, dim);
                        build_fp16_blob(&wv_t)
                    }
                    "@model_path/weights/wo.bin" => {
                        let wo_t = transpose_weight(&lw.wo, dim, attn_dim);
                        build_fp16_blob(&wo_t)
                    }
                    _ => build_fp16_blob(&vec![0.0f32; 16]),
                })
                .collect();

            let names: Vec<&str> = result.weight_names.iter().copied().collect();
            let datas: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();

            let k = super::ane_bridge::AneKernel::compile_multi_weights(
                &result.mil_text,
                &names,
                &datas,
                &[result.input_bytes],
                &[result.output_bytes],
            )
            .map_err(|e| format!("layer {l} fused_attn_gqa_bwd compile: {e}"))?;

            kernels.push(Some(k));
        }

        let elapsed = t0.elapsed();
        let attn_layers = kernels.iter().filter(|k| k.is_some()).count();
        tracing::info!(
            "primed {attn_layers}/{n_layers} per-layer fused backward attention GQA kernels in {:.1}ms ({:.1}ms/layer)",
            elapsed.as_secs_f64() * 1000.0,
            if attn_layers > 0 { elapsed.as_secs_f64() * 1000.0 / attn_layers as f64 } else { 0.0 },
        );
        self.bwd_fused_attn_gqa_kernels = Some(kernels);
        Ok(())
    }

    /// Evaluate fused backward attention GQA on per-layer kernel.
    ///
    /// Packs dx2, Q_rot, K_rot, V, [pre_gate, gate_raw] into the input buffer,
    /// evals, and returns dx_attn [dim, seq] fp32. Returns None if not primed.
    pub fn eval_bwd_fused_attn_gqa(
        &self,
        layer: usize,
        dx2: &[f32],
        ac: &super::ane_forward::LayerActivations,
        cfg: &super::ane_mil::MilConfig,
    ) -> Option<Result<Vec<f32>, String>> {
        let kernels = self.bwd_fused_attn_gqa_kernels.as_ref()?;
        let kernel = kernels[layer].as_ref()?;

        let dim = cfg.dim;
        let ad = cfg.attn_dim();
        let kvd = cfg.kv_dim();
        let seq = cfg.seq_len;
        let has_gate = cfg.attn_output_gate;

        // Pack input: dx2[dim,seq] | Q_rot[ad,seq] | K_rot[kvd,seq] | V[kvd,seq]
        //             [| pre_gate[ad,seq] | gate_raw[ad,seq]]
        let in_ch = if has_gate {
            dim + 3 * ad + 2 * kvd
        } else {
            dim + ad + 2 * kvd
        };
        let mut input = Vec::with_capacity(in_ch * seq);
        input.extend_from_slice(dx2);
        input.extend_from_slice(&ac.q[..ad * seq]);
        input.extend_from_slice(&ac.k[..kvd * seq]);
        input.extend_from_slice(&ac.v[..kvd * seq]);
        if has_gate {
            if let (Some(pg), Some(gr)) = (ac.attn_pre_gate.as_ref(), ac.attn_gate.as_ref()) {
                input.extend_from_slice(pg);
                input.extend_from_slice(gr);
            } else {
                return Some(Err("gate activations missing for fused bwd attn".into()));
            }
        }

        let input_bytes =
            unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
        kernel.write_input(0, input_bytes);

        if let Err(e) = kernel.eval() {
            return Some(Err(format!(
                "per-layer fused_attn_gqa_bwd eval layer {layer}: {e}"
            )));
        }

        let mut buf = vec![0u8; dim * seq * 4];
        kernel.read_output(0, &mut buf);
        let dx_attn: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Some(Ok(dx_attn))
    }

    /// Prime per-layer Wot backward kernels with BLOBFILE weights.
    ///
    /// Each MHA layer gets a compiled kernel with Wo transposed baked in.
    /// Eliminates DynMatmul packing overhead (~16MB memcpy per layer per step).
    pub fn prime_bwd_wot_kernels<W: WeightSource>(
        &mut self,
        cfg: &super::ane_mil::MilConfig,
        model: &W,
    ) -> Result<(), String> {
        let n_layers = model.n_layers();
        let dim = cfg.dim;
        let ad = cfg.attn_dim();
        let seq = cfg.seq_len;

        // Wot blob is dim*ad fp16 = dim*ad*2 bytes. Skip if exceeds ANE SRAM.
        let wot_blob_bytes = dim * ad * 2;
        if wot_blob_bytes > Self::ANE_MAX_BLOBFILE_BYTES {
            tracing::debug!(
                "skipping Wot bwd: blob {:.1}MB exceeds SRAM limit",
                wot_blob_bytes as f64 / (1024.0 * 1024.0),
            );
            self.bwd_wot_kernels = Some((0..n_layers).map(|_| None).collect());
            return Ok(());
        }

        let result = super::ane_mil::gen_wot_bwd_blob(dim, ad, seq);
        let t0 = std::time::Instant::now();

        let mut kernels = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let lw_cow = model.layer(l);
            let lw = &*lw_cow;
            if lw.gdn.is_some() || lw.wo.is_empty() {
                kernels.push(None);
                continue;
            }
            // Wo is [dim, ad] row-major. Wot = transpose(Wo) = [ad, dim].
            let wot = transpose_weight(&lw.wo, dim, ad);
            let blob = build_fp16_blob(&wot);
            let names: Vec<&str> = result.weight_names.iter().copied().collect();
            let k = super::ane_bridge::AneKernel::compile_multi_weights(
                &result.mil_text,
                &names,
                &[&blob],
                &[result.input_bytes],
                &[result.output_bytes],
            )
            .map_err(|e| format!("layer {l} bwd_wot compile: {e}"))?;
            kernels.push(Some(k));
        }

        let elapsed = t0.elapsed();
        let count = kernels.iter().filter(|k| k.is_some()).count();
        tracing::info!(
            "primed {count}/{n_layers} per-layer Wot backward kernels in {:.1}ms",
            elapsed.as_secs_f64() * 1000.0
        );
        self.bwd_wot_kernels = Some(kernels);
        Ok(())
    }

    /// Evaluate per-layer Wot backward: da = Wo^T @ dx2.
    /// Returns None if not primed.
    pub fn eval_bwd_wot(
        &self,
        layer: usize,
        dx2: &[f32],
        cfg: &super::ane_mil::MilConfig,
    ) -> Option<Result<Vec<f32>, String>> {
        let kernels = self.bwd_wot_kernels.as_ref()?;
        let kernel = kernels[layer].as_ref()?;
        let bytes = unsafe { std::slice::from_raw_parts(dx2.as_ptr() as *const u8, dx2.len() * 4) };
        kernel.write_input(0, bytes);
        if let Err(e) = kernel.eval() {
            return Some(Err(format!("bwd_wot eval layer {layer}: {e}")));
        }
        let ad = cfg.attn_dim();
        let seq = cfg.seq_len;
        let mut buf = vec![0u8; ad * seq * 4];
        kernel.read_output(0, &mut buf);
        Some(Ok(buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()))
    }

    /// Prime per-layer fused Wot+SDPA backward kernels (2 BLOBFILEs: Wo + mask).
    ///
    /// Merges the Wo^T dispatch into the SDPA backward kernel, reducing
    /// attention backward from 3 dispatches (Wot | SDPA | QKV) to 2 (Wot+SDPA | QKV).
    /// Each MHA layer gets a compiled kernel with Wo baked in as BLOBFILE.
    /// Max single BLOBFILE size the ANE compiler can handle (~32 MB SRAM).
    /// We use a conservative 16 MB threshold — anything above triggers the
    /// "MIR attributes missing from RESHAPE" compiler internal error (Bug 14).
    const ANE_MAX_BLOBFILE_BYTES: usize = 16 * 1024 * 1024;

    pub fn prime_bwd_wot_sdpa_kernels<W: WeightSource>(
        &mut self,
        cfg: &super::ane_mil::MilConfig,
        model: &W,
    ) -> Result<(), String> {
        let n_layers = model.n_layers();
        let dim = cfg.dim;
        let ad = cfg.attn_dim();

        // Wo blob is dim*ad fp16 values = dim*ad*2 bytes.
        // If it exceeds ANE SRAM the compiler will fail noisily — skip silently.
        let wo_blob_bytes = dim * ad * 2;
        if wo_blob_bytes > Self::ANE_MAX_BLOBFILE_BYTES {
            tracing::debug!(
                "skipping fused Wot+SDPA bwd: Wo blob {:.1}MB exceeds {:.0}MB SRAM limit",
                wo_blob_bytes as f64 / (1024.0 * 1024.0),
                Self::ANE_MAX_BLOBFILE_BYTES as f64 / (1024.0 * 1024.0),
            );
            self.bwd_wot_sdpa_kernels = Some((0..n_layers).map(|_| None).collect());
            return Ok(());
        }

        let has_gate = cfg.attn_output_gate;
        let result = super::ane_mil::gen_wot_sdpa_bwd(cfg, has_gate);
        let mask_blob = super::ane_mil::build_causal_mask_blob(cfg.seq_len);
        let t0 = std::time::Instant::now();

        let mut kernels = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let lw_cow = model.layer(l);
            let lw = &*lw_cow;
            if lw.gdn.is_some() || lw.wo.is_empty() {
                kernels.push(None);
                continue;
            }
            // Wo is [dim, ad] row-major. MIL expects [ad, dim] for transpose_y=bT matmul.
            let wo_ad_dim = transpose_weight(&lw.wo, dim, ad);
            let wo_blob = build_fp16_blob(&wo_ad_dim);
            let names: Vec<&str> = result.weight_names.iter().copied().collect();
            match super::ane_bridge::AneKernel::compile_multi_weights(
                &result.mil_text,
                &names,
                &[&wo_blob, &mask_blob],
                &[result.input_bytes],
                &[result.output_bytes],
            ) {
                Ok(k) => kernels.push(Some(k)),
                Err(e) => {
                    tracing::warn!("layer {l} bwd_wot_sdpa compile failed: {e}");
                    kernels.push(None);
                }
            }
        }

        let elapsed = t0.elapsed();
        let count = kernels.iter().filter(|k| k.is_some()).count();
        tracing::info!(
            "primed {count}/{n_layers} per-layer Wot+SDPA backward kernels in {:.1}ms",
            elapsed.as_secs_f64() * 1000.0
        );
        self.bwd_wot_sdpa_kernels = Some(kernels);
        Ok(())
    }

    /// Check if fused Wot+SDPA backward kernel is primed for a given layer.
    pub fn has_bwd_wot_sdpa(&self, layer: usize) -> bool {
        self.bwd_wot_sdpa_kernels
            .as_ref()
            .and_then(|ks| ks.get(layer))
            .and_then(|k| k.as_ref())
            .is_some()
    }

    /// Evaluate per-layer fused Wot+SDPA backward.
    ///
    /// Input: `dx2[dim,seq] | Q_rot[ad,seq] | K_exp[ad,seq] | V_exp[ad,seq]` concatenated f32.
    /// With gate: append `pre_gate[ad,seq] | gate_raw[ad,seq]`.
    /// Output: `[1, N*H, S, hd]` fp32 — dQ|dK|dV [|d_gate] on head axis.
    ///   N=3 (non-gated) or N=4 (gated, 4th block is d_gate).
    /// Returns None if not primed.
    pub fn eval_bwd_wot_sdpa(
        &self,
        layer: usize,
        input: &[f32],
        cfg: &super::ane_mil::MilConfig,
    ) -> Option<Result<Vec<f32>, String>> {
        let kernels = self.bwd_wot_sdpa_kernels.as_ref()?;
        let kernel = kernels[layer].as_ref()?;
        let bytes =
            unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
        kernel.write_input(0, bytes);
        if let Err(e) = kernel.eval() {
            return Some(Err(format!("bwd_wot_sdpa eval layer {layer}: {e}")));
        }
        let heads = cfg.n_heads;
        let hd = cfg.head_dim();
        let seq = cfg.seq_len;
        let n_blocks = if cfg.attn_output_gate { 4 } else { 3 };
        let out_elems = n_blocks * heads * seq * hd;
        let mut buf = vec![0u8; out_elems * 4];
        kernel.read_output(0, &mut buf);
        Some(Ok(buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()))
    }

    /// Prime per-layer QKV backward kernels with BLOBFILE weights.
    ///
    /// Each MHA layer gets a compiled kernel with WQ^T, WK^T, WV^T baked in.
    /// When WQ^T fits in SRAM (≤16MB), uses a single kernel with 3 BLOBFILEs.
    /// When WQ^T exceeds SRAM (e.g. 32MB at 35B), splits into 2 kernels of
    /// 2 BLOBFILEs each. Eliminates DynMatmul packing (~48MB per layer).
    pub fn prime_bwd_qkvb_kernels<W: WeightSource>(
        &mut self,
        cfg: &super::ane_mil::MilConfig,
        model: &W,
    ) -> Result<(), String> {
        let n_layers = model.n_layers();
        let dim = cfg.dim;
        let qpd = cfg.q_proj_dim();
        let ad = cfg.attn_dim();

        let max_blob_bytes = dim * qpd.max(ad) * 2;
        let needs_split = max_blob_bytes > Self::ANE_MAX_BLOBFILE_BYTES;

        // Check if split halves also exceed SRAM (would need further splitting)
        if needs_split {
            let half_qpd = qpd / 2;
            let half_blob_bytes = dim * half_qpd.max(ad) * 2;
            if half_blob_bytes > Self::ANE_MAX_BLOBFILE_BYTES {
                tracing::debug!(
                    "skipping QKV bwd: even split half {:.1}MB exceeds SRAM limit",
                    half_blob_bytes as f64 / (1024.0 * 1024.0),
                );
                self.bwd_qkvb_kernels = Some((0..n_layers).map(|_| None).collect());
                return Ok(());
            }
            tracing::info!(
                "QKV bwd: WQ^T {:.1}MB exceeds SRAM, using split path (2 × {:.1}MB)",
                max_blob_bytes as f64 / (1024.0 * 1024.0),
                half_blob_bytes as f64 / (1024.0 * 1024.0),
            );
        }

        let t0 = std::time::Instant::now();
        let mut kernels = Vec::with_capacity(n_layers);

        if needs_split {
            let mil_a = super::ane_mil::gen_qkvb_blob_split_half(cfg, 0);
            let mil_b = super::ane_mil::gen_qkvb_blob_split_half(cfg, 1);
            let half_qpd = qpd / 2;

            for l in 0..n_layers {
                let lw_cow = model.layer(l);
                let lw = &*lw_cow;
                if lw.gdn.is_some() || lw.wq.is_empty() {
                    kernels.push(None);
                    continue;
                }
                // WQ^T = [dim, qpd], split into [dim, half_qpd] halves
                let wqt = transpose_weight(&lw.wq, qpd, dim);
                let wqt_h0: Vec<f32> = wqt
                    .chunks_exact(qpd)
                    .flat_map(|row| &row[..half_qpd])
                    .copied()
                    .collect();
                let wqt_h1: Vec<f32> = wqt
                    .chunks_exact(qpd)
                    .flat_map(|row| &row[half_qpd..])
                    .copied()
                    .collect();
                let wqt_h0_blob = build_fp16_blob(&wqt_h0);
                let wqt_h1_blob = build_fp16_blob(&wqt_h1);
                // WK^T and WV^T
                let wkt = transpose_weight(&lw.wk, ad, dim);
                let wkt_blob = build_fp16_blob(&wkt);
                let wvt = transpose_weight(&lw.wv, ad, dim);
                let wvt_blob = build_fp16_blob(&wvt);

                // Compile kernel A: Wq_h0 + Wk
                let names_a: Vec<&str> = mil_a.weight_names.iter().copied().collect();
                let ka = super::ane_bridge::AneKernel::compile_multi_weights(
                    &mil_a.mil_text,
                    &names_a,
                    &[&wqt_h0_blob, &wkt_blob],
                    &[mil_a.input_bytes],
                    &[mil_a.output_bytes],
                );
                // Compile kernel B: Wq_h1 + Wv
                let names_b: Vec<&str> = mil_b.weight_names.iter().copied().collect();
                let kb = super::ane_bridge::AneKernel::compile_multi_weights(
                    &mil_b.mil_text,
                    &names_b,
                    &[&wqt_h1_blob, &wvt_blob],
                    &[mil_b.input_bytes],
                    &[mil_b.output_bytes],
                );

                match (ka, kb) {
                    (Ok(a), Ok(b)) => kernels.push(Some(QkvbKernel::Split {
                        half_a: a,
                        half_b: b,
                    })),
                    (Err(e), _) | (_, Err(e)) => {
                        tracing::debug!("layer {l} bwd_qkvb split compile failed: {e}");
                        kernels.push(None);
                    }
                }
            }
        } else {
            let result = super::ane_mil::gen_qkvb_blob(cfg);
            for l in 0..n_layers {
                let lw_cow = model.layer(l);
                let lw = &*lw_cow;
                if lw.gdn.is_some() || lw.wq.is_empty() {
                    kernels.push(None);
                    continue;
                }
                let wqt = transpose_weight(&lw.wq, qpd, dim);
                let wqt_blob = build_fp16_blob(&wqt);
                let wkt = transpose_weight(&lw.wk, ad, dim);
                let wkt_blob = build_fp16_blob(&wkt);
                let wvt = transpose_weight(&lw.wv, ad, dim);
                let wvt_blob = build_fp16_blob(&wvt);

                let names: Vec<&str> = result.weight_names.iter().copied().collect();
                match super::ane_bridge::AneKernel::compile_multi_weights(
                    &result.mil_text,
                    &names,
                    &[&wqt_blob, &wkt_blob, &wvt_blob],
                    &[result.input_bytes],
                    &[result.output_bytes],
                ) {
                    Ok(k) => kernels.push(Some(QkvbKernel::Single(k))),
                    Err(e) => {
                        tracing::debug!("layer {l} bwd_qkvb compile failed: {e}");
                        kernels.push(None);
                    }
                }
            }
        }

        let elapsed = t0.elapsed();
        let count = kernels.iter().filter(|k| k.is_some()).count();
        let mode = if needs_split { "split" } else { "single" };
        tracing::info!(
            "primed {count}/{n_layers} per-layer QKV backward kernels ({mode}) in {:.1}ms",
            elapsed.as_secs_f64() * 1000.0
        );
        self.bwd_qkvb_kernels = Some(kernels);
        Ok(())
    }

    /// Evaluate per-layer QKV backward: dx = WQ^T@dQ + WK^T@dK + WV^T@dV.
    /// Input: dQ[qpd,seq] | dK[ad,seq] | dV[ad,seq] concatenated.
    /// Returns None if not primed.
    ///
    /// Dispatches `QkvbKernel::Single` (1 eval) or `QkvbKernel::Split` (2 evals + sum).
    pub fn eval_bwd_qkvb(
        &self,
        layer: usize,
        dqkv: &[f32],
        cfg: &super::ane_mil::MilConfig,
    ) -> Option<Result<Vec<f32>, String>> {
        let kernels = self.bwd_qkvb_kernels.as_ref()?;
        let qkvb = kernels[layer].as_ref()?;
        let dim = cfg.dim;
        let seq = cfg.seq_len;
        let out_floats = dim * seq;

        match qkvb {
            QkvbKernel::Single(kernel) => {
                let bytes = unsafe {
                    std::slice::from_raw_parts(dqkv.as_ptr() as *const u8, dqkv.len() * 4)
                };
                kernel.write_input(0, bytes);
                if let Err(e) = kernel.eval() {
                    return Some(Err(format!("bwd_qkvb eval layer {layer}: {e}")));
                }
                let mut buf = vec![0u8; out_floats * 4];
                kernel.read_output(0, &mut buf);
                Some(Ok(buf
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()))
            }
            QkvbKernel::Split { half_a, half_b } => {
                let qpd = cfg.q_proj_dim();
                let ad = cfg.attn_dim();
                let half_qpd = qpd / 2;

                // Build input A: dQ[0..half_qpd] | dK
                // dqkv layout: dQ[qpd*seq] | dK[ad*seq] | dV[ad*seq]
                let dq_offset = 0;
                let dk_offset = qpd * seq;
                let dv_offset = (qpd + ad) * seq;

                let input_a_len = (half_qpd + ad) * seq;
                let mut input_a = Vec::with_capacity(input_a_len);
                input_a.extend_from_slice(&dqkv[dq_offset..dq_offset + half_qpd * seq]);
                input_a.extend_from_slice(&dqkv[dk_offset..dk_offset + ad * seq]);

                let bytes_a = unsafe {
                    std::slice::from_raw_parts(input_a.as_ptr() as *const u8, input_a.len() * 4)
                };
                half_a.write_input(0, bytes_a);
                if let Err(e) = half_a.eval() {
                    return Some(Err(format!(
                        "bwd_qkvb split half_a eval layer {layer}: {e}"
                    )));
                }

                // Build input B: dQ[half_qpd..qpd] | dV
                let input_b_len = (half_qpd + ad) * seq;
                let mut input_b = Vec::with_capacity(input_b_len);
                input_b.extend_from_slice(&dqkv[dq_offset + half_qpd * seq..dq_offset + qpd * seq]);
                input_b.extend_from_slice(&dqkv[dv_offset..dv_offset + ad * seq]);

                let bytes_b = unsafe {
                    std::slice::from_raw_parts(input_b.as_ptr() as *const u8, input_b.len() * 4)
                };
                half_b.write_input(0, bytes_b);
                if let Err(e) = half_b.eval() {
                    return Some(Err(format!(
                        "bwd_qkvb split half_b eval layer {layer}: {e}"
                    )));
                }

                // Read both outputs and sum
                let mut buf_a = vec![0u8; out_floats * 4];
                let mut buf_b = vec![0u8; out_floats * 4];
                half_a.read_output(0, &mut buf_a);
                half_b.read_output(0, &mut buf_b);

                let dx: Vec<f32> = buf_a
                    .chunks_exact(4)
                    .zip(buf_b.chunks_exact(4))
                    .map(|(a, b)| {
                        f32::from_le_bytes([a[0], a[1], a[2], a[3]])
                            + f32::from_le_bytes([b[0], b[1], b[2], b[3]])
                    })
                    .collect();

                Some(Ok(dx))
            }
        }
    }

    /// Prime per-layer RMSNorm kernels (attention + FFN) with baked weights.
    ///
    /// Each layer's RMSNorm weight vector is compiled into a per-layer kernel via
    /// `compile_multi_weights` hitting the delta cache (same MIL, different weights).
    pub fn prime_rmsnorm_kernels<W: WeightSource>(
        &mut self,
        cfg: &super::ane_mil::MilConfig,
        model: &W,
    ) -> Result<(), String> {
        let n_layers = model.n_layers();
        let dim = cfg.dim;
        let seq = cfg.seq_len;
        let result = super::ane_mil::gen_rmsnorm_fwd(dim, seq, cfg.rms_eps);

        let mut att_kernels = Vec::with_capacity(n_layers);
        let mut ffn_kernels = Vec::with_capacity(n_layers);
        let t0 = std::time::Instant::now();

        for l in 0..n_layers {
            let fetch_t0 = std::time::Instant::now();
            let lw_cow = model.layer(l);
            let fetch_ms = fetch_t0.elapsed().as_millis();
            let lw = &*lw_cow;
            let layer_t0 = std::time::Instant::now();

            // Attention RMSNorm weight
            let att_t0 = std::time::Instant::now();
            let att_blob = build_fp16_blob(&lw.rms_att);
            let names: Vec<&str> = result.weight_names.iter().copied().collect();
            let k = super::ane_bridge::AneKernel::compile_multi_weights(
                &result.mil_text,
                &names,
                &[&att_blob],
                &[result.input_bytes],
                &[result.output_bytes],
            )
            .map_err(|e| format!("layer {l} rmsnorm_att compile: {e}"))?;
            att_kernels.push(k);
            bench_trace_weights(format!(
                "prime_rmsnorm_fwd:att layer={} fetch_ms={} elapsed_ms={}",
                l,
                fetch_ms,
                att_t0.elapsed().as_millis()
            ));

            // FFN RMSNorm weight
            let ffn_t0 = std::time::Instant::now();
            let ffn_blob = build_fp16_blob(&lw.rms_ffn);
            let k = super::ane_bridge::AneKernel::compile_multi_weights(
                &result.mil_text,
                &names,
                &[&ffn_blob],
                &[result.input_bytes],
                &[result.output_bytes],
            )
            .map_err(|e| format!("layer {l} rmsnorm_ffn compile: {e}"))?;
            ffn_kernels.push(k);
            bench_trace_weights(format!(
                "prime_rmsnorm_fwd:ffn layer={} elapsed_ms={} layer_total_ms={}",
                l,
                ffn_t0.elapsed().as_millis(),
                layer_t0.elapsed().as_millis()
            ));
        }

        let elapsed = t0.elapsed();
        tracing::info!(
            "primed {} per-layer RMSNorm kernels (att+ffn) in {:.1}ms ({:.1}ms/layer)",
            n_layers * 2,
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_secs_f64() * 1000.0 / n_layers as f64,
        );
        self.rmsnorm_att_kernels = Some(att_kernels);
        self.rmsnorm_ffn_kernels = Some(ffn_kernels);
        Ok(())
    }

    /// Prime per-layer RMSNorm backward kernels (dx only, no dw).
    pub fn prime_rmsnorm_bwd_kernels<W: WeightSource>(
        &mut self,
        cfg: &super::ane_mil::MilConfig,
        model: &W,
    ) -> Result<(), String> {
        let n_layers = model.n_layers();
        let dim = cfg.dim;
        let seq = cfg.seq_len;
        let result = super::ane_mil::gen_rmsnorm_bwd(dim, seq, cfg.rms_eps);

        let mut att_kernels = Vec::with_capacity(n_layers);
        let mut ffn_kernels = Vec::with_capacity(n_layers);
        let t0 = std::time::Instant::now();

        for l in 0..n_layers {
            let fetch_t0 = std::time::Instant::now();
            let lw_cow = model.layer(l);
            let fetch_ms = fetch_t0.elapsed().as_millis();
            let lw = &*lw_cow;
            let layer_t0 = std::time::Instant::now();

            let att_t0 = std::time::Instant::now();
            let att_blob = build_fp16_blob(&lw.rms_att);
            let names: Vec<&str> = result.weight_names.iter().copied().collect();
            let k = super::ane_bridge::AneKernel::compile_multi_weights(
                &result.mil_text,
                &names,
                &[&att_blob],
                &[result.input_bytes],
                &[result.output_bytes],
            )
            .map_err(|e| format!("layer {l} rmsnorm_bwd_att compile: {e}"))?;
            att_kernels.push(k);
            bench_trace_weights(format!(
                "prime_rmsnorm_bwd:att layer={} fetch_ms={} elapsed_ms={}",
                l,
                fetch_ms,
                att_t0.elapsed().as_millis()
            ));

            let ffn_t0 = std::time::Instant::now();
            let ffn_blob = build_fp16_blob(&lw.rms_ffn);
            let k = super::ane_bridge::AneKernel::compile_multi_weights(
                &result.mil_text,
                &names,
                &[&ffn_blob],
                &[result.input_bytes],
                &[result.output_bytes],
            )
            .map_err(|e| format!("layer {l} rmsnorm_bwd_ffn compile: {e}"))?;
            ffn_kernels.push(k);
            bench_trace_weights(format!(
                "prime_rmsnorm_bwd:ffn layer={} elapsed_ms={} layer_total_ms={}",
                l,
                ffn_t0.elapsed().as_millis(),
                layer_t0.elapsed().as_millis()
            ));
        }

        let elapsed = t0.elapsed();
        tracing::info!(
            "primed {} per-layer RMSNorm backward kernels (att+ffn) in {:.1}ms",
            n_layers * 2,
            elapsed.as_secs_f64() * 1000.0,
        );
        self.rmsnorm_bwd_att_kernels = Some(att_kernels);
        self.rmsnorm_bwd_ffn_kernels = Some(ffn_kernels);
        Ok(())
    }

    /// Evaluate RMSNorm backward on ANE per-layer kernel.
    ///
    /// Input: dy[dim,seq] and x[dim,seq] (the forward input).
    /// Returns dx[dim,seq] or None if not primed.
    pub fn eval_rmsnorm_bwd(
        &self,
        layer: usize,
        dy: &[f32],
        x: &[f32],
        which_ffn: bool,
    ) -> Option<Result<Vec<f32>, String>> {
        let kernels = if which_ffn {
            self.rmsnorm_bwd_ffn_kernels.as_ref()?
        } else {
            self.rmsnorm_bwd_att_kernels.as_ref()?
        };
        let kernel = &kernels[layer];

        // Pack input: dy | x concatenated on channel axis
        let n = dy.len();
        let mut input = Vec::with_capacity(2 * n);
        input.extend_from_slice(dy);
        input.extend_from_slice(x);

        let input_bytes =
            unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
        kernel.write_input(0, input_bytes);

        if let Err(e) = kernel.eval() {
            return Some(Err(format!(
                "rmsnorm_bwd eval layer {layer} (ffn={which_ffn}): {e}"
            )));
        }

        let mut buf = vec![0u8; n * 4];
        kernel.read_output(0, &mut buf);
        let out: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Some(Ok(out))
    }

    /// Evaluate RMSNorm on per-layer kernel. Returns None if not primed.
    ///
    /// `which`: `false` for attention RMSNorm, `true` for FFN RMSNorm.
    pub fn eval_rmsnorm(
        &self,
        layer: usize,
        x: &[f32],
        which_ffn: bool,
    ) -> Option<Result<Vec<f32>, String>> {
        let kernels = if which_ffn {
            self.rmsnorm_ffn_kernels.as_ref()?
        } else {
            self.rmsnorm_att_kernels.as_ref()?
        };
        let kernel = &kernels[layer];

        let input_bytes =
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
        kernel.write_input(0, input_bytes);

        if let Err(e) = kernel.eval() {
            return Some(Err(format!(
                "rmsnorm eval layer {layer} (ffn={which_ffn}): {e}"
            )));
        }

        let mut buf = vec![0u8; x.len() * 4];
        kernel.read_output(0, &mut buf);
        let out: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Some(Ok(out))
    }

    /// Prime per-layer fused FFN backward kernels (W2^T + SiLU bwd + W13^T).
    pub fn prime_fused_ffn_bwd_kernels<W: WeightSource>(
        &mut self,
        cfg: &super::ane_mil::MilConfig,
        model: &W,
    ) -> Result<(), String> {
        let n_layers = model.n_layers();
        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;
        let result = super::ane_mil::gen_fused_ffn_bwd(cfg);

        let mut kernels = Vec::with_capacity(n_layers);
        let t0 = std::time::Instant::now();

        for l in 0..n_layers {
            let fetch_t0 = std::time::Instant::now();
            let lw_cow = model.layer(l);
            let fetch_ms = fetch_t0.elapsed().as_millis();
            let lw = &*lw_cow;
            let layer_t0 = std::time::Instant::now();

            // W2 is stored [dim, hidden] row-major. MIL kernel expects W2^T = [hidden, dim].
            let w2t = transpose_weight(&lw.w2, dim, hidden);
            let w2t_blob = build_fp16_blob(&w2t);
            // W1 is [hidden, dim]. MIL expects W1^T = [dim, hidden].
            let w1t = transpose_weight(&lw.w1, hidden, dim);
            let w1t_blob = build_fp16_blob(&w1t);
            // W3 is [hidden, dim]. MIL expects W3^T = [dim, hidden].
            let w3t = transpose_weight(&lw.w3, hidden, dim);
            let w3t_blob = build_fp16_blob(&w3t);
            bench_trace_weights(format!(
                "prime_fused_ffn_bwd:blobs layer={} fetch_ms={} elapsed_ms={}",
                l,
                fetch_ms,
                layer_t0.elapsed().as_millis()
            ));

            let names: Vec<&str> = result.weight_names.iter().copied().collect();
            let datas: Vec<&[u8]> = vec![&w2t_blob, &w1t_blob, &w3t_blob];

            let compile_t0 = std::time::Instant::now();
            let k = super::ane_bridge::AneKernel::compile_multi_weights(
                &result.mil_text,
                &names,
                &datas,
                &[result.input_bytes],
                &[result.output_bytes],
            )
            .map_err(|e| format!("layer {l} fused_ffn_bwd compile: {e}"))?;

            kernels.push(k);
            bench_trace_weights(format!(
                "prime_fused_ffn_bwd:compile layer={} elapsed_ms={} layer_total_ms={}",
                l,
                compile_t0.elapsed().as_millis(),
                layer_t0.elapsed().as_millis()
            ));
        }

        let elapsed = t0.elapsed();
        tracing::info!(
            "primed {} per-layer fused FFN backward kernels in {:.1}ms ({:.1}ms/layer)",
            n_layers,
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_secs_f64() * 1000.0 / n_layers.max(1) as f64,
        );
        self.fused_ffn_bwd_kernels = Some(kernels);
        Ok(())
    }

    /// Prime shared fused FFN backward kernel (1 ANE program slot for all layers).
    /// Uses `reload_weights()` to hotswap per-layer weights before each eval.
    /// Only uses 1 ANE program slot instead of N (critical for 119-slot budget).
    pub fn prime_fused_ffn_bwd_shared<W: WeightSource>(
        &mut self,
        cfg: &super::ane_mil::MilConfig,
        model: &W,
    ) -> Result<(), String> {
        let n_layers = model.n_layers();
        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;
        let result = super::ane_mil::gen_fused_ffn_bwd(cfg);
        let t0 = std::time::Instant::now();

        // Pre-build weight blobs for all layers
        let mut all_blobs = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let lw_cow = model.layer(l);
            let lw = &*lw_cow;
            let w2t = transpose_weight(&lw.w2, dim, hidden);
            let w1t = transpose_weight(&lw.w1, hidden, dim);
            let w3t = transpose_weight(&lw.w3, hidden, dim);
            all_blobs.push([
                build_fp16_blob(&w2t),
                build_fp16_blob(&w1t),
                build_fp16_blob(&w3t),
            ]);
        }

        // Compile ONE kernel with layer 0's weights
        let names: Vec<&str> = result.weight_names.iter().copied().collect();
        let datas: Vec<&[u8]> = vec![&all_blobs[0][0], &all_blobs[0][1], &all_blobs[0][2]];
        let kernel = super::ane_bridge::AneKernel::compile_multi_weights(
            &result.mil_text,
            &names,
            &datas,
            &[result.input_bytes],
            &[result.output_bytes],
        )
        .map_err(|e| format!("fused_ffn_bwd shared compile: {e}"))?;

        let elapsed = t0.elapsed();
        tracing::info!(
            "primed shared FFN bwd kernel (1 slot, {} layers hotswap) in {:.1}ms",
            n_layers,
            elapsed.as_secs_f64() * 1000.0,
        );
        self.fused_ffn_bwd_shared = Some(kernel);
        self.fused_ffn_bwd_weight_blobs = Some(all_blobs);
        Ok(())
    }

    /// Evaluate fused FFN backward using shared kernel with weight hotswap.
    pub fn eval_fused_ffn_bwd_hotswap(
        &self,
        layer: usize,
        dx_ffn: &[f32],
        h1: &[f32],
        h3: &[f32],
        dim: usize,
        hidden: usize,
        seq: usize,
    ) -> Option<Result<(Vec<f32>, Vec<f32>), String>> {
        let kernel = self.fused_ffn_bwd_shared.as_ref()?;
        let blobs = self.fused_ffn_bwd_weight_blobs.as_ref()?;
        let layer_blobs = &blobs[layer];

        // Hotswap weights for this layer
        let weight_datas: Vec<&[u8]> = vec![&layer_blobs[0], &layer_blobs[1], &layer_blobs[2]];
        if let Err(e) = kernel.delta_reload(&weight_datas) {
            return Some(Err(format!("fused_ffn_bwd hotswap layer {layer}: {e}")));
        }

        // Pack input: dx_ffn | h1 | h3
        let mut input = Vec::with_capacity(dx_ffn.len() + h1.len() + h3.len());
        input.extend_from_slice(dx_ffn);
        input.extend_from_slice(h1);
        input.extend_from_slice(h3);

        let input_bytes =
            unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
        kernel.write_input(0, input_bytes);

        if let Err(e) = kernel.eval() {
            return Some(Err(format!(
                "fused_ffn_bwd hotswap eval layer {layer}: {e}"
            )));
        }

        let out_ch = dim + hidden;
        let mut buf = vec![0u8; out_ch * seq * 4];
        kernel.read_output(0, &mut buf);
        let out: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let dx = out[..dim * seq].to_vec();
        let dsilu = out[dim * seq..].to_vec();
        Some(Ok((dx, dsilu)))
    }

    /// Evaluate fused FFN backward on per-layer kernel.
    ///
    /// Input: dx_ffn[dim*seq], h1[hidden*seq], h3[hidden*seq]
    /// Output: (dx[dim*seq], dsilu[hidden*seq]) — dx for backward chain, dsilu for LoRA W2 grads.
    /// Returns None if not primed.
    pub fn eval_fused_ffn_bwd(
        &self,
        layer: usize,
        dx_ffn: &[f32],
        h1: &[f32],
        h3: &[f32],
        dim: usize,
        hidden: usize,
        seq: usize,
    ) -> Option<Result<(Vec<f32>, Vec<f32>), String>> {
        let kernels = self.fused_ffn_bwd_kernels.as_ref()?;
        let kernel = &kernels[layer];

        // Pack input: dx_ffn | h1 | h3
        let mut input = Vec::with_capacity(dx_ffn.len() + h1.len() + h3.len());
        input.extend_from_slice(dx_ffn);
        input.extend_from_slice(h1);
        input.extend_from_slice(h3);

        let input_bytes =
            unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
        kernel.write_input(0, input_bytes);

        if let Err(e) = kernel.eval() {
            return Some(Err(format!("fused_ffn_bwd eval layer {layer}: {e}")));
        }

        let out_ch = dim + hidden;
        let mut buf = vec![0u8; out_ch * seq * 4];
        kernel.read_output(0, &mut buf);
        let out: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let dx = out[..dim * seq].to_vec();
        let dsilu = out[dim * seq..].to_vec();
        Some(Ok((dx, dsilu)))
    }

    /// Prime split FFN backward kernels (2 shallower BLOBFILE kernels, 2 ANE slots).
    ///
    /// Kernel A: W2^T + SiLU bwd → dh1|dh3|dsilu (1 weight per layer).
    /// Kernel B: W1^T + W3^T → dx (2 weights per layer).
    ///
    /// Uses shared hotswap: 2 ANE program slots total, weight blobs swapped per layer.
    /// This compiles at 35B dims where the monolithic 3-matmul `gen_fused_ffn_bwd` fails.
    pub fn prime_split_ffn_bwd<W: WeightSource>(
        &mut self,
        cfg: &super::ane_mil::MilConfig,
        model: &W,
    ) -> Result<(), String> {
        let n_layers = model.n_layers();
        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;
        let t0 = std::time::Instant::now();

        let result_a = super::ane_mil::gen_ffn_bwd_w2t_silu_blob(cfg);
        let result_b = super::ane_mil::gen_ffn_bwd_w13t_blob(cfg);

        // Pre-build weight blobs for all layers
        let mut a_blobs = Vec::with_capacity(n_layers);
        let mut b_blobs = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let lw_cow = model.layer(l);
            let lw = &*lw_cow;
            // Kernel A: W2 [dim, hidden] (NOT transposed — BLOBFILE must be y-operand)
            a_blobs.push(build_fp16_blob(&lw.w2));
            // Kernel B: W1 [hidden, dim], W3 [hidden, dim] (NOT transposed)
            b_blobs.push([build_fp16_blob(&lw.w1), build_fp16_blob(&lw.w3)]);
        }

        // Compile kernel A with layer 0's weights
        let names_a: Vec<&str> = result_a.weight_names.iter().copied().collect();
        let datas_a: Vec<&[u8]> = vec![&a_blobs[0]];
        let kernel_a = super::ane_bridge::AneKernel::compile_multi_weights(
            &result_a.mil_text,
            &names_a,
            &datas_a,
            &[result_a.input_bytes],
            &[result_a.output_bytes],
        )
        .map_err(|e| format!("split_ffn_bwd kernel A compile: {e}"))?;

        // Compile kernel B with layer 0's weights
        let names_b: Vec<&str> = result_b.weight_names.iter().copied().collect();
        let datas_b: Vec<&[u8]> = vec![&b_blobs[0][0], &b_blobs[0][1]];
        let kernel_b = super::ane_bridge::AneKernel::compile_multi_weights(
            &result_b.mil_text,
            &names_b,
            &datas_b,
            &[result_b.input_bytes],
            &[result_b.output_bytes],
        )
        .map_err(|e| format!("split_ffn_bwd kernel B compile: {e}"))?;

        let elapsed = t0.elapsed();
        tracing::info!(
            "primed split FFN bwd kernels (2 slots, {} layers hotswap) in {:.1}ms",
            n_layers,
            elapsed.as_secs_f64() * 1000.0,
        );

        self.split_ffn_bwd_a_shared = Some(kernel_a);
        self.split_ffn_bwd_b_shared = Some(kernel_b);
        self.split_ffn_bwd_a_blobs = Some(a_blobs);
        self.split_ffn_bwd_b_blobs = Some(b_blobs);
        Ok(())
    }

    /// Evaluate split FFN backward (2 dispatches) with weight hotswap.
    ///
    /// Returns `(dx, dsilu)` — same interface as `eval_fused_ffn_bwd`.
    pub fn eval_split_ffn_bwd(
        &self,
        layer: usize,
        dx_ffn: &[f32],
        h1: &[f32],
        h3: &[f32],
        dim: usize,
        hidden: usize,
        seq: usize,
    ) -> Option<Result<(Vec<f32>, Vec<f32>), String>> {
        let ka = self.split_ffn_bwd_a_shared.as_ref()?;
        let kb = self.split_ffn_bwd_b_shared.as_ref()?;
        let a_blobs = self.split_ffn_bwd_a_blobs.as_ref()?;
        let b_blobs = self.split_ffn_bwd_b_blobs.as_ref()?;

        // Kernel A: hotswap W2^T, eval dx_ffn|h1|h3 → dh1|dh3|dsilu
        if let Err(e) = ka.delta_reload(&[&a_blobs[layer]]) {
            return Some(Err(format!("split_ffn_bwd_a hotswap layer {layer}: {e}")));
        }

        let mut input_a = Vec::with_capacity(dx_ffn.len() + h1.len() + h3.len());
        input_a.extend_from_slice(dx_ffn);
        input_a.extend_from_slice(h1);
        input_a.extend_from_slice(h3);
        let input_a_bytes =
            unsafe { std::slice::from_raw_parts(input_a.as_ptr() as *const u8, input_a.len() * 4) };
        ka.write_input(0, input_a_bytes);

        if let Err(e) = ka.eval() {
            return Some(Err(format!("split_ffn_bwd_a eval layer {layer}: {e}")));
        }

        let out_a_ch = 3 * hidden;
        let mut buf_a = vec![0u8; out_a_ch * seq * 4];
        ka.read_output(0, &mut buf_a);
        let out_a: Vec<f32> = buf_a
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let dh1 = &out_a[..hidden * seq];
        let dh3 = &out_a[hidden * seq..2 * hidden * seq];
        let dsilu = out_a[2 * hidden * seq..].to_vec();

        // Kernel B: hotswap W1^T + W3^T, eval dh1|dh3 → dx
        let layer_b = &b_blobs[layer];
        if let Err(e) = kb.delta_reload(&[&layer_b[0], &layer_b[1]]) {
            return Some(Err(format!("split_ffn_bwd_b hotswap layer {layer}: {e}")));
        }

        let mut input_b = Vec::with_capacity(dh1.len() + dh3.len());
        input_b.extend_from_slice(dh1);
        input_b.extend_from_slice(dh3);
        let input_b_bytes =
            unsafe { std::slice::from_raw_parts(input_b.as_ptr() as *const u8, input_b.len() * 4) };
        kb.write_input(0, input_b_bytes);

        if let Err(e) = kb.eval() {
            return Some(Err(format!("split_ffn_bwd_b eval layer {layer}: {e}")));
        }

        let mut buf_b = vec![0u8; dim * seq * 4];
        kb.read_output(0, &mut buf_b);
        let dx: Vec<f32> = buf_b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        Some(Ok((dx, dsilu)))
    }

    /// Prime per-layer fused full-layer forward kernels (MHA layers only).
    ///
    /// Each kernel does: RMSNorm + QKV + RoPE + SDPA + Wo + residual + RMSNorm + FFN + residual
    /// in a single ANE dispatch. Training mode outputs packed activations for backward.
    /// Uses 12 BLOBFILE weights per kernel (within 16-slot limit).
    pub fn prime_fused_layer_fwd<W: WeightSource>(
        &mut self,
        cfg: &super::ane_mil::MilConfig,
        model: &W,
        rope_cos: &[u8],
        rope_sin: &[u8],
        mask: &[u8],
        train: bool,
    ) -> Result<(), String> {
        let n_layers = model.n_layers();
        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;
        let t0 = std::time::Instant::now();

        let mut train_cfg = cfg.clone();
        train_cfg.has_lm_head = train;
        let result = super::ane_mil::gen_fused_layer_fwd(&train_cfg);
        let names: Vec<&str> = result.weight_names.iter().copied().collect();

        let mut kernels = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let lw_cow = model.layer(l);
            let lw = &*lw_cow;

            // Skip GDN layers (fused layer handles both MHA and GQA)
            if lw.gdn.is_some() {
                continue;
            }

            // Build weight blobs: transpose to [in, out] layout for matmul pattern
            // Derive dims from actual weight sizes to handle attn_output_gate (Wq doubles)
            let qpd = lw.wq.len() / dim; // q_proj_dim (attn_dim or 2*attn_dim with gate)
            let kvd = lw.wk.len() / dim; // kv_dim
            let ad = lw.wo.len() / dim; // attn_dim
            let wq = build_fp16_blob(&transpose_weight(&lw.wq, qpd, dim));
            let wk = build_fp16_blob(&transpose_weight(&lw.wk, kvd, dim));
            let wv = build_fp16_blob(&transpose_weight(&lw.wv, kvd, dim));
            let wo = build_fp16_blob(&transpose_weight(&lw.wo, dim, ad));
            let w1 = build_fp16_blob(&transpose_weight(&lw.w1, hidden, dim));
            let w3 = build_fp16_blob(&transpose_weight(&lw.w3, hidden, dim));
            let w2 = build_fp16_blob(&transpose_weight(&lw.w2, dim, hidden));
            let rms_att = build_fp16_blob(&lw.rms_att);
            let rms_ffn = build_fp16_blob(&lw.rms_ffn);

            let mut datas: Vec<&[u8]> = Vec::new();
            for name in &names {
                match *name {
                    "@model_path/weights/rms_att.bin" => datas.push(&rms_att),
                    "@model_path/weights/rms_ffn.bin" => datas.push(&rms_ffn),
                    "@model_path/weights/wq.bin" => datas.push(&wq),
                    "@model_path/weights/wk.bin" => datas.push(&wk),
                    "@model_path/weights/wv.bin" => datas.push(&wv),
                    "@model_path/weights/wo.bin" => datas.push(&wo),
                    "@model_path/weights/w1.bin" => datas.push(&w1),
                    "@model_path/weights/w3.bin" => datas.push(&w3),
                    "@model_path/weights/w2.bin" => datas.push(&w2),
                    "@model_path/weights/rope_cos.bin" => datas.push(rope_cos),
                    "@model_path/weights/rope_sin.bin" => datas.push(rope_sin),
                    "@model_path/weights/mask.bin" => datas.push(mask),
                    _ => datas.push(&rms_att),
                }
            }

            let kernel = super::ane_bridge::AneKernel::compile_multi_weights(
                &result.mil_text,
                &names,
                &datas,
                &[result.input_bytes],
                &[result.output_bytes],
            )
            .map_err(|e| format!("fused_layer_fwd layer {l}: {e}"))?;

            kernels.push(kernel);
        }

        let elapsed = t0.elapsed();
        tracing::info!(
            "primed {} fused layer fwd kernels ({}) in {:.1}ms",
            kernels.len(),
            if train { "train" } else { "infer" },
            elapsed.as_secs_f64() * 1000.0,
        );
        self.fused_layer_fwd_output_bytes = result.output_bytes;
        self.fused_layer_fwd_kernels = Some(kernels);
        Ok(())
    }

    /// Evaluate fused full-layer forward for an MHA layer.
    ///
    /// Returns `(next_x, packed_acts)` where:
    /// - `next_x`: `[dim * seq]` fp32 — the layer output (for next layer input)
    /// - `packed_acts`: `[(7*dim + 2*hidden) * seq]` fp32 — packed activations for backward
    ///   Layout: `xout[dim] | xnorm[dim] | qf[dim] | kf[dim] | vf[dim] | x2[dim] | h1[hidden] | h3[hidden]`
    pub fn eval_fused_layer_fwd(
        &self,
        layer: usize,
        x_cur: &[f32],
        cfg: &super::ane_mil::MilConfig,
    ) -> Option<Result<(Vec<f32>, Vec<f32>), String>> {
        let kernels = self.fused_layer_fwd_kernels.as_ref()?;
        // MHA layers are stored sequentially in the kernels vec (GDN layers skipped)
        // For now, assume layer index maps directly
        if layer >= kernels.len() {
            return None;
        }
        let kernel = &kernels[layer];
        let dim = cfg.dim;
        let seq = cfg.seq_len;

        // Write input
        let input_bytes =
            unsafe { std::slice::from_raw_parts(x_cur.as_ptr() as *const u8, x_cur.len() * 4) };
        kernel.write_input(0, input_bytes);

        if let Err(e) = kernel.eval() {
            return Some(Err(format!("fused_layer_fwd eval layer {layer}: {e}")));
        }

        // Read output
        let mut buf = vec![0u8; self.fused_layer_fwd_output_bytes];
        kernel.read_output(0, &mut buf);
        let full: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Split: first dim*seq is the layer output, rest is packed activations
        let next_x = full[..dim * seq].to_vec();
        let acts = full[dim * seq..].to_vec();
        Some(Ok((next_x, acts)))
    }

    /// Prime per-layer fused FFN forward kernels (RMSNorm + W1×SiLU×W3 + W2 + residual).
    pub fn prime_fused_ffn_fwd<W: WeightSource>(
        &mut self,
        cfg: &super::ane_mil::MilConfig,
        model: &W,
        train: bool,
    ) -> Result<(), String> {
        let n_layers = model.n_layers();
        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;
        let t0 = std::time::Instant::now();

        let mut train_cfg = cfg.clone();
        train_cfg.has_lm_head = train;
        let result = super::ane_mil::gen_fused_ffn_fwd_blob(&train_cfg);
        let names: Vec<&str> = result.weight_names.iter().copied().collect();

        let mut kernels = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let lw_cow = model.layer(l);
            let lw = &*lw_cow;
            if lw.gdn.is_some() {
                continue;
            } // GDN layers use separate path

            let rms = build_fp16_blob(&lw.rms_ffn);
            let w1 = build_fp16_blob(&transpose_weight(&lw.w1, hidden, dim));
            let w3 = build_fp16_blob(&transpose_weight(&lw.w3, hidden, dim));
            let w2 = build_fp16_blob(&transpose_weight(&lw.w2, dim, hidden));

            let datas: Vec<&[u8]> = vec![&rms, &w1, &w3, &w2];
            let kernel = super::ane_bridge::AneKernel::compile_multi_weights(
                &result.mil_text,
                &names,
                &datas,
                &[result.input_bytes],
                &[result.output_bytes],
            )
            .map_err(|e| format!("fused_ffn_fwd layer {l}: {e}"))?;
            kernels.push(kernel);
        }

        let elapsed = t0.elapsed();
        tracing::info!(
            "primed {} fused FFN fwd kernels ({}) in {:.1}ms",
            kernels.len(),
            if train { "train" } else { "infer" },
            elapsed.as_secs_f64() * 1000.0,
        );
        self.fused_ffn_fwd_output_bytes = result.output_bytes;
        self.fused_ffn_fwd_kernels = Some(kernels);
        Ok(())
    }

    /// Evaluate fused FFN forward for an MHA layer.
    pub fn eval_fused_ffn_fwd(
        &self,
        layer: usize,
        x2: &[f32],
        cfg: &super::ane_mil::MilConfig,
    ) -> Option<Result<(Vec<f32>, Vec<f32>), String>> {
        let kernels = self.fused_ffn_fwd_kernels.as_ref()?;
        if layer >= kernels.len() {
            return None;
        }
        let kernel = &kernels[layer];
        let dim = cfg.dim;
        let seq = cfg.seq_len;

        let bytes = unsafe { std::slice::from_raw_parts(x2.as_ptr() as *const u8, x2.len() * 4) };
        kernel.write_input(0, bytes);
        if let Err(e) = kernel.eval() {
            return Some(Err(format!("fused_ffn_fwd eval layer {layer}: {e}")));
        }

        let mut buf = vec![0u8; self.fused_ffn_fwd_output_bytes];
        kernel.read_output(0, &mut buf);
        let full: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let next_x = full[..dim * seq].to_vec();
        let acts = full[dim * seq..].to_vec();
        Some(Ok((next_x, acts)))
    }

    pub fn has_fused_ffn_fwd(&self) -> bool {
        self.fused_ffn_fwd_kernels.is_some()
    }

    /// Prime per-layer fused GDN projection kernels (QKV+A+B+Z) and O projection.
    pub fn prime_gdn_proj_kernels<W: WeightSource>(
        &mut self,
        cfg: &super::ane_mil::MilConfig,
        model: &W,
    ) -> Result<(), String> {
        let n_layers = model.n_layers();
        let dim = cfg.dim;
        let h_k = cfg.linear_n_heads;
        let d_k = cfg.linear_head_dim;
        let h_v = cfg.linear_n_value_heads;
        let d_v = cfg.linear_value_head_dim;
        let key_dim = h_k * d_k;
        let value_dim = h_v * d_v;
        let qkv_dim = 2 * key_dim + value_dim;

        let proj_result = super::ane_mil::gen_fused_gdn_proj(cfg);
        let o_result = super::ane_mil::gen_blobfile_matmul(value_dim, dim, cfg.seq_len);

        let mut proj_kernels = Vec::with_capacity(n_layers);
        let mut o_kernels = Vec::with_capacity(n_layers);
        let t0 = std::time::Instant::now();

        for l in 0..n_layers {
            let lw_cow = model.layer(l);
            let lw = &*lw_cow;

            if lw.gdn.is_none() {
                proj_kernels.push(None);
                o_kernels.push(None);
                continue;
            }
            let gdn = lw.gdn.as_ref().unwrap();

            // Transpose weights to [in, out] for BLOBFILE matmul layout
            let wqkv_t = transpose_weight(&gdn.qkv_proj, qkv_dim, dim);
            let wa_t = transpose_weight(&gdn.a_proj, h_v, dim);
            let wb_t = transpose_weight(&gdn.b_proj, h_v, dim);
            let wz_t = transpose_weight(&gdn.z_proj, value_dim, dim);
            let wo_t = transpose_weight(&gdn.o_proj, dim, value_dim);

            let wqkv_blob = build_fp16_blob(&wqkv_t);
            let wa_blob = build_fp16_blob(&wa_t);
            let wb_blob = build_fp16_blob(&wb_t);
            let wz_blob = build_fp16_blob(&wz_t);

            let names: Vec<&str> = proj_result.weight_names.iter().copied().collect();
            let datas: Vec<&[u8]> = vec![&wqkv_blob, &wa_blob, &wb_blob, &wz_blob];
            match super::ane_bridge::AneKernel::compile_multi_weights(
                &proj_result.mil_text,
                &names,
                &datas,
                &[proj_result.input_bytes],
                &[proj_result.output_bytes],
            ) {
                Ok(k) => proj_kernels.push(Some(k)),
                Err(_) => {
                    proj_kernels.push(None);
                    o_kernels.push(None);
                    continue; // skip O proj too
                }
            }

            // O projection: BLOBFILE matmul on ANE (~0.5ms vs ~5ms CPU cblas_sgemm).
            // Budget: 30 GDN o_proj + 63 existing = 93, well under ~119 loaded limit.
            let wo_blob = build_fp16_blob(&wo_t);
            let o_names: Vec<&str> = o_result.weight_names.iter().copied().collect();
            let o_datas: Vec<&[u8]> = vec![&wo_blob];
            match super::ane_bridge::AneKernel::compile_multi_weights(
                &o_result.mil_text,
                &o_names,
                &o_datas,
                &[o_result.input_bytes],
                &[o_result.output_bytes],
            ) {
                Ok(k) => o_kernels.push(Some(k)),
                Err(_) => o_kernels.push(None), // graceful fallback to CPU matmul
            }
        }

        let gdn_count = proj_kernels.iter().filter(|k| k.is_some()).count();
        let o_count = o_kernels.iter().filter(|k| k.is_some()).count();
        let elapsed = t0.elapsed();
        tracing::info!(
            "primed {gdn_count}/{n_layers} GDN proj + {o_count} o_proj in {:.1}ms",
            elapsed.as_secs_f64() * 1000.0,
        );
        self.fused_gdn_proj_kernels = Some(proj_kernels);
        self.gdn_o_proj_kernels = Some(o_kernels);
        Ok(())
    }

    /// Compile per-layer GDN pre-recurrence split kernels with real weights.
    ///
    /// Each GDN layer gets its own fused kernel (4 BLOBFILEs baked at compile time).
    /// conv_bias fix: models without conv bias store an empty Vec — pad to expected size.
    /// Max seq_len for non-chunked GDN pre-recurrence (ANE SRAM limit at 35B dims).
    const GDN_PRE_RECUR_MAX_SEQ: usize = 256;

    pub fn prime_gdn_pre_recurrence_kernels<W: WeightSource>(
        &mut self,
        cfg: &super::ane_mil::MilConfig,
        model: &W,
    ) {
        let n_layers = model.n_layers();
        let full_seq = cfg.seq_len;
        let kernel_size = cfg.conv_kernel_size;

        // Pre-compute expected bias size for empty-bias fix
        let h_k = cfg.linear_n_heads;
        let d_k = cfg.linear_head_dim;
        let h_v = cfg.linear_n_value_heads;
        let d_v = cfg.linear_value_head_dim;
        let qkv_dim = 2 * h_k * d_k + h_v * d_v;

        // For large seq_lens, use split approach: CPU conv + ANE post-conv.
        // ANE can't run conv with pad=[0,0,0,0] (Program Inference error),
        // so we run conv+SiLU on CPU and feed post-SiLU data to ANE post-conv kernel.
        let chunked = full_seq > Self::GDN_PRE_RECUR_MAX_SEQ;
        let (chunk_size, overlap) = if chunked {
            let chunk = Self::GDN_PRE_RECUR_MAX_SEQ;
            let olap = kernel_size.saturating_sub(1);
            tracing::info!(
                "GDN pre-recurrence: chunked mode seq={full_seq} → chunk={chunk} overlap={olap} (CPU conv + ANE post-conv)",
            );
            (chunk, olap)
        } else {
            (0, 0)
        };

        // Generate the appropriate MIL
        let pre_r = if chunked {
            let mut chunk_cfg = cfg.clone();
            chunk_cfg.seq_len = Self::GDN_PRE_RECUR_MAX_SEQ;
            // Post-conv kernel: no conv, just SiLU/RMSNorm/GQA/decay/gate (2 BLOBFILEs)
            super::ane_mil::gen_gdn_post_conv_fwd(&chunk_cfg)
        } else {
            // Fused kernel: conv + SiLU + everything (4 BLOBFILEs)
            super::ane_mil::gen_gdn_pre_recurrence_fwd(cfg)
        };

        let mut kernels: Vec<Option<super::ane_bridge::AneKernel>> = Vec::with_capacity(n_layers);
        let mut conv_weights_per_layer: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(n_layers);
        let t0 = std::time::Instant::now();

        for l in 0..n_layers {
            // Only GDN (linear attention) layers have pre-recurrence.
            // MHA layers may have gdn weights populated but different head geometry.
            if !cfg.is_linear_attn_layer(l) {
                kernels.push(None);
                if chunked {
                    conv_weights_per_layer.push((vec![], vec![]));
                }
                continue;
            }
            let lw_cow = model.layer(l);
            let lw = &*lw_cow;
            if let Some(gdn) = lw.gdn.as_ref() {
                let conv_bias = if gdn.conv_bias.is_empty() {
                    vec![0.0f32; qkv_dim]
                } else {
                    gdn.conv_bias.clone()
                };

                let (blobs, names): (Vec<Vec<u8>>, Vec<&str>) = if chunked {
                    // Post-conv kernel: only a_log + dt_bias
                    (
                        vec![build_fp16_blob(&gdn.a_log), build_fp16_blob(&gdn.dt_bias)],
                        pre_r.weight_names.iter().copied().collect(),
                    )
                } else {
                    // Fused kernel: conv_w + conv_b + a_log + dt_bias
                    (
                        vec![
                            build_fp16_blob(&gdn.conv_weight),
                            build_fp16_blob(&conv_bias),
                            build_fp16_blob(&gdn.a_log),
                            build_fp16_blob(&gdn.dt_bias),
                        ],
                        pre_r.weight_names.iter().copied().collect(),
                    )
                };

                // Store conv weights for CPU conv in chunked mode
                if chunked {
                    conv_weights_per_layer.push((gdn.conv_weight.clone(), conv_bias));
                }

                let datas: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();
                match super::ane_bridge::AneKernel::compile_multi_weights(
                    &pre_r.mil_text,
                    &names,
                    &datas,
                    &[pre_r.input_bytes],
                    &[pre_r.output_bytes],
                ) {
                    Ok(k) => kernels.push(Some(k)),
                    Err(e) => {
                        tracing::warn!("GDN pre-recurrence layer {l} compile failed: {e}");
                        kernels.push(None);
                    }
                }
            } else {
                kernels.push(None);
                if chunked {
                    conv_weights_per_layer.push((vec![], vec![]));
                }
            }
        }

        let count = kernels.iter().filter(|k| k.is_some()).count();
        let elapsed = t0.elapsed();
        tracing::info!(
            "GDN pre-recurrence: {count}/{n_layers} per-layer kernels in {:.1}ms",
            elapsed.as_secs_f64() * 1000.0
        );
        self.gdn_pre_recur_output_bytes = pre_r.output_bytes;
        self.gdn_pre_recur_input_bytes = pre_r.input_bytes;
        self.gdn_pre_recur_chunk = chunk_size;
        self.gdn_pre_recur_overlap = overlap;
        self.gdn_pre_recur_per_layer = Some(kernels);
        self.gdn_pre_recur_conv_weights = if chunked {
            Some(conv_weights_per_layer)
        } else {
            None
        };
    }

    /// Evaluate fused GDN pre-recurrence on ANE.
    ///
    /// Input: qkv|a|b concatenated `[in_ch, seq]` fp32.
    /// For large seq_lens, automatically chunks into smaller pieces with conv overlap.
    /// Returns: (q_exp, k_exp, v, g, beta) — same as `cpu_gdn_pre_recurrence`.
    pub fn eval_gdn_pre_recurrence(
        &self,
        layer: usize,
        qkv: &[f32],
        a: &[f32],
        b: &[f32],
        cfg: &super::ane_mil::MilConfig,
    ) -> Option<Result<super::ane_forward::GdnPreRecurrenceOutput, String>> {
        let kernels = self.gdn_pre_recur_per_layer.as_ref()?;
        let kernel = kernels[layer].as_ref()?;

        let h_k = cfg.linear_n_heads;
        let d_k = cfg.linear_head_dim;
        let h_v = cfg.linear_n_value_heads;
        let d_v = cfg.linear_value_head_dim;
        let key_dim = h_k * d_k;
        let value_dim = h_v * d_v;
        let qkv_dim = 2 * key_dim + value_dim;
        let seq = cfg.seq_len;

        if self.gdn_pre_recur_chunk > 0 {
            return self.eval_gdn_pre_recurrence_chunked(
                kernel, qkv, a, b, cfg, qkv_dim, key_dim, value_dim, layer,
            );
        }

        // Non-chunked path: single kernel call
        let in_ch = qkv_dim + 2 * h_v;
        let mut input = Vec::with_capacity(in_ch * seq);
        input.extend_from_slice(qkv);
        input.extend_from_slice(a);
        input.extend_from_slice(b);
        debug_assert_eq!(input.len(), in_ch * seq);

        let input_bytes =
            unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
        kernel.write_input(0, input_bytes);

        if let Err(e) = kernel.eval() {
            return Some(Err(format!("pre-recurrence eval layer {layer}: {e}")));
        }

        let mut out_buf = vec![0u8; self.gdn_pre_recur_output_bytes];
        kernel.read_output(0, &mut out_buf);
        let out: Vec<f32> = out_buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let q_dim = h_v * d_k;
        let mut offset = 0;
        let q_exp = out[offset..offset + q_dim * seq].to_vec();
        offset += q_dim * seq;
        let k_exp = out[offset..offset + q_dim * seq].to_vec();
        offset += q_dim * seq;
        let v_raw = out[offset..offset + value_dim * seq].to_vec();
        offset += value_dim * seq;
        let g = out[offset..offset + h_v * seq].to_vec();
        offset += h_v * seq;
        let beta = out[offset..offset + h_v * seq].to_vec();

        Some(Ok(super::ane_forward::GdnPreRecurrenceOutput {
            q_exp,
            k_exp,
            v_raw,
            g,
            beta,
        }))
    }

    /// Chunked eval: CPU conv+SiLU per-chunk with overlap, then ANE post-conv kernel.
    ///
    /// ANE can't run conv with pad=[0,0,0,0] (Program Inference error), so we run the
    /// causal depthwise conv1d + SiLU on CPU with proper overlap context from previous
    /// chunk data, then feed the post-SiLU QKV + A + B to the ANE post-conv kernel which
    /// does RMSNorm + GQA expansion + decay/gate computation.
    fn eval_gdn_pre_recurrence_chunked(
        &self,
        kernel: &super::ane_bridge::AneKernel,
        qkv: &[f32],
        a: &[f32],
        b: &[f32],
        cfg: &super::ane_mil::MilConfig,
        qkv_dim: usize,
        _key_dim: usize,
        value_dim: usize,
        layer: usize,
    ) -> Option<Result<super::ane_forward::GdnPreRecurrenceOutput, String>> {
        let h_v = cfg.linear_n_value_heads;
        let d_k = cfg.linear_head_dim;
        let seq = cfg.seq_len;
        let chunk = self.gdn_pre_recur_chunk;
        let overlap = self.gdn_pre_recur_overlap;
        let conv_kernel = cfg.conv_kernel_size;

        // Get conv weights for this layer
        let conv_ws = self.gdn_pre_recur_conv_weights.as_ref()?;
        let (conv_w, conv_b) = &conv_ws[layer];
        if conv_w.is_empty() {
            return None; // not a GDN layer
        }

        // Post-conv kernel input: [qkv_dim + 2*h_v, chunk] (qkv_silu | a | b)
        let in_ch = qkv_dim + 2 * h_v;

        // Output accumulators
        let q_dim = h_v * d_k;
        let out_ch = 2 * q_dim + value_dim + 2 * h_v;
        let mut full_output = vec![0.0f32; out_ch * seq];

        let n_chunks = (seq + chunk - 1) / chunk;
        for ci in 0..n_chunks {
            let chunk_start = ci * chunk;
            let chunk_end = (chunk_start + chunk).min(seq);
            let actual_chunk = chunk_end - chunk_start;

            // 1. CPU: causal depthwise conv1d + SiLU on QKV for this chunk
            //    with overlap context from previous timesteps
            let mut qkv_silu = vec![0.0f32; qkv_dim * chunk];
            for c in 0..qkv_dim {
                for t in 0..actual_chunk {
                    let global_t = chunk_start + t;
                    let mut acc = 0.0f32;
                    for ki in 0..conv_kernel {
                        let src_t = global_t as isize - ki as isize;
                        let val = if src_t >= 0 && (src_t as usize) < seq {
                            qkv[c * seq + src_t as usize]
                        } else {
                            0.0
                        };
                        acc += val * conv_w[c * conv_kernel + ki];
                    }
                    if c < conv_b.len() {
                        acc += conv_b[c];
                    }
                    // SiLU: x * sigmoid(x)
                    qkv_silu[c * chunk + t] = acc / (1.0 + (-acc).exp());
                }
            }

            // 2. Build post-conv ANE input: [qkv_dim + 2*h_v, chunk]
            //    = qkv_silu | a_chunk | b_chunk
            let mut chunk_input = Vec::with_capacity(in_ch * chunk);
            chunk_input.extend_from_slice(&qkv_silu);
            // A: [h_v, chunk]
            for ch in 0..h_v {
                for t in 0..chunk {
                    let global_t = chunk_start + t;
                    let val = if global_t < seq {
                        a[ch * seq + global_t]
                    } else {
                        0.0
                    };
                    chunk_input.push(val);
                }
            }
            // B: [h_v, chunk]
            for ch in 0..h_v {
                for t in 0..chunk {
                    let global_t = chunk_start + t;
                    let val = if global_t < seq {
                        b[ch * seq + global_t]
                    } else {
                        0.0
                    };
                    chunk_input.push(val);
                }
            }

            debug_assert_eq!(chunk_input.len(), in_ch * chunk);

            // 3. ANE: post-conv kernel (RMSNorm + GQA + decay/gate)
            let input_bytes = unsafe {
                std::slice::from_raw_parts(chunk_input.as_ptr() as *const u8, chunk_input.len() * 4)
            };
            kernel.write_input(0, input_bytes);

            if let Err(e) = kernel.eval() {
                return Some(Err(format!(
                    "pre-recurrence post-conv eval layer {layer} chunk {ci}: {e}"
                )));
            }

            // Read chunk output: [out_ch, chunk]
            let mut out_buf = vec![0u8; self.gdn_pre_recur_output_bytes];
            kernel.read_output(0, &mut out_buf);
            let chunk_out: Vec<f32> = out_buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            // Copy valid portion into full output
            for ch in 0..out_ch {
                let src_off = ch * chunk;
                let dst_off = ch * seq + chunk_start;
                full_output[dst_off..dst_off + actual_chunk]
                    .copy_from_slice(&chunk_out[src_off..src_off + actual_chunk]);
            }
        }

        // Unpack from full_output [out_ch, seq]
        let mut offset = 0;
        let q_exp = full_output[offset..offset + q_dim * seq].to_vec();
        offset += q_dim * seq;
        let k_exp = full_output[offset..offset + q_dim * seq].to_vec();
        offset += q_dim * seq;
        let v_raw = full_output[offset..offset + value_dim * seq].to_vec();
        offset += value_dim * seq;
        let g = full_output[offset..offset + h_v * seq].to_vec();
        offset += h_v * seq;
        let beta = full_output[offset..offset + h_v * seq].to_vec();

        Some(Ok(super::ane_forward::GdnPreRecurrenceOutput {
            q_exp,
            k_exp,
            v_raw,
            g,
            beta,
        }))
    }

    /// Evaluate fused GDN projections on per-layer kernel.
    /// Returns (qkv_raw, a_raw, b_raw, z) or None if not primed.
    pub fn eval_gdn_proj(
        &self,
        layer: usize,
        xnorm: &[f32],
        cfg: &super::ane_mil::MilConfig,
    ) -> Option<Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), String>> {
        let kernels = self.fused_gdn_proj_kernels.as_ref()?;
        let kernel = kernels[layer].as_ref()?;

        let input_bytes =
            unsafe { std::slice::from_raw_parts(xnorm.as_ptr() as *const u8, xnorm.len() * 4) };
        kernel.write_input(0, input_bytes);

        if let Err(e) = kernel.eval() {
            return Some(Err(format!("fused_gdn_proj eval layer {layer}: {e}")));
        }

        let h_k = cfg.linear_n_heads;
        let d_k = cfg.linear_head_dim;
        let h_v = cfg.linear_n_value_heads;
        let d_v = cfg.linear_value_head_dim;
        let key_dim = h_k * d_k;
        let value_dim = h_v * d_v;
        let qkv_dim = 2 * key_dim + value_dim;
        let seq = cfg.seq_len;

        let out_ch = qkv_dim + 2 * h_v + value_dim;
        let mut buf = vec![0u8; out_ch * seq * 4];
        kernel.read_output(0, &mut buf);
        let out: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let qkv = out[..qkv_dim * seq].to_vec();
        let a = out[qkv_dim * seq..(qkv_dim + h_v) * seq].to_vec();
        let b = out[(qkv_dim + h_v) * seq..(qkv_dim + 2 * h_v) * seq].to_vec();
        let z = out[(qkv_dim + 2 * h_v) * seq..].to_vec();
        Some(Ok((qkv, a, b, z)))
    }

    /// Evaluate GDN O projection on per-layer kernel.
    /// `out_dim` is the model dim (output size of O projection).
    pub fn eval_gdn_o_proj(
        &self,
        layer: usize,
        gated: &[f32],
        out_dim: usize,
        seq: usize,
    ) -> Option<Result<Vec<f32>, String>> {
        let kernels = self.gdn_o_proj_kernels.as_ref()?;
        let kernel = kernels[layer].as_ref()?;

        let input_bytes =
            unsafe { std::slice::from_raw_parts(gated.as_ptr() as *const u8, gated.len() * 4) };
        kernel.write_input(0, input_bytes);

        if let Err(e) = kernel.eval() {
            return Some(Err(format!("gdn_o_proj eval layer {layer}: {e}")));
        }

        let mut buf = vec![0u8; out_dim * seq * 4];
        kernel.read_output(0, &mut buf);
        let out: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Some(Ok(out))
    }

    /// Evaluate forward fused FFN on per-layer kernel (IOSurface-resident weights).
    ///
    /// Only patches the activation slice via strided write (~512KB for seq=128),
    /// instead of copying the full 13MB buffer. Returns None if per-layer kernels
    /// aren't primed (caller should fall back to shared kernel path).
    pub fn eval_fwd_fused(
        &self,
        layer: usize,
        xnorm: &[f32],
        cfg: &super::ane_mil::MilConfig,
    ) -> Option<Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), String>> {
        let kernels = self.fwd_fused_kernels.as_ref()?;
        let kernel = &kernels[layer];
        let dim = cfg.dim;
        let seq = self.seq_len;
        let hidden = cfg.hidden_dim;
        let sp = seq + 3 * hidden;

        // Strided write: patch xnorm rows into IOSurface activation columns.
        // Each of dim rows has seq floats at the start, then 3*hidden weight floats.
        // xnorm is [dim, seq] contiguous. IOSurface stride is sp*4, source stride is seq*4.
        let xnorm_bytes =
            unsafe { std::slice::from_raw_parts(xnorm.as_ptr() as *const u8, xnorm.len() * 4) };
        kernel.write_input_strided(
            0,      // input idx
            0,      // dst_offset (activation starts at beginning of each row)
            sp * 4, // dst_stride (bytes per row in IOSurface)
            xnorm_bytes,
            seq * 4, // src_stride (bytes per row in xnorm)
            seq * 4, // chunk_bytes (one row of activation)
            dim,     // n_chunks (number of rows)
        );

        let spec =
            super::ane_mil::KernelSpec::for_kernel(cfg, super::ane_mil::KernelType::FusedFfn);
        if let Err(e) = kernel.eval() {
            return Some(Err(format!("per-layer fwd fused eval layer {layer}: {e}")));
        }
        let mut out_buf = vec![0u8; spec.output_bytes];
        kernel.read_output(0, &mut out_buf);
        Some(Ok(unpack_fused_ffn(&out_buf, cfg)))
    }

    /// Evaluate backward W2^T on per-layer kernel (IOSurface-resident weights).
    pub fn eval_bwd_w2t(
        &self,
        layer: usize,
        dffn: &[f32],
        cfg: &super::ane_mil::MilConfig,
    ) -> Option<Result<Vec<f32>, String>> {
        let kernels = self.bwd_w2t_kernels.as_ref()?;
        let kernel = &kernels[layer];
        let dim = cfg.dim;
        let seq = self.seq_len;
        let hidden = cfg.hidden_dim;
        let sp = seq + hidden;

        let dffn_bytes =
            unsafe { std::slice::from_raw_parts(dffn.as_ptr() as *const u8, dffn.len() * 4) };
        kernel.write_input_strided(0, 0, sp * 4, dffn_bytes, seq * 4, seq * 4, dim);

        let spec =
            super::ane_mil::KernelSpec::for_kernel(cfg, super::ane_mil::KernelType::FfnBwdW2t);
        if let Err(e) = kernel.eval() {
            return Some(Err(format!("per-layer bwd w2t eval layer {layer}: {e}")));
        }
        let mut out_buf = vec![0u8; spec.output_bytes];
        kernel.read_output(0, &mut out_buf);
        Some(Ok(out_buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()))
    }

    /// Evaluate backward W1^T+W3^T on per-layer kernel (IOSurface-resident weights).
    pub fn eval_bwd_w13t(
        &self,
        layer: usize,
        dh1: &[f32],
        dh3: &[f32],
        cfg: &super::ane_mil::MilConfig,
    ) -> Option<Result<Vec<f32>, String>> {
        let kernels = self.bwd_w13t_kernels.as_ref()?;
        let kernel = &kernels[layer];
        let hidden = cfg.hidden_dim;
        let seq = self.seq_len;
        let dim = cfg.dim;
        let sp = 2 * seq + 2 * dim;

        // dh1 goes to columns [0..seq], dh3 to columns [seq..2*seq]
        let dh1_bytes =
            unsafe { std::slice::from_raw_parts(dh1.as_ptr() as *const u8, dh1.len() * 4) };
        let dh3_bytes =
            unsafe { std::slice::from_raw_parts(dh3.as_ptr() as *const u8, dh3.len() * 4) };
        kernel.write_input_strided(0, 0, sp * 4, dh1_bytes, seq * 4, seq * 4, hidden);
        kernel.write_input_strided(
            0,
            seq * 4, // dh3 starts after dh1 in each row
            sp * 4,
            dh3_bytes,
            seq * 4,
            seq * 4,
            hidden,
        );

        let spec =
            super::ane_mil::KernelSpec::for_kernel(cfg, super::ane_mil::KernelType::FfnBwdW13t);
        if let Err(e) = kernel.eval() {
            return Some(Err(format!("per-layer bwd w13t eval layer {layer}: {e}")));
        }
        let mut out_buf = vec![0u8; spec.output_bytes];
        kernel.read_output(0, &mut out_buf);
        Some(Ok(out_buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()))
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
            moe: None,
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
            moe: None,
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
                moe: None,
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
            vocab_clusters: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub(crate) fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = vec![0u8; data.len() * 4];
    for (dst, &value) in bytes.chunks_exact_mut(4).zip(data.iter()) {
        dst.copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub(crate) fn bytes_to_f32_vec(data: &[u8]) -> Vec<f32> {
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
    fn test_qkvb_pack_eval_over_parameterized_attention() {
        init_ane();

        let mut cfg = test_cfg();
        cfg.dim = 64;
        cfg.n_heads = 4;
        cfg.head_dim_explicit = 32;
        cfg.attn_output_gate = true;
        cfg.seq_len = 16;

        let dim = cfg.dim;
        let ad = cfg.attn_dim();
        let qpd = cfg.q_proj_dim();
        let seq = cfg.seq_len;

        let dq: Vec<f32> = (0..qpd * seq).map(|i| (i % 41) as f32 * 0.001).collect();
        let dk: Vec<f32> = (0..ad * seq).map(|i| (i % 29) as f32 * 0.001).collect();
        let dv: Vec<f32> = (0..ad * seq).map(|i| (i % 37) as f32 * 0.001).collect();

        let mut wq_t = vec![0.0f32; qpd * dim];
        let mut wk_t = vec![0.0f32; ad * dim];
        let mut wv_t = vec![0.0f32; ad * dim];
        for i in 0..dim {
            wq_t[i * dim + i] = 1.0;
            if i < ad {
                wk_t[i * dim + (i % dim)] = 1.0;
                wv_t[i * dim + (i % dim)] = 1.0;
            }
        }

        let input_buf = pack_qkvb(&dq, &dk, &dv, &wq_t, &wk_t, &wv_t, &cfg);
        let spec = KernelSpec::for_kernel(&cfg, KernelType::Qkvb);
        let kernel = AneKernel::compile(
            &spec.mil_text,
            None,
            &[spec.input_bytes],
            &[spec.output_bytes],
        )
        .expect("qkvb compile failed for over-parameterized attention");

        kernel.write_input(0, &input_buf);
        kernel
            .eval()
            .expect("qkvb eval failed for over-parameterized attention");

        let mut out_buf = vec![0u8; spec.output_bytes];
        kernel.read_output(0, &mut out_buf);
        let output = bytes_to_f32_vec(&out_buf);

        assert_eq!(output.len(), dim * seq);
        let nonzero = output.iter().filter(|v| v.abs() > 1e-10).count();
        assert!(
            nonzero > 0,
            "qkvb output is all zeros for over-parameterized attention"
        );
    }

    /// Split QKV backward: 2 half-kernels produce same result as single kernel.
    ///
    /// Verifies the mathematical equivalence:
    ///   dx_single = WQ^T@dQ + WK^T@dK + WV^T@dV
    ///   dx_split  = (WQ_h0^T@dQ_first + WK^T@dK) + (WQ_h1^T@dQ_second + WV^T@dV)
    #[test]
    fn test_qkvb_split_matches_single() {
        init_ane();

        // Over-parameterized: qpd > ad (same as 35B structure)
        let mut cfg = test_cfg();
        cfg.dim = 64;
        cfg.n_heads = 4;
        cfg.head_dim_explicit = 32;
        cfg.seq_len = 16;

        let dim = cfg.dim;
        let ad = cfg.attn_dim(); // 4 * 32 = 128
        let qpd = cfg.q_proj_dim(); // 4 * 32 = 128 (same here, but split logic works regardless)
        let seq = cfg.seq_len;
        let half_qpd = qpd / 2;

        // Random-ish weights (non-identity to test real matmul)
        let wqt: Vec<f32> = (0..dim * qpd)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
            .collect();
        let wkt: Vec<f32> = (0..dim * ad)
            .map(|i| ((i % 83) as f32 - 41.0) * 0.01)
            .collect();
        let wvt: Vec<f32> = (0..dim * ad)
            .map(|i| ((i % 71) as f32 - 35.0) * 0.01)
            .collect();

        // Gradients
        let dq: Vec<f32> = (0..qpd * seq)
            .map(|i| ((i % 61) as f32 - 30.0) * 0.001)
            .collect();
        let dk: Vec<f32> = (0..ad * seq)
            .map(|i| ((i % 47) as f32 - 23.0) * 0.001)
            .collect();
        let dv: Vec<f32> = (0..ad * seq)
            .map(|i| ((i % 53) as f32 - 26.0) * 0.001)
            .collect();

        // --- Single kernel path ---
        let single_mil = gen_qkvb_blob(&cfg);
        let wqt_blob = build_fp16_blob(&wqt);
        let wkt_blob = build_fp16_blob(&wkt);
        let wvt_blob = build_fp16_blob(&wvt);
        let names: Vec<&str> = single_mil.weight_names.iter().copied().collect();
        let single_kernel = AneKernel::compile_multi_weights(
            &single_mil.mil_text,
            &names,
            &[&wqt_blob, &wkt_blob, &wvt_blob],
            &[single_mil.input_bytes],
            &[single_mil.output_bytes],
        )
        .expect("single qkvb compile failed");

        let mut single_input = Vec::with_capacity((qpd + 2 * ad) * seq);
        single_input.extend_from_slice(&dq);
        single_input.extend_from_slice(&dk);
        single_input.extend_from_slice(&dv);
        let single_bytes = unsafe {
            std::slice::from_raw_parts(single_input.as_ptr() as *const u8, single_input.len() * 4)
        };
        single_kernel.write_input(0, single_bytes);
        single_kernel.eval().expect("single qkvb eval failed");
        let mut single_out = vec![0u8; dim * seq * 4];
        single_kernel.read_output(0, &mut single_out);
        let dx_single: Vec<f32> = single_out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // --- Split kernel path ---
        let mil_a = gen_qkvb_blob_split_half(&cfg, 0);
        let mil_b = gen_qkvb_blob_split_half(&cfg, 1);

        // Split Wq^T into halves along reduction axis (columns)
        let wqt_h0: Vec<f32> = wqt
            .chunks_exact(qpd)
            .flat_map(|row| &row[..half_qpd])
            .copied()
            .collect();
        let wqt_h1: Vec<f32> = wqt
            .chunks_exact(qpd)
            .flat_map(|row| &row[half_qpd..])
            .copied()
            .collect();
        let wqt_h0_blob = build_fp16_blob(&wqt_h0);
        let wqt_h1_blob = build_fp16_blob(&wqt_h1);

        // Kernel A: Wq_h0 + Wk
        let names_a: Vec<&str> = mil_a.weight_names.iter().copied().collect();
        let kernel_a = AneKernel::compile_multi_weights(
            &mil_a.mil_text,
            &names_a,
            &[&wqt_h0_blob, &wkt_blob],
            &[mil_a.input_bytes],
            &[mil_a.output_bytes],
        )
        .expect("split kernel A compile failed");

        // Kernel B: Wq_h1 + Wv
        let names_b: Vec<&str> = mil_b.weight_names.iter().copied().collect();
        let kernel_b = AneKernel::compile_multi_weights(
            &mil_b.mil_text,
            &names_b,
            &[&wqt_h1_blob, &wvt_blob],
            &[mil_b.input_bytes],
            &[mil_b.output_bytes],
        )
        .expect("split kernel B compile failed");

        // Input A: dQ[0..half_qpd] | dK
        let mut input_a = Vec::with_capacity((half_qpd + ad) * seq);
        input_a.extend_from_slice(&dq[..half_qpd * seq]);
        input_a.extend_from_slice(&dk);
        let bytes_a =
            unsafe { std::slice::from_raw_parts(input_a.as_ptr() as *const u8, input_a.len() * 4) };
        kernel_a.write_input(0, bytes_a);
        kernel_a.eval().expect("split kernel A eval failed");

        // Input B: dQ[half_qpd..qpd] | dV
        let mut input_b = Vec::with_capacity((half_qpd + ad) * seq);
        input_b.extend_from_slice(&dq[half_qpd * seq..]);
        input_b.extend_from_slice(&dv);
        let bytes_b =
            unsafe { std::slice::from_raw_parts(input_b.as_ptr() as *const u8, input_b.len() * 4) };
        kernel_b.write_input(0, bytes_b);
        kernel_b.eval().expect("split kernel B eval failed");

        // Read and sum
        let mut out_a = vec![0u8; dim * seq * 4];
        let mut out_b = vec![0u8; dim * seq * 4];
        kernel_a.read_output(0, &mut out_a);
        kernel_b.read_output(0, &mut out_b);
        let dx_split: Vec<f32> = out_a
            .chunks_exact(4)
            .zip(out_b.chunks_exact(4))
            .map(|(a, b)| {
                f32::from_le_bytes([a[0], a[1], a[2], a[3]])
                    + f32::from_le_bytes([b[0], b[1], b[2], b[3]])
            })
            .collect();

        // Verify equivalence (fp16 quantization allows small error)
        assert_eq!(dx_single.len(), dx_split.len());
        let max_err = dx_single
            .iter()
            .zip(dx_split.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_val = dx_single.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            max_err < max_val * 0.01,
            "split QKV backward diverges from single: max_err={max_err}, max_val={max_val}, ratio={:.4}",
            max_err / max_val.max(1e-10),
        );
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
            moe: None,
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
            moe: None,
        };

        let base = ModelWeights {
            cfg: cfg.clone(),
            layers: vec![make_layer(0.0), make_layer(0.5)],
            rms_final: (0..dim).map(|i| 1.0 + i as f32 * 0.01).collect(),
            embed: (0..vocab * dim).map(|i| i as f32 * 0.001).collect(),
            vocab_size: vocab,
            lm_head: None,
            vocab_clusters: None,
        };

        let current = ModelWeights {
            cfg: cfg.clone(),
            layers: vec![make_layer(0.1), make_layer(0.6)],
            rms_final: (0..dim).map(|i| 1.1 + i as f32 * 0.01).collect(),
            embed: (0..vocab * dim).map(|i| 0.01 + i as f32 * 0.001).collect(),
            vocab_size: vocab,
            lm_head: None,
            vocab_clusters: None,
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

    #[test]
    fn test_dequant_3bit_contiguous_packing() {
        // Simulate MLX 3-bit quantization with contiguous bit packing.
        // 16 logical values packed into ceil(16*3/32) = 2 u32 words.
        let bits: usize = 3;
        let group_size: usize = 8;
        let cols: usize = 16;
        let rows: usize = 2;
        let packed_cols = (cols * bits + 31) / 32; // 2 u32 words per row
        assert_eq!(packed_cols, 2);

        // Pack values 0..15 (mod 8 to fit in 3 bits) contiguously into u32 words.
        // Value i occupies bits [i*3 .. i*3+3).
        let mut words = vec![0u32; rows * packed_cols];
        let vals: Vec<u32> = (0..cols as u32).map(|v| v % 8).collect();
        for r in 0..rows {
            for c in 0..cols {
                let bit_offset = c * bits;
                let word_idx = bit_offset / 32;
                let bit_within_word = bit_offset % 32;
                words[r * packed_cols + word_idx] |= vals[c] << bit_within_word;
                // Handle spanning: value 10 starts at bit 30, needs 2 bits in next word
                if bit_within_word + bits > 32 {
                    let overflow_bits = bit_within_word + bits - 32;
                    words[r * packed_cols + word_idx + 1] |= vals[c] >> (bits - overflow_bits);
                }
            }
        }

        // Convert words to bytes (LE)
        let mut weight_bytes = vec![0u8; words.len() * 4];
        for (i, w) in words.iter().enumerate() {
            weight_bytes[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }

        // Scales = 1.0, biases = 0.0 (identity transform)
        let n_groups = cols / group_size; // 2
        let scales = vec![1.0f32; rows * n_groups];
        let biases = vec![0.0f32; rows * n_groups];

        let out = super::dequant_nbit(
            &weight_bytes,
            &scales,
            &biases,
            rows,
            cols,
            group_size,
            bits,
        );
        assert_eq!(out.len(), rows * cols);

        // Check that all values match
        for r in 0..rows {
            for c in 0..cols {
                let expected = (c % 8) as f32;
                let got = out[r * cols + c];
                assert_eq!(
                    got, expected,
                    "row={r} col={c}: expected {expected}, got {got}"
                );
            }
        }
    }

    #[test]
    fn test_dequant_4bit_non_spanning() {
        // 4-bit: 8 values per u32, no spanning.
        let bits: usize = 4;
        let group_size: usize = 8;
        let cols: usize = 16;
        let rows: usize = 1;
        let packed_cols = (cols * bits + 31) / 32; // 2
        assert_eq!(packed_cols, 2);

        let vals: Vec<u32> = (0..16).map(|v| v % 16).collect();
        let mut words = vec![0u32; rows * packed_cols];
        for c in 0..cols {
            let elems_per_u32 = 32 / bits;
            let word_idx = c / elems_per_u32;
            let elem_idx = c % elems_per_u32;
            words[word_idx] |= vals[c] << (elem_idx * bits);
        }

        let mut weight_bytes = vec![0u8; words.len() * 4];
        for (i, w) in words.iter().enumerate() {
            weight_bytes[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }

        let n_groups = cols / group_size;
        let scales = vec![1.0f32; rows * n_groups];
        let biases = vec![0.0f32; rows * n_groups];

        let out = super::dequant_nbit(
            &weight_bytes,
            &scales,
            &biases,
            rows,
            cols,
            group_size,
            bits,
        );
        assert_eq!(out.len(), rows * cols);
        for c in 0..cols {
            assert_eq!(out[c], (c % 16) as f32, "col={c}");
        }
    }

    #[test]
    fn test_dense_cache_budget_plan_reserves_inference_memory() {
        let plan = super::DenseCacheBudgetPlan::from_policy(
            16 * super::GIB_BYTES,
            2 * super::GIB_BYTES,
            12 * super::GIB_BYTES,
            super::DenseCacheBudgetPolicy {
                fixed_headroom_bytes: 2 * super::GIB_BYTES,
                inference_reserve_bytes: 3 * super::GIB_BYTES,
                cache_fraction: 0.5,
                explicit_budget_bytes: None,
            },
        );

        assert_eq!(plan.reserved_bytes, 7 * super::GIB_BYTES);
        assert_eq!(plan.available_after_reserve_bytes, 9 * super::GIB_BYTES);
        assert_eq!(
            plan.dense_cache_budget_bytes,
            4 * super::GIB_BYTES + 512 * super::MIB_BYTES
        );
    }

    #[test]
    fn test_dense_cache_budget_plan_explicit_override_clamps_to_dense_total() {
        let plan = super::DenseCacheBudgetPlan::from_policy(
            32 * super::GIB_BYTES,
            super::GIB_BYTES,
            3 * super::GIB_BYTES,
            super::DenseCacheBudgetPolicy {
                fixed_headroom_bytes: 0,
                inference_reserve_bytes: 0,
                cache_fraction: 1.0,
                explicit_budget_bytes: Some(10 * super::GIB_BYTES),
            },
        );

        assert_eq!(plan.dense_cache_budget_bytes, 3 * super::GIB_BYTES);
        assert_eq!(plan.total_dense_layer_bytes, 3 * super::GIB_BYTES);
    }
}
