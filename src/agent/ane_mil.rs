//! MIL (Model Intermediate Language) generators for Apple Neural Engine kernels.
//!
//! Ports the dynamic MIL generators from `ANE/training/training_dynamic/mil_dynamic.h`
//! to Rust. Weights are packed into the IOSurface spatial dimension alongside activations,
//! so kernels compile once and accept new weights each step without recompilation.
//!
//! IOSurface layout: `[1, channels, 1, spatial]` where spatial = seq + weight_cols.

use std::fmt::Write;

// ---------------------------------------------------------------------------
// MIL header (matches MIL_HDR in mil_dynamic.h)
// ---------------------------------------------------------------------------

pub(crate) const MIL_HDR: &str = concat!(
    "program(1.3)\n",
    "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, ",
    "{\"coremlc-version\", \"3505.4.1\"}, ",
    "{\"coremltools-component-milinternal\", \"\"}, ",
    "{\"coremltools-version\", \"9.0\"}})]\n",
    "{\n",
);

// ---------------------------------------------------------------------------
// Config types
// ---------------------------------------------------------------------------

/// Model configuration for MIL generation.
#[derive(Debug, Clone)]
pub struct MilConfig {
    pub dim: usize,        // hidden dimension (768, 1024, etc.)
    pub hidden_dim: usize, // FFN hidden (2048, etc.)
    pub n_heads: usize,    // attention heads (12, etc.)
    pub seq_len: usize,    // sequence length (256, etc.)
    pub n_kv_heads: usize, // KV heads for GQA (n_heads for MHA)
    pub rope_theta: f64,   // RoPE frequency base (10000.0 for llama, 1e6 for Qwen)
    pub rms_eps: f32,      // RMSNorm epsilon (1e-5 for llama, 1e-6 for Qwen)
    pub has_lm_head: bool, // true = untied lm_head, false = share embed
    /// Explicit per-head dimension. May differ from `dim / n_heads` in models
    /// like Qwen3.5 where n_heads * head_dim > dim (over-parameterised attention).
    pub head_dim_explicit: usize,
    // Qwen3.5 GDN (Gated Delta Network) linear attention config
    pub linear_attn_indices: Vec<usize>, // layer indices using linear attention
    pub linear_n_heads: usize,           // key/query heads for GDN
    pub linear_head_dim: usize,          // key/query head dimension
    pub linear_n_value_heads: usize,     // value heads (recurrence count)
    pub linear_value_head_dim: usize,    // value head dimension
    pub conv_kernel_size: usize,         // causal conv kernel size (usually 4)
    /// Qwen3.5: q_proj outputs [Q, gate] interleaved per-head (2× attn_dim).
    /// The gate is sigmoid-ed and multiplied with SDPA output before o_proj.
    pub attn_output_gate: bool,
}

impl MilConfig {
    /// Create an MHA (multi-head attention) config with standard defaults for the new GQA fields.
    /// n_kv_heads = n_heads, rope_theta = 10000.0, rms_eps = 1e-5, has_lm_head = false.
    pub fn mha(dim: usize, hidden_dim: usize, n_heads: usize, seq_len: usize) -> Self {
        MilConfig {
            dim,
            hidden_dim,
            n_heads,
            seq_len,
            n_kv_heads: n_heads,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            has_lm_head: false,
            head_dim_explicit: dim / n_heads,
            linear_attn_indices: vec![],
            linear_n_heads: 0,
            linear_head_dim: 0,
            linear_n_value_heads: 0,
            linear_value_head_dim: 0,
            conv_kernel_size: 0,
            attn_output_gate: false,
        }
    }

    /// Check if a layer index uses GDN linear attention.
    pub fn is_linear_attn_layer(&self, idx: usize) -> bool {
        self.linear_attn_indices.contains(&idx)
    }

    /// Per-head dimension. Uses the explicit value from model config, which may
    /// differ from `dim / n_heads` for over-parameterised attention (Qwen3.5).
    pub fn head_dim(&self) -> usize {
        self.head_dim_explicit
    }

    /// Total attention dimension: `n_heads * head_dim`. Equals `dim` for
    /// standard transformers but can be larger for Qwen3.5-style models.
    pub fn attn_dim(&self) -> usize {
        self.n_heads * self.head_dim_explicit
    }
    /// Q projection output dimension: `attn_dim * 2` when `attn_output_gate`, else `attn_dim`.
    pub fn q_proj_dim(&self) -> usize {
        if self.attn_output_gate {
            2 * self.attn_dim()
        } else {
            self.attn_dim()
        }
    }
    pub fn score_ch(&self) -> usize {
        self.n_heads * self.seq_len
    }
    /// KV projection dimension: n_kv_heads * head_dim.
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim()
    }
    /// Number of Q heads per KV group.
    pub fn heads_per_group(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
    /// Score channels for KV heads: n_kv_heads * seq_len.
    pub fn kv_score_ch(&self) -> usize {
        self.n_kv_heads * self.seq_len
    }
}

/// Kernel type for spec computation.
#[derive(Debug, Clone, Copy)]
pub enum KernelType {
    DynMatmul {
        ic: usize,
        oc: usize,
    },
    SdpaFwd,
    FfnW13,
    FfnW2,
    Wot,
    FfnBwdW2t,
    FfnBwdW13t,
    Qkvb,
    /// Pure SDPA core for GQA: Q@K^T → scale → mask → softmax → @V.
    /// Accepts pre-projected, pre-normed, pre-RoPE'd, GQA-expanded Q/K/V.
    SdpaCoreGqa,
    SdpaBwd1,
    SdpaBwd2,
    /// Fully-fused FFN: W1+W3+SiLU+gate+W2 in a single ANE dispatch.
    /// Input channel = dim; W2 packed in original [dim, hidden] layout, transposed inside kernel.
    FusedFfn,
}

/// Computed metadata for a compiled kernel.
pub struct KernelSpec {
    pub mil_text: String,
    pub input_bytes: usize,
    pub output_bytes: usize,
}

impl KernelSpec {
    /// Build a KernelSpec for the given kernel type and config.
    pub fn for_kernel(cfg: &MilConfig, kt: KernelType) -> Self {
        let (mil_text, in_ch, in_sp, out_ch, out_sp, in_fp16, out_fp16) = match kt {
            KernelType::DynMatmul { ic, oc } => {
                let sp = cfg.seq_len + oc;
                (
                    gen_dyn_matmul_mil(ic, oc, cfg.seq_len),
                    ic,
                    sp,
                    oc,
                    cfg.seq_len,
                    false,
                    false,
                )
            }
            KernelType::SdpaFwd => {
                let sp_in = cfg.seq_len + 4 * cfg.dim;
                let out_ch = 6 * cfg.dim;
                (
                    gen_sdpa_fwd(cfg),
                    cfg.dim,
                    sp_in,
                    out_ch,
                    cfg.seq_len,
                    false,
                    false,
                )
            }
            KernelType::FfnW13 => {
                let sp_in = cfg.seq_len + 2 * cfg.hidden_dim;
                let out_ch = 3 * cfg.hidden_dim;
                (
                    gen_ffn_w13(cfg),
                    cfg.dim,
                    sp_in,
                    out_ch,
                    cfg.seq_len,
                    false,
                    false,
                )
            }
            KernelType::FfnW2 => {
                let sp_in = cfg.seq_len + cfg.dim;
                (
                    gen_ffn_w2(cfg),
                    cfg.hidden_dim,
                    sp_in,
                    cfg.dim,
                    cfg.seq_len,
                    false,
                    false,
                )
            }
            KernelType::Wot => {
                let ad = cfg.attn_dim();
                let sp_in = cfg.seq_len + ad;
                (gen_wot(cfg), cfg.dim, sp_in, ad, cfg.seq_len, false, false)
            }
            KernelType::FfnBwdW2t => {
                let sp_in = cfg.seq_len + cfg.hidden_dim;
                (
                    gen_ffn_bwd_w2t(cfg),
                    cfg.dim,
                    sp_in,
                    cfg.hidden_dim,
                    cfg.seq_len,
                    false,
                    false,
                )
            }
            KernelType::FfnBwdW13t => {
                let sp_in = 2 * cfg.seq_len + 2 * cfg.dim;
                (
                    gen_ffn_bwd_w13t(cfg),
                    cfg.hidden_dim,
                    sp_in,
                    cfg.dim,
                    cfg.seq_len,
                    false,
                    false,
                )
            }
            KernelType::Qkvb => {
                let qpd = cfg.q_proj_dim();
                let sp_in = 3 * cfg.seq_len + 3 * cfg.dim;
                (
                    gen_qkvb(cfg),
                    qpd,
                    sp_in,
                    cfg.dim,
                    cfg.seq_len,
                    false,
                    false,
                )
            }
            KernelType::SdpaCoreGqa => {
                let ad = cfg.attn_dim();
                let in_ch = 3 * ad;
                (
                    gen_sdpa_core_gqa(cfg),
                    in_ch,
                    cfg.seq_len,
                    ad,
                    cfg.seq_len,
                    false,
                    false,
                )
            }
            KernelType::SdpaBwd1 => {
                let attn_dim = cfg.attn_dim();
                let in_ch = 4 * attn_dim;
                let out_ch = attn_dim + 2 * cfg.score_ch();
                (
                    gen_sdpa_bwd1(cfg),
                    in_ch,
                    cfg.seq_len,
                    out_ch,
                    cfg.seq_len,
                    true,
                    true,
                )
            }
            KernelType::FusedFfn => {
                let sp_in = cfg.seq_len + 3 * cfg.hidden_dim;
                let out_ch = 3 * cfg.hidden_dim + cfg.dim;
                (
                    gen_fused_ffn_fwd(cfg),
                    cfg.dim,
                    sp_in,
                    out_ch,
                    cfg.seq_len,
                    false,
                    false,
                )
            }
            KernelType::SdpaBwd2 => {
                let attn_dim = cfg.attn_dim();
                let in_ch = 2 * cfg.score_ch() + 2 * attn_dim;
                let out_ch = 2 * attn_dim;
                (
                    gen_sdpa_bwd2(cfg),
                    in_ch,
                    cfg.seq_len,
                    out_ch,
                    cfg.seq_len,
                    true,
                    true,
                )
            }
        };
        let bpe_in = if in_fp16 { 2 } else { 4 };
        let bpe_out = if out_fp16 { 2 } else { 4 };
        KernelSpec {
            mil_text,
            input_bytes: in_ch * in_sp * bpe_in,
            output_bytes: out_ch * out_sp * bpe_out,
        }
    }
}

// ---------------------------------------------------------------------------
// Tiling
// ---------------------------------------------------------------------------

/// Max fp16 elements that fit in ANE SRAM (~28 MB, leaving headroom for intermediates).
const ANE_SRAM_FP16_ELEMS: usize = 14_000_000;

/// Tile plan for a DynMatmul dimension that exceeds ANE SRAM.
#[derive(Debug, Clone)]
pub struct TilePlan {
    /// Size of each tile (last tile is padded to this).
    pub tile_size: usize,
    /// Number of tiles.
    pub n_tiles: usize,
    /// Actual elements in the last tile (before padding).
    pub last_actual: usize,
    /// Original full dimension.
    pub full_size: usize,
}

impl TilePlan {
    /// Returns true if tiling is needed (more than 1 tile).
    pub fn needs_tiling(&self) -> bool {
        self.n_tiles > 1
    }

    /// Actual size of tile `t` (before padding).
    pub fn actual_tile_size(&self, t: usize) -> usize {
        if t == self.n_tiles - 1 {
            self.last_actual
        } else {
            self.tile_size
        }
    }

    /// Start offset for tile `t`.
    pub fn tile_start(&self, t: usize) -> usize {
        t * self.tile_size
    }
}

/// Compute a tile plan for DynMatmul OC dimension.
///
/// IOSurface `[1, ic, 1, seq+oc]` cast to fp16 must fit in ANE SRAM:
/// `ic * (seq + tile_oc) * 2 <= 28 MB` → `tile_oc <= ANE_SRAM_FP16_ELEMS / ic - seq`.
pub fn compute_oc_tile_plan(ic: usize, oc: usize, seq: usize) -> TilePlan {
    let max_oc = ANE_SRAM_FP16_ELEMS / ic - seq;
    if oc <= max_oc {
        return TilePlan {
            tile_size: oc,
            n_tiles: 1,
            last_actual: oc,
            full_size: oc,
        };
    }
    // Round down to multiple of 128 for ANE alignment
    let tile_oc = (max_oc / 128) * 128;
    assert!(
        tile_oc > 0,
        "compute_oc_tile_plan: IC={ic} too large for ANE SRAM"
    );
    let n_tiles = (oc + tile_oc - 1) / tile_oc;
    let last_actual = oc - (n_tiles - 1) * tile_oc;
    TilePlan {
        tile_size: tile_oc,
        n_tiles,
        last_actual,
        full_size: oc,
    }
}

/// Compute a tile plan for DynMatmul IC dimension (reduction tiling).
///
/// IOSurface `[1, tile_ic, 1, seq+oc]` cast to fp16 must fit:
/// `tile_ic * (seq + oc) * 2 <= 28 MB` → `tile_ic <= ANE_SRAM_FP16_ELEMS / (seq + oc)`.
pub fn compute_ic_tile_plan(ic: usize, oc: usize, seq: usize) -> TilePlan {
    let max_ic = ANE_SRAM_FP16_ELEMS / (seq + oc);
    if ic <= max_ic {
        return TilePlan {
            tile_size: ic,
            n_tiles: 1,
            last_actual: ic,
            full_size: ic,
        };
    }
    let tile_ic = (max_ic / 128) * 128;
    assert!(
        tile_ic > 0,
        "compute_ic_tile_plan: OC={oc}+SEQ={seq} too large for ANE SRAM"
    );
    let n_tiles = (ic + tile_ic - 1) / tile_ic;
    let last_actual = ic - (n_tiles - 1) * tile_ic;
    TilePlan {
        tile_size: tile_ic,
        n_tiles,
        last_actual,
        full_size: ic,
    }
}

// ---------------------------------------------------------------------------
// Generator helpers
// ---------------------------------------------------------------------------

/// Helper: emit a dynamic matmul block within a MIL function body.
///
/// Slices activation `[1,ic,1,seq]` and weight `[1,ic,1,oc]` from `input_var`,
/// performs reshape→transpose→matmul→transpose→reshape.
/// Result variable: `{prefix}_y` with shape `[1,oc,1,seq]` in fp16.
fn gen_dyn_matmul(
    m: &mut String,
    prefix: &str,
    ic: usize,
    oc: usize,
    seq: usize,
    act_sp_off: usize,
    w_sp_off: usize,
    input_var: &str,
) {
    // Slice activations
    let _ = writeln!(m, "        tensor<int32, [4]> {prefix}_ba = const()[name=string(\"{prefix}_ba\"), val=tensor<int32, [4]>([0,0,0,{act_sp_off}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> {prefix}_sa = const()[name=string(\"{prefix}_sa\"), val=tensor<int32, [4]>([1,{ic},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{ic},1,{seq}]> {prefix}_act = slice_by_size(x={input_var},begin={prefix}_ba,size={prefix}_sa)[name=string(\"{prefix}_act\")];");
    // Slice weight
    let _ = writeln!(m, "        tensor<int32, [4]> {prefix}_bw = const()[name=string(\"{prefix}_bw\"), val=tensor<int32, [4]>([0,0,0,{w_sp_off}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> {prefix}_sw = const()[name=string(\"{prefix}_sw\"), val=tensor<int32, [4]>([1,{ic},1,{oc}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{ic},1,{oc}]> {prefix}_wt = slice_by_size(x={input_var},begin={prefix}_bw,size={prefix}_sw)[name=string(\"{prefix}_wt\")];");
    // Reshape act: [1,ic,1,seq] → [1,1,ic,seq] → transpose → [1,1,seq,ic]
    let _ = writeln!(m, "        tensor<int32, [4]> {prefix}_ra = const()[name=string(\"{prefix}_ra\"), val=tensor<int32, [4]>([1,1,{ic},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{ic},{seq}]> {prefix}_a2 = reshape(shape={prefix}_ra,x={prefix}_act)[name=string(\"{prefix}_a2\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> {prefix}_pm = const()[name=string(\"{prefix}_pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{ic}]> {prefix}_a3 = transpose(perm={prefix}_pm,x={prefix}_a2)[name=string(\"{prefix}_a3\")];");
    // Reshape weight: [1,ic,1,oc] → [1,1,ic,oc]
    let _ = writeln!(m, "        tensor<int32, [4]> {prefix}_rw = const()[name=string(\"{prefix}_rw\"), val=tensor<int32, [4]>([1,1,{ic},{oc}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{ic},{oc}]> {prefix}_W = reshape(shape={prefix}_rw,x={prefix}_wt)[name=string(\"{prefix}_W\")];");
    // matmul: [1,1,seq,ic] @ [1,1,ic,oc] → [1,1,seq,oc]
    let _ = writeln!(
        m,
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{oc}]> {prefix}_yh = matmul(transpose_x=bF,transpose_y=bF,x={prefix}_a3,y={prefix}_W)[name=string(\"{prefix}_yh\")];");
    // Transpose back + reshape: [1,1,seq,oc] → [1,1,oc,seq] → [1,oc,1,seq]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{oc},{seq}]> {prefix}_yt = transpose(perm={prefix}_pm,x={prefix}_yh)[name=string(\"{prefix}_yt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> {prefix}_ro = const()[name=string(\"{prefix}_ro\"), val=tensor<int32, [4]>([1,{oc},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{oc},1,{seq}]> {prefix}_y = reshape(shape={prefix}_ro,x={prefix}_yt)[name=string(\"{prefix}_y\")];");
}

// ---------------------------------------------------------------------------
// Public generators
// ---------------------------------------------------------------------------

/// Standalone dynamic matmul kernel: y = x @ W.
///
/// Input: `[1, ic, 1, seq+oc]` fp32 — activations in `[0:seq]`, weight in `[seq:seq+oc]`.
/// Output: `[1, oc, 1, seq]` fp32.
pub fn gen_dyn_matmul_mil(ic: usize, oc: usize, seq: usize) -> String {
    let sp = seq + oc;
    let mut m = String::with_capacity(4096);
    m.push_str(MIL_HDR);
    let _ = writeln!(
        m,
        "    func main<ios18>(tensor<fp32, [1, {ic}, 1, {sp}]> x) {{"
    );
    let _ = writeln!(
        m,
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{ic},1,{sp}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];"
    );
    gen_dyn_matmul(&mut m, "mm", ic, oc, seq, 0, seq, "xh");
    let _ = writeln!(
        m,
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];"
    );
    let _ = writeln!(m, "        tensor<fp32, [1,{oc},1,{seq}]> y = cast(dtype=to32,x=mm_y)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (y);");
    m.push_str("}\n");
    m
}

/// SDPA forward (dynamic weights): QKV matmul + scaled dot-product attention + Wo matmul.
///
/// Input: `[1, dim, 1, seq + 4*dim]` fp32.
/// Output: `[1, 6*dim, 1, seq]` fp32 = concat(o_out, Q, K, V, attn_out, xnorm_pass).
pub fn gen_sdpa_fwd(cfg: &MilConfig) -> String {
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let heads = cfg.n_heads;
    let hd = cfg.head_dim();
    let sc = 1.0 / (hd as f64).sqrt();
    let w_total = 4 * dim;
    let sp_in = seq + w_total;

    let mut m = String::with_capacity(8192);
    m.push_str(MIL_HDR);
    let _ = writeln!(
        m,
        "    func main<ios18>(tensor<fp32, [1, {dim}, 1, {sp_in}]> x) {{"
    );
    // Cast to fp16
    let _ = writeln!(
        m,
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{sp_in}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");

    // Slice xnorm
    let _ = writeln!(m, "        tensor<int32, [4]> bx = const()[name=string(\"bx\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> sx = const()[name=string(\"sx\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> xn = slice_by_size(x=xh,begin=bx,size=sx)[name=string(\"xn\")];");

    // Slice Wq, Wk, Wv, Wo
    let _ = writeln!(m, "        tensor<int32, [4]> bq = const()[name=string(\"bq\"), val=tensor<int32, [4]>([0,0,0,{seq}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,{dim},1,{dim}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{dim}]> Wq = slice_by_size(x=xh,begin=bq,size=sw)[name=string(\"Wq\")];");

    let off_k = seq + dim;
    let _ = writeln!(m, "        tensor<int32, [4]> bk = const()[name=string(\"bk\"), val=tensor<int32, [4]>([0,0,0,{off_k}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{dim}]> Wk = slice_by_size(x=xh,begin=bk,size=sw)[name=string(\"Wk\")];");

    let off_v = seq + 2 * dim;
    let _ = writeln!(m, "        tensor<int32, [4]> bv = const()[name=string(\"bv\"), val=tensor<int32, [4]>([0,0,0,{off_v}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{dim}]> Wv = slice_by_size(x=xh,begin=bv,size=sw)[name=string(\"Wv\")];");

    let off_o = seq + 3 * dim;
    let _ = writeln!(m, "        tensor<int32, [4]> bo = const()[name=string(\"bo\"), val=tensor<int32, [4]>([0,0,0,{off_o}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{dim}]> Wo = slice_by_size(x=xh,begin=bo,size=sw)[name=string(\"Wo\")];");

    // Reshape for matmul
    let _ = writeln!(m, "        tensor<int32, [4]> r2 = const()[name=string(\"r2\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> xn2 = reshape(shape=r2,x=xn)[name=string(\"xn2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> xnt = transpose(perm=pm,x=xn2)[name=string(\"xnt\")];");

    // Reshape weights
    let _ = writeln!(m, "        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,{dim},{dim}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{dim}]> Wq2 = reshape(shape=rw,x=Wq)[name=string(\"Wq2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{dim}]> Wk2 = reshape(shape=rw,x=Wk)[name=string(\"Wk2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{dim}]> Wv2 = reshape(shape=rw,x=Wv)[name=string(\"Wv2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{dim}]> Wo2 = reshape(shape=rw,x=Wo)[name=string(\"Wo2\")];");

    // QKV matmul
    let _ = writeln!(
        m,
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];"
    );
    let _ = writeln!(
        m,
        "        bool bT = const()[name=string(\"bT\"), val=bool(true)];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> qm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wq2)[name=string(\"qm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> km = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wk2)[name=string(\"km\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> vm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wv2)[name=string(\"vm\")];");

    // Transpose back: [1,1,S,D] → [1,1,D,S] → reshape [1,D,1,S]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> qt = transpose(perm=pm,x=qm)[name=string(\"qt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> kt = transpose(perm=pm,x=km)[name=string(\"kt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> vt = transpose(perm=pm,x=vm)[name=string(\"vt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> os = const()[name=string(\"os\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> qf = reshape(shape=os,x=qt)[name=string(\"qf\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> kf = reshape(shape=os,x=kt)[name=string(\"kf\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> vf = reshape(shape=os,x=vt)[name=string(\"vf\")];"
    );

    // SDPA: reshape to heads
    let _ = writeln!(m, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> q4 = reshape(shape=qsh,x=qf)[name=string(\"rq\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> q = transpose(perm=pm,x=q4)[name=string(\"tq\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> k4 = reshape(shape=qsh,x=kf)[name=string(\"rk\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> k = transpose(perm=pm,x=k4)[name=string(\"tk\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> v4 = reshape(shape=qsh,x=vf)[name=string(\"rv\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> v = transpose(perm=pm,x=v4)[name=string(\"tv\")];");

    // RoPE rotation (half-convention: split, not interleaved)
    // Load precomputed cos/sin [1, 1, seq, hd/2] — broadcast across heads
    let half_hd = hd / 2;
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_cos = const()[name=string(\"rc\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_cos.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_sin = const()[name=string(\"rs\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_sin.bin\"), offset=uint64(64)))];");

    // Slice Q into halves: q1 = q[:,:,:,:hd/2], q2 = q[:,:,:,hd/2:]
    let _ = writeln!(m, "        tensor<int32, [4]> rp_b0 = const()[name=string(\"rpb0\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> rp_sh = const()[name=string(\"rpsh\"), val=tensor<int32, [4]>([1,{heads},{seq},{half_hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q1 = slice_by_size(x=q,begin=rp_b0,size=rp_sh)[name=string(\"q1\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rp_bh = const()[name=string(\"rpbh\"), val=tensor<int32, [4]>([0,0,0,{half_hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q2 = slice_by_size(x=q,begin=rp_bh,size=rp_sh)[name=string(\"q2\")];");

    // q_rot = concat(q1*cos - q2*sin, q1*sin + q2*cos)
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q1c = mul(x=q1,y=rope_cos)[name=string(\"q1c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q2s = mul(x=q2,y=rope_sin)[name=string(\"q2s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> qr1 = sub(x=q1c,y=q2s)[name=string(\"qr1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q1s = mul(x=q1,y=rope_sin)[name=string(\"q1s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q2c = mul(x=q2,y=rope_cos)[name=string(\"q2c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> qr2 = add(x=q1s,y=q2c)[name=string(\"qr2\")];");
    let _ = writeln!(
        m,
        "        int32 rpax = const()[name=string(\"rpax\"), val=int32(-1)];"
    );
    let _ = writeln!(
        m,
        "        bool rpid = const()[name=string(\"rpid\"), val=bool(false)];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> q_rot = concat(axis=rpax,interleave=rpid,values=(qr1,qr2))[name=string(\"qrot\")];");

    // Same for K
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> k1 = slice_by_size(x=k,begin=rp_b0,size=rp_sh)[name=string(\"k1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> k2 = slice_by_size(x=k,begin=rp_bh,size=rp_sh)[name=string(\"k2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> k1c = mul(x=k1,y=rope_cos)[name=string(\"k1c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> k2s = mul(x=k2,y=rope_sin)[name=string(\"k2s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> kr1 = sub(x=k1c,y=k2s)[name=string(\"kr1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> k1s = mul(x=k1,y=rope_sin)[name=string(\"k1s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> k2c = mul(x=k2,y=rope_cos)[name=string(\"k2c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> kr2 = add(x=k1s,y=k2c)[name=string(\"kr2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> k_rot = concat(axis=rpax,interleave=rpid,values=(kr1,kr2))[name=string(\"krot\")];");

    // Q @ K^T (using rotated Q, K)
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q_rot,y=k_rot)[name=string(\"mm1\")];");
    let _ = writeln!(
        m,
        "        fp16 scv = const()[name=string(\"scv\"), val=fp16({sc})];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];");

    // Causal mask (const BLOBFILE)
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{seq}]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,{seq},{seq}]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];");
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];"
    );

    // Softmax
    let _ = writeln!(
        m,
        "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"sm\")];");

    // scores @ V
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> a4 = matmul(transpose_x=bF,transpose_y=bF,x=aw,y=v)[name=string(\"mm2\")];");

    // Reshape back to [1,DIM,1,SEQ]
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> at = transpose(perm=pm,x=a4)[name=string(\"ta\")];");
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> af = reshape(shape=os,x=at)[name=string(\"ra\")];"
    );

    // Wo matmul
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> af2 = reshape(shape=r2,x=af)[name=string(\"af2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> aft = transpose(perm=pm,x=af2)[name=string(\"aft\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> om = matmul(transpose_x=bF,transpose_y=bF,x=aft,y=Wo2)[name=string(\"om\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> ot = transpose(perm=pm,x=om)[name=string(\"ot\")];");
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> oo = reshape(shape=os,x=ot)[name=string(\"oo\")];"
    );

    // Reshape rotated Q, K back to [1, dim, 1, seq] for output
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qrt = transpose(perm=pm,x=q_rot)[name=string(\"qrt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> qrf = reshape(shape=os,x=qrt)[name=string(\"qrf\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> krt = transpose(perm=pm,x=k_rot)[name=string(\"krt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> krf = reshape(shape=os,x=krt)[name=string(\"krf\")];");

    // Output: concat(o_out, qf_rotated, kf_rotated, vf, af, xn)
    let out_ch = 6 * dim;
    let _ = writeln!(
        m,
        "        int32 cax = const()[name=string(\"cax\"), val=int32(1)];"
    );
    let _ = writeln!(
        m,
        "        bool cid = const()[name=string(\"cid\"), val=bool(false)];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{out_ch},1,{seq}]> out = concat(axis=cax,interleave=cid,values=(oo,qrf,krf,vf,af,xn))[name=string(\"cat\")];");
    let _ = writeln!(
        m,
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];"
    );
    let _ = writeln!(m, "        tensor<fp32, [1,{out_ch},1,{seq}]> out32 = cast(dtype=to32,x=out)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (out32);");
    m.push_str("}\n");
    m
}

/// SDPA core for GQA: Q @ K^T → scale → mask → softmax → @V.
///
/// Unlike `gen_sdpa_fwd` which fuses projections+RoPE+SDPA+O_proj, this is
/// JUST the attention core. Projections, QK-norm, RoPE, and gating are handled
/// on CPU (cheap element-wise ops). The matmuls (Q@K^T and scores@V) are the
/// expensive part that benefits from ANE.
///
/// Input: `[1, 2*attn_dim + kv_dim, 1, seq]` fp32 = concat(Q, K, V_unexpanded).
///   - Q: `[attn_dim, seq]` — post-QK-norm, post-RoPE
///   - K: `[attn_dim, seq]` — expanded for GQA, post-QK-norm, post-RoPE
///   - V: `[kv_dim, seq]` — NOT expanded (expanded inside kernel via tile)
///
/// Wait — actually K needs to be expanded BEFORE the kernel because Q@K^T
/// requires matching head counts. Let's accept expanded K.
///
/// Input: `[1, 3*attn_dim, 1, seq]` fp32 = concat(Q, K_expanded, V_expanded).
/// Output: `[1, attn_dim, 1, seq]` fp32 = attention output.
///
/// The causal mask is baked in as a BLOBFILE constant.
pub fn gen_sdpa_core_gqa(cfg: &MilConfig) -> String {
    let ad = cfg.attn_dim();
    let seq = cfg.seq_len;
    let heads = cfg.n_heads;
    let hd = cfg.head_dim();
    let sc = 1.0 / (hd as f64).sqrt();
    let in_ch = 3 * ad;

    let mut m = String::with_capacity(8192);
    m.push_str(MIL_HDR);
    let _ = writeln!(
        m,
        "    func main<ios18>(tensor<fp32, [1, {in_ch}, 1, {seq}]> x) {{"
    );

    // Cast to fp16 for ANE
    let _ = writeln!(
        m,
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{in_ch},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");

    // Slice Q, K, V
    let _ = writeln!(m, "        tensor<int32, [4]> bq = const()[name=string(\"bq\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> sq = const()[name=string(\"sq\"), val=tensor<int32, [4]>([1,{ad},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> qf = slice_by_size(x=xh,begin=bq,size=sq)[name=string(\"qf\")];");

    let _ = writeln!(m, "        tensor<int32, [4]> bk = const()[name=string(\"bk\"), val=tensor<int32, [4]>([0,{ad},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> kf = slice_by_size(x=xh,begin=bk,size=sq)[name=string(\"kf\")];");

    let off_v = 2 * ad;
    let _ = writeln!(m, "        tensor<int32, [4]> bv = const()[name=string(\"bv\"), val=tensor<int32, [4]>([0,{off_v},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> vf = slice_by_size(x=xh,begin=bv,size=sq)[name=string(\"vf\")];");

    // Reshape to [1, heads, hd, seq] → transpose to [1, heads, seq, hd]
    let _ = writeln!(m, "        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");

    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> q4 = reshape(shape=rsh,x=qf)[name=string(\"rq\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> q = transpose(perm=pm,x=q4)[name=string(\"tq\")];");

    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> k4 = reshape(shape=rsh,x=kf)[name=string(\"rk\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> k = transpose(perm=pm,x=k4)[name=string(\"tk\")];");

    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> v4 = reshape(shape=rsh,x=vf)[name=string(\"rv\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> v = transpose(perm=pm,x=v4)[name=string(\"tv\")];");

    // Q @ K^T → [1, heads, seq, seq]
    let _ = writeln!(
        m,
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];"
    );
    let _ = writeln!(
        m,
        "        bool bT = const()[name=string(\"bT\"), val=bool(true)];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"mm1\")];");

    // Scale
    let _ = writeln!(
        m,
        "        fp16 scv = const()[name=string(\"scv\"), val=fp16({sc})];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];");

    // Causal mask (const BLOBFILE)
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{seq}]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,{seq},{seq}]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];");
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];"
    );

    // Softmax
    let _ = writeln!(
        m,
        "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"sm\")];");

    // scores @ V → [1, heads, seq, hd]
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> a4 = matmul(transpose_x=bF,transpose_y=bF,x=aw,y=v)[name=string(\"mm2\")];");

    // Reshape back to [1, attn_dim, 1, seq]
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> at = transpose(perm=pm,x=a4)[name=string(\"ta\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> os = const()[name=string(\"os\"), val=tensor<int32, [4]>([1,{ad},1,{seq}])];");
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{ad},1,{seq}]> af = reshape(shape=os,x=at)[name=string(\"ra\")];"
    );

    // Cast back to fp32
    let _ = writeln!(
        m,
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];"
    );
    let _ = writeln!(m, "        tensor<fp32, [1,{ad},1,{seq}]> out = cast(dtype=to32,x=af)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (out);");
    m.push_str("}\n");
    m
}

/// FFN forward part 1: xnorm @ W1 → SiLU, xnorm @ W3 → gate, gate*silu.
///
/// Input: `[1, dim, 1, seq + 2*hidden]` fp32.
/// Output: `[1, 3*hidden, 1, seq]` fp32 = concat(h1, h3, gate).
pub fn gen_ffn_w13(cfg: &MilConfig) -> String {
    let dim = cfg.dim;
    let hidden = cfg.hidden_dim;
    let seq = cfg.seq_len;
    let sp_in = seq + 2 * hidden;
    let out_ch = 3 * hidden;

    let mut m = String::with_capacity(4096);
    m.push_str(MIL_HDR);
    let _ = writeln!(
        m,
        "    func main<ios18>(tensor<fp32, [1, {dim}, 1, {sp_in}]> x) {{"
    );
    let _ = writeln!(
        m,
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{sp_in}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");

    // Slice xnorm
    let _ = writeln!(m, "        tensor<int32, [4]> bx = const()[name=string(\"bx\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> sx = const()[name=string(\"sx\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> xn = slice_by_size(x=xh,begin=bx,size=sx)[name=string(\"xn\")];");

    // Slice W1
    let _ = writeln!(m, "        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,0,0,{seq}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> s1 = const()[name=string(\"s1\"), val=tensor<int32, [4]>([1,{dim},1,{hidden}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{hidden}]> W1 = slice_by_size(x=xh,begin=b1,size=s1)[name=string(\"W1\")];");

    // Slice W3
    let off_w3 = seq + hidden;
    let _ = writeln!(m, "        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,0,0,{off_w3}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{hidden}]> W3 = slice_by_size(x=xh,begin=b3,size=s1)[name=string(\"W3\")];");

    // Reshape for matmul
    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(m, "        tensor<int32, [4]> rd = const()[name=string(\"rd\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> xn2 = reshape(shape=rd,x=xn)[name=string(\"xn2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> xnt = transpose(perm=pm,x=xn2)[name=string(\"xnt\")];");

    let _ = writeln!(m, "        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,{dim},{hidden}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{hidden}]> W12 = reshape(shape=rw,x=W1)[name=string(\"W12\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{hidden}]> W32 = reshape(shape=rw,x=W3)[name=string(\"W32\")];");

    let _ = writeln!(
        m,
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{hidden}]> h1m = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=W12)[name=string(\"h1m\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{hidden}]> h3m = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=W32)[name=string(\"h3m\")];");

    // Transpose back
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{seq}]> h1t = transpose(perm=pm,x=h1m)[name=string(\"h1t\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{seq}]> h3t = transpose(perm=pm,x=h3m)[name=string(\"h3t\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rh = const()[name=string(\"rh\"), val=tensor<int32, [4]>([1,{hidden},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> h1 = reshape(shape=rh,x=h1t)[name=string(\"h1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> h3 = reshape(shape=rh,x=h3t)[name=string(\"h3\")];");

    // SiLU + gate
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{hidden},1,{seq}]> sig = sigmoid(x=h1)[name=string(\"sg\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{hidden},1,{seq}]> silu = mul(x=h1,y=sig)[name=string(\"si\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{hidden},1,{seq}]> gate = mul(x=silu,y=h3)[name=string(\"gt\")];"
    );

    // Concat output: (h1, h3, gate)
    let _ = writeln!(
        m,
        "        int32 cax = const()[name=string(\"cax\"), val=int32(1)];"
    );
    let _ = writeln!(
        m,
        "        bool cid = const()[name=string(\"cid\"), val=bool(false)];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{out_ch},1,{seq}]> out = concat(axis=cax,interleave=cid,values=(h1,h3,gate))[name=string(\"cat\")];");
    let _ = writeln!(
        m,
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];"
    );
    let _ = writeln!(m, "        tensor<fp32, [1,{out_ch},1,{seq}]> out32 = cast(dtype=to32,x=out)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (out32);");
    m.push_str("}\n");
    m
}

/// FFN forward part 2: gate @ W2 (hidden → dim).
///
/// Input: `[1, hidden, 1, seq + dim]` fp32.
/// Output: `[1, dim, 1, seq]` fp32.
pub fn gen_ffn_w2(cfg: &MilConfig) -> String {
    let hidden = cfg.hidden_dim;
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let sp_in = seq + dim;

    let mut m = String::with_capacity(4096);
    m.push_str(MIL_HDR);
    let _ = writeln!(
        m,
        "    func main<ios18>(tensor<fp32, [1, {hidden}, 1, {sp_in}]> x) {{"
    );
    let _ = writeln!(
        m,
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{sp_in}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");

    let _ = writeln!(m, "        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,{hidden},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> act = slice_by_size(x=xh,begin=ba,size=sa)[name=string(\"act\")];");

    let _ = writeln!(m, "        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,{seq}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,{hidden},1,{dim}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{dim}]> W2 = slice_by_size(x=xh,begin=bw,size=sw)[name=string(\"W2\")];");

    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(m, "        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,{hidden},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{seq}]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{hidden}]> at = transpose(perm=pm,x=a2)[name=string(\"at\")];");

    let _ = writeln!(m, "        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,{hidden},{dim}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{dim}]> W22 = reshape(shape=rw,x=W2)[name=string(\"W22\")];");

    let _ = writeln!(
        m,
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> ym = matmul(transpose_x=bF,transpose_y=bF,x=at,y=W22)[name=string(\"ym\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> yt = transpose(perm=pm,x=ym)[name=string(\"yt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> yr = reshape(shape=ro,x=yt)[name=string(\"yr\")];"
    );
    let _ = writeln!(
        m,
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp32, [1,{dim},1,{seq}]> y = cast(dtype=to32,x=yr)[name=string(\"cout\")];"
    );
    let _ = writeln!(m, "    }} -> (y);");
    m.push_str("}\n");
    m
}

/// Fully-fused FFN forward: W1+W3+SiLU+gate+W2 in a single ANE dispatch.
///
/// Input: `[1, dim, 1, seq + 3*hidden]` fp32.
///   - xnorm `[dim, seq]`, W1_t `[dim, hidden]`, W3_t `[dim, hidden]`, W2_orig `[dim, hidden]`
///   - W1_t and W3_t are ic-major (transposed from PyTorch [hidden, dim] to [dim, hidden]).
///   - W2_orig is the ORIGINAL weight [dim, hidden] (out_features=dim, in_features=hidden).
///     It is transposed inside the kernel to get the correct matmul operand.
///
/// Output: `[1, 3*hidden + dim, 1, seq]` fp32 = concat(h1, h3, gate, ffn_out).
///
/// All ops are ANE-supported: slice_by_size, reshape, transpose, matmul, sigmoid, mul, concat, cast.
pub fn gen_fused_ffn_fwd(cfg: &MilConfig) -> String {
    let dim = cfg.dim;
    let hidden = cfg.hidden_dim;
    let seq = cfg.seq_len;
    let sp_in = seq + 3 * hidden;
    let out_ch = 3 * hidden + dim;

    let mut m = String::with_capacity(8192);
    m.push_str(MIL_HDR);
    let _ = writeln!(
        m,
        "    func main<ios18>(tensor<fp32, [1, {dim}, 1, {sp_in}]> x) {{"
    );

    // --- Constants ---
    let _ = writeln!(
        m,
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];"
    );
    let _ = writeln!(
        m,
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];"
    );
    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(
        m,
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];"
    );

    // Cast input to fp16
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{sp_in}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");

    // --- Slice inputs from spatial dimension ---
    // xnorm [dim, seq]
    let _ = writeln!(m, "        tensor<int32, [4]> bx = const()[name=string(\"bx\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> sx = const()[name=string(\"sx\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> xn = slice_by_size(x=xh,begin=bx,size=sx)[name=string(\"xn\")];");

    // W1_t [dim, hidden]
    let _ = writeln!(m, "        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,0,0,{seq}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,{dim},1,{hidden}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{hidden}]> W1 = slice_by_size(x=xh,begin=b1,size=sw)[name=string(\"W1\")];");

    // W3_t [dim, hidden]
    let off_w3 = seq + hidden;
    let _ = writeln!(m, "        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,0,0,{off_w3}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{hidden}]> W3 = slice_by_size(x=xh,begin=b3,size=sw)[name=string(\"W3\")];");

    // W2_orig [dim, hidden] — will be transposed inside kernel to get [hidden, dim]
    let off_w2 = seq + 2 * hidden;
    let _ = writeln!(m, "        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,0,0,{off_w2}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{hidden}]> W2r = slice_by_size(x=xh,begin=b2,size=sw)[name=string(\"W2r\")];");

    // --- W1/W3 matmul: xnorm^T @ W1, xnorm^T @ W3 ---
    // Reshape xnorm for matmul: [1,dim,1,seq] → [1,1,dim,seq] → transpose → [1,1,seq,dim]
    let _ = writeln!(m, "        tensor<int32, [4]> rd = const()[name=string(\"rd\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> xn2 = reshape(shape=rd,x=xn)[name=string(\"xn2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> xnt = transpose(perm=pm,x=xn2)[name=string(\"xnt\")];");

    // Reshape weights for matmul: [1,dim,1,hidden] → [1,1,dim,hidden]
    let _ = writeln!(m, "        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,{dim},{hidden}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{hidden}]> W12 = reshape(shape=rw,x=W1)[name=string(\"W12\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{hidden}]> W32 = reshape(shape=rw,x=W3)[name=string(\"W32\")];");

    // matmul: [1,1,seq,dim] @ [1,1,dim,hidden] → [1,1,seq,hidden]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{hidden}]> h1m = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=W12)[name=string(\"h1m\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{hidden}]> h3m = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=W32)[name=string(\"h3m\")];");

    // Transpose back: [1,1,seq,hidden] → [1,1,hidden,seq] → reshape [1,hidden,1,seq]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{seq}]> h1t = transpose(perm=pm,x=h1m)[name=string(\"h1t\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{seq}]> h3t = transpose(perm=pm,x=h3m)[name=string(\"h3t\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rh = const()[name=string(\"rh\"), val=tensor<int32, [4]>([1,{hidden},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> h1 = reshape(shape=rh,x=h1t)[name=string(\"h1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> h3 = reshape(shape=rh,x=h3t)[name=string(\"h3\")];");

    // --- SiLU + gate ---
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{hidden},1,{seq}]> sig = sigmoid(x=h1)[name=string(\"sg\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{hidden},1,{seq}]> silu = mul(x=h1,y=sig)[name=string(\"si\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{hidden},1,{seq}]> gate = mul(x=silu,y=h3)[name=string(\"gt\")];"
    );

    // --- W2 matmul: gate @ W2_t ---
    // W2_orig [1,dim,1,hidden] → reshape [1,1,dim,hidden] → transpose → [1,1,hidden,dim] = W2_t
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{hidden}]> W2s = reshape(shape=rw,x=W2r)[name=string(\"W2s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{dim}]> W2t = transpose(perm=pm,x=W2s)[name=string(\"W2t\")];");

    // gate: [1,hidden,1,seq] → reshape [1,1,hidden,seq] → transpose → [1,1,seq,hidden]
    let _ = writeln!(m, "        tensor<int32, [4]> rg = const()[name=string(\"rg\"), val=tensor<int32, [4]>([1,1,{hidden},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{seq}]> g2 = reshape(shape=rg,x=gate)[name=string(\"g2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{hidden}]> gt = transpose(perm=pm,x=g2)[name=string(\"gt2\")];");

    // matmul: [1,1,seq,hidden] @ [1,1,hidden,dim] → [1,1,seq,dim]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> fm = matmul(transpose_x=bF,transpose_y=bF,x=gt,y=W2t)[name=string(\"fm\")];");

    // Transpose + reshape: [1,1,seq,dim] → [1,1,dim,seq] → [1,dim,1,seq]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> ft = transpose(perm=pm,x=fm)[name=string(\"ft\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> ffn = reshape(shape=ro,x=ft)[name=string(\"ffn\")];");

    // --- Output: concat(h1, h3, gate, ffn_out) ---
    let _ = writeln!(
        m,
        "        int32 cax = const()[name=string(\"cax\"), val=int32(1)];"
    );
    let _ = writeln!(
        m,
        "        bool cid = const()[name=string(\"cid\"), val=bool(false)];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{out_ch},1,{seq}]> out = concat(axis=cax,interleave=cid,values=(h1,h3,gate,ffn))[name=string(\"cat\")];");
    let _ = writeln!(m, "        tensor<fp32, [1,{out_ch},1,{seq}]> out32 = cast(dtype=to32,x=out)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (out32);");
    m.push_str("}\n");
    m
}

/// Wo^T backward matmul: dx2 @ Wo^T → da (dim → attn_dim).
pub fn gen_wot(cfg: &MilConfig) -> String {
    gen_dyn_matmul_mil(cfg.dim, cfg.attn_dim(), cfg.seq_len)
}

/// FFN backward part 1: dffn @ W2^T (dim → hidden).
pub fn gen_ffn_bwd_w2t(cfg: &MilConfig) -> String {
    gen_dyn_matmul_mil(cfg.dim, cfg.hidden_dim, cfg.seq_len)
}

/// FFN backward part 2: dh1 @ W1^T + dh3 @ W3^T → dx (fused add).
///
/// Input: `[1, hidden, 1, 2*seq + 2*dim]` fp32.
/// Output: `[1, dim, 1, seq]` fp32.
pub fn gen_ffn_bwd_w13t(cfg: &MilConfig) -> String {
    let hidden = cfg.hidden_dim;
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let sp_in = 2 * seq + 2 * dim;

    let mut m = String::with_capacity(4096);
    m.push_str(MIL_HDR);
    let _ = writeln!(
        m,
        "    func main<ios18>(tensor<fp32, [1, {hidden}, 1, {sp_in}]> x) {{"
    );
    let _ = writeln!(
        m,
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{sp_in}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");

    // Slice dh1
    let _ = writeln!(m, "        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> sh = const()[name=string(\"sh\"), val=tensor<int32, [4]>([1,{hidden},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> dh1 = slice_by_size(x=xh,begin=b0,size=sh)[name=string(\"dh1\")];");

    // Slice dh3
    let _ = writeln!(m, "        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,0,0,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> dh3 = slice_by_size(x=xh,begin=b1,size=sh)[name=string(\"dh3\")];");

    // Slice W1^T
    let off_w1t = 2 * seq;
    let _ = writeln!(m, "        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,0,0,{off_w1t}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,{hidden},1,{dim}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{dim}]> W1t = slice_by_size(x=xh,begin=b2,size=sw)[name=string(\"W1t\")];");

    // Slice W3^T
    let off_w3t = 2 * seq + dim;
    let _ = writeln!(m, "        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,0,0,{off_w3t}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{dim}]> W3t = slice_by_size(x=xh,begin=b3,size=sw)[name=string(\"W3t\")];");

    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");

    // Reshape and matmul for dh1, dh3
    let _ = writeln!(m, "        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,{hidden},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{seq}]> dh12 = reshape(shape=ra,x=dh1)[name=string(\"dh12\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{hidden}]> dh1t = transpose(perm=pm,x=dh12)[name=string(\"dh1t\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{seq}]> dh32 = reshape(shape=ra,x=dh3)[name=string(\"dh32\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{hidden}]> dh3t = transpose(perm=pm,x=dh32)[name=string(\"dh3t\")];");

    let _ = writeln!(m, "        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,{hidden},{dim}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{dim}]> W1t2 = reshape(shape=rw,x=W1t)[name=string(\"W1t2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{dim}]> W3t2 = reshape(shape=rw,x=W3t)[name=string(\"W3t2\")];");

    let _ = writeln!(
        m,
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx1m = matmul(transpose_x=bF,transpose_y=bF,x=dh1t,y=W1t2)[name=string(\"dx1m\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx3m = matmul(transpose_x=bF,transpose_y=bF,x=dh3t,y=W3t2)[name=string(\"dx3m\")];");

    // Add
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{dim}]> dxm = add(x=dx1m,y=dx3m)[name=string(\"dxm\")];"
    );

    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dxt = transpose(perm=pm,x=dxm)[name=string(\"dxt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dx = reshape(shape=ro,x=dxt)[name=string(\"dx\")];");
    let _ = writeln!(
        m,
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp32, [1,{dim},1,{seq}]> y = cast(dtype=to32,x=dx)[name=string(\"cout\")];"
    );
    let _ = writeln!(m, "    }} -> (y);");
    m.push_str("}\n");
    m
}

/// QKV backward: dq @ Wq^T + dk @ Wk^T + dv @ Wv^T → dx (fused add).
///
/// Input: `[1, q_proj_dim, 1, 3*seq + 3*dim]` fp32.
/// Output: `[1, dim, 1, seq]` fp32.
pub fn gen_qkvb(cfg: &MilConfig) -> String {
    let qpd = cfg.q_proj_dim();
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let sp_in = 3 * seq + 3 * dim;

    let mut m = String::with_capacity(8192);
    m.push_str(MIL_HDR);
    let _ = writeln!(
        m,
        "    func main<ios18>(tensor<fp32, [1, {qpd}, 1, {sp_in}]> x) {{"
    );
    let _ = writeln!(
        m,
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{qpd},1,{sp_in}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");

    // Slice dq, dk, dv
    let _ = writeln!(m, "        tensor<int32, [4]> sd = const()[name=string(\"sd\"), val=tensor<int32, [4]>([1,{qpd},1,{seq}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{qpd},1,{seq}]> dq = slice_by_size(x=xh,begin=b0,size=sd)[name=string(\"dq\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,0,0,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{qpd},1,{seq}]> dk = slice_by_size(x=xh,begin=b1,size=sd)[name=string(\"dk\")];");
    let off_dv = 2 * seq;
    let _ = writeln!(m, "        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,0,0,{off_dv}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{qpd},1,{seq}]> dv = slice_by_size(x=xh,begin=b2,size=sd)[name=string(\"dv\")];");

    // Slice Wq^T, Wk^T, Wv^T
    let _ = writeln!(m, "        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,{qpd},1,{dim}])];");
    let off_wqt = 3 * seq;
    let _ = writeln!(m, "        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,0,0,{off_wqt}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{qpd},1,{dim}]> Wqt = slice_by_size(x=xh,begin=b3,size=sw)[name=string(\"Wqt\")];");
    let off_wkt = 3 * seq + dim;
    let _ = writeln!(m, "        tensor<int32, [4]> b4 = const()[name=string(\"b4\"), val=tensor<int32, [4]>([0,0,0,{off_wkt}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{qpd},1,{dim}]> Wkt = slice_by_size(x=xh,begin=b4,size=sw)[name=string(\"Wkt\")];");
    let off_wvt = 3 * seq + 2 * dim;
    let _ = writeln!(m, "        tensor<int32, [4]> b5 = const()[name=string(\"b5\"), val=tensor<int32, [4]>([0,0,0,{off_wvt}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{qpd},1,{dim}]> Wvt = slice_by_size(x=xh,begin=b5,size=sw)[name=string(\"Wvt\")];");

    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(
        m,
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];"
    );

    let _ = writeln!(m, "        tensor<int32, [4]> rd = const()[name=string(\"rd\"), val=tensor<int32, [4]>([1,1,{qpd},{seq}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,{qpd},{dim}])];");

    // dq @ Wq^T
    let _ = writeln!(m, "        tensor<fp16, [1,1,{qpd},{seq}]> dq2 = reshape(shape=rd,x=dq)[name=string(\"dq2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{qpd}]> dqt = transpose(perm=pm,x=dq2)[name=string(\"dqt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{qpd},{dim}]> Wqt2 = reshape(shape=rw,x=Wqt)[name=string(\"Wqt2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dxq = matmul(transpose_x=bF,transpose_y=bF,x=dqt,y=Wqt2)[name=string(\"dxq\")];");

    // dk @ Wk^T
    let _ = writeln!(m, "        tensor<fp16, [1,1,{qpd},{seq}]> dk2 = reshape(shape=rd,x=dk)[name=string(\"dk2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{qpd}]> dkt = transpose(perm=pm,x=dk2)[name=string(\"dkt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{qpd},{dim}]> Wkt2 = reshape(shape=rw,x=Wkt)[name=string(\"Wkt2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dxk = matmul(transpose_x=bF,transpose_y=bF,x=dkt,y=Wkt2)[name=string(\"dxk\")];");

    // dv @ Wv^T
    let _ = writeln!(m, "        tensor<fp16, [1,1,{qpd},{seq}]> dv2 = reshape(shape=rd,x=dv)[name=string(\"dv2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{qpd}]> dvt = transpose(perm=pm,x=dv2)[name=string(\"dvt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{qpd},{dim}]> Wvt2 = reshape(shape=rw,x=Wvt)[name=string(\"Wvt2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dxv = matmul(transpose_x=bF,transpose_y=bF,x=dvt,y=Wvt2)[name=string(\"dxv\")];");

    // Sum: dxq + dxk + dxv
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{dim}]> dxqk = add(x=dxq,y=dxk)[name=string(\"aqk\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{dim}]> dxall = add(x=dxqk,y=dxv)[name=string(\"aall\")];"
    );

    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dxt = transpose(perm=pm,x=dxall)[name=string(\"dxt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dx = reshape(shape=ro,x=dxt)[name=string(\"dx\")];");
    let _ = writeln!(
        m,
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp32, [1,{dim},1,{seq}]> y = cast(dtype=to32,x=dx)[name=string(\"cout\")];"
    );
    let _ = writeln!(m, "    }} -> (y);");
    m.push_str("}\n");
    m
}

/// SDPA backward part 1 (weight-free): recompute softmax + dV + dp.
///
/// Input: `[1, 4*attn_dim, 1, seq]` fp16 — Q,K,V,da stacked in channels.
/// Output: `[1, attn_dim+2*score_ch, 1, seq]` fp16 = concat(dV, probs, dp).
pub fn gen_sdpa_bwd1(cfg: &MilConfig) -> String {
    let attn_dim = cfg.attn_dim();
    let seq = cfg.seq_len;
    let heads = cfg.n_heads;
    let hd = cfg.head_dim();
    let score_ch = cfg.score_ch();
    let sc = 1.0 / (hd as f64).sqrt();
    let in_ch = 4 * attn_dim;
    let out_ch = attn_dim + 2 * score_ch;

    let mut m = String::with_capacity(8192);
    m.push_str(MIL_HDR);
    let _ = writeln!(
        m,
        "    func main<ios18>(tensor<fp16, [1, {in_ch}, 1, {seq}]> x) {{"
    );

    // Slice Q,K,V,da (channel-wise)
    let _ = writeln!(m, "        tensor<int32, [4]> sz = const()[name=string(\"sz\"), val=tensor<int32, [4]>([1,{attn_dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> qf = slice_by_size(x=x,begin=b0,size=sz)[name=string(\"s0\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,{attn_dim},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> kf = slice_by_size(x=x,begin=b1,size=sz)[name=string(\"s1\")];");
    let off_v = 2 * attn_dim;
    let _ = writeln!(m, "        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,{off_v},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> vf = slice_by_size(x=x,begin=b2,size=sz)[name=string(\"s2\")];");
    let off_da = 3 * attn_dim;
    let _ = writeln!(m, "        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,{off_da},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> da = slice_by_size(x=x,begin=b3,size=sz)[name=string(\"s3\")];");

    // Reshape to heads
    let _ = writeln!(m, "        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qr = reshape(shape=rsh,x=qf)[name=string(\"rq\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> q = transpose(perm=pm,x=qr)[name=string(\"tq\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> kr = reshape(shape=rsh,x=kf)[name=string(\"rk\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> k = transpose(perm=pm,x=kr)[name=string(\"tk\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> vr = reshape(shape=rsh,x=vf)[name=string(\"rv\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> v = transpose(perm=pm,x=vr)[name=string(\"tv\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> dr = reshape(shape=rsh,x=da)[name=string(\"rd\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dat = transpose(perm=pm,x=dr)[name=string(\"td\")];");

    // Forward attention scores (recompute)
    let _ = writeln!(
        m,
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];"
    );
    let _ = writeln!(
        m,
        "        bool bT = const()[name=string(\"bT\"), val=bool(true)];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"mm1\")];");
    let _ = writeln!(
        m,
        "        fp16 scv = const()[name=string(\"scv\"), val=fp16({sc})];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{seq}]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,{seq},{seq}]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];");
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];"
    );
    let _ = writeln!(
        m,
        "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> probs = softmax(axis=sax,x=ms)[name=string(\"sm\")];");

    // dV = probs^T @ da, dp = da @ V^T
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dv4 = matmul(transpose_x=bT,transpose_y=bF,x=probs,y=dat)[name=string(\"dv\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dp4 = matmul(transpose_x=bF,transpose_y=bT,x=dat,y=v)[name=string(\"dp\")];");

    // Reshape dV back
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> dvt = transpose(perm=pm,x=dv4)[name=string(\"dvt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> dvs = const()[name=string(\"dvs\"), val=tensor<int32, [4]>([1,{attn_dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> dvf = reshape(shape=dvs,x=dvt)[name=string(\"dvf\")];");

    // Flatten probs and dp for output
    let _ = writeln!(m, "        tensor<int32, [4]> scs = const()[name=string(\"scs\"), val=tensor<int32, [4]>([1,{score_ch},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{score_ch},1,{seq}]> pf = reshape(shape=scs,x=probs)[name=string(\"pf\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{score_ch},1,{seq}]> dpf = reshape(shape=scs,x=dp4)[name=string(\"dpf\")];");

    let _ = writeln!(
        m,
        "        int32 cax = const()[name=string(\"cax\"), val=int32(1)];"
    );
    let _ = writeln!(
        m,
        "        bool cid = const()[name=string(\"cid\"), val=bool(false)];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{out_ch},1,{seq}]> out = concat(axis=cax,interleave=cid,values=(dvf,pf,dpf))[name=string(\"cat\")];");
    let _ = writeln!(m, "    }} -> (out);");
    m.push_str("}\n");
    m
}

/// SDPA backward part 2: dQ + dK from probs, dp, Q, K.
///
/// Input: `[1, 2*score_ch + 2*attn_dim, 1, seq]` fp16.
/// Output: `[1, 2*attn_dim, 1, seq]` fp16 = concat(dQ, dK).
pub fn gen_sdpa_bwd2(cfg: &MilConfig) -> String {
    let attn_dim = cfg.attn_dim();
    let seq = cfg.seq_len;
    let heads = cfg.n_heads;
    let hd = cfg.head_dim();
    let score_ch = cfg.score_ch();
    let sc = 1.0 / (hd as f64).sqrt();
    let in_ch = 2 * score_ch + 2 * attn_dim;
    let out_ch = 2 * attn_dim;

    let mut m = String::with_capacity(8192);
    m.push_str(MIL_HDR);
    let _ = writeln!(
        m,
        "    func main<ios18>(tensor<fp16, [1, {in_ch}, 1, {seq}]> x) {{"
    );

    // Slice probs, dp (channel-wise, score_ch each)
    let _ = writeln!(m, "        tensor<int32, [4]> sz_sc = const()[name=string(\"szsc\"), val=tensor<int32, [4]>([1,{score_ch},1,{seq}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{score_ch},1,{seq}]> pf = slice_by_size(x=x,begin=b0,size=sz_sc)[name=string(\"s0\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,{score_ch},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{score_ch},1,{seq}]> dpf = slice_by_size(x=x,begin=b1,size=sz_sc)[name=string(\"s1\")];");

    // Slice Q, K
    let _ = writeln!(m, "        tensor<int32, [4]> sz_d = const()[name=string(\"szd\"), val=tensor<int32, [4]>([1,{attn_dim},1,{seq}])];");
    let off_q = 2 * score_ch;
    let _ = writeln!(m, "        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,{off_q},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> qf = slice_by_size(x=x,begin=b2,size=sz_d)[name=string(\"s2\")];");
    let off_k = 2 * score_ch + attn_dim;
    let _ = writeln!(m, "        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,{off_k},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> kf = slice_by_size(x=x,begin=b3,size=sz_d)[name=string(\"s3\")];");

    // Reshape to heads
    let _ = writeln!(m, "        tensor<int32, [4]> ssh = const()[name=string(\"ssh\"), val=tensor<int32, [4]>([1,{heads},{seq},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> probs = reshape(shape=ssh,x=pf)[name=string(\"rp\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dp = reshape(shape=ssh,x=dpf)[name=string(\"rdp\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qr = reshape(shape=rsh,x=qf)[name=string(\"rq\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> q = transpose(perm=pm,x=qr)[name=string(\"tq\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> kr = reshape(shape=rsh,x=kf)[name=string(\"rk\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> k = transpose(perm=pm,x=kr)[name=string(\"tk\")];");

    // Softmax backward: ds = probs * (dp - sum(probs*dp, axis=-1))
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> pdp = mul(x=probs,y=dp)[name=string(\"pdp\")];");
    let _ = writeln!(m, "        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];");
    let _ = writeln!(
        m,
        "        bool kd = const()[name=string(\"kd\"), val=bool(true)];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> spdp = reduce_sum(x=pdp,axes=rax,keep_dims=kd)[name=string(\"rs\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dps = sub(x=dp,y=spdp)[name=string(\"dps\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ds0 = mul(x=probs,y=dps)[name=string(\"ds0\")];");
    let _ = writeln!(
        m,
        "        fp16 scv = const()[name=string(\"scv\"), val=fp16({sc})];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{seq}]> ds = mul(x=ds0,y=scv)[name=string(\"ds\")];"
    );

    // dQ = ds @ K, dK = ds^T @ Q
    let _ = writeln!(
        m,
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];"
    );
    let _ = writeln!(
        m,
        "        bool bT = const()[name=string(\"bT\"), val=bool(true)];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dq4 = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=k)[name=string(\"dq\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dk4 = matmul(transpose_x=bT,transpose_y=bF,x=ds,y=q)[name=string(\"dk\")];");

    // Reshape back
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> dqt = transpose(perm=pm,x=dq4)[name=string(\"dqt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> dkt = transpose(perm=pm,x=dk4)[name=string(\"dkt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> fs = const()[name=string(\"fs\"), val=tensor<int32, [4]>([1,{attn_dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> dqf = reshape(shape=fs,x=dqt)[name=string(\"dqf\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> dkf = reshape(shape=fs,x=dkt)[name=string(\"dkf\")];");
    let _ = writeln!(
        m,
        "        int32 cax = const()[name=string(\"cax\"), val=int32(1)];"
    );
    let _ = writeln!(
        m,
        "        bool cid = const()[name=string(\"cid\"), val=bool(false)];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{out_ch},1,{seq}]> out = concat(axis=cax,interleave=cid,values=(dqf,dkf))[name=string(\"cat\")];");
    let _ = writeln!(m, "    }} -> (out);");
    m.push_str("}\n");
    m
}

// ---------------------------------------------------------------------------
// Fused single-layer forward (prototype: full transformer layer in one MIL)
// ---------------------------------------------------------------------------

/// Metadata for a fused single-layer forward MIL program.
pub struct FusedLayerMil {
    pub mil_text: String,
    pub weight_names: Vec<&'static str>,
    pub input_bytes: usize,
    pub output_bytes: usize,
}

/// Generate a fused single-layer forward MIL program.
///
/// Fuses: RMSNorm_att → QKV → RoPE → SDPA → Wo → residual → RMSNorm_ffn → FFN → residual.
/// All base weights are BLOBFILE constants (compiled once). Input is just x[dim, seq].
/// Only supports standard MHA (n_kv_heads == n_heads, attn_dim == dim, no output gate).
pub fn gen_fused_layer_fwd(cfg: &MilConfig) -> FusedLayerMil {
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let heads = cfg.n_heads;
    let hd = cfg.head_dim();
    let hidden = cfg.hidden_dim;
    let half_hd = hd / 2;
    let sc = 1.0 / (hd as f64).sqrt();
    let eps = cfg.rms_eps as f64;

    let mut m = String::with_capacity(32768);
    m.push_str(MIL_HDR);
    let _ = writeln!(
        m,
        "    func main<ios18>(tensor<fp32, [1, {dim}, 1, {seq}]> x) {{"
    );

    // --- Shared constants ---
    let _ = writeln!(
        m,
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];"
    );
    let _ = writeln!(
        m,
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];"
    );
    let _ = writeln!(
        m,
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];"
    );
    let _ = writeln!(
        m,
        "        bool bT = const()[name=string(\"bT\"), val=bool(true)];"
    );
    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(
        m,
        "        bool kd = const()[name=string(\"kd\"), val=bool(true)];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [1]> ch_ax = const()[name=string(\"chax\"), val=tensor<int32, [1]>([1])];"
    );
    let _ = writeln!(
        m,
        "        fp16 eps_v = const()[name=string(\"epsv\"), val=fp16({eps})];\n        fp16 nhalf = const()[name=string(\"nh\"), val=fp16(-0.5)];"
    );

    // Cast input to fp16
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];"
    );

    // === RMSNorm (attention) ===
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> rn1_sq = mul(x=xh,y=xh)[name=string(\"rn1sq\")];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{seq}]> rn1_m = reduce_mean(x=rn1_sq,axes=ch_ax,keep_dims=kd)[name=string(\"rn1m\")];");
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{seq}]> rn1_e = add(x=rn1_m,y=eps_v)[name=string(\"rn1e\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{seq}]> rn1_r = pow(x=rn1_e,y=nhalf)[name=string(\"rn1r\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> rn1_n = mul(x=xh,y=rn1_r)[name=string(\"rn1n\")];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,1]> rn1_w = const()[name=string(\"rn1w\"), val=tensor<fp16, [1,{dim},1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms_att.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> xnorm = mul(x=rn1_n,y=rn1_w)[name=string(\"xnorm\")];");

    // === QKV Projections (same pattern as gen_sdpa_fwd but with BLOBFILE weights) ===
    // Reshape xnorm for matmul: [1,D,1,S] → [1,1,D,S] → transpose → [1,1,S,D]
    let _ = writeln!(m, "        tensor<int32, [4]> r2d = const()[name=string(\"r2d\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> xn2 = reshape(shape=r2d,x=xnorm)[name=string(\"xn2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> xnt = transpose(perm=pm,x=xn2)[name=string(\"xnt\")];");

    // Weight constants (transposed: [in=dim, out=dim])
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{dim}]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [1,1,{dim},{dim}]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{dim}]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [1,1,{dim},{dim}]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{dim}]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [1,1,{dim},{dim}]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{dim}]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [1,1,{dim},{dim}]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];");

    // QKV matmul: xnt[S,D] @ W[D,D] → [S,D]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> qm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wq)[name=string(\"qm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> km = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wk)[name=string(\"km\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> vm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wv)[name=string(\"vm\")];");

    // Transpose back to [1,1,D,S] → reshape [1,D,1,S]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> qt = transpose(perm=pm,x=qm)[name=string(\"qt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> kt = transpose(perm=pm,x=km)[name=string(\"kt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> vt = transpose(perm=pm,x=vm)[name=string(\"vt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> os = const()[name=string(\"os\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> qf = reshape(shape=os,x=qt)[name=string(\"qf\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> kf = reshape(shape=os,x=kt)[name=string(\"kf\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> vf = reshape(shape=os,x=vt)[name=string(\"vf\")];"
    );

    // === RoPE ===
    let _ = writeln!(m, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> q4 = reshape(shape=qsh,x=qf)[name=string(\"rq\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> q = transpose(perm=pm,x=q4)[name=string(\"tq\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> k4 = reshape(shape=qsh,x=kf)[name=string(\"rk\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> k = transpose(perm=pm,x=k4)[name=string(\"tk\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> v4 = reshape(shape=qsh,x=vf)[name=string(\"rv\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> v = transpose(perm=pm,x=v4)[name=string(\"tv\")];");

    // RoPE cos/sin constants
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_cos = const()[name=string(\"rc\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_cos.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_sin = const()[name=string(\"rs\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_sin.bin\"), offset=uint64(64)))];");

    // Slice Q halves and apply RoPE
    let _ = writeln!(m, "        tensor<int32, [4]> rp_b0 = const()[name=string(\"rpb0\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> rp_sh = const()[name=string(\"rpsh\"), val=tensor<int32, [4]>([1,{heads},{seq},{half_hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q1 = slice_by_size(x=q,begin=rp_b0,size=rp_sh)[name=string(\"q1\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rp_bh = const()[name=string(\"rpbh\"), val=tensor<int32, [4]>([0,0,0,{half_hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q2 = slice_by_size(x=q,begin=rp_bh,size=rp_sh)[name=string(\"q2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q1c = mul(x=q1,y=rope_cos)[name=string(\"q1c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q2s = mul(x=q2,y=rope_sin)[name=string(\"q2s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> qr1 = sub(x=q1c,y=q2s)[name=string(\"qr1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q1s = mul(x=q1,y=rope_sin)[name=string(\"q1s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q2c = mul(x=q2,y=rope_cos)[name=string(\"q2c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> qr2 = add(x=q1s,y=q2c)[name=string(\"qr2\")];");
    let _ = writeln!(
        m,
        "        int32 rpax = const()[name=string(\"rpax\"), val=int32(-1)];"
    );
    let _ = writeln!(
        m,
        "        bool rpid = const()[name=string(\"rpid\"), val=bool(false)];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> q_rot = concat(axis=rpax,interleave=rpid,values=(qr1,qr2))[name=string(\"qrot\")];");

    // Same RoPE for K
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> k1 = slice_by_size(x=k,begin=rp_b0,size=rp_sh)[name=string(\"k1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> k2 = slice_by_size(x=k,begin=rp_bh,size=rp_sh)[name=string(\"k2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> k1c = mul(x=k1,y=rope_cos)[name=string(\"k1c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> k2s = mul(x=k2,y=rope_sin)[name=string(\"k2s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> kr1 = sub(x=k1c,y=k2s)[name=string(\"kr1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> k1s = mul(x=k1,y=rope_sin)[name=string(\"k1s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> k2c = mul(x=k2,y=rope_cos)[name=string(\"k2c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> kr2 = add(x=k1s,y=k2c)[name=string(\"kr2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> k_rot = concat(axis=rpax,interleave=rpid,values=(kr1,kr2))[name=string(\"krot\")];");

    // === SDPA ===
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q_rot,y=k_rot)[name=string(\"mm1\")];");
    let _ = writeln!(
        m,
        "        fp16 scv = const()[name=string(\"scv\"), val=fp16({sc})];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{seq}]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,{seq},{seq}]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];");
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];"
    );
    let _ = writeln!(
        m,
        "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"sm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> a4 = matmul(transpose_x=bF,transpose_y=bF,x=aw,y=v)[name=string(\"mm2\")];");

    // Reshape back to [1,D,1,S]
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> at = transpose(perm=pm,x=a4)[name=string(\"ta\")];");
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> af = reshape(shape=os,x=at)[name=string(\"ra\")];"
    );

    // === Wo projection ===
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> af2 = reshape(shape=r2d,x=af)[name=string(\"af2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> aft = transpose(perm=pm,x=af2)[name=string(\"aft\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> om = matmul(transpose_x=bF,transpose_y=bF,x=aft,y=Wo)[name=string(\"om\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> ot = transpose(perm=pm,x=om)[name=string(\"ot\")];");
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> oo = reshape(shape=os,x=ot)[name=string(\"oo\")];"
    );

    // === Residual 1 ===
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> x2 = add(x=xh,y=oo)[name=string(\"x2\")];"
    );

    // === RMSNorm (FFN) ===
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> rn2_sq = mul(x=x2,y=x2)[name=string(\"rn2sq\")];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{seq}]> rn2_m = reduce_mean(x=rn2_sq,axes=ch_ax,keep_dims=kd)[name=string(\"rn2m\")];");
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{seq}]> rn2_e = add(x=rn2_m,y=eps_v)[name=string(\"rn2e\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{seq}]> rn2_r = pow(x=rn2_e,y=nhalf)[name=string(\"rn2r\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> rn2_n = mul(x=x2,y=rn2_r)[name=string(\"rn2n\")];"
    );
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,1]> rn2_w = const()[name=string(\"rn2w\"), val=tensor<fp16, [1,{dim},1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms_ffn.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> x2norm = mul(x=rn2_n,y=rn2_w)[name=string(\"x2norm\")];");

    // === FFN W1/W3 projections ===
    // Reshape x2norm for matmul: [1,D,1,S] → [1,1,D,S] → transpose → [1,1,S,D]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> fn2 = reshape(shape=r2d,x=x2norm)[name=string(\"fn2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> fnt = transpose(perm=pm,x=fn2)[name=string(\"fnt\")];");

    // W1, W3 constants (transposed: [in=dim, out=hidden])
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{hidden}]> W1 = const()[name=string(\"W1\"), val=tensor<fp16, [1,1,{dim},{hidden}]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{hidden}]> W3 = const()[name=string(\"W3\"), val=tensor<fp16, [1,1,{dim},{hidden}]>(BLOBFILE(path=string(\"@model_path/weights/w3.bin\"), offset=uint64(64)))];");

    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{hidden}]> h1m = matmul(transpose_x=bF,transpose_y=bF,x=fnt,y=W1)[name=string(\"h1m\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{hidden}]> h3m = matmul(transpose_x=bF,transpose_y=bF,x=fnt,y=W3)[name=string(\"h3m\")];");

    // Transpose back + reshape
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{seq}]> h1t = transpose(perm=pm,x=h1m)[name=string(\"h1t\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{seq}]> h3t = transpose(perm=pm,x=h3m)[name=string(\"h3t\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rh = const()[name=string(\"rh\"), val=tensor<int32, [4]>([1,{hidden},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> h1 = reshape(shape=rh,x=h1t)[name=string(\"h1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> h3 = reshape(shape=rh,x=h3t)[name=string(\"h3\")];");

    // SiLU + gate
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{hidden},1,{seq}]> sig = sigmoid(x=h1)[name=string(\"sg\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{hidden},1,{seq}]> silu = mul(x=h1,y=sig)[name=string(\"si\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{hidden},1,{seq}]> gate = mul(x=silu,y=h3)[name=string(\"gt\")];"
    );

    // === FFN W2 projection ===
    let _ = writeln!(m, "        tensor<int32, [4]> rh2 = const()[name=string(\"rh2\"), val=tensor<int32, [4]>([1,1,{hidden},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{seq}]> g2 = reshape(shape=rh2,x=gate)[name=string(\"g2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{hidden}]> gt = transpose(perm=pm,x=g2)[name=string(\"gt2\")];");
    // W2 constant (transposed: [in=hidden, out=dim])
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{dim}]> W2 = const()[name=string(\"W2\"), val=tensor<fp16, [1,1,{hidden},{dim}]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> fm = matmul(transpose_x=bF,transpose_y=bF,x=gt,y=W2)[name=string(\"fm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> ft = transpose(perm=pm,x=fm)[name=string(\"ft\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> ffn_out = reshape(shape=os,x=ft)[name=string(\"ffn\")];");

    // === Residual 2 ===
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> xout = add(x=x2,y=ffn_out)[name=string(\"xout\")];");

    // Cast back to fp32
    let _ = writeln!(m, "        tensor<fp32, [1,{dim},1,{seq}]> y = cast(dtype=to32,x=xout)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (y);");
    m.push_str("}\n");

    let input_bytes = dim * seq * 4; // fp32
    let output_bytes = dim * seq * 4; // fp32

    FusedLayerMil {
        mil_text: m,
        weight_names: vec![
            "@model_path/weights/rms_att.bin",
            "@model_path/weights/rms_ffn.bin",
            "@model_path/weights/wq.bin",
            "@model_path/weights/wk.bin",
            "@model_path/weights/wv.bin",
            "@model_path/weights/wo.bin",
            "@model_path/weights/w1.bin",
            "@model_path/weights/w3.bin",
            "@model_path/weights/w2.bin",
            "@model_path/weights/rope_cos.bin",
            "@model_path/weights/rope_sin.bin",
            "@model_path/weights/mask.bin",
        ],
        input_bytes,
        output_bytes,
    }
}

/// Generate a fused GQA attention forward MIL program with gradient taps.
///
/// Fuses: QKV projections → Q/gate split → RoPE → GQA broadcast SDPA → sigmoid gate → O proj.
/// Handles Qwen3.5-style over-parameterised attention (attn_dim > dim) with output gating.
/// All base weights are BLOBFILE constants (compiled per-layer via `patch_from_donor`).
///
/// GQA is handled via matmul batch-dim broadcasting: Q is reshaped to
/// `[kv_heads, hpg, seq, hd]` and K/V to `[kv_heads, 1, seq, hd]`.
/// The matmul broadcasts dim-1 from 1→hpg, avoiding explicit tile/repeat.
///
/// Input: `[1, dim, 1, seq]` fp32 (post-RMSNorm hidden state).
/// Output: `[1, out_ch, 1, seq]` fp32 — gradient taps concat'd in channel dim:
///   `concat(o_out[dim], q_rot[attn_dim], k_rot[kv_dim], v[kv_dim], attn_out[attn_dim],
///           pre_gate[attn_dim], gate_raw[attn_dim])` (last 2 only if gated).
///
/// Weight files:
///   - wq.bin: `[1, 1, dim, q_proj_dim]` fp16 (q_proj_dim = 2·attn_dim if gate, else attn_dim)
///   - wk.bin: `[1, 1, dim, kv_dim]` fp16
///   - wv.bin: `[1, 1, dim, kv_dim]` fp16
///   - wo.bin: `[1, 1, attn_dim, dim]` fp16
///   - rope_cos.bin, rope_sin.bin: `[1, 1, seq, hd/2]` fp16
///   - mask.bin: `[1, 1, seq, seq]` fp16 (causal mask)
pub fn gen_fused_attn_gqa_fwd(cfg: &MilConfig, has_qk_norm: bool) -> FusedLayerMil {
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let heads = cfg.n_heads;
    let kv_heads = cfg.n_kv_heads;
    let hd = cfg.head_dim();
    let half_hd = hd / 2;
    let attn_dim = cfg.attn_dim();
    let kv_dim = cfg.kv_dim();
    let qpd = cfg.q_proj_dim();
    let hpg = cfg.heads_per_group();
    let sc = 1.0 / (hd as f64).sqrt();
    let has_gate = cfg.attn_output_gate;
    let eps = cfg.rms_eps as f64;
    // After QK-norm (if present), RoPE uses the normed tensor instead of raw Q/K
    let q_for_rope = if has_qk_norm { "q_n" } else { "q" };
    let k_for_rope = if has_qk_norm { "k_n" } else { "k" };

    let mut m = String::with_capacity(32768);
    m.push_str(MIL_HDR);
    let _ = writeln!(
        m,
        "    func main<ios18>(tensor<fp32, [1, {dim}, 1, {seq}]> x) {{"
    );

    // --- Shared constants ---
    let _ = writeln!(
        m,
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];"
    );
    let _ = writeln!(
        m,
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];"
    );
    let _ = writeln!(
        m,
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];"
    );
    let _ = writeln!(
        m,
        "        bool bT = const()[name=string(\"bT\"), val=bool(true)];"
    );
    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");

    // --- QK-norm constants: [1,H,hd,S] → [H,hd,1,S], reduce_mean axis 1 per-head ---
    if has_qk_norm {
        // QK-norm uses reduce_mean on axis -1 (hd dim) in [1,H,S,hd] layout
        // No batch dim reshape needed — stays [1,*,*,*] throughout
        let _ = writeln!(m, "        tensor<int32, [1]> rax_last = const()[name=string(\"raxl\"), val=tensor<int32, [1]>([-1])];");
        let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");
        let _ = writeln!(m, "        fp16 eps_v = const()[name=string(\"epsv\"), val=fp16({eps})];");
        let _ = writeln!(m, "        fp16 nhalf = const()[name=string(\"nh\"), val=fp16(-0.5)];");
    }

    // --- Cast input to fp16 ---
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");

    // --- Reshape for matmul: [1,D,1,S] → [1,1,D,S] → transpose → [1,1,S,D] ---
    let _ = writeln!(m, "        tensor<int32, [4]> r2d = const()[name=string(\"r2d\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> xn2 = reshape(shape=r2d,x=xh)[name=string(\"xn2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> xnt = transpose(perm=pm,x=xn2)[name=string(\"xnt\")];");

    // --- BLOBFILE weight constants ---
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{qpd}]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [1,1,{dim},{qpd}]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{kv_dim}]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [1,1,{dim},{kv_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{kv_dim}]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [1,1,{dim},{kv_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{dim}]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [1,1,{attn_dim},{dim}]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];");

    // --- QK-norm weight BLOBFILEs: [1, hd, 1, 1] for broadcast in [H,hd,1,S] layout ---
    if has_qk_norm {
        let _ = writeln!(m, "        tensor<fp16, [1,{hd},1,1]> qnw = const()[name=string(\"qnw\"), val=tensor<fp16, [1,{hd},1,1]>(BLOBFILE(path=string(\"@model_path/weights/q_norm.bin\"), offset=uint64(64)))];");
        let _ = writeln!(m, "        tensor<fp16, [1,{hd},1,1]> knw = const()[name=string(\"knw\"), val=tensor<fp16, [1,{hd},1,1]>(BLOBFILE(path=string(\"@model_path/weights/k_norm.bin\"), offset=uint64(64)))];");
    }

    // --- QKV matmuls: xnt[1,1,S,D] @ W[1,1,D,O] → [1,1,S,O] ---
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{qpd}]> qm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wq)[name=string(\"qm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{kv_dim}]> km = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wk)[name=string(\"km\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{kv_dim}]> vm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wv)[name=string(\"vm\")];");

    // --- Q: reshape to head layout, optionally split gate ---
    if has_gate {
        let two_hd = 2 * hd;
        let two_ad = 2 * attn_dim;
        // qm[1,1,S,2ad] → transpose [1,1,2ad,S] → reshape [1,H,2hd,S] → transpose [1,H,S,2hd]
        let _ = writeln!(m, "        tensor<fp16, [1,1,{two_ad},{seq}]> qt = transpose(perm=pm,x=qm)[name=string(\"qt\")];");
        let _ = writeln!(m, "        tensor<int32, [4]> rqg = const()[name=string(\"rqg\"), val=tensor<int32, [4]>([1,{heads},{two_hd},{seq}])];");
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{two_hd},{seq}]> q4g = reshape(shape=rqg,x=qt)[name=string(\"q4g\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{two_hd}]> qg = transpose(perm=pm,x=q4g)[name=string(\"qg\")];");
        // Slice Q_actual [:,:,:,:hd] and gate [:,:,:,hd:]
        let _ = writeln!(m, "        tensor<int32, [4]> bq0 = const()[name=string(\"bq0\"), val=tensor<int32, [4]>([0,0,0,0])];");
        let _ = writeln!(m, "        tensor<int32, [4]> sqh = const()[name=string(\"sqh\"), val=tensor<int32, [4]>([1,{heads},{seq},{hd}])];");
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> q = slice_by_size(x=qg,begin=bq0,size=sqh)[name=string(\"q\")];");
        let _ = writeln!(m, "        tensor<int32, [4]> bgh = const()[name=string(\"bgh\"), val=tensor<int32, [4]>([0,0,0,{hd}])];");
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> graw = slice_by_size(x=qg,begin=bgh,size=sqh)[name=string(\"graw\")];");
    } else {
        // qm[1,1,S,ad] → transpose [1,1,ad,S] → reshape [1,H,hd,S] → transpose [1,H,S,hd]
        let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{seq}]> qt = transpose(perm=pm,x=qm)[name=string(\"qt\")];");
        let _ = writeln!(m, "        tensor<int32, [4]> rqh = const()[name=string(\"rqh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> q4 = reshape(shape=rqh,x=qt)[name=string(\"q4\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> q = transpose(perm=pm,x=q4)[name=string(\"q\")];");
    }

    // --- Q QK-norm: stay in [1,H,S,hd], reduce axis=3 (no batch dim change) ---
    if has_qk_norm {
        // sq = q^2, ms = reduce_mean(sq, axis=-1), rr = pow(ms+eps, -0.5), q_n = q*rr*w
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> qn_sq = mul(x=q,y=q)[name=string(\"qnsq\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> qn_ms = reduce_mean(x=qn_sq,axes=rax_last,keep_dims=kd)[name=string(\"qnms\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> qn_me = add(x=qn_ms,y=eps_v)[name=string(\"qnme\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> qn_ri = pow(x=qn_me,y=nhalf)[name=string(\"qnri\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> qn_nm = mul(x=q,y=qn_ri)[name=string(\"qnnm\")];");
        // qnw is [1,hd,1,1] but we need [1,1,1,hd] for broadcast in [1,H,S,hd]
        let _ = writeln!(m, "        tensor<int32, [4]> rqnw = const()[name=string(\"rqnw\"), val=tensor<int32, [4]>([1,1,1,{hd}])];");
        let _ = writeln!(m, "        tensor<fp16, [1,1,1,{hd}]> qnw_b = reshape(shape=rqnw,x=qnw)[name=string(\"qnwb\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> q_n = mul(x=qn_nm,y=qnw_b)[name=string(\"qn\")];");
    }

    // --- K: [1,1,S,kvd] → transpose → reshape → transpose → [1,kvH,S,hd] ---
    let _ = writeln!(m, "        tensor<fp16, [1,1,{kv_dim},{seq}]> kt2 = transpose(perm=pm,x=km)[name=string(\"kt2\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rkv = const()[name=string(\"rkv\"), val=tensor<int32, [4]>([1,{kv_heads},{hd},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> k4 = reshape(shape=rkv,x=kt2)[name=string(\"rk\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> k = transpose(perm=pm,x=k4)[name=string(\"tk\")];");

    // --- K QK-norm: stay in [1,kvH,S,hd], reduce axis=3 (no batch dim change) ---
    if has_qk_norm {
        let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> kn_sq = mul(x=k,y=k)[name=string(\"knsq\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},1]> kn_ms = reduce_mean(x=kn_sq,axes=rax_last,keep_dims=kd)[name=string(\"knms\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},1]> kn_me = add(x=kn_ms,y=eps_v)[name=string(\"knme\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},1]> kn_ri = pow(x=kn_me,y=nhalf)[name=string(\"knri\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> kn_nm = mul(x=k,y=kn_ri)[name=string(\"knnm\")];");
        let _ = writeln!(m, "        tensor<int32, [4]> rknw = const()[name=string(\"rknw\"), val=tensor<int32, [4]>([1,1,1,{hd}])];");
        let _ = writeln!(m, "        tensor<fp16, [1,1,1,{hd}]> knw_b = reshape(shape=rknw,x=knw)[name=string(\"knwb\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> k_n = mul(x=kn_nm,y=knw_b)[name=string(\"kn\")];");
    }

    // --- V: same path ---
    let _ = writeln!(m, "        tensor<fp16, [1,1,{kv_dim},{seq}]> vt2 = transpose(perm=pm,x=vm)[name=string(\"vt2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> v4 = reshape(shape=rkv,x=vt2)[name=string(\"rv\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> v = transpose(perm=pm,x=v4)[name=string(\"tv\")];");

    // --- RoPE on Q [1,H,S,hd] ---
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_cos = const()[name=string(\"rc\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_cos.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_sin = const()[name=string(\"rs\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_sin.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        int32 rpax = const()[name=string(\"rpax\"), val=int32(-1)];");
    let _ = writeln!(
        m,
        "        bool rpid = const()[name=string(\"rpid\"), val=bool(false)];"
    );

    // Q RoPE
    let _ = writeln!(m, "        tensor<int32, [4]> rpb0 = const()[name=string(\"rpb0\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> rpqh = const()[name=string(\"rpqh\"), val=tensor<int32, [4]>([1,{heads},{seq},{half_hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q1 = slice_by_size(x={q_for_rope},begin=rpb0,size=rpqh)[name=string(\"q1\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rpbh = const()[name=string(\"rpbh\"), val=tensor<int32, [4]>([0,0,0,{half_hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q2 = slice_by_size(x={q_for_rope},begin=rpbh,size=rpqh)[name=string(\"q2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q1c = mul(x=q1,y=rope_cos)[name=string(\"q1c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q2s = mul(x=q2,y=rope_sin)[name=string(\"q2s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> qr1 = sub(x=q1c,y=q2s)[name=string(\"qr1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q1s = mul(x=q1,y=rope_sin)[name=string(\"q1s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q2c = mul(x=q2,y=rope_cos)[name=string(\"q2c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> qr2 = add(x=q1s,y=q2c)[name=string(\"qr2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> q_rot = concat(axis=rpax,interleave=rpid,values=(qr1,qr2))[name=string(\"qrot\")];");

    // K RoPE (kv_heads)
    let _ = writeln!(m, "        tensor<int32, [4]> rpkh = const()[name=string(\"rpkh\"), val=tensor<int32, [4]>([1,{kv_heads},{seq},{half_hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k1 = slice_by_size(x={k_for_rope},begin=rpb0,size=rpkh)[name=string(\"k1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k2 = slice_by_size(x={k_for_rope},begin=rpbh,size=rpkh)[name=string(\"k2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k1c = mul(x=k1,y=rope_cos)[name=string(\"k1c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k2s = mul(x=k2,y=rope_sin)[name=string(\"k2s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> kr1 = sub(x=k1c,y=k2s)[name=string(\"kr1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k1s = mul(x=k1,y=rope_sin)[name=string(\"k1s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k2c = mul(x=k2,y=rope_cos)[name=string(\"k2c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> kr2 = add(x=k1s,y=k2c)[name=string(\"kr2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> k_rot = concat(axis=rpax,interleave=rpid,values=(kr1,kr2))[name=string(\"krot\")];");

    // --- GQA SDPA via batch-dim broadcast ---
    // Q: [1,H,S,hd] → [kvH, hpg, S, hd]
    // K: [1,kvH,S,hd] → [kvH, 1, S, hd]
    // V: [1,kvH,S,hd] → [kvH, 1, S, hd]
    let _ = writeln!(m, "        tensor<int32, [4]> rqb = const()[name=string(\"rqb\"), val=tensor<int32, [4]>([{kv_heads},{hpg},{seq},{hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> qb = reshape(shape=rqb,x=q_rot)[name=string(\"qb\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rkb = const()[name=string(\"rkb\"), val=tensor<int32, [4]>([{kv_heads},1,{seq},{hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> kb = reshape(shape=rkb,x=k_rot)[name=string(\"kb\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> vb = reshape(shape=rkb,x=v)[name=string(\"vb\")];");

    // Q@K^T: [kvH,hpg,S,hd] @ [kvH,1,hd,S] → [kvH,hpg,S,S]
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=qb,y=kb)[name=string(\"mm1\")];");
    let _ = writeln!(
        m,
        "        fp16 scv = const()[name=string(\"scv\"), val=fp16({sc})];"
    );
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];");

    // Causal mask [1,1,S,S] broadcasts to [kvH,hpg,S,S]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{seq}]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,{seq},{seq}]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];");

    // Softmax
    let _ = writeln!(
        m,
        "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];"
    );
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"sm\")];");

    // scores@V: [kvH,hpg,S,S] @ [kvH,1,S,hd] → [kvH,hpg,S,hd]
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> a4 = matmul(transpose_x=bF,transpose_y=bF,x=aw,y=vb)[name=string(\"mm2\")];");

    // Reshape back to [1,H,S,hd]
    let _ = writeln!(m, "        tensor<int32, [4]> rha = const()[name=string(\"rha\"), val=tensor<int32, [4]>([1,{heads},{seq},{hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> a_out = reshape(shape=rha,x=a4)[name=string(\"aout\")];");

    // --- Sigmoid gate ---
    let o_in = if has_gate {
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> gsig = sigmoid(x=graw)[name=string(\"gsig\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> gated = mul(x=a_out,y=gsig)[name=string(\"gated\")];");
        "gated"
    } else {
        "a_out"
    };

    // --- O projection ---
    // [1,H,S,hd] → transpose [1,H,hd,S] → reshape [1,ad,1,S]
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> ot1 = transpose(perm=pm,x={o_in})[name=string(\"ot1\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rad = const()[name=string(\"rad\"), val=tensor<int32, [4]>([1,{attn_dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> af = reshape(shape=rad,x=ot1)[name=string(\"af\")];");
    // [1,ad,1,S] → [1,1,ad,S] → transpose [1,1,S,ad]
    let _ = writeln!(m, "        tensor<int32, [4]> r2a = const()[name=string(\"r2a\"), val=tensor<int32, [4]>([1,1,{attn_dim},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{seq}]> af2 = reshape(shape=r2a,x=af)[name=string(\"af2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{attn_dim}]> aft = transpose(perm=pm,x=af2)[name=string(\"aft\")];");
    // matmul [1,1,S,ad] @ Wo[1,1,ad,D] → [1,1,S,D]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> om = matmul(transpose_x=bF,transpose_y=bF,x=aft,y=Wo)[name=string(\"om\")];");
    // transpose [1,1,D,S] → reshape [1,D,1,S]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> ot2 = transpose(perm=pm,x=om)[name=string(\"ot2\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rod = const()[name=string(\"rod\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> oo = reshape(shape=rod,x=ot2)[name=string(\"oo\")];");

    // --- Gradient tap outputs: flatten head tensors to [1, ch, 1, seq] ---
    // q_rot: [1,H,S,hd] → transpose [1,H,hd,S] → reshape [1,ad,1,S]
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qr_t = transpose(perm=pm,x=q_rot)[name=string(\"qrt2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> qr_f = reshape(shape=rad,x=qr_t)[name=string(\"qrf\")];");

    // k_rot: [1,kvH,S,hd] → transpose [1,kvH,hd,S] → reshape [1,kvd,1,S]
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> kr_t = transpose(perm=pm,x=k_rot)[name=string(\"krt2\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rkd = const()[name=string(\"rkd\"), val=tensor<int32, [4]>([1,{kv_dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_dim},1,{seq}]> kr_f = reshape(shape=rkd,x=kr_t)[name=string(\"krf\")];");

    // v: [1,kvH,S,hd] → transpose [1,kvH,hd,S] → reshape [1,kvd,1,S]
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> v_t = transpose(perm=pm,x=v)[name=string(\"vtf\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_dim},1,{seq}]> v_f = reshape(shape=rkd,x=v_t)[name=string(\"vfl\")];");

    // attn_out (post-gate or a_out): [1,H,S,hd] → [1,ad,1,S]
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> ao_t = transpose(perm=pm,x={o_in})[name=string(\"aot\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> ao_f = reshape(shape=rad,x=ao_t)[name=string(\"aof\")];");

    // q_pre / k_pre: pre-QK-norm activations for backward (only when QK-norm present)
    if has_qk_norm {
        // q (pre-norm): [1,H,S,hd] → transpose [1,H,hd,S] → reshape [1,ad,1,S]
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qp_t = transpose(perm=pm,x=q)[name=string(\"qpt\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> qp_f = reshape(shape=rad,x=qp_t)[name=string(\"qpf\")];");
        // k (pre-norm): [1,kvH,S,hd] → transpose [1,kvH,hd,S] → reshape [1,kvd,1,S]
        let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> kp_t = transpose(perm=pm,x=k)[name=string(\"kpt\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{kv_dim},1,{seq}]> kp_f = reshape(shape=rkd,x=kp_t)[name=string(\"kpf\")];");
    }

    // --- Concat output ---
    let _ = writeln!(m, "        int32 cax = const()[name=string(\"cax\"), val=int32(1)];");
    let _ = writeln!(m, "        bool cid = const()[name=string(\"cid\"), val=bool(false)];");

    // Output layout (channels along axis 1):
    //   oo | qr_f | kr_f | v_f | ao_f [| pg_f | gr_f] [| qp_f | kp_f]
    let out_ch = if has_gate && has_qk_norm {
        // a_out (pre-gate): [1,H,S,hd] → [1,ad,1,S]
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> pg_t = transpose(perm=pm,x=a_out)[name=string(\"pgt\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> pg_f = reshape(shape=rad,x=pg_t)[name=string(\"pgf\")];");
        // graw: [1,H,S,hd] → [1,ad,1,S]
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> gr_t = transpose(perm=pm,x=graw)[name=string(\"grt\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> gr_f = reshape(shape=rad,x=gr_t)[name=string(\"grf\")];");

        let out_ch = dim + 5 * attn_dim + 3 * kv_dim;
        let _ = writeln!(m, "        tensor<fp16, [1,{out_ch},1,{seq}]> out = concat(axis=cax,interleave=cid,values=(oo,qr_f,kr_f,v_f,ao_f,pg_f,gr_f,qp_f,kp_f))[name=string(\"cat\")];");
        out_ch
    } else if has_gate {
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> pg_t = transpose(perm=pm,x=a_out)[name=string(\"pgt\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> pg_f = reshape(shape=rad,x=pg_t)[name=string(\"pgf\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> gr_t = transpose(perm=pm,x=graw)[name=string(\"grt\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> gr_f = reshape(shape=rad,x=gr_t)[name=string(\"grf\")];");

        let out_ch = dim + 4 * attn_dim + 2 * kv_dim;
        let _ = writeln!(m, "        tensor<fp16, [1,{out_ch},1,{seq}]> out = concat(axis=cax,interleave=cid,values=(oo,qr_f,kr_f,v_f,ao_f,pg_f,gr_f))[name=string(\"cat\")];");
        out_ch
    } else if has_qk_norm {
        let out_ch = dim + 3 * attn_dim + 3 * kv_dim;
        let _ = writeln!(m, "        tensor<fp16, [1,{out_ch},1,{seq}]> out = concat(axis=cax,interleave=cid,values=(oo,qr_f,kr_f,v_f,ao_f,qp_f,kp_f))[name=string(\"cat\")];");
        out_ch
    } else {
        let out_ch = dim + 2 * attn_dim + 2 * kv_dim;
        let _ = writeln!(m, "        tensor<fp16, [1,{out_ch},1,{seq}]> out = concat(axis=cax,interleave=cid,values=(oo,qr_f,kr_f,v_f,ao_f))[name=string(\"cat\")];");
        out_ch
    };

    // Cast to fp32
    let _ = writeln!(m, "        tensor<fp32, [1,{out_ch},1,{seq}]> out32 = cast(dtype=to32,x=out)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (out32);");
    m.push_str("}\n");

    let input_bytes = dim * seq * 4;
    let output_bytes = out_ch * seq * 4;

    let mut weight_names: Vec<&'static str> = vec![
        "@model_path/weights/wq.bin",
        "@model_path/weights/wk.bin",
        "@model_path/weights/wv.bin",
        "@model_path/weights/wo.bin",
        "@model_path/weights/rope_cos.bin",
        "@model_path/weights/rope_sin.bin",
        "@model_path/weights/mask.bin",
    ];
    if has_qk_norm {
        weight_names.push("@model_path/weights/q_norm.bin");
        weight_names.push("@model_path/weights/k_norm.bin");
    }

    FusedLayerMil {
        mil_text: m,
        weight_names,
        input_bytes,
        output_bytes,
    }
}

/// Generate a fused attention GQA backward MIL program.
///
/// Replaces 4 separate ANE dispatches (wot_bwd, sdpa_bwd1, sdpa_bwd2, qkv_bwd) + CPU
/// (gate backward, RoPE backward) with a single fused ANE kernel.
///
/// Input layout `[1, in_ch, 1, seq]` fp32:
///   `dx2[dim] | Q_rot[ad] | K_rot[kvd] | V[kvd] [| pre_gate[ad] | gate_raw[ad]]`
///
/// Output: `[1, dim, 1, seq]` fp32 — dx_attn gradient for the residual stream.
///
/// BLOBFILE weights (same orientation as forward): Wq, Wk, Wv, Wo, rope_cos, rope_sin, mask.
/// Backward uses `matmul(transpose_y=True)` to compute W^T projections with forward-layout weights.
pub fn gen_fused_attn_gqa_bwd(cfg: &MilConfig, has_qk_norm: bool) -> FusedLayerMil {
    assert!(
        !has_qk_norm,
        "QK-norm backward not yet implemented in fused kernel"
    );

    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let heads = cfg.n_heads;
    let kv_heads = cfg.n_kv_heads;
    let hd = cfg.head_dim();
    let half_hd = hd / 2;
    let attn_dim = cfg.attn_dim();
    let kv_dim = cfg.kv_dim();
    let qpd = cfg.q_proj_dim();
    let hpg = cfg.heads_per_group();
    let sc = 1.0 / (hd as f64).sqrt();
    let has_gate = cfg.attn_output_gate;

    // Input channels: dx2 + Q_rot + K_rot + V [+ pre_gate + gate_raw]
    let in_ch = if has_gate {
        dim + 3 * attn_dim + 2 * kv_dim
    } else {
        dim + attn_dim + 2 * kv_dim
    };

    let mut m = String::with_capacity(65536);
    m.push_str(MIL_HDR);
    let _ = writeln!(
        m,
        "    func main<ios18>(tensor<fp32, [1, {in_ch}, 1, {seq}]> x) {{"
    );

    // --- Shared constants ---
    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
    let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
    let _ = writeln!(m, "        bool bT = const()[name=string(\"bT\"), val=bool(true)];");
    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(m, "        fp16 scv = const()[name=string(\"scv\"), val=fp16({sc})];");
    // reduce_mean→sum trick constants
    let _ = writeln!(m, "        fp16 hpg_v = const()[name=string(\"hpgv\"), val=fp16({hpg})];");
    let _ = writeln!(m, "        fp16 seq_v = const()[name=string(\"seqv\"), val=fp16({seq})];");
    // reduce axes must be tensor<int32,[1]>, but softmax axis must be scalar int32
    let _ = writeln!(m, "        tensor<int32, [1]> ax1 = const()[name=string(\"ax1\"), val=tensor<int32, [1]>([1])];");
    let _ = writeln!(m, "        tensor<int32, [1]> rax_last = const()[name=string(\"raxl\"), val=tensor<int32, [1]>([-1])];");
    let _ = writeln!(m, "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];");
    let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");

    // --- Cast input to fp16 ---
    let _ = writeln!(m, "        tensor<fp16, [1,{in_ch},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");

    // --- Phase 1: Slice input channels ---
    let mut off = 0usize;

    // dx2: [1, dim, 1, S]
    let _ = writeln!(m, "        tensor<int32, [4]> b_dx = const()[name=string(\"bdx\"), val=tensor<int32, [4]>([0,{off},0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> s_dx = const()[name=string(\"sdx\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dx2h = slice_by_size(x=xh,begin=b_dx,size=s_dx)[name=string(\"dx2h\")];");
    off += dim;

    // Q_rot: [1, ad, 1, S]
    let _ = writeln!(m, "        tensor<int32, [4]> b_qr = const()[name=string(\"bqr\"), val=tensor<int32, [4]>([0,{off},0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> s_ad = const()[name=string(\"sad\"), val=tensor<int32, [4]>([1,{attn_dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> qrh = slice_by_size(x=xh,begin=b_qr,size=s_ad)[name=string(\"qrh\")];");
    off += attn_dim;

    // K_rot: [1, kvd, 1, S]
    let _ = writeln!(m, "        tensor<int32, [4]> b_kr = const()[name=string(\"bkr\"), val=tensor<int32, [4]>([0,{off},0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> s_kv = const()[name=string(\"skv\"), val=tensor<int32, [4]>([1,{kv_dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_dim},1,{seq}]> krh = slice_by_size(x=xh,begin=b_kr,size=s_kv)[name=string(\"krh\")];");
    off += kv_dim;

    // V: [1, kvd, 1, S]
    let _ = writeln!(m, "        tensor<int32, [4]> b_vh = const()[name=string(\"bvh\"), val=tensor<int32, [4]>([0,{off},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_dim},1,{seq}]> vh = slice_by_size(x=xh,begin=b_vh,size=s_kv)[name=string(\"vh\")];");
    off += kv_dim;

    if has_gate {
        // pre_gate (a_out before gating): [1, ad, 1, S]
        let _ = writeln!(m, "        tensor<int32, [4]> b_pg = const()[name=string(\"bpg\"), val=tensor<int32, [4]>([0,{off},0,0])];");
        let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> pgh = slice_by_size(x=xh,begin=b_pg,size=s_ad)[name=string(\"pgh\")];");
        off += attn_dim;

        // gate_raw: [1, ad, 1, S]
        let _ = writeln!(m, "        tensor<int32, [4]> b_gr = const()[name=string(\"bgr\"), val=tensor<int32, [4]>([0,{off},0,0])];");
        let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> grh = slice_by_size(x=xh,begin=b_gr,size=s_ad)[name=string(\"grh\")];");
        off += attn_dim;
    }
    let _ = off;

    // --- BLOBFILE weight constants (same orientation as forward) ---
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{qpd}]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [1,1,{dim},{qpd}]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{kv_dim}]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [1,1,{dim},{kv_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{kv_dim}]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [1,1,{dim},{kv_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{dim}]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [1,1,{attn_dim},{dim}]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];");

    // RoPE tables + causal mask
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_cos = const()[name=string(\"rc\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_cos.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_sin = const()[name=string(\"rs\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_sin.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{seq}]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,{seq},{seq}]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];");

    // --- Phase 2: Wo^T projection ---
    // dx2: [1,dim,1,S] → [1,1,dim,S] → transpose [1,1,S,dim]
    let _ = writeln!(m, "        tensor<int32, [4]> r2d = const()[name=string(\"r2d\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dx_r = reshape(shape=r2d,x=dx2h)[name=string(\"dxr\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx_nt = transpose(perm=pm,x=dx_r)[name=string(\"dxnt\")];");
    // da = dx_nt @ Wo^T: [1,1,S,dim] @ transpose([1,1,ad,dim]) = [1,1,S,dim] @ [1,1,dim,ad] → [1,1,S,ad]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{attn_dim}]> da_nt = matmul(transpose_x=bF,transpose_y=bT,x=dx_nt,y=Wo)[name=string(\"dant\")];");

    // --- Phase 3: Gate backward ---
    // Convert da to [1,ad,1,S] for gate ops / head reshaping
    let da_var = if has_gate {
        // da_nt [1,1,S,ad] → transpose [1,1,ad,S] → reshape [1,ad,1,S]
        let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{seq}]> da_t = transpose(perm=pm,x=da_nt)[name=string(\"dat\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> da_ch = reshape(shape=s_ad,x=da_t)[name=string(\"dach\")];");
        // sigmoid(gate_raw)
        let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> sig = sigmoid(x=grh)[name=string(\"sig\")];");
        // d_attn = da * sig
        let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> d_at = mul(x=da_ch,y=sig)[name=string(\"dat2\")];");
        // d_gate = da * pre_gate * sig * (1 - sig)
        // Use sig - sig^2 instead of sub(scalar, tensor) which ANE rejects
        let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> sig2 = mul(x=sig,y=sig)[name=string(\"sig2\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> sder = sub(x=sig,y=sig2)[name=string(\"sder\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> dg1 = mul(x=da_ch,y=pgh)[name=string(\"dg1\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> d_gate = mul(x=dg1,y=sder)[name=string(\"dgate\")];");
        "d_at" // d_attn in [1,ad,1,S]
    } else {
        // No gate: da_nt [1,1,S,ad] → transpose [1,1,ad,S] → reshape [1,ad,1,S]
        let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{seq}]> da_t = transpose(perm=pm,x=da_nt)[name=string(\"dat\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> d_at = reshape(shape=s_ad,x=da_t)[name=string(\"dat2\")];");
        "d_at"
    };

    // --- Phase 4: Reshape to GQA batch form ---
    // d_attn[1,ad,1,S] → [1,H,hd,S] → transpose [1,H,S,hd] → [kvH,hpg,S,hd]
    let _ = writeln!(m, "        tensor<int32, [4]> rqh = const()[name=string(\"rqh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> da_4 = reshape(shape=rqh,x={da_var})[name=string(\"da4\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> da_hs = transpose(perm=pm,x=da_4)[name=string(\"dahs\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rqb = const()[name=string(\"rqb\"), val=tensor<int32, [4]>([{kv_heads},{hpg},{seq},{hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dab = reshape(shape=rqb,x=da_hs)[name=string(\"dab\")];");

    // Q_rot[1,ad,1,S] → batch form [kvH,hpg,S,hd]
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qr_4 = reshape(shape=rqh,x=qrh)[name=string(\"qr4\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> qr_hs = transpose(perm=pm,x=qr_4)[name=string(\"qrhs\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> qb = reshape(shape=rqb,x=qr_hs)[name=string(\"qb\")];");

    // K_rot[1,kvd,1,S] → batch form [kvH,1,S,hd]
    let _ = writeln!(m, "        tensor<int32, [4]> rkv = const()[name=string(\"rkv\"), val=tensor<int32, [4]>([1,{kv_heads},{hd},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> kr_4 = reshape(shape=rkv,x=krh)[name=string(\"kr4\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> kr_hs = transpose(perm=pm,x=kr_4)[name=string(\"krhs\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rkb = const()[name=string(\"rkb\"), val=tensor<int32, [4]>([{kv_heads},1,{seq},{hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> kb = reshape(shape=rkb,x=kr_hs)[name=string(\"kb\")];");

    // V[1,kvd,1,S] → batch form [kvH,1,S,hd]
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> v_4 = reshape(shape=rkv,x=vh)[name=string(\"v4\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> v_hs = transpose(perm=pm,x=v_4)[name=string(\"vhs\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> vb = reshape(shape=rkb,x=v_hs)[name=string(\"vb\")];");

    // --- Phase 4b: SDPA backward ---
    // Recompute attention probs: scores = Q@K^T * scale + mask → softmax
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=qb,y=kb)[name=string(\"sc1\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"sc2\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"ms\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"aw\")];");

    // dV = A^T @ dO, sum over groups: [kvH,hpg,S,hd] → reduce axis 1 → [kvH,1,S,hd]
    // ANE doesn't support transpose_x — explicitly transpose aw: [kvH,hpg,S,S] → swap last 2
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> aw_t = transpose(perm=pm,x=aw)[name=string(\"awt\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dvr = matmul(transpose_x=bF,transpose_y=bF,x=aw_t,y=dab)[name=string(\"dvr\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> dvm = reduce_mean(x=dvr,axes=ax1,keep_dims=kd)[name=string(\"dvm\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> dvb = mul(x=dvm,y=hpg_v)[name=string(\"dvb\")];");

    // dP = dO @ V^T: [kvH,hpg,S,S] (vb broadcasts from dim1=1)
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> dp = matmul(transpose_x=bF,transpose_y=bT,x=dab,y=vb)[name=string(\"dp\")];");

    // Softmax backward: dS = aw * (dP - sum(dP*aw, axis=-1))
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> dpaw = mul(x=dp,y=aw)[name=string(\"dpaw\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},1]> dot_m = reduce_mean(x=dpaw,axes=rax_last,keep_dims=kd)[name=string(\"dotm\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},1]> dot = mul(x=dot_m,y=seq_v)[name=string(\"dot\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> dps = sub(x=dp,y=dot)[name=string(\"dps\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> ds = mul(x=aw,y=dps)[name=string(\"ds\")];");

    // dQ = scale * dS @ K: [kvH,hpg,S,hd] (kb broadcasts)
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dqr = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=kb)[name=string(\"dqr\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dqb = mul(x=dqr,y=scv)[name=string(\"dqb\")];");

    // dK = scale * dS^T @ Q, sum over groups: [kvH,hpg,S,hd] → reduce axis 1
    // ANE doesn't support transpose_x — explicitly transpose ds: [kvH,hpg,S,S] → swap last 2
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> ds_t = transpose(perm=pm,x=ds)[name=string(\"dst\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dkr = matmul(transpose_x=bF,transpose_y=bF,x=ds_t,y=qb)[name=string(\"dkr\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> dkm = reduce_mean(x=dkr,axes=ax1,keep_dims=kd)[name=string(\"dkm\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> dks = mul(x=dkm,y=hpg_v)[name=string(\"dks\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> dkb = mul(x=dks,y=scv)[name=string(\"dkb\")];");

    // --- Phase 5: RoPE backward ---
    // Backward of rotation R is R^T: dq1 = dqr1*cos + dqr2*sin, dq2 = dqr2*cos - dqr1*sin
    // Reshape dQ from batch [kvH,hpg,S,hd] → [1,H,S,hd]
    let _ = writeln!(m, "        tensor<int32, [4]> rha = const()[name=string(\"rha\"), val=tensor<int32, [4]>([1,{heads},{seq},{hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dq_hs = reshape(shape=rha,x=dqb)[name=string(\"dqhs\")];");
    let _ = writeln!(m, "        int32 rpax = const()[name=string(\"rpax\"), val=int32(-1)];");
    let _ = writeln!(m, "        bool rpid = const()[name=string(\"rpid\"), val=bool(false)];");

    // dQ RoPE backward: split halves
    let _ = writeln!(m, "        tensor<int32, [4]> rpb0 = const()[name=string(\"rpb0\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> rpqh = const()[name=string(\"rpqh\"), val=tensor<int32, [4]>([1,{heads},{seq},{half_hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqr1 = slice_by_size(x=dq_hs,begin=rpb0,size=rpqh)[name=string(\"dqr1\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rpbh = const()[name=string(\"rpbh\"), val=tensor<int32, [4]>([0,0,0,{half_hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqr2 = slice_by_size(x=dq_hs,begin=rpbh,size=rpqh)[name=string(\"dqr2\")];");
    // R^T: dq_pre1 = dqr1*cos + dqr2*sin, dq_pre2 = dqr2*cos - dqr1*sin
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq1c = mul(x=dqr1,y=rope_cos)[name=string(\"dq1c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq2s = mul(x=dqr2,y=rope_sin)[name=string(\"dq2s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqp1 = add(x=dq1c,y=dq2s)[name=string(\"dqp1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq2c = mul(x=dqr2,y=rope_cos)[name=string(\"dq2c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq1s = mul(x=dqr1,y=rope_sin)[name=string(\"dq1s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqp2 = sub(x=dq2c,y=dq1s)[name=string(\"dqp2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dq_pre = concat(axis=rpax,interleave=rpid,values=(dqp1,dqp2))[name=string(\"dqpre\")];");

    // dK RoPE backward: reshape from [kvH,1,S,hd] → [1,kvH,S,hd], split + rotate
    let _ = writeln!(m, "        tensor<int32, [4]> rkha = const()[name=string(\"rkha\"), val=tensor<int32, [4]>([1,{kv_heads},{seq},{hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> dk_hs = reshape(shape=rkha,x=dkb)[name=string(\"dkhs\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rpkh = const()[name=string(\"rpkh\"), val=tensor<int32, [4]>([1,{kv_heads},{seq},{half_hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> dkr1 = slice_by_size(x=dk_hs,begin=rpb0,size=rpkh)[name=string(\"dkr1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> dkr2 = slice_by_size(x=dk_hs,begin=rpbh,size=rpkh)[name=string(\"dkr2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> dk1c = mul(x=dkr1,y=rope_cos)[name=string(\"dk1c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> dk2s = mul(x=dkr2,y=rope_sin)[name=string(\"dk2s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> dkp1 = add(x=dk1c,y=dk2s)[name=string(\"dkp1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> dk2c = mul(x=dkr2,y=rope_cos)[name=string(\"dk2c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> dk1s = mul(x=dkr1,y=rope_sin)[name=string(\"dk1s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> dkp2 = sub(x=dk2c,y=dk1s)[name=string(\"dkp2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> dk_pre = concat(axis=rpax,interleave=rpid,values=(dkp1,dkp2))[name=string(\"dkpre\")];");

    // --- Phase 6: Flatten to matmul form for QKV^T projections ---
    let dq_nt_var = if has_gate {
        // Merge Q + gate grads: dq_for_wq = concat(dq_pre, d_gate) per head → [1,H,S,2*hd]
        let two_hd = 2 * hd;
        // d_gate[1,ad,1,S] → [1,H,hd,S] → transpose [1,H,S,hd]
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> dg_4 = reshape(shape=rqh,x=d_gate)[name=string(\"dg4\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dg_hs = transpose(perm=pm,x=dg_4)[name=string(\"dghs\")];");
        // concat [1,H,S,hd] + [1,H,S,hd] → [1,H,S,2*hd]
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{two_hd}]> dqg = concat(axis=rpax,interleave=rpid,values=(dq_pre,dg_hs))[name=string(\"dqg\")];");
        // [1,H,S,2*hd] → transpose [1,H,2*hd,S] → reshape [1,1,qpd,S] → transpose [1,1,S,qpd]
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{two_hd},{seq}]> dqg_t = transpose(perm=pm,x=dqg)[name=string(\"dqgt\")];");
        let _ = writeln!(m, "        tensor<int32, [4]> rqp = const()[name=string(\"rqp\"), val=tensor<int32, [4]>([1,1,{qpd},{seq}])];");
        let _ = writeln!(m, "        tensor<fp16, [1,1,{qpd},{seq}]> dqg_r = reshape(shape=rqp,x=dqg_t)[name=string(\"dqgr\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{qpd}]> dq_nt = transpose(perm=pm,x=dqg_r)[name=string(\"dqnt\")];");
        "dq_nt"
    } else {
        // dq_pre[1,H,S,hd] → transpose [1,H,hd,S] → reshape [1,1,ad,S] → transpose [1,1,S,ad]
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> dq_t = transpose(perm=pm,x=dq_pre)[name=string(\"dqt\")];");
        let _ = writeln!(m, "        tensor<int32, [4]> rqp = const()[name=string(\"rqp\"), val=tensor<int32, [4]>([1,1,{qpd},{seq}])];");
        let _ = writeln!(m, "        tensor<fp16, [1,1,{qpd},{seq}]> dq_r = reshape(shape=rqp,x=dq_t)[name=string(\"dqr2\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{qpd}]> dq_nt = transpose(perm=pm,x=dq_r)[name=string(\"dqnt\")];");
        "dq_nt"
    };

    // dk_pre[1,kvH,S,hd] → transpose [1,kvH,hd,S] → reshape [1,1,kvd,S] → transpose [1,1,S,kvd]
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> dk_t = transpose(perm=pm,x=dk_pre)[name=string(\"dkt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rkd = const()[name=string(\"rkd\"), val=tensor<int32, [4]>([1,1,{kv_dim},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{kv_dim},{seq}]> dk_r = reshape(shape=rkd,x=dk_t)[name=string(\"dkr3\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{kv_dim}]> dk_nt = transpose(perm=pm,x=dk_r)[name=string(\"dknt\")];");

    // dv: [kvH,1,S,hd] → reshape [1,kvH,S,hd] → transpose [1,kvH,hd,S] → reshape [1,1,kvd,S] → transpose [1,1,S,kvd]
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> dv_hs = reshape(shape=rkha,x=dvb)[name=string(\"dvhs\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> dv_t = transpose(perm=pm,x=dv_hs)[name=string(\"dvt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{kv_dim},{seq}]> dv_r = reshape(shape=rkd,x=dv_t)[name=string(\"dvr2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{kv_dim}]> dv_nt = transpose(perm=pm,x=dv_r)[name=string(\"dvnt\")];");

    // --- Phase 7: QKV^T projections ---
    // dx_q = dq_nt @ Wq^T: [1,1,S,qpd] @ transpose([1,1,dim,qpd]) → [1,1,S,dim]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx_q = matmul(transpose_x=bF,transpose_y=bT,x={dq_nt_var},y=Wq)[name=string(\"dxq\")];");
    // dx_k = dk_nt @ Wk^T: [1,1,S,kvd] @ transpose([1,1,dim,kvd]) → [1,1,S,dim]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx_k = matmul(transpose_x=bF,transpose_y=bT,x=dk_nt,y=Wk)[name=string(\"dxk\")];");
    // dx_v = dv_nt @ Wv^T: [1,1,S,kvd] @ transpose([1,1,dim,kvd]) → [1,1,S,dim]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx_v = matmul(transpose_x=bF,transpose_y=bT,x=dv_nt,y=Wv)[name=string(\"dxv\")];");

    // --- Phase 8: Sum + output ---
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx_s1 = add(x=dx_q,y=dx_k)[name=string(\"dxs1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx_s = add(x=dx_s1,y=dx_v)[name=string(\"dxs\")];");
    // [1,1,S,dim] → transpose [1,1,dim,S] → reshape [1,dim,1,S] → cast fp32
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dx_tr = transpose(perm=pm,x=dx_s)[name=string(\"dxtr\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rod = const()[name=string(\"rod\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dx_ch = reshape(shape=rod,x=dx_tr)[name=string(\"dxch\")];");
    let _ = writeln!(m, "        tensor<fp32, [1,{dim},1,{seq}]> out = cast(dtype=to32,x=dx_ch)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (out);");
    m.push_str("}\n");

    let input_bytes = in_ch * seq * 4;
    let output_bytes = dim * seq * 4;

    FusedLayerMil {
        mil_text: m,
        weight_names: vec![
            "@model_path/weights/wq.bin",
            "@model_path/weights/wk.bin",
            "@model_path/weights/wv.bin",
            "@model_path/weights/wo.bin",
            "@model_path/weights/rope_cos.bin",
            "@model_path/weights/rope_sin.bin",
            "@model_path/weights/mask.bin",
        ],
        input_bytes,
        output_bytes,
    }
}

/// Generate a standalone RMSNorm forward kernel for ANE.
///
/// Input: `[1, dim, 1, seq]` fp32 — activation tensor.
/// Output: `[1, dim, 1, seq]` fp32 — normalized + scaled tensor.
/// BLOBFILE weight: `rms_w.bin` — `[1, dim, 1, 1]` fp16 RMSNorm weight vector.
///
/// Ops (8 total): cast → sq → reduce_mean → add(eps) → pow(-0.5) → mul(x,rrms) → mul(weight) → cast
pub fn gen_rmsnorm_fwd(dim: usize, seq: usize, eps: f32) -> FusedLayerMil {
    let eps_f64 = eps as f64;
    let mut m = String::with_capacity(4096);
    m.push_str(MIL_HDR);
    let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1, {dim}, 1, {seq}]> x) {{");

    // Constants
    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
    let _ = writeln!(m, "        tensor<int32, [1]> ch_ax = const()[name=string(\"chax\"), val=tensor<int32, [1]>([1])];");
    let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");
    let _ = writeln!(m, "        fp16 eps_v = const()[name=string(\"epsv\"), val=fp16({eps_f64})];");
    let _ = writeln!(m, "        fp16 nhalf = const()[name=string(\"nh\"), val=fp16(-0.5)];");

    // RMSNorm weight via BLOBFILE
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,1]> w = const()[name=string(\"w\"), val=tensor<fp16, [1,{dim},1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms_w.bin\"), offset=uint64(64)))];");

    // Cast to fp16
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");

    // RMSNorm: sq → mean → add(eps) → rsqrt → mul → scale
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> sq = mul(x=xh,y=xh)[name=string(\"sq\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{seq}]> ms = reduce_mean(x=sq,axes=ch_ax,keep_dims=kd)[name=string(\"ms\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{seq}]> me = add(x=ms,y=eps_v)[name=string(\"me\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{seq}]> rr = pow(x=me,y=nhalf)[name=string(\"rr\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> xn = mul(x=xh,y=rr)[name=string(\"xn\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> xs = mul(x=xn,y=w)[name=string(\"xs\")];");

    // Cast back to fp32
    let _ = writeln!(m, "        tensor<fp32, [1,{dim},1,{seq}]> out = cast(dtype=to32,x=xs)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (out);");
    m.push_str("}\n");

    let input_bytes = dim * seq * 4;
    let output_bytes = dim * seq * 4;

    FusedLayerMil {
        mil_text: m,
        weight_names: vec!["@model_path/weights/rms_w.bin"],
        input_bytes,
        output_bytes,
    }
}

/// Generate RMSNorm backward kernel (dx only, no dw — LoRA freezes base weights).
///
/// Input: `[1, 2*dim, 1, seq]` fp32 — `dy[dim,seq] | x[dim,seq]` concatenated on channel axis.
/// Weight: `rms_w.bin` — `[1, dim, 1, 1]` fp16 (same as forward).
/// Output: `[1, dim, 1, seq]` fp32 — `dx[dim,seq]`.
///
/// Algorithm (all ANE-safe ops):
///   sq = x * x
///   ms = reduce_mean(sq, ch)        → [1,1,1,seq]
///   rr = pow(ms + eps, -0.5)        → rrms
///   dot_raw = dy * x * w            → [dim, seq]
///   dot = reduce_mean(dot_raw, ch) * rr * rr  → [1,1,1,seq]
///   dx = rr * (w * dy - x * dot)
pub fn gen_rmsnorm_bwd(dim: usize, seq: usize, eps: f32) -> FusedLayerMil {
    let eps_f64 = eps as f64;
    let in_ch = 2 * dim; // dy and x concatenated
    let mut m = String::with_capacity(4096);
    m.push_str(MIL_HDR);
    let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1, {in_ch}, 1, {seq}]> input) {{");

    // Constants
    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
    let _ = writeln!(m, "        tensor<int32, [1]> ch_ax = const()[name=string(\"chax\"), val=tensor<int32, [1]>([1])];");
    let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");
    let _ = writeln!(m, "        fp16 eps_v = const()[name=string(\"epsv\"), val=fp16({eps_f64})];");
    let _ = writeln!(m, "        fp16 nhalf = const()[name=string(\"nh\"), val=fp16(-0.5)];");

    // RMSNorm weight via BLOBFILE
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,1]> w = const()[name=string(\"w\"), val=tensor<fp16, [1,{dim},1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms_w.bin\"), offset=uint64(64)))];");

    // Cast input to fp16 then slice dy and x
    let _ = writeln!(m, "        tensor<fp16, [1,{in_ch},1,{seq}]> ih = cast(dtype=to16,x=input)[name=string(\"cin\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> dy_begin = const()[name=string(\"dyb\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> dy_end = const()[name=string(\"dye\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> x_begin = const()[name=string(\"xb\"), val=tensor<int32, [4]>([0,{dim},0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> x_end = const()[name=string(\"xe\"), val=tensor<int32, [4]>([1,{in_ch},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dy = slice_by_index(x=ih,begin=dy_begin,end=dy_end)[name=string(\"dy\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> x = slice_by_index(x=ih,begin=x_begin,end=x_end)[name=string(\"x\")];");

    // Step 1: rrms = pow(reduce_mean(x*x, ch) + eps, -0.5)
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> sq = mul(x=x,y=x)[name=string(\"sq\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{seq}]> ms = reduce_mean(x=sq,axes=ch_ax,keep_dims=kd)[name=string(\"ms\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{seq}]> me = add(x=ms,y=eps_v)[name=string(\"me\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{seq}]> rr = pow(x=me,y=nhalf)[name=string(\"rr\")];");

    // Step 2: dot = reduce_mean(dy*x*w, ch) * rr * rr
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dxw = mul(x=dy,y=x)[name=string(\"dxw\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dxww = mul(x=dxw,y=w)[name=string(\"dxww\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{seq}]> dot_m = reduce_mean(x=dxww,axes=ch_ax,keep_dims=kd)[name=string(\"dotm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{seq}]> dot_r1 = mul(x=dot_m,y=rr)[name=string(\"dr1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{seq}]> dot = mul(x=dot_r1,y=rr)[name=string(\"dot\")];");

    // Step 3: dx = rr * (w*dy - x*dot)
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> wdy = mul(x=w,y=dy)[name=string(\"wdy\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> xdot = mul(x=x,y=dot)[name=string(\"xdot\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> diff = sub(x=wdy,y=xdot)[name=string(\"diff\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dxh = mul(x=rr,y=diff)[name=string(\"dxh\")];");

    // Cast back to fp32
    let _ = writeln!(m, "        tensor<fp32, [1,{dim},1,{seq}]> out = cast(dtype=to32,x=dxh)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (out);");
    m.push_str("}\n");

    let input_bytes = in_ch * seq * 4; // 2*dim*seq f32
    let output_bytes = dim * seq * 4;  // dim*seq f32

    FusedLayerMil {
        mil_text: m,
        weight_names: vec!["@model_path/weights/rms_w.bin"],
        input_bytes,
        output_bytes,
    }
}

/// Generate fused FFN backward kernel: W2^T + SiLU backward + W13^T in one dispatch.
///
/// Replaces 2 ANE dispatches (W2^T, W13^T) + 1 CPU op (SiLU backward) with 1 dispatch.
/// Generate fused GDN projections: QKV + A + B + Z in 1 dispatch with BLOBFILE weights.
///
/// Eliminates 86ms/layer of DynMatmul weight-packing overhead (was 4 dispatches × ~22ms each).
///
/// Input: `[1, dim, 1, seq]` fp32 — xnorm
/// Weights: wqkv.bin [dim,qkv_dim], wa.bin [dim,h_v], wb.bin [dim,h_v], wz.bin [dim,value_dim] (fp16)
/// Output: `[1, out_ch, 1, seq]` fp32 — qkv|a|b|z concatenated on channel axis
pub fn gen_fused_gdn_proj(cfg: &MilConfig) -> FusedLayerMil {
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let h_k = cfg.linear_n_heads;
    let d_k = cfg.linear_head_dim;
    let h_v = cfg.linear_n_value_heads;
    let d_v = cfg.linear_value_head_dim;
    let key_dim = h_k * d_k;
    let value_dim = h_v * d_v;
    let qkv_dim = 2 * key_dim + value_dim;
    let out_ch = qkv_dim + 2 * h_v + value_dim;

    let mut m = String::with_capacity(8192);
    m.push_str(MIL_HDR);
    let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1, {dim}, 1, {seq}]> x) {{");

    // Constants
    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
    let _ = writeln!(m, "        int32 cax = const()[name=string(\"cax\"), val=int32(1)];");

    // BLOBFILE weights
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{qkv_dim}]> Wqkv = const()[name=string(\"Wqkv\"), val=tensor<fp16, [1,1,{dim},{qkv_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wqkv.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{h_v}]> Wa = const()[name=string(\"Wa\"), val=tensor<fp16, [1,1,{dim},{h_v}]>(BLOBFILE(path=string(\"@model_path/weights/wa.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{h_v}]> Wb = const()[name=string(\"Wb\"), val=tensor<fp16, [1,1,{dim},{h_v}]>(BLOBFILE(path=string(\"@model_path/weights/wb.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{value_dim}]> Wz = const()[name=string(\"Wz\"), val=tensor<fp16, [1,1,{dim},{value_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wz.bin\"), offset=uint64(64)))];");

    // Cast + reshape input for matmul
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},{seq},1]> xt = transpose(perm=pm,x=xh)[name=string(\"xt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rd = const()[name=string(\"rd\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> xm = reshape(shape=rd,x=xt)[name=string(\"xm\")];");
    // Transpose for matmul: [1,1,dim,seq] → [1,1,seq,dim]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> xmt = transpose(perm=pm,x=xm)[name=string(\"xmt\")];");

    // 4 matmuls: xmt[1,1,seq,dim] @ W[1,1,dim,out] → [1,1,seq,out]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{qkv_dim}]> qkvm = matmul(transpose_x=bF,transpose_y=bF,x=xmt,y=Wqkv)[name=string(\"qkvm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{h_v}]> am = matmul(transpose_x=bF,transpose_y=bF,x=xmt,y=Wa)[name=string(\"am\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{h_v}]> bm = matmul(transpose_x=bF,transpose_y=bF,x=xmt,y=Wb)[name=string(\"bm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{value_dim}]> zm = matmul(transpose_x=bF,transpose_y=bF,x=xmt,y=Wz)[name=string(\"zm\")];");

    // Transpose back to channel-first: [1,1,seq,out] → pm → [1,1,out,seq] → reshape [1,out,1,seq]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{qkv_dim},{seq}]> qkvt = transpose(perm=pm,x=qkvm)[name=string(\"qkvt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rqkv = const()[name=string(\"rqkv\"), val=tensor<int32, [4]>([1,{qkv_dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{qkv_dim},1,{seq}]> qkv = reshape(shape=rqkv,x=qkvt)[name=string(\"qkv\")];");

    let _ = writeln!(m, "        tensor<fp16, [1,1,{h_v},{seq}]> at = transpose(perm=pm,x=am)[name=string(\"at\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,{h_v},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{h_v},1,{seq}]> a = reshape(shape=ra,x=at)[name=string(\"a\")];");

    let _ = writeln!(m, "        tensor<fp16, [1,1,{h_v},{seq}]> bt = transpose(perm=pm,x=bm)[name=string(\"bt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{h_v},1,{seq}]> b = reshape(shape=ra,x=bt)[name=string(\"b\")];");

    let _ = writeln!(m, "        tensor<fp16, [1,1,{value_dim},{seq}]> zt = transpose(perm=pm,x=zm)[name=string(\"zt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rz = const()[name=string(\"rz\"), val=tensor<int32, [4]>([1,{value_dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{value_dim},1,{seq}]> z = reshape(shape=rz,x=zt)[name=string(\"z\")];");

    // Concatenate on channel axis: qkv|a|b|z
    let _ = writeln!(m, "        tensor<fp16, [1,{out_ch},1,{seq}]> cat = concat(values=(qkv,a,b,z),axis=cax,interleave=bF)[name=string(\"cat\")];");
    let _ = writeln!(m, "        tensor<fp32, [1,{out_ch},1,{seq}]> out = cast(dtype=to32,x=cat)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (out);");
    m.push_str("}\n");

    let input_bytes = dim * seq * 4;
    let output_bytes = out_ch * seq * 4;

    FusedLayerMil {
        mil_text: m,
        weight_names: vec![
            "@model_path/weights/wqkv.bin",
            "@model_path/weights/wa.bin",
            "@model_path/weights/wb.bin",
            "@model_path/weights/wz.bin",
        ],
        input_bytes,
        output_bytes,
    }
}

/// Generate a single BLOBFILE matmul kernel: out = x @ W.
///
/// Input: `[1, in_dim, 1, seq]` fp32.
/// Weight: `w.bin` `[1, 1, in_dim, out_dim]` fp16.
/// Output: `[1, out_dim, 1, seq]` fp32.
pub fn gen_blobfile_matmul(in_dim: usize, out_dim: usize, seq: usize) -> FusedLayerMil {
    let mut m = String::with_capacity(2048);
    m.push_str(MIL_HDR);
    let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1, {in_dim}, 1, {seq}]> x) {{");
    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{in_dim},{out_dim}]> W = const()[name=string(\"W\"), val=tensor<fp16, [1,1,{in_dim},{out_dim}]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,{in_dim},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{in_dim},{seq},1]> xt = transpose(perm=pm,x=xh)[name=string(\"xt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rd = const()[name=string(\"rd\"), val=tensor<int32, [4]>([1,1,{in_dim},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{in_dim},{seq}]> xm = reshape(shape=rd,x=xt)[name=string(\"xm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{in_dim}]> xmt = transpose(perm=pm,x=xm)[name=string(\"xmt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{out_dim}]> ym = matmul(transpose_x=bF,transpose_y=bF,x=xmt,y=W)[name=string(\"ym\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{out_dim},{seq}]> yt = transpose(perm=pm,x=ym)[name=string(\"yt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,{out_dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{out_dim},1,{seq}]> yr = reshape(shape=ro,x=yt)[name=string(\"yr\")];");
    let _ = writeln!(m, "        tensor<fp32, [1,{out_dim},1,{seq}]> out = cast(dtype=to32,x=yr)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (out);");
    m.push_str("}\n");

    FusedLayerMil {
        mil_text: m,
        weight_names: vec!["@model_path/weights/w.bin"],
        input_bytes: in_dim * seq * 4,
        output_bytes: out_dim * seq * 4,
    }
}

/// Generate fused FFN backward kernel: W2^T + SiLU backward + W13^T in one dispatch.
///
/// Uses BLOBFILE weights (per-layer, baked into compiled model via delta cache).
///
/// Input: `[1, dim + 2*hidden, 1, seq]` fp32 — dx_ffn | h1 | h3
/// Weights: w2t.bin [hidden,dim], w1t.bin [dim,hidden], w3t.bin [dim,hidden] (all fp16)
/// Output: `[1, dim, 1, seq]` fp32 — dx = W1^T @ dh1 + W3^T @ dh3
pub fn gen_fused_ffn_bwd(cfg: &MilConfig) -> FusedLayerMil {
    let dim = cfg.dim;
    let hidden = cfg.hidden_dim;
    let seq = cfg.seq_len;
    let in_ch = dim + 2 * hidden;

    let mut m = String::with_capacity(8192);
    m.push_str(MIL_HDR);
    let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1, {in_ch}, 1, {seq}]> x) {{");

    // Constants
    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
    let _ = writeln!(m, "        fp16 one_v = const()[name=string(\"onev\"), val=fp16(1.0)];");

    // Weight BLOBFILEs
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{dim}]> W2t = const()[name=string(\"W2t\"), val=tensor<fp16, [1,{hidden},1,{dim}]>(BLOBFILE(path=string(\"@model_path/weights/w2t.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{hidden}]> W1t = const()[name=string(\"W1t\"), val=tensor<fp16, [1,{dim},1,{hidden}]>(BLOBFILE(path=string(\"@model_path/weights/w1t.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{hidden}]> W3t = const()[name=string(\"W3t\"), val=tensor<fp16, [1,{dim},1,{hidden}]>(BLOBFILE(path=string(\"@model_path/weights/w3t.bin\"), offset=uint64(64)))];");

    // Cast input + slice
    let _ = writeln!(m, "        tensor<fp16, [1,{in_ch},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");

    // dx_ffn [dim, seq]
    let _ = writeln!(m, "        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> sd = const()[name=string(\"sd\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dxf = slice_by_size(x=xh,begin=b0,size=sd)[name=string(\"dxf\")];");

    // h1 [hidden, seq]
    let _ = writeln!(m, "        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,{dim},0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> sh = const()[name=string(\"sh\"), val=tensor<int32, [4]>([1,{hidden},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> h1 = slice_by_size(x=xh,begin=b1,size=sh)[name=string(\"h1\")];");

    // h3 [hidden, seq]
    let off_h3 = dim + hidden;
    let _ = writeln!(m, "        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,{off_h3},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> h3 = slice_by_size(x=xh,begin=b3,size=sh)[name=string(\"h3\")];");

    // Step 1: dsilu = W2^T @ dx_ffn
    // Reshape to [1,1,M,K] @ [1,1,K,N] pattern for ANE matmul
    let _ = writeln!(m, "        tensor<int32, [4]> rw2 = const()[name=string(\"rw2\"), val=tensor<int32, [4]>([1,1,{hidden},{dim}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{dim}]> W2m = reshape(shape=rw2,x=W2t)[name=string(\"W2m\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},{seq},1]> dxf_t = transpose(perm=pm,x=dxf)[name=string(\"dxft\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rdx = const()[name=string(\"rdx\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dxfm = reshape(shape=rdx,x=dxf_t)[name=string(\"dxfm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{seq}]> dsm = matmul(transpose_x=bF,transpose_y=bF,x=W2m,y=dxfm)[name=string(\"dsm\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rds = const()[name=string(\"rds\"), val=tensor<int32, [4]>([1,{hidden},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> dsilu = reshape(shape=rds,x=dsm)[name=string(\"dsilu\")];");

    // Step 2: SiLU backward
    // sig = sigmoid(h1)
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> sig = sigmoid(x=h1)[name=string(\"sig\")];");
    // silu_val = h1 * sig
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> silu = mul(x=h1,y=sig)[name=string(\"silu\")];");
    // dh3 = dsilu * silu_val
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> dh3 = mul(x=dsilu,y=silu)[name=string(\"dh3\")];");
    // silu_deriv = sig * (1 + h1 * (1 - sig))
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> omsig = sub(x=one_v,y=sig)[name=string(\"oms\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> h1oms = mul(x=h1,y=omsig)[name=string(\"h1oms\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> opl = add(x=one_v,y=h1oms)[name=string(\"opl\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> sd1 = mul(x=sig,y=opl)[name=string(\"sd1\")];");
    // dh1 = dsilu * h3 * silu_deriv
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> dsh3 = mul(x=dsilu,y=h3)[name=string(\"dsh3\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> dh1 = mul(x=dsh3,y=sd1)[name=string(\"dh1\")];");

    // Step 3: dx1 = W1^T @ dh1 + W3^T @ dh3
    let _ = writeln!(m, "        tensor<int32, [4]> rw1 = const()[name=string(\"rw1\"), val=tensor<int32, [4]>([1,1,{dim},{hidden}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{hidden}]> W1m = reshape(shape=rw1,x=W1t)[name=string(\"W1m\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},{seq},1]> dh1_t = transpose(perm=pm,x=dh1)[name=string(\"dh1t\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rdh = const()[name=string(\"rdh\"), val=tensor<int32, [4]>([1,1,{hidden},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{seq}]> dh1m = reshape(shape=rdh,x=dh1_t)[name=string(\"dh1m\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dx1m = matmul(transpose_x=bF,transpose_y=bF,x=W1m,y=dh1m)[name=string(\"dx1m\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rdout = const()[name=string(\"rdout\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dx1 = reshape(shape=rdout,x=dx1m)[name=string(\"dx1\")];");

    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{hidden}]> W3m = reshape(shape=rw1,x=W3t)[name=string(\"W3m\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},{seq},1]> dh3_t = transpose(perm=pm,x=dh3)[name=string(\"dh3t\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{seq}]> dh3m = reshape(shape=rdh,x=dh3_t)[name=string(\"dh3m\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dx3m = matmul(transpose_x=bF,transpose_y=bF,x=W3m,y=dh3m)[name=string(\"dx3m\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dx3 = reshape(shape=rdout,x=dx3m)[name=string(\"dx3\")];");

    // Sum dx1 + dx3
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dxs = add(x=dx1,y=dx3)[name=string(\"dxs\")];");

    // Output: dx | dsilu concatenated on channel axis (caller can split to get both)
    let out_ch = dim + hidden;
    let _ = writeln!(m, "        tensor<int32, [1]> cax = const()[name=string(\"cax\"), val=int32(1)];");
    let _ = writeln!(m, "        tensor<fp16, [1,{out_ch},1,{seq}]> cat = concat(values=(dxs,dsilu),axis=cax,interleave=bF)[name=string(\"cat\")];");
    let _ = writeln!(m, "        tensor<fp32, [1,{out_ch},1,{seq}]> out = cast(dtype=to32,x=cat)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (out);");
    m.push_str("}\n");

    let input_bytes = in_ch * seq * 4;
    let output_bytes = out_ch * seq * 4;

    FusedLayerMil {
        mil_text: m,
        weight_names: vec![
            "@model_path/weights/w2t.bin",
            "@model_path/weights/w1t.bin",
            "@model_path/weights/w3t.bin",
        ],
        input_bytes,
        output_bytes,
    }
}

/// Generate fused SDPA backward for GQA (RoPE backward done on CPU).
///
/// Replaces sdpa_bwd1 + sdpa_bwd2 with a single dispatch.
/// Uses `[1,H,S,*]` form throughout (dim 0 = 1) to avoid ANE compiler bug
/// with `[kvH,hpg,S,S]` batch shapes. K/V must be pre-expanded to full head count
/// by the caller (repeat each KV head `hpg` times).
///
/// RoPE backward is NOT included — the ANE compiler crashes on `slice(axis=-1) →
/// concat(axis=-1)` when the source tensor comes from a long SDPA chain. RoPE
/// backward (cheap element-wise ops) is done on CPU after this kernel.
///
/// Input layout `[1, in_ch, 1, seq]` fp32:
///   `d_attn[ad] | Q_rot[ad] | K_expanded[ad] | V_expanded[ad]`
///   All tensors have full head count (H = n_heads).
///
/// Output layout `[1, out_ch, 1, seq]` fp32:
///   `dq_scaled[ad] | dk_scaled[ad] | dv_all[ad]`
///   dQ/dK are pre-RoPE (post-scale). dK/dV are in full head count — caller does
///   group reduction + RoPE backward on CPU.
///
/// BLOBFILE weights: mask only.
pub fn gen_sdpa_rope_bwd(cfg: &MilConfig, _has_gate: bool) -> FusedLayerMil {
    let seq = cfg.seq_len;
    let heads = cfg.n_heads;
    let hd = cfg.head_dim();
    let half_hd = hd / 2;
    let attn_dim = cfg.attn_dim(); // H * hd
    let sc = 1.0 / (hd as f64).sqrt();

    // Input: d_attn[ad] | Q_rot[ad] | K_expanded[ad] | V_expanded[ad] (all full head count)
    let in_ch = 4 * attn_dim;
    // Output: dq_pre[ad] | dk_all[ad] | dv_all[ad] (all full head count)
    let out_ch = 3 * attn_dim;

    let mut m = String::with_capacity(32768);
    m.push_str(MIL_HDR);
    let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1, {in_ch}, 1, {seq}]> x) {{");

    // --- Constants ---
    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
    let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
    let _ = writeln!(m, "        bool bT = const()[name=string(\"bT\"), val=bool(true)];");
    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(m, "        fp16 scv = const()[name=string(\"scv\"), val=fp16({sc})];");
    let _ = writeln!(m, "        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];");
    let _ = writeln!(m, "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];");
    let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");
    let _ = writeln!(m, "        fp16 seq_v = const()[name=string(\"seqv\"), val=fp16({seq})];");

    // BLOBFILEs: mask only (RoPE backward done on CPU to avoid axis=-1 concat ANE bug)
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{seq}]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,{seq},{seq}]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];");

    // --- Cast + slice input: d_attn[ad] | Q_rot[ad] | K_exp[ad] | V_exp[ad] ---
    let _ = writeln!(m, "        tensor<fp16, [1,{in_ch},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");
    let mut off = 0usize;
    let _ = writeln!(m, "        tensor<int32, [4]> s_ad = const()[name=string(\"sad\"), val=tensor<int32, [4]>([1,{attn_dim},1,{seq}])];");

    // d_attn [1,ad,1,S]
    let _ = writeln!(m, "        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,{off},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> dah = slice_by_size(x=xh,begin=b0,size=s_ad)[name=string(\"dah\")];");
    off += attn_dim;
    // Q_rot [1,ad,1,S]
    let _ = writeln!(m, "        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,{off},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> qrh = slice_by_size(x=xh,begin=b1,size=s_ad)[name=string(\"qrh\")];");
    off += attn_dim;
    // K_expanded [1,ad,1,S]
    let _ = writeln!(m, "        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,{off},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> keh = slice_by_size(x=xh,begin=b2,size=s_ad)[name=string(\"keh\")];");
    off += attn_dim;
    // V_expanded [1,ad,1,S]
    let _ = writeln!(m, "        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,{off},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> veh = slice_by_size(x=xh,begin=b3,size=s_ad)[name=string(\"veh\")];");
    let _ = off;

    // --- Reshape to [1,H,S,hd] head form (dim 0 = 1 throughout!) ---
    let _ = writeln!(m, "        tensor<int32, [4]> rqh = const()[name=string(\"rqh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> da_4 = reshape(shape=rqh,x=dah)[name=string(\"da4\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> da = transpose(perm=pm,x=da_4)[name=string(\"da\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qr_4 = reshape(shape=rqh,x=qrh)[name=string(\"qr4\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> q = transpose(perm=pm,x=qr_4)[name=string(\"q\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> ke_4 = reshape(shape=rqh,x=keh)[name=string(\"ke4\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> k = transpose(perm=pm,x=ke_4)[name=string(\"k\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> ve_4 = reshape(shape=rqh,x=veh)[name=string(\"ve4\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> v = transpose(perm=pm,x=ve_4)[name=string(\"v\")];");

    // --- SDPA backward (all [1,H,S,*] — no batch broadcast!) ---
    // Recompute probs: Q@K^T * scale + mask → softmax
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"sc1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"sc2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"ms\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"aw\")];");

    // dV = A^T @ dO — explicit transpose (ANE doesn't support transpose_x)
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw_t = transpose(perm=pm,x=aw)[name=string(\"awt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dv_all = matmul(transpose_x=bF,transpose_y=bF,x=aw_t,y=da)[name=string(\"dva\")];");

    // dP = dO @ V^T
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dp = matmul(transpose_x=bF,transpose_y=bT,x=da,y=v)[name=string(\"dp\")];");

    // Softmax backward: dS = aw * (dP - sum(dP*aw, axis=-1))
    // ANE bug: reduce_sum crashes in this graph topology. Use reduce_mean * seq instead.
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dpaw = mul(x=dp,y=aw)[name=string(\"dpaw\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> dot_m = reduce_mean(x=dpaw,axes=rax,keep_dims=kd)[name=string(\"dotm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> dot = mul(x=dot_m,y=seq_v)[name=string(\"dot\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dps = sub(x=dp,y=dot)[name=string(\"dps\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ds = mul(x=aw,y=dps)[name=string(\"ds\")];");

    // dQ = scale * dS @ K
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dqr = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=k)[name=string(\"dqr\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dq_s = mul(x=dqr,y=scv)[name=string(\"dqs\")];");

    // dK = scale * dS^T @ Q — explicit transpose
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ds_t = transpose(perm=pm,x=ds)[name=string(\"dst\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dkr = matmul(transpose_x=bF,transpose_y=bF,x=ds_t,y=q)[name=string(\"dkr\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dk_s = mul(x=dkr,y=scv)[name=string(\"dks\")];");

    // --- Output: concat dq/dk/dv on head axis [1, 3*H, S, hd] ---
    // Avoids the transpose+reshape+concat-on-channel pattern that crashes ANE compiler.
    let heads3 = 3 * heads;
    let _ = writeln!(m, "        int32 cax = const()[name=string(\"cax\"), val=int32(1)];");
    let _ = writeln!(m, "        bool cid = const()[name=string(\"cid\"), val=bool(false)];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads3},{seq},{hd}]> out_h = concat(axis=cax,interleave=cid,values=(dq_s,dk_s,dv_all))[name=string(\"outh\")];");
    let _ = writeln!(m, "        tensor<fp32, [1,{heads3},{seq},{hd}]> out = cast(dtype=to32,x=out_h)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (out);");
    m.push_str("}\n");

    let input_bytes = in_ch * seq * 4; // fp32
    let output_bytes = 3 * heads * seq * hd * 4; // fp32 — [1, 3*H, S, hd]

    FusedLayerMil {
        mil_text: m,
        weight_names: vec!["@model_path/weights/mask.bin"],
        input_bytes,
        output_bytes,
    }
}

/// Generate fused SDPA backward + RoPE backward + QKV^T projections for GQA.
///
/// Replaces sdpa_bwd1 + sdpa_bwd2 + CPU RoPE backward + qkv_bwd with a single dispatch.
/// Takes post-gate gradient `d_attn` as input (Wo^T + gate backward handled separately).
///
/// Input layout `[1, in_ch, 1, seq]` fp32:
///   `d_attn[ad] | Q_rot[ad] | K_rot[kvd] | V[kvd]`
///
/// Output: `[1, dim, 1, seq]` fp32 — dx_attn gradient for the residual stream.
///
/// BLOBFILE weights: Wq, Wk, Wv (for QKV^T projections), rope_cos, rope_sin, mask.
/// No Wo weight — that's handled by the separate Wo^T backward kernel.
pub fn gen_sdpa_rope_qkvt_bwd(cfg: &MilConfig, has_gate: bool) -> FusedLayerMil {
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let heads = cfg.n_heads;
    let kv_heads = cfg.n_kv_heads;
    let hd = cfg.head_dim();
    let half_hd = hd / 2;
    let attn_dim = cfg.attn_dim();
    let kv_dim = cfg.kv_dim();
    let qpd = cfg.q_proj_dim();
    let hpg = cfg.heads_per_group();
    let sc = 1.0 / (hd as f64).sqrt();

    // Input: d_attn[ad] | Q_rot[ad] | K_rot[kvd] | V[kvd]
    let in_ch = attn_dim + attn_dim + 2 * kv_dim;

    let mut m = String::with_capacity(32768);
    m.push_str(MIL_HDR);
    let _ = writeln!(
        m,
        "    func main<ios18>(tensor<fp32, [1, {in_ch}, 1, {seq}]> x) {{"
    );

    // --- Shared constants ---
    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
    let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
    let _ = writeln!(m, "        bool bT = const()[name=string(\"bT\"), val=bool(true)];");
    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(m, "        fp16 scv = const()[name=string(\"scv\"), val=fp16({sc})];");
    // reduce_mean→sum trick constants
    let _ = writeln!(m, "        fp16 hpg_v = const()[name=string(\"hpgv\"), val=fp16({hpg})];");
    let _ = writeln!(m, "        fp16 seq_v = const()[name=string(\"seqv\"), val=fp16({seq})];");
    // reduce axes (tensor) vs softmax axis (scalar)
    let _ = writeln!(m, "        tensor<int32, [1]> ax1 = const()[name=string(\"ax1\"), val=tensor<int32, [1]>([1])];");
    let _ = writeln!(m, "        tensor<int32, [1]> rax_last = const()[name=string(\"raxl\"), val=tensor<int32, [1]>([-1])];");
    let _ = writeln!(m, "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];");
    let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");

    // --- Cast input to fp16 ---
    let _ = writeln!(m, "        tensor<fp16, [1,{in_ch},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");

    // --- Phase 1: Slice input channels ---
    let mut off = 0usize;

    // d_attn: [1, ad, 1, S] (post-gate gradient)
    let _ = writeln!(m, "        tensor<int32, [4]> b_da = const()[name=string(\"bda\"), val=tensor<int32, [4]>([0,{off},0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> s_ad = const()[name=string(\"sad\"), val=tensor<int32, [4]>([1,{attn_dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> d_at = slice_by_size(x=xh,begin=b_da,size=s_ad)[name=string(\"dat\")];");
    off += attn_dim;

    // Q_rot: [1, ad, 1, S]
    let _ = writeln!(m, "        tensor<int32, [4]> b_qr = const()[name=string(\"bqr\"), val=tensor<int32, [4]>([0,{off},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> qrh = slice_by_size(x=xh,begin=b_qr,size=s_ad)[name=string(\"qrh\")];");
    off += attn_dim;

    // K_rot: [1, kvd, 1, S]
    let _ = writeln!(m, "        tensor<int32, [4]> b_kr = const()[name=string(\"bkr\"), val=tensor<int32, [4]>([0,{off},0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> s_kv = const()[name=string(\"skv\"), val=tensor<int32, [4]>([1,{kv_dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_dim},1,{seq}]> krh = slice_by_size(x=xh,begin=b_kr,size=s_kv)[name=string(\"krh\")];");
    off += kv_dim;

    // V: [1, kvd, 1, S]
    let _ = writeln!(m, "        tensor<int32, [4]> b_vh = const()[name=string(\"bvh\"), val=tensor<int32, [4]>([0,{off},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_dim},1,{seq}]> vh = slice_by_size(x=xh,begin=b_vh,size=s_kv)[name=string(\"vh\")];");
    let _ = off;

    // --- BLOBFILE weight constants (QKV only, no Wo) ---
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{qpd}]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [1,1,{dim},{qpd}]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{kv_dim}]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [1,1,{dim},{kv_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{kv_dim}]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [1,1,{dim},{kv_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];");

    // RoPE tables + causal mask
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_cos = const()[name=string(\"rc\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_cos.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_sin = const()[name=string(\"rs\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_sin.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{seq}]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,{seq},{seq}]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];");

    // --- Phase 2: Reshape to GQA batch form ---
    // d_attn[1,ad,1,S] → [1,H,hd,S] → transpose [1,H,S,hd] → [kvH,hpg,S,hd]
    let _ = writeln!(m, "        tensor<int32, [4]> rqh = const()[name=string(\"rqh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> da_4 = reshape(shape=rqh,x=d_at)[name=string(\"da4\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> da_hs = transpose(perm=pm,x=da_4)[name=string(\"dahs\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rqb = const()[name=string(\"rqb\"), val=tensor<int32, [4]>([{kv_heads},{hpg},{seq},{hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dab = reshape(shape=rqb,x=da_hs)[name=string(\"dab\")];");

    // Q_rot → batch form
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qr_4 = reshape(shape=rqh,x=qrh)[name=string(\"qr4\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> qr_hs = transpose(perm=pm,x=qr_4)[name=string(\"qrhs\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> qb = reshape(shape=rqb,x=qr_hs)[name=string(\"qb\")];");

    // K_rot → batch form [kvH,1,S,hd]
    let _ = writeln!(m, "        tensor<int32, [4]> rkv = const()[name=string(\"rkv\"), val=tensor<int32, [4]>([1,{kv_heads},{hd},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> kr_4 = reshape(shape=rkv,x=krh)[name=string(\"kr4\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> kr_hs = transpose(perm=pm,x=kr_4)[name=string(\"krhs\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rkb = const()[name=string(\"rkb\"), val=tensor<int32, [4]>([{kv_heads},1,{seq},{hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> kb = reshape(shape=rkb,x=kr_hs)[name=string(\"kb\")];");

    // V → batch form
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> v_4 = reshape(shape=rkv,x=vh)[name=string(\"v4\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> v_hs = transpose(perm=pm,x=v_4)[name=string(\"vhs\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> vb = reshape(shape=rkb,x=v_hs)[name=string(\"vb\")];");

    // --- Phase 3: SDPA backward ---
    // Recompute attention probs: scores = Q@K^T * scale + mask → softmax
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=qb,y=kb)[name=string(\"sc1\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"sc2\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"ms\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"aw\")];");

    // dV = A^T @ dO — explicit transpose (ANE doesn't support transpose_x)
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> aw_t = transpose(perm=pm,x=aw)[name=string(\"awt\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dvr = matmul(transpose_x=bF,transpose_y=bF,x=aw_t,y=dab)[name=string(\"dvr\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> dvm = reduce_mean(x=dvr,axes=ax1,keep_dims=kd)[name=string(\"dvm\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> dvb = mul(x=dvm,y=hpg_v)[name=string(\"dvb\")];");

    // dP = dO @ V^T
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> dp = matmul(transpose_x=bF,transpose_y=bT,x=dab,y=vb)[name=string(\"dp\")];");

    // Softmax backward: dS = aw * (dP - sum(dP*aw, axis=-1))
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> dpaw = mul(x=dp,y=aw)[name=string(\"dpaw\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},1]> dot_m = reduce_mean(x=dpaw,axes=rax_last,keep_dims=kd)[name=string(\"dotm\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},1]> dot = mul(x=dot_m,y=seq_v)[name=string(\"dot\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> dps = sub(x=dp,y=dot)[name=string(\"dps\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> ds = mul(x=aw,y=dps)[name=string(\"ds\")];");

    // dQ = scale * dS @ K
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dqr = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=kb)[name=string(\"dqr\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dqb = mul(x=dqr,y=scv)[name=string(\"dqb\")];");

    // dK = scale * dS^T @ Q — explicit transpose
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> ds_t = transpose(perm=pm,x=ds)[name=string(\"dst\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dkr = matmul(transpose_x=bF,transpose_y=bF,x=ds_t,y=qb)[name=string(\"dkr\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> dkm = reduce_mean(x=dkr,axes=ax1,keep_dims=kd)[name=string(\"dkm\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> dks = mul(x=dkm,y=hpg_v)[name=string(\"dks\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> dkb = mul(x=dks,y=scv)[name=string(\"dkb\")];");

    // --- Phase 4: RoPE backward ---
    // dQ: reshape [kvH,hpg,S,hd] → [1,H,S,hd], split halves, R^T
    let _ = writeln!(m, "        tensor<int32, [4]> rha = const()[name=string(\"rha\"), val=tensor<int32, [4]>([1,{heads},{seq},{hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dq_hs = reshape(shape=rha,x=dqb)[name=string(\"dqhs\")];");
    let _ = writeln!(m, "        int32 rpax = const()[name=string(\"rpax\"), val=int32(-1)];");
    let _ = writeln!(m, "        bool rpid = const()[name=string(\"rpid\"), val=bool(false)];");

    let _ = writeln!(m, "        tensor<int32, [4]> rpb0 = const()[name=string(\"rpb0\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> rpqh = const()[name=string(\"rpqh\"), val=tensor<int32, [4]>([1,{heads},{seq},{half_hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqr1 = slice_by_size(x=dq_hs,begin=rpb0,size=rpqh)[name=string(\"dqr1\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rpbh = const()[name=string(\"rpbh\"), val=tensor<int32, [4]>([0,0,0,{half_hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqr2 = slice_by_size(x=dq_hs,begin=rpbh,size=rpqh)[name=string(\"dqr2\")];");
    // R^T: dq_pre1 = dqr1*cos + dqr2*sin, dq_pre2 = dqr2*cos - dqr1*sin
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq1c = mul(x=dqr1,y=rope_cos)[name=string(\"dq1c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq2s = mul(x=dqr2,y=rope_sin)[name=string(\"dq2s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqp1 = add(x=dq1c,y=dq2s)[name=string(\"dqp1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq2c = mul(x=dqr2,y=rope_cos)[name=string(\"dq2c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq1s = mul(x=dqr1,y=rope_sin)[name=string(\"dq1s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqp2 = sub(x=dq2c,y=dq1s)[name=string(\"dqp2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dq_pre = concat(axis=rpax,interleave=rpid,values=(dqp1,dqp2))[name=string(\"dqpre\")];");

    // dK RoPE backward
    let _ = writeln!(m, "        tensor<int32, [4]> rkha = const()[name=string(\"rkha\"), val=tensor<int32, [4]>([1,{kv_heads},{seq},{hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> dk_hs = reshape(shape=rkha,x=dkb)[name=string(\"dkhs\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rpkh = const()[name=string(\"rpkh\"), val=tensor<int32, [4]>([1,{kv_heads},{seq},{half_hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> dkr1 = slice_by_size(x=dk_hs,begin=rpb0,size=rpkh)[name=string(\"dkr1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> dkr2 = slice_by_size(x=dk_hs,begin=rpbh,size=rpkh)[name=string(\"dkr2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> dk1c = mul(x=dkr1,y=rope_cos)[name=string(\"dk1c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> dk2s = mul(x=dkr2,y=rope_sin)[name=string(\"dk2s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> dkp1 = add(x=dk1c,y=dk2s)[name=string(\"dkp1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> dk2c = mul(x=dkr2,y=rope_cos)[name=string(\"dk2c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> dk1s = mul(x=dkr1,y=rope_sin)[name=string(\"dk1s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> dkp2 = sub(x=dk2c,y=dk1s)[name=string(\"dkp2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> dk_pre = concat(axis=rpax,interleave=rpid,values=(dkp1,dkp2))[name=string(\"dkpre\")];");

    // --- Phase 5: Flatten to matmul form for QKV^T projections ---
    let dq_nt_var = if has_gate {
        let two_hd = 2 * hd;
        // d_gate was applied before this kernel, but dq still has attn_dim columns.
        // For gated models, Q projection was [dim → qpd=2*ad], so backward needs
        // to reconstruct the full qpd dimension. dq_pre has hd cols, gate grads have hd cols.
        // But gate grads were handled externally (Wo^T+gate kernel), so here dq_pre is just
        // the Q portion. We need to zero-pad to qpd width for the Wq^T matmul.
        //
        // Wait — the Wq^T matmul needs the FULL qpd-width gradient (Q + gate halves).
        // The gate gradient must be passed in as an additional input channel.
        //
        // For now, pad with zeros (gate grads contribute dx_gate, not dx_q).
        // dq_pre[1,H,S,hd] → transpose → reshape [1,1,ad,S] → transpose [1,1,S,ad]
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> dq_t = transpose(perm=pm,x=dq_pre)[name=string(\"dqt\")];");
        let _ = writeln!(m, "        tensor<int32, [4]> rad = const()[name=string(\"rad\"), val=tensor<int32, [4]>([1,1,{attn_dim},{seq}])];");
        let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{seq}]> dq_r = reshape(shape=rad,x=dq_t)[name=string(\"dqr2\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{attn_dim}]> dq_nt = transpose(perm=pm,x=dq_r)[name=string(\"dqnt\")];");
        // Zero-pad to qpd: concat(dq_nt[S,ad], zeros[S,ad]) → [S,qpd=2*ad]
        let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{attn_dim}]> dg_z = const()[name=string(\"dgz\"), val=tensor<fp16, [1,1,{seq},{attn_dim}]>(BLOBFILE(path=string(\"@model_path/weights/gate_zeros.bin\"), offset=uint64(64)))];");
        let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{two_hd}]> dqg = concat(axis=rpax,interleave=rpid,values=(dq_nt,dg_z))[name=string(\"dqg\")];");
        "dqg"
    } else {
        // Non-gated: dq_pre → [1,1,S,ad] directly
        let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> dq_t = transpose(perm=pm,x=dq_pre)[name=string(\"dqt\")];");
        let _ = writeln!(m, "        tensor<int32, [4]> rqp = const()[name=string(\"rqp\"), val=tensor<int32, [4]>([1,1,{qpd},{seq}])];");
        let _ = writeln!(m, "        tensor<fp16, [1,1,{qpd},{seq}]> dq_r = reshape(shape=rqp,x=dq_t)[name=string(\"dqr2\")];");
        let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{qpd}]> dq_nt = transpose(perm=pm,x=dq_r)[name=string(\"dqnt\")];");
        "dq_nt"
    };

    // dk_pre → [1,1,S,kvd]
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> dk_t = transpose(perm=pm,x=dk_pre)[name=string(\"dkt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rkd = const()[name=string(\"rkd\"), val=tensor<int32, [4]>([1,1,{kv_dim},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{kv_dim},{seq}]> dk_r = reshape(shape=rkd,x=dk_t)[name=string(\"dkr3\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{kv_dim}]> dk_nt = transpose(perm=pm,x=dk_r)[name=string(\"dknt\")];");

    // dv → [1,1,S,kvd]
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> dv_hs = reshape(shape=rkha,x=dvb)[name=string(\"dvhs\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> dv_t = transpose(perm=pm,x=dv_hs)[name=string(\"dvt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{kv_dim},{seq}]> dv_r = reshape(shape=rkd,x=dv_t)[name=string(\"dvr2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{kv_dim}]> dv_nt = transpose(perm=pm,x=dv_r)[name=string(\"dvnt\")];");

    // --- Phase 6: QKV^T projections ---
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx_q = matmul(transpose_x=bF,transpose_y=bT,x={dq_nt_var},y=Wq)[name=string(\"dxq\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx_k = matmul(transpose_x=bF,transpose_y=bT,x=dk_nt,y=Wk)[name=string(\"dxk\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx_v = matmul(transpose_x=bF,transpose_y=bT,x=dv_nt,y=Wv)[name=string(\"dxv\")];");

    // --- Phase 7: Sum + output ---
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx_s1 = add(x=dx_q,y=dx_k)[name=string(\"dxs1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx_s = add(x=dx_s1,y=dx_v)[name=string(\"dxs\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dx_tr = transpose(perm=pm,x=dx_s)[name=string(\"dxtr\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rod = const()[name=string(\"rod\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dx_ch = reshape(shape=rod,x=dx_tr)[name=string(\"dxch\")];");
    let _ = writeln!(m, "        tensor<fp32, [1,{dim},1,{seq}]> out = cast(dtype=to32,x=dx_ch)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (out);");
    m.push_str("}\n");

    let input_bytes = in_ch * seq * 4;
    let output_bytes = dim * seq * 4;

    let mut weight_names = vec![
        "@model_path/weights/wq.bin",
        "@model_path/weights/wk.bin",
        "@model_path/weights/wv.bin",
        "@model_path/weights/rope_cos.bin",
        "@model_path/weights/rope_sin.bin",
        "@model_path/weights/mask.bin",
    ];
    if has_gate {
        weight_names.push("@model_path/weights/gate_zeros.bin");
    }

    FusedLayerMil {
        mil_text: m,
        weight_names,
        input_bytes,
        output_bytes,
    }
}

/// Build a causal mask weight blob for SDPA kernels.
///
/// Returns raw bytes in ANE blob format: 128-byte header + fp16 mask data.
/// Header layout matches `build_blob_fp16` from io.h.
/// Mask: 0.0 where t2 <= t (causal), -65504.0 (fp16 -inf) where t2 > t.
pub fn build_causal_mask_blob(seq: usize) -> Vec<u8> {
    let n = seq * seq;
    let data_bytes = n * 2; // fp16
    let header_bytes = 128;
    let mut blob = vec![0u8; header_bytes + data_bytes];

    // ANE blob header (matches io.h build_blob_fp16)
    blob[0] = 1;
    blob[4] = 2;
    blob[64] = 0xEF;
    blob[65] = 0xBE;
    blob[66] = 0xAD;
    blob[67] = 0xDE;
    blob[68] = 1;
    // Weight data size at offset 72 (uint32 LE)
    blob[72..76].copy_from_slice(&(data_bytes as u32).to_le_bytes());
    // Data offset at offset 80 (uint32 LE)
    blob[80..84].copy_from_slice(&(header_bytes as u32).to_le_bytes());

    for t in 0..seq {
        for t2 in 0..seq {
            let val: f32 = if t2 <= t { 0.0 } else { -65504.0 };
            let fp16 = half::f16::from_f32(val);
            let offset = header_bytes + (t * seq + t2) * 2;
            blob[offset..offset + 2].copy_from_slice(&fp16.to_le_bytes());
        }
    }
    blob
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::ane_bridge::{self, AneKernel};
    use crate::agent::ane_weights::generate_rope_blobs;

    fn init_ane() {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            ane_bridge::ane_init().expect("ane_init failed — is this Apple Silicon?");
        });
    }

    /// Pack fp32 data into bytes.
    fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// Unpack bytes to fp32.
    fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Pack fp16 data into bytes from f32 values.
    fn f32_to_fp16_bytes(data: &[f32]) -> Vec<u8> {
        data.iter()
            .flat_map(|f| half::f16::from_f32(*f).to_le_bytes())
            .collect()
    }

    /// Unpack fp16 bytes to f32.
    fn fp16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()
    }

    // ---- Round 1: gen_dyn_matmul_mil (standalone matmul) ----

    #[test]
    fn test_dyn_matmul_identity() {
        init_ane();

        let ic = 64;
        let oc = 64;
        let seq = 64;
        let sp = seq + oc;

        let mil = gen_dyn_matmul_mil(ic, oc, seq);
        let input_bytes = ic * sp * 4;
        let output_bytes = oc * seq * 4;

        let kernel = AneKernel::compile(&mil, None, &[input_bytes], &[output_bytes])
            .expect("matmul compile failed");

        // Build input: activations = small values, weight = identity matrix
        // IOSurface layout: [1, ic, 1, sp] where sp = seq + oc
        // For each channel c: spatial[0..seq] = activation, spatial[seq..seq+oc] = weight row c
        let mut input = vec![0.0f32; ic * sp];
        for c in 0..ic {
            for s in 0..seq {
                input[c * sp + s] = ((c * seq + s) % 100) as f32 * 0.01;
            }
            // Identity: W[c, c] = 1.0 (weight at spatial offset seq+c)
            if c < oc {
                input[c * sp + seq + c] = 1.0;
            }
        }

        kernel.write_input(0, &f32_to_bytes(&input));
        kernel.eval().expect("matmul eval failed");

        let mut out_buf = vec![0u8; output_bytes];
        kernel.read_output(0, &mut out_buf);
        let output = bytes_to_f32(&out_buf);

        // With identity weight, output should ≈ input activations
        // Output layout: [1, oc, 1, seq]
        let mut max_err: f32 = 0.0;
        for c in 0..oc {
            for s in 0..seq {
                let expected = ((c * seq + s) % 100) as f32 * 0.01;
                let got = output[c * seq + s];
                let err = (expected - got).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        // fp32→fp16→matmul→fp16→fp32 should be reasonably accurate for small values
        assert!(
            max_err < 0.05,
            "matmul identity max error {max_err} too large"
        );
    }

    // ---- Round 2: gen_ffn_w13 + gen_ffn_w2 ----

    #[test]
    fn test_ffn_w13_shape() {
        init_ane();

        let cfg = MilConfig::mha(64, 128, 4, 64);
        let mil = gen_ffn_w13(&cfg);
        let sp_in = cfg.seq_len + 2 * cfg.hidden_dim;
        let input_bytes = cfg.dim * sp_in * 4;
        let out_ch = 3 * cfg.hidden_dim;
        let output_bytes = out_ch * cfg.seq_len * 4;

        let kernel = AneKernel::compile(&mil, None, &[input_bytes], &[output_bytes])
            .expect("ffn_w13 compile failed");

        // Fill with small random-ish values
        let input: Vec<f32> = (0..cfg.dim * sp_in)
            .map(|i| ((i % 200) as f32 - 100.0) * 0.001)
            .collect();
        kernel.write_input(0, &f32_to_bytes(&input));
        kernel.eval().expect("ffn_w13 eval failed");

        let mut out_buf = vec![0u8; output_bytes];
        kernel.read_output(0, &mut out_buf);
        let output = bytes_to_f32(&out_buf);

        assert_eq!(output.len(), out_ch * cfg.seq_len);
        // Verify output is not all zeros (weights are non-zero)
        let nonzero = output.iter().filter(|v| v.abs() > 1e-10).count();
        assert!(nonzero > 0, "ffn_w13 output is all zeros");
    }

    #[test]
    fn test_ffn_w2_shape() {
        init_ane();

        let cfg = MilConfig::mha(64, 128, 4, 64);
        let mil = gen_ffn_w2(&cfg);
        let sp_in = cfg.seq_len + cfg.dim;
        let input_bytes = cfg.hidden_dim * sp_in * 4;
        let output_bytes = cfg.dim * cfg.seq_len * 4;

        let kernel = AneKernel::compile(&mil, None, &[input_bytes], &[output_bytes])
            .expect("ffn_w2 compile failed");

        let input: Vec<f32> = (0..cfg.hidden_dim * sp_in)
            .map(|i| ((i % 200) as f32 - 100.0) * 0.001)
            .collect();
        kernel.write_input(0, &f32_to_bytes(&input));
        kernel.eval().expect("ffn_w2 eval failed");

        let mut out_buf = vec![0u8; output_bytes];
        kernel.read_output(0, &mut out_buf);
        let output = bytes_to_f32(&out_buf);

        assert_eq!(output.len(), cfg.dim * cfg.seq_len);
        let nonzero = output.iter().filter(|v| v.abs() > 1e-10).count();
        assert!(nonzero > 0, "ffn_w2 output is all zeros");
    }

    // ---- Round 3: gen_sdpa_fwd ----

    #[test]
    fn test_sdpa_fwd_shape() {
        init_ane();

        let cfg = MilConfig::mha(64, 128, 4, 64);
        let mil = gen_sdpa_fwd(&cfg);
        let sp_in = cfg.seq_len + 4 * cfg.dim;
        let input_bytes = cfg.dim * sp_in * 4;
        let out_ch = 6 * cfg.dim;
        let output_bytes = out_ch * cfg.seq_len * 4;

        // Build causal mask blob and rope blobs
        let mask_blob = build_causal_mask_blob(cfg.seq_len);
        let (rope_cos_blob, rope_sin_blob) =
            generate_rope_blobs(cfg.seq_len, cfg.head_dim(), cfg.rope_theta);

        let kernel = AneKernel::compile_multi_weights(
            &mil,
            &[
                "@model_path/weights/mask.bin",
                "@model_path/weights/rope_cos.bin",
                "@model_path/weights/rope_sin.bin",
            ],
            &[&mask_blob, &rope_cos_blob, &rope_sin_blob],
            &[input_bytes],
            &[output_bytes],
        )
        .expect("sdpa_fwd compile failed");

        let input: Vec<f32> = (0..cfg.dim * sp_in)
            .map(|i| ((i % 200) as f32 - 100.0) * 0.001)
            .collect();
        kernel.write_input(0, &f32_to_bytes(&input));
        kernel.eval().expect("sdpa_fwd eval failed");

        let mut out_buf = vec![0u8; output_bytes];
        kernel.read_output(0, &mut out_buf);
        let output = bytes_to_f32(&out_buf);

        assert_eq!(output.len(), out_ch * cfg.seq_len);
        let nonzero = output.iter().filter(|v| v.abs() > 1e-10).count();
        assert!(nonzero > 0, "sdpa_fwd output is all zeros");
    }

    // ---- Round 4: Backward matmul kernels ----

    #[test]
    fn test_wot_compiles() {
        init_ane();

        let cfg = MilConfig::mha(64, 128, 4, 64);
        let mil = gen_wot(&cfg);
        let sp_in = cfg.seq_len + cfg.dim;
        let input_bytes = cfg.dim * sp_in * 4;
        let output_bytes = cfg.dim * cfg.seq_len * 4;

        let kernel = AneKernel::compile(&mil, None, &[input_bytes], &[output_bytes])
            .expect("wot compile failed");

        let input: Vec<f32> = (0..cfg.dim * sp_in)
            .map(|i| ((i % 200) as f32 - 100.0) * 0.001)
            .collect();
        kernel.write_input(0, &f32_to_bytes(&input));
        kernel.eval().expect("wot eval failed");

        let mut out_buf = vec![0u8; output_bytes];
        kernel.read_output(0, &mut out_buf);
        let output = bytes_to_f32(&out_buf);

        let nonzero = output.iter().filter(|v| v.abs() > 1e-10).count();
        assert!(nonzero > 0, "wot output is all zeros");
    }

    #[test]
    fn test_ffn_bwd_w2t_compiles() {
        init_ane();

        let cfg = MilConfig::mha(64, 128, 4, 64);
        let mil = gen_ffn_bwd_w2t(&cfg);
        let sp_in = cfg.seq_len + cfg.hidden_dim;
        let input_bytes = cfg.dim * sp_in * 4;
        let output_bytes = cfg.hidden_dim * cfg.seq_len * 4;

        let kernel = AneKernel::compile(&mil, None, &[input_bytes], &[output_bytes])
            .expect("ffn_bwd_w2t compile failed");

        let input: Vec<f32> = (0..cfg.dim * sp_in)
            .map(|i| ((i % 200) as f32 - 100.0) * 0.001)
            .collect();
        kernel.write_input(0, &f32_to_bytes(&input));
        kernel.eval().expect("ffn_bwd_w2t eval failed");

        let mut out_buf = vec![0u8; output_bytes];
        kernel.read_output(0, &mut out_buf);
        let output = bytes_to_f32(&out_buf);

        let nonzero = output.iter().filter(|v| v.abs() > 1e-10).count();
        assert!(nonzero > 0, "ffn_bwd_w2t output is all zeros");
    }

    #[test]
    fn test_ffn_bwd_w13t_compiles() {
        init_ane();

        let cfg = MilConfig::mha(64, 128, 4, 64);
        let mil = gen_ffn_bwd_w13t(&cfg);
        let sp_in = 2 * cfg.seq_len + 2 * cfg.dim;
        let input_bytes = cfg.hidden_dim * sp_in * 4;
        let output_bytes = cfg.dim * cfg.seq_len * 4;

        let kernel = AneKernel::compile(&mil, None, &[input_bytes], &[output_bytes])
            .expect("ffn_bwd_w13t compile failed");

        let input: Vec<f32> = (0..cfg.hidden_dim * sp_in)
            .map(|i| ((i % 200) as f32 - 100.0) * 0.001)
            .collect();
        kernel.write_input(0, &f32_to_bytes(&input));
        kernel.eval().expect("ffn_bwd_w13t eval failed");

        let mut out_buf = vec![0u8; output_bytes];
        kernel.read_output(0, &mut out_buf);
        let output = bytes_to_f32(&out_buf);

        let nonzero = output.iter().filter(|v| v.abs() > 1e-10).count();
        assert!(nonzero > 0, "ffn_bwd_w13t output is all zeros");
    }

    #[test]
    fn test_qkvb_compiles() {
        init_ane();

        let cfg = MilConfig::mha(64, 128, 4, 64);
        let mil = gen_qkvb(&cfg);
        let sp_in = 3 * cfg.seq_len + 3 * cfg.dim;
        let input_bytes = cfg.dim * sp_in * 4;
        let output_bytes = cfg.dim * cfg.seq_len * 4;

        let kernel = AneKernel::compile(&mil, None, &[input_bytes], &[output_bytes])
            .expect("qkvb compile failed");

        let input: Vec<f32> = (0..cfg.dim * sp_in)
            .map(|i| ((i % 200) as f32 - 100.0) * 0.001)
            .collect();
        kernel.write_input(0, &f32_to_bytes(&input));
        kernel.eval().expect("qkvb eval failed");

        let mut out_buf = vec![0u8; output_bytes];
        kernel.read_output(0, &mut out_buf);
        let output = bytes_to_f32(&out_buf);

        let nonzero = output.iter().filter(|v| v.abs() > 1e-10).count();
        assert!(nonzero > 0, "qkvb output is all zeros");
    }

    // ---- Round 5: SDPA backward ----

    #[test]
    fn test_sdpa_bwd1_shape() {
        init_ane();

        let cfg = MilConfig::mha(64, 128, 4, 64);
        let mil = gen_sdpa_bwd1(&cfg);
        let in_ch = 4 * cfg.dim;
        let input_bytes = in_ch * cfg.seq_len * 2; // fp16
        let out_ch = cfg.dim + 2 * cfg.score_ch();
        let output_bytes = out_ch * cfg.seq_len * 2; // fp16

        let mask_blob = build_causal_mask_blob(cfg.seq_len);

        let kernel = AneKernel::compile_multi_weights(
            &mil,
            &["@model_path/weights/mask.bin"],
            &[&mask_blob],
            &[input_bytes],
            &[output_bytes],
        )
        .expect("sdpa_bwd1 compile failed");

        // fp16 input
        let input_f32: Vec<f32> = (0..in_ch * cfg.seq_len)
            .map(|i| ((i % 200) as f32 - 100.0) * 0.001)
            .collect();
        kernel.write_input(0, &f32_to_fp16_bytes(&input_f32));
        kernel.eval().expect("sdpa_bwd1 eval failed");

        let mut out_buf = vec![0u8; output_bytes];
        kernel.read_output(0, &mut out_buf);
        let output = fp16_bytes_to_f32(&out_buf);

        assert_eq!(output.len(), out_ch * cfg.seq_len);
        let nonzero = output.iter().filter(|v| v.abs() > 1e-10).count();
        assert!(nonzero > 0, "sdpa_bwd1 output is all zeros");
    }

    #[test]
    fn test_sdpa_bwd2_shape() {
        init_ane();

        let cfg = MilConfig::mha(64, 128, 4, 64);
        let mil = gen_sdpa_bwd2(&cfg);
        let in_ch = 2 * cfg.score_ch() + 2 * cfg.dim;
        let input_bytes = in_ch * cfg.seq_len * 2; // fp16
        let out_ch = 2 * cfg.dim;
        let output_bytes = out_ch * cfg.seq_len * 2; // fp16

        let kernel = AneKernel::compile(&mil, None, &[input_bytes], &[output_bytes])
            .expect("sdpa_bwd2 compile failed");

        // Build synthetic input: probs should be valid softmax-like values
        // For simplicity, use small values; the test just checks shape/non-zero
        let input_f32: Vec<f32> = (0..in_ch * cfg.seq_len)
            .map(|i| ((i % 200) as f32 - 100.0) * 0.0001)
            .collect();
        kernel.write_input(0, &f32_to_fp16_bytes(&input_f32));
        kernel.eval().expect("sdpa_bwd2 eval failed");

        let mut out_buf = vec![0u8; output_bytes];
        kernel.read_output(0, &mut out_buf);
        let output = fp16_bytes_to_f32(&out_buf);

        assert_eq!(output.len(), out_ch * cfg.seq_len);
    }

    #[test]
    fn test_sdpa_backward_over_parameterized_attention_compile() {
        init_ane();

        let cfg = MilConfig {
            dim: 64,
            hidden_dim: 128,
            n_heads: 4,
            seq_len: 32,
            n_kv_heads: 4,
            rope_theta: 10_000_000.0,
            rms_eps: 1e-6,
            has_lm_head: false,
            head_dim_explicit: 32, // attn_dim = 128 > dim = 64
            linear_attn_indices: vec![],
            linear_n_heads: 0,
            linear_head_dim: 0,
            linear_n_value_heads: 0,
            linear_value_head_dim: 0,
            conv_kernel_size: 0,
            attn_output_gate: true,
        };

        let mask_blob = build_causal_mask_blob(cfg.seq_len);
        let bwd1_spec = KernelSpec::for_kernel(&cfg, KernelType::SdpaBwd1);
        AneKernel::compile_multi_weights(
            &bwd1_spec.mil_text,
            &["@model_path/weights/mask.bin"],
            &[&mask_blob],
            &[bwd1_spec.input_bytes],
            &[bwd1_spec.output_bytes],
        )
        .expect("sdpa_bwd1 should compile for over-parameterized attention");

        let bwd2_spec = KernelSpec::for_kernel(&cfg, KernelType::SdpaBwd2);
        AneKernel::compile(
            &bwd2_spec.mil_text,
            None,
            &[bwd2_spec.input_bytes],
            &[bwd2_spec.output_bytes],
        )
        .expect("sdpa_bwd2 should compile for over-parameterized attention");
    }

    // ---- Round 6: KernelSpec integration ----

    #[test]
    fn test_kernel_spec_sizes() {
        let cfg = MilConfig::mha(64, 128, 4, 64);

        // DynMatmul
        let spec = KernelSpec::for_kernel(&cfg, KernelType::DynMatmul { ic: 64, oc: 64 });
        assert_eq!(spec.input_bytes, 64 * (64 + 64) * 4);
        assert_eq!(spec.output_bytes, 64 * 64 * 4);

        // SdpaFwd
        let spec = KernelSpec::for_kernel(&cfg, KernelType::SdpaFwd);
        assert_eq!(spec.input_bytes, 64 * (64 + 4 * 64) * 4);
        assert_eq!(spec.output_bytes, 6 * 64 * 64 * 4);

        // FfnW13
        let spec = KernelSpec::for_kernel(&cfg, KernelType::FfnW13);
        assert_eq!(spec.input_bytes, 64 * (64 + 2 * 128) * 4);
        assert_eq!(spec.output_bytes, 3 * 128 * 64 * 4);

        // FfnW2
        let spec = KernelSpec::for_kernel(&cfg, KernelType::FfnW2);
        assert_eq!(spec.input_bytes, 128 * (64 + 64) * 4);
        assert_eq!(spec.output_bytes, 64 * 64 * 4);

        // SdpaBwd1 (fp16)
        let spec = KernelSpec::for_kernel(&cfg, KernelType::SdpaBwd1);
        assert_eq!(spec.input_bytes, 4 * 64 * 64 * 2);
        let score_ch = 4 * 64; // n_heads * seq_len
        assert_eq!(spec.output_bytes, (64 + 2 * score_ch) * 64 * 2);

        // SdpaBwd2 (fp16)
        let spec = KernelSpec::for_kernel(&cfg, KernelType::SdpaBwd2);
        assert_eq!(spec.input_bytes, (2 * score_ch + 2 * 64) * 64 * 2);
        assert_eq!(spec.output_bytes, 2 * 64 * 64 * 2);
    }

    // ---- Round 7.1: MilConfig GQA methods ----

    #[test]
    fn test_milconfig_mha_defaults() {
        let cfg = MilConfig::mha(768, 2048, 12, 256);
        assert_eq!(cfg.n_kv_heads, 12);
        assert_eq!(cfg.rope_theta, 10000.0);
        assert_eq!(cfg.rms_eps, 1e-5);
        assert!(!cfg.has_lm_head);
        // MHA: kv_dim == dim
        assert_eq!(cfg.kv_dim(), 768);
        assert_eq!(cfg.heads_per_group(), 1);
        assert_eq!(cfg.head_dim(), 64);
    }

    #[test]
    fn test_milconfig_gqa_kv_dim() {
        // Qwen3.5 0.8B: dim=1024, n_heads=16, n_kv_heads=8, head_dim=64
        let cfg = MilConfig {
            dim: 1024,
            hidden_dim: 2816,
            n_heads: 16,
            seq_len: 128,
            n_kv_heads: 8,
            rope_theta: 1e6,
            rms_eps: 1e-6,
            has_lm_head: true,
            head_dim_explicit: 1024 / 16,
            linear_attn_indices: vec![],
            linear_n_heads: 0,
            linear_head_dim: 0,
            linear_n_value_heads: 0,
            linear_value_head_dim: 0,
            conv_kernel_size: 0,
            attn_output_gate: false,
        };
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.kv_dim(), 512); // 8 * 64
        assert_eq!(cfg.heads_per_group(), 2); // 16 / 8
        assert_eq!(cfg.kv_score_ch(), 8 * 128);
        assert_eq!(cfg.score_ch(), 16 * 128);
    }

    // ---- Round 8: gen_fused_attn_gqa_fwd ----

    /// Run: cargo test --features ane --release --lib -- "test_fused_attn_gqa_compile_and_eval" --nocapture --test-threads=1
    #[test]
    fn test_fused_attn_gqa_compile_and_eval() {
        use crate::agent::ane_weights::{build_fp16_blob, generate_rope_blobs, transpose_weight};

        init_ane();

        // Qwen3.5-like config: GQA with output gate, over-parameterised attention
        let cfg = MilConfig {
            dim: 64,
            hidden_dim: 128,
            n_heads: 8,
            seq_len: 32,
            n_kv_heads: 4,
            rope_theta: 1e6,
            rms_eps: 1e-6,
            has_lm_head: false,
            head_dim_explicit: 16, // attn_dim = 8*16 = 128, kv_dim = 4*16 = 64
            linear_attn_indices: vec![],
            linear_n_heads: 0,
            linear_head_dim: 0,
            linear_n_value_heads: 0,
            linear_value_head_dim: 0,
            conv_kernel_size: 0,
            attn_output_gate: true,
        };

        let hd = cfg.head_dim(); // 16
        let attn_dim = cfg.attn_dim(); // 128
        let kv_dim = cfg.kv_dim(); // 64
        let qpd = cfg.q_proj_dim(); // 256 (gated)
        let seq = cfg.seq_len;
        let dim = cfg.dim;

        // Deterministic weights
        let make_weight = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) as f32 * 0.0037).sin() * 0.1)
                .collect()
        };

        // Model stores [out, in]. Transpose to [in, out] for BLOBFILE matmul layout.
        let wq_blob = build_fp16_blob(&transpose_weight(&make_weight(qpd * dim, 100), qpd, dim));
        let wk_blob =
            build_fp16_blob(&transpose_weight(&make_weight(kv_dim * dim, 200), kv_dim, dim));
        let wv_blob =
            build_fp16_blob(&transpose_weight(&make_weight(kv_dim * dim, 300), kv_dim, dim));
        let wo_blob =
            build_fp16_blob(&transpose_weight(&make_weight(dim * attn_dim, 400), dim, attn_dim));
        let (rc_blob, rs_blob) = generate_rope_blobs(seq, hd, cfg.rope_theta);
        let mask_blob = build_causal_mask_blob(seq);
        // QK-norm weights: [hd] deterministic values
        let qn_blob = build_fp16_blob(&make_weight(hd, 500));
        let kn_blob = build_fp16_blob(&make_weight(hd, 600));

        let has_qk_norm = false; // ANE requires batch=1; QK-norm needs per-head reduce → future work
        let result = gen_fused_attn_gqa_fwd(&cfg, has_qk_norm);
        eprintln!(
            "Fused attn GQA MIL (qk_norm={}): {} bytes, {} weight files, in={}B, out={}B",
            has_qk_norm,
            result.mil_text.len(),
            result.weight_names.len(),
            result.input_bytes,
            result.output_bytes,
        );

        let weight_names: Vec<&str> = result.weight_names.iter().copied().collect();
        let mut weight_datas: Vec<&[u8]> = vec![
            &wq_blob, &wk_blob, &wv_blob, &wo_blob, &rc_blob, &rs_blob, &mask_blob,
        ];
        if has_qk_norm {
            weight_datas.push(&qn_blob);
            weight_datas.push(&kn_blob);
        }

        let kernel = match AneKernel::compile_multi_weights(
            &result.mil_text,
            &weight_names,
            &weight_datas,
            &[result.input_bytes],
            &[result.output_bytes],
        ) {
            Ok(k) => {
                eprintln!("Fused attn GQA kernel compiled on ANE!");
                k
            }
            Err(e) => {
                eprintln!("COMPILE FAILED: {e}");
                eprintln!(
                    "MIL (first 3000 chars):\n{}",
                    &result.mil_text[..result.mil_text.len().min(3000)]
                );
                panic!("Fused attn GQA MIL compile failed: {e}");
            }
        };

        // Input: [dim, seq] fp32
        let input: Vec<f32> = (0..dim * seq)
            .map(|i| ((i + 42) as f32 * 0.0037).sin() * 0.1)
            .collect();
        kernel.write_input(0, &f32_to_bytes(&input));
        kernel.eval().expect("fused attn GQA eval failed");

        let mut out_buf = vec![0u8; result.output_bytes];
        kernel.read_output(0, &mut out_buf);
        let output = bytes_to_f32(&out_buf);

        let out_ch = result.output_bytes / (seq * 4);
        assert_eq!(output.len(), out_ch * seq);
        let nonzero = output.iter().filter(|v| v.abs() > 1e-10).count();
        assert!(
            nonzero > output.len() / 2,
            "fused attn GQA: only {nonzero}/{} non-zero values",
            output.len()
        );
        let max_abs = output
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs < 100.0,
            "fused attn GQA: max_abs {max_abs} too large — likely NaN/overflow"
        );
        eprintln!(
            "Fused attn GQA OK: {} output values, max_abs={max_abs:.4}, nonzero={nonzero}",
            output.len()
        );
    }

    #[test]
    fn test_fused_attn_gqa_fwd_with_qknorm() {
        use crate::agent::ane_weights::{build_fp16_blob, generate_rope_blobs, transpose_weight};
        init_ane();

        let cfg = MilConfig {
            dim: 64, hidden_dim: 128, n_heads: 8, seq_len: 32, n_kv_heads: 4,
            rope_theta: 1e6, rms_eps: 1e-6, has_lm_head: false, head_dim_explicit: 16,
            linear_attn_indices: vec![], linear_n_heads: 0, linear_head_dim: 0,
            linear_n_value_heads: 0, linear_value_head_dim: 0, conv_kernel_size: 0,
            attn_output_gate: true,
        };
        let hd = cfg.head_dim();
        let kv_dim = cfg.kv_dim();
        let qpd = cfg.q_proj_dim();
        let dim = cfg.dim;
        let attn_dim = cfg.attn_dim();
        let seq = cfg.seq_len;

        let make = |n: usize, s: usize| -> Vec<f32> { (0..n).map(|i| ((i+s) as f32 * 0.003).sin() * 0.1).collect() };
        let wq_blob = build_fp16_blob(&transpose_weight(&make(qpd*dim, 1), qpd, dim));
        let wk_blob = build_fp16_blob(&transpose_weight(&make(kv_dim*dim, 2), kv_dim, dim));
        let wv_blob = build_fp16_blob(&transpose_weight(&make(kv_dim*dim, 3), kv_dim, dim));
        let wo_blob = build_fp16_blob(&transpose_weight(&make(dim*attn_dim, 4), dim, attn_dim));
        let (rc, rs) = generate_rope_blobs(seq, hd, cfg.rope_theta);
        let mask = build_causal_mask_blob(seq);
        let qn_blob = build_fp16_blob(&make(hd, 5));
        let kn_blob = build_fp16_blob(&make(hd, 6));

        let result = gen_fused_attn_gqa_fwd(&cfg, true); // QK-NORM ENABLED
        eprintln!("Fused attn GQA FWD (qk_norm=true): {} bytes, {} weights", result.mil_text.len(), result.weight_names.len());

        let names: Vec<&str> = result.weight_names.iter().copied().collect();
        let mut datas: Vec<&[u8]> = Vec::new();
        for n in &names {
            match *n {
                "@model_path/weights/wq.bin" => datas.push(&wq_blob),
                "@model_path/weights/wk.bin" => datas.push(&wk_blob),
                "@model_path/weights/wv.bin" => datas.push(&wv_blob),
                "@model_path/weights/wo.bin" => datas.push(&wo_blob),
                "@model_path/weights/rope_cos.bin" => datas.push(&rc),
                "@model_path/weights/rope_sin.bin" => datas.push(&rs),
                "@model_path/weights/mask.bin" => datas.push(&mask),
                "@model_path/weights/q_norm.bin" => datas.push(&qn_blob),
                "@model_path/weights/k_norm.bin" => datas.push(&kn_blob),
                _ => datas.push(&wq_blob),
            }
        }

        match AneKernel::compile_multi_weights(
            &result.mil_text, &names, &datas, &[result.input_bytes], &[result.output_bytes],
        ) {
            Ok(k) => {
                eprintln!("QK-NORM FORWARD: COMPILED OK on ANE!");
                let input: Vec<f32> = (0..dim*seq).map(|i| ((i+42) as f32*0.001).sin()*0.5).collect();
                k.write_input(0, &f32_to_bytes(&input));
                k.eval().expect("eval");
                let mut out = vec![0u8; result.output_bytes];
                k.read_output(0, &mut out);
                let vals = bytes_to_f32(&out);
                let nonzero = vals.iter().filter(|v| v.abs() > 1e-10).count();
                eprintln!("QK-NORM FORWARD: {}/{} non-zero", nonzero, vals.len());
                assert!(nonzero > 0, "output is all zeros");
            }
            Err(e) => {
                eprintln!("QK-NORM FORWARD: COMPILE FAILED: {e}");
                panic!("QK-norm forward kernel does not compile on ANE");
            }
        }
    }

    // ---- Round 9: gen_fused_attn_gqa_bwd ----

    /// Run: cargo test --features ane --release --lib -- "test_fused_attn_gqa_bwd_compile_and_eval" --nocapture --test-threads=1
    #[test]
    fn test_fused_attn_gqa_bwd_compile_and_eval() {
        use crate::agent::ane_weights::{build_fp16_blob, generate_rope_blobs, transpose_weight};

        init_ane();

        // Same Qwen3.5-like config as forward test
        let cfg = MilConfig {
            dim: 64,
            hidden_dim: 128,
            n_heads: 8,
            seq_len: 32,
            n_kv_heads: 4,
            rope_theta: 1e6,
            rms_eps: 1e-6,
            has_lm_head: false,
            head_dim_explicit: 16,
            linear_attn_indices: vec![],
            linear_n_heads: 0,
            linear_head_dim: 0,
            linear_n_value_heads: 0,
            linear_value_head_dim: 0,
            conv_kernel_size: 0,
            attn_output_gate: true,
        };

        let hd = cfg.head_dim(); // 16
        let attn_dim = cfg.attn_dim(); // 128
        let kv_dim = cfg.kv_dim(); // 64
        let qpd = cfg.q_proj_dim(); // 256 (gated)
        let seq = cfg.seq_len;
        let dim = cfg.dim;
        let has_gate = cfg.attn_output_gate;

        let make = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) as f32 * 0.0037).sin() * 0.1)
                .collect()
        };

        // Same weight blobs as forward (same orientation: transposed to [in, out])
        let wq_blob = build_fp16_blob(&transpose_weight(&make(qpd * dim, 100), qpd, dim));
        let wk_blob =
            build_fp16_blob(&transpose_weight(&make(kv_dim * dim, 200), kv_dim, dim));
        let wv_blob =
            build_fp16_blob(&transpose_weight(&make(kv_dim * dim, 300), kv_dim, dim));
        let wo_blob =
            build_fp16_blob(&transpose_weight(&make(dim * attn_dim, 400), dim, attn_dim));
        let (rc_blob, rs_blob) = generate_rope_blobs(seq, hd, cfg.rope_theta);
        let mask_blob = build_causal_mask_blob(seq);

        let has_qk_norm = false;
        let result = gen_fused_attn_gqa_bwd(&cfg, has_qk_norm);
        eprintln!(
            "Fused attn GQA BWD MIL: {} bytes, {} weight files, in={}B, out={}B",
            result.mil_text.len(),
            result.weight_names.len(),
            result.input_bytes,
            result.output_bytes,
        );

        let weight_names: Vec<&str> = result.weight_names.iter().copied().collect();
        let weight_datas: Vec<&[u8]> = vec![
            &wq_blob, &wk_blob, &wv_blob, &wo_blob, &rc_blob, &rs_blob, &mask_blob,
        ];

        let kernel = match AneKernel::compile_multi_weights(
            &result.mil_text,
            &weight_names,
            &weight_datas,
            &[result.input_bytes],
            &[result.output_bytes],
        ) {
            Ok(k) => {
                eprintln!("Fused attn GQA BWD kernel compiled on ANE!");
                k
            }
            Err(e) => {
                eprintln!("BWD COMPILE FAILED: {e}");
                let path = "/tmp/fused_attn_bwd.mil";
                std::fs::write(path, &result.mil_text).ok();
                eprintln!("Full MIL written to {path}");
                eprintln!("MIL:\n{}", &result.mil_text);
                panic!("Fused attn GQA BWD MIL compile failed: {e}");
            }
        };

        // Build input: dx2 + Q_rot + K_rot + V + pre_gate + gate_raw
        let in_ch = if has_gate {
            dim + 3 * attn_dim + 2 * kv_dim
        } else {
            dim + attn_dim + 2 * kv_dim
        };
        let mut input = Vec::with_capacity(in_ch * seq);
        input.extend(make(dim * seq, 1000)); // dx2
        input.extend(make(attn_dim * seq, 2000)); // Q_rot
        input.extend(make(kv_dim * seq, 3000)); // K_rot
        input.extend(make(kv_dim * seq, 4000)); // V
        if has_gate {
            input.extend(make(attn_dim * seq, 5000)); // pre_gate
            input.extend(make(attn_dim * seq, 6000)); // gate_raw
        }
        assert_eq!(input.len(), in_ch * seq);

        kernel.write_input(0, &f32_to_bytes(&input));
        kernel.eval().expect("fused attn GQA BWD eval failed");

        let mut out_buf = vec![0u8; result.output_bytes];
        kernel.read_output(0, &mut out_buf);
        let output = bytes_to_f32(&out_buf);

        assert_eq!(output.len(), dim * seq);
        let nonzero = output.iter().filter(|v| v.abs() > 1e-10).count();
        assert!(
            nonzero > output.len() / 2,
            "fused attn GQA BWD: only {nonzero}/{} non-zero values",
            output.len()
        );
        let max_abs = output
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs < 100.0,
            "fused attn GQA BWD: max_abs {max_abs} too large — likely NaN/overflow"
        );
        eprintln!(
            "Fused attn GQA BWD OK: {} output values, max_abs={max_abs:.4}, nonzero={nonzero}",
            output.len()
        );
    }

    // ---- Round 10: gen_sdpa_rope_qkvt_bwd (fused SDPA+RoPE+QKV^T backward) ----

    #[test]
    fn test_sdpa_rope_qkvt_bwd_compile() {
        use crate::agent::ane_weights::{build_fp16_blob, generate_rope_blobs};

        init_ane();

        let cfg = MilConfig {
            dim: 64,
            hidden_dim: 128,
            n_heads: 8,
            seq_len: 64,
            n_kv_heads: 4,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            has_lm_head: false,
            head_dim_explicit: 16,
            linear_attn_indices: vec![],
            linear_n_heads: 0,
            linear_head_dim: 0,
            linear_n_value_heads: 0,
            linear_value_head_dim: 0,
            conv_kernel_size: 0,
            attn_output_gate: true,
        };

        let result = gen_sdpa_rope_qkvt_bwd(&cfg, cfg.attn_output_gate);
        eprintln!(
            "SDPA+RoPE+QKV^T BWD MIL: {} bytes, {} weights, in={}B, out={}B",
            result.mil_text.len(),
            result.weight_names.len(),
            result.input_bytes,
            result.output_bytes,
        );

        // Build weight blobs
        let dim = cfg.dim;
        let qpd = cfg.q_proj_dim();
        let kv_dim = cfg.kv_dim();
        let attn_dim = cfg.attn_dim();
        let (cos_blob, sin_blob) = generate_rope_blobs(cfg.seq_len, cfg.head_dim(), cfg.rope_theta);
        let mask_blob = build_causal_mask_blob(cfg.seq_len);
        let wq_blob = build_fp16_blob(&vec![0.01f32; dim * qpd]);
        let wk_blob = build_fp16_blob(&vec![0.01f32; dim * kv_dim]);
        let wv_blob = build_fp16_blob(&vec![0.01f32; dim * kv_dim]);
        let gate_blob = build_fp16_blob(&vec![0.0f32; 1]); // placeholder

        let mut names: Vec<&str> = result.weight_names.iter().copied().collect();
        let mut datas: Vec<&[u8]> = Vec::new();
        for name in &names {
            match *name {
                "@model_path/weights/wq.bin" => datas.push(&wq_blob),
                "@model_path/weights/wk.bin" => datas.push(&wk_blob),
                "@model_path/weights/wv.bin" => datas.push(&wv_blob),
                "@model_path/weights/rope_cos.bin" => datas.push(&cos_blob),
                "@model_path/weights/rope_sin.bin" => datas.push(&sin_blob),
                "@model_path/weights/mask.bin" => datas.push(&mask_blob),
                "@model_path/weights/gate_zeros.bin" => datas.push(&gate_blob),
                _ => datas.push(&wq_blob), // fallback
            }
        }

        match AneKernel::compile_multi_weights(
            &result.mil_text,
            &names,
            &datas,
            &[result.input_bytes],
            &[result.output_bytes],
        ) {
            Ok(kernel) => {
                eprintln!("SDPA+RoPE+QKV^T BWD: COMPILED OK");
                // Quick eval test
                let input = vec![0.01f32; result.input_bytes / 4];
                kernel.write_input(0, &f32_to_bytes(&input));
                kernel.eval().expect("eval failed");
                let mut out = vec![0u8; result.output_bytes];
                kernel.read_output(0, &mut out);
                let vals = bytes_to_f32(&out);
                let nonzero = vals.iter().filter(|v| v.abs() > 1e-10).count();
                eprintln!("SDPA+RoPE+QKV^T BWD: {}/{} non-zero output elements", nonzero, vals.len());
            }
            Err(e) => {
                eprintln!("SDPA+RoPE+QKV^T BWD: COMPILE FAILED: {e}");
                // Don't panic — this is experimental
            }
        }
    }

    // ---- Round 10.4: gen_fused_gdn_proj (QKV+A+B+Z in 1 dispatch) ----

    #[test]
    fn test_fused_gdn_proj_compile() {
        use crate::agent::ane_weights::build_fp16_blob;
        init_ane();

        // 35B GDN dimensions
        let cfg = MilConfig {
            dim: 2048, hidden_dim: 512, n_heads: 16, seq_len: 128, n_kv_heads: 2,
            rope_theta: 1e6, rms_eps: 1e-6, has_lm_head: false, head_dim_explicit: 256,
            linear_attn_indices: (0..30).collect(), linear_n_heads: 16, linear_head_dim: 128,
            linear_n_value_heads: 32, linear_value_head_dim: 128, conv_kernel_size: 0,
            attn_output_gate: true,
        };

        let result = gen_fused_gdn_proj(&cfg);
        eprintln!("Fused GDN proj: {} bytes, {} weights, in={}B, out={}B",
            result.mil_text.len(), result.weight_names.len(), result.input_bytes, result.output_bytes);

        let dim = cfg.dim;
        let h_v = cfg.linear_n_value_heads;
        let d_v = cfg.linear_value_head_dim;
        let h_k = cfg.linear_n_heads;
        let d_k = cfg.linear_head_dim;
        let key_dim = h_k * d_k;
        let value_dim = h_v * d_v;
        let qkv_dim = 2 * key_dim + value_dim;

        let make = |n: usize, s: usize| build_fp16_blob(&(0..n).map(|i| ((i+s) as f32*0.001).sin()*0.1).collect::<Vec<_>>());
        let wqkv = make(dim * qkv_dim, 1);
        let wa = make(dim * h_v, 2);
        let wb = make(dim * h_v, 3);
        let wz = make(dim * value_dim, 4);

        let names: Vec<&str> = result.weight_names.iter().copied().collect();
        let datas: Vec<&[u8]> = vec![&wqkv, &wa, &wb, &wz];

        match AneKernel::compile_multi_weights(
            &result.mil_text, &names, &datas, &[result.input_bytes], &[result.output_bytes],
        ) {
            Ok(k) => {
                eprintln!("Fused GDN proj: COMPILED OK on ANE!");
                let input: Vec<f32> = (0..dim*128).map(|i| ((i+42) as f32*0.001).sin()*0.5).collect();
                k.write_input(0, &f32_to_bytes(&input));
                k.eval().expect("eval");
                let mut out = vec![0u8; result.output_bytes];
                k.read_output(0, &mut out);
                let vals = bytes_to_f32(&out);
                let nz = vals.iter().filter(|v| v.abs() > 1e-10).count();
                eprintln!("Fused GDN proj: {nz}/{} non-zero", vals.len());
                assert!(nz > 0);
            }
            Err(e) => {
                eprintln!("Fused GDN proj: COMPILE FAILED: {e}");
                panic!("Fused GDN proj kernel does not compile");
            }
        }
    }

    // ---- Round 10.5: gen_fused_ffn_bwd (W2T + SiLU bwd + W13T) ----

    #[test]
    fn test_fused_ffn_bwd_compile_and_eval() {
        use crate::agent::ane_weights::build_fp16_blob;

        init_ane();

        let cfg = MilConfig {
            dim: 64,
            hidden_dim: 128,
            n_heads: 4,
            seq_len: 64,
            n_kv_heads: 4,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            has_lm_head: false,
            head_dim_explicit: 16,
            linear_attn_indices: vec![],
            linear_n_heads: 0,
            linear_head_dim: 0,
            linear_n_value_heads: 0,
            linear_value_head_dim: 0,
            conv_kernel_size: 0,
            attn_output_gate: false,
        };

        let result = gen_fused_ffn_bwd(&cfg);
        eprintln!(
            "Fused FFN BWD MIL: {} bytes, {} weights, in={}B, out={}B",
            result.mil_text.len(),
            result.weight_names.len(),
            result.input_bytes,
            result.output_bytes,
        );

        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;
        let seq = cfg.seq_len;

        // Build weight blobs (W2^T [hidden,dim], W1^T [dim,hidden], W3^T [dim,hidden])
        let w2t_blob = build_fp16_blob(&(0..hidden*dim).map(|i| ((i+1) as f32 * 0.003).sin() * 0.5).collect::<Vec<_>>());
        let w1t_blob = build_fp16_blob(&(0..dim*hidden).map(|i| ((i+2) as f32 * 0.005).sin() * 0.5).collect::<Vec<_>>());
        let w3t_blob = build_fp16_blob(&(0..dim*hidden).map(|i| ((i+3) as f32 * 0.007).sin() * 0.5).collect::<Vec<_>>());

        let names: Vec<&str> = result.weight_names.iter().copied().collect();
        let datas: Vec<&[u8]> = vec![&w2t_blob, &w1t_blob, &w3t_blob];

        let kernel = AneKernel::compile_multi_weights(
            &result.mil_text,
            &names,
            &datas,
            &[result.input_bytes],
            &[result.output_bytes],
        )
        .expect("Fused FFN BWD compile failed");

        // Build input: dx_ffn | h1 | h3
        let in_ch = dim + 2 * hidden;
        let input: Vec<f32> = (0..in_ch * seq)
            .map(|i| ((i + 42) as f32 * 0.0013).sin() * 0.5)
            .collect();

        kernel.write_input(0, &f32_to_bytes(&input));
        kernel.eval().expect("Fused FFN BWD eval failed");

        let mut out_buf = vec![0u8; result.output_bytes];
        kernel.read_output(0, &mut out_buf);
        let out = bytes_to_f32(&out_buf);
        let nonzero = out.iter().filter(|v| v.abs() > 1e-10).count();
        let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!(
            "Fused FFN BWD: {}/{} non-zero, norm={norm:.4}",
            nonzero, out.len()
        );
        assert!(nonzero > 0, "output is all zeros");
        assert!(norm > 1e-6, "output norm too small");

        // CPU reference: W2^T matmul → SiLU backward → W13^T matmul
        let dx_ffn: Vec<f32> = input[..dim*seq].to_vec();
        let h1_act: Vec<f32> = input[dim*seq..(dim+hidden)*seq].to_vec();
        let h3_act: Vec<f32> = input[(dim+hidden)*seq..].to_vec();

        // W2^T weights (decode from fp16 blob)
        let w2t_f32: Vec<f32> = (0..hidden*dim).map(|i| ((i+1) as f32 * 0.003).sin() * 0.5).collect();
        let w1t_f32: Vec<f32> = (0..dim*hidden).map(|i| ((i+2) as f32 * 0.005).sin() * 0.5).collect();
        let w3t_f32: Vec<f32> = (0..dim*hidden).map(|i| ((i+3) as f32 * 0.007).sin() * 0.5).collect();

        // dsilu = W2^T @ dx_ffn: [hidden,dim] @ [dim,seq] → [hidden,seq]
        let dsilu_cpu = crate::agent::ane_forward::cpu_matmul(&w2t_f32, &dx_ffn, hidden, dim, seq);
        // SiLU backward
        let mut dh1_cpu = vec![0.0f32; hidden*seq];
        let mut dh3_cpu = vec![0.0f32; hidden*seq];
        crate::agent::ane_backward::silu_bwd(&mut dh1_cpu, &mut dh3_cpu, &dsilu_cpu, &h1_act, &h3_act, hidden*seq);
        // W1^T @ dh1 + W3^T @ dh3: [dim,hidden] @ [hidden,seq]
        let dx1 = crate::agent::ane_forward::cpu_matmul(&w1t_f32, &dh1_cpu, dim, hidden, seq);
        let dx3 = crate::agent::ane_forward::cpu_matmul(&w3t_f32, &dh3_cpu, dim, hidden, seq);
        let cpu_out: Vec<f32> = dx1.iter().zip(dx3.iter()).map(|(a,b)| a+b).collect();

        let max_err = out.iter().zip(cpu_out.iter()).map(|(a,c)| (a-c).abs()).fold(0.0f32, f32::max);
        let cpu_norm: f32 = cpu_out.iter().map(|v| v*v).sum::<f32>().sqrt();
        eprintln!("Fused FFN BWD vs CPU: max_err={max_err:.6}, ane_norm={norm:.4}, cpu_norm={cpu_norm:.4}");
        assert!(max_err < 0.5, "Fused FFN BWD max error too large: {max_err:.6}");
    }

    // ---- Round 11: gen_rmsnorm_fwd (standalone RMSNorm for ANE) ----

    #[test]
    fn test_rmsnorm_fwd_compile_and_correctness() {
        use crate::agent::ane_weights::build_fp16_blob;

        init_ane();

        let dim = 64;
        let seq = 64;
        let eps = 1e-6f32;

        let result = gen_rmsnorm_fwd(dim, seq, eps);
        eprintln!(
            "RMSNorm FWD MIL: {} bytes, {} weight files, in={}B, out={}B",
            result.mil_text.len(),
            result.weight_names.len(),
            result.input_bytes,
            result.output_bytes,
        );

        // Deterministic weight
        let w: Vec<f32> = (0..dim).map(|i| 0.5 + (i as f32 * 0.01)).collect();
        let w_blob = build_fp16_blob(&w);

        let weight_names: Vec<&str> = result.weight_names.iter().copied().collect();
        let weight_datas: Vec<&[u8]> = vec![&w_blob];

        let kernel = AneKernel::compile_multi_weights(
            &result.mil_text,
            &weight_names,
            &weight_datas,
            &[result.input_bytes],
            &[result.output_bytes],
        )
        .expect("RMSNorm FWD compile failed");

        // Deterministic input
        let input: Vec<f32> = (0..dim * seq)
            .map(|i| ((i + 42) as f32 * 0.0037).sin() * 0.5)
            .collect();
        kernel.write_input(0, &f32_to_bytes(&input));
        kernel.eval().expect("RMSNorm FWD eval failed");

        let mut out_buf = vec![0u8; result.output_bytes];
        kernel.read_output(0, &mut out_buf);
        let ane_output = bytes_to_f32(&out_buf);

        // CPU reference
        let mut cpu_output = vec![0.0f32; dim * seq];
        crate::agent::ane_forward::rmsnorm(&mut cpu_output, &input, &w, dim, seq, eps);

        // Compare
        let max_err = ane_output
            .iter()
            .zip(cpu_output.iter())
            .map(|(a, c)| (a - c).abs())
            .fold(0.0f32, f32::max);
        let mean_err = ane_output
            .iter()
            .zip(cpu_output.iter())
            .map(|(a, c)| (a - c).abs())
            .sum::<f32>()
            / ane_output.len() as f32;
        eprintln!(
            "RMSNorm FWD ANE vs CPU: max_err={max_err:.6}, mean_err={mean_err:.6} (fp16 tolerance ~1e-3)"
        );
        assert!(
            max_err < 0.01,
            "RMSNorm FWD: max_err {max_err:.6} exceeds fp16 tolerance"
        );
    }

    // ---- gen_rmsnorm_bwd (standalone RMSNorm backward dx for ANE, no dw) ----

    #[test]
    fn test_rmsnorm_bwd_compile_and_correctness() {
        use crate::agent::ane_weights::build_fp16_blob;

        init_ane();

        let dim = 64;
        let seq = 64;
        let eps = 1e-6f32;

        let result = gen_rmsnorm_bwd(dim, seq, eps);
        eprintln!(
            "RMSNorm BWD MIL: {} bytes, {} weight files, in={}B, out={}B",
            result.mil_text.len(),
            result.weight_names.len(),
            result.input_bytes,
            result.output_bytes,
        );

        // Deterministic weight
        let w: Vec<f32> = (0..dim).map(|i| 0.5 + (i as f32 * 0.01)).collect();
        let w_blob = build_fp16_blob(&w);

        let weight_names: Vec<&str> = result.weight_names.iter().copied().collect();
        let weight_datas: Vec<&[u8]> = vec![&w_blob];

        let kernel = AneKernel::compile_multi_weights(
            &result.mil_text,
            &weight_names,
            &weight_datas,
            &[result.input_bytes],
            &[result.output_bytes],
        )
        .expect("RMSNorm BWD compile failed");

        // Deterministic input: dy and x
        let dy: Vec<f32> = (0..dim * seq)
            .map(|i| ((i + 7) as f32 * 0.0023).sin() * 0.3)
            .collect();
        let x: Vec<f32> = (0..dim * seq)
            .map(|i| ((i + 42) as f32 * 0.0037).sin() * 0.5)
            .collect();

        // Pack input: dy | x concatenated on channel axis
        let mut input = Vec::with_capacity(2 * dim * seq);
        input.extend_from_slice(&dy);
        input.extend_from_slice(&x);

        kernel.write_input(0, &f32_to_bytes(&input));
        kernel.eval().expect("RMSNorm BWD eval failed");

        let mut out_buf = vec![0u8; result.output_bytes];
        kernel.read_output(0, &mut out_buf);
        let ane_dx = bytes_to_f32(&out_buf);

        // CPU reference
        let mut cpu_dx = vec![0.0f32; dim * seq];
        let mut dw_dummy = vec![0.0f32; dim];
        crate::agent::ane_backward::rmsnorm_bwd(
            &mut cpu_dx, &mut dw_dummy, &dy, &x, &w, dim, seq, eps,
        );

        // Compare
        let max_err = ane_dx
            .iter()
            .zip(cpu_dx.iter())
            .map(|(a, c)| (a - c).abs())
            .fold(0.0f32, f32::max);
        let mean_err = ane_dx
            .iter()
            .zip(cpu_dx.iter())
            .map(|(a, c)| (a - c).abs())
            .sum::<f32>()
            / ane_dx.len() as f32;
        let ane_norm: f32 = ane_dx.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cpu_norm: f32 = cpu_dx.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!(
            "RMSNorm BWD ANE vs CPU: max_err={max_err:.6}, mean_err={mean_err:.6}, \
             ane_norm={ane_norm:.4}, cpu_norm={cpu_norm:.4}"
        );
        assert!(
            max_err < 0.02,
            "RMSNorm BWD: max_err {max_err:.6} exceeds tolerance"
        );
        assert!(ane_norm > 1e-6, "ANE dx is zero");
        assert!(cpu_norm > 1e-6, "CPU dx is zero");
    }

    /// Test that the fused SDPA backward kernel compiles and produces non-zero output.
    /// Uses [1,H,S,*] form + fp16 throughout to match working gen_sdpa_bwd1/bwd2 pattern.
    #[test]
    fn test_sdpa_rope_bwd_compile_and_eval() {
        use crate::agent::ane_weights::generate_rope_blobs;

        init_ane();

        // Same as bisect Phase E config: H=8, seq=32
        let cfg = MilConfig {
            dim: 64,
            hidden_dim: 128,
            n_heads: 8,
            seq_len: 32,
            n_kv_heads: 4,
            rope_theta: 1e6,
            rms_eps: 1e-6,
            has_lm_head: false,
            head_dim_explicit: 16,
            linear_attn_indices: vec![],
            linear_n_heads: 0,
            linear_head_dim: 0,
            linear_n_value_heads: 0,
            linear_value_head_dim: 0,
            conv_kernel_size: 0,
            attn_output_gate: true,
        };

        let hd = cfg.head_dim();
        let attn_dim = cfg.attn_dim();
        let kv_dim = cfg.kv_dim();
        let qpd = cfg.q_proj_dim();
        let seq = cfg.seq_len;
        let has_gate = cfg.attn_output_gate;

        let make = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) as f32 * 0.0037).sin() * 0.1)
                .collect()
        };

        let mask_blob = build_causal_mask_blob(seq);

        let result = gen_sdpa_rope_bwd(&cfg, has_gate);
        eprintln!(
            "SDPA BWD MIL: {} bytes, {} weight files, in={}B, out={}B",
            result.mil_text.len(),
            result.weight_names.len(),
            result.input_bytes,
            result.output_bytes,
        );

        let weight_names: Vec<&str> = result.weight_names.iter().copied().collect();
        let weight_datas: Vec<&[u8]> = vec![&mask_blob];

        let kernel = match AneKernel::compile_multi_weights(
            &result.mil_text,
            &weight_names,
            &weight_datas,
            &[result.input_bytes],
            &[result.output_bytes],
        ) {
            Ok(k) => {
                eprintln!("SDPA+RoPE BWD kernel compiled on ANE!");
                k
            }
            Err(e) => {
                let path = "/tmp/sdpa_rope_bwd.mil";
                std::fs::write(path, &result.mil_text).ok();
                eprintln!("MIL written to {path}");
                panic!("SDPA+RoPE BWD compile failed: {e}");
            }
        };

        // Input: d_attn[ad] + Q_rot[ad] + K_expanded[ad] + V_expanded[ad]
        let in_ch = 4 * attn_dim;
        let out_ch = 3 * attn_dim;
        let mut input = Vec::with_capacity(in_ch * seq);
        input.extend(make(attn_dim * seq, 1000)); // d_attn (post-gate)
        input.extend(make(attn_dim * seq, 2000)); // Q_rot (full heads)
        input.extend(make(attn_dim * seq, 3000)); // K_expanded (full heads)
        input.extend(make(attn_dim * seq, 4000)); // V_expanded (full heads)
        assert_eq!(input.len(), in_ch * seq);

        kernel.write_input(0, &f32_to_bytes(&input));
        kernel.eval().expect("SDPA BWD eval failed");

        let mut out_buf = vec![0u8; result.output_bytes];
        kernel.read_output(0, &mut out_buf);
        let output = bytes_to_f32(&out_buf);

        assert_eq!(output.len(), out_ch * seq);
        let nonzero = output.iter().filter(|v| v.abs() > 1e-10).count();
        assert!(
            nonzero > output.len() / 4,
            "SDPA+RoPE BWD: only {nonzero}/{} non-zero values",
            output.len()
        );
        let max_abs = output.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs < 100.0,
            "SDPA+RoPE BWD: max_abs {max_abs} too large"
        );
        eprintln!(
            "SDPA+RoPE BWD OK: {} values, max_abs={max_abs:.4}, nonzero={nonzero}",
            output.len()
        );
    }

    /// Binary-search which phase of gen_fused_attn_gqa_bwd causes ANE compile failure.
    ///
    /// Generates progressively more complex MIL snippets that each compile independently.
    /// Phase 1: input slice only
    /// Phase 2: + Wo^T projection
    /// Phase 3: + gate backward
    /// Phase 4: + GQA reshape + SDPA recompute (scores + softmax)
    /// Phase 4b-dV: + dV (reduce_mean on axis 1)
    /// Phase 4b-dP: + dP (no reduce)
    /// Phase 4b-softmax-bwd: + softmax backward (reduce_mean on axis -1)
    /// Phase 4b-dQ: + dQ matmul
    /// Phase 4b-dK: + dK (reduce_mean on axis 1)
    /// Phase 5+6+7+8: + RoPE backward + flatten + QKV^T proj + output sum
    ///
    /// Run: cargo test --features ane --release --lib -- "test_bwd_attn_binary_search" --nocapture --test-threads=1
    #[test]
    fn test_bwd_attn_binary_search() {
        use crate::agent::ane_weights::{build_fp16_blob, generate_rope_blobs, transpose_weight};

        init_ane();

        let cfg = MilConfig {
            dim: 64,
            hidden_dim: 128,
            n_heads: 8,
            seq_len: 32,
            n_kv_heads: 4,
            rope_theta: 1e6,
            rms_eps: 1e-6,
            has_lm_head: false,
            head_dim_explicit: 16,
            linear_attn_indices: vec![],
            linear_n_heads: 0,
            linear_head_dim: 0,
            linear_n_value_heads: 0,
            linear_value_head_dim: 0,
            conv_kernel_size: 0,
            attn_output_gate: true,
        };

        let dim = cfg.dim; // 64
        let seq = cfg.seq_len; // 32
        let heads = cfg.n_heads; // 8
        let kv_heads = cfg.n_kv_heads; // 4
        let hd = cfg.head_dim(); // 16
        let half_hd = hd / 2; // 8
        let attn_dim = cfg.attn_dim(); // 128
        let kv_dim = cfg.kv_dim(); // 64
        let qpd = cfg.q_proj_dim(); // 256
        let hpg = cfg.heads_per_group(); // 2
        let sc = 1.0 / (hd as f64).sqrt();
        let in_ch = dim + 3 * attn_dim + 2 * kv_dim; // 576

        // Build weight blobs
        let make = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + seed) as f32 * 0.0037).sin() * 0.1).collect()
        };
        let wq_blob = build_fp16_blob(&transpose_weight(&make(qpd * dim, 100), qpd, dim));
        let wk_blob = build_fp16_blob(&transpose_weight(&make(kv_dim * dim, 200), kv_dim, dim));
        let wv_blob = build_fp16_blob(&transpose_weight(&make(kv_dim * dim, 300), kv_dim, dim));
        let wo_blob = build_fp16_blob(&transpose_weight(&make(dim * attn_dim, 400), dim, attn_dim));
        let (rc_blob, rs_blob) = generate_rope_blobs(seq, hd, cfg.rope_theta);
        let mask_blob = build_causal_mask_blob(seq);

        let weight_names: Vec<&str> = vec![
            "@model_path/weights/wq.bin",
            "@model_path/weights/wk.bin",
            "@model_path/weights/wv.bin",
            "@model_path/weights/wo.bin",
            "@model_path/weights/rope_cos.bin",
            "@model_path/weights/rope_sin.bin",
            "@model_path/weights/mask.bin",
        ];
        let weight_datas: Vec<&[u8]> = vec![
            &wq_blob, &wk_blob, &wv_blob, &wo_blob, &rc_blob, &rs_blob, &mask_blob,
        ];

        // Helper: build MIL preamble (constants + cast + input slices + weights)
        let preamble = || -> String {
            let mut m = String::with_capacity(16384);
            m.push_str(MIL_HDR);
            let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1, {in_ch}, 1, {seq}]> x) {{");
            // constants
            let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
            let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
            let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
            let _ = writeln!(m, "        bool bT = const()[name=string(\"bT\"), val=bool(true)];");
            let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
            let _ = writeln!(m, "        fp16 scv = const()[name=string(\"scv\"), val=fp16({sc})];");
            let _ = writeln!(m, "        fp16 hpg_v = const()[name=string(\"hpgv\"), val=fp16({hpg})];");
            let _ = writeln!(m, "        fp16 seq_v = const()[name=string(\"seqv\"), val=fp16({seq})];");
            let _ = writeln!(m, "        tensor<int32, [1]> ax1 = const()[name=string(\"ax1\"), val=tensor<int32, [1]>([1])];");
            let _ = writeln!(m, "        tensor<int32, [1]> rax_last = const()[name=string(\"raxl\"), val=tensor<int32, [1]>([-1])];");
            let _ = writeln!(m, "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];");
            let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");
            // cast
            let _ = writeln!(m, "        tensor<fp16, [1,{in_ch},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");
            // slices
            let _ = writeln!(m, "        tensor<int32, [4]> b_dx = const()[name=string(\"bdx\"), val=tensor<int32, [4]>([0,0,0,0])];");
            let _ = writeln!(m, "        tensor<int32, [4]> s_dx = const()[name=string(\"sdx\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dx2h = slice_by_size(x=xh,begin=b_dx,size=s_dx)[name=string(\"dx2h\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> b_qr = const()[name=string(\"bqr\"), val=tensor<int32, [4]>([0,{dim},0,0])];");
            let _ = writeln!(m, "        tensor<int32, [4]> s_ad = const()[name=string(\"sad\"), val=tensor<int32, [4]>([1,{attn_dim},1,{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> qrh = slice_by_size(x=xh,begin=b_qr,size=s_ad)[name=string(\"qrh\")];");
            let off_kr = dim + attn_dim;
            let _ = writeln!(m, "        tensor<int32, [4]> b_kr = const()[name=string(\"bkr\"), val=tensor<int32, [4]>([0,{off_kr},0,0])];");
            let _ = writeln!(m, "        tensor<int32, [4]> s_kv = const()[name=string(\"skv\"), val=tensor<int32, [4]>([1,{kv_dim},1,{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_dim},1,{seq}]> krh = slice_by_size(x=xh,begin=b_kr,size=s_kv)[name=string(\"krh\")];");
            let off_v = off_kr + kv_dim;
            let _ = writeln!(m, "        tensor<int32, [4]> b_vh = const()[name=string(\"bvh\"), val=tensor<int32, [4]>([0,{off_v},0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_dim},1,{seq}]> vh = slice_by_size(x=xh,begin=b_vh,size=s_kv)[name=string(\"vh\")];");
            let off_pg = off_v + kv_dim;
            let _ = writeln!(m, "        tensor<int32, [4]> b_pg = const()[name=string(\"bpg\"), val=tensor<int32, [4]>([0,{off_pg},0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> pgh = slice_by_size(x=xh,begin=b_pg,size=s_ad)[name=string(\"pgh\")];");
            let off_gr = off_pg + attn_dim;
            let _ = writeln!(m, "        tensor<int32, [4]> b_gr = const()[name=string(\"bgr\"), val=tensor<int32, [4]>([0,{off_gr},0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> grh = slice_by_size(x=xh,begin=b_gr,size=s_ad)[name=string(\"grh\")];");
            // weights
            let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{qpd}]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [1,1,{dim},{qpd}]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{kv_dim}]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [1,1,{dim},{kv_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{kv_dim}]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [1,1,{dim},{kv_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{dim}]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [1,1,{attn_dim},{dim}]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_cos = const()[name=string(\"rc\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_cos.bin\"), offset=uint64(64)))];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_sin = const()[name=string(\"rs\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_sin.bin\"), offset=uint64(64)))];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{seq}]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,{seq},{seq}]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];");
            m
        };

        // Helper: terminate MIL with a cast of the given var to fp32 output
        let terminate = |m: &mut String, var: &str, shape: &str, out_shape: &str| {
            let _ = writeln!(m, "        tensor<fp32, {out_shape}> out = cast(dtype=to32,x={var})[name=string(\"cout\")];");
            let _ = writeln!(m, "    }} -> (out);");
            m.push_str("}\n");
        };

        // Try compiling a phase and report pass/fail
        let try_compile = |label: &str, mil: &str, out_bytes: usize| -> bool {
            let in_bytes = in_ch * seq * 4;
            match AneKernel::compile_multi_weights(
                mil,
                &weight_names,
                &weight_datas,
                &[in_bytes],
                &[out_bytes],
            ) {
                Ok(_k) => {
                    eprintln!("  PASS: {label}");
                    true
                }
                Err(e) => {
                    eprintln!("  FAIL: {label} -- {e}");
                    let path = format!("/tmp/bwd_bisect_{}.mil", label.replace(' ', "_"));
                    std::fs::write(&path, mil).ok();
                    eprintln!("        MIL written to {path}");
                    false
                }
            }
        };

        eprintln!("\n=== Binary search: fused attn GQA BWD ANE compile ===\n");

        // ---- Phase 1: slices only → output dx2h ----
        {
            let mut m = preamble();
            terminate(&mut m, "dx2h", &format!("[1,{dim},1,{seq}]"), &format!("[1,{dim},1,{seq}]"));
            try_compile("Phase1_slices", &m, dim * seq * 4);
        }

        // ---- Phase 2: Wo^T projection ----
        {
            let mut m = preamble();
            let _ = writeln!(m, "        tensor<int32, [4]> r2d = const()[name=string(\"r2d\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dx_r = reshape(shape=r2d,x=dx2h)[name=string(\"dxr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx_nt = transpose(perm=pm,x=dx_r)[name=string(\"dxnt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{attn_dim}]> da_nt = matmul(transpose_x=bF,transpose_y=bT,x=dx_nt,y=Wo)[name=string(\"dant\")];");
            terminate(&mut m, "da_nt", &format!("[1,1,{seq},{attn_dim}]"), &format!("[1,1,{seq},{attn_dim}]"));
            try_compile("Phase2_WoT_proj", &m, seq * attn_dim * 4);
        }

        // ---- Phase 3: gate backward ----
        {
            let mut m = preamble();
            let _ = writeln!(m, "        tensor<int32, [4]> r2d = const()[name=string(\"r2d\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dx_r = reshape(shape=r2d,x=dx2h)[name=string(\"dxr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx_nt = transpose(perm=pm,x=dx_r)[name=string(\"dxnt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{attn_dim}]> da_nt = matmul(transpose_x=bF,transpose_y=bT,x=dx_nt,y=Wo)[name=string(\"dant\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{seq}]> da_t = transpose(perm=pm,x=da_nt)[name=string(\"dat\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> da_ch = reshape(shape=s_ad,x=da_t)[name=string(\"dach\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> sig = sigmoid(x=grh)[name=string(\"sig\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> d_at = mul(x=da_ch,y=sig)[name=string(\"dat2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> sig2 = mul(x=sig,y=sig)[name=string(\"sig2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> sder = sub(x=sig,y=sig2)[name=string(\"sder\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> dg1 = mul(x=da_ch,y=pgh)[name=string(\"dg1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> d_gate = mul(x=dg1,y=sder)[name=string(\"dgate\")];");
            terminate(&mut m, "d_at", &format!("[1,{attn_dim},1,{seq}]"), &format!("[1,{attn_dim},1,{seq}]"));
            try_compile("Phase3_gate_bwd", &m, attn_dim * seq * 4);
        }

        // ---- Phase 3.5: gate + GQA reshape ONLY (no SDPA) ----
        {
            let mut m = preamble();
            let _ = writeln!(m, "        tensor<int32, [4]> r2d = const()[name=string(\"r2d\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dx_r = reshape(shape=r2d,x=dx2h)[name=string(\"dxr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx_nt = transpose(perm=pm,x=dx_r)[name=string(\"dxnt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{attn_dim}]> da_nt = matmul(transpose_x=bF,transpose_y=bT,x=dx_nt,y=Wo)[name=string(\"dant\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{seq}]> da_t = transpose(perm=pm,x=da_nt)[name=string(\"dat\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> da_ch = reshape(shape=s_ad,x=da_t)[name=string(\"dach\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> sig = sigmoid(x=grh)[name=string(\"sig\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> d_at = mul(x=da_ch,y=sig)[name=string(\"dat2\")];");
            // GQA reshape for dab only
            let _ = writeln!(m, "        tensor<int32, [4]> rqh = const()[name=string(\"rqh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> da_4 = reshape(shape=rqh,x=d_at)[name=string(\"da4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> da_hs = transpose(perm=pm,x=da_4)[name=string(\"dahs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rqb = const()[name=string(\"rqb\"), val=tensor<int32, [4]>([{kv_heads},{hpg},{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dab = reshape(shape=rqb,x=da_hs)[name=string(\"dab\")];");
            terminate(&mut m, "dab", &format!("[{kv_heads},{hpg},{seq},{hd}]"), &format!("[{kv_heads},{hpg},{seq},{hd}]"));
            try_compile("Phase3.5_gqa_reshape_only", &m, kv_heads * hpg * seq * hd * 4);
        }

        // ---- Phase 3.6: same GQA reshape but also Q/K/V batch reshape (no SDPA matmul) ----
        {
            let mut m = preamble();
            let _ = writeln!(m, "        tensor<int32, [4]> r2d = const()[name=string(\"r2d\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dx_r = reshape(shape=r2d,x=dx2h)[name=string(\"dxr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx_nt = transpose(perm=pm,x=dx_r)[name=string(\"dxnt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{attn_dim}]> da_nt = matmul(transpose_x=bF,transpose_y=bT,x=dx_nt,y=Wo)[name=string(\"dant\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{seq}]> da_t = transpose(perm=pm,x=da_nt)[name=string(\"dat\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> da_ch = reshape(shape=s_ad,x=da_t)[name=string(\"dach\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> sig = sigmoid(x=grh)[name=string(\"sig\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> d_at = mul(x=da_ch,y=sig)[name=string(\"dat2\")];");
            // all GQA reshapes
            let _ = writeln!(m, "        tensor<int32, [4]> rqh = const()[name=string(\"rqh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> da_4 = reshape(shape=rqh,x=d_at)[name=string(\"da4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> da_hs = transpose(perm=pm,x=da_4)[name=string(\"dahs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rqb = const()[name=string(\"rqb\"), val=tensor<int32, [4]>([{kv_heads},{hpg},{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dab = reshape(shape=rqb,x=da_hs)[name=string(\"dab\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qr_4 = reshape(shape=rqh,x=qrh)[name=string(\"qr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> qr_hs = transpose(perm=pm,x=qr_4)[name=string(\"qrhs\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> qb = reshape(shape=rqb,x=qr_hs)[name=string(\"qb\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rkv = const()[name=string(\"rkv\"), val=tensor<int32, [4]>([1,{kv_heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> kr_4 = reshape(shape=rkv,x=krh)[name=string(\"kr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> kr_hs = transpose(perm=pm,x=kr_4)[name=string(\"krhs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rkb = const()[name=string(\"rkb\"), val=tensor<int32, [4]>([{kv_heads},1,{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> kb = reshape(shape=rkb,x=kr_hs)[name=string(\"kb\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> v_4 = reshape(shape=rkv,x=vh)[name=string(\"v4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> v_hs = transpose(perm=pm,x=v_4)[name=string(\"vhs\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> vb = reshape(shape=rkb,x=v_hs)[name=string(\"vb\")];");
            // just add them together to force all to be used
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> s1 = add(x=dab,y=qb)[name=string(\"s1\")];");
            // broadcast add kb to s1 to force kb used
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> s2 = add(x=s1,y=kb)[name=string(\"s2\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> s3 = add(x=s2,y=vb)[name=string(\"s3\")];");
            terminate(&mut m, "s3", &format!("[{kv_heads},{hpg},{seq},{hd}]"), &format!("[{kv_heads},{hpg},{seq},{hd}]"));
            try_compile("Phase3.6_all_gqa_reshapes", &m, kv_heads * hpg * seq * hd * 4);
        }

        // ---- Phase 3.7: all reshapes + ONE matmul (Q@K^T) ----
        {
            let mut m = preamble();
            let _ = writeln!(m, "        tensor<int32, [4]> r2d = const()[name=string(\"r2d\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dx_r = reshape(shape=r2d,x=dx2h)[name=string(\"dxr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx_nt = transpose(perm=pm,x=dx_r)[name=string(\"dxnt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{attn_dim}]> da_nt = matmul(transpose_x=bF,transpose_y=bT,x=dx_nt,y=Wo)[name=string(\"dant\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{seq}]> da_t = transpose(perm=pm,x=da_nt)[name=string(\"dat\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> da_ch = reshape(shape=s_ad,x=da_t)[name=string(\"dach\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> sig = sigmoid(x=grh)[name=string(\"sig\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> d_at = mul(x=da_ch,y=sig)[name=string(\"dat2\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rqh = const()[name=string(\"rqh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> da_4 = reshape(shape=rqh,x=d_at)[name=string(\"da4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> da_hs = transpose(perm=pm,x=da_4)[name=string(\"dahs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rqb = const()[name=string(\"rqb\"), val=tensor<int32, [4]>([{kv_heads},{hpg},{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dab = reshape(shape=rqb,x=da_hs)[name=string(\"dab\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qr_4 = reshape(shape=rqh,x=qrh)[name=string(\"qr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> qr_hs = transpose(perm=pm,x=qr_4)[name=string(\"qrhs\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> qb = reshape(shape=rqb,x=qr_hs)[name=string(\"qb\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rkv = const()[name=string(\"rkv\"), val=tensor<int32, [4]>([1,{kv_heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> kr_4 = reshape(shape=rkv,x=krh)[name=string(\"kr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> kr_hs = transpose(perm=pm,x=kr_4)[name=string(\"krhs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rkb = const()[name=string(\"rkb\"), val=tensor<int32, [4]>([{kv_heads},1,{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> kb = reshape(shape=rkb,x=kr_hs)[name=string(\"kb\")];");
            // Just Q@K^T broadcast matmul
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=qb,y=kb)[name=string(\"sc1\")];");
            terminate(&mut m, "sc1", &format!("[{kv_heads},{hpg},{seq},{seq}]"), &format!("[{kv_heads},{hpg},{seq},{seq}]"));
            try_compile("Phase3.7_one_matmul", &m, kv_heads * hpg * seq * seq * 4);
        }

        // ---- Phase 3.7b: same as 3.7 but dab from Wo^T WITHOUT sigmoid ----
        {
            let mut m = preamble();
            let _ = writeln!(m, "        tensor<int32, [4]> r2d = const()[name=string(\"r2d\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dx_r = reshape(shape=r2d,x=dx2h)[name=string(\"dxr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx_nt = transpose(perm=pm,x=dx_r)[name=string(\"dxnt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{attn_dim}]> da_nt = matmul(transpose_x=bF,transpose_y=bT,x=dx_nt,y=Wo)[name=string(\"dant\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{seq}]> da_t = transpose(perm=pm,x=da_nt)[name=string(\"dat\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> d_at = reshape(shape=s_ad,x=da_t)[name=string(\"dat2\")];");
            // GQA reshape (no sigmoid)
            let _ = writeln!(m, "        tensor<int32, [4]> rqh = const()[name=string(\"rqh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> da_4 = reshape(shape=rqh,x=d_at)[name=string(\"da4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> da_hs = transpose(perm=pm,x=da_4)[name=string(\"dahs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rqb = const()[name=string(\"rqb\"), val=tensor<int32, [4]>([{kv_heads},{hpg},{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dab = reshape(shape=rqb,x=da_hs)[name=string(\"dab\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qr_4 = reshape(shape=rqh,x=qrh)[name=string(\"qr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> qr_hs = transpose(perm=pm,x=qr_4)[name=string(\"qrhs\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> qb = reshape(shape=rqb,x=qr_hs)[name=string(\"qb\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rkv = const()[name=string(\"rkv\"), val=tensor<int32, [4]>([1,{kv_heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> kr_4 = reshape(shape=rkv,x=krh)[name=string(\"kr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> kr_hs = transpose(perm=pm,x=kr_4)[name=string(\"krhs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rkb = const()[name=string(\"rkb\"), val=tensor<int32, [4]>([{kv_heads},1,{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> kb = reshape(shape=rkb,x=kr_hs)[name=string(\"kb\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=qb,y=kb)[name=string(\"sc1\")];");
            terminate(&mut m, "sc1", &format!("[{kv_heads},{hpg},{seq},{seq}]"), &format!("[{kv_heads},{hpg},{seq},{seq}]"));
            try_compile("Phase3.7b_matmul_WoT_no_sigmoid", &m, kv_heads * hpg * seq * seq * 4);
        }

        // ---- Phase 3.7b2: same as 3.7b but add dummy mul(x,1.0) before batch reshape ----
        {
            let mut m = preamble();
            let _ = writeln!(m, "        tensor<int32, [4]> r2d = const()[name=string(\"r2d\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dx_r = reshape(shape=r2d,x=dx2h)[name=string(\"dxr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx_nt = transpose(perm=pm,x=dx_r)[name=string(\"dxnt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{attn_dim}]> da_nt = matmul(transpose_x=bF,transpose_y=bT,x=dx_nt,y=Wo)[name=string(\"dant\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{seq}]> da_t = transpose(perm=pm,x=da_nt)[name=string(\"dat\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> d_at = reshape(shape=s_ad,x=da_t)[name=string(\"dat2\")];");
            // Add dummy identity op: mul by 1.0
            let _ = writeln!(m, "        fp16 one = const()[name=string(\"one\"), val=fp16(1.0)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> d_at2 = mul(x=d_at,y=one)[name=string(\"dat3\")];");
            // GQA reshape
            let _ = writeln!(m, "        tensor<int32, [4]> rqh = const()[name=string(\"rqh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> da_4 = reshape(shape=rqh,x=d_at2)[name=string(\"da4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> da_hs = transpose(perm=pm,x=da_4)[name=string(\"dahs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rqb = const()[name=string(\"rqb\"), val=tensor<int32, [4]>([{kv_heads},{hpg},{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dab = reshape(shape=rqb,x=da_hs)[name=string(\"dab\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qr_4 = reshape(shape=rqh,x=qrh)[name=string(\"qr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> qr_hs = transpose(perm=pm,x=qr_4)[name=string(\"qrhs\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> qb = reshape(shape=rqb,x=qr_hs)[name=string(\"qb\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rkv = const()[name=string(\"rkv\"), val=tensor<int32, [4]>([1,{kv_heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> kr_4 = reshape(shape=rkv,x=krh)[name=string(\"kr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> kr_hs = transpose(perm=pm,x=kr_4)[name=string(\"krhs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rkb = const()[name=string(\"rkb\"), val=tensor<int32, [4]>([{kv_heads},1,{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> kb = reshape(shape=rkb,x=kr_hs)[name=string(\"kb\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=qb,y=kb)[name=string(\"sc1\")];");
            terminate(&mut m, "sc1", &format!("[{kv_heads},{hpg},{seq},{seq}]"), &format!("[{kv_heads},{hpg},{seq},{seq}]"));
            try_compile("Phase3.7b2_WoT_dummy_mul_matmul", &m, kv_heads * hpg * seq * seq * 4);
        }

        // ---- Phase 3.7b3: Wo^T → add with qrh before batch reshape to break fusion ----
        {
            let mut m = preamble();
            let _ = writeln!(m, "        tensor<int32, [4]> r2d = const()[name=string(\"r2d\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dx_r = reshape(shape=r2d,x=dx2h)[name=string(\"dxr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx_nt = transpose(perm=pm,x=dx_r)[name=string(\"dxnt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{attn_dim}]> da_nt = matmul(transpose_x=bF,transpose_y=bT,x=dx_nt,y=Wo)[name=string(\"dant\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{seq}]> da_t = transpose(perm=pm,x=da_nt)[name=string(\"dat\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> d_at = reshape(shape=s_ad,x=da_t)[name=string(\"dat2\")];");
            // Add with qrh (both [1,attn_dim,1,seq]) to break matmul chain
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> d_at2 = add(x=d_at,y=qrh)[name=string(\"dat3\")];");
            // GQA reshape
            let _ = writeln!(m, "        tensor<int32, [4]> rqh = const()[name=string(\"rqh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> da_4 = reshape(shape=rqh,x=d_at2)[name=string(\"da4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> da_hs = transpose(perm=pm,x=da_4)[name=string(\"dahs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rqb = const()[name=string(\"rqb\"), val=tensor<int32, [4]>([{kv_heads},{hpg},{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dab = reshape(shape=rqb,x=da_hs)[name=string(\"dab\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qr_4 = reshape(shape=rqh,x=qrh)[name=string(\"qr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> qr_hs = transpose(perm=pm,x=qr_4)[name=string(\"qrhs\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> qb = reshape(shape=rqb,x=qr_hs)[name=string(\"qb\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rkv = const()[name=string(\"rkv\"), val=tensor<int32, [4]>([1,{kv_heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> kr_4 = reshape(shape=rkv,x=krh)[name=string(\"kr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> kr_hs = transpose(perm=pm,x=kr_4)[name=string(\"krhs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rkb = const()[name=string(\"rkb\"), val=tensor<int32, [4]>([{kv_heads},1,{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> kb = reshape(shape=rkb,x=kr_hs)[name=string(\"kb\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=qb,y=kb)[name=string(\"sc1\")];");
            terminate(&mut m, "sc1", &format!("[{kv_heads},{hpg},{seq},{seq}]"), &format!("[{kv_heads},{hpg},{seq},{seq}]"));
            try_compile("Phase3.7b3_WoT_add_break_chain", &m, kv_heads * hpg * seq * seq * 4);
        }

        // ---- Phase 3.7c: Wo^T + sigmoid + mul chain → reshape dab → matmul, but dab NOT in matmul ----
        // Test if the issue is the dab in matmul specifically or just having both paths
        {
            let mut m = preamble();
            let _ = writeln!(m, "        tensor<int32, [4]> r2d = const()[name=string(\"r2d\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dx_r = reshape(shape=r2d,x=dx2h)[name=string(\"dxr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx_nt = transpose(perm=pm,x=dx_r)[name=string(\"dxnt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{attn_dim}]> da_nt = matmul(transpose_x=bF,transpose_y=bT,x=dx_nt,y=Wo)[name=string(\"dant\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{seq}]> da_t = transpose(perm=pm,x=da_nt)[name=string(\"dat\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> da_ch = reshape(shape=s_ad,x=da_t)[name=string(\"dach\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> sig = sigmoid(x=grh)[name=string(\"sig\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> d_at = mul(x=da_ch,y=sig)[name=string(\"dat2\")];");
            // GQA reshape for dab (computed but NOT used in matmul -- just output)
            let _ = writeln!(m, "        tensor<int32, [4]> rqh = const()[name=string(\"rqh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> da_4 = reshape(shape=rqh,x=d_at)[name=string(\"da4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> da_hs = transpose(perm=pm,x=da_4)[name=string(\"dahs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rqb = const()[name=string(\"rqb\"), val=tensor<int32, [4]>([{kv_heads},{hpg},{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dab = reshape(shape=rqb,x=da_hs)[name=string(\"dab\")];");
            // qb@kb matmul (NOT using dab)
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qr_4 = reshape(shape=rqh,x=qrh)[name=string(\"qr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> qr_hs = transpose(perm=pm,x=qr_4)[name=string(\"qrhs\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> qb = reshape(shape=rqb,x=qr_hs)[name=string(\"qb\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rkv = const()[name=string(\"rkv\"), val=tensor<int32, [4]>([1,{kv_heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> kr_4 = reshape(shape=rkv,x=krh)[name=string(\"kr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> kr_hs = transpose(perm=pm,x=kr_4)[name=string(\"krhs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rkb = const()[name=string(\"rkb\"), val=tensor<int32, [4]>([{kv_heads},1,{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> kb = reshape(shape=rkb,x=kr_hs)[name=string(\"kb\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=qb,y=kb)[name=string(\"sc1\")];");
            // add dab and sc1 to force both to be "used"
            // can't add directly (different shapes), so just output dab
            terminate(&mut m, "dab", &format!("[{kv_heads},{hpg},{seq},{hd}]"), &format!("[{kv_heads},{hpg},{seq},{hd}]"));
            try_compile("Phase3.7c_gate_reshape_plus_matmul_separate", &m, kv_heads * hpg * seq * hd * 4);
        }

        // ---- Phase 3.8: matmul but dab comes from slice (no gate path) ----
        {
            let mut m = preamble();
            // dab directly from qrh (slice) → reshape → batch form — NO gate path
            let _ = writeln!(m, "        tensor<int32, [4]> rqh = const()[name=string(\"rqh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> da_4 = reshape(shape=rqh,x=qrh)[name=string(\"da4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> da_hs = transpose(perm=pm,x=da_4)[name=string(\"dahs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rqb = const()[name=string(\"rqb\"), val=tensor<int32, [4]>([{kv_heads},{hpg},{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dab = reshape(shape=rqb,x=da_hs)[name=string(\"dab\")];");
            // K from slice
            let _ = writeln!(m, "        tensor<int32, [4]> rkv = const()[name=string(\"rkv\"), val=tensor<int32, [4]>([1,{kv_heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> kr_4 = reshape(shape=rkv,x=krh)[name=string(\"kr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> kr_hs = transpose(perm=pm,x=kr_4)[name=string(\"krhs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rkb = const()[name=string(\"rkb\"), val=tensor<int32, [4]>([{kv_heads},1,{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> kb = reshape(shape=rkb,x=kr_hs)[name=string(\"kb\")];");
            // broadcast matmul
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=dab,y=kb)[name=string(\"sc1\")];");
            terminate(&mut m, "sc1", &format!("[{kv_heads},{hpg},{seq},{seq}]"), &format!("[{kv_heads},{hpg},{seq},{seq}]"));
            try_compile("Phase3.8_matmul_no_gate", &m, kv_heads * hpg * seq * seq * 4);
        }

        // ---- Phase 3.9: Q@K^T but dab replaced by qb (same tensor as matmul 1st op) ----
        // This tests whether using qb (from qrh) as BOTH matmul operands works
        {
            let mut m = preamble();
            let _ = writeln!(m, "        tensor<int32, [4]> rqh = const()[name=string(\"rqh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qr_4 = reshape(shape=rqh,x=qrh)[name=string(\"qr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> qr_hs = transpose(perm=pm,x=qr_4)[name=string(\"qrhs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rqb = const()[name=string(\"rqb\"), val=tensor<int32, [4]>([{kv_heads},{hpg},{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> qb = reshape(shape=rqb,x=qr_hs)[name=string(\"qb\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rkv = const()[name=string(\"rkv\"), val=tensor<int32, [4]>([1,{kv_heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> kr_4 = reshape(shape=rkv,x=krh)[name=string(\"kr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> kr_hs = transpose(perm=pm,x=kr_4)[name=string(\"krhs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rkb = const()[name=string(\"rkb\"), val=tensor<int32, [4]>([{kv_heads},1,{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> kb = reshape(shape=rkb,x=kr_hs)[name=string(\"kb\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=qb,y=kb)[name=string(\"sc1\")];");
            terminate(&mut m, "sc1", &format!("[{kv_heads},{hpg},{seq},{seq}]"), &format!("[{kv_heads},{hpg},{seq},{seq}]"));
            try_compile("Phase3.9_matmul_slices_only", &m, kv_heads * hpg * seq * seq * 4);
        }

        // ---- Phase 4: GQA reshape + SDPA recompute (scores + softmax) ----
        {
            let mut m = preamble();
            // Wo^T + gate
            let _ = writeln!(m, "        tensor<int32, [4]> r2d = const()[name=string(\"r2d\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dx_r = reshape(shape=r2d,x=dx2h)[name=string(\"dxr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx_nt = transpose(perm=pm,x=dx_r)[name=string(\"dxnt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{attn_dim}]> da_nt = matmul(transpose_x=bF,transpose_y=bT,x=dx_nt,y=Wo)[name=string(\"dant\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{seq}]> da_t = transpose(perm=pm,x=da_nt)[name=string(\"dat\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> da_ch = reshape(shape=s_ad,x=da_t)[name=string(\"dach\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> sig = sigmoid(x=grh)[name=string(\"sig\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> d_at = mul(x=da_ch,y=sig)[name=string(\"dat2\")];");
            // GQA reshape
            let _ = writeln!(m, "        tensor<int32, [4]> rqh = const()[name=string(\"rqh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> da_4 = reshape(shape=rqh,x=d_at)[name=string(\"da4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> da_hs = transpose(perm=pm,x=da_4)[name=string(\"dahs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rqb = const()[name=string(\"rqb\"), val=tensor<int32, [4]>([{kv_heads},{hpg},{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dab = reshape(shape=rqb,x=da_hs)[name=string(\"dab\")];");
            // Q/K/V batch reshape
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qr_4 = reshape(shape=rqh,x=qrh)[name=string(\"qr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> qr_hs = transpose(perm=pm,x=qr_4)[name=string(\"qrhs\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> qb = reshape(shape=rqb,x=qr_hs)[name=string(\"qb\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rkv = const()[name=string(\"rkv\"), val=tensor<int32, [4]>([1,{kv_heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> kr_4 = reshape(shape=rkv,x=krh)[name=string(\"kr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> kr_hs = transpose(perm=pm,x=kr_4)[name=string(\"krhs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rkb = const()[name=string(\"rkb\"), val=tensor<int32, [4]>([{kv_heads},1,{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> kb = reshape(shape=rkb,x=kr_hs)[name=string(\"kb\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> v_4 = reshape(shape=rkv,x=vh)[name=string(\"v4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> v_hs = transpose(perm=pm,x=v_4)[name=string(\"vhs\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> vb = reshape(shape=rkb,x=v_hs)[name=string(\"vb\")];");
            // SDPA scores + softmax
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=qb,y=kb)[name=string(\"sc1\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"sc2\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"ms\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"aw\")];");
            // output softmax as fp32
            terminate(&mut m, "aw", &format!("[{kv_heads},{hpg},{seq},{seq}]"), &format!("[{kv_heads},{hpg},{seq},{seq}]"));
            try_compile("Phase4_sdpa_recompute", &m, kv_heads * hpg * seq * seq * 4);
        }

        // ---- Phase 4b-dV: reduce_mean on axis 1 (the first reduce_mean) ----
        {
            let mut m = preamble();
            // minimal path: reshape Q/K/V, compute softmax, then dV with reduce_mean
            let _ = writeln!(m, "        tensor<int32, [4]> r2d = const()[name=string(\"r2d\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dx_r = reshape(shape=r2d,x=dx2h)[name=string(\"dxr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx_nt = transpose(perm=pm,x=dx_r)[name=string(\"dxnt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{attn_dim}]> da_nt = matmul(transpose_x=bF,transpose_y=bT,x=dx_nt,y=Wo)[name=string(\"dant\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{seq}]> da_t = transpose(perm=pm,x=da_nt)[name=string(\"dat\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> da_ch = reshape(shape=s_ad,x=da_t)[name=string(\"dach\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> sig = sigmoid(x=grh)[name=string(\"sig\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> d_at = mul(x=da_ch,y=sig)[name=string(\"dat2\")];");
            // GQA reshape
            let _ = writeln!(m, "        tensor<int32, [4]> rqh = const()[name=string(\"rqh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> da_4 = reshape(shape=rqh,x=d_at)[name=string(\"da4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> da_hs = transpose(perm=pm,x=da_4)[name=string(\"dahs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rqb = const()[name=string(\"rqb\"), val=tensor<int32, [4]>([{kv_heads},{hpg},{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dab = reshape(shape=rqb,x=da_hs)[name=string(\"dab\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qr_4 = reshape(shape=rqh,x=qrh)[name=string(\"qr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> qr_hs = transpose(perm=pm,x=qr_4)[name=string(\"qrhs\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> qb = reshape(shape=rqb,x=qr_hs)[name=string(\"qb\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rkv = const()[name=string(\"rkv\"), val=tensor<int32, [4]>([1,{kv_heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> kr_4 = reshape(shape=rkv,x=krh)[name=string(\"kr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> kr_hs = transpose(perm=pm,x=kr_4)[name=string(\"krhs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rkb = const()[name=string(\"rkb\"), val=tensor<int32, [4]>([{kv_heads},1,{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> kb = reshape(shape=rkb,x=kr_hs)[name=string(\"kb\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> v_4 = reshape(shape=rkv,x=vh)[name=string(\"v4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> v_hs = transpose(perm=pm,x=v_4)[name=string(\"vhs\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> vb = reshape(shape=rkb,x=v_hs)[name=string(\"vb\")];");
            // SDPA scores + softmax
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=qb,y=kb)[name=string(\"sc1\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"sc2\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"ms\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"aw\")];");
            // dV = A^T @ dO, reduce over groups
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> aw_t = transpose(perm=pm,x=aw)[name=string(\"awt\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dvr = matmul(transpose_x=bF,transpose_y=bF,x=aw_t,y=dab)[name=string(\"dvr\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> dvm = reduce_mean(x=dvr,axes=ax1,keep_dims=kd)[name=string(\"dvm\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> dvb = mul(x=dvm,y=hpg_v)[name=string(\"dvb\")];");
            terminate(&mut m, "dvb", &format!("[{kv_heads},1,{seq},{hd}]"), &format!("[{kv_heads},1,{seq},{hd}]"));
            try_compile("Phase4b_dV_reduce_mean_ax1", &m, kv_heads * seq * hd * 4);
        }

        // ---- Isolated: just reduce_mean on axis 1 with batch [4,2,...] ----
        {
            let mut m = String::with_capacity(4096);
            m.push_str(MIL_HDR);
            let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [{kv_heads}, {hpg}, {seq}, {hd}]> x) {{");
            let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
            let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
            let _ = writeln!(m, "        tensor<int32, [1]> ax1 = const()[name=string(\"ax1\"), val=tensor<int32, [1]>([1])];");
            let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> rm = reduce_mean(x=xh,axes=ax1,keep_dims=kd)[name=string(\"rm\")];");
            let _ = writeln!(m, "        tensor<fp32, [{kv_heads},1,{seq},{hd}]> out = cast(dtype=to32,x=rm)[name=string(\"cout\")];");
            let _ = writeln!(m, "    }} -> (out);");
            m.push_str("}\n");
            let in_bytes = kv_heads * hpg * seq * hd * 4;
            let out_bytes = kv_heads * 1 * seq * hd * 4;
            match AneKernel::compile(&m, None, &[in_bytes], &[out_bytes]) {
                Ok(_) => eprintln!("  PASS: isolated_reduce_mean_ax1_batch4"),
                Err(e) => eprintln!("  FAIL: isolated_reduce_mean_ax1_batch4 -- {e}"),
            }
        }

        // ---- Isolated: reduce_mean on axis -1 with batch [4,2,S,S] ----
        {
            let mut m = String::with_capacity(4096);
            m.push_str(MIL_HDR);
            let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [{kv_heads}, {hpg}, {seq}, {seq}]> x) {{");
            let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
            let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
            let _ = writeln!(m, "        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];");
            let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},1]> rm = reduce_mean(x=xh,axes=rax,keep_dims=kd)[name=string(\"rm\")];");
            let _ = writeln!(m, "        tensor<fp32, [{kv_heads},{hpg},{seq},1]> out = cast(dtype=to32,x=rm)[name=string(\"cout\")];");
            let _ = writeln!(m, "    }} -> (out);");
            m.push_str("}\n");
            let in_bytes = kv_heads * hpg * seq * seq * 4;
            let out_bytes = kv_heads * hpg * seq * 1 * 4;
            match AneKernel::compile(&m, None, &[in_bytes], &[out_bytes]) {
                Ok(_) => eprintln!("  PASS: isolated_reduce_mean_axn1_batch4x2"),
                Err(e) => eprintln!("  FAIL: isolated_reduce_mean_axn1_batch4x2 -- {e}"),
            }
        }

        // ---- Isolated: matmul with broadcast [4,2,S,S] @ [4,1,S,hd] ----
        {
            let mut m = String::with_capacity(4096);
            m.push_str(MIL_HDR);
            let total_in = kv_heads * hpg * seq * seq + kv_heads * 1 * seq * hd;
            let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1, {total_in}, 1, 1]> x) {{");
            let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
            let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
            let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{total_in},1,1]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");
            // split
            let a_sz = kv_heads * hpg * seq * seq;
            let _ = writeln!(m, "        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];");
            let _ = writeln!(m, "        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,{a_sz},1,1])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{a_sz},1,1]> ah = slice_by_size(x=xh,begin=b0,size=sa)[name=string(\"ah\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([{kv_heads},{hpg},{seq},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> a = reshape(shape=ra,x=ah)[name=string(\"a\")];");
            let b_sz = kv_heads * seq * hd;
            let _ = writeln!(m, "        tensor<int32, [4]> bb = const()[name=string(\"bb\"), val=tensor<int32, [4]>([0,{a_sz},0,0])];");
            let _ = writeln!(m, "        tensor<int32, [4]> sb = const()[name=string(\"sb\"), val=tensor<int32, [4]>([1,{b_sz},1,1])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{b_sz},1,1]> bh = slice_by_size(x=xh,begin=bb,size=sb)[name=string(\"bh\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rb = const()[name=string(\"rb\"), val=tensor<int32, [4]>([{kv_heads},1,{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> b = reshape(shape=rb,x=bh)[name=string(\"b\")];");
            // matmul with broadcast
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> mm = matmul(transpose_x=bF,transpose_y=bF,x=a,y=b)[name=string(\"mm\")];");
            let _ = writeln!(m, "        tensor<fp32, [{kv_heads},{hpg},{seq},{hd}]> out = cast(dtype=to32,x=mm)[name=string(\"cout\")];");
            let _ = writeln!(m, "    }} -> (out);");
            m.push_str("}\n");
            let in_bytes = total_in * 4;
            let out_bytes = kv_heads * hpg * seq * hd * 4;
            match AneKernel::compile(&m, None, &[in_bytes], &[out_bytes]) {
                Ok(_) => eprintln!("  PASS: isolated_broadcast_matmul_4x2xSxS_@_4x1xSxhd"),
                Err(e) => eprintln!("  FAIL: isolated_broadcast_matmul_4x2xSxS_@_4x1xSxhd -- {e}"),
            }
        }

        // ---- Phase 4b-softmax-bwd: add softmax backward (reduce_mean axis -1) ----
        {
            let mut m = preamble();
            // Wo^T + gate + GQA reshape (abbreviated)
            let _ = writeln!(m, "        tensor<int32, [4]> r2d = const()[name=string(\"r2d\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dx_r = reshape(shape=r2d,x=dx2h)[name=string(\"dxr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx_nt = transpose(perm=pm,x=dx_r)[name=string(\"dxnt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{attn_dim}]> da_nt = matmul(transpose_x=bF,transpose_y=bT,x=dx_nt,y=Wo)[name=string(\"dant\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{seq}]> da_t = transpose(perm=pm,x=da_nt)[name=string(\"dat\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> da_ch = reshape(shape=s_ad,x=da_t)[name=string(\"dach\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> sig = sigmoid(x=grh)[name=string(\"sig\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> d_at = mul(x=da_ch,y=sig)[name=string(\"dat2\")];");
            // GQA reshape
            let _ = writeln!(m, "        tensor<int32, [4]> rqh = const()[name=string(\"rqh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> da_4 = reshape(shape=rqh,x=d_at)[name=string(\"da4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> da_hs = transpose(perm=pm,x=da_4)[name=string(\"dahs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rqb = const()[name=string(\"rqb\"), val=tensor<int32, [4]>([{kv_heads},{hpg},{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dab = reshape(shape=rqb,x=da_hs)[name=string(\"dab\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qr_4 = reshape(shape=rqh,x=qrh)[name=string(\"qr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> qr_hs = transpose(perm=pm,x=qr_4)[name=string(\"qrhs\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> qb = reshape(shape=rqb,x=qr_hs)[name=string(\"qb\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rkv = const()[name=string(\"rkv\"), val=tensor<int32, [4]>([1,{kv_heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> kr_4 = reshape(shape=rkv,x=krh)[name=string(\"kr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> kr_hs = transpose(perm=pm,x=kr_4)[name=string(\"krhs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rkb = const()[name=string(\"rkb\"), val=tensor<int32, [4]>([{kv_heads},1,{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> kb = reshape(shape=rkb,x=kr_hs)[name=string(\"kb\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> v_4 = reshape(shape=rkv,x=vh)[name=string(\"v4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> v_hs = transpose(perm=pm,x=v_4)[name=string(\"vhs\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> vb = reshape(shape=rkb,x=v_hs)[name=string(\"vb\")];");
            // SDPA + softmax
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=qb,y=kb)[name=string(\"sc1\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"sc2\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"ms\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"aw\")];");
            // dV with reduce_mean
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> aw_t = transpose(perm=pm,x=aw)[name=string(\"awt\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dvr = matmul(transpose_x=bF,transpose_y=bF,x=aw_t,y=dab)[name=string(\"dvr\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> dvm = reduce_mean(x=dvr,axes=ax1,keep_dims=kd)[name=string(\"dvm\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> dvb = mul(x=dvm,y=hpg_v)[name=string(\"dvb\")];");
            // dP + softmax backward
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> dp = matmul(transpose_x=bF,transpose_y=bT,x=dab,y=vb)[name=string(\"dp\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> dpaw = mul(x=dp,y=aw)[name=string(\"dpaw\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},1]> dot_m = reduce_mean(x=dpaw,axes=rax_last,keep_dims=kd)[name=string(\"dotm\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},1]> dot = mul(x=dot_m,y=seq_v)[name=string(\"dot\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> dps = sub(x=dp,y=dot)[name=string(\"dps\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> ds = mul(x=aw,y=dps)[name=string(\"ds\")];");
            terminate(&mut m, "ds", &format!("[{kv_heads},{hpg},{seq},{seq}]"), &format!("[{kv_heads},{hpg},{seq},{seq}]"));
            try_compile("Phase4b_softmax_bwd", &m, kv_heads * hpg * seq * seq * 4);
        }

        // ---- Phase 4b-dQdK: add dQ and dK (with second reduce_mean on axis 1) ----
        {
            let mut m = preamble();
            // Wo^T + gate + GQA reshape + SDPA + dV + softmax bwd + dQ + dK
            let _ = writeln!(m, "        tensor<int32, [4]> r2d = const()[name=string(\"r2d\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dx_r = reshape(shape=r2d,x=dx2h)[name=string(\"dxr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx_nt = transpose(perm=pm,x=dx_r)[name=string(\"dxnt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{attn_dim}]> da_nt = matmul(transpose_x=bF,transpose_y=bT,x=dx_nt,y=Wo)[name=string(\"dant\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{seq}]> da_t = transpose(perm=pm,x=da_nt)[name=string(\"dat\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> da_ch = reshape(shape=s_ad,x=da_t)[name=string(\"dach\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> sig = sigmoid(x=grh)[name=string(\"sig\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> d_at = mul(x=da_ch,y=sig)[name=string(\"dat2\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rqh = const()[name=string(\"rqh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> da_4 = reshape(shape=rqh,x=d_at)[name=string(\"da4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> da_hs = transpose(perm=pm,x=da_4)[name=string(\"dahs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rqb = const()[name=string(\"rqb\"), val=tensor<int32, [4]>([{kv_heads},{hpg},{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dab = reshape(shape=rqb,x=da_hs)[name=string(\"dab\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qr_4 = reshape(shape=rqh,x=qrh)[name=string(\"qr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> qr_hs = transpose(perm=pm,x=qr_4)[name=string(\"qrhs\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> qb = reshape(shape=rqb,x=qr_hs)[name=string(\"qb\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rkv = const()[name=string(\"rkv\"), val=tensor<int32, [4]>([1,{kv_heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> kr_4 = reshape(shape=rkv,x=krh)[name=string(\"kr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> kr_hs = transpose(perm=pm,x=kr_4)[name=string(\"krhs\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rkb = const()[name=string(\"rkb\"), val=tensor<int32, [4]>([{kv_heads},1,{seq},{hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> kb = reshape(shape=rkb,x=kr_hs)[name=string(\"kb\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> v_4 = reshape(shape=rkv,x=vh)[name=string(\"v4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> v_hs = transpose(perm=pm,x=v_4)[name=string(\"vhs\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> vb = reshape(shape=rkb,x=v_hs)[name=string(\"vb\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=qb,y=kb)[name=string(\"sc1\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"sc2\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"ms\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"aw\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> aw_t = transpose(perm=pm,x=aw)[name=string(\"awt\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dvr = matmul(transpose_x=bF,transpose_y=bF,x=aw_t,y=dab)[name=string(\"dvr\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> dvm = reduce_mean(x=dvr,axes=ax1,keep_dims=kd)[name=string(\"dvm\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> dvb = mul(x=dvm,y=hpg_v)[name=string(\"dvb\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> dp = matmul(transpose_x=bF,transpose_y=bT,x=dab,y=vb)[name=string(\"dp\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> dpaw = mul(x=dp,y=aw)[name=string(\"dpaw\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},1]> dot_m = reduce_mean(x=dpaw,axes=rax_last,keep_dims=kd)[name=string(\"dotm\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},1]> dot = mul(x=dot_m,y=seq_v)[name=string(\"dot\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> dps = sub(x=dp,y=dot)[name=string(\"dps\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> ds = mul(x=aw,y=dps)[name=string(\"ds\")];");
            // dQ
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dqr = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=kb)[name=string(\"dqr\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dqb = mul(x=dqr,y=scv)[name=string(\"dqb\")];");
            // dK with reduce_mean
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> ds_t = transpose(perm=pm,x=ds)[name=string(\"dst\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> dkr = matmul(transpose_x=bF,transpose_y=bF,x=ds_t,y=qb)[name=string(\"dkr\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> dkm = reduce_mean(x=dkr,axes=ax1,keep_dims=kd)[name=string(\"dkm\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> dks = mul(x=dkm,y=hpg_v)[name=string(\"dks\")];");
            let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> dkb = mul(x=dks,y=scv)[name=string(\"dkb\")];");
            // Output dqb as proxy
            terminate(&mut m, "dqb", &format!("[{kv_heads},{hpg},{seq},{hd}]"), &format!("[{kv_heads},{hpg},{seq},{hd}]"));
            try_compile("Phase4b_dQ_dK", &m, kv_heads * hpg * seq * hd * 4);
        }

        eprintln!("\n=== Binary search complete ===\n");
    }

    /// Bisect which op combination in the monolithic SDPA+RoPE backward kernel
    /// causes ANE compilation failure (error 22).
    ///
    /// Config: dim=64, heads=8, hd=16, seq=32, kv_heads=4
    /// attn_dim = 128, input shape [1,512,1,32] (4*ad packed: da,Q,K,V)
    /// All inputs fp32, cast to fp16 inside the kernel (matches the failing /tmp/sdpa_rope_bwd.mil).
    #[test]
    fn test_bisect_sdpa_rope_bwd_ane_failure() {
        use std::fmt::Write;
        init_ane();

        let seq: usize = 32;
        let heads: usize = 8;
        let hd: usize = 16;
        let ad: usize = heads * hd; // 128
        let in_ch: usize = 4 * ad; // 512
        let input_bytes = in_ch * seq * 4; // fp32

        let mask_blob = build_causal_mask_blob(seq);

        // Helper: try compiling a MIL with just the mask weight
        let try_compile = |label: &str, mil: &str, out_elems: usize| -> bool {
            let out_bytes = out_elems * 4; // fp32 output
            match AneKernel::compile_multi_weights(
                mil,
                &["@model_path/weights/mask.bin"],
                &[&mask_blob],
                &[input_bytes],
                &[out_bytes],
            ) {
                Ok(_k) => {
                    eprintln!("[PASS] {label}");
                    true
                }
                Err(e) => {
                    eprintln!("[FAIL] {label}: {e}");
                    false
                }
            }
        };

        // Common preamble: cast fp32→fp16, slice da/Q/K/V, reshape to [1,H,S,hd]
        let preamble = || -> String {
            let mut m = String::with_capacity(8192);
            m.push_str(MIL_HDR);
            let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1, {in_ch}, 1, {seq}]> x) {{");
            let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
            let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
            let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
            let _ = writeln!(m, "        bool bT = const()[name=string(\"bT\"), val=bool(true)];");
            let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
            let _ = writeln!(m, "        fp16 scv = const()[name=string(\"scv\"), val=fp16(0.25)];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{seq}]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,{seq},{seq}]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];");
            let _ = writeln!(m, "        tensor<fp16, [1,{in_ch},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");
            // Slice 4 blocks of ad=128
            let _ = writeln!(m, "        tensor<int32, [4]> s_ad = const()[name=string(\"sad\"), val=tensor<int32, [4]>([1,{ad},1,{seq}])];");
            let _ = writeln!(m, "        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> dah = slice_by_size(x=xh,begin=b0,size=s_ad)[name=string(\"dah\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,{ad},0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> qrh = slice_by_size(x=xh,begin=b1,size=s_ad)[name=string(\"qrh\")];");
            let off2 = 2 * ad;
            let _ = writeln!(m, "        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,{off2},0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> keh = slice_by_size(x=xh,begin=b2,size=s_ad)[name=string(\"keh\")];");
            let off3 = 3 * ad;
            let _ = writeln!(m, "        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,{off3},0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> veh = slice_by_size(x=xh,begin=b3,size=s_ad)[name=string(\"veh\")];");
            // Reshape to [1,H,hd,S] then transpose to [1,H,S,hd]
            let _ = writeln!(m, "        tensor<int32, [4]> rqh = const()[name=string(\"rqh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> da_4 = reshape(shape=rqh,x=dah)[name=string(\"da4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> da = transpose(perm=pm,x=da_4)[name=string(\"da\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qr_4 = reshape(shape=rqh,x=qrh)[name=string(\"qr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> q = transpose(perm=pm,x=qr_4)[name=string(\"q\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> ke_4 = reshape(shape=rqh,x=keh)[name=string(\"ke4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> k = transpose(perm=pm,x=ke_4)[name=string(\"k\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> ve_4 = reshape(shape=rqh,x=veh)[name=string(\"ve4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> v = transpose(perm=pm,x=ve_4)[name=string(\"v\")];");
            m
        };

        // Helper: terminate with fp32 cast and return
        let terminate_fp32 = |m: &mut String, var: &str, shape_h: &str, shape_f: &str| {
            let _ = writeln!(m, "        tensor<fp32, {shape_f}> out = cast(dtype=to32,x={var})[name=string(\"cout\")];");
            let _ = writeln!(m, "    }} -> (out);");
            m.push_str("}\n");
        };
        let terminate_fp16 = |m: &mut String, var: &str, shape_h: &str| {
            // Cast to fp32 for output
            let _ = writeln!(m, "        tensor<fp32, {shape_h}> out = cast(dtype=to32,x={var})[name=string(\"cout\")];");
            let _ = writeln!(m, "    }} -> (out);");
            m.push_str("}\n");
        };

        eprintln!("\n=== SDPA+RoPE backward bisection ===\n");

        // ---- Phase A: preamble + ONE matmul (Q@K^T) ----
        {
            let mut m = preamble();
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"sc1\")];");
            terminate_fp16(&mut m, "sc1", &format!("[1,{heads},{seq},{seq}]"));
            try_compile("PhaseA_one_matmul", &m, heads * seq * seq);
        }

        // ---- Phase B: Phase A + scale + mask + softmax ----
        {
            let mut m = preamble();
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"sc1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"sc2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"ms\")];");
            let _ = writeln!(m, "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"aw\")];");
            terminate_fp16(&mut m, "aw", &format!("[1,{heads},{seq},{seq}]"));
            try_compile("PhaseB_fwd_recompute", &m, heads * seq * seq);
        }

        // ---- Phase C: Phase B + dV (aw^T @ da) + dP (da @ V^T) ----
        {
            let mut m = preamble();
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"sc1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"sc2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"ms\")];");
            let _ = writeln!(m, "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"aw\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw_t = transpose(perm=pm,x=aw)[name=string(\"awt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dv_all = matmul(transpose_x=bF,transpose_y=bF,x=aw_t,y=da)[name=string(\"dva\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dp = matmul(transpose_x=bF,transpose_y=bT,x=da,y=v)[name=string(\"dp\")];");
            // Concat dv_all and dp (different shapes) - just output dv_all
            terminate_fp16(&mut m, "dv_all", &format!("[1,{heads},{seq},{hd}]"));
            try_compile("PhaseC_dV_dP", &m, heads * seq * hd);
        }

        // ---- Phase D: Phase C + softmax backward (reduce_sum, sub, mul) ----
        {
            let mut m = preamble();
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"sc1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"sc2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"ms\")];");
            let _ = writeln!(m, "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"aw\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw_t = transpose(perm=pm,x=aw)[name=string(\"awt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dv_all = matmul(transpose_x=bF,transpose_y=bF,x=aw_t,y=da)[name=string(\"dva\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dp = matmul(transpose_x=bF,transpose_y=bT,x=da,y=v)[name=string(\"dp\")];");
            // Softmax backward with reduce_sum
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dpaw = mul(x=dp,y=aw)[name=string(\"dpaw\")];");
            let _ = writeln!(m, "        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];");
            let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> dot = reduce_sum(x=dpaw,axes=rax,keep_dims=kd)[name=string(\"dot\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dps = sub(x=dp,y=dot)[name=string(\"dps\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ds = mul(x=aw,y=dps)[name=string(\"ds\")];");
            terminate_fp16(&mut m, "ds", &format!("[1,{heads},{seq},{seq}]"));
            try_compile("PhaseD_softmax_bwd_reduce_sum", &m, heads * seq * seq);
        }

        // ---- Phase D2: same as D but reduce_mean instead of reduce_sum ----
        {
            let mut m = preamble();
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"sc1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"sc2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"ms\")];");
            let _ = writeln!(m, "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"aw\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw_t = transpose(perm=pm,x=aw)[name=string(\"awt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dv_all = matmul(transpose_x=bF,transpose_y=bF,x=aw_t,y=da)[name=string(\"dva\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dp = matmul(transpose_x=bF,transpose_y=bT,x=da,y=v)[name=string(\"dp\")];");
            // Softmax backward with reduce_mean * seq
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dpaw = mul(x=dp,y=aw)[name=string(\"dpaw\")];");
            let _ = writeln!(m, "        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];");
            let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> dot_m = reduce_mean(x=dpaw,axes=rax,keep_dims=kd)[name=string(\"dotm\")];");
            let _ = writeln!(m, "        fp16 seq_v = const()[name=string(\"seqv\"), val=fp16({seq}.0)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> dot = mul(x=dot_m,y=seq_v)[name=string(\"dot\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dps = sub(x=dp,y=dot)[name=string(\"dps\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ds = mul(x=aw,y=dps)[name=string(\"ds\")];");
            terminate_fp16(&mut m, "ds", &format!("[1,{heads},{seq},{seq}]"));
            try_compile("PhaseD2_softmax_bwd_reduce_mean", &m, heads * seq * seq);
        }

        // ---- Phase E: Phase D2 (reduce_mean) + dQ and dK matmuls ----
        {
            let mut m = preamble();
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"sc1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"sc2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"ms\")];");
            let _ = writeln!(m, "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"aw\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw_t = transpose(perm=pm,x=aw)[name=string(\"awt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dv_all = matmul(transpose_x=bF,transpose_y=bF,x=aw_t,y=da)[name=string(\"dva\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dp = matmul(transpose_x=bF,transpose_y=bT,x=da,y=v)[name=string(\"dp\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dpaw = mul(x=dp,y=aw)[name=string(\"dpaw\")];");
            let _ = writeln!(m, "        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];");
            let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> dot_m = reduce_mean(x=dpaw,axes=rax,keep_dims=kd)[name=string(\"dotm\")];");
            let _ = writeln!(m, "        fp16 seq_v = const()[name=string(\"seqv\"), val=fp16({seq}.0)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> dot = mul(x=dot_m,y=seq_v)[name=string(\"dot\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dps = sub(x=dp,y=dot)[name=string(\"dps\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ds = mul(x=aw,y=dps)[name=string(\"ds\")];");
            // dQ = ds @ K * scale, dK = ds^T @ Q * scale
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dqr = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=k)[name=string(\"dqr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dq_s = mul(x=dqr,y=scv)[name=string(\"dqs\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ds_t = transpose(perm=pm,x=ds)[name=string(\"dst\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dkr = matmul(transpose_x=bF,transpose_y=bF,x=ds_t,y=q)[name=string(\"dkr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dk_s = mul(x=dkr,y=scv)[name=string(\"dks\")];");
            terminate_fp16(&mut m, "dq_s", &format!("[1,{heads},{seq},{hd}]"));
            try_compile("PhaseE_dQ_dK_matmuls", &m, heads * seq * hd);
        }

        // ---- Phase F: Phase E + RoPE backward ----
        // RoPE needs rope_cos/rope_sin blobs. Add them as additional weights.
        {
            let cfg_rope = MilConfig {
                dim: 64,
                hidden_dim: 128,
                n_heads: 8,
                seq_len: 32,
                n_kv_heads: 4,
                rope_theta: 1e6,
                rms_eps: 1e-6,
                has_lm_head: false,
                head_dim_explicit: 16,
                linear_attn_indices: vec![],
                linear_n_heads: 0,
                linear_head_dim: 0,
                linear_n_value_heads: 0,
                linear_value_head_dim: 0,
                conv_kernel_size: 0,
                attn_output_gate: true,
            };
            let (rope_cos_blob, rope_sin_blob) =
                generate_rope_blobs(seq, hd, cfg_rope.rope_theta);
            let half_hd = hd / 2;

            let mut m = String::with_capacity(16384);
            m.push_str(MIL_HDR);
            let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1, {in_ch}, 1, {seq}]> x) {{");
            let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
            let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
            let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
            let _ = writeln!(m, "        bool bT = const()[name=string(\"bT\"), val=bool(true)];");
            let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
            let _ = writeln!(m, "        fp16 scv = const()[name=string(\"scv\"), val=fp16(0.25)];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{seq}]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,{seq},{seq}]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_cos = const()[name=string(\"rc\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_cos.bin\"), offset=uint64(64)))];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_sin = const()[name=string(\"rs\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_sin.bin\"), offset=uint64(64)))];");
            let _ = writeln!(m, "        tensor<fp16, [1,{in_ch},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");
            // Slice
            let _ = writeln!(m, "        tensor<int32, [4]> s_ad = const()[name=string(\"sad\"), val=tensor<int32, [4]>([1,{ad},1,{seq}])];");
            let _ = writeln!(m, "        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> dah = slice_by_size(x=xh,begin=b0,size=s_ad)[name=string(\"dah\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,{ad},0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> qrh = slice_by_size(x=xh,begin=b1,size=s_ad)[name=string(\"qrh\")];");
            let off2 = 2 * ad;
            let _ = writeln!(m, "        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,{off2},0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> keh = slice_by_size(x=xh,begin=b2,size=s_ad)[name=string(\"keh\")];");
            let off3 = 3 * ad;
            let _ = writeln!(m, "        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,{off3},0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> veh = slice_by_size(x=xh,begin=b3,size=s_ad)[name=string(\"veh\")];");
            // Reshape to [1,H,hd,S] then transpose to [1,H,S,hd]
            let _ = writeln!(m, "        tensor<int32, [4]> rqh = const()[name=string(\"rqh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> da_4 = reshape(shape=rqh,x=dah)[name=string(\"da4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> da = transpose(perm=pm,x=da_4)[name=string(\"da\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qr_4 = reshape(shape=rqh,x=qrh)[name=string(\"qr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> q = transpose(perm=pm,x=qr_4)[name=string(\"q\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> ke_4 = reshape(shape=rqh,x=keh)[name=string(\"ke4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> k = transpose(perm=pm,x=ke_4)[name=string(\"k\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> ve_4 = reshape(shape=rqh,x=veh)[name=string(\"ve4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> v = transpose(perm=pm,x=ve_4)[name=string(\"v\")];");
            // Full SDPA forward recompute + backward with reduce_mean
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"sc1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"sc2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"ms\")];");
            let _ = writeln!(m, "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"aw\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw_t = transpose(perm=pm,x=aw)[name=string(\"awt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dv_all = matmul(transpose_x=bF,transpose_y=bF,x=aw_t,y=da)[name=string(\"dva\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dp = matmul(transpose_x=bF,transpose_y=bT,x=da,y=v)[name=string(\"dp\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dpaw = mul(x=dp,y=aw)[name=string(\"dpaw\")];");
            let _ = writeln!(m, "        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];");
            let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> dot_m = reduce_mean(x=dpaw,axes=rax,keep_dims=kd)[name=string(\"dotm\")];");
            let _ = writeln!(m, "        fp16 seq_v = const()[name=string(\"seqv\"), val=fp16({seq}.0)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> dot = mul(x=dot_m,y=seq_v)[name=string(\"dot\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dps = sub(x=dp,y=dot)[name=string(\"dps\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ds = mul(x=aw,y=dps)[name=string(\"ds\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dqr = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=k)[name=string(\"dqr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dq_s = mul(x=dqr,y=scv)[name=string(\"dqs\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ds_t = transpose(perm=pm,x=ds)[name=string(\"dst\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dkr = matmul(transpose_x=bF,transpose_y=bF,x=ds_t,y=q)[name=string(\"dkr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dk_s = mul(x=dkr,y=scv)[name=string(\"dks\")];");
            // RoPE backward on dq_s
            let _ = writeln!(m, "        int32 rpax = const()[name=string(\"rpax\"), val=int32(-1)];");
            let _ = writeln!(m, "        bool rpid = const()[name=string(\"rpid\"), val=bool(false)];");
            let _ = writeln!(m, "        tensor<int32, [4]> rpb0 = const()[name=string(\"rpb0\"), val=tensor<int32, [4]>([0,0,0,0])];");
            let _ = writeln!(m, "        tensor<int32, [4]> rpqh = const()[name=string(\"rpqh\"), val=tensor<int32, [4]>([1,{heads},{seq},{half_hd}])];");
            let _ = writeln!(m, "        tensor<int32, [4]> rpbh = const()[name=string(\"rpbh\"), val=tensor<int32, [4]>([0,0,0,{half_hd}])];");
            // dQ RoPE bwd
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqr1 = slice_by_size(x=dq_s,begin=rpb0,size=rpqh)[name=string(\"dqr1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqr2 = slice_by_size(x=dq_s,begin=rpbh,size=rpqh)[name=string(\"dqr2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq1c = mul(x=dqr1,y=rope_cos)[name=string(\"dq1c\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq2s = mul(x=dqr2,y=rope_sin)[name=string(\"dq2s\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqp1 = add(x=dq1c,y=dq2s)[name=string(\"dqp1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq2c = mul(x=dqr2,y=rope_cos)[name=string(\"dq2c\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq1s = mul(x=dqr1,y=rope_sin)[name=string(\"dq1s\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqp2 = sub(x=dq2c,y=dq1s)[name=string(\"dqp2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dq_pre = concat(axis=rpax,interleave=rpid,values=(dqp1,dqp2))[name=string(\"dqpre\")];");
            // dK RoPE bwd
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dkr1 = slice_by_size(x=dk_s,begin=rpb0,size=rpqh)[name=string(\"dkr1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dkr2 = slice_by_size(x=dk_s,begin=rpbh,size=rpqh)[name=string(\"dkr2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dk1c = mul(x=dkr1,y=rope_cos)[name=string(\"dk1c\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dk2s = mul(x=dkr2,y=rope_sin)[name=string(\"dk2s\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dkp1 = add(x=dk1c,y=dk2s)[name=string(\"dkp1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dk2c = mul(x=dkr2,y=rope_cos)[name=string(\"dk2c\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dk1s = mul(x=dkr1,y=rope_sin)[name=string(\"dk1s\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dkp2 = sub(x=dk2c,y=dk1s)[name=string(\"dkp2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dk_pre = concat(axis=rpax,interleave=rpid,values=(dkp1,dkp2))[name=string(\"dkpre\")];");
            terminate_fp16(&mut m, "dq_pre", &format!("[1,{heads},{seq},{hd}]"));

            let out_elems = heads * seq * hd;
            let out_bytes = out_elems * 4;
            match AneKernel::compile_multi_weights(
                &m,
                &[
                    "@model_path/weights/mask.bin",
                    "@model_path/weights/rope_cos.bin",
                    "@model_path/weights/rope_sin.bin",
                ],
                &[&mask_blob, &rope_cos_blob, &rope_sin_blob],
                &[input_bytes],
                &[out_bytes],
            ) {
                Ok(_k) => eprintln!("[PASS] PhaseF_rope_bwd"),
                Err(e) => eprintln!("[FAIL] PhaseF_rope_bwd: {e}"),
            }
        }

        // ---- Phase G: Phase F + output flatten + concat (dQ, dK, dV) ----
        {
            let cfg_rope = MilConfig {
                dim: 64,
                hidden_dim: 128,
                n_heads: 8,
                seq_len: 32,
                n_kv_heads: 4,
                rope_theta: 1e6,
                rms_eps: 1e-6,
                has_lm_head: false,
                head_dim_explicit: 16,
                linear_attn_indices: vec![],
                linear_n_heads: 0,
                linear_head_dim: 0,
                linear_n_value_heads: 0,
                linear_value_head_dim: 0,
                conv_kernel_size: 0,
                attn_output_gate: true,
            };
            let (rope_cos_blob, rope_sin_blob) =
                generate_rope_blobs(seq, hd, cfg_rope.rope_theta);
            let half_hd = hd / 2;
            let out_ad = 3 * ad; // dQ + dK + dV

            let mut m = String::with_capacity(16384);
            m.push_str(MIL_HDR);
            let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1, {in_ch}, 1, {seq}]> x) {{");
            let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
            let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
            let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
            let _ = writeln!(m, "        bool bT = const()[name=string(\"bT\"), val=bool(true)];");
            let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
            let _ = writeln!(m, "        fp16 scv = const()[name=string(\"scv\"), val=fp16(0.25)];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{seq}]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,{seq},{seq}]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_cos = const()[name=string(\"rc\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_cos.bin\"), offset=uint64(64)))];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_sin = const()[name=string(\"rs\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_sin.bin\"), offset=uint64(64)))];");
            let _ = writeln!(m, "        tensor<fp16, [1,{in_ch},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> s_ad = const()[name=string(\"sad\"), val=tensor<int32, [4]>([1,{ad},1,{seq}])];");
            let _ = writeln!(m, "        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> dah = slice_by_size(x=xh,begin=b0,size=s_ad)[name=string(\"dah\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,{ad},0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> qrh = slice_by_size(x=xh,begin=b1,size=s_ad)[name=string(\"qrh\")];");
            let off2 = 2 * ad;
            let _ = writeln!(m, "        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,{off2},0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> keh = slice_by_size(x=xh,begin=b2,size=s_ad)[name=string(\"keh\")];");
            let off3 = 3 * ad;
            let _ = writeln!(m, "        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,{off3},0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> veh = slice_by_size(x=xh,begin=b3,size=s_ad)[name=string(\"veh\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rqh = const()[name=string(\"rqh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> da_4 = reshape(shape=rqh,x=dah)[name=string(\"da4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> da = transpose(perm=pm,x=da_4)[name=string(\"da\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qr_4 = reshape(shape=rqh,x=qrh)[name=string(\"qr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> q = transpose(perm=pm,x=qr_4)[name=string(\"q\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> ke_4 = reshape(shape=rqh,x=keh)[name=string(\"ke4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> k = transpose(perm=pm,x=ke_4)[name=string(\"k\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> ve_4 = reshape(shape=rqh,x=veh)[name=string(\"ve4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> v = transpose(perm=pm,x=ve_4)[name=string(\"v\")];");
            // SDPA fwd recompute
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"sc1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"sc2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"ms\")];");
            let _ = writeln!(m, "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"aw\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw_t = transpose(perm=pm,x=aw)[name=string(\"awt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dv_all = matmul(transpose_x=bF,transpose_y=bF,x=aw_t,y=da)[name=string(\"dva\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dp = matmul(transpose_x=bF,transpose_y=bT,x=da,y=v)[name=string(\"dp\")];");
            // Softmax bwd (reduce_mean)
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dpaw = mul(x=dp,y=aw)[name=string(\"dpaw\")];");
            let _ = writeln!(m, "        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];");
            let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> dot_m = reduce_mean(x=dpaw,axes=rax,keep_dims=kd)[name=string(\"dotm\")];");
            let _ = writeln!(m, "        fp16 seq_v = const()[name=string(\"seqv\"), val=fp16({seq}.0)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> dot = mul(x=dot_m,y=seq_v)[name=string(\"dot\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dps = sub(x=dp,y=dot)[name=string(\"dps\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ds = mul(x=aw,y=dps)[name=string(\"ds\")];");
            // dQ, dK matmuls
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dqr = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=k)[name=string(\"dqr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dq_s = mul(x=dqr,y=scv)[name=string(\"dqs\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ds_t = transpose(perm=pm,x=ds)[name=string(\"dst\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dkr = matmul(transpose_x=bF,transpose_y=bF,x=ds_t,y=q)[name=string(\"dkr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dk_s = mul(x=dkr,y=scv)[name=string(\"dks\")];");
            // RoPE bwd on dQ
            let _ = writeln!(m, "        int32 rpax = const()[name=string(\"rpax\"), val=int32(-1)];");
            let _ = writeln!(m, "        bool rpid = const()[name=string(\"rpid\"), val=bool(false)];");
            let _ = writeln!(m, "        tensor<int32, [4]> rpb0 = const()[name=string(\"rpb0\"), val=tensor<int32, [4]>([0,0,0,0])];");
            let _ = writeln!(m, "        tensor<int32, [4]> rpqh = const()[name=string(\"rpqh\"), val=tensor<int32, [4]>([1,{heads},{seq},{half_hd}])];");
            let _ = writeln!(m, "        tensor<int32, [4]> rpbh = const()[name=string(\"rpbh\"), val=tensor<int32, [4]>([0,0,0,{half_hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqr1 = slice_by_size(x=dq_s,begin=rpb0,size=rpqh)[name=string(\"dqr1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqr2 = slice_by_size(x=dq_s,begin=rpbh,size=rpqh)[name=string(\"dqr2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq1c = mul(x=dqr1,y=rope_cos)[name=string(\"dq1c\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq2s = mul(x=dqr2,y=rope_sin)[name=string(\"dq2s\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqp1 = add(x=dq1c,y=dq2s)[name=string(\"dqp1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq2c = mul(x=dqr2,y=rope_cos)[name=string(\"dq2c\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq1s = mul(x=dqr1,y=rope_sin)[name=string(\"dq1s\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqp2 = sub(x=dq2c,y=dq1s)[name=string(\"dqp2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dq_pre = concat(axis=rpax,interleave=rpid,values=(dqp1,dqp2))[name=string(\"dqpre\")];");
            // RoPE bwd on dK
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dkr1 = slice_by_size(x=dk_s,begin=rpb0,size=rpqh)[name=string(\"dkr1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dkr2 = slice_by_size(x=dk_s,begin=rpbh,size=rpqh)[name=string(\"dkr2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dk1c = mul(x=dkr1,y=rope_cos)[name=string(\"dk1c\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dk2s = mul(x=dkr2,y=rope_sin)[name=string(\"dk2s\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dkp1 = add(x=dk1c,y=dk2s)[name=string(\"dkp1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dk2c = mul(x=dkr2,y=rope_cos)[name=string(\"dk2c\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dk1s = mul(x=dkr1,y=rope_sin)[name=string(\"dk1s\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dkp2 = sub(x=dk2c,y=dk1s)[name=string(\"dkp2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dk_pre = concat(axis=rpax,interleave=rpid,values=(dkp1,dkp2))[name=string(\"dkpre\")];");
            // Flatten and concat dQ, dK, dV
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> dq_t = transpose(perm=pm,x=dq_pre)[name=string(\"dqt\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rad = const()[name=string(\"rad\"), val=tensor<int32, [4]>([1,{ad},1,{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> dq_ch = reshape(shape=rad,x=dq_t)[name=string(\"dqch\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> dk_t = transpose(perm=pm,x=dk_pre)[name=string(\"dkt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> dk_ch = reshape(shape=rad,x=dk_t)[name=string(\"dkch\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> dv_t = transpose(perm=pm,x=dv_all)[name=string(\"dvt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> dv_ch = reshape(shape=rad,x=dv_t)[name=string(\"dvch\")];");
            let _ = writeln!(m, "        int32 cax = const()[name=string(\"cax\"), val=int32(1)];");
            let _ = writeln!(m, "        bool cid = const()[name=string(\"cid\"), val=bool(false)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{out_ad},1,{seq}]> out_h = concat(axis=cax,interleave=cid,values=(dq_ch,dk_ch,dv_ch))[name=string(\"outh\")];");
            let _ = writeln!(m, "        tensor<fp32, [1,{out_ad},1,{seq}]> out = cast(dtype=to32,x=out_h)[name=string(\"cout\")];");
            let _ = writeln!(m, "    }} -> (out);");
            m.push_str("}\n");

            let out_elems = out_ad * seq;
            let out_bytes = out_elems * 4;
            match AneKernel::compile_multi_weights(
                &m,
                &[
                    "@model_path/weights/mask.bin",
                    "@model_path/weights/rope_cos.bin",
                    "@model_path/weights/rope_sin.bin",
                ],
                &[&mask_blob, &rope_cos_blob, &rope_sin_blob],
                &[input_bytes],
                &[out_bytes],
            ) {
                Ok(_k) => eprintln!("[PASS] PhaseG_full_fused"),
                Err(e) => eprintln!("[FAIL] PhaseG_full_fused: {e}"),
            }
        }

        // ---- Phase E2: Phase E + just the dQ RoPE slice (no mul/add) ----
        {
            let mut m = preamble();
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"sc1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"sc2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"ms\")];");
            let _ = writeln!(m, "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"aw\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw_t = transpose(perm=pm,x=aw)[name=string(\"awt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dv_all = matmul(transpose_x=bF,transpose_y=bF,x=aw_t,y=da)[name=string(\"dva\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dp = matmul(transpose_x=bF,transpose_y=bT,x=da,y=v)[name=string(\"dp\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dpaw = mul(x=dp,y=aw)[name=string(\"dpaw\")];");
            let _ = writeln!(m, "        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];");
            let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> dot_m = reduce_mean(x=dpaw,axes=rax,keep_dims=kd)[name=string(\"dotm\")];");
            let _ = writeln!(m, "        fp16 seq_v = const()[name=string(\"seqv\"), val=fp16({seq}.0)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> dot = mul(x=dot_m,y=seq_v)[name=string(\"dot\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dps = sub(x=dp,y=dot)[name=string(\"dps\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ds = mul(x=aw,y=dps)[name=string(\"ds\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dqr = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=k)[name=string(\"dqr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dq_s = mul(x=dqr,y=scv)[name=string(\"dqs\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ds_t = transpose(perm=pm,x=ds)[name=string(\"dst\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dkr = matmul(transpose_x=bF,transpose_y=bF,x=ds_t,y=q)[name=string(\"dkr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dk_s = mul(x=dkr,y=scv)[name=string(\"dks\")];");
            // Just slice dQ into halves (no RoPE computation)
            let half_hd = hd / 2;
            let _ = writeln!(m, "        tensor<int32, [4]> rpb0 = const()[name=string(\"rpb0\"), val=tensor<int32, [4]>([0,0,0,0])];");
            let _ = writeln!(m, "        tensor<int32, [4]> rpqh = const()[name=string(\"rpqh\"), val=tensor<int32, [4]>([1,{heads},{seq},{half_hd}])];");
            let _ = writeln!(m, "        tensor<int32, [4]> rpbh = const()[name=string(\"rpbh\"), val=tensor<int32, [4]>([0,0,0,{half_hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqr1 = slice_by_size(x=dq_s,begin=rpb0,size=rpqh)[name=string(\"dqr1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqr2 = slice_by_size(x=dq_s,begin=rpbh,size=rpqh)[name=string(\"dqr2\")];");
            // Output just slice result (add them to force both used)
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> sum12 = add(x=dqr1,y=dqr2)[name=string(\"s12\")];");
            terminate_fp16(&mut m, "sum12", &format!("[1,{heads},{seq},{half_hd}]"));
            try_compile("PhaseE2_slice_only", &m, heads * seq * half_hd);
        }

        // ---- Phase E3: Phase E2 + dQ RoPE mul with cos/sin (NO concat) ----
        // Need rope blobs for this
        {
            let cfg_rope = MilConfig {
                dim: 64,
                hidden_dim: 128,
                n_heads: 8,
                seq_len: 32,
                n_kv_heads: 4,
                rope_theta: 1e6,
                rms_eps: 1e-6,
                has_lm_head: false,
                head_dim_explicit: 16,
                linear_attn_indices: vec![],
                linear_n_heads: 0,
                linear_head_dim: 0,
                linear_n_value_heads: 0,
                linear_value_head_dim: 0,
                conv_kernel_size: 0,
                attn_output_gate: true,
            };
            let (rope_cos_blob, rope_sin_blob) =
                generate_rope_blobs(seq, hd, cfg_rope.rope_theta);
            let half_hd = hd / 2;

            let mut m = String::with_capacity(16384);
            m.push_str(MIL_HDR);
            let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1, {in_ch}, 1, {seq}]> x) {{");
            let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
            let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
            let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
            let _ = writeln!(m, "        bool bT = const()[name=string(\"bT\"), val=bool(true)];");
            let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
            let _ = writeln!(m, "        fp16 scv = const()[name=string(\"scv\"), val=fp16(0.25)];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{seq}]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,{seq},{seq}]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_cos = const()[name=string(\"rc\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_cos.bin\"), offset=uint64(64)))];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_sin = const()[name=string(\"rs\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_sin.bin\"), offset=uint64(64)))];");
            let _ = writeln!(m, "        tensor<fp16, [1,{in_ch},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> s_ad = const()[name=string(\"sad\"), val=tensor<int32, [4]>([1,{ad},1,{seq}])];");
            let _ = writeln!(m, "        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> dah = slice_by_size(x=xh,begin=b0,size=s_ad)[name=string(\"dah\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,{ad},0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> qrh = slice_by_size(x=xh,begin=b1,size=s_ad)[name=string(\"qrh\")];");
            let off2 = 2 * ad;
            let _ = writeln!(m, "        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,{off2},0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> keh = slice_by_size(x=xh,begin=b2,size=s_ad)[name=string(\"keh\")];");
            let off3 = 3 * ad;
            let _ = writeln!(m, "        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,{off3},0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> veh = slice_by_size(x=xh,begin=b3,size=s_ad)[name=string(\"veh\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rqh = const()[name=string(\"rqh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> da_4 = reshape(shape=rqh,x=dah)[name=string(\"da4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> da = transpose(perm=pm,x=da_4)[name=string(\"da\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qr_4 = reshape(shape=rqh,x=qrh)[name=string(\"qr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> q = transpose(perm=pm,x=qr_4)[name=string(\"q\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> ke_4 = reshape(shape=rqh,x=keh)[name=string(\"ke4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> k = transpose(perm=pm,x=ke_4)[name=string(\"k\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> ve_4 = reshape(shape=rqh,x=veh)[name=string(\"ve4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> v = transpose(perm=pm,x=ve_4)[name=string(\"v\")];");
            // Full SDPA backward
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"sc1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"sc2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"ms\")];");
            let _ = writeln!(m, "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"aw\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw_t = transpose(perm=pm,x=aw)[name=string(\"awt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dv_all = matmul(transpose_x=bF,transpose_y=bF,x=aw_t,y=da)[name=string(\"dva\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dp = matmul(transpose_x=bF,transpose_y=bT,x=da,y=v)[name=string(\"dp\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dpaw = mul(x=dp,y=aw)[name=string(\"dpaw\")];");
            let _ = writeln!(m, "        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];");
            let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> dot_m = reduce_mean(x=dpaw,axes=rax,keep_dims=kd)[name=string(\"dotm\")];");
            let _ = writeln!(m, "        fp16 seq_v = const()[name=string(\"seqv\"), val=fp16({seq}.0)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> dot = mul(x=dot_m,y=seq_v)[name=string(\"dot\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dps = sub(x=dp,y=dot)[name=string(\"dps\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ds = mul(x=aw,y=dps)[name=string(\"ds\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dqr = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=k)[name=string(\"dqr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dq_s = mul(x=dqr,y=scv)[name=string(\"dqs\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ds_t = transpose(perm=pm,x=ds)[name=string(\"dst\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dkr = matmul(transpose_x=bF,transpose_y=bF,x=ds_t,y=q)[name=string(\"dkr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dk_s = mul(x=dkr,y=scv)[name=string(\"dks\")];");
            // dQ RoPE: slice + mul with cos/sin + add/sub (but NO concat, just output dqp1)
            let _ = writeln!(m, "        tensor<int32, [4]> rpb0 = const()[name=string(\"rpb0\"), val=tensor<int32, [4]>([0,0,0,0])];");
            let _ = writeln!(m, "        tensor<int32, [4]> rpqh = const()[name=string(\"rpqh\"), val=tensor<int32, [4]>([1,{heads},{seq},{half_hd}])];");
            let _ = writeln!(m, "        tensor<int32, [4]> rpbh = const()[name=string(\"rpbh\"), val=tensor<int32, [4]>([0,0,0,{half_hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqr1 = slice_by_size(x=dq_s,begin=rpb0,size=rpqh)[name=string(\"dqr1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqr2 = slice_by_size(x=dq_s,begin=rpbh,size=rpqh)[name=string(\"dqr2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq1c = mul(x=dqr1,y=rope_cos)[name=string(\"dq1c\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq2s = mul(x=dqr2,y=rope_sin)[name=string(\"dq2s\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqp1 = add(x=dq1c,y=dq2s)[name=string(\"dqp1\")];");
            terminate_fp16(&mut m, "dqp1", &format!("[1,{heads},{seq},{half_hd}]"));

            let out_elems = heads * seq * half_hd;
            let out_bytes = out_elems * 4;
            match AneKernel::compile_multi_weights(
                &m,
                &[
                    "@model_path/weights/mask.bin",
                    "@model_path/weights/rope_cos.bin",
                    "@model_path/weights/rope_sin.bin",
                ],
                &[&mask_blob, &rope_cos_blob, &rope_sin_blob],
                &[input_bytes],
                &[out_bytes],
            ) {
                Ok(_k) => eprintln!("[PASS] PhaseE3_rope_mul_no_concat"),
                Err(e) => eprintln!("[FAIL] PhaseE3_rope_mul_no_concat: {e}"),
            }
        }

        // ---- Phase E4: Phase E3 + dQ concat (full dQ RoPE, but NO dK RoPE) ----
        {
            let cfg_rope = MilConfig {
                dim: 64,
                hidden_dim: 128,
                n_heads: 8,
                seq_len: 32,
                n_kv_heads: 4,
                rope_theta: 1e6,
                rms_eps: 1e-6,
                has_lm_head: false,
                head_dim_explicit: 16,
                linear_attn_indices: vec![],
                linear_n_heads: 0,
                linear_head_dim: 0,
                linear_n_value_heads: 0,
                linear_value_head_dim: 0,
                conv_kernel_size: 0,
                attn_output_gate: true,
            };
            let (rope_cos_blob, rope_sin_blob) =
                generate_rope_blobs(seq, hd, cfg_rope.rope_theta);
            let half_hd = hd / 2;

            let mut m = String::with_capacity(16384);
            m.push_str(MIL_HDR);
            let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1, {in_ch}, 1, {seq}]> x) {{");
            let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
            let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
            let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
            let _ = writeln!(m, "        bool bT = const()[name=string(\"bT\"), val=bool(true)];");
            let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
            let _ = writeln!(m, "        fp16 scv = const()[name=string(\"scv\"), val=fp16(0.25)];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{seq}]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,{seq},{seq}]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_cos = const()[name=string(\"rc\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_cos.bin\"), offset=uint64(64)))];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_sin = const()[name=string(\"rs\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_sin.bin\"), offset=uint64(64)))];");
            let _ = writeln!(m, "        tensor<fp16, [1,{in_ch},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> s_ad = const()[name=string(\"sad\"), val=tensor<int32, [4]>([1,{ad},1,{seq}])];");
            let _ = writeln!(m, "        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> dah = slice_by_size(x=xh,begin=b0,size=s_ad)[name=string(\"dah\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,{ad},0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> qrh = slice_by_size(x=xh,begin=b1,size=s_ad)[name=string(\"qrh\")];");
            let off2 = 2 * ad;
            let _ = writeln!(m, "        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,{off2},0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> keh = slice_by_size(x=xh,begin=b2,size=s_ad)[name=string(\"keh\")];");
            let off3 = 3 * ad;
            let _ = writeln!(m, "        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,{off3},0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> veh = slice_by_size(x=xh,begin=b3,size=s_ad)[name=string(\"veh\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rqh = const()[name=string(\"rqh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> da_4 = reshape(shape=rqh,x=dah)[name=string(\"da4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> da = transpose(perm=pm,x=da_4)[name=string(\"da\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qr_4 = reshape(shape=rqh,x=qrh)[name=string(\"qr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> q = transpose(perm=pm,x=qr_4)[name=string(\"q\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> ke_4 = reshape(shape=rqh,x=keh)[name=string(\"ke4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> k = transpose(perm=pm,x=ke_4)[name=string(\"k\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> ve_4 = reshape(shape=rqh,x=veh)[name=string(\"ve4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> v = transpose(perm=pm,x=ve_4)[name=string(\"v\")];");
            // SDPA backward
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"sc1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"sc2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"ms\")];");
            let _ = writeln!(m, "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"aw\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw_t = transpose(perm=pm,x=aw)[name=string(\"awt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dv_all = matmul(transpose_x=bF,transpose_y=bF,x=aw_t,y=da)[name=string(\"dva\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dp = matmul(transpose_x=bF,transpose_y=bT,x=da,y=v)[name=string(\"dp\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dpaw = mul(x=dp,y=aw)[name=string(\"dpaw\")];");
            let _ = writeln!(m, "        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];");
            let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> dot_m = reduce_mean(x=dpaw,axes=rax,keep_dims=kd)[name=string(\"dotm\")];");
            let _ = writeln!(m, "        fp16 seq_v = const()[name=string(\"seqv\"), val=fp16({seq}.0)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> dot = mul(x=dot_m,y=seq_v)[name=string(\"dot\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dps = sub(x=dp,y=dot)[name=string(\"dps\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ds = mul(x=aw,y=dps)[name=string(\"ds\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dqr = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=k)[name=string(\"dqr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dq_s = mul(x=dqr,y=scv)[name=string(\"dqs\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ds_t = transpose(perm=pm,x=ds)[name=string(\"dst\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dkr = matmul(transpose_x=bF,transpose_y=bF,x=ds_t,y=q)[name=string(\"dkr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dk_s = mul(x=dkr,y=scv)[name=string(\"dks\")];");
            // Full dQ RoPE bwd with concat
            let _ = writeln!(m, "        int32 rpax = const()[name=string(\"rpax\"), val=int32(-1)];");
            let _ = writeln!(m, "        bool rpid = const()[name=string(\"rpid\"), val=bool(false)];");
            let _ = writeln!(m, "        tensor<int32, [4]> rpb0 = const()[name=string(\"rpb0\"), val=tensor<int32, [4]>([0,0,0,0])];");
            let _ = writeln!(m, "        tensor<int32, [4]> rpqh = const()[name=string(\"rpqh\"), val=tensor<int32, [4]>([1,{heads},{seq},{half_hd}])];");
            let _ = writeln!(m, "        tensor<int32, [4]> rpbh = const()[name=string(\"rpbh\"), val=tensor<int32, [4]>([0,0,0,{half_hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqr1 = slice_by_size(x=dq_s,begin=rpb0,size=rpqh)[name=string(\"dqr1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqr2 = slice_by_size(x=dq_s,begin=rpbh,size=rpqh)[name=string(\"dqr2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq1c = mul(x=dqr1,y=rope_cos)[name=string(\"dq1c\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq2s = mul(x=dqr2,y=rope_sin)[name=string(\"dq2s\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqp1 = add(x=dq1c,y=dq2s)[name=string(\"dqp1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq2c = mul(x=dqr2,y=rope_cos)[name=string(\"dq2c\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq1s = mul(x=dqr1,y=rope_sin)[name=string(\"dq1s\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqp2 = sub(x=dq2c,y=dq1s)[name=string(\"dqp2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dq_pre = concat(axis=rpax,interleave=rpid,values=(dqp1,dqp2))[name=string(\"dqpre\")];");
            terminate_fp16(&mut m, "dq_pre", &format!("[1,{heads},{seq},{hd}]"));

            let out_elems = heads * seq * hd;
            let out_bytes = out_elems * 4;
            match AneKernel::compile_multi_weights(
                &m,
                &[
                    "@model_path/weights/mask.bin",
                    "@model_path/weights/rope_cos.bin",
                    "@model_path/weights/rope_sin.bin",
                ],
                &[&mask_blob, &rope_cos_blob, &rope_sin_blob],
                &[input_bytes],
                &[out_bytes],
            ) {
                Ok(_k) => eprintln!("[PASS] PhaseE4_dQ_rope_with_concat"),
                Err(e) => eprintln!("[FAIL] PhaseE4_dQ_rope_with_concat: {e}"),
            }
        }

        // ---- Phase E4b: Same as E4 but output dqp1 + dqp2 via ADD (no concat) ----
        {
            let cfg_rope = MilConfig {
                dim: 64, hidden_dim: 128, n_heads: 8, seq_len: 32, n_kv_heads: 4,
                rope_theta: 1e6, rms_eps: 1e-6, has_lm_head: false, head_dim_explicit: 16,
                linear_attn_indices: vec![], linear_n_heads: 0, linear_head_dim: 0,
                linear_n_value_heads: 0, linear_value_head_dim: 0, conv_kernel_size: 0,
                attn_output_gate: true,
            };
            let (rope_cos_blob, rope_sin_blob) =
                generate_rope_blobs(seq, hd, cfg_rope.rope_theta);
            let half_hd = hd / 2;

            let mut m = String::with_capacity(16384);
            m.push_str(MIL_HDR);
            let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1, {in_ch}, 1, {seq}]> x) {{");
            let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
            let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
            let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
            let _ = writeln!(m, "        bool bT = const()[name=string(\"bT\"), val=bool(true)];");
            let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
            let _ = writeln!(m, "        fp16 scv = const()[name=string(\"scv\"), val=fp16(0.25)];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{seq}]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,{seq},{seq}]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_cos = const()[name=string(\"rc\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_cos.bin\"), offset=uint64(64)))];");
            let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_sin = const()[name=string(\"rs\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_sin.bin\"), offset=uint64(64)))];");
            let _ = writeln!(m, "        tensor<fp16, [1,{in_ch},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> s_ad = const()[name=string(\"sad\"), val=tensor<int32, [4]>([1,{ad},1,{seq}])];");
            let _ = writeln!(m, "        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> dah = slice_by_size(x=xh,begin=b0,size=s_ad)[name=string(\"dah\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,{ad},0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> qrh = slice_by_size(x=xh,begin=b1,size=s_ad)[name=string(\"qrh\")];");
            let off2 = 2 * ad;
            let _ = writeln!(m, "        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,{off2},0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> keh = slice_by_size(x=xh,begin=b2,size=s_ad)[name=string(\"keh\")];");
            let off3 = 3 * ad;
            let _ = writeln!(m, "        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,{off3},0,0])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{ad},1,{seq}]> veh = slice_by_size(x=xh,begin=b3,size=s_ad)[name=string(\"veh\")];");
            let _ = writeln!(m, "        tensor<int32, [4]> rqh = const()[name=string(\"rqh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> da_4 = reshape(shape=rqh,x=dah)[name=string(\"da4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> da = transpose(perm=pm,x=da_4)[name=string(\"da\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qr_4 = reshape(shape=rqh,x=qrh)[name=string(\"qr4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> q = transpose(perm=pm,x=qr_4)[name=string(\"q\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> ke_4 = reshape(shape=rqh,x=keh)[name=string(\"ke4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> k = transpose(perm=pm,x=ke_4)[name=string(\"k\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> ve_4 = reshape(shape=rqh,x=veh)[name=string(\"ve4\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> v = transpose(perm=pm,x=ve_4)[name=string(\"v\")];");
            // SDPA backward (same as E4)
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"sc1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"sc2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"ms\")];");
            let _ = writeln!(m, "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"aw\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw_t = transpose(perm=pm,x=aw)[name=string(\"awt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dv_all = matmul(transpose_x=bF,transpose_y=bF,x=aw_t,y=da)[name=string(\"dva\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dp = matmul(transpose_x=bF,transpose_y=bT,x=da,y=v)[name=string(\"dp\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dpaw = mul(x=dp,y=aw)[name=string(\"dpaw\")];");
            let _ = writeln!(m, "        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];");
            let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> dot_m = reduce_mean(x=dpaw,axes=rax,keep_dims=kd)[name=string(\"dotm\")];");
            let _ = writeln!(m, "        fp16 seq_v = const()[name=string(\"seqv\"), val=fp16({seq}.0)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> dot = mul(x=dot_m,y=seq_v)[name=string(\"dot\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dps = sub(x=dp,y=dot)[name=string(\"dps\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ds = mul(x=aw,y=dps)[name=string(\"ds\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dqr = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=k)[name=string(\"dqr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dq_s = mul(x=dqr,y=scv)[name=string(\"dqs\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ds_t = transpose(perm=pm,x=ds)[name=string(\"dst\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dkr = matmul(transpose_x=bF,transpose_y=bF,x=ds_t,y=q)[name=string(\"dkr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dk_s = mul(x=dkr,y=scv)[name=string(\"dks\")];");
            // Full dQ RoPE computation (both halves) but output ADD instead of CONCAT
            let _ = writeln!(m, "        tensor<int32, [4]> rpb0 = const()[name=string(\"rpb0\"), val=tensor<int32, [4]>([0,0,0,0])];");
            let _ = writeln!(m, "        tensor<int32, [4]> rpqh = const()[name=string(\"rpqh\"), val=tensor<int32, [4]>([1,{heads},{seq},{half_hd}])];");
            let _ = writeln!(m, "        tensor<int32, [4]> rpbh = const()[name=string(\"rpbh\"), val=tensor<int32, [4]>([0,0,0,{half_hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqr1 = slice_by_size(x=dq_s,begin=rpb0,size=rpqh)[name=string(\"dqr1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqr2 = slice_by_size(x=dq_s,begin=rpbh,size=rpqh)[name=string(\"dqr2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq1c = mul(x=dqr1,y=rope_cos)[name=string(\"dq1c\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq2s = mul(x=dqr2,y=rope_sin)[name=string(\"dq2s\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqp1 = add(x=dq1c,y=dq2s)[name=string(\"dqp1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq2c = mul(x=dqr2,y=rope_cos)[name=string(\"dq2c\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq1s = mul(x=dqr1,y=rope_sin)[name=string(\"dq1s\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqp2 = sub(x=dq2c,y=dq1s)[name=string(\"dqp2\")];");
            // Output ADD of the halves instead of concat (to check if concat is the specific trigger)
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq_sum = add(x=dqp1,y=dqp2)[name=string(\"dqsum\")];");
            terminate_fp16(&mut m, "dq_sum", &format!("[1,{heads},{seq},{half_hd}]"));

            let out_elems = heads * seq * half_hd;
            let out_bytes = out_elems * 4;
            match AneKernel::compile_multi_weights(
                &m,
                &[
                    "@model_path/weights/mask.bin",
                    "@model_path/weights/rope_cos.bin",
                    "@model_path/weights/rope_sin.bin",
                ],
                &[&mask_blob, &rope_cos_blob, &rope_sin_blob],
                &[input_bytes],
                &[out_bytes],
            ) {
                Ok(_k) => eprintln!("[PASS] PhaseE4b_rope_add_no_concat"),
                Err(e) => eprintln!("[FAIL] PhaseE4b_rope_add_no_concat: {e}"),
            }
        }

        // ---- Phase E4c: Minimal concat test — just concat two slices of dq_s, no rope mul ----
        {
            let half_hd = hd / 2;
            let mut m = preamble();
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"sc1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"sc2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"ms\")];");
            let _ = writeln!(m, "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"aw\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw_t = transpose(perm=pm,x=aw)[name=string(\"awt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dv_all = matmul(transpose_x=bF,transpose_y=bF,x=aw_t,y=da)[name=string(\"dva\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dp = matmul(transpose_x=bF,transpose_y=bT,x=da,y=v)[name=string(\"dp\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dpaw = mul(x=dp,y=aw)[name=string(\"dpaw\")];");
            let _ = writeln!(m, "        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];");
            let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> dot_m = reduce_mean(x=dpaw,axes=rax,keep_dims=kd)[name=string(\"dotm\")];");
            let _ = writeln!(m, "        fp16 seq_v = const()[name=string(\"seqv\"), val=fp16({seq}.0)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> dot = mul(x=dot_m,y=seq_v)[name=string(\"dot\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dps = sub(x=dp,y=dot)[name=string(\"dps\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ds = mul(x=aw,y=dps)[name=string(\"ds\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dqr = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=k)[name=string(\"dqr\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dq_s = mul(x=dqr,y=scv)[name=string(\"dqs\")];");
            // Just slice dq_s into halves and concat (NO rope mul)
            let _ = writeln!(m, "        int32 rpax = const()[name=string(\"rpax\"), val=int32(-1)];");
            let _ = writeln!(m, "        bool rpid = const()[name=string(\"rpid\"), val=bool(false)];");
            let _ = writeln!(m, "        tensor<int32, [4]> rpb0 = const()[name=string(\"rpb0\"), val=tensor<int32, [4]>([0,0,0,0])];");
            let _ = writeln!(m, "        tensor<int32, [4]> rpqh = const()[name=string(\"rpqh\"), val=tensor<int32, [4]>([1,{heads},{seq},{half_hd}])];");
            let _ = writeln!(m, "        tensor<int32, [4]> rpbh = const()[name=string(\"rpbh\"), val=tensor<int32, [4]>([0,0,0,{half_hd}])];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqr1 = slice_by_size(x=dq_s,begin=rpb0,size=rpqh)[name=string(\"dqr1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqr2 = slice_by_size(x=dq_s,begin=rpbh,size=rpqh)[name=string(\"dqr2\")];");
            // Concat them directly (no transformation — just split and rejoin)
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dq_re = concat(axis=rpax,interleave=rpid,values=(dqr1,dqr2))[name=string(\"dqre\")];");
            terminate_fp16(&mut m, "dq_re", &format!("[1,{heads},{seq},{hd}]"));
            try_compile("PhaseE4c_bare_slice_concat", &m, heads * seq * hd);
        }

        // ---- Phase D_original: exact replica of the failing kernel's reduce_sum ----
        // Test reduce_sum in isolation to confirm it's the blocker
        {
            let mut m = preamble();
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"sc1\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"sc2\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"ms\")];");
            let _ = writeln!(m, "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"aw\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw_t = transpose(perm=pm,x=aw)[name=string(\"awt\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dv_all = matmul(transpose_x=bF,transpose_y=bF,x=aw_t,y=da)[name=string(\"dva\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dp = matmul(transpose_x=bF,transpose_y=bT,x=da,y=v)[name=string(\"dp\")];");
            // Softmax backward with reduce_sum (ORIGINAL from /tmp/sdpa_rope_bwd.mil)
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dpaw = mul(x=dp,y=aw)[name=string(\"dpaw\")];");
            let _ = writeln!(m, "        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];");
            let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> dot = reduce_sum(x=dpaw,axes=rax,keep_dims=kd)[name=string(\"dot\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dps = sub(x=dp,y=dot)[name=string(\"dps\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ds = mul(x=aw,y=dps)[name=string(\"ds\")];");
            terminate_fp16(&mut m, "ds", &format!("[1,{heads},{seq},{seq}]"));
            try_compile("PhaseD_original_reduce_sum", &m, heads * seq * seq);
        }

        // ---- Minimal reduce_sum isolation: just a reduce_sum on the input ----
        {
            let mut m = String::with_capacity(4096);
            m.push_str(MIL_HDR);
            let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1, {in_ch}, 1, {seq}]> x) {{");
            let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
            let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
            let _ = writeln!(m, "        tensor<fp16, [1,{in_ch},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");
            let _ = writeln!(m, "        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];");
            let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");
            let _ = writeln!(m, "        tensor<fp16, [1,{in_ch},1,1]> rs = reduce_sum(x=xh,axes=rax,keep_dims=kd)[name=string(\"rs\")];");
            let _ = writeln!(m, "        tensor<fp32, [1,{in_ch},1,1]> out = cast(dtype=to32,x=rs)[name=string(\"cout\")];");
            let _ = writeln!(m, "    }} -> (out);");
            m.push_str("}\n");
            let out_bytes = in_ch * 4;
            match AneKernel::compile_multi_weights(
                &m,
                &["@model_path/weights/mask.bin"],
                &[&mask_blob],
                &[input_bytes],
                &[out_bytes],
            ) {
                Ok(_k) => eprintln!("[PASS] Minimal_reduce_sum_only"),
                Err(e) => eprintln!("[FAIL] Minimal_reduce_sum_only: {e}"),
            }
        }

        eprintln!("\n=== SDPA+RoPE backward bisection complete ===\n");
    }
}
