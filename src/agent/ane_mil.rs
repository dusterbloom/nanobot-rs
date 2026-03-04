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

const MIL_HDR: &str = concat!(
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
    pub dim: usize,            // hidden dimension (768, 1024, etc.)
    pub hidden_dim: usize,     // FFN hidden (2048, etc.)
    pub n_heads: usize,        // attention heads (12, etc.)
    pub seq_len: usize,        // sequence length (256, etc.)
    pub n_kv_heads: usize,     // KV heads for GQA (n_heads for MHA)
    pub rope_theta: f64,       // RoPE frequency base (10000.0 for llama, 1e6 for Qwen)
    pub rms_eps: f32,          // RMSNorm epsilon (1e-5 for llama, 1e-6 for Qwen)
    pub has_lm_head: bool,     // true = untied lm_head, false = share embed
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
        }
    }

    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
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
    DynMatmul { ic: usize, oc: usize },
    SdpaFwd,
    FfnW13,
    FfnW2,
    Wot,
    FfnBwdW2t,
    FfnBwdW13t,
    Qkvb,
    SdpaBwd1,
    SdpaBwd2,
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
                (gen_dyn_matmul_mil(ic, oc, cfg.seq_len), ic, sp, oc, cfg.seq_len, false, false)
            }
            KernelType::SdpaFwd => {
                let sp_in = cfg.seq_len + 4 * cfg.dim;
                let out_ch = 6 * cfg.dim;
                (gen_sdpa_fwd(cfg), cfg.dim, sp_in, out_ch, cfg.seq_len, false, false)
            }
            KernelType::FfnW13 => {
                let sp_in = cfg.seq_len + 2 * cfg.hidden_dim;
                let out_ch = 3 * cfg.hidden_dim;
                (gen_ffn_w13(cfg), cfg.dim, sp_in, out_ch, cfg.seq_len, false, false)
            }
            KernelType::FfnW2 => {
                let sp_in = cfg.seq_len + cfg.dim;
                (gen_ffn_w2(cfg), cfg.hidden_dim, sp_in, cfg.dim, cfg.seq_len, false, false)
            }
            KernelType::Wot => {
                let sp_in = cfg.seq_len + cfg.dim;
                (gen_wot(cfg), cfg.dim, sp_in, cfg.dim, cfg.seq_len, false, false)
            }
            KernelType::FfnBwdW2t => {
                let sp_in = cfg.seq_len + cfg.hidden_dim;
                (gen_ffn_bwd_w2t(cfg), cfg.dim, sp_in, cfg.hidden_dim, cfg.seq_len, false, false)
            }
            KernelType::FfnBwdW13t => {
                let sp_in = 2 * cfg.seq_len + 2 * cfg.dim;
                (gen_ffn_bwd_w13t(cfg), cfg.hidden_dim, sp_in, cfg.dim, cfg.seq_len, false, false)
            }
            KernelType::Qkvb => {
                let sp_in = 3 * cfg.seq_len + 3 * cfg.dim;
                (gen_qkvb(cfg), cfg.dim, sp_in, cfg.dim, cfg.seq_len, false, false)
            }
            KernelType::SdpaBwd1 => {
                let in_ch = 4 * cfg.dim;
                let out_ch = cfg.dim + 2 * cfg.score_ch();
                (gen_sdpa_bwd1(cfg), in_ch, cfg.seq_len, out_ch, cfg.seq_len, true, true)
            }
            KernelType::SdpaBwd2 => {
                let in_ch = 2 * cfg.score_ch() + 2 * cfg.dim;
                let out_ch = 2 * cfg.dim;
                (gen_sdpa_bwd2(cfg), in_ch, cfg.seq_len, out_ch, cfg.seq_len, true, true)
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
    let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
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
    let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1, {ic}, 1, {sp}]> x) {{");
    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{ic},1,{sp}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");
    gen_dyn_matmul(&mut m, "mm", ic, oc, seq, 0, seq, "xh");
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
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
    let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1, {dim}, 1, {sp_in}]> x) {{");
    // Cast to fp16
    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
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
    let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
    let _ = writeln!(m, "        bool bT = const()[name=string(\"bT\"), val=bool(true)];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> qm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wq2)[name=string(\"qm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> km = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wk2)[name=string(\"km\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> vm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wv2)[name=string(\"vm\")];");

    // Transpose back: [1,1,S,D] → [1,1,D,S] → reshape [1,D,1,S]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> qt = transpose(perm=pm,x=qm)[name=string(\"qt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> kt = transpose(perm=pm,x=km)[name=string(\"kt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> vt = transpose(perm=pm,x=vm)[name=string(\"vt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> os = const()[name=string(\"os\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> qf = reshape(shape=os,x=qt)[name=string(\"qf\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> kf = reshape(shape=os,x=kt)[name=string(\"kf\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> vf = reshape(shape=os,x=vt)[name=string(\"vf\")];");

    // SDPA: reshape to heads
    let _ = writeln!(m, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> q4 = reshape(shape=qsh,x=qf)[name=string(\"rq\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> q = transpose(perm=pm,x=q4)[name=string(\"tq\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> k4 = reshape(shape=qsh,x=kf)[name=string(\"rk\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> k = transpose(perm=pm,x=k4)[name=string(\"tk\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> v4 = reshape(shape=qsh,x=vf)[name=string(\"rv\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> v = transpose(perm=pm,x=v4)[name=string(\"tv\")];");

    // Q @ K^T
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"mm1\")];");
    let _ = writeln!(m, "        fp16 scv = const()[name=string(\"scv\"), val=fp16({sc})];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];");

    // Causal mask (const BLOBFILE)
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{seq}]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,{seq},{seq}]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];");

    // Softmax
    let _ = writeln!(m, "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"sm\")];");

    // scores @ V
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> a4 = matmul(transpose_x=bF,transpose_y=bF,x=aw,y=v)[name=string(\"mm2\")];");

    // Reshape back to [1,DIM,1,SEQ]
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> at = transpose(perm=pm,x=a4)[name=string(\"ta\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> af = reshape(shape=os,x=at)[name=string(\"ra\")];");

    // Wo matmul
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> af2 = reshape(shape=r2,x=af)[name=string(\"af2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> aft = transpose(perm=pm,x=af2)[name=string(\"aft\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> om = matmul(transpose_x=bF,transpose_y=bF,x=aft,y=Wo2)[name=string(\"om\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> ot = transpose(perm=pm,x=om)[name=string(\"ot\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> oo = reshape(shape=os,x=ot)[name=string(\"oo\")];");

    // Output: concat(o_out, qf, kf, vf, af, xn)
    let out_ch = 6 * dim;
    let _ = writeln!(m, "        int32 cax = const()[name=string(\"cax\"), val=int32(1)];");
    let _ = writeln!(m, "        bool cid = const()[name=string(\"cid\"), val=bool(false)];");
    let _ = writeln!(m, "        tensor<fp16, [1,{out_ch},1,{seq}]> out = concat(axis=cax,interleave=cid,values=(oo,qf,kf,vf,af,xn))[name=string(\"cat\")];");
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
    let _ = writeln!(m, "        tensor<fp32, [1,{out_ch},1,{seq}]> out32 = cast(dtype=to32,x=out)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (out32);");
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
    let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1, {dim}, 1, {sp_in}]> x) {{");
    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
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

    let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{hidden}]> h1m = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=W12)[name=string(\"h1m\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{hidden}]> h3m = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=W32)[name=string(\"h3m\")];");

    // Transpose back
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{seq}]> h1t = transpose(perm=pm,x=h1m)[name=string(\"h1t\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{seq}]> h3t = transpose(perm=pm,x=h3m)[name=string(\"h3t\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rh = const()[name=string(\"rh\"), val=tensor<int32, [4]>([1,{hidden},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> h1 = reshape(shape=rh,x=h1t)[name=string(\"h1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> h3 = reshape(shape=rh,x=h3t)[name=string(\"h3\")];");

    // SiLU + gate
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> sig = sigmoid(x=h1)[name=string(\"sg\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> silu = mul(x=h1,y=sig)[name=string(\"si\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> gate = mul(x=silu,y=h3)[name=string(\"gt\")];");

    // Concat output: (h1, h3, gate)
    let _ = writeln!(m, "        int32 cax = const()[name=string(\"cax\"), val=int32(1)];");
    let _ = writeln!(m, "        bool cid = const()[name=string(\"cid\"), val=bool(false)];");
    let _ = writeln!(m, "        tensor<fp16, [1,{out_ch},1,{seq}]> out = concat(axis=cax,interleave=cid,values=(h1,h3,gate))[name=string(\"cat\")];");
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
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
    let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1, {hidden}, 1, {sp_in}]> x) {{");
    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
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

    let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> ym = matmul(transpose_x=bF,transpose_y=bF,x=at,y=W22)[name=string(\"ym\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> yt = transpose(perm=pm,x=ym)[name=string(\"yt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> yr = reshape(shape=ro,x=yt)[name=string(\"yr\")];");
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
    let _ = writeln!(m, "        tensor<fp32, [1,{dim},1,{seq}]> y = cast(dtype=to32,x=yr)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (y);");
    m.push_str("}\n");
    m
}

/// Wo^T backward matmul: dx2 @ Wo^T → da (dim → dim).
pub fn gen_wot(cfg: &MilConfig) -> String {
    gen_dyn_matmul_mil(cfg.dim, cfg.dim, cfg.seq_len)
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
    let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1, {hidden}, 1, {sp_in}]> x) {{");
    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
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

    let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx1m = matmul(transpose_x=bF,transpose_y=bF,x=dh1t,y=W1t2)[name=string(\"dx1m\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dx3m = matmul(transpose_x=bF,transpose_y=bF,x=dh3t,y=W3t2)[name=string(\"dx3m\")];");

    // Add
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dxm = add(x=dx1m,y=dx3m)[name=string(\"dxm\")];");

    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dxt = transpose(perm=pm,x=dxm)[name=string(\"dxt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dx = reshape(shape=ro,x=dxt)[name=string(\"dx\")];");
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
    let _ = writeln!(m, "        tensor<fp32, [1,{dim},1,{seq}]> y = cast(dtype=to32,x=dx)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (y);");
    m.push_str("}\n");
    m
}

/// QKV backward: dq @ Wq^T + dk @ Wk^T + dv @ Wv^T → dx (fused add).
///
/// Input: `[1, dim, 1, 3*seq + 3*dim]` fp32.
/// Output: `[1, dim, 1, seq]` fp32.
pub fn gen_qkvb(cfg: &MilConfig) -> String {
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let sp_in = 3 * seq + 3 * dim;

    let mut m = String::with_capacity(8192);
    m.push_str(MIL_HDR);
    let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1, {dim}, 1, {sp_in}]> x) {{");
    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{sp_in}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");

    // Slice dq, dk, dv
    let _ = writeln!(m, "        tensor<int32, [4]> sd = const()[name=string(\"sd\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dq = slice_by_size(x=xh,begin=b0,size=sd)[name=string(\"dq\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,0,0,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dk = slice_by_size(x=xh,begin=b1,size=sd)[name=string(\"dk\")];");
    let off_dv = 2 * seq;
    let _ = writeln!(m, "        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,0,0,{off_dv}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dv = slice_by_size(x=xh,begin=b2,size=sd)[name=string(\"dv\")];");

    // Slice Wq^T, Wk^T, Wv^T
    let _ = writeln!(m, "        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,{dim},1,{dim}])];");
    let off_wqt = 3 * seq;
    let _ = writeln!(m, "        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,0,0,{off_wqt}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{dim}]> Wqt = slice_by_size(x=xh,begin=b3,size=sw)[name=string(\"Wqt\")];");
    let off_wkt = 3 * seq + dim;
    let _ = writeln!(m, "        tensor<int32, [4]> b4 = const()[name=string(\"b4\"), val=tensor<int32, [4]>([0,0,0,{off_wkt}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{dim}]> Wkt = slice_by_size(x=xh,begin=b4,size=sw)[name=string(\"Wkt\")];");
    let off_wvt = 3 * seq + 2 * dim;
    let _ = writeln!(m, "        tensor<int32, [4]> b5 = const()[name=string(\"b5\"), val=tensor<int32, [4]>([0,0,0,{off_wvt}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{dim}]> Wvt = slice_by_size(x=xh,begin=b5,size=sw)[name=string(\"Wvt\")];");

    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");

    let _ = writeln!(m, "        tensor<int32, [4]> rd = const()[name=string(\"rd\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,{dim},{dim}])];");

    // dq @ Wq^T
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dq2 = reshape(shape=rd,x=dq)[name=string(\"dq2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dqt = transpose(perm=pm,x=dq2)[name=string(\"dqt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{dim}]> Wqt2 = reshape(shape=rw,x=Wqt)[name=string(\"Wqt2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dxq = matmul(transpose_x=bF,transpose_y=bF,x=dqt,y=Wqt2)[name=string(\"dxq\")];");

    // dk @ Wk^T
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dk2 = reshape(shape=rd,x=dk)[name=string(\"dk2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dkt = transpose(perm=pm,x=dk2)[name=string(\"dkt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{dim}]> Wkt2 = reshape(shape=rw,x=Wkt)[name=string(\"Wkt2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dxk = matmul(transpose_x=bF,transpose_y=bF,x=dkt,y=Wkt2)[name=string(\"dxk\")];");

    // dv @ Wv^T
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dv2 = reshape(shape=rd,x=dv)[name=string(\"dv2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dvt = transpose(perm=pm,x=dv2)[name=string(\"dvt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{dim}]> Wvt2 = reshape(shape=rw,x=Wvt)[name=string(\"Wvt2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dxv = matmul(transpose_x=bF,transpose_y=bF,x=dvt,y=Wvt2)[name=string(\"dxv\")];");

    // Sum: dxq + dxk + dxv
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dxqk = add(x=dxq,y=dxk)[name=string(\"aqk\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> dxall = add(x=dxqk,y=dxv)[name=string(\"aall\")];");

    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dxt = transpose(perm=pm,x=dxall)[name=string(\"dxt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dx = reshape(shape=ro,x=dxt)[name=string(\"dx\")];");
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
    let _ = writeln!(m, "        tensor<fp32, [1,{dim},1,{seq}]> y = cast(dtype=to32,x=dx)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (y);");
    m.push_str("}\n");
    m
}

/// SDPA backward part 1 (weight-free): recompute softmax + dV + dp.
///
/// Input: `[1, 4*dim, 1, seq]` fp16 — Q,K,V,da stacked in channels.
/// Output: `[1, dim+2*score_ch, 1, seq]` fp16 = concat(dV, probs, dp).
pub fn gen_sdpa_bwd1(cfg: &MilConfig) -> String {
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let heads = cfg.n_heads;
    let hd = cfg.head_dim();
    let score_ch = cfg.score_ch();
    let sc = 1.0 / (hd as f64).sqrt();
    let in_ch = 4 * dim;
    let out_ch = dim + 2 * score_ch;

    let mut m = String::with_capacity(8192);
    m.push_str(MIL_HDR);
    let _ = writeln!(m, "    func main<ios18>(tensor<fp16, [1, {in_ch}, 1, {seq}]> x) {{");

    // Slice Q,K,V,da (channel-wise)
    let _ = writeln!(m, "        tensor<int32, [4]> sz = const()[name=string(\"sz\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> qf = slice_by_size(x=x,begin=b0,size=sz)[name=string(\"s0\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,{dim},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> kf = slice_by_size(x=x,begin=b1,size=sz)[name=string(\"s1\")];");
    let off_v = 2 * dim;
    let _ = writeln!(m, "        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,{off_v},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> vf = slice_by_size(x=x,begin=b2,size=sz)[name=string(\"s2\")];");
    let off_da = 3 * dim;
    let _ = writeln!(m, "        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,{off_da},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> da = slice_by_size(x=x,begin=b3,size=sz)[name=string(\"s3\")];");

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
    let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
    let _ = writeln!(m, "        bool bT = const()[name=string(\"bT\"), val=bool(true)];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"mm1\")];");
    let _ = writeln!(m, "        fp16 scv = const()[name=string(\"scv\"), val=fp16({sc})];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{seq}]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,{seq},{seq}]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];");
    let _ = writeln!(m, "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> probs = softmax(axis=sax,x=ms)[name=string(\"sm\")];");

    // dV = probs^T @ da, dp = da @ V^T
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dv4 = matmul(transpose_x=bT,transpose_y=bF,x=probs,y=dat)[name=string(\"dv\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dp4 = matmul(transpose_x=bF,transpose_y=bT,x=dat,y=v)[name=string(\"dp\")];");

    // Reshape dV back
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> dvt = transpose(perm=pm,x=dv4)[name=string(\"dvt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> dvs = const()[name=string(\"dvs\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dvf = reshape(shape=dvs,x=dvt)[name=string(\"dvf\")];");

    // Flatten probs and dp for output
    let _ = writeln!(m, "        tensor<int32, [4]> scs = const()[name=string(\"scs\"), val=tensor<int32, [4]>([1,{score_ch},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{score_ch},1,{seq}]> pf = reshape(shape=scs,x=probs)[name=string(\"pf\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{score_ch},1,{seq}]> dpf = reshape(shape=scs,x=dp4)[name=string(\"dpf\")];");

    let _ = writeln!(m, "        int32 cax = const()[name=string(\"cax\"), val=int32(1)];");
    let _ = writeln!(m, "        bool cid = const()[name=string(\"cid\"), val=bool(false)];");
    let _ = writeln!(m, "        tensor<fp16, [1,{out_ch},1,{seq}]> out = concat(axis=cax,interleave=cid,values=(dvf,pf,dpf))[name=string(\"cat\")];");
    let _ = writeln!(m, "    }} -> (out);");
    m.push_str("}\n");
    m
}

/// SDPA backward part 2: dQ + dK from probs, dp, Q, K.
///
/// Input: `[1, 2*score_ch + 2*dim, 1, seq]` fp16.
/// Output: `[1, 2*dim, 1, seq]` fp16 = concat(dQ, dK).
pub fn gen_sdpa_bwd2(cfg: &MilConfig) -> String {
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let heads = cfg.n_heads;
    let hd = cfg.head_dim();
    let score_ch = cfg.score_ch();
    let sc = 1.0 / (hd as f64).sqrt();
    let in_ch = 2 * score_ch + 2 * dim;
    let out_ch = 2 * dim;

    let mut m = String::with_capacity(8192);
    m.push_str(MIL_HDR);
    let _ = writeln!(m, "    func main<ios18>(tensor<fp16, [1, {in_ch}, 1, {seq}]> x) {{");

    // Slice probs, dp (channel-wise, score_ch each)
    let _ = writeln!(m, "        tensor<int32, [4]> sz_sc = const()[name=string(\"szsc\"), val=tensor<int32, [4]>([1,{score_ch},1,{seq}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{score_ch},1,{seq}]> pf = slice_by_size(x=x,begin=b0,size=sz_sc)[name=string(\"s0\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,{score_ch},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{score_ch},1,{seq}]> dpf = slice_by_size(x=x,begin=b1,size=sz_sc)[name=string(\"s1\")];");

    // Slice Q, K
    let _ = writeln!(m, "        tensor<int32, [4]> sz_d = const()[name=string(\"szd\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let off_q = 2 * score_ch;
    let _ = writeln!(m, "        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,{off_q},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> qf = slice_by_size(x=x,begin=b2,size=sz_d)[name=string(\"s2\")];");
    let off_k = 2 * score_ch + dim;
    let _ = writeln!(m, "        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,{off_k},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> kf = slice_by_size(x=x,begin=b3,size=sz_d)[name=string(\"s3\")];");

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
    let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> spdp = reduce_sum(x=pdp,axes=rax,keep_dims=kd)[name=string(\"rs\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dps = sub(x=dp,y=spdp)[name=string(\"dps\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ds0 = mul(x=probs,y=dps)[name=string(\"ds0\")];");
    let _ = writeln!(m, "        fp16 scv = const()[name=string(\"scv\"), val=fp16({sc})];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> ds = mul(x=ds0,y=scv)[name=string(\"ds\")];");

    // dQ = ds @ K, dK = ds^T @ Q
    let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
    let _ = writeln!(m, "        bool bT = const()[name=string(\"bT\"), val=bool(true)];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dq4 = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=k)[name=string(\"dq\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dk4 = matmul(transpose_x=bT,transpose_y=bF,x=ds,y=q)[name=string(\"dk\")];");

    // Reshape back
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> dqt = transpose(perm=pm,x=dq4)[name=string(\"dqt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> dkt = transpose(perm=pm,x=dk4)[name=string(\"dkt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> fs = const()[name=string(\"fs\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dqf = reshape(shape=fs,x=dqt)[name=string(\"dqf\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dkf = reshape(shape=fs,x=dkt)[name=string(\"dkf\")];");
    let _ = writeln!(m, "        int32 cax = const()[name=string(\"cax\"), val=int32(1)];");
    let _ = writeln!(m, "        bool cid = const()[name=string(\"cid\"), val=bool(false)];");
    let _ = writeln!(m, "        tensor<fp16, [1,{out_ch},1,{seq}]> out = concat(axis=cax,interleave=cid,values=(dqf,dkf))[name=string(\"cat\")];");
    let _ = writeln!(m, "    }} -> (out);");
    m.push_str("}\n");
    m
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
        assert!(max_err < 0.05, "matmul identity max error {max_err} too large");
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

        // Build causal mask blob
        let mask_blob = build_causal_mask_blob(cfg.seq_len);

        let kernel = AneKernel::compile_multi_weights(
            &mil,
            &["@model_path/weights/mask.bin"],
            &[&mask_blob],
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
        };
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.kv_dim(), 512);      // 8 * 64
        assert_eq!(cfg.heads_per_group(), 2); // 16 / 8
        assert_eq!(cfg.kv_score_ch(), 8 * 128);
        assert_eq!(cfg.score_ch(), 16 * 128);
    }
}
