//! MLX GPU LoRA training for quantized models.
//!
//! Uses mlx-rs for Apple Silicon GPU-accelerated LoRA fine-tuning:
//! - Base model stays quantized (8-bit) and frozen via `QuantizedLinear`
//! - LoRA adapters (A, B matrices) are trainable `Linear` layers
//! - `nn::value_and_grad()` provides autograd through the LoRA path
//! - `AdamW` optimizer with gradient clipping
//!
//! Mirrors the Python JIT LoRA trainer:
//! - rank=32, alpha=32, LR=5e-4, grad_clip=1.0
//! - Targets: q_proj, v_proj, o_proj (attention) + down_proj (FFN)
//! - Batch size 1, epoch-style training

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::{Module, ModuleParameters, ModuleParamMut, ModuleParamRef, Param};
use mlx_rs::nested::{NestedHashMap, NestedValue};
use mlx_rs::nn::{self, Embedding, Linear, QuantizedEmbedding, QuantizedLinear, RmsNorm};
use mlx_rs::optimizers::{AdamW, Optimizer};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::losses::{CrossEntropyBuilder, LossReduction};
use mlx_rs::{array, Array};

// ---------------------------------------------------------------------------
// Tokenizer wrapper (reads tokenizer.json from model directory)
// ---------------------------------------------------------------------------

pub struct MlxTokenizer {
    inner: tokenizers::Tokenizer,
}

impl MlxTokenizer {
    pub fn load(model_dir: &Path) -> Result<Self, anyhow::Error> {
        let path = model_dir.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {}: {e}", path.display()))?;
        Ok(MlxTokenizer { inner })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i32>, anyhow::Error> {
        let encoding = self.inner.encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenizer encode failed: {e}"))?;
        Ok(encoding.get_ids().iter().map(|&id| id as i32).collect())
    }

    pub fn decode(&self, ids: &[i32]) -> Result<String, anyhow::Error> {
        let u32_ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        let text = self.inner.decode(&u32_ids, true)
            .map_err(|e| anyhow::anyhow!("Tokenizer decode failed: {e}"))?;
        Ok(text)
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        let vocab = self.inner.get_vocab(true);
        // Prefer <|im_end|> for chat models, then general EOS markers
        for name in &["<|im_end|>", "<|endoftext|>", "</s>", "<eos>"] {
            if let Some(&id) = vocab.get(*name) {
                return Some(id);
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// LoRA configuration
// ---------------------------------------------------------------------------

pub struct LoraConfig {
    pub rank: i32,
    pub alpha: f32,
    pub lr: f32,
    pub weight_decay: f32,
    pub grad_clip: f32,
}

impl Default for LoraConfig {
    fn default() -> Self {
        LoraConfig {
            rank: 32,
            alpha: 32.0,
            lr: 5e-4,
            weight_decay: 0.01,
            grad_clip: 1.0,
        }
    }
}

impl LoraConfig {
    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

// ---------------------------------------------------------------------------
// Helper: build NestedHashMap from sub-modules
// ---------------------------------------------------------------------------

fn nested_ref_from<'a>(
    entries: Vec<(&str, ModuleParamRef<'a>)>,
) -> NestedHashMap<Rc<str>, &'a Array> {
    let mut map = NestedHashMap::new();
    for (name, params) in entries {
        map.insert(Rc::from(name), params.into());
    }
    map
}

fn nested_mut_from<'a>(
    entries: Vec<(&str, ModuleParamMut<'a>)>,
) -> NestedHashMap<Rc<str>, &'a mut Array> {
    let mut map = NestedHashMap::new();
    for (name, params) in entries {
        map.insert(Rc::from(name), params.into());
    }
    map
}

// ---------------------------------------------------------------------------
// LoRA Linear: quantized base + trainable low-rank adapters
// ---------------------------------------------------------------------------

/// LoRA-wrapped quantized linear layer.
///
/// Forward: y = QuantizedLinear(x) + scale * B(A(x))
///
/// The base QuantizedLinear is frozen. Only A and B are trainable.
#[derive(Debug)]
pub struct LoraLinear {
    pub base: QuantizedLinear,
    pub lora_a: Linear,
    pub lora_b: Linear,
    pub scale: f32,
}

impl LoraLinear {
    pub fn new(base: QuantizedLinear, rank: i32, scale: f32) -> Result<Self, Exception> {
        let weight_shape = base.inner.weight.shape();
        let output_dims = weight_shape[0] as i32;
        let input_dims = (weight_shape[1] as i32) * 32 / base.bits;

        let lora_a = Linear {
            weight: Param::new(mlx_rs::random::normal::<f32>(
                &[rank, input_dims],
                None,
                Some(1.0 / (input_dims as f32).sqrt()),
                None,
            )?),
            bias: Param::new(None),
        };

        let lora_b = Linear {
            weight: Param::new(mlx_rs::ops::zeros::<f32>(&[output_dims, rank])?),
            bias: Param::new(None),
        };

        Ok(LoraLinear {
            base,
            lora_a,
            lora_b,
            scale,
        })
    }
}

impl ModuleParameters for LoraLinear {
    fn num_parameters(&self) -> usize {
        self.base.num_parameters() + self.lora_a.num_parameters() + self.lora_b.num_parameters()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        nested_ref_from(vec![
            ("base", self.base.parameters()),
            ("lora_a", self.lora_a.parameters()),
            ("lora_b", self.lora_b.parameters()),
        ])
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        nested_mut_from(vec![
            ("base", self.base.parameters_mut()),
            ("lora_a", self.lora_a.parameters_mut()),
            ("lora_b", self.lora_b.parameters_mut()),
        ])
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        nested_ref_from(vec![
            ("lora_a", self.lora_a.trainable_parameters()),
            ("lora_b", self.lora_b.trainable_parameters()),
        ])
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        self.base.freeze_parameters(recursive);
        self.lora_a.freeze_parameters(recursive);
        self.lora_b.freeze_parameters(recursive);
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        self.lora_a.unfreeze_parameters(recursive);
        self.lora_b.unfreeze_parameters(recursive);
    }

    fn all_frozen(&self) -> Option<bool> {
        Some(
            self.lora_a.all_frozen().unwrap_or(true)
                && self.lora_b.all_frozen().unwrap_or(true),
        )
    }

    fn any_frozen(&self) -> Option<bool> {
        Some(true) // base is always frozen
    }
}

impl Module<&Array> for LoraLinear {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let base_out = self.base.forward(x)?;
        let h = self.lora_a.forward(x)?;
        let delta = self.lora_b.forward(&h)?;
        let scaled = delta.multiply(&array!(self.scale))?;
        base_out.add(&scaled)
    }

    fn training_mode(&mut self, mode: bool) {
        self.base.training_mode(mode);
        self.lora_a.training_mode(mode);
        self.lora_b.training_mode(mode);
    }
}

// ---------------------------------------------------------------------------
// Weight loading helpers
// ---------------------------------------------------------------------------

pub fn load_weights(model_dir: &Path) -> Result<HashMap<String, Array>, anyhow::Error> {
    let mut weights = HashMap::new();
    let mut files: Vec<_> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
        .map(|e| e.path())
        .collect();
    files.sort();

    if files.is_empty() {
        anyhow::bail!("No .safetensors files in {}", model_dir.display());
    }

    for path in &files {
        let file_weights = Array::load_safetensors(path)
            .map_err(|e| anyhow::anyhow!("Failed to load {}: {}", path.display(), e))?;
        weights.extend(file_weights);
    }

    Ok(weights)
}

fn take(weights: &mut HashMap<String, Array>, key: &str) -> Result<Array, anyhow::Error> {
    weights
        .remove(key)
        .ok_or_else(|| anyhow::anyhow!("Weight not found: {}", key))
}

fn load_quantized_linear(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<QuantizedLinear, anyhow::Error> {
    let weight = take(weights, &format!("{prefix}.weight"))?;
    let scales = take(weights, &format!("{prefix}.scales"))?;
    let biases = take(weights, &format!("{prefix}.biases"))?;
    let bias = weights.remove(&format!("{prefix}.bias"));

    let inner = Linear {
        weight: Param::new(weight),
        bias: Param::new(bias),
    };

    let mut ql = QuantizedLinear {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner,
    };
    ql.freeze_parameters(true);
    Ok(ql)
}

fn load_rms_norm(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    eps: f32,
) -> Result<RmsNorm, anyhow::Error> {
    let weight = take(weights, &format!("{prefix}.weight"))?;
    let mut norm = RmsNorm {
        weight: Param::new(weight),
        eps,
    };
    norm.freeze_parameters(true);
    Ok(norm)
}

/// Load embedding — returns QuantizedEmbedding if scales/biases present, else regular.
enum MlxEmbedding {
    Plain(Embedding),
    Quantized(QuantizedEmbedding),
}

impl std::fmt::Debug for MlxEmbedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MlxEmbedding::Plain(_) => write!(f, "Embedding(plain)"),
            MlxEmbedding::Quantized(_) => write!(f, "Embedding(quantized)"),
        }
    }
}

impl MlxEmbedding {
    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        match self {
            MlxEmbedding::Plain(e) => e.forward(x),
            MlxEmbedding::Quantized(e) => e.forward(x),
        }
    }
}

impl ModuleParameters for MlxEmbedding {
    fn num_parameters(&self) -> usize {
        match self { MlxEmbedding::Plain(e) => e.num_parameters(), MlxEmbedding::Quantized(e) => e.num_parameters() }
    }
    fn parameters(&self) -> ModuleParamRef<'_> {
        match self { MlxEmbedding::Plain(e) => e.parameters(), MlxEmbedding::Quantized(e) => e.parameters() }
    }
    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        match self { MlxEmbedding::Plain(e) => e.parameters_mut(), MlxEmbedding::Quantized(e) => e.parameters_mut() }
    }
    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        match self { MlxEmbedding::Plain(e) => e.trainable_parameters(), MlxEmbedding::Quantized(e) => e.trainable_parameters() }
    }
    fn freeze_parameters(&mut self, r: bool) {
        match self { MlxEmbedding::Plain(e) => e.freeze_parameters(r), MlxEmbedding::Quantized(e) => e.freeze_parameters(r) }
    }
    fn unfreeze_parameters(&mut self, r: bool) {
        match self { MlxEmbedding::Plain(e) => e.unfreeze_parameters(r), MlxEmbedding::Quantized(e) => e.unfreeze_parameters(r) }
    }
    fn all_frozen(&self) -> Option<bool> {
        match self { MlxEmbedding::Plain(e) => e.all_frozen(), MlxEmbedding::Quantized(e) => e.all_frozen() }
    }
    fn any_frozen(&self) -> Option<bool> {
        match self { MlxEmbedding::Plain(e) => e.any_frozen(), MlxEmbedding::Quantized(e) => e.any_frozen() }
    }
}

fn load_embedding(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<MlxEmbedding, anyhow::Error> {
    let weight = take(weights, &format!("{prefix}.weight"))?;
    if weights.contains_key(&format!("{prefix}.scales")) {
        let scales = take(weights, &format!("{prefix}.scales"))?;
        let biases = take(weights, &format!("{prefix}.biases"))?;
        let inner = Embedding { weight: Param::new(weight) };
        let mut qe = QuantizedEmbedding {
            group_size, bits,
            scales: Param::new(scales),
            biases: Param::new(biases),
            inner,
        };
        qe.freeze_parameters(true);
        Ok(MlxEmbedding::Quantized(qe))
    } else {
        let mut emb = Embedding { weight: Param::new(weight) };
        emb.freeze_parameters(true);
        Ok(MlxEmbedding::Plain(emb))
    }
}

// ---------------------------------------------------------------------------
// Qwen3 model with LoRA
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub group_size: i32,
    pub bits: i32,
    /// Weight key prefix: "model" for Qwen3, "language_model.model" for Qwen3.5
    pub weight_prefix: &'static str,
    /// Fraction of head_dim that gets RoPE (1.0 for Qwen3, 0.25 for Qwen3.5)
    pub partial_rotary_factor: f32,
    /// q_proj output is doubled for output gating (Qwen3.5 full_attention)
    pub attn_output_gate: bool,
    /// Layer indices that use linear (Mamba) attention instead of full attention.
    /// Empty for pure-transformer models like Qwen3.
    pub linear_attn_indices: Vec<usize>,
    /// Linear attention head config (only used when linear_attn_indices is non-empty)
    pub linear_n_heads: usize,
    pub linear_head_dim: usize,
    pub conv_kernel_size: usize,
}

impl ModelConfig {
    pub fn qwen3_1_7b() -> Self {
        ModelConfig {
            dim: 2048, hidden_dim: 6144,
            n_heads: 16, n_kv_heads: 8, n_layers: 28,
            vocab_size: 151936, head_dim: 128,
            rope_theta: 1_000_000.0, rms_eps: 1e-6,
            group_size: 64, bits: 8,
            weight_prefix: "model",
            partial_rotary_factor: 1.0,
            attn_output_gate: false,
            linear_attn_indices: vec![],
            linear_n_heads: 0, linear_head_dim: 0, conv_kernel_size: 0,
        }
    }

    pub fn qwen3_5_2b() -> Self {
        // Hybrid Mamba-Transformer: 18 linear_attn + 6 full_attn (every 4th)
        let linear_indices: Vec<usize> = (0..24)
            .filter(|i| i % 4 != 3)
            .collect();
        ModelConfig {
            dim: 2048, hidden_dim: 6144,
            n_heads: 8, n_kv_heads: 2, n_layers: 24,
            vocab_size: 248320, head_dim: 256,
            rope_theta: 10_000_000.0, rms_eps: 1e-6,
            group_size: 64, bits: 8,
            weight_prefix: "language_model.model",
            partial_rotary_factor: 0.25,
            attn_output_gate: true,
            linear_attn_indices: linear_indices,
            linear_n_heads: 16, linear_head_dim: 128, conv_kernel_size: 4,
        }
    }

    pub fn rope_dims(&self) -> i32 {
        (self.head_dim as f32 * self.partial_rotary_factor) as i32
    }

    pub fn is_linear_attn_layer(&self, idx: usize) -> bool {
        self.linear_attn_indices.contains(&idx)
    }
}

// --- Attention ---

#[derive(Debug)]
pub struct MlxLoraAttention {
    pub q_proj: LoraLinear,
    pub k_proj: QuantizedLinear,
    pub v_proj: LoraLinear,
    pub o_proj: LoraLinear,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub attn_scale: f32,
    pub rope_dims: i32,
    pub rope_base: f32,
    pub attn_output_gate: bool,
}

impl MlxLoraAttention {
    pub fn load(
        weights: &mut HashMap<String, Array>,
        prefix: &str,
        cfg: &ModelConfig,
        lora_cfg: &LoraConfig,
    ) -> Result<Self, anyhow::Error> {
        let gs = cfg.group_size;
        let bits = cfg.bits;
        let scale = lora_cfg.scale();
        let rank = lora_cfg.rank;

        Ok(MlxLoraAttention {
            q_proj: LoraLinear::new(
                load_quantized_linear(weights, &format!("{prefix}.q_proj"), gs, bits)?,
                rank, scale,
            ).map_err(|e| anyhow::anyhow!("LoRA q_proj: {e}"))?,
            k_proj: load_quantized_linear(weights, &format!("{prefix}.k_proj"), gs, bits)?,
            v_proj: LoraLinear::new(
                load_quantized_linear(weights, &format!("{prefix}.v_proj"), gs, bits)?,
                rank, scale,
            ).map_err(|e| anyhow::anyhow!("LoRA v_proj: {e}"))?,
            o_proj: LoraLinear::new(
                load_quantized_linear(weights, &format!("{prefix}.o_proj"), gs, bits)?,
                rank, scale,
            ).map_err(|e| anyhow::anyhow!("LoRA o_proj: {e}"))?,
            q_norm: load_rms_norm(weights, &format!("{prefix}.q_norm"), cfg.rms_eps)?,
            k_norm: load_rms_norm(weights, &format!("{prefix}.k_norm"), cfg.rms_eps)?,
            num_heads: cfg.n_heads as i32,
            num_kv_heads: cfg.n_kv_heads as i32,
            head_dim: cfg.head_dim as i32,
            attn_scale: 1.0 / (cfg.head_dim as f32).sqrt(),
            rope_dims: cfg.rope_dims(),
            rope_base: cfg.rope_theta,
            attn_output_gate: cfg.attn_output_gate,
        })
    }

    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        let shape = x.shape();
        let (batch, seq_len) = (shape[0], shape[1]);

        let q_raw = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // When attn_output_gate is enabled, q_proj outputs [Q, gate] interleaved per-head.
        // Python: reshape to [B, S, H, 2*D] then split → per-head [B, S, H, D] each.
        // Memory layout is [Q_h0, Gate_h0, Q_h1, Gate_h1, ...] NOT [Q_all | Gate_all].
        let (q, output_gate) = if self.attn_output_gate {
            let q4d = q_raw.reshape(&[batch, seq_len, self.num_heads, self.head_dim * 2])?;
            let parts = q4d.split(2, -1)?; // [B, S, H, D] × 2
            let gate = parts[1].reshape(&[batch, seq_len, self.num_heads * self.head_dim])?;
            (parts[0].clone(), Some(gate))
        } else {
            (q_raw, None)
        };

        // QK norm — when gated, q is already [B, S, H, D] from the per-head split
        let q = if self.attn_output_gate {
            // q is [B, S, H, D] — RMSNorm applies on last dim (head_dim), which matches weight shape
            self.q_norm.forward(&q)?
        } else {
            let q_norm_size = self.q_norm.weight.shape()[0] as i32;
            if q_norm_size == self.head_dim {
                let q = q.reshape(&[batch * seq_len * self.num_heads, self.head_dim])?;
                let q = self.q_norm.forward(&q)?;
                q.reshape(&[batch, seq_len, self.num_heads * self.head_dim])?
            } else {
                self.q_norm.forward(&q)?
            }
        };
        let k_norm_size = self.k_norm.weight.shape()[0] as i32;
        let k = if k_norm_size == self.head_dim {
            let k = k.reshape(&[batch * seq_len * self.num_kv_heads, self.head_dim])?;
            let k = self.k_norm.forward(&k)?;
            k.reshape(&[batch, seq_len, self.num_kv_heads * self.head_dim])?
        } else {
            self.k_norm.forward(&k)?
        };

        // Reshape to [B, S, H, D] — q may already be 4D if gated
        let q = if self.attn_output_gate {
            q // already [B, S, H, D]
        } else {
            q.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?
        };
        let k = k.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?;
        let v = v.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?;

        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        let q = mlx_rs::fast::rope(&q, self.rope_dims, false, self.rope_base, 1.0, 0, None::<&Array>)?;
        let k = mlx_rs::fast::rope(&k, self.rope_dims, false, self.rope_base, 1.0, 0, None::<&Array>)?;

        // MLX SDPA handles GQA natively (num_heads != num_kv_heads) — no expansion needed
        use mlx_rs::fast::ScaledDotProductAttentionMask;
        let attn = if let Some(m) = mask {
            let m = m.as_dtype(q.dtype())?;
            mlx_rs::fast::scaled_dot_product_attention(
                &q, &k, &v, self.attn_scale,
                ScaledDotProductAttentionMask::Array(&m),
            )?
        } else {
            mlx_rs::fast::scaled_dot_product_attention(
                &q, &k, &v, self.attn_scale,
                ScaledDotProductAttentionMask::Causal,
            )?
        };

        let mut attn = attn
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, self.num_heads * self.head_dim])?;

        if let Some(gate) = output_gate {
            attn = attn.multiply(&nn::sigmoid(&gate)?)?;
        }

        self.o_proj.forward(&attn)
    }
}

impl ModuleParameters for MlxLoraAttention {
    fn num_parameters(&self) -> usize {
        self.q_proj.num_parameters() + self.k_proj.num_parameters()
            + self.v_proj.num_parameters() + self.o_proj.num_parameters()
            + self.q_norm.num_parameters() + self.k_norm.num_parameters()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        nested_ref_from(vec![
            ("q_proj", self.q_proj.parameters()),
            ("k_proj", self.k_proj.parameters()),
            ("v_proj", self.v_proj.parameters()),
            ("o_proj", self.o_proj.parameters()),
            ("q_norm", self.q_norm.parameters()),
            ("k_norm", self.k_norm.parameters()),
        ])
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        nested_mut_from(vec![
            ("q_proj", self.q_proj.parameters_mut()),
            ("k_proj", self.k_proj.parameters_mut()),
            ("v_proj", self.v_proj.parameters_mut()),
            ("o_proj", self.o_proj.parameters_mut()),
            ("q_norm", self.q_norm.parameters_mut()),
            ("k_norm", self.k_norm.parameters_mut()),
        ])
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        nested_ref_from(vec![
            ("q_proj", self.q_proj.trainable_parameters()),
            ("v_proj", self.v_proj.trainable_parameters()),
            ("o_proj", self.o_proj.trainable_parameters()),
        ])
    }

    fn freeze_parameters(&mut self, r: bool) {
        self.q_proj.freeze_parameters(r);
        self.k_proj.freeze_parameters(r);
        self.v_proj.freeze_parameters(r);
        self.o_proj.freeze_parameters(r);
        self.q_norm.freeze_parameters(r);
        self.k_norm.freeze_parameters(r);
    }

    fn unfreeze_parameters(&mut self, r: bool) {
        self.q_proj.unfreeze_parameters(r);
        self.v_proj.unfreeze_parameters(r);
        self.o_proj.unfreeze_parameters(r);
    }

    fn all_frozen(&self) -> Option<bool> {
        Some(
            self.q_proj.all_frozen().unwrap_or(true)
                && self.v_proj.all_frozen().unwrap_or(true)
                && self.o_proj.all_frozen().unwrap_or(true),
        )
    }

    fn any_frozen(&self) -> Option<bool> {
        Some(true)
    }
}

// --- MLP ---

#[derive(Debug)]
pub struct MlxLoraMLP {
    pub gate_proj: QuantizedLinear,
    pub up_proj: QuantizedLinear,
    pub down_proj: LoraLinear,
}

impl MlxLoraMLP {
    pub fn load(
        weights: &mut HashMap<String, Array>,
        prefix: &str,
        cfg: &ModelConfig,
        lora_cfg: &LoraConfig,
    ) -> Result<Self, anyhow::Error> {
        let gs = cfg.group_size;
        let bits = cfg.bits;

        Ok(MlxLoraMLP {
            gate_proj: load_quantized_linear(weights, &format!("{prefix}.gate_proj"), gs, bits)?,
            up_proj: load_quantized_linear(weights, &format!("{prefix}.up_proj"), gs, bits)?,
            down_proj: LoraLinear::new(
                load_quantized_linear(weights, &format!("{prefix}.down_proj"), gs, bits)?,
                lora_cfg.rank, lora_cfg.scale(),
            ).map_err(|e| anyhow::anyhow!("LoRA down_proj: {e}"))?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x)?;
        let gate = nn::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        let h = gate.multiply(&up)?;
        self.down_proj.forward(&h)
    }
}

impl ModuleParameters for MlxLoraMLP {
    fn num_parameters(&self) -> usize {
        self.gate_proj.num_parameters() + self.up_proj.num_parameters()
            + self.down_proj.num_parameters()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        nested_ref_from(vec![
            ("gate_proj", self.gate_proj.parameters()),
            ("up_proj", self.up_proj.parameters()),
            ("down_proj", self.down_proj.parameters()),
        ])
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        nested_mut_from(vec![
            ("gate_proj", self.gate_proj.parameters_mut()),
            ("up_proj", self.up_proj.parameters_mut()),
            ("down_proj", self.down_proj.parameters_mut()),
        ])
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        nested_ref_from(vec![
            ("down_proj", self.down_proj.trainable_parameters()),
        ])
    }

    fn freeze_parameters(&mut self, r: bool) {
        self.gate_proj.freeze_parameters(r);
        self.up_proj.freeze_parameters(r);
        self.down_proj.freeze_parameters(r);
    }

    fn unfreeze_parameters(&mut self, r: bool) {
        self.down_proj.unfreeze_parameters(r);
    }

    fn all_frozen(&self) -> Option<bool> {
        self.down_proj.all_frozen()
    }

    fn any_frozen(&self) -> Option<bool> {
        Some(true)
    }
}

// --- Linear (Mamba2 SSM) attention ---

#[derive(Debug)]
pub struct MlxLinearAttention {
    pub in_proj_qkv: QuantizedLinear,
    pub in_proj_a: QuantizedLinear,
    pub in_proj_b: QuantizedLinear,
    pub in_proj_z: QuantizedLinear,
    pub out_proj: QuantizedLinear,
    pub conv1d_weight: Param<Array>,
    pub a_log: Param<Array>,
    pub dt_bias: Param<Array>,
    pub norm: RmsNorm,
    pub n_heads: i32,
    pub head_dim: i32,
    pub conv_kernel: i32,
}

impl MlxLinearAttention {
    pub fn load(
        weights: &mut HashMap<String, Array>,
        prefix: &str,
        cfg: &ModelConfig,
    ) -> Result<Self, anyhow::Error> {
        let gs = cfg.group_size;
        let bits = cfg.bits;

        let mut attn = MlxLinearAttention {
            in_proj_qkv: load_quantized_linear(weights, &format!("{prefix}.in_proj_qkv"), gs, bits)?,
            in_proj_a: load_quantized_linear(weights, &format!("{prefix}.in_proj_a"), gs, bits)?,
            in_proj_b: load_quantized_linear(weights, &format!("{prefix}.in_proj_b"), gs, bits)?,
            in_proj_z: load_quantized_linear(weights, &format!("{prefix}.in_proj_z"), gs, bits)?,
            out_proj: load_quantized_linear(weights, &format!("{prefix}.out_proj"), gs, bits)?,
            conv1d_weight: Param::new(take(weights, &format!("{prefix}.conv1d.weight"))?),
            a_log: Param::new(take(weights, &format!("{prefix}.A_log"))?),
            dt_bias: Param::new(take(weights, &format!("{prefix}.dt_bias"))?),
            norm: load_rms_norm(weights, &format!("{prefix}.norm"), cfg.rms_eps)?,
            n_heads: cfg.linear_n_heads as i32,
            head_dim: cfg.linear_head_dim as i32,
            conv_kernel: cfg.conv_kernel_size as i32,
        };
        attn.freeze_parameters(true);
        Ok(attn)
    }

    /// Gated Delta Net forward (linear attention with delta rule).
    ///
    /// Implements the recurrence from `mlx_lm.models.gated_delta`:
    ///   g = exp(-exp(A_log) * softplus(a + dt_bias))    // decay
    ///   beta = sigmoid(b)                                 // write gate
    ///   for each token:
    ///     state *= g; delta = (v - state@k) * beta; state += outer(k, delta); y = state@q
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        let (batch, seq_len) = (shape[0], shape[1]);
        let h = self.n_heads;
        let d = self.head_dim;
        let key_dim = h * d;  // 2048 for Qwen3.5-2B

        // 1. Project QKV, alpha, beta, z
        let qkv = self.in_proj_qkv.forward(x)?;     // [B, L, key_dim*2 + value_dim]
        let a = self.in_proj_a.forward(x)?;           // [B, L, H]
        let b = self.in_proj_b.forward(x)?;           // [B, L, H]

        // 2. Causal depthwise conv1d on QKV
        //    Weight shape from safetensors: [conv_dim, kernel, 1] (depthwise: groups=conv_dim)
        let conv_dim = qkv.shape()[2];  // key_dim*2 + value_dim
        let kernel = self.conv_kernel;
        // Left-pad for causal convolution (kernel-1 on left, 0 on right)
        let pad_widths: &[(i32, i32)] = &[(0, 0), (kernel - 1, 0), (0, 0)];
        let qkv_padded = mlx_rs::ops::pad(&qkv, pad_widths, None, None)?;
        // Depthwise conv1d: groups = conv_dim, weight [conv_dim, kernel, 1]
        let qkv_conv = mlx_rs::ops::conv1d(
            &qkv_padded, &*self.conv1d_weight, None, None, None, conv_dim as i32,
        )?;
        let qkv_conv = nn::silu(&qkv_conv)?;

        // 3. Split into Q, K, V (NOT equal thirds)
        let parts = qkv_conv.split_axis(&[key_dim, 2 * key_dim], -1)?;
        let q = parts[0].reshape(&[batch, seq_len, h, d])?;
        let k = parts[1].reshape(&[batch, seq_len, h, d])?;
        let v = parts[2].reshape(&[batch, seq_len, h, d])?;

        // 4. QK RMS normalization (weight-free, just normalize)
        let inv_scale = (d as f32).powf(-0.5);
        let ones_d = mlx_rs::ops::ones::<f32>(&[d])?;
        let q_flat = q.reshape(&[-1, d])?;
        let k_flat = k.reshape(&[-1, d])?;
        let q_norm = mlx_rs::fast::rms_norm(&q_flat, &ones_d, 1e-6)?
            .reshape(&[batch, seq_len, h, d])?;
        let k_norm = mlx_rs::fast::rms_norm(&k_flat, &ones_d, 1e-6)?
            .reshape(&[batch, seq_len, h, d])?;
        let q = q_norm.multiply(array!(inv_scale * inv_scale))?;
        let k = k_norm.multiply(array!(inv_scale))?;

        // 5. Compute decay g and write gate beta
        //    g = exp(-exp(A_log) * softplus(a + dt_bias))
        let a_plus_bias = a.add(&*self.dt_bias)?;           // [B, L, H]
        let sp = nn::softplus(&a_plus_bias)?;
        let decay_rate = mlx_rs::ops::exp(&*self.a_log)?.multiply(&sp)?;
        let g = mlx_rs::ops::exp(&decay_rate.negative()?)?;  // [B, L, H]
        let beta = nn::sigmoid(&b)?;                          // [B, L, H]

        // 6. Gated delta recurrence
        //    State: [B, H, D_v, D_k] — we accumulate outer products
        let mut state = mlx_rs::ops::zeros::<f32>(&[batch, h, d, d])?;
        let mut outputs: Vec<Array> = Vec::with_capacity(seq_len as usize);

        for t in 0..seq_len {
            // Slice [B, H] for this timestep
            let g_t = g.index((.., t, ..));       // [B, H]
            let beta_t = beta.index((.., t, ..)); // [B, H]
            let q_t = q.index((.., t, .., ..));   // [B, H, D]
            let k_t = k.index((.., t, .., ..));   // [B, H, D]
            let v_t = v.index((.., t, .., ..));   // [B, H, D]

            // state *= g (broadcast [B, H, 1, 1])
            let g_t = g_t.reshape(&[batch, h, 1, 1])?;
            state = state.multiply(&g_t)?;

            // kv_mem = (state * k[..., None, :]).sum(-1) → [B, H, D_v]
            let k_expanded = k_t.expand_dims(-2)?;  // [B, H, 1, D_k]
            let kv_mem = state.multiply(&k_expanded)?.sum_axes(&[-1], false)?; // [B, H, D_v]

            // delta = (v - kv_mem) * beta[..., None]
            let beta_t = beta_t.reshape(&[batch, h, 1])?;
            let delta = v_t.subtract(&kv_mem)?.multiply(&beta_t)?; // [B, H, D_v]

            // state += k[..., None, :] * delta[..., :, None]  (outer product update)
            let delta_expanded = delta.expand_dims(-1)?;   // [B, H, D_v, 1]
            state = state.add(&k_expanded.multiply(&delta_expanded)?)?;

            // y_t = (state * q[..., None, :]).sum(-1) → [B, H, D_v]
            let q_expanded = q_t.expand_dims(-2)?;  // [B, H, 1, D_k]
            let y_t = state.multiply(&q_expanded)?.sum_axes(&[-1], false)?; // [B, H, D_v]
            outputs.push(y_t);
        }

        // Stack: [B, H, D_v] * L → [B, L, H, D_v]
        let y = mlx_rs::ops::stack_axis(&outputs, 1)?;  // [B, L, H, D]

        // 7. RMSNormGated: rms_norm(out) on last dim (D), then silu(z) * normed
        //    z is [B, L, H, D], norm weight is [D=128]
        let z = self.in_proj_z.forward(x)?.reshape(&[batch, seq_len, h, d])?;
        let y_flat = y.reshape(&[-1, d])?;
        let y_normed = self.norm.forward(&y_flat)?
            .reshape(&[batch, seq_len, h, d])?;
        let y = nn::silu(&z)?.multiply(&y_normed)?;
        let y = y.reshape(&[batch, seq_len, h * d])?;

        self.out_proj.forward(&y)
    }
}

impl ModuleParameters for MlxLinearAttention {
    fn num_parameters(&self) -> usize {
        self.in_proj_qkv.num_parameters() + self.in_proj_a.num_parameters()
            + self.in_proj_b.num_parameters() + self.in_proj_z.num_parameters()
            + self.out_proj.num_parameters() + self.norm.num_parameters() + 3
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        nested_ref_from(vec![
            ("in_proj_qkv", self.in_proj_qkv.parameters()),
            ("in_proj_a", self.in_proj_a.parameters()),
            ("in_proj_b", self.in_proj_b.parameters()),
            ("in_proj_z", self.in_proj_z.parameters()),
            ("out_proj", self.out_proj.parameters()),
            ("norm", self.norm.parameters()),
        ])
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        nested_mut_from(vec![
            ("in_proj_qkv", self.in_proj_qkv.parameters_mut()),
            ("in_proj_a", self.in_proj_a.parameters_mut()),
            ("in_proj_b", self.in_proj_b.parameters_mut()),
            ("in_proj_z", self.in_proj_z.parameters_mut()),
            ("out_proj", self.out_proj.parameters_mut()),
            ("norm", self.norm.parameters_mut()),
        ])
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        NestedHashMap::new() // all frozen
    }

    fn freeze_parameters(&mut self, r: bool) {
        self.in_proj_qkv.freeze_parameters(r);
        self.in_proj_a.freeze_parameters(r);
        self.in_proj_b.freeze_parameters(r);
        self.in_proj_z.freeze_parameters(r);
        self.out_proj.freeze_parameters(r);
        self.norm.freeze_parameters(r);
    }

    fn unfreeze_parameters(&mut self, _r: bool) {}
    fn all_frozen(&self) -> Option<bool> { Some(true) }
    fn any_frozen(&self) -> Option<bool> { Some(true) }
}

// --- Decoder layer ---

/// Attention type: either full transformer attention (with LoRA) or linear Mamba SSM (frozen).
#[derive(Debug)]
enum AttentionKind {
    Full(MlxLoraAttention),
    Linear(MlxLinearAttention),
}

#[derive(Debug)]
pub struct MlxLoraDecoderLayer {
    attn: AttentionKind,
    pub mlp: MlxLoraMLP,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
}

impl MlxLoraDecoderLayer {
    pub fn load(
        weights: &mut HashMap<String, Array>,
        prefix: &str,
        layer_idx: usize,
        cfg: &ModelConfig,
        lora_cfg: &LoraConfig,
    ) -> Result<Self, anyhow::Error> {
        let attn = if cfg.is_linear_attn_layer(layer_idx) {
            AttentionKind::Linear(MlxLinearAttention::load(
                weights, &format!("{prefix}.linear_attn"), cfg,
            )?)
        } else {
            AttentionKind::Full(MlxLoraAttention::load(
                weights, &format!("{prefix}.self_attn"), cfg, lora_cfg,
            )?)
        };

        Ok(MlxLoraDecoderLayer {
            attn,
            mlp: MlxLoraMLP::load(weights, &format!("{prefix}.mlp"), cfg, lora_cfg)?,
            input_layernorm: load_rms_norm(weights, &format!("{prefix}.input_layernorm"), cfg.rms_eps)?,
            post_attention_layernorm: load_rms_norm(weights, &format!("{prefix}.post_attention_layernorm"), cfg.rms_eps)?,
        })
    }

    /// Test-only: run the linear_attn sub-module directly on pre-layernormed input.
    #[cfg(test)]
    pub fn forward_linear_attn(&mut self, x: &Array) -> Result<Array, Exception> {
        match &mut self.attn {
            AttentionKind::Linear(attn) => attn.forward(x),
            AttentionKind::Full(_) => panic!("not a linear_attn layer"),
        }
    }

    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        let residual = x;
        let h = self.input_layernorm.forward(x)?;
        let h = match &mut self.attn {
            AttentionKind::Full(attn) => attn.forward(&h, mask)?,
            AttentionKind::Linear(attn) => attn.forward(&h)?,
        };
        let x = h.add(residual)?;

        let residual = &x;
        let h = self.post_attention_layernorm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        h.add(residual)
    }
}

impl ModuleParameters for MlxLoraDecoderLayer {
    fn num_parameters(&self) -> usize {
        let attn_params = match &self.attn {
            AttentionKind::Full(a) => a.num_parameters(),
            AttentionKind::Linear(a) => a.num_parameters(),
        };
        attn_params + self.mlp.num_parameters()
            + self.input_layernorm.num_parameters() + self.post_attention_layernorm.num_parameters()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let attn_key = match &self.attn {
            AttentionKind::Full(a) => ("self_attn", a.parameters()),
            AttentionKind::Linear(a) => ("linear_attn", a.parameters()),
        };
        nested_ref_from(vec![
            attn_key,
            ("mlp", self.mlp.parameters()),
            ("input_layernorm", self.input_layernorm.parameters()),
            ("post_attention_layernorm", self.post_attention_layernorm.parameters()),
        ])
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let attn_key = match &mut self.attn {
            AttentionKind::Full(a) => ("self_attn", a.parameters_mut()),
            AttentionKind::Linear(a) => ("linear_attn", a.parameters_mut()),
        };
        nested_mut_from(vec![
            attn_key,
            ("mlp", self.mlp.parameters_mut()),
            ("input_layernorm", self.input_layernorm.parameters_mut()),
            ("post_attention_layernorm", self.post_attention_layernorm.parameters_mut()),
        ])
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        let mut entries = vec![
            ("mlp", self.mlp.trainable_parameters()),
        ];
        if let AttentionKind::Full(a) = &self.attn {
            entries.push(("self_attn", a.trainable_parameters()));
        }
        nested_ref_from(entries)
    }

    fn freeze_parameters(&mut self, r: bool) {
        match &mut self.attn {
            AttentionKind::Full(a) => a.freeze_parameters(r),
            AttentionKind::Linear(a) => a.freeze_parameters(r),
        }
        self.mlp.freeze_parameters(r);
        self.input_layernorm.freeze_parameters(r);
        self.post_attention_layernorm.freeze_parameters(r);
    }

    fn unfreeze_parameters(&mut self, r: bool) {
        if let AttentionKind::Full(a) = &mut self.attn {
            a.unfreeze_parameters(r);
        }
        self.mlp.unfreeze_parameters(r);
    }

    fn all_frozen(&self) -> Option<bool> {
        let attn_frozen = match &self.attn {
            AttentionKind::Full(a) => a.all_frozen().unwrap_or(true),
            AttentionKind::Linear(_) => true,
        };
        Some(attn_frozen && self.mlp.all_frozen().unwrap_or(true))
    }

    fn any_frozen(&self) -> Option<bool> {
        Some(true)
    }
}

// --- Full model ---

#[derive(Debug)]
pub struct MlxLoraModel {
    pub embed_tokens: MlxEmbedding,
    pub layers: Vec<MlxLoraDecoderLayer>,
    pub norm: RmsNorm,
    /// None = tied to embed_tokens (use QuantizedEmbedding::as_linear)
    pub lm_head: Option<QuantizedLinear>,
}

impl MlxLoraModel {
    pub fn load(
        model_dir: &Path,
        cfg: &ModelConfig,
        lora_cfg: &LoraConfig,
    ) -> Result<Self, anyhow::Error> {
        eprintln!("loading weights from {}...", model_dir.display());
        let t0 = std::time::Instant::now();
        let mut weights = load_weights(model_dir)?;
        eprintln!("loaded {} tensors in {}ms", weights.len(), t0.elapsed().as_millis());

        let pfx = cfg.weight_prefix;

        let embed_tokens = load_embedding(&mut weights, &format!("{pfx}.embed_tokens"), cfg.group_size, cfg.bits)?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(MlxLoraDecoderLayer::load(
                &mut weights, &format!("{pfx}.layers.{i}"), i, cfg, lora_cfg,
            )?);
        }

        let norm = load_rms_norm(&mut weights, &format!("{pfx}.norm"), cfg.rms_eps)?;

        let lm_head_key = if pfx == "model" { "lm_head" } else { "language_model.lm_head" };
        let lm_head = if weights.contains_key(&format!("{lm_head_key}.weight")) {
            Some(load_quantized_linear(&mut weights, lm_head_key, cfg.group_size, cfg.bits)?)
        } else {
            None // tied to embed_tokens
        };

        // Filter out vision_tower weights (not needed for text LoRA)
        weights.retain(|k, _| !k.starts_with("vision_tower"));

        Ok(MlxLoraModel { embed_tokens, layers, norm, lm_head })
    }

    /// Greedy autoregressive text generation.
    ///
    /// Takes a prompt token sequence and generates up to `max_tokens` new tokens.
    /// Uses temperature sampling when temp > 0, greedy argmax otherwise.
    /// Returns the generated token IDs (NOT including the prompt).
    pub fn generate(
        &mut self,
        prompt_tokens: &[i32],
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[i32],
    ) -> Result<Vec<i32>, Exception> {
        let mut tokens = prompt_tokens.to_vec();
        let mut generated = Vec::with_capacity(max_tokens);

        for _ in 0..max_tokens {
            let input = Array::from_slice(&tokens, &[1, tokens.len() as i32]);
            let logits = self.forward_logits(&input)?;

            // Take logits for last position: [1, seq, vocab] → [1, vocab]
            let last_logits = logits.index((.., -1, ..));

            let next_token = if temperature <= 0.0 || temperature < 1e-6 {
                // Greedy
                mlx_rs::ops::indexing::argmax_axis(&last_logits, -1, false)?
            } else {
                // Temperature sampling
                let scaled = last_logits.multiply(array!(1.0 / temperature))?;
                mlx_rs::random::categorical(&scaled, None, None, None)?
            };

            let next_token = next_token.as_dtype(mlx_rs::Dtype::Int32)?;
            let token_id: i32 = next_token.as_slice::<i32>()[0];

            if stop_tokens.contains(&token_id) {
                break;
            }

            generated.push(token_id);
            tokens.push(token_id);
        }

        Ok(generated)
    }

    pub fn generate_text(
        &mut self,
        tokenizer: &MlxTokenizer,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<String, anyhow::Error> {
        let prompt_tokens = tokenizer.encode(prompt)?;
        let eos = tokenizer.eos_token_id().unwrap_or(248046) as i32;
        let generated = self.generate(&prompt_tokens, max_tokens, temperature, &[eos])
            .map_err(|e| anyhow::anyhow!("Generation failed: {e}"))?;
        tokenizer.decode(&generated)
    }

    pub fn forward_logits(&mut self, tokens: &Array) -> Result<Array, Exception> {
        let seq_len = tokens.shape()[1] as i32;
        let mut h = self.embed_tokens.forward(tokens)?;
        let mask = create_causal_mask(seq_len)?;

        for layer in &mut self.layers {
            h = layer.forward(&h, Some(&mask))?;
        }

        h = self.norm.forward(&h)?;
        if let Some(lm) = &mut self.lm_head {
            lm.forward(&h)
        } else {
            match &self.embed_tokens {
                MlxEmbedding::Quantized(qe) => qe.as_linear(&h),
                MlxEmbedding::Plain(e) => {
                    // Tied weights: x @ embed.T
                    mlx_rs::ops::matmul(&h, &e.weight.transpose_axes(&[1, 0])?)
                }
            }
        }
    }
}

impl ModuleParameters for MlxLoraModel {
    fn num_parameters(&self) -> usize {
        self.embed_tokens.num_parameters()
            + self.layers.iter().map(|l| l.num_parameters()).sum::<usize>()
            + self.norm.num_parameters()
            + self.lm_head.as_ref().map_or(0, |l| l.num_parameters())
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut map = NestedHashMap::new();
        map.insert(Rc::from("embed_tokens"), self.embed_tokens.parameters().into());
        let mut layers_entries = HashMap::new();
        for (i, l) in self.layers.iter().enumerate() {
            layers_entries.insert(Rc::from(i.to_string().as_str()), l.parameters().into());
        }
        map.insert(Rc::from("layers"), NestedValue::Map(layers_entries));
        map.insert(Rc::from("norm"), self.norm.parameters().into());
        if let Some(lm) = &self.lm_head {
            map.insert(Rc::from("lm_head"), lm.parameters().into());
        }
        map
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut map = NestedHashMap::new();
        map.insert(Rc::from("embed_tokens"), self.embed_tokens.parameters_mut().into());
        let mut layers_entries = HashMap::new();
        for (i, l) in self.layers.iter_mut().enumerate() {
            layers_entries.insert(Rc::from(i.to_string().as_str()), l.parameters_mut().into());
        }
        map.insert(Rc::from("layers"), NestedValue::Map(layers_entries));
        map.insert(Rc::from("norm"), self.norm.parameters_mut().into());
        if let Some(lm) = &mut self.lm_head {
            map.insert(Rc::from("lm_head"), lm.parameters_mut().into());
        }
        map
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        let mut map = NestedHashMap::new();
        let mut layers_entries = HashMap::new();
        for (i, l) in self.layers.iter().enumerate() {
            layers_entries.insert(Rc::from(i.to_string().as_str()), l.trainable_parameters().into());
        }
        map.insert(Rc::from("layers"), NestedValue::Map(layers_entries));
        map
    }

    fn freeze_parameters(&mut self, r: bool) {
        self.embed_tokens.freeze_parameters(r);
        for l in &mut self.layers { l.freeze_parameters(r); }
        self.norm.freeze_parameters(r);
        if let Some(lm) = &mut self.lm_head { lm.freeze_parameters(r); }
    }

    fn unfreeze_parameters(&mut self, r: bool) {
        for l in &mut self.layers { l.unfreeze_parameters(r); }
    }

    fn all_frozen(&self) -> Option<bool> {
        Some(self.layers.iter().all(|l| l.all_frozen().unwrap_or(true)))
    }

    fn any_frozen(&self) -> Option<bool> {
        Some(true)
    }
}

// ---------------------------------------------------------------------------
// Causal mask
// ---------------------------------------------------------------------------

fn create_causal_mask(seq_len: i32) -> Result<Array, Exception> {
    let data: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..seq_len).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY })
        })
        .collect();
    Ok(Array::from_slice(&data, &[1, 1, seq_len, seq_len]))
}

// ---------------------------------------------------------------------------
// Cross-entropy loss
// ---------------------------------------------------------------------------

/// Cross-entropy loss: logits [batch, seq, vocab], targets [batch, seq] -> scalar.
///
/// Uses mlx-rs built-in CrossEntropy which computes `logsumexp(logits) - logits[target]`
/// instead of materializing the full log_softmax gradient over the vocab dimension.
/// This avoids NaN gradients on large vocab (248k) with real-length sequences.
pub fn cross_entropy_loss(logits: &Array, targets: &Array) -> Result<Array, Exception> {
    let shape = logits.shape();
    let vocab = shape[shape.len() - 1];

    let flat_logits = logits.reshape(&[-1, vocab])?;
    let flat_targets = targets.reshape(&[-1])?.as_dtype(mlx_rs::Dtype::Int32)?;

    let ce = CrossEntropyBuilder::new()
        .reduction(LossReduction::Mean)
        .build()?;

    ce.apply(&flat_logits, &flat_targets)
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

pub fn train_step(
    model: &mut MlxLoraModel,
    optimizer: &mut AdamW,
    tokens: &Array,
    targets: &Array,
    grad_clip: f32,
) -> Result<(f32, f32), anyhow::Error> {
    let loss_fn =
        |model: &mut MlxLoraModel, (toks, tgts): (&Array, &Array)| -> Result<Array, Exception> {
            let logits = model.forward_logits(toks)?;
            cross_entropy_loss(&logits, tgts)
        };

    let mut vg = nn::value_and_grad(loss_fn);

    let (loss, grads) = vg(model, (tokens, targets))
        .map_err(|e| anyhow::anyhow!("value_and_grad: {e}"))?;

    let (clipped, total_norm) = mlx_rs::optimizers::clip_grad_norm(&grads, grad_clip)
        .map_err(|e| anyhow::anyhow!("clip_grad_norm: {e}"))?;

    // Skip optimizer update if gradient is NaN (prevents permanent weight corruption).
    if total_norm.is_finite() {
        let owned_grads: HashMap<Rc<str>, Array> =
            clipped.into_iter().map(|(k, v)| (k, v.into_owned())).collect();

        optimizer
            .update(model, &owned_grads)
            .map_err(|e| anyhow::anyhow!("optimizer update: {e}"))?;
    } else {
        eprintln!("    [skipped NaN gradient update]");
    }

    // Evaluate loss, model parameters, AND optimizer state — matching Python mlx_lm.
    let params = model.parameters().flatten();
    let mut eval_targets: Vec<&Array> = vec![&loss];
    eval_targets.extend(params.values());
    mlx_rs::transforms::eval(eval_targets.into_iter())
        .map_err(|e| anyhow::anyhow!("eval: {e}"))?;

    let loss_val: f32 = loss.as_slice::<f32>()[0];
    Ok((loss_val, total_norm))
}

/// Optional callback invoked after each training step with (step, loss, grad_norm).
pub type TrainCallback = Box<dyn FnMut(usize, f32, f32) + Send>;

pub fn train_loop(
    model: &mut MlxLoraModel,
    tokens_list: &[Array],
    targets_list: &[Array],
    config: &LoraConfig,
    max_steps: usize,
    early_stop_loss: f32,
    patience: usize,
) -> Result<Vec<f32>, anyhow::Error> {
    train_loop_with_callback(model, tokens_list, targets_list, config, max_steps, early_stop_loss, patience, None)
}

pub fn train_loop_with_callback(
    model: &mut MlxLoraModel,
    tokens_list: &[Array],
    targets_list: &[Array],
    config: &LoraConfig,
    max_steps: usize,
    early_stop_loss: f32,
    patience: usize,
    mut on_step: Option<TrainCallback>,
) -> Result<Vec<f32>, anyhow::Error> {
    let mut optimizer = AdamW::new(config.lr);
    optimizer.weight_decay = array!(config.weight_decay);

    let mut losses = Vec::with_capacity(max_steps);
    let mut steps_without_improvement = 0;
    let mut best_loss = f32::MAX;
    let n_examples = tokens_list.len();

    for step in 0..max_steps {
        let idx = step % n_examples;
        let t0 = std::time::Instant::now();

        let (loss, grad_norm) = train_step(
            model, &mut optimizer,
            &tokens_list[idx], &targets_list[idx],
            config.grad_clip,
        )?;

        let ms = t0.elapsed().as_millis();
        eprintln!("  step {step}: loss={loss:.4}, grad_norm={grad_norm:.4}, time={ms}ms");
        losses.push(loss);

        if let Some(ref mut cb) = on_step {
            cb(step, loss, grad_norm);
        }

        if loss < best_loss {
            best_loss = loss;
            steps_without_improvement = 0;
        } else {
            steps_without_improvement += 1;
        }

        if loss < early_stop_loss {
            eprintln!("  early stop: loss {loss:.4} < threshold {early_stop_loss:.4}");
            break;
        }

        if steps_without_improvement >= patience {
            eprintln!("  early stop: no improvement for {patience} steps");
            break;
        }
    }

    Ok(losses)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask() {
        let mask = create_causal_mask(3).unwrap();
        assert_eq!(mask.shape(), &[1, 1, 3, 3]);
        let data: &[f32] = mask.as_slice();
        assert_eq!(data[0], 0.0);
        assert!(data[1].is_infinite());
        assert_eq!(data[3], 0.0);
        assert_eq!(data[4], 0.0);
        assert!(data[5].is_infinite());
    }

    #[test]
    fn test_lora_linear_shapes() {
        let ql = QuantizedLinear::new(128, 64).unwrap();
        let lora = LoraLinear::new(ql, 4, 1.0).unwrap();
        assert_eq!(lora.lora_a.weight.shape(), &[4, 128]);
        assert_eq!(lora.lora_b.weight.shape(), &[64, 4]);
    }

    #[test]
    fn test_lora_linear_forward() {
        let ql = QuantizedLinear::new(128, 64).unwrap();
        let mut lora = LoraLinear::new(ql, 4, 1.0).unwrap();
        let x = Array::from_slice(&vec![1.0f32; 128], &[1, 128]);
        let out = lora.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_lora_linear_zero_init() {
        let ql = QuantizedLinear::new(128, 64).unwrap();
        let mut lora = LoraLinear::new(ql, 4, 1.0).unwrap();
        let x = Array::from_slice(&vec![1.0f32; 128], &[1, 128]);
        let base_out = lora.base.forward(&x).unwrap();
        let lora_out = lora.forward(&x).unwrap();
        let diff = lora_out.subtract(&base_out).unwrap().abs().unwrap().sum(None).unwrap();
        let diff_val: f32 = diff.as_slice::<f32>()[0];
        assert!(diff_val < 1e-6, "LoRA should not change output when B=0, got diff={diff_val}");
    }

    #[test]
    fn test_cross_entropy_loss() {
        let logits = Array::from_slice(
            &[0.0f32, 0.0, 10.0, 0.0,  10.0, 0.0, 0.0, 0.0],
            &[1, 2, 4],
        );
        let targets = Array::from_slice(&[2i32, 0], &[1, 2]);
        let loss = cross_entropy_loss(&logits, &targets).unwrap();
        let loss_val: f32 = loss.as_slice::<f32>()[0];
        assert!(loss_val < 0.1, "loss should be small, got {loss_val}");
    }

    /// Regression test: our old log_softmax-based CE produced NaN gradients on
    /// large vocab (248k) with sequences longer than ~10 tokens. The built-in
    /// CrossEntropy uses logsumexp - score which avoids materializing the full
    /// log_softmax gradient.
    #[test]
    fn test_cross_entropy_grad_large_vocab_no_nan() {
        let vocab = 248320i32;
        let seq_len = 64;
        // Random logits in a realistic range
        let logits = mlx_rs::random::normal::<f32>(
            &[1, seq_len, vocab],
            None, None, None,
        ).unwrap();
        // Random target indices in [0, vocab)
        let targets = mlx_rs::random::randint::<_, i32>(
            0, vocab, &[1, seq_len], None,
        ).unwrap();

        // Forward: loss must be finite
        let loss = cross_entropy_loss(&logits, &targets).unwrap();
        mlx_rs::transforms::eval(std::iter::once(&loss)).unwrap();
        let loss_val: f32 = loss.as_slice::<f32>()[0];
        assert!(loss_val.is_finite(), "loss should be finite, got {loss_val}");
        eprintln!("large vocab CE loss = {loss_val:.4}");

        // Backward: gradient via grad() must be finite
        let mut grad_fn = mlx_rs::transforms::grad(
            |logits: &Array| -> Result<Array, Exception> {
                cross_entropy_loss(logits, &targets)
            },
        );
        let grad = grad_fn(&logits).unwrap();
        mlx_rs::transforms::eval(std::iter::once(&grad)).unwrap();
        let grad_sum: f32 = grad.sum(None).unwrap().as_slice::<f32>()[0];
        assert!(grad_sum.is_finite(), "gradient should be finite, got {grad_sum}");
        eprintln!("large vocab CE grad_sum = {grad_sum:.6}");
    }

    #[test]
    fn test_trainable_param_count() {
        let ql = QuantizedLinear::new(128, 64).unwrap();
        let lora = LoraLinear::new(ql, 4, 1.0).unwrap();
        let trainable = lora.trainable_parameters().flatten();
        assert_eq!(trainable.len(), 2, "expected 2 trainable params, got {:?}", trainable.keys().collect::<Vec<_>>());
    }

    #[test]
    fn test_lora_linear_grad() {
        let ql = QuantizedLinear::new(128, 64).unwrap();
        let mut lora = LoraLinear::new(ql, 4, 1.0).unwrap();
        let x = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[1, 128], None).unwrap();

        let loss_fn = |model: &mut LoraLinear, x: &Array| -> Result<Array, Exception> {
            model.forward(x)?.sum(None)
        };

        let mut vg = nn::value_and_grad(loss_fn);
        let (val, grads) = vg(&mut lora, &x).unwrap();

        assert!(val.as_slice::<f32>()[0].is_finite(), "loss should be finite");
        assert!(grads.len() >= 2, "should have grads for LoRA A and B, got {}", grads.len());
    }

    #[test]
    fn test_e2e_qwen3_lora_mlx() {
        let model_dir = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/lmstudio-community/Qwen3-1.7B-MLX-8bit");
        if !model_dir.exists() {
            eprintln!("Qwen3-1.7B-MLX-8bit not found, skipping E2E test");
            return;
        }

        let cfg = ModelConfig::qwen3_1_7b();
        let lora_cfg = LoraConfig::default();

        let mut model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg)
            .expect("model load failed");

        let trainable = model.trainable_parameters().flatten();
        let total_trainable: usize = trainable.values()
            .map(|a| a.shape().iter().product::<i32>() as usize)
            .sum();
        eprintln!("trainable parameters: {total_trainable}");
        assert_eq!(trainable.len(), 224, "expected 224 trainable params");

        let tokens = Array::from_slice(&[100i32, 101, 102, 103], &[1, 4]);
        let targets = Array::from_slice(&[101i32, 102, 103, 104], &[1, 4]);

        let losses = train_loop(
            &mut model, &[tokens], &[targets],
            &lora_cfg, 3, 0.5, 10,
        ).expect("training failed");

        assert!(losses.len() >= 2);
        let first = losses[0];
        let last = losses[losses.len() - 1];
        eprintln!("loss: {first:.4} -> {last:.4}");
        assert!(first.is_finite());
        assert!(last < first, "loss should decrease: {first:.4} -> {last:.4}");
    }

    #[test]
    fn test_e2e_qwen3_5_2b_lora_mlx() {
        let model_dir = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-2B-MLX-8bit");
        if !model_dir.exists() {
            eprintln!("Qwen3.5-2B-MLX-8bit not found, skipping E2E test");
            return;
        }

        let cfg = ModelConfig::qwen3_5_2b();
        let lora_cfg = LoraConfig::default();

        eprintln!("loading Qwen3.5-2B hybrid model (18 linear + 6 full attn layers)...");
        let mut model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg)
            .expect("model load failed");

        let trainable = model.trainable_parameters().flatten();
        let total_trainable: usize = trainable.values()
            .map(|a| a.shape().iter().product::<i32>() as usize)
            .sum();
        eprintln!("trainable parameters: {total_trainable} ({} param tensors)", trainable.len());

        // 6 full-attn layers × 3 LoRA targets (q, v, o) × 2 (A+B) = 36
        // 24 layers × 1 LoRA target (down_proj) × 2 (A+B) = 48
        // Total: 84
        let expected_lora_params = 84;
        assert_eq!(
            trainable.len(), expected_lora_params,
            "expected {expected_lora_params} trainable params, got {}", trainable.len()
        );

        let tokens = Array::from_slice(&[100i32, 101, 102, 103], &[1, 4]);
        let targets = Array::from_slice(&[101i32, 102, 103, 104], &[1, 4]);

        let losses = train_loop(
            &mut model, &[tokens], &[targets],
            &lora_cfg, 3, 0.5, 10,
        ).expect("training failed");

        assert!(losses.len() >= 2, "expected at least 2 training steps");
        let first = losses[0];
        let last = losses[losses.len() - 1];
        eprintln!("loss: {first:.4} -> {last:.4}");
        assert!(first.is_finite(), "first loss should be finite");
        assert!(last < first, "loss should decrease: {first:.4} -> {last:.4}");
    }

    /// Regression test: train on a real-length tokenized sequence (64 tokens)
    /// to verify no NaN gradients. Previously failed with log_softmax-based CE.
    #[test]
    fn test_e2e_qwen3_5_2b_long_sequence_no_nan() {
        let model_dir = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-2B-MLX-8bit");
        if !model_dir.exists() {
            eprintln!("Qwen3.5-2B-MLX-8bit not found, skipping");
            return;
        }

        let cfg = ModelConfig::qwen3_5_2b();
        let lora_cfg = LoraConfig { lr: 1e-5, ..LoraConfig::default() };

        eprintln!("loading model for long-sequence NaN regression test...");
        let mut model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg)
            .expect("model load failed");

        // Simulate a real conversation: 64 random token IDs in valid vocab range
        let seq: Vec<i32> = (0..65).map(|i| ((i * 7919 + 1337) % 248320) as i32).collect();
        let tokens = Array::from_slice(&seq[..64], &[1, 64]);
        let targets = Array::from_slice(&seq[1..65], &[1, 64]);

        let losses = train_loop(
            &mut model, &[tokens], &[targets],
            &lora_cfg, 3, 0.5, 10,
        ).expect("training failed — likely NaN gradient regression");

        for (i, l) in losses.iter().enumerate() {
            eprintln!("  step {i}: loss={l:.4}");
            assert!(l.is_finite(), "step {i} loss is NaN/Inf: {l}");
        }
        assert!(losses.len() >= 2);
        assert!(losses.last().unwrap() < &losses[0], "loss should decrease");
    }

    /// Load a raw f32 binary tensor from a reference directory under tests/
    fn load_reference_tensor_from(dir: &str, name: &str, shape: &[i32]) -> Array {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(dir)
            .join(format!("{name}.bin"));
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}\nRun the appropriate reference script", path.display()));
        let floats: Vec<f32> = bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Array::from_slice(&floats, shape)
    }

    /// Compare two arrays element-wise, return (max_abs_diff, mean_abs_diff).
    fn compare_arrays(a: &Array, b: &Array) -> (f32, f32) {
        let diff = a.subtract(b).unwrap();
        let abs_diff = mlx_rs::ops::abs(&diff).unwrap();
        let max_diff: f32 = abs_diff.max(None).unwrap().as_slice::<f32>()[0];
        let mean_diff: f32 = abs_diff.mean(None).unwrap().as_slice::<f32>()[0];
        (max_diff, mean_diff)
    }

    #[test]
    fn test_gdn_numerical_vs_python_reference() {
        // Check reference files exist
        let ref_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/gdn_reference_raw");
        if !ref_dir.join("manifest.json").exists() {
            eprintln!("Reference tensors not found. Run: python3 tests/gdn_reference.py");
            return;
        }

        let model_dir = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-2B-MLX-8bit");
        if !model_dir.exists() {
            eprintln!("Qwen3.5-2B-MLX-8bit not found, skipping");
            return;
        }

        // Load model (same weights as Python reference)
        let cfg = ModelConfig::qwen3_5_2b();
        let lora_cfg = LoraConfig::default();
        let mut model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg)
            .expect("model load failed");

        // Load reference input [1, 4, 2048] and expected output [1, 4, 2048]
        let ref_input = load_reference_tensor_from("tests/gdn_reference_raw","input", &[1, 4, 2048]);
        let ref_output = load_reference_tensor_from("tests/gdn_reference_raw","output", &[1, 4, 2048]);

        // Run our forward on layer 0 (which is a linear_attn layer)
        // The reference was computed on raw input (before layernorm), so we must
        // call the linear_attn sub-module directly
        let our_output = model.layers[0].forward_linear_attn(&ref_input)
            .expect("forward failed");

        // Compare
        let (max_diff, mean_diff) = compare_arrays(&our_output, &ref_output);
        eprintln!("GDN output vs Python reference:");
        eprintln!("  max_abs_diff:  {max_diff:.6e}");
        eprintln!("  mean_abs_diff: {mean_diff:.6e}");
        eprintln!("  our shape:     {:?}", our_output.shape());
        eprintln!("  ref shape:     {:?}", ref_output.shape());

        // Tolerance: quantized model with bf16 intermediates — expect ~1e-2 to 1e-3
        // Python runs bf16 on GPU, our Rust runs bf16 quantized path on GPU
        assert!(
            max_diff < 0.05,
            "max diff {max_diff:.6e} exceeds tolerance 0.05 — implementation mismatch"
        );
        assert!(
            mean_diff < 0.01,
            "mean diff {mean_diff:.6e} exceeds tolerance 0.01 — implementation mismatch"
        );
        eprintln!("PASS: Gated Delta Net matches Python reference within tolerance");

        // Also compare intermediate tensors for debugging
        let ref_recurrence = load_reference_tensor_from("tests/gdn_reference_raw","recurrence_out", &[1, 4, 16, 128]);
        eprintln!("\nIntermediate reference tensors (for debugging):");
        eprintln!("  recurrence_out sample: {:?}",
            &ref_recurrence.as_slice::<f32>()[..8]);
    }

    #[test]
    fn test_full_attn_numerical_vs_python_reference() {
        let ref_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/full_attn_reference_raw");
        if !ref_dir.join("manifest.json").exists() {
            eprintln!("Reference tensors not found. Run: python3 tests/full_attn_reference.py");
            return;
        }

        let model_dir = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-2B-MLX-8bit");
        if !model_dir.exists() {
            eprintln!("Qwen3.5-2B-MLX-8bit not found, skipping");
            return;
        }

        let cfg = ModelConfig::qwen3_5_2b();
        let lora_cfg = LoraConfig::default();
        let mut model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg)
            .expect("model load failed");

        let d = "tests/full_attn_reference_raw";
        let ref_input = load_reference_tensor_from(d, "attn_input", &[1, 4, 2048]);
        let ref_output = load_reference_tensor_from(d, "attn_output", &[1, 4, 2048]);

        // Call forward() directly — it uses Causal mask, matching the reference
        let our_output = match &mut model.layers[3].attn {
            AttentionKind::Full(attn) => attn.forward(&ref_input, None).expect("forward failed"),
            AttentionKind::Linear(_) => panic!("layer 3 should be full attention"),
        };

        let (max_diff, mean_diff) = compare_arrays(&our_output, &ref_output);
        eprintln!("Full attention output vs Python reference:");
        eprintln!("  max_abs_diff:  {max_diff:.6e}");
        eprintln!("  mean_abs_diff: {mean_diff:.6e}");

        assert!(
            max_diff < 0.05,
            "max diff {max_diff:.6e} exceeds tolerance 0.05 — implementation mismatch"
        );
        assert!(
            mean_diff < 0.01,
            "mean diff {mean_diff:.6e} exceeds tolerance 0.01 — implementation mismatch"
        );
    }

    #[test]
    fn test_generate_greedy() {
        let model_dir = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-2B-MLX-8bit");
        if !model_dir.exists() {
            eprintln!("Qwen3.5-2B-MLX-8bit not found, skipping");
            return;
        }

        let cfg = ModelConfig::qwen3_5_2b();
        let lora_cfg = LoraConfig::default();
        let mut model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg)
            .expect("model load failed");

        // "The capital of France is" → Python produces token 11751 (Paris) first
        let prompt = &[760i32, 6511, 314, 9338, 369];

        let eos = 248046;

        // Layer-by-layer comparison with Python
        {
            let input = Array::from_slice(prompt, &[1, 5]);
            let mut h = model.embed_tokens.forward(&input).unwrap();
            for (i, layer) in model.layers.iter_mut().enumerate() {
                h = layer.forward(&h, None).unwrap();
                if [0,1,2,3,5,11,23].contains(&i) {
                    let hf = h.as_dtype(mlx_rs::Dtype::Float32).unwrap();
                    let flat = hf.reshape(&[-1]).unwrap();
                    let n = hf.shape().iter().product::<i32>();
                    // last position (idx 4), first 3 dims: offset = 4*2048
                    let off = 4 * 2048;
                    let v: Vec<f32> = (off..off+3).map(|j| flat.index(j as i32).as_slice::<f32>()[0]).collect();
                    eprintln!("layer {i:2}: {:?}", v);
                }
            }
        }

        // Reload model for clean forward
        let mut model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg).expect("reload");

        // First check logits match Python reference
        let input = Array::from_slice(prompt, &[1, 5]);
        let logits = model.forward_logits(&input).expect("forward failed");
        let last_logits = logits.index((.., -1, ..)); // [1, vocab]
        let argmax_token = mlx_rs::ops::indexing::argmax_axis(&last_logits, -1, false)
            .unwrap().as_dtype(mlx_rs::Dtype::Int32).unwrap();
        let argmax_id: i32 = argmax_token.as_slice::<i32>()[0];
        eprintln!("argmax token: {argmax_id} (expected 11751 = 'Paris')");

        // Check top-5 logit values
        let last_flat = last_logits.reshape(&[-1]).unwrap();
        for &tid in &[11751i32, 279, 264, 3750, 13] {
            let val: f32 = last_flat.index(tid).as_slice::<f32>()[0];
            eprintln!("  token {tid}: logit {val:.4}");
        }

        eprintln!("\ngenerating (greedy, max 20 tokens)...");
        let t0 = std::time::Instant::now();
        let generated = model.generate(prompt, 20, 0.0, &[eos])
            .expect("generation failed");
        let elapsed = t0.elapsed();

        eprintln!("generated {} tokens in {:.1}s ({:.0} ms/token)",
            generated.len(), elapsed.as_secs_f64(),
            elapsed.as_millis() as f64 / generated.len().max(1) as f64);
        eprintln!("tokens: {:?}", generated);

        assert!(!generated.is_empty(), "should generate at least one token");
        assert_eq!(generated[0], 11751,
            "first generated token should be 11751 (Paris), got {}", generated[0]);
    }

    #[test]
    fn test_tokenizer_roundtrip() {
        let model_dir = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-2B-MLX-8bit");
        if !model_dir.exists() {
            eprintln!("Qwen3.5-2B-MLX-8bit not found, skipping");
            return;
        }

        let tok = MlxTokenizer::load(&model_dir).expect("tokenizer load failed");

        // Encode/decode roundtrip
        let text = "The capital of France is Paris.";
        let ids = tok.encode(text).expect("encode failed");
        eprintln!("encoded '{}' -> {:?} ({} tokens)", text, ids, ids.len());
        assert!(!ids.is_empty());

        let decoded = tok.decode(&ids).expect("decode failed");
        eprintln!("decoded -> '{}'", decoded);
        assert_eq!(decoded, text);

        // Verify known prompt matches hardcoded tokens from test_generate_greedy
        let prompt = "The capital of France is";
        let prompt_ids = tok.encode(prompt).expect("encode failed");
        eprintln!("prompt '{}' -> {:?}", prompt, prompt_ids);
        assert_eq!(prompt_ids, vec![760, 6511, 314, 9338, 369],
            "tokenizer should match hardcoded prompt tokens");

        // EOS token
        let eos = tok.eos_token_id();
        eprintln!("eos_token_id: {:?}", eos);
        assert!(eos.is_some(), "should find an EOS token");
    }

    #[test]
    fn test_generate_text_e2e() {
        let model_dir = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-2B-MLX-8bit");
        if !model_dir.exists() {
            eprintln!("Qwen3.5-2B-MLX-8bit not found, skipping");
            return;
        }

        let cfg = ModelConfig::qwen3_5_2b();
        let lora_cfg = LoraConfig::default();
        let mut model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg)
            .expect("model load failed");
        let tok = MlxTokenizer::load(&model_dir).expect("tokenizer load failed");

        let t0 = std::time::Instant::now();
        let response = model.generate_text(&tok, "The capital of France is", 10, 0.0)
            .expect("generate_text failed");
        let elapsed = t0.elapsed();

        eprintln!("generate_text: '{}' ({:.1}s)", response, elapsed.as_secs_f64());
        assert!(response.contains("Paris"),
            "response should mention Paris, got: '{}'", response);
    }
}
