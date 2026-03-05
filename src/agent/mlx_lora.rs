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
use mlx_rs::{array, Array};

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

        // When attn_output_gate is enabled, q_proj outputs [Q, gate] concatenated.
        // Split: first half is Q, second half is the sigmoid gate for the output.
        let (q, output_gate) = if self.attn_output_gate {
            let parts = q_raw.split(2, -1)?;
            (parts[0].clone(), Some(parts[1].clone()))
        } else {
            (q_raw, None)
        };

        // QK norm before reshape (weight shape matches projected dim, not head_dim)
        let q_norm_size = self.q_norm.weight.shape()[0] as i32;
        let q = if q_norm_size == self.head_dim {
            let q = q.reshape(&[batch * seq_len * self.num_heads, self.head_dim])?;
            let q = self.q_norm.forward(&q)?;
            q.reshape(&[batch, seq_len, self.num_heads * self.head_dim])?
        } else {
            self.q_norm.forward(&q)?
        };
        let k_norm_size = self.k_norm.weight.shape()[0] as i32;
        let k = if k_norm_size == self.head_dim {
            let k = k.reshape(&[batch * seq_len * self.num_kv_heads, self.head_dim])?;
            let k = self.k_norm.forward(&k)?;
            k.reshape(&[batch, seq_len, self.num_kv_heads * self.head_dim])?
        } else {
            self.k_norm.forward(&k)?
        };

        let q = q.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let k = k.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?;
        let v = v.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?;

        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        let q = mlx_rs::fast::rope(&q, self.rope_dims, false, self.rope_base, 1.0, 0, None::<&Array>)?;
        let k = mlx_rs::fast::rope(&k, self.rope_dims, false, self.rope_base, 1.0, 0, None::<&Array>)?;

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
                None::<ScaledDotProductAttentionMask>,
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

    /// Simplified Mamba2 SSM forward.
    ///
    /// Uses causal attention as an approximation of the full selective scan.
    /// Not numerically identical to the real SSM but produces correct-shape
    /// output and meaningful gradients through the residual+MLP path.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        let (batch, seq_len) = (shape[0], shape[1]);

        // Project to Q, K, V and split
        let qkv = self.in_proj_qkv.forward(x)?;
        let parts = qkv.split(3, -1)?;
        let (q, k, v) = (&parts[0], &parts[1], &parts[2]);

        // Reshape to heads: [B, L, H, D] then [B, H, L, D]
        let q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Causal attention (simplified stand-in for SSM selective scan)
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mask = create_causal_mask(seq_len as i32)?;
        use mlx_rs::fast::ScaledDotProductAttentionMask;
        let attn = mlx_rs::fast::scaled_dot_product_attention(
            &q, &k, &v, scale,
            ScaledDotProductAttentionMask::Array(&mask.as_dtype(q.dtype())?),
        )?;

        // Back to [B, L, H*D]
        let y = attn.transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, self.n_heads * self.head_dim])?;

        // Per-head group norm
        let y_flat = y.reshape(&[batch * seq_len * self.n_heads, self.head_dim])?;
        let y_normed = self.norm.forward(&y_flat)?;
        let y = y_normed.reshape(&[batch, seq_len, self.n_heads * self.head_dim])?;

        // Output gate
        let z = nn::silu(&self.in_proj_z.forward(x)?)?;
        let y = y.multiply(&z)?;

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
pub fn cross_entropy_loss(logits: &Array, targets: &Array) -> Result<Array, Exception> {
    let shape = logits.shape();
    let vocab = shape[shape.len() - 1];

    let flat_logits = logits.reshape(&[-1, vocab])?;
    let flat_targets = targets.reshape(&[-1])?;

    let log_probs = nn::log_softmax(&flat_logits, -1)?;

    // Gather log-probs at target indices
    let target_indices = flat_targets.reshape(&[-1, 1])?.as_dtype(mlx_rs::Dtype::Int32)?;
    let selected = log_probs.take_along_axis_device(&target_indices, -1, mlx_rs::StreamOrDevice::default())?;

    // Mean negative log-likelihood
    selected.mean(None)?.negative()
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

    let owned_grads: HashMap<Rc<str>, Array> =
        clipped.into_iter().map(|(k, v)| (k, v.into_owned())).collect();

    optimizer
        .update(model, &owned_grads)
        .map_err(|e| anyhow::anyhow!("optimizer update: {e}"))?;

    mlx_rs::transforms::eval(std::iter::once(&loss))
        .map_err(|e| anyhow::anyhow!("eval: {e}"))?;

    let loss_val: f32 = loss.as_slice::<f32>()[0];
    Ok((loss_val, total_norm))
}

pub fn train_loop(
    model: &mut MlxLoraModel,
    tokens_list: &[Array],
    targets_list: &[Array],
    config: &LoraConfig,
    max_steps: usize,
    early_stop_loss: f32,
    patience: usize,
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
}
