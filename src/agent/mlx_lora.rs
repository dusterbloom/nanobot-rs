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
use mlx_rs::losses::{CrossEntropyBuilder, LossReduction};
use mlx_rs::module::{Module, ModuleParamMut, ModuleParamRef, ModuleParameters, Param};
use mlx_rs::nested::{NestedHashMap, NestedValue};
use mlx_rs::nn::{self, Embedding, Linear, QuantizedEmbedding, QuantizedLinear, RmsNorm};
use mlx_rs::ops::indexing::{put_along_axis, IndexOp, TryIndexMutOp};
use mlx_rs::optimizers::{AdamW, Optimizer};
use mlx_rs::transforms::compile::{clear_cache, compile_with_state};
use mlx_rs::transforms::{async_eval, eval};
use mlx_rs::utils::Updatable;
use mlx_rs::{array, Array, Dtype};

const KV_CACHE_MATERIALIZE_INTERVAL: usize = 32;

fn compiled_decode_enabled() -> bool {
    std::env::var("NANOBOT_MLX_COMPILED_DECODE")
        .ok()
        .map(|raw| {
            let raw = raw.trim();
            raw == "1"
                || raw.eq_ignore_ascii_case("true")
                || raw.eq_ignore_ascii_case("yes")
                || raw.eq_ignore_ascii_case("on")
        })
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Tokenizer wrapper (reads tokenizer.json from model directory)
// ---------------------------------------------------------------------------

pub struct MlxTokenizer {
    inner: tokenizers::Tokenizer,
}

impl MlxTokenizer {
    pub fn load(model_dir: &Path) -> Result<Self, anyhow::Error> {
        let path = model_dir.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&path).map_err(|e| {
            anyhow::anyhow!("Failed to load tokenizer from {}: {e}", path.display())
        })?;
        Ok(MlxTokenizer { inner })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i32>, anyhow::Error> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenizer encode failed: {e}"))?;
        Ok(encoding.get_ids().iter().map(|&id| id as i32).collect())
    }

    pub fn decode(&self, ids: &[i32]) -> Result<String, anyhow::Error> {
        let u32_ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        let text = self
            .inner
            .decode(&u32_ids, true)
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

#[derive(Clone)]
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
    adapter_active: bool,
    training: bool,
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
            adapter_active: false,
            training: false,
        })
    }

    fn uses_adapter_path(&self) -> bool {
        self.training || self.adapter_active
    }

    pub fn set_adapter_active(&mut self, active: bool) {
        self.adapter_active = active;
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
        Some(self.lora_a.all_frozen().unwrap_or(true) && self.lora_b.all_frozen().unwrap_or(true))
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
        if !self.uses_adapter_path() {
            return Ok(base_out);
        }
        let h = self.lora_a.forward(x)?;
        let delta = self.lora_b.forward(&h)?;
        let scaled = delta.multiply(&array!(self.scale))?;
        base_out.add(&scaled)
    }

    fn training_mode(&mut self, mode: bool) {
        self.training = mode;
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
    // Try exact key first, then try adding/removing "language_model." prefix.
    if let Some(v) = weights.remove(key) {
        return Ok(v);
    }
    // If key starts with "language_model.", try without prefix
    if let Some(stripped) = key.strip_prefix("language_model.") {
        if let Some(v) = weights.remove(stripped) {
            return Ok(v);
        }
    }
    // If key starts with "model.", try with "language_model." prefix
    if key.starts_with("model.") {
        let prefixed = format!("language_model.{key}");
        if let Some(v) = weights.remove(&prefixed) {
            return Ok(v);
        }
    }
    Err(anyhow::anyhow!("Weight not found: {}", key))
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

/// Load a fused `gate_up_proj` QuantizedLinear and split into (gate, up).
///
/// Some models (distilled) store `gate_proj` and `up_proj` as a single
/// `gate_up_proj` with shape `[2*hidden_dim, dim]`. This splits the packed
/// weight, scales, and biases at the row midpoint.
fn split_fused_gate_up(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<(QuantizedLinear, QuantizedLinear), anyhow::Error> {
    let fused =
        load_quantized_linear(weights, &format!("{prefix}.gate_up_proj"), group_size, bits)?;

    // Split weight [2*H, packed_cols], scales [2*H, n_groups], biases [2*H, n_groups] at row midpoint
    let w_parts = fused.inner.weight.split(2, 0)?;
    let s_parts = fused.scales.split(2, 0)?;
    let b_parts = fused.biases.split(2, 0)?;

    let make_ql = |w: &Array, s: &Array, b: &Array| -> QuantizedLinear {
        let mut ql = QuantizedLinear {
            group_size,
            bits,
            scales: Param::new(s.clone()),
            biases: Param::new(b.clone()),
            inner: Linear {
                weight: Param::new(w.clone()),
                bias: Param::new(None),
            },
        };
        ql.freeze_parameters(true);
        ql
    };

    let gate = make_ql(&w_parts[0], &s_parts[0], &b_parts[0]);
    let up = make_ql(&w_parts[1], &s_parts[1], &b_parts[1]);
    Ok((gate, up))
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
        match self {
            MlxEmbedding::Plain(e) => e.num_parameters(),
            MlxEmbedding::Quantized(e) => e.num_parameters(),
        }
    }
    fn parameters(&self) -> ModuleParamRef<'_> {
        match self {
            MlxEmbedding::Plain(e) => e.parameters(),
            MlxEmbedding::Quantized(e) => e.parameters(),
        }
    }
    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        match self {
            MlxEmbedding::Plain(e) => e.parameters_mut(),
            MlxEmbedding::Quantized(e) => e.parameters_mut(),
        }
    }
    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        match self {
            MlxEmbedding::Plain(e) => e.trainable_parameters(),
            MlxEmbedding::Quantized(e) => e.trainable_parameters(),
        }
    }
    fn freeze_parameters(&mut self, r: bool) {
        match self {
            MlxEmbedding::Plain(e) => e.freeze_parameters(r),
            MlxEmbedding::Quantized(e) => e.freeze_parameters(r),
        }
    }
    fn unfreeze_parameters(&mut self, r: bool) {
        match self {
            MlxEmbedding::Plain(e) => e.unfreeze_parameters(r),
            MlxEmbedding::Quantized(e) => e.unfreeze_parameters(r),
        }
    }
    fn all_frozen(&self) -> Option<bool> {
        match self {
            MlxEmbedding::Plain(e) => e.all_frozen(),
            MlxEmbedding::Quantized(e) => e.all_frozen(),
        }
    }
    fn any_frozen(&self) -> Option<bool> {
        match self {
            MlxEmbedding::Plain(e) => e.any_frozen(),
            MlxEmbedding::Quantized(e) => e.any_frozen(),
        }
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
        let inner = Embedding {
            weight: Param::new(weight),
        };
        let mut qe = QuantizedEmbedding {
            group_size,
            bits,
            scales: Param::new(scales),
            biases: Param::new(biases),
            inner,
        };
        qe.freeze_parameters(true);
        Ok(MlxEmbedding::Quantized(qe))
    } else {
        let mut emb = Embedding {
            weight: Param::new(weight),
        };
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
    /// Linear attention key head config (only used when linear_attn_indices is non-empty)
    pub linear_n_heads: usize,
    pub linear_head_dim: usize,
    /// Linear attention value head config.  Defaults to key config when equal
    /// (Qwen3.5-2B), but can differ (Qwen3.5-9B: 32 value heads vs 16 key heads).
    pub linear_n_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub conv_kernel_size: usize,
    /// Thinking model: use /nothink system instruction instead of empty think prefill
    pub thinking_model: bool,
    /// Mixture-of-Experts model: in-process training not supported (use vllm-mlx for inference)
    pub is_moe: bool,
}

impl ModelConfig {
    pub fn qwen3_0_6b() -> Self {
        ModelConfig {
            dim: 1024,
            hidden_dim: 3072,
            n_heads: 16,
            n_kv_heads: 8,
            n_layers: 28,
            vocab_size: 151936,
            head_dim: 128,
            rope_theta: 1_000_000.0,
            rms_eps: 1e-6,
            group_size: 64,
            bits: 8,
            weight_prefix: "model",
            partial_rotary_factor: 1.0,
            attn_output_gate: false,
            linear_attn_indices: vec![],
            linear_n_heads: 0,
            linear_head_dim: 0,
            linear_n_value_heads: 0,
            linear_value_head_dim: 0,
            conv_kernel_size: 0,
            thinking_model: true,
            is_moe: false,
        }
    }

    pub fn qwen3_1_7b() -> Self {
        ModelConfig {
            dim: 2048,
            hidden_dim: 6144,
            n_heads: 16,
            n_kv_heads: 8,
            n_layers: 28,
            vocab_size: 151936,
            head_dim: 128,
            rope_theta: 1_000_000.0,
            rms_eps: 1e-6,
            group_size: 64,
            bits: 8,
            weight_prefix: "model",
            partial_rotary_factor: 1.0,
            attn_output_gate: false,
            linear_attn_indices: vec![],
            linear_n_heads: 0,
            linear_head_dim: 0,
            linear_n_value_heads: 0,
            linear_value_head_dim: 0,
            conv_kernel_size: 0,
            thinking_model: false,
            is_moe: false,
        }
    }

    pub fn qwen3_4b() -> Self {
        ModelConfig {
            dim: 2560,
            hidden_dim: 9728,
            n_heads: 32,
            n_kv_heads: 8,
            n_layers: 36,
            vocab_size: 151936,
            head_dim: 128,
            rope_theta: 5_000_000.0,
            rms_eps: 1e-6,
            group_size: 64,
            bits: 4,
            weight_prefix: "model",
            partial_rotary_factor: 1.0,
            attn_output_gate: false,
            linear_attn_indices: vec![],
            linear_n_heads: 0,
            linear_head_dim: 0,
            linear_n_value_heads: 0,
            linear_value_head_dim: 0,
            conv_kernel_size: 0,
            thinking_model: true,
            is_moe: false,
        }
    }

    pub fn qwen3_8b() -> Self {
        ModelConfig {
            dim: 4096,
            hidden_dim: 12288,
            n_heads: 32,
            n_kv_heads: 8,
            n_layers: 36,
            vocab_size: 151936,
            head_dim: 128,
            rope_theta: 1_000_000.0,
            rms_eps: 1e-6,
            group_size: 64,
            bits: 4,
            weight_prefix: "model",
            partial_rotary_factor: 1.0,
            attn_output_gate: false,
            linear_attn_indices: vec![],
            linear_n_heads: 0,
            linear_head_dim: 0,
            linear_n_value_heads: 0,
            linear_value_head_dim: 0,
            conv_kernel_size: 0,
            thinking_model: true,
            is_moe: false,
        }
    }

    pub fn qwen3_5_2b() -> Self {
        // Hybrid Mamba-Transformer: 18 linear_attn + 6 full_attn (every 4th)
        let linear_indices: Vec<usize> = (0..24).filter(|i| i % 4 != 3).collect();
        ModelConfig {
            dim: 2048,
            hidden_dim: 6144,
            n_heads: 8,
            n_kv_heads: 2,
            n_layers: 24,
            vocab_size: 248320,
            head_dim: 256,
            rope_theta: 10_000_000.0,
            rms_eps: 1e-6,
            group_size: 64,
            bits: 8,
            weight_prefix: "language_model.model",
            partial_rotary_factor: 0.25,
            attn_output_gate: true,
            linear_attn_indices: linear_indices,
            linear_n_heads: 16,
            linear_head_dim: 128,
            linear_n_value_heads: 16,
            linear_value_head_dim: 128,
            conv_kernel_size: 4,
            thinking_model: false,
            is_moe: false,
        }
    }

    pub fn qwen3_5_4b() -> Self {
        // Hybrid GDN: 24 linear_attn + 8 full_attn (every 4th), 32 layers
        let linear_indices: Vec<usize> = (0..32).filter(|i| i % 4 != 3).collect();
        ModelConfig {
            dim: 2560,
            hidden_dim: 9216,
            n_heads: 16,
            n_kv_heads: 4,
            n_layers: 32,
            vocab_size: 248320,
            head_dim: 256,
            rope_theta: 10_000_000.0,
            rms_eps: 1e-6,
            group_size: 64,
            bits: 4,
            weight_prefix: "language_model.model",
            partial_rotary_factor: 0.25,
            attn_output_gate: true,
            linear_attn_indices: linear_indices,
            linear_n_heads: 16,
            linear_head_dim: 128,
            linear_n_value_heads: 32,
            linear_value_head_dim: 128,
            conv_kernel_size: 4,
            thinking_model: true,
            is_moe: false,
        }
    }

    pub fn qwen3_5_9b() -> Self {
        // Hybrid GDN: 24 linear_attn + 8 full_attn (every 4th), 32 layers
        let linear_indices: Vec<usize> = (0..32).filter(|i| i % 4 != 3).collect();
        ModelConfig {
            dim: 4096,
            hidden_dim: 12288,
            n_heads: 16,
            n_kv_heads: 4,
            n_layers: 32,
            vocab_size: 248320,
            head_dim: 256,
            rope_theta: 10_000_000.0,
            rms_eps: 1e-6,
            group_size: 64,
            bits: 4,
            weight_prefix: "language_model.model",
            partial_rotary_factor: 0.25,
            attn_output_gate: true,
            linear_attn_indices: linear_indices,
            linear_n_heads: 16,
            linear_head_dim: 128,
            linear_n_value_heads: 32,
            linear_value_head_dim: 128,
            conv_kernel_size: 4,
            thinking_model: true,
            is_moe: false,
        }
    }

    /// Build `ModelConfig` by reading the model's `config.json` from disk.
    ///
    /// Supports both Qwen3 (pure transformer) and Qwen3.5 (hybrid GDN) models.
    /// Falls back gracefully: if `config.json` is missing or unparseable,
    /// returns `None` so callers can fall back to a preset.
    pub fn from_config_json(model_dir: &Path) -> Option<Self> {
        let config_path = model_dir.join("config.json");
        let data = std::fs::read_to_string(&config_path).ok()?;
        let root: serde_json::Value = serde_json::from_str(&data).ok()?;

        // Qwen3.5 nests architecture under "text_config"; Qwen3 puts it at root.
        let tc = root.get("text_config").unwrap_or(&root);

        let model_type = root
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let is_qwen35 = model_type.starts_with("qwen3_5")
            || model_type.starts_with("qwen3.5")
            || root
                .get("architectures")
                .and_then(|a| a.as_array())
                .map(|a| {
                    a.iter()
                        .any(|v| v.as_str().map(|s| s.contains("Qwen3_5")).unwrap_or(false))
                })
                .unwrap_or(false);

        let dim = tc.get("hidden_size")?.as_u64()? as usize;
        // Dense models use intermediate_size; MoE models use moe_intermediate_size.
        let hidden_dim = tc
            .get("intermediate_size")
            .or_else(|| tc.get("moe_intermediate_size"))
            .and_then(|v| v.as_u64())? as usize;
        let n_heads = tc.get("num_attention_heads")?.as_u64()? as usize;
        let n_kv_heads = tc
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(n_heads as u64) as usize;
        let n_layers = tc.get("num_hidden_layers")?.as_u64()? as usize;
        let vocab_size = tc.get("vocab_size")?.as_u64()? as usize;
        let head_dim = tc
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or((dim / n_heads) as u64) as usize;

        // RoPE theta: try text_config.rope_parameters.rope_theta first,
        // then text_config.rope_theta, then root.rope_theta
        let rope_theta = tc
            .get("rope_parameters")
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .or_else(|| tc.get("rope_theta").and_then(|v| v.as_f64()))
            .or_else(|| root.get("rope_theta").and_then(|v| v.as_f64()))
            .unwrap_or(1_000_000.0) as f32;

        let rms_eps = tc
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6) as f32;

        // Quantization: check root.quantization or root.quantization_config
        let quant = root
            .get("quantization")
            .or_else(|| root.get("quantization_config"));
        let group_size = quant
            .and_then(|q| q.get("group_size"))
            .and_then(|v| v.as_i64())
            .unwrap_or(64) as i32;
        let bits = quant
            .and_then(|q| q.get("bits"))
            .and_then(|v| v.as_i64())
            .unwrap_or(4) as i32;

        // Qwen3.5 hybrid architecture
        let partial_rotary_factor = if is_qwen35 {
            tc.get("rope_parameters")
                .and_then(|rp| rp.get("partial_rotary_factor"))
                .and_then(|v| v.as_f64())
                .or_else(|| tc.get("partial_rotary_factor").and_then(|v| v.as_f64()))
                .unwrap_or(0.25) as f32
        } else {
            1.0
        };

        let attn_output_gate = tc
            .get("attn_output_gate")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Linear attention layers (Qwen3.5 hybrid)
        let layer_types: Vec<String> = tc
            .get("layer_types")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let linear_attn_indices: Vec<usize> = layer_types
            .iter()
            .enumerate()
            .filter(|(_, t)| t.as_str() == "linear_attention")
            .map(|(i, _)| i)
            .collect();

        let linear_n_heads = tc
            .get("linear_num_key_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let linear_head_dim = tc
            .get("linear_key_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        // Value heads can differ from key heads (e.g. Qwen3.5-9B: 32 vs 16).
        let linear_n_value_heads = tc
            .get("linear_num_value_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(linear_n_heads as u64) as usize;
        let linear_value_head_dim = tc
            .get("linear_value_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(linear_head_dim as u64) as usize;
        let conv_kernel_size = tc
            .get("linear_conv_kernel_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let weight_prefix = if is_qwen35 {
            "language_model.model"
        } else {
            "model"
        };

        // Detect thinking model: Qwen3/3.5 templates default to enable_thinking=true,
        // so ALL Qwen3.x models are thinking-capable regardless of name.
        let dir_name = model_dir
            .file_name()
            .map(|n| n.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        let thinking_model = model_type.starts_with("qwen3")
            || dir_name.contains("qwen3")
            || dir_name.contains("thinking")
            || dir_name.contains("reasoning")
            || dir_name.contains("distill");

        let is_moe = tc.get("num_experts").and_then(|v| v.as_u64()).unwrap_or(0) > 1;

        tracing::info!(
            model_type,
            dim,
            n_layers,
            n_heads,
            bits,
            is_qwen35,
            linear_layers = linear_attn_indices.len(),
            "auto-detected model config from config.json"
        );

        Some(ModelConfig {
            dim,
            hidden_dim,
            n_heads,
            n_kv_heads,
            n_layers,
            vocab_size,
            head_dim,
            rope_theta,
            rms_eps,
            group_size,
            bits,
            weight_prefix,
            partial_rotary_factor,
            attn_output_gate,
            linear_attn_indices,
            linear_n_heads,
            linear_head_dim,
            linear_n_value_heads,
            linear_value_head_dim,
            conv_kernel_size,
            thinking_model,
            is_moe,
        })
    }

    /// Convert to the ANE-compatible MilConfig used by the training pipeline.
    #[cfg(feature = "ane")]
    pub fn to_mil_config(&self, seq_len: usize) -> super::ane_mil::MilConfig {
        super::ane_mil::MilConfig {
            dim: self.dim,
            hidden_dim: self.hidden_dim,
            n_heads: self.n_heads,
            seq_len,
            n_kv_heads: self.n_kv_heads,
            rope_theta: self.rope_theta as f64,
            rms_eps: self.rms_eps,
            has_lm_head: false,
            head_dim_explicit: self.head_dim,
            linear_attn_indices: self.linear_attn_indices.clone(),
            linear_n_heads: self.linear_n_heads,
            linear_head_dim: self.linear_head_dim,
            linear_n_value_heads: self.linear_n_value_heads,
            linear_value_head_dim: self.linear_value_head_dim,
            conv_kernel_size: self.conv_kernel_size,
            attn_output_gate: self.attn_output_gate,
        }
    }

    pub fn rope_dims(&self) -> i32 {
        (self.head_dim as f32 * self.partial_rotary_factor) as i32
    }

    pub fn is_linear_attn_layer(&self, idx: usize) -> bool {
        self.linear_attn_indices.contains(&idx)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct KvCacheConfig {
    pub bits: Option<i32>,
    pub group_size: i32,
    pub quantized_start: i32,
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        let bits = std::env::var("NANOBOT_MLX_KV_BITS")
            .ok()
            .and_then(|raw| raw.parse::<i32>().ok())
            .filter(|bits| *bits > 0);
        let group_size = std::env::var("NANOBOT_MLX_KV_GROUP_SIZE")
            .ok()
            .and_then(|raw| raw.parse::<i32>().ok())
            .filter(|group_size| *group_size > 0)
            .unwrap_or(64);
        let quantized_start = std::env::var("NANOBOT_MLX_KV_START")
            .ok()
            .and_then(|raw| raw.parse::<i32>().ok())
            .filter(|start| *start >= 0)
            .unwrap_or(0);

        KvCacheConfig {
            bits,
            group_size,
            quantized_start,
        }
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

fn dense_scaled_dot_product_attention(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    mask: Option<&Array>,
    offset: i32,
    seq_len: i32,
) -> Result<Array, Exception> {
    use mlx_rs::fast::ScaledDotProductAttentionMask;

    if let Some(mask) = mask {
        let mask = mask.as_dtype(queries.dtype())?;
        mlx_rs::fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale,
            ScaledDotProductAttentionMask::Array(&mask),
        )
    } else if offset == 0 {
        mlx_rs::fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale,
            ScaledDotProductAttentionMask::Causal,
        )
    } else if seq_len == 1 {
        mlx_rs::fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale,
            None::<ScaledDotProductAttentionMask<'_>>,
        )
    } else {
        debug_assert!(
            false,
            "multi-token cached decode requires an explicit rectangular mask"
        );
        mlx_rs::fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale,
            ScaledDotProductAttentionMask::Causal,
        )
    }
}

fn quantized_scaled_dot_product_attention(
    queries: &Array,
    keys: &QuantizedArray,
    values: &QuantizedArray,
    scale: f32,
    mask: Option<&Array>,
    group_size: i32,
    bits: i32,
) -> Result<Array, Exception> {
    let shape = queries.shape();
    let (batch, n_q_heads, seq_len, head_dim) = (shape[0], shape[1], shape[2], shape[3]);
    let n_kv_heads = keys.packed.shape()[1];
    let n_repeats = n_q_heads / n_kv_heads;

    let queries = queries.multiply(array!(scale))?;
    let (queries, keys, values) = if n_repeats > 1 {
        (
            queries.reshape(&[batch, n_kv_heads, n_repeats, seq_len, head_dim])?,
            keys.expand_dims(-3)?,
            values.expand_dims(-3)?,
        )
    } else {
        (queries, keys.clone(), values.clone())
    };

    let mut scores = mlx_rs::ops::quantized_matmul(
        &queries,
        &keys.packed,
        &keys.scales,
        &keys.biases,
        true,
        group_size,
        bits,
    )?;

    if let Some(mask) = mask {
        if mask.dtype() == Dtype::Bool {
            let fill = array!(f32::NEG_INFINITY).as_dtype(scores.dtype())?;
            scores = mlx_rs::ops::r#where(mask, &scores, &fill)?;
        } else {
            let mask = mask.as_dtype(scores.dtype())?;
            scores = scores.add(&mask)?;
        }
    }

    let scores = mlx_rs::ops::softmax_axis(&scores, -1, true)?;
    let mut out = mlx_rs::ops::quantized_matmul(
        &scores,
        &values.packed,
        &values.scales,
        &values.biases,
        false,
        group_size,
        bits,
    )?;

    if n_repeats > 1 {
        out = out.reshape(&[batch, n_q_heads, seq_len, head_dim])?;
    }

    Ok(out)
}

type CompiledPlainKvDecodeFn = dyn for<'a> FnMut(
    &mut PlainKvCache,
    (&'a Array, &'a Array, &'a Array),
) -> Result<Array, Exception>;

fn make_compiled_plain_kv_decode(attn_scale: f32) -> Box<CompiledPlainKvDecodeFn> {
    Box::new(compile_with_state(
        move |cache: &mut PlainKvCache, (q, k, v): (&Array, &Array, &Array)| {
            compiled_plain_kv_decode_step(cache, q, k, v, attn_scale)
        },
        true,
    ))
}

fn compiled_plain_kv_decode_step(
    cache: &mut PlainKvCache,
    queries: &Array,
    new_keys: &Array,
    new_values: &Array,
    attn_scale: f32,
) -> Result<Array, Exception> {
    let storage_keys = cache
        .keys
        .as_ref()
        .ok_or_else(|| Exception::custom("compiled decode requires initialized KV keys"))?;
    let storage_values = cache
        .values
        .as_ref()
        .ok_or_else(|| Exception::custom("compiled decode requires initialized KV values"))?;
    let len = cache
        .len_array
        .as_ref()
        .ok_or_else(|| Exception::custom("compiled decode requires initialized KV length state"))?;
    let positions = cache.positions.as_ref().ok_or_else(|| {
        Exception::custom("compiled decode requires initialized KV position state")
    })?;

    let index = len.reshape(&[1, 1, 1, 1])?.as_dtype(Dtype::Int32)?;
    let index = mlx_rs::ops::broadcast_to(&index, new_keys.shape())?;
    let keys = put_along_axis(storage_keys, &index, new_keys, 2)?;
    let values = put_along_axis(storage_values, &index, new_values, 2)?;

    let next_len = len.add(&array!(1).as_dtype(Dtype::Int32)?)?;
    let next_len = next_len.as_dtype(Dtype::Int32)?;
    let active = positions.lt(&next_len.reshape(&[1, 1, 1, 1])?)?;
    let zeros = array!(0.0).as_dtype(queries.dtype())?;
    let neg_inf = array!(f32::NEG_INFINITY).as_dtype(queries.dtype())?;
    let mask = mlx_rs::ops::r#where(&active, &zeros, &neg_inf)?;

    // compile_with_state only persists arrays that are written back into the
    // tracked state object. Keep KV storage and length in the cache so the
    // next compiled step sees the updated prefix instead of the stale inputs.
    *cache
        .keys
        .as_mut()
        .ok_or_else(|| Exception::custom("compiled decode requires initialized KV keys"))? =
        keys.clone();
    *cache
        .values
        .as_mut()
        .ok_or_else(|| Exception::custom("compiled decode requires initialized KV values"))? =
        values.clone();
    *cache.len_array.as_mut().ok_or_else(|| {
        Exception::custom("compiled decode requires initialized KV length state")
    })? = next_len.clone();

    dense_scaled_dot_product_attention(
        queries,
        &keys,
        &values,
        attn_scale,
        Some(&mask),
        0,
        queries.shape()[2],
    )
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
                rank,
                scale,
            )
            .map_err(|e| anyhow::anyhow!("LoRA q_proj: {e}"))?,
            k_proj: load_quantized_linear(weights, &format!("{prefix}.k_proj"), gs, bits)?,
            v_proj: LoraLinear::new(
                load_quantized_linear(weights, &format!("{prefix}.v_proj"), gs, bits)?,
                rank,
                scale,
            )
            .map_err(|e| anyhow::anyhow!("LoRA v_proj: {e}"))?,
            o_proj: LoraLinear::new(
                load_quantized_linear(weights, &format!("{prefix}.o_proj"), gs, bits)?,
                rank,
                scale,
            )
            .map_err(|e| anyhow::anyhow!("LoRA o_proj: {e}"))?,
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
        let k_raw = self.k_proj.forward(x)?;
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
            let q4d = q_raw.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
            let q_norm_size = self.q_norm.weight.shape()[0] as i32;
            let q = if q_norm_size == self.head_dim {
                self.q_norm.forward(&q4d)?
            } else {
                self.q_norm.forward(&q_raw)?.reshape(&[
                    batch,
                    seq_len,
                    self.num_heads,
                    self.head_dim,
                ])?
            };
            (q, None)
        };

        // QK norm — when gated, q is already [B, S, H, D] from the per-head split
        let q = if self.attn_output_gate {
            // q is [B, S, H, D] — RMSNorm applies on last dim (head_dim), which matches weight shape
            self.q_norm.forward(&q)?
        } else {
            q
        };
        let k4d = k_raw.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?;
        let k_norm_size = self.k_norm.weight.shape()[0] as i32;
        let k = if k_norm_size == self.head_dim {
            self.k_norm.forward(&k4d)?
        } else {
            self.k_norm.forward(&k_raw)?.reshape(&[
                batch,
                seq_len,
                self.num_kv_heads,
                self.head_dim,
            ])?
        };

        // Reshape to [B, S, H, D] — q may already be 4D if gated
        let q = if self.attn_output_gate {
            q // already [B, S, H, D]
        } else {
            q
        };
        let k = k;
        let v = v.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?;

        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        let q = mlx_rs::fast::rope(
            &q,
            self.rope_dims,
            false,
            self.rope_base,
            1.0,
            0,
            None::<&Array>,
        )?;
        let k = mlx_rs::fast::rope(
            &k,
            self.rope_dims,
            false,
            self.rope_base,
            1.0,
            0,
            None::<&Array>,
        )?;

        let attn =
            dense_scaled_dot_product_attention(&q, &k, &v, self.attn_scale, mask, 0, seq_len)?;

        let mut attn = attn.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ])?;

        if let Some(gate) = output_gate {
            attn = attn.multiply(&nn::sigmoid(&gate)?)?;
        }

        self.o_proj.forward(&attn)
    }

    /// Forward with KV cache for incremental generation.
    ///
    /// `cache`: per-layer KV cache, updated in-place.
    /// `offset`: position offset for RoPE (= number of previously cached tokens).
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut KvCache,
        compiled_plain_kv_decode: Option<&mut CompiledPlainKvDecodeFn>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let (batch, seq_len) = (shape[0], shape[1]);
        let offset = cache.len();

        let q_raw = self.q_proj.forward(x)?;
        let k_raw = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let (q, output_gate) = if self.attn_output_gate {
            let q4d = q_raw.reshape(&[batch, seq_len, self.num_heads, self.head_dim * 2])?;
            let parts = q4d.split(2, -1)?;
            let gate = parts[1].reshape(&[batch, seq_len, self.num_heads * self.head_dim])?;
            (parts[0].clone(), Some(gate))
        } else {
            let q4d = q_raw.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
            let q_norm_size = self.q_norm.weight.shape()[0] as i32;
            let q = if q_norm_size == self.head_dim {
                self.q_norm.forward(&q4d)?
            } else {
                self.q_norm.forward(&q_raw)?.reshape(&[
                    batch,
                    seq_len,
                    self.num_heads,
                    self.head_dim,
                ])?
            };
            (q, None)
        };

        // QK norm
        let q = if self.attn_output_gate {
            self.q_norm.forward(&q)?
        } else {
            q
        };
        let k4d = k_raw.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?;
        let k_norm_size = self.k_norm.weight.shape()[0] as i32;
        let k = if k_norm_size == self.head_dim {
            self.k_norm.forward(&k4d)?
        } else {
            self.k_norm.forward(&k_raw)?.reshape(&[
                batch,
                seq_len,
                self.num_kv_heads,
                self.head_dim,
            ])?
        };

        let q = if self.attn_output_gate { q } else { q };
        let k = k;
        let v = v.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?;

        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // RoPE with position offset for cached positions
        let q = mlx_rs::fast::rope(
            &q,
            self.rope_dims,
            false,
            self.rope_base,
            1.0,
            offset,
            None::<&Array>,
        )?;
        let k = mlx_rs::fast::rope(
            &k,
            self.rope_dims,
            false,
            self.rope_base,
            1.0,
            offset,
            None::<&Array>,
        )?;

        let attn = if seq_len == 1 && mask.is_none() {
            if let Some(compiled) = compiled_plain_kv_decode {
                if let KvCache::Plain(plain) = cache {
                    if plain.compiled_decode_ready() {
                        let attn = compiled(plain, (&q, &k, &v))?;
                        plain.finish_compiled_decode_step(1)?;
                        attn
                    } else {
                        let (keys, values) = plain.update(&k, &v)?;
                        dense_scaled_dot_product_attention(
                            &q,
                            &keys,
                            &values,
                            self.attn_scale,
                            None,
                            offset,
                            seq_len,
                        )?
                    }
                } else {
                    let kv_view = cache.update(&k, &v)?;
                    match kv_view {
                        KvCacheView::Plain { keys, values } => dense_scaled_dot_product_attention(
                            &q,
                            &keys,
                            &values,
                            self.attn_scale,
                            None,
                            offset,
                            seq_len,
                        )?,
                        KvCacheView::Quantized(view) => quantized_scaled_dot_product_attention(
                            &q,
                            &view.keys,
                            &view.values,
                            self.attn_scale,
                            None,
                            view.group_size,
                            view.bits,
                        )?,
                    }
                }
            } else {
                let kv_view = cache.update(&k, &v)?;
                match kv_view {
                    KvCacheView::Plain { keys, values } => dense_scaled_dot_product_attention(
                        &q,
                        &keys,
                        &values,
                        self.attn_scale,
                        None,
                        offset,
                        seq_len,
                    )?,
                    KvCacheView::Quantized(view) => quantized_scaled_dot_product_attention(
                        &q,
                        &view.keys,
                        &view.values,
                        self.attn_scale,
                        None,
                        view.group_size,
                        view.bits,
                    )?,
                }
            }
        } else {
            let kv_view = cache.update(&k, &v)?;
            match kv_view {
                KvCacheView::Plain { keys, values } => dense_scaled_dot_product_attention(
                    &q,
                    &keys,
                    &values,
                    self.attn_scale,
                    mask,
                    offset,
                    seq_len,
                )?,
                KvCacheView::Quantized(view) => {
                    if offset == 0 {
                        dense_scaled_dot_product_attention(
                            &q,
                            &k,
                            &v,
                            self.attn_scale,
                            mask,
                            offset,
                            seq_len,
                        )?
                    } else {
                        let resolved_mask = if let Some(mask) = mask {
                            Some(mask.clone())
                        } else if seq_len > 1 {
                            Some(create_causal_mask_cached(
                                seq_len,
                                view.keys.packed.shape()[2],
                            )?)
                        } else {
                            None
                        };
                        quantized_scaled_dot_product_attention(
                            &q,
                            &view.keys,
                            &view.values,
                            self.attn_scale,
                            resolved_mask.as_ref(),
                            view.group_size,
                            view.bits,
                        )?
                    }
                }
            }
        };

        let mut attn = attn.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ])?;

        if let Some(gate) = output_gate {
            attn = attn.multiply(&nn::sigmoid(&gate)?)?;
        }

        self.o_proj.forward(&attn)
    }

    fn set_lora_training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
    }
}

impl ModuleParameters for MlxLoraAttention {
    fn num_parameters(&self) -> usize {
        self.q_proj.num_parameters()
            + self.k_proj.num_parameters()
            + self.v_proj.num_parameters()
            + self.o_proj.num_parameters()
            + self.q_norm.num_parameters()
            + self.k_norm.num_parameters()
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

        // Fallback chain: dense → shared_expert (MoE) → fused gate_up_proj
        let mut try_load_mlp =
            |pfx: &str| -> Option<(QuantizedLinear, QuantizedLinear, QuantizedLinear)> {
                let g =
                    load_quantized_linear(weights, &format!("{pfx}.gate_proj"), gs, bits).ok()?;
                let u = load_quantized_linear(weights, &format!("{pfx}.up_proj"), gs, bits).ok()?;
                let d =
                    load_quantized_linear(weights, &format!("{pfx}.down_proj"), gs, bits).ok()?;
                Some((g, u, d))
            };
        let (gate_proj, up_proj, down_proj_base) = if let Some(mlp) = try_load_mlp(prefix) {
            mlp
        } else {
            // prefix is like "model.layers.0.mlp" — try "model.layers.0.mlp.shared_expert"
            let se_prefix = format!("{prefix}.shared_expert");
            if let Some(mlp) = try_load_mlp(&se_prefix) {
                mlp
            } else {
                let (g, u) = split_fused_gate_up(weights, prefix, gs, bits)?;
                let d = load_quantized_linear(weights, &format!("{prefix}.down_proj"), gs, bits)?;
                (g, u, d)
            }
        };

        Ok(MlxLoraMLP {
            gate_proj,
            up_proj,
            down_proj: LoraLinear::new(down_proj_base, lora_cfg.rank, lora_cfg.scale())
                .map_err(|e| anyhow::anyhow!("LoRA down_proj: {e}"))?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        self.forward_profiled(x, None)
    }

    fn forward_profiled(
        &mut self,
        x: &Array,
        mut profile: Option<&mut LinearDecodeSubProfile>,
    ) -> Result<Array, Exception> {
        let gate_proj_t0 = std::time::Instant::now();
        let gate = self.gate_proj.forward(x)?;
        if let Some(profile) = profile.as_deref_mut() {
            gate.eval()?;
            profile.mlp_gate_proj_ms = gate_proj_t0.elapsed().as_secs_f64() * 1000.0;
        }
        let up_proj_t0 = std::time::Instant::now();
        let up = self.up_proj.forward(x)?;
        if let Some(profile) = profile.as_deref_mut() {
            up.eval()?;
            profile.mlp_up_proj_ms = up_proj_t0.elapsed().as_secs_f64() * 1000.0;
        }
        let act_mul_t0 = std::time::Instant::now();
        let gate = nn::silu(&gate)?;
        let h = gate.multiply(&up)?;
        if let Some(profile) = profile.as_deref_mut() {
            h.eval()?;
            profile.mlp_act_mul_ms = act_mul_t0.elapsed().as_secs_f64() * 1000.0;
        }
        let down_proj_t0 = std::time::Instant::now();
        let out = self.down_proj.forward(&h)?;
        if let Some(profile) = profile.as_deref_mut() {
            out.eval()?;
            profile.mlp_down_proj_ms = down_proj_t0.elapsed().as_secs_f64() * 1000.0;
        }
        Ok(out)
    }

    fn set_lora_training_mode(&mut self, mode: bool) {
        self.down_proj.training_mode(mode);
    }
}

impl ModuleParameters for MlxLoraMLP {
    fn num_parameters(&self) -> usize {
        self.gate_proj.num_parameters()
            + self.up_proj.num_parameters()
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
        nested_ref_from(vec![("down_proj", self.down_proj.trainable_parameters())])
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
    /// Number of key/query heads (may differ from value heads in GQA-style GDN).
    pub n_heads: i32,
    pub head_dim: i32,
    /// Number of value heads — the recurrence runs with this many heads.
    /// K/Q are repeated n_value_heads/n_heads times to match.
    pub n_value_heads: i32,
    pub value_head_dim: i32,
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
            in_proj_qkv: load_quantized_linear(
                weights,
                &format!("{prefix}.in_proj_qkv"),
                gs,
                bits,
            )?,
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
            n_value_heads: cfg.linear_n_value_heads as i32,
            value_head_dim: cfg.linear_value_head_dim as i32,
            conv_kernel: cfg.conv_kernel_size as i32,
        };
        attn.freeze_parameters(true);
        Ok(attn)
    }

    pub fn apply_decode_conv1d_step(
        &self,
        qkv: &Array,
        cache: &mut GdnCache,
    ) -> Result<Array, Exception> {
        apply_decode_conv1d_step_with_weight(&*self.conv1d_weight, self.conv_kernel, qkv, cache)
    }

    #[doc(hidden)]
    pub fn apply_decode_conv1d_step_reference(
        &self,
        qkv: &Array,
        cache: &mut GdnCache,
    ) -> Result<Array, Exception> {
        apply_decode_conv1d_step_reference_with_weight(
            &*self.conv1d_weight,
            self.conv_kernel,
            qkv,
            cache,
        )
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
        let h_k = self.n_heads; // key/query heads
        let d_k = self.head_dim; // key/query head dim
        let h_v = self.n_value_heads; // value heads (recurrence runs at this count)
        let d_v = self.value_head_dim;
        let key_dim = h_k * d_k;
        let value_dim = h_v * d_v;
        let kv_repeat = h_v / h_k; // GQA repeat factor (1 when symmetric)

        // 1. Project QKV, alpha, beta, z
        let qkv = self.in_proj_qkv.forward(x)?; // [B, L, 2*key_dim + value_dim]
        let a = self.in_proj_a.forward(x)?; // [B, L, H_v]
        let b = self.in_proj_b.forward(x)?; // [B, L, H_v]

        // 2. Causal depthwise conv1d on QKV
        let conv_dim = qkv.shape()[2];
        let kernel = self.conv_kernel;
        let pad_widths: &[(i32, i32)] = &[(0, 0), (kernel - 1, 0), (0, 0)];
        let qkv_padded = mlx_rs::ops::pad(&qkv, pad_widths, None, None)?;
        let qkv_conv = mlx_rs::ops::conv1d(
            &qkv_padded,
            &*self.conv1d_weight,
            None,
            None,
            None,
            conv_dim as i32,
        )?;
        let qkv_conv = nn::silu(&qkv_conv)?;

        // 3. Split into Q, K, V — Q and K share key_dim, V uses value_dim
        let parts = qkv_conv.split_axis(&[key_dim, 2 * key_dim], -1)?;
        let q = parts[0].reshape(&[batch, seq_len, h_k, d_k])?;
        let k = parts[1].reshape(&[batch, seq_len, h_k, d_k])?;
        let v = parts[2].reshape(&[batch, seq_len, h_v, d_v])?;

        // 4. QK RMS normalization (weight-free)
        let inv_scale = (d_k as f32).powf(-0.5);
        let ones_d = mlx_rs::ops::ones::<f32>(&[d_k])?;
        let q_flat = q.reshape(&[-1, d_k])?;
        let k_flat = k.reshape(&[-1, d_k])?;
        let q_norm =
            mlx_rs::fast::rms_norm(&q_flat, &ones_d, 1e-6)?.reshape(&[batch, seq_len, h_k, d_k])?;
        let k_norm =
            mlx_rs::fast::rms_norm(&k_flat, &ones_d, 1e-6)?.reshape(&[batch, seq_len, h_k, d_k])?;

        let q = q_norm.multiply(Array::from_f32(inv_scale * inv_scale))?;
        let k = k_norm.multiply(Array::from_f32(inv_scale))?;
        let v = v.reshape(&[batch, seq_len, h_k, kv_repeat, d_v])?;

        // 5. Compute decay g and write gate beta
        let a_plus_bias = a.add(&*self.dt_bias)?; // [B, L, H_v]
        let sp = nn::softplus(&a_plus_bias)?;
        let decay_rate = mlx_rs::ops::exp(&*self.a_log)?.multiply(&sp)?;
        let g = mlx_rs::ops::exp(&decay_rate.negative()?)?; // [B, L, H_v]
        let beta = nn::sigmoid(&b)?; // [B, L, H_v]
        let g = g.reshape(&[batch, seq_len, h_k, kv_repeat])?;
        let beta = beta.reshape(&[batch, seq_len, h_k, kv_repeat])?;

        // 6. Gated delta recurrence — grouped state avoids materializing repeated Q/K heads.
        let mut state = mlx_rs::ops::zeros::<f32>(&[batch, h_k, kv_repeat, d_v, d_k])?;
        let mut outputs: Vec<Array> = Vec::with_capacity(seq_len as usize);

        for t in 0..seq_len {
            let g_t = g
                .index((.., t, .., ..))
                .reshape(&[batch, h_k, kv_repeat, 1, 1])?;
            let beta_t = beta
                .index((.., t, .., ..))
                .reshape(&[batch, h_k, kv_repeat, 1])?;
            let q_t = q.index((.., t, .., ..)).reshape(&[batch, h_k, 1, 1, d_k])?;
            let k_t = k.index((.., t, .., ..)).reshape(&[batch, h_k, 1, 1, d_k])?;
            let v_t = v.index((.., t, .., .., ..)); // [B, H_k, R, D_v]

            state = state.multiply(&g_t)?;
            let kv_mem = state.multiply(&k_t)?.sum_axes(&[-1], false)?; // [B, H_k, R, D_v]
            let delta = v_t.subtract(&kv_mem)?.multiply(&beta_t)?;
            state = state.add(&k_t.multiply(&delta.expand_dims(-1)?)?)?;
            let y_t = state
                .multiply(&q_t)?
                .sum_axes(&[-1], false)?
                .reshape(&[batch, h_v, d_v])?;
            outputs.push(y_t);
        }

        // Stack: [B, H_v, D_v] * L → [B, L, H_v, D_v]
        let y = mlx_rs::ops::stack_axis(&outputs, 1)?;

        // 7. RMSNormGated: norm on last dim (D_v), then silu(z) * normed
        let z = self
            .in_proj_z
            .forward(x)?
            .reshape(&[batch, seq_len, h_v, d_v])?;
        let y_flat = y.reshape(&[-1, d_v])?;
        let y_normed = self
            .norm
            .forward(&y_flat)?
            .reshape(&[batch, seq_len, h_v, d_v])?;
        let y = nn::silu(&z)?.multiply(&y_normed)?;
        let y = y.reshape(&[batch, seq_len, value_dim])?;

        self.out_proj.forward(&y)
    }

    /// Forward with GDN cache for incremental generation.
    ///
    /// On prefill (seq_len > 1): runs full recurrence and stores final state + conv buffer.
    /// On decode (seq_len == 1): runs single-step update using cached state.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        cache: &mut GdnCache,
    ) -> Result<Array, Exception> {
        self.forward_with_cache_internal(x, cache, None)
    }

    fn forward_with_cache_internal(
        &mut self,
        x: &Array,
        cache: &mut GdnCache,
        mut profile: Option<&mut LinearDecodeSubProfile>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let (batch, seq_len) = (shape[0], shape[1]);
        let h_k = self.n_heads;
        let d_k = self.head_dim;
        let h_v = self.n_value_heads;
        let d_v = self.value_head_dim;
        let key_dim = h_k * d_k;
        let kv_repeat = h_v / h_k;

        // 1. Project QKV, alpha, beta, z
        let qkv_proj_t0 = std::time::Instant::now();
        let qkv = self.in_proj_qkv.forward(x)?;
        if let Some(profile) = profile.as_deref_mut() {
            qkv.eval()?;
            profile.qkv_proj_ms = qkv_proj_t0.elapsed().as_secs_f64() * 1000.0;
        }
        let ab_proj_t0 = std::time::Instant::now();
        let a = self.in_proj_a.forward(x)?;
        let b = self.in_proj_b.forward(x)?;
        if let Some(profile) = profile.as_deref_mut() {
            eval([&a, &b])?;
            profile.ab_proj_ms = ab_proj_t0.elapsed().as_secs_f64() * 1000.0;
        }
        let conv_dim = qkv.shape()[2];
        let kernel = self.conv_kernel;

        // 2. Conv1d — different path for prefill vs decode
        let conv_t0 = std::time::Instant::now();
        let qkv_conv = if seq_len == 1 {
            self.apply_decode_conv1d_step(&qkv, cache)?
        } else {
            let pad_widths: &[(i32, i32)] = &[(0, 0), (kernel - 1, 0), (0, 0)];
            let qkv_padded = mlx_rs::ops::pad(&qkv, pad_widths, None, None)?;
            let qkv_conv = mlx_rs::ops::conv1d(
                &qkv_padded,
                &*self.conv1d_weight,
                None,
                None,
                None,
                conv_dim as i32,
            )?;
            let history_len = kernel.saturating_sub(1);
            if history_len > 0 {
                let mut history =
                    mlx_rs::ops::zeros_dtype(&[batch, history_len, conv_dim], qkv.dtype())?;
                let tail_len = seq_len.min(history_len);
                if tail_len > 0 {
                    let tail = qkv.index((.., seq_len - tail_len.., ..));
                    history.try_index_mut((.., 0..tail_len, ..), &tail)?;
                }
                history.eval()?;
                cache.conv_buf = Some(history);
                cache.conv_pos = tail_len - 1;
            } else {
                cache.conv_buf = None;
                cache.conv_pos = -1;
            }
            nn::silu(&qkv_conv)?
        };
        if let Some(profile) = profile.as_deref_mut() {
            qkv_conv.eval()?;
            profile.conv_ms = conv_t0.elapsed().as_secs_f64() * 1000.0;
        }

        // 3. Split Q, K, V + normalize + GQA repeat
        let qk_norm_t0 = std::time::Instant::now();
        let parts = qkv_conv.split_axis(&[key_dim, 2 * key_dim], -1)?;
        let q = parts[0].reshape(&[batch, seq_len, h_k, d_k])?;
        let k = parts[1].reshape(&[batch, seq_len, h_k, d_k])?;
        let v = parts[2].reshape(&[batch, seq_len, h_v, d_v])?;

        let inv_scale = (d_k as f32).powf(-0.5);
        let ones_d = mlx_rs::ops::ones::<f32>(&[d_k])?;
        let q_flat = q.reshape(&[-1, d_k])?;
        let k_flat = k.reshape(&[-1, d_k])?;
        let q_norm =
            mlx_rs::fast::rms_norm(&q_flat, &ones_d, 1e-6)?.reshape(&[batch, seq_len, h_k, d_k])?;
        let k_norm =
            mlx_rs::fast::rms_norm(&k_flat, &ones_d, 1e-6)?.reshape(&[batch, seq_len, h_k, d_k])?;

        let (q, k) = if kv_repeat > 1 {
            let q_exp = mlx_rs::ops::broadcast_to(
                &q_norm.expand_dims(3)?,
                &[batch, seq_len, h_k, kv_repeat, d_k],
            )?
            .reshape(&[batch, seq_len, h_v, d_k])?;
            let k_exp = mlx_rs::ops::broadcast_to(
                &k_norm.expand_dims(3)?,
                &[batch, seq_len, h_k, kv_repeat, d_k],
            )?
            .reshape(&[batch, seq_len, h_v, d_k])?;
            (q_exp, k_exp)
        } else {
            (q_norm, k_norm)
        };
        let q = q.multiply(Array::from_f32(inv_scale * inv_scale))?;
        let k = k.multiply(Array::from_f32(inv_scale))?;
        if let Some(profile) = profile.as_deref_mut() {
            eval([&q, &k, &v])?;
            profile.qk_norm_ms = qk_norm_t0.elapsed().as_secs_f64() * 1000.0;
        }

        // 4. Decay and beta gates
        let gate_t0 = std::time::Instant::now();
        let a_plus_bias = a.add(&*self.dt_bias)?;
        let sp = nn::softplus(&a_plus_bias)?;
        let decay_rate = mlx_rs::ops::exp(&*self.a_log)?.multiply(&sp)?;
        let g = mlx_rs::ops::exp(&decay_rate.negative()?)?;
        let beta = nn::sigmoid(&b)?;
        if let Some(profile) = profile.as_deref_mut() {
            eval([&g, &beta])?;
            profile.gate_ms = gate_t0.elapsed().as_secs_f64() * 1000.0;
        }

        // 5. Recurrence with state caching — state: [B, H_v, D_v, D_k]
        let recurrence_t0 = std::time::Instant::now();
        let mut state = cache
            .state
            .take()
            .unwrap_or_else(|| mlx_rs::ops::zeros::<f32>(&[batch, h_v, d_v, d_k]).unwrap());
        let mut outputs: Vec<Array> = Vec::with_capacity(seq_len as usize);

        for t in 0..seq_len {
            let g_t = g.index((.., t, ..)).reshape(&[batch, h_v, 1, 1])?;
            let beta_t = beta.index((.., t, ..)).reshape(&[batch, h_v, 1])?;
            let q_t = q.index((.., t, .., ..));
            let k_t = k.index((.., t, .., ..));
            let v_t = v.index((.., t, .., ..));

            state = state.multiply(&g_t)?;
            let k_expanded = k_t.expand_dims(-2)?;
            let kv_mem = state.multiply(&k_expanded)?.sum_axes(&[-1], false)?;
            let delta = v_t.subtract(&kv_mem)?.multiply(&beta_t)?;
            let delta_expanded = delta.expand_dims(-1)?;
            state = state.add(&k_expanded.multiply(&delta_expanded)?)?;
            let q_expanded = q_t.expand_dims(-2)?;
            let y_t = state.multiply(&q_expanded)?.sum_axes(&[-1], false)?;
            outputs.push(y_t);
        }

        // Save final recurrent state
        state.eval()?;
        cache.state = Some(state);
        if let Some(profile) = profile.as_deref_mut() {
            profile.recurrence_ms = recurrence_t0.elapsed().as_secs_f64() * 1000.0;
        }

        // 6. Stack + RMSNormGated + out_proj
        let y = mlx_rs::ops::stack_axis(&outputs, 1)?;
        let value_dim = h_v * d_v;
        let z_proj_t0 = std::time::Instant::now();
        let z = self
            .in_proj_z
            .forward(x)?
            .reshape(&[batch, seq_len, h_v, d_v])?;
        if let Some(profile) = profile.as_deref_mut() {
            z.eval()?;
            profile.z_proj_ms = z_proj_t0.elapsed().as_secs_f64() * 1000.0;
        }
        let output_t0 = std::time::Instant::now();
        let y_flat = y.reshape(&[-1, d_v])?;
        let y_normed = self
            .norm
            .forward(&y_flat)?
            .reshape(&[batch, seq_len, h_v, d_v])?;
        let y = nn::silu(&z)?.multiply(&y_normed)?;
        let y = y.reshape(&[batch, seq_len, value_dim])?;
        let out = self.out_proj.forward(&y)?;
        if let Some(profile) = profile.as_deref_mut() {
            out.eval()?;
            profile.output_ms = output_t0.elapsed().as_secs_f64() * 1000.0;
        }
        Ok(out)
    }
}

fn apply_decode_conv1d_step_with_weight(
    conv1d_weight: &Array,
    conv_kernel: i32,
    qkv: &Array,
    cache: &mut GdnCache,
) -> Result<Array, Exception> {
    let shape = qkv.shape();
    let (batch, seq_len, conv_dim) = (shape[0], shape[1], shape[2]);
    debug_assert_eq!(seq_len, 1, "decode conv expects a single token");

    let history_len = conv_kernel.saturating_sub(1);
    let qkv_flat = qkv.reshape(&[batch, conv_dim])?;
    let current_weight = conv1d_weight.index((.., conv_kernel - 1, 0));
    let mut conv_flat = qkv_flat.multiply(&current_weight)?;

    if history_len > 0 {
        if let Some(history) = cache.conv_buf.as_ref() {
            debug_assert_eq!(history.shape(), &[batch, history_len, conv_dim]);
            for lag in 0..history_len {
                if cache.conv_pos < 0 {
                    break;
                }
                let idx = (cache.conv_pos - lag).rem_euclid(history_len);
                let prev = history
                    .index((.., idx..idx + 1, ..))
                    .reshape(&[batch, conv_dim])?;
                let weight = conv1d_weight.index((.., history_len - 1 - lag, 0));
                conv_flat = conv_flat.add(&prev.multiply(&weight)?)?;
            }
        }

        let history = if let Some(history) = cache.conv_buf.as_mut() {
            history
        } else {
            cache.conv_buf = Some(mlx_rs::ops::zeros_dtype(
                &[batch, history_len, conv_dim],
                qkv.dtype(),
            )?);
            cache
                .conv_buf
                .as_mut()
                .ok_or_else(|| Exception::custom("decode conv history buffer missing"))?
        };
        let next_pos = if cache.conv_pos < 0 {
            0
        } else {
            (cache.conv_pos + 1).rem_euclid(history_len)
        };
        history.try_index_mut((.., next_pos..next_pos + 1, ..), qkv)?;
        history.eval()?;
        cache.conv_pos = next_pos;
    }

    nn::silu(&conv_flat.reshape(&[batch, 1, conv_dim])?)
}

fn apply_decode_conv1d_step_reference_with_weight(
    conv1d_weight: &Array,
    conv_kernel: i32,
    qkv: &Array,
    cache: &mut GdnCache,
) -> Result<Array, Exception> {
    let shape = qkv.shape();
    let (batch, seq_len, conv_dim) = (shape[0], shape[1], shape[2]);
    debug_assert_eq!(seq_len, 1, "decode conv expects a single token");
    let history_len = conv_kernel.saturating_sub(1);

    let buf = if let Some(ref cb) = cache.conv_buf {
        let combined = mlx_rs::ops::concatenate_axis(&[cb, qkv], 1)?;
        let new_buf = combined.index((.., 1.., ..));
        new_buf.eval()?;
        cache.conv_buf = Some(new_buf);
        combined
    } else {
        let zeros = mlx_rs::ops::zeros_dtype(&[batch, history_len, conv_dim], qkv.dtype())?;
        let combined = mlx_rs::ops::concatenate_axis(&[&zeros, qkv], 1)?;
        let new_buf = combined.index((.., 1.., ..));
        new_buf.eval()?;
        cache.conv_buf = Some(new_buf);
        combined
    };
    let conv_out = mlx_rs::ops::conv1d(&buf, conv1d_weight, None, None, None, conv_dim as i32)?;
    nn::silu(&conv_out)
}

impl ModuleParameters for MlxLinearAttention {
    fn num_parameters(&self) -> usize {
        self.in_proj_qkv.num_parameters()
            + self.in_proj_a.num_parameters()
            + self.in_proj_b.num_parameters()
            + self.in_proj_z.num_parameters()
            + self.out_proj.num_parameters()
            + self.norm.num_parameters()
            + 3
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
    fn all_frozen(&self) -> Option<bool> {
        Some(true)
    }
    fn any_frozen(&self) -> Option<bool> {
        Some(true)
    }
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
                weights,
                &format!("{prefix}.linear_attn"),
                cfg,
            )?)
        } else {
            AttentionKind::Full(MlxLoraAttention::load(
                weights,
                &format!("{prefix}.self_attn"),
                cfg,
                lora_cfg,
            )?)
        };

        Ok(MlxLoraDecoderLayer {
            attn,
            mlp: MlxLoraMLP::load(weights, &format!("{prefix}.mlp"), cfg, lora_cfg)?,
            input_layernorm: load_rms_norm(
                weights,
                &format!("{prefix}.input_layernorm"),
                cfg.rms_eps,
            )?,
            post_attention_layernorm: load_rms_norm(
                weights,
                &format!("{prefix}.post_attention_layernorm"),
                cfg.rms_eps,
            )?,
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

    pub fn linear_attention(&self) -> Option<&MlxLinearAttention> {
        match &self.attn {
            AttentionKind::Linear(attn) => Some(attn),
            AttentionKind::Full(_) => None,
        }
    }

    pub fn linear_attention_mut(&mut self) -> Option<&mut MlxLinearAttention> {
        match &mut self.attn {
            AttentionKind::Linear(attn) => Some(attn),
            AttentionKind::Full(_) => None,
        }
    }

    /// Apply LoRA weight arrays to a specific target in this layer.
    /// Returns true if applied, false if the target doesn't exist (e.g. attention on GDN layer).
    pub fn apply_lora_weights(&mut self, target: &str, new_a: Array, new_b: Array) -> bool {
        match target {
            "q_proj" | "v_proj" | "o_proj" => {
                let attn = match &mut self.attn {
                    AttentionKind::Full(a) => a,
                    AttentionKind::Linear(_) => return false,
                };
                let ll = match target {
                    "q_proj" => &mut attn.q_proj,
                    "v_proj" => &mut attn.v_proj,
                    "o_proj" => &mut attn.o_proj,
                    _ => unreachable!(),
                };
                *ll.lora_a.weight = new_a;
                *ll.lora_b.weight = new_b;
                ll.set_adapter_active(true);
                true
            }
            "down_proj" => {
                *self.mlp.down_proj.lora_a.weight = new_a;
                *self.mlp.down_proj.lora_b.weight = new_b;
                self.mlp.down_proj.set_adapter_active(true);
                true
            }
            _ => false,
        }
    }

    pub fn is_full_attention(&self) -> bool {
        matches!(&self.attn, AttentionKind::Full(_))
    }

    fn set_lora_training_mode(&mut self, mode: bool) {
        if let AttentionKind::Full(attn) = &mut self.attn {
            attn.set_lora_training_mode(mode);
        }
        self.mlp.set_lora_training_mode(mode);
    }

    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        let residual = x;
        let h = self.input_layernorm.forward(x)?;
        let h = match &mut self.attn {
            AttentionKind::Full(attn) => attn.forward(&h, mask)?,
            AttentionKind::Linear(attn) => mlx_rs::stop_gradient(&attn.forward(&h)?)?,
        };
        let x = h.add(residual)?;

        let residual = &x;
        let h = self.post_attention_layernorm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        h.add(residual)
    }

    /// Forward with layer cache for incremental generation.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut LayerCache>,
        compiled_plain_kv_decode: Option<&mut CompiledPlainKvDecodeFn>,
        mut linear_profile: Option<&mut LinearDecodeSubProfile>,
    ) -> Result<Array, Exception> {
        let residual = x;
        let input_norm_t0 = std::time::Instant::now();
        let h = self.input_layernorm.forward(x)?;
        if let Some(profile) = linear_profile.as_deref_mut() {
            h.eval()?;
            profile.input_norm_ms = input_norm_t0.elapsed().as_secs_f64() * 1000.0;
        }
        let h = match (&mut self.attn, cache) {
            (AttentionKind::Full(attn), Some(LayerCache::FullAttn(c))) => {
                attn.forward_with_cache(&h, mask, c, compiled_plain_kv_decode)?
            }
            (AttentionKind::Linear(attn), Some(LayerCache::LinearAttn(c))) => {
                mlx_rs::stop_gradient(&attn.forward_with_cache_internal(
                    &h,
                    c,
                    linear_profile.as_deref_mut(),
                )?)?
            }
            (AttentionKind::Full(attn), _) => attn.forward(&h, mask)?,
            (AttentionKind::Linear(attn), _) => mlx_rs::stop_gradient(&attn.forward(&h)?)?,
        };
        let attn_residual_t0 = std::time::Instant::now();
        let x = h.add(residual)?;
        if let Some(profile) = linear_profile.as_deref_mut() {
            x.eval()?;
            profile.attn_residual_ms = attn_residual_t0.elapsed().as_secs_f64() * 1000.0;
        }

        let residual = &x;
        let post_attn_norm_t0 = std::time::Instant::now();
        let h = self.post_attention_layernorm.forward(&x)?;
        if let Some(profile) = linear_profile.as_deref_mut() {
            h.eval()?;
            profile.post_attn_norm_ms = post_attn_norm_t0.elapsed().as_secs_f64() * 1000.0;
        }
        let h = self
            .mlp
            .forward_profiled(&h, linear_profile.as_deref_mut())?;
        let final_residual_t0 = std::time::Instant::now();
        let out = h.add(residual)?;
        if let Some(profile) = linear_profile.as_deref_mut() {
            out.eval()?;
            profile.final_residual_ms = final_residual_t0.elapsed().as_secs_f64() * 1000.0;
        }
        Ok(out)
    }
}

impl ModuleParameters for MlxLoraDecoderLayer {
    fn num_parameters(&self) -> usize {
        let attn_params = match &self.attn {
            AttentionKind::Full(a) => a.num_parameters(),
            AttentionKind::Linear(a) => a.num_parameters(),
        };
        attn_params
            + self.mlp.num_parameters()
            + self.input_layernorm.num_parameters()
            + self.post_attention_layernorm.num_parameters()
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
            (
                "post_attention_layernorm",
                self.post_attention_layernorm.parameters(),
            ),
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
            (
                "post_attention_layernorm",
                self.post_attention_layernorm.parameters_mut(),
            ),
        ])
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        let mut entries = vec![("mlp", self.mlp.trainable_parameters())];
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CachedDecodeLayerKind {
    FullAttention,
    LinearAttention,
}

impl CachedDecodeLayerKind {
    pub fn label(self) -> &'static str {
        match self {
            CachedDecodeLayerKind::FullAttention => "full-attn",
            CachedDecodeLayerKind::LinearAttention => "linear-attn",
        }
    }
}

#[derive(Clone, Debug)]
pub struct CachedDecodeLayerProfile {
    pub layer_idx: usize,
    pub kind: CachedDecodeLayerKind,
    pub cache_len_before: i32,
    pub quantized_kv: bool,
    pub compiled_decode: bool,
    pub total_ms: f64,
    pub linear_decode: Option<LinearDecodeSubProfile>,
}

#[derive(Clone, Debug, Default)]
pub struct LinearDecodeSubProfile {
    pub input_norm_ms: f64,
    pub qkv_proj_ms: f64,
    pub ab_proj_ms: f64,
    pub conv_ms: f64,
    pub qk_norm_ms: f64,
    pub gate_ms: f64,
    pub recurrence_ms: f64,
    pub z_proj_ms: f64,
    pub output_ms: f64,
    pub attn_residual_ms: f64,
    pub post_attn_norm_ms: f64,
    pub mlp_gate_proj_ms: f64,
    pub mlp_up_proj_ms: f64,
    pub mlp_act_mul_ms: f64,
    pub mlp_down_proj_ms: f64,
    pub final_residual_ms: f64,
}

impl LinearDecodeSubProfile {
    pub fn total_ms(&self) -> f64 {
        self.input_norm_ms
            + self.qkv_proj_ms
            + self.ab_proj_ms
            + self.conv_ms
            + self.qk_norm_ms
            + self.gate_ms
            + self.recurrence_ms
            + self.z_proj_ms
            + self.output_ms
            + self.attn_residual_ms
            + self.post_attn_norm_ms
            + self.mlp_gate_proj_ms
            + self.mlp_up_proj_ms
            + self.mlp_act_mul_ms
            + self.mlp_down_proj_ms
            + self.final_residual_ms
    }
}

#[derive(Clone, Debug, Default)]
pub struct CachedDecodeProfile {
    pub total_ms: f64,
    pub embed_ms: f64,
    pub mask_ms: f64,
    pub layer_profiles: Vec<CachedDecodeLayerProfile>,
    pub final_norm_ms: f64,
    pub logits_ms: f64,
}

impl CachedDecodeProfile {
    pub fn layer_total_ms(&self) -> f64 {
        self.layer_profiles.iter().map(|layer| layer.total_ms).sum()
    }

    pub fn kind_total_ms(&self, kind: CachedDecodeLayerKind) -> f64 {
        self.layer_profiles
            .iter()
            .filter(|layer| layer.kind == kind)
            .map(|layer| layer.total_ms)
            .sum()
    }

    pub fn unattributed_ms(&self) -> f64 {
        (self.total_ms
            - self.embed_ms
            - self.mask_ms
            - self.layer_total_ms()
            - self.final_norm_ms
            - self.logits_ms)
            .max(0.0)
    }

    pub fn linear_decode_stage_totals(&self) -> LinearDecodeSubProfile {
        let mut totals = LinearDecodeSubProfile::default();
        for layer in &self.layer_profiles {
            let Some(linear) = layer.linear_decode.as_ref() else {
                continue;
            };
            totals.input_norm_ms += linear.input_norm_ms;
            totals.qkv_proj_ms += linear.qkv_proj_ms;
            totals.ab_proj_ms += linear.ab_proj_ms;
            totals.conv_ms += linear.conv_ms;
            totals.qk_norm_ms += linear.qk_norm_ms;
            totals.gate_ms += linear.gate_ms;
            totals.recurrence_ms += linear.recurrence_ms;
            totals.z_proj_ms += linear.z_proj_ms;
            totals.output_ms += linear.output_ms;
            totals.attn_residual_ms += linear.attn_residual_ms;
            totals.post_attn_norm_ms += linear.post_attn_norm_ms;
            totals.mlp_gate_proj_ms += linear.mlp_gate_proj_ms;
            totals.mlp_up_proj_ms += linear.mlp_up_proj_ms;
            totals.mlp_act_mul_ms += linear.mlp_act_mul_ms;
            totals.mlp_down_proj_ms += linear.mlp_down_proj_ms;
            totals.final_residual_ms += linear.final_residual_ms;
        }
        totals
    }
}

#[derive(Debug)]
pub struct MlxLoraModel {
    pub embed_tokens: MlxEmbedding,
    pub layers: Vec<MlxLoraDecoderLayer>,
    pub norm: RmsNorm,
    /// None = tied to embed_tokens (use QuantizedEmbedding::as_linear)
    pub lm_head: Option<QuantizedLinear>,
    pub kv_cache_config: KvCacheConfig,
}

impl MlxLoraModel {
    pub fn load(
        model_dir: &Path,
        cfg: &ModelConfig,
        lora_cfg: &LoraConfig,
    ) -> Result<Self, anyhow::Error> {
        tracing::info!(path = %model_dir.display(), "loading MLX weights");
        let t0 = std::time::Instant::now();
        let mut weights = load_weights(model_dir)?;
        tracing::info!(
            tensors = weights.len(),
            ms = t0.elapsed().as_millis(),
            "weights loaded"
        );

        let pfx = cfg.weight_prefix;

        let embed_tokens = load_embedding(
            &mut weights,
            &format!("{pfx}.embed_tokens"),
            cfg.group_size,
            cfg.bits,
        )?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(MlxLoraDecoderLayer::load(
                &mut weights,
                &format!("{pfx}.layers.{i}"),
                i,
                cfg,
                lora_cfg,
            )?);
        }

        let norm = load_rms_norm(&mut weights, &format!("{pfx}.norm"), cfg.rms_eps)?;

        let lm_head_key = if pfx == "model" {
            "lm_head"
        } else {
            "language_model.lm_head"
        };
        let lm_head = if weights.contains_key(&format!("{lm_head_key}.weight")) {
            Some(load_quantized_linear(
                &mut weights,
                lm_head_key,
                cfg.group_size,
                cfg.bits,
            )?)
        } else {
            None // tied to embed_tokens
        };

        // Filter out vision_tower weights (not needed for text LoRA)
        weights.retain(|k, _| !k.starts_with("vision_tower"));

        Ok(MlxLoraModel {
            embed_tokens,
            layers,
            norm,
            lm_head,
            kv_cache_config: KvCacheConfig::default(),
        })
    }

    fn set_lora_training_mode(&mut self, mode: bool) {
        for layer in &mut self.layers {
            layer.set_lora_training_mode(mode);
        }
    }

    fn mark_adapters_active(&mut self) {
        for layer in &mut self.layers {
            if let AttentionKind::Full(attn) = &mut layer.attn {
                attn.q_proj.set_adapter_active(true);
                attn.v_proj.set_adapter_active(true);
                attn.o_proj.set_adapter_active(true);
            }
            layer.mlp.down_proj.set_adapter_active(true);
        }
    }

    fn build_layer_caches(&self, capacity: i32) -> Vec<LayerCache> {
        self.layers
            .iter()
            .map(|l| {
                if l.is_full_attention() {
                    if let Some(bits) = self.kv_cache_config.bits {
                        if self.kv_cache_config.quantized_start > 0 {
                            LayerCache::FullAttn(KvCache::new_promotable(
                                capacity,
                                self.kv_cache_config.group_size,
                                bits,
                                self.kv_cache_config.quantized_start,
                            ))
                        } else {
                            LayerCache::FullAttn(KvCache::new_quantized(
                                capacity,
                                self.kv_cache_config.group_size,
                                bits,
                            ))
                        }
                    } else {
                        LayerCache::FullAttn(KvCache::new(capacity))
                    }
                } else {
                    LayerCache::LinearAttn(GdnCache::new())
                }
            })
            .collect()
    }

    pub fn prefill(
        &mut self,
        prompt_tokens: &[i32],
        max_tokens: usize,
    ) -> Result<PrefillState, Exception> {
        let capacity = (prompt_tokens.len().saturating_add(max_tokens)) as i32;
        let mut caches = self.build_layer_caches(capacity);
        let input = Array::from_slice(prompt_tokens, &[1, prompt_tokens.len() as i32]);
        let logits = self.forward_logits_cached(&input, &mut caches)?;
        let last_logits = logits.index((.., -1, ..));
        last_logits.eval()?;
        Ok(PrefillState {
            caches,
            last_logits,
            prompt_len: prompt_tokens.len(),
        })
    }

    pub fn ensure_prefill_capacity(
        &mut self,
        state: &mut PrefillState,
        max_tokens: usize,
    ) -> Result<(), Exception> {
        let required_capacity = (state.prompt_len.saturating_add(max_tokens)) as i32;
        for cache in &mut state.caches {
            if let LayerCache::FullAttn(kv) = cache {
                kv.reserve(required_capacity)?;
            }
        }
        Ok(())
    }

    pub fn generate_from_prefill(
        &mut self,
        state: PrefillState,
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[i32],
    ) -> Result<Vec<i32>, Exception> {
        self.decode_from_prefill(state, max_tokens, temperature, stop_tokens, |_| true)
    }

    pub fn generate_stream_from_prefill<F>(
        &mut self,
        state: PrefillState,
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[i32],
        on_token: F,
    ) -> Result<Vec<i32>, Exception>
    where
        F: FnMut(i32) -> bool,
    {
        self.decode_from_prefill(state, max_tokens, temperature, stop_tokens, on_token)
    }

    fn decode_from_prefill<F>(
        &mut self,
        mut state: PrefillState,
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[i32],
        mut on_token: F,
    ) -> Result<Vec<i32>, Exception>
    where
        F: FnMut(i32) -> bool,
    {
        self.ensure_prefill_capacity(&mut state, max_tokens)?;
        let compiled_decode = compiled_decode_enabled();
        if compiled_decode {
            for cache in &mut state.caches {
                if let LayerCache::FullAttn(KvCache::Plain(kv)) = cache {
                    kv.prepare_compiled_decode_state()?;
                }
            }
        }
        let mut caches = state.caches;
        let mut generated = Vec::with_capacity(max_tokens);
        if max_tokens == 0 {
            return Ok(generated);
        }

        let first_full_attn_scale = self.layers.iter().find_map(|layer| match &layer.attn {
            AttentionKind::Full(attn) => Some(attn.attn_scale),
            AttentionKind::Linear(_) => None,
        });
        let mut compiled_plain_kv_decode = if compiled_decode && first_full_attn_scale.is_some() {
            Some(make_compiled_plain_kv_decode(
                first_full_attn_scale.unwrap(),
            ))
        } else {
            None
        };

        let mut current_token = sample_next_token(&state.last_logits, temperature)?;

        for step in 0..max_tokens {
            let next_token = if step + 1 < max_tokens {
                let input = current_token.reshape(&[1, 1])?;
                let logits = self.forward_logits_cached_with_compiled(
                    &input,
                    &mut caches,
                    &mut compiled_plain_kv_decode,
                )?;
                let last_logits = logits.index((.., -1, ..));
                let token = sample_next_token(&last_logits, temperature)?;
                async_eval([&token])?;
                Some(token)
            } else {
                None
            };

            let token_id: i32 = current_token.as_slice::<i32>()[0];

            if stop_tokens.contains(&token_id) {
                break;
            }

            generated.push(token_id);
            if !on_token(token_id) {
                break;
            }

            if let Some(token) = next_token {
                current_token = token;
            } else {
                break;
            }

            if step > 0 && step % 256 == 0 {
                clear_cache();
            }
        }

        Ok(generated)
    }

    /// Autoregressive text generation with KV cache.
    ///
    /// Phase 1 (prefill): full prompt processed in one forward pass, KV cached.
    /// Phase 2 (decode): one token at a time, reusing cached KV.
    pub fn generate(
        &mut self,
        prompt_tokens: &[i32],
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[i32],
    ) -> Result<Vec<i32>, Exception> {
        let state = self.prefill(prompt_tokens, max_tokens)?;
        self.generate_from_prefill(state, max_tokens, temperature, stop_tokens)
    }

    /// Forward pass with layer caches for incremental generation.
    fn forward_logits_cached(
        &mut self,
        tokens: &Array,
        caches: &mut [LayerCache],
    ) -> Result<Array, Exception> {
        self.forward_logits_cached_internal(tokens, caches, &mut None, None)
    }

    pub fn profile_forward_logits_cached_step(
        &mut self,
        tokens: &Array,
        caches: &mut [LayerCache],
    ) -> Result<(Array, CachedDecodeProfile), Exception> {
        let compiled_decode = compiled_decode_enabled();
        if compiled_decode {
            for cache in caches.iter_mut() {
                if let LayerCache::FullAttn(KvCache::Plain(kv)) = cache {
                    kv.prepare_compiled_decode_state()?;
                }
            }
        }

        let first_full_attn_scale = self.layers.iter().find_map(|layer| match &layer.attn {
            AttentionKind::Full(attn) => Some(attn.attn_scale),
            AttentionKind::Linear(_) => None,
        });
        let mut compiled_plain_kv_decode = if compiled_decode && first_full_attn_scale.is_some() {
            Some(make_compiled_plain_kv_decode(
                first_full_attn_scale.unwrap(),
            ))
        } else {
            None
        };

        let mut profile = CachedDecodeProfile::default();
        let logits = self.forward_logits_cached_internal(
            tokens,
            caches,
            &mut compiled_plain_kv_decode,
            Some(&mut profile),
        )?;
        Ok((logits, profile))
    }

    fn forward_logits_cached_with_compiled(
        &mut self,
        tokens: &Array,
        caches: &mut [LayerCache],
        compiled_plain_kv_decode: &mut Option<Box<CompiledPlainKvDecodeFn>>,
    ) -> Result<Array, Exception> {
        self.forward_logits_cached_internal(tokens, caches, compiled_plain_kv_decode, None)
    }

    fn forward_logits_cached_internal(
        &mut self,
        tokens: &Array,
        caches: &mut [LayerCache],
        compiled_plain_kv_decode: &mut Option<Box<CompiledPlainKvDecodeFn>>,
        mut profile: Option<&mut CachedDecodeProfile>,
    ) -> Result<Array, Exception> {
        let total_t0 = std::time::Instant::now();
        let profiling_enabled = profile.is_some();
        let q_len = tokens.shape()[1] as i32;
        let embed_t0 = std::time::Instant::now();
        let mut h = self.embed_tokens.forward(tokens)?;
        if let Some(profile) = profile.as_deref_mut() {
            h.eval()?;
            profile.embed_ms = embed_t0.elapsed().as_secs_f64() * 1000.0;
        }

        // Determine total KV length from first full-attention cache that has data
        let kv_len = caches
            .iter()
            .find_map(|c| {
                if let LayerCache::FullAttn(kv) = c {
                    Some(kv.len() + q_len)
                } else {
                    None
                }
            })
            .unwrap_or(q_len);

        let mask_t0 = std::time::Instant::now();
        let mask = if needs_explicit_cached_mask(q_len, kv_len) {
            Some(create_causal_mask_cached(q_len, kv_len)?)
        } else {
            None
        };
        if let Some(profile) = profile.as_deref_mut() {
            if let Some(mask) = mask.as_ref() {
                mask.eval()?;
            }
            profile.mask_ms = mask_t0.elapsed().as_secs_f64() * 1000.0;
        }

        for (layer_idx, (layer, cache)) in self.layers.iter_mut().zip(caches.iter_mut()).enumerate()
        {
            let kind = if layer.is_full_attention() {
                CachedDecodeLayerKind::FullAttention
            } else {
                CachedDecodeLayerKind::LinearAttention
            };
            let (cache_len_before, quantized_kv) = match cache {
                LayerCache::FullAttn(KvCache::Plain(kv)) => (kv.len(), false),
                LayerCache::FullAttn(KvCache::Quantized(kv)) => (kv.len(), true),
                LayerCache::LinearAttn(_) => (0, false),
            };
            let compiled_decode = q_len == 1
                && mask.is_none()
                && compiled_plain_kv_decode.is_some()
                && matches!(cache, LayerCache::FullAttn(KvCache::Plain(kv)) if kv.compiled_decode_ready());
            let layer_t0 = std::time::Instant::now();
            let mut linear_decode =
                if profiling_enabled && kind == CachedDecodeLayerKind::LinearAttention {
                    Some(LinearDecodeSubProfile::default())
                } else {
                    None
                };
            h = layer.forward_with_cache(
                &h,
                mask.as_ref(),
                Some(cache),
                compiled_plain_kv_decode.as_deref_mut(),
                linear_decode.as_mut(),
            )?;
            if let Some(profile) = profile.as_deref_mut() {
                h.eval()?;
                profile.layer_profiles.push(CachedDecodeLayerProfile {
                    layer_idx,
                    kind,
                    cache_len_before,
                    quantized_kv,
                    compiled_decode,
                    total_ms: layer_t0.elapsed().as_secs_f64() * 1000.0,
                    linear_decode,
                });
            }
        }

        let norm_t0 = std::time::Instant::now();
        h = self.norm.forward(&h)?;
        if let Some(profile) = profile.as_deref_mut() {
            h.eval()?;
            profile.final_norm_ms = norm_t0.elapsed().as_secs_f64() * 1000.0;
        }
        let logits_t0 = std::time::Instant::now();
        let logits = if let Some(lm) = &mut self.lm_head {
            lm.forward(&h)
        } else {
            match &self.embed_tokens {
                MlxEmbedding::Quantized(qe) => qe.as_linear(&h),
                MlxEmbedding::Plain(e) => {
                    mlx_rs::ops::matmul(&h, &e.weight.transpose_axes(&[1, 0])?)
                }
            }
        }?;
        if let Some(profile) = profile.as_deref_mut() {
            logits.eval()?;
            profile.logits_ms = logits_t0.elapsed().as_secs_f64() * 1000.0;
            profile.total_ms = total_t0.elapsed().as_secs_f64() * 1000.0;
        }
        Ok(logits)
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
        let generated = self
            .generate(&prompt_tokens, max_tokens, temperature, &[eos])
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
            + self
                .layers
                .iter()
                .map(|l| l.num_parameters())
                .sum::<usize>()
            + self.norm.num_parameters()
            + self.lm_head.as_ref().map_or(0, |l| l.num_parameters())
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut map = NestedHashMap::new();
        map.insert(
            Rc::from("embed_tokens"),
            self.embed_tokens.parameters().into(),
        );
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
        map.insert(
            Rc::from("embed_tokens"),
            self.embed_tokens.parameters_mut().into(),
        );
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
            layers_entries.insert(
                Rc::from(i.to_string().as_str()),
                l.trainable_parameters().into(),
            );
        }
        map.insert(Rc::from("layers"), NestedValue::Map(layers_entries));
        map
    }

    fn freeze_parameters(&mut self, r: bool) {
        self.embed_tokens.freeze_parameters(r);
        for l in &mut self.layers {
            l.freeze_parameters(r);
        }
        self.norm.freeze_parameters(r);
        if let Some(lm) = &mut self.lm_head {
            lm.freeze_parameters(r);
        }
    }

    fn unfreeze_parameters(&mut self, r: bool) {
        for l in &mut self.layers {
            l.unfreeze_parameters(r);
        }
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
        .flat_map(|i| (0..seq_len).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
        .collect();
    Ok(Array::from_slice(&data, &[1, 1, seq_len, seq_len]))
}

fn needs_explicit_cached_mask(q_len: i32, kv_len: i32) -> bool {
    q_len > 1 && q_len != kv_len
}

/// Create a causal mask for cached generation.
///
/// During prefill (no cache), this is the standard `[1,1,S,S]` causal mask.
/// During decode (with cache), the query has `q_len` new tokens attending to
/// `kv_len` total cached+new tokens: shape `[1,1,q_len,kv_len]`.
fn create_causal_mask_cached(q_len: i32, kv_len: i32) -> Result<Array, Exception> {
    if q_len == kv_len {
        return create_causal_mask(q_len);
    }
    // Decode step: q_len new tokens can attend to all kv_len tokens,
    // with causal constraint: query position i attends to kv positions ≤ (kv_len - q_len + i).
    let offset = kv_len - q_len;
    let data: Vec<f32> = (0..q_len)
        .flat_map(|i| {
            (0..kv_len).map(move |j| {
                if j <= offset + i {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Ok(Array::from_slice(&data, &[1, 1, q_len, kv_len]))
}

// ---------------------------------------------------------------------------
// KV Cache + GDN Cache
// ---------------------------------------------------------------------------

/// Per-layer cache: KV for full attention, GDN state for linear attention.
#[derive(Clone, Debug)]
pub enum LayerCache {
    /// Full attention: cached K/V tensors.
    FullAttn(KvCache),
    /// Linear attention (GDN): recurrent state + conv1d buffer.
    LinearAttn(GdnCache),
}

#[derive(Clone, Debug)]
pub enum KvCacheView {
    Plain { keys: Array, values: Array },
    Quantized(QuantizedKvView),
}

#[derive(Clone, Debug)]
pub struct QuantizedKvView {
    pub keys: QuantizedArray,
    pub values: QuantizedArray,
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Clone, Debug)]
pub struct QuantizedArray {
    pub packed: Array,
    pub scales: Array,
    pub biases: Array,
}

#[derive(Clone, Copy, Debug)]
struct QuantizedKvPromotionConfig {
    group_size: i32,
    bits: i32,
    start: i32,
}

#[derive(Clone, Debug)]
pub struct PlainKvCache {
    pub keys: Option<Array>,
    pub values: Option<Array>,
    len: i32,
    capacity: i32,
    pending_updates: usize,
    promotion: Option<QuantizedKvPromotionConfig>,
    len_array: Option<Array>,
    positions: Option<Array>,
}

#[derive(Clone, Debug)]
pub struct QuantizedKvCache {
    pub keys: Option<QuantizedArray>,
    pub values: Option<QuantizedArray>,
    len: i32,
    capacity: i32,
    pending_updates: usize,
    group_size: i32,
    bits: i32,
}

/// Per-layer KV cache for incremental generation.
#[derive(Clone, Debug)]
pub enum KvCache {
    Plain(PlainKvCache),
    Quantized(QuantizedKvCache),
}

/// GDN (Gated Delta Net) recurrent state for incremental generation.
///
/// Stores the recurrent state matrix and conv1d circular buffer.
#[derive(Clone, Debug)]
pub struct GdnCache {
    /// Recurrent state: `[B, H, D_v, D_k]`
    pub state: Option<Array>,
    /// Conv1d circular buffer: last `kernel_size - 1` inputs, `[B, kernel-1, conv_dim]`
    pub conv_buf: Option<Array>,
    /// Index of the most recent token in `conv_buf` when present.
    pub conv_pos: i32,
}

fn quantized_layout_dims(
    head_dim: i32,
    group_size: i32,
    bits: i32,
) -> Result<(i32, i32), Exception> {
    if !(bits == 2 || bits == 4 || bits == 8) {
        return Err(Exception::custom(format!(
            "unsupported quantized KV bit-width {bits}; expected 2, 4, or 8"
        )));
    }
    let el_per_int = 32 / bits;
    if head_dim % group_size != 0 {
        return Err(Exception::custom(format!(
            "head_dim {head_dim} must be divisible by KV group_size {group_size}"
        )));
    }
    if head_dim % el_per_int != 0 {
        return Err(Exception::custom(format!(
            "head_dim {head_dim} must be divisible by packed width {el_per_int}"
        )));
    }
    Ok((head_dim / el_per_int, head_dim / group_size))
}

impl QuantizedArray {
    fn zeros_storage(
        batch: i32,
        heads: i32,
        seq: i32,
        head_dim: i32,
        value_dtype: Dtype,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, Exception> {
        let (packed_dim, scale_dim) = quantized_layout_dims(head_dim, group_size, bits)?;
        Ok(QuantizedArray {
            packed: mlx_rs::ops::zeros_dtype(&[batch, heads, seq, packed_dim], Dtype::Uint32)?,
            scales: mlx_rs::ops::zeros_dtype(&[batch, heads, seq, scale_dim], value_dtype)?,
            biases: mlx_rs::ops::zeros_dtype(&[batch, heads, seq, scale_dim], value_dtype)?,
        })
    }

    fn from_dense(dense: &Array, group_size: i32, bits: i32) -> Result<Self, Exception> {
        let shape = dense.shape();
        if shape.len() != 4 {
            return Err(Exception::custom(format!(
                "quantized KV expects a 4D tensor, got shape {:?}",
                shape
            )));
        }

        let (batch, heads, seq, head_dim) = (shape[0], shape[1], shape[2], shape[3]);
        let (packed_dim, scale_dim) = quantized_layout_dims(head_dim, group_size, bits)?;
        let flat = dense.reshape(&[-1, head_dim])?;
        let (packed, scales, biases) = mlx_rs::ops::quantize(&flat, group_size, bits)?;

        Ok(QuantizedArray {
            packed: packed.reshape(&[batch, heads, seq, packed_dim])?,
            scales: scales.reshape(&[batch, heads, seq, scale_dim])?,
            biases: biases.reshape(&[batch, heads, seq, scale_dim])?,
        })
    }

    fn with_capacity(&self, capacity: i32) -> Result<Self, Exception> {
        let packed_shape = self.packed.shape();
        let scale_shape = self.scales.shape();
        Ok(QuantizedArray {
            packed: mlx_rs::ops::zeros_dtype(
                &[packed_shape[0], packed_shape[1], capacity, packed_shape[3]],
                self.packed.dtype(),
            )?,
            scales: mlx_rs::ops::zeros_dtype(
                &[scale_shape[0], scale_shape[1], capacity, scale_shape[3]],
                self.scales.dtype(),
            )?,
            biases: mlx_rs::ops::zeros_dtype(
                &[scale_shape[0], scale_shape[1], capacity, scale_shape[3]],
                self.biases.dtype(),
            )?,
        })
    }

    fn view_prefix(&self, len: i32) -> Self {
        QuantizedArray {
            packed: self.packed.index((.., .., ..len, ..)),
            scales: self.scales.index((.., .., ..len, ..)),
            biases: self.biases.index((.., .., ..len, ..)),
        }
    }

    fn expand_dims(&self, axis: i32) -> Result<Self, Exception> {
        Ok(QuantizedArray {
            packed: self.packed.expand_dims(axis)?,
            scales: self.scales.expand_dims(axis)?,
            biases: self.biases.expand_dims(axis)?,
        })
    }

    fn write_range(&mut self, start: i32, end: i32, src: &QuantizedArray) -> Result<(), Exception> {
        self.packed
            .try_index_mut((.., .., start..end, ..), &src.packed)?;
        self.scales
            .try_index_mut((.., .., start..end, ..), &src.scales)?;
        self.biases
            .try_index_mut((.., .., start..end, ..), &src.biases)?;
        Ok(())
    }

    fn copy_prefix_to(&self, dst: &mut QuantizedArray, len: i32) -> Result<(), Exception> {
        let packed = self.packed.index((.., .., ..len, ..));
        let scales = self.scales.index((.., .., ..len, ..));
        let biases = self.biases.index((.., .., ..len, ..));
        dst.packed.try_index_mut((.., .., ..len, ..), &packed)?;
        dst.scales.try_index_mut((.., .., ..len, ..), &scales)?;
        dst.biases.try_index_mut((.., .., ..len, ..), &biases)?;
        Ok(())
    }

    fn eval_all(&self) -> Result<(), Exception> {
        eval([&self.packed, &self.scales, &self.biases])?;
        Ok(())
    }

    #[cfg(test)]
    fn dequantize(&self, group_size: i32, bits: i32) -> Result<Array, Exception> {
        let packed_shape = self.packed.shape();
        let rows = packed_shape[0] * packed_shape[1] * packed_shape[2];
        let head_dim = self.scales.shape()[3] * group_size;
        let packed = self.packed.reshape(&[rows, packed_shape[3]])?;
        let scales = self.scales.reshape(&[rows, self.scales.shape()[3]])?;
        let biases = self.biases.reshape(&[rows, self.biases.shape()[3]])?;
        let dense = mlx_rs::ops::dequantize(&packed, &scales, &biases, group_size, bits)?;
        dense.reshape(&[packed_shape[0], packed_shape[1], packed_shape[2], head_dim])
    }
}

impl PlainKvCache {
    pub fn new(capacity: i32) -> Self {
        Self::new_with_promotion(capacity, None)
    }

    pub fn new_promotable(capacity: i32, group_size: i32, bits: i32, start: i32) -> Self {
        Self::new_with_promotion(
            capacity,
            Some(QuantizedKvPromotionConfig {
                group_size,
                bits,
                start,
            }),
        )
    }

    fn new_with_promotion(capacity: i32, promotion: Option<QuantizedKvPromotionConfig>) -> Self {
        PlainKvCache {
            keys: None,
            values: None,
            len: 0,
            capacity: capacity.max(1),
            pending_updates: 0,
            promotion,
            len_array: None,
            positions: None,
        }
    }

    fn sync_len_array(&mut self) -> Result<(), Exception> {
        if let Some(len_array) = self.len_array.as_mut() {
            *len_array = Array::from_int(self.len).as_dtype(Dtype::Int32)?;
            len_array.eval()?;
        }
        Ok(())
    }

    fn rebuild_positions(&mut self) -> Result<(), Exception> {
        if self.positions.is_some() {
            let positions = Array::arange::<_, i32>(0, self.capacity, None)?.reshape(&[
                1,
                1,
                1,
                self.capacity,
            ])?;
            positions.eval()?;
            self.positions = Some(positions);
        }
        Ok(())
    }

    fn prepare_compiled_forward_state(&mut self) -> Result<(), Exception> {
        if self.keys.is_none() || self.values.is_none() || self.promotion.is_some() {
            return Ok(());
        }
        if self.len_array.is_none() {
            self.len_array = Some(Array::from_int(self.len).as_dtype(Dtype::Int32)?);
        }
        if let Some(len_array) = self.len_array.as_ref() {
            len_array.eval()?;
        }
        Ok(())
    }

    fn prepare_compiled_decode_state(&mut self) -> Result<(), Exception> {
        self.prepare_compiled_forward_state()?;
        if self.positions.is_none() {
            self.positions = Some(Array::arange::<_, i32>(0, self.capacity, None)?.reshape(&[
                1,
                1,
                1,
                self.capacity,
            ])?);
        }
        if let Some(positions) = self.positions.as_ref() {
            positions.eval()?;
        }
        Ok(())
    }

    fn compiled_decode_ready(&self) -> bool {
        self.keys.is_some()
            && self.values.is_some()
            && self.promotion.is_none()
            && self.len_array.is_some()
            && self.positions.is_some()
    }

    fn finish_compiled_decode_step(&mut self, added: i32) -> Result<(), Exception> {
        let next_len = self.len + added;
        if next_len > self.capacity {
            return Err(Exception::custom(format!(
                "compiled KV cache overflow: need {next_len} slots, capacity {}",
                self.capacity
            )));
        }
        self.len += added;
        self.pending_updates = 0;
        Ok(())
    }

    fn ensure_storage(&mut self, new_keys: &Array, new_values: &Array) -> Result<(), Exception> {
        if self.keys.is_some() && self.values.is_some() {
            return Ok(());
        }

        let shape = new_keys.shape();
        let alloc_len = self.capacity.max(shape[2]);
        let alloc_shape = [shape[0], shape[1], alloc_len, shape[3]];
        let keys = mlx_rs::ops::zeros_dtype(&alloc_shape, new_keys.dtype())?;
        let values = mlx_rs::ops::zeros_dtype(&alloc_shape, new_values.dtype())?;
        eval([&keys, &values])?;

        self.capacity = alloc_len;
        self.keys = Some(keys);
        self.values = Some(values);
        self.rebuild_positions()?;
        self.sync_len_array()?;
        Ok(())
    }

    pub fn reserve(&mut self, required_capacity: i32) -> Result<(), Exception> {
        if required_capacity <= self.capacity {
            return Ok(());
        }
        self.capacity = required_capacity.max(1);
        if self.len == 0 {
            self.keys = None;
            self.values = None;
            return Ok(());
        }

        let old_keys = self
            .keys
            .as_ref()
            .ok_or_else(|| Exception::custom("KV cache keys missing during reserve"))?;
        let old_values = self
            .values
            .as_ref()
            .ok_or_else(|| Exception::custom("KV cache values missing during reserve"))?;
        let old_shape = old_keys.shape();
        let alloc_shape = [old_shape[0], old_shape[1], self.capacity, old_shape[3]];

        let mut keys = mlx_rs::ops::zeros_dtype(&alloc_shape, old_keys.dtype())?;
        let mut values = mlx_rs::ops::zeros_dtype(&alloc_shape, old_values.dtype())?;
        let active_keys = old_keys.index((.., .., ..self.len, ..));
        let active_values = old_values.index((.., .., ..self.len, ..));
        keys.try_index_mut((.., .., ..self.len, ..), &active_keys)?;
        values.try_index_mut((.., .., ..self.len, ..), &active_values)?;
        eval([&keys, &values])?;
        self.keys = Some(keys);
        self.values = Some(values);
        self.pending_updates = 0;
        self.rebuild_positions()?;
        self.sync_len_array()?;
        Ok(())
    }

    fn current_views(&self) -> Result<(Array, Array), Exception> {
        let keys = self
            .keys
            .as_ref()
            .ok_or_else(|| Exception::custom("KV cache keys not initialized"))?;
        let values = self
            .values
            .as_ref()
            .ok_or_else(|| Exception::custom("KV cache values not initialized"))?;
        Ok((
            keys.index((.., .., ..self.len, ..)),
            values.index((.., .., ..self.len, ..)),
        ))
    }

    pub fn update(
        &mut self,
        new_keys: &Array,
        new_values: &Array,
    ) -> Result<(Array, Array), Exception> {
        self.ensure_storage(new_keys, new_values)?;

        let new_seq = new_keys.shape()[2];
        let next_len = self.len + new_seq;
        if next_len > self.capacity {
            return Err(Exception::custom(format!(
                "KV cache overflow: need {next_len} slots, capacity {}",
                self.capacity
            )));
        }

        {
            let keys = self
                .keys
                .as_mut()
                .ok_or_else(|| Exception::custom("KV cache keys not allocated"))?;
            keys.try_index_mut((.., .., self.len..next_len, ..), new_keys)?;
        }
        {
            let values = self
                .values
                .as_mut()
                .ok_or_else(|| Exception::custom("KV cache values not allocated"))?;
            values.try_index_mut((.., .., self.len..next_len, ..), new_values)?;
        }

        self.len = next_len;
        self.pending_updates += 1;
        let should_materialize =
            new_seq > 1 || self.pending_updates >= KV_CACHE_MATERIALIZE_INTERVAL;
        if should_materialize {
            let keys = self
                .keys
                .as_ref()
                .ok_or_else(|| Exception::custom("KV cache keys missing during eval"))?;
            let values = self
                .values
                .as_ref()
                .ok_or_else(|| Exception::custom("KV cache values missing during eval"))?;
            eval([keys, values])?;
            self.pending_updates = 0;
        }
        self.sync_len_array()?;

        self.current_views()
    }

    pub fn len(&self) -> i32 {
        self.len
    }

    fn promotion_config(&self) -> Option<QuantizedKvPromotionConfig> {
        self.promotion
            .filter(|promotion| self.len >= promotion.start)
    }

    fn to_quantized(&self, group_size: i32, bits: i32) -> Result<QuantizedKvCache, Exception> {
        let mut quantized = QuantizedKvCache::new(self.capacity, group_size, bits);
        if self.len == 0 {
            return Ok(quantized);
        }

        let keys = self
            .keys
            .as_ref()
            .ok_or_else(|| Exception::custom("KV cache keys missing during promotion"))?
            .index((.., .., ..self.len, ..));
        let values = self
            .values
            .as_ref()
            .ok_or_else(|| Exception::custom("KV cache values missing during promotion"))?
            .index((.., .., ..self.len, ..));

        quantized.ensure_storage(&keys, &values)?;
        let q_keys = QuantizedArray::from_dense(&keys, group_size, bits)?;
        let q_values = QuantizedArray::from_dense(&values, group_size, bits)?;
        {
            let storage = quantized.keys.as_mut().ok_or_else(|| {
                Exception::custom("quantized KV cache keys not allocated during promotion")
            })?;
            storage.write_range(0, self.len, &q_keys)?;
        }
        {
            let storage = quantized.values.as_mut().ok_or_else(|| {
                Exception::custom("quantized KV cache values not allocated during promotion")
            })?;
            storage.write_range(0, self.len, &q_values)?;
        }
        quantized.len = self.len;
        if let Some(keys) = quantized.keys.as_ref() {
            keys.eval_all()?;
        }
        if let Some(values) = quantized.values.as_ref() {
            values.eval_all()?;
        }
        Ok(quantized)
    }
}

impl Updatable for PlainKvCache {
    fn updatable_states_len(&self) -> usize {
        usize::from(self.keys.is_some())
            + usize::from(self.values.is_some())
            + usize::from(self.len_array.is_some())
            + usize::from(self.positions.is_some())
    }

    fn updatable_states(&self) -> impl IntoIterator<Item = &Array> {
        let mut states = Vec::with_capacity(self.updatable_states_len());
        if let Some(keys) = self.keys.as_ref() {
            states.push(keys);
        }
        if let Some(values) = self.values.as_ref() {
            states.push(values);
        }
        if let Some(len_array) = self.len_array.as_ref() {
            states.push(len_array);
        }
        if let Some(positions) = self.positions.as_ref() {
            states.push(positions);
        }
        states
    }

    fn updatable_states_mut(&mut self) -> impl IntoIterator<Item = &mut Array> {
        let mut states = Vec::with_capacity(self.updatable_states_len());
        if let Some(keys) = self.keys.as_mut() {
            states.push(keys);
        }
        if let Some(values) = self.values.as_mut() {
            states.push(values);
        }
        if let Some(len_array) = self.len_array.as_mut() {
            states.push(len_array);
        }
        if let Some(positions) = self.positions.as_mut() {
            states.push(positions);
        }
        states
    }
}

impl QuantizedKvCache {
    pub fn new(capacity: i32, group_size: i32, bits: i32) -> Self {
        QuantizedKvCache {
            keys: None,
            values: None,
            len: 0,
            capacity: capacity.max(1),
            pending_updates: 0,
            group_size,
            bits,
        }
    }

    fn ensure_storage(&mut self, new_keys: &Array, new_values: &Array) -> Result<(), Exception> {
        if self.keys.is_some() && self.values.is_some() {
            return Ok(());
        }

        let shape = new_keys.shape();
        let alloc_len = self.capacity.max(shape[2]);
        let keys = QuantizedArray::zeros_storage(
            shape[0],
            shape[1],
            alloc_len,
            shape[3],
            new_keys.dtype(),
            self.group_size,
            self.bits,
        )?;
        let values = QuantizedArray::zeros_storage(
            shape[0],
            shape[1],
            alloc_len,
            new_values.shape()[3],
            new_values.dtype(),
            self.group_size,
            self.bits,
        )?;
        keys.eval_all()?;
        values.eval_all()?;

        self.capacity = alloc_len;
        self.keys = Some(keys);
        self.values = Some(values);
        Ok(())
    }

    pub fn reserve(&mut self, required_capacity: i32) -> Result<(), Exception> {
        if required_capacity <= self.capacity {
            return Ok(());
        }
        self.capacity = required_capacity.max(1);
        if self.len == 0 {
            self.keys = None;
            self.values = None;
            return Ok(());
        }

        let old_keys = self
            .keys
            .as_ref()
            .ok_or_else(|| Exception::custom("quantized KV cache keys missing during reserve"))?;
        let old_values = self
            .values
            .as_ref()
            .ok_or_else(|| Exception::custom("quantized KV cache values missing during reserve"))?;
        let mut keys = old_keys.with_capacity(self.capacity)?;
        let mut values = old_values.with_capacity(self.capacity)?;
        old_keys.copy_prefix_to(&mut keys, self.len)?;
        old_values.copy_prefix_to(&mut values, self.len)?;
        keys.eval_all()?;
        values.eval_all()?;
        self.keys = Some(keys);
        self.values = Some(values);
        self.pending_updates = 0;
        Ok(())
    }

    fn current_views(&self) -> Result<QuantizedKvView, Exception> {
        let keys = self
            .keys
            .as_ref()
            .ok_or_else(|| Exception::custom("quantized KV cache keys not initialized"))?;
        let values = self
            .values
            .as_ref()
            .ok_or_else(|| Exception::custom("quantized KV cache values not initialized"))?;
        Ok(QuantizedKvView {
            keys: keys.view_prefix(self.len),
            values: values.view_prefix(self.len),
            group_size: self.group_size,
            bits: self.bits,
        })
    }

    pub fn update(
        &mut self,
        new_keys: &Array,
        new_values: &Array,
    ) -> Result<QuantizedKvView, Exception> {
        self.ensure_storage(new_keys, new_values)?;

        let new_seq = new_keys.shape()[2];
        let next_len = self.len + new_seq;
        if next_len > self.capacity {
            return Err(Exception::custom(format!(
                "quantized KV cache overflow: need {next_len} slots, capacity {}",
                self.capacity
            )));
        }

        let q_keys = QuantizedArray::from_dense(new_keys, self.group_size, self.bits)?;
        let q_values = QuantizedArray::from_dense(new_values, self.group_size, self.bits)?;

        {
            let keys = self
                .keys
                .as_mut()
                .ok_or_else(|| Exception::custom("quantized KV cache keys not allocated"))?;
            keys.write_range(self.len, next_len, &q_keys)?;
        }
        {
            let values = self
                .values
                .as_mut()
                .ok_or_else(|| Exception::custom("quantized KV cache values not allocated"))?;
            values.write_range(self.len, next_len, &q_values)?;
        }

        self.len = next_len;
        self.pending_updates += 1;
        let should_materialize =
            new_seq > 1 || self.pending_updates >= KV_CACHE_MATERIALIZE_INTERVAL;
        if should_materialize {
            let keys = self
                .keys
                .as_ref()
                .ok_or_else(|| Exception::custom("quantized KV cache keys missing during eval"))?;
            let values = self.values.as_ref().ok_or_else(|| {
                Exception::custom("quantized KV cache values missing during eval")
            })?;
            keys.eval_all()?;
            values.eval_all()?;
            self.pending_updates = 0;
        }

        self.current_views()
    }

    pub fn len(&self) -> i32 {
        self.len
    }
}

impl KvCache {
    pub fn new(capacity: i32) -> Self {
        KvCache::Plain(PlainKvCache::new(capacity))
    }

    pub fn new_promotable(capacity: i32, group_size: i32, bits: i32, start: i32) -> Self {
        KvCache::Plain(PlainKvCache::new_promotable(
            capacity, group_size, bits, start,
        ))
    }

    pub fn new_quantized(capacity: i32, group_size: i32, bits: i32) -> Self {
        KvCache::Quantized(QuantizedKvCache::new(capacity, group_size, bits))
    }

    pub fn reserve(&mut self, required_capacity: i32) -> Result<(), Exception> {
        match self {
            KvCache::Plain(cache) => cache.reserve(required_capacity),
            KvCache::Quantized(cache) => cache.reserve(required_capacity),
        }
    }

    pub fn update(
        &mut self,
        new_keys: &Array,
        new_values: &Array,
    ) -> Result<KvCacheView, Exception> {
        match self {
            KvCache::Plain(cache) => {
                let (keys, values, promoted) = {
                    let (keys, values) = cache.update(new_keys, new_values)?;
                    let promoted = if let Some(promotion) = cache.promotion_config() {
                        Some(cache.to_quantized(promotion.group_size, promotion.bits)?)
                    } else {
                        None
                    };
                    (keys, values, promoted)
                };

                if let Some(quantized) = promoted {
                    let view = quantized.current_views()?;
                    *self = KvCache::Quantized(quantized);
                    Ok(KvCacheView::Quantized(view))
                } else {
                    Ok(KvCacheView::Plain { keys, values })
                }
            }
            KvCache::Quantized(cache) => {
                Ok(KvCacheView::Quantized(cache.update(new_keys, new_values)?))
            }
        }
    }

    pub fn len(&self) -> i32 {
        match self {
            KvCache::Plain(cache) => cache.len(),
            KvCache::Quantized(cache) => cache.len(),
        }
    }

    #[cfg(test)]
    fn is_quantized(&self) -> bool {
        matches!(self, KvCache::Quantized(_))
    }
}

impl GdnCache {
    pub fn new() -> Self {
        GdnCache {
            state: None,
            conv_buf: None,
            conv_pos: -1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PrefillState {
    pub caches: Vec<LayerCache>,
    pub last_logits: Array,
    pub prompt_len: usize,
}

fn sample_next_token(logits: &Array, temperature: f32) -> Result<Array, Exception> {
    let token = if temperature <= 0.0 || temperature < 1e-6 {
        mlx_rs::ops::indexing::argmax_axis(logits, -1, false)?
    } else {
        let scaled = logits.multiply(array!(1.0 / temperature))?;
        mlx_rs::random::categorical(&scaled, None, None, None)?
    };
    token.as_dtype(mlx_rs::Dtype::Int32)
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
    model.set_lora_training_mode(true);
    let result = (|| -> Result<(f32, f32), anyhow::Error> {
        let loss_fn = |model: &mut MlxLoraModel,
                       (toks, tgts): (&Array, &Array)|
         -> Result<Array, Exception> {
            let logits = model.forward_logits(toks)?;
            cross_entropy_loss(&logits, tgts)
        };

        let mut vg = nn::value_and_grad(loss_fn);

        let (loss, grads) =
            vg(model, (tokens, targets)).map_err(|e| anyhow::anyhow!("value_and_grad: {e}"))?;

        let (clipped, total_norm) = mlx_rs::optimizers::clip_grad_norm(&grads, grad_clip)
            .map_err(|e| anyhow::anyhow!("clip_grad_norm: {e}"))?;

        // Skip optimizer update if gradient is NaN (prevents permanent weight corruption).
        if total_norm.is_finite() {
            let owned_grads: HashMap<Rc<str>, Array> = clipped
                .into_iter()
                .map(|(k, v)| (k, v.into_owned()))
                .collect();

            optimizer
                .update(model, &owned_grads)
                .map_err(|e| anyhow::anyhow!("optimizer update: {e}"))?;
            model.mark_adapters_active();
        } else {
            tracing::warn!("skipped NaN gradient update");
        }

        // Evaluate loss and model parameters to materialize the updated graph.
        let params = model.parameters().flatten();
        let mut eval_targets: Vec<&Array> = vec![&loss];
        eval_targets.extend(params.values());
        mlx_rs::transforms::eval(eval_targets.into_iter())
            .map_err(|e| anyhow::anyhow!("eval: {e}"))?;

        let loss_val: f32 = loss.as_slice::<f32>()[0];
        Ok((loss_val, total_norm))
    })();
    model.set_lora_training_mode(false);
    result
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
    train_loop_with_callback(
        model,
        tokens_list,
        targets_list,
        config,
        max_steps,
        early_stop_loss,
        patience,
        None,
    )
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
            model,
            &mut optimizer,
            &tokens_list[idx],
            &targets_list[idx],
            config.grad_clip,
        )?;

        let ms = t0.elapsed().as_millis();
        tracing::debug!(
            step,
            loss = format!("{loss:.4}"),
            grad_norm = format!("{grad_norm:.4}"),
            ms,
            "train step"
        );
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
            tracing::info!(
                loss = format!("{loss:.4}"),
                threshold = format!("{early_stop_loss:.4}"),
                "early stop: loss below threshold"
            );
            break;
        }

        if steps_without_improvement >= patience {
            tracing::info!(patience, "early stop: no improvement");
            break;
        }
    }

    Ok(losses)
}

// ---------------------------------------------------------------------------
// Adapter export (mlx-lm compatible safetensors)
// ---------------------------------------------------------------------------

/// LoRA target keys as they appear in mlx-lm adapter_config.json.
const LORA_KEYS: &[&str] = &[
    "self_attn.q_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.down_proj",
];

/// Export trained LoRA adapters in mlx-lm format.
///
/// Produces two files in `output_dir`:
///   - `adapters.safetensors` — trainable weights with mlx-lm key paths
///   - `adapter_config.json` — LoRA config for `mlx_lm.server --adapter-path`
///
/// The weight keys are prefixed with `weight_prefix` so mlx-lm's
/// `model.load_weights()` can match them to the right layers.
pub fn export_adapters(
    model: &MlxLoraModel,
    config: &LoraConfig,
    model_config: &ModelConfig,
    output_dir: &Path,
) -> Result<usize, anyhow::Error> {
    use mlx_rs::module::ModuleParameters;
    use std::collections::HashMap;

    std::fs::create_dir_all(output_dir)?;

    // Get trainable params: keys like "layers.0.self_attn.q_proj.lora_a.weight"
    let trainable = model.trainable_parameters().flatten();

    // Eval all arrays to materialize lazy computation graphs
    mlx_rs::transforms::eval(trainable.values().copied())
        .map_err(|e| anyhow::anyhow!("eval trainable params: {e}"))?;

    // Prefix keys with model weight_prefix for mlx-lm compatibility
    let pfx = model_config.weight_prefix;
    let prefixed: HashMap<String, &Array> = trainable
        .iter()
        .map(|(k, v)| (format!("{pfx}.{k}"), *v))
        .collect();

    let n_params = prefixed.len();
    let safetensors_path = output_dir.join("adapters.safetensors");
    Array::save_safetensors(
        prefixed.iter().map(|(k, v)| (k.as_str(), *v)),
        None,
        &safetensors_path,
    )
    .map_err(|e| anyhow::anyhow!("save safetensors: {e}"))?;

    // Count LoRA layers (layers that have at least one trainable param)
    let lora_layers = (0..model_config.n_layers)
        .filter(|i| !model_config.is_linear_attn_layer(*i))
        .count();

    // Write adapter_config.json
    let adapter_config = serde_json::json!({
        "alpha": config.alpha,
        "dropout": 0.0,
        "keys": LORA_KEYS,
        "lora_layers": lora_layers,
        "rank": config.rank,
        "num_layers": model_config.n_layers,
    });
    let config_path = output_dir.join("adapter_config.json");
    std::fs::write(&config_path, serde_json::to_string_pretty(&adapter_config)?)?;

    tracing::info!(tensors = n_params, path = %output_dir.display(), "exported adapter tensors");

    Ok(n_params)
}

/// Export ANE-trained LoRA weights directly in mlx-lm adapter format.
///
/// Unlike [`export_adapters`], this does not require an in-process MLX model.
/// It writes the same `adapters.safetensors` + `adapter_config.json` pair from
/// the ANE LoRA tensors so managed mlx-lm inference can reload them live.
#[cfg(feature = "ane")]
pub fn export_ane_adapters(
    lora: &crate::agent::ane_lora::LoraModel,
    model_config: &ModelConfig,
    linear_attn_indices: Option<&[usize]>,
    output_dir: &Path,
) -> Result<usize, anyhow::Error> {
    std::fs::create_dir_all(output_dir)?;

    let linear = linear_attn_indices.unwrap_or(&model_config.linear_attn_indices);
    let pfx = model_config.weight_prefix;
    let mut tensors: Vec<(String, Array)> = Vec::new();

    for (layer_idx, layer) in lora.layers.iter().enumerate() {
        let is_linear = linear.contains(&layer_idx);

        let mut push_adapter = |path: &str, adapter: &crate::agent::ane_lora::LoraAdapter| {
            tensors.push((
                format!("{pfx}.layers.{layer_idx}.{path}.lora_a.weight"),
                Array::from_slice(&adapter.a, &[adapter.rank as i32, adapter.d_in as i32]),
            ));
            tensors.push((
                format!("{pfx}.layers.{layer_idx}.{path}.lora_b.weight"),
                Array::from_slice(&adapter.b, &[adapter.d_out as i32, adapter.rank as i32]),
            ));
        };

        if !is_linear {
            if let Some(adapter) = &layer.wq {
                push_adapter("self_attn.q_proj", adapter);
            }
            if let Some(adapter) = &layer.wv {
                push_adapter("self_attn.v_proj", adapter);
            }
            if let Some(adapter) = &layer.wo {
                push_adapter("self_attn.o_proj", adapter);
            }
        }

        if let Some(adapter) = &layer.w2 {
            push_adapter("mlp.down_proj", adapter);
        }
    }

    let n_params = tensors.len();
    let safetensors_path = output_dir.join("adapters.safetensors");
    Array::save_safetensors(
        tensors.iter().map(|(k, v)| (k.as_str(), v)),
        None,
        &safetensors_path,
    )
    .map_err(|e| anyhow::anyhow!("save safetensors: {e}"))?;

    let lora_layers = (0..model_config.n_layers)
        .filter(|i| !model_config.is_linear_attn_layer(*i))
        .count();
    let adapter_config = serde_json::json!({
        "alpha": lora.config.alpha,
        "dropout": 0.0,
        "keys": LORA_KEYS,
        "lora_layers": lora_layers,
        "rank": lora.config.rank,
        "num_layers": model_config.n_layers,
    });
    let config_path = output_dir.join("adapter_config.json");
    std::fs::write(&config_path, serde_json::to_string_pretty(&adapter_config)?)?;

    tracing::info!(tensors = n_params, path = %output_dir.display(), "exported ANE adapter tensors");

    Ok(n_params)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_config_json_qwen3() {
        // Auto-detect from Qwen3-4B config.json if available
        let model_dir = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/lmstudio-community/Qwen3-4B-Thinking-2507-MLX-4bit");
        if !model_dir.exists() {
            eprintln!("Qwen3-4B not found, skipping from_config_json test");
            return;
        }
        let cfg = ModelConfig::from_config_json(&model_dir).expect("should parse config.json");
        assert_eq!(cfg.dim, 2560);
        assert_eq!(cfg.hidden_dim, 9728);
        assert_eq!(cfg.n_heads, 32);
        assert_eq!(cfg.n_kv_heads, 8);
        assert_eq!(cfg.n_layers, 36);
        assert_eq!(cfg.vocab_size, 151936);
        assert_eq!(cfg.bits, 4);
        assert!(
            cfg.linear_attn_indices.is_empty(),
            "Qwen3 has no linear attention"
        );
        assert_eq!(cfg.weight_prefix, "model");
        assert!(cfg.thinking_model, "name contains 'Thinking'");
    }

    #[test]
    fn test_from_config_json_qwen3_5() {
        let model_dir = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-2B-MLX-8bit");
        if !model_dir.exists() {
            eprintln!("Qwen3.5-2B not found, skipping from_config_json test");
            return;
        }
        let cfg = ModelConfig::from_config_json(&model_dir).expect("should parse config.json");
        assert_eq!(cfg.dim, 2048);
        assert_eq!(cfg.n_layers, 24);
        assert_eq!(cfg.bits, 8);
        assert_eq!(cfg.weight_prefix, "language_model.model");
        assert!(cfg.attn_output_gate);
        assert_eq!(
            cfg.linear_attn_indices.len(),
            18,
            "18 of 24 layers are linear"
        );
        assert_eq!(cfg.linear_n_heads, 16);
        assert_eq!(cfg.conv_kernel_size, 4);
    }

    #[test]
    fn test_from_config_json_qwen3_5_moe() {
        // MoE models (e.g. Qwen3.5-35B-A3B) use moe_intermediate_size instead of intermediate_size
        let model_dir = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-35B-A3B-4bit");
        if !model_dir.exists() {
            eprintln!("Qwen3.5-35B-A3B not found, skipping MoE config test");
            return;
        }
        let cfg = ModelConfig::from_config_json(&model_dir).expect("should parse MoE config.json");
        assert_eq!(cfg.dim, 2048);
        assert_eq!(cfg.hidden_dim, 512); // moe_intermediate_size, not intermediate_size
        assert_eq!(cfg.n_layers, 40);
        assert_eq!(cfg.n_heads, 16);
        assert_eq!(cfg.n_kv_heads, 2);
        assert_eq!(cfg.vocab_size, 248320);
        assert_eq!(cfg.bits, 4);
        assert_eq!(cfg.weight_prefix, "language_model.model");
        assert!(cfg.attn_output_gate);
        // 30 of 40 layers are linear attention
        assert_eq!(cfg.linear_attn_indices.len(), 30);
    }

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
    fn test_kv_cache_preallocated_append() {
        let mut cache = KvCache::new(4);
        let keys1 = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 2, 1]);
        let values1 = Array::from_slice(&[10.0f32, 20.0], &[1, 1, 2, 1]);
        let (k1, v1) = match cache.update(&keys1, &values1).unwrap() {
            KvCacheView::Plain { keys, values } => (keys, values),
            KvCacheView::Quantized(_) => panic!("expected plain KV cache"),
        };
        k1.eval().unwrap();
        v1.eval().unwrap();
        assert_eq!(cache.len(), 2);
        assert_eq!(k1.shape(), &[1, 1, 2, 1]);
        assert_eq!(v1.shape(), &[1, 1, 2, 1]);
        assert_eq!(k1.as_slice::<f32>(), &[1.0, 2.0]);
        assert_eq!(v1.as_slice::<f32>(), &[10.0, 20.0]);

        let keys2 = Array::from_slice(&[3.0f32], &[1, 1, 1, 1]);
        let values2 = Array::from_slice(&[30.0f32], &[1, 1, 1, 1]);
        let (k2, v2) = match cache.update(&keys2, &values2).unwrap() {
            KvCacheView::Plain { keys, values } => (keys, values),
            KvCacheView::Quantized(_) => panic!("expected plain KV cache"),
        };
        k2.eval().unwrap();
        v2.eval().unwrap();
        assert_eq!(cache.len(), 3);
        assert_eq!(k2.shape(), &[1, 1, 3, 1]);
        assert_eq!(v2.shape(), &[1, 1, 3, 1]);
        assert_eq!(k2.as_slice::<f32>(), &[1.0, 2.0, 3.0]);
        assert_eq!(v2.as_slice::<f32>(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_quantized_kv_cache_preallocated_append_round_trip() {
        let mut cache = KvCache::new_quantized(4, 64, 8);
        let keys1: Vec<f32> = (0..128).map(|i| i as f32 / 32.0).collect();
        let values1: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 24.0).collect();
        let keys1 = Array::from_slice(&keys1, &[1, 1, 2, 64]);
        let values1 = Array::from_slice(&values1, &[1, 1, 2, 64]);

        let view = match cache.update(&keys1, &values1).unwrap() {
            KvCacheView::Quantized(view) => view,
            KvCacheView::Plain { .. } => panic!("expected quantized KV cache"),
        };

        let dense_keys = view.keys.dequantize(view.group_size, view.bits).unwrap();
        let dense_values = view.values.dequantize(view.group_size, view.bits).unwrap();
        assert_eq!(dense_keys.shape(), &[1, 1, 2, 64]);
        assert_eq!(dense_values.shape(), &[1, 1, 2, 64]);

        let key_diff = dense_keys
            .subtract(&keys1)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        let value_diff = dense_values
            .subtract(&values1)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(
            key_diff < 0.05,
            "quantized key max diff too large: {key_diff}"
        );
        assert!(
            value_diff < 0.05,
            "quantized value max diff too large: {value_diff}"
        );

        let keys2: Vec<f32> = (0..64).map(|i| (i as f32 + 7.0) / 19.0).collect();
        let values2: Vec<f32> = (0..64).map(|i| (32.0 - i as f32) / 11.0).collect();
        let keys2 = Array::from_slice(&keys2, &[1, 1, 1, 64]);
        let values2 = Array::from_slice(&values2, &[1, 1, 1, 64]);
        let view = match cache.update(&keys2, &values2).unwrap() {
            KvCacheView::Quantized(view) => view,
            KvCacheView::Plain { .. } => panic!("expected quantized KV cache"),
        };
        let dense_keys = view.keys.dequantize(view.group_size, view.bits).unwrap();
        let dense_values = view.values.dequantize(view.group_size, view.bits).unwrap();
        assert_eq!(cache.len(), 3);
        assert_eq!(dense_keys.shape(), &[1, 1, 3, 64]);
        assert_eq!(dense_values.shape(), &[1, 1, 3, 64]);
    }

    #[test]
    fn test_kv_cache_promotes_to_quantized_after_threshold() {
        let mut cache = KvCache::new_promotable(6, 64, 8, 3);

        let keys1: Vec<f32> = (0..128).map(|i| i as f32 / 41.0).collect();
        let values1: Vec<f32> = (0..128).map(|i| (i as f32 - 23.0) / 17.0).collect();
        let keys1 = Array::from_slice(&keys1, &[1, 1, 2, 64]);
        let values1 = Array::from_slice(&values1, &[1, 1, 2, 64]);
        let view1 = cache.update(&keys1, &values1).unwrap();
        assert!(matches!(view1, KvCacheView::Plain { .. }));
        assert!(!cache.is_quantized());

        let keys2: Vec<f32> = (0..64).map(|i| (i as f32 + 5.0) / 13.0).collect();
        let values2: Vec<f32> = (0..64).map(|i| (31.0 - i as f32) / 9.0).collect();
        let keys2 = Array::from_slice(&keys2, &[1, 1, 1, 64]);
        let values2 = Array::from_slice(&values2, &[1, 1, 1, 64]);
        let view2 = cache.update(&keys2, &values2).unwrap();
        let view2 = match view2 {
            KvCacheView::Quantized(view) => view,
            KvCacheView::Plain { .. } => panic!("expected promotion to quantized cache"),
        };
        assert!(cache.is_quantized());
        assert_eq!(cache.len(), 3);

        let dense_keys = view2.keys.dequantize(view2.group_size, view2.bits).unwrap();
        let dense_values = view2
            .values
            .dequantize(view2.group_size, view2.bits)
            .unwrap();
        assert_eq!(dense_keys.shape(), &[1, 1, 3, 64]);
        assert_eq!(dense_values.shape(), &[1, 1, 3, 64]);
    }

    #[test]
    fn test_compiled_plain_kv_decode_matches_dense_step() {
        let mut regular = PlainKvCache::new(6);
        let mut compiled = PlainKvCache::new(6);

        let init_keys: Vec<f32> = (0..128).map(|i| (i as f32 - 32.0) / 29.0).collect();
        let init_values: Vec<f32> = (0..128).map(|i| (i as f32 + 11.0) / 23.0).collect();
        let init_keys = Array::from_slice(&init_keys, &[1, 1, 2, 64]);
        let init_values = Array::from_slice(&init_values, &[1, 1, 2, 64]);
        regular.update(&init_keys, &init_values).unwrap();
        compiled.update(&init_keys, &init_values).unwrap();
        compiled.prepare_compiled_decode_state().unwrap();

        let queries: Vec<f32> = (0..64).map(|i| ((i % 17) as f32 - 8.0) / 13.0).collect();
        let new_keys: Vec<f32> = (0..64).map(|i| ((i % 19) as f32 - 9.0) / 11.0).collect();
        let new_values: Vec<f32> = (0..64).map(|i| ((i % 23) as f32 - 11.0) / 7.0).collect();
        let queries = Array::from_slice(&queries, &[1, 1, 1, 64]);
        let new_keys = Array::from_slice(&new_keys, &[1, 1, 1, 64]);
        let new_values = Array::from_slice(&new_values, &[1, 1, 1, 64]);
        let scale = 1.0 / (64.0f32).sqrt();

        let offset = regular.len();
        let (expected_keys, expected_values) = regular.update(&new_keys, &new_values).unwrap();
        let expected = dense_scaled_dot_product_attention(
            &queries,
            &expected_keys,
            &expected_values,
            scale,
            None,
            offset,
            1,
        )
        .unwrap();

        let mut compiled_step = make_compiled_plain_kv_decode(scale);
        let actual = compiled_step(&mut compiled, (&queries, &new_keys, &new_values)).unwrap();
        compiled.finish_compiled_decode_step(1).unwrap();

        let max_diff = actual
            .subtract(&expected)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(
            max_diff < 1e-4,
            "compiled decode drift too large: {max_diff}"
        );

        let (compiled_keys, compiled_values) = compiled.current_views().unwrap();
        let key_diff = compiled_keys
            .subtract(&expected_keys)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        let value_diff = compiled_values
            .subtract(&expected_values)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(key_diff < 1e-6, "compiled keys diverged: {key_diff}");
        assert!(value_diff < 1e-6, "compiled values diverged: {value_diff}");
        assert_eq!(compiled.len(), 3);
    }

    #[test]
    fn test_compiled_plain_kv_decode_matches_dense_multiple_steps() {
        let mut regular = PlainKvCache::new(8);
        let mut compiled = PlainKvCache::new(8);

        let init_keys: Vec<f32> = (0..128).map(|i| (i as f32 - 41.0) / 17.0).collect();
        let init_values: Vec<f32> = (0..128).map(|i| (i as f32 + 7.0) / 13.0).collect();
        let init_keys = Array::from_slice(&init_keys, &[1, 1, 2, 64]);
        let init_values = Array::from_slice(&init_values, &[1, 1, 2, 64]);
        regular.update(&init_keys, &init_values).unwrap();
        compiled.update(&init_keys, &init_values).unwrap();
        compiled.prepare_compiled_decode_state().unwrap();

        let scale = 1.0 / (64.0f32).sqrt();
        let mut compiled_step = make_compiled_plain_kv_decode(scale);

        for step in 0..3 {
            let queries: Vec<f32> = (0..64)
                .map(|i| (((i + step * 3) % 17) as f32 - 8.0) / 9.0)
                .collect();
            let new_keys: Vec<f32> = (0..64)
                .map(|i| (((i + step * 5) % 19) as f32 - 9.0) / 7.0)
                .collect();
            let new_values: Vec<f32> = (0..64)
                .map(|i| (((i + step * 7) % 23) as f32 - 11.0) / 5.0)
                .collect();
            let queries = Array::from_slice(&queries, &[1, 1, 1, 64]);
            let new_keys = Array::from_slice(&new_keys, &[1, 1, 1, 64]);
            let new_values = Array::from_slice(&new_values, &[1, 1, 1, 64]);

            let offset = regular.len();
            let (expected_keys, expected_values) = regular.update(&new_keys, &new_values).unwrap();
            let expected = dense_scaled_dot_product_attention(
                &queries,
                &expected_keys,
                &expected_values,
                scale,
                None,
                offset,
                1,
            )
            .unwrap();

            let actual = compiled_step(&mut compiled, (&queries, &new_keys, &new_values)).unwrap();
            compiled.finish_compiled_decode_step(1).unwrap();

            let max_diff = actual
                .subtract(&expected)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>();
            assert!(
                max_diff < 1e-4,
                "compiled multi-step decode drift too large at step {step}: {max_diff}"
            );

            let (compiled_keys, compiled_values) = compiled.current_views().unwrap();
            let key_diff = compiled_keys
                .subtract(&expected_keys)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>();
            let value_diff = compiled_values
                .subtract(&expected_values)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>();
            assert!(
                key_diff < 1e-6,
                "compiled keys diverged at step {step}: {key_diff}"
            );
            assert!(
                value_diff < 1e-6,
                "compiled values diverged at step {step}: {value_diff}"
            );
        }

        assert_eq!(compiled.len(), regular.len());
    }

    #[test]
    fn test_quantized_attention_matches_dense_attention() {
        let scale = 1.0 / (64.0f32).sqrt();
        let queries: Vec<f32> = (0..512).map(|i| ((i % 37) as f32 - 18.0) / 23.0).collect();
        let keys: Vec<f32> = (0..384).map(|i| ((i % 29) as f32 - 14.0) / 19.0).collect();
        let values: Vec<f32> = (0..384).map(|i| ((i % 31) as f32 - 15.0) / 17.0).collect();
        let queries = Array::from_slice(&queries, &[1, 4, 2, 64]);
        let keys = Array::from_slice(&keys, &[1, 2, 3, 64]);
        let values = Array::from_slice(&values, &[1, 2, 3, 64]);
        let q_keys = QuantizedArray::from_dense(&keys, 64, 8).unwrap();
        let q_values = QuantizedArray::from_dense(&values, 64, 8).unwrap();
        let zero_mask = mlx_rs::ops::zeros::<f32>(&[1, 1, 2, 3]).unwrap();

        let dense = dense_scaled_dot_product_attention(
            &queries,
            &keys,
            &values,
            scale,
            Some(&zero_mask),
            0,
            2,
        )
        .unwrap();
        let quantized = quantized_scaled_dot_product_attention(
            &queries,
            &q_keys,
            &q_values,
            scale,
            Some(&zero_mask),
            64,
            8,
        )
        .unwrap();
        assert_eq!(dense.shape(), quantized.shape());

        let max_diff = quantized
            .subtract(&dense)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(
            max_diff < 0.08,
            "quantized attention drift too large: {max_diff}"
        );
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
        let diff = lora_out
            .subtract(&base_out)
            .unwrap()
            .abs()
            .unwrap()
            .sum(None)
            .unwrap();
        let diff_val: f32 = diff.as_slice::<f32>()[0];
        assert!(
            diff_val < 1e-6,
            "LoRA should not change output when B=0, got diff={diff_val}"
        );
    }

    #[test]
    fn test_cross_entropy_loss() {
        let logits = Array::from_slice(&[0.0f32, 0.0, 10.0, 0.0, 10.0, 0.0, 0.0, 0.0], &[1, 2, 4]);
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
        let logits = mlx_rs::random::normal::<f32>(&[1, seq_len, vocab], None, None, None).unwrap();
        // Random target indices in [0, vocab)
        let targets = mlx_rs::random::randint::<_, i32>(0, vocab, &[1, seq_len], None).unwrap();

        // Forward: loss must be finite
        let loss = cross_entropy_loss(&logits, &targets).unwrap();
        mlx_rs::transforms::eval(std::iter::once(&loss)).unwrap();
        let loss_val: f32 = loss.as_slice::<f32>()[0];
        assert!(
            loss_val.is_finite(),
            "loss should be finite, got {loss_val}"
        );
        eprintln!("large vocab CE loss = {loss_val:.4}");

        // Backward: gradient via grad() must be finite
        let mut grad_fn = mlx_rs::transforms::grad(|logits: &Array| -> Result<Array, Exception> {
            cross_entropy_loss(logits, &targets)
        });
        let grad = grad_fn(&logits).unwrap();
        mlx_rs::transforms::eval(std::iter::once(&grad)).unwrap();
        let grad_sum: f32 = grad.sum(None).unwrap().as_slice::<f32>()[0];
        assert!(
            grad_sum.is_finite(),
            "gradient should be finite, got {grad_sum}"
        );
        eprintln!("large vocab CE grad_sum = {grad_sum:.6}");
    }

    #[test]
    fn test_trainable_param_count() {
        let ql = QuantizedLinear::new(128, 64).unwrap();
        let lora = LoraLinear::new(ql, 4, 1.0).unwrap();
        let trainable = lora.trainable_parameters().flatten();
        assert_eq!(
            trainable.len(),
            2,
            "expected 2 trainable params, got {:?}",
            trainable.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_trainable_param_keys_match_mlx_lm() {
        // Verify that flattened trainable param keys follow mlx-lm naming convention.
        let ql = QuantizedLinear::new(128, 64).unwrap();
        let lora = LoraLinear::new(ql, 4, 1.0).unwrap();
        let trainable = lora.trainable_parameters().flatten();
        let keys: Vec<String> = trainable.keys().map(|k| k.to_string()).collect();
        assert!(
            keys.contains(&"lora_a.weight".to_string()),
            "expected lora_a.weight, got {keys:?}"
        );
        assert!(
            keys.contains(&"lora_b.weight".to_string()),
            "expected lora_b.weight, got {keys:?}"
        );
    }

    #[test]
    fn test_adapter_config_json_format() {
        let config = LoraConfig::default();
        let model_config = ModelConfig::qwen3_1_7b();

        let adapter_config = serde_json::json!({
            "alpha": config.alpha,
            "dropout": 0.0,
            "keys": LORA_KEYS,
            "lora_layers": model_config.n_layers,
            "rank": config.rank,
            "num_layers": model_config.n_layers,
        });

        let json_str = serde_json::to_string_pretty(&adapter_config).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed["alpha"], 32.0);
        assert_eq!(parsed["rank"], 32);
        assert_eq!(parsed["lora_layers"], 28);
        let keys: Vec<String> = parsed["keys"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert!(keys.contains(&"self_attn.q_proj".to_string()));
        assert!(keys.contains(&"mlp.down_proj".to_string()));
    }

    #[test]
    fn test_lora_linear_grad() {
        let ql = QuantizedLinear::new(128, 64).unwrap();
        let mut lora = LoraLinear::new(ql, 4, 1.0).unwrap();
        lora.training_mode(true);
        let x = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[1, 128], None).unwrap();

        let loss_fn = |model: &mut LoraLinear, x: &Array| -> Result<Array, Exception> {
            model.forward(x)?.sum(None)
        };

        let mut vg = nn::value_and_grad(loss_fn);
        let (val, grads) = vg(&mut lora, &x).unwrap();

        assert!(
            val.as_slice::<f32>()[0].is_finite(),
            "loss should be finite"
        );
        assert!(
            grads.len() >= 2,
            "should have grads for LoRA A and B, got {}",
            grads.len()
        );
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

        let mut model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg).expect("model load failed");

        let trainable = model.trainable_parameters().flatten();
        let total_trainable: usize = trainable
            .values()
            .map(|a| a.shape().iter().product::<i32>() as usize)
            .sum();
        eprintln!("trainable parameters: {total_trainable}");
        assert_eq!(trainable.len(), 224, "expected 224 trainable params");

        let tokens = Array::from_slice(&[100i32, 101, 102, 103], &[1, 4]);
        let targets = Array::from_slice(&[101i32, 102, 103, 104], &[1, 4]);

        let losses = train_loop(&mut model, &[tokens], &[targets], &lora_cfg, 3, 0.5, 10)
            .expect("training failed");

        assert!(losses.len() >= 2);
        let first = losses[0];
        let last = losses[losses.len() - 1];
        eprintln!("loss: {first:.4} -> {last:.4}");
        assert!(first.is_finite());
        assert!(
            last < first,
            "loss should decrease: {first:.4} -> {last:.4}"
        );
    }

    #[test]
    fn test_e2e_export_adapters_qwen3() {
        let model_dir = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/lmstudio-community/Qwen3-1.7B-MLX-8bit");
        if !model_dir.exists() {
            eprintln!("Qwen3-1.7B-MLX-8bit not found, skipping export test");
            return;
        }

        let cfg = ModelConfig::qwen3_1_7b();
        let lora_cfg = LoraConfig::default();

        let model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg).expect("model load failed");

        let tmpdir = tempfile::tempdir().unwrap();
        let n = export_adapters(&model, &lora_cfg, &cfg, tmpdir.path()).expect("export failed");

        assert!(n > 0, "should export at least 1 tensor");
        assert_eq!(n, 224, "Qwen3-1.7B has 224 trainable tensors");

        // Verify safetensors file exists and can be loaded back
        let st_path = tmpdir.path().join("adapters.safetensors");
        assert!(st_path.exists(), "adapters.safetensors not found");
        let loaded = Array::load_safetensors(&st_path).expect("load back failed");
        assert_eq!(loaded.len(), n);

        // Verify keys have the correct prefix
        for key in loaded.keys() {
            assert!(
                key.starts_with("model.layers."),
                "key should start with 'model.layers.', got: {key}"
            );
            assert!(
                key.ends_with(".weight"),
                "key should end with '.weight', got: {key}"
            );
        }

        // Verify adapter_config.json
        let config_path = tmpdir.path().join("adapter_config.json");
        assert!(config_path.exists(), "adapter_config.json not found");
        let config_json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
        assert_eq!(config_json["rank"], 32);
        assert_eq!(config_json["alpha"], 32.0);
        assert_eq!(config_json["lora_layers"], 28);

        eprintln!(
            "exported {} adapter tensors to {}",
            n,
            tmpdir.path().display()
        );
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
        let mut model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg).expect("model load failed");

        let trainable = model.trainable_parameters().flatten();
        let total_trainable: usize = trainable
            .values()
            .map(|a| a.shape().iter().product::<i32>() as usize)
            .sum();
        eprintln!(
            "trainable parameters: {total_trainable} ({} param tensors)",
            trainable.len()
        );

        // 6 full-attn layers × 3 LoRA targets (q, v, o) × 2 (A+B) = 36
        // 24 layers × 1 LoRA target (down_proj) × 2 (A+B) = 48
        // Total: 84
        let expected_lora_params = 84;
        assert_eq!(
            trainable.len(),
            expected_lora_params,
            "expected {expected_lora_params} trainable params, got {}",
            trainable.len()
        );

        let tokens = Array::from_slice(&[100i32, 101, 102, 103], &[1, 4]);
        let targets = Array::from_slice(&[101i32, 102, 103, 104], &[1, 4]);

        let losses = train_loop(&mut model, &[tokens], &[targets], &lora_cfg, 3, 0.5, 10)
            .expect("training failed");

        assert!(losses.len() >= 2, "expected at least 2 training steps");
        let first = losses[0];
        let last = losses[losses.len() - 1];
        eprintln!("loss: {first:.4} -> {last:.4}");
        assert!(first.is_finite(), "first loss should be finite");
        assert!(
            last < first,
            "loss should decrease: {first:.4} -> {last:.4}"
        );
    }

    #[test]
    fn test_e2e_export_adapters_qwen3_5_2b() {
        let model_dir = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-2B-MLX-8bit");
        if !model_dir.exists() {
            eprintln!("Qwen3.5-2B-MLX-8bit not found, skipping export test");
            return;
        }

        let cfg = ModelConfig::qwen3_5_2b();
        let lora_cfg = LoraConfig::default();

        let model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg).expect("model load failed");

        let tmpdir = tempfile::tempdir().unwrap();
        let n = export_adapters(&model, &lora_cfg, &cfg, tmpdir.path()).expect("export failed");

        // Qwen3.5-2B: 84 trainable tensors (6 full-attn × 3 LoRA + 24 × 1 MLP)
        assert_eq!(n, 84, "Qwen3.5-2B has 84 trainable tensors");

        let st_path = tmpdir.path().join("adapters.safetensors");
        let loaded = Array::load_safetensors(&st_path).expect("load back failed");

        // Verify language_model.model prefix for Qwen3.5
        for key in loaded.keys() {
            assert!(
                key.starts_with("language_model.model.layers."),
                "key should start with 'language_model.model.layers.', got: {key}"
            );
        }

        // Verify lora_layers count (only full-attn layers have all 4 LoRA targets)
        let config_json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(tmpdir.path().join("adapter_config.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(config_json["lora_layers"], 6, "only 6 full-attn layers");
        assert_eq!(config_json["num_layers"], 24);

        eprintln!("exported {} adapter tensors (Qwen3.5-2B)", n);
    }

    #[cfg(feature = "ane")]
    #[test]
    fn test_export_ane_adapters_qwen3_5_2b_layout() {
        let cfg = ModelConfig::qwen3_5_2b();
        let lora = crate::agent::ane_lora::LoraModel::with_full_dims(
            crate::agent::ane_lora::LoraConfig::default(),
            cfg.n_layers,
            cfg.dim,
            cfg.n_kv_heads * cfg.head_dim,
            cfg.n_heads * cfg.head_dim,
            if cfg.attn_output_gate {
                2 * cfg.n_heads * cfg.head_dim
            } else {
                cfg.n_heads * cfg.head_dim
            },
            cfg.hidden_dim,
        );

        let tmpdir = tempfile::tempdir().unwrap();
        let n = export_ane_adapters(&lora, &cfg, Some(&cfg.linear_attn_indices), tmpdir.path())
            .expect("ANE export failed");

        assert_eq!(
            n, 84,
            "Qwen3.5-2B ANE export should match mlx-lm tensor count"
        );

        let st_path = tmpdir.path().join("adapters.safetensors");
        let loaded = Array::load_safetensors(&st_path).expect("load back failed");
        assert_eq!(loaded.len(), n);

        for key in loaded.keys() {
            assert!(
                key.starts_with("language_model.model.layers."),
                "key should start with Qwen3.5 weight prefix, got: {key}"
            );
            assert!(
                key.ends_with(".weight"),
                "key should end with .weight, got: {key}"
            );
        }
        assert!(
            loaded
                .keys()
                .any(|k| k.contains(".mlp.down_proj.lora_a.weight")),
            "expected down_proj LoRA weights in export"
        );
        assert!(
            !loaded
                .keys()
                .any(|k| k.contains(".layers.0.self_attn.q_proj.lora_a.weight")),
            "linear-attention layers should not export attention LoRA weights"
        );
        assert!(
            loaded
                .keys()
                .any(|k| k.contains(".layers.3.self_attn.q_proj.lora_a.weight")),
            "full-attention layers should export attention LoRA weights"
        );

        let config_json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(tmpdir.path().join("adapter_config.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(config_json["rank"], 32);
        assert_eq!(config_json["alpha"], 32.0);
        assert_eq!(config_json["lora_layers"], 6);
        assert_eq!(config_json["num_layers"], 24);
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
        let lora_cfg = LoraConfig {
            lr: 1e-5,
            ..LoraConfig::default()
        };

        eprintln!("loading model for long-sequence NaN regression test...");
        let mut model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg).expect("model load failed");

        // Simulate a real conversation: 64 random token IDs in valid vocab range
        let seq: Vec<i32> = (0..65)
            .map(|i| ((i * 7919 + 1337) % 248320) as i32)
            .collect();
        let tokens = Array::from_slice(&seq[..64], &[1, 64]);
        let targets = Array::from_slice(&seq[1..65], &[1, 64]);

        let losses = train_loop(&mut model, &[tokens], &[targets], &lora_cfg, 3, 0.5, 10)
            .expect("training failed — likely NaN gradient regression");

        for (i, l) in losses.iter().enumerate() {
            eprintln!("  step {i}: loss={l:.4}");
            assert!(l.is_finite(), "step {i} loss is NaN/Inf: {l}");
        }
        assert!(losses.len() >= 2);
        assert!(losses.last().unwrap() < &losses[0], "loss should decrease");
    }

    /// Regression test: train on real ChatML-tokenized conversation.
    /// Previously produced NaN gradients because the GDN recurrence backward
    /// has exponentially growing gradients (verified: inf at T=64 in Python).
    /// Fixed by applying stop_gradient to GDN attention output.
    #[test]
    fn test_e2e_qwen3_5_2b_chatml_no_nan() {
        let model_dir = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-2B-MLX-8bit");
        if !model_dir.exists() {
            eprintln!("Qwen3.5-2B-MLX-8bit not found, skipping");
            return;
        }

        let cfg = ModelConfig::qwen3_5_2b();
        let lora_cfg = LoraConfig {
            lr: 1e-5,
            ..LoraConfig::default()
        };

        eprintln!("loading model for ChatML NaN regression test...");
        let mut model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg).expect("model load failed");

        let tokenizer = MlxTokenizer::load(&model_dir).expect("tokenizer load failed");

        // Real ChatML conversation — the exact pattern that previously caused NaN
        let text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
                    <|im_start|>user\nWhat is the capital of France?<|im_end|>\n\
                    <|im_start|>assistant\nThe capital of France is Paris.<|im_end|>\n\
                    <|im_start|>user\nWhat about Germany?<|im_end|>\n\
                    <|im_start|>assistant\nThe capital of Germany is Berlin.<|im_end|>\n";
        let all_tokens = tokenizer.encode(text).expect("tokenize failed");
        eprintln!(
            "  ChatML tokens: {}, first 10: {:?}",
            all_tokens.len(),
            &all_tokens[..10.min(all_tokens.len())]
        );

        let input = Array::from_slice(
            &all_tokens[..all_tokens.len() - 1],
            &[1, (all_tokens.len() - 1) as i32],
        );
        let target = Array::from_slice(&all_tokens[1..], &[1, (all_tokens.len() - 1) as i32]);

        let losses = train_loop(&mut model, &[input], &[target], &lora_cfg, 3, 0.5, 10)
            .expect("training failed — likely NaN gradient regression");

        for (i, l) in losses.iter().enumerate() {
            eprintln!("  step {i}: loss={l:.4}");
            assert!(l.is_finite(), "step {i} loss is NaN/Inf: {l}");
        }
        assert!(losses.len() >= 2);
        assert!(
            losses.last().unwrap() < &losses[0],
            "loss should decrease: {:.4} -> {:.4}",
            losses[0],
            losses.last().unwrap()
        );
    }

    /// Load a raw f32 binary tensor from a reference directory under tests/
    fn load_reference_tensor_from(dir: &str, name: &str, shape: &[i32]) -> Array {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(dir)
            .join(format!("{name}.bin"));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| {
            panic!(
                "cannot read {}: {e}\nRun the appropriate reference script",
                path.display()
            )
        });
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
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
        let ref_dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/gdn_reference_raw");
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
        let mut model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg).expect("model load failed");

        // Load reference input [1, 4, 2048] and expected output [1, 4, 2048]
        let ref_input =
            load_reference_tensor_from("tests/gdn_reference_raw", "input", &[1, 4, 2048]);
        let ref_output =
            load_reference_tensor_from("tests/gdn_reference_raw", "output", &[1, 4, 2048]);

        // Run our forward on layer 0 (which is a linear_attn layer)
        // The reference was computed on raw input (before layernorm), so we must
        // call the linear_attn sub-module directly
        let our_output = model.layers[0]
            .forward_linear_attn(&ref_input)
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
        let ref_recurrence = load_reference_tensor_from(
            "tests/gdn_reference_raw",
            "recurrence_out",
            &[1, 4, 16, 128],
        );
        eprintln!("\nIntermediate reference tensors (for debugging):");
        eprintln!(
            "  recurrence_out sample: {:?}",
            &ref_recurrence.as_slice::<f32>()[..8]
        );
    }

    #[test]
    fn test_full_attn_numerical_vs_python_reference() {
        let ref_dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/full_attn_reference_raw");
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
        let mut model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg).expect("model load failed");

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
        let mut model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg).expect("model load failed");

        // "The capital of France is" → Python produces token 11751 (Paris) first
        let prompt = &[760i32, 6511, 314, 9338, 369];

        let eos = 248046;

        // Layer-by-layer comparison with Python
        {
            let input = Array::from_slice(prompt, &[1, 5]);
            let mut h = model.embed_tokens.forward(&input).unwrap();
            for (i, layer) in model.layers.iter_mut().enumerate() {
                h = layer.forward(&h, None).unwrap();
                if [0, 1, 2, 3, 5, 11, 23].contains(&i) {
                    let hf = h.as_dtype(mlx_rs::Dtype::Float32).unwrap();
                    let flat = hf.reshape(&[-1]).unwrap();
                    let n = hf.shape().iter().product::<i32>();
                    // last position (idx 4), first 3 dims: offset = 4*2048
                    let off = 4 * 2048;
                    let v: Vec<f32> = (off..off + 3)
                        .map(|j| flat.index(j as i32).as_slice::<f32>()[0])
                        .collect();
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
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Int32)
            .unwrap();
        let argmax_id: i32 = argmax_token.as_slice::<i32>()[0];
        eprintln!("argmax token: {argmax_id} (expected 11751 = 'Paris')");

        // Check top-5 logit values
        let last_flat = last_logits.reshape(&[-1]).unwrap();
        for &tid in &[11751i32, 279, 264, 3750, 13] {
            let val: f32 = last_flat.index(tid).as_slice::<f32>()[0];
            eprintln!("  token {tid}: logit {val:.4}");
        }

        // Greedy (temp=0) causes repetition on Qwen3.5 — use temp=0.6 per HF guidance
        eprintln!("\ngenerating (temp=0.6, max 20 tokens)...");
        let t0 = std::time::Instant::now();
        let generated = model
            .generate(prompt, 20, 0.6, &[eos])
            .expect("generation failed");
        let elapsed = t0.elapsed();

        eprintln!(
            "generated {} tokens in {:.1}s ({:.0} ms/token)",
            generated.len(),
            elapsed.as_secs_f64(),
            elapsed.as_millis() as f64 / generated.len().max(1) as f64
        );
        eprintln!("tokens: {:?}", generated);

        assert!(!generated.is_empty(), "should generate at least one token");
        assert_eq!(
            generated[0], 11751,
            "first generated token should be 11751 (Paris), got {}",
            generated[0]
        );
    }

    #[test]
    fn test_profiled_cached_decode_matches_regular_step() {
        let model_dir = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-2B-MLX-8bit");
        if !model_dir.exists() {
            eprintln!("Qwen3.5-2B-MLX-8bit not found, skipping");
            return;
        }

        let cfg = ModelConfig::qwen3_5_2b();
        let lora_cfg = LoraConfig::default();
        let mut model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg).expect("model load failed");
        let prompt = &[760i32, 6511, 314, 9338, 369];
        let state = model.prefill(prompt, 2).expect("prefill");
        let token = sample_next_token(&state.last_logits, 0.0)
            .expect("sample")
            .reshape(&[1, 1])
            .expect("reshape");

        let mut regular_caches = state.caches.clone();
        let regular = model
            .forward_logits_cached(&token, &mut regular_caches)
            .expect("regular cached decode");

        let mut profiled_caches = state.caches.clone();
        let (profiled, profile) = model
            .profile_forward_logits_cached_step(&token, &mut profiled_caches)
            .expect("profiled cached decode");

        let max_diff = profiled
            .subtract(&regular)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(
            max_diff < 1e-4,
            "profiled cached decode drift too large: {max_diff}"
        );
        assert_eq!(profile.layer_profiles.len(), model.layers.len());
        assert!(
            profile.total_ms > 0.0,
            "profile should record non-zero total time"
        );
        assert!(
            profile.embed_ms > 0.0,
            "profile should record embedding time"
        );
        assert!(profile.logits_ms > 0.0, "profile should record logits time");
        assert!(
            profile
                .layer_profiles
                .iter()
                .any(|layer| layer.kind == CachedDecodeLayerKind::FullAttention),
            "expected at least one full-attention layer in cached decode profile"
        );
        assert!(
            profile
                .layer_profiles
                .iter()
                .any(|layer| layer.kind == CachedDecodeLayerKind::LinearAttention),
            "expected at least one linear-attention layer in cached decode profile"
        );
        assert!(
            profile
                .layer_profiles
                .iter()
                .filter(|layer| layer.kind == CachedDecodeLayerKind::LinearAttention)
                .all(|layer| layer.linear_decode.is_some()),
            "expected linear-attention layers to carry a decode subprofile"
        );
    }

    #[test]
    fn test_decode_conv1d_helper_matches_reference_path() {
        let conv_dim = 2;
        let kernel = 4;
        let conv_weight = Array::from_slice(
            &[
                0.5f32, -0.25, 1.5, 2.0, // channel 0
                -1.0, 0.75, 0.25, 1.25, // channel 1
            ],
            &[conv_dim, kernel, 1],
        );

        let prefill_tail = Array::from_slice(
            &[
                1.0f32, 10.0, //
                2.0, 20.0, //
                3.0, 30.0,
            ],
            &[1, kernel - 1, conv_dim],
        );
        let decode_steps = [
            Array::from_slice(&[4.0f32, 40.0], &[1, 1, conv_dim]),
            Array::from_slice(&[5.0f32, 50.0], &[1, 1, conv_dim]),
            Array::from_slice(&[6.0f32, 60.0], &[1, 1, conv_dim]),
        ];

        let mut optimized_cache = GdnCache {
            state: None,
            conv_buf: Some(prefill_tail.clone()),
            conv_pos: kernel - 2,
        };
        let mut reference_cache = GdnCache {
            state: None,
            conv_buf: Some(prefill_tail),
            conv_pos: -1,
        };

        for (step_idx, qkv) in decode_steps.iter().enumerate() {
            let optimized = apply_decode_conv1d_step_with_weight(
                &conv_weight,
                kernel,
                qkv,
                &mut optimized_cache,
            )
            .expect("optimized decode conv");
            let reference = apply_decode_conv1d_step_reference_with_weight(
                &conv_weight,
                kernel,
                qkv,
                &mut reference_cache,
            )
            .expect("reference decode conv");
            let (max_diff, mean_diff) = compare_arrays(&optimized, &reference);
            assert!(
                max_diff < 1e-6 && mean_diff < 1e-6,
                "decode conv drift at step {step_idx}: max={max_diff:.3e} mean={mean_diff:.3e}"
            );
        }
    }

    #[test]
    fn test_linear_attn_cached_step_matches_full_forward_last_token() {
        let model_dir = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-2B-MLX-8bit");
        if !model_dir.exists() {
            eprintln!("Qwen3.5-2B-MLX-8bit not found, skipping");
            return;
        }

        let cfg = ModelConfig::qwen3_5_2b();
        let lora_cfg = LoraConfig::default();
        let mut model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg).expect("model load failed");
        let hidden: Vec<f32> = (0..(5 * cfg.dim))
            .map(|i| ((i as f32) * 0.013 - 0.7).sin() * 0.5)
            .collect();
        let input = Array::from_slice(&hidden, &[1, 5, cfg.dim as i32]);

        let linear_idx = model
            .layers
            .iter()
            .position(|layer| matches!(layer.attn, AttentionKind::Linear(_)))
            .expect("expected at least one linear-attention layer");
        let layer = &mut model.layers[linear_idx];
        let full = layer.forward_linear_attn(&input).expect("full forward");
        let expected_last = full.index((.., 4..5, ..));

        let prefill = input.index((.., ..4, ..));
        let decode = input.index((.., 4..5, ..));
        let mut cache = GdnCache::new();
        let prefill_out = match &mut layer.attn {
            AttentionKind::Linear(attn) => attn
                .forward_with_cache(&prefill, &mut cache)
                .expect("prefill cache forward"),
            AttentionKind::Full(_) => panic!("expected linear-attention layer"),
        };
        assert_eq!(prefill_out.shape(), &[1, 4, cfg.dim as i32]);
        let actual_last = match &mut layer.attn {
            AttentionKind::Linear(attn) => attn
                .forward_with_cache(&decode, &mut cache)
                .expect("decode cache forward"),
            AttentionKind::Full(_) => panic!("expected linear-attention layer"),
        };

        let (max_diff, mean_diff) = compare_arrays(&actual_last, &expected_last);
        assert!(
            max_diff < 1e-4 && mean_diff < 1e-5,
            "linear-attn cached-step drift too large: max={max_diff:.3e} mean={mean_diff:.3e}"
        );
    }

    #[test]
    fn test_full_model_cached_first_step_parity_long_prompt() {
        let model_dir = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-2B-MLX-8bit");
        if !model_dir.exists() {
            eprintln!("Qwen3.5-2B-MLX-8bit not found, skipping");
            return;
        }

        const BASE_SENTENCE: &str = concat!(
            "Rust favors explicit data movement over hidden allocations. ",
            "Quantized checkpoints reduce bandwidth pressure during inference. ",
            "Hybrid attention models mix full attention with gated delta recurrence. "
        );

        let cfg = ModelConfig::qwen3_5_2b();
        let lora_cfg = LoraConfig::default();
        let mut model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg).expect("model load failed");
        let tokenizer = MlxTokenizer::load(&model_dir).expect("tokenizer load failed");
        let mut text = String::new();
        let mut prompt = Vec::new();
        while prompt.len() < 128 {
            text.push_str(BASE_SENTENCE);
            prompt = tokenizer.encode(&text).expect("encode");
        }
        prompt.truncate(128);

        let prefill = model.prefill(&prompt, 2).expect("prefill");
        let token = sample_next_token(&prefill.last_logits, 0.0)
            .expect("sample")
            .reshape(&[1, 1])
            .expect("reshape");
        let token_id = token.as_slice::<i32>()[0];

        let mut cached_caches = prefill.caches.clone();
        let cached_logits = model
            .forward_logits_cached(&token, &mut cached_caches)
            .expect("cached decode");
        let cached_last = cached_logits.index((.., -1, ..));

        let mut profiled_caches = prefill.caches.clone();
        let (profiled_logits, _) = model
            .profile_forward_logits_cached_step(&token, &mut profiled_caches)
            .expect("profiled cached decode");
        let profiled_last = profiled_logits.index((.., -1, ..));

        let mut prompt_plus_first = prompt;
        prompt_plus_first.push(token_id);
        let no_cache_input =
            Array::from_slice(&prompt_plus_first, &[1, prompt_plus_first.len() as i32]);
        let no_cache_logits = model
            .forward_logits(&no_cache_input)
            .expect("no-cache forward");
        let no_cache_last = no_cache_logits.index((.., -1, ..));

        let (cached_max_diff, cached_mean_diff) = compare_arrays(&cached_last, &no_cache_last);
        let (profiled_max_diff, profiled_mean_diff) =
            compare_arrays(&profiled_last, &no_cache_last);
        let (profile_delta_max, profile_delta_mean) = compare_arrays(&profiled_last, &cached_last);
        eprintln!(
            "long-prompt full-model parity: cached/no-cache max={cached_max_diff:.3e} mean={cached_mean_diff:.3e}; profiled/no-cache max={profiled_max_diff:.3e} mean={profiled_mean_diff:.3e}; profiled/cached max={profile_delta_max:.3e} mean={profile_delta_mean:.3e}"
        );
        assert!(
            profile_delta_max < 1e-4 && profile_delta_mean < 1e-5,
            "profiled cached-step drift too large on long prompt: max={profile_delta_max:.3e} mean={profile_delta_mean:.3e}"
        );
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
        assert_eq!(
            prompt_ids,
            vec![760, 6511, 314, 9338, 369],
            "tokenizer should match hardcoded prompt tokens"
        );

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
        let mut model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg).expect("model load failed");
        let tok = MlxTokenizer::load(&model_dir).expect("tokenizer load failed");

        let t0 = std::time::Instant::now();
        let response = model
            .generate_text(&tok, "The capital of France is", 10, 0.6)
            .expect("generate_text failed");
        let elapsed = t0.elapsed();

        eprintln!(
            "generate_text: '{}' ({:.1}s)",
            response,
            elapsed.as_secs_f64()
        );
        assert!(
            response.contains("Paris"),
            "response should mention Paris, got: '{}'",
            response
        );
    }

    /// KV cache correctness: pure transformer (no GDN) should produce coherent text.
    /// Compares cached (generate) vs non-cached (forward_logits) first-token agreement.
    #[test]
    fn test_kv_cache_qwen3_pure_transformer() {
        let model_dir = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/mlx-community/Qwen3-0.6B-8bit");
        if !model_dir.exists() {
            eprintln!("Qwen3-0.6B-8bit not found, skipping");
            return;
        }

        let cfg = ModelConfig::qwen3_0_6b();
        let lora_cfg = LoraConfig::default();
        let tok = MlxTokenizer::load(&model_dir).expect("tokenizer load failed");
        let eos = tok.eos_token_id().unwrap_or(151645) as i32;
        let prompt = tok.encode("The capital of France is").expect("encode");

        // Non-cached: get argmax of last token logits
        let mut model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg).expect("load");
        let input = Array::from_slice(&prompt, &[1, prompt.len() as i32]);
        let logits = model.forward_logits(&input).expect("forward");
        let last = logits.index((.., -1, ..));
        let argmax_nocache = mlx_rs::ops::indexing::argmax_axis(&last, -1, false)
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Int32)
            .unwrap();
        let token_nocache: i32 = argmax_nocache.as_slice::<i32>()[0];
        eprintln!("non-cached argmax: {token_nocache}");

        // Cached: generate with temp=0.6 (greedy causes repetition on Qwen3)
        let mut model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg).expect("reload");
        let t0 = std::time::Instant::now();
        let generated = model.generate(&prompt, 30, 0.6, &[eos]).expect("generate");
        let elapsed = t0.elapsed();

        let text = tok.decode(&generated).unwrap_or_default();
        eprintln!(
            "KV-cached generation ({:.0}ms/tok): '{}'",
            elapsed.as_millis() as f64 / generated.len().max(1) as f64,
            text
        );
        eprintln!("tokens: {:?}", &generated[..generated.len().min(10)]);

        // First token should match non-cached argmax (high probability)
        assert!(!generated.is_empty(), "should generate tokens");
        // With temp=0.6, first token is very likely the argmax
        eprintln!(
            "cached first token: {}, non-cached argmax: {}",
            generated[0], token_nocache
        );

        // Check for repetition: no single token should appear > 50% of the time
        let mut counts = std::collections::HashMap::new();
        for &t in &generated {
            *counts.entry(t).or_insert(0usize) += 1;
        }
        let max_count = counts.values().max().copied().unwrap_or(0);
        let max_pct = max_count as f64 / generated.len() as f64 * 100.0;
        eprintln!(
            "max token repetition: {max_count}/{} ({max_pct:.0}%)",
            generated.len()
        );
        assert!(
            max_pct < 60.0,
            "excessive repetition ({max_pct:.0}%) suggests KV cache bug"
        );
    }
}
