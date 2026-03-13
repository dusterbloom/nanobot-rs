//! Performance ceiling benchmark — measures current kernel ceilings against
//! measured and assumed memory rooflines.
//!
//! This benchmark is intentionally explicit about what it is and is not:
//!   - It validates kernel correctness against scalar references before timing.
//!   - It times the production CPU SGEMM path and the production GDN recurrence.
//!   - It reports measured streaming-memory proxies separately from the assumed
//!     peak DRAM bandwidth used for theoretical roofline comparisons.
//!   - When built with `mlx`, it also measures a checkpoint-backed prefill +
//!     cached decode path on the real model.
//!   - Its roofline model still does not claim to be a full end-to-end decode bound.
//!
//! Usage:
//!   cargo run --features ane --release --bin perf_ceiling
//!   cargo run --features ane --release --bin perf_ceiling -- \
//!       --model-size-gb 2.0 --peak-bandwidth-gbps 273 --layers 24

use std::env;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

#[cfg(feature = "mlx")]
use mlx_rs::module::{Module, ModuleParameters};

#[cfg(feature = "mlx")]
use mlx_rs::ops::indexing::IndexOp;

const DEFAULT_MODEL_SIZE_GB: f64 = 2.0;
const DEFAULT_PEAK_BANDWIDTH_GBPS: f64 = 273.0;
const DEFAULT_LAYERS: usize = 24;
const DEFAULT_MEMORY_SIZE_MB: usize = 512;
const DEFAULT_DECODE_PROMPT_TOKENS: usize = 128;
const DEFAULT_DECODE_STEPS: usize = 16;
const QUANTIZED_GROUP_SIZE: usize = 64;
const MATMUL_CONFIGS: &[(usize, usize, usize, &str)] = &[
    (2048, 2048, 4, "QKV proj (dim x dim, seq=4)"),
    (2048, 2048, 64, "QKV proj (dim x dim, seq=64)"),
    (6144, 2048, 4, "QKV combined (3*dim x dim, seq=4)"),
    (5632, 2048, 4, "FFN gate (hidden x dim, seq=4)"),
    (2048, 5632, 4, "FFN down (dim x hidden, seq=4)"),
];

fn black_box<T>(x: T) -> T {
    std::hint::black_box(x)
}

#[derive(Debug, Clone)]
struct Options {
    model_size_gb: f64,
    peak_bandwidth_gbps: f64,
    layers: usize,
    memory_size_mb: usize,
    decode_prompt_tokens: usize,
    decode_steps: usize,
    checkpoint_dir: Option<PathBuf>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            model_size_gb: DEFAULT_MODEL_SIZE_GB,
            peak_bandwidth_gbps: DEFAULT_PEAK_BANDWIDTH_GBPS,
            layers: DEFAULT_LAYERS,
            memory_size_mb: DEFAULT_MEMORY_SIZE_MB,
            decode_prompt_tokens: DEFAULT_DECODE_PROMPT_TOKENS,
            decode_steps: DEFAULT_DECODE_STEPS,
            checkpoint_dir: None,
        }
    }
}

impl Options {
    fn parse() -> Self {
        let mut options = Self::default();
        let mut args = env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model-size-gb" => {
                    options.model_size_gb = parse_value(args.next(), "--model-size-gb", parse_f64);
                }
                "--peak-bandwidth-gbps" => {
                    options.peak_bandwidth_gbps =
                        parse_value(args.next(), "--peak-bandwidth-gbps", parse_f64);
                }
                "--layers" => {
                    options.layers = parse_value(args.next(), "--layers", parse_usize);
                }
                "--memory-size-mb" => {
                    options.memory_size_mb =
                        parse_value(args.next(), "--memory-size-mb", parse_usize);
                }
                "--decode-prompt-tokens" => {
                    options.decode_prompt_tokens =
                        parse_value(args.next(), "--decode-prompt-tokens", parse_usize);
                }
                "--decode-steps" => {
                    options.decode_steps = parse_value(args.next(), "--decode-steps", parse_usize);
                }
                "--checkpoint-dir" => {
                    options.checkpoint_dir = Some(PathBuf::from(parse_value(
                        args.next(),
                        "--checkpoint-dir",
                        |s| {
                            if s.is_empty() {
                                Err("path must not be empty".to_string())
                            } else {
                                Ok(s.to_string())
                            }
                        },
                    )));
                }
                "--help" | "-h" => {
                    print_usage();
                    process::exit(0);
                }
                _ => {
                    eprintln!("Unknown argument: {arg}");
                    eprintln!();
                    print_usage();
                    process::exit(2);
                }
            }
        }

        if options.model_size_gb <= 0.0 {
            eprintln!("--model-size-gb must be > 0");
            process::exit(2);
        }
        if options.peak_bandwidth_gbps <= 0.0 {
            eprintln!("--peak-bandwidth-gbps must be > 0");
            process::exit(2);
        }
        if options.layers == 0 {
            eprintln!("--layers must be > 0");
            process::exit(2);
        }
        if options.memory_size_mb == 0 {
            eprintln!("--memory-size-mb must be > 0");
            process::exit(2);
        }
        if options.decode_prompt_tokens == 0 {
            eprintln!("--decode-prompt-tokens must be > 0");
            process::exit(2);
        }
        if options.decode_steps == 0 {
            eprintln!("--decode-steps must be > 0");
            process::exit(2);
        }

        options
    }
}

fn parse_value<T>(
    raw: Option<String>,
    flag: &str,
    parser: impl FnOnce(&str) -> Result<T, String>,
) -> T {
    let value = raw.unwrap_or_else(|| {
        eprintln!("Missing value for {flag}");
        process::exit(2);
    });
    parser(&value).unwrap_or_else(|err| {
        eprintln!("Invalid value for {flag}: {err}");
        process::exit(2);
    })
}

fn parse_f64(raw: &str) -> Result<f64, String> {
    raw.parse::<f64>()
        .map_err(|e| format!("{raw:?} is not a number ({e})"))
}

fn parse_usize(raw: &str) -> Result<usize, String> {
    raw.parse::<usize>()
        .map_err(|e| format!("{raw:?} is not an integer ({e})"))
}

fn print_usage() {
    eprintln!("Usage: perf_ceiling [options]");
    eprintln!();
    eprintln!("Options:");
    eprintln!(
        "  --model-size-gb <f64>         Assumed model size for tok/s roofline (default: {DEFAULT_MODEL_SIZE_GB})"
    );
    eprintln!(
        "  --peak-bandwidth-gbps <f64>   Assumed peak DRAM bandwidth for theoretical roofline (default: {DEFAULT_PEAK_BANDWIDTH_GBPS})"
    );
    eprintln!(
        "  --layers <usize>              Assumed decoder layer count (default: {DEFAULT_LAYERS})"
    );
    eprintln!(
        "  --memory-size-mb <usize>      Buffer size for memory streaming tests (default: {DEFAULT_MEMORY_SIZE_MB})"
    );
    eprintln!(
        "  --decode-prompt-tokens <usize>  Prompt length for checkpoint-backed decode benchmark (default: {DEFAULT_DECODE_PROMPT_TOKENS})"
    );
    eprintln!(
        "  --decode-steps <usize>        Cached decode steps for checkpoint-backed decode benchmark (default: {DEFAULT_DECODE_STEPS})"
    );
    eprintln!(
        "  --checkpoint-dir <path>       MLX checkpoint dir for real quantized benchmark (auto-discovers when built with `mlx`)"
    );
}

struct MatmulBenchResult {
    label: &'static str,
    elapsed_ms: f64,
    gflops: f64,
}

struct GdnBenchResult {
    scalar_ms: f64,
    kernel_ms: f64,
    scalar_gflops: f64,
    kernel_gflops: f64,
    max_abs_diff: f32,
}

struct QuantizedBenchResult {
    label: &'static str,
    dense_ms: f64,
    materialized_ms: f64,
    pipeline_ms: f64,
    compression_ratio: f64,
    max_abs_diff: f32,
}

struct QuantizedGdnBenchResult {
    dense_ms: f64,
    materialized_ms: f64,
    pipeline_ms: f64,
    compression_ratio: f64,
    max_abs_diff: f32,
}

struct QuantizedLayerForwardBenchResult {
    dense_ms: f64,
    materialized_ms: f64,
    pipeline_ms: f64,
    compression_ratio: f64,
    max_abs_diff: f32,
}

#[cfg(feature = "mlx")]
struct CheckpointBenchCase {
    label: String,
    dense_ms: f64,
    materialized_ms: f64,
    pipeline_ms: f64,
    compression_ratio: f64,
    max_abs_diff: f32,
}

#[cfg(feature = "mlx")]
struct CheckpointBenchResult {
    model_dir: PathBuf,
    cases: Vec<CheckpointBenchCase>,
}

#[cfg(feature = "mlx")]
struct CheckpointDecodeBenchResult {
    model_dir: PathBuf,
    prompt_tokens: usize,
    decode_steps: usize,
    prefill_ms: f64,
    cached_step_ms: f64,
    no_cache_next_ms: f64,
    first_step_max_diff: f32,
    full_generate_ms: f64,
    compiled_decode: bool,
    kv_cache_bits: Option<i32>,
    kv_cache_group_size: i32,
    kv_cache_quantized_start: i32,
    first_token_id: i32,
    profile: nanobot::agent::mlx_lora::CachedDecodeProfile,
    conv_step_bench: Option<LinearDecodeConvBenchResult>,
    mlp_bench: Option<LinearMlpBenchResult>,
    aux_proj_bench: Option<LinearAuxProjBenchResult>,
}

#[cfg(feature = "mlx")]
struct LinearDecodeConvBenchResult {
    layer_idx: usize,
    current_ms: f64,
    reference_ms: f64,
    max_abs_diff: f32,
    cache_max_abs_diff: f32,
    cache_pos_match: bool,
    conv_dim: i32,
    history_len: i32,
}

#[cfg(feature = "mlx")]
struct LinearMlpBenchResult {
    layer_idx: usize,
    separate_ms: f64,
    fused_ms: f64,
    max_abs_diff: f32,
    input_dim: i32,
    hidden_dim: i32,
}

#[cfg(feature = "mlx")]
struct LinearAuxProjBenchResult {
    layer_idx: usize,
    separate_ms: f64,
    fused_ms: f64,
    max_abs_diff: f32,
    input_dim: i32,
    a_dim: i32,
    b_dim: i32,
    z_dim: i32,
}

struct TransposedMatmulBenchResult {
    label: &'static str,
    old_ms: f64,
    new_ms: f64,
    max_abs_diff: f32,
}

struct QuantizedTransposedMatmulBenchResult {
    label: &'static str,
    old_ms: f64,
    new_ms: f64,
    max_abs_diff: f32,
}

struct MemoryBenchResult {
    size_mb: usize,
    read_ms: f64,
    read_gbps: f64,
    copy_ms: f64,
    copy_gbps: f64,
}

struct BorrowedSingleLayerModel<'a> {
    cfg: nanobot::agent::ane_mil::MilConfig,
    layer: &'a nanobot::agent::ane_weights::LayerWeights,
    rms_final: &'a [f32],
    embed: &'a [f32],
    vocab_size: usize,
    lm_head: Option<&'a [f32]>,
}

impl<'a> nanobot::agent::ane_weights::WeightSource for BorrowedSingleLayerModel<'a> {
    fn cfg(&self) -> &nanobot::agent::ane_mil::MilConfig {
        &self.cfg
    }

    fn cfg_mut(&mut self) -> &mut nanobot::agent::ane_mil::MilConfig {
        &mut self.cfg
    }

    fn n_layers(&self) -> usize {
        1
    }

    fn layer(&self, l: usize) -> std::borrow::Cow<'_, nanobot::agent::ane_weights::LayerWeights> {
        assert_eq!(l, 0);
        std::borrow::Cow::Borrowed(self.layer)
    }

    fn embed(&self) -> &[f32] {
        self.embed
    }

    fn rms_final(&self) -> &[f32] {
        self.rms_final
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn lm_head(&self) -> Option<&[f32]> {
        self.lm_head
    }

    fn actual_dim(&self) -> usize {
        self.cfg.dim
    }

    fn actual_hidden_dim(&self) -> usize {
        self.cfg.hidden_dim
    }
}

#[cfg(feature = "mlx")]
fn single_layer_quantized_model(
    model: &nanobot::agent::ane_weights::QuantizedModelWeights,
    layer_idx: usize,
) -> nanobot::agent::ane_weights::QuantizedModelWeights {
    nanobot::agent::ane_weights::QuantizedModelWeights {
        cfg: model.cfg.clone(),
        layers: vec![model.layers[layer_idx].clone()],
        rms_final: model.rms_final.clone(),
        embed: model.embed.clone(),
        vocab_size: model.vocab_size,
        lm_head: model.lm_head.clone(),
        heads_per_group: model.heads_per_group,
    }
}

fn print_separator() {
    eprintln!("{}", "─".repeat(80));
}

fn print_header(title: &str) {
    eprintln!();
    print_separator();
    eprintln!("  {title}");
    print_separator();
}

fn fill_random(buf: &mut [f32], seed: u64) {
    let mut state = seed;
    for v in buf.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *v = ((state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0;
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max)
}

fn measure_average_ms(warmup: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

#[cfg(feature = "mlx")]
fn compare_latency(candidate_ms: f64, baseline_ms: f64) -> String {
    if candidate_ms <= baseline_ms {
        format!(
            "{:.2}x faster",
            baseline_ms / candidate_ms.max(f64::MIN_POSITIVE)
        )
    } else {
        format!(
            "{:.2}x slower",
            candidate_ms / baseline_ms.max(f64::MIN_POSITIVE)
        )
    }
}

#[cfg(feature = "mlx")]
fn max_abs_diff_array(
    lhs: &mlx_rs::Array,
    rhs: &mlx_rs::Array,
) -> Result<f32, mlx_rs::error::Exception> {
    let delta = lhs.subtract(rhs)?;
    let delta = delta.abs()?;
    let delta = delta.max(None)?;
    Ok(delta.item::<f32>())
}

#[cfg(feature = "mlx")]
fn combine_quantized_linear_pair(
    lhs: &mlx_rs::nn::QuantizedLinear,
    rhs: &mlx_rs::nn::QuantizedLinear,
) -> Result<mlx_rs::nn::QuantizedLinear, mlx_rs::error::Exception> {
    use mlx_rs::module::Param;

    if lhs.group_size != rhs.group_size || lhs.bits != rhs.bits {
        return Err(mlx_rs::error::Exception::custom(
            "cannot fuse quantized linears with different group sizes or bit widths",
        ));
    }
    if lhs.inner.weight.shape()[1] != rhs.inner.weight.shape()[1] {
        return Err(mlx_rs::error::Exception::custom(
            "cannot fuse quantized linears with different packed input widths",
        ));
    }

    let weight = mlx_rs::ops::concatenate_axis(&[&lhs.inner.weight, &rhs.inner.weight], 0)?;
    let scales = mlx_rs::ops::concatenate_axis(&[&lhs.scales, &rhs.scales], 0)?;
    let biases = mlx_rs::ops::concatenate_axis(&[&lhs.biases, &rhs.biases], 0)?;
    let lhs_rows = lhs.inner.weight.shape()[0];
    let rhs_rows = rhs.inner.weight.shape()[0];
    let bias_dtype = lhs
        .inner
        .bias
        .value
        .as_ref()
        .map(|bias| bias.dtype())
        .or_else(|| rhs.inner.bias.value.as_ref().map(|bias| bias.dtype()))
        .unwrap_or(scales.dtype());
    let lhs_bias = match lhs.inner.bias.value.as_ref() {
        Some(bias) => bias.clone(),
        None => mlx_rs::ops::zeros_dtype(&[lhs_rows], bias_dtype)?,
    };
    let rhs_bias = match rhs.inner.bias.value.as_ref() {
        Some(bias) => bias.clone(),
        None => mlx_rs::ops::zeros_dtype(&[rhs_rows], bias_dtype)?,
    };
    let bias = if lhs.inner.bias.value.is_none() && rhs.inner.bias.value.is_none() {
        None
    } else {
        Some(mlx_rs::ops::concatenate_axis(&[&lhs_bias, &rhs_bias], 0)?)
    };

    let inner = mlx_rs::nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(bias),
    };
    let mut fused = mlx_rs::nn::QuantizedLinear {
        group_size: lhs.group_size,
        bits: lhs.bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner,
    };
    fused.freeze_parameters(true);
    Ok(fused)
}

#[cfg(feature = "mlx")]
fn combine_quantized_linear_triplet(
    first: &mlx_rs::nn::QuantizedLinear,
    second: &mlx_rs::nn::QuantizedLinear,
    third: &mlx_rs::nn::QuantizedLinear,
) -> Result<mlx_rs::nn::QuantizedLinear, mlx_rs::error::Exception> {
    let first_two = combine_quantized_linear_pair(first, second)?;
    combine_quantized_linear_pair(&first_two, third)
}

#[cfg(feature = "mlx")]
fn canonicalize_linear_decode_cache(
    cache: &nanobot::agent::mlx_lora::GdnCache,
) -> Result<nanobot::agent::mlx_lora::GdnCache, mlx_rs::error::Exception> {
    let mut normalized = cache.clone();
    let Some(history) = cache.conv_buf.as_ref() else {
        normalized.conv_pos = -1;
        return Ok(normalized);
    };
    let history_len = history.shape()[1];
    if history_len <= 1 || cache.conv_pos < 0 {
        normalized.conv_pos = history_len.saturating_sub(1);
        return Ok(normalized);
    }
    let next_pos = (cache.conv_pos + 1).rem_euclid(history_len);
    if next_pos == 0 {
        normalized.conv_pos = history_len - 1;
        return Ok(normalized);
    }
    let tail = history.index((.., next_pos.., ..));
    let head = history.index((.., ..next_pos, ..));
    let reordered = mlx_rs::ops::concatenate_axis(&[&tail, &head], 1)?;
    reordered.eval()?;
    normalized.conv_buf = Some(reordered);
    normalized.conv_pos = history_len - 1;
    Ok(normalized)
}

#[cfg(feature = "mlx")]
fn sort_decode_layers_by_time(
    profile: &nanobot::agent::mlx_lora::CachedDecodeProfile,
) -> Vec<nanobot::agent::mlx_lora::CachedDecodeLayerProfile> {
    let mut layers = profile.layer_profiles.clone();
    layers.sort_by(|lhs, rhs| {
        rhs.total_ms
            .partial_cmp(&lhs.total_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    layers
}

#[cfg(feature = "mlx")]
fn bench_checkpoint_linear_mlp_fusion(
    model: &mut nanobot::agent::mlx_lora::MlxLoraModel,
) -> Result<Option<LinearMlpBenchResult>, mlx_rs::error::Exception> {
    let Some(layer_idx) = model
        .layers
        .iter()
        .position(|layer| layer.linear_attention().is_some())
    else {
        return Ok(None);
    };
    let mlp = &mut model.layers[layer_idx].mlp;
    let hidden_dim = mlp.gate_proj.inner.weight.shape()[0];
    let input_dim = mlp.gate_proj.inner.weight.shape()[1] * 32 / mlp.gate_proj.bits;
    let mut fused = combine_quantized_linear_pair(&mlp.gate_proj, &mlp.up_proj)?;
    let mut input = vec![0.0f32; input_dim as usize];
    fill_random(&mut input, 0x5eed_cafe);
    let x = mlx_rs::Array::from_slice(&input, &[1, 1, input_dim]);

    const WARMUP: usize = 5;
    const ITERS: usize = 100;

    let separate_ms = measure_average_ms(WARMUP, ITERS, || {
        let out = mlp.forward(&x).expect("separate MLP forward");
        out.eval().expect("eval separate MLP output");
        black_box(out);
    });
    let fused_ms = measure_average_ms(WARMUP, ITERS, || {
        let gate_up = fused.forward(&x).expect("fused gate+up forward");
        let parts = gate_up
            .split_axis(&[hidden_dim], -1)
            .expect("split fused gate+up");
        let gate = mlx_rs::nn::silu(&parts[0]).expect("silu fused gate");
        let h = gate.multiply(&parts[1]).expect("fused gate*up");
        let out = mlp.down_proj.forward(&h).expect("fused MLP down_proj");
        out.eval().expect("eval fused MLP output");
        black_box(out);
    });

    let separate_out = mlp.forward(&x)?;
    separate_out.eval()?;
    let gate_up = fused.forward(&x)?;
    let parts = gate_up.split_axis(&[hidden_dim], -1)?;
    let gate = mlx_rs::nn::silu(&parts[0])?;
    let h = gate.multiply(&parts[1])?;
    let fused_out = mlp.down_proj.forward(&h)?;
    fused_out.eval()?;

    Ok(Some(LinearMlpBenchResult {
        layer_idx,
        separate_ms,
        fused_ms,
        max_abs_diff: max_abs_diff_array(&separate_out, &fused_out)?,
        input_dim,
        hidden_dim,
    }))
}

#[cfg(feature = "mlx")]
fn bench_checkpoint_linear_aux_proj_fusion(
    model: &mut nanobot::agent::mlx_lora::MlxLoraModel,
) -> Result<Option<LinearAuxProjBenchResult>, mlx_rs::error::Exception> {
    let Some(layer_idx) = model
        .layers
        .iter()
        .position(|layer| layer.linear_attention().is_some())
    else {
        return Ok(None);
    };
    let Some(attn) = model.layers[layer_idx].linear_attention_mut() else {
        return Ok(None);
    };

    let a_dim = attn.in_proj_a.inner.weight.shape()[0];
    let b_dim = attn.in_proj_b.inner.weight.shape()[0];
    let z_dim = attn.in_proj_z.inner.weight.shape()[0];
    let input_dim = attn.in_proj_a.inner.weight.shape()[1] * 32 / attn.in_proj_a.bits;
    let mut fused =
        combine_quantized_linear_triplet(&attn.in_proj_a, &attn.in_proj_b, &attn.in_proj_z)?;
    let mut input = vec![0.0f32; input_dim as usize];
    fill_random(&mut input, 0xabad_1dea);
    let x = mlx_rs::Array::from_slice(&input, &[1, 1, input_dim]);

    const WARMUP: usize = 5;
    const ITERS: usize = 100;

    let separate_ms = measure_average_ms(WARMUP, ITERS, || {
        let a = attn.in_proj_a.forward(&x).expect("aux a");
        let b = attn.in_proj_b.forward(&x).expect("aux b");
        let z = attn.in_proj_z.forward(&x).expect("aux z");
        mlx_rs::transforms::eval([&a, &b, &z]).expect("eval separate aux");
        black_box((a, b, z));
    });
    let fused_ms = measure_average_ms(WARMUP, ITERS, || {
        let abz = fused.forward(&x).expect("fused abz");
        let parts = abz
            .split_axis(&[a_dim, a_dim + b_dim], -1)
            .expect("split fused abz");
        mlx_rs::transforms::eval([&parts[0], &parts[1], &parts[2]]).expect("eval fused aux");
        black_box(parts);
    });

    let a = attn.in_proj_a.forward(&x)?;
    let b = attn.in_proj_b.forward(&x)?;
    let z = attn.in_proj_z.forward(&x)?;
    mlx_rs::transforms::eval([&a, &b, &z])?;
    let abz = fused.forward(&x)?;
    let parts = abz.split_axis(&[a_dim, a_dim + b_dim], -1)?;
    mlx_rs::transforms::eval([&parts[0], &parts[1], &parts[2]])?;
    let max_abs_diff = max_abs_diff_array(&a, &parts[0])?
        .max(max_abs_diff_array(&b, &parts[1])?)
        .max(max_abs_diff_array(&z, &parts[2])?);

    Ok(Some(LinearAuxProjBenchResult {
        layer_idx,
        separate_ms,
        fused_ms,
        max_abs_diff,
        input_dim,
        a_dim,
        b_dim,
        z_dim,
    }))
}

#[cfg(feature = "mlx")]
fn bench_checkpoint_linear_decode_conv_step(
    model: &nanobot::agent::mlx_lora::MlxLoraModel,
    caches: &[nanobot::agent::mlx_lora::LayerCache],
) -> Result<Option<LinearDecodeConvBenchResult>, mlx_rs::error::Exception> {
    use nanobot::agent::mlx_lora::LayerCache;

    let Some(layer_idx) = model
        .layers
        .iter()
        .position(|layer| layer.linear_attention().is_some())
    else {
        return Ok(None);
    };
    let Some(attn) = model.layers[layer_idx].linear_attention() else {
        return Ok(None);
    };
    if attn.conv_kernel <= 1 {
        return Ok(None);
    }
    let LayerCache::LinearAttn(base_cache) = &caches[layer_idx] else {
        return Ok(None);
    };
    let reference_base_cache = canonicalize_linear_decode_cache(base_cache)?;

    let conv_dim = attn.conv1d_weight.shape()[0];
    let history_len = attn.conv_kernel.saturating_sub(1);
    let qkv = if let Some(history) = reference_base_cache.conv_buf.as_ref() {
        let pos = history.shape()[1].saturating_sub(1);
        Ok(history.index((.., pos..pos + 1, ..)))
    } else {
        mlx_rs::ops::zeros_dtype(&[1, 1, conv_dim], attn.conv1d_weight.dtype())
    }?;
    qkv.eval()?;

    const WARMUP: usize = 5;
    const ITERS: usize = 100;

    let current_ms = measure_average_ms(WARMUP, ITERS, || {
        let mut cache = base_cache.clone();
        let out = attn
            .apply_decode_conv1d_step(&qkv, &mut cache)
            .expect("current decode conv step");
        out.eval().expect("eval current decode conv output");
        black_box((out, cache.conv_pos));
    });
    let reference_ms = measure_average_ms(WARMUP, ITERS, || {
        let mut cache = reference_base_cache.clone();
        let out = attn
            .apply_decode_conv1d_step_reference(&qkv, &mut cache)
            .expect("reference decode conv step");
        out.eval().expect("eval reference decode conv output");
        black_box((out, cache.conv_pos));
    });

    let mut current_cache = base_cache.clone();
    let current_out = attn.apply_decode_conv1d_step(&qkv, &mut current_cache)?;
    current_out.eval()?;
    let mut reference_cache = reference_base_cache.clone();
    let reference_out = attn.apply_decode_conv1d_step_reference(&qkv, &mut reference_cache)?;
    reference_out.eval()?;
    let max_abs_diff = max_abs_diff_array(&current_out, &reference_out)?;
    let current_cache = canonicalize_linear_decode_cache(&current_cache)?;
    let reference_cache = canonicalize_linear_decode_cache(&reference_cache)?;
    let cache_max_abs_diff = match (&current_cache.conv_buf, &reference_cache.conv_buf) {
        (Some(lhs), Some(rhs)) => max_abs_diff_array(lhs, rhs)?,
        (None, None) => 0.0,
        _ => f32::INFINITY,
    };

    Ok(Some(LinearDecodeConvBenchResult {
        layer_idx,
        current_ms,
        reference_ms,
        max_abs_diff,
        cache_max_abs_diff,
        cache_pos_match: current_cache.conv_pos == reference_cache.conv_pos,
        conv_dim,
        history_len,
    }))
}

fn quantize_affine_u8(
    dense: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> nanobot::agent::ane_weights::QuantizedTensor {
    assert_eq!(dense.len(), rows * cols);
    assert_eq!(cols % group_size, 0);

    let n_groups = cols / group_size;
    let mut data = vec![0u8; dense.len()];
    let mut scales = vec![0.0f32; rows * n_groups];
    let mut biases = vec![0.0f32; rows * n_groups];

    for r in 0..rows {
        for g in 0..n_groups {
            let start = r * cols + g * group_size;
            let group = &dense[start..start + group_size];
            let mut min_v = f32::INFINITY;
            let mut max_v = f32::NEG_INFINITY;
            for &value in group {
                min_v = min_v.min(value);
                max_v = max_v.max(value);
            }

            let scale = if max_v > min_v {
                (max_v - min_v) / 255.0
            } else {
                0.0
            };
            let idx = r * n_groups + g;
            scales[idx] = scale;
            biases[idx] = min_v;

            for (offset, &value) in group.iter().enumerate() {
                let q = if scale > 0.0 {
                    ((value - min_v) / scale).round().clamp(0.0, 255.0)
                } else {
                    0.0
                };
                data[start + offset] = q as u8;
            }
        }
    }

    nanobot::agent::ane_weights::QuantizedTensor {
        data,
        scales,
        biases,
        rows,
        cols,
        group_size,
        bits: 8,
    }
}

// ---------------------------------------------------------------------------
// Naive scalar matmul (baseline)
// ---------------------------------------------------------------------------

fn naive_matmul(w: &[f32], x: &[f32], m: usize, n: usize, s: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * s];
    for mi in 0..m {
        for si in 0..s {
            let mut acc = 0.0f32;
            for ni in 0..n {
                acc += w[mi * n + ni] * x[ni * s + si];
            }
            out[mi * s + si] = acc;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Scalar GDN recurrence reference
// ---------------------------------------------------------------------------

fn scalar_gdn_recurrence(
    q_exp: &[f32],
    k_exp: &[f32],
    v_raw: &[f32],
    g: &[f32],
    beta: &[f32],
    h_v: usize,
    d_k: usize,
    d_v: usize,
    seq: usize,
) -> Vec<f32> {
    let mut state = vec![0.0f32; h_v * d_v * d_k];
    let mut y = vec![0.0f32; h_v * d_v * seq];

    for t in 0..seq {
        for h in 0..h_v {
            let g_t = g[h * seq + t];
            let beta_t = beta[h * seq + t];
            for dv in 0..d_v {
                for dk in 0..d_k {
                    state[h * d_v * d_k + dv * d_k + dk] *= g_t;
                }
            }
            for dv in 0..d_v {
                let mut kv_mem = 0.0f32;
                for dk in 0..d_k {
                    kv_mem +=
                        state[h * d_v * d_k + dv * d_k + dk] * k_exp[(h * d_k + dk) * seq + t];
                }
                let v_t = v_raw[(h * d_v + dv) * seq + t];
                let delta = (v_t - kv_mem) * beta_t;
                for dk in 0..d_k {
                    state[h * d_v * d_k + dv * d_k + dk] += k_exp[(h * d_k + dk) * seq + t] * delta;
                }
            }
            for dv in 0..d_v {
                let mut y_val = 0.0f32;
                for dk in 0..d_k {
                    y_val += state[h * d_v * d_k + dv * d_k + dk] * q_exp[(h * d_k + dk) * seq + t];
                }
                y[(h * d_v + dv) * seq + t] = y_val;
            }
        }
    }

    y
}

// ---------------------------------------------------------------------------
// Matmul benchmark: naive vs cpu_matmul (Accelerate SGEMM)
// ---------------------------------------------------------------------------

fn bench_matmul() -> Vec<MatmulBenchResult> {
    let mut results = Vec::new();

    print_header("MATMUL: validated naive scalar vs production cpu_matmul");
    eprintln!(
        "  {:40} {:>10} {:>10} {:>8} {:>8} {:>10}",
        "Config", "Naive(ms)", "SGEMM(ms)", "Speedup", "GFLOPS", "Max|diff|"
    );
    print_separator();

    let warmup = 2;
    let iters = 10;

    for &(m, n, s, label) in MATMUL_CONFIGS {
        let mut w = vec![0.0f32; m * n];
        let mut x = vec![0.0f32; n * s];
        fill_random(&mut w, 42);
        fill_random(&mut x, 123);

        let naive_ref = naive_matmul(&w, &x, m, n, s);
        let sgemm_ref = nanobot::agent::ane_forward::cpu_matmul(&w, &x, m, n, s);
        let max_diff = max_abs_diff(&naive_ref, &sgemm_ref);

        let flops = 2.0 * m as f64 * n as f64 * s as f64;

        let naive_ms = measure_average_ms(warmup, iters, || {
            black_box(naive_matmul(&w, &x, m, n, s));
        });

        let sgemm_ms = measure_average_ms(warmup, iters, || {
            black_box(nanobot::agent::ane_forward::cpu_matmul(&w, &x, m, n, s));
        });
        let sgemm_gflops = flops / (sgemm_ms / 1000.0) / 1e9;
        let speedup = naive_ms / sgemm_ms;

        eprintln!(
            "  {:40} {:>9.3}ms {:>9.3}ms {:>7.1}x {:>7.1} {:>10.3e}",
            label, naive_ms, sgemm_ms, speedup, sgemm_gflops, max_diff
        );

        results.push(MatmulBenchResult {
            label,
            elapsed_ms: sgemm_ms,
            gflops: sgemm_gflops,
        });
    }

    results
}

// ---------------------------------------------------------------------------
// Quantized path benchmark: QuantizedTensor::dequantize + cpu_matmul
// ---------------------------------------------------------------------------

fn bench_quantized_path(matmul_results: &[MatmulBenchResult]) -> Vec<QuantizedBenchResult> {
    let mut results = Vec::new();
    let labels = [
        "QKV proj (dim x dim, seq=4)",
        "FFN gate (hidden x dim, seq=4)",
        "FFN down (dim x hidden, seq=4)",
    ];

    print_header(
        "QUANTIZED PATH: synthetic grouped-u8 weights through current blocked quantized matmul",
    );
    eprintln!(
        "  {:32} {:>10} {:>10} {:>12} {:>8} {:>8} {:>10}",
        "Config", "Dense(ms)", "Old(ms)", "Current(ms)", "Gain", "Compr", "Max|diff|"
    );
    print_separator();

    for &(m, n, s, label) in MATMUL_CONFIGS {
        if !labels.contains(&label) {
            continue;
        }

        let dense_ms = matmul_results
            .iter()
            .find(|result| result.label == label)
            .expect("dense SGEMM result missing")
            .elapsed_ms;

        let mut w = vec![0.0f32; m * n];
        let mut x = vec![0.0f32; n * s];
        fill_random(&mut w, 777);
        fill_random(&mut x, 888);

        let quantized = quantize_affine_u8(&w, m, n, QUANTIZED_GROUP_SIZE);
        let dequantized = quantized.dequantize();
        let materialized_ref = nanobot::agent::ane_forward::cpu_matmul(&dequantized, &x, m, n, s);
        let current_ref = nanobot::agent::ane_forward::cpu_quantized_matmul(&quantized, &x, s);
        let max_diff = max_abs_diff(&materialized_ref, &current_ref);

        let materialized_ms = measure_average_ms(2, 10, || {
            let dense = quantized.dequantize();
            black_box(nanobot::agent::ane_forward::cpu_matmul(&dense, &x, m, n, s));
        });
        let pipeline_ms = measure_average_ms(2, 10, || {
            black_box(nanobot::agent::ane_forward::cpu_quantized_matmul(
                &quantized, &x, s,
            ));
        });

        let compression_ratio =
            (w.len() * std::mem::size_of::<f32>()) as f64 / quantized.quantized_bytes() as f64;
        let speedup = materialized_ms / pipeline_ms;

        eprintln!(
            "  {:32} {:>9.3}ms {:>9.3}ms {:>11.3}ms {:>7.2}x {:>7.2}x {:>10.3e}",
            label, dense_ms, materialized_ms, pipeline_ms, speedup, compression_ratio, max_diff
        );

        results.push(QuantizedBenchResult {
            label,
            dense_ms,
            materialized_ms,
            pipeline_ms,
            compression_ratio,
            max_abs_diff: max_diff,
        });
    }

    results
}

fn bench_quantized_gdn_path() -> QuantizedGdnBenchResult {
    print_header("QUANTIZED GDN PATH: synthetic hybrid layer through production forward");
    eprintln!(
        "  {:32} {:>10} {:>10} {:>12} {:>8} {:>8} {:>10}",
        "Config", "Dense(ms)", "Old(ms)", "Current(ms)", "Gain", "Compr", "Max|diff|"
    );
    print_separator();

    let cfg = nanobot::agent::ane_mil::MilConfig {
        dim: 2048,
        hidden_dim: 5632,
        n_heads: 16,
        seq_len: 4,
        n_kv_heads: 16,
        rope_theta: 1_000_000.0,
        rms_eps: 1e-6,
        has_lm_head: false,
        head_dim_explicit: 2048 / 16,
        linear_attn_indices: vec![0],
        linear_n_heads: 16,
        linear_head_dim: 128,
        linear_n_value_heads: 16,
        linear_value_head_dim: 128,
        conv_kernel_size: 4,
        attn_output_gate: false,
    };
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let h_v = cfg.linear_n_value_heads;
    let d_v = cfg.linear_value_head_dim;
    let value_dim = h_v * d_v;
    let qkv_dim = 2 * cfg.linear_n_heads * cfg.linear_head_dim + value_dim;

    let mut qkv_proj = vec![0.0f32; qkv_dim * dim];
    let mut a_proj = vec![0.0f32; h_v * dim];
    let mut b_proj = vec![0.0f32; h_v * dim];
    let mut z_proj = vec![0.0f32; value_dim * dim];
    let mut o_proj = vec![0.0f32; dim * value_dim];
    let mut a_log = vec![0.0f32; h_v];
    let mut dt_bias = vec![0.0f32; h_v];
    let mut norm_weight = vec![0.0f32; value_dim];
    let mut conv_weight = vec![0.0f32; qkv_dim * cfg.conv_kernel_size];
    let mut conv_bias = vec![0.0f32; qkv_dim];
    let mut xnorm = vec![0.0f32; dim * seq];
    fill_random(&mut qkv_proj, 11);
    fill_random(&mut a_proj, 12);
    fill_random(&mut b_proj, 13);
    fill_random(&mut z_proj, 14);
    fill_random(&mut o_proj, 15);
    fill_random(&mut a_log, 16);
    fill_random(&mut dt_bias, 17);
    fill_random(&mut norm_weight, 18);
    fill_random(&mut conv_weight, 19);
    fill_random(&mut conv_bias, 20);
    fill_random(&mut xnorm, 21);

    let quantized = nanobot::agent::ane_weights::QuantizedGdnLayerWeights {
        qkv_proj: quantize_affine_u8(&qkv_proj, qkv_dim, dim, QUANTIZED_GROUP_SIZE),
        a_proj: quantize_affine_u8(&a_proj, h_v, dim, QUANTIZED_GROUP_SIZE),
        b_proj: quantize_affine_u8(&b_proj, h_v, dim, QUANTIZED_GROUP_SIZE),
        z_proj: quantize_affine_u8(&z_proj, value_dim, dim, QUANTIZED_GROUP_SIZE),
        o_proj: quantize_affine_u8(&o_proj, dim, value_dim, QUANTIZED_GROUP_SIZE),
        a_log,
        dt_bias,
        norm_weight,
        conv_weight,
        conv_bias,
    };
    let dense = nanobot::agent::ane_weights::GdnLayerWeights {
        qkv_proj: quantized.qkv_proj.dequantize(),
        a_proj: quantized.a_proj.dequantize(),
        b_proj: quantized.b_proj.dequantize(),
        z_proj: quantized.z_proj.dequantize(),
        o_proj: quantized.o_proj.dequantize(),
        a_log: quantized.a_log.clone(),
        dt_bias: quantized.dt_bias.clone(),
        norm_weight: quantized.norm_weight.clone(),
        conv_weight: quantized.conv_weight.clone(),
        conv_bias: quantized.conv_bias.clone(),
    };

    let dense_ref = nanobot::agent::ane_forward::cpu_gdn_forward_bench(&dense, &xnorm, &cfg);
    let pipeline_ref =
        nanobot::agent::ane_forward::cpu_quantized_gdn_forward(&quantized, &xnorm, &cfg);
    let max_diff = max_abs_diff(&dense_ref, &pipeline_ref);

    let dense_ms = measure_average_ms(1, 5, || {
        black_box(nanobot::agent::ane_forward::cpu_gdn_forward_bench(
            &dense, &xnorm, &cfg,
        ));
    });
    let materialized_ms = measure_average_ms(1, 5, || {
        let dense_gdn = nanobot::agent::ane_weights::GdnLayerWeights {
            qkv_proj: quantized.qkv_proj.dequantize(),
            a_proj: quantized.a_proj.dequantize(),
            b_proj: quantized.b_proj.dequantize(),
            z_proj: quantized.z_proj.dequantize(),
            o_proj: quantized.o_proj.dequantize(),
            a_log: quantized.a_log.clone(),
            dt_bias: quantized.dt_bias.clone(),
            norm_weight: quantized.norm_weight.clone(),
            conv_weight: quantized.conv_weight.clone(),
            conv_bias: quantized.conv_bias.clone(),
        };
        black_box(nanobot::agent::ane_forward::cpu_gdn_forward_bench(
            &dense_gdn, &xnorm, &cfg,
        ));
    });
    let pipeline_ms = measure_average_ms(1, 5, || {
        black_box(nanobot::agent::ane_forward::cpu_quantized_gdn_forward(
            &quantized, &xnorm, &cfg,
        ));
    });

    let dense_bytes =
        (qkv_proj.len() + a_proj.len() + b_proj.len() + z_proj.len() + o_proj.len()) * 4;
    let quantized_bytes = quantized.qkv_proj.quantized_bytes()
        + quantized.a_proj.quantized_bytes()
        + quantized.b_proj.quantized_bytes()
        + quantized.z_proj.quantized_bytes()
        + quantized.o_proj.quantized_bytes();
    let compression_ratio = dense_bytes as f64 / quantized_bytes as f64;
    let speedup = materialized_ms / pipeline_ms;

    eprintln!(
        "  {:32} {:>9.3}ms {:>9.3}ms {:>11.3}ms {:>7.2}x {:>7.2}x {:>10.3e}",
        "GDN layer (seq=4)",
        dense_ms,
        materialized_ms,
        pipeline_ms,
        speedup,
        compression_ratio,
        max_diff
    );

    QuantizedGdnBenchResult {
        dense_ms,
        materialized_ms,
        pipeline_ms,
        compression_ratio,
        max_abs_diff: max_diff,
    }
}

fn bench_quantized_layer_forward_path() -> QuantizedLayerForwardBenchResult {
    use nanobot::agent::ane_forward::forward_cpu_generic;
    use nanobot::agent::ane_mil::MilConfig;
    use nanobot::agent::ane_weights::{QuantizedLayerWeights, QuantizedModelWeights};

    print_header("QUANTIZED LAYER FORWARD: production one-layer MHA/GQA forward path");
    eprintln!(
        "  {:32} {:>10} {:>10} {:>12} {:>8} {:>8} {:>10}",
        "Config", "Dense(ms)", "Old(ms)", "Current(ms)", "Gain", "Compr", "Max|diff|"
    );
    print_separator();

    let cfg = MilConfig {
        dim: 2048,
        hidden_dim: 5632,
        n_heads: 16,
        seq_len: 4,
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
    let dim = cfg.dim;
    let hidden = cfg.hidden_dim;
    let kv_dim = cfg.kv_dim();
    let vocab = 128;

    let mut wq = vec![0.0f32; dim * dim];
    let mut wk = vec![0.0f32; kv_dim * dim];
    let mut wv = vec![0.0f32; kv_dim * dim];
    let mut wo = vec![0.0f32; dim * dim];
    let mut w1 = vec![0.0f32; hidden * dim];
    let mut w2 = vec![0.0f32; dim * hidden];
    let mut w3 = vec![0.0f32; hidden * dim];
    let mut rms_att = vec![0.0f32; dim];
    let mut rms_ffn = vec![0.0f32; dim];
    let mut q_norm = vec![0.0f32; cfg.head_dim()];
    let mut k_norm = vec![0.0f32; cfg.head_dim()];
    let mut rms_final = vec![0.0f32; dim];
    let mut embed = vec![0.0f32; vocab * dim];
    fill_random(&mut wq, 31);
    fill_random(&mut wk, 32);
    fill_random(&mut wv, 33);
    fill_random(&mut wo, 34);
    fill_random(&mut w1, 35);
    fill_random(&mut w2, 36);
    fill_random(&mut w3, 37);
    fill_random(&mut rms_att, 38);
    fill_random(&mut rms_ffn, 39);
    fill_random(&mut q_norm, 40);
    fill_random(&mut k_norm, 41);
    fill_random(&mut rms_final, 42);
    fill_random(&mut embed, 43);
    for v in rms_att.iter_mut() {
        *v = v.abs() + 0.5;
    }
    for v in rms_ffn.iter_mut() {
        *v = v.abs() + 0.5;
    }
    for v in q_norm.iter_mut() {
        *v = v.abs() + 0.5;
    }
    for v in k_norm.iter_mut() {
        *v = v.abs() + 0.5;
    }
    for v in rms_final.iter_mut() {
        *v = v.abs() + 0.5;
    }

    let quantized = QuantizedModelWeights {
        cfg: cfg.clone(),
        layers: vec![QuantizedLayerWeights {
            wq: quantize_affine_u8(&wq, dim, dim, QUANTIZED_GROUP_SIZE),
            wk: quantize_affine_u8(&wk, kv_dim, dim, QUANTIZED_GROUP_SIZE),
            wv: quantize_affine_u8(&wv, kv_dim, dim, QUANTIZED_GROUP_SIZE),
            wo: quantize_affine_u8(&wo, dim, dim, QUANTIZED_GROUP_SIZE),
            w1: quantize_affine_u8(&w1, hidden, dim, QUANTIZED_GROUP_SIZE),
            w2: quantize_affine_u8(&w2, dim, hidden, QUANTIZED_GROUP_SIZE),
            w3: quantize_affine_u8(&w3, hidden, dim, QUANTIZED_GROUP_SIZE),
            rms_att: rms_att.clone(),
            rms_ffn: rms_ffn.clone(),
            q_norm: Some(q_norm.clone()),
            k_norm: Some(k_norm.clone()),
            gdn: None,
        }],
        rms_final: rms_final.clone(),
        embed: embed.clone(),
        vocab_size: vocab,
        lm_head: None,
        heads_per_group: cfg.heads_per_group(),
    };
    let dense_layer = quantized.dequantize_layer(0);
    let dense_model = BorrowedSingleLayerModel {
        cfg: cfg.clone(),
        layer: &dense_layer,
        rms_final: &rms_final,
        embed: &embed,
        vocab_size: vocab,
        lm_head: None,
    };
    let tokens = vec![1u16, 2, 3, 4];
    let targets = vec![2u16, 3, 4, 5];

    let dense_ref = forward_cpu_generic(&dense_model, None, &tokens, &targets);
    let current_ref = forward_cpu_generic(&quantized, None, &tokens, &targets);
    let max_diff = max_abs_diff(&dense_ref.base.classifier_dy, &current_ref.base.classifier_dy);

    let dense_ms = measure_average_ms(1, 5, || {
        black_box(forward_cpu_generic(&dense_model, None, &tokens, &targets));
    });
    let materialized_ms = measure_average_ms(1, 5, || {
        let old_layer = quantized.dequantize_layer(0);
        let old_model = BorrowedSingleLayerModel {
            cfg: cfg.clone(),
            layer: &old_layer,
            rms_final: &rms_final,
            embed: &embed,
            vocab_size: vocab,
            lm_head: None,
        };
        black_box(forward_cpu_generic(&old_model, None, &tokens, &targets));
    });
    let pipeline_ms = measure_average_ms(1, 5, || {
        black_box(forward_cpu_generic(&quantized, None, &tokens, &targets));
    });

    let dense_bytes = (wq.len() + wk.len() + wv.len() + wo.len() + w1.len() + w2.len() + w3.len())
        * std::mem::size_of::<f32>();
    let quantized_bytes = quantized.layers[0].wq.quantized_bytes()
        + quantized.layers[0].wk.quantized_bytes()
        + quantized.layers[0].wv.quantized_bytes()
        + quantized.layers[0].wo.quantized_bytes()
        + quantized.layers[0].w1.quantized_bytes()
        + quantized.layers[0].w2.quantized_bytes()
        + quantized.layers[0].w3.quantized_bytes();
    let compression_ratio = dense_bytes as f64 / quantized_bytes as f64;
    let speedup = materialized_ms / pipeline_ms;

    eprintln!(
        "  {:32} {:>9.3}ms {:>9.3}ms {:>11.3}ms {:>7.2}x {:>7.2}x {:>10.3e}",
        "MHA layer (seq=4, GQA)",
        dense_ms,
        materialized_ms,
        pipeline_ms,
        speedup,
        compression_ratio,
        max_diff
    );

    QuantizedLayerForwardBenchResult {
        dense_ms,
        materialized_ms,
        pipeline_ms,
        compression_ratio,
        max_abs_diff: max_diff,
    }
}

#[cfg(feature = "mlx")]
fn preferred_checkpoint_dir(explicit: Option<&std::path::Path>) -> Option<PathBuf> {
    if let Some(dir) = explicit {
        return Some(dir.to_path_buf());
    }

    let mut models = nanobot::agent::mlx_lm::discover_mlx_models();
    models.sort_by_key(|model| {
        let name = model.name.to_lowercase();
        if name.contains("qwen3.5-2b") {
            0
        } else if name.contains("qwen3-0.6b") {
            1
        } else if name.contains("qwen3-1.7b") {
            2
        } else if name.contains("qwen3.5-4b") {
            3
        } else if name.contains("qwen3.5-9b") {
            4
        } else if name.contains("qwen3-4b") {
            5
        } else if name.contains("qwen3-8b") {
            6
        } else {
            100
        }
    });
    models.into_iter().next().map(|model| model.path)
}

#[cfg(feature = "mlx")]
fn mlx_compiled_decode_enabled() -> bool {
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

#[cfg(feature = "mlx")]
fn build_decode_prompt_tokens(
    tokenizer: &nanobot::agent::mlx_lora::MlxTokenizer,
    target_tokens: usize,
) -> Result<Vec<i32>, String> {
    const BASE_SENTENCE: &str = concat!(
        "Rust favors explicit data movement over hidden allocations. ",
        "Quantized checkpoints reduce bandwidth pressure during inference. ",
        "Hybrid attention models mix full attention with gated delta recurrence. "
    );

    let mut text = String::new();
    let mut tokens = Vec::new();
    while tokens.len() < target_tokens {
        text.push_str(BASE_SENTENCE);
        tokens = tokenizer
            .encode(&text)
            .map_err(|err| format!("tokenizer encode failed: {err}"))?;
    }
    tokens.truncate(target_tokens);
    if tokens.is_empty() {
        return Err("tokenizer produced an empty prompt".to_string());
    }
    Ok(tokens)
}

#[cfg(feature = "mlx")]
fn measure_prefill_decode_ms(
    model: &mut nanobot::agent::mlx_lora::MlxLoraModel,
    prompt_tokens: &[i32],
    max_tokens: usize,
    warmup: usize,
    iters: usize,
) -> Result<f64, mlx_rs::error::Exception> {
    let warmup_states: Vec<_> = (0..warmup)
        .map(|_| model.prefill(prompt_tokens, max_tokens))
        .collect::<Result<_, _>>()?;
    for state in warmup_states {
        let generated = model.generate_from_prefill(state, max_tokens, 0.0, &[])?;
        debug_assert_eq!(generated.len(), max_tokens);
        black_box(generated);
    }

    let measure_states: Vec<_> = (0..iters)
        .map(|_| model.prefill(prompt_tokens, max_tokens))
        .collect::<Result<_, _>>()?;
    let t0 = Instant::now();
    for state in measure_states {
        let generated = model.generate_from_prefill(state, max_tokens, 0.0, &[])?;
        debug_assert_eq!(generated.len(), max_tokens);
        black_box(generated);
    }
    Ok(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64)
}

#[cfg(feature = "mlx")]
fn quantized_mha_weight_bytes(
    layer: &nanobot::agent::ane_weights::QuantizedLayerWeights,
) -> (usize, usize) {
    let dense_bytes = (layer.wq.rows * layer.wq.cols
        + layer.wk.rows * layer.wk.cols
        + layer.wv.rows * layer.wv.cols
        + layer.wo.rows * layer.wo.cols
        + layer.w1.rows * layer.w1.cols
        + layer.w2.rows * layer.w2.cols
        + layer.w3.rows * layer.w3.cols)
        * std::mem::size_of::<f32>();
    let quantized_bytes = layer.wq.quantized_bytes()
        + layer.wk.quantized_bytes()
        + layer.wv.quantized_bytes()
        + layer.wo.quantized_bytes()
        + layer.w1.quantized_bytes()
        + layer.w2.quantized_bytes()
        + layer.w3.quantized_bytes();
    (dense_bytes, quantized_bytes)
}

#[cfg(feature = "mlx")]
fn quantized_gdn_weight_bytes(
    layer: &nanobot::agent::ane_weights::QuantizedGdnLayerWeights,
) -> (usize, usize) {
    let dense_bytes = (layer.qkv_proj.rows * layer.qkv_proj.cols
        + layer.a_proj.rows * layer.a_proj.cols
        + layer.b_proj.rows * layer.b_proj.cols
        + layer.z_proj.rows * layer.z_proj.cols
        + layer.o_proj.rows * layer.o_proj.cols)
        * std::mem::size_of::<f32>();
    let quantized_bytes = layer.qkv_proj.quantized_bytes()
        + layer.a_proj.quantized_bytes()
        + layer.b_proj.quantized_bytes()
        + layer.z_proj.quantized_bytes()
        + layer.o_proj.quantized_bytes();
    (dense_bytes, quantized_bytes)
}

#[cfg(feature = "mlx")]
fn bench_checkpoint_quantized_forward_path(options: &Options) -> Option<CheckpointBenchResult> {
    use nanobot::agent::ane_forward::forward_cpu_generic;
    use nanobot::agent::ane_weights::QuantizedModelWeights;

    let model_dir = match preferred_checkpoint_dir(options.checkpoint_dir.as_deref()) {
        Some(dir) => dir,
        None => {
            eprintln!();
            eprintln!("  CHECKPOINT-BACKED QUANTIZED FORWARD");
            eprintln!("  No MLX checkpoint found on disk; skipping real checkpoint benchmark.");
            return None;
        }
    };
    let model_cfg = match nanobot::agent::mlx_lora::ModelConfig::from_config_json(&model_dir) {
        Some(cfg) => cfg,
        None => {
            eprintln!();
            eprintln!("  CHECKPOINT-BACKED QUANTIZED FORWARD");
            eprintln!(
                "  Could not parse config.json for {}; skipping real checkpoint benchmark.",
                model_dir.display()
            );
            return None;
        }
    };
    let mil_cfg = model_cfg.to_mil_config(4);
    let quantized = match QuantizedModelWeights::from_mlx_safetensors(&model_dir, &mil_cfg) {
        Ok(model) => model,
        Err(err) => {
            eprintln!();
            eprintln!("  CHECKPOINT-BACKED QUANTIZED FORWARD");
            eprintln!(
                "  Failed to load quantized weights from {}: {}; skipping real checkpoint benchmark.",
                model_dir.display(),
                err
            );
            return None;
        }
    };
    let tokens = vec![1u16, 2, 3, 4];
    let targets = vec![2u16, 3, 4, 5];

    print_header("CHECKPOINT-BACKED QUANTIZED FORWARD: real MLX model on disk");
    eprintln!("  Model dir: {}", model_dir.display());
    eprintln!(
        "  Auto-detected config: dim={} hidden={} layers={} linear_layers={}",
        model_cfg.dim,
        model_cfg.hidden_dim,
        model_cfg.n_layers,
        model_cfg.linear_attn_indices.len()
    );
    eprintln!(
        "  {:32} {:>10} {:>10} {:>12} {:>8} {:>8} {:>10}",
        "Config", "Dense(ms)", "Old(ms)", "Current(ms)", "Gain", "Compr", "Max|diff|"
    );
    print_separator();

    let mut cases = Vec::new();
    let mut selected = Vec::new();
    if let Some(idx) = (0..quantized.layers.len()).find(|&idx| quantized.layers[idx].gdn.is_none())
    {
        selected.push((idx, "Checkpoint MHA layer"));
    }
    if let Some(idx) = (0..quantized.layers.len()).find(|&idx| quantized.layers[idx].gdn.is_some())
    {
        selected.push((idx, "Checkpoint GDN layer"));
    }

    for (layer_idx, label) in selected {
        let single_layer = single_layer_quantized_model(&quantized, layer_idx);
        let dense_layer = single_layer.dequantize_layer(0);
        let dense_model = BorrowedSingleLayerModel {
            cfg: single_layer.cfg.clone(),
            layer: &dense_layer,
            rms_final: &single_layer.rms_final,
            embed: &single_layer.embed,
            vocab_size: single_layer.vocab_size,
            lm_head: single_layer.lm_head.as_deref(),
        };

        let dense_ref = forward_cpu_generic(&dense_model, None, &tokens, &targets);
        let current_ref = forward_cpu_generic(&single_layer, None, &tokens, &targets);
        let max_diff = max_abs_diff(&dense_ref.base.classifier_dy, &current_ref.base.classifier_dy);

        let dense_ms = measure_average_ms(1, 5, || {
            black_box(forward_cpu_generic(&dense_model, None, &tokens, &targets));
        });
        let materialized_ms = measure_average_ms(1, 5, || {
            let old_layer = single_layer.dequantize_layer(0);
            let old_model = BorrowedSingleLayerModel {
                cfg: single_layer.cfg.clone(),
                layer: &old_layer,
                rms_final: &single_layer.rms_final,
                embed: &single_layer.embed,
                vocab_size: single_layer.vocab_size,
                lm_head: single_layer.lm_head.as_deref(),
            };
            black_box(forward_cpu_generic(&old_model, None, &tokens, &targets));
        });
        let pipeline_ms = measure_average_ms(1, 5, || {
            black_box(forward_cpu_generic(&single_layer, None, &tokens, &targets));
        });

        let (dense_bytes, quantized_bytes) = if let Some(gdn) = &single_layer.layers[0].gdn {
            quantized_gdn_weight_bytes(gdn)
        } else {
            quantized_mha_weight_bytes(&single_layer.layers[0])
        };
        let compression_ratio = dense_bytes as f64 / quantized_bytes as f64;
        let speedup = materialized_ms / pipeline_ms;

        eprintln!(
            "  {:32} {:>9.3}ms {:>9.3}ms {:>11.3}ms {:>7.2}x {:>7.2}x {:>10.3e}",
            format!("{label} #{layer_idx}"),
            dense_ms,
            materialized_ms,
            pipeline_ms,
            speedup,
            compression_ratio,
            max_diff
        );

        cases.push(CheckpointBenchCase {
            label: format!("{label} #{layer_idx}"),
            dense_ms,
            materialized_ms,
            pipeline_ms,
            compression_ratio,
            max_abs_diff: max_diff,
        });
    }

    if cases.is_empty() {
        None
    } else {
        Some(CheckpointBenchResult { model_dir, cases })
    }
}

#[cfg(feature = "mlx")]
fn bench_checkpoint_decode_path(options: &Options) -> Option<CheckpointDecodeBenchResult> {
    use mlx_rs::ops::indexing::{argmax_axis, IndexOp};
    use mlx_rs::{Array, Dtype};
    use nanobot::agent::mlx_lora::{CachedDecodeLayerKind, LoraConfig, MlxLoraModel, MlxTokenizer};

    let model_dir = match preferred_checkpoint_dir(options.checkpoint_dir.as_deref()) {
        Some(dir) => dir,
        None => {
            eprintln!();
            eprintln!("  CHECKPOINT-BACKED DECODE");
            eprintln!("  No MLX checkpoint found on disk; skipping decode benchmark.");
            return None;
        }
    };
    let model_cfg = match nanobot::agent::mlx_lora::ModelConfig::from_config_json(&model_dir) {
        Some(cfg) => cfg,
        None => {
            eprintln!();
            eprintln!("  CHECKPOINT-BACKED DECODE");
            eprintln!(
                "  Could not parse config.json for {}; skipping decode benchmark.",
                model_dir.display()
            );
            return None;
        }
    };
    let tokenizer = match MlxTokenizer::load(&model_dir) {
        Ok(tokenizer) => tokenizer,
        Err(err) => {
            eprintln!();
            eprintln!("  CHECKPOINT-BACKED DECODE");
            eprintln!(
                "  Failed to load tokenizer from {}: {}; skipping decode benchmark.",
                model_dir.display(),
                err
            );
            return None;
        }
    };
    let prompt_tokens = match build_decode_prompt_tokens(&tokenizer, options.decode_prompt_tokens) {
        Ok(tokens) => tokens,
        Err(err) => {
            eprintln!();
            eprintln!("  CHECKPOINT-BACKED DECODE");
            eprintln!(
                "  Failed to build decode prompt for {}: {}; skipping decode benchmark.",
                model_dir.display(),
                err
            );
            return None;
        }
    };

    let lora_cfg = LoraConfig::default();
    let mut model = match MlxLoraModel::load(&model_dir, &model_cfg, &lora_cfg) {
        Ok(model) => model,
        Err(err) => {
            eprintln!();
            eprintln!("  CHECKPOINT-BACKED DECODE");
            eprintln!(
                "  Failed to load MLX model from {}: {}; skipping decode benchmark.",
                model_dir.display(),
                err
            );
            return None;
        }
    };

    let total_generate_tokens = options.decode_steps + 1;
    let first_prefill = match model.prefill(&prompt_tokens, total_generate_tokens) {
        Ok(state) => state,
        Err(err) => {
            eprintln!();
            eprintln!("  CHECKPOINT-BACKED DECODE");
            eprintln!(
                "  Prefill failed for {}: {}; skipping decode benchmark.",
                model_dir.display(),
                err
            );
            return None;
        }
    };
    let conv_step_bench =
        match bench_checkpoint_linear_decode_conv_step(&model, &first_prefill.caches) {
            Ok(result) => result,
            Err(err) => {
                eprintln!(
                    "  Linear-attn conv-step microbench failed for {}: {}; continuing without it.",
                    model_dir.display(),
                    err
                );
                None
            }
        };
    let mlp_bench = match bench_checkpoint_linear_mlp_fusion(&mut model) {
        Ok(result) => result,
        Err(err) => {
            eprintln!(
                "  Linear-attn MLP fusion microbench failed for {}: {}; continuing without it.",
                model_dir.display(),
                err
            );
            None
        }
    };
    let aux_proj_bench = match bench_checkpoint_linear_aux_proj_fusion(&mut model) {
        Ok(result) => result,
        Err(err) => {
            eprintln!(
                "  Linear-attn aux projection fusion microbench failed for {}: {}; continuing without it.",
                model_dir.display(),
                err
            );
            None
        }
    };
    let first_token = match argmax_axis(&first_prefill.last_logits, -1, false)
        .and_then(|token| token.as_dtype(Dtype::Int32))
    {
        Ok(token) => token,
        Err(err) => {
            eprintln!();
            eprintln!("  CHECKPOINT-BACKED DECODE");
            eprintln!(
                "  Failed to extract first token from prefill logits for {}: {}; skipping decode benchmark.",
                model_dir.display(),
                err
            );
            return None;
        }
    };
    let first_token_id = first_token.as_slice::<i32>()[0];
    let mut prompt_plus_first = prompt_tokens.clone();
    prompt_plus_first.push(first_token_id);
    let no_cache_input =
        Array::from_slice(&prompt_plus_first, &[1, prompt_plus_first.len() as i32]);

    const WARMUP: usize = 1;
    const ITERS: usize = 3;

    print_header("CHECKPOINT-BACKED DECODE: MLX prefill + cached decode");
    eprintln!("  Model dir: {}", model_dir.display());
    eprintln!(
        "  Prompt tokens: {}  Cached decode steps: {}  Total generated/run: {}",
        prompt_tokens.len(),
        options.decode_steps,
        total_generate_tokens
    );
    eprintln!(
        "  Compiled decode: {}  KV cache: {}",
        if mlx_compiled_decode_enabled() {
            "on"
        } else {
            "off"
        },
        match model.kv_cache_config.bits {
            Some(bits) => format!(
                "{bits}-bit (group_size={}, start={})",
                model.kv_cache_config.group_size, model.kv_cache_config.quantized_start
            ),
            None => "plain fp16/f32".to_string(),
        }
    );
    eprintln!("  First generated token id (greedy): {first_token_id}");

    let prefill_ms = measure_average_ms(WARMUP, ITERS, || {
        black_box(
            model
                .prefill(&prompt_tokens, total_generate_tokens)
                .expect("prefill"),
        );
    });
    let sample_only_ms =
        match measure_prefill_decode_ms(&mut model, &prompt_tokens, 1, WARMUP, ITERS) {
            Ok(ms) => ms,
            Err(err) => {
                eprintln!();
                eprintln!("  CHECKPOINT-BACKED DECODE");
                eprintln!(
                    "  Cached sample-only decode failed for {}: {}; skipping decode benchmark.",
                    model_dir.display(),
                    err
                );
                return None;
            }
        };
    let cached_series_ms = match measure_prefill_decode_ms(
        &mut model,
        &prompt_tokens,
        total_generate_tokens,
        WARMUP,
        ITERS,
    ) {
        Ok(ms) => ms,
        Err(err) => {
            eprintln!();
            eprintln!("  CHECKPOINT-BACKED DECODE");
            eprintln!(
                "  Cached decode failed for {}: {}; skipping decode benchmark.",
                model_dir.display(),
                err
            );
            return None;
        }
    };
    let no_cache_next_ms = measure_average_ms(WARMUP, ITERS, || {
        black_box(
            model
                .forward_logits(&no_cache_input)
                .expect("forward_logits"),
        );
    });
    let full_generate_ms = measure_average_ms(WARMUP, ITERS, || {
        let generated = model
            .generate(&prompt_tokens, total_generate_tokens, 0.0, &[])
            .expect("generate");
        debug_assert_eq!(generated.len(), total_generate_tokens);
        black_box(generated);
    });
    let profile_input = first_token.reshape(&[1, 1]).expect("reshape first token");
    let mut profile_caches = first_prefill.caches.clone();
    let (profiled_logits, profile) =
        match model.profile_forward_logits_cached_step(&profile_input, &mut profile_caches) {
            Ok(result) => result,
            Err(err) => {
                eprintln!();
                eprintln!("  CHECKPOINT-BACKED DECODE");
                eprintln!(
                    "  Cached decode profiling failed for {}: {}; skipping decode benchmark.",
                    model_dir.display(),
                    err
                );
                return None;
            }
        };
    let profiled_last = profiled_logits.index((.., -1, ..));
    let no_cache_logits = match model.forward_logits(&no_cache_input) {
        Ok(logits) => logits,
        Err(err) => {
            eprintln!();
            eprintln!("  CHECKPOINT-BACKED DECODE");
            eprintln!(
                "  No-cache validation forward failed for {}: {}; skipping decode benchmark.",
                model_dir.display(),
                err
            );
            return None;
        }
    };
    let no_cache_last = no_cache_logits.index((.., -1, ..));
    let first_step_max_diff = match profiled_last.subtract(&no_cache_last) {
        Ok(delta) => match delta.abs() {
            Ok(delta) => match delta.max(None) {
                Ok(delta) => delta.item::<f32>(),
                Err(err) => {
                    eprintln!();
                    eprintln!("  CHECKPOINT-BACKED DECODE");
                    eprintln!(
                        "  First-step cached/no-cache reduction failed for {}: {}; skipping decode benchmark.",
                        model_dir.display(),
                        err
                    );
                    return None;
                }
            },
            Err(err) => {
                eprintln!();
                eprintln!("  CHECKPOINT-BACKED DECODE");
                eprintln!(
                    "  First-step cached/no-cache abs failed for {}: {}; skipping decode benchmark.",
                    model_dir.display(),
                    err
                );
                return None;
            }
        },
        Err(err) => {
            eprintln!();
            eprintln!("  CHECKPOINT-BACKED DECODE");
            eprintln!(
                "  First-step cached/no-cache diff check failed for {}: {}; skipping decode benchmark.",
                model_dir.display(),
                err
            );
            return None;
        }
    };
    black_box(profiled_logits);

    let cached_compute_ms = (cached_series_ms - sample_only_ms).max(0.0);
    let cached_step_ms = cached_compute_ms / options.decode_steps as f64;
    let cached_tok_s = if cached_compute_ms > 0.0 {
        options.decode_steps as f64 / (cached_compute_ms / 1000.0)
    } else {
        0.0
    };
    let full_generate_tok_s = total_generate_tokens as f64 / (full_generate_ms / 1000.0);
    let prefill_tok_s = prompt_tokens.len() as f64 / (prefill_ms / 1000.0);
    let cached_vs_no_cache = compare_latency(cached_step_ms, no_cache_next_ms);
    let full_attn_ms = profile.kind_total_ms(CachedDecodeLayerKind::FullAttention);
    let linear_attn_ms = profile.kind_total_ms(CachedDecodeLayerKind::LinearAttention);
    let full_attn_layers = profile
        .layer_profiles
        .iter()
        .filter(|layer| layer.kind == CachedDecodeLayerKind::FullAttention)
        .count();
    let linear_attn_layers = profile
        .layer_profiles
        .iter()
        .filter(|layer| layer.kind == CachedDecodeLayerKind::LinearAttention)
        .count();
    let linear_stage_totals = profile.linear_decode_stage_totals();
    let linear_unprofiled_ms = (linear_attn_ms - linear_stage_totals.total_ms()).max(0.0);
    let compiled_layers = profile
        .layer_profiles
        .iter()
        .filter(|layer| layer.compiled_decode)
        .count();
    let quantized_kv_layers = profile
        .layer_profiles
        .iter()
        .filter(|layer| layer.quantized_kv)
        .count();
    let hottest_layers = sort_decode_layers_by_time(&profile);

    eprintln!(
        "  {:30} {:>10} {:>10} {:>12}",
        "Metric", "Time(ms)", "tok/s", "Notes"
    );
    print_separator();
    eprintln!(
        "  {:30} {:>9.3} {:>10.1} {:>12}",
        format!("Prefill ({} tok)", prompt_tokens.len()),
        prefill_ms,
        prefill_tok_s,
        "prompt -> cache"
    );
    eprintln!(
        "  {:30} {:>9.3} {:>10} {:>12}",
        "Sample from prefill only", sample_only_ms, "n/a", "argmax only"
    );
    eprintln!(
        "  {:30} {:>9.3} {:>10.1} {:>12}",
        format!("Cached decode ({} tok)", total_generate_tokens),
        cached_series_ms,
        total_generate_tokens as f64 / (cached_series_ms / 1000.0),
        "prefill reused"
    );
    eprintln!(
        "  {:30} {:>9.3} {:>10.1} {:>12}",
        "Cached step (isolated)", cached_step_ms, cached_tok_s, "series-sample"
    );
    eprintln!(
        "  {:30} {:>9.3} {:>10} {:>12}",
        "No-cache full forward +1", no_cache_next_ms, "n/a", "prompt+1 token"
    );
    eprintln!(
        "  {:30} {:>9.3} {:>10.1} {:>12}",
        format!("Full generate ({} tok)", total_generate_tokens),
        full_generate_ms,
        full_generate_tok_s,
        "prefill+decode"
    );
    eprintln!(
        "  Cached step vs no-cache next-token baseline: {}",
        cached_vs_no_cache
    );
    eprintln!(
        "  First cached step vs no-cache prompt+1 max |diff|: {:.3e}",
        first_step_max_diff
    );
    eprintln!();
    eprintln!(
        "  Profiled single cached step (forced layer eval; attribution only, not wall-clock):"
    );
    eprintln!("  {:30} {:>10} {:>10}", "Stage", "Time(ms)", "Share");
    print_separator();
    eprintln!(
        "  {:30} {:>9.3} {:>9.1}%",
        "Embed",
        profile.embed_ms,
        profile.embed_ms * 100.0 / profile.total_ms.max(f64::MIN_POSITIVE)
    );
    eprintln!(
        "  {:30} {:>9.3} {:>9.1}%",
        "Mask",
        profile.mask_ms,
        profile.mask_ms * 100.0 / profile.total_ms.max(f64::MIN_POSITIVE)
    );
    eprintln!(
        "  {:30} {:>9.3} {:>9.1}%",
        format!("Full-attn layers ({full_attn_layers})"),
        full_attn_ms,
        full_attn_ms * 100.0 / profile.total_ms.max(f64::MIN_POSITIVE)
    );
    eprintln!(
        "  {:30} {:>9.3} {:>9.1}%",
        format!("Linear-attn layers ({linear_attn_layers})"),
        linear_attn_ms,
        linear_attn_ms * 100.0 / profile.total_ms.max(f64::MIN_POSITIVE)
    );
    eprintln!(
        "  {:30} {:>9.3} {:>9.1}%",
        "Final norm",
        profile.final_norm_ms,
        profile.final_norm_ms * 100.0 / profile.total_ms.max(f64::MIN_POSITIVE)
    );
    eprintln!(
        "  {:30} {:>9.3} {:>9.1}%",
        "Logits head",
        profile.logits_ms,
        profile.logits_ms * 100.0 / profile.total_ms.max(f64::MIN_POSITIVE)
    );
    eprintln!(
        "  {:30} {:>9.3} {:>9.1}%",
        "Unattributed",
        profile.unattributed_ms(),
        profile.unattributed_ms() * 100.0 / profile.total_ms.max(f64::MIN_POSITIVE)
    );
    eprintln!(
        "  {:30} {:>9.3} {:>9}",
        "Profile total", profile.total_ms, "n/a"
    );
    eprintln!(
        "  Full-attn compiled decode layers: {}  Quantized KV layers: {}",
        compiled_layers, quantized_kv_layers
    );
    if linear_attn_layers > 0 {
        eprintln!();
        eprintln!("  Linear-attn cached step breakdown (forced eval inside GDN attention path):");
        eprintln!("  {:30} {:>10} {:>10}", "Stage", "Time(ms)", "Share");
        print_separator();
        eprintln!(
            "  {:30} {:>9.3} {:>9.1}%",
            "Input layernorm",
            linear_stage_totals.input_norm_ms,
            linear_stage_totals.input_norm_ms * 100.0 / linear_attn_ms.max(f64::MIN_POSITIVE)
        );
        eprintln!(
            "  {:30} {:>9.3} {:>9.1}%",
            "QKV projection",
            linear_stage_totals.qkv_proj_ms,
            linear_stage_totals.qkv_proj_ms * 100.0 / linear_attn_ms.max(f64::MIN_POSITIVE)
        );
        eprintln!(
            "  {:30} {:>9.3} {:>9.1}%",
            "A/B projections",
            linear_stage_totals.ab_proj_ms,
            linear_stage_totals.ab_proj_ms * 100.0 / linear_attn_ms.max(f64::MIN_POSITIVE)
        );
        eprintln!(
            "  {:30} {:>9.3} {:>9.1}%",
            "Conv1d + buffer",
            linear_stage_totals.conv_ms,
            linear_stage_totals.conv_ms * 100.0 / linear_attn_ms.max(f64::MIN_POSITIVE)
        );
        eprintln!(
            "  {:30} {:>9.3} {:>9.1}%",
            "Q/K norm + repeat",
            linear_stage_totals.qk_norm_ms,
            linear_stage_totals.qk_norm_ms * 100.0 / linear_attn_ms.max(f64::MIN_POSITIVE)
        );
        eprintln!(
            "  {:30} {:>9.3} {:>9.1}%",
            "Decay/beta gates",
            linear_stage_totals.gate_ms,
            linear_stage_totals.gate_ms * 100.0 / linear_attn_ms.max(f64::MIN_POSITIVE)
        );
        eprintln!(
            "  {:30} {:>9.3} {:>9.1}%",
            "Recurrence",
            linear_stage_totals.recurrence_ms,
            linear_stage_totals.recurrence_ms * 100.0 / linear_attn_ms.max(f64::MIN_POSITIVE)
        );
        eprintln!(
            "  {:30} {:>9.3} {:>9.1}%",
            "Z projection",
            linear_stage_totals.z_proj_ms,
            linear_stage_totals.z_proj_ms * 100.0 / linear_attn_ms.max(f64::MIN_POSITIVE)
        );
        eprintln!(
            "  {:30} {:>9.3} {:>9.1}%",
            "Norm/gate/out_proj",
            linear_stage_totals.output_ms,
            linear_stage_totals.output_ms * 100.0 / linear_attn_ms.max(f64::MIN_POSITIVE)
        );
        eprintln!(
            "  {:30} {:>9.3} {:>9.1}%",
            "Attn residual add",
            linear_stage_totals.attn_residual_ms,
            linear_stage_totals.attn_residual_ms * 100.0 / linear_attn_ms.max(f64::MIN_POSITIVE)
        );
        eprintln!(
            "  {:30} {:>9.3} {:>9.1}%",
            "Post-attn layernorm",
            linear_stage_totals.post_attn_norm_ms,
            linear_stage_totals.post_attn_norm_ms * 100.0 / linear_attn_ms.max(f64::MIN_POSITIVE)
        );
        eprintln!(
            "  {:30} {:>9.3} {:>9.1}%",
            "MLP gate_proj",
            linear_stage_totals.mlp_gate_proj_ms,
            linear_stage_totals.mlp_gate_proj_ms * 100.0 / linear_attn_ms.max(f64::MIN_POSITIVE)
        );
        eprintln!(
            "  {:30} {:>9.3} {:>9.1}%",
            "MLP up_proj",
            linear_stage_totals.mlp_up_proj_ms,
            linear_stage_totals.mlp_up_proj_ms * 100.0 / linear_attn_ms.max(f64::MIN_POSITIVE)
        );
        eprintln!(
            "  {:30} {:>9.3} {:>9.1}%",
            "MLP activation + mul",
            linear_stage_totals.mlp_act_mul_ms,
            linear_stage_totals.mlp_act_mul_ms * 100.0 / linear_attn_ms.max(f64::MIN_POSITIVE)
        );
        eprintln!(
            "  {:30} {:>9.3} {:>9.1}%",
            "MLP down_proj",
            linear_stage_totals.mlp_down_proj_ms,
            linear_stage_totals.mlp_down_proj_ms * 100.0 / linear_attn_ms.max(f64::MIN_POSITIVE)
        );
        eprintln!(
            "  {:30} {:>9.3} {:>9.1}%",
            "Final residual add",
            linear_stage_totals.final_residual_ms,
            linear_stage_totals.final_residual_ms * 100.0 / linear_attn_ms.max(f64::MIN_POSITIVE)
        );
        eprintln!(
            "  {:30} {:>9.3} {:>9.1}%",
            "Unprofiled linear layer time",
            linear_unprofiled_ms,
            linear_unprofiled_ms * 100.0 / linear_attn_ms.max(f64::MIN_POSITIVE)
        );
    }
    for layer in hottest_layers.iter().take(3) {
        let cache_note = if layer.kind == CachedDecodeLayerKind::FullAttention {
            format!(
                "kv_len={} kv={} compiled={}",
                layer.cache_len_before,
                if layer.quantized_kv { "quant" } else { "plain" },
                if layer.compiled_decode { "yes" } else { "no" }
            )
        } else {
            "gdn cache".to_string()
        };
        eprintln!(
            "  Hot layer #{:<2} {:<11} {:>9.3}ms  {}",
            layer.layer_idx,
            layer.kind.label(),
            layer.total_ms,
            cache_note
        );
    }
    if let Some(conv_step_bench) = conv_step_bench.as_ref() {
        eprintln!();
        eprintln!("  Linear-attn decode conv microbench (real checkpoint cache state):");
        eprintln!(
            "  {:30} {:>10} {:>10} {:>12}",
            "Variant", "Time(ms)", "Speed", "Notes"
        );
        print_separator();
        eprintln!(
            "  {:30} {:>9.4} {:>10} {:>12}",
            "Current circular-buffer helper",
            conv_step_bench.current_ms,
            "baseline",
            format!("#{}", conv_step_bench.layer_idx)
        );
        eprintln!(
            "  {:30} {:>9.4} {:>10} {:>12}",
            "Reference tiny-window conv1d",
            conv_step_bench.reference_ms,
            compare_latency(conv_step_bench.reference_ms, conv_step_bench.current_ms),
            format!(
                "dim={} hist={}",
                conv_step_bench.conv_dim, conv_step_bench.history_len
            )
        );
        eprintln!(
            "  Conv-step max |diff|: {:.3e}  cache max |diff|: {:.3e}  cache pos match: {}",
            conv_step_bench.max_abs_diff,
            conv_step_bench.cache_max_abs_diff,
            if conv_step_bench.cache_pos_match {
                "yes"
            } else {
                "no"
            }
        );
    }
    if let Some(mlp_bench) = mlp_bench.as_ref() {
        eprintln!();
        eprintln!("  Linear-attn MLP microbench (frozen gate/up fusion spike):");
        eprintln!(
            "  {:30} {:>10} {:>10} {:>12}",
            "Variant", "Time(ms)", "Speed", "Notes"
        );
        print_separator();
        eprintln!(
            "  {:30} {:>9.4} {:>10} {:>12}",
            "Current gate+up+down",
            mlp_bench.separate_ms,
            "baseline",
            format!("#{}", mlp_bench.layer_idx)
        );
        eprintln!(
            "  {:30} {:>9.4} {:>10} {:>12}",
            "Fused gate/up + down",
            mlp_bench.fused_ms,
            compare_latency(mlp_bench.fused_ms, mlp_bench.separate_ms),
            format!("in={} hid={}", mlp_bench.input_dim, mlp_bench.hidden_dim)
        );
        eprintln!("  MLP max |diff|: {:.3e}", mlp_bench.max_abs_diff);
    }
    if let Some(aux_proj_bench) = aux_proj_bench.as_ref() {
        eprintln!();
        eprintln!("  Linear-attn aux projection microbench (frozen A/B/Z fusion spike):");
        eprintln!(
            "  {:30} {:>10} {:>10} {:>12}",
            "Variant", "Time(ms)", "Speed", "Notes"
        );
        print_separator();
        eprintln!(
            "  {:30} {:>9.4} {:>10} {:>12}",
            "Current A + B + Z",
            aux_proj_bench.separate_ms,
            "baseline",
            format!("#{}", aux_proj_bench.layer_idx)
        );
        eprintln!(
            "  {:30} {:>9.4} {:>10} {:>12}",
            "Fused A/B/Z split",
            aux_proj_bench.fused_ms,
            compare_latency(aux_proj_bench.fused_ms, aux_proj_bench.separate_ms),
            format!(
                "in={} a={} b={} z={}",
                aux_proj_bench.input_dim,
                aux_proj_bench.a_dim,
                aux_proj_bench.b_dim,
                aux_proj_bench.z_dim
            )
        );
        eprintln!("  Aux-proj max |diff|: {:.3e}", aux_proj_bench.max_abs_diff);
    }

    Some(CheckpointDecodeBenchResult {
        model_dir,
        prompt_tokens: prompt_tokens.len(),
        decode_steps: options.decode_steps,
        prefill_ms,
        cached_step_ms,
        no_cache_next_ms,
        first_step_max_diff,
        full_generate_ms,
        compiled_decode: mlx_compiled_decode_enabled(),
        kv_cache_bits: model.kv_cache_config.bits,
        kv_cache_group_size: model.kv_cache_config.group_size,
        kv_cache_quantized_start: model.kv_cache_config.quantized_start,
        first_token_id,
        profile,
        conv_step_bench,
        mlp_bench,
        aux_proj_bench,
    })
}

// ---------------------------------------------------------------------------
// Backward hot path benchmark: explicit transpose + SGEMM vs transposed SGEMM
// ---------------------------------------------------------------------------

fn bench_transposed_matmul() -> Vec<TransposedMatmulBenchResult> {
    let mut results = Vec::new();
    let configs = [
        (2048, 5632, 4, "W2^T @ dffn"),
        (5632, 2048, 4, "W1/W3^T @ dh"),
        (2048, 2048, 4, "Wo/Wq/Wk/Wv^T @ d?"),
    ];

    print_header("TRANSPOSE-FREE BACKWARD: explicit transpose + SGEMM vs transposed SGEMM");
    eprintln!(
        "  {:24} {:>11} {:>11} {:>8} {:>10}",
        "Config", "Old(ms)", "New(ms)", "Speedup", "Max|diff|"
    );
    print_separator();

    for &(rows, cols, s, label) in &configs {
        let mut w = vec![0.0f32; rows * cols];
        let mut x = vec![0.0f32; rows * s];
        fill_random(&mut w, 9001);
        fill_random(&mut x, 1337);

        let explicit = {
            let wt = nanobot::agent::ane_weights::transpose_weight(&w, rows, cols);
            nanobot::agent::ane_forward::cpu_matmul(&wt, &x, cols, rows, s)
        };
        let direct = nanobot::agent::ane_forward::cpu_matmul_lhs_transposed(&w, rows, cols, &x, s);
        let max_diff = max_abs_diff(&explicit, &direct);

        let old_ms = measure_average_ms(2, 10, || {
            let wt = nanobot::agent::ane_weights::transpose_weight(&w, rows, cols);
            black_box(nanobot::agent::ane_forward::cpu_matmul(
                &wt, &x, cols, rows, s,
            ));
        });
        let new_ms = measure_average_ms(2, 10, || {
            black_box(nanobot::agent::ane_forward::cpu_matmul_lhs_transposed(
                &w, rows, cols, &x, s,
            ));
        });
        let speedup = old_ms / new_ms;

        eprintln!(
            "  {:24} {:>10.3}ms {:>10.3}ms {:>7.2}x {:>10.3e}",
            label, old_ms, new_ms, speedup, max_diff
        );

        results.push(TransposedMatmulBenchResult {
            label,
            old_ms,
            new_ms,
            max_abs_diff: max_diff,
        });
    }

    results
}

// ---------------------------------------------------------------------------
// Quantized backward hot path: materialized vs blocked transposed matmul
// ---------------------------------------------------------------------------

fn bench_quantized_transposed_matmul() -> Vec<QuantizedTransposedMatmulBenchResult> {
    let mut results = Vec::new();
    let configs = [
        (2048, 5632, 4, "Quantized W2^T @ dffn"),
        (5632, 2048, 4, "Quantized W1/W3^T @ dh"),
        (2048, 2048, 4, "Quantized Wo/Wq^T @ d?"),
    ];

    print_header(
        "QUANTIZED BACKWARD: materialized dense matmul vs blocked quantized transpose matmul",
    );
    eprintln!(
        "  {:24} {:>11} {:>11} {:>8} {:>10}",
        "Config", "Old(ms)", "New(ms)", "Speedup", "Max|diff|"
    );
    print_separator();

    for &(rows, cols, s, label) in &configs {
        let mut w = vec![0.0f32; rows * cols];
        let mut x = vec![0.0f32; rows * s];
        fill_random(&mut w, 2027);
        fill_random(&mut x, 3031);

        let quantized = quantize_affine_u8(&w, rows, cols, QUANTIZED_GROUP_SIZE);
        let dequantized = quantized.dequantize();
        let explicit =
            nanobot::agent::ane_forward::cpu_matmul_lhs_transposed(&dequantized, rows, cols, &x, s);
        let direct =
            nanobot::agent::ane_forward::cpu_quantized_matmul_lhs_transposed(&quantized, &x, s);
        let max_diff = max_abs_diff(&explicit, &direct);

        let old_ms = measure_average_ms(2, 10, || {
            let dense = quantized.dequantize();
            black_box(nanobot::agent::ane_forward::cpu_matmul_lhs_transposed(
                &dense, rows, cols, &x, s,
            ));
        });
        let new_ms = measure_average_ms(2, 10, || {
            black_box(
                nanobot::agent::ane_forward::cpu_quantized_matmul_lhs_transposed(&quantized, &x, s),
            );
        });
        let speedup = old_ms / new_ms;

        eprintln!(
            "  {:24} {:>10.3}ms {:>10.3}ms {:>7.2}x {:>10.3e}",
            label, old_ms, new_ms, speedup, max_diff
        );

        results.push(QuantizedTransposedMatmulBenchResult {
            label,
            old_ms,
            new_ms,
            max_abs_diff: max_diff,
        });
    }

    results
}

// ---------------------------------------------------------------------------
// GDN recurrence benchmark: scalar reference vs production kernel
// ---------------------------------------------------------------------------

fn bench_gdn_recurrence() -> GdnBenchResult {
    print_header("GDN RECURRENCE: validated scalar reference vs production kernel");

    let seq = 64;
    let h_v = 16;
    let d_k = 128;
    let d_v = 128;

    let n_q = h_v * d_k * seq;
    let n_v = h_v * d_v * seq;
    let n_g = h_v * seq;

    let mut q = vec![0.0f32; n_q];
    let mut k = vec![0.0f32; n_q];
    let mut v = vec![0.0f32; n_v];
    let mut g_arr = vec![0.0f32; n_g];
    let mut beta_arr = vec![0.0f32; n_g];
    fill_random(&mut q, 1);
    fill_random(&mut k, 2);
    fill_random(&mut v, 3);
    fill_random(&mut g_arr, 4);
    fill_random(&mut beta_arr, 5);
    for val in g_arr.iter_mut() {
        *val = val.abs().min(0.999);
    }
    for val in beta_arr.iter_mut() {
        *val = 1.0 / (1.0 + (-*val).exp());
    }

    let scalar_ref = scalar_gdn_recurrence(&q, &k, &v, &g_arr, &beta_arr, h_v, d_k, d_v, seq);
    let kernel_ref = nanobot::agent::ane_forward::cpu_gdn_recurrence_bench(
        &q, &k, &v, &g_arr, &beta_arr, h_v, d_k, d_v, seq,
    );
    let max_diff = max_abs_diff(&scalar_ref, &kernel_ref);

    let iters = 5;
    let scalar_ms = measure_average_ms(1, iters, || {
        black_box(scalar_gdn_recurrence(
            &q, &k, &v, &g_arr, &beta_arr, h_v, d_k, d_v, seq,
        ));
    });
    let kernel_ms = measure_average_ms(1, iters, || {
        black_box(nanobot::agent::ane_forward::cpu_gdn_recurrence_bench(
            &q, &k, &v, &g_arr, &beta_arr, h_v, d_k, d_v, seq,
        ));
    });

    let flops_per_token = 7.0 * h_v as f64 * d_v as f64 * d_k as f64;
    let total_flops = flops_per_token * seq as f64;
    let scalar_gflops = total_flops / (scalar_ms / 1000.0) / 1e9;
    let kernel_gflops = total_flops / (kernel_ms / 1000.0) / 1e9;
    let speedup = scalar_ms / kernel_ms;

    let state_kb = h_v * d_v * d_k * 4 / 1024;
    let kernel_label = if cfg!(target_arch = "aarch64") {
        "Production kernel (NEON)"
    } else {
        "Production kernel (scalar)"
    };

    eprintln!("  Dims:              h_v={h_v}, d_k={d_k}, d_v={d_v}, seq={seq}");
    eprintln!("  State per layer:   {} KB", state_kb);
    eprintln!("  Max |diff| vs scalar reference: {:.3e}", max_diff);
    eprintln!();
    eprintln!(
        "  {:24} {:>10} {:>10} {:>8}",
        "", "Time(ms)", "GFLOPS", "tok/ms"
    );
    print_separator();
    eprintln!(
        "  {:24} {:>9.2}ms {:>9.2} {:>7.1}",
        "Scalar reference",
        scalar_ms,
        scalar_gflops,
        seq as f64 / scalar_ms
    );
    eprintln!(
        "  {:24} {:>9.2}ms {:>9.2} {:>7.1}",
        kernel_label,
        kernel_ms,
        kernel_gflops,
        seq as f64 / kernel_ms
    );
    eprintln!("  Speedup:           {:.1}x", speedup);
    eprintln!();
    eprintln!("  18 GDN layers (scalar):  {:.1} ms", scalar_ms * 18.0);
    eprintln!("  18 GDN layers (kernel):  {:.1} ms", kernel_ms * 18.0);

    GdnBenchResult {
        scalar_ms,
        kernel_ms,
        scalar_gflops,
        kernel_gflops,
        max_abs_diff: max_diff,
    }
}

// ---------------------------------------------------------------------------
// Memory bandwidth benchmark
// ---------------------------------------------------------------------------

fn streaming_read_pass(data: &[f32]) -> f32 {
    let mut sums = [0.0f32; 8];
    let mut chunks = data.chunks_exact(8);
    for chunk in &mut chunks {
        sums[0] += chunk[0];
        sums[1] += chunk[1];
        sums[2] += chunk[2];
        sums[3] += chunk[3];
        sums[4] += chunk[4];
        sums[5] += chunk[5];
        sums[6] += chunk[6];
        sums[7] += chunk[7];
    }
    let mut total = sums.into_iter().sum::<f32>();
    for &v in chunks.remainder() {
        total += v;
    }
    black_box(total)
}

fn bench_memory_bandwidth(options: &Options) -> MemoryBenchResult {
    print_header("MEMORY BANDWIDTH (measured proxies vs assumed peak)");

    let size_mb = options.memory_size_mb;
    let n = size_mb * 1024 * 1024 / 4;
    let mut src = vec![0.0f32; n];
    let mut dst = vec![0.0f32; n];
    fill_random(&mut src, 999);

    let iters = 5;
    black_box(streaming_read_pass(&src));
    let read_ms = measure_average_ms(1, iters, || {
        black_box(streaming_read_pass(&src));
    });
    let bytes = size_mb as f64 * 1024.0 * 1024.0;
    let read_gbps = bytes / (read_ms / 1000.0) / 1e9;

    dst.copy_from_slice(&src);
    let copy_ms = measure_average_ms(1, iters, || {
        dst.copy_from_slice(&src);
        black_box(dst[0]);
    });
    let copy_gbps = (2.0 * bytes) / (copy_ms / 1000.0) / 1e9;

    eprintln!("  Buffer size:       {} MB", size_mb);
    eprintln!(
        "  {:24} {:>10} {:>10} {:>10}",
        "Kernel", "Time(ms)", "GB/s", "vs peak"
    );
    print_separator();
    eprintln!(
        "  {:24} {:>9.2} {:>10.1} {:>9.1}%",
        "Streaming read (8 accum)",
        read_ms,
        read_gbps,
        read_gbps / options.peak_bandwidth_gbps * 100.0
    );
    eprintln!(
        "  {:24} {:>9.2} {:>10.1} {:>9.1}%",
        "memcpy-style copy",
        copy_ms,
        copy_gbps,
        copy_gbps / options.peak_bandwidth_gbps * 100.0
    );
    eprintln!(
        "  Assumed peak DRAM BW: {:.1} GB/s (--peak-bandwidth-gbps overrides)",
        options.peak_bandwidth_gbps
    );
    eprintln!("  Note: copy bandwidth counts both read and write traffic.");

    MemoryBenchResult {
        size_mb,
        read_ms,
        read_gbps,
        copy_ms,
        copy_gbps,
    }
}

// ---------------------------------------------------------------------------
// Inference throughput estimate
// ---------------------------------------------------------------------------

fn estimate_inference(
    matmul_results: &[MatmulBenchResult],
    quantized_results: &[QuantizedBenchResult],
    memory: &MemoryBenchResult,
    options: &Options,
) {
    print_header("ROOFLINE ESTIMATE (measured kernels + explicit assumptions)");

    let sgemm_qkv = matmul_results
        .iter()
        .find(|r| r.label == "QKV proj (dim x dim, seq=4)")
        .expect("QKV result missing");
    let sgemm_gate = matmul_results
        .iter()
        .find(|r| r.label == "FFN gate (hidden x dim, seq=4)")
        .expect("FFN gate result missing");
    let sgemm_down = matmul_results
        .iter()
        .find(|r| r.label == "FFN down (dim x hidden, seq=4)")
        .expect("FFN down result missing");
    let quantized_qkv = quantized_results
        .iter()
        .find(|r| r.label == "QKV proj (dim x dim, seq=4)")
        .expect("quantized QKV result missing");
    let quantized_gate = quantized_results
        .iter()
        .find(|r| r.label == "FFN gate (hidden x dim, seq=4)")
        .expect("quantized FFN gate result missing");
    let quantized_down = quantized_results
        .iter()
        .find(|r| r.label == "FFN down (dim x hidden, seq=4)")
        .expect("quantized FFN down result missing");

    let mha_layer_ms =
        4.0 * sgemm_qkv.elapsed_ms + 2.0 * sgemm_gate.elapsed_ms + sgemm_down.elapsed_ms;
    let matmul_only_token_ms = mha_layer_ms * options.layers as f64;
    let matmul_only_tok_s = 1000.0 / matmul_only_token_ms;
    let quantized_layer_ms = 4.0 * quantized_qkv.pipeline_ms
        + 2.0 * quantized_gate.pipeline_ms
        + quantized_down.pipeline_ms;
    let quantized_token_ms = quantized_layer_ms * options.layers as f64;
    let quantized_tok_s = 1000.0 / quantized_token_ms;

    let measured_read_tok_s = memory.read_gbps / options.model_size_gb;
    let measured_copy_tok_s = memory.copy_gbps / options.model_size_gb;
    let theoretical_tok_s = options.peak_bandwidth_gbps / options.model_size_gb;

    let ceilings = [
        ("Dense SGEMM-only ceiling", matmul_only_tok_s),
        ("Current quantized pipeline ceiling", quantized_tok_s),
        ("Measured streaming-read roofline", measured_read_tok_s),
        ("Measured copy roofline proxy", measured_copy_tok_s),
        ("Assumed peak-bandwidth roofline", theoretical_tok_s),
    ];
    let (tightest_name, tightest_tok_s) = ceilings
        .iter()
        .copied()
        .min_by(|lhs, rhs| {
            lhs.1
                .partial_cmp(&rhs.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("at least one ceiling");

    eprintln!(
        "  Assumed model size:          {:.2} GB (--model-size-gb overrides)",
        options.model_size_gb
    );
    eprintln!(
        "  Assumed layer count:         {} (--layers overrides)",
        options.layers
    );
    eprintln!(
        "  Assumed peak bandwidth:      {:.1} GB/s (--peak-bandwidth-gbps overrides)",
        options.peak_bandwidth_gbps
    );
    eprintln!();
    eprintln!("  MHA layer matmul cost:       {:.2} ms", mha_layer_ms);
    eprintln!(
        "  Dense SGEMM token ceiling:   {:.1} tok/s ({:.1} ms/token)",
        matmul_only_tok_s, matmul_only_token_ms
    );
    eprintln!(
        "  Quantized pipeline ceiling:  {:.1} tok/s ({:.1} ms/token)",
        quantized_tok_s, quantized_token_ms
    );
    eprintln!(
        "  Streaming-read roofline:     {:.1} tok/s (measured {} MB stream)",
        measured_read_tok_s, memory.size_mb
    );
    eprintln!(
        "  Copy roofline proxy:         {:.1} tok/s (memcpy-style stream)",
        measured_copy_tok_s
    );
    eprintln!(
        "  Peak-bandwidth roofline:     {:.1} tok/s (assumed)",
        theoretical_tok_s
    );
    eprintln!();
    eprintln!(
        "  Tightest modeled ceiling:    {tightest_name} @ {:.1} tok/s",
        tightest_tok_s
    );
    eprintln!(
        "  Current quantized path vs peak roofline: {:.1}%",
        quantized_tok_s / theoretical_tok_s * 100.0
    );
    eprintln!();
    eprintln!("  Included in current quantized ceiling:");
    eprintln!(
        "    - synthetic QuantizedTensor blocked quantized matmul without full-layer materialization"
    );
    eprintln!("  Not included in token ceilings:");
    eprintln!("    - production one-layer quantized MHA path (benchmarked separately below)");
    eprintln!("    - hybrid GDN layer mix (benchmarked separately, not modeled here)");
    eprintln!("    - RMSNorm, RoPE, SDPA, conv1d, gating, logits, sampling");
    eprintln!("    - KV-cache traffic and scheduler/runtime overhead");
    eprintln!("    - checkpoint-backed quantized weights and any extra runtime packing costs");
}

fn print_missing_pieces(checkpoint_backed_covered: bool, decode_covered: bool) {
    print_header("COVERAGE AND MISSING PIECES");
    eprintln!("  Benchmark coverage:");
    eprintln!("    [x] f32 SGEMM kernel benchmark");
    eprintln!("    [x] blocked quantized matmul benchmark via QuantizedTensor");
    eprintln!("    [x] production one-layer quantized MHA forward benchmark");
    eprintln!("    [x] blocked quantized GDN forward benchmark via production path");
    if checkpoint_backed_covered {
        eprintln!("    [x] checkpoint-backed quantized forward benchmark");
    }
    if decode_covered {
        eprintln!("    [x] checkpoint-backed end-to-end decode benchmark");
    }
    eprintln!("    [x] transpose-free backward matmul benchmark");
    eprintln!("    [x] quantized transpose-free backward matmul benchmark");
    eprintln!("    [x] measured streaming-read and copy bandwidth proxies");
    eprintln!();
    eprintln!("  Benchmark gaps:");
    if !checkpoint_backed_covered {
        eprintln!(
            "    [ ] checkpoint-backed quantized benchmark (not just synthetic grouped-u8 weights)"
        );
    }
    if !decode_covered {
        eprintln!("    [ ] end-to-end single-token decode benchmark with KV-cache");
    }
    eprintln!("    [ ] separate accounting for RoPE, RMSNorm, SDPA, conv1d, and sampling");
    eprintln!("    [ ] machine-readable output for regression tracking in CI");
    eprintln!();
    eprintln!("  Optimization gaps:");
    eprintln!(
        "    [ ] broader fused dequant + matmul coverage in the live training/inference paths"
    );
    eprintln!("    [ ] weight prepacking / persistent packed layouts for SGEMM");
    eprintln!("    [ ] fused norm/activation epilogues to cut memory traffic");
    eprintln!("    [ ] ANE conv1x1 and more operator offload beyond current CPU kernels");
    eprintln!("    [ ] overlap layer N+1 dequant with layer N compute");
    eprintln!("    [ ] tighter KV-cache-aware decode kernels and cache traffic analysis");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let options = Options::parse();

    eprintln!();
    eprintln!("  PERFORMANCE CEILING BENCHMARK");
    eprintln!("  Hardware: Apple Silicon / current host");
    eprintln!("  Model reference: Qwen3.5-2B-style shapes (dim=2048, hidden=5632)");

    let matmul_results = bench_matmul();
    let quantized_results = bench_quantized_path(&matmul_results);
    let quantized_layer_forward_result = bench_quantized_layer_forward_path();
    let quantized_gdn_result = bench_quantized_gdn_path();
    #[cfg(feature = "mlx")]
    let checkpoint_bench = bench_checkpoint_quantized_forward_path(&options);
    #[cfg(not(feature = "mlx"))]
    let checkpoint_bench: Option<()> = None;
    #[cfg(feature = "mlx")]
    let checkpoint_decode_bench = bench_checkpoint_decode_path(&options);
    #[cfg(not(feature = "mlx"))]
    let checkpoint_decode_bench: Option<()> = None;
    let transposed_results = bench_transposed_matmul();
    let quantized_transposed_results = bench_quantized_transposed_matmul();
    let gdn_result = bench_gdn_recurrence();
    let memory_result = bench_memory_bandwidth(&options);
    estimate_inference(
        &matmul_results,
        &quantized_results,
        &memory_result,
        &options,
    );
    print_missing_pieces(
        checkpoint_bench.is_some(),
        checkpoint_decode_bench.is_some(),
    );

    let best_sgemm_gflops = matmul_results
        .iter()
        .map(|r| r.gflops)
        .fold(0.0f64, f64::max);
    let avg_sgemm_gflops =
        matmul_results.iter().map(|r| r.gflops).sum::<f64>() / matmul_results.len() as f64;
    let avg_quantized_overhead = quantized_results
        .iter()
        .map(|r| r.pipeline_ms / r.dense_ms)
        .sum::<f64>()
        / quantized_results.len() as f64;
    let avg_quantized_materialized_overhead = quantized_results
        .iter()
        .map(|r| r.materialized_ms / r.dense_ms)
        .sum::<f64>()
        / quantized_results.len() as f64;
    let avg_quantized_speedup = quantized_results
        .iter()
        .map(|r| r.materialized_ms / r.pipeline_ms)
        .sum::<f64>()
        / quantized_results.len() as f64;
    let avg_quantized_compression = quantized_results
        .iter()
        .map(|r| r.compression_ratio)
        .sum::<f64>()
        / quantized_results.len() as f64;
    let max_quantized_diff = quantized_results
        .iter()
        .map(|r| r.max_abs_diff)
        .fold(0.0f32, f32::max);
    let avg_transpose_speedup = transposed_results
        .iter()
        .map(|r| r.old_ms / r.new_ms)
        .sum::<f64>()
        / transposed_results.len() as f64;
    let avg_quantized_transpose_speedup = quantized_transposed_results
        .iter()
        .map(|r| r.old_ms / r.new_ms)
        .sum::<f64>()
        / quantized_transposed_results.len() as f64;
    let max_transposed_diff = transposed_results
        .iter()
        .map(|r| r.max_abs_diff)
        .fold(0.0f32, f32::max);
    let max_quantized_transposed_diff = quantized_transposed_results
        .iter()
        .map(|r| r.max_abs_diff)
        .fold(0.0f32, f32::max);
    let slowest_transpose_case = transposed_results
        .iter()
        .max_by(|lhs, rhs| {
            lhs.new_ms
                .partial_cmp(&rhs.new_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|r| r.label)
        .unwrap_or("n/a");
    let slowest_quantized_transpose_case = quantized_transposed_results
        .iter()
        .max_by(|lhs, rhs| {
            lhs.new_ms
                .partial_cmp(&rhs.new_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|r| r.label)
        .unwrap_or("n/a");

    print_header("OPTIMIZATION STATUS");
    eprintln!("  [x] cpu_matmul -> Accelerate SGEMM");
    eprintln!("  [x] current quantized path benchmarked via blocked quantized matmul");
    eprintln!(
        "  [x] current one-layer quantized MHA forward path benchmarked via production forward"
    );
    eprintln!("  [x] current quantized GDN path benchmarked via production forward");
    if checkpoint_bench.is_some() {
        eprintln!("  [x] checkpoint-backed quantized forward benchmark executed");
    }
    if checkpoint_decode_bench.is_some() {
        eprintln!("  [x] checkpoint-backed end-to-end decode benchmark executed");
        if checkpoint_decode_bench
            .as_ref()
            .and_then(|bench| bench.conv_step_bench.as_ref())
            .is_some()
        {
            eprintln!("  [x] checkpoint-backed linear-attn conv-step microbench executed");
        }
        if checkpoint_decode_bench
            .as_ref()
            .and_then(|bench| bench.mlp_bench.as_ref())
            .is_some()
        {
            eprintln!("  [x] checkpoint-backed linear-attn MLP microbench executed");
        }
        if checkpoint_decode_bench
            .as_ref()
            .and_then(|bench| bench.aux_proj_bench.as_ref())
            .is_some()
        {
            eprintln!("  [x] checkpoint-backed linear-attn aux-projection microbench executed");
        }
    }
    eprintln!(
        "  [x] transpose-free backward matmul path benchmarked against old transpose + SGEMM path"
    );
    eprintln!("  [x] GDN recurrence benchmark now times the production kernel");
    eprintln!(
        "  [x] SGEMM kernels: avg {:.1} GFLOPS, best {:.1} GFLOPS",
        avg_sgemm_gflops, best_sgemm_gflops
    );
    eprintln!(
        "  [x] quantized path: avg {:.2}x faster than old materialize+SGEMM, {:.2}x slower than dense SGEMM, old path was {:.2}x slower than dense, avg compression {:.2}x",
        avg_quantized_speedup,
        avg_quantized_overhead,
        avg_quantized_materialized_overhead,
        avg_quantized_compression
    );
    eprintln!(
        "  [x] one-layer quantized MHA forward path: {:.2}x faster than old materialize+forward, {:.2}x slower than dense forward, compression {:.2}x",
        quantized_layer_forward_result.materialized_ms / quantized_layer_forward_result.pipeline_ms,
        quantized_layer_forward_result.pipeline_ms / quantized_layer_forward_result.dense_ms,
        quantized_layer_forward_result.compression_ratio
    );
    #[cfg(feature = "mlx")]
    if let Some(checkpoint_bench) = &checkpoint_bench {
        eprintln!(
            "  [x] checkpoint source: {}",
            checkpoint_bench.model_dir.display()
        );
        for case in &checkpoint_bench.cases {
            eprintln!(
                "  [x] checkpoint-backed {}: {:.2}x faster than old materialize+forward, {:.2}x slower than dense forward, compression {:.2}x",
                case.label,
                case.materialized_ms / case.pipeline_ms,
                case.pipeline_ms / case.dense_ms,
                case.compression_ratio
            );
        }
    }
    #[cfg(feature = "mlx")]
    if let Some(decode_bench) = &checkpoint_decode_bench {
        let kv_cache = match decode_bench.kv_cache_bits {
            Some(bits) => format!(
                "{bits}-bit group_size={} start={}",
                decode_bench.kv_cache_group_size, decode_bench.kv_cache_quantized_start
            ),
            None => "plain fp16/f32".to_string(),
        };
        let cached_tok_s = 1000.0 / decode_bench.cached_step_ms.max(f64::MIN_POSITIVE);
        let full_generate_tok_s =
            (decode_bench.decode_steps + 1) as f64 / (decode_bench.full_generate_ms / 1000.0);
        let prefill_tok_s = decode_bench.prompt_tokens as f64 / (decode_bench.prefill_ms / 1000.0);
        let full_attn_ms = decode_bench
            .profile
            .kind_total_ms(nanobot::agent::mlx_lora::CachedDecodeLayerKind::FullAttention);
        let linear_attn_ms = decode_bench
            .profile
            .kind_total_ms(nanobot::agent::mlx_lora::CachedDecodeLayerKind::LinearAttention);
        let linear_stage_totals = decode_bench.profile.linear_decode_stage_totals();
        let (linear_attention_bottleneck_label, linear_attention_bottleneck_ms) = [
            ("Input layernorm", linear_stage_totals.input_norm_ms),
            ("QKV projection", linear_stage_totals.qkv_proj_ms),
            ("A/B projections", linear_stage_totals.ab_proj_ms),
            ("Conv1d + buffer", linear_stage_totals.conv_ms),
            ("Q/K norm + repeat", linear_stage_totals.qk_norm_ms),
            ("Decay/beta gates", linear_stage_totals.gate_ms),
            ("Recurrence", linear_stage_totals.recurrence_ms),
            ("Z projection", linear_stage_totals.z_proj_ms),
            ("Norm/gate/out_proj", linear_stage_totals.output_ms),
            ("Attn residual add", linear_stage_totals.attn_residual_ms),
            ("Post-attn layernorm", linear_stage_totals.post_attn_norm_ms),
            ("MLP gate_proj", linear_stage_totals.mlp_gate_proj_ms),
            ("MLP up_proj", linear_stage_totals.mlp_up_proj_ms),
            ("MLP activation + mul", linear_stage_totals.mlp_act_mul_ms),
            ("MLP down_proj", linear_stage_totals.mlp_down_proj_ms),
            ("Final residual add", linear_stage_totals.final_residual_ms),
        ]
        .into_iter()
        .max_by(|lhs, rhs| {
            lhs.1
                .partial_cmp(&rhs.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(("n/a", 0.0));
        let hottest_layer = sort_decode_layers_by_time(&decode_bench.profile)
            .into_iter()
            .next()
            .map(|layer| {
                format!(
                    "#{} {} {:.1}ms",
                    layer.layer_idx,
                    layer.kind.label(),
                    layer.total_ms
                )
            })
            .unwrap_or_else(|| "n/a".to_string());
        eprintln!(
            "  [x] checkpoint-backed decode source: {}",
            decode_bench.model_dir.display()
        );
        eprintln!(
            "  [x] checkpoint-backed decode: prefill {:.1} tok/s, cached decode {:.1} tok/s, end-to-end generate {:.1} tok/s, cached step {} than no-cache next-token baseline, first-step max |diff| {:.3e}, compiled decode {}, KV cache {}, first token {}",
            prefill_tok_s,
            cached_tok_s,
            full_generate_tok_s,
            compare_latency(decode_bench.cached_step_ms, decode_bench.no_cache_next_ms),
            decode_bench.first_step_max_diff,
            if decode_bench.compiled_decode {
                "on"
            } else {
                "off"
            },
            kv_cache,
            decode_bench.first_token_id
        );
        eprintln!(
            "  [x] cached decode profile: forced-eval step {:.1}ms with full-attn {:.1}ms, linear-attn {:.1}ms, largest linear-attn substage {} {:.1}ms, hottest layer {}",
            decode_bench.profile.total_ms,
            full_attn_ms,
            linear_attn_ms,
            linear_attention_bottleneck_label,
            linear_attention_bottleneck_ms,
            hottest_layer
        );
        if let Some(conv_step_bench) = &decode_bench.conv_step_bench {
            eprintln!(
                "  [x] linear-attn conv step microbench: current helper {:.4} ms, tiny-window conv1d {:.4} ms ({}), max |diff| {:.3e}, cache max |diff| {:.3e}, cache pos match {}",
                conv_step_bench.current_ms,
                conv_step_bench.reference_ms,
                compare_latency(conv_step_bench.reference_ms, conv_step_bench.current_ms),
                conv_step_bench.max_abs_diff,
                conv_step_bench.cache_max_abs_diff,
                if conv_step_bench.cache_pos_match {
                    "yes"
                } else {
                    "no"
                }
            );
        }
        if let Some(mlp_bench) = &decode_bench.mlp_bench {
            eprintln!(
                "  [x] linear-attn MLP microbench: current gate+up+down {:.4} ms, fused gate/up + down {:.4} ms ({}), max |diff| {:.3e}",
                mlp_bench.separate_ms,
                mlp_bench.fused_ms,
                compare_latency(mlp_bench.fused_ms, mlp_bench.separate_ms),
                mlp_bench.max_abs_diff
            );
        }
        if let Some(aux_proj_bench) = &decode_bench.aux_proj_bench {
            eprintln!(
                "  [x] linear-attn aux-projection microbench: current A+B+Z {:.4} ms, fused A/B/Z {:.4} ms ({}), max |diff| {:.3e}",
                aux_proj_bench.separate_ms,
                aux_proj_bench.fused_ms,
                compare_latency(aux_proj_bench.fused_ms, aux_proj_bench.separate_ms),
                aux_proj_bench.max_abs_diff
            );
        }
    }
    eprintln!(
        "  [x] quantized GDN path: {:.2}x faster than old materialize+run, {:.2}x slower than dense GDN, compression {:.2}x",
        quantized_gdn_result.materialized_ms / quantized_gdn_result.pipeline_ms,
        quantized_gdn_result.pipeline_ms / quantized_gdn_result.dense_ms,
        quantized_gdn_result.compression_ratio
    );
    eprintln!(
        "  [x] transpose-free backward matmul: avg {:.2}x speedup, slowest case {}",
        avg_transpose_speedup, slowest_transpose_case
    );
    eprintln!(
        "  [x] quantized backward matmul: avg {:.2}x speedup vs materialized path, slowest case {}",
        avg_quantized_transpose_speedup, slowest_quantized_transpose_case
    );
    eprintln!(
        "  [x] GDN kernel: {:.1}x speedup ({:.2} -> {:.2} ms, {:.2} -> {:.2} GFLOPS)",
        gdn_result.scalar_ms / gdn_result.kernel_ms,
        gdn_result.scalar_ms,
        gdn_result.kernel_ms,
        gdn_result.scalar_gflops,
        gdn_result.kernel_gflops
    );
    eprintln!(
        "  [x] correctness validation: max |diff| GDN kernel vs scalar = {:.3e}",
        gdn_result.max_abs_diff
    );
    eprintln!(
        "  [x] correctness validation: max |diff| quantized path = {:.3e}, one-layer quantized MHA = {:.3e}, quantized GDN path = {:.3e}, transpose-free path = {:.3e}, quantized transpose-free path = {:.3e}",
        max_quantized_diff,
        quantized_layer_forward_result.max_abs_diff,
        quantized_gdn_result.max_abs_diff,
        max_transposed_diff,
        max_quantized_transposed_diff
    );
    #[cfg(feature = "mlx")]
    if let Some(checkpoint_bench) = &checkpoint_bench {
        let max_checkpoint_diff = checkpoint_bench
            .cases
            .iter()
            .map(|case| case.max_abs_diff)
            .fold(0.0f32, f32::max);
        eprintln!(
            "  [x] correctness validation: max |diff| checkpoint-backed forward = {:.3e}",
            max_checkpoint_diff
        );
    }
    eprintln!(
        "  [x] measured memory proxies: read {:.1} GB/s ({:.2} ms), copy {:.1} GB/s ({:.2} ms)",
        memory_result.read_gbps,
        memory_result.read_ms,
        memory_result.copy_gbps,
        memory_result.copy_ms
    );
    eprintln!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_abs_diff() {
        let a = [1.0f32, -2.0, 3.0];
        let b = [1.25f32, -2.0, 2.5];
        assert!((max_abs_diff(&a, &b) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_streaming_read_pass_matches_sum() {
        let data: Vec<f32> = (0..33).map(|i| i as f32 * 0.5).collect();
        let expected: f32 = data.iter().sum();
        let got = streaming_read_pass(&data);
        assert!((got - expected).abs() < 1e-4);
    }

    #[test]
    fn test_parse_helpers() {
        assert_eq!(parse_usize("42").unwrap(), 42);
        assert!((parse_f64("2.5").unwrap() - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_quantize_affine_u8_round_trips_group_endpoints() {
        let dense = vec![0.0f32, 255.0, -10.0, 10.0];
        let quantized = quantize_affine_u8(&dense, 1, 4, 2);
        let dequantized = quantized.dequantize();
        assert_eq!(quantized.rows, 1);
        assert_eq!(quantized.cols, 4);
        assert!(max_abs_diff(&dense, &dequantized) < 1e-5);
    }
}
