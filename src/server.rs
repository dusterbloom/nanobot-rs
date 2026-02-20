#![allow(dead_code)]
//! Local LLM utilities: health checks, GGUF parser, model listing, context sizing.
//!
//! Server lifecycle (spawning, process management) is handled by LM Studio via
//! the `lms` module. This module provides shared utilities used across the codebase.

use std::collections::HashMap;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Command;

use tracing::debug;

// ============================================================================
// Constants
// ============================================================================

pub(crate) const DEFAULT_LOCAL_MODEL: &str = "NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf";

// ============================================================================
// GGUF Metadata
// ============================================================================

pub(crate) struct GgufModelInfo {
    pub n_layers: u32,
    pub n_kv_heads: u32,
    pub n_heads: u32,
    pub embedding_dim: u32,
    pub context_length: u32,
}

// ============================================================================
// Port & Model Utilities
// ============================================================================

/// Find an available TCP port starting from `start`.
///
/// Scans ports in the range `[start, start+99]` and returns the first one
/// that can be bound. Falls back to `start` if all are occupied.
pub(crate) fn find_available_port(start: u16) -> u16 {
    for port in start..=start.saturating_add(99) {
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return port;
        }
    }
    start // fallback
}

/// List all GGUF models in `~/models/`, sorted by filename.
pub(crate) fn list_local_models() -> Vec<PathBuf> {
    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return vec![],
    };
    let models_dir = home.join("models");
    let mut models: Vec<PathBuf> = std::fs::read_dir(&models_dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("gguf"))
        .collect();
    models.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    models
}

/// Resolve a local model path given a filename or absolute/relative path.
pub(crate) fn resolve_local_model_path(name: &str) -> Option<PathBuf> {
    if name.trim().is_empty() {
        return None;
    }

    let expanded = if name.starts_with("~/") || name == "~" {
        if let Some(home) = dirs::home_dir() {
            let without_tilde = name.trim_start_matches("~/");
            if without_tilde.is_empty() {
                home
            } else {
                home.join(without_tilde)
            }
        } else {
            PathBuf::from(name)
        }
    } else {
        PathBuf::from(name)
    };

    if expanded.exists() {
        return Some(expanded);
    }

    if let Some(home) = dirs::home_dir() {
        let candidate = home.join("models").join(name);
        if candidate.exists() {
            return Some(candidate);
        }
    }

    None
}

// ============================================================================
// GGUF Parser
// ============================================================================

/// Parse architecture-specific metadata from a GGUF file header.
pub(crate) fn parse_gguf_metadata(path: &Path) -> Option<GgufModelInfo> {
    use std::io::{Read, Seek, SeekFrom};

    let mut f = std::fs::File::open(path).ok()?;
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    // Magic "GGUF"
    f.read_exact(&mut buf4).ok()?;
    if &buf4 != b"GGUF" {
        return None;
    }

    // Version (u32 LE) — we support v2 and v3
    f.read_exact(&mut buf4).ok()?;
    let version = u32::from_le_bytes(buf4);
    if version < 2 {
        return None;
    }

    // tensor_count (u64), kv_count (u64)
    f.read_exact(&mut buf8).ok()?;
    let _tensor_count = u64::from_le_bytes(buf8);
    f.read_exact(&mut buf8).ok()?;
    let kv_count = u64::from_le_bytes(buf8);

    fn gguf_read_string(f: &mut std::fs::File) -> Option<String> {
        let mut b8 = [0u8; 8];
        f.read_exact(&mut b8).ok()?;
        let len = u64::from_le_bytes(b8) as usize;
        if len > 256 {
            f.seek(SeekFrom::Current(len as i64)).ok()?;
            return Some(String::new());
        }
        let mut s = vec![0u8; len];
        f.read_exact(&mut s).ok()?;
        String::from_utf8(s).ok()
    }

    fn gguf_skip_value(f: &mut std::fs::File, vtype: u32) -> Option<()> {
        match vtype {
            0 | 1 | 7 => {
                let mut b = [0u8; 1];
                f.read_exact(&mut b).ok()?;
            }
            2 | 3 => {
                let mut b = [0u8; 2];
                f.read_exact(&mut b).ok()?;
            }
            4 | 5 | 6 => {
                let mut b = [0u8; 4];
                f.read_exact(&mut b).ok()?;
            }
            8 => {
                gguf_read_string(f)?;
            }
            9 => {
                let mut tb = [0u8; 4];
                f.read_exact(&mut tb).ok()?;
                let elem_type = u32::from_le_bytes(tb);
                let mut cb = [0u8; 8];
                f.read_exact(&mut cb).ok()?;
                let count = u64::from_le_bytes(cb);
                for _ in 0..count {
                    gguf_skip_value(f, elem_type)?;
                }
            }
            10 | 11 | 12 => {
                let mut b = [0u8; 8];
                f.read_exact(&mut b).ok()?;
            }
            _ => return None,
        }
        Some(())
    }

    let mut arch = String::new();
    let mut n_layers: Option<u32> = None;
    let mut n_kv_heads: Option<u32> = None;
    let mut n_heads: Option<u32> = None;
    let mut embedding_dim: Option<u32> = None;
    let mut context_length: Option<u32> = None;

    for _ in 0..kv_count {
        let key = match gguf_read_string(&mut f) {
            Some(k) => k,
            None => return None,
        };

        // Read value type
        f.read_exact(&mut buf4).ok()?;
        let vtype = u32::from_le_bytes(buf4);

        if key == "general.architecture" && vtype == 8 {
            arch = gguf_read_string(&mut f)?;
            continue;
        }

        // Check for u32 metadata fields (type 4 = u32, type 5 = i32)
        if (vtype == 4 || vtype == 5) && !arch.is_empty() {
            let mut vb = [0u8; 4];
            f.read_exact(&mut vb).ok()?;
            let val = u32::from_le_bytes(vb);
            if key == format!("{}.block_count", arch) {
                n_layers = Some(val);
            } else if key == format!("{}.attention.head_count_kv", arch) {
                n_kv_heads = Some(val);
            } else if key == format!("{}.attention.head_count", arch) {
                n_heads = Some(val);
            } else if key == format!("{}.embedding_length", arch) {
                embedding_dim = Some(val);
            } else if key == format!("{}.context_length", arch) {
                context_length = Some(val);
            }
            continue;
        }

        // Skip values we don't need
        gguf_skip_value(&mut f, vtype)?;
    }

    Some(GgufModelInfo {
        n_layers: n_layers?,
        n_kv_heads: n_kv_heads?,
        n_heads: n_heads?,
        embedding_dim: embedding_dim?,
        context_length: context_length?,
    })
}

// ============================================================================
// Memory & Context Sizing
// ============================================================================

/// Detect available VRAM (via nvidia-smi) and RAM (via /proc/meminfo).
/// Returns (vram_bytes, ram_bytes).
pub(crate) fn detect_available_memory() -> (Option<u64>, u64) {
    let vram = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|out| {
            if !out.status.success() {
                return None;
            }
            let s = String::from_utf8_lossy(&out.stdout);
            s.trim().lines().next()?.trim().parse::<u64>().ok()
        })
        .map(|mib| mib * 1024 * 1024);

    let ram = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|contents| {
            for line in contents.lines() {
                if line.starts_with("MemAvailable:") {
                    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
                    return Some(kb * 1024);
                }
            }
            None
        })
        .unwrap_or(8 * 1024 * 1024 * 1024); // 8 GB fallback

    (vram, ram)
}

/// Practical context cap based on model file size (proxy for parameter count).
pub(crate) fn practical_context_cap(model_file_size_bytes: u64) -> usize {
    let gb = model_file_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    if gb < 2.0 {
        8192
    } else if gb < 4.0 {
        16384
    } else if gb < 8.0 {
        32768
    } else if gb < 16.0 {
        65536
    } else {
        usize::MAX
    }
}

/// Compute optimal context size for a GGUF model given available system resources.
pub(crate) fn compute_optimal_context_size(model_path: &Path) -> usize {
    const OVERHEAD: u64 = 512 * 1024 * 1024;
    const FALLBACK_CTX: usize = 16384;

    let model_file_size = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);

    let gguf = match parse_gguf_metadata(model_path) {
        Some(info) => info,
        None => {
            let cap = practical_context_cap(model_file_size).min(FALLBACK_CTX);
            debug!(
                "GGUF parse failed for {}, using {}K context (file size: {:.1}GB)",
                model_path.display(),
                cap / 1024,
                model_file_size as f64 / 1e9
            );
            return cap;
        }
    };

    let head_dim = gguf.embedding_dim / gguf.n_heads;
    let kv_per_token = 2u64 * gguf.n_layers as u64 * gguf.n_kv_heads as u64 * head_dim as u64 * 2;

    let (vram, ram) = detect_available_memory();

    let available_for_kv = if let Some(vram_bytes) = vram {
        vram_bytes
            .saturating_sub(model_file_size)
            .saturating_sub(OVERHEAD)
    } else {
        ram.saturating_sub(OVERHEAD)
    };

    if kv_per_token == 0 {
        let cap = practical_context_cap(model_file_size).min(FALLBACK_CTX);
        debug!("KV per token is 0, using {}K context", cap / 1024);
        return cap;
    }

    let max_ctx_from_memory = (available_for_kv / kv_per_token) as usize;
    let cap = practical_context_cap(model_file_size);
    let ctx = max_ctx_from_memory
        .max(4096)
        .min(gguf.context_length as usize)
        .min(cap);
    let ctx = (ctx / 1024) * 1024;

    let mem_source = if vram.is_some() { "VRAM" } else { "RAM" };
    debug!(
        "Auto-sized context: {} tokens ({}K) — kv/tok={}B, available {}={:.1}GB, model={:.1}GB, practical_cap={}K",
        ctx, ctx / 1024, kv_per_token,
        mem_source, available_for_kv as f64 / 1e9,
        model_file_size as f64 / 1e9,
        cap / 1024,
    );

    ctx
}

// ============================================================================
// VRAM Budget Types
// ============================================================================

/// Hardware-relevant metadata for a single model (extracted once, reused for budget computation).
#[derive(Debug, Clone)]
pub(crate) struct ModelProfile {
    pub name: String,
    /// Model file size in bytes (determines base VRAM for weights).
    pub file_size_bytes: u64,
    /// KV cache cost per token in bytes.
    /// Computed from GGUF: 2 * n_layers * n_kv_heads * head_dim * 2.
    pub kv_bytes_per_token: u64,
    /// Maximum context length the model architecture supports.
    pub max_context_length: usize,
    /// Practical context cap from file-size heuristic.
    pub practical_cap: usize,
}

/// Input to the VRAM budget solver. All fields are plain data -- no I/O.
#[derive(Debug, Clone)]
pub(crate) struct VramBudgetInput {
    /// Available VRAM in bytes (or RAM if no GPU).
    pub available_memory: u64,
    /// Hard cap on total VRAM usage (default: 16GB from config).
    pub vram_cap: u64,
    /// Per-server overhead in bytes (default: 512MB).
    pub overhead_per_model: u64,
    /// Main model profile (always present in local mode).
    pub main: ModelProfile,
    /// Router model profile (None if trio disabled or no router set).
    pub router: Option<ModelProfile>,
    /// Specialist model profile (None if trio disabled or no specialist set).
    pub specialist: Option<ModelProfile>,
}

/// Per-model breakdown in a VRAM budget result.
#[derive(Debug, Clone)]
pub(crate) struct VramModelBreakdown {
    pub role: String,
    pub name: String,
    pub weights_bytes: u64,
    pub kv_cache_bytes: u64,
    pub context_tokens: usize,
    pub overhead_bytes: u64,
}

/// Computed context sizes and VRAM usage for each role.
#[derive(Debug, Clone)]
pub(crate) struct VramBudgetResult {
    pub main_ctx: usize,
    pub router_ctx: usize,
    pub specialist_ctx: usize,
    pub total_vram_bytes: u64,
    pub effective_limit_bytes: u64,
    pub fits: bool,
    pub breakdown: Vec<VramModelBreakdown>,
}

// ============================================================================
// VRAM Budget Pure Functions
// ============================================================================

/// Build a ModelProfile from GGUF metadata and file size. Pure function.
pub(crate) fn build_model_profile(
    name: &str,
    file_size_bytes: u64,
    gguf: Option<&GgufModelInfo>,
) -> ModelProfile {
    let (kv_bytes_per_token, max_context_length) = match gguf {
        Some(info) => {
            let head_dim = if info.n_heads > 0 {
                info.embedding_dim / info.n_heads
            } else {
                128 // fallback
            };
            let kv = 2u64
                * info.n_layers as u64
                * info.n_kv_heads as u64
                * head_dim as u64
                * 2; // fp16
            (kv, info.context_length as usize)
        }
        None => (0, 32768),
    };
    ModelProfile {
        name: name.to_string(),
        file_size_bytes,
        kv_bytes_per_token,
        max_context_length,
        practical_cap: practical_context_cap(file_size_bytes),
    }
}

/// Estimate file size from model name (e.g. "qwen3-1.7b" -> ~1.1GB Q4_K_M).
pub(crate) fn estimate_file_size_from_name(name: &str) -> u64 {
    let lower = name.to_lowercase();
    // Extract param count from common patterns: "1.7b", "3b", "8b", "9b", "14b", "70b"
    let param_billions: Option<f64> = {
        let mut result = None;
        // Try patterns like "-1.7b", "-3b", "_8b", etc.
        for separator in &["-", "_", " "] {
            for part in lower.split(separator) {
                if let Some(num_str) = part.strip_suffix('b') {
                    if let Ok(n) = num_str.parse::<f64>() {
                        if n > 0.1 && n < 1000.0 {
                            result = Some(n);
                            break;
                        }
                    }
                }
            }
            if result.is_some() {
                break;
            }
        }
        result
    };

    match param_billions {
        Some(params) => {
            // Q4_K_M is roughly 0.6 bytes per param
            (params * 0.6 * 1e9) as u64
        }
        None => {
            // Conservative fallback: assume ~3B model
            2_000_000_000
        }
    }
}

/// Compute optimal context sizes for all models to fit within VRAM budget.
///
/// Priority order: main gets the largest share, specialist next, router least.
/// Each role has a minimum context (main: 4096, specialist: 4096, router: 2048).
pub(crate) fn compute_vram_budget(input: &VramBudgetInput) -> VramBudgetResult {
    const MIN_MAIN_CTX: usize = 4096;
    const MIN_ROUTER_CTX: usize = 2048;
    const MIN_SPECIALIST_CTX: usize = 4096;

    let effective_limit = input.available_memory.min(input.vram_cap);

    // Count models and compute fixed costs (weights + overhead)
    let mut model_count: u64 = 1; // main always
    let mut fixed_cost: u64 = input.main.file_size_bytes + input.overhead_per_model;

    if let Some(ref router) = input.router {
        model_count += 1;
        fixed_cost += router.file_size_bytes + input.overhead_per_model;
    }
    if let Some(ref specialist) = input.specialist {
        model_count += 1;
        fixed_cost += specialist.file_size_bytes + input.overhead_per_model;
    }
    let _ = model_count; // used for breakdown

    // If fixed costs alone exceed limit, we can't fit even with minimum contexts
    if fixed_cost >= effective_limit {
        let main_ctx = MIN_MAIN_CTX;
        let router_ctx = if input.router.is_some() { MIN_ROUTER_CTX } else { 0 };
        let specialist_ctx = if input.specialist.is_some() { MIN_SPECIALIST_CTX } else { 0 };

        let main_kv = main_ctx as u64 * input.main.kv_bytes_per_token;
        let router_kv = input.router.as_ref().map_or(0, |r| router_ctx as u64 * r.kv_bytes_per_token);
        let spec_kv = input.specialist.as_ref().map_or(0, |s| specialist_ctx as u64 * s.kv_bytes_per_token);
        let total = fixed_cost + main_kv + router_kv + spec_kv;

        let breakdown = build_breakdown(input, main_ctx, router_ctx, specialist_ctx);

        return VramBudgetResult {
            main_ctx,
            router_ctx,
            specialist_ctx,
            total_vram_bytes: total,
            effective_limit_bytes: effective_limit,
            fits: false,
            breakdown,
        };
    }

    let remaining = effective_limit - fixed_cost;

    // Weighted proportional allocation: main=4, specialist=2, router=1
    let main_weight: u64 = 4;
    let router_weight: u64 = if input.router.is_some() { 1 } else { 0 };
    let specialist_weight: u64 = if input.specialist.is_some() { 2 } else { 0 };
    let total_weight = main_weight + router_weight + specialist_weight;

    // Allocate remaining bytes proportionally
    let main_kv_budget = remaining * main_weight / total_weight;
    let router_kv_budget = if input.router.is_some() {
        remaining * router_weight / total_weight
    } else {
        0
    };
    let specialist_kv_budget = if input.specialist.is_some() {
        remaining * specialist_weight / total_weight
    } else {
        0
    };

    // Convert bytes to tokens, clamp, and round
    let main_ctx = bytes_to_ctx(
        main_kv_budget,
        input.main.kv_bytes_per_token,
        MIN_MAIN_CTX,
        input.main.practical_cap.min(input.main.max_context_length),
    );

    let router_ctx = if let Some(ref router) = input.router {
        bytes_to_ctx(
            router_kv_budget,
            router.kv_bytes_per_token,
            MIN_ROUTER_CTX,
            router.practical_cap.min(router.max_context_length),
        )
    } else {
        0
    };

    let specialist_ctx = if let Some(ref specialist) = input.specialist {
        bytes_to_ctx(
            specialist_kv_budget,
            specialist.kv_bytes_per_token,
            MIN_SPECIALIST_CTX,
            specialist.practical_cap.min(specialist.max_context_length),
        )
    } else {
        0
    };

    // Compute actual total
    let main_kv = main_ctx as u64 * input.main.kv_bytes_per_token;
    let router_kv = input.router.as_ref().map_or(0, |r| router_ctx as u64 * r.kv_bytes_per_token);
    let spec_kv = input.specialist.as_ref().map_or(0, |s| specialist_ctx as u64 * s.kv_bytes_per_token);
    let total = fixed_cost + main_kv + router_kv + spec_kv;

    let breakdown = build_breakdown(input, main_ctx, router_ctx, specialist_ctx);

    VramBudgetResult {
        main_ctx,
        router_ctx,
        specialist_ctx,
        total_vram_bytes: total,
        effective_limit_bytes: effective_limit,
        fits: total <= effective_limit,
        breakdown,
    }
}

/// Convert a byte budget to context tokens, clamped and rounded down to 1K.
fn bytes_to_ctx(budget_bytes: u64, kv_per_token: u64, min_ctx: usize, max_ctx: usize) -> usize {
    if kv_per_token == 0 {
        return min_ctx;
    }
    let raw = (budget_bytes / kv_per_token) as usize;
    let clamped = raw.max(min_ctx).min(max_ctx);
    // Round down to nearest 1024
    (clamped / 1024) * 1024
}

fn build_breakdown(
    input: &VramBudgetInput,
    main_ctx: usize,
    router_ctx: usize,
    specialist_ctx: usize,
) -> Vec<VramModelBreakdown> {
    let mut breakdown = vec![VramModelBreakdown {
        role: "main".to_string(),
        name: input.main.name.clone(),
        weights_bytes: input.main.file_size_bytes,
        kv_cache_bytes: main_ctx as u64 * input.main.kv_bytes_per_token,
        context_tokens: main_ctx,
        overhead_bytes: input.overhead_per_model,
    }];

    if let Some(ref router) = input.router {
        breakdown.push(VramModelBreakdown {
            role: "router".to_string(),
            name: router.name.clone(),
            weights_bytes: router.file_size_bytes,
            kv_cache_bytes: router_ctx as u64 * router.kv_bytes_per_token,
            context_tokens: router_ctx,
            overhead_bytes: input.overhead_per_model,
        });
    }

    if let Some(ref specialist) = input.specialist {
        breakdown.push(VramModelBreakdown {
            role: "specialist".to_string(),
            name: specialist.name.clone(),
            weights_bytes: specialist.file_size_bytes,
            kv_cache_bytes: specialist_ctx as u64 * specialist.kv_bytes_per_token,
            context_tokens: specialist_ctx,
            overhead_bytes: input.overhead_per_model,
        });
    }

    breakdown
}

/// Resolve a ModelProfile by reading GGUF metadata from disk.
pub(crate) fn resolve_model_profile_from_path(name: &str, path: &Path) -> ModelProfile {
    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let gguf = parse_gguf_metadata(path);
    build_model_profile(name, file_size, gguf.as_ref())
}

/// Estimate a ModelProfile from an LMS model identifier.
pub(crate) fn estimate_model_profile_from_name(name: &str) -> ModelProfile {
    let estimated_size = estimate_file_size_from_name(name);
    build_model_profile(name, estimated_size, None)
}

// ============================================================================
// Health Checks
// ============================================================================

/// Quick health check: is the local server at the given base URL alive?
///
/// Sends a GET to `/health` with a 2-second timeout.
pub(crate) async fn check_health(api_base: &str) -> bool {
    let base = api_base
        .trim_end_matches('/')
        .trim_end_matches("/v1")
        .trim_end_matches('/');
    let url = format!("{}/health", base);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap_or_default();
    match client.get(&url).send().await {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
}

/// Deep health check: verify server can handle a chat completion.
pub(crate) async fn check_chat_health(port: &str) -> bool {
    let url = format!("http://localhost:{}/v1/chat/completions", port);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_default();

    let body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1
    });

    match client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
    {
        Ok(resp) => {
            if !resp.status().is_success() {
                return false;
            }
            if let Ok(text) = resp.text().await {
                text.contains("\"choices\"") || text.contains("\"content\"")
            } else {
                false
            }
        }
        Err(_) => false,
    }
}

// ============================================================================
// Context Size Query
// ============================================================================

/// Query the local server for its actual context size (`n_ctx`).
///
/// Returns the server's context window with 5% headroom subtracted.
pub(crate) fn query_local_context_size(port: &str) -> Option<usize> {
    let url = format!("http://localhost:{}/props", port);
    let props = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::try_current();
        match rt {
            Ok(handle) => handle.block_on(async {
                reqwest::Client::new()
                    .get(&url)
                    .send()
                    .await
                    .ok()?
                    .json::<serde_json::Value>()
                    .await
                    .ok()
            }),
            Err(_) => reqwest::blocking::get(&url)
                .ok()?
                .json::<serde_json::Value>()
                .ok(),
        }
    })?;
    let n_ctx = props
        .get("default_generation_settings")
        .and_then(|v| v.get("n_ctx"))
        .and_then(|v| v.as_u64())
        .or_else(|| props.get("n_ctx").and_then(|v| v.as_u64()))? as usize;
    let n_parallel = props
        .get("default_generation_settings")
        .and_then(|v| v.get("n_parallel"))
        .and_then(|v| v.as_u64())
        .or_else(|| props.get("n_parallel").and_then(|v| v.as_u64()))
        .unwrap_or(1)
        .max(1) as usize;
    let per_request_ctx = (n_ctx / n_parallel).max(1);
    Some((per_request_ctx as f64 * 0.95) as usize)
}

pub(crate) async fn query_local_context_size_async(port: &str) -> Option<usize> {
    let url = format!("http://localhost:{}/props", port);
    let props = reqwest::Client::new()
        .get(&url)
        .send()
        .await
        .ok()?
        .json::<serde_json::Value>()
        .await
        .ok()?;
    let n_ctx = props
        .get("default_generation_settings")
        .and_then(|v| v.get("n_ctx"))
        .and_then(|v| v.as_u64())
        .or_else(|| props.get("n_ctx").and_then(|v| v.as_u64()))? as usize;
    let n_parallel = props
        .get("default_generation_settings")
        .and_then(|v| v.get("n_parallel"))
        .and_then(|v| v.as_u64())
        .or_else(|| props.get("n_parallel").and_then(|v| v.as_u64()))
        .unwrap_or(1)
        .max(1) as usize;
    let per_request_ctx = (n_ctx / n_parallel).max(1);
    Some((per_request_ctx as f64 * 0.95) as usize)
}

// ============================================================================
// Health Watchdog
// ============================================================================

/// Request to restart a specific server role.
#[derive(Debug, Clone)]
pub struct RestartRequest {
    pub role: String,
}

/// Background health watchdog with auto-repair for local server processes.
///
/// Pings `/health` every 30 seconds on all active server ports. When a server
/// fails 3 consecutive health checks, sends a restart request through `restart_tx`.
pub(crate) fn start_health_watchdog_with_autorepair(
    ports: Vec<(String, String)>, // (role, port)
    alert_tx: tokio::sync::mpsc::UnboundedSender<String>,
    restart_tx: tokio::sync::mpsc::UnboundedSender<RestartRequest>,
    inference_active: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_default();

        let mut was_healthy: HashMap<String, bool> =
            ports.iter().map(|(role, _)| (role.clone(), true)).collect();

        let mut consecutive_failures: HashMap<String, u32> =
            ports.iter().map(|(role, _)| (role.clone(), 0)).collect();

        loop {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;

            if inference_active.load(std::sync::atomic::Ordering::Relaxed) {
                continue;
            }

            for (role, port) in &ports {
                let healthy = if role == "main" {
                    check_chat_health(port).await
                } else {
                    let url = format!("http://localhost:{}/health", port);
                    match client.get(&url).send().await {
                        Ok(resp) => resp.status().is_success(),
                        Err(_) => false,
                    }
                };

                let prev = was_healthy.get(role).copied().unwrap_or(true);

                if !healthy {
                    *consecutive_failures.get_mut(role).unwrap() += 1;
                    let failures = consecutive_failures[role];

                    if failures < 3 && prev {
                        let msg = format!(
                            "\x1b[RAW]\n  \x1b[33m\u{25cf}\x1b[0m \x1b[1m{} server\x1b[0m (port {}) \x1b[33munhealthy\x1b[0m (attempt {}/3)\n",
                            role, port, failures
                        );
                        let _ = alert_tx.send(msg);
                    } else if failures >= 3 {
                        let msg = format!(
                            "\x1b[RAW]\n  \x1b[33m\u{25cf}\x1b[0m \x1b[1m{} server\x1b[0m auto-restarting...\n",
                            role
                        );
                        let _ = alert_tx.send(msg);
                        let _ = restart_tx.send(RestartRequest { role: role.clone() });
                        consecutive_failures.insert(role.clone(), 0);
                    }
                } else {
                    let was_failed = consecutive_failures[role] > 0;
                    consecutive_failures.insert(role.clone(), 0);

                    if !prev {
                        let msg = format!(
                            "\x1b[RAW]\n  \x1b[32m\u{25cf}\x1b[0m \x1b[1m{} server\x1b[0m \x1b[32mrecovered\x1b[0m\n",
                            role
                        );
                        let _ = alert_tx.send(msg);
                    } else if was_failed {
                        let msg = format!(
                            "\x1b[RAW]\n  \x1b[32m\u{25cf}\x1b[0m \x1b[1m{} server\x1b[0m \x1b[32mhealthy\x1b[0m\n",
                            role
                        );
                        let _ = alert_tx.send(msg);
                    }
                }

                was_healthy.insert(role.clone(), healthy);
            }
        }
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn can_bind_localhost() -> bool {
        std::net::TcpListener::bind(("127.0.0.1", 0)).is_ok()
    }

    // -- practical_context_cap tests --

    #[test]
    fn test_practical_context_cap_tiny_model() {
        assert_eq!(practical_context_cap(1_500_000_000), 8192);
    }

    #[test]
    fn test_practical_context_cap_small_model() {
        assert_eq!(practical_context_cap(3_000_000_000), 16384);
    }

    #[test]
    fn test_practical_context_cap_medium_model() {
        assert_eq!(practical_context_cap(6_000_000_000), 32768);
    }

    #[test]
    fn test_practical_context_cap_large_model() {
        assert_eq!(practical_context_cap(12_000_000_000), 65536);
    }

    #[test]
    fn test_practical_context_cap_xlarge_model() {
        assert_eq!(practical_context_cap(20_000_000_000), usize::MAX);
    }

    #[test]
    fn test_practical_context_cap_boundary_2gb() {
        let two_gb = 2 * 1024 * 1024 * 1024u64;
        assert_eq!(practical_context_cap(two_gb), 16384);
        assert_eq!(practical_context_cap(two_gb - 1), 8192);
    }

    // -- find_available_port tests --

    #[test]
    fn test_find_available_port_returns_valid_port() {
        if !can_bind_localhost() {
            return;
        }
        let port = find_available_port(19000);
        assert!(port >= 19000);
        assert!(port <= 19099);
        assert!(TcpListener::bind(("127.0.0.1", port)).is_ok());
    }

    #[test]
    fn test_find_available_port_skips_occupied() {
        if !can_bind_localhost() {
            return;
        }
        let listener = TcpListener::bind(("127.0.0.1", 19200)).unwrap();
        let port = find_available_port(19200);
        assert!(port > 19200);
        drop(listener);
    }

    #[test]
    fn test_find_port_within_range() {
        let start = 40000;
        let port = find_available_port(start);
        assert!(port >= start);
        assert!(port < start + 100);
    }

    #[test]
    fn test_find_port_skips_consecutive_occupied() {
        let base: u16 = 48500;
        let l1 = TcpListener::bind(("127.0.0.1", base));
        let l2 = TcpListener::bind(("127.0.0.1", base + 1));

        if let (Ok(_l1), Ok(_l2)) = (l1, l2) {
            let found = find_available_port(base);
            assert!(found >= base + 2);
        }
    }

    #[test]
    fn test_find_port_high_start_no_overflow() {
        let port = find_available_port(65500);
        assert!(port >= 65500);
    }

    // -- parse_gguf_metadata tests --

    fn build_synthetic_gguf(
        arch: &str,
        n_layers: u32,
        n_kv_heads: u32,
        n_heads: u32,
        embedding_dim: u32,
        context_length: u32,
    ) -> Vec<u8> {
        let mut buf = Vec::new();

        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&6u64.to_le_bytes());

        fn write_string_kv(buf: &mut Vec<u8>, key: &str, value: &str) {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&8u32.to_le_bytes());
            buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
            buf.extend_from_slice(value.as_bytes());
        }

        fn write_u32_kv(buf: &mut Vec<u8>, key: &str, value: u32) {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&4u32.to_le_bytes());
            buf.extend_from_slice(&value.to_le_bytes());
        }

        write_string_kv(&mut buf, "general.architecture", arch);
        write_u32_kv(&mut buf, &format!("{}.block_count", arch), n_layers);
        write_u32_kv(
            &mut buf,
            &format!("{}.attention.head_count_kv", arch),
            n_kv_heads,
        );
        write_u32_kv(&mut buf, &format!("{}.attention.head_count", arch), n_heads);
        write_u32_kv(
            &mut buf,
            &format!("{}.embedding_length", arch),
            embedding_dim,
        );
        write_u32_kv(
            &mut buf,
            &format!("{}.context_length", arch),
            context_length,
        );

        buf
    }

    #[test]
    fn test_parse_gguf_metadata_synthetic() {
        let data = build_synthetic_gguf("llama", 32, 8, 32, 4096, 131072);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&data).unwrap();

        let info = parse_gguf_metadata(&path).expect("Should parse synthetic GGUF");
        assert_eq!(info.n_layers, 32);
        assert_eq!(info.n_kv_heads, 8);
        assert_eq!(info.n_heads, 32);
        assert_eq!(info.embedding_dim, 4096);
        assert_eq!(info.context_length, 131072);
    }

    #[test]
    fn test_parse_gguf_metadata_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.gguf");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"NOT_GGUF_DATA").unwrap();
        assert!(parse_gguf_metadata(&path).is_none());
    }

    #[test]
    fn test_parse_gguf_metadata_nonexistent() {
        assert!(parse_gguf_metadata(Path::new("/nonexistent/model.gguf")).is_none());
    }

    #[test]
    fn test_parse_gguf_small_model() {
        let data = build_synthetic_gguf("qwen2", 28, 4, 16, 1024, 32768);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("qwen.gguf");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&data).unwrap();

        let info = parse_gguf_metadata(&path).unwrap();
        assert_eq!(info.n_layers, 28);
        assert_eq!(info.n_kv_heads, 4);
        assert_eq!(info.embedding_dim, 1024);
        assert_eq!(info.context_length, 32768);
    }

    // -- list_local_models tests --

    #[test]
    fn test_list_local_models_returns_sorted() {
        let models = list_local_models();
        for (i, model) in models.iter().enumerate() {
            assert_eq!(model.extension().and_then(|e| e.to_str()), Some("gguf"));
            if i > 0 {
                assert!(model.file_name() >= models[i - 1].file_name());
            }
        }
    }

    // -- query_local_context_size tests --

    #[test]
    fn test_query_local_context_size_no_server() {
        assert!(query_local_context_size("59999").is_none());
    }

    // -- detect_available_memory --

    #[test]
    fn test_detect_available_memory_sane() {
        let (vram, ram) = detect_available_memory();
        assert!(ram >= 1_000_000_000, "RAM {} seems too low", ram);
        if let Some(v) = vram {
            assert!(v >= 256 * 1024 * 1024, "VRAM {} seems too low", v);
        }
    }

    // -- VRAM budget tests --

    const GB: u64 = 1_000_000_000;
    const MB: u64 = 1_000_000;

    fn make_profile(name: &str, file_size: u64, kv_per_token: u64) -> ModelProfile {
        ModelProfile {
            name: name.to_string(),
            file_size_bytes: file_size,
            kv_bytes_per_token: kv_per_token,
            max_context_length: 131072,
            practical_cap: practical_context_cap(file_size),
        }
    }

    // -- build_model_profile tests --

    #[test]
    fn test_build_model_profile_from_gguf() {
        let gguf = GgufModelInfo {
            n_layers: 32,
            n_kv_heads: 8,
            n_heads: 32,
            embedding_dim: 4096,
            context_length: 131072,
        };
        let profile = build_model_profile("test-8b", 5 * GB, Some(&gguf));
        // head_dim = 4096/32 = 128, kv = 2 * 32 * 8 * 128 * 2 = 131072
        assert_eq!(profile.kv_bytes_per_token, 131072);
        assert_eq!(profile.max_context_length, 131072);
        assert_eq!(profile.practical_cap, 32768); // 5GB -> <8GB bracket
    }

    #[test]
    fn test_build_model_profile_no_gguf() {
        let profile = build_model_profile("unknown", 2 * GB, None);
        assert_eq!(profile.kv_bytes_per_token, 0);
        assert_eq!(profile.max_context_length, 32768);
    }

    // -- estimate_file_size_from_name tests --

    #[test]
    fn test_estimate_file_size_1_7b() {
        let size = estimate_file_size_from_name("qwen3-1.7b");
        assert!(
            size > 900 * MB && size < 1500 * MB,
            "Expected ~1.1GB for 1.7B Q4, got {}",
            size
        );
    }

    #[test]
    fn test_estimate_file_size_8b() {
        let size = estimate_file_size_from_name("nvidia_orchestrator-8b");
        assert!(
            size > 4 * GB && size < 6 * GB,
            "Expected ~4.8GB for 8B Q4, got {}",
            size
        );
    }

    #[test]
    fn test_estimate_file_size_3b() {
        let size = estimate_file_size_from_name("ministral-3b-instruct");
        assert!(
            size > 1500 * MB && size < 2500 * MB,
            "Expected ~1.8GB for 3B Q4, got {}",
            size
        );
    }

    #[test]
    fn test_estimate_file_size_unknown() {
        let size = estimate_file_size_from_name("mystery-model");
        assert!(size > 0, "Should return conservative fallback");
    }

    // -- compute_vram_budget tests --

    #[test]
    fn test_budget_single_model_fits_easily() {
        let input = VramBudgetInput {
            available_memory: 16 * GB,
            vram_cap: 16 * GB,
            overhead_per_model: 512 * MB,
            main: make_profile("test-8b", 5 * GB, 131072),
            router: None,
            specialist: None,
        };
        let result = compute_vram_budget(&input);
        assert!(result.fits);
        assert!(result.main_ctx >= 4096);
        assert!(result.main_ctx <= 32768); // capped by practical_cap (5GB -> 32K)
        assert_eq!(result.router_ctx, 0);
        assert_eq!(result.specialist_ctx, 0);
    }

    #[test]
    fn test_budget_trio_fits_in_16gb() {
        // Typical trio: 9B main (5.5GB) + 8B router (4.9GB) + 3B specialist (1.9GB)
        let input = VramBudgetInput {
            available_memory: 16 * GB,
            vram_cap: 16 * GB,
            overhead_per_model: 512 * MB,
            main: make_profile("main-9b", 5_500 * MB, 131072),
            router: Some(make_profile("router-8b", 4_900 * MB, 131072)),
            specialist: Some(make_profile("specialist-3b", 1_900 * MB, 32768)),
        };
        let result = compute_vram_budget(&input);
        assert!(result.fits, "Trio should fit in 16GB");
        assert!(result.main_ctx >= 4096);
        assert!(result.router_ctx >= 2048);
        assert!(result.specialist_ctx >= 4096);
        // Main should get the biggest share
        assert!(
            result.main_ctx >= result.router_ctx,
            "main_ctx {} should >= router_ctx {}",
            result.main_ctx,
            result.router_ctx
        );
    }

    #[test]
    fn test_budget_trio_does_not_fit() {
        // 8GB VRAM with three large models -> doesn't fit
        let input = VramBudgetInput {
            available_memory: 8 * GB,
            vram_cap: 8 * GB,
            overhead_per_model: 512 * MB,
            main: make_profile("main-8b", 4_900 * MB, 131072),
            router: Some(make_profile("router-8b", 4_900 * MB, 131072)),
            specialist: Some(make_profile("spec-8b", 4_900 * MB, 131072)),
        };
        let result = compute_vram_budget(&input);
        assert!(!result.fits, "Three 5GB models can't fit in 8GB");
    }

    #[test]
    fn test_budget_respects_vram_cap() {
        let input = VramBudgetInput {
            available_memory: 24 * GB,
            vram_cap: 12 * GB,
            overhead_per_model: 512 * MB,
            main: make_profile("main", 5 * GB, 131072),
            router: None,
            specialist: None,
        };
        let result = compute_vram_budget(&input);
        assert_eq!(result.effective_limit_bytes, 12 * GB);
    }

    #[test]
    fn test_budget_priority_order() {
        // With constrained budget, main should get more than specialist, specialist more than router
        let input = VramBudgetInput {
            available_memory: 16 * GB,
            vram_cap: 16 * GB,
            overhead_per_model: 512 * MB,
            main: make_profile("main", 5 * GB, 131072),
            router: Some(make_profile("router", 5 * GB, 131072)),
            specialist: Some(make_profile("spec", 5 * GB, 131072)),
        };
        let result = compute_vram_budget(&input);
        // main_weight=4, spec_weight=2, router_weight=1
        assert!(
            result.main_ctx >= result.specialist_ctx,
            "main {} should >= specialist {}",
            result.main_ctx,
            result.specialist_ctx
        );
        assert!(
            result.specialist_ctx >= result.router_ctx,
            "specialist {} should >= router {}",
            result.specialist_ctx,
            result.router_ctx
        );
    }

    #[test]
    fn test_budget_zero_kv_fallback() {
        // When kv_bytes_per_token is 0 (no GGUF), should get minimum context
        let input = VramBudgetInput {
            available_memory: 16 * GB,
            vram_cap: 16 * GB,
            overhead_per_model: 512 * MB,
            main: ModelProfile {
                name: "no-gguf".to_string(),
                file_size_bytes: 2 * GB,
                kv_bytes_per_token: 0,
                max_context_length: 32768,
                practical_cap: 16384,
            },
            router: None,
            specialist: None,
        };
        let result = compute_vram_budget(&input);
        assert!(result.fits);
        assert_eq!(result.main_ctx, 4096); // minimum when kv=0
    }

    #[test]
    fn test_budget_round_down_to_1k() {
        // Context sizes should be multiples of 1024
        let input = VramBudgetInput {
            available_memory: 16 * GB,
            vram_cap: 16 * GB,
            overhead_per_model: 512 * MB,
            main: make_profile("main", 3 * GB, 131072),
            router: Some(make_profile("router", 3 * GB, 131072)),
            specialist: None,
        };
        let result = compute_vram_budget(&input);
        assert_eq!(result.main_ctx % 1024, 0, "main_ctx should be multiple of 1024");
        assert_eq!(result.router_ctx % 1024, 0, "router_ctx should be multiple of 1024");
    }

    #[test]
    fn test_budget_breakdown_has_all_roles() {
        let input = VramBudgetInput {
            available_memory: 16 * GB,
            vram_cap: 16 * GB,
            overhead_per_model: 512 * MB,
            main: make_profile("main-model", 3 * GB, 131072),
            router: Some(make_profile("router-model", 2 * GB, 32768)),
            specialist: Some(make_profile("spec-model", 2 * GB, 32768)),
        };
        let result = compute_vram_budget(&input);
        assert_eq!(result.breakdown.len(), 3);
        assert_eq!(result.breakdown[0].role, "main");
        assert_eq!(result.breakdown[1].role, "router");
        assert_eq!(result.breakdown[2].role, "specialist");
    }
}
