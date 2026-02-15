//! Local LLM server management: llama-server spawn/health, GGUF parser, context sizing.
//!
//! Handles spawning and managing llama.cpp server processes (main model + compaction),
//! parsing GGUF model metadata for auto-sizing context windows, and detecting available
//! system memory (VRAM/RAM).

use std::collections::HashMap;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;

use tracing::debug;

// ============================================================================
// Constants
// ============================================================================

pub(crate) const DEFAULT_LOCAL_MODEL: &str = "NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf";

const COMPACTION_MODEL_URL: &str =
    "https://huggingface.co/MaziyarPanahi/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B.Q4_K_M.gguf";
const COMPACTION_MODEL_FILENAME: &str = "Qwen3-0.6B.Q4_K_M.gguf";

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
// Function Implementations
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

/// Ensure the dedicated compaction model is available locally.
///
/// Downloads Qwen3-0.6B Q4_K_M (~500MB) to `~/.nanobot/models/` if not already
/// present. Returns `None` on failure (graceful degradation — compaction just
/// gets skipped and the system falls back to `trim_to_fit`).
pub(crate) fn ensure_compaction_model() -> Option<PathBuf> {
    let models_dir = dirs::home_dir()?.join(".nanobot").join("models");
    std::fs::create_dir_all(&models_dir).ok()?;

    let model_path = models_dir.join(COMPACTION_MODEL_FILENAME);
    if model_path.exists() {
        return Some(model_path);
    }

    println!(
        "  {}{}Downloading{} compaction model (Qwen3-0.6B, ~500MB)...",
        crate::tui::BOLD,
        crate::tui::YELLOW,
        crate::tui::RESET
    );

    let tmp_path = models_dir.join(format!("{}.downloading", COMPACTION_MODEL_FILENAME));
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let mut resp = reqwest::blocking::get(COMPACTION_MODEL_URL)?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()).into());
        }
        let mut file = std::fs::File::create(&tmp_path)?;
        resp.copy_to(&mut file)?;
        std::fs::rename(&tmp_path, &model_path)?;
        Ok(())
    })();

    match result {
        Ok(()) => {
            println!(
                "  {}{}Done{} — saved to {}",
                crate::tui::BOLD,
                crate::tui::GREEN,
                crate::tui::RESET,
                model_path.display()
            );
            Some(model_path)
        }
        Err(e) => {
            println!(
                "  {}{}Download failed:{} {} (compaction will use trim_to_fit fallback)",
                crate::tui::BOLD,
                crate::tui::YELLOW,
                crate::tui::RESET,
                e
            );
            // Clean up partial download
            let _ = std::fs::remove_file(&tmp_path);
            None
        }
    }
}

/// Start the dedicated compaction server if the model is available.
///
/// Downloads the model on first run, spawns a GPU-accelerated llama-server on
/// port 8090+ with context matching the main model, and stores the process
/// handle and port. Gracefully degrades if anything fails.
pub(crate) async fn start_compaction_if_available(
    compaction_process: &mut Option<Child>,
    compaction_port: &mut Option<String>,
    main_ctx_size: usize,
) {
    // Already running?
    if compaction_process.is_some() {
        return;
    }

    let model_path = match ensure_compaction_model() {
        Some(p) => p,
        None => return,
    };

    let port = find_available_port(8090);
    println!(
        "  {}{}Starting{} compaction server on port {} (ctx: {}K, GPU)...",
        crate::tui::BOLD,
        crate::tui::YELLOW,
        crate::tui::RESET,
        port,
        main_ctx_size / 1024,
    );

    match spawn_compaction_server(port, &model_path, main_ctx_size) {
        Ok(child) => {
            record_server_pid("compaction", child.id());
            *compaction_process = Some(child);
            if wait_for_server_ready(port, 15, compaction_process).await {
                *compaction_port = Some(port.to_string());
                println!(
                    "  {}{}Compaction server ready{} (Qwen3-0.6B on GPU)",
                    crate::tui::BOLD,
                    crate::tui::GREEN,
                    crate::tui::RESET
                );
            } else {
                println!(
                    "  {}{}Compaction server failed to start{} (using trim_to_fit fallback)",
                    crate::tui::BOLD,
                    crate::tui::YELLOW,
                    crate::tui::RESET
                );
                if let Some(ref mut child) = compaction_process {
                    child.kill().ok();
                    child.wait().ok();
                }
                *compaction_process = None;
            }
        }
        Err(e) => {
            println!(
                "  {}{}Compaction server failed:{} {} (using trim_to_fit fallback)",
                crate::tui::BOLD,
                crate::tui::YELLOW,
                crate::tui::RESET,
                e
            );
        }
    }
}

/// Stop the compaction server and clear state.
pub(crate) fn stop_compaction_server(
    compaction_process: &mut Option<Child>,
    compaction_port: &mut Option<String>,
) {
    if let Some(ref mut child) = compaction_process {
        let pid = child.id();
        child.kill().ok();
        child.wait().ok();
        unrecord_server_pid(pid);
    }
    *compaction_process = None;
    *compaction_port = None;
}

// ============================================================================
// Delegation server (auto-spawned for tool delegation in local mode)
// ============================================================================

/// Model preferences for the auto-spawned delegation server, in priority order.
/// The first model found in `~/models/` wins.
const DELEGATION_MODEL_PREFERENCES: &[&str] = &[
    "Ministral-3",
    "Qwen3-0.6B",
    "Ministral-8B",
    "Nemotron-Nano-9B",
];

/// Find a suitable delegation model from `~/models/` using the preference list.
///
/// Scans `list_local_models()` and returns the first match against
/// `DELEGATION_MODEL_PREFERENCES` (case-insensitive substring match).
pub(crate) fn find_delegation_model() -> Option<PathBuf> {
    pick_preferred_model(&list_local_models(), DELEGATION_MODEL_PREFERENCES)
}

/// Pure priority-matching: given a list of model paths and an ordered preference
/// list, return the first model that matches the highest-priority preference
/// (case-insensitive substring match on filename).
fn pick_preferred_model(models: &[PathBuf], preferences: &[&str]) -> Option<PathBuf> {
    for pref in preferences {
        let pref_lower = pref.to_lowercase();
        if let Some(m) = models.iter().find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.to_lowercase().contains(&pref_lower))
                .unwrap_or(false)
        }) {
            return Some(m.clone());
        }
    }
    None
}

/// Start the dedicated delegation server if a suitable model is available.
///
/// Spawns a GPU-accelerated llama-server on port 8091+ with a small context
/// window (4096 tokens — tool work is short). Uses 10 GPU layers to avoid
/// competing with the main model for VRAM.
pub(crate) async fn start_delegation_if_available(
    delegation_process: &mut Option<Child>,
    delegation_port: &mut Option<String>,
) {
    // Already running?
    if delegation_process.is_some() {
        return;
    }

    let model_path = match find_delegation_model() {
        Some(p) => p,
        None => return,
    };

    let model_name = model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("delegation-model");
    let port = find_available_port(8091);
    let ctx_size = 4096; // Tool work is short — no need for large context

    // Compute GPU layers based on available VRAM (after main model is loaded).
    let gpu_layers = compute_gpu_layers_for_model(&model_path, ctx_size);
    let gpu_label = if gpu_layers >= 99 {
        "full GPU".to_string()
    } else if gpu_layers == 0 {
        "CPU".to_string()
    } else {
        format!("{} GPU layers", gpu_layers)
    };

    println!(
        "  {}{}Starting{} delegation server ({}, port {}, ctx: {}K, {})...",
        crate::tui::BOLD,
        crate::tui::YELLOW,
        crate::tui::RESET,
        model_name,
        port,
        ctx_size / 1024,
        gpu_label,
    );

    match spawn_delegation_server(port, &model_path, ctx_size, gpu_layers) {
        Ok(child) => {
            record_server_pid("delegation", child.id());
            *delegation_process = Some(child);
            if wait_for_server_ready(port, 30, delegation_process).await {
                *delegation_port = Some(port.to_string());
                println!(
                    "  {}{}Delegation server ready{} ({} on GPU)",
                    crate::tui::BOLD,
                    crate::tui::GREEN,
                    crate::tui::RESET,
                    model_name,
                );
            } else {
                println!(
                    "  {}{}Delegation server failed to start{} (tool delegation will use main model)",
                    crate::tui::BOLD,
                    crate::tui::YELLOW,
                    crate::tui::RESET,
                );
                if let Some(ref mut child) = delegation_process {
                    child.kill().ok();
                    child.wait().ok();
                }
                *delegation_process = None;
            }
        }
        Err(e) => {
            println!(
                "  {}{}Delegation server failed:{} {} (tool delegation will use main model)",
                crate::tui::BOLD,
                crate::tui::YELLOW,
                crate::tui::RESET,
                e,
            );
        }
    }
}

/// Stop the delegation server and clear state.
pub(crate) fn stop_delegation_server(
    delegation_process: &mut Option<Child>,
    delegation_port: &mut Option<String>,
) {
    if let Some(ref mut child) = delegation_process {
        let pid = child.id();
        child.kill().ok();
        child.wait().ok();
        unrecord_server_pid(pid);
    }
    *delegation_process = None;
    *delegation_port = None;
}

/// Path to the PID file for tracking spawned llama-server processes.
fn pid_file_path() -> Option<PathBuf> {
    Some(dirs::home_dir()?.join(".nanobot").join(".server-pids"))
}

/// Record a server PID in the tracking file (format: `role:pid` per line).
pub(crate) fn record_server_pid(role: &str, pid: u32) {
    if let Some(path) = pid_file_path() {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let _ = writeln!(f, "{}:{}", role, pid);
        }
    }
}

/// Remove a specific PID from the tracking file.
pub(crate) fn unrecord_server_pid(pid: u32) {
    if let Some(path) = pid_file_path() {
        if let Ok(contents) = std::fs::read_to_string(&path) {
            let remaining: Vec<&str> = contents
                .lines()
                .filter(|line| {
                    line.split(':')
                        .nth(1)
                        .and_then(|p| p.parse::<u32>().ok())
                        != Some(pid)
                })
                .collect();
            if remaining.is_empty() {
                std::fs::remove_file(&path).ok();
            } else {
                std::fs::write(&path, remaining.join("\n") + "\n").ok();
            }
        }
    }
}

/// Kill llama-server processes tracked from previous nanobot runs.
///
/// Only kills PIDs recorded in `~/.nanobot/.server-pids`, not system-wide.
/// Replaces the former `pkill -f llama-server` approach.
pub(crate) fn kill_tracked_servers() {
    let path = match pid_file_path() {
        Some(p) => p,
        None => return,
    };
    let contents = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return, // No PID file — nothing to kill
    };

    for line in contents.lines() {
        if let Some(pid_str) = line.split(':').nth(1) {
            if let Ok(pid) = pid_str.parse::<i32>() {
                // Send SIGKILL to the specific PID (safe: only our tracked PIDs)
                unsafe {
                    libc::kill(pid, libc::SIGKILL);
                }
            }
        }
    }

    // Clean up the file
    std::fs::remove_file(&path).ok();
    // Brief pause to let ports be released
    std::thread::sleep(std::time::Duration::from_millis(300));
}

/// Kill any orphaned llama-server processes from previous runs.
///
/// Uses PID-based tracking. Falls back to the tracking file only.
pub(crate) fn kill_stale_llama_servers() {
    kill_tracked_servers();
}

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

/// Detect available VRAM (via nvidia-smi) and RAM (via /proc/meminfo).
/// Returns (vram_bytes, ram_bytes).
pub(crate) fn detect_available_memory() -> (Option<u64>, u64) {
    // Try VRAM via nvidia-smi
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

    // RAM via /proc/meminfo
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
///
/// Small models become unresponsive with very large contexts — attention is O(n²)
/// and they lack the capacity to utilize long contexts effectively.
pub(crate) fn practical_context_cap(model_file_size_bytes: u64) -> usize {
    let gb = model_file_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    if gb < 2.0 {
        8192
    }
    // tiny (~1-3B heavy quant)
    else if gb < 4.0 {
        16384
    }
    // small (~3-7B)
    else if gb < 8.0 {
        32768
    }
    // medium (~7-14B)
    else if gb < 16.0 {
        65536
    }
    // large (~14-30B)
    else {
        usize::MAX
    } // xlarge (30B+) — no cap
}

/// Compute optimal --ctx-size for a GGUF model given available system resources.
pub(crate) fn compute_optimal_context_size(model_path: &Path) -> usize {
    const OVERHEAD: u64 = 512 * 1024 * 1024; // 512 MB
    const FALLBACK_CTX: usize = 16384;

    let model_file_size = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);

    let gguf = match parse_gguf_metadata(model_path) {
        Some(info) => info,
        None => {
            // GGUF parse failed — use practical cap based on file size, or fallback
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
    // KV cache per token (FP16): 2 (K+V) × layers × kv_heads × head_dim × 2 bytes
    let kv_per_token = 2u64 * gguf.n_layers as u64 * gguf.n_kv_heads as u64 * head_dim as u64 * 2;

    let (vram, ram) = detect_available_memory();

    let available_for_kv = if let Some(vram_bytes) = vram {
        // GPU mode: VRAM must hold model weights + KV cache
        vram_bytes
            .saturating_sub(model_file_size)
            .saturating_sub(OVERHEAD)
    } else {
        // CPU mode: weights are mmap'd, RAM mainly for KV cache
        ram.saturating_sub(OVERHEAD)
    };

    if kv_per_token == 0 {
        let cap = practical_context_cap(model_file_size).min(FALLBACK_CTX);
        debug!("KV per token is 0, using {}K context", cap / 1024);
        return cap;
    }

    let max_ctx_from_memory = (available_for_kv / kv_per_token) as usize;
    let cap = practical_context_cap(model_file_size);
    // Clamp: at least 4096, at most min(memory allows, GGUF native, practical cap)
    let ctx = max_ctx_from_memory
        .max(4096)
        .min(gguf.context_length as usize)
        .min(cap);
    // Round down to nearest 1024
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

/// Spawn a llama-server for context compaction (summarization).
///
/// Uses GPU acceleration and matches the main model's context size so
/// large conversations can be summarized in a single LLM call.
/// Estimate VRAM needed to fully offload a GGUF model (weights + small KV cache).
///
/// Returns bytes needed for full GPU offload. Uses file size as a proxy for
/// weight memory (GGUF file ≈ quantized weights + small overhead).
pub(crate) fn estimate_model_vram(model_path: &Path, ctx_size: usize) -> u64 {
    let file_size = std::fs::metadata(model_path)
        .map(|m| m.len())
        .unwrap_or(5 * 1024 * 1024 * 1024); // 5GB fallback

    // Parse GGUF for KV cache estimation if possible.
    let kv_bytes = match parse_gguf_metadata(model_path) {
        Some(meta) => {
            let head_dim = meta.embedding_dim / meta.n_heads.max(1);
            let kv_per_token = 2 * meta.n_layers * meta.n_kv_heads * head_dim * 2;
            (kv_per_token as u64 * ctx_size as u64)
        }
        None => {
            // Rough estimate: ~2MB per 1K context for 8B model
            (ctx_size as u64 / 1024) * 2 * 1024 * 1024
        }
    };

    file_size + kv_bytes + 256 * 1024 * 1024 // weights + KV + 256MB overhead
}

/// Compute how many GPU layers a model can use given available VRAM.
///
/// If the model fits entirely, returns 99 (full offload). Otherwise,
/// proportionally allocates layers based on how much VRAM is free.
pub(crate) fn compute_gpu_layers_for_model(model_path: &Path, ctx_size: usize) -> u32 {
    let (vram, _ram) = detect_available_memory();
    let Some(free_vram) = vram else {
        return 0; // No GPU detected
    };

    let needed = estimate_model_vram(model_path, ctx_size);

    if free_vram >= needed {
        99 // Full offload — plenty of room
    } else if free_vram < 512 * 1024 * 1024 {
        0 // Less than 512MB free — CPU only
    } else {
        // Proportional: if we have 60% of needed VRAM, use ~60% of layers.
        // Parse layer count from GGUF, fall back to 32 (typical for 8B models).
        let total_layers = parse_gguf_metadata(model_path)
            .map(|m| m.n_layers)
            .unwrap_or(32) as u64;
        let proportion = free_vram as f64 / needed as f64;
        let layers = (total_layers as f64 * proportion).floor() as u32;
        layers.max(1) // At least 1 layer on GPU
    }
}

/// Configuration for spawning a llama-server process.
pub(crate) struct SpawnConfig<'a> {
    pub port: u16,
    pub model_path: &'a Path,
    pub ctx_size: usize,
    pub gpu_layers: u32,
    pub role: &'a str, // "main", "compaction", "delegation" — for error messages
}

/// Spawn a llama-server with the given configuration.
///
/// Single implementation replacing the former per-role spawn functions.
pub(crate) fn spawn_server(config: &SpawnConfig) -> Result<Child, String> {
    let home = dirs::home_dir().ok_or("Could not determine home directory")?;
    let server_path = home.join("llama.cpp/build/bin/llama-server");

    if !server_path.exists() {
        return Err(format!(
            "llama-server not found at {}",
            server_path.display()
        ));
    }
    if !config.model_path.exists() {
        return Err(format!(
            "{} model not found at {}",
            capitalize(config.role),
            config.model_path.display()
        ));
    }

    Command::new(&server_path)
        .arg("--model")
        .arg(config.model_path)
        .arg("--port")
        .arg(config.port.to_string())
        .arg("--ctx-size")
        .arg(config.ctx_size.to_string())
        .arg("--parallel")
        .arg("1")
        .arg("--n-gpu-layers")
        .arg(config.gpu_layers.to_string())
        .arg("--flash-attn")
        .arg("on")
        // Jinja required for tool calling (Mistral/Ministral templates).
        .arg("--jinja")
        // Prevent prefill errors when conversation ends with tool results.
        .arg("--no-prefill-assistant")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn {} server: {}", config.role, e))
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

/// Convenience wrapper: spawn the main model server (full GPU offload).
pub(crate) fn spawn_llama_server(
    port: u16,
    model_path: &Path,
    ctx_size: usize,
) -> Result<Child, String> {
    spawn_server(&SpawnConfig {
        port,
        model_path,
        ctx_size,
        gpu_layers: 99,
        role: "main",
    })
}

/// Convenience wrapper: spawn the compaction server (10 GPU layers).
pub(crate) fn spawn_compaction_server(
    port: u16,
    model_path: &Path,
    ctx_size: usize,
) -> Result<Child, String> {
    spawn_server(&SpawnConfig {
        port,
        model_path,
        ctx_size,
        gpu_layers: 10,
        role: "compaction",
    })
}

/// Convenience wrapper: spawn the delegation server (configurable GPU layers).
pub(crate) fn spawn_delegation_server(
    port: u16,
    model_path: &Path,
    ctx_size: usize,
    gpu_layers: u32,
) -> Result<Child, String> {
    spawn_server(&SpawnConfig {
        port,
        model_path,
        ctx_size,
        gpu_layers,
        role: "delegation",
    })
}

/// Wait for a llama-server to be healthy (responds to /health).
///
/// Polls the server's health endpoint for up to `timeout_secs`, displaying
/// a progress bar. If the server process exits unexpectedly, returns `false`.
pub(crate) async fn wait_for_server_ready(
    port: u16,
    timeout_secs: u64,
    llama_process: &mut Option<Child>,
) -> bool {
    use std::io::Write;

    // Drain stderr in a background thread so the pipe buffer doesn't block the server.
    let stderr_lines: Arc<std::sync::Mutex<Vec<String>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    if let Some(ref mut child) = llama_process {
        if let Some(stderr) = child.stderr.take() {
            let lines = stderr_lines.clone();
            std::thread::spawn(move || {
                use std::io::BufRead;
                let reader = std::io::BufReader::new(stderr);
                for line in reader.lines() {
                    if let Ok(l) = line {
                        lines.lock().unwrap().push(l);
                    }
                }
            });
        }
    }

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/health", port);
    let start = std::time::Instant::now();
    let deadline = start + std::time::Duration::from_secs(timeout_secs);
    let bar_width = 24usize;

    print!("{}", crate::tui::HIDE_CURSOR);
    std::io::stdout().flush().ok();

    while std::time::Instant::now() < deadline {
        // Check if server process crashed
        if let Some(ref mut child) = llama_process {
            if let Ok(Some(_)) = child.try_wait() {
                // Clear the bar line, show error
                print!(
                    "\r{}{}{}  ",
                    crate::tui::SHOW_CURSOR,
                    crate::tui::RESET,
                    " ".repeat(bar_width + 30)
                );
                print!(
                    "\r  {}Server exited unexpectedly{}\n",
                    crate::tui::YELLOW,
                    crate::tui::RESET
                );
                // Show last few stderr lines as hint
                let lines = stderr_lines.lock().unwrap();
                if let Some(last) = lines.last() {
                    println!("  {}{}{}", crate::tui::DIM, last, crate::tui::RESET);
                }
                std::io::stdout().flush().ok();
                return false;
            }
        }

        // Draw progress bar
        let elapsed = start.elapsed().as_secs_f64();
        let frac = (elapsed / timeout_secs as f64).min(1.0);
        let filled = (frac * bar_width as f64) as usize;
        let empty = bar_width - filled;
        print!(
            "\r  {}Loading model [{}{}{}{}{}] {:.0}s{}",
            crate::tui::DIM,
            crate::tui::RESET,
            crate::tui::CYAN,
            "\u{2588}".repeat(filled), // █
            "\u{2591}".repeat(empty),  // ░
            crate::tui::DIM,
            elapsed,
            crate::tui::RESET,
        );
        std::io::stdout().flush().ok();

        if let Ok(resp) = client.get(&url).send().await {
            if resp.status().is_success() {
                // Fill bar to 100% briefly
                print!(
                    "\r  {}Loading model [{}{}{}] done{}",
                    crate::tui::DIM,
                    crate::tui::RESET,
                    crate::tui::CYAN,
                    "\u{2588}".repeat(bar_width),
                    crate::tui::RESET,
                );
                std::io::stdout().flush().ok();
                std::thread::sleep(std::time::Duration::from_millis(200));
                print!("\r{}{}\r", crate::tui::SHOW_CURSOR, " ".repeat(bar_width + 30));
                std::io::stdout().flush().ok();
                return true;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    print!("\r{}{}\r", crate::tui::SHOW_CURSOR, " ".repeat(bar_width + 30));
    std::io::stdout().flush().ok();
    false
}

/// Quick health check: is the local llama-server at the given base URL alive?
///
/// Sends a GET to `/health` with a 2-second timeout. Returns `true` if the
/// server responds with 200, `false` otherwise.
pub(crate) async fn check_health(api_base: &str) -> bool {
    // api_base is typically "http://localhost:PORT/v1" — strip to get the root.
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

/// Background health watchdog for local llama-server processes.
///
/// Pings `/health` every 30 seconds on all active server ports. When a server
/// goes down, sends a one-line warning to `alert_tx` (displayed in the REPL
/// before the next prompt).
pub(crate) fn start_health_watchdog(
    ports: Vec<(String, String)>, // (role, port)
    alert_tx: tokio::sync::mpsc::UnboundedSender<String>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3))
            .build()
            .unwrap_or_default();

        // Track which servers were healthy on last check.
        let mut was_healthy: HashMap<String, bool> = ports
            .iter()
            .map(|(role, _)| (role.clone(), true))
            .collect();

        loop {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;

            for (role, port) in &ports {
                let url = format!("http://localhost:{}/health", port);
                let healthy = match client.get(&url).send().await {
                    Ok(resp) => resp.status().is_success(),
                    Err(_) => false,
                };

                let prev = was_healthy.get(role).copied().unwrap_or(true);
                if !healthy && prev {
                    // Server just went down — alert once.
                    let msg = format!(
                        "\x1b[RAW]\n  \x1b[31m\u{25cf}\x1b[0m \x1b[1m{} server (port {})\x1b[0m \x1b[31mDOWN\x1b[0m — use /restart or /local to recover\n",
                        role, port
                    );
                    let _ = alert_tx.send(msg);
                } else if healthy && !prev {
                    // Server recovered.
                    let msg = format!(
                        "\x1b[RAW]\n  \x1b[32m\u{25cf}\x1b[0m \x1b[1m{} server (port {})\x1b[0m \x1b[32mrecovered\x1b[0m\n",
                        role, port
                    );
                    let _ = alert_tx.send(msg);
                }

                was_healthy.insert(role.clone(), healthy);
            }
        }
    })
}

/// Query the local llama.cpp server for its actual context size (`n_ctx`).
///
/// Returns the server's context window with 5% headroom subtracted (to account
/// for token-estimation drift). Falls back to `None` if the server is
/// unreachable or the response is unexpected.
pub(crate) fn query_local_context_size(port: &str) -> Option<usize> {
    let url = format!("http://localhost:{}/props", port);
    let props = reqwest::blocking::get(&url)
        .ok()?
        .json::<serde_json::Value>()
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
    // Apply 5% headroom — our char/4 estimator can overshoot slightly.
    Some((per_request_ctx as f64 * 0.95) as usize)
}

// ============================================================================
// Tests (RED — these should FAIL until we move the implementations)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // -- practical_context_cap tests --

    #[test]
    fn test_practical_context_cap_tiny_model() {
        // < 2 GB → 8192
        assert_eq!(practical_context_cap(1_500_000_000), 8192);
    }

    #[test]
    fn test_practical_context_cap_small_model() {
        // 2-4 GB → 16384
        assert_eq!(practical_context_cap(3_000_000_000), 16384);
    }

    #[test]
    fn test_practical_context_cap_medium_model() {
        // 4-8 GB → 32768
        assert_eq!(practical_context_cap(6_000_000_000), 32768);
    }

    #[test]
    fn test_practical_context_cap_large_model() {
        // 8-16 GB → 65536
        assert_eq!(practical_context_cap(12_000_000_000), 65536);
    }

    #[test]
    fn test_practical_context_cap_xlarge_model() {
        // > 16 GB → usize::MAX (no cap)
        assert_eq!(practical_context_cap(20_000_000_000), usize::MAX);
    }

    #[test]
    fn test_practical_context_cap_boundary_2gb() {
        // Exactly at 2 GB boundary
        let two_gb = 2 * 1024 * 1024 * 1024u64;
        assert_eq!(practical_context_cap(two_gb), 16384);
        assert_eq!(practical_context_cap(two_gb - 1), 8192);
    }

    // -- find_available_port tests --

    #[test]
    fn test_find_available_port_returns_valid_port() {
        let port = find_available_port(19000);
        assert!(port >= 19000);
        assert!(port <= 19099);
        // Should be bindable
        assert!(TcpListener::bind(("127.0.0.1", port)).is_ok());
    }

    #[test]
    fn test_find_available_port_skips_occupied() {
        // Bind a port, then ask for one starting at the same number
        let listener = TcpListener::bind(("127.0.0.1", 19200)).unwrap();
        let port = find_available_port(19200);
        // Should get the next one since 19200 is occupied
        assert!(port > 19200);
        drop(listener);
    }

    // -- parse_gguf_metadata tests (synthetic GGUF) --

    /// Build a minimal GGUF v3 file header with the metadata fields we care about.
    fn build_synthetic_gguf(
        arch: &str,
        n_layers: u32,
        n_kv_heads: u32,
        n_heads: u32,
        embedding_dim: u32,
        context_length: u32,
    ) -> Vec<u8> {
        let mut buf = Vec::new();

        // Magic
        buf.extend_from_slice(b"GGUF");
        // Version 3
        buf.extend_from_slice(&3u32.to_le_bytes());
        // tensor_count = 0
        buf.extend_from_slice(&0u64.to_le_bytes());
        // kv_count = 6 (arch + 5 fields)
        buf.extend_from_slice(&6u64.to_le_bytes());

        // Helper: write a string-typed KV pair
        fn write_string_kv(buf: &mut Vec<u8>, key: &str, value: &str) {
            // key length + key bytes
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            // value type = 8 (string)
            buf.extend_from_slice(&8u32.to_le_bytes());
            // value: length + bytes
            buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
            buf.extend_from_slice(value.as_bytes());
        }

        // Helper: write a u32-typed KV pair
        fn write_u32_kv(buf: &mut Vec<u8>, key: &str, value: u32) {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            // value type = 4 (u32)
            buf.extend_from_slice(&4u32.to_le_bytes());
            buf.extend_from_slice(&value.to_le_bytes());
        }

        // 1. general.architecture = arch
        write_string_kv(&mut buf, "general.architecture", arch);
        // 2. {arch}.block_count
        write_u32_kv(&mut buf, &format!("{}.block_count", arch), n_layers);
        // 3. {arch}.attention.head_count_kv
        write_u32_kv(&mut buf, &format!("{}.attention.head_count_kv", arch), n_kv_heads);
        // 4. {arch}.attention.head_count
        write_u32_kv(&mut buf, &format!("{}.attention.head_count", arch), n_heads);
        // 5. {arch}.embedding_length
        write_u32_kv(&mut buf, &format!("{}.embedding_length", arch), embedding_dim);
        // 6. {arch}.context_length
        write_u32_kv(&mut buf, &format!("{}.context_length", arch), context_length);

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
        // Qwen3-0.6B style: small model with fewer layers
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
        // This is a filesystem test — it may return empty if ~/models/ doesn't exist
        let models = list_local_models();
        // If any models exist, they should be sorted and all .gguf
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
        // No server running on this port, should return None gracefully
        assert!(query_local_context_size("59999").is_none());
    }

    // -- find_available_port (additional) --

    #[test]
    fn test_find_port_within_range() {
        let start = 40000;
        let port = find_available_port(start);
        assert!(port >= start, "Port {} < start {}", port, start);
        assert!(port < start + 100, "Port {} >= start + 100", port);
    }

    #[test]
    fn test_find_port_skips_consecutive_occupied() {
        let base: u16 = 48500;
        let l1 = TcpListener::bind(("127.0.0.1", base));
        let l2 = TcpListener::bind(("127.0.0.1", base + 1));

        if let (Ok(_l1), Ok(_l2)) = (l1, l2) {
            let found = find_available_port(base);
            assert!(
                found >= base + 2,
                "Should skip both occupied ports, got {}",
                found
            );
        }
    }

    #[test]
    fn test_find_port_high_start_no_overflow() {
        let port = find_available_port(65500);
        assert!(port >= 65500);
    }

    // -- spawn_llama_server --

    #[test]
    fn test_spawn_server_errors_when_binary_missing() {
        let home = dirs::home_dir().unwrap();
        let server_path = home.join("llama.cpp/build/bin/llama-server");

        if !server_path.exists() {
            let fake_model = home.join("models/nonexistent.gguf");
            let result = spawn_llama_server(19876, &fake_model, 8192);
            assert!(result.is_err());
            assert!(
                result.unwrap_err().contains("llama-server not found"),
                "Should report missing binary"
            );
        }
    }

    #[test]
    fn test_spawn_server_errors_when_model_missing() {
        let home = dirs::home_dir().unwrap();
        let server_path = home.join("llama.cpp/build/bin/llama-server");
        let model_path = home.join("models/nonexistent-test-model.gguf");

        if server_path.exists() {
            let result = spawn_llama_server(19877, &model_path, 8192);
            assert!(result.is_err());
            assert!(
                result.unwrap_err().contains("model not found"),
                "Should report missing model"
            );
        }
    }

    // -- wait_for_server_ready --

    #[tokio::test]
    async fn test_wait_timeout_when_no_server() {
        let mut proc = None;
        let result = wait_for_server_ready(19999, 1, &mut proc).await;
        assert!(!result, "Should return false when no server running");
    }

    #[tokio::test]
    async fn test_wait_zero_timeout_returns_false() {
        let mut proc = None;
        let result = wait_for_server_ready(19998, 0, &mut proc).await;
        assert!(!result, "Should return false immediately with zero timeout");
    }

    #[tokio::test]
    async fn test_wait_finds_healthy_server() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        tokio::spawn(async move {
            loop {
                if let Ok((mut stream, _)) = listener.accept().await {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    let mut buf = [0u8; 1024];
                    let _ = stream.read(&mut buf).await;
                    let resp =
                        "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 2\r\n\r\nok";
                    stream.write_all(resp.as_bytes()).await.ok();
                }
            }
        });

        let mut proc = None;
        let result = wait_for_server_ready(port, 5, &mut proc).await;
        assert!(result, "Should detect the healthy server");
    }

    #[tokio::test]
    async fn test_wait_retries_on_503_then_succeeds() {
        use std::sync::atomic::AtomicUsize;

        let request_count = Arc::new(AtomicUsize::new(0));
        let count_clone = request_count.clone();

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        tokio::spawn(async move {
            loop {
                if let Ok((mut stream, _)) = listener.accept().await {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    let mut buf = [0u8; 1024];
                    let _ = stream.read(&mut buf).await;

                    let n = count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let resp = if n < 2 {
                        "HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\nContent-Length: 7\r\n\r\nloading"
                    } else {
                        "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 2\r\n\r\nok"
                    };
                    stream.write_all(resp.as_bytes()).await.ok();
                }
            }
        });

        let mut proc = None;
        let result = wait_for_server_ready(port, 10, &mut proc).await;
        assert!(result, "Should succeed after retries");
        assert!(
            request_count.load(std::sync::atomic::Ordering::SeqCst) >= 3,
            "Should have retried at least 3 times"
        );
    }

    // -- Integration: real GGUF model parsing --

    #[test]
    fn test_parse_real_gguf_if_available() {
        let models = list_local_models();
        if models.is_empty() {
            eprintln!("Skipping: no GGUF models found in ~/models/");
            return;
        }

        for model_path in &models {
            let name = model_path.file_name().unwrap().to_string_lossy();
            match parse_gguf_metadata(model_path) {
                Some(info) => {
                    // Sanity checks — all real models should have sane values
                    assert!(info.n_layers > 0, "{}: n_layers should be > 0", name);
                    assert!(info.n_layers <= 256, "{}: n_layers {} seems too high", name, info.n_layers);
                    assert!(info.n_heads > 0, "{}: n_heads should be > 0", name);
                    assert!(info.n_kv_heads > 0, "{}: n_kv_heads should be > 0", name);
                    assert!(info.n_kv_heads <= info.n_heads, "{}: n_kv_heads {} > n_heads {}", name, info.n_kv_heads, info.n_heads);
                    assert!(info.embedding_dim >= 64, "{}: embedding_dim {} too small", name, info.embedding_dim);
                    assert!(info.context_length >= 512, "{}: context_length {} too small", name, info.context_length);
                    eprintln!(
                        "  OK: {} — layers={}, heads={}/{}, embed={}, ctx={}",
                        name, info.n_layers, info.n_heads, info.n_kv_heads,
                        info.embedding_dim, info.context_length
                    );
                }
                None => {
                    eprintln!("  SKIP: {} — could not parse GGUF header", name);
                }
            }
        }
    }

    #[test]
    fn test_compute_optimal_context_real_models() {
        let models = list_local_models();
        if models.is_empty() {
            eprintln!("Skipping: no GGUF models found in ~/models/");
            return;
        }

        for model_path in &models {
            let name = model_path.file_name().unwrap().to_string_lossy();
            let ctx = compute_optimal_context_size(model_path);
            // Context should be at least 4096 and a multiple of 1024
            assert!(ctx >= 4096, "{}: ctx {} < 4096", name, ctx);
            assert_eq!(ctx % 1024, 0, "{}: ctx {} not aligned to 1024", name, ctx);
            let file_size = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);
            eprintln!("  OK: {} ({:.1}GB) → {}K context", name, file_size as f64 / 1e9, ctx / 1024);
        }
    }

    #[test]
    fn test_detect_available_memory_sane() {
        let (vram, ram) = detect_available_memory();
        // RAM should be at least 1 GB on any modern system
        assert!(ram >= 1_000_000_000, "RAM {} seems too low", ram);
        if let Some(v) = vram {
            // If VRAM detected, should be at least 256 MB
            assert!(v >= 256 * 1024 * 1024, "VRAM {} seems too low", v);
            eprintln!("  VRAM: {:.1} GB, RAM: {:.1} GB", v as f64 / 1e9, ram as f64 / 1e9);
        } else {
            eprintln!("  No GPU detected. RAM: {:.1} GB", ram as f64 / 1e9);
        }
    }

    // -- find_delegation_model / pick_preferred_model tests --

    #[test]
    fn test_find_delegation_model_preference_constants() {
        assert_eq!(DELEGATION_MODEL_PREFERENCES[0], "Ministral-3");
        assert_eq!(DELEGATION_MODEL_PREFERENCES[1], "Qwen3-0.6B");
        assert_eq!(DELEGATION_MODEL_PREFERENCES[2], "Ministral-8B");
        assert_eq!(DELEGATION_MODEL_PREFERENCES[3], "Nemotron-Nano-9B");
    }

    #[test]
    fn test_pick_preferred_model_empty_list() {
        let models: Vec<PathBuf> = vec![];
        assert!(pick_preferred_model(&models, DELEGATION_MODEL_PREFERENCES).is_none());
    }

    #[test]
    fn test_pick_preferred_model_no_match() {
        let models = vec![
            PathBuf::from("/models/llama-70B.gguf"),
            PathBuf::from("/models/qwen-7B.gguf"),
        ];
        assert!(pick_preferred_model(&models, DELEGATION_MODEL_PREFERENCES).is_none());
    }

    #[test]
    fn test_pick_preferred_model_single_match() {
        let models = vec![
            PathBuf::from("/models/llama-70B.gguf"),
            PathBuf::from("/models/Nemotron-Nano-9B-Q4.gguf"),
        ];
        let result = pick_preferred_model(&models, DELEGATION_MODEL_PREFERENCES);
        assert_eq!(
            result.as_deref(),
            Some(Path::new("/models/Nemotron-Nano-9B-Q4.gguf"))
        );
    }

    #[test]
    fn test_pick_preferred_model_priority_ordering() {
        // Both Ministral-3 and Nemotron present — Ministral-3 wins (higher priority)
        let models = vec![
            PathBuf::from("/models/Nemotron-Nano-9B-v2-Q4_K_M.gguf"),
            PathBuf::from("/models/Ministral-3-Instruct-Q4_K_M.gguf"),
        ];
        let result = pick_preferred_model(&models, DELEGATION_MODEL_PREFERENCES);
        assert!(
            result.as_ref().unwrap().to_string_lossy().contains("Ministral-3"),
            "Ministral-3 should beat Nemotron, got: {:?}",
            result
        );
    }

    #[test]
    fn test_pick_preferred_model_all_three_present() {
        let models = vec![
            PathBuf::from("/models/Nemotron-Nano-9B.gguf"),
            PathBuf::from("/models/Ministral-3-Q4.gguf"),
            PathBuf::from("/models/Ministral-8B-Q4.gguf"),
        ];
        let result = pick_preferred_model(&models, DELEGATION_MODEL_PREFERENCES);
        assert!(
            result.as_ref().unwrap().to_string_lossy().contains("Ministral-3"),
            "Ministral-3 is highest priority, got: {:?}",
            result
        );
    }

    #[test]
    fn test_pick_preferred_model_case_insensitive() {
        let models = vec![
            PathBuf::from("/models/MINISTRAL-8B-INSTRUCT.gguf"),
        ];
        let result = pick_preferred_model(&models, DELEGATION_MODEL_PREFERENCES);
        assert!(result.is_some(), "Should match case-insensitively");
    }

    #[test]
    fn test_pick_preferred_model_mixed_case() {
        let models = vec![
            PathBuf::from("/models/ministral-8b-instruct-Q4_K_M.gguf"),
        ];
        let result = pick_preferred_model(&models, DELEGATION_MODEL_PREFERENCES);
        assert!(result.is_some(), "Should match lowercase filename against mixed-case preference");
    }

    #[test]
    fn test_pick_preferred_model_substring_match() {
        // "Ministral-3" should match "Ministral-3B-Instruct-Q4.gguf" via substring
        let models = vec![
            PathBuf::from("/models/Ministral-3B-Instruct-Q4.gguf"),
        ];
        let result = pick_preferred_model(&models, DELEGATION_MODEL_PREFERENCES);
        assert!(result.is_some(), "Substring match should work");
    }

    #[test]
    fn test_pick_preferred_model_empty_preferences() {
        let models = vec![PathBuf::from("/models/anything.gguf")];
        let prefs: &[&str] = &[];
        assert!(pick_preferred_model(&models, prefs).is_none());
    }

    #[test]
    fn test_find_delegation_model_uses_real_filesystem() {
        // Integration test: delegates to pick_preferred_model with real ~/models/
        let result = find_delegation_model();
        if let Some(ref path) = result {
            let name = path.file_name().unwrap().to_string_lossy().to_lowercase();
            let matches_any = DELEGATION_MODEL_PREFERENCES
                .iter()
                .any(|pref| name.contains(&pref.to_lowercase()));
            assert!(matches_any, "Real match '{}' should be in preferences", name);
            eprintln!("  Found delegation model: {}", path.display());
        } else {
            eprintln!("  No delegation model in ~/models/ (OK in CI)");
        }
    }

    // -- stop_delegation_server tests --

    #[test]
    fn test_stop_delegation_server_clears_state() {
        let mut process: Option<Child> = None;
        let mut port: Option<String> = Some("8091".to_string());

        stop_delegation_server(&mut process, &mut port);

        assert!(process.is_none());
        assert!(port.is_none());
    }

    #[test]
    fn test_stop_delegation_server_noop_when_empty() {
        let mut process: Option<Child> = None;
        let mut port: Option<String> = None;

        // Should not panic when nothing to stop
        stop_delegation_server(&mut process, &mut port);

        assert!(process.is_none());
        assert!(port.is_none());
    }

    #[test]
    fn test_stop_delegation_server_kills_real_process() {
        // Spawn a harmless process, then stop it
        let child = Command::new("sleep")
            .arg("60")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();

        if let Ok(child) = child {
            let pid = child.id();
            let mut process = Some(child);
            let mut port = Some("8091".to_string());

            stop_delegation_server(&mut process, &mut port);

            assert!(process.is_none());
            assert!(port.is_none());

            // Verify the process is actually dead
            let status = Command::new("kill")
                .args(["-0", &pid.to_string()])
                .status();
            // kill -0 should fail (process doesn't exist)
            if let Ok(s) = status {
                assert!(!s.success(), "Process {} should be dead after stop", pid);
            }
        }
    }

    // -- Delegation vs compaction server symmetry --

    #[test]
    fn test_stop_delegation_mirrors_stop_compaction() {
        // Both stop functions should behave identically for state cleanup
        let mut d_process: Option<Child> = None;
        let mut d_port: Option<String> = Some("8091".to_string());
        let mut c_process: Option<Child> = None;
        let mut c_port: Option<String> = Some("8090".to_string());

        stop_delegation_server(&mut d_process, &mut d_port);
        stop_compaction_server(&mut c_process, &mut c_port);

        assert_eq!(d_process.is_none(), c_process.is_none());
        assert_eq!(d_port.is_none(), c_port.is_none());
    }
}
