//! Local LLM server management: llama-server spawn/health, GGUF parser, context sizing.
//!
//! Handles spawning and managing llama.cpp server processes (main model + compaction),
//! parsing GGUF model metadata for auto-sizing context windows, and detecting available
//! system memory (VRAM/RAM).

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
        child.kill().ok();
        child.wait().ok();
    }
    *compaction_process = None;
    *compaction_port = None;
}

/// Kill any orphaned llama-server processes from previous runs.
pub(crate) fn kill_stale_llama_servers() {
    // Kill any orphaned llama-server processes from previous runs
    let _ = Command::new("pkill")
        .args(["-f", "llama-server"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    // Brief pause to let ports be released
    std::thread::sleep(std::time::Duration::from_millis(300));
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
pub(crate) fn spawn_compaction_server(
    port: u16,
    model_path: &Path,
    ctx_size: usize,
) -> Result<Child, String> {
    let home = dirs::home_dir().ok_or("Could not determine home directory")?;
    let server_path = home.join("llama.cpp/build/bin/llama-server");

    if !server_path.exists() {
        return Err(format!(
            "llama-server not found at {}",
            server_path.display()
        ));
    }
    if !model_path.exists() {
        return Err(format!(
            "Compaction model not found at {}",
            model_path.display()
        ));
    }

    Command::new(&server_path)
        .arg("--model")
        .arg(model_path)
        .arg("--port")
        .arg(port.to_string())
        .arg("--ctx-size")
        .arg(ctx_size.to_string())
        .arg("--parallel")
        .arg("1")
        .arg("--n-gpu-layers")
        .arg("10")
        .arg("--flash-attn")
        .arg("on")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn compaction server: {}", e))
}

/// Spawn a llama-server for the main model.
pub(crate) fn spawn_llama_server(
    port: u16,
    model_path: &Path,
    ctx_size: usize,
) -> Result<Child, String> {
    let home = dirs::home_dir().ok_or("Could not determine home directory")?;
    let server_path = home.join("llama.cpp/build/bin/llama-server");

    if !server_path.exists() {
        return Err(format!(
            "llama-server not found at {}",
            server_path.display()
        ));
    }
    if !model_path.exists() {
        return Err(format!("Model not found at {}", model_path.display()));
    }

    Command::new(&server_path)
        .arg("--model")
        .arg(model_path)
        .arg("--port")
        .arg(port.to_string())
        .arg("--ctx-size")
        .arg(ctx_size.to_string())
        .arg("--parallel")
        .arg("1")
        .arg("--n-gpu-layers")
        .arg("99")
        .arg("--flash-attn")
        .arg("on")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn llama-server: {}", e))
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
                result.unwrap_err().contains("Model not found"),
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
}
