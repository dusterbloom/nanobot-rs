//! Local LLM utilities: health checks, GGUF parser, model listing, context sizing.
//!
//! Server lifecycle (spawning, process management) is handled by LM Studio via
//! the `lms` module. This module provides shared utilities used across the codebase.

// Interactive/app boundary (error-protocol layer 3 backlog): printing IS the
// product here (REPL/TUI/CLI), and the thin glue code keeps pragmatic
// unwraps on always-set state (rl, runtime, static regexes). The deny regime
// in Cargo.toml stays live for the core; this module lands on the regime
// when its backlog is migrated.
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::shadow_reuse,
    clippy::shadow_unrelated,
    clippy::shadow_same,
    clippy::format_push_string,
    clippy::string_add
)]
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use tracing::debug;

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
    // Apple Silicon: unified memory, no discrete VRAM and no /proc/meminfo.
    // Report total physical memory as the RAM figure; callers apply their own
    // headroom reserve. Without this, macOS falls through to the 8 GB fallback
    // below and wildly under-budgets context on 32/64/128 GB machines.
    #[cfg(target_os = "macos")]
    {
        let total = Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .filter(|out| out.status.success())
            .and_then(|out| {
                String::from_utf8_lossy(&out.stdout)
                    .trim()
                    .parse::<u64>()
                    .ok()
            })
            .unwrap_or(8 * 1024 * 1024 * 1024);
        (None, total)
    }

    #[cfg(not(target_os = "macos"))]
    {
        detect_available_memory_linux()
    }
}

#[cfg(not(target_os = "macos"))]
fn detect_available_memory_linux() -> (Option<u64>, u64) {
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

// ============================================================================
// VRAM Budget Pure Functions
// ============================================================================

// ============================================================================
// Health Checks
// ============================================================================

/// Quick health check: is the local server at the given base URL alive?
///
/// Sends a GET to `/health` with the specified timeout.
pub(crate) async fn check_health(api_base: &str, timeout_secs: u64) -> bool {
    let base = api_base
        .trim_end_matches('/')
        .trim_end_matches("/v1")
        .trim_end_matches('/');
    let url = format!("{}/health", base);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .build()
        .unwrap_or_default();
    match client.get(&url).send().await {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
}

/// Lightweight health check for a local server by port number.
///
/// Delegates to `check_health` with a localhost URL.
pub(crate) async fn check_local_health(port: &str) -> bool {
    let base = format!("http://localhost:{}", port);
    check_health(&base, 5).await
}

// ============================================================================
// Context Size Query
// ============================================================================

/// Strip a trailing `/v1` (and trailing slashes) from a base URL.
fn strip_v1_suffix(api_base: &str) -> &str {
    let trimmed = api_base.trim_end_matches('/');
    trimmed
        .strip_suffix("/v1")
        .unwrap_or(trimmed)
        .trim_end_matches('/')
}

/// Parse `/props` JSON into a per-request context size with 5% headroom.
fn parse_props_n_ctx(props: &serde_json::Value) -> Option<usize> {
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

/// Query a `/props` endpoint at an arbitrary base URL.
///
/// Returns the server's per-request context window with 5% headroom subtracted.
/// Works for any llama.cpp-style server (localhost or cluster peer).
pub(crate) fn query_context_size_from_url(api_base: &str) -> Option<usize> {
    let url = format!("{}/props", strip_v1_suffix(api_base));
    let props = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::try_current();
        match rt {
            Ok(handle) => handle.block_on(async {
                reqwest::Client::new()
                    .get(&url)
                    .timeout(std::time::Duration::from_secs(3))
                    .send()
                    .await
                    .ok()?
                    .json::<serde_json::Value>()
                    .await
                    .ok()
            }),
            Err(_) => reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(3))
                .build()
                .ok()?
                .get(&url)
                .send()
                .ok()?
                .json::<serde_json::Value>()
                .ok(),
        }
    })?;
    parse_props_n_ctx(&props)
}

/// Query the local server for its actual context size (`n_ctx`).
///
/// Returns the server's context window with 5% headroom subtracted.
pub(crate) fn query_local_context_size(port: &str) -> Option<usize> {
    query_context_size_from_url(&format!("http://localhost:{}", port))
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
/// Pings `/health` every `poll_interval_secs` on all active server ports. When a server
/// fails `degraded_threshold` consecutive health checks, sends a restart request through
/// `restart_tx`.
pub(crate) fn start_health_watchdog_with_autorepair(
    ports: Vec<(String, String)>, // (role, port)
    alert_tx: tokio::sync::mpsc::UnboundedSender<String>,
    restart_tx: tokio::sync::mpsc::UnboundedSender<RestartRequest>,
    inference_active: std::sync::Arc<std::sync::atomic::AtomicBool>,
    poll_interval_secs: u64,
    degraded_threshold: u32,
    health_check_timeout_secs: u64,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(health_check_timeout_secs))
            .build()
            .unwrap_or_default();

        let mut was_healthy: HashMap<String, bool> =
            ports.iter().map(|(role, _)| (role.clone(), true)).collect();

        let mut consecutive_failures: HashMap<String, u32> =
            ports.iter().map(|(role, _)| (role.clone(), 0)).collect();

        // Track whether each server has ever responded successfully.
        // Until a server passes its first health check, we assume it's still
        // starting up (loading model weights) and don't count failures.
        let mut seen_healthy: HashMap<String, bool> = ports
            .iter()
            .map(|(role, _)| (role.clone(), false))
            .collect();

        let mut cooldown_remaining: u32 = 0;

        loop {
            tokio::time::sleep(std::time::Duration::from_secs(poll_interval_secs)).await;

            if inference_active.load(std::sync::atomic::Ordering::Relaxed) {
                // Reset stale failure counts during inference so they don't
                // trigger a restart the moment inference ends.
                for v in consecutive_failures.values_mut() {
                    *v = 0;
                }
                cooldown_remaining = 2; // grace period after inference ends
                continue;
            }

            if cooldown_remaining > 0 {
                cooldown_remaining -= 1;
                continue;
            }

            for (role, port) in &ports {
                let url = format!("http://localhost:{}/health", port);
                let healthy = match client.get(&url).send().await {
                    Ok(resp) => resp.status().is_success(),
                    Err(_) => false,
                };

                let prev = was_healthy.get(role).copied().unwrap_or(true);

                if !healthy {
                    // Server hasn't passed a health check yet — still in
                    // initial startup (loading model weights). Don't count
                    // these as failures.
                    if !seen_healthy.get(role).copied().unwrap_or(false) {
                        continue;
                    }

                    *consecutive_failures.get_mut(role).unwrap() += 1;
                    let failures = consecutive_failures[role];

                    if failures < degraded_threshold && prev {
                        let msg = format!(
                            "\x1b[RAW]\n  \x1b[33m\u{25cf}\x1b[0m \x1b[1m{} server\x1b[0m (port {}) \x1b[33munhealthy\x1b[0m (attempt {}/{})\n",
                            role, port, failures, degraded_threshold
                        );
                        let _ = alert_tx.send(msg);
                    } else if failures >= degraded_threshold {
                        let msg = format!(
                            "\x1b[RAW]\n  \x1b[33m\u{25cf}\x1b[0m \x1b[1m{} server\x1b[0m auto-restarting...\n",
                            role
                        );
                        let _ = alert_tx.send(msg);
                        let _ = restart_tx.send(RestartRequest { role: role.clone() });
                        consecutive_failures.insert(role.clone(), 0);
                    }
                } else {
                    if !seen_healthy.get(role).copied().unwrap_or(false) {
                        seen_healthy.insert(role.clone(), true);
                    }
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
}
