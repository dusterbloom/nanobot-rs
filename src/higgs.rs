//! Higgs inference server sidecar management.
//!
//! Higgs is a pure-Rust MLX inference server for Apple Silicon. When
//! `localBackend == "higgs"`, nanobot auto-starts Higgs as a detached
//! background daemon and manages its lifecycle via PID file.
//!
//! The server persists between nanobot sessions so model loading only
//! happens once. Subsequent `nanobot agent -l` invocations detect the
//! running instance and skip startup.

use std::fs;
use std::path::{Path, PathBuf};

/// Find the `higgs` binary on the system.
///
/// Search order:
/// 1. `HIGGS_BIN` env var (explicit override, e.g. for development builds)
/// 2. `~/.cargo/bin/higgs` (cargo install location)
/// 3. `higgs` on PATH (via `which`)
pub(crate) fn find_binary() -> Option<PathBuf> {
    // 1. HIGGS_BIN env var (highest priority — dev builds and CI overrides)
    if let Ok(bin) = std::env::var("HIGGS_BIN") {
        let path = PathBuf::from(bin);
        if path.exists() {
            return Some(path);
        }
    }

    // 2. ~/.cargo/bin/higgs (most common for Rust tools installed via cargo install)
    if let Some(home) = dirs::home_dir() {
        let cargo_bin = home.join(".cargo/bin/higgs");
        if cargo_bin.exists() {
            return Some(cargo_bin);
        }
    }

    // 3. Check PATH
    if let Ok(output) = std::process::Command::new("which").arg("higgs").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }

    None
}

/// PID file path: `~/.nanobot/higgs.pid`.
fn pid_path() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".nanobot").join("higgs.pid")
}

/// Log file path: `~/.nanobot/higgs.log`.
fn log_path() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".nanobot").join("higgs.log")
}

/// Read the stored PID, if any.
fn read_pid() -> Option<u32> {
    fs::read_to_string(pid_path())
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

/// Check if a process is alive.
fn pid_is_alive(pid: u32) -> bool {
    platform::is_process_alive(pid)
}

/// Outcome of `server_start`.
pub(crate) enum StartResult {
    /// Server is healthy and ready.
    Ready,
    /// Process spawned and alive but not yet healthy (large model still loading).
    /// Caller may still route requests — they will retry until the server responds.
    Loading { pid: u32, port: u16 },
}

/// Start Higgs serving a model on the given port.
///
/// If Higgs is already running and healthy, returns `Ready` immediately.
/// Otherwise spawns a new detached instance and waits for readiness.
///
/// Returns `Loading` when the process is alive but health check timed out
/// (large model still loading). Returns `Err` for hard failures.
pub(crate) async fn server_start(
    bin: &Path,
    port: u16,
    model_dir: &str,
) -> Result<StartResult, String> {
    // Already running and healthy?
    if let Some(pid) = read_pid() {
        if pid_is_alive(pid) && wait_for_ready(port, 3).await {
            if is_serving_expected_model(port, model_dir).await {
                tracing::info!(pid, port, "higgs already running");
                return Ok(StartResult::Ready);
            }
            tracing::info!(
                pid,
                port,
                "higgs running but serving wrong model, restarting"
            );
            server_stop()?;
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        } else {
            let _ = fs::remove_file(pid_path());
        }
    }

    if wait_for_ready(port, 1).await {
        if is_serving_expected_model(port, model_dir).await {
            tracing::info!(port, "higgs port already responding (externally managed)");
            return Ok(StartResult::Ready);
        }
        return Err(format!(
            "port {port} is serving a different model than configured.\n\
             Stop the existing server and retry, or update mlxModelDir to match."
        ));
    }

    if let Some(existing_pid) = find_existing_higgs_process() {
        return Err(format!(
            "another Higgs process is already running (pid {existing_pid}).\n\
             Stop it first, or point localApiBase at its port to reuse it.\n\
             To force nanobot to start its own instance: kill {existing_pid}"
        ));
    }

    if let Some(parent) = pid_path().parent() {
        let _ = fs::create_dir_all(parent);
    }

    let log_file =
        fs::File::create(log_path()).map_err(|e| format!("failed to create higgs log: {e}"))?;
    let log_err = log_file
        .try_clone()
        .map_err(|e| format!("failed to clone log handle: {e}"))?;

    let devnull =
        fs::File::open("/dev/null").map_err(|e| format!("failed to open /dev/null: {e}"))?;

    let mut cmd = std::process::Command::new(bin);
    // Maximise local-MLX throughput: throughput profile, TurboQuant KV @ 4-bit.
    // TurboQuant activates above its internal threshold (2048 tokens) so short
    // turns are unaffected; long-context decode benefits from compressed KV.
    cmd.args([
        "serve",
        "--model",
        model_dir,
        "--port",
        &port.to_string(),
        "--mlx-profile",
        "throughput",
        "--kv-cache",
        "turboquant",
        "--kv-bits",
        "8",
    ]);
    cmd.stdin(devnull);
    cmd.stdout(log_file);
    cmd.stderr(log_err);

    platform::set_new_session(&mut cmd);

    let child = cmd
        .spawn()
        .map_err(|e| format!("failed to spawn higgs: {e}"))?;

    let child_pid = child.id();

    fs::write(pid_path(), child_pid.to_string())
        .map_err(|e| format!("failed to write higgs PID file: {e}"))?;

    // Reaper thread: wait for higgs to exit and log how it died.
    //
    // Higgs has been observed to die silently mid-request with no trace in
    // its own log, no `ReportCrash` artifact, and no kernel-level kill
    // message. Logging the exit status from the parent gives us the
    // POSIX signal (SIGKILL, SIGSEGV, SIGBUS, etc.) which is the missing
    // signal in the crash forensics.
    std::thread::spawn(move || {
        let mut child = child;
        match child.wait() {
            Ok(status) => {
                #[cfg(unix)]
                {
                    use std::os::unix::process::ExitStatusExt as _;
                    let code = status.code();
                    let signal = status.signal();
                    let core_dumped = status.core_dumped();
                    tracing::warn!(
                        pid = child_pid,
                        ?code,
                        ?signal,
                        core_dumped,
                        "higgs child exited",
                    );
                }
                #[cfg(not(unix))]
                {
                    tracing::warn!(pid = child_pid, ?status, "higgs child exited");
                }
            }
            Err(e) => {
                tracing::warn!(pid = child_pid, error = %e, "higgs child wait failed");
            }
        }
    });

    tracing::info!(pid = child_pid, port, model_dir, "higgs spawned");

    if !wait_for_ready(port, 60).await {
        if pid_is_alive(child_pid) {
            tracing::warn!(
                pid = child_pid,
                port,
                "higgs still loading after 60s, routing requests anyway"
            );
            return Ok(StartResult::Loading {
                pid: child_pid,
                port,
            });
        }
        return Err(format!(
            "higgs (pid {child_pid}) exited before becoming healthy on port {port}\n\
             check log: {}",
            log_path().display()
        ));
    }

    Ok(StartResult::Ready)
}

/// Stop a running Higgs instance.
pub(crate) fn server_stop() -> Result<(), String> {
    let Some(pid) = read_pid() else {
        return Ok(());
    };

    if !pid_is_alive(pid) {
        let _ = fs::remove_file(pid_path());
        return Ok(());
    }

    platform::send_signal(pid, libc::SIGTERM);

    // Wait briefly for graceful shutdown
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while pid_is_alive(pid) {
        if std::time::Instant::now() >= deadline {
            tracing::warn!(pid, "higgs still running after SIGTERM, sending SIGKILL");
            platform::send_signal(pid, libc::SIGKILL);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    let _ = fs::remove_file(pid_path());
    tracing::info!(pid, "higgs stopped");
    Ok(())
}

/// Check if any `higgs serve` process is already running on this system.
///
/// Uses `pgrep` to find processes matching "higgs". Returns the first PID found,
/// excluding our own nanobot-managed PID (if still tracked and alive on the
/// expected port, we already returned early in `server_start`).
fn find_existing_higgs_process() -> Option<u32> {
    let output = std::process::Command::new("pgrep")
        .args(["-f", "higgs serve"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None; // no matches
    }

    let text = String::from_utf8_lossy(&output.stdout);
    // Return the first PID that isn't our own process
    let my_pid = std::process::id();
    text.lines()
        .filter_map(|l| l.trim().parse::<u32>().ok())
        .find(|&pid| pid != my_pid)
}

/// Wait for the Higgs health endpoint to respond.
async fn wait_for_ready(port: u16, timeout_secs: u64) -> bool {
    let url = format!("http://127.0.0.1:{port}/health");
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);

    while tokio::time::Instant::now() < deadline {
        if let Ok(resp) = reqwest::get(&url).await {
            if resp.status().is_success() {
                return true;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    false
}

/// Resolve the model directory for Higgs from config.
///
/// Priority: `mlxModelDir` → error.
pub(crate) fn resolve_model_dir(config: &crate::config::schema::Config) -> Result<String, String> {
    if let Some(ref dir) = config.agents.defaults.mlx_model_dir {
        if !dir.is_empty() && *dir != "auto" {
            return Ok(dir.clone());
        }
    }

    Err("no model directory configured for Higgs.\n\
         Set mlxModelDir in ~/.nanobot/config.json to the path of an MLX model directory\n\
         (e.g. ~/.cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit)"
        .to_string())
}

/// Restart Higgs with a (potentially different) model.
///
/// Stops the running instance, waits briefly for port release, then starts
/// with the new model directory.
pub(crate) async fn server_restart(
    bin: &Path,
    port: u16,
    model_dir: &str,
) -> Result<StartResult, String> {
    server_stop()?;
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    server_start(bin, port, model_dir).await
}

/// Check if the running Higgs is serving a model that matches `expected_dir`.
///
/// Scans the FULL set of served models (an externally-managed Higgs can serve
/// several at once) and matches each against the last path component of
/// `expected_dir`. Returns `true` if any served model matches, or if the
/// server can't be queried (optimistic fallback).
async fn is_serving_expected_model(port: u16, expected_dir: &str) -> bool {
    let served = list_served_models(port).await;
    if served.is_empty() {
        return true; // Can't query → assume ok
    }
    let expected_name = Path::new(expected_dir)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    if expected_name.is_empty() {
        return true;
    }
    served.iter().any(|id| model_id_matches(id, &expected_name))
}

/// Case-insensitive fuzzy match between two model identifiers.
///
/// Model ids reach us as full paths, directory basenames, or short names, so a
/// match is accepted when either id contains the other after lowercasing
/// (e.g. dir basename `MiniCPM5-1B-4bit` matches served id `minicpm5-1b`).
fn model_id_matches(a: &str, b: &str) -> bool {
    let a = a.to_lowercase();
    let b = b.to_lowercase();
    !a.is_empty() && !b.is_empty() && (a.contains(&b) || b.contains(&a))
}

/// List every model id a running Higgs instance is serving (via /v1/models).
pub(crate) async fn list_served_models(port: u16) -> Vec<String> {
    let url = format!("http://127.0.0.1:{port}/v1/models");
    let Ok(resp) = reqwest::get(&url).await else {
        return Vec::new();
    };
    let Ok(json) = resp.json::<serde_json::Value>().await else {
        return Vec::new();
    };
    json.get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|id| id.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Resolve the served model id that best matches `preferred`.
///
/// nanobot's own auto-started Higgs serves a single model named after its
/// model directory, so `preferred` matches directly. An externally-managed
/// Higgs may serve SEVERAL models under arbitrary ids — there we must pick the
/// one the user actually configured rather than blindly taking the first
/// entry, otherwise nanobot requests a model id Higgs has not loaded (→ 404).
///
/// Falls back to the first served model when nothing matches `preferred`.
pub(crate) async fn resolve_served_model(port: u16, preferred: &str) -> Option<String> {
    let served = list_served_models(port).await;
    served
        .iter()
        .find(|id| model_id_matches(id, preferred))
        .cloned()
        .or_else(|| served.into_iter().next())
}

#[allow(unsafe_code)]
mod platform {
    /// Check if a process is alive (signal 0 checks existence without sending a signal).
    pub(super) fn is_process_alive(pid: u32) -> bool {
        unsafe { libc::kill(pid as i32, 0) == 0 }
    }

    /// Send a signal to a process.
    pub(super) fn send_signal(pid: u32, signal: libc::c_int) {
        unsafe {
            libc::kill(pid as i32, signal);
        }
    }

    /// Configure a Command to create a new session (setsid) via pre_exec.
    pub(super) fn set_new_session(cmd: &mut std::process::Command) {
        // SAFETY: setsid is async-signal-safe per POSIX.
        unsafe {
            std::os::unix::process::CommandExt::pre_exec(cmd, || {
                libc::setsid();
                Ok(())
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pid_path_under_nanobot_dir() {
        let path = pid_path();
        assert!(path.to_string_lossy().contains(".nanobot"));
        assert!(path.to_string_lossy().ends_with("higgs.pid"));
    }

    #[test]
    fn test_read_pid_returns_none_when_no_file() {
        // PID file likely doesn't exist in test environment
        // Just verify it doesn't panic
        let _ = read_pid();
    }

    #[tokio::test]
    async fn test_wait_for_ready_unreachable_returns_false() {
        let result = wait_for_ready(19998, 1).await;
        assert!(!result, "unreachable port should return false");
    }

    #[test]
    fn test_model_id_matches() {
        // The regression that motivated this: a model-dir basename must match
        // the shorter served id (and vice versa), case-insensitively.
        assert!(model_id_matches("MiniCPM5-1B-4bit", "minicpm5-1b"));
        assert!(model_id_matches("minicpm5-1b", "MiniCPM5-1B-4bit"));
        // Symmetric and case-insensitive in general.
        assert!(model_id_matches("Qwen3-8B", "qwen3-8b"));
        // Distinct models served side-by-side must NOT match each other.
        assert!(!model_id_matches("bonsai-8b-mlx", "minicpm5-1b"));
        // Empty ids never match (avoids vacuous substring hits).
        assert!(!model_id_matches("", "minicpm5-1b"));
        assert!(!model_id_matches("minicpm5-1b", ""));
    }
}
