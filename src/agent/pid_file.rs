// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(clippy::as_conversions)]
//! PID file management for managed child processes.
//!
//! Tracks child process PIDs in `~/.nanobot/pids/{name}-{port}.pid` so that
//! stale processes from previous crashed/killed nanobot runs can be cleaned up
//! on next startup.

use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Directory for PID files: `~/.nanobot/pids/`.
pub fn pids_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_default()
        .join(".nanobot")
        .join("pids")
}

/// Send SIGTERM, wait up to 2s, then SIGKILL if still alive.
fn graceful_kill(pid: u32) {
    if !platform::is_process_alive(pid) {
        return; // already dead
    }
    platform::send_signal(pid, libc::SIGTERM);
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(100));
        if !platform::is_process_alive(pid) {
            return; // died from SIGTERM
        }
    }
    // Still alive after grace period — force kill
    platform::send_signal(pid, libc::SIGKILL);
    std::thread::sleep(Duration::from_millis(50));
}

/// Scan `~/.nanobot/pids/*.pid`, kill any still-alive processes, remove all
/// stale PID files. Called at startup before spawning new servers.
pub fn cleanup_stale_pids() {
    let dir = pids_dir();
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return, // directory doesn't exist yet — nothing to clean
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("pid") {
            continue;
        }
        let contents = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => {
                let _ = std::fs::remove_file(&path);
                continue;
            }
        };
        if let Ok(pid) = contents.trim().parse::<u32>() {
            if platform::is_process_alive(pid) {
                tracing::info!(pid, file = %path.display(), "killing stale child process");
                graceful_kill(pid);
            }
        }
        let _ = std::fs::remove_file(&path);
    }
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
}

// ---------------------------------------------------------------------------
// Main agent singleton guard
// ---------------------------------------------------------------------------

/// Path to the main agent PID file: `~/.nanobot/agent.pid`.
/// Kept outside `pids/` so `cleanup_stale_pids()` (which kills child servers)
/// does not accidentally kill the running agent itself.
fn agent_pid_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_default()
        .join(".nanobot")
        .join("agent.pid")
}

/// If a previous agent process is still alive, kill it gracefully before we
/// take over. Then write our own PID so the *next* launch can do the same.
/// Call this early in `cmd_agent` / `run_gateway_async`.
pub fn acquire_agent_singleton() {
    let path = agent_pid_path();
    if let Ok(contents) = std::fs::read_to_string(&path) {
        if let Ok(old_pid) = contents.trim().parse::<u32>() {
            if platform::is_process_alive(old_pid) && old_pid != std::process::id() {
                tracing::warn!(old_pid, "killing stale agent process (singleton guard)");
                graceful_kill(old_pid);
            }
        }
    }
    let _ = std::fs::create_dir_all(path.parent().unwrap_or(&PathBuf::from(".")));
    let _ = std::fs::write(&path, std::process::id().to_string());
    tracing::debug!(pid = std::process::id(), "agent singleton acquired");
}

/// Remove the agent PID file on clean shutdown.
pub fn release_agent_singleton() {
    let path = agent_pid_path();
    // Only remove if the file contains our own PID (another instance may have
    // already overwritten it).
    if let Ok(contents) = std::fs::read_to_string(&path) {
        if let Ok(pid) = contents.trim().parse::<u32>() {
            if pid == std::process::id() {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static AGENT_SINGLETON_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_agent_singleton_acquire_release() {
        let _guard = AGENT_SINGLETON_TEST_LOCK.lock().unwrap();
        let _ = std::fs::remove_file(agent_pid_path());

        // acquire writes our PID, release removes it.
        acquire_agent_singleton();
        let contents = std::fs::read_to_string(agent_pid_path()).unwrap();
        assert_eq!(contents.trim().parse::<u32>().unwrap(), std::process::id());

        release_agent_singleton();
        assert!(!agent_pid_path().exists());
    }

    #[test]
    fn test_agent_singleton_stale_pid_cleaned() {
        let _guard = AGENT_SINGLETON_TEST_LOCK.lock().unwrap();
        let _ = std::fs::remove_file(agent_pid_path());

        // Write a dead PID, acquire should overwrite it with ours.
        let _ = std::fs::write(agent_pid_path(), "4000000");
        acquire_agent_singleton();
        let contents = std::fs::read_to_string(agent_pid_path()).unwrap();
        assert_eq!(contents.trim().parse::<u32>().unwrap(), std::process::id());
        release_agent_singleton();
    }

    #[test]
    fn test_pid_file_roundtrip_and_stale_cleanup() {
        // Use a unique port unlikely to collide with parallel tests.
        let name = "test-roundtrip";
        let port = 59997;
        let dir = pids_dir();
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join(format!("{name}-{port}.pid"));

        // Write a PID for a process that doesn't exist, then verify cleanup
        // removes the stale file through the real startup path.
        std::fs::write(&path, "4000000").unwrap();
        cleanup_stale_pids();
        assert!(!path.exists());
    }
}
