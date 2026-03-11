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

/// Write a PID file for a managed child process.
pub fn write_pid(name: &str, port: u16, pid: u32) {
    let dir = pids_dir();
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join(format!("{name}-{port}.pid"));
    let _ = std::fs::write(&path, pid.to_string());
    tracing::debug!(pid, path = %path.display(), "wrote PID file");
}

/// Read a PID from a PID file, returning `None` if missing or malformed.
pub fn read_pid(name: &str, port: u16) -> Option<u32> {
    let path = pids_dir().join(format!("{name}-{port}.pid"));
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

/// Remove a PID file (ignores errors if file doesn't exist).
pub fn remove_pid(name: &str, port: u16) {
    let path = pids_dir().join(format!("{name}-{port}.pid"));
    let _ = std::fs::remove_file(&path);
    tracing::debug!(path = %path.display(), "removed PID file");
}

/// Send SIGTERM, wait up to 2s, then SIGKILL if still alive.
fn graceful_kill(pid: u32) {
    let pid_i32 = pid as i32;
    unsafe {
        // Check if process is alive
        if libc::kill(pid_i32, 0) != 0 {
            return; // already dead
        }
        libc::kill(pid_i32, libc::SIGTERM);
    }
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(100));
        unsafe {
            if libc::kill(pid_i32, 0) != 0 {
                return; // died from SIGTERM
            }
        }
    }
    // Still alive after grace period — force kill
    unsafe {
        libc::kill(pid_i32, libc::SIGKILL);
    }
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
            let alive = unsafe { libc::kill(pid as i32, 0) == 0 };
            if alive {
                tracing::info!(pid, file = %path.display(), "killing stale child process");
                graceful_kill(pid);
            }
        }
        let _ = std::fs::remove_file(&path);
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
            let alive = unsafe { libc::kill(old_pid as i32, 0) == 0 };
            if alive && old_pid != std::process::id() {
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

    #[test]
    fn test_agent_singleton_acquire_release() {
        // acquire writes our PID, release removes it.
        acquire_agent_singleton();
        let contents = std::fs::read_to_string(agent_pid_path()).unwrap();
        assert_eq!(contents.trim().parse::<u32>().unwrap(), std::process::id());

        release_agent_singleton();
        assert!(!agent_pid_path().exists());
    }

    #[test]
    fn test_agent_singleton_stale_pid_cleaned() {
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

        // 1) roundtrip: write → read → remove → read
        write_pid(name, port, 12345);
        assert_eq!(read_pid(name, port), Some(12345));
        remove_pid(name, port);
        assert_eq!(read_pid(name, port), None);

        // 2) stale cleanup: write a PID for a process that doesn't exist,
        //    verify cleanup removes the file.
        write_pid(name, port, 4_000_000);
        cleanup_stale_pids();
        assert_eq!(read_pid(name, port), None);
    }
}
