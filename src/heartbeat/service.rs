//! Heartbeat service -- periodic agent wake-up to check for tasks.
//!
//! The agent reads `HEARTBEAT.md` from the workspace and executes any tasks
//! listed there. If nothing needs attention it replies `HEARTBEAT_OK`.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::time::Duration;
use tracing::{debug, error, info};

/// Default heartbeat interval: 30 minutes.
pub const DEFAULT_HEARTBEAT_INTERVAL_S: u64 = 30 * 60;

/// The prompt sent to the agent during a heartbeat.
pub const HEARTBEAT_PROMPT: &str = "Read HEARTBEAT.md in your workspace (if it exists).\n\
    Follow any instructions or tasks listed there.\n\
    If nothing needs attention, reply with just: HEARTBEAT_OK";

/// Token that indicates "nothing to do".
pub const HEARTBEAT_OK_TOKEN: &str = "HEARTBEAT_OK";

// ---------------------------------------------------------------------------
// Callback type
// ---------------------------------------------------------------------------

/// Async callback invoked on each heartbeat.
/// Receives the heartbeat prompt and returns the agent response.
pub type HeartbeatCallback =
    Arc<dyn Fn(String) -> Pin<Box<dyn Future<Output = Option<String>> + Send>> + Send + Sync>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check whether a `HEARTBEAT.md` file has no actionable content.
///
/// Lines that are empty, headers (`#`), HTML comments (`<!--`), or checkbox
/// items (`- [ ]`, `- [x]`, `* [ ]`, `* [x]`) are skipped.  If only such
/// lines are present the file is considered empty.
fn is_heartbeat_empty(content: Option<&str>) -> bool {
    let content = match content {
        Some(c) => c,
        None => return true,
    };

    let skip_exact: &[&str] = &["- [ ]", "* [ ]", "- [x]", "* [x]"];

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('#') {
            continue;
        }
        if trimmed.starts_with("<!--") {
            continue;
        }
        if skip_exact.contains(&trimmed) {
            continue;
        }
        // Found an actionable line.
        return false;
    }

    true
}

// ---------------------------------------------------------------------------
// HeartbeatService
// ---------------------------------------------------------------------------

/// Periodic heartbeat service that wakes the agent to check for tasks.
pub struct HeartbeatService {
    /// Root workspace directory (contains `HEARTBEAT.md`).
    pub workspace: PathBuf,
    /// Callback invoked on each heartbeat tick.
    on_heartbeat: Option<HeartbeatCallback>,
    /// Interval between heartbeats in seconds.
    pub interval_s: u64,
    /// Whether the service is enabled at all.
    pub enabled: bool,
    /// Shared flag used to stop the background loop.
    running: Arc<AtomicBool>,
    /// Handle to the spawned background task.
    task_handle: tokio::sync::Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl HeartbeatService {
    /// Create a new `HeartbeatService`.
    pub fn new(
        workspace: PathBuf,
        on_heartbeat: Option<HeartbeatCallback>,
        interval_s: u64,
        enabled: bool,
    ) -> Self {
        Self {
            workspace,
            on_heartbeat,
            interval_s,
            enabled,
            running: Arc::new(AtomicBool::new(false)),
            task_handle: tokio::sync::Mutex::new(None),
        }
    }

    /// Path to `HEARTBEAT.md` inside the workspace.
    pub fn heartbeat_file(&self) -> PathBuf {
        self.workspace.join("HEARTBEAT.md")
    }

    /// Read the contents of `HEARTBEAT.md`, if it exists.
    fn read_heartbeat_file(&self) -> Option<String> {
        let path = self.heartbeat_file();
        if path.exists() {
            std::fs::read_to_string(&path).ok()
        } else {
            None
        }
    }

    /// Start the heartbeat background loop.
    pub async fn start(&self) {
        if !self.enabled {
            info!("Heartbeat disabled");
            return;
        }

        self.running.store(true, Ordering::Relaxed);
        info!("Heartbeat started (every {}s)", self.interval_s);

        let running = Arc::clone(&self.running);
        let interval_s = self.interval_s;
        let workspace = self.workspace.clone();
        let on_heartbeat = self.on_heartbeat.clone();

        let handle = tokio::spawn(async move {
            Self::run_loop(running, interval_s, workspace, on_heartbeat).await;
        });

        let mut guard = self.task_handle.lock().await;
        *guard = Some(handle);
    }

    /// Stop the heartbeat service and cancel the background task.
    pub async fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
        let mut guard = self.task_handle.lock().await;
        if let Some(h) = guard.take() {
            h.abort();
        }
    }

    /// Manually trigger a heartbeat right now, bypassing the interval.
    ///
    /// Returns the agent response (if any).
    pub async fn trigger_now(&self) -> Option<String> {
        if let Some(ref cb) = self.on_heartbeat {
            let result = cb(HEARTBEAT_PROMPT.to_string()).await;
            return result;
        }
        None
    }

    // -----------------------------------------------------------------------
    // Internal loop
    // -----------------------------------------------------------------------

    async fn run_loop(
        running: Arc<AtomicBool>,
        interval_s: u64,
        workspace: PathBuf,
        on_heartbeat: Option<HeartbeatCallback>,
    ) {
        loop {
            tokio::time::sleep(Duration::from_secs(interval_s)).await;

            if !running.load(Ordering::Relaxed) {
                break;
            }

            // Read HEARTBEAT.md
            let content = {
                let path = workspace.join("HEARTBEAT.md");
                if path.exists() {
                    std::fs::read_to_string(&path).ok()
                } else {
                    None
                }
            };

            if is_heartbeat_empty(content.as_deref()) {
                debug!("Heartbeat: no tasks (HEARTBEAT.md empty)");
                continue;
            }

            info!("Heartbeat: checking for tasks...");

            if let Some(ref cb) = on_heartbeat {
                match cb(HEARTBEAT_PROMPT.to_string()).await {
                    Some(response) => {
                        // Normalize both sides for comparison (strip underscores, uppercase).
                        let normalized = response.to_uppercase().replace('_', "");
                        let token_normalized = HEARTBEAT_OK_TOKEN.replace('_', "");
                        if normalized.contains(&token_normalized) {
                            info!("Heartbeat: OK (no action needed)");
                        } else {
                            info!("Heartbeat: completed task");
                        }
                    }
                    None => {
                        info!("Heartbeat: callback returned no response");
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_heartbeat_empty_none() {
        assert!(is_heartbeat_empty(None));
    }

    #[test]
    fn test_is_heartbeat_empty_blank() {
        assert!(is_heartbeat_empty(Some("")));
        assert!(is_heartbeat_empty(Some("  \n\n  ")));
    }

    #[test]
    fn test_is_heartbeat_empty_headers_only() {
        assert!(is_heartbeat_empty(Some("# Tasks\n## Sub\n")));
    }

    #[test]
    fn test_is_heartbeat_empty_comments_only() {
        assert!(is_heartbeat_empty(Some("<!-- nothing -->\n")));
    }

    #[test]
    fn test_is_heartbeat_empty_checkboxes_only() {
        assert!(is_heartbeat_empty(Some("- [ ]\n* [x]\n")));
    }

    #[test]
    fn test_is_heartbeat_not_empty() {
        assert!(!is_heartbeat_empty(Some("Do the thing\n")));
        assert!(!is_heartbeat_empty(Some("# Tasks\n- Buy milk\n")));
    }
}
