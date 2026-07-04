//! Auto-manage a local crw-server (fastCRW) for web fetching.
//!
//! fastCRW is a single static binary — no Docker, no sidecars — serving
//! `/v1/scrape` on port 3000. `ensure_crw()` runs once at startup: if the
//! server is not reachable and the binary is installed, spawn it detached
//! and wait for `/health`. Non-fatal on every path: web_fetch falls back
//! to the plain HTTP fetcher when crw is unavailable.

use std::path::PathBuf;
use std::time::Duration;
use tokio::time::timeout;
use tracing::{debug, info};

const HEALTH_TIMEOUT: Duration = Duration::from_secs(2);
const READY_POLL_INTERVAL: Duration = Duration::from_millis(500);
const READY_POLL_MAX: Duration = Duration::from_secs(10);

/// Ensure a crw-server is reachable at `url`. Spawns the installed binary
/// when necessary. Err is informational — the caller logs and moves on.
pub async fn ensure_crw(url: &str) -> Result<(), String> {
    if health_check(url).await {
        debug!("crw-server already reachable at {url}");
        return Ok(());
    }

    let Some(bin) = find_crw_binary() else {
        return Err(
            "crw-server not installed — web_fetch uses the plain fetcher. \
             Install: curl -fsSL https://fastcrw.com/install | CRW_BINARY=crw-server sh"
                .to_string(),
        );
    };

    info!("Starting crw-server from {}", bin.display());
    std::process::Command::new(&bin)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("failed to spawn {}: {e}", bin.display()))?;

    let start = std::time::Instant::now();
    while start.elapsed() < READY_POLL_MAX {
        if health_check(url).await {
            info!("crw-server ready at {url}");
            return Ok(());
        }
        tokio::time::sleep(READY_POLL_INTERVAL).await;
    }
    Err(format!(
        "crw-server did not become ready within {}s",
        READY_POLL_MAX.as_secs()
    ))
}

/// `/health` returns `{"status":"ok",...}` when the server is up.
async fn health_check(url: &str) -> bool {
    let client = match reqwest::Client::builder().timeout(HEALTH_TIMEOUT).build() {
        Ok(c) => c,
        Err(_) => return false,
    };
    let resp = timeout(HEALTH_TIMEOUT, client.get(format!("{url}/health")).send()).await;
    match resp {
        Ok(Ok(r)) if r.status().is_success() => {
            parse_health(&r.text().await.unwrap_or_default())
        }
        _ => false,
    }
}

/// True when the health body reports `"status":"ok"`.
fn parse_health(body: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("status").and_then(|s| s.as_str()).map(|s| s == "ok"))
        .unwrap_or(false)
}

/// Locate the crw-server binary: $PATH first, then the common install dirs
/// the official installer and package managers use.
fn find_crw_binary() -> Option<PathBuf> {
    if let Ok(output) = std::process::Command::new("which").arg("crw-server").output() {
        if output.status.success() {
            let p = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !p.is_empty() {
                return Some(PathBuf::from(p));
            }
        }
    }
    candidate_paths().into_iter().find(|p| p.is_file())
}

/// Install locations checked when the binary is not on $PATH.
fn candidate_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(home) = dirs::home_dir() {
        paths.push(home.join(".local/bin/crw-server"));
        paths.push(home.join(".cargo/bin/crw-server"));
    }
    paths.push(PathBuf::from("/opt/homebrew/bin/crw-server"));
    paths.push(PathBuf::from("/usr/local/bin/crw-server"));
    paths
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_health_accepts_ok_status_only() {
        assert!(parse_health(r#"{"active_crawl_jobs":0,"status":"ok","version":"0.20.0"}"#));
        assert!(!parse_health(r#"{"status":"degraded"}"#));
        assert!(!parse_health("not json"));
        assert!(!parse_health(""));
    }

    #[test]
    fn candidate_paths_cover_installer_and_brew_locations() {
        let paths: Vec<String> = candidate_paths()
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();
        assert!(paths.iter().any(|p| p.ends_with(".local/bin/crw-server")));
        assert!(paths.iter().any(|p| p == "/opt/homebrew/bin/crw-server"));
        assert!(paths.iter().any(|p| p == "/usr/local/bin/crw-server"));
    }

    #[tokio::test]
    async fn health_check_unreachable_is_false() {
        assert!(!health_check("http://127.0.0.1:19998").await);
    }
}
