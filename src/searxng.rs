//! Auto-manage SearXNG Docker container for web search.
//!
//! `ensure_searxng()` runs once at startup: checks if SearXNG is reachable,
//! and if not, attempts to start or create a Docker container. All Docker
//! commands use aggressive timeouts to avoid blocking when Docker Desktop
//! is stuck or unresponsive.

use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;
use tracing::{debug, info};

const CONTAINER_NAME: &str = "nanobot-searxng";
const DOCKER_CMD_TIMEOUT: Duration = Duration::from_secs(5);
const DOCKER_RUN_TIMEOUT: Duration = Duration::from_secs(30);
const HEALTH_CHECK_TIMEOUT: Duration = Duration::from_secs(2);
const READY_POLL_INTERVAL: Duration = Duration::from_secs(1);
const READY_POLL_MAX: Duration = Duration::from_secs(20);
const DOCKER_DESKTOP_STARTUP_TIMEOUT: Duration = Duration::from_secs(45);
const DOCKER_DESKTOP_POLL_INTERVAL: Duration = Duration::from_secs(2);

/// Outcome of probing the JSON search endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearxHealth {
    /// JSON search works — nothing to do.
    Ready,
    /// The server answered but refused the API call (403/429: bot detection,
    /// or 403 because the `json` format is not enabled). Fixable by rewriting
    /// the container's settings.yml.
    Blocked,
    /// No HTTP answer at all (wrong port, container down, still starting).
    Unreachable,
}

/// Ensure SearXNG is running and its JSON API is usable. Non-fatal: returns
/// Err with a human-readable message on failure (caller logs warning, search
/// falls back).
///
/// Self-healing by reconciliation, not only on creation: a container that
/// exists but rejects API calls (created by docker compose, or by an older
/// nanobot without the JSON/limiter config) is reconfigured in place. The
/// previous design only configured containers it created itself, so a
/// misconfigured pre-existing container 403'd forever.
pub async fn ensure_searxng(searxng_url: &str) -> Result<(), String> {
    // 1. Quick probe — already usable?
    if probe(searxng_url).await == SearxHealth::Ready {
        debug!("SearXNG already reachable at {searxng_url}");
        return Ok(());
    }

    // 2. Docker binary + daemon
    let docker = find_docker().await?;
    check_docker_daemon(&docker).await?;

    // 3. Make sure the container exists and is running
    match container_status(&docker).await.as_deref() {
        Some("running") => {}
        // Crash loop — usually a broken settings.yml (e.g. the image's
        // placeholder secret_key, which SearXNG refuses to boot with).
        // Rewrite our known-good config via `docker cp` (works in any
        // container state, unlike `docker exec`) and let it come back.
        Some("restarting") => configure_for_local_use(&docker).await,
        Some("exited") | Some("created") => start_container(&docker).await?,
        _ => create_container(&docker, searxng_url).await?,
    }

    // 4. Port reconciliation: the configured URL must actually point at the
    //    container, or every later step waits on the wrong port.
    if let (Some(host_port), Some(cfg_port)) = (
        container_host_port(&docker).await,
        url_port(searxng_url),
    ) {
        if host_port != cfg_port {
            return Err(format!(
                "SearXNG container '{CONTAINER_NAME}' publishes port {host_port}, but \
                 config points at {searxng_url}. Set tools.web.search.searxngUrl to \
                 \"http://localhost:{host_port}\" in ~/.nanobot/config.json (or remove \
                 the container and restart nanobot to recreate it on {cfg_port})."
            ));
        }
    }

    // 5. Wait until the server answers, then heal a blocked API in place.
    match wait_for_http(searxng_url).await {
        SearxHealth::Ready => Ok(()),
        SearxHealth::Blocked => {
            info!("SearXNG is up but rejects API calls — rewriting container config...");
            configure_for_local_use(&docker).await;
            match wait_for_http(searxng_url).await {
                SearxHealth::Ready => {
                    info!("SearXNG healed and ready at {searxng_url}");
                    Ok(())
                }
                other => Err(format!(
                    "SearXNG still not usable after reconfiguration ({other:?}) — \
                     check `docker logs {CONTAINER_NAME}`"
                )),
            }
        }
        SearxHealth::Unreachable => Err(format!(
            "SearXNG did not become reachable within {}s",
            READY_POLL_MAX.as_secs()
        )),
    }
}

/// GET the JSON search endpoint and classify the result.
async fn probe(searxng_url: &str) -> SearxHealth {
    let url = format!("{searxng_url}/search?q=test&format=json");
    let client = match reqwest::Client::builder()
        .timeout(HEALTH_CHECK_TIMEOUT)
        .build()
    {
        Ok(c) => c,
        Err(_) => return SearxHealth::Unreachable,
    };
    match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => SearxHealth::Ready,
        Ok(_) => SearxHealth::Blocked,
        Err(_) => SearxHealth::Unreachable,
    }
}

/// Extract the port from a base URL like `http://localhost:8888/`.
fn url_port(url: &str) -> Option<u16> {
    url.rsplit(':')
        .next()
        .and_then(|p| p.trim_end_matches('/').parse::<u16>().ok())
}

/// Locate the `docker` binary via `which`.
async fn find_docker() -> Result<String, String> {
    let result = timeout(Duration::from_secs(3), async {
        Command::new("which")
            .arg("docker")
            .kill_on_drop(true)
            .output()
            .await
    })
    .await;

    match result {
        Ok(Ok(output)) if output.status.success() => {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if path.is_empty() {
                Err("docker not found on PATH".to_string())
            } else {
                Ok(path)
            }
        }
        Ok(Ok(_)) => Err("docker not found on PATH".to_string()),
        Ok(Err(e)) => Err(format!("failed to run `which docker`: {e}")),
        Err(_) => Err("timed out searching for docker binary".to_string()),
    }
}

/// Check if Docker daemon is responsive with `docker info`.
async fn docker_daemon_ready(docker: &str) -> bool {
    let result = timeout(DOCKER_CMD_TIMEOUT, async {
        Command::new(docker)
            .arg("info")
            .kill_on_drop(true)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await
    })
    .await;

    matches!(result, Ok(Ok(status)) if status.success())
}

/// Ensure the Docker daemon is running. If not, attempt to start Docker Desktop
/// (macOS) or the docker service (Linux), then poll until it's ready.
async fn check_docker_daemon(docker: &str) -> Result<(), String> {
    if docker_daemon_ready(docker).await {
        return Ok(());
    }

    // Docker daemon not running — try to start it
    info!("Docker daemon not running, attempting to start...");

    if cfg!(target_os = "macos") {
        // Force-kill all Docker Desktop processes (handles zombie/stuck state
        // where the backend is alive but the VM/daemon isn't running).
        for proc in [
            "com.docker.backend",
            "com.docker.vmnetd",
            "Docker Desktop",
            "Docker",
        ] {
            let _ = Command::new("killall")
                .args(["-9", proc])
                .kill_on_drop(true)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .await;
        }
        tokio::time::sleep(Duration::from_secs(2)).await;

        // macOS: open Docker Desktop app fresh
        let _ = Command::new("open")
            .args(["-a", "Docker"])
            .kill_on_drop(true)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await;
        info!("Launched Docker Desktop, waiting for daemon...");
    } else {
        // Linux: try systemctl, then dockerd
        let systemctl = Command::new("systemctl")
            .args(["start", "docker"])
            .kill_on_drop(true)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await;

        if !matches!(systemctl, Ok(s) if s.success()) {
            // Fallback: try starting dockerd directly (rootless or manual setups)
            let _ = Command::new("dockerd")
                .kill_on_drop(true)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn(); // fire-and-forget
        }
        info!("Starting Docker daemon, waiting...");
    }

    // Poll until daemon is ready
    let start = std::time::Instant::now();
    while start.elapsed() < DOCKER_DESKTOP_STARTUP_TIMEOUT {
        if docker_daemon_ready(docker).await {
            info!("Docker daemon is ready");
            return Ok(());
        }
        tokio::time::sleep(DOCKER_DESKTOP_POLL_INTERVAL).await;
    }

    Err(format!(
        "Docker daemon did not start within {}s — check Docker Desktop",
        DOCKER_DESKTOP_STARTUP_TIMEOUT.as_secs()
    ))
}

/// Query container status via `docker inspect`.
async fn container_status(docker: &str) -> Option<String> {
    let result = timeout(DOCKER_CMD_TIMEOUT, async {
        Command::new(docker)
            .args(["inspect", CONTAINER_NAME, "--format", "{{.State.Status}}"])
            .kill_on_drop(true)
            .output()
            .await
    })
    .await;

    match result {
        Ok(Ok(output)) if output.status.success() => {
            let status = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if status.is_empty() {
                None
            } else {
                Some(status)
            }
        }
        _ => None,
    }
}

/// Host port the container publishes for its internal 8080/tcp endpoint.
async fn container_host_port(docker: &str) -> Option<u16> {
    let result = timeout(DOCKER_CMD_TIMEOUT, async {
        Command::new(docker)
            .args([
                "inspect",
                CONTAINER_NAME,
                "--format",
                r#"{{(index (index .NetworkSettings.Ports "8080/tcp") 0).HostPort}}"#,
            ])
            .kill_on_drop(true)
            .output()
            .await
    })
    .await;

    match result {
        Ok(Ok(output)) if output.status.success() => String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse::<u16>()
            .ok(),
        _ => None,
    }
}

/// Start an existing stopped container.
async fn start_container(docker: &str) -> Result<(), String> {
    info!("Starting existing SearXNG container...");
    let result = timeout(DOCKER_CMD_TIMEOUT, async {
        Command::new(docker)
            .args(["start", CONTAINER_NAME])
            .kill_on_drop(true)
            .output()
            .await
    })
    .await;

    match result {
        Ok(Ok(output)) if output.status.success() => Ok(()),
        Ok(Ok(output)) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("failed to start SearXNG container: {stderr}"))
        }
        Ok(Err(e)) => Err(format!("failed to run `docker start`: {e}")),
        Err(_) => Err("timed out starting SearXNG container".to_string()),
    }
}

/// Create and start a new SearXNG container with `docker run`.
async fn create_container(docker: &str, searxng_url: &str) -> Result<(), String> {
    info!("Creating SearXNG container (may pull image on first run)...");

    // Extract port from URL, default to 8888
    let port = url_port(searxng_url).unwrap_or(8888);

    let port_mapping = format!("{port}:8080");
    let base_url = format!("SEARXNG_BASE_URL=http://localhost:{port}");

    let result = timeout(DOCKER_RUN_TIMEOUT, async {
        Command::new(docker)
            .args([
                "run",
                "-d",
                "--name",
                CONTAINER_NAME,
                "-p",
                &port_mapping,
                "-e",
                &base_url,
                "--restart",
                "unless-stopped",
                "searxng/searxng:latest",
            ])
            .kill_on_drop(true)
            .output()
            .await
    })
    .await;

    match result {
        Ok(Ok(output)) if output.status.success() => {
            info!("SearXNG container created successfully");
            Ok(())
        }
        Ok(Ok(output)) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("failed to create SearXNG container: {stderr}"))
        }
        Ok(Err(e)) => Err(format!("failed to run `docker run`: {e}")),
        Err(_) => Err("timed out creating SearXNG container (image pull may be slow)".to_string()),
    }
}

/// The docker image ships this placeholder; SearXNG refuses to boot with it.
const PLACEHOLDER_SECRET: &str = "ultrasecretkey";

/// Pull the current secret_key out of a settings.yml, rejecting the image's
/// placeholder (propagating it crash-loops the container).
fn extract_secret_key(yaml: &str) -> Option<String> {
    let key = yaml
        .lines()
        .find_map(|l| l.trim_start().strip_prefix("secret_key:"))
        .map(|v| v.trim().trim_matches('"').to_string())?;
    if key.is_empty() || key == PLACEHOLDER_SECRET {
        None
    } else {
        Some(key)
    }
}

/// Minimal settings.yml for local API use. `use_default_settings: true`
/// keeps everything else on upstream defaults.
fn render_local_settings(secret_key: &str) -> String {
    format!(
        "# Managed by nanobot (rewritten when the JSON API is blocked).\n\
         use_default_settings: true\n\
         server:\n\
         \x20 secret_key: \"{secret_key}\"\n\
         \x20 limiter: true\n\
         \x20 public_instance: false\n\
         redis:\n\
         \x20 url: valkey://valkey:6379/0\n\
         search:\n\
         \x20 formats:\n\
         \x20   - html\n\
         \x20   - json\n\
         \x20 ban_time_on_fail: 5\n\
         \x20 max_ban_time_on_fail: 120\n\
         \x20 suspended_times:\n\
         \x20   SearxEngineAccessDenied: 600\n\
         \x20   SearxEngineCaptcha: 1800\n\
         \x20   SearxEngineTooManyRequests: 600\n"
    )
}

/// Rewrite the container's settings.yml for local API use, then restart it.
///
/// Runs whenever the API is blocked or the container crash-loops — regardless
/// of who created it (nanobot, docker compose, an older nanobot). The config:
/// - disables the limiter entirely (bot detection 403s local API calls that
///   lack browser-like headers — pointless protection on localhost now that
///   WebSearchTool sends a real User-Agent + Sec-Fetch/Accept headers)
/// - enables the `json` output format (off by default → 403 on format=json)
/// - shortens engine suspension times so one upstream 403/CAPTCHA doesn't
///   bench an engine for up to a day; with several engines active, a single
///   ban never zeroes the results
///
/// Uses `docker cp` (not `exec`) so it also works on a container that is
/// stopped or stuck restarting. An existing non-placeholder secret_key is
/// preserved; otherwise a fresh one is generated.
async fn configure_for_local_use(docker: &str) {
    debug!("Configuring SearXNG for local API use...");
    let tmp = std::env::temp_dir().join("nanobot-searxng-settings.yml");

    // Read the existing file (best-effort) to preserve a real secret_key.
    let _ = timeout(DOCKER_CMD_TIMEOUT, async {
        Command::new(docker)
            .args([
                "cp",
                &format!("{CONTAINER_NAME}:/etc/searxng/settings.yml"),
                &tmp.to_string_lossy(),
            ])
            .kill_on_drop(true)
            .output()
            .await
    })
    .await;
    let old = std::fs::read_to_string(&tmp).unwrap_or_default();
    let key = extract_secret_key(&old)
        .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());

    if std::fs::write(&tmp, render_local_settings(&key)).is_err() {
        debug!("could not write temp settings.yml");
        return;
    }
    let copied = timeout(DOCKER_CMD_TIMEOUT, async {
        Command::new(docker)
            .args([
                "cp",
                &tmp.to_string_lossy(),
                &format!("{CONTAINER_NAME}:/etc/searxng/settings.yml"),
            ])
            .kill_on_drop(true)
            .output()
            .await
    })
    .await;
    if !matches!(&copied, Ok(Ok(o)) if o.status.success()) {
        debug!("docker cp of settings.yml into container failed");
    }

    // Restart to pick up config changes
    let _ = timeout(DOCKER_RUN_TIMEOUT, async {
        Command::new(docker)
            .args(["restart", CONTAINER_NAME])
            .kill_on_drop(true)
            .output()
            .await
    })
    .await;
}

/// Poll until SearXNG answers HTTP (Ready or Blocked) or the timeout lapses.
/// A `Blocked` answer returns immediately — the server is up, only its config
/// needs fixing, and waiting longer would not change that.
async fn wait_for_http(searxng_url: &str) -> SearxHealth {
    let start = std::time::Instant::now();
    while start.elapsed() < READY_POLL_MAX {
        match probe(searxng_url).await {
            SearxHealth::Unreachable => tokio::time::sleep(READY_POLL_INTERVAL).await,
            reachable => return reachable,
        }
    }
    SearxHealth::Unreachable
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_secret_key_rejects_placeholder() {
        // The image placeholder must never be propagated — SearXNG refuses
        // to boot with it and the container crash-loops.
        assert_eq!(
            extract_secret_key("server:\n  secret_key: \"ultrasecretkey\"\n"),
            None
        );
        assert_eq!(extract_secret_key("server:\n  secret_key: \"\"\n"), None);
        assert_eq!(extract_secret_key("no key here"), None);
        assert_eq!(
            extract_secret_key("server:\n  secret_key: \"abc123\"\n"),
            Some("abc123".to_string())
        );
    }

    #[test]
    fn test_render_local_settings_enables_json_api() {
        let s = render_local_settings("k");
        assert!(s.contains("use_default_settings: true"));
        assert!(s.contains("- json"), "json format must be enabled");
        assert!(s.contains("secret_key: \"k\""));
        assert!(
            s.contains("suspended_times"),
            "fast engine-ban recovery must be configured"
        );
    }

    #[test]
    fn test_url_port_extraction() {
        assert_eq!(url_port("http://localhost:8888"), Some(8888));
        assert_eq!(url_port("http://localhost:9999/"), Some(9999));
        assert_eq!(url_port("http://localhost"), None);
    }

    #[tokio::test]
    #[ignore] // requires Docker and the local nanobot-searxng container
    async fn live_ensure_searxng_heals_container() {
        ensure_searxng("http://localhost:8080").await.unwrap();
        assert_eq!(probe("http://localhost:8080").await, SearxHealth::Ready);
    }

    #[tokio::test]
    async fn test_probe_unreachable() {
        // A port nothing listens on must classify as Unreachable, not Blocked.
        assert_eq!(
            probe("http://127.0.0.1:19999").await,
            SearxHealth::Unreachable
        );
    }
}
