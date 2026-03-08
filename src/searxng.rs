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

/// Ensure SearXNG is running and reachable. Non-fatal: returns Err with a
/// human-readable message on failure (caller logs warning, search falls back).
pub async fn ensure_searxng(searxng_url: &str) -> Result<(), String> {
    // 1. Quick health check — already running?
    if health_check(searxng_url).await {
        debug!("SearXNG already reachable at {searxng_url}");
        return Ok(());
    }

    // 2. Find docker binary
    let docker = find_docker().await?;

    // 3. Check Docker daemon is responsive
    check_docker_daemon(&docker).await?;

    // 4. Check container state and act accordingly
    let created_fresh = match container_status(&docker).await {
        Some(status) if status == "running" => {
            // Container running but health check failed — maybe still starting up
            info!("SearXNG container is running but not yet responding, waiting...");
            false
        }
        Some(status) if status == "exited" || status == "created" => {
            start_container(&docker).await?;
            false
        }
        _ => {
            // Not found — create new container
            create_container(&docker, searxng_url).await?;
            true
        }
    };

    // 5. Configure JSON format on fresh containers
    if created_fresh {
        configure_for_local_use(&docker).await;
    }

    // 6. Wait for ready
    wait_for_ready(searxng_url).await
}

/// GET the search endpoint to verify SearXNG is alive.
async fn health_check(searxng_url: &str) -> bool {
    let url = format!("{searxng_url}/search?q=test&format=json");
    let client = match reqwest::Client::builder()
        .timeout(HEALTH_CHECK_TIMEOUT)
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    match client.get(&url).send().await {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
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
    let port = searxng_url
        .rsplit(':')
        .next()
        .and_then(|p| p.trim_end_matches('/').parse::<u16>().ok())
        .unwrap_or(8888);

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

/// Configure SearXNG for local API use (fresh container only).
///
/// 1. Enable JSON output format in settings.yml
/// 2. Create a permissive limiter.toml so bot detection doesn't block local requests
async fn configure_for_local_use(docker: &str) {
    debug!("Configuring SearXNG for local API use...");

    // Python script to fix settings.yml (enable json format) and create
    // a permissive limiter.toml (disable bot detection for local use).
    let script = r#"
import re, pathlib

# 1. Enable json format in settings.yml
p = pathlib.Path('/etc/searxng/settings.yml')
t = p.read_text()
# Match the indented `  formats:` block and its list items
t = re.sub(
    r'(?m)^(  formats:\s*\n(?:    - \w+\n)*)',
    '  formats:\n    - html\n    - json\n',
    t,
)
p.write_text(t)

# 2. Create permissive limiter.toml — SearXNG bot detection rejects
#    requests without X-Forwarded-For headers, which blocks local API calls.
limiter = pathlib.Path('/etc/searxng/limiter.toml')
limiter.write_text("""
[botdetection.ip_limit]
link_token = false

[botdetection.ip_lists]
pass_ip = [
  '0.0.0.0/0',
  '::/0',
]
""")
"#;

    let result = timeout(DOCKER_CMD_TIMEOUT, async {
        Command::new(docker)
            .args(["exec", CONTAINER_NAME, "python3", "-c", script])
            .kill_on_drop(true)
            .output()
            .await
    })
    .await;

    if let Ok(Ok(output)) = &result {
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            debug!("SearXNG config script failed: {stderr}");
        }
    }

    // Restart to pick up config changes
    let _ = timeout(DOCKER_CMD_TIMEOUT, async {
        Command::new(docker)
            .args(["restart", CONTAINER_NAME])
            .kill_on_drop(true)
            .output()
            .await
    })
    .await;
}

/// Poll health check until SearXNG responds or we time out.
async fn wait_for_ready(searxng_url: &str) -> Result<(), String> {
    let start = std::time::Instant::now();
    while start.elapsed() < READY_POLL_MAX {
        if health_check(searxng_url).await {
            info!("SearXNG is ready at {searxng_url}");
            return Ok(());
        }
        tokio::time::sleep(READY_POLL_INTERVAL).await;
    }
    Err(format!(
        "SearXNG did not become ready within {}s",
        READY_POLL_MAX.as_secs()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_port_extraction_from_url() {
        let url = "http://localhost:8888";
        let port = url
            .rsplit(':')
            .next()
            .and_then(|p| p.trim_end_matches('/').parse::<u16>().ok())
            .unwrap_or(8888);
        assert_eq!(port, 8888);

        let url2 = "http://localhost:9999/";
        let port2 = url2
            .rsplit(':')
            .next()
            .and_then(|p| p.trim_end_matches('/').parse::<u16>().ok())
            .unwrap_or(8888);
        assert_eq!(port2, 9999);
    }

    #[tokio::test]
    async fn test_health_check_unreachable() {
        // Health check against a port nothing listens on should return false
        assert!(!health_check("http://127.0.0.1:19999").await);
    }
}
