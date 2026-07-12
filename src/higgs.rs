//! Higgs inference server sidecar management.
//!
//! Higgs is a pure-Rust MLX inference server for Apple Silicon. When
//! `localBackend == "higgs"`, nanobot auto-starts Higgs as a detached
//! background daemon and manages its lifecycle via PID file.
//!
//! Clean nanobot exits stop the managed Higgs PID. If nanobot dies before
//! cleanup, subsequent `nanobot agent -l` invocations detect the running
//! instance and reuse or replace it through the PID file.

use std::collections::HashSet;
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

/// PID file path for a named role: `~/.nanobot/higgs.pid` (main) or
/// `~/.nanobot/higgs-{role}.pid` (e.g. compaction sidecar).
fn pid_path_for(role: &str) -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let name = if role == "higgs" {
        "higgs.pid".to_string()
    } else {
        format!("higgs-{role}.pid")
    };
    home.join(".nanobot").join(name)
}

/// Log file path for a named role.
fn log_path_for(role: &str) -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let name = if role == "higgs" {
        "higgs.log".to_string()
    } else {
        format!("higgs-{role}.log")
    };
    home.join(".nanobot").join(name)
}

/// Backward-compat wrappers for the main ("higgs") instance.
#[allow(dead_code)]
fn pid_path() -> PathBuf {
    pid_path_for("higgs")
}

/// Read the stored PID for the main instance, if any.
#[allow(dead_code)]
fn read_pid() -> Option<u32> {
    fs::read_to_string(pid_path())
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

/// Read the stored PID for a named role, if any.
fn read_pid_for(role: &str) -> Option<u32> {
    fs::read_to_string(pid_path_for(role))
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
    local_model: &str,
) -> Result<StartResult, String> {
    server_start_role(bin, port, model_dir, local_model, "higgs").await
}

/// Start a Higgs instance for a named role. `role == "higgs"` is the main
/// instance (writes `~/.nanobot/higgs.pid`); any other tag (e.g.
/// `"compaction"`) gets its own PID/log files so a sidecar can coexist with
/// the main server without colliding on the singleton PID file.
pub(crate) async fn server_start_role(
    bin: &Path,
    port: u16,
    model_dir: &str,
    local_model: &str,
    role: &str,
) -> Result<StartResult, String> {
    // Already running and healthy?
    if let Some(pid) = read_pid_for(role) {
        if pid_is_alive(pid) && wait_for_ready(port, 3).await {
            if is_serving_expected_model(port, model_dir, local_model).await {
                tracing::info!(pid, port, role, "higgs already running");
                return Ok(StartResult::Ready);
            }
            tracing::info!(
                pid,
                port,
                role,
                "higgs running but serving wrong model, restarting"
            );
            server_stop_role(role)?;
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        } else {
            let _ = fs::remove_file(pid_path_for(role));
        }
    }

    if wait_for_ready(port, 1).await {
        // An externally-managed higgs (one nanobot did not spawn) is the user's
        // server, reached via the localApiBase they configured. Trust it: a
        // model-name mismatch is a warning, not a hard error. Returning Err here
        // made the caller clear localApiBase and silently fall back to a dead
        // default port (the :8080 bug).
        if !is_serving_expected_model(port, model_dir, local_model).await {
            tracing::warn!(
                port,
                "externally-managed higgs is serving a model that does not match the \
                 configured mlxModelDir/localModel; using it anyway (run /model to switch)"
            );
        } else {
            tracing::info!(port, "higgs port already responding (externally managed)");
        }
        return Ok(StartResult::Ready);
    }

    // Only the main instance owns the singleton pgrep guard; a sidecar
    // (e.g. compaction) is expected to coexist with a main `higgs serve`.
    if role == "higgs" {
        if let Some(existing_pid) = find_existing_higgs_process() {
            return Err(format!(
                "another Higgs process is already running (pid {existing_pid}).\n\
                 Stop it first, or point localApiBase at its port to reuse it.\n\
                 To force nanobot to start its own instance: kill {existing_pid}"
            ));
        }
    }

    if let Some(parent) = pid_path_for(role).parent() {
        let _ = fs::create_dir_all(parent);
    }

    let log_file =
        fs::File::create(log_path_for(role)).map_err(|e| format!("failed to create higgs log: {e}"))?;
    let log_err = log_file
        .try_clone()
        .map_err(|e| format!("failed to clone log handle: {e}"))?;

    let devnull =
        fs::File::open("/dev/null").map_err(|e| format!("failed to open /dev/null: {e}"))?;

    let mut cmd = std::process::Command::new(bin);
    // Throughput profile only. TurboQuant KV @ 4-bit was tried here and
    // removed: the dequant overhead on long-context decode made everything
    // slower in practice on this workload — full-precision KV wins.
    // HIGGS_CHUNKED_PREFILL_CHUNK_SIZE overrides the hardcoded 1024 floor
    // for throughput profile (384 *max(1024) = 1024 for huge+MoE).
    cmd.env("HIGGS_CHUNKED_PREFILL_CHUNK_SIZE", "4096");
    cmd.args([
        "serve",
        "--model",
        model_dir,
        "--port",
        &port.to_string(),
        "--mlx-profile",
        "throughput",
    ]);
    cmd.stdin(devnull);
    cmd.stdout(log_file);
    cmd.stderr(log_err);

    platform::set_new_session(&mut cmd);

    let child = cmd
        .spawn()
        .map_err(|e| format!("failed to spawn higgs: {e}"))?;

    let child_pid = child.id();

    fs::write(pid_path_for(role), child_pid.to_string())
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
            log_path_for(role).display()
        ));
    }

    Ok(StartResult::Ready)
}

/// Stop a running Higgs instance (main role).
pub(crate) fn server_stop() -> Result<(), String> {
    server_stop_role("higgs")
}

/// Stop a running Higgs instance for a named role.
pub(crate) fn server_stop_role(role: &str) -> Result<(), String> {
    let Some(pid) = read_pid_for(role) else {
        return Ok(());
    };

    if !pid_is_alive(pid) {
        let _ = fs::remove_file(pid_path_for(role));
        return Ok(());
    }

    platform::send_signal(pid, libc::SIGTERM);

    // Wait briefly for graceful shutdown
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while pid_is_alive(pid) {
        if std::time::Instant::now() >= deadline {
            tracing::warn!(pid, role, "higgs still running after SIGTERM, sending SIGKILL");
            platform::send_signal(pid, libc::SIGKILL);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    let _ = fs::remove_file(pid_path_for(role));
    tracing::info!(pid, role, "higgs stopped");
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

/// Lifecycle spec for the compaction Higgs sidecar.
///
/// In always-on mode the sidecar is started once at REPL startup and kept
/// resident. In on-demand mode (the default) it is spawned just before a
/// compaction pass and stopped right after, so the lightweight compaction
/// model never competes with the main model for unified memory between
/// compactions.
///
/// On-demand uses a reference-counted LEASE so concurrent gateway sessions
/// compacting at the same time can't stop the sidecar out from under each
/// other: `ensure_up` takes a lease, `release` drops one, and the sidecar is
/// only stopped when the last lease is dropped. `ensure_up` is idempotent (a
/// running sidecar is a no-op), and `release`/`ensure_up` are no-ops in
/// always-on mode. Always held behind an `Arc` (one per AgentLoop).
pub(crate) struct CompactionSidecarSpec {
    pub bin: PathBuf,
    pub port: u16,
    pub dir: String,
    pub model: String,
    pub on_demand: bool,
}

/// Process-global outstanding leases on the compaction sidecar. Global (not
/// per-`CompactionSidecarSpec`) because cron reflection, REPL `/learn`, exit
/// reflection, and the LCM compaction task each construct their own spec from
/// config — independent per-spec counters couldn't coordinate and one could
/// `release()` the sidecar while another is mid-summarize. There is only one
/// compaction sidecar (role `"compaction"`) per process, so one shared counter
/// is correct. The sidecar is stopped only when the LAST lease drops.
static COMPACTION_SIDE_LEASES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Serializes sidecar lifecycle transitions (start, stop). Without it, a
/// release could CAS the lease count 1→0 and stop the sidecar in the window
/// between a concurrent ensure_up's health check and its lease increment.
/// Held briefly when the sidecar is already up; only the (rare) cold spawn
/// holds it for the load duration.
static COMPACTION_LIFECYCLE: std::sync::LazyLock<tokio::sync::Mutex<()>> =
    std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));

impl std::fmt::Debug for CompactionSidecarSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactionSidecarSpec")
            .field("port", &self.port)
            .field("model", &self.model)
            .field("on_demand", &self.on_demand)
            .field(
                "active_leases",
                &COMPACTION_SIDE_LEASES.load(std::sync::atomic::Ordering::Relaxed),
            )
            .finish()
    }
}

impl CompactionSidecarSpec {
    /// Bind an LCM endpoint that targets this managed sidecar to the literal
    /// model id the sidecar loaded. Higgs' chat endpoint does not guarantee
    /// the transport alias `active`; using the known id avoids a late 404 and
    /// silent deterministic-compaction fallback.
    pub(crate) fn bind_lcm_endpoint_model(
        &self,
        lcm: &mut crate::config::schema::LcmSchemaConfig,
    ) {
        let Some(endpoint) = lcm.compaction_endpoint.as_mut() else {
            return;
        };
        let Ok(url) = url::Url::parse(&endpoint.url) else {
            return;
        };
        let targets_this_sidecar = url.port_or_known_default() == Some(self.port)
            && url.host_str().is_some_and(|host| {
                host.eq_ignore_ascii_case("localhost")
                    || host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback())
            });
        if targets_this_sidecar {
            endpoint.model.clone_from(&self.model);
        }
    }

    /// Build from config. Returns `None` when there is no compaction sidecar
    /// (no port), no spawnable model directory, or no `higgs` binary on disk —
    /// in all those cases compaction falls back to the main model.
    pub(crate) fn from_config(config: &crate::config::schema::Config) -> Option<Self> {
        let port = config.agents.defaults.higgs_compaction_port?;
        // Don't build a sidecar spec (and later spawn it) when the memory
        // provider targets a non-localhost endpoint — the sidecar wouldn't be
        // used, and cron/`/learn`/exit-reflection would waste up to 60s
        // spawning a model the Reflector never calls. If a memory provider is
        // configured, it must point at localhost for the sidecar to apply.
        let memory_targets_remote = config
            .memory
            .provider
            .as_ref()
            .and_then(|p| p.api_base.as_deref())
            .is_some_and(|b| !b.contains("127.0.0.1") && !b.contains("localhost"));
        if memory_targets_remote {
            return None;
        }
        let bin = find_binary()?;
        let (dir, model) = compaction_sidecar_config(config);
        let dir = dir?;
        let on_demand = config.agents.defaults.higgs_compaction_on_demand.unwrap_or(true);
        Some(Self {
            bin,
            port,
            dir,
            model,
            on_demand,
        })
    }

    /// Ensure the sidecar is reachable before a compaction/memory call, and
    /// take a lease on the global counter. Idempotent: a running sidecar
    /// (always-on, or a prior on-demand spawn) returns `Ready` immediately. On
    /// failure NO lease is taken (so a matching `release()` is a no-op).
    pub(crate) async fn ensure_up(&self) -> Result<(), String> {
        if !self.on_demand {
            return Ok(());
        }
        // Serialize start vs stop so a concurrent release can't kill the
        // sidecar between this health check and the lease increment.
        let _guard = COMPACTION_LIFECYCLE.lock().await;
        match server_start_role(&self.bin, self.port, &self.dir, &self.model, "compaction").await {
            Ok(_) => {
                COMPACTION_SIDE_LEASES.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Drop a lease taken by `ensure_up`. The sidecar is stopped only when the
    /// LAST lease drops. Safe against an unheld release (e.g. a caller whose
    /// `ensure_up` failed): the CAS loop refuses to decrement past zero. The
    /// lifecycle lock serializes the stop against concurrent starts.
    pub(crate) async fn release(&self) {
        if !self.on_demand {
            return;
        }
        let _guard = COMPACTION_LIFECYCLE.lock().await;
        loop {
            let cur = COMPACTION_SIDE_LEASES.load(std::sync::atomic::Ordering::SeqCst);
            if cur == 0 {
                return; // no lease held — nothing to release (ensure_up failed)
            }
            if COMPACTION_SIDE_LEASES
                .compare_exchange(cur, cur - 1, std::sync::atomic::Ordering::SeqCst, std::sync::atomic::Ordering::SeqCst)
                .is_ok()
            {
                if cur == 1 {
                    let _ = server_stop_role("compaction");
                }
                return;
            }
        }
    }
}

/// Resolve the compaction sidecar's `(model_dir, model)` from config,
/// mirroring the main-instance fallbacks: configured compaction dir, else the
/// main model dir; configured compaction model, else the main local model.
/// Single source of truth for the spawn path and `/restart`.
pub(crate) fn compaction_sidecar_config(
    config: &crate::config::schema::Config,
) -> (Option<String>, String) {
    let dir = config
        .agents
        .defaults
        .higgs_compaction_model_dir
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| resolve_model_dir(config).ok());
    let model = config
        .agents
        .defaults
        .higgs_compaction_model
        .clone()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| config.agents.defaults.local_model.clone());
    (dir, model)
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

/// A Higgs model that nanobot can offer in `/model`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RuntimeModelCandidate {
    /// User-facing picker id.
    pub id: String,
    /// Path/HF id sent to Higgs' runtime model endpoints.
    pub path: String,
    /// Runtime model name exposed by Higgs after load.
    pub name: String,
}

/// Discover local MLX model directories that Higgs can switch to at runtime.
///
/// Higgs' runtime endpoint accepts either an existing model directory or a
/// cached Hugging Face id. `mlxModelDir` remains the startup/default model;
/// for the picker it may also point at a parent folder to scan.
pub(crate) fn discover_runtime_model_candidates(
    config: &crate::config::schema::Config,
) -> Vec<RuntimeModelCandidate> {
    let mut candidates = Vec::new();
    let mut seen = HashSet::new();
    let mut roots = Vec::new();

    if let Some(ref dir) = config.agents.defaults.mlx_model_dir {
        let dir = dir.trim();
        if !dir.is_empty() && dir != "auto" {
            let expanded = crate::utils::helpers::expand_tilde(dir);
            if looks_like_mlx_model_dir(&expanded) {
                let id = model_id_from_spec(dir);
                let preferred_name =
                    preferred_runtime_name(config).unwrap_or_else(|| model_name(&id));
                push_candidate(
                    &mut candidates,
                    &mut seen,
                    id,
                    dir.to_string(),
                    preferred_name,
                );

                if let Some(parent) = expanded.parent() {
                    roots.push(parent.to_path_buf());
                    if let Some(grandparent) = parent.parent() {
                        roots.push(grandparent.to_path_buf());
                    }
                }
            } else {
                roots.push(expanded);
            }
        }
    }

    for config_path in higgs_config_paths() {
        collect_higgs_config_models(&config_path, &mut candidates, &mut seen, &mut roots);
    }

    if let Some(home) = dirs::home_dir() {
        roots.push(home.join(".cache/lm-studio/models"));
    }

    let mut seen_roots = HashSet::new();
    for root in roots {
        if seen_roots.insert(root.clone()) {
            collect_mlx_dirs(&root, 3, &mut candidates, &mut seen);
        }
    }

    if let Some(cache) = hf_cache_root() {
        collect_hf_cache_models(&cache, &mut candidates, &mut seen);
    }

    candidates.sort_by(|a, b| a.id.cmp(&b.id));
    candidates
}

fn higgs_config_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(path) = std::env::var("HIGGS_CONFIG") {
        paths.push(crate::utils::helpers::expand_tilde(&path));
    }
    if let Ok(path) = std::env::var("HIGGS_CONFIG_PATH") {
        paths.push(crate::utils::helpers::expand_tilde(&path));
    }
    if let Some(home) = dirs::home_dir() {
        paths.push(home.join(".config/higgs/config.toml"));
    }
    paths
}

fn collect_higgs_config_models(
    config_path: &Path,
    candidates: &mut Vec<RuntimeModelCandidate>,
    seen: &mut HashSet<String>,
    roots: &mut Vec<PathBuf>,
) {
    let Ok(text) = fs::read_to_string(config_path) else {
        return;
    };

    for model in parse_higgs_config_models(&text) {
        let id = model_id_from_spec(&model.path);
        let expanded = crate::utils::helpers::expand_tilde(&model.path);
        let model_path = if model_spec_is_local_path(&model.path) {
            if looks_like_mlx_model_dir(&expanded) {
                if let Some(parent) = expanded.parent() {
                    roots.push(parent.to_path_buf());
                }
                Some(model.path.clone())
            } else if expanded.is_dir() {
                roots.push(expanded);
                None
            } else {
                None
            }
        } else {
            hf_cached_model_path_for_id(&model.path).map(|path| path.display().to_string())
        };

        let Some(model_path) = model_path else {
            continue;
        };
        let name = if model.name.is_empty() {
            model_name(&id)
        } else {
            model.name
        };
        push_candidate(candidates, seen, id, model_path, name);
    }
}

fn model_spec_is_local_path(spec: &str) -> bool {
    spec.starts_with("~/") || spec.starts_with('/')
}

fn hf_cached_model_path_for_id(model_id: &str) -> Option<PathBuf> {
    let (org, name) = model_id.split_once('/')?;
    if org.is_empty() || name.is_empty() {
        return None;
    }
    let cache = hf_cache_root()?;
    let repo = cache.join(format!("models--{org}--{name}"));
    usable_hf_cached_model_dir(&repo)
}

fn usable_hf_cached_model_dir(repo: &Path) -> Option<PathBuf> {
    if looks_like_mlx_model_dir(repo) {
        return Some(repo.to_path_buf());
    }

    let main_ref = fs::read_to_string(repo.join("refs/main")).ok()?;
    let revision = main_ref.trim();
    if revision.is_empty() || revision.contains('/') {
        return None;
    }
    let snapshot = repo.join("snapshots").join(revision);
    if looks_like_mlx_model_dir(&snapshot) {
        Some(snapshot)
    } else {
        None
    }
}

fn hf_id_from_cache_repo(path: &Path) -> Option<(String, String)> {
    let file_name = path.file_name().and_then(|n| n.to_str())?;
    let rest = file_name.strip_prefix("models--")?;
    let mut parts = rest.splitn(2, "--");
    let (Some(org), Some(name)) = (parts.next(), parts.next()) else {
        return None;
    };
    if org.is_empty() || name.is_empty() {
        None
    } else {
        Some((org.to_string(), name.to_string()))
    }
}

fn path_has_safetensors_payload(path: &Path) -> bool {
    let Ok(entries) = fs::read_dir(path) else {
        return false;
    };
    entries.flatten().any(|entry| {
        entry
            .file_name()
            .to_string_lossy()
            .ends_with(".safetensors")
    })
}

fn looks_like_mlx_model_dir(path: &Path) -> bool {
    if !path.is_dir() || !path.join("config.json").is_file() {
        return false;
    }
    if !(path.join("tokenizer.json").is_file() || path.join("tokenizer.model").is_file()) {
        return false;
    }
    path_has_safetensors_payload(path)
}

fn collect_hf_cache_models(
    cache: &Path,
    candidates: &mut Vec<RuntimeModelCandidate>,
    seen: &mut HashSet<String>,
) {
    let Ok(entries) = fs::read_dir(cache) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some((org, name)) = hf_id_from_cache_repo(&path) else {
            continue;
        };
        let Some(model_dir) = usable_hf_cached_model_dir(&path) else {
            continue;
        };
        let id = format!("{org}/{name}");
        push_candidate(
            candidates,
            seen,
            id.clone(),
            model_dir.display().to_string(),
            name,
        );
    }
}

fn hf_cache_root() -> Option<PathBuf> {
    if let Ok(cache) = std::env::var("HF_HUB_CACHE") {
        return Some(crate::utils::helpers::expand_tilde(&cache));
    }
    if let Ok(cache) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return Some(crate::utils::helpers::expand_tilde(&cache));
    }
    if let Ok(home) = std::env::var("HF_HOME") {
        return Some(crate::utils::helpers::expand_tilde(&home).join("hub"));
    }
    dirs::home_dir().map(|home| home.join(".cache/huggingface/hub"))
}

fn model_id_from_spec(spec: &str) -> String {
    if model_spec_is_local_path(spec) {
        model_id_from_path(&crate::utils::helpers::expand_tilde(spec))
    } else {
        spec.to_string()
    }
}

fn model_id_from_path(path: &Path) -> String {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("model");
    let Some(parent) = path.parent() else {
        return name.to_string();
    };
    let Some(org) = parent.file_name().and_then(|n| n.to_str()) else {
        return name.to_string();
    };
    let is_lmstudio_models_root = parent
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        == Some("models");
    if is_lmstudio_models_root {
        format!("{org}/{name}")
    } else {
        name.to_string()
    }
}

fn model_name(id: &str) -> String {
    id.rsplit('/').next().unwrap_or(id).to_string()
}

fn models_url_from_base(api_base: &str) -> String {
    let base = api_base.trim_end_matches('/');
    if base.ends_with("/v1") {
        format!("{base}/models")
    } else {
        format!("{base}/v1/models")
    }
}

fn switch_url_from_base(api_base: &str) -> String {
    let base = api_base.trim_end_matches('/');
    if base.ends_with("/v1") {
        format!("{base}/models/switch")
    } else {
        format!("{base}/v1/models/switch")
    }
}

fn health_url_from_base(api_base: &str) -> String {
    let base = api_base.trim_end_matches('/');
    let base = base.strip_suffix("/v1").unwrap_or(base);
    format!("{base}/health")
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HiggsConfigModel {
    path: String,
    name: String,
}

fn parse_higgs_config_models(text: &str) -> Vec<HiggsConfigModel> {
    let mut models = Vec::new();
    let mut in_model = false;
    let mut path: Option<String> = None;
    let mut name: Option<String> = None;

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let header = line.split('#').next().unwrap_or(line).trim();
        if header.starts_with("[[") {
            push_parsed_higgs_model(&mut models, &mut path, &mut name);
            in_model = header == "[[models]]";
            continue;
        }

        if !in_model {
            continue;
        }

        let Some((key, raw_value)) = line.split_once('=') else {
            continue;
        };
        let Some(value) = parse_toml_string_value(raw_value) else {
            continue;
        };

        match key.trim() {
            "path" => path = Some(value),
            "name" => name = Some(value),
            _ => {}
        }
    }

    push_parsed_higgs_model(&mut models, &mut path, &mut name);
    models
}

fn push_parsed_higgs_model(
    models: &mut Vec<HiggsConfigModel>,
    path: &mut Option<String>,
    name: &mut Option<String>,
) {
    let Some(path_value) = path.take() else {
        *name = None;
        return;
    };
    let path_value = path_value.trim();
    if !path_value.is_empty() {
        models.push(HiggsConfigModel {
            path: path_value.to_string(),
            name: name.take().unwrap_or_default(),
        });
    } else {
        *name = None;
    }
}

fn parse_toml_string_value(raw: &str) -> Option<String> {
    let value = raw.trim();
    if let Some(rest) = value.strip_prefix('"') {
        return parse_basic_quoted_string(rest);
    }
    if let Some(rest) = value.strip_prefix('\'') {
        return rest.split_once('\'').map(|(parsed, _)| parsed.to_string());
    }

    let value = value.split('#').next().unwrap_or(value).trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

fn parse_basic_quoted_string(rest: &str) -> Option<String> {
    let mut parsed = String::new();
    let mut escaped = false;
    for ch in rest.chars() {
        if escaped {
            let value = match ch {
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                '"' => '"',
                '\\' => '\\',
                other => other,
            };
            parsed.push(value);
            escaped = false;
            continue;
        }

        match ch {
            '\\' => escaped = true,
            '"' => return Some(parsed),
            other => parsed.push(other),
        }
    }
    None
}

fn preferred_runtime_name(config: &crate::config::schema::Config) -> Option<String> {
    let lms = config.agents.defaults.lms_main_model.trim();
    if !lms.is_empty() && lms != "active" {
        return Some(model_name(lms));
    }
    let local = config.agents.defaults.local_model.trim();
    if !local.is_empty() && local != "active" && !local.ends_with(".gguf") {
        return Some(model_name(local));
    }
    None
}

fn push_candidate(
    candidates: &mut Vec<RuntimeModelCandidate>,
    seen: &mut HashSet<String>,
    id: String,
    path: String,
    name: String,
) {
    let key = path.to_ascii_lowercase();
    if seen.insert(key) {
        candidates.push(RuntimeModelCandidate { id, path, name });
    }
}

fn collect_mlx_dirs(
    root: &Path,
    depth: usize,
    candidates: &mut Vec<RuntimeModelCandidate>,
    seen: &mut HashSet<String>,
) {
    if looks_like_mlx_model_dir(root) {
        let id = model_id_from_path(root);
        push_candidate(
            candidates,
            seen,
            id.clone(),
            root.display().to_string(),
            model_name(&id),
        );
        return;
    }
    if depth == 0 || !root.is_dir() {
        return;
    }

    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_mlx_dirs(&path, depth - 1, candidates, seen);
        }
    }
}

/// List models from a Higgs OpenAI-compatible base URL, with optional auth.
pub(crate) async fn list_served_models_at(api_base: &str, api_key: &str) -> Vec<String> {
    let url = models_url_from_base(api_base);
    let client = reqwest::Client::new();
    let mut req = client.get(&url);
    if !api_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {api_key}"));
    }
    let Ok(resp) = req.timeout(std::time::Duration::from_secs(3)).send().await else {
        return Vec::new();
    };
    let Ok(json) = resp.json::<serde_json::Value>().await else {
        return Vec::new();
    };
    model_ids_from_models_json(&json)
}

/// List resident model ids, dropping entries the endpoint health marks unavailable.
pub(crate) async fn list_available_served_models_at(api_base: &str, api_key: &str) -> Vec<String> {
    let served = list_served_models_at(api_base, api_key).await;
    if served.is_empty() {
        return served;
    }
    let unavailable = unavailable_models_at(api_base, api_key).await;
    filter_available_model_ids(served, &unavailable)
}

/// Probe whether this endpoint implements Higgs' runtime switch API.
///
/// A real switch-capable Higgs answers GET /v1/models/switch with 405 and
/// Allow: POST. Plain OpenAI-compatible resident endpoints commonly return
/// 404; those must not receive filesystem switch candidates.
pub(crate) async fn supports_runtime_model_switch_at(api_base: &str, api_key: &str) -> bool {
    let url = switch_url_from_base(api_base);
    let client = reqwest::Client::new();
    let mut req = client.get(&url);
    if !api_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {api_key}"));
    }
    let Ok(resp) = req.timeout(std::time::Duration::from_secs(3)).send().await else {
        return false;
    };
    if resp.status() != reqwest::StatusCode::METHOD_NOT_ALLOWED {
        return false;
    }
    resp.headers()
        .get(reqwest::header::ALLOW)
        .and_then(|value| value.to_str().ok())
        .map(|allow| {
            allow
                .split(',')
                .any(|method| method.trim().eq_ignore_ascii_case("POST"))
        })
        .unwrap_or(true)
}

/// Switch Higgs to one runtime model using the free-then-load endpoint.
pub(crate) async fn switch_runtime_model(
    api_base: &str,
    api_key: &str,
    path: &str,
    name: &str,
    timeout_secs: u64,
) -> Result<String, String> {
    let url = switch_url_from_base(api_base);
    let mut body = serde_json::json!({ "path": path });
    if !name.is_empty() {
        body["name"] = serde_json::json!(name);
    }

    let client = reqwest::Client::new();
    let mut req = client.post(&url).json(&body);
    if !api_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {api_key}"));
    }
    let resp = req
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .send()
        .await
        .map_err(|e| format!("Higgs switch request failed: {e}"))?;

    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format!(
            "Higgs switch failed (HTTP {status}): {}",
            text.trim()
        ));
    }

    let loaded = serde_json::from_str::<serde_json::Value>(&text)
        .ok()
        .and_then(|json| json.get("id").and_then(|id| id.as_str()).map(String::from))
        .unwrap_or_else(|| name.to_string());
    Ok(loaded)
}

/// Restart Higgs with a (potentially different) model.
///
/// Stops the running instance, waits briefly for port release, then starts
/// with the new model directory.
pub(crate) async fn server_restart(
    bin: &Path,
    port: u16,
    model_dir: &str,
    local_model: &str,
) -> Result<StartResult, String> {
    server_restart_role(bin, port, model_dir, local_model, "higgs").await
}

/// Restart a Higgs instance for a named role (stop then start).
pub(crate) async fn server_restart_role(
    bin: &Path,
    port: u16,
    model_dir: &str,
    local_model: &str,
    role: &str,
) -> Result<StartResult, String> {
    server_stop_role(role)?;
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    server_start_role(bin, port, model_dir, local_model, role).await
}

/// Check if the running Higgs is serving a model that matches `expected_dir`.
///
/// Scans the FULL set of served models (an externally-managed Higgs can serve
/// several at once) and matches each against the last path component of
/// `expected_dir`. Returns `true` if any served model matches, or if the
/// server can't be queried (optimistic fallback).
async fn is_serving_expected_model(port: u16, expected_dir: &str, preferred: &str) -> bool {
    let served = list_served_models(port).await;
    if served.is_empty() {
        return true; // Can't query → assume ok
    }
    let expected_name = Path::new(expected_dir)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    if expected_name.is_empty() && preferred.is_empty() {
        return true;
    }
    // Accept if any served model matches the model-dir basename OR the configured
    // model name. Matching `preferred` (localModel) is essential when Higgs serves
    // a custom-named model (e.g. "qwen36-35b") that doesn't correspond to the
    // mlxModelDir basename — otherwise nanobot wrongly rejects a healthy server.
    served.iter().any(|id| {
        (!expected_name.is_empty() && model_id_matches(id, &expected_name))
            || (!preferred.is_empty() && model_id_matches(id, preferred))
    })
}

/// Case-insensitive fuzzy match between two model identifiers.
///
/// Model ids reach us as full paths, directory basenames, or short names, so a
/// match is accepted when either id contains the other after normalization
/// (e.g. dir basename `MiniCPM5-1B-4bit` matches served id `minicpm5-1b`).
/// Separator punctuation is ignored so a served alias like `qwen36-35b` still
/// matches the model dir `Qwen3.6-35B-A3B-4bit` — a false mismatch here made
/// startup restart a healthy server or reject its served id.
pub(crate) fn model_id_matches(a: &str, b: &str) -> bool {
    let a = normalize_model_id(a);
    let b = normalize_model_id(b);
    !a.is_empty() && !b.is_empty() && (a.contains(&b) || b.contains(&a))
}

/// Lowercase and drop everything but ASCII alphanumerics for fuzzy id compares.
fn normalize_model_id(id: &str) -> String {
    id.chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// Decide which served model id nanobot should adopt at startup.
///
/// `None` = keep the configured id: the server reported nothing (down or not
/// OpenAI-compatible), or the configured id already IS a served id verbatim.
/// `Some(id)` = the endpoint is healthy but the configured id is stale or
/// non-canonical — use `id` (the server's own identifier) so the first
/// request cannot 404 on a name the server never loaded.
pub(crate) fn adopt_served_model(configured: &str, served: &[String]) -> Option<String> {
    match served.iter().find(|id| model_id_matches(id, configured)) {
        Some(id) if id.as_str() == configured => None,
        Some(id) => Some(id.clone()),
        None => served.first().cloned(),
    }
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
    model_ids_from_models_json(&json)
}

fn model_ids_from_models_json(json: &serde_json::Value) -> Vec<String> {
    json.get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|id| id.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

async fn unavailable_models_at(api_base: &str, api_key: &str) -> Vec<String> {
    let url = health_url_from_base(api_base);
    let client = reqwest::Client::new();
    let mut req = client.get(&url);
    if !api_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {api_key}"));
    }
    let Ok(resp) = req.timeout(std::time::Duration::from_secs(3)).send().await else {
        return Vec::new();
    };
    let Ok(json) = resp.json::<serde_json::Value>().await else {
        return Vec::new();
    };
    unavailable_models_from_health_json(&json)
}

fn unavailable_models_from_health_json(json: &serde_json::Value) -> Vec<String> {
    json.get("models")
        .and_then(|models| models.as_array())
        .map(|models| {
            models
                .iter()
                .filter(|model| {
                    model
                        .get("available")
                        .and_then(|available| available.as_bool())
                        == Some(false)
                })
                .filter_map(|model| {
                    model
                        .get("name")
                        .and_then(|name| name.as_str())
                        .map(String::from)
                })
                .collect()
        })
        .unwrap_or_default()
}

fn filter_available_model_ids(models: Vec<String>, unavailable: &[String]) -> Vec<String> {
    if unavailable.is_empty() {
        return models;
    }
    models
        .into_iter()
        .filter(|id| {
            !unavailable
                .iter()
                .any(|blocked| model_id_matches(id, blocked))
        })
        .collect()
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
    fn compaction_sidecar_binds_lcm_to_loaded_model_id() {
        let spec = CompactionSidecarSpec {
            bin: PathBuf::from("/tmp/higgs"),
            port: 8001,
            dir: "/models/Bonsai".to_string(),
            model: "Bonsai-1.7B-mlx-1bit".to_string(),
            on_demand: true,
        };
        let mut lcm = crate::config::schema::LcmSchemaConfig {
            compaction_endpoint: Some(crate::config::schema::ModelEndpoint {
                url: "http://127.0.0.1:8001/v1".to_string(),
                model: "active".to_string(),
            }),
            ..Default::default()
        };

        spec.bind_lcm_endpoint_model(&mut lcm);

        assert_eq!(
            lcm.compaction_endpoint.as_ref().unwrap().model,
            "Bonsai-1.7B-mlx-1bit"
        );
    }

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

    #[test]
    fn model_id_matches_ignores_separator_punctuation() {
        // Served alias "qwen36-35b" vs model dir "Qwen3.6-35B-A3B-4bit": the
        // dot in "3.6" must not defeat the match (regression: startup
        // restarted a healthy Higgs over this false mismatch).
        assert!(model_id_matches("qwen36-35b", "Qwen3.6-35B-A3B-4bit"));
        assert!(model_id_matches("Qwen3.6-35B-A3B-4bit", "qwen36-35b"));
        // Distinct models still don't match.
        assert!(!model_id_matches("bonsai-8b-mlx", "Qwen3.6-35B-A3B-4bit"));
    }

    #[test]
    fn adopt_served_model_replaces_stale_configured_name() {
        let served = vec!["qwen36-35b".to_string()];
        // Stale configured name → adopt what the server actually serves.
        assert_eq!(
            adopt_served_model("vibethinker3", &served),
            Some("qwen36-35b".to_string())
        );
        // Configured id is already the served id verbatim → keep.
        assert_eq!(adopt_served_model("qwen36-35b", &served), None);
        // Fuzzy match to a different canonical id → adopt the server's form.
        assert_eq!(
            adopt_served_model("Qwen3.6-35B-A3B-4bit", &served),
            Some("qwen36-35b".to_string())
        );
        // Server reported nothing → no adoption, keep configured.
        assert_eq!(adopt_served_model("vibethinker3", &[]), None);
        // Higgs "active" transport already configured and served → keep.
        assert_eq!(
            adopt_served_model("active", &["active".to_string()]),
            None
        );
    }

    #[test]
    fn test_runtime_model_id_from_lmstudio_path() {
        let path = Path::new("/tmp/cache/lm-studio/models/mlx-community/Qwen3-4bit");
        assert_eq!(model_id_from_path(path), "mlx-community/Qwen3-4bit");
    }

    #[test]
    fn test_parse_higgs_config_models_ignores_commented_entries() {
        let models = parse_higgs_config_models(
            r#"
            #[[models]]
            #path = "/tmp/commented"
            #name = "commented"

            [[models]]
            path = "/tmp/active"
            name = "active-model"

            [[routes]]
            pattern = "gpt-.*"
            provider = "openai"
            "#,
        );

        assert_eq!(
            models,
            vec![HiggsConfigModel {
                path: "/tmp/active".to_string(),
                name: "active-model".to_string()
            }]
        );
    }

    #[test]
    fn test_collect_higgs_config_models_keeps_alias_and_scan_root() {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp
            .path()
            .join("lm-studio/models/mlx-community/Qwen3-Test-4bit");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("config.json"), "{}").unwrap();
        fs::write(model_dir.join("tokenizer.json"), "{}").unwrap();
        fs::write(model_dir.join("model.safetensors"), "").unwrap();

        let config_path = tmp.path().join("config.toml");
        fs::write(
            &config_path,
            format!(
                r#"
                [[models]]
                path = "{}"
                name = "qwen-test"
                "#,
                model_dir.display()
            ),
        )
        .unwrap();

        let mut candidates = Vec::new();
        let mut seen = HashSet::new();
        let mut roots = Vec::new();
        collect_higgs_config_models(&config_path, &mut candidates, &mut seen, &mut roots);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].name, "qwen-test");
        assert_eq!(candidates[0].path, model_dir.display().to_string());
        assert!(roots.iter().any(|root| root.ends_with("mlx-community")));
    }

    #[test]
    fn test_collect_higgs_config_models_skips_missing_local_paths() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.toml");
        fs::write(
            &config_path,
            format!(
                r#"
                [[models]]
                path = "{}"
                name = "old-missing"
                "#,
                tmp.path().join("old-model").display()
            ),
        )
        .unwrap();

        let mut candidates = Vec::new();
        let mut seen = HashSet::new();
        let mut roots = Vec::new();
        collect_higgs_config_models(&config_path, &mut candidates, &mut seen, &mut roots);

        assert!(
            candidates.is_empty(),
            "missing local config paths are stale"
        );
        assert!(
            roots.is_empty(),
            "missing local config paths should not seed scans"
        );
    }

    #[test]
    fn test_collect_mlx_dirs_scans_lmstudio_parent_folder() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("lm-studio/models");
        let model_dir = root.join("mlx-community/Qwen3-Test-4bit");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("config.json"), "{}").unwrap();
        fs::write(model_dir.join("tokenizer.json"), "{}").unwrap();
        fs::write(model_dir.join("model.safetensors"), "").unwrap();

        let mut candidates = Vec::new();
        let mut seen = HashSet::new();
        collect_mlx_dirs(&root, 3, &mut candidates, &mut seen);

        assert!(candidates
            .iter()
            .any(|candidate| candidate.id == "mlx-community/Qwen3-Test-4bit"));
    }

    #[test]
    fn test_collect_mlx_dirs_requires_safetensors_payload() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("lm-studio/models");
        let model_dir = root.join("old-org/IndexOnly-4bit");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("config.json"), "{}").unwrap();
        fs::write(model_dir.join("tokenizer.json"), "{}").unwrap();
        fs::write(model_dir.join("model.safetensors.index.json"), "{}").unwrap();

        let mut candidates = Vec::new();
        let mut seen = HashSet::new();
        collect_mlx_dirs(&root, 3, &mut candidates, &mut seen);

        assert!(
            candidates.is_empty(),
            "index-only dirs should not appear as loadable models"
        );
    }

    #[test]
    fn test_collect_hf_cache_models_uses_only_real_main_snapshots() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path().join("hub");

        let stale = cache.join("models--old-org--OldModel");
        fs::create_dir_all(stale.join("refs")).unwrap();
        fs::write(stale.join("refs/main"), "deadbeef\n").unwrap();

        let index_only = cache.join("models--old-org--IndexOnly");
        let index_snapshot = index_only.join("snapshots/feedface");
        fs::create_dir_all(index_only.join("refs")).unwrap();
        fs::create_dir_all(&index_snapshot).unwrap();
        fs::write(index_only.join("refs/main"), "feedface\n").unwrap();
        fs::write(index_snapshot.join("config.json"), "{}").unwrap();
        fs::write(index_snapshot.join("tokenizer.json"), "{}").unwrap();
        fs::write(index_snapshot.join("model.safetensors.index.json"), "{}").unwrap();

        let real = cache.join("models--good-org--RealModel-4bit");
        let real_snapshot = real.join("snapshots/cafebabe");
        fs::create_dir_all(real.join("refs")).unwrap();
        fs::create_dir_all(&real_snapshot).unwrap();
        fs::write(real.join("refs/main"), "cafebabe\n").unwrap();
        fs::write(real_snapshot.join("config.json"), "{}").unwrap();
        fs::write(real_snapshot.join("tokenizer.json"), "{}").unwrap();
        fs::write(real_snapshot.join("model.safetensors"), "").unwrap();

        let mut candidates = Vec::new();
        let mut seen = HashSet::new();
        collect_hf_cache_models(&cache, &mut candidates, &mut seen);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].id, "good-org/RealModel-4bit");
        assert_eq!(candidates[0].name, "RealModel-4bit");
        assert_eq!(candidates[0].path, real_snapshot.display().to_string());
    }

    #[test]
    fn test_runtime_model_urls_from_base() {
        assert_eq!(
            models_url_from_base("http://127.0.0.1:8091/v1"),
            "http://127.0.0.1:8091/v1/models"
        );
        assert_eq!(
            switch_url_from_base("http://127.0.0.1:8091"),
            "http://127.0.0.1:8091/v1/models/switch"
        );
    }

    #[test]
    fn test_model_ids_from_models_json() {
        let json = serde_json::json!({
            "object": "list",
            "data": [
                { "id": "qwen", "object": "model" },
                { "id": "llama", "object": "model" },
                { "object": "model" }
            ]
        });
        assert_eq!(
            model_ids_from_models_json(&json),
            vec!["qwen".to_string(), "llama".to_string()]
        );
    }

    #[test]
    fn test_health_url_from_base() {
        assert_eq!(
            health_url_from_base("http://127.0.0.1:8000/v1"),
            "http://127.0.0.1:8000/health"
        );
        assert_eq!(
            health_url_from_base("http://127.0.0.1:8000"),
            "http://127.0.0.1:8000/health"
        );
    }

    #[test]
    fn test_unavailable_models_from_health_json() {
        let json = serde_json::json!({
            "status": "fm serve is running",
            "models": [
                { "name": "system", "available": true },
                { "name": "pcc", "available": false, "reason": "not available here" }
            ]
        });
        assert_eq!(
            unavailable_models_from_health_json(&json),
            vec!["pcc".to_string()]
        );
    }

    #[test]
    fn test_filter_available_model_ids_uses_health_unavailable() {
        let models = vec!["system".to_string(), "pcc".to_string()];
        let unavailable = vec!["PCC".to_string()];
        assert_eq!(
            filter_available_model_ids(models, &unavailable),
            vec!["system".to_string()]
        );
    }
}
