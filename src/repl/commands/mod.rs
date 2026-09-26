//! Command dispatch and handlers for the REPL.
//!
//! Contains `ReplContext`, `normalize_alias()`, `dispatch()`, and all
//! `cmd_xxx()` command handlers (split across submodules).

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
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use rustyline::error::ReadlineError;
use rustyline::ExternalPrinter as _;
use tokio::sync::mpsc;

use crate::agent::agent_loop::{AgentLoop, SharedCoreHandle};
use crate::agent::audit::AuditLog;
use crate::agent::provenance::{ClaimStatus, ClaimVerifier};
use crate::cli;
use crate::config::loader::{load_config, save_config};
use crate::config::schema::{Config, EmailConfig};
use crate::cron::service::CronService;
use crate::server;
use crate::tui;

// ============================================================================
// Submodule declarations
// ============================================================================

mod channels;
mod cluster;
mod lifecycle;
mod mutation;
mod read;

// ============================================================================
// ReplContext — all mutable state for the REPL command handlers
// ============================================================================

/// Mutable state for the REPL main loop and slash-command handlers.
///
/// Single flat struct (not sub-structs) because commands like `/local` need
/// mutable access to srv, agent_loop, core_handle, and config simultaneously.
pub(crate) struct ReplContext {
    pub config: Config,
    pub core_handle: SharedCoreHandle,
    pub agent_loop: AgentLoop,
    pub session_id: String,
    #[cfg_attr(not(feature = "voice"), allow(dead_code))]
    pub lang: Option<String>,
    pub srv: super::ServerState,
    pub current_model_path: PathBuf,
    pub active_channels: Vec<super::ActiveChannel>,
    pub display_tx: mpsc::UnboundedSender<String>,
    pub display_rx: mpsc::UnboundedReceiver<String>,
    pub cron_service: Arc<CronService>,
    pub email_config: Option<EmailConfig>,
    pub rl: Option<rustyline::DefaultEditor>,
    /// Health watchdog task handle — aborted on mode switch, restarted on `/local`.
    pub watchdog_handle: Option<tokio::task::JoinHandle<()>>,
    /// Sender for auto-restart requests (passed to watchdog).
    pub restart_tx: mpsc::UnboundedSender<crate::server::RestartRequest>,
    /// Receiver for auto-restart requests from the health watchdog.
    pub restart_rx: mpsc::UnboundedReceiver<crate::server::RestartRequest>,
    /// Health probe registry for endpoint liveness (shown in /status).
    pub health_registry: Option<Arc<crate::heartbeat::health::HealthRegistry>>,
    #[cfg(feature = "voice")]
    pub voice_session: Option<crate::voice_pipeline::VoiceSession>,
    /// Cluster state for /cluster command (peer discovery, model listing).
    #[cfg(feature = "cluster")]
    pub cluster_state: Option<Arc<crate::cluster::state::ClusterState>>,
}

/// Whether the configured policy authorizes the watchdog to recreate the
/// currently selected backend. Commands and background repair share this
/// `localAutostart` authority; `localBackend` is runtime identity only.
fn autonomous_restart_allowed(config: &Config) -> bool {
    let is_higgs = crate::config::schema::is_higgs_backend(&config.agents.defaults.local_backend);
    matches!(
        (config.agents.defaults.local_autostart, is_higgs),
        (crate::config::schema::LocalAutostart::Higgs, true)
            | (crate::config::schema::LocalAutostart::Lmstudio, false)
    )
}

// ============================================================================
// Unified model picker types
// ============================================================================

/// Where a model comes from in the unified picker.
#[derive(Debug, Clone)]
enum ModelSource {
    /// Local LM Studio managed instance.
    LocalLms { port: u16 },
    /// Remote server (cluster peer or manual `local_api_base`).
    Remote {
        endpoint: String,
        #[cfg(feature = "cluster")]
        peer_type: crate::cluster::state::PeerType,
        #[cfg(not(feature = "cluster"))]
        peer_type: (),
    },
    /// Filesystem GGUF file.
    File { path: PathBuf },
    /// Higgs server. `path` is present for runtime switchable models.
    Higgs {
        endpoint: String,
        path: Option<String>,
        name: String,
    },
}

/// A model entry from any source, used by the unified model picker.
#[derive(Debug, Clone)]
pub(crate) struct ModelEntry {
    /// Model identifier (name or path stem).
    pub(crate) id: String,
    /// Where it comes from.
    source: ModelSource,
    /// Currently selected model.
    pub(crate) is_active: bool,
    /// Currently loaded in memory (LMS only).
    pub(crate) is_loaded: bool,
}

impl ModelEntry {
    #[cfg(test)]
    pub(crate) fn test_local(id: &str) -> Self {
        Self {
            id: id.to_string(),
            source: ModelSource::LocalLms { port: 1234 },
            is_active: false,
            is_loaded: false,
        }
    }

    #[cfg(test)]
    pub(crate) fn test_higgs(id: &str, name: &str) -> Self {
        Self {
            id: id.to_string(),
            source: ModelSource::Higgs {
                endpoint: "http://127.0.0.1:9000/v1".to_string(),
                path: Some(id.to_string()),
                name: name.to_string(),
            },
            is_active: false,
            is_loaded: false,
        }
    }

    /// Human-facing model label; Higgs ids can be path-derived while `name`
    /// is the runtime model name users expect to see.
    pub(crate) fn display_name(&self) -> &str {
        match &self.source {
            ModelSource::Higgs { name, .. } if !name.is_empty() => name,
            _ => &self.id,
        }
    }

    /// Short, human label for the model's source (for the TUI picker).
    pub(crate) fn source_tag(&self) -> String {
        match &self.source {
            ModelSource::LocalLms { port } => format!("LM Studio :{port}"),
            ModelSource::Remote { endpoint, .. } => crate::tui::shorten_url(endpoint)
                .split('/')
                .next()
                .unwrap_or(endpoint)
                .to_string(),
            ModelSource::File { .. } => "file".to_string(),
            ModelSource::Higgs { endpoint, .. } => {
                let short = crate::tui::shorten_url(endpoint);
                let host = short.split('/').next().unwrap_or(&short);
                format!("Higgs {host}")
            }
        }
    }
}

fn model_short_id(id: &str) -> &str {
    id.rsplit('/').next().unwrap_or(id)
}

fn normalized_endpoint(endpoint: &str) -> String {
    endpoint.trim().trim_end_matches('/').to_ascii_lowercase()
}

fn endpoint_is_loopback(endpoint: &str) -> bool {
    let lower = endpoint.trim().to_ascii_lowercase();
    let host_port = lower
        .strip_prefix("http://")
        .or_else(|| lower.strip_prefix("https://"))
        .unwrap_or(&lower);
    host_port.starts_with("localhost:")
        || host_port.starts_with("127.0.0.1:")
        || host_port.starts_with("[::1]:")
}

fn push_unique_endpoint(bases: &mut Vec<String>, endpoint: String) {
    if endpoint.trim().is_empty() {
        return;
    }
    let normalized = normalized_endpoint(&endpoint);
    if bases
        .iter()
        .any(|base| normalized_endpoint(base) == normalized)
    {
        return;
    }
    bases.push(endpoint.trim().to_string());
}

fn higgs_model_bases(
    current_base: &str,
    configured_higgs_base: &str,
    backend_is_higgs: bool,
) -> Vec<String> {
    let mut bases = Vec::new();
    let current_base = current_base.trim();
    let current_norm = normalized_endpoint(current_base);
    let configured_norm = normalized_endpoint(configured_higgs_base);

    if backend_is_higgs {
        if current_base.is_empty() {
            push_unique_endpoint(&mut bases, configured_higgs_base.to_string());
        } else {
            push_unique_endpoint(&mut bases, current_base.to_string());
            if endpoint_is_loopback(current_base) && current_norm != configured_norm {
                push_unique_endpoint(&mut bases, configured_higgs_base.to_string());
            }
        }
    } else if !current_base.is_empty() && current_norm == configured_norm {
        push_unique_endpoint(&mut bases, current_base.to_string());
    }

    bases
}

fn higgs_model_entries(
    endpoint: &str,
    active_hint: &str,
    catalog: Option<crate::higgs::AvailableModelCatalog>,
    resident: Vec<String>,
) -> Vec<ModelEntry> {
    let Some(catalog) = catalog.filter(|catalog| catalog.runtime_model_load) else {
        return resident
            .into_iter()
            .filter(|id| !id.is_empty() && !id.to_lowercase().contains("embedding"))
            .map(|id| ModelEntry {
                is_active: crate::lms::is_model_available(std::slice::from_ref(&id), active_hint),
                source: ModelSource::Higgs {
                    endpoint: endpoint.to_string(),
                    path: None,
                    name: id.clone(),
                },
                id,
                is_loaded: true,
            })
            .collect();
    };

    catalog
        .models
        .into_iter()
        .map(|model| {
            let is_active = crate::lms::is_model_available(
                &[model.id.clone(), model.stable_id.clone()],
                active_hint,
            );
            ModelEntry {
                id: model.id.clone(),
                source: ModelSource::Higgs {
                    endpoint: endpoint.to_string(),
                    path: Some(model.path),
                    name: model.id,
                },
                is_active,
                is_loaded: model.loaded,
            }
        })
        .collect()
}

fn extend_unique_higgs_entries(
    entries: &mut Vec<ModelEntry>,
    seen_paths: &mut HashSet<String>,
    candidates: Vec<ModelEntry>,
) {
    entries.extend(candidates.into_iter().filter(|entry| match &entry.source {
        ModelSource::Higgs {
            path: Some(path), ..
        } => seen_paths.insert(path.clone()),
        _ => true,
    }));
}

fn model_direct_match_rank(entry: &ModelEntry, query: &str) -> Option<usize> {
    let query = query.trim().to_ascii_lowercase();
    if query.is_empty() {
        return None;
    }

    let id = entry.id.to_ascii_lowercase();
    let short = model_short_id(&entry.id).to_ascii_lowercase();

    if id == query {
        Some(0)
    } else if short == query {
        Some(1)
    } else if id.starts_with(&query) {
        Some(2)
    } else if short.starts_with(&query) {
        Some(3)
    } else if id.contains(&query) {
        Some(4)
    } else if short.contains(&query) {
        Some(5)
    } else {
        None
    }
}

pub(crate) fn unique_direct_model_match<'a>(
    entries: &[&'a ModelEntry],
    query: &str,
) -> Option<&'a ModelEntry> {
    let mut best_rank = usize::MAX;
    let mut best = None;
    let mut ties = 0usize;

    for entry in entries {
        let Some(rank) = model_direct_match_rank(entry, query) else {
            continue;
        };
        if rank < best_rank {
            best_rank = rank;
            best = Some(*entry);
            ties = 1;
        } else if rank == best_rank {
            ties += 1;
        }
    }

    if ties == 1 {
        best
    } else {
        None
    }
}

// ============================================================================
// DRY helpers — replace 3–10x copy-paste patterns
// ============================================================================

impl ReplContext {
    /// Rebuild the agent loop after a server or config change.
    pub fn rebuild_agent_loop(&mut self) {
        self.agent_loop = cli::create_agent_loop(
            self.core_handle.clone(),
            &self.config,
            Some(self.cron_service.clone()),
            self.email_config.clone(),
            Some(self.display_tx.clone()),
            self.health_registry.clone(),
        );
    }

    /// (Re)start the health watchdog for all active local servers.
    ///
    /// Aborts any previous watchdog task, collects current server ports, and
    /// spawns a fresh watchdog with auto-repair. Called on REPL init and on `/local` toggle-on.
    /// No-op when using a remote local server — nothing to watch.
    /// A Higgs endpoint is monitored only when `localAutostart: "higgs"`
    /// authorizes background repair.
    pub fn restart_watchdog(&mut self) {
        if let Some(handle) = self.watchdog_handle.take() {
            handle.abort();
        }
        if !autonomous_restart_allowed(&self.config) {
            return;
        }
        // Remote server: nothing to watch locally — unless it's Higgs (managed sidecar).
        let is_higgs =
            crate::config::schema::is_higgs_backend(&self.config.agents.defaults.local_backend);
        if !is_higgs && !self.config.agents.defaults.local_api_base.is_empty() {
            return;
        }
        // Discovery is runtime truth. A manually started Higgs may answer on a
        // port different from the stale `higgsPort` hint.
        let port = self.srv.local_port.clone();
        let ports = vec![("main".to_string(), port)];
        self.watchdog_handle = Some(crate::server::start_health_watchdog_with_autorepair(
            ports,
            self.display_tx.clone(),
            self.restart_tx.clone(),
            Arc::clone(&self.core_handle.counters.inference_active),
            self.config.monitoring.health_poll_interval_secs,
            self.config.monitoring.degraded_threshold,
            self.config.monitoring.health_check_timeout_secs,
        ));
    }

    /// Stop the health watchdog (e.g. when switching to cloud mode).
    pub fn stop_watchdog(&mut self) {
        if let Some(handle) = self.watchdog_handle.take() {
            handle.abort();
        }
    }

    /// Check for and handle any pending auto-restart requests from the watchdog.
    ///
    /// Returns true if a restart was performed.
    pub async fn handle_restart_requests(&mut self) -> bool {
        if !autonomous_restart_allowed(&self.config) {
            while self.restart_rx.try_recv().is_ok() {}
            return false;
        }
        // When using a remote local server, there are no local server processes
        // to restart — drain and ignore any stale requests.
        // A Higgs endpoint can be restarted only under explicit Higgs
        // autostart authority (checked above).
        let is_higgs =
            crate::config::schema::is_higgs_backend(&self.config.agents.defaults.local_backend);
        if !is_higgs && !self.config.agents.defaults.local_api_base.is_empty() {
            while self.restart_rx.try_recv().is_ok() {}
            return false;
        }
        let mut restarted = false;
        while let Ok(req) = self.restart_rx.try_recv() {
            if req.role == "main" {
                if server::check_local_health(&self.srv.local_port).await {
                    continue;
                }
                let _ = self.display_tx.send(format!(
                    "\x1b[RAW]\n  \x1b[33m\u{25cf}\x1b[0m Auto-restarting main server...\n"
                ));
                self.cmd_restart().await;

                // Wait for server to be ready
                for i in 0..10 {
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    if server::check_local_health(&self.srv.local_port).await {
                        let _ = self.display_tx.send(format!(
                            "\x1b[RAW]\n  \x1b[32m\u{25cf}\x1b[0m Main server \x1b[32mready\x1b[0m\n"
                        ));
                        break;
                    }
                    if i == 9 {
                        let _ = self.display_tx.send(format!(
                            "\x1b[RAW]\n  \x1b[31m\u{25cf}\x1b[0m Main server failed to start\n"
                        ));
                    }
                }
                restarted = true;
            }
        }
        restarted
    }

    /// Drain pending display messages from background channels/subagents.
    ///
    /// Replaces the 3x copy-pasted `while let Ok(line) = display_rx.try_recv()` pattern.
    pub fn drain_display(&mut self) {
        while let Ok(line) = self.display_rx.try_recv() {
            if line.starts_with("\x1b[RAW]") {
                print!("\r{}", &line[6..]);
            } else {
                print!("\r{}", crate::syntax::render_response(&line));
            }
        }
    }

    /// Async readline that drains display messages while waiting for input.
    ///
    /// Uses rustyline's `ExternalPrinter` to safely output subagent results
    /// while `readline()` blocks on a background thread.  This way results
    /// appear immediately instead of waiting until the next user input.
    pub async fn readline_async(&mut self, prompt: &str) -> Result<String, ReadlineError> {
        // Take the editor out of the Option — no placeholder DefaultEditor created,
        // so no second SIGWINCH handler is registered.
        let mut rl = self.rl.take().expect("editor already borrowed");
        let printer_result = rl.create_external_printer();

        let prompt_owned = prompt.to_string();
        let handle = tokio::task::spawn_blocking(move || {
            let result = rl.readline(&prompt_owned);
            (rl, result)
        });
        tokio::pin!(handle);

        let result = if let Ok(mut printer) = printer_result {
            loop {
                tokio::select! {
                    res = &mut handle => {
                        let (rl_back, readline_res) = res.expect("readline task panicked");
                        self.rl = Some(rl_back);
                        break readline_res;
                    }
                    Some(line) = self.display_rx.recv() => {
                        let rendered = if line.starts_with("\x1b[RAW]") {
                            line[6..].to_string()
                        } else {
                            crate::syntax::render_response(&line)
                        };
                        let _ = printer.print(rendered);
                    }
                }
            }
        } else {
            // No external printer available — just wait for readline.
            let (rl_back, readline_res) = handle.await.expect("readline task panicked");
            self.rl = Some(rl_back);
            readline_res
        };

        result
    }

    /// Print the status bar after a response completes.
    ///
    /// Replaces the 3x copy-pasted block that gets subagent count, retains
    /// finished channels, collects channel names, and calls `tui::print_status_bar`.
    #[cfg_attr(not(feature = "voice"), allow(dead_code))]
    pub async fn print_status_bar(&mut self) {
        let sa_count = self.agent_loop.subagent_manager().get_running_count().await;
        self.active_channels.retain(|ch| !ch.handle.is_finished());
        let ch_names: Vec<&str> = self
            .active_channels
            .iter()
            .map(|c| super::short_channel_name(&c.name))
            .collect();
        tui::print_status_bar(&self.core_handle, &ch_names, sa_count);
    }

    /// Rebuild the shared core from current ServerState and then rebuild the agent loop.
    ///
    /// Combines `apply_server_change` + `rebuild_agent_loop` into one call.
    /// Reads `is_local` from the current core to pass through, unless the caller
    /// is about to change modes — in which case use `apply_and_rebuild_with`.
    pub fn apply_and_rebuild(&mut self) {
        // migrated from swappable().is_local — phase 09-03
        let is_local = self.core_handle.swappable().mode().is_local();
        self.apply_and_rebuild_with(is_local);
    }

    /// Like `apply_and_rebuild` but with an explicit `is_local` override.
    /// Use when toggling between local and cloud mode.
    pub fn apply_and_rebuild_with(&mut self, is_local: bool) {
        super::apply_server_change(
            &self.srv,
            &self.current_model_path,
            &self.core_handle,
            &self.config,
            is_local,
        );
        self.rebuild_agent_loop();
    }

    /// Extract port from an endpoint URL like "http://192.168.1.50:1234/v1".
    fn extract_endpoint_port(endpoint: &str) -> Option<u16> {
        endpoint
            .trim_start_matches("http://")
            .trim_start_matches("https://")
            .split(':')
            .nth(1)
            .and_then(|p| p.split('/').next())
            .and_then(|p| p.parse::<u16>().ok())
    }

    /// Check if the current `local_api_base` points at a remote LM Studio peer.
    #[cfg(feature = "cluster")]
    async fn is_remote_lms_peer(&self) -> bool {
        use crate::cluster::state::PeerType;
        let base = &self.config.agents.defaults.local_api_base;
        if base.is_empty() {
            return false;
        }
        // Check cluster state for known peer type
        if let Some(ref cs) = self.cluster_state {
            let peers = cs.get_all_peers().await;
            for peer in &peers {
                if peer.endpoint == *base && peer.peer_type == PeerType::LMStudio {
                    return true;
                }
            }
        }
        // Fallback: port 1234 is conventionally LM Studio
        Self::extract_endpoint_port(base) == Some(1234)
    }

    /// Aggregate models from all available sources into a unified list.
    /// Whether the model picker applies (local mode, or a cluster is present).
    pub(crate) fn model_picker_available(&self) -> bool {
        #[cfg(feature = "cluster")]
        let has_cluster = self.cluster_state.is_some();
        #[cfg(not(feature = "cluster"))]
        let has_cluster = false;
        self.core_handle.swappable().mode().is_local() || has_cluster
    }

    /// Toggle voice mode without printing or switching screens (for the TUI).
    /// Returns the new state (`true` = on). Mirrors `cmd_voice` sans output.
    #[cfg(feature = "voice")]
    pub(crate) async fn toggle_voice(&mut self) -> bool {
        use std::sync::atomic::Ordering;
        if self.voice_session.is_some() {
            if let Some(ref mut vs) = self.voice_session {
                vs.stop_playback();
            }
            self.voice_session = None;
            self.core_handle
                .counters
                .suppress_thinking_in_tts
                .store(false, Ordering::Relaxed);
            return false;
        }
        let mut voice_config = self.config.voice.clone();
        if voice_config.language.is_none() {
            voice_config.language = self.lang.clone();
        }
        match crate::voice_pipeline::VoiceSession::with_voice_config(&voice_config).await {
            Ok(vs) => {
                self.voice_session = Some(vs);
                self.core_handle
                    .counters
                    .suppress_thinking_in_tts
                    .store(true, Ordering::Relaxed);
                true
            }
            Err(_) => false,
        }
    }

    pub(crate) async fn collect_all_models(&self) -> Vec<ModelEntry> {
        self.collect_all_models_with_local_files(None).await
    }

    async fn collect_all_models_with_local_files(
        &self,
        local_files: Option<&[PathBuf]>,
    ) -> Vec<ModelEntry> {
        let mut entries = Vec::new();
        let current_model = if !self.config.agents.defaults.lms_main_model.is_empty() {
            self.config.agents.defaults.lms_main_model.clone()
        } else {
            self.config.agents.defaults.local_model.clone()
        };
        let current_base = &self.config.agents.defaults.local_api_base;
        let backend_is_higgs =
            crate::config::schema::is_higgs_backend(&self.config.agents.defaults.local_backend);
        let configured_higgs_base = format!(
            "http://127.0.0.1:{}/v1",
            self.config.agents.defaults.higgs_port
        );
        let higgs_bases = higgs_model_bases(current_base, &configured_higgs_base, backend_is_higgs);
        // Older configs may still say "omlx" after pointing localApiBase at the
        // managed Higgs port. Treat that exact localhost endpoint as Higgs so
        // the picker can show runtime-switchable model directories. For sticky
        // Higgs configs that point at another loopback service, query that
        // resident endpoint too, but only add Higgs switch candidates if the
        // endpoint proves it supports /v1/models/switch.
        let use_higgs_model_discovery = !higgs_bases.is_empty();

        // 1. Local LMS (if lms_managed)
        let mut covered_endpoint: Option<String> = None;
        if self.srv.lms_managed {
            let lms_port = self.config.agents.defaults.lms_port;
            let available = crate::lms::list_available("", lms_port).await;
            let loaded = crate::lms::list_loaded("", lms_port).await;
            covered_endpoint = Some(format!("http://{}:{}/v1", crate::lms::api_host(), lms_port));
            for name in &available {
                if name.to_lowercase().contains("embedding") {
                    continue;
                }
                let is_loaded = loaded
                    .iter()
                    .any(|l| l.contains(name.as_str()) || name.contains(l.as_str()));
                let is_active =
                    crate::lms::is_model_available(std::slice::from_ref(name), &current_model);
                entries.push(ModelEntry {
                    id: name.clone(),
                    source: ModelSource::LocalLms { port: lms_port },
                    is_active,
                    is_loaded,
                });
            }
        }

        // 2. Cluster peers (if cluster feature and state exists)
        #[cfg(feature = "cluster")]
        if let Some(ref cs) = self.cluster_state {
            let peers = cs.get_healthy_peers().await;
            for peer in &peers {
                // Skip if this peer's endpoint was already covered by local LMS
                if let Some(ref covered) = covered_endpoint {
                    if peer.endpoint == *covered {
                        continue;
                    }
                }
                // The local Higgs sidecar may also appear as a healthy cluster
                // peer. Let the Higgs-specific branch own it so it can include
                // filesystem candidates that are switchable via /v1/models/switch.
                if use_higgs_model_discovery
                    && higgs_bases.iter().any(|base| {
                        normalized_endpoint(&peer.endpoint) == normalized_endpoint(base)
                    })
                {
                    continue;
                }
                // Skip if endpoint matches current local_api_base AND we already
                // added models from it via the remote-server branch in step 1
                for model in &peer.models {
                    let is_active = peer.endpoint == *current_base
                        && crate::lms::is_model_available(&[model.id.clone()], &current_model);
                    entries.push(ModelEntry {
                        id: model.id.clone(),
                        source: ModelSource::Remote {
                            endpoint: peer.endpoint.clone(),
                            peer_type: peer.peer_type.clone(),
                        },
                        is_active,
                        is_loaded: false,
                    });
                }
            }
        }

        // 2.5. Configured local_api_base remote endpoint (when not covered above)
        //
        // When the user points `local_api_base` at a remote server (e.g. LM Studio
        // on another machine), the cluster peer list may not include it.
        {
            let is_higgs = use_higgs_model_discovery;
            let base = current_base.trim().to_string();
            let already_covered = covered_endpoint
                .as_deref()
                .map(|c| normalized_endpoint(c) == normalized_endpoint(&base))
                .unwrap_or(false);

            #[cfg(feature = "cluster")]
            let covered_by_cluster = if !already_covered {
                if let Some(ref cs) = self.cluster_state {
                    let peers = cs.get_healthy_peers().await;
                    peers
                        .iter()
                        .any(|p| normalized_endpoint(&p.endpoint) == normalized_endpoint(&base))
                } else {
                    false
                }
            } else {
                false
            };
            #[cfg(not(feature = "cluster"))]
            let covered_by_cluster = false;

            if is_higgs {
                let api_key = &self.config.agents.defaults.local_api_key;
                let active_hint = if current_model == "active" {
                    self.config.agents.defaults.local_model.as_str()
                } else {
                    current_model.as_str()
                };
                let mut seen_higgs_paths = HashSet::new();
                for base in &higgs_bases {
                    let already_covered = covered_endpoint
                        .as_deref()
                        .map(|c| normalized_endpoint(c) == normalized_endpoint(base))
                        .unwrap_or(false);
                    if base.is_empty() || already_covered {
                        continue;
                    }
                    let catalog = crate::higgs::available_model_catalog_at(base, api_key).await;
                    let resident =
                        crate::higgs::list_available_served_models_at(base, api_key).await;
                    extend_unique_higgs_entries(
                        &mut entries,
                        &mut seen_higgs_paths,
                        higgs_model_entries(base, active_hint, catalog, resident),
                    );
                }
            } else if !base.is_empty() && !already_covered && !covered_by_cluster {
                let api_key = &self.config.agents.defaults.local_api_key;
                let models_url = {
                    let b = base.trim_end_matches('/');
                    if b.ends_with("/v1") {
                        format!("{}/models", b)
                    } else {
                        format!("{}/v1/models", b)
                    }
                };

                let client = reqwest::Client::new();
                if let Ok(resp) = client
                    .get(&models_url)
                    .header("Authorization", format!("Bearer {}", api_key))
                    .timeout(Duration::from_secs(3))
                    .send()
                    .await
                {
                    if let Ok(json) = resp.json::<serde_json::Value>().await {
                        if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
                            for item in data {
                                let id = item
                                    .get("id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                if id.is_empty() || id.to_lowercase().contains("embedding") {
                                    continue;
                                }
                                let is_active =
                                    crate::lms::is_model_available(&[id.clone()], &current_model);
                                let source = ModelSource::Remote {
                                    endpoint: base.clone(),
                                    #[cfg(feature = "cluster")]
                                    peer_type: crate::cluster::state::PeerType::Unknown,
                                    #[cfg(not(feature = "cluster"))]
                                    peer_type: (),
                                };
                                entries.push(ModelEntry {
                                    id,
                                    source,
                                    is_active,
                                    is_loaded: false,
                                });
                            }
                        }
                    }
                }
            }
        }

        // 3. Filesystem GGUF fallback for non-Higgs local modes only.
        if !self.srv.lms_managed && entries.is_empty() && !use_higgs_model_discovery {
            let discovered;
            let models = if let Some(files) = local_files {
                files
            } else {
                discovered = crate::server::list_local_models();
                &discovered
            };
            for path in models {
                let name = path.file_name().unwrap().to_string_lossy().to_string();
                let is_active = *path == self.current_model_path;
                entries.push(ModelEntry {
                    id: name,
                    source: ModelSource::File { path: path.clone() },
                    is_active,
                    is_loaded: false,
                });
            }
        }

        // 4. MLX models (safetensors dirs, served via managed mlx-lm server)

        entries
    }
}

// ============================================================================
// Alias resolution
// ============================================================================

/// Normalize command aliases to their canonical form.
pub(crate) fn normalize_alias(cmd: &str) -> &str {
    match cmd {
        "/l" => "/local",
        "/m" => "/model",
        "/t" | "/thinking" => "/think",
        "/nt" => "/nothink",
        "/v" => "/voice",
        "/ss" => "/sessions",
        "/wa" => "/whatsapp",
        "/tg" => "/telegram",
        "/p" | "/prov" => "/provenance",
        "/h" | "/?" => "/help",
        "/a" => "/agents",
        "/s" => "/status",
        "/rd" => "/restart",
        "/ctx-info" => "/context",
        "/c" => "/clear",
        "/cl" => "/cluster",
        "/sk" => "/skill",
        other => other,
    }
}

// ============================================================================
// Dispatch — routes slash commands to handlers
// ============================================================================

impl ReplContext {
    /// Dispatch a slash command. Returns `true` if the input was a recognized command.
    pub async fn dispatch(&mut self, input: &str) -> bool {
        // Reset scroll region and ensure cooked terminal mode before printing
        // command output.  render_input_bar() sets a scroll region and various
        // code paths may leave the terminal in raw mode — without this guard
        // every println!() inside a command handler would staircase (\n without
        // \r) or be constrained to the scroll region.
        crate::tui::reset_scroll_region();
        crate::tui::clear_input_bar();
        crate::tui::force_exit_raw_mode();
        crate::tui::ensure_cooked_output();

        let (cmd, arg) = input
            .split_once(' ')
            .map(|(c, a)| (c, a.trim()))
            .unwrap_or((input, ""));
        let cmd = normalize_alias(cmd);
        match cmd {
            "/help" => {
                super::print_help();
            }
            "/think" => {
                self.cmd_think(arg);
            }
            "/nothink" => {
                self.cmd_nothink();
            }
            "/status" => {
                self.cmd_status().await;
            }
            "/context" => {
                self.cmd_context().await;
            }
            "/compact" => {
                self.cmd_compact().await;
            }
            "/memory" => {
                self.cmd_memory().await;
            }
            "/learn" | "/reflect" => {
                self.cmd_learn().await;
            }
            "/agents" => {
                self.cmd_agents().await;
            }
            "/audit" => {
                self.cmd_audit();
            }
            "/verify" => {
                self.cmd_verify().await;
            }
            "/kill" => {
                self.cmd_kill(arg).await;
            }
            "/stop" => {
                self.cmd_stop().await;
            }

            "/replay" => {
                self.cmd_replay(arg).await;
            }
            "/long" => {
                self.cmd_long(arg);
            }
            "/provenance" => {
                self.cmd_provenance();
            }
            "/restart" => {
                self.cmd_restart().await;
            }
            "/ctx" => {
                self.cmd_ctx(arg).await;
            }
            "/model" => {
                self.cmd_model(arg).await;
            }
            "/local" => {
                self.cmd_local().await;
            }
            "/whatsapp" => {
                self.cmd_whatsapp();
            }
            "/telegram" => {
                self.cmd_telegram();
            }
            "/email" => {
                self.cmd_email();
            }
            "/lane" => {
                self.cmd_lane(arg);
            }
            "/sessions" => {
                self.cmd_sessions(arg).await;
            }
            "/clear" => {
                self.cmd_clear().await;
            }
            #[cfg(feature = "voice")]
            "/voice" => {
                self.cmd_voice().await;
            }
            "/cluster" => {
                self.cmd_cluster(arg).await;
            }
            "/lcm" => {
                self.cmd_lcm(arg);
            }
            "/skill" | "/skills" => {
                self.cmd_skill(arg).await;
            }
            _ => {
                return false;
            }
        }
        true
    }
}

// ============================================================================
// Utility
// ============================================================================

impl ReplContext {
    /// Whether voice mode is currently active.
    pub fn voice_on(&self) -> bool {
        #[cfg(feature = "voice")]
        {
            self.voice_session.is_some()
        }
        #[cfg(not(feature = "voice"))]
        {
            false
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    async fn spawn_model_catalog_server(
        catalog: Option<serde_json::Value>,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                let mut request = [0_u8; 4096];
                let Ok(read) = stream.read(&mut request).await else {
                    continue;
                };
                let request = String::from_utf8_lossy(&request[..read]);
                let path = request
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or("");
                let (status, body) = match path {
                    "/v1/models/available" => catalog
                        .as_ref()
                        .map(|value| (200, value.to_string()))
                        .unwrap_or_else(|| (404, "{}".to_string())),
                    "/v1/models" => (200, r#"{"data":[]}"#.to_string()),
                    "/health" => (200, r#"{"models":[]}"#.to_string()),
                    _ => (404, "{}".to_string()),
                };
                let reason = if status == 200 { "OK" } else { "Not Found" };
                let response = format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
            }
        });
        (format!("http://{address}/v1"), task)
    }

    fn test_repl_context(config: Config) -> (ReplContext, tempfile::TempDir) {
        let temp = tempfile::tempdir().unwrap();
        let core_handle = crate::cli::build_core_handle(&config, "0", None, false);
        let agent_loop =
            crate::cli::create_agent_loop(core_handle.clone(), &config, None, None, None, None);
        let (display_tx, display_rx) = mpsc::unbounded_channel();
        let (restart_tx, restart_rx) = mpsc::unbounded_channel();
        let context = ReplContext {
            config,
            core_handle,
            agent_loop,
            session_id: "model-catalog-test".to_string(),
            lang: None,
            srv: super::super::ServerState::new("0".to_string()),
            current_model_path: PathBuf::new(),
            active_channels: Vec::new(),
            display_tx,
            display_rx,
            cron_service: Arc::new(CronService::new(temp.path().join("cron.json"))),
            email_config: None,
            rl: None,
            watchdog_handle: None,
            restart_tx,
            restart_rx,
            health_registry: None,
            #[cfg(feature = "voice")]
            voice_session: None,
            #[cfg(feature = "cluster")]
            cluster_state: None,
        };
        (context, temp)
    }

    #[test]
    fn test_normalize_alias_all_aliases() {
        assert_eq!(normalize_alias("/l"), "/local");
        assert_eq!(normalize_alias("/m"), "/model");
        assert_eq!(normalize_alias("/t"), "/think");
        assert_eq!(normalize_alias("/thinking"), "/think");
        assert_eq!(normalize_alias("/nt"), "/nothink");
        assert_eq!(normalize_alias("/v"), "/voice");
        assert_eq!(normalize_alias("/wa"), "/whatsapp");
        assert_eq!(normalize_alias("/tg"), "/telegram");
        assert_eq!(normalize_alias("/prov"), "/provenance");
        assert_eq!(normalize_alias("/h"), "/help");
        assert_eq!(normalize_alias("/?"), "/help");
        assert_eq!(normalize_alias("/a"), "/agents");
        assert_eq!(normalize_alias("/s"), "/status");
        assert_eq!(normalize_alias("/rd"), "/restart");
        assert_eq!(normalize_alias("/ctx-info"), "/context");
        assert_eq!(normalize_alias("/ss"), "/sessions");
        assert_eq!(normalize_alias("/c"), "/clear");
    }

    #[test]
    fn test_normalize_alias_passthrough() {
        assert_eq!(normalize_alias("/status"), "/status");
        assert_eq!(normalize_alias("/help"), "/help");
        assert_eq!(normalize_alias("/local"), "/local");
        assert_eq!(normalize_alias("/unknown"), "/unknown");
        assert_eq!(normalize_alias("hello"), "hello");
    }

    #[test]
    fn test_watchdog_autorestart_requires_matching_autostart_authority() {
        let mut config = Config::default();
        config.agents.defaults.local_backend = "higgs".to_string();

        config.agents.defaults.local_autostart = crate::config::schema::LocalAutostart::Off;
        assert!(!autonomous_restart_allowed(&config));

        config.agents.defaults.local_autostart = crate::config::schema::LocalAutostart::Lmstudio;
        assert!(!autonomous_restart_allowed(&config));

        config.agents.defaults.local_autostart = crate::config::schema::LocalAutostart::Higgs;
        assert!(autonomous_restart_allowed(&config));

        config.agents.defaults.local_backend = "lmstudio".to_string();
        config.agents.defaults.local_autostart = crate::config::schema::LocalAutostart::Off;
        assert!(!autonomous_restart_allowed(&config));

        config.agents.defaults.local_autostart = crate::config::schema::LocalAutostart::Higgs;
        assert!(!autonomous_restart_allowed(&config));

        config.agents.defaults.local_autostart = crate::config::schema::LocalAutostart::Lmstudio;
        assert!(autonomous_restart_allowed(&config));
    }

    fn test_model_entry(id: &str) -> ModelEntry {
        ModelEntry {
            id: id.to_string(),
            source: ModelSource::LocalLms { port: 1234 },
            is_active: false,
            is_loaded: false,
        }
    }

    #[test]
    fn test_unique_direct_model_match_short_name() {
        let qwen = test_model_entry("mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit");
        let nemotron = test_model_entry("nvidia/Nemotron-Nano-12B");
        let entries = vec![&qwen, &nemotron];

        let selected = unique_direct_model_match(&entries, "qwen3-coder").unwrap();

        assert_eq!(selected.id, qwen.id);
    }

    #[test]
    fn test_unique_direct_model_match_refuses_ambiguous_prefix() {
        let coder = test_model_entry("mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit");
        let chat = test_model_entry("mlx-community/Qwen3-14B-Instruct-4bit");
        let entries = vec![&coder, &chat];

        assert!(unique_direct_model_match(&entries, "qwen3").is_none());
    }

    #[test]
    fn test_unique_direct_model_match_prefers_exact_over_contains() {
        let exact = test_model_entry("Qwen3");
        let longer = test_model_entry("mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit");
        let entries = vec![&longer, &exact];

        let selected = unique_direct_model_match(&entries, "qwen3").unwrap();

        assert_eq!(selected.id, exact.id);
    }

    #[test]
    fn test_higgs_model_bases_adds_configured_sidecar_for_sticky_loopback() {
        let bases = higgs_model_bases("http://127.0.0.1:1976/v1", "http://127.0.0.1:8000/v1", true);

        assert_eq!(
            bases,
            vec![
                "http://127.0.0.1:1976/v1".to_string(),
                "http://127.0.0.1:8000/v1".to_string()
            ]
        );
    }

    #[test]
    fn test_higgs_model_bases_respects_remote_higgs_endpoint() {
        let bases = higgs_model_bases(
            "http://192.168.1.22:8000/v1",
            "http://127.0.0.1:8000/v1",
            true,
        );

        assert_eq!(bases, vec!["http://192.168.1.22:8000/v1".to_string()]);
    }

    #[test]
    fn test_higgs_model_bases_accepts_exact_configured_endpoint_for_old_backend_tag() {
        let bases = higgs_model_bases(
            "http://127.0.0.1:8000/v1",
            "http://127.0.0.1:8000/v1",
            false,
        );

        assert_eq!(bases, vec!["http://127.0.0.1:8000/v1".to_string()]);
    }

    #[test]
    fn higgs_catalog_entries_use_server_names_paths_and_loaded_state() {
        let catalog = crate::higgs::AvailableModelCatalog {
            runtime_model_load: true,
            models: vec![
                crate::higgs::AvailableRuntimeModel {
                    id: "ternary-bonsai2-27b-2bit".to_string(),
                    stable_id: "NexVeridian/ternary-bonsai2-27b-2bit".to_string(),
                    path: "/models/ternary".to_string(),
                    model_type: "qwen3".to_string(),
                    adapter: "qwen3".to_string(),
                    loaded: true,
                },
                crate::higgs::AvailableRuntimeModel {
                    id: "Nanbeige4.1-3B".to_string(),
                    stable_id: "Nanbeige/Nanbeige4.1-3B".to_string(),
                    path: "/custom-hf/nanbeige".to_string(),
                    model_type: "nanbeige".to_string(),
                    adapter: "nanbeige".to_string(),
                    loaded: false,
                },
            ],
        };

        let entries = higgs_model_entries(
            "http://127.0.0.1:9000/v1",
            "ternary-bonsai2-27b-2bit",
            Some(catalog),
            vec!["must-not-be-added-twice".to_string()],
        );

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].display_name(), "ternary-bonsai2-27b-2bit");
        assert!(entries[0].is_loaded);
        assert!(entries[0].is_active);
        assert_eq!(entries[1].display_name(), "Nanbeige4.1-3B");
        match &entries[1].source {
            ModelSource::Higgs { path, name, .. } => {
                assert_eq!(path.as_deref(), Some("/custom-hf/nanbeige"));
                assert_eq!(name, "Nanbeige4.1-3B");
            }
            source => panic!("unexpected source: {source:?}"),
        }
    }

    #[test]
    fn old_higgs_falls_back_to_resident_models_only() {
        let entries = higgs_model_entries(
            "http://127.0.0.1:9000/v1",
            "resident",
            None,
            vec!["resident".to_string()],
        );

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, "resident");
        assert!(entries[0].is_loaded);
        match &entries[0].source {
            ModelSource::Higgs { path, .. } => assert!(path.is_none()),
            source => panic!("unexpected source: {source:?}"),
        }
    }

    #[test]
    fn disabled_runtime_loading_ignores_unloaded_catalog_entries() {
        let catalog = crate::higgs::AvailableModelCatalog {
            runtime_model_load: false,
            models: vec![crate::higgs::AvailableRuntimeModel {
                id: "Nanbeige4.1-3B".to_string(),
                stable_id: "Nanbeige/Nanbeige4.1-3B".to_string(),
                path: "/custom-hf/nanbeige".to_string(),
                model_type: "nanbeige".to_string(),
                adapter: "nanbeige".to_string(),
                loaded: false,
            }],
        };

        let entries = higgs_model_entries(
            "http://127.0.0.1:9000/v1",
            "resident",
            Some(catalog),
            vec!["resident".to_string()],
        );

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, "resident");
    }

    #[tokio::test]
    async fn collect_all_models_old_higgs_never_falls_through_to_local_files() {
        let (base, server) = spawn_model_catalog_server(None).await;
        let mut config = Config::default();
        config.agents.defaults.local_backend = "higgs".to_string();
        config.agents.defaults.local_api_base = base.clone();
        config.agents.defaults.higgs_port = base
            .trim_end_matches("/v1")
            .rsplit(':')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        let (context, _temp) = test_repl_context(config);

        let entries = context
            .collect_all_models_with_local_files(Some(&[PathBuf::from("/tmp/orphan.gguf")]))
            .await;

        server.abort();
        assert!(entries.is_empty());
    }

    #[tokio::test]
    async fn collect_all_models_deduplicates_catalog_path_across_higgs_bases() {
        let canonical_path = "/models/shared-canonical";
        let catalog = |id: &str| {
            serde_json::json!({
                "runtime_model_load": true,
                "data": [{
                    "id": id,
                    "stable_id": "publisher/shared",
                    "path": canonical_path,
                    "model_type": "qwen3",
                    "adapter": "qwen3",
                    "loaded": false
                }]
            })
        };
        let (first_base, first_server) =
            spawn_model_catalog_server(Some(catalog("first-base-name"))).await;
        let (second_base, second_server) =
            spawn_model_catalog_server(Some(catalog("second-base-name"))).await;
        let mut config = Config::default();
        config.agents.defaults.local_backend = "higgs".to_string();
        config.agents.defaults.local_api_base = first_base.clone();
        config.agents.defaults.higgs_port = second_base
            .trim_end_matches("/v1")
            .rsplit(':')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        let (context, _temp) = test_repl_context(config);

        let entries = context.collect_all_models_with_local_files(Some(&[])).await;

        first_server.abort();
        second_server.abort();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, "first-base-name");
        match &entries[0].source {
            ModelSource::Higgs { endpoint, path, .. } => {
                assert_eq!(endpoint.as_str(), first_base.as_str());
                assert_eq!(path.as_deref(), Some(canonical_path));
            }
            source => panic!("unexpected source: {source:?}"),
        }
    }

    #[test]
    fn test_command_arg_parsing() {
        // Verify split_once behavior used in dispatch
        let input = "/ctx 32K";
        let (cmd, arg) = input
            .split_once(' ')
            .map(|(c, a)| (c, a.trim()))
            .unwrap_or((input, ""));
        assert_eq!(cmd, "/ctx");
        assert_eq!(arg, "32K");

        // No arg
        let input2 = "/status";
        let (cmd2, arg2) = input2
            .split_once(' ')
            .map(|(c, a)| (c, a.trim()))
            .unwrap_or((input2, ""));
        assert_eq!(cmd2, "/status");
        assert_eq!(arg2, "");
    }
}
