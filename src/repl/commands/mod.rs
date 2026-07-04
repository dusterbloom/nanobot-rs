//! Command dispatch and handlers for the REPL.
//!
//! Contains `ReplContext`, `normalize_alias()`, `dispatch()`, and all
//! `cmd_xxx()` command handlers (split across submodules).

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
mod trio_helpers;

pub(crate) use trio_helpers::*;

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
        #[allow(dead_code)]
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

fn push_remote_model_entry(
    entries: &mut Vec<ModelEntry>,
    endpoint: &str,
    id: String,
    active_hint: &str,
    is_loaded: bool,
) {
    if id.is_empty() || id.to_lowercase().contains("embedding") {
        return;
    }
    let is_active = crate::lms::is_model_available(std::slice::from_ref(&id), active_hint);
    let source = ModelSource::Remote {
        endpoint: endpoint.to_string(),
        #[cfg(feature = "cluster")]
        peer_type: crate::cluster::state::PeerType::Unknown,
        #[cfg(not(feature = "cluster"))]
        peer_type: (),
    };
    entries.push(ModelEntry {
        id,
        source,
        is_active,
        is_loaded,
    });
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
    /// Exception: Higgs is a managed sidecar that uses local_api_base but still
    /// needs watchdog monitoring.
    pub fn restart_watchdog(&mut self) {
        if let Some(handle) = self.watchdog_handle.take() {
            handle.abort();
        }
        // Remote server: nothing to watch locally — unless it's Higgs (managed sidecar).
        let is_higgs =
            crate::config::schema::is_higgs_backend(&self.config.agents.defaults.local_backend);
        if !is_higgs && !self.config.agents.defaults.local_api_base.is_empty() {
            return;
        }
        let port = if is_higgs {
            self.config.agents.defaults.higgs_port.to_string()
        } else {
            self.srv.local_port.clone()
        };
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
        // When using a remote local server, there are no local server processes
        // to restart — drain and ignore any stale requests.
        // Exception: Higgs is managed and CAN be restarted.
        let is_higgs =
            crate::config::schema::is_higgs_backend(&self.config.agents.defaults.local_backend);
        if !is_higgs && !self.config.agents.defaults.local_api_base.is_empty() {
            while self.restart_rx.try_recv().is_ok() {}
            return false;
        }
        let mut restarted = false;
        while let Ok(req) = self.restart_rx.try_recv() {
            if req.role == "main" {
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
                for base in &higgs_bases {
                    let already_covered = covered_endpoint
                        .as_deref()
                        .map(|c| normalized_endpoint(c) == normalized_endpoint(base))
                        .unwrap_or(false);
                    if base.is_empty() || already_covered {
                        continue;
                    }
                    let loaded = crate::higgs::list_available_served_models_at(base, api_key).await;
                    let switch_supported =
                        crate::higgs::supports_runtime_model_switch_at(base, api_key).await;

                    if !switch_supported {
                        for id in loaded {
                            push_remote_model_entry(&mut entries, base, id, active_hint, true);
                        }
                        continue;
                    }

                    let mut covered_loaded = HashSet::new();
                    for candidate in crate::higgs::discover_runtime_model_candidates(&self.config) {
                        let is_loaded = loaded.iter().any(|id| {
                            crate::lms::model_matches(id, &candidate.id)
                                || crate::lms::model_matches(id, &candidate.name)
                        });
                        if is_loaded {
                            covered_loaded.insert(candidate.id.clone());
                            covered_loaded.insert(candidate.name.clone());
                        }
                        let is_active = crate::lms::is_model_available(
                            &[candidate.id.clone(), candidate.name.clone()],
                            active_hint,
                        );
                        entries.push(ModelEntry {
                            id: candidate.id,
                            source: ModelSource::Higgs {
                                endpoint: base.clone(),
                                path: Some(candidate.path),
                                name: candidate.name,
                            },
                            is_active,
                            is_loaded,
                        });
                    }

                    for id in loaded {
                        if id.to_lowercase().contains("embedding")
                            || covered_loaded
                                .iter()
                                .any(|known| crate::lms::model_matches(known, &id))
                        {
                            continue;
                        }
                        let is_active = crate::lms::is_model_available(&[id.clone()], active_hint);
                        entries.push(ModelEntry {
                            id: id.clone(),
                            source: ModelSource::Higgs {
                                endpoint: base.clone(),
                                path: None,
                                name: id,
                            },
                            is_active,
                            is_loaded: true,
                        });
                    }
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

        // 3. Filesystem GGUF fallback (only if no LMS and no cluster models)
        if !self.srv.lms_managed && entries.is_empty() {
            let models = crate::server::list_local_models();
            for path in &models {
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
            "/memory" => {
                self.cmd_memory();
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
            "/trio" => {
                self.cmd_trio(arg).await;
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

    /// Compute the current VRAM budget from live config and model paths.
    fn compute_current_vram_budget(&self) -> crate::server::VramBudgetResult {
        let (vram, ram) = server::detect_available_memory();
        let available = vram.unwrap_or(ram);

        let main_profile = if self.srv.lms_managed {
            let model_name = if !self.config.agents.defaults.lms_main_model.is_empty() {
                &self.config.agents.defaults.lms_main_model
            } else {
                &self.config.agents.defaults.local_model
            };
            server::estimate_model_profile_from_name(model_name)
        } else {
            server::resolve_model_profile_from_path(
                &self.current_model_path.display().to_string(),
                &self.current_model_path,
            )
        };

        let router_profile =
            if self.config.trio.enabled && !self.config.trio.router_model.is_empty() {
                Some(server::estimate_model_profile_from_name(
                    &self.config.trio.router_model,
                ))
            } else {
                None
            };

        let specialist_profile =
            if self.config.trio.enabled && !self.config.trio.specialist_model.is_empty() {
                Some(server::estimate_model_profile_from_name(
                    &self.config.trio.specialist_model,
                ))
            } else {
                None
            };

        let input = crate::server::VramBudgetInput {
            available_memory: available,
            vram_cap: (self.config.trio.vram_cap_gb * 1e9) as u64,
            overhead_per_model: 512 * 1_000_000,
            main: main_profile,
            router: router_profile,
            specialist: specialist_profile,
        };

        server::compute_vram_budget(&input)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::DelegationMode;

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

    #[test]
    fn test_trio_passthrough_not_aliased() {
        assert_eq!(normalize_alias("/trio"), "/trio");
    }

    #[test]
    fn test_trio_subcommand_arg_parsing() {
        let input = "/trio status";
        let (cmd, arg) = input
            .split_once(' ')
            .map(|(c, a)| (c, a.trim()))
            .unwrap_or((input, ""));
        assert_eq!(cmd, "/trio");
        assert_eq!(arg, "status");

        let input2 = "/trio router";
        let (_, arg2) = input2
            .split_once(' ')
            .map(|(c, a)| (c, a.trim()))
            .unwrap_or((input2, ""));
        assert_eq!(arg2, "router");

        // No subcommand = toggle
        let input3 = "/trio";
        let (_, arg3) = input3
            .split_once(' ')
            .map(|(c, a)| (c, a.trim()))
            .unwrap_or((input3, ""));
        assert_eq!(arg3, "");
    }

    // ---- trio toggle: pure config state transitions ----

    #[test]
    fn test_trio_enable_sets_all_strict_flags() {
        use crate::config::schema::Config;

        let mut cfg = Config::default();
        trio_enable(&mut cfg);

        assert!(cfg.trio.enabled);
        assert_eq!(cfg.tool_delegation.mode, DelegationMode::Trio);
        assert!(
            cfg.tool_delegation.enabled,
            "delegation must stay on for trio"
        );
        assert!(cfg.tool_delegation.strict_no_tools_main);
        assert!(cfg.tool_delegation.strict_router_schema);
        assert!(cfg.tool_delegation.role_scoped_context_packs);
    }

    #[test]
    fn test_trio_disable_clears_strict_flags() {
        use crate::config::schema::Config;

        let mut cfg = Config::default();
        trio_enable(&mut cfg);
        trio_disable(&mut cfg);

        assert!(!cfg.trio.enabled);
        assert_eq!(cfg.tool_delegation.mode, DelegationMode::Inline);
        assert!(!cfg.tool_delegation.enabled, "inline disables delegation");
        assert!(!cfg.tool_delegation.strict_no_tools_main);
        assert!(!cfg.tool_delegation.strict_router_schema);
        assert!(!cfg.tool_delegation.role_scoped_context_packs);
    }

    #[test]
    fn test_trio_double_enable_is_idempotent() {
        use crate::config::schema::Config;

        let mut cfg = Config::default();
        trio_enable(&mut cfg);
        trio_enable(&mut cfg);

        assert!(cfg.trio.enabled);
        assert_eq!(cfg.tool_delegation.mode, DelegationMode::Trio);
        assert!(cfg.tool_delegation.strict_no_tools_main);
    }

    #[test]
    fn test_trio_double_disable_is_idempotent() {
        use crate::config::schema::Config;

        let mut cfg = Config::default();
        trio_disable(&mut cfg);

        assert!(!cfg.trio.enabled);
        assert_eq!(cfg.tool_delegation.mode, DelegationMode::Inline);
        assert!(!cfg.tool_delegation.strict_no_tools_main);
    }

    #[test]
    fn test_trio_enable_preserves_model_names() {
        use crate::config::schema::Config;

        let mut cfg = Config::default();
        cfg.trio.router_model = "qwen3-1.7b".to_string();
        cfg.trio.specialist_model = "ministral-3-8b".to_string();

        trio_enable(&mut cfg);

        assert_eq!(cfg.trio.router_model, "qwen3-1.7b");
        assert_eq!(cfg.trio.specialist_model, "ministral-3-8b");
    }

    #[test]
    fn test_trio_disable_preserves_model_names() {
        use crate::config::schema::Config;

        let mut cfg = Config::default();
        cfg.trio.router_model = "qwen3-1.7b".to_string();
        cfg.trio.specialist_model = "ministral-3-8b".to_string();
        trio_enable(&mut cfg);
        trio_disable(&mut cfg);

        // Models stay configured even when trio is off
        assert_eq!(cfg.trio.router_model, "qwen3-1.7b");
        assert_eq!(cfg.trio.specialist_model, "ministral-3-8b");
    }

    #[test]
    fn test_trio_enable_with_empty_models_still_enables() {
        use crate::config::schema::Config;

        let mut cfg = Config::default();
        assert!(cfg.trio.router_model.is_empty());
        assert!(cfg.trio.specialist_model.is_empty());

        let needs_warning = trio_enable(&mut cfg);
        assert!(cfg.trio.enabled, "should enable even with empty models");
        assert!(needs_warning, "should warn about missing models");
    }

    #[test]
    fn test_trio_enable_with_both_models_no_warning() {
        use crate::config::schema::Config;

        let mut cfg = Config::default();
        cfg.trio.router_model = "qwen3-1.7b".to_string();
        cfg.trio.specialist_model = "ministral-3-8b".to_string();

        let needs_warning = trio_enable(&mut cfg);
        assert!(!needs_warning);
    }

    // ---- trio role assignment ----

    #[test]
    fn test_set_trio_router_model() {
        use crate::config::schema::Config;

        let mut cfg = Config::default();
        set_trio_role_model_pure(&mut cfg, "router", "qwen3-1.7b");

        assert_eq!(cfg.trio.router_model, "qwen3-1.7b");
        assert!(cfg.trio.specialist_model.is_empty(), "specialist untouched");
    }

    #[test]
    fn test_set_trio_specialist_model() {
        use crate::config::schema::Config;

        let mut cfg = Config::default();
        set_trio_role_model_pure(&mut cfg, "specialist", "ministral-3-8b");

        assert_eq!(cfg.trio.specialist_model, "ministral-3-8b");
        assert!(cfg.trio.router_model.is_empty(), "router untouched");
    }

    #[test]
    fn test_set_trio_role_overwrites_previous() {
        use crate::config::schema::Config;

        let mut cfg = Config::default();
        set_trio_role_model_pure(&mut cfg, "router", "old-model");
        set_trio_role_model_pure(&mut cfg, "router", "new-model");

        assert_eq!(cfg.trio.router_model, "new-model");
    }

    // ---- persist_trio_fields: what gets copied to the disk config ----

    #[test]
    fn test_persist_trio_fields_copies_all_required() {
        use crate::config::schema::{Config, DelegationMode};

        let mut live = Config::default();
        live.trio.enabled = true;
        live.trio.router_model = "qwen3-1.7b".to_string();
        live.trio.specialist_model = "ministral-3-8b".to_string();
        live.tool_delegation.mode = DelegationMode::Trio;
        live.tool_delegation.strict_no_tools_main = true;
        live.tool_delegation.strict_router_schema = true;
        live.tool_delegation.role_scoped_context_packs = true;

        let mut disk = Config::default();
        persist_trio_fields(&live, &mut disk);

        assert!(disk.trio.enabled);
        assert_eq!(disk.trio.router_model, "qwen3-1.7b");
        assert_eq!(disk.trio.specialist_model, "ministral-3-8b");
        assert_eq!(disk.tool_delegation.mode, DelegationMode::Trio);
        assert!(disk.tool_delegation.strict_no_tools_main);
        assert!(disk.tool_delegation.strict_router_schema);
        assert!(disk.tool_delegation.role_scoped_context_packs);
    }

    #[test]
    fn test_persist_trio_fields_does_not_clobber_other_config() {
        use crate::config::schema::Config;

        let live = Config::default();
        let mut disk = Config::default();
        disk.agents.defaults.model = "custom-cloud-model".to_string();
        disk.providers.openrouter.api_key = "my-key".to_string();

        persist_trio_fields(&live, &mut disk);

        assert_eq!(
            disk.agents.defaults.model, "custom-cloud-model",
            "should not clobber model"
        );
        assert_eq!(
            disk.providers.openrouter.api_key, "my-key",
            "should not clobber api key"
        );
    }

    // ---- roundtrip: enable → disable → enable preserves full config ----

    #[test]
    fn test_trio_roundtrip_enable_disable_enable() {
        use crate::config::schema::Config;

        let mut cfg = Config::default();
        cfg.trio.router_model = "qwen3-1.7b".to_string();
        cfg.trio.specialist_model = "ministral-3-8b".to_string();

        trio_enable(&mut cfg);
        assert!(cfg.trio.enabled);
        assert!(cfg.tool_delegation.strict_no_tools_main);

        trio_disable(&mut cfg);
        assert!(!cfg.trio.enabled);
        assert!(!cfg.tool_delegation.strict_no_tools_main);

        trio_enable(&mut cfg);
        assert!(cfg.trio.enabled);
        assert!(cfg.tool_delegation.strict_no_tools_main);
        assert_eq!(
            cfg.trio.router_model, "qwen3-1.7b",
            "models survive roundtrip"
        );
    }

    // ---- apply_trio_param tests ----

    #[test]
    fn test_apply_trio_param_router_temp() {
        use crate::config::schema::Config;
        let mut config = Config::default();
        let result = apply_trio_param(&mut config, "router", "temperature", "0.3");
        assert!(result.is_ok());
        assert!((config.trio.router_temperature - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_apply_trio_param_specialist_temp() {
        use crate::config::schema::Config;
        let mut config = Config::default();
        let result = apply_trio_param(&mut config, "specialist", "temperature", "0.5");
        assert!(result.is_ok());
        assert!((config.trio.specialist_temperature - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_apply_trio_param_invalid_temp() {
        use crate::config::schema::Config;
        let mut config = Config::default();
        let result = apply_trio_param(&mut config, "router", "temperature", "abc");
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_trio_param_temp_clamps() {
        use crate::config::schema::Config;
        let mut config = Config::default();
        let result = apply_trio_param(&mut config, "router", "temperature", "5.0");
        assert!(result.is_ok());
        assert!((config.trio.router_temperature - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_apply_trio_param_router_ctx() {
        use crate::config::schema::Config;
        let mut config = Config::default();
        let result = apply_trio_param(&mut config, "router", "ctx", "8K");
        assert!(result.is_ok());
        assert_eq!(config.trio.router_ctx_tokens, 8192);
    }

    #[test]
    fn test_apply_trio_param_specialist_ctx() {
        use crate::config::schema::Config;
        let mut config = Config::default();
        let result = apply_trio_param(&mut config, "specialist", "ctx", "16384");
        assert!(result.is_ok());
        assert_eq!(config.trio.specialist_ctx_tokens, 16384);
    }

    #[test]
    fn test_apply_trio_param_vram_cap() {
        use crate::config::schema::Config;
        let mut config = Config::default();
        let result = apply_trio_param(&mut config, "trio", "vram_cap", "12");
        assert!(result.is_ok());
        assert!((config.trio.vram_cap_gb - 12.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_apply_trio_param_vram_cap_out_of_range() {
        use crate::config::schema::Config;
        let mut config = Config::default();
        let result = apply_trio_param(&mut config, "trio", "vram_cap", "0.1");
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_trio_param_router_nothink_toggle() {
        use crate::config::schema::Config;
        let mut config = Config::default();
        assert!(config.trio.router_no_think); // default true
        let result = apply_trio_param(&mut config, "router", "no_think", "toggle");
        assert!(result.is_ok());
        assert!(!config.trio.router_no_think);
        let _ = apply_trio_param(&mut config, "router", "no_think", "toggle");
        assert!(config.trio.router_no_think);
    }

    #[test]
    fn test_apply_trio_param_main_nothink_toggle() {
        use crate::config::schema::Config;
        let mut config = Config::default();
        assert!(config.trio.main_no_think);
        let result = apply_trio_param(&mut config, "main", "no_think", "toggle");
        assert!(result.is_ok());
        assert!(!config.trio.main_no_think);
    }

    #[test]
    fn test_apply_trio_param_unknown() {
        use crate::config::schema::Config;
        let mut config = Config::default();
        let result = apply_trio_param(&mut config, "router", "unknown_param", "x");
        assert!(result.is_err());
    }

    // ---- format_vram_budget tests ----

    #[test]
    fn test_format_vram_budget_contains_role_names() {
        let result = crate::server::VramBudgetResult {
            main_ctx: 16384,
            router_ctx: 4096,
            specialist_ctx: 8192,
            total_vram_bytes: 12_000_000_000,
            effective_limit_bytes: 16_000_000_000,
            fits: true,
            breakdown: vec![
                crate::server::VramModelBreakdown {
                    role: "main".to_string(),
                    name: "test-main".to_string(),
                    weights_bytes: 5_000_000_000,
                    kv_cache_bytes: 1_000_000_000,
                    context_tokens: 16384,
                    overhead_bytes: 512_000_000,
                },
                crate::server::VramModelBreakdown {
                    role: "router".to_string(),
                    name: "test-router".to_string(),
                    weights_bytes: 3_000_000_000,
                    kv_cache_bytes: 200_000_000,
                    context_tokens: 4096,
                    overhead_bytes: 512_000_000,
                },
                crate::server::VramModelBreakdown {
                    role: "specialist".to_string(),
                    name: "test-specialist".to_string(),
                    weights_bytes: 2_000_000_000,
                    kv_cache_bytes: 500_000_000,
                    context_tokens: 8192,
                    overhead_bytes: 512_000_000,
                },
            ],
        };
        let output = format_vram_budget(&result);
        assert!(output.contains("MAIN"), "Should contain MAIN role");
        assert!(output.contains("ROUTER"), "Should contain ROUTER role");
        assert!(
            output.contains("SPECIALIST"),
            "Should contain SPECIALIST role"
        );
        assert!(output.contains("OK"), "Should show OK when fits=true");
        assert!(output.contains("TOTAL"), "Should contain TOTAL line");
    }

    #[test]
    fn test_format_vram_budget_over_shows_over() {
        let result = crate::server::VramBudgetResult {
            main_ctx: 4096,
            router_ctx: 2048,
            specialist_ctx: 4096,
            total_vram_bytes: 20_000_000_000,
            effective_limit_bytes: 16_000_000_000,
            fits: false,
            breakdown: vec![crate::server::VramModelBreakdown {
                role: "main".to_string(),
                name: "big-model".to_string(),
                weights_bytes: 10_000_000_000,
                kv_cache_bytes: 500_000_000,
                context_tokens: 4096,
                overhead_bytes: 512_000_000,
            }],
        };
        let output = format_vram_budget(&result);
        assert!(output.contains("OVER"), "Should show OVER when fits=false");
    }

    // ---- should_auto_activate_trio ----

    #[test]
    fn test_should_auto_activate_trio_happy_path() {
        use crate::config::schema::DelegationMode;
        assert!(should_auto_activate_trio(
            true,
            "qwen3-1.7b",
            "ministral-3-8b",
            false,
            false,
            &DelegationMode::Delegated,
        ));
    }

    #[test]
    fn test_should_auto_activate_trio_not_local() {
        use crate::config::schema::DelegationMode;
        assert!(!should_auto_activate_trio(
            false,
            "qwen3-1.7b",
            "ministral-3-8b",
            false,
            false,
            &DelegationMode::Delegated,
        ));
    }

    #[test]
    fn test_should_auto_activate_trio_empty_router() {
        use crate::config::schema::DelegationMode;
        assert!(!should_auto_activate_trio(
            true,
            "",
            "ministral-3-8b",
            false,
            false,
            &DelegationMode::Delegated,
        ));
    }

    #[test]
    fn test_should_auto_activate_trio_empty_specialist() {
        use crate::config::schema::DelegationMode;
        assert!(!should_auto_activate_trio(
            true,
            "qwen3-1.7b",
            "",
            false,
            false,
            &DelegationMode::Delegated,
        ));
    }

    #[test]
    fn test_should_auto_activate_trio_already_trio() {
        use crate::config::schema::DelegationMode;
        assert!(!should_auto_activate_trio(
            true,
            "qwen3-1.7b",
            "ministral-3-8b",
            false,
            false,
            &DelegationMode::Trio,
        ));
    }

    #[test]
    fn test_should_auto_activate_trio_explicit_inline_opt_out() {
        use crate::config::schema::DelegationMode;
        // User explicitly set mode: "inline" — must never be overridden by auto-activation.
        assert!(!should_auto_activate_trio(
            true,
            "qwen3-1.7b",
            "ministral-3-8b",
            false,
            false,
            &DelegationMode::Inline,
        ));
    }

    #[test]
    fn test_should_auto_activate_trio_both_empty() {
        use crate::config::schema::DelegationMode;
        assert!(!should_auto_activate_trio(
            true,
            "",
            "",
            false,
            false,
            &DelegationMode::Delegated,
        ));
    }

    #[test]
    fn test_should_auto_activate_trio_with_explicit_endpoints() {
        use crate::config::schema::DelegationMode;
        // Empty GGUF model names but explicit endpoints → should activate
        assert!(should_auto_activate_trio(
            true,
            "",
            "",
            true,
            true,
            &DelegationMode::Delegated,
        ));
        // Neither GGUF nor endpoints → should NOT activate
        assert!(!should_auto_activate_trio(
            true,
            "",
            "",
            false,
            false,
            &DelegationMode::Delegated,
        ));
        // GGUF models (no endpoint) → still activates (existing behavior preserved)
        assert!(should_auto_activate_trio(
            true,
            "router.gguf",
            "spec.gguf",
            false,
            false,
            &DelegationMode::Delegated,
        ));
    }

    // ---- pick_trio_models ----

    #[test]
    fn test_pick_trio_models_orchestrator_and_instruct() {
        let available = vec![
            "Nemotron-Orchestrator-8B".to_string(),
            "Ministral-3-3B".to_string(),
            "Qwen3-14B".to_string(),
        ];
        let (router, specialist) = pick_trio_models(&available, "Qwen3-14B");
        assert_eq!(router.as_deref(), Some("Nemotron-Orchestrator-8B"));
        assert_eq!(specialist.as_deref(), Some("Ministral-3-3B"));
    }

    #[test]
    fn test_pick_trio_models_skips_main() {
        let available = vec!["Nemotron-Orchestrator-8B".to_string()];
        // Main model matches the only orchestrator — no router picked
        let (router, specialist) = pick_trio_models(&available, "Nemotron-Orchestrator-8B");
        assert!(router.is_none());
        assert!(specialist.is_none());
    }

    #[test]
    fn test_pick_trio_models_router_only() {
        let available = vec![
            "Nemotron-Orchestrator-8B".to_string(),
            "Qwen3-14B".to_string(),
        ];
        let (router, specialist) = pick_trio_models(&available, "Qwen3-14B");
        assert_eq!(router.as_deref(), Some("Nemotron-Orchestrator-8B"));
        assert!(specialist.is_none());
    }

    #[test]
    fn test_pick_trio_models_specialist_only() {
        let available = vec![
            "Qwen3-14B".to_string(),
            "Hermes-3-Llama-instruct-8B".to_string(),
        ];
        let (router, specialist) = pick_trio_models(&available, "Qwen3-14B");
        assert!(router.is_none());
        assert_eq!(specialist.as_deref(), Some("Hermes-3-Llama-instruct-8B"));
    }

    #[test]
    fn test_pick_trio_models_empty_list() {
        let (router, specialist) = pick_trio_models(&[], "Qwen3-14B");
        assert!(router.is_none());
        assert!(specialist.is_none());
    }

    #[test]
    fn test_pick_trio_models_function_calling_preferred() {
        let available = vec![
            "Qwen3-14B".to_string(),
            "Hermes-function-calling-3B".to_string(),
            "Llama-instruct-8B".to_string(),
        ];
        let (_, specialist) = pick_trio_models(&available, "Qwen3-14B");
        assert_eq!(specialist.as_deref(), Some("Hermes-function-calling-3B"));
    }

    #[test]
    fn test_pick_trio_models_case_insensitive() {
        let available = vec![
            "nemotron-ORCHESTRATOR-8b".to_string(),
            "MINISTRAL-3-3B".to_string(),
            "Qwen3-14B".to_string(),
        ];
        let (router, specialist) = pick_trio_models(&available, "Qwen3-14B");
        assert_eq!(router.as_deref(), Some("nemotron-ORCHESTRATOR-8b"));
        assert_eq!(specialist.as_deref(), Some("MINISTRAL-3-3B"));
    }

    #[test]
    fn test_pick_trio_models_no_duplicate_assignment() {
        // A model matching both router and specialist patterns should only be assigned once
        let available = vec![
            "orchestrator-instruct-8B".to_string(),
            "Qwen3-14B".to_string(),
        ];
        let (router, specialist) = pick_trio_models(&available, "Qwen3-14B");
        assert_eq!(router.as_deref(), Some("orchestrator-instruct-8B"));
        // specialist must not reuse the router
        assert!(specialist.is_none());
    }

    // ---- e2e: pick → fill config → should_auto_activate ----

    #[test]
    fn test_pick_trio_e2e_auto_activates() {
        use crate::config::schema::{Config, DelegationMode};

        let available = vec![
            "Qwen3-14B".to_string(),
            "Nemotron-Orchestrator-8B".to_string(),
            "Ministral-3-3B".to_string(),
        ];
        let main_model = "Qwen3-14B";

        // Step 1: pick
        let (auto_router, auto_specialist) = pick_trio_models(&available, main_model);

        // Step 2: fill empty config slots (mirrors mod.rs wiring)
        let mut config = Config::default();
        assert!(config.trio.router_model.is_empty());
        assert!(config.trio.specialist_model.is_empty());

        if config.trio.router_model.is_empty() {
            if let Some(r) = auto_router {
                config.trio.router_model = r;
            }
        }
        if config.trio.specialist_model.is_empty() {
            if let Some(s) = auto_specialist {
                config.trio.specialist_model = s;
            }
        }

        assert_eq!(config.trio.router_model, "Nemotron-Orchestrator-8B");
        assert_eq!(config.trio.specialist_model, "Ministral-3-3B");

        // Step 3: should_auto_activate now returns true
        assert!(should_auto_activate_trio(
            true,
            &config.trio.router_model,
            &config.trio.specialist_model,
            false,
            false,
            &DelegationMode::Delegated,
        ));

        // Step 4: trio_enable activates without warning
        let needs_warning = trio_enable(&mut config);
        assert!(!needs_warning);
        assert!(config.trio.enabled);
        assert_eq!(config.tool_delegation.mode, DelegationMode::Trio);
    }

    #[test]
    fn test_pick_trio_e2e_explicit_config_not_overridden() {
        use crate::config::schema::Config;

        let available = vec![
            "Qwen3-14B".to_string(),
            "Nemotron-Orchestrator-8B".to_string(),
            "Ministral-3-3B".to_string(),
        ];

        let mut config = Config::default();
        config.trio.router_model = "my-custom-router".to_string();
        config.trio.specialist_model = "my-custom-specialist".to_string();

        let (auto_router, auto_specialist) = pick_trio_models(&available, "Qwen3-14B");
        // Only fill empty slots
        if config.trio.router_model.is_empty() {
            if let Some(r) = auto_router {
                config.trio.router_model = r;
            }
        }
        if config.trio.specialist_model.is_empty() {
            if let Some(s) = auto_specialist {
                config.trio.specialist_model = s;
            }
        }

        // Explicit config preserved
        assert_eq!(config.trio.router_model, "my-custom-router");
        assert_eq!(config.trio.specialist_model, "my-custom-specialist");
    }

    // ---- fuzzy main-model exclusion (real LM Studio keys have org prefixes) ----

    #[test]
    fn test_pick_trio_models_org_prefix_skips_main() {
        // LMS returns "nvidia/nvidia-nemotron-nano-12b-v2-vl" but main_model
        // was resolved to "nvidia-nemotron-nano-12b-v2-vl" (without org prefix).
        let available = vec![
            "nvidia/nvidia-nemotron-nano-12b-v2-vl".to_string(),
            "nvidia/nemotron-orchestrator-8b".to_string(),
        ];
        let (router, specialist) = pick_trio_models(&available, "nvidia-nemotron-nano-12b-v2-vl");
        assert_eq!(router.as_deref(), Some("nvidia/nemotron-orchestrator-8b"));
        // Main model must NOT be picked as specialist even with org prefix mismatch
        assert!(specialist.is_none());
    }

    #[test]
    fn test_pick_trio_models_main_unresolved_hint() {
        // main_model is the original GGUF hint that didn't resolve (returned as-is).
        // available has the real LMS key. Substring match should still exclude it.
        let available = vec![
            "nvidia/nvidia-nemotron-nano-12b-v2-vl".to_string(),
            "nvidia/nemotron-orchestrator-8b".to_string(),
            "bartowski/ministral-3-3b".to_string(),
        ];
        let (router, specialist) = pick_trio_models(&available, "NVIDIA-Nemotron-Nano-12B-v2");
        assert_eq!(router.as_deref(), Some("nvidia/nemotron-orchestrator-8b"));
        // "nvidia-nemotron-nano-12b-v2" substring-matches available[0], so skip it
        assert_eq!(specialist.as_deref(), Some("bartowski/ministral-3-3b"));
    }

    #[test]
    fn test_pick_trio_models_real_lms_model_list() {
        // Mimics the user's actual LM Studio model list from error output
        let available = vec![
            "nemotron-orchestrator-8b-claude-4.5-opus-distill".to_string(),
            "nvidia-nemotron-nano-12b-v2-vl".to_string(),
            "nvidia_orchestrator-8b".to_string(),
        ];
        let (router, specialist) = pick_trio_models(&available, "nvidia-nemotron-nano-12b-v2-vl");
        // Should pick an orchestrator (not the main model)
        assert_eq!(
            router.as_deref(),
            Some("nemotron-orchestrator-8b-claude-4.5-opus-distill")
        );
        // No specialist patterns match any available model
        assert!(specialist.is_none());
    }
}
