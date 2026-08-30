// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(clippy::shadow_reuse)]
#![allow(dead_code)]
//! Main agent loop that consumes inbound messages and produces responses.
//!
//! Ported from Python `agent/loop.py`.
//!
//! The agent loop uses a fan-out pattern for concurrent message processing:
//! messages from different sessions run in parallel (up to `max_concurrent_chats`),
//! while messages within the same session are serialized to preserve ordering.

use std::collections::HashMap;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
#[cfg(test)]
use serde_json::Value;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::{Mutex, Semaphore};
use tracing::{debug, error, info, warn};

use crate::agent::reflector::Reflector;
use crate::agent::subagent::SubagentManager;
use crate::agent::system_state::SystemState;
use crate::bus::events::{InboundMessage, OutboundMessage};
use crate::config::schema::{EmailConfig, LcmSchemaConfig, ProprioceptionConfig};
use crate::cron::service::CronService;

// ---------------------------------------------------------------------------
// Core types re-exported from agent_core module
// ---------------------------------------------------------------------------
pub use crate::agent::agent_core::SharedCoreHandle;
// Re-export for test use (agent_loop_tests.rs uses `use super::*`).
#[cfg(test)]
pub(crate) use crate::agent::agent_core::{
    build_swappable_core, AgentHandle, SwappableCore, SwappableCoreConfig,
};

// ---------------------------------------------------------------------------
// Submodules
// ---------------------------------------------------------------------------

mod budget;
mod compaction;
mod heuristics;
mod local_stream;
mod response;
mod shared;

pub(crate) use heuristics::appears_incomplete;
pub(crate) use response::RetryState;
pub(crate) use shared::*;
// Re-export remaining heuristic functions at module-private level for use
// within this module and its submodules (shared uses them via super::).
#[cfg(test)]
use heuristics::adaptive_max_tokens;
use heuristics::{last_user_message, render_via_protocol, should_strip_tools_for_trio};

// ---------------------------------------------------------------------------
// Tool proxy wrappers
// ---------------------------------------------------------------------------
//
// Because `Arc<MessageTool>` etc. don't implement `Tool` directly (the trait
// requires owned `Box<dyn Tool>`), we create thin proxy wrappers that
// implement `Tool` by delegating to the inner `Arc`.

// ---------------------------------------------------------------------------
// AgentLoop (owns the receiver + orchestrates concurrency)
// ---------------------------------------------------------------------------

/// The core agent loop.
///
/// Consumes [`InboundMessage`]s from the bus, runs the LLM + tool loop, and
/// publishes [`OutboundMessage`]s when the agent produces a response.
///
/// In gateway mode, messages for different sessions run concurrently (up to
/// `max_concurrent_chats`), while messages within the same session are
/// serialized to preserve conversation ordering.
pub struct AgentLoop {
    shared: Arc<AgentLoopShared>,
    bus_inbound_rx: UnboundedReceiver<InboundMessage>,
    running: Arc<AtomicBool>,
    max_concurrent_chats: usize,
    reflection_spawned: AtomicBool,
}

impl AgentLoop {
    /// Create a new `AgentLoop`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        core_handle: SharedCoreHandle,
        bus_inbound_rx: UnboundedReceiver<InboundMessage>,
        bus_outbound_tx: UnboundedSender<OutboundMessage>,
        bus_inbound_tx: UnboundedSender<InboundMessage>,
        cron_service: Option<Arc<CronService>>,
        max_concurrent_chats: usize,
        email_config: Option<EmailConfig>,
        repl_display_tx: Option<UnboundedSender<String>>,
        providers_config: Option<crate::config::schema::ProvidersConfig>,
        proprioception_config: ProprioceptionConfig,
        lcm_config: LcmSchemaConfig,
        health_registry: Option<Arc<crate::heartbeat::health::HealthRegistry>>,
    ) -> Self {
        // Read core to initialize the subagent manager.
        let core = core_handle.swappable();
        let mut subagent_mgr = SubagentManager::new(
            core.provider.clone(),
            core.workspace.clone(),
            bus_inbound_tx.clone(),
            core.model.clone(),
            core.brave_api_key.clone(),
            core.exec_timeout,
            core.restrict_to_workspace,
            // migrated from swappable().is_local — phase 09-03
            core.mode().is_local(),
            core.max_tool_result_chars,
        )
        .with_search_config(
            core.search_provider.clone(),
            core.searxng_url.clone(),
            core.search_max_results,
        );
        if let Some(pc) = providers_config {
            subagent_mgr = subagent_mgr.with_providers_config(pc);
        }
        // Wire up the cheap default model for subagents from config.
        // Resolve "local" to the delegation model name so it's a real model
        // name that oMLX/servers recognise, not a literal "local" string.
        let subagent_model = {
            let raw = &core.tool_delegation_config.default_subagent_model;
            if raw.eq_ignore_ascii_case("local") {
                core.tool_runner_model
                    .clone()
                    .unwrap_or_else(|| core.model.clone())
            } else {
                raw.clone()
            }
        };
        subagent_mgr = subagent_mgr.with_default_subagent_model(subagent_model);
        // Wire up subagent tuning from config.
        subagent_mgr =
            subagent_mgr.with_subagent_tuning(core.tool_delegation_config.subagent.clone());
        if let Some(ref dtx) = repl_display_tx {
            subagent_mgr = subagent_mgr.with_display_tx(dtx.clone());
        }
        // migrated from swappable().is_local — phase 09-03
        if core.mode().is_local() {
            subagent_mgr = subagent_mgr.with_local_context_limit(core.token_budget.max_context());
        }

        // Create aha channel before subagent manager so we can pass the sender.
        let (aha_tx, aha_rx) = tokio::sync::mpsc::unbounded_channel();
        if proprioception_config.aha_channel {
            subagent_mgr = subagent_mgr.with_aha_tx(aha_tx.clone());
        }

        let subagents = Arc::new(subagent_mgr);

        let system_state = Arc::new(arc_swap::ArcSwap::from_pointee(SystemState::default()));

        let shared = Arc::new(AgentLoopShared {
            core_handle,
            subagents,
            bus_outbound_tx,
            cron_service,
            email_config,
            repl_display_tx,
            system_state,
            proprioception_config,
            idle: crate::agent::idle::IdleRuntime::default(),
            aha_rx: Arc::new(Mutex::new(aha_rx)),
            aha_tx,
            session_policies: Arc::new(Mutex::new(HashMap::new())),
            continuity_notes: Arc::new(Mutex::new(HashMap::new())),
            lcm_engines: Arc::new(Mutex::new(HashMap::new())),
            compaction_handles: Arc::new(Mutex::new(HashMap::new())),
            lcm_config,
            health_registry,
            knowledge_store: crate::agent::knowledge_store::KnowledgeStore::open_default()
                .ok()
                .map(|ks| Arc::new(parking_lot::Mutex::new(ks))),
        });

        Self {
            shared,
            bus_inbound_rx,
            running: Arc::new(AtomicBool::new(false)),
            max_concurrent_chats,
            reflection_spawned: AtomicBool::new(false),
        }
    }

    /// Set the cluster router for distributed inference routing.
    ///
    /// Must be called before `run()` or `process_direct()` to take effect.
    #[cfg(feature = "cluster")]
    pub fn set_cluster_router(&mut self, router: Arc<crate::cluster::router::ClusterRouter>) {
        // Only the subagent manager consumes the router (per-model peer
        // resolution); the former Shared.cluster_router field was a dead
        // store — written, never read (removed v0.5 E5).
        // SAFETY: we hold &mut self so no concurrent access exists yet.
        let shared = Arc::get_mut(&mut self.shared)
            .expect("set_cluster_router called after shared Arc was cloned");
        let subagents = Arc::get_mut(&mut shared.subagents)
            .expect("set_cluster_router: subagents Arc already shared");
        subagents.cluster_router = Some(router);
    }

    /// Wire idle-window agency (gateway only, before `run()`). The returned
    /// tracker Arc is the same one the idle timer task polls; `run()` notes
    /// real inbound activity into it.
    pub fn set_idle_runtime(
        &mut self,
        runtime: crate::agent::idle::IdleRuntime,
    ) -> Arc<crate::agent::idle::IdleTracker> {
        // SAFETY: we hold &mut self so no concurrent access exists yet.
        let shared = Arc::get_mut(&mut self.shared)
            .expect("set_idle_runtime called after shared Arc was cloned");
        shared.idle = runtime;
        shared.idle.tracker.clone()
    }

    /// Check whether the in-process MLX provider is set.

    /// Spawn a background reflection task if completed working sessions exceed
    /// the configured threshold.
    fn spawn_background_reflection(shared: &Arc<AgentLoopShared>) {
        let core = shared.core_handle.swappable();
        if !core.memory_enabled {
            return;
        }
        tokio::spawn(async move {
            if !Reflector::should_reflect_sessions(&core.sessions, core.reflection_threshold).await
            {
                return;
            }
            let reflector = Reflector::new(
                core.memory_provider.clone(),
                core.memory_model.clone(),
                &core.workspace,
                core.reflection_threshold,
                core.sessions.clone(),
                core.memory_file_max_words,
            );
            info!("Background: reflecting on completed SQLite working memory...");
            let result = reflector.reflect().await;
            if let Err(error) = result {
                tracing::warn!(%error, "Background reflection failed");
            } else {
                info!("Background reflection complete — MEMORY.md updated");
            }
        });
    }

    /// Run the main agent loop until stopped.
    ///
    /// Messages for different sessions are processed concurrently (up to
    /// `max_concurrent_chats`). Messages within the same session are serialized.
    pub async fn run(&mut self) {
        self.running.store(true, Ordering::SeqCst);
        info!(
            "Agent loop started (max_concurrent_chats={})",
            self.max_concurrent_chats
        );

        // Spawn background reflection if completed SQLite working memory has accumulated.
        Self::spawn_background_reflection(&self.shared);

        let semaphore = Arc::new(Semaphore::new(self.max_concurrent_chats));
        // Per-session locks to serialize messages within the same conversation.
        let session_locks: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        // Coalescing can observe the first message for another session. Retain
        // it for the next loop iteration so it traverses the exact same
        // system/command/lock/permit path instead of a parallel side path
        // (see main b9d0055).
        let mut pending_msg = None;

        while self.running.load(Ordering::SeqCst) {
            let msg = if let Some(msg) = pending_msg.take() {
                msg
            } else {
                match tokio::time::timeout(Duration::from_secs(1), self.bus_inbound_rx.recv()).await
                {
                    Ok(Some(msg)) => msg,
                    Ok(None) => {
                        info!("Inbound channel closed, stopping agent loop");
                        break;
                    }
                    Err(_) => continue, // timeout - loop and check running flag
                }
            };

            // Coalesce rapid messages from the same session (Telegram, WhatsApp).
            // Waits up to 400ms for follow-up messages before processing.
            // Idle turns never coalesce: their observation must not be glued
            // onto a real user message arriving in the window.
            let msg = if crate::bus::events::should_coalesce(&msg.channel)
                && !msg.content.trim_start().starts_with('/')
                && !crate::agent::idle::is_idle_message(&msg)
            {
                let session = msg.session_key();
                let mut batch = vec![msg];
                let deadline = tokio::time::Instant::now() + Duration::from_millis(400);
                loop {
                    match tokio::time::timeout_at(deadline, self.bus_inbound_rx.recv()).await {
                        Ok(Some(next)) if next.session_key() == session => {
                            batch.push(next);
                        }
                        Ok(Some(other)) => {
                            // Preserve the different-session message for the
                            // next normal iteration. This is the in-process
                            // equivalent of push-back without a second gateway
                            // execution pipeline (see main b9d0055) — a side
                            // path here would skip is_system + /-command
                            // interception, letting a /clear from another
                            // session reach the LLM as plain text.
                            pending_msg = Some(other);
                            break;
                        }
                        _ => break, // timeout or channel closed
                    }
                }
                if batch.len() > 1 {
                    debug!("Coalesced {} messages for session", batch.len());
                }
                crate::bus::events::coalesce_messages(batch)
            } else {
                msg
            };

            // System messages (subagent announces) are handled inline (fast).
            let is_system = msg
                .metadata
                .get("is_system")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            if is_system {
                debug!(
                    "Processing system message: {}",
                    &msg.content[..msg.content.len().min(80)]
                );
                let outbound = OutboundMessage::new(&msg.channel, &msg.chat_id, &msg.content);
                if let Err(e) = self.shared.bus_outbound_tx.send(outbound) {
                    error!("Failed to publish outbound message: {}", e);
                }
                continue;
            }

            // Idle-turn activity bookkeeping: record real inbound for the
            // idle timer's designated-session resolution and backoff reset.
            // Placed after the is_system branch (internal events are not
            // user activity) and skipped for idle turns themselves — an
            // agent's own thoughts must not reset its idle backoff.
            if !crate::agent::idle::is_idle_message(&msg) {
                self.shared
                    .idle
                    .tracker
                    .note_inbound(&msg.channel, &msg.chat_id);
            }

            // Get or create the per-session lock.
            let session_key = msg.session_key();
            let session_lock = {
                let mut locks = session_locks.lock().await;
                locks
                    .entry(session_key)
                    .or_insert_with(|| Arc::new(Mutex::new(())))
                    .clone()
            };

            let shared = self.shared.clone();
            let outbound_tx = self.shared.bus_outbound_tx.clone();
            let display_tx = self.shared.repl_display_tx.clone();
            let request_sem = semaphore.clone();

            tokio::spawn(async move {
                // Serialize within the same session.
                let _session_guard = session_lock.lock().await;

                // Commands such as /clear mutate session history and retained
                // prompt state, so dispatch them under the same lock as normal
                // turns — dispatching on the run() loop outside this lock lets
                // a /clear race an in-flight turn and resurrect cleared
                // history via later persists (see main 3cbc59d).
                if msg.content.trim().starts_with('/') {
                    if let Some(response_text) =
                        crate::agent::gateway_commands::dispatch(&shared, &msg).await
                    {
                        let outbound = crate::bus::events::OutboundMessage::new(
                            &msg.channel,
                            &msg.chat_id,
                            &response_text,
                        );
                        if let Err(error) = outbound_tx.send(outbound) {
                            tracing::error!(%error, "Failed to send command response");
                        }
                        return;
                    }
                }

                // Permit acquired post-lock so exhaustion can't stall command
                // dispatch or other sessions; recognized commands never need
                // one at all (see main 3fb926d).
                let permit = match request_sem.acquire_owned().await {
                    Ok(permit) => permit,
                    Err(_) => {
                        error!("Semaphore closed unexpectedly");
                        return;
                    }
                };

                // Notify REPL about inbound channel message.
                if let Some(ref dtx) = display_tx {
                    let preview = if msg.content.len() > 120 {
                        let end = crate::utils::helpers::floor_char_boundary(&msg.content, 120);
                        format!("{}...", &msg.content[..end])
                    } else {
                        msg.content.clone()
                    };
                    let _ = dtx.send(format!(
                        "\x1b[2m[{}]\x1b[0m \x1b[36m{}\x1b[0m: {}",
                        msg.channel, msg.sender_id, preview
                    ));
                }

                // For Telegram: set up streaming with typing indicator + progressive edits.
                // The actual streaming logic (typing action, placeholder, throttled edits)
                // lives in channels/telegram.rs::spawn_stream_editor so the hot path
                // stays channel-agnostic. Idle turns skip it: no typing indicators
                // or placeholder edits for self-directed turns (quiet by default).
                let stream_tx =
                    if msg.channel == "telegram" && !crate::agent::idle::is_idle_message(&msg) {
                        let bot_token = msg
                            .metadata
                            .get("bot_token")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        crate::channels::telegram::spawn_stream_editor(bot_token, &msg.chat_id)
                    } else {
                        None
                    };
                let stream_is_telegram = stream_tx.is_some();

                let response = shared
                    .process_message(&msg, stream_tx, None, None, None)
                    .await;

                // Quiet by default for idle turns: the final reply is logged
                // instead of sent — the agent reaches the human only through
                // an explicit `message` tool call (Q2 decision, v0.5 E1).
                if crate::agent::idle::is_idle_message(&msg) {
                    match response {
                        Some(outbound) => info!(
                            channel = %outbound.channel,
                            "idle turn completed (reply suppressed)"
                        ),
                        None => warn!("idle turn errored during processing"),
                    }
                    drop(permit);
                    return;
                }

                let outbound = match response {
                    Some(mut outbound) => {
                        if stream_is_telegram {
                            outbound
                                .metadata
                                .insert("streaming_handled".to_string(), serde_json::json!(true));
                        }
                        outbound
                    }
                    None => {
                        error!(
                            channel = %msg.channel,
                            chat_id = %msg.chat_id,
                            "process_message returned None; sending error feedback to user"
                        );
                        crate::bus::events::OutboundMessage::new(
                            &msg.channel,
                            &msg.chat_id,
                            "[nanobot] Sorry, I encountered an error processing your message. Please try again.",
                        )
                    }
                };

                // Notify REPL about outbound response.
                if let Some(ref dtx) = display_tx {
                    let preview = if outbound.content.len() > 120 {
                        let end =
                            crate::utils::helpers::floor_char_boundary(&outbound.content, 120);
                        format!("{}...", &outbound.content[..end])
                    } else {
                        outbound.content.clone()
                    };
                    let _ = dtx.send(format!(
                        "\x1b[2m[{}]\x1b[0m \x1b[33mbot\x1b[0m: {}",
                        outbound.channel, preview
                    ));
                }
                if let Err(e) = outbound_tx.send(outbound) {
                    error!("Failed to publish outbound message: {}", e);
                }

                drop(permit); // release concurrency slot
            });
        }

        info!("Agent loop stopped");
    }

    /// Return a handle to the subagent manager.
    pub fn subagent_manager(&self) -> Arc<SubagentManager> {
        self.shared.subagents.clone()
    }

    /// Build the current local prompt runtime blocks for inspection/debugging.
    pub async fn local_prompt_runtime_blocks(
        &self,
        session_key: &str,
    ) -> Vec<crate::agent::context::PromptBlock> {
        let core = self.shared.core_handle.swappable();
        let session = core.sessions.get_or_resume(session_key).await;
        self.shared
            .build_local_runtime_blocks(&core, &session.id)
            .await
    }

    /// Cancel generation and join every compaction job owned by this loop.
    pub(crate) async fn drain_compaction_jobs(&self) {
        let handles = {
            let handles = self.shared.compaction_handles.lock().await;
            handles.values().cloned().collect::<Vec<_>>()
        };
        futures_util::future::join_all(handles.iter().map(|handle| handle.cancel_and_reap())).await;
    }

    /// Atomically clear all state owned by one concrete interactive session.
    pub async fn clear_session_state(&self, session_key: &str) {
        let core = self.shared.core_handle.swappable();
        let session = core.sessions.get_or_resume(session_key).await;
        let session_id = session.id;
        let (compaction, closing) = self.begin_session_clear(&session_id).await;

        compaction.cancel_and_reap().await;
        *compaction.slot.lock().await = None;
        if self
            .shared
            .lcm_engines
            .lock()
            .await
            .remove(&session_id)
            .is_some()
        {
            debug!(%session_id, "LCM engine cleared");
        }
        if core.memory_enabled {
            if let Err(error) = core.working_memory.clear(&session_id).await {
                warn!(%error, %session_id, "failed to clear working memory");
            }
        }
        core.sessions.clear_history(&session_id).await;

        let counters = &self.shared.core_handle.counters;
        counters.reset_session_prompt_state(session_key);
        counters.last_context_used.store(0, Ordering::Relaxed);
        counters.last_message_count.store(0, Ordering::Relaxed);
        counters
            .last_working_memory_tokens
            .store(0, Ordering::Relaxed);
        self.retire_session_clear(&session_id, &compaction, closing)
            .await;
    }

    async fn begin_session_clear(
        &self,
        session_id: &str,
    ) -> (CompactionHandle, compaction::CompactionClosingAdmission) {
        loop {
            let candidate = self.shared.compaction_handle_for_session(session_id).await;
            if let Some(closing) = candidate.begin_close().await {
                return (candidate, closing);
            }
            self.shared
                .remove_compaction_handle_if_owned(session_id, &candidate)
                .await;
        }
    }

    async fn retire_session_clear(
        &self,
        session_id: &str,
        compaction: &CompactionHandle,
        closing: compaction::CompactionClosingAdmission,
    ) {
        // Retire and remove only the handle whose closing admission spans this
        // clear. Releasing the permit last wakes preparations against a fully
        // cleared durable session; stale waiters then retry through the map.
        let mut handles = self.shared.compaction_handles.lock().await;
        closing.retire();
        if handles
            .get(session_id)
            .is_some_and(|current| current.same_owner(compaction))
        {
            handles.remove(session_id);
        }
        drop(handles);
        drop(closing);
    }

    /// Signal the agent loop to stop.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Process a message directly (for CLI / cron usage) without going through
    /// the bus.
    pub async fn process_direct(
        &self,
        content: &str,
        session_key: &str,
        channel: &str,
        chat_id: &str,
    ) -> String {
        self.process_direct_with_lang(content, session_key, channel, chat_id, None)
            .await
    }

    /// Like `process_direct` but allows passing a detected language code
    /// (e.g. "it", "es") so the LLM responds in that language.
    pub async fn process_direct_with_lang(
        &self,
        content: &str,
        session_key: &str,
        channel: &str,
        chat_id: &str,
        detected_language: Option<&str>,
    ) -> String {
        // Spawn background reflection once per session (on first message).
        if !self.reflection_spawned.swap(true, Ordering::SeqCst) {
            Self::spawn_background_reflection(&self.shared);
        }

        let mut msg = InboundMessage::new(channel, "user", chat_id, content);
        msg.metadata
            .insert("session_key".to_string(), json!(session_key));
        if let Some(lang) = detected_language {
            msg.metadata
                .insert("detected_language".to_string(), json!(lang));
        }

        match self
            .shared
            .process_message(&msg, None, None, None, None)
            .await
        {
            Some(response) => response.content,
            None => String::new(),
        }
    }

    /// Like `process_direct_with_lang` but streams text deltas to `text_delta_tx`
    /// as they arrive from the LLM. Returns the full response text.
    pub async fn process_direct_streaming(
        &self,
        content: &str,
        session_key: &str,
        channel: &str,
        chat_id: &str,
        detected_language: Option<&str>,
        text_delta_tx: tokio::sync::mpsc::UnboundedSender<String>,
        tool_event_tx: Option<tokio::sync::mpsc::UnboundedSender<crate::agent::audit::ToolEvent>>,
        cancellation_token: Option<tokio_util::sync::CancellationToken>,
        priority_rx: Option<tokio::sync::mpsc::UnboundedReceiver<String>>,
        media_paths: Option<&[String]>,
    ) -> String {
        if !self.reflection_spawned.swap(true, Ordering::SeqCst) {
            Self::spawn_background_reflection(&self.shared);
        }

        let msg = Self::build_direct_message(
            channel,
            chat_id,
            content,
            session_key,
            detected_language,
            media_paths,
        );

        match self
            .shared
            .process_message(
                &msg,
                Some(text_delta_tx),
                tool_event_tx,
                cancellation_token,
                priority_rx,
            )
            .await
        {
            Some(response) => response.content,
            None => String::new(),
        }
    }

    /// Build the `InboundMessage` for a direct (CLI/TUI) turn. Shared by
    /// [`Self::process_direct_streaming`] and [`Self::spawn_direct_streaming`].
    fn build_direct_message(
        channel: &str,
        chat_id: &str,
        content: &str,
        session_key: &str,
        detected_language: Option<&str>,
        media_paths: Option<&[String]>,
    ) -> InboundMessage {
        let mut msg = InboundMessage::new(channel, "user", chat_id, content);
        msg.metadata
            .insert("session_key".to_string(), json!(session_key));
        if let Some(lang) = detected_language {
            msg.metadata
                .insert("detected_language".to_string(), json!(lang));
        }
        if let Some(media) = media_paths.filter(|paths| !paths.is_empty()) {
            msg.metadata.insert("media".to_string(), json!(media));
        }
        msg
    }

    /// Spawn a direct streaming turn on its own task, returning a handle to the
    /// final response text. Unlike [`Self::process_direct_streaming`] (awaited
    /// inline), the agent work runs on a separate task — so a caller driving its
    /// own loop (the TUI render loop) keeps redrawing while a CPU-heavy turn
    /// runs, and the animation never stalls behind the model. Args are owned so
    /// the spawned future is `'static`.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_direct_streaming(
        &self,
        content: String,
        session_key: String,
        channel: String,
        chat_id: String,
        detected_language: Option<String>,
        text_delta_tx: tokio::sync::mpsc::UnboundedSender<String>,
        tool_event_tx: Option<tokio::sync::mpsc::UnboundedSender<crate::agent::audit::ToolEvent>>,
        cancellation_token: Option<tokio_util::sync::CancellationToken>,
        media_paths: Option<Vec<String>>,
    ) -> tokio::task::JoinHandle<String> {
        if !self.reflection_spawned.swap(true, Ordering::SeqCst) {
            Self::spawn_background_reflection(&self.shared);
        }
        let shared = self.shared.clone();
        tokio::spawn(async move {
            let msg = Self::build_direct_message(
                &channel,
                &chat_id,
                &content,
                &session_key,
                detected_language.as_deref(),
                media_paths.as_deref(),
            );
            match shared
                .process_message(
                    &msg,
                    Some(text_delta_tx),
                    tool_event_tx,
                    cancellation_token,
                    None,
                )
                .await
            {
                Some(response) => response.content,
                None => String::new(),
            }
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
