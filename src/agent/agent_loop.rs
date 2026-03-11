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

use serde_json::{json, Value};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::{Mutex, Semaphore};
use tracing::{debug, error, info};

use crate::agent::compaction::ContextCompactor;
use crate::agent::reflector::Reflector;
use crate::agent::subagent::SubagentManager;
use crate::agent::system_state::SystemState;
use crate::bus::events::{InboundMessage, OutboundMessage};
use crate::config::schema::{EmailConfig, LcmSchemaConfig, ProprioceptionConfig};
use crate::cron::service::CronService;

// ---------------------------------------------------------------------------
// Core types re-exported from agent_core module
// ---------------------------------------------------------------------------
pub use crate::agent::agent_core::{
    build_swappable_core, AgentHandle, RuntimeCounters, SharedCoreHandle, SwappableCore,
    SwappableCoreConfig,
};
// Re-export for test use (agent_loop_tests.rs uses `use super::*`).
pub(crate) use crate::agent::agent_core::{history_limit, provenance_warning_role};

// ---------------------------------------------------------------------------
// Submodules (loaded via #[path] because agent_loop is a file, not a dir)
// ---------------------------------------------------------------------------

#[path = "agent_shared.rs"]
mod agent_shared;
pub(crate) use agent_shared::*;

#[path = "agent_heuristics.rs"]
mod agent_heuristics;
pub(crate) use agent_heuristics::appears_incomplete;
// Re-export remaining heuristic functions at module-private level for use
// within this module and its submodules (agent_shared uses them via super::).
use agent_heuristics::{
    adaptive_max_tokens, last_user_message, render_via_protocol, should_strip_tools_for_trio,
};

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
            core.is_local,
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
        subagent_mgr = subagent_mgr.with_default_subagent_model(
            core.tool_delegation_config.default_subagent_model.clone(),
        );
        // Wire up subagent tuning from config.
        subagent_mgr =
            subagent_mgr.with_subagent_tuning(core.tool_delegation_config.subagent.clone());
        if let Some(ref dtx) = repl_display_tx {
            subagent_mgr = subagent_mgr.with_display_tx(dtx.clone());
        }
        if core.is_local {
            subagent_mgr = subagent_mgr.with_local_context_limit(core.token_budget.max_context());
        }

        // Create aha channel before subagent manager so we can pass the sender.
        let (aha_tx, aha_rx) = tokio::sync::mpsc::unbounded_channel();
        if proprioception_config.aha_channel {
            subagent_mgr = subagent_mgr.with_aha_tx(aha_tx.clone());
        }

        let subagents = Arc::new(subagent_mgr);

        // Load persisted bulletin from disk (warm start).
        let bulletin_cache = {
            let core = core_handle.swappable();
            let cache = crate::agent::bulletin::BulletinCache::new();
            if let Some(persisted) =
                crate::agent::bulletin::load_persisted_bulletin(&core.workspace)
            {
                cache.update(persisted);
            }
            cache.handle()
        };

        let system_state = Arc::new(arc_swap::ArcSwap::from_pointee(SystemState::default()));

        // Build dedicated LCM compactor when compaction_endpoint is configured.
        let lcm_compactor = lcm_config.compaction_endpoint.as_ref().map(|ep| {
            let provider: Arc<dyn crate::providers::base::LLMProvider> =
                crate::providers::factory::create_openai_compat(
                    crate::providers::factory::ProviderSpec {
                        api_key: "lcm-compactor".to_string(),
                        api_base: Some(ep.url.clone()),
                        model: Some(ep.model.clone()),
                        jit_gate: None,
                        retry: crate::config::schema::RetryConfig::default(),
                        timeout_secs: 120,
                        lms_native_probe_secs: 2,
                    },
                );
            Arc::new(ContextCompactor::new(
                provider,
                ep.model.clone(),
                lcm_config.compaction_context_size,
            ))
        });

        let mut shared = Arc::new(AgentLoopShared {
            core_handle,
            subagents,
            bus_outbound_tx,
            bus_inbound_tx,
            cron_service,
            email_config,
            repl_display_tx,
            bulletin_cache,
            system_state,
            proprioception_config,
            aha_rx: Arc::new(Mutex::new(aha_rx)),
            aha_tx,
            session_policies: Arc::new(Mutex::new(HashMap::new())),
            lcm_engines: Arc::new(Mutex::new(HashMap::new())),
            lcm_config,
            lcm_compactor,
            health_registry,
            calibrator: {
                let cal: Option<Arc<parking_lot::Mutex<_>>> =
                    match crate::agent::budget_calibrator::BudgetCalibrator::open_default() {
                        Ok(c) => Some(Arc::new(parking_lot::Mutex::new(c))),
                        Err(e) => {
                            tracing::warn!(
                                "BudgetCalibrator init failed, recording disabled: {}",
                                e
                            );
                            None
                        }
                    };
                cal
            },
            learn_loop: {
                // Placeholder -- rebuilt in set_perplexity_gate / after all fields are set.
                // At construction time, calibrator and experience_buffer are not yet Arc-cloneable
                // from the struct itself, so we initialize with empty observers.
                Arc::new(crate::agent::learn_loop::DefaultLearnLoop {
                    calibrator: None,
                    experience_buffer: None,
                    perplexity_gate_config: Default::default(),
                    #[cfg(feature = "mlx")]
                    mlx_provider: None,
                    training_counters: None,
                    ane_model_dir: None,
                })
            },
            #[cfg(feature = "cluster")]
            cluster_router: None,
            knowledge_store: crate::agent::knowledge_store::KnowledgeStore::open_default()
                .ok()
                .map(|ks| Arc::new(parking_lot::Mutex::new(ks))),
            experience_buffer: crate::agent::lora_bridge::ExperienceBuffer::open_default()
                .ok()
                .map(|eb| Arc::new(parking_lot::Mutex::new(eb))),
            perplexity_gate_config: Default::default(),
            #[cfg(feature = "mlx")]
            mlx_provider: None,
            ane_model_dir: None,
        });
        // Rebuild learn_loop now that shared fields are accessible.
        {
            let s = Arc::get_mut(&mut shared).expect("learn_loop init: shared Arc not yet cloned");
            s.learn_loop = Arc::new(crate::agent::learn_loop::DefaultLearnLoop {
                calibrator: s.calibrator.clone(),
                experience_buffer: s.experience_buffer.clone(),
                perplexity_gate_config: s.perplexity_gate_config.clone(),
                #[cfg(feature = "mlx")]
                mlx_provider: s.mlx_provider.clone(),
                training_counters: Some(s.core_handle.counters.clone()),
                ane_model_dir: s.ane_model_dir.clone(),
            });
        }

        Self {
            shared,
            bus_inbound_rx,
            running: Arc::new(AtomicBool::new(false)),
            max_concurrent_chats,
            reflection_spawned: AtomicBool::new(false),
        }
    }

    /// Configure the perplexity gate for automatic online learning.
    ///
    /// Must be called before `run()` or `process_direct()` to take effect.
    pub fn set_perplexity_gate(&mut self, config: crate::config::schema::PerplexityGateConfig) {
        let shared = Arc::get_mut(&mut self.shared)
            .expect("set_perplexity_gate called after shared Arc was cloned");
        shared.perplexity_gate_config = config;
        // Rebuild learn_loop with updated config.
        shared.learn_loop = Arc::new(crate::agent::learn_loop::DefaultLearnLoop {
            calibrator: shared.calibrator.clone(),
            experience_buffer: shared.experience_buffer.clone(),
            perplexity_gate_config: shared.perplexity_gate_config.clone(),
            #[cfg(feature = "mlx")]
            mlx_provider: shared.mlx_provider.clone(),
            training_counters: Some(shared.core_handle.counters.clone()),
            ane_model_dir: shared.ane_model_dir.clone(),
        });
    }

    /// Set the in-process MLX provider for direct perplexity + training.
    #[cfg(feature = "mlx")]
    pub fn set_mlx_provider(&mut self, provider: Arc<crate::providers::mlx::MlxProvider>) {
        let shared = Arc::get_mut(&mut self.shared)
            .expect("set_mlx_provider called after shared Arc was cloned");
        shared.mlx_provider = Some(provider.clone());
        // Rebuild learn_loop with updated MLX provider.
        shared.learn_loop = Arc::new(crate::agent::learn_loop::DefaultLearnLoop {
            calibrator: shared.calibrator.clone(),
            experience_buffer: shared.experience_buffer.clone(),
            perplexity_gate_config: shared.perplexity_gate_config.clone(),
            mlx_provider: Some(provider),
            training_counters: Some(shared.core_handle.counters.clone()),
            ane_model_dir: shared.ane_model_dir.clone(),
        });
    }

    /// Set the model directory for standalone ANE training (no in-process MLX).
    pub fn set_ane_model_dir(&mut self, dir: Option<std::path::PathBuf>) {
        let shared = Arc::get_mut(&mut self.shared)
            .expect("set_ane_model_dir called after shared Arc was cloned");
        shared.ane_model_dir = dir.clone();
        // Rebuild learn_loop with updated model dir.
        shared.learn_loop = Arc::new(crate::agent::learn_loop::DefaultLearnLoop {
            calibrator: shared.calibrator.clone(),
            experience_buffer: shared.experience_buffer.clone(),
            perplexity_gate_config: shared.perplexity_gate_config.clone(),
            #[cfg(feature = "mlx")]
            mlx_provider: shared.mlx_provider.clone(),
            training_counters: Some(shared.core_handle.counters.clone()),
            ane_model_dir: dir,
        });
    }

    /// Set the cluster router for distributed inference routing.
    ///
    /// Must be called before `run()` or `process_direct()` to take effect.
    #[cfg(feature = "cluster")]
    pub fn set_cluster_router(&mut self, router: Arc<crate::cluster::router::ClusterRouter>) {
        // SAFETY: we hold &mut self so no concurrent access exists yet.
        let shared = Arc::get_mut(&mut self.shared)
            .expect("set_cluster_router called after shared Arc was cloned");
        shared.cluster_router = Some(router.clone());
        // Also pass the router down to the subagent manager.
        let subagents = Arc::get_mut(&mut shared.subagents)
            .expect("set_cluster_router: subagents Arc already shared");
        subagents.cluster_router = Some(router);
    }

    /// Check whether the perplexity gate is enabled on this agent loop.
    pub fn has_perplexity_gate(&self) -> bool {
        self.shared.perplexity_gate_config.enabled
    }

    /// Check whether the in-process MLX provider is set.
    #[cfg(all(test, feature = "mlx"))]
    pub fn has_mlx_provider(&self) -> bool {
        self.shared.mlx_provider.is_some()
    }

    /// Spawn a periodic bulletin refresh task (compaction model, when idle).
    fn spawn_bulletin_refresh(shared: &Arc<AgentLoopShared>, running: &Arc<AtomicBool>) {
        let core = shared.core_handle.swappable();
        if !core.memory_enabled {
            return;
        }
        let provider = core.memory_provider.clone();
        let model = core.memory_model.clone();
        let workspace = core.workspace.clone();
        let cache = shared.bulletin_cache.clone();
        let running = running.clone();

        tokio::spawn(async move {
            // Initial delay: let the system settle before first bulletin.
            tokio::time::sleep(Duration::from_secs(5 * 60)).await;

            while running.load(Ordering::Relaxed) {
                debug!("Bulletin: refreshing...");
                if let Err(e) = crate::agent::bulletin::refresh_bulletin(
                    provider.as_ref(),
                    &model,
                    &workspace,
                    &cache,
                )
                .await
                {
                    tracing::warn!("Bulletin refresh failed: {}", e);
                }
                // Sleep until next refresh.
                tokio::time::sleep(Duration::from_secs(
                    crate::agent::bulletin::DEFAULT_BULLETIN_INTERVAL_S,
                ))
                .await;
            }
        });
        info!(
            "Bulletin refresh task spawned (every {}min)",
            crate::agent::bulletin::DEFAULT_BULLETIN_INTERVAL_S / 60
        );
    }

    /// Spawn a background reflection task if observations exceed threshold.
    fn spawn_background_reflection(shared: &Arc<AgentLoopShared>) {
        let core = shared.core_handle.swappable();
        if !core.memory_enabled {
            return;
        }
        let reflector = Reflector::new(
            core.memory_provider.clone(),
            core.memory_model.clone(),
            &core.workspace,
            core.reflection_threshold,
        );
        if reflector.should_reflect() {
            tokio::spawn(async move {
                info!("Background: reflecting on accumulated observations...");
                if let Err(e) = reflector.reflect().await {
                    tracing::warn!("Background reflection failed: {}", e);
                } else {
                    info!("Background reflection complete — MEMORY.md updated");
                }
            });
        }
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

        // Spawn background reflection if observations have accumulated.
        Self::spawn_background_reflection(&self.shared);

        // Spawn periodic bulletin refresh (compaction model synthesizes briefing).
        Self::spawn_bulletin_refresh(&self.shared, &self.running);

        let semaphore = Arc::new(Semaphore::new(self.max_concurrent_chats));
        // Per-session locks to serialize messages within the same conversation.
        let session_locks: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        while self.running.load(Ordering::SeqCst) {
            let msg = match tokio::time::timeout(Duration::from_secs(1), self.bus_inbound_rx.recv())
                .await
            {
                Ok(Some(msg)) => msg,
                Ok(None) => {
                    info!("Inbound channel closed, stopping agent loop");
                    break;
                }
                Err(_) => continue, // timeout - loop and check running flag
            };

            // Coalesce rapid messages from the same session (Telegram, WhatsApp).
            // Waits up to 400ms for follow-up messages before processing.
            let msg = if crate::bus::events::should_coalesce(&msg.channel)
                && !msg.content.trim_start().starts_with('/')
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
                            // Different session — coalesce what we have, push other back.
                            // Can't push back into mpsc, so process inline as separate spawn.
                            let other_key = other.session_key();
                            let other_lock = {
                                let mut locks = session_locks.lock().await;
                                locks
                                    .entry(other_key)
                                    .or_insert_with(|| Arc::new(Mutex::new(())))
                                    .clone()
                            };
                            let other_shared = self.shared.clone();
                            let other_outbound_tx = self.shared.bus_outbound_tx.clone();
                            let _other_display_tx = self.shared.repl_display_tx.clone();
                            let other_sem = semaphore.clone();
                            tokio::spawn(async move {
                                if let Ok(permit) = other_sem.acquire_owned().await {
                                    let _guard = other_lock.lock().await;
                                    if let Some(resp) = other_shared
                                        .process_message(&other, None, None, None, None)
                                        .await
                                    {
                                        let _ = other_outbound_tx.send(resp);
                                    }
                                    drop(permit);
                                }
                            });
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

            // Gateway slash command interception — handle before LLM processing.
            if msg.content.trim().starts_with('/') {
                if let Some(response_text) =
                    crate::agent::gateway_commands::dispatch(&self.shared, &msg).await
                {
                    let outbound = crate::bus::events::OutboundMessage::new(
                        &msg.channel,
                        &msg.chat_id,
                        &response_text,
                    );
                    if let Err(e) = self.shared.bus_outbound_tx.send(outbound) {
                        tracing::error!("Failed to send command response: {}", e);
                    }
                    continue;
                }
            }

            // Acquire a concurrency permit.
            let permit = match semaphore.clone().acquire_owned().await {
                Ok(p) => p,
                Err(_) => {
                    error!("Semaphore closed unexpectedly");
                    break;
                }
            };

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

            tokio::spawn(async move {
                // Serialize within the same session.
                let _session_guard = session_lock.lock().await;

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
                let (stream_tx, stream_is_telegram) = if msg.channel == "telegram" {
                    let bot_token = msg
                        .metadata
                        .get("bot_token")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let chat_id_str = msg.chat_id.clone();
                    if !bot_token.is_empty() {
                        let chat_id_num: i64 = chat_id_str.parse().unwrap_or(0);
                        let (delta_tx, mut delta_rx) =
                            tokio::sync::mpsc::unbounded_channel::<String>();
                        let stream_client = reqwest::Client::new();
                        let stream_token = bot_token.clone();
                        tokio::spawn(async move {
                            crate::channels::telegram::tg_send_typing_action(
                                &stream_client,
                                &stream_token,
                                chat_id_num,
                            )
                            .await;
                            let msg_id = crate::channels::telegram::tg_send_placeholder(
                                &stream_client,
                                &stream_token,
                                chat_id_num,
                            )
                            .await;
                            let Some(message_id) = msg_id else {
                                while delta_rx.recv().await.is_some() {}
                                return;
                            };
                            let mut accumulated = String::new();
                            let mut dirty = false;
                            let mut interval =
                                tokio::time::interval(std::time::Duration::from_millis(500));
                            interval
                                .set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                            loop {
                                tokio::select! {
                                    delta = delta_rx.recv() => {
                                        match delta {
                                            Some(chunk) => {
                                                accumulated.push_str(&chunk);
                                                dirty = true;
                                            }
                                            None => {
                                                if dirty && !accumulated.is_empty() {
                                                    crate::channels::telegram::tg_edit_message(
                                                        &stream_client,
                                                        &stream_token,
                                                        chat_id_num,
                                                        message_id,
                                                        &accumulated,
                                                    )
                                                    .await;
                                                }
                                                break;
                                            }
                                        }
                                    }
                                    _ = interval.tick() => {
                                        if dirty && !accumulated.is_empty() {
                                            crate::channels::telegram::tg_edit_message(
                                                &stream_client,
                                                &stream_token,
                                                chat_id_num,
                                                message_id,
                                                &accumulated,
                                            )
                                            .await;
                                            dirty = false;
                                        }
                                    }
                                }
                            }
                        });
                        (Some(delta_tx), true)
                    } else {
                        (None, false)
                    }
                } else {
                    (None, false)
                };

                let response = shared
                    .process_message(&msg, stream_tx, None, None, None)
                    .await;

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
        self.shared
            .build_local_runtime_blocks(&core, session_key)
            .await
    }

    /// Clear the LCM engine for a session (e.g. on /clear command).
    ///
    /// This resets the summary DAG and active context so stale summaries
    /// don't pollute fresh conversations after /clear.
    pub async fn clear_lcm_engine(&self, session_key: &str) {
        let mut engines = self.shared.lcm_engines.lock().await;
        if engines.remove(session_key).is_some() {
            debug!(session = %session_key, "LCM engine cleared");
        }
    }

    /// Clear the bulletin cache (e.g. on /clear command).
    pub fn clear_bulletin_cache(&self) {
        self.shared.bulletin_cache.store(Arc::new(String::new()));
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
    ) -> String {
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
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "agent_loop_tests.rs"]
mod tests;
