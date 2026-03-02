//! Phase 1 of message processing: build the [`TurnContext`] from an inbound message.
//!
//! Extracted from `agent_loop.rs` to keep that file focused on the iteration
//! state machine. This module contains only the context-construction logic.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use serde_json::json;

use crate::agent::agent_core::{append_to_system_prompt, history_limit};
use crate::agent::agent_loop::{AgentLoopShared, CompactionHandle, FlowControl, TurnContext};
use crate::agent::audit::AuditLog;
use crate::agent::context_gate::ContentGate;
use crate::agent::policy;
use crate::agent::protocol::{CloudProtocol, ConversationProtocol, LocalProtocol};
use crate::agent::taint::TaintState;
use crate::agent::token_budget::TokenBudget;
use crate::agent::tool_guard::ToolGuard;
use crate::bus::events::InboundMessage;

impl AgentLoopShared {
    /// Phase 1: Build the [`TurnContext`] from the inbound message.
    ///
    /// Snapshots the swappable core, extracts session info, builds tools,
    /// loads history, constructs the message array, and initialises all
    /// per-turn tracking state.
    pub(crate) async fn prepare_context(
        &self,
        msg: &InboundMessage,
        text_delta_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
        tool_event_tx: Option<tokio::sync::mpsc::UnboundedSender<crate::agent::audit::ToolEvent>>,
        cancellation_token: Option<tokio_util::sync::CancellationToken>,
        priority_rx: Option<tokio::sync::mpsc::UnboundedReceiver<String>>,
    ) -> TurnContext {
        let streaming = text_delta_tx.is_some();

        // Snapshot core — instant Arc clone under brief read lock.
        let core = self.core_handle.swappable();
        let counters = &self.core_handle.counters;
        let turn_count = counters
            .learning_turn_counter
            .fetch_add(1, Ordering::Relaxed)
            + 1;
        if turn_count % 50 == 0 {
            core.learning.prune();
        }

        let session_key = msg
            .metadata
            .get("session_key")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{}:{}", msg.channel, msg.chat_id));

        let session_policy = {
            let mut map = self.session_policies.lock().await;
            let entry = map.entry(session_key.clone()).or_default();
            if core.tool_delegation_config.strict_local_only {
                entry.local_only = true;
            }
            policy::update_from_user_text(entry, &msg.content);
            entry.clone()
        };
        let strict_local_only =
            core.tool_delegation_config.strict_local_only || session_policy.local_only;

        tracing::debug!(
            "Processing message{} from {} on {}: {}",
            if streaming { " (streaming)" } else { "" },
            msg.sender_id,
            msg.channel,
            &msg.content[..msg.content.len().min(80)]
        );

        // Create audit log if provenance is enabled.
        let audit = if core.provenance_config.enabled && core.provenance_config.audit_log {
            Some(AuditLog::new(&core.workspace, &session_key))
        } else {
            None
        };

        // Build per-message tools with context baked in.
        // The reasoning engine is returned alongside the registry so it can be
        // stored in TurnContext for plan-guided execution and backtracking.
        let (mut tools, reasoning_engine) =
            self.build_tools(&core, &msg.channel, &msg.chat_id).await;

        // Register lcm_expand tool when LCM is enabled.
        // Bug 4 fix: do NOT insert a fresh engine here. Only look up an
        // existing engine so that step_pre_call's rebuild_from_turns path
        // is not bypassed on session restart. If no engine exists yet, skip
        // registration — step_pre_call will create/rebuild the engine on the
        // first iteration and the tool will be available from the next turn.
        if self.lcm_config.enabled {
            let lcm_engine = {
                let engines = self.lcm_engines.lock().await;
                engines.get(&session_key).cloned()
            };
            if let Some(engine) = lcm_engine {
                use crate::agent::lcm::LcmExpandTool;
                tools.register(Box::new(LcmExpandTool::new(engine)));
            }
            // Engine will be created/rebuilt in step_pre_call if needed.
        }

        // Resolve or create session for this key.
        let session_meta = core.sessions.get_or_resume(&session_key).await;
        let session_id = session_meta.id.clone();

        // Get session history. Track count so we know where new messages start.
        let history = core
            .sessions
            .get_history(
                &session_id,
                history_limit(core.token_budget.max_context()),
                core.max_history_turns,
            )
            .await;
        // Track where new (unsaved) messages start. Updated after compaction
        // swaps to avoid re-persisting already-saved messages.
        let new_start = 1 + history.len();

        // Extract media paths.
        let media_paths: Vec<String> = msg
            .metadata
            .get("media")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        // Build messages.
        let is_voice_message = msg
            .metadata
            .get("voice_message")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let detected_language = msg
            .metadata
            .get("detected_language")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        // /no_think is handled at the provider level (system prompt via native
        // LMS API for Nemotron) — never inject it into user content where the
        // model treats it as literal text and leaks it into tool arguments.
        let user_content = msg.content.clone();
        let mut messages = core.context.build_messages(
            &history,
            &user_content,
            None,
            if media_paths.is_empty() {
                None
            } else {
                Some(&media_paths)
            },
            Some(&msg.channel),
            Some(&msg.chat_id),
            is_voice_message,
            detected_language.as_deref(),
        );

        // Inject per-session working memory into the system message.
        if core.memory_enabled {
            let mut wm = core
                .working_memory
                .get_context(&session_key, core.working_memory_budget);
            // Append learning context (tool patterns) if available.
            let learning_ctx = core.learning.get_learning_context();
            if !learning_ctx.is_empty() {
                wm.push_str("\n\n## Tool Patterns\n\n");
                wm.push_str(&learning_ctx);
            }
            if !wm.is_empty() {
                append_to_system_prompt(
                    &mut messages,
                    &format!("\n\n---\n\n# Working Memory (Current Session)\n\n{}", wm),
                );
            }
        }

        // Inject recent daily notes for local mode continuity.
        if core.is_local && core.memory_enabled {
            let memory_store = crate::agent::memory::MemoryStore::new(&core.workspace);
            let notes = memory_store.read_recent_daily_notes(3);
            if !notes.is_empty() {
                append_to_system_prompt(
                    &mut messages,
                    &format!("\n\n## Recent Notes\n\n{}", notes),
                );
            }
        }

        // Auto-inject background task status into system prompt so the agent
        // is naturally aware of running/completed subagents without explicit tool calls.
        {
            let running = self.subagents.list_running().await;
            let recent =
                crate::agent::subagent::SubagentManager::read_recent_completed(&core.workspace, 5);
            let status = crate::agent::subagent::format_status_block(&running, &recent);
            if !status.is_empty() {
                append_to_system_prompt(&mut messages, &status);
            }
        }

        // Inject memory bulletin if available (zero-cost Arc load).
        {
            let bulletin = self.bulletin_cache.load_full();
            if !bulletin.is_empty() {
                append_to_system_prompt(
                    &mut messages,
                    &format!("\n\n## Memory Briefing\n\n{}", &*bulletin),
                );
            }
        }

        // Tag the current user message (last in the array) with turn number
        // for age-based eviction in trim_to_fit.
        if let Some(last) = messages.last_mut() {
            last["_turn"] = json!(turn_count);
        }

        // Background compaction state.
        let compaction_slot: Arc<tokio::sync::Mutex<Option<crate::agent::agent_core::PendingCompaction>>> =
            Arc::new(tokio::sync::Mutex::new(None));
        let compaction_in_flight = Arc::new(std::sync::atomic::AtomicBool::new(false));

        // Context gate: budget-aware content sizing for this turn.
        let cache_dir = crate::utils::helpers::get_data_path()
            .join("cache")
            .join("tool_outputs");
        let mut content_gate = ContentGate::new(
            core.token_budget.max_context(),
            0.20,
            cache_dir,
        );
        // Pre-consume the tokens already used by system prompt + history.
        let initial_tokens = TokenBudget::estimate_tokens(&messages);
        content_gate.budget.consume(initial_tokens);

        let tool_guard = ToolGuard::new(core.tool_delegation_config.max_same_tool_call_per_turn);

        let request_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
        tracing::info!(
            request_id = %request_id,
            model = %core.model,
            session = %session_key,
            turn = turn_count,
            "request_started"
        );

        // Select conversation protocol based on whether we're talking to a local model.
        // Protocol correctness is enforced at render time — no repair needed.
        let protocol: Arc<dyn ConversationProtocol> = if core.is_local {
            Arc::new(LocalProtocol::auto_for_model(&core.model))
        } else {
            Arc::new(CloudProtocol)
        };

        TurnContext {
            core,
            request_id,
            session_key,
            session_id,
            session_policy,
            strict_local_only,
            turn_count,
            streaming,
            audit,
            tools,
            user_content,
            channel: msg.channel.clone(),
            chat_id: msg.chat_id.clone(),
            is_voice_message,
            detected_language,
            text_delta_tx,
            tool_event_tx,
            cancellation_token,
            priority_rx,
            messages,
            new_start,
            rendered_messages: Vec::new(),
            protocol,
            used_tools: std::collections::HashSet::new(),
            final_content: String::new(),
            turn_tool_entries: Vec::new(),
            iterations_used: 0,
            turn_start: std::time::Instant::now(),
            compaction: CompactionHandle {
                slot: compaction_slot,
                in_flight: compaction_in_flight,
            },
            content_gate,
            lcm_synced_to: None,
            counters: self.core_handle.counters.clone(),
            flow: FlowControl {
                force_response: false,
                router_preflight_done: false,
                tool_guard,
                iterations_since_compaction: 0,
                forced_finalize_attempted: false,
                content_was_streamed: false,
                consecutive_all_blocked: 0,
                llm_call_start: None,
                agent_retry_attempted: false,
                continuations_used: 0,
                validation_retries: 0,
            },
            health_registry: self.health_registry.clone(),
            taint_state: TaintState::new(),
            reasoning: reasoning_engine,
        }
    }
}
