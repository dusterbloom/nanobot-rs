//! Phase 1 of message processing: build the [`TurnContext`] from an inbound message.
//!
//! Extracted from `agent_loop.rs` to keep that file focused on the iteration
//! state machine. This module contains only the context-construction logic.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use serde_json::json;

use crate::agent::agent_core::{history_limit, SwappableCore};
use crate::agent::agent_loop::{AgentLoopShared, CompactionHandle, FlowControl, TurnContext};
use crate::agent::audit::AuditLog;
use crate::agent::context::PromptBlock;
use crate::agent::context_gate::ContentGate;
use crate::agent::memory_ladder::{MemoryLadder, MemoryLayer, MemoryQuery};
use crate::agent::policy;
use crate::agent::prompt_contract::{PromptSection, SectionEntry, SectionSource};
use crate::agent::protocol::{CloudProtocol, ConversationProtocol, LocalProtocol};
use crate::agent::taint::TaintState;
use crate::agent::token_budget::TokenBudget;
use crate::agent::tool_guard::ToolGuard;
use crate::bus::events::InboundMessage;

impl AgentLoopShared {
    pub(crate) async fn build_local_runtime_blocks(
        &self,
        core: &Arc<SwappableCore>,
        session_key: &str,
    ) -> Vec<crate::agent::context::PromptBlock> {
        let mut blocks = Vec::new();

        if core.memory_enabled {
            let ladder = MemoryLadder::new(
                &core.workspace,
                &core.working_memory,
                None,
                &core.sessions,
            );
            let memory_multiplier = core.lane.policy().memory.budget_multiplier;
            let adjusted_budget = ((core.working_memory_budget as f64 * memory_multiplier) as usize).min(200);
            let results = ladder
                .query(&MemoryQuery {
                    session_key,
                    query: "",
                    total_budget: adjusted_budget,
                });
            for result in results {
                if !result.content.is_empty() {
                    let title = match result.layer {
                        MemoryLayer::WorkingSession => "Working Memory",
                        _ => "Memory Briefing",
                    };
                    blocks.push(crate::agent::context::PromptBlock::new(
                        title,
                        &result.content,
                    ));
                }
            }
        }

        let learning_ctx = core.learning.get_learning_context();
        if !learning_ctx.is_empty() {
            blocks.push(crate::agent::context::PromptBlock::new(
                "Tool Patterns",
                learning_ctx,
            ));
        }

        let running = self.subagents.list_running().await;
        let recent =
            crate::agent::subagent::SubagentManager::read_recent_completed(&core.workspace, 5);
        let status = crate::agent::subagent::format_status_block(&running, &recent);
        if !status.is_empty() {
            blocks.push(crate::agent::context::PromptBlock::new(
                "Background Tasks",
                status,
            ));
        }

        // Filter blocks by lane prompt profile (e.g. Answer lane excludes
        // ToolPatterns and BackgroundTasks).
        let prompt_profile = core.lane.policy().prompt;
        blocks.retain(|block| {
            let section = match block.title() {
                "Tool Patterns" => PromptSection::ToolPatterns,
                "Background Tasks" => PromptSection::BackgroundTasks,
                _ => return true, // unknown title => keep
            };
            prompt_profile.includes(section)
        });

        blocks
    }

    /// Collect runtime sections for the cloud prompt path as typed `SectionEntry` values.
    ///
    /// Replaces the 4 former `append_to_system_prompt()` calls. Content is
    /// pre-fetched here; the assembler handles ordering, budgeting, and overflow.
    pub(crate) async fn collect_cloud_runtime_sections(
        &self,
        core: &Arc<SwappableCore>,
        session_key: &str,
    ) -> Vec<SectionEntry> {
        let mut sections = Vec::new();

        // 1. Memory layers via MemoryLadder (replaces direct working memory + bulletin cache).
        if core.memory_enabled {
            let ks_guard;
            let ks_ref = if let Some(ref ks_arc) = self.knowledge_store {
                ks_guard = ks_arc.lock();
                Some(&*ks_guard)
            } else {
                None
            };
            let ladder = MemoryLadder::new(
                &core.workspace,
                &core.working_memory,
                ks_ref,
                &core.sessions,
            );
            let memory_multiplier = core.lane.policy().memory.budget_multiplier;
            let adjusted_budget = (core.working_memory_budget as f64 * memory_multiplier) as usize;
            let results = ladder
                .query(&MemoryQuery {
                    session_key,
                    query: "",
                    total_budget: adjusted_budget,
                });

            for result in results {
                let (section, title) = match result.layer {
                    MemoryLayer::WorkingSession => {
                        (PromptSection::WorkingMemory, "Working Memory (Current Session)")
                    }
                    _ => (PromptSection::MemoryBriefing, "Memory Briefing"),
                };
                if !result.content.is_empty() {
                    sections.push(SectionEntry {
                        section,
                        block: PromptBlock::new(title, &result.content),
                        allocated_tokens: 0,
                        actual_tokens: 0,
                        source: SectionSource::Runtime(format!(
                            "memory-ladder:{:?}",
                            result.layer
                        )),
                        included: true,
                        shrinkable: section.shrinkable(),
                    });
                }
            }

            // Tool patterns remain as a separate section (not a memory layer).
            let learning_ctx = core.learning.get_learning_context();
            if !learning_ctx.is_empty() {
                sections.push(SectionEntry {
                    section: PromptSection::ToolPatterns,
                    block: PromptBlock::new("Tool Patterns", &learning_ctx),
                    allocated_tokens: 0,
                    actual_tokens: 0,
                    source: SectionSource::Runtime("learning patterns".to_string()),
                    included: true,
                    shrinkable: PromptSection::ToolPatterns.shrinkable(),
                });
            }
        }

        // 2. Recent daily notes (cloud mode, local-backend only).
        if core.is_local && core.memory_enabled {
            let memory_store = crate::agent::memory::MemoryStore::new(&core.workspace);
            let notes = memory_store.read_recent_daily_notes(3);
            if !notes.is_empty() {
                sections.push(SectionEntry {
                    section: PromptSection::RecentNotes,
                    block: PromptBlock::new("Recent Notes", &notes),
                    allocated_tokens: 0,
                    actual_tokens: 0,
                    source: SectionSource::File("daily notes".to_string()),
                    included: true,
                    shrinkable: PromptSection::RecentNotes.shrinkable(),
                });
            }
        }

        // 3. Background task status (subagent status).
        {
            let running = self.subagents.list_running().await;
            let recent =
                crate::agent::subagent::SubagentManager::read_recent_completed(&core.workspace, 5);
            let status = crate::agent::subagent::format_status_block(&running, &recent);
            if !status.is_empty() {
                sections.push(SectionEntry {
                    section: PromptSection::BackgroundTasks,
                    block: PromptBlock::new("Background Tasks", &status),
                    allocated_tokens: 0,
                    actual_tokens: 0,
                    source: SectionSource::Runtime("subagent status".to_string()),
                    included: true,
                    shrinkable: PromptSection::BackgroundTasks.shrinkable(),
                });
            }
        }

        // 4. Memory bulletin/briefing — replaced by GroundTruth layer in MemoryLadder above.

        // Filter sections by lane prompt profile (e.g. Answer lane excludes
        // ToolPatterns and BackgroundTasks).
        let prompt_profile = core.lane.policy().prompt;
        sections.retain(|entry| prompt_profile.includes(entry.section));

        sections
    }

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

        let local_runtime_blocks = if core.context.local_prompt_mode {
            self.build_local_runtime_blocks(&core, &session_key).await
        } else {
            Vec::new()
        };

        // Collect runtime sections and inject into the developer message.
        // All 4 former append_to_system_prompt() calls are now pre-fetched as
        // typed SectionEntry values and appended to the developer content block.
        if !core.context.local_prompt_mode {
            let runtime_sections = self
                .collect_cloud_runtime_sections(&core, &session_key)
                .await;
            if !runtime_sections.is_empty() {
                core.context
                    .inject_runtime_sections(&mut messages, &runtime_sections);
            }
        }

        if core.context.local_prompt_mode {
            let rebuilt = core.context.build_local_system_prompt(
                None,
                Some(&msg.channel),
                Some(&msg.chat_id),
                is_voice_message,
                detected_language.as_deref(),
                &local_runtime_blocks,
            );
            if let Some(first) = messages.first_mut() {
                first["content"] = json!(rebuilt);
            }
        }

        // Tag the current user message (last in the array) with turn number
        // for age-based eviction in trim_to_fit.
        if let Some(last) = messages.last_mut() {
            last["_turn"] = json!(turn_count);
        }

        // Background compaction state.
        let compaction_slot: Arc<
            tokio::sync::Mutex<Option<crate::agent::agent_core::PendingCompaction>>,
        > = Arc::new(tokio::sync::Mutex::new(None));
        let compaction_in_flight = Arc::new(std::sync::atomic::AtomicBool::new(false));

        // Context gate: budget-aware content sizing for this turn.
        let mut content_gate = ContentGate::new(core.token_budget.max_context(), 0.20);
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
        // MLX models are in-process and speak cloud protocol (proper tool_calls),
        // so they use CloudProtocol even though is_local=true for context sizing.
        let protocol: Arc<dyn ConversationProtocol> =
            if core.is_local && !core.model.starts_with("mlx:") {
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
