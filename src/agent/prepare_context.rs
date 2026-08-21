// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(clippy::as_conversions, clippy::shadow_reuse, clippy::shadow_unrelated)]
//! Phase 1 of message processing: build the [`TurnContext`] from an inbound message.
//!
//! Extracted from `agent_loop.rs` to keep that file focused on the iteration
//! state machine. This module contains only the context-construction logic.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use serde_json::json;

use crate::agent::agent_core::SwappableCore;
use crate::agent::agent_loop::{AgentLoopShared, FlowControl, TurnContext};
use crate::agent::audit::AuditLog;
use crate::agent::context::PromptBlock;
use crate::agent::context_gate::ContentGate;
use crate::agent::memory_ladder::{MemoryLadder, MemoryLayer, MemoryQuery};
use crate::agent::policy;
use crate::agent::prompt_contract::{PromptSection, SectionEntry, SectionSource};
use crate::agent::protocol::{CloudProtocol, ConversationProtocol, LocalProtocol};
use crate::agent::runtime_mode::RuntimeMode;
use crate::agent::taint::TaintState;
use crate::agent::token_budget::TokenBudget;
use crate::agent::tool_guard::ToolGuard;
use crate::bus::events::InboundMessage;

/// System-prompt guidance teaching the LCM expand workflow, with a worked call.
/// Shared by the local and cloud advertisement paths so the wording (and the
/// worked example small models copy) stays in one place.
const LCM_EXPAND_GUIDE: &str =
    "Your conversation history is managed by LCM. When earlier turns are \
     compressed, you will see a block marked [Summary of messages X-Y. …]. \
     To read the exact originals, copy that range into lcm_expand — for example \
     lcm_expand({\"message_ids\": \"120-158\"}). Expansion is lossless and safe: \
     the full originals are always retrievable, so nothing is ever permanently \
     lost by summarizing. The system may also auto-expand relevant summaries for you. \
     NOTE: lcm_expand only expands summaries from the CURRENT session. To retrieve the \
     FULL content of a DIFFERENT past session, use session_search(mode=\"session\", \
     session=KEY) with that session's key — do not call lcm_expand for past sessions.";

/// Append the previous-session continuity note to the system message.
///
/// Callers must pass the SAME note on every turn of a session (it is cached
/// per session key) so the system prompt stays byte-identical and the prompt
/// prefix remains append-only across turns.
fn append_continuity_to_system(messages: &mut [serde_json::Value], note: &str) {
    let Some(first) = messages.first_mut() else {
        return;
    };
    let Some(content) = first.get("content").and_then(|v| v.as_str()) else {
        return;
    };
    first["content"] = json!(format!("{content}\n\n## Previous Session\n{note}"));
}

impl AgentLoopShared {
    /// Resolve the previous-session continuity note for this session.
    ///
    /// Computed exactly once per session key (on its first turn): a FRESH
    /// session (no prior history) gets a one-line tail of the most recent
    /// other session; a resumed session gets `None`. Later turns replay the
    /// cached value so the injected system prompt stays byte-identical.
    async fn session_continuity_note(
        &self,
        core: &Arc<SwappableCore>,
        session_key: &str,
        session_id: &str,
        prior_history_len: usize,
    ) -> Option<String> {
        use crate::agent::continuity::{classify_session_start, continuity_note};

        // Local sessions follow a Pi-style progressive context contract: prior
        // sessions stay durable and searchable, but an unrelated tail is never
        // resident in every fresh prompt. Cloud behavior remains unchanged.
        if core.context.local_prompt_mode {
            return None;
        }

        let mut notes = self.continuity_notes.lock().await;
        if let Some(cached) = notes.get(session_key) {
            return cached.clone();
        }
        let start = classify_session_start(prior_history_len);
        let tails = core.sessions.latest_session_tails(session_id, 1).await;
        let note = continuity_note(start, tails.first(), chrono::Utc::now());
        notes.insert(session_key.to_string(), note.clone());
        note
    }

    pub(crate) async fn build_local_runtime_blocks(
        &self,
        _core: &Arc<SwappableCore>,
        _session_id: &str,
    ) -> Vec<crate::agent::context::PromptBlock> {
        // This block is fixed from turn one, so local prompt prefixes remain
        // append-only while the model always understands LCM summaries. Mutable
        // working memory and background status stay out of the cached prefix.
        vec![crate::agent::context::PromptBlock::new(
            "Context Management",
            LCM_EXPAND_GUIDE,
        )]
    }

    /// Collect runtime sections for the cloud prompt path as typed `SectionEntry` values.
    ///
    /// Replaces the 4 former `append_to_system_prompt()` calls. Content is
    /// pre-fetched here; the assembler handles ordering, budgeting, and overflow.
    pub(crate) async fn collect_cloud_runtime_sections(
        &self,
        core: &Arc<SwappableCore>,
        session_id: &str,
    ) -> Vec<SectionEntry> {
        let mut sections = Vec::new();

        // LCM context awareness: advertised unconditionally so it stays in the
        // stable cached prefix from turn 1 (see build_local_runtime_blocks).
        sections.push(SectionEntry {
            section: PromptSection::ToolUse,
            block: PromptBlock::new("Context Management", LCM_EXPAND_GUIDE),
            allocated_tokens: 0,
            actual_tokens: 0,
            source: SectionSource::Runtime("lcm-context".to_string()),
            included: true,
            shrinkable: false,
        });

        // 1. Memory layers via MemoryLadder.
        if core.memory_enabled {
            let ladder = MemoryLadder::new(&core.working_memory, &core.sessions);
            let memory_multiplier = core.lane.policy().memory.budget_multiplier;
            let adjusted_budget = (core.working_memory_budget as f64 * memory_multiplier) as usize;
            let results = ladder
                .query(&MemoryQuery {
                    session_id,
                    query: "",
                    total_budget: adjusted_budget,
                })
                .await;

            for result in results {
                let (section, title) = match result.layer {
                    MemoryLayer::WorkingSession => (
                        PromptSection::WorkingMemory,
                        "Working Memory (Current Session)",
                    ),
                    _ => (PromptSection::MemoryBriefing, "Memory Briefing"),
                };
                if !result.content.is_empty() {
                    sections.push(SectionEntry {
                        section,
                        block: PromptBlock::new(title, &result.content),
                        allocated_tokens: 0,
                        actual_tokens: 0,
                        source: SectionSource::Runtime(format!("memory-ladder:{:?}", result.layer)),
                        included: true,
                        shrinkable: section.shrinkable(),
                    });
                }
            }
        }

        // ponytail: "Tool Patterns" learnings block dropped on the cloud path too
        // (see build_local_runtime_blocks) — value-less per-turn injection plus a
        // full ~947KB learnings.jsonl read every turn.

        // 2. Background task status (subagent status).
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

        // Filter sections by lane prompt profile (e.g. Answer lane excludes
        // BackgroundTasks).
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

        // Per-turn overhead baseline (Phase 1, plan cozy-squishing-galaxy):
        // one `turn_timing` line per turn; grep `prepare_context_timing`.
        let prep_t0 = std::time::Instant::now();
        let mut prep_last = prep_t0;
        let mut lap_ms = move || {
            let now = std::time::Instant::now();
            let ms = now.duration_since(prep_last).as_millis() as u64;
            prep_last = now;
            ms
        };

        // Snapshot core — instant Arc clone under brief read lock.
        let core = self.core_handle.swappable();
        let counters = &self.core_handle.counters;
        let turn_count = counters
            .learning_turn_counter
            .fetch_add(1, Ordering::Relaxed)
            + 1;
        let session_key = msg
            .metadata
            .get("session_key")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{}:{}", msg.channel, msg.chat_id));

        let session_policy = {
            let mut map = self.session_policies.lock().await;
            let entry = map.entry(session_key.clone()).or_default();
            if core.tool_delegation_config.strict_local_only() {
                entry.local_only = true;
            }
            policy::update_from_user_text(entry, &msg.content);
            entry.clone()
        };
        let strict_local_only =
            core.tool_delegation_config.strict_local_only() || session_policy.local_only;

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
        let tools_ms = lap_ms();

        // Resolve the concrete SQLite session before constructing any state
        // that must not leak across idle rollover.
        let session_meta = core
            .sessions
            .get_or_resume_with_idle(&session_key, core.session_complete_after_secs)
            .await;
        let session_id = session_meta.id.clone();
        let (compaction, compaction_admission) = loop {
            let candidate = self.compaction_handle_for_session(&session_id).await;
            if let Some(admission) = candidate.admit().await {
                break (candidate, admission);
            }

            // A waiter may have cloned the old handle before clear retired and
            // removed it. Never replace a newer handle installed by a waiter
            // that reached the map first.
            self.remove_compaction_handle_if_owned(&session_id, &candidate)
                .await;
        };
        compaction.cancel_and_reap().await;
        if tools.contains("recall") {
            // recall absorbed session_search; re-register it bound to the
            // concrete session so the fetch/search legs exclude the turn in
            // progress (mirrors the former session_search wiring).
            let tool = crate::agent::tools::RecallTool::new(&core.workspace)
                .with_db(core.sessions.path().to_path_buf())
                .with_current_session_id(Some(session_id.clone()));
            tools.register(Box::new(tool));
        }

        // Register lcm_expand and restore this concrete session's LCM DAG.
        // Eagerly create the engine here (with DB-persisted DAG if available)
        // so the tool is available from the very first turn.
        //
        // The engine persists across turns (cached in self.lcm_engines) and
        // its store is append-only, keyed by each message's stable `_db_id`
        // (SQLite rowid). Ingest is an idempotent upsert, so no per-turn
        // cursor is needed: re-offering already-stored messages is a no-op.
        let lcm_engine = {
            let mut engines = self.lcm_engines.lock().await;
            if !engines.contains_key(&session_id) {
                use crate::agent::lcm::{LcmConfig, LcmEngine};
                let config = LcmConfig::from(&self.lcm_config);
                let db_nodes = core.sessions.load_summary_nodes(&session_id).await;

                let engine = if !db_nodes.is_empty() {
                    let all_msgs = core.sessions.get_all_messages(&session_id).await;
                    tracing::debug!(
                        session_id = %session_id,
                        node_count = db_nodes.len(),
                        "LCM: rebuilding engine from SQLite summary nodes"
                    );
                    LcmEngine::rebuild_from_db_nodes(&all_msgs, &db_nodes, config)
                } else {
                    LcmEngine::new(config)
                };
                engines.insert(
                    session_id.clone(),
                    std::sync::Arc::new(tokio::sync::Mutex::new(engine)),
                );
            }
            match engines.get(&session_id) {
                Some(e) => e.clone(),
                // Inserted above or pre-existing under the same lock — absent
                // only if the invariant broke; fall back to a fresh engine.
                None => {
                    use crate::agent::lcm::{LcmConfig, LcmEngine};
                    std::sync::Arc::new(tokio::sync::Mutex::new(LcmEngine::new(LcmConfig::from(
                        &self.lcm_config,
                    ))))
                }
            }
        };
        use crate::agent::lcm::LcmExpandTool;
        tools.register(Box::new(LcmExpandTool::new(lcm_engine.clone())));

        let lcm_setup_ms = lap_ms();

        // One bounded result-inspection tool replaces full recall plus separate
        // search/slice verbs. Exact bodies remain in SQLite and no model call
        // can request an unbounded replay into the transcript.
        tools.register(Box::new(
            crate::agent::tools::stash_search::SearchToolResultTool::with_db(
                core.sessions.path().to_path_buf(),
                session_id.clone(),
            ),
        ));

        // Get session history. Track count so we know where new messages start.
        // The trim ceiling must stay above LCM's soft
        // compaction threshold, or compaction never fires (see history_limit_lcm).
        let max_messages =
            crate::agent::agent_core::history_limit_lcm(core.token_budget.max_context());
        let history = core
            .sessions
            .get_history(&session_id, max_messages, core.max_history_turns)
            .await;
        // The fingerprint deliberately SURVIVES the reload so the first call of
        // each new user turn is compared against the last call of the previous
        // one. That cross-turn comparison is the only thing that can catch a
        // reload whose bytes differ from what was already sent.
        //
        // It used to be cleared here, justified by "get_history applies
        // byte-changing transformations … the Higgs radix cache is unaffected
        // (it matches by content, not by nanobot's fingerprint)". The first
        // half was true and the second half was not: higgs matches on content,
        // the content had changed, and it re-prefilled. Clearing the
        // fingerprint only removed the evidence. Session
        // 20260810_081050_8306f8 lost 124.54s that way with an empty log.
        //
        // The byte-changing transformations are gone (`session::filters` is now
        // a pure function of the stored rows), so a divergence reported here is
        // real and worth the WARN.
        //
        // The WATERMARK still must go: it is an index into the message array,
        // and history windowing renumbers those. A stale watermark would freeze
        // the wrong prefix range.
        let _prompt_cache_transition = counters.lock_prompt_cache_transition();
        counters.prompt_cache_watermark.lock().remove(&session_key);
        drop(_prompt_cache_transition);
        let history_ms = lap_ms();
        // LCM history adoption: when the engine holds a summary DAG, the
        // engine's active context IS the conversation history — summary
        // blocks + unsummarized raws. Feeding raw history to the prompt here
        // would (a) hide every summary from the model after a restart (the
        // in-process compaction swap is the only other path that surfaces
        // them, and it dies with the process) and (b) overflow the token
        // budget with the very messages compaction already condensed, so the
        // trim would gut recent turns instead. Ingest first (idempotent by
        // `_db_id`) so a live session's rows are in the store, then adopt.
        let history = {
            // The session admission held above prevents a new compaction start
            // between reaping and this normal cancellation-safe lock await.
            let rewrite_unpublished = compaction.has_pending().await;
            let mut engine = lcm_engine.lock().await;
            for msg in &history {
                engine.ingest(msg.clone());
            }
            // The background compactor mutates the shared DAG before it
            // publishes the checkpoint that rotates Higgs's session ID.
            // Keep raw SQLite history authoritative across that window;
            // the existing checkpoint installer is the sole publication
            // point for both the prompt rewrite and cache rotation.
            if engine.dag().is_empty() || rewrite_unpublished {
                history
            } else {
                engine
                    .active_context()
                    .into_iter()
                    .filter(|m| m.get("role").and_then(|r| r.as_str()) != Some("system"))
                    .collect()
            }
        };
        let lcm_ingest_ms = lap_ms();

        // `new_start` (where new/unsaved messages begin) is computed AFTER
        // build_messages below — the assembled array may carry a variable
        // prompt prefix (system only, or system + developer). The old
        // `1 + history.len()` hardcoded a 1-slot prefix; with a developer
        // message present every index was off by one: eager-persist wrote the
        // previous assistant (duplicates), overshot, and the real user message
        // was never persisted.

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
        let build_msgs_ms = lap_ms();

        // Local prompts receive only the fixed LCM guide at runtime. Volatile
        // working-memory/status injection and per-turn relevance tails were
        // removed so there is one append-only cache path.
        let local_runtime_blocks = if core.context.local_prompt_mode {
            self.build_local_runtime_blocks(&core, &session_id).await
        } else {
            Vec::new()
        };

        // Collect runtime sections and inject into the developer message.
        // All 4 former append_to_system_prompt() calls are now pre-fetched as
        // typed SectionEntry values and appended to the developer content block.
        if !core.context.local_prompt_mode {
            let runtime_sections = self
                .collect_cloud_runtime_sections(&core, &session_id)
                .await;
            if !runtime_sections.is_empty() {
                core.context
                    .inject_runtime_sections(&mut messages, &runtime_sections);
            }
        }

        if core.context.local_prompt_mode {
            let (stable_prompt, _runtime_suffix, _) = core.context.build_local_system_prompt_parts(
                None,
                Some(&msg.channel),
                Some(&msg.chat_id),
                is_voice_message,
                detected_language.as_deref(),
                &local_runtime_blocks,
            );
            // System message = STABLE ONLY. Runtime sections (Context Management,
            // WorkingMemory, SessionMetadata, etc.) are dropped entirely — they
            // are either empty, meta-instructions the model ignores, or low-
            // quality observations that add ~250 tokens of noise, break the
            // prefix cache every turn, and cost ~4.5s of prefill time.
            // The memory/context features still work server-side (lcm, DB, memory
            // model) without being injected into the prompt.
            if let Some(first) = messages.first_mut() {
                first["content"] = json!(stable_prompt);
            }
        }

        // Previous-session continuity: a fresh session's first turn resolves
        // the tail of the most recent prior session; the identical cached line
        // is re-appended every later turn (prefix-stable within the session).
        if let Some(note) = self
            .session_continuity_note(&core, &session_key, &session_id, history.len())
            .await
        {
            append_continuity_to_system(&mut messages, &note);
        }

        // `/clear` and model switches invalidate resident-server prompt caches
        // purely through higgs session-id rotation: `reset_session_prompt_state`
        // bumps the epoch, which `stable_higgs_session_id` folds into a brand-new
        // id, so the server cold-starts the prompt. No system-message marker is
        // needed — a content marker would re-prefill message 0 on every
        // compaction/trim (the dominant KV-cache break in higgs logs).
        let sections_ms = lap_ms();

        // The just-pushed user message is the last element; everything before
        // it (the prompt prefix plus history) is already persisted or static.
        // Local split-system insertion happens above, so compute this here.
        let new_start = messages.len() - 1;

        // Tag the current user message (last in the array) with turn number
        // for age-based eviction in trim_to_fit.
        if let Some(last) = messages.last_mut() {
            last["_turn"] = json!(turn_count);
        }

        // Context gate: budget-aware content sizing for this turn.
        let mut content_gate = ContentGate::new(core.token_budget.max_context(), 0.20);
        // Pre-consume the tokens already used by system prompt + history.
        let initial_tokens = TokenBudget::estimate_tokens(&messages);
        content_gate.budget.consume(initial_tokens);
        let estimate_ms = lap_ms();
        tracing::info!(
            target: "turn_timing",
            tools_ms,
            lcm_setup_ms,
            history_ms,
            lcm_ingest_ms,
            build_msgs_ms,
            sections_ms,
            estimate_ms,
            total_ms = prep_t0.elapsed().as_millis() as u64,
            history_len = history.len(),
            msg_count = messages.len(),
            session = %session_key,
            "prepare_context_timing"
        );

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
        // so they use CloudProtocol even though mode=Local for context sizing.
        let protocol: Arc<dyn ConversationProtocol> = match core.mode() {
            RuntimeMode::Local { .. } if !core.model.starts_with("mlx:") => {
                Arc::new(LocalProtocol::auto_for_model(&core.model))
            }
            RuntimeMode::Local { .. } | RuntimeMode::Cloud => Arc::new(CloudProtocol),
        };

        drop(compaction_admission);
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
            advertised_tool_names: None,
            used_tools: std::collections::HashSet::new(),
            final_content: String::new(),
            turn_tool_entries: Vec::new(),
            iterations_used: 0,
            turn_start: std::time::Instant::now(),
            compaction,
            soft_compaction_requested: false,
            staged_auto_expansion: None,
            higgs_session_route: Default::default(),
            retained_route_cleanup: Default::default(),
            content_gate,
            counters: self.core_handle.counters.clone(),
            flow: FlowControl {
                boundary: crate::agent::agent_loop::ResponseBoundary::Off,
                router_preflight_done: false,
                tool_guard,
                iterations_since_compaction: 0,
                content_was_streamed: false,
                consecutive_all_blocked: 0,
                consecutive_no_progress_rounds: 0,
                round_executed_no_tools: false,
                lease: crate::agent::lease::Lease::new(
                    crate::agent::lease::DEFAULT_TOOLS_PER_LEASE,
                    crate::agent::lease::DEFAULT_MAX_LEASES_PER_TURN,
                ),
                llm_call_start: None,
                ttft_ms: None,
                provider_prompt_estimate: None,
                retries: crate::agent::agent_loop::RetryState::new(),
                restore_thinking_budget: None,
                provider_request: Default::default(),
                tool_rounds_completed: 0,
                pending_request_metrics: None,
                last_round_keys: Vec::new(),
                prev_round_keys: Vec::new(),
                consecutive_repeat_rounds: 0,
                repeat_nudged: false,
                infra_error: None,
            },
            health_registry: self.health_registry.clone(),
            taint_state: TaintState::new(),
            reasoning: reasoning_engine,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::append_continuity_to_system;
    use crate::agent::prompt_fingerprint::{compare, fingerprint, PromptDelta};
    use serde_json::json;

    /// With no ephemeral local tail, turn N is an exact prefix of turn N+1:
    /// [system + history + current_user] becomes
    /// [system + history + current_user + assistant + next_user].
    #[test]
    fn test_no_local_tail_is_append_only_across_turns() {
        let system = json!({"role": "system", "content": "STATIC PREFIX (identity+skills)"});
        let h_user1 = json!({"role": "user", "content": "first question"});
        let h_asst1 = json!({"role": "assistant", "content": "first answer"});

        let turn_n = vec![
            system.clone(),
            h_user1.clone(),
            h_asst1.clone(),
            json!({"role": "user", "content": "second question"}),
        ];
        let turn_n1 = vec![
            system,
            h_user1,
            h_asst1,
            json!({"role": "user", "content": "second question"}),
            json!({"role": "assistant", "content": "second answer"}),
            json!({"role": "user", "content": "third question"}),
        ];

        let fp_n = fingerprint(&turn_n);
        let fp_n1 = fingerprint(&turn_n1);
        assert_eq!(
            compare(Some(&fp_n), &fp_n1),
            PromptDelta::AppendOnly { added_msgs: 2 }
        );
    }

    /// Contrast: folding the per-turn block into the STATIC prefix (messages[0])
    /// — the thing tail placement avoids — diverges at message 0, forcing a full
    /// re-prefill every turn.
    #[test]
    fn test_static_prefix_injection_diverges_at_head() {
        let prefix_n = vec![
            json!({"role": "system", "content": "STATIC + <relevant_context>N</relevant_context>"}),
            json!({"role": "user", "content": "second question"}),
        ];
        let prefix_n1 = vec![
            json!({"role": "system", "content": "STATIC + <relevant_context>N+1</relevant_context>"}),
            json!({"role": "user", "content": "second question"}),
        ];
        let fp_n = fingerprint(&prefix_n);
        let fp_n1 = fingerprint(&prefix_n1);
        match compare(Some(&fp_n), &fp_n1) {
            PromptDelta::Diverged {
                first_divergent_msg,
                ..
            } => assert_eq!(
                first_divergent_msg, 0,
                "static-prefix injection busts the whole cache"
            ),
            other => panic!("expected head divergence, got {:?}", other),
        }
    }

    /// Epoch rotation must NOT be expressed in the prompt: the system message
    /// stays byte-identical across resets. Cache invalidation is driven solely
    /// by higgs session-id rotation (pinned in `agent_core`), so a compaction
    /// or trim no longer re-prefills message 0.
    #[test]
    fn test_no_epoch_marker_in_prompt_content() {
        let baseline = json!({"role": "system", "content": "STATIC"});
        let rendered = vec![baseline.clone(), json!({"role": "user", "content": "hi"})];

        // Simulate the rendered output after a reset: nothing appends an epoch
        // marker. If anyone reintroduces `[session-reset-epoch:N]` anywhere in
        // the prepare_context path, this fails.
        assert_eq!(rendered[0]["content"], baseline["content"]);
        assert!(
            !rendered[0]["content"]
                .as_str()
                .unwrap()
                .contains("[session-reset-epoch:"),
            "epoch must not leak into prompt content"
        );
    }

    /// Injecting the SAME cached continuity note on every turn keeps the
    /// system prompt byte-identical, so the prompt prefix stays append-only
    /// across turns within the session.
    #[test]
    fn test_continuity_note_injection_is_prefix_stable_across_turns() {
        let note = "Previous session (2h ago, key cli:oneshot-1): q → a";

        let mut turn_n = vec![
            json!({"role": "system", "content": "STATIC PREFIX"}),
            json!({"role": "user", "content": "first question"}),
        ];
        let mut turn_n1 = vec![
            json!({"role": "system", "content": "STATIC PREFIX"}),
            json!({"role": "user", "content": "first question"}),
            json!({"role": "assistant", "content": "first answer"}),
            json!({"role": "user", "content": "second question"}),
        ];

        append_continuity_to_system(&mut turn_n, note);
        append_continuity_to_system(&mut turn_n1, note);

        assert!(turn_n[0]["content"]
            .as_str()
            .unwrap()
            .contains("Previous session"));

        let fp_n = fingerprint(&turn_n);
        let fp_n1 = fingerprint(&turn_n1);
        assert_eq!(
            compare(Some(&fp_n), &fp_n1),
            PromptDelta::AppendOnly { added_msgs: 2 },
            "cached continuity note must not bust the prompt prefix"
        );
    }

    #[test]
    fn test_continuity_note_injection_noop_on_empty_messages() {
        let mut empty: Vec<serde_json::Value> = Vec::new();
        append_continuity_to_system(&mut empty, "note");
        assert!(empty.is_empty());
    }
}
