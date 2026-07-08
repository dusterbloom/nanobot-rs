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
use crate::agent::memory_ladder::{LayerResult, MemoryLadder, MemoryLayer, MemoryQuery};
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
     compressed, you will see a block marked [Summary of messages X-Y (IDs: …)]. \
     To read the exact originals, call lcm_expand with those IDs — for example \
     lcm_expand({\"message_ids\": [5, 6, 7, 8]}). Expansion is lossless and safe: \
     the full originals are always retrievable, so nothing is ever permanently \
     lost by summarizing. The system may also auto-expand relevant summaries for you.";

/// Query the memory ladder for the local prompt path, capped to the local
/// working-memory budget (the same `(working_memory_budget * multiplier).min(200)`
/// formula the always-on prefix path has always used).
///
/// Shared by the legacy always-on prefix injection (`build_local_runtime_blocks`,
/// `query = ""`) and the per-turn tail block (`query` = the user's turn) so the
/// two cannot drift. Returns an empty vec when memory is disabled.
fn local_memory_results(
    core: &Arc<SwappableCore>,
    session_key: &str,
    query: &str,
) -> Vec<LayerResult> {
    if !core.memory_enabled {
        return Vec::new();
    }
    let ladder = MemoryLadder::new(&core.workspace, &core.working_memory, None, &core.sessions);
    let memory_multiplier = core.lane.policy().memory.budget_multiplier;
    let adjusted_budget =
        ((core.working_memory_budget as f64 * memory_multiplier) as usize).min(200);
    ladder.query(&MemoryQuery {
        session_key,
        query,
        total_budget: adjusted_budget,
    })
}

/// Reduce a user turn to a compact retrieval query: trimmed and truncated to a
/// char boundary. Relevance comes from the opening of the turn, not a multi-KB
/// paste, and a bounded query keeps embedding/search cheap.
fn turn_query(content: &str) -> String {
    const MAX: usize = 256;
    let trimmed = content.trim();
    if trimmed.len() <= MAX {
        return trimmed.to_string();
    }
    let end = crate::utils::helpers::floor_char_boundary(trimmed, MAX);
    trimmed[..end].to_string()
}

/// Build the per-turn local tail block: query-relevant skills (compact
/// one-liners) + query-aware memory, rendered as a single `<relevant_context>`
/// string. Returns an empty string when nothing relevant is found.
fn build_local_tail(core: &Arc<SwappableCore>, session_key: &str, query: &str) -> String {
    let mut parts: Vec<String> = Vec::new();

    // Relevant skills — names + descriptions; the model reads full SKILL.md on
    // demand via read_skill. The static name index still lists every skill.
    let skill_names = core.context.skills.relevant(query, 3);
    tracing::debug!(
        ?skill_names,
        query_len = query.len(),
        "local_tail: query-relevant skills selected"
    );
    let skill_lines = core.context.skills.compact_lines(&skill_names);
    if !skill_lines.is_empty() {
        parts.push(format!("## Possibly-relevant skills\n{}", skill_lines));
    }

    // Query-aware memory (Scratch session-search is query-aware even without the
    // `semantic` feature; SearchIndex/KG layers add embedding/graph recall when
    // their features are on).
    let mut mem_lines: Vec<String> = Vec::new();
    for result in local_memory_results(core, session_key, query) {
        if result.content.is_empty() {
            continue;
        }
        let title = match result.layer {
            MemoryLayer::WorkingSession => "Working Memory",
            _ => "Memory",
        };
        mem_lines.push(format!("### {}\n{}", title, result.content));
    }
    if !mem_lines.is_empty() {
        parts.push(format!(
            "## Possibly-relevant memory\n{}",
            mem_lines.join("\n\n")
        ));
    }

    if parts.is_empty() {
        return String::new();
    }
    format!(
        "<relevant_context>\n{}\n</relevant_context>",
        parts.join("\n\n")
    )
}

fn local_tail_enabled() -> bool {
    std::env::var("NANOBOT_LOCAL_TAIL")
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

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

fn apply_session_prompt_epoch(messages: &mut [serde_json::Value], epoch: u64) {
    if epoch == 0 {
        return;
    }
    let Some(first) = messages.first_mut() else {
        return;
    };
    let Some(content) = first.get("content").and_then(|v| v.as_str()) else {
        return;
    };
    first["content"] = json!(format!("{content}\n\n[session-reset-epoch:{epoch}]"));
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
        core: &Arc<SwappableCore>,
        session_key: &str,
    ) -> Vec<crate::agent::context::PromptBlock> {
        let mut blocks = Vec::new();

        // LCM context awareness for local models.
        //
        // Advertised unconditionally (not gated on has_summaries) so the block
        // lives in the stable system-prompt prefix from turn 1. Adding it later,
        // once a summary appears, would rewrite the prefix and bust the prompt
        // cache — the exact failure the old cache_warm gate tried to avoid, at
        // the cost of leaving small models staring at an unexplained [Summary…]
        // block. A fixed ~50-token block in the cold prefix is the cheaper trade.
        if self.lcm_enabled.load(Ordering::Relaxed) {
            blocks.push(crate::agent::context::PromptBlock::new(
                "Context Management",
                LCM_EXPAND_GUIDE,
            ));
        }

        // Only durable, cross-session memory ("Memory Briefing") goes in the
        // cold prefix — it's stable, so it stays in the cached prefix.
        //
        // ponytail: the WorkingSession layer is deliberately excluded. It mutates
        // every turn (the background reflector rewrites it) and accretes across a
        // session, which (a) bloats the cold prefix and (b) is the one thing that
        // can bust a warm prefix cache mid-session. The live conversation already
        // carries the session's content; per-turn relevance comes from the
        // query-aware tail (NANOBOT_LOCAL_TAIL) and the `recall` tool, not an
        // always-on dump.
        for result in local_memory_results(core, session_key, "") {
            if result.content.is_empty() || matches!(result.layer, MemoryLayer::WorkingSession) {
                continue;
            }
            blocks.push(crate::agent::context::PromptBlock::new(
                "Memory Briefing",
                &result.content,
            ));
        }

        // ponytail: dropped the "Tool Patterns" learnings block. It injected
        // vague global success/failure counts ("- exec: 8/10 succeeded recently")
        // with no measurable value, and get_learning_context() read the entire
        // ~947KB learnings.jsonl every turn to keep the last 50. Recording stays;
        // only this value-less per-turn prompt injection is removed.

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

        // LCM context awareness: advertised unconditionally so it stays in the
        // stable cached prefix from turn 1 (see build_local_runtime_blocks).
        if self.lcm_enabled.load(Ordering::Relaxed) {
            sections.push(SectionEntry {
                section: PromptSection::ToolUse,
                block: PromptBlock::new("Context Management", LCM_EXPAND_GUIDE),
                allocated_tokens: 0,
                actual_tokens: 0,
                source: SectionSource::Runtime("lcm-context".to_string()),
                included: true,
                shrinkable: false,
            });
        }

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
            let results = ladder.query(&MemoryQuery {
                session_key,
                query: "",
                total_budget: adjusted_budget,
            });

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

        // 2. Recent daily notes (cloud mode, local-backend only).
        if core.mode().is_local() && core.memory_enabled {
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
        // Eagerly create the engine here (with DB-persisted DAG if available)
        // so the tool is available from the very first turn.
        //
        // The engine persists across turns (cached in self.lcm_engines) and
        // its store is append-only, keyed by each message's stable `_db_id`
        // (SQLite rowid). Ingest is an idempotent upsert, so no per-turn
        // cursor is needed: re-offering already-stored messages is a no-op.
        if self.lcm_enabled.load(Ordering::Relaxed) {
            let lcm_engine = {
                let mut engines = self.lcm_engines.lock().await;
                if !engines.contains_key(&session_key) {
                    use crate::agent::lcm::{LcmConfig, LcmEngine};
                    let config = LcmConfig::from(&self.lcm_config);

                    // Try to restore DAG from SQLite summary_nodes table.
                    let session_meta_tmp = core
                        .sessions
                        .get_or_resume_with_idle(&session_key, core.session_complete_after_secs)
                        .await;
                    let db_nodes = core.sessions.load_summary_nodes(&session_meta_tmp.id).await;

                    let engine = if !db_nodes.is_empty() {
                        // Restore from DB-persisted DAG + raw messages.
                        let all_msgs = core.sessions.get_all_messages(&session_meta_tmp.id).await;
                        tracing::debug!(
                            session = %session_key,
                            node_count = db_nodes.len(),
                            "LCM: rebuilding engine from DB summary nodes"
                        );
                        LcmEngine::rebuild_from_db_nodes(&all_msgs, &db_nodes, config)
                    } else {
                        // No persisted DAG. Legacy `role:"summary"` rows in the
                        // messages table carry positional source_ids that
                        // cannot be resolved against the db-id-keyed store, so
                        // they are deliberately NOT loaded into the DAG (their
                        // text remains in session history).
                        LcmEngine::new(config)
                    };

                    engines.insert(
                        session_key.clone(),
                        std::sync::Arc::new(tokio::sync::Mutex::new(engine)),
                    );
                }
                engines.get(&session_key).cloned().unwrap()
            };
            use crate::agent::lcm::LcmExpandTool;
            tools.register(Box::new(LcmExpandTool::new(lcm_engine)));
        }

        // Resolve or create session for this key.
        // If the session has been idle longer than session_complete_after_secs,
        // start fresh instead of loading stale history into the context.
        let session_meta = core
            .sessions
            .get_or_resume_with_idle(&session_key, core.session_complete_after_secs)
            .await;
        let session_id = session_meta.id.clone();

        // Get session history. Track count so we know where new messages start.
        // With LCM enabled the trim ceiling must stay above LCM's soft
        // compaction threshold, or compaction never fires (see history_limit_lcm).
        let max_messages = if self.lcm_enabled.load(Ordering::Relaxed) {
            crate::agent::agent_core::history_limit_lcm(core.token_budget.max_context())
        } else {
            history_limit(core.token_budget.max_context())
        };
        let history = core
            .sessions
            .get_history(&session_id, max_messages, core.max_history_turns)
            .await;
        // LCM history adoption: when the engine holds a summary DAG, the
        // engine's active context IS the conversation history — summary
        // blocks + unsummarized raws. Feeding raw history to the prompt here
        // would (a) hide every summary from the model after a restart (the
        // in-process compaction swap is the only other path that surfaces
        // them, and it dies with the process) and (b) overflow the token
        // budget with the very messages compaction already condensed, so the
        // trim would gut recent turns instead. Ingest first (idempotent by
        // `_db_id`) so a live session's rows are in the store, then adopt.
        let history = if self.lcm_enabled.load(Ordering::Relaxed) {
            let engine_arc = self.lcm_engines.lock().await.get(&session_key).cloned();
            match engine_arc {
                Some(engine_arc) => {
                    let mut engine = engine_arc.lock().await;
                    for msg in &history {
                        engine.ingest(msg.clone());
                    }
                    if engine.dag().is_empty() {
                        history
                    } else {
                        engine
                            .active_context()
                            .into_iter()
                            .filter(|m| {
                                m.get("role").and_then(|r| r.as_str()) != Some("system")
                            })
                            .collect()
                    }
                }
                None => history,
            }
        } else {
            history
        };

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
        // The just-pushed user message is the last element; everything before
        // it (any prompt prefix + history) is already persisted or ephemeral.
        // Mutable because the per-turn local tail block (inserted before the
        // user message) bumps it so the tail is never written to history.
        let mut new_start = messages.len() - 1;

        // Append-only local prompt (default): keep per-turn-volatile runtime
        // blocks (Working Memory, Background Tasks, Tool Patterns, LCM hints)
        // OUT of the system prefix so turn N stays a byte-prefix of turn N+1 and
        // Higgs reuses the prior turn's KV cache (flat TTFT on long sessions —
        // measured ~1.9s full prefill → ~0.07s reuse). Long-term MEMORY.md stays
        // in the static prefix (build_developer_context); recent context comes
        // from the append-only history; older/cross-session facts via `recall`.
        // Opt back into always-on injection with NANOBOT_LOCAL_ALWAYS_ON_MEMORY=1.
        let local_runtime_blocks = if core.context.local_prompt_mode
            && std::env::var("NANOBOT_LOCAL_ALWAYS_ON_MEMORY").is_ok()
        {
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

        // Previous-session continuity: a fresh session's first turn resolves
        // the tail of the most recent prior session; the identical cached line
        // is re-appended every later turn (prefix-stable within the session).
        if let Some(note) = self
            .session_continuity_note(&core, &session_key, &session_id, history.len())
            .await
        {
            append_continuity_to_system(&mut messages, &note);
        }

        // `/clear` and model switches must invalidate resident-server prompt
        // caches even when the user starts with identical text (`hi` after
        // `hi`). Keep the marker stable within an epoch so later turns remain
        // append-only, but make each new epoch a cold prompt prefix.
        apply_session_prompt_epoch(&mut messages, counters.session_prompt_epoch(&session_key));

        // Optional per-turn query-aware TAIL block (local only): relevant skills
        // + memory placed AFTER history, immediately before the user message.
        //
        // This is opt-in because local protocol merges consecutive user messages;
        // if the tail is excluded from session persistence, turn N sends
        // "tail + user" while turn N+1 replays only "user", forcing a prefix-cache
        // divergence at the prior user message every turn. Exact Higgs/LM Studio
        // cache reuse is the default; set NANOBOT_LOCAL_TAIL=1 to trade that for
        // per-turn retrieved context. `new_start` is bumped so the tail remains
        // ephemeral when enabled.
        if core.context.local_prompt_mode
            && std::env::var("NANOBOT_LOCAL_ALWAYS_ON_MEMORY").is_err()
            && local_tail_enabled()
        {
            let query = turn_query(&msg.content);
            let tail = build_local_tail(&core, &session_key, &query);
            if !tail.is_empty() {
                tracing::debug!(
                    session = %session_key,
                    tail_tokens = crate::agent::token_budget::TokenBudget::estimate_str_tokens(&tail),
                    at_msg = messages.len().saturating_sub(1),
                    "local_tail_injected — relevant skills+memory before user msg"
                );
                core.context.insert_tail_before_user(&mut messages, &tail);
                new_start += 1;
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
        // so they use CloudProtocol even though mode=Local for context sizing.
        let protocol: Arc<dyn ConversationProtocol> = match core.mode() {
            RuntimeMode::Local { .. } if !core.model.starts_with("mlx:") => {
                Arc::new(LocalProtocol::auto_for_model(&core.model))
            }
            RuntimeMode::Local { .. } | RuntimeMode::Cloud => Arc::new(CloudProtocol),
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
            counters: self.core_handle.counters.clone(),
            flow: FlowControl {
                boundary: crate::agent::agent_loop::ResponseBoundary::Off,
                router_preflight_done: false,
                tool_guard,
                iterations_since_compaction: 0,
                content_was_streamed: false,
                consecutive_all_blocked: 0,
                round_executed_no_tools: false,
                llm_call_start: None,
                ttft_ms: None,
                retries: crate::agent::agent_loop::RetryState::new(),
                restore_thinking_budget: None,
            },
            health_registry: self.health_registry.clone(),
            taint_state: TaintState::new(),
            reasoning: reasoning_engine,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{append_continuity_to_system, apply_session_prompt_epoch, turn_query};
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

    /// Tail placement is still useful as an opt-in relevance tradeoff, but the
    /// local protocol merge means the prior user message changes on replay.
    /// This documents why the local tail is not the default cache path.
    #[test]
    fn test_local_tail_diverges_at_prior_user_message() {
        let system = json!({"role": "system", "content": "STATIC PREFIX (identity+skills)"});
        let h_user1 = json!({"role": "user", "content": "first question"});
        let h_asst1 = json!({"role": "assistant", "content": "first answer"});

        // Turn N rendered wire: tail merged into the user message (LocalProtocol).
        let turn_n = vec![
            system.clone(),
            h_user1.clone(),
            h_asst1.clone(),
            json!({"role": "user", "content": "<relevant_context>N</relevant_context>\n\nsecond question"}),
        ];
        // Turn N+1: turn-N's user/assistant are now RAW history; a fresh tail
        // precedes the new user message.
        let turn_n1 = vec![
            system.clone(),
            h_user1.clone(),
            h_asst1.clone(),
            json!({"role": "user", "content": "second question"}),
            json!({"role": "assistant", "content": "second answer"}),
            json!({"role": "user", "content": "<relevant_context>N+1</relevant_context>\n\nthird question"}),
        ];

        let fp_n = fingerprint(&turn_n);
        let fp_n1 = fingerprint(&turn_n1);
        match compare(Some(&fp_n), &fp_n1) {
            PromptDelta::Diverged {
                first_divergent_msg,
                ..
            } => {
                // Indices 0,1,2 (system + older history) are byte-identical and
                // stay cached, but the previous user message itself re-prefills.
                assert_eq!(
                    first_divergent_msg, 3,
                    "ephemeral local tail makes the prior user message diverge"
                );
            }
            other => panic!("expected Diverged at the tail, got {:?}", other),
        }
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

    #[test]
    fn test_session_prompt_epoch_busts_prompt_prefix_after_clear() {
        let mut before = vec![
            json!({"role": "system", "content": "STATIC"}),
            json!({"role": "user", "content": "hi"}),
        ];
        let mut after = before.clone();

        apply_session_prompt_epoch(&mut before, 0);
        apply_session_prompt_epoch(&mut after, 1);

        assert_eq!(before[0]["content"], "STATIC");
        assert!(after[0]["content"]
            .as_str()
            .unwrap()
            .contains("[session-reset-epoch:1]"));

        let fp_before = fingerprint(&before);
        let fp_after = fingerprint(&after);
        match compare(Some(&fp_before), &fp_after) {
            PromptDelta::Diverged {
                first_divergent_msg,
                ..
            } => assert_eq!(
                first_divergent_msg, 0,
                "clear/model switch epoch must invalidate the stale prompt head"
            ),
            other => panic!("expected epoch marker to diverge at the prompt head, got {other:?}"),
        }
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

    #[test]
    fn test_turn_query_passes_short_input_trimmed() {
        assert_eq!(
            turn_query("  how do I parse JSON?  "),
            "how do I parse JSON?"
        );
    }

    #[test]
    fn test_turn_query_truncates_long_input() {
        let long = "a".repeat(1000);
        let q = turn_query(&long);
        assert!(
            q.len() <= 256,
            "query should be capped at 256, got {}",
            q.len()
        );
        assert!(
            q.len() >= 250,
            "should keep most of the cap, got {}",
            q.len()
        );
    }

    #[test]
    fn test_turn_query_truncates_on_char_boundary() {
        // Multibyte chars must not be split mid-byte (would panic on slice).
        let s = "é".repeat(500); // 2 bytes each → 1000 bytes
        let q = turn_query(&s);
        assert!(q.len() <= 256);
        // Re-encoding round-trips cleanly (valid UTF-8, no partial char).
        assert!(q.chars().all(|c| c == 'é'));
    }

    #[test]
    fn test_turn_query_empty() {
        assert_eq!(turn_query("   "), "");
    }
}
