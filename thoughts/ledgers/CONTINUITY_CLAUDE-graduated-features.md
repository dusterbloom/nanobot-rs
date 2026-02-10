# Continuity Ledger: Graduating nanobot - Agent Resilience

## Goal
Make nanobot's agent loop production-resilient: context budget management, protocol-safe messages, efficient token usage, outcome-aware learning, bypass-resistant sandbox, smart summarization, and concurrent chat processing.

## Constraints
- No new crate dependencies (character-based token estimation, not tiktoken)
- All changes backward-compatible with existing config.json
- Each phase independently useful and testable
- Existing tests must keep passing

## Key Decisions
- **char-based estimation (~4 chars/token)**: No tokenizer dependency, good enough for budget management [2026-02-09]
- **3-stage overflow recovery**: Soft (truncate tool results) → Medium (drop old history) → Hard (system + summary + last message). Inspired by BabyAGI [2026-02-09]
- **Thread repair always-on**: Not just after crashes — trimming can also create orphans [2026-02-09]
- **Keyword-based tool selection over embedding**: Simple, fast, no ML deps. Core tools always included, others keyword-triggered [2026-02-09]
- **File-based learning (JSON), not DB**: Keeps it simple, lives alongside memory files [2026-02-09]
- **Full command + per-segment deny checking**: Full command catches cross-pipe patterns (curl|sh), per-segment catches hidden commands after semicolons [2026-02-09]
- **LLM-powered context compaction**: Summarize old messages via cheap LLM call before truncation. Falls back to hard trim on failure. Preserves key facts vs lossy truncation. [2026-02-09]
- **Per-message ToolRegistry**: Create fresh tool instances per message (bake channel/chat_id at construction) instead of shared set_context() calls. Eliminates race conditions under concurrency. [2026-02-09]
- **Arc<AgentLoopShared> extraction**: Shared state (provider, sessions, config) wrapped in Arc for concurrent task access. `run()` owns the receiver, spawned tasks share the rest. [2026-02-09]
- **Semaphore + per-session Mutex**: Semaphore limits total concurrent chats (default 4), per-session Mutex serializes messages within the same conversation. No dashmap dep — tokio::sync::Mutex<HashMap> for session locks. [2026-02-09]
- **SessionManager async locking**: Wrapped cache in tokio::sync::Mutex, all public methods take &self. New async API: get_history(), add_message_and_save(), add_messages_and_save(). [2026-02-09]

## State
- Done:
  - [x] Phase 1: Context Window Management — `token_budget.rs`, config `maxContextTokens`, 3-stage trim in agent_loop
  - [x] Phase 2: Thread Repair — `thread_repair.rs`, repair orphaned tool calls/results, merge consecutive users
  - [x] Phase 3: Tool Selection Per Turn — `get_relevant_definitions()` on ToolRegistry, keyword triggers, used-tool tracking
  - [x] Phase 4: Learning Model — `learning.rs`, records to `{workspace}/memory/learnings.json`, injects into system prompt
  - [x] Phase 5: Better Sandbox — normalize commands, split compounds, new deny patterns (curl|sh, base64|sh, chmod+xs)
  - [x] Phase 6: Context Compaction — `compaction.rs`, LLM-powered summarization before truncation, falls back to trim_to_fit
  - [x] Phase 7: Concurrent Message Processing — fan-out dispatcher, per-session serialization, per-message tool instances
  - [x] All tests passing (362), release build clean, pushed to main
- Now: [DONE]
- Next: End-to-end testing with real LLM conversations
- Remaining:
  - [ ] Test compaction with real conversations (verify summary quality)
  - [ ] Tune maxConcurrentChats under load (4 may be conservative)
  - [ ] Consider auto-detecting maxContextTokens per model

## Open Questions
- UNCONFIRMED: Does `maxContextTokens: 128000` work well for local models? May need auto-detection or per-model override
- UNCONFIRMED: Should learning store auto-prune on startup? `prune()` exists but is never called automatically
- Tool selection keyword list may need tuning based on real usage patterns
- UNCONFIRMED: Is compaction summary quality good enough with cheap models? May need model-specific prompt tuning

## Working Set
- Files: `src/agent/{token_budget,thread_repair,learning,compaction}.rs`, `src/agent/{agent_loop,context,mod}.rs`, `src/agent/tools/{registry,shell}.rs`, `src/config/schema.rs`, `src/session/manager.rs`, `src/main.rs`
- Branch: main
- Commit: `8e5f797`
- Test command: `cargo test` (362 tests)
- Build: `cargo build --release`

## New Files Created
| File | Lines | Tests | Purpose |
|------|-------|-------|---------|
| `src/agent/token_budget.rs` | 332 | 7 | Token estimation + 3-stage overflow recovery |
| `src/agent/thread_repair.rs` | 360 | 7 | Message protocol repair before LLM calls |
| `src/agent/learning.rs` | 265 | 6 | Tool outcome tracking + system prompt injection |
| `src/agent/compaction.rs` | ~330 | 3 | LLM-powered context summarization with fallback |

## Architecture Changes (Phase 6-7)

### Context Compaction (`compaction.rs`)
- `ContextCompactor` holds `Arc<dyn LLMProvider>`, model name, summary_max_tokens
- `compact()`: estimate tokens → if over budget → split old/recent → summarize old → reassemble
- `summarize()`: LLM call with focused prompt ("key facts, decisions, pending actions, <500 words")
- Fallback: on any error → `TokenBudget::trim_to_fit()` (existing behavior)
- Called once before the agent tool-call loop; `trim_to_fit()` still runs each iteration as safety net

### Concurrent Processing (`agent_loop.rs` rewrite)
- **AgentLoopShared**: extracted shared state (provider, workspace, model, sessions, subagents, learning, config values)
- **build_tools(channel, chat_id)**: creates per-message ToolRegistry with context baked into MessageTool/SpawnTool/CronTool
- **process_message(&self)**: now takes `&self` (was `&mut self`), fully concurrent-safe
- **run()**: fan-out with `Semaphore::new(max_concurrent_chats)` + per-session `Mutex<()>`
- System messages handled inline (no concurrency slot needed)

### SessionManager thread-safety
- `cache: Mutex<HashMap<String, Session>>` (was plain HashMap)
- All `&mut self` methods → `&self` with internal locking
- New async API: `get_history()`, `add_message_and_save()`, `add_messages_and_save()`

## Testing Notes
- Unit tests cover all new modules (362 total tests pass)
- Compaction test uses MockProvider to verify summary is injected
- Compaction fallback test uses FailingProvider to verify trim_to_fit kicks in
- Session manager test uses UUID keys to avoid stale data from disk
- End-to-end testing: use `RUST_LOG=debug nanobot agent -m "..."` for compaction/concurrency verification
- Gateway concurrency: start gateway with 2+ channels, send simultaneous messages, verify parallel processing
