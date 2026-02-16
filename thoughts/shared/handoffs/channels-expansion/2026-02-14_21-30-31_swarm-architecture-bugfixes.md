---
date: 2026-02-14T21:30:31+0100
session_name: "channels-expansion"
researcher: claude
git_commit: d9cbc0f1e75df83ee98ac0dee90d4aadc46f2948
branch: main
repository: nanobot
topic: "Swarm Architecture Implementation + Bug Fixes"
tags: [implementation, swarm, delegation, rlm, async-display, tool-runner, repl]
status: complete
last_updated: 2026-02-14
last_updated_by: claude
type: implementation_strategy
root_span_id: ""
turn_span_id: ""
---

# Handoff: Swarm Architecture Implementation + Two Bug Fixes

## Task(s)

### 1. Swarm Architecture Implementation [COMPLETED]
Implemented the full swarm architecture plan from `/home/peppi/.nanobot/workspace/plans/swarm-architecture.md`. This transforms nanobot's RLM (Recursive Language Model) tool delegation system from a single summarizer into a programmable swarm of micro-agents.

**6 micro-tools implemented across 3 phases:**
- Phase 1: `mem_store` / `mem_recall` (working memory for delegation sessions) + `verify` (shell command verification)
- Phase 2: `python_eval` (sandboxed Python execution) + `fmt_convert` (JSON/CSV/table conversion)
- Phase 3: `delegate` (recursive worker spawning) + `Budget` struct for resource control

All 749 tests pass.

### 2. Fix RLM Delegation Defaults [COMPLETED]
The RLM delegation system was inert by default:
- `toolDelegation.enabled` defaulted to `false` -- changed to `true`
- `max_iterations` defaulted to 3 -- raised to 10
- Iteration 0 suppressed micro-tools -- removed suppression so RLM can inspect results immediately

### 3. Fix Async Display Drain for Subagent Results [COMPLETED]
Subagent results were invisible in the REPL because `readline()` blocked the thread and `drain_display()` only ran at 3 discrete moments via `try_recv()`. Implemented async readline using rustyline's `ExternalPrinter` + `tokio::select!` so subagent output renders while the user is at the prompt.

## Critical References
- `/home/peppi/.nanobot/workspace/plans/swarm-architecture.md` -- the original swarm architecture plan
- `src/agent/tool_runner.rs` -- core delegation loop, Budget struct, delegate tool
- `src/repl/commands.rs` -- ReplContext, drain_display, readline_async

## Recent changes

**New files:**
- `src/agent/worker_tools.rs` -- 804 lines: verify, python_eval, diff_apply, fmt_convert async worker tools

**Modified files:**
- `src/agent/context_store.rs:24-120` -- Added `memory: HashMap<String, String>`, `mem_store()`, `mem_recall()`, `mem_keys()` methods, extended `MICRO_TOOLS` and `execute_micro_tool` (now takes `&mut self`)
- `src/agent/tool_runner.rs:1-50` -- Added `Budget` struct with `root()`, `child()`, `can_delegate()` methods
- `src/agent/tool_runner.rs:200-250` -- Wired worker tools into `allowed_tools` and tool definitions
- `src/agent/tool_runner.rs:550-563` -- Removed iteration 0 tool suppression (was `if iteration == 0 { None }`)
- `src/agent/tool_runner.rs:600-650` -- Added delegate dispatch with `Box::pin(run_tool_loop(...))` for recursive async
- `src/config/schema.rs:583-584` -- Changed `default_td_max_iterations` from 3 to 10
- `src/config/schema.rs:598-601` -- Changed `enabled` default from `false` to `true` via `default_true`
- `src/config/schema.rs:655-680` -- Added `WorkerConfig` struct
- `src/agent/mod.rs:19` -- Added `pub mod worker_tools`
- `src/repl/commands.rs:70-115` -- Added `readline_async()` method using `ExternalPrinter` + `spawn_blocking` + `tokio::select!`
- `src/repl/mod.rs:743-763` -- Replaced `ctx.rl.readline(&prompt)` with `ctx.readline_async(&prompt).await` in both cfg branches

## Learnings

1. **rustyline 15 ExternalPrinter**: `create_external_printer(&mut self)` returns an owned `ExternalPrinter` that communicates via internal pipe. The printer can safely output text while `readline()` is active on a different thread -- it clears the line, prints the message, and redraws the prompt. `fn print(&mut self, msg: String)` takes owned `String`.

2. **Moving rustyline Editor to spawn_blocking**: `DefaultEditor` is `Send`, so it can be moved to `tokio::task::spawn_blocking`. The `ExternalPrinter` pipe survives the move because pipe fds are integers that survive bit-copy. Use `std::mem::replace` with a temp `DefaultEditor::new()` to take the editor out of the context.

3. **execute_micro_tool mutability**: Adding `mem_store` required changing `execute_micro_tool` from `&ContextStore` to `&mut ContextStore`. This cascades to all callers in `tool_runner.rs`.

4. **Recursive async for delegate tool**: `run_tool_loop` calling itself requires `Box::pin()` for the recursive future: `Box::pin(run_tool_loop(...)).await`.

5. **RLM iteration 0 suppression was counterproductive**: Suppressing tools on iteration 0 forced the model to summarize without being able to inspect results first. Removing this lets the RLM use ctx_slice/ctx_grep before deciding.

## Post-Mortem (Required for Artifact Index)

### What Worked
- Implementing all 6 tools in a single session with parallel agent research
- Using oracle agents to diagnose both bugs concurrently -- each produced actionable root cause analysis
- rustyline's `ExternalPrinter` was the exact right abstraction for async display drain
- The `Budget` struct with `child()` halving pattern cleanly prevents infinite recursion

### What Failed
- Multiple explore/oracle agents failed with `classifyHandoffIfNeeded is not defined` hook error -- this is a hook infrastructure issue, not a code problem. Work completed before the error fired.
- Initial attempt to understand the display drain bug required reading multiple files across repl/, agent/, and cli.rs to trace the full channel wiring

### Key Decisions
- Decision: Use `ExternalPrinter` + `spawn_blocking` + `select!` for async readline
  - Alternatives considered: background drain thread, polling with timeout, direct stdout writes during readline
  - Reason: ExternalPrinter is the officially supported way to output during readline without terminal corruption
- Decision: Default `toolDelegation.enabled` to `true`
  - Alternatives considered: Keep false, require explicit config
  - Reason: The swarm system is the whole point of the delegation infrastructure; defaulting to off makes it inert
- Decision: Remove iteration 0 tool suppression entirely
  - Alternatives considered: Suppress only on first call, suppress only summarize-type tools
  - Reason: The RLM needs micro-tools to inspect results before summarizing; forced summarization produces blind summaries

## Artifacts
- `src/agent/worker_tools.rs` -- NEW: all async worker tools
- `src/agent/context_store.rs:90-120` -- mem_store/mem_recall methods
- `src/agent/tool_runner.rs:1-50` -- Budget struct
- `src/agent/tool_runner.rs:200-250` -- worker tool + delegate wiring
- `src/config/schema.rs:655-680` -- WorkerConfig struct
- `src/repl/commands.rs:70-115` -- readline_async method
- `/home/peppi/.nanobot/workspace/plans/swarm-architecture.md` -- original plan (all phases complete)

## Action Items & Next Steps
1. **Commit these changes** -- all work is uncommitted on main branch
2. **Test RLM delegation end-to-end** -- spawn a subagent (e.g. `/spawn nytimes-headlines`) and verify: (a) the RLM actually uses micro-tools to inspect results, (b) subagent output appears at the REPL prompt without waiting for user input
3. **Test voice mode display drain** -- voice mode still uses the old blocking path for `voice_read_input()`; consider adding async drain there too if subagent results don't appear during voice recording
4. **Consider adding delegation provider auto-detection** -- if no delegation provider is configured and no local model is available, delegation should gracefully fall back to the main model
5. **Monitor RLM iteration count** -- with `max_iterations` raised to 10, verify the RLM doesn't burn excessive iterations on simple tasks

## Other Notes
- The `classifyHandoffIfNeeded is not defined` error on many agents is a hook infrastructure issue, not related to code changes. It fires in the hook system after agents complete.
- The `display_tx`/`display_rx` channel is `tokio::sync::mpsc::unbounded_channel::<String>` created at `src/repl/mod.rs:653`. It flows through `cli::create_agent_loop` and into `SubagentManager` for subagent result delivery.
- In CLI mode, `bus_tx` path is dead (`_outbound_rx` is dropped at `src/cli.rs:485`), so all CLI-mode results must flow through `display_tx`.
- The `WorkerConfig` struct in `schema.rs` controls the swarm system separately from `ToolDelegationConfig`. They're independent: delegation controls the RLM, workers control recursive sub-workers.
