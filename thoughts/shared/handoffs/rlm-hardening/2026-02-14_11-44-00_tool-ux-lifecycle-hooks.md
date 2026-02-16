---
date: 2026-02-14T11:44:00+00:00
session_name: "rlm-hardening"
researcher: peppi
git_commit: 0045b134739eca5b37ea988d5d7fc3f513da9a94
branch: main
repository: nanobot
topic: "Tool Execution UX Lifecycle Hooks"
tags: [ux, tool-events, exec, repl, cancellation, hooks]
status: planned
last_updated: 2026-02-14
last_updated_by: peppi
type: implementation_strategy
root_span_id: ""
turn_span_id: ""
---

# Handoff: Tool Execution UX + Lifecycle Hooks

## Task(s)

### Completed (this session)
1. **RLM Hardening** [DONE] — Full delegation model safety overhaul:
   - Tool filtering (delegation can only use tools from initial request)
   - Stronger ExecTool deny patterns (rm variants, find -delete, shred, sudo)
   - Short-circuit for trivial outputs (<200 chars)
   - Antifragile delegation (health tracking, auto-fallback, re-probe)
   - Dynamic VRAM GPU allocation
   - Iteration scaling (cloud: up to 50, local: capped at 15)
   - dom_smoothie HTML extraction replacing scraper
   - /restart command and delegation health in /status

### Planned (next session)
2. **Tool Execution UX Improvement** [PLANNED] — User reported two critical UX problems:
   - **Problem 1: No progress visibility** — After `▶ exec({"command":"..."})` displays, user sees nothing until the tool finishes or times out. No streaming output, no elapsed time, no spinner.
   - **Problem 2: No cancellation** — User can't cancel a stuck tool without Ctrl+C which kills the entire process and breaks session continuity.
   - **User's requirement**: "structurally SOLID / DRY way so that before and after tools hooks can work and users can see what is going on and eventually stop it without breaking continuity"

## Critical References
- `src/repl.rs:175-305` — Current REPL tool event display (print_task loop)
- `src/agent/audit.rs:329-347` — ToolEvent enum (CallStart/CallEnd only, no progress events)
- `src/agent/agent_loop.rs:1004-1110` — Inline tool execution path (where exec runs)

## Recent Changes
- `src/agent/tool_runner.rs:66-73` — Tool filtering (allowed_tools HashSet)
- `src/agent/tool_runner.rs:91-107` — Filtered tool definitions for delegation
- `src/agent/tool_runner.rs:224-237` — Block uninvited tools in chained calls
- `src/agent/tools/shell.rs:14-54` — Expanded deny patterns
- `src/cli.rs:217-240` — effective_max_iterations scaling
- `src/agent/agent_loop.rs:745-753` — short_circuit_chars: 200 in runner config

## Learnings

### Architecture for tool UX (research findings)
1. **ToolEvent is the right abstraction** — Already has CallStart/CallEnd, needs new variants:
   - `Progress { tool_name, tool_call_id, elapsed_ms, output_preview }` — periodic updates during execution
   - Possibly `Cancellation { tool_call_id }` — user requested stop

2. **ExecTool is the bottleneck** — `src/agent/tools/shell.rs:299-335`. Uses `tokio::time::timeout` wrapping `Command::new("sh").output()`. The `.output()` call blocks until completion — no streaming. To show progress:
   - Replace `.output()` with `.spawn()` + async read from stdout/stderr
   - Emit periodic Progress events on the tool_event_tx channel
   - Show elapsed time or last output line in the REPL

3. **Cancellation architecture** — The exec tool uses `tokio::time::timeout` but there's no way for the REPL to signal "cancel this tool". Options:
   - Pass a `CancellationToken` (from `tokio_util`) into tool execution
   - The REPL listens for Ctrl+C during tool execution, signals the token
   - Tool implementations check the token and abort gracefully
   - The agent loop receives a partial result and continues (session not broken)

4. **DRY concern** — The tool event display logic is inline in the REPL. The before/after hook pattern should be:
   - `Tool` trait gets optional `fn supports_streaming(&self) -> bool`
   - Streaming tools emit progress on a channel
   - Non-streaming tools get a generic spinner with elapsed time
   - All tools check a cancellation token

5. **Delegation path already emits events** — `agent_loop.rs:756-768` emits CallStart for delegated tools. But delegation runs its own tool loop (`tool_runner.rs`) which does NOT emit events — the REPL sees "tool started" then silence until delegation finishes.

### Key pattern: the two execution paths
Both inline and delegated tool execution need the same UX treatment:
- **Inline** (`agent_loop.rs:1004-1110`): tools.execute() with ToolEvent emission
- **Delegated** (`agent_loop.rs:722-978` → `tool_runner.rs`): tool_runner::run_tool_loop() — currently no progress events

### Critical files for implementation
- `src/agent/tools/base.rs` — Tool trait definition (add streaming/cancellation support)
- `src/agent/tools/shell.rs` — ExecTool (primary target for streaming output)
- `src/agent/audit.rs` — ToolEvent enum (add Progress variant)
- `src/repl.rs:186-262` — print_task loop (add Progress handling, cancellation keybinding)
- `src/agent/agent_loop.rs:1004-1110` — Inline execution (pass cancellation token)
- `src/agent/tool_runner.rs:138-155` — Delegated execution (emit progress events)

## Post-Mortem (Required for Artifact Index)

### What Worked
- **Tool filtering as prompt injection defense**: HashSet of allowed tool names from initial_tool_calls, checked at both definition-level (LLM can't see tools) and execution-level (blocked even if hallucinated). Clean, zero-overhead defense.
- **effective_max_iterations pattern**: Scaling with context size at the cli.rs callsite rather than inside build_shared_core keeps the core generic while callsites adapt to their environment.
- **replace_all for test config updates**: When adding a new struct field to ToolRunnerConfig, using Edit with replace_all on `max_tool_result_chars: 30000,` pattern updated all 11 test configs in one operation.

### What Failed
- **Hardcoded SHORT_RESULT_THRESHOLD constant**: Initially used `const SHORT_RESULT_THRESHOLD: usize = 200;` instead of `config.short_circuit_chars`. Caused 10 test failures because CountingTool returns 16-char results, triggering short-circuit in all tests. Fixed by using config field with `short_circuit_chars: 0` in tests.
- **rm deny pattern too narrow**: Original `\brm\s+-[rf]{1,2}\b` missed `rm -rv`, `rm -rfi`, `rm --recursive`. The `{1,2}` quantifier + `\b` word boundary meant any flag combo > 2 chars or with non-rf chars was unmatched.

### Key Decisions
- **Decision**: Tool filtering in tool_runner, not in ToolRegistry
  - Alternatives: Could have added filtering to ToolRegistry.get_definitions(), or used a separate "safe registry"
  - Reason: Filtering belongs in the delegation context, not the registry. The main model should see all tools; only the delegation model is restricted.
- **Decision**: Scale iterations in cli.rs, not in build_shared_core
  - Alternatives: Could have added is_local logic inside build_shared_core
  - Reason: build_shared_core is a pure builder; policy decisions (scaling) belong in the callsite that knows the deployment context.
- **Decision**: Block sudo entirely in deny patterns
  - Alternatives: Could allow sudo for specific safe commands
  - Reason: The bot should never need privilege escalation. If it does, the user can run the command manually.

## Artifacts
- `src/agent/tool_runner.rs` — Tool filtering, short-circuit, loop detection, truncation (all new)
- `src/agent/agent_loop.rs` — Delegation health, auto-fallback, instructions passthrough
- `src/agent/tools/shell.rs` — Expanded deny patterns (rm variants, find, shred, sudo)
- `src/agent/tools/web.rs` — dom_smoothie replacement
- `src/cli.rs` — effective_max_iterations
- `src/server.rs` — Dynamic VRAM GPU allocation, model preferences
- `src/repl.rs` — /restart command, delegation health in /status
- `src/config/schema.rs` — td_max_iterations default: 3
- `thoughts/ledgers/CONTINUITY_CLAUDE-rlm-hardening.md` — Continuity ledger

## Action Items & Next Steps

### Phase 1: ToolEvent Progress variant
1. Add `Progress { tool_name, tool_call_id, elapsed_ms, output_preview: Option<String> }` to `ToolEvent` enum in `src/agent/audit.rs`
2. Handle `Progress` in REPL print_task loop (`src/repl.rs:210-257`) — show elapsed time, optionally last output line

### Phase 2: ExecTool streaming
3. Replace `.output()` with `.spawn()` + async stdout/stderr reading in `src/agent/tools/shell.rs:299-335`
4. Emit periodic Progress events (every 1-2 seconds) with elapsed time and last output line
5. Need to thread `tool_event_tx` through to tool execution — currently tools don't have access to the event channel

### Phase 3: Cancellation support
6. Add `CancellationToken` (from `tokio_util::sync`) to tool execution path
7. REPL catches Ctrl+C during tool execution, signals the token (instead of killing process)
8. ExecTool kills the child process on cancellation, returns partial output
9. Agent loop receives partial result and continues normally

### Phase 4: DRY tool lifecycle
10. Add `fn supports_streaming(&self) -> bool` to Tool trait (default false)
11. Non-streaming tools get generic spinner with elapsed time in REPL
12. Create `ToolExecutionContext` struct passed to `execute()` with event_tx + cancellation_token
13. Both inline and delegated paths use the same context

### Design consideration: Tool trait signature change
The current `Tool::execute(&self, params: HashMap<String, Value>) -> String` has no way to receive an event channel or cancellation token. Options:
- **Option A**: Add optional `context` parameter: `execute(&self, params, ctx: Option<&ToolContext>)`
- **Option B**: Builder pattern: tools that support streaming implement `StreamingTool` trait
- **Option C**: Pass event_tx when constructing tools (like MessageTool/SpawnTool already do)
- **Recommended**: Option A is simplest and backward-compatible (None for tests, Some for production)

## Other Notes

### Current exec timeout behavior
`src/agent/tools/shell.rs:299`: `tokio::time::timeout(Duration::from_secs(self.timeout), ...)` — default timeout is `config.tools.exec_.timeout` (30s from schema defaults). The user sees nothing for up to 30 seconds.

### The delegation path is blind
When tools are delegated, `tool_runner.rs` executes tools but has no `tool_event_tx`. The REPL only sees CallStart events emitted by agent_loop.rs before delegation starts, then silence until the full delegation loop completes. To fix:
- Thread `tool_event_tx` through to `run_tool_loop`
- Emit CallStart/CallEnd/Progress events from within the tool runner loop

### 668 tests passing
All changes from this session maintain test suite integrity. New tests: 13 for tool filtering, 10 for shell deny patterns, 6 for iteration scaling.
