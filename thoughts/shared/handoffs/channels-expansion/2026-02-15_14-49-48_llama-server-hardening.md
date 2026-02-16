---
date: "2026-02-15T14:49:48+0100"
session_name: channels-expansion
researcher: claude-opus
git_commit: bd60211
branch: main
repository: nanobot
topic: "llama.cpp Server Management Hardening"
tags: [implementation, local-llm, server-management, health-checks, pid-tracking]
status: complete
last_updated: "2026-02-15"
last_updated_by: claude-opus
type: implementation_strategy
root_span_id: ""
turn_span_id: ""
---

# Handoff: llama.cpp Server Management Hardening (All 6 Phases Complete)

## Task(s)

Implemented a 6-phase hardening plan for local LLM server management. All phases complete, all 844 tests passing.

- **Phase 1: finish_reason error check** [COMPLETED] - Added `finish_reason == "error"` guard in `agent_loop.rs` and `subagent.rs` to prevent LLM error text from being treated as model output.
- **Phase 2: Unified spawn_server()** [COMPLETED] - Created `SpawnConfig` struct and single `spawn_server()` function, replaced 3 duplicate spawn functions with thin wrappers.
- **Phase 3: PID-based cleanup** [COMPLETED] - Replaced `pkill -f llama-server` with PID tracking via `~/.nanobot/.server-pids` file. Only kills our tracked PIDs.
- **Phase 4: Health check on LLM error** [COMPLETED] - Added `check_health()` to `server.rs` and `get_api_base()` to `LLMProvider` trait. On error in local mode, pings `/health` to detect server crashes.
- **Phase 5: Periodic health watchdog** [COMPLETED] - Background tokio task pings `/health` every 30s, sends alerts through `display_tx` channel when servers go down or recover.
- **Phase 6: Enhanced /status display** [COMPLETED] - `/status` now shows `SERVERS` line with live health status for all active servers (main, compact, deleg).

**Plan document:** `/home/peppi/.claude/plans/wondrous-purring-mitten.md`

## Critical References

- `src/server.rs` — All server spawn/kill/health logic lives here
- `src/agent/agent_loop.rs` — Main agent loop with finish_reason guard and health-check-on-error

## Recent changes

- `src/agent/agent_loop.rs:760-778` — Phase 1 + Phase 4: finish_reason error check with local health ping
- `src/agent/subagent.rs:495-500` — Phase 1: finish_reason error check (returns Err)
- `src/server.rs:674-770` — Phase 2: `SpawnConfig` struct + `spawn_server()` + thin wrappers
- `src/server.rs:348-420` — Phase 3: `pid_file_path()`, `record_server_pid()`, `unrecord_server_pid()`, `kill_tracked_servers()`
- `src/server.rs:198-208,335-345` — Phase 3: `stop_compaction_server`/`stop_delegation_server` now unrecord PIDs
- `src/server.rs:968-983` — Phase 4: `check_health()` async function
- `src/server.rs:985-1038` — Phase 5: `start_health_watchdog()` background task
- `src/providers/base.rs:93-96` — Phase 4: `get_api_base()` trait method with default None
- `src/providers/openai_compat.rs:272-274` — Phase 4: `get_api_base()` returns `Some(&self.api_base)`
- `src/repl/mod.rs:457-465` — Phase 3: `kill_current()` uses `unrecord_server_pid` + `kill_tracked_servers`
- `src/repl/mod.rs:467-477` — Phase 3: `shutdown()` unrecords PIDs
- `src/repl/mod.rs:489-491` — Phase 3: `try_start_server` records PID on spawn
- `src/repl/mod.rs:740-750` — Phase 5: Starts health watchdog when in local mode
- `src/repl/commands.rs:275-327` — Phase 6: `/status` shows SERVERS line with live health

## Learnings

1. **Error text as model output is a real problem**: `openai_compat.rs` wraps HTTP/parse errors as `Ok(LLMResponse { finish_reason: "error" })`. Without explicit checking, agent_loop treats it as a valid response. This was the highest-priority fix.

2. **The three spawn functions differed only in gpu_layers**: `spawn_llama_server` (99), `spawn_compaction_server` (10), `spawn_delegation_server` (configurable). The unified `SpawnConfig` + `spawn_server()` eliminates triple bug surface.

3. **`pkill -f llama-server` is dangerous**: It kills ALL llama-server processes system-wide. PID tracking via `~/.nanobot/.server-pids` (format: `role:pid` per line) is safer. Uses `libc::kill(pid, SIGKILL)` which was already a dependency.

4. **LLMProvider trait doesn't expose api_base**: Added `get_api_base() -> Option<&str>` with default None to the trait, implemented on `OpenAICompatProvider`. This allows health checks without knowing the concrete provider type.

5. **Test assertion change**: The unified `spawn_server` changes error message from "Model not found" to "Main model not found" (capitalize of role). Updated the test assertion to `contains("model not found")` (lowercase substring match).

## Post-Mortem (Required for Artifact Index)

### What Worked
- Direct implementation of well-researched plan — all 6 phases completed in a single session
- Keeping the three spawn functions as thin wrappers preserved all callers unchanged
- Using `libc::kill` (already a dependency) for PID-based kill avoided adding new deps
- Adding `get_api_base()` as a trait method with default None was minimally invasive

### What Failed
- No failures. The plan was well-researched and the implementation was straightforward.
- One test needed updating (`test_spawn_server_errors_when_model_missing`) due to capitalized error message — caught immediately by test run.

### Key Decisions
- Decision: Keep spawn wrappers (`spawn_llama_server`, etc.) instead of changing all callers
  - Alternatives: Inline `spawn_server(SpawnConfig{...})` at each call site
  - Reason: Smaller diff, existing callers don't need changes, and the wrappers serve as documentation
- Decision: Use `libc::kill` for PID cleanup instead of `Command::new("kill")`
  - Alternatives: Shell out to `kill`, use nix crate
  - Reason: `libc` already a dependency, direct syscall is more reliable
- Decision: Add `get_api_base()` to LLMProvider trait vs storing api_base on SwappableCore
  - Alternatives: Add field to SwappableCore, downcast provider
  - Reason: Trait method is cleanest, default None means only OpenAICompatProvider needs impl

## Artifacts

- `/home/peppi/.claude/plans/wondrous-purring-mitten.md` — Implementation plan (all 6 phases)
- `src/server.rs` — Unified spawn, PID tracking, health check, watchdog
- `src/agent/agent_loop.rs` — finish_reason guard + health-check-on-error
- `src/agent/subagent.rs` — finish_reason guard
- `src/providers/base.rs` — `get_api_base()` trait method
- `src/providers/openai_compat.rs` — `get_api_base()` implementation
- `src/repl/mod.rs` — PID recording/cleanup wiring, watchdog startup
- `src/repl/commands.rs` — Enhanced `/status` with SERVERS health display

## Action Items & Next Steps

1. **Manual testing**: Start nanobot in local mode, kill llama-server mid-conversation, verify `[LLM Error]` message appears and `/status` shows `DOWN`.
2. **Verify PID file**: After local startup, check `~/.nanobot/.server-pids` contains correct PIDs.
3. **Test external llama-server survival**: Start an external llama-server, then start nanobot — the external one should survive nanobot's cleanup.
4. **Consider**: The health watchdog currently only starts at REPL init. If the user switches to local mode via `/local`, the watchdog doesn't start. This could be wired into the `/local` command handler for dynamic watchdog management.

## Other Notes

- The `check_health()` function strips `/v1` from the api_base URL to construct the health endpoint URL (`/health` lives at the server root, not under `/v1`).
- The watchdog sends alerts via the same `display_tx` channel used by subagents — alerts appear before the next prompt, using `\x1b[RAW]` prefix to bypass markdown rendering.
- PID file uses simple `role:pid` format (one per line). `unrecord_server_pid` removes by PID match. `kill_tracked_servers` kills all and deletes the file.
