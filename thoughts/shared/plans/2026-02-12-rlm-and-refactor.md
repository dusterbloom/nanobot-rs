# Plan: RLM Tool Delegation + main.rs Refactor

**Status**: SAVED FOR LATER
**Date**: 2026-02-12

## Phase 1: Refactor main.rs (P0)

`main.rs` is 3,565 lines / 64 functions. `cmd_agent()` alone is 1,091 lines.

### Extractions

| New module | What moves | ~Lines |
|---|---|---|
| `src/repl.rs` | REPL loop + command dispatch | 600 |
| `src/tui.rs` | ANSI helpers, status bar, splash | 200 |
| `src/server.rs` | llama-server spawn, health, GGUF parser | 500 |
| `src/cli.rs` | `cmd_onboard`, `cmd_status`, `cmd_tune`, cron | 300 |

### DRY fixes
- Merge `process_message()` / `process_message_streaming()` in agent_loop.rs (~280 dup lines)
- Extract `restart_with_fallback()` for 4x copy-pasted server restart logic
- Generic `ArcToolProxy<T>` to eliminate 3 proxy struct boilerplate

## Phase 2: RLM Tool Delegation

Add subagent-based tool execution so main orchestrator context isn't consumed by tool loops.

### Changes (~200-300 lines)
- Add `toolRunnerProvider` + `toolRunnerModel` to config/SharedCore
- Add `SubagentManager::delegate_tool_loop()` method
- Optional delegation in `process_message()` after LLM returns tool calls
- Cheap model (Qwen 0.5b) runs tool loop independently, returns aggregated results
