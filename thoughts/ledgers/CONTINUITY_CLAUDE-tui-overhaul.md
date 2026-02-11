# Continuity Ledger: TUI Overhaul

## Goal
Replace emoji-based TUI with typography-first ASCII design. Add context window visibility, subagent management commands, and a compact status bar after every response.

**Done looks like**: Zero emoji in REPL output. Status bar shows ctx%, channels, agents, turn after every response. `/agents` and `/kill` commands work. `/status` shows rich breakdown.

## Constraints
- No new dependencies
- No new files (all changes in existing modules)
- ASCII/typographic indicators only (no emoji anywhere)
- Sparse color accents: green (ok), yellow (warning), red (pressure), cyan (channels), magenta (voice)
- Status bar must be dim/unobtrusive

## Key Decisions
- **Prompts use single-char indicators**: `>` cloud, `L>` local, `~>` voice. Removed "You:" label for cleaner look. [2026-02-11]
- **Context tracking via AtomicU64**: Added `last_context_used`/`last_context_max` to SharedCore rather than passing through return values. Interior mutability avoids changing process_message signature. [2026-02-11]
- **SubagentInfo as separate Clone struct**: Stored alongside JoinHandle in running_tasks HashMap. Info is cloneable, handle is not. [2026-02-11]
- **cancel() uses prefix match**: `/kill a3f2` matches full ID `a3f2c8e1`. User convenience. [2026-02-11]
- **print_status_bar takes &[&str] channel names**: Decoupled from ActiveChannel struct which stays local to cmd_agent. [2026-02-11]
- **ObservationStore::count() for /status**: Simple file count, no token estimation. Fast. [2026-02-11]

## State
- Done:
  - [x] Phase 1: Data plumbing (token_budget getter, context atomics, SubagentInfo, list_running/cancel, subagent_manager accessor)
  - [x] Phase 2: TUI redesign (remove emoji, new prompts, status bar fn, RED color)
  - [x] Phase 3: New commands (/agents, /kill, enhanced /status, updated /help)
  - [x] Phase 4: Wire status bar into REPL loop (3 locations: normal, paste, voice)
  - [x] Build: cargo build clean (43 warnings, all pre-existing)
  - [x] Tests: 433 passed, 0 failed
- Now: [DONE] All phases complete
- Next: Manual verification, commit

## Open Questions
- UNCONFIRMED: Gateway mode (`nanobot gateway`) still uses `LOGO` ("*") in println. Acceptable or should it say "nanobot" instead?
- The `cmd_onboard` still prints `LOGO` ("*") in "* nanobot is ready!" — acceptable ASCII

## Working Set
- Files changed this session:
  - `src/agent/token_budget.rs` — max_context() getter
  - `src/agent/agent_loop.rs` — context atomics, subagent_manager()
  - `src/agent/subagent.rs` — SubagentInfo, list_running(), cancel()
  - `src/agent/observer.rs` — count() method
  - `src/main.rs` — emoji removal, prompts, status bar, /agents, /kill, /status, /help
- Branch: main (uncommitted)
- Test: `cargo test` (433 pass)
- Build: `cargo build` (clean)
