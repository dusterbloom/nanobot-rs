---
phase: 04-learn-loop
plan: 01
subsystem: agent
tags: [learn-loop, trait-dispatch, observer-pattern, perplexity-gate, audit]

# Dependency graph
requires:
  - phase: 03-memory-ladder
    provides: "MemoryLadder context assembly"
provides:
  - "TurnOutcome struct for cross-thread turn data transfer"
  - "LearnLoop trait with observe_immediate/observe_async dispatch"
  - "DefaultLearnLoop dispatching to audit, calibrator, perplexity gate"
  - "Zero direct store writes in finalize_response.rs"
affects: [04-learn-loop]

# Tech tracking
tech-stack:
  added: []
  patterns: [observer-trait-dispatch, sync-async-observer-split]

key-files:
  created:
    - src/agent/learn_loop.rs
  modified:
    - src/agent/finalize_response.rs
    - src/agent/agent_loop.rs
    - src/agent/agent_shared.rs
    - src/agent/mod.rs
    - src/agent/lora_bridge.rs

key-decisions:
  - "Pre-compute cost_usd in finalize_response (async context) and pass via TurnOutcome"
  - "Calibrator field changed to Arc<Mutex<...>> for shared ownership between AgentLoopShared and LearnLoop"
  - "LearnLoop rebuilt on set_perplexity_gate/set_mlx_provider to capture updated config"
  - "Moved query_perplexity, build_ane_training_config, try_mlx_or_http_train to learn_loop.rs"

patterns-established:
  - "Observer trait dispatch: all turn observations flow through LearnLoop trait"
  - "Sync/async observer split: immediate observers run inline, async spawned via tokio"

requirements-completed: [LEARN-01, LEARN-02, LEARN-03]

# Metrics
duration: 8min
completed: 2026-03-07
---

# Phase 4 Plan 1: LearnLoop Summary

**LearnLoop trait with TurnOutcome struct replacing 200+ lines of scattered observers in finalize_response**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-07T16:25:47Z
- **Completed:** 2026-03-07T16:33:47Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- TurnOutcome struct with 18 flat owned fields capturing all observer-needed turn data
- LearnLoop trait with observe_immediate (sync: audit + calibrator + structured log) and observe_async (perplexity gate + LoRA training)
- finalize_response.rs reduced by 136 net lines, zero direct store writes remaining
- Full test suite passes (2038 tests, 0 failures)

## Task Commits

Each task was committed atomically:

1. **Task 1: TurnOutcome, LearnLoop trait, DefaultLearnLoop** - `4dd95fb` (feat)
2. **Task 2: Wire LearnLoop into finalize_response and agent_loop** - `b0a70fe` (feat)

## Files Created/Modified
- `src/agent/learn_loop.rs` - TurnOutcome struct, LearnLoop trait, DefaultLearnLoop impl, helper functions
- `src/agent/finalize_response.rs` - Refactored to construct TurnOutcome and call LearnLoop dispatch
- `src/agent/agent_loop.rs` - Constructs DefaultLearnLoop, rebuilds on config changes
- `src/agent/agent_shared.rs` - Added learn_loop field, changed calibrator to Arc-wrapped
- `src/agent/mod.rs` - Added learn_loop module declaration
- `src/agent/lora_bridge.rs` - Updated test reference from finalize_response to learn_loop

## Decisions Made
- Pre-compute cost_usd in finalize_response.rs (which has async context) rather than blocking in observe_immediate
- Changed calibrator from `Option<Mutex<...>>` to `Option<Arc<Mutex<...>>>` for shared ownership
- LearnLoop is rebuilt when set_perplexity_gate or set_mlx_provider is called (captures updated config)
- Helper functions (query_perplexity, build_ane_training_config, try_mlx_or_http_train) moved to learn_loop.rs

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Updated lora_bridge test reference**
- **Found during:** Task 2 (wiring)
- **Issue:** Test in lora_bridge.rs referenced `finalize_response::query_perplexity` which was moved to learn_loop
- **Fix:** Updated import path to `learn_loop::query_perplexity`
- **Files modified:** src/agent/lora_bridge.rs
- **Verification:** cargo test passes
- **Committed in:** b0a70fe (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary path update after function relocation. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- LearnLoop trait established as single dispatch point for all turn observations
- Ready for additional observers to be added via the trait
- finalize_response.rs is clean of direct store writes

---
*Phase: 04-learn-loop*
*Completed: 2026-03-07*
