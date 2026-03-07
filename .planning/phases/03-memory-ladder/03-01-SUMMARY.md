---
phase: 03-memory-ladder
plan: 01
subsystem: memory
tags: [memory-ladder, priority-waterfall, budget-allocation, feature-gates]

# Dependency graph
requires:
  - phase: 01-prompt-contract
    provides: PromptSection enum, SectionEntry struct, PromptBlock
provides:
  - MemoryLayer enum with 5 priority-ordered variants
  - MemoryLadder facade with budget-aware query()
  - Feature-gated available_layers() for knowledge-graph and semantic
  - truncate_to_token_budget helper for line-based truncation
affects: [04-learn-loop, 05-lane-split]

# Tech tracking
tech-stack:
  added: []
  patterns: [priority-waterfall-budget, 50pct-soft-cap, feature-gated-layers]

key-files:
  created:
    - src/agent/memory_ladder.rs
  modified:
    - src/agent/mod.rs
    - src/agent/prepare_context.rs

key-decisions:
  - "Made query() synchronous to avoid Send issues with parking_lot MutexGuard across await points"
  - "Tool patterns extracted into separate PromptSection::ToolPatterns instead of merged into WorkingMemory"
  - "Scratch layer uses block_in_place for async session_db.search_messages call"

patterns-established:
  - "Priority waterfall: iterate layers in Ord order, allocate min(remaining, total/2) per layer"
  - "Feature-gated layer enumeration: cfg(feature) guards in available_layers()"

requirements-completed: [MEM-01, MEM-02, MEM-03]

# Metrics
duration: 8min
completed: 2026-03-07
---

# Phase 3 Plan 1: Memory Ladder Summary

**MemoryLadder facade with 5 priority-ordered layers, 50% soft-cap waterfall budget, and feature-gated layer enumeration wired into prepare_context**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-07T15:27:09Z
- **Completed:** 2026-03-07T15:35:09Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- MemoryLayer enum with GroundTruth > WorkingSession > DurablePersonal > SearchIndex > Scratch priority ordering
- MemoryLadder::query() implements waterfall budget with 50% soft cap per layer
- prepare_context.rs fully migrated from direct store calls to MemoryLadder in both cloud and local paths
- 8 unit tests covering priority ordering, feature gates, budget exhaustion, and soft cap enforcement

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement MemoryLadder types, query logic, and unit tests** - `946e5c2` (feat)
2. **Task 2: Wire MemoryLadder into prepare_context.rs** - `d15f1c8` (feat)

## Files Created/Modified
- `src/agent/memory_ladder.rs` - MemoryLayer enum, MemoryQuery, LayerResult, MemoryLadder struct with query() and available_layers()
- `src/agent/mod.rs` - Added pub mod memory_ladder declaration
- `src/agent/prepare_context.rs` - Replaced direct working_memory/bulletin_cache calls with MemoryLadder queries

## Decisions Made
- Made query() synchronous to avoid Send issues with parking_lot::MutexGuard held across .await points in tokio::spawn contexts
- Separated tool patterns into own PromptSection::ToolPatterns entry instead of merging into WorkingMemory section (per plan: "Tool Patterns continues to be injected independently")
- Scratch layer uses tokio::task::block_in_place + Handle::current().block_on() for the async session_db.search_messages call

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Made query() synchronous instead of async**
- **Found during:** Task 2 (wiring into prepare_context)
- **Issue:** parking_lot::MutexGuard for KnowledgeStore is !Send, held across .await in tokio::spawn context causes compilation error
- **Fix:** Changed query() and fetch_layer() from async to sync; Scratch layer uses block_in_place for its async call
- **Files modified:** src/agent/memory_ladder.rs
- **Verification:** cargo build succeeds, all tests pass
- **Committed in:** d15f1c8

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary for compilation. No functional difference since both call sites pass query="" (Scratch returns empty for empty queries). No scope creep.

## Issues Encountered
None beyond the Send safety issue documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- MemoryLadder provides the foundation for Phase 4 (LearnLoop) which needs to know which memory layers are active
- available_layers() enables Phase 5 (Lane Split) to route based on active features
- All tests pass, no regressions

---
*Phase: 03-memory-ladder*
*Completed: 2026-03-07*
