---
phase: 01-prompt-contract
plan: 01
subsystem: agent
tags: [prompt, context, assembler, overflow, token-budget]

# Dependency graph
requires: []
provides:
  - "PromptSection enum with 13 variants in fixed discriminant order"
  - "SectionEntry, SectionSource, AssemblyContext, AssemblyResult types"
  - "PromptAssembler trait with CloudAssembler and LocalAssembler"
  - "Two-pass overflow: drop non-shrinkable tail-first, then shrink"
affects: [01-prompt-contract plan 02, context wiring, provider selection]

# Tech tracking
tech-stack:
  added: []
  patterns: [two-pass-overflow, assembler-trait, discriminant-ordered-enum]

key-files:
  created:
    - src/agent/prompt_contract.rs
  modified:
    - src/agent/context.rs
    - src/agent/mod.rs

key-decisions:
  - "Pass 1 skips shrinkable sections, giving them a chance to be truncated in Pass 2 before dropping"
  - "PromptBlock::content() accessor added as pub(crate) for shrink pass truncation"
  - "PromptBlockReport extended with allocated_tokens and source fields (backward-compatible)"

patterns-established:
  - "Assembler pattern: PromptAssembler trait with struct impls (CloudAssembler, LocalAssembler)"
  - "Two-pass overflow: drop non-shrinkable from tail, then shrink last shrinkable"
  - "Proportional budget: section percentages scaled against context window cap"

requirements-completed: [PROMPT-01, PROMPT-02, PROMPT-04]

# Metrics
duration: 7min
completed: 2026-03-07
---

# Phase 1 Plan 1: Prompt Contract Types Summary

**Typed prompt contract with 13-variant PromptSection enum, two-pass overflow (drop then shrink), and CloudAssembler/LocalAssembler producing separate system/developer content splits**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-07T13:22:23Z
- **Completed:** 2026-03-07T13:29:01Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- PromptSection enum with 13 variants in fixed discriminant order, each mapped to PromptBlockKind and budget percentage
- PromptAssembler trait with CloudAssembler (system/developer split) and LocalAssembler (single concatenated prompt)
- Two-pass overflow: drops non-shrinkable sections from tail first, then shrinks lowest remaining shrinkable
- 13 unit tests covering ordering, kind mapping, shrinkability, budgets, assembly, overflow, and empty section handling

## Task Commits

Each task was committed atomically:

1. **Task 1: Define PromptSection enum, SectionEntry, SectionSource, and AssemblyContext types** - `973f929` (test)
2. **Task 2: Implement PromptAssembler trait with CloudAssembler and LocalAssembler** - `596828e` (feat)

## Files Created/Modified
- `src/agent/prompt_contract.rs` - PromptSection enum, SectionEntry, SectionSource, AssemblyContext, AssemblyResult, PromptAssembler trait, CloudAssembler, LocalAssembler, enforce_budget, 13 tests
- `src/agent/context.rs` - PromptBlock::content() accessor, PromptBlockReport extended with allocated_tokens and source
- `src/agent/mod.rs` - pub mod prompt_contract declaration

## Decisions Made
- Pass 1 of overflow skips shrinkable sections, preserving them for Pass 2 truncation rather than dropping outright
- Added PromptBlock::content() as pub(crate) accessor rather than making the field public
- PromptBlockReport extended with allocated_tokens (usize) and source (String) -- existing construction sites provide defaults (0, "")

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Two-pass overflow dropping shrinkable sections prematurely**
- **Found during:** Task 2 (assembler implementation)
- **Issue:** Original plan described Pass 1 as "drop the last included section" which would drop shrinkable sections before they get a chance to be truncated in Pass 2
- **Fix:** Pass 1 now skips shrinkable sections. Pass 1b drops excess shrinkable sections but keeps at least one for Pass 2 to truncate
- **Files modified:** src/agent/prompt_contract.rs
- **Verification:** test_overflow_shrinks_after_drop_fails passes
- **Committed in:** 596828e

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Necessary for correct overflow behavior. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All types and assemblers ready for Plan 02 (wiring into ContextBuilder)
- PromptAssembler trait is the integration surface -- ContextBuilder will populate AssemblyContext and call assemble()

## Self-Check: PASSED

- src/agent/prompt_contract.rs: FOUND
- src/agent/context.rs: FOUND
- src/agent/mod.rs: FOUND
- Commit 973f929: FOUND
- Commit 596828e: FOUND

---
*Phase: 01-prompt-contract*
*Completed: 2026-03-07*
