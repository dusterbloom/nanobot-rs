---
phase: 01-prompt-contract
plan: 02
subsystem: agent
tags: [prompt-assembly, context-builder, cloud-assembler, local-assembler, section-entry]

# Dependency graph
requires:
  - phase: 01-prompt-contract
    provides: PromptSection enum, SectionEntry, AssemblyContext, CloudAssembler, LocalAssembler in prompt_contract.rs
provides:
  - build_messages() delegates to CloudAssembler for cloud prompt assembly
  - build_local_system_prompt() delegates to LocalAssembler for local prompt assembly
  - collect_cloud_runtime_sections() pre-fetches working memory, daily notes, subagent status, bulletin as typed SectionEntry values
  - collect_static_sections() converts identity, verification, workspace context, memory, skills into SectionEntry values
  - inject_runtime_sections() appends rendered sections to developer message
  - Zero append_to_system_prompt() calls in prepare_context.rs
affects: [05-lane-split, prompt-contract]

# Tech tracking
tech-stack:
  added: []
  patterns: [section-entry-pipeline, assembler-delegation, pre-fetch-then-assemble]

key-files:
  created: []
  modified:
    - src/agent/prepare_context.rs
    - src/agent/context.rs
    - src/agent/agent_core.rs

key-decisions:
  - "Runtime sections injected into developer message (not system message) for cloud path"
  - "Working memory and tool patterns merged into single WorkingMemory SectionEntry"
  - "Local path converts PromptBlock runtime_blocks to SectionEntry via title-based section mapping"
  - "Context window reverse-engineered from system_prompt_cap for assembler budget calculation"

patterns-established:
  - "Section pipeline: collect as Vec<SectionEntry> -> pass to assembler -> render into messages"
  - "inject_runtime_sections() pattern: find developer message, append rendered content with separators"

requirements-completed: [PROMPT-03]

# Metrics
duration: 8min
completed: 2026-03-07
---

# Phase 1 Plan 2: Assembler Wiring Summary

**CloudAssembler and LocalAssembler wired into ContextBuilder, eliminating all 4 append_to_system_prompt() calls from prepare_context.rs with typed SectionEntry pipeline**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-07T13:33:24Z
- **Completed:** 2026-03-07T13:41:30Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- All 4 append_to_system_prompt() calls in prepare_context.rs replaced with typed SectionEntry pre-fetching via collect_cloud_runtime_sections()
- build_messages() cloud path delegates to CloudAssembler.assemble() for budget-aware system+developer message construction
- build_local_system_prompt() delegates to LocalAssembler.assemble() with section-based assembly
- REPL /context command backward-compatible (PromptBlockReport fields unchanged)
- 4190 tests pass, zero regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Refactor prepare_context.rs to pre-fetch sections into AssemblyContext** - `52f2181` (feat)
2. **Task 2: Wire PromptAssembler into ContextBuilder** - `fd588af` (feat)
3. **Task 3: Verify REPL /context command and run full test suite** - verification only, no code changes

## Files Created/Modified
- `src/agent/prepare_context.rs` - collect_cloud_runtime_sections() replaces 4 append_to_system_prompt() calls; inject_runtime_sections() called for cloud path
- `src/agent/context.rs` - collect_static_sections(), _build_skills_content(), _collect_local_sections(), inject_runtime_sections() added; build_messages() and build_local_system_prompt() delegate to assemblers
- `src/agent/agent_core.rs` - append_to_system_prompt() documented as sole remaining caller: agent_shared.rs trio orchestration

## Decisions Made
- Runtime sections go into the developer message (not system) for cloud path, preserving the system+developer role split
- Working memory and tool patterns are merged into a single WorkingMemory SectionEntry (matching the pre-existing behavior where tool patterns were appended to working memory text)
- Context window for the assembler is reverse-engineered from system_prompt_cap (cap = 40% of window for cloud, 30% for local)
- Local prompt path converts PromptBlock runtime_blocks to SectionEntry via title-based section mapping rather than requiring callers to change

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- The 4 append_to_system_prompt() calls were already failing to compile (function not imported after Plan 01 changes). This made the refactor straightforward since the old code path was already broken.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All prompt content now flows through typed SectionEntry -> PromptAssembler pipeline
- agent_shared.rs trio orchestration injection is the sole remaining post-assembly mutation (documented, Phase 5 scope)
- Phase 1 (Prompt Contract) is complete: types defined (Plan 01) and wired (Plan 02)

## Self-Check: PASSED

- src/agent/prepare_context.rs: FOUND
- src/agent/context.rs: FOUND
- src/agent/agent_core.rs: FOUND
- Commit 52f2181: FOUND
- Commit fd588af: FOUND

---
*Phase: 01-prompt-contract*
*Completed: 2026-03-07*
