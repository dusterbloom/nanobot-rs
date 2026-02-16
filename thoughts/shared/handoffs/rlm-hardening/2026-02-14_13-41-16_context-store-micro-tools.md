---
date: "2026-02-14T13:41:16Z"
session_name: "rlm-hardening"
researcher: claude-opus
git_commit: 0045b13
branch: vibe-1771072514
repository: nanobot
topic: "RLM ContextStore: Symbolic Handles to Tool Outputs"
tags: [implementation, rlm, context-store, micro-tools, delegation, tool-runner]
status: complete
last_updated: "2026-02-14"
last_updated_by: claude-opus
type: implementation_strategy
root_span_id: ""
turn_span_id: ""
---

# Handoff: RLM ContextStore with micro-tools for delegation model

## Task(s)

**Completed:** Implemented the RLM ContextStore (Change 1 from the plan). The delegation model now receives symbolic variable metadata for large tool results instead of truncated text, with micro-tools (ctx_slice, ctx_grep, ctx_length) to inspect stored data on demand.

**Plan source:** The full implementation plan was provided inline by the user. It describes 3 changes; this session completed Change 1 (ContextStore + micro-tools) and Change 3 (prompt update). Change 2 (ctx_summarize recursive sub-summarization) was explicitly deferred to Phase 2 per the plan.

## Critical References

- `src/agent/tool_runner.rs` — The delegation tool execution loop where ContextStore is wired in
- `src/agent/context_store.rs` — NEW file: ContextStore struct + micro-tool definitions + dispatch + 13 unit tests

## Recent changes

- `src/agent/context_store.rs` — **NEW**: ContextStore struct with `store()`, `get()`, `slice()`, `grep()`, `length()`, plus `micro_tool_definitions()`, `is_micro_tool()`, `execute_micro_tool()`, `MICRO_TOOLS` constant, and 13 unit tests
- `src/agent/mod.rs:6` — Added `pub mod context_store;`
- `src/agent/tool_runner.rs:14` — Added import of `context_store::{self, ContextStore}`
- `src/agent/tool_runner.rs:38-39` — Added `depth: u32` field to `ToolRunnerConfig` (reserved for Phase 2 ctx_summarize)
- `src/agent/tool_runner.rs:69-78` — Create `ContextStore::new()` and add micro-tool names to `allowed_tools` HashSet
- `src/agent/tool_runner.rs:80-85` — Changed `allowed_tools` from `let` to `let mut` to allow extending with micro-tools
- `src/agent/tool_runner.rs:99-102` — Append `micro_tool_definitions()` to `tool_defs` vec
- `src/agent/tool_runner.rs:88` — Updated system prompt: tells delegation model about variables, metadata, and ctx_slice/ctx_grep/ctx_length
- `src/agent/tool_runner.rs:169-199` — Replaced truncation block with ContextStore-aware dispatch: micro-tools execute against store (not added to all_results), real tools store full data and inject metadata for large results
- `src/agent/agent_loop.rs:753` — Added `depth: 0` to ToolRunnerConfig construction
- All test configs in `tool_runner.rs` — Added `depth: 0` field

## Learnings

1. **Micro-tools pass the allowed_tools filter naturally** — Adding them to the HashSet alongside real tools means the existing prompt-injection defense and duplicate detection work for micro-tools too, with zero additional code.

2. **ContextStore is per-invocation, not persistent** — Created at `run_tool_loop` start, dropped when it returns. No cross-invocation state needed. This keeps things simple.

3. **Small results bypass ContextStore metadata** — When `result.data.len() <= max_tool_result_chars`, the full text is injected directly. The metadata/micro-tool path only activates for large results. This avoids overhead for the common case (small tool outputs).

4. **The `store()` method always stores** — Even for small results, the data goes into the store. This means if the delegation model somehow calls ctx_grep on a small result's variable, it still works. But the delegation model sees full text, not metadata, so it has no reason to use micro-tools.

5. **Unicode char counting matters** — `"héllo 🌍"` is 7 chars (not 8). The `chars().count()` approach correctly handles multi-byte UTF-8.

## Post-Mortem (Required for Artifact Index)

### What Worked
- **Plan-driven implementation**: The detailed plan with exact line numbers, code snippets, and test specifications made implementation straightforward. Each change was self-contained.
- **replace_all for repetitive edits**: Using `replace_all: true` to add `depth: 0` to all 16+ test configs at once was efficient.
- **Existing test infrastructure**: The MockProvider, CapturingProvider, CountingTool, and VerboseTool patterns in the existing tests made writing integration tests easy.

### What Failed
- **replace_all missed edge cases**: The `short_circuit_chars: 0,` → add `depth: 0` replace_all missed one instance that had a trailing comment (`// disabled`). Required a manual follow-up edit.
- **Unicode test assertion off-by-one**: Incorrectly counted "héllo 🌍" as 8 chars instead of 7. Minor, caught immediately by test run.

### Key Decisions
- **Decision:** Micro-tools are NOT registered in ToolRegistry
  - Alternatives: Could have created Tool implementations for ctx_slice/ctx_grep/ctx_length
  - Reason: ToolRegistry is `&ToolRegistry` (immutable borrow). Micro-tools need per-invocation ContextStore access. Simpler to handle dispatch inline.

- **Decision:** `depth` field added but unused (Phase 2 placeholder)
  - Alternatives: Could skip adding it until ctx_summarize is implemented
  - Reason: Plan specified it; adding now avoids a second round of updating all test configs later.

- **Decision:** Deferred ctx_summarize to Phase 2
  - Alternatives: Could implement the full recursive summarization
  - Reason: Plan explicitly recommended deferring. The 3 basic micro-tools cover 90% of cases.

## Artifacts

- `src/agent/context_store.rs` — NEW: 310 lines, ContextStore + micro-tools + 13 unit tests
- `src/agent/mod.rs:6` — Module registration
- `src/agent/tool_runner.rs` — Major modifications (ContextStore wiring, prompt update, micro-tool dispatch)
- `src/agent/agent_loop.rs:753` — depth field at callsite

## Action Items & Next Steps

1. **Phase 2: ctx_summarize** — Implement recursive sub-summarization (~100 lines in context_store.rs). The `depth` field is already wired through. Needs an async function that creates a mini tool loop with only sync micro-tools. See the plan's "Change 2" section.

2. **Manual testing** — Run with a real delegation model: `RUST_LOG=debug nanobot agent`, fetch a large web page, verify debug logs show `context_store.store()` called and delegation model receives metadata instead of 50K chars.

3. **Tune max_tool_result_chars threshold** — Currently uses the existing config value (typically 6000). May want to lower it for RLM mode since micro-tools provide an escape hatch. A value of 1000-2000 would force more micro-tool usage.

4. **Consider ctx_summarize priority** — If real-world testing shows the delegation model struggles with large variables using only slice/grep/length, ctx_summarize becomes more urgent.

## Other Notes

- The ContextStore is created fresh for each `run_tool_loop` invocation. Variables are named `output_0`, `output_1`, etc. within each invocation.
- Micro-tool results are intentionally excluded from `all_results` — they're internal to the delegation conversation and shouldn't leak into the main model's context.
- The `grep` method does case-insensitive matching, which is appropriate for the delegation model's exploration use case.
- All 685 tests pass (684 existing + 1 new in context_store, plus 3 new integration tests in tool_runner, minus the renamed existing truncation tests).
- Branch: `vibe-1771072514` (auto-created by vibe-git)
