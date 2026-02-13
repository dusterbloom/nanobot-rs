---
date: "2026-02-13T14:46:57Z"
session_name: "channels-expansion"
researcher: Claude
git_commit: e69877c3caca33d444d5378e16be5bf095457da1
branch: vibe-1770922299
repository: nanobot
topic: "Agent Provenance Protocol (APP) Implementation"
tags: [implementation, provenance, audit-log, claim-verification, tool-visibility]
status: complete
last_updated: "2026-02-13"
last_updated_by: Claude
type: implementation_strategy
root_span_id: ""
turn_span_id: ""
---

# Handoff: Agent Provenance Protocol - Full 5-Phase Implementation

## Task(s)

Implemented the full Agent Provenance Protocol (APP) across 5 phases, following the plan at `~/.claude/plans/frolicking-growing-sphinx.md`. All phases are **COMPLETED**:

- [x] **Phase 1a**: ProvenanceConfig added to `src/config/schema.rs`, wired into SharedCore
- [x] **Phase 1b**: AuditLog module created at `src/agent/audit.rs` with SHA-256 hash chain
- [x] **Phase 1c**: Audit recording wired into `process_message()` for both inline and delegated tool paths
- [x] **Phase 2**: Verification Protocol rules injected into system prompt via `ContextBuilder`
- [x] **Phase 3**: Tool call visibility in REPL - `tool_event_tx` plumbed through agent loop, combined `tokio::select!` rendering
- [x] **Phase 4**: Mechanical claim verification - `ClaimVerifier` with regex extraction, `render_turn_with_provenance()` wired into REPL
- [x] **Phase 5**: REPL commands `/audit`, `/verify`, `/provenance` + help text updated

All 534 tests pass, 0 failures.

## Critical References

- Plan document: `~/.claude/plans/frolicking-growing-sphinx.md` (the full APP implementation plan)
- Protocol spec: `~/.nanobot/workspace/docs/agent-provenance-protocol.md`
- Config pattern reference: `src/config/schema.rs` (MemoryConfig/ToolDelegationConfig patterns)

## Recent changes

- `src/config/schema.rs`: Added `ProvenanceConfig` struct (6 fields) and `pub provenance: ProvenanceConfig` to root Config
- `src/agent/audit.rs` (NEW): `AuditLog` + `AuditEntry` + `ToolEvent` enum (~485 lines) with hash chain, file locking, JSONL append
- `src/agent/provenance.rs` (NEW): `ClaimVerifier` + `ClaimStatus` + `AnnotatedClaim` (~333 lines) with 5 regex extraction methods
- `src/agent/mod.rs`: Registered `pub mod audit;` and `pub mod provenance;`
- `src/agent/agent_loop.rs:20`: Changed import to `use crate::agent::audit::{AuditLog, ToolEvent};`
- `src/agent/agent_loop.rs:350-355`: Added `tool_event_tx: Option<UnboundedSender<ToolEvent>>` to `process_message()`
- `src/agent/agent_loop.rs:380-384`: Audit log creation after session_key
- `src/agent/agent_loop.rs:725-748`: Inline tool path: CallStart/CallEnd events + audit recording
- `src/agent/agent_loop.rs:586-615`: Delegated tool path: CallStart events before run_tool_loop, CallEnd events in results loop
- `src/agent/agent_loop.rs:1088-1096`: Added `tool_event_tx` param to `process_direct_streaming()`
- `src/agent/context.rs`: Added `pub provenance_enabled: bool` field, verification rules in `build_system_prompt()`
- `src/cli.rs`: Both `build_core_handle()` and `rebuild_core()` pass `config.provenance.clone()` to `build_shared_core()`
- `src/syntax.rs:71-123`: Added `render_turn_with_provenance()` with colored markers
- `src/repl.rs:17-19`: Added imports for `AuditLog`, `ToolEvent`, `ClaimVerifier`, `ClaimStatus`
- `src/repl.rs:107-230`: Rewrote `stream_and_render()` with tool event channel, `tokio::select!` combined print task, provenance re-render
- `src/repl.rs:1176-1270`: Added `/audit`, `/verify`, `/provenance` command handlers
- `Cargo.toml`: Added `sha2 = "0.10"` dependency

## Learnings

1. **SharedCore doesn't implement Clone** (has `AtomicU64`/`AtomicBool` fields). Cannot create modified copies directly. The `/provenance` toggle rebuilds the entire core via `cli::rebuild_core()` with a cloned+modified `Config`.

2. **`tc.arguments` is `HashMap<String, Value>`**, not `serde_json::Value`. Required `serde_json::to_value()` conversion before passing to `audit.record()`.

3. **Delegated tool runner doesn't track per-tool timing**. CallEnd events for delegated tools use `duration_ms: 0`. Individual timing is only available for the inline path.

4. **File locking pattern** from `LearningStore` (`src/agent/learning.rs:237-259`) — `create_new` lockfile, drop removes. Reused for AuditLog.

5. **`stream_and_render()` line counting for erasure** — tool event lines printed during streaming must be added to text lines when calculating how many terminal rows to erase before re-rendering.

## Post-Mortem (Required for Artifact Index)

### What Worked
- Parallel phase execution: Phases 1a and 1b were independent and ran as parallel subagents, saving time
- The existing config pattern (`MemoryConfig`/`ToolDelegationConfig`) made `ProvenanceConfig` straightforward to wire
- Surgical edits to `process_message()` kept the blast radius small — tool event emission is opt-in via `Option<UnboundedSender>`
- `tokio::select!` with biased polling for the combined delta/tool-event print task works cleanly since text deltas and tool events are naturally interleaved (never concurrent)

### What Failed
- Subagent tasks reported "failed" status due to a hook error (`classifyHandoffIfNeeded is not defined`) even though the implementation work completed successfully. Had to verify manually.
- Initial attempt to use `HashMap<String, Value>` directly as `&Value` for audit args — type mismatch. Fixed by adding `serde_json::to_value()` conversion.
- First attempt at `/provenance` toggle tried to clone SharedCore directly — doesn't work. Changed to full `rebuild_core()` approach.

### Key Decisions
- Decision: Use channel-based tool event emission (`UnboundedSender<ToolEvent>`) rather than direct stdout printing from agent_loop
  - Alternatives: Direct printing, shared mutable state
  - Reason: Maintains separation between agent logic and display layer; REPL owns rendering
- Decision: ClaimVerifier uses pure regex + string matching, no LLM
  - Alternatives: LLM-based verification
  - Reason: Using an LLM to verify LLM output reintroduces the same trust problem
- Decision: Audit log uses SHA-256 hash chain with pipe-delimited fields
  - Alternatives: Simple append-only without integrity verification
  - Reason: Tamper detection is core to the provenance guarantee
- Decision: `/provenance` toggle rebuilds the entire SharedCore
  - Alternatives: Atomic flag on SharedCore
  - Reason: SharedCore doesn't impl Clone; rebuild_core is the established pattern for runtime config changes

## Artifacts

- `src/config/schema.rs` — ProvenanceConfig struct and root Config field
- `src/agent/audit.rs` (NEW) — AuditLog, AuditEntry, ToolEvent
- `src/agent/provenance.rs` (NEW) — ClaimVerifier, ClaimStatus, AnnotatedClaim
- `src/agent/mod.rs` — Module registrations
- `src/agent/agent_loop.rs` — tool_event_tx plumbing, audit recording, event emission
- `src/agent/context.rs` — provenance_enabled flag, verification rules in system prompt
- `src/cli.rs` — provenance config passthrough to build_shared_core
- `src/syntax.rs` — render_turn_with_provenance()
- `src/repl.rs` — stream_and_render() rewrite, /audit /verify /provenance commands
- `Cargo.toml` — sha2 dependency
- Plan: `~/.claude/plans/frolicking-growing-sphinx.md`

## Action Items & Next Steps

All 5 phases are complete. Potential future work:

1. **Manual testing** — Run with `provenance.enabled: true` in `~/.nanobot/config.json` to verify tool calls appear in REPL, audit JSONL written, and claim verification markers display correctly
2. **Tune ClaimVerifier regex patterns** — The 5 extraction patterns may need refinement based on real-world agent output
3. **Audit log cleanup** — Add a pruning mechanism or size cap for `{workspace}/memory/audit/` JSONL files over time
4. **Integration with session persistence** — Currently audit logs are separate from session JSONL; could cross-reference for richer history

## Other Notes

- Storage location for audit logs: `{workspace}/memory/audit/{session_key}.jsonl`
- ProvenanceConfig defaults: `enabled: false`, `audit_log: true`, `show_tool_calls: true`, `verify_claims: false`, `strict_mode: false`, `system_prompt_rules: true`
- To enable in config: `"provenance": {"enabled": true}` in `~/.nanobot/config.json`
- The `sha2` crate was already a transitive dependency (in Cargo.lock) — now direct
- The `regex` crate was already a direct dependency — used by ClaimVerifier
- Tool event rendering format: `  ▶ tool_name(args_preview)  ✓ 12ms` (dim cyan + green/red status)
- Claim status markers: ✓ green (Observed), ~ blue (Derived), ⚠ yellow (Claimed), ◇ dim (Recalled)
