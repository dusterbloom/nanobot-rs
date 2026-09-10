# Deterministic Fold and Constrained Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make required Higgs tool calls fail closed and make Nanobot's pressure-path compaction a deterministic, reversible, integrity-addressed fold.

**Architecture:** Keep both release hot paths. Higgs strengthens its existing `ConstrainedGenerator` and chat materialization contract. Nanobot strengthens `LcmEngine::compact` and `deterministic_recovery_index`; existing SQLite rows, summary nodes, and `lcm_expand` remain the only storage and recall mechanisms.

**Tech Stack:** Rust 2021, Tokio, MLX, Outlines FSM, SQLite, SHA-256, existing LCM DAG.

## Global Constraints

- Test first and observe the intended failure before production edits.
- Release builds and tests only.
- Do not change ordinary `tool_choice=auto` semantics.
- Do not change `render_tool_result_handle` or its byte format.
- Do not add a service, database table, feature flag, or new compaction module.
- Keep assistant tool-call carriers and matching tool results in one fold unit.
- Exact source rows remain append-only and recoverable through `lcm_expand`.

---

### Task 1: Fail-closed required tool calls in Higgs

**Files:**
- Modify: `/Users/peppi/Dev/higgs/crates/higgs-engine/src/constrained.rs`
- Modify: `/Users/peppi/Dev/higgs/crates/higgs-engine/src/simple.rs`
- Modify: `/Users/peppi/Dev/higgs/crates/higgs-engine/src/batch_engine.rs`
- Modify: `/Users/peppi/Dev/higgs/crates/higgs/src/routes/chat.rs`

**Interfaces:**
- Consumes: existing `ConstrainedGenerator`, `ToolChoicePlan::requires_call`, and `parse_tool_calls`.
- Produces: a required/named request cannot return success without exactly one
  parser-visible declared tool call whose arguments are an object. The existing
  grammar FSM, not a second schema validator, owns parameter-schema validity.

- [x] Add a real `constrained.rs` test whose FSM has no allowed token and assert `apply_mask` returns an error rather than unchanged logits.
- [x] Run `cargo test --release -p higgs-engine constrained -- --nocapture` and confirm the new test fails for the current fail-open behavior.
- [x] Make dead grammar states, empty allowlists, empty logits vocabularies, and all-OOV allowlists return a descriptive typed `EngineError`; never return unmasked logits.
- [x] Add tests for rejected FSM advancement, then make all five decode call sites terminate through a typed error instead of ignoring `advance(false)`. In `BatchEngine`, use the existing per-request terminal-error response path so one invalid constrained request does not abort unrelated batched requests.
- [x] Add blocking and streaming route tests proving `Required` rejects visible
  text, zero calls, undeclared/named-wrong calls, non-object arguments, and
  multiple calls while accepting exactly one grammar-produced declared call.
- [x] Add one private helper beside response materialization that checks the required-call postcondition for both routes; do not create a module or second parser.
- [x] Run focused release tests for `higgs-engine` and `higgs` chat routes, then `cargo build --release`.
- [x] Run GitNexus change detection and record the exact affected flows before committing.

### Task 2: Deterministic reversible fold in Nanobot

**Files:**
- Modify: `src/agent/lcm.rs`
- Test: `src/agent/lcm.rs`
- Test only if the public seam requires it: `tests/lcm_e2e_tests.rs`

**Interfaces:**
- Consumes: `LcmEngine::compact`, its existing complete-block selection, summary DAG nodes, and `lcm_expand`.
- Produces: `CompactionFailureMode::Deterministic` performs no provider call, replaces the complete compactible prefix once, preserves the recent raw tail, and always leaves an exact recall pointer when shrinking is possible.

- [x] Add a failing adversarial test containing user messages, assistant tool-call carriers, matching results, follow-up conclusions, and a protected newest user request.
- [x] Assert the deterministic fold makes zero provider calls, preserves the newest request raw, never splits any call/result pair, emits one level-0 node, shrinks active tokens, and expands every covered ID byte-for-byte.
- [x] Add a repeated-render assertion: identical source rows and configuration produce byte-identical checkpoint text.
- [x] Run the focused release test and confirm the missing integrity metadata or invariant causes failure.
- [x] Implement the minimum change inside `deterministic_recovery_index` and the existing block-selection path. Reuse `format_id_ranges`, `summary_wire_message`, `SummaryNode`, and `lcm_expand`; do not change the tool-result renderer.
- [x] Ensure the `<200 tokens` model-economics shortcut does not prevent deterministic pressure recovery when a fold can shrink the prompt.
- [x] Run focused LCM release tests and `tests/lcm_e2e_tests.rs`.

### Task 3: Integrity-addressed checkpoint metadata

**Files:**
- Modify: `src/agent/lcm.rs`
- Test: `src/agent/lcm.rs`

**Interfaces:**
- Consumes: the exact `(MessageId, Value)` rows already collected for deterministic recovery.
- Produces: checkpoint text containing `version=1`, exact compact source ranges, `revision=<greatest MessageId>`, `source_sha256=<64 lowercase hex>`, and the existing copyable `lcm_expand` call.

- [x] Extend the Task 2 test first to assert the exact metadata fields and verify the digest changes when any covered role, content, tool-call ID, or result byte changes.
- [x] Canonicalize by sorting covered rows by `MessageId`, then hash each ID and `serde_json::to_vec(message)` with unambiguous length framing using the already-installed `sha2` crate.
- [x] Render only bounded metadata plus newest-first orientation; do not copy raw tool bodies into the checkpoint.
- [x] Render the exact compact source-ID ranges on the wire and retain the identical source-ID set in the DAG node; never widen a fragmented set into a min-max range.
- [x] Run the complete `lcm` unit suite and LCM end-to-end release tests.
- [x] Run `scripts/turn_bench.sh` if the agent-loop/context build path changed; otherwise document why the pure LCM edit cannot affect ordinary turns before compaction.
- [x] Run GitNexus change detection, `cargo test --release`, and `cargo build --release` with zero new warnings before committing.

### Final integration and proof

- [x] Review each task diff against the design and reject new flags, modules, tables, or duplicated recall paths.
- [x] Apply only task commits over the current dirty baselines; preserve unrelated user work.
- [x] Run GitNexus change detection in both repositories.
- [x] Run both complete release test suites and builds.
- [x] Rebuild/install the exact Higgs and Nanobot binaries used by the end-to-end test.
- [x] In tmux, run Higgs plus Nanobot through a long tool-heavy session, required recovery call, capacity-triggered deterministic fold, exact `lcm_expand`, and resume. Capture logs proving no compactor inference ran on the deterministic path.

## Verification Record — 2026-09-10

- Higgs required/named tool calls fail closed through one grammar and one route
  postcondition. Focused suites passed: 17 constrained-engine tests and 59 chat
  route tests. The complete release suite and build passed with zero warnings.
- Nanobot deterministic pressure recovery made zero compactor calls and folded
  24,014 estimated tokens to a 973-token level-0 recovery node in under one
  second. Its mandatory header contains exact ranges, revision, SHA-256, and one
  copyable `lcm_expand` call.
- The production incident session resumed from a 31,822-token cold prompt, then
  folded under a reduced live capacity envelope and continued with a 4,159-token
  provider prompt.
- A restart audit found and fixed a representation split: the live engine had
  hashed replay-projected rows while restart expansion used raw SQLite rows.
  Compaction now hydrates only already-active IDs from SQLite immediately before
  folding; restart keeps exact durable rows in the immutable store while syncing
  only the provider-facing raw tail to bounded replay handles.
- The restart regression includes a cache-replay scaffold and a 13 KB tool body.
  It proves all covered IDs expand byte-for-byte from SQLite while the active
  prompt contains the bounded tool-result handle. The final complete Nanobot
  release suite passed 3,040 executed tests with zero failures or warnings.
- The ordinary no-DAG turn path does not load durable history or run active
  synchronization. Full-session hydration occurs only inside an actual
  compaction job, so the change does not add I/O to ordinary turns.
