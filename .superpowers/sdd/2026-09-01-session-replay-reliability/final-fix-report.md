# Final whole-branch review fix

Status: complete

## Result

- Duplicate/all-blocked router rounds no longer inject a synthetic final-answer
  scaffold or return a static `Finished` reply. They persist their rejected
  receipts, set `round_executed_no_tools`, and converge through the shared
  at-most-once terminal `ToolChoice::None` authority recorded as
  `ModelCallPurpose::Continuation`.
- Router and lease classification now build one pending protocol group before
  any decision event or allowed side effect. The complete carrier, every
  rejected receipt, and each byte-identical `ok=false` raw row commit in one
  transaction; rejected decisions follow; only then may allowed calls enter
  the existing Ready/execution path.
- A failed protocol transaction discards the same pending messages in memory,
  preventing finalization from leaking a carrier or receipt through a later
  messages-only retry.
- A legacy immutable row with matching tool name/body and `ok IS NULL` is
  replay-compatible with an inferred status without a database backfill.
  Explicit `Some(false)`/`Some(true)` disagreement remains a conflict.
- Removed the obsolete zero-caller `journal_tool_call_carrier` wrapper. The
  canonical append helper and persistence implementations remain unchanged.

## RED evidence

- Duplicate regression returned `Tool calls were blocked after repeated
  duplicates...` instead of the scripted terminal summary and never issued the
  required terminal authority.
- Router raw-row fault observed `(carrier=true, receipt=true, raw=false)`;
  finalization had retried the failed pending receipt through the messages-only
  path.
- Lease message fault observed `(carrier=true, receipt=false, raw=false)`.
- Legacy NULL-status reopen replay left the tool message `ok` absent instead of
  inferred false and therefore could not render the large body as a handle.

## GREEN evidence

All commands used release mode.

- `cached_duplicate_rounds_use_one_terminal_none_without_scaffold_or_static_break`:
  passed; exactly one `ToolChoice::None`, one Continuation request with
  `tool_choice == "none"`, no retired scaffold/static reply persisted.
- `router_rejection_raw_and_decision_faults_preserve_transaction_order`:
  passed for raw-row and later decision-event faults.
- `lease_rejection_message_raw_and_decision_faults_preserve_transaction_order`:
  passed for message, raw-row, and later decision-event faults; no allowed
  side-effect file and no Ready event after any fault.
- Existing mixed guard, carrier failure, all-guard-blocked, terminal matrix
  (8/8), compound replay, immutable explicit-status conflict, and old-binary
  omitted-status migration regressions passed.
- `legacy_null_status_replays_large_failure_as_typed_handle_without_backfill`:
  passed after reopen; replay attaches false and renders a handle while the DB
  row remains `ok=NULL`.
- Unrestricted `cargo test --release`: 2,850 library tests passed, 0 failed,
  23 ignored; 9 LCM E2E, 6 protocol invariants, and 24 protocol tests passed;
  every remaining integration/doc target finished with zero failures.
- Unrestricted `cargo build --release`: passed.
- `rustfmt --check` passed for the changed router/loop/test/tool-engine files.
  `session/db.rs` retains only its two pre-existing formatting deviations
  outside this change; the new test hunk is rustfmt-shaped.
- `git diff --check`: passed.
- Final staged `detect-changes --repo /private/tmp/nr-srr`: LOW, 6 files / 9
  indexed symbols / 0 affected processes. A separate unstaged scan reports
  only the pre-existing `AGENTS.md` and `CLAUDE.md`; neither is part of this
  commit.

## GitNexus impact

- `route_tool_calls`: CRITICAL, 576 upstream, 1 direct, 58 processes, 20 modules.
- `RoutedToolBatch`: CRITICAL, 270 upstream, 1 direct, 51 processes, 20 modules.
- `journal_tool_call_carrier`: CRITICAL in the stale graph, 1 direct and 49
  processes; the Rust compiler showed zero callers after the reviewed ordering
  change, so only that obsolete wrapper was removed.
- `get_history`: LOW, 1 direct caller.
- `step_execute_tools`, `persist_pending_protocol_group`,
  `add_rejected_tool_messages_checked`, and
  `store_tool_result_immutable_locked`: UNKNOWN/unindexed.

All HIGH/CRITICAL warnings were reported to and acknowledged by the parent
before edits/removal. The changes stay on the existing channel → agent loop →
provider → tools → reply hot path and add no flag, router fallback, replay
pipeline, or schema migration.

## Files

- `src/agent/router.rs`
- `src/agent/agent_loop/shared.rs`
- `src/agent/agent_loop/tests.rs`
- `src/agent/tool_engine.rs`
- `src/session/db.rs`
- `.superpowers/sdd/2026-09-01-session-replay-reliability/final-fix-report.md`

`AGENTS.md` and `CLAUDE.md` are pre-existing unstaged user edits and were not
modified, staged, or reverted.

## Residual risks

- The GitNexus worktree index is stale/noisy for the hottest symbols; final
  change detection is therefore paired with the full unrestricted release
  suite and deterministic compound replay.
- A rejected-decision journal fault deliberately leaves the already committed
  carrier/receipt/raw group without that later audit decision, but no allowed
  call reaches Ready or execution and the turn ends as an infrastructure error.
- Legacy NULL status remains ambiguous on disk by design for rollback
  compatibility. Replay infers status from the exact body in memory and does
  not mutate the old row.
