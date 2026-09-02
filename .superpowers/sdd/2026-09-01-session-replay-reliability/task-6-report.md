# Task 6 Report — Compound deterministic replay gate

## Scope

- Added one test-only `CompoundReplayProvider` inside
  `compound_session_failure_replay`.
- Used the existing production `AgentLoop`, tool registry/engine, SQLite session
  journal, replay loader, lease state machine, and terminal finalizer.
- Used an in-process `127.0.0.1:0` HTTP fixture. No external network, user DB,
  or workspace state was read or written.
- Review exposed a production replay gap: rejected tool receipts were durable
  messages but had no immutable `tool_results` row. The narrow production fix
  touches `src/agent/agent_loop/shared.rs` and `src/session/db.rs`: lease,
  router-guard, and terminal rejections now commit carrier/receipt messages and
  exact `ok=false` raw rows in one SQLite transaction. The ordinary executed
  result writer and `add_messages_checked` contract remain unchanged.

## Covered compound sequence

1. Successful evidence-producing `curl` result.
2. Upstream `false | curl ...` pipeline failure, proving pipefail semantics.
3. Zero-exit GitHub primary rate-limit JSON classified as failure.
4. Zero-exit GitHub secondary rate-limit JSON classified as failure.
5. Eight varied metered loopback `curl` calls, filling the normal 12-call lease.
6. Four further calls (a repeated pair and two distinct calls) rejected by the
   lease without reaching the server.
7. Exactly one buffered `ToolChoice::None` terminal call producing prose.
8. A separate streaming request whose terminal response contains no content,
   persisting the `empty` turn outcome.

The test asserts:

- successful turn outcome `finished` and empty turn outcome `empty`;
- exactly one terminal `None` call and one emitted terminal prose delta;
- terminal replay purpose `Continuation`, `tool_choice == "none"`,
  `streaming == false`, and byte-equal serialized tool catalogs versus the
  preceding streaming main request;
- pipeline and both GitHub failures have `ok == false` in `ToolExecute`, raw
  `tool_results`, and model-visible receipts;
- successful evidence has `ok == true`;
- carrier and receipt call vectors have equal length, unique IDs on both sides,
  and exactly equal ID multisets;
- all four blocked calls have durable rejected pre-execution decisions, false
  receipts, byte-equal false `tool_results` rows, no execution event, and no
  loopback request;
- exactly 12 network executions, within
  `DEFAULT_TOOLS_PER_LEASE * (1 + DEFAULT_MAX_LEASES_PER_TURN)`;
- no retired response-boundary or repeat-nudge scaffold text persists.

## Load-bearing mutation proof

After the first GREEN run, `apply_patch` temporarily removed only:

```rust
|| normalized.contains("secondary rate limit")
```

from `detect_api_error_body`. The unchanged compound test failed as required:

```text
assertion `left == right` failed: event status for tc_compound_secondary
  left: Some(true)
 right: Some(false)
test result: FAILED. 0 passed; 1 failed
```

The exact arm was restored with `apply_patch`. `git diff --
src/agent/tool_engine.rs` then produced no output. No mutation was committed.

## Review-driven RED and transactional fix

The strengthened compound assertion first failed against production as
required:

```text
blocked call lacks durable raw tool_results row:
tc_compound_blocked_repeat_1
test result: FAILED. 0 passed; 1 failed
```

`SessionDb::add_rejected_tool_messages_checked` now uses one transaction for
the protocol messages and every `role=tool, ok=false` immutable row. It reuses
the exact existing insert/read-back/digest/name/body/status comparison through
an extracted transaction-scoped helper. `Stored` and byte-identical retries
commit; `Conflict`/`Failed` roll back the complete batch. Direct and loop-level
regressions prove:

- terminal, router-guard, and lease rejections persist false raw rows whose
  bytes equal their model-visible receipts;
- a message insert fault rolls back the carrier, receipt, and raw row;
- an immutable raw-row conflict rolls back newly inserted carrier/receipt
  messages and preserves the original row;
- a later terminal-decision journal fault does not erase the already committed
  atomic protocol group.

GitNexus impact was run before production edits. The immutable result writer
was **CRITICAL** (282 upstream symbols, 51 processes, 20 modules), and the
shared message insertion helper was **HIGH** (9 upstream symbols, one app-run
process). This was reported before proceeding. The implementation therefore
extracts the old semantics verbatim and adds one dedicated transaction API
instead of changing the ordinary writer's public behavior.

## GREEN verification

```text
cargo test --release compound_session_failure_replay -- --nocapture --test-threads=1
  1 passed; 0 failed; production-loop test body 0.59s

cargo test --release exact_turn_replay_survives_workspace_prompt_changes -- --nocapture
  1 passed; 0 failed

cargo test --release terminal_no_tools_ -- --nocapture --test-threads=1
  8 passed; 0 failed

cargo test --release mixed_guard_ -- --nocapture --test-threads=1
  2 passed; 0 failed

cargo test --release lease -- --nocapture --test-threads=1
  25 passed; 0 failed

cargo test --release exact_replay_ -- --nocapture --test-threads=1
  6 passed; 0 failed

cargo test --release rejected_message_batch_rolls_back_when_raw_row_conflicts -- --nocapture --test-threads=1
  1 passed; 0 failed

cargo test --release --test protocol_invariants -- --nocapture
  6 passed; 0 failed

cargo test --release --test protocol_tests -- --nocapture
  24 passed; 0 failed

cargo test --release --test lcm_e2e_tests -- --nocapture
  9 passed; 0 failed

cargo test --release
  2,886 passed; 0 failed; 27 ignored across all targets

cargo build --release
  exit 0; release profile finished successfully

rustfmt --edition 2021 --check src/agent/agent_loop/shared.rs src/agent/agent_loop/tests.rs
  exit 0

rustfmt --edition 2021 --check src/session/db.rs
  reports only two pre-existing deviations at lines 5122 and 5191, outside
  every Task 6 hunk; no global formatting rewrite was applied

git diff --check
  exit 0

node /Users/peppi/Dev/nanobot-rs/.gitnexus/run.cjs detect-changes --repo /private/tmp/nr-srr
  6 files / 15 symbols; 0 affected processes; LOW risk
```

GitNexus included pre-existing user changes to `AGENTS.md` and `CLAUDE.md` in
its six-file count. Task 6 does not stage or modify those files. All full-build
warnings are pre-existing unused/dead-code warnings outside Task 6 ownership.

## Risks and limitations

- The test intentionally uses the local OS `curl` executable because this is
  the real audited execution path. CI must provide `curl`, as existing project
  behavior and release environments already do.
- Loopback binding can be denied by an unusually restrictive test sandbox;
  the test makes no outbound request and binds only an ephemeral localhost
  port.
- The transaction API intentionally recognizes rejected rows by the existing
  protocol shape (`role=tool`, `ok=false`). A future new rejection source must
  route its protocol group through the rejected persistence method.
- This gate and fix add no playback framework, alternate production pipeline,
  or duplicate model-visible message.
