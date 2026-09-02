# Task 6 Report — Compound deterministic replay gate

## Scope

- Modified only `src/agent/agent_loop/tests.rs`.
- Added one test-only `CompoundReplayProvider` inside
  `compound_session_failure_replay`; no production helper or behavior changed.
- Used the existing production `AgentLoop`, tool registry/engine, SQLite session
  journal, replay loader, lease state machine, and terminal finalizer.
- Used an in-process `127.0.0.1:0` HTTP fixture. No external network, user DB,
  or workspace state was read or written.

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
- every tool receipt has an assistant carrier, with equal carrier/receipt
  cardinality;
- all four blocked calls have durable rejected pre-execution decisions, false
  receipts, no execution event, and no loopback request;
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

## GREEN verification

```text
cargo test --release compound_session_failure_replay -- --nocapture --test-threads=1
  1 passed; 0 failed; production-loop test body 0.59s

cargo test --release exact_turn_replay_survives_workspace_prompt_changes -- --nocapture
  1 passed; 0 failed

cargo test --release --test protocol_invariants -- --nocapture
  6 passed; 0 failed

cargo test --release --test protocol_tests -- --nocapture
  24 passed; 0 failed

cargo test --release --test lcm_e2e_tests -- --nocapture
  9 passed; 0 failed

rustfmt --edition 2021 --check src/agent/agent_loop/tests.rs
  exit 0

git diff --check
  exit 0

node /Users/peppi/Dev/nanobot-rs/.gitnexus/run.cjs detect-changes --repo nanobot-rs
  No changes detected.
```

The post-restore compound test was rerun once more after formatting and passed
fresh. One earlier post-restore invocation received a test-environment
`EPERM` while binding `127.0.0.1:0` before any test logic ran; an immediate
rerun with the identical already-built binary passed in 0.59s. Both initial
GREEN and mutation RED runs had also bound the same fixture successfully.

Only pre-existing compiler warnings were emitted (unused/dead code outside
Task 6 ownership).

## Risks and limitations

- The test intentionally uses the local OS `curl` executable because this is
  the real audited execution path. CI must provide `curl`, as existing project
  behavior and release environments already do.
- Loopback binding can be denied by an unusually restrictive test sandbox;
  the test makes no outbound request and binds only an ephemeral localhost
  port.
- This gate protects the compound interaction among existing components. It
  deliberately adds no playback framework or alternate production pipeline.
