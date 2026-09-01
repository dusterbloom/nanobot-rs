# Task 3 implementation report

Status: complete

Commit: `a8591c22c3fe1f943a3f9ea7d22038026679f74f` (`fix(agent): fail closed and preserve turn outcomes`)

Deep-review follow-up commit: this commit (`fix(agent): close replay lifecycle gaps`)

Second deep-review follow-up commit: this commit (`fix(agent): preserve routed tool batches`)

## Second deep-review follow-up

The second independent review found two remaining router protocol defects. Both
were verified against the prior implementation before production changes:

1. Router-generated tool-call ids were derived only from turn and target, so a
   same-turn preflight/post-tool subagent pair or repeated planned tool target
   reused one replay lifecycle id. A per-turn monotonic allocator now supplies
   every synthetic subagent, pipeline, direct-tool, and planned-tool id.
2. ToolGuard rejections were removed before the common carrier. No-cache
   rejections disappeared entirely, and mixed batches carried only allowed
   calls. `RoutedToolBatch` now preserves original order and a typed
   execute/reject disposition for every routed call. The common hot path first
   persists one complete assistant carrier, then records every rejection,
   atomically persists false-status receipts, and only then permits allowed
   tools to enter lease enforcement/execution. Either persistence failure ends
   the turn as `Error` before an allowed side effect.

No parallel router pipeline, semantic flag, SQL outcome, or schema change was
introduced. Repeated blocked rounds emit the scaffold once (at round two)
rather than duplicating it at round three; this retains the newly mandatory
carrier/receipt bytes without crossing the history window and preserves the KV
prefix invariant.

Second follow-up staged paths are exactly:

- `src/agent/agent_loop/shared.rs`
- `src/agent/agent_loop/tests.rs`
- `src/agent/prepare_context.rs`
- `src/agent/router.rs`
- `.superpowers/sdd/2026-09-01-session-replay-reliability/task-3-report.md`

### Second follow-up RED evidence

- Preflight/post-tool subagent reproduction generated the same
  `router-7-subagent-spawn` id twice.
- All-blocked/no-cache reproduction had neither assistant carrier nor tool
  receipt for `tc_guard_4`.
- Mixed reproduction carried only the allowed id and dropped the rejected id.
- The first full release run exposed one prefix regression after mandatory
  receipts increased durable history: a redundant third-round scaffold pushed
  reload across the history window. Restricting the scaffold to its intended
  second round made the prefix regression green.

### Second follow-up GREEN evidence

- Exact-replay synthetic id matrix: 2 passed, 0 failed (same-turn
  preflight/post subagent; repeated same-target planned tool).
- Guard protocol matrix: 2 passed, 0 failed (all-blocked/no-cache and mixed
  allowed/rejected), including full carrier membership, false rejected
  receipts, allowed execution, provider pairing, and
  `ReplayAvailability::Exact`.
- Rejected-receipt fault injection: 1 passed, 0 failed; the allowed member had
  no pre-execute event, provider call count stopped at four, and terminal
  outcome was `error`.
- Router suite: 65 passed, 0 failed, 1 ignored.
- Prefix circuit-breaker regression: 1 passed, 0 failed.
- `cargo test --release`: 2,874 passed, 0 failed, 27 ignored across unit,
  integration, and doc-test targets.
- `cargo build --release`: passed.
- `git diff --check`: passed.

GitNexus second-follow-up impacts were LOW for `router_preflight` (3 upstream),
`route_tool_calls` (4), and `step_execute_tools` (3). `TurnContext` resolved LOW
for the struct and UNKNOWN for its impl; `RouteResult` and the new allocator
were UNKNOWN/unindexed. The index was three commits behind. No HIGH or CRITICAL
impact was returned for this follow-up.

Pre-commit `detect-changes --repo nanobot-rs` reported the expected MEDIUM
scope: 5 files, 38 indexed symbols, and the two `step_execute_tools` execution
flows (`Now` and `Commit`).

Second-follow-up residual risks:

- A rejected-receipt database failure can leave the already durable carrier
  without its receipt, but the turn is terminal `Error` and no allowed tool or
  later provider call runs; the carrier cannot be safely rolled back after it
  has become the lifecycle prerequisite.
- The monotonic id counter uses checked addition and fails before emitting a
  duplicate if a single turn somehow exhausts all `u64` ids.
- `scripts/turn_bench.sh` requires a live local provider and was not run in the
  offline verification environment; the cache-prefix regression directly
  validates the changed prompt-history behavior.

## Deep-review follow-up

The independent review found seven additional convergence and replay gaps. All
were reproduced or verified against the old implementation before the hot path
was changed:

1. Sequential inline and delegated batches could execute later calls after a
   raw-result or post-result write failed. Production now executes one serial
   call or one bounded adjacent `ParallelSafe` chunk, persists the entire chunk
   lifecycle, and only then starts the next chunk. Configured delegation is a
   thin wrapper over this same inline executor; its alternate tool-runner loop
   was removed.
2. Strict-router preflight tool/subagent/pipeline actions and post-tool subagent
   actions executed outside the carrier lifecycle. They now produce ordinary
   `ToolCallRequest` values which traverse canonicalization, guard, carrier,
   pre-execute, raw-result, model-visible result, and post-result persistence.
3. Main-model failure journaling and empty-response rescue journaling no longer
   permit retries or fallbacks after their prerequisite write fails.
4. Auxiliary errors use `AuxiliaryCallError::{Persistence,Call}`. Provider text
   cannot impersonate a persistence fault through a magic string prefix, and
   every persistence variant maps to terminal `Error`.
5. If the final streamed response cannot be journaled, the stream-control
   contract emits `RetractReply`, clears streamed state, and only then emits the
   infrastructure error.
6. Per-iteration completion carries its candidate `TurnOutcome`; the outer loop
   installs it only on actual terminal exit, so an empty plan step cannot poison
   a later successful step.
7. The post-reply `turn_finished` fault test now explicitly asserts
   `ReplayAvailability::Incomplete` while preserving delivery of the already
   durable reply.

Follow-up staged paths are exactly:

- `src/agent/agent_loop/response.rs`
- `src/agent/agent_loop/shared.rs`
- `src/agent/agent_loop/tests.rs`
- `src/agent/router.rs`
- `src/agent/tool_engine.rs`
- `.superpowers/sdd/2026-09-01-session-replay-reliability/task-3-report.md`

No SQL outcome value or schema changed.

### Follow-up RED evidence

- Raw-result two-call reproduction: the second sequential tool executed.
- Post-result two-call reproduction: the second sequential tool executed.
- Delegated reproduction: the second write existed after the first raw-result
  fault.
- Strict router preflight reproduction: direct dispatch bypassed the injected
  pre-execute failure and reached the main response.
- Model-failure reproduction: a retry occurred after `model_failed` journaling
  failed.
- Empty-rescue reproduction: fallback continued after rescue-response
  journaling failed.
- Stream reproduction: visible deltas were not retracted after response
  journaling failed.
- Plan reproduction: an empty first step left the later successful turn with
  outcome `empty`.

### Follow-up GREEN evidence

- Review fault/terminal matrix: 9 passed, 0 failed (raw, post, delegated,
  strict-router lifecycle, model-failure, empty-rescue, stream retraction,
  empty-then-success, and post-reply incomplete replay).
- Typed provider-prefix regression: 1 passed, 0 failed.
- Subagent/pipeline ordinary spawn-call builders: 2 passed, 0 failed.
- `cargo test --release`: 2,869 passed, 0 failed, 27 ignored across unit,
  integration, and doc-test targets.
- `cargo build --release`: passed.
- `git diff --check`: passed.

GitNexus follow-up impact warnings were HIGH for
`execute_tool_calls_ordered`, `request_strict_router_decision`,
`journal_aux_request`, `journal_aux_terminal`, `persist_model_failure`, and
`run_agent_loop`; the shared worktree `detect-changes` result was CRITICAL
(27 files, 243 symbols, 171 flows) because it included concurrent Task 1/2
changes. The follow-up commit is intentionally limited to the six paths listed
above.

## Files

- `src/agent/agent_loop/shared.rs`
- `src/agent/agent_loop/response.rs`
- `src/agent/agent_loop/tests.rs`
- `src/agent/prepare_context.rs`
- `src/agent/finalize_response.rs`
- `src/agent/router.rs`
- `src/agent/tool_engine.rs`

`src/session/db.rs` and the SQL outcome schema were intentionally unchanged.

## RED evidence

Tests were added before production changes and run in release mode.

- `cargo test --release persistence_failure_prevents -- --nocapture` compiled and exposed the missing fail-closed behavior: inbound persistence failure still called the provider, and assistant carrier persistence failure still executed the tool. The existing pre-execute boundary already passed.
- `cargo test --release router_journal_failure -- --nocapture` failed because the router provider continued after its request artifact could not be recorded.
- The provider-failure regression showed that rendered error text was being classified as a successful `finished` turn.
- The raw-result and post-result fault tests initially used relative paths, which created fixtures outside their temporary workspaces. After correcting those paths to absolute temporary paths, both tests exercised only the intended persistence faults.

## GREEN evidence

All commands were release-only.

- `persistence_failure_prevents`: 3 passed, 0 failed
- `result_persistence_failure`: 2 passed, 0 failed
- `router_journal_failure`: 1 passed, 0 failed
- `test_failed_local_call_does_not_seed_prompt_cache_marker`: 1 passed, 0 failed
- `empty_provider_content_persists_empty_outcome`: 1 passed, 0 failed
- `cancelled_turn_persists_cancelled_outcome_without_provider_call`: 1 passed, 0 failed
- `plain_text_response_is_final_answer`: 1 passed, 0 failed
- `iteration_limit_persists_limit_exhausted_outcome`: 1 passed, 0 failed
- `turn_finish_journal_failure_still_returns_the_reply`: 1 passed, 0 failed
- `test_tool_call_carrier_persists_before_tool_result`: 1 passed, 0 failed
- `test_tool_round_is_durable_before_next_provider_call_completes`: 1 passed, 0 failed
- `checked_message_batch_rolls_back_when_middle_insert_fails`: 1 passed, 0 failed
- `cargo test --release`: 2,875 passed, 0 failed, 27 ignored across unit, integration, and doc-test targets
- `cargo build --release`: passed
- `cargo check --release`: passed
- `git diff --check`: passed

Only pre-existing warnings appeared (`unused_mut` in `cua.rs` and test-only `agent_core.rs`, plus the unused `Emergency` retention variant). The repository-wide `cargo fmt --all -- --check` encounters existing formatting drift outside Task 3's diff, so no bulk formatter was applied.

## Persistence and side-effect boundaries

| Fault boundary | Observed calls after fault | Durable terminal outcome |
| --- | --- | --- |
| Inbound user batch | provider 0, tool 0 | `error` |
| Router auxiliary request | auxiliary provider 0 | `error` |
| Assistant tool-call carrier | main provider 1, tool 0 | `error` |
| Tool pre-execute decision | main provider 1, tool 0 | `error` |
| Raw tool-result event | unavoidable tool call exactly once, main provider remains 1 | `error` |
| Model-visible post-result receipt | unavoidable tool call exactly once, main provider remains 1 | `error` |
| Final assistant message | main provider 1; undurable reply is replaced by session error | `error` |
| Post-reply `turn_finished` journal | no new side effect; already durable reply is still returned | replay remains incomplete by design |

Auxiliary request journaling now fails before an auxiliary provider call. Auxiliary response/failure journaling returns `Result`, and an unrecorded provider result is not consumed by the turn. Delegated and inline tool paths stop before any subsequent provider or tool side effect when carrier, decision, raw-result, or model-visible receipt persistence fails.

## Terminal outcomes

- `Finished`: ordinary non-empty final prose.
- `Error`: provider errors and protocol/session persistence failures, even when the rendered text is non-empty.
- `Cancelled`: cancellation before the provider boundary, with zero provider calls.
- `Empty`: empty content/SSE recovery exhaustion, including the user-facing fallback text.
- `LimitExhausted`: iteration budget exhausted after a replay-safe tool round.

Finalization writes `TurnOutcome::wire_str()` directly and no longer infers success from reply text. The special post-reply journal-failure rule is preserved: once the assistant reply itself is durable, a failing `turn_finished` write warns but does not discard the reply.

Provider error rendering preserves the exact `[LLM Error] {provider detail}` evidence. A failed local health probe appends only that the health endpoint was unavailable; it does not claim that the server crashed.

## GitNexus and scope

Pre-edit impact results included:

- `persist_pending_protocol_messages`: CRITICAL, 27 impacts / 8 direct / 8 processes / 3 modules
- `process_message`: CRITICAL, 64 impacts / 4 direct / 10 processes / 4 modules
- `run_agent_loop`: HIGH, 17 impacts / 1 direct / 4 processes / 3 modules
- `journal_aux_request`: HIGH, 10 impacts / 2 direct
- `journal_aux_terminal`: HIGH, 10 impacts / 2 direct
- `request_strict_router_decision`: HIGH, 9 impacts / 5 direct
- `recorded_auxiliary_chat`: LOW, 4 impacts / 2 direct / 1 process / 2 modules
- `journal_tool_call_carrier`: LOW, 3 impacts
- tool execution helpers: LOW
- new typed outcome/result symbols: UNKNOWN until indexed

The index was three commits behind during impact analysis. All HIGH/CRITICAL findings were reported before editing and acknowledged.

Pre-commit `detect-changes --repo nanobot-rs` reported the expected CRITICAL hot-path scope: 7 files, 42 indexed symbols, and 19 affected execution flows. The mapped flows cover response handling, finalization, routing, protocol persistence, and inline/delegated tool execution; no unrelated source file was changed.

## Residual risks

- Post-side-effect persistence can only stop subsequent work; it cannot undo a tool side effect that already occurred. Fault tests prove the tool runs once and no later provider/tool call follows.
- A final `turn_finished` journal failure intentionally remains non-fatal after the reply message is durable, so replay is incomplete while the user still receives the reply.
- Parallel-safe siblings in one already-started chunk are all-settled before
  persistence is evaluated; a post-side-effect failure prevents every later
  chunk but cannot undo sibling calls already started concurrently.
- Configured tool delegation now uses the common inline executor. This removes
  the old scratch-pad model's ability to invent extra internal tool calls in
  exchange for one auditable lifecycle for every provider-selected call.
- The GitNexus index lag may make line-level mappings incomplete, although its reported critical flows match the explicitly edited hot path.
