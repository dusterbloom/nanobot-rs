# Task 4 implementation report

Status: complete

Commit: this commit (`fix(agent): replace convergence prompts with typed limit`)

## Result

The response-boundary and repeated-result prompt scaffolds were removed from
the production loop. Every natural iteration limit, repeated-call stop, and
no-progress hard stop now converges through one shared at-most-once authority:
an enum-selected blocking provider request with `ToolChoice::None`, thinking
disabled, `streaming:false`, and `ModelCallPurpose::Continuation`.

Terminal mode calls the provider directly. It does not re-enter preparation,
strict-trio router preflight, response processing, validation, forced-tool
recovery, continuation, routing, or tool execution. It freshly renders the
current message tail and reuses the exact frozen tool catalog and response
budget from the last normal provider request. The OpenAI-compatible HTTP path
disables its retry wrapper for `ToolChoice::None`; normal and required calls
retain their prior retry behavior.

## RED evidence

All implementation tests were written and run in release mode before the
production changes.

- The terminal provider matrix observed zero terminal calls and returned the
  static iteration fallback rather than the scripted terminal prose.
- The ignored-`None` case had no durable terminal carrier/rejected receipt
  because no typed terminal request existed.
- The provider error/empty cases observed zero terminal calls instead of one
  bounded attempt.
- The OpenAI-compatible retry reproduction observed two HTTP requests after a
  retryable `503` with `ToolChoice::None`; the contract requires exactly one.
- The provider-body characterization was corrected to compare the normal and
  terminal serialized catalogs after the existing schema normalization; this
  pins byte-stable tool arrays rather than the unnormalized test input.

## GREEN evidence

- `terminal_no_tools`: 5 passed, 0 failed. This covers prose, ignored tool
  calls, provider error/empty, stable serialized tools, and strict-trio router
  bypass.
- `terminal_none_request_is_not_retried_after_retryable_http_failure`: passed;
  the capture server observed exactly one HTTP request.
- `convergence_loop_terminates_without_mutating_tool_catalog`: passed and now
  asserts the looping provider observes exactly one `ToolChoice::None` call.
- `wire_prefix_stable_across_turn_after_side_effect`: passed.
- `cached_duplicate_tool_receipts_trip_loop_circuit_breaker`: passed.
- Repeat-breaker matrix: 4 passed, 0 failed.
- Lease-renewal matrix: 3 passed, 0 failed.
- Cancellation regressions for pre-provider cancellation, soft compaction, and
  hard compaction: 3 passed, 0 failed.
- Protocol integration suites: 6 passed and 24 passed, 0 failed.
- Final unrestricted `cargo test --release`: 2,835 passed, 0 failed, 23 ignored
  in the library target; every integration and doc-test target also completed
  with zero failures.
- `cargo build --release`: passed.
- `git diff --check`: passed.

Only pre-existing warnings appeared (`unused_mut` and existing dead-code
warnings outside the Task 4 paths).

## Terminal outcome matrix

| Terminal result | Durable protocol | Tool execution | Turn outcome |
| --- | --- | --- | --- |
| Non-empty prose | request + response | none | `finished` |
| Provider returns tool calls despite `None` | complete assistant carrier, rejected pre-execute decision, and matching `ok:false` tool receipt for every call | none | `limit_exhausted` |
| Empty content or provider-declared failed response | request + response | none | `limit_exhausted` |
| Provider call error | request + model failure | none; no HTTP/loop retry | `limit_exhausted` |
| Request/response/carrier/rejection/receipt persistence fault | stops at the failed durable boundary | none | `error` |
| Cancellation before terminal authority | no terminal request | none | `cancelled` |

Terminal prose is emitted through the existing final delta path exactly once.
Ignored terminal tool calls never enter the router, guard, lease, delegated, or
inline executors.

## Scaffold and catalog checks

Production searches return no matches for `ResponseBoundary`,
`advance_response_boundary`, `repeat_nudged`, `RepeatBreakerAction::Nudge`, or
the response-boundary wording. The three retired synthetic prompt strings
remain only as negative assertions over persisted messages.

The terminal request records the same final `tools` array used by the last
normal call. The provider-body test also compares the fully serialized normal
and terminal arrays. Strict-trio terminal mode bypasses router preflight; a
dedicated regression starts directly at the limit and proves router call count
remains zero while the main provider receives one terminal `None` call.

Lease behavior is unchanged:

- `DEFAULT_MAX_LEASES_PER_TURN = 3`
- `MAX_LEASE_RENEWAL_REJECTIONS = 2`

## Review and GitNexus

The independent review found one critical issue in the first implementation:
terminal mode re-entered `run_iteration`, so strict-trio preflight could return
or execute before the provider call while consuming the one-shot authority.
The final implementation directly invokes the terminal provider boundary over
fresh messages and the cached frozen catalog. The follow-up review confirmed
the critical issue closed; its two stale-comment findings were also removed.

Pre-edit impact results included:

- `run_agent_loop`: HIGH, 17 upstream / 1 direct / 4 processes
- `normalize_call_key`: HIGH, 38 upstream / 4 direct / 2 processes; intentionally unchanged
- `evaluate_repeated_tool_round`: MEDIUM, 11 upstream
- `step_call_llm`: LOW, 3 upstream
- `should_arm_boundary`: LOW, 4 upstream
- `step_pre_call`: LOW, 3 upstream
- tool-engine and provider helpers: LOW or UNKNOWN where newly introduced

The GitNexus index was three commits behind. Two refresh attempts generated an
index but could not finalize the worktree registry, so all impacts and the
pre-commit scope check retain that staleness caveat. The mandatory final
`detect-changes --repo nanobot-rs` reported the expected CRITICAL hot-path
scope: 6 source files, 49 indexed symbols, and 133 affected flows. No unrelated
source path was changed.

## Files

- `src/agent/agent_loop/heuristics.rs`
- `src/agent/agent_loop/shared.rs`
- `src/agent/agent_loop/tests.rs`
- `src/agent/prepare_context.rs`
- `src/agent/tool_engine.rs`
- `src/providers/openai_compat.rs`
- `.superpowers/sdd/2026-09-01-session-replay-reliability/task-4-report.md`

`src/agent/router.rs` required no production change: terminal mode now bypasses
the router entirely.

## Residual risks

- The blocking provider trait has no mid-request cancellation primitive.
  Cancellation is checked before terminal authority is consumed, but a cancel
  arriving after the terminal HTTP call starts cannot abort that one attempt.
- A persistence failure after the assistant carrier is durable can leave that
  carrier without every rejected receipt. The turn fails closed as `error`, and
  no terminal tool or later provider call executes.
- Terminal failure deliberately has no retry, recovery, validation, or
  continuation path. Availability is traded for the required hard convergence
  bound.
- `scripts/turn_bench.sh` requires a live local provider and was not run in the
  offline environment. Stable serialized tool arrays and wire-prefix
  regressions directly cover the changed cache-sensitive behavior.
