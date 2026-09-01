# Session Replay Reliability and Bounded Turn Convergence

Date: 2026-09-01
Status: Approved design; awaiting written-spec review
Scope: Agent-loop convergence, tool outcome truthfulness, SQLite replay, terminal outcomes, and release verification

## Problem

The latest persisted session, `20260901_161823_fbb4e8`, demonstrates a compound
failure rather than one isolated bug. The agent eventually gathered the requested
evidence, but it never returned it to the user. During the turn it made 99 model
requests and executed 82 tools, exhausted GitHub's rate limit, persisted several
failed shell/API operations as successful, injected 30 response-boundary
scaffolds, and finally recorded a provider failure as a finished turn.

The session database is structurally healthy and its assistant/tool-call pairing
is complete. The failure is in the production control path and the truthfulness
of the state written to that database:

- `curl` is classified as a freely renewable read-only exec command, so varied
  network calls can bypass the intended per-turn lease.
- shell pipelines inherit the last program's exit code, allowing an upstream
  `jq` failure to be reported as success;
- structured API error bodies such as GitHub's rate-limit response are also
  accepted as success;
- the response-boundary prompt usually produces more tool use, not a report;
- protocol-persistence failures are logged but do not consistently stop the
  provider call or tool side effect they were meant to precede;
- provider errors lose their terminal type when final text is rendered;
- one failed health probe is presented as proof that the local server crashed;
- mechanical LCM headlines can discard a single-line `TOOL_RESULT_HANDLE`;
- `tool_results.ok` is not durably populated;
- the committed stash lookup discards an explicit range when a query misses.

The current working tree already contains two user-owned changes. The stash
query-to-range fallback is part of this design's verification target. The
opt-in provider prompt dump is not part of the release path and must be removed
before shipping; exact replay artifacts already provide the required diagnostic
record without a second prompt-capture mechanism.

## Production Evidence About Synthetic Prompts

An independent read-only audit classified the immediate model response after
every exact response-boundary scaffold in the live SQLite history:

| Outcome | Count | Share |
|---|---:|---:|
| Tool-free response | 98 | 18.6% |
| Text plus another tool | 228 | 43.3% |
| Silent tool call | 173 | 32.8% |
| Interrupted/no response | 28 | 5.3% |

The prompt occurred 527 times across 130 sessions. Of 100 calls actually
rejected by the boundary, only six produced a tool-free response on the next
round. In the latest session, 30 scaffolds yielded one final answer, no
substantive tool-free progress report, 17 text-plus-tool responses, ten silent
tool calls, and two interruptions.

The longer current wording performs worse observationally than the former short
wording, although model and version changes prevent a causal claim about the
wording itself. The supported conclusion is narrower: production evidence does
not justify shipping the current boundary scaffold or trying another unmeasured
rewording.

Other scaffolds are judged by their own purpose rather than by final-answer
rate. Lease renewal restored intended tool use in 11 of 17 observed cases and
remains bounded by the existing two-renewal cap. Repeat-result nudges produced
no observed convergence and are replaced by deterministic limits.

## Invariants

1. The production path remains channel → agent loop → provider → tools → reply.
2. No provider call or tool side effect occurs unless its required protocol
   prefix is durable in SQLite.
3. One raw tool execution has one truthful outcome used by the model message,
   tool event, replay event, and `tool_results.ok` value.
4. A failed tool never resets no-progress state as though useful evidence was
   gathered.
5. Turn completion preserves a typed outcome. Error, cancellation, empty output,
   and limit exhaustion are never rewritten to `finished` merely because a
   user-visible explanation exists.
6. Tool definitions remain byte-stable during a retained local session. A
   terminal prose request keeps the definitions and uses `tool_choice: none`.
7. All convergence limits have explicit units and per-turn reset semantics.
8. A rejected tool call still receives its matching protocol-valid result
   receipt before the turn stops.
9. SQLite replay remains readable by the previously installed binary. Any
   schema evolution is additive and ignored safely by that binary; the release
   does not require a destructive database downgrade on rollback.
10. Tests may provide fixtures and scripted providers, but no alternate replay
    pipeline or protocol mode enters production.

## Considered Convergence Approaches

### Reword the response-boundary prompt

Rejected. The audit shows at most a small, confounded shift from silent tool
calls to narrated tool calls. It does not produce reliable reports, and its
persisted bytes compound across later requests.

### Strip tool definitions for the reporting call

Rejected. The local chat template renders the tool block near the prompt head.
Removing it invalidates the retained prefix and can force a full long-context
prefill.

### Remove all convergence enforcement

Rejected. Fixing `curl`, error classification, and leases removes the largest
amplifiers, but an uncooperative model still needs a deterministic terminal
bound.

### Typed limits with a terminal `tool_choice: none` call

Selected. Ordinary work keeps the stable tool array and existing lease. At a
typed terminal limit, Nanobot records one final model request with the same
messages and tool definitions but explicitly sets `tool_choice: none`. This
changes request policy, not the serialized tool prefix. The call is bounded and
does not depend on a synthetic user message.

The earlier stable-tool-prefix design rejected `tool_choice: none` as the
ordinary lease-exhaustion path because it adds an inference and cannot guarantee
low latency. This design uses it only once as terminal recovery after an already
reached hard limit. The terminal call may use the existing blocking tool-choice
provider path; its normal provider timeout is the latency bound, and failure is
persisted as `limit_exhausted` rather than retried.

If a provider ignores `tool_choice: none` and returns another tool call, Nanobot
does not execute it. It persists the matching rejection receipt and terminates
with `limit_exhausted`. The fallback is truthful and bounded; it is not treated
as a model-authored final answer.

## Design

### Truthful tool outcomes

The existing tool-execution chokepoint classifies the raw result exactly once
before any downstream representation is written. Its result drives all of:

- the provider-facing tool receipt;
- `ToolEvent::CallEnd.ok`;
- the exact replay event;
- `tool_results.ok`;
- lease/no-progress accounting.

Shell execution must surface failure from any pipeline stage instead of trusting
only the final command. Structured stdout/stderr is inspected after process
status classification so a zero exit status cannot bless a known API error.
The API classifier covers the established error envelopes plus HTTP-style
`message` bodies for rate limits and authentication failures. It must remain
conservative: an arbitrary successful JSON object containing ordinary prose is
not an error.

`curl` remains available through `exec` under normal safety policy, but it is no
longer a read-only auto-renewal command. Each invocation consumes the ordinary
lease. This bounds varied URLs and arguments that exact-call deduplication cannot
recognize as one logical retry storm.

### Persistence fails closed before external work

`persist_pending_protocol_messages` and the tool-call carrier journal return a
typed result. Their callers stop the turn on failure instead of logging and
continuing.

The required ordering is:

1. persist inbound user/protocol messages;
2. persist the exact model-request artifact;
3. call the provider;
4. persist the assistant tool-call carrier and pre-execution decision;
5. execute the tool;
6. persist the raw result and its model-facing projection;
7. continue the loop or finalize.

A failure before steps 3 or 5 prevents that external operation. A failure after
a provider response or tool execution cannot undo completed work; it becomes an
infrastructure error and stops further work. Failure of the final
`turn_finished` journal after an already persisted reply may still deliver that
reply while replay is marked incomplete, because suppressing the generated reply
would not restore durability.

Tests inject locked-database, forced write-failure, and disk-full-equivalent
errors at the user-message, model-request, tool-carrier, pre-execute, raw-result,
and finalization boundaries. The critical assertion is the count of provider and
tool invocations after each injected failure.

### Typed turn outcomes

The hot path carries an internal enum through finalization rather than flattening
errors into `final_content`. Its minimal wire outcomes are:

- `finished` — a normal model-authored final response;
- `error` — provider or infrastructure failure;
- `cancelled` — user or job cancellation;
- `empty` — provider returned no usable content or tool payload;
- `limit_exhausted` — the deterministic convergence bound was reached without a
  model-authored final response.

The user-visible error text remains separate from the outcome. A local health
probe may add evidence such as "the backend health endpoint was unavailable",
but it does not replace the provider's precise error with "server crashed".

These outcomes use the existing string field in replay events and require no
outcome-schema change. Compatibility verification loads newly written events
through the prior installed binary before deployment.

### Deterministic convergence

The response-boundary enum, arming logic, synthetic user message, and execution
rejection path are removed together. Repeat-result scaffolds are also removed.

The existing `FlowControl` hot path remains the single authority for:

- model requests consumed per turn;
- successfully executed tools per turn;
- consecutive zero-progress rounds;
- consecutive repeated normalized call batches;
- lease use and at most two renewals.

No new guard module or protocol mode is introduced. Each counter is initialized
with the turn and reset only by its documented evidence event. A failed or
rejected tool is zero progress.

When a hard convergence limit is reached, the loop makes at most one terminal
provider request using the unchanged tool array and `tool_choice: none`. It
reuses the existing `Continuation` replay purpose and is distinguished by the
recorded `tool_choice: "none"`; adding a new serialized enum variant would make
the rollback binary reject the event. Providers that cannot enforce tool choice
are still safe because returned tool calls are rejected rather than executed.

The physical `tool_results` table regains its nullable `ok` column through the
prior additive migration pattern. Fresh databases create it directly; existing
databases use `ALTER TABLE ... ADD COLUMN` only when absent. The status-aware
immutable store treats status as part of the stored result. Older binaries ignore
the extra column, so this is compatible in both directions and requires no
destructive migration or historical backfill.

### Replay and compaction integrity

Mechanical LCM headlines preserve the bounded excerpt from a one-line
`TOOL_RESULT_HANDLE` instead of skipping the complete line. The regression test
uses the actual handle shape produced by the tool engine.

The dirty stash-search change remains narrow: when a literal query misses and a
valid range was supplied, inspection falls back to that range. Query hits remain
unchanged. This is verified against the call shape persisted in the failed
session.

The prompt-dump diagnostic is removed before release. Exact model request and
response artifacts in SQLite are the one diagnostic and replay source.

## End-to-End Replay

The deterministic replay is an integration test around the production
`AgentLoop`, not a second runtime. It supplies:

- a temporary SQLite database and workspace;
- the real context, routing, tool-engine, persistence, and finalization code;
- a scripted provider returning recorded failure shapes;
- loopback HTTP fixtures for successful JSON, GitHub 429/secondary-limit,
  authentication failure, malformed JSON, and empty SSE;
- a safe temporary command workspace for pipeline success and upstream failure.

The fixture sequence covers the compound failure, not merely isolated helpers:

1. successful evidence-producing tools;
2. a pipeline whose upstream stage fails;
3. a zero-exit structured rate-limit body;
4. repeated distinct network-tool attempts;
5. terminal convergence;
6. an empty provider stream;
7. finalization and exact replay reload.

Assertions cover call counts, tool receipts, `ok` values, replay ordering,
terminal outcome, absence of response-boundary scaffold rows, and compatibility
with session reload. Separate focused tests cover each persistence fault and LCM
handle compaction.

Scripted responses prove control flow but cannot prove that the deployed local
model produces a useful answer. A second semantic gate replays the latest task
in a new session against the verified release binary and local model, using
bounded local fixtures rather than the public GitHub API. The task-specific
oracle checks that the answer contains the complete recorded 25-item result,
contains no unsupported claims, and is delivered as a tool-free final response.

An independent reviewer receives the baseline and candidate persisted
transcripts in blinded order. Its semantic classification is secondary to the
hard oracle and structural database assertions; it cannot override a failed
mechanical gate.

## Verification Gates

### Gate A: Baseline

- Take a SQLite online backup of the live database, including any WAL state.
- Record the installed binary hash and the freshly built candidate hash.
- Capture matched turn-benchmark and prompt-cache metrics before production
  changes.
- Preserve the failed session's exact replay artifacts as the immutable oracle.

### Gate B: Truthfulness and persistence

- Pipeline and structured API failures have `ok=false` in every representation.
- GitHub primary and secondary rate-limit fixtures are typed failures.
- An injected precondition persistence failure produces zero later provider or
  tool calls.
- Provider, infrastructure, empty-stream, and cancellation outcomes do not
  appear as `finished`.

### Gate C: Bounded convergence

- No response-boundary or repeat-result scaffold is persisted.
- `curl` consumes the normal lease and cannot auto-renew it.
- Lease renewal remains capped at two.
- Limit exhaustion performs at most one `tool_choice: none` call.
- A provider that ignores `none` causes no tool execution and ends as
  `limit_exhausted`.

### Gate D: Full replay and performance

- The deterministic compound replay passes through the production agent loop.
- The fresh local-model replay returns the complete correct final answer.
- `cargo build --release` passes.
- `cargo test --release` passes.
- A matched 20-turn `scripts/turn_bench.sh` comparison has zero additional
  failures. Median wall time and median TTFT may not regress by more than 10%; a
  larger first result is rerun twice under the same machine/model/power state and
  is a no-go if the median regression persists.
- Tool-definition hashes remain byte-identical. Cache-read efficiency may not
  fall by more than five percentage points in the matched run.
- `git diff --check` and GitNexus `detect_changes` show only expected scope.

### Gate E: Deployment

- The previous binary reads sessions containing every new outcome string.
- The release binary hash matches the verified candidate before and after
  installation.
- The live SQLite backup completes before process replacement.
- Restarted health checks and a smoke turn pass.
- Post-deploy session rows have truthful terminal and tool outcomes.

Any masked success, unbounded call sequence, missing tool receipt, response-
boundary scaffold, incompatible replay row, cache collapse, or incorrect live
answer is a no-go.

## Deployment and Rollback

Deployment backs up SQLite through its online backup mechanism and copies the
currently installed binary to a hash-addressed rollback artifact. The verified
candidate is installed through a temporary sibling followed by atomic rename.
The agent is restarted, its health is checked, and one bounded smoke turn is
queried from SQLite before declaring the release live.

A normal release failure rolls back only the binary and restarts it. The only
schema change is an additive column the prior binary ignores, so restoring the
database would discard new user messages without benefit. The database backup is
restored only for demonstrated corruption or a compatibility failure, and only
after the affected processes are stopped.

## Code Touch Points

Expected production files, subject to impact analysis before each symbol edit:

- `src/agent/tools/shell.rs` — truthful pipeline status;
- `src/agent/tool_engine.rs` — API-body classification, `curl` lease class,
  unified outcome projection, and boundary removal;
- `src/agent/agent_loop/shared.rs` — persistence propagation, typed limits,
  terminal `tool_choice: none`, and scaffold removal;
- `src/agent/finalize_response.rs` — typed terminal outcome persistence;
- `src/agent/lcm.rs` — one-line handle headline preservation;
- `src/agent/tools/stash_search.rs` — retain and verify query-range fallback;
- `src/providers/openai_compat.rs` and provider trait code only as required to
  carry a terminal `ToolChoice::None`; remove the dirty prompt dump;
- `src/session/db.rs` — restore the rollback-safe additive `tool_results.ok`
  migration, populate truthful values, and add fault-injection/replay coverage;
- existing agent-loop, provider, session, and LCM test modules for regressions.

No production replay module, new feature flag, parallel agent loop, or
incompatible/destructive SQL migration is planned.

## Explicit Non-Goals

- Rewriting the agent loop or provider abstraction.
- Preventing all network use through `exec`.
- Inferring arbitrary semantic success or failure from free-form command output.
- Backfilling every historical nullable `tool_results.ok` row.
- Treating an LLM reviewer as the release oracle.
- Restoring the live database during routine binary rollback.
