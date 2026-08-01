# Stable Tool Prefix and Bounded Lease Exhaustion

Date: 2026-08-02
Status: Approved design; awaiting written-spec review
Scope: Nanobot agent loop, tool-result shaping, and Higgs retained-session safety

## Problem

Nanobot treats tool definitions as both provider protocol and flow-control state.
During a live local session, exhausting the tool lease and then producing two
blocked tool rounds makes `step_pre_call` clear `tool_defs`. The local chat
template renders that tool block near the prompt head, so changing six tools to
zero tools invalidates nearly the entire retained prefix. Restoring the tools on
the next user turn invalidates it again.

The 2026-08-01 session `cli:oneshot-1785618334653` demonstrated the cost:

- Higgs retained approximately 24,600 tokens before the lease transition.
- Removing the tool block reduced the common prefix to 244 tokens and forced an
  exact bootstrap of the full prompt.
- Restoring the tool block caused another full bootstrap.
- Long-context recovery generations then took several minutes and were
  repeatedly cancelled.

The investigation also found that the lease contract is implemented
inconsistently. Delegated tool results include `Lease::progress_signal`, but
inline tool results do not. The affected session used inline execution, so the
model received no warning on the final allowed tool result before its next tool
call was rejected.

## Invariants

1. Within one local retained-session epoch, the serialized tool-definition
   array is byte-stable.
2. Behavioral flow control may append messages or reject execution, but it may
   not add, remove, reorder, or rewrite tool definitions.
3. A legitimate tool-topology change starts a new retained-session epoch before
   the changed request reaches Higgs.
4. Lease exhaustion remains structurally bounded even when a model ignores the
   lease protocol.
5. A blocked tool call always receives its matching tool-result receipt before
   the turn terminates, preserving assistant/tool message-array validity and
   replay safety.
6. Lease admission is atomic for one assistant tool-call batch: either every
   call in the batch is admitted or none is executed.

These invariants apply to the production path. They do not introduce a mode
flag or an alternate agent pipeline.

## Considered Approaches

### Remove the schema strip and retain the current no-progress limit

This preserves the prefix, but an uncooperative model may consume four more
provider calls before `NO_PROGRESS_HARD_STOP` terminates the turn. At long
context, each call may take minutes. This is bounded but not operationally
acceptable.

### Keep tools present and force `tool_choice: none`

This could grammar-constrain prose while retaining the tool array. It requires
plumbing tool choice through the streaming provider path, depends on backend
semantics outside Nanobot, and adds another inference call. The live session's
blocking forced-tool recovery already timed out after 120 seconds, so this
approach does not provide a reliable latency bound.

### Stable schema with an explicit final lease signal and immediate violation stop

This is the selected approach. The model receives the lease boundary before it
must decide what to do. It may renew or answer normally. If it ignores that
contract and requests another tool, Nanobot rejects the call and terminates
without another provider invocation.

## Design

### One lease annotation path

Move tool-result lease decoration behind one `Lease` formatter used by both the
inline and delegated execution paths. The annotation describes lease usage
after the admitted batch, so concurrent calls cannot all claim the same
per-call ordinal. A normal result reports the current usage. A result from the
batch that consumes the remaining allowance adds an explicit transition:

```text
[Lease usage after this batch: 12 of 12 calls — 3 renewals remaining. Lease
exhausted: your next response must be either a final answer or a renewal
checkpoint containing findings:/next:/will:. Do not request another tool
before renewal.]
```

The wording distinguishes future renewals from the current lease and does not
claim that tools disappeared. The complete tool schema remains attached to the
following provider request.

### Renewal and final-answer path

The existing text-response path remains authoritative:

- A valid `findings:/next:/will:` checkpoint renews the lease, appends the
  existing renewal confirmation, and continues.
- An incomplete checkpoint receives the existing bounded correction.
- Plain text finishes the turn.
- A checkpoint that attempts renewal after all renewals are consumed finishes
  under the existing out-of-leases behavior.

Renewal changes execution authority only. It does not rebuild or mutate the
tool definitions.

### Ignored-lease path

Lease admission occurs once per assistant tool-call batch, before any member of
the batch executes. `Lease::admit_batch(count)` returns a typed
`BatchAdmission` outcome instead of requiring the caller to loop over a
stateful single-call API or interpret a boolean:

- If the complete batch fits, admission consumes its call count atomically and
  every call executes.
- If the complete batch does not fit, admission consumes nothing and no call in
  that batch executes. This avoids a partially executed assistant message and
  orphaned tool-result IDs at the lease boundary.

If the model emits an over-budget tool-call batch:

1. Do not execute any call in the batch.
2. Append one assistant message containing the complete original tool-call
   array.
3. Append one matching rejection result for every tool-call ID, explaining that
   the lease was exhausted or that the batch exceeded the remaining allowance.
4. Persist the complete assistant/results protocol group through the existing
   persistence path.
5. Finish the turn immediately with a deterministic user-visible explanation.

The terminal explanation must state that Nanobot stopped an over-budget tool
request. It must not expose accompanying model prose as a final answer because
such prose is commonly a preamble such as "let me inspect that".

A response containing both a renewal checkpoint and a tool call is invalid for
this protocol: renewal must occur before another tool request. It follows the
same rejection-and-stop path.

This replaces the lease-specific two-round strip and the four-round wait. The
general `NO_PROGRESS_HARD_STOP` remains for unrelated zero-progress paths such
as response-boundary rejection.

### Stable retained-session epochs

The existing per-session tool hash becomes an enforced retained-session
boundary for Higgs-capable providers:

- The first request records the serialized tool-array hash.
- An equal hash continues the current epoch.
- A different hash is treated as a sanctioned prompt-head rewrite: rotate the
  retained-session epoch and queue the old concrete Higgs session ID for drop
  before attaching the next session marker.
- Non-Higgs providers retain diagnostic behavior without Higgs-specific
  rotation.

The comparison must release the tool-hash lock before calling cache
invalidation because epoch reset clears the same map. After rotation, record
the new hash as the baseline for the new epoch.

This is defense in depth for legitimate topology transitions such as a model or
trio-routing state change. Lease state must never reach this fallback because
it no longer changes the tool array.

## Code Changes

Expected production touch points:

- `src/agent/lease.rs`
  - Update the stale contract that instructs callers to strip tools.
  - Replace stateful per-call admission with atomic batch admission.
  - Provide the shared result annotation, including the final-batch exhaustion
    directive.
- `src/agent/tool_engine.rs`
  - Use the shared annotation in both inline and delegated result injection.
- `src/agent/agent_loop/shared.rs`
  - Remove `LEASE_BLOCKS_BEFORE_STRIP`, `consecutive_lease_blocks`,
    `lease_forced_text_only`, and router-restore exceptions tied to stripping.
  - Terminate immediately after persisting a lease-exhausted tool rejection.
  - Enforce Higgs epoch rotation when the serialized tool hash changes.
- `src/agent/prepare_context.rs`
  - Remove initialization of retired lease-strip state.
- `src/turn_stream.rs` and `src/tui_app/app.rs`
  - Add the explicit `ToolTopology` cache-reset reason and render it as
    `cache reset · tool topology` when a real schema transition rotates the
    Higgs epoch.

No new module, configuration field, protocol mode, or provider API is needed.

## Testing

Implementation follows test-first development.

### Lease unit tests

- Inline and delegated result bodies receive the same annotation format.
- Batches before the limit report normal post-batch usage.
- The batch consuming the remaining allowance reports exhaustion and the exact
  renewal/final-answer choices.
- A batch larger than the remaining allowance consumes no calls.
- Renewal resets lease usage while decrementing remaining renewals.

### Agent-loop convergence tests

- An adversarial provider emits distinct tool calls forever. Assert:
  - every provider call receives a non-empty tool array;
  - every serialized tool array is byte-identical;
  - exactly the configured allowed calls execute;
  - the first post-exhaustion batch is rejected atomically;
  - no further provider call occurs;
  - the turn returns the deterministic over-budget explanation.
- A multi-call response crossing the lease boundary executes none of its calls,
  persists one assistant tool-call array plus one result per ID, and terminates
  without a subsequent provider request.
- A provider emits a valid renewal checkpoint after the final allowed result.
  Assert the next tool executes and the tool array remains byte-identical.
- Existing legitimate bounded exploration reaches its model-authored final
  answer without a guard firing.
- Existing duplicate-call and response-boundary convergence tests continue to
  pass, proving the universal no-progress guard still covers independent
  failure classes.

### Retained-session tests

- Equal tool hashes preserve the same Higgs epoch and queue no drop.
- A changed tool hash rotates the epoch before the request marker is derived,
  queues the old concrete session ID, and records the new hash.
- The rotation path does not deadlock while clearing tool-hash bookkeeping.
- A lease-exhaustion sequence never rotates because its tool hash never
  changes.

### Verification

Run targeted lease, tool-engine, agent-loop convergence, prompt-cache, and
Higgs-session tests, followed by:

```bash
cargo test
cargo build
scripts/turn_bench.sh
```

The matched turn benchmark is required because this changes the agent loop and
provider-request cache boundary.

## Rollout Evidence

The change is successful when a reproduced exhausted-lease trace shows:

- no `tool_lease_stripping_after_blocks` event;
- no lease-driven `tool_block_changed` event;
- stable Higgs `common_prefix_tokens` across the exhaustion boundary;
- no provider request after the first rejected over-budget call;
- protocol-valid persisted assistant/tool receipt pairing;
- normal renewal and final-answer behavior.

## Explicit Non-Goals

- Changing lease size or renewal count.
- Removing tool leases.
- Reworking duplicate-call caching or tool-result stashing.
- Changing Higgs decoding, TurboQuant thresholds, or MLX profiles.
- Fixing the separate blocking forced-tool-recovery timeout. That path has a
  distinct cause and requires its own design and regression tests.
