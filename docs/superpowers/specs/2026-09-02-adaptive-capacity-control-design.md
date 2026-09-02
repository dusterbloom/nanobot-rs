# Adaptive Model Capacity and Thrash Prevention

Date: 2026-09-02
Status: Approved design; awaiting written-spec review
Scope: Higgs local inference capacity, Nanobot context admission, and automatic compaction

## Problem

Choosing a local model currently leaves several independent numeric limits for
the user to reconcile manually. Higgs owns model weights, MLX allocations,
retained-session KV, and prefix caches. Nanobot separately owns semantic context,
output reserve, and compaction. Neither side publishes or consumes one measured
hardware-safe capacity.

The failure is especially costly on Apple Silicon because CPU and GPU use unified
memory. A prompt may fit the model's architectural context and every configured
token limit while still pushing the machine into compression and swap. Once that
happens, prefill can take many minutes even though inference remains technically
alive.

The current code has useful but disconnected pieces:

- Nanobot has a VRAM/context solver for local GGUF and Trio operation, but its
  macOS input is total physical RAM rather than current pressure or available
  working set. The normal Higgs path ultimately clamps to the server-advertised
  context and `localMaxContextTokens`.
- Higgs already knows model metadata, model weight bytes, MLX active memory,
  Metal's recommended working-set size, and conservative retained/radix-cache
  bytes. Its session and cache limits are static configuration values.
- Higgs metrics expose request latency and cache residency, but not one live
  admission envelope that an agent client can obey.
- Nanobot can compact a conversation, but it does not receive a typed budget
  reduction from Higgs before an unsafe prefill.

The desired default is that selecting a model is sufficient. The system starts
immediately with conservative limits, learns from real work on that exact
machine, and automatically compacts before an expensive request can cause
thrashing. User-supplied numeric values are ceilings, not promises that force an
unsafe allocation.

## Invariants

1. Higgs is the sole authority for measured model-and-hardware capacity.
2. Nanobot is the sole authority for semantic conversation compaction and retry.
3. Prompt tokens, requested output, retained growth, and transient prefill
   memory are admitted as one resource envelope.
4. Missing tuning values select automatic sizing. Explicit numeric values cap
   automatic sizing but never disable safety reductions.
5. Model selection does not block on a synthetic benchmark. A conservative
   capacity is available immediately and real turns refine it in the background.
6. Capacity rises slowly after repeated clean observations and falls immediately
   on pressure, swap growth, or an unexpected allocation peak.
7. No prompt is silently truncated. Capacity-driven prompt reduction uses
   Nanobot's normal durable compaction path.
8. A capacity retry preserves one logical user turn and cannot duplicate tool
   side effects.
9. Learned data is reused only for an exact hardware, build, model, and execution
   fingerprint.
10. Higgs remains a final fail-closed admission guard for clients that do not
    implement the Nanobot contract.

No controller can prevent an unrelated process from consuming memory after a
request begins. The production guarantee is therefore operational: Higgs does
not knowingly admit or continue new expensive allocation while its working-set
or system-pressure evidence says the request is unsafe.

## Considered Approaches

### Higgs-only guard

Higgs could monitor pressure, evict caches, and reject unsafe prompts. This is a
necessary final defense, but it reacts after Nanobot has already assembled and
submitted an oversized semantic context. Higgs cannot decide what conversation
content should be summarized.

### Shared capacity contract

This is the selected approach. Higgs publishes a live capacity profile for the
loaded model. Nanobot adopts the lower of that profile and its configured
ceilings, then compacts before admission. Higgs independently enforces the same
profile at request time.

### External supervisor

A separate daemon could sample both processes and rewrite their configuration.
It adds another lifecycle, duplicates ownership, races live state, and violates
the codebase goal of one production path per concern.

## Architecture

The runtime flow is:

```text
model load
  -> Higgs publishes conservative capacity
  -> Nanobot adopts live context/output ceilings
  -> real turns update the learned envelope
  -> pressure lowers the Higgs capacity generation
  -> Nanobot compacts before the next expensive request
  -> Higgs performs final request admission
```

Higgs computes effective limits in memory. It does not rewrite `config.toml`
during operation. Nanobot likewise updates its live `TokenBudget` without
rewriting `config.json`. Learned measurements live in a separate derived-state
file and are never confused with user intent.

### Capacity profile

Higgs exposes one endpoint for the served model:

```http
GET /v1/capacity?model=escha-35b-a3b
```

The response is an extension API, not part of standard OpenAI compatibility:

```json
{
  "model": "escha-35b-a3b",
  "generation": 7,
  "pressure": "normal",
  "safeTotalTokens": 53248,
  "recommendedOutputTokens": 4096,
  "maxPromptTokens": 49152,
  "retainedSessionTokens": 49152,
  "retainedBytes": 2147483648,
  "prefixCacheBytes": 1073741824,
  "basis": "learned"
}
```

`safeTotalTokens` is the primary authority. The prompt and output fields are a
recommended split for ordinary chat turns. Higgs still validates the actual
requested output reserve for every request. `generation` increases whenever a
field that affects admission changes, allowing Nanobot to ignore identical
polls cheaply.

`basis` is `conservative` until enough real observations exist for the exact
fingerprint, then `learned` after the first prompt-size band satisfies the
three-clean-observation rule below. The capacity endpoint requires the same
authentication policy as chat completion and returns 404 for an unknown or
unloaded model.

### Fingerprint and persistence

The learned profile key includes:

- hardware identifier and physical memory;
- Metal recommended working-set size;
- Higgs build identity and profile schema version;
- model content identity, not only its display name;
- quantization and native/affine execution path;
- KV representation and relevant cache settings;
- draft/prefill model identities when present.

Profiles are written atomically beneath Higgs's state directory. A missing,
corrupt, incomplete, or mismatched profile restores the conservative cold-start
envelope. A clean process restart may reuse a matching profile only after
checking that current startup headroom is at least as large as the persisted
observation baseline.

## Capacity Controller

### Immediate conservative envelope

After model load, Higgs derives an initial envelope without a long calibration:

- actual resident model and sidecar weight bytes;
- architectural context and KV geometry;
- measured MLX allocation after load;
- Metal `recommendedMaxWorkingSetSize`;
- a protected OS/application reserve;
- user ceilings, if present.

On Apple Silicon the starting process envelope is the smaller nonzero value of
the MLX memory limit and Metal recommended working set, minus a protected reserve
equal to the larger of 4 GiB or 20% of that envelope. Current warning/critical
pressure can only reduce this result. The loaded model's measured MLX allocation
is then subtracted before token capacity is calculated. Other platforms supply
the equivalent backend working-set authority and reserve through the same pure
controller input.

Apple documents `recommendedMaxWorkingSetSize` as the approximate allocation
level below which GPU runtime performance should remain unaffected:
<https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize>.
It is the hardware ceiling input, not total physical RAM.

The cold envelope must reserve enough space for the requested completion and a
conservative transient-prefill estimate. If exact KV geometry is unavailable,
Higgs uses an intentionally high bytes-per-token estimate until real
measurements replace it.

### Learning from real turns

Every successful real request contributes a content-free observation:

- prompt, suffix, and requested-output token counts;
- peak MLX allocation during prefill and generation;
- retained and radix-cache byte deltas;
- time to first token, prefill duration/rate, and decode rate;
- pressure state plus compressor and swap deltas over the request.

The controller maintains a conservative upper envelope rather than trusting an
average. It may raise capacity only after three clean cold-prefill observations
in the relevant power-of-two prompt-size band and five continuous minutes with
normal pressure and no new swap-outs. One increase is the smaller of 4096 tokens
or 12.5% of the current total-token envelope, rounded down to 1024 tokens. It
never raises beyond the next unobserved size band, and it does not learn upward
from a cache hit that avoided the allocation being estimated.

Slow prefill alone is not classified as thrashing. A reduction requires memory
evidence such as system pressure, swap growth, compressor growth, working-set
headroom loss, or an unexpected allocation peak. This avoids shrinking context
merely because a very long exact prefill is naturally slow.

No background synthetic prompts are required for production learning. A
separate explicit benchmark may validate a release, but it does not compete
with foreground work or gate model availability.

### Live pressure and hysteresis

On macOS, Higgs consumes the operating system's normal, warning, and critical
memory-pressure events. Apple provides these through
`DISPATCH_SOURCE_TYPE_MEMORYPRESSURE`:
<https://developer.apple.com/documentation/dispatch/dispatch_source_type_memorypressure>.
It also samples VM compressor and swap counters as deltas; a historical nonzero
swap total is not itself a failure.

The controller has three observable states:

- `normal`: admit within the envelope and cautiously collect upward evidence.
- `constrained`: stop upward learning, evict unleased optional radix/prefix
  entries, lower the next-turn envelope, and request Nanobot compaction.
- `critical`: clear optional caches, reject new expensive admission, and retain
  only the minimum state needed for durable recovery.

Downshifts are immediate. Recovery uses the five-minute/three-observation rule
above, preventing limits from oscillating as memory pressure moves around a
threshold.

### Admission and reservations

Higgs evaluates:

```text
prompt
+ requested output reserve
+ predicted retained-session growth
+ predicted transient prefill peak
<= current safe working envelope
```

Admission reserves the predicted peak before model execution. Cache eviction is
attempted before rejection, but active leased session state is not silently
discarded. Bounded prefill chunks provide opportunities to recheck pressure
between allocations. If an unrelated process creates critical pressure after
admission, Higgs reclaims optional state and terminates at the next safe
boundary instead of continuing unchecked allocation.

Reservations are global to the loaded model, not per HTTP connection. Concurrent
Nanobot agents therefore cannot each admit against the same free bytes. Higgs
atomically subtracts an in-flight reservation before execution and releases it
on success, error, or cancellation. A request that is individually safe but
temporarily blocked by another reservation waits in the existing cancellable
request lifecycle; it is not compacted merely because the model is busy.

The existing retained-session token, retained-byte, prefix-cache-byte, session
count, and suffix-prefill limits become effective outputs of this same
controller. Explicit configured values remain upper bounds. There is no second
independent cache-sizing pipeline.

## Nanobot Integration

Nanobot reads capacity:

- after endpoint/model discovery;
- after a model switch or Higgs restart;
- before every local provider request, including tool-loop continuations and
  compaction requests;
- after any typed capacity rejection.

The endpoint is local and content-free. Nanobot reuses the installed budget when
the returned generation is unchanged.

The effective local token budget is the minimum of the Higgs profile and user
ceilings. Nanobot reserves the planned completion and protocol overhead before
calculating conversation room. It invokes ordinary LCM compaction when the
active context exceeds that room.

Capacity changes do not mutate the persisted user configuration. `/status` and
`/context` display configured ceilings separately from current effective
limits, including the pressure state and capacity basis.

### Compaction and retry

An over-budget request returns HTTP 413 with an OpenAI-shaped typed body:

```json
{
  "error": {
    "type": "higgs_capacity_exceeded",
    "code": "compact_and_retry",
    "safePromptTokens": 36864,
    "safeTotalTokens": 40960,
    "generation": 8
  }
}
```

Nanobot handles only this exact error as automatic capacity recovery:

1. Preserve the pending user turn in SQLite.
2. Compact to the supplied prompt budget using the normal semantic compactor.
3. Rotate/drop the retained Higgs session epoch because compaction changes old
   prompt bytes.
4. Retry once with the same logical task and newly serialized context.
5. Persist the capacity decision and final result.

The retry receives a stable logical-turn identifier so persistence and tool
execution cannot duplicate side effects. A second capacity rejection does not
loop. Nanobot leaves the turn pending and reports the current safe budget.

### Capacity-safe compaction

Compaction must not require sending the same oversized context through the newly
reduced envelope. Nanobot first installs an already completed LCM checkpoint if
one covers the required durable message span. Otherwise it uses ordinary
model-authored LCM summarization only when that summarization request itself
fits the current Higgs budget.

If neither condition holds, Nanobot performs the existing deterministic LCM
level-3 reduction locally, marks it as a capacity-emergency compaction, and keeps
all original messages addressable in SQLite. This fallback makes no provider
call, cannot recursively hit the capacity error, and is visible in the TUI and
audit log. It is not silent prompt truncation: the active prompt receives a
summary marker with durable source IDs so later recall/expansion can recover the
original material.

The minimum working request is Nanobot's immutable system/tool prefix plus a
1024-token completion reserve. If even that request cannot fit, Higgs returns a
distinct `capacity_unavailable` error. Nanobot does not compact repeatedly. It
keeps the turn durable until pressure recovers or the user cancels it.

If critical pressure occurs after streaming begins, Higgs emits a typed terminal
stream error with the generated-token count. Nanobot stores partial output as
incomplete rather than presenting it as a final answer. Recovery resumes from
durable turn state; partial prose is not silently promoted to success.

### Compatibility

A managed Nanobot release should spawn a compatible Higgs binary. When connected
to an older external Higgs that lacks `/v1/capacity`, Nanobot uses a deliberately
conservative 16,384-token total envelope, reserves at most 4096 of those tokens
for output, applies any lower configured ceiling, and clearly reports that
adaptive capacity is unavailable. It must not equate an architectural context
advertised by `/v1/models` with a hardware-safe context.

## Configuration Semantics

Automatic sizing is the production default. Absence means `auto`; a numeric
value is an upper bound. No `unsafe`, `disableSafety`, or alternate controller
mode is added.

Existing numeric configuration remains accepted for compatibility. Loading it
produces a ceiling, while effective runtime values may be smaller. Serialization
must not write learned values back into those fields.

The implementation should converge the following settings under one computed
profile:

- Nanobot context and default output reserve;
- Higgs retained-session token and byte limits;
- Higgs prefix-cache byte limit;
- retained-session count and idle retention where concurrency requires it;
- maximum exact suffix prefill admitted without compaction.

## Observability

Every capacity transition emits one structured record containing:

- model fingerprint and capacity generation;
- old and new token/byte envelopes;
- pressure state and triggering evidence;
- cache bytes reclaimed;
- whether Nanobot compacted, retried, waited, or failed;
- prompt/output reservation used for admission.

Logs and the TUI use direct language such as:

```text
capacity 49K -> 36K · memory pressure warning · compacting and retrying
```

Metrics expose controller state, capacity generation, current effective limits,
downshift count, capacity rejection count, automatic compaction count, retry
outcomes, peak MLX allocation, and swap/compressor deltas. They contain no
prompt content.

## Expected Code Boundaries

Higgs work is expected near its existing model tuning, model state, request
route, metrics, and doctor/config validation paths. The capacity controller must
reuse existing weight metadata, MLX allocation queries, cache byte accounting,
and request metrics rather than build parallel instrumentation.

Nanobot work is expected near `src/higgs.rs`, local provider construction,
`TokenBudget`, context preparation/compaction, typed provider errors, runtime
status, and session persistence. Its existing Trio-only memory solver should be
reused where its pure calculations remain authoritative or retired where Higgs
now supplies better evidence; two competing local-capacity solvers must not
remain in production.

Exact symbol changes belong in the implementation plan after fresh GitNexus
impact analysis in each repository.

## Testing

Implementation is test-first and covers four layers.

### Pure controller tests

- Missing configuration yields a conservative automatic envelope.
- Numeric configuration can lower but never raise the safe result.
- Prompt plus output and transient reserves are accounted together.
- Pressure and swap deltas downshift immediately.
- Recovery requires hysteresis and raises only one bounded step.
- Cache hits cannot train an unsafe cold-prefill estimate.
- Overflow, missing metadata, and corrupt learned profiles fail conservative.
- Fingerprint changes invalidate persisted observations.

### Higgs integration tests

Use an injected pressure-observation interface and an artificially small
working-set budget. Tests do not force the development Mac into real swapping.

- Cache reclamation precedes rejection.
- A request outside the envelope is rejected before model allocation.
- Warning and critical signals update the capacity generation.
- Admission reservations include requested output.
- Concurrent requests cannot reserve the same working-set bytes, and cancelled
  requests release their reservations.
- Mid-prefill critical pressure stops at a bounded safe checkpoint.
- `/v1/capacity`, metrics, and typed errors report the same effective values.

### Nanobot replay tests

- Capacity discovery installs the lower live `TokenBudget` without changing
  `config.json`.
- A proactive over-budget turn compacts before its first provider call.
- An oversized compactor request uses the deterministic capacity-emergency
  reduction without calling Higgs recursively.
- `capacity_exceeded -> compact -> retained epoch rotation -> retry` produces
  one durable user turn and one final assistant result.
- Tool calls completed before recovery are not executed twice.
- A second rejection converges without an infinite retry loop.
- `capacity_unavailable` leaves the work durable and visibly pending.
- A terminal streaming capacity error stores partial output as incomplete.
- An old Higgs endpoint selects the conservative compatibility budget.

### Real hardware validation

Run progressively larger genuine agent sessions against EschaMoE on the target
Mac. Synthetic prompts may support diagnostics but cannot be the only shipping
evidence.

Measure request-scoped deltas for MLX allocation, VM compression, swap-ins,
swap-outs, TTFT, prefill rate, and decode rate. Existing historical swap usage is
ignored; the gate is whether the replay causes new swap activity or sustained
pressure.

## Release Gates

1. Selecting only a model produces a safe usable configuration.
2. Foreground model startup does not wait for calibration.
3. The standard long-session replay creates no new swap-outs.
4. A memory-pressure warning lowers capacity before the next expensive prefill.
5. Critical pressure prevents new unsafe allocation.
6. Nanobot compacts and completes the original logical task automatically.
7. Explicit context/cache settings act as ceilings.
8. A matching learned profile survives restart; any fingerprint change
   invalidates it.
9. Logs and TUI explain every reduction, eviction, compaction, wait, and retry.
10. Ordinary warm turns show no material regression in the matched turn
    benchmark.
11. Higgs release checks and `higgs doctor` validate all changed configuration
    semantics.
12. Nanobot passes release build, regression tests, matched turn benchmark, and
    end-to-end replay against the synchronized Higgs build.

Higgs and Nanobot ship this contract in synchronized releases. Temporary
observation diagnostics are removed or folded into the production metrics path;
no permanent shadow controller or safety-off path remains.

## Non-Goals

- Maximizing context at the cost of system responsiveness.
- Rewriting user configuration with learned numbers.
- Treating total RAM or advertised architectural context as sufficient capacity
  evidence.
- Running mandatory synthetic calibration before the model can answer.
- Silently truncating prompts or generated output.
- Depending on Nanobot cooperation for Higgs process safety.
- Adding a separate supervisor daemon or alternate inference pipeline.
