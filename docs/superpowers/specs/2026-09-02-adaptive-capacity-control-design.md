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
  "schemaVersion": 1,
  "model": "escha-35b-a3b",
  "modelFingerprint": "sha256:7b2f5c8ae91a5b1d83f1364c2023e5e53b5530d0461a4193cf9bd37f4e70d821",
  "bootId": "01993654-8af2-7b31-a420-c52ebc349287",
  "generation": 7,
  "availability": "available",
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
polls cheaply. Higgs creates a random `bootId` for every process. A generation
is comparable only within one boot ID, so a restart cannot make Nanobot reuse a
capacity cached from an older process whose retained state no longer exists.

`basis` is `conservative` until enough real observations exist for the exact
fingerprint, then `learned` after the first prompt-size band satisfies the
three-clean-observation rule below. The capacity endpoint requires the same
authentication policy as chat completion and returns 404 for an unknown or
unloaded model. `schemaVersion` is `1`; `availability` is `available` or
`unavailable`; `pressure` is `normal`, `constrained`, or `critical`; and `basis`
is `conservative` or `learned`. `retainedSessionTokens` is the effective
per-session cap, while `retainedBytes` and `prefixCacheBytes` are process-wide
effective byte caps.

### Fingerprint and persistence

The learned profile key includes:

- hardware identifier and physical memory;
- operating-system version and build;
- Metal recommended working-set size;
- Higgs build identity and profile schema version;
- model content identity, not only its display name;
- quantization and native/affine execution path;
- KV representation and relevant cache settings;
- draft/prefill model identities when present.

Profiles are written atomically beneath Higgs's state directory. They persist
model cost observations, not a promise of currently available capacity. Higgs
recomputes the live envelope from current pressure and working-set headroom on
every start. A missing, corrupt, incomplete, or mismatched profile restores the
conservative cold-start cost model. A clean process restart may reuse a matching
profile only after checking that current startup headroom is at least as large
as the persisted observation baseline.

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

The cold envelope reserves space for the requested completion and a conservative
transient-prefill estimate. If exact KV geometry is unavailable, the loader uses
the representation's documented upper bound; if that is also unavailable, it
publishes only the minimum working request until real measurements replace it.

Before loading weights, Higgs compares an artifact-specific resident estimate
plus a loader-path workspace bound with the safe process envelope. Each known
loader reports its largest simultaneous shard/conversion allocation; an unknown
loader uses twice the artifact weight bytes as its workspace bound. A model that
is definitely too large fails before allocation. Loading proceeds at bounded shard
boundaries with pressure checks; warning stops optional prefetch and critical
pressure aborts the load. After load, Higgs replaces the estimate with measured
MLX residency. If measured residency plus Nanobot's immutable system/tool prefix
and a 1024-token completion reserve cannot fit, loading fails with typed
`insufficient_capacity` and releases the model. Higgs never publishes a positive
`safeTotalTokens` below that minimum working request.

### Worked EschaMoE cold-start tiers

The selected Qwen3.6 35B-A3B Escha artifact contains 12,296,952,480 bytes of
safetensors and Higgs documents roughly 11 GiB native residency. It is a hybrid
model: ten of forty layers use full attention, with two KV heads of width 256.
Its dense full-attention fp16 KV term is therefore:

```text
10 layers * 2 (K,V) * 2 KV heads * 256 width * 2 bytes
= 20,480 bytes/token
```

Linear-attention recurrent state is charged as a fixed per-session term by the
engine cost model rather than hidden inside this token slope. At 49,152 tokens,
the dense KV term is 960 MiB.

For an illustrative 32 GiB tier whose Metal API reports a 24 GiB recommended
working set, the starting process envelope is `24 * 0.8 = 19.2 GiB`. After an
11 GiB loaded-model baseline and a conservative 4 GiB prefill transient bound,
4.2 GiB remains for live/output KV, recurrent state, retained sessions, and
prefix cache. The controller can admit a 49,152-token dense KV term only if its
measured fixed state and automatically reduced cache budgets fit the remaining
approximately 3.26 GiB; otherwise it publishes a smaller token envelope.

For an illustrative 64 GiB tier whose Metal API reports 48 GiB, the process
envelope is `48 * 0.8 = 38.4 GiB`, leaving 23.4 GiB after the same model and
transient terms. These are examples of the calculation, not hardcoded hardware
tables. Release Gate 1 records the actual Metal limit, measured residency,
fixed-state cost, transient bound, cache allocation, and resulting token limit
for each tested Mac tier.

### One byte-domain cost model

Admission comparisons use bytes only. Every Higgs engine path supplies one
memory cost model from its real cache and execution geometry:

- loaded model/sidecar baseline bytes;
- fixed bytes per live session, including recurrent state;
- persistent bytes per prompt and output token for the selected KV
  representation;
- retained/radix duplication or sharing from existing cache `estimated_bytes`
  accounting;
- decode/sampling workspace;
- worst-case transient bytes for `(full_prompt_tokens, prefill_chunk_tokens)`.

For conventional attention, the initial persistent slope is the sum across
cache-bearing layers of K and V elements multiplied by actual storage width.
Hybrid, MLA, TurboQuant, and paired-drafter paths report their own fixed and
token-linear components rather than being forced through a dense-transformer
formula. Runtime high-water observations raise any underestimated coefficient
immediately; lowering a coefficient requires the normal clean-evidence window.

The controller selects a bounded prefill chunk whose predicted transient peak
fits the remaining byte envelope. The initial transient function comes from
model/engine geometry and is replaced conservatively by the maximum observed
high-water value in each prompt/chunk band. If an engine path cannot provide a
safe initial bound, it exposes only the minimum working request until real
allocation-bearing requests establish one.

Published token fields are derived from this byte solver by finding the largest
1024-token-aligned value that satisfies the byte inequality after output and
fixed-state reserves. Request admission calls the same solver; there is no
separate token heuristic. All byte arithmetic is checked `u64` arithmetic and
overflow produces `capacity_unavailable`.

### Learning from real turns

Every successful real request contributes a content-free observation tagged by
its `cold`, `retained_suffix`, or `radix_hit` execution path:

- prompt, suffix, and requested-output token counts;
- peak MLX allocation during prefill and generation;
- retained and radix-cache byte deltas;
- time to first token, prefill duration/rate, and decode rate;
- pressure state plus compressor and swap deltas over the request.

The controller maintains a conservative upper envelope rather than trusting an
average. Cold observations are banded by full logical prompt size and train
transient-prefill cost. Retained/radix hits are also banded by full prompt size;
they train actual persistent/retained residency and suffix cost but cannot lower
the cold-prefill coefficient. This prevents a cheap suffix turn from teaching
Higgs that a later cache-reset bootstrap will also be cheap.

Capacity may rise only after three clean allocation-bearing observations in the
current power-of-two prompt-size band, spanning five continuous minutes with
normal pressure and no new swap-outs. Idle time before the first or after the
last observation does not count. One increase is the smaller of 4096 tokens or
12.5% of the current total-token envelope, rounded down to 1024 tokens. Clean
cold observations at the current boundary may open one step into the next band
using the conservative static slope; no observation permits a jump over an
entire unobserved band.

A single prefix-heavy session cannot raise its cold-bootstrap limit from cache
hits. The learning-liveness replay therefore uses genuine agent sessions with
repeated warm tool turns and at least three naturally cold session starts in the
boundary band. It must demonstrate upward movement without synthetic prompts
while preserving the colder transient-prefill coefficient.

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
swap total is not itself a failure. System-wide pressure from another process
reduces the current live envelope and freezes upward learning, but does not
persistently rewrite learned model-cost coefficients.

The controller has three observable states:

- `normal`: admit within the envelope and cautiously collect upward evidence.
- `constrained`: stop upward learning, evict unleased optional radix/prefix
  entries, increase the protected reserve from 20% to 30%, and recompute the
  next-turn envelope.
- `critical`: clear optional caches, reject new expensive admission, and retain
  only the minimum state needed for durable recovery.

Downshifts are immediate. Recovery uses the five-minute/three-observation rule
above, preventing limits from oscillating as memory pressure moves around a
threshold. After eviction and byte-ledger recomputation, warning commits the
smaller of the recomputed total or 75% of the previous token envelope. Critical
commits the smaller of the recomputed total or 50% of the previous envelope,
then returns `capacity_unavailable` while critical pressure remains. Values are
rounded down to 1024 tokens. Any new swap-out is critical for new admission
until pressure is normal and swap-out counters remain unchanged for one minute.
An unexpected allocator peak raises the matching cost coefficient to the
observed high-water value plus 10% and recomputes capacity immediately.

### Admission and reservations

Higgs applies the same admission path to local `/v1/chat/completions`,
`/v1/completions`, and `/v1/messages` requests. It converts request requirements
to bytes and evaluates:

```text
committed bytes
+ request reservation bytes
<= usable process envelope bytes
```

`usable` is the smaller nonzero MLX/Metal working limit minus the protected OS
reserve. `committed` is non-evictable loaded-model residency, retained/leased KV,
radix residency after permitted eviction, all active reservation bytes, and the
positive difference between measured MLX active memory and all accounted
residency. A request reservation is the larger of the static
cost model and the learned high-water estimate for its execution path. It
includes KV for prompt plus requested output, uncached-suffix prefill workspace,
decode/sampling workspace, and only actual post-turn retained duplication not
already charged in the KV term.

Admission reserves the predicted peak before model execution. Cache eviction is
attempted before rejection, but active leased session state is not silently
discarded. Bounded prefill chunks provide opportunities to recheck pressure
between allocations. If an unrelated process creates critical pressure after
admission, Higgs reclaims optional state and terminates at the next safe
boundary instead of continuing unchecked allocation.

Reservations are process-wide, not per model or HTTP connection, because Higgs
may keep multiple engines resident while MLX and Metal memory are global.
Concurrent Nanobot agents therefore cannot each admit against the same bytes.
"Individually safe" means the request fits after persistent process/model/cache
bytes but before other in-flight reservations. If it is individually safe yet
does not fit after outstanding reservations, it waits in a FIFO cancellable
admission queue instead of receiving a compaction error. A waiter rechecks
pressure, boot ID, generation, persistent bytes, and its prediction at dequeue.

An in-process RAII guard owned by the actual inference worker holds every byte
reservation and releases it only when that worker returns on success, engine
error, unwind, or acknowledged cancellation. There is no TTL that can free
bytes while a kernel still uses them. Client disconnect, server timeout, or a
no-progress watchdog using the configured request timeout signals cancellation;
simple and batch engines check it
before each bounded prefill chunk and decode step. The worker drops its guard
only after allocation has stopped. Model unload/switch waits for guards to drain
or cancels and joins their workers before releasing weights.

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

The endpoint is local and content-free. Nanobot keys its cache by endpoint,
schema version, boot ID, and model fingerprint, then reuses the installed budget
only when that tuple and generation are unchanged. A boot-ID change invalidates
the old capacity snapshot and rotates Nanobot's retained-session epoch because
the restarted Higgs process cannot own the previous retained KV.

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
    "bootId": "01993654-8af2-7b31-a420-c52ebc349287",
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
loop. Nanobot leaves the turn pending and reports the current safe budget. The
typed provider error retains `safePromptTokens`, `safeTotalTokens`, `bootId`,
and `generation`; recovery never parses numbers from an error message.

The 413 values are computed after permitted eviction. For the same boot ID and
generation, a rewritten request at or below both published token limits is
individually admissible by construction. If another request temporarily owns
the remaining bytes, it queues rather than receiving another 413. Only a newer
capacity generation or worsened pressure may invalidate the figures; that case
may produce the one terminal second rejection described above.

Capacity recovery retries only the current provider request and is allowed only
before response processing reaches tool execution. It never restarts the outer
user-turn loop. No `ToolPreExecute` or `ToolExecute` journal entry may precede a
capacity 413; completed tool calls from earlier iterations remain committed and
are not replayed.

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
1024-token completion reserve. If even that request cannot fit, the capacity
endpoint reports `availability: "unavailable"` with zero prompt/output fields
and Higgs returns HTTP 503 with this body before model allocation:

```json
{
  "error": {
    "type": "higgs_capacity_unavailable",
    "code": "capacity_unavailable",
    "bootId": "01993654-8af2-7b31-a420-c52ebc349287",
    "generation": 9,
    "retryAfterMs": 5000
  }
}
```

Nanobot does not compact repeatedly. It keeps the turn durable and polls after
5 seconds, backing off to at most 30 seconds. Once the same endpoint reports an
available profile that fits the minimum request, Nanobot resumes the pending
turn automatically unless the user cancelled it.

If critical pressure occurs after streaming begins, Higgs emits a typed terminal
SSE event followed by the normal `[DONE]` terminator:

```text
data: {"error":{"type":"higgs_capacity_interrupted","code":"capacity_interrupted","bootId":"01993654-8af2-7b31-a420-c52ebc349287","generation":10,"partialOutputTokens":317}}

data: [DONE]
```

Nanobot records the partial bytes in its model-failure journal as an incomplete
artifact, retracts any transient TUI rendering, and does not commit them as a
successful assistant message. After pressure recovery it regenerates from the
last committed conversation state. Version 1 does not claim exact continuation
from an arbitrary generated-token boundary.

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
outcomes, peak MLX allocation, swap/compressor deltas, outstanding reservation
count/bytes/oldest age, queued waiters, and cancellation/watchdog outcomes. They
contain no prompt content.

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
- The token fields published by `/v1/capacity` resolve through the same byte
  ledger used by request admission.
- Pressure and swap deltas downshift immediately.
- Warning and critical downshift formulas round and floor exactly as specified.
- Recovery requires hysteresis and raises only one bounded step.
- Cache hits may train observed retained residency but cannot lower an unsafe
  cold-prefill estimate.
- Overflow, missing metadata, and corrupt learned profiles fail conservative.
- Fingerprint changes invalidate persisted observations.
- A repeated generation under a new boot ID invalidates the Nanobot snapshot.
- The 32/64 GiB Escha examples reproduce from injected measurement inputs.

### Higgs integration tests

Use an injected pressure-observation interface and an artificially small
working-set budget. Tests do not force the development Mac into real swapping.

- Cache reclamation precedes rejection.
- A request outside the envelope is rejected before model allocation.
- Warning and critical signals update the capacity generation.
- Admission reservations include requested output.
- Concurrent requests cannot reserve the same working-set bytes, and cancelled
  requests release their reservations.
- Individually safe contention waits without returning 413, then revalidates at
  dequeue.
- Client disconnect while queued, mid-prefill, and mid-decode releases the RAII
  reservation only after the worker stops allocating.
- Engine error, unwind, and stall-watchdog cancellation cannot leak or expire a
  live reservation.
- Model switch waits for or joins reservation-owning workers.
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
- A same-generation 413 retry at the published limits is admitted; concurrent
  busyness queues instead of causing another compaction.
- `capacity_unavailable` leaves the work durable and visibly pending.
- Capacity recovery never reruns the outer user turn or a committed tool call.
- A terminal streaming capacity error stores partial output as incomplete.
- An old Higgs endpoint selects the conservative compatibility budget.

### Real hardware validation

Run progressively larger genuine agent sessions against EschaMoE on the target
Mac. Synthetic prompts may support diagnostics but cannot be the only shipping
evidence.

The learning-liveness replay interleaves prefix-cache-heavy tool turns with
three genuine cold session starts in the boundary band and proves a bounded
upward step. A cache-only variant proves cold-prefill cost never becomes more
optimistic without cold evidence.

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
9. Reusing a generation number under a new boot ID cannot reuse old capacity or
   retained-session state.
10. Logs and TUI explain every reduction, eviction, compaction, wait, and retry.
11. Ordinary warm turns show no material regression in the matched turn
    benchmark.
12. Higgs release checks and `higgs doctor` validate all changed configuration
    semantics.
13. Nanobot passes release build, regression tests, matched turn benchmark, and
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
