# Admission-Owned Compaction and Memory Reclamation

Date: 2026-09-09
Status: Approved design; awaiting written-spec review
Scope: Nanobot prompt admission and compaction; Higgs request lifetime,
cancellation, and retained-session reclamation

## Problem

The production failure on 2026-09-09 combined a prompt-capacity collapse with a
request-lifetime gap. Nanobot had an approximately 31K-token active prompt when
Higgs reduced its safe envelope from 51.2K to 20.5K, then zero, then 8.2K. The
client timed out while Higgs continued a retained-session prefill. Nanobot then
entered model-based LCM compaction even though the same MLX process was still
working on the disconnected request. Several slow summary calls followed while
the foreground turn remained unavailable.

This exposes two incorrect dependencies:

- hard-pressure recovery depends on successful model inference; and
- the HTTP response lifetime does not completely own the corresponding Higgs
  worker and capacity reservation.

The repair must make every allocation owned and accounted until it is released,
and must make foreground compaction independent of model availability.

## Evidence

The incident logs establish the following sequence without inference about the
model's behavior:

1. Higgs accepted a retained-session request near 31K prompt tokens.
2. Nanobot's 120-second inactivity timeout elapsed before Higgs finished.
3. Higgs continued for about 250 seconds and reported approximately 245 seconds
   of prefill.
4. During that work, the capacity envelope collapsed to 8.2K.
5. A deterministic recovery node was persisted, but foreground recovery
   continued into several model compaction calls instead of terminating.
6. The retained-session prefill path did not use the same closed-stream
   observation installed on the cold chunked-prefill path.

The capacity reservation was present in Higgs's registry until worker exit, so
the allocation was not absent from accounting. It was orphaned from the client
request lifecycle: dropping the response did not promptly stop and join the
worker that owned the reservation.

## Goals

1. An over-capacity prompt never enters model inference.
2. Foreground recovery completes without an LLM, network request, or MLX
   allocation.
3. Every generation allocation has one visible owner from admission through
   worker completion.
4. Capacity collapse produces one bounded checkpoint and at most one retry.
5. Exact history and tool output bytes remain durable and recoverable.
6. Semantic summaries may improve recovery quality but never gate user work.
7. The production path remains channel -> agent loop -> provider -> tools ->
   reply, with no sidecar protocol or parallel compaction pipeline.

## Non-Goals

- Killing an individual uninterruptible MLX kernel inside the process.
- Predicting future memory consumption by unrelated applications.
- Replacing SQLite history with generated summaries.
- Keeping a dedicated compactor resident before measurements justify its cost.
- Adding a separate tokenizer-reservation or compactor HTTP service.

## Invariants

### Prompt admission

1. Only a server-tokenized, atomically reserved prompt may enter inference.
2. Nanobot estimates prompt size before every provider call and refreshes the
   current Higgs capacity generation.
3. When the estimate exceeds the current prompt allowance, Nanobot checkpoints
   before sending the request.
4. Higgs remains the exact authority. A request that passes the estimate but
   fails exact admission returns typed `higgs_capacity_exceeded` before model
   execution.
5. Nanobot consumes the returned exact allowance, checkpoints once, and retries
   once. Another rejection suspends the durable pending turn; it does not loop.

An oversized serialized request may reach Higgs's admission endpoint because
Nanobot's estimate cannot be exact for every provider template. It may not reach
the inference engine. Avoiding the HTTP payload itself would require a separate
reservation ticket and introduce a time-of-check/time-of-use race without
improving memory safety.

### Request ownership

1. One Higgs generation owns one generation ID, capacity reservation,
   `GenerationStop`, response-stream lifetime, and worker completion handle.
2. Dropping the response stream signals `ClientDisconnect`.
3. Cold prefill, retained prefill, and decode observe the same stop state.
4. Closed progress or token receivers are cancellation, not successful
   completion.
5. The reservation remains accounted until the worker acknowledges cancellation
   and exits.
6. New same-session inference and model refinement cannot start while the old
   worker owns the session reservation.
7. Retained cache retirement and MLX cache reclamation occur only after that
   worker is quiescent.

An MLX operation may be uninterruptible until its current dispatch returns.
During that interval its memory remains owned and competing work is not
admitted. Cancellation is checked at every bounded prefill chunk and decode
step. A hard wall-clock kill would require process isolation and is deferred
unless bounded cancellation proves insufficient in measurement.

### Foreground compaction

1. Hard-pressure compaction is one local SQLite transaction.
2. It makes no provider call and requires no MLX memory.
3. It replaces the entire compactible prefix in one pass rather than repeatedly
   compacting prompt-sized batches.
4. It preserves the newest complete user turn and atomic assistant-tool-result
   groups needed to resume it.
5. Replaced messages and exact tool bytes remain in append-only SQLite history.
6. The active window receives one recovery checkpoint no larger than 256
   estimated tokens.
7. The checkpoint records the session, exact covered message ranges, source
   hash, current objective, and recall instruction.
8. Publication is atomic. Failure rolls back to the previous active window.
9. Rewriting the active prompt retires its old Higgs retained-session identity
   before retry, preventing prefix reuse against different bytes.
10. At most one foreground compaction job exists per session.

The target is p99 below 250 ms on the supported Mac and a measured end-to-end
bound below one second. Correctness tests assert absence of model/network waits;
performance tests report the wall-clock distribution without making ordinary
CI timing-sensitive.

### Semantic refinement

1. The deterministic recovery checkpoint is sufficient for correctness.
2. A semantic refiner may run only when no foreground generation owns the MLX
   execution lease.
3. Foreground work cancels and reaps refinement before inference admission.
4. Refinement reads exact source records from SQLite and writes a tentative
   result separately.
5. Publication requires a complete response, size validation, source-range
   validation, and an atomic compare against the checkpoint revision.
6. Timeout, cancellation, malformed output, capacity rejection, or process
   restart leaves the deterministic checkpoint intact.
7. The refiner uses the existing OpenAI-compatible provider path and an optional
   model name. It does not introduce model-directory, port, or protocol modes.

## Production Flow

```text
prepare prompt
  -> refresh capacity
  -> local estimate fits?
       no  -> deterministic checkpoint
  -> exact Higgs tokenization and atomic reservation
       413 -> checkpoint once -> retry once
  -> admitted generation
  -> stream owns cancellation and worker lifetime
  -> response or typed terminal outcome
  -> optional idle semantic refinement
```

When capacity collapses during an active request:

```text
capacity generation narrows
  -> request stop is signalled
  -> local checkpoint can be persisted immediately
  -> wait for the generation reservation to quiesce
  -> retire incompatible retained cache
  -> render bounded active prompt
  -> exact admission or durable suspension
```

Compaction does not need to wait for MLX in order to persist its checkpoint.
Resume inference does wait for the previous generation to release its owned
reservation. This prevents recovery from contending with the allocation it is
trying to escape.

## Minimal-Prompt Failure

The immutable minimum is the system prompt, stable tool catalog, newest user
request, required protocol receipts, and output reserve. If that minimum cannot
fit the exact current allowance, Nanobot persists one suspended pending turn
with the typed capacity reason. It makes no provider retry until a later capacity
generation can admit the minimum.

This is the only legitimate capacity wait. Ordinary oversized history is always
reduced locally first.

## Optional Compactor Model

The compactor model is a quality optimization after recovery. It is selected by
measurement on actual Nanobot sessions, including decision retention, open-loop
retention, source-ID fidelity, malformed output rate, latency, peak MLX memory,
and cancellation latency.

Provisional candidates are:

1. `mlx-community/Qwen3.5-0.8B-5bit`, the same current model family and first
   quality candidate.
2. `mlx-community/Qwen3-0.6B-4bit`, the smaller latency and memory control.
3. `mlx-community/LFM2-1.2B-4bit`, an independent edge-model challenger.

Local inventory on 2026-09-09 found no usable small Qwen weights. The Hugging
Face cache entry for `mlx-community/Qwen3-0.6B-4bit` is a 12 KB metadata record
whose snapshot symlink points to the deleted LM Studio directory
`~/.cache/lm-studio/models/mlx-community/Qwen3-0.6B-4bit`. The cached
`Qwen/Qwen2-0.5B` contains only a tokenizer. `MiniCPM5-1B-4bit` is likewise an
empty model directory. The benchmark therefore requires downloading a complete
candidate artifact and must verify required files before advertising it as
available.

No compactor is loaded concurrently automatically. A missing candidate leaves
the deterministic checkpoint path fully operational.

## Recovery Checkpoint Shape

The serialized form is versioned and bounded. A representative logical form is:

```text
RECOVERY_CHECKPOINT v1
session=<session key>
covered=<first message id>..<last message id>
source_sha256=<digest>
objective=<bounded latest objective>
resume=Use recall for exact prior messages and tool outputs.
```

The concrete implementation reuses the existing LCM recovery node and session
history schema where possible. It does not create a second memory store. Tool
result handles remain the active representation for large raw outputs.

## Telemetry

Each incident records enough facts to establish ownership and bounded recovery:

- capacity boot ID and generation;
- exact prompt and output allowance;
- generation ID and session ID;
- reserved bytes and active reservation count;
- cancellation reason, request time, acknowledgement time, and worker exit;
- checkpoint source range, before/after prompt estimate, and duration;
- retained-session retirement;
- exact admission outcome and retry count;
- semantic refinement model, status, duration, and peak allocation.

Logging contents or logits is not required to diagnose this systems failure.
Existing exact request/response artifacts remain the source for model-behavior
reports.

## Acceptance Tests

### Higgs request lifetime

- A retained 31K prompt stalls beyond the client timeout. Dropping the stream
  signals `ClientDisconnect`, the worker exits, reservation count reaches zero,
  and active MLX allocation returns to the measured baseline before competing
  work is admitted.
- Cold and retained prefill observe cancellation at every chunk boundary.
- A closed receiver cannot produce a successful generation result.
- A same-session request arriving before worker exit receives a typed busy or
  capacity outcome and cannot overlap the reservation.

### Nanobot recovery

- Capacity falls from 51.2K to 8.2K with a 31K active prompt. One deterministic
  checkpoint brings the active prompt below the effective prompt allowance and
  makes zero model-compaction calls.
- An authoritative 413 after a passing estimate causes one checkpoint and one
  retry. A second rejection durably suspends the turn.
- User input during compaction coalesces with the pending turn and does not start
  another compaction or duplicate provider request.
- Rewriting the prompt retires the previous retained-session identity before
  retry, with no token-mismatch recovery prefill.
- Assistant tool calls and their results remain atomic; exact tool bytes are
  recoverable by handle after checkpointing.
- Transaction failure exposes neither a partial checkpoint nor a partially
  replaced active window.

### Optional refinement

- A refiner that hangs, disconnects, returns malformed output, exceeds its size
  bound, or receives 413 cannot delay or corrupt foreground recovery.
- New foreground work cancels and reaps refinement before admission.
- A valid summary atomically replaces only the checkpoint revision from which it
  was derived.

### End-to-end adversarial replay

Replay the production incident with scripted capacity transitions, stalled
retained prefill, a client timeout, tool-heavy history, and a user continuation.
Assert:

- zero unowned reservations;
- one foreground checkpoint;
- no foreground model-compaction requests;
- at most one generation retry;
- no duplicate tool execution;
- bounded local checkpoint duration;
- a final response or one typed durable suspension.

## Implementation Boundaries

Nanobot keeps capacity comparison, deterministic recovery, and retry ordering in
the existing agent-loop hot path. Shared calculations are extracted on their
second use, without adding `_gate`, `_guard`, or parallel recovery modules.

Higgs reuses `RequestReservation` and `GenerationStop`. The response-stream drop
observer and the unified prefill cancellation check belong beside the existing
route and generation implementations. No new daemon or service is introduced.

## Rollout Gates

1. Land and verify request ownership and retained-prefill cancellation in Higgs.
2. Land deterministic one-pass foreground checkpointing and bounded retry in
   Nanobot.
3. Pass focused release tests, full `cargo test --release`, release builds, and
   the production incident replay with zero warnings.
4. Run matched turn benchmarks for agent-loop and provider changes.
5. Download and benchmark small compactor candidates only after the correctness
   path is proven.
6. Enable semantic refinement only if it improves the quality gates without
   violating foreground cancellation and memory bounds.
