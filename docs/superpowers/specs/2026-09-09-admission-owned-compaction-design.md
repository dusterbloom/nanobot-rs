# Admission-Owned Compaction and Memory Reclamation

Date: 2026-09-10
Status: Approved for tasks 1-3
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

The foreground representation is a deterministic fold, not a generated
summary. It reuses the existing LCM summary node and `lcm_expand` interface:
the active prompt receives a compact pointer while SQLite retains every source
row. The pointer includes the exact source-ID ranges, the greatest covered
message ID as its revision, and a SHA-256 over the canonical covered rows.
Rendering the same rows therefore produces byte-identical output, and recall
can verify that it returned the same evidence. Existing tool-result handle
bytes are frozen; folding references them without changing their renderer.

The fold unit is the existing complete LCM block. Selection may end before an
assistant tool call or after its matching tool result, but never between them.
The protected recent tail and newest user request remain raw. No classifier,
new memory table, background service, or second compaction protocol is added.

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

Semantic refinement is outside tasks 1-3. A later experiment may run merged
`encoder_v0` on macOS 27 Core AI while the foreground lease is idle. Prior
branches already prove small-Qwen and real-Escha operations can execute on ANE,
but they also show that conversion, synchronization, and shared-memory costs can
erase operator-level gains. Core AI adds stateful KV, preallocated/direct
values, asynchronous compute streams, and ahead-of-time specialization; those
features make a bounded batch refiner worth measuring without reviving the
removed compactor sidecar. The experiment requires Xcode 27, which is not yet
installed on the test Mac.

Core AI's documented Neural Engine choice is a preference and may fall back to
GPU. ANE is also a separate compute engine, not separate memory. Promotion
therefore requires measured ANE placement with no GPU fallback, reference
parity, bounded memory release, no loss of the 45K Higgs envelope, no swap-out,
and no more than 3% Escha decode regression. If Core AI cannot exclude GPU, test
the public Core ML `cpuAndNeuralEngine` path instead. Private ANE APIs and an
HTTP sidecar remain excluded from production.

### Constrained tool-call contract

Nanobot's ordinary `tool_choice=auto` path remains model-selected. Required and
named calls, including recovery and a future semantic refiner, are hard
protocol contracts:

1. An empty grammar mask is an error; logits are never returned unmasked.
2. Rejecting a sampled token while advancing the grammar terminates generation.
3. A required or named request succeeds only with exactly one parser-visible,
   schema-valid tool call.
4. Blocking and streaming routes enforce the same postcondition.
5. A malformed required call is a typed terminal error and is never exposed as
   a successful assistant response.

Task 1 implements these guarantees in Higgs. Constraining ordinary automatic
choice would require a tool-or-final response envelope or an extra routing
inference and is deferred until matched evaluation justifies that protocol
change.

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

The first benchmark set uses task-specific fine-tunes rather than generic
instruct models:

1. `fuhao23/encoder_v0` is the best available task-fit candidate below 1B. It
   is an 8.7 MB LoRA for Qwen2.5-0.5B-Instruct trained to turn conversation
   segments into typed memory records with source spans. Its model card reports
   a 519-segment held-out evaluation, 3.4-second p50 on its reference hardware,
   and strong atomicity and self-containedness. It also documents weak entity
   coverage and a fall from 100% to about 64% schema compliance without
   constrained decoding. Evaluation therefore requires merging the adapter,
   converting the merged model to MLX, and enforcing the Nanobot recovery
   schema during decoding.
2. `Harsha901/qwen2.5-0.5b-kd-merged-cnndm-50k` is the best evidence-backed
   ordinary summarizer at this size. The merged 0.5B model publishes
   cross-domain ROUGE and BERTScore results on XSum, SAMSum, and DialogSum. It
   remains a news-trained model with a 1,536-token training sequence limit, so
   it is a control rather than the assumed winner. Its F16 safetensors can be
   converted and quantized for MLX.
3. `wallster88888/Qwen2.5-1.5B-Instruct-Summarizer-4bit` is the ready-made MLX
   integration control. It is an approximately 880 MB 4-bit artifact derived
   from `agentlans/Qwen2.5-1.5B-Instruct-Summarizer`. Neither repository
   publishes a meaningful summarization evaluation, so native format alone is
   not evidence of quality.
4. `ericflo/qwen3-0.6b-summarizer` is a useful technical-headline control. It
   was distilled on 6,720 software and project summaries, but is trained for one
   sentence and recommends only about 2,000 input characters. The release is
   GGUF plus a custom LoRA, not a native MLX artifact, and is unsuitable as the
   sole full-session compactor.

The 2026 deterministic-memory search reinforces the foreground fold rather than
replacing it. Zero-Mem keeps original traces as the source of record while
using deterministic retrieval, but its public repository is still empty as of
2026-09-10. The newer
Compaction Cliff paper releases a classifier and reference implementation; its
typed deterministic operators preserve safety rules far better than uniform
LLM summaries. Those systems add retrieval and classification machinery that
the pressure path does not need, but they support the same rule: exact evidence
stays durable and compaction changes only the bounded working representation.

Fidelity Before Structure likewise reports that verbatim chunks beat generated
structured artifacts on LoCoMo and LongMemEval-S. LycheeMemory V2 and SimpleMem
are released background-memory systems, but their learned extraction cannot
guarantee complete tool bytes or bounded recovery. Parallel Context Compaction
can hide some summarization latency, yet remains learned and lossy. Agent Zero
Memory adds strong provenance and citation locks, but publishes no code artifact
on its arXiv page. `paritok-4b-v1` is the strongest released tool-trajectory
compressor found; it retains about 25.7% of input while preserving 86.5% of
uncompressed SWE-bench solve quality, which is useful as an upper-bound control
but is neither small nor lossless.

Relevant artifacts:

- `https://github.com/Zero-Mem/Zero-mem` (public placeholder; no files yet)
- `https://github.com/searchsim-org/cikm26-knowledge-triage`
- `https://huggingface.co/datasets/searchsim/AgentArtifactCorpus`

`IAAR-Shanghai/MemReader-0.6B` would be the strongest specialized candidate on
the published memory benchmarks: the MemReader card reports 79.56% LOCOMO,
80.20% LongMemEval, and 93.76% HaluMem extraction F1 for the 0.6B variant.
However, Hugging Face exposes no public 0.6B checkpoint as of 2026-09-10; only
`IAAR-Shanghai/MemReader-4B-thinking` is downloadable. The 4B model is an
upper-bound quality control, not a pressure-path dependency.

Generic `mlx-community/Qwen3.5-0.8B-5bit` and
`mlx-community/Qwen3-0.6B-4bit` remain untuned baselines. They are not described
as compactor fine-tunes.

Local inventory on 2026-09-10 found no `encoder_v0` artifact. The ordinary
Hugging Face cache link for `mlx-community/Qwen3-0.6B-4bit` is broken, but a
complete 320 MB copy exists under `/Users/peppi/AI-Models/shared/huggingface/`
and is the integration control. The benchmark must verify required files at the
resolved path before advertising any candidate as available.

No compactor is loaded concurrently automatically. A missing candidate leaves
the deterministic checkpoint path fully operational.

## Recovery Checkpoint Shape

The serialized form is versioned and bounded. A representative logical form is:

```text
version=1 ranges=<exact compact ID ranges>
revision=<greatest covered message ID> source_sha256=<digest>
lcm_expand({"message_ids":"<the same exact ranges>"})
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
