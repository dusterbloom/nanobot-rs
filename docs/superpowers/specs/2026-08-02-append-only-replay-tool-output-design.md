# Append-Only Replay and Durable Tool Outputs

**Date:** 2026-08-02
**Status:** Approved

## Problem

A live Nanobot/Higgs session exposed two related long-session failures.

First, Nanobot reloaded a shorter or byte-changed prompt from SQLite while
continuing to use the same Higgs retained-session identifier. Higgs found only
a 244-token common prefix, rejected continuation, and performed a complete
17,867-token prefill. The request took 311 seconds. This was not an LCM
checkpoint or a tool-schema change. It was an ordinary cross-turn history
reload whose filtering changed bytes that Higgs had already cached.

Second, individually bounded tool outputs still accumulated in the active
prompt. The observed prompt contained 19 raw tool results totalling 17,992
bytes, with a largest individual result of 7,227 bytes. Per-result limits did
not bound the aggregate, and repeated exploration increased both tool-call
count and prompt size.

The two failures share one architectural mistake: persisted conversation
history is being used both as an audit record and as a mutable model-facing
projection. Recomputing that projection on reload can change an already-sent
prefix.

## Goals

- Make every ordinary provider request an exact append-only extension of the
  previous request for the same retained Higgs session.
- Store every tool output durably and immutably before prompt shaping.
- Keep model-facing tool results bounded without losing exact retrieval.
- Preserve the newest tool evidence needed for the next reasoning step.
- Bound local-model tool loops to one 12-call lease with no automatic renewal.
- Leave the existing LCM compaction policy and thresholds unchanged.
- Keep the advertised local tool schema byte-stable across a session.

## Non-goals

- Redesigning LCM summarization or changing its thresholds.
- Dynamically adding and removing tool definitions per request.
- Asking the main model to summarize every tool output.
- Retrofitting old prompt messages on every turn to maintain a sliding output
  window.
- Guaranteeing warm-cache continuation across an explicit `/clear`, model
  switch, tool-topology change, or installed LCM checkpoint.

## Core Contract

### 1. Previously sent prompt rows are immutable

Once a model-facing message has been sent and persisted, routine session
reload must reproduce the same provider-visible bytes. Reload may project away
database-only metadata, but it must not:

- truncate or digest a stored tool result;
- replace a recalled result with a reference;
- drop an old turn because of a message or token window;
- rewrite a synthetic message that was previously provider-visible;
- change protocol pairing between an assistant tool call and its result.

The SQLite message row stores the canonical model-facing representation.
Filtering and prompt shaping happen before that row is first persisted, never
later during replay.

Existing LCM checkpoint installation and explicit session operations are the
only sanctioned history replacements. They already own cache invalidation.

### 2. Raw tool output and prompt representation are separate records

Every completed tool call, including errors, stores its full raw output in the
immutable `tool_results` table under `(session_id, tool_call_id)` before its
model-facing result is committed.

The associated `messages` row stores only the representation sent to the
model:

- a bounded deterministic preview when budget permits; or
- a deterministic handle containing tool name, call identifier, status,
  byte/character counts, digest, selected scalar arguments, and a short
  excerpt.

The full raw body never needs to be reconstructed from the prompt. Existing
`search_tool_result`, `slice_tool_result`, and `recall_tool_result` tools are
the retrieval path.

Storage is fail-closed. If the raw body cannot be proven durable, Nanobot must
not publish a handle that claims it can be retrieved and must not re-run a
side-effecting tool to repair the failure.

### 3. Output budgets apply only to newly appended results

Prompt shaping uses both a per-result limit and a shared budget for each tool
batch/turn. The newest one or two tool rounds receive the useful bounded
preview budget; additional results become compact handles.

The budget is applied while appending a new result. It never reaches backward
and rewrites a previously sent tool message. Consequently, aggregate tool
detail may grow across several user turns until ordinary LCM compaction runs,
but its rate is bounded and every exact body remains externally retrievable.

This is intentionally different from a sliding global cap: a sliding cap would
continually mutate the warm prefix and defeat Higgs retention.

### 4. Normal turns do not rotate the Higgs session

Fast TTFT comes from preserving the retained session and sending only the new
suffix. Routine history reload therefore neither clears prompt fingerprints
nor changes the prompt epoch.

Immediately before a provider request, Nanobot compares the provider-visible
request with the prior fingerprint:

- `First`: establish the baseline;
- `AppendOnly`: reuse the existing Higgs session;
- `Diverged`: report an invariant violation and recover safely by rotating the
  Higgs epoch before the request.

Rotation on divergence is a last-resort correctness mechanism, not the normal
history policy. It prevents sending incompatible bytes under an old retained
session identifier, but it does not make a cold prefill fast. Tests and warning
telemetry must make this path exceptional.

### 5. Sanctioned rewrites remain explicit

The following operations may replace the prompt and rotate/drop retained
state:

- LCM checkpoint installation;
- `/clear` or equivalent logical session reset;
- model switch;
- a real advertised tool-topology change;
- explicit emergency recovery after an append-only invariant violation.

LCM behavior and thresholds are unchanged by this work. Its cold request is
acceptable because it replaces a large history with a much smaller summary,
unlike the observed failure that cold-prefilled almost the entire long prompt.

### 6. Tool-loop convergence for local models

Local mode receives one atomic lease of 12 executed calls per user turn. There
is no automatic renewal. A batch that would exceed the remaining allowance is
rejected atomically and the turn ends with a concise explanation.

Existing protections remain:

- duplicate calls within a batch are collapsed;
- successful identical read-only calls within the turn use a compact cached
  receipt rather than executing again;
- repeated no-progress rounds terminate;
- the advertised tool schema remains unchanged while execution is blocked.

Cloud-mode policy is outside this change unless the same invariant is required
for shared code correctness.

## Why Not the Common Alternatives?

### Re-truncate history on every reload

This is simple for stateless APIs but changes old bytes. Under retained Higgs
sessions it creates exactly the expensive reset being fixed.

### Always rotate once per user turn

This is correct but discards nearly all retained-cache value. It converts every
turn into a cold prefill.

### Keep all raw output in prompt until context pressure

This preserves immediate detail but lets tools dominate the context and makes
small local models less reliable. The live session demonstrated this failure.

### LLM-summarize every tool result

Per-result summarization adds latency, model calls, nondeterminism, and another
failure path. Deterministic previews plus exact targeted retrieval are cheaper
and cache-stable. LCM remains responsible for narrative conversation
summarization.

### Dynamically advertise only currently relevant tools

Changing definitions rewrites the prompt head and forces retained-cache
rotation. Nanobot instead keeps a small stable native surface with stable
discovery/proxy tools.

## Industry Review

Local inference servers such as Ollama, llama.cpp, vLLM, and LM Studio mostly
serialize tool calls/results and leave retention policy to the client. Agent
harnesses add their own controls:

- Pi truncates individual outputs and writes the full body to a temporary file.
- Codex truncates tool output at history ingestion and compacts conversation
  history separately.
- Goose combines externalized large outputs, older-tool summarization, and
  context-pressure compaction.
- Gemini CLI combines per-output distillation, recent-output protection,
  older-output masking, and history compression.
- OpenHands uses threshold-driven event-history condensation.

Nanobot adopts the shared durable-body/bounded-view pattern, but shapes once at
ingestion to preserve retained-prefix bytes. Its SQLite result store and
targeted retrieval tools avoid temporary-file lifetime problems.

## Failure Handling

- Immutable result-store conflict: abort the turn with an infrastructure error.
- SQLite write failure: do not publish the prompt result; do not retry a
  side-effecting tool.
- Missing stored body during retrieval: return an exact error identifying the
  unavailable call ID.
- Unexpected prompt divergence: log the first divergent message class and
  hashes, rotate the retained epoch, queue the old Higgs session for drop, and
  continue once on the new session.
- Lease exhaustion: do not change tool definitions; reject the entire batch and
  finish the turn.

## Verification

Regression tests must prove:

1. Two ordinary user turns produce append-only provider requests and retain one
   Higgs epoch.
2. A stored model-facing tool result reloads byte-identically.
3. Raw tool output is stored for small, medium, large, successful, and failed
   results.
4. Multiple medium results share a bounded newly-appended budget and remain
   individually retrievable.
5. The newest tool rounds retain their deterministic previews.
6. Replay performs no tool-body capping, recall rewriting, or routine history
   window trimming.
7. An injected replay mutation is detected before provider I/O and rotates the
   Higgs epoch as emergency recovery.
8. Existing LCM checkpoint tests still rotate exactly once and otherwise remain
   unchanged.
9. Local mode admits at most 12 tool calls and offers no renewal path.
10. Stable tool definitions remain byte-identical through lease exhaustion.

Validation uses `cargo test`, `cargo build`, and `scripts/turn_bench.sh` because
the affected code is on the agent/provider/context hot path.

## Operational Signals

Keep one-line structured diagnostics for:

- raw tool bytes stored and model-facing bytes appended;
- shared preview budget consumed per batch/turn;
- prompt delta (`first`, `append_only`, `diverged`);
- retained epoch rotation reason;
- old Higgs session IDs queued/dropped;
- local lease usage and rejected batches.

A healthy long session should show append-only deltas on ordinary turns, small
prefilled suffixes, stable tool hashes, and no replay-time tool transformations.
