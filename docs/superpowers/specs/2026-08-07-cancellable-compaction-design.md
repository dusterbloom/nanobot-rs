# Cancellable Foreground-Priority Compaction Design

Date: 2026-08-07

## Objective

Keep interactive turns and terminal shutdown responsive when LCM compaction is
slow, without exposing a non-durable compaction mutation or running background
summarization ahead of foreground inference.

## Proven Failure

Session `20260807_142618_98f6b8` crossed the LCM soft limit during a tool loop.
The detached soft-compaction task acquired the session's `LcmEngine` mutex and
held it across three LFM generations. The next foreground iteration blocked on
that mutex for more than three minutes.

Escape and Ctrl-C fired the turn cancellation token, but `TurnStream` then
waited indefinitely for the blocked agent task. Ctrl-D had no streaming quit
mapping at all.

## Interaction Contract

- Escape cancels the current turn and returns to the prompt.
- The first Ctrl-C cancels the current turn and returns to the prompt.
- A repeated Ctrl-C while cancellation is pending cancels and exits.
- Ctrl-D while streaming cancels and exits.
- Partial assistant output from a cancelled turn remains discarded.
- Cancellation has a bounded fallback if an agent task does not cooperate.

## Compaction Contract

- Foreground work always has priority over soft compaction.
- Each concrete session owns at most one soft-compaction job.
- The session retains that job's cancellation token and join handle; no
  compaction task is detached.
- Before foreground preparation or an LCM pre-call step waits on the engine,
  it cancels and reaps an active soft job.
- Hard-pressure compaction may block inference because the raw prompt cannot be
  admitted safely, but it observes the current turn's cancellation token.
- Cancelling model generation drops the tentative `LcmCompactionMutation`,
  restoring the prior DAG and active window.
- SQLite checkpoint persistence and publication remain one short atomic
  boundary. Cancellation is checked before that boundary, not through it.
- Interactive session clear cancels and reaps active jobs before deleting state.

## Structure

Replace `CompactionHandle`'s detached `in_flight` bit with explicit owned job
state shared by every turn of the concrete session. A soft job contains a
cancellation token and join handle and writes a successful result into the
existing pending checkpoint slot.

Pre-call accounting records that soft compaction is needed, but starts it only
after the foreground agent loop finishes. Foreground preparation and
`manage_compaction` use one helper to reap a finished job or cancel and await a
running job before acquiring the LCM engine. The helper keeps the job in an
async lifecycle mutex while awaiting it, so a cancelled reaper cannot detach
the task and concurrent starts cannot observe a false idle state.

`execute_lcm_compaction` accepts a cancellation token. Engine acquisition and
LLM summarization are cancellation-aware. Once a valid compacted state is ready,
the existing transaction persists the summary node and working-memory snapshot
before the in-memory mutation becomes observable.

`TurnStream` gives cooperative cancellation a short grace period. If the
foreground agent task still has not resolved, it aborts and joins that task
before reporting the turn finished. The background compaction task itself is
never force-aborted after checkpoint publication can begin: model generation
observes its token, while SQLite publication and pending-checkpoint handoff run
to completion under retained session ownership. Explicit quit escalation does
not wait for another key cycle.

## Rejected Alternatives

- A UI-only timeout leaves detached compaction holding the LCM lock and GPU, so
  the next turn freezes again.
- Optimistic snapshot/generate/validate compaction needs DAG revision and merge
  machinery while still requiring GPU preemption. It is unnecessary for the
  foreground-priority behavior requested here.
- Disabling LLM compaction or always truncating would trade responsiveness for
  silent context loss.

## Tests

- A non-cooperative `TurnStream` agent is aborted and finishes within the
  cancellation bound.
- Streaming Escape, Ctrl-C escalation, and Ctrl-D produce the specified actions.
- A stalled soft compactor is cancelled and reaped before foreground LCM access;
  the foreground does not wait for the model-generation gate.
- Cancelling hard compaction restores the prior DAG and does not publish a
  checkpoint.
- Clear and shutdown cannot leave or revive a session compaction job.

Use blocking test providers and synchronization barriers; no live model is
required. Run focused tests first, then the complete release library suite and
release build.

## Scope

Expected production files:

- `src/agent/agent_loop/compaction.rs`
- `src/agent/agent_loop/shared.rs`
- `src/agent/prepare_context.rs`
- `src/agent/agent_loop/mod.rs`
- `src/repl/commands/read.rs`
- `src/turn_stream.rs`
- `src/tui_app/app.rs`
- `src/tui_app/mod.rs`

Tests remain in the existing colocated modules. Gateway scheduling, channel
shutdown, provider protocol, LCM summary format, SQLite schema, and Higgs engine
behavior are not changed.
