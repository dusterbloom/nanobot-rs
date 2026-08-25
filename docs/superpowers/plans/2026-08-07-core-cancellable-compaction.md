# Core Cancellable Compaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent background LCM compaction from blocking an interactive foreground turn while preserving atomic checkpoint publication.

**Architecture:** Each concrete session owns at most one soft-compaction task. Pre-call accounting requests soft work, but the task starts only after the foreground agent loop finishes. The next foreground turn cancels and joins generation before acquiring the LCM engine; publication and pending-checkpoint handoff are never force-aborted.

**Tech Stack:** Rust 2021, Tokio `JoinHandle`, `CancellationToken`, existing LCM mutation rollback and SQLite checkpoint transaction.

## Global Constraints

- Do not modify gateway scheduling, gateway command routing, channel shutdown, Higgs, provider protocol, LCM wire format, or SQLite schema.
- Soft compaction must never run ahead of foreground model inference.
- Escape/Ctrl-C remains bounded by Task 1 even if foreground cleanup does not cooperate.
- Cancellation before publication restores the prior DAG and active window.
- Publication and pending-checkpoint handoff must not be force-aborted.
- Hard-pressure compaction observes the current turn cancellation token.
- Interactive `/clear` reaps compaction before clearing durable state.
- Keep production changes small; delete obsolete detached-task coordination rather than layering over it.
- Do not commit or stage without explicit user instruction.

---

### Task 1: Owned Cancellation-Safe Compaction Job

**Files:**
- Modify: `src/agent/agent_loop/compaction.rs`
- Modify: `src/agent/agent_loop/shared.rs` only for owned-job call-site migration
- Modify: `src/agent/prepare_context.rs`
- Modify: `src/agent/agent_loop/mod.rs`
- Modify: `src/repl/commands/read.rs`
- Test: colocated tests in `src/agent/agent_loop/shared.rs` and `src/agent/agent_loop/tests.rs`

**Interfaces:**
- `CompactionHandle::new()`
- async `CompactionHandle::has_job()`, `has_pending()`, `try_start(F)`, and `cancel_and_reap()`
- cancellation-aware `execute_lcm_compaction(..., cancellation, publication_phase)`

- [ ] Add a failing test where cancellation wins before the engine lock and a later threshold check still requests soft compaction.
- [ ] Add a failing test where a cancelled reaper future is dropped, then a later reaper still joins the same task.
- [ ] Add a failing publication-handoff test: once publication starts, dropping the final handle cannot abort a task blocked on the pending slot.
- [ ] Run each focused test and record the expected RED.
- [ ] Replace `AtomicBool + detached spawn` ownership with a session-owned job containing token, handle, and a `Generating/Publishing/Aborting` phase handshake.
- [ ] Keep the job inside an async lifecycle mutex while joining so cancelled reapers cannot detach it and starts cannot observe false idle.
- [ ] Make `try_start` reject both an existing job and an uninstalled pending checkpoint.
- [ ] Move `request_async_compaction()` behind successful cancellable engine acquisition.
- [ ] Select cancellation while acquiring the engine and awaiting `LcmEngine::compact`; check/claim publication before SQLite persistence and do not select through persistence.
- [ ] In `prepare_context`, cancel and reap before acquiring the session engine.
- [ ] In interactive clear, cancel/reap and discard pending state before removing the engine; call this before working-memory/history deletion.
- [ ] Run focused lifecycle/checkpoint/clear suites and full release library tests.

---

### Task 2: Foreground-Priority Scheduling

**Files:**
- Modify: `src/agent/agent_core.rs` for prefix-independent pending snapshot matching
- Modify: `src/agent/agent_loop/shared.rs`
- Test: `src/agent/agent_loop/tests.rs`

**Interfaces:**
- `TurnContext::soft_compaction_requested: bool`
- `AgentLoopShared::spawn_requested_soft_compaction(&mut TurnContext)`
- Task 1's owned job API

- [ ] Add a coordinated provider regression: turn one finishes and starts blocked soft generation; turn two for the same concrete session must cancel it and reach its foreground provider call within two seconds.
- [ ] Add a hard-pressure regression: cancelling the turn during compaction returns promptly, restores the engine, and persists no summary checkpoint.
- [ ] Run both regressions and record expected RED against the current pre-call detached scheduler.
- [ ] At the start of compaction management, reap/cancel any prior soft job before any engine lock.
- [ ] Change `CompactionAction::Async` from immediate spawn to setting `soft_compaction_requested = true`.
- [ ] After `run_agent_loop` finishes, spawn requested soft work only if the turn was not cancelled and no pending checkpoint/job exists.
- [ ] Pass the current turn cancellation token to blocking compaction instead of a fresh token.
- [ ] Preserve `install_pending_compaction` as the sole prompt rewrite/cache-rotation publication path.
- [ ] Run focused soft/hard compaction, DAG-adoption, prompt-cache, and full release library tests.

---

### Task 3: Verification

- [ ] Run `git diff --check` on touched files.
- [ ] Run all Task 1 TUI cancellation suites.
- [ ] Run all Task 1-2 compaction regressions.
- [ ] Run `cargo test --release --lib` and `cargo build --release`.
- [ ] Refresh GitNexus and run `detect_changes` against `main`; affected flows must remain interactive TUI cancellation, LCM preparation/compaction, and interactive clear only.
- [ ] Review the final diff for gateway or unrelated-file changes; none are allowed.
