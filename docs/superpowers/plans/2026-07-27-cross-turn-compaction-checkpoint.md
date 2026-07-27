# Cross-Turn Compaction Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent a compacted LCM DAG from reaching a warm Higgs session before
the pending checkpoint rotates that session.

**Architecture:** Give each concrete SQLite session one shared
`CompactionHandle`. During preparation, ingest durable history but use raw
history whenever that handle is in flight or contains a pending checkpoint;
the existing checkpoint installer remains the only rewrite publication path.

**Tech Stack:** Rust 2021, Tokio `Mutex`, atomics, SQLite session history,
existing LCM engine and Higgs cache rotation.

## Global Constraints

- Keep one production compaction and checkpoint-installation path.
- Do not modify `src/agent/lcm.rs` or its current uncommitted changes.
- Do not change checkpoint persistence or summary semantics.
- Run Cargo tests and builds with `--release`.
- Preserve concrete-session isolation across idle rollover.

---

### Task 1: Session-Scoped Compaction Handle

**Files:**
- Modify: `src/agent/agent_loop/compaction.rs:19`
- Modify: `src/agent/agent_loop/shared.rs:76`
- Modify: `src/agent/agent_loop/mod.rs:162`
- Modify: `src/agent/prepare_context.rs:263`
- Test: `src/agent/agent_loop/tests.rs`

**Interfaces:**
- Consumes: concrete `session_id: String`.
- Produces: a cloned `CompactionHandle` shared by every `TurnContext` for that
  concrete session.

- [x] **Step 1: Write the failing handle-reuse test**

Add a Tokio test using the existing local inline harness. Prepare two contexts
with the same `session_key` and assert:

```rust
assert!(Arc::ptr_eq(
    &first.compaction.in_flight,
    &second.compaction.in_flight
));
assert!(Arc::ptr_eq(&first.compaction.slot, &second.compaction.slot));
```

Then create an idle-rollover harness and assert the new concrete session does
not share either `Arc` with the expired session.

- [x] **Step 2: Run the focused release test and confirm RED**

```bash
cargo test --release --lib compaction_handle -- --nocapture
```

Expected: same-session contexts currently hold different Arcs.

- [x] **Step 3: Add the session handle map**

Make the existing handle cloneable:

```rust
#[derive(Clone)]
pub(crate) struct CompactionHandle {
    pub(crate) slot: Arc<tokio::sync::Mutex<Option<PendingCompaction>>>,
    pub(crate) in_flight: Arc<AtomicBool>,
}
```

Add to `AgentLoopShared`:

```rust
pub(crate) compaction_handles: Arc<Mutex<HashMap<String, CompactionHandle>>>,
```

Initialize it beside `lcm_engines` in `AgentLoop::new`. In `prepare_context`,
after resolving `session_id`, retrieve or insert one handle and clone it into
the returned `TurnContext`. Remove the two per-turn Arc allocations.

- [x] **Step 4: Run the focused handle tests**

```bash
cargo test --release --lib compaction_handle -- --nocapture
```

Expected: reuse and idle-rollover isolation tests pass.

### Task 2: Atomic DAG Adoption Gate

**Files:**
- Modify: `src/agent/prepare_context.rs:357`
- Test: `src/agent/agent_loop/tests.rs`

**Interfaces:**
- Consumes: raw SQLite `history`, the session `LcmEngine`, and its shared
  `CompactionHandle`.
- Produces: raw history while a rewrite is unpublished; otherwise the engine's
  durable active context.

- [x] **Step 1: Write the failing in-flight adoption test**

Use the harness to persist identifiable raw messages and obtain the
session-scoped engine. Mutate the engine so `active_context()` contains an LCM
summary, set:

```rust
handle.in_flight.store(true, Ordering::Release);
```

Prepare the next turn and assert the messages contain the raw marker and do not
contain the summary wire marker.

- [x] **Step 2: Write the failing pending-checkpoint adoption test**

Put a `PendingCompaction` into `handle.slot`, prepare another turn, and assert
the same raw-history property. The pending snapshot must match the raw message
prefix so the test represents an installable checkpoint.

- [x] **Step 3: Run the focused release tests and confirm RED**

```bash
cargo test --release --lib dag_adoption -- --nocapture
```

Expected: preparation currently adopts `engine.active_context()` despite the
shared rewrite state.

- [x] **Step 4: Gate active-context adoption**

After idempotently ingesting raw history and while holding the engine lock,
compute:

```rust
let rewrite_unpublished =
    compaction.in_flight.load(Ordering::Acquire)
        || compaction.slot.lock().await.is_some();
```

Return raw `history` when `engine.dag().is_empty()` or
`rewrite_unpublished`; otherwise return the filtered
`engine.active_context()`. This check occurs after waiting for the engine lock,
so either the background task still reports in-flight or has already published
its pending slot.

- [x] **Step 5: Run the focused adoption tests**

```bash
cargo test --release --lib dag_adoption -- --nocapture
```

Expected: both unpublished-rewrite cases preserve raw history.

### Task 3: Cross-Turn Checkpoint Publication

**Files:**
- Test: `src/agent/agent_loop/tests.rs`
- No additional production files.

**Interfaces:**
- Consumes: the existing `install_pending_compaction` path.
- Produces: one cache rotation followed by compacted-context adoption.

- [ ] **Step 1: Add the cross-turn regression**

Seed a warm prompt-cache watermark, a mutated engine DAG, and a pending
checkpoint on the shared handle. Prepare the next turn, call the existing
installation step with checkpoint permission, and assert:

```rust
assert!(installed);
assert_ne!(old_higgs_session_id, new_higgs_session_id);
assert!(ctx.messages.iter().any(|m| m.get("_lcm_summary").is_some()));
assert!(ctx.compaction.slot.lock().await.is_none());
```

Prepare one later turn and assert it adopts the compacted DAG without another
session rotation.

- [ ] **Step 2: Run the cross-turn release test**

```bash
cargo test --release --lib cross_turn_compaction -- --nocapture
```

Expected: the complete prepare/install/adopt sequence passes.

- [ ] **Step 3: Run release verification**

```bash
cargo fmt --all -- --check
cargo test --release --lib
cargo build --release
scripts/turn_bench.sh
```

Expected: formatting, all library tests, the release build, and the matched
turn benchmark pass.

- [ ] **Step 4: Check graph scope**

Run GitNexus change detection for the working tree and confirm only the
expected preparation, compaction coordination, and test flows changed.
