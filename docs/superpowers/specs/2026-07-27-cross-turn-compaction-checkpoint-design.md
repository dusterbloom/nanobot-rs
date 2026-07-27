# Cross-Turn Compaction Checkpoint Design

## Objective

Make asynchronous LCM compaction and the Higgs retained-prefix cache obey one
atomic visibility boundary across Nanobot turns.

## Proven Failure

The SQLite history and Higgs trace establish this ordering:

```text
turn N: raw history is sent under Higgs session H
turn N: async LCM mutates the shared engine and persists a summary
turn N: the pending checkpoint is deferred to preserve H's warm prefix
turn N+1: prepare_context adopts engine.active_context()
turn N+1: compacted history is sent under the unchanged H
Higgs: retained prompt is not a prefix -> ExactBootstrap/not_growing
```

The LCM engine is session-scoped, but `CompactionHandle` is currently created
inside `prepare_context` and is turn-scoped. A later turn therefore observes
the mutated engine without observing the in-flight or pending checkpoint that
must govern its installation.

## Session Coordination Invariant

Each concrete SQLite session owns:

- one `LcmEngine`;
- one `CompactionHandle`, containing its pending-result slot and in-flight bit.

Preparation always ingests durable SQLite rows into the engine. It may adopt
`engine.active_context()` only when the same session handle reports neither an
in-flight mutation nor a pending checkpoint. Otherwise it builds the next turn
from raw SQLite history, preserving the append-only prompt expected by the
current Higgs session.

The existing `install_pending_compaction` path remains the only way to publish
the rewritten active window. It validates the source snapshot, rotates the
Higgs session through `invalidate_prompt_cache_for_rewrite`, and applies the
checkpoint exactly once.

This preserves the intended behavior:

- soft compaction can finish in the background;
- a warm foreground turn remains append-only until safe installation;
- hard pressure can wait and install immediately;
- restart reconstruction still adopts durable summary nodes when no in-process
  checkpoint is pending.

## Scope

Modify only:

- `src/agent/agent_loop/compaction.rs`
- `src/agent/agent_loop/shared.rs`
- `src/agent/agent_loop/mod.rs`
- `src/agent/prepare_context.rs`
- `src/agent/agent_loop/tests.rs`

Do not alter LCM summarization, range selection, SQLite checkpoint format,
Higgs session-ID derivation, or the user's uncommitted `src/agent/lcm.rs`
changes.

GitNexus reports CRITICAL upstream impact for `CompactionHandle`,
`install_pending_compaction`, and `execute_lcm_compaction` because they feed the
main agent loop. The implementation therefore adds coordination only and keeps
the current checkpoint installation path intact.

## Tests

Write regressions before production edits:

- two turns for one concrete session receive the same `CompactionHandle`;
- while that handle is in flight, a mutated DAG is not adopted by
  `prepare_context`;
- while a checkpoint is pending, raw history remains the prepared prompt;
- after safe installation, the rewrite rotates cache state once and later
  preparation adopts the compacted DAG;
- idle rollover receives an independent engine and handle.

Run every test and build in release mode:

```bash
cargo test --release --lib agent::agent_loop::tests -- --nocapture
cargo test --release --lib
cargo build --release
```

Run `scripts/turn_bench.sh` after correctness tests, then use GitNexus
`detect-changes` to confirm the affected flow is limited to session preparation
and compaction installation.
