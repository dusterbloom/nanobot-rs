# KV cache and compaction review — September 12, 2026

The architecture has acquired unnecessary coupling: a cache optimization now constrains whether context can be reclaimed, while reclamation, admission, retry, and persistence maintain overlapping state. Keep the durable-history and protocol-integrity work. Simplify the policies around it before adding another compactor or cache representation.

This is a focused risk review of the month's history and current KV/compaction paths, not a claim that every changed line in both repositories was audited.

## Scope and branch identity

| Repository/ref | Reviewed tip | Commits since Aug 12 | Since Sep 5 |
|---|---|---:|---:|
| nanobot-rs/main | `9fecb314c5864f7c67c48ec7664f6e51b58601ad` | 136 | 29 |
| higgs/main | `c915efabd97880254967edd6d3161bed40b9e03f` | 0 | 0 |
| higgs/nightly, requested development branch | `42abcf64b` | 197 | 24 |

Counts use local refs and September 12's available history, with Aug 12/Sep 5 midnight Europe/Rome cutoffs. Nanobot local main is 89 commits ahead of origin/main. Higgs main's last commit is July 14; the working checkout and recent development are nightly. The user clarified that nightly is the intended Higgs review target. **The Higgs findings below concern nightly.** No fetch or branch changes were performed. Existing dirty files were preserved.

## Actionable findings

### 1. [P1] A permanent prefix pin makes a feasible capacity recovery impossible

Nanobot, introduced by `33a060d`, September 11. [lcm.rs:809](/Users/peppi/Dev/nanobot-rs/src/agent/lcm.rs:809).

Every compaction span is clamped past `pinned_prefix_entries`, including deterministic recovery under a smaller budget. The first successful prefix-preserving cut commits this lifetime pin; subsequent cuts cannot retire it. A mutable server capacity envelope and an irreducible old conversation prefix are incompatible.

**Executed reproduction:** 80 alternating persisted user/assistant messages total 40,400 estimated tokens. Perform a deterministic fold under a 49,152-token context with 4,096 output reserved, then reduce to `TokenBudget::new(4000, 1000)` (3,000 prompt tokens). With the default `keep_prefix_fraction=0.35`, recovery stops at **14,374 tokens**; seven further compactions return `None`. The identical fixture with fraction `0.0` reaches **1,248 tokens**. The active raw evidence remains recoverable; the pin alone prevents fitting it.

The admission gate can consequently park work despite a feasible smaller projection. Repeating the same fold after a retry does not remove this obstruction. Make prefix preservation revocable when admission needs a deeper cut. Budget the retained head in absolute tokens against the current envelope; a historical fraction is not an admission invariant.

Reproducer: [pin_repro.rs](/Users/peppi/Dev/nanobot-rs/docs/reviews/2026-09-12-kv-compaction/pin_repro.rs). This exercises the actual release LcmEngine without a model or network.

### 2. [P2] The eager session-release request addresses a nonexistent route

Nanobot, introduced by `9fecb31`, September 11. [openai_compat.rs:1610](/Users/peppi/Dev/nanobot-rs/src/providers/openai_compat.rs:1610).

The provider uses a versioned API base for chat, such as `http://host:port/v1`, but the new method appends `/v1/sessions/drop`. A localhost mock receiving the actual release provider call recorded:

```text
POST /v1/v1/sessions/drop HTTP/1.1
result=Err(HttpError("session drop failed: 404 "))
```

The asynchronous caller discards that error. This defeats the new eager-reclamation mechanism; the older piggybacked release remains the fallback. It does not prove every retained cache leaks indefinitely, but it invalidates the claim that this hook frees the old KV before the next prefill.

Use the same version-aware URL construction already used by `capacity_url_from_base`. Verify request routing end to end, rather than only checking that a callback fired. Fire-and-forget execution also provides no ordering guarantee that reclamation finishes before the next prefill; if that ordering is needed for memory admission, make it explicit.

Reproducer: [drop_repro.rs](/Users/peppi/Dev/nanobot-rs/docs/reviews/2026-09-12-kv-compaction/drop_repro.rs). It uses an ephemeral localhost server, an empty credential, and no real model.

### 3. [P2] New prefix state is outside the compaction transaction

Nanobot, introduced by `33a060d`, September 11. [lcm.rs:625](/Users/peppi/Dev/nanobot-rs/src/agent/lcm.rs:625).

`LcmCompactionState` snapshots only the DAG and active messages. `compact()` updates `pinned_prefix_entries` before SQLite publication. If checkpoint persistence fails, `LcmCompactionMutation::drop` restores the DAG and messages but leaves the pin from the rejected checkpoint. Cancellation can similarly leave `pending_pinned_prefix` behind.

Future folds then skip raw history based on a checkpoint that never became durable; rebuilding from the database derives different state. Include both pin fields in snapshot/restore, or make the cut local to the pending transaction and publish all of it together. This finding is verified by tracing the mutation and rollback code; no database-failure injection was executed.

### 4. [P1, Higgs nightly] Hybrid disk restore loses recurrent shape and precision

Introduced by `8d994070e`, September 11. [disk_prefix_cache.rs:688](/Users/peppi/Dev/higgs/crates/higgs-engine/src/cache/disk_prefix_cache.rs:688).

The new hybrid snapshot restores convolution and SSM state as Float16 arrays of shape `[1, len]`. Production GDN convolution state is rank three; its SSM state is rank four and Float32. The actual consumer explicitly rejects the wrong dtype and shape in [qwen3_next.rs:2369](/Users/peppi/Dev/higgs/crates/higgs-models/src/qwen3_next.rs:2369).

Trigger: enable/use hybrid disk-prefix persistence, save an aligned prefix, restart, restore it, and continue inference. Preserve the original dimensions and required state precision in the format. Casting a previously rounded SSM back to Float32 would not restore the original values.

Two existing release hybrid snapshot tests pass because their toy states already have rank two and they compare flattened values. They do not test continuation through a real GDN layer. The incompatible producer/consumer contract is source-verified; a real-model restart was not run in this review.

### 5. [P2, Higgs nightly] Session release blocks an async executor worker

Introduced by `10b845e2b`, September 11. [chat.rs:2022](/Users/peppi/Dev/higgs/crates/higgs/src/routes/chat.rs:2022).

The async route directly invokes synchronous `drop_retained_session`, which waits on the session's standard mutex in [simple.rs:4545](/Users/peppi/Dev/higgs/crates/higgs-engine/src/simple.rs:4545). An in-flight session does not promptly report `false` as the endpoint comments claim. A drop can occupy a Tokio worker until generation finishes; concurrent drops can exhaust the worker pool and disrupt unrelated HTTP/stream progress.

The existing release test `simple_engine_drop_retained_session_waits_for_session_lock` passes and explicitly verifies the waiting behavior. Prefer a nonblocking drop implementing the advertised busy response. If waiting is intentional, move it off the async executor and document the actual behavior. Fix this together with Nanobot's URL: correcting the client activates the problematic server path.

### 6. [P2, Higgs nightly] Documented disk-cache knobs do not configure the server cache

Introduced into nightly by `84af237d5`, August 20, merging the August 15 disk-store work. [config.rs:576](/Users/peppi/Dev/higgs/crates/higgs/src/config.rs:576).

`kv_disk_dir` and `kv_disk_space_mb` are exposed, validated by doctor, and recommended in the daemon configuration example. However, the server loader calls `disk_prefix_cache_config` at [state.rs:1499](/Users/peppi/Dev/higgs/crates/higgs/src/state.rs:1499), and that method reads only the separate `disk_cache_enabled`, `disk_cache_path`, and block-limit settings at [config.rs:841](/Users/peppi/Dev/higgs/crates/higgs/src/config.rs:841). Setting the directory and byte budget alone therefore leaves persistence disabled.

The imported `DiskPrefixStore` implementation has no production callers. A separate public engine-load adapter also discards the byte budget and treats the directory as a file path, but that adapter is not the current server loading path. Consolidate these settings into the active implementation and preserve an explicit storage-budget contract. This is verified by tracing all configuration consumers and the server constructor; no new model-load experiment was run.

## What the history and session evidence support

Late August improved append-only prompt discipline and separated sent messages from drafts. September 1–4 added durable replay and live-capacity handling. September 5–8 repaired several distinct loss/order/retry cases. September 10 added deterministic folding and exact checkpoint sources. September 11 then added permanent pinning plus fresh admission, foreground waiting and deferred resumption in the same Nanobot commit, and hybrid persistence plus eager release in Higgs nightly. The latest findings concentrate in that last layer of policy.

These are valuable foundations to retain: durable original messages, atomic tool-call/result groups, typed capacity failures, checkpoint publication before use, and exact recall of covered source rows. The review does not justify reverting the month's reliability work wholesale.

The implementation has nonetheless grown substantially. Including embedded tests, Nanobot `shared.rs` grew from 3,513 to 8,386 lines over the month; `lcm.rs` from 5,093 to 6,377; `agent_core.rs` from 1,296 to 3,362. These counts are a maintenance signal, not proof that individual checks are unnecessary. The stronger evidence is that new state escaped the existing transaction, and an optimization can now defeat recovery.

Read-only SQLite evidence covers 8,073 messages in 195 sessions since Aug 12 and 1,631 messages in 39 sessions since Sep 5. Last week contains 57 model-failure events: 29 typed capacity errors, 9 rendered-prompt limit errors, 10 cancellations, and 9 local transport errors. These include development and stress tests and are **not production failed-turn rates**.

One independently checked recent chain: session `20260911_153952_c40bf8`, events 18470–18474. On Sep 11 at 19:52:16 UTC Higgs rejected **40,989 rendered tokens against 40,960 allowed**. A retry began immediately and ended at 19:55:20 with the turn marked cancelled. This postdates the latest commit timestamps but the DB does not establish the installed binary, nor that cancellation was caused by a particular bug.

The stored evaluation reports generally disclose their limitations honestly:

| Evidence | What it demonstrates | What it does not establish |
|---|---|---|
| [39K incident run](/Users/peppi/Dev/nanobot-rs/experiments/capacity-incident/RESULTS.md) | 10 user turns, 18 calls, exact recall through 39,315 rendered tokens | Zero compactions; no exhaustion/compaction endurance proof |
| [Announcement comparison](/Users/peppi/Dev/nanobot-rs/experiments/context-recovery/ANNOUNCEMENT-RESULTS.md) | Both arms reached six correct submissions | Both suspended before the 20-update target |
| [Scheduled reset control](/Users/peppi/Dev/nanobot-rs/experiments/context-recovery/FEASIBILITY-RESULTS.md) | 20 updates executed, 13 exact snapshots | Correct recovery on every boundary; the first checksum error happened despite correct notes being supplied |
| [Commit validation](/Users/peppi/Dev/nanobot-rs/experiments/capacity-incident/COMMIT-VALIDATION.md) | Release regressions passed for that revision | Explicitly no new live speed/endurance claim |

The Codex task “Compare context management” also claimed broad unit-suite and adversarial-review success. Such claims were treated as investigation leads. The new reproductions show why they cannot substitute for boundary-specific tests.

The Higgs September 11 pressure-policy change (`3d455f046`) was also checked. It removes pressure-triggered cancellation, but request admission still enforces the current token ceiling and byte ledger. No concrete admission bypass was established; this review does not classify that deliberate policy change as a bug.

The expanded nightly review also covered August retained-session continuation/leases and disk-store integration, September 2–3 admission revalidation, September 6 retained-memory accounting, September 7 prefill progress/error propagation, and September 8 stale-session rejection. No additional current defect was established in the retained-accounting or stale-prefix validation paths. Historical faults that these changes corrected are not counted as outstanding findings.

## Simplification I would make

1. **Keep SQLite authoritative and use one active context projection.** Source rows and the current assistant/tool transaction remain durable; summaries/checkpoints are replaceable projections. Put every field affecting that projection inside its existing publication transaction.
2. **Give Higgs ownership of physical capacity and exact rendered admission.** Nanobot needs a conservative estimate to schedule work, but estimates, server rejection, and retry ceilings must converge on one request envelope. Avoid independently evolving pressure rules that can disagree about whether another attempt is useful.
3. **Make deterministic folding the bounded recovery operation.** It must fit the requested target or identify the irreducible current transaction. An old prefix pin must yield. Optional semantic refinement should not be required for urgent reclamation.
4. **Treat KV reuse as an optimization with measured coverage.** Rotation does not by itself prove zero reuse: independent radix/disk checkpoints and expansion leases exist. However, ordinary retained-session AR/MTP generation disables prefix-cache publication at [simple.rs:7430](/Users/peppi/Dev/higgs/crates/higgs-engine/src/simple.rs:7430). A fresh session can consume an existing prefix, but preserving an arbitrary head does not create the matching recurrent checkpoint. Require measured hit tokens and prefill cost before trading reclaimable context for a pinned head.
5. **Use one explicit retained-session retirement contract.** Identify the old model/session, make busy versus released observable, and acknowledge completion where memory admission depends on it. Avoid two best-effort release mechanisms silently disagreeing.

Start with the six concrete fixes, including consolidation of the disk-cache configuration, then test one bounded transition: full window → deeper fold → durable checkpoint → old-session release → smaller admitted request → restart → exact recall. Add capacity collapse, publication failure, in-flight release, and realistic hybrid-state restoration as adversarial cases. Only after this passes should matched latency measurements decide whether permanent-looking cache optimizations are worthwhile.

## Verification and limits

- Current Nanobot `cargo build --release`: passed.
- Current Nanobot `cargo test --release --lib`: **3,005 passed, 0 failed, 27 ignored**.
- Current Nanobot protocol/LCM integration suites: **40 passed** (10 LCM E2E, 6 protocol-invariant, 24 protocol tests).
- Executed both standalone reproductions against the release library; the client URL test used an isolated localhost mock.
- First-fold LCM restart equality passed in a standalone probe with pinning enabled and disabled. This does not cover rejected publication or all repeated-fold cases.
- Higgs targeted existing release binaries: two hybrid snapshot tests and the session-lock waiting test passed. No new full Higgs build, full suite, live-model restart, or matched performance benchmark was run.
- GitNexus CLI was consulted; graph coverage was stale/incomplete for the newest symbols. Current source and git diffs determined findings. No production source edits or commits were made.
- SQLite was opened with `mode=ro&immutable=1` because normal readonly access needed journal permissions. Counts describe that retained database view; uncheckpointed WAL data may be absent. No database writes or service changes were made. Raw trials were not all independently rescored.
- Existing dirty work was left alone. Added only this review and its standalone reproducers.

Reproducer build pattern after `cargo build --release` (select the matching release dependency hashes from the successful build):

```sh
rustc --edition=2021 pin_repro.rs -L dependency=/Users/peppi/Dev/nanobot-rs/target/release/deps --extern nanobot=/path/to/matching/libnanobot-HASH.rlib -o /tmp/pin_repro
rustc --edition=2021 drop_repro.rs -L dependency=/Users/peppi/Dev/nanobot-rs/target/release/deps --extern nanobot=/path/to/matching/libnanobot-HASH.rlib --extern tokio=/path/to/matching/libtokio-HASH.rlib -o /tmp/drop_repro
```

The current machine additionally needed the clang-21 library search argument already recorded in the repository's `.cargo/config.toml` when linking the standalone probes.
