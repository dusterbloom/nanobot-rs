# Higgs Adaptive Capacity Enforcement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Higgs publish and enforce a measured, process-wide memory-safe capacity for every local generation route so selecting EschaMoE cannot knowingly drive the Mac into compression or swap thrashing.

**Architecture:** Add one pure byte-domain capacity controller to the Higgs server process, backed by narrow engine memory facts and a macOS pressure observer. Every loaded engine, retained cache, and active request participates in the same ledger. Admission occurs after exact tokenization and before inference starts; a FIFO reservation moves into the actual worker and remains held until allocation has stopped. The controller publishes the same snapshot used by admission through `/v1/capacity` and typed OpenAI-shaped errors.

**Tech Stack:** Rust 2021, Axum, Tokio, MLX/Metal, `dispatch2` on macOS, serde JSON, UUID boot IDs, existing Higgs metrics and cache APIs, release-only Cargo validation.

## Global Constraints

1. Work in an isolated Higgs worktree. Do not alter `/Users/peppi/Dev/higgs` or its dirty `.omen/search.db`.
2. Read the Higgs `AGENTS.md` and applicable skills before editing. Use test-driven development for each behavior change.
3. Before editing any function, method, or type, run GitNexus upstream impact for that symbol. Report HIGH or CRITICAL blast radius before proceeding. Known CRITICAL surfaces are `build_engine`, `chat_completions_stream`, and the session streaming wrapper.
4. Keep one production inference path. Do not add a safety-off mode, alternate router, expiring reservation, or token-only shadow heuristic.
5. All arithmetic entering admission is checked `u64` byte arithmetic. Missing facts, overflow, corrupt profiles, and critical pressure fail conservatively.
6. Run only release builds/tests. Long-running replay or server processes belong in named `tmux` sessions.
7. Before every commit, run `detect_changes({scope: "compare", base_ref: "main"})`, inspect the exact diff, and stage only the task-owned files.

---

### Task 1: Freeze the v1 wire contract with red tests

**Owner:** routine worker; Higgs server tests only.

**Files:**

- Add: `crates/higgs/src/capacity.rs`
- Modify: `crates/higgs/src/lib.rs`
- Modify: `crates/higgs/src/routes/mod.rs`
- Modify: the existing route test module that constructs `AppState`

- [ ] **Step 1: Run GitNexus impact analysis.**

  Analyze `AppState`, the root router constructor, `INFRASTRUCTURE_PATHS`, `ServerError`, and every test helper whose signature will change. Warn before editing any HIGH or CRITICAL result.

- [ ] **Step 2: Add serialization-only capacity contract types.**

  Define closed enums and camelCase serde records for schema version 1: `CapacityAvailability`, `MemoryPressure`, `CapacityBasis`, `CapacitySnapshot`, `CapacityExceededError`, `CapacityUnavailableError`, and `CapacityInterruptedError`. Include `modelFingerprint`, `bootId`, and `generation` exactly as specified in the approved design.

- [ ] **Step 3: Add failing JSON shape tests.**

  Assert the exact successful snapshot and exact OpenAI-shaped 413/503 bodies. Assert unknown enum values fail deserialization in the client-facing compatibility fixture. Assert generation equality is meaningful only with the same boot ID.

- [ ] **Step 4: Reserve the route and metrics identity.**

  Add `/v1/capacity` to `INFRASTRUCTURE_PATHS`. Freeze distinct fixtures for: 200 available; 200 unavailable for a known-but-unloaded model; typed 404 `higgs_capacity_model_not_found` for an unknown model; and generic route absence used only by legacy clients. Do not add policy yet.

- [ ] **Step 5: Run the focused release tests.**

  Run: `cargo test --release -p higgs capacity -- --nocapture`

  Expected: wire tests pass; endpoint-availability tests remain red until Task 5.

- [ ] **Step 6: Commit the contract seam.**

  Commit: `feat(capacity): define higgs wire contract`

---

### Task 2: Expose narrow engine memory and cost facts

**Owner:** routine worker; owns only `higgs-engine` memory fact APIs and tests.

**Files:**

- Modify: `crates/higgs-engine/src/mlx_tuning.rs`
- Modify: `crates/higgs-engine/src/simple.rs`
- Modify: `crates/higgs-engine/src/lib.rs`
- Modify: relevant engine test modules

- [ ] **Step 1: Run GitNexus impact analysis.**

  Analyze `model_weight_bytes`, `pflash_free_memory_mb`, `set_wired_limit_to_max`, `CacheStats`, and the public engine trait/type used by all local engines. Keep the public seam smaller if exposing the main engine trait is HIGH or CRITICAL.

- [ ] **Step 2: Add failing tests for injected measurements.**

  Cover measured MLX active bytes, MLX memory limit, Metal recommended working set, retained/radix resident bytes, and checked overflow. Include an Escha geometry fixture proving the dense KV slope is 20,480 bytes/token and 49,152 tokens consume 960 MiB.

- [ ] **Step 3: Add narrow immutable fact records.**

  Introduce `MlxMemorySnapshot`, `ModelFootprint`, and an engine cost description containing fixed live-session bytes, persistent bytes per token, decode workspace, and a bounded transient-prefill estimate. Reuse existing metadata and cache `estimated_bytes`; do not expose the private loader metadata wholesale.

- [ ] **Step 4: Add request high-water sampling.**

  Provide a content-free before/after measurement primitive that can record active/peak allocation around prefill and decode. Keep platform calls behind an injectable probe for deterministic tests.

- [ ] **Step 5: Run release validation.**

  Run: `cargo test --release -p higgs-engine memory -- --nocapture`

  Run: `cargo build --release -p higgs-engine`

- [ ] **Step 6: Commit the measurement seam.**

  Commit: `feat(engine): expose capacity memory facts`

---

### Task 3: Implement the pure byte-domain controller and profile store

**Owner:** routine worker; owns `capacity.rs` policy and its tests.

**Files:**

- Modify: `crates/higgs/src/capacity.rs`
- Modify: `crates/higgs/Cargo.toml`
- Modify: `Cargo.lock`

- [ ] **Step 1: Run GitNexus impact analysis for the new call sites.**

  Analyze the state/config constructors that will supply controller inputs. Make `dispatch2` a direct target-specific macOS dependency only when Task 4 begins using it.

- [ ] **Step 2: Write failing pure-controller tests.**

  Cover: smaller nonzero MLX/Metal limit; protected reserve `max(4 GiB, 20%)`; numeric config as ceilings; prompt/output/transient accounting in one ledger; 1024-token rounding; 32/64 GiB Escha examples; warning `min(recomputed, 75%)`; critical `min(recomputed, 50%)`; allocator peak `+10%`; corrupt/mismatched fingerprint fallback; checked-arithmetic failure; three observations over five continuous minutes; one-step rise bounded by 4096 tokens or 12.5%; and cache hits never lowering cold-prefill cost.

- [ ] **Step 3: Implement immutable inputs and one solver.**

  Define `CapacityInputs`, `RequestCost`, `ByteLedger`, `CapacityDecision`, and `CapacityController`. Derive published token limits by searching the same byte inequality used by admission. Do not duplicate token math in the route.

- [ ] **Step 4: Implement content-free learning.**

  Tag observations `cold`, `retained_suffix`, or `radix_hit`; maintain conservative per-band high-water coefficients; freeze rises under pressure; downshift immediately. Use a clock trait in tests.

- [ ] **Step 5: Implement atomic profile persistence.**

  Key by hardware/OS/Metal/build/model/quantization/execution/KV/drafter fingerprint. Persist cost evidence and baselines, never live capacity. Write temp file, sync, then atomic rename. A new random boot ID is created on every process start.

- [ ] **Step 6: Run focused and crate tests.**

  Run: `cargo test --release -p higgs capacity:: -- --nocapture`

  Run: `cargo test --release -p higgs --lib`

- [ ] **Step 7: Commit the controller.**

  Commit: `feat(capacity): add adaptive byte controller`

---

### Task 4: Observe macOS pressure and apply deterministic downshifts

**Owner:** routine worker; owns platform observer and controller notification tests.

**Files:**

- Modify: `crates/higgs/src/capacity.rs`
- Modify: `crates/higgs/Cargo.toml`
- Modify: `Cargo.lock`
- Modify: `crates/higgs/src/main.rs`

- [ ] **Step 1: Run impact analysis.**

  Analyze server startup/shutdown and the controller methods receiving observations.

- [ ] **Step 2: Add failing observer tests behind an injected source.**

  Prove warning, critical, return-to-normal, swap-out delta, compressor delta, and shutdown are delivered once and do not block the runtime.

- [ ] **Step 3: Add the production macOS source.**

  Use `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` through `dispatch2`; sample VM swap/compressor counters as deltas. Treat a new swap-out as critical for new admission until normal pressure plus one minute without another swap-out.

- [ ] **Step 4: Start and stop one observer with the server.**

  Feed the process-wide controller. No per-model observers and no config rewrites.

- [ ] **Step 5: Validate.**

  Run: `cargo test --release -p higgs capacity::pressure -- --nocapture`

  Run: `cargo build --release -p higgs`

- [ ] **Step 6: Commit.**

  Commit: `feat(capacity): track mac memory pressure`

---

### Task 5: Bind capacity to model lifecycle and expose `/v1/capacity`

**Owner:** deep worker because `build_engine` is CRITICAL.

**Files:**

- Modify: `crates/higgs/src/state.rs`
- Modify: `crates/higgs/src/routes/models.rs`
- Modify: `crates/higgs/src/routes/mod.rs`
- Add or modify: `crates/higgs/src/routes/capacity.rs`
- Modify: all existing `AppState` test fixtures

- [ ] **Step 1: Run and report impact analysis before edits.**

  Analyze `build_engine`, `AppState`, boot loading, runtime `load_model`, unload/switch, and router construction. Explicitly report the CRITICAL `build_engine` blast radius before proceeding.

- [ ] **Step 2: Add failing lifecycle tests.**

  Cover conservative snapshot immediately after load, exact content fingerprint, random boot ID, runtime load/unload registration, pre-load definite-too-large rejection, and post-load minimum-working-request rejection. A known-but-unloaded model must expose 200 unavailable; an unknown name must produce the typed model-not-found 404.

- [ ] **Step 3: Add one process-wide controller to `AppState`.**

  Initialize it once, register model facts after load, and expose a lifecycle seam that Task 6 will bind to reservation drain before final unregister. Provide a test constructor so fixtures do not clone policy setup.

- [ ] **Step 4: Bound model loading.**

  Apply the loader workspace estimate and pressure checks at known shard/conversion boundaries. Warning disables optional prefetch; critical aborts and releases partially loaded state. Replace estimates with measured MLX residency after load.

- [ ] **Step 5: Finish `/v1/capacity`.**

  Require chat-equivalent authentication, return 404 for unknown/unloaded model, and serialize the controller’s current snapshot without recomputing policy in the route.

- [ ] **Step 6: Validate.**

  Run: `cargo test --release -p higgs capacity -- --nocapture`

  Run: `cargo test --release -p higgs routes::models -- --nocapture`

- [ ] **Step 7: Commit.**

  Commit: `feat(capacity): bind model lifecycle`

---

### Task 6: Add FIFO admission and worker-owned reservations to every generation route

**Owner:** deep worker; owns admission, reservation lifetime, and local route integration.

**Files:**

- Modify: `crates/higgs/src/capacity.rs`
- Modify: `crates/higgs/src/routes/chat.rs`
- Modify: `crates/higgs/src/routes/completions.rs`
- Modify: `crates/higgs/src/routes/anthropic.rs`
- Modify: `crates/higgs/src/error.rs`
- Modify: route integration tests

- [ ] **Step 1: Run and report CRITICAL impacts.**

  Analyze `chat_completions_stream`, non-streaming chat, completions, messages, `spawn_blocking` worker helpers, and the session streaming wrapper. Preserve the current continuation handshake ordering.

- [ ] **Step 2: Add failing contention and rejection tests.**

  With a tiny injected envelope, prove cache reclamation precedes rejection; output reserve is charged; over-budget requests receive exact 413 before model allocation; unavailable receives exact 503 with `retryAfterMs: 5000`; two concurrent requests cannot reserve the same bytes; individually safe contention queues FIFO; queued cancellation removes a waiter; dequeue revalidates boot ID, generation, pressure, and cost.

- [ ] **Step 3: Implement process-wide cancellable FIFO admission.**

  Reserve the larger static/learned request peak. Charge unaccounted positive MLX active bytes. Evict optional radix then eligible unleased retained state before rejection. Config values remain ceilings.

- [ ] **Step 4: Move the RAII guard into the inference worker.**

  Create admission after exact tokenization and before `spawn_blocking`, but transfer ownership into the blocking worker. Release only after success, engine error, unwind, or acknowledged cancellation has stopped allocation. Never use a TTL.

- [ ] **Step 5: Bind unload to the reservation registry.**

  Prove model unload waits for active worker reservations to drain, or signals cancellation and joins those workers, before unregistering capacity state or releasing weights. This is deliberately sequenced here because Task 5 only established the lifecycle seam.

- [ ] **Step 6: Apply the same helper to all three local APIs.**

  Route chat completions, legacy completions, and Anthropic-compatible messages through the same controller decision and typed error mapping.

- [ ] **Step 7: Validate focused concurrency behavior.**

  Run: `cargo test --release -p higgs capacity_admission -- --nocapture --test-threads=1`

  Run: `cargo test --release -p higgs routes -- --nocapture`

- [ ] **Step 8: Commit.**

  Commit: `feat(capacity): enforce request admission`

---

### Task 7: Make disconnect, timeout, and critical pressure stop live allocation safely

**Owner:** deep worker; crosses server, engine, and model chunk loops.

**Files:**

- Modify: `crates/higgs-engine/src/simple.rs`
- Modify: `crates/higgs-engine/src/batch_engine.rs`
- Modify: `crates/higgs-models/src/progress.rs`
- Modify: `crates/higgs-models/src/lib.rs`
- Modify: `crates/higgs-models/src/qwen3_next.rs`
- Modify: `crates/higgs/src/routes/chat.rs`
- Modify: cancellation/stream tests

- [ ] **Step 1: Run impact analysis for every changed loop and callback.**

  Analyze the progress sink, the two real prefill chunk loops, decode loops, pending batch-prefill advancement, session streaming wrapper, and terminal SSE mapping. Warn on CRITICAL results.

- [ ] **Step 2: Add red lifecycle tests.**

  Prove queued disconnect, mid-prefill disconnect, mid-decode timeout, engine error, panic/unwind, and a no-progress watchdog cannot leak or prematurely release a live reservation. Add the missing `response_tx.is_closed()` coverage before batch pending-prefill start/advance/decode.

- [ ] **Step 3: Add a shared cancellation observation.**

  Extend the existing thread-local prefill progress mechanism or an equally narrow engine callback so bounded prefill chunks and decode steps can observe client cancellation, timeout, watchdog, and critical pressure. Do not poll platform state inside model kernels.

- [ ] **Step 4: Join before release.**

  The HTTP task signals cancellation; the inference worker acknowledges it at a safe boundary, stops allocating, returns, and only then drops its reservation. Model unload follows the same join rule.

- [ ] **Step 5: Emit the exact terminal capacity SSE.**

  After streaming begins, emit `higgs_capacity_interrupted` with boot ID, generation, and partial-output token count, then the normal `[DONE]`. Do not map it to generic `generation_error`.

- [ ] **Step 6: Validate.**

  Run: `cargo test --release -p higgs-engine cancellation -- --nocapture --test-threads=1`

  Run: `cargo test --release -p higgs capacity_interrupted -- --nocapture --test-threads=1`

  Run: `cargo test --release -p higgs-models prefill -- --nocapture`

- [ ] **Step 7: Commit.**

  Commit: `fix(capacity): cancel inference before release`

---

### Task 8: Add observability, doctor output, and release validation

**Owner:** routine worker for observability; root agent owns final hardware replay.

**Files:**

- Modify: `crates/higgs/src/metrics.rs`
- Modify: `crates/higgs/src/routes/metrics.rs`
- Modify: `crates/higgs/src/doctor.rs`
- Modify: relevant snapshot tests and documentation

- [ ] **Step 1: Run impact analysis.**

  Analyze metrics registration/rendering and doctor model checks before edits.

- [ ] **Step 2: Add red observability tests.**

  Require boot ID/generation, effective byte/token limits, pressure, basis, downshifts, rejections, active reservation bytes/count/oldest age, queued waiters, eviction, watchdog/cancellation, peak MLX allocation, and swap/compressor deltas. Metrics contain no prompt content.

- [ ] **Step 3: Add structured transitions and doctor checks.**

  Emit one record per capacity transition with old/new envelopes and cause. Replace the Escha RAM-only warning with measured Metal/MLX facts and validate configured ceilings.

- [ ] **Step 4: Run full release validation.**

  Run: `cargo fmt --all -- --check`

  Run: `cargo test --release`

  Run: `cargo build --release`

  Run the repository’s Higgs doctor command against the Escha configuration.

- [ ] **Step 5: Run real-hardware replay in `tmux`.**

  Record Metal limit, model residency, fixed-state cost, transient bound, cache allocation, and published limit. Replay progressively larger genuine sessions, including warm tool turns plus three genuine cold boundary-band starts. Gate on zero new swap-outs and no sustained warning/critical pressure. A cache-only control must not lower cold-prefill cost.

- [ ] **Step 6: Review and commit.**

  Run GitNexus `detect_changes` against `main`, request an independent diff review, resolve all correctness findings, and commit: `feat(capacity): expose adaptive diagnostics`

## Higgs Completion Gate

- `/v1/capacity` and request admission are two views of the same byte solver.
- All local generation routes enforce it.
- Contention waits FIFO; semantic oversize returns one typed 413; pressure unavailability returns typed 503.
- No reservation can expire while inference still allocates.
- Real Escha replay causes no new swap-outs and the conservative limit can rise only from qualifying real observations.
- `cargo test --release` and `cargo build --release` pass from a clean worktree.
