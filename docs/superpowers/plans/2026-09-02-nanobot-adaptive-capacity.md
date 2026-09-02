# Nanobot Adaptive Capacity and Durable Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Nanobot consume Higgs's measured live capacity, compact before unsafe local calls, and recover one typed rejection without rewriting configuration, losing the user turn, or duplicating tool effects.

**Architecture:** Add a typed Higgs capacity client and one live runtime snapshot beside the existing immutable configured `TokenBudget`. The agent loop resolves the effective budget immediately before each local provider request, installs an existing durable LCM checkpoint or performs capacity-safe reduction, and reissues only the rejected provider call at most once. All prompt rewrites, retained-session rotation, model-call journaling, tool boundaries, and incomplete streaming output continue through the existing hot path.

**Tech Stack:** Rust 2021, Tokio, reqwest, serde JSON, existing OpenAI-compatible provider, SQLite session/replay journal, LCM compaction, TUI control markers, release-only Cargo validation.

## Global Constraints

1. Work in an isolated Nanobot worktree and preserve unrelated changes in `AGENTS.md`, `CLAUDE.md`, `PLAN.md`, and the durable-subagent plan.
2. Read the repository `AGENTS.md` and applicable skills before editing. Use test-driven development for every behavior change.
3. Before editing any symbol, run GitNexus upstream impact and report HIGH or CRITICAL results. Known CRITICAL surfaces include `build_swappable_core` and `apply_emergency_trim`; avoid mutating or rebuilding the immutable core.
4. Capacity behavior applies only to providers that identify as Higgs-compatible. Cloud providers and LM Studio keep their existing behavior.
5. User configuration is a ceiling. Never persist learned/effective capacity into `config.json`.
6. A capacity retry reissues only the current provider request before response processing reaches tool execution. It never restarts `process_message`, the outer user turn, a tool carrier, or a tool implementation.
7. Every prompt rewrite uses `TurnContext::rewrite_committed`, the existing LCM install path, and prompt-cache/retained-session invalidation. No silent truncation or parallel compaction pipeline.
8. Run only release builds/tests. Use `tmux` for persistent servers and replays.
9. Before every commit, run GitNexus `detect_changes({scope: "compare", base_ref: "main"})`, inspect the exact diff, and stage only task-owned files.

## Cross-Repository Prerequisite

Freeze schema version 1 before production integration. Nanobot fixtures must match Higgs exactly:

- `GET /v1/capacity?model=...` with `schemaVersion`, `modelFingerprint`, `bootId`, `generation`, availability, pressure, basis, and safe token fields;
- HTTP 413 `higgs_capacity_exceeded` / `compact_and_retry`;
- HTTP 503 `higgs_capacity_unavailable` / `capacity_unavailable` with schema-v1 `retryAfterMs: 5000`;
- terminal SSE `higgs_capacity_interrupted`, followed by `[DONE]`.

The fixture also distinguishes 200 unavailable for a known-but-unloaded model,
typed 404 `higgs_capacity_model_not_found` for an unknown model, and a generic
route-not-found response from an older Higgs. Only the last case selects the
legacy 16K/4K envelope after ordinary Higgs discovery has succeeded.

Mock-based Nanobot work may proceed in parallel with Higgs after this fixture freezes. The real HTTP replay waits for both implementations.

---

### Task 1: Define the typed contract and pure effective-budget policy

**Owner:** routine worker; owns `agent/capacity.rs` and token-budget helpers.

**Files:**

- Add: `src/agent/capacity.rs`
- Modify: `src/agent/mod.rs`
- Modify: `src/agent/token_budget.rs`
- Modify: focused unit tests in those modules

- [ ] **Step 1: Run GitNexus impact analysis.**

  Analyze `TokenBudget`, its constructors, and `build_swappable_core`. Report the known CRITICAL core-builder blast radius, but do not change it in this task.

- [ ] **Step 2: Add failing schema and policy tests.**

  Cover schema mismatch, empty fingerprint/boot ID, invalid field relationships, checked overflow, unavailable capacity, config ceilings that cannot raise server limits, same generation with a new boot ID, and the legacy Higgs fallback of 16,384 total tokens with at most 4096 output tokens.

- [ ] **Step 3: Add closed contract types.**

  Define `HiggsCapacityProfile`, `CapacityAvailability`, `CapacityPressure`, `CapacityBasis`, and `EffectiveCapacity` with camelCase serde. Validate schema version 1 and all numeric relationships at construction.

- [ ] **Step 4: Add one pure effective-budget conversion.**

  Compute total as the minimum of the Higgs snapshot and configured context ceiling. Compute output as the minimum of recommended output, configured output, and room after the immutable prefix. `prompt_room(planned_output, protocol_overhead)` uses checked arithmetic. Add a narrow `TokenBudget` constructor/view if required; do not make the configured object mutable.

- [ ] **Step 5: Validate.**

  Run: `cargo test --release capacity --lib -- --nocapture`

  Run: `cargo test --release token_budget --lib -- --nocapture`

- [ ] **Step 6: Commit.**

  Commit: `feat(capacity): add typed live budget policy`

---

### Task 2: Fetch capacity and classify typed HTTP/SSE failures

**Owner:** routine worker; owns Higgs client/provider error mapping.

**Files:**

- Modify: `src/higgs.rs`
- Modify: `src/providers/openai_compat.rs`
- Modify: `src/errors.rs`
- Modify: provider and mock-HTTP tests

- [ ] **Step 1: Run impact analysis.**

  Analyze `OpenAICompatProvider`, `map_status_to_provider_error`, `ProviderError`, the streaming SSE parser, and `supports_higgs_session_cache`.

- [ ] **Step 2: Add red mock-server tests.**

  Prove base URL normalization, auth parity, model query encoding, known-but-unloaded 200, typed unknown-model 404, generic route-absent old-Higgs fallback, malformed profile rejection, exact 413 classification, exact 503 classification including `retryAfterMs: 5000`, unrelated/malformed 413 preserving the old generic behavior, and terminal capacity SSE never producing a successful `LLMResponse`.

- [ ] **Step 3: Implement the bounded client.**

  Fetch capacity only for Higgs-compatible providers using the existing authenticated `reqwest::Client`. Do not add a generic capacity method to all providers unless the existing trait makes a narrow capability impossible.

- [ ] **Step 4: Add structural error variants.**

  Add `ProviderError::HiggsCapacityExceeded`, `HiggsCapacityUnavailable`, and `HiggsCapacityInterrupted` carrying integer fields plus boot ID/generation. Parse typed JSON only when status, `type`, and `code` all match; never recover by parsing display strings.

- [ ] **Step 5: Preserve incomplete stream bytes privately.**

  On typed terminal SSE, retain accumulated bytes for the replay journal but return an error rather than `Done` or a successful assistant result.

- [ ] **Step 6: Validate and commit.**

  Run: `cargo test --release higgs --lib -- --nocapture`

  Run: `cargo test --release openai_compat --lib -- --nocapture`

  Commit: `feat(higgs): consume capacity contract`

---

### Task 3: Install live capacity without rebuilding the core or rewriting config

**Owner:** deep worker because the immutable core builder is CRITICAL.

**Files:**

- Modify: `src/agent/agent_core.rs`
- Modify: `src/cli/core_builder.rs`
- Modify: `src/agent/agent_loop/shared.rs`
- Modify: `src/repl/commands/lifecycle.rs`
- Modify: focused core/loop tests

- [ ] **Step 1: Run and report impact analysis.**

  Analyze `build_swappable_core`, `RuntimeCounters`, `step_pre_call`, model-switch lifecycle, and retained-session reset. Explicitly warn before touching the CRITICAL core builder.

- [ ] **Step 2: Add failing runtime-state tests.**

  Cover initial discovery, unchanged tuple avoiding a refetch, refresh before each main/tool-continuation/compaction request, boot-ID/model switch invalidation, retained epoch rotation, cloud provider bypass, and old Higgs selecting the conservative fallback. Assert the config file remains byte-for-byte unchanged.

- [ ] **Step 3: Add one live snapshot holder.**

  Place an `Arc<CapacityRuntime>` beside `RuntimeCounters` in shared loop state. Key the installed snapshot by endpoint, schema version, model fingerprint, boot ID, and generation. Keep `SwappableCore.token_budget` immutable and derive the effective request budget at use time.

- [ ] **Step 4: Refresh at safe boundaries.**

  Refresh after endpoint/model discovery, after model switch/restart, before every Higgs provider request, and after a typed rejection. A boot-ID change queues the old retained ID for drop through the existing reservation/epoch path.

- [ ] **Step 5: Validate and commit.**

  Run: `cargo test --release capacity_runtime --lib -- --nocapture`

  Run: `cargo test --release build_core --lib -- --nocapture`

  Commit: `feat(agent): adopt live Higgs capacity`

---

### Task 4: Compact proactively without making an oversized compactor request

**Owner:** deep worker; sole owner of `agent_loop/shared.rs` during this task.

**Files:**

- Modify: `src/agent/agent_loop/shared.rs`
- Modify: `src/agent/agent_loop/budget.rs`
- Modify: `src/agent/lcm.rs` only if the existing deterministic level-3 reducer needs a narrow exposure
- Modify: `src/agent/agent_loop/tests.rs`

- [ ] **Step 1: Run and report impact analysis.**

  Analyze `step_pre_call`, `manage_compaction`, `install_pending_compaction`, the deterministic LCM reducer, and `TurnContext::rewrite_committed`. Verify that the capacity path does not modify or call the CRITICAL `apply_emergency_trim` symbol.

- [ ] **Step 2: Add a typed turn-local recovery state.**

  Freeze an enum such as `Idle`, `PreflightCompacted { generation }`, `RetryIssued { generation }`, and `Terminal { generation }`; Task 4 exercises only its preflight states and Task 5 owns retry/terminal transitions. Do not overload a generic retry counter or add a behavior-selecting boolean.

- [ ] **Step 3: Add failing preflight tests.**

  Prove: a completed pending LCM checkpoint installs first; an over-budget turn compacts before the first provider call; model-authored summary runs only if its own request fits; otherwise deterministic level-3 reduction makes zero provider calls; all durable source message IDs remain recallable; retained prompt epoch rotates once; and an immutable prefix that still cannot fit becomes pending/unavailable rather than recursively compacting.

- [ ] **Step 4: Implement the sanctioned reduction order.**

  In `step_pre_call`, snapshot capacity after frozen tool definitions are known, then: install a covering checkpoint; otherwise use ordinary LCM only when it fits; otherwise use the existing local deterministic level-3 reduction with durable source handles. If that still cannot fit the immutable request, leave the turn pending/unavailable. Do not call `apply_emergency_trim` from this capacity path unless a later spec explicitly proves it produces equivalent durable source markers.

- [ ] **Step 5: Invalidate caches through the existing path.**

  Every rewrite goes through `rewrite_committed` or LCM install plus `invalidate_prompt_cache_for_rewrite`, ensuring the Higgs request reservation drops before the new request and the old retained session is queued for deletion.

- [ ] **Step 6: Validate and commit.**

  Run: `cargo test --release capacity_preflight --lib -- --nocapture`

  Run: `cargo test --release hard_lcm_checkpoint --lib -- --nocapture`

  Run: `cargo test --release prompt_cache --lib -- --nocapture`

  Commit: `feat(agent): compact before unsafe prefill`

---

### Task 5: Recover exactly one 413 without replaying the user turn or tools

**Owner:** deep worker; continues sole ownership of agent-loop recovery code.

**Files:**

- Modify: `src/agent/agent_loop/shared.rs`
- Modify: `src/agent/agent_loop/response.rs` only if typed failures currently pass into response processing
- Modify: `src/agent/agent_loop/tests.rs`

- [ ] **Step 1: Run impact analysis.**

  Analyze `step_call_llm`, `attempt_overflow_recovery`, provider-error handling, `step_process_response`, and `step_execute_tools`.

- [ ] **Step 2: Add event-order replay tests.**

  Require `ModelRequest -> ModelFailed(413) -> ModelRequest -> ModelResponse -> TurnFinished` under one logical turn ID; exactly one user row; a rotated retained epoch between requests; zero tool-preexecute/tool-execute entries before the rejection; an earlier committed tool from a prior iteration remains exactly once; a second 413 produces no third request and leaves the turn visibly pending.

- [ ] **Step 3: Implement a dedicated typed branch.**

  On `HiggsCapacityExceeded`, journal the failed model call, refresh capacity, compact to the typed safe prompt budget, let `HiggsSessionRequestReservation` drop, and re-enter only the pre-call phase. Do not call `process_message`, append the user message, invoke generic retry, or rerun the outer loop.

- [ ] **Step 4: Enforce the one-retry boundary.**

  Transition `Idle/PreflightCompacted -> RetryIssued` once. A second typed rejection becomes terminal/pending with current safe values. Existing generic context-length recovery remains independent and unchanged.

- [ ] **Step 5: Validate and commit.**

  Run: `cargo test --release capacity_exceeded --lib -- --nocapture`

  Run: `cargo test --release exact_replay --lib -- --nocapture`

  Run: `cargo test --release tool_lifecycle --lib -- --nocapture`

  Commit: `fix(agent): retry one capacity rejection safely`

---

### Task 6: Persist unavailable and interrupted turns without pretending success

**Owner:** deep worker; starts only after Task 5 is merged and then owns replay plus loop integration exclusively.

**Files:**

- Modify: `src/session/db.rs`
- Modify: replay event/recorder source under `src/session/` or `src/agent/`
- Modify: `src/agent/agent_loop/shared.rs`
- Modify: `src/agent/agent_loop/response.rs`
- Modify: relevant DB/replay/stream tests

- [ ] **Step 1: Run impact analysis.**

  Analyze `SessionEventPayload`, `TurnReplayRecorder`, `RecordedModelCall`, provider-error handling, and the streaming response fold.

- [ ] **Step 2: Add red durability tests.**

  Prove partial SSE bytes persist as an incomplete artifact, no assistant success row is created, replay remains incomplete, transient TUI output retracts, unavailable retains the user turn as resumable pending work, backoff is 5 seconds up to 30 seconds, cancellation stops polling, and a static unavailable profile creates no busy-loop model requests.

- [ ] **Step 3: Add an explicit incomplete model event.**

  Extend the existing journal with a typed interrupted record containing call ID, error digest, partial-output artifact digest, and token count. Do not hide bytes in a display string. Fold it as incomplete, never successful.

- [ ] **Step 4: Defer unavailable work durably.**

  Use a dedicated resumable event/status if existing `TurnFinished` semantics are terminal. Poll only from the outer scheduler/REPL lifecycle, without holding a tool loop or DB transaction. Honor schema-v1 `retryAfterMs: 5000`; clamp compatible future values to 5–30 seconds and back off exponentially. Resume automatically after an available snapshot fits the minimum request unless the user cancels.

- [ ] **Step 5: Validate and commit.**

  Run: `cargo test --release interrupted --lib -- --nocapture`

  Run: `cargo test --release replay --lib -- --nocapture`

  Run: `cargo test --release chat_stream --lib -- --nocapture`

  Commit: `feat(replay): persist capacity-interrupted turns`

---

### Task 7: Explain configured versus effective capacity in the TUI and REPL

**Owner:** routine worker; may proceed after Task 3 while Tasks 4–6 own the loop.

**Files:**

- Modify: `src/turn_stream.rs`
- Modify: `src/tui_app/app.rs`
- Modify: `src/tui_app/mod.rs`
- Modify: `src/repl/mod.rs`
- Modify: `src/repl/commands/lifecycle.rs`
- Modify: focused marker/UI tests

- [ ] **Step 1: Run impact analysis.**

  Analyze `ControlMarker`, `App::on_delta`, context/status commands, and the status-bar renderer.

- [ ] **Step 2: Add one typed capacity marker.**

  Carry prior/current total, prompt/output limits, pressure, basis, generation, and action. Never carry raw server display strings or prompt content.

- [ ] **Step 3: Add red rendering tests.**

  Require marker round-trip, `capacity 49K -> 36K · constrained · compacting`, configured/effective distinction, legacy fallback labeling, interrupted-output retraction, and absence of prompt content in logs.

- [ ] **Step 4: Render current state.**

  `/context` and `/status` show configured ceilings separately from effective live limits plus adaptive/legacy, basis, pressure, boot/generation. The active turn line explains fetch, reduction, wait, retry, unavailable, and recovery states.

- [ ] **Step 5: Validate and commit.**

  Run: `cargo test --release turn_stream --lib -- --nocapture`

  Run: `cargo test --release tui_app --lib -- --nocapture`

  Commit: `feat(tui): explain adaptive capacity`

---

### Task 8: Run synchronized HTTP replay and release gates

**Owner:** root integration agent; independent reviewers own no production files.

**Files:**

- Modify only existing Nanobot integration/replay fixtures if needed
- Update release notes/config documentation only for proven shipped behavior

- [ ] **Step 1: Run cross-binary contract conformance.**

  Start the synchronized release Higgs binary in `tmux` with injectable capacity/pressure test inputs. Run Nanobot through real HTTP, not mocks. Prove schema agreement, captured request budget, 413 fields, retained-session drop/rotation, one logical turn/result, and no tool duplication.

- [ ] **Step 2: Run compatibility scenarios.**

  Prove a fresh Higgs boot invalidates cached capacity and retained state; old Higgs uses the 16K/4K fallback; malformed typed bodies do not trigger recovery; unavailable work survives restart/cancellation semantics.

- [ ] **Step 3: Run full release validation.**

  Run: `cargo fmt --all -- --check`

  Run: `cargo test --release`

  Run: `cargo build --release`

- [ ] **Step 4: Run matched turn benchmarks.**

  Run baseline and adaptive builds with `scripts/turn_bench.sh`, using separate output directories under `/private/tmp`. Ordinary warm turns must show no material regression.

- [ ] **Step 5: Run the target-Mac Escha replay alone.**

  Grow genuine agent history through cold, cache-warm, tool-heavy, compacted, warning, and critical cases. Record request-scoped MLX peak, compressor/swap deltas, TTFT, prefill/decode rate, capacity generations, and UI decisions. The gate is zero new swap-outs in the clean replay and safe refusal/reduction under injected pressure; synthetic prompts may diagnose but cannot satisfy the gate.

- [ ] **Step 6: Review before shipping.**

  Run GitNexus `detect_changes` against `main`; request independent Higgs accounting/RAII review, Nanobot durability review, and cross-wire review. Resolve all correctness findings and repeat release tests and the real replay.

## Parallel Execution Waves

1. Contract fixture freezes first.
2. Higgs pure controller/engine work and Nanobot Tasks 1–2 proceed in parallel using fixtures.
3. Nanobot Task 3 installs the runtime seam; TUI Task 7 can then run in parallel with loop work.
4. One deep worker owns Tasks 4–5 sequentially. Task 6 starts only after Task 5 is merged; its worker then owns both replay and loop integration, avoiding concurrent edits to `shared.rs`/`response.rs`. Do not split prompt recovery from tool-idempotence review.
5. Higgs admission and Nanobot recovery merge before the real HTTP replay. Real hardware replay runs alone so other agents do not confound memory pressure.

## Nanobot Completion Gate

- Effective live budget can decrease without a config rewrite or core rebuild.
- Proactive reduction never makes an oversized compactor request.
- A typed 413 produces at most one reissued provider request and never repeats the outer turn or a committed tool.
- A 503 remains durable and resumable; terminal SSE bytes remain incomplete, never a successful assistant message.
- TUI/status state explains configured and effective limits.
- Full release tests/build, matched benchmark, synchronized HTTP replay, and target-Mac Escha replay pass.
