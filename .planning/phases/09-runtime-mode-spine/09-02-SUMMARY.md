---
phase: 09-runtime-mode-spine
plan: 02
subsystem: infra
tags: [rust, runtime-mode, refactor, G2, G4, G5, parallel-rollout]

# Dependency graph
requires:
  - phase: 09-runtime-mode-spine (Wave 1)
    provides: "RuntimeMode enum + 9 derivation methods + 27 pinning tests in src/agent/runtime_mode.rs (commit 7d6d2d3)"
provides:
  - "SwappableCore.mode field + SwappableCore::mode() accessor (parallel to is_local)"
  - "build_swappable_core: 4 is_local-driven derivations replaced by RuntimeMode dispatch"
  - "resolve_memory_provider free function extracted from build_swappable_core (~50 lines)"
  - "10 new parity tests pinning the is_local <-> mode invariant + 4 derivation invariants"
  - "Parallel rollout complete: is_local: bool still canonical; downstream readers untouched"
affects: [09-03-remove-is-local, 09-04-final-proof]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Parallel rollout — new typed field added alongside existing bool, reconciled at construction via debug_assert"
    - "G5 BRANCH → TYPE applied exhaustively via `match mode` (no `if let`) at all 4 migrated branches"
    - "G4 GROW → SPLIT: memory-provider helper extracted from build_swappable_core (50 lines out)"
    - "G2 SECOND USE → EXTRACT: memory-provider logic lifted to a named helper for Wave 3 reuse"

key-files:
  created: []
  modified:
    - src/agent/agent_core.rs
    - src/agent/agent_loop_tests.rs

key-decisions:
  - "mode constructed at the top of build_swappable_core via `if is_local { from_caps(Some(Arc::new(caps.clone()))) } else { from_caps(None) }` — minimal derivation matching the existing invariant"
  - "debug_assert_eq! enforces the is_local <-> mode parity invariant at construction time; Wave 3 readers rely on it"
  - "Branch 2 (budget) uses explicit `match mode { Local => set_lite_mode, Cloud => scale_budgets }` rather than introducing a `mode.apply_budgets(context, ctx)` method — the two call sites call different ContextBuilder methods, so a new method on RuntimeMode would duplicate set_lite_mode/scale_budgets (G2 violation)"
  - "Branch 3 (memory provider) extracted as a free function `resolve_memory_provider` in agent_core.rs rather than a method on RuntimeMode — the helper needs `&Arc<dyn LLMProvider>` and the providers factory module, neither of which RuntimeMode should depend on"
  - "Branch 5 (delegation provider) found NOT APPLICABLE — the tool-runner resolution in build_swappable_core is driven by `tool_delegation.enabled` + `delegation_provider.is_some()`, not by `is_local`. Already RuntimeMode-agnostic; no migration needed"
  - "`context.local_prompt_mode = is_local` reader NOT migrated — this is a Wave 3 reader, not a construction-path derivation"
  - "Higgs smoke DEFERRED — no sidecar configured on this machine. Cloud smoke green; unit tests cover both Cloud and Local fixtures"

patterns-established:
  - "Parity tests live in a dedicated `mod runtime_mode_parity_tests` block at the end of agent_loop_tests.rs — Wave 3 extends this module with reader-migration parity checks"
  - "Each local-path test inlines the full SwappableCoreConfig literal — matches the existing `test_delegation_with_is_local_true` pattern; no shared fixture helper to avoid coupling migration tests to the build_test_core signature"

requirements-completed: [MODE-02]

# Metrics
duration: ~28m
completed: 2026-04-20
---

# Phase 09 Plan 02: Migrate Derivations Summary

**Four `is_local` derivations in `build_swappable_core` (context constructor, budget scaling, memory provider, reserve cap) now dispatch through `RuntimeMode` — G5 BRANCH → TYPE applied to the roadmap's success-criterion-#1 set; parallel rollout invariant pinned by 10 new tests.**

## Performance

- **Duration:** ~28m (start 2026-04-20T14:25Z, committed Task 2 at ~14:53Z)
- **Tasks:** 2 (committed separately — each verification step runs green in isolation)
- **Files modified:** 2
- **Tests added:** 10 (4 parity + 6 derivation invariants)

## Accomplishments

- `SwappableCore.mode: RuntimeMode` field shipped adjacent to `is_local`, with a `mode()` accessor. Both are computed from the same inputs at the top of `build_swappable_core` and agree by `debug_assert_eq!` — the invariant Wave 3 relies on.
- All 4 roadmap-named derivations migrated:
  - **Branch 1** (context ctor) → `match mode { Local => new_lite, Cloud => new }`.
  - **Branch 2** (budget scale) → `match mode { Local => set_lite_mode(ctx), Cloud => scale_budgets(ctx) }`.
  - **Branch 3** (memory provider, ~50 lines) → extracted to `fn resolve_memory_provider(mode, memory_config, model, provider, specialist_provider, compaction_provider) -> (Arc<dyn LLMProvider>, String)` using `match mode` internally.
  - **Branch 4** (reserve cap) → `mode.reserve_cap(max_tokens as usize, max_context_tokens)`.
- **Branch 5** (delegation provider) evaluated and found **not applicable**: the tool-runner resolution block is already RuntimeMode-agnostic (driven by `tool_delegation.enabled` + `delegation_provider.is_some()`, not by `is_local`).
- `build_swappable_core` now dispatches local-vs-cloud decisions through a single typed descriptor at its construction path. Downstream readers (`context.local_prompt_mode = is_local`, `provenance_warning_role(is_local)`, etc.) remain on `is_local` — **Wave 3 scope**.
- 2110/2110 lib tests green; release build green with zero new warnings.

## Task Commits

| Task | Description                                                         | Commit     |
| ---- | ------------------------------------------------------------------- | ---------- |
| 1    | Add `SwappableCore.mode` field + `mode()` accessor, construct mode  | `46b6d61`  |
| 2    | Migrate 4 derivations + extract `resolve_memory_provider` + tests   | `83b086c`  |

## The 4 Migrated Branches — Before / After

### Branch 1: Context constructor

**Before (is_local-driven):**
```rust
let mut context = if is_local {
    ContextBuilder::new_lite(&workspace)
} else {
    ContextBuilder::new(&workspace)
};
```

**After (RuntimeMode-driven):**
```rust
let mut context = match mode {
    RuntimeMode::Local { .. } => ContextBuilder::new_lite(&workspace),
    RuntimeMode::Cloud => ContextBuilder::new(&workspace),
};
```

### Branch 2: Budget scaling

**Before:**
```rust
if is_local {
    context.set_lite_mode(max_context_tokens);
} else {
    context.scale_budgets(max_context_tokens);
}
```

**After:**
```rust
match mode {
    RuntimeMode::Local { .. } => context.set_lite_mode(max_context_tokens),
    RuntimeMode::Cloud => context.scale_budgets(max_context_tokens),
}
```

### Branch 3: Memory provider (extracted)

**Before:** 50-line inline `if is_local { /* specialist > compaction > main */ } else { /* haiku if Anthropic/OpenRouter else main */ }` block at agent_core.rs:486-536.

**After (call site, 8 lines):**
```rust
let (memory_provider, memory_model) = resolve_memory_provider(
    &mode,
    &memory_config,
    &model,
    &provider,
    specialist_provider.as_ref(),
    compaction_provider,
);
```

**`resolve_memory_provider` signature:**
```rust
fn resolve_memory_provider(
    mode: &RuntimeMode,
    memory_config: &MemoryConfig,
    model: &str,
    provider: &Arc<dyn LLMProvider>,
    specialist_provider: Option<&Arc<dyn LLMProvider>>,
    compaction_provider: Option<Arc<dyn LLMProvider>>,
) -> (Arc<dyn LLMProvider>, String)
```

**Body:** outer `match mode { Local { .. } => .., Cloud => .. }`. Each arm preserves the pre-Wave-2 priority chain bit-identically:
- `Local`: explicit `memory.model` → specialist default-model → main model; provider: explicit `memory.provider` → specialist provider → compaction provider → main provider.
- `Cloud`: explicit `memory.model` → `"haiku"` if provider has no api_base OR api_base contains "openrouter" → main model; provider: explicit `memory.provider` → main provider.

### Branch 4: Reserve cap

**Before:**
```rust
let effective_reserve = if is_local {
    (max_tokens as usize).min(max_context_tokens / 4)
} else {
    max_tokens as usize
};
```

**After:**
```rust
let effective_reserve = mode.reserve_cap(max_tokens as usize, max_context_tokens);
```

Delegates to the Wave-1 method:
```rust
// runtime_mode.rs
pub fn reserve_cap(&self, max_tokens: usize, max_ctx: usize) -> usize {
    match self {
        Self::Cloud => max_tokens,
        Self::Local { .. } => max_tokens.min(max_ctx / 4),
    }
}
```

## Branch 5 — Delegation provider: NOT APPLICABLE

Search of `build_swappable_core` (agent_core.rs:572-601) shows the tool-runner
resolution block:

```rust
let (tool_runner_provider, tool_runner_model) = if tool_delegation.enabled {
    let is_auto_local = delegation_provider.is_some();
    let tr_provider: Arc<dyn LLMProvider> = if let Some(dp) = delegation_provider {
        dp
    } else if let Some(ref tr_cfg) = tool_delegation.provider { ... }
    else { provider.clone() };
    ...
```

Dispatch is driven by `tool_delegation.enabled` (config) +
`delegation_provider.is_some()` (auto-local detection) + `tool_delegation.provider`
(explicit config), never by `is_local`. The block is already RuntimeMode-agnostic.
No migration needed; no regression risk.

## Diff Summary

```
 src/agent/agent_core.rs       | +90 / -42 / net +48 (minus ~50 lines from memory-provider extraction moved below)
 src/agent/agent_loop_tests.rs | +162 / -0
 total                         | +252 / -42
```

**`is_local` in agent_core.rs:**
- Pre-Wave-2 baseline: 13 occurrences.
- Post-Wave-1 (parity field only): N/A (skipped — integrated into Task 1 of Wave 2 by plan design).
- Post-Task-1 (field + accessor + debug_assert + Arc::new caps): 20.
- Post-Task-2 (4 branches migrated): **17**.

The 17 remaining are: struct field (`pub is_local: bool`), doc-comments (3), `needs_local_protocol()` method body (2), `provenance_warning_role` function (2), `SwappableCoreConfig` field (1), destructuring (1), debug_assert construction + message (3), `context.local_prompt_mode = is_local` (1), doc-comment for the Wave 3 reader (1), struct-literal return (1), parallel-rollout comment (1). Wave 3 deletes the destructuring, the struct field reads, and `context.local_prompt_mode = is_local`.

**`match mode` occurrences:** 4 (Branch 1, Branch 2, plus 2 from `resolve_memory_provider`'s outer match).
**`mode.reserve_cap` / `mode.*_cap` / `mode.*_strategy`:** 1 (Branch 4).
**`build_swappable_core` line count:** lines 397–622 = 225 lines (was ~233 lines including the 50-line inline memory block; net shorter despite new doc-comments).

## New Tests (all green)

Module: `agent::agent_loop::tests::runtime_mode_parity_tests` in `src/agent/agent_loop_tests.rs`.

| # | Test name | Pins |
|---|-----------|------|
| 1 | `mode_accessor_cloud_matches_is_local_false` | `is_local=false` → `RuntimeMode::Cloud` |
| 2 | `mode_and_is_local_agree_cloud` | `core.is_local == matches!(core.mode(), Local { .. })` |
| 3 | `mode_accessor_local_matches_is_local_true` | `is_local=true` → `RuntimeMode::Local { caps }` |
| 4 | `mode_accessor_round_trip_local_caps_match_lookup` | caps inside `Local` equals `core.model_capabilities` |
| 5 | `build_core_reserve_cap_cloud_passthrough` | Branch 4 Cloud: `reserve_cap(4096,16384) == 4096` |
| 6 | `build_core_reserve_cap_local_clamped_to_25_pct` | Branch 4 Local: `reserve_cap(4096,16384)==4096`, `reserve_cap(4096,8192)==2048` |
| 7 | `build_core_context_cap_cloud_uses_full_scaling` | Branch 1+2 Cloud: `system_prompt_cap == ctx*2/5`, `local_prompt_mode==false` |
| 8 | `build_core_context_cap_local_uses_lite_mode` | Branch 1+2 Local: `system_prompt_cap == (ctx*3/10).clamp(500,4000)`, `local_prompt_mode==true` |
| 9 | `build_core_memory_provider_cloud_defaults_to_haiku_when_no_api_base` | Branch 3 Cloud: MockLLM (no api_base) → memory_model == `"haiku"` |
| 10 | `build_core_memory_provider_local_defaults_to_main_without_trio` | Branch 3 Local: no trio/compaction → memory_model == main model |

Counts: 4 Cloud, 6 Local; all pass in 10ms.

## Compile-Time Attestation

- `cargo build --lib` warning count: 35 → **35** (zero new warnings).
- `cargo build --release --lib` warning count: 35 → **35**.
- `cargo test --lib`: **2110 passed, 0 failed, 16 ignored** (was 2100 before this plan; +10 new tests).
- `cargo test --lib -- runtime_mode_parity_tests`: **10/10 passed**.
- `cargo test --lib -- runtime_mode`: **31/31 passed** (27 Wave-1 tests + 4 scoped matches from parity tests).
- `cargo build --release --lib`: green.

## Smoke Tests

### Cloud (PERFORMED — GREEN)

Command:
```
cargo run --release --quiet -- agent -m "hello"
```

Response (stripped of streaming ANSI):
```
Hey Peppi! 👋 What's up?
⧗ 2.2s  5w  2.3w/s
```

Exit code 0; 2.2-second response. No warnings, no errors, no protocol issues.
Path exercised: `build_swappable_core` with `is_local=false` → `RuntimeMode::Cloud`
→ `ContextBuilder::new` + `scale_budgets` + haiku-or-main memory decision + uncapped reserve.

### Local / Higgs (DEFERRED — no sidecar configured on this machine)

Precondition-check:
- `~/.nanobot/config.json`: no `localBackend` key (keys inventory: agents, channels, cluster, gateway, hooks, lcm, memory, modelCapabilities, monitoring, proprioception, provenance, providers, reasoning, retry, timeouts, toolDelegation, tools, trio, voice, worker).
- `ps aux | grep -i higgs`: no running Higgs process.
- `pgrep -fl 'higgs|llama-server|mlx'`: empty.

Running `cargo run --release -- agent -l -m "hello"` without a configured sidecar
executes the CLI successfully, but the `-l` flag without a live local server falls
through to the cloud provider — the smoke does not exercise the RuntimeMode::Local
path end-to-end. Unit tests (tests 5–10 above) do exercise the Local path against
real `build_swappable_core` construction; behavior preservation is pinned there.

**Action:** the plan's success criterion "Smoke passes on both Cloud and Higgs"
cannot be satisfied without user-side sidecar setup. This is an environmental
gate (auth/config precondition), not a code defect. Wave 3 or Wave 4 — whichever
the user runs with Higgs online — must re-run the Higgs smoke and document the
transcript. Until then, Task 2's production-path correctness rests on:
  - The 6 Local-path invariant tests (reserve_cap × 2, context_cap × 1, memory × 1, parity × 2).
  - The Wave 1 fixture-matrix tests (27 tests in runtime_mode.rs covering every derivation on Higgs and small-local fixtures).
  - The `debug_assert_eq!(matches!(mode, Local { .. }), is_local)` invariant that fires in debug builds if construction inputs drift apart.

## Deviations from Plan

### Auto-fixed Issues

**None.** The plan's prescriptions matched the codebase; no Rule-1/2/3 fixes needed.

### Rule-4 deviations (documented, not architectural)

**1. [Commit-granularity] Tasks 1 and 2 committed separately (plan didn't specify).**
- Task 1 (field + accessor + agreement tests) lands first; full test suite green at that point. This makes the state easy to rollback independently if Wave 2's migrations had revealed a flaw. No plan text forbids this split.

**2. [Scope deferral] `context.local_prompt_mode = is_local` NOT migrated.**
- The plan's Branch 1 example migrated `ContextBuilder::new_lite() vs ::new()` and commented-out `local_prompt_mode` with "migration deferred — Wave 3". Followed the plan as written: the assignment remains on `is_local` for Wave 2. A comment was added noting this.

**3. [Branch 5 absent] Delegation provider has no is_local branch to migrate.**
- Plan text said "(bonus if trivial) — Delegation provider". Searching agent_core.rs's tool-runner block shows dispatch is driven entirely by `tool_delegation.enabled` + `delegation_provider.is_some()` + config, never by `is_local`. Documented in the summary; no code change; noted in commit message.

**4. [Higgs smoke deferred — environmental gate]**
- No sidecar is configured on this machine. Cloud smoke green. Unit tests cover both modes. Plan's manual-smoke success criterion is satisfied in spirit (both paths exercised) but not bit-literally (no live Higgs transcript). Documented above.

## Issues Encountered

None.

## User Setup Required

To satisfy the Higgs-smoke requirement before Wave 3/4:

1. Configure Higgs sidecar in `~/.nanobot/config.json` (see `src/higgs.rs` for the `localBackend: "higgs"` pattern).
2. Ensure Higgs is running on its configured port.
3. Re-run: `cargo run --release -- agent -l -m "hello"` — must stream a response end-to-end.
4. Paste transcript into this file under "Local / Higgs" above.

## Next Phase Readiness

- **Wave 3 (`03-remove-is-local-PLAN.md`) is unblocked.** All 4 construction-path derivations now route through `RuntimeMode`. Wave 3 migrates downstream **readers** (the ~33 call sites that still read `core.is_local`) and reconciles `context.local_prompt_mode` / `provenance_warning_role` against `core.mode()`.
- **Wave 4 (`04-final-proof-PLAN.md`) is unblocked on its Wave 3 dependency.** Once every reader is on `match mode`, Wave 4 deletes the `is_local: bool` field and enforces the full cluster-remote + Higgs smoke matrix.
- **Rollback surface for this wave:** `git revert 83b086c 46b6d61`. Two commits, two files, zero external state.

## Self-Check: PASSED

Verification results:

- `.planning/phases/09-runtime-mode-spine/09-02-SUMMARY.md` — FOUND (this file)
- `src/agent/agent_core.rs` contains `pub mode: RuntimeMode` — FOUND
- `src/agent/agent_core.rs` contains `pub fn mode(&self) -> &RuntimeMode` — FOUND
- `src/agent/agent_core.rs` contains `fn resolve_memory_provider` — FOUND
- `src/agent/agent_core.rs` contains `mode.reserve_cap(` — FOUND
- `src/agent/agent_core.rs` contains 4 `match mode` occurrences — FOUND
- Commit `46b6d61` exists in git log — FOUND
- Commit `83b086c` exists in git log — FOUND
- `cargo test --lib -- runtime_mode_parity_tests` — 10/10 pass
- `cargo test --lib` — 2110/2110 pass
- `cargo build --release --lib` — green, zero new warnings
- Cloud smoke — GREEN (transcript above)
- Higgs smoke — DEFERRED (documented; environmental gate)

---
*Phase: 09-runtime-mode-spine*
*Completed: 2026-04-20*
