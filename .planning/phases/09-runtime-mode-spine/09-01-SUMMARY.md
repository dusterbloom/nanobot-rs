---
phase: 09-runtime-mode-spine
plan: 01
subsystem: infra
tags: [rust, enums, runtime-mode, type-safety, G1, G5, parallel-rollout]

# Dependency graph
requires:
  - phase: 09-runtime-mode-spine (Wave 0)
    provides: "Pinned-behavior unit tests in agent_shared / agent_heuristics — the safety net this wave's derivations must remain compatible with."
provides:
  - "Typed RuntimeMode { Cloud | Local { caps: Arc<ModelCapabilities> } } enum"
  - "Nine derivation methods centralising the is_local critical-branch table"
  - "ThinkingCapPolicy + BudgetShares helper types (G1/G5 compliant)"
  - "27 unit tests pinning every derivation output the refactor must preserve"
  - "Module re-export; zero production callsite migrated (parallel rollout)"
affects: [09-02-migrate-derivations, 09-03-remove-is-local, 09-04-final-proof]

# Tech tracking
tech-stack:
  added: []  # No new crates — std::sync::Mutex used for env-test serialisation
  patterns:
    - "Parallel-rollout enum introduction — new type lives alongside the bool field it replaces"
    - "Derivation methods dispatch via `match` only (no `if let`), exhaustiveness is the Nyquist filter for future variants"
    - "env > caps > default priority resolver extracted as a free function (G4)"
    - "Env-touching tests serialise on a module-local Mutex to keep parallel cargo test deterministic"

key-files:
  created:
    - src/agent/runtime_mode.rs
  modified:
    - src/agent/mod.rs

key-decisions:
  - "Two-variant shape locked per 09-CONTEXT.md; no Cluster/Remote variants introduced this wave"
  - "Local wraps Arc<ModelCapabilities> (shared handle, cheap clone) — matches the locked spec even though SwappableCore currently stores caps by value"
  - "protocol() returns Option<LocalReplayMode>; Cloud yields None rather than a sentinel variant — keeps LocalReplayMode cloud-agnostic"
  - "Reused existing LocalToolMode (config::schema) for tool_def_mode() rather than inventing ToolDefMode — minimises churn"
  - "Introduced new local enum ThinkingCapPolicy + struct BudgetShares rather than tuples — aligns with CLAUDE.md G1/G5"
  - "Env-override resolver uses the same parsing rules as protocol.rs::LocalReplayMode::from_env (accepts native|tool_calls|native_tool_calls and text|textual|textual_replay) so Wave 3 can swap the old helper for this one without semantic drift"

patterns-established:
  - "New runtime-policy method lives next to constants it owns (LOCAL_CONTEXT_CAP_LITE etc.) — G4 locality, no cross-module magic numbers"
  - "Test helpers grouped in `mod fixtures` sub-module — cloud_mode / higgs_mode / small_mode builders are the canonical fixture trio for Waves 2/3"
  - "Env-test discipline: every test that reads protocol() holds `env_mutex()` via `lock_env_cleared()` — pattern for any future process-env-coupled tests"

requirements-completed: [MODE-01, MODE-02]

# Metrics
duration: 7m1s
completed: 2026-04-20
---

# Phase 09 Plan 01: RuntimeMode Type Summary

**Typed `RuntimeMode` enum with 9 derivation methods and 27 invariant tests — parallel-rollout foundation for collapsing 33 `is_local` reads into a single dispatching type.**

## Performance

- **Duration:** 7m 01s
- **Started:** 2026-04-20T14:10:46Z
- **Completed:** 2026-04-20T14:17:47Z
- **Tasks:** 2 (committed as one atomic change — module declaration and enum are inseparable for the test build to compile)
- **Files modified:** 2 (1 created, 1 edited)

## Accomplishments

- Shipped `RuntimeMode` with the two-variant shape locked by 09-CONTEXT.md (`Cloud | Local { caps: Arc<ModelCapabilities> }`).
- Implemented all 9 derivation methods from the critical-branch table with `match`-only dispatch.
- Added two new helper types (`ThinkingCapPolicy`, `BudgetShares`) that absorb what would otherwise have been scattered bools and tuples.
- Landed 27 passing unit tests covering variant resolution, every method on all three canonical fixtures (Cloud 200K / Higgs 32K / small-local 8K), and the `NANOBOT_LOCAL_PROTOCOL_MODE` env-override path.
- Zero consumer callsites migrated — rollback is a single-file revert.

## Task Commits

Each task was committed atomically (Tasks 1 and 2 combined into one commit — the module declaration and the test-bearing file must land together for the lib test target to compile):

1. **Task 1 + Task 2: RuntimeMode enum + module re-export** — `7d6d2d3` (feat)

_Note: Original plan listed these as two tasks. Task 2 (re-export in `mod.rs`) is a one-line change with no independent verification path — splitting the commit would create a commit where tests cannot be run, which violates "commit each task only when verification passes". Combined commit retains full auditability._

## Enum Definition (copied from source)

```rust
#[derive(Debug, Clone)]
pub enum RuntimeMode {
    /// Cloud / remote managed API (Anthropic, OpenAI, OpenRouter, ...).
    Cloud,
    /// Any locally-reachable backend (Higgs sidecar, LM Studio, vLLM,
    /// cluster peer). Capabilities carry the finer-grained differentiation.
    Local {
        /// Cheap handle to the resolved model capabilities. Shared via `Arc`
        /// so `RuntimeMode` stays `Clone` without deep-copying the struct.
        caps: Arc<ModelCapabilities>,
    },
}
```

## Derivation Invariants — Feeds Wave 2 Migration

> Table: per-method output on each canonical fixture, plus the test function that pins it. Wave 2 must reproduce these rows exactly when migrating the production call sites.

| Method                              | Cloud (200K)                 | Local-Higgs (32K, tool_calling=true, large) | Local-Small (8K, tool_calling=false, small) | Pinning Test                                              |
| ----------------------------------- | ---------------------------- | ------------------------------------------- | ------------------------------------------- | --------------------------------------------------------- |
| `is_local()`                        | `false`                      | `true`                                      | `true`                                      | `resolves_cloud_variant`, `resolves_small_local_variant`  |
| `context_cap(max)`                  | `max` (200_000)              | `800`                                       | `800`                                       | `context_cap_cloud_uncapped`, `context_cap_local_clamped_to_lite` |
| `reserve_cap(max_tokens, max_ctx)`  | `max_tokens` (8_192)         | `min(8_192, 32_768/4) = 8_192`              | `min(8_192, 8_192/4) = 2_048`               | `reserve_cap_cloud_passthrough`, `reserve_cap_local_higgs_at_25_pct_boundary`, `reserve_cap_local_small_clamped` |
| `max_iterations(cfg, max_ctx)`      | `max(cfg, ctx/4000 capped at 50)` | `min(cfg, 15)`                         | `min(cfg, 15)`                              | `max_iterations_cloud_scales_up`, `max_iterations_local_clamped_to_15` |
| `budget_strategy(max_ctx)`          | `2/1/4/2 %`                  | `2/1/2/1 %`                                 | `2/1/2/1 %`                                 | `budget_strategy_cloud_ratios`, `budget_strategy_local_ratios` |
| `protocol()` (no env override)      | `None`                       | `Some(NativeToolCalls)`                     | `Some(TextualReplay)`                       | `protocol_cloud_is_none`, `protocol_derives_from_caps_native_when_tool_calling`, `protocol_derives_from_caps_textual_when_no_tool_calling` |
| `protocol()` (env override wins)    | `None` (env ignored)         | env-forced                                  | env-forced                                  | `protocol_env_override_beats_caps`                        |
| `tool_def_mode()`                   | `Full`                       | `Slim`                                      | `Proxy`                                     | `tool_def_mode_cloud_is_full`, `tool_def_mode_local_slim_when_tool_calling`, `tool_def_mode_local_proxy_when_no_tool_calling` |
| `needs_anti_drift()`                | `false`                      | `true`                                      | `true`                                      | `needs_anti_drift_cloud_false`, `needs_anti_drift_local_true_for_all_sizes` |
| `grounding_role()`                  | `"system"`                   | `"user"`                                    | `"user"`                                    | `grounding_role_cloud_is_system`, `grounding_role_local_is_user` |
| `thinking_cap_policy()`             | `Uncapped`                   | `Uncapped` (large)                          | `Hard(2048)` (small)                        | `thinking_cap_cloud_uncapped`, `thinking_cap_local_large_uncapped`, `thinking_cap_local_small_is_hard` |
| `needs_local_protocol()`            | `false`                      | `true`                                      | `true`                                      | `needs_local_protocol_matches_protocol_some`              |

Totals: 27 unit tests, 12 distinct methods / convenience wrappers pinned.

## Compile-Time Attestation

Release build output tail (Linker stage, post-change tree):

```
warning: `nanobot` (lib) generated 35 warnings (run `cargo fix --lib -p nanobot` to apply 15 suggestions)
    Finished `release` profile [optimized] target(s) in 23.53s
```

- `cargo build --lib` warning count: **36 → 36** (zero new warnings from `runtime_mode.rs`).
- `cargo build --release --lib` warning count: **35 → 35** (zero new warnings).
- `cargo clippy --lib --no-deps` — no `runtime_mode.rs`-originating diagnostics (pre-existing errors in unrelated files remain, SCOPE BOUNDARY leaves them).
- `./quality-sentinel.sh` — `All sentinel checks passed.` (zero new G1 mutable-bool flags, zero new G5 else-if chains).
- `cargo test --lib runtime_mode` — **27 passed, 0 failed**.
- `cargo test --lib` — **2100 passed, 0 failed, 16 ignored** (full library suite).

## Files Created/Modified

- `src/agent/runtime_mode.rs` **(created, 605 lines)** — the enum, helper types, derivation methods, env-resolver, and inline `#[cfg(test)] mod tests` with canonical fixtures and 27 pinning tests.
- `src/agent/mod.rs` **(edited, +1 line)** — `pub mod runtime_mode;` declaration inserted alphabetically between `router_fallback` and `sanitize`.

## Decisions Made

All decisions recorded in frontmatter `key-decisions`. The two worth highlighting here:

1. **`LocalReplayMode` already exists; the plan text referenced a non-existent `LocalProtocolMode`.** Used the real type (`LocalReplayMode` from `protocol.rs`). `protocol()` returns `Option<LocalReplayMode>` where `None` represents Cloud, rather than inventing a third variant in a cross-cutting enum. This keeps `LocalReplayMode` a pure local-model concept.

2. **`ModelCapabilities` lives in `agent::model_capabilities`, not `config::schema` as the plan stated.** Imported from the correct path (`crate::agent::model_capabilities::{ModelCapabilities, ModelSizeClass}`). Also noted the capability struct also exposes `thinking`, `needs_native_lms_api`, `strict_alternation`, `reader_tier`, and `parser` fields not mentioned in the plan interfaces block — test fixtures construct the full struct so they stay compile-safe as the struct evolves.

## Deviations from Plan

All deviations are Rule-1 / Rule-3 corrections to plan inaccuracies discovered against the actual codebase. No new external dependencies; no architectural changes.

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Plan referenced `LocalProtocolMode`; the actual type is `LocalReplayMode`**
- **Found during:** Task 1 (enum definition)
- **Issue:** The plan's `<interfaces>` block said `LocalProtocolMode` (`NativeToolCalls | TextualReplay`) exists in `src/agent/protocol.rs`. Grep found the real name is `LocalReplayMode`. Using the plan's name verbatim would not compile.
- **Fix:** Imported `LocalReplayMode` throughout. `protocol()` returns `Option<LocalReplayMode>`. Documented in module-level doc-comment.
- **Files modified:** `src/agent/runtime_mode.rs` (throughout).
- **Verification:** `cargo build --lib` green; all 27 tests reference `LocalReplayMode::NativeToolCalls` / `::TextualReplay` and pass.
- **Committed in:** `7d6d2d3`

**2. [Rule 3 - Blocking] Plan pointed `ModelCapabilities` at `src/config/schema.rs`; the real path is `src/agent/model_capabilities.rs`**
- **Found during:** Task 1 (fixture setup)
- **Issue:** Plan claimed `ModelCapabilities` lives in `config::schema`. Grep located the definition at `src/agent/model_capabilities.rs:28` (with extra fields beyond what the plan enumerated — `thinking`, `needs_native_lms_api`, `strict_alternation`, `reader_tier`, `parser`).
- **Fix:** Imported from `crate::agent::model_capabilities::{ModelCapabilities, ModelSizeClass}`. Fixture builders construct the full struct literal.
- **Files modified:** `src/agent/runtime_mode.rs`.
- **Verification:** `cargo build --lib` green; fixtures resolve.
- **Committed in:** `7d6d2d3`

**3. [Rule 1 - Bug] Parallel-test env-var race in `protocol()` tests**
- **Found during:** Task 1 verification (first `cargo test --lib runtime_mode` run)
- **Issue:** Five tests exercise `protocol()` and the env-override path. `cargo test` runs tests in parallel; without a lock, `protocol_env_override_beats_caps` consistently failed because a sibling test cleared `NANOBOT_LOCAL_PROTOCOL_MODE` mid-assertion.
- **Fix:** Introduced a module-local `std::sync::OnceLock<Mutex<()>>` (`env_mutex()`). Added `lock_env_cleared()` helper that every `protocol()`-reading test holds for the duration of its assertions. Env-override test also holds the lock while mutating the var.
- **Files modified:** `src/agent/runtime_mode.rs` (test module only).
- **Verification:** All 27 runtime_mode tests green, repeatedly; full library suite (2100 tests) green.
- **Committed in:** `7d6d2d3`

**4. [Commit-granularity deviation] Task 1 and Task 2 combined into one commit**
- **Found during:** Final commit preparation
- **Issue:** Task 2 (re-export in `src/agent/mod.rs`) is a one-line `pub mod runtime_mode;` declaration. Without it, the file added in Task 1 is not compiled by `cargo test --lib` and Task 1's tests cannot run. Committing Task 1 alone would produce a commit where the verification step fails.
- **Fix:** Committed both together as `feat(09-01): introduce RuntimeMode enum with derivation methods`. Commit message enumerates both task outputs.
- **Files modified:** both files staged in one commit.
- **Verification:** `cargo test --lib runtime_mode` green at commit HEAD.
- **Committed in:** `7d6d2d3`

---

**Total deviations:** 4 — 2 blocking plan-vs-reality corrections, 1 test-parallelism bug fix, 1 commit-granularity judgement call.
**Impact on plan:** All fixes are mechanical and required. No scope creep. Every deviation keeps the plan's locked invariants (two-variant shape, `match`-only dispatch, no new capability struct, env > caps > default priority) intact.

## Issues Encountered

None that required problem-solving beyond the deviations documented above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **Wave 2 (`02-migrate-derivations-PLAN.md`) is unblocked.** The nine derivation methods on `RuntimeMode` are implemented and test-pinned; Wave 2 can now migrate the memory-provider and delegation-provider construction sites in `agent_core.rs::build_swappable_core` to consult `RuntimeMode` instead of `is_local`.
- **Wave 3 (`03-remove-is-local-PLAN.md`) is unblocked on its Wave 2 dependency.** The Cloud-branch / Local-branch invariants needed for the 33-call-site migration are now pinned.
- **Zero production callsite has been touched** — rollback of this wave is `git revert 7d6d2d3`, a single-commit operation.

## Self-Check: PASSED

Verification results:

- `.planning/phases/09-runtime-mode-spine/09-01-SUMMARY.md` — FOUND (this file)
- `src/agent/runtime_mode.rs` — FOUND
- `src/agent/mod.rs` contains `pub mod runtime_mode;` — FOUND
- Commit `7d6d2d3` exists in git log — FOUND
- `cargo test --lib runtime_mode` — 27/27 pass
- `cargo test --lib` — 2100/2100 pass
- `cargo build --release --lib` — green, zero new warnings
- `./quality-sentinel.sh` — `All sentinel checks passed.`

---
*Phase: 09-runtime-mode-spine*
*Completed: 2026-04-20*
