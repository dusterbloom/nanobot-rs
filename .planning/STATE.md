---
gsd_state_version: 1.0
milestone: v0.4
milestone_name: milestone
current_plan: 3
status: "RuntimeMode type landed as parallel structure; Wave 2 ready to migrate derivations in agent_core.rs::build_swappable_core"
last_updated: "2026-04-20T14:20:24.353Z"
last_activity: 2026-04-20
progress:
  total_phases: 3
  completed_phases: 0
  total_plans: 5
  completed_plans: 3
  percent: 40
---

# State: nanobot

## Current Position

Milestone v0.4.0 Lean Runtime Refactor — IN PROGRESS (pivoted)
Phase: 09 (Runtime Mode Spine) — Waves 0 and 1 complete (2 of 5 plans done)
Current Plan: 3
Total Plans in Phase: 5
Status: RuntimeMode type landed as parallel structure; Wave 2 ready to migrate derivations in agent_core.rs::build_swappable_core
Progress: [████░░░░░░] 40%
Last activity: 2026-04-20

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-21)

**Core value:** Personal AI substrate across any model backend with persistent, adaptive memory
**Current focus:** Typed runtime boundaries (Phase 09) and smaller orchestration seams (Phase 10). In-process training and MLX/candle compute are no longer part of this tree — they live in `~/Dev/higgs` and are consumed via the Higgs sidecar.

## Accumulated Context

### Pivot (2026-03-25 → 2026-04-03)

- **Decision:** cut all training from nanobot; move to `~/Dev/higgs` with a train endpoint.
- **`d3f7a1f`:** Higgs managed sidecar (`src/higgs.rs` 307 LOC, auto-start, PID tracking, REPL lifecycle).
- **`1f9e1d5`:** 86,856-line prune across 64 files — training modules, MLX provider/server, ANE compute, candle MoE, eval suite, router adapter, factored vocab, Feishu channel all deleted. SOLID/DRY audit H1/H7/M1/M5/M6/M7 fixed.
- **Phase consequences:** 11a (SharedLoraState), 11b (ModelDef), 11c (MLX boundary), 12 (in-process MoE) all cancelled — their subject matter no longer exists in-tree.

### Still true from pre-pivot

- 6 typed contracts established (PromptContract, ToolGate, MemoryLadder, LearnLoop, ParserCanon, LaneSplit)
- 3 low-severity integration gaps deferred (ParsedAction consumers, GATE-02 config plumbing, LearnProfile/ParserProfile consumers)
- v0.2.0 State-Driven Architecture: phase 06 planned (4 waves) but not executed, deferred
- **CACHE-01/CACHE-02 shipped** (a74ac78): stable prefix ordering enables oMLX/vLLM prefix cache hits
- **Session hardened**: token budget in filter_history (7211010), overflow/orphan guards (9910684), stale auto-expire (d936b96)
- **LCM fix**: system prompt excluded from compaction threshold (bbaf7be)
- **Two token counters**: `token_budget.rs` uses tiktoken (accurate), `filters.rs` uses chars/4 (drifts 60%) — carried to v0.5.0 as PERF-02
- Streaming path has `llm_stream_started` with `ttfb_ms` but NO completion telemetry — carried to v0.5.0 as PERF-01
- Phase 06 state-driven refactor audit identified `is_local` cascades as the best refactor leverage — this is now Phase 09's target
- v0.4.0 intentionally prioritizes seam extraction and behavior preservation over broad cleanup

### Post-pivot hotspot sizes

- `agent_loop.rs` 845 LOC (Phase 10 target)
- `agent_core.rs` 772 LOC (Phase 09 target)
- `agent_shared.rs` 1,799 LOC (Phase 10 target)
- `providers/mlx.rs` — **deleted** (was 1,833 LOC pre-pivot)

### Phase 09 progress (active)

- **Wave 0 (commits `5c4fa7d`, `4c7d652`, `dbd68f4`):** Inline unit tests pinning the 11 + 5 `is_local` branch reads in `agent_shared.rs` / `agent_heuristics.rs`, plus 2 `_cloud` fixture variants in `agent_loop_tests.rs` that rebalance the 8:2 local/cloud asymmetry to 8:4. SUMMARY at `.planning/phases/09-runtime-mode-spine/09-00-SUMMARY.md`.
- **Wave 1 (commit `7d6d2d3`):** `RuntimeMode` enum + 9 derivation methods + 27 invariant tests in `src/agent/runtime_mode.rs`. Parallel-rollout foundation; zero production callsite migrated.
- **Open decisions from Wave 1:**
  - Plan text referenced `LocalProtocolMode`; real type is `LocalReplayMode`. Real name used throughout. Wave 2 can assume the existing enum.
  - Plan text pointed `ModelCapabilities` at `config::schema`; real path is `agent::model_capabilities`. Real path used. Fixture builders construct the full struct (caps contain `thinking`, `needs_native_lms_api`, `strict_alternation`, `reader_tier`, `parser` fields the plan did not enumerate).
  - Env-var tests serialise on a module-local `Mutex` — any future env-coupled test in this module should follow the same `lock_env_cleared()` pattern.
- **Wave 2 next:** migrate memory-provider / delegation-provider construction in `agent_core.rs::build_swappable_core` to consult `RuntimeMode` instead of `is_local`.

### v0.5.0 readiness

- Plan drafted at `.planning/self-evolving-harness-plan.md` (11 S-phases, 2026-04-20).
- Dependencies on v0.4: S01 + S02 need Phase 09's runtime descriptor; S04 benefits from Phase 10's event extraction.
- Do not start S-phases until v0.4.0 completes.
