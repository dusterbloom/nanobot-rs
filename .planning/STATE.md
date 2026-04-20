---
gsd_state_version: 1.0
milestone: v0.4.0
milestone_name: Lean Runtime Refactor
status: in_progress
stopped_at: pivoted to Higgs sidecar; retroactive Phase 08 documented; ready to plan Phase 09
last_updated: "2026-04-20"
last_activity: 2026-04-20 — Higgs pivot captured in RETROSPECTIVE; REQUIREMENTS/ROADMAP pruned (11a/11b/11c/12 cancelled); Phase 08 Prune & Externalize added retroactively
progress:
  total_phases: 3
  completed_phases: 1
  total_plans: 0
  completed_plans: 0
---

# State: nanobot

## Current Position

Milestone v0.4.0 Lean Runtime Refactor — IN PROGRESS (pivoted)
Phase: 09 (Runtime Mode Spine) — research done, ready to plan
Status: v0.4.0 scope halved by Higgs pivot; 2 of 6 original phases remain active (09, 10); Phase 08 retroactively captures the pivot work (Higgs sidecar + ~86k-line prune)
Last activity: 2026-04-20 — retrospective + roadmap/requirements updates to reflect the pivot; close-v0.4 plan ready

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

### v0.5.0 readiness

- Plan drafted at `.planning/self-evolving-harness-plan.md` (11 S-phases, 2026-04-20).
- Dependencies on v0.4: S01 + S02 need Phase 09's runtime descriptor; S04 benefits from Phase 10's event extraction.
- Do not start S-phases until v0.4.0 completes.
