---
gsd_state_version: 1.0
milestone: v0.4
milestone_name: milestone
current_plan: 4
status: "Phase 09 Wave 3 COMPLETE — 0 swappable().is_local reads remain in src/. Only Wave 4 (delete now-dead is_local/local_tool_mode fields + provenance_warning_role bool) remains. Branch has grown ~6 features beyond the refactor; priority is now LANDING — 208 commits ahead of main, none merged since 2026-03-15."
last_updated: "2026-06-19T00:00:00.000Z"
last_activity: 2026-06-19
progress:
  total_phases: 3
  completed_phases: 0
  total_plans: 5
  completed_plans: 4
  percent: 90
---

# State: nanobot

## Current Position

Milestone v0.4.0 Lean Runtime Refactor — refactor essentially DONE; branch became a feature drop
Phase: 09 (Runtime Mode Spine) — Waves 0–3 complete (Wave 4 field-deletion + summaries remain)
Current Plan: 04 (final-proof)
Total Plans in Phase: 5
Status: Wave 3 complete in code (commits 3b83b0d, e7a3aef, 6a9cc8d) — `rg 'swappable().is_local' src/` = 0. Wave 4 (delete dead fields, let rustc prove zero readers) is the only remainder. Tech-debt Commits A/B/C also shipped (8d8b5f0, 878368e, etc.).
Progress: [█████████░] 90%
Last activity: 2026-06-19
Build: green · Lib tests: 2057 passed / 0 failed (2026-06-19)

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
- **Wave 2 (commits `46b6d61`, `83b086c`):** SwappableCore.mode field + `mode()` accessor added alongside `is_local`; 4 roadmap-named derivations migrated (context constructor, budget scaling, memory provider, reserve cap). `resolve_memory_provider` free function extracted (~50 lines out of build_swappable_core). `match mode` occurrences in agent_core.rs: 4. 10 new invariant tests (2110 lib tests passing, zero new warnings). Cloud smoke green; Higgs smoke deferred (no sidecar on this machine). SUMMARY at `.planning/phases/09-runtime-mode-spine/09-02-SUMMARY.md`.
- **Open decisions from Wave 2:**
  - Branch 5 (delegation provider) evaluated NOT APPLICABLE — tool-runner resolution is driven by `tool_delegation.enabled` + `delegation_provider.is_some()`, not `is_local`. Already RuntimeMode-agnostic.
  - `context.local_prompt_mode = is_local` reader NOT migrated — Wave 3 scope.
  - `is_local: bool` remains canonical on SwappableCore; both fields reconciled at construction via `debug_assert_eq!(matches!(mode, Local {..}), is_local)`. Wave 3 reader migration relies on this invariant.
  - Higgs smoke transcript still owed — user must configure sidecar and re-run before Wave 4's "final proof" phase.
- **Wave 3 DONE** (commits `3b83b0d`, `e7a3aef`, `6a9cc8d`): all ~33 downstream `is_local` readers migrated to `core.mode()` dispatch. `rg 'swappable().is_local' src/` returns 0. Note: `09-03-SUMMARY.md` was never written — owed (Day 2). The mandated 3-way smoke (Cloud + Higgs + cluster-remote) was also never captured — owed (Day 5).
- **Wave 4 remaining:** delete `is_local: bool` + `local_tool_mode` from `SwappableCore` (still present at `agent_core.rs:63`), migrate `provenance_warning_role(is_local: bool)` at `agent_core.rs:374` to take `&RuntimeMode`. Single rustc-driven commit — compiler proves zero readers.

## Reality reconciliation — 2026-06-19

STATE.md had drifted ~2 months. Ground truth as of this session:

- **The documented v0.4 refactor is essentially done.** Phase 09 Waves 0–3 complete; TECH_DEBT_AUDIT Commits A/B/C shipped (parsers/ deleted, dead-code sweep, truncation helpers consolidated). Only Wave 4 (cheap) + summaries remain.
- **The branch outgrew its name.** `refactoring/maximum-speed-with-less-code` now also carries: full ratatui TUI rewrite (`tui_app/` ≈6.7k LOC), voice mode, trio/Apple-FM routing, Higgs sidecar integration, local grammar-constrained tool-calling (Tier 1/2), query-aware skills + lean context, tool observability, and (committed this session) native TUI session snapshot/resume/export + vision detection.
- **Headline risk = unlanded value.** 208 commits ahead of main; main frozen at `4252a4f` (2026-03-15). Net −20k LOC. Build green, 2057 lib tests green → the branch is landable today.
- **New code reintroduced gate debt:** `./quality-sentinel.sh` = 11 hits (3 G1 mutable bools, 8 G5 else-if chains), all in new `tui_app/` + `cmd_mutation`; `tui_app/app.rs` is 4,924 LOC (larger than any old refactor target).

### Week plan (set 2026-06-19)
- **Day 1 [DONE]:** committed WIP feature (`bc62b77`), GitNexus doc refresh (`41bec80`), ignored tooling cruft, moved scratch/SESSION_RESULTS into `.planning/`, this STATE rewrite.
- **Day 2:** Phase 09 Wave 4 (delete dead fields) + write `09-03/09-04-SUMMARY.md`.
- **Day 3:** clear the 11 sentinel violations; carve natural seams out of `app.rs` (no full SRP rewrite).
- **Day 4:** LAND IT — merge strategy decision (rec: squash-drop whole branch vs stack-split), rebase onto main, full suite + release build.
- **Day 5:** 3-way smoke (Cloud + Higgs + cluster-remote) with transcripts; refresh GitNexus index; buffer.
- **Deferred (post-merge):** tech-debt D/E/F (require_str, SSE dedup, chat_stream extraction); v0.5.0 self-evolving-harness (blocked on v0.4 landing).

### v0.5.0 readiness

- Plan drafted at `.planning/self-evolving-harness-plan.md` (11 S-phases, 2026-04-20).
- Dependencies on v0.4: S01 + S02 need Phase 09's runtime descriptor; S04 benefits from Phase 10's event extraction.
- Do not start S-phases until v0.4.0 completes.
