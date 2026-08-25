---
gsd_state_version: 1.0
milestone: v0.4
milestone_name: milestone
current_plan: 4
status: "Phase 09 COMPLETE (Waves 0–4). Turn replay shipped. Error protocol Phase 0–1 done, Phase 2 partial (18 tool files unmigrated + 4 legacy deletions + trait flip). cua tool shipped with vision injection. Branch is 60 ahead / 22 BEHIND a diverged main (merge-base d7a801c, 2026-08-02) — priority is LANDING via squash-merge. See PLAN.md."
last_updated: "2026-08-25T00:00:00.000Z"
last_activity: 2026-08-25
progress:
  total_phases: 3
  completed_phases: 0
  total_plans: 5
  completed_plans: 5
  percent: 95
---

# State: nanobot

## Current Position

Milestone v0.4 Lean Runtime Refactor — refactor DONE in code; branch is a feature drop awaiting landing
Phase: 09 (Runtime Mode Spine) — COMPLETE, Waves 0–4
Current Plan: see PLAN.md (root) — landing + post-landing protocol, all pending
Status: Phase 09 Wave 4 shipped (`local_tool_mode` and `provenance_warning_role` deleted; zero
`swappable().is_local` code reads — only `migrated from swappable().is_local` comments remain).
Dead `parsers/` module deleted. Exact Turn Replay complete (PLAN.md items all done, including
review fixes). Error protocol: Phase 0–1 done (`ToolResult = Result<ToolOutput, ToolError>` in
`src/agent/tools/base.rs`; `execute_typed` migration seam live; `ToolError` used in 7 tool files);
Phase 2 codemod partial — exact remainder below. `cua` tool shipped (screenshot + vision turn
injection, `src/agent/tools/cua.rs`). Owed from June plan: `09-03/09-04-SUMMARY.md` never written;
3-way smoke transcripts never captured.
Progress: [█████████▌] 95%
Last activity: 2026-08-25
Build: last verified green · 2057 lib tests (2026-06-19) — re-verify at landing gate

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-21)

**Core value:** Personal AI substrate across any model backend with persistent, adaptive memory
**Current focus:** landing the branch (squash onto main), then error-protocol Phases 2–4 and a
fresh tech-debt audit. In-process training and MLX/candle compute are no longer part of this tree —
they live in `~/Dev/higgs` and are consumed via the Higgs sidecar.

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
- Phase 06 state-driven refactor audit identified `is_local` cascades as the best refactor leverage — delivered by Phase 09
- v0.4.0 intentionally prioritizes seam extraction and behavior preservation over broad cleanup

### Post-pivot hotspot sizes (re-measured 2026-08-25)

- `agent_loop.rs` and `agent_shared.rs` no longer exist — split into `src/agent/agent_loop/` module
  (20,204 LOC total: tests.rs 9,864 · shared.rs 5,946 · response.rs 1,643 · heuristics.rs 867 · mod.rs 747)
- `agent_core.rs` 3,351 LOC (grew from 772; absorbed Wave-2/3 migration surface)
- TUI moved `tui_app/` → `src/tui_app/` (3 files, 8,651 LOC; `app.rs` 6,709 — largest file in tree)
- `providers/mlx.rs` — deleted (was 1,833 LOC pre-pivot)

### Phase 09 (COMPLETE)

- **Wave 0 (commits `5c4fa7d`, `4c7d652`, `dbd68f4`):** tests pinning the 11 + 5 `is_local` branch reads + 2 `_cloud` fixture variants. SUMMARY at `09-00-SUMMARY.md`.
- **Wave 1 (commit `7d6d2d3`):** `RuntimeMode` enum + 9 derivation methods + 27 invariant tests in `src/agent/runtime_mode.rs`. Real type is `LocalReplayMode` (not plan-text `LocalProtocolMode`); `ModelCapabilities` lives at `agent::model_capabilities`. SUMMARY at `09-01-SUMMARY.md`.
- **Wave 2 (commits `46b6d61`, `83b086c`):** SwappableCore.mode field + `mode()` accessor; 4 roadmap-named derivations migrated; `resolve_memory_provider` extracted. Branch 5 (delegation provider) N/A — driven by `tool_delegation.enabled`, not `is_local`. SUMMARY at `09-02-SUMMARY.md`.
- **Wave 3 (commits `3b83b0d`, `e7a3aef`, `6a9cc8d`):** all ~33 downstream `is_local` readers migrated to `core.mode()` dispatch. `rg 'swappable().is_local' src/` = 0 code reads (20 `migrated from …` comments remain as breadcrumbs). SUMMARY `09-03-SUMMARY.md` never written — owed.
- **Wave 4 (DONE):** `is_local: bool`, `local_tool_mode`, and `provenance_warning_role` deleted from src/; rustc proved zero readers. SUMMARY `09-04-SUMMARY.md` never written — owed.
- Owed alongside: 3-way smoke transcripts (Cloud + Higgs + cluster-remote) never captured.

### Error protocol (Phase 0–1 done, Phase 2 partial)

- **Phase 0–1 shipped** (~commit `8b1c8f9` area): `pub type ToolResult = Result<ToolOutput, crate::errors::ToolError>` (`base.rs:173`); additive `execute_typed -> ToolResult` seam with `funnel_legacy` default (`base.rs:341`); `execute_with_result`/`execute_with_result_and_context` render into legacy `ToolExecutionResult`. `FinishReason` live in `src/providers/openai_compat.rs`.
- **Phase 2 exact remainder (counted 2026-08-25):** trait methods `execute` and `execute_with_context` in `base.rs` still return `String`. 22 tool files still override `execute -> String`; of those, 4 (message, recall, remember, spawn) already override `execute_typed` and only need their legacy `execute` deleted at trait flip; 18 files need full migration. `code_execution.rs` and `registry.rs` construct `ToolError` without overriding `execute_typed`.
- **Phases 3–4:** not started — finish AFTER squash per owner decision.

## Reality reconciliation — 2026-08-25

- **Branch topology changed since June.** Not "208 commits ahead of a frozen main". Now: 60 commits
  ahead, 22 BEHIND — main advanced to `c6103a2` (2026-08-02, delegated-tool invariants, lease
  accounting, replay verification); merge-base is `d7a801c` (2026-08-02). Squash-merge must
  reconcile those 22 main-side commits; it is not a fast-forward.
- **Everything hard from the June plan happened.** Phase 09 Wave 4 shipped; parsers/ deleted; Exact
  Turn Replay shipped with review fixes (durable compaction on journal failure, fail-soft aux
  lanes, `StreamCancelGuard`, single-query artifact verification, real delegated timings); cua
  tool shipped with vision-capable screenshot injection.
- **Turn bench never ran** — local inference server down. Plan: bench against a reachable cloud
  provider first; restore the local server only if a regression shows.
- **June week-plan superseded** by PLAN.md (root): squash-merge first, error-protocol Phases 2–4
  after, tech-debt audit re-run, cloud turn bench.
- **GitNexus-index numbers from the 2026-06-19 note (gate hits, LOC) are stale**; re-run
  `quality-sentinel.sh` and the audit fresh rather than trusting old counts.

### v0.5.0 readiness

- Plan drafted at `.planning/self-evolving-harness-plan.md` (11 S-phases, 2026-04-20).
- Dependencies on v0.4: S01 + S02 need Phase 09's runtime descriptor (now shipped); S04 benefits from Phase 10's event extraction.
- Do not start S-phases until v0.4.0 lands.
