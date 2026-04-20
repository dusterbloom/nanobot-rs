---
phase: 08-prune-and-externalize
plan: retroactive
subsystem: agent,providers,config,repl
tags: [pivot, prune, higgs, sidecar, external-backend, audit-cleanup]
retroactive: true

# Dependency graph
requires: []
provides:
  - "Higgs managed sidecar for local inference (src/higgs.rs)"
  - "Single-backend runtime — only Higgs (local) and cloud providers remain in-tree"
  - "Removal of in-process ANE compute, MLX provider/server, candle MoE, eval suite, training modules"
  - "SOLID/DRY audit fixes H1/H7/M1/M5/M6/M7 landed alongside the prune"
affects: [v0.4.0 roadmap (11a/11b/11c/12 cancelled), v0.5.0 plan (baseline is one-backend codebase)]

# Tech tracking
tech-stack:
  added:
    - "Higgs HTTP sidecar integration (external process, port 8091)"
  removed:
    - "candle Metal MoE + attention modules"
    - "ANE compute/training kernel surface (GDN, LoRA backward, weights, bridge)"
    - "MLX Python server integration + providers/mlx.rs"
    - "In-tree eval suite (hanoi, haystack, learning, sprint, runner)"
    - "router_adapter, factored_vocab, split_silicon, learn_loop, lora_bridge, lora_merge"
    - "Feishu channel + realtime voice/WebSocket server"
  patterns: [managed-sidecar-lifecycle, pid-tracking, external-backend-boundary]

key-files:
  created:
    - src/higgs.rs
  modified:
    - src/cli/mod.rs
    - src/config/schema.rs
    - src/main.rs
    - src/repl/cmd_cluster.rs
    - src/repl/cmd_lifecycle.rs
    - src/repl/commands.rs
    - src/repl/mod.rs
    - src/tui.rs
  deleted (partial list, full set in 1f9e1d5 commit):
    - src/agent/ane_backward.rs, ane_bridge.rs, ane_decode.rs, ane_forward.rs, ane_lora.rs, ane_mil.rs, ane_mlx_bridge.rs, ane_train.rs, ane_weights.rs
    - src/agent/candle_attn.rs, candle_moe.rs, candle_moe_cache.rs
    - src/agent/eval/* (mod, hanoi, haystack, learning, results, runner, sprint, eval_summary.md)
    - src/agent/learn_loop.rs, lora_bridge.rs, lora_merge.rs, mlx_lm.rs, mlx_lora.rs, mlx_server.rs
    - src/agent/factored_vocab.rs, router_adapter.rs, finalize_response.rs (recreated later)
    - src/providers/mlx.rs
    - src/realtime/* (mod, session, voice_agent, ws_server)
    - src/channels/feishu.rs
    - tests/memory_pipeline_e2e.rs

key-decisions:
  - "Cut in-process training entirely — move to ~/Dev/higgs exposed via train endpoint"
  - "Local inference delegated to Higgs managed sidecar with PID tracking; nanobot is a client, not a host"
  - "One-backend codebase preferred over maintaining two-clean-backends seam (LoRA state, model def, MoE routing) — the seam cost exceeded the value"
  - "Phase 11a/11b/11c/12 of v0.4.0 roadmap cancelled — subject matter deleted"
  - "Higgs persists across nanobot sessions (daemon model loads once)"

patterns-established:
  - "Externalize-and-prune as a refactor primitive — when a subsystem has a clean external implementation, delete the in-process version"
  - "PID-file-driven sidecar lifecycle (template for future managed subprocess integrations)"
  - "OOM-safe spawn via pgrep — refuse to start if Higgs already running"

requirements-completed: [PRUNE-01, BACK-01]

# Metrics
commits: 2
  - d3f7a1f (2026-03-25) — Higgs managed sidecar, 9 files, +636/-118
  - 1f9e1d5 (2026-04-03) — 50k-line prune of dead/obsolete modules, 64 files, +45/-86,856
lines_deleted: 86,856
lines_added: 681
net_change: -86,175
duration: 10 days (2026-03-25 → 2026-04-03)
completed: 2026-04-03
retroactive_documentation_date: 2026-04-20
---

# Phase 08: Prune & Externalize (retroactive)

**Executed the mid-milestone pivot that moved training and local inference to an external Higgs sidecar and deleted ~86k lines of now-redundant in-process compute/training/eval code. Captures as a shipped phase what was originally unplanned.**

## Retroactive Documentation

This phase is documented **after** the work shipped. The original v0.4.0 roadmap (2026-03-21) did not include it — it planned a two-backend (ANE+MLX) seam cleanup across Phases 11a/11b/11c/12. Between 2026-03-25 and 2026-04-03 we decided that cost was misaligned with value, pivoted to a one-backend model (Higgs-owned), and executed the pivot in two commits.

Documenting this as Phase 08 closes the gap between what the roadmap claimed and what the codebase actually shows. Without this phase, the milestone audit would mark five requirements "Pending" against subject matter that no longer exists in-tree.

## Performance

- **Duration:** 10 days
- **Started:** 2026-03-25
- **Completed:** 2026-04-03
- **Commits:** 2 (both on branch `refactoring/maximum-speed-with-less-code`)
- **Files changed:** 73 total (9 + 64)
- **Net lines:** −86,175

## Accomplishments

### Higgs Managed Sidecar (`d3f7a1f`, 2026-03-25)

- New `src/higgs.rs` (307 LOC at commit, 374 LOC as of 2026-04-20) with `find_binary`, `server_start`, `server_stop`, PID tracking, OOM safety via `pgrep` (refuses spawn if Higgs already running).
- Config: added `higgsPort` (default 8091) and `is_higgs_backend()` helper to `src/config/schema.rs`.
- REPL: auto-start on startup and on `/l` toggle, Higgs splash screen, `/m` model switch preserves "higgs" backend, `/l off` preserves it, `/adapt` correctly blocks Higgs (no LoRA API).
- Fixed stale `has_remote_local` that previously prevented Higgs startup.
- Higgs persists between nanobot sessions — model loading only happens once.

### ~86k-Line Prune (`1f9e1d5`, 2026-04-03)

- **Training modules:** `learn_loop.rs` (1,518 LOC), `lora_bridge.rs` (2,048), `lora_merge.rs` (1,461), `ane_train.rs` (1,370), `ane_backward.rs` (5,312), `ane_lora.rs` (5,701), `training_eval` paths — all deleted.
- **MLX backend:** `mlx_lora.rs` (6,478), `mlx_server.rs` (1,576), `mlx_lm.rs` (740), `providers/mlx.rs` (1,870) — all deleted.
- **ANE compute:** `ane_bridge.rs` (1,573), `ane_decode.rs` (7,217), `ane_forward.rs` (9,651), `ane_mil.rs` (11,302), `ane_mlx_bridge.rs` (8,086), `ane_weights.rs` (7,150) — all deleted.
- **Candle:** `candle_attn.rs` (396), `candle_moe.rs` (249), `candle_moe_cache.rs` (298) — all deleted.
- **Eval suite:** `eval/{mod,hanoi,haystack,learning,results,runner,sprint}.rs` + `eval_summary.md` — all deleted.
- **Realtime:** `realtime/{mod,session,voice_agent,ws_server}.rs` — all deleted.
- **Misc:** `split_silicon`, `router_adapter`, `factored_vocab`, `perf_ceiling`, Feishu channel + FeishuConfig — all deleted.

### SOLID/DRY Audit Fixes (landed alongside prune)

- **H1:** Deleted `FeishuConfig` struct, Debug/Default impls, field, all 9 references.
- **H7:** Removed dead `bus_inbound_tx` field from `AgentLoopShared`.
- **M1:** Extracted `ChannelsConfig::enable_exclusive()` replacing 6 copy-pasted blocks.
- **M5:** Extracted `needs_mlx_inprocess()` predicate, fixed missing guards in `voice.rs`.
- **M6+M7:** Removed 3 blanket `#![allow(dead_code)]` from repl modules; deleted dead functions (`fetch_lms_loaded_models`, `lms_model_matches`, `is_first_line`); added targeted `#[cfg_attr]` for feature-gated items; fixed 5 unused imports.

## Files Created/Modified/Deleted

### Created (net new)
- `src/higgs.rs` — Higgs sidecar lifecycle, PID tracking, start/stop/probe.

### Modified
- `src/cli/mod.rs` — Higgs-aware backend selection (+9 lines).
- `src/config/schema.rs` — `higgsPort`, `is_higgs_backend()`, Feishu removal (net −230).
- `src/main.rs` — Dead module imports removed (−243).
- `src/repl/cmd_lifecycle.rs` — Higgs-aware start/stop, `/l` toggle, status display (+307/−~130).
- `src/repl/mod.rs`, `src/repl/commands.rs`, `src/repl/cmd_cluster.rs`, `src/tui.rs` — splash, toggles, display.

### Deleted
- 39+ source files, full list in `1f9e1d5` commit stats; summarized in "Accomplishments" above.

## Decisions Made

1. **Move training to Higgs.** The on-device training work (REINFORCE router training, LoRA backward, chained delta merge) had proven itself but had an 86k-line footprint in a general-purpose agent codebase. Moving it to a dedicated project with a train endpoint separates concerns and lets nanobot stay a thin orchestration layer.
2. **Delete MLX provider/server.** Once Higgs owns local inference, the Python `mlx_lm` runtime path and the in-tree `providers/mlx.rs` (1,870 LOC of subprocess supervision, request queueing, response parsing) became redundant.
3. **Cancel v0.4.0 Phases 11a/11b/11c/12.** Their entire premise (share state between two in-process backends) evaporated with the single-backend decision.
4. **Keep the audit fixes in the same commit.** The prune commit was large anyway; bundling H1/H7/M1/M5/M6/M7 avoided a follow-up audit-cleanup phase and kept the "dead code deleted + patterns repaired" narrative unified.
5. **Document retroactively as Phase 08.** The alternative — leaving it unphased — left ROADMAP.md and REQUIREMENTS.md lying to the milestone audit. Phase 08 closes the gap.

## Deviations from Plan

**N/A** — this phase had no plan. It was an unplanned mid-milestone pivot documented retroactively.

## Issues Encountered

- **Stale `has_remote_local` blocking Higgs startup** — discovered during REPL lifecycle work in `d3f7a1f`. Fixed in the same commit.
- **Voice+cluster feature build broke after realtime prune** — caught later and fixed in `c012d84` (`fix: restore voice+cluster feature build after realtime prune`). Noted here for traceability; full fix lives outside the two Phase 08 commits.

## User Setup Required

- **Higgs binary must be installed** at a discoverable path (`find_binary` searches `~/Dev/higgs/target/release/higgs`, PATH, etc).
- **Config change:** `localBackend: "higgs"` in `~/.nanobot/config.json` to opt into the sidecar.
- No migration needed — existing sessions continue to work; the cloud path is unaffected.

## Next Phase Readiness

- **Phase 09 (Runtime Mode Spine)** proceeds against a simpler tree. The `is_local` predicate is now effectively "is Higgs configured," which should make the typed runtime descriptor cleaner than originally scoped.
- `agent_loop.rs` (845 LOC), `agent_core.rs` (772), `agent_shared.rs` (1,799) remain the Phase 09/10 hotspots — unchanged by the prune.
- v0.5.0 self-evolving-harness-plan has its baseline set: one-backend + external sidecar.

## Self-Check: PASSED

- `src/higgs.rs`: FOUND (374 LOC)
- `src/agent/ane_*.rs`: ALL DELETED (confirmed via shell glob miss)
- `src/agent/mlx_*.rs`, `learn_loop.rs`, `lora_*.rs`: ALL DELETED
- `src/providers/mlx.rs`: DELETED
- `src/agent/eval/`: DELETED
- `src/realtime/`: DELETED
- `src/channels/feishu.rs`: DELETED
- Commit `d3f7a1f`: FOUND on branch `refactoring/maximum-speed-with-less-code`
- Commit `1f9e1d5`: FOUND on branch `refactoring/maximum-speed-with-less-code`
- `cargo check`: GREEN (warnings only, pre-existing)

---
*Phase: 08-prune-and-externalize*
*Completed: 2026-04-03*
*Retroactively documented: 2026-04-20*
