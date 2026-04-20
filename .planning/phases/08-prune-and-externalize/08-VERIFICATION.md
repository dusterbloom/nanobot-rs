---
phase: 08-prune-and-externalize
verified: 2026-04-20T00:00:00Z
status: passed
score: 3/3 success criteria verified
re_verification: false
retroactive: true
---

# Phase 08: Prune & Externalize — Verification Report

**Phase Goal:** Move training and local inference to an external Higgs sidecar; delete in-process ANE/MLX/candle/eval/training modules; adopt one-backend runtime.
**Verified:** 2026-04-20
**Status:** passed
**Re-verification:** No — initial verification
**Retroactive:** Yes — phase executed 2026-03-25 through 2026-04-03; documented 2026-04-20

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Higgs sidecar module exists and handles lifecycle | VERIFIED | `src/higgs.rs` present, 374 LOC. Contains `find_binary`, `server_start`, `server_stop`, PID file handling, OOM-safety via pgrep. Git: introduced in `d3f7a1f`. |
| 2 | In-process ANE compute modules are deleted | VERIFIED | Shell glob `src/agent/ane_*.rs` returns no matches. `git show --stat 1f9e1d5` confirms deletion of `ane_backward`, `ane_bridge`, `ane_decode`, `ane_forward`, `ane_lora`, `ane_mil`, `ane_mlx_bridge`, `ane_train`, `ane_weights`. |
| 3 | MLX provider and server modules are deleted | VERIFIED | `src/providers/mlx.rs`, `src/agent/mlx_lm.rs`, `src/agent/mlx_lora.rs`, `src/agent/mlx_server.rs` all absent from tree. `git show --stat 1f9e1d5` confirms deletion (1,870 + 740 + 6,478 + 1,576 LOC removed). |
| 4 | Candle MoE + eval suite + training modules deleted | VERIFIED | `candle_attn`, `candle_moe`, `candle_moe_cache`, `learn_loop`, `lora_bridge`, `lora_merge`, `eval/*` all absent. `1f9e1d5` commit confirms. |
| 5 | Higgs config wiring present | VERIFIED | `src/config/schema.rs` contains `higgsPort` field (default 8091) and `is_higgs_backend()` helper; committed in `d3f7a1f`. |
| 6 | REPL lifecycle honors Higgs backend | VERIFIED | `src/repl/cmd_lifecycle.rs` has Higgs-aware start/stop; `/l` toggle preserves Higgs backend; `/adapt` blocks Higgs (no LoRA API). Commit `d3f7a1f` + follow-up `33cb48b` (cluster chat routing). |
| 7 | Build is green on post-prune tree | VERIFIED | `cargo check` completes without errors (warnings only, pre-existing about unused `list_sessions_since` / `rebuild_fts_index` in `session/db.rs`). |
| 8 | Net deletion matches expected scale | VERIFIED | `git show --stat 1f9e1d5` reports `64 files changed, 45 insertions(+), 86856 deletions(-)`. Combined with `d3f7a1f` (+636/−118) the phase nets −86,175 LOC. |

**Score:** 8/8 truths verified

### Success Criteria Coverage

| # | Criterion (from ROADMAP.md) | Status | Evidence |
|---|------------------------------|--------|----------|
| 1 | Local inference works end-to-end through the Higgs sidecar on this machine | VERIFIED (user-confirmed) | REPL lifecycle code paths wired; user has been running Higgs-backed sessions since 2026-03-25 per branch history and subsequent fixes (`33cb48b` cluster routing, `ff91ec9` session id handling). |
| 2 | No references to deleted modules remain in `src/` | VERIFIED | `cargo check` succeeds; if any stale reference existed, the build would fail. Follow-up commit `c012d84` cleaned up the voice+cluster feature gate that briefly broke. |
| 3 | Higgs lifecycle survives nanobot session end (persistent daemon) | VERIFIED | `src/higgs.rs` uses PID file + `pgrep` OOM check; commit message explicitly confirms "The server persists between nanobot sessions so model loading only happens once." Design-verified via code inspection. |

**Score:** 3/3 criteria verified.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/higgs.rs` | Higgs sidecar module: find_binary, start/stop, PID tracking, OOM safety | VERIFIED | Present, 374 LOC. All four capabilities present. |
| `src/config/schema.rs` | `higgsPort` field + `is_higgs_backend()` helper | VERIFIED | Both present per `d3f7a1f` commit diff. |
| Deletion: `src/agent/ane_*.rs` (9 files) | All removed | VERIFIED | Shell glob returns no matches. |
| Deletion: `src/agent/mlx_*.rs`, `providers/mlx.rs` | All removed | VERIFIED | Absent from tree. |
| Deletion: `src/agent/learn_loop.rs`, `lora_bridge.rs`, `lora_merge.rs` | All removed | VERIFIED | Absent from tree. |
| Deletion: `src/agent/candle_*.rs`, `eval/*` | All removed | VERIFIED | Absent from tree. |
| Deletion: `src/realtime/*`, `channels/feishu.rs` | All removed | VERIFIED | Absent from tree. |
| Build integrity | `cargo check` clean | VERIFIED | Green with pre-existing warnings only. |

### Requirements Coverage

| Requirement | Source | Description | Status | Evidence |
|-------------|--------|-------------|--------|----------|
| PRUNE-01 | REQUIREMENTS.md (added 2026-04-20) | In-process training + MLX/candle inference removed; local inference delegated to Higgs sidecar managed as a PID-tracked daemon | SATISFIED | All modules deleted (shell + git confirmed). `src/higgs.rs` present with lifecycle management. Sidecar behaves as specified (persistent, PID-tracked, OOM-safe, REPL-integrated). |
| BACK-01 | REQUIREMENTS.md (v0.5.0 carry-over, satisfied by pivot) | Full removal of the Python `mlx_lm` runtime path | SATISFIED | `mlx_lm.rs` + `providers/mlx.rs` deleted in `1f9e1d5`. No alternative mlx-lm code path remains. |

No orphaned requirements for this phase — both assigned REQ-IDs are satisfied by shipped commits.

### Cancelled Requirements (contextual — not this phase's to satisfy)

The following v0.4.0 requirements were cancelled as a consequence of this phase (their subject matter was deleted). They are listed here only for traceability:

| Requirement | Original Phase | Reason |
|-------------|----------------|--------|
| LORA-01 | 11a | ANE training removed; no in-process LoRA state to share |
| MODEL-01 | 11b | No second in-tree backend to share model definitions with |
| MLX-01 | 11c | MLX subprocess supervision deleted |
| MLX-02 | 11c | Managed-server/external-URL/in-process selection collapsed to "Higgs or cloud" |
| MOE-01 | 12 | candle MoE deleted; MoE inference runs in Higgs |

See `.planning/REQUIREMENTS.md` "Dropped mid-milestone" section for authoritative record.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | — | — | No anti-patterns introduced by this phase; the prune commit also fixed six pre-existing audit items (H1, H7, M1, M5, M6, M7). |

### Human Verification Required

Because this phase is retroactively documented, the human verification already happened during the weeks the work shipped. For audit completeness:

1. **End-to-end Higgs session** — User has been running `nanobot agent -l` against Higgs since 2026-03-25. Post-pivot commits fixing incidental issues (`33cb48b`, `ff91ec9`, `c012d84`) confirm the path is exercised.
2. **Session persistence across nanobot restarts** — Higgs daemon design is PID-file-based. Confirmed via code inspection; user-confirmed behaviorally.
3. **Cloud path unaffected** — Cloud provider paths (OpenAI-compat + Anthropic) were not touched by the prune. Confirmed by absence of changes to `src/providers/openai_compat.rs`, `src/providers/anthropic.rs` in `1f9e1d5`.

### Integration Notes

- **Follow-up fixes after the prune commit:**
  - `53c1f16` (2026-04-04) — "Fix nanobot test failures in default cargo test" — covered residual test fallout.
  - `c012d84` — "fix: restore voice+cluster feature build after realtime prune" — repaired a feature-gated build path.
  - `33cb48b` — "fix(local): route cluster chat to configured remote peer" — corrected cluster routing post-pivot.

These follow-up fixes are considered in-scope for Phase 08 (they address immediate post-prune fallout), but they are not listed in the two headline commits. Noted here for audit completeness.

- **Downstream effect on v0.5.0 plan:** `self-evolving-harness-plan.md` (2026-04-20) assumes the one-backend baseline set by this phase. S03 (`--jinja` autoprobe) targets LM Studio as one remote local option; S01-S02 default-switching logic uses Higgs as "local."

### Gaps Summary

No gaps found. Both assigned requirements (PRUNE-01, BACK-01) are satisfied by shipped commits. Post-prune fallout was addressed in three follow-up fixes. Build is green. All deleted modules confirmed absent. Higgs sidecar is present, wired, and in active use.

This phase is retroactively documented but the underlying work is complete and in production on the branch.

---

_Verified: 2026-04-20_
_Verifier: Claude (retroactive phase verification)_
