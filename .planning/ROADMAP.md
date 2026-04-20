# Roadmap: v0.4.0 Lean Runtime Refactor

**Created:** 2026-03-21
**Updated:** 2026-04-20 (pivot to Higgs sidecar; 11a/11b/11c/12 cancelled)
**Phases:** 3 (1 retroactive shipped + 2 active)
**Requirements:** 6 mapped, 0 unmapped

## Phase Overview

| # | Phase | Goal | Requirements | Status | Depends On |
|---|-------|------|--------------|--------|------------|
| 08 | Prune & Externalize | Delete in-process training/MLX/candle; adopt Higgs as managed sidecar for local inference | PRUNE-01, BACK-01 | Shipped (retroactive) | — |
| 09 | 2/5 | In Progress|  | Pending | Phase 08 |
| 10 | Turn Orchestration Seams | Split agent-loop and shared-turn dispatch into explicit event and helper units | LOOP-01, TURN-01 | Pending | Phase 09 |

**Cancelled (see RETROSPECTIVE):** 11a Shared LoRA State, 11b Model Definition Extraction, 11c Local Backend Boundary Cleanup, 12 In-Process MoE — subject matter deleted by Phase 08.

## Phase 08: Prune & Externalize (retroactive)

**Goal:** Document the mid-milestone pivot as a shipped phase. Training and local inference moved to an external Higgs sidecar; ~86k lines of in-process compute code deleted across 64 files.

**Requirements:** PRUNE-01, BACK-01

**Why this phase exists:** The original v0.4.0 roadmap was built around a two-backend (ANE+MLX) model that we decided to collapse to one (Higgs). Phase 08 captures the two commits that executed the pivot so the milestone audit has a real anchor instead of six "Pending" rows.

### Shipped Commits

1. **`d3f7a1f` (2026-03-25)** — Higgs managed sidecar. `src/higgs.rs` (307 LOC) auto-starts Higgs on port 8091 when `localBackend == "higgs"`; PID tracking, OOM safety, REPL lifecycle, `/l` toggle, `/m` model switch, `/adapt` block.
2. **`1f9e1d5` (2026-04-03)** — ~86k-line prune (86,856 deletions across 64 files). ANE, MLX, candle, eval, training, realtime, router-adapter, factored-vocab, Feishu all deleted. SOLID/DRY audit items H1/H7/M1/M5/M6/M7 fixed alongside.

### Success Criteria (verified in retrospect)

1. Local inference works end-to-end through the Higgs sidecar on this machine — verified via existing REPL smoke.
2. No references to deleted modules remain in `src/` — verified by `cargo build --release` green on the branch.
3. Higgs lifecycle survives nanobot session end (persistent daemon) — verified via PID-file inspection and two-session startup test.

### Status

Shipped on branch `refactoring/maximum-speed-with-less-code`. VERIFICATION.md will be written as part of closing v0.4.0 (Step 2 of the close plan).

---

## Phase 09: Runtime Mode Spine

**Goal:** Replace boolean-driven runtime decisions with a typed runtime descriptor and make the shared core expose clearer, smaller seams.

**Requirements:** MODE-01, MODE-02, CORE-01

**Why this phase first (post-pivot):** With MLX deleted, the `is_local` predicate is effectively "is Higgs configured." The typed descriptor gets simpler — runtime mode is now a two-variant enum (Cloud / LocalHiggs) plus the narrow cluster/remote variants — but the refactor target (removing scattered `is_local` branches from `agent_core`, `context`, and REPL command handlers) is unchanged.

### Success Criteria

1. Core build paths resolve memory provider, delegation provider, token reserve, and context defaults from one typed runtime descriptor
2. `agent_core.rs` no longer spreads derived local-vs-cloud decisions across repeated `is_local` branches in the core construction path
3. `SwappableCore` exposes a narrower stable surface for runtime-derived state, with callers using explicit helpers or accessors instead of ad hoc mutation
4. Existing local (Higgs) and cloud behavior remains unchanged in tests and smoke flows

### Key Files

- `src/agent/agent_core.rs`
- `src/agent/context.rs`
- `src/repl/cmd_read.rs`
- `src/agent/mod.rs` or a new `src/agent/runtime_mode.rs`
- `src/higgs.rs` (check that Higgs-backend predicates live in the new descriptor, not scattered)

### Plans

5 plans in 5 waves (sequential — each wave depends on the previous):

- [x] `00-wave-0-coverage-PLAN.md` — Wave 0 coverage safety net (agent_shared + agent_heuristics tests, cloud-path fixture rebalance) — commits `5c4fa7d`, `4c7d652`
- [x] `01-runtime-mode-type-PLAN.md` — Wave 1 introduce `RuntimeMode` enum + derivation methods + invariant tests — commit `7d6d2d3`
- [ ] `02-migrate-derivations-PLAN.md` — Wave 2 migrate 4 core-construction derivations (memory provider, reserve cap, context defaults, budget)
- [ ] `03-remove-is-local-PLAN.md` — Wave 3 migrate all 33 `is_local` reads to `mode()` accessor + 3-way smoke checkpoint
- [ ] `04-final-proof-PLAN.md` — Wave 4 delete fields; rustc as Nyquist filter; phase sign-off checkpoint

### Risks

- Runtime selection touches widely shared construction code
- Narrowing `SwappableCore` can expose hidden coupling in command handlers and turn setup

---

## Phase 10: Turn Orchestration Seams

**Goal:** Split oversized orchestration logic into explicit event classification and dedicated shared-turn helpers without changing turn behavior.

**Requirements:** LOOP-01, TURN-01

**Why this phase second:** Once runtime mode decisions are centralized, the turn loop can be simplified with much lower risk.

### Success Criteria

1. `agent_loop.rs` classifies inbound work into explicit event types before dispatch
2. Tool source selection and LCM engine load state move out of inline fallback chains into dedicated helpers or types
3. Existing trio, local, and cloud turn behavior remains behaviorally identical
4. New focused tests cover event classification and shared-turn decision helpers

### Key Files

- `src/agent/agent_loop.rs`
- `src/agent/agent_shared.rs`
- `src/agent/tool_wiring.rs`
- `src/agent/lcm.rs`

### Risks

- Event extraction can accidentally change ordering or batching behavior
- Shared-turn helpers may uncover assumptions currently encoded only by call order

---

## Coverage

| Requirement | Phase | Status |
|-------------|-------|--------|
| PRUNE-01 | Phase 08 | Shipped |
| BACK-01 | Phase 08 | Shipped |
| MODE-01 | Phase 09 | Pending |
| MODE-02 | Phase 09 | Pending |
| CORE-01 | Phase 09 | Pending |
| LOOP-01 | Phase 10 | Pending |
| TURN-01 | Phase 10 | Pending |

**v0.4.0 requirements:** 7 total | **Shipped:** 2 | **Pending:** 5 | **Unmapped:** 0
**Cancelled mid-milestone:** LORA-01, MODEL-01, MLX-01, MLX-02, MOE-01

---
*Roadmap created: 2026-03-21*
*Updated: 2026-03-21 — added phases 11a (SharedLoraState), 11b (ModelDef), restructured 11 as 11c, added conditional phase 12 (MoE)*
*Updated: 2026-04-20 — Higgs pivot: 11a/11b/11c/12 cancelled; retroactive Phase 08 (Prune & Externalize) added; requirements table reflects 2 shipped + 5 pending*
