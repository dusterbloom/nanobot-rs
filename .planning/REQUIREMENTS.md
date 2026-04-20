# Requirements: nanobot

**Defined:** 2026-03-21
**Updated:** 2026-04-20 — pivoted mid-milestone; training/inference externalized to Higgs; LORA/MODEL/MLX/MOE requirements dropped
**Core Value:** The system works as a personal AI substrate across any model backend -- local or cloud, small or large -- with memory that persists and adapts across sessions.

## v0.4.0 Requirements

Requirements for the Lean Runtime Refactor milestone. Each maps to one roadmap phase. The milestone pivoted on 2026-03-25: training and local inference were externalized to a Higgs sidecar (`~/Dev/higgs`), so the original two-backend seam work was replaced by a prune-and-externalize phase. See `RETROSPECTIVE.md` for the full pivot narrative.

### Prune & Externalize (shipped)

- [x] **PRUNE-01**: In-process training (ANE/MLX LoRA), MLX inference provider/server, candle MoE, eval suite, and related dead modules are removed from the nanobot tree. Local inference is delegated to an externally-owned Higgs sidecar via HTTP, managed as a PID-tracked daemon with lifecycle parity to the prior in-process backend. *(Shipped via `d3f7a1f` Higgs sidecar + `1f9e1d5` ~50k-line prune.)*

### Runtime Mode

- [x] **MODE-01**: Operator can switch between local and cloud runtimes without leaving provider selection, token reserve, or context-budget state inconsistent
- [x] **MODE-02**: Runtime-specific defaults (context window, prompt cap, delegation model, memory provider) resolve from one typed runtime descriptor instead of duplicated boolean branches
- [x] **CORE-01**: Runtime-derived state is exposed through a narrower stable `SwappableCore` surface so commands and turn handlers do not mutate it ad hoc

### Turn Orchestration

- [ ] **LOOP-01**: Agent loop classifies inbound work into explicit event types before dispatching turn handling
- [ ] **TURN-01**: Tool source selection and LCM engine load state are resolved by dedicated helpers or types with focused tests rather than inline fallback chains

### Dropped mid-milestone (see RETROSPECTIVE)

The following requirements were cancelled on 2026-04-03 when the Higgs pivot deleted their subject matter:

- ~~**LORA-01**~~: Cancelled — ANE training removed from nanobot; no in-process LoRA state to share. Equivalent capability now lives in Higgs.
- ~~**MODEL-01**~~: Cancelled — no second compute backend in-tree to share a model definition with.
- ~~**MLX-01**~~: Superseded by PRUNE-01 — MLX subprocess supervision deleted entirely; the boundary is now the Higgs HTTP interface.
- ~~**MLX-02**~~: Superseded by PRUNE-01 — managed-server/external-URL/in-process selection collapsed to "Higgs or cloud."
- ~~**MOE-01**~~: Cancelled — candle MoE deleted; MoE inference runs in Higgs, not nanobot.

## v0.5.0 Requirements

Deferred until Phases 09 and 10 close. The detailed plan lives in `.planning/self-evolving-harness-plan.md` (11 S-phases targeting local-model harness parity + self-evolution).

### Performance Carry-Over

- **PERF-01**: Streaming path emits end-of-stream telemetry with TTFT, elapsed time, and tok/s
- **PERF-02**: Session and context budgeting use actual token counts instead of chars/4 heuristics

### Broader Simplification

- **MODE-03**: Remaining bool-flag state machines and duplicated fallback chains outside the runtime spine are simplified after core runtime boundaries are stable

### Shipped in v0.4.0 via pivot

- [x] **BACK-01**: Full removal of the Python `mlx_lm` runtime path. *(Shipped via `1f9e1d5` — MLX provider/server deleted; local inference is Higgs.)*

## Out of Scope

| Feature | Reason |
|---------|--------|
| New user-facing features | This milestone is strictly about refactoring runtime internals |
| Whole-codebase cleanup sweep | Lean scope favors the highest-leverage seams only |
| Realtime voice completion | Separate feature track, not part of this refactor milestone |
| TTFT/performance tuning | Deferred until runtime seams are smaller and easier to instrument |
| In-process ANE training | Moved to Higgs (`~/Dev/higgs`), exposed via train endpoint |
| In-tree MLX / candle compute | Deleted; owned by Higgs |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PRUNE-01 | Phase 08 (retroactive) | Shipped |
| MODE-01 | Phase 09 | Complete |
| MODE-02 | Phase 09 | Complete |
| CORE-01 | Phase 09 | Complete |
| LOOP-01 | Phase 10 | Pending |
| TURN-01 | Phase 10 | Pending |
| BACK-01 | Phase 08 (retroactive) | Shipped |

**Coverage:**
- v0.4.0 active requirements: 6 (1 shipped via pivot, 5 pending)
- Dropped mid-milestone: 5 (LORA-01, MODEL-01, MLX-01, MLX-02, MOE-01)
- Mapped to phases: 6
- Unmapped: 0

---
*Requirements defined: 2026-03-21*
*Updated: 2026-03-21 -- added LORA-01, MODEL-01, MOE-01; restructured Phase 11 into 11a/11b/11c; added ANE/mlx-lm to Out of Scope*
*Updated: 2026-04-20 -- Higgs pivot; added PRUNE-01; dropped LORA/MODEL/MLX/MOE; BACK-01 marked shipped*
