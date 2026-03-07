# Requirements: nanobot

**Defined:** 2026-03-07
**Core Value:** Personal AI substrate across any model backend with persistent, adaptive memory

## v0.1.1 Requirements

Requirements for Contract Architecture milestone. Each maps to roadmap phases.

### Prompt Contract

- [x] **PROMPT-01**: System prompt sections are typed variants of PromptSection enum with fixed ordering
- [x] **PROMPT-02**: Each section has a PromptBudget with allocated tokens, actual tokens, and source
- [ ] **PROMPT-03**: PromptAssembler performs single-pass assembly, replacing all append_to_system_prompt() calls
- [x] **PROMPT-04**: Total prompt cost enforced against context window cap, sections dropped tail-first

### Tool Gate

- [ ] **GATE-01**: Tool definitions filtered by model size class (Small=5 tools, Medium=10, Large=all)
- [ ] **GATE-02**: Explicit toolsFilter config overrides size-class gate

### Memory Ladder

- [ ] **MEM-01**: 5 named memory layers (Scratch, WorkingSession, DurablePersonal, SearchIndex, GroundTruth) with priority ordering
- [ ] **MEM-02**: Memory queries stop pulling from lower-priority layers when token budget exhausted
- [ ] **MEM-03**: available_layers() respects feature gates (no KG layer without knowledge-graph feature)

### Learn Loop

- [ ] **LEARN-01**: TurnOutcome struct captures tool outcomes, response meta, budget, model, and lane per turn
- [ ] **LEARN-02**: Immediate observers run synchronously, async observers spawned as tokio tasks
- [ ] **LEARN-03**: finalize_response.rs constructs TurnOutcome and calls LearnLoop — zero direct store writes

### Parser Canon

- [ ] **PARSE-01**: ParsedAction enum (Final/Call/Ask) is the canonical output type from response parsing

### Lane Split

- [ ] **LANE-01**: Lane enum (Answer/Action) with LanePolicy struct configuring all contracts
- [ ] **LANE-02**: Per-contract profile types (PromptProfile, ToolGateProfile, MemoryProfile, LearnProfile, ParserProfile)

## v0.2.0 Requirements

Deferred to next milestone. Tracked but not in current roadmap.

### Tool Gate

- **GATE-03**: Tool call to gated-out tool returns ToolNotFound error (not silently dropped)

### Parser Canon

- **PARSE-02**: Agent loop uses parser.parse(response) match — zero inline regex outside parsers/
- **PARSE-03**: Parser selected by model name, unknown falls back to Hermes

### Lane Split

- **LANE-03**: Config-based lane selection via default_lane in AgentsConfig
- **LANE-04**: Automatic lane routing via heuristic or router model

### Mode Descriptor

- **MODE-01**: Typed mode descriptor for orchestrator, trust_domain, engine, fabric, capability_profile axes
- **MODE-02**: Mode descriptor replaces is_local confusion across core_builder, agent_core, prepare_context

### Automation

- **AUTO-01**: Unified routine engine wrapping cron, heartbeat, pipeline, subagent

## Out of Scope

| Feature | Reason |
|---------|--------|
| New memory stores | MemoryLadder wraps existing 7, doesn't add new ones |
| New learning mechanisms | LearnLoop dispatches to existing 7, doesn't add new ones |
| New tools | Contract milestone, not feature milestone |
| Eval framework changes | Contracts enable per-lane eval in v0.2.0 |
| UX/REPL changes | Beyond lane config, no user-facing changes |
| Cluster/cloud changes | Still work, just not surfaced in contract layer |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| PROMPT-01 | Phase 1 | Complete |
| PROMPT-02 | Phase 1 | Complete |
| PROMPT-03 | Phase 1 | Pending |
| PROMPT-04 | Phase 1 | Complete |
| GATE-01 | Phase 2 | Pending |
| GATE-02 | Phase 2 | Pending |
| PARSE-01 | Phase 2 | Pending |
| MEM-01 | Phase 3 | Pending |
| MEM-02 | Phase 3 | Pending |
| MEM-03 | Phase 3 | Pending |
| LEARN-01 | Phase 4 | Pending |
| LEARN-02 | Phase 4 | Pending |
| LEARN-03 | Phase 4 | Pending |
| LANE-01 | Phase 5 | Pending |
| LANE-02 | Phase 5 | Pending |

**Coverage:**
- v0.1.1 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0

---
*Requirements defined: 2026-03-07*
*Last updated: 2026-03-07 after roadmap creation*
