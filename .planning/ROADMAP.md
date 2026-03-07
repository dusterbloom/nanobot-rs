# Roadmap: nanobot v0.1.1 — Contract Architecture

**Created:** 2026-03-07
**Phases:** 5
**Requirements:** 15 mapped, 0 unmapped

## Phase Overview

| # | Phase | Goal | Requirements | Depends On |
|---|-------|------|--------------|------------|
| 1 | Prompt Contract | Every prompt section has a name, budget, assembly order | PROMPT-01..04 | — |
| 2 | 1/2 | In Progress|  | — |
| 3 | Memory Ladder | Single memory facade with named layers, priority, budget | MEM-01..03 | Phase 1 |
| 4 | Learn Loop | One dispatch point for all turn observations | LEARN-01..03 | Phase 3 |
| 5 | Lane Split | Answer/Action as parameter sets over all contracts | LANE-01..02 | Phases 1-4 |

## Phase 1: Prompt Contract

**Goal:** Every prompt section has a name, budget, and assembly order. Zero ad-hoc injections.

**Requirements:** PROMPT-01, PROMPT-02, PROMPT-03, PROMPT-04

**Complexity:** High (~200 new LOC, biggest refactor)

**Key files:**
- `agent/prompt_contract.rs` — **New.** PromptSection enum, PromptBudget struct, PromptAssembler trait + impl
- `agent/context.rs` — Replace ContextBuilder internals with PromptAssembler. Keep public API stable.
- `agent/prepare_context.rs` — Remove scattered memory/learning/bulletin injection. Replace with assembler.assemble().
- `agent/mod.rs` — Add pub mod prompt_contract

**Success criteria:**
1. System prompt sections are typed PromptSection variants with fixed ordering (Identity first, History last)
2. Each section reports allocated token budget, actual measured tokens, and source after assembly
3. All system prompt content flows through PromptAssembler — zero append_to_system_prompt() calls remain
4. When total prompt cost exceeds context window cap, sections dropped tail-first (History shrinks before Identity)

---

## Phase 2: Tool Gate + Response Types

**Goal:** Tool availability varies by model capability. Small models see 5 tools, not 16. Canonical response parsing type defined.

**Requirements:** GATE-01, GATE-02, PARSE-01

**Complexity:** Low (~105 new LOC combined)

**Key files:**
- `agent/tool_gate.rs` — **New.** ToolGate struct, TINY/BALANCED constants, filter()
- `agent/tools/registry.rs` — Add definitions_for(names) method
- `agent/agent_loop.rs` — Route get_definitions() through ToolGate::filter()
- `agent/parsers/registry.rs` — Add ParsedAction enum (Final/Call/Ask)

**Success criteria:**
1. Small models see exactly the tiny tool set (5 tools), medium see balanced (10), large see all
2. toolsFilter config override takes precedence over size-class gate
3. ParsedAction enum (Final/Call/Ask) exists as the canonical output type from response parsing

---

## Phase 3: Memory Ladder

**Goal:** 5 named layers with explicit precedence and a query contract. No ambiguity about what's truth, cache, or projection.

**Requirements:** MEM-01, MEM-02, MEM-03

**Complexity:** Medium (~160 new LOC)

**Depends on:** Phase 1 (memory sections are typed PromptSection slots)

**Key files:**
- `agent/memory_ladder.rs` — **New.** MemoryLayer enum, MemoryQuery, MemoryResult, MemoryLadder trait + impl
- `agent/prepare_context.rs` — Replace scattered memory injection with memory_ladder.query()
- `agent/prompt_contract.rs` — PromptSection::Memory and ::WorkingMemory populated from MemoryLadder results

**Existing stores NOT modified:** memory.rs, working_memory.rs, knowledge_store.rs, knowledge_graph.rs, session/db.rs, lcm.rs

**Success criteria:**
1. Five named memory layers (Scratch, WorkingSession, DurablePersonal, SearchIndex, GroundTruth) with explicit priority ordering
2. Memory queries stop pulling from lower-priority layers when token budget exhausted
3. available_layers() returns only layers whose feature gates are active

---

## Phase 4: Learn Loop

**Goal:** Every observation from a completed turn flows through one dispatch point. Zero direct writes from finalize_response.rs.

**Requirements:** LEARN-01, LEARN-02, LEARN-03

**Complexity:** Medium (~90 new LOC)

**Depends on:** Phase 3 (needs to know which stores to write to)

**Key files:**
- `agent/learn_loop.rs` — **New.** TurnOutcome struct, LearnLoop trait, default impl dispatching to existing stores
- `agent/finalize_response.rs` — Extract observation logic into TurnOutcome construction. Call learn_loop.observe_immediate() + spawn observe_async()
- `agent/agent_loop.rs` — Pass LearnLoop instance to finalize

**Existing stores NOT modified:** learning.rs, budget_calibrator.rs, anti_drift.rs, model_feature_cache.rs, reflector.rs, lora_bridge.rs

**Success criteria:**
1. TurnOutcome struct captures tool outcomes, response meta, budget meta, model info, and lane per turn
2. Immediate observers run synchronously; async observers (LoRA, Reflector) spawned as tokio tasks
3. finalize_response.rs constructs TurnOutcome and calls LearnLoop — zero direct store writes remain

**Plans:** 1 plan

Plans:
- [ ] 04-01-PLAN.md — TurnOutcome + LearnLoop trait + wire into finalize_response

---

## Phase 5: Lane Split

**Goal:** Answer and Action are different parameter sets for the same contracts. Not separate infrastructure.

**Requirements:** LANE-01, LANE-02

**Complexity:** Medium (~140 new LOC)

**Depends on:** Phases 1-4 (consumes all contracts)

**Key files:**
- `agent/lane.rs` — **New.** Lane enum, LanePolicy struct, profile types
- `agent/agent_loop.rs` — Pass lane.policy() to PromptAssembler, ToolGate, MemoryLadder, LearnLoop
- `agent/mod.rs` — Add pub mod lane

**Success criteria:**
1. Lane enum (Answer/Action) with LanePolicy struct configuring all contracts
2. Per-contract profile types (PromptProfile, ToolGateProfile, MemoryProfile, LearnProfile, ParserProfile) produce measurably different pipelines
3. Lane::Answer produces fewer prompt sections, fewer tools, lighter learning than Lane::Action

---

## Coverage

| Requirement | Phase | Status |
|-------------|-------|--------|
| PROMPT-01 | Phase 1 | Pending |
| PROMPT-02 | Phase 1 | Pending |
| PROMPT-03 | Phase 1 | Pending |
| PROMPT-04 | Phase 1 | Pending |
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

**v0.1.1 requirements:** 15 total | **Mapped:** 15 | **Unmapped:** 0

---
*Roadmap created: 2026-03-07*
*Last updated: 2026-03-07 after initial creation*
