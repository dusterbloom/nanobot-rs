---
gsd_state_version: 1.0
milestone: v0.3.0
milestone_name: Inference Speed + Cache Prefill
status: planning
stopped_at: Milestone started, defining requirements
last_updated: "2026-03-15"
last_activity: 2026-03-15 — Milestone v0.3.0 started
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
---

# State: nanobot

## Current Position

Milestone v0.3.0 Inference Speed + Cache Prefill — PLANNING
Phase: Not started (defining requirements)
Status: Defining requirements
Last activity: 2026-03-15 — Milestone v0.3.0 started

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-15)

**Core value:** Personal AI substrate across any model backend with persistent, adaptive memory
**Current focus:** Eliminate prefill latency via KV cache reuse, fix token drift, add streaming telemetry

## Accumulated Context

- 6 typed contracts established (PromptContract, ToolGate, MemoryLadder, LearnLoop, ParserCanon, LaneSplit)
- 3 low-severity integration gaps deferred (ParsedAction consumers, GATE-02 config plumbing, LearnProfile/ParserProfile consumers)
- v0.2.0 State-Driven Architecture: phase 06 planned (4 waves) but not executed, deferred
- **Performance baseline (2026-03-15):** 9B dense 0.5-3.4 w/s, 35B MoE 6-9.7 w/s, 60% token drift, zero cache reuse
- Streaming path has no `llm_call_complete` telemetry — only failures logged
- LM Studio + oMLX are the active local backends; aux mlx-lm server (:8090) never auto-starts
