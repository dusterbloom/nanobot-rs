# Architecture Review Memo — nanobot-rs config & product structure

**Date:** 2026-07-17  
**Authors:** opencode (primary review) + codex (second opinion) + owner (corrections)  
**Status:** assessment + plan. No code changed by this memo.

## Context

On 2026-07-17 the live `~/.nanobot/config.json` was silently gutted: API keys zeroed, several sections dropped, the local model name drifted (`bonsai-27b-mlx` vs the actual `bonsai-27b`), and traffic was routed to a dead port (`:1234` instead of higgs `:8000`). The owner couldn't tell what had happened. This sparked a fresh-eyes review of whether the config and product are over-built. This memo reconciles three independent passes.

## The three structural diseases

### 1. Redundant orchestration (multiple systems doing the same job)
Five subsystems decide "who runs this work": `trio` (3-model router), `toolDelegation` (subagents), `worker` (another delegator), `cluster` (multi-machine discovery), `reasoning` (autoDecompose).

**Reconciled verdict:**
- `toolDelegation` — **keep, harden.** First-class, integrated in the hot path (`core_builder.rs:143-179`). The owner wants it to "just work."
- `trio` — **deprecate→sunset (phased).** Overlaps `toolDelegation` semantically; a different mechanism in code, but a second orchestrator is unjustified.
- `worker` — **audit, then cut.** Appears config-only (not seen in the hot path); confirm before deleting.
- `reasoning` — **product decision, NOT a blind cut.** Codex corrected the primary review: it is wired into execution (`agent_loop/shared.rs:878-895`, checkpointing). Removing it is a behavior change.
- `cluster` — **make opt-in, don't exile.** Also active (`core_builder.rs:676-703`). The 11-port LAN scan every 60s is log noise on a single-machine setup, but the code is live.

### 2. Memory systems
`memory` (15+ fields, own model + port), `lcm` (compaction), `proprioception` (research hyperparameters).

**Reconciled verdict:**
- `lcm` — **keep.** Owner correction confirmed by codex: `lcm` is a *distinct* runtime flow (`resolve_memory_provider` separation), not a duplicate of `memory`. Significant work went into it.
- `memory` — **keep, trim.** The canonical memory system; collapse its 15+ knobs to ~5.
- `proprioception` — **confirm call-sites before cutting.** Looks research-y; may be load-bearing.

### 3. The `agents.defaults` god-bag
~40 flat fields mixing identity (3 model-name fields, 4 port fields, 3 dir fields), sampling, context, thinking, and 5 `adaptive*` heuristics. The multi-model-name/multi-port ambiguity is **exactly what caused the misrouting outage.**

**Reconciled verdict (all three agree):** collapse into a `models[]` table + `sampling`/`context`/`thinking` sub-objects. Kill the `adaptive*` heuristics (internalize or delete).

## The actual root cause of the outage (codex find)

Not "config too big" — a **deserialization-to-default anti-pattern** in the loader:

> `load_config` falls back to `Config::default()` on parse error (`config/loader.rs:97-113`). Defaults have empty secret strings → the next save serializes those empties, **zeroing keys and dropping any field not in the current schema.**

This is the bug that made today's outage possible, *independent of how the config is structured*. **Fix #1 (highest priority, S, low-risk):** fail-closed on parse error + preserve a backup + reserialize as a known-field merge onto raw JSON (so unknown/legacy keys survive).

## `lms` → OpenAI/Anthropic endpoint compatibility

Owner direction: standardize backend connectors on the OpenAI-compatible (and Anthropic) API standard instead of LM-Studio-specific knobs/ports.

Codex confirms it's **mostly already done** — provider creation routes through `providers/openai_compat.rs` / `factory.rs`. Delta is config-layer cleanup: replace `lmsPort`/`lmsMainModel` with generic `apiBase`/`apiKey`/provider+model + a legacy `lms*` shim. **Small runtime change, medium config migration.**

## Triangulated cut / keep / collapse list

| item | verdict | basis |
|---|---|---|
| `agents.defaults` god-bag → `models[]` + sub-objects | **collapse** | all three agree; caused the outage |
| duplicated model names / ports | **collapse** | all three agree |
| `adaptive*` heuristics | **cut** | all three agree |
| `lms` legacy knobs | **standardize on OpenAI/Anthropic compat** | owner + codex |
| `toolDelegation` | **keep, harden** | owner + codex (hot-path integrated) |
| `lcm` | **keep** | owner + codex (distinct from memory) |
| `memory` | **keep, trim** | all three |
| `trio` | **deprecate→sunset (phased)** | all three |
| `worker` | **audit, then cut** | primary + codex (likely unused) |
| `reasoning` | **product decision (active in code)** | codex correction |
| `cluster` | **opt-in, not exile (active in code)** | codex correction |
| `proprioception` | **confirm call-sites first** | codex |
| `provenance` | **trim flags, don't remove (active)** | codex |
| `channels` (whatsapp/telegram/email) | **separate plugin** | primary |
| `voice` | **separate plugin** | primary |
| secrets in config.json | **separate store/env** | primary + codex |

## Prioritized plan

| # | change | size | risk | safe-now? |
|---|---|---|---|---|
| 1 | **Fix the loader** (fail-closed + backup + merge-on-reserialize) | S | low | ✅ do first |
| 2 | Deprecate-don't-remove god-bag duplicates (read legacy aliases, emit warnings) | S | low | ✅ |
| 3 | Split `agents.defaults` → `models[]` + `sampling`/`context`/`thinking` (with aliases) | M | medium | migration |
| 4 | Standardize backends on OpenAI/Anthropic endpoint fields + `lms` shim | M | medium | migration |
| 5 | Audit `worker`; cut if unused | M | low-med | audit first |
| 6 | Sunset `trio` behind a migration | M | medium | phased |
| 7 | `reasoning`/`cluster`/`proprioception`/`provenance` → product decisions | L | high | decide, don't refactor blindly |

## Governing principle

> **Default config = a working minimal agent. Every additional subsystem is opt-in, has exactly one enable flag, and is not in the shipped default file. No second way to do the same job. And "looks redundant" ≠ "is redundant" — verify a subsystem is dead in the hot path before cutting it.**

The product isn't bad — it does a lot — but it has no spine: rival subsystems compete for each role (orchestration, memory) and all demand config. The fix is a clear core (one model endpoint, one orchestrator, one memory) with experiments opt-in — **and a loader that never silently falls back to defaults.**
