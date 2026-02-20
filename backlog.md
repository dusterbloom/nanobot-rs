# Nanobot Backlog

> Single source of truth for all actionable work.
> ROADMAP.md = vision. This file = what to do next.

---

## Phase 0: Foundation (current)

### 🔴 Blocking — do first

- [ ] **B1: 132 compiler warnings** — `cargo fix` pass to clear `unused_imports`, `dead_code`, `unused_variables`. Real errors invisible under noise.
- [ ] **B2: 2 test failures** — `test_web_search_no_api_key` (assumes no BRAVE_API_KEY), `test_normalize_alias_all_aliases` (stale `/prov` alias). Quick fixes.
- [ ] **B3: Wire Nanbeige4.1-3B as RLM model** — Update `DELEGATION_MODEL_PREFERENCES` (top of list), update `DEFAULT_LOCAL_MODEL`. Test tool calling with nanobot's `<tool_call>` format. _Ref: `docs/plans/local-model-matrix.md`_
- [ ] **B4: Multi-model config schema** — Add `local.main`, `local.rlm`, `local.memory` to config. Each slot: `{ model, path, gpu, context_size, temperature }`. Server manager spawns up to 3 llama-server instances. _Ref: `docs/plans/local-model-matrix.md`_

### 🟡 Important — do soon

- [ ] **I1: Local role/protocol crashes** — Fix `system` role crash, alternation crash, orphan tool messages. Thread repair pipeline exists but needs hardening. _Ref: `docs/plans/local-trio-strategy-2026-02-18.md`, `docs/plans/local-model-reliability-tdd.md`_
- [ ] **I2: Non-blocking compaction** — Spawn compaction as background task via `tokio::spawn`, swap result when done. Three tiers: background (80%), aggressive (85%), emergency truncation (95%). _(Spacebot idea)_
- [ ] **I3: Context Gate** — Replace dumb char-limit truncation with `ContentGate`: pass raw / structural briefing / drill-down. Zero agent-facing API changes. _Ref: `docs/plans/context-gate.md`, `docs/plans/context-protocol.md`_
- [ ] **I4: Multi-provider refactor** — Break up `SwappableCore` god struct, extensible provider registry, fallback chains. _Ref: `docs/plans/multi-provider-refactor.md`, `docs/plans/nanobot_architecture_review.md`_
- [ ] **I5: Dynamic model router** — Prompt complexity scoring (light/standard/heavy), auto-downgrade simple messages to cheaper models. _Ref: `docs/plans/dynamic-model-router.md`_

### 🟢 Nice to have — Phase 0

- [ ] **N1: Auto hardware detection** — Detect VRAM/RAM/CPU, auto-assign tier (Potato/Sweet/Power/Beast), select quant level. `nanobot doctor` command.
- [ ] **N2: `nanobot setup`** — Interactive first-run: detect hardware, download models, generate optimal config.
- [ ] **N3: Streaming rewrite** — Incremental markdown renderer, line-by-line syntax highlighting, no full-response rerender. _Ref: `docs/plans/streaming-rewrite.md`_
- [ ] **N4: Full-duplex REPL** — ESC+ESC instant cancel, backtick injection prompt, priority message channel. _Ref: `docs/plans/full-duplex-repl.md`_
- [ ] **N5: Thinking toggle** — `/think` command + Ctrl+T toggle for extended thinking mode. _Ref: `docs/plans/thinking-toggle.md`_
- [ ] **N6: Status injection** — Auto-inject background worker status into context each turn. _(Spacebot idea)_
- [ ] **N7: Message coalescing** — Batch rapid messages in channels into single LLM turn. _(Spacebot idea)_
- [ ] **N8: Narration stress test** — Validate narration compliance across local models. _Ref: `docs/plans/narration-stress-test.md`_

---

## Phase 1: Million-Token Context (next)

- [ ] **P1.1: File-backed volumes** — `MappedVolume` struct with mmap + line-offset index
- [ ] **P1.2: Chunk index** — 4K-char chunks, simhash signatures, `ctx_search`
- [ ] **P1.3: Semantic index** — Optional e5-small embeddings, vector similarity
- [ ] **P1.4: Proof** — Needle-in-haystack at 1M tokens, 95%+ recall, <60s

---

## Phase 2: Million-Step Processes (later)

- [ ] **P2.0: Calibration run** — Measure Nanbeige4.1-3B per-step `p` on 1K-10K steps
- [ ] **P2.1: MAKER voting** — `first_to_ahead_by_k`, red-flagging, output token cap
- [ ] **P2.2: MAD decomposition** — Atomic step definitions per domain
- [ ] **P2.3: Process tree** — Persistent execution tree, checkpoint/resume
- [ ] **P2.4: RLM completion** — `ctx_summarize`, recursive depth, smart short-circuit. _Ref: `docs/plans/rlm-completion-proposal.md`, `docs/plans/adaptive_rlm_design.md`_
- [ ] **P2.5: Swarm architecture** — Workers spawn Workers, budget propagation. _Ref: `docs/plans/swarm-architecture.md`_
- [ ] **P2.6: Event log pipeline** — Append-only JSONL, pipeline runner. _Ref: `docs/plans/event-log-pipeline.md`_
- [ ] **P2.7: Proof** — Towers of Hanoi 20 disks, 1M+ steps, zero errors, local only

---

## Phase 3: Self-Evolving Agent (future)

- [ ] **P3.1: Trace logger** — Structured JSONL per process
- [ ] **P3.2: Skill crystallization** — Auto-create skills from repeated successes
- [ ] **P3.3: Budget calibration** — Per-task-type stats in SQLite
- [ ] **P3.4: LoRA distillation** — Export traces → Zero pipeline → hot-swap LoRA

---

## Spacebot Ideas (parking lot)

Captured from [spacebot](https://github.com/spacedriveapp/spacebot). Ideas only, no code.

| Idea | Status | Mapped to |
|------|--------|-----------|
| Non-blocking compaction | Backlog I2 | Phase 0 |
| Status injection | Backlog N6 | Phase 0 |
| Message coalescing | Backlog N7 | Phase 0 |
| Branch concept (context-fork) | Not started | Phase 2 (related to swarm) |
| Prompt complexity routing | Backlog I5 | Phase 0 |
| Memory bulletin (Cortex) | Not started | Phase 3 (related to memory) |

---

## Done ✅

- ~~Fix stale comment in `ensure_compaction_model`~~ (2026-02-17)
- ~~Raise tool result truncation threshold~~ (2026-02-17)
- ~~Document multi-session CONTEXT.md race~~ (2026-02-17)
- ~~Input box disappears during streaming~~ (2026-02-17)
- ~~Agent interruption too slow~~ (2026-02-17)
- ~~Subagent improvements (wait, output files, budget, compaction)~~ (2026-02-18)
- ~~Tool runner infinite loop fix~~ (2026-02-18)
