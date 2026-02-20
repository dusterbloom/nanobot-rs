# Nanobot Backlog

> Single source of truth for all actionable work.
> ROADMAP.md = vision. This file = what to do next.

---

## Phase 0: Foundation (current)

### 🔴 Blocking — do first

- [ ] **B3: Update default local trio** — New defaults: 1) Main: `gemma-3n-e4b-it`, 2) Orchestrator: `nvidia_orchestrator-8b`, 3) Specialist: `ministral-3-8b-instruct-2512`. Update `agents.json`, `DEFAULT_LOCAL_MODEL`, config schema defaults, and `TrioConfig`. _Ref: experiments/tool-calling/_
- [ ] **B3.1: Smarter deterministic fallback** — `router_fallback.rs` only has 2 patterns (URL→web_fetch, "latest news"→spawn). Fails on "research report + URLs" which should spawn a researcher. Add pattern: research/report/summarize + URLs → `Subagent(researcher)`. _Ref: `src/agent/router_fallback.rs`_
- [ ] **B4: Multi-model config schema** — Add `local.main`, `local.rlm`, `local.memory` to config. Each slot: `{ model, path, gpu, context_size, temperature }`. Server manager spawns up to 3 llama-server instances. _Ref: `docs/plans/local-model-matrix.md`_
- [ ] **B5: RLM model evaluation** — Systematic experiments to find best RLM model per VRAM tier. Critical for "3 impossible things". See experiment plan below.

### 🟡 Important — do soon

- [ ] **I0: Trio pipeline actions** — Router can only emit ONE action per turn. Multi-step tasks (research + synthesize) fail because the router picks one tool and stops. Need pipeline-as-first-class router output + shared scratchpad between trio roles. _Ref: `thoughts/shared/plans/2026-02-20-trio-pipeline-architecture.md`_
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

- [ ] **P2.0: Calibration run** — Measure per-step `p` on 1K-10K steps using winning RLM model from E3
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

## Experiment Plan: Local Trio Evaluation

> Reduce assumptions one at a time. No coding until we know what works.

### New Default Trio (RTX 3090, 24GB VRAM)

| Role | Model | Size | Why |
|------|-------|------|-----|
| **Main** | gemma-3n-e4b-it | ~4B effective | Fast, good chat, small footprint |
| **Orchestrator** | nvidia_orchestrator-8b | 8B | 10/10 routing accuracy (proven in experiments/) |
| **Specialist** | ministral-3-8b-instruct-2512 | 8B | Strong tool-calling, instruction following |

### What We Know
- Nemotron Orchestrator: 10/10 routing (vs NanBeige 6/10). Purpose-built. **Proven.**
- NanBeige 3B: Good with `<think>\n</think>\n\n` prefill, but weak as router.
- Main + Orchestrator work well together in practice.
- Sequential self-routing would add latency vs parallel separation. Keep roles split.
- **Router single-action bottleneck**: `request_strict_router_decision()` returns ONE `RouterDecision`. Multi-step tasks (fetch 2 URLs + synthesize) cannot be expressed. The router picks one tool and the pipeline stalls.
- **Deterministic fallback too narrow**: `router_fallback.rs` has 2 patterns only — URLs→web_fetch (first URL), "latest news"→spawn. Everything else→ask_user. Misses research/report patterns.
- **Specialist has no tools**: `dispatch_specialist()` sends a single-shot chat — no tool access. Can synthesize given context but cannot fetch/execute.
- **Trio never tested end-to-end**: As of 2026-02-19 handoff, the full trio flow (Main→Router→Specialist) has never completed a real task through LM Studio.

### Experiments Needed (one assumption at a time)

#### E1: Role Evaluation Matrix
Test each candidate model in each role independently.

| Model | As Main | As Orchestrator | As Specialist | As RLM |
|-------|---------|-----------------|---------------|--------|
| gemma-3n-e4b-it | ? | ? | ? | ? |
| nvidia_orchestrator-8b | ✅ 10/10 routing | ✅ proven | ? | ? |
| ministral-3-8b-instruct-2512 | ? | ? | ? | ? |
| nanbeige4.1-3b | ? | 6/10 | ? | ? |

Test bench per role:
- **Main**: 10 conversation tasks (chat quality, coherence, narration compliance)
- **Orchestrator**: 10 routing cases (existing test suite from experiments/)
- **Specialist**: 10 tool-calling tasks (file ops, exec, multi-step)
- **RLM**: 5 delegation loops (multi-step file edit, research, build cycle)

#### E2: VRAM Profile Testing
Critical for "3 impossible things" — must work across hardware tiers.

| Tier | VRAM | Trio Budget | Candidate Combos |
|------|------|-------------|------------------|
| Potato | 4-6 GB | ~4B total | 1 model does all? |
| Sweet | 8-12 GB | ~12B total | 2 small models |
| Power | 16-24 GB | ~24B total | Full trio (current target) |
| Beast | 48+ GB | Unlimited | Bigger specialists |

#### E3: RLM Model Shootout
The key unknown. Test candidates on delegation loop benchmarks:
- Multi-step file edit (read → plan → edit → verify)
- Web research synthesis (search → fetch → summarize)
- Build cycle (edit → compile → fix errors → retry)

Metrics: completion rate, token cost, latency, error recovery.

#### E4: Integration Test
Once E1-E3 identify winners, run full nanobot session with the new trio.
Compare against current setup on real tasks.

### Experiment Order
1. **E1** first — know what each model can do in each role
2. **E3** next — find the RLM (biggest unknown)
3. **E2** then — scale findings across VRAM tiers
4. **E4** last — validate the winning combo end-to-end

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

- ~~B1: 132 compiler warnings~~ → 0 warnings (2026-02-20)
- ~~B2: 2 test failures~~ → 1248 pass, 0 fail (2026-02-20)
- ~~Fix: Subprocess stdin steal~~ — `.stdin(Stdio::null())` on all 4 spawn sites in shell.rs + worker_tools.rs (2026-02-20)
- ~~Fix: Esc-mashing freezes REPL~~ — drain_stdin() after cancel (2026-02-20, commit 57ec883)
- ~~Fix stale comment in `ensure_compaction_model`~~ (2026-02-17)
- ~~Raise tool result truncation threshold~~ (2026-02-17)
- ~~Document multi-session CONTEXT.md race~~ (2026-02-17)
- ~~Input box disappears during streaming~~ (2026-02-17)
- ~~Agent interruption too slow~~ (2026-02-17)
- ~~Subagent improvements (wait, output files, budget, compaction)~~ (2026-02-18)
- ~~Tool runner infinite loop fix~~ (2026-02-18)
