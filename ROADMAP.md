# ROADMAP: Local-First AI That Kicks Ass

> The first local AI that's structurally superior to cloud.
> Not bigger models. Smarter architecture.

## Vision

Cloud AI has one advantage: big models. Local AI can have every other advantage:
latency (<50ms vs 500ms+), unlimited tool rounds (500+ vs ~20), zero cost,
privacy, learning (skills + LoRA), and unlimited parallelism.

**The bet:** MAKER's framework + Nanbeige4.1-3B on a single RTX 3090 =
the first local-hardware million-step zero-error system.

---

## Phase 0: Foundation ← **WE ARE HERE**
> Make local mode actually good with the right models.

**Goal:** `nanobot setup` on any machine → working local AI in 5 minutes.

| Area | What | Detail docs |
|------|------|-------------|
| Models | Wire Nanbeige4.1-3B as RLM, Qwen3-30B-A3B as main, Qwen3-0.6B as memory | `docs/plans/local-model-matrix.md` |
| Config | Multi-model config schema (3 named slots: main/rlm/memory) | — |
| Hardware | Auto-detect VRAM/RAM, assign tier, select quants | — |
| Onboarding | `nanobot setup` interactive first-run | — |
| Reliability | Fix local role crashes, tool loops, drift | `docs/plans/local-trio-strategy-2026-02-18.md`, `docs/plans/local-model-reliability-tdd.md` |
| Tech debt | 132 compiler warnings, 2 test failures | — |
| Architecture | Break up SwappableCore, provider registry | `docs/plans/multi-provider-refactor.md`, `docs/plans/nanobot_architecture_review.md` |

---

## Phase 1: Million-Token Context
> Process 1M tokens with 8K-window models.

**Goal:** `nanobot ingest ./my-codebase` → searchable 1M-token context, any model.

| Area | What | Detail docs |
|------|------|-------------|
| Storage | File-backed volumes (mmap) + line-offset index | `docs/plans/context-gate.md` |
| Search | Chunk index with simhash signatures | — |
| Semantic | Optional e5-small embeddings | — |
| Context mgmt | ContentGate: pass raw / brief / drill-down | `docs/plans/context-gate.md`, `docs/plans/context-protocol.md` |
| Proof | Needle-in-haystack at 1M: 95%+ recall, <60s | — |

---

## Phase 2: Million-Step Processes
> Execute 1M coherent steps without losing state.

**Goal:** `nanobot process "refactor this codebase" --budget 1000000` → runs overnight, resumes on crash.

| Area | What | Detail docs |
|------|------|-------------|
| Calibration | Measure Nanbeige4.1-3B per-step error rate `p` | `docs/plans/three-impossible-things.md` |
| Voting | MAKER first-to-ahead-by-k + red-flagging | `docs/plans/event-log-pipeline.md` |
| Decomposition | Maximal Agentic Decomposition (MAD) | `docs/plans/swarm-architecture.md` |
| Process tree | Persistent execution tree, checkpoint/resume | — |
| RLM | Complete ctx_summarize, recursive depth, smart short-circuit | `docs/plans/rlm-completion-proposal.md`, `docs/plans/adaptive_rlm_design.md` |
| Proof | Towers of Hanoi 20 disks (1,048,575 steps), zero errors, local only | — |

---

## Phase 3: Self-Evolving Agent
> 10x improvement over 1000 tasks.

**Goal:** Leave nanobot running overnight with 1000 tasks. Morning: measurably better.

| Area | What | Detail docs |
|------|------|-------------|
| Traces | Structured JSONL trace logging per process | — |
| Skills | Auto-crystallize successful patterns into reusable skills | — |
| Budget | Per-task-type stats in SQLite, budget optimizer | — |
| LoRA | Export traces → Zero pipeline → hot-swap LoRA | — |
| Proof | Plot learning curve: 5x fewer steps, 10x faster at task 1000 | — |

---

## Phase 4: Integration & Polish
> Make it real, make it shippable.

| Area | What |
|------|------|
| End-to-end | Phases 0-3 working together |
| Benchmarks | Compare vs Claude Code, Cursor, Aider, OpenHands |
| Docs | Architecture guide, setup guide, challenge writeups |
| Release | Pre-built binaries, model auto-download, `nanobot setup` cross-platform |

---

## Timeline (Aggressive)

```
         Feb                    Mar                    Apr
Week:  3    4    |    1    2    3    4    |    1    2    3
       ─────────┼────────────────────────┼───────────────
P0:    ████████ |                        |
P1:         ████|████████               |
P2:              |     ████████████      |
P3:              |              ████████████████
P4:              |                       |  ████████████
```

---

## Principles

1. Each phase ships something usable.
2. Local-first, cloud-optional.
3. Prove it or it didn't happen.
4. Honest about limitations.
5. Accessible — the Potato tier (8GB, no GPU) must work.

---

## Reference

All detailed design docs live in `docs/plans/`. Archived research in `archive/research/`.
Completed/superseded plans in `archive/completed/`.
