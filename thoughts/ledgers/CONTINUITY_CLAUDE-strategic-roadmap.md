# Continuity Ledger: Strategic Roadmap 2026-02-19

## Goal
Stop feature accumulation. Stabilize, simplify, then build toward 3 impossible challenges
with a local-first trio architecture that runs on 24GB VRAM (WSL) and 32GB M4.

## Constraints
- SOLID / DRY / KISS — no over-engineering
- Don't reinvent wheels (UTCP, ZeptoClaw patterns)
- Design for small models first (if 1B works, everything works)
- Cross-platform: RTX 3090 24GB + M4 32GB
- 66 uncommitted GLM5 files need review-then-commit, not rewrite

## Key Decisions
- **NanBeige 3B as main model candidate**: Beats 32B models on tool calling (BFCL-V4: 56.50), sustains 500+ tool rounds [2026-02-19]
- **Nemotron-30B-A3B rejected for trio**: 19.3GB + KV cache = OOM on 24GB, crashed WSL [2026-02-19]
- **mistral.rs rejected**: Spent full day testing, could not get stable operation — reverted to LM Studio [2026-02-20]
- **LM Studio as inference engine**: JIT model loading, OpenAI-compat API, proven stable with trio [2026-02-20]
- **Three trio combos to test empirically**: A (NanBeige+Orch+Ministral ~12.4GB), B (NanBeige+LFM1.2B+Ministral ~8.4GB), C (NanBeige-does-all+Ministral ~8.1GB) [2026-02-19]
- **Voice mode**: Only VAD remains as future work [2026-02-19]
- **KIK/evolution**: Deferred until challenges 1+2 won [2026-02-19]
- **GLM5 code**: Review module-by-module (option A), good quality, all directed work [2026-02-19]

## State
- Done:
  - [x] Strategic assessment — 7 ledgers reviewed, all feature streams catalogued
  - [x] GLM5 session forensics — root causes identified (ctx overflow, provider chaos, router lockout)
  - [x] Model inventory — 18 GGUFs catalogued with VRAM budgets
  - [x] Trio combos designed — 3 options within hardware budget
  - [x] Plan written — .sisyphus/plans/2026-02-19-strategic-roadmap.md
- Now:
  - [->] Phase 0: Stabilize (fix build, SOLID/DRY/KISS audit, circuit breaker, incremental commits)
- Next:
  - [ ] Phase 1.1: Empirical trio testing with LM Studio (3 combos vs reliability gates)
  - [ ] Phase 1.2: Context diet (system prompt budget, tool result sanitization)
  - [ ] Phase 1.3: Evaluate UTCP and ZeptoClaw patterns
- Remaining:
  - [ ] Phase 2: North star challenges (infinite context, 100k workflows, self-evolution)
  - [ ] Phase 3: Product positioning (single binary, community adoption)

## Open Questions
- UNCONFIRMED: Is LFM2.5-1.2B-Nova good enough as router (vs proven Nemotron-Orchestrator)?
- UNCONFIRMED: What made zeptoclaw + llama3.2-1B work perfectly in first session but fail in second?
- RESOLVED: mistral.rs — rejected after full-day testing, unstable [2026-02-20]

## Working Set
- Files: .sisyphus/plans/2026-02-19-strategic-roadmap.md
- Branch: main
- Build: `cargo build && cargo test`
- Models: ~/models/*.gguf
- Hardware: RTX 3090 24GB (WSL), M4 32GB (macOS)
- Reference: ~/.nanobot/workspace/plans/local-model-matrix.md
