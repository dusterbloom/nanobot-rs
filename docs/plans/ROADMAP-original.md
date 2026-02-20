# ROADMAP: Local-First AI That Kicks Ass

> The first local AI that's structurally superior to cloud.
> Not bigger models. Smarter architecture.

## Where We Are (February 2026)

### What Works
- **34K lines of Rust** — stable, modular codebase
- **Multi-provider architecture** — Anthropic, OpenAI-compat, Claude Code, local llama-server
- **Local mode** — auto-spawns llama-server, manages main + delegation + compaction (3 server slots)
- **Tool delegation (RLM)** — subagent spawning, tool runner (3K lines), context store with micro-tools
- **ContextStore** — 701 lines, in-memory HashMap with ctx_slice, ctx_grep, ctx_length, ctx_summarize, mem_store, mem_recall
- **Memory system** — sessions (JSONL), MEMORY.md, skills, scratchpad, semantic recall
- **Channels** — CLI, Telegram, WhatsApp, Email, voice pipeline
- **Streaming** — progress line (no scrollback pollution), tool event display, syntax highlighting

### What's Chosen But Not Wired
- **Nanbeige4.1-3B** as RLM model (beats Qwen3-32B on tool calling, 500+ rounds)
- **Qwen3-30B-A3B-2507** as main model (GPT-4o class, 3.3B active params)
- **Qwen3-0.6B** as memory model (already used for compaction)
- **Hardware tier matrix** defined (Potato→Sweet→Power→Beast→Hybrid)
- Delegation model prefs still list old models (Ministral-3, not Nanbeige)
- Default model still Nemotron-Nano-9B

### What's Missing
- No file-backed volumes (mmap) — ContextStore is in-memory only
- No process tree / checkpointing / resumable execution
- No trace logging or automatic skill crystallization
- No multi-model config schema (3 named slots)
- No auto hardware detection
- No LoRA integration

---

## The Thesis

Cloud AI has one advantage: big models. Local AI can have every other advantage:

| | Cloud | Local (nanobot) |
|---|---|---|
| Model size | 400B+ | 3B-30B |
| Latency per tool call | 500ms-2s (network) | <50ms (localhost) |
| Tool rounds before timeout | ~20 | 500+ |
| Context strategy | Cram into window | Index + slice (unlimited) |
| Privacy | Your data on their servers | Your data on your disk |
| Cost per 1000 tasks | $50-500 | $0 (electricity) |
| Learning | None (stateless) | Accumulates skills + LoRA |
| Parallelism | Rate-limited | Limited only by GPU/CPU |

**The structural advantages compound.** A 3B model doing 500 tool rounds at 50ms each finishes in 25 seconds. A cloud model doing 20 rounds at 1s each takes 20 seconds — but solved 25x less of the problem.

### Validated by MAKER (2025)

The MAKER paper (arXiv:2511.09030) proved that million-step zero-error execution is
achievable — but only tested cloud models. Their key finding: **small non-reasoning models
outperform expensive reasoning models** for step execution. However, they found llama-3.2-3B
(p=0.0) completely failed, concluding 3B models can't do it.

**Nanbeige4.1-3B changes this calculus entirely.** Released after MAKER, it is a fundamentally
different class of 3B model:

| Benchmark | llama-3.2-3B | Nanbeige4.1-3B | Qwen3-32B (10x larger) |
|---|---|---|---|
| BFCL-V4 (tool use) | not tested | **56.50** | 47.90 |
| AIME 2026 I (math) | N/A | **87.40** | 75.83 |
| GPQA (science) | N/A | **83.8** | 68.4 |
| Arena-Hard-v2 | N/A | **73.2** | 56.0 |
| Sustained tool rounds | fails | **500+** | not tested |
| GAIA (with tools) | N/A | **69.90** | 30.17 |

Nanbeige4.1-3B is the first small model to natively sustain 500+ rounds of tool invocations.
Combined with MAKER's voting framework, this creates a path to **million-step local execution**
that didn't exist when either paper was published alone.

**The bet:** MAKER's framework + Nanbeige4.1-3B on a single RTX 3090 = the first local-hardware
million-step zero-error system. Calibration run (Phase 2.0) will determine feasibility.

---

## Phase 0: Foundation (1-2 weeks)
> Make local mode actually good with the right models.

### 0.1 Update Model Matrix
- [ ] Add `"Nanbeige4.1"` to `DELEGATION_MODEL_PREFERENCES` (top of list)
- [ ] Update `DEFAULT_LOCAL_MODEL` to Nanbeige4.1-3B (or auto-detect best available)
- [ ] Test Nanbeige tool calling with nanobot's `<tool_call>` format
- [ ] Verify 500+ round sustained tool chains locally

### 0.2 Multi-Model Config Schema
- [ ] Add `local.main`, `local.rlm`, `local.memory` to config schema
- [ ] Each slot: `{ model, path, gpu, context_size, temperature }`
- [ ] Server manager spawns up to 3 llama-server instances (different ports)
- [ ] Graceful degradation: if only 1 model available, use it for all slots

### 0.3 Auto Hardware Detection
- [ ] Detect available VRAM (nvidia-smi), RAM, CPU cores
- [ ] Auto-assign hardware tier (Potato/Sweet/Power/Beast)
- [ ] Auto-select quant level based on available memory
- [ ] `nanobot doctor` command: shows detected hardware, models, config

### 0.4 Local Onboarding
- [ ] `nanobot setup` — interactive first-run: detect hardware, download models, configure
- [ ] Download Nanbeige4.1-3B automatically (2GB, fast)
- [ ] For Tier 2+: offer to download Qwen3-30B-A3B-2507 (17GB, warn about size)
- [ ] Generate optimal config.json based on hardware

**Deliverable:** `nanobot setup` on any machine → working local AI in 5 minutes.

---

## Phase 1: Million-Token Context (2-3 weeks)
> Challenge 1: Process 1M tokens with 8K-window models.

### 1.1 File-Backed Volumes
- [ ] `MappedVolume` struct: mmap file + line-offset index
- [ ] Ingest pipeline: file/directory → volume with metadata
- [ ] `ctx_slice(volume, start, end)` → O(1) random access
- [ ] `ctx_grep(volume, pattern)` → scan mmap, return matches with line numbers
- [ ] Volume handles in ContextStore alongside existing HashMap

### 1.2 Chunk Index
- [ ] Split volumes into 4K-char chunks on ingest
- [ ] Generate chunk signatures (simhash or 3-word RLM summary)
- [ ] `ctx_search(volume, query)` → scan chunk summaries → return relevant chunk IDs
- [ ] Worker reads only relevant chunks (2-4K tokens each)

### 1.3 Semantic Index (optional, bridges to Zero)
- [ ] Embed chunks with e5-small (33M params, runs on CPU)
- [ ] Store embeddings alongside chunks
- [ ] `ctx_search` becomes vector similarity when embeddings available
- [ ] Graceful fallback to simhash when no embedding model

### 1.4 Proof: Needle-in-a-Haystack at 1M
- [ ] Generate 1M tokens synthetic data (code + docs + logs)
- [ ] Insert 100 needles at random positions
- [ ] Benchmark: recall, precision, latency
- [ ] Target: 95%+ recall, <60 seconds, single GPU
- [ ] Publish results

**Deliverable:** `nanobot ingest ./my-codebase` → searchable 1M-token context, any model.

---

## Phase 2: Million-Step Processes (2-3 weeks)
> Challenge 2: Execute 1M coherent steps without losing state.
>
> **Grounded in:** MAKER (Meyerson et al., 2025) — "Solving a Million-Step
> LLM Task with Zero Errors" (arXiv:2511.09030). First system to achieve
> 1M+ LLM steps with zero errors via Massively Decomposed Agentic Processes.

### 2.0 Calibration Run (NEW — from MAKER)
> Before committing to large runs, measure the actual per-step error rate.
> This is the single most important number for the entire phase.

- [ ] Design representative micro-step task (e.g., single code edit + verify)
- [ ] Sample 1K-10K random steps, run Nanbeige4.1-3B on each independently
- [ ] Measure per-step success rate `p` (correct output / total attempts)
- [ ] Compute `k_min` — minimum votes needed for target reliability (Eq. 18 from MAKER)
- [ ] Project total cost: `steps × k_min × cost_per_sample`
- [ ] Store calibration data in `stats.db` for budget optimizer (Phase 3)

**Why this matters:** MAKER showed that cost varies 40x between models depending on `p`.
gpt-4.1-nano (p=0.64) needed k=29 votes → $41.9K. gpt-4.1-mini (p=0.998) needed k=3 → $3.5K.
Measuring `p` first prevents catastrophic budget waste.

**Expected for Nanbeige4.1-3B:** Given its benchmarks (BFCL-V4: 56.50, 500+ sustained tool
rounds, AIME 2026: 87.40, LiveCodeBench-Pro: 81.4), we expect p ≥ 0.95 on structured
micro-steps, which would require k ≈ 3-5. This makes 1M steps feasible on local hardware.

### 2.1 MAKER-Style Voting (NEW — core error correction)
> The key insight from MAKER: error correction at the *micro-step* level,
> not the task level. Each atomic step gets independent votes.

- [ ] Implement `first_to_ahead_by_k` voting: first answer to lead by k votes wins
- [ ] Red-flagging: discard responses with format errors, excessive length, or structural anomalies *before* they vote (format errors correlate with reasoning errors — proven in MAKER)
- [ ] Output token cap: MAKER found error rate spikes when response > 700 tokens. Enforce max output per micro-step
- [ ] Parallel sampling: for each step, fire k+1 requests concurrently (local: k+1 sequential with Nanbeige, still fast at ~50ms each)
- [ ] Temperature strategy: first vote at τ=0 (best guess), subsequent at τ=0.1 (diversity for decorrelation)

**Scaling law (from MAKER Eq. 18):**
```
P(all steps correct) = (1 - P(step error with voting))^s
P(step error with voting) ≈ (1-p)^k for large k when p > 0.5
With p=0.99, k=3: P(step error) ≈ 10^-6 → 1M steps feasible
With p=0.95, k=5: P(step error) ≈ 3×10^-7 → 1M steps feasible
```

### 2.2 Maximal Agentic Decomposition (MAD)
> From MAKER: decompose to the *smallest possible* subtask. One step per agent.
> Not "break into 5 subtasks." Break into *atomic* steps.

- [ ] Define what "one step" means for each task domain:
  - Code: read file → identify function → generate edit → apply → verify compilation (5 atomic steps)
  - Research: search → fetch → extract → store (4 atomic steps)
  - Refactor: for each function: read → analyze → edit → verify (4 atomic steps per function)
- [ ] Each micro-agent gets minimal context: current state + strategy + one instruction
- [ ] State passed between steps via structured handoff (not growing context window)
- [ ] Strategy/plan provided in every micro-agent's prompt (MAKER: "the overall strategy is provided in the prompt for each agent")

### 2.3 Process Tree (unchanged but informed by MAKER)
- [ ] `Process` struct: persistent execution tree with `WorkerNode`s
- [ ] Status tracking: pending → running → completed/failed
- [ ] Serialize tree to disk (JSON) after every node completion
- [ ] `nanobot resume <process-id>` — restart from last checkpoint

### 2.4 Parallel Execution
- [ ] Concurrency semaphore (configurable, default = CPU cores)
- [ ] Independent subtrees run in parallel
- [ ] Fan-out configurable per node (default 10)
- [ ] Global budget: atomic step counter, token counter, wall clock limit

### 2.5 Hierarchical Coherence
- [ ] Each Worker stores results via mem_store → parent's context
- [ ] Parent Workers summarize children's results before continuing
- [ ] Root has summary of everything (never raw data)
- [ ] Plan decomposition: recursive until each leaf is trivial

### 2.6 Observability
- [ ] Live process tree in TUI: progress bars, status per node
- [ ] `nanobot status <process-id>` — snapshot of tree state
- [ ] Error aggregation: which subtrees failed, why, retry count
- [ ] Per-step vote counts: track how many votes each step needed (detect hard steps)

### 2.7 Self-Healing
- [ ] Failed node → parent retries with different strategy
- [ ] Configurable: retry N times, then skip/escalate
- [ ] Partial results preserved (tree shows exactly where it stopped)
- [ ] Prompt paraphrasing on retry (MAKER: decorrelates errors for stubborn steps)

### 2.8 Proof: Graduated Scale-Up
> MAKER's approach: calibrate small → project large → execute full.
> We do the same, with concrete milestones.

**Phase 2.8a: 1K steps (baseline)**
- [ ] Towers of Hanoi with 10 disks (1,023 steps) — direct comparison with MAKER
- [ ] Measure Nanbeige4.1-3B per-step `p` on this domain
- [ ] Compare with MAKER's model table (gpt-4.1-mini p=0.998, llama-3.2-3B p=0.0)
- [ ] Target: zero errors with k ≤ 5

**Phase 2.8b: 10K-100K steps (practical)**
- [ ] The Refactoring Challenge: real codebase, 500+ files
- [ ] Task: "Add comprehensive error handling to all functions"
- [ ] Measure: files processed, compilation success, test pass rate, total steps
- [ ] Target: 500+ files, 95%+ compilation success, <30 minutes

**Phase 2.8c: 1M steps (the big one)**
- [ ] Towers of Hanoi 20 disks (1,048,575 steps) — exact MAKER benchmark
- [ ] Run on RTX 3090 with Nanbeige4.1-3B
- [ ] Projected time: ~42 hours at 50ms/step × 3 votes (weekend run)
- [ ] Target: zero errors, local hardware only, no cloud
- [ ] **If successful: first local-hardware 1M-step zero-error result**

**Deliverable:** `nanobot process "refactor this codebase" --budget 1000000` → runs overnight, resumes on crash.

### MAKER Paper Reference Card
> Key numbers for quick reference during implementation.

```
Paper: "Solving a Million-Step LLM Task with Zero Errors"
Authors: Meyerson et al. (Cognizant AI Lab + UT Austin), 2025
URL: https://arxiv.org/abs/2511.09030

Framework: MDAP (Massively Decomposed Agentic Processes)
Implementation: MAKER (MAD + first-to-ahead-by-K voting + Red-flagging)

Key results:
  - 20-disk Towers of Hanoi: 1,048,575 steps, zero errors
  - Best model: gpt-4.1-mini (non-reasoning), p=0.998, k=3, cost=$3.5K
  - Small models CAN work: gpt-oss-20B at $1.7K projected
  - Reasoning models NOT needed (and more expensive)
  - llama-3.2-3B FAILED (p=0.0) — but this predates Nanbeige4.1-3B
  - Per-step error rate is STABLE as problem size grows (encouraging)
  - Exponential convergence: after first k rounds, undecided steps decay exponentially
  - Red-flagging: format errors correlate with reasoning errors — discard them
  - Error decorrelation: zero steps had errors in both independent runs

Key equations:
  - P(success) = (1 - P_err)^s where s = total steps
  - P_err with voting ≈ (1-p)^k (simplified, for p > 0.5)
  - Expected votes per step ≈ k + (1-p)/(2p-1) (for first-to-ahead-by-k)
  - Cost = s × E[votes_per_step] × cost_per_sample

Implications for nanobot:
  - Nanbeige4.1-3B (500+ tool rounds, BFCL 56.50) likely has p >> llama-3.2-3B
  - If p ≥ 0.99: k=3, 1M steps in ~42 hours on RTX 3090
  - If p ≥ 0.95: k=5, 1M steps in ~70 hours on RTX 3090
  - If p ≥ 0.90: k=8, 1M steps in ~112 hours — still feasible (5 days)
  - Calibration run (Phase 2.0) determines which scenario we're in
```

---

## Phase 3: Self-Evolving Agent (3-4 weeks)
> Challenge 3: 10x improvement over 1000 tasks.

### 3.1 Trace Logger
- [ ] Record every Worker execution: task, tools called, results, success/fail, timing
- [ ] Store as structured JSONL (one file per process, indexed)
- [ ] Trace viewer: `nanobot traces --last 100`

### 3.2 Skill Crystallization (Loop 1)
- [ ] Pattern extractor: successful trace → reusable recipe
- [ ] Auto-create skills from repeated successful patterns
- [ ] Skill matcher: new task → check if a crystallized skill applies
- [ ] Confidence scoring: only crystallize after N successful executions
- [ ] Skills stored in existing `workspace/skills/` system

### 3.3 Budget Calibration (Loop 2)
- [ ] Track stats per task type: avg steps, avg time, success rate, optimal depth/fan-out
- [ ] Store in SQLite (`stats.db`)
- [ ] Budget optimizer: allocate steps based on historical data, not heuristics
- [ ] Anomaly detection: flag tasks that take 10x more than expected

### 3.4 LoRA Distillation (Loop 3) — Zero Integration
- [ ] Export successful traces as training examples (input → output pairs)
- [ ] Zero's pipeline: L0 exact → L0.5 facts → L1 semantic → L2 delta
- [ ] Train LoRA adapter on accumulated examples (periodic, e.g. every 500 tasks)
- [ ] Hot-swap LoRA in llama-server without restart
- [ ] A/B testing: measure new LoRA vs baseline on held-out tasks

### 3.5 Proof: The Learning Curve
- [ ] Generate 1000 tasks across 10 categories
- [ ] Run sequentially, measure per-task metrics
- [ ] Plot learning curve: performance vs task number
- [ ] Target at task 1000 vs task 1:
  - Steps per task: 50 → 10 (5x reduction)
  - Wall clock: 30s → 3s (10x reduction)
  - Success rate: 60% → 95%
  - Skill cache hit: 0% → 60%
- [ ] Publish results

**Deliverable:** Leave nanobot running overnight with 1000 tasks. Morning: measurably better.

---

## Phase 4: Integration & Polish (2-3 weeks)
> Make it real, make it shippable.

### 4.1 The Full Stack
- [ ] Phase 0 + 1 + 2 + 3 working together end-to-end
- [ ] Ingest codebase → process with million steps → learn from execution → get faster
- [ ] Cross-channel: start task on CLI, monitor on Telegram, results on email

### 4.2 Benchmarks & Comparison
- [ ] Compare against: Claude Code, Cursor, Aider, OpenHands
- [ ] Same tasks, measure: accuracy, speed, cost, privacy
- [ ] Document where local wins and where it doesn't (honest)
- [ ] Focus on structural advantages: tool rounds, latency, learning

### 4.3 Documentation
- [ ] Architecture guide (how the pieces fit)
- [ ] Setup guide (hardware → working system in 5 min)
- [ ] Challenge writeups (methodology, results, reproducibility)
- [ ] Blog post / video: "The first local AI that outgrows its model"

### 4.4 Release
- [ ] `nanobot setup` works on Linux, macOS, WSL2
- [ ] Pre-built binaries
- [ ] Model auto-download
- [ ] First public release with challenge results

---

## Timeline (Aggressive but Realistic)

```
         Feb                    Mar                    Apr
Week:  3    4    |    1    2    3    4    |    1    2    3
       ─────────┼────────────────────────┼───────────────
P0:    ████████ |                        |
       Foundation                        |
       ─────────┼────────────────────────┼───────────────
P1:         ████|████████               |
       Million-Token Context             |
       ─────────┼────────────────────────┼───────────────
P2:              |     ████████████      |
       Million-Step Process              |
       ─────────┼────────────────────────┼───────────────
P3:              |              ████████████████
       Self-Evolving Agent               |
       ─────────┼────────────────────────┼───────────────
P4:              |                       |  ████████████
       Integration & Polish
```

**Total: ~10-12 weeks** (part-time, single developer + AI assist)

---

## Principles

1. **Each phase ships something usable.** No "wait 3 months for anything to work."
2. **Local-first, cloud-optional.** Every feature works offline. Cloud is an accelerator, not a dependency.
3. **Prove it or it didn't happen.** Each challenge has a concrete benchmark with published results.
4. **Honest about limitations.** Cloud models are smarter. We win on architecture, not intelligence.
5. **Accessible.** The Potato tier (8GB, no GPU) must work. Not everyone has a 3090.

---

## The Pitch

> Everyone's building bigger models with bigger context windows.
> We built a swarm of tiny models that processes million-token
> contexts, executes million-step plans, and teaches itself to
> get better — all on a single consumer GPU.
>
> The context window isn't in the model. It's in the architecture.
> The intelligence isn't in the weights. It's in the system.
>
> **nanobot: the local AI that outgrows its model.**
