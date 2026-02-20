# Three Impossible Things Before Breakfast

## What if a local AI on a single GPU could do what no one has done?

Not by having a bigger model. Not by renting cloud GPUs. By being
*architecturally* smarter — a swarm of cheap Workers with a shared
context store, running on one RTX 3090.

Three challenges. Each sounds impossible. Each is achievable with
the Worker + ContextStore + micro-tools architecture.

---

## Experimental Validation: ArXiv Large-Context Test (2026-02-16)

Before diving into the challenges, here's what we've already proven:

**Experiment:** Process 88 ArXiv papers (~100K chars CSV) using only
local models on a single RTX 3090. Identify top authors, extract
keyword coverage, determine if emergence is discussed.

**Models:** Ministral-8B (main agent, port 8080) + Ministral-3B
(subagent, port 8083). Total VRAM: ~7GB.

**Result: 93.3% overall score — PASS**
- Author coverage: 80% (4/5 top authors found, missed Juneyoung Park)
- Keyword coverage: 100% (emergence, scaling, chain-of-thought)
- Emergence discussed: Yes

**What this proved:**
1. **Subagent delegation works end-to-end.** Main agent coordinates,
   spawns subagents to 3B model, aggregates results. No human in loop.
2. **Context slicing pattern validated.** Main agent never loaded the
   full 100K CSV. Used `exec` with grep/awk to extract slices, then
   delegated subagents to process each slice.
3. **Multi-model tiering works.** 8B brain + 3B hands. The 3B model
   executed tool calls reliably when given explicit instructions.
4. **Pipeline and loop actions battle-tested.** 6 progressive tests,
   5 infrastructure bugs found and fixed (Jinja alternation, config
   nesting, is_local detection, context blowout, consecutive messages).

**Key SLM insight:** Small models need explicit exec commands — they
use placeholder text if told "analyze this" but succeed when told
"Use exec tool to run: `grep -c 'emergence' data.csv`". This is
exactly the kind of pattern that Challenge 3's skill crystallizer
should capture automatically.

**What remains to prove:** Scale from 100K → 1M tokens. The pattern
works; the ContextStore infrastructure (mmap, indexing, chunking)
is what enables the 10x jump.

---

## Challenge 1: Million-Token Context

**The claim:** Process 1,000,000 tokens of context using models with
8K context windows. No information loss. Verifiable retrieval accuracy.

### Why it sounds impossible

The largest local models top out at 32K-128K context. Most usable
local models (3B-8B) are reliable at 4K-8K. A million tokens is
~750K words — roughly 10 novels, or an entire mid-size codebase,
or a year of chat logs. No local model can hold this.

### Why it's not

**No single Worker holds the full context. The context is a database.**

The ContextStore becomes a file-backed, indexed storage layer. Workers
access it through micro-tools: grep, slice, summarize, index queries.
Each Worker only loads the slice it needs — 2K chars at a time.

```
1M tokens in ContextStore (on disk, memory-mapped)
     │
     │  ctx_index("function", "error handling")
     │  → returns: [{file: "auth.rs", line: 42, score: 0.95}, ...]
     │
     │  ctx_slice("auth.rs", 40, 60)
     │  → returns: 20 lines of code (< 1K tokens)
     │
     │  Worker processes 1K tokens. Never sees the other 999K.
```

### Architecture

**Layer 0: Raw Storage**
```
ContextStore {
    // Small data: in-memory HashMap (existing)
    variables: HashMap<String, String>,

    // Large data: memory-mapped files
    // Each "volume" is a file on disk with an offset index
    volumes: HashMap<String, MappedVolume>,
}

struct MappedVolume {
    mmap: Mmap,                          // memory-mapped file
    line_offsets: Vec<u64>,              // byte offset of each line
    chunk_index: Vec<(u64, u64, u32)>,  // (start, end, chunk_id)
}
```

**Layer 1: Line Index**
Every volume has a line-offset table built on ingest. This makes
`ctx_slice(vol, line_start, line_end)` O(1) instead of O(N).
`ctx_grep(vol, pattern)` scans the mmap directly — no copying.
On a 3090, grepping 1M chars takes <10ms.

**Layer 2: Chunk Index**
Large volumes are split into chunks (e.g. 4K chars). Each chunk
gets a lightweight signature (simhash, or a 3-word summary from
the RLM itself). This enables:
```
ctx_search(volume, "error handling patterns")
→ scans chunk summaries (fast, small)
→ returns relevant chunk IDs
→ Worker calls ctx_slice on those chunks
```

**Layer 3: Semantic Index (optional, future)**
Embed chunks with a small embedding model (e5-small, 33M params).
Store embeddings alongside chunks. `ctx_search` becomes vector
similarity. This is the bridge to the Zero/kernel work.

### The Ingest Pipeline

When a large input arrives (file, codebase, document dump):

```
Input (1M tokens)
  │
  ├─ Write to disk as memory-mapped volume
  ├─ Build line-offset index (instant, single pass)
  ├─ Split into 4K chunks
  ├─ [Optional] Generate chunk summaries via RLM
  ├─ [Optional] Embed chunks for semantic search
  │
  └─ Return handle to main agent:
     "Volume 'codebase': 1,042,567 chars, 247 files, 1,203 chunks.
      Top-level: src/ (142 files), tests/ (38 files), docs/ (67 files)"
```

The main agent never sees the million tokens. It sees a handle and
metadata. It delegates Workers to explore specific parts.

### Proof Protocol (graduated)

**Phase 0: 100K — Pattern Validation ✅ DONE (2026-02-16)**
- 88 ArXiv papers, ~100K chars CSV
- Ministral-8B (main) + Ministral-3B (subagent) on RTX 3090
- Result: 93.3% score (80% author coverage, 100% keywords)
- Proved: subagent delegation, context slicing via exec/grep,
  multi-model tiering, pipeline/loop infrastructure
- See: `experiments/large-context-test/`

**Phase 1: 500K — Stress Test (next)**
1. Scale ArXiv dataset to 500+ papers (~500K chars)
2. Same model setup, same validation framework
3. Test: does accuracy hold? Does latency scale linearly?
4. Target: >80% accuracy, <5 minutes wall clock

**Phase 2: 1M — Full ContextStore (the real test)**
1. Build MappedVolume + line index + chunk index
2. Ingest 1M tokens (full codebase or document corpus)
3. Needle-in-a-haystack: insert 100 needles at random positions
4. Workers use ctx_grep, ctx_slice, ctx_search to find them
5. Target: 95%+ recall, <60 seconds, single 3090

**How Phase 2 works internally:**
```
Main agent: "Find all security vulnerabilities mentioned in the codebase"
  → Worker 0: ctx_search("codebase", "vulnerability") → 23 chunks
  → delegate 23 Workers, each reads one chunk
  → Each Worker: ctx_slice + ctx_grep for specific patterns
  → Each Worker: mem_store("finding_N", result)
  → Worker 0: aggregates, deduplicates, returns report
```

23 Workers × 4K context each = 92K tokens processed.
But they searched through 1M tokens via the index.
No single model held more than 8K.

### Implementation Cost (updated)

| Component | Effort | Status |
|-----------|--------|--------|
| Subagent delegation + pipeline | — | ✅ DONE (ArXiv experiment) |
| Context slicing via exec/grep | — | ✅ DONE (ArXiv experiment) |
| Multi-model tiering (8B+3B) | — | ✅ DONE (ArXiv experiment) |
| Scale test to 500K (Phase 1) | 2 hours | pending |
| MappedVolume + line index | 3 hours | pending |
| ctx_search (chunk-based) | 2 hours | pending (needs MappedVolume) |
| Ingest pipeline | 2 hours | pending (needs MappedVolume) |
| Needle-in-haystack benchmark | 2 hours | pending (needs all above) |
| Semantic index (optional) | 4 hours | future |

**Total remaining: ~11 hours** (was ~13, saved ~2 from validated infra)

---

## Challenge 2: Million-Step Processes

**The claim:** Execute a coherent 1,000,000-step process — a single
task that requires a million LLM calls, tool executions, or Worker
operations — without losing coherence, state, or progress.

**Grounded in:** MAKER (Meyerson et al., 2025) — first system to solve
1M+ LLM steps with zero errors. arXiv:2511.09030.

### Why it sounds impossible

Current AI agents lose coherence after ~50 turns. Context windows
fill up. The model forgets what it was doing. State accumulates as
unstructured text. One error cascades. No agent system today can
reliably execute even 10,000 steps.

### Why it's not (and now we have proof)

**MAKER proved it in October 2025.** 20-disk Towers of Hanoi,
1,048,575 steps, zero errors, using gpt-4.1-mini ($3.5K total).

The method is surprisingly simple — three components:

1. **Maximal Agentic Decomposition (MAD):** Break into the smallest
   possible subtask. One step per agent. Each agent gets minimal
   context: current state + strategy + one instruction.

2. **First-to-ahead-by-k Voting:** For each micro-step, sample
   multiple independent responses. First answer to lead by k votes
   wins. With p=0.998 per step, k=3 suffices for 1M steps.

3. **Red-flagging:** Discard any response with format errors,
   excessive length, or structural anomalies *before* it votes.
   Key finding: format errors correlate with reasoning errors.

**The critical insight:** You don't need a smarter model. You need
error correction at the micro-step level. Just like digital
communication pretends to be deterministic (while bits flip all
the time), MAKER makes LLM execution deterministic through voting.

### MAKER's Key Results

```
Model comparison (1M steps, t=0.95 target):
  gpt-4.1-mini (τ=0.1):  p=0.998, k=3,  cost=$3.5K  ← WINNER
  gpt-4.1-nano:          p=0.643, k=29, cost=$41.9K  ← too expensive
  o3-mini (reasoning):   p=0.998, k=3,  cost=$9.4K   ← works but 3x price
  llama-3.2-3B:          p=0.000, k=∞,  IMPOSSIBLE   ← too weak
  gpt-oss-20B:           p=0.964, k=6,  cost=$1.7K   ← cheapest overall

Key findings:
  - Reasoning models NOT needed (and more expensive)
  - Per-step error rate is STABLE as problem size grows
  - Exponential convergence after first k rounds
  - Format errors → reasoning errors (red-flag and discard)
  - Zero correlated errors across independent runs
```

### Why Nanbeige4.1-3B Changes Everything

MAKER concluded 3B models can't do it — but they tested llama-3.2-3B,
which is a completely different beast from Nanbeige4.1-3B:

```
                        llama-3.2-3B    Nanbeige4.1-3B    Qwen3-32B
BFCL-V4 (tool use):     N/A             56.50             47.90
AIME 2026 I (math):     N/A             87.40             75.83
GPQA (science):         N/A             83.8              68.4
Arena-Hard-v2:          N/A             73.2              56.0
Sustained tool rounds:  fails           500+              N/A
GAIA (with tools):      N/A             69.90             30.17
Deep search (xBench):   N/A             75                39

Nanbeige4.1-3B is the first 3B model that:
  - Sustains 500+ rounds of tool invocations natively
  - Beats models 10x its size on reasoning AND tool use
  - Was specifically trained for sustained agentic execution
```

**The prediction:** Nanbeige4.1-3B's per-step `p` on structured
micro-tasks will be comparable to gpt-4.1-mini (p ≈ 0.99+), making
million-step local execution feasible with k=3-5 voting.

**Projected local execution (RTX 3090):**
```
If p ≥ 0.99: k=3, 1M steps × 3 votes × 50ms = ~42 hours
If p ≥ 0.95: k=5, 1M steps × 5 votes × 50ms = ~70 hours
If p ≥ 0.90: k=8, 1M steps × 8 votes × 50ms = ~112 hours (5 days)
```

All feasible as weekend/week-long runs. **No cloud required.**

### Why it's not

**Steps don't need to be sequential. State doesn't need to be in context.**

A Worker tree with fan-out 10 and depth 5 is 100,000 leaf Workers.
Each does 10 steps. That's 1M steps. The tree structure provides
natural checkpointing — each node's mem_store is a checkpoint.

But the real insight is: **coherence comes from structure, not memory.**

A million-step process isn't one agent thinking for a million turns.
It's a plan that decomposes recursively until each leaf is trivial.

```
Step 1: "Refactor this 100K-line codebase"
  → Plan: 50 modules, each needs refactoring
  → Step 2-51: delegate per module
    → Each module: 20 files
    → Step 52-1051: delegate per file
      → Each file: read, analyze, diff, verify
      → Step 1052-5051: 4 operations per file × 1000 files
        → Each operation: 1-3 tool calls
        → Step 5052-20051: individual tool calls

20,000 steps. Each one trivial. The whole thing coherent because
the tree structure IS the plan.
```

### Architecture (MAKER-informed)

**Step 0: Calibration (before any large run)**

```
1. Pick representative task domain (e.g., code edits)
2. Define atomic step (e.g., "given state X, produce edit Y")
3. Sample 1K-10K random steps
4. Run Nanbeige4.1-3B on each independently
5. Measure p = correct / total
6. Compute k_min from MAKER Eq. 18
7. Project cost: steps × E[votes_per_step] × time_per_sample
8. Go/no-go decision
```

**The Process — a persistent, resumable execution tree with MAKER voting:**

```rust
struct Process {
    id: ProcessId,
    root: WorkerNode,
    budget: GlobalBudget,
    store: Arc<ProcessStore>,  // shared persistent state
    checkpoint_dir: PathBuf,
}

struct WorkerNode {
    id: NodeId,
    task: String,
    status: Status,           // pending, running, completed, failed
    result: Option<String>,
    children: Vec<WorkerNode>,
    checkpoint: Option<Checkpoint>,
}

struct GlobalBudget {
    total_steps: AtomicU64,     // counts down from 1M
    total_tokens: AtomicU64,    // optional token budget
    wall_clock: Instant,        // when did we start
    max_wall_clock: Duration,   // hard time limit
}

enum Status { Pending, Running, Completed, Failed, Paused }
```

**MAKER Voting Layer (NEW):**

```rust
struct VotingConfig {
    k: u32,                       // ahead-by-k threshold (default 3)
    max_output_tokens: u32,       // red-flag cutoff (default 750)
    temperature_first: f32,       // first vote (default 0.0)
    temperature_rest: f32,        // subsequent votes (default 0.1)
}

struct StepVoter {
    config: VotingConfig,
    model: Arc<LocalModel>,       // Nanbeige4.1-3B
}

impl StepVoter {
    /// MAKER Algorithm 2: first-to-ahead-by-k voting
    fn vote(&self, state: &StepState) -> StepResult {
        let mut votes: HashMap<String, u32> = HashMap::new();
        loop {
            let response = self.get_vote(state);  // Algorithm 3
            *votes.entry(response.answer.clone()).or_default() += 1;
            let max_other = votes.iter()
                .filter(|(k, _)| *k != &response.answer)
                .map(|(_, v)| *v).max().unwrap_or(0);
            if votes[&response.answer] >= self.config.k + max_other {
                return response;
            }
        }
    }

    /// MAKER Algorithm 3: get one valid vote (with red-flagging)
    fn get_vote(&self, state: &StepState) -> StepResult {
        loop {
            let raw = self.model.generate(state.to_prompt());
            // Red-flag checks:
            if raw.len() > self.config.max_output_tokens as usize { continue; }
            if !raw.contains("move =") || !raw.contains("next_state =") { continue; }
            if let Ok(parsed) = parse_structured_output(&raw) {
                return parsed;
            }
            // Format error = reasoning error. Discard silently.
        }
    }
}
```

**Key properties:**

1. **Persistent.** The process tree is serialized to disk after every
   node completion. Kill the process, restart, it resumes from the
   last checkpoint. `nanobot resume <process-id>`.

2. **Parallel.** Independent subtrees run concurrently. A 10-way
   fan-out at depth 2 means 100 Workers can run simultaneously
   (bounded by a concurrency semaphore).

3. **Budget-aware.** Every Worker decrements the global step counter
   atomically. When it hits zero, all Workers gracefully stop and
   return partial results. The tree shows exactly where it stopped.

4. **Observable.** The process tree is a live data structure. A TUI
   or web UI can show progress in real-time:
   ```
   Process: refactor-codebase [████████░░] 67% (670,432 / 1,000,000 steps)
   ├─ module auth [████████████] 100% (23 files, 0 errors)
   ├─ module api  [███████░░░░░] 58% (12/20 files done)
   │   ├─ api/routes.rs [running] step 4/4: verify compilation
   │   ├─ api/models.rs [pending]
   │   └─ ...
   ├─ module db   [░░░░░░░░░░░░] 0% (queued)
   └─ ...
   ```

5. **Self-healing.** When a Worker fails, its parent can retry with
   a different strategy, skip it, or escalate to a higher-level
   Worker. The tree doesn't collapse from one failure.

### The Step Counter

The million-step claim needs to be precise. What counts as a "step"?

```
1 step = 1 atomic operation:
  - 1 LLM call (any model, any size)
  - 1 tool execution (exec, read_file, etc.)
  - 1 micro-tool call (ctx_grep, mem_store, etc.)
  - 1 delegate call (spawning a child = 1 step for the parent)
```

A Worker that makes 3 LLM calls and 5 tool calls = 8 steps.
A tree of 10,000 Workers averaging 100 steps each = 1M steps.

### Coherence at Scale

The hard problem isn't executing 1M steps. It's maintaining coherence.
How does step 999,999 know what step 1 decided?

**Answer: hierarchical summarization through the tree.**

```
Leaf Workers: produce raw results (diffs, findings, data)
     ↑ mem_store into parent's context
Parent Workers: summarize children's results
     ↑ mem_store into grandparent's context
Root Worker: has a summary of everything

At any depth, a Worker knows:
- Its specific task (from the delegate call)
- Its children's results (from mem_store)
- The global plan (from the root's context, passed down)
```

No Worker needs to know everything. Each knows its neighborhood.
Coherence is maintained by the tree structure, not by any single
model's context window.

### Proof Protocol (graduated, MAKER-style)

**Phase 0: Infrastructure Validation ✅ DONE (2026-02-16)**
- Pipeline action (`action: "pipeline"`) — working with tool-equipped steps
- Loop action (`action: "loop"`) — iterative refinement working
- Subagent delegation to local models — working (Ministral-8B → 3B)
- Depth limits + budget propagation — implemented
- Strict alternation repair for local models — fixed and tested
- 6 progressive tests, 5 bugs fixed, 93.3% accuracy on real task
- **This validates the execution layer that Challenge 2 builds on.**

**Phase A: Calibration (1 day)**
1. Run Nanbeige4.1-3B on 1K random Towers of Hanoi steps (10-disk)
2. Measure per-step success rate `p`
3. Compare with MAKER's model table
4. Compute k_min, project feasibility

**Phase B: 1K Steps — Direct MAKER Comparison (1 day)**
1. 10-disk Towers of Hanoi (1,023 steps)
2. MAKER voting with measured k_min
3. Target: zero errors
4. Compare cost/time with MAKER's cloud results

**Phase C: 10K-100K Steps — Practical Task (1 week)**
1. The Refactoring Challenge: real codebase, 500+ files
2. Task: "Add comprehensive error handling to all functions"
3. MAKER voting on each atomic code edit step
4. Measure: files processed, compilation success, test pass rate
5. Target: 500+ files, 95%+ compilation success, <30 minutes

**Phase D: 1M Steps — The Big One (weekend run)**
1. 20-disk Towers of Hanoi (1,048,575 steps) — exact MAKER benchmark
2. Run on RTX 3090 with Nanbeige4.1-3B, MAKER voting
3. Target: zero errors, local hardware only, no cloud
4. **If successful: first local-hardware million-step zero-error result**

**Target:** Process 500+ files, 95%+ compilation success, complete
in <30 minutes on a single 3090 with a 3B model.

### Implementation Cost

| Component | Effort | Depends on |
|-----------|--------|------------|
| Calibration harness (sample random steps) | 2 hours | Nanbeige integration |
| StepVoter (first-to-ahead-by-k) | 2 hours | nothing |
| Red-flagging parser | 1 hour | nothing |
| Process struct + serialization | 3 hours | Worker abstraction |
| GlobalBudget with atomic counters | 1 hour | nothing |
| Checkpoint/resume | 3 hours | Process struct |
| Parallel execution (semaphore) | 2 hours | Process struct |
| Progress UI (TUI) | 3 hours | Process struct |
| Self-healing (retry/skip + prompt paraphrase) | 2 hours | Process struct |
| Towers of Hanoi benchmark adapter | 2 hours | StepVoter |

**Total: ~21 hours** (was ~14, +7 for MAKER voting + calibration)

Depends on: Challenge 1 (ContextStore) + Swarm Phase 3 (delegate)

---

## Challenge 3: Self-Evolving Agent

**The claim:** A local agent that measurably improves at its own tasks
over time — not by swapping models, but by learning from its execution
traces. After 1000 tasks, it's 10x faster and more accurate than on
task 1. On one GPU. No retraining.

### Why it sounds impossible

LLMs don't learn from experience. Each conversation starts fresh.
Fine-tuning requires expensive retraining. LoRA helps but still needs
a training loop. No local agent system today gets better by being used.

### Why it's not

**The model doesn't need to improve. The system around it does.**

A Worker that fails leaves a trace. A Worker that succeeds leaves a
recipe. Over time, the system accumulates:
- **Skills:** proven tool sequences for common tasks
- **Shortcuts:** cached results for repeated queries
- **Anti-patterns:** things that don't work (avoid these)
- **Calibration:** which tasks need more budget, which need less

The model stays the same. The prompts, tool selection, budget
allocation, and context preparation all evolve.

### The Three Learning Loops

**Loop 1: Skill Crystallization (no training, immediate)**

When a Worker tree completes successfully, extract the execution
pattern and save it as a Skill:

```
Execution trace:
  1. read_file("Cargo.toml")
  2. ctx_grep(output_0, "dependencies")
  3. python_eval("parse toml...")
  4. mem_store("deps", parsed_list)
  5. return formatted dependency list

Crystallized skill:
  name: "list-rust-dependencies"
  trigger: "list dependencies" + Cargo.toml exists
  recipe: [read_file, ctx_grep, python_eval, mem_store, return]
  avg_steps: 5
  avg_time: 2.3s
  success_rate: 98%
```

Next time someone asks "list dependencies," the system doesn't
reason from scratch. It loads the skill and executes the recipe.
The model just fills in the parameters.

**This is already half-built.** Nanobot has a skill system
(`workspace/skills/`). The gap is: skills are created manually.
The evolution is: skills are created automatically from successful
execution traces.

**First candidate for auto-crystallization (from ArXiv experiment):**
The experiment discovered that SLMs need explicit exec commands —
Ministral-8B used placeholder text when told "analyze this" but
succeeded when told "Use exec tool to run: `grep -c 'emergence'
data.csv`". This is a *prompting strategy* that should be captured
as a skill: "When delegating to SLMs, provide the exact command
rather than a high-level instruction." The skill crystallizer
should detect this pattern from the execution trace (failed attempt
→ rephrased prompt → success) and encode it for future use.

**Loop 2: Budget Calibration (no training, statistical)**

Track execution statistics per task type:

```rust
struct TaskStats {
    task_pattern: String,        // e.g. "read file and summarize"
    avg_steps: f32,
    avg_time: Duration,
    success_rate: f32,
    optimal_depth: u32,
    optimal_fan_out: u32,
    failures: Vec<FailureMode>,  // common failure patterns
}
```

After 100 tasks, the system knows:
- "File summarization needs 5 steps and depth 1"
- "Codebase refactoring needs 2000 steps and depth 3"
- "Web research needs 50 steps but high fan-out"

Budget allocation becomes data-driven instead of heuristic.
Over-budget tasks get more. Under-budget tasks get less.
Total efficiency improves without any model change.

**Loop 3: LoRA Distillation (training, periodic)**

This is where Zero connects.

Every successful Worker execution is a training example:
- Input: task description + context
- Output: tool calls + final result

Accumulate 1000 examples. Fine-tune a LoRA adapter on the local
model. The model itself gets better at the specific tasks this
user runs.

```
Week 1: 3B model + no LoRA → 60% success rate, 50 steps avg
Week 2: 3B model + LoRA v1 (500 examples) → 75% success, 30 steps
Week 4: 3B model + LoRA v2 (2000 examples) → 90% success, 15 steps
```

The model literally learns from its own successes. Failed traces
are negative examples. The LoRA gets better at:
- Choosing the right tools
- Generating correct tool arguments
- Knowing when to delegate vs. do it directly
- Producing better summaries

**This is the Zero connection.** Zero's instant learning pipeline
(L0 exact → L0.5 facts → L1 semantic → L2 delta) becomes the
training pipeline for Worker LoRAs.

### Architecture

```
┌─────────────────────────────────────────────────┐
│                  Execution Layer                  │
│                                                   │
│  Worker trees execute tasks, produce traces       │
│                                                   │
└──────────────┬────────────────────────────────────┘
               │ traces (task, tools, result, success/fail)
               ▼
┌─────────────────────────────────────────────────┐
│                  Learning Layer                   │
│                                                   │
│  ┌─────────────┐ ┌──────────────┐ ┌───────────┐ │
│  │   Skill     │ │   Budget     │ │   LoRA    │ │
│  │ Crystallizer│ │ Calibrator   │ │ Distiller │ │
│  │             │ │              │ │           │ │
│  │ Extracts    │ │ Tracks stats │ │ Trains    │ │
│  │ recipes     │ │ per task type│ │ adapters  │ │
│  │ from traces │ │ optimizes    │ │ from      │ │
│  │             │ │ allocation   │ │ successes │ │
│  └──────┬──────┘ └──────┬───────┘ └─────┬─────┘ │
│         │               │               │        │
└─────────┼───────────────┼───────────────┼────────┘
          │               │               │
          ▼               ▼               ▼
┌─────────────────────────────────────────────────┐
│                 Knowledge Layer                   │
│                                                   │
│  skills/          stats.db          loras/        │
│  ├─ list-deps/    task_type: ...    ├─ base.gguf  │
│  ├─ refactor/     avg_steps: ...    ├─ v1.gguf    │
│  ├─ research/     success: ...      ├─ v2.gguf    │
│  └─ ...           budget: ...       └─ ...        │
│                                                   │
└─────────────────────────────────────────────────┘
```

### The Flywheel

This is the key: the three loops compound.

1. User runs tasks → Workers execute → traces accumulate
2. Skills crystallize → common tasks become instant → more capacity
   for novel tasks → more traces
3. Budget calibrates → less waste → more tasks per hour → more traces
4. LoRA trains → model gets better → higher success rate → better
   traces → better skills → better LoRA

**Each loop accelerates the others.** After 1000 tasks, the system
isn't 10% better. It's 10x better. Because:
- 60% of tasks hit a cached skill (instant, no LLM call)
- 30% of tasks use calibrated budgets (3x more efficient)
- 10% of tasks are novel (but the LoRA handles them better)

### Proof Protocol

**The Learning Curve Benchmark:**
1. Generate 1000 tasks from 10 categories (file ops, web research,
   code analysis, data transformation, etc.)
2. Run all 1000 tasks sequentially
3. Measure per-task: steps, time, success/failure
4. Plot the learning curve: performance vs. task number

**Target metrics at task 1000 vs. task 1:**
- Steps per task: 5x reduction (50 → 10)
- Wall clock per task: 10x reduction (30s → 3s)
- Success rate: 60% → 95%
- Skill cache hit rate: 0% → 60%

**The overnight test:**
Leave nanobot running overnight with a queue of 1000 tasks.
Come back in the morning. Check the learning curve. The system
should be measurably better at the end than at the beginning.

### Implementation Cost

| Component | Effort | Depends on |
|-----------|--------|------------|
| Trace logger (execution recording) | 2 hours | Worker abstraction |
| Skill crystallizer (pattern extraction) | 4 hours | Trace logger |
| Skill matcher (trigger detection) | 3 hours | Skill crystallizer |
| Budget calibrator (stats tracking) | 3 hours | Trace logger |
| Budget optimizer (allocation tuning) | 2 hours | Budget calibrator |
| LoRA training pipeline | 6 hours | Zero integration |
| LoRA hot-swap (load new adapter) | 2 hours | llama.cpp integration |
| Learning curve benchmark | 3 hours | all above |

**Total: ~25 hours**

Depends on: Challenges 1 & 2 + Zero's training pipeline

---

## The Grand Unification

These three challenges aren't separate. They're layers of the same system.

```
Challenge 1 (Context)  → the system can SEE everything
Challenge 2 (Process)  → the system can DO everything
Challenge 3 (Learning) → the system IMPROVES at everything
```

Together they create something that doesn't exist yet:

**A local AI that can process unlimited context, execute million-step
plans, and get better every time it runs. On one GPU. No cloud.**

The foundation is the Worker abstraction from the swarm plan.
Everything builds on: `Worker = (task, tools, context, budget) → result`

### Timeline (updated 2026-02-16)

```
✅ DONE: Swarm infrastructure (pipeline, loop, delegate, subagent)
         → Workers can do things (validated in ArXiv experiment)

✅ DONE: Phase 0 proof (100K context, multi-agent, 93.3% accuracy)
         → Pattern works at small scale

Week 1: Challenge 1 Phase 1 (scale to 500K, stress test)
         → Workers can see more things

Week 2: Challenge 1 Phase 2 (ContextStore: mmap, indexing, 1M tokens)
         → Workers can see everything

Week 3: Challenge 2 (Process tree, MAKER voting, checkpointing)
         → Workers can do big things

Week 4: Challenge 3 (trace logging, skill crystallization, budget calibration)
         → Workers get better at things

Month 2: LoRA distillation pipeline (Zero integration)
          → The model itself gets better

Month 3: Benchmarks, optimization, documentation
          → Prove it works, ship it
```

### The Pitch

> "Everyone's building bigger models with bigger context windows.
> We built a swarm of tiny models that processes million-token
> contexts, executes million-step plans, and teaches itself to
> get better — all on a single consumer GPU.
>
> The context window isn't in the model. It's in the architecture.
> The intelligence isn't in the weights. It's in the system.
>
> nanobot: the local AI that outgrows its model."
