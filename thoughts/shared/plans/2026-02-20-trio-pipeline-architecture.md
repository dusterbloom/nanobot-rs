# Trio Pipeline Architecture

**Date**: 2026-02-20
**Status**: Design draft
**Depends on**: B3 (default trio), I1 (role/protocol crashes), LM Studio trio integration
**Blocks**: P2.4 (RLM completion), P2.5 (swarm architecture)

## Problem

The trio architecture (Main 3B + Router 8B + Specialist 8B) can only execute **one action per router decision**. Multi-step tasks like "research RLM from this paper and this crate" require:

1. Fetch URL A
2. Fetch URL B
3. Synthesize a report from both

The router picks ONE of these and the pipeline stalls. The deterministic fallback (`router_fallback.rs`) is even worse — it matches the first URL and does a bare `web_fetch`.

No single local model can do this alone. But three local models collaborating through shared context **can** — if the orchestration supports it.

## Design Principles

1. **No cloud dependency** — the trio must be self-sufficient on local hardware
2. **Each model does what it's best at** — 3B talks, 8B plans and writes, Rust executes
3. **Shared context, not shared history** — models see only what they need via scratchpad
4. **Recursive but bounded** — loops have circuit breakers (max rounds, token budget, timeout)
5. **Incremental, not big-bang** — build on existing `router.rs`, `toolplan.rs`, `role_policy.rs`

## Architecture

### Current Flow (single action)

```
User → Main(3B) → Router(8B) → ONE action → done or stall
```

### Proposed Flow (pipeline)

```
User → Main(3B) → Router(8B) → Pipeline [s1, s2, s3...] → Executor → done
                                    │                           │
                                    │    ┌──────────────────────┘
                                    │    │
                                    ▼    ▼
                              Shared Scratchpad
                              ┌─────────────────┐
                              │ s1: (tool result)│
                              │ s2: (tool result)│
                              │ s3: (specialist) │
                              └─────────────────┘
```

### Role Responsibilities

| Role | Model | Sees | Does |
|------|-------|------|------|
| **Main** | 3B | User message, final result | Parse intent, present output |
| **Router** | 8B | Intent, tools, scratchpad status | Plan pipeline, review quality |
| **Specialist** | 8B | Scratchpad contents for its step | Synthesize, analyze, generate |
| **Executor** | Rust | Pipeline DAG | Run tools, manage scratchpad, schedule steps |

### Pipeline Schema

```rust
struct PipelineStep {
    id: String,                    // "s1", "s2", etc.
    action: PipelineAction,        // Tool | Specialist | RouterReview
    target: String,                // tool name or specialist task type
    args: Value,                   // tool args or specialist prompt template
    needs: Vec<String>,            // dependency step IDs (DAG edges)
    max_chars: usize,              // context budget for this step's input
}

enum PipelineAction {
    Tool,           // Execute a tool, store result in scratch[id]
    Specialist,     // Call specialist with scratch[needs] as context
    RouterReview,   // Router evaluates results, may extend pipeline
}

struct Pipeline {
    steps: Vec<PipelineStep>,
    max_rounds: u8,                // circuit breaker for recursive review
    token_budget: usize,           // total token budget across all steps
}
```

### Router Output (extended)

The `route_decision` tool gets a new action:

```json
{
  "action": "pipeline",
  "steps": [
    {"id": "s1", "action": "tool", "target": "web_fetch", "args": {"url": "https://arxiv.org/..."}, "needs": []},
    {"id": "s2", "action": "tool", "target": "web_fetch", "args": {"url": "https://crates.io/..."}, "needs": []},
    {"id": "s3", "action": "specialist", "target": "synthesize", "args": {"task": "Write a research report combining s1 and s2"}, "needs": ["s1", "s2"]}
  ],
  "confidence": 0.85
}
```

Steps with no `needs` run in parallel. Steps with `needs` wait for dependencies.

### Execution Flow (RLM example)

```
t0  Main(3B): classifies intent → "multi-step research with URLs"     ~50 tok
t1  Router(8B): plans pipeline [s1: fetch, s2: fetch, s3: specialist] ~200 tok
t2  Executor: runs s1 ∥ s2 in parallel (pure Rust, 0 tokens)
    scratch["s1"] = arxiv content (truncated to max_chars)
    scratch["s2"] = crate docs (truncated to max_chars)
t3  Specialist(8B): receives s1+s2 content, writes report             ~1500 tok
t4  Main(3B): wraps specialist output, presents to user               ~100 tok

Total: ~1850 tokens, ~8 seconds, fully local
```

### Recursive Refinement (optional)

If a pipeline includes a `RouterReview` step, the router can inspect results and extend the pipeline:

```
Round 1: Router plans → Executor runs → Specialist writes
Round 2: Router reviews scratch["s3"]
         → "Report is missing crate usage examples"
         → Adds s4: web_search("rlm-cli usage examples")
         → Adds s5: specialist(synthesize s3+s4)
Round 3: Router reviews scratch["s5"] → "Complete" → done
```

Circuit breaker: `max_rounds` (default 3), total token budget, wall-clock timeout.

### Scratchpad Context Injection

Each model gets a **filtered view** of the scratchpad:

- **Router** sees: step IDs, status (pending/done/failed), output size — NOT full content
- **Specialist** sees: full content of steps listed in its `needs` — truncated to `max_chars`
- **Main** sees: final step output only

This keeps context windows small. A 3B model with 4K context can still present a report that was synthesized by an 8B model with 32K context.

### Deterministic Fallback (enhanced)

When the router fails to produce a valid pipeline, `router_fallback.rs` gets new patterns:

```rust
// Research/report + URLs → pipeline template
if (has_research_intent && has_urls && has_tool("web_fetch")) {
    let urls = extract_urls(user_text);
    let mut steps: Vec<PipelineStep> = urls.iter().enumerate().map(|(i, url)| {
        PipelineStep {
            id: format!("s{}", i + 1),
            action: PipelineAction::Tool,
            target: "web_fetch".into(),
            args: json!({"url": url}),
            needs: vec![],
            max_chars: 4000,
        }
    }).collect();
    let dep_ids: Vec<String> = steps.iter().map(|s| s.id.clone()).collect();
    steps.push(PipelineStep {
        id: format!("s{}", steps.len() + 1),
        action: PipelineAction::Specialist,
        target: "synthesize".into(),
        args: json!({"task": user_text}),
        needs: dep_ids,
        max_chars: 8000,
    });
    return Pipeline { steps, max_rounds: 1, token_budget: 4000 };
}
```

## Implementation Plan

### Step 1: Pipeline types (~50 lines)
Add `PipelineStep`, `PipelineAction`, `Pipeline` to `toolplan.rs`. Conversion from `RouterDecision` with `action: "pipeline"`.

### Step 2: Pipeline executor (~200 lines)
New `src/agent/pipeline_executor.rs`:
- DAG scheduler (topological sort on `needs`)
- Parallel tool execution via `tokio::JoinSet`
- Scratchpad storage (in-memory `HashMap<String, String>`)
- Specialist dispatch reusing existing `dispatch_specialist()`
- Context injection: build specialist prompt from scratchpad entries

### Step 3: Router schema update (~30 lines)
Add `"pipeline"` to the `action` enum in `request_strict_router_decision()`. Add `steps` array to the tool schema. Keep backward compat — single actions still work.

### Step 4: Integration into `router.rs` (~50 lines)
In `router_preflight()` and `route_tool_calls()`: when decision is `pipeline`, hand off to executor instead of dispatching a single action.

### Step 5: Enhanced fallback patterns (~40 lines)
Add research/report, multi-URL, and build-cycle templates to `router_fallback.rs`.

### Step 6: Recursive review (optional, ~80 lines)
`RouterReview` step type. Router inspects scratchpad, may append steps. Bounded by `max_rounds`.

**Total estimate: ~450 lines of new code**, mostly in a new file. No major refactoring of existing modules.

## Relationship to Existing Work

- **B3 (default trio)**: Must land first — need working trio before adding pipelines
- **I0 (this item)**: The core pipeline work
- **I1 (role/protocol crashes)**: Must be stable before multi-step flows
- **I3 (Context Gate)**: Scratchpad truncation is a simpler version of ContentGate
- **P2.3 (process tree)**: Pipeline is a lightweight precursor — same DAG concept, smaller scope
- **P2.4 (RLM completion)**: Pipelines enable recursive multi-step completion
- **P2.5 (swarm)**: Pipeline executor is a single-agent swarm — same scheduling patterns

## Open Questions

1. **Should the router learn pipeline planning, or should we template it?** The 8B router might not reliably produce valid pipeline JSON. Templated pipelines (detect pattern → fill template) are more reliable but less flexible.

2. **How big can scratchpad entries be?** A web_fetch result can be 50KB+. Need truncation strategy per step. The `max_chars` field handles this, but optimal values need tuning.

3. **What if the specialist needs tools?** Current design keeps specialist tool-free (focused synthesis). But some tasks (e.g., "write code and test it") need tool access. Could add a `SpecialistWithTools` action that gives the specialist a mini tool loop.

4. **Token accounting across models**: Three models running means 3x the token tracking. Need unified budget that spans the pipeline.
