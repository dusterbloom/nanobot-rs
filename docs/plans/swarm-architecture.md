# Swarm Architecture: RLM Micro-Agent Network

## Vision

Transform the RLM tool delegation system from a single summarizer into a
programmable swarm of cheap micro-agents. Each micro-agent is a small model
(3B) with specialized tools. The main agent coordinates, the swarm does the
grunt work.

## Current State

### What exists
- **ContextStore** (`context_store.rs`): Variable storage + micro-tools
  (ctx_slice, ctx_grep, ctx_length, ctx_summarize). Max 2000 char slices,
  20 grep matches. Recursive summarization with depth guard.
- **ToolRunner** (`tool_runner.rs`): Delegation loop with dedup, loop
  detection, cancellation, verbatim mode. Allowed-tools safety filter.
  Short-circuit for small results.
- **SubagentManager** (`subagent.rs`): Background task spawner. Has
  read/write/exec/web tools. 15 iteration limit. Announces results via bus.

### What's missing
1. No memory persistence across RLM calls
2. No code execution (python/eval)
3. No diff/patch tool
4. No verification loop
5. No format conversion tools
6. No self-delegation (RLM spawning RLM)

## Core Abstraction: The Worker

Everything is a **Worker**. A Worker is:

```
Worker = (task, tools[], context, budget) → result
```

That's it. A Worker receives a task description, a set of tools it can
use, optional context data, and a resource budget. It runs until it
produces a result or exhausts its budget.

**The main agent is a Worker.** A subagent is a Worker. The tool runner's
delegation model is a Worker. A child spawned by delegate is a Worker.
They differ only in their tools, context, and budget — not in kind.

### Worker Properties

```rust
struct Worker {
    provider: Arc<dyn LLMProvider>,  // which model to use
    model: String,                    // model name
    tools: Vec<ToolDef>,             // available tools
    context: ContextStore,           // variables + memory
    budget: Budget,                  // iterations, depth, timeouts
}

struct Budget {
    max_iterations: u32,             // how many LLM calls
    max_depth: u32,                  // how deep can it delegate
    current_depth: u32,              // where are we now
    timeout: Duration,               // wall-clock limit
}
```

### The Recursion

When a Worker calls `delegate`, it creates a child Worker with:
- **Inherited:** provider, model (or overridden)
- **Specified:** tools (subset of parent's + micro-tools)
- **Passed:** context (explicit, not implicit)
- **Reduced:** budget (depth + 1, iterations halved, timeout shared)

The child runs synchronously from the parent's perspective — it's a
tool call that returns a string. But internally it's a full agent loop.

```
Worker A (depth=0, budget=20)
  calls delegate("analyze file X", tools=["read_file", "ctx_grep"])
    → Worker B (depth=1, budget=10)
        reads file X, greps for patterns
        calls delegate("parse this JSON block", tools=["python_eval"])
          → Worker C (depth=2, budget=5)
              runs python to parse JSON
              returns structured data
        returns analysis
  continues with B's result
```

### Why This Abstraction Matters

1. **No special cases.** The tool runner, subagent, and swarm delegate
   are all the same thing. One implementation, one test suite, one
   mental model.

2. **Composable.** Any tool can be given to any Worker. New tools
   automatically work at any depth. No wiring needed.

3. **Budget propagation is automatic.** A Worker can't create children
   that outlive its own budget. The tree is self-limiting.

4. **Provider-agnostic.** A Worker at depth 0 might be Claude Opus.
   Depth 1 might be Ministral-8B. Depth 2 might be Qwen-0.5B. Each
   level gets cheaper. Or they're all the same model. The abstraction
   doesn't care.

5. **Testable.** Mock the provider, give it tools, assert on the result.
   Same test harness works for a single Worker or a tree of 50.

### Architecture Diagram

```
User Request
    │
    ▼
┌─────────────────────────────────┐
│  Worker (main agent)            │
│  model: claude-opus             │
│  tools: [all]                   │
│  budget: {iter:50, depth:0/3}   │
│                                 │
│  tool_call: exec("cargo build") │
│      │                          │
│      ▼                          │
│  ┌──────────────────────┐       │
│  │ Worker (tool runner)  │       │
│  │ model: ministral-8b   │       │
│  │ tools: [exec,ctx_*]   │       │
│  │ budget: {iter:5,d:1/3}│       │
│  │                       │       │
│  │ ctx_grep → ctx_slice  │       │
│  │ → summarize           │       │
│  └───────┬──────────────┘       │
│          │ result                │
│          ▼                       │
│  tool_call: spawn("research")   │
│      │                          │
│      ▼                          │
│  ┌──────────────────────┐       │
│  │ Worker (subagent)     │  async│
│  │ model: ministral-8b   │       │
│  │ tools: [web,read,ctx] │       │
│  │ budget: {iter:15,d:1} │       │
│  │                       │       │
│  │ delegate("fetch p1")  │       │
│  │    │                  │       │
│  │    ▼                  │       │
│  │ ┌────────────────┐    │       │
│  │ │ Worker (child)  │    │       │
│  │ │ model: qwen-3b  │    │       │
│  │ │ tools: [web,ctx] │    │       │
│  │ │ budget:{i:5,d:2} │    │       │
│  │ └────────────────┘    │       │
│  │                       │       │
│  │ delegate("fetch p2")  │       │
│  │    │                  │       │
│  │    ▼                  │       │
│  │ ┌────────────────┐    │       │
│  │ │ Worker (child)  │    │       │
│  │ │ ...same shape   │    │       │
│  │ └────────────────┘    │       │
│  │                       │       │
│  │ mem_store + synthesize│       │
│  └───────┬──────────────┘       │
│          │ announced via bus     │
│          ▼                       │
│  continues reasoning...          │
└─────────────────────────────────┘
```

### Unification Path

Currently there are 3 separate implementations:
- `agent_loop.rs` — main agent loop
- `tool_runner.rs` — delegation loop
- `subagent.rs` — background agent loop

All three are Workers with different configs. The refactor path:

1. **Phase 1 (now):** Add new micro-tools to existing ContextStore +
   ToolRunner. Add `delegate` as a tool that creates a child ToolRunner.
   This works today without restructuring.

2. **Phase 2 (later):** Extract the common Worker abstraction. Unify
   tool_runner and subagent into a single `Worker::run()` with different
   configs. The main agent loop stays separate (it has UI, streaming,
   session management) but uses Worker internally for delegation.

3. **Phase 3 (future):** Worker becomes the unit of distribution.
   Workers can run on different machines, different GPUs, different
   models. The swarm becomes a distributed system. But the abstraction
   is the same.

## New Micro-Tools (6 tools, ~400 lines total)

### 1. `mem_store` / `mem_recall` — Persistent Scratchpad
**Purpose:** RLM remembers findings across tool calls within a delegation session.

```rust
// In context_store.rs — extend ContextStore
pub struct ContextStore {
    variables: HashMap<String, String>,   // existing: tool outputs
    memory: HashMap<String, String>,      // NEW: persistent key-value store
    counter: usize,
}
```

**Tool definitions:**
```json
{
  "name": "mem_store",
  "description": "Store a key-value pair in working memory. Persists across tool calls in this session.",
  "parameters": {
    "key": {"type": "string", "description": "Memory key (e.g. 'findings', 'urls')"},
    "value": {"type": "string", "description": "Value to store (overwrites existing)"}
  }
}
{
  "name": "mem_recall",
  "description": "Recall a value from working memory by key. Returns empty string if not found.",
  "parameters": {
    "key": {"type": "string", "description": "Memory key to recall"}
  }
}
```

**Why it matters:** Currently each tool call in the delegation loop is
stateless — the model can only see what's in the conversation history.
With mem_store/mem_recall, the RLM can build up structured findings:
- Call 1: fetch page → mem_store("headlines", "...")
- Call 2: fetch another page → mem_store("prices", "...")
- Call 3: mem_recall("headlines") + mem_recall("prices") → synthesize

**Implementation:** ~40 lines. Add `memory` HashMap to ContextStore,
add 2 tool definitions, add 2 match arms in execute_micro_tool.

**Estimated effort:** 30 minutes

---

### 2. `python_eval` — Sandboxed Code Execution
**Purpose:** RLM can do math, parse JSON, transform data via code instead
of trying to reason about it in text.

```json
{
  "name": "python_eval",
  "description": "Execute a Python expression or short script. Returns stdout. Max 5 second timeout. No network, no file I/O.",
  "parameters": {
    "code": {"type": "string", "description": "Python code to execute (print() for output)"}
  }
}
```

**Implementation:** Shell out to `python3 -c` with:
- 5 second timeout
- No network (`--network=none` if available, or just timeout)
- Capture stdout only
- Max 2000 chars output

```rust
async fn execute_python_eval(code: &str) -> String {
    let output = tokio::process::Command::new("python3")
        .arg("-c")
        .arg(code)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .output();

    match tokio::time::timeout(Duration::from_secs(5), output).await {
        Ok(Ok(out)) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            if out.status.success() {
                truncate(&stdout, 2000)
            } else {
                format!("Error: {}", truncate(&stderr, 500))
            }
        }
        Ok(Err(e)) => format!("Error: {}", e),
        Err(_) => "Error: timeout (5s)".to_string(),
    }
}
```

**Why it matters:** Small models are terrible at arithmetic and data
transformation via text. But they're fine at writing simple Python.
"Calculate the average of [1,2,3,4,5]" → `print(sum([1,2,3,4,5])/5)`.
The model doesn't need to be smart, it just needs to delegate to Python.

**Security:** This runs in the delegation context, not the main agent.
The delegation model can only call tools it was given. We can further
sandbox with `bwrap` or `nsjail` if needed, but timeout + no-network
is sufficient for v1.

**Estimated effort:** 1 hour (including sandbox considerations)

---

### 3. `diff_apply` — Surgical File Editing
**Purpose:** The main agent describes a change, the RLM reads the file,
generates a diff, and applies it — without the main agent ever loading
the full file into its context.

```json
{
  "name": "diff_apply",
  "description": "Apply a unified diff patch to a file. Returns success/failure with context.",
  "parameters": {
    "path": {"type": "string", "description": "File path to patch"},
    "diff": {"type": "string", "description": "Unified diff format (--- a/file\\n+++ b/file\\n@@ ... @@)"}
  }
}
```

**Implementation:** Write the diff to a temp file, run `patch -p0 < diff`.
Alternatively, implement a simple line-based patcher in Rust (more reliable).

```rust
async fn execute_diff_apply(path: &str, diff: &str) -> String {
    // Write diff to temp file
    let diff_path = format!("/tmp/nanobot_diff_{}.patch", uuid::Uuid::new_v4());
    std::fs::write(&diff_path, diff).ok();

    let output = tokio::process::Command::new("patch")
        .arg("--forward")
        .arg("--no-backup-if-mismatch")
        .arg(path)
        .arg(&diff_path)
        .output()
        .await;

    std::fs::remove_file(&diff_path).ok();

    match output {
        Ok(out) if out.status.success() => "Patch applied successfully.".to_string(),
        Ok(out) => format!("Patch failed: {}", String::from_utf8_lossy(&out.stderr)),
        Err(e) => format!("Error: {}", e),
    }
}
```

**Why it matters:** Currently editing a file requires:
1. Main agent reads entire file (burns context)
2. Main agent generates edit_file call
3. Tool runner executes it

With diff_apply, the flow becomes:
1. Main agent says "add error handling to function X in file Y"
2. RLM reads file Y (stays in RLM context, not main)
3. RLM generates diff
4. RLM applies diff
5. Main agent gets "Patch applied successfully" — never saw the file

**Estimated effort:** 1 hour

---

### 4. `verify` — Test and Validate
**Purpose:** RLM runs a command and checks if the output matches
expectations. Returns pass/fail with details.

```json
{
  "name": "verify",
  "description": "Run a command and check if output contains expected patterns. Returns PASS/FAIL with details.",
  "parameters": {
    "command": {"type": "string", "description": "Shell command to run"},
    "expect_contains": {"type": "array", "items": {"type": "string"}, "description": "Strings that must appear in output"},
    "expect_exit_code": {"type": "integer", "description": "Expected exit code (default 0)"}
  }
}
```

**Implementation:**
```rust
async fn execute_verify(command: &str, expect_contains: &[String], expect_exit: i32) -> String {
    let output = exec_with_timeout(command, 30).await;
    let mut failures = Vec::new();

    if output.exit_code != expect_exit {
        failures.push(format!("Exit code: got {}, expected {}", output.exit_code, expect_exit));
    }

    for pattern in expect_contains {
        if !output.stdout.contains(pattern) && !output.stderr.contains(pattern) {
            failures.push(format!("Missing pattern: '{}'", pattern));
        }
    }

    if failures.is_empty() {
        format!("PASS: command succeeded, all {} patterns found", expect_contains.len())
    } else {
        format!("FAIL:\n{}\n\nOutput preview:\n{}", failures.join("\n"), truncate(&output.stdout, 500))
    }
}
```

**Why it matters:** Write-test loops become cheap. The main agent writes
code, the RLM verifies it compiles and passes tests. Each verify call
is one cheap RLM iteration instead of the main agent burning context on
full build output.

**Estimated effort:** 45 minutes

---

### 5. `fmt_convert` — Format Transformation
**Purpose:** Convert between data formats without the model needing to
reason about syntax.

```json
{
  "name": "fmt_convert",
  "description": "Convert data between formats. Supports: json→csv, csv→json, json→yaml, yaml→json, md_table→json, json→md_table",
  "parameters": {
    "input": {"type": "string", "description": "Input data or variable name (e.g. 'output_0')"},
    "from": {"type": "string", "enum": ["json", "csv", "yaml", "md_table"]},
    "to": {"type": "string", "enum": ["json", "csv", "yaml", "md_table"]}
  }
}
```

**Implementation:** Use serde_json + csv + serde_yaml crates (already
in the dependency tree or trivially added). For md_table, simple string
parsing.

**Why it matters:** The RLM fetches a web page with a table, converts
it to JSON, greps for specific fields, returns structured data. No LLM
reasoning needed for the format conversion — it's mechanical.

**Estimated effort:** 2 hours (multiple format pairs)

---

### 6. `delegate` — Spawn a Child Worker
**Purpose:** The recursive primitive. A Worker spawns a child Worker.
The swarm emerges from this single operation.

```json
{
  "name": "delegate",
  "description": "Spawn a child worker for a sub-task. Inherits your model. Budget is halved. Returns the child's result.",
  "parameters": {
    "task": {"type": "string", "description": "What the child should accomplish"},
    "tools": {"type": "array", "items": {"type": "string"}, "description": "Tools for the child (e.g. ['read_file', 'ctx_grep', 'python_eval'])"},
    "context": {"type": "string", "description": "Data to seed the child's context (optional)"}
  }
}
```

**Implementation:** `delegate` is just `Worker::run()` with inherited
config and reduced budget. It's the same code path as the tool runner
and subagent — not a new system.

```rust
// This is conceptually what happens. The actual implementation
// reuses run_tool_loop or a shared Worker::run() function.
async fn execute_delegate(task, tools, context, parent_budget) -> String {
    let child_budget = Budget {
        max_iterations: parent_budget.max_iterations / 2,
        max_depth: parent_budget.max_depth,
        current_depth: parent_budget.current_depth + 1,
        timeout: parent_budget.remaining_time(),
    };

    if child_budget.current_depth > child_budget.max_depth {
        return "Error: depth limit reached.";
    }

    Worker::run(task, tools, context, child_budget).await
}
```

**The key insight:** delegate doesn't need special machinery. It's just
"run another Worker." The budget halving and depth increment are the
only controls needed — everything else (tool safety, loop detection,
cancellation) is inherited from the Worker abstraction.

**Emergent behaviors from this single primitive:**

- **Fan-out:** Worker reads a directory, delegates one child per file
- **Pipeline:** Worker A fetches data, delegates to Worker B to parse,
  delegates to Worker C to validate
- **Map-reduce:** Worker delegates N children, collects results in
  mem_store, synthesizes
- **Recursive search:** Worker searches a tree, delegates to children
  for subtrees

None of these patterns need to be designed. They emerge from:
`delegate + mem_store + the model's ability to plan`

**Estimated effort:** 2 hours (reuses existing Worker/ToolRunner loop)

---

## Implementation Order

### Phase 1: Memory + Verify (Day 1) — Foundation
1. **mem_store / mem_recall** — 30 min, stateful Workers
2. **verify** — 45 min, self-checking Workers

A Worker that can remember and verify is already 10x more useful.

### Phase 2: Code + Convert (Day 2) — Capabilities
3. **python_eval** — 1 hour, computational Workers
4. **fmt_convert** — 2 hours, data transformation Workers

Workers can now compute and transform, not just search and summarize.

### Phase 3: Delegate + Diff (Day 3) — The Swarm
5. **delegate** — 2 hours, recursive Workers (reuses existing loop)
6. **diff_apply** — 1 hour, surgical editing Workers

One Worker becomes many. The swarm is born.

### Total: ~8 hours across 3 days

## Integration

### Files touched

**`context_store.rs`** — Add `memory` HashMap, `mem_store()`,
`mem_recall()`, update `MICRO_TOOLS` and `execute_micro_tool`.

**`src/agent/worker_tools.rs`** (NEW) — All new tool implementations:
`python_eval`, `diff_apply`, `verify`, `fmt_convert`, `delegate`.
Pure functions, no state. Each takes args and returns a String.

**`tool_runner.rs`** — Import worker_tools, add dispatch for new tools
in the execution loop. Pass `Budget` (replaces raw depth/iterations).

**`schema.rs`** — Add `WorkerConfig` section:
```rust
struct WorkerConfig {
    enabled: bool,          // default true
    max_depth: u32,         // default 3
    python: bool,           // default true
    delegate: bool,         // default true
    budget_multiplier: f32, // default 0.5 (children get half)
}
```

## Safety Model

Safety emerges from the Worker abstraction, not from per-tool rules:

1. **Budget is conserved.** A Worker's total work (iterations * depth)
   is bounded. Children share the parent's budget, they don't add to it.
   A depth-3 tree with halving: 20 + 10 + 5 + 2 = 37 total iterations max.

2. **Tools are explicitly granted.** A child Worker only has tools the
   parent listed in the `delegate` call. No discovery, no escalation.
   Micro-tools (ctx_*, mem_*) are always available — they're internal.

3. **Context is explicitly passed.** Children don't inherit parent memory.
   The parent chooses what to share via the `context` parameter.
   This prevents information leakage between unrelated subtasks.

4. **Timeouts cascade.** A child's wall-clock timeout is the parent's
   remaining time. The whole tree finishes within the root's timeout.

5. **The existing safety filters still apply.** Exec deny-patterns,
   workspace restrictions, loop detection, dedup — all inherited by
   every Worker at every depth. One security model, applied uniformly.

## What Emerges

With just these 6 tools + the Worker abstraction, the system can:

| Pattern | How it works |
|---------|-------------|
| **Fan-out** | delegate N children, each handles one item |
| **Pipeline** | delegate A → A delegates B → B delegates C |
| **Map-reduce** | delegate N, mem_store results, synthesize |
| **Search** | delegate into subtrees recursively |
| **Write-verify** | diff_apply, then verify, retry on FAIL |
| **Compute** | python_eval for math/parsing, ctx_* for text |
| **Research** | web_fetch + delegate per-source + mem_store |

None of these are designed. They emerge from the primitives.

A 3B model that can `delegate + mem_store + python_eval + verify` is
not a chatbot. It's a **programmable workforce.**
