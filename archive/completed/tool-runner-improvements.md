# Tool Runner Improvements

Based on the infinite loop observed (15x identical `exec` calls), here are
proposed improvements ranked by impact.

## Root Cause Analysis

The delegation model (local RLM) re-requested the same `exec` command 15 times.
The existing `seen_calls` dedup uses string comparison of `name:json_args`, but
JSON serialization differences (whitespace, key ordering) can bypass this.

Additionally, the local model may not understand when to stop because:
1. The system prompt is too vague ("Execute tools as requested")
2. The user continuation message is open-ended ("do you need to call more tools?")
3. Small models have weak instruction-following for tool use stop conditions

---

## Fix 1: Normalize JSON for Dedup (Quick Win)

**Problem:** `{"command":"echo hi"}` vs `{"command": "echo hi"}` are different strings.

**Fix:** Parse and re-serialize with sorted keys before inserting into `seen_calls`:

```rust
fn normalize_call_key(name: &str, arguments: &HashMap<String, Value>) -> String {
    // Sort keys and use compact serialization
    let mut sorted: Vec<_> = arguments.iter().collect();
    sorted.sort_by_key(|(k, _)| k.clone());
    let normalized = serde_json::to_string(&sorted).unwrap_or_default();
    format!("{}:{}", name, normalized)
}
```

---

## Fix 2: Improve System Prompt (High Impact)

**Current:**
```
Execute tools as requested. When done, summarize findings with specific 
data points, values, and counts. Do not ask questions.
```

**Proposed:**
```
You are a tool execution assistant. Your ONLY job is to summarize tool results.

RULES:
1. The tools have ALREADY been executed. The results are shown above.
2. DO NOT call any tools. Just summarize what the results show.
3. Respond with a brief summary of the tool outputs.
4. Never re-execute a tool that has already been run.
5. If results are clear, summarize them in 1-2 sentences.
```

**Why:** The current prompt says "Execute tools as requested" which the model
interprets as permission/instruction to keep executing. The model needs to
understand its role is to SUMMARIZE, not to EXECUTE.

---

## Fix 3: Don't Pass Tools on Summary Turn (High Impact)

After the initial tool calls are executed, the delegation model is asked
"do you need more tools?" with tool definitions still available. Small models
see available tools and use them reflexively.

**Fix:** On the first iteration (when we just want a summary of the initial
results), don't pass tool definitions:

```rust
// First iteration: just summarize, don't offer tools
let tools_for_call = if iteration == 0 && !config.allow_chaining {
    None
} else {
    tool_defs_opt
};
```

Or simpler: add a `allow_chaining: bool` config flag. When false, never pass
tools to the delegation model — it can only summarize.

---

## Fix 4: Improve User Continuation Message (Medium Impact)

**Current:**
```
Based on the tool results above, decide: do you need to call more tools, 
or can you provide a summary of what was found?
```

**Proposed:**
```
The tool has been executed and the results are shown above. 
Provide a brief summary of the results. Do NOT call any more tools.
```

**Why:** The current message explicitly offers the choice to call more tools.
Small models with weak instruction-following will take the "call more tools"
path because it's the path of least resistance (they see tool definitions and
a permission to use them).

---

## Fix 5: Result-Based Loop Detection (Medium Impact)

Beyond dedup on call signature, detect when results are identical:

```rust
// Track result hashes
let result_hash = hash(&result.data);
if last_result_hashes.contains(&(tc.name.clone(), result_hash)) {
    warn!("Same tool producing identical results — stopping");
    break;
}
last_result_hashes.insert((tc.name.clone(), result_hash));
```

---

## Fix 6: Exponential Backoff on Iterations (Low Priority)

Add increasing delays between iterations to prevent rapid-fire loops:

```rust
if iteration > 0 {
    tokio::time::sleep(Duration::from_millis(100 * 2u64.pow(iteration))).await;
}
```

---

## Fix 7: Two-Phase Architecture (Best Long-Term)

Separate the tool runner into two distinct phases:

### Phase 1: Execute
Run all tool calls from the main model. No LLM involved. Just execute and
collect results.

### Phase 2: Summarize (optional)
If results are large (> threshold), ask the RLM to summarize. But:
- Do NOT pass tool definitions
- Use a pure summarization prompt
- Single-shot, no loop

```rust
pub async fn run_tools_and_summarize(
    config: &ToolRunnerConfig,
    tool_calls: &[ToolCallRequest],
    tools: &ToolRegistry,
) -> ToolRunResult {
    // Phase 1: Execute all tools (no LLM)
    let mut results = Vec::new();
    for tc in tool_calls {
        let result = tools.execute(&tc.name, tc.arguments.clone()).await;
        results.push((tc.id.clone(), tc.name.clone(), result.data));
    }
    
    // Phase 2: Summarize if needed (no tools available)
    let total_chars: usize = results.iter().map(|(_, _, d)| d.len()).sum();
    let summary = if total_chars > config.summarize_threshold {
        summarize_results(&config, &results).await
    } else {
        None
    };
    
    ToolRunResult { tool_results: results, summary, iterations_used: 1 }
}
```

This eliminates the loop entirely for the common case (main model requests
tools, tools run, results go back). Chaining is only needed when the
delegation model genuinely needs multi-step reasoning, which is rare for
small models.

---

## Fix 8: Lower max_iterations Default

Current default is unclear from config. For local models, `max_iterations: 3`
is plenty. Most legitimate chains are 1-2 steps.

---

## Priority Order

1. **Fix 3** (don't pass tools on summary) — eliminates the root cause
2. **Fix 2** (better system prompt) — reinforces correct behavior  
3. **Fix 7** (two-phase architecture) — best long-term solution
4. **Fix 1** (normalize JSON dedup) — catches edge cases
5. **Fix 4** (better continuation message) — belt and suspenders
6. **Fix 5** (result-based loop detection) — extra safety net
7. **Fix 8** (lower max_iterations) — damage control
8. **Fix 6** (exponential backoff) — nice to have

Fixes 2+3 together would likely eliminate the infinite loop problem entirely.
Fix 7 is the cleanest long-term architecture.
