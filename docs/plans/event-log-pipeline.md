# Event Log & Pipeline Runner

## Status: PLAN (not yet implemented)

## Problem

Every `spawn` creates a `scratch/subagent-{id}.md` file. These accumulate
indefinitely. In a MAKER-scale scenario (10M steps), this means 10M files.
Even in normal use, a single conversation created 6 files.

## Design Principle

**One append-only JSONL file replaces all scratch files.** The event log is
infrastructure — no model ever reads or writes it directly. SLMs see the
exact same `spawn`/`wait` interface they already use.

---

## Part 1: Event Log (replaces scratch files)

### File

```
{workspace}/scratch/events.jsonl
```

### Struct (in subagent.rs)

```rust
#[derive(serde::Serialize, serde::Deserialize)]
struct SubagentEvent {
    t: String,                    // ISO 8601 timestamp
    id: String,                   // 8-char task ID
    kind: String,                 // "completed" | "failed" | "cancelled"
    pipeline: Option<String>,     // pipeline ID if part of a pipeline
    step: Option<u64>,            // step number within pipeline
    parent: Option<String>,       // parent task ID (for chaining)
    model: String,                // model used
    label: String,                // display label
    task: String,                 // task description (truncated to 200 chars)
    result: Option<String>,       // result text (truncated to 2000 chars for log)
    full_result_len: usize,       // original result length before truncation
    error: Option<String>,        // error message if failed
}
```

### Functions

```rust
/// Append one event to events.jsonl. ~10 lines.
fn append_event(workspace: &Path, event: &SubagentEvent) -> io::Result<()> {
    let path = workspace.join("scratch/events.jsonl");
    fs::create_dir_all(path.parent().unwrap())?;
    let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
    writeln!(file, "{}", serde_json::to_string(event)?)?;
    Ok(())
}

/// Rotate events.jsonl if over max_bytes. Called at startup. ~15 lines.
fn rotate_events_if_needed(workspace: &Path, max_bytes: u64) -> io::Result<()> {
    let path = workspace.join("scratch/events.jsonl");
    if path.exists() && path.metadata()?.len() > max_bytes {
        let archive = workspace.join(format!(
            "scratch/events.{}.jsonl",
            chrono::Utc::now().format("%Y%m%d-%H%M%S")
        ));
        fs::rename(&path, &archive)?;
        // Optional: compress in background (shell `zstd`)
    }
    Ok(())
}

/// Read result for a task ID from events.jsonl (fallback for wait). ~15 lines.
fn read_event_result(workspace: &Path, task_id: &str) -> Option<String> {
    let path = workspace.join("scratch/events.jsonl");
    if !path.exists() { return None; }
    // Read from end (most recent first) for efficiency
    let content = fs::read_to_string(&path).ok()?;
    for line in content.lines().rev() {
        if line.contains(task_id) {
            if let Ok(event) = serde_json::from_str::<SubagentEvent>(line) {
                if event.id.starts_with(task_id) {
                    return event.result;
                }
            }
        }
    }
    None
}
```

### Changes to existing code

**`_write_result_file()` → `_append_event()`**

Replace the function body. Instead of writing a markdown file, construct a
`SubagentEvent` and call `append_event()`. The full result is still passed
via broadcast channel (in-memory) — the event log stores a truncated version
for forensics.

**`wait_for()` fallback**

Currently scans scratch dir for `subagent-{id}.md` files. Change to call
`read_event_result()` instead. The in-memory broadcast path stays unchanged.

**Startup cleanup**

Call `rotate_events_if_needed(workspace, 100_000_000)` (100MB) at startup.
Delete any legacy `scratch/subagent-*.md` files.

---

## Part 2: Pipeline Runner (Rust-native, not LLM)

### New tool: `pipeline`

Added to spawn.rs as a new action: `action: "pipeline"`.

```json
{
    "action": "pipeline",
    "plan": ["step 1 instruction", "step 2 instruction", ...],
    "model": "local",
    "pipeline_id": "hanoi-001",
    "state": "{initial state as string}",
    "options": {
        "voting_k": 3,
        "max_retries": 3,
        "state_template": "Previous state: {state}\n\nTask: {step}\n\nReturn only the new state."
    }
}
```

### Runner (new function in subagent.rs, ~80 lines)

```rust
async fn run_pipeline(
    &self,
    pipeline_id: &str,
    plan: Vec<String>,
    model: &str,
    initial_state: Option<String>,
    voting_k: u32,         // MAKER-style first-to-ahead-by-k
    max_retries: u32,
    state_template: &str,  // "{state}" and "{step}" placeholders
    origin_channel: &str,
    origin_chat_id: &str,
) -> String {
    let mut state = initial_state.unwrap_or_default();
    let total = plan.len();
    
    for (i, step) in plan.iter().enumerate() {
        let task = state_template
            .replace("{state}", &state)
            .replace("{step}", step)
            .replace("{i}", &i.to_string())
            .replace("{total}", &total.to_string());
        
        // Spawn worker, wait for result
        let result = self.spawn_and_wait(&task, model, voting_k, max_retries).await;
        
        match result {
            Ok(new_state) => {
                // Append event
                append_event(workspace, &SubagentEvent {
                    pipeline: Some(pipeline_id.to_string()),
                    step: Some(i as u64),
                    kind: "completed".to_string(),
                    result: Some(new_state.clone()),
                    // ... other fields
                });
                state = new_state;
            }
            Err(e) => {
                append_event(workspace, &SubagentEvent {
                    pipeline: Some(pipeline_id.to_string()),
                    step: Some(i as u64),
                    kind: "failed".to_string(),
                    error: Some(e.to_string()),
                    // ...
                });
                return format!("Pipeline failed at step {}/{}: {}", i, total, e);
            }
        }
        
        // Progress reporting every N steps
        if i % 100 == 0 {
            info!("Pipeline {} progress: {}/{}", pipeline_id, i, total);
        }
    }
    
    format!("Pipeline {} completed: {}/{} steps. Final state: {}", 
            pipeline_id, total, total, truncate_for_display(&state, 10, 500))
}
```

### Voting (MAKER-style, ~30 lines)

```rust
/// Run a single step with first-to-ahead-by-k voting.
async fn spawn_and_wait_voted(
    &self, task: &str, model: &str, k: u32, max_retries: u32
) -> Result<String> {
    let mut votes: HashMap<String, u32> = HashMap::new();
    let mut attempts = 0;
    
    loop {
        let result = self._run_single_step(task, model).await?;
        
        // Red-flagging: if result looks malformed, don't count it
        if self.is_red_flagged(&result) {
            attempts += 1;
            if attempts > max_retries * k { return Err("too many red flags"); }
            continue;
        }
        
        *votes.entry(result.clone()).or_insert(0) += 1;
        
        // Check if any answer is ahead by k
        let max_votes = votes.values().max().unwrap_or(&0);
        let second_max = votes.values().sorted().rev().nth(1).unwrap_or(&0);
        if max_votes - second_max >= k {
            return Ok(votes.into_iter().max_by_key(|(_, v)| *v).unwrap().0);
        }
        
        attempts += 1;
        if attempts > max_retries * k * 2 { return Err("no consensus"); }
    }
}
```

### Resume support (~10 lines)

```rust
/// Find last completed step for a pipeline in the event log.
fn find_resume_point(workspace: &Path, pipeline_id: &str) -> Option<u64> {
    let path = workspace.join("scratch/events.jsonl");
    if !path.exists() { return None; }
    let content = fs::read_to_string(&path).ok()?;
    content.lines().rev()
        .filter_map(|l| serde_json::from_str::<SubagentEvent>(l).ok())
        .filter(|e| e.pipeline.as_deref() == Some(pipeline_id) && e.kind == "completed")
        .map(|e| e.step.unwrap_or(0))
        .next()
}
```

---

## Part 3: SLM Compatibility

### Key principle: SLMs never see the event log

The event log is written by Rust code, not by any model. The `spawn`/`wait`
interface is unchanged. An SLM worker receives a plain text task and returns
a plain text result. It doesn't know about pipelines, events, or logs.

### Pipeline tool usage by model tier

| Model tier | Can use `spawn`? | Can use `pipeline`? | Role |
|-----------|-----------------|-------------------|------|
| 3B (Nanbeige) | As worker only | No (too complex) | Pure function worker |
| 8-14B (Mistral, Qwen) | Yes | Simple plans only | Light orchestration |
| Haiku+ | Yes | Full plans + voting | Full orchestration |
| Opus | Yes | Everything | Architecture + debug |

### Worker prompt template (skill)

```markdown
# skills/pipeline-worker/SKILL.md

You receive a task with STATE and ACTION.
Execute the ACTION on the STATE.
Return ONLY the new state. No explanation. No commentary.
Format: exact same format as the input state.
```

### Orchestrator prompt template (skill)

```markdown
# skills/pipeline-orchestrator/SKILL.md

You are a pipeline planner. Given a goal:
1. Break it into sequential steps
2. Each step must be a self-contained instruction
3. Each step receives the previous step's output as state
4. Return a JSON array of step strings

Format: ["step 1", "step 2", ...]
```

---

## Diff Summary

### subagent.rs (~100 lines changed)

1. **Add** `SubagentEvent` struct with serde derive (~15 lines)
2. **Add** `append_event()` function (~10 lines)
3. **Add** `rotate_events_if_needed()` function (~15 lines)
4. **Add** `read_event_result()` function (~15 lines)
5. **Replace** `_write_result_file()` body → construct event + call append (~10 lines)
6. **Modify** `wait_for()` fallback → call `read_event_result()` instead of scanning dir (~5 lines)

### subagent.rs (~120 lines added for pipeline)

7. **Add** `run_pipeline()` method (~80 lines)
8. **Add** `spawn_and_wait_voted()` method (~30 lines)
9. **Add** `find_resume_point()` function (~10 lines)

### spawn.rs (~30 lines added)

10. **Add** `"pipeline"` action branch in `execute()` (~25 lines)
11. **Add** `pipeline_callback` type + setter (~5 lines)

### main.rs or startup code (~5 lines)

12. **Call** `rotate_events_if_needed()` at startup
13. **Cleanup** legacy `scratch/subagent-*.md` files

### New files

14. `skills/pipeline-worker/SKILL.md` (~10 lines)
15. `skills/pipeline-orchestrator/SKILL.md` (~15 lines)

### Total: ~270 lines changed/added. Zero new dependencies.

---

## Implementation Order

1. **Event log** (Part 1) — smallest change, immediate value, removes scratch file accumulation
2. **Pipeline runner** (Part 2) — builds on event log, enables MAKER-scale
3. **Skills** (Part 3) — just markdown files, enables SLM participation

Each part is independently shippable and testable.

---

## What This Enables

- **Normal use:** `spawn + wait` works exactly as today, but no file accumulation
- **Debugging:** `cat events.jsonl | grep pipeline-id` — full audit trail
- **Resume:** Crash at step 847,293 → restart from 847,293 automatically
- **Analytics:** `jq '.model' events.jsonl | sort | uniq -c` — cost per model
- **MAKER-scale:** 10M steps × ~300 bytes = ~3GB JSONL, rotated at 100MB
- **SLM-friendly:** Nanbeige 3B does single steps. Rust does the loop.
