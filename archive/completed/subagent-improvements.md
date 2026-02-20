# Subagent Reliability Improvements

## Problem Statement

When the parent agent spawns subagents, it wastes tool iterations polling `spawn list`
and can lose subagent results to context compaction. This burns 30+ iterations out of
a 20-iteration budget just checking status, leaving no budget for actual work.

## Four Improvements

### 1. `spawn wait <task_id>` — Block until subagent completes ⭐ HIGH IMPACT

**What:** Add `action: "wait"` to the spawn tool that blocks (with timeout) until
a specific subagent finishes, then returns its result directly.

**Files to modify:**
- `src/agent/tools/spawn.rs` — Add "wait" action, new `WaitCallback` type
- `src/agent/subagent.rs` — Add `wait_for()` method using `tokio::sync::oneshot`
- `src/agent/agent_loop.rs` — Wire up the wait callback

**Approach:**
- In `SubagentManager::spawn()`, store a `broadcast::Sender<String>` alongside the JoinHandle
- `wait_for(task_id, timeout)` subscribes to the broadcast and awaits the result
- The spawned task sends its result through the broadcast channel before removing itself
- Default timeout: 120s (configurable)
- Returns the full result text, not just status

**Risk:** LOW — additive change, no existing code paths modified

### 2. Subagent output files — Write results to disk automatically ⭐ HIGH IMPACT

**What:** Subagents automatically write their final output to
`{workspace}/scratch/subagent-{task_id}.md` before announcing.

**Files to modify:**
- `src/agent/subagent.rs` — Write result file in the spawned task, after `_run_subagent`

**Approach:**
- After `_run_subagent` returns, write to `{workspace}/scratch/subagent-{task_id}.md`
- Include metadata header: task, label, status, timestamp
- The `spawn wait` response can reference this file path
- The `spawn list` response shows file paths for completed tasks
- Files are ephemeral — cleaned up after 24h by cron or on next start

**Risk:** VERY LOW — purely additive, just a file write after existing logic

### 3. Tool budget counter — Show remaining iterations ⭐ MEDIUM IMPACT

**What:** Inject remaining iteration count into tool responses so the agent
can plan accordingly.

**Files to modify:**
- `src/agent/agent_loop.rs` — Add iteration info to system context

**Approach:**
- After each iteration in the main loop (line ~605), inject a small note into
  the response boundary message: `[iteration X/20, Y remaining]`
- Alternatively, add it as metadata in the tool result messages
- Keep it minimal — just a number, not a whole paragraph

**Risk:** LOW — small addition to existing response boundary logic

### 4. Compaction preserves subagent results ⭐ MEDIUM IMPACT

**What:** Tag subagent result messages as high-priority so the compactor
doesn't summarize them away.

**Files to modify:**
- `src/agent/compaction.rs` — Respect priority tags during compaction
- `src/agent/agent_loop.rs` — Tag subagent-origin messages

**Approach:**
- When subagent results are injected into the conversation (via tool runner
  summary or direct injection), mark them with a metadata flag
- The compactor preserves messages with this flag, treating them like
  system messages that shouldn't be summarized
- Note: Currently subagent results go to the BUS as OutboundMessage, not
  into the agent's conversation. The real fix is #1 (wait) which puts results
  directly into the tool response. This improvement is a safety net for
  the case where results ARE in context.

**Risk:** MEDIUM — touches compaction logic which is sensitive

## Implementation Order

1. **#2 (output files)** ✅ DONE — subagent.rs: writes to `workspace/scratch/subagent-{id}.md`
2. **#1 (spawn wait)** ✅ DONE — spawn.rs + subagent.rs + agent_loop.rs: `action: "wait"` with broadcast channel
3. **#3 (budget counter)** ✅ DONE — agent_loop.rs: budget warning when ≤5 iterations remain
4. **#4 (compaction)** ✅ DONE — compaction.rs: subagent result messages force-included in "recent" set

## Test Results

- 965 passed, 1 failed (pre-existing: `test_web_search_no_api_key`), 4 ignored
- All subagent tests (4) pass
- All spawn tests (16) pass
- All compaction tests (13) pass
- Binary builds cleanly (139 pre-existing warnings, 0 errors)
