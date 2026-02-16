# Large-Context Multi-Agent Experiment: Results Memo

**Date**: 2026-02-16
**Branch**: vibe-1771266698
**Result**: 93.3% overall score — PASS

## What We Tested

Can nanobot process 88 ArXiv papers (large CSV) using only local SLMs on an RTX 3090, with multi-agent orchestration (main model + subagents)?

- **Main model**: Ministral-8B on port 8080 (14K context)
- **Subagent model**: Ministral-3B on port 8083 (8K context)
- **Data**: 88 papers from ArXiv, CSV format with titles, authors, abstracts
- **Validation**: Author coverage, keyword coverage, emergence topic detection

## What We Found

### Final Scores
| Metric | Score | Detail |
|--------|-------|--------|
| Author Coverage | 80% | Found 4/5 top authors (Zhan Qu, Michael Farber, Yue Huang, +2 others) |
| Keyword Coverage | 100% | emergence, scaling, chain-of-thought all detected |
| Emergence Discussed | Yes | Identified "Constraint-Rectified Training for Efficient Chain-of-Thought" |
| **Overall** | **93.3%** | **PASS** |

### Test Suite Results (6 tests)
| Test | Description | Result |
|------|-------------|--------|
| 1 | Basic exec (smoke test) | PASS |
| 2 | Author extraction via exec | PASS — found top 2 correctly |
| 3 | Keyword search via exec | PASS — found emergence papers |
| 4 | Subagent spawn to Ministral-3B | PASS — Jinja fix verified |
| 5 | Pipeline action (3-step) | SOFT PASS — model confused by JSON prompt |
| 6 v2 | Full workflow (exec + synthesis) | PASS — 93.3% validated |

## Bugs Fixed (5 total)

### Bug 1: Pipeline missing strict alternation repair
`execute_step_with_tools()` in `pipeline.rs` didn't call `repair_for_strict_alternation()` for local models. Tool result messages (`role: "tool"`) broke Jinja templates.

### Bug 2: Context overflow from workspace files
The default workspace at `~/.nanobot/workspace` contains 96KB of markdown (AGENTS.md, SOUL.md, research docs, pricing guides). This gets loaded into the system prompt, blowing the 16K context budget. Fixed with a minimal 1-line workspace.

### Bug 3: Config field at wrong JSON level
`defaultSubagentModel` was placed at the JSON root level. Serde expects it inside the `toolDelegation` object. No error — silently ignored, causing subagents to fall back to the expensive main model.

### Bug 4: Provider-routed local models detected as cloud
When a model uses a provider prefix like `groq/Ministral-3B`, the code set `routed_to_cloud = true` even though the groq provider points to `localhost:8083`. This skipped strict alternation repair, causing Jinja errors.

### Bug 5: Consecutive user messages after repair (ROOT CAUSE)
This was the elusive Jinja error that persisted through all other fixes. The execution flow was:

```
1. Model returns tool call → add assistant message + tool results
2. repair_for_strict_alternation() → converts tool messages to user role
3. Append user continuation: "Based on the tool results above..."
```

After step 2, the thread ends with a user message (converted tool result). Step 3 adds *another* user message, creating:
```
["system", "user", "assistant", "user", "user"]
                                         ^^^^^ DUPLICATE
```

The Jinja template rejects consecutive same-role messages. The fix: remove step 3 entirely — the repair already ensures the thread ends with a user message.

The `agent_loop.rs` already had a comment warning about this exact issue ("Do NOT add extra user continuation — it would create consecutive user messages") but `subagent.rs` had the old pattern.

## Key Learnings

1. **SLMs need explicit tool instructions**: "Use the exec tool to run this exact command" works. "Run these commands" gets placeholder text.
2. **Debug with message roles**: Adding `roles={:?}` logging to the provider instantly revealed the consecutive-user bug that direct API testing couldn't find.
3. **Fix everywhere rule pays off**: The agent_loop.rs already had the correct pattern; the bug was only in subagent.rs. Grep-based verification caught it.
4. **Silent config errors are deadly**: Serde's `deny_unknown_fields` would have caught the misplaced `defaultSubagentModel` immediately.
5. **Minimal workspaces for small models**: 96KB of workspace files in a 16K context budget leaves almost nothing for the actual task.

## Files Changed

| File | Change |
|------|--------|
| `src/agent/subagent.rs` | `resolve_provider_for_model` returns `targets_local`; removed redundant user continuation after repair; updated `is_local` detection at both call sites |
| `src/agent/pipeline.rs` | Added `thread_repair` import and strict alternation repair in tool loop |
| `src/providers/openai_compat.rs` | No permanent changes (temp debug logging removed) |
| `experiments/large-context-test/` | New test scripts, workspace, results |
| `thoughts/ledgers/CONTINUITY_CLAUDE-large-context-experiment.md` | Updated with results |
