# Narration Compliance Stress Test

**Status:** TODO
**Priority:** High — must validate before shipping narration as a feature
**Created:** 2026-02-16
**Context:** SOUL.md now has "Streaming Narration" rules (checkpoint every 3 calls, no silent chains). Works great with Opus/Sonnet. Will it survive local models?

## The Problem

Cloud models (Opus, Sonnet, even Haiku) follow system prompt instructions reliably.
Local 3B-8B models often:
- Ignore soft instructions under cognitive load
- Drop narration when deep in multi-step tool use
- Produce garbage narration ("Let me check... let me check... let me check...")
- Skip narration entirely and just emit tool calls

If narration only works on $15/MTok models, it's not a real feature.

## Test Matrix

### Models to test
| Model | Size | Expected difficulty |
|-------|------|-------------------|
| Nanbeige4.1-3B | 3B | 🔴 Hard — smallest, most likely to fail |
| Qwen3-30B-A3B | 30B (3B active) | 🟡 Medium — MoE, good benchmarks but active params are small |
| Qwen3-0.6B | 0.6B | 🔴 Extreme — memory model, probably can't narrate at all |
| Llama-3.2-8B | 8B | 🟡 Medium — baseline local model |
| glm-4.5-air (cloud) | ? | 🟢 Easy — cloud RLM baseline, should work |

### Tasks to test (increasing difficulty)
1. **Simple:** "Read this file and summarize it" (1 tool call — narration is just before/after)
2. **Medium:** "Find all TODO comments in src/" (grep + read 3-5 files — tests checkpoint rule)
3. **Hard:** "Refactor this function into a separate module" (read + write + edit + test — 6+ tools)
4. **Extreme:** "Review the codebase and suggest improvements" (open-ended, 10+ tool calls)

### What to measure
- **Compliance rate:** % of tool batches preceded by narration text
- **Quality:** Is the narration useful or just filler? ("I'll check" vs "The config has 3 sections, now updating the parser")
- **Silent chain max:** Longest streak of consecutive tool calls with zero text between them
- **Degradation pattern:** Does narration start strong then fade? Or never appear?

## Fallback Strategies (if local models fail)

### Option A: Runtime-forced narration
If model emits 3+ tool calls without text, the agent loop injects a synthetic
"What are you doing?" prompt. Forces the model to narrate. Adds latency but
guarantees compliance.

### Option B: Simplified narration for small models
Different SOUL.md section for local models:
- "Before tools, write ONE WORD about what you're doing: reading/writing/searching/testing"
- Lower the bar from sentences to keywords
- Even "reading..." before a read_file is better than silence

### Option C: Display-layer narration
Don't ask the model to narrate — instead, the display layer generates
human-readable descriptions from tool events:
- `read_file(src/main.rs)` → "📖 Reading src/main.rs..."
- `exec(cargo test)` → "🔨 Running tests..."
- `edit_file(...)` → "✏️ Editing src/config.rs..."
This is model-independent. Could combine with model narration when available.

### Option D: Hybrid (recommended guess)
- Option C as baseline (always works, zero model dependency)
- Option B for local models (low-effort narration keywords)
- Full SOUL.md narration for cloud models (rich context)
- Option A as safety net (runtime injection if all else fails)

## Test Protocol

1. Start llama-server with each model
2. Run each task 3 times (variance check)
3. Record full session logs (already in ~/.nanobot/sessions/)
4. Score each session on the metrics above
5. Write results in this file under "## Results"

## Results

*(to be filled after testing)*

## Notes

- The display-layer narration (Option C) is probably worth building regardless —
  it's the Layer 2 status bar work we already planned
- If Nanbeige4.1-3B can do even basic narration, that's a win — it's our primary
  local RLM and it handles 500+ tool rounds
- Qwen3-0.6B is the memory model, not expected to use tools — skip if irrelevant
