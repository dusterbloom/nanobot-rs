# CONTEXT.md Protocol — Implementation Plan

## Problems (Verified by Research)

### 1. Duplication Map (token waste per turn)
| Concept | Source 1 | Source 2 | Source 3 | Waste |
|---------|----------|----------|----------|-------|
| "Assumptions forbidden" | AGENTS.md L7 | MEMORY.md L5 | — | exact dup |
| Model hierarchy (zhipu/haiku/sonnet/opus) | AGENTS.md L18-53 | MEMORY.md L2,L3 | — | near dup |
| "Always pass model to spawn" | AGENTS.md L50 | MEMORY.md L3 | — | exact dup |
| Narration rules | SOUL.md L28-66 | AGENTS.md L11 | — | cross-ref |
| Concise/voice behavior | SOUL.md L9-11 | context.rs identity | build_messages voice block | triple |
| Verification protocol | AGENTS.md L55-81 | context.rs provenance | — | near dup |
| Memory architecture | SOUL.md L95-99 | TOOLS.md L87-97 | context.rs identity | triple, TOOLS.md is STALE |
| ZhiPu model notes | AGENTS.md L42-47 | MEMORY.md L102-109 | — | near dup |

### 2. Working Memory = Garbage
Actual content of `SESSION_41308070.md` (the active session):
```
## Compaction Summary (2026-02-16T11:39:46Z)
<thinking>
Okay, let's see. The user wants me to read files directly instead of
using subagents. They mentioned that the summaries are losing verbatim
content, so I need to use the read_file tool...
</thinking>
The conversation history highlights the need to read files directly
instead of using subagents. The user requested file paths but hasn't
provided them. I should prompt them to specify the file locations...
```
- `<thinking>` tags leak through (Qwen3-0.6B doesn't strip them)
- Summaries are self-referential ("I should prompt them")
- No structured extraction of facts/decisions/pending items
- Appended indefinitely — grows every compaction

### 3. Qwen3-0.6B Is Too Small for Summarization
- 0.6B params cannot reliably follow template instructions
- No thinking-tag suppression
- Produces repetitive, noisy, self-referential summaries
- Research shows minimum viable: Qwen3-1.7B (~1.1GB Q4) or Gemma 3 1B (~0.7GB Q4)

## Solution: Three Layers + Better Model

### Layer 1: MEMORY.md (Long-term facts only)
- **What:** Permanent user preferences, project facts, learned rules
- **Updated by:** Reflector only
- **Format:** Flat bullet points, no narrative, no session state
- **Rule:** Strip everything that's already in AGENTS.md/SOUL.md/USER.md

### Layer 2: CONTEXT.md (Structured session state)
- **What:** Current task, decisions, facts found, pending actions
- **Updated by:** Compactor (replaces working memory appending)
- **Format:** Strict template (below) — any SLM can fill it
- **Rule:** Overwritten each compaction. One file, not accumulated sections.

### Layer 3: Session JSONL (Raw history)
- **What:** Every message, tool call, result — verbatim
- **Updated by:** Nanobot core (automatic)
- **Rule:** Never summarized in-place

## CONTEXT.md Template (SLM-friendly)

```markdown
# Context

## Task
[One sentence: what the user is working on right now]

## Decisions
- [Decision 1]
- [Decision 2]

## Facts
- [Important fact discovered]

## Pending
- [ ] [What's still to do]

## Errors
- [Error and resolution]
```

## Compaction Prompt (replaces SUMMARIZE_PROMPT)

```
Extract facts from this conversation into the template below.
Rules:
1. Copy technical terms, file paths, numbers EXACTLY
2. One sentence per bullet, max 10 bullets per section
3. Skip anything you're unsure about
4. No meta-commentary ("I should...", "Let me...")
5. No thinking tags
6. ONLY output the filled template, nothing else

# Context

## Task
[What is the user doing?]

## Decisions
[What was decided?]

## Facts
[What was discovered?]

## Pending
[What's still to do?]

## Errors
[What went wrong?]
```

## Compaction Model Upgrade

### Current: Qwen3-0.6B Q4_K_M (500MB)
- Pros: Tiny, fast
- Cons: Thinking tags leak, can't follow templates, self-referential output

### Recommended: Qwen3-1.7B Q4_K_M (~1.1GB)
- Available: `Qwen/Qwen3-1.7B-GGUF` on HuggingFace
- 29+ languages, strong instruction-following at 1.7B
- GGUF Q4_K_M: ~1.1GB (only 600MB more than current)
- Significant quality jump over 0.6B for structured extraction

### Alternative: Gemma 3 1B Q4 (~0.7GB)
- 140+ languages (best multilingual coverage)
- Only 200MB more than Qwen3-0.6B
- But: less proven for structured extraction than Qwen3

### Decision: Ship Qwen3-1.7B as default, test Gemma 3 1B later

## Deduplication Rules

### System prompt `_get_identity()` keeps:
- Model identity (1 line)
- Current time
- Workspace path
- "Be concise, use tools, never fabricate" (3 behavioral rules)
- Working memory / long-term memory pointers
- Cost delegation hint (for expensive models)

### System prompt REMOVES:
- Memory architecture explanation (TOOLS.md has it)
- Voice mode rules (already injected by build_messages when relevant)

### MEMORY.md REMOVES (already in AGENTS.md):
- "Assumptions are forbidden" (line 5 → AGENTS.md L7)
- "SAVE TOKENS / delegate" (line 2 → AGENTS.md L20-34)
- "ALWAYS pass model to spawn" (line 3 → AGENTS.md L50)
- Anthropic model IDs (line 4 → not needed, agents.md has hierarchy)
- Session history (lines 22-130 → belong in CONTEXT.md or nowhere)

### TOOLS.md FIXES:
- Remove stale "Observations" layer (line 90 — removed from code)
- Remove stale "Semantic retrieval" layer (line 91-92 — not implemented)
- Compress tool examples to one-liners

### AGENTS.md: Remove verification protocol (lines 55-81)
- Already injected by `provenance_enabled` in context.rs
- If provenance is off, these rules shouldn't be there anyway

## Implementation Phases

### Phase 1: Compaction model + prompt (code changes)
**Files:**
- `src/server.rs`: Change COMPACTION_MODEL_URL/FILENAME to Qwen3-1.7B
- `src/agent/compaction.rs`: Replace SUMMARIZE_PROMPT with template prompt
- `src/agent/compaction.rs`: Replace MERGE_SUMMARIES_PROMPT with merge-into-template variant

### Phase 2: CONTEXT.md integration (code changes)
**Files:**
- `src/agent/compaction.rs`: Write output to `{workspace}/CONTEXT.md` (overwrite, not append)
- `src/agent/context.rs`: Load CONTEXT.md as its own system prompt section
- `src/agent/working_memory.rs`: Stop accumulating compaction summaries (optional: keep for backup)
- `src/agent/agent_loop.rs`: Wire new compaction output path

### Phase 3: Deduplicate workspace files (file edits)
**Files:**
- `~/.nanobot/workspace/MEMORY.md`: Strip duplicates and session history
- `~/.nanobot/workspace/TOOLS.md`: Fix stale memory description, compress examples
- `~/.nanobot/workspace/AGENTS.md`: Remove verification protocol section

### Phase 4: System prompt cleanup (code changes)
**Files:**
- `src/agent/context.rs`: Remove duplicated info from `_get_identity()`
- Verify no truncation markers in any path

## Token Budget (After)

| Component | Before | After |
|-----------|--------|-------|
| _get_identity() | ~120 tok | ~100 tok |
| AGENTS.md (bootstrap) | ~510 tok | ~350 tok (no verification dup) |
| SOUL.md (bootstrap) | ~625 tok | ~625 tok (unchanged) |
| USER.md (bootstrap) | ~275 tok | ~275 tok (unchanged) |
| TOOLS.md (bootstrap) | ~625 tok | ~350 tok (compressed) |
| MEMORY.md (long-term) | ~400 tok (tail) | ~200 tok (facts only) |
| Working memory (session) | ~600 tok (garbage) | ~200 tok (CONTEXT.md template) |
| **Total overhead** | **~3155 tok** | **~2100 tok** |
| **Savings** | | **~1000 tok/turn + quality** |

The real win isn't just tokens — it's that the model sees CLEAN, STRUCTURED session state instead of leaked thinking tags and self-referential noise.

## Verification

1. `cargo test` — all existing tests pass
2. `cargo build` — compiles with new model URL
3. Manual: trigger compaction, check CONTEXT.md output is clean template
4. Manual: verify MEMORY.md has no session history
5. Manual: verify system prompt has no duplication
6. Test at 16K/64K/1M context: all components fit cleanly
