# Local Model Architecture: First-Principles Redesign

**Author:** Architectural analysis, Feb 2026
**Status:** Proposal — not yet implemented
**Scope:** `is_local = true` paths only. Cloud mode is left untouched.

---

## 1. The Core Problem

The current local mode has a fundamental category error: it takes a cloud architecture and tries to bolt on compression. That is the wrong direction. The right direction is to start from what small models *actually are* and build from there.

### What small models (1–7B) actually are

- Attention window is physically short (4K–32K tokens typically in practice, even if the rope limit is higher)
- Attention *quality* degrades rapidly with distance — primacy and recency effects dominate; the middle is largely lost
- They cannot reliably perform metacognitive operations ("I notice my context is compressed, I should call `lcm_expand`")
- They are brittle under role ambiguity: mixing persona text, tool schema, memory summaries, and history in one block causes incoherence
- Tool results are ephemeral. A small model cannot reliably extract signal from a tool result 20 turns back. It is noise, not signal, and should be treated as such

### What the current system does wrong

**Double compaction conflict.** `trim_to_fit_with_age` at line 830 of `agent_loop.rs` fires *before* LCM ingest at line 854. Messages that get mechanically dropped by the token trimmer never enter the LCM store. The lossless guarantee is silently broken on every small-window model. These two systems are fighting each other.

**Wrong model for summaries.** `summarize_for_lcm()` in `compaction.rs` uses `SUMMARIZE_PROMPT` (standard 8–70B prompt) regardless of reader capability. `SUMMARIZE_PROMPT_MINIMAL` exists and is perfect for 3B target models, but `select_summarize_prompt()` is never called from the LCM path. The compaction model produces a summary optimised for a large reader and gives it to a small one.

**Summaries injected in the wrong place.** LCM injects summaries as `role: "user"` messages in the middle of conversation history. For a small model, the middle is where attention goes to die. The summary needs to be at the *start*, not the middle.

**LCM DAG is ephemeral.** The DAG lives only in `LcmEngine`'s in-memory `HashMap`. On process restart or session reload, the whole DAG is lost. `lcm_expand` calls after a restart silently fail or return nothing.

**τ_soft at 0.5 is wrong.** At 4K context, 50% threshold = 2K tokens, which is roughly 13 messages. The trigger fires before meaningful history builds up. At 128K context, 50% is 64K tokens and the trigger almost never fires. The threshold needs to be context-window-relative.

**The router is context-blind.** `build_conversation_tail` filters out *all* tool messages. On a code-heavy task, the router sees a conversation with no tool calls — it literally cannot know the model is in the middle of a multi-step tool loop.

**Trio main model has no tools but does have a 2000-token system prompt.** For a 1–3B router that receives an 800-char snippet, the instruction overhead dominates. The router needs a sub-200-token operational prompt, not a full personality prompt.

---

## 2. First-Principles Architecture for Local Models

The mental model is simple: **a small local model is a reactive function, not a cognitive agent**. Treat it like one.

```
┌─────────────────────────────────────────────────────────┐
│  SYSTEM[0]  (FIXED — never changes within a session)    │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ IDENTITY (200 tokens max — who you are + one rule)  │ │
│  ├─────────────────────────────────────────────────────┤ │
│  │ SESSION STATE (extracted each turn, ~300 tokens)    │ │
│  │  task: <current goal>                               │ │
│  │  done: [step1, step2]                               │ │
│  │  pending: [step3]                                   │ │
│  │  facts: [key=val, ...]                              │ │
│  │  last_tool: <name> → <digest 50 chars>              │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│  ROLLING WINDOW  (last N raw turns, evict oldest first) │
│  N = floor(0.4 × max_context_tokens / avg_turn_tokens)  │
│  Min 3 turns, Max 10 turns                              │
│                                                         │
│  Tool results: kept only in the turn they were issued.  │
│  On eviction: tool result → digest(50 chars) in state.  │
└─────────────────────────────────────────────────────────┘
```

The key insight: **state lives in the system prompt, not in history**. History is just the last few raw exchanges so the model knows what was just said. Everything else goes into a structured state block that is rewritten each turn by a dedicated extraction pass.

---

## 3. Concrete Changes Required

### 3.1 Kill the double compaction conflict

**File:** `src/agent/agent_loop.rs`, pre-call step (around line 830)

**Current order:**
1. `hygiene_pipeline`
2. `trim_to_fit_with_age`  ← drops messages silently
3. LCM ingest  ← never sees the dropped messages

**Required order:**
1. `hygiene_pipeline`
2. LCM ingest (feed all messages into immutable store FIRST)
3. LCM compact if needed (reduces active window)
4. `trim_to_fit_with_age` as hard safety net ONLY — it must never fire before LCM has had a chance to ingest

The trim function should operate on the LCM active context, not the raw message list. In practice this means: if `lcm.enabled`, skip `trim_to_fit_with_age` entirely and let LCM control the window. Add a hard-cap override only if LCM produces something that still exceeds `max_context_tokens × 0.95`.

```rust
// Proposed ordering in agent_loop pre_call():
if self.lcm_config.enabled {
    // 1. Ingest first (idempotent, never drops)
    lcm_engine.lock().await.ingest_batch(&ctx.messages);
    // 2. Compact if needed (reduces active window)
    lcm_engine.lock().await.maybe_compact().await;
    // 3. Sync active context back (what LCM decided to keep)
    ctx.messages = lcm_engine.lock().await.active_messages();
    // 4. Hard safety net ONLY — should almost never trigger
    if TokenBudget::estimate_tokens(&ctx.messages) > hard_cap {
        ctx.messages = ctx.core.token_budget.trim_to_fit(&ctx.messages, tool_def_tokens);
    }
} else {
    // Non-LCM path: trim as before
    ctx.messages = ctx.core.token_budget.trim_to_fit_with_age(...);
}
```

### 3.2 Fix the SUMMARIZE_PROMPT_MINIMAL wiring

**File:** `src/agent/compaction.rs`, `summarize_for_lcm()`

**Current (broken):**
```rust
let prompt = match mode {
    "preserve_details" => SUMMARIZE_PROMPT,   // always Standard
    "bullet_points"    => SUMMARIZE_PROMPT_ADVANCED,
    _                  => SUMMARIZE_PROMPT,
};
```

**Required:** Pass `ReaderCapability` through to `summarize_for_lcm`. The reader is the **main model** (the one that will consume the summary), not the compaction model.

```rust
pub async fn summarize_for_lcm(
    messages: &[Value],
    mode: &str,
    reader: ReaderCapability,  // ADD THIS
) -> Result<String> {
    let prompt = match (mode, reader) {
        (_, ReaderCapability::Minimal) => SUMMARIZE_PROMPT_MINIMAL,
        ("bullet_points", _)           => SUMMARIZE_PROMPT_ADVANCED,
        _                              => SUMMARIZE_PROMPT,
    };
    // ... rest unchanged
}
```

`ReaderCapability` should be derived from `model_capabilities` at call site:
- context window ≤ 8K → `Minimal`
- context window 8K–64K → `Standard`
- context window > 64K → `Advanced`

### 3.3 Fix summary injection role and position

**File:** `src/agent/lcm.rs`, `compact()` method

**Current (broken):**
```rust
json!({
    "role": "user",
    "content": "[Summary of messages ...]"
})
```

**Problems:** `role: "user"` is semantically wrong — the user did not write this. More importantly, it's injected in the middle of history where small models lose it.

**Required:** Summaries should never live in the rolling conversation window. They should be folded into the `SESSION STATE` block in the system prompt. The LCM engine's `compact()` should:

1. Summarise the oldest raw block (same as now)
2. **Call `extract_session_state(summary_text)` to update the structured state**
3. Remove those messages from the active context window
4. Let the state block in `system[0]` carry the information forward

This means LCM changes from a "compress into middle" strategy to a "extract forward into system" strategy. The rolling window shrinks; the system prompt's state block grows (with a hard cap at ~400 tokens for state).

```rust
// In LcmEngine::compact():
let summary_text = self.escalated_summary(&block).await?;
// Extract structured state delta from summary
let state_delta = self.extract_state_delta(&summary_text).await;
// Merge into session state (always at system[0])
self.session_state.merge(state_delta);
// Remove the compacted block from active[]
self.active.retain(|e| !compacted_ids.contains(&e.msg_id()));
// Rebuild system[0] with updated state
self.rebuild_system_message();
```

### 3.4 Persist the LCM DAG to disk

**File:** `src/agent/lcm.rs`, `LcmEngine`

**Current:** The DAG (`store: HashMap<MsgId, LcmMessage>`, `nodes: Vec<SummaryNode>`) is in-memory only.

**Required:** Persist to `{workspace}/memory/sessions/LCM_{session_hash}.json` after every compact operation. Load on engine creation if file exists.

```rust
impl LcmEngine {
    pub fn persist(&self, path: &Path) -> Result<()> {
        let data = serde_json::to_string(&LcmSnapshot {
            store: &self.store,
            nodes: &self.nodes,
            session_state: &self.session_state,
        })?;
        std::fs::write(path, data)?;
        Ok(())
    }

    pub fn load(path: &Path, config: LcmConfig) -> Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let snapshot: LcmSnapshot = serde_json::from_str(&data)?;
        Ok(Self {
            store: snapshot.store,
            nodes: snapshot.nodes,
            session_state: snapshot.session_state,
            config,
            ..Self::default()
        })
    }
}
```

This also means `lcm_expand` calls after a restart actually work.

### 3.5 Make τ_soft context-window-relative

**File:** `src/config/schema.rs` / `src/agent/lcm.rs`

**Current:** `tau_soft = 0.5` fixed.

**Required:** τ_soft should be the fill fraction at which the *rolling window* exceeds `max_window_turns`. The rolling window size should be derived from context window, not a fixed fraction of it.

```rust
pub fn tau_soft_for_context(max_context_tokens: usize) -> f32 {
    // Target: keep rolling window at ~6 turns for small models, ~15 for large
    let target_turns = if max_context_tokens <= 8_000 { 4 }
                       else if max_context_tokens <= 32_000 { 8 }
                       else { 15 };
    // Trigger compaction when active window would need to exceed target
    // Expressed as fraction of max_context: assume 200 tokens/turn avg
    let target_tokens = target_turns * 200;
    (target_tokens as f32 / max_context_tokens as f32).clamp(0.15, 0.70)
}
```

For a 4K model this gives τ_soft ≈ 0.20 (trigger at 800 tokens = 4 turns — aggressive, intentional).
For a 32K model this gives τ_soft ≈ 0.05 (trigger at ~1600 tokens — wait for a reasonable window to build up first, then compact).

Wait — actually for 32K: `8 * 200 / 32000 = 0.05`. That's too early. Let me reconsider:

The τ_soft formula needs to be: "trigger compaction when active context reaches X% of window, where X is calibrated so that after compaction the remaining active window is small enough for the model to handle effectively."

Better formulation:
- Minimum safe rolling window: `max_window = min(max_context * 0.6, target_turns * avg_turn_tokens)`
- τ_soft = `max_window / max_context`
- For 4K, 4 turns × 200 = 800 tokens → τ_soft = 0.20
- For 16K, 6 turns × 200 = 1200 tokens → τ_soft = 0.075 (too aggressive)
- Better: scale τ_soft between 0.30 and 0.60 based on context size

Simplest correct formula:
```rust
pub fn tau_soft_for_context(max_context_tokens: usize) -> f32 {
    // Keep 30-40% of context for rolling history regardless of window size.
    // Smaller models get lower threshold (trigger sooner) because quality
    // degrades faster with context fill.
    match max_context_tokens {
        0..=8_192   => 0.30,   // 4K: trigger at ~1200 tokens ≈ 6 turns
        8_193..=32_768  => 0.40,   // 16K: trigger at ~6500 tokens ≈ 32 turns
        32_769..=131_072 => 0.50,  // 64K: default behaviour
        _           => 0.60,   // 128K+: give it more room
    }
}
```

### 3.6 Fix the trio router context — include tool digests

**File:** `src/agent/router.rs`, `build_conversation_tail()`

**Current:** Filters out `system` and `tool` messages entirely. Router is blind to tool execution history.

**Required:** Include a compact representation of recent tool calls and their outcomes.

```rust
pub fn build_conversation_tail(
    messages: &[Value],
    max_pairs: usize,
    max_msg_chars: usize,
    max_chars: usize,
) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut chars_used = 0;

    for msg in messages.iter().rev() {
        let role = msg["role"].as_str().unwrap_or("");
        match role {
            "system" => continue,
            "tool" => {
                // Include a digest: "[tool: <name> → <first 60 chars of result>]"
                let name = msg.get("name").and_then(|n| n.as_str()).unwrap_or("?");
                let content = msg["content"].as_str().unwrap_or("");
                let digest = &content[..content.len().min(60)];
                let line = format!("[tool:{} → {}]", name, digest);
                if chars_used + line.len() < max_chars {
                    parts.push(line);
                    chars_used += line.len();
                }
            }
            "assistant" | "user" => {
                let content = msg["content"].as_str().unwrap_or("");
                let truncated = &content[..content.len().min(max_msg_chars)];
                let line = format!("[{}] {}", role, truncated);
                if chars_used + line.len() < max_chars {
                    parts.push(line);
                    chars_used += line.len();
                } else {
                    break;
                }
            }
            _ => {}
        }
        if parts.len() >= max_pairs * 3 { break; }
    }

    parts.reverse();
    parts.join("\n")
}
```

### 3.7 Reduce the trio router system prompt

**File:** `src/agent/role_policy.rs`

**Current:** Router receives the full nanobot identity prompt + conversation tail. This is likely 1000+ tokens for a message that needs to produce a 20-token JSON decision.

**Required:** Router system prompt should be ~150 tokens maximum:

```
ROLE=ROUTER. Output ONLY JSON: {"action":"tool|specialist|respond","target":"...","confidence":0.0}

Actions: tool=execute a tool directly, specialist=delegate to specialist model, respond=answer inline.
Current task context:
{tail}
```

No personality. No memory. No skills. Just the decision schema and the context tail.

This is especially critical for Qwen3-0.6B or similar tiny router models. A 1000-token instruction prompt leaves 3000 tokens for context and response in a 4K model — that is not enough.

### 3.8 Session state extraction — the new "working memory" for local models

This is the key new concept. After each assistant turn that produces a tool call or a substantive response, run a lightweight extraction pass using the specialist/compaction model:

```rust
/// Extract structured session state from the latest exchange.
/// Called after each turn when is_local=true.
/// Output is a small YAML block that replaces the state section in system[0].
async fn extract_session_state(
    recent_turns: &[Value],
    compaction_provider: &Arc<dyn LLMProvider>,
    model: &str,
) -> String {
    let prompt = r#"Extract the current task state from this conversation.
Output ONLY this YAML block, no other text:

task: <one sentence: what is being worked on>
done: [<completed step>, ...]  # max 5 items
pending: [<next step>, ...]    # max 3 items
facts: [<key=value>, ...]      # max 5 critical facts
last_tool: <tool_name> → <result digest 50 chars>
"#;
    // Call compaction/specialist model with recent_turns + prompt
    // Returns the YAML block
    // Hard cap at 400 tokens — truncate if needed
}
```

This extraction runs async after each turn (not blocking the response). The result is written to `session_state` in the LCM engine, which then rebuilds `system[0]` with the updated state block prepended.

The state block always occupies the *first 400 tokens* of the system prompt — highest attention position. Older facts get evicted when the block fills (LRU eviction on `facts[]`).

---

## 4. The Revised Local Turn Sequence

```
Turn N begins:
│
├─ 1. hygiene_pipeline (dedup, orphan cleanup) — unchanged
│
├─ 2. LCM ingest all messages into immutable store (if lcm.enabled)
│
├─ 3. LCM threshold check → compact if needed
│      - summarize oldest block using SUMMARIZE_PROMPT_MINIMAL
│      - extract state delta → merge into session_state
│      - remove compacted messages from active window
│      - rebuild system[0] with updated state block
│
├─ 4. hard safety trim (last resort, should rarely fire)
│
├─ 5. trio router preflight (if trio mode)
│      - receives: 150-token system + tail (user/assistant + tool digests)
│      - returns: action JSON
│
├─ 6. LLM call (main model)
│      - sees: system[IDENTITY + SESSION_STATE] + rolling_window[3-8 turns]
│      - no LCM summaries injected mid-history
│
├─ 7. tool execution / specialist dispatch
│
└─ 8. async state extraction (non-blocking)
       - extract_session_state(recent_turns) → update session_state
       - persist LCM DAG to disk
```

---

## 5. What to Keep vs What to Throw Away

**Keep:**
- `LcmEngine` struct and DAG concept — it is the right data structure for the immutable store
- `LcmExpandTool` — useful when the main model explicitly needs to retrieve a past turn
- `hygiene_pipeline` — good, necessary, keep as step 1
- `SpecialistMemory` ring buffer — good concept, increase to 5 entries × 400 chars
- `apply_compaction_result` watermark pattern — correct, keep
- `deterministic_truncate` — the Level 3 fallback, keep
- `circuit_breaker` on router — correct, keep

**Fix (not throw away):**
- Ordering: LCM ingest before trim (see §3.1)
- `SUMMARIZE_PROMPT_MINIMAL` wiring (see §3.2)
- Summary injection: fold into state block, not mid-history `user` message (see §3.3)
- LCM persistence (see §3.4)
- τ_soft formula (see §3.5)
- Router context to include tool digests (see §3.6)
- Router system prompt size (see §3.7)

**Add:**
- `extract_session_state()` async extraction after each turn (see §3.8)
- `SessionState` struct with merge + LRU eviction
- `rebuild_system_message()` on LCM engine that recomposes `system[0]`
- `tau_soft_for_context()` config helper

**Remove / disable by default:**
- `trim_to_fit_with_age` as primary context management when LCM is enabled — it fights LCM
- `SUMMARIZE_PROMPT_ADVANCED` from the local compaction path (it's for cloud readers)
- The `_turn`-tagged age eviction in trim, when LCM is handling eviction

---

## 6. Config Implications

A minimal `~/.nanobot/config.json` for local single mode with these fixes:

```json
{
  "model": "qwen3-4b",
  "lm_studio_api_key": "lm-studio",
  "memory": {
    "enabled": true,
    "compaction_threshold_percent": 30
  },
  "lcm": {
    "enabled": true,
    "tau_soft": 0.0,
    "tau_hard": 0.85,
    "deterministic_target": 512,
    "compaction_endpoint": ""
  }
}
```

Setting `tau_soft: 0.0` means "use the auto-derived value from `tau_soft_for_context()`" — a zero is the sentinel for "calculate from window size". The current code treats 0.0 as "never trigger", which is wrong; it should be treated as "use default for this window size".

---

## 7. Priority Order for Implementation

These are ordered by impact-to-effort ratio:

1. **§3.1 — Fix ordering** (1–2 hours, `agent_loop.rs`). Highest impact. Without this, LCM is broken for all small-window models.

2. **§3.2 — Wire SUMMARIZE_PROMPT_MINIMAL** (30 min, `compaction.rs`). Already written, just not connected.

3. **§3.5 — τ_soft auto-derivation** (1 hour, `schema.rs` + `lcm.rs`). Fixes the "never triggers on 128K, fires immediately on 4K" problem.

4. **§3.6 — Router context with tool digests** (1 hour, `router.rs`). Fixes the router blindness problem.

5. **§3.7 — Reduce router system prompt** (1 hour, `role_policy.rs`). Critical for sub-3B router models.

6. **§3.4 — Persist LCM DAG** (2–3 hours, `lcm.rs`). Correctness fix — `lcm_expand` is currently useless after restart.

7. **§3.3 — Summary into state block not mid-history** (3–4 hours). Requires new `SessionState` struct and `rebuild_system_message`. Biggest architectural change but highest ROI for small model coherence.

8. **§3.8 — Async state extraction** (4–6 hours). Depends on §3.3. This is the "new working memory" concept and requires new code paths.

---

## 8. One-Line Summary

The current local mode is a cloud architecture with compression bolted on. The correct architecture for local models is: **fixed structured state in system prompt, short rolling window, tools as ephemeral, compaction extracts forward into state rather than compressing backward into history.**
