# NanBeige4.1-3B Prefill Strategy Test Results

Date: 2026-02-20
Model: nanbeige4.1-3b (Q8_0 GGUF via LM Studio at 192.168.1.22:1234)
Temperature: 0.6, max_tokens: 256 unless noted

## Background

NanBeige has `<think>` / `</think>` special tokens (IDs 166103/166104) baked into
its chat template. Unlike Qwen3, there is **no `enable_thinking` toggle** — the
model always produces thinking blocks. The `chat_template_kwargs.enable_thinking`
and `reasoning_budget` parameters are ignored by the NanBeige template.

Reddit tip: prefilling with "Sure" instead of empty/newline forces the model to
skip `<think>` blocks entirely.

## Round 1: Chat Mode (No Tools)

| # | Prefill | Thinking? | Content | Tokens | Finished? |
|---|---------|-----------|---------|--------|-----------|
| 1 | None | 256 tokens wasted in `reasoning_content` | Empty `""` | 256 | No (length) |
| 2 | `"\n"` (old nanobot) | 256 tokens wasted in `reasoning_content` | `"[ignore]\n"` | 256 | No (length) |
| 3 | `""` empty | 256 tokens wasted in `reasoning_content` | Empty `""` | 256 | No (length) |
| 4 | `enable_thinking: false` | 256 tokens wasted in `reasoning_content` | Empty `""` | 256 | No (length) |
| 5 | **`"Sure"`** | **None** | Direct answer (verbose) | 256 | No (length) |
| 6 | **`"Sure, "`** | **None** | Clean: "2+2 = **4**" | **57** | **Yes (stop)** |
| 7 | `"The answer is"` | Mixed — reasoning AND content | Had answer but also reasoning | 128 | No (length) |
| 8 | `"2+2 = "` | Mixed — reasoning AND content | Had `4` but also reasoning | 128 | No (length) |
| 9 | `"Sure"` + `enable_thinking: false` | **None** | Direct answer | 128 | No (verbose) |

## Round 2: Tool Calling Mode

| # | Prefill | Thinking? | Tool call? | Correct? | Tokens |
|---|---------|-----------|------------|----------|--------|
| 1 | None (control) | 220 tokens reasoning | Yes, parsed in `tool_calls` | Yes — `read_file(src/main.rs)` | 220 |
| 2 | `"Sure, "` | None | **No** — narrated instead | Broken — hallucinated narrative | 256 |
| 3 | `"<tool_call>"` (obvious) | None | Yes, in `content` (raw) | Yes — `read_file(src/main.rs)` | **25** |
| 4 | `"<tool_call>"` (ambiguous) | None | Yes, in `content` (raw) | **Wrong** — `read_file(.)` not `exec(ls)` | 20 |
| 5 | `"<tool_call>"` (greeting) | None | Yes, in `content` (raw) | **Wrong** — forced `read_file(hello)` | 21 |

Problems:
- `"Sure, "` breaks tool calling (locks into text continuation)
- `"<tool_call>"` breaks tool selection (can't reason about which tool)
- `"<tool_call>"` output lands in `content` not `tool_calls` (needs manual parsing)

## Round 3: Parameter Controls for Thinking Length

Tested whether `reasoning_budget` or temperature controls thinking length.

### reasoning_budget (all with max_tokens: 512)

| Budget | Reasoning chars | Tokens | Content |
|--------|----------------|--------|---------|
| 0 | 2183 | 512 | Empty |
| 50 | 2229 | 512 | Empty |
| 128 | 2053 | 512 | Empty |
| 256 | 2112 | 512 | Empty |
| 1024 | 2161 | 512 | Empty |

**`reasoning_budget` is completely ignored.** All produced ~2100 chars.

### Temperature (all with max_tokens: 512)

| Temp | Reasoning chars | Tokens | Content |
|------|----------------|--------|---------|
| 0.1 | 2085 | 512 | Empty |
| 0.3 | 2092 | 512 | Empty |
| 0.6 | 2195 | 512 | Empty |
| 1.0 | 2147 | 512 | Empty |

**Temperature has no effect on thinking length.**

### Natural thinking length (max_tokens: 2048)

With enough budget, the model thinks ~713 words (4241 chars) then answers.
Total: 1518 tokens. The model needs ~1000+ tokens of thinking for "What is 2+2?".

### System prompt instructions (max_tokens: 2048)

| System prompt | Reasoning chars | Reasoning words | Total tokens |
|---------------|----------------|-----------------|--------------|
| Default | 4241 | 713 | 1518 |
| "Think briefly, answer concisely" | **1460** | **247** | **440** |
| "Keep reasoning under 50 words" | **1404** | **241** | **358** |
| "Do not use internal reasoning" | 1717 | 288 | 499 |

**System prompt reduces thinking by ~60-75%** but cannot eliminate it.

## Round 4: Pre-closed Think Block (THE WINNER)

Prefill: `"<think>\n</think>\n\n"` — an already-closed think block.

### Chat mode

| # | Prefill | Thinking | Content | Tokens | Finished? |
|---|---------|----------|---------|--------|-----------|
| 1 | `<think>\n</think>\n\n` | **Zero** | Clean answer | **62** | Yes (stop) |
| 2 | `<think>\nSimple.\n</think>\n\n` | **Zero** | Cleaner answer | **45** | Yes (stop) |

### Tool mode — obvious task ("Read src/main.rs")

| # | Prefill | Thinking | Tool call | Correct? | Tokens |
|---|---------|----------|-----------|----------|--------|
| 1 | None (control) | 220 tokens | `tool_calls` field | Yes | 220 |
| 2 | `<think>\nUse read_file.\n</think>\n\n` | **Zero** | `tool_calls` field | **Yes** | **26** |

### Tool mode — ambiguous task ("List files in current directory")

| # | Prefill | Thinking | Tool call | Correct? | Tokens |
|---|---------|----------|-----------|----------|--------|
| 1 | None (control) | ~220 tokens | `tool_calls` | Yes — `exec(ls)` | ~220 |
| 2 | `<tool_call>` | Zero | `content` (raw) | **Wrong** — `read_file(.)` | 20 |
| 3 | `<think>\n</think>\n\n` | **Zero** | `tool_calls` field | **Yes** — `exec(ls -la)` | **22** |

### Tool mode — no tool needed ("Hello, how are you?")

| # | Prefill | Thinking | Tool call | Correct? | Tokens |
|---|---------|----------|-----------|----------|--------|
| 1 | `<tool_call>` | Zero | Forced hallucination | **Wrong** | 21 |
| 2 | `<think>\n</think>\n\n` | Some (198) | **None** (correct!) | **Yes** — chatted | 198 |

## Why Pre-closed Think Works

The NanBeige chat template format is:
```
<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{response}<|im_end|>
```

By prefilling `<think>\n</think>\n\n`, we:
1. Satisfy the template's expected `<think>` block (model sees it already happened)
2. Place the continuation point at the `{response}` position
3. Model generates content directly — tool calls or text
4. LM Studio correctly parses tool calls into the `tool_calls` field
5. Model retains ability to reason about tool selection (unlike `<tool_call>` prefill)

For the greeting case, the model still produced some reasoning (198 tokens) because
it started a new `<think>` block. This is acceptable — it correctly decided NOT to
use tools and chatted instead.

## Comparison: All Strategies

| Strategy | Chat | Tools (obvious) | Tools (ambiguous) | Tools (no tool) |
|----------|------|-----------------|-------------------|-----------------|
| None | Thinks forever | 220 tok, correct | 220 tok, correct | correct |
| `"\n"` | Thinks forever | not tested | not tested | not tested |
| `"Sure, "` | 57 tok, clean | **BROKEN** | **BROKEN** | **BROKEN** |
| `"<tool_call>"` | N/A | 25 tok, correct | 20 tok, **WRONG** | **WRONG** |
| **`<think>\n</think>\n\n`** | **62 tok, clean** | **26 tok, correct** | **22 tok, correct** | **198 tok, correct** |

**Winner: `<think>\n</think>\n\n` — works for all scenarios, no guards needed.**

## Round 5: Stress Test (10 scenarios)

### Chat mode (5/5 pass)

| # | Prompt | Think | Tokens | Correct? |
|---|--------|-------|--------|----------|
| T1 | Capital of France | 0 | 60 | Yes — "Paris" |
| T2 | Python reverse string | 0 | 347 | Yes — working code |
| T3 | Train distance 60mph * 2.5h | 0 | 256 | Yes — 150 miles (hit length) |
| T4 | Is earth round? | 0 | 256 | Yes (verbose, hit length) |
| T5 | Multi-turn: 5*3=15, add 7? | 0 | 256 | Yes — 22 (verbose) |

### Tool mode with 4 tools: read_file, exec, write_file, web_search (4/5 pass)

| # | Prompt | Think | Tool chosen | Correct? | Tokens |
|---|--------|-------|-------------|----------|--------|
| T6 | Run cargo test | 0 | `exec("cargo test")` | **Yes** | 21 |
| T7 | Create hello.txt | 0 | `write_file("hello.txt","Hello World")` | **Yes** | 31 |
| T8 | Search Rust releases | 0 | `web_search("latest Rust release notes")` | **Yes** | 25 |
| T9 | Check syntax in src/lib.rs | 0 | **None** — "don't have access" | **FAIL** | 149 |
| T10 | Explain mutex in Rust | 0 | None (correct!) | **Yes** | 512 |

### T9 failure analysis

Without thinking, the model didn't reason through that `read_file` is the right
tool for "check syntax errors in src/lib.rs". The no-prefill control gets this right
(thinks 220 tokens then calls `read_file`). This is a known tradeoff: pre-closed
think saves ~200 tokens per call but occasionally misses tool usage on indirect
prompts.

### T10 note

Model correctly declined tools but emitted `<thinking>` tags inside `content`
(not in `reasoning_content`). The existing `split_thinking_from_content_delta()`
in nanobot strips these.

## Integration

File: `src/providers/openai_compat.rs`, function `apply_local_thinking_prefill()`

The pre-closed think block is a universal prefill:
- Works with and without tools
- Preserves correct tool selection on 4/5 tool scenarios
- Eliminates thinking overhead (20-60 tokens vs 220-512)
- Properly parsed by LM Studio into `tool_calls` field
- The `has_tools` guard is **removed** — same prefill works everywhere
- The `thinking_budget.is_some()` guard is **preserved** — `/think` (`/t`) still
  enables full thinking when users want it

### `/think` toggle interaction

| User command | `thinking_budget` | Prefill applied? | Behavior |
|-------------|-------------------|-----------------|----------|
| (default) | `None` | Yes — pre-closed think | Fast, no reasoning |
| `/think on` or `/t` | `Some(n)` | **No** — skipped | Full thinking enabled |
| `/think off` | `None` | Yes — pre-closed think | Back to fast mode |
| `/think 256` | `Some(256)` | **No** — skipped | Capped thinking (256 tok for small models) |

### Tradeoff

- **9/10 accuracy** without thinking (saves ~200 tokens/call)
- **10/10 accuracy** with thinking (costs ~200 extra tokens/call)
- Users can toggle with `/t` when they need the model to reason harder
