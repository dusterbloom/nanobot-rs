# NanBeige4.1-3B vs nvidia_orchestrator-8b: Router Comparison

Date: 2026-02-20
LM Studio: 192.168.1.22:1234

## Setup

Both models tested with the exact router protocol from nanobot's
`request_strict_router_decision()`:
- System prompt: routing agent instructions with 4 actions
- Tool definition: `route_decision(action, target, args, confidence)`
- User content: real `router_pack` format with available tools listed
- Temperature: 0.2, max_tokens: 256

Available tools provided: read_file, write_file, list_files, exec,
web_search, web_browse, send_message, spawn, cron_schedule, cron_list, cron_delete

NanBeige uses pre-closed think block prefill (`<think>\n</think>\n\n`).
Nemotron uses `/no_think` prefix.

## Head-to-Head Results (10 test cases)

| # | Prompt | Expected | Nemotron-8B | NanBeige-3B |
|---|--------|----------|-------------|-------------|
| T1 | "Hello, how are you today?" | respond | OK — respond/none | OK — respond/User |
| T2 | "Read the file src/main.rs" | tool:read_file | OK — tool/read_file | **WRONG** — respond/router |
| T3 | "Run cargo test" | tool:exec | OK — tool/exec | **WRONG** — respond/router |
| T4 | "Write comprehensive analysis of Rust ownership" | specialist | OK — specialist | **WRONG** — respond/assistant |
| T5 | "What is 2+2?" | respond | OK — respond/direct_answer | OK — respond |
| T6 | "Search web for latest Rust release" | tool:web_search | OK — tool/web_search | OK — tool/web_search |
| T7 | "Refactor auth to JWT" | specialist | OK — specialist | **WRONG** — respond/router |
| T8 | "Tell me a joke" | respond | OK — respond/joke | OK — respond/joke |
| T9 | "Create notes.txt with shopping list" | tool:write_file | OK — tool/write_file | OK — tool/write_file |
| T10 | "hmm I'm not sure what to do" | ask_user | OK — ask_user/user | OK — ask_user/User |

### Scores

|                | Accuracy | Tokens | Avg Latency |
|----------------|----------|--------|-------------|
| Nemotron-8B    | **10/10** | 578 | 571ms |
| NanBeige-3B    | 6/10 | 448 | 338ms |

## NanBeige Failure Pattern

NanBeige without thinking defaults to `respond` for any task requiring inference
about which tool to use:
- "Read the file src/main.rs" → should map to `read_file` tool, NanBeige says respond
- "Run cargo test" → should map to `exec` tool, NanBeige says respond
- Both specialist tasks → NanBeige says respond

The model can handle explicit tool matches (T6: "search the web" → web_search,
T9: "create a file" → write_file) but fails on tasks requiring reasoning about
tool mapping.

## NanBeige With Thinking Enabled

Tested the 4 failing cases with full thinking (no prefill, max_tokens: 1024):

| # | Prompt | No-Think | With-Think | Think Cost |
|---|--------|----------|------------|------------|
| T2 | Read src/main.rs | WRONG (respond) | **OK** (tool/read_file) | 2048ch, 577tok, 3.5s |
| T3 | Run cargo test | WRONG (respond) | WRONG (ask_user) | 3518ch, 852tok, 5.2s |
| T4 | Analysis of ownership | WRONG (respond) | WRONG (hit token limit) | 4823ch, 1024tok, 10.1s |
| T7 | Refactor to JWT | WRONG (respond) | WRONG (ask_user) | 3852ch, 970tok, 6.0s |

**Thinking only fixes 1/4 failing cases** and costs 577-1024 tokens (vs 40 tokens
without thinking). Even with thinking, NanBeige gets only 7/10 on routing.

### Ironic detail

On T7 without thinking, NanBeige narrates in the target field: "This is a complex
multi-step reasoning task... should delegate to specialist" — but still routes to
`respond`. The model KNOWS the right answer but can't express it through the
route_decision tool call without thinking.

## Why Nemotron Wins

nvidia_orchestrator-8b is **purpose-built** for routing/orchestration:

1. **Direct action mapping**: Calls `tool(target="read_file")` and
   `specialist(reason="...")` as function names instead of using `route_decision`.
   This is its native format — the model was trained for orchestration.

2. **Tool reasoning**: Correctly maps "Run cargo test" → `exec` and
   "Read the file" → `read_file` without needing to think.

3. **Specialist detection**: Correctly identifies "write a comprehensive analysis"
   and "refactor auth module" as specialist tasks.

4. **Consistent output**: 10/10 across all test categories. No category failures.

NanBeige is a **general-purpose** 3B chat model. It handles simple routing
(greetings, explicit tool mentions, obvious ask_user) but lacks the trained
orchestration capability for ambiguous or indirect tool selection.

## Conclusion

**Keep nvidia_orchestrator-8b as the router.** It is 10/10 accurate and purpose-built
for the job. NanBeige should remain the main/RLM model where the pre-closed think
block strategy (from nanbeige-prefill-tests.md) gives it 9/10 accuracy at low
token cost.

### Role assignment (unchanged)

| Role | Model | Why |
|------|-------|-----|
| Router | nvidia_orchestrator-8b | 10/10 routing accuracy, purpose-built |
| Main (RLM) | nanbeige4.1-3b | Fast chat + tool calling with pre-closed think |
| Specialist | ministral-3-8b | Complex multi-step reasoning |
