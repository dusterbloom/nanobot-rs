# Free Inference Orchestration Plan

## Goal
Minimize expensive Claude Opus tokens by using me (Opus) ONLY as an orchestrator/director, delegating all actual work to free or cheap inference providers.

## Current Assets

### 🆓 Confirmed Free Providers

| Provider | Models Available | Best For | Rate Limits |
|----------|-----------------|----------|-------------|
| **Groq** | llama-3.3-70b-versatile, llama-3.1-8b-instant, qwen3-32b, llama-4-maverick-17b, llama-4-scout-17b, kimi-k2-instruct, gpt-oss-120b, gpt-oss-20b | Fast inference, tool calling, reasoning | Free tier (rate limited per minute) |
| **Gemini** | gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash, gemini-3-pro-preview, gemini-3-flash-preview | Long context, multimodal, reasoning | Free tier (generous for flash models) |
| **Claude Code** | claude-opus-4 (via Max plan) | Complex coding, architecture, deep reasoning | Unlimited via Max subscription |
| **Local (RTX 3090)** | Nemotron-12B, Ministral-3B, Qwen3-Coder-30B-A3B | Offline, private, no rate limits | Hardware limited only |

### 💰 Paid (Avoid Unless Necessary)

| Provider | Status | Notes |
|----------|--------|-------|
| **OpenAI** | Has API key but costs money | Only use if free models can't handle it |
| **Anthropic API** | Has key but costs money | NEVER use — use Claude Code instead |
| **OpenRouter** | No free models available | DROPPED — not giving free inference |

## Architecture: Opus as Director

```
User Request
    │
    ▼
┌─────────────────────┐
│  Claude Opus (me)    │  ← DIRECTOR ONLY
│  - Parse intent      │  ← Minimal token use
│  - Pick best agent   │  ← Route to free provider
│  - Validate result   │
└─────────┬───────────┘
          │
    ┌─────┴─────┬──────────┬──────────┐
    ▼           ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
│ Groq   │ │ Gemini │ │ Claude │ │ Local    │
│ Free   │ │ Free   │ │ Code   │ │ Models   │
│        │ │        │ │ (Max)  │ │          │
└────────┘ └────────┘ └────────┘ └──────────┘
```

## Routing Rules

### Task → Provider Mapping

| Task Type | Primary | Fallback | Why |
|-----------|---------|----------|-----|
| **Quick Q&A, summaries** | Groq (llama-3.1-8b) | Gemini flash | Speed |
| **Reasoning, analysis** | Groq (qwen3-32b or llama-3.3-70b) | Gemini 2.5-pro | Quality |
| **Tool calling** | Groq (llama-4-scout/maverick) | Local (Nemotron) | Tool format support |
| **Complex coding** | Claude Code (cli) | Groq (gpt-oss-120b) | Max plan = free |
| **Long context (>32K)** | Gemini 2.5-flash (1M ctx) | Gemini 2.5-pro | Only Gemini does 1M |
| **Code review/debug** | Claude Code | Local (Qwen-Coder) | Quality |
| **Summarization/compaction** | Groq (llama-3.1-8b) | Local (Ministral-3B) | Fast & cheap |
| **Memory/reflection** | Gemini flash | Local (Ministral-3B) | Bulk processing |
| **Web research** | Gemini (grounding) | Groq + manual fetch | Gemini has search |
| **Private/sensitive** | Local models ONLY | Never cloud | Privacy |

## Implementation Steps

### Phase 1: Claude Code as Tool (IMMEDIATE)
1. Create a `claude_code` tool that shells out to `claude` CLI
2. Usage: `claude -p "prompt" --output-format json`
3. This gives us Opus-level coding for FREE via Max plan
4. No API tokens burned — it's included in subscription

### Phase 2: Groq Integration (IMMEDIATE)  
1. Groq already has API key configured
2. Add Groq as a provider in nanobot's routing
3. Primary workhorse for most tasks
4. Best models: llama-3.3-70b (reasoning), llama-3.1-8b (speed), qwen3-32b (balanced)

### Phase 3: Gemini Integration (IMMEDIATE)
1. Gemini API key already configured
2. Use for long-context tasks and multimodal
3. Free tier is generous for flash models
4. Best models: gemini-2.5-flash (workhorse), gemini-2.5-pro (quality)

### Phase 4: Smart Routing (NEXT)
1. Implement task classifier (can run on Groq llama-3.1-8b)
2. Auto-route based on task type, context length, and provider health
3. Fallback chains: if Groq rate-limited → Gemini → Local

### Phase 5: Opus Token Budget (NEXT)
1. Set max Opus tokens per conversation (e.g., 2000 for routing)
2. All heavy lifting delegated to free providers
3. Opus only: parses intent, picks agent, validates output
4. Target: 90%+ of tokens on free inference

## Config Changes Needed

### Remove/Deprecate
- OpenRouter provider (no free models)
- Anthropic API direct calls (use Claude Code instead)

### Add
- Claude Code tool (shell out to `claude` CLI)
- Groq provider routing (OpenAI-compatible API)
- Gemini provider routing (Google AI API)
- Task classifier for auto-routing

### Update
- Default model: keep Opus for orchestration but with strict token budget
- Tool delegation model: switch from Opus to Groq llama-3.1-8b
- Memory/compaction model: switch to Groq llama-3.1-8b or local Ministral

## Token Economics

| Role | Current | Proposed | Savings |
|------|---------|----------|---------|
| Main reasoning | Opus ($15/M in) | Opus (director only) | ~80% reduction |
| Tool delegation | Opus | Groq llama-3.1-8b (FREE) | 100% savings |
| Summarization | Opus | Groq llama-3.1-8b (FREE) | 100% savings |
| Coding | Opus | Claude Code CLI (FREE via Max) | 100% savings |
| Long context | Opus | Gemini flash (FREE) | 100% savings |
| Memory ops | Opus | Local Ministral-3B (FREE) | 100% savings |

## Priority Order
1. **Claude Code as tool** — biggest win, zero cost coding
2. **Groq for delegation/summarization** — already have key, instant
3. **Gemini for long context** — already have key, instant  
4. **Smart routing** — needs code changes in nanobot
5. **Local models** — already configured, need to spin up llama.cpp servers
