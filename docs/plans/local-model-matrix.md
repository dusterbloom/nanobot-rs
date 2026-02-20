# Local Model Matrix — Hardware-Optimized Combos

> Last updated: 2026-02-14
> Based on HuggingFace benchmarks, verified GGUF sizes

## Architecture: 3 Roles, 3 Models

Every nanobot local deployment runs up to 3 model slots:

| Slot | Role | Needs | Frequency |
|------|------|-------|-----------|
| **Main** | Conversation, reasoning, code gen | Best general intelligence that fits | Every user message |
| **RLM** | Tool delegation, multi-step agent loops | Tool calling, structured output, sustained chains | Every tool call |
| **Memory** | Summarization, fact extraction, compaction | Text compression, template following | Background, periodic |

On constrained hardware, slots can share a single model.

---

## The Models

### RLM: Nanbeige4.1-3B (THE choice)

**Why it wins — benchmarks vs everything else at 3B-32B:**

| Benchmark | Nanbeige4.1-3B | Qwen3-4B | Qwen3-8B | Qwen3-14B | Qwen3-32B | Qwen3-30B-A3B |
|-----------|---------------|----------|----------|-----------|-----------|---------------|
| BFCL-V4 (tool use) | **56.50** | 44.87 | 42.20 | 45.14 | 47.90 | 48.6 |
| Tau2-Bench (agent) | **48.57** | 45.9 | 42.06 | 44.96 | 45.26 | 47.70 |
| xBench-DeepSearch | **75** | 34 | 31 | 34 | 39 | 25 |
| GAIA (multi-step) | **69.90** | 28.33 | 19.53 | 30.23 | 30.17 | 31.63 |
| Browse-Comp | **19.12** | 1.57 | 0.79 | 2.36 | 3.15 | 1.57 |
| AIME 2026 I | **87.40** | 81.46 | 70.42 | 76.46 | 75.83 | 87.30 |
| GPQA | **83.8** | 65.8 | 62.0 | 63.38 | 68.4 | 73.4 |
| Arena-Hard-v2 | **73.2** | 34.9 | 26.3 | 36.9 | 56.0 | 60.2 |

Key facts:
- **First general small model** to natively support deep-search tasks
- Sustains **500+ rounds of tool invocations** reliably
- Native `<tool_call>` / `<tool_response>` XML format in chat template
- LLaMA architecture (broad llama.cpp compatibility)
- Bilingual EN/ZH, Apache 2.0 license
- Q8_0 GGUF: ~3.5GB, Q4: ~2GB

### Main: Qwen3-30B-A3B-Instruct-2507

**Why this version (2507, not original):**

| Benchmark | Original 30B-A3B | **2507 version** | vs DeepSeek-V3 | vs GPT-4o |
|-----------|------------------|-------------------|----------------|-----------|
| AIME25 | 21.6 | **61.3** | 46.6 | 26.7 |
| Arena-Hard v2 | 24.8 | **69.0** | 45.6 | 61.9 |
| BFCL-v3 (tools) | 58.6 | **65.1** | 64.7 | 66.5 |
| ZebraLogic | 33.2 | **90.0** | 83.4 | 52.6 |
| LiveCodeBench v6 | 29.0 | **43.2** | 45.2 | 35.8 |
| IFEval | 83.7 | **84.7** | 82.3 | 83.9 |
| Creative Writing v3 | 68.1 | **86.0** | 81.6 | 84.9 |
| WritingBench | 72.2 | **85.5** | 74.5 | 75.5 |

Key facts:
- 30.5B total params, **3.3B activated** (MoE with 128 experts, 8 active)
- Non-thinking mode (fast, no `<think>` overhead)
- 262K native context
- Q4_K_S GGUF: ~17GB
- Beats GPT-4o on reasoning, writing, logic while activating only 3B params

### Memory: Qwen3-0.6B

- Tiny (Q4: ~0.4GB)
- Good at following templates, summarization
- Runs on literally anything
- Handles: session compaction, fact extraction, memory updates

### Considered but rejected for main slot:

**Qwen3-Next-80B-A3B-Instruct** — Amazing model (beats 235B on coding, 82.7 Arena-Hard) but:
- Q4_K_S = 42GB (won't fit any consumer hardware)
- Q2_K = 27GB (still too big with workers)
- Even IQ1_S = 21GB (bigger than 30B-A3B Q4, terrible quality)
- 512 experts × storage = too many dormant weights
- Designed for multi-GPU cloud, not local

---

## Hardware Tiers

### Tier 0: Potato (8GB RAM, no GPU)
> Base MacBook Air, cheap laptop, Raspberry Pi 5

```
Model: Nanbeige4.1-3B Q4_K_M (~2GB)
Roles: Main + RLM + Memory (single model, all roles)
RAM:   ~2GB model + ~4GB OS = 6GB total
```

**What you get:** An agent that beats Qwen3-32B on tool calling and sustains 500+ tool rounds. On a $999 laptop. This is the killer onboarding story.

### Tier 1: Sweet (16GB RAM)
> MacBook Air M2/M3 16GB, gaming laptop with 16GB

```
Main:   Qwen3-8B-Instruct Q4_K_M (~5GB)
RLM:    Nanbeige4.1-3B Q4_K_M (~2GB)
Memory: Qwen3-0.6B Q4_K_M (~0.4GB)
Total:  ~7.4GB + OS
```

**What you get:** Full 3-model setup. Strong conversation + unstoppable agent + background memory. All local, all free.

### Tier 2: Power (24GB — RTX 3090, M2 Pro 24GB)
> Peppi's desktop GPU

```
Main:   Qwen3-30B-A3B-Instruct-2507 Q4_K_S (~17GB) [GPU]
RLM:    Nanbeige4.1-3B Q8_0 (~3.5GB) [CPU]
Memory: Qwen3-0.6B Q4_K_M (~0.4GB) [CPU]
Total:  ~21GB
```

**What you get:** GPT-4o class main model + best-in-class agent + memory. On the GPU: 30B MoE flies. On CPU: 3B models are instant. 3GB headroom.

### Tier 3: Beast (32GB — M4 32GB, dual GPU)
> Peppi's MacBook

```
Main:   Qwen3-30B-A3B-Instruct-2507 Q4_K_M (~17GB)
RLM:    Nanbeige4.1-3B Q8_0 (~3.5GB)
Memory: Qwen3-0.6B Q4_K_M (~0.4GB)
Total:  ~21GB, 11GB free for context/batching
```

**What you get:** Same as Tier 2 but with massive headroom. Can run longer contexts, batch requests, or upgrade to Q5/Q6 quants for better quality.

### Tier 4: Hybrid (any hardware + API key)
> Users who want cloud main + local workers

```
Main:   Cloud (Claude Opus, GPT-4o, Gemini) [API]
RLM:    Nanbeige4.1-3B Q8_0 (~3.5GB) [local]
Memory: Qwen3-0.6B Q4_K_M (~0.4GB) [local]
Total:  ~4GB local
```

**What you get:** Best cloud reasoning + local agent execution (private, fast, no round-trips for tool calls) + local memory (no data leaves machine).

---

## Download Guide

### Essential (everyone needs these):
```bash
# RLM model — the star
# Already have: ~/models/nanbeige4.1-3b-q8_0.gguf

# Memory model
# Already have: ~/.nanobot/models/Qwen3-0.6B.Q4_K_M.gguf
```

### Main model (Tier 2+):
```bash
# Download in LM Studio: search "Qwen3-30B-A3B-Instruct-2507"
# Pick Q4_K_S (~17GB) or Q4_K_M (~18GB)
# Or via huggingface-cli:
huggingface-cli download unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF \
  --include "Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf" \
  --local-dir ~/models/
```

---

## Config Integration

### nanobot config.json model slots:
```json
{
  "local": {
    "main": {
      "model": "Qwen3-30B-A3B-Instruct-2507",
      "path": "~/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf",
      "gpu": true
    },
    "rlm": {
      "model": "Nanbeige4.1-3B",
      "path": "~/models/nanbeige4.1-3b-q8_0.gguf",
      "gpu": false
    },
    "memory": {
      "model": "Qwen3-0.6B",
      "path": "~/.nanobot/models/Qwen3-0.6B.Q4_K_M.gguf",
      "gpu": false
    }
  }
}
```

### Auto-detection (future):
- Scan for `.gguf` files in known paths
- Detect available RAM/VRAM
- Auto-assign models to slots based on hardware tier
- `nanobot doctor` shows detected config

---

## Inference Settings

### Nanbeige4.1-3B (RLM):
- Temperature: 0.6
- Top-p: 0.95
- Repeat penalty: 1.0
- Max tokens: 131072 (supports very long agent chains)
- Tool format: `<tool_call>{"name": ..., "arguments": ...}</tool_call>`

### Qwen3-30B-A3B-2507 (Main):
- Temperature: 0.7
- Top-p: 0.8
- Repeat penalty: 1.05
- Max tokens: 16384 (conversation)
- Non-thinking mode (no `enable_thinking` needed, it's the default)

### Qwen3-0.6B (Memory):
- Temperature: 0.3 (deterministic for summaries)
- Top-p: 0.9
- Max tokens: 2048 (summaries are short)

---

## Roadmap

### Now
- [x] Nanbeige4.1-3B identified as RLM model
- [x] Qwen3-30B-A3B-2507 identified as main model
- [x] Hardware tier matrix defined
- [ ] Download Qwen3-30B-A3B-Instruct-2507 GGUF
- [ ] Test Nanbeige tool calling with nanobot's `<tool_call>` format

### Next
- [ ] Multi-model llama-server management (start/stop per slot)
- [ ] Config schema for 3-slot model assignment
- [ ] Auto hardware detection + tier assignment
- [ ] Benchmark: Nanbeige vs Ministral-3B vs Gemma-3n on actual nanobot RLM tasks

### Future
- [ ] LoRA adapters per role (fine-tune Nanbeige for nanobot-specific tool patterns)
- [ ] Dynamic model swapping (load/unload based on activity)
- [ ] Learned routing (confidence-based escalation to cloud)
