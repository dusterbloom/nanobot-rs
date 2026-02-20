# Inference Engine Options for Nanobot

**Date**: 2026-02-18  
**Context**: llama.cpp server instability issues - server becomes unhealthy, requires frequent restarts  
**Goal**: Evaluate alternatives and recommend a path forward

---

## Current State: llama.cpp Server

### Configuration (as of 2026-02-18)
```
--port 8080
--ctx-size <adaptive>
--parallel 1
--n-gpu-layers <computed>
--threads <cpu_count - 2>
--batch-size 256 / --ubatch-size 128
--flash-attn on
--jinja
--no-prefill-assistant
--mlock
--cache-reuse 256
```

### Observed Issues
1. **Health check passes but inference fails** - `/health` returns 200 but chat completions error
2. **Frequent restarts needed** - watchdog triggers auto-repair ~daily
3. **VRAM pressure** - 30B model with 512K context pushes memory limits
4. **Race conditions** - server reports healthy before model fully loaded

### Root Causes
- llama.cpp designed for batch inference, not long-running server
- No built-in process supervision
- Limited error recovery
- Memory fragmentation over time

---

## Evaluated Alternatives

### 1. mistral.rs ⭐ RECOMMENDED

| Aspect | Rating | Notes |
|--------|--------|-------|
| Stability | ✅ Excellent | Rust memory safety, proper error handling |
| Performance | ✅ Fast | FlashAttention V2/V3, continuous batching |
| Memory | ✅ Good | PagedAttention prevents OOM |
| OpenAI API | ✅ Full | Drop-in compatible |
| Rust Integration | ✅ Native | `mistralrs` crate available |
| Tool Calling | ✅ Built-in | Native support, no jinja hacks |
| Maintenance | ✅ Active | 6.6k stars, 76 contributors |

**Pros**:
- Pure Rust - can integrate directly into nanobot binary
- PagedAttention - handles memory pressure gracefully
- Continuous batching - better multi-request throughput
- Built-in tool calling support
- Active development (v0.7.0 released Jan 2026)

**Cons**:
- Newer project, may have edge case bugs
- Smaller community than llama.cpp

**Integration Options**:
- **A**: Run as subprocess (like llama.cpp now) - drop-in replacement
- **B**: Native Rust SDK - embed directly in nanobot, no subprocess

```rust
// Option B: Native integration
use mistralrs::{VisionModelBuilder, IsqType, TextMessages};

let model = VisionModelBuilder::new("path/to/model.gguf")
    .with_isq(IsqType::Q3KLarge)
    .build()
    .await?;

let response = model.send_chat_completion_request(request).await?;
```

---

### 2. vLLM

| Aspect | Rating | Notes |
|--------|--------|-------|
| Stability | ✅ Excellent | Production-grade, used by major companies |
| Performance | ✅ Fastest | Optimized PagedAttention, CUDA graphs |
| Memory | ⚠️ Higher | Python overhead, larger baseline |
| OpenAI API | ✅ Full | Drop-in compatible |
| Rust Integration | ❌ None | Python-only |
| Tool Calling | ⚠️ Limited | Via chat templates |

**Pros**:
- Most mature PagedAttention implementation
- Best throughput for concurrent requests
- Industry standard for production

**Cons**:
- Python-only (requires separate process)
- Higher memory overhead
- No Rust SDK

**Command**:
```bash
pip install vllm
vllm serve ~/models/model.gguf --port 8080 --gpu-memory-utilization 0.9
```

---

### 3. Ollama

| Aspect | Rating | Notes |
|--------|--------|-------|
| Stability | ✅ Excellent | Wraps llama.cpp with better process mgmt |
| Performance | ⚠️ Good | Slight overhead from Go wrapper |
| Memory | ✅ Good | Same as llama.cpp |
| OpenAI API | ✅ Full | Port 11434, /v1/... endpoints |
| Rust Integration | ❌ None | Subprocess only |
| Tool Calling | ⚠️ Via templates | |

**Pros**:
- Drop-in replacement, zero config
- Auto-restarts on crash
- Built-in model management
- Large community

**Cons**:
- Still uses llama.cpp under the hood
- Go-based, no Rust integration
- Abstractions hide tuning options

**Command**:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama run nemotron-mini
# API at http://localhost:11434/v1
```

---

### 4. LocalAI

| Aspect | Rating | Notes |
|--------|--------|-------|
| Stability | ✅ Good | Multiple backend support |
| Performance | ⚠️ Varies | Depends on backend chosen |
| Memory | ⚠️ Higher | Go + backend overhead |
| OpenAI API | ✅ Full | Designed as drop-in |
| Rust Integration | ❌ None | Subprocess only |

**Pros**:
- Can use llama.cpp, transformers, or other backends
- Drop-in OpenAI replacement
- Active community

**Cons**:
- Abstraction adds complexity
- Go-based
- Not as fast as native implementations

---

### 5. ExLlamaV2

| Aspect | Rating | Notes |
|--------|--------|-------|
| Stability | ✅ Good | NVIDIA-focused, well-optimized |
| Performance | ✅ Fastest | Best 4-bit inference for NVIDIA |
| Memory | ⚠️ Medium | NVIDIA-only, no CPU fallback |
| OpenAI API | ⚠️ Partial | Requires additional layer |
| Rust Integration | ❌ None | Python-only |

**Pros**:
- Best performance for 4-bit quantization
- Excellent NVIDIA GPU utilization

**Cons**:
- NVIDIA GPUs only (no AMD, no CPU)
- Python-only
- Limited model format support (EXL2)

---

## Comparison Matrix

| Feature | llama.cpp | mistral.rs | vLLM | Ollama |
|---------|-----------|------------|------|--------|
| **Language** | C++ | Rust | Python | Go+C++ |
| **Stability** | ⚠️ Medium | ✅ Good | ✅ Excellent | ✅ Good |
| **Memory Efficiency** | ✅ Best | ✅ Good | ⚠️ Higher | ✅ Good |
| **PagedAttention** | ❌ | ✅ | ✅ | ❌ |
| **Continuous Batching** | ⚠️ | ✅ | ✅ | ⚠️ |
| **OpenAI API** | ✅ | ✅ | ✅ | ✅ |
| **Rust SDK** | ❌ | ✅ | ❌ | ❌ |
| **Tool Calling** | ⚠️ Jinja | ✅ Native | ⚠️ Templates | ⚠️ Templates |
| **GGUF Support** | ✅ Native | ✅ | ✅ | ✅ |
| **GPU Support** | ✅ CUDA/Metal | ✅ CUDA/Metal | ✅ CUDA | ✅ CUDA/Metal |
| **Active Dev** | ✅ | ✅ | ✅ | ✅ |

---

## Recommendation

### Short-term (This Week)
1. **Try mistral.rs as subprocess** - drop-in replacement for llama.cpp
   ```bash
   mistralrs serve -m ~/models/model.gguf --port 8080
   ```
2. Compare stability over 2-3 days

### Medium-term (Next Sprint)
If mistral.rs works well, **integrate the Rust SDK natively**:

```rust
// Cargo.toml
mistralrs = { version = "0.7", features = ["cuda"] }

// src/providers/mistralrs.rs
pub struct MistralRsProvider {
    model: mistralrs::Model,
}

impl LLMProvider for MistralRsProvider {
    async fn chat(&self, messages: &[Value], tools: Option<&[Value]>) -> Result<LLMResponse> {
        self.model.send_chat_completion_request(...).await
    }
}
```

Benefits:
- No subprocess management
- Single binary deployment
- Better error propagation
- Lower latency (no HTTP overhead)

### Long-term
Consider **hybrid approach**:
- mistral.rs for local inference (Rust native)
- vLLM as alternative for high-throughput scenarios (subprocess)

---

## Implementation Plan

### Phase 1: Evaluate mistral.rs (1 day)
- [ ] Install mistral.rs CLI
- [ ] Run with current model on port 8080
- [ ] Monitor stability for 2-3 days
- [ ] Compare memory usage and latency

### Phase 2: Subprocess Integration (2 days)
- [ ] Add `MistralRsProvider` alongside `OpenAICompatProvider`
- [ ] Add config option to select inference engine
- [ ] Auto-detect mistral.rs binary
- [ ] Fallback to llama.cpp if not available

### Phase 3: Native Integration (3-5 days)
- [ ] Add `mistralrs` crate dependency
- [ ] Implement `MistralRsNativeProvider`
- [ ] Handle model loading/unloading
- [ ] Integrate with existing tool calling infrastructure
- [ ] Test with all current models

### Phase 4: Deprecate llama.cpp (1 day)
- [ ] Remove llama.cpp spawn code
- [ ] Update documentation
- [ ] Migration guide for users

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| mistral.rs missing features | Medium | Low | Keep llama.cpp as fallback |
| Performance regression | Low | Medium | Benchmark before switching |
| Model compatibility issues | Low | High | Test with all current models |
| Native SDK bugs | Medium | Medium | Start with subprocess mode |

---

## Decision

**Recommendation**: Proceed with **mistral.rs** evaluation, targeting native Rust integration.

**Rationale**:
1. Pure Rust aligns with nanobot's architecture
2. Native SDK enables single-binary deployment
3. PagedAttention addresses memory stability
4. Active development and growing community
5. Built-in tool calling reduces complexity

**Next Step**: Install and test mistral.rs as drop-in replacement

```bash
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.sh | sh
mistralrs serve -m ~/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Q3_K_L.gguf --port 8080
```
