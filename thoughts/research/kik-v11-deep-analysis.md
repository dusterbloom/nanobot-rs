# KIK Edge v11 Deep Analysis — For Nanobot Integration

## Executive Summary

KIK Edge v11 is a 4-layer memory hierarchy that augments a frozen small LM (SmolLM-135M) with infinite-context memory, achieving 100% copy accuracy and 100% QA accuracy on synthetic evals. Total memory footprint: ~437 MB on a 24 GB GPU.

**What it does that nanobot doesn't:** Token-level memory (L0/L1/L2), surprise-gated learning, delta-rule compressed state, LoRA-based continual weight updates, and dual-embedding (H_0/H_30) retrieval.

**What nanobot already has that KIK doesn't:** LLM-based summarization, multi-provider routing, session management, per-channel isolation, tool outcome tracking, audience-aware compaction.

**Key insight for composition:** KIK operates at the *sub-token* level (embedding space, logit mixing). Nanobot operates at the *message* level (text summarization, session snapshots). They are complementary layers, not competing ones.

---

## Source Files Reference

| File | LOC | Layer | Path |
|------|-----|-------|------|
| `kik_edge_v11.py` | 3340 | L0/L0.5/L1/L2/L3 + orchestrator | `C:\Users\PC\Dev\KIK\zero\` |
| `timewarp.py` | 397 | SurpriseGate + MemoryBuffer | `C:\Users\PC\Dev\KIK\zero\` |
| `continual_learner.py` | 954 | LoRA + NREM consolidation | `C:\Users\PC\Dev\KIK\zero\` |
| `sam_core.py` | ~400 | L1 mmap foundation | `C:\Users\PC\Dev\KIK\zero\` |
| `fast_memory.py` | 557 | HNSW-accelerated L0.5 | `C:\Users\PC\Dev\KIK\zero\` |
| `buffer_base.py` | 26 | SOLID buffer interface | `C:\Users\PC\Dev\KIK\zero\` |
| `memory_store.py` | 750 | SQLite episodic store (L3+) | `C:\Users\PC\Dev\KIK\zero\` |
| `stream_processor.py` | 267 | Document chunking | `C:\Users\PC\Dev\KIK\zero\` |

---

## Architecture: The 5-Layer Hierarchy

```
Token Stream
    |
    v
[L0]  ExactPrefixCache      — hash(last_3_tokens) -> next_token, O(1) lookup
    |                          ~8 MB, volatile, 100K entries max
    |
[L0.5] FactMemory/HNSW      — semantic fact store, dual H_0/H_30 embeddings
    |                          ~1 KB/fact, 1000 facts max, HNSW O(log N) search
    |                          Score: 0.3*h0_sim + 0.7*h30_sim + overlap_boost
    |
[L1]  SAMGatedMemory         — Wk-based KNN over mmap vault, surprise-gated writes
    |                          ~80 MB, persistent (mmap file), 10K entries
    |                          Gate: w=(0.3,0.4,0.3) * (top_agree, kl_sim, confidence)
    |
[L2]  DeltaStateMemory       — Fixed matrix V via error-correcting delta rule
    |                          ~37 MB, volatile, 16K slots (product keys)
    |                          Gate: 1 - cos_sim(prediction, target), DeltaNet-style
    |
[L3]  ArchivedVault          — Bloom filter + sharded binary files
                               O(1) per query, persistent (disk)
```

### Query Priority Cascade

```
prompt -> tokenize -> token_list
    |
    v
L0.5 FIRST (before token loop):
    forward_cached(prompt) -> H_0, H_30
    fact_memory.retrieve_best(H_0, tokens, H_30)
    if conf > threshold:
        augmented_prompt = "Fact: {fact} Question: {prompt} Answer:"
        model.generate(augmented_prompt) -> RETURN immediately
    |
    v [for each of max_new tokens]
L0: l0_cache.lookup(token_list[-3:])
    if HIT: next_token = cached_id -> RETURN token
    |
L1: forward pass -> H_0[-1] as query
    retrieve_gated(query, kernel_logits)
    if gate > 0.3:
        scale = max(|ker_logits|) / max(syn_logits)
        combined = (1-gate)*ker_logits + gate*(syn_logits*scale)
        next_token = argmax(combined) -> RETURN token
    |
L2: retrieve_token_logits(query, embed_weight)
    if conf > 0.2:
        l2_gate = conf * 0.5
        combined = (1-l2_gate)*ker_logits + l2_gate*l2_logits
        next_token = argmax(combined) -> RETURN token
    |
L3: maybe_retrieve(encoded_key)
    if result:
        combined = 0.7*ker_logits + 0.3*l3_logits
    else:
        combined = ker_logits (kernel only)
    next_token = argmax(combined)
```

### Imprint (Write) Flow

```
text -> tokenize -> model(input_ids, output_hidden_states=True)
    |
    ├─ H_0 = hidden_states[0]    (layer 0, form-invariant)
    └─ H_30 = hidden_states[-1]  (final layer, semantic)
         |
         ├─ logits -> probs -> surprise_i = -log(p(next_token))
         |   mask_i = (surprise_i > 0.5)
         |
         ├─ L0: for t: l0_cache.add(token_ids[t-2:t+1], token_ids[t+1])
         |
         ├─ L0.5: fact_memory.add(text, H_0, token_ids, H_30)
         |   mean_pool -> normalize -> check supersession -> append/evict
         |
         ├─ L1: if mask[i]: l1_memory.add(H_0[i], target_id)
         |   encode_key -> normalized float16 -> write to mmap vault
         |
         └─ L2: l2_memory.update(H_0[i], target_embedding)  [ALWAYS]
             encode_key -> IDW scores -> prediction = scores @ V
             gate = 1 - cos_sim(prediction, target)
             V -= gate * lr * outer(scores, error)
             z += scores * gate
```

**BUG (line 2050):** In `imprint()`, the L2 update is indented inside `if mask[i]`, so L2 only updates on surprising tokens. The comment says "ALWAYS update." `imprint_silent()` does NOT have this bug.

---

## Core Classes — Detailed Reference

### ExactPrefixCache (L0) — `kik_edge_v11.py:345-378`

```python
class ExactPrefixCache:
    def __init__(self, n_gram=3, max_entries=100_000)
    def add(self, prefix_ids: List[int], target_id: int)      # L354
    def lookup(self, prefix_ids: List[int]) -> (int, bool)     # L368
    def __len__(self) -> int                                    # L377
```

Pure dict-based. Key = tuple of last `n_gram` token IDs. FIFO eviction. No persistence.

### FactMemory (L0.5) — `kik_edge_v11.py:385-862`

```python
class FactMemory:
    def __init__(self, tokenizer=None, max_facts=1000, confidence_threshold=0.4, tau_half=100)
    def add(self, text, h0_embeddings, token_ids=None, h30_embeddings=None)  # L478
    def retrieve(self, query_h0, query_tokens=None, top_k=3, query_h30=None) # L538
    def retrieve_best(self, query_h0, query_tokens=None, query_h30=None)     # L673
    def save(self, filepath)                                                  # L714
    def load(self, filepath) -> bool                                          # L748
```

**Scoring formula:**
```
overlap_bonus = 0.3 * overlap_count
base_score = overlap_bonus + max(0, h30_sim) * 0.7
temporal_weight = 1 / (1 + age / tau_half)    # hyperbolic
score = base_score * temporal_weight
```

**Supersession:** `_find_supersedable()` (L437) — if existing fact has 2+ token overlap AND H_30 cosine sim > 0.85, replace in-place instead of appending.

**HNSW variant:** `HNSWFactMemory` in `fast_memory.py:50-556` — same interface, O(log N) via `hnswlib`. Score: `(0.3*h0 + 0.7*h30 + min(0.5, overlap*0.15)) * 0.5^(age/tau_half)`.

### SAMGatedMemory (L1) — `kik_edge_v11.py:869-1025`

```python
class SAMGatedMemory:
    def __init__(self, wk_weight, wq_weight, embed_weight, config, vocab_size, device='cpu')
    def encode_key(self, hidden_state) -> Tensor          # L940
    def add(self, hidden_state, target_id)                 # L955
    def retrieve_gated(self, query, kernel_logits=None)    # L988
    def flush(self)                                         # L1019
    def close(self)                                         # L1022
```

**Mmap vault format:** 8-byte header (uint64 count) + entries of `(float16 key [d_key*2 bytes], uint32 token_id [4 bytes])`.

**Gate formula:**
```
top_agree = float(argmax(ker_logits) == argmax(syn_logits))
kl_div = KL(ker_probs || syn_probs*10)
kl_similarity = exp(-kl_div * 0.1)
mem_confidence = weights[0]  # top KNN weight
gate = 0.3*top_agree + 0.4*kl_similarity + 0.3*mem_confidence
gate = clamp(gate, 0.0, 1.0)
```

**Logit blending:**
```
scale = max(|ker_logits|) / max(syn_logits)
combined = (1 - gate) * ker_logits + gate * (syn_logits * scale)
```

### DeltaStateMemory (L2) — `kik_edge_v11.py:1032-1346`

```python
class DeltaStateMemory:
    def __init__(self, wk_weight, config, vocab_size, device='cpu')
    def update(self, hidden_state, target_embedding)                    # L1178
    def retrieve(self, hidden_state) -> (Tensor, float)                 # L1270
    def retrieve_token_logits(self, hidden_state, embed_weight)         # L1319
```

**Product key decomposition:** `K_full[i*sqrt_n + j] = K1[i] + K2[j]` gives 16384 slots from 256 stored vectors.

**IDW attention:** `scores = softmax((1/(dist_sq + 1e-6)) / temperature)` with temperature=0.1.

**Delta rule update:**
```
prediction = scores @ V
gate = clamp(1 - cos_sim(prediction, target), 0.01, 1.0)
error = prediction - target
V -= gate * lr * outer(scores, error)
z += scores * gate    # Infini-attention normalization
```

**Retrieval with temporal decay:**
```
ages = global_tau - tau_last
temporal_w = exp(-ages / tau_half)
weighted_scores = normalize(scores * temporal_w)
raw = weighted_scores @ V
retrieved = raw / (weighted_scores @ z + 1e-8)    # Infini-attention z-norm
```

**Confidence:** entropy-based: `conf = 1 - entropy(weighted_scores) / log(n_slots)`

### ArchivedVault (L3) — `kik_edge_v11.py:1423-1557`

```python
class ArchivedVault:
    def __init__(self, config, d_key, vocab_size)
    def archive(self, key, token_id)                      # L1467
    def maybe_retrieve(self, key, k=5)                    # L1494
```

Bloom filter (MD5+SHA1 double-hashing) for fast negative lookups. 256 shard files with binary entries. O(1) per query via Bloom, linear scan within shard on positive.

### BloomFilter — `kik_edge_v11.py:1352-1420`

```python
class BloomFilter:
    def __init__(self, capacity=10_000_000, error_rate=0.01)
    def add(self, key: bytes)                              # L1401
    def maybe_contains(self, key: bytes) -> bool           # L1407
```

`m = -(n * ln(p)) / ln(2)^2`, `k = (m/n) * ln(2)`.

---

## Continual Learning System

### SurpriseGate — `timewarp.py:50-159`

```python
class SurpriseGate(nn.Module):
    def __init__(self, threshold=1.0, temperature=1.0)
    def compute_surprise(self, logits, targets) -> Tensor     # L89
    def forward(self, logits, targets) -> (gate, surprise)    # L112
```

**Algorithm:**
1. `surprise = cross_entropy(logits, targets, reduction='none')` — per-token
2. EMA update (alpha=0.1): `running_mean = 0.9*mean + 0.1*batch_mean`
3. Normalize: `z = (surprise - running_mean) / sqrt(running_var)`
4. Gate: `sigmoid((z - threshold) / temperature)`

Gate > 0.5 means token is > 1 std above running mean surprise.

### MemoryBuffer — `timewarp.py:162-334`

```python
class MemoryBuffer:
    def __init__(self, max_size=10000)
    def add(self, input_ids, target_ids, surprise, memory_confidence=0.0)  # L203
    def sample(self, n, weighted=True) -> List[TaggedMemory]                # L241
    def get_batch(self, n, device="cuda") -> Optional[Dict]                 # L277
    def clear(self)                                                         # L310
```

Min-heap ordered by surprise. Full buffer: new entry replaces minimum only if more surprising. Sampling: weighted by raw surprise (proportional probability).

### TaggedMemory — `timewarp.py:31-47`

```python
@dataclass
class TaggedMemory:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    surprise: float
    memory_confidence: float
    timestamp: int
```

### ContinualLearner — `continual_learner.py:182-954`

```python
class ContinualLearner:
    def __init__(self, config, edge, device="cuda")
    def on_learn(self, text, verbose=False, surprise_score=None)           # L236
    def on_correction(self, original_context, wrong_response, correction)  # L306
    def consolidate(self, verbose=False) -> Dict                            # L572
    def save_checkpoint(self)                                               # L795
    def load_checkpoint(self) -> bool                                       # L882
```

**Two paths:**

**Fast path** (on_correction/on_learn, ~10ms):
1. Tokenize (chat template if instruct, mask prompt with -100)
2. Compute surprise via SurpriseGate
3. If surprise > 0: buffer to MemoryBuffer
4. If `update_l2_immediately`: forward pass -> update L2 delta memory per token
5. If `instant_mode`: apply LoRA, 3 SGD steps at 5x LR, save checkpoint

**Slow path** (consolidate, ~5-30s):
1. Apply LoRA (rank-16, q_proj/k_proj/v_proj, never merged)
2. AdamW optimizer on LoRA params
3. Loop up to 50 steps: sample batch from buffer, CE loss (label_smoothing=0.1)
4. Optional: self-distillation (KL with frozen teacher)
5. NTE protection: EMA of grad^2, scale gradients for important params (90% suppression)
6. Early stop at loss < 0.1
7. Clear buffer, save checkpoint

### LoRALayer — `continual_learner.py:86-134`

```python
class LoRALayer(nn.Module):
    def __init__(self, base_layer, rank=16, alpha=16.0, dropout=0.0)
    def forward(self, x) -> Tensor    # L126
```

A = Kaiming init, B = zeros. Scaling = alpha/rank = 1.0. **Never merged into base weights.** BFloat16 safe (explicit dtype cast).

### ImportanceTracker — `continual_learner.py:141-175`

```python
class ImportanceTracker:
    def __init__(self, model, decay=0.99)
    def update(self, model)                                    # L155
    def scale_gradients(self, model, scale_factor=0.1)         # L164
```

Fisher Information via EMA of grad^2. High-importance params get 90% gradient suppression.

---

## Key Hyperparameters (All Validated Values)

### EdgeV11Config — `kik_edge_v11.py:241-338`

| Param | Default | Purpose |
|-------|---------|---------|
| `n_gram` | 3 | L0 prefix size |
| `l0_max_entries` | 100,000 | L0 cache cap |
| `l1_max_memories` | 10,000 | L1 vault cap |
| `k_neighbors` | 16 | L1 KNN count |
| `surprise_threshold` | 0.5 | L1 write gate |
| `l1_temperature` | 0.05 | L1 softmax temp |
| `l1_gate_weights` | (0.3, 0.4, 0.3) | (agree, kl_sim, confidence) |
| `l2_n_slots` | 16,384 | L2 product key slots |
| `l2_learning_rate` | 0.1 | L2 delta rule LR |
| `l2_temperature` | 0.1 | L2 IDW softmax temp |
| `l2_gate_min/max` | 0.01/1.0 | L2 adaptive gate bounds |
| `fact_max_facts` | 1,000 | L0.5 capacity |
| `fact_confidence_threshold` | 0.35 | L0.5 retrieval gate |
| `l0_5_tau_half` | 100 | L0.5 decay (hyperbolic) |
| `l1_tau_half` | 1,000 | L1 decay |
| `l2_tau_half` | 10,000 | L2 decay (exponential) |
| `l3_bloom_capacity` | 10,000,000 | L3 Bloom cap |
| `l3_n_shards` | 256 | L3 shard count |
| `yarn_scale_factor` | 1 | YaRN context extension |

### ContinualLearnerConfig — `continual_learner.py:40-79`

| Param | Default | Purpose |
|-------|---------|---------|
| `surprise_threshold` | 1.0 | SurpriseGate z-score threshold |
| `buffer_max_size` | 1,000 | MemoryBuffer heap cap |
| `consolidation_threshold` | 5 | Min buffered before consolidation |
| `lora_rank` | 16 | LoRA rank r |
| `lora_alpha` | 16.0 | LoRA scaling (alpha/rank = 1.0) |
| `lora_target_modules` | (q_proj, k_proj, v_proj) | LoRA injection targets |
| `nrem_max_steps` | 50 | Max consolidation steps |
| `nrem_learning_rate` | 5e-5 | Slow path LR (instant = 5x) |
| `nrem_batch_size` | 8 | Replay batch size |
| `nrem_loss_threshold` | 0.1 | Early stop threshold |
| `nte_decay` | 0.99 | Fisher importance EMA |
| `instant_steps` | 3 | Fast path gradient steps |

### Temporal Decay Constants

```
L0.5: tau_half = 100 facts,   decay = hyperbolic: 1/(1 + age/tau)
L1:   tau_half = 1000 tokens,  decay = hyperbolic
L2:   tau_half = 10000 tokens, decay = exponential: exp(-age/tau)
```

### Scoring Weights

```
L0.5: 0.3 * h0_sim + 0.7 * h30_sim + 0.3 * overlap_count
L1 gate: 0.3 * top_agree + 0.4 * kl_similarity + 0.3 * mem_confidence
L2 blending: l2_gate = confidence * 0.5
L3 blending: fixed 0.7 * kernel + 0.3 * archive
```

---

## Persistence Summary

| Layer | Persistent? | Format | Auto-save? |
|-------|-------------|--------|------------|
| L0 | No | Python dict in RAM | N/A |
| L0.5 | Yes | `torch.save()` to `.pt` | Manual (`save_facts()`) |
| L1 | Yes | mmap binary file | Auto (every `add()`) |
| L2 | No | GPU tensors | N/A |
| L3 | Yes | Binary shard files + Bloom (RAM only) | Auto (every `archive()`) |
| LoRA | Yes | `torch.save()` to `.pt` | After consolidation |

---

## Helper Functions

| Function | File:Line | Purpose |
|----------|-----------|---------|
| `temporal_weight(age, tau_half, decay_type, legacy_weight)` | v11:68 | Unified hyperbolic/exponential decay |
| `canon_smooth(hidden_states, window=3)` | v11:110 | 1D conv smoothing (UNUSED) |
| `apply_yarn_scaling(model, scale_factor, original_max_pos)` | v11:157 | NTK-aware RoPE extension |
| `_update_rotary_emb(rotary, scale_factor, rope_theta)` | v11:206 | RoPE frequency recomputation |

---

## Nanobot's Existing Memory (For Comparison)

### What Nanobot Has

| Component | File | What It Does |
|-----------|------|-------------|
| `ContextCompactor` | `compaction.rs` | LLM-summarizes old messages at 66.6% context |
| `WorkingMemoryStore` | `working_memory.rs` | Per-session snapshots (SESSION_*.md) |
| `MemoryStore` | `memory.rs` | Long-term MEMORY.md (permanent facts) |
| `Reflector` | `reflector.rs` | Background: completed sessions -> MEMORY.md |
| `LearningStore` | `learning.rs` | Tool outcome tracking (learnings.jsonl) |
| `RecallTool` | `recall.rs` | Agent-callable search across all memory |
| `SessionIndexer` | `session_indexer.rs` | JSONL -> SESSION_*.md conversion |
| `ContextStore` | `context_store.rs` | Ephemeral KV store for large tool outputs |

### Layer Comparison

| Concern | KIK v11 | Nanobot | Gap |
|---------|---------|---------|-----|
| **Exact recall** | L0 n-gram cache | None | KIK wins |
| **Fact retrieval** | L0.5 HNSW + dual embedding | RecallTool (BM25/grep) | KIK has embedding search |
| **Associative memory** | L1 mmap KNN | None | KIK unique |
| **Compressed state** | L2 delta rule matrix | None | KIK unique |
| **Archival** | L3 Bloom + shards | SESSION_*.md archived | Different approaches |
| **Summarization** | None | ContextCompactor (LLM) | Nanobot wins |
| **Session tracking** | None | WorkingMemoryStore | Nanobot wins |
| **Reflection** | None | Reflector (LLM distills facts) | Nanobot wins |
| **Tool learning** | None | LearningStore | Nanobot wins |
| **Continual learning** | LoRA + NREM + SurpriseGate | None | KIK unique |
| **Gradient compaction** | None | 3-tier (raw/light/facts) | Nanobot wins |
| **Multi-user** | memory_store.py (SQLite) | Per-channel isolation | Different |

### Key Architectural Difference

**KIK operates in embedding space** — it intercepts hidden states (H_0, H_30), modifies logits, and updates weight matrices. It requires direct access to the model's internals (hidden_states, embed_weight, attention projections).

**Nanobot operates in text space** — it manipulates message strings, calls LLMs as black boxes via API, and stores/retrieves text. It has no access to model internals.

**This means:** KIK's L0/L1/L2 layers CANNOT be ported to nanobot's cloud/API mode. They only make sense for **local mode** where nanobot hosts the model directly and has access to hidden states.

---

## Composition Strategy (Preliminary)

### What to Bring to Nanobot Local Mode

**High value, low complexity:**
1. **SurpriseGate** — port to Rust. Lightweight (cross-entropy + EMA + sigmoid). Use it to decide which messages/facts are worth remembering. Currently nanobot's Reflector uses a flat "dump everything" approach.

2. **FactMemory with temporal decay** — port the hyperbolic decay + supersession logic. Nanobot's MEMORY.md uses tail truncation (drop oldest). Temporal decay is better — facts fade gracefully, frequently-recalled facts persist.

3. **Delta rule compressed state (L2 concept)** — NOT the tensor-level implementation, but the IDEA: maintain a fixed-size "compressed understanding" that updates incrementally. Could be implemented as a fixed-budget summary that gets updated via LLM (not gradient descent) after each conversation turn.

**Medium value, medium complexity:**
4. **MemoryBuffer with surprise-weighted sampling** — for replay/reflection. Instead of reflecting on ALL completed sessions, reflect on the most surprising ones first.

5. **Dual-strategy retrieval** — nanobot's RecallTool uses BM25 or grep. Adding embedding-based search (form-invariant addressing) would solve the paraphrase problem ("Alice is an engineer" vs "What is Alice's job?").

**Low value for nanobot (but high value for a dedicated local LLM):**
6. L0 n-gram cache — only useful when generating tokens directly
7. L1 mmap KNN — requires hidden state access
8. L2 delta matrix — requires hidden state access
9. LoRA continual learning — requires weight access

### Proposed New Architecture for Nanobot Local Mode

```
Message arrives
    |
    v
[Existing] ContextCompactor — keeps conversation within context
    |
    v
[NEW] SurpriseEstimator — uses perplexity/novelty of response
    |   to gate what enters memory (port of SurpriseGate concept)
    |
    v
[Existing] WorkingMemoryStore — per-session snapshots
    |
[NEW] TemporalFactStore — facts with decay, supersession
    |   replaces flat MEMORY.md with time-aware store
    |   hyperbolic decay: w = 1/(1 + age/tau_half)
    |
[Existing] Reflector — but now weighted by surprise
    |   reflects on high-surprise sessions first
    |
[NEW for local-only] EmbeddingMemory — if running local LLM
    |   with hidden_state access, add L1-style KNN retrieval
    |   as an additional recall strategy alongside BM25/grep
```

### What NOT to Do

- Don't try to port the full 5-layer hierarchy. It's designed for a 135M model that needs external memory to function. Nanobot's local models (8B-72B) have much more parametric knowledge.
- Don't add LoRA/NREM to nanobot. The continual learning system assumes a tiny base model that benefits from weight updates. 8B+ models don't need this.
- Don't duplicate the L3 Bloom+shard system. Nanobot already has disk-based session archives.
- Don't add product key decomposition. It's an optimization for tensor-space memory that doesn't apply to text-space.

---

## Known Bugs and Issues

1. **L2 update indentation bug** (`kik_edge_v11.py:2050`) — L2 only updates on surprising tokens despite comment saying "ALWAYS". `imprint_silent()` is correct.

2. **SQL injection in memory_store.py** (`memory_store.py:487`) — `role_filter` is string-interpolated into SQL. Not exploitable in practice (internal use only) but should be parameterized.

3. **Auto-consolidation not called** (`continual_learner.py:364-372`) — `on_correction` checks threshold but only logs, doesn't call `consolidate()`. Consolidation only happens via session-end hook or manual call.

4. **`wrong_response` ignored** (`continual_learner.py:307`) — Parameter accepted but never used in tokenization or loss.

5. **`canon_smooth` unused** (`kik_edge_v11.py:110`) — Experimental function, not called anywhere.

6. **L3 Bloom filter not persisted** — Rebuilt on every restart. On-disk shards survive but Bloom index is lost.
