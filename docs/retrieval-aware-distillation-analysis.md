# Retrieval-Aware Distillation for Transformer-SSM Hybrids — Analysis

**Paper**: [arXiv:2602.11374](https://arxiv.org/abs/2602.11374)  
**Authors**: Eric P. Xing (CMU/MBZUAI), Albert Gu (CMU/Cartesia AI)  
**Date**: Feb 11, 2026  
**Reviewed**: Feb 14, 2026

---

## TL;DR

In a 1B Transformer with 512 attention heads, **only 10 heads (~2%) are doing retrieval**. The other 98% can be replaced with SSMs (Mamba-2) with essentially no performance loss. Once those 10 heads handle retrieval, the SSM state dimension can shrink 8× (64 → 8). Result: 5–6× more memory-efficient than comparable hybrids.

---

## Core Findings

### 1. The 2% Rule

The paper identifies a small subset of attention heads called **Gather-and-Aggregate (G&A)** heads that are solely responsible for in-context retrieval. These implement a two-phase pattern:

- **Gather Heads**: Compress local information into "transport" tokens. E.g., in `"scallops:50\n"`, a Gather head moves semantic info from `scallops` and `50` into the `\n` token, creating a summary vector.
- **Aggregate Heads**: Perform global retrieval by attending to transport tokens across the full sequence, matching the query and extracting the stored value.

SSMs approximate this implicitly but can't match it. The paper's solution: don't make them try — just keep the 10 heads that do it well.

### 2. The Numbers (Table 1 — Hybrid-Llama-1B)

| # Attention Heads | KV-Retrieval | SWDE  | Retrieval Coverage |
|-------------------|-------------|-------|-------------------|
| 0 (pure SSM)      | 13.2%       | 27.7% | 49.2%             |
| 5                  | 90.0%       | 66.0% | 88.3%             |
| **10 (2%)**        | **99.0%**   | **71.1%** | **95.0%**     |
| 20                 | 99.3%       | 72.5% | 95.8%             |
| 512 (full Transformer) | 99.4%  | 75.3% | 98.2%             |

Going from 10 → 512 heads barely moves the needle. Knowledge-focused tasks (ARC, PIQA, HellaSwag, Winogrande) are **unaffected** — the SSM handles them perfectly at 0 heads (~101% coverage).

Same pattern holds for Qwen2.5-1.5B: 10 heads out of 336 → 96.4% retrieval coverage.

### 3. SSM State Shrinks 8×

Once retrieval is handled by G&A heads, the SSM backbone doesn't need large states. State dimension can go from 64 → 8 with limited degradation. The large states were **compensating for missing retrieval capability** — they're no longer needed.

### 4. Method: Retrieval-Aware Distillation

Built on the MOHAWK framework (3-stage distillation):

1. **Ablation scoring**: Run a synthetic KV-retrieval probe on every attention head. Measure accuracy drop when each head is zeroed out. Rank by retrieval importance.
2. **Architecture construction**: Keep top-k G&A heads, replace the rest with DiscreteMamba2 (SSM) heads. Uses a Static LayerNorm adapter to align distributions between attention and SSM outputs.
3. **Distillation**: Standard MOHAWK pipeline — matrix orientation → hidden-state alignment → weight transfer + logit KD.

Key insight: attention placement is **non-uniform** (not every-nth-layer). The G&A heads cluster in specific layers, and the method preserves them wherever they are.

---

## Connection to Kernel Architecture

This paper empirically validates the kernel architecture from the opposite direction — we designed it from theory (HRR + chirp + operator memory), they discovered it by ablating attention heads in Transformers.

### G&A ↔ Kernel Memory Mapping

| Paper Concept | Kernel Equivalent | Function |
|---------------|-------------------|----------|
| Gather Heads | Chirp encoding | Compress local info into transport representations |
| Aggregate Heads | Associative retrieval | Match query to stored keys, extract value |
| SSM backbone | Kernel recurrent state | General language modeling in compressed form |
| SSM state (dim 8) | L0 working memory (8-16 slots) | Minimal precise state |
| G&A attention cache | L1 episodic (chirp-addressed bank) | Exact retrieval when needed |

### The Kernel Epiphany, Validated

> "The LLM is the interface, not the mind. Real intelligence lives in a kernel that operates in non-linguistic representations."

This paper proves the architectural version:
- **The kernel = SSM backbone** — handles 98% of computation in compressed recurrent state (non-linguistic)
- **The interface = 10 G&A heads** — the ONLY part that needs to look back at raw tokens
- **Everything else works from compressed state** — exactly the holographic/ternary representation hypothesis

### Superposition Capacity Connection

Our kernel-bridge experiment found: naive superposition capacity ~ O(sqrt(D)). This paper finds: SSM state dim 8 suffices when retrieval is handled separately. These are the same finding from different angles — you don't need huge states for general pattern matching, only for retrieval. Separate the retrieval, and small states work fine.

---

## Actionable Next Steps

### 1. Create a Hybrid Local Model

Recipe for Nanbeige4.1-3B (target local model):
1. Run KV-retrieval ablation to find its G&A heads
2. Replace 98% of attention with Mamba-2/SSM
3. Distill with MOHAWK (3 stages, parallelizable per-layer)
4. Result: ~3B hybrid, much faster inference, much less KV-cache on RTX 3090

### 2. CALM Bridge Targeting

The CALM bridge (cross-attention between frozen LLM and augmenting model) should attach specifically to the G&A heads, not the whole model. This dramatically reduces bridge parameters:
- Instead of bridging all 512 heads → bridge only 10
- The SSM backbone doesn't need bridging (it handles general modeling fine)
- Bridge = kernel's chirp-addressed memory ↔ G&A retrieval heads

### 3. LoRA Efficiency

For Zero's instant learning:
- LoRA updates only need to touch the SSM backbone (new knowledge) and G&A heads (new retrieval patterns)
- With 98% of heads replaced by SSMs, the LoRA parameter count drops dramatically
- SSM parameters are more LoRA-friendly than attention (no KV-cache interaction)

### 4. Kernel Bridge Experiment Update

The next kernel-bridge experiment should test:
- Chirp-addressed bank as the G&A replacement (can 10 chirp-addressed slots match 10 attention heads?)
- Operator memory as the SSM replacement (can matrix memory handle general modeling?)
- CALM-AE round-trip with only G&A bridge points

---

## Key References

- **MOHAWK**: Bick et al., 2024 — "Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models"
- **G&A Mechanism**: Bick et al., 2025b — "Understanding the Skill Gap in Recurrent Language Models"
- **Llamba**: Bick et al., 2025a — "Scaling Distilled Recurrent Models for Efficient Language Processing"
- **Mamba-2/SSD**: Dao & Gu, 2024 — "Transformers are SSMs: Generalized Models and Efficient Algorithms through Structured State Space Duality"
- **CALM**: https://shaochenze.github.io/blog/2025/CALM/ — Cross-attention bridge between frozen LLM and augmenting model
- **Existing CALM-AE experiments**: `/home/peppi/research/KIK/codex_plastic_calm.py`, `/home/peppi/research/KIK/redux_version_9.py`

---

## Summary

This is the most important architecture paper for our kernel work since the original Mamba paper. It proves that:

1. **Retrieval is a separable function** — 10 heads out of 512, identifiable by ablation
2. **SSMs handle everything else** — knowledge, reasoning, pattern completion
3. **Compressed state suffices** — 8 dimensions, not 64, once retrieval is separate
4. **The hybrid is 5-6× more efficient** — less memory, faster inference

This directly validates the kernel architecture: a small, precise retrieval mechanism (chirp/G&A) + a large, compressed pattern engine (SSM/operator memory) + a bridge between them (CALM/attention). The paper gives us the empirical recipe; our theory gives us the generalization path.
