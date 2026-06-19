# Session 9 Knowledge Transfer: ANE Training on Apple Silicon

## What We Built

Nanobot trains Qwen3.5-35B-A3B entirely on-device on an M4 MacBook using the Apple Neural Engine. No GPU, no cloud. LoRA fine-tuning on the user's own conversation data so the assistant learns their patterns.

This session took the 35B training step from **2443ms to 1148ms** (2.1x) through three orthogonal optimizations, each attacking a different bottleneck.

## The Three Wins

### 1. Parallel GDN Recurrence (2443ms -> 2037ms, -406ms)

**Problem:** Qwen3.5 uses Griffin-style GDN (Gated Delta Network) layers for 30 of its 40 layers. The GDN recurrence scans sequentially over sequence positions, updating a hidden state per head. With 32 value heads and seq=128, this was the forward pass bottleneck at ~530ms.

**Solution:** The recurrence across heads is independent. `rayon::par_iter` over the 32 value heads gives ~3x speedup on the recurrence. The parallelism is limited by M4's 10 performance cores, but 32 heads provide enough work to saturate them.

**Key insight:** The recurrence within each head is inherently sequential (each position depends on the previous). But heads are independent. This is embarrassingly parallel along the head dimension.

### 2. Per-Layer BLOBFILE Backward Kernels (2037ms -> 1680ms, -287ms on backward)

**Problem:** The MHA attention backward used DynMatmul kernels for Wo^T and QKV^T projections. DynMatmul packs weight matrices into the IOSurface input alongside activations. At 35B: Wo is 16MB, WQ+WK+WV is 48MB. That's 64MB of CPU memcpy per MHA layer, per step, for 10 layers = 640MB of redundant weight copies.

**Solution:** Pre-compile per-layer ANE kernels with weights baked in as BLOBFILE (fp16). At eval time, only the 1MB activation input needs to be written to the IOSurface. The weights are already in the compiled program.

Two new MIL generators:
- `gen_wot_bwd_blob`: Wo^T matmul, 1 BLOBFILE (8MB fp16). Input: dx2. Output: da.
- `gen_qkvb_blob`: WQ^T + WK^T + WV^T fused, 3 BLOBFILEs (24MB fp16 total). Input: dQ|dK|dV concatenated. Output: dx (sum of three back-projections).

Both compile at 35B dimensions. The monolithic fused backward attention kernel (all 4 weights + RoPE + mask in one program) was tried earlier and FAILED because 4 weight matrices at 2048x2048 fp16 exceeded the ANE's 32MB SRAM. Splitting into separate Wot and QKV kernels keeps each within budget.

**Key insight:** The ANE compiler rejects programs whose total BLOBFILE weight exceeds ~32MB. Two kernels with 8MB and 24MB each both pass. The fused 32MB+ kernel doesn't. Sometimes splitting is the only viable path.

### 3. Adaptive Layer Drop (automatic, -530ms with 22 layers)

**Problem:** Gradient norms reveal that layers 0-21 are effectively dead during LoRA fine-tuning (<0.01% of max gradient norm after 5 steps). Training these layers wastes compute with zero learning signal.

**Solution:** `adaptive_layer_drop: true` in config. After each backward pass, `lora_per_layer_grad_norms()` computes per-layer gradient norms. Layers below 0.01% of max are added to `NANOBOT_SKIP_LAYERS`. The forward pass skips them entirely (identity pass-through). The backward pass skips them too.

At 35B, 22 of 40 layers are dead. Skipping them saves ~530ms: fwd goes from 1500ms to 1044ms, bwd from 200ms to 103ms.

**Key insight:** This is model-specific. Qwen3.5-35B uses MoE with 3B active params. The first 22 layers (all GDN-type) have near-zero gradient signal for LoRA on Wo+W2. The final 18 layers (10 MHA + 8 GDN) carry all the learning. A different model or different LoRA targets would have a different dead layer set.

## Combined Performance

| Configuration | Step (ms) | tok/sec | Loss |
|---|---|---|---|
| Session start baseline | 2443 | 52 | 10.9 |
| + Parallel GDN | 2037 | 60 | 10.9 |
| + Per-layer BWD kernels | 1680 | 74 | 10.9 |
| + 22-layer adaptive drop | 1148 | 109 | 10.3 |

Loss *improves* with layer drop (10.9 -> 10.3). Removing dead layers removes noise from their near-zero gradients.

## Hard-Won Constraints (Don't Re-Learn These)

### ANE Compiler Limits
- **16 BLOBFILE-weighted ops per program.** The 17th causes silent rejection.
- **~32MB total BLOBFILE weight per program.** Exceeding it yields CompilationFailure.
- **~119 loaded program limit** per process. After that, new compiles fail.
- **No fp32 mid-graph.** fp32 matmul, sigmoid, or cast mid-pipeline = CompilationFailure. Cast fp32->fp16 at input, do everything in fp16, cast back at output.
- **No rsqrt.** Use `pow(x, -0.5)`.
- **No reduce_sum.** Use `reduce_mean(x, axis) * dim_size`.
- **BLOBFILE matmul shape:** Always reshape to `[1,1,M,K]` for the weight. This is the universal pattern that ANE accepts.

### Bug 12: tmpDir Deletion
`_ANEModel`'s `-unloadWithQoS:` unconditionally deletes its tmpDir. This means:
- Delta reload (unload -> patch BLOBFILE on disk -> reload) costs O(weight_size) in disk I/O because the tmpDir must be recreated each time.
- BLOBFILE hotswap is only viable for blobs under ~20MB at training rates.
- The classifier (128MB per tile) cannot use delta reload. Pre-compiled per-tile kernels or CPU BLAS are the only viable paths.

### Classifier Bottleneck: Compute, Not Bandwidth
The classifier does embed[vocab, dim] @ x[dim, seq]. At 35B: vocab=151936, dim=2048, seq=128. Tiled into 5 chunks of 32768 rows.

Each tile GEMM is M=32768, N=128, K=2048. This is tall-skinny and the AMX achieves only ~10% utilization (294 GFLOP/s vs ~3 TFLOP/s theoretical). The bottleneck is compute efficiency, not memory bandwidth.

Results of every approach tried:
- ANE multi-kernel: 573-978ms (high variance from slot contention with transformer kernels)
- fp16 CPU: ~690ms (NEON fp16->fp32 conversion overhead negates bandwidth savings)
- fp32 CPU cblas_sgemm: ~665ms (most stable, no contention)

**Decision: CPU BLAS wins.** The classifier is 585-665ms and that's the floor without a fundamentally different GEMM implementation (tiled AMX intrinsics, or waiting for Apple to ship cblas_hgemm).

### DynMatmul vs BLOBFILE
Two ways to get weight matrices onto ANE:

**DynMatmul:** Pack weights into the IOSurface input alongside activations. MIL slices them apart, does the matmul. Pro: one kernel handles any layer. Con: copies the full weight matrix every eval call.

**BLOBFILE:** Bake weights into the compiled program. MIL references them as `BLOBFILE(path=...)`. Pro: zero-copy at eval time. Con: one compiled kernel per layer (N layers = N ANE programs loaded).

At 35B, DynMatmul copies 64MB/layer/step for MHA backward. BLOBFILE eliminates this entirely. The tradeoff is ANE program slots (~119 limit), but 10 MHA layers x 2 kernels = 20 slots, well within budget.

## Architecture Overview

```
Training Step Pipeline:
  Forward (1044ms with 22-layer drop):
    for each active layer:
      MHA layer: fused attn GQA kernel (ANE) + Wo (ANE) + fused FFN (ANE)
      GDN layer: pre-recurrence (ANE) + parallel recurrence (CPU, rayon) + output proj (CPU)
    Classifier: tiled cblas_sgemm (CPU)
    Cross-entropy loss (CPU)

  Backward (103ms with 22-layer drop):
    dlogits -> classifier backward (CPU)
    for each active layer (reverse):
      MHA: per-layer Wot (ANE BLOBFILE) + fused SDPA bwd (ANE) + per-layer QKV (ANE BLOBFILE)
      GDN: CPU backward (sequential, gradient norms near-zero anyway)
      RMSNorm bwd: per-layer kernel (ANE) or CPU fallback
      FFN bwd: per-layer W2t + W13t kernels (ANE BLOBFILE)
      LoRA weight grads (ANE)

  Update (28ms):
    AdamW on LoRA parameters (CPU)
```

## Key Files

| File | What |
|---|---|
| `src/agent/ane_forward.rs` | Forward pass. Fused layer fast path, 3-dispatch path, GDN recurrence. |
| `src/agent/ane_backward.rs` | Backward pass. 4-tier attention backward (fused -> BLOBFILE -> DynMatmul -> CPU). |
| `src/agent/ane_mil.rs` | MIL program generators. All ANE kernel definitions live here. |
| `src/agent/ane_weights.rs` | `PrePackedWeights` struct. Per-layer kernel priming, eval, and weight management. |
| `src/agent/ane_bridge.rs` | Rust FFI to `ane_bridge.m`. `AneKernel` struct wrapping `_ANEModel`. |
| `src/agent/ane_bridge.m` | Objective-C bridge to Apple private ANE frameworks. |
| `src/agent/ane_mlx_bridge.rs` | MLX model loading + 35B benchmark harness. |
| `src/agent/ane_lora.rs` | LoRA adapter implementation. Forward/backward on CPU, weight grads on ANE. |
| `memory/ane-compiler-bugs.md` | Bugs 1-12 with workarounds. Read before writing new MIL. |

## What's Next (Remaining Optimization Paths)

| Target | Mechanism | Estimated Gain | Complexity |
|---|---|---|---|
| Classifier fp16 BNNS | BNNSMatMul with fp16 weights, AMX native fp16 | -200 to -340ms | Low |
| More aggressive layer drop | 30-layer skip (all GDN), keep 10 MHA only | -130ms over 22-drop | Needs loss validation |
| GDN backward parallel | Same rayon pattern as forward recurrence | -30ms (small, most GDN skipped) | Low |
| Fused RMSNorm into FFN fwd | Merge pre-FFN norm into fused FFN kernel | -15ms (10 layers x 1.5ms) | Medium |

The honest read: at 1148ms/109 tok/sec with layer drop, the system is fast enough for real-time on-device personalization. The remaining wins are incremental. The classifier at ~600ms is the largest single component but requires either Apple shipping fp16 BLAS or custom AMX intrinsics to move significantly.

## How to Run

```bash
# Build
cargo build --features ane,mlx --release

# Unit tests (serial, ANE hardware contention)
cargo test --features ane --lib -- "ane_" --test-threads=1

# 35B benchmark (requires Qwen3.5-35B-A3B via MLX)
cargo test --features ane,mlx --release --lib -- "bench_35b_ane_per_step" --nocapture --test-threads=1 --ignored

# With layer drop
NANOBOT_SKIP_LAYERS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21" cargo test ...

# With backward profiling
NANOBOT_PROFILE_BWD=1 cargo test ...
```

## The Pattern That Worked

Every session that shipped real gains followed the same loop:

1. **Profile with real numbers.** Not estimates. Run the benchmark, read the timings.
2. **Identify the actual bottleneck.** The classifier was "obviously" the problem (675ms!). But the real win was in the backward pass's DynMatmul packing (64MB/layer of redundant memcpy).
3. **Try the simplest thing first.** Per-layer BLOBFILE kernels are conceptually simple: bake weights at compile time, skip the packing at eval. The MIL generators are 40 lines each.
4. **Accept what the hardware tells you.** The ANE has hard SRAM limits. The monolithic fused backward kernel doesn't fit. Two smaller kernels do. Ship the thing that compiles, not the thing that's theoretically optimal.
5. **Document the dead ends.** Bug 12 (tmpDir deletion), classifier compute bottleneck, fp16 conversion overhead. Future sessions don't re-run these experiments.
