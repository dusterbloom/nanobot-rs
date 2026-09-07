# Adversarial review — FP32 D256 fused SDPA attempts (base M4, Q16/KV2, H16)

## 1. Strongest source-proven findings

**Correctness: explicitly none.** I found no missed correctness bug: dispatch is consistent, mask/stride semantics agree, tail shapes pass, synchronization is sound (§4). What the evidence does prove:

### 1.1 The benchmark cannot support a "~6% slower" conclusion (flaw, proven)
probe.cpp:28-31 runs arms {0,1,2,2,1,0}, discards one rep, times 3, reports a median; ADVERSARIAL-REVIEW.md:5 averages two such medians. Control (arm0) drift exceeds the claimed effect:
- tile16x8/probe-1024-32768.jsonl:3,8 — arm0 427.289→374.052 ms (−12.5%) within one run.
- probe-1024-45056.jsonl:3 vs tile16x8/probe-1024-45056.jsonl:3,8 — arm0 = 804.969 / 526.788 / 605.634 ms across sessions (+15% intra-run drift at K45056).
- arm0 at K32768 spans 358.629 ms (register-1024-32768.jsonl:3) to 427.289 ms — 19% session spread.
- One bracket is provably contaminated: tile16x8/probe-32-1024.jsonl:3 arm0 = 2.202 ms vs 0.747 ms (line 8) — a 3× artifact that survived the single discarded rep.

The candidate loses 11 of 12 bracket pairs, so the rejection is directionally robust — but its margin is smaller than every noise term above.

### 1.2 Existing biases favor the candidate; they cannot explain the loss
probe.cpp:10 sets a 256 MB allocator cache; the fallback's 2 GiB score buffer (arms 0/1) therefore evicts and is re-allocated on every timed eval, while arm2 allocates only a 16 MiB output that stays cached. The fallback arms also rebuild multi-node decomposed graphs per rep; arm2 builds a 4-node graph. Both inflate arms 0/1. The fused candidate still lost — the rejection is conservative, and "the fused kernel was unfairly timed" is not supported.

### 1.3 Register-Q's 2× loss is designed-in (source-proven arithmetic; no counters exist)
- register-q/mlx/.../steel_attention.h:188 — `MMATile Qtiles[TD]` (TD=32) pins 64 extra float registers/thread on top of Otile's 64, i.e. ~140+ live registers × 128 threads at wm4; at 31/1055 it is competitive (register-31-1055.jsonl:5-6), confirming the loss at long K is occupancy/pressure, not semantics.
- steel_attention.h:111 — `padQ = BD==256 ? 0 : 16/sizeof(T)` gives LDQ_tgp=256; a 256-float row stride lands every row on the same smem banks → conflicts in the register-load fanout (lines 219-224).
- steel_attention.h:125-127 — aliased buffer = max(32×256, 20×256)×4 = 32768 B, exactly the device's 32 KiB threadgroup budget → minimal residency.

### 1.4 One untried point in the tile space, adjacent to the best attempt
Best fused = BQ16×BK8×WM2: 438.151/410.953 ms vs fallback 427.289/374.052 (K32768). BQ16×BK16×WM2 was never tried. With register-q's aliasing trick (single shared buffer, padQ=0) smem = max(16×256, 20×256)×4 = 20.5 KiB < 32 KiB, ~80 regs/thread at 64 threads, and per-key barrier overhead halves (5 threadgroup barriers per iteration — upstream-steel_attention.h:247,257,367,418,427 — amortized over 16 keys instead of 8). No source obstacle.

### 1.5 Measured flaw in the shipped path itself
At the identical shape (1024×32768), explicit sliced-mask 128-blocking (strided.cpp:26,30; tile −128) = 288.808/296.456 ms beats the shipped causal mode with KV prefix slicing (banked-memory.patch:36-44; strided tile +128) = 317.669/321.952 ms by ~8% — while doing slightly *more* key work (262,144 vs 258,560 block-key slots). The causal branch's per-block mask synthesis in the decomposed path costs more than the prefix slice saves. Caveat: the decomposed composition (fast.cpp) is not in the bundle, so mechanism is inferred; the delta is measured.

## 2. Top three minimal experiments

**E1 — BQ16×BK16×WM2 aliased kernel (the untried config).**
Change: copy the tile16x8/ set; d256.metal → `instantiate_attn(float32, float, 16, 16, 256, 2, 1, float32, float)` (+bool_ line); in the header adopt register-q's aliased single smem buffer + padQ=0 but keep upstream's per-iteration Qtile smem loads (drop `Qtiles[TD]`).
Shapes: 1024×{8192, 32768, 45056}, 31×1055, 32×1024.
Controls: add a 4th probe arm = shipped +128 query blocking; 6 alternating brackets.
Gates: numerics max_abs ≤ 1e-5, rel L2 ≤ 1e-5 (probe.cpp:26); candidate beats arm0 in ≥5/6 bracket pairs and lands ≤302 ms at K32768 (≥5% under the 318 ms shipped reference).
Falsification: still ≥5% behind arm0, or fails to close half the gap to shipped → declare flash-at-D256-FP32 dead in this design; stop sweeping tiles.

**E2 — Explicit sliced-mask blocking on the shipped path (no kernel work).**
Change: in `dense_prefill_attention` (banked-memory.patch:36-44), build one `create_causal_mask(query_len, Some(key_len−query_len))` and per block pass `AttentionMask::Array(mask.index((start..end, ..)))` with full-length keys/values — mirroring strided tile −128.
Gates: existing patch tests (lines 124-202) for numerics and <384 MiB scratch (lines 145-149; a 1024×45056 bool mask is 45 MB, safe); synthetic: ≥5% under +128 across ≥3 brackets; then one matched 45K serving pair with TTFT improvement ≥1%.
Falsification: advantage vanishes when the mask is built in the Rust path, or TTFT within paired-run noise → revert the mask branch.

**E3 — GQA-reload sensitivity probe (decides whether any structural rescue exists).**
Change (probe-only): add an arm passing K/V as {1,16,nk,256} (gqa_factor=1) — identical FLOPs, 8× K/V device traffic.
Gate: if arm2(gqa=1) is within ±5% of arm2(gqa=8) at K32768, K/V reload is not binding → GQA-reuse/split-K redesigns are pointless; fused stays rejected. If ≥15% slower, a multi-head threadgroup sharing K/V smem could transform the economics — but that is a rewrite, not one inch.

## 3. One-inch-away assessment
Against the old fallback at K≥32K: plausibly one inch — E1 could reach parity (the fallback itself degrades to 805 ms in one session). Against the shipped baseline: not one inch. Shipped +128 measures 317.7/321.9 ms at K32768 and 462.8/463.7 ms at K45056 (tiled-1024-45056.jsonl:7,12) vs best fused 411-438 / 580-619 ms: fused needs 25-29%, and no in-evidence mechanism that large is a small change (double-buffered K/V pipelining or GQA multi-head threadgroups are restructures). E2's ~8% on the shipped path is the cheapest real win.

## 4. Verification checklist from the task (all clean)
- Dispatch: attention-probe.cpp:196-199 ↔ d256.metal:2 (BQ8/BK16/WM1); tile16x8/attention-probe.cpp:196-199 ↔ tile16x8/d256.metal:2 (16/8/WM2); attention-register.cpp:196-199 ↔ register-q.metal:3 (32/16/WM4). Env gate (attention-probe.cpp:624-632) correct; `-DMLX_METAL_NO_NAX` + `is_nax_available` guard (line 178) route correctly.
- Compiler: probe metal flags (build*.py:6) match the upstream default `-fno-fast-math` (upstream-CMakeLists.txt:18) — no unfair flag.
- Mask/stride equivalence: candidate causal vs explicit offset-causal mask agrees to 6.5e-7 max_abs (probe-1024-32768.jsonl:1-2); tails 31/1055 and 32/1024 pass; strided.cpp:13's transposed-Q layout is handled (stride(-1)==1, Q_strides honored).
- Synchronization: barriers in both headers are correctly paired; register-q's Q→register handoff (barriers at 218/224) is safe.
- Timing/JIT: eval-synced wall clock is fine; one discarded rep is not (§1.1).
- Anomaly: probe-1024-45056.jsonl ends after arm0's first median (804.969) with empty run.stderr — the bq8 run never completed that shape; excluded from conclusions.

## 5. Whole-request bound (evidence-permitted)
Single-shot attention at the serving shape costs ~0.29-0.32 s (shipped) of a ~290 s TTFT; 45 model chunks imply ~10-14 s of attention prefill (~4-5% of TTFT). A zero-cost fused kernel is therefore bounded by ~5% TTFT; realistic E1 success nets <1% (~2-3 s) — at or below the 1.4 s (0.5%) paired delta already recorded (STATUS.md:9-18), consistent with "latency flat". E2 (~1 s, free of kernel risk) is the only change worth carrying into a production test.
