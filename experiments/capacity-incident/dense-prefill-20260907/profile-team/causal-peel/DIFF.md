# Causal-tail peel diagnostic

Baseline: `/private/tmp/higgs-fused-v2/reg` with BQ=16, BK=16, WM=2.

The copied attention header keeps Q staged once in threadgroup memory and held
in `Qtiles` registers. Only the KV loop changes:

- causal bulk: `kb = 0 .. kb_tail`, compiled with no causal-mask body;
- causal tail: `kb = kb_tail .. kb_lim`, compiled with the causal mask;
- noncausal: the original single `0 .. kb_lim` loop with no causal mask.

`kb_tail = max(kb_lim - ((BQ + BK - 1) / BK) - int(!align_K), 0)`.
The unguarded local include holds the KV block body once, so K/V loader,
barrier, accumulation, softmax, output-update, and loader-advance order remain
identical in every arm. `STEEL_APPLY_CAUSAL_MASK` is defined only around each
include and consumed by `if constexpr`.

Impact lookup returned `UNKNOWN`: the nanobot graph does not index the
external MLX symbols `sdpa_full_self_attention` or `attention`. Source search
confirmed the copied `attention` kernel is instantiated only by the two
BQ16/BK16/BD256/WM2 entries in `reg.metal`.

Risks: repeated in-function inclusion depends on Metal's normal C++
preprocessor behavior; compilation must confirm it accepts the unguarded body
include. Numerical tests must cover `kb_tail == 0`, aligned and partial K,
nonzero `qL_off`, and amplified QK values.
