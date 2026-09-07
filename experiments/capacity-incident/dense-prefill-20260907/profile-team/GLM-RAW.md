Read both files. The two specializations differ in-loop only at steel_attention.h:310-330 (causal block) versus :333-374 (bool-mask block); everything else — including `Qtiles[TD]` loaded once at :220-223 and live across the whole KV loop — is identical source, and static resources match your measurement.

**Discriminator: peel the causal tail.** Split the KV loop at the (loop-invariant, compile-time-resolvable) boundary:

```cpp
constexpr int kCausalBlocks = (BQ + BK - 1) / BK + int(!align_K);
int kb_full = kb_lim;
if (do_causal) kb_full = max(kb_lim - kCausalBlocks, 0);

for (int kb = 0; kb < kb_full; kb++) { /* body minus causal block */ }
for (int kb = kb_full; kb < kb_lim; kb++) { /* identical body with causal block */ }
```

**Why it is semantics-preserving:**
- The peel point is exactly the existing guard `kb >= (kb_lim - ((BQ+BK-1)/BK) - int(!align_K))`, so every `(row, col)` gets masked in the same iteration and the same order as before; masking math, softmax recurrence, and accumulation order per element are untouched. Output is bit-identical by construction, not by tolerance.
- No new shared-memory traffic: `Qtiles[TD]` stays register-resident in both loops. Q is never re-read from `shared_qkv`, so the K/V-over-Q aliasing hazard (Qs == Ks == Vs, :127-130) is not reintroduced.
- Loop bounds depend only on `kb_lim`, `BQ`, `BK`, `align_K` — no runtime behavior change for either specialization.

**Decision rule (one run each, native-causal vs bool-mask, same shapes):**
- **Gap closes (~1x):** the cause is code-shape/live-range interaction — the causal block's presence in the hot iteration body (taken on ≤ `kCausalBlocks` of ~NKiters) perturbs scheduling/allocation around the long-lived `Qtiles[TD]` and `Otile` accumulator chain in the `has_mask=false` specialization. The peel is then also the fix; keep it.
- **Gap persists (~2.5x):** the cause is function-constant specialization itself — the `do_causal=true, has_mask=false` instantiation generates structurally worse code regardless of body shape, and the next lever is inspecting the `.air`/metallib for the two specializations, not more source surgery.

It discriminates exactly your two hypotheses because it is the minimal edit that changes in-loop code shape *without* changing which function constants are set — a pure body-shape perturbation, so only one hypothesis can survive the measurement.

One caveat, in scope of the source: note native-causal also truncates `kb_lim` (:248-252) while a full bool-mask causal pass does not, so the two configs execute very different per-threadgroup trip counts. If the peel leaves the gap intact, re-run with the bool mask config's `kb_lim` forced (loop-range probe) before concluding specialization — it is the same zero-cost edit and removes that confound from the decision.