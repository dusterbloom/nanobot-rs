## Verdict

This run is **position-limited, not config-limited**: full-prompt average will land near **~98 tok/s** (projected ~460 s for 45,003), with instantaneous throughput falling from **~148 tok/s peak (2–4k)** to **~75 tok/s at 39k** — roughly a **2× decay** over the run. The published card's 264 tok/s (M4 base, 512-tok, short prompt) is **not comparable** to this snapshot; no gap or speedup claim is supportable yet.

## Measured numbers (snapshot to 38,912 tok)

| Position (tok) | Cumulative tok/s | Interval tok/s |
|---|---|---|
| 1,024 | 105.0 | 105.0 (incl. startup) |
| 4,096 | 128.1 | 146.1 |
| 8,192 | 132.8 ← peak | 133.4 |
| 16,384 | 127.2 | 115.3 |
| 24,576 | 118.4 | 97.9 |
| 32,768 | 109.5 | 84.0 |
| 38,912 | 103.1 | 75.1 |
| **45,003 (proj.)** | **~98** | ~72 |

Shape: ~1.5 s startup, ramp to peak by ~4k, then smooth monotonic decline; interval time grows ~7.0 s → ~13.6 s per 1,024-tok chunk, i.e., roughly 2× for 19× context growth — a strongly positive but sublinear position term.

## Why it slows (candidates, not conclusions)

Per-chunk work is constant (1,024 new tokens); only context length changes, and interval cost tracks position almost linearly after warm-up. Consistent candidates: (a) per-token attention/KV cost growing with sequence length, (b) KV working set exceeding cache tiers and shifting to bandwidth-bound reads, (c) a mix. No cliff or step anywhere in the curve — this argues **against** a discrete swap/spill event and for a continuous length-dependent cost, but the snapshot alone cannot split (a) from (b).

## Which comparisons are valid

| Comparison | Valid? | Why |
|---|---|---|
| Card 264 tok/s vs this run | **No** | Differs in chunk size (512 vs 1024), prompt length (short vs 45k), position mix — directionally meaningless both ways |
| Earlier 171 s "summary" test | **No** | ~3.4k is estimated source tokens, not model prompt tokens; unusable as throughput (~20 tok/s if taken literally, which it shouldn't be) |
| Internal: early vs late intervals of this run | **Yes** | Same hardware, config, prompt — isolates position-dependence |
| Projected ~98 avg vs card 264 | **No** | Unmatched hardware/prompt; would be an invalid ~2.7× shortfall claim |

## Smallest next matched benchmark

Two runs, same build, same M4 base, same tokenizer/content:

1. **512-token prompt, chunk512** — one number, directly comparable to the card's 264 tok/s. Confirms or refutes the published figure under identical conditions.
2. **Same 45k prompt, chunk1024** (this run, completed) — already yields per-position interval tok/s for free.

The pair gives the position slope at matched config; comparing run 1's number against this run's first-1k intervals localizes the gap to prompt length vs runtime config. If the slope needs a mechanism split (attention growth vs bandwidth), the next step would be a KV-size sweep at fixed chunk — but that is run 3, not needed to quantify the gap.