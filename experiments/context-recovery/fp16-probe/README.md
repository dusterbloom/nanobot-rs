# Native Escha FP16 KV and attention feasibility probes

Date: 2026-09-05. Host: Apple M4, 32 GB. Model:
`Qwen3.6-35B-A3B-Escha-W2` in native Escha mode.

## Candidate identity and scope

- Worktree: `/private/tmp/higgs-fp16-probe`
- Base commit: `890039d7f294d4a861f381bff03f4ab0396d7ed9`
- Patch: `fp16-kv-probe.patch`
- Patch SHA-256: `eb9ea6166cdc47df183db73cb4821bbeac35dc25001485c7b655c903dfe933c4`
- Benchmark binary: `bench_frontier-fp16`
- Binary SHA-256: `547131b279c8dc348f2a6ffb32a7b8eb72efa4b2921657eab1fe7b78c7429664`
- Diagnostic flag: `HIGGS_FP16_KV_PROBE=1`; absent or any other value leaves the baseline path unchanged.
- Only newly-created `SteppingKeyValueCache` dense K/V allocations become FP16.
  Restored caches retain their serialized dtype. The native packed W2 weights
  are unchanged; surrounding activations and GDN/recurrent state remain FP32.
- This candidate is not installed and is not intended to ship as a flag.

GitNexus rated `SteppingKeyValueCache::update_dense` HIGH risk: 8 upstream
symbols, one direct caller, and three cache-test process groups. Whole-process
discovery was truncated, so all textual `update_and_view` consumers were also
audited. The patch remains an isolated feasibility probe.

The installed and preserved recovery server remained unchanged:
`e7a811f1dec325741d4a5d69fa91fcdd93a9f9b0d4040a325f7056d72feec7c7`.

## MLX mixed-dtype behavior

The MLX revision in `Cargo.lock` is
`b46423d2447d3db354c134a0ef25ff55dfdfe8b6`. A direct GQA probe accepted FP32
Q with FP16 K/V and returned FP32. The local MLX C++ source explains the cost:
`mlx/fast.cpp` computes `result_type(queries, keys, values)` and applies
`astype` to all three inputs. With native Escha FP32 Q, it materializes FP32
K/V inside SDPA. Thus this patch halves retained dense K/V but does not keep
attention's transient K/V in FP16.

## Matched 1K/2K benchmark

All three matched rows used the same candidate binary. Order was ON, OFF, ON
to bracket time drift. Each run used one sweep, a 1024-token chunk, and a
32-token greedy decode probe:

```bash
HIGGS_ESCHA_NATIVE=1 HIGGS_FP16_KV_PROBE={0|1} /usr/bin/time -l \
  /private/tmp/higgs-fp16-artifacts/bench_frontier-fp16 \
  --model-dir /Users/peppi/.cache/lm-studio/models/EschaLabs/Qwen3.6-35B-A3B-Escha-W2 \
  --frontiers 1024,2048 --probe-tokens 32 --runs 1 \
  --prefill-chunk-size 1024 --format json
```

| Mode | Frontier | Incremental prefill ms | Decode tok/s | Retained bytes | Greedy FNV-1a64 |
|---|---:|---:|---:|---:|---:|
| FP16 ON #1 | 1024 | 9510.939 | 14.706 | 86,835,200 | 12792644362472167692 |
| FP16 OFF | 1024 | 7899.096 | 17.570 | 107,806,720 | 12792644362472167692 |
| FP16 ON #2 | 1024 | 10044.117 | 14.982 | 86,835,200 | 12792644362472167692 |
| FP16 ON #1 | 2048 | 7895.744 | 15.544 | 107,806,720 | 7617371700643931454 |
| FP16 OFF | 2048 | 6231.341 | 18.992 | 149,749,760 | 7617371700643931454 |
| FP16 ON #2 | 2048 | 7341.396 | 15.367 | 107,806,720 | 7617371700643931454 |

The dense attention KV slope fell exactly from 40,960 to 20,480 bytes/token.
The remaining 65,863,680 bytes are fixed recurrent/GDN cache state. Therefore:

- 1K total retained cache fell by 20,971,520 bytes (19.5%).
- 2K total retained cache fell by 41,943,040 bytes (28.0%).
- At 16K, the same layout projects from 702.8125 MiB to 382.8125 MiB total
  retained cache, assuming unchanged fixed state.
- The two FP16 runs averaged 23.8% slower 1K prefill and 22.3% slower 2K
  incremental prefill than the bracketed OFF run.
- Decode averaged 15.5% slower at 1K and 18.6% slower at 2K.
- End-to-end duration averaged 29.781 s versus 26.411 s, 12.8% slower.
- Peak process footprint averaged 19.444 GB versus 19.505 GB, only 0.31%
  lower. Maximum RSS was lower, but varied substantially between processes and
  is not strong evidence of peak allocation savings.

Raw files: `fp16-bench.{json,log}`, `matched-off-bench.{json,log}`, and
`repeat-on-bench.{json,log}`. A separate recovery-worktree baseline is in
`baseline-bench.{json,log}`; its byte counts and greedy digests match the
same-binary OFF control, but timing conclusions use only the ON/OFF/ON bracket.

## Focused correctness checks

Release test command:

```bash
HIGGS_FP16_KV_PROBE=1 CARGO_TARGET_DIR=/Users/peppi/Dev/higgs/target \
  cargo test -p higgs-models --release \
  cache::tests::fp16_probe_stores_and_returns_fp16 -- --exact --nocapture
```

It passed and verified internal and returned K/V are FP16, allocated bytes are
two bytes/element, append offset/shape are preserved, and stored values equal
the expected FP16 round trip. See `/private/tmp/higgs-fp16-final-test.log`.

The existing native fixture was then run twice from the same release test
binary: flag OFF saved full logits plus a 129-token greedy trajectory; flag ON
compared against both. Results:

- Maximum absolute logit drift: `0.002208`
- Reference maximum absolute logit: `9.884724`
- Relative max drift: `2.234e-4`
- All 129 greedy token IDs matched exactly.

See `/private/tmp/higgs-fp16-fixture-off.log` and
`/private/tmp/higgs-fp16-fixture-on.log`.

The frontier sweep appends the second 1024-token segment to the live first
frontier cache and performs decode plus rollback at each frontier. This is KV
append/continuation evidence, not a full HTTP retained-session certification.
The short fixture covers 128 autoregressive continuation steps. This bounded
probe does not establish long-context accuracy.

## KV-storage-only decision

FP16 dense KV storage is numerically feasible for this bounded native Escha
probe, and it halves the variable retained-cache bytes. It should not be the
release default in its current form: FP32 Q causes MLX to promote the whole K/V
view inside SDPA, eliminating meaningful peak-footprint improvement at 2K and
producing a repeatable throughput regression. A future experiment would need
FP16 Q/attention plus an FP32 output cast, which changes attention arithmetic
and requires broader quality validation.

## Follow-up: true FP16 attention boundary

The follow-up candidate adds `HIGGS_FP16_ATTN_PROBE=1` at the two dense SDPA
entry points used by native single-request Qwen3Next attention. It requires the
KV-storage probe, casts Q to FP16, calls SDPA with FP16 Q/K/V, verifies that the
logical SDPA result is FP16, and restores that smaller result to the original
FP32 activation dtype before the attention gate and output projection. Packed
W2 weights and GDN/recurrent state are unchanged.

MLX revision `b46423d2447d3db354c134a0ef25ff55dfdfe8b6` computes a common
`result_type` and applies `astype` to Q/K/V inside `mlx/fast.cpp`. Supplying all
three operands as FP16 therefore removes the KV-only experiment's internal
FP32 dtype promotion. This does not prove that the implementation creates no
other transient buffers.

GitNexus could not resolve either existing entry point (`attend_one_query` or
`forward_scheduled`) and returned risk `UNKNOWN`. A text audit confirmed the
former serves canonical one-row/decode calls and the latter serves the native
single-request multi-token path. This bounded patch does not modify or validate
DenseMTP attention or the batched-engine attention bypass.

Candidate identity:

- Combined patch: `fp16-attn-probe.patch`
- Combined patch SHA-256: `01b3e7411b4ea997387008e765e7cc495c9582697f8273739c00cf1acb5262f9`
- Diagnostic flags: `HIGGS_FP16_KV_PROBE=1 HIGGS_FP16_ATTN_PROBE=1`
- Flag-off reference: both values `0`, using the same executable
- Benchmark binary: `bench_frontier-fp16-attn`
- Binary SHA-256: `db8dfb78ba8382d16fc96b8a2430fc00f8967ff1cf21b1c5840504cc3e2db9c1`

The ON/OFF/ON command matched the earlier frontier configuration:

```bash
HIGGS_ESCHA_NATIVE=1 \
HIGGS_FP16_KV_PROBE={0|1} HIGGS_FP16_ATTN_PROBE={0|1} \
/usr/bin/time -l \
  /private/tmp/higgs-fp16-artifacts/bench_frontier-fp16-attn \
  --model-dir /Users/peppi/.cache/lm-studio/models/EschaLabs/Qwen3.6-35B-A3B-Escha-W2 \
  --frontiers 1024,2048 --probe-tokens 32 --runs 1 \
  --prefill-chunk-size 1024 --format json
```

| Mode | Frontier | Incremental prefill ms | Decode tok/s | Retained bytes | Peak footprint bytes | Greedy FNV-1a64 |
|---|---:|---:|---:|---:|---:|---:|
| FP16 attention ON #1 | 1024 | 14082.964 | 15.757 | 86,835,200 | 19,195,996,096 | 12792644362472167692 |
| FP16 attention OFF | 1024 | 11030.331 | 14.203 | 107,806,720 | 19,514,042,112 | 12792644362472167692 |
| FP16 attention ON #2 | 1024 | 11007.707 | 16.953 | 86,835,200 | 19,197,093,776 | 12792644362472167692 |
| FP16 attention ON #1 | 2048 | 7286.074 | 15.969 | 107,806,720 | 19,195,996,096 | 7617371700643931454 |
| FP16 attention OFF | 2048 | 8050.449 | 14.987 | 149,749,760 | 19,514,042,112 | 7617371700643931454 |
| FP16 attention ON #2 | 2048 | 7279.069 | 16.450 | 107,806,720 | 19,197,093,776 | 7617371700643931454 |

The first ON process paid a large first-use cost at the 1K frontier. The repeat
ON process was 0.2% faster than OFF at 1K prefill and 9.6% faster at the 2K
increment, while decode was 19.4% faster at 1K and 9.8% faster at 2K. Both ON
processes had about 1.62% less peak process footprint than OFF. These are single
runs in one ON/OFF/ON order, so warm state, kernel compilation, thermal drift,
and process order remain confounded. The numbers support a promising bounded
follow-up, not a production speed guarantee or a global optimum claim.

Compared with the earlier KV-storage-only bracket, true FP16 attention avoids
the full-cache FP32 dtype promotion and changes the observed direction of the
decode result. Cross-bracket timing is contextual because the controls were
run at different times; only each experiment's same-binary bracket is matched.

Focused release tests passed:

```bash
HIGGS_FP16_KV_PROBE=1 CARGO_TARGET_DIR=/Users/peppi/Dev/higgs/target \
  cargo test -p higgs-models --release \
  cache::tests::fp16_probe_stores_and_returns_fp16 -- --exact --nocapture

HIGGS_FP16_ATTN_PROBE=1 CARGO_TARGET_DIR=/Users/peppi/Dev/higgs/target \
  cargo test -p higgs-models --release \
  qwen3_next::tests::fp16_attention_probe_restores_activation_dtype \
  -- --exact --nocapture
```

The first test verifies half-sized retained storage, append/readback, offsets,
and shapes. The second exercises GQA shapes with FP32 Q and FP16 K/V, verifies
the probe's FP16 SDPA result contract, and verifies the returned activation is
FP32 with its original shape.

The real-model fixture used one 16-token prompt and 128 greedy continuation
steps, first with both flags OFF to save the reference and then with both flags
ON to compare:

- Maximum absolute logit drift: `0.003558`
- Reference maximum absolute logit: `9.884724`
- Relative maximum drift: `3.599e-4`
- Greedy continuation: all `129/129` token IDs matched exactly

The earlier KV-storage-only fixture measured `0.002208` maximum absolute drift
and `2.234e-4` relative drift, also with all 129 tokens matching. The follow-up
therefore remains behaviorally feasible at this short fixture while changing
attention arithmetic more than storage-only FP16. It is not long-context or
model-quality certification.

Follow-up raw files are `attn-{on1,off,on2}.{json,log}`,
`attn-off-logits.safetensors`, `attn-off-tokens.txt`, and the copied
`attn-{test-cache,test-dtype,fixture-off,fixture-on}.log` files.

## Overall decision

True FP16 Q/K/V attention is the first variant in this probe to show both the
expected retained-cache reduction and a promising decode improvement. The
evidence is too narrow to change the production default: it covers one Apple
M4, one native Escha W2 model, 1K/2K frontiers, one Simple single-request path,
and a short greedy fixture. DenseMTP, batching, longer contexts, broader prompts,
and repeated performance trials remain unvalidated. Both switches remain
diagnostic-only candidate code and no runtime binary was installed.

Final `git diff --check` and `cargo fmt --all -- --check` passed. GitNexus
post-change analysis reported 2 files, 7 indexed symbols, 12 affected process
flows, and HIGH risk; it still did not resolve the Qwen3Next helper additions.
The installed `/Users/peppi/.local/bin/higgs`, preserved
`/private/tmp/higgs-recovery/higgs-hardened`, baseline copy, and shared-target
`release/higgs` all retained SHA-256
`e7a811f1dec325741d4a5d69fa91fcdd93a9f9b0d4040a325f7056d72feec7c7`.
After the candidate tests, the unchanged recovery `cache.rs` and
`qwen3_next.rs` were touched so the next shared-target build must rebuild the
baseline sources rather than reuse the candidate model library.

Repository evidence: this directory preserves reports, patches, token IDs and raw JSON/logs. Full binaries and logits are also copied to the git-ignored sibling `../endurance-fp16-artifacts/`; their original `/private/tmp` paths above describe the actual commands run.
