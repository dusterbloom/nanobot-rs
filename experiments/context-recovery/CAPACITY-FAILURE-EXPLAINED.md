# What the observed capacity failure establishes

The policy-discriminator runs ended in Higgs HTTP503 higgs_capacity_unavailable,
translated by nanobot into a durable pending capacity turn. They did not record
an allocator OOM, process crash, or physical free-memory measurement of zero.
The reports must not be read as proof that the machine exhausted its RAM.

For the decision run (boot241c86ca-bc1f-413b-8529-e16af3994e1f), recorded values:

| Quantity | Value |
|---|---:|
| Machine physical memory, earlier host measurement | 32 GiB |
| MLX configured memory authority at boot | 30.40 GiB |
| Metal recommended working-set authority at boot | 24.96 GiB |
| Governor normal usable budget | 19.97 GiB |
| Governor constrained usable budget | 17.47 GiB |
| Maximum sampled MLX active allocation | 12.02 GiB |
| Maximum reported MLX peak allocation | 14.69 GiB |
| Final effective pressure state | critical |
| Last typed failure | capacity unavailable, generation11, retry5000ms |

MLX allocation is not total process or system RAM usage. The usable budget is a
policy allocation budget, not an OS free-memory counter. On critical state,
Higgs publishes unavailable and can set its decision byte/token fields to zero.

Source trace in the exact hardened Higgs worktree:
- crates/higgs/src/capacity/pressure.rs:175-196: receives OS pressure callbacks,
  samples system-wide cumulative swapouts and compressions every second, then
  computes deltas. Compressions is activity, not net compressor-resident growth.
- pressure.rs:350-395: Darwin dispatch memory-pressure flags map to states.
- capacity.rs:1000-1067: any recent swap-out makes effective pressure critical,
  sticky for at least60 seconds until conditions permit recovery. Compression
  activity under nominally normal OS pressure becomes constrained.
- capacity.rs:690-714: use the smaller MLX/Metal authority, reserving20% under
  normal pressure or30% under constrained/critical, at least4GiB.
- capacity.rs:1136-1205: a constrained downshift takes75% of the previous token
  envelope; critical takes50% and disables availability. Normal transitions do
  not immediately restore the previous allowance. Separate recovery/evidence
  logic governs rises. Repeated new episodes can therefore reduce token capacity
  without a corresponding monotonic increase in model allocation.
- capacity/registry.rs:1314-1361: propagates the effective pressure to each model,
  records last deltas, requests active-reservation stops on critical, and recomputes
  admission. The exported pressure is effective policy state, not raw OS state.
- nanobot src/providers/openai_compat.rs:921 maps the503 into its typed error.

The decision-run total-token envelope fell51200→37888→27648→20480→10240→0.
The server stayed alive and advertised8192 tokens again at12:18:32UTC. The exact
failure occurred at12:17:41UTC. These are signs of protective admission and recovery,
not evidence of allocator exhaustion.

All69 five-second telemetry samples for this run show swapOutDelta=0, while some
show positive compressorDelta. This does NOT prove no swap happened: the exported
delta is only the latest observer interval, and polling every five seconds can
miss a one-second spike. Metrics also omit the raw OS pressure flag, so existing
artifacts cannot distinguish an OS-critical event from a recent-swap-induced
critical state. They cannot attribute system compression to Higgs versus another
process, or show total system available memory at that instant.

Correct conclusion: protective governor rejection triggered by its effective
critical-pressure state. Physical RAM exhaustion, the identity of the initiating
pressure source, and whether the rejection was overly conservative are unresolved.
To resolve them, a subsequent matched reproduction must timestamp raw OS pressure
events, cumulative VM counters/free and compressor pages, process resident/physical
footprint, MLX active/peak/cache bytes, and governor decisions at each transition.
Do not disable the safeguards or infer an OOM solely from the word capacity.

Follow-up source/history/session study: [HIGGS-ALLOCATION-AUDIT.md](HIGGS-ALLOCATION-AUDIT.md).
It identifies the test cache-byte cutoff, a runtime/estimated KV slope mismatch,
unconnected live learning, and that allocator metrics expose last-published
process-wide counters rather than a per-request trace.
