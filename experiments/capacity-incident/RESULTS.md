# Long-context incident — verified recovery and limits

2026-09-06. The earlier one-request recovery claim was insufficient. The final installed pair passed **10 user turns / 18 model calls through39,315 rendered prompt tokens**: nine archive turns with exact recall, followed by real Hacker News/article browsing and the same exact recall.

## What actually broke

| Failure | Evidence | Correction |
|---|---|---|
| Conflicting client/server budgets | Nanobot advertised configured room and requested6144 output against a4096 server ceiling | Effective prompt limit, bounded output, preserved typed rejection ceiling and one-retry state, truthful footer/error |
| Synthetic pressure and unusable envelope | Compression activity promoted normal OS pressure; public available profile could contain zero prompt room | Compression only disqualifies upward learning; zero-prompt public profiles normalize to unavailable; swap/OS protections retained |
| Direct trellis memory leak |32 dropped calls retained2,167,424B; full first prompt exhausted Metal memory | Release MLX C input/output containers and config; free failed output extraction |
| Duplicate retained KV charge | Already-budgeted4096 KV tokens reduced published capacity by4096; live limit shrank after every turn | Charge only retained high-water beyond prompt/output/fixed session bytes already budgeted |
| Unused allocator buffers consumed RAM |23.918GiB physical with14.479GiB active and9.219GiB cached; default cache limit31129MiB | Existing wired-memory path now sets256MiB allocator-cache target automatically |
| Silent history loss | SQLite loader dropped batch1 at37977 heuristic tokens despite ample real context; recall took201s and falsely claimed key never supplied | Remove competing `max_messages * 150` token filter; explicit row/turn limits and agent TokenBudget/LCM remain |

## Final installed validation

Normal launch: `HIGGS_ENABLE_THINKING=1 higgs serve --mlx-profile throughput`. No diagnostic kernel override. Runtime confirms scratch_matmul,1024-token chunks,256MiB allocator-cache target. Retained sessions:49152tokens/2GiB; prefix cache128MiB. Allocator buffers and retained KV are different caches.

Artifacts: `long-context-default-final/`, server log `/private/tmp/higgs-live-final.log`.

- Nine turns passed exact answers/facts. Final prompt33507tokens; final recall reused33449tokens and prefetched only58 new tokens.
- Recall answer: LANTERN-4729-QZ, COPPER-8163-VX, Alder.10.05seconds instead of the failed201-second cold replay.
- No compaction, capacity rejection or pressure downshift. Normal pressure throughout; system swap-out counter remained3278240.
- Maximum5-second-sampled physical footprint18.102GiB; this is sampled, not an exact continuous peak. Allocator-cache samples peaked316.49MiB despite the256MiB target, so do not describe it as a strict byte ceiling.
- Prompt allowance remained47104tokens. This run proves33.5K, not the full advertised limit or arbitrary endurance.
- Incremental ~3.9K-token batches took25–40seconds under default scratch prefill. No claim that this is the fastest kernel or best hardware/model combination.

## Verification and provenance

- Nanobot release library suite2965passed,0failed,27ignored; release executable build passed. Four obsolete tests for the removed competing token policy were replaced by long-history and intact/orphaned tool-pair coverage.
- Higgs capacity/server release suite778passed. Escha release suite52passed, including leak regression and numerical parity. Release executable build passed; allocator default validated in the live sequence.
- Repository short-turn speed check: before11.170s/1.463s, after10.954s/1.463s. Only two turns per binary; not a statistical performance claim.
- Nanobot installed SHA256: `8eadd230c5a8532b56b3cae6f55ea6c5d6c3188794a9f68a17afe825f01924f3`.
- Higgs installed SHA256: `64f58982d4dc64293654f05bafba78915782942da99d9894a976e75a56da1259`.
- Engine build identity: `sha256:b570369276b008196ba035ee35dc857316275a35baaaf0d3d203e1787ed07cc9`.
- Prior binaries and config backed up under their `.before-*20260906` paths. No user session deleted; no unrelated application stopped.
- Architecture companions regenerated. GitNexus changes inspected, but its stale graph cannot establish all Higgs callers; UNKNOWN results were cross-checked in source. No clean-graph or completed external-review claim.

## Real browsing and final live state

Turn10 fetched Hacker News, autonomously selected Cloud in a Bottle, fetched the article, read its stored content, and produced a grounded two-sentence summary plus all three exact original archive facts. Recorded excerpts support the summary. Final prompt39315tokens; cumulative18main model calls, zero compaction. Final live metrics show zero capacity rejections, normal pressure, no active reservations, unchanged swap-out counter3278240 and47104prompt allowance. Higgs remains running in tmux `higgs-live-final`, listener/binary hashes saved in `final-live-state.json`; source branch is `nightly`.

This follow-up took359.46seconds, including about227seconds of cold replay caused by the test launcher using a different working directory and thus changing the system prompt. It is a successful cold-replay/tool-use correctness check, **not** a warm-turn latency measurement. Subsequent calls reused the prefix. Launcher corrected for future use. Cold replay of33K remains slow; the fix prevents needless replay but does not make it instantaneous.

## Remaining boundary

Sustained operation beyond39.3K and correctness after genuine context exhaustion/LCM compaction are not established by this run. Earlier LCM-versus-notes experiments remain separate. No public nightly release was published by this incident work.
