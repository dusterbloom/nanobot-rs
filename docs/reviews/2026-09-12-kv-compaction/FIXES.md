# KV-cache and compaction fixes — September 12, 2026

Implementation follows the six findings in [REVIEW.md](REVIEW.md). Changes target nanobot `main` and Higgs `nightly`; preexisting dirty work is preserved. The six original fixes and the subsequently approved fixed-context simplification passed release verification. Live inference, disk restart reuse, and durable recovery checks also passed on battery with explicit user authorization.

## Changes

1. **Recoverable prefix retention.** LCM may release a previously pinned prefix when the active conversation exceeds available room. Normal fitting compaction still preserves the optimization. Both committed and pending pins participate in rollback. Durable SQLite history remains authoritative.
2. **Complete request budgeting.** The compaction caller reserves leading system/developer messages, ephemeral tail messages, and the frozen tool definitions before compacting. The blocking recovery decision checks the complete wire estimate. This closes the case where conversation history fits but the actual request remains oversized.
3. **Correct eager-release URL.** Nanobot reuses its existing version-aware endpoint helper; root and `/v1` API bases all emit `/v1/sessions/drop`.
4. **Nonblocking eager reclamation.** Higgs returns false when generation owns the session mutex. Nanobot retains the pending cleanup for a later request. Ordinary explicit reset keeps its existing wait behavior.
5. **Reconstructible hybrid snapshots.** Disk format v3 records recurrent shapes and dtypes and stores Float32 SSM values without f16 rounding. Old formats are invalidated for a cold prefill. Attention KV keeps its preexisting disk representation.
6. **Active disk configuration and byte bounds.** `kv_disk_dir` enables the existing prefix cache, with model-path-specific filenames and `kv_disk_space_mb` converted to a checked whole-file ceiling. Legacy explicit cache paths retain their behavior. The existing append log discards older entries when the next fitting snapshot would cross the ceiling; an individually oversized snapshot is rejected without deleting the usable cache. Doctor, startup validation, init template, and README share the corrected configuration contract.

## Verification

Focused failing-before/passing-after regressions cover pin contraction, rollback, complete-wire overhead, actual HTTP request targets, session mutex contention, recurrent geometry/precision through disk reopen, and obsolete format invalidation. All 22 Higgs disk-cache tests passed, including whole-file byte bounds and Float32/Float16/Bfloat16 recurrent restart cases. Nanobot release build passed; its full library suite passed 3,010 tests with 27 ignored. All 40 Nanobot LCM/protocol integration tests also passed. Higgs release build and full server tests passed: 811 library, 8 binary, and 107 integration tests (926 total; 10 ignored). Final targeted rechecks after all lint cleanup passed: 22 disk tests, 6 config/doctor tests, and the release build.

Independent source review found no additional actionable issue in the Nanobot compaction/URL changes and the Higgs eager-release path. GitNexus was refreshed, but cache symbols remain outside its graph coverage; source and caller inspection supplement the graph.

Formatting checks in both repositories report differences already present at HEAD. Those unrelated formatting changes are intentionally left untouched. Required `cargo clippy --release -p higgs` failed in the unchanged `higgs-models` dependency with 128 errors and 206 warnings; it is not a clean lint result. Direct-crate checks also fail on existing lint debt: Higgs 79 errors/137 warnings and engine 148 errors/183 warnings. New-hunk diagnostics were corrected and checked again; no clean whole-crate lint claim is made.

## Fixed-context follow-up

The user approved removing the adaptive memory policy after the original live attempt received HTTP 503 under macOS warning pressure. That refusal happened before inference and did not demonstrate an actual allocation failure.

Higgs now checks the complete tokenized prompt plus requested output against `min(max_context_tokens, architectural limit)`, with a default context of 32,768. Explicit output requests remain unchanged; an omitted output budget uses the configured default bounded by remaining context. Learned profiles, predictive byte admission, pressure cancellation, and adaptive cache redistribution are removed. Static retained/prefix/disk cache bounds, worker ownership, unload cancellation, and actual error cleanup remain.

Nanobot consumes the fixed server context and compacts against that limit. Old learned or unavailable snapshots do not shrink its configured budget. Pressure parking, persisted retry writers, pollers, and nested capacity retries are removed. Historical SQLite suspension events remain readable, and existing user databases are untouched. An optional retained-history expansion that does not fit is discarded in favor of compacted history.

Final verification passed 2,996 Nanobot unit tests plus 40 integration tests, and 681 Higgs library tests plus 8 binary and 107 integration tests: 3,832 passed in total. Both release builds passed. The prior 22 engine disk-cache regressions remain applicable; this follow-up did not change engine cache code. GitNexus reports critical scope across both dirty repositories, consistent with the shared agent and request paths; graph line drift and missing engine symbols require direct caller inspection as well. See [LIVE-VALIDATION.md](LIVE-VALIDATION.md) for live results as they become available.

Live verification is complete: exact answer before/after restart with 1,472 reused prompt tokens; checkpoint/reset and durable LCM exact-recall artifacts both passed. The new three-turn smoke completed successfully. The old binary cannot consume the configured capacity response, so no matched speedup is claimed. The isolated test server has been stopped. See [LIVE-VALIDATION.md](LIVE-VALIDATION.md) and [LIVE-RESULTS.json](LIVE-RESULTS.json).
