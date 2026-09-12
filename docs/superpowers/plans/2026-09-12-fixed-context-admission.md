# Fixed context admission implementation plan

**Goal:** Replace speculative memory admission with a fixed model context limit and ordinary request errors.
**Architecture:** Higgs advertises and enforces `prompt + requested output <= min(max_context_tokens, architecture)`. The model config defaults to32768 total tokens. Nanobot narrows its configured context only to this fixed server limit and compacts normally. OS pressure and allocation measurements are diagnostic; cache ownership and configured bounds remain enforced.
**Tech stack:** Existing Rust/MLX/Tokio/SQLite implementations; no new dependencies or policy modes.

The user explicitly approved this design with “please go for it.” No second design approval or commit is required. Preserve all earlier KV/LCM fixes and preexisting dirty work. AC power is already authorized for live tests.

## Ownership and steps

- [x] Server admission worker: replace capacity controller/registry decisions with fixed context checks, update chat/completions routes, remove runtime learning/pressure/reservation enforcement. Preserve model lifecycle and request cancellation for real user disconnect/watchdog events. Add red/green tests: warning/critical pressure does not reject a fitting request; full rendered prompt plus output exceeding fixed context is rejected even with cached prefix; both streaming and ordinary routes use the same check.
- [x] Startup/config worker: add validated model `max_context_tokens` default 32768 capped by architectural maximum, populate fixed facts, remove load predictions and learned profile initialization; preserve bounded static caches and actual model load failures. Update doctor, init template, and README.
- [x] Nanobot worker: consume configured server limits, delete pressure-driven shrinking/recovery/parking from provider and agent loop, keep actual errors and ordinary compaction. Test fixed limits and pressure-only/old adaptive snapshots against the real client flow.
- [x] Root: remove obsolete bus/session/REPL parking callers and misleading adaptive status text while preserving historical SQLite event readability. Verify engine bounds and request-local cache cleanup remain correct.
- [x] Run upstream GitNexus impact before modifying existing symbols and report HIGH/CRITICAL risk. Supplement missing/stale graph entries with direct callers; run detect-changes and direct diff checks before completion.
- [x] Observe targeted regressions fail before implementation; coordinate all release builds/tests in tmux. Run library/protocol/config/admission/cache regressions and independent diff review. Record inherited lint/format failures rather than broad unrelated changes.
- [x] With compilers stopped, run isolated live inference under actual OS pressure, hybrid disk restart, Nanobot durable recovery, and matched turn benchmark. Inspect artifacts, not only exit codes. Stop owned services afterward.

## Evidence and decisions

Prior-stage diffs are saved in `/tmp/kiss-capacity-20260912/{nanobot,higgs}-before.patch` so this simplification can be reviewed separately. Earlier release test results and live refusal are in the KV review folder.

Implementation checkpoint:
- Fixed server admission, static cache bounds, configured client discovery, and removal of persisted parking writers are implemented. Historical suspension events remain readable; existing databases are not migrated or erased.
- Targeted old-behavior tests failed before fixes: pressure rejected a fitting server request; the client parked an otherwise valid turn; LCM retried an actual summary error. Selected client capacity suite passed 62 tests afterward.
- Full release suites are running. Review identified two follow-ups: omitted output limits must fit remaining fixed context, and an oversized optional history expansion must fall back to compacted history. Neither explicit output requests nor genuinely oversized compacted prompts will be silently truncated.
- Live inference remains pending final binaries; earlier live 503 was an admission refusal, not evidence of a model allocation failure.

Final release verification:
- Nanobot: 2,996 unit + 40 integration tests passed; release build passed. Binary SHA-256 `f062a2f1d15fffdd6ac2b63e83c4112cc08fd4c006bcac720e809fd4150aa6ca`.
- Higgs: 681 library + 8 binary + 107 integration tests passed, 10 ignored; release build passed. Logs: `/tmp/kiss-capacity-20260912/server-final2-{test,build}.log`.
- Optional raw-history expansion fallback and omitted-output default both failed before correction and passed after. Test fixtures now explicitly seed required retained sessions; actual continuation errors remain enforced.
- No commits. Existing dirty work remains. Final diff whitespace checks passed.
- Pre-live check found no compiler/server process and port 9000 free, but `pmset -g batt` reported Battery Power at 39%. User notified to reconnect AC. Model has not been loaded for this new run.
- Saved old Nanobot does not understand the new configured capacity basis; matched benchmark may be protocol-incompatible. Verify actual old-run result before drawing any performance conclusion.

Completed live checks on battery after explicit user instruction. Exact inference and disk-restart reuse passed (1,472/1,479 cached prompt tokens); both durable recovery artifacts passed. New three-turn smoke passed, but old binary capacity-protocol incompatibility prevents a matched speed comparison. Owned server stopped. Structured evidence is in `docs/reviews/2026-09-12-kv-compaction/LIVE-RESULTS.json`.
