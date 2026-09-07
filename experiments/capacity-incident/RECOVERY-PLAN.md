# Pressure recovery and faithful compaction

User authorized parallel implementation and primary-agent quality review on 2026-09-06.

- [x] Higgs: restore normal-pressure usable context under current memory accounting; preserve critical-pressure protection.
- [x] Nanobot: automatic interactive retry with bounded backoff, cancellation and no duplicate completed tools.
- [x] LCM: model-authored, budget-fitting summaries for long/tool-heavy spans; no silent oldest-first clipping.
- [x] Review integration, failure paths, source coverage and regression tests.
- [x] Serialize release tests/builds; do not compete with live inference or launch the singleton CLI beside the user's session.
- [x] Record exact validation, deployment state and remaining limits.

Ownership: capacity_recovery owns Higgs capacity sources/tests; interactive_retry owns nanobot agent loop/TUI/session retry; faithful_compaction owns lcm.rs and compaction.rs. Primary agent owns integration and this record. Preserve unrelated dirty files. No commit or installation until reviewed and validated.

Evidence motivating changes: session 20260906_172536_478870 suspended at 17:47 UTC, pending retry count remained zero; level-3 summary covered 135 messages in 1,029 tokens, bypassing the model and dropping later findings. Prior 39.3K success did not prove pressure recovery or real summarization quality.

## Validation / review ledger

- Higgs valid RED: quiet ample-memory recovery returned 24,576 total instead of configured 49,152 total / 45,056 prompt. Original synthetic memory-bound fixture had an incorrect expected budget; corrected before accepting it.
- Main rejected an initial swap-only recovery patch that added a compression blocker. Final production change is one existing Normal branch returning the current static ledger decision; all swap/critical predicates remain unchanged.
- Higgs final: 140 capacity tests and 782 library tests passed; release binary built. Evidence `/private/tmp/higgs-capacity-final2.log` and `.exit` (0).
- Installed Higgs SHA256 `7c88eacfc8b991a4158d74def94215181f6dfd341c4140694024bdb687182342`, backup `~/.local/bin/higgs.before-recovery-20260906`; restarted in tmux `higgs-recovery-live` with original thinking flag and throughput profile. Live validation pending.
- Nanobot seven behavioral regressions confirmed RED against old code. Evidence `/private/tmp/nanobot-combined-red-direct.log`. First fixture import error was corrected; compilation failure was not counted as behavioral RED.
- Main review corrections: actual summary retries obey live total and separate prompt ceilings; prefix search avoids quadratic tokenization; newer pending work supersedes only after durable inbound persistence.
- Peer review uncovered gateway handoff/replay races and non-atomic pending replacement, plus ordinary retention trimming before/after failed LCM. Agents are fixing those integration paths before GREEN validation.
- Live user nanobot PID53654 remains untouched; do not launch singleton CLI benchmarks beside it. Builds/inference are serialized. Current compilation paused for installed Higgs live test.
- First live ~45K check used non-streaming and hit Tower's 300s deadline (504), not capacity rejection. Sampled footprint peaked ~16.7GiB; normal pressure/no new swap-outs. The server continued the owned generation after the client timeout, then released it. This is NOT a recall pass. Production streaming/progress is required for the rerun. Script `/private/tmp/recovery45k.py` now uses streaming/progress and ~45,000 native user tokens.
- Higgs is currently stopped again (after confirming idle) for final nanobot release compilation. Restart using `/private/tmp/higgs-recovery-launch.py` in tmux when heavy builds finish.
- Nanobot first GREEN targeted run: all nine retry/gateway/DB filters passed, eight of ten compaction filters passed. Two failures investigated: test tool-pair must lie within actually covered prefix (fixture corrected); more importantly `history_limit_lcm(8192)=38` discarded the oldest failure before compaction. Main rejected a 40→36-row fixture workaround. Approved removing guessed context/150 row cap via existing max_messages=0 LCM loads, retaining explicit max_history_turns and protocol filters. HIGH context hot-path warning given; graph UNKNOWN was textually corroborated.
- Peer review corrections now include durable gateway resume identity, stale wake suppression, canonical session/sender/voice, transactional pending replacement, and cleanup only after a durable terminal result. Failed pending writes report Session Error. Review found no remaining HIGH/MEDIUM in those corrected hunks; loader amendment still pending final verification.

## Final release verification

- Nanobot: 20 focused regression filters pass; full release library suite 2,975 passed, 0 failed, 27 ignored. Final release build passed. Logs `/private/tmp/nanobot-combined-green-{targeted,full,compile}.log` and `/private/tmp/nanobot-final-release-build.log`. SHA256 `e015ce897604d070580296eabde0dfb74d91f551b3654abd408eed0afa2cf1d6`.
- Higgs restarted with reviewed binary for isolated model-backed validation; no overlapping build. User nanobot PID 53654 remains untouched.
- Final dirty-tree graph analysis reports CRITICAL aggregate blast radius across 19 files, including unrelated existing edits; it is not an isolated change verdict or an all-clear. Narrow peer review and regression tests cover the edited paths.
- Pre-existing follow-up: `history_window_near` receives process-global learning count, never reset by compaction. Current explicit maxHistoryTurns=600 makes compaction attempts sticky after call 599; model work occurs only when eligible raw exceeds protected tail/minimum. No new silent clipping, but possible premature blocking/latency. Replace with session-local accounting in a separate verified correction.
- Matched CLI speed benchmark deferred because starting another agent CLI would terminate the user’s interactive singleton. No speed-regression claim.

## Live validation / activation

- Installed nanobot atomically at `~/.local/bin/nanobot`; verified SHA256 e015ce897604d070580296eabde0dfb74d91f551b3654abd408eed0afa2cf1d6. Backup `nanobot.before-recovery-20260906`. Existing interactive PID 53654 still uses its old executable until relaunched.
- Live failed_action A: actual model summary (finish_reason=stop), 171.34s preparation, 65.40s resumed turn, one durable summary. Correct operation EX-7041, null receipt, request_permission, zero forbidden actions. Strict evaluator FAIL: status was `Failed (ERROR permission denied)` rather than exact `failed`. The Rust test itself exits 0 while recording pass=false; do not count its exit as benchmark success. Artifacts `/private/tmp/nanobot-faithful-live-1788726642`, log `/private/tmp/nanobot-faithful-live.log`. This demonstrates semantic failure-state retention for one case, not full correctness or fast compaction.
- Streaming ~45K smoke launched alone against Higgs PID 69778; output `/private/tmp/higgs-recovery-45k-stream-result`, pending.
- Independent live review: one summary node covers IDs 1–17, rows 18–20 retained raw; one successful submit_result lifecycle, no export/retry. Exact status vocabulary was not enumerated in the original request, so strict equality is an output-conformance finding, not demonstrated semantic recovery failure. Summary contains two minor inaccuracies: attributes a suggested retry when source only forbids retry, and says six snippets while listing seven. Neither changed the operation state, but do not claim perfect summary fidelity.
- Streaming 45K check FAILED at ~303s: server SSE generation_error `Prefill cancelled by observer`. Last samples normal pressure, no new swap-outs, no admission rejection; cold cache=0. Streaming alone does not remove the actual execution deadline. Reopened Higgs investigation with capacity_recovery; do not claim 45K success. Artifacts preserved unchanged.

## Long-prefill watchdog follow-up

- Live stop outcome `no_progress_watchdog` identified actual cause. `GenerationStop` already tracks last progress; simple-engine chunked prefill omitted `note_progress`, while decode and batch prefill renew it. No change to timeout configuration or pressure guard is needed.
- Regression RED: progressing chunks beyond total watchdog age returned `Some(NoProgressWatchdog)`; `/private/tmp/higgs-watchdog-red.log`, exit101. Renewal GREEN: 3 focused tests pass, `/private/tmp/higgs-watchdog-green.log`, exit0. Stops are checked before renewal, retaining stalled/client-cancel behavior.
- Peer review identified pre-existing MLX wrapping of PrefillCancelled losing typed stop classification. Approved narrow exact `Exception::what()` sentinel mapping in existing simple-engine path; unrelated MLX failures remain unchanged. Final tests/build/live rerun pending.
- Final watchdog+typed cancellation review: exact MLX sentinel matches real model exception conversion; stored stop reason preserved, unrelated MLX error unchanged. Six focused release tests pass, `/private/tmp/higgs-watchdog-final.log`. Full higgs-engine library: 641 passed, 0 failed, 5 ignored, `/private/tmp/higgs-watchdog-validation.log`. Server library tests/release build next.
- Final watchdog build verified: engine 641 passed/5 ignored; Higgs server 782 passed; release binary build exit0. `/private/tmp/higgs-watchdog-validation.log`. Installed SHA256 `81385258b48b2fd736f9a3864e97e0eeea19dc2c1b65d370c89997f09dcbb4f7`, superseding earlier 7c88 build. Running PID75074 in tmux `higgs-watchdog-live`, log `/private/tmp/higgs-watchdog-live.log`, boot b8b73d97-39ef-4d99-a2f7-2732740a95d9. Final identical cold45K run `/private/tmp/higgs-recovery-45k-final-result` started with reservations0/queued0, prompt ceiling47104.

## Final measured outcome

- Cold 45K PASS: 45,003 prompt tokens; 3/3 literal recall, final stop, 475.795s end-to-end. Reported prefill44,992tokens/467.320s =96.28tok/s. Peak sampled physical footprint17.96GiB, normal pressure throughout, zero new swap-outs. Proves backend long-prompt completion/recall, not multi-turn nanobot endurance or best performance. Durable artifacts in `recovery-validation-20260906/`.
- GLM review verified provider/model zai-coding-plan/glm-5.3-flash via session export, no fallback. Review saved with corrections needed: position-correlated decline does not exclude config/kernel limitations; 512-token input does not establish 512 chunk size; matching prompt length alone does not create identical benchmark conditions.
- Short probe507actual prompt tokens completed in7.53s; server emitted no finished-prefill event for subchunk prompt, so no pure-prefill rate claimed.
- EschaLabs primary published M4base24GB:264tok/s at512 and2048; paired output-fused512 result289.79tok/s. https://github.com/EschaLabs/escha-mlx/blob/main/docs/PERFORMANCE.md . No matching45KbaseM4 published result located.
- User added ANE investigation; capacity_recovery assigned read-only code/docs/git study of both repos and maderix part4b.
- ANE study complete, `recovery-validation-20260906/ANE-PREFILL-INVESTIGATION.md`: old branches/archived experiments found; no demonstrated current Escha W2 end-to-end gain. Article supports narrow fixed-shape kernels, not native trellis/45K path. Recommend measuring current GPU operator fractions, then one copy/sync/memory-inclusive T1024 projection experiment before production integration. No ANE code change.
- Final health: installed hashes verified, Higgs idle normal pressure, zero active/queued reservations, stopOutcomes empty; original nanobot PID53654 preserved.
