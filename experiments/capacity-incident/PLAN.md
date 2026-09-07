# Live capacity incident — 2026-09-06

- [x] Confirm no benchmark jobs remain running; identify installed binaries.
- [x] Observe available/4096 total/4096 output/0 prompt envelope under normal pressure; direct 17-token prompt answers successfully.
- [x] Back up user Higgs config, restore measured cache limits and restart service; 18432 total / 14336 prompt, READY through installed nanobot passed.
- [x] Verify ordinary nanobot request end to end: three HN headlines and links, 85.8 seconds.
- [x] Fix and regress client wire budget, retry ceiling/state and visible diagnostic.
- [ ] Finish secondary Higgs public-envelope patch: uninstalled, four legacy test expectations unresolved.
- [x] Nanobot release library tests/build, installed binary provenance and final live service verification.
- [ ] Repeat matched speed checks later without overlapping builds and live inference; earlier pressure-limited attempt is invalid.

Installed binaries at incident: nanobot e26c4082bff1dea1540a593cc176e4fd092b3f12e7fb9504a1b9717fc42755a0; Higgs fd9747c68f4f85770eb47d30cf7c077b9a014e559a71dc03e2b46380dcfa86fc. Live PID 30417 maps installed Higgs. Existing user configuration had 49152 retained tokens, 2GiB retained budget and 1GiB prefix budget. Preserve existing unrelated workspace changes and all user session data.

HN follow-up fetched stories but failed final generation. Its raw wire control still advertised 45056 prompt tokens from configured core budget. Both failed requests were byte-identical and server rejected with 8192 prompt / 12288 total. Fixes under release validation: wire effective budget, retain typed rejection ceiling after discovery, preserve one-retry state through compaction, truthful visible terminal reason, Higgs output-only envelope normalization.

## Reopened after real two-turn failure
- [ ] Isolate pressure observation/static ledger behind 73728 -> 8192 collapse.
- [ ] Resolve configured-versus-effective footer ambiguity.
- [ ] Trace HTTP400 rendered-prefix overflow, repeated compaction and task loss.
- [ ] Validate multi-turn browsing/task retention before claiming recovery.

## KISS implementation contract

One existing Higgs capacity envelope remains authoritative. Normal OS pressure plus global compression activity must not reduce it; compression still disqualifies upward learning, and real pressure/swap plus allocation admission remain enforced. Nanobot displays effective prompt room and explicit output reserve. No new mode, flags, controller, or fallback pipeline. Preserve durable session history.

- [x] Regress normal/compression oscillation, then remove synthetic pressure promotion.
- [x] Complete public output-only envelope normalization while asserting internal registration semantics separately.
- [ ] Show effective prompt ceiling in footer and label output reserve explicitly.
- [ ] Serialize release checks/builds with server stopped.
- [ ] Restart exact installed binaries; validate sustained multi-turn context with literal recall and capacity telemetry above the previous 6K cliff.

Fixed-limit overrides would hide bad policy and bypass live evidence. Replacing the whole controller would enlarge the risk. Chosen change removes the observed false signal at its source and retains the existing memory admission path.

## User interruption: recover missing conversation

Stopped queued nanobot validation and OpenCode review at user correction. Long-context check has NOT run. Higgs library 777 passed and release build succeeded; installed SHA256 ad60c84abed93de88ba54d3a626171d15cdb769e031e1171bc6b8e927d1217f6, backed up old binary as ~/.local/bin/higgs.before-kiss-20260906, serving in tmux higgs-live-kiss. Nanobot footer changes remain unvalidated/uninstalled.

Verified all 29 rows of user session 20260906_143059_14bf08 remain in SQLite; exported messages and metadata to /private/tmp/nanobot-hackernews-recovered-20260906.json and readable .md. Plain nanobot launch creates a fresh ephemeral key; explicit --resume is needed. Reopened exact session in tmux nanobot-recovered without sending a user message. Preserve this user's conversation and do not resume benchmark/build traffic without reassessing the user's priority.

## Active priority: usable sustained context

User explicitly redirected from chat recovery back to long-context reliability. First installed-pair test stopped at turn 1: user input 3916 native tokens; adaptive max_tokens6144 exceeded Higgs output ceiling4096. Server advertised79872prompt/83968total and rejected with413, so retry/LCM cannot help. Correct compute_adaptive_max_tokens to respect existing live capacity description. Added wire-output regression (live end-to-end reproduction failed before patch). Release nano suite/build running in tmux kiss-output-build with Higgs stopped. Restart service and rerun nine-turn long test after install.

The production CLI singleton kills a previously running agent; initial test terminated our recovered TUI. Test driver now refuses to start if the recorded agent PID is alive. Do not launch test CLI concurrently with an interactive agent.

Installed nanobot d8dc4659a8943a8e1270b37be53eeaef10bdba9ba0cecb0bfac97f3936c83fe8: 2968 release library tests passed,27ignored; release build passed. Nine-turn rerun reached17689 rendered prompt, four exact acknowledgments, then fifth request503 after global swap counter increased28868pages while OS pressure epoch0 and raw normal. MLX prefill peak16669341732B. Retention dropped at fourth turn under16K/768MiB configuration.

Auto-review rejected proposed removal of swap-derived critical state as a memory-exhaustion safety tradeoff requiring explicit approval. No part of that rejected edit executed; source swap protection intact. Safer alternative applied: retained tokens49152, retained bytes2147483648 (backup config.toml.before-long-retention-20260906); prefill chunk and threshold256 via existing runtime env, all pressure/admission protections intact. Live server tmux higgs-long-chunk256. Run identical long test next.

## Direct trellis allocation investigation

Scratch chunk256 reduced peaks but failed on turn3 under actual constrained pressure; no long-context pass. Direct SIMD QGEMM crashed on the first5866-token prompt at24.6GiB physical footprint. Forcing per-layer evaluation also crashed, disproving that evaluation cadence alone fixes it. Found missing MLX C config/input/output container frees in eschamoe_gather_qgemm_simd; neighbor scalar wrapper releases these handles. GitNexus HIGH impact reported before edits. Added repeated-call active-memory regression: RED before67732B after2235156B over32 calls. Applied matching ownership cleanup, including failed output extraction. Escha release regressions running, model stopped. No pressure protections changed.

- [ ] Pass repeated-call memory regression and Escha numerical parity.
- [ ] Build/install leak-fixed Higgs; rerun exact nine-turn long context with QGEMM and physical memory telemetry.
- [ ] Validate real tool follow-ups plus early fact recall; leave verified server running.

Leak fix GREEN: all52 Escha tests passed; release Higgs433005be6a54dc0fa7b7dcdbb8565734c8785561541b73a0a6c3597daa750535 installed. Same QGEMM sequence now passes five turns through21621 prompt tokens. Sixth starts compaction, then pressure cancellation/503. No COMPLETE. Known retained KV is also charged as learned_retained_bytes in addition to prompt/output/fixed costs; capacity dropped each turn from51200 to25600total. Added regression asserting known KV growth alone does not lower capacity. GitNexus ledger target UNKNOWN (index36commitsstale); textual callers are capacity solve and admission. Server stopped; RED test compiling. Planned correction deducts already-charged session bytes from retained evidence, retains any positive excess, and preserves all pressure rules.

Retained-KV regression RED with memory-bound fixture: observing4096 known KV tokens lowered total82944→78848. Correction in byte_ledger_with_evidence_for_policy charges only retained high-water beyond prompt/output/fixed session bytes already in ledger. All778 Higgs release library tests pass, including excess-component downshift and pressure protections. Release executable building. No pressure-policy removal.

Ledger-corrected installed Higgs d459eca0dff47a3c55c505f329f667f4a5f0e17fb391f4dae83e33f8244fd729 held47104prompt across firstfourturns. Then pressure/503 atturn5. Telemetry atturn4:23.918GiB physical,14.479GiB active,9.219GiB allocator cache. Retention/KV cache and MLX allocator free-buffer cache are distinct. Existing WiredLimit branch raises wiring but leaves allocator cache at large MLX default. After impact UNKNOWN + textual confirmation(simple and batch initializers), capped reusable allocator cache at256MiB in existingdefaultbranch; no live-array limit or pressure-rule changes. Releasebuildrunning; live rerun is the regression check.

Allocator cap run: eightexactackturnspassthrough33444prompt,25–35sec/turn,normalpressure,47104promptcapacity. NinthrecallFAILED after200.98sec coldreplay: nanobotSQLitefilter silentlydroppedbatch1 at37977estimatedhistorytokens > max_messages229*150=34350. Outputwronglyclaimslaunchkeynotprovided. NoCOMPLETE. Graphfilter_historyCRITICALreported. Regressionaddedfor8fatturnsunder229message/600turnlimits; releaseREDcompilingwithserverstopped. PlannedremovalofredundantStage6tokenbudget(andprivateestimator), preservingexplicitrow/turnwindows,protocolfilters,andagent/LCMbudgetauthority.

## Final default-configuration proof

- [x] Nanobot2965release tests passed;27ignored;releasebinarybuilt/installed8eadd230c5a8532b56b3cae6f55ea6c5d6c3188794a9f68a17afe825f01924f3.
- [x] Higgs64f58982d4dc64293654f05bafba78915782942da99d9894a976e75a56da1259 runningnormal scratch_matmul/throughput/1024chunk/automatic256MiBallocatorcache;tmuxhiggs-live-final.
- [x] Matched scripts/turn_bench.sh two turns each:before11.170/1.463s,after10.954/1.463s.
- [x] Nine-turn default-final COMPLETE:33507renderedtokens,allkeys/projectcorrect,zeroLCM/capacityerrors,normalpressure,no swap-out increase,18.102GiBmaxsampledphysical. Recall10.05s,warm33449tokenreuse/58prefill.
- [ ] RealHN/articletooluseandkeyrecallinthesamelongsession. Firstfollowuplauncherinheritedwrongcwd,changingCwdfieldandforcingcold33500replay; disclosedtesterror, correctedscriptforfuturelaunches. Do not attribute thatcoldreset toproductiontool-schema instability or countitslatencyaswarm.
- [ ] Finalresults/provenanceupdateandlivehealthcheck.

- [x] Real browsing passed:HNfrontpage→selectedCloudinaBottle→articlefetched/read→groundedsummary+threeexactkeys.39315prompttokens,18maincalls,zeroLCM/capacityrejections.
- [x] Final live state healthy/idle,47,104Higgspromptallowance,normalpressure,swapcounterunchanged3278240;hashesverified;tmuxhiggs-live-final;nightlybranch.
- [x] Results updated with all failures, corrections, proofs, test-launcher cwd mistake and remaining cold-replay cost. Beyond39.3K and actualexhaustion/LCM correctness remain unproven, not silently claimed.
