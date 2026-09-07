# Higgs measured roofline campaign

User authorized optimization across GPU, CPU and ANE on 2026-09-06. Preserve exact checkpoint and long-context recovery.

- [x] Snapshot validated baseline in isolated worktree, preserve unrelated changes.
- [x] Measure current hardware/runtime state and cold/warm prefill/decode baseline.
- [x] Attribute biggest measurable loss; rank whole-request upside rather than theoretical TOPS.
- [x] Test bounded candidate with differential correctness and real workload timing.
- [x] Keep only repeatable gains, release-test, install verified binary, report remaining gap.

Hardware slots serialized; agents read code only. No public publication. Rooflines are measured per operation/dtype/shape; shared memory bandwidth is not additive across accelerators. Preserve user nanobot process.

16K instrumented run completed: 16008 prompt tokens, 143.901s reported prefill (16000 processed), 168.860s total, 256 completion tokens. Trace attribution only, not clean timing baseline. Existing W2 component breakdown building in tmux higgs-roofline-breakdown; validated server automatically restored afterward.

INVALID: existing bench_prefill_breakdown unchunked/full-vocabulary diagnostic triggered swapping (system swapouts3283831 ->3824247), interrupted with SIGINT at T5120 before completion. Do not use any component timings as evidence. First restore attempt rejected transient critical pressure; frontier build running without model resident, restore retried after build. Need bounded serving-path profiling with memory abort.

Clean live16K baseline completed: 16008prompt,16000processed/121.184s prefill=132.031tok/s, TTFT122.546s,128completion/8.632s decode interval~14.713tok/s,131.178s request, peak14.897GiB, normalpressure/no newswapouts. Agent fusion work interrupted by usage limit; main resumed review/testing.
Audit correction: SliceUpdate calls copy_gpu, but Metal copy_gpu invokes set_copy_output_data and RETURNS WITHOUT COPYING if input is donatable with same dtype. Whole-KV-copy-per-token claim is unproven; do not optimize cache on that assumption. Need donation/append timing evidence.

Fusion screening: first shared-memory7-stage version exact on tested values but slower(0.60–0.81x operator baseline), rejected. Revised first5stages SIMDshuffles +2crossgroup sharedstages: bit-identical across141312 tested f32 values, 2.03–2.46x operator speedup for4096/8192rows bothwidths; small8rows0.94–1.15x noisy/neutral. Source snapshot fusion-shuffle.patch. Candidate fullserver build then matched OFF/ON/ON/OFF16K gates pending.
Measured Metal streaming copy256MiB input+256MiB output,20timedGPU samples: median5.444562ms,98.6068GB/s decimal. This is read+write operation bandwidth, not inference roofline.

Critical benchmark condition discovered: macOS battery31%, Battery lowpowermode1, AC lowpowermode0. Existing measurements are power-limited, not maximum-performance ceiling. User asked asynchronously to connect AC; power watcher logs transitions, reject any timing run spanning power-source change. Do not silently change system power preferences.

Profiler audit: existing HIGGS_PROFILE append_ms evaluates K/V first, so it isolates append+eval; attn_ms timer starts BEFORE those evaluations and append, so its label "attn(kernels)" is inclusive and cannot be read as pure SDPA time. Use correct interpretation or fix diagnostic timer before claiming cache-vs-attention ratio. No production cache change made.

Battery sweep incomplete: OFF140.147s,ON128.353s,ON135.148s; outputs identical. AC detected at1788734388.5 beforefourthtimedrun. FourthOFFdiscarded, benchmarkguardaborted new1020swapoutpages; installedserverrestored. Re-run entirematchedset onAC required; no conclusion from incompletebatterybracket.

AC fusion OFF/ON/ON/OFF complete: controlmean92.603s vsfusion91.565s (1.12%reduction); prefill1.56%reduction,decode3.65%slower. Allanswersidentical/no swaps. Inadequateend-to-endgate: rejected default, productionfusion edits removed fromworktree; preserved output-fusion-rejected.patch andcandidatebinary. Only new profilingtimerfix remains. Screeningexistingchunk2048 andpackedGEMM next.

Existing controls onAC: chunk2048/scratch82.157s,chunk1024/GEMM81.923s,following1024/scratch93.935s. Repetition: combo2048/GEMM80.395s,1024/GEMM82.577s,followingcontrol89.829s. Allmatched16Koutputsidentical. Warmshort512: GEMM8.403s TTFT3.419s,control9.222s TTFT4.254s. Prefer1024/GEMM for long-context margin; combo adds~2GiB for~2.2s gain. 45Kcontrol/GEMM pair withcachedownerfollowup nowrunning.
Planned conditionaldefault if45Kpasses: baseM4 nativeEscha GEMM auto; keep1024chunks. ShareexistingcachedCPUbrandhelper throughutils (no duplicatehardwarediscovery), singlepureGEMMresolver for execution andruntime_identity (currentlyidentity reparsesENV==1; mustfixthat duplication). Otherhardwarekeepsolddefaultpendingevidence. Requiredmodel/kernelregressiontests andcorrectbinaryinstallation aftergraphreview.

45Kstatelesscontrol passed3facts314.515s; statelessfollowup correctNeri319.190s butcache0. Fixtureomittedsession_id (Nanobotdoesprovideit); not a retained-cache test. Stopped supervisor91542 duringnextcandidatecold, fixedfixtureusingnumeric session_id2026090701 andassert>44000cachedtokens onfollowup. Rerunmatchedretained45Kpair afterrestoration.

Retained45K passed both arms: control307.813s/TTFT304.451s, GEMM286.659s/TTFT283.255s (6.87% total reduction). Both all3facts, followup cache45037 tokens, control1.378s vsGEMM1.164s. No new swapouts. Retained path crossed300s successfully; suspected watchdog failure not reproduced. Behavioral gate and default unit/release checks pending.

Current integration checklist:
- [x] Retained45K comparison and cached followup (no swap, correct facts).
- [x] Complete seven-case matched behavior check (control currently6/7: inventory96 instead103).
- [x] GitNexus refresh with explicit inner-shell cwd (first attempt inherited deleted tmux cwd).
- [x] Run expected-failing M4 default test, then apply one resolver policy and make it green.
- [x] Run existing kernel/oracle, base-M4 selection, identity and engine/server release regressions.
- [x] Review GLM feedback; inspect diff and call graph; preserve baseline recovery changes.
- [x] Build final binary/metallib, validate no-ENV selection and real serving profile.
- [x] Integrate to Higgs nightly and install verified pair; preserve Nanobot PID53654.
- [x] Record fresh hashes, measured limits and next bottleneck in durable STATUS.

Behavior completed6/7 each, identical answers; same inventory96/103 failure. DefaultRED failed expectedNone/M4; defaultGREEN passed. Existing oracle default-scratch assertion failed after accuracy bound PASSED; updated dispatch assertion and explicitly checked BOTH kernels with unchanged2e-3 oracle bound, thenpassed. Kernelbitpreservation, GEMMcomparison, M4policy,642engine,3identity,782server tests passed. GLM foundno blockingcodeissue, docsupdated; finalbuildpending.

Final binary fbf8b3e8... passes no-override retained45K293.456s TTFT290.135s, all3facts; continuation1.208s cached45037. Profiling and installation next.

Installed fbf8b3e8... + matching metallib, normal capacity47104, actualAPI READY0.737s, Nanobot53654 preserved. Final profiler and AC bandwidth99.43GB/s recorded; STATUS.md is authoritative. No commit/public release in this continuation. Roofline saturation and ANE benefit remain unproven.
