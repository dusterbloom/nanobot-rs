# Adversarial fused-kernel follow-up

User requested GLM review. Launch rejected by automatic approval review: requires explicit authorization of curated private kernel source/build options/synthetic timings to Z.ai GLM-5.3-Flash. User subsequently explicitly approved the payload. GLM review launched through OpenCode in tmux higgs-glm-fused-review, session ses_f849d45e6ffeGUz3sV7ikquCi1. Curated bundle /private/tmp/higgs-glm-fused-review; source/build/synthetic evidence only, shell and external-directory access denied. Installed d77352ef8 remains intact.

Local preliminary audit: first BQ16/BK8 fused candidate is much closer than the later register-Q candidate. At Q1024/K32768 its bracket mean is424.552ms vs explicit-mask400.671ms (~6.0% slower) and native-causal392.142ms (~8.3% slower). At K45056 it is599.685ms vs explicit566.211ms (~5.9% slower) and causal564.574ms (~6.2% slower). Controls drift ~12–15% between bracket endpoints. These are means of two per-arm medians, not confidence intervals. There is insufficient evidence to declare a small optimization impossible, or a speed win established.

Probe evaluates inputs and numerical checks before timing, discards the first iteration per arm, and mirrors arm order; useful but only three timed repetitions per block. No GPU counters establish occupancy or spilling. Candidate and pinned upstream both specify -fno-fast-math, so that flag alone is not a demonstrated unfair comparison. Candidate numerical checks use random normal Q/K/V and do not establish full model behavior.

Next GLM review should prioritize source-proven problems, dispatch verification, benchmark drift, and one minimal falsifiable experiment; compare against the now-shipped query-block baseline as well as old fallback. No new GPU test or production edit performed in this follow-up yet.

Review completed: see GLM-FUSED-REVIEW.md for verified conclusions and corrections; GLM-FUSED-RAW.md preserves the authored report. No follow-up kernel implemented yet.
