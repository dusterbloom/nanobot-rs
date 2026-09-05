# Scheduled-reset feasibility control

Approved by user 2026-09-06. Establish physical and protocol feasibility before autonomous strategy ranking.

Same frozen 20 updates, 12288 context ceiling, 2048 output reserve, native FP32 Escha, corrected installed binaries. Harness policy prescribes notes write + reset after each submission; pointer-only next-turn handoff. No model-authored reset timing, no mandatory post-submit context_status. This deliberately changes timing/inspection policy, not task facts. Model still authors notes and must execute durable writes/reset tools. Score every boundary; absence of a prescribed reset fails the control. Preserve raw failures. No pressure-policy bypass, no in-arm server restarts, no compiler or other GPU workload during inference.

Telemetry: server envelope/MLX counters plus proc_pid_rusage physical footprint/resident size, vm_stat free/available-category/compressor pages and sysctl swap. Raw free pages do not alone establish allocatable RAM.

- [x] Inspect existing harness and graph; run_turn HIGH (four evaluation callers); endurance_eval_live UNKNOWN confirmed ignored test invoked by Python driver.
- [x] Add scheduled policy with regression and unchanged autonomous prompts.
- [x] Validate release harness and memory sampling.
- [ ] Run 20-update control in tmux; audit exact correctness, reset/handoff count, tool receipts, headroom, peak footprint and capacity failures.
- [ ] If control passes, compare corrected autonomous arms with same memory telemetry; otherwise report the exact limiting failure without claiming feasibility.
- [ ] Restore installed Higgs defaults and record results/provenance.
