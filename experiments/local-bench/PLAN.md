# Local benchmark pilot — 2026-09-06

Hardware: Apple M4, 32 GiB unified memory (Mac16,1). No superiority claim is pre-assumed.

- [ ] Finish the corrected 20-update scheduled recovery control first; independently score complete snapshots and report any remaining failures.
- [ ] Install isolated, version-recorded EvalScope and Harbor tools.
- [ ] Run EvalScope against local Higgs: sequential streaming requests, fixed prompt/output budgets, cold and warm observations labeled separately. Record errors and physical footprint; this is an engine baseline, not a correctness benchmark or competitor win.
- [ ] Validate a real Terminal-Bench task with Harbor's oracle on local Docker/ARM64 before model evaluation. Select tasks on compatibility grounds before inspecting model outcomes; preserve exclusions.
- [ ] Start the Harbor reference-agent local-Escha pilot. Adapt nanobot to the same task environment before comparing harnesses; do not mislabel Terminus results as nanobot results.
- [ ] Record exact tool versions, source revisions, installed binaries, model artifact identity, effective generation configuration, budgets, task revisions, commands and raw results.

Keep one GPU workload active. No cloud model assistance or LLM judges in the local pilot. Docker task resources count toward the same machine budget. Production notes/reset and FP16 settings are unchanged.

External references: https://www.harborframework.com/docs/agents/terminus-2 ; https://www.harborframework.com/docs/tutorials/running-terminal-bench ; https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/examples.html
