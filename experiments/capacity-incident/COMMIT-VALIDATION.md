# Nanobot open-work commit validation — 2026-09-07

Scope: outstanding capacity retry/pending-turn durability, LCM summary fit/fidelity, history retention, regression tests, README/instructions, evaluation corrections and local inference benchmark evidence. Exclude unrelated untracked Exo setup.sh. Higgs production work already committed separately; no runtime installation or push in this task.

- [x] Reviewed pending Rust diff with independent deep_worker.
- [x] Found/fixed capacity-wake coalescing race: wake metadata could contaminate fresh steering and cause its loss.
- [x] Added gateway regression for both message arrival orders; RED observed timeout against unfixed code (exit101).
- [x] Reviewer confirmed shared coalescing predicate resolves blocker; no other blockers found.
- [x] Fresh GREEN regression (both arrival orders), release build and full release tests: 3,015 passed, zero failed, 31 ignored; exit 0.
- [x] Refreshed exact-checkout GitNexus analysis before code commit: 143 changed symbols, 141 affected processes, critical risk; no partial/truncated result. Staged documentation/evidence analysis: 423 changed symbols, 16 affected processes, 520 changed text files, critical risk; no partial/truncated result. Stored patch files can contain indexed code; they are evidence, not release source.
- [x] Credential-pattern audit of text artifacts and three benchmark SQLite databases found no matches. This is a scoped content check, not a general security audit.
- [x] Production source and tracked prose whitespace checks pass. Archived raw logs, patches and generated benchmark reports retain original whitespace warnings for evidence fidelity.
- [x] Code/regression committed separately as 6235fca; documentation/evidence accompanies this record.

Validation limits: no live scripts/turn_bench.sh run; it launches nanobot agent against the shared singleton and could displace the user's interactive session. No speed-regression or new endurance correctness claim. Historical experiment reports retain their own failures and caveats. No promise that the current capacity UI/admission incident is solved by committing these changes.

Logs and full graph outputs: /private/tmp/nanobot-commit-check. Final source hashes recorded to ensure committed code matches tested code. Initial graph was stale and had schema degradation; rebuilt index successfully before editing the coalescing path. Fresh targeted AgentLoop.run impact LOW, one indexed caller/flow; broader recovery diff remains a critical hot-path change.
