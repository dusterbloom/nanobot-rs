# KV and compaction fixes implementation plan

**Goal:** Correct the six verified findings in the September 12 review on nanobot main and higgs nightly.
**Architecture:** Preserve SQLite authority and existing cache implementations. Make old-prefix retention revocable for recovery, rollback all projection state, normalize release routing, restore exact hybrid arrays, and use a nonblocking busy-release contract. Map documented disk settings to the active disk cache rather than adding another pipeline.
**Tech stack:** Rust, Tokio, SQLite, MLX, existing release test suites.

The user approved the review's fixes. This plan executes that existing design; no additional feature or redesign approval is needed.

## Constraints

- Preserve existing dirty files; no unrelated source edits.
- Run GitNexus upstream impact for each modified existing symbol, inspect callers, and report risk before edits. Use current /opt/homebrew/bin/gitnexus and absolute repository paths.
- Add focused regressions and observe their failure before implementing fixes.
- Build/test only with --release; long processes run under tmux.
- No model-backed E2E or speed runs until the user confirms AC power.
- Keep changes uncommitted and reviewable unless integration requires a commit.

## Tasks

- [x] Refresh stale graphs, preserving user instruction files, and record impact.
- [x] Nanobot lcm.rs: reproduce reduced-capacity stall with alternating durable turns; revoke pinned head when recovery needs it; include committed/pending pins in LcmCompactionState rollback. Test restart and rollback behavior.
- [x] Nanobot provider/higgs URL: use the existing version-aware endpoint helper for sessions/drop; test actual emitted HTTP route against both root/versioned bases.
- [x] Higgs disk_prefix_cache.rs: persist required recurrent array geometry/dtype without Float32 loss; invalidate incompatible prior format safely; use realistic rank-three conv/rank-four SSM tests through disk reopen.
- [x] Higgs simple.rs/session release: return false promptly for a busy session, keep idle release behavior; test lock contention without holding an async runtime worker.
- [x] Higgs config/state/doctor/docs: connect kv_disk_dir and kv_disk_space_mb to the active disk config, preserve explicit existing disk config compatibility, honor disk budget, validate collisions or invalid settings. Update init template and README.
- [x] Review integrated diffs and run release builds, focused regressions, full relevant suites, required Higgs lint/format checks, and GitNexus detect-changes.
- [x] Prepare isolated live E2E: repeated compaction/capacity collapse/restart/recall plus hybrid disk restart; request AC readiness before execution.

## Validation evidence

Record commands, failing-before/passing-after results, limits and any remaining live validation below as work proceeds.

### Focused regression evidence (implementation in progress)

- Graphs refreshed with instruction-file bytes preserved. Cache symbols remain absent from the Higgs graph; direct source/caller inspection supplements it. CRITICAL compaction and HIGH provider/cache/config risks were reported before edits.
- Nanobot pin contraction and rollback tests failed before implementation, then passed. Full-wire SQLite regression exposed a second caller defect (21,882 estimated tokens against 17,000 room); system/developer/ephemeral messages and tool definitions are now reserved before compacting. Both overhead fixtures passed.
- Actual HTTP sessions/drop test failed for versioned API bases before normalization; passing-after log is `/tmp/kv-review-20260912/provider-drop-green.log`.
- Hybrid disk restart test failed on restored convolution geometry `[1,24]` versus `[1,3,8]`; v2 invalidation also failed before the v3 fix. Initial 19 disk tests passed; the expanded f16/bf16 convolution cases will be rerun with final budget tests.
- Eager-release contention regression timed out with the old blocking call, then passed using the nonblocking path. Ordinary reset retains its blocking semantics.
- Documented-directory configuration regressions failed before wiring. Whole-file byte-budget regressions and integrated validation remain in progress.
- No live model has been loaded. AC-power confirmation remains pending.

- Integrated Nanobot release build passed; full release library suite: **3,010 passed, 0 failed, 27 ignored**.
- Final Higgs disk-cache suite: **22 passed**, including byte-budget regressions and f16/bf16 convolution extensions.
- Nanobot fmt differences were reproduced against HEAD for all seven reported files; no unrelated formatting sweep.

- Final Nanobot integration targets `lcm_e2e_tests`, `protocol_invariants`, and `protocol_tests`: **40 passed**. Final graph detect-changes completed; 117 affected flows include preexisting instruction-file changes. Direct source diff review and `git diff --check` are clean.

- Higgs release server build passed. Full server test command passed **926 tests** (811 lib, 8 binary, 107 integration), **10 ignored**, no failures.
- Required Higgs Clippy failed in unchanged `higgs-models`: **128 errors, 206 warnings**. `git diff --exit-code -- crates/higgs-models` confirms no tracked changes there. Direct-crate lint checks and final targeted rechecks after new-code lint cleanup are pending.

- Independent cross-review completed for all changed subsystems; no additional actionable defects found. Final post-lint-cleanup disk suite again passed all 22 tests.

- User confirmed AC; `pmset` verified AC power. Isolated testserver started, first request refused503 before inference with constrained pressure/zero safe capacity. Stopped only owned server to release model memory until final compilation ends.
- All new-hunk direct Clippy diagnostics corrected; inherited direct lint debt remains (Higgs79 errors/137 warnings; engine148 errors/183 warnings). Final source frozen, direct diff check passed and graph scope rerun; final release rerun in progress.

- Final frozen-source rechecks passed: disk22, config/doctor6, release build. Nine source hashes match the frozen record; binary SHA256 `ebf4ac548731a30be2af3c7dcd23a7a20ea02de86c74b6a7a89cfebf1dacffeb`. No compiler processes remain.
- Live final-source server startup passed, but idle capacity remains unavailable/constrained with zero safe tokens. Independent macOS query reports warning level2. Asked user to free memory; live inference, restart reuse, recovery and matched timing remain unverified.
- [ ] Execute and score live tests once host memory pressure clears; stop owned testserver afterward.

The final capacity recheck still reported constrained pressure and zero safe tokens. Stopped the owned isolated test server to release its model memory while waiting for the user to free host memory. No live inference passed; restart reuse, exact live recovery, and matched timing remain pending. All test configuration/artifacts are preserved under `/tmp/kv-review-20260912/live/`.
