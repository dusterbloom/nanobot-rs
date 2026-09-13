# Tool recovery validation

Worktree: `/private/tmp/nanobot-tool-recovery`, branch `codex/tool-recovery`, base `f32f5be`.

The implementation keeps recovery within a user turn. A new user turn does not automatically inherit cached tool outcomes or an old pending-call blocker. Automatic approval review rejected that cross-turn restoration design because it could reuse stale evidence or block an entire session. The proposed restoration API and its tests were removed; no such path was applied to context preparation.

## Baseline

The unchanged release library binary passed **2,996 tests, 27 ignored** in 33.96 seconds. A sandboxed run had 27 failures caused by tests being unable to create local mock HTTP servers; rerunning with local socket access passed all tests. Evidence: `/private/tmp/nanobot-study/baseline-full-lib-unrestricted.log`.

The new DB tests first failed on noncanonical argument serialization and the absence of a distinct cached-replay disposition. Both passed after those fixes. An injected interruption-journal failure then reproduced the missing persistence error propagation. Logs: `db-red.log` and `db-green-interrupted-red.log` in the evidence directory.

## Exploratory Escha probe

Six isolated inference requests used the loaded `escha-35b-a3b` model, temperature 0, max_tokens 512, and baseline/candidate tool descriptions with selected recorded tool responses. Generated tool calls were recorded, **never executed**. These are small constructed next-action probes, not full task runs, randomized trials, or a speed benchmark. Candidate wording was still being refined during implementation.

| Probe | Baseline | Candidate |
|---|---|---|
| Before Git use | Correctly chose `git log --oneline -50` | Correctly chose `git log --oneline -n 50` |
| After useful output plus exit 141 | Proposed counting Git commits | Proposed a filtered Git pipeline ending in `head -30`; guidance did not eliminate this risk |
| Stored result ends but file has more lines | Claimed “Now I have the complete plan” and moved to implementation search | Timed out after 180 seconds; comparison inconclusive |

The probe does not prove a behavioral improvement or speedup. It supports retaining structural limits alongside better guidance and testing exact file/result boundaries. Raw requests and responses: `/private/tmp/nanobot-study/guidance-evaluation.json` and `eval-*-request.json`.

The real CLI `scripts/turn_bench.sh` uses the user's configured database and service by default. It was not used to modify the active environment. Final release correctness and build results will be recorded below; no inference speed claim is made.

## Scope of proof

Unit/integration regressions protect exact command strings, actual tool execution counts, original result provenance, successful-versus-failed outcomes, argument-order compatibility, tool-call/result pairing, and bounded attempts. They do not establish why the previously running binary executed two identical explicit-cwd successes before rejecting the third. That live discrepancy remains unproven; the source contract is protected by a full-path regression and diagnostic correlation.

## Combined verification

On the local macOS arm64 machine, the initial candidate `cargo build --release` passed in 2m30s. A newly unused cache-text accessor was then restricted to test builds; production uses the provenance-bearing accessor. Only the three pre-existing retention dead-code warnings remain.

The first full release library run passed 3,018 tests with 27 ignored and exposed three regressions missed by focused tests: a stale final-answer wording assertion, a circuit-breaker fixture primed without the now-materialized working directory, and a reused-call-ID conflict test reaching the next model response. The first two required fixture/wording corrections. SQLite events showed that background compaction consumed the reused-ID test’s queued conflict response, so the foreground turn never executed that call. That fixture now has sufficient context plus assertions for the seed answer and provider-call count; the immutable-store conflict assertions remain intact. Separately, all 40 release protocol and compaction integration tests passed (10 LCM, 6 protocol-invariant, 24 protocol tests).

## Final acceptance

- `cargo test --release`: **3,061 passed, 0 failed, 31 ignored** across library, binary, integration, and documentation targets. The library contributed 3,021 passes and 27 ignored. Compile: 2m48s; library execution: 15.50s. Evidence: `/private/tmp/nanobot-study/final-release-tests-verified.log`.
- `cargo build --release`: passed on the final source after the complete suite. Evidence: `/private/tmp/nanobot-study/final-release-build-verified.log`.
- `rustfmt --edition 2021 --check` for all 12 changed Rust files and `git diff --check`: passed.
- GitNexus change analysis: expected 12 source/test files, 108 symbols, 42 affected flows, critical aggregate impact. Direct consumers were updated and exercised; the index misses some receiver-typed Rust edges and skips the large test module, so graph counts are not complete coverage.
- Independent Sol review findings were resolved. Final test-fixture corrections preserve production behavior and strengthen the reused-ID fixture’s queue assertions.

The branch is `codex/tool-recovery`. The installed binary, active session, and original working tree were not modified. No model speedup or universal elimination of reasoning loops is claimed.
