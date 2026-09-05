# Higgs allocation and runtime-default repair plan

Goal: close the allocation audit gaps, validate native Escha defaults, and run the verified nightly binaries.

Architecture: retain the existing governor and worker ownership. Correct byte facts without changing cache precision; feed bounded worker observations into the existing controller. Add honest diagnostics without relaxing pressure safety.

Constraints: release builds/tests only; persistent jobs in tmux; impact checks before symbol edits; graph change analysis before commits. Preserve main-checkout/user changes. Serialize GPU experiments.

- [x] Locate hardened revision and nightly ancestry: nightly 5c7605f91 is an ancestor, 30 commits behind 327e5021e. Installed Higgs dates August 29; experiment runs separate higgs-hardened.
- [x] Refresh isolated worktree graph and inspect affected loader call chain (CRITICAL).
- [x] Correct native Escha FP32 persistent-cache slope and account for 256-slot allocation slack; regression against recorded 40,960 bytes/token geometry.
- [x] Expose oversized retained-cache drops; align the 16K endurance fixture byte limit with measured layout.
- [x] Wire serialized prefill/decode observations and completion into capacity learning, preserving cancellation ownership and dirty-request exclusion.
- [x] Export raw pressure, cumulative VM activity, measurement freshness and allocator cached bytes alongside existing effective pressure.
- [x] Verify prefill/decode ENV selections: real-model QGEMM was 34.48% slower, so preserve scratch default; native QMV decode and established safe defaults remain. Numerical/regression tests pass; raw benchmark artifacts saved.
- [ ] Run release regression/build checks and real request/long-session validation, recording hashes, source revision, kernel choices and synchronized telemetry.
- [ ] Review graph changes, integrate all hardened commits plus fixes into nightly, build/install matching executable and Metal library, restart tmux service and verify active process provenance.

Ownership: parent handles estimator, allocator snapshots, integration/provenance and live validation; kernel_defaults handles Escha kernel selection; live_learning handles engine receipts/registry/retention diagnostics; pressure_telemetry handles pressure observation and diagnostic types. Shared-file changes are coordinated explicitly.

Validation commands: `cargo test --release -p higgs-engine mlx_tuning`, `cargo test --release -p higgs capacity`, model-specific Escha numerical tests, `cargo build --release -p higgs`, then end-to-end replay with no kernel ENV overrides. Build target and exact binary hashes are recorded with results rather than inferred from PATH.

Progress: installed nanobot now hashes to 18d5c9bc123d686596b62bf5aa53d843477db9082a5753e8353ff016486c86bc (validated release). Local Higgs nightly fast-forwarded to 327e5021e; new fixes on fix/capacity-runtime-integration await combined validation. Review found and is correcting partial decode measurement, transient byte labeling, and nested capture cleanup.

Installed and pushed nightly `890039d7f`; complete release suites and real-model retention check pass. Correct installed Higgs process and matching artifacts verified. Default auto profile resolves throughput/scratch/native Escha without kernel overrides. Spaced allocation probe and then installed CLI + paired decision-policy endurance are running serialized in tmux (`higgs-allocation-probe`, `higgs-endurance-final`).

## End-to-end regression discovered after installation

- [x] Fix nanobot safety-compaction replay losing the current authoritative user update: SQLite retains revision4/5 but outbound request contains old summary plus Continue. Add a failing regression, fix the existing hot path, release-test/build and install nanobot, then replay endurance. Owner: live_learning; parent owns live validation and integration.

- [x] Fix repeated CapacityExceeded recovery reporting saved pending work without writing a pending record; share the existing persistence path, add a regression proving durable pending bytes, release-validate and include in nanobot installation.

Nanobot core repairs committed `5d4a680`, full library2,951pass/buildpass, installed exact release hash8503b965…. Post-fix live endurance remains in progress; do not mark overall endurance complete from unit tests.
