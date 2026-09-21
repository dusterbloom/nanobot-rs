# Higgs Safe Model Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Higgs publish, and Nanobot display, every metadata-compatible local model exactly once under a stable correct name.

**Architecture:** Higgs owns scanning, canonical identity, loader compatibility, and naming through one typed catalog shared by CLI and HTTP. Nanobot consumes that catalog and falls back to resident-only listings for older servers.

**Tech Stack:** Rust, Axum, Serde, Tokio, existing Higgs adapter registry, Cargo tests.

## Global Constraints

- Tests must fail for the observed bug before production changes.
- No new dependency.
- Canonical path is identity; display name is not identity.
- Metadata validation never loads model weights.
- Old Higgs fallback is resident-only, never unvalidated disk discovery.

---

### Task 1: Authoritative Higgs catalog

**Files:**
- Modify: `crates/higgs/src/retention_plan.rs`
- Test: `crates/higgs/src/retention_plan.rs`

**Interfaces:**
- Produces: `AvailableModel` and `scan_models(roots: &[PathBuf]) -> Vec<AvailableModel>`.

- [ ] Add failing tests for Nanbeige acceptance, unsupported/malformed rejection, canonical-path deduplication, HF snapshot naming, and nested LM Studio naming.
- [ ] Run the exact catalog test filter and confirm failures describe current unsafe behavior.
- [ ] Implement canonicalization, adapter `detect`/`resolve`, artifact checks, and stable naming with existing standard-library and adapter helpers.
- [ ] Run the catalog tests and existing retention planner tests to green.
- [ ] Commit the isolated Higgs catalog change.

### Task 2: Higgs HTTP catalog contract

**Files:**
- Modify: `crates/higgs/src/routes/models.rs`
- Modify: `crates/higgs/src/router.rs` or the existing route registration module
- Modify: `crates/higgs/src/state.rs` only if configured roots/loaded canonical paths are not already exposed
- Test: existing model-route test module

**Interfaces:**
- Produces: authenticated `GET /v1/models/available` returning `runtime_model_load` and typed catalog records.

- [ ] Add a failing route test proving one record per canonical artifact, correct stable names, and exact loaded marking.
- [ ] Run the route test and confirm the endpoint/schema is missing.
- [ ] Add the minimal route using Task 1's catalog and current configured/default roots.
- [ ] Run route, API-contract, and model-switch tests to green.
- [ ] Commit the Higgs HTTP contract.

### Task 3: Nanobot catalog consumption

**Files:**
- Modify: `src/higgs.rs`
- Modify: `src/repl/commands/mod.rs`
- Test: existing unit-test modules in both files

**Interfaces:**
- Consumes: `GET /v1/models/available`.
- Produces: one `ModelEntry` per canonical catalog path with server-provided ID and loaded state.

- [ ] Add failing tests reproducing the duplicate Ternary/Nanbeige label, orphan nested variant name, canonical duplicate, and old-server fallback.
- [ ] Run exact filters and verify each fails for the intended reason.
- [ ] Implement catalog parsing and mapping; remove runtime filesystem guessing from the Higgs picker path.
- [ ] Run model discovery, selector, provider-contract, and retained-contract tests to green.
- [ ] Commit the Nanobot client change on `main` without staging unrelated dirty files.

### Task 4: Release and end-to-end proof

**Files:**
- Update installed binaries only after all tests pass.

**Interfaces:**
- Consumes: release binaries from Tasks 1-3.
- Produces: verified live `/model` behavior on localhost.

- [ ] Run GitNexus change detection in both repositories and review every direct dependent.
- [ ] Build release Higgs and Nanobot with the pinned local toolchain.
- [ ] Atomically install both binaries in `~/.local/bin`, restart `higgs-nightly`, and verify health.
- [ ] Query `/v1/models/available` and assert unique canonical paths, correct names, and supported adapters.
- [ ] Exercise Nanobot's real model collection/picker path and confirm Nanbeige once, Ternary once loaded, no `8bit`, and no unsupported artifacts.
- [ ] Mark the goal complete only after the live evidence satisfies every acceptance item.
