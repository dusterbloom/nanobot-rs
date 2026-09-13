# Higgs Runtime Model Contract Implementation Plan

> **For agentic workers:** execute this plan in the current checkout and preserve unrelated dirty work.

**Goal:** Make every model Higgs can serve a first-class nanobot citizen by having Higgs publish the runtime facts that nanobot needs, then making nanobot choose reasoning and tool transport from those facts without putting capability data into the prompt or adding model-name registries.

**Architecture:** Add an additive `capabilities` object to Higgs `/v1/models`. Nanobot fetches it once when constructing or switching a Higgs provider, keeps it on the provider/core, and applies it at the existing protocol and local-reasoning decision points. Unknown or incomplete metadata remains safe: retain existing heuristics and textual tool replay rather than failing model startup. The existing OpenAI tool-call parser and Higgs cache/session extensions remain unchanged.

**Tech Stack:** Rust 2021, serde/reqwest, Axum, existing nanobot provider/core/protocol paths, release-mode Cargo tests.

## Constraints

- Work in `/Users/peppi/Dev/nanobot-rs`; do not touch unrelated dirty files.
- Preserve the stable prompt prefix: no capability blob is injected into system messages.
- Keep one capability contract and one protocol decision path; do not add per-model nanobot registries.
- Preserve backward compatibility for Higgs clients that ignore unknown `/v1/models` fields.
- Use TDD for each behavior change, GitNexus impact before editing symbols, and GitNexus change detection before commit.

## Tasks

### 1. Define the shared runtime contract and merge semantics

- Add serializable tool-mode and thinking-mode enums plus a compact runtime contract beside the existing model capabilities.
- Add deterministic merge logic: runtime metadata overrides only the protocol/reasoning/vision/context facts it actually provides; existing nanobot defaults remain for absent fields.
- Add unit tests for complete, partial, unknown, and malformed capability payloads.

### 2. Publish Higgs model facts through `/v1/models`

- Extend Higgs `ModelObject` additively with optional capability metadata.
- Build the metadata from the already-loaded engine/template facts and configured context limit; do not add a second model registry.
- Keep list and load responses consistent and retain existing model IDs and vision behavior.
- Add route/type tests proving old clients still deserialize the response and nanobot can consume the new fields.

### 3. Adopt the contract during local provider construction

- Fetch the selected Higgs model’s metadata once alongside existing context discovery.
- Carry the contract through `LocalProviders` and `SwappableCoreConfig`, including `/model` rebuilds.
- Attach it to the OpenAI-compatible provider so request construction uses the same immutable contract as the core.
- Fall back cleanly when the endpoint is not Higgs or metadata is absent.

### 4. Use one contract-driven protocol and reasoning path

- Make local protocol selection consume the core’s resolved capabilities/contract, while retaining name-based lookup only for non-Higgs fallback paths.
- Send the contract’s thinking mode to Higgs through the existing `chat_template_kwargs` mechanism; explicit `/think` budgets continue to win.
- Select native tools, textual replay, or no tools from the published tool mode without changing the stable prompt prefix.
- Add focused request-body and protocol tests for Qwen-like native tools, Nanbeige-like textual fallback, and thinking-disabled defaults.

### 5. Verify and integrate

- Run focused red/green tests while implementing, then `cargo fmt --all`, `cargo test --release`, and `cargo build --release` in nanobot.
- Run the corresponding release tests/build in Higgs if available without disturbing its unrelated dirty files.
- Run `gitnexus_detect_changes` and inspect the diff for scope, cache-prefix stability, and duplicate logic.
- Commit only the focused changes; report any Higgs-side change that cannot be written or verified because it is outside the current writable root.
