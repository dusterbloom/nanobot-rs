# Manual Compact and Exact Prompt Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a foreground `/compact` command and make automatic LCM hard admission use the rendered-prompt estimate that the provider receives, without making the immutable prefix trigger premature soft checkpoints.

**Architecture:** Keep one LCM compaction implementation. The agent loop will estimate rendered messages plus tool definitions for hard-cap admission while retaining conversation-only soft hysteresis. A maintenance entry point will force that same compaction path without sending a synthetic model turn. In-flight generation may yield to a foreground turn, while the atomic publication phase remains protected and its pending checkpoint is installed at the next safe boundary.

**Tech Stack:** Rust 2021, Tokio, serde_json, existing LCM/session/prompt-cache code, cargo release tests.

## Global Constraints

- Preserve the immutable system/developer prompt prefix byte-for-byte.
- Do not add a second compaction algorithm, context store, or provider path.
- Keep all existing dirty user files untouched.
- Use `cargo build --release` and `cargo test --release` for validation.

---

### Task 1: Make LCM thresholds accept the exact prompt estimate

**Files:**
- Modify: `src/agent/lcm.rs`
- Modify: `src/agent/agent_loop/shared.rs`
- Test: existing unit-test modules in those files

**Interfaces:**
- Add an LCM threshold method that receives `(available_prompt_tokens, rendered_prompt_tokens)`.
- Keep the existing conversation-only method for focused LCM tests and compatibility.
- Automatic compaction passes rendered protocol messages plus the selected tool-definition estimate.

- [x] **Step 1: Write the failing threshold regression test**

  Add a test proving a prompt estimate that includes the otherwise-excluded fixed prefix can reach `Blocking` at the configured hard threshold.

- [x] **Step 2: Run the focused test and verify it fails**

  Run: `cargo test --release lcm::tests::<exact_prompt_threshold_test> -- --exact`

  Expected: compile failure because the exact-prompt threshold method does not exist.

- [x] **Step 3: Implement the smallest exact-prompt threshold method**

  Factor only the threshold decision needed by the new method; leave the existing conversation-only API behavior unchanged.

- [x] **Step 4: Use the exact rendered estimate in `manage_compaction`**

  Render `ctx.messages` through `ctx.protocol`, add `tool_def_tokens`, compare against the prompt budget after the response reserve, and use that value for soft/hard threshold decisions and the raw over-capacity guard.

- [x] **Step 5: Run focused tests and formatter**

  Run: `cargo test --release lcm::tests::<exact_prompt_threshold_test> -- --exact`

  Run: `cargo fmt --all -- --check`

  Expected: the regression test passes and formatting is clean.

### Task 2: Preserve background checkpoints across the next foreground turn

**Files:**
- Modify: `src/agent/agent_loop/shared.rs`
- Modify: `src/agent/agent_loop/compaction.rs` only if the existing lifecycle API needs a narrow wait helper
- Test: `src/agent/agent_loop/tests.rs` or the focused compaction tests

**Interfaces:**
- A foreground pre-call must not discard a publishable soft checkpoint; it installs a pending result when safe and only cancels still-generating work so user turns are not blocked indefinitely.

- [x] **Step 1: Existing lifecycle regressions cover publication handoff**

  The existing `published_soft_checkpoint_replays_and_installs_on_next_turn` and publication-survival tests cover this boundary.

- [x] **Step 2: Run the focused lifecycle tests**

  Run: `cargo test --release agent_loop::tests::<background_checkpoint_boundary_test> -- --exact`

  The pre-change behavior and the old foreground-preemption test were used to validate the lifecycle contract.

- [x] **Step 3: Preserve publication handoff**

  Keep the existing cancellation-safe lifecycle: generation can yield, publication cannot; install pending results before another fold and preserve the existing hard-pressure/cache ceremony.

- [x] **Step 4: Run the focused regression tests**

  Run: `cargo test --release agent_loop::tests::<background_checkpoint_boundary_test> -- --exact`

  Expected: PASS.

### Task 3: Add `/compact` as a forced foreground LCM command

**Files:**
- Modify: `src/agent/agent_loop/mod.rs`
- Modify: `src/agent/agent_loop/shared.rs`
- Modify: `src/repl/commands/mutation.rs`
- Modify: `src/repl/commands/mod.rs`
- Modify: `src/repl/mod.rs`
- Modify: `src/tui_app/app.rs`
- Test: command/agent-loop unit tests adjacent to the changed code

**Interfaces:**
- Add a crate-visible `AgentLoop::compact_session_now(session_key)` maintenance method returning before/after token counts.
- The command invokes that method, reports no-op versus reduced context, and never sends `/compact` to the model.

- [x] **Step 1: Write the failing command-surface test**

  Assert `/compact` is listed in autocomplete/help and dispatches to the maintenance handler rather than falling through as a user prompt.

- [x] **Step 2: Run the focused test and verify it fails**

  Run: `cargo test --release repl::commands::<compact_command_test> -- --exact`

  Expected: failure because the command is not registered.

- [x] **Step 3: Add the maintenance entry point**

  Build the existing context through the shared preparation path, remove the synthetic empty user draft, resolve live capacity, force the existing `PreserveContext` LCM path, wait for completion, and return token counts.

- [x] **Step 4: Register and render `/compact`**

  Add the dispatch arm, autocomplete label, and help text. Print a short progress/result message while preserving the existing terminal suspend/resume behavior.

- [x] **Step 5: Run focused command and compaction tests**

  Run: `cargo test --release repl::commands -- --test-threads=1`

  Run: `cargo test --release agent_loop -- --test-threads=1`

  Expected: PASS.

### Task 4: Full verification and graph review

**Files:**
- No additional production files.

- [x] **Step 1: Run full release build and tests**

  Run: `cargo build --release`

  Run: `cargo test --release`

- [x] **Step 2: Check the diff**

  Run: `git diff --check`

- [x] **Step 3: Analyze graph changes before committing**

  Run: `npx gitnexus detect-changes --scope staged`

  Expected: no partial/truncated result; review the affected compaction and REPL flows.

- [x] **Step 4: Commit only the implementation files and this plan**

  Stage the files listed above plus this plan, leaving all pre-existing dirty files unstaged, then create one focused commit.
