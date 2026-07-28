# Structural LCM and Proxy Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make proxy calls semantically canonical, derive LCM pressure only from the active model context, and run LCM on the active main model with no compaction sidecar path.

**Architecture:** `ToolRegistry` becomes the single proxy-envelope decoder used by both routing and execution. `manage_compaction` uses only `TokenBudget::available_budget`, while `SwappableCore` owns a main-model compactor and a separately named memory provider/model for reflection. The optional compaction provider, sidecar manager, lease lifecycle, and obsolete configuration are removed.

**Tech Stack:** Rust 2021, Tokio, serde/serde_json, existing `ToolRegistry`, `TokenBudget`, `ContextCompactor`, and Rust unit/integration tests.

## Global Constraints

- Preserve the production path: channel → agent_loop → provider → tools → reply.
- Maintain one production path; do not retain sidecar or retained-cap alternatives behind flags.
- Run GitNexus upstream impact before editing every symbol and warn before any HIGH or CRITICAL edit.
- Preserve existing user changes in `budget.rs`, `shared.rs`, `repl/mod.rs`, and other dirty files.
- Use `cargo build`, `cargo test`, and `scripts/turn_bench.sh` for final verification.
- Do not commit implementation files from this dirty worktree; several required files contain pre-existing user edits. Run GitNexus `detect_changes` and leave the reviewed diff unstaged.

---

### Task 1: Canonical Proxy Resolution

**Files:**
- Modify: `src/agent/tools/registry.rs`
- Modify: `src/agent/router.rs`

**Interfaces:**
- Produces: private `ProxyCall` resolution variants for catalog, inspect, dispatch, missing selector, and invalid arguments.
- Produces: `ToolRegistry::canonical_proxy_dispatch(&self, outer_name: &str, params: &HashMap<String, Value>) -> Option<(String, HashMap<String, Value>)>`.
- Consumes: both current `tool_name` / `tool_args` and legacy `name` / `args` envelopes, including object, JSON-string, and validated flattened arguments.

- [x] **Step 1: Add the failing current-envelope routing test**

Add beside the existing proxy canonicalization tests in `src/agent/router.rs`:

```rust
#[test]
fn test_canonicalize_proxy_execution_current_envelope() {
    let registry = ToolRegistry::new();
    let tc = canonicalize_proxy_execution(
        &registry,
        ToolCallRequest {
            id: "tc_proxy_web".to_string(),
            name: "tool".to_string(),
            arguments: HashMap::from([
                ("tool_name".to_string(), json!("web_fetch")),
                (
                    "tool_args".to_string(),
                    json!({"url": "https://example.com"}),
                ),
            ]),
        },
    );

    assert_eq!(tc.name, "web_fetch");
    assert_eq!(
        tc.arguments.get("url"),
        Some(&json!("https://example.com"))
    );
}
```

The production change this catches is restoring a router-only legacy decoder
that leaves current proxy calls named `tool`.

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
cargo test --lib agent::router::tests::test_canonicalize_proxy_execution_current_envelope -- --exact
```

Expected: compile failure because `canonicalize_proxy_execution` does not accept
the registry yet, or assertion failure because `tool_name` is not decoded.

- [x] **Step 3: Move all envelope interpretation into `ToolRegistry`**

In `src/agent/tools/registry.rs`, add a private resolution enum beside
`ToolRegistry`:

```rust
enum ProxyCall {
    Catalog,
    Inspect { tool_name: String },
    Dispatch {
        tool_name: String,
        arguments: HashMap<String, Value>,
    },
    MissingSelector,
    InvalidArguments,
}
```

Implement one private resolver that:

1. prefers `tool_name` over legacy `name`;
2. prefers `tool_args` over legacy `args`;
3. accepts an object or a JSON string containing an object;
4. uses `flattened_proxy_args_for_dispatch` only when no envelope exists;
5. returns `Inspect` when a valid selector has no dispatch arguments;
6. returns `MissingSelector` when invocation intent exists without a selector;
7. returns `Catalog` for the empty discovery call; and
8. returns `InvalidArguments` for a non-object envelope.

Expose only the canonical dispatch projection:

```rust
pub(crate) fn canonical_proxy_dispatch(
    &self,
    outer_name: &str,
    params: &HashMap<String, Value>,
) -> Option<(String, HashMap<String, Value>)> {
    if outer_name != "tool" {
        return Some((outer_name.to_string(), params.clone()));
    }
    match self.resolve_proxy_call(params) {
        ProxyCall::Dispatch {
            tool_name,
            arguments,
        } => Some((tool_name, arguments)),
        _ => None,
    }
}
```

Rewrite `execute_proxy` as a match on the same resolution enum. Preserve the
existing catalog, schema inspection, missing-selector error, write-file staged
piece limit, and `execute_inner` behavior.

- [x] **Step 4: Make routing consume the registry projection**

Change the router helper to:

```rust
fn canonicalize_proxy_execution(
    registry: &ToolRegistry,
    mut tc: ToolCallRequest,
) -> ToolCallRequest {
    let Some((name, arguments)) =
        registry.canonical_proxy_dispatch(&tc.name, &tc.arguments)
    else {
        return tc;
    };
    tc.name = name;
    tc.arguments = arguments;
    tc
}
```

Update the routing iterator to call it through `ctx.tools`, and pass a
`ToolRegistry` to the existing legacy, flattened, inspect, and JSON-string unit
tests.

- [x] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
cargo test --lib agent::router::tests::test_canonicalize_proxy_execution -- --nocapture
cargo test --lib agent::tools::registry::tests::proxy_ -- --nocapture
```

Expected: all matching proxy routing and execution tests pass.

---

### Task 2: Model-Context-Only LCM Pressure

**Files:**
- Modify: `src/agent/agent_loop/budget.rs`
- Modify: `src/agent/agent_loop/shared.rs`
- Modify: `src/agent/agent_core.rs`
- Modify: `src/agent/agent_loop/response.rs`

**Interfaces:**
- Produces: `should_allow_checkpoint(pressure: f32, tau_hard: f64) -> bool`.
- Consumes: `TokenBudget::available_budget(tool_def_tokens)` as the only LCM denominator.
- Removes: retained-cap constants, environment overrides, admission structures, prompt calibration, retained pressure, and effective-budget wrappers.

- [x] **Step 1: Reverse the retained-pressure regression test**

Replace the retained-pressure checkpoint test in `shared.rs` with:

```rust
#[test]
fn checkpoint_is_not_forced_before_model_context_pressure() {
    assert!(!should_allow_checkpoint(0.40, 0.85, Some(0.99)));
}
```

The production change this catches is reintroducing transport-cache capacity as
a semantic compaction trigger.

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
cargo test --lib agent::agent_loop::shared::cache_pressure_tests::checkpoint_is_not_forced_before_model_context_pressure -- --exact
```

Expected: assertion failure because current retained pressure forces the
checkpoint.

- [x] **Step 3: Remove retained-session admission from LCM**

In `budget.rs`:

- delete `DEFAULT_HIGGS_RETAINED_SESSION_CAP_TOKENS` and all retained admission constants;
- delete `HiggsRetainedAdmission`;
- delete `system_token_count`, `higgs_retained_session_cap_tokens`,
  `higgs_retained_admission_ratio`, `calibrated_higgs_prompt_tokens`,
  `higgs_retained_admission`, `retained_conversation_available`,
  `effective_lcm_available_budget`, and `retained_context_pressure`;
- simplify checkpoint policy to:

```rust
pub(super) fn should_allow_checkpoint(pressure: f32, tau_hard: f64) -> bool {
    (pressure as f64) >= tau_hard
}
```

Update the module comment to describe token accounting and cache invalidation,
not retained-session admission.

In `shared.rs`, calculate:

```rust
let available = budget.available_budget(tool_def_tokens);
let action = engine.check_thresholds_with_available(available);
let hard_limit = (available as f64 * engine.tau_hard()) as usize;
let soft_limit = (available as f64 * engine.tau_soft()) as usize;
let raw_hard = conversation_token_count(&ctx.messages) > hard_limit;
```

Remove every retained-cap branch and log field. At pending-checkpoint install,
pass only model context pressure to `should_allow_checkpoint`.

- [x] **Step 4: Remove now-dead prompt calibration**

In `agent_core.rs`, delete `PromptCalibration`, the
`prompt_calibrations` counter field, initialization/reset logic,
`record_prompt_calibration`, `prompt_calibration`, and its unit test.

In `response.rs`, keep the public actual/estimated prompt telemetry atomics but
remove the per-session `record_prompt_calibration` call.

- [x] **Step 5: Replace retained tests with context-authority tests**

Keep the existing `checkpoint_deferred_below_tau_hard`,
`checkpoint_forced_at_tau_hard_boundary`, and
`checkpoint_forced_above_tau_hard` tests, updating them to the two-argument
signature. Delete tests whose only subject is retained capacity or calibration.

- [x] **Step 6: Run focused tests and verify GREEN**

Run:

```bash
cargo test --lib agent::agent_loop::shared::cache_pressure_tests -- --nocapture
cargo test --lib agent::agent_core::tests -- --nocapture
```

Expected: context-pressure and runtime-counter tests pass with no retained LCM
policy remaining.

---

### Task 3: Main-Model LCM and Sidecar Removal

**Files:**
- Modify: `src/agent/agent_core.rs`
- Modify: `src/agent/compaction.rs`
- Modify: `src/agent/agent_loop/compaction.rs`
- Modify: `src/cli/core_builder.rs`
- Modify: `src/agent/agent_loop/mod.rs`
- Modify: `src/repl/commands/mutation.rs`
- Modify: `src/cli/mod.rs`
- Modify: `src/repl/mod.rs`
- Modify: `src/config/schema.rs`
- Modify: `src/heartbeat/health.rs`
- Modify: `src/higgs.rs`
- Modify: `src/agent/agent_loop/tests.rs`

**Interfaces:**
- Produces: `SwappableCore::memory_model: String`, paired with
  `memory_provider` for reflection.
- Produces: `ContextCompactor::new(provider.clone(), model.clone(), max_context_tokens)`
  from the main core provider/model only.
- Removes: `SwappableCoreConfig::compaction_provider`,
  `SwappableCoreConfig::compaction_manager`, `SwappableCore::compaction_manager`,
  `LocalProviders` compaction fields, sidecar config schema, manager, leases,
  acquisition timeouts, and shutdown paths.

- [x] **Step 1: Add a failing provider-separation regression**

Add an async LCM test using two `WireRecordingProvider` instances:

```rust
#[tokio::test]
async fn lcm_compaction_uses_main_provider_not_memory_provider() {
    let main = Arc::new(WireRecordingProvider::new(
        "main",
        vec![
            WireRecordingProvider::text_response("foreground reply"),
            WireRecordingProvider::text_response("- faithful summary"),
        ],
    ));
    let memory = Arc::new(WireRecordingProvider::new(
        "memory",
        vec![WireRecordingProvider::text_response("- memory summary")],
    ));
    // Build with tau_soft low, tau_hard above 1.0, and memory as the specialist
    // fallback. Persist enough padded turns to exceed MIN_COMPACTION_TOKENS,
    // call process_direct, then wait for the soft job.
    assert!(main.calls().len() > 1);
    assert!(memory.calls().is_empty());
}
```

The production change this catches is constructing LCM's `ContextCompactor`
from the memory/specialist provider instead of the active main provider.

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
cargo test --lib agent::agent_loop::tests::lcm_compaction_uses_main_provider_not_memory_provider -- --exact --nocapture
```

Expected: timeout/assertion failure because current LCM sends its summary to the
memory provider.

- [x] **Step 3: Separate compaction and reflection identities in the core**

In `agent_core.rs`:

- add `pub memory_model: String` beside `memory_provider`;
- remove both compaction inputs from `SwappableCoreConfig`;
- remove compaction-provider precedence from `resolve_memory_provider`;
- build `ContextCompactor` from the main `provider`, `model`, and
  `max_context_tokens`;
- store the resolved `memory_model` separately.

In `compaction.rs`, update the struct comment to say active main endpoint and
remove the now-unused `for_model` clone helper.

- [x] **Step 4: Collapse LCM execution to the main compactor**

In `agent_loop/compaction.rs`, delete `main_provider_compactor`,
`COMPACTION_ACQUIRE_TIMEOUT`, sidecar imports, acquisition, fallback, and lease
release. Begin `execute_lcm_compaction` with:

```rust
let resolved_compactor = core.compactor.clone();
```

Preserve the engine lock, `LcmCompactionMutation`, summary escalation, SQLite
checkpoint, rollback, and statistics unchanged.

- [x] **Step 5: Remove sidecar provider construction**

In `cli/core_builder.rs`:

- remove `make_managed_compaction_provider` and
  `normalize_memory_model_for_sidecar`;
- remove compaction fields from `LocalProviders`;
- remove compaction arguments from `core_config_from`;
- pass `config.memory.clone()` unchanged;
- build and rebuild only main, delegation, and specialist providers.

Update all `SwappableCoreConfig` initializers in tests to remove the two deleted
fields.

- [x] **Step 6: Make every reflection caller direct**

For background reflection, cron reflection, `/learn`, and exit reflection,
construct:

```rust
Reflector::new(
    core.memory_provider.clone(),
    core.memory_model.clone(),
    &core.workspace,
    threshold,
    core.sessions.clone(),
)
```

Remove manager checks, acquisition/release, sidecar-specific warnings, and
shutdown. Preserve existing reflection thresholds, error reporting, and exit
timeout.

- [x] **Step 7: Delete the sidecar surface**

In `config/schema.rs`, remove:

- `AgentsDefaults::{higgs_compaction_port, higgs_compaction_model_dir}`;
- `LcmSchemaConfig::{compaction_model_dir, compaction_port}`;
- corresponding defaults and tests.

Old JSON keys remain harmless unknown fields under existing serde behavior.

In `higgs.rs`, delete `CompactionSidecarSpec`, state, registry,
`mlx_model_max_context`, manager, acquisition guard, lease, retry helpers, model
resolution wait helper, and their tests. Retain main Higgs process management,
runtime model discovery/switching, `model_id_matches`, and served-model helpers.
Remove now-unused `HashMap`, `Arc`, `OnceLock`, and `Weak` imports.

Delete the obsolete health-registry test that configures an on-demand
compactor, and remove sidecar-specific keepalive comments.

- [x] **Step 8: Update and run focused tests**

Update memory-provider tests to assert:

```rust
assert_eq!(core.compactor.model(), "main-model");
assert_eq!(core.memory_model, "haiku-or-explicit-memory-model");
```

Delete tests whose contract is sidecar lifecycle. Run:

```bash
cargo test --lib agent::agent_loop::tests::lcm_compaction_uses_main_provider_not_memory_provider -- --exact --nocapture
cargo test --lib agent::agent_loop::tests -- --nocapture
cargo test --lib config::schema::tests -- --nocapture
cargo test --lib higgs::tests -- --nocapture
```

Expected: all focused tests pass and no source reference to
`CompactionSidecarManager`, `compaction_manager`, `compaction_provider`,
`compactionPort`, or `compactionModelDir` remains.

---

### Task 4: Full Verification and Scope Audit

**Files:**
- Update checkboxes in this plan only.

**Interfaces:**
- Consumes: completed Tasks 1–3.
- Produces: fresh build, test, benchmark, and GitNexus evidence.

- [x] **Step 1: Format and inspect**

Run:

```bash
cargo fmt --all -- --check
git diff --check
rg -n "CompactionSidecarManager|compaction_manager|compaction_provider|compactionPort|compactionModelDir|DEFAULT_HIGGS_RETAINED_SESSION_CAP_TOKENS|higgs_retained_admission|prompt_calibration" src
```

Expected: formatting and whitespace checks succeed; the symbol search is empty.

Actual: every repair-owned Rust file is formatted and `git diff --check`
passes. The repository-wide format check still reports only pre-existing/user
formatting in `finalize_response.rs`, `protocol.rs`, `tool_engine.rs`,
`tui_app/app.rs`, and `turn_stream.rs`; those files were intentionally not
rewritten. Removed runtime symbols are absent. Legacy compaction JSON spellings
remain only in the schema regression proving they deserialize as ignored
unknown fields and are never serialized.

- [x] **Step 2: Run the complete test and build tracks**

Run:

```bash
cargo test
cargo build
```

Expected: both commands exit zero.

Actual: `cargo build` and `cargo build --release` pass. The final library run is
green (`2602 passed, 23 ignored`). The complete track passes the library and all
task-related integration suites; its sole failure is the unchanged baseline
`tests/protocol_invariants.rs::local_last_message_is_user_after_tool_results`
(`tool` versus expected `user`), reproduced before this implementation.

- [x] **Step 3: Run the matched turn benchmark**

Run:

```bash
scripts/turn_bench.sh
```

Expected: the correctness track passes and no material turn-speed regression is
reported.

Actual: after rebuilding release, 20/20 turns succeeded. Cold elapsed was
16.425s; warm elapsed median 1.035s and p95 1.306s; warm TTFT median 661ms and
p95 1.016s. Context preparation median was 49ms. The run is within the
checked-in warm total envelope; model identities differ from the checked-in
baseline, so no cross-model quality claim is made.

- [x] **Step 4: Run GitNexus change detection**

Run:

```bash
node .gitnexus/run.cjs detect-changes --repo nanobot-rs
node .gitnexus/run.cjs detect-changes --repo nanobot-rs --scope compare --base-ref main
```

If the installed CLI lacks `detect-changes`, use the equivalent GitNexus MCP
tool. Inspect every changed symbol and affected flow; do not treat command
availability as verification.

Actual: the CLI lacked this subcommand, so both MCP scopes were run. Worktree
scope reports 418 changed and 36 affected symbols across 23 dirty files;
compare-to-main reports 2150 changed and 154 affected symbols across the
long-lived branch. Both are CRITICAL because core agent/LCM flows are affected,
as expected and warned before edits. Scope includes unrelated pre-existing user
changes; the repair audit was performed against the task-owned diff.

- [x] **Step 5: Audit the specification requirement by requirement**

Verify directly that:

1. routing and execution call the same proxy resolver;
2. LCM thresholds and checkpoint installation have no retained-cap input;
3. LCM's compactor is constructed from main provider/model/context;
4. reflection has a separate memory provider/model pair;
5. no sidecar runtime/config path remains; and
6. cancellation-safe LCM rollback and SQLite checkpointing are unchanged.

Leave implementation changes unstaged for user review because required files
overlap pre-existing user edits.

Actual: all six invariants hold. `ToolRegistry::resolve_proxy_call` feeds both
routing and execution; thresholds consume only `TokenBudget`; the compactor is
constructed from foreground provider/model/context; reflection alone owns the
memory provider/model pair; sidecar runtime/config code is deleted; and the
same mutation rollback plus atomic SQLite checkpoint path remains covered by
passing regression tests. Implementation changes remain unstaged.
