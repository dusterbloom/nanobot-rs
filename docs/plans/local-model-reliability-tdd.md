# Local Reliability Plan (Red -> Green)

## Goal
Resolve the current failure clusters and guarantee nanobot can run in `local-only` mode without cloud provider coupling.

## Root Causes and Fix Plan

1. Archive/move failures (`EXDEV`) in memory pipeline
- Problem: `fs::rename` across directories fails in this runtime for archive paths.
- Files: `src/agent/observer.rs`, `src/agent/working_memory.rs`, `src/agent/reflector.rs`.
- Plan:
  - Add a shared file-move helper with fallback (`rename` then `copy+remove` on cross-device errors).
  - Use it in both observation and working-memory archive paths.
  - Stop swallowing observer archive errors when correctness depends on a successful move.

2. Web search test depends on process env
- Problem: `test_web_search_no_api_key` fails when `BRAVE_API_KEY` is set in test environment.
- File: `src/agent/tools/web.rs`.
- Plan:
  - Make test hermetic (temporarily clear env var and restore it).
  - Keep behavior contract: empty runtime key should return `"BRAVE_API_KEY not configured"`.

3. Alias regression (`/prov` not mapped)
- Problem: test expects `/prov -> /provenance`, implementation only maps `/p`.
- File: `src/repl/commands.rs`.
- Plan:
  - Align implementation with test contract (support `/prov` alias).
  - Keep `/p` behavior explicit.

4. Socket-bind tests fail under restricted runtime
- Problem: server tests assume bind permission unconditionally.
- File: `src/server.rs`.
- Plan:
  - Add a small test-only capability probe for localhost bind.
  - Skip bind-dependent tests when the environment forbids sockets.

5. Provider selection mismatch for `"none"` sentinel values
- Problem: `get_api_key()` treats `"none"` as disabled, but `get_api_base()` does not for higher-priority providers.
- File: `src/config/schema.rs`.
- Plan:
  - Make `get_api_base()` use the same disabled-key predicate as `get_api_key()`.
  - Add regression test for local vLLM selection when cloud providers are explicitly disabled.

## TDD Matrix

### Red phase (must fail before fixes)
Use:

```bash
./scripts/tdd_local_models_only.sh red
```

Expected failing contracts:
- alias normalization (`/prov`)
- web no-key path under env contamination
- archive move path
- reflector archive completion path
- bind-dependent server path
- local-only provider/base mismatch with `"none"` sentinel

### Green phase (must pass after fixes)
Use:

```bash
./scripts/tdd_local_models_only.sh green
```

Green criteria:
- all root-cause regressions pass
- local-only wiring tests pass:
  - `cli::tests::test_build_core_handle_local_forces_local_provider_even_with_cloud_keys`
  - `cli::tests::test_make_eval_provider_local_uses_local_endpoint`
  - `cli::tests::test_eval_model_name_local_is_port_scoped`
  - `config::schema::tests::test_local_vllm_provider_selected_when_cloud_disabled`

## Local-only Acceptance Contract

When local mode is enabled:
- core provider base must be `http://localhost:<port>/v1`
- model label must be `local:<model>`
- cloud keys configured in config must not alter provider wiring for local mode
- eval local mode must use local endpoint and not require cloud API keys
