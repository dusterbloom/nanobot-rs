# Local Reliability Regression Plan (Red -> Green)

> The red phase is for a pre-fix checkout. On the current tree, run the green
> phase; every filter is exact, the script uses an isolated temporary HOME, and
> it fails if a named test is missing.

## Goal
Keep local-only startup independent from cloud providers while preserving the
canonical SQLite working-memory and reflection path.

## Root Causes and Fix Plan

1. Session memory lifecycle
- Problem: the former observer and `SESSION_*.md` archive checks described a
  storage path that no longer exists.
- Files: `src/session/db.rs`, `src/agent/working_memory.rs`,
  `src/agent/reflector.rs`.
- Contract:
  - Session-scoped working memory lives in SQLite.
  - Reflection reads completed rows, updates `MEMORY.md`, and marks those rows
    reflected without recreating per-session Markdown files.

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

6. Local discovery and spawn authority
- Problem: endpoint identity, model identity, and spawn policy must not drift
  between Higgs and LM Studio.
- Files: `src/local_discovery.rs`, `src/higgs.rs`.
- Contract:
  - Discovery adopts endpoint and model together.
  - `localAutostart: "off"` never spawns; explicit Higgs autostart does.
  - Higgs and LM Studio use distinct configured ports, with LM Studio on 1234
    by default.
  - The compaction sidecar only gains spawn authority from explicit Higgs
    autostart, never from LM Studio autostart.

## TDD Matrix

### Red phase (must fail before fixes)
Use:

```bash
./scripts/tdd_local_models_only.sh red
```

Expected failing contracts:
- alias normalization (`/prov`)
- web no-key path under env contamination
- SQLite working-memory lifecycle
- reflector completion/reflection lifecycle
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
  - `config::schema::tests::test_local_vllm_provider_selected_when_cloud_disabled`
  - `local_discovery::tests::test_decide_no_server_and_autostart_off_is_note_not_spawn`
  - `local_discovery::tests::test_decide_no_server_spawns_only_with_explicit_autostart`
  - `local_discovery::tests::test_candidates_cover_configured_higgs_lms_and_cluster`
  - `higgs::tests::compaction_manager_respects_explicit_higgs_autostart`
  - `higgs::tests::compaction_manager_never_spawns_for_lmstudio_autostart`

## Local-only Acceptance Contract

When local mode is enabled:
- endpoint and served model are adopted as one discovery result
- a healthy explicit `localApiBase` wins, followed by Higgs and then LM Studio
- Higgs uses `higgsPort`; LM Studio uses `lmsPort` (default 1234)
- `localAutostart: "off"` does not spawn a server
- `localAutostart: "higgs"` authorizes Higgs spawning
- model label remains `local:<model>`
- cloud keys configured in config must not alter provider wiring for local mode
- compaction-sidecar spawning is not authorized by LM Studio autostart
