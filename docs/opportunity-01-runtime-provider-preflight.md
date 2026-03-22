# Opportunity 1: Runtime Provider Preflight

Fresh read: many of the failures we treated as "model quality" or "context" problems are failing before the real task even starts.

## LEAN proof

- `~/.nanobot/nanobot.log` contains 72 repeated local-model 404s caused by invalid model ids plus 2 auth failures.
- `~/.nanobot/logs/nanobot.log.2026-03-04` contains 2 retries against a server that explicitly said no models were loaded.
- `~/.nanobot/sessions/cli_default_2026-02-18.jsonl` shows 7 user-visible provider 404 failures in one session.
- `~/.nanobot/sessions/cli_default_2026-02-27.jsonl` shows the "short context" investigation ending in a 401 before any real diagnosis happened.
- We already wrote the right check in tests: `src/agent/agent_loop_tests.rs:1082-1165` probes `/v1/models` and warms up all trio roles.
- Runtime construction in `src/cli/core_builder.rs:159-164`, `src/cli/core_builder.rs:195-232`, and `src/providers/factory.rs:90-113` creates providers but never runs that preflight.

### Smallest confirming experiment

- Run the existing preflight logic before the first turn for every local or trio provider.
- Expected result on the bad setups above: one startup failure, zero user-visible turn failures, zero retries against unloaded or unauthorized servers.

### Success signal

- Same broken config produces one structured startup error and no downstream `llm_stream_call_failed` for the same root cause.

## First draft implementation

1. Add `src/providers/preflight.rs` with `ProviderPreflightReport`.
2. For OpenAI-compatible local providers, perform:
   - `GET /v1/models`
   - auth/header validation
   - requested-model lookup using stripped aliases
   - a 1-token warmup chat request
3. Call preflight from:
   - `build_core_handle`
   - `build_core_handle_mlx`
   - `rebuild_core`
   - `rebuild_core_mlx`
4. Extend `ProviderSpec` in `src/providers/factory.rs` so preflight receives the exact expected model id.
5. Cache successful reports by `(api_base, model)` with a short TTL so the REPL stays fast.
6. Return a typed startup error instead of entering the loop:
   - `model_not_available`
   - `auth_rejected`
   - `server_reachable_but_empty`
   - `warmup_failed`
7. Promote the helper logic from `src/agent/agent_loop_tests.rs:1082-1165` into reusable runtime code and keep the end-to-end test as the regression harness.
