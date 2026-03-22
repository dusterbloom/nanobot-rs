# Opportunity 4: Always-On Failure Envelope

Fresh read: we do not have one reliable artifact that links a failed user turn to the backend failure that caused it.

## LEAN proof

- `~/.nanobot/traces` currently has 5 chrome trace `.json` files and 1 of them is zero bytes.
- The same directory has 0 router decision `.jsonl` files, even though `src/agent/trace_store.rs:86-135` can write them.
- Router trace emission is config-gated: `TrioConfig.trace_log` defaults to `false` in `src/config/schema.rs:1001` and `src/config/schema.rs:1078`.
- Chrome traces are written separately in `src/main.rs:500-585`, so we already have two uncorrelated trace channels.
- Session logs, app logs, and trace files do not share a stable `run_id`, which makes cross-artifact diagnosis slower than it should be.

### Smallest confirming experiment

- Write one tiny JSONL failure envelope on every failed run, regardless of full tracing settings.
- Fields: `run_id`, `session_key`, `session_id`, `model`, `api_base`, `phase`, `error_class`, `error`, `ts`.

### Success signal

- Every reproduced failure yields exactly one grep-able envelope even if full trace capture is disabled or chrome trace flushing fails.

## First draft implementation

1. Add `src/agent/run_audit.rs` with an always-on `append_failure_envelope()` helper.
2. Generate a `run_id` at turn or session entry and thread it through:
   - session metadata
   - provider logs
   - router traces
   - chrome trace filenames
3. Use `src/agent/trace_store.rs` for structured JSONL and reserve chrome traces for deep profiling only.
4. On failure paths in:
   - `src/agent/agent_shared.rs`
   - `src/providers/openai_compat.rs`
   - `src/providers/mlx.rs`
   - `src/repl/mod.rs`
   append one normalized failure envelope.
5. Keep full router decision traces behind `trace_log`, but make the minimal failure envelope unconditional.
6. Add a startup warning when chrome tracing is enabled and the previous trace file was empty, so silent trace loss becomes visible.
