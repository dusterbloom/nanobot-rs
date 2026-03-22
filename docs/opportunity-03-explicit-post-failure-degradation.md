# Opportunity 3: Explicit Post-Failure Degradation

Fresh read: once the primary LLM path is unhealthy, we still run optional subsystems and create secondary failures that blur the real cause.

## LEAN proof

- `~/.nanobot/logs/nanobot.log.*` contains 26 `Exit reflection failed` warnings.
- The same logs contain 4 `perplexity_gate: ANE training failed, experiences NOT marked exported` warnings.
- Trace `~/.nanobot/traces/nanobot-20260312-120554.json` shows a connection failure to `http://localhost:8000/v1/chat/completions` followed by `Exit reflection failed` in the same run.
- `src/repl/mod.rs:2004-2029` always attempts exit reflection whenever memory is enabled.
- `src/agent/learn_loop.rs:425-444` schedules ANE export work and only then records failure.
- These are secondary failures. They do not explain the original outage, but they do make the system harder to reason about.

### Smallest confirming experiment

- After the first provider or backend failure in a run, set a degraded state and skip optional work that depends on the broken capability.

### Success signal

- A broken run produces one primary error and zero trailing reflection or export failures.

## First draft implementation

1. Add a run-scoped `CapabilityState` or `DegradedMode` to `SwappableCore` in `src/agent/agent_core.rs`.
2. When `llm_stream_call_failed` fires in `src/agent/agent_shared.rs`, set the degraded reason:
   - `main_provider_down`
   - `backend_incompatible`
   - `auth_invalid`
   - `warmup_failed`
3. In `src/repl/mod.rs`, skip exit reflection unless `reflection_safe` is true.
4. In `src/agent/learn_loop.rs`, do not spawn ANE or export work unless `ane_train` and `perplexity_gate` are healthy.
5. Record skipped work once per run:
   - `exit_reflection_skipped_due_to_degraded_provider`
   - `ane_export_skipped_due_to_incompatible_model`
6. Clear degraded mode only after a successful preflight or warmup, not after a single retry.
