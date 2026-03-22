# Opportunity 2: Capability-Verified Fallback Graph

Fresh read: our fallback chain is optimistic. It chooses another backend before proving that backend can actually serve the selected model.

## LEAN proof

- Chrome traces `~/.nanobot/traces/nanobot-20260312-112716.json` and `~/.nanobot/traces/nanobot-20260312-113026.json` show the same chain:
  - `vllm-mlx server failed to start`
  - `in-process model load failed: Weight not found ...`
  - `llm_stream_call_failed`
- Rotated logs add scale:
  - 7 `vllm-mlx server failed to start`
  - 49 `in-process model load failed: Weight not found: ...`
  - 9 `in-process model not loaded`
  - 3 `ANE train: failed to load weights: missing tensor: model.embed_tokens`
  - 4 `perplexity_gate: ANE training failed, experiences NOT marked exported`
- `~/.nanobot/vllm-mlx.log` is full of missing or renamed tensor keys, so this is compatibility drift, not random flakiness.
- `src/providers/mlx.rs:216-242` always falls back to in-process.
- `src/agent/mlx_server.rs:282-287` turns unsupported model loads into `None`, which delays failure until a user turn.
- `src/agent/agent_shared.rs:1393-1406` discovers the broken state only while serving a request.

### Smallest confirming experiment

- Probe inference, training, perplexity, and ANE support once at backend startup.
- If no valid fallback exists, reject the backend immediately instead of entering the loop.

### Success signal

- Unsupported models fail once at startup and never produce later `in-process model not loaded` errors.

## First draft implementation

1. Add a per-model `CapabilityMatrix` covering:
   - `stream_inference`
   - `train_lora`
   - `perplexity_gate`
   - `ane_train`
   - `reflection_safe`
2. Build the matrix during MLX startup in `src/providers/mlx.rs`:
   - try managed server start
   - if that fails, run a lightweight in-process tensor probe before accepting fallback
   - if both fail, return `backend_unavailable`
3. Change `run_model_worker` in `src/agent/mlx_server.rs` to return a typed availability result instead of silently storing `None`.
4. Add tensor alias support in the MLX loader path for known schema drift:
   - `language_model.model.*`
   - `model.*`
   - split vs fused MLP projections
   - q/k norm variants
5. Gate ANE training in `src/agent/ane_mlx_bridge.rs` on the same capability matrix so unsupported models are skipped, not attempted.
6. Persist probe results under `~/.nanobot/cache/model_capabilities.json` and invalidate on model file mtime change.
