# Higgs Safe Model Catalog Design

## Goal

Nanobot's `/model` picker displays every locally installed model that Higgs can select through a compatible loader, exactly once, under a stable correct name.

## Authority

Higgs owns model compatibility and naming. Nanobot must not duplicate Higgs architecture, quantization, or filesystem-layout rules.

Higgs exposes an authenticated `GET /v1/models/available` catalog. Each record contains:

- `id`: stable display and selection name;
- `path`: canonical local artifact path used for switching;
- `model_type`: detected effective model type;
- `adapter`: resolved Higgs loader adapter;
- `loaded`: whether this exact canonical artifact is resident.

The catalog is returned only from metadata-compatible artifacts. A candidate must have a readable bounded `config.json`, tokenizer metadata, a safetensors payload, and pass `higgs_models::adapter::detect` plus `adapter::resolve`. This is a zero-weight compatibility guarantee, not a promise that current free memory can hold the model; runtime admission remains authoritative for memory.

## Discovery and identity

Higgs scans its configured model paths, their recognized Hugging Face cache root, and the standard Hugging Face and LM Studio roots. Paths are canonicalized before deduplication. Symlinks and repeated roots therefore produce one record.

Names are derived in this order:

1. explicit active/configured alias;
2. useful `config.json` `_name_or_path` identity;
3. Hugging Face cache `publisher/model` identity;
4. LM Studio `publisher/model[/variant]` identity;
5. directory leaf as the final fallback.

The same naming function is used for catalog entries and runtime-loaded models. A nested variant such as `LiquidAI/LFM2.5-2.6B-MLX/8bit` must never appear merely as `8bit`.

## Nanobot behavior

When Higgs advertises runtime switching, Nanobot requests `/v1/models/available` and maps each record directly to one picker entry. It deduplicates defensively by canonical path, and uses the server's `id` without substituting `localModel`, `lmsMainModel`, or `mlxModelDir` names.

For older Higgs servers without the endpoint, Nanobot lists only `/v1/models` resident models. It does not fall back to unsafe filesystem guessing.

## Tests and live acceptance

Red-to-green tests cover supported Nanbeige discovery, unsupported/malformed artifact rejection, canonical-path deduplication, Hugging Face and nested LM Studio naming, loaded identity, Nanobot catalog parsing, and picker deduplication.

Live acceptance on this machine requires:

- Nanbeige appears once under its Nanbeige name;
- the resident Ternary Bonsai model appears once and loaded;
- no orphan `8bit` entry appears;
- audio, embedding, diffusion, drafter-only, malformed, and unsupported artifacts do not appear;
- selecting a catalog entry sends its exact catalog path/name to Higgs.
