# ANE Generative UI — Implementation Plan

## Thesis

ANE is a stateless dispatch surface. Two compiled MIL programs (draft decoder, diffusion UI generator) share it via preemptive scheduling. A 0.5B diffusion model generates structured component JSON — not raw HTML — which a deterministic renderer expands. EGGROLL trains on user feedback during idle time. Shared components become both artifacts and training data.

## Architecture

```
                    ┌─────────────────────┐
                    │   ANE Task Queue    │
                    │  (preemptive, ~30MB │
                    │   per kernel set)   │
                    └────┬───────────┬────┘
                         │           │
              ┌──────────▼──┐  ┌─────▼──────────┐
              │ Draft Decode │  │ Diffusion UI   │
              │ (priority)   │  │ (background)   │
              │ <10ms budget │  │ yield-per-step │
              └──────────────┘  └────────────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │ Component JSON   │
                              │ (~20 types)      │
                              └────────┬─────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │ Renderer (Rust)  │
                              │ JSON → HTML/CSS  │
                              └──────────────────┘
```

## Phases

### Phase 1 — ANE scheduler (higgs)

**Goal:** Preemptive dual-mode ANE dispatch.

- Add `AneTaskQueue` to higgs engine — priority lane (draft decode) preempts background lane (diffusion)
- Make `DiffusionEngine::generate()` yield-able: check between denoising steps, pause/resume via `(canvas: Vec<u32>, step: usize)` state
- Both kernel sets stay compiled in memory (~60MB total); switching is just calling `ane_bridge_eval` on a different handle

**Where:** `crates/higgs-engine/src/ane_scheduler.rs` (new), modify `crates/higgs-models/src/diffusion.rs`

**Constraint:** Draft decode latency must stay <10ms p99. Diffusion tolerates 500ms+ interruptions — each denoising step is independent.

### Phase 2 — Component schema + renderer (nanobot)

**Goal:** Structured generation that a 0.5B model can reliably produce.

- Define ~20 component types as a JSON schema: `card`, `metric`, `chart`, `table`, `form`, `list`, `code`, `image`, `nav`, `alert`, `progress`, `toggle`, `input`, `select`, `tabs`, `accordion`, `timeline`, `badge`, `avatar`, `dialog`
- Diffusion model generates JSON with masked scaffolding — structural tokens (braces, keys, type names) are prefilled, values are masked
- Deterministic Rust renderer expands JSON → HTML/CSS using a bundled design system (single CSS file, no JS framework)

**Where:** `src/agent/tools/generate_ui.rs` (new tool), `src/ui/schema.rs`, `src/ui/render.rs`

**Key insight:** The model's search space is ~20 type names + natural language values. Not arbitrary HTML. This is why 0.5B works.

### Phase 3 — Weight loading + validation (higgs)

**Goal:** Load `dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1` into existing diffusion engine.

- Map Qwen2.5 config → `DiffusionConfig` (dim=896, layers=24, heads=14, kv_heads=2, hd=64, inter=4864, vocab=151936)
- Adjust `diffusion_ane.rs` MIL generation for Qwen2.5 GQA ratio (14:2 vs Qwen3's 16:8)
- Validate: prefill component JSON skeleton → mask values → denoise → assert valid JSON output

**Where:** Modify `crates/higgs-models/src/diffusion.rs` (config detection), `crates/higgs-models/src/diffusion_ane.rs` (GQA params)

**Models already downloaded:**
- `~/.cache/huggingface/hub/models--dllm-hub--Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1/snapshots/a284e895.../`
- `~/.cache/huggingface/hub/models--dllm-hub--Qwen3-0.6B-diffusion-mdlm-v0.1/`

### Phase 4 — EGGROLL fine-tuning loop (higgs)

**Goal:** On-device training from user feedback, zero backprop.

- Fitness signal: user kept the UI (positive) vs regenerated (negative)
- EGGROLL perturbs via `perturbed_qlinear_forward` — no dequant/requant, ~90MB overhead
- Training runs during ANE idle time (background lane, lower priority than inference)
- Periodic merge: `dequant(W) + delta → requant` replaces base weights

**Where:** Already built in `crates/higgs-models/src/diffusion_eggroll.rs` + `eggroll_quantized.rs`. Wire fitness signal from nanobot → higgs `/train` endpoint.

### Phase 5 — Component commons (nanobot)

**Goal:** Users share and discover components.

- Components saved as JSON + YAML frontmatter (reuse skills pattern): `~/.nanobot/workspace/components/{name}/COMPONENT.json`
- Sharing protocol: git-backed registry (simplest), or exchange via nanobot message channels
- Imported components become: (a) usable artifacts the renderer can instantiate, (b) positive training examples for EGGROLL
- Discovery: `ComponentRegistryTool` — agent can search/list/import from commons

**Where:** `src/agent/tools/component_registry.rs` (new tool), components follow existing skills directory convention

## Dependencies

```
Phase 1 ← nothing (higgs-only)
Phase 2 ← nothing (nanobot-only)
Phase 3 ← Phase 1 (needs scheduler to test dual-mode)
Phase 4 ← Phase 3 (needs working diffusion inference)
Phase 5 ← Phase 2 (needs schema + renderer)
```

Phases 1 and 2 are fully parallel. Phase 3 merges them. Phases 4 and 5 are independent of each other.

## Non-goals

- Raw HTML generation (search space too large for 0.5B)
- JavaScript framework dependency in rendered output (static HTML/CSS only, progressive enhancement later)
- Cloud training or centralized model hosting
- Vision/screenshot-to-UI (different problem, requires multimodal model)

## Models

| Role | Model | Params | Path |
|------|-------|--------|------|
| UI generation | `dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1` | 0.5B | HF cache (downloaded) |
| Fallback / comparison | `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` | 0.6B | HF cache (downloaded) |
| Draft decode | Same 0.5B or dedicated draft | shared ANE kernels | — |

## References

- [dLLM framework](https://github.com/ZHZisZZ/dllm) — AR→diffusion conversion recipes
- [MDLM paper](https://arxiv.org/abs/2406.07524) — masked diffusion language models
- [EGGROLL paper](https://arxiv.org/abs/2511.16652) — forward-pass-only training
- [OpenGenerativeUI](https://github.com/CopilotKit/OpenGenerativeUI) — pattern reference (skills → LLM → iframe)
- [Tesslate UIGEN](https://huggingface.co/Tesslate) — UI generation training methodology (3B+, dataset approach worth replicating)
