# Local Portal-style workers: assessment, 2026-09-05

Status: read-only investigation and proposed experiment. No model download, inference, service restart, build, production edit, or configuration change was performed. The running Higgs endurance experiment was left alone.

## Conclusion

The mechanism is feasible locally. Implement two bounded tool operations that send source bytes to small models and return compact evidence or a generated-file receipt to the main agent. For v1, the reader is an **extractive evidence worker**: source excerpts and validated references are the useful output, with any explanatory summary secondary. The writer produces an **untrusted staged artifact or patch** that the host validates and applies through the current write policy. Keep the current provider, tool execution, and durable result paths. This can reduce the main model's context growth; it does not itself recover forgotten session facts, replace LCM, or prove lower total inference time.

For a first distinct-model experiment, use **Qwen3.5-4B at MLX 4-bit for bulk reading** and **Qwen2.5-Coder-3B-Instruct at MLX 4-bit for template-driven generation**. This is a provisional, testable pairing, not a measured ranking. Their published artifacts total roughly 4.8 GB before inference state. Static arithmetic on a 32 GB machine with the approximately 12.9 GB Escha35B workload suggests possible coexistence at short worker contexts, but the main experiment is already investigating memory-pressure failures. **Do not add two permanent residents on that evidence.** Start with one helper loaded at a time after the current run, explicitly await its unload before changing helper, and measure both the load cost and displaced Escha cache. Serialization of inference alone does not remove resident-weight pressure. If even one helper is rejected by capacity policy, use a smaller candidate or postpone local delegation; do not reduce the existing safeguards to force the experiment through.

The simplest control is one Qwen3.5-4B checkpoint serving both role prompts. Two roles do not require two resident checkpoints; keep the second checkpoint only if it improves measured quality or latency enough to justify its memory.

## What Spotify actually implemented

The September 3 article describes two AiKA modes, both using Gemini 2.5 Flash in its examples: a question-focused bulk reader, and a boilerplate writer supplied with a specification and reference file. Each call is independent. The reader returns compact findings; the writer can place its output directly on disk so Claude never receives the full generated code. Hooks redirect large full-file reads while permitting targeted reads. The author explicitly excludes debugging, architectural judgment, and safety-critical reasoning, reports unreliable summary line numbers, and describes 10–30 second delegation latency. The claimed approximately 90% saving concerns Claude's bulk-read context in a small set of scenarios, not all models' combined tokens. [Spotify article](https://engineering.atspotify.com/2026/9/portal-by-spotify-cut-my-claude-code-token-usage-by-90)

The linked implementation is `shunt`, in the article author's `add-shunt-claude` fork. It has hooks, CLI wrapper scripts, and skills; the reader threshold defaults to 350 lines. Writer routing is advisory. Its README reports three Java reading cases: 33,684→5,737, 75,990→4,148, and 16,221→821 estimated tokens, plus a separate generation case. Portal modes are reusable configurations rather than a necessary algorithmic component. A local HTTP worker can fill that role. [Linked shunt implementation](https://github.com/sorantis/portal-ai-plugins/tree/add-shunt-claude/plugins/shunt), [AiKA mode documentation](https://backstage.spotify.com/docs/portal/core-features-and-plugins/aika/modes)

The public benchmark definition uses `chars / 4` and TypeScript fixtures, not the Java corpus behind the README results. [Benchmark definitions](https://raw.githubusercontent.com/sorantis/portal-ai-plugins/add-shunt-claude/plugins/shunt/evals/benchmarks.json)

The benchmark runner suppresses worker errors with `|| true`, measures response length without checking answer correctness, assigns the writer zero main-context tokens, and weights its baseline generation tokens by five. Consequently an empty failed response can look like excellent savings. These scripts demonstrate the accounting idea, but do not establish quality-preserving end-to-end savings. [Benchmark runner](https://raw.githubusercontent.com/sorantis/portal-ai-plugins/add-shunt-claude/plugins/shunt/evals/run.sh)

## Existing integration points and boundaries

Source observations refer to the current nanobot workspace and `/private/tmp/higgs-recovery` at the supplied `327e5021ef957a7d6968f6780a7644f00b410a9e` revision.

| Existing code | Consequence for implementation |
|---|---|
| `src/agent/tools/filesystem/mod.rs`, `ReadFileTool` | Reads already return bounded, contiguous numbered windows, approximately 7,000 characters by default, with content hashes and continuation ranges. The current baseline is considerably more economical than indiscriminate whole-file reads. |
| `src/agent/tools/registry.rs`, `execute_inner` / `run_pre_hook`; `src/agent/hooks.rs` | One execution boundary already applies permissions and a PreToolUse hook. Hooks receive `NANOBOT_TOOL_NAME`/`NANOBOT_TOOL_PARAMS`; their protocol differs from Claude Code's hook JSON. Hook crashes/timeouts fail open. |
| `src/providers/base.rs`, `LLMProvider::chat` | Already accepts an explicit model and generation budget and returns typed finish reasons and usage. Suitable transport for a bounded worker call. |
| `src/agent/subagent.rs`, `resolve_spawn_settings` / `resolve_provider_for_model`; `agent_profiles.rs:230` | Profiles, explicit model IDs and provider resolution exist. Local `haiku`, `sonnet`, `opus`, and `local` aliases all resolve to the main local model; use concrete helper IDs. General subagents also bring a tool loop, context assembly, and temperature 0.7, which are unnecessary for these one-shot jobs. |
| `src/agent/tool_engine.rs:643`, `execute_tools_delegated` | Despite its name, this production function ignores its delegation provider/model arguments and executes tools inline. Existing `tool_runner`/`ctx_summarize` machinery is not an enabled solution to this task. Avoid reviving a second agent pipeline. |
| `src/agent/tool_engine.rs:252`, `store_then_render_tool_result` | Durable exact results and stable handles/excerpts already protect replay. Worker outputs should enter this same ingestion path once and remain immutable in replay. |
| Higgs `crates/higgs/src/router.rs`, `routes/models.rs`, `state.rs` | Higgs supports multiple named local engines, runtime load/unload, and capacity-aware publication. A second inference server or learned router is unnecessary. Its memory authority includes MLX limits and Metal's recommended working set. |
| Higgs `routes/chat.rs` | Per-request or per-model `enable_thinking` settings exist. Use worker-specific generation settings rather than changing Escha's global environment. |

Graph exploration: bound repository `nanobot-rs`; index `c59bb8f`, September 4, is 36 commits behind the observed `fa53da4` HEAD. The default CLI chose an incompatible database runtime (storage 40 versus database 43). Using the exact indexed CLI artifact restored query/context/impact reads:

`/Users/peppi/.hermes/node/bin/node /Users/peppi/.npm/_npx/5e786f48223a616c/node_modules/gitnexus/dist/cli/index.js`

`execute_ctx_summarize` context identifies `run_tool_loop` and `analyze_via_scratch_pad` callers; source search confirms these are confined to the legacy worker implementation and tests, while the release entry above executes inline. Impact for `ToolRegistry::execute_inner` is **HIGH**: 58 upstream impacted symbols, three direct callers (`execute`, `execute_proxy`, `execute_with_context`), and three reported process groups, including subagent `run_loop`, scratch-pad analysis, and proxy tests. This is stale graph evidence, not a clean current blast-radius approval. Reindex and repeat before any dispatcher modification. Higgs' registered indexes point at other revisions/worktrees; this assessment inspected the requested worktree directly and makes no clean Higgs graph claim. No reindex was run during the live experiment.

## Minimal proposed behavior

1. Add a narrow `bulk_read(question, paths)` tool alongside the filesystem tools. Canonicalize and authorize each path using the existing policy, impose byte/token limits, snapshot source with hashes, and number lines deterministically. Send only that corpus and question to the reader, with no tools and no history. Return bounded claims, source references, and an explicit incomplete/unknown status. Preserve the source snapshot or durable handle for exact retrieval. Validate quoted spans against source bytes; model-supplied references alone do not prove a claim.
2. Add `code_write(spec, reference_paths, target)` beside the existing write tool. The main agent supplies intended behavior, interfaces, and any required source paths, not source contents pasted into its call. Give the writer no exec/network tools. Require a complete successful response; reject truncation, malformed wrappers, or empty output. Stage an untrusted candidate artifact or patch, validate it with the project’s existing checks in a constrained environment, then apply through the same authorized write behavior. The worker never bypasses file restrictions or directly overwrites the repository. Start with new boilerplate files; edits to existing logic remain exact targeted reads plus normal edits, and any later patch support must check source hashes before application. Return path, hash, size, and concrete check results without dumping the generated file into main context.
3. Route by explicit task and input size, not an additional LLM classifier. Preserve cheap direct reads/searches. Initial trial thresholds: reader corpus 4–16K input tokens, output cap 512–768; writer input cap 8K and output cap 2K. These are proposed experimental settings, not measured optima. Request limits include prompt overhead and output reserve. If a task needs more, split at source boundaries and disclose coverage.
4. Keep one normal tool-call/result pair in the main history. Log helper model ID, revision, source hashes, input/output usage, finish reason, elapsed time, and artifact/summary receipt. Do not inject a changing summary into previously cached turns. Workers do not rewrite SQLite history, LCM summaries, or MEMORY.md.
5. Workers inherit source taint; wrapping source in XML or JSON does not make its instructions trusted. File policy and generated-code validation remain host responsibilities. Worker failure returns a typed failure plus source handles/targeted-read guidance; never silently treat an empty answer as success or route to a different model behind the user's back.

A script-plus-skill prototype using the existing hook is possible and avoids Rust changes, but shell arguments are a poor transport for entire corpora and direct script writes must still enforce workspace policy. For the release path, the narrow tool approach better preserves existing permissions and durable result semantics. Do not insert an LLM call into every file read or copy the article's line threshold verbatim: nanobot's first read page is already bounded, and a hook redirect adds another main-model turn.

## Model candidates and what is actually present

| Role | Candidate and evidence | Limitations |
|---|---|---|
| Bulk reader | **Qwen3.5-4B, MLX 4-bit.** Official model card gives a hybrid architecture, 32 layers with eight full-attention layers, and published comprehension/coding results. The MLX artifact page lists about 3.03 GB. | Use text-only input and disable thinking for this extraction experiment. Published BF16/standard benchmarks do not validate the chosen quantization, Higgs path, Rust understanding, or 16K coverage. [Official model](https://huggingface.co/Qwen/Qwen3.5-4B), [MLX artifact](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit) |
| Template writer | **Qwen2.5-Coder-3B-Instruct, MLX 4-bit.** Official 3.09B code model with 32,768 context; the MLX artifact is listed at 1.74 GB. | Older model, selected for bounded code generation and modest memory, not because it is proved superior to Qwen3.5-4B. Rust and repository-specific pass rates must decide. [Official model](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct), [MLX artifact](https://huggingface.co/mlx-community/Qwen2.5-Coder-3B-Instruct-4bit) |
| Smaller reader challenger | MiniCPM5-1B has 24 layers, two KV heads, 128K advertised context, switchable thinking, and an official MLX 4-bit release. | Plausible for narrow factual extraction; a 1B model is not established as adequate for cross-file semantic analysis. Its Llama family is supported by Higgs, but the exact checkpoint/template still needs a smoke test. [Official model](https://huggingface.co/openbmb/MiniCPM5-1B), [official MLX release](https://huggingface.co/openbmb/MiniCPM5-1B-MLX) |

Read-only artifact inventory, using actual file `stat` rather than directory names:

- Escha Qwen3.6-35B-A3B W2: three safetensors files totaling **12.297 GB on disk** under `~/.cache/lm-studio/models/EschaLabs/`. The approximately 12.9 GB resident allowance supplied for this assessment is a different quantity.
- Qwen3.5-9B MLX 4-bit: **5.950 GB** of safetensors, present. Too large to be the initial cheap helper alongside the other proposed residents.
- LiquidAI LFM2.5-2.6B MLX 8-bit: **2.866 GB**, present. It is not in the inspected Higgs supported-family registry (`lfm2` absent); using it would require another backend or engine work. Its maker also discourages agentic coding and describes this checkpoint as always-thinking, so it is not a drop-in fast writer. [LiquidAI model card](https://huggingface.co/LiquidAI/LFM2.5-2.6B)
- Hugging Face cache directories for Qwen3.5-4B MLX 4-bit and MiniCPM5-1B 4-bit contain refs only, not snapshots/weights. SmolLM3-3B has a config snapshot but no safetensors in that snapshot.
- `~/Models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf`, both checked Qwen3-1.7B GGUF links, and the checked LFM2-350M link are dangling. They must not be presented as ready-to-run models.

This inventory checks common local cache roots, not every mounted disk. It does not imply a complete artifact has passed a loader or inference check.

## Memory and latency envelope

Use the supplied 32 GB machine constraint; sandboxed `sysctl hw.memsize` was unavailable, so this investigation did not independently confirm its hardware SKU. `vm_stat` worked, but a single free-page sample during active work is not an admission budget.

Illustrative budget, decimal GB unless marked GiB:

| Allocation | Planning amount |
|---|---:|
| Existing Escha workload allowance | ~12.9 GB |
| Two proposed helper weight artifacts | ~4.8 GB |
| OS and other applications, reserved assumption | 6 GB |
| Inference state, prefill activations, caches, allocator slack | Remainder, roughly 8 GB before respecting the backend's potentially lower limit |

The above is arithmetic, not proof of fit. At FP16, basic attention KV storage is `tokens × full_attention_layers × KV_heads × head_dim × 2(K,V) × 2(bytes)`. From the official Qwen3.5-4B architecture, its attention cache alone is **0.5 GiB at 16K tokens** and 1 GiB at 32K; recurrent states and workspaces add more. The local Escha config has 40 layers, full-attention interval four, two KV heads and dimension 256: its basic attention KV is approximately **1.25 GiB at 64K**, before recurrent state, cache snapshots, and temporary allocations. The coder's conventional GQA cache also grows with context. Long advertised model contexts are not suggested resident allocations.

Higgs memory policy may reduce caches or reject a helper load before physical RAM is exhausted. Loading a helper can displace valuable Escha prefixes even when resident weight totals look safe. Begin with sequential helper residency and requests; evaluate dual residency only as a later explicitly measured case. A smaller MiniCPM5 reader plus the coder is a lower-footprint challenger if extraction quality survives the tests. Do not infer shared-GPU speedup from model parameter counts: Escha is a sparse MoE while the proposed helpers are dense. A 4B dense model need not prefill faster than a 35B-A3B model.

Measure `helper prefill + helper decode + main resumption + cache refill` against the direct-read baseline. Repeated file questions resend their corpus to a one-shot worker; that is real local work. Cache only by exact corpus hashes, question, model revision, and generation settings if measurements subsequently justify it.

## Experiment that can justify shipping

Run after the current endurance experiment finishes, with pinned repository revisions and identical main model/settings:

1. Compare current nanobot targeted reads/searches, one shared 4B helper serving both roles, and the proposed two-model pair. Keep a direct exact-read control for difficult tasks. Separate cold-load and warm-resident runs; repeat to capture variance.
2. Reader set: at least 30 answerable code questions across single-file, cross-file, source-plus-test, constants/config, and negative queries. Include distractors, late-file answers, conflicting definitions, stale snapshots, UTF-8, minified long lines, and a planted subtle concurrency issue that requires escalation. Score supported-fact precision, required-fact recall, source-reference validity, and incorrect certainty. An answer that compresses away the requested fact fails.
3. Writer set: at least 20 small new-file tasks with existing references and explicit interfaces. Validate syntax/build and held-out behavior assertions; for generated tests, introduce known defects and require tests to catch them. Passing generated tests alone is insufficient. Track targeted review tokens, rejected artifacts, and repair rounds.
4. Record exact tokenizer/API usage for **all** main and helper calls: input, output, reasoning, cached and uncached tokens where exposed. Include hook redirects, skill/tool schemas, errors, retries, review reads, and later cache refill. Report main-model savings separately from all-model work and peak main context size.
5. Report task success first, then P50/P95 completion time, helper timeouts, prompt/cache hit statistics, peak process/MLX memory, swap growth, capacity rejections, and OOMs. Timeout/empty/truncated/error responses are failed tasks; they never count as a 100% saving.
6. Provisional acceptance criteria: all designated critical-fact and protocol checks pass; no extra out-of-policy writes or memory failures; at least 95% required-fact recall on the finite reader set; no hidden-behavior regression in accepted generated files; and at least 50% median main-context reduction on eligible bulk tasks. These are proposed gates, not results. Retain the second helper only if it adds measurable value over one shared helper. Report uncertainty rather than generalizing a small pass set to arbitrary code.
7. Re-run long-session recovery with helpers enabled: old raw facts remain retrievable after compaction, source handles survive restart, the main prefix remains stable, worker artifacts do not pollute MEMORY.md, and capacity changes do not disturb the recovery guarantees.

This investigation supports a controlled prototype. It does not establish 90% savings, superior small-model accuracy, measured coexistence on this machine, or production readiness.
