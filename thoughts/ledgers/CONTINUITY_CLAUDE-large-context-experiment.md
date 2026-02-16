# Continuity Ledger: Large-Context Multi-Agent Experiment

## Goal
Get the `experiments/large-context-test/` experiment running successfully. The test validates nanobot's ability to process 88 ArXiv papers via local models on RTX 3090, spawning parallel subagents for extraction.

**Success criteria**: Author coverage > 70% AND emergence topic discussed, validated against `results/ground_truth.json`.

**Result**: 93.3% overall score — PASS (Author 80%, Keywords 100%, Emergence Yes)

## Constraints
- All local models on RTX 3090 (24GB VRAM)
- Must use `exec` tool for CSV processing (not `read_file` — context too large)
- Subagents need strict alternation repair for Ministral/Jinja templates
- Pipeline/loop features were implemented in the overhaul session (Phase 1-4)

## Key Decisions
- Pipeline callback is now wired — `action: "pipeline"` works with tool-equipped steps
- Loop action added — `action: "loop"` for iterative refinement
- Depth limit = 3, budget halves per depth level
- Delegation diagnostics improved (model name, iteration count, error vs empty vs timeout)
- **SLMs need explicit exec commands**: Ministral-8B uses placeholder text if not told "Use exec tool to run this exact command" [2026-02-16]
- **Minimal workspace for experiments**: Default workspace has 96KB of markdown, blows 16K context budget [2026-02-16]

## State
- Done:
  - [x] Phase 1a: Wire pipeline callback in agent_loop.rs
  - [x] Phase 1b: Add delegation diagnostics to tool_runner.rs
  - [x] Phase 2: Tool-equipped pipeline steps with context chaining
  - [x] Phase 3: Agent loop action ("loop") in subagent
  - [x] Phase 4: Depth limits & budget propagation
  - [x] Build passes (release), 990/991 tests pass (1 pre-existing failure)
  - [x] Fix: Pipeline strict alternation repair for local models
  - [x] Fix: `defaultSubagentModel` must be inside `toolDelegation` in config JSON
  - [x] Fix: `is_local` detection for provider-routed local models (groq/ → localhost)
  - [x] Fix: Consecutive user messages after repair — removed redundant user continuation in `_run_subagent()`
  - [x] Test 1-3: Basic exec, author extraction, keyword search — all PASS
  - [x] Test 4: Subagent spawn to Ministral-3B on port 8083 — PASS
  - [x] Test 5: Pipeline action (model confused, didn't execute) — SOFT PASS
  - [x] Test 6 v2: Full workflow with explicit exec prompts — PASS (93.3%)
- Now: [DONE] — Experiment validated successfully

## Bugs Fixed (This Session)

### 1. Pipeline Jinja fix (pipeline.rs)
`execute_step_with_tools()` didn't call `repair_for_strict_alternation()` for local models.
Fixed by adding `thread_repair` import and repair call gated on `is_local && iteration > 0`.

### 2. Context size exceeded
Workspace at `~/.nanobot/workspace` has 96KB of markdown files loaded into system prompt.
Fixed by creating minimal workspace at `experiments/large-context-test/workspace/` with one-line SOUL.md.

### 3. defaultSubagentModel silently ignored
Config had `defaultSubagentModel` at JSON top level; serde expects it inside `toolDelegation`.
Fixed by moving field inside `toolDelegation` in config.local.json.

### 4. is_local false for provider-routed local models
When model has prefix like `groq/...`, `resolve_provider_for_model` set `routed_to_cloud = true`.
Fixed by adding `targets_local` return value that checks if base URL contains localhost.

### 5. Consecutive user messages after repair (ROOT CAUSE of Jinja errors)
`repair_for_strict_alternation()` converts tool→user, then code ALSO appended user continuation.
Result: `["system", "user", "assistant", "user", "user"]` — violates strict alternation.
Fixed by removing the user continuation in `_run_subagent()` — repair already ends with user message.

## Validation Results (test6_v2)
- **Author Coverage: 80%** — Found: Zhan Qu (3), Michael Färber (3), Yue Huang (2), Xiangliang Zhang (2), Theresa Schmiedel (2). Missed: Juneyoung Park (2).
- **Keyword Coverage: 100%** — emergence, scaling, chain-of-thought all found
- **Emergence Discussed: Yes**
- **Overall Score: 93.3% — PASS**

## Working Set
- **Branch**: `vibe-1771266698`
- **Experiment dir**: `experiments/large-context-test/`
- **Config**: `experiments/large-context-test/config.local.json`
- **Ground truth**: `experiments/large-context-test/results/ground_truth.json`
- **Model files**: `~/models/` (Ministral-8B on :8080, Ministral-3B on :8083)
- **Build**: `cargo build --release` (clean)
- **Validation**: `python3 experiments/large-context-test/scripts/validate.py results/v3_test6_v2.txt`

## Open Questions
- Pipeline action (test 5) was confused by the JSON prompt format — SLM couldn't parse tool args
- NanBeige 3B available but needs thinking disabled — not tested yet
- `DEFAULT_LOCAL_MODEL` in server.rs is hardcoded to Nemotron-9B regardless of actual server model (cosmetic)
