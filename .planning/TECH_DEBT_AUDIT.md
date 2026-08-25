# Tech Debt Audit — Aug 2026 refresh (`refactoring/maximum-speed-with-less-code`)

Supersedes the June audit (in git history). Methodology unchanged: four threads
(dead_code / dry / solid / over_engineering), each finding re-verified by hunting
for the caller/test/impl before it keeps `safe_to_act=true`. Scope: `src/` only
(169 files, ~157k LOC). Build warning-clean; all dead items verified by rg
caller-counting, not compiler hints.

## 1. Headline — LOC removable (verified safe only)

**Total verified-safe removable: ~486 LOC** (plus ~110 LOC cfg(test)-only-live
pub API held back under June's coverage rule).

| Category | Verified-safe LOC | Largest single win |
|---|---|---|
| Dead code (incl. dead speculative abstractions) | ~462 | `bus/queue.rs` MessageBus = 234 |
| DRY (de-duplication) | ~21 | router directive-pack twins = 9 |
| SOLID (dead-param cleanup) | ~3 | `tool_preflight_result` = 3 |
| **Total** | **~486** | MessageBus deletion |

De-dup: `ToolsetsConfig` and `admit_with_specialist` flagged by both dead_code
and over_engineering; each counted once in dead_code.

## 2. Full findings — sorted LOC desc

| # | Title | Location | Kind | LOC | Risk | Effort | Action | safe |
|---|---|---|---|---|---|---|---|---|
| 1 | `MessageBus` orphaned — whole module, zero external refs | `src/bus/queue.rs` (all 233) + `pub mod queue;` bus/mod.rs:2 | dead_code | 234 | low | trivial | Delete file + mod decl. Channels wire directly via ChannelManager + bus::events. | ✅ |
| 2 | `GroqTranscriptionProvider` orphaned by voice_pipeline | `src/providers/transcription.rs` (all 110) + providers/mod.rs:9 | dead_code | 111 | low | trivial | Delete file + mod decl. Telegram/WhatsApp transcribe via voice_pipeline.transcribe_file (telegram.rs:295). | ✅ |
| 3 | `admit_with_specialist` — dead "semantic gate" parallel path | `src/agent/context_gate.rs:103` (+ test :789) | dead_code/over_eng | 43 (+~55 test) | low | small | Delete method + test. Production gates go through admit_simple/admit; the "second parallel pipeline" AGENTS.md forbids. | ✅ |
| 4 | `toolsets` config — parsed, never read | schema.rs:856 ToolsetsConfig, :938-940, :37/:79/:130, Debug :89/:157 | dead_code/over_eng | ~28 | low | small | Delete struct + 4 fields + Debug lines. No reader anywhere; no deny_unknown_fields → old configs load. | ✅ |
| 5 | `Turn` helpers: is_user/is_clear/assistant_text | `src/agent/turn.rs:116,126,131` | dead_code | 13 | low | trivial | Delete all three (is_assistant/is_summary/tool_calls stay — live). | ✅ |
| 6 | get_memory_path/get_skills_path | `src/utils/helpers.rs:81,90` | dead_code | 14 | low | trivial | Delete both; memory/skills resolve their own paths. | ✅ |
| 7 | ChannelManager::get_status | `src/channels/manager.rs:178-192` | dead_code | 15 | low | trivial | Delete; cli uses enabled_channels/start_all/stop_all only. | ✅ |
| 8 | lms.rs vestigial timeout params (June #12 residue) | `src/lms.rs:180,216,217,233` + ~25 call sites | over_eng | 12 | low | small | June only `_`-prefixed them. Drop params + args at call sites. | ✅ |
| 9 | Directive-pack action/target twin blocks | `src/agent/router.rs:438-456` | dry | 9 | low | trivial | Extract `fn take_token(pack, key) -> Option<String>`. | ✅ |
| 10 | ToolErrorKind::ExecutionFailed never constructed | `src/errors.rs:200` + arm :413 + test :791 | dead_code | 6 | low | trivial | Delete variant + arm + test line. ToolError::Execution is live — keep. | ✅ |
| 11 | tool_preflight_result ignores 2 of 3 params | `src/agent/router.rs:914` | solid | 3 | low | trivial | Drop _tool_name/_tool_result + caller args. | ✅ |
| 12 | cfg(test)-only-live pub API (10 fns) | knowledge_store.rs:468, audit.rs:357, context_gate.rs, db.rs:3266, registry.rs:296,909, base.rs:131, circuit_breaker.rs:37, knowledge_graph.rs:210, token_budget.rs:268 | dead_code | ~110 | low | small | June precedent: deleting removes live-type coverage → default leave. Keep `failure_with_kind` — planned Phase-2 seam. | ❌ |
| 13 | God fns new since June: step_call_llm 824, run_tool_loop 613, raw_json_tool_call_span 534, execute_tools_delegated 465, prepare_context 461, router_preflight 404, route_tool_calls 401 | shared.rs:3084, tool_runner/mod.rs:556, validation.rs:185, tool_engine.rs:589, prepare_context.rs:203, router.rs:972,1415 | solid | 0 (relocation) | high | large | All carry invariants as hot-path comments; no internal duplication. Only raw_json_tool_call_span has extractable pure sub-scanners. Needs per-fn plan + replay-byte regression first. | ❌ |
| 14 | cmd_agent grew 1179 → 1302 | `src/repl/mod.rs:1454` | solid | 0 | high | large | June's needs-care verdict stands, worsened. 3 behavioral differences block mechanical merge. | ❌ |
| 15 | God files: session/db.rs 6239, tui_app/app.rs ~3460, agent_loop/shared.rs 5946, lcm.rs 5224 | as listed | solid | 0 (relocation) | medium | large | db.rs split (events/replay vs session CRUD) highest-value relocation. | ❌ |
| 16 | Stop/filler-word lists ×7 files | heuristics.rs:228, anti_drift.rs:38, lcm.rs:1745, validation.rs:101, db.rs:2764, context.rs, model_capabilities.rs | dry | ~60 | high | medium | Deliberately different vocabularies; unifying changes heuristic behavior. Keep, document divergence. | ❌ |

Folded as trivia: syntax.rs theme getters, tool-trait boilerplate (~28 groups),
test-fixture dup (~160 LOC, test-only).

## 3. Delta vs June

**Done since June** (all 17 safe wins executed, ~1,202 LOC landed): parsers/,
runtime_mode dead methods+tests, execute_inner/proxy collapse (Option<&ToolContext>
at registry.rs:669), require_str (macro + worker_tools), lcm rebuild tail,
anthropic retry chain, strip_tags, needs_local_protocol copy, truncate twins,
redact_opt, from_provider_config, SSE parse_tool_arguments, single parse_sse_stream,
build_chat_request shared (openai_compat.rs:1071/1256), cargo-fix sweep.
June #18 landed differently (SectionEntry driver at context.rs:447).
**Half-done:** June #12 — lms timeout params `_`-prefixed but still passed (row 8).
**Stale/worse:** cmd_agent +123 LOC; stream_and_render_inner, VoicePipeline,
PromptAssembler, filters.rs chars/4 estimator — June verdicts still apply.

**New since June:** architecture orphans, not new slop — MessageBus bypassed by
direct channel→agent wiring, GroqTranscriptionProvider by voice_pipeline, toolsets
never got a consumer, admit_with_specialist is a dead specialist-gate lane,
ExecutionFailed is Phase-0 residue. New subsystems (replay/journal, tool
pre/execute/post, cua, python_kernel, tui_app) came out clean. Net: debt moved
from "speculative abstraction layers" to "concentration in hot-path god
functions". The deletion well is ~500 LOC, safe to drain in one commit; the
splits need their own plans.
