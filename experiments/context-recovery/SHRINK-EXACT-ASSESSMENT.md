# Can nanobot become much smaller with exactly the same behavior?

Assessment date: 2026-09-05. Read-only investigation of production code; no builds, benchmarks, service changes, model loads, or production edits performed by this investigator.

**There is real cleanup available, including a dormant delegated execution implementation. This audit does not justify promising a 50% reduction in the actual production source while preserving current behavior.** The defensible identified pool is roughly 1–2% of non-test source, much of which should become a test reference rather than disappear. A larger percentage requires finding and proving additional redundancy; it is not established by the current code or by passing the existing tests.

The hardened recovery path is valuable behavior, not an interchangeable implementation detail. A simpler fresh-window policy, fewer recovery routes, different tool descriptions, or a smaller set of supported configurations would be a product change. Those can be evaluated separately, but cannot be credited as exact-preserving shrinkage.

## Measurement and scope

Pinned baseline: `fa53da492f100a7388571913f425c7419e1db1c3` on main in `/Users/peppi/Dev/nanobot-rs`. Existing dirty edits and new recovery/endurance harness files were preserved. The concurrent experiment changes those files, so stable source counts below come from `git show <baseline>:<path>`, not a moving worktree.

Counts are physical Rust lines, including comments and blank lines. A tree-sitter Rust item walk separates `#[cfg(test)]`, `#[cfg(all(test, ...))]`, and test-attributed item ranges, plus the separate `tests.rs` modules. **Splitting a file at its first `#[cfg(test)]` is incorrect here:** several files contain small test helpers before thousands of production lines. The parser reports unsupported syntax around a cfg-attributed match pattern in `repl/commands/lifecycle.rs:532`; its test modules at 1637 and 1721 remain separately parsed. Treat the production/test division as a source classification, not compiler-proved reachability. It counts optional feature implementations and dormant functions in the non-test pool.

| Baseline source | Physical lines | Interpretation |
| --- | ---: | --- |
| All 169 Rust files under `src/` | 170,758 | Actual tracked Rust source, not 170k of runtime logic |
| Test items inside `src/` | 77,026 | 45.1% of that total; preserve hardening assertions |
| Non-test items under `src/` | 93,732 | Includes comments, optional features, and dormant implementations |
| `src/bin/trio_bench.rs` | 770 | Benchmark executable, included in preceding row |
| Application/library non-test pool excluding that benchmark | 92,962 | Useful denominator for production-source discussion |
| Separate Rust integration tests under `tests/` | 2,484 | In addition to the 77,026 above |
| `build.rs` | 158 | Build/packaging behavior, separate from runtime |

The working tree initially added about 1,050 physical Rust lines, almost entirely experiment test code and its cfg wiring; it continued changing during this investigation. Do not attribute this ongoing experiment growth to production complexity.

| Area | Non-test physical lines | Test physical lines |
| --- | ---: | ---: |
| `src/agent` | 51,718 | 55,697 |
| `src/repl` | 8,515 | 1,755 |
| `src/tui_app` | 5,709 | 3,218 |
| `src/session` | 4,497 | 4,256 |
| `src/providers` | 3,342 | 3,932 |
| `src/config` | 2,822 | 975 |
| `src/cli` | 2,351 | 460 |
| `src/channels` | 2,039 | 1,156 |

The agent-loop directory itself contains approximately 9,569 non-test lines and 17,135 test lines at the pinned baseline. Its 14,418-line `tests.rs` is entirely test code. Large production concentrations are `agent_loop/shared.rs` (5,848), `session/db.rs` (4,162), `tui_app/app.rs` (4,112), and `providers/openai_compat.rs` (2,482). Moving tests out of mixed files makes those files easier to navigate but does not reduce total source or runtime behavior.

Other size categories, measured separately:

- Tracked `archive/`: 37 files, about 1.62 MB and 41,348 newline bytes. Much is historical patches, including stashes and an uncommitted-cancellation patch; these are not compiled source. Preserve recoverability before moving historical material elsewhere.
- Tracked `experiments/`: 66 files and about 3.59 MB; includes Python, results, and model tensors. Current on-disk experiments were about 42 MB because new evidence is untracked. They are not application Rust.
- Tracked `tests/`: about 3.07 MB, including binary reference data. Binary newline counts are not LOC.
- `bridge/whatsapp/index.js` is live channel support, not vendor code. `bridge/ane` contains roughly 1,820 physical lines of an older Objective-C/header/build bridge; no current Cargo linkage was found, but its standalone Makefile is a separate executable surface, not an automatic deletion.
- No vendored Rust subtree or generated Rust file under `src/` was identified. Cargo registry dependencies and generated build outputs are outside this source denominator. `Cargo.lock` is generated dependency metadata, not redundant application code.
- On-disk `target/` was 5.4 GB, `.gitnexus/` 250 MB, and `.git/` 83 MB. Disk cleanup could dwarf every source reduction, but would not simplify the code. Do not clean an active build/test target during the current experiments.

## Graph evidence and limitations

Bound graph repository: **nanobot-rs**, same worktree. Index commit: `c59bb8f1bf34dc31110afcc25057636dd58e20d7`, indexed 2026-09-04, **36 commits behind HEAD**. CLI used the installed Node 22 runner and GitNexus 1.6.10. Queries covered agent-loop/compaction/recovery/tool execution, legacy compatibility, and delegated tool execution; symbol contexts and upstream impacts disambiguated file/UID where needed.

No reindex was run because this investigator owns only this report and the main agent is operating concurrently. The stale graph is discovery evidence, not an edit clearance. Any implementation must refresh the graph, repeat impact checks, and run complete graph change analysis before committing. No commit or graph-change success is claimed here.

| Prospective symbol | Upstream graph result | Current-source confirmation |
| --- | --- | --- |
| `router_fallback::route` | **HIGH**, 20 impacted, 17 direct; `route_tool_calls`, `step_execute_tools`, `run_agent_loop` processes | Called at `router.rs:1463` and `1488`. Active fallback behavior, despite the architectural preference against fallback pipelines. |
| `ContextBuilder::add_tool_result_immutable_with_status` | **HIGH**, 12 impacted, 4 direct; injection and loop processes | Same JSON body as regular status appender, but live protocol callers require exact-byte tests. |
| `proactive::has_path_like` | LOW, 14 impacted, 1 direct; pre-call and loop processes | Duplicate predicate in fallback router. Tiny extraction candidate, not evidence of duplicated routing engines. |
| `ContextBuilder::assemble_local_prompt_report` | **UNKNOWN**, zero resolved callers | Current source search finds only definition at `context.rs:1413`; private inherent method. Strong deletion candidate after refreshed analysis and compiler check. Zero graph edges alone were not treated as unused. |
| `heuristics::proactive_recall` | **UNKNOWN**, zero resolved callers | Current source search finds only definition at `heuristics.rs:396`; restricted `pub(super)` visibility. Same caveat. |
| `compact_inline_tool_result` | LOW, 1 resolved caller, no production process | Source references are truncation tests; comment explicitly identifies it as retained reference implementation. |
| `get_slim_definitions` | LOW, 3 resolved callers, test processes | Current references are test-only comparisons; production uses other tool-presentation paths. |
| `run_tool_loop` | **CRITICAL**, 32 direct callers, test processes | Current references are tests and self-recursion; the old runner has no release entry found. Preserve its reference coverage. |
| `execute_tools_delegated` | LOW, 3 impacted, 1 direct; execution/loop processes | Active wrapper, with behavior in its caller that prevents trivial removal. |

**HIGH and CRITICAL results are warnings, not permission to remove these implementations.** The graph includes inline test callers in its counts. Conversely, a low result does not mean an edit to an execution-order boundary is low consequence.

## Concrete reduction candidates

1. **Remove genuinely unreferenced private helpers, approximately 100 lines before comment/import cleanup.** `assemble_local_prompt_report` is 58 lines, `proactive_recall` 29, and `_available_bootstrap_files` 13. The first two have graph impacts above; all three have only their definition in the current source search. No public library API is implicated by their effective visibility. Keep the actively used prompt assembly and knowledge retrieval paths. This is the strongest small net-deletion pool identified, still pending compiler/differential verification.

2. **Classify retained test references as test code, approximately 120 lines plus exclusive helpers.** `sanitize_tool_result` (33), `compact_inline_tool_result` (33), `get_slim_definitions` (31), and `format_results_for_context` (24) currently have only test consumers. Use test-only placement/attributes after checking feature and macro references. Keep the tests and their independent reference logic. This reduces the non-test source pool; it is not a net repository LOC reduction and may not change the optimized executable because dead-code elimination already applies.

3. **Isolate the dormant delegated runner, roughly 1,100–1,400 non-test lines including exclusive support.** `run_tool_loop` alone is 624 lines; its scratch-pad analysis, budgets, result-formatting, and retry helpers form a larger closed implementation. `context_store::execute_ctx_summarize` adds 120 lines with callers found inside that old runner. The containing module is `pub(crate)`, so these functions are not accessible through the public crate API merely because their own declarations say `pub`.

   **Do not delete `tool_runner` wholesale.** `normalize_call_key` is used by the main loop and router. `process_tool_response` is used by live subagents and pipelines; GitNexus context independently confirms both callers. Preserve `ToolRunOutcome` where tests assert structured status/timing. Preserve any exposed micro-tool names, schema bytes, config fields, and serialized replay enum variants such as `ModelCallPurpose::ToolRunner`. A first safe version would retain the dormant code and its 2,652-line test module as a test reference, while leaving the live helpers in the production path. Net repository shrink would require a separate decision about retiring historical tests, which this assessment does not recommend as a shortcut.

4. **Consolidate a few exact duplicate bodies, tens of lines.** The regular and immutable status result appenders construct identical JSON; preserving both entrypoints while forwarding one to the other removes a few lines. `has_path_like` duplicates a 13-line predicate in `proactive.rs` and `router_fallback.rs`; sharing one implementation saves roughly 10 lines. A whole-function body scan found no large identical production implementation vein. It is not a general semantic clone detector, so this does not prove deeper duplication absent.

5. **Review repetitive test fixture setup, with no quantified saving yet.** The large agent-loop suite repeats provider/core/session construction. A shared fixture can retain every scenario, assertion, failure schedule, and name while reducing setup. Do not merge distinct failure cases into a single happy-path test or parameterize away the readable incident narrative. This may yield more net LOC than the tiny production duplicates, but no measured amount is promised by this audit.

These pools overlap: the old runner includes its formatting helpers. Do not add their upper bounds together as an estimate of guaranteed savings. The concrete investigated pool is about **1.2–1.6k non-test lines (roughly 1–2%)**, mainly reclassification; the clearly identified net deletion/consolidation pool is only about **100–150 lines** before implementation. Further **5–15%** production shrink is an investigation hypothesis, not an evidence-backed forecast. A **30–50%** production target is presently speculative and would probably require narrowing supported behavior, removing evidence, or introducing compressed abstractions whose correctness is harder to inspect.

## The delegation trap: same callee does not imply same behavior

`tool_engine.rs:643` accepts delegation provider/model arguments but calls the durable inline executor. That is deliberate hardening: raw/post-result persistence must succeed before the next effectful call.

However, `agent_loop/shared.rs:5857` takes an early return after this wrapper. The ordinary inline path below it additionally handles:

- automatic pre-execution checkpoints at approximately 5882;
- metrics based on actual new tool entries rather than routed-call count;
- pending priority user messages at approximately 5922;
- different tracing spans and surrounding state bookkeeping.

Replacing both branches with one unconditional inline call can therefore change checkpoint state, metric values, and the next prompt even though both execute the same tools. Provider initialization, health counters, and accepted delegation configuration may also remain observable. Consolidate only after defining and testing each of these current branch semantics. If a discrepancy is a bug, fix it explicitly as a behavior change; do not conceal it in a shrink patch.

## What exact behavior must mean

For the same starting state, configuration, input sequence, provider responses/stream chunks, nondeterministic inputs, and injected failures, the candidate should produce the same observable transition sequence. Successful final answers alone are not enough.

| Boundary | Required equality |
| --- | --- |
| LLM protocol | Ordered messages, role repair, tool schemas/order, synthetic markers, raw argument strings, tool-call/result pairing, missing versus null fields, request parameters, and prompt prefix bytes. Preserve both model-visible strings and serialized wire form where the transport contract requires it. |
| Tool effects and errors | Names/arguments and execution order; deny patterns, path validation, taint and permissions; structured `ok` status independent of display text; byte-stable `Error: ...` rendering; cancellation/timeout/retry decisions and side-effect counts. |
| Persistence/replay | Ordered journal events; raw/result artifact hashes and bytes; session scoping; migrations and old configuration/session loading; carrier → pre-execution → raw → post-result ordering; incomplete/corrupt replay classification; immutable result status; duplicate-effect prevention after failures. |
| LCM/capacity recovery | Same compaction trigger, selected messages, summaries, retained/active routes, route retirement, preflight results, cache reset/retraction events, capacity interruption outcome, retry budget, and pending-turn recovery. Failed or cancelled publication must not leak a committed summary or rotate state incorrectly. |
| Cancellation and concurrency | Same allowed operations before/after cancellation boundaries; same partial durable results; no later sequential side effect after persistence failure; preserved priority-message consumption and compaction task ownership. Compare allowed schedules with deterministic control, not one lucky thread interleaving. |
| Surfaces and configuration | CLI/TUI commands, channel allowlists and media/reply behavior, provider selection/endpoints, defaults/aliases/serialization, local/cloud modes, optional feature surfaces, public crate APIs, config acceptance, diagnostics relied on by scripts. |
| Performance | No correctness-neutral speed regression under the repository rule; compare matched release context-build time, TTFT, elapsed time, token counts, failures, and cache reuse. Smaller source is not evidence of faster inference or a smaller executable. |

Exact equality of wall-clock nanoseconds, OS scheduling, random UUIDs, or live stochastic LLM samples is not a sensible cross-build guarantee. Control those inputs in differential tests and separately evaluate real performance. A fixed model seed alone does not guarantee identical GPU inference. Finite test success is evidence over the exercised states, not proof for every possible conversation and machine failure.

## Smallest first slice and proof gate

Start with only the **58-line private `assemble_local_prompt_report` deletion** after a refreshed impact analysis and current-source/compiler confirmation. It has no located callers, does not remove a test oracle, and avoids touching tool/recovery state transitions. Its value is establishing the equivalence workflow, not a headline LOC win. Do not combine it with policy changes, delegation branch collapse, config cleanup, or service/model upgrades.

For subsequent live-path consolidations:

1. Freeze baseline commit, dirty patch, feature/config matrix, model/provider settings, prompt/bootstrap files, environment inputs, and hashes of independent temporary initial databases/workspaces. Keep this session's experiments intact.
2. Reuse the existing `SessionDb` exact replay artifacts and recording provider as inputs/oracles. Existing replay loading verifies recorded events and artifacts; it does **not** by itself rerun the candidate agent and prove new control-flow equivalence. Add the smallest driver needed to feed the same provider responses/stream chunks to baseline and candidate, intercept effects, and compare request/effect/event sequences after every boundary.
3. Compare full request bodies and prompt strings, effect receipts/status, journal order and artifact hashes, session state after reopen, LCM/route state, and terminal outcomes. Normalize only explicitly controlled nondeterministic identifiers/timestamps; never normalize tool argument strings, error strings, role ordering, omitted fields, or output bodies. For concurrency, test controlled interruption/publication schedules and observable ordering constraints.
4. Exercise empty/invalid responses, malformed tools, denied calls, writes followed by reads, same-tool duplicates, huge/Unicode outputs, missing/corrupt artifacts, DB failures before/after effects, stream retraction, timeout/cancellation, capacity interruption, retained preflight failure, compaction failure, session reset/reopen, and delegation-enabled priority/checkpoint cases. Include the recovery/endurance scenarios currently being developed, without treating those task-level scores as exact replay proof.
5. Run `cargo build --release` and appropriate `cargo test --release` checks, including protocol, session/replay, LCM and loop regressions. Expand to affected supported feature combinations; default features alone cannot establish whole-product equality. Run `scripts/turn_bench.sh` matched before/after when the loop/provider/context builder changes, per AGENTS.md and CONTRIBUTING.md. The AGENTS.md release-only instruction overrides older debug-mode examples in CONTRIBUTING.md.
6. Refresh/detect graph changes with `scope: all`; partial/truncated results are not a clean check. Review the diff for removed contracts and retain one independently revertible change per slice. Report measured LOC change separately for production, tests, and archived material.

The existing exact replay API, typed error/status machinery, and the many hardening tests make controlled simplification practical. Removing those to make the project look small would discard the strongest mechanism available for proving that it still works.
