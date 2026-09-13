# Nanobot model behavior and recovery study

Read-only study, 13 September 2026. No source changes, model experiments, configuration changes, or messages to the running session.

**Conclusion:** the observed failures combine model reasoning mistakes, missing or ambiguous tool guidance, and gaps between loop protection and useful recovery. Predictable mistakes can be prevented or bounded without weakening execution, evidence, or replay invariants. This trace establishes concrete interface defects and repeated behavior; it does not establish how much a guidance change will improve this model without a controlled comparison.

## Evidence and scope

Session `20260912_223710_6c7830`, model `local:escha-35b-a3b`, temperature 0. The user asked the model to inspect existing recall/remember, durable records, `/learn`, and commit history before proposing RSI work. The frozen current-turn sample spans message 72591 through 72665, 23:17:10–23:27:15 UTC: **37 tool calls: 24 exec, 5 read_file, 8 inspect_tool_result.** Additional read-only observations through message 72687 show a second search repetition loop. Not every call was redundant: different commits, advancing result pages, and checking current implementation were legitimate work.

Evidence snapshots: [messages](/private/tmp/nanobot-study/messages.json), [events](/private/tmp/nanobot-study/events.json), [first model request](/private/tmp/nanobot-study/first-request.json), [later model request](/private/tmp/nanobot-study/last-request.json). The requests contain the actual model-facing schemas and history. They do not prove that the inference backend consumed its retained KV state correctly. No cache-disabled model comparison was run. The repeated calls have distinct model-generated call IDs and argument variants; this is not evidence of duplicated streaming fragments.

## What the model did, and what the interface contributed

1. **Useful output looked like a generic failure.** `git log --oneline --all | head -50` produced 50 useful lines plus exit 141 under pipefail. A read-only reproduction returned 141; `git log --oneline --all -50` returned 0 with 50 lines. The trace contains repeated variants of early-closing Git pipelines. Neither the system prompt nor the full exec description explains pipefail. The executor returns `Error: Command failed` and the code, without explaining possible downstream early closure. This is actionable missing feedback, but 141 must remain a failure status: it is not universally harmless.

2. **The model also repeated successful searches.** Messages 72610 and 72612 request the same command and explicit working directory, receive the same commit list, and 72614 repeats it again. The third call is rejected. The model then changes syntax (`wc`, `head`, redirection, filters) rather than resolving the question. Later it repeats a source grep, adds line numbers, then redirection, repeatedly finding the same comment. Thus SIGPIPE alone cannot explain the behavior.

3. **There is a genuine reasoning error.** The model says Git `-S` is matching “recovery” rather than “rsi.” `-S` concerns occurrence changes in diffs, not a substring filter on commit subjects. Better error wording cannot guarantee correct Git semantics. The model eventually distinguishes historical plans from current implementation, which is positive, but takes an unnecessarily long route.

4. **Source guidance is not the delivered contract.** [registry.rs:557](/Users/peppi/Dev/nanobot-rs/src/agent/tools/registry.rs:557) truncates tool descriptions to two sentences. The actual exec schema contains only its purpose and the zsh/BSD note. Later skill and file-tool guidance in [shell.rs:414](/Users/peppi/Dev/nanobot-rs/src/agent/tools/shell.rs:414) disappears. Pipefail guidance is absent even before truncation. Essential operating rules need an explicit concise representation that survives serialization, rather than relying on sentence position.

5. **Discovery and completion markers add friction.** A 4.6 KB skill catalog is externalized, but its receipt excerpt is just `<skills>`. The model queries that tag and gets only that tag, then pages to discover the names. A file read later says `139 more lines; next: read_file lines="167:305"`, while its inspection wrapper says `END OF SOURCE` and “answer from what you have.” End of the stored result is not end of the file. The raw file marker is in [filesystem/mod.rs:1793](/Users/peppi/Dev/nanobot-rs/src/agent/tools/filesystem/mod.rs:1793); wrapper wording is in [stash_search.rs:420](/Users/peppi/Dev/nanobot-rs/src/agent/tools/stash_search.rs:420). These are demonstrable ambiguities; their causal contribution is not yet measured.

6. **Clear bounded instructions often worked.** The model correctly followed `next_char` through 3713, 7423, and completion, and followed contiguous read_file line ranges. This supports improving local recovery instructions, though it is not a controlled comparison with the search failures.

## Guard and invariant findings

The source already has exact-call replay, duplicate limits, a consecutive zero-execution stop, repeated-round detection, and a 96-success lease. Those are useful safeguards, but they do not establish evidence progress. Variants reset exact-match streaks; failed executions release success-lease reservations. Main iteration limits still exist, so the trace should not be described as an unbounded infinite loop.

Two issues need investigation before increasing reliance on replay:

- [tool_guard.rs:118](/Users/peppi/Dev/nanobot-rs/src/agent/tool_guard.rs:118) says an exact cached success is replayed before checking duplicate limits. The observed identical explicit-cwd successes instead execute twice and then reject. Both successes have durable execute/post events with the same result digest; the third has a rejection and no execution. The per-turn guard lifecycle and its read-cache invalidation do not explain this under current source. This is an unresolved runtime/source discrepancy, not proof of an old binary or a known root cause.
- [shared.rs:5337](/Users/peppi/Dev/nanobot-rs/src/agent/agent_loop/shared.rs:5337) routes and deduplicates before injecting an omitted working directory at line 5364; [tool_engine.rs:1213](/Users/peppi/Dev/nanobot-rs/src/agent/tool_engine.rs:1213) caches the resulting arguments. Consequently omitted-cwd lookup and storage keys can differ. This source defect does **not** explain the explicit-cwd trace above.

Replay also loses useful durable typing: the live Replay disposition is folded into rejected-call persistence, and short cached results lack a source-call reference. Preserve a distinct `CachedReplay` disposition with source call ID and result digest; never describe it as a newly executed operation. An additional canonical call-key digest would improve correlation without rewriting command bytes.

An adjacent truthfulness issue is outside the observed loop: capacity-interruption handling ignores a persistence failure while saying partial output “was saved” ([shared.rs:2480](/Users/peppi/Dev/nanobot-rs/src/agent/agent_loop/shared.rs:2480)). Recovery must not make that claim unless persistence succeeded.

## Proposed prevention and recovery

**First, repair and verify the execution contract.** Resolve execution defaults once before guard lookup and execution, preserving the original requested arguments separately where needed. Test the complete route→execute→persist→replay path. Resolve the explicit-cwd discrepancy with canonical-key and source-result diagnostics. Preserve cache invalidation and state freshness; do not expand reuse to arbitrary “similar” commands.

**Before use, deliver concise, explicit operating guidance.** Keep essential rules in the actual wire schema. Explain pipeline behavior and producer-native output limits. Make skill discovery expose a useful bounded names/descriptions index, while retaining full instructions on demand. Describe when to list versus load a known skill. Distinguish stored-result completion from file completion and give the applicable next action.

**During use, supply truthful structured outcomes and recovery hints.** Retain exact stdout, stderr, exit status, and durable raw results. Explain possible SIGPIPE without declaring success or automatically changing the command. Structured search can distinguish no matches from an execution failure; arbitrary exec exit 1 cannot be globally reclassified. Duplicate feedback should identify the prior evidence and ask what remains unresolved. Replace unsupported “you already have all the data” assertions with a request to synthesize what is known and state gaps.

**Use progress signals for bounded correction, not unsafe deduplication.** Repeated identical evidence from recognized read-only operations can trigger a nudge to change the investigation or synthesize. Advancing cursors, different commits, and changed workspace state count differently. Equal output alone does not establish equivalent operations: unrelated searches can both be empty. Never skip, retry, or rewrite arbitrary shell commands or writes using this heuristic.

**Bound recovery structurally.** Add or consolidate a durable per-turn admitted-attempt bound that counts failed as well as successful executions, alongside the existing success lease. Couple persistent lack of progress to a bounded recovery opportunity, then an honest partial answer stating what could not be established. Preserve budget accounting through batches, interruption, compaction, and restart. Avoid a second execution pipeline or permanent model-specific behavior flags.

Non-negotiable invariants: original tool bytes/status remain attributable; cached evidence has explicit provenance and valid freshness; no duplicate side effects from retries; every call has correctly paired ordered protocol records; failures are not relabeled successes; workspace, permission, and taint checks remain intact; incomplete work is never reported as complete.

## How to establish that the changes work

Use the frozen model-request prefixes and reproducible read-only tool responses with the same Escha model/settings. Compare baseline, guidance-only, feedback-only, and the combined bounded recovery design in an evaluation harness. This study has not run those interventions.

Measure correct answers about current implementation, redundant calls, new evidence acquired, model tokens, wall time, false stops, and unresolved claims. In integration tests cover three identical successful calls, omitted versus explicit cwd, legitimate new reads after mutation, advancing pagination, equal empty results for different queries, real pipeline failures, writes, failed persistence, and restart/compaction. Assert execution counts, durable receipts, source IDs, and budgets—not just final wording.

This would distinguish an ergonomics improvement from a model-capability limit or inference-state defect. The goal is fewer wasted decisions with the same or stronger evidence and execution guarantees, not a larger prompt or a weaker guard.
