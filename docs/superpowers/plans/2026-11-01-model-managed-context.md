# Model-managed context (nodecode port) — implementation plan

Date: 2026-11-01. Status: planned, not implemented. No config knobs, no feature flags.

## What ships

1. Every tool result on the wire ends with `[seq N]`, N = the message's SQLite rowid (`_db_id`) — the same id
   `lcm_expand` and `[msg N]` already use. One id namespace.
2. One tool `context({drop, keep, note})`. `drop` = `"all"` | `"12-56"` | `[12,13]`; `keep` = exemptions;
   `note` = the model's own words, appended as an assistant message `[context note] …`. Dropping a call drops
   its result and vice versa (pairs stay intact). Lossless: dropped rows stay in the LCM store; `lcm_expand`
   reads them back (no new recall tool).
3. Edits persist as an append-only `context_drops` table (monotonic ⇒ projection = union). The prompt is a
   projection of the message log minus drops; untouched blocks are byte-identical; a note-only edit is a
   pure append (prefix preserved).
4. Standing ~100-word skill in the static system prompt block that already carries `LCM_EXPAND_GUIDE`.
5. Backstop unchanged: LCM folds the remainder at `tau_hard`; the mechanical checkpoint keeps
   `[context note]` text whole (6000 chars, nodecode's `*eviction-stub-context-note-chars*`); the cloud
   handoff prompt is told to copy notes verbatim.

## Design decisions (with evidence)

**Where `_db_id` becomes known vs. when a tool result is rendered.** Tool results are injected into the
draft and persisted in the same step: `src/agent/tool_engine.rs:1075-1095` (`add_tool_result_*` then
`persist_pending_protocol_messages`), which tags each row via `set_meta(index, "_db_id", row_id)`
(`src/agent/agent_loop/shared.rs:698-700`; doc at `:602-607` "tool results immediately after injection").
The next render (`render_via_protocol`, e.g. `shared.rs:3479`) happens after that. On reload
`reconstruct_message` sets `_db_id` (`src/session/db.rs:4123-4142`) and `filter_history` preserves it
(`src/session/filters.rs:208-213`). So **the first wire render already carries the id** and a restart
renders the same bytes. Therefore the tag is applied at render time, in the one message→`Turn` chokepoint
`turn_from_legacy` (`src/agent/turn.rs:144-200`), used by both `render_to_wire` (`protocol.rs:803`) and
`render_via_protocol` (`agent_loop/heuristics.rs:42`). Messages without `_db_id` (subagent/pipeline,
un-persisted) get no tag — unchanged behaviour. Stored bytes are never mutated, so stash digests and
`is_stable_tool_result_representation` checks (`db.rs:2553-2580`) are untouched.

**Where the edit projection hooks in.** Two places, one predicate (`LcmEngine::is_dropped`):
- In-turn: after `execute_tools_inline` in `step_execute_tools` (`shared.rs:5078-5092`), the loop takes the
  edit queued by the tool, lets the engine resolve it against `ctx.messages` (every message there carries
  `_db_id` by then), then swaps the log through `TurnContext::rewrite_committed` (`shared.rs:579-600`) —
  the sanctioned hatch that prices the cache reset only when warm bytes changed. A note-only edit never
  calls it (draft append ⇒ `PrefixPreserved`).
- Turn start / restart: `prepare_context.rs:442-467` ingests history then adopts either raw `history`
  (empty DAG) or `engine.active_context()`. `active` is physically retained (dropped rows never fold into
  summaries), and the raw branch is filtered with the same predicate. Engine creation
  (`prepare_context.rs:326-343`) restores the drop set from SQLite for both the `new` and `rebuild` paths.
  `ingest` (`lcm.rs:640-650`) stores a dropped id but does not push it to `active`.

**Persistence.** New table `context_drops(session_id, message_id, created_at, PK(session_id, message_id))`
added to `SCHEMA` (`db.rs:55-183`, bootstrapped by `execute_batch(SCHEMA)` at `:1168`, no versioned
migration needed). Not `session_events`: its payload enum drives a strict replay state machine
(`db.rs:1782-2052`) and an unknown kind would need arms everywhere. Not row metadata: raw rows are
immutable by invariant. Notes are ordinary assistant rows (normal persistence, normal `_db_id`).

**Ordering vs. LCM folds.** Model edits apply first and shrink `active`; `check_thresholds` reads
`conversation_tokens()` over `active` (`lcm.rs:757-781`), so pressure drops before LCM looks. `compact()`
folds only what remains (`find_oldest_raw_block_impl`, `lcm.rs:1304`); summary `source_ids` never include
dropped ids. Dropping an id already inside a summary is a no-op (summaries have no seq; v1 does not drop
summary blocks). `lcm_expand` reads `store` (`format_expanded`, `lcm.rs:1369`) — dropped rows remain.

**Prefix-cache impact (local).** Seq tags: ~4 tokens per tool result, present from first render. Note-only
edit: append-only. Drop: re-prefill from the first dropped block (llama.cpp/LM Studio reuse the longest
common prefix); priced as `CacheResetReason::ContextEdit` via `rewrite_committed`. Standing prompt and the
new native tool schema change `messages[0]`/tool block once per binary upgrade. Existing contract test
`test_local_wire_prompt_prefix_stable_across_turns` (`agent_loop/tests.rs:5432`) must keep passing.

## Tasks

Legend: `haiku` = 1-file mechanical; `sonnet` = multi-file/logic. Independent tasks run in parallel.

### T1 — `[seq N]` on tool results · haiku · no deps
- File: `src/agent/turn.rs` (`turn_from_legacy`, `"tool"` arm ~L180-200).
- Change: when `v["_db_id"]` is a u64, `result = format!("{result}\n[seq {id}]")`. Nothing else.
- Test (turn.rs): a tool message with `_db_id: 57` → `Turn::ToolResult.result` ends with `"\n[seq 57]"`;
  the same message without `_db_id` → unchanged. Then `cargo test prefix_stable` (existing) must pass.

### T2 — `context_drops` persistence · haiku · no deps
- File: `src/session/db.rs`. Add table to `SCHEMA`; `pub async fn add_context_drops(&self, session_id,
  ids: &[usize])` (one tx, `INSERT OR IGNORE`); `pub async fn load_context_drops(&self, session_id) ->
  Vec<usize>` ordered by `message_id`. Mirror `save_summary_node`/`load_summary_nodes` (`db.rs:3403/3467`).
- Test (db.rs): add `[5,3]` then `[3,7]` → load returns `[3,5,7]`; reopen the same file → same result.

### T3 — Engine drop state + edit resolution · sonnet · no deps
- File: `src/agent/lcm.rs`. Add to `LcmEngine`: `dropped: BTreeSet<MessageId>`,
  `pending_edit: Option<ContextEdit>`. New types: `enum DropSpec { All, Ids(Vec<MessageId>) }`,
  `struct ContextEdit { drop: Option<DropSpec>, keep: Vec<MessageId>, note: Option<String> }`.
- Methods: `queue_context_edit(ContextEdit)`, `take_context_edit() -> Option<ContextEdit>`,
  `resolve_and_drop(&mut self, edit: &ContextEdit, log: &[Value], below: MessageId) -> Vec<MessageId>`
  (candidates = log rows with `_db_id < below`, role ∉ {system, developer}, not synthetic; apply
  All/Ids, subtract keep, close over `tool_calls[].id` ↔ `tool_call_id` pairs; extend `dropped`; retain
  `active`), `is_dropped(&Value) -> bool`, `restore_dropped(ids)` (sets + retains `active`), and the
  `ingest` guard (store but skip `active` when dropped). `rebuild_from_db_nodes` keeps working: restore
  runs after it. Make `parse_message_ids` `pub(crate)` (reused by T4).
- Test (lcm.rs): ingest user(1), carrier(2: calls c1,c2), results(3,4), user(5), carrier(6), result(7).
  `resolve_and_drop(drop=Ids[3], keep=[], below=6)` → returns {2,3,4} (pair closure); `active_context()`
  excludes them, `format_expanded(&[3])` still returns the body, `conversation_tokens()` decreased, and
  re-`ingest` of row 3 does not reappear in `active`. `All` with `keep=[4]` keeps only 4 among rows < 6.

### T4 — `context` tool + registration · sonnet · deps T3
- Files: new `src/agent/tools/context_edit.rs` (`ContextTool { engine: Arc<Mutex<LcmEngine>> }`, name
  `"context"`, description + schema in lcm_expand's lenient style; `execute` parses `drop` ("all" |
  string | array via `parse_message_ids`), `keep`, `note`; requires at least one of drop/note; queues the
  edit; returns one line "context edit queued: drop … keep … note …; applied before your next request").
  `src/agent/tools/mod.rs` (export). `src/agent/prepare_context.rs:361-362` (register next to
  `LcmExpandTool`). `src/agent/tools/registry.rs:742-749` add `"context"` to `CORE_NATIVE_TOOLS` and
  `:808-817` `KEEP_PARAM_DESCRIPTIONS` (local models need the native schema; cf. comment at `:733-735`).
- Test (context_edit.rs): `{"drop":"12-56","keep":[57],"note":"x"}` queues
  `ContextEdit{Ids(12..=56), keep [57], note x}`; `{"drop":"all"}` queues `All`; `{}` → `InvalidArgs`.

### T5 — Apply the edit in the loop · sonnet · deps T2, T3, T4
- Files: `src/turn_stream.rs:586` add `CacheResetReason::ContextEdit` (+ its display arm);
  `src/agent/agent_loop/shared.rs` new `async fn apply_pending_context_edit(&self, ctx)` called right after
  the infra-error check in `step_execute_tools` (`:5092`). Body: engine via the existing per-session
  lookup (`shared.rs:444-447`); `take_context_edit()` else return; `below` = `_db_id` of the carrier that
  issued the `context` call; `ids = resolve_and_drop(...)`; if non-empty: `sessions.add_context_drops`,
  `new = ctx.messages.iter().filter(|m| !engine.is_dropped(m))`, `ctx.rewrite_committed(new, ContextEdit)`;
  if note: `push_draft(assistant "[context note] {note}")` then `persist_pending_protocol_messages()`.
  (Both protocols already append a user continuation after a trailing assistant: `protocol.rs:108,244+`.)
- Test (agent_loop/tests.rs, local harness, `WireRecordingProvider`): responses = [tool_call
  `read_file` big file, tool_call `context{drop:"<seq of that result>", note:"N"}`, text]. Assert on the
  recorded calls: call 3 contains no message whose content/tool_call_id belongs to the read_file pair;
  contains an assistant message starting `[context note] N`; every message of call 3 before the first
  dropped index equals call 2's byte-for-byte (`assert_wire_prefix`-style loop); the context carrier and
  its `[seq …]` result survive. A second variant (note only) must satisfy `assert_wire_prefix(call2, call3)`.

### T6 — Turn-start projection + restart · sonnet · deps T2, T3
- File: `src/agent/prepare_context.rs`. At `:326-343` after constructing the engine (both branches):
  `engine.restore_dropped(core.sessions.load_context_drops(&session_id).await)`. At `:442-467` filter the
  raw `history` branch with `engine.is_dropped` (the `active_context()` branch is already retained).
- Test (agent_loop/tests.rs, `build_shared_db_harness` `:5486` pattern): core A runs the T5 scenario;
  core B (fresh process-equivalent over the same DB) runs one more turn: its first wire call lacks the
  dropped pair, keeps the note, and `lcm_expand` on the dropped seq returns the original body.

### T7 — Standing skill in the static prompt · haiku · no deps (names the tool from T4)
- File: `src/agent/prepare_context.rs:37-46`. Append to `LCM_EXPAND_GUIDE` (same "Context Management"
  block on both local `:113` and cloud `:132` paths) this text, adapted from nodecode `cells/context/cell.lisp`:
  > During substantial tool work, edit your context at useful checkpoints: after settling a subproblem (a
  > fix proved, a file understood) and before your final answer; not after every tool call, not while a
  > subproblem is still open (if you must, keep blocks you still use). Every tool result ends with its
  > seq, [seq 57]. Edit in one call: context({"drop":"12-56","note":"…"}) drops those blocks;
  > context({"drop":"all","keep":"57","note":"…"}) keeps only seq 57. The note is your own words, kept
  > at the end: constraints, decisions, verified state, open work, evidence pointers. Window headroom is
  > not the test: kept blocks ride every request, so drop settled ones you will not reread; a drop is
  > lossless, lcm_expand({"message_ids":"57"}) reads one back. Correctness outranks size.
- Test: existing `test_local_wire_prompt_prefix_stable_across_turns` and
  `test_continuity_note_injection_is_prefix_stable_across_turns` pass (prompt stays byte-stable across
  turns); `tests.rs:2581/2737` substring guards still hold.

### T8 — Notes survive folds · haiku · no deps
- Files: `src/agent/lcm.rs:2114` `mechanical_assistant_line`: text starting `[context note]` clips at a new
  `MECHANICAL_NOTE_MAX_CHARS = 6_000` instead of `MECHANICAL_TEXT_MAX_CHARS`. `src/agent/compaction.rs:48-72`
  `SUMMARIZE_PROMPT` (and `_ADVANCED`): add bullet "- every `[context note]` block verbatim: it is the
  model's own checkpoint of what was leaving".
- Test (lcm.rs, next to `mechanical_checkpoint_keeps_reasoning_and_drops_large_results` `:4425`): a span
  with a 3000-char `[context note]` assistant row and a 3000-char ordinary assistant row → the checkpoint
  contains the full note and the ordinary text clipped to 1200.

### T9 — Benchmark (before/after) · sonnet · deps T1–T8
- Add `experiments/model-managed-context/run.sh` + `README.md`: one matched tool-heavy task (e.g. 12 steps:
  read 6 files under `src/agent/tools/`, run `cargo test -- lcm::tests` twice via `exec`, then answer a
  factual question whose answer lives in step 2's file) driven by `nanobot agent -m` against the same
  session twice per arm. Arm A = `main` binary (git worktree build), arm B = this branch. Same model, same
  server, cold server per arm.
- Metrics from `~/.nanobot/metrics.jsonl` (`RequestMetrics.prompt_tokens`, `cache_read_tokens`,
  `src/agent/metrics.rs:16-45`) and `sessions.db` (`session_events` model_request count; reuse the
  `summarize()` reader in `experiments/context-recovery/endurance.py:16`). If the local server reports no
  cached-token usage, use the `PromptDelta` log (`AppendOnly` vs `Diverged`) as the cache-hit proxy.
- Report table (local; cloud too if an API key is present):

  | arm | requests | mean prompt tok/request | prefill cache hit % | total prompt tok | task success |
  |-----|----------|-------------------------|---------------------|------------------|--------------|

  Lead with the end-to-end numbers. nodecode claims no cost improvement; a loss here is a valid result and
  must be stated, not buried. Acceptance: script runs both arms unattended and prints the table.

## Dependency graph

```
T1  T2  T3  T7  T8        (parallel, wave 1)
     \  / \
      T4   T6             (wave 2: T4 needs T3; T6 needs T2+T3)
       |
      T5                  (wave 3: T2+T3+T4)
       |
      T9                  (wave 4)
```

## Out of scope (v1)
Dropping or restoring summary blocks; file-based context editing; undo/restore of drops (recall covers
reads); per-turn eviction stubs beyond the existing LCM checkpoint; any config surface.

## Quality gates to watch
G1: `DropSpec`/`ContextEdit` are enums/structs, not bool flags. G2: one predicate `is_dropped`, one
resolver, one tag site (`turn_from_legacy`). G4: `apply_pending_context_edit` ≤ 40 lines — resolution lives
in the engine. Run `./quality-sentinel.sh --full` before each commit; run GitNexus `impact` on
`turn_from_legacy`, `step_execute_tools`, `LcmEngine::ingest`, `rebuild_from_db_nodes` before editing.
