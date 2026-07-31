# Memory & Session Subsystem — Harden + Rationalize (Design)

**Date:** 2026-07-31
**Status:** Design (awaiting user review)
**Scope decision:** harden + rationalize the tool API; defer the vector/semantic pipeline and the MEMORY.md structured-facts migration.
**Approach:** B — trust-ranked unified `recall` + structural `MissingArg` + recovery rename with alias shim.

## 1. Context & goals

The memory/search surface has 8 tools across 3 stores with no shared query layer.
Recent incidents showed weak local models (qwen36-35b-a3b, temp 0) repeatedly
emit empty-arg or wrong-tool calls because the call shape is not shown where
they look, and the corrective signals (empty-arg errors, dedup messages) were
passive rather than corrective. Three parallel research agents mapped the
subsystem end-to-end (write path, index/store backends, tool API surface); the
findings below reframe the redesign around its real leverage: **the data layer
is solid and already indexed; the failure surface is the tool API.**

### Goals (the five questions)
1. **Ingest:** what we store per turn (verbatim, append-only) — established (§2).
2. **Indexing:** what gets indexed, when — established (§2).
3. **Unify search** into a simple, powerful API across all stored data.
4. **Batched ops** to cut round-trips (and footgun exposure).
5. **Ergonomics** so failure is rare to non-existent.

### Non-goals (deferred)
- Wiring the live embedding/vector pipeline (`chunks_vec` exists but is dead in
  prod; `semantic` feature is off). The `recall` API is shaped to accept it
  later without a redesign.
- Restructuring MEMORY.md into indexed structured facts (tags/categories).
  `remember` keeps the bullet format; an optional `tags` field may later
  serialize as `#tag` markers with no migration.

## 2. Current state (as-is, verified)

### Three stores, three backends, no shared query layer
| Store | Index | Populated | Read by |
|---|---|---|---|
| `sessions.db` (`messages`) | FTS5 `porter unicode61`, AFTER-INSERT trigger | inline, per message | `session_search` only |
| `knowledge.db` (`chunks`) | FTS5 bare `unicode61` (no Porter) + vec0 cosine 384d | inline; CLI `ingest` = text-only | `recall` (fallback), CLI `search` |
| `MEMORY.md` | none — linear substring scan of `- ` lines | inline atomic write | `recall` (primary) |

### Ingest guarantees (Q1)
`messages` is **true append-only** (zero `UPDATE`; deletes only in admin `nuke`).
User + assistant text stored verbatim. Tool results capped @ 8000 B in
`messages`, but **full raw bytes** survive in `tool_results`, recoverable via
`recall_tool_result`. LCM compaction *appends* `summary_nodes`; never mutates
raw rows.

### Indexing (Q2)
All indexing is **inline** (no background task). Session FTS fires via SQL
trigger on INSERT. Knowledge FTS fires via trigger on chunk INSERT. MEMORY.md
has **no index** (linear scan). `chunks_vec` is created at open time but
`ingest_with_embeddings` is called only from tests — vector/hybrid search
silently degrades to BM25 in production.

### Failure surface (why loops happen)
- `remember` has `required:[]` **and** its empty-arg path returns SUCCESS
  (defaults to `list`) → positive feedback → loop. Its error phrase
  ("Missing required parameter") bypasses the worked-example substring gate.
- `lcm_expand` bypasses the same gate ("no valid message IDs provided").
- The worked-example augmentation gates on the literal substring `"is required"`
  (registry.rs:677) — fragile; two tools already drift from it.
- Only 2/7 tools have a worked call-shape in the system-prompt Memory section.
- No batch anywhere — 5 facts = 5 calls = 5 footgun exposures.
- Naming collisions: `recall` vs `recall_tool_result`; four `search_*` tools.

## 3. Target design

### 3.1 Tool surface (8 → 6; search surface 3 → 1)

| New tool | Role | Absorbs / renames | Hidden alias? |
|---|---|---|---|
| `recall` | unified retrieve: query-search across all stores **or** fetch by `session`/`message_ids` | absorbs `session_search` + `search_context` | `session_search`, `search_context` → routed to `recall` |
| `remember` | write facts (batch, hardened) | — (stays) | — |
| `lcm_expand` | decode compressed summary blocks by ID | — (stays; distinct op) | — |
| `fetch_tool_output` | recover a stashed truncated output by `tool_call_id` | renames `recall_tool_result` | yes |
| `grep_tool_output` | grep within a stashed output | renames `search_tool_result` | yes |
| `slice_tool_output` | line-range a stashed output | renames `slice_tool_result` | yes |

The confusing part of the surface was never the mechanical retrieval tools — it
was **which search tool?** That collapses to always-`recall`.

### 3.2 `recall` contract

| param | type | when |
|---|---|---|
| `query` | str | search intent (required unless `session`/`message_ids`/`mode` present) |
| `scope` | enum `memory`\|`files`\|`sessions`\|`all` | optional, **default `all` (trust-ranked)** |
| `n` | int | optional per-source cap |
| `session` | str | fetch mode — dump one session by key |
| `message_ids` | str | extract mode — e.g. `"5-12"` |
| `mode` | `"latest"` | list recent sessions |

- `required: []` (multi-mode). Made safe by the structural `MissingArg` fix:
  `recall({})` → typed error enumerating the three entry params, each with a
  worked shape. Empty-arg is self-correcting, not a loop.
- **Trust-ranking (the dissolve safety net):** when `scope=all`, query each
  store, merge **curated memory > knowledge docs > workspace files > raw
  sessions**, per-source cap ~3, total cap ~10, output cap 8000 chars. Section
  headers in the result (`## Curated memory`, `## Knowledge docs`, `## Workspace
  files`, `## Past conversations`) make source + trust visible. Canonical facts
  surface first and cannot be drowned by stale transcripts — the guardrail's
  intent preserved via ranking instead of a hard tool boundary.
- Param-dispatch picks intent: `query`→search; `session`/`message_ids`→fetch;
  `mode:"latest"`→list. One tool, params decide. Fetch modes inherit the
  existing session-dump cap (16000 chars) and extract cap (≤20 ids).
- `scope=memory` covers both MEMORY.md and knowledge.db (curated + ingested
  reference), matching today's `recall` primary+fallback behavior.

### 3.3 `remember` contract

| param | type | notes |
|---|---|---|
| `facts` | array[str] | **batch write**; ≤20 facts, each ≤180 chars (kept — per-line MEMORY.md format constraint) |
| `action` | enum `add`(default)\|`replace`\|`delete`\|`dedupe` | maintenance stays, `add` is the dominant path |
| `old_fact`/`new_fact` | str | for `replace` |
| `limit` | int | for `dedupe` |

Three contract changes (each kills a documented failure mode):
- **`list` action removed** → reads go to `recall(scope=memory)`. Enforces
  write/read symmetry; removes the action-enum ambiguity.
- **empty-arg → `MissingArg` ERROR** (was: *success* defaulting to `list` — the
  single worst loop hazard). For `action=add`, `facts` is required.
- **batch `facts:[...]`** — one atomic write for N facts.

### 3.4 Structural `MissingArg` (root-cause fix)

`Tool::execute` changes from `String` to `Result<String, ToolError>`, where:

```rust
enum ToolError {
    MissingArg { param: String, example: String },
    Other(String),
}
```

The registry matches `ToolError::MissingArg { param, example }` (not prose) and
appends the canonical call-shape from `example`. Every tool's empty-arg path
becomes self-correcting in one mechanism — `recall`, `remember`, `lcm_expand`
fixed together. Tools return `Ok(String)` for success, `Err(ToolError::MissingArg{..})`
for a correctable empty/wrong-arg call, `Err(ToolError::Other(s))` for other
errors (the existing "Error: "-prefixed strings).

### 3.5 Unify the two FTS tokenizers
Align knowledge.db's `chunks_fts` from bare `unicode61` to `porter unicode61` to
match `messages_fts`, so the same query matches consistently across stores now
that `recall` searches both. Because `chunks_fts` is external-content
(`content='chunks'`), migration is drop+recreate+reindex with **no data loss**;
auto-detect the mismatch at open and rebuild once (idempotent).

### 3.6 Worked call-shapes for all tools (proactive coverage)
Every memory/search tool gets a verified worked call-shape in (a) its
`description()` and (b) the system-prompt Memory section (context.rs), generated
from the tool's schema as the single source of truth so it cannot drift.
`MissingArg` is reactive (fixes the error); worked-shapes are proactive (prevent
it).

### 3.7 Alias shim (replay back-compat)
Registry holds a hidden alias map: `session_search`/`search_context`/
`recall_tool_result`/`search_tool_result`/`slice_tool_result` → their new
targets. Aliases are registered (so old session `tool_calls` rows replay) but
**excluded from the catalog** (never advertised to the model). `session_search`
routes by which param is present (`query`→recall search;
`session`/`message_ids`→recall fetch).

## 4. Verification — unit tests AND live e2e (mandatory)

**Rule: no contract change is "done" until the live e2e passes.** Unit tests
prove general logic; live e2e proves behavior against `local:qwen36-35b-a3b`.
The e2e harness is the `nanobot agent -l -s <session> -m "<prompt>"` subcommand,
whose audit trail at `~/.nanobot/workspace/memory/audit/<session>.jsonl`
(hash-chained `tool_name` + `arguments` + `result_data`) is the deterministic
evidence source.

| # | Change | Unit test (general logic) | Live e2e (behavioral proof) |
|---|---|---|---|
| 1 | `MissingArg` is structural | parametrized: any tool returning `MissingArg{param,example}` → registry appends the worked shape from `example` | provoke an empty-arg call via a weak prompt; audit shows the corrective example appended AND the model's next call is well-formed (no loop) |
| 2 | `remember` empty-arg is ERROR | `remember({})` action=add → `MissingArg`; `action=list` rejected | ask the live model to "remember AGI bonsai"; audit shows `remember({"facts":["..."]})` first-try, not an empty-call loop |
| 3 | trust-ranking protects facts | `recall(scope=all)` matching MEMORY.md + sessions → memory hits first | seed a MEMORY.md fact + a session on the same topic; ask "what do you know about X"; audit shows `recall` and the model cites the canonical fact, not stale chat |
| 4 | alias shim resolves | old `recall_tool_result({tool_call_id})` runs via alias; alias names absent from catalog | replay an old session tool_call through the new build; it still resolves; `get_tools` catalog omits the old names |
| 5 | batch write | `remember({facts:[a,b,c]})` writes 3 atomically | "remember these three things…"; audit shows one `remember` call with a 3-fact array, MEMORY.md gains all three |
| 6 | FTS tokenizer parity | stemming-sensitive query ("running"↔"ran") matches in both stores post-rebuild | ingest a doc containing "running"; `recall({"query":"ran"})` finds it |

## 5. Migration & risk

- **Trait change is the widest blast radius.** `String → Result<String, ToolError>`
  touches ~all tool impls. Land as **one atomic commit; let `rustc` drive** —
  change the trait, the compiler enumerates every impl/caller to fix (same
  technique as the Phase-09 Wave-4 field deletion). Existing string-asserting
  tests adjust mechanically.
- **`remember` list removal / `session_search`+`search_context` dissolution** —
  logic migrates into `recall`; old tool files become thin alias dispatchers (or
  delete + alias-map entries). No internal caller depends on `remember(action=list)`.
- **knowledge.db FTS rebuild** — auto on open, idempotent, no data loss.
- **Aliases** keep old session `tool_calls` replaying — no history migration.

## 6. Order of operations (for the implementation plan)

1. Trait change + `MissingArg` (foundation, compiler-driven) + e2e #1.
2. `remember` hardening + batch + e2e #2, #5.
3. Dissolve into `recall` + alias shim + trust-ranking + e2e #3, #4.
4. FTS tokenizer unification + worked-shapes + e2e #6.

Each step ships independently green (unit + e2e).

## 7. Open questions for review

- Trust-ranking source order (memory > knowledge > files > sessions) — confirm
  or reorder.
- Per-source cap (~3) and total cap (~10) — confirm numbers.
- `remember` 180-char per-fact cap — keep as the format constraint, or raise?
