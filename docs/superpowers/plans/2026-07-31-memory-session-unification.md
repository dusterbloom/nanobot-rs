# Memory & Session Unification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden + rationalize the memory/search tool API — one trust-ranked `recall`, batched `remember`, structural `MissingArg`, renamed recovery tools with alias back-compat — so weak local models stop empty-arg looping.

**Architecture:** Extend the existing `ToolErrorKind`/`error_kind` slot (no `execute()` signature change) to make missing-arg errors self-correcting; dissolve `session_search`+`search_context` into `recall` with a trust-ranked merge that preserves the "canonical facts outrank stale transcripts" guardrail; reuse the existing `normalize_tool_request` for the recovery-tool rename aliases; unify the two FTS tokenizers; add worked call-shapes to both identity paths.

**Tech Stack:** Rust 2021, async-trait + tokio, SQLite (rusqlite), FTS5 (`porter unicode61`), serde_json. Tools return `String`; structured outcomes via `ToolExecutionResult { ok, data, error, error_kind }`.

**Spec:** `docs/superpowers/specs/2026-07-31-memory-session-unification-design.md`

## Global Constraints

- Rust 2021, `snake_case` fns, `PascalCase` types, `SCREAMING_SNAKE` consts. `anyhow::Result` at app layer; tools return `String`, prefix errors with `"Error: "`.
- "Touch only what you must." "The second occurrence of a piece of logic triggers extraction, not the third." Comment only where the LLM-protocol contract is non-obvious.
- Build: `cargo build`. Tests: `cargo test --lib` (full), plus focused modules while iterating.
- **Every task ends with a live e2e** against `local:qwen36-35b-a3b` — no task is "done" until the audit trail proves it. Harness: `./target/release/nanobot agent -l -s cli:e2e-<task> -m "<prompt>"`; evidence at `~/.nanobot/workspace/memory/audit/cli_e2e-<task>.jsonl` (hash-chained `tool_name`+`arguments`+`result_data`). Rebuild before each e2e: `cargo build --release`.
- One commit per task. Branch already exists (`refactoring/maximum-speed-with-less-code`); commit on it.

## File Structure

- **Modify** `src/errors.rs` — add `ToolErrorKind::MissingArg { param, example }`.
- **Modify** `src/agent/tools/registry.rs` — switch the augmentation gate from substring to `error_kind`; add rename/dissolution aliases to `normalize_tool_request`; update `RARELY_ADVERTISED_TOOLS`.
- **Modify** `src/agent/tools/remember.rs` — remove `list`, add batch `facts`, set `MissingArg` on empty-arg.
- **Modify** `src/agent/memory.rs` — add `append_facts_batch` (atomic read-modify-write).
- **Modify** `src/agent/tools/recall.rs` — absorb `session_search`+`search_context`; add `scope`, trust-ranking, fetch modes.
- **Delete** `src/agent/tools/session_search.rs`, `src/agent/tools/search_context.rs` — logic migrated to `recall`; aliases in `normalize_tool_request` keep old names resolving.
- **Modify** `src/agent/tools/recall_tool_result.rs`, `src/agent/tools/stash_search.rs` — `name()` returns new names (`fetch_tool_output` / `grep_tool_output` / `slice_tool_output`).
- **Modify** `src/agent/knowledge_store.rs` — `chunks_fts` tokenizer → `porter unicode61`; rebuild-on-open if mismatch.
- **Modify** `src/agent/context.rs` — worked call-shapes in **both** identity paths (local lines ~1272-1288 + non-local Memory section ~1322-1332).

---

## Task 1: Structural `MissingArg` (foundation)

**Files:**
- Modify: `src/errors.rs:90-117` (enum), `src/agent/tools/registry.rs:677-682` (gate), `src/agent/tools/registry.rs:3259-3308` (test).
- Test: `src/agent/tools/registry.rs` (unit), live e2e (regression).

**Interfaces:**
- Produces: `crate::errors::ToolErrorKind::MissingArg { param: String, example: String }`; registry appends `example` for any `ToolExecutionResult` whose `error_kind` is `MissingArg` (and, back-compat, any legacy `"is required"` string).

- [ ] **Step 1: Add the enum variant.** In `src/errors.rs`, inside `pub enum ToolErrorKind` (after `ServiceUnavailable`):

```rust
    #[error("Missing required argument '{param}'; call as {example}")]
    MissingArg { param: String, example: String },
```

(`PartialEq, Eq` derive already present on the enum; `String` fields are `Eq` — compiles.)

- [ ] **Step 2: Write the failing test.** Add to `src/agent/tools/registry.rs` tests module:

```rust
    /// Structural MissingArg: a tool that sets error_kind=MissingArg gets the
    /// worked example appended EVEN when its data string lacks "is required"
    /// (the failure mode that left remember/lcm_expand unaugmented).
    #[tokio::test]
    async fn missing_arg_error_kind_appends_structured_example() {
        struct StructuredMissingArg;
        #[async_trait]
        impl Tool for StructuredMissingArg {
            fn name(&self) -> &str { "structured_missing_arg" }
            fn description(&self) -> &str { "test" }
            fn parameters(&self) -> serde_json::Value {
                serde_json::json!({"type":"object","properties":{"facts":{"type":"array"}}})
            }
            async fn execute(&self, _: HashMap<String, serde_json::Value>) -> String {
                "Error: provide facts".to_string() // no "is required" substring
            }
            async fn execute_with_result(
                &self,
                _: HashMap<String, serde_json::Value>,
            ) -> ToolExecutionResult {
                ToolExecutionResult {
                    ok: false,
                    data: "Error: provide facts".to_string(),
                    error: None,
                    error_kind: Some(crate::errors::ToolErrorKind::MissingArg {
                        param: "facts".to_string(),
                        example: r#"structured_missing_arg({"facts":["..."]})"#.to_string(),
                    }),
                }
            }
        }
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(StructuredMissingArg));
        let result = registry.execute("structured_missing_arg", HashMap::new()).await;
        assert!(!result.ok, "must still report failure");
        assert!(
            result.data.contains(r#"Call as structured_missing_arg({"facts":["..."]})"#),
            "structural MissingArg must append the example from error_kind: {}", result.data
        );
    }
```

- [ ] **Step 3: Run test to verify it fails.** `cargo test --lib missing_arg_error_kind_appends_structured_example` → FAIL (no augmentation: the data has no "is required" substring, so the old gate doesn't fire).

- [ ] **Step 4: Switch the gate.** Replace the block at `registry.rs:677-682`:

```rust
        // Before (substring gate):
        // if !result.ok && result.data.contains("is required") {
        //     if let Some(example) = Self::worked_example_call(&name, &tool.parameters()) {
        //         let base = result.data.trim_end_matches('.');
        //         result.data = format!("{}. Call as {}.", base, example);
        //     }
        // }

        // After (structural, with substring back-compat):
        if !result.ok {
            let example = match &result.error_kind {
                Some(crate::errors::ToolErrorKind::MissingArg { example, .. }) => Some(example.clone()),
                _ if result.data.contains("is required") => Self::worked_example_call(&name, &tool.parameters()),
                _ => None,
            };
            if let Some(example) = example {
                let base = result.data.trim_end_matches('.');
                result.data = format!("{}. Call as {}.", base, example);
            }
        }
```

- [ ] **Step 5: Run the new test + the existing augmentation test.**
`cargo test --lib missing_arg_error_kind_appends_structured_example` → PASS.
`cargo test --lib test_missing_required_arg_error_appends_schema_derived_example` → PASS (back-compat path intact).

- [ ] **Step 6: Full lib suite.** `cargo test --lib` → green (no regressions; existing substring-based tools unchanged).

- [ ] **Step 7: Commit.**
```bash
git add src/errors.rs src/agent/tools/registry.rs
git commit -m "feat(tools): structural MissingArg via ToolErrorKind

Add ToolErrorKind::MissingArg{param,example}; registry augmentation gate
now matches error_kind structurally (with substring back-compat). Tools
that set MissingArg get a corrective worked example even when their error
string lacks 'is required' — fixes the remember/lcm_expand blind spot."
```

- [ ] **Step 8: e2e (regression).** `cargo build --release && ./target/release/nanobot agent -l -s cli:e2e-t1 -m "recall an empty topic:"`. The structural path isn't exercised by a live tool yet (no tool sets `MissingArg` until Task 2) — this e2e only confirms existing substring-path augmentation (recall's `'query' is required`) still fires live. Audit `cli_e2e-t1.jsonl`: if recall was called empty, its result_data contains "Call as recall(...)". (The structural-path live proof lands in Task 2.)

---

## Task 2: Harden `remember` + batch writes

**Files:**
- Modify: `src/agent/memory.rs` (add `append_facts_batch`), `src/agent/tools/remember.rs` (schema, dispatch, MissingArg, batch).
- Test: `src/agent/tools/remember.rs` unit tests; live e2e.

**Interfaces:**
- Consumes: `ToolErrorKind::MissingArg` (Task 1).
- Produces: `MemoryStore::append_facts_batch(&self, facts: Vec<String>) -> Result<usize, String>`; `remember` schema drops `list`, adds `facts: array`.

- [ ] **Step 1: Add `append_facts_batch` to `MemoryStore`.** In `src/agent/memory.rs`, alongside `write_long_term`:

```rust
    /// Append N facts atomically (read-modify-write under the file mutex +
    /// atomic rename). Each fact becomes one `- ` bullet. Returns count added.
    pub async fn append_facts_batch(&self, facts: Vec<String>) -> Result<usize, String> {
        let _guard = self.long_term_lock.lock().await;
        let path = self.long_term_path();
        let mut content = tokio::fs::read_to_string(&path)
            .await
            .map_err(|e| format!("read MEMORY.md: {e}"))?;
        let mut added = 0usize;
        for fact in facts {
            let fact = fact.trim();
            if fact.is_empty() { continue; }
            if !content.ends_with('\n') && !content.is_empty() { content.push('\n'); }
            content.push_str("- "); content.push_str(fact); content.push('\n');
            added += 1;
        }
        let tmp = path.with_extension("md.tmp");
        tokio::fs::write(&tmp, &content).await.map_err(|e| format!("write tmp: {e}"))?;
        tokio::fs::rename(&tmp, &path).await.map_err(|e| format!("rename: {e}"))?;
        Ok(added)
    }
```

(If `MemoryStore`'s mutex/path accessors differ from `long_term_lock`/`long_term_path`, adapt to the actual private fields — the existing `write_long_term` shows the canonical pattern at memory.rs:47-53.)

- [ ] **Step 2: Write failing tests** in `remember.rs` tests module:

```rust
    #[tokio::test]
    async fn empty_add_returns_structural_missing_arg() {
        let dir = tempfile::tempdir().unwrap();
        let tool = RememberTool::new(dir.path().to_path_buf());
        let res = tool.execute_with_result(HashMap::new()).await;
        assert!(!res.ok);
        assert!(matches!(
            res.error_kind,
            Some(crate::errors::ToolErrorKind::MissingArg { ref param, .. }) if param == "facts"
        ));
    }

    #[tokio::test]
    async fn batch_add_writes_all_facts_atomically() {
        let dir = tempfile::tempdir().unwrap();
        let tool = RememberTool::new(dir.path().to_path_buf());
        let mut args = HashMap::new();
        args.insert("facts".to_string(), json!(["alpha", "bravo", "charlie"]));
        let out = tool.execute(args).await;
        assert!(out.starts_with("Remembered 3 fact"));
        let mem = tokio::fs::read_to_string(dir.path().join("memory").join("MEMORY.md"))
            .await.unwrap();
        assert!(mem.contains("- alpha") && mem.contains("- bravo") && mem.contains("- charlie"));
    }

    #[tokio::test]
    async fn list_action_is_rejected_with_recall_redirect() {
        let dir = tempfile::tempdir().unwrap();
        let tool = RememberTool::new(dir.path().to_path_buf());
        let mut args = HashMap::new();
        args.insert("action".to_string(), json!("list"));
        let out = tool.execute(args).await;
        assert!(out.contains("recall") && out.contains("scope"), "must redirect to recall: {out}");
    }
```

- [ ] **Step 3: Run, verify fail.** `cargo test --lib remember::` → FAIL (old `list`-default returns success; no batch).

- [ ] **Step 4: Rewrite `parameters()`** — drop `list` from the enum, add `facts`:

```rust
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "replace", "delete", "dedupe"],
                    "description": "Memory operation. Default: add. (Reading memory: use recall with scope=\"memory\".)"
                },
                "facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 20,
                    "description": "One or more concise facts to add (batch). Each ≤180 chars."
                },
                "old_fact": {"type": "string", "description": "Exact fact to replace when action='replace'"},
                "new_fact": {"type": "string", "description": "Replacement fact when action='replace'"},
                "limit": {"type": "integer", "description": "Max facts for dedupe. Default 50."}
            },
            "required": []
        })
    }
```

- [ ] **Step 5: Rewrite the dispatch + add an `execute_with_result` override** that sets `MissingArg` when there's nothing to act on:

```rust
    async fn execute_with_result(&self, args: HashMap<String, Value>) -> ToolExecutionResult {
        let has_input = args.contains_key("facts")
            || args.contains_key("fact")
            || args.contains_key("old_fact");
        let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("add");
        if action == "add" && !has_input {
            return ToolExecutionResult {
                ok: false,
                data: "Error: nothing to remember.".to_string(),
                error: None,
                error_kind: Some(crate::errors::ToolErrorKind::MissingArg {
                    param: "facts".to_string(),
                    example: r#"remember({"facts":["a concise fact"]})"#.to_string(),
                }),
            };
        }
        ToolExecutionResult::from_output(self.execute(args).await)
    }
```

In `execute()`: reject `list` early (`return "Error: remember has no 'list' action. Read memory with recall({\"query\":\"...\",\"scope\":\"memory\"}).".to_string()`); route `add` to a `do_add` that collects `facts` (array) + optional legacy single `fact`, enforces ≤20 facts and ≤`MAX_FACT_CHARS` chars each, then calls `self.store.append_facts_batch(facts).await`. `replace`/`delete`/`dedupe` keep their current single-fact paths (unchanged).

- [ ] **Step 6: Run tests, verify pass.** `cargo test --lib remember::` → PASS. `cargo test --lib` → green.

- [ ] **Step 7: Commit.**
```bash
git add src/agent/memory.rs src/agent/tools/remember.rs
git commit -m "feat(memory): batch remember + structural empty-arg error

remember accepts facts:[...] (≤20, batch atomic write); list action removed
(reads -> recall scope=memory); empty-arg add now returns ToolErrorKind::MissingArg
with a corrective example instead of silently succeeding as a list (the loop root cause)."
```

- [ ] **Step 8: e2e.** `cargo build --release && ./target/release/nanobot agent -l -s cli:e2e-t2 -m "please remember this moment as AGI bonsai, and that I prefer Rust, and that today is Friday"`. Audit `cli_e2e-t2.jsonl` must show a SINGLE `remember({"facts":[...]})` call with ≥2 facts (batch), `result_ok:true`, and NO empty `remember({})` loop. This is the structural-path live proof (Task 1's mechanism exercised by a real tool).

---

## Task 3: Dissolve into unified `recall` (+ alias shim)

This is the largest task. Sub-steps each carry their own test cycle. The alias shim (3a) and rename (3b) ship first so old session `tool_calls` keep resolving; then `recall` absorbs the search and fetch modes (3c–3e); then the old files retire (3f).

**Files:**
- Modify: `src/agent/tools/registry.rs` (`normalize_tool_request` aliases, `RARELY_ADVERTISED_TOOLS`).
- Modify: `src/agent/tools/recall_tool_result.rs`, `src/agent/tools/stash_search.rs` (`name()`).
- Modify: `src/agent/tools/recall.rs` (scope, trust-ranking, fetch modes; absorb session_search/search_context logic).
- Delete: `src/agent/tools/session_search.rs`, `src/agent/tools/search_context.rs`.
- Test: `recall.rs` unit tests; live e2e.

**Interfaces:**
- Consumes: Task 1 `MissingArg`.
- Produces: `recall({query, scope?, n?, session?, message_ids?, mode?})`; aliases `session_search`/`search_context`/`recall_tool_result`/`search_tool_result`/`slice_tool_result` → new targets via `normalize_tool_request`.

### 3a — Alias shim

- [ ] **Step 1: Extend `normalize_tool_request`** (registry.rs:117-139) name-alias match to route the old names to the new ones:

```rust
        let canonical_name = match name {
            "wait" | "check" | "list" | "cancel" => "spawn",
            // Recovery-tool renames (old session tool_calls keep resolving).
            "recall_tool_result" => "fetch_tool_output",
            "search_tool_result" => "grep_tool_output",
            "slice_tool_result" => "slice_tool_output",
            // Dissolution: session_search and search_context collapse into recall.
            // Routing by param happens in recall's dispatch (query -> search;
            // session/message_ids -> fetch). Plain name rewrite here.
            "session_search" | "search_context" => "recall",
            other => other,
        };
```

(For `session_search`'s `mode:"latest"`/`session`/`in_session`/`extract`, recall's dispatch in 3e reads the same param keys, so a plain name rewrite is sufficient — no param rewrite needed.)

- [ ] **Step 2: Test alias resolution.** In registry.rs tests:

```rust
    #[test]
    fn normalize_routes_old_tool_names_to_new_targets() {
        for (old, new) in [
            ("recall_tool_result", "fetch_tool_output"),
            ("search_tool_result", "grep_tool_output"),
            ("slice_tool_result", "slice_tool_output"),
            ("session_search", "recall"),
            ("search_context", "recall"),
        ] {
            let (c, _) = ToolRegistry::normalize_tool_request(old, HashMap::new()).unwrap();
            assert_eq!(c, new, "alias {old} -> {new}");
        }
    }
```

`cargo test --lib normalize_routes_old_tool_names_to_new_targets` → PASS once 3a lands.

### 3b — Rename the recovery tools

- [ ] **Step 3:** In `recall_tool_result.rs`, change `fn name(&self) -> &str { "fetch_tool_output" }`. In `stash_search.rs`, change the two tool impls: the grep one → `"grep_tool_output"`, the slice one → `"slice_tool_output"`.
- [ ] **Step 4:** In `registry.rs`, update `RARELY_ADVERTISED_TOOLS` (lines 746-757): replace `"recall_tool_result"`, `"search_tool_result"`, `"slice_tool_result"` with their new names; remove `"search_context"` (dissolved). Keep `session_search` OUT entirely (dissolved).
- [ ] **Step 5:** `cargo build` (catch any leftover references; the old `register_standard_tools` calls and `should_include(...)` gates may need the new names). `cargo test --lib` → green. Commit:

```bash
git add src/agent/tools/registry.rs src/agent/tools/recall_tool_result.rs src/agent/tools/stash_search.rs
git commit -m "refactor(tools): rename recovery trio, alias old names

recall_tool_result->fetch_tool_output, search_tool_result->grep_tool_output,
slice_tool_result->slice_tool_output. Old names kept as hidden aliases via
normalize_tool_request so stored session tool_calls keep resolving. Kills
the recall/recall_tool_result naming collision."
```

### 3c–3e — `recall` absorbs search + fetch modes

- [ ] **Step 6 (scope + sessions query):** Extend `recall`'s schema with `scope` (`memory|files|sessions|all`, default `all`), `n`, `session`, `message_ids`, `mode`. In `execute()`, dispatch by param presence:
  - `mode == "latest"` OR `session` present OR `message_ids` present → fetch path (migrate the logic from `session_search.rs`: `latest` lists recent sessions; `session` dumps one; `message_ids` extracts). Reuse `SessionDb::search_conversation_messages` and friends (the existing calls inside session_search.rs).
  - `query` present → search path: query each store selected by `scope`, collect top-N per source.
  - none of the above → return `MissingArg { param: "query", example: r#"recall({"query":"..."})"# }` via `execute_with_result` override (enumerating the three entry params in the data string).

- [ ] **Step 7 (trust-ranking merge):** When `scope=all`, merge results in trust order with section headers, per-source cap ~3, total cap ~10, 8000-char output cap (existing). Preserve the recall.rs:227-235 invariant: curated memory first; knowledge/sessions never drown it.

```rust
    // Trust order (the dissolve safety net — preserves the guardrail's intent):
    //   curated memory (MEMORY.md) > knowledge docs > workspace files > raw sessions
    fn merge_trust_ranked(&self, query: &str, n: usize) -> Vec<String> {
        let mut sections = Vec::new();
        if let Some(m) = self.grep_memory(query, n).await.ok_truncated() { sections.push(("Curated memory", m)); }
        if let Some(k) = self.knowledge_search(query, n, mode) { sections.push(("Knowledge docs", k)); }
        if let Some(f) = self.files_search(query, n)    { sections.push(("Workspace files", f)); }
        if let Some(s) = self.sessions_search(query, n) { sections.push(("Past conversations", s)); }
        sections.into_iter().map(|(h, b)| format!("## {h}\n{b}")).collect()
    }
```

(`grep_memory`/`knowledge_search` already exist on `RecallTool`; `files_search` wraps the `SearchFilesTool` call from `search_context.rs`; `sessions_search` wraps `SessionDb::search_conversation_messages` from `session_search.rs`. "ok_truncated" = take non-empty results, each pre-capped.)

- [ ] **Step 8 (tests):**
  - `recall({})` → `MissingArg`, data names the three entry params.
  - `recall({"query":"x","scope":"memory"})` queries MEMORY.md + knowledge only (no sessions).
  - `recall({"query":"x"})` (scope=all) → trust-ranked: seed MEMORY.md with a matching fact and a session row matching the same term; assert the "## Curated memory" section precedes "## Past conversations" and the canonical fact is present.
  - `recall({"session":"<key>"})` dumps that session; `recall({"message_ids":"1-3"})` extracts.

- [ ] **Step 9:** `cargo test --lib recall::` → PASS; `cargo test --lib` → green.

### 3f — Retire the old files

- [ ] **Step 10:** Delete `src/agent/tools/session_search.rs` and `src/agent/tools/search_context.rs`; remove their `register(...)` calls + `mod`/`use` lines. Their logic now lives in `recall`. The aliases in 3a keep old names dispatching to `recall`. `cargo build` must be clean (rustc proves zero remaining references).

- [ ] **Step 11: Commit.**
```bash
git add -A src/agent/tools/
git commit -m "refactor(memory): dissolve session_search + search_context into recall

recall now serves all retrieval: query-search across memory/knowledge/files/
sessions (trust-ranked so canonical facts outrank stale transcripts) OR
fetch by session/message_ids. session_search and search_context deleted;
old names alias to recall via normalize_tool_request for replay back-compat."
```

- [ ] **Step 12: e2e (two checks).**
  - **Trust-ranking:** seed a MEMORY.md fact and a session on the same topic; `./target/release/nanobot agent -l -s cli:e2e-t3a -m "what do you know about <topic>?"`; audit shows a `recall` call and the reply cites the canonical fact.
  - **Alias:** replay an old session that contains a `recall_tool_result` tool_call through the new build; it must still resolve (no "Tool not found"). (`./target/release/nanobot agent -l -s cli:e2e-t3b -m "show me the truncated output you fetched earlier"` — if the model emits the old name, the alias resolves.)

---

## Task 4: FTS tokenizer unification + worked-shapes (both identity paths)

**Files:**
- Modify: `src/agent/knowledge_store.rs:106-110` (FTS create), open path (rebuild-on-mismatch).
- Modify: `src/agent/context.rs:1272-1288` (local) and `:1322-1332` (non-local Memory section).
- Test: knowledge_store unit test (tokenizer parity); context test (worked-shape presence); live e2e.

**Interfaces:** none new.

- [ ] **Step 1: Align the knowledge FTS tokenizer.** In `knowledge_store.rs:106-110`, change the `chunks_fts` CREATE to `tokenize='porter unicode61'` to match `messages_fts`. Because `chunks_fts` is external-content (`content='chunks'`), the rebuild loses no data.

- [ ] **Step 2: Rebuild-on-mismatch at open.** After opening/ensuring the schema, detect if the existing `chunks_fts` used the old tokenizer (query `SELECT sql FROM sqlite_master WHERE name='chunks_fts'`; if the SQL lacks `porter`, rebuild): drop `chunks_fts`, recreate with `porter unicode61`, then `INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')` (FTS5 external-content rebuild idiom). Idempotent — no-op when already correct.

- [ ] **Step 3: Test tokenizer parity.** In `knowledge_store.rs` tests: ingest a chunk containing "running"; assert `store.search("ran", 1)` finds it (Porter stemming). Before the change this fails (bare unicode61 doesn't stem).

- [ ] **Step 4: Worked-shapes in BOTH identity paths.** In `context.rs`:
  - Non-local Memory section (~1322-1332): add a worked shape for every memory/search tool — `remember`, `recall`, `fetch_tool_output`, `lcm_expand` — alongside the existing `recall`/`session_search` examples (drop the `session_search` example, replaced by `recall` with `scope`).
  - Local identity (~1272-1288): the lean inline tool list (`recall; session_search.`) currently has NO shapes — add `remember({"facts":[...]})`, `recall({"query":"...","scope":"all"})`, and note `fetch_tool_output`/`grep_tool_output`/`slice_tool_output` for truncated-output recovery.

- [ ] **Step 5: Test worked-shape presence.** In `context.rs` tests: build the system prompt for both local and non-local identities; assert each rendered Memory block contains a `remember({...})` and a `recall({...})` call-shape substring.

- [ ] **Step 6:** `cargo test --lib` → green.

- [ ] **Step 7: Commit.**
```bash
git add src/agent/knowledge_store.rs src/agent/context.rs
git commit -m "feat(memory): unify FTS tokenizers + worked call-shapes

knowledge.db chunks_fts -> porter unicode61 to match sessions FTS (rebuild
on open, no data loss). System prompt Memory section (both local and
non-local identity paths) now shows a worked call-shape for every memory
tool — proactive coverage so weak models emit the right shape first try."
```

- [ ] **Step 8: e2e.** Delete `~/.nanobot/knowledge.db`, run `./target/release/nanobot ingest <some file containing 'running'>`, then `./target/release/nanobot agent -l -s cli:e2e-t4 -m "use recall to find anything about ran"`; audit shows `recall({"query":"ran",...})` and the result surfaces the "running" doc (stemming parity proven live).

---

## Self-review notes (plan ↔ spec)

- **Spec coverage:** §3.1 surface → Tasks 3a/3b/3f; §3.2 recall contract → 3c-3e; §3.3 remember → Task 2; §3.4 MissingArg → Task 1; §3.5 FTS unification → Task 4; §3.6 worked-shapes → Task 4; §3.7 alias shim → 3a. §4 verification table rows → e2e steps (1→T1, 2/5→T2, 3/4→T3, 6→T4). All covered.
- **Refinement vs spec §3.4:** uses the existing `error_kind` slot instead of changing `execute()`'s signature — same structural contract, far smaller blast radius (spec already updated in commit b8ed9d7).
- **Type consistency:** `MissingArg { param: String, example: String }` used identically in Task 1 (enum), Task 2 (remember override), Task 3 (recall override). New tool names (`fetch_tool_output`/`grep_tool_output`/`slice_tool_output`) identical in 3a alias map, 3b rename, 3b catalog list, Task 4 worked-shapes.
- **Risk:** Task 3 is the largest; 3a/3b ship the back-compat aliases first so old sessions keep replaying before any file is deleted (3f). Each sub-step compiles + tests green before the next.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-31-memory-session-unification.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks (each ends with its live e2e gate), fast iteration.
2. **Inline Execution** — execute tasks in this session via executing-plans, batch with checkpoints.

Which approach?
