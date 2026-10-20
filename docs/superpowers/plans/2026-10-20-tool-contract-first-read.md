# Tool contract and first-read results implementation plan

> **For agentic workers:** Use subagent-driven-development to execute tasks in order. Implementation model: **gpt-6-luna, xhigh**. The parent agent reviews specification compliance and code quality after each task and the complete diff at the end.

**Goal:** Remove conflicting tool instructions and make ordinary file/web reads useful immediately, while keeping exact evidence, bounded context, and replay stability.

**Architecture:** Keep the existing tool registry, SQLite artifacts, inspector, and live/replay ingestion chokepoint. Supplied schemas are the authority for invocation. File pages fit the existing inline byte ceiling; new web results retain provenance/completeness and expose a bounded first page with an artifact continuation. Historical artifacts retain their existing rendering rules.

**Tech stack:** Rust 2021, Tokio, serde_json, existing SQLite SessionDb and reqwest. No new dependencies or schema migrations.

## Global constraints

- Worktree: `/private/tmp/nanobot-tool-contract-20261020`, branch `codex/tool-contract-first-read`, base `e4db535b256c1ef435d154ddfbb62e28b2af31d8`.
- User-approved order: instruction cleanup (previous discussion item 3), then first-read results (item 2). Search/provider changes are excluded.
- Subsequent search work should prefer self-hosted search, with an external API only as fallback, per the user's preference.
- Follow repository AGENTS.md. Release builds/tests only. Do not install a binary, restart Higgs/Nanobot, reset sessions, or alter model/configuration.
- Original checkout contains unrelated untracked files; leave them alone. Do not merge or push automatically.
- Before editing a function, run GitNexus upstream impact against `/Users/peppi/Dev/nanobot-rs`; its indexed base matches this worktree. For new/unresolved symbols confirm with source search. Parent has already warned user about HIGH/CRITICAL impact.
- Existing impact: `_get_identity` LOW (three prompt-building callers); `render_range` CRITICAL (read execute/range paths, seven processes); `extract_web_content` HIGH (injection); `store_then_render_tool_result` HIGH (live injection and get_history); `get_proxy_definition_excluding` HIGH (schema generation).
- Preserve legacy proxy parsing aliases for existing callers; stop teaching aliases as competing current protocols.
- Ordinary inline ceiling remains 4,096 bytes. Replay ceiling remains 12,000 bytes. Do not globally raise either to hide the mismatch.
- New output budgets count UTF-8 bytes including headers/footers, and cuts occur at UTF-8 boundaries. Cursor coordinates retain documented semantics.
- No generic new pipeline, policy flags, automatic inspection loop, whole-page archival, or extra model supervisor.
- Persist source truncation separately from displayed-page truncation. An absent legacy completeness flag means unknown. Never imply discarded text is retrievable from SQLite.
- Never rewrite committed messages, mutate old artifacts, or change a historical handle/excerpt into a new presentation solely based on tool name. Version new web evidence explicitly so new rendering is opt-in by stored format.
- Existing taint, immutable-store conflict handling, error status, protocol pairing and source digest remain intact.
- Shared release target may be reused at `/Users/peppi/Dev/nanobot-rs/target` after the baseline job finishes; use sequential builds and sandbox escalation where needed. Record command output in `/tmp`, not the live transcript DB.

## Task 1: One authoritative tool invocation contract

**Files:** `src/agent/context.rs`, `src/agent/tools/registry.rs` only if shared guidance is needed, `src/cli/mod.rs` for new-workspace guidance, associated in-file tests. Also a surgical cleanup of `/Users/peppi/.nanobot/workspace/AGENTS.md` after tests, with an exact backup in `/tmp/nanobot-agents-before-tool-contract.md`.

**Interface:** The existing `get_tools` schema exposes `tool_name` and `tool_args`; native schemas accept direct arguments. Keep runtime dispatch aliases compatible. Generated prompts should defer to supplied tool schemas rather than repeat a manually maintained native-tool list. The registry owns exact discovery/inspection/invocation examples.

- [x] Add a regression verifying identity no longer enumerates a hard-coded native catalog or teaches a second invocation syntax, while registry schema still exposes get_tools/tool_name/tool_args. Extend onboarding tests to verify its instructions defer to active schemas, and existing custom workspace files are not overwritten.

```rust
let (_tmp, cb) = make_context();
let identity = cb._get_identity(true);
assert!(identity.contains("schemas"));
assert!(!identity.contains("Native tools: read_file,"));
assert!(!identity.contains("Other tools via `tool`"));
```

- [x] Run the focused tests with `cargo test --release --lib test_local_identity` and `cargo test --release --lib test_create_workspace_templates`; capture the intended failing assertion before changing production code.
- [x] Remove duplicated catalogs and obsolete proxy examples. Keep concise error/evidence discipline and optional inspection guidance. Update descriptions that falsely say all large reads require an inspection. Do not remove safety/workflow/persona content.
- [x] Run focused tests and existing registry proxy tests. Back up live workspace AGENTS exactly, replace only its obsolete Tools invocation/catalog paragraphs with schema-authoritative guidance, and preserve all other sections verbatim. No bulk migration of users' files and no runtime regex stripping of custom instructions.
- [x] Record tests, precise live-file diff/backup path, and changed files in the task report. Parent reviews before Task 2.

## Task 2: Direct file pages fit the inline contract

**Files:** `src/agent/tools/filesystem/mod.rs`; `src/agent/context_hygiene.rs` only if sharing an existing ceiling requires visibility changes; regression in `src/agent/tool_engine.rs` or session DB tests to exercise real ingestion/reload.

**Interface:** Preserve `ReadFileTool::new(usize)`, `path`, `lines`, `max_lines`, numbered contiguous lines, digest, and exact next line-range semantics. Constructor/render budget uses the existing inline ceiling instead of the replay ceiling. Explicit range reads remain bounded pages. An exceptional single oversized line must remain lossless using the existing handle/inspector path, with truthful guidance; do not silently cut it or advertise skipped source as consumed.

- [x] Add failing cases for a normal ASCII file larger than 6 KB, Unicode-heavy lines, long source paths, and an oversized single line. For ordinary lines, assert real returned envelope fits 4,096 bytes, contains useful numbered source and exact continuation, and stays inline through store_then_render_tool_result.

```rust
let body = (0..400).map(|i| format!("item {i}: useful evidence\n")).collect::<String>();
let hash = sha256_hex(body.as_bytes());
let page = render_range(&body, 1, 400, "source.txt", 400, 4_096, &hash);
assert!(page.len() <= crate::agent::context_hygiene::INLINE_TOOL_RESULT_MAX_BYTES);
assert!(page.contains("item 0: useful evidence"));
assert!(page.contains("next: read_file lines="));
assert!(!page.contains("[truncated:"));
```

- [x] Run `cargo test --release --lib read_file` and the new renderer/ingestion tests to observe the current mismatch.
- [x] Clamp the production read budget to the inline ceiling with explicit wrapper headroom. Update stale 7,000-character/replay-safe descriptions and comments. Reuse the current contiguous-line renderer and existing overflow recovery.
- [x] Verify paging reaches a required fact after the first page without gaps/overlap; Unicode reconstruction stays exact; empty/binary/error behavior remains truthful. Verify first live result equals reloaded history bytes, including provenance wrapper when enabled.
- [x] Re-run focused renderer, read-file, ingestion and history tests. Record exceptions honestly and parent-review this task before Task 3.

## Task 3: Web provenance and useful bounded first page

**Files:** `src/agent/tools/web.rs`, `src/agent/tool_engine.rs`, `src/agent/tools/stash_search.rs` only for sharing the current paging helper, `src/session/db.rs` tests/replay integration only as necessary. No web_search behavior change.

**Interfaces:** Existing web_fetch arguments and fetch size limits remain. New fetched evidence has a versioned stored representation containing requested URL, final URL, HTTP status, extractor, and source-truncated true/false/unknown. A clean textual envelope is sufficient; use existing serde support if structured parsing is clearer. Extraction paths include plain HTTP and configured CRW. `web_search` and unrelated JSON must retain existing handling.

- [x] Add a failing extraction test showing URL/status/truncated survive ingestion, plus a large fetched-text fixture whose first model-visible result contains real article text and a valid original artifact id/cursor.

```rust
let raw = serde_json::json!({
    "url":"https://example.test/start", "finalUrl":"https://example.test/article",
    "status":200, "extractor":"readability", "truncated":true,
    "text":"First paragraph.\nRequired fact later."
}).to_string();
let evidence = extract_web_fetch_evidence(&raw);
assert!(evidence.contains("https://example.test/article"));
assert!(evidence.contains("200"));
assert!(evidence.contains("truncated"));
assert!(evidence.contains("First paragraph."));
```

- [x] Test full small result, large stored body, source-truncated body, unknown legacy metadata, CRW result, empty/error/malformed input, and multibyte text. Observe failures before production edits.
- [x] Preserve metadata in new versioned web evidence before storage. Teach the shared live/reload chokepoint to render only this new format as a useful contiguous first page with original artifact identity and explicit continuation. Small results remain complete inline. Prefer reuse of inspector paging; ensure new first-page path enforces a byte budget rather than assuming inspector's character cap is byte-safe.
- [x] Inspector EOF says end of retained result, not complete page. Continuation walks retained evidence exactly; when source extraction already truncated, metadata says refetch with a larger cap is required for omitted text. Keep current extraction limits and do not store whole pages beyond them.
- [x] Add integration tests: ingest -> inspect continuation -> close/reopen DB -> get_history is byte-identical; old ordinary result handles and old explicit excerpts retain their old bytes. Inspect a required fact after the first page and after a persisted/reloaded context boundary. Check tool-call/result pairing remains valid.
- [x] Run release focused tests: `cargo test --release --lib web_fetch`, `cargo test --release --lib inspect`, `cargo test --release --lib get_history`, then protocol and LCM integration targets. Parent reviews all changes.

## Final validation and review

- [x] Run `cargo fmt --all -- --check`, `cargo build --release`, and `cargo test --release`. Compare failures to the saved baseline; do not label pre-existing failures fixed or introduce unrelated repairs.
- [x] Existing `scripts/turn_bench.sh` is only a short-reply timing smoke test. For behavior evidence use isolated sessions and static file/web fixtures on the actual LFM/Higgs stack; compare answer correctness, source completeness claims, repeated source reads versus legitimate paging, inspector calls, model round trips, input/cache tokens, TTFT and elapsed time. Do not use the user's active conversation or change power/model settings. If a live dependency is unavailable, report the measurement as not run rather than infer a speedup from deterministic tests.
- [x] Parent performs specification and code-quality review; all implementation fixes return to Luna xhigh. No parent production edits.
- [x] Run GitNexus `detect-changes --scope all` on worktree before any commit. A partial/truncated result is unresolved; use the CLI's worktree-aware options and rerun with adequate bounds. No automatic commit until a clean graph-change receipt. If graph tooling cannot cover the worktree, report it and leave reviewable uncommitted changes.
- [x] Final handoff names worktree/branch, test evidence, live AGENTS backup/change, unresolved limits and whether live model behavior was measured. Do not merge, push, install, or restart services.

## Final review receipt — 2026-10-20

Implementation by gpt-6-luna at xhigh; parent reviewed each task and the whole diff. All requested fixes were returned to Luna. No unresolved code-review findings remain.

- `cargo build --release`: passed (45.14s); log `/tmp/nanobot-tool-contract-final-build.log`.
- `cargo test --release`: passed, 2,661 tests passed, 0 failed, 21 ignored. Library: 2,621 passed; LCM integration: 10; protocol invariants: 6; protocol integration: 24. Default feature set, no ignored/live-endpoint tests enabled. Log `/tmp/nanobot-tool-contract-final-tests.log`.
- All six touched Rust files pass targeted rustfmt. `git diff --check` passes. `cargo fmt --all -- --check` still reports exactly the same 21 pre-existing hunks as the original checkout, byte-identical after normalizing checkout paths. Baseline/final logs: `/tmp/nanobot-tool-contract-baseline-fmt.log`, `/tmp/nanobot-tool-contract-final-fmt.log`.
- GitNexus branch index refreshed at 16:39:53 UTC; hashes match all six reviewed source files. `detect_changes(scope=all)` reports 52 symbols, 25 affected flows, critical risk, with no partial/truncated flag. Receipt `/tmp/nanobot-tool-contract-final-graph.json`. The analyzer separately warns that process discovery is bounded and cross-language receiver inference is incomplete; absence from graph results does not prove a path is absent.
- Regression evidence: complete Unicode continuation reconstruction, late-fact inspection after DB reopen, identical first-page bytes through history reload, retained empty-fetch provenance, oversized-metadata fallback, and legacy handle/excerpt stability. This establishes deterministic behavior under these fixtures, not a measured model cache-hit ratio or latency improvement.
- Live-model benchmark not run: the CLI acquires a singleton and can terminate the active Nanobot process. No CLI agent run or `turn_bench` invocation was made. The live LFM session, model settings, installed binary and running services were preserved.
- The only live edit was the workspace `AGENTS.md` Tools guidance. Exact backup `/tmp/nanobot-agents-before-tool-contract.md`; final readback confirms all content outside Tools is unchanged.
- Changes remain uncommitted in this worktree on `codex/tool-contract-first-read`. No merge, push, installation, or restart. Original checkout retains only its four pre-existing untracked items. Temporary redundant build artifacts were removed; the shared release cache and this reviewable worktree remain.
