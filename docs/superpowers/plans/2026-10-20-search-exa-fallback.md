# Reliable local-first web search implementation plan

> Implementation: gpt-6-luna xhigh through the existing implementation agent; parent reviews and integrates. User authorized fixing search, committing, merging locally, and installing the latest binary. No extra approval gate is needed.

**Goal:** Make `web_search` succeed when the local metasearch server is healthy but its upstream engines are blocked, then deliver this and the previously reviewed tool improvements.

**Architecture:** Retain SearXNG as primary. Its existing executor returns an error for degraded health, transport/HTTP failure, malformed JSON/schema, or blocked empty results. One decision in `Tool::execute` chooses Exa when configured, otherwise the existing Brave fallback. A genuine empty response does not incur an external API call. Use existing reqwest and config plumbing; no new dependency, registry, persistent retry state, or provider framework.

## Evidence and constraints

- Worktree `/private/tmp/nanobot-tool-contract-20261020`, branch `codex/tool-contract-first-read`, base `e4db535`. Preserve the six reviewed source edits and prior plan.
- Live probe: local `http://localhost:8080/healthz` HTTP200; `/search` HTTP200 with zero results and upstream failures: Brave rate limit, DuckDuckGo CAPTCHA, Startpage CAPTCHA. The existing blocked-empty and degraded-health paths bypass fallback. JSON parse errors also lack the `Error:` prefix used by the shared boundary.
- The exported key exists in `~/.zshrc` but is absent from the current app process environment. A direct Exa POST returned HTTP200, three official Rust URLs and highlights in1.291s; reported cost$0.007. Never print, commit, or place the key in a command argument.
- Official endpoint contract: https://exa.ai/docs/reference/search — POST `https://api.exa.ai/search`, `x-api-key`, JSON `query`, `numResults`, `type:auto`, `contents.highlights:true`.
- Release builds/tests only. Reuse `/Users/peppi/Dev/nanobot-rs/target` with sandbox escalation. No CLI agent invocation: its singleton guard terminates an existing agent.
- GitNexus impacts before edits: `core_config_from` CRITICAL18impacts/7flows; `build_swappable_core` CRITICAL99/21; `with_standard_tools` HIGH22/4; `ToolConfig` HIGH12; `WebSearchTool::new` HIGH39. Parent warned user. `execute` is MEDIUM with unresolved trait receivers; config and chained builder UNKNOWN were confirmed by source search, not assumed unused.
- Keep `apiKey` as the existing Brave credential. Add `exaApiKey`, default empty, redact Debug; nonempty explicit config overrides `EXA_API_KEY`. Thread through normal builds, model switches and both subagent execution paths.
- Clamp result count to1..10. Exa results retain titles, URLs and bounded highlights/snippets. Surface sanitized HTTP/auth/quota/network/malformed errors. Do not expose request headers or raw upstream error bodies; no Exa-to-Brave cascade.
- Keep the user's active session/model and services running. Only search credential config and the installed binary may change, as now authorized. Preserve original-checkout untracked files.

## Task 1 — Luna implements and proves the fix

**Ownership:** `src/agent/tools/web.rs`, `src/config/schema.rs`, `src/cli/core_builder.rs`, `src/agent/agent_core.rs`, `src/agent/agent_loop/mod.rs`, `src/agent/subagent.rs`, `src/agent/tool_wiring.rs`, `src/agent/tools/registry.rs`; mechanical input fixtures in `src/agent/agent_loop/tests.rs`, `response.rs`, `recovery_eval.rs`, `src/agent/router.rs` as required. No unrelated formatting cleanup.

- [x] Add focused failing regressions for local-first fallback across blocked engines/health/transport/HTTP/parse/schema failures, genuine empty/no paid fallback, Exa request/result/error behavior, config precedence/redaction and root/subagent plumbing. Use an existing or minimal local HTTP fixture; no dependency. Capture meaningful RED before production changes.
- [x] Keep `WebSearchTool::new` compatibility where practical; add a narrow Exa builder/config input. Use the single existing execution boundary for fallback; convert all SearXNG errors to its existing typed error representation. For valid `results:[]` and no engine failures, return `No results` successfully.
- [x] Add Exa POST implementation with timeout, capped count and highlights. A selected fallback must be named in successful output. Failed fallback must remain a typed error, not a success saying it fell back.
- [x] Carry the credential through the existing config/core/tool/subagent fields. Add only mechanical fixture fields beyond this path. Empty config allows env fallback; tests should avoid global env races.
- [x] Add one ignored tool-level live search smoke that exercises local SearXNG then Exa with an inherited key, without launching an agent or opening a session DB. Parent supplies the key privately.
- [x] Run focused release tests, targeted rustfmt and diff-check. Provide source-stable checkpoint, changed functions, and RED/GREEN logs. Parent reviews; all code fixes return to Luna.

## Task 2 — Parent verifies, commits, merges and installs

- [x] Review the source diff and evidence. Run live search through the Rust tool itself, verifying usable URLs/snippets and explicit fallback. This is tool reliability evidence, not model latency/cache evidence.
- [x] Back up `~/.nanobot/config.json` with private permissions; surgically add only `tools.web.search.exaApiKey` from the literal `.zshrc` assignment. Preserve all other fields and existing provider/searxngUrl. Verify with redacted output only.
- [x] Run `cargo build --release`, `cargo test --release`, targeted rustfmt, full fmt comparison against the existing21-hunk baseline, and `git diff --check`.
- [x] Refresh the worktree GitNexus index with `--index-only --max-file-size 2048`; run machine-readable `detect_changes(scope=all)` with no partial/truncated flag before staging/commit. Preserve the analyzer's separate bounded-flow caveat.
- [ ] Commit all reviewed tool/read/search changes with explicit paths. Verify staged diff for accidental credentials and unrelated files. Merge into the original checkout's `main` with `--ff-only`, preserving its four unrelated untracked items. No push requested.
- [ ] Re-run release tests and release build on merged main. Atomically copy the resulting binary to `~/.local/bin/nanobot`, keeping a rollback copy of the previous executable. Verify source/installed SHA256 match and run only safe `--version`, without restarting any active agent.
- [ ] Record commit, merged tree, test/live-search receipts, config/binary backup paths, hash verification, and remaining limitations. The running process uses its existing image until the user next restarts it; do not force a restart.

## Validation receipts before integration

- Meaningful RED: malformed SearXNG JSON was incorrectly successful (`/tmp/nanobot-exa-red.log`); missing results and discarded `exaApiKey` also reproduced before the fix.
- Focused final release GREEN: 24 passed, 0 failed, 1 ignored (`/tmp/nanobot-exa-green-compression.log`). Config precedence, root registry and subagent availability have separate passing release receipts.
- Live Rust tool smoke: 1 passed, 0 failed in 0.92s; local SearXNG's blocked-empty response selected Exa, which returned five usable results (`/tmp/nanobot-exa-live-tool.log`). This validates the tool/backend path, not end-to-end model behavior or prompt-cache reuse.
- The first Rust live probe exposed a real compression mismatch: browser defaults advertised gzip while this reqwest build has no decompressor. Exa and Brave now explicitly request JSON with identity encoding; local SearXNG headers remain unchanged. No dependency added.
- Full rustfmt check has exactly the same 21 pre-existing hunks as baseline; contents match after normalizing checkout paths and shifted line numbers. `git diff --check` passes. Unrelated recursive formatter changes were removed before the final source freeze.
- Config backup: `/Users/peppi/.nanobot/config.before-exa-20261020T170744Z.json`; config and backup mode 0600. Only `tools.web.search.exaApiKey` changed. The provider remains SearXNG.
- The CLI agent benchmark is intentionally deferred because its singleton guard would terminate the user's running agent. No matched model-speed or cache-reuse claim is made.

- Parent final branch verification: `cargo build --release` PASS (`/tmp/nanobot-exa-final-build.log`); `cargo test --release` PASS, 2675 passed / 0 failed / 22 ignored across all targets (`/tmp/nanobot-exa-final-tests.log`).
- Fresh branch GitNexus index matches all 17 changed Rust SHA256 hashes. Machine-readable change analysis reports 156 changed symbols / 178 affected flows / critical risk, `partial:false`, `truncated:false` (`/tmp/nanobot-exa-final-graph.json`). Critical scope reflects the previously warned shared config/tool constructors. Analyzer flow discovery remains bounded, so absent paths are not treated as unused.
- Exact private key-value scan of all 19 intended source/plan files passes. No credential is included. Integration/install results will be recorded in the ignored ledger and deployment receipt after this commit.
