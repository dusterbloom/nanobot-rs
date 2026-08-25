# Error-protocol layer-3 backlog

Tracking artifact for the per-module `#![allow(...)]` blocks that carry the
`Error-protocol layer-3 backlog` marker comment. The deny regime in `Cargo.toml`
(`[lints.clippy]`) is live; each module below still carries pre-existing violations
of the listed lints. Remove the module's allow block as the module migrates onto the
regime (research doc: `docs/research/2026-08-06-error-conventions-and-host-bridge.md`
§3.6).

## Status

- [x] **Phase 3 milestone**: host bridge adopted — `SpawnTool`/`MessageTool`
  route through `HostBridge`/`MessageHost`; legacy callbacks, the
  `HostBridgeAdapter`, and the 8 `build_tools` closures are deleted (research
  doc §3.7 Steps 2-3); the `host_bridge.rs` dead-code allow is gone and
  `set_*_callback` is gated by `clippy.toml` + the quality sentinel.
- [ ] **Phase 3 milestone**: zero modules carry the `Error-protocol layer-3 backlog` allow
- [ ] **Phase 3 milestone**: `from_output` / `classify_tool_error` fully removed (see `scripts/quality-sentinel.sh`)

## Per-module backlog (78 modules)

Check a module off as its `#![allow(...)]` block is deleted and the module passes
`cargo clippy --all-targets` under the full deny regime.

| Module | Allowed lints (backlog) | Migrated |
|---|---|---|
| **`src/agent`** | | |
| [`src/agent/agent_core.rs`](src/agent/agent_core.rs) | `clippy::as_conversions`, `clippy::indexing_slicing`, `clippy::shadow_unrelated` | [ ] |
| [`src/agent/agent_loop/budget.rs`](src/agent/agent_loop/budget.rs) | `clippy::as_conversions`, `clippy::shadow_reuse`, `clippy::shadow_same` | [ ] |
| [`src/agent/agent_loop/compaction.rs`](src/agent/agent_loop/compaction.rs) | `clippy::indexing_slicing` | [ ] |
| [`src/agent/agent_loop/heuristics.rs`](src/agent/agent_loop/heuristics.rs) | `clippy::as_conversions`, `clippy::indexing_slicing`, `clippy::format_push_string`, `clippy::string_add` | [ ] |
| [`src/agent/agent_loop/local_stream.rs`](src/agent/agent_loop/local_stream.rs) | `clippy::as_conversions`, `clippy::shadow_reuse` | [ ] |
| [`src/agent/agent_loop/mod.rs`](src/agent/agent_loop/mod.rs) | `clippy::shadow_reuse` | [ ] |
| [`src/agent/agent_loop/response.rs`](src/agent/agent_loop/response.rs) | `clippy::as_conversions`, `clippy::shadow_reuse` | [ ] |
| [`src/agent/agent_loop/shared.rs`](src/agent/agent_loop/shared.rs) | `clippy::as_conversions`, `clippy::indexing_slicing`, `clippy::shadow_reuse`, `clippy::shadow_unrelated` | [ ] |
| [`src/agent/agent_profiles.rs`](src/agent/agent_profiles.rs) | `clippy::shadow_reuse` | [ ] |
| [`src/agent/anti_drift.rs`](src/agent/anti_drift.rs) | `clippy::as_conversions`, `clippy::indexing_slicing` | [ ] |
| [`src/agent/compaction.rs`](src/agent/compaction.rs) | `clippy::as_conversions`, `clippy::shadow_reuse`, `clippy::print_stderr` | [ ] |
| [`src/agent/context.rs`](src/agent/context.rs) | `clippy::as_conversions`, `clippy::indexing_slicing`, `clippy::shadow_reuse`, `clippy::format_push_string`, `clippy::string_add` | [ ] |
| [`src/agent/context_gate.rs`](src/agent/context_gate.rs) | `clippy::as_conversions`, `clippy::indexing_slicing` | [ ] |
| [`src/agent/context_hygiene.rs`](src/agent/context_hygiene.rs) | `clippy::indexing_slicing` | [ ] |
| [`src/agent/context_store.rs`](src/agent/context_store.rs) | `clippy::as_conversions` | [ ] |
| [`src/agent/embedder.rs`](src/agent/embedder.rs) | `clippy::indexing_slicing` | [ ] |
| [`src/agent/finalize_response.rs`](src/agent/finalize_response.rs) | `clippy::as_conversions`, `clippy::indexing_slicing` | [ ] |
| [`src/agent/gateway_commands.rs`](src/agent/gateway_commands.rs) | `clippy::as_conversions` | [ ] |
| [`src/agent/instructions.rs`](src/agent/instructions.rs) | `clippy::indexing_slicing`, `clippy::shadow_reuse` | [ ] |
| [`src/agent/knowledge_store.rs`](src/agent/knowledge_store.rs) | `clippy::as_conversions`, `clippy::indexing_slicing` | [ ] |
| [`src/agent/lcm.rs`](src/agent/lcm.rs) | `clippy::as_conversions`, `clippy::indexing_slicing`, `clippy::shadow_reuse`, `clippy::format_push_string` | [ ] |
| [`src/agent/memory_ladder.rs`](src/agent/memory_ladder.rs) | `clippy::shadow_reuse` | [ ] |
| [`src/agent/model_capabilities.rs`](src/agent/model_capabilities.rs) | `clippy::as_conversions`, `clippy::indexing_slicing` | [ ] |
| [`src/agent/model_prices.rs`](src/agent/model_prices.rs) | `clippy::as_conversions` | [ ] |
| [`src/agent/pid_file.rs`](src/agent/pid_file.rs) | `clippy::as_conversions` | [ ] |
| [`src/agent/pipeline.rs`](src/agent/pipeline.rs) | `clippy::as_conversions`, `clippy::indexing_slicing`, `clippy::format_push_string` | [ ] |
| [`src/agent/prepare_context.rs`](src/agent/prepare_context.rs) | `clippy::as_conversions`, `clippy::shadow_reuse`, `clippy::shadow_unrelated` | [ ] |
| [`src/agent/prompt_contract.rs`](src/agent/prompt_contract.rs) | `clippy::as_conversions`, `clippy::indexing_slicing` | [ ] |
| [`src/agent/prompt_fingerprint.rs`](src/agent/prompt_fingerprint.rs) | `clippy::shadow_reuse` | [ ] |
| [`src/agent/protocol.rs`](src/agent/protocol.rs) | `clippy::indexing_slicing`, `clippy::shadow_reuse`, `clippy::format_push_string`, `clippy::string_add` | [ ] |
| [`src/agent/reasoning.rs`](src/agent/reasoning.rs) | `clippy::shadow_reuse` | [ ] |
| [`src/agent/reflector.rs`](src/agent/reflector.rs) | `clippy::indexing_slicing`, `clippy::shadow_reuse` | [ ] |
| [`src/agent/router.rs`](src/agent/router.rs) | `clippy::as_conversions`, `clippy::indexing_slicing`, `clippy::shadow_reuse`, `clippy::format_push_string` | [ ] |
| [`src/agent/runtime_mode.rs`](src/agent/runtime_mode.rs) | `clippy::shadow_reuse` | [ ] |
| [`src/agent/sanitize.rs`](src/agent/sanitize.rs) | `clippy::shadow_reuse` | [ ] |
| [`src/agent/skills.rs`](src/agent/skills.rs) | `clippy::format_push_string` | [ ] |
| [`src/agent/subagent.rs`](src/agent/subagent.rs) | `clippy::as_conversions`, `clippy::indexing_slicing`, `clippy::shadow_reuse`, `clippy::format_push_string` | [ ] |
| [`src/agent/system_state.rs`](src/agent/system_state.rs) | `clippy::as_conversions` | [ ] |
| [`src/agent/token_budget.rs`](src/agent/token_budget.rs) | `clippy::as_conversions`, `clippy::indexing_slicing` | [ ] |
| [`src/agent/tool_engine.rs`](src/agent/tool_engine.rs) | `clippy::as_conversions`, `clippy::indexing_slicing`, `clippy::shadow_reuse`, `clippy::format_push_string` | [ ] |
| [`src/agent/tool_runner/mod.rs`](src/agent/tool_runner/mod.rs) | `clippy::as_conversions` | [ ] |
| [`src/agent/tool_wiring.rs`](src/agent/tool_wiring.rs) | `clippy::shadow_reuse`, `clippy::shadow_unrelated` | [~] — partially cleared in Phase 3: `as_conversions` (→ `u32::try_from`) and `format_push_string` (→ `writeln!`) removed; `shadow_*` remain |
| [`src/agent/tools/apply_patch.rs`](src/agent/tools/apply_patch.rs) | `clippy::indexing_slicing`, `clippy::shadow_reuse` | [ ] |
| [`src/agent/tools/browser.rs`](src/agent/tools/browser.rs) | `clippy::shadow_reuse` | [ ] |
| [`src/agent/tools/cron_tool.rs`](src/agent/tools/cron_tool.rs) | `clippy::shadow_reuse` | [ ] |
| [`src/agent/tools/file_preview.rs`](src/agent/tools/file_preview.rs) | `clippy::as_conversions`, `clippy::format_push_string` | [ ] |
| [`src/agent/tools/filesystem/mod.rs`](src/agent/tools/filesystem/mod.rs) | `clippy::as_conversions`, `clippy::indexing_slicing`, `clippy::shadow_reuse`, `clippy::shadow_unrelated`, `clippy::format_push_string` | [ ] |
| [`src/agent/tools/filesystem/write.rs`](src/agent/tools/filesystem/write.rs) | `clippy::as_conversions` | [ ] |
| [`src/agent/tools/reasoning_tools.rs`](src/agent/tools/reasoning_tools.rs) | `clippy::as_conversions`, `clippy::indexing_slicing` | [ ] |
| [`src/agent/tools/recall.rs`](src/agent/tools/recall.rs) | `clippy::as_conversions`, `clippy::shadow_reuse`, `clippy::format_push_string` | [ ] |
| [`src/agent/tools/registry.rs`](src/agent/tools/registry.rs) | `clippy::indexing_slicing`, `clippy::shadow_reuse` | [ ] |
| [`src/agent/tools/remember.rs`](src/agent/tools/remember.rs) | `clippy::indexing_slicing`, `clippy::shadow_reuse` | [ ] |
| [`src/agent/tools/shell.rs`](src/agent/tools/shell.rs) | `clippy::as_conversions`, `clippy::shadow_reuse` | [ ] |
| [`src/agent/tools/stash_search.rs`](src/agent/tools/stash_search.rs) | `clippy::as_conversions`, `clippy::shadow_reuse`, `clippy::format_push_string` | [ ] |
| [`src/agent/tools/system_info.rs`](src/agent/tools/system_info.rs) | `clippy::as_conversions` | [ ] |
| [`src/agent/tools/todo.rs`](src/agent/tools/todo.rs) | `clippy::as_conversions`, `clippy::shadow_reuse` | [ ] |
| [`src/agent/tools/tool_status.rs`](src/agent/tools/tool_status.rs) | `clippy::as_conversions`, `clippy::shadow_reuse`, `clippy::format_push_string` | [ ] |
| [`src/agent/tools/web.rs`](src/agent/tools/web.rs) | `clippy::as_conversions`, `clippy::shadow_reuse`, `clippy::shadow_unrelated`, `clippy::format_push_string` | [ ] |
| [`src/agent/tuning.rs`](src/agent/tuning.rs) | `clippy::shadow_reuse` | [ ] |
| [`src/agent/turn.rs`](src/agent/turn.rs) | `clippy::as_conversions`, `clippy::shadow_unrelated` | [ ] |
| [`src/agent/validation.rs`](src/agent/validation.rs) | `clippy::shadow_reuse` | [ ] |
| [`src/agent/worker_tools.rs`](src/agent/worker_tools.rs) | `clippy::as_conversions`, `clippy::indexing_slicing` | [ ] |
| **`src/bus`** | | |
| [`src/bus/events.rs`](src/bus/events.rs) | `clippy::indexing_slicing` | [ ] |
| **`src/cron`** | | |
| [`src/cron/executor.rs`](src/cron/executor.rs) | `clippy::as_conversions`, `clippy::shadow_reuse` | [ ] |
| **`src/heartbeat`** | | |
| [`src/heartbeat/health.rs`](src/heartbeat/health.rs) | `clippy::as_conversions` | [ ] |
| [`src/heartbeat/service.rs`](src/heartbeat/service.rs) | `clippy::shadow_reuse` | [ ] |
| **`src/providers`** | | |
| [`src/providers/anthropic.rs`](src/providers/anthropic.rs) | `clippy::as_conversions`, `clippy::indexing_slicing`, `clippy::shadow_reuse`, `clippy::shadow_unrelated` | [ ] |
| [`src/providers/factory.rs`](src/providers/factory.rs) | `clippy::shadow_reuse` | [ ] |
| [`src/providers/jit_gate.rs`](src/providers/jit_gate.rs) | `clippy::shadow_unrelated` | [ ] |
| [`src/providers/openai_compat.rs`](src/providers/openai_compat.rs) | `clippy::as_conversions`, `clippy::indexing_slicing`, `clippy::shadow_reuse`, `clippy::shadow_unrelated` | [ ] |
| [`src/providers/retry.rs`](src/providers/retry.rs) | `clippy::shadow_reuse` | [ ] |
| **`src/session`** | | |
| [`src/session/db.rs`](src/session/db.rs) | `clippy::as_conversions`, `clippy::indexing_slicing`, `clippy::shadow_reuse`, `clippy::shadow_unrelated` | [ ] |
| [`src/session/filters.rs`](src/session/filters.rs) | `clippy::indexing_slicing`, `clippy::shadow_reuse` | [ ] |
| **`src/src-root`** | | |
| [`src/higgs.rs`](src/higgs.rs) | `clippy::as_conversions`, `clippy::indexing_slicing`, `clippy::shadow_reuse`, `clippy::shadow_same` | [ ] |
| [`src/lms.rs`](src/lms.rs) | `clippy::as_conversions`, `clippy::indexing_slicing` | [ ] |
| [`src/local_discovery.rs`](src/local_discovery.rs) | `clippy::shadow_reuse` | [ ] |
| [`src/syntax.rs`](src/syntax.rs) | `clippy::indexing_slicing`, `clippy::format_push_string` | [ ] |
| **`src/utils`** | | |
| [`src/utils/helpers.rs`](src/utils/helpers.rs) | `clippy::indexing_slicing`, `clippy::shadow_reuse`, `clippy::print_stderr` | [ ] |

## Host bridge migration (Phase 3 — complete)

The typed host bridge (research doc §3.2-§3.3) replaced the legacy callback
channel; doc §3.7 Steps 2-3 are done:

- [x] `SpawnTool` holds `Arc<dyn HostBridge>`: `execute` parses `SpawnAction`
      and calls `host.call()`; the 7 callback fields, 7 setters and
      `set_context` are deleted (`SpawnAction` preserves every legacy
      validation string byte-for-byte).
- [x] `MessageTool` holds `Arc<dyn MessageHost>`: `send_callback`,
      `set_send_callback` and `set_context` are deleted.
- [x] `HostBridgeAdapter` deleted — its byte-stability proof was ported into
      permanent tests that drive `SpawnTool`/`MessageTool` through the
      dispatcher over mock hosts emitting the exact pre-bridge wire strings.
- [x] The 8 `build_tools` closures deleted — the logic lives in the named,
      unit-testable `AgentHost` methods (the byte-identical port).
- [x] `src/agent/host_bridge.rs` transitional `#![cfg_attr(not(test),
      allow(dead_code))]` removed — the bridge is live in production.
- [x] `clippy.toml` `disallowed-methods` gate the deleted `set_*_callback`
      API (inert entries block reintroduction); `scripts/quality-sentinel.sh`
      greps zero `set_*_callback` in `src/`.

## Lint legend

| Lint | Meaning |
|---|---|
| `clippy::as_conversions` | `as` casts between numeric/pointer types |
| `clippy::indexing_slicing` | `arr[i]` indexing / slicing (use iterators + get) |
| `clippy::shadow_reuse` | reusing a binding name where the old value is used |
| `clippy::shadow_unrelated` | shadowing without using the old value |
| `clippy::shadow_same` | shadowing with the same value |
| `clippy::format_push_string` | `format!` result pushed into another `String` |
| `clippy::string_add` | `String + String` (use `push_str`/`format!`) |
| `clippy::print_stderr` | `eprintln!`/`eprint!` (use `tracing`) |

## How to migrate a module

1. Delete the module's backlog comment + `#![allow(...)]` block (and its
   `Tracking:` line).
2. Run `cargo clippy --all-targets`; fix each newly surfaced violation with a
   typed-error / iterator / `?`-based fix (doc §3.6), not a new allow.
3. Run the module's tests: `cargo test --lib <module>`.
4. Tick the module's checkbox above and commit.
