# Design: Cua Driver — local desktop computer-use for nanobot

Date: 2026-08-19
Status: Approved (scope amended: vision deferred to follow-up)

## Summary

Add a single `cua` tool that gives the nanobot agent the ability to drive native
GUI apps on the host machine (macOS/Windows/Linux) in the background via
[trycua/cua](https://github.com/trycua/cua) `cua-driver` — click, type,
screenshot, invoke menus, drive browser pages, verify state — without stealing
focus or the cursor.

Decisions locked in with the user:

- **Surface**: local desktop control via cua-driver (not sandboxes, not Lume,
  not cua-bench).
- **Transport**: CLI subprocess (`cua-driver call <tool> <json-args>`), not an
  MCP client, not a generic MCP bridge.
- **Tool shape**: one passthrough `cua` tool (tool + args), mirroring the CLI
  1:1 so the surface stays in sync with cua-driver releases. The cua-driver
  skill pack teaches the LLM which tools to call.
- **Daemon auto-start**: on by default (`daemonAutoStart: true`). The tool
  ensures the daemon is running before each call.
- **Vision**: **deferred to a follow-up**. This change is text-only. Screenshots
  are saved to the workspace and their path returned. Feeding images back to
  vision-capable models is a separate workstream because it touches the provider
  layer and the Higgs KV prefix-cache contract.

## Background

### cua-driver

`cua-driver` is the background computer-use driver from trycua/cua. It speaks
MCP over stdio, but for shell-oriented agents it exposes a CLI:

```
cua-driver call <tool-name> <json-args>
cua-driver call <tool-name> --screenshot-out-file <path>   # write first image block
cua-driver list-tools
cua-driver describe <tool-name>
cua-driver status        # daemon running?
cua-driver doctor        # diagnostics incl. platform + permissions
```

`call` requires a running daemon. Daemon lifecycle:

- macOS: launch via `CuaDriver.app` (`open -n -g -a CuaDriver --args serve`) so
  Accessibility/Screen Recording TCC grants retain the app identity. Raw
  `cua-driver serve` outside the app is unsupported on macOS for TCC reasons.
- Linux/Windows: `cua-driver serve [--permission-mode <mode>]`.

Permission modes: `standard` (promptless default for normal automation),
`bounded` (capability-manifest ceiling), `unrestricted`
(`--dangerously-bypass-approvals`). The mode is fixed at daemon launch.

MCP tool surface (subset): `list_apps`, `launch_app`, `list_windows`,
`get_window_state`, `get_app_state`, `get_desktop_state`, `get_browser_state`,
`browser_prepare/navigate/click/type`, `click`, `double_click`, `right_click`,
`type_text`, `press_key`, `hotkey`, `scroll`, `drag`, `move_cursor`,
`set_window_frame`, `invoke_menu`, `show_menu`, `clipboard_read/write`,
`screenshot`, `verify_state`, session tools, history tools, recording tools.

The agent skill pack (`SKILL.md` + `MACOS.md`/`WINDOWS.md`/`LINUX.md`, published
as `@cua/driver`, MIT-0) teaches the required loop: **snapshot-before-action** —
take an accessibility snapshot, act through snapshot-bound element tokens or
exact geometry, verify from fresh state. The primary grounding is the
accessibility tree (text), so text-only operation is fully supported.

### nanobot

- Tools live in `src/agent/tools/*.rs`, implement the `Tool` trait, register in
  `ToolRegistry::register_standard_tools` behind `should_include(name)`.
- Tool output is a bounded string; results are appended as
  `{"role":"tool","tool_call_id":…,"name":…,"content":…}` via
  `ContextBuilder::add_tool_result*`.
- Skills live at `{workspace}/skills/{name}/SKILL.md` with YAML frontmatter;
  `ReadSkillTool` loads them.
- Providers are text-only today (no image content blocks in OpenAI-compat or
  Anthropic clients). Tool results are strings.
- The Higgs KV prefix-cache contract pins that the rendered wire prompt for
  turn N is a byte-prefix of turn N+1; per-turn content goes in TAIL blocks,
  never `messages[0]`. Enforced by
  `test_local_wire_prompt_prefix_stable_across_turns`.

## Architecture

```
LLM ──tool_call──▶ cua {tool:"click", args:{element_token:"…"}}
                        │ 1. ensure daemon running (status → auto-start)
                        ▼
                   cua-driver call click '{"element_token":"…"}'
                        │ 2. text output (bounded)
                        ▼
                   reply; screenshots → workspace/cua/*.png (path returned)
```

## Components

### 1. `CuaTool` — `src/agent/tools/cua.rs`

One file, following the `web.rs`/`shell.rs` patterns.

- `name()` → `"cua"`.
- `description()` → "Drive a native GUI app on this machine via cua-driver
  (snapshot before acting; prefer accessibility-tree element tokens over pixel
  coordinates). Tools: list_apps, launch_app, list_windows, get_window_state,
  click, type_text, press_key, hotkey, scroll, drag, invoke_menu,
  clipboard_read/write, screenshot, browser_*, verify_state, …"
- `parameters()` →
  ```json
  {
    "type": "object",
    "properties": {
      "tool": {"type": "string", "description": "cua-driver MCP tool name to invoke"},
      "args": {"type": "object", "description": "JSON args for that tool (per its schema)"}
    },
    "required": ["tool"]
  }
  ```
- `PermissionLevel::System` — it controls the host desktop.
- `ToolConcurrency::Sequential` — stateful desktop automation; never parallel
  with other calls.

Execution path (`execute_with_context`, honoring cancellation):

1. **Ensure daemon**: run `cua-driver status`. If not running and
   `daemonAutoStart` is enabled (default true), launch it:
   - macOS: `open -n -g -a CuaDriver --args serve` (preserves TCC app identity)
   - Linux/Windows: `cua-driver serve --permission-mode <mode>` detached
   - Poll `status` until ready (bounded retries).
   If auto-start is off and daemon is down → error string with the exact
   platform launch command.
2. **Invoke**: `cua-driver call <tool> <args-json>` via
   `std::process::Command` (argv array — never a shell). Apply `exec_timeout`.
   For image-producing tools (`screenshot`, `get_window_state`), pass
   `--screenshot-out-file <workspace>/cua/<call_id>.png`; return the file path
   alongside any text output.
3. **Bound output** to `max_tool_result_chars` like every other tool.
4. **Discovery fallback**: if `tool` is missing/unknown, run
   `cua-driver list-tools` and return the list so the agent can self-correct.

### 2. Config — `src/config/schema.rs`

camelCase, following `ExecToolConfig`:

```json
"cua": {
  "enabled": true,
  "binaryPath": "cua-driver",
  "permissionMode": "standard",
  "daemonAutoStart": true,
  "screenshotDir": null
}
```

- `binaryPath` — override for the `cua-driver` binary (default resolved from
  PATH).
- `permissionMode` — `standard` (default) | `bounded` | `unrestricted`; only
  applied at daemon launch.
- `daemonAutoStart` — default **true** (user decision).
- `screenshotDir` — defaults to `{workspace}/cua`.

Registered in `register_standard_tools` behind `should_include("cua")`, gated on
`config.cua.enabled`.

### 3. Skill pack install (one-time, documented)

Copy cua-driver's `SKILL.md` + platform docs into
`{workspace}/skills/cua-driver/`. Nanobot's `ReadSkillTool` loads it as a
regular skill; the LLM then knows the snapshot-before-action loop and the tool
surface. Documented in the README as a one-time setup step (and optionally
offered by `cua-driver doctor`-style guidance in the tool's error output).

## Error handling & safety

- Daemon down + auto-start failed → `Error: cua-driver daemon not running.
  Start it with: <exact platform command>`.
- Binary missing → actionable error pointing at the official installer
  (`curl -fsSL https://cua.ai/driver/install.sh | bash` / PowerShell equivalent).
- macOS TCC: never spawn a raw `serve` with arbitrary path identity; launch via
  `CuaDriver.app`. `--direct`/embedded nuances are the user's choice, not the
  tool's default.
- `args` must parse as JSON; tool name and args are passed as argv, never
  through a shell (no injection surface).
- Screenshots are written only under the configured screenshot dir (default
  workspace); paths are bounded.
- Output bounded by `max_tool_result_chars`.

## Explicitly out of scope (follow-ups)

- **Vision workstream** (deferred): image content blocks in providers
  (OpenAI-compat `image_url`, Anthropic `image`), `supportsVision` capability
  gate, image-bearing TAIL blocks under the Higgs prefix-cache contract,
  provider rendering tests + prefix-stability regression test with image tail.
  Touches the provider layer and Higgs hot path — separate plan.
- cua sandboxes / cloud fleet (Python SDK) — not now.
- Generic MCP bridge — rejected; cua only.
- Curated nanobot-native tool wrappers — rejected; passthrough only.

## Testing

- Unit (in `cua.rs`): arg building, daemon-status parsing, auto-start command
  selection per platform, screenshot path handling. Use a mock `cua-driver`
  shim (a small script/bin in tests that echoes args and exits per script).
- Registry test: `cua` registered when enabled, absent when disabled.
- Config test: schema parses `cua` block with defaults.
- Validation: `cargo build`, `cargo test`, manual `cua-driver doctor` smoke on
  the host.

## Open questions resolved

- Autostart default → true (user decision).
- Vision in this change → no, follow-up (user decision).
