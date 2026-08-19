# Cua Driver (Local Desktop Computer-Use) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the nanobot agent a `cua` tool that drives native GUI apps on the host machine (macOS/Windows/Linux) in the background via `cua-driver` — click, type, screenshot, menus, browser pages — with daemon auto-start and the cua-driver skill pack installed into the workspace.

**Architecture:** One passthrough `cua` tool (`src/agent/tools/cua.rs`) that shells out to the `cua-driver` CLI (`status` / `call` / `list-tools`) as argv arrays (never a shell). Config flows `Config.tools.cua` → `SwappableCoreConfig`/`SwappableCore` → `ToolConfig.cua` → `register_standard_tools`, mirroring the `code_execution`/`python_kernel` chain exactly. Daemon ensure runs `cua-driver status` (exit 0 = running); if down and `daemonAutoStart` (default true), launch (macOS: `open -n -g -a CuaDriver --args serve` for TCC app identity; else `serve --permission-mode <mode>` detached) and poll. Screenshots: `--screenshot-out-file` always passed; path returned only if the file is written. Text-only (vision is a separate follow-up workstream touching the provider layer and Higgs prefix-cache).

**Tech Stack:** Rust 2021, tokio (`tokio::process::Command`), serde/serde_json, `async_trait`. No new dependencies.

## Global Constraints

- Rust 2021; `snake_case` fns, `camelCase` serde keys (copy from existing `ExecToolConfig`).
- Lints in Cargo.toml: `unwrap_used`, `expect_used`, `panic`, `todo`, `indexing_slicing`, `as_conversions` are `deny` in non-test code (tests may use `unwrap()` like `agent_loop/tests.rs` does).
- Tools return `String`, errors prefixed `"Error: "` (use `require_str!` for required params — byte-identical error strings).
- Subprocess invocation must use `std::process::Command`/`tokio::process::Command` with an argv array — never a shell string.
- `PermissionLevel::System` (drives the host desktop), `ToolConcurrency::Sequential` (stateful; never parallel).
- Provider layer stays untouched: text-only output in this change. No image content blocks.
- No new `bool` params — the permission mode is a `String` enum value from config (`standard` | `bounded` | `unrestricted`).
- `cua-driver status` contract (verified from driver source `serve.rs`): exit 0 + stdout `Cua Driver daemon is running` when up; exit 1 + stderr `Cua Driver daemon is not running` when down. `cua-driver call` exits 1 with the same stderr when the daemon is down.

---
---

### Task 1: `CuaToolConfig` schema + `ToolsConfig.cua` field

**Files:**
- Modify: `src/config/schema.rs` (add struct near `ExecToolConfig` ~line 783; add field to `ToolsConfig` ~line 886; add test in `mod tests` ~line 2477)

**Interfaces:**
- Produces: `pub struct CuaToolConfig` (fields below), `impl Default for CuaToolConfig`, `ToolsConfig.cua: CuaToolConfig`. Later tasks consume these exact names.

- [ ] **Step 1: Write the failing test**

Append inside `mod tests` in `src/config/schema.rs`:

```rust
#[test]
fn test_cua_config_roundtrip() {
    // Defaults.
    let default = CuaToolConfig::default();
    assert!(default.enabled);
    assert_eq!(default.permission_mode, "standard");
    assert!(default.daemon_auto_start);
    assert_eq!(default.binary_path, None);
    assert_eq!(default.screenshot_dir, None);

    // Explicit JSON (camelCase) parses.
    let json = r#"{
        "tools": {
            "cua": {
                "enabled": false,
                "binaryPath": "/opt/bin/cua-driver",
                "permissionMode": "bounded",
                "daemonAutoStart": false,
                "screenshotDir": "/tmp/shots"
            }
        }
    }"#;
    let cfg: Config = serde_json::from_str(json).unwrap();
    let cua = &cfg.tools.cua;
    assert!(!cua.enabled);
    assert_eq!(cua.binary_path.as_deref(), Some("/opt/bin/cua-driver"));
    assert_eq!(cua.permission_mode, "bounded");
    assert!(!cua.daemon_auto_start);
    assert_eq!(
        cua.screenshot_dir.as_deref(),
        Some(std::path::Path::new("/tmp/shots"))
    );

    // Missing block falls back to defaults.
    let cfg2: Config = serde_json::from_str(r#"{"tools": {}}"#).unwrap();
    assert!(cfg2.tools.cua.enabled);
    assert!(cfg2.tools.cua.daemon_auto_start);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib config::schema::tests::test_cua_config_roundtrip`
Expected: FAIL — `CuaToolConfig` not found (compile error), and `tools.cua` field missing.

- [ ] **Step 3: Implement the struct and field**

In `src/config/schema.rs`, right after the `ExecToolConfig` impl block (after line ~802):

```rust
/// Cua driver (local desktop computer-use) tool configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CuaToolConfig {
    /// When false, the `cua` tool is not registered.
    #[serde(default = "default_cua_enabled")]
    pub enabled: bool,
    /// Path to the cua-driver binary. `None` resolves `cua-driver` on PATH.
    #[serde(default)]
    pub binary_path: Option<String>,
    /// Daemon permission mode applied at launch: standard | bounded | unrestricted.
    #[serde(default = "default_cua_permission_mode")]
    pub permission_mode: String,
    /// Auto-start the cua-driver daemon when a call finds it not running.
    #[serde(default = "default_cua_daemon_auto_start")]
    pub daemon_auto_start: bool,
    /// Directory for screenshots. `None` defaults to `<workspace>/cua`.
    #[serde(default)]
    pub screenshot_dir: Option<PathBuf>,
}

fn default_cua_enabled() -> bool {
    true
}

fn default_cua_permission_mode() -> String {
    "standard".to_string()
}

fn default_cua_daemon_auto_start() -> bool {
    true
}

impl Default for CuaToolConfig {
    fn default() -> Self {
        Self {
            enabled: default_cua_enabled(),
            binary_path: None,
            permission_mode: default_cua_permission_mode(),
            daemon_auto_start: default_cua_daemon_auto_start(),
            screenshot_dir: None,
        }
    }
}
```

Then add the field to `ToolsConfig` (after `pub code_execution: CodeExecutionConfig,` ~line 898):

```rust
    /// Cua driver (local desktop computer-use) tool settings.
    #[serde(default)]
    pub cua: CuaToolConfig,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib config::schema::tests::test_cua_config_roundtrip`
Expected: PASS.

- [ ] **Step 5: Run the full suite once**

Run: `cargo test`
Expected: all pass (the new field is serde-defaulted, so existing config tests are unaffected).

- [ ] **Step 6: Commit**

```bash
git add src/config/schema.rs
git commit -m "feat(config): add cua-driver tool config (CuaToolConfig)"
```

---
---

### Task 2: Plumb `cua` config through the core chain

**Files:**
- Modify: `src/agent/agent_core.rs` — `SwappableCoreConfig` struct (~line 1380, after `python_kernel`), `SwappableCore` struct (~line 119, after `python_kernel`), destructure in `build_swappable_core` (~line 1419), `SwappableCore { ... }` literal (~line 1573)
- Modify: `src/cli/core_builder.rs:456` (pass `cua` into `SwappableCoreConfig`)
- Modify: `src/agent/tools/registry.rs:33` — add `cua` to `ToolConfig`, set in `ToolConfig::new`
- Modify: `src/agent/tool_wiring.rs:326` — set `cua: core.cua.clone()` in the `ToolConfig` literal
- Modify: `src/agent/agent_loop/tests.rs` — add `CuaToolConfig` to the import at line 12, insert `cua: CuaToolConfig::default(),` after every `python_kernel:` line (22 sites; sed below)

**Interfaces:**
- Consumes: `CuaToolConfig` from Task 1.
- Produces: `SwappableCoreConfig.cua: CuaToolConfig`, `SwappableCore.cua: CuaToolConfig`, `ToolConfig.cua: CuaToolConfig` (defaulted in `ToolConfig::new`). Task 3/4 consume `ToolConfig.cua`.

- [ ] **Step 1: Add the field to both core structs**

In `src/agent/agent_core.rs`, after the `python_kernel` field in `SwappableCoreConfig` (line ~1379):

```rust
    /// Cua driver (local desktop computer-use) tool settings.
    pub cua: crate::config::schema::CuaToolConfig,
```

Same field (with the same comment) after `python_kernel` in `SwappableCore` (line ~119).

- [ ] **Step 2: Destructure + forward in `build_swappable_core`**

In `build_swappable_core`, add `cua,` to the `let SwappableCoreConfig { ... } = cfg;` destructure (after `python_kernel,` ~line 1420), and add `cua,` to the `SwappableCore { ... }` literal (after `python_kernel,` ~line 1573).

- [ ] **Step 3: Wire the source in core_builder**

In `src/cli/core_builder.rs` (in `build_swappable_core`'s caller, after `python_kernel: config.tools.python_kernel.clone(),` ~line 457):

```rust
        cua: config.tools.cua.clone(),
```

- [ ] **Step 4: Add `cua` to `ToolConfig` + default**

In `src/agent/tools/registry.rs`, add the field to `pub struct ToolConfig` (after `pub python_kernel: PythonKernelConfig,`):

```rust
    /// Cua driver (local desktop computer-use) tool settings.
    pub cua: crate::config::schema::CuaToolConfig,
```

In `ToolConfig::new` (after the `python_kernel:` line):

```rust
            cua: crate::config::schema::CuaToolConfig::default(),
```

- [ ] **Step 5: Forward in `build_tools` (tool_wiring.rs)**

In `src/agent/tool_wiring.rs` `build_tools`, after `python_kernel: core.python_kernel.clone(),` (line ~343):

```rust
            cua: core.cua.clone(),
```

- [ ] **Step 6: Fix the 22 test literals + import**

Add `CuaToolConfig` to the schema import in `src/agent/agent_loop/tests.rs` (line 12-14):

```rust
use crate::config::schema::{
    AdaptiveTokenConfig, CodeExecutionConfig, CuaToolConfig, MemoryConfig, ProvenanceConfig,
    ProviderConfig, PythonKernelConfig, ToolDelegationConfig, TrioConfig,
};
```

Insert `cua: CuaToolConfig::default(),` after every `python_kernel: PythonKernelConfig::default(),` line (22 sites, 2 indent depths):

```bash
sed -i '' -E 's/^( *)python_kernel: PythonKernelConfig::default\(\),$/\1python_kernel: PythonKernelConfig::default(),\n\1cua: CuaToolConfig::default(),/' src/agent/agent_loop/tests.rs
grep -c "cua: CuaToolConfig::default()" src/agent/agent_loop/tests.rs   # expect 22
```

- [ ] **Step 7: Build + test**

Run: `cargo build` then `cargo test`
Expected: clean build; all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/agent/agent_core.rs src/cli/core_builder.rs src/agent/tools/registry.rs src/agent/tool_wiring.rs src/agent/agent_loop/tests.rs
git commit -m "feat(config): thread cua-driver config through core to ToolConfig"
```

---
---

### Task 3: `CuaTool` implementation (daemon ensure + call + screenshot + list-tools)

**Files:**
- Create: `src/agent/tools/cua.rs` (tool + inline `#[cfg(test)] mod tests`)
- Modify: `src/agent/tools/mod.rs` (add `pub mod cua;` and re-export)

**Interfaces:**
- Consumes: `CuaToolConfig` (Task 1), `Tool` trait from `super::base`.
- Produces: `pub struct CuaTool` with `pub fn new(config: &CuaToolConfig, workspace: &Path) -> Self` and a `#[cfg(test)]` helper `with_binary(binary: &str) -> Self`. `execute_with_context` returns the tool-result `String`. Later task registers it.

- [ ] **Step 1: Write the failing tests (tool surface + helpers)**

Create `src/agent/tools/cua.rs` with the module doc, imports, and this test module first (implementation below will fill in the gaps):

```rust
//! Cua driver tool: background computer-use on the host desktop via the
//! `cua-driver` CLI. Daemon contract (from cua-driver serve.rs): `status`
//! exits 0 when the daemon is listening, 1 otherwise; `call` exits 1 with
//! "Cua Driver daemon is not running" on stderr when the daemon is down.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use tokio::process::Command;

use super::base::{require_str, PermissionLevel, Tool, ToolConcurrency, ToolContext};
use crate::config::schema::CuaToolConfig;

const DEFAULT_BINARY: &str = "cua-driver";
const DEFAULT_TIMEOUT_SECS: u64 = 30;
const DAEMON_READY_POLLS: u32 = 10;
const DAEMON_READY_POLL_MS: u64 = 200;

pub struct CuaTool {
    binary: String,
    permission_mode: String,
    daemon_auto_start: bool,
    screenshot_dir: PathBuf,
    timeout: Duration,
}

impl CuaTool {
    pub fn new(config: &CuaToolConfig, workspace: &Path) -> Self {
        let screenshot_dir = config
            .screenshot_dir
            .clone()
            .unwrap_or_else(|| workspace.join("cua"));
        Self {
            binary: config
                .binary_path
                .clone()
                .unwrap_or_else(|| DEFAULT_BINARY.to_string()),
            permission_mode: config.permission_mode.clone(),
            daemon_auto_start: config.daemon_auto_start,
            screenshot_dir,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
        }
    }

    /// Test seam: pin the binary to an absolute path (e.g. a shim script).
    #[cfg(test)]
    fn with_binary(binary: &str) -> Self {
        Self {
            binary: binary.to_string(),
            permission_mode: "standard".to_string(),
            daemon_auto_start: false,
            screenshot_dir: std::env::temp_dir().join("cua-test-shots"),
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    /// Write an executable `cua-driver` shim into a temp dir. Behavior driven
    /// by marker files: `status` exits 0 when `.running` exists; `call` echoes
    /// its argv to `.last_call` and (when `--screenshot-out-file` is present)
    /// creates that file; `list-tools` prints a fixed list; `serve` creates
    /// `.running`.
    fn make_shim(dir: &Path) -> PathBuf {
        let bin = dir.join("cua-driver");
        let script = format!(
            r#"#!/bin/sh
if [ "$1" = "status" ]; then
  if [ -f "{d}/.running" ]; then echo "Cua Driver daemon is running"; exit 0; fi
  echo "Cua Driver daemon is not running" >&2; exit 1
fi
if [ "$1" = "call" ]; then
  echo "$@" > "{d}/.last_call"
  prev=""
  for a in "$@"; do
    if [ "$prev" = "--screenshot-out-file" ]; then touch "$a"; fi
    prev="$a"
  done
  echo "OK"
  exit 0
fi
if [ "$1" = "list-tools" ]; then echo "list_apps"; echo "click"; echo "type_text"; exit 0; fi
if [ "$1" = "serve" ]; then touch "{d}/.running"; exit 0; fi
echo "unknown command: $1" >&2; exit 1
"#,
            d = dir.display()
        );
        fs::write(&bin, script).unwrap();
        #[cfg(unix)]
        {
            let mut perms = fs::metadata(&bin).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&bin, perms).unwrap();
        }
        bin
    }

    #[test]
    fn test_cua_tool_name_and_schema() {
        let tool = CuaTool::with_binary("cua-driver");
        assert_eq!(tool.name(), "cua");
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        assert_eq!(params["properties"]["tool"]["type"], "string");
        assert_eq!(
            params["required"],
            serde_json::json!(["tool"])
        );
        assert_eq!(tool.permission(), PermissionLevel::System);
        assert_eq!(tool.concurrency(), ToolConcurrency::Sequential);
    }

    #[tokio::test]
    async fn test_missing_tool_param_lists_tools() {
        let dir = std::env::temp_dir().join(format!("cua-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let bin = make_shim(&dir);
        let tool = CuaTool::with_binary(bin.to_str().unwrap());
        let result = tool.execute(HashMap::new()).await;
        assert!(result.contains("list_apps"), "got: {result}");
        assert!(result.contains("click"), "got: {result}");
        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_call_passes_tool_and_args() {
        let dir = std::env::temp_dir().join(format!("cua-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let bin = make_shim(&dir);
        fs::write(dir.join(".running"), "").unwrap(); // daemon "up"
        let tool = CuaTool::with_binary(bin.to_str().unwrap());
        let mut params = HashMap::new();
        params.insert("tool".to_string(), Value::String("click".to_string()));
        params.insert(
            "args".to_string(),
            serde_json::json!({"element_token": "@e3"}),
        );
        let result = tool.execute(params).await;
        assert!(result.contains("OK"), "got: {result}");
        let last = fs::read_to_string(dir.join(".last_call")).unwrap();
        assert!(last.contains("call"), "got: {last}");
        assert!(last.contains("click"), "got: {last}");
        assert!(last.contains("element_token"), "got: {last}");
        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_daemon_down_no_autostart_errors_with_launch_hint() {
        let dir = std::env::temp_dir().join(format!("cua-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let bin = make_shim(&dir); // no .running marker → status exit 1
        let tool = CuaTool::with_binary(bin.to_str().unwrap()); // daemon_auto_start=false
        let mut params = HashMap::new();
        params.insert("tool".to_string(), Value::String("click".to_string()));
        let result = tool.execute(params).await;
        assert!(result.starts_with("Error:"), "got: {result}");
        assert!(
            result.contains("cua-driver") || result.contains("CuaDriver"),
            "expected launch hint, got: {result}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_screenshot_out_file_returned_when_written() {
        let dir = std::env::temp_dir().join(format!("cua-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let bin = make_shim(&dir);
        fs::write(dir.join(".running"), "").unwrap();
        let tool = CuaTool::with_binary(bin.to_str().unwrap());
        let mut params = HashMap::new();
        params.insert("tool".to_string(), Value::String("screenshot".to_string()));
        let result = tool.execute(params).await;
        assert!(result.contains("OK"), "got: {result}");
        assert!(
            result.contains("Screenshot saved"),
            "expected screenshot path in result, got: {result}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_daemon_launch_args_pure() {
        // macOS: launch via CuaDriver.app so TCC grants keep app identity.
        let mac = daemon_launch_args("cua-driver", "standard", true);
        assert_eq!(
            mac,
            vec![
                "open".to_string(),
                "-n".to_string(),
                "-g".to_string(),
                "-a".to_string(),
                "CuaDriver".to_string(),
                "--args".to_string(),
                "serve".to_string()
            ]
        );
        // Linux/Windows: serve directly with the permission mode.
        let other = daemon_launch_args("/opt/cua/cua-driver", "bounded", false);
        assert_eq!(
            other,
            vec![
                "/opt/cua/cua-driver".to_string(),
                "serve".to_string(),
                "--permission-mode".to_string(),
                "bounded".to_string()
            ]
        );
    }

    #[test]
    fn test_binary_present_absolute_and_missing() {
        let dir = std::env::temp_dir().join(format!("cua-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let bin = make_shim(&dir);
        let tool = CuaTool::with_binary(bin.to_str().unwrap());
        assert!(tool.binary_present());
        let missing = CuaTool::with_binary("/nonexistent/cua-driver");
        assert!(!missing.binary_present());
        fs::remove_dir_all(&dir).ok();
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib agent::tools::cua`
Expected: FAIL — `daemon_launch_args` not found, `binary_present` not found, and `execute` returns the default trait error.

- [ ] **Step 3: Implement the tool**

Add to `src/agent/tools/cua.rs` (before the test module):

```rust
impl CuaTool {
    /// Whether the configured binary resolves (absolute path exists, or on PATH).
    fn binary_present(&self) -> bool {
        let p = Path::new(&self.binary);
        if p.is_absolute() {
            return p.is_file();
        }
        std::env::var_os("PATH").map_or(false, |paths| {
            std::env::split_paths(&paths).any(|d| d.join(&self.binary).is_file())
        })
    }

    /// Run a driver subcommand; returns (stdout, stderr, success).
    async fn run_driver(&self, args: &[&str]) -> (String, String, bool) {
        let result = tokio::time::timeout(
            self.timeout,
            Command::new(&self.binary).args(args).output(),
        )
        .await;
        match result {
            Ok(Ok(output)) => (
                String::from_utf8_lossy(&output.stdout).to_string(),
                String::from_utf8_lossy(&output.stderr).to_string(),
                output.status.success(),
            ),
            Ok(Err(e)) => (
                String::new(),
                format!("failed to run cua-driver: {e}"),
                false,
            ),
            Err(_) => (
                String::new(),
                format!("cua-driver call timed out after {}s", self.timeout.as_secs()),
                false,
            ),
        }
    }

    /// `cua-driver status` exits 0 when the daemon is listening.
    async fn daemon_running(&self) -> bool {
        let (_out, _err, ok) = self.run_driver(&["status"]).await;
        ok
    }

    /// Launch command argv, pure for testability. macOS uses `open -a
    /// CuaDriver` so Accessibility/Screen Recording grants keep the app
    /// identity (raw `serve` outside the app is unsupported on macOS).
    fn daemon_launch_args(&self) -> Vec<String> {
        daemon_launch_args(&self.binary, &self.permission_mode, cfg!(target_os = "macos"))
    }

    /// Ensure the daemon is up before calling. Auto-start is bounded: launch,
    /// then poll `status` up to DAEMON_READY_POLLS times.
    async fn ensure_daemon(&self) -> Result<(), String> {
        if self.daemon_running().await {
            return Ok(());
        }
        if !self.daemon_auto_start {
            return Err(format!(
                "Error: cua-driver daemon is not running. Start it with: {}",
                self.daemon_launch_args().join(" ")
            ));
        }
        let args = self.daemon_launch_args();
        let spawn = std::process::Command::new(&args[0])
            .args(&args[1..])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
        if spawn.is_err() {
            return Err(format!(
                "Error: failed to launch cua-driver daemon ({}). Run `cua-driver doctor`.",
                self.binary
            ));
        }
        for _ in 0..DAEMON_READY_POLLS {
            tokio::time::sleep(Duration::from_millis(DAEMON_READY_POLL_MS)).await;
            if self.daemon_running().await {
                return Ok(());
            }
        }
        Err("Error: cua-driver daemon did not become ready after launch. Run `cua-driver doctor`.".to_string())
    }

    /// Run `cua-driver list-tools` and return the output (discovery fallback).
    async fn list_tools(&self) -> String {
        let (out, _err, _ok) = self.run_driver(&["list-tools"]).await;
        out
    }
}

/// Pure launch-argv builder (see `CuaTool::daemon_launch_args`).
fn daemon_launch_args(binary: &str, permission_mode: &str, macos: bool) -> Vec<String> {
    if macos {
        vec![
            "open".to_string(),
            "-n".to_string(),
            "-g".to_string(),
            "-a".to_string(),
            "CuaDriver".to_string(),
            "--args".to_string(),
            "serve".to_string(),
        ]
    } else {
        vec![
            binary.to_string(),
            "serve".to_string(),
            "--permission-mode".to_string(),
            permission_mode.to_string(),
        ]
    }
}

#[async_trait]
impl Tool for CuaTool {
    fn name(&self) -> &str {
        "cua"
    }

    fn description(&self) -> &str {
        "Drive a native GUI app on this machine via cua-driver (background \
         computer-use: click, type, screenshot, menus, browser pages, verify \
         state). Snapshot before acting and prefer accessibility-tree element \
         tokens over pixel coordinates. Tools: list_apps, launch_app, \
         list_windows, get_window_state, get_desktop_state, click, \
         double_click, right_click, type_text, press_key, hotkey, scroll, \
         drag, invoke_menu, clipboard_read, clipboard_write, screenshot, \
         browser_navigate, browser_click, browser_type, verify_state, ..."
    }

    fn permission(&self) -> PermissionLevel {
        PermissionLevel::System
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Sequential
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "description": "cua-driver MCP tool name to invoke (run cua with no args to list available tools)"
                },
                "args": {
                    "type": "object",
                    "description": "JSON arguments for that tool, per its input schema"
                }
            },
            "required": ["tool"]
        })
    }

    async fn execute(&self, params: HashMap<String, Value>) -> String {
        let (event_tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let ctx = ToolContext::new(
            None,
            event_tx,
            tokio_util::sync::CancellationToken::new(),
            String::new(),
        );
        self.execute_with_context(params, &ctx).await
    }

    async fn execute_with_context(
        &self,
        params: HashMap<String, Value>,
        _ctx: &ToolContext,
    ) -> String {
        if !self.binary_present() {
            return "Error: cua-driver binary not found. Install it with: \
                    curl -fsSL https://cua.ai/driver/install.sh | bash"
                .to_string();
        }

        let Some(tool) = params.get("tool").and_then(|v| v.as_str()) else {
            // Discovery fallback: missing tool param → surface the surface.
            let list = self.list_tools().await;
            return format!(
                "Error: 'tool' parameter is required. Available cua-driver tools:\n{list}"
            );
        };

        let args_json = match params.get("args") {
            Some(v) => serde_json::to_string(v).unwrap_or_else(|_| "{}".to_string()),
            None => "{}".to_string(),
        };

        if let Err(e) = self.ensure_daemon().await {
            return e;
        }

        // Always pass --screenshot-out-file: the driver only writes the file
        // when the response contains an image block, so it is harmless for
        // non-image tools.
        let call_id = _ctx.call_id();
        let shot_name = if call_id.is_empty() {
            "screenshot.png".to_string()
        } else {
            let safe: String = call_id
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect();
            format!("cua-{safe}.png")
        };
        let shot_path = self.screenshot_dir.join(shot_name);
        let shot_str = shot_path.to_string_lossy().to_string();
        let shot_arg = "--screenshot-out-file";
        // Ensure the screenshot dir exists before the driver writes into it.
        if let Err(e) = std::fs::create_dir_all(&self.screenshot_dir) {
            return format!("Error: cannot create screenshot dir {}: {e}", self.screenshot_dir.display());
        }
        let (_out, _err, ok) = self
            .run_driver(&["call", tool, &args_json, shot_arg, &shot_str])
            .await;
        let mut out = _out;
        if !ok {
            if _err.contains("daemon is not running") {
                return format!(
                    "Error: cua-driver daemon is not running. Start it with: {}",
                    self.daemon_launch_args().join(" ")
                );
            }
            let detail = if _err.is_empty() { out.clone() } else { _err };
            return format!("Error: cua-driver call '{tool}' failed: {detail}");
        }
        if shot_path.is_file() {
            out.push_str(&format!("\n\nScreenshot saved: {shot_str}"));
        }
        out
    }
}
```

Notes on the code above (read before typing):
- `run_driver` takes `&[&str]`; the call site builds a temporary array `["call", tool, &args_json, shot_arg, &shot_str]` — all elements live long enough for the awaited call, and the `&str` refs borrow from locals. If borrowck complains, change `run_driver` to take `&[String]` and pass `[args...].map(String::from)`.
- `require_str!` from `super::base` is imported but unused in the final shape (tool-missing handled via discovery fallback); drop the import if the compiler flags it. Keep the `Error: ` prefix on every failure path.

- [ ] **Step 4: Register the module**

In `src/agent/tools/mod.rs`, add `pub mod cua;` (alphabetical, after `code_execution`) and `pub use cua::CuaTool;` in the re-export block.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --lib agent::tools::cua`
Expected: all 7 tests PASS.

- [ ] **Step 6: Clippy + full build**

Run: `cargo build` and `cargo clippy`
Expected: clean (no new warnings; fix any lint the tool trips — e.g. unused import, `format_push_string`).

- [ ] **Step 7: Commit**

```bash
git add src/agent/tools/cua.rs src/agent/tools/mod.rs
git commit -m "feat(tools): add cua tool for cua-driver desktop computer-use"
```

---
---

### Task 4: Register `cua` in `register_standard_tools` + registry tests

**Files:**
- Modify: `src/agent/tools/registry.rs` — `register_standard_tools` (after the `browser` block ~line 392) + test module

**Interfaces:**
- Consumes: `CuaTool::new(&CuaToolConfig, &Path)` from Task 3; `ToolConfig.cua` from Task 2.
- Produces: the `cua` tool in the standard registry, gated on `config.cua.enabled`.

- [ ] **Step 1: Write the failing registry tests**

Append to `mod tests` in `src/agent/tools/registry.rs`:

```rust
#[test]
fn test_cua_registered_when_enabled() {
    let ws = std::path::Path::new("/tmp");
    let mut cfg = ToolConfig::new(ws);
    cfg.cua.enabled = true;
    let reg = ToolRegistry::with_standard_tools(&cfg);
    assert!(reg.contains_key("cua"), "cua should be registered when enabled");
}

#[test]
fn test_cua_not_registered_when_disabled() {
    let ws = std::path::Path::new("/tmp");
    let mut cfg = ToolConfig::new(ws);
    cfg.cua.enabled = false;
    let reg = ToolRegistry::with_standard_tools(&cfg);
    assert!(!reg.contains_key("cua"), "cua should be absent when disabled");
}

#[test]
fn test_cua_excluded_by_tools_filter() {
    let ws = std::path::Path::new("/tmp");
    let mut cfg = ToolConfig::new(ws);
    cfg.tools_filter = Some(vec!["read_file".to_string()]);
    let reg = ToolRegistry::with_standard_tools(&cfg);
    assert!(!reg.contains_key("cua"));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib agent::tools::registry::tests::test_cua`
Expected: FAIL — `cua` not in registry.

- [ ] **Step 3: Register the tool**

In `register_standard_tools`, after the `browser` block:

```rust
        if should_include("cua") && config.cua.enabled {
            self.register(Box::new(CuaTool::new(&config.cua, &config.workspace)));
        }
```

Add `use super::cua::CuaTool;` to the imports in `registry.rs` (or `crate::agent::tools::cua::CuaTool`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib agent::tools::registry::tests::test_cua`
Expected: all 3 PASS. Also `cargo test` full — the new registration must not disturb existing toolset/filter tests.

- [ ] **Step 5: Commit**

```bash
git add src/agent/tools/registry.rs
git commit -m "feat(tools): register cua tool behind config.cua.enabled"
```

---
---

### Task 5: Skill pack install docs + README section

**Files:**
- Modify: `README.md` (add a "Cua Driver (local desktop computer-use)" section after the tools/config area)

**Interfaces:** none (documentation only).

- [ ] **Step 1: Write the README section**

Add to `README.md`:

```markdown
## Cua Driver — local desktop computer-use

The `cua` tool lets the agent drive native GUI apps on this machine in the
background (click, type, screenshot, menus, browser pages) via
[cua-driver](https://github.com/trycua/cua) from trycua/cua.

### One-time setup

1. Install cua-driver:

   ```bash
   curl -fsSL https://cua.ai/driver/install.sh | bash
   ```

   (Windows: `irm https://cua.ai/driver/install.ps1 | iex`)

2. Grant macOS Accessibility + Screen Recording to `CuaDriver.app` when
   prompted, then start it once (`open -n -g -a CuaDriver --args serve`).
   The tool auto-starts the daemon on first use; keep `daemonAutoStart`
   enabled unless you manage the daemon yourself.

3. Install the cua-driver agent skill pack so the model knows the
   snapshot-before-action loop:

   ```bash
   mkdir -p ~/.nanobot/workspace/skills/cua-driver
   curl -fsSL https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/rust/Skills/cua-driver/SKILL.md -o ~/.nanobot/workspace/skills/cua-driver/SKILL.md
   # optional platform docs:
   curl -fsSL https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/rust/Skills/cua-driver/MACOS.md -o ~/.nanobot/workspace/skills/cua-driver/MACOS.md
   ```

### Configuration

```json
{
  "tools": {
    "cua": {
      "enabled": true,
      "binaryPath": "cua-driver",
      "permissionMode": "standard",
      "daemonAutoStart": true,
      "screenshotDir": null
    }
  }
}
```

`screenshotDir` defaults to `<workspace>/cua`. `permissionMode` is
`standard` (default), `bounded` (requires a capability manifest), or
`unrestricted` (requires `--dangerously-bypass-approvals` at daemon launch).
Screenshots are saved to a file and their path returned; feeding images back
to vision models is a planned follow-up.
```

- [ ] **Step 2: Verify the markdown renders**

Run: `grep -n "Cua Driver" README.md` — confirm the section header exists once.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document cua-driver setup, skill pack, and config"
```

---
---

### Task 6: End-to-end validation

**Files:** none (verification only).

- [ ] **Step 1: Full build + tests**

Run: `cargo build` and `cargo test`
Expected: clean build; full suite passes (including the new cua/config/registry tests and the Higgs prefix-stability regression — untouched by this change).

- [ ] **Step 2: Clippy**

Run: `cargo clippy`
Expected: no new warnings.

- [ ] **Step 3: Manual smoke (host has cua-driver installed)**

```bash
cua-driver doctor
cargo run -- agent -m "Use the cua tool to list what apps are running on this machine"
```

Expected: the agent calls `cua` → daemon auto-starts → `list_apps` result returns.
If `cua-driver` is not installed, expect the install-hint error string instead (that path is also verified by `test_missing_binary`-style behavior in the tool).

- [ ] **Step 4: Final review**

Run: `git log --oneline -8` — expect the 6 feature commits from Tasks 1-5 plus this validation. Review the diff with `git diff main..HEAD --stat`.

- [ ] **Step 5: Verify against spec**

Open `docs/superpowers/specs/2026-08-19-cua-driver-design.md` and confirm every component has a task:
- `CuaTool` passthrough → Task 3 ✅
- daemon ensure + auto-start (default true) → Task 3 ✅
- `--screenshot-out-file` always passed, path returned if written → Task 3 ✅
- config block (`enabled/binaryPath/permissionMode/daemonAutoStart/screenshotDir`) → Tasks 1-2 ✅
- registration behind `should_include("cua")` + `config.cua.enabled` → Task 4 ✅
- skill pack install documented → Task 5 ✅
- argv-only invocation, bounded output, `Error:` prefixes → Task 3 ✅
- vision workstream explicitly deferred → no task here (documented in spec) ✅
