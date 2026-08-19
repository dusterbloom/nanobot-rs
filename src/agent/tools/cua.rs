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

use super::base::{PermissionLevel, Tool, ToolConcurrency, ToolContext};
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

    /// Whether the configured binary resolves (absolute path exists, or on PATH).
    fn binary_present(&self) -> bool {
        let p = Path::new(&self.binary);
        if p.is_absolute() {
            return p.is_file();
        }
        std::env::var_os("PATH").is_some_and(|paths| {
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
    /// then poll `status` up to `DAEMON_READY_POLLS` times.
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
        let Some((first, rest)) = args.split_first() else {
            return Err("Error: cua-driver daemon launch arguments are empty.".to_string());
        };
        let spawn = std::process::Command::new(first)
            .args(rest)
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
    fn name(&self) -> &'static str {
        "cua"
    }

    fn description(&self) -> &'static str {
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

        let args_json = params.get("args").map_or_else(
            || "{}".to_string(),
            |v| serde_json::to_string(v).unwrap_or_else(|_| "{}".to_string()),
        );

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
        let (raw_out, raw_err, ok) = self
            .run_driver(&["call", tool, &args_json, shot_arg, &shot_str])
            .await;
        let mut out = raw_out;
        if !ok {
            if raw_err.contains("daemon is not running") {
                return format!(
                    "Error: cua-driver daemon is not running. Start it with: {}",
                    self.daemon_launch_args().join(" ")
                );
            }
            let detail = if raw_err.is_empty() { out.clone() } else { raw_err };
            return format!("Error: cua-driver call '{tool}' failed: {detail}");
        }
        if shot_path.is_file() {
            out.push_str("\n\nScreenshot saved: ");
            out.push_str(&shot_str);
        }
        out
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
        let dir = std::env::temp_dir().join(format!("cua-test-{}-missing_tool", std::process::id()));
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
        let dir = std::env::temp_dir().join(format!("cua-test-{}-call_passes", std::process::id()));
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
        let dir = std::env::temp_dir().join(format!("cua-test-{}-daemon_down", std::process::id()));
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
        let dir = std::env::temp_dir().join(format!("cua-test-{}-screenshot", std::process::id()));
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
        let dir = std::env::temp_dir().join(format!("cua-test-{}-binary_present", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let bin = make_shim(&dir);
        let tool = CuaTool::with_binary(bin.to_str().unwrap());
        assert!(tool.binary_present());
        let missing = CuaTool::with_binary("/nonexistent/cua-driver");
        assert!(!missing.binary_present());
        fs::remove_dir_all(&dir).ok();
    }
}
