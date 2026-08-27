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

use super::base::{PermissionLevel, Tool, ToolConcurrency, ToolContext, ToolResult};
use crate::config::schema::CuaToolConfig;
use crate::errors::ToolError;

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
    /// `exec_timeout` seconds bounds every driver subprocess; 0 falls back to
    /// the default (a zero timeout would make every call abort instantly).
    pub fn new(config: &CuaToolConfig, workspace: &Path, exec_timeout: u64) -> Self {
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
            timeout: Duration::from_secs(if exec_timeout == 0 {
                DEFAULT_TIMEOUT_SECS
            } else {
                exec_timeout
            }),
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

    /// Test seam: like `with_binary`, but with daemon auto-start enabled so
    /// the tool launches the daemon (shim `serve` arm) and polls until ready.
    /// macOS is excluded: the launch argv there is `open -a CuaDriver`, which
    /// would launch the real app from a test.
    #[cfg(test)]
    #[cfg(not(target_os = "macos"))]
    fn with_binary_autostart(binary: &str) -> Self {
        Self {
            binary: binary.to_string(),
            permission_mode: "standard".to_string(),
            daemon_auto_start: true,
            screenshot_dir: std::env::temp_dir().join("cua-test-shots"),
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
        }
    }

    /// Whether the configured binary resolves (absolute path exists, or on
    /// PATH; Windows also probes `cua-driver.exe`).
    fn binary_present(&self) -> bool {
        let p = Path::new(&self.binary);
        if p.is_absolute() {
            return p.is_file();
        }
        std::env::var_os("PATH").is_some_and(|paths| {
            std::env::split_paths(&paths).any(|d| {
                d.join(&self.binary).is_file()
                    || (cfg!(windows) && d.join(format!("{}.exe", self.binary)).is_file())
            })
        })
    }

    /// Run a driver subcommand; returns (stdout, stderr, success).
    async fn run_driver(&self, args: &[&str]) -> (String, String, bool) {
        let result =
            tokio::time::timeout(self.timeout, Command::new(&self.binary).args(args).output())
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
                format!(
                    "cua-driver call timed out after {}s",
                    self.timeout.as_secs()
                ),
                false,
            ),
        }
    }

    /// `cua-driver status` exits 0 when the daemon is listening.
    async fn daemon_running(&self) -> bool {
        let (_out, _err, ok) = self.run_driver(&["status"]).await;
        ok
    }

    /// Run `cua-driver call`, racing the timeout against cooperative
    /// cancellation. `kill_on_drop` aborts the child on the timeout/cancel
    /// path (same pattern as `code_execution.rs`). `None` means cancelled.
    async fn run_call(&self, args: &[&str], ctx: &ToolContext) -> Option<(String, String, bool)> {
        let child = match Command::new(&self.binary)
            .args(args)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                return Some((
                    String::new(),
                    format!("failed to run cua-driver: {e}"),
                    false,
                ))
            }
        };
        tokio::select! {
            output = tokio::time::timeout(self.timeout, child.wait_with_output()) => {
                match output {
                    Ok(Ok(out)) => Some((
                        String::from_utf8_lossy(&out.stdout).to_string(),
                        String::from_utf8_lossy(&out.stderr).to_string(),
                        out.status.success(),
                    )),
                    Ok(Err(e)) => Some((
                        String::new(),
                        format!("failed to run cua-driver: {e}"),
                        false,
                    )),
                    Err(_) => Some((
                        String::new(),
                        format!(
                            "cua-driver call timed out after {}s",
                            self.timeout.as_secs()
                        ),
                        false,
                    )),
                }
            }
            () = ctx.cancellation_token().cancelled() => None,
        }
    }

    /// Launch command argv, pure for testability. macOS uses `open -a
    /// CuaDriver` so Accessibility/Screen Recording grants keep the app
    /// identity (raw `serve` outside the app is unsupported on macOS).
    fn daemon_launch_args(&self) -> Vec<String> {
        daemon_launch_args(
            &self.binary,
            &self.permission_mode,
            cfg!(target_os = "macos"),
        )
    }

    /// Ensure the daemon is up before calling. Auto-start is bounded: launch,
    /// then poll `status` up to `DAEMON_READY_POLLS` times. Cooperative
    /// cancellation is honored before and after the poll loop (the `status`
    /// probes themselves are short — no per-iteration check).
    async fn ensure_daemon(&self, ctx: &ToolContext) -> Result<(), String> {
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
        // Cancellation check before the poll loop: a cancelled call should not
        // keep waiting for the daemon to come up.
        if ctx.cancellation_token().is_cancelled() {
            return Err("Error: cua-driver call cancelled".to_string());
        }
        for _ in 0..DAEMON_READY_POLLS {
            tokio::time::sleep(Duration::from_millis(DAEMON_READY_POLL_MS)).await;
            if self.daemon_running().await {
                return Ok(());
            }
        }
        // Cancellation check after the poll loop: prefer the cancellation
        // signal over the generic not-ready error.
        if ctx.cancellation_token().is_cancelled() {
            return Err("Error: cua-driver call cancelled".to_string());
        }
        Err(
            "Error: cua-driver daemon did not become ready after launch. Run `cua-driver doctor`."
                .to_string(),
        )
    }

    /// Run `cua-driver list-tools` and return the output (discovery fallback).
    async fn list_tools(&self) -> String {
        let (out, _err, _ok) = self.run_driver(&["list-tools"]).await;
        out
    }

    /// Run one `cua-driver call` with screenshot capture, mapping failures to
    /// user-facing errors: daemon-down gets the launch hint, other failures
    /// append the available-tools list (discovery fallback), cancellation
    /// short-circuits.
    async fn call_tool(&self, tool: &str, args_json: &str, ctx: &ToolContext) -> String {
        // Always pass --screenshot-out-file: the driver only writes the file
        // when the response contains an image block, so it is harmless for
        // non-image tools.
        let call_id = ctx.call_id();
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
            return format!(
                "Error: cannot create screenshot dir {}: {e}",
                self.screenshot_dir.display()
            );
        }
        let Some((raw_out, raw_err, ok)) = self
            .run_call(&["call", tool, args_json, shot_arg, &shot_str], ctx)
            .await
        else {
            return "Error: cua-driver call cancelled".to_string();
        };
        let mut out = raw_out;
        if !ok {
            if raw_err.contains("daemon is not running") {
                return format!(
                    "Error: cua-driver daemon is not running. Start it with: {}",
                    self.daemon_launch_args().join(" ")
                );
            }
            let detail = if raw_err.is_empty() {
                out.clone()
            } else {
                raw_err
            };
            // Unknown/missing tool names surface the tool list so the model
            // learns the surface (discovery fallback).
            let list = self.list_tools().await;
            return format!(
                "Error: cua-driver call '{tool}' failed: {detail}\n\nAvailable cua-driver tools:\n{list}"
            );
        }
        // Compact driver JSON at the source: raw list_apps (29KB) and AX
        // window trees blew the result cap, stashed behind a handle whose
        // excerpt was just "{", forcing an inspect_tool_result round-trip
        // per cua action (ergonomics fix 1). Plain-text driver replies
        // (shims, future shapes) pass through untouched.
        let mut out = compact_driver_output(tool, &out);
        if shot_path.is_file() {
            out.push_str("\n\nScreenshot saved: ");
            out.push_str(&shot_str);
        }
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

// ---------------------------------------------------------------------------
// Driver-output compaction (v0.5 ergonomics fix 1)
// ---------------------------------------------------------------------------

/// Char ceiling for compacted cua output. Comfortably under the tool-result
/// cap so compacted output stays INLINE instead of being stashed behind a
/// handle — the inspect round-trip tax that halved the GUI lease budget
/// (session 20260826_120633_51cbb8: 7 of 12 slots spent on
/// `inspect_tool_result` fetches of `cua` handles).
const COMPACT_MAX_CHARS: usize = 8_000;
/// Per-element char ceiling for one AX element line.
const ELEMENT_LINE_MAX: usize = 160;
/// Max element lines rendered before truncating with a hint.
const ELEMENT_LINES_MAX: usize = 80;

/// Compact a cua-driver JSON payload into a small model-ready projection.
/// Pure; non-JSON input passes through unchanged (the stash pipeline still
/// bounds it).
///
/// Known shapes (verified against driver output in the session DB):
/// - `list_apps`: `{"apps": [{name, bundle_id, pid, running, active, ...}]}`
///   → one line per app.
/// - window/desktop state: `{degraded, degraded_reason, element_count,
///   elements: [...], escalation: {...}, windows: [...]}` → diagnostics kept,
///   elements rendered as bounded role/name/value lines.
/// Anything else JSON-shaped keeps scalar diagnostics + any recognized
/// array, dropping opaque blobs.
fn compact_driver_output(tool: &str, raw: &str) -> String {
    let Ok(value) = serde_json::from_str::<Value>(raw) else {
        return raw.to_string();
    };
    let Some(obj) = value.as_object() else {
        // Non-object JSON (rare): leave to the normal bounding pipeline.
        return raw.to_string();
    };

    let mut out = String::with_capacity(1024);
    out.push_str(&format!(
        "[cua:{tool} compacted from {} chars]\n",
        raw.chars().count()
    ));

    // Scalar diagnostics worth teaching the model (truncated hard).
    for key in ["degraded_reason", "escalation", "error", "message"] {
        if let Some(text) = obj.get(key).and_then(Value::as_str) {
            let t: String = text.chars().take(400).collect();
            out.push_str(&format!("{key}: {t}\n"));
        }
    }
    // The driver also sends escalation as an object ({reason: ...}) on
    // degraded snapshots — surface its reason the same way.
    if let Some(reason) = obj
        .get("escalation")
        .and_then(Value::as_object)
        .and_then(|esc| esc.get("reason"))
        .and_then(Value::as_str)
    {
        let t: String = reason.chars().take(400).collect();
        out.push_str(&format!("escalation: {t}\n"));
    }
    for key in [
        "degraded",
        "element_count",
        "elements_complete",
        "current_space_id",
    ] {
        if let Some(v) = obj.get(key) {
            if !v.is_object() && !v.is_array() {
                out.push_str(&format!("{key}: {v}\n"));
            }
        }
    }
    // The driver's _note teaches bounding args — keep one compressed hint.
    if obj.contains_key("elements") {
        out.push_str("note: prefer elements; pass max_elements/max_depth to bound large AX trees\n");
    }

    // Known payload arrays.
    if let Some(apps) = obj.get("apps").and_then(Value::as_array) {
        out.push_str(&format!("apps ({}):\n", apps.len()));
        for app in apps.iter().take(60) {
            let name = str_field(app, &["name", "app_name"]).unwrap_or_else(|| "?".to_string());
            let bid = str_field(app, &["bundle_id"]).unwrap_or_default();
            let pid = app.get("pid").map(|v| v.to_string()).unwrap_or_default();
            let state = if app.get("active").and_then(Value::as_bool) == Some(true) {
                "active"
            } else if app.get("running").and_then(Value::as_bool) == Some(true) {
                "running"
            } else {
                "not-running"
            };
            let windows = app
                .get("windows")
                .and_then(Value::as_array)
                .map(|w| w.len())
                .unwrap_or(0);
            out.push_str(&format!(
                "- {name} ({bid}, pid {pid}, {state}, {windows} window(s))\n"
            ));
        }
        if apps.len() > 60 {
            out.push_str(&format!("… {} more apps not shown\n", apps.len() - 60));
        }
    }

    if let Some(windows) = obj.get("windows").and_then(Value::as_array) {
        out.push_str(&format!("windows ({}):\n", windows.len()));
        for w in windows.iter().take(40) {
            let app = str_field(w, &["app_name", "name", "title"]).unwrap_or_else(|| "?".to_string());
            let on_screen = w.get("is_on_screen").and_then(Value::as_bool);
            let id = w.get("window_id").map(|v| v.to_string()).unwrap_or_default();
            let space = w.get("space_id").map(|v| v.to_string()).unwrap_or_default();
            out.push_str(&format!(
                "- {app} (id {id}, space {space}, {})\n",
                if on_screen == Some(true) {
                    "on-screen"
                } else {
                    "off-screen"
                }
            ));
        }
        if windows.len() > 40 {
            out.push_str(&format!(
                "… {} more windows not shown\n",
                windows.len() - 40
            ));
        }
    }

    if let Some(elements) = obj.get("elements").and_then(Value::as_array) {
        if elements.is_empty() {
            out.push_str("elements: none (tree empty — see degraded_reason above)\n");
        } else {
            out.push_str("elements:\n");
            let mut lines = 0;
            for el in elements {
                if lines >= ELEMENT_LINES_MAX {
                    out.push_str(&format!(
                        "… {} more elements not shown (pass max_elements to narrow)\n",
                        elements.len() - lines
                    ));
                    break;
                }
                if let Some(line) = element_line(el) {
                    out.push_str(&line);
                    out.push('\n');
                    lines += 1;
                }
            }
        }
    }

    // Hard ceiling with an actionable hint.
    if out.chars().count() > COMPACT_MAX_CHARS {
        let cut: String = out.chars().take(COMPACT_MAX_CHARS).collect();
        return format!(
            "{cut}\n[cua output truncated at {COMPACT_MAX_CHARS} chars — narrow the call with max_elements/max_depth or a more specific tool]"
        );
    }
    out
}

/// First present string field among candidates.
fn str_field(obj: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|k| obj.get(*k).and_then(Value::as_str))
        .map(|s| s.chars().take(60).collect())
}

/// One compact line for an AX element: identity fields + child count.
fn element_line(el: &Value) -> Option<String> {
    let obj = el.as_object()?;
    let role = str_field(el, &["role", "role_description"]).unwrap_or_default();
    let name = str_field(el, &["name", "label", "title", "ax_name"]);
    let text = str_field(el, &["value", "text", "ax_value"]);
    let children = obj
        .get("children")
        .and_then(Value::as_array)
        .map(|c| format!(" +{}ch", c.len()))
        .unwrap_or_default();
    let mut line = format!("- {role}");
    if let Some(n) = &name {
        line.push_str(&format!(" \"{n}\""));
    }
    if let Some(t) = &text {
        line.push_str(&format!(" = \"{t}\""));
    }
    line.push_str(&children);
    Some(line.chars().take(ELEMENT_LINE_MAX).collect())
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

    async fn execute(&self, params: HashMap<String, Value>, ctx: &ToolContext) -> ToolResult {
        let out = self.run(params, ctx).await;
        // One boundary: driver-protocol strings split the legacy error
        // channel here (python_kernel pattern). Nothing downstream consumes
        // per-kind telemetry for these, so Execution preserves bytes.
        match out.strip_prefix("Error:").map(str::trim) {
            Some(err) => Err(ToolError::Execution {
                message: err.to_string(),
            }),
            None => Ok(out.into()),
        }
    }
}

impl CuaTool {
    /// Legacy String body, called only by [`CuaTool::execute_typed`].
    async fn run(&self, params: HashMap<String, Value>, ctx: &ToolContext) -> String {
        // Honor cooperative cancellation before doing any driver work.
        if ctx.cancellation_token().is_cancelled() {
            return "Error: cua-driver call cancelled".to_string();
        }
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

        if let Err(e) = self.ensure_daemon(ctx).await {
            return e;
        }

        self.call_tool(tool, &args_json, ctx).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;


    // -------------------------------------------------------------------
    // Driver-output compaction (ergonomics fix 1)
    // -------------------------------------------------------------------

    #[test]
    fn compact_list_apps_shrinks_29k_json_to_per_app_lines() {
        // Shape verified against the real driver output in session
        // 20260826_120633_51cbb8: per-app objects with windows arrays.
        let mut apps = Vec::new();
        for i in 0..40 {
            apps.push(serde_json::json!({
                "active": i == 0,
                "bundle_id": format!("com.example.app{i}"),
                "kind": "desktop",
                "last_used": "2026-07-17T05:14:16Z",
                "launch_path": format!("/Applications/App{i}.app"),
                "name": format!("App {i}"),
                "pid": 600 + i,
                "running": true,
                "windows": [serde_json::json!({"bounds": {"x": 0.0, "y": 0.0, "width": 800.0, "height": 600.0}, "is_on_screen": i % 2 == 0, "window_id": 1000 + i})]
            }));
        }
        let raw = serde_json::to_string_pretty(&serde_json::json!({"apps": apps})).unwrap();
        assert!(raw.chars().count() > 10_000, "fixture must be the big shape");

        let out = compact_driver_output("list_apps", &raw);
        assert!(out.contains("[cua:list_apps compacted from"), "{out}");
        assert!(out.contains("apps (40):"), "{out}");
        assert!(out.contains("- App 0 (com.example.app0, pid 600, active, 1 window(s))"), "{out}");
        assert!(out.contains("- App 1 (com.example.app1, pid 601, running, 1 window(s))"), "{out}");
        assert!(out.chars().count() < 3_000, "must stay inline-sized, got {}", out.chars().count());
        // No handle-bait: the raw JSON blob is gone.
        assert!(!out.contains("\"launch_path\""), "{out}");
    }

    #[test]
    fn compact_window_state_keeps_diagnostics_and_bounded_elements() {
        // Shape from the same session: degraded snapshot with escalation and
        // routes; elements may be empty or a tree.
        let raw = serde_json::to_string_pretty(&serde_json::json!({
            "_note": "Prefer elements — tree_markdown will continue to work. Issue #22865: use max_elements / max_depth to bound the AX walk.",
            "background_input": {"exact_window": {"pid": 41460, "status": "ax_unresolved", "window_id": 16964}},
            "degraded": true,
            "degraded_reason": "ax_window_unresolved: window_id 16964 exists but none of the AXWindow elements report this CGWindowID.",
            "element_count": 0,
            "elements": [],
            "elements_complete": false,
            "escalation": {"reason": "observation-only: re-snapshot after the app settles, or act with delivery_mode:\"foreground\"."}
        }))
        .unwrap();

        let out = compact_driver_output("get_window_state", &raw);
        assert!(out.contains("degraded: true"), "{out}");
        assert!(out.contains("degraded_reason: ax_window_unresolved"), "{out}");
        assert!(out.contains("escalation: observation-only"), "{out}");
        assert!(out.contains("elements: none"), "{out}");
        assert!(!out.contains("background_input"), "opaque blob dropped: {out}");
    }

    #[test]
    fn compact_elements_renders_identity_lines_and_caps() {
        let elements: Vec<_> = (0..100)
            .map(|i| {
                serde_json::json!({
                    "role": "AXButton",
                    "name": format!("button {i}"),
                    "value": format!("value {i}"),
                    "children": [{"role": "AXStaticText", "name": "child"}]
                })
            })
            .collect();
        let raw = serde_json::to_string(&serde_json::json!({
            "elements": elements, "element_count": 100
        }))
        .unwrap();

        let out = compact_driver_output("snapshot", &raw);
        assert!(out.contains("- AXButton \"button 0\" = \"value 0\" +1ch"), "{out}");
        assert!(out.contains("more elements not shown (pass max_elements"), "{out}");
    }

    #[test]
    fn compact_passes_through_non_json_untouched() {
        let raw = "OK";
        assert_eq!(compact_driver_output("click", raw), "OK");
        let text = "plain driver text\nsecond line";
        assert_eq!(compact_driver_output("type_text", text), text);
        // Non-object JSON passes through too.
        assert_eq!(compact_driver_output("scroll", "[1,2,3]"), "[1,2,3]");
    }

    #[test]
    fn compact_hard_ceiling_with_actionable_hint() {
        // Long name AND value so each element line approaches the 160-char
        // per-line cap; 80 lines then exceed the 8,000-char hard ceiling and
        // exercise the truncation hint. (Short values render ~80-char lines
        // and never reach the ceiling.)
        let elements: Vec<_> = (0..600)
            .map(|i| serde_json::json!({"role": "AXRow", "name": "x".repeat(150), "value": format!("row {i} {}", "y".repeat(150))}))
            .collect();
        let raw = serde_json::to_string(&serde_json::json!({"elements": elements})).unwrap();
        let out = compact_driver_output("get_desktop_state", &raw);
        assert!(out.chars().count() < COMPACT_MAX_CHARS + 200, "ceiling honored");
        assert!(out.contains("truncated at"), "{out}");
    }

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
  if [ "$2" = "nonexistent_tool" ]; then echo "unknown tool: $2" >&2; exit 1; fi
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
        assert_eq!(params["required"], serde_json::json!(["tool"]));
        assert_eq!(tool.permission(), PermissionLevel::System);
        assert_eq!(tool.concurrency(), ToolConcurrency::Sequential);
    }

    #[tokio::test]
    async fn test_missing_tool_param_lists_tools() {
        let dir =
            std::env::temp_dir().join(format!("cua-test-{}-missing_tool", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let bin = make_shim(&dir);
        let tool = CuaTool::with_binary(bin.to_str().unwrap());
        let result = crate::agent::tools::base::render_result(
            tool.execute(
                HashMap::new(),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );
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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(result.starts_with("Error:"), "got: {result}");
        assert!(
            result.contains("cua-driver") || result.contains("CuaDriver"),
            "expected launch hint, got: {result}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    #[cfg(not(target_os = "macos"))]
    async fn test_autostart_launches_daemon_and_call_succeeds() {
        // macOS excluded: the launch argv there is `open -a CuaDriver`, which
        // would launch the real app. On Linux/Windows the tool runs
        // `[binary, serve, --permission-mode, standard]` against the shim.
        let dir = std::env::temp_dir().join(format!("cua-test-{}-autostart", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let bin = make_shim(&dir); // no .running marker initially → status exit 1
        let tool = CuaTool::with_binary_autostart(bin.to_str().unwrap());
        let mut params = HashMap::new();
        params.insert("tool".to_string(), Value::String("click".to_string()));
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(
            result.contains("OK"),
            "call should succeed after the tool auto-starts the daemon, got: {result}"
        );
        assert!(
            dir.join(".running").is_file(),
            "tool should have launched the daemon (shim serve arm creates .running)"
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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(result.contains("OK"), "got: {result}");
        assert!(
            result.contains("Screenshot saved"),
            "expected screenshot path in result, got: {result}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_unknown_tool_appends_available_tools() {
        let dir =
            std::env::temp_dir().join(format!("cua-test-{}-unknown_tool", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let bin = make_shim(&dir); // call with nonexistent_tool exits 1
        fs::write(dir.join(".running"), "").unwrap(); // daemon up
        let tool = CuaTool::with_binary(bin.to_str().unwrap());
        let mut params = HashMap::new();
        params.insert(
            "tool".to_string(),
            Value::String("nonexistent_tool".to_string()),
        );
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(
            result.contains("cua-driver call 'nonexistent_tool' failed"),
            "got: {result}"
        );
        assert!(
            result.contains("Available cua-driver tools"),
            "expected discovery fallback, got: {result}"
        );
        assert!(result.contains("list_apps"), "got: {result}");
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

    #[tokio::test]
    async fn test_binary_present_absolute_and_missing() {
        let dir =
            std::env::temp_dir().join(format!("cua-test-{}-binary_present", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let bin = make_shim(&dir);
        let tool = CuaTool::with_binary(bin.to_str().unwrap());
        assert!(tool.binary_present());
        let missing = CuaTool::with_binary("/nonexistent/cua-driver");
        assert!(!missing.binary_present());
        // A missing binary short-circuits before tool-param discovery, so the
        // empty-params call must return the spec'd install hint.
        let result = crate::agent::tools::base::render_result(
            missing
                .execute(
                    HashMap::new(),
                    &crate::agent::tools::base::ToolContext::sandbox(),
                )
                .await,
        );
        assert!(
            result.contains("cua-driver binary not found"),
            "got: {result}"
        );
        assert!(
            result.contains("curl -fsSL https://cua.ai/driver/install.sh"),
            "install hint missing, got: {result}"
        );
        fs::remove_dir_all(&dir).ok();
    }
}
