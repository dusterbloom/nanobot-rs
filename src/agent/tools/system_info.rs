// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(clippy::as_conversions)]
//! System information tool: quick runtime, process, and disk snapshots.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::process::Command;

use super::base::{Tool, ToolContext, ToolResult};
use crate::errors::ToolError;

/// Tool to inspect process/disk/runtime state without hand-written shell calls.
pub struct SystemInfoTool;

#[async_trait]
impl Tool for SystemInfoTool {
    fn name(&self) -> &str {
        "system_info"
    }

    fn description(&self) -> &str {
        "Inspect local runtime state: overview, running processes, and disk usage. Use this instead of repeated exec calls for common environment checks."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["overview", "processes", "disk", "all"],
                    "description": "Information to return. Default: overview"
                },
                "path": {
                    "type": "string",
                    "description": "Path for disk usage checks. Default: current directory"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum process rows for action=processes/all. Default: 20, max: 100"
                }
            }
        })
    }

    async fn execute(&self, params: HashMap<String, Value>, _ctx: &ToolContext) -> ToolResult {
        let action = params
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("overview");
        if !matches!(action, "overview" | "processes" | "disk" | "all") {
            return Err(ToolError::InvalidArgs {
                message: "'action' must be one of: overview, processes, disk, all".to_string(),
            });
        }

        let limit = params
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| (v as usize).clamp(1, 100))
            .unwrap_or(20);
        let path = params.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        match action {
            "overview" => Ok(overview().into()),
            "processes" => Ok(process_snapshot(limit).await.into()),
            "disk" => disk_snapshot(path).await,
            "all" => {
                let mut out = overview();
                out.push_str("\n\n## Processes\n");
                out.push_str(&process_snapshot(limit).await);
                out.push_str("\n\n## Disk\n");
                // Legacy embedded a rendered disk error inline in the
                // success report (the funnel only saw the final string);
                // preserve those bytes.
                out.push_str(&match disk_snapshot(path).await {
                    Ok(o) => o.text,
                    Err(e) => e.render(),
                });
                Ok(out.into())
            }
            // Defensive: action was validated above; unknown values are a
            // programming error, report rather than panic.
            _ => Err(ToolError::InvalidArgs {
                message: "'action' must be one of: overview, processes, disk, all".to_string(),
            }),
        }
    }
}

fn overview() -> String {
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "(unknown)".to_string());
    let home = dirs::home_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "(unknown)".to_string());
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "(unset)".to_string());
    let path = std::env::var("PATH").unwrap_or_else(|_| "(unset)".to_string());
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    format!(
        "OS: {}\nArch: {}\nPID: {}\nCWD: {}\nHome: {}\nTemp: {}\nShell: {}\nCPUs: {}\nPATH: {}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::process::id(),
        cwd,
        home,
        std::env::temp_dir().display(),
        shell,
        cpus,
        path
    )
}

async fn process_snapshot(limit: usize) -> String {
    #[cfg(unix)]
    {
        let output = run_command(
            "ps",
            &["-axo", "pid,ppid,stat,%cpu,%mem,comm", "-r"],
            Duration::from_secs(5),
        )
        .await;
        return match output {
            Ok(text) => text.lines().take(limit + 1).collect::<Vec<_>>().join("\n"),
            Err(e) => format!(
                "Process listing unavailable: {}\nCurrent process:\nPID: {}\nCWD: {}",
                e,
                std::process::id(),
                std::env::current_dir()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|_| "(unknown)".to_string())
            ),
        };
    }

    #[cfg(not(unix))]
    {
        let _ = limit;
        "Process listing is not available on this platform.".to_string()
    }
}

async fn disk_snapshot(path: &str) -> ToolResult {
    let target = expand_path(path);
    if !target.exists() {
        return Err(ToolError::NotFound(format!(
            "Path not found for disk check: {}",
            path
        )));
    }

    #[cfg(unix)]
    {
        let path_arg = target.to_string_lossy().to_string();
        let output = run_command("df", &["-k", &path_arg], Duration::from_secs(5)).await;
        // Legacy returned the informational fallback as the success channel
        // (only a leading "Error:" prefix is an error) — keep those bytes.
        return Ok(output
            .unwrap_or_else(|e| format!("Error reading disk usage: {}", e))
            .into());
    }

    #[cfg(not(unix))]
    {
        Ok(format!(
            "Disk usage is not available on this platform. Path exists: {}",
            target.display()
        )
        .into())
    }
}

async fn run_command(command: &str, args: &[&str], timeout: Duration) -> Result<String, String> {
    let output = tokio::time::timeout(
        timeout,
        Command::new(command)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .output(),
    )
    .await
    .map_err(|_| format!("{} timed out", command))?
    .map_err(|e| e.to_string())?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if output.status.success() {
        Ok(stdout)
    } else if stderr.trim().is_empty() {
        Err(stdout.trim().to_string())
    } else {
        Err(stderr.trim().to_string())
    }
}

fn expand_path(path: &str) -> PathBuf {
    if path.starts_with('~') {
        return crate::utils::helpers::expand_tilde(path);
    }
    let p = PathBuf::from(path);
    if p.is_absolute() {
        p
    } else {
        std::env::current_dir().map(|cwd| cwd.join(&p)).unwrap_or(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name() {
        let tool = SystemInfoTool;
        assert_eq!(tool.name(), "system_info");
    }

    #[test]
    fn test_parameters_schema() {
        let tool = SystemInfoTool;
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["action"].is_object());
    }

    #[tokio::test]
    async fn test_overview_contains_runtime_fields() {
        let tool = SystemInfoTool;
        let mut params = HashMap::new();
        params.insert("action".to_string(), json!("overview"));
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(result.contains("OS:"), "got: {}", result);
        assert!(result.contains("CWD:"), "got: {}", result);
    }

    #[tokio::test]
    async fn test_processes_returns_snapshot_or_fallback() {
        let tool = SystemInfoTool;
        let mut params = HashMap::new();
        params.insert("action".to_string(), json!("processes"));
        params.insert("limit".to_string(), json!(3));
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(
            result.contains("PID") || result.contains("pid"),
            "got: {}",
            result
        );
    }
}
