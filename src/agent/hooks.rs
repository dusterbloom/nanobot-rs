//! PreToolUse / PostToolUse hook system.
//!
//! Hooks are external shell scripts that run before and after tool execution.
//! They receive context via environment variables and can influence execution:
//! - **PreToolUse**: can block a tool call by exiting non-zero
//! - **PostToolUse**: observes the result (exit code is informational only)

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tracing::warn;

/// Which phase of tool execution the hook runs in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum HookPhase {
    PreToolUse,
    PostToolUse,
}

/// Outcome of running a hook script.
#[derive(Debug, Clone)]
pub struct HookResult {
    /// Whether the hook allowed the action to proceed.
    pub allowed: bool,
    /// Combined stdout + stderr from the hook (truncated).
    pub output: String,
}

/// Maximum time a hook script is allowed to run before being killed.
const HOOK_TIMEOUT: Duration = Duration::from_secs(10);

/// Maximum bytes of hook output to capture.
const MAX_OUTPUT_BYTES: usize = 4096;

/// Run a hook script for the given phase.
///
/// Environment variables set for the hook:
/// - `NANOBOT_HOOK_PHASE`: "pre_tool_use" or "post_tool_use"
/// - `NANOBOT_TOOL_NAME`: name of the tool being called
/// - `NANOBOT_TOOL_PARAMS`: JSON-encoded tool parameters
/// - `NANOBOT_TOOL_RESULT` (post only): the tool's output string
/// - `NANOBOT_TOOL_OK` (post only): "true" or "false"
///
/// Returns `None` if the hook script doesn't exist or isn't configured.
pub async fn run_hook(
    script: &Path,
    phase: HookPhase,
    tool_name: &str,
    params: &HashMap<String, serde_json::Value>,
    result: Option<(&str, bool)>,
) -> Option<HookResult> {
    if !script.exists() {
        return None;
    }

    let phase_str = match phase {
        HookPhase::PreToolUse => "pre_tool_use",
        HookPhase::PostToolUse => "post_tool_use",
    };

    let params_json = serde_json::to_string(params).unwrap_or_default();

    let mut cmd = Command::new(script);
    cmd.env("NANOBOT_HOOK_PHASE", phase_str)
        .env("NANOBOT_TOOL_NAME", tool_name)
        .env("NANOBOT_TOOL_PARAMS", &params_json);

    if let Some((output, ok)) = result {
        cmd.env("NANOBOT_TOOL_RESULT", output);
        cmd.env("NANOBOT_TOOL_OK", if ok { "true" } else { "false" });
    }

    let output = match tokio::time::timeout(HOOK_TIMEOUT, cmd.output()).await {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => {
            warn!("Hook {} failed: {}", script.display(), e);
            return Some(HookResult {
                allowed: true, // fail-open: don't block on hook errors
                output: format!("hook error: {}", e),
            });
        }
        Err(_) => {
            warn!(
                "Hook {} timed out after {:?}",
                script.display(),
                HOOK_TIMEOUT
            );
            return Some(HookResult {
                allowed: true, // fail-open on timeout
                output: "hook timed out".to_string(),
            });
        }
    };

    let mut combined = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.is_empty() {
        if !combined.is_empty() {
            combined.push('\n');
        }
        combined.push_str(&stderr);
    }
    combined.truncate(MAX_OUTPUT_BYTES);

    let allowed = match phase {
        HookPhase::PreToolUse => output.status.success(),
        HookPhase::PostToolUse => true, // post hooks are observational
    };

    Some(HookResult {
        allowed,
        output: combined,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_hook_phase_serde() {
        let pre: HookPhase = serde_json::from_str("\"preToolUse\"").unwrap();
        assert_eq!(pre, HookPhase::PreToolUse);
        let post: HookPhase = serde_json::from_str("\"postToolUse\"").unwrap();
        assert_eq!(post, HookPhase::PostToolUse);
    }

    #[tokio::test]
    async fn test_nonexistent_script_returns_none() {
        let result = run_hook(
            Path::new("/nonexistent/hook.sh"),
            HookPhase::PreToolUse,
            "read_file",
            &HashMap::new(),
            None,
        )
        .await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_pre_hook_allows_on_exit_zero() {
        let dir = tempfile::TempDir::new().unwrap();
        let script = dir.path().join("pre_hook.sh");
        std::fs::write(&script, "#!/bin/sh\nexit 0\n").unwrap();
        make_executable(&script);

        let result = run_hook(
            &script,
            HookPhase::PreToolUse,
            "exec",
            &HashMap::new(),
            None,
        )
        .await
        .unwrap();

        assert!(result.allowed);
    }

    #[tokio::test]
    async fn test_pre_hook_blocks_on_nonzero_exit() {
        let dir = tempfile::TempDir::new().unwrap();
        let script = dir.path().join("pre_hook.sh");
        std::fs::write(&script, "#!/bin/sh\necho 'blocked by policy'\nexit 1\n").unwrap();
        make_executable(&script);

        let result = run_hook(
            &script,
            HookPhase::PreToolUse,
            "exec",
            &HashMap::new(),
            None,
        )
        .await
        .unwrap();

        assert!(!result.allowed);
        assert!(result.output.contains("blocked by policy"));
    }

    #[tokio::test]
    async fn test_post_hook_always_allows() {
        let dir = tempfile::TempDir::new().unwrap();
        let script = dir.path().join("post_hook.sh");
        std::fs::write(&script, "#!/bin/sh\nexit 1\n").unwrap();
        make_executable(&script);

        let result = run_hook(
            &script,
            HookPhase::PostToolUse,
            "exec",
            &HashMap::new(),
            Some(("output", true)),
        )
        .await
        .unwrap();

        // Post hooks are observational — always allowed even on non-zero exit.
        assert!(result.allowed);
    }

    #[tokio::test]
    async fn test_hook_receives_env_vars() {
        let dir = tempfile::TempDir::new().unwrap();
        let script = dir.path().join("env_hook.sh");
        std::fs::write(
            &script,
            "#!/bin/sh\necho \"phase=$NANOBOT_HOOK_PHASE tool=$NANOBOT_TOOL_NAME\"\n",
        )
        .unwrap();
        make_executable(&script);

        let result = run_hook(
            &script,
            HookPhase::PreToolUse,
            "write_file",
            &HashMap::new(),
            None,
        )
        .await
        .unwrap();

        assert!(result.output.contains("phase=pre_tool_use"));
        assert!(result.output.contains("tool=write_file"));
    }

    #[tokio::test]
    async fn test_hook_receives_post_result_env() {
        let dir = tempfile::TempDir::new().unwrap();
        let script = dir.path().join("post_env.sh");
        std::fs::write(
            &script,
            "#!/bin/sh\necho \"ok=$NANOBOT_TOOL_OK result=$NANOBOT_TOOL_RESULT\"\n",
        )
        .unwrap();
        make_executable(&script);

        let result = run_hook(
            &script,
            HookPhase::PostToolUse,
            "exec",
            &HashMap::new(),
            Some(("hello world", true)),
        )
        .await
        .unwrap();

        assert!(result.output.contains("ok=true"));
        assert!(result.output.contains("result=hello world"));
    }

    #[tokio::test]
    async fn test_hook_receives_params_json() {
        let dir = tempfile::TempDir::new().unwrap();
        let script = dir.path().join("params_hook.sh");
        std::fs::write(&script, "#!/bin/sh\necho \"params=$NANOBOT_TOOL_PARAMS\"\n").unwrap();
        make_executable(&script);

        let mut params = HashMap::new();
        params.insert(
            "path".to_string(),
            serde_json::Value::String("/tmp/test.txt".to_string()),
        );

        let result = run_hook(&script, HookPhase::PreToolUse, "read_file", &params, None)
            .await
            .unwrap();

        assert!(result.output.contains("/tmp/test.txt"));
    }

    /// Helper to set executable permission on a script.
    fn make_executable(path: &PathBuf) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(path).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(path, perms).unwrap();
        }
    }
}
