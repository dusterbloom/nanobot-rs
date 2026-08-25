// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(clippy::shadow_reuse)]
//! Cron tool for scheduling reminders and tasks.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use super::base::{require_param, PermissionLevel, Tool, ToolContext, ToolResult};
use crate::cron::executor::initial_next_run;
use crate::cron::service::CronService;
use crate::cron::types::CronSchedule;
use crate::errors::ToolError;

/// Tool to schedule reminders and recurring tasks.
pub struct CronScheduleTool {
    cron_service: Arc<CronService>,
    channel: Arc<Mutex<String>>,
    chat_id: Arc<Mutex<String>>,
}

impl CronScheduleTool {
    /// Create a new cron schedule tool.
    pub fn new(cron_service: Arc<CronService>) -> Self {
        Self {
            cron_service,
            channel: Arc::new(Mutex::new(String::new())),
            chat_id: Arc::new(Mutex::new(String::new())),
        }
    }

    /// Set the current session context for delivery.
    pub async fn set_context(&self, channel: &str, chat_id: &str) {
        *self.channel.lock().await = channel.to_string();
        *self.chat_id.lock().await = chat_id.to_string();
    }

    /// Handle the "add" action: persist a real job through the shared service.
    async fn add_job(
        &self,
        message: &str,
        every_seconds: Option<i64>,
        cron_expr: Option<&str>,
    ) -> ToolResult {
        if message.is_empty() {
            return Err(ToolError::InvalidArgs {
                message: "message is required for add".to_string(),
            });
        }

        let channel = self.channel.lock().await.clone();
        let chat_id = self.chat_id.lock().await.clone();

        if channel.is_empty() || chat_id.is_empty() {
            return Err(ToolError::Execution {
                message: "no session context (channel/chat_id)".to_string(),
            });
        }

        let Some(schedule) = build_schedule(every_seconds, cron_expr) else {
            return Err(ToolError::InvalidArgs {
                message: "either every_seconds or cron_expr is required".to_string(),
            });
        };

        // Validate before persisting: an unschedulable job (bad cron expr,
        // non-positive interval) would sit in the store and never fire.
        let now_ms = chrono::Local::now().timestamp_millis();
        let Some(next_run_ms) = initial_next_run(&schedule, now_ms) else {
            return Err(ToolError::InvalidArgs {
                message: "invalid schedule (check cron expression / interval)".to_string(),
            });
        };

        // Truncate name to 30 chars.
        let name: String = message.chars().take(30).collect();

        // Deliver the fired reminder back to the chat that scheduled it.
        let job = self.cron_service.add_job(
            &name,
            schedule,
            message,
            true,
            Some(&channel),
            Some(&chat_id),
            false,
        );

        Ok(format!(
            "Scheduled '{}' (id: {}, next run: {})",
            job.name,
            job.id,
            format_local(next_run_ms)
        )
        .into())
    }

    /// Handle the "list" action.
    async fn list_jobs(&self) -> ToolResult {
        let jobs = self.cron_service.list_jobs(false);
        if jobs.is_empty() {
            return Ok("No scheduled jobs.".into());
        }
        let lines: Vec<String> = jobs
            .iter()
            .map(|j| format!("- {} (id: {}, {})", j.name, j.id, j.schedule.kind))
            .collect();
        Ok(format!("Scheduled jobs:\n{}", lines.join("\n")).into())
    }

    /// Handle the "remove" action.
    async fn remove_job(&self, job_id: Option<&str>) -> ToolResult {
        let job_id = match job_id {
            Some(id) if !id.is_empty() => id,
            _ => {
                return Err(ToolError::InvalidArgs {
                    message: "job_id is required for remove".to_string(),
                })
            }
        };
        if self.cron_service.remove_job(job_id) {
            Ok(format!("Removed job {}", job_id).into())
        } else {
            // Legacy quirk preserved: no "Error:" prefix → success channel.
            Ok(format!("Job {} not found", job_id).into())
        }
    }
}

/// Map tool parameters onto a schedule; `None` when neither is given.
fn build_schedule(every_seconds: Option<i64>, cron_expr: Option<&str>) -> Option<CronSchedule> {
    match (every_seconds, cron_expr) {
        (Some(secs), _) => Some(CronSchedule {
            kind: "every".to_string(),
            every_ms: Some(secs.saturating_mul(1000)),
            ..Default::default()
        }),
        (None, Some(expr)) if !expr.is_empty() => Some(CronSchedule {
            kind: "cron".to_string(),
            expr: Some(expr.to_string()),
            ..Default::default()
        }),
        _ => None,
    }
}

/// Human-readable local timestamp for the model to relay.
fn format_local(ms: i64) -> String {
    chrono::DateTime::from_timestamp_millis(ms)
        .map(|dt| {
            dt.with_timezone(&chrono::Local)
                .format("%Y-%m-%d %H:%M")
                .to_string()
        })
        .unwrap_or_else(|| format!("{} ms", ms))
}

#[async_trait]
impl Tool for CronScheduleTool {
    fn name(&self) -> &str {
        "cron"
    }

    fn permission(&self) -> PermissionLevel {
        PermissionLevel::System
    }

    fn description(&self) -> &str {
        "Schedule reminders and recurring tasks. Actions: add, list, remove."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "remove"],
                    "description": "Action to perform"
                },
                "message": {
                    "type": "string",
                    "description": "Reminder message (for add)"
                },
                "every_seconds": {
                    "type": "integer",
                    "description": "Interval in seconds (for recurring tasks)"
                },
                "cron_expr": {
                    "type": "string",
                    "description": "Cron expression like '0 9 * * *' (for scheduled tasks)"
                },
                "job_id": {
                    "type": "string",
                    "description": "Job ID (for remove)"
                }
            },
            "required": ["action"]
        })
    }

    async fn execute(
        &self,
        params: HashMap<String, serde_json::Value>,
        _ctx: &ToolContext,
    ) -> ToolResult {
        let action = require_param!(params, "action");

        match action {
            "add" => {
                let message = params.get("message").and_then(|v| v.as_str()).unwrap_or("");
                let every_seconds = params.get("every_seconds").and_then(|v| v.as_i64());
                let cron_expr = params.get("cron_expr").and_then(|v| v.as_str());
                self.add_job(message, every_seconds, cron_expr).await
            }
            "list" => self.list_jobs().await,
            "remove" => {
                let job_id = params.get("job_id").and_then(|v| v.as_str());
                self.remove_job(job_id).await
            }
            // Legacy quirk preserved: no "Error:" prefix → success channel.
            other => Ok(format!("Unknown action: {}", other).into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::NamedTempFile;

    fn temp_tool() -> (CronScheduleTool, Arc<CronService>, NamedTempFile) {
        let tmp = NamedTempFile::new().expect("temp file");
        std::fs::remove_file(tmp.path()).ok();
        let service = Arc::new(CronService::new(tmp.path().to_path_buf()));
        (CronScheduleTool::new(service.clone()), service, tmp)
    }

    fn params(pairs: &[(&str, serde_json::Value)]) -> HashMap<String, serde_json::Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[tokio::test]
    async fn test_add_interval_job_persists_through_service() {
        let (tool, service, _tmp) = temp_tool();
        tool.set_context("telegram", "12345").await;

        let result = crate::agent::tools::base::render_result(
            tool.execute(
                params(&[
                    ("action", json!("add")),
                    ("message", json!("water break")),
                    ("every_seconds", json!(3600)),
                ]),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );

        let jobs = service.list_jobs(true);
        assert_eq!(jobs.len(), 1, "tool add must persist a real job");
        let job = &jobs[0];
        assert_eq!(job.schedule.kind, "every");
        assert_eq!(job.schedule.every_ms, Some(3_600_000));
        assert_eq!(job.payload.kind, "agent_turn");
        assert_eq!(job.payload.message, "water break");
        // Reminder is delivered back to the chat that scheduled it.
        assert!(job.payload.deliver);
        assert_eq!(job.payload.channel.as_deref(), Some("telegram"));
        assert_eq!(job.payload.to.as_deref(), Some("12345"));
        // The model needs id + next-run time to report back to the user.
        assert!(
            result.contains(&job.id),
            "result must contain job id: {result}"
        );
        assert!(
            result.contains("next run"),
            "result must report next run: {result}"
        );
        assert!(
            !result.contains("use CLI") && !result.contains("nanobot cron"),
            "placeholder text must be gone: {result}"
        );
    }

    #[tokio::test]
    async fn test_add_cron_expr_job_persists() {
        let (tool, service, _tmp) = temp_tool();
        tool.set_context("whatsapp", "+491234").await;

        let result = crate::agent::tools::base::render_result(
            tool.execute(
                params(&[
                    ("action", json!("add")),
                    ("message", json!("morning digest")),
                    ("cron_expr", json!("0 9 * * *")),
                ]),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );

        let jobs = service.list_jobs(true);
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].schedule.kind, "cron");
        assert_eq!(jobs[0].schedule.expr.as_deref(), Some("0 9 * * *"));
        assert!(result.contains(&jobs[0].id), "got: {result}");
        assert!(result.contains("next run"), "got: {result}");
    }

    #[tokio::test]
    async fn test_add_invalid_cron_expr_rejected_without_persisting() {
        let (tool, service, _tmp) = temp_tool();
        tool.set_context("telegram", "1").await;

        let result = crate::agent::tools::base::render_result(
            tool.execute(
                params(&[
                    ("action", json!("add")),
                    ("message", json!("bad")),
                    ("cron_expr", json!("not a cron")),
                ]),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );

        assert!(result.starts_with("Error"), "got: {result}");
        assert!(
            service.list_jobs(true).is_empty(),
            "invalid job must not persist"
        );
    }

    #[tokio::test]
    async fn test_remove_job_actually_removes() {
        let (tool, service, _tmp) = temp_tool();
        let job = service.add_job(
            "temp",
            crate::cron::types::CronSchedule {
                kind: "every".to_string(),
                every_ms: Some(60_000),
                ..Default::default()
            },
            "msg",
            false,
            None,
            None,
            false,
        );

        let result = crate::agent::tools::base::render_result(
            tool.execute(
                params(&[("action", json!("remove")), ("job_id", json!(job.id))]),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );

        assert!(result.contains("Removed"), "got: {result}");
        assert!(service.list_jobs(true).is_empty());

        // Unknown id: clear feedback, no panic.
        let missing = crate::agent::tools::base::render_result(
            tool.execute(
                params(&[("action", json!("remove")), ("job_id", json!("nope1234"))]),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );
        assert!(missing.contains("not found"), "got: {missing}");
    }
}
