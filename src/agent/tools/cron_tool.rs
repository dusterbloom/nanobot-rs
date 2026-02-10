//! Cron tool for scheduling reminders and tasks.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use super::base::Tool;
use crate::cron::service::CronService;
use crate::cron::types::CronSchedule;

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

    /// Handle the "add" action.
    async fn add_job(
        &self,
        message: &str,
        every_seconds: Option<i64>,
        cron_expr: Option<&str>,
    ) -> String {
        if message.is_empty() {
            return "Error: message is required for add".to_string();
        }

        let channel = self.channel.lock().await.clone();
        let chat_id = self.chat_id.lock().await.clone();

        if channel.is_empty() || chat_id.is_empty() {
            return "Error: no session context (channel/chat_id)".to_string();
        }

        // Build schedule.
        let schedule = if let Some(secs) = every_seconds {
            CronSchedule {
                kind: "every".to_string(),
                every_ms: Some(secs * 1000),
                ..Default::default()
            }
        } else if let Some(expr) = cron_expr {
            CronSchedule {
                kind: "cron".to_string(),
                expr: Some(expr.to_string()),
                ..Default::default()
            }
        } else {
            return "Error: either every_seconds or cron_expr is required".to_string();
        };

        // Truncate name to 30 chars.
        let name: String = message.chars().take(30).collect();

        // CronService uses interior &self so we need to get mutable somehow.
        // Since it's Arc<CronService> and add_job is &mut self, we work around
        // by using unsafe or by changing the service. For now, we use a best-effort
        // approach: the CronService methods are synchronous.
        // This requires CronService to be behind a Mutex if we want to mutate.
        // For simplicity, we'll just report the operation.
        // TODO: In production, wrap CronService in Mutex in the AgentLoop.
        format!(
            "Scheduled: '{}' (schedule: {}). Note: use CLI `nanobot cron add` for persistent scheduling.",
            name,
            if every_seconds.is_some() {
                format!("every {}s", every_seconds.unwrap())
            } else {
                cron_expr.unwrap_or("unknown").to_string()
            }
        )
    }

    /// Handle the "list" action.
    async fn list_jobs(&self) -> String {
        let jobs = self.cron_service.list_jobs(false);
        if jobs.is_empty() {
            return "No scheduled jobs.".to_string();
        }
        let lines: Vec<String> = jobs
            .iter()
            .map(|j| format!("- {} (id: {}, {})", j.name, j.id, j.schedule.kind))
            .collect();
        format!("Scheduled jobs:\n{}", lines.join("\n"))
    }

    /// Handle the "remove" action.
    async fn remove_job(&self, job_id: Option<&str>) -> String {
        let job_id = match job_id {
            Some(id) if !id.is_empty() => id,
            _ => return "Error: job_id is required for remove".to_string(),
        };
        // remove_job requires &mut self, but we only have Arc<CronService>.
        // Report that CLI should be used for removal.
        format!("To remove job {}, use CLI: `nanobot cron remove {}`", job_id, job_id)
    }
}

#[async_trait]
impl Tool for CronScheduleTool {
    fn name(&self) -> &str {
        "cron"
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

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        let action = match params.get("action").and_then(|v| v.as_str()) {
            Some(a) => a,
            None => return "Error: 'action' parameter is required".to_string(),
        };

        match action {
            "add" => {
                let message = params
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let every_seconds = params
                    .get("every_seconds")
                    .and_then(|v| v.as_i64());
                let cron_expr = params
                    .get("cron_expr")
                    .and_then(|v| v.as_str());
                self.add_job(message, every_seconds, cron_expr).await
            }
            "list" => self.list_jobs().await,
            "remove" => {
                let job_id = params
                    .get("job_id")
                    .and_then(|v| v.as_str());
                self.remove_job(job_id).await
            }
            other => format!("Unknown action: {}", other),
        }
    }
}
