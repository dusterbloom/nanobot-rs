//! Cron types – schedule definitions, payloads, job state, and persistence.

use serde::{Deserialize, Serialize};

/// Schedule definition for a cron job.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CronSchedule {
    /// One of `"at"`, `"every"`, or `"cron"`.
    pub kind: String,
    /// For `"at"`: absolute timestamp in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub at_ms: Option<i64>,
    /// For `"every"`: interval in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub every_ms: Option<i64>,
    /// For `"cron"`: a cron expression (e.g. `"0 9 * * *"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expr: Option<String>,
    /// Timezone for cron expressions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tz: Option<String>,
}

impl Default for CronSchedule {
    fn default() -> Self {
        Self {
            kind: "every".to_string(),
            at_ms: None,
            every_ms: None,
            expr: None,
            tz: None,
        }
    }
}

/// What to do when the job runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CronPayload {
    /// `"system_event"` or `"agent_turn"`.
    #[serde(default = "default_payload_kind")]
    pub kind: String,
    /// The message/prompt to send.
    #[serde(default)]
    pub message: String,
    /// Whether to deliver the response to a channel.
    #[serde(default)]
    pub deliver: bool,
    /// Target channel name (e.g. `"whatsapp"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channel: Option<String>,
    /// Recipient identifier (e.g. a phone number).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
}

fn default_payload_kind() -> String {
    "agent_turn".to_string()
}

impl Default for CronPayload {
    fn default() -> Self {
        Self {
            kind: default_payload_kind(),
            message: String::new(),
            deliver: false,
            channel: None,
            to: None,
        }
    }
}

/// Runtime state of a job.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CronJobState {
    /// Next scheduled run time in milliseconds since epoch.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_run_at_ms: Option<i64>,
    /// Last completed run time in milliseconds since epoch.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_run_at_ms: Option<i64>,
    /// `"ok"`, `"error"`, or `"skipped"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_status: Option<String>,
    /// Error message from the last run, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
}

/// A scheduled job.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CronJob {
    pub id: String,
    pub name: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub schedule: CronSchedule,
    #[serde(default)]
    pub payload: CronPayload,
    #[serde(default)]
    pub state: CronJobState,
    #[serde(default)]
    pub created_at_ms: i64,
    #[serde(default)]
    pub updated_at_ms: i64,
    #[serde(default)]
    pub delete_after_run: bool,
}

fn default_true() -> bool {
    true
}

/// Persistent store for cron jobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CronStore {
    #[serde(default = "default_version")]
    pub version: i32,
    #[serde(default)]
    pub jobs: Vec<CronJob>,
}

fn default_version() -> i32 {
    1
}

impl Default for CronStore {
    fn default() -> Self {
        Self {
            version: 1,
            jobs: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── CronSchedule serialization ────────────────────────────────

    #[test]
    fn test_cron_schedule_interval_roundtrip() {
        let schedule = CronSchedule {
            kind: "every".to_string(),
            every_ms: Some(300_000),
            at_ms: None,
            expr: None,
            tz: None,
        };

        let json = serde_json::to_string(&schedule).expect("serialize");
        let parsed: CronSchedule = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.kind, "every");
        assert_eq!(parsed.every_ms, Some(300_000));
        assert!(parsed.at_ms.is_none());
        assert!(parsed.expr.is_none());
        assert!(parsed.tz.is_none());
    }

    #[test]
    fn test_cron_schedule_cron_expr_roundtrip() {
        let schedule = CronSchedule {
            kind: "cron".to_string(),
            expr: Some("0 9 * * 1-5".to_string()),
            tz: Some("America/New_York".to_string()),
            at_ms: None,
            every_ms: None,
        };

        let json = serde_json::to_string(&schedule).expect("serialize");
        let parsed: CronSchedule = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.kind, "cron");
        assert_eq!(parsed.expr.as_deref(), Some("0 9 * * 1-5"));
        assert_eq!(parsed.tz.as_deref(), Some("America/New_York"));
        assert!(parsed.at_ms.is_none());
        assert!(parsed.every_ms.is_none());
    }

    #[test]
    fn test_cron_schedule_at_variant() {
        let schedule = CronSchedule {
            kind: "at".to_string(),
            at_ms: Some(1_700_000_000_000),
            every_ms: None,
            expr: None,
            tz: None,
        };

        let json = serde_json::to_string(&schedule).expect("serialize");
        let parsed: CronSchedule = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.kind, "at");
        assert_eq!(parsed.at_ms, Some(1_700_000_000_000));
    }

    #[test]
    fn test_cron_schedule_default() {
        let schedule = CronSchedule::default();
        assert_eq!(schedule.kind, "every");
        assert!(schedule.at_ms.is_none());
        assert!(schedule.every_ms.is_none());
        assert!(schedule.expr.is_none());
        assert!(schedule.tz.is_none());
    }

    #[test]
    fn test_cron_schedule_skip_serializing_none_fields() {
        let schedule = CronSchedule {
            kind: "every".to_string(),
            every_ms: Some(60_000),
            at_ms: None,
            expr: None,
            tz: None,
        };

        let json = serde_json::to_string(&schedule).expect("serialize");
        // None fields should be omitted due to skip_serializing_if.
        assert!(!json.contains("atMs"));
        assert!(!json.contains("expr"));
        assert!(!json.contains("tz"));
        // Present field should be there (camelCase).
        assert!(json.contains("everyMs"));
    }

    // ── CronSchedule uses camelCase ───────────────────────────────

    #[test]
    fn test_cron_schedule_camel_case_keys() {
        let schedule = CronSchedule {
            kind: "every".to_string(),
            every_ms: Some(1000),
            at_ms: Some(999),
            expr: None,
            tz: None,
        };
        let val: serde_json::Value = serde_json::to_value(&schedule).expect("to_value");
        // Rust field `every_ms` should serialize as `everyMs`.
        assert!(val.get("everyMs").is_some());
        assert!(val.get("atMs").is_some());
        // Rust-style snake_case should not appear.
        assert!(val.get("every_ms").is_none());
        assert!(val.get("at_ms").is_none());
    }

    // ── CronPayload ───────────────────────────────────────────────

    #[test]
    fn test_cron_payload_default() {
        let payload = CronPayload::default();
        assert_eq!(payload.kind, "agent_turn");
        assert_eq!(payload.message, "");
        assert!(!payload.deliver);
        assert!(payload.channel.is_none());
        assert!(payload.to.is_none());
    }

    #[test]
    fn test_cron_payload_roundtrip() {
        let payload = CronPayload {
            kind: "system_event".to_string(),
            message: "Run report".to_string(),
            deliver: true,
            channel: Some("email".to_string()),
            to: Some("user@example.com".to_string()),
        };

        let json = serde_json::to_string(&payload).expect("serialize");
        let parsed: CronPayload = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.kind, "system_event");
        assert_eq!(parsed.message, "Run report");
        assert!(parsed.deliver);
        assert_eq!(parsed.channel.as_deref(), Some("email"));
        assert_eq!(parsed.to.as_deref(), Some("user@example.com"));
    }

    #[test]
    fn test_cron_payload_deserialize_with_defaults() {
        // When deserializing minimal JSON, defaults should kick in.
        let json = r#"{}"#;
        let payload: CronPayload = serde_json::from_str(json).expect("deserialize");
        assert_eq!(payload.kind, "agent_turn");
        assert_eq!(payload.message, "");
        assert!(!payload.deliver);
    }

    // ── CronJobState ──────────────────────────────────────────────

    #[test]
    fn test_cron_job_state_default() {
        let state = CronJobState::default();
        assert!(state.next_run_at_ms.is_none());
        assert!(state.last_run_at_ms.is_none());
        assert!(state.last_status.is_none());
        assert!(state.last_error.is_none());
    }

    #[test]
    fn test_cron_job_state_roundtrip() {
        let state = CronJobState {
            next_run_at_ms: Some(1_700_000_060_000),
            last_run_at_ms: Some(1_700_000_000_000),
            last_status: Some("ok".to_string()),
            last_error: None,
        };

        let json = serde_json::to_string(&state).expect("serialize");
        let parsed: CronJobState = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.next_run_at_ms, Some(1_700_000_060_000));
        assert_eq!(parsed.last_run_at_ms, Some(1_700_000_000_000));
        assert_eq!(parsed.last_status.as_deref(), Some("ok"));
        assert!(parsed.last_error.is_none());
    }

    #[test]
    fn test_cron_job_state_skip_serializing_none() {
        let state = CronJobState::default();
        let json = serde_json::to_string(&state).expect("serialize");
        // All fields are None, so they should all be skipped.
        assert_eq!(json, "{}");
    }

    // ── CronJob ───────────────────────────────────────────────────

    #[test]
    fn test_cron_job_full_roundtrip() {
        let job = CronJob {
            id: "abc12345".to_string(),
            name: "My Job".to_string(),
            enabled: true,
            schedule: CronSchedule {
                kind: "every".to_string(),
                every_ms: Some(120_000),
                at_ms: None,
                expr: None,
                tz: None,
            },
            payload: CronPayload {
                kind: "agent_turn".to_string(),
                message: "check status".to_string(),
                deliver: false,
                channel: None,
                to: None,
            },
            state: CronJobState::default(),
            created_at_ms: 1_700_000_000_000,
            updated_at_ms: 1_700_000_000_000,
            delete_after_run: false,
        };

        let json = serde_json::to_string_pretty(&job).expect("serialize");
        let parsed: CronJob = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.id, "abc12345");
        assert_eq!(parsed.name, "My Job");
        assert!(parsed.enabled);
        assert_eq!(parsed.schedule.kind, "every");
        assert_eq!(parsed.schedule.every_ms, Some(120_000));
        assert_eq!(parsed.payload.message, "check status");
        assert_eq!(parsed.created_at_ms, 1_700_000_000_000);
        assert!(!parsed.delete_after_run);
    }

    #[test]
    fn test_cron_job_enabled_defaults_to_true() {
        // When "enabled" is missing from JSON, it should default to true.
        let json = r#"{
            "id": "x",
            "name": "test",
            "schedule": {"kind": "every"},
            "payload": {},
            "state": {},
            "createdAtMs": 0,
            "updatedAtMs": 0,
            "deleteAfterRun": false
        }"#;
        let job: CronJob = serde_json::from_str(json).expect("deserialize");
        assert!(job.enabled);
    }

    // ── CronStore ─────────────────────────────────────────────────

    #[test]
    fn test_cron_store_default_is_empty() {
        let store = CronStore::default();
        assert_eq!(store.version, 1);
        assert!(store.jobs.is_empty());
    }

    #[test]
    fn test_cron_store_roundtrip() {
        let store = CronStore {
            version: 1,
            jobs: vec![CronJob {
                id: "job1".to_string(),
                name: "Test".to_string(),
                enabled: true,
                schedule: CronSchedule::default(),
                payload: CronPayload::default(),
                state: CronJobState::default(),
                created_at_ms: 0,
                updated_at_ms: 0,
                delete_after_run: false,
            }],
        };

        let json = serde_json::to_string(&store).expect("serialize");
        let parsed: CronStore = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.jobs.len(), 1);
        assert_eq!(parsed.jobs[0].id, "job1");
    }

    #[test]
    fn test_cron_store_deserialize_empty_json_uses_defaults() {
        let json = r#"{}"#;
        let store: CronStore = serde_json::from_str(json).expect("deserialize");
        assert_eq!(store.version, 1);
        assert!(store.jobs.is_empty());
    }
}
