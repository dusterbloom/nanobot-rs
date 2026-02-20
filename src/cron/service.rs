#![allow(dead_code)]
//! Cron service for managing scheduled jobs.

use std::path::PathBuf;

use chrono::Local;
use tracing::{info, warn};
use uuid::Uuid;

use crate::cron::types::{CronJob, CronJobState, CronPayload, CronSchedule, CronStore};

fn now_ms() -> i64 {
    Local::now().timestamp_millis()
}

/// Service that manages cron jobs with file-based persistence.
pub struct CronService {
    store_path: PathBuf,
    store: CronStore,
    running: bool,
}

impl CronService {
    /// Create a new `CronService` with the given store file path.
    pub fn new(store_path: PathBuf) -> Self {
        let store = if store_path.exists() {
            std::fs::read_to_string(&store_path)
                .ok()
                .and_then(|c| serde_json::from_str(&c).ok())
                .unwrap_or_default()
        } else {
            CronStore::default()
        };
        Self {
            store_path,
            store,
            running: false,
        }
    }

    /// Start the cron service.
    pub async fn start(&mut self) {
        self.running = true;
        info!("Cron service started with {} jobs", self.store.jobs.len());
    }

    /// Stop the cron service.
    pub fn stop(&mut self) {
        self.running = false;
    }

    /// Add a new cron job and persist the store.
    pub fn add_job(
        &mut self,
        name: &str,
        schedule: CronSchedule,
        message: &str,
        deliver: bool,
        channel: Option<&str>,
        to: Option<&str>,
        delete_after_run: bool,
    ) -> CronJob {
        let now = now_ms();
        let id = Uuid::new_v4().to_string();
        let short_id = id[..8].to_string();

        let job = CronJob {
            id: short_id,
            name: name.to_string(),
            enabled: true,
            schedule,
            payload: CronPayload {
                kind: "agent_turn".to_string(),
                message: message.to_string(),
                deliver,
                channel: channel.map(|s| s.to_string()),
                to: to.map(|s| s.to_string()),
            },
            state: CronJobState::default(),
            created_at_ms: now,
            updated_at_ms: now,
            delete_after_run,
        };

        self.store.jobs.push(job.clone());
        self.persist();
        info!("Cron: added job '{}' ({})", job.name, job.id);
        job
    }

    /// List all registered jobs.
    pub fn list_jobs(&self, include_disabled: bool) -> Vec<CronJob> {
        if include_disabled {
            self.store.jobs.clone()
        } else {
            self.store
                .jobs
                .iter()
                .filter(|j| j.enabled)
                .cloned()
                .collect()
        }
    }

    /// Remove a job by its ID. Returns `true` if a job was removed.
    pub fn remove_job(&mut self, job_id: &str) -> bool {
        let before = self.store.jobs.len();
        self.store.jobs.retain(|j| j.id != job_id);
        let removed = self.store.jobs.len() < before;
        if removed {
            self.persist();
            info!("Cron: removed job {}", job_id);
        }
        removed
    }

    /// Enable or disable a job.
    pub fn enable_job(&mut self, job_id: &str, enabled: bool) -> Option<CronJob> {
        let job = self.store.jobs.iter_mut().find(|j| j.id == job_id)?;
        job.enabled = enabled;
        job.updated_at_ms = now_ms();
        let result = job.clone();
        self.persist();
        Some(result)
    }

    /// Get service status.
    pub fn status(&self) -> serde_json::Value {
        serde_json::json!({
            "enabled": self.running,
            "jobs": self.store.jobs.len(),
        })
    }

    /// Serialize the current store to disk.
    fn persist(&self) {
        if let Some(parent) = self.store_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        if let Ok(json) = serde_json::to_string_pretty(&self.store) {
            if let Err(e) = std::fs::write(&self.store_path, json) {
                warn!("Failed to persist cron store: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    /// Helper: build a simple "every 60 s" schedule for tests.
    fn every_60s() -> CronSchedule {
        CronSchedule {
            kind: "every".to_string(),
            every_ms: Some(60_000),
            ..CronSchedule::default()
        }
    }

    /// Helper: build a cron-expression schedule for tests.
    fn cron_9am() -> CronSchedule {
        CronSchedule {
            kind: "cron".to_string(),
            expr: Some("0 9 * * *".to_string()),
            tz: Some("UTC".to_string()),
            ..CronSchedule::default()
        }
    }

    /// Helper: create a CronService backed by a fresh temp file.
    fn temp_service() -> (CronService, NamedTempFile) {
        let tmp = NamedTempFile::new().expect("failed to create temp file");
        // Remove the file so CronService starts with an empty default store.
        std::fs::remove_file(tmp.path()).ok();
        let svc = CronService::new(tmp.path().to_path_buf());
        (svc, tmp)
    }

    // ── Basic creation ────────────────────────────────────────────

    #[test]
    fn test_new_service_has_empty_state() {
        let (svc, _tmp) = temp_service();
        assert_eq!(svc.list_jobs(true).len(), 0);
        assert!(!svc.running);
    }

    // ── add_job / list_jobs ───────────────────────────────────────

    #[test]
    fn test_add_job_appears_in_list() {
        let (mut svc, _tmp) = temp_service();
        let job = svc.add_job(
            "Morning check",
            every_60s(),
            "Good morning!",
            false,
            None,
            None,
            false,
        );
        assert_eq!(job.name, "Morning check");
        assert!(job.enabled);
        assert!(!job.delete_after_run);
        assert_eq!(job.payload.message, "Good morning!");
        assert_eq!(job.payload.kind, "agent_turn");

        let jobs = svc.list_jobs(true);
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].id, job.id);
    }

    #[test]
    fn test_add_job_with_channel_and_to() {
        let (mut svc, _tmp) = temp_service();
        let job = svc.add_job(
            "Notify",
            cron_9am(),
            "Daily digest",
            true,
            Some("whatsapp"),
            Some("+1234567890"),
            true,
        );
        assert!(job.payload.deliver);
        assert_eq!(job.payload.channel.as_deref(), Some("whatsapp"));
        assert_eq!(job.payload.to.as_deref(), Some("+1234567890"));
        assert!(job.delete_after_run);
    }

    // ── remove_job ────────────────────────────────────────────────

    #[test]
    fn test_remove_existing_job() {
        let (mut svc, _tmp) = temp_service();
        let job = svc.add_job("temp", every_60s(), "msg", false, None, None, false);
        assert!(svc.remove_job(&job.id));
        assert_eq!(svc.list_jobs(true).len(), 0);
    }

    #[test]
    fn test_remove_nonexistent_job_returns_false() {
        let (mut svc, _tmp) = temp_service();
        assert!(!svc.remove_job("does-not-exist"));
    }

    // ── enable_job (enable / disable) ─────────────────────────────

    #[test]
    fn test_disable_and_enable_job() {
        let (mut svc, _tmp) = temp_service();
        let job = svc.add_job("toggle", every_60s(), "msg", false, None, None, false);

        // Disable.
        let updated = svc.enable_job(&job.id, false).expect("job should exist");
        assert!(!updated.enabled);

        // The disabled job should be hidden from the default list.
        assert_eq!(svc.list_jobs(false).len(), 0);
        // But visible when include_disabled = true.
        assert_eq!(svc.list_jobs(true).len(), 1);

        // Re-enable.
        let updated = svc.enable_job(&job.id, true).expect("job should exist");
        assert!(updated.enabled);
        assert_eq!(svc.list_jobs(false).len(), 1);
    }

    #[test]
    fn test_enable_nonexistent_job_returns_none() {
        let (mut svc, _tmp) = temp_service();
        assert!(svc.enable_job("no-such-id", true).is_none());
    }

    // ── list_jobs with include_disabled ────────────────────────────

    #[test]
    fn test_list_jobs_include_disabled_filtering() {
        let (mut svc, _tmp) = temp_service();
        let j1 = svc.add_job("a", every_60s(), "m", false, None, None, false);
        let _j2 = svc.add_job("b", every_60s(), "m", false, None, None, false);

        svc.enable_job(&j1.id, false);

        assert_eq!(svc.list_jobs(true).len(), 2);
        assert_eq!(svc.list_jobs(false).len(), 1);
        assert_eq!(svc.list_jobs(false)[0].name, "b");
    }

    // ── status ────────────────────────────────────────────────────

    #[test]
    fn test_status_when_stopped() {
        let (svc, _tmp) = temp_service();
        let st = svc.status();
        assert_eq!(st["enabled"], serde_json::json!(false));
        assert_eq!(st["jobs"], serde_json::json!(0));
    }

    #[tokio::test]
    async fn test_status_when_running_with_jobs() {
        let (mut svc, _tmp) = temp_service();
        svc.add_job("j", every_60s(), "m", false, None, None, false);
        svc.start().await;

        let st = svc.status();
        assert_eq!(st["enabled"], serde_json::json!(true));
        assert_eq!(st["jobs"], serde_json::json!(1));
    }

    // ── start / stop ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_start_and_stop() {
        let (mut svc, _tmp) = temp_service();
        assert!(!svc.running);
        svc.start().await;
        assert!(svc.running);
        svc.stop();
        assert!(!svc.running);
    }

    // ── persistence ───────────────────────────────────────────────

    #[test]
    fn test_persistence_roundtrip() {
        let tmp = NamedTempFile::new().expect("failed to create temp file");
        let path = tmp.path().to_path_buf();
        // Remove so we start fresh.
        std::fs::remove_file(&path).ok();

        // Service 1: add two jobs.
        let (job1_id, job2_id) = {
            let mut svc = CronService::new(path.clone());
            let j1 = svc.add_job("alpha", every_60s(), "hello", false, None, None, false);
            let j2 = svc.add_job(
                "beta",
                cron_9am(),
                "world",
                true,
                Some("slack"),
                None,
                false,
            );
            (j1.id, j2.id)
        };

        // Service 2: load from the same path and verify the jobs survived.
        let svc2 = CronService::new(path);
        let jobs = svc2.list_jobs(true);
        assert_eq!(jobs.len(), 2);
        assert_eq!(jobs[0].id, job1_id);
        assert_eq!(jobs[0].name, "alpha");
        assert_eq!(jobs[1].id, job2_id);
        assert_eq!(jobs[1].name, "beta");
        assert!(jobs[1].payload.deliver);
        assert_eq!(jobs[1].payload.channel.as_deref(), Some("slack"));
    }

    #[test]
    fn test_persistence_after_remove() {
        let tmp = NamedTempFile::new().expect("failed to create temp file");
        let path = tmp.path().to_path_buf();
        std::fs::remove_file(&path).ok();

        let job_id = {
            let mut svc = CronService::new(path.clone());
            let j = svc.add_job("ephemeral", every_60s(), "x", false, None, None, false);
            svc.remove_job(&j.id);
            j.id
        };

        let svc2 = CronService::new(path);
        assert_eq!(svc2.list_jobs(true).len(), 0);
        // Make sure the removed ID is truly gone.
        assert!(svc2.list_jobs(true).iter().all(|j| j.id != job_id));
    }

    // ── Job ID format ─────────────────────────────────────────────

    #[test]
    fn test_job_id_is_short_uuid_prefix() {
        let (mut svc, _tmp) = temp_service();
        let job = svc.add_job("id-test", every_60s(), "m", false, None, None, false);
        // The id should be the first 8 characters of a UUID v4 string.
        assert_eq!(job.id.len(), 8);
        assert!(job.id.chars().all(|c| c.is_ascii_hexdigit() || c == '-'));
    }

    // ── Timestamps ────────────────────────────────────────────────

    #[test]
    fn test_created_and_updated_timestamps_set() {
        let (mut svc, _tmp) = temp_service();
        let job = svc.add_job("ts", every_60s(), "m", false, None, None, false);
        assert!(job.created_at_ms > 0);
        assert_eq!(job.created_at_ms, job.updated_at_ms);
    }
}
