#![allow(dead_code)]
//! Cron service for managing scheduled jobs.
//!
//! Interior mutability (`Mutex<CronStore>` + `AtomicBool`) so the service can
//! be shared as `Arc<CronService>` between the agent loop, the CLI, and the
//! background executor (`cron::executor`) — the historical `&mut self` API is
//! why jobs were never fired: mutation was impossible through the `Arc`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use chrono::Local;
use parking_lot::Mutex;
use tracing::{info, warn};
use uuid::Uuid;

use crate::cron::executor;
use crate::cron::types::{CronJob, CronPayload, CronSchedule, CronStore, PayloadKind};

fn now_ms() -> i64 {
    Local::now().timestamp_millis()
}

/// Service that manages cron jobs with file-based persistence.
pub struct CronService {
    store_path: PathBuf,
    store: Mutex<CronStore>,
    running: AtomicBool,
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
            store: Mutex::new(store),
            running: AtomicBool::new(false),
        }
    }

    /// Start the cron service.
    pub async fn start(&self) {
        self.running.store(true, Ordering::Relaxed);
        info!(
            "Cron service started with {} jobs",
            self.store.lock().jobs.len()
        );
    }

    /// Stop the cron service.
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }

    /// Add a new `agent_turn` cron job and persist the store.
    #[allow(clippy::too_many_arguments)]
    pub fn add_job(
        &self,
        name: &str,
        schedule: CronSchedule,
        message: &str,
        deliver: bool,
        channel: Option<&str>,
        to: Option<&str>,
        delete_after_run: bool,
    ) -> CronJob {
        let payload = CronPayload {
            kind: PayloadKind::AgentTurn.as_str().to_string(),
            message: message.to_string(),
            deliver,
            channel: channel.map(str::to_string),
            to: to.map(str::to_string),
        };
        self.add_job_with_payload(name, schedule, payload, delete_after_run)
    }

    /// Add a cron job with an explicit payload (any [`PayloadKind`]).
    pub fn add_job_with_payload(
        &self,
        name: &str,
        schedule: CronSchedule,
        payload: CronPayload,
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
            payload,
            state: Default::default(),
            created_at_ms: now,
            updated_at_ms: now,
            delete_after_run,
        };

        let mut store = self.store.lock();
        store.jobs.push(job.clone());
        self.write_store(&store);
        info!("Cron: added job '{}' ({})", job.name, job.id);
        job
    }

    /// List all registered jobs.
    pub fn list_jobs(&self, include_disabled: bool) -> Vec<CronJob> {
        let store = self.store.lock();
        if include_disabled {
            store.jobs.clone()
        } else {
            store.jobs.iter().filter(|j| j.enabled).cloned().collect()
        }
    }

    /// Remove a job by its ID. Returns `true` if a job was removed.
    pub fn remove_job(&self, job_id: &str) -> bool {
        let mut store = self.store.lock();
        let before = store.jobs.len();
        store.jobs.retain(|j| j.id != job_id);
        let removed = store.jobs.len() < before;
        if removed {
            self.write_store(&store);
            info!("Cron: removed job {}", job_id);
        }
        removed
    }

    /// Enable or disable a job.
    pub fn enable_job(&self, job_id: &str, enabled: bool) -> Option<CronJob> {
        let mut store = self.store.lock();
        let job = store.jobs.iter_mut().find(|j| j.id == job_id)?;
        job.enabled = enabled;
        job.updated_at_ms = now_ms();
        let result = job.clone();
        self.write_store(&store);
        Some(result)
    }

    /// Get service status.
    pub fn status(&self) -> serde_json::Value {
        serde_json::json!({
            "enabled": self.running.load(Ordering::Relaxed),
            "jobs": self.store.lock().jobs.len(),
        })
    }

    /// Executor entry point: initialize fresh jobs, advance every due job's
    /// state (misfire policy: recompute from `now_ms`), honor
    /// `delete_after_run`, persist once, and return the due jobs for firing.
    pub fn due_jobs_and_advance(&self, now_ms: i64) -> Vec<CronJob> {
        let mut store = self.store.lock();
        let initialized = executor::init_missing_next_runs(&mut store.jobs, now_ms);
        let due = executor::advance_due_jobs(&mut store, now_ms);
        if initialized || !due.is_empty() {
            self.write_store(&store);
        }
        due
    }

    /// Earliest `next_run_at_ms` across enabled jobs (executor sleep target).
    pub fn next_wakeup_ms(&self) -> Option<i64> {
        self.store
            .lock()
            .jobs
            .iter()
            .filter(|j| j.enabled)
            .filter_map(|j| j.state.next_run_at_ms)
            .min()
    }

    /// Serialize the given (already locked) store to disk.
    fn write_store(&self, store: &CronStore) {
        if let Some(parent) = self.store_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        if let Ok(json) = serde_json::to_string_pretty(store) {
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
        assert!(!svc.running.load(Ordering::Relaxed));
    }

    // ── add_job / list_jobs ───────────────────────────────────────

    #[test]
    fn test_add_job_appears_in_list() {
        let (svc, _tmp) = temp_service();
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
        let (svc, _tmp) = temp_service();
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
        let (svc, _tmp) = temp_service();
        let job = svc.add_job("temp", every_60s(), "msg", false, None, None, false);
        assert!(svc.remove_job(&job.id));
        assert_eq!(svc.list_jobs(true).len(), 0);
    }

    #[test]
    fn test_remove_nonexistent_job_returns_false() {
        let (svc, _tmp) = temp_service();
        assert!(!svc.remove_job("does-not-exist"));
    }

    // ── enable_job (enable / disable) ─────────────────────────────

    #[test]
    fn test_disable_and_enable_job() {
        let (svc, _tmp) = temp_service();
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
        let (svc, _tmp) = temp_service();
        assert!(svc.enable_job("no-such-id", true).is_none());
    }

    // ── list_jobs with include_disabled ────────────────────────────

    #[test]
    fn test_list_jobs_include_disabled_filtering() {
        let (svc, _tmp) = temp_service();
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
        let (svc, _tmp) = temp_service();
        svc.add_job("j", every_60s(), "m", false, None, None, false);
        svc.start().await;

        let st = svc.status();
        assert_eq!(st["enabled"], serde_json::json!(true));
        assert_eq!(st["jobs"], serde_json::json!(1));
    }

    // ── start / stop ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_start_and_stop() {
        let (svc, _tmp) = temp_service();
        assert!(!svc.running.load(Ordering::Relaxed));
        svc.start().await;
        assert!(svc.running.load(Ordering::Relaxed));
        svc.stop();
        assert!(!svc.running.load(Ordering::Relaxed));
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
            let svc = CronService::new(path.clone());
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
            let svc = CronService::new(path.clone());
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
        let (svc, _tmp) = temp_service();
        let job = svc.add_job("id-test", every_60s(), "m", false, None, None, false);
        // The id should be the first 8 characters of a UUID v4 string.
        assert_eq!(job.id.len(), 8);
        assert!(job.id.chars().all(|c| c.is_ascii_hexdigit() || c == '-'));
    }

    // ── Timestamps ────────────────────────────────────────────────

    #[test]
    fn test_created_and_updated_timestamps_set() {
        let (svc, _tmp) = temp_service();
        let job = svc.add_job("ts", every_60s(), "m", false, None, None, false);
        assert!(job.created_at_ms > 0);
        assert_eq!(job.created_at_ms, job.updated_at_ms);
    }
}
