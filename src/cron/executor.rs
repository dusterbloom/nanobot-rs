//! Cron job executor — computes due jobs, fires them, and advances schedules.
//!
//! ## Where this runs
//!
//! The executor is spawned from **gateway mode** (`run_gateway_async` in
//! `src/cli/mod.rs`), the only mode that consumes the inbound message bus:
//! `agent_turn` payloads are injected as [`InboundMessage`]s on the same
//! channel `AgentLoop::run()` drains, so they flow through the normal
//! concurrency/session machinery.
//!
//! **REPL/TUI limitation (deliberate):** in REPL/TUI mode the agent is driven
//! via `AgentLoop::process_direct()`; `run()` is never called and
//! `create_agent_loop` (core_builder.rs) drops its `outbound_rx` immediately.
//! An injected inbound message would sit unconsumed forever and its response
//! would go to a closed channel. Rather than a fragile side-channel hack,
//! jobs added from the REPL persist to the store and fire the next time a
//! gateway runs (including the REPL-spawned background gateway, which uses
//! `run_gateway_async` with a stop signal).
//!
//! ## Misfire policy
//!
//! A job whose due time passed while nanobot wasn't running fires exactly
//! **once** on the next tick after startup, not N times: when a job fires,
//! its next run is recomputed from *now*, not from the missed slot(s).
//!
//! ## Tick cadence
//!
//! The loop sleeps until the earliest `next_run_at_ms` across enabled jobs,
//! clamped to `[MIN_TICK_MS, MAX_TICK_MS]`. The 30 s cap bounds the latency
//! for noticing jobs added mid-sleep through the shared service handle
//! (the agent's cron tool / future REPL additions); the 250 ms floor keeps
//! clock jitter from busy-spinning the task. Between those bounds the sleep
//! is exact, so second-granularity interval jobs still fire on time — a
//! fixed 30 s tick would quantize them.

use std::future::Future;
use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use chrono::{Local, TimeZone, Utc};
use tokio::sync::mpsc::UnboundedSender;
use tracing::{debug, info, warn};

use crate::bus::events::InboundMessage;
use crate::cron::service::CronService;
use crate::cron::types::{CronJob, CronSchedule, CronStore, PayloadKind};

/// Floor for the executor sleep — avoids busy-spin on clock jitter.
pub(crate) const MIN_TICK_MS: i64 = 250;
/// Cap for the executor sleep — bounds pickup latency for newly added jobs.
pub(crate) const MAX_TICK_MS: i64 = 30_000;

// ---------------------------------------------------------------------------
// Pure scheduling logic (deterministic: `now_ms` is always injected)
// ---------------------------------------------------------------------------

/// First `next_run_at_ms` for a job that has none yet.
///
/// - `"at"`: the absolute timestamp, **even if already past** — a one-shot
///   scheduled while nanobot was down still fires once (misfire policy).
/// - `"every"` / `"cron"`: the next occurrence strictly after `now_ms`
///   (a fresh interval job does not fire immediately on creation).
pub fn initial_next_run(schedule: &CronSchedule, now_ms: i64) -> Option<i64> {
    match schedule.kind.as_str() {
        "at" => schedule.at_ms,
        _ => next_run_after(schedule, now_ms),
    }
}

/// Next run strictly after `now_ms`, or `None` when the schedule is
/// exhausted (`"at"` one-shots) or invalid.
pub fn next_run_after(schedule: &CronSchedule, now_ms: i64) -> Option<i64> {
    match schedule.kind.as_str() {
        // One-shot: only a future timestamp still counts as a next run.
        "at" => schedule.at_ms.filter(|&at| at > now_ms),
        "every" => schedule
            .every_ms
            .filter(|&every| every > 0)
            .map(|every| now_ms + every),
        "cron" => cron_expr_next_ms(schedule.expr.as_deref(), schedule.tz.as_deref(), now_ms),
        other => {
            warn!(
                "Cron: unknown schedule kind '{}' — job will never fire",
                other
            );
            None
        }
    }
}

/// IDs of jobs that are due at `now_ms`: enabled, with a computed
/// `next_run_at_ms <= now_ms`. Pure — no clock reads, no I/O.
pub fn due_job_ids(jobs: &[CronJob], now_ms: i64) -> Vec<String> {
    jobs.iter()
        .filter(|j| is_due(j, now_ms))
        .map(|j| j.id.clone())
        .collect()
}

fn is_due(job: &CronJob, now_ms: i64) -> bool {
    job.enabled && job.state.next_run_at_ms.is_some_and(|next| next <= now_ms)
}

/// Give every enabled job without a `next_run_at_ms` its first one.
/// Returns `true` if anything changed (caller persists).
pub(crate) fn init_missing_next_runs(jobs: &mut [CronJob], now_ms: i64) -> bool {
    let mut changed = false;
    for job in jobs
        .iter_mut()
        .filter(|j| j.enabled && j.state.next_run_at_ms.is_none())
    {
        job.state.next_run_at_ms = initial_next_run(&job.schedule, now_ms);
        changed |= job.state.next_run_at_ms.is_some();
    }
    changed
}

/// Advance every due job's state and return clones for firing.
///
/// Per due job: `last_run_at_ms = now`, `next_run_at_ms` recomputed **from
/// `now_ms`** (misfire policy: overdue-by-N-periods fires once, next run
/// lands in the future). `delete_after_run` jobs are removed; exhausted
/// one-shots (`"at"` with no future run) are disabled.
pub(crate) fn advance_due_jobs(store: &mut CronStore, now_ms: i64) -> Vec<CronJob> {
    let mut due = Vec::new();
    for job in store.jobs.iter_mut().filter(|j| is_due(j, now_ms)) {
        job.state.last_run_at_ms = Some(now_ms);
        job.state.last_status = Some("ok".to_string());
        // Misfire policy: recompute from `now_ms`, never from the missed slot.
        job.state.next_run_at_ms = next_run_after(&job.schedule, now_ms);
        if job.state.next_run_at_ms.is_none() && !job.delete_after_run {
            // Exhausted one-shot: keep for history, never fire again.
            job.enabled = false;
        }
        job.updated_at_ms = now_ms;
        due.push(job.clone());
    }
    if due.iter().any(|j| j.delete_after_run) {
        let fired: Vec<&str> = due.iter().map(|j| j.id.as_str()).collect();
        store
            .jobs
            .retain(|j| !(j.delete_after_run && fired.contains(&j.id.as_str())));
    }
    due
}

/// How long the executor should sleep: until the earliest wakeup, clamped
/// to `[MIN_TICK_MS, MAX_TICK_MS]`. `None` (no scheduled jobs) sleeps the cap.
pub(crate) fn sleep_ms_until(next_wakeup_ms: Option<i64>, now_ms: i64) -> i64 {
    match next_wakeup_ms {
        Some(next) => (next - now_ms).clamp(MIN_TICK_MS, MAX_TICK_MS),
        None => MAX_TICK_MS,
    }
}

// ---------------------------------------------------------------------------
// Executor task
// ---------------------------------------------------------------------------

/// Async hook that runs a memory reflection (built by the gateway from the
/// live core handle so `/model` swaps are honored). Kept as a boxed closure
/// so the cron module stays decoupled from `agent::reflector`.
pub type ReflectFn = Arc<dyn Fn() -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

/// Side-effect hooks the executor fires payloads through.
pub struct ExecutorHooks {
    /// Sender side of the bus `AgentLoop::run()` consumes (`agent_turn`).
    pub inbound_tx: Option<UnboundedSender<InboundMessage>>,
    /// Memory reflection runner (`reflect`).
    pub reflect: Option<ReflectFn>,
}

/// Spawn the background executor. Stop it by aborting the returned handle —
/// job state is persisted at every advance, so an abort loses nothing.
pub fn spawn_executor(
    service: Arc<CronService>,
    hooks: ExecutorHooks,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(run_loop(service, hooks))
}

async fn run_loop(service: Arc<CronService>, hooks: ExecutorHooks) {
    info!("Cron executor started");
    loop {
        let now = Local::now().timestamp_millis();
        for job in service.due_jobs_and_advance(now) {
            info!("Cron: firing job '{}' ({})", job.name, job.id);
            fire_job(&job, &hooks);
        }
        let sleep_ms = sleep_ms_until(service.next_wakeup_ms(), Local::now().timestamp_millis());
        tokio::time::sleep(Duration::from_millis(sleep_ms as u64)).await;
    }
}

/// Dispatch a due job's payload. Never panics: unknown kinds and missing
/// hooks are logged skips.
pub(crate) fn fire_job(job: &CronJob, hooks: &ExecutorHooks) {
    let Some(kind) = PayloadKind::parse(&job.payload.kind) else {
        warn!(
            "Cron: job '{}' ({}) has unknown payload kind '{}' — skipped",
            job.name, job.id, job.payload.kind
        );
        return;
    };
    match kind {
        PayloadKind::AgentTurn => fire_agent_turn(job, hooks.inbound_tx.as_ref()),
        PayloadKind::Reflect => fire_reflect(job, hooks.reflect.as_ref()),
    }
}

fn fire_agent_turn(job: &CronJob, inbound_tx: Option<&UnboundedSender<InboundMessage>>) {
    let Some(tx) = inbound_tx else {
        warn!(
            "Cron: no inbound bus wired — agent_turn job '{}' skipped",
            job.id
        );
        return;
    };
    let (channel, chat_id) = delivery_target(job);
    let mut msg = InboundMessage::new(channel, "cron", chat_id, &job.payload.message);
    msg.metadata
        .insert("cron_job_id".to_string(), serde_json::json!(job.id));
    if tx.send(msg).is_err() {
        warn!(
            "Cron: inbound bus closed — agent_turn job '{}' dropped",
            job.id
        );
    }
}

/// Deliverable jobs route to their channel/recipient; internal reminders use
/// the synthetic `"cron"` channel (responses to it are dropped by the
/// channel manager, but session history still records the turn).
fn delivery_target(job: &CronJob) -> (String, String) {
    if job.payload.deliver {
        if let (Some(channel), Some(to)) = (&job.payload.channel, &job.payload.to) {
            return (channel.clone(), to.clone());
        }
    }
    ("cron".to_string(), job.id.clone())
}

fn fire_reflect(job: &CronJob, reflect: Option<&ReflectFn>) {
    let Some(reflect) = reflect else {
        debug!(
            "Cron: no reflect hook wired — reflect job '{}' skipped",
            job.id
        );
        return;
    };
    // Reflection calls the memory model — run it off the executor tick.
    tokio::spawn(reflect());
}

// ---------------------------------------------------------------------------
// Cron-expression evaluation
// ---------------------------------------------------------------------------

/// Next occurrence of `expr` after `now_ms`, in the schedule's timezone.
/// Invalid expressions return `None` (logged; the job simply never fires).
fn cron_expr_next_ms(expr: Option<&str>, tz: Option<&str>, now_ms: i64) -> Option<i64> {
    let expr = expr?;
    let normalized = normalize_cron_expr(expr);
    let schedule = match cron::Schedule::from_str(&normalized) {
        Ok(s) => s,
        Err(e) => {
            warn!(
                "Cron: invalid expression '{}': {} — job will never fire",
                expr, e
            );
            return None;
        }
    };
    // Only chrono's built-in zones are available (no chrono-tz dependency).
    match tz.map(str::to_ascii_lowercase).as_deref() {
        Some("utc") => next_in_tz(&schedule, Utc, now_ms),
        None | Some("local") => next_in_tz(&schedule, Local, now_ms),
        Some(other) => {
            warn!("Cron: unsupported timezone '{}' — using local time", other);
            next_in_tz(&schedule, Local, now_ms)
        }
    }
}

/// The `cron` crate requires a seconds field; user-facing 5-field
/// expressions get `"0 "` prepended so both forms are accepted.
fn normalize_cron_expr(expr: &str) -> String {
    match expr.split_whitespace().count() {
        5 => format!("0 {}", expr.trim()),
        _ => expr.trim().to_string(),
    }
}

fn next_in_tz<Tz: TimeZone>(schedule: &cron::Schedule, tz: Tz, now_ms: i64) -> Option<i64> {
    let now = tz.timestamp_millis_opt(now_ms).single()?;
    schedule.after(&now).next().map(|dt| dt.timestamp_millis())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cron::types::{CronJobState, CronPayload};

    /// 2024-01-01T00:00:00Z — fixed `now` so every test is deterministic.
    const NOW: i64 = 1_704_067_200_000;
    const HOUR_MS: i64 = 3_600_000;

    fn every(ms: i64) -> CronSchedule {
        CronSchedule {
            kind: "every".to_string(),
            every_ms: Some(ms),
            ..Default::default()
        }
    }

    fn at(ts_ms: i64) -> CronSchedule {
        CronSchedule {
            kind: "at".to_string(),
            at_ms: Some(ts_ms),
            ..Default::default()
        }
    }

    fn cron(expr: &str, tz: Option<&str>) -> CronSchedule {
        CronSchedule {
            kind: "cron".to_string(),
            expr: Some(expr.to_string()),
            tz: tz.map(str::to_string),
            ..Default::default()
        }
    }

    fn job(id: &str, schedule: CronSchedule, next_run: Option<i64>, enabled: bool) -> CronJob {
        CronJob {
            id: id.to_string(),
            name: id.to_string(),
            enabled,
            schedule,
            payload: CronPayload::default(),
            state: CronJobState {
                next_run_at_ms: next_run,
                ..Default::default()
            },
            created_at_ms: 0,
            updated_at_ms: 0,
            delete_after_run: false,
        }
    }

    fn store_of(jobs: Vec<CronJob>) -> CronStore {
        CronStore { version: 1, jobs }
    }

    // ── Due-job selection (pure) ──────────────────────────────────

    #[test]
    fn test_due_selection_returns_exactly_due_jobs() {
        let jobs = vec![
            job("overdue", every(60_000), Some(NOW - 1), true),
            job("due-now", every(60_000), Some(NOW), true),
            job("future", every(60_000), Some(NOW + 1), true),
            job("disabled-overdue", every(60_000), Some(NOW - 1), false),
            job("uninitialized", every(60_000), None, true),
        ];
        assert_eq!(due_job_ids(&jobs, NOW), vec!["overdue", "due-now"]);
        assert!(due_job_ids(&jobs, NOW - 10).is_empty());
    }

    // ── Next-run computation ──────────────────────────────────────

    #[test]
    fn test_next_run_interval() {
        assert_eq!(next_run_after(&every(60_000), NOW), Some(NOW + 60_000));
        // Non-positive or missing intervals are invalid → no next run.
        assert_eq!(next_run_after(&every(0), NOW), None);
        let mut missing = every(1);
        missing.every_ms = None;
        assert_eq!(next_run_after(&missing, NOW), None);
    }

    #[test]
    fn test_next_run_cron_expr_utc() {
        // NOW is midnight UTC; next "0 9 * * *" is 09:00 the same day.
        let s = cron("0 9 * * *", Some("UTC"));
        assert_eq!(next_run_after(&s, NOW), Some(NOW + 9 * HOUR_MS));
        // And from 09:00 sharp, the next is 09:00 tomorrow (strictly after).
        assert_eq!(
            next_run_after(&s, NOW + 9 * HOUR_MS),
            Some(NOW + 33 * HOUR_MS)
        );
    }

    #[test]
    fn test_next_run_cron_invalid_expr_is_none() {
        assert_eq!(next_run_after(&cron("not a cron", Some("UTC")), NOW), None);
        let mut missing = cron("0 9 * * *", None);
        missing.expr = None;
        assert_eq!(next_run_after(&missing, NOW), None);
    }

    #[test]
    fn test_next_run_at_is_one_shot() {
        // A fired (or passed) "at" schedule has no next run.
        assert_eq!(next_run_after(&at(NOW - 1), NOW), None);
        assert_eq!(next_run_after(&at(NOW), NOW), None);
        assert_eq!(next_run_after(&at(NOW + 5_000), NOW), Some(NOW + 5_000));
    }

    #[test]
    fn test_initial_next_run() {
        // Fresh interval jobs schedule ahead — they don't fire on creation.
        assert_eq!(initial_next_run(&every(60_000), NOW), Some(NOW + 60_000));
        // A past "at" still fires once (missed while down = misfire-once).
        assert_eq!(initial_next_run(&at(NOW - 10_000), NOW), Some(NOW - 10_000));
        // Unknown schedule kinds never fire.
        let unknown = CronSchedule {
            kind: "someday".to_string(),
            ..Default::default()
        };
        assert_eq!(initial_next_run(&unknown, NOW), None);
    }

    // ── Misfire policy ────────────────────────────────────────────

    #[test]
    fn test_misfire_overdue_by_3_periods_fires_once() {
        let mut store = store_of(vec![job("j", every(60_000), Some(NOW - 180_000), true)]);
        let fired = advance_due_jobs(&mut store, NOW);
        assert_eq!(fired.len(), 1, "overdue job fires exactly once");
        let next = store.jobs[0].state.next_run_at_ms.expect("next run set");
        assert!(
            next > NOW,
            "next run lands in the future, not a catch-up slot"
        );
        assert!(next <= NOW + 60_000);
        assert_eq!(store.jobs[0].state.last_run_at_ms, Some(NOW));
        // Same instant again: nothing further to fire.
        assert!(advance_due_jobs(&mut store, NOW).is_empty());
    }

    // ── delete_after_run / one-shot exhaustion ────────────────────

    #[test]
    fn test_delete_after_run_removes_job_after_firing() {
        let mut oneshot = job("gone", at(NOW - 1_000), Some(NOW - 1_000), true);
        oneshot.delete_after_run = true;
        let mut store = store_of(vec![
            oneshot,
            job("stays", every(60_000), Some(NOW + 1), true),
        ]);
        let fired = advance_due_jobs(&mut store, NOW);
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].id, "gone");
        assert_eq!(store.jobs.len(), 1);
        assert_eq!(store.jobs[0].id, "stays");
    }

    #[test]
    fn test_exhausted_one_shot_without_delete_is_disabled() {
        let mut store = store_of(vec![job("once", at(NOW - 1), Some(NOW - 1), true)]);
        let fired = advance_due_jobs(&mut store, NOW);
        assert_eq!(fired.len(), 1);
        assert_eq!(store.jobs.len(), 1, "kept for history");
        assert!(!store.jobs[0].enabled, "disabled so it never fires again");
        assert!(store.jobs[0].state.next_run_at_ms.is_none());
    }

    // ── init_missing_next_runs ────────────────────────────────────

    #[test]
    fn test_init_missing_next_runs_only_touches_uninitialized_enabled() {
        let mut jobs = vec![
            job("fresh", every(60_000), None, true),
            job("set", every(60_000), Some(NOW + 5), true),
            job("disabled", every(60_000), None, false),
        ];
        assert!(init_missing_next_runs(&mut jobs, NOW));
        assert_eq!(jobs[0].state.next_run_at_ms, Some(NOW + 60_000));
        assert_eq!(jobs[1].state.next_run_at_ms, Some(NOW + 5));
        assert_eq!(jobs[2].state.next_run_at_ms, None);
        // Second pass: nothing left to initialize.
        assert!(!init_missing_next_runs(&mut jobs, NOW));
    }

    // ── Sleep clamping ────────────────────────────────────────────

    #[test]
    fn test_sleep_ms_clamped_to_tick_bounds() {
        assert_eq!(sleep_ms_until(None, NOW), MAX_TICK_MS);
        assert_eq!(sleep_ms_until(Some(NOW - 5_000), NOW), MIN_TICK_MS);
        assert_eq!(sleep_ms_until(Some(NOW + 5_000), NOW), 5_000);
        assert_eq!(sleep_ms_until(Some(NOW + 90_000), NOW), MAX_TICK_MS);
    }

    // ── Payload dispatch never panics ─────────────────────────────

    #[test]
    fn test_fire_job_unknown_kind_is_logged_skip_not_panic() {
        let mut bogus = job("b", every(60_000), Some(NOW), true);
        bogus.payload.kind = "system_event".to_string();
        let hooks = ExecutorHooks {
            inbound_tx: None,
            reflect: None,
        };
        fire_job(&bogus, &hooks); // must not panic
                                  // Known kinds with missing hooks are also skips, not panics.
        let plain = job("p", every(60_000), Some(NOW), true);
        fire_job(&plain, &hooks);
        let mut refl = job("r", every(60_000), Some(NOW), true);
        refl.payload.kind = "reflect".to_string();
        fire_job(&refl, &hooks);
    }

    // ── End-to-end: executor puts an InboundMessage on the bus ────

    #[tokio::test]
    async fn test_executor_sends_inbound_message_for_due_agent_turn() {
        let tmp = tempfile::NamedTempFile::new().expect("temp file");
        std::fs::remove_file(tmp.path()).ok();
        let service = Arc::new(CronService::new(tmp.path().to_path_buf()));

        // One-shot already due (at in the past) → fires on the first tick.
        let payload = CronPayload {
            kind: "agent_turn".to_string(),
            message: "ping from cron".to_string(),
            deliver: false,
            channel: None,
            to: None,
        };
        let past = Local::now().timestamp_millis() - 1_000;
        service.add_job_with_payload("wakeup", at(past), payload, true);

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
        let handle = spawn_executor(
            service.clone(),
            ExecutorHooks {
                inbound_tx: Some(tx),
                reflect: None,
            },
        );

        let msg = tokio::time::timeout(Duration::from_secs(3), rx.recv())
            .await
            .expect("executor fired within the first tick")
            .expect("channel open");
        handle.abort();

        assert_eq!(msg.content, "ping from cron");
        assert_eq!(msg.channel, "cron");
        assert_eq!(msg.sender_id, "cron");
        assert!(msg.metadata.contains_key("cron_job_id"));
        // delete_after_run honored end-to-end.
        assert!(service.list_jobs(true).is_empty());
    }
}
