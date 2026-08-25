// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(clippy::as_conversions, clippy::shadow_reuse)]
//! Local-backend stream progress tracking, backend-activity heartbeat, and
//! stream-abort diagnostics.
//!
//! Extracted verbatim from `shared.rs`.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::mpsc::UnboundedSender;

use crate::agent::agent_loop::heuristics::{
    local_artifact_action_with_sticky, LocalArtifactAction,
};
use crate::turn_stream::{BackendActivity, ControlMarker};

use super::shared::TurnContext;

const LOCAL_STREAM_NO_PROGRESS_TIMEOUT: Duration = Duration::from_secs(120);
const BACKEND_ACTIVITY_HEARTBEAT: Duration = Duration::from_secs(5);
fn backend_activity_code(phase: BackendActivity) -> u8 {
    match phase {
        BackendActivity::WaitingForHeaders => 0,
        BackendActivity::Prefill => 1,
        BackendActivity::AwaitingToolPayload => 2,
        BackendActivity::ToolPayload => 3,
        BackendActivity::Decoding => 4,
    }
}

fn backend_activity_from_code(code: u8) -> BackendActivity {
    match code {
        1 => BackendActivity::Prefill,
        2 => BackendActivity::AwaitingToolPayload,
        3 => BackendActivity::ToolPayload,
        4 => BackendActivity::Decoding,
        _ => BackendActivity::WaitingForHeaders,
    }
}

fn send_backend_activity_marker(
    tx: &UnboundedSender<String>,
    phase: BackendActivity,
    idle_ms: u64,
) {
    let _ = tx.send(ControlMarker::BackendActivity { phase, idle_ms }.encode());
}

pub(super) struct BackendActivityHeartbeat {
    tx: UnboundedSender<String>,
    stop: Arc<AtomicBool>,
    phase: Arc<AtomicU8>,
    last_progress_ms: Arc<AtomicU64>,
    started: Instant,
    task: tokio::task::JoinHandle<()>,
}

impl BackendActivityHeartbeat {
    pub(super) fn start(tx: Option<UnboundedSender<String>>, enabled: bool) -> Option<Self> {
        if !enabled {
            return None;
        }
        let tx = tx?;
        let stop = Arc::new(AtomicBool::new(false));
        let phase = Arc::new(AtomicU8::new(backend_activity_code(
            BackendActivity::WaitingForHeaders,
        )));
        let last_progress_ms = Arc::new(AtomicU64::new(0));
        let started = Instant::now();
        send_backend_activity_marker(&tx, BackendActivity::WaitingForHeaders, 0);

        let task_stop = stop.clone();
        let task_phase = phase.clone();
        let task_last_progress = last_progress_ms.clone();
        let task_tx = tx.clone();
        let task = tokio::spawn(async move {
            loop {
                tokio::time::sleep(BACKEND_ACTIVITY_HEARTBEAT).await;
                if task_stop.load(Ordering::Relaxed) {
                    break;
                }
                let elapsed_ms = started.elapsed().as_millis() as u64;
                let last_ms = task_last_progress.load(Ordering::Relaxed);
                let idle_ms = elapsed_ms.saturating_sub(last_ms);
                send_backend_activity_marker(
                    &task_tx,
                    backend_activity_from_code(task_phase.load(Ordering::Relaxed)),
                    idle_ms,
                );
            }
        });

        Some(Self {
            tx,
            stop,
            phase,
            last_progress_ms,
            started,
            task,
        })
    }

    pub(super) fn mark_progress(&self, phase: BackendActivity) {
        let code = backend_activity_code(phase);
        let old = self.phase.swap(code, Ordering::Relaxed);
        self.last_progress_ms
            .store(self.started.elapsed().as_millis() as u64, Ordering::Relaxed);
        if old != code {
            send_backend_activity_marker(&self.tx, phase, 0);
        }
    }
}

impl Drop for BackendActivityHeartbeat {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.task.abort();
    }
}

#[derive(Debug)]
pub(super) struct LocalStreamProgress;

impl LocalStreamProgress {
    pub(super) fn new() -> Self {
        Self
    }

    pub(super) fn on_prefill_progress(&mut self, processed: u64, total: u64) -> BackendActivity {
        if total > 0 && processed >= total {
            BackendActivity::Decoding
        } else {
            BackendActivity::Prefill
        }
    }

    pub(super) fn on_text_or_thinking_delta(&mut self) -> BackendActivity {
        BackendActivity::Decoding
    }

    pub(super) fn on_tool_call_delta(&mut self) -> BackendActivity {
        BackendActivity::ToolPayload
    }
}

pub(super) fn local_artifact_action_for_turn(ctx: &TurnContext) -> Option<LocalArtifactAction> {
    if !ctx.core.mode().is_local() {
        return None;
    }
    let sticky_action = ctx
        .counters
        .local_artifact_intent_is_rich(&ctx.session_key, ctx.turn_count)
        .map(|is_rich| {
            if is_rich {
                LocalArtifactAction::Rich
            } else {
                LocalArtifactAction::Simple
            }
        });
    let action = local_artifact_action_with_sticky(&ctx.user_content, sticky_action);
    if let Some(action) = action {
        ctx.counters.record_local_artifact_intent(
            &ctx.session_key,
            ctx.turn_count,
            matches!(action, LocalArtifactAction::Rich),
        );
    }
    action
}

pub(super) fn local_stream_no_progress_timeout(ctx: &TurnContext) -> Option<Duration> {
    if !ctx.core.mode().is_local() {
        return None;
    }
    // Evaluate for the intent-recording side effect; rich and simple turns use
    // the same sliding transport-inactivity deadline.
    let _ = local_artifact_action_for_turn(ctx);
    Some(LOCAL_STREAM_NO_PROGRESS_TIMEOUT)
}

pub(super) fn local_no_stream_headers_error(timeout: Duration) -> String {
    format!(
        "The local backend produced no stream headers for {}s, so I aborted the inactive turn.",
        timeout.as_secs()
    )
}

pub(super) fn local_no_stream_progress_error(timeout: Duration) -> String {
    format!(
        "The local backend produced no stream progress for {}s, so I aborted the inactive turn.",
        timeout.as_secs()
    )
}

pub(super) fn emit_stream_abort_metrics(ctx: &TurnContext, detail: &str) {
    let metrics = crate::agent::metrics::RequestMetrics {
        timestamp: chrono::Local::now().to_rfc3339(),
        request_id: ctx.request_id.clone(),
        logical_session: ctx.session_id.clone(),
        cache_route: "active".into(),
        role: "main".into(),
        model: ctx.core.model.clone(),
        provider_base: ctx.core.provider.get_api_base().unwrap_or("unknown").into(),
        elapsed_ms: ctx
            .flow
            .llm_call_start
            .map_or(0, |t| t.elapsed().as_millis() as u64),
        ttft_ms: ctx.flow.ttft_ms,
        prompt_tokens: 0,
        completion_tokens: 0,
        cache_read_tokens: None,
        cache_creation_tokens: None,
        status: "error".into(),
        error_detail: Some(detail.to_string()),
        raw_response: None,
        anti_drift_score: None,
        anti_drift_signals: None,
        tool_calls_requested: 0,
        tool_calls_executed: 0,
        validation_result: None,
    };
    ctx.counters.record_cache_metrics(
        &metrics.logical_session,
        metrics.prompt_tokens,
        metrics.cache_read_tokens,
        metrics.cache_creation_tokens,
    );
    crate::agent::metrics::emit(&metrics);
}

#[cfg(test)]
mod stream_progress_tests {
    use super::{BackendActivity, LocalStreamProgress};

    #[test]
    fn healthy_artifact_progress_has_no_second_wall_clock_deadline() {
        let mut progress = LocalStreamProgress::new();

        assert_eq!(
            progress.on_prefill_progress(40, 100),
            BackendActivity::Prefill
        );
        assert_eq!(
            progress.on_prefill_progress(100, 100),
            BackendActivity::Decoding
        );
        assert_eq!(
            progress.on_text_or_thinking_delta(),
            BackendActivity::Decoding
        );
        assert_eq!(progress.on_tool_call_delta(), BackendActivity::ToolPayload);
    }
}
