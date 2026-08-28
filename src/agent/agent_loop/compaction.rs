// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(clippy::indexing_slicing)]
//! LCM compaction execution: cancellation-safe engine mutation and checkpoint
//! persistence through the foreground provider/model.
//!
//! Extracted verbatim from `shared.rs`.

use std::future::Future;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use serde_json::Value;
use tracing::warn;

use crate::agent::agent_core::{PendingCompaction, SwappableCore};
use crate::agent::lcm::{CompactionFailureMode, LcmCompactionState, LcmEngine};
use crate::agent::token_budget::TokenBudget;
use crate::session::db::{ModelCallPurpose, ReplayRecordingProvider, TurnReplayRecorder};

#[derive(Clone, Copy)]
enum CompactionPhase {
    Generating,
    Publishing,
    Aborting,
}

impl CompactionPhase {
    const fn encoded(self) -> u8 {
        match self {
            Self::Generating => 0,
            Self::Publishing => 1,
            Self::Aborting => 2,
        }
    }

    const fn from_encoded(value: u8) -> Self {
        match value {
            1 => Self::Publishing,
            2 => Self::Aborting,
            _ => Self::Generating,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Generating => "generating",
            Self::Publishing => "publishing",
            Self::Aborting => "aborting",
        }
    }
}

/// Atomic handoff between cancellable generation and durable publication.
pub(super) struct CompactionPublication {
    phase: AtomicU8,
}

impl CompactionPublication {
    pub(super) const fn new() -> Self {
        Self {
            phase: AtomicU8::new(CompactionPhase::Generating.encoded()),
        }
    }

    pub(super) fn begin_publication(&self) -> bool {
        self.phase
            .compare_exchange(
                CompactionPhase::Generating.encoded(),
                CompactionPhase::Publishing.encoded(),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    fn abort_generation(&self) -> bool {
        match self.phase.compare_exchange(
            CompactionPhase::Generating.encoded(),
            CompactionPhase::Aborting.encoded(),
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => true,
            Err(phase) => phase == CompactionPhase::Aborting.encoded(),
        }
    }

    fn phase(&self) -> CompactionPhase {
        CompactionPhase::from_encoded(self.phase.load(Ordering::Acquire))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CompactionAdmissionState {
    Open,
    Closing,
    Retired,
}

impl CompactionAdmissionState {
    const fn encoded(self) -> u8 {
        match self {
            Self::Open => 0,
            Self::Closing => 1,
            Self::Retired => 2,
        }
    }

    const fn from_encoded(value: u8) -> Self {
        match value {
            1 => Self::Closing,
            2 => Self::Retired,
            _ => Self::Open,
        }
    }
}

/// Shared handles for background compaction coordination.
#[derive(Clone)]
pub(crate) struct CompactionHandle {
    pub(crate) slot: Arc<tokio::sync::Mutex<Option<PendingCompaction>>>,
    lifecycle: Arc<CompactionLifecycle>,
}

struct CompactionLifecycle {
    job: tokio::sync::Mutex<Option<CompactionJob>>,
    admission: Arc<tokio::sync::Semaphore>,
    admission_state: AtomicU8,
    session_id: Arc<str>,
}

struct CompactionJob {
    cancellation: tokio_util::sync::CancellationToken,
    publication: Arc<CompactionPublication>,
    handle: tokio::task::JoinHandle<()>,
    completion: tokio::sync::watch::Receiver<CompactionCompletionState>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CompactionCompletionState {
    Running,
    Complete,
}

struct CompactionCompletion(tokio::sync::watch::Sender<CompactionCompletionState>);

impl Drop for CompactionCompletion {
    fn drop(&mut self) {
        self.0.send_replace(CompactionCompletionState::Complete);
    }
}

pub(in crate::agent) struct CompactionAdmission {
    lifecycle: Arc<CompactionLifecycle>,
    _permit: tokio::sync::OwnedSemaphorePermit,
}

pub(super) struct CompactionClosingAdmission {
    lifecycle: Arc<CompactionLifecycle>,
    _permit: tokio::sync::OwnedSemaphorePermit,
}

impl CompactionClosingAdmission {
    pub(super) fn retire(&self) {
        self.lifecycle.admission_state.store(
            CompactionAdmissionState::Retired.encoded(),
            Ordering::Release,
        );
    }
}

impl Drop for CompactionClosingAdmission {
    fn drop(&mut self) {
        let state = CompactionAdmissionState::from_encoded(
            self.lifecycle.admission_state.load(Ordering::Acquire),
        );
        if state == CompactionAdmissionState::Closing {
            self.lifecycle
                .admission_state
                .store(CompactionAdmissionState::Open.encoded(), Ordering::Release);
        }
    }
}

#[derive(Clone, Copy)]
enum CompactionReapMode {
    Wait,
    CancelGeneration,
}

impl CompactionHandle {
    pub(crate) fn new() -> Self {
        Self::for_session("unknown")
    }

    pub(super) fn for_session(session_id: impl Into<Arc<str>>) -> Self {
        Self {
            slot: Arc::new(tokio::sync::Mutex::new(None)),
            lifecycle: Arc::new(CompactionLifecycle {
                job: tokio::sync::Mutex::new(None),
                admission: Arc::new(tokio::sync::Semaphore::new(1)),
                admission_state: AtomicU8::new(CompactionAdmissionState::Open.encoded()),
                session_id: session_id.into(),
            }),
        }
    }

    pub(crate) async fn has_job(&self) -> bool {
        self.lifecycle.job.lock().await.is_some()
    }

    pub(crate) async fn has_pending(&self) -> bool {
        self.slot.lock().await.is_some()
    }

    pub(in crate::agent) async fn admit(&self) -> Option<CompactionAdmission> {
        let Ok(permit) = self.lifecycle.admission.clone().acquire_owned().await else {
            return None;
        };
        let state = CompactionAdmissionState::from_encoded(
            self.lifecycle.admission_state.load(Ordering::Acquire),
        );
        (state == CompactionAdmissionState::Open).then(|| CompactionAdmission {
            lifecycle: self.lifecycle.clone(),
            _permit: permit,
        })
    }

    pub(super) async fn begin_close(&self) -> Option<CompactionClosingAdmission> {
        let permit = self
            .lifecycle
            .admission
            .clone()
            .acquire_owned()
            .await
            .ok()?;
        let state = CompactionAdmissionState::from_encoded(
            self.lifecycle.admission_state.load(Ordering::Acquire),
        );
        if state != CompactionAdmissionState::Open {
            return None;
        }
        self.lifecycle.admission_state.store(
            CompactionAdmissionState::Closing.encoded(),
            Ordering::Release,
        );
        Some(CompactionClosingAdmission {
            lifecycle: self.lifecycle.clone(),
            _permit: permit,
        })
    }

    pub(super) fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.lifecycle, &other.lifecycle)
    }

    pub(super) async fn try_start<F, Fut>(&self, run: F) -> bool
    where
        F: FnOnce(tokio_util::sync::CancellationToken, Arc<CompactionPublication>) -> Fut
            + Send
            + 'static,
        Fut: Future<Output = Option<PendingCompaction>> + Send + 'static,
    {
        let Some(admission) = self.admit().await else {
            return false;
        };
        self.try_start_admitted(&admission, run).await
    }

    pub(super) async fn try_start_admitted<F, Fut>(
        &self,
        admission: &CompactionAdmission,
        run: F,
    ) -> bool
    where
        F: FnOnce(tokio_util::sync::CancellationToken, Arc<CompactionPublication>) -> Fut
            + Send
            + 'static,
        Fut: Future<Output = Option<PendingCompaction>> + Send + 'static,
    {
        debug_assert!(Arc::ptr_eq(&self.lifecycle, &admission.lifecycle));
        let mut job = self.lifecycle.job.lock().await;
        if job.is_some() || self.slot.lock().await.is_some() {
            return false;
        }

        let cancellation = tokio_util::sync::CancellationToken::new();
        let publication = Arc::new(CompactionPublication::new());
        let task_cancellation = cancellation.clone();
        let task_publication = publication.clone();
        let slot = self.slot.clone();
        let (completion_tx, completion) =
            tokio::sync::watch::channel(CompactionCompletionState::Running);
        let handle = tokio::spawn(async move {
            let _completion = CompactionCompletion(completion_tx);
            if let Some(pending) = run(task_cancellation, task_publication).await {
                *slot.lock().await = Some(pending);
            }
        });
        *job = Some(CompactionJob {
            cancellation,
            publication,
            handle,
            completion,
        });
        true
    }

    async fn reap(&self, mode: CompactionReapMode) {
        let mut guard = self.lifecycle.job.lock().await;
        let Some(job) = guard.as_mut() else {
            return;
        };
        if matches!(mode, CompactionReapMode::CancelGeneration)
            && !job.handle.is_finished()
            && job.publication.abort_generation()
        {
            job.cancellation.cancel();
        }
        if let Err(error) = (&mut job.handle).await {
            warn!(
                %error,
                session_id = %self.lifecycle.session_id,
                phase = %job.publication.phase().as_str(),
                "owned compaction task failed"
            );
        }
        *guard = None;
    }

    pub(crate) async fn wait_for_completion(&self) {
        let mut completion = {
            let guard = self.lifecycle.job.lock().await;
            let Some(job) = guard.as_ref() else {
                return;
            };
            let completion = job.completion.clone();
            drop(guard);
            completion
        };
        let state = *completion.borrow();
        if state != CompactionCompletionState::Complete {
            let _ = completion.changed().await;
        }
        self.reap(CompactionReapMode::Wait).await;
    }

    pub(crate) async fn cancel_and_reap(&self) {
        self.reap(CompactionReapMode::CancelGeneration).await;
    }
}

impl Drop for CompactionLifecycle {
    fn drop(&mut self) {
        let Some(job) = self.job.get_mut().as_mut() else {
            return;
        };
        if job.publication.abort_generation() {
            job.cancellation.cancel();
            job.handle.abort();
        }
    }
}

/// Cancellation-safe ownership of one tentative LCM mutation. Unless SQLite
/// commit explicitly succeeds, dropping the future restores the prior DAG and
/// active window before releasing the per-session engine lock.
pub(super) struct LcmCompactionMutation<'a> {
    engine: &'a mut LcmEngine,
    rollback: Option<LcmCompactionState>,
}

impl<'a> LcmCompactionMutation<'a> {
    pub(super) fn new(engine: &'a mut LcmEngine) -> Self {
        Self {
            rollback: Some(engine.compaction_state()),
            engine,
        }
    }

    pub(super) fn engine(&self) -> &LcmEngine {
        self.engine
    }

    pub(super) fn engine_mut(&mut self) -> &mut LcmEngine {
        self.engine
    }

    fn commit(&mut self) {
        self.rollback = None;
    }
}

impl Drop for LcmCompactionMutation<'_> {
    fn drop(&mut self) {
        if let Some(state) = self.rollback.take() {
            self.engine.restore_compaction_state(state);
        }
    }
}

/// Execute one LCM compaction and persist its lossless summary metadata.
/// The session handle owns this future for both hard and soft pressure so a
/// dropped foreground waiter cannot interrupt publication.
#[allow(clippy::too_many_arguments)]
pub(super) async fn execute_lcm_compaction(
    core: Arc<SwappableCore>,
    session_id: String,
    lcm: Arc<tokio::sync::Mutex<LcmEngine>>,
    messages: Vec<Value>,
    session_turn: u64,
    failure_mode: CompactionFailureMode,
    cancellation: tokio_util::sync::CancellationToken,
    publication: Arc<CompactionPublication>,
) -> Option<PendingCompaction> {
    // Keep the engine locked through SQLite commit. A failed checkpoint rolls
    // back the DAG/active window before any other task can observe or extend
    // the non-durable state.
    let mut engine = tokio::select! {
        biased;
        () = cancellation.cancelled() => return None,
        engine = lcm.lock() => engine,
    };
    // Stamp the session turn so the new summary node records its creation
    // turn. auto_expand's fresh-summary cooldown uses this to prevent the
    // just-compacted originals from being reinjected on the very next turn
    // (live failure 2026-07-27 12:13:06).
    let mut mutation = LcmCompactionMutation::new(&mut engine);
    mutation.engine_mut().set_current_turn(session_turn);
    if failure_mode == CompactionFailureMode::PreserveContext {
        mutation.engine_mut().request_async_compaction();
    }
    let replay = TurnReplayRecorder::new(
        Arc::clone(&core.sessions),
        session_id.clone(),
        format!("compaction:{session_turn}:{}", uuid::Uuid::new_v4()),
        session_turn,
    );
    let recorded_provider: Arc<dyn crate::providers::base::LLMProvider> =
        Arc::new(ReplayRecordingProvider::new(
            Arc::clone(&core.provider),
            replay.clone(),
            ModelCallPurpose::Compaction,
        ));
    let compactor = core.compactor.with_provider(recorded_provider);
    let summary_turn = tokio::select! {
        biased;
        // Cancellation after `request_async_compaction()` must clear the
        // pending flag, or check_thresholds_with_available stops returning
        // Async for the whole session life (soft-compaction limbo).
        () = cancellation.cancelled() => {
            mutation.engine_mut().clear_async_compaction_pending();
            None
        }
        summary = mutation.engine_mut().compact(
            Some(&compactor),
            &core.token_budget,
            0,
            failure_mode,
        ) => summary,
    };
    let compacted_conversation = mutation.engine().active_context();
    let summary_node = summary_turn.as_ref().and_then(|turn| {
        let crate::agent::turn::Turn::Summary {
            text,
            source_ids,
            level,
        } = turn
        else {
            return None;
        };
        let node = mutation.engine().dag().newest()?;
        let node_id = node.id;
        let child_ids = node.child_summaries.clone();
        let manifest = node.manifest.clone();
        Some((
            node_id,
            source_ids.clone(),
            child_ids,
            text.clone(),
            TokenBudget::estimate_str_tokens(text),
            *level,
            manifest,
        ))
    });

    // LCM stores only durable conversation rows. Reattach the complete
    // non-durable prompt prefix (system + optional developer) and any
    // same-snapshot ephemeral tail so installing the checkpoint cannot erase
    // instructions or scaffolding that the engine intentionally did not ingest.
    let prompt_prefix_len = messages
        .iter()
        .take_while(|message| {
            matches!(
                message.get("role").and_then(Value::as_str),
                Some("system" | "developer")
            )
        })
        .count();
    let ephemeral_tail = messages[prompt_prefix_len..]
        .iter()
        .filter(|message| {
            message.get("_db_id").is_none()
                && !message
                    .get("_lcm_summary")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
        })
        .cloned();
    let mut compacted_messages =
        Vec::with_capacity(prompt_prefix_len + compacted_conversation.len() + messages.len());
    compacted_messages.extend_from_slice(&messages[..prompt_prefix_len]);
    compacted_messages.extend(compacted_conversation);
    compacted_messages.extend(ephemeral_tail);

    // A failed/no-op soft compaction must preserve the active array byte for
    // byte. Length alone is invalid here because the LCM context deliberately
    // omits system/developer messages.
    let reduced =
        TokenBudget::estimate_tokens(&compacted_messages) < TokenBudget::estimate_tokens(&messages);
    let summary_node_id = summary_node.as_ref().map(|node| node.0);
    let pending = match (summary_node_id, reduced) {
        (Some(summary_node_id), true) => Some(PendingCompaction {
            result: crate::agent::compaction::CompactionResult {
                messages: compacted_messages,
            },
            snapshot: messages,
            summary_node_id,
        }),
        _ => None,
    };

    if pending.is_some() && (cancellation.is_cancelled() || !publication.begin_publication()) {
        let _ = replay.turn_finished("cancelled_before_publication").await;
        return None;
    }

    // Summary nodes, rather than synthetic message rows, persist the DAG. The
    // summary and working-memory snapshot commit together before this function
    // returns the in-memory checkpoint: foreground inference must never install
    // an LCM state that cannot be reconstructed after restart.
    let compacted = summary_turn.is_some();
    let checkpoint_persisted =
        if let (Some((node_id, source_ids, child_ids, text, tokens, level, manifest)), true) =
            (summary_node, pending.is_some())
        {
            let working_memory = core.memory_enabled.then_some((text.as_str(), session_turn));
            match core
                .sessions
                .save_compaction_checkpoint(
                    &session_id,
                    node_id,
                    &source_ids,
                    &child_ids,
                    &text,
                    tokens,
                    level,
                    &manifest,
                    working_memory,
                )
                .await
            {
                Ok(()) => true,
                Err(error) => {
                    warn!(%error, %session_id, node_id, "lcm checkpoint persistence failed");
                    false
                }
            }
        } else {
            !compacted
        };
    if checkpoint_persisted {
        mutation.commit();
    }
    drop(mutation);
    drop(engine);
    // The checkpoint itself is durable (engine commit + SQLite row above),
    // so a failed replay finalization must not discard it. Warn and hand the
    // pending compaction to the foreground; the session's replay degrades to
    // Incomplete, which is exactly what that availability level exists for.
    if let Err(error) = replay
        .turn_finished(if checkpoint_persisted {
            "finished"
        } else {
            "checkpoint_failed"
        })
        .await
    {
        warn!(
            %error,
            %session_id,
            "compaction replay finalization failed; replay degrades to incomplete"
        );
    }
    if checkpoint_persisted {
        pending
    } else {
        None
    }
}
