//! LCM compaction execution: sidecar lease acquisition, cancellation-safe
//! engine mutation, and checkpoint persistence.
//!
//! Extracted verbatim from `shared.rs`.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use tracing::warn;

use crate::agent::agent_core::{PendingCompaction, SwappableCore};
use crate::agent::compaction::ContextCompactor;
use crate::agent::lcm::{CompactionFailureMode, LcmCompactionState, LcmEngine};
use crate::agent::token_budget::TokenBudget;

/// Shared handles for background compaction coordination.
pub(crate) struct CompactionHandle {
    pub(crate) slot: Arc<tokio::sync::Mutex<Option<PendingCompaction>>>,
    pub(crate) in_flight: Arc<AtomicBool>,
}

/// Build a compactor bound directly to the main provider/model, bypassing a
/// managed compaction sidecar that is unreachable (or was never configured
/// for a fresh attempt). Reuses the sidecar-tuned context-window budget so
/// summarization chunking stays unchanged; only the endpoint and served
/// model differ. Never `None` — this is what keeps compaction from silently
/// degrading to truncation when the sidecar is down.
fn main_provider_compactor(core: &SwappableCore) -> ContextCompactor {
    ContextCompactor::new(
        core.provider.clone(),
        core.model.clone(),
        core.compactor.context_size(),
    )
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

const COMPACTION_ACQUIRE_TIMEOUT: Duration = Duration::from_secs(75);

/// Execute one LCM compaction and persist its lossless summary metadata.
/// The caller decides whether to await this future (hard pressure) or spawn it
/// (soft pressure); keeping the operation itself shared prevents the two paths
/// from drifting.
pub(super) async fn execute_lcm_compaction(
    core: Arc<SwappableCore>,
    session_id: String,
    lcm: Arc<tokio::sync::Mutex<LcmEngine>>,
    messages: Vec<Value>,
    session_turn: u64,
    failure_mode: CompactionFailureMode,
) -> Option<PendingCompaction> {
    let (lease, resolved_compactor): (_, ContextCompactor) = match core.compaction_manager.as_ref()
    {
        Some(manager) => {
            match tokio::time::timeout(COMPACTION_ACQUIRE_TIMEOUT, manager.acquire()).await {
                Ok(Ok(lease)) => {
                    let compactor = core.compactor.for_model(lease.served_model().to_string());
                    (Some(lease), compactor)
                }
                Ok(Err(error)) => {
                    tracing::warn!(
                        %error,
                        ?failure_mode,
                        model = %core.model,
                        "compaction sidecar unavailable — compacting with main model"
                    );
                    (None, main_provider_compactor(&core))
                }
                Err(_) => {
                    tracing::warn!(
                        timeout_secs = COMPACTION_ACQUIRE_TIMEOUT.as_secs(),
                        ?failure_mode,
                        model = %core.model,
                        "compaction sidecar unavailable — compacting with main model"
                    );
                    (None, main_provider_compactor(&core))
                }
            }
        }
        // No managed sidecar configured at all — the core's base
        // compactor already resolves to the right endpoint (explicit
        // memory.provider / specialist / main provider).
        None => (None, core.compactor.clone()),
    };

    // Keep the engine locked through SQLite commit. A failed checkpoint rolls
    // back the DAG/active window before any other task can observe or extend
    // the non-durable state.
    let mut engine = lcm.lock().await;
    let mut mutation = LcmCompactionMutation::new(&mut engine);
    let summary_turn = mutation
        .engine_mut()
        .compact(
            Some(&resolved_compactor),
            &core.token_budget,
            0,
            failure_mode,
        )
        .await;
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
        Some((
            node_id,
            source_ids.clone(),
            child_ids,
            text.clone(),
            TokenBudget::estimate_str_tokens(text),
            *level,
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
    let pending = (summary_turn.is_some() && reduced).then(|| PendingCompaction {
        result: crate::agent::compaction::CompactionResult {
            messages: compacted_messages,
        },
        snapshot: messages,
    });

    // Summary nodes, rather than synthetic message rows, persist the DAG. The
    // summary and working-memory snapshot commit together before this function
    // returns the in-memory checkpoint: foreground inference must never install
    // an LCM state that cannot be reconstructed after restart.
    let compacted = summary_turn.is_some();
    let checkpoint_persisted =
        if let (Some((node_id, source_ids, child_ids, text, tokens, level)), true) =
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
    // Drop is the cancellation-safe release path; it schedules last-lease
    // shutdown without delaying foreground checkpoint installation.
    drop(lease);
    if checkpoint_persisted {
        pending
    } else {
        None
    }
}
