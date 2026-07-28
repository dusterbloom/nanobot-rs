//! LCM compaction execution: cancellation-safe engine mutation and checkpoint
//! persistence through the foreground provider/model.
//!
//! Extracted verbatim from `shared.rs`.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use serde_json::Value;
use tracing::warn;

use crate::agent::agent_core::{PendingCompaction, SwappableCore};
use crate::agent::lcm::{CompactionFailureMode, LcmCompactionState, LcmEngine};
use crate::agent::token_budget::TokenBudget;

/// Shared handles for background compaction coordination.
#[derive(Clone)]
pub(crate) struct CompactionHandle {
    pub(crate) slot: Arc<tokio::sync::Mutex<Option<PendingCompaction>>>,
    pub(crate) in_flight: Arc<AtomicBool>,
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
    // Keep the engine locked through SQLite commit. A failed checkpoint rolls
    // back the DAG/active window before any other task can observe or extend
    // the non-durable state.
    let mut engine = lcm.lock().await;
    // Stamp the session turn so the new summary node records its creation
    // turn. auto_expand's fresh-summary cooldown uses this to prevent the
    // just-compacted originals from being reinjected on the very next turn
    // (live failure 2026-07-27 12:13:06).
    engine.set_current_turn(session_turn);
    let mut mutation = LcmCompactionMutation::new(&mut engine);
    let summary_turn = mutation
        .engine_mut()
        .compact(Some(&core.compactor), &core.token_budget, 0, failure_mode)
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
    if checkpoint_persisted {
        pending
    } else {
        None
    }
}
