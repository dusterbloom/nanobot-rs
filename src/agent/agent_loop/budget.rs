// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(
    clippy::as_conversions,
    clippy::shadow_reuse,
    clippy::shadow_same,
)]
//! Prompt-cache invalidation markers, token accounting, and compaction
//! checkpoint policy.
//!
//! Extracted verbatim from `shared.rs`.

use std::collections::HashSet;

use serde_json::{json, Value};

use crate::agent::system_state;
use crate::agent::token_budget::TokenBudget;
use crate::turn_stream::{CacheResetReason, CacheStatus, ControlMarker};

use super::shared::TurnContext;

pub(super) fn send_cache_reset_marker(
    tx: &Option<tokio::sync::mpsc::UnboundedSender<String>>,
    reason: CacheResetReason,
) {
    if let Some(tx) = tx {
        let _ = tx.send(ControlMarker::CacheStatus(CacheStatus::Reset { reason }).encode());
    }
}

/// Emit a compaction lifecycle marker so the TUI can render a live progress
/// indicator instead of a silent freeze. See `CompactionStatus` for the
/// state machine.
pub(super) fn send_compaction_marker(
    tx: &Option<tokio::sync::mpsc::UnboundedSender<String>>,
    status: crate::turn_stream::CompactionStatus,
) {
    if let Some(tx) = tx {
        let _ = tx.send(ControlMarker::Compaction(status).encode());
    }
}

pub(super) fn clear_prompt_cache_state(ctx: &TurnContext) -> bool {
    ctx.counters.clear_local_prompt_cache(&ctx.session_key)
}

/// Invalidate prompt-cache state after a sanctioned prompt rewrite (budget
/// trim, emergency trim, LCM compaction). When the provider supports the higgs
/// retained-session protocol, the retained session is ROTATED (epoch bump +
/// queued drop) so the server cold-starts the rewritten prompt instead of
/// rejecting a shrunken prompt as "not_growing" and re-prefilling under the
/// stale session id. Otherwise only local bookkeeping is cleared. The system
/// message is never mutated — rotation alone invalidates the cache.
pub(super) fn invalidate_prompt_cache_for_rewrite(
    ctx: &mut TurnContext,
    reason: CacheResetReason,
) -> bool {
    let rotate = ctx.core.mode().is_local() && ctx.core.provider.supports_higgs_session_cache();
    ctx.counters
        .invalidate_prompt_cache(&ctx.session_key, rotate);
    ctx.counters
        .note_cache_reset(&ctx.session_key, reason.as_wire());
    send_cache_reset_marker(&ctx.text_delta_tx, reason);
    rotate
}

pub(super) fn send_retract_reply_marker(tx: &Option<tokio::sync::mpsc::UnboundedSender<String>>) {
    if let Some(tx) = tx {
        let _ = tx.send(ControlMarker::RetractReply.encode());
    }
}

/// Classify a divergent message for the `prompt_prefix_diverged` WARN: role +
/// structural kind tags + a truncated content snippet. Names the cache-busting
/// class (synthetic / cache-replay / lcm-summary / tool-result / persisted) so
/// the higgs `token_mismatch` cause is identifiable from the log line alone
/// without a separate message dump.
///
/// After the trim/compaction fixes, every sanctioned rewrite clears the prompt
/// fingerprint, so a `PromptDelta::Diverged` is by definition UNSANCTIONED — a
/// message whose rendered bytes changed across turns under the same session id.
/// This digest names which structural kind diverged so the root cause (e.g. a
/// transient synthetic that was sent but not replayed, or a re-summarized LCM
/// block) is obvious.
pub(super) fn divergent_message_digest(msg: &Value) -> String {
    let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("?");
    let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
    let snippet: String = content.chars().take(70).collect();

    let mut tags: Vec<&str> = Vec::new();
    if msg
        .get("_synthetic")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        tags.push("synthetic");
    }
    if msg
        .get("_cache_replay")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        tags.push("cache-replay");
    }
    if msg
        .get("_lcm_summary")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        tags.push("lcm-summary");
    }
    if msg.get("tool_call_id").is_some() {
        tags.push("tool-result");
    }
    if msg.get("_db_id").is_some() {
        tags.push("persisted");
    }

    if tags.is_empty() {
        format!("[{role}] {snippet}")
    } else {
        format!("[{role}] ({}) {snippet}", tags.join(","))
    }
}

pub(super) fn attach_higgs_session_marker(
    messages: &mut [Value],
    session_id: u64,
    drop_session_ids: &[u64],
) {
    if let Some(first) = messages.first_mut().and_then(Value::as_object_mut) {
        first.insert(
            crate::providers::openai_compat::NANOBOT_HIGGS_SESSION_ID_FIELD.to_string(),
            json!(session_id),
        );
        match drop_session_ids {
            [] => {}
            [drop_session_id] => {
                first.insert(
                    crate::providers::openai_compat::NANOBOT_HIGGS_DROP_SESSION_ID_FIELD
                        .to_string(),
                    json!(drop_session_id),
                );
            }
            drop_session_ids => {
                first.insert(
                    crate::providers::openai_compat::NANOBOT_HIGGS_DROP_SESSION_IDS_FIELD
                        .to_string(),
                    json!(drop_session_ids),
                );
            }
        }
    }
}

pub(super) fn proactive_grounding_preserves_prefix_cache(is_local: bool) -> bool {
    !is_local
}

/// Decide whether to inject a heartbeat `[grounding]` message this iteration.
///
/// Pure policy: combines the existing cadence/pressure rule (`should_ground`)
/// with the prefix-cache preservation rule (`proactive_grounding_preserves_prefix_cache`).
/// Local models skip heartbeat grounding because every synthetic user turn
/// diverges the warm prefix cache.
pub(super) fn should_inject_heartbeat_grounding(
    iteration: u32,
    interval: u32,
    pressure: f32,
    is_local: bool,
) -> bool {
    system_state::should_ground(iteration, interval, pressure)
        && proactive_grounding_preserves_prefix_cache(is_local)
}

pub(super) fn conversation_token_count(messages: &[Value]) -> usize {
    let conversation: Vec<Value> = messages
        .iter()
        .filter(|message| message.get("role").and_then(Value::as_str) != Some("system"))
        .cloned()
        .collect();
    TokenBudget::estimate_tokens(&conversation)
}

pub(super) fn advertised_tool_names(tool_defs: &[Value]) -> HashSet<String> {
    tool_defs
        .iter()
        .filter_map(|def| {
            def.pointer("/function/name")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .collect()
}

/// Decide whether to accept the prefix-cache re-prefill cost and install a
/// pending compaction result rather than deferring it.
///
/// Below `tau_hard` the cache-warm deferral is honoured. At or above
/// `tau_hard` compaction is forced through: the one-time re-prefill of a
/// compacted context is strictly cheaper than the unbounded growth that
/// starves compaction until the model hits max tokens and fails.
pub(super) fn should_allow_checkpoint(pressure: f32, tau_hard: f64) -> bool {
    (pressure as f64) >= tau_hard
}
