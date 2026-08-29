// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(clippy::as_conversions, clippy::shadow_reuse, clippy::shadow_same)]
//! Prompt-cache invalidation markers, token accounting, and compaction
//! checkpoint policy.
//!
//! Extracted verbatim from `shared.rs`.

use std::collections::HashSet;

use serde_json::{json, Value};

use crate::agent::agent_core::HiggsSessionControl;
#[cfg(test)]
use crate::agent::agent_core::HiggsSessionReusePolicy;
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

pub(super) fn attach_higgs_session_control(messages: &mut [Value], control: &HiggsSessionControl) {
    if let Some(first) = messages.first_mut().and_then(Value::as_object_mut) {
        first.insert(
            crate::providers::openai_compat::NANOBOT_HIGGS_SESSION_ID_FIELD.to_string(),
            json!(control.active_id),
        );
        match control.drop_ids.as_slice() {
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
        if let Some(lease) = control.session_lease {
            first.insert(
                crate::providers::openai_compat::NANOBOT_HIGGS_SESSION_LEASE_FIELD.to_string(),
                json!({
                    "session_id": lease.session_id,
                    "ttl_seconds": lease.ttl_seconds,
                }),
            );
        }
        first.insert(
            crate::providers::openai_compat::NANOBOT_HIGGS_SESSION_CACHE_POLICY_FIELD.to_string(),
            json!(control.reuse_policy.as_wire()),
        );
        first.insert(
            crate::providers::openai_compat::NANOBOT_HIGGS_MAX_PROMPT_TOKENS_FIELD.to_string(),
            json!(control.max_prompt_tokens),
        );
    }
}

pub(super) fn strip_higgs_session_lease_control(messages: &mut [Value]) {
    if let Some(first) = messages.first_mut().and_then(Value::as_object_mut) {
        first.remove(crate::providers::openai_compat::NANOBOT_HIGGS_SESSION_LEASE_FIELD);
    }
}

#[cfg(test)]
pub(super) fn attach_higgs_session_marker(
    messages: &mut [Value],
    session_id: u64,
    drop_session_ids: &[u64],
) {
    attach_higgs_session_control(
        messages,
        &HiggsSessionControl {
            active_id: session_id,
            drop_ids: drop_session_ids.to_vec(),
            session_lease: None,
            reuse_policy: HiggsSessionReusePolicy::BestEffort,
            max_prompt_tokens: 0,
        },
    );
}

#[cfg(test)]
mod history_window_near_tests {
    use super::history_window_near;

    #[test]
    fn window_is_far_when_messages_and_turns_have_headroom() {
        // 32.8K context → history_limit_lcm = 32800*7/10/150 = 152 messages.
        assert!(!history_window_near(100, 5, 32_768, 60));
    }

    #[test]
    fn message_window_binds_inside_one_turn_of_headroom() {
        // 32.8K context → history_limit_lcm = 152 messages, headroom
        // 152/4 clamped to 16: near from message 136.
        assert!(history_window_near(136, 5, 32_768, 60));
        assert!(!history_window_near(135, 5, 32_768, 60));
    }

    #[test]
    fn turn_window_binds_one_turn_before_the_limit() {
        assert!(history_window_near(10, 59, 32_768, 60));
        assert!(!history_window_near(10, 58, 32_768, 60));
    }

    #[test]
    fn tiny_contexts_keep_plain_drop_behavior() {
        // Harness-sized windows bind every couple of turns; enforcing the
        // ordering there would block-compact on nearly every turn.
        assert!(!history_window_near(19, 1, 4_096, 60));
        assert!(!history_window_near(0, 0, 4_096, 60));
    }
}

#[cfg(test)]
mod lease_control_tests {
    use super::strip_higgs_session_lease_control;
    use serde_json::json;

    #[test]
    fn forced_recovery_messages_strip_only_the_one_shot_lease() {
        let mut messages = vec![json!({
            "role": "system",
            "content": "stable prefix",
            crate::providers::openai_compat::NANOBOT_HIGGS_SESSION_ID_FIELD: 42_u64,
            crate::providers::openai_compat::NANOBOT_HIGGS_SESSION_LEASE_FIELD: {
                "session_id": 41_u64,
                "ttl_seconds": 300_u32,
            },
            crate::providers::openai_compat::NANOBOT_HIGGS_SESSION_CACHE_POLICY_FIELD: "best_effort",
            crate::providers::openai_compat::NANOBOT_HIGGS_MAX_PROMPT_TOKENS_FIELD: 31_744_u32,
        })];

        strip_higgs_session_lease_control(&mut messages);

        assert!(messages[0]
            .get(crate::providers::openai_compat::NANOBOT_HIGGS_SESSION_LEASE_FIELD)
            .is_none());
        assert_eq!(
            messages[0][crate::providers::openai_compat::NANOBOT_HIGGS_SESSION_ID_FIELD],
            json!(42)
        );
        assert_eq!(
            messages[0][crate::providers::openai_compat::NANOBOT_HIGGS_SESSION_CACHE_POLICY_FIELD],
            json!("best_effort")
        );
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

/// Whether the hard history window (`filter_history`'s max-messages /
/// max-turns drop at the next reload) is close enough to bind that a
/// pending compaction must install NOW instead of deferring for the warm
/// cache. If compaction loses this race, the next reload drops aged-out
/// turns WITHOUT summarizing them: the same forced cold prefill, plus
/// permanent content loss — strictly worse than installing now.
///
/// Policy invariant: the token-pressure deferral (`should_allow_checkpoint`)
/// may only hold while the message/turn window cannot bind first.
pub(super) fn history_window_near(
    messages_len: usize,
    turn_count: u64,
    max_context_tokens: usize,
    max_history_turns: usize,
) -> bool {
    // Tiny contexts (test harnesses, toy profiles) have disposable history:
    // their message window binds every couple of turns, so a blocking
    // compaction per turn would be pathological. The ordering invariant only
    // applies where a compaction round has room to matter.
    const MIN_ENFORCE_CONTEXT_TOKENS: usize = 8_192;
    if max_context_tokens < MIN_ENFORCE_CONTEXT_TOKENS {
        return false;
    }
    let max_messages = crate::agent::agent_core::history_limit_lcm(max_context_tokens);
    // Turn headroom proportional to the window, clamped: enough that "near"
    // fires while one turn of tool rounds still fits, never so much that it
    // fences a large window.
    let turn_headroom = (max_messages / 4).clamp(2, 16);
    let message_near = messages_len.saturating_add(turn_headroom) >= max_messages;
    let turn_near = (turn_count as usize).saturating_add(1) >= max_history_turns;
    message_near || turn_near
}

/// Margin under the server-enforced prompt cap for [`overflow_trim_threshold`].
/// Sized to absorb the tokenizer-estimate error: the client estimates with
/// cl100k while local servers tokenize with their own BPE (Qwen), which
/// measured ~4% higher on live content (session 20260827_102059: cl100k
/// passed a prompt the server counted at 3972 vs a 3952 cap). Under-trimming
/// kills the turn with a 400; over-trimming only sheds an old message — so the
/// margin errs conservative.
const OVERFLOW_TRIM_MARGIN: f64 = 0.93;

/// Trim target for server-oracle overflow recovery (see
/// `attempt_overflow_recovery`), as a fraction of the SMALLEST possible
/// prompt cap (window minus the BASE response budget). The server already
/// proved the client estimate wrong for this content — the bias measured up
/// to ~20% on entity-heavy content — so the recovery target sits far enough
/// under every possible cap that even a badly biased estimate lands safely.
pub(super) const OVERFLOW_RECOVERY_MARGIN: f64 = 0.80;

/// Headroom under the server-reported cap for the RATIO-based recovery
/// target (the error carries exact token counts). 10% absorbs the tool-def
/// count the server renders alongside the messages plus per-message density
/// variation between the trimmed tail and the kept head.
pub(super) const OVERFLOW_RECOVERY_HEADROOM: f64 = 0.90;

/// Max server-oracle overflow recoveries per turn. Each attempt strictly
/// shrinks the context (or gives up), so this only bounds the prefill cost
/// of pathological grow-overflow-shrink cycles, not correctness.
pub(super) const MAX_OVERFLOW_RECOVERIES: u32 = 3;

/// Emergency-trim gate for a given prompt cap (context window minus the
/// response budget the request will actually send — what higgs receives as
/// `max_prompt_tokens`). Gating on the raw window instead lets a prompt slip
/// between the client gate and the server cap and die in a
/// `context_length_exceeded` 400 (session 20260827_083227: ~31,054 estimated
/// tokens passed `0.95 × 32,768` and hit the 30,720 server cap). The margin
/// absorbs tokenizer-estimate error — trimming one message early is cheap;
/// overflowing kills the turn.
pub(super) fn overflow_trim_threshold(prompt_cap: usize) -> usize {
    (prompt_cap as f64 * OVERFLOW_TRIM_MARGIN) as usize
}
