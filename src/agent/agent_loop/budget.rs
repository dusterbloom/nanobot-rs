//! Prompt-cache invalidation markers, token accounting, and higgs
//! retained-session admission / compaction-pressure policy.
//!
//! Extracted verbatim from `shared.rs`.

use std::collections::HashSet;

use serde_json::{json, Value};

use crate::agent::system_state;
use crate::agent::token_budget::TokenBudget;
use crate::turn_stream::{CacheResetReason, CacheStatus, ControlMarker};

use super::shared::TurnContext;

const DEFAULT_HIGGS_RETAINED_SESSION_CAP_TOKENS: usize = 24_576;
const HIGGS_RETAINED_CAP_ENV: &str = "NANOBOT_HIGGS_RETAINED_CAP_TOKENS";
const DEFAULT_HIGGS_RETAINED_ADMISSION_RATIO: f64 = 0.80;
const HIGGS_RETAINED_ADMISSION_RATIO_ENV: &str = "NANOBOT_HIGGS_RETAINED_ADMISSION_RATIO";
const HIGGS_PROMPT_TOKEN_RATIO_CEILING: f64 = 2.50;

#[derive(Clone, Copy, Debug)]
pub(super) struct HiggsRetainedAdmission {
    pub(super) estimated_prompt_tokens: usize,
    pub(super) calibrated_prompt_tokens: usize,
    pub(super) admission_limit_tokens: usize,
    pub(super) observed_token_ratio: f64,
    pub(super) force_blocking: bool,
}

pub(super) fn send_cache_reset_marker(
    tx: &Option<tokio::sync::mpsc::UnboundedSender<String>>,
    reason: CacheResetReason,
) {
    if let Some(tx) = tx {
        let _ = tx.send(ControlMarker::CacheStatus(CacheStatus::Reset { reason }).encode());
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

fn system_token_count(messages: &[Value]) -> usize {
    let system: Vec<Value> = messages
        .iter()
        .filter(|message| message.get("role").and_then(Value::as_str) == Some("system"))
        .cloned()
        .collect();
    TokenBudget::estimate_tokens(&system)
}

/// Retained-session token cap used for admission pressure. Engaged when the
/// provider advertises the higgs retained-session protocol
/// (`supports_higgs_session_cache`, set for any `localBackend=higgs` regardless
/// of port), or overridden explicitly via `HIGGS_RETAINED_CAP_ENV`. The
/// capability flag — not a hardcoded port — decides engagement, so a
/// higgs-nightly server on port 8092 still participates.
pub(super) fn higgs_retained_session_cap_tokens(higgs_capable: bool) -> Option<usize> {
    match std::env::var(HIGGS_RETAINED_CAP_ENV) {
        Ok(raw) => return raw.trim().parse::<usize>().ok().filter(|cap| *cap > 0),
        Err(std::env::VarError::NotPresent) => {}
        Err(std::env::VarError::NotUnicode(_)) => return None,
    }

    higgs_capable.then_some(DEFAULT_HIGGS_RETAINED_SESSION_CAP_TOKENS)
}

fn higgs_retained_admission_ratio() -> f64 {
    match std::env::var(HIGGS_RETAINED_ADMISSION_RATIO_ENV) {
        Ok(raw) => raw
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|ratio| ratio.is_finite() && (0.10..=0.98).contains(ratio))
            .unwrap_or(DEFAULT_HIGGS_RETAINED_ADMISSION_RATIO),
        Err(_) => DEFAULT_HIGGS_RETAINED_ADMISSION_RATIO,
    }
}

fn calibrated_higgs_prompt_tokens(
    estimated_prompt_tokens: usize,
    last_estimated_prompt_tokens: u64,
    last_actual_prompt_tokens: u64,
) -> (usize, f64) {
    if last_estimated_prompt_tokens == 0 || last_actual_prompt_tokens == 0 {
        return (estimated_prompt_tokens, 1.0);
    }

    let observed_ratio = (last_actual_prompt_tokens as f64 / last_estimated_prompt_tokens as f64)
        .clamp(1.0, HIGGS_PROMPT_TOKEN_RATIO_CEILING);
    (
        ((estimated_prompt_tokens as f64) * observed_ratio).ceil() as usize,
        observed_ratio,
    )
}

pub(super) fn higgs_retained_admission(
    retained_cap_tokens: Option<usize>,
    messages: &[Value],
    tool_def_tokens: usize,
    last_estimated_prompt_tokens: u64,
    last_actual_prompt_tokens: u64,
) -> Option<HiggsRetainedAdmission> {
    let cap = retained_cap_tokens?;
    if cap == 0 {
        return None;
    }

    let estimated_prompt_tokens =
        TokenBudget::estimate_tokens(messages).saturating_add(tool_def_tokens);
    let (calibrated_prompt_tokens, observed_token_ratio) = calibrated_higgs_prompt_tokens(
        estimated_prompt_tokens,
        last_estimated_prompt_tokens,
        last_actual_prompt_tokens,
    );
    let admission_limit_tokens = ((cap as f64) * higgs_retained_admission_ratio()).floor() as usize;

    Some(HiggsRetainedAdmission {
        estimated_prompt_tokens,
        calibrated_prompt_tokens,
        admission_limit_tokens,
        observed_token_ratio,
        force_blocking: calibrated_prompt_tokens >= admission_limit_tokens,
    })
}

fn retained_conversation_available(
    retained_cap_tokens: usize,
    messages: &[Value],
    tool_def_tokens: usize,
) -> usize {
    retained_cap_tokens
        .saturating_sub(system_token_count(messages))
        .saturating_sub(tool_def_tokens)
}

pub(super) fn effective_lcm_available_budget(
    model_available: usize,
    messages: &[Value],
    tool_def_tokens: usize,
    retained_cap_tokens: Option<usize>,
) -> (usize, Option<usize>) {
    let Some(retained_cap_tokens) = retained_cap_tokens else {
        return (model_available, None);
    };
    let retained_available =
        retained_conversation_available(retained_cap_tokens, messages, tool_def_tokens);
    // LCM thresholds use the MODEL's full context budget, NOT the
    // retained-session cap. The retained cap (24K on default higgs
    // config) was clamping LCM to compact at 12K conversation tokens
    // even on 120K-context models — suffocating legitimate long
    // sessions. The retained-admission check (separate, at shared.rs)
    // still forces blocking compaction when the retained session is
    // under pressure, so the safety net is preserved. Cold-prefills
    // above the retained cap are the accepted tradeoff.
    (model_available, Some(retained_available))
}

pub(super) fn retained_context_pressure(
    retained_cap_tokens: Option<usize>,
    messages: &[Value],
    tool_def_tokens: usize,
) -> Option<f32> {
    let cap = retained_cap_tokens?;
    if cap == 0 {
        return None;
    }
    let used = TokenBudget::estimate_tokens(messages).saturating_add(tool_def_tokens);
    Some((used as f32) / (cap as f32))
}

/// Decide whether to accept the prefix-cache re-prefill cost and install a
/// pending compaction result rather than deferring it.
///
/// Below `tau_hard` the cache-warm deferral is honoured. At or above
/// `tau_hard` compaction is forced through: the one-time re-prefill of a
/// compacted context is strictly cheaper than the unbounded growth that
/// starves compaction until the model hits max tokens and fails.
pub(super) fn should_allow_checkpoint(
    pressure: f32,
    tau_hard: f64,
    retained_pressure: Option<f32>,
) -> bool {
    (pressure as f64) >= tau_hard
        || retained_pressure.is_some_and(|pressure| (pressure as f64) >= tau_hard)
}
