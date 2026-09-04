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

/// Trim margin for server-oracle overflow recovery (see
/// `attempt_overflow_recovery`), as a fraction of the prompt cap the failed
/// request actually sent: the context window minus the call's
/// `effective_max_tokens`. The server already proved the client estimate
/// wrong for this content — the bias measured up to ~20% on entity-heavy
/// content — so the recovery target sits far enough under the call's real
/// cap that even a badly biased estimate lands safely. Computing this from
/// the *base* `ctx.core.max_tokens` while the failed request was sent with
/// the widened *effective* budget leaves the target above the server's true
/// prompt cap (`window − effective`), which makes `apply_emergency_trim`'s
/// no-shrink guard short-circuit and turns the overflow into a fatal
/// `context_length_exceeded` 400 — see [`overflow_recovery_fallback_budget`].
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

/// `None`-arm fallback target for `attempt_overflow_recovery`: the context
/// window minus the failed call's `effective_max_tokens`, scaled by
/// [`OVERFLOW_RECOVERY_MARGIN`].
///
/// The failed request was sent with `max_tokens = effective_max_tokens`
/// (possibly widened above the static base `ctx.core.max_tokens` by `/long`,
/// a long-form prompt, a local Rich-artifact action, or a thinking budget),
/// so the server enforced the real prompt cap `window − effective` — not
/// `window − base`. Deriving the target from the base budget instead can
/// leave it *above* that real cap, which makes `apply_emergency_trim`'s
/// no-shrink guard (`rendered_before <= message_budget`) short-circuit and
/// turns the overflow into a fatal `context_length_exceeded` 400 even though
/// trimming further would fit. Mirrors how the preflight gate
/// ([`overflow_trim_threshold`]) derives its cap from the same effective
/// budget; when `effective_max_tokens == base` the result is identical to
/// the old base-derived computation, so unwidened turns are unaffected.
pub(super) fn overflow_recovery_fallback_budget(
    max_context: usize,
    effective_max_tokens: u32,
) -> usize {
    ((max_context.saturating_sub(effective_max_tokens as usize)) as f64 * OVERFLOW_RECOVERY_MARGIN)
        as usize
}

#[cfg(test)]
mod overflow_recovery_fallback_budget_tests {
    use super::{
        overflow_recovery_fallback_budget, overflow_trim_threshold, OVERFLOW_RECOVERY_MARGIN,
    };

    // Production defaults: base max_tokens B = 4096, adaptive_long_mode_min_tokens
    // = 12288 (what /long and a local Rich-artifact action widen `effective` to),
    // and the local default window is 32,768.
    const BASE: u32 = 4096;
    const EFFECTIVE_LONG: u32 = 12288;
    const LOCAL_WINDOW: usize = 32_768;

    #[test]
    fn uses_effective_max_tokens_so_target_lands_under_the_real_cap() {
        // The failed request sent max_tokens = EFFECTIVE_LONG, so the server's
        // real prompt cap is window − effective, NOT window − base.
        let real_cap = LOCAL_WINDOW.saturating_sub(EFFECTIVE_LONG as usize);
        assert_eq!(real_cap, 20_480);

        let target = overflow_recovery_fallback_budget(LOCAL_WINDOW, EFFECTIVE_LONG);
        assert_eq!(
            target,
            ((LOCAL_WINDOW - EFFECTIVE_LONG as usize) as f64 * OVERFLOW_RECOVERY_MARGIN) as usize
        );
        assert_eq!(target, 16_384);
        assert!(
            target < real_cap,
            "fixed target must sit under the server's real prompt cap: {target} >= {real_cap}"
        );
    }

    #[test]
    fn guard_no_longer_short_circuits_on_the_firing_window() {
        // After the preflight gate trims, rendered_before sits at the preflight
        // clamp = (window − effective) · OVERFLOW_TRIM_MARGIN. apply_emergency_trim
        // short-circuits (refuses to trim) when rendered_before <= message_budget,
        // which with the BUGGY base-derived target made recovery bail to a fatal
        // 400 on the local / non-higgs firing window.
        let clamp = overflow_trim_threshold(LOCAL_WINDOW.saturating_sub(EFFECTIVE_LONG as usize));
        assert_eq!(clamp, 19_046);

        // Fixed (effective-derived) target is below the clamp → trim proceeds.
        let fixed = overflow_recovery_fallback_budget(LOCAL_WINDOW, EFFECTIVE_LONG);
        assert!(
            fixed < clamp,
            "fixed target must sit below the preflight clamp so the no-shrink guard fires the \
             trim: {fixed} >= {clamp}"
        );

        // Regression guard: the pre-fix base-derived target overshot BOTH the
        // clamp and the real cap — exactly the bug.
        let pre_fix = ((LOCAL_WINDOW.saturating_sub(BASE as usize)) as f64
            * OVERFLOW_RECOVERY_MARGIN) as usize;
        assert_eq!(pre_fix, 22_937);
        assert!(
            pre_fix > clamp,
            "pre-fix target sat above the preflight clamp: the no-shrink guard short-circuited \
             (rendered_before ({clamp}) <= {pre_fix})"
        );
        assert!(
            pre_fix > LOCAL_WINDOW.saturating_sub(EFFECTIVE_LONG as usize),
            "pre-fix target sat above the real cap: even a successful trim would 400 again"
        );
    }

    #[test]
    fn unchanged_when_effective_equals_base() {
        // No widening (E == B): the fix is byte-identical to the old
        // base-derived computation, so unwidened turns keep their behaviour.
        let window = 32_768usize;
        let budget: u32 = 2_048;
        let fixed = overflow_recovery_fallback_budget(window, budget);
        let old =
            ((window.saturating_sub(budget as usize)) as f64 * OVERFLOW_RECOVERY_MARGIN) as usize;
        assert_eq!(fixed, old);
        assert_eq!(fixed, 24_576);
    }

    #[test]
    fn target_always_sits_under_the_real_cap_and_preflight_clamp() {
        // General invariant: as a fraction (0.80) of (window − effective), the
        // target is always strictly under the real cap (window − effective)
        // and under the preflight clamp (window − effective) · 0.93 whenever
        // that cap is positive — so the trimmed retry fits and the no-shrink
        // guard does not short-circuit.
        for (window, effective) in [
            (128_000usize, 12_288u32), // cloud default + /long (non-firing row)
            (32_768, 12_288),          // local default + /long (firing row)
            (32_768, 6_144),           // local default + long-form prompt
            (65_536, 12_288),          // mid-size cloud + /long
            (8_192, 4_096),            // tiny harness window
        ] {
            let real_cap = window.saturating_sub(effective as usize);
            assert!(real_cap > 0, "test vector must leave room for a prompt");
            let target = overflow_recovery_fallback_budget(window, effective);
            assert!(
                target < real_cap,
                "({window}, {effective}): target {target} must sit under real cap {real_cap}"
            );
            let clamp = overflow_trim_threshold(real_cap);
            assert!(
                target < clamp,
                "({window}, {effective}): target {target} must sit under preflight clamp {clamp} \
                 so the no-shrink guard does not short-circuit"
            );
        }
    }

    #[test]
    fn saturates_to_zero_when_effective_budget_meets_or_exceeds_window() {
        // When the response budget consumes the whole window no prompt can be
        // sent: saturating_sub clamps to zero, the target is zero, and the
        // no-shrink guard (rendered_before <= 0) leaves the turn to the normal
        // error path — no panic, no unsigned wraparound.
        assert_eq!(overflow_recovery_fallback_budget(8_192, 8_192), 0);
        assert_eq!(overflow_recovery_fallback_budget(8_192, 16_384), 0);
    }
}
