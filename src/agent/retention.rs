//! Single owner for message-retention policy.
//!
//! Five mechanisms decide "what stays in `messages[]`" across a turn:
//! `session::filters::filter_history` (DB load), `context_hygiene::hygiene_pipeline`
//! (structural cleanup), `anti_drift::pre_completion_pipeline` (local-only quality
//! cleanup), `LcmEngine` (semantic summarization — a separate job, untouched
//! here), and `token_budget::trim_to_fit_with_age_preserving_prefix` (hard
//! token-budget eviction). This module centralizes the config lookup and call
//! surface for the hygiene/anti-drift/budget trio so callers stop reading
//! `hygiene_keep_last_messages` / `anti_drift` / `max_message_age_turns`
//! directly off `SwappableCore` — without changing any pipeline's internal
//! logic, order, or call site. `LcmEngine` and `filter_history` are out of
//! scope: they run at different points in the turn lifecycle for unrelated
//! reasons (session load, summarization) and are not "retention limits".
//!
//! [`RetentionPolicy::apply_shaping`] wraps the `step_prepare` call site
//! (hygiene, then local-only anti-drift). [`RetentionPolicy::apply_budget`]
//! wraps the `step_pre_call` call site (token-budget trim, normal or
//! emergency). Both read the same policy struct, built once from config in
//! `agent_core::build_swappable_core`.

use serde_json::Value;

use crate::agent::anti_drift;
use crate::agent::context_hygiene;
use crate::agent::token_budget::{PrefixTrimDisposition, TokenBudget};
use crate::config::schema::{AntiDriftConfig, MemoryConfig};

/// Single source of retention knobs, built once from config.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Recent messages kept untruncated by context hygiene
    /// (`memory.hygiene.keepLastMessages`).
    pub keep_last_messages: usize,
    /// Turn-age ceiling past which token-budget trim prefers eviction
    /// (`memory.maxMessageAgeTurns`).
    pub max_message_age_turns: usize,
    /// Quality-based cleanup config for local models (`trio.antiDrift`).
    pub anti_drift: AntiDriftConfig,
}

/// Which budget-trim call site is invoking [`RetentionPolicy::apply_budget`].
///
/// The two pre-extraction call sites in `step_pre_call` pass different
/// arguments to `trim_to_fit_with_age_preserving_prefix`: the normal trim
/// uses the policy's age ceiling and the real turn number, while the
/// emergency (context-overflow) trim deliberately ignores age — ties
/// `tool_def_tokens`/`current_turn`/`max_age_turns` to 0 so it trims more
/// aggressively than the normal path. This enum keeps that distinction
/// explicit at the call site instead of a bare bool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetMode {
    /// Standard pre-call trim: age-aware eviction using this turn's number.
    Normal { turn_count: u64 },
    /// Emergency pre-flight trim when the rendered prompt is about to exceed
    /// the model's context window. Ignores age preference and tool-def
    /// token reservation (matches the original call site's `(0, 0, 0)`).
    Emergency,
}

impl RetentionPolicy {
    /// Build the policy once from config. Called from
    /// `agent_core::build_swappable_core` so every consumer shares one
    /// source of truth instead of reading `MemoryConfig`/`AntiDriftConfig`
    /// fields ad hoc.
    pub fn from_config(memory_config: &MemoryConfig, anti_drift: &AntiDriftConfig) -> Self {
        Self {
            keep_last_messages: memory_config.hygiene.keep_last_messages,
            max_message_age_turns: memory_config.max_message_age_turns,
            anti_drift: anti_drift.clone(),
        }
    }

    /// Structural + quality shaping: hygiene pipeline, then (caller-gated)
    /// anti-drift pipeline. Order and behavior are unchanged from the
    /// pre-extraction call site — only the config lookup is centralized.
    /// `run_anti_drift` mirrors the original
    /// `ctx.core.mode().needs_anti_drift() && ctx.core.anti_drift.enabled`
    /// gate, which depends on runtime mode this module doesn't own.
    pub fn apply_shaping(
        &self,
        messages: &mut Vec<Value>,
        iteration: u32,
        run_anti_drift: bool,
        tools_active: bool,
    ) {
        context_hygiene::hygiene_pipeline(messages, self.keep_last_messages);
        if run_anti_drift {
            anti_drift::pre_completion_pipeline(
                messages,
                iteration,
                &self.anti_drift,
                tools_active,
            );
        }
    }

    /// Token-budget trim, preserving the frozen KV-cache prefix.
    pub fn apply_budget(
        &self,
        token_budget: &TokenBudget,
        messages: &[Value],
        tool_def_tokens: usize,
        mode: BudgetMode,
        frozen_prefix: usize,
    ) -> (Vec<Value>, PrefixTrimDisposition) {
        match mode {
            BudgetMode::Normal { turn_count } => token_budget
                .trim_to_fit_with_age_preserving_prefix(
                    messages,
                    tool_def_tokens,
                    turn_count,
                    self.max_message_age_turns,
                    frozen_prefix,
                ),
            BudgetMode::Emergency => token_budget.trim_to_fit_with_age_preserving_prefix(
                messages,
                0,
                0,
                0,
                frozen_prefix,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn policy() -> RetentionPolicy {
        RetentionPolicy {
            keep_last_messages: 2,
            max_message_age_turns: 5,
            anti_drift: AntiDriftConfig {
                enabled: true,
                anchor_interval: 0,
                pollution_threshold: 0.0, // evict everything text-only for the test
                babble_max_tokens: 500,
                repetition_min_count: 3,
            },
        }
    }

    #[test]
    fn test_from_config_reads_the_three_knobs() {
        let mut memory_config = MemoryConfig {
            max_message_age_turns: 42,
            ..Default::default()
        };
        memory_config.hygiene.keep_last_messages = 9;
        let anti_drift = AntiDriftConfig {
            anchor_interval: 7,
            ..Default::default()
        };

        let p = RetentionPolicy::from_config(&memory_config, &anti_drift);
        assert_eq!(p.max_message_age_turns, 42);
        assert_eq!(p.keep_last_messages, 9);
        assert_eq!(p.anti_drift.anchor_interval, 7);
    }

    #[test]
    fn test_apply_shaping_runs_hygiene_always_and_anti_drift_when_gated() {
        let p = policy();
        // Orphaned tool result (hygiene should remove) + a filler-heavy
        // assistant turn well outside the safe window (anti-drift should evict).
        let mut messages = vec![
            json!({"role": "system", "content": "system"}),
            json!({"role": "tool", "content": "orphan", "tool_call_id": "tc_orphan"}),
            json!({"role": "user", "content": "hello"}),
            json!({"role": "assistant", "content": "Certainly! Absolutely! Of course! Well, basically, honestly, I understand!"}),
            json!({"role": "user", "content": "next"}),
            json!({"role": "assistant", "content": "ok"}),
            json!({"role": "user", "content": "next2"}),
            json!({"role": "assistant", "content": "ok2"}),
        ];

        p.apply_shaping(&mut messages, 1, true, false);

        assert!(
            !messages
                .iter()
                .any(|m| m.get("content").and_then(|c| c.as_str()) == Some("orphan")),
            "hygiene must remove the orphaned tool result"
        );
        assert!(
            messages.iter().any(|m| m
                .get("content")
                .and_then(|c| c.as_str())
                .map(|s| s.contains("[low-quality response removed]"))
                .unwrap_or(false)),
            "gated anti-drift must evict the polluted turn"
        );
    }

    #[test]
    fn test_apply_shaping_skips_anti_drift_when_not_gated() {
        let p = policy();
        let mut messages = vec![
            json!({"role": "system", "content": "system"}),
            json!({"role": "user", "content": "hello"}),
            json!({"role": "assistant", "content": "Certainly! Absolutely! Of course! Well, basically, honestly, I understand!"}),
            json!({"role": "user", "content": "next"}),
            json!({"role": "assistant", "content": "ok"}),
            json!({"role": "user", "content": "next2"}),
        ];
        let before = messages.clone();

        p.apply_shaping(&mut messages, 1, false, false);

        // Hygiene alone is a no-op on this already-clean conversation, and
        // anti-drift never ran, so nothing should have changed.
        assert_eq!(messages, before);
    }

    #[test]
    fn test_apply_budget_normal_uses_policy_age_ceiling() {
        let p = policy();
        let token_budget = TokenBudget::new(2_000, 500);
        let fat = "word ".repeat(1200);
        let mut messages = vec![json!({"role": "system", "content": "sys"})];
        for i in 0..5 {
            messages.push(json!({"role": "user", "content": format!("q{i}")}));
            messages.push(json!({"role": "assistant", "content": fat.clone()}));
        }
        assert!(
            TokenBudget::estimate_tokens(&messages) > 1500,
            "test setup must be over budget"
        );

        let (trimmed, _) = p.apply_budget(
            &token_budget,
            &messages,
            0,
            BudgetMode::Normal { turn_count: 5 },
            0,
        );
        assert!(TokenBudget::estimate_tokens(&trimmed) <= 1500);
    }

    #[test]
    fn test_apply_budget_emergency_matches_zeroed_direct_call() {
        let p = policy();
        let token_budget = TokenBudget::new(2_000, 500);
        let fat = "word ".repeat(1200);
        let mut messages = vec![json!({"role": "system", "content": "sys"})];
        for i in 0..5 {
            messages.push(json!({"role": "user", "content": format!("q{i}")}));
            messages.push(json!({"role": "assistant", "content": fat.clone()}));
        }

        let (via_policy, disp_policy) =
            p.apply_budget(&token_budget, &messages, 0, BudgetMode::Emergency, 0);
        let (direct, disp_direct) =
            token_budget.trim_to_fit_with_age_preserving_prefix(&messages, 0, 0, 0, 0);

        assert_eq!(via_policy, direct);
        assert_eq!(disp_policy, disp_direct);
    }
}
