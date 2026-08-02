//! Tool leases — structural loop prevention.
//!
//! See `docs/superpowers/specs/2026-07-27-tool-leases-design.md`.
//!
//! Each user turn starts with a tool lease of `TOOLS_PER_LEASE` tool
//! iterations. After exhaustion, the model must produce a final text answer or
//! emit a renewal checkpoint (findings + remaining question + next bounded
//! actions). Together with `ToolGuard`'s per-key identical-call counter (which
//! bounds the live 13-call identical-`exec grep` loop, session
//! `20260727_161730_6e61a0`) and the no-progress hard stop, this makes runaway
//! tool loops structurally impossible without a coarse-family cap (retired
//! 2026-07-30 — it over-fired on legitimate exploration and busted the prefix
//! cache).

/// Default per-lease tool budget. Tuned for coding tasks where the
/// model needs to read multiple files, run searches, and exec commands
/// in one turn. 12 is enough for a typical "explore 3-4 files +
/// summarize" workflow without hitting the cap. The original 5 was
/// too tight and suffocated legitimate exploration on 120K-context
/// models.
pub const DEFAULT_TOOLS_PER_LEASE: u32 = 12;

/// Default cap on lease renewals per turn. Three leases × twelve tools
/// = 36 tool calls per turn at maximum, each renewal gated by a
/// validated checkpoint.
pub const DEFAULT_MAX_LEASES_PER_TURN: u32 = 3;

// ---------------------------------------------------------------------------
// Lease state machine
// ---------------------------------------------------------------------------
// Lease state machine
// ---------------------------------------------------------------------------

/// A per-turn tool lease. Counts tool iterations within the current
/// lease; on exhaustion the caller requires the model to either answer or emit
/// a renewal checkpoint without changing the advertised tool schema.
///
/// Renewal checkpoints must contain all three of `findings`, `next`,
/// `will` (any case, colon-terminated). Anything else is rejected with
/// the missing field named — small models produce messy text, so the
/// validator is regex-tolerant but the contract is strict: a renewal
/// must say what was learned, what's missing, and what's planned.
#[derive(Debug, Clone)]
pub struct Lease {
    lease_size: u32,
    max_renewals: u32,
    iterations_used: u32,
    renewals_used: u32,
}

/// Atomic admission outcome for one assistant tool-call batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchAdmission {
    Admitted,
    Rejected { remaining: u32 },
}

/// Outcome of `record_tool_call`. The `reason` is a stable machine-readable
/// code (`lease_exhausted`) so callers can route receipts without parsing
/// prose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallResult {
    pub allowed: bool,
    pub reason: Option<&'static str>,
}

impl ToolCallResult {
    fn allowed() -> Self {
        Self {
            allowed: true,
            reason: None,
        }
    }
    fn blocked(reason: &'static str) -> Self {
        Self {
            allowed: false,
            reason: Some(reason),
        }
    }
}

/// Outcome of `try_renew`. `missing_field` is `"findings"`, `"next"`,
/// `"will"`, or `"out_of_leases"` — the caller renders it into a
/// human-readable rejection that names exactly what's missing.
///
/// `attempted` distinguishes a real checkpoint attempt (some labels
/// present) from plain final text (no labels at all). An exhausted lease
/// must let the model choose to stop: nagging every text response would,
/// once the strip is made sticky, loop until max iterations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenewalResult {
    valid: bool,
    missing_field: &'static str,
    attempted: bool,
}

impl RenewalResult {
    fn accepted() -> Self {
        Self {
            valid: true,
            missing_field: "",
            attempted: true,
        }
    }
    fn rejected(missing: &'static str) -> Self {
        Self {
            valid: false,
            missing_field: missing,
            attempted: true,
        }
    }
    /// Plain text with no checkpoint labels — the model chose to answer,
    /// not to renew. `missing_field` is empty so this is distinct from
    /// both a field rejection and the `"out_of_leases"` cap.
    fn not_attempted() -> Self {
        Self {
            valid: false,
            missing_field: "",
            attempted: false,
        }
    }
    pub fn is_valid(&self) -> bool {
        self.valid
    }
    pub fn missing_field(&self) -> &str {
        self.missing_field
    }
    /// Did the text look like a checkpoint at all? Callers use this to
    /// separate "model tried to renew but missed a field" (nudge with the
    /// missing field) from "model just wrote its final answer" (let it
    /// finish).
    pub fn was_attempted(&self) -> bool {
        self.attempted
    }
}

impl Lease {
    pub fn new(lease_size: u32, max_renewals: u32) -> Self {
        Self {
            lease_size: lease_size.max(1),
            max_renewals,
            iterations_used: 0,
            renewals_used: 0,
        }
    }

    /// Admit a complete assistant tool-call batch or reject it without
    /// consuming the remaining lease allowance.
    pub fn admit_batch(&mut self, count: u32) -> BatchAdmission {
        let remaining = self.lease_size.saturating_sub(self.iterations_used);
        if count > remaining {
            return BatchAdmission::Rejected { remaining };
        }
        self.iterations_used = self.iterations_used.saturating_add(count);
        BatchAdmission::Admitted
    }

    /// Prefix a prompt-visible tool result with the post-batch lease state.
    pub fn annotate_result(&self, body: &str) -> String {
        let renewals_remaining = self.max_renewals.saturating_sub(self.renewals_used);
        let mut signal = format!(
            "[Lease usage after this batch: {} of {} calls — {} renewals remaining.",
            self.iterations_used, self.lease_size, renewals_remaining
        );
        if self.is_exhausted() {
            signal.push_str(
                " Lease exhausted: your next response must be either a final answer or a \
                 renewal checkpoint containing findings:/next:/will:. Do not request \
                 another tool before renewal.",
            );
        }
        signal.push(']');
        format!("{signal}\n{body}")
    }

    /// Record one tool call against the per-lease budget. Returns whether the
    /// call is allowed; once `lease_size` calls have been used this returns
    /// `lease_exhausted`. Retained temporarily while production callers migrate
    /// to atomic batch admission.
    ///
    /// There is NO consecutive-same-family cap anymore: it over-fired on
    /// legitimate exploration (N different greps) and busted the prompt-prefix
    /// cache when it triggered a strip (2026-07-30). Identical-call loops are
    /// bounded by `ToolGuard`'s per-key `seen` counter (tool_guard.rs, applies
    /// to all tools); different-args exploration is bounded by this per-lease
    /// budget plus the no-progress hard stop. See
    /// docs/superpowers/plans/2026-07-30-reuse-not-rerun-tool-dedup.md.
    pub fn record_tool_call(&mut self) -> ToolCallResult {
        if self.iterations_used >= self.lease_size {
            return ToolCallResult::blocked("lease_exhausted");
        }
        self.iterations_used += 1;
        ToolCallResult::allowed()
    }

    pub fn is_exhausted(&self) -> bool {
        self.iterations_used >= self.lease_size
    }

    pub fn renewals_used(&self) -> u32 {
        self.renewals_used
    }

    /// Configured per-lease tool budget. Exposed so the renewal nudge
    /// can tell the model exactly how many calls it has after renewal.
    pub fn lease_size(&self) -> u32 {
        self.lease_size
    }

    /// Try to renew the lease with a model-emitted checkpoint. The
    /// checkpoint must contain `findings:`, `next:`, and `will:` (any
    /// case). Returns `RenewalResult::accepted()` and resets the
    /// iteration budget on success; on failure returns which field is
    /// missing.
    ///
    /// Text with no checkpoint labels at all returns `not_attempted()`
    /// — the model is writing a final answer, not requesting more tools,
    /// and the lease must allow that exit.
    pub fn try_renew(&mut self, checkpoint: &str) -> RenewalResult {
        if self.renewals_used >= self.max_renewals {
            return RenewalResult::rejected("out_of_leases");
        }
        let lower = checkpoint.to_lowercase();
        let has_findings = ["findings:", "learned:", "discovered:"]
            .iter()
            .any(|k| lower.contains(k));
        let has_next = ["next:", "still need:"].iter().any(|k| lower.contains(k));
        let has_will = ["will:", "plan:"].iter().any(|k| lower.contains(k));
        // No checkpoint labels = plain final text. Treat as a choice to
        // stop, not a failed renewal.
        if !has_findings && !has_next && !has_will {
            return RenewalResult::not_attempted();
        }
        let missing = if !has_findings {
            "findings"
        } else if !has_next {
            "next"
        } else if !has_will {
            "will"
        } else {
            ""
        };
        if !missing.is_empty() {
            return RenewalResult::rejected(missing);
        }
        self.renewals_used += 1;
        self.iterations_used = 0;
        RenewalResult::accepted()
    }

    /// Format the progress signal for inclusion in tool results.
    ///
    /// `L = max_renewals - renewals_used` (future leases still obtainable
    /// this turn). The call index `N` is `iterations_used` — the call that
    /// just executed (record_tool_call* was already called by the time
    /// tool_engine adds the result). For the first call this is 1, etc.
    pub fn progress_signal(&self) -> String {
        let current_call = self.iterations_used;
        let leases_remaining = self.max_renewals.saturating_sub(self.renewals_used);
        format!(
            "[Tool call {} of {} this lease — {} leases remaining]",
            current_call, self.lease_size, leases_remaining
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Advance the lease by one call. Used by the state-machine tests below to
    /// exercise exhaustion/renewal/progress in isolation.
    fn tick(lease: &mut Lease) -> bool {
        lease.admit_batch(1) == BatchAdmission::Admitted
    }

    // -----------------------------------------------------------------
    // RED tests — write first, watch fail, then GREEN each
    // -----------------------------------------------------------------

    // -----------------------------------------------------------------
    // Lease state machine
    // -----------------------------------------------------------------

    /// A fresh lease allows `lease_size` tool iterations. The
    /// `(iteration + 1) == lease_size`-th call must report exhaustion
    /// (tools should be stripped by the caller on the next LLM request).
    #[test]
    fn lease_allows_n_tool_calls_then_reports_exhausted() {
        let mut lease = Lease::new(3, 2);
        assert!(tick(&mut lease));
        assert!(tick(&mut lease));
        assert!(tick(&mut lease));
        // Three tools consumed the lease; the 4th call must be rejected.
        assert!(
            !tick(&mut lease),
            "lease must be exhausted after lease_size tool calls"
        );
        assert!(lease.is_exhausted());
    }

    /// After exhaustion, the model must either answer or request a
    /// renewal. A valid checkpoint (findings + next + will) re-arms the
    /// lease for another `lease_size` iterations.
    #[test]
    fn lease_renewal_restores_budget_when_checkpoint_is_valid() {
        let mut lease = Lease::new(2, 3);
        tick(&mut lease);
        tick(&mut lease);
        assert!(lease.is_exhausted());

        let checkpoint = "Findings: located tau_soft in config/schema.rs.\n\
                          Next: need to verify the default value.\n\
                          Will: grep for default_lcm_tau_soft.";
        assert!(
            lease.try_renew(checkpoint).is_valid(),
            "valid checkpoint (findings + next + will) must renew the lease"
        );
        assert!(!lease.is_exhausted());
        assert_eq!(lease.renewals_used(), 1);
    }

    /// Renewal is the only synth-injection-free checkpoint mechanism —
    /// it must reject checkpoints that lack findings, next, or will.
    /// Each rejection path is named so the model can correct.
    #[test]
    fn lease_renewal_rejected_when_any_required_field_is_missing() {
        let mut lease = Lease::new(1, 3);
        tick(&mut lease);
        assert!(lease.is_exhausted());

        // Missing findings.
        let no_findings = "Next: still need context size.\nWill: grep for it.";
        let result = lease.try_renew(no_findings);
        assert!(
            !result.is_valid(),
            "checkpoint without findings must be rejected"
        );
        assert!(
            result.missing_field().contains("findings"),
            "rejection must name 'findings' as missing, got: {:?}",
            result.missing_field()
        );

        // Missing next.
        let no_next = "Findings: located the file.\nWill: grep again.";
        let result = lease.try_renew(no_next);
        assert!(!result.is_valid());
        assert!(result.missing_field().contains("next"));

        // Missing will.
        let no_will = "Findings: located the file.\nNext: still need value.";
        let result = lease.try_renew(no_will);
        assert!(!result.is_valid());
        assert!(result.missing_field().contains("will"));

        // Lease is still exhausted after every rejection.
        assert!(lease.is_exhausted());
        assert_eq!(lease.renewals_used(), 0);
    }

    /// The lease count is capped: even with valid checkpoints, the model
    /// cannot loop indefinitely. After `max_renewals` renewals, further
    /// renewals are rejected with `out_of_leases`.
    #[test]
    fn lease_renewal_capped_at_max_renewals() {
        let mut lease = Lease::new(1, 2);
        for lease_num in 0..2 {
            tick(&mut lease);
            assert!(lease.is_exhausted());
            let valid = format!(
                "Findings: did step {lease_num}.\nNext: step {}.\nWill: continue.",
                lease_num + 1
            );
            assert!(
                lease.try_renew(&valid).is_valid(),
                "renewal {} of {} must succeed with valid checkpoint",
                lease_num + 1,
                2
            );
        }
        // Used both renewals; the third must be rejected for lease-count
        // reasons, not checkpoint-quality reasons.
        tick(&mut lease);
        assert!(lease.is_exhausted());
        let valid = "Findings: x.\nNext: y.\nWill: z.";
        let result = lease.try_renew(valid);
        assert!(
            !result.is_valid(),
            "must not exceed max_renewals even with valid checkpoints"
        );
        assert!(
            result.missing_field().contains("out_of_leases"),
            "rejection must name 'out_of_leases', got: {:?}",
            result.missing_field()
        );
    }

    /// Plain final text (no checkpoint labels) is NOT a renewal attempt.
    /// `try_renew` returns `not_attempted`: invalid, but `was_attempted()`
    /// is false and `missing_field` is empty. This is the exit that lets an
    /// exhausted lease finish the turn instead of nagging every text
    /// response — required for convergence once the strip is sticky, since
    /// otherwise a plain answer would be rejected forever.
    #[test]
    fn lease_plain_text_is_not_a_renewal_attempt() {
        let mut lease = Lease::new(1, 3);
        tick(&mut lease);
        assert!(lease.is_exhausted());

        // Plain prose, no findings:/next:/will: labels.
        let plain = "Based on what I read, the loop lives in agent_loop/mod.rs. \
                     It's a fan-out pattern over inbound messages.";
        let result = lease.try_renew(plain);
        assert!(!result.is_valid(), "plain text must not renew");
        assert!(
            !result.was_attempted(),
            "plain text must report was_attempted=false so the caller finishes the turn"
        );
        assert!(
            result.missing_field().is_empty(),
            "plain text is neither a field rejection nor out_of_leases; got: {:?}",
            result.missing_field()
        );
        // Lease state is unchanged — the model did not request more tools.
        assert!(lease.is_exhausted());
        assert_eq!(lease.renewals_used(), 0);
    }

    /// A checkpoint with some-but-not-all labels IS an attempt. The caller
    /// nudge path keys off `was_attempted()`, so a partial checkpoint must
    /// still report true (and name the missing field) while plain text
    /// reports false. This is the discriminator the response handler uses.
    #[test]
    fn lease_partial_checkpoint_is_attempted_and_names_missing_field() {
        let mut lease = Lease::new(1, 3);
        tick(&mut lease);
        assert!(lease.is_exhausted());

        // Has findings + will, missing next.
        let partial = "Findings: located the loop.\nWill: read it next.";
        let result = lease.try_renew(partial);
        assert!(!result.is_valid());
        assert!(
            result.was_attempted(),
            "a checkpoint with any labels counts as an attempt"
        );
        assert!(result.missing_field().contains("next"));

        // A fully valid checkpoint is also an attempt (trivially).
        let mut lease2 = Lease::new(1, 3);
        tick(&mut lease2);
        let valid_result = lease2.try_renew("Findings: a.\nNext: b.\nWill: c.");
        assert!(valid_result.is_valid());
        assert!(valid_result.was_attempted());
    }

    #[test]
    fn batch_admission_is_atomic_at_remaining_boundary() {
        let mut lease = Lease::new(3, 1);
        assert_eq!(lease.admit_batch(2), BatchAdmission::Admitted);
        assert_eq!(
            lease.admit_batch(2),
            BatchAdmission::Rejected { remaining: 1 }
        );
        assert_eq!(lease.iterations_used, 2, "rejection must consume nothing");
        assert_eq!(lease.admit_batch(1), BatchAdmission::Admitted);
        assert!(lease.is_exhausted());
    }

    #[test]
    fn admitted_multi_call_batch_consumes_every_call() {
        let mut lease = Lease::new(5, 2);
        assert_eq!(lease.admit_batch(3), BatchAdmission::Admitted);
        assert_eq!(lease.iterations_used, 3);
        assert_eq!(lease.admit_batch(2), BatchAdmission::Admitted);
        assert_eq!(lease.iterations_used, 5);
    }

    #[test]
    fn result_annotation_reports_post_batch_usage() {
        let mut lease = Lease::new(5, 2);
        assert_eq!(lease.admit_batch(2), BatchAdmission::Admitted);
        assert_eq!(
            lease.annotate_result("payload"),
            "[Lease usage after this batch: 2 of 5 calls — 2 renewals remaining.]\npayload"
        );
    }

    #[test]
    fn final_batch_annotation_requires_answer_or_renewal() {
        let mut lease = Lease::new(2, 3);
        assert_eq!(lease.admit_batch(2), BatchAdmission::Admitted);
        let annotated = lease.annotate_result("payload");
        assert!(annotated.contains("Lease usage after this batch: 2 of 2 calls"));
        assert!(annotated.contains("Lease exhausted"));
        assert!(annotated.contains("findings:/next:/will:"));
        assert!(annotated.contains("Do not request another tool before renewal"));
        assert!(annotated.ends_with("\npayload"));
    }
}
