//! Tool leases — structural loop prevention.
//!
//! See `docs/superpowers/specs/2026-07-27-tool-leases-design.md`.
//!
//! Each user turn has one 96-success execution budget. A call reserves a slot,
//! and only a durably persisted successful result consumes it. Failed calls
//! release their reservation so recovery can retry without lease ceremonies.

/// One turn budget. This is deliberately a single number: a tool-heavy turn
/// should not have to negotiate a sequence of subleases with the model.
pub const DEFAULT_TOOLS_PER_LEASE: u32 = 96;

/// Default cap on lease renewals per turn. Three leases × twelve tools
/// = 36 tool calls per turn at maximum, each renewal gated by a
/// validated checkpoint.
/// Kept for decoding and replaying historical turns. New turns do not renew
/// leases; `Lease::new(DEFAULT_TOOLS_PER_LEASE, 0)` is the active policy.
pub const DEFAULT_MAX_LEASES_PER_TURN: u32 = 0;

// Lease state machine
// ---------------------------------------------------------------------------

/// A per-turn tool budget. It retains checkpoint parsing solely so old session
/// receipts remain decodable; active turns never renew this budget.
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
    /// Calls accepted but whose durable result has not arrived yet. Reserving
    /// here prevents a parallel batch from exceeding the cap.
    in_flight: u32,
    successful_executions: u32,
    renewals_used: u32,
    /// Checkpoint-free renewals spent on read-only tools. Separate from
    /// `renewals_used` so exploration can't starve the write path.
    read_only_renewals_used: u32,
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
            in_flight: 0,
            successful_executions: 0,
            renewals_used: 0,
            read_only_renewals_used: 0,
        }
    }

    /// Reserve one tool call against the single successful-execution budget.
    /// The reservation is settled with [`record_tool_result`]. Failed calls
    /// release their reservation and can be retried; only successfully
    /// persisted results consume the turn budget.
    ///
    /// There is NO consecutive-same-family cap anymore: it over-fired on
    /// legitimate exploration (N different greps) and busted the prompt-prefix
    /// cache when it triggered a strip (2026-07-30). Identical-call loops are
    /// bounded by `ToolGuard`'s per-key `seen` counter (tool_guard.rs, applies
    /// to all tools); different-args exploration is bounded by this per-lease
    /// budget plus the no-progress hard stop. See
    /// docs/superpowers/plans/2026-07-30-reuse-not-rerun-tool-dedup.md.
    pub fn record_tool_call(&mut self) -> ToolCallResult {
        if self.successful_executions.saturating_add(self.in_flight) >= self.lease_size {
            return ToolCallResult::blocked("lease_exhausted");
        }
        self.in_flight += 1;
        ToolCallResult::allowed()
    }

    /// Settle a previously reserved call after its result has been durably
    /// persisted. A failed result is retryable and therefore does not count.
    pub fn record_tool_result(&mut self, ok: bool) {
        self.in_flight = self.in_flight.saturating_sub(1);
        if ok {
            self.successful_executions = self.successful_executions.saturating_add(1);
        }
    }

    /// Release reservations for calls that never entered the tool executor,
    /// such as a protocol persistence failure.
    pub fn release_pending(&mut self, count: u32) {
        self.in_flight = self.in_flight.saturating_sub(count);
    }

    pub fn successful_executions(&self) -> u32 {
        self.successful_executions
    }

    pub fn is_exhausted(&self) -> bool {
        self.successful_executions.saturating_add(self.in_flight) >= self.lease_size
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
        self.in_flight = 0;
        self.successful_executions = 0;
        RenewalResult::accepted()
    }

    /// Auto-renew the lease for read-only tools without requiring a checkpoint.
    /// Read-only tools (read_file, list_dir, etc.) can't cause destructive
    /// loops, so blocking them behind a manual checkpoint ceremony wastes
    /// 3 round-trips on legitimate multi-file exploration. Returns false
    /// when out of auto-renewals.
    ///
    /// Spends a SEPARATE budget from `try_renew`: a read-heavy turn that
    /// auto-renewed its way through the checkpoint budget would then hard-stop
    /// on its first `write_file`, which is exactly backwards — the reads are
    /// the cheap part. Both budgets are capped at `max_renewals`, so the worst
    /// case is 2× leases per turn, half of them read-only.
    pub fn auto_renew_for_read_only(&mut self) -> bool {
        if self.read_only_renewals_used >= self.max_renewals {
            return false;
        }
        self.read_only_renewals_used += 1;
        self.in_flight = 0;
        self.successful_executions = 0;
        true
    }

    /// Format the progress signal for inclusion in tool results.
    ///
    /// Reports the MINIMUM of the two remaining-lease counters (checkpoint
    /// and read-only auto-renewal). Both are capped at `max_renewals`; the
    /// model must see the binding constraint so it can plan tool use before
    /// hitting a hard block with no warning.
    ///
    /// `N` is `iterations_used` — the call that just executed
    /// (`record_tool_call` was already called by the time `tool_engine`
    /// adds the result). For the first call this is 1, etc.
    pub fn progress_signal(&self) -> String {
        let current_call = self.successful_executions.saturating_add(self.in_flight);
        let checkpoint_remaining = self.max_renewals.saturating_sub(self.renewals_used);
        let read_only_remaining = self
            .max_renewals
            .saturating_sub(self.read_only_renewals_used);
        let leases_remaining = std::cmp::min(checkpoint_remaining, read_only_remaining);
        format!(
            "[Tool call {} of {} this lease — {} leases remaining]",
            current_call, self.lease_size, leases_remaining
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exhausting the read-only auto-renewal budget must leave the
    /// checkpoint budget untouched: a turn that explored its way to the cap
    /// can still renew for the write it was exploring toward.
    #[test]
    fn read_only_auto_renewals_do_not_spend_the_checkpoint_budget() {
        let mut lease = Lease::new(2, 2);
        for _ in 0..2 {
            assert!(lease.auto_renew_for_read_only());
        }
        assert!(
            !lease.auto_renew_for_read_only(),
            "read-only budget must be capped at max_renewals"
        );
        assert_eq!(lease.renewals_used(), 0);

        let checkpoint = "findings: read 4 files. next: confirm. will: write the fix.";
        for _ in 0..2 {
            assert!(lease.try_renew(checkpoint).is_valid());
        }
        assert_eq!(lease.try_renew(checkpoint).missing_field(), "out_of_leases");
    }

    /// Advance the lease by one call. Used by the state-machine tests below to
    /// exercise exhaustion/renewal/progress in isolation.
    fn tick(lease: &mut Lease) -> bool {
        lease.record_tool_call().allowed
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

    /// There is NO consecutive-same-family cap anymore (retired 2026-07-30 — it
    /// over-fired on legitimate exploration like N different greps and busted
    /// the prompt-prefix cache). The lease now bounds only by `lease_size`;
    /// identical-call loops are bounded by `ToolGuard`'s per-key counter. So
    /// many calls up to `lease_size` are all allowed regardless of family, and
    /// the only block reason is `lease_exhausted`.
    #[test]
    fn record_tool_call_allows_up_to_lease_size_with_no_family_cap() {
        let mut lease = Lease::new(5, 2);
        // Five calls — all allowed, no matter how "same-family" they'd be.
        for _ in 0..5 {
            assert!(
                lease.record_tool_call().allowed,
                "calls within lease_size must be allowed (no family cap)"
            );
        }
        // The 6th exhausts the lease — that is the only block path now.
        let blocked = lease.record_tool_call();
        assert!(
            !blocked.allowed,
            "lease_size+1 must be blocked by exhaustion"
        );
        assert_eq!(
            blocked.reason,
            Some("lease_exhausted"),
            "the only block reason is lease_exhausted; no coarse_family_cap"
        );
    }

    #[test]
    fn single_turn_budget_allows_96_successes_and_rejects_97th() {
        let mut lease = Lease::new(96, 0);
        for _ in 0..96 {
            assert!(lease.record_tool_call().allowed);
            lease.record_tool_result(true);
        }
        assert_eq!(lease.successful_executions(), 96);
        let blocked = lease.record_tool_call();
        assert_eq!(blocked.reason, Some("lease_exhausted"));
    }

    #[test]
    fn failed_execution_releases_budget_and_can_be_retried() {
        let mut lease = Lease::new(1, 0);
        assert!(lease.record_tool_call().allowed);
        lease.record_tool_result(false);
        assert_eq!(lease.successful_executions(), 0);
        assert!(lease.record_tool_call().allowed);
        lease.record_tool_result(true);
        assert!(!lease.record_tool_call().allowed);
    }

    #[test]
    fn mixed_batch_reserves_only_the_successful_budget() {
        let mut lease = Lease::new(2, 0);
        assert!(lease.record_tool_call().allowed);
        assert!(lease.record_tool_call().allowed);
        assert!(!lease.record_tool_call().allowed);
        lease.record_tool_result(false);
        assert!(lease.record_tool_call().allowed);
        lease.record_tool_result(true);
        lease.record_tool_result(true);
        assert_eq!(lease.successful_executions(), 2);
        assert!(!lease.record_tool_call().allowed);
    }

    /// Tool results include the progress signal
    /// `[Tool call N of M this lease — L leases remaining]`. The model
    /// uses this to self-regulate instead of being interrupted.
    /// `L = max_renewals - renewals_used` (number of future leases still
    /// obtainable this turn). `N = iterations_used` — the call that just
    /// ran (record_tool_call* was called before the result is formatted).
    #[test]
    fn lease_progress_signal_format() {
        let mut lease = Lease::new(5, 3);
        // First call records, then signal describes that call.
        tick(&mut lease);
        let s1 = lease.progress_signal();
        assert!(s1.contains("Tool call 1 of 5"), "got: {s1}");
        assert!(s1.contains("3 leases remaining"), "got: {s1}");

        tick(&mut lease);
        let s2 = lease.progress_signal();
        assert!(s2.contains("Tool call 2 of 5"), "got: {s2}");
        assert!(s2.contains("3 leases remaining"), "got: {s2}");

        // Renew — now in lease 2; 2 future leases obtainable.
        for _ in 0..3 {
            tick(&mut lease);
        }
        assert!(lease.is_exhausted());
        lease.try_renew("Findings: x.\nNext: y.\nWill: z.");
        // After renewal, the next call records as call 1 of the new lease.
        tick(&mut lease);
        let s_after = lease.progress_signal();
        assert!(
            s_after.contains("Tool call 1 of 5"),
            "renewal must reset the per-lease counter, got: {s_after}"
        );
        assert!(
            s_after.contains("2 leases remaining"),
            "one less lease remaining after renewal, got: {s_after}"
        );
    }
}
