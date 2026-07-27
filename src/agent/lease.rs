//! Tool leases — structural loop prevention.
//!
//! See `docs/superpowers/specs/2026-07-27-tool-leases-design.md`.
//!
//! Each user turn starts with a tool lease of `TOOLS_PER_LEASE` tool
//! iterations. After exhaustion, tool definitions are stripped from the
//! request: the model must produce a final text answer OR emit a renewal
//! checkpoint (findings + remaining question + next bounded actions). The
//! design makes the live 13-call `exec grep` loop (session
//! `20260727_161730_6e61a0`, 2026-07-27 16:19-16:21) structurally
//! impossible.

use serde_json::Value;

/// Default per-lease tool budget. Tuned so a typical lookup-and-answer
/// turn completes in one lease (read file → answer), but multi-step
/// exploration must checkpoint to continue.
pub const DEFAULT_TOOLS_PER_LEASE: u32 = 5;

/// Default cap on lease renewals per turn. Three leases × five tools =
/// 15 tool calls per turn at maximum, each renewal gated by a validated
/// checkpoint. Real work almost never exceeds two leases; the cap exists
/// to bound a model that learns to emit minimal checkpoints.
pub const DEFAULT_MAX_LEASES_PER_TURN: u32 = 3;

/// Default cap on consecutive same-coarse-family tool calls within a
/// lease. Three calls to `exec:grep` is fine; the fourth is blocked with
/// a receipt. The model must switch family or checkpoint. This is the
/// direct prevention for the live `13 × exec:grep` failure.
pub const DEFAULT_COARSE_FAMILY_CAP: u32 = 3;

/// Read-only exec commands (by leading word). Writes and unknown
/// commands are treated as their own family — `exec:write` or
/// `exec:other`. Read-only classification is deliberately narrow so the
/// coarse-family cap cannot let a write slip past as a "read".
const READONLY_EXEC_FIRST_WORDS: &[&str] = &[
    "grep", "rg", "find", "ls", "cat", "head", "tail", "wc", "which", "file", "stat",
    "tree",
    // `git <subcommand>` is read-only only for these subcommands. Bare
    // `git` (e.g. `git add`, `git commit`) is a write.
];

/// `git` read-only subcommands. Matched when the command starts with
/// `git <sub>`.
const READONLY_GIT_SUBCOMMANDS: &[&str] = &[
    "status",
    "log",
    "diff",
    "show",
    "blame",
    "ls-files",
    "rev-parse",
];

/// Coarse family of a tool call. Used by the consecutive-cap enforcer.
///
/// The family string is intentionally human-readable (logged and visible
/// to the model in receipts) so debugging is straightforward.
///
/// `exec` is classified by first word of the command (`exec:grep`,
/// `exec:find`, `exec:git`), so the cap distinguishes "3 greps then a
/// find" (allowed — different commands, the model is exploring) from
/// "4 greps" (blocked — the live 13×grep failure pattern). The
/// `git` subcommand is part of the family only when ambiguous on the
/// first word — e.g. `git status` and `git log` are both `exec:git`
/// (the model switching subcommands under one binary is the loop shape
/// we cap).
pub fn coarse_family(tool_name: &str, args: &std::collections::HashMap<String, Value>) -> String {
    match tool_name {
        "exec" => {
            let command = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
            exec_coarse_family(command)
        }
        "read_file" | "list_dir" | "find_files" | "search_files" | "recall" | "read_skill" => {
            "read".to_string()
        }
        "web_search" | "web_fetch" => "web".to_string(),
        _ => format!("tool:{tool_name}"),
    }
}

/// Classify an exec command into a coarse family by first word.
///
/// The family is `exec:{first_word}` (e.g. `exec:grep`, `exec:git`,
/// `exec:npm`). The first word is the actual binary being invoked, so
/// `grep` and `find` are different families — switching from one to the
/// other is genuine exploration, not loop repetition. Trailing `&` is
/// stripped so backgrounded commands cluster with their foreground
/// equivalents.
fn exec_coarse_family(command: &str) -> String {
    let trimmed = command.trim();
    let first_word = trimmed
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_end_matches('&');
    if first_word.is_empty() {
        return "exec:other".to_string();
    }
    format!("exec:{first_word}")
}

/// True iff the exec command is read-only (no filesystem mutation).
/// Used by the future A3 readonly-exec result cache. Distinct from
/// `coarse_family` — the cap treats `exec:grep` and `exec:find` as
/// different families, but BOTH are readonly for caching purposes.
#[allow(dead_code)]
pub fn is_readonly_exec(command: &str) -> bool {
    let trimmed = command.trim();
    let first_word = trimmed
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_end_matches('&');
    if READONLY_EXEC_FIRST_WORDS.contains(&first_word) {
        return true;
    }
    if first_word == "git" {
        let sub = trimmed
            .split_whitespace()
            .nth(1)
            .unwrap_or("")
            .trim_end_matches('&');
        return READONLY_GIT_SUBCOMMANDS.contains(&sub);
    }
    false
}

// ---------------------------------------------------------------------------
// Lease state machine
// ---------------------------------------------------------------------------

/// A per-turn tool lease. Counts tool iterations within the current
/// lease; on exhaustion the caller must strip tool definitions and
/// require the model to either answer or emit a renewal checkpoint.
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
    /// Coarse family of the most recent tool call, with its consecutive
    /// count. Reset on a different family, on renewal, or when the cap
    /// blocks a call (so the model can switch family on the next call).
    last_family: Option<String>,
    consecutive_same_family: u32,
    coarse_family_cap: u32,
}

/// Outcome of `record_tool_call_in_family`. The `reason` is a stable
/// machine-readable code (`lease_exhausted`, `coarse_family_cap`) so
/// callers can route receipts without parsing prose.
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenewalResult {
    valid: bool,
    missing_field: &'static str,
}

impl RenewalResult {
    fn accepted() -> Self {
        Self {
            valid: true,
            missing_field: "",
        }
    }
    fn rejected(missing: &'static str) -> Self {
        Self {
            valid: false,
            missing_field: missing,
        }
    }
    pub fn is_valid(&self) -> bool {
        self.valid
    }
    pub fn missing_field(&self) -> &str {
        self.missing_field
    }
}

impl Lease {
    pub fn new(lease_size: u32, max_renewals: u32) -> Self {
        Self {
            lease_size: lease_size.max(1),
            max_renewals,
            iterations_used: 0,
            renewals_used: 0,
            last_family: None,
            consecutive_same_family: 0,
            coarse_family_cap: DEFAULT_COARSE_FAMILY_CAP,
        }
    }

    /// Record a tool call without family tracking (for callers that
    /// don't use the coarse cap). Returns `false` when the lease is
    /// exhausted — the caller should then strip tools from the next
    /// request and require a renewal-or-answer.
    pub fn record_tool_call(&mut self) -> bool {
        if self.iterations_used >= self.lease_size {
            return false;
        }
        self.iterations_used += 1;
        true
    }

    /// Record a tool call with coarse-family tracking. Returns whether
    /// the call was allowed plus the rejection reason if not. A blocked
    /// call does NOT consume the per-lease budget — the model is told
    /// it hit the cap and is expected to switch family or checkpoint.
    pub fn record_tool_call_in_family(
        &mut self,
        tool_name: &str,
        args: &std::collections::HashMap<String, Value>,
    ) -> ToolCallResult {
        if self.iterations_used >= self.lease_size {
            return ToolCallResult::blocked("lease_exhausted");
        }
        let family = coarse_family(tool_name, args);
        if family == self.last_family.as_deref().unwrap_or("")
            && self.consecutive_same_family >= self.coarse_family_cap
        {
            return ToolCallResult::blocked("coarse_family_cap");
        }
        if Some(&family) == self.last_family.as_ref() {
            self.consecutive_same_family += 1;
        } else {
            self.last_family = Some(family);
            self.consecutive_same_family = 1;
        }
        self.iterations_used += 1;
        ToolCallResult::allowed()
    }

    pub fn is_exhausted(&self) -> bool {
        self.iterations_used >= self.lease_size
    }

    pub fn iterations_used(&self) -> u32 {
        self.iterations_used
    }

    pub fn renewals_used(&self) -> u32 {
        self.renewals_used
    }

    /// Configured per-lease tool budget. Exposed so the renewal nudge
    /// can tell the model exactly how many calls it has after renewal.
    pub fn lease_size(&self) -> u32 {
        self.lease_size
    }

    /// Coarse-family cap (consecutive same-family calls allowed within
    /// a lease). Exposed for logging/receipts — modification is via
    /// `new()` defaults only for now.
    pub fn coarse_family_cap(&self) -> u32 {
        self.coarse_family_cap
    }

    /// Try to renew the lease with a model-emitted checkpoint. The
    /// checkpoint must contain `findings:`, `next:`, and `will:` (any
    /// case). Returns `RenewalResult::accepted()` and resets the
    /// iteration budget on success; on failure returns which field is
    /// missing.
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
        self.last_family = None;
        self.consecutive_same_family = 0;
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
    use std::collections::HashMap;
    use Value;

    fn args(pairs: &[(&str, &str)]) -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), Value::String(v.to_string())))
            .collect()
    }

    // -----------------------------------------------------------------
    // RED tests — write first, watch fail, then GREEN each
    // -----------------------------------------------------------------

    /// The live failure was 13 `exec grep` calls. The cap keys on
    /// `exec:{first_word}`, so all `grep` invocations cluster as
    /// `exec:grep` (regardless of args). `find`, `rg`, `cat` are
    /// separate families — switching between them is genuine
    /// exploration, not loop repetition.
    #[test]
    fn coarse_family_classifies_exec_grep_as_distinct_per_first_word() {
        let g = coarse_family("exec", &args(&[("command", "grep pattern path")]));
        let r = coarse_family("exec", &args(&[("command", "rg pattern path")]));
        let f = coarse_family("exec", &args(&[("command", "find . -name x")]));
        let l = coarse_family("exec", &args(&[("command", "ls -la")]));
        let c = coarse_family("exec", &args(&[("command", "cat file.txt")]));
        assert_eq!(g, "exec:grep", "grep must be exec:grep");
        assert_eq!(r, "exec:rg", "rg must be exec:rg");
        assert_eq!(f, "exec:find", "find must be exec:find");
        assert_eq!(l, "exec:ls", "ls must be exec:ls");
        assert_eq!(c, "exec:cat", "cat must be exec:cat");
        // All distinct — switching between them does NOT count as
        // repetition. The live loop was 13× THE SAME grep, which
        // the cap fires on.
        let families = [g, r, f, l, c];
        let unique: std::collections::HashSet<_> = families.iter().collect();
        assert_eq!(
            families.len(),
            unique.len(),
            "all five must be distinct families"
        );
    }

    /// Writes (`rm`, `mv`, `git add`, `npm install`) classify by first
    /// word just like reads — there is no read/write split in the
    /// coarse family. A write after three reads is NOT blocked (different
    /// family); a write after three of THE SAME write IS blocked.
    #[test]
    fn coarse_family_distinguishes_readonly_exec_from_write() {
        let rm = coarse_family("exec", &args(&[("command", "rm file")]));
        let grep = coarse_family("exec", &args(&[("command", "grep x y")]));
        assert_eq!(rm, "exec:rm");
        assert_eq!(grep, "exec:grep");
        assert_ne!(rm, grep, "rm and grep must be different families");
        // `is_readonly_exec` is the separate concern (for A3 caching),
        // NOT the coarse family — different functions, different layers.
        assert!(!is_readonly_exec("rm file"));
        assert!(is_readonly_exec("grep x y"));
    }

    /// `git status`, `git log`, `git diff` are common read-only
    /// inspections via `is_readonly_exec`. They classify as `exec:git`
    /// for coarse-family purposes (one family per binary), but the
    /// readonly check distinguishes them from `git add` / `git commit`.
    #[test]
    fn coarse_family_git_subcommands_classified_correctly() {
        // Coarse family — all `git …` are `exec:git` (the model
        // switching subcommands under one binary is the loop shape we
        // cap; if the model runs `git status` three times in a row the
        // cap fires correctly).
        let status = coarse_family("exec", &args(&[("command", "git status")]));
        let log = coarse_family("exec", &args(&[("command", "git log --oneline")]));
        let add = coarse_family("exec", &args(&[("command", "git add file")]));
        assert_eq!(status, "exec:git");
        assert_eq!(log, "exec:git");
        assert_eq!(add, "exec:git");

        // Readonly check — distinguishes read subcommands from writes.
        assert!(is_readonly_exec("git status"));
        assert!(is_readonly_exec("git log --oneline"));
        assert!(is_readonly_exec("git diff HEAD~1"));
        assert!(!is_readonly_exec("git add file"));
        assert!(!is_readonly_exec("git commit -m x"));
    }

    /// Non-exec read tools all share the `read` family so the cap
    /// catches back-to-back `read_file`/`list_dir`/`recall` cycles too.
    #[test]
    fn coarse_family_clusters_non_exec_read_tools() {
        for tool in [
            "read_file",
            "list_dir",
            "find_files",
            "recall",
            "read_skill",
        ] {
            assert_eq!(
                coarse_family(tool, &HashMap::new()),
                "read",
                "{tool} should be in the 'read' family"
            );
        }
    }

    /// Write tools (`write_file`, `edit_file`, `apply_patch`) each get
    /// their own family so they don't share a cap with reads. The
    /// generic fallback is `tool:<name>` so every tool has a family.
    #[test]
    fn coarse_family_write_tools_get_own_family() {
        let w = coarse_family("write_file", &HashMap::new());
        let e = coarse_family("edit_file", &HashMap::new());
        let a = coarse_family("apply_patch", &HashMap::new());
        assert_eq!(w, "tool:write_file");
        assert_eq!(e, "tool:edit_file");
        assert_eq!(a, "tool:apply_patch");
        assert_ne!(w, e);
    }

    // -----------------------------------------------------------------
    // Lease state machine — these go RED until the `Lease` struct exists.
    // -----------------------------------------------------------------

    /// A fresh lease allows `lease_size` tool iterations. The
    /// `(iteration + 1) == lease_size`-th call must report exhaustion
    /// (tools should be stripped by the caller on the next LLM request).
    #[test]
    fn lease_allows_n_tool_calls_then_reports_exhausted() {
        let mut lease = Lease::new(3, 2);
        assert!(lease.record_tool_call());
        assert!(lease.record_tool_call());
        assert!(lease.record_tool_call());
        // Three tools consumed the lease; the 4th call must be rejected.
        assert!(
            !lease.record_tool_call(),
            "lease must be exhausted after lease_size tool calls"
        );
        assert!(lease.is_exhausted());
        assert_eq!(lease.iterations_used(), 3);
    }

    /// After exhaustion, the model must either answer or request a
    /// renewal. A valid checkpoint (findings + next + will) re-arms the
    /// lease for another `lease_size` iterations.
    #[test]
    fn lease_renewal_restores_budget_when_checkpoint_is_valid() {
        let mut lease = Lease::new(2, 3);
        lease.record_tool_call();
        lease.record_tool_call();
        assert!(lease.is_exhausted());

        let checkpoint = "Findings: located tau_soft in config/schema.rs.\n\
                          Next: need to verify the default value.\n\
                          Will: grep for default_lcm_tau_soft.";
        assert!(
            lease.try_renew(checkpoint).is_valid(),
            "valid checkpoint (findings + next + will) must renew the lease"
        );
        assert!(!lease.is_exhausted());
        assert_eq!(lease.iterations_used(), 0, "renewal resets the budget");
        assert_eq!(lease.renewals_used(), 1);
    }

    /// Renewal is the only synth-injection-free checkpoint mechanism —
    /// it must reject checkpoints that lack findings, next, or will.
    /// Each rejection path is named so the model can correct.
    #[test]
    fn lease_renewal_rejected_when_any_required_field_is_missing() {
        let mut lease = Lease::new(1, 3);
        lease.record_tool_call();
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
            lease.record_tool_call();
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
        lease.record_tool_call();
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

    /// Within a lease, the coarse-family cap blocks the 4th consecutive
    /// same-family call. A different family is still allowed.
    #[test]
    fn coarse_family_cap_blocks_fourth_consecutive_grep() {
        let mut lease = Lease::new(5, 2);
        let grep_args = args(&[("command", "grep pattern path")]);
        let find_args = args(&[("command", "find . -name x")]);

        // Three consecutive `readonly_search` calls are allowed.
        assert!(lease.record_tool_call_in_family("exec", &grep_args).allowed);
        assert!(lease.record_tool_call_in_family("exec", &grep_args).allowed);
        assert!(lease.record_tool_call_in_family("exec", &grep_args).allowed);

        // The 4th same-family call is blocked.
        let blocked = lease.record_tool_call_in_family("exec", &grep_args);
        assert!(
            !blocked.allowed,
            "4th consecutive readonly_search must be blocked by the cap"
        );
        assert_eq!(
            blocked.reason,
            Some("coarse_family_cap"),
            "blocked reason must name the cap"
        );

        // A different family is still allowed within the same lease.
        let allowed = lease.record_tool_call_in_family("exec", &find_args);
        assert!(allowed.allowed, "different family must still be allowed");
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
        lease.record_tool_call();
        let s1 = lease.progress_signal();
        assert!(s1.contains("Tool call 1 of 5"), "got: {s1}");
        assert!(s1.contains("3 leases remaining"), "got: {s1}");

        lease.record_tool_call();
        let s2 = lease.progress_signal();
        assert!(s2.contains("Tool call 2 of 5"), "got: {s2}");
        assert!(s2.contains("3 leases remaining"), "got: {s2}");

        // Renew — now in lease 2; 2 future leases obtainable.
        for _ in 0..3 {
            lease.record_tool_call();
        }
        assert!(lease.is_exhausted());
        lease.try_renew("Findings: x.\nNext: y.\nWill: z.");
        // After renewal, the next call records as call 1 of the new lease.
        lease.record_tool_call();
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
