//! Mechanical claim verification for agent provenance.
//!
//! Extracts claims from agent text output and verifies them against
//! the audit log. No LLM involved — purely regex + string matching.

use regex::Regex;

use crate::agent::audit::AuditEntry;

/// Classification of how well a claim is supported by the audit log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimStatus {
    /// Claim matches audit log data exactly (20+ char exact match).
    Observed,
    /// Claim partially matches audit data (substring overlap).
    Derived,
    /// Claim references a tool call but content doesn't match.
    Claimed,
    /// Claim references memory/history, not current tool calls.
    Recalled,
}

/// A claim extracted from agent text with its verification status.
#[derive(Debug, Clone)]
pub struct AnnotatedClaim {
    /// Byte range in the original text.
    pub span: (usize, usize),
    /// Type of claim: "file_ref", "command_ref", "action_claim", "quoted_output", "numeric".
    pub claim_type: String,
    /// The verification status.
    pub status: ClaimStatus,
    /// The matched text from the agent's response.
    pub text: String,
}

/// Verifies claims in agent output against audit log entries.
pub struct ClaimVerifier<'a> {
    entries: &'a [AuditEntry],
}

impl<'a> ClaimVerifier<'a> {
    pub fn new(entries: &'a [AuditEntry]) -> Self {
        Self { entries }
    }

    /// Extract and verify all claims in the given text.
    pub fn verify(&self, text: &str) -> Vec<AnnotatedClaim> {
        let mut claims = Vec::new();

        // Extract file path references.
        claims.extend(self.extract_file_refs(text));

        // Extract command references.
        claims.extend(self.extract_command_refs(text));

        // Extract quoted output (backtick blocks after "output/result/returns/shows").
        claims.extend(self.extract_quoted_output(text));

        // Extract action claims ("I read/wrote/created/...").
        claims.extend(self.extract_action_claims(text));

        // Extract numeric assertions.
        claims.extend(self.extract_numeric_claims(text));

        // Extract outcome claims ("Build succeeded", "install worked", etc.).
        claims.extend(self.extract_outcome_claims(text));

        // Extract timestamp claims (HH:MM patterns).
        claims.extend(self.extract_timestamp_claims(text));

        // Deduplicate overlapping spans (keep the most specific).
        claims.sort_by_key(|c| c.span.0);
        claims.dedup_by(|a, b| a.span.0 == b.span.0 && a.span.1 == b.span.1);

        claims
    }

    /// Check if any claims are unverified (Claimed status).
    pub fn has_unverified(&self, claims: &[AnnotatedClaim]) -> bool {
        claims.iter().any(|c| c.status == ClaimStatus::Claimed)
    }

    /// Get a summary of unverified claims for strict mode.
    pub fn unverified_summary(&self, claims: &[AnnotatedClaim]) -> String {
        let unverified: Vec<&AnnotatedClaim> = claims
            .iter()
            .filter(|c| c.status == ClaimStatus::Claimed)
            .collect();

        if unverified.is_empty() {
            return String::new();
        }

        let mut lines = vec![format!("{} unverified claim(s):", unverified.len())];
        for claim in &unverified {
            let preview: String = claim.text.chars().take(60).collect();
            lines.push(format!("  ⚠ [{}] {}", claim.claim_type, preview));
        }
        lines.join("\n")
    }

    // --- Extraction methods ---

    fn extract_file_refs(&self, text: &str) -> Vec<AnnotatedClaim> {
        let mut claims = Vec::new();
        // Match patterns like: read/wrote/created/edited + path-like string
        let re = Regex::new(
            r"(?i)\b(read|wrote|created|edited|deleted|found|opened)\b[^`\n]{0,20}[`]([/~][^\s`]+)[`]"
        ).unwrap();

        for cap in re.captures_iter(text) {
            if let (Some(full), Some(path)) = (cap.get(0), cap.get(2)) {
                let path_str = path.as_str();
                let status = self.match_against_entries(path_str, Some("file"));
                claims.push(AnnotatedClaim {
                    span: (full.start(), full.end()),
                    claim_type: "file_ref".to_string(),
                    status,
                    text: full.as_str().to_string(),
                });
            }
        }
        claims
    }

    fn extract_command_refs(&self, text: &str) -> Vec<AnnotatedClaim> {
        let mut claims = Vec::new();
        // Match: ran/executed + backtick content
        let re = Regex::new(r"(?i)\b(ran|executed|running)\b[^`\n]{0,20}`([^`]+)`").unwrap();

        for cap in re.captures_iter(text) {
            if let (Some(full), Some(cmd)) = (cap.get(0), cap.get(2)) {
                let cmd_str = cmd.as_str();
                let status = self.match_against_entries(cmd_str, Some("exec"));
                claims.push(AnnotatedClaim {
                    span: (full.start(), full.end()),
                    claim_type: "command_ref".to_string(),
                    status,
                    text: full.as_str().to_string(),
                });
            }
        }
        claims
    }

    fn extract_quoted_output(&self, text: &str) -> Vec<AnnotatedClaim> {
        let mut claims = Vec::new();
        // Match: "output/result/returns/shows" followed by backtick block
        let re = Regex::new(
            r"(?i)\b(output|result|returns?|shows?|returned|produced)\b[:\s]*\n?```[^\n]*\n([\s\S]*?)```"
        ).unwrap();

        for cap in re.captures_iter(text) {
            if let (Some(full), Some(content)) = (cap.get(0), cap.get(2)) {
                let content_str = content.as_str().trim();
                if content_str.len() < 5 {
                    continue; // too short to verify
                }
                let status = self.match_content_against_results(content_str);
                claims.push(AnnotatedClaim {
                    span: (full.start(), full.end()),
                    claim_type: "quoted_output".to_string(),
                    status,
                    text: content_str.chars().take(100).collect(),
                });
            }
        }
        claims
    }

    fn extract_action_claims(&self, text: &str) -> Vec<AnnotatedClaim> {
        let mut claims = Vec::new();

        // Pattern 1: "I read/wrote/ran/..." (past-tense self-attribution)
        let re_past = Regex::new(
            r"(?i)\bI (read|wrote|created|deleted|executed|searched|fetched|edited|ran|modified|updated|removed|checked|verified|built|compiled|installed|copied)\b[^.\n]{0,80}"
        ).unwrap();

        for cap in re_past.captures_iter(text) {
            if let (Some(full), Some(action)) = (cap.get(0), cap.get(1)) {
                let action_str = action.as_str().to_lowercase();
                let tool_hint = Self::action_to_tool_hint(&action_str);
                let status = self.match_action_against_entries(&action_str, tool_hint);
                claims.push(AnnotatedClaim {
                    span: (full.start(), full.end()),
                    claim_type: "action_claim".to_string(),
                    status,
                    text: full.as_str().to_string(),
                });
            }
        }

        // Pattern 2: "Let me check/run/verify/look" (present-tense with implied result)
        let re_present = Regex::new(
            r"(?i)\b[Ll]et me (check|run|verify|look|see|test|try|build|install|copy)\b[^.\n]{0,80}"
        ).unwrap();

        for cap in re_present.captures_iter(text) {
            if let (Some(full), Some(action)) = (cap.get(0), cap.get(1)) {
                let action_str = action.as_str().to_lowercase();
                let tool_hint = Self::action_to_tool_hint(&action_str);
                let status = self.match_action_against_entries(&action_str, tool_hint);
                claims.push(AnnotatedClaim {
                    span: (full.start(), full.end()),
                    claim_type: "action_claim".to_string(),
                    status,
                    text: full.as_str().to_string(),
                });
            }
        }

        // Pattern 3: "When I run X" / "If I run X" (implied action)
        let re_when = Regex::new(
            r"(?i)\b(?:when|if|after)\s+I\s+(run|check|build|test|execute)\b[^.\n]{0,80}",
        )
        .unwrap();

        for cap in re_when.captures_iter(text) {
            if let (Some(full), Some(action)) = (cap.get(0), cap.get(1)) {
                let action_str = action.as_str().to_lowercase();
                let tool_hint = Self::action_to_tool_hint(&action_str);
                let status = self.match_action_against_entries(&action_str, tool_hint);
                claims.push(AnnotatedClaim {
                    span: (full.start(), full.end()),
                    claim_type: "action_claim".to_string(),
                    status,
                    text: full.as_str().to_string(),
                });
            }
        }

        claims
    }

    /// Map an action verb to the tool it implies.
    fn action_to_tool_hint(action: &str) -> Option<&'static str> {
        match action {
            "read" => Some("read_file"),
            "wrote" | "created" | "modified" | "updated" => Some("write_file"),
            "deleted" | "removed" => Some("exec"),
            "executed" | "ran" | "run" | "checked" | "check" | "verified" | "verify" | "built"
            | "build" | "compiled" | "installed" | "install" | "copied" | "copy" | "tested"
            | "test" | "try" | "look" | "see" => Some("exec"),
            "searched" => Some("web_search"),
            "fetched" => Some("web_fetch"),
            "edited" => Some("edit_file"),
            _ => None,
        }
    }

    fn extract_numeric_claims(&self, text: &str) -> Vec<AnnotatedClaim> {
        let mut claims = Vec::new();
        let re = Regex::new(
            r"\b(\d+)\s+(files?|lines?|errors?|tests?|warnings?|results?|matches?|items?)\b",
        )
        .unwrap();

        for cap in re.captures_iter(text) {
            if let (Some(full), Some(num)) = (cap.get(0), cap.get(1)) {
                let num_str = num.as_str();
                // Check if any audit entry result contains this number in context
                let status = if self.entries.iter().any(|e| e.result_data.contains(num_str)) {
                    ClaimStatus::Derived
                } else if self.entries.is_empty() {
                    ClaimStatus::Recalled
                } else {
                    ClaimStatus::Claimed
                };
                claims.push(AnnotatedClaim {
                    span: (full.start(), full.end()),
                    claim_type: "numeric".to_string(),
                    status,
                    text: full.as_str().to_string(),
                });
            }
        }
        claims
    }

    fn extract_outcome_claims(&self, text: &str) -> Vec<AnnotatedClaim> {
        let mut claims = Vec::new();
        // Match outcome assertions: "Build succeeded", "copy worked", "install completed", etc.
        let re = Regex::new(
            r"(?i)\b(build|compile|install|copy|cp|mv|mkdir|chmod|sudo|cargo|npm|pip|make|test|deploy|push|pull|merge)\b[^.\n]{0,30}\b(succeeded|failed|worked|completed|finished|passed|done|ready|updated|error|broke|broken|permission denied|not found|timed? ?out)\b"
        ).unwrap();

        for cap in re.captures_iter(text) {
            if let Some(full) = cap.get(0) {
                let claim_text = full.as_str();
                // Check if any audit entry contains evidence matching this outcome.
                let status = self.match_outcome_against_entries(claim_text);
                claims.push(AnnotatedClaim {
                    span: (full.start(), full.end()),
                    claim_type: "outcome".to_string(),
                    status,
                    text: claim_text.to_string(),
                });
            }
        }
        claims
    }

    fn extract_timestamp_claims(&self, text: &str) -> Vec<AnnotatedClaim> {
        let mut claims = Vec::new();
        // Match time patterns: "17:45", "from 17:36", "at 3:30 PM"
        let re = Regex::new(r"\b(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?)\b").unwrap();

        for cap in re.captures_iter(text) {
            if let Some(full) = cap.get(0) {
                let time_str = full.as_str();
                // Check if any audit entry result contains this timestamp.
                let status = if self
                    .entries
                    .iter()
                    .any(|e| e.result_data.contains(time_str))
                {
                    ClaimStatus::Derived
                } else if self.entries.is_empty() {
                    ClaimStatus::Recalled
                } else {
                    ClaimStatus::Claimed
                };
                claims.push(AnnotatedClaim {
                    span: (full.start(), full.end()),
                    claim_type: "timestamp".to_string(),
                    status,
                    text: time_str.to_string(),
                });
            }
        }
        claims
    }

    // --- Matching helpers ---

    fn match_against_entries(&self, needle: &str, tool_hint: Option<&str>) -> ClaimStatus {
        if self.entries.is_empty() {
            // No tool calls at all = phantom action, not legitimate recall.
            return ClaimStatus::Claimed;
        }

        for entry in self.entries.iter().rev() {
            // Check if tool name matches hint
            let tool_matches = tool_hint
                .map(|hint| entry.tool_name.contains(hint))
                .unwrap_or(true);

            if !tool_matches {
                continue;
            }

            // Check arguments for the needle
            let args_str = serde_json::to_string(&entry.arguments).unwrap_or_default();
            if args_str.contains(needle) {
                return ClaimStatus::Observed;
            }

            // Check result_data for the needle
            if entry.result_data.contains(needle) {
                return ClaimStatus::Observed;
            }
        }

        // Check if any tool call exists with matching hint (but content doesn't match)
        if let Some(hint) = tool_hint {
            if self.entries.iter().any(|e| e.tool_name.contains(hint)) {
                return ClaimStatus::Claimed;
            }
        }

        ClaimStatus::Claimed
    }

    fn match_content_against_results(&self, content: &str) -> ClaimStatus {
        if self.entries.is_empty() {
            return ClaimStatus::Claimed;
        }

        // Try exact substring match (20+ chars).
        let check_len = content.len().min(80);
        let check_str = &content[..check_len];

        for entry in self.entries.iter().rev() {
            if entry.result_data.contains(check_str) {
                return ClaimStatus::Observed;
            }
        }

        // Try shorter substring match (partial).
        if content.len() >= 20 {
            let short = &content[..crate::utils::helpers::floor_char_boundary(content, 20)];
            for entry in self.entries.iter().rev() {
                if entry.result_data.contains(short) {
                    return ClaimStatus::Derived;
                }
            }
        }

        ClaimStatus::Claimed
    }

    fn match_outcome_against_entries(&self, claim_text: &str) -> ClaimStatus {
        if self.entries.is_empty() {
            return ClaimStatus::Claimed;
        }

        let lower = claim_text.to_lowercase();
        let claims_success = lower.contains("succeeded")
            || lower.contains("worked")
            || lower.contains("completed")
            || lower.contains("finished")
            || lower.contains("passed")
            || lower.contains("done")
            || lower.contains("ready")
            || lower.contains("updated");
        let claims_failure = lower.contains("failed")
            || lower.contains("error")
            || lower.contains("broke")
            || lower.contains("broken")
            || lower.contains("permission denied")
            || lower.contains("not found")
            || lower.contains("timed out")
            || lower.contains("timeout");

        // Look at exec tool results to verify outcome claims.
        let exec_entries: Vec<&AuditEntry> = self
            .entries
            .iter()
            .filter(|e| e.tool_name == "exec")
            .collect();

        if exec_entries.is_empty() {
            return ClaimStatus::Claimed;
        }

        // Check if the claimed outcome matches actual tool results.
        let last_exec = exec_entries.last().unwrap();
        if claims_success && last_exec.result_ok {
            return ClaimStatus::Derived;
        }
        if claims_failure && !last_exec.result_ok {
            return ClaimStatus::Derived;
        }

        // Outcome contradicts actual result → fabrication.
        ClaimStatus::Claimed
    }

    fn match_action_against_entries(&self, action: &str, tool_hint: Option<&str>) -> ClaimStatus {
        // When there are NO audit entries and the agent claims an action,
        // this is a phantom action (fabrication), not a recall.
        if self.entries.is_empty() {
            return ClaimStatus::Claimed;
        }

        // Check if there's a matching tool call.
        if let Some(hint) = tool_hint {
            if self.entries.iter().any(|e| e.tool_name.contains(hint)) {
                return ClaimStatus::Observed;
            }
        }

        // Broad check: does any tool name relate to the action?
        let related_tools: Vec<&str> = match action {
            "read" => vec!["read_file", "read"],
            "wrote" | "created" | "modified" | "updated" => {
                vec!["write_file", "write", "edit_file"]
            }
            "deleted" | "removed" => vec!["exec"],
            "executed" | "ran" | "run" | "checked" | "check" | "verified" | "verify" | "built"
            | "build" | "compiled" | "installed" | "install" | "copied" | "copy" | "tested"
            | "test" | "try" | "look" | "see" => vec!["exec"],
            "searched" => vec!["web_search", "search"],
            "fetched" => vec!["web_fetch", "fetch"],
            "edited" => vec!["edit_file", "edit"],
            _ => vec![],
        };

        for entry in self.entries {
            if related_tools.iter().any(|t| entry.tool_name.contains(t)) {
                return ClaimStatus::Observed;
            }
        }

        ClaimStatus::Claimed
    }
}

/// Verify claims in a response against audit entries from this turn.
///
/// Returns annotated claims and a flag indicating whether any fabrication was detected.
pub fn verify_turn_claims(response: &str, entries: &[AuditEntry]) -> (Vec<AnnotatedClaim>, bool) {
    let verifier = ClaimVerifier::new(entries);
    let claims = verifier.verify(response);
    let has_fabrication = verifier.has_unverified(&claims);
    (claims, has_fabrication)
}

/// Redact unverified claims from text, replacing each span with a placeholder.
///
/// Returns the redacted text and the number of redactions made.
/// Claims are processed from end to start to preserve byte offsets.
pub fn redact_fabrications(text: &str, claims: &[AnnotatedClaim]) -> (String, usize) {
    // Filter to Claimed status only.
    let mut to_redact: Vec<&AnnotatedClaim> = claims
        .iter()
        .filter(|c| c.status == ClaimStatus::Claimed)
        .collect();

    if to_redact.is_empty() {
        return (text.to_string(), 0);
    }

    // Sort by span start descending (process from end to preserve offsets).
    to_redact.sort_by(|a, b| b.span.0.cmp(&a.span.0));

    let mut result = text.to_string();
    let mut count = 0usize;

    for claim in &to_redact {
        let (start, end) = claim.span;
        if start <= result.len() && end <= result.len() && start <= end {
            result.replace_range(start..end, "[unverified claim removed]");
            count += 1;
        }
    }

    (result, count)
}

/// Result of phantom detection.
pub struct PhantomDetection {
    /// Warning text for the system message (next turn).
    pub system_warning: String,
    /// Matched phantom patterns.
    pub matched_patterns: Vec<String>,
}

/// Detect phantom tool call claims — when the LLM mentions tool actions
/// but no tools were actually called this turn.
///
/// Returns `Some(PhantomDetection)` if phantom claims are detected.
pub fn detect_phantom_claims(response: &str, tools_called: &[String]) -> Option<PhantomDetection> {
    if !tools_called.is_empty() {
        return None; // Tools were actually called, no phantom possible.
    }

    // Patterns that suggest the LLM is referencing tool results.
    let phantom_patterns = [
        // Past-tense claims (the model already "did" something)
        "I read the file",
        "I ran the command",
        "I executed",
        "I searched for",
        "I wrote to",
        "I created the file",
        "I checked",
        "I verified",
        "I updated",
        "I edited",
        "I deleted",
        // Result claims without tool calls
        "the file contains",
        "the output shows",
        "the result shows",
        "the command returned",
        "the search returned",
        "here's what I found",
        "here is the content",
        "the content is",
        // Promise-then-fabricate patterns ("Let me X" → fake output)
        "let me check",
        "let me read",
        "let me write",
        "let me create",
        "let me run",
        "let me search",
        "let me verify",
        "let me look",
        // Fake tool output indicators
        "```\n$",     // fake shell prompt in code block
        "exit code:", // fake exec output
        "successfully edited",
        "successfully created",
        "successfully wrote",
    ];

    let lower = response.to_lowercase();
    let matches: Vec<String> = phantom_patterns
        .iter()
        .filter(|p| lower.contains(&p.to_lowercase()))
        .map(|p| p.to_string())
        .collect();

    if matches.is_empty() {
        return None;
    }

    let system_warning = format!(
        "CRITICAL: You did not call any tools this turn, but your response contains \
         {} phantom claim(s): [{}]. You MUST use tool calls to perform actions. \
         Do not fabricate tool outputs. If you need to read a file, call read_file. \
         If you need to run a command, call exec. Never generate fake results.",
        matches.len(),
        matches.join(", ")
    );

    Some(PhantomDetection {
        system_warning,
        matched_patterns: matches,
    })
}

/// Prepend a visible phantom warning to the response text so the user
/// knows the response contains unverified claims.
pub fn annotate_phantom_response(response: &str, detection: &PhantomDetection) -> String {
    format!(
        "[PHANTOM WARNING: This response claims tool results but NO tools were called. \
         Matched patterns: {}. Do not trust claims about file contents, command outputs, \
         or system state in this response.]\n\n{}",
        detection.matched_patterns.join(", "),
        response
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_entry(
        tool_name: &str,
        tool_call_id: &str,
        args: serde_json::Value,
        result: &str,
        ok: bool,
    ) -> AuditEntry {
        AuditEntry {
            seq: 0,
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            tool_name: tool_name.to_string(),
            tool_call_id: tool_call_id.to_string(),
            arguments: args,
            result_data: result.to_string(),
            result_ok: ok,
            duration_ms: 10,
            executor: "inline".to_string(),
            hash: String::new(),
            prev_hash: String::new(),
        }
    }

    #[test]
    fn test_observed_file_ref() {
        let entries = vec![make_entry(
            "read_file",
            "c1",
            json!({"path": "/home/user/code.rs"}),
            "fn main() {}",
            true,
        )];
        let verifier = ClaimVerifier::new(&entries);
        let claims =
            verifier.verify("I read `/home/user/code.rs` and it contains a main function.");
        assert!(!claims.is_empty());
        let file_claims: Vec<_> = claims
            .iter()
            .filter(|c| c.claim_type == "file_ref")
            .collect();
        assert!(!file_claims.is_empty());
        assert_eq!(file_claims[0].status, ClaimStatus::Observed);
    }

    #[test]
    fn test_claimed_no_tool_call() {
        let entries: Vec<AuditEntry> = vec![];
        let verifier = ClaimVerifier::new(&entries);
        let claims = verifier.verify("I read `/tmp/secret.txt` and found passwords.");
        let file_claims: Vec<_> = claims
            .iter()
            .filter(|c| c.claim_type == "file_ref")
            .collect();
        if !file_claims.is_empty() {
            // No tool calls = phantom action, should be Claimed (fabrication).
            assert_eq!(file_claims[0].status, ClaimStatus::Claimed);
        }
    }

    #[test]
    fn test_command_ref_observed() {
        let entries = vec![make_entry(
            "exec",
            "c1",
            json!({"command": "cargo test"}),
            "test result: ok. 42 passed",
            true,
        )];
        let verifier = ClaimVerifier::new(&entries);
        let claims = verifier.verify("I ran `cargo test` and all tests passed.");
        let cmd_claims: Vec<_> = claims
            .iter()
            .filter(|c| c.claim_type == "command_ref")
            .collect();
        assert!(!cmd_claims.is_empty());
        assert_eq!(cmd_claims[0].status, ClaimStatus::Observed);
    }

    #[test]
    fn test_action_claim_observed() {
        let entries = vec![make_entry(
            "write_file",
            "c1",
            json!({"path": "/tmp/out.txt"}),
            "ok",
            true,
        )];
        let verifier = ClaimVerifier::new(&entries);
        let claims = verifier.verify("I wrote the output to a file.");
        let action_claims: Vec<_> = claims
            .iter()
            .filter(|c| c.claim_type == "action_claim")
            .collect();
        assert!(!action_claims.is_empty());
        assert_eq!(action_claims[0].status, ClaimStatus::Observed);
    }

    #[test]
    fn test_action_claim_no_matching_tool() {
        let entries = vec![make_entry("read_file", "c1", json!({}), "data", true)];
        let verifier = ClaimVerifier::new(&entries);
        let claims = verifier.verify("I deleted the temporary files.");
        let action_claims: Vec<_> = claims
            .iter()
            .filter(|c| c.claim_type == "action_claim")
            .collect();
        assert!(!action_claims.is_empty());
        // "deleted" maps to "exec" tool, but we only have read_file
        assert_eq!(action_claims[0].status, ClaimStatus::Claimed);
    }

    #[test]
    fn test_numeric_claim_derived() {
        let entries = vec![make_entry(
            "exec",
            "c1",
            json!({}),
            "total 42 files found",
            true,
        )];
        let verifier = ClaimVerifier::new(&entries);
        let claims = verifier.verify("There are 42 files in the directory.");
        let num_claims: Vec<_> = claims
            .iter()
            .filter(|c| c.claim_type == "numeric")
            .collect();
        assert!(!num_claims.is_empty());
        assert_eq!(num_claims[0].status, ClaimStatus::Derived);
    }

    #[test]
    fn test_has_unverified() {
        let entries = vec![make_entry("read_file", "c1", json!({}), "data", true)];
        let verifier = ClaimVerifier::new(&entries);
        let claims = verifier.verify("I deleted the old backups.");
        assert!(verifier.has_unverified(&claims));
    }

    #[test]
    fn test_unverified_summary() {
        let entries = vec![make_entry("read_file", "c1", json!({}), "data", true)];
        let verifier = ClaimVerifier::new(&entries);
        let claims = verifier.verify("I deleted the old backups.");
        let summary = verifier.unverified_summary(&claims);
        assert!(summary.contains("unverified"));
        assert!(summary.contains("action_claim"));
    }

    #[test]
    fn test_empty_text_no_claims() {
        let entries: Vec<AuditEntry> = vec![];
        let verifier = ClaimVerifier::new(&entries);
        let claims = verifier.verify("");
        assert!(claims.is_empty());
    }

    #[test]
    fn test_plain_text_no_claims() {
        let entries: Vec<AuditEntry> = vec![];
        let verifier = ClaimVerifier::new(&entries);
        let claims = verifier.verify("Hello! How can I help you today?");
        assert!(claims.is_empty());
    }

    // --- phantom action detection ---

    #[test]
    fn test_phantom_action_let_me_check() {
        // Agent says "Let me check" but never called any tool.
        let entries: Vec<AuditEntry> = vec![];
        let verifier = ClaimVerifier::new(&entries);
        let claims = verifier.verify("Let me check the time. It's 5:35 PM.");
        // Should detect "Let me check" as a phantom action (Claimed).
        let action_claims: Vec<_> = claims
            .iter()
            .filter(|c| c.claim_type == "action_claim")
            .collect();
        assert!(!action_claims.is_empty());
        assert_eq!(action_claims[0].status, ClaimStatus::Claimed);
    }

    #[test]
    fn test_phantom_action_let_me_run() {
        let entries: Vec<AuditEntry> = vec![];
        let verifier = ClaimVerifier::new(&entries);
        let claims = verifier.verify("Let me run date to get the current time.");
        let action_claims: Vec<_> = claims
            .iter()
            .filter(|c| c.claim_type == "action_claim")
            .collect();
        assert!(!action_claims.is_empty());
        assert_eq!(action_claims[0].status, ClaimStatus::Claimed);
    }

    #[test]
    fn test_let_me_check_with_actual_tool_call() {
        // Agent says "Let me check" AND actually called exec.
        let entries = vec![make_entry(
            "exec",
            "c1",
            json!({"command": "date"}),
            "Fri Feb 13 17:35:00 CET 2026",
            true,
        )];
        let verifier = ClaimVerifier::new(&entries);
        let claims = verifier.verify("Let me check the time. It's 17:35.");
        let action_claims: Vec<_> = claims
            .iter()
            .filter(|c| c.claim_type == "action_claim")
            .collect();
        assert!(!action_claims.is_empty());
        // Tool WAS called → Observed, not Claimed.
        assert_eq!(action_claims[0].status, ClaimStatus::Observed);
    }

    // --- outcome claims ---

    #[test]
    fn test_outcome_claim_success_matches_ok_tool() {
        let entries = vec![make_entry(
            "exec",
            "c1",
            json!({"command": "cargo build"}),
            "Compiling...\nFinished",
            true,
        )];
        let verifier = ClaimVerifier::new(&entries);
        let claims = verifier.verify("Build succeeded. Now let me copy the binary.");
        let outcome_claims: Vec<_> = claims
            .iter()
            .filter(|c| c.claim_type == "outcome")
            .collect();
        assert!(!outcome_claims.is_empty());
        assert_eq!(outcome_claims[0].status, ClaimStatus::Derived);
    }

    #[test]
    fn test_outcome_claim_success_contradicts_failed_tool() {
        let entries = vec![make_entry(
            "exec",
            "c1",
            json!({"command": "cargo build"}),
            "error[E0308]: mismatched types",
            false,
        )];
        let verifier = ClaimVerifier::new(&entries);
        let claims = verifier.verify("Build succeeded. Now let me install.");
        let outcome_claims: Vec<_> = claims
            .iter()
            .filter(|c| c.claim_type == "outcome")
            .collect();
        assert!(!outcome_claims.is_empty());
        assert_eq!(outcome_claims[0].status, ClaimStatus::Claimed);
    }

    #[test]
    fn test_outcome_claim_failure_matches_failed_tool() {
        let entries = vec![make_entry(
            "exec",
            "c1",
            json!({"command": "cargo build"}),
            "error: aborting due to previous error",
            false,
        )];
        let verifier = ClaimVerifier::new(&entries);
        let claims = verifier.verify("Build failed. Let me fix the error.");
        let outcome_claims: Vec<_> = claims
            .iter()
            .filter(|c| c.claim_type == "outcome")
            .collect();
        assert!(!outcome_claims.is_empty());
        assert_eq!(outcome_claims[0].status, ClaimStatus::Derived);
    }

    // --- timestamp claims ---

    #[test]
    fn test_timestamp_claim_from_tool_output() {
        let entries = vec![make_entry(
            "exec",
            "c1",
            json!({"command": "stat file"}),
            "Modify: 2026-02-13 17:36:00",
            true,
        )];
        let verifier = ClaimVerifier::new(&entries);
        let claims = verifier.verify("The binary is from 17:36.");
        let ts_claims: Vec<_> = claims
            .iter()
            .filter(|c| c.claim_type == "timestamp")
            .collect();
        assert!(!ts_claims.is_empty());
        assert_eq!(ts_claims[0].status, ClaimStatus::Derived);
    }

    #[test]
    fn test_timestamp_claim_fabricated() {
        let entries = vec![make_entry(
            "exec",
            "c1",
            json!({"command": "stat file"}),
            "Modify: 2026-02-13 17:36:00",
            true,
        )];
        let verifier = ClaimVerifier::new(&entries);
        let claims = verifier.verify("The binary is from 17:45.");
        let ts_claims: Vec<_> = claims
            .iter()
            .filter(|c| c.claim_type == "timestamp")
            .collect();
        assert!(!ts_claims.is_empty());
        // 17:45 is NOT in any tool output → fabricated
        assert_eq!(ts_claims[0].status, ClaimStatus::Claimed);
    }

    // --- verify_turn_claims ---

    #[test]
    fn test_verify_turn_claims_no_fabrication() {
        let entries = vec![make_entry(
            "write_file",
            "c1",
            json!({"path": "/tmp/out.txt"}),
            "ok",
            true,
        )];
        let (claims, has_fab) = verify_turn_claims("I wrote the output to a file.", &entries);
        assert!(!claims.is_empty());
        assert!(!has_fab);
    }

    #[test]
    fn test_verify_turn_claims_with_fabrication() {
        let entries = vec![make_entry("read_file", "c1", json!({}), "data", true)];
        let (claims, has_fab) = verify_turn_claims("I deleted the old backups.", &entries);
        assert!(has_fab);
    }

    // --- redact_fabrications ---

    #[test]
    fn test_redact_fabrications_empty_input() {
        let (result, count) = redact_fabrications("", &[]);
        assert_eq!(result, "");
        assert_eq!(count, 0);
    }

    #[test]
    fn test_redact_fabrications_no_claimed() {
        let claims = vec![AnnotatedClaim {
            span: (0, 5),
            claim_type: "action_claim".to_string(),
            status: ClaimStatus::Observed,
            text: "hello".to_string(),
        }];
        let (result, count) = redact_fabrications("hello world", &claims);
        assert_eq!(result, "hello world");
        assert_eq!(count, 0);
    }

    #[test]
    fn test_redact_fabrications_single_span() {
        let text = "I deleted the old backups and cleaned up.";
        let claims = vec![AnnotatedClaim {
            span: (0, 25),
            claim_type: "action_claim".to_string(),
            status: ClaimStatus::Claimed,
            text: "I deleted the old backups".to_string(),
        }];
        let (result, count) = redact_fabrications(text, &claims);
        assert_eq!(count, 1);
        assert!(result.contains("[unverified claim removed]"));
        assert!(result.contains("and cleaned up."));
    }

    #[test]
    fn test_redact_fabrications_preserves_offsets() {
        // Two claims: the second claim's offsets should still be valid after
        // the first is redacted (because we process from end to start).
        let text = "AAA BBB CCC";
        let claims = vec![
            AnnotatedClaim {
                span: (0, 3),
                claim_type: "numeric".to_string(),
                status: ClaimStatus::Claimed,
                text: "AAA".to_string(),
            },
            AnnotatedClaim {
                span: (8, 11),
                claim_type: "numeric".to_string(),
                status: ClaimStatus::Claimed,
                text: "CCC".to_string(),
            },
        ];
        let (result, count) = redact_fabrications(text, &claims);
        assert_eq!(count, 2);
        assert!(result.contains("BBB"));
        assert!(!result.contains("AAA"));
        assert!(!result.contains("CCC"));
    }

    // --- detect_phantom_claims ---

    #[test]
    fn test_phantom_detection_no_tools_with_claims() {
        let result = detect_phantom_claims(
            "I read the file and the output shows the configuration.",
            &[], // no tools called
        );
        assert!(result.is_some(), "should detect phantom claims");
        let detection = result.unwrap();
        assert!(detection
            .matched_patterns
            .iter()
            .any(|p| p.contains("I read the file")));
        assert!(detection
            .matched_patterns
            .iter()
            .any(|p| p.contains("the output shows")));
    }

    #[test]
    fn test_phantom_detection_no_tools_no_claims() {
        let result = detect_phantom_claims("Here is a general explanation of Rust ownership.", &[]);
        assert!(
            result.is_none(),
            "should not detect phantom when no tool language used"
        );
    }

    #[test]
    fn test_phantom_detection_with_tools() {
        let result = detect_phantom_claims(
            "I read the file and it contains configuration data.",
            &["read_file".to_string()],
        );
        assert!(
            result.is_none(),
            "should not flag when tools were actually called"
        );
    }

    #[test]
    fn test_phantom_detection_case_insensitive() {
        let result = detect_phantom_claims("I EXECUTED the command successfully.", &[]);
        assert!(result.is_some(), "should be case-insensitive");
    }

    #[test]
    fn test_phantom_detection_let_me_patterns() {
        let result = detect_phantom_claims(
            "Let me check the file and show you the contents.\n\nThe file contains: foo bar baz",
            &[],
        );
        assert!(
            result.is_some(),
            "should detect 'let me check' phantom pattern"
        );
        let detection = result.unwrap();
        assert!(detection
            .matched_patterns
            .iter()
            .any(|p| p.contains("let me check")));
    }

    #[test]
    fn test_phantom_annotation() {
        let detection = PhantomDetection {
            system_warning: "test warning".to_string(),
            matched_patterns: vec!["I read the file".to_string()],
        };
        let annotated = annotate_phantom_response("I read the file and it says hello.", &detection);
        assert!(annotated.starts_with("[PHANTOM WARNING:"));
        assert!(annotated.contains("I read the file"));
        assert!(annotated.contains("and it says hello"));
    }
}
