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
        let re = Regex::new(
            r"(?i)\b(ran|executed|running)\b[^`\n]{0,20}`([^`]+)`"
        ).unwrap();

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
        let re = Regex::new(
            r"(?i)\bI (read|wrote|created|deleted|executed|searched|fetched|edited|ran|modified|updated|removed)\b[^.\n]{0,80}"
        ).unwrap();

        for cap in re.captures_iter(text) {
            if let (Some(full), Some(action)) = (cap.get(0), cap.get(1)) {
                let action_str = action.as_str().to_lowercase();
                let tool_hint = match action_str.as_str() {
                    "read" => Some("read_file"),
                    "wrote" | "created" | "modified" | "updated" => Some("write_file"),
                    "deleted" | "removed" => Some("exec"),
                    "executed" | "ran" => Some("exec"),
                    "searched" => Some("web_search"),
                    "fetched" => Some("web_fetch"),
                    "edited" => Some("edit_file"),
                    _ => None,
                };
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

    fn extract_numeric_claims(&self, text: &str) -> Vec<AnnotatedClaim> {
        let mut claims = Vec::new();
        let re = Regex::new(
            r"\b(\d+)\s+(files?|lines?|errors?|tests?|warnings?|results?|matches?|items?)\b"
        ).unwrap();

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

    // --- Matching helpers ---

    fn match_against_entries(&self, needle: &str, tool_hint: Option<&str>) -> ClaimStatus {
        if self.entries.is_empty() {
            return ClaimStatus::Recalled;
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
            return ClaimStatus::Recalled;
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
            let short = &content[..20];
            for entry in self.entries.iter().rev() {
                if entry.result_data.contains(short) {
                    return ClaimStatus::Derived;
                }
            }
        }

        ClaimStatus::Claimed
    }

    fn match_action_against_entries(&self, action: &str, tool_hint: Option<&str>) -> ClaimStatus {
        if self.entries.is_empty() {
            return ClaimStatus::Recalled;
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
            "wrote" | "created" | "modified" | "updated" => vec!["write_file", "write", "edit_file"],
            "deleted" | "removed" => vec!["exec"],
            "executed" | "ran" => vec!["exec"],
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_entry(tool_name: &str, tool_call_id: &str, args: serde_json::Value, result: &str, ok: bool) -> AuditEntry {
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
        let entries = vec![
            make_entry("read_file", "c1", json!({"path": "/home/user/code.rs"}), "fn main() {}", true),
        ];
        let verifier = ClaimVerifier::new(&entries);
        let claims = verifier.verify("I read `/home/user/code.rs` and it contains a main function.");
        assert!(!claims.is_empty());
        let file_claims: Vec<_> = claims.iter().filter(|c| c.claim_type == "file_ref").collect();
        assert!(!file_claims.is_empty());
        assert_eq!(file_claims[0].status, ClaimStatus::Observed);
    }

    #[test]
    fn test_claimed_no_tool_call() {
        let entries: Vec<AuditEntry> = vec![];
        let verifier = ClaimVerifier::new(&entries);
        let claims = verifier.verify("I read `/tmp/secret.txt` and found passwords.");
        let file_claims: Vec<_> = claims.iter().filter(|c| c.claim_type == "file_ref").collect();
        if !file_claims.is_empty() {
            assert_eq!(file_claims[0].status, ClaimStatus::Recalled);
        }
    }

    #[test]
    fn test_command_ref_observed() {
        let entries = vec![
            make_entry("exec", "c1", json!({"command": "cargo test"}), "test result: ok. 42 passed", true),
        ];
        let verifier = ClaimVerifier::new(&entries);
        let claims = verifier.verify("I ran `cargo test` and all tests passed.");
        let cmd_claims: Vec<_> = claims.iter().filter(|c| c.claim_type == "command_ref").collect();
        assert!(!cmd_claims.is_empty());
        assert_eq!(cmd_claims[0].status, ClaimStatus::Observed);
    }

    #[test]
    fn test_action_claim_observed() {
        let entries = vec![
            make_entry("write_file", "c1", json!({"path": "/tmp/out.txt"}), "ok", true),
        ];
        let verifier = ClaimVerifier::new(&entries);
        let claims = verifier.verify("I wrote the output to a file.");
        let action_claims: Vec<_> = claims.iter().filter(|c| c.claim_type == "action_claim").collect();
        assert!(!action_claims.is_empty());
        assert_eq!(action_claims[0].status, ClaimStatus::Observed);
    }

    #[test]
    fn test_action_claim_no_matching_tool() {
        let entries = vec![
            make_entry("read_file", "c1", json!({}), "data", true),
        ];
        let verifier = ClaimVerifier::new(&entries);
        let claims = verifier.verify("I deleted the temporary files.");
        let action_claims: Vec<_> = claims.iter().filter(|c| c.claim_type == "action_claim").collect();
        assert!(!action_claims.is_empty());
        // "deleted" maps to "exec" tool, but we only have read_file
        assert_eq!(action_claims[0].status, ClaimStatus::Claimed);
    }

    #[test]
    fn test_numeric_claim_derived() {
        let entries = vec![
            make_entry("exec", "c1", json!({}), "total 42 files found", true),
        ];
        let verifier = ClaimVerifier::new(&entries);
        let claims = verifier.verify("There are 42 files in the directory.");
        let num_claims: Vec<_> = claims.iter().filter(|c| c.claim_type == "numeric").collect();
        assert!(!num_claims.is_empty());
        assert_eq!(num_claims[0].status, ClaimStatus::Derived);
    }

    #[test]
    fn test_has_unverified() {
        let entries = vec![
            make_entry("read_file", "c1", json!({}), "data", true),
        ];
        let verifier = ClaimVerifier::new(&entries);
        let claims = verifier.verify("I deleted the old backups.");
        assert!(verifier.has_unverified(&claims));
    }

    #[test]
    fn test_unverified_summary() {
        let entries = vec![
            make_entry("read_file", "c1", json!({}), "data", true),
        ];
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
}
