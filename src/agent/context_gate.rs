// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(clippy::as_conversions, clippy::indexing_slicing)]
//! Context Gate: intelligent content management for LLM agents.
//!
//! Instead of uniform char-limit truncation, the gate makes context-aware
//! decisions based on the model's token budget:
//! - **Pass raw** when content fits
//! - **Briefing** (structural summary via compactor) when it doesn't

use serde_json::Value;

use crate::agent::token_budget::TokenBudget;

// ---------------------------------------------------------------------------
// GateResult
// ---------------------------------------------------------------------------

/// What the gate returns after processing content.
pub enum GateResult {
    /// Content fits — pass through unchanged.
    Raw(String),
    /// Content was too large — summarized.
    Briefing { summary: String },
}

impl GateResult {
    /// Consume into the text string.
    pub fn into_text(self) -> String {
        match self {
            GateResult::Raw(s) => s,
            GateResult::Briefing { summary } => summary,
        }
    }
}

// ---------------------------------------------------------------------------
// ContentGate
// ---------------------------------------------------------------------------

/// Single entry point for ALL content entering the agent's context.
///
/// Decides: pass raw (fits) or produce a briefing (doesn't fit).
pub struct ContentGate {
    pub budget: TokenBudget,
}

impl ContentGate {
    pub fn new(max_tokens: usize, output_reserve: f32) -> Self {
        Self {
            budget: TokenBudget::with_output_reserve(max_tokens, output_reserve),
        }
    }

    /// Gate content entering the agent's context.
    ///
    /// If content fits the remaining budget, passes it through raw.
    /// Otherwise, caches the full content and produces a structural briefing.
    ///
    /// `briefing_fn` is called only when content doesn't fit — it receives
    /// the content and a target token count, and should return a structural
    /// summary. This keeps ContentGate decoupled from the compactor.
    pub fn admit<F>(&mut self, content: &str, briefing_fn: F) -> GateResult
    where
        F: FnOnce(&str, usize) -> String,
    {
        let tokens = TokenBudget::estimate_str_tokens(content);
        let available = self.budget.available();

        if tokens <= available {
            self.budget.consume(tokens);
            return GateResult::Raw(content.to_string());
        }

        // Content doesn't fit — produce a briefing.
        let target_tokens = available / 2; // briefing should use ~half remaining budget
        let summary = briefing_fn(content, target_tokens);
        let summary_tokens = TokenBudget::estimate_str_tokens(&summary);
        self.budget.consume(summary_tokens);

        GateResult::Briefing { summary }
    }

    /// Admit content with a simple fallback briefing (no LLM summarization).
    ///
    /// Produces a mechanical structural map: line counts, first/last lines,
    /// and a navigation hint. Useful when no compactor is available.
    pub fn admit_simple(&mut self, content: &str) -> GateResult {
        self.admit(content, |c, target_tokens| {
            if let Some(json_summary) = build_json_briefing(c, target_tokens) {
                json_summary
            } else {
                build_simple_briefing(c, target_tokens)
            }
        })
    }
}

/// Build a mechanical briefing without LLM summarization.
///
/// Includes: total lines, byte size, structural sketch (first N lines),
/// and a navigation hint for drill-down.
fn build_simple_briefing(content: &str, target_tokens: usize) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();
    let total_bytes = content.len();

    // Estimate how many preview lines fit in target_tokens.
    // ~4 chars per token, conservative.
    let char_budget = target_tokens * 3;

    let mut preview = String::new();
    let mut chars_used = 0;

    // Show first lines within budget.
    let mut shown = 0;
    for (i, line) in lines.iter().enumerate() {
        let entry = format!("{:>4}: {}\n", i + 1, line);
        if chars_used + entry.len() > char_budget / 2 {
            break;
        }
        preview.push_str(&entry);
        chars_used += entry.len();
        shown = i + 1;
    }

    // If we didn't show everything, add last few lines.
    if shown < total_lines {
        preview.push_str("  ...\n");
        let tail_start = total_lines.saturating_sub(5);
        if tail_start > shown {
            for i in tail_start..total_lines {
                let entry = format!("{:>4}: {}\n", i + 1, lines[i]);
                if chars_used + entry.len() > char_budget {
                    break;
                }
                preview.push_str(&entry);
                chars_used += entry.len();
            }
        }
    }

    format!(
        "# Content Summary ({} lines, {} bytes)\n\n\
         ## Preview\n```\n{}```\n\n\
         To inspect a section, use: read_file with lines parameter (e.g. lines=\"{}:{}\")",
        total_lines,
        total_bytes,
        preview,
        shown.max(1),
        (shown + 50).min(total_lines),
    )
}

fn parse_json_content(content: &str) -> Option<Value> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        return Some(v);
    }

    if trimmed.starts_with("```") {
        let first_nl = trimmed.find('\n')?;
        let last_fence = trimmed.rfind("```")?;
        if last_fence > first_nl {
            let body = &trimmed[first_nl + 1..last_fence];
            if let Ok(v) = serde_json::from_str::<Value>(body.trim()) {
                return Some(v);
            }
        }
    }

    None
}

fn signal_key(key: &str) -> bool {
    let k = key.to_ascii_lowercase();
    k.contains("id")
        || k.contains("error")
        || k.contains("status")
        || k.contains("path")
        || k.contains("request")
        || k.contains("latency")
        || k.contains("count")
        || k.contains("total")
        || k.contains("fail")
        || k.contains("code")
        || k.contains("max")
        || k.contains("min")
        || k.contains("message")
        || k.contains("msg")
        || k.contains("reason")
        || k.contains("detail")
}

fn scalar_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => {
            if s.is_empty() {
                None
            } else {
                Some(format!("\"{}\"", s.chars().take(180).collect::<String>()))
            }
        }
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        Value::Null => Some("null".to_string()),
        _ => None,
    }
}

fn is_ok_status(s: &str) -> bool {
    matches!(
        s.to_ascii_lowercase().as_str(),
        "ok" | "success" | "passed" | "pass" | "none"
    )
}

/// Extract a human-readable error message from an `error` field that may be
/// either a plain string (`"error": "checksum mismatch"`) or an object shaped
/// like `{"error": {"code": 500, "message": "..."}}` (common in batch/REST
/// responses). For objects, prefers `message`, then `reason`, `detail`, `msg`.
/// Returns the truncated message without surrounding quotes.
fn error_message_value(error: &Value) -> Option<String> {
    match error {
        Value::String(s) => {
            if s.trim().is_empty() {
                None
            } else {
                Some(s.chars().take(120).collect())
            }
        }
        Value::Object(map) => ["message", "reason", "detail", "msg"]
            .iter()
            .find_map(|k| map.get(*k).and_then(|v| v.as_str()))
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.chars().take(120).collect()),
        _ => None,
    }
}

fn anomaly_line_for_object(path: &str, map: &serde_json::Map<String, Value>) -> Option<String> {
    let status = map.get("status").and_then(|v| v.as_str());
    let error = map.get("error").and_then(error_message_value);
    let latency = map.get("latencyMs").and_then(|v| v.as_i64());

    let mut notes = Vec::new();
    if let Some(s) = status {
        if !is_ok_status(s) {
            notes.push(format!("status=\"{}\"", s));
        }
    }
    if let Some(e) = error {
        if !e.trim().is_empty() {
            notes.push(format!(
                "error=\"{}\"",
                e.chars().take(120).collect::<String>()
            ));
        }
    }
    if let Some(ms) = latency {
        if ms >= 1000 {
            notes.push(format!("latencyMs={}", ms));
        }
    }

    if notes.is_empty() {
        None
    } else {
        Some(format!("{}: {}", path, notes.join(", ")))
    }
}

fn collect_signal_facts(value: &Value, path: &str, out: &mut Vec<String>, limit: usize) {
    if out.len() >= limit {
        return;
    }

    match value {
        Value::Object(map) => {
            let mut scalar_facts = Vec::new();
            for (k, v) in map {
                if signal_key(k) {
                    if let Some(s) = scalar_string(v) {
                        let child = if path.is_empty() {
                            k.to_string()
                        } else {
                            format!("{}.{}", path, k)
                        };
                        scalar_facts.push((k.to_ascii_lowercase(), format!("{} = {}", child, s)));
                    }
                }
            }

            scalar_facts.sort_by_key(|(k, _)| {
                if k.contains("requestid") || k == "request" {
                    0usize
                } else if k.contains("error") {
                    1usize
                } else if k.contains("status") {
                    2usize
                } else if k.contains("path") {
                    3usize
                } else if k.contains("latency") || k.contains("max") {
                    4usize
                } else {
                    10usize
                }
            });

            for (_, line) in scalar_facts {
                if out.len() >= limit {
                    return;
                }
                out.push(line);
            }

            if let Some(anomaly) = anomaly_line_for_object(path, map) {
                out.push(anomaly);
                if out.len() >= limit {
                    return;
                }
            }

            for (k, v) in map {
                if out.len() >= limit {
                    return;
                }
                let child = if path.is_empty() {
                    k.to_string()
                } else {
                    format!("{}.{}", path, k)
                };
                if matches!(v, Value::Object(_) | Value::Array(_)) {
                    collect_signal_facts(v, &child, out, limit);
                }
            }
        }
        Value::Array(arr) => {
            // First pass: anomalies across the full array so we do not miss critical rows.
            let mut anomaly_indexes = std::collections::BTreeSet::new();
            for (i, v) in arr.iter().enumerate() {
                if out.len() >= limit {
                    return;
                }
                if let Value::Object(map) = v {
                    let child = format!("{}[{}]", path, i);
                    if let Some(anomaly) = anomaly_line_for_object(&child, map) {
                        out.push(anomaly);
                        anomaly_indexes.insert(i);
                        for key in ["id", "invoiceId", "path", "error", "latencyMs", "status"] {
                            if out.len() >= limit {
                                return;
                            }
                            if let Some(val) = map.get(key).and_then(scalar_string) {
                                out.push(format!("{}.{} = {}", child, key, val));
                            }
                        }
                        // Object-valued errors (e.g. `{"error":{"code":500,"message":"..."}}`)
                        // are dropped by `scalar_string` above; descend one level so the
                        // nested `code`/`message`/`reason`/`detail` survive in Extracted
                        // Facts. This replaces the need to recurse the whole anomaly item
                        // (which would duplicate the scalar sub-facts already emitted).
                        if let Some(error_map) = map.get("error").and_then(Value::as_object) {
                            for key in ["code", "message", "reason", "detail", "msg"] {
                                if out.len() >= limit {
                                    return;
                                }
                                if let Some(val) = error_map.get(key).and_then(scalar_string) {
                                    out.push(format!("{}.error.{} = {}", child, key, val));
                                }
                            }
                        }
                    }
                }
            }

            for (i, v) in arr.iter().enumerate() {
                if out.len() >= limit {
                    return;
                }

                if anomaly_indexes.contains(&i) {
                    continue;
                }

                let should_scan = i < 4 || i + 1 == arr.len();
                if !should_scan {
                    if let Value::Object(map) = v {
                        if anomaly_line_for_object(&format!("{}[{}]", path, i), map).is_none() {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }

                let child = format!("{}[{}]", path, i);
                collect_signal_facts(v, &child, out, limit);
            }
        }
        _ => {}
    }
}

fn collect_key_lines(content: &str, limit: usize) -> Vec<String> {
    let mut out = Vec::new();
    for line in content.lines() {
        if out.len() >= limit {
            break;
        }
        let lower = line.to_ascii_lowercase();
        if lower.contains("\"error\"")
            || lower.contains("\"status\"")
            || lower.contains("\"requestid\"")
            || lower.contains("\"latency")
            || lower.contains("\"path\"")
            || lower.contains("\"invoice")
            || lower.contains("\"message\"")
            || lower.contains("\"reason\"")
            || lower.contains("\"detail\"")
        {
            out.push(line.trim().chars().take(220).collect::<String>());
        }
    }
    out
}

fn build_json_briefing(content: &str, target_tokens: usize) -> Option<String> {
    let parsed = parse_json_content(content)?;
    let target = target_tokens.max(200);
    let fact_limit = (target / 24).clamp(10, 40);

    let mut structure = Vec::new();
    match &parsed {
        Value::Object(map) => {
            let keys = map.keys().take(20).cloned().collect::<Vec<_>>();
            structure.push("root: object".to_string());
            if !keys.is_empty() {
                structure.push(format!("top_keys: {}", keys.join(", ")));
            }
            for (k, v) in map {
                if let Value::Array(_) = v {
                    structure.push(format!("array {}", k));
                }
            }
        }
        Value::Array(_) => structure.push("root: array".to_string()),
        other => structure.push(format!("root: {}", other)),
    }

    let mut facts = Vec::new();
    collect_signal_facts(&parsed, "$", &mut facts, fact_limit);
    let key_lines = collect_key_lines(content, (fact_limit / 2).max(4));

    let mut out = String::new();
    out.push_str("# JSON Summary\n\n");
    out.push_str("## Structure\n");
    for item in structure.into_iter().take(12) {
        out.push_str("- ");
        out.push_str(&item);
        out.push('\n');
    }

    if !facts.is_empty() {
        out.push_str("\n## Extracted Facts\n");
        for fact in facts.into_iter().take(fact_limit) {
            out.push_str("- ");
            out.push_str(&fact);
            out.push('\n');
        }
    }

    if !key_lines.is_empty() {
        out.push_str("\n## Key Source Lines\n");
        out.push_str("```json\n");
        for line in key_lines {
            out.push_str(&line);
            out.push('\n');
        }
        out.push_str("```\n");
    }

    Some(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- TokenBudget (with_output_reserve) tests --

    #[test]
    fn test_budget_available_basic() {
        let budget = TokenBudget::with_output_reserve(10_000, 0.20);
        // ceiling = 10_000 * 0.80 = 8_000
        assert_eq!(budget.available(), 8_000);
    }

    #[test]
    fn test_budget_consume_and_available() {
        let mut budget = TokenBudget::with_output_reserve(10_000, 0.20);
        budget.consume(3_000);
        assert_eq!(budget.available(), 5_000);
        assert_eq!(budget.used(), 3_000);
    }

    #[test]
    fn test_budget_saturating_consume() {
        let mut budget = TokenBudget::with_output_reserve(1_000, 0.20);
        budget.consume(900);
        // available = 800 - 900 = 0 (saturating)
        assert_eq!(budget.available(), 0);
    }

    #[test]
    fn test_budget_reset() {
        let mut budget = TokenBudget::with_output_reserve(10_000, 0.20);
        budget.consume(5_000);
        budget.reset_used(1_000);
        assert_eq!(budget.used(), 1_000);
        assert_eq!(budget.available(), 7_000);
    }

    #[test]
    fn test_budget_clamp_output_reserve() {
        let budget = TokenBudget::with_output_reserve(10_000, 1.5); // clamped to 0.95
        assert_eq!(budget.available(), 500); // 10_000 * 0.05
    }

    #[test]
    fn test_estimate_tokens_delegates_to_bpe() {
        let tokens = TokenBudget::estimate_str_tokens("hello world");
        assert!(tokens > 0 && tokens < 10);
    }

    // -- ContentGate tests --

    #[test]
    fn test_gate_raw_when_fits() {
        let mut gate = ContentGate::new(100_000, 0.20);

        let content = "short content";
        let result = gate.admit_simple(content);
        match result {
            GateResult::Raw(s) => assert_eq!(s, content),
            GateResult::Briefing { .. } => panic!("expected Raw"),
        }
    }

    #[test]
    fn test_gate_briefing_when_too_large() {
        // Tiny budget: 100 tokens, 20% reserve = 80 available.
        let mut gate = ContentGate::new(100, 0.20);

        // Content that definitely exceeds 80 tokens.
        let content = "x\n".repeat(500);
        let result = gate.admit_simple(&content);
        match result {
            GateResult::Raw(_) => panic!("expected Briefing"),
            GateResult::Briefing { summary } => {
                assert!(summary.contains("Content Summary"));
                assert!(summary.contains("read_file"));
            }
        }
    }

    #[test]
    fn test_gate_custom_briefing_fn() {
        let mut gate = ContentGate::new(50, 0.20);

        let content = "a\n".repeat(200);
        let result = gate.admit(&content, |_c, _target| "custom briefing".to_string());
        match result {
            GateResult::Briefing { summary } => {
                assert_eq!(summary, "custom briefing");
            }
            GateResult::Raw(_) => panic!("expected Briefing"),
        }
    }

    #[test]
    fn test_gate_result_into_text() {
        let raw = GateResult::Raw("hello".to_string());
        assert_eq!(raw.into_text(), "hello");

        let briefing = GateResult::Briefing {
            summary: "summary".to_string(),
        };
        assert_eq!(briefing.into_text(), "summary");
    }

    #[test]
    fn test_simple_briefing_format() {
        let content = (1..=100)
            .map(|i| format!("line number {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        let briefing = build_simple_briefing(&content, 200);

        assert!(briefing.contains("100 lines"));
        assert!(briefing.contains("Content Summary"));
        assert!(briefing.contains("read_file"));
        assert!(briefing.contains("line number 1"));
    }

    #[test]
    fn test_json_briefing_extracts_signal_values() {
        let content = r#"{
  "requestId": "req-7f3d9ab2",
  "entries": [
    {"id": 1, "status": "ok", "latencyMs": 140, "error": ""},
    {"id": 173, "status": "error", "latencyMs": 9821, "error": "checksum mismatch for invoice hash", "invoiceId": "INV-2026-0173", "path": "/v1/invoices/reconcile"}
  ]
}"#;
        let briefing = build_json_briefing(content, 360).expect("json briefing expected");
        assert!(briefing.contains("JSON Summary"));
        assert!(briefing.contains("req-7f3d9ab2"));
        assert!(briefing.contains("INV-2026-0173"));
        assert!(briefing.contains("checksum mismatch for invoice hash"));
    }

    #[test]
    fn test_json_briefing_parses_fenced_json() {
        let content = "```json\n{\"status\":\"error\",\"requestId\":\"abc-1\"}\n```";
        let briefing = build_json_briefing(content, 220).expect("fenced json should parse");
        assert!(briefing.contains("abc-1"));
        assert!(briefing.contains("status"));
    }

    #[test]
    fn test_budget_tracks_across_multiple_admits() {
        let mut gate = ContentGate::new(10_000, 0.20);
        // available = 8_000

        let initial = gate.budget.available();
        let content = "hello world";
        let _ = gate.admit_simple(content);
        let after = gate.budget.available();

        assert!(after < initial, "budget should decrease after admit");
    }

    // -- JSON briefing: object-valued embedded errors (the bug) --

    #[test]
    fn test_json_briefing_object_error_extracts_code_and_message() {
        // Object-valued `error` (common in batch/REST responses) must surface
        // both its `code` and `message` in Extracted Facts — not just
        // `status="error"`. Previously every extractor assumed `error` was a
        // string and silently dropped the object case.
        let content = r#"{
  "requestId": "batch-42",
  "results": [
    {"id": 1, "status": "ok", "latencyMs": 11},
    {"id": 13, "status": "error", "error": {"code": 500, "message": "database connection refused"}}
  ]
}"#;
        let briefing = build_json_briefing(content, 360).expect("json briefing expected");
        assert!(briefing.contains("JSON Summary"));
        assert!(
            briefing.contains("error.code = 500"),
            "nested error.code missing from Extracted Facts: {briefing}"
        );
        assert!(
            briefing.contains("error.message = \"database connection refused\""),
            "nested error.message missing from Extracted Facts: {briefing}"
        );
        // The anomaly line must carry the message too, not just `status="error"`.
        assert!(
            briefing.contains("error=\"database connection refused\""),
            "anomaly line dropped object-valued error message: {briefing}"
        );
    }

    #[test]
    fn test_json_briefing_object_error_reason_and_detail_variants() {
        // error objects that use `reason`/`detail` (or `msg`) instead of
        // `message` should also surface, and the anomaly line picks the first
        // available message-like field.
        let content = r#"{"results":[{"id":1,"status":"error","error":{"code":503,"reason":"upstream timeout","detail":"db pool exhausted"}}]}"#;
        let briefing = build_json_briefing(content, 360).expect("json briefing expected");
        assert!(
            briefing.contains("error.code = 503"),
            "code missing: {briefing}"
        );
        assert!(
            briefing.contains("error.reason = \"upstream timeout\""),
            "reason missing: {briefing}"
        );
        assert!(
            briefing.contains("error.detail = \"db pool exhausted\""),
            "detail missing: {briefing}"
        );
        // anomaly line prefers message -> reason -> detail -> msg; `reason` wins.
        assert!(
            briefing.contains("error=\"upstream timeout\""),
            "anomaly line missing reason: {briefing}"
        );
    }

    #[test]
    fn test_json_briefing_pretty_printed_message_survives_in_key_lines() {
        // Pretty-printed JSON (the output format of several tools in this
        // codebase) places `"message": "..."` on its own line. That line must
        // survive in Key Source Lines — previously `collect_key_lines` did not
        // match `"message"`, so the message was dropped for pretty-print bodies.
        let content = r#"{
  "requestId": "batch-42",
  "results": [
    {"id": 1, "status": "ok", "latencyMs": 11},
    {
      "id": 13,
      "status": "error",
      "error": {
        "code": 500,
        "message": "database connection refused"
      }
    }
  ]
}"#;
        let briefing = build_json_briefing(content, 360).expect("json briefing expected");
        assert!(
            briefing.contains("\"message\": \"database connection refused\""),
            "message line dropped from Key Source Lines: {briefing}"
        );
        assert!(
            briefing.contains("error.code = 500"),
            "nested error.code missing: {briefing}"
        );
    }

    #[test]
    fn test_json_briefing_root_object_message_extracted() {
        // `signal_key` now treats `message` as a signal key, so a message
        // anywhere in the structure (here at the root, outside any array anomaly)
        // is recovered by the recursive pass rather than silently dropped. Use
        // the exact Extracted Facts form so the assertion isolates the
        // `signal_key` path from any Key Source Lines leak.
        let content = r#"{"status":"error","message":"root failure","requestId":"r-1"}"#;
        let briefing = build_json_briefing(content, 240).expect("json briefing expected");
        assert!(
            briefing.contains("$.message = \"root failure\""),
            "root-level message not extracted as a signal fact: {briefing}"
        );
        assert!(briefing.contains("r-1"), "requestId regression: {briefing}");
    }

    #[test]
    fn test_json_briefing_string_error_still_surfaces() {
        // Regression guard: the original string-valued `error` shape that the
        // summarizer was written against must keep working unchanged.
        let content = r#"{"results":[{"id":7,"status":"error","error":"checksum mismatch"}]}"#;
        let briefing = build_json_briefing(content, 240).expect("json briefing expected");
        assert!(
            briefing.contains("checksum mismatch"),
            "string-valued error regressed: {briefing}"
        );
        assert!(
            briefing.contains("error=\"checksum mismatch\""),
            "string-valued anomaly line regressed: {briefing}"
        );
    }

    #[test]
    fn test_e2e_depleted_budget_embedded_error() {
        // End-to-end via `admit_simple` with a depleted budget, mirroring the
        // bug report scenario: an inline (<= 4096-byte) JSON body containing an
        // embedded error object is summarized by `build_json_briefing`. Both the
        // error code and human-readable message must survive the briefing.
        let mut ok_items = String::new();
        for i in 1..=12 {
            ok_items.push_str(&format!(
                "    {{\"id\": {}, \"status\": \"ok\", \"latencyMs\": {}}},\n",
                i,
                10 + i
            ));
        }
        let content = format!(
            "{{\n  \"requestId\": \"batch-42\",\n  \"results\": [\n{}    {{\"id\": 13, \"status\": \"error\", \"error\": {{\"code\": 500, \"message\": \"database connection refused\"}}}}\n  ]\n}}",
            ok_items
        );
        assert!(
            content.len() <= 4096,
            "body must be inline-sized to reach build_json_briefing: {} bytes",
            content.len()
        );

        // 8000-token context, 20% reserve -> ceiling 6400. Deplete so the
        // remaining budget is far below the body's token estimate.
        let mut gate = ContentGate::new(8_000, 0.20);
        gate.budget.consume(6_370);
        assert!(
            TokenBudget::estimate_str_tokens(&content) > gate.budget.available(),
            "body must exceed remaining budget to trigger the briefing path"
        );

        let text = gate.admit_simple(&content).into_text();
        assert!(
            text.contains("JSON Summary"),
            "expected JSON briefing path to run: {text}"
        );
        assert!(
            text.contains("database connection refused"),
            "error message lost in briefing: {text}"
        );
        assert!(
            text.contains("error.code = 500"),
            "error code lost in briefing: {text}"
        );
    }
}
