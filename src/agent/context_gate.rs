#![allow(dead_code)]
//! Context Gate: intelligent content management for LLM agents.
//!
//! Instead of uniform char-limit truncation, the gate makes context-aware
//! decisions based on the model's token budget:
//! - **Pass raw** when content fits
//! - **Briefing** (structural summary via compactor) when it doesn't,
//!   with full content cached to disk for drill-down via `read_file(lines=...)`

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use serde_json::Value;

use crate::agent::compaction::ContextCompactor;
use crate::agent::token_budget::TokenBudget;

// ---------------------------------------------------------------------------
// CacheRef
// ---------------------------------------------------------------------------

/// Stable reference to a cached tool output on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheRef {
    /// Unique identifier (monotonic counter + timestamp hash).
    pub id: String,
    /// Path to the cached file.
    pub path: PathBuf,
}

// ---------------------------------------------------------------------------
// OutputCache
// ---------------------------------------------------------------------------

static CACHE_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Persists full tool outputs to disk, serves them back by reference or line range.
pub struct OutputCache {
    cache_dir: PathBuf,
}

impl OutputCache {
    pub fn new(cache_dir: PathBuf) -> Self {
        let _ = std::fs::create_dir_all(&cache_dir);
        Self { cache_dir }
    }

    /// Store content, return a stable reference.
    pub fn store(&self, content: &str) -> CacheRef {
        let seq = CACHE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let id = format!("gate_{}_{}", ts, seq);
        let path = self.cache_dir.join(&id);
        let _ = std::fs::write(&path, content);
        CacheRef { id, path }
    }

    /// Retrieve full content by reference.
    pub fn get(&self, cache_ref: &CacheRef) -> Option<String> {
        std::fs::read_to_string(&cache_ref.path).ok()
    }

    /// Retrieve a line range (1-indexed, inclusive).
    pub fn get_lines(&self, cache_ref: &CacheRef, start: usize, end: usize) -> Option<String> {
        let content = self.get(cache_ref)?;
        let lines: Vec<&str> = content.lines().collect();
        let s = start.max(1) - 1; // convert to 0-indexed
        let e = end.min(lines.len());
        if s >= lines.len() || s >= e {
            return Some(String::new());
        }
        Some(lines[s..e].join("\n"))
    }

    /// Remove entries older than `max_age`.
    pub fn gc(&self, max_age: Duration) {
        let Ok(entries) = std::fs::read_dir(&self.cache_dir) else {
            return;
        };
        let now = std::time::SystemTime::now();
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                let age = meta
                    .modified()
                    .ok()
                    .and_then(|m| now.duration_since(m).ok())
                    .unwrap_or_default();
                if age > max_age {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GateResult
// ---------------------------------------------------------------------------

/// What the gate returns after processing content.
pub enum GateResult {
    /// Content fits — pass through unchanged.
    Raw(String),
    /// Content was too large — summarized, full version cached.
    Briefing {
        summary: String,
        cache_ref: CacheRef,
        original_size: usize,
    },
}

impl GateResult {
    /// Get the text to inject into the agent's context.
    pub fn text(&self) -> &str {
        match self {
            GateResult::Raw(s) => s,
            GateResult::Briefing { summary, .. } => summary,
        }
    }

    /// Consume into the text string.
    pub fn into_text(self) -> String {
        match self {
            GateResult::Raw(s) => s,
            GateResult::Briefing { summary, .. } => summary,
        }
    }
}

// ---------------------------------------------------------------------------
// ContentGate
// ---------------------------------------------------------------------------

/// Single entry point for ALL content entering the agent's context.
///
/// Decides: pass raw (fits) or produce a briefing (doesn't fit).
/// The briefing is a structural summary; full content is cached for drill-down.
pub struct ContentGate {
    pub budget: TokenBudget,
    pub cache: OutputCache,
}

impl ContentGate {
    pub fn new(max_tokens: usize, output_reserve: f32, cache_dir: PathBuf) -> Self {
        Self {
            budget: TokenBudget::with_output_reserve(max_tokens, output_reserve),
            cache: OutputCache::new(cache_dir),
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

        // Content doesn't fit — cache full version, produce briefing.
        let cache_ref = self.cache.store(content);
        let target_tokens = available / 2; // briefing should use ~half remaining budget
        let summary = briefing_fn(content, target_tokens);
        let summary_tokens = TokenBudget::estimate_str_tokens(&summary);
        self.budget.consume(summary_tokens);

        GateResult::Briefing {
            summary,
            cache_ref,
            original_size: content.len(),
        }
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

    /// Gate content using the LLM compactor for briefings.
    ///
    /// Like `admit`, but uses the compactor's `summarize_for_briefing` when
    /// content doesn't fit. Falls back to simple briefing if compactor fails.
    pub async fn admit_with_compactor(
        &mut self,
        content: &str,
        compactor: &ContextCompactor,
    ) -> GateResult {
        let tokens = TokenBudget::estimate_str_tokens(content);
        let available = self.budget.available();

        if tokens <= available {
            self.budget.consume(tokens);
            return GateResult::Raw(content.to_string());
        }

        let cache_ref = self.cache.store(content);
        let target_tokens = available / 2;

        // Try LLM briefing, fall back to mechanical briefing on failure.
        let summary = match compactor
            .summarize_for_briefing(content, target_tokens)
            .await
        {
            Ok(s) => s,
            Err(e) => {
                tracing::debug!("Briefing LLM failed, using simple briefing: {}", e);
                build_simple_briefing(content, target_tokens)
            }
        };

        let summary_tokens = TokenBudget::estimate_str_tokens(&summary);
        self.budget.consume(summary_tokens);

        GateResult::Briefing {
            summary,
            cache_ref,
            original_size: content.len(),
        }
    }

    /// Gate content using the specialist provider for semantically aware summarization.
    /// Falls back to `admit_simple()` on failure.
    pub async fn admit_with_specialist(
        &mut self,
        content: &str,
        provider: &dyn crate::providers::base::LLMProvider,
        model: &str,
    ) -> GateResult {
        let tokens = crate::agent::token_budget::TokenBudget::estimate_str_tokens(content);
        let available = self.budget.available();

        // If it fits in budget, pass through raw.
        if tokens <= available {
            self.budget.consume(tokens);
            return GateResult::Raw(content.to_string());
        }

        let cache_ref = self.cache.store(content);
        let target_tokens = available / 2;

        // JSON tool output is handled deterministically to avoid model drift.
        // This path preserves exact values and avoids hallucinated fields.
        if let Some(summary) = build_json_briefing(content, target_tokens) {
            let summary_tokens = crate::agent::token_budget::TokenBudget::estimate_str_tokens(&summary);
            self.budget.consume(summary_tokens);
            return GateResult::Briefing {
                summary,
                cache_ref,
                original_size: content.len(),
            };
        }

        // Try specialist summarization, fall back to mechanical briefing.
        let summary = match specialist_summarize(provider, model, content, target_tokens).await {
            Ok(s) => s,
            Err(e) => {
                tracing::debug!("Specialist summarization failed, using simple briefing: {}", e);
                build_simple_briefing(content, target_tokens)
            }
        };

        let summary_tokens = crate::agent::token_budget::TokenBudget::estimate_str_tokens(&summary);
        self.budget.consume(summary_tokens);

        GateResult::Briefing {
            summary,
            cache_ref,
            original_size: content.len(),
        }
    }

    /// Gate content using deterministic extraction only (no LLM calls).
    pub fn admit_with_deterministic(&mut self, content: &str) -> GateResult {
        let tokens = crate::agent::token_budget::TokenBudget::estimate_str_tokens(content);
        let available = self.budget.available();

        if tokens <= available {
            self.budget.consume(tokens);
            return GateResult::Raw(content.to_string());
        }

        let cache_ref = self.cache.store(content);
        let target_tokens = available / 2;
        let summary = if let Some(json_summary) = build_json_briefing(content, target_tokens) {
            json_summary
        } else {
            build_deterministic_tool_briefing(content, target_tokens)
        };
        let summary_tokens = crate::agent::token_budget::TokenBudget::estimate_str_tokens(&summary);
        self.budget.consume(summary_tokens);

        GateResult::Briefing {
            summary,
            cache_ref,
            original_size: content.len(),
        }
    }

    /// Hybrid gate: deterministic extraction first, specialist fallback only
    /// when deterministic signal is weak.
    pub async fn admit_with_hybrid(
        &mut self,
        content: &str,
        provider: &dyn crate::providers::base::LLMProvider,
        model: &str,
    ) -> GateResult {
        let tokens = crate::agent::token_budget::TokenBudget::estimate_str_tokens(content);
        let available = self.budget.available();

        if tokens <= available {
            self.budget.consume(tokens);
            return GateResult::Raw(content.to_string());
        }

        let cache_ref = self.cache.store(content);
        let target_tokens = available / 2;

        if let Some(summary) = build_json_briefing(content, target_tokens) {
            let summary_tokens = crate::agent::token_budget::TokenBudget::estimate_str_tokens(&summary);
            self.budget.consume(summary_tokens);
            return GateResult::Briefing {
                summary,
                cache_ref,
                original_size: content.len(),
            };
        }

        let (det_summary, det_signal) = build_deterministic_tool_briefing_with_score(content, target_tokens);
        if should_use_deterministic_in_hybrid(&det_summary, content, det_signal) {
            let summary_tokens = crate::agent::token_budget::TokenBudget::estimate_str_tokens(&det_summary);
            self.budget.consume(summary_tokens);
            return GateResult::Briefing {
                summary: det_summary,
                cache_ref,
                original_size: content.len(),
            };
        }

        let summary = match specialist_summarize(provider, model, content, target_tokens).await {
            Ok(s) => s,
            Err(_) => det_summary,
        };
        let summary_tokens = crate::agent::token_budget::TokenBudget::estimate_str_tokens(&summary);
        self.budget.consume(summary_tokens);

        GateResult::Briefing {
            summary,
            cache_ref,
            original_size: content.len(),
        }
    }

    /// Run garbage collection on the cache.
    pub fn gc(&self, max_age: Duration) {
        self.cache.gc(max_age);
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

fn anomaly_line_for_object(path: &str, map: &serde_json::Map<String, Value>) -> Option<String> {
    let status = map.get("status").and_then(|v| v.as_str());
    let error = map.get("error").and_then(|v| v.as_str());
    let latency = map.get("latencyMs").and_then(|v| v.as_i64());

    let mut notes = Vec::new();
    if let Some(s) = status {
        if !is_ok_status(s) {
            notes.push(format!("status=\"{}\"", s));
        }
    }
    if let Some(e) = error {
        if !e.trim().is_empty() {
            notes.push(format!("error=\"{}\"", e.chars().take(120).collect::<String>()));
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

fn build_deterministic_tool_briefing(content: &str, target_tokens: usize) -> String {
    build_deterministic_tool_briefing_with_score(content, target_tokens).0
}

fn build_deterministic_tool_briefing_with_score(content: &str, target_tokens: usize) -> (String, usize) {
    let lines: Vec<&str> = content.lines().collect();
    let lower = content.to_ascii_lowercase();

    let mut key_facts: Vec<String> = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    let mut next_checks: Vec<String> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut key_fact_shapes: BTreeSet<String> = BTreeSet::new();

    let mut push_unique = |bucket: &mut Vec<String>, value: String| {
        if value.trim().is_empty() {
            return;
        }
        if seen.insert(value.clone()) {
            bucket.push(value);
        }
    };

    let signal_indexes: BTreeSet<usize> = lines
        .iter()
        .enumerate()
        .filter_map(|(idx, line)| {
            if is_deterministic_signal_line(line) {
                Some(idx)
            } else {
                None
            }
        })
        .collect();

    for (idx, line) in lines.iter().enumerate() {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }

        let has_signal = signal_indexes.contains(&idx);
        let neighbor_signal = (idx > 0 && signal_indexes.contains(&(idx - 1)))
            || signal_indexes.contains(&(idx + 1));

        // Keep exact neighboring context when it carries concrete literals
        // (file paths, ids, endpoints) tied to a signal line.
        if !has_signal && !(neighbor_signal && has_literal_anchor(l)) {
            continue;
        }

        let clipped: String = l.chars().take(220).collect();
        if is_failure_line(l) {
            push_unique(&mut failures, clipped);
        } else {
            let shape = normalize_line_shape(l);
            if !key_fact_shapes.insert(shape) {
                continue;
            }
            push_unique(&mut key_facts, clipped);
        }
    }

    if failures.is_empty() {
        if lower.contains("test") && lower.contains("shard") {
            push_unique(
                &mut next_checks,
                "Continue test shard run and capture final test result line".to_string(),
            );
        }
        if lower.contains("running") && !lower.contains("failed") {
            push_unique(
                &mut next_checks,
                "No explicit failure line detected in captured segment; fetch later lines".to_string(),
            );
        }
    } else {
        push_unique(
            &mut next_checks,
            "Inspect referenced files and paths from failure lines to identify root cause".to_string(),
        );
        push_unique(
            &mut next_checks,
            "Re-run only failing command or test with full output capture".to_string(),
        );
    }

    let line_budget = (target_tokens / 18).clamp(6, 24);
    key_facts.truncate(line_budget);
    failures.truncate(line_budget);
    next_checks.truncate(6);
    let signal = (failures.len().min(6) + key_facts.len().min(6)).min(12);

    let mut out = String::new();
    out.push_str("Key Facts\n");
    if key_facts.is_empty() {
        out.push_str("- No high-signal facts extracted from this segment.\n");
    } else {
        for item in key_facts {
            out.push_str("- ");
            out.push_str(&item);
            out.push('\n');
        }
    }
    out.push_str("\nFailures\n");
    if failures.is_empty() {
        out.push_str("- No explicit failure line in captured segment.\n");
    } else {
        for item in failures {
            out.push_str("- ");
            out.push_str(&item);
            out.push('\n');
        }
    }
    out.push_str("\nNext Checks\n");
    if next_checks.is_empty() {
        out.push_str("- Drill into cached raw output for exact failing section.\n");
    } else {
        for item in next_checks {
            out.push_str("- ");
            out.push_str(&item);
            out.push('\n');
        }
    }

    (out, signal)
}

fn is_deterministic_signal_line(line: &str) -> bool {
    let ll = line.trim().to_ascii_lowercase();
    ll.contains("error")
        || ll.contains("failed")
        || ll.contains("panic")
        || ll.contains("aborting")
        || ll.contains("final_status=")
        || ll.contains("requestid")
        || ll.contains("latency")
        || ll.contains("assertion failed")
        || ll.contains("http://")
        || ll.contains("https://")
        || ll.contains("test result:")
        || ll.contains("[stage:")
        || ll.contains("fail:")
        || ll.contains("skipped")
}

fn has_literal_anchor(line: &str) -> bool {
    let l = line.trim();
    let ll = l.to_ascii_lowercase();
    ll.contains("src/")
        || ll.contains("tests/")
        || ll.contains("http://")
        || ll.contains("https://")
        || ll.contains("final_status=")
        || ll.contains("requestid")
        || ll.contains("invoice")
        || ll.contains("error[")
        || (ll.contains("::") && l.chars().any(|c| c.is_ascii_digit()))
        || (ll.contains('/') && l.chars().any(|c| c.is_ascii_digit()))
}

fn is_failure_line(line: &str) -> bool {
    let ll = line.trim().to_ascii_lowercase();
    ll.contains("error")
        || ll.contains("failed")
        || ll.contains("panic")
        || ll.contains("assertion failed")
        || ll.contains("aborting")
        || ll.contains("fail:")
        || ll.contains("final_status=failed")
}

fn normalize_line_shape(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    for ch in line.trim().to_ascii_lowercase().chars() {
        if ch.is_ascii_digit() {
            out.push('#');
        } else if ch.is_ascii_whitespace() {
            if !out.ends_with(' ') {
                out.push(' ');
            }
        } else {
            out.push(ch);
        }
    }
    out.trim().to_string()
}

fn should_use_deterministic_in_hybrid(summary: &str, source: &str, det_signal: usize) -> bool {
    if det_signal < 3 {
        return false;
    }

    let summary_lower = summary.to_ascii_lowercase();
    let source_lower = source.to_ascii_lowercase();

    if summary_lower.contains("\n- .") {
        return false;
    }

    let has_anchor = summary_lower.contains("src/")
        || summary_lower.contains("tests/")
        || summary_lower.contains("http://")
        || summary_lower.contains("https://")
        || summary_lower.contains("error[")
        || summary_lower.contains("assertion failed")
        || summary_lower.contains("final_status=")
        || summary_lower.contains("requestid")
        || summary_lower.contains("/v1/");
    if !has_anchor {
        return false;
    }

    let source_has_failure_markers = source_lower.contains("failed")
        || source_lower.contains("error")
        || source_lower.contains("panic")
        || source_lower.contains("final_status=")
        || source_lower.contains("test result:");
    let summary_claims_no_failure = summary_lower.contains("no explicit failure line");
    if source_has_failure_markers && summary_claims_no_failure {
        return false;
    }

    let summary_has_failure = summary_lower.contains("\nfailures\n-") && !summary_claims_no_failure;
    if summary_has_failure {
        return det_signal >= 2;
    }

    det_signal >= 5
}

fn normalize_specialist_summary(text: &str, target_tokens: usize) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }

    let lower = trimmed.to_ascii_lowercase();
    let first_header = ["key facts", "failures", "next checks"]
        .iter()
        .filter_map(|h| lower.find(h))
        .min();

    let mut out = if let Some(idx) = first_header {
        trimmed[idx..].to_string()
    } else {
        trimmed.to_string()
    };

    let meta_markers = [
        "we are given",
        "let's",
        "steps:",
        "analysis:",
        "the task is to",
        "we must output",
    ];
    let has_meta = meta_markers
        .iter()
        .any(|m| out.to_ascii_lowercase().contains(m));
    if has_meta && first_header.is_none() {
        return None;
    }

    if !out.to_ascii_lowercase().contains("key facts") {
        return None;
    }

    let max_chars = target_tokens.saturating_mul(7).max(280);
    out = out.chars().take(max_chars).collect();
    Some(out)
}

/// Ask the specialist provider to summarize a tool result.
async fn specialist_summarize(
    provider: &dyn crate::providers::base::LLMProvider,
    model: &str,
    content: &str,
    target_tokens: usize,
) -> Result<String, String> {
    use serde_json::json;
    let caps = crate::agent::model_capabilities::lookup(model, &std::collections::HashMap::new());
    let thinking_budget = if caps.thinking {
        // Enable hidden reasoning only for thinking-capable models.
        // Budget scales with requested summary size but stays bounded.
        Some((target_tokens as u32).saturating_mul(2).clamp(160, 768))
    } else {
        None
    };

    let messages = vec![
        json!({
            "role": "system",
            "content": format!(
                "You are a strict incident summarizer for tool output.\n\
                 Return ONLY final answer (no preamble) in <= {} tokens.\n\
                 Rules:\n\
                 1) Copy exact literals: numbers, IDs, paths, error strings.\n\
                 2) If exact copy is not possible, omit that fact.\n\
                 3) No meta text (no 'we are given', no 'let's', no planning).\n\
                 4) No JSON output unless input is JSON.\n\
                 5) Output exactly these sections:\n\
                    - Key Facts\n\
                    - Failures\n\
                    - Next Checks",
                target_tokens
            )
        }),
        json!({
            "role": "user",
            "content": content
        }),
    ];
    // Keep output budget tight to avoid runaway local generation latency.
    // Previously this allowed 2x target tokens, then clamped locally, wasting
    // generation time and hurting p95 latency on small local models.
    let max_response_tokens = (target_tokens as u32).saturating_add(48).clamp(96, 320);
    let resp = provider
        .chat(
            &messages,
            None,
            Some(model),
            max_response_tokens,
            0.2,
            thinking_budget,
            None,
        )
        .await
        .map_err(|e| format!("specialist chat failed: {}", e))?;
    let raw = resp
        .content
        .ok_or_else(|| "specialist returned no content".to_string())?;
    let cleaned = crate::agent::compaction::strip_thinking_tags(&raw);
    let max_chars = target_tokens.saturating_mul(6).max(240);
    let clamped: String = cleaned.chars().take(max_chars).collect();
    normalize_specialist_summary(&clamped, target_tokens)
        .ok_or_else(|| "specialist returned non-final/meta output".to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::path::Path;

    use crate::providers::base::{LLMProvider, LLMResponse};

    struct PanicProvider;

    struct MarkerProvider;

    #[async_trait]
    impl LLMProvider for PanicProvider {
        async fn chat(
            &self,
            _messages: &[serde_json::Value],
            _tools: Option<&[serde_json::Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
        ) -> anyhow::Result<LLMResponse> {
            panic!("provider should not be called for JSON deterministic briefing")
        }

        fn get_default_model(&self) -> &str {
            "panic"
        }
    }

    #[async_trait]
    impl LLMProvider for MarkerProvider {
        async fn chat(
            &self,
            _messages: &[serde_json::Value],
            _tools: Option<&[serde_json::Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
        ) -> anyhow::Result<LLMResponse> {
            Ok(LLMResponse {
                content: Some("Key Facts\n- PROVIDER_MARKER\n\nFailures\n- none\n\nNext Checks\n- none".to_string()),
                tool_calls: Vec::new(),
                finish_reason: "stop".to_string(),
                usage: std::collections::HashMap::new(),
            })
        }

        fn get_default_model(&self) -> &str {
            "marker"
        }
    }

    fn test_cache_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("nanobot_gate_test_{}_{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&dir); // clean slate
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    fn cleanup(dir: &Path) {
        let _ = std::fs::remove_dir_all(dir);
    }

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

    // -- OutputCache tests --

    #[test]
    fn test_cache_store_and_get() {
        let dir = test_cache_dir("store_get");
        let cache = OutputCache::new(dir.clone());

        let content = "line 1\nline 2\nline 3";
        let cache_ref = cache.store(content);
        assert!(cache_ref.path.exists());

        let retrieved = cache.get(&cache_ref).unwrap();
        assert_eq!(retrieved, content);

        cleanup(&dir);
    }

    #[test]
    fn test_cache_get_lines() {
        let dir = test_cache_dir("get_lines");
        let cache = OutputCache::new(dir.clone());

        let content = "alpha\nbeta\ngamma\ndelta\nepsilon";
        let cache_ref = cache.store(content);

        // Lines 2-4 (1-indexed, inclusive)
        let range = cache.get_lines(&cache_ref, 2, 4).unwrap();
        assert_eq!(range, "beta\ngamma\ndelta");

        // Line 1 only
        let first = cache.get_lines(&cache_ref, 1, 1).unwrap();
        assert_eq!(first, "alpha");

        // Out of bounds clamped
        let all = cache.get_lines(&cache_ref, 1, 100).unwrap();
        assert_eq!(all, content);

        // Empty range
        let empty = cache.get_lines(&cache_ref, 10, 20).unwrap();
        assert_eq!(empty, "");

        cleanup(&dir);
    }

    #[test]
    fn test_cache_gc() {
        let dir = test_cache_dir("gc");
        let cache = OutputCache::new(dir.clone());

        let ref1 = cache.store("old content");
        assert!(ref1.path.exists());

        // GC with zero max_age should remove everything.
        cache.gc(Duration::from_secs(0));
        assert!(!ref1.path.exists());

        cleanup(&dir);
    }

    // -- ContentGate tests --

    #[test]
    fn test_gate_raw_when_fits() {
        let dir = test_cache_dir("raw_fits");
        let mut gate = ContentGate::new(100_000, 0.20, dir.clone());

        let content = "short content";
        let result = gate.admit_simple(content);
        match result {
            GateResult::Raw(s) => assert_eq!(s, content),
            GateResult::Briefing { .. } => panic!("expected Raw"),
        }

        cleanup(&dir);
    }

    #[test]
    fn test_gate_briefing_when_too_large() {
        let dir = test_cache_dir("briefing_large");
        // Tiny budget: 100 tokens, 20% reserve = 80 available.
        let mut gate = ContentGate::new(100, 0.20, dir.clone());

        // Content that definitely exceeds 80 tokens.
        let content = "x\n".repeat(500);
        let result = gate.admit_simple(&content);
        match result {
            GateResult::Raw(_) => panic!("expected Briefing"),
            GateResult::Briefing {
                summary,
                cache_ref,
                original_size,
            } => {
                assert!(summary.contains("Content Summary"));
                assert!(summary.contains("read_file"));
                assert_eq!(original_size, content.len());
                // Full content is cached
                let cached = gate.cache.get(&cache_ref).unwrap();
                assert_eq!(cached, content);
            }
        }

        cleanup(&dir);
    }

    #[test]
    fn test_gate_custom_briefing_fn() {
        let dir = test_cache_dir("custom_fn");
        let mut gate = ContentGate::new(50, 0.20, dir.clone());

        let content = "a\n".repeat(200);
        let result = gate.admit(&content, |_c, _target| "custom briefing".to_string());
        match result {
            GateResult::Briefing { summary, .. } => {
                assert_eq!(summary, "custom briefing");
            }
            GateResult::Raw(_) => panic!("expected Briefing"),
        }

        cleanup(&dir);
    }

    #[test]
    fn test_gate_result_text() {
        let raw = GateResult::Raw("hello".to_string());
        assert_eq!(raw.text(), "hello");

        let briefing = GateResult::Briefing {
            summary: "summary".to_string(),
            cache_ref: CacheRef {
                id: "test".to_string(),
                path: PathBuf::from("/tmp/test"),
            },
            original_size: 1000,
        };
        assert_eq!(briefing.text(), "summary");
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
        let dir = test_cache_dir("multi_admits");
        let mut gate = ContentGate::new(10_000, 0.20, dir.clone());
        // available = 8_000

        let initial = gate.budget.available();
        let content = "hello world";
        let _ = gate.admit_simple(content);
        let after = gate.budget.available();

        assert!(after < initial, "budget should decrease after admit");

        cleanup(&dir);
    }

    #[tokio::test]
    async fn test_admit_with_specialist_uses_deterministic_json_path() {
        let dir = test_cache_dir("specialist_json_bypass");
        let mut gate = ContentGate::new(40, 0.20, dir.clone());
        let content = r#"{
  "requestId": "req-abc",
  "entries": [
    {"status": "ok", "latencyMs": 10, "error": ""},
    {"status": "error", "latencyMs": 5000, "error": "bad checksum", "path": "/v1/jobs/reconcile"}
  ]
}"#;

        let result = gate
            .admit_with_specialist(content, &PanicProvider, "any-model")
            .await;

        match result {
            GateResult::Raw(_) => panic!("expected briefing for oversized content"),
            GateResult::Briefing { summary, .. } => {
                assert!(summary.contains("JSON Summary"));
                assert!(summary.contains("bad checksum"));
                assert!(summary.contains("/v1/jobs/reconcile"));
            }
        }

        cleanup(&dir);
    }

    #[test]
    fn test_deterministic_briefing_extracts_failures() {
        let content = "running 10 tests\nassertion failed: expected foo\nerror[E0502]: cannot borrow\nfinal_status=failed\n";
        let briefing = build_deterministic_tool_briefing(content, 300);
        assert!(briefing.contains("Key Facts"));
        assert!(briefing.contains("Failures"));
        assert!(briefing.contains("assertion failed"));
        assert!(briefing.contains("error[E0502]"));
        assert!(!briefing.contains("Input size:"));
    }

    #[test]
    fn test_deterministic_briefing_keeps_adjacent_literal_context() {
        let content = "error[E0502]: cannot borrow `state`\n  --> src/agent/router.rs:412:21\n";
        let briefing = build_deterministic_tool_briefing(content, 280);
        assert!(briefing.contains("src/agent/router.rs:412:21"));
    }

    #[test]
    fn test_deterministic_briefing_dedupes_repetitive_retry_lines() {
        let content = "retry 1: waiting for local endpoint http://127.0.0.1:1234/v1\nretry 2: waiting for local endpoint http://127.0.0.1:1234/v1\nretry 3: waiting for local endpoint http://127.0.0.1:1234/v1\n";
        let briefing = build_deterministic_tool_briefing(content, 260);
        let kept = briefing
            .lines()
            .filter(|l| l.contains("waiting for local endpoint"))
            .count();
        assert_eq!(kept, 1);
    }

    #[tokio::test]
    async fn test_admit_with_hybrid_uses_deterministic_for_strong_signal() {
        let dir = test_cache_dir("hybrid_strong_signal");
        let mut gate = ContentGate::new(20, 0.20, dir.clone());
        let content = "running 134 tests\nassertion failed: expected assistant tool_calls while calling http://127.0.0.1:1234/v1\nerror[E0597]: payload does not live long enough\n";
        let result = gate
            .admit_with_hybrid(content, &MarkerProvider, "qwen/qwen3-4b-thinking-2507")
            .await;

        match result {
            GateResult::Raw(_) => panic!("expected briefing for oversized content"),
            GateResult::Briefing { summary, .. } => {
                assert!(summary.contains("Failures"));
            }
        }
        cleanup(&dir);
    }

    #[test]
    fn test_hybrid_rejects_broken_deterministic_fragment() {
        let summary = "Key Facts\n- retry 1: waiting for local endpoint http://127.0\n- .0.1:1234/v1\n";
        let source = "retry 1: waiting for local endpoint http://127.0.0.1:1234/v1\n";
        assert!(!should_use_deterministic_in_hybrid(summary, source, 6));
    }
}
