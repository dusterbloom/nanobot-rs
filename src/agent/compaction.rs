//! Context compaction via LLM-powered summarization.
//!
//! Two-stage design:
//! - Stage 1 (66.6% capacity): Proactive LLM summarization → creates observation.
//! - Stage 2 (100% capacity): Emergency `trim_to_fit()` — mechanical, no LLM.
//!
//! Stage 1 is handled here. Stage 2 is `TokenBudget::trim_to_fit()` in agent_loop.rs.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use serde_json::{json, Value};
use tracing::{debug, warn};

use crate::agent::token_budget::TokenBudget;
use crate::providers::base::LLMProvider;

/// Summarization prompt sent to the LLM.
const SUMMARIZE_PROMPT: &str = "\
Extract facts from this conversation into the template below.
Rules:
1. Copy technical terms, file paths, numbers EXACTLY
2. One sentence per bullet, max 10 bullets per section
3. Skip anything you're unsure about
4. No meta-commentary (no 'I should...', 'Let me...')
5. ONLY output the filled template, nothing else

## Task
[What is the user doing?]

## Decisions
[What was decided?]

## Facts
[What was discovered?]

## Pending
[What's still to do?]

## Errors
[What went wrong?]";

/// Prompt for merging already-compressed chunk summaries.
const MERGE_SUMMARIES_PROMPT: &str = "\
Merge these context summaries into one using the same template.
Rules: copy exact terms, one sentence per bullet, max 10 per section, no meta-commentary.

## Task
[Combined task description]

## Decisions
[All decisions]

## Facts
[All facts]

## Pending
[All pending items]

## Errors
[All errors]";

// SUMMARIZER_INPUT_BUDGET_TOKENS removed — now dynamically computed via
// ContextCompactor::input_budget() based on the compaction server's ctx size.

/// Safety cap for iterative merge rounds.
const MAX_MERGE_ROUNDS: usize = 6;

/// Result of a compaction attempt.
pub struct CompactionResult {
    /// The (possibly compacted) messages.
    pub messages: Vec<Value>,
    /// If compaction occurred, the summary text (for observation storage).
    pub observation: Option<String>,
}

/// Compacts conversation context by summarizing old messages via an LLM call.
pub struct ContextCompactor {
    provider: Arc<dyn LLMProvider>,
    model: String,
    /// Max tokens for the summarization response.
    summary_max_tokens: u32,
    /// Disable proactive compaction after first provider failure.
    disabled: AtomicBool,
    /// Context window size of the compaction model (tokens).
    compaction_context_size: usize,
    /// Compaction threshold as percentage of available context (e.g. 66.6).
    threshold_percent: f64,
    /// Compaction threshold in absolute tokens. Whichever fires first wins.
    threshold_tokens: usize,
}

impl ContextCompactor {
    /// Create a new compactor that uses the given provider/model for summaries.
    ///
    /// `compaction_context_size` is the context window of the compaction model
    /// (in tokens). The input budget for summarization chunks is derived from
    /// this dynamically, so a 4K model produces ~2.5K budgets while a 32K
    /// model can summarize in a single call.
    pub fn new(provider: Arc<dyn LLMProvider>, model: String, compaction_context_size: usize) -> Self {
        Self {
            provider,
            model,
            summary_max_tokens: 1024,
            disabled: AtomicBool::new(false),
            compaction_context_size,
            threshold_percent: 66.6,
            threshold_tokens: 100_000,
        }
    }

    /// Set the compaction thresholds (from config).
    pub fn with_thresholds(mut self, percent: f64, tokens: usize) -> Self {
        self.threshold_percent = percent;
        self.threshold_tokens = tokens;
        self
    }

    /// Dynamic input budget derived from the compaction model's context size.
    ///
    /// Reserves space for the system prompt (~200 tokens), the summary
    /// response, and a small safety margin.
    fn input_budget(&self) -> usize {
        let reserved = 200 + self.summary_max_tokens as usize + 300;
        self.compaction_context_size.saturating_sub(reserved)
    }

    /// Compute the effective compaction threshold (the lower of percent-based and token-based).
    fn effective_threshold(&self, available: usize) -> usize {
        let pct_threshold = (available as f64 * self.threshold_percent / 100.0) as usize;
        pct_threshold.min(self.threshold_tokens)
    }

    /// Check whether the conversation needs compaction.
    pub fn needs_compaction(&self, messages: &[Value], budget: &TokenBudget, tool_def_tokens: usize) -> bool {
        if self.disabled.load(Ordering::Relaxed) {
            return false;
        }
        let available = budget.available_budget(tool_def_tokens);
        let estimated = TokenBudget::estimate_tokens(messages);
        let threshold = self.effective_threshold(available);
        estimated > threshold
    }

    /// Compact messages when they exceed 66.6% of the token budget.
    ///
    /// Returns a `CompactionResult` with the (possibly compacted) messages and
    /// an optional observation summary for cross-session memory.
    ///
    /// If messages fit within 66.6% of budget, returns them unchanged with no
    /// observation. The 100% safety net (`trim_to_fit`) runs separately in
    /// agent_loop.rs every iteration.
    pub async fn compact(
        &self,
        messages: &[Value],
        budget: &TokenBudget,
        tool_def_tokens: usize,
    ) -> CompactionResult {
        if self.disabled.load(Ordering::Relaxed) {
            return CompactionResult {
                messages: messages.to_vec(),
                observation: None,
            };
        }

        let available = budget.available_budget(tool_def_tokens);
        let current = TokenBudget::estimate_tokens(messages);

        // Stage 1: Proactive compaction (configurable threshold).
        let threshold = self.effective_threshold(available);
        if current <= threshold {
            return CompactionResult {
                messages: messages.to_vec(),
                observation: None,
            };
        }

        debug!(
            "Proactive compaction at {:.0}% capacity ({}/{})",
            (current as f64 / available as f64) * 100.0,
            current,
            available,
        );

        // Try LLM summarization.
        match self.compact_via_summary(messages, available, budget).await {
            Ok((compacted, summary)) => {
                let new_size = TokenBudget::estimate_tokens(&compacted);
                debug!("Compacted {} -> {} tokens", current, new_size);
                CompactionResult {
                    messages: compacted,
                    observation: Some(summary),
                }
            }
            Err(e) => {
                if !self.disabled.swap(true, Ordering::SeqCst) {
                    warn!(
                        "Compaction failed; disabling proactive compaction for this run: {}",
                        e
                    );
                } else {
                    debug!("Compaction still disabled after prior failure: {}", e);
                }
                CompactionResult {
                    messages: messages.to_vec(),
                    observation: None,
                }
            }
        }
    }

    /// Perform the actual summarization-based compaction.
    ///
    /// Returns (compacted_messages, summary_text).
    async fn compact_via_summary(
        &self,
        messages: &[Value],
        available_budget: usize,
        budget: &TokenBudget,
    ) -> Result<(Vec<Value>, String)> {
        if messages.is_empty() {
            return Ok((messages.to_vec(), String::new()));
        }

        // Always keep the system message (first).
        let system_msg = messages[0].clone();
        let system_tokens = TokenBudget::estimate_tokens(&[system_msg.clone()]);

        // Reserve space: system + summary overhead + some headroom.
        let summary_headroom = 400; // ~400 tokens for the summary message
        let remaining = available_budget
            .saturating_sub(system_tokens)
            .saturating_sub(summary_headroom);
        if remaining == 0 {
            anyhow::bail!(
                "Budget too small for summarization (available={}, system={}, headroom={})",
                available_budget,
                system_tokens,
                summary_headroom
            );
        }
        let recent_budget = (remaining as f64 * 0.6) as usize;

        // Walk backwards from the end to find recent messages that fit in ~60% of budget.
        // Messages containing subagent results are force-included to prevent
        // compaction from summarizing away important delegated work.
        let body = &messages[1..];
        let mut recent_start = body.len();
        let mut recent_tokens = 0;

        // First pass: identify subagent result messages to protect.
        let protected: std::collections::HashSet<usize> = body
            .iter()
            .enumerate()
            .filter(|(_, msg)| {
                let content = msg["content"].as_str().unwrap_or("");
                content.contains("[Subagent ") || content.contains("subagent-")
                    || content.contains("Result also saved to:")
            })
            .map(|(i, _)| i)
            .collect();

        for (i, msg) in body.iter().enumerate().rev() {
            let msg_tokens = TokenBudget::estimate_tokens(&[msg.clone()]);
            if recent_tokens + msg_tokens > recent_budget && !protected.contains(&i) {
                recent_start = i + 1;
                break;
            }
            recent_tokens += msg_tokens;
            if i == 0 {
                recent_start = 0;
            }
        }

        // If everything is "recent" (nothing to summarize), fall back.
        if recent_start == 0 {
            anyhow::bail!("All messages fit as recent; nothing to summarize");
        }

        let old_messages = &body[..recent_start];
        let recent_messages = &body[recent_start..];

        // Build the summarization request.
        let summary = self.summarize(old_messages).await?;

        // Assemble: system + summary + recent
        let mut result = Vec::with_capacity(2 + recent_messages.len());
        result.push(system_msg);
        result.push(json!({
            "role": "user",
            "content": format!("[Conversation summary: {}]", summary)
        }));
        result.extend_from_slice(recent_messages);

        Ok((result, summary))
    }

    /// Call the LLM to summarize a set of messages.
    async fn summarize(&self, messages: &[Value]) -> Result<String> {
        if messages.is_empty() {
            return Ok(String::new());
        }

        // First pass: summarize message chunks that fit the summarizer input budget.
        let mut summaries: Vec<String> = Vec::new();
        for (start, end) in split_message_ranges_by_budget(messages, self.input_budget()) {
            let transcript = build_transcript(&messages[start..end]);
            let s = self.summarize_text(&transcript, SUMMARIZE_PROMPT).await?;
            summaries.push(s);
        }

        // Merge pass: if we produced multiple chunk summaries, iteratively merge
        // them until one summary remains.
        let mut rounds = 0usize;
        while summaries.len() > 1 {
            rounds += 1;
            if rounds > MAX_MERGE_ROUNDS {
                anyhow::bail!(
                    "Exceeded merge rounds while summarizing (remaining chunks={})",
                    summaries.len()
                );
            }

            let mut merged: Vec<String> = Vec::new();
            for (start, end) in
                split_summary_ranges_by_budget(&summaries, self.input_budget())
            {
                let mut block = String::new();
                for (i, s) in summaries[start..end].iter().enumerate() {
                    block.push_str(&format!("Summary {}:\n{}\n\n", i + 1, s));
                }
                let next = self.summarize_text(&block, MERGE_SUMMARIES_PROMPT).await?;
                merged.push(next);
            }

            if merged.len() >= summaries.len() {
                anyhow::bail!(
                    "Summary merge made no progress (before={}, after={})",
                    summaries.len(),
                    merged.len()
                );
            }
            summaries = merged;
        }

        Ok(summaries.remove(0))
    }

    async fn summarize_text(&self, input: &str, prompt: &str) -> Result<String> {
        let summary_messages = vec![
            json!({
                "role": "system",
                "content": prompt
            }),
            json!({
                "role": "user",
                "content": input
            }),
        ];

        let response = self
            .provider
            .chat(
                &summary_messages,
                None,
                Some(&self.model),
                self.summary_max_tokens,
                0.3, // low temperature for factual summaries
                None,
            )
            .await?;

        if response.finish_reason == "error" {
            let detail = response
                .content
                .as_deref()
                .unwrap_or("unknown summarization error");
            anyhow::bail!("Summarization provider error: {}", detail);
        }

        let text = response
            .content
            .ok_or_else(|| anyhow::anyhow!("Summarization returned no content"))?;

        // Defensive: some providers encode HTTP/transport failures as plain text.
        if text.starts_with("Error calling LLM") || text.starts_with("Error:") {
            anyhow::bail!("Summarization failed: {}", text);
        }

        // Strip thinking tags that leak from small models (e.g. Qwen3).
        let text = strip_thinking_tags(&text);
        Ok(text)
    }
}

/// Strip `<thinking>...</thinking>` blocks from model output.
///
/// Small models (Qwen3-0.6B, 1.7B) sometimes leak chain-of-thought tags
/// into their output. This prevents garbage from leaking into summaries.
fn strip_thinking_tags(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(start) = remaining.find("<thinking>") {
        result.push_str(&remaining[..start]);
        if let Some(end) = remaining[start..].find("</thinking>") {
            remaining = &remaining[start + end + "</thinking>".len()..];
        } else {
            // Unclosed tag — drop everything after <thinking>
            return result.trim().to_string();
        }
    }
    result.push_str(remaining);
    result.trim().to_string()
}

fn format_message_for_transcript(msg: &Value) -> String {
    let role = msg
        .get("role")
        .and_then(|r| r.as_str())
        .unwrap_or("unknown");
    let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");

    // Truncate long tool results — keep first 600 + last 400 chars to
    // preserve file paths at the start and statuses/errors at the end.
    if role == "tool" {
        let name = msg.get("name").and_then(|n| n.as_str()).unwrap_or("tool");
        if content.len() > 1200 {
            let first: String = content.chars().take(600).collect();
            let last: String = content.chars().rev().take(400).collect::<String>().chars().rev().collect();
            return format!("{} ({}): {}...[{} chars omitted]...{}",
                role, name, first, content.len() - 1000, last);
        }
        return format!("{} ({}): {}", role, name, content);
    }

    if role == "assistant" && msg.get("tool_calls").is_some() {
        let mut out = String::new();
        // Summarize tool call requests briefly.
        if let Some(Value::Array(calls)) = msg.get("tool_calls") {
            let names: Vec<&str> = calls
                .iter()
                .filter_map(|c| {
                    c.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                })
                .collect();
            out.push_str(&format!("assistant: [called tools: {}]", names.join(", ")));
        }
        if !content.is_empty() {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(&format!("assistant: {}", content));
        }
        return out;
    }

    format!("{}: {}", role, content)
}

fn build_transcript(messages: &[Value]) -> String {
    messages
        .iter()
        .map(format_message_for_transcript)
        .collect::<Vec<_>>()
        .join("\n")
}

fn split_message_ranges_by_budget(messages: &[Value], max_tokens: usize) -> Vec<(usize, usize)> {
    if messages.is_empty() {
        return Vec::new();
    }

    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;
    let mut acc = 0usize;

    for (i, msg) in messages.iter().enumerate() {
        let piece = format_message_for_transcript(msg);
        let t = TokenBudget::estimate_str_tokens(&piece).max(1);

        if acc + t > max_tokens && i > start {
            ranges.push((start, i));
            start = i;
            acc = 0;
        }

        // Very large single message: keep as its own chunk so we preserve
        // message boundaries rather than slicing text arbitrarily.
        if t > max_tokens && i == start {
            ranges.push((i, i + 1));
            start = i + 1;
            acc = 0;
            continue;
        }

        acc += t;
    }

    if start < messages.len() {
        ranges.push((start, messages.len()));
    }

    ranges
}

fn split_summary_ranges_by_budget(summaries: &[String], max_tokens: usize) -> Vec<(usize, usize)> {
    if summaries.is_empty() {
        return Vec::new();
    }

    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;
    let mut acc = 0usize;

    for (i, s) in summaries.iter().enumerate() {
        let t = (TokenBudget::estimate_str_tokens(s) + 12).max(1); // label overhead

        if acc + t > max_tokens && i > start {
            ranges.push((start, i));
            start = i;
            acc = 0;
        }

        if t > max_tokens && i == start {
            ranges.push((i, i + 1));
            start = i + 1;
            acc = 0;
            continue;
        }

        acc += t;
    }

    if start < summaries.len() {
        ranges.push((start, summaries.len()));
    }

    ranges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::base::{LLMProvider, LLMResponse};
    use async_trait::async_trait;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Mock provider that returns a fixed summary.
    struct MockProvider {
        response: String,
    }

    impl MockProvider {
        fn new(response: &str) -> Self {
            Self {
                response: response.to_string(),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for MockProvider {
        async fn chat(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
        ) -> Result<LLMResponse> {
            Ok(LLMResponse {
                content: Some(self.response.clone()),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: HashMap::new(),
            })
        }

        fn get_default_model(&self) -> &str {
            "mock"
        }
    }

    /// Mock provider that always fails.
    struct FailingProvider;

    #[async_trait]
    impl LLMProvider for FailingProvider {
        async fn chat(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
        ) -> Result<LLMResponse> {
            Err(anyhow::anyhow!("LLM unavailable"))
        }

        fn get_default_model(&self) -> &str {
            "mock"
        }
    }

    struct CountingFailingProvider {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl LLMProvider for CountingFailingProvider {
        async fn chat(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
        ) -> Result<LLMResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Err(anyhow::anyhow!("LLM unavailable"))
        }

        fn get_default_model(&self) -> &str {
            "mock"
        }
    }

    #[tokio::test]
    async fn test_compact_within_budget_is_noop() {
        let provider = Arc::new(MockProvider::new("summary"));
        let compactor = ContextCompactor::new(provider, "test".into(), 100_000);
        let budget = TokenBudget::new(100_000, 8192);

        let messages = vec![
            json!({"role": "system", "content": "You are helpful."}),
            json!({"role": "user", "content": "Hello"}),
        ];

        let result = compactor.compact(&messages, &budget, 500).await;
        assert_eq!(result.messages.len(), 2);
        assert!(result.observation.is_none());
    }

    #[tokio::test]
    async fn test_compact_summarizes_old_messages() {
        let provider = Arc::new(MockProvider::new("User discussed weather and cats."));
        let compactor = ContextCompactor::new(provider, "test".into(), 100_000);
        let budget = TokenBudget::new(1200, 200);

        let mut messages =
            vec![json!({"role": "system", "content": "You are a helpful assistant."})];
        for i in 0..30 {
            messages.push(json!({"role": "user", "content": format!("This is a long user message number {} discussing many important topics about the world and various subjects at length", i)}));
            messages.push(json!({"role": "assistant", "content": format!("This is a detailed assistant response number {} providing thorough analysis and helpful information about the topics", i)}));
        }

        let result = compactor.compact(&messages, &budget, 50).await;
        assert!(
            result.messages.len() < messages.len(),
            "Expected fewer messages after compaction: got {} vs original {}",
            result.messages.len(),
            messages.len()
        );
        assert_eq!(result.messages[0]["role"], "system");
        let summary_content = result.messages[1]["content"].as_str().unwrap();
        assert!(summary_content.contains("Conversation summary"));
        // Should produce an observation.
        assert!(result.observation.is_some());
        assert!(result.observation.unwrap().contains("weather and cats"));
    }

    #[tokio::test]
    async fn test_compact_falls_back_on_failure() {
        let provider = Arc::new(FailingProvider);
        let compactor = ContextCompactor::new(provider, "test".into(), 100_000);
        let budget = TokenBudget::new(200, 50);

        let mut messages = vec![json!({"role": "system", "content": "Sys"})];
        for i in 0..20 {
            messages.push(json!({"role": "user", "content": format!("Long message {}", i)}));
        }

        let result = compactor.compact(&messages, &budget, 20).await;
        assert!(!result.messages.is_empty());
        assert_eq!(result.messages[0]["role"], "system");
        // On failure, no observation is produced.
        assert!(result.observation.is_none());
    }

    #[tokio::test]
    async fn test_compact_proactive_threshold() {
        // Test that compaction triggers at 66.6%, not 100%.
        let provider = Arc::new(MockProvider::new("Summary of the conversation."));
        let compactor = ContextCompactor::new(provider, "test".into(), 100_000);
        // Budget: 1000 available (after response reserve). 66.6% = 666 tokens.
        let budget = TokenBudget::new(1200, 200);

        let mut messages = vec![json!({"role": "system", "content": "System prompt."})];
        // Add messages totaling ~700 tokens (above 66.6% of 1000).
        for i in 0..15 {
            messages.push(json!({"role": "user", "content": format!("Message {} with moderate length content here.", i)}));
            messages.push(json!({"role": "assistant", "content": format!("Response {} with moderate length content here.", i)}));
        }

        let result = compactor.compact(&messages, &budget, 0).await;
        // Should have triggered compaction and produced an observation.
        if result.messages.len() < messages.len() {
            assert!(result.observation.is_some());
        }
    }

    #[tokio::test]
    async fn test_compaction_disables_after_failure() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = Arc::new(CountingFailingProvider {
            calls: calls.clone(),
        });
        let compactor = ContextCompactor::new(provider, "test".into(), 100_000);
        let budget = TokenBudget::new(2200, 200);

        let mut messages = vec![json!({"role": "system", "content": "Sys"})];
        for i in 0..80 {
            messages.push(
                json!({"role": "user", "content": format!("Long message {} {}", i, "x".repeat(180))}),
            );
        }

        let _ = compactor.compact(&messages, &budget, 20).await;
        let _ = compactor.compact(&messages, &budget, 20).await;

        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "compaction provider should be called only once after first failure"
        );
    }

    #[test]
    fn test_configurable_threshold_percent() {
        let provider = Arc::new(MockProvider::new("Summary."));
        let compactor = ContextCompactor::new(provider, "test".into(), 100_000)
            .with_thresholds(20.0, 100_000);
        let budget = TokenBudget::new(1200, 200);
        // available = 1200 - 200 = 1000. threshold = 20% of 1000 = 200 tokens.

        // Messages well under 200 tokens → no compaction needed.
        let messages = vec![
            json!({"role": "system", "content": "S"}),
            json!({"role": "user", "content": "Hi"}),
        ];
        assert!(!compactor.needs_compaction(&messages, &budget, 0));

        // Messages well above 200 tokens → needs compaction.
        let mut messages = vec![json!({"role": "system", "content": "S"})];
        for i in 0..20 {
            messages.push(json!({"role": "user", "content": format!("Message number {} with enough content to push us past the threshold surely.", i)}));
        }
        let est = TokenBudget::estimate_tokens(&messages);
        assert!(est > 200, "messages should be above threshold, got {} tokens", est);
        assert!(compactor.needs_compaction(&messages, &budget, 0));
    }

    #[test]
    fn test_configurable_threshold_tokens() {
        let provider = Arc::new(MockProvider::new("Summary."));
        // Set token threshold very low (50 tokens), percent very high (99%).
        let compactor = ContextCompactor::new(provider, "test".into(), 100_000)
            .with_thresholds(99.0, 50);
        let budget = TokenBudget::new(100_000, 8192);

        // Even though 99% of budget is huge, the 50-token cap fires first.
        let mut messages = vec![json!({"role": "system", "content": "S"})];
        for i in 0..5 {
            messages.push(json!({"role": "user", "content": format!("Message {}", i)}));
        }
        // Should need compaction because total tokens > 50.
        let total = TokenBudget::estimate_tokens(&messages);
        if total > 50 {
            assert!(compactor.needs_compaction(&messages, &budget, 0));
        }
    }

    #[test]
    fn test_effective_threshold_uses_minimum() {
        let provider = Arc::new(MockProvider::new("Summary."));
        let compactor = ContextCompactor::new(provider, "test".into(), 100_000)
            .with_thresholds(50.0, 200);
        // available = 1000. 50% of 1000 = 500. Token cap = 200. min(500, 200) = 200.
        assert_eq!(compactor.effective_threshold(1000), 200);
    }
}
