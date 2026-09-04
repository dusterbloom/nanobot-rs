// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(clippy::as_conversions, clippy::indexing_slicing)]
//! Anti-drift hook pipeline for SLM context stabilization.
//!
//! Small local models (3B–8B) degrade as context fills with noise — filler
//! responses, repeated failed attempts, babble without action. The existing
//! hygiene/compaction/trim pipeline fixes structural problems (orphans,
//! duplicates, overflow) but doesn't assess **quality**.
//!
//! This module adds PreCompletion and PostCompletion hooks that:
//! - Score turn quality via heuristics (no LLM calls)
//! - Evict polluted (low-quality) assistant turns
//! - Collapse repetitive tool-call attempts
//! - Re-inject format anchors periodically
//! - Collapse babble in text-only responses

use std::collections::HashSet;

use serde_json::{json, Value};
use tracing::debug;

use crate::config::schema::AntiDriftConfig;

// ---------------------------------------------------------------------------
// Pollution scoring
// ---------------------------------------------------------------------------

/// Heuristic quality score for a single assistant message.
struct PollutionScore {
    score: f32,
}

/// Filler phrases that inflate responses without adding information.
const FILLER_PHRASES: &[&str] = &[
    "certainly",
    "of course",
    "i'd be happy to",
    "i would be happy to",
    "sure thing",
    "absolutely",
    "great question",
    "that's a great",
    "let me help",
    "i can help",
    "no problem",
    "happy to help",
    "i understand",
    "thank you for",
    "thanks for",
    "as an ai",
    "as a language model",
    "let me think",
    "well",
    "so",
    "basically",
    "essentially",
    "actually",
    "honestly",
    "to be honest",
];

/// Score how "polluted" an assistant message is.
///
/// Returns 0.0 (clean) to 1.0 (pure noise). Weighted heuristics:
/// - `filler_heavy` (0.3): >30% filler words
/// - `repetitive` (0.3): trigram Jaccard > 0.4 with recent messages
/// - `babble_no_action` (0.2): >150 tokens, no tool calls, no code blocks
/// - `hallucination_marker` (0.2): fake tool XML / claim phrases
fn score_message(msg: &Value, prev_assistant_msgs: &[&Value]) -> PollutionScore {
    let content = msg_content(msg);
    let mut score = 0.0f32;

    // Signal 1: filler-heavy
    if filler_ratio(&content) > 0.30 {
        score += 0.3;
    }

    // Signal 2: repetitive (trigram overlap with recent assistant messages)
    for prev in prev_assistant_msgs.iter().take(3) {
        let prev_content = msg_content(prev);
        if trigram_jaccard(&content, &prev_content) > 0.4 {
            score += 0.3;
            break;
        }
    }

    // Signal 3: babble without action
    let word_count = content.split_whitespace().count();
    let has_tool_calls = msg
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .map_or(false, |a| !a.is_empty());
    let has_code_block = content.contains("```");
    if word_count > 150 && !has_tool_calls && !has_code_block {
        score += 0.2;
    }

    // Signal 4: hallucination markers
    let has_fake_tool_xml = content.contains("[Called")
        || content.contains("<tool_call>")
        || content.contains("<function_call>");
    let has_claim_without_evidence = content.contains("I ran") || content.contains("I executed");
    // Only flag claims if there's no preceding tool result in the message itself
    if has_fake_tool_xml || (has_claim_without_evidence && !has_tool_calls) {
        score += 0.2;
    }

    // Signal 5: condensed response markers from previous babble collapse
    if content.contains("[response condensed]") {
        score += 0.4;
    }

    PollutionScore {
        score: score.min(1.0),
    }
}

/// Reject-threshold for scoring standalone text (e.g. a produced LCM
/// compaction summary) as babble/degenerate. Mirrors `AntiDriftConfig`'s
/// default `pollution_threshold` — summaries don't carry a dedicated config
/// knob; reusing the same score and threshold as the per-turn pollution gate
/// keeps a single quality bar instead of a second tunable.
pub(crate) const SUMMARY_BABBLE_THRESHOLD: f32 = 0.6;

/// Score arbitrary text (e.g. a produced LCM summary) for pollution signals
/// using the same heuristics applied to per-turn assistant messages. Returns
/// `Some(score)` when the text crosses the babble/degenerate threshold —
/// callers must not persist it — or `None` when it's clean enough to keep.
pub(crate) fn score_summary_babble(text: &str) -> Option<f32> {
    let synthetic = json!({"role": "assistant", "content": text});
    let scored = score_message(&synthetic, &[]);
    (scored.score >= SUMMARY_BABBLE_THRESHOLD).then_some(scored.score)
}

/// Fraction of words that are filler phrases.
fn filler_ratio(content: &str) -> f32 {
    let lower = content.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    if words.is_empty() {
        return 0.0;
    }

    let mut filler_count = 0usize;
    for phrase in FILLER_PHRASES {
        let phrase_words: Vec<&str> = phrase.split_whitespace().collect();
        if phrase_words.len() == 1 {
            filler_count += words.iter().filter(|w| **w == phrase_words[0]).count();
        } else {
            // Multi-word phrase: count substring occurrences
            filler_count += lower.matches(phrase).count() * phrase_words.len();
        }
    }

    filler_count as f32 / words.len() as f32
}

/// Lexical diversity: fraction of distinct words over total words.
/// Degenerate babble repeats a handful of words (low diversity); a story,
/// explanation, or prose uses many distinct words (high diversity).
fn lexical_diversity(content: &str) -> f32 {
    let words: Vec<&str> = content.split_whitespace().collect();
    if words.is_empty() {
        return 1.0;
    }
    let unique = words.iter().collect::<std::collections::HashSet<_>>().len();
    unique as f32 / words.len() as f32
}

/// Decide whether a long assistant response is degenerate babble that must be
/// collapsed. Length alone is NOT enough: a long story, explanation, or code
/// block is legitimate and must be preserved verbatim so it survives into
/// sessions.db and stays recallable via extract / lcm_expand. We collapse only
/// when the text is genuinely low-quality: filler-heavy, lexically repetitive,
/// or carrying hallucination markers (fake tool calls / unbacked "I ran" claims).
fn is_babble(content: &str, max_tokens: usize) -> bool {
    let word_count = content.split_whitespace().count();
    if word_count <= max_tokens {
        return false;
    }
    if content.contains("```") {
        // Code blocks are legitimate long content — never collapse them.
        return false;
    }
    if filler_ratio(content) > 0.30 {
        return true;
    }
    if lexical_diversity(content) < 0.45 {
        return true;
    }
    let has_fake_tool = content.contains("[Called")
        || content.contains("<tool_call>")
        || content.contains("<function_call>");
    let has_claim = content.contains("I ran") || content.contains("I executed");
    has_fake_tool || has_claim
}

/// Word-trigram Jaccard similarity between two strings.
fn trigram_jaccard(a: &str, b: &str) -> f32 {
    let trigrams_a = word_trigrams(a);
    let trigrams_b = word_trigrams(b);
    if trigrams_a.is_empty() && trigrams_b.is_empty() {
        return 0.0;
    }
    let intersection = trigrams_a.intersection(&trigrams_b).count();
    let union = trigrams_a.union(&trigrams_b).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f32 / union as f32
}

fn word_trigrams(text: &str) -> HashSet<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 3 {
        return HashSet::new();
    }
    words
        .windows(3)
        .map(|w| format!("{} {} {}", w[0], w[1], w[2]).to_lowercase())
        .collect()
}

// ---------------------------------------------------------------------------
// PreCompletion pipeline
// ---------------------------------------------------------------------------

/// Run the pre-completion anti-drift pipeline on the message history.
///
/// Three steps:
/// 1. Evict polluted turns (replace with placeholder)
/// 2. Collapse repetitive tool-call attempts
/// 3. Inject format anchor (every N iterations)
pub fn pre_completion_pipeline(
    messages: &mut Vec<Value>,
    iteration: u32,
    config: &AntiDriftConfig,
    tools_active: bool,
) {
    let evicted = evict_polluted_turns(messages, config.pollution_threshold);
    let collapsed = collapse_repetitive_attempts(messages, config.repetition_min_count);
    let anchored = inject_format_anchor(messages, iteration, config.anchor_interval, tools_active);

    if evicted > 0 || collapsed > 0 || anchored {
        debug!(
            "Anti-drift pre: evicted={}, collapsed={}, anchored={}",
            evicted, collapsed, anchored
        );
    }
}

/// Replace polluted assistant messages with a placeholder.
///
/// Skips the safe window (last 4 messages) and messages with tool_calls.
fn evict_polluted_turns(messages: &mut Vec<Value>, threshold: f32) -> usize {
    let len = messages.len();
    let safe_window = 4;
    if len <= safe_window + 1 {
        return 0;
    }

    // Collect recent assistant messages as owned clones (avoids borrow conflict)
    let recent_assistant: Vec<Value> = messages
        .iter()
        .rev()
        .filter(|m| msg_role(m) == "assistant")
        .take(3)
        .cloned()
        .collect();
    let recent_refs: Vec<&Value> = recent_assistant.iter().collect();

    let mut evicted = 0;
    // Skip system (index 0) and safe window (last 4)
    let scan_end = len.saturating_sub(safe_window);
    for i in 1..scan_end {
        if msg_role(&messages[i]) != "assistant" {
            continue;
        }
        // Never evict messages with real tool calls
        let has_tool_calls = messages[i]
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .map_or(false, |a| !a.is_empty());
        if has_tool_calls {
            continue;
        }

        let score = score_message(&messages[i], &recent_refs);
        if score.score >= threshold {
            messages[i]["content"] = Value::String("[low-quality response removed]".to_string());
            evicted += 1;
        }
    }
    evicted
}

/// Collapse repetitive assistant messages (may be interleaved with other roles).
///
/// Fingerprints assistant messages by sorted tool names + first 50 chars.
/// Non-assistant messages (tool results, user messages) are transparent —
/// they don't break a run. When >= min_count matches in a window, replaces
/// all but last with placeholder.
fn collapse_repetitive_attempts(messages: &mut Vec<Value>, min_count: usize) -> usize {
    if messages.len() < 2 || min_count < 2 {
        return 0;
    }

    // Build (index, fingerprint) pairs for assistant messages only
    let assistant_fps: Vec<(usize, String)> = messages
        .iter()
        .enumerate()
        .filter_map(|(i, m)| {
            if msg_role(m) != "assistant" {
                return None;
            }
            let mut fp = String::new();
            if let Some(tcs) = m.get("tool_calls").and_then(|v| v.as_array()) {
                // Fingerprint each call by name AND arguments. Keying on the
                // name alone conflates distinct calls — reading five different
                // files all fingerprint as "read_file" and get collapsed as
                // "repetitive", which both discards real history AND rewrites
                // the prompt prefix every iteration (busting the inference
                // server's cache → full re-prefill). Only genuinely identical
                // calls (same tool, same args) are drift.
                let mut calls: Vec<String> = tcs
                    .iter()
                    .filter_map(|tc| {
                        let func = tc.get("function")?;
                        let name = func.get("name").and_then(|n| n.as_str())?;
                        let args = func.get("arguments").and_then(|a| a.as_str()).unwrap_or("");
                        Some(format!("{name}({args})"))
                    })
                    .collect();
                calls.sort();
                fp.push_str(&calls.join(","));
            }
            fp.push('|');
            let content = msg_content(m);
            let truncated: String = content.chars().take(50).collect();
            fp.push_str(&truncated);
            Some((i, fp))
        })
        .collect();

    let mut collapsed = 0;
    let mut k = 0;
    while k < assistant_fps.len() {
        let (_, ref fp) = assistant_fps[k];
        // Find run of identical fingerprints in the assistant-only sequence
        let mut run_end = k + 1;
        while run_end < assistant_fps.len() && assistant_fps[run_end].1 == *fp {
            run_end += 1;
        }
        let run_len = run_end - k;
        if run_len >= min_count {
            // Replace all but last in the run (using original message indices).
            // Keep `tool_calls` on the collapsed assistants: this pass only
            // walks assistant messages, so a `role=tool` result interleaved
            // after a collapsed assistant is never touched here. Removing the
            // `tool_calls` would orphan that result (its `tool_call_id` would
            // no longer match any assistant call) — and hygiene's
            // `remove_orphaned_tool_results` runs BEFORE anti-drift in
            // `retention::apply_shaping`, so orphans introduced here are not
            // cleaned up before the wire render on this iteration. Keeping the
            // calls preserves the protocol pairing invariant; only the verbose
            // preamble text is replaced with a compact placeholder.
            for j in k..(run_end - 1) {
                let msg_idx = assistant_fps[j].0;
                messages[msg_idx]["content"] = Value::String(format!(
                    "[{} previous similar attempts removed]",
                    run_len - 1
                ));
                collapsed += 1;
            }
            k = run_end;
        } else {
            k += 1;
        }
    }
    collapsed
}

/// Inject a format anchor when the model is drifting, at most once per
/// `interval` iterations.
///
/// The anchor corrects text-instead-of-tools drift, so it is **need-based, not
/// blind cadence**: a model that is actively calling tools is on-task, and
/// re-anchoring it every N iterations is pure context noise that also nudges it
/// into report→act cycles that burn iterations (observed: 5 anchors in a single
/// 15-iteration turn while the model called a tool every step). The interval is
/// only a floor on how often the anchor *can* fire; the drift check gates
/// whether it actually does.
fn inject_format_anchor(
    messages: &mut Vec<Value>,
    iteration: u32,
    interval: u32,
    tools_active: bool,
) -> bool {
    if interval == 0 || iteration == 0 || iteration % interval != 0 {
        return false;
    }
    // On-task models don't get nagged: skip when a recent assistant turn made a
    // tool call. Only genuine drift (recent turns all text-only) earns an anchor.
    // `tools_active` is computed on the FULL message array (before the
    // frozen-prefix split in prefix_guard) so the check sees recent assistant
    // turns even when they live in the cached prefix and are invisible to the
    // tail-only view inside `with_frozen_prefix`.
    if tools_active {
        return false;
    }
    // Don't inject if last message already contains an anchor
    if let Some(last) = messages.last() {
        let content = msg_content(last);
        if content.contains("[format-anchor]") {
            return false;
        }
    }
    messages.push(crate::agent::markers::scaffold_user(
        "[format-anchor] Reminder: use tool calls for actions, not text descriptions. Be concise. No XML imitation. If unsure, ask.",
    ));
    true
}

/// True if any of the last two assistant turns made a tool call — i.e. the model
/// is acting, not drifting into narration. Two turns (not one) so a single
/// sanctioned text report after a boundary nudge doesn't read as drift.
pub(crate) fn recent_assistant_used_tools(messages: &[Value]) -> bool {
    messages
        .iter()
        .rev()
        .filter(|m| msg_role(m) == "assistant")
        .take(2)
        .any(|m| {
            m.get("tool_calls")
                .and_then(|v| v.as_array())
                .is_some_and(|a| !a.is_empty())
        })
}

// ---------------------------------------------------------------------------
// PostCompletion pipeline
// ---------------------------------------------------------------------------

/// Run the post-completion anti-drift pipeline on the response content.
///
/// Collapses babble only when it is genuinely low-quality (see `is_babble`):
/// filler-heavy, lexically repetitive, or carrying hallucination markers. A
/// long story, explanation, or code block is preserved verbatim so the original
/// survives into sessions.db and stays recallable via extract / lcm_expand.
pub fn post_completion_pipeline(
    content: &mut String,
    _messages: &[Value],
    config: &AntiDriftConfig,
) {
    if is_babble(content, config.babble_max_tokens) {
        collapse_babble(content, config.babble_max_tokens);
    }
}

/// Truncate babble responses to first 2 sentences + marker.
///
/// When >=3 sentences, keeps first 2. When <=2 sentences but still over
/// max_tokens, keeps only the first sentence.
fn collapse_babble(content: &mut String, max_tokens: usize) {
    let word_count = content.split_whitespace().count();
    if word_count <= max_tokens {
        return;
    }

    let sentences: Vec<&str> = split_sentences(content);
    let condensed = if sentences.len() >= 3 {
        format!("{} {} [response condensed]", sentences[0], sentences[1])
    } else if sentences.len() == 2 {
        format!("{} [response condensed]", sentences[0])
    } else {
        // Single giant sentence — truncate by word count
        let truncated: String = content
            .split_whitespace()
            .take(max_tokens)
            .collect::<Vec<&str>>()
            .join(" ");
        format!("{} [response condensed]", truncated)
    };
    *content = condensed;
}

/// Simple sentence splitter (splits on `. `, `! `, `? ` boundaries).
fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    for i in 0..bytes.len().saturating_sub(1) {
        if (bytes[i] == b'.' || bytes[i] == b'!' || bytes[i] == b'?') && bytes[i + 1] == b' ' {
            let end = i + 1;
            let s = &text[start..end];
            if !s.trim().is_empty() {
                sentences.push(s.trim());
            }
            start = end + 1;
        }
    }
    // Trailing content
    if start < text.len() {
        let s = &text[start..];
        if !s.trim().is_empty() {
            sentences.push(s.trim());
        }
    }
    sentences
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn msg_role(msg: &Value) -> &str {
    msg.get("role").and_then(|v| v.as_str()).unwrap_or("")
}

fn msg_content(msg: &Value) -> String {
    msg.get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_filler_ratio_empty() {
        assert_eq!(filler_ratio(""), 0.0);
    }

    #[test]
    fn test_filler_ratio_high() {
        let text = "certainly absolutely well basically honestly actually so essentially";
        let ratio = filler_ratio(text);
        assert!(ratio > 0.5, "Expected high filler ratio, got {}", ratio);
    }

    #[test]
    fn test_filler_ratio_clean() {
        let text = "The function reads from disk and returns a parsed config struct.";
        let ratio = filler_ratio(text);
        assert!(ratio < 0.15, "Expected low filler ratio, got {}", ratio);
    }

    #[test]
    fn test_trigram_jaccard_identical() {
        let text = "the quick brown fox jumps over the lazy dog";
        let sim = trigram_jaccard(text, text);
        assert!((sim - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_trigram_jaccard_disjoint() {
        let a = "the quick brown fox jumps";
        let b = "a lazy red cat sleeps quietly nearby";
        let sim = trigram_jaccard(a, b);
        assert!(sim < 0.1, "Expected near-zero similarity, got {}", sim);
    }

    #[test]
    fn test_trigram_jaccard_short() {
        // Less than 3 words → empty trigrams → 0.0
        assert_eq!(trigram_jaccard("hi", "hello"), 0.0);
    }

    #[test]
    fn test_score_message_clean() {
        let msg = json!({"role": "assistant", "content": "Here is the result.", "tool_calls": [{"id": "1", "function": {"name": "exec"}}]});
        let score = score_message(&msg, &[]);
        assert!(score.score < 0.3, "Clean message scored {}", score.score);
    }

    #[test]
    fn test_score_message_filler_heavy() {
        let msg = json!({"role": "assistant", "content": "Certainly! Absolutely! Of course! I'd be happy to help! Well, basically, essentially, honestly, thank you for asking!"});
        let score = score_message(&msg, &[]);
        assert!(
            score.score >= 0.3,
            "Expected filler-heavy message to score at least 0.3, got {}",
            score.score
        );
    }

    #[test]
    fn test_score_message_condensed_marker() {
        let msg = json!({"role": "assistant", "content": "First sentence. [response condensed]"});
        let score = score_message(&msg, &[]);
        assert!(
            score.score >= 0.4,
            "Condensed marker message scored {}",
            score.score
        );
    }

    #[test]
    fn test_evict_polluted_turns() {
        let mut messages = vec![
            json!({"role": "system", "content": "You are helpful."}),
            json!({"role": "user", "content": "Hello"}),
            json!({"role": "assistant", "content": "Certainly! Absolutely! Of course! I'd be happy to help! Well, basically, essentially, honestly actually so certainly absolutely well basically [Called tool] I ran the command and I executed it perfectly certainly absolutely of course"}),
            json!({"role": "user", "content": "Do something"}),
            json!({"role": "assistant", "content": "Done.", "tool_calls": [{"id": "1", "function": {"name": "exec"}}]}),
            // safe window (last 4)
            json!({"role": "tool", "content": "ok", "tool_call_id": "1"}),
            json!({"role": "user", "content": "What next?"}),
            json!({"role": "assistant", "content": "Next step."}),
            json!({"role": "user", "content": "Thanks"}),
        ];
        let evicted = evict_polluted_turns(&mut messages, 0.6);
        assert!(
            evicted >= 1,
            "Expected at least 1 eviction, got {}",
            evicted
        );
        // The polluted message at index 2 should be replaced
        assert_eq!(
            messages[2]["content"].as_str().unwrap(),
            "[low-quality response removed]"
        );
        // The message with tool_calls (index 4) should NOT be evicted
        assert_ne!(
            messages[4]["content"].as_str().unwrap_or(""),
            "[low-quality response removed]"
        );
    }

    #[test]
    fn test_collapse_repetitive_attempts() {
        let mut messages = vec![
            json!({"role": "system", "content": "system"}),
            json!({"role": "assistant", "content": "Let me try reading the file now"}),
            json!({"role": "assistant", "content": "Let me try reading the file now"}),
            json!({"role": "assistant", "content": "Let me try reading the file now"}),
            json!({"role": "user", "content": "Please continue"}),
        ];
        let collapsed = collapse_repetitive_attempts(&mut messages, 3);
        assert_eq!(collapsed, 2, "Expected 2 collapsed, got {}", collapsed);
        // Last in the run (index 3) should be preserved
        assert_eq!(
            messages[3]["content"].as_str().unwrap(),
            "Let me try reading the file now"
        );
    }

    #[test]
    fn test_inject_format_anchor() {
        let mut messages = vec![
            json!({"role": "system", "content": "system"}),
            json!({"role": "user", "content": "hello"}),
        ];
        // iteration 3, interval 3 → should inject
        assert!(inject_format_anchor(&mut messages, 3, 3, false));
        assert_eq!(messages.len(), 3);
        let anchor = messages[2]["content"].as_str().unwrap();
        assert!(anchor.contains("[format-anchor]"));

        // Double injection should be prevented
        assert!(!inject_format_anchor(&mut messages, 6, 3, false));
    }

    #[test]
    fn test_format_anchor_skipped_when_model_using_tools() {
        // Need-based gate: a model that just made a tool call is on-task, so the
        // anchor must NOT fire even on an interval boundary — this is the root
        // cause of the scaffolding flood (an anchor every 3 iterations while the
        // model called a tool every step).
        let mut acting = vec![
            json!({"role": "user", "content": "read the files"}),
            json!({"role": "assistant", "content": "Reading.", "tool_calls": [{"id": "t1", "function": {"name": "read_file"}}]}),
            json!({"role": "tool", "content": "contents", "tool_call_id": "t1"}),
        ];
        assert!(
            !inject_format_anchor(&mut acting, 3, 3, true),
            "anchor must be skipped while the model is actively calling tools"
        );

        // Genuine drift (consecutive text-only turns) still earns an anchor.
        let mut drifting = vec![
            json!({"role": "user", "content": "do it"}),
            json!({"role": "assistant", "content": "Let me think about how I would approach this."}),
            json!({"role": "user", "content": "go on"}),
            json!({"role": "assistant", "content": "I believe it is probably fine, basically."}),
        ];
        assert!(
            inject_format_anchor(&mut drifting, 3, 3, false),
            "anchor should fire when recent assistant turns are all text-only"
        );
    }

    #[test]
    fn test_collapse_babble() {
        let mut content = "First sentence here. Second sentence here. Third long sentence that goes on and on with many many words to push the count over the limit and trigger condensation of the overall response text. Fourth sentence adds even more words to make it clear. Fifth for good measure with padding words.".to_string();
        collapse_babble(&mut content, 20);
        assert!(
            content.contains("[response condensed]"),
            "Expected condensed marker in: {}",
            content
        );
        assert!(
            content.starts_with("First sentence here."),
            "Expected to keep first sentence"
        );
    }

    #[test]
    fn test_collapse_babble_short_passes_through() {
        let mut content = "Short response.".to_string();
        collapse_babble(&mut content, 500);
        assert_eq!(content, "Short response.");
    }

    // -----------------------------------------------------------------------
    // Integration / end-to-end tests
    // -----------------------------------------------------------------------

    /// Build a realistic drifting conversation: system + 5 turns of user/assistant
    /// with degrading quality, interleaved tool calls and results.
    fn build_drifting_conversation() -> Vec<Value> {
        vec![
            json!({"role": "system", "content": "You are a helpful assistant."}),
            // Turn 1: clean
            json!({"role": "user", "content": "Read the config file"}),
            json!({"role": "assistant", "content": "Reading now.", "tool_calls": [{"id": "tc1", "function": {"name": "read_file"}}]}),
            json!({"role": "tool", "content": "key=value", "tool_call_id": "tc1"}),
            // Turn 2: slightly filler-heavy
            json!({"role": "user", "content": "Parse it"}),
            json!({"role": "assistant", "content": "Certainly! I'd be happy to help! Of course, absolutely, let me parse that config file for you right away, no problem at all, basically I understand what you need."}),
            // Turn 3: repeated attempt (same as turn 2 text)
            json!({"role": "user", "content": "Try again"}),
            json!({"role": "assistant", "content": "Certainly! I'd be happy to help! Of course, absolutely, let me parse that config file for you right away, no problem at all, basically I understand what you need."}),
            // Turn 4: repeated attempt again
            json!({"role": "user", "content": "Once more"}),
            json!({"role": "assistant", "content": "Certainly! I'd be happy to help! Of course, absolutely, let me parse that config file for you right away, no problem at all, basically I understand what you need."}),
            // Turn 5: hallucination marker
            json!({"role": "user", "content": "Did it work?"}),
            json!({"role": "assistant", "content": "[Called read_file] I ran the command and I executed the parsing. The results are ready."}),
            // Turn 6: STILL drifting — text-only narration, no tool call. This is
            // exactly when a format anchor helps (two consecutive text-only turns).
            json!({"role": "user", "content": "Show me"}),
            json!({"role": "assistant", "content": "I believe the values are probably correct based on my understanding."}),
            json!({"role": "user", "content": "Thanks"}),
        ]
    }

    #[test]
    fn test_e2e_full_drift_scenario_evicts_and_anchors() {
        let mut messages = build_drifting_conversation();
        let config = AntiDriftConfig::default(); // threshold=0.6, interval=3
        let original_len = messages.len();

        // Iteration 3 triggers anchor injection
        pre_completion_pipeline(&mut messages, 3, &config, false);

        // At least one polluted turn should be evicted
        let evicted_count = messages
            .iter()
            .filter(|m| msg_content(m) == "[low-quality response removed]")
            .count();
        assert!(
            evicted_count >= 1,
            "Expected at least 1 evicted turn in drifting conversation, got {}",
            evicted_count
        );

        // Format anchor should be injected at iteration 3
        let has_anchor = messages
            .iter()
            .any(|m| msg_content(m).contains("[format-anchor]"));
        assert!(has_anchor, "Expected format anchor at iteration 3");

        // Message count should grow by 1 (anchor) but no messages are deleted
        assert_eq!(
            messages.len(),
            original_len + 1,
            "Pipeline should add anchor but not delete messages"
        );
    }

    #[test]
    fn test_e2e_disabled_config_is_noop() {
        let mut messages = build_drifting_conversation();
        let original = messages.clone();
        let _config = AntiDriftConfig {
            enabled: false,
            ..Default::default()
        };

        // Even though we call the pipeline, with enabled=false the caller
        // would skip it. Test the pipeline directly — it should still work
        // but in practice the guard is in agent_loop.rs. Test by verifying
        // that a default config DOES modify and a custom threshold of 1.1
        // (impossible to reach) effectively disables eviction.
        let no_evict_config = AntiDriftConfig {
            pollution_threshold: 1.1,  // impossible score
            anchor_interval: 0,        // no anchors
            repetition_min_count: 999, // no collapse
            ..Default::default()
        };
        pre_completion_pipeline(&mut messages, 3, &no_evict_config, false);
        assert_eq!(
            messages, original,
            "Impossible thresholds should produce no changes"
        );
    }

    #[test]
    fn test_e2e_repetitive_attempts_with_interleaved_messages() {
        // Real-world pattern: assistant tries → tool fails → user says retry → repeat
        let mut messages = vec![
            json!({"role": "system", "content": "system"}),
            json!({"role": "assistant", "content": "Let me read the file.", "tool_calls": [{"id": "t1", "function": {"name": "read_file"}}]}),
            json!({"role": "tool", "content": "Error: file not found", "tool_call_id": "t1"}),
            json!({"role": "user", "content": "Try /tmp instead"}),
            json!({"role": "assistant", "content": "Let me read the file.", "tool_calls": [{"id": "t2", "function": {"name": "read_file"}}]}),
            json!({"role": "tool", "content": "Error: file not found", "tool_call_id": "t2"}),
            json!({"role": "user", "content": "Try again"}),
            json!({"role": "assistant", "content": "Let me read the file.", "tool_calls": [{"id": "t3", "function": {"name": "read_file"}}]}),
            json!({"role": "tool", "content": "Error: file not found", "tool_call_id": "t3"}),
            json!({"role": "user", "content": "Continue"}),
        ];

        let config = AntiDriftConfig::default();
        pre_completion_pipeline(&mut messages, 1, &config, false);

        // The 3 identical assistant attempts should be collapsed (all but last replaced)
        let collapsed_count = messages
            .iter()
            .filter(|m| msg_content(m).contains("previous similar attempts removed"))
            .count();
        assert!(
            collapsed_count >= 2,
            "Expected 2 collapsed attempts in interleaved conversation, got {}",
            collapsed_count
        );

        // The collapse must not orphan the interleaved tool results: collapsing
        // strips only the verbose preamble, leaving each `tool_call_id` paired
        // with its announcing assistant `tool_calls` entry.
        let orphans = orphaned_tool_result_ids(&messages);
        assert!(
            orphans.is_empty(),
            "interleaved tool results orphaned after collapse: {:?}",
            orphans
        );
    }

    /// Collect tool_call_ids announced by assistant messages *in order* so a
    /// caller can verify every `role=tool` result still has a matching call.
    fn announced_tool_call_ids(messages: &[Value]) -> HashSet<String> {
        messages
            .iter()
            .filter(|m| msg_role(m) == "assistant")
            .filter_map(|m| m.get("tool_calls").and_then(|v| v.as_array()))
            .flatten()
            .filter_map(|tc| tc.get("id").and_then(|i| i.as_str()).map(String::from))
            .collect()
    }

    /// Every `role=tool` message must have its `tool_call_id` present in some
    /// assistant `tool_calls` entry — otherwise it is an orphan result.
    fn orphaned_tool_result_ids(messages: &[Value]) -> Vec<String> {
        let known = announced_tool_call_ids(messages);
        messages
            .iter()
            .filter(|m| msg_role(m) == "tool")
            .filter_map(|m| {
                m.get("tool_call_id")
                    .and_then(|i| i.as_str())
                    .filter(|id| !known.contains(*id))
                    .map(String::from)
            })
            .collect()
    }

    #[test]
    fn test_collapse_repetitive_attempts_keeps_tool_result_pairing() {
        // The collapse pass must not orphan `role=tool` results: it walks
        // assistant messages only, so a `role=tool` result interleaved after a
        // collapsed assistant is never touched. Stripping `tool_calls` from
        // the assistant then leaves the result's `tool_call_id` unmatched.
        // Hygiene (which removes orphans) runs BEFORE anti-drift, so orphans
        // introduced here survive to the wire on this iteration.
        let mut messages = vec![
            json!({"role": "system", "content": "system"}),
            json!({"role": "assistant", "content": "Let me read the file.", "tool_calls": [{"id": "t1", "function": {"name": "read_file", "arguments": "{\"path\":\"/a\"}"}}]}),
            json!({"role": "tool", "content": "Error: file not found", "tool_call_id": "t1"}),
            json!({"role": "user", "content": "Try /tmp instead"}),
            json!({"role": "assistant", "content": "Let me read the file.", "tool_calls": [{"id": "t2", "function": {"name": "read_file", "arguments": "{\"path\":\"/a\"}"}}]}),
            json!({"role": "tool", "content": "Error: file not found", "tool_call_id": "t2"}),
            json!({"role": "user", "content": "Try again"}),
            json!({"role": "assistant", "content": "Let me read the file.", "tool_calls": [{"id": "t3", "function": {"name": "read_file", "arguments": "{\"path\":\"/a\"}"}}]}),
            json!({"role": "tool", "content": "Error: file not found", "tool_call_id": "t3"}),
            json!({"role": "user", "content": "Continue"}),
        ];

        let collapsed = collapse_repetitive_attempts(&mut messages, 3);
        assert_eq!(
            collapsed, 2,
            "the first two of three identical attempts collapse"
        );

        // No `role=tool` result may be left without its announcing assistant call.
        let orphans = orphaned_tool_result_ids(&messages);
        assert!(
            orphans.is_empty(),
            "collapse orphaned tool results {:?} — every tool_call_id must keep its assistant tool_calls entry",
            orphans
        );

        // The collapsed assistants must still carry their tool_calls so the
        // pairing above holds; only their textual preamble is placeholdered.
        for m in messages.iter().filter(|m| msg_role(m) == "assistant") {
            if msg_content(m).contains("previous similar attempts removed") {
                let has_calls = m
                    .get("tool_calls")
                    .and_then(|v| v.as_array())
                    .is_some_and(|a| !a.is_empty());
                assert!(
                    has_calls,
                    "collapsed assistant must keep tool_calls to avoid orphaning its tool result"
                );
            }
        }
    }

    #[test]
    fn test_collapse_distinguishes_tool_arguments() {
        let asst = |path: &str| {
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": path,
                    "function": {"name": "read_file", "arguments": format!("{{\"path\":\"{path}\"}}")}
                }]
            })
        };

        // Same tool, DIFFERENT args = legitimate multi-file work, NOT drift.
        // Must not collapse — collapsing both discards history and rewrites
        // the prompt prefix (busting the server's cache).
        let mut distinct = vec![asst("src/tui.rs"), asst("src/main.rs"), asst("src/lib.rs")];
        let original = distinct.clone();
        let collapsed = collapse_repetitive_attempts(&mut distinct, 2);
        assert_eq!(
            collapsed, 0,
            "distinct-arg calls must not collapse as drift"
        );
        assert_eq!(distinct, original, "distinct-arg history must be untouched");

        // Same tool, SAME args = genuine drift loop. Still collapses (all but
        // last), preserving the anti-drift behavior for real repetition.
        let mut same = vec![asst("a.rs"), asst("a.rs"), asst("a.rs")];
        let collapsed_same = collapse_repetitive_attempts(&mut same, 2);
        assert_eq!(collapsed_same, 2, "identical repeated calls still collapse");
        assert!(msg_content(&same[0]).contains("previous similar attempts removed"));
    }

    #[test]
    fn test_e2e_filler_ratio_never_exceeds_one() {
        // Overlapping multi-word filler phrases that could produce ratio > 1.0
        let text = "I'd be happy to help you. Happy to help! Certainly, of course, absolutely, I'd be happy to help again. Thank you for asking, thanks for that. I understand, let me help.";
        let ratio = filler_ratio(text);
        assert!(
            ratio <= 1.0,
            "filler_ratio must be capped at 1.0, got {}",
            ratio
        );
    }

    #[test]
    fn test_e2e_babble_collapse_two_long_sentences() {
        // Two extremely long sentences that exceed babble_max_tokens.
        // The function should still condense when only 2 sentences exist but they're huge.
        let long_sentence_a = format!(
            "This is the first sentence with {} padding words.",
            "very ".repeat(300)
        );
        let long_sentence_b = format!(
            "And this is the second sentence with {} more padding.",
            "extra ".repeat(300)
        );
        let mut content = format!("{} {}", long_sentence_a, long_sentence_b);
        let word_count = content.split_whitespace().count();
        assert!(
            word_count > 500,
            "Test setup: need >500 words, got {}",
            word_count
        );

        collapse_babble(&mut content, 500);

        // Should be condensed even with just 2 sentences, since total exceeds max
        assert!(
            content.len() < long_sentence_a.len() + long_sentence_b.len(),
            "Two-sentence babble should be condensed, but content length {} >= original {}",
            content.len(),
            long_sentence_a.len() + long_sentence_b.len()
        );
    }

    #[test]
    fn test_e2e_thinking_tags_then_babble_collapse() {
        // Simulates the agent_loop.rs flow: strip thinking → then babble collapse
        let mut content = format!(
            "<thinking>Internal reasoning that takes up 300 words {}</thinking>Short answer.",
            "blah ".repeat(60)
        );

        // Step 1: strip thinking tags (as agent_loop.rs does)
        let cleaned = crate::agent::compaction::strip_thinking_tags(&content);
        content = cleaned;

        // Step 2: post-completion pipeline
        let config = AntiDriftConfig::default();
        post_completion_pipeline(&mut content, &[], &config);

        // After stripping thinking, "Short answer." is < 500 words → no babble collapse
        assert_eq!(
            content, "Short answer.",
            "After stripping thinking, short content should pass through unchanged"
        );
    }

    #[test]
    fn test_e2e_thinking_tags_then_babble_collapse_still_long() {
        // When content AFTER thinking strip is still long, babble should fire
        let long_text = format!(
            "First sentence here. Second sentence here. {}",
            "And then more and more words follow in the third fourth fifth sentences. ".repeat(40)
        );
        let mut content = format!("<thinking>Some thought</thinking>{}", long_text);

        // Step 1: strip thinking tags
        content = crate::agent::compaction::strip_thinking_tags(&content);
        assert!(
            content.split_whitespace().count() > 500,
            "Setup: content after strip should exceed 500 words, got {}",
            content.split_whitespace().count()
        );

        // Step 2: post-completion pipeline
        let config = AntiDriftConfig::default();
        post_completion_pipeline(&mut content, &[], &config);

        assert!(
            content.contains("[response condensed]"),
            "Long content after thinking strip should be condensed"
        );
    }

    #[test]
    fn test_e2e_long_story_preserved_not_collapsed() {
        // A long coherent story must survive verbatim so it stays recallable.
        let story = "The operator asked nano32 to recall the morning. Clouds drifted \
            above the harbor while the agent summarized three conversations. A lighthouse \
            blinked across the bay. The model retrieved the diary entry without hesitation. \
            Wind carried salt through the open window. Memory crystals hummed in the server \
            rack. She replied with a quiet parable about two threads of time. The user smiled \
            and closed the notebook. Nothing was lost in the archive. A second story followed \
            about the river and the signal. The ember glowed beneath the lattice. Horizons \
            expanded as the chronicle turned. Quantum whispers reached the beacon. The spiral \
            held steady through the cascade. Verse after verse the agent remembered everything.";

        // Use a low threshold so a readable story still exceeds babble_max_tokens.
        let mut config = AntiDriftConfig::default();
        config.babble_max_tokens = 20;

        let mut content = story.to_string();
        post_completion_pipeline(&mut content, &[], &config);

        assert_eq!(
            content, story,
            "A long coherent story must NOT be collapsed — it must stay recallable"
        );
    }

    #[test]
    fn test_is_babble_gate() {
        // Lower threshold so readable content still trips the length check.
        let mut config = AntiDriftConfig::default();
        config.babble_max_tokens = 30;

        // Long, lexically diverse prose is NOT babble — preserved verbatim.
        let story = "The operator asked nano32 to recall the morning. Clouds drifted \
            above the harbor while the agent summarized three conversations. A lighthouse \
            blinked across the bay. The model retrieved the diary entry without hesitation. \
            Wind carried salt through the open window. Memory crystals hummed in the server \
            rack. She replied with a quiet parable about two threads of time. The user smiled \
            and closed the notebook. Nothing was lost in the archive.";
        let mut content = story.to_string();
        post_completion_pipeline(&mut content, &[], &config);
        assert_eq!(content, story, "diverse long prose preserved");

        // Long repetitive padding IS babble and collapses.
        let mut padded = format!(
            "First sentence here. Second sentence here. {}",
            "And then more and more words follow in the third fourth fifth sentences. ".repeat(40)
        );
        post_completion_pipeline(&mut padded, &[], &config);
        assert!(
            padded.contains("[response condensed]"),
            "repetitive padding collapses"
        );

        // Long code block is NOT babble (legitimate long content).
        let code = format!("```\n{}\n```", "fn main() {}\n".repeat(60));
        let mut content = code.clone();
        post_completion_pipeline(&mut content, &[], &config);
        assert_eq!(content, code, "long code block preserved");
    }

    #[test]
    fn test_e2e_hygiene_then_anti_drift_compose() {
        // Run both pipelines in sequence (as agent_loop does) on a messy conversation
        let mut messages = vec![
            json!({"role": "system", "content": "system"}),
            // Orphaned tool result (hygiene should remove)
            json!({"role": "tool", "content": "orphan", "tool_call_id": "tc_orphan"}),
            json!({"role": "user", "content": "hello"}),
            // Filler-heavy (anti-drift should evict)
            json!({"role": "assistant", "content": "Certainly! Absolutely! Of course! Well, basically, honestly, I understand! I'd be happy to help! Thank you for asking! No problem!"}),
            json!({"role": "user", "content": "Do it"}),
            json!({"role": "assistant", "content": "[Called read_file] I ran the command and I executed it."}),
            // Safe window — still drifting (text-only), so the anchor fires.
            json!({"role": "user", "content": "next"}),
            json!({"role": "assistant", "content": "I think it is basically done now, probably."}),
            json!({"role": "user", "content": "thanks"}),
        ];

        // Step 1: hygiene (structural)
        crate::agent::context_hygiene::hygiene_pipeline(&mut messages, 20);

        // Orphan should be removed
        let has_orphan = messages.iter().any(|m| msg_content(m) == "orphan");
        assert!(!has_orphan, "Hygiene should remove orphaned tool results");

        // Step 2: anti-drift (quality)
        let config = AntiDriftConfig::default();
        pre_completion_pipeline(&mut messages, 3, &config, false);

        // Filler-heavy message should be evicted (if not in safe window)
        let evicted = messages
            .iter()
            .filter(|m| msg_content(m) == "[low-quality response removed]")
            .count();
        assert!(
            evicted >= 1,
            "Anti-drift should evict filler-heavy messages"
        );

        // Anchor should be injected at iteration 3
        let has_anchor = messages
            .iter()
            .any(|m| msg_content(m).contains("[format-anchor]"));
        assert!(
            has_anchor,
            "Anti-drift should inject format anchor at iteration 3"
        );
    }

    #[test]
    fn test_e2e_safe_window_never_touched() {
        // Even heavily polluted messages in the last 4 positions should survive
        let mut messages = vec![
            json!({"role": "system", "content": "system"}),
            json!({"role": "user", "content": "hello"}),
            json!({"role": "assistant", "content": "ok"}),
            // Safe window starts here (last 4)
            json!({"role": "user", "content": "do stuff"}),
            json!({"role": "assistant", "content": "Certainly! Absolutely! Of course! I'd be happy to! Well basically honestly [Called tool] I ran it!"}),
            json!({"role": "user", "content": "more"}),
            json!({"role": "assistant", "content": "Certainly again! Absolutely! Basically!"}),
        ];

        let config = AntiDriftConfig {
            pollution_threshold: 0.0,
            ..Default::default()
        }; // evict everything
        pre_completion_pipeline(&mut messages, 1, &config, false);

        // Messages in safe window (last 4, indices 3-6) should NOT be replaced
        assert_ne!(
            messages[4]["content"].as_str().unwrap(),
            "[low-quality response removed]",
            "Safe window message should not be evicted"
        );
        assert_ne!(
            messages[6]["content"].as_str().unwrap(),
            "[low-quality response removed]",
            "Safe window message should not be evicted"
        );
    }

    #[test]
    fn test_e2e_score_multi_signal_accumulation() {
        // Message that triggers filler + hallucination + babble = 0.7
        // Need >150 words for babble_no_action signal
        let filler_and_hallucination =
            "Certainly! Absolutely! Of course! Well, basically, honestly, I understand! "
                .repeat(20);
        let content = format!(
            "{} [Called read_file] I ran the command and I executed it successfully.",
            filler_and_hallucination
        );
        assert!(
            content.split_whitespace().count() > 150,
            "Setup: need >150 words for babble signal"
        );
        let msg = json!({"role": "assistant", "content": content});
        let score = score_message(&msg, &[]);
        assert!(
            score.score >= 0.6,
            "Message with multiple signals should score >= 0.6, got {}",
            score.score
        );
    }

    #[test]
    fn test_e2e_config_roundtrip_through_trio() {
        let trio_json = r#"{
            "enabled": true,
            "antiDrift": {
                "enabled": true,
                "anchorInterval": 5,
                "pollutionThreshold": 0.7,
                "babbleMaxTokens": 300,
                "repetitionMinCount": 4
            }
        }"#;
        let trio: crate::config::schema::TrioConfig = serde_json::from_str(trio_json).unwrap();
        assert!(trio.anti_drift.enabled);
        assert_eq!(trio.anti_drift.anchor_interval, 5);
        assert!((trio.anti_drift.pollution_threshold - 0.7).abs() < f32::EPSILON);
        assert_eq!(trio.anti_drift.babble_max_tokens, 300);
        assert_eq!(trio.anti_drift.repetition_min_count, 4);

        // Roundtrip
        let json = serde_json::to_string(&trio).unwrap();
        let trio2: crate::config::schema::TrioConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(trio2.anti_drift.anchor_interval, 5);
    }

    #[test]
    fn test_e2e_config_defaults_through_root() {
        let json = r#"{}"#;
        let cfg: crate::config::schema::Config = serde_json::from_str(json).unwrap();
        assert!(cfg.trio.anti_drift.enabled);
        assert_eq!(cfg.trio.anti_drift.anchor_interval, 3);
        assert!((cfg.trio.anti_drift.pollution_threshold - 0.6).abs() < f32::EPSILON);
        assert_eq!(cfg.trio.anti_drift.babble_max_tokens, 500);
        assert_eq!(cfg.trio.anti_drift.repetition_min_count, 3);
    }
}
