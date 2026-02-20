#![allow(dead_code)]
//! Context Gate: intelligent content management for LLM agents.
//!
//! Instead of uniform char-limit truncation, the gate makes context-aware
//! decisions based on the model's token budget:
//! - **Pass raw** when content fits
//! - **Briefing** (structural summary via compactor) when it doesn't,
//!   with full content cached to disk for drill-down via `read_file(lines=...)`

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

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
            build_simple_briefing(c, target_tokens)
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

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
}
