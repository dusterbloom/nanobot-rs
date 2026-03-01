//! Proactive information gathering before LLM tool calls.
//!
//! Classifies user intent from raw text, extracts search terms, retrieves
//! grounding from the knowledge store and learning context, and formats a
//! pre-prompt snippet that the caller can inject before invoking the LLM.

use tracing::debug;

use crate::agent::knowledge_store::KnowledgeStore;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum IntentCategory {
    FileOperation,
    CodeExecution,
    WebResearch,
    KnowledgeQuery,
    Communication,
    MemoryRecall,
    Ambiguous,
}

pub struct IntentSignal {
    pub category: IntentCategory,
    pub likely_tools: Vec<String>,
    pub search_terms: Vec<String>,
    pub confidence: f32,
}

pub struct GroundingPayload {
    pub knowledge_snippets: Vec<String>,
    pub tool_hints: String,
    pub estimated_tokens: usize,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Classify the user's intent from raw text.
///
/// Patterns are checked most-specific-first. A single category is returned
/// together with a list of likely tools and search terms.
pub fn extract_intent(user_text: &str) -> IntentSignal {
    let lower = user_text.to_lowercase();

    // 1. Memory recall — must precede knowledge-query to capture "what did we"
    if lower.contains("remember when")
        || lower.contains("recall")
        || lower.contains("what did we")
        || lower.contains("previously")
    {
        debug!("intent: MemoryRecall");
        return IntentSignal {
            category: IntentCategory::MemoryRecall,
            likely_tools: vec![],
            search_terms: extract_search_terms(user_text),
            confidence: 0.7,
        };
    }

    // 2. File operations — path-like token + action verb
    if has_path_like(&lower) {
        let tools = if lower.contains("read ") || lower.contains("show ") || lower.contains("cat ") {
            vec!["read_file".to_string()]
        } else if lower.contains("write ") || lower.contains("create ") {
            vec!["write_file".to_string()]
        } else if lower.contains("edit ")
            || lower.contains("fix ")
            || lower.contains("modify ")
        {
            vec!["edit_file".to_string(), "read_file".to_string()]
        } else if lower.contains("list ") {
            vec!["list_dir".to_string()]
        } else {
            // Path-like but no clear verb — still a file op
            vec!["read_file".to_string()]
        };

        // Only treat as FileOperation when at least one recognisable file verb is present
        let has_file_verb = lower.contains("read ")
            || lower.contains("show ")
            || lower.contains("write ")
            || lower.contains("edit ")
            || lower.contains("fix ")
            || lower.contains("list ")
            || lower.contains("cat ");

        if has_file_verb {
            debug!("intent: FileOperation");
            return IntentSignal {
                category: IntentCategory::FileOperation,
                likely_tools: tools,
                search_terms: extract_search_terms(user_text),
                confidence: 0.6,
            };
        }
    }

    // 3. Code execution
    let code_keywords = [
        "run ", "execute ", "cargo ", "npm ", "git ", "build ", "compile ", "test ",
    ];
    // Also match keyword at end of string (trim handles trailing whitespace)
    let trimmed = lower.trim_end();
    let code_match = code_keywords
        .iter()
        .any(|kw| lower.contains(kw) || trimmed.ends_with(kw.trim_end()));
    if code_match {
        debug!("intent: CodeExecution");
        return IntentSignal {
            category: IntentCategory::CodeExecution,
            likely_tools: vec!["exec".to_string()],
            search_terms: extract_search_terms(user_text),
            confidence: 0.6,
        };
    }

    // 4. Web research
    let has_url = lower.contains("http://") || lower.contains("https://");
    let has_web_kw = lower.contains("search for")
        || lower.contains("look up")
        || lower.contains("research");
    if has_url || has_web_kw {
        debug!("intent: WebResearch");
        return IntentSignal {
            category: IntentCategory::WebResearch,
            likely_tools: vec!["web_search".to_string(), "web_fetch".to_string()],
            search_terms: extract_search_terms(user_text),
            confidence: 0.5,
        };
    }

    // 5. Communication
    if lower.contains("send ") || lower.contains("message ") || lower.contains("email ") {
        debug!("intent: Communication");
        return IntentSignal {
            category: IntentCategory::Communication,
            likely_tools: vec!["send_message".to_string()],
            search_terms: extract_search_terms(user_text),
            confidence: 0.5,
        };
    }

    // 6. Knowledge query
    if lower.contains("what is")
        || lower.contains("how does")
        || lower.contains("explain")
        || lower.contains("tell me about")
    {
        debug!("intent: KnowledgeQuery");
        return IntentSignal {
            category: IntentCategory::KnowledgeQuery,
            likely_tools: vec![],
            search_terms: extract_search_terms(user_text),
            confidence: 0.4,
        };
    }

    // 7. Ambiguous fallback
    debug!("intent: Ambiguous");
    IntentSignal {
        category: IntentCategory::Ambiguous,
        likely_tools: vec![],
        search_terms: extract_search_terms(user_text),
        confidence: 0.1,
    }
}

/// Extract meaningful search terms from raw text.
///
/// Lowercases, splits on whitespace, strips punctuation, removes stop words
/// and common tool-verb noise, then returns up to 8 terms.
pub fn extract_search_terms(text: &str) -> Vec<String> {
    const STOP_WORDS: &[&str] = &[
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "it", "this", "that", "what",
        "how", "when", "where", "who", "which", "me", "about",
    ];
    const TOOL_VERBS: &[&str] = &[
        "read", "write", "edit", "fix", "show", "list", "run", "execute",
        "search", "send", "message", "explain", "tell", "look", "up",
    ];

    text.to_lowercase()
        .split_whitespace()
        .map(|token| {
            token
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|t| !t.is_empty())
        .filter(|t| !STOP_WORDS.contains(&t.as_str()))
        .filter(|t| !TOOL_VERBS.contains(&t.as_str()))
        .take(8)
        .collect()
}

/// Gather grounding material relevant to the intent.
///
/// Queries the knowledge store (if provided) and filters the learning context
/// for lines mentioning the likely tools. Returns an empty payload when there
/// is not enough signal (Ambiguous with low confidence, or no search terms).
pub fn retrieve_grounding(
    intent: &IntentSignal,
    knowledge_store: Option<&KnowledgeStore>,
    learning_context: &str,
    max_tokens: usize,
) -> GroundingPayload {
    // Skip grounding for low-signal intents
    if intent.category == IntentCategory::Ambiguous && intent.confidence < 0.2 {
        return GroundingPayload {
            knowledge_snippets: vec![],
            tool_hints: String::new(),
            estimated_tokens: 0,
        };
    }

    if intent.search_terms.is_empty() {
        return GroundingPayload {
            knowledge_snippets: vec![],
            tool_hints: String::new(),
            estimated_tokens: 0,
        };
    }

    let token_budget_chars = max_tokens * 4; // chars ≈ tokens * 4
    let half_budget = token_budget_chars / 2;

    // --- Knowledge store snippets ---
    let mut knowledge_snippets: Vec<String> = vec![];
    if let Some(ks) = knowledge_store {
        let query = intent.search_terms.join(" ");
        debug!("proactive: searching knowledge store for {:?}", query);
        if let Ok(hits) = ks.search(&query, 5) {
            let mut chars_used = 0usize;
            for hit in hits {
                if chars_used + hit.snippet.len() > half_budget {
                    break;
                }
                chars_used += hit.snippet.len();
                knowledge_snippets.push(hit.snippet);
            }
        }
    }

    // --- Tool hints from learning context ---
    let tool_hints: String = if intent.likely_tools.is_empty() {
        String::new()
    } else {
        let mut lines_used = 0usize;
        learning_context
            .lines()
            .filter(|line| {
                intent
                    .likely_tools
                    .iter()
                    .any(|tool| line.contains(tool.as_str()))
            })
            .take_while(|line| {
                lines_used += line.len() + 1;
                lines_used <= half_budget
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    let estimated_tokens = (knowledge_snippets.iter().map(|s| s.len()).sum::<usize>()
        + tool_hints.len())
        / 4;

    GroundingPayload {
        knowledge_snippets,
        tool_hints,
        estimated_tokens,
    }
}

/// Format the grounding payload as an optional pre-prompt message.
///
/// Returns `None` when there is nothing to inject.
pub fn format_grounding_message(payload: &GroundingPayload) -> Option<String> {
    if payload.knowledge_snippets.is_empty() && payload.tool_hints.is_empty() {
        return None;
    }

    let mut out = String::from("[grounding] Relevant knowledge:\n");
    for snippet in &payload.knowledge_snippets {
        out.push_str("- ");
        out.push_str(snippet);
        out.push('\n');
    }
    if !payload.tool_hints.is_empty() {
        out.push_str("\nTool context:\n");
        out.push_str(&payload.tool_hints);
        out.push('\n');
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Heuristic: does the lowercased text look like it contains a file path?
fn has_path_like(lower: &str) -> bool {
    lower.contains('/')
        || lower.contains(".rs")
        || lower.contains(".txt")
        || lower.contains(".md")
        || lower.contains(".json")
        || lower.contains(".py")
        || lower.contains(".js")
        || lower.contains(".ts")
        || lower.contains(".toml")
        || lower.contains(".yaml")
        || lower.contains(".yml")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Phase 1: Intent Detection ----

    #[test]
    fn test_extract_intent_file_read() {
        let intent = extract_intent("read src/main.rs");
        assert_eq!(intent.category, IntentCategory::FileOperation);
        assert!(intent.likely_tools.contains(&"read_file".to_string()));
        assert!(intent.confidence >= 0.5);
    }

    #[test]
    fn test_extract_intent_file_edit() {
        let intent = extract_intent("fix the bug in config.json");
        assert_eq!(intent.category, IntentCategory::FileOperation);
        assert!(intent.likely_tools.contains(&"edit_file".to_string()));
        assert!(intent.likely_tools.contains(&"read_file".to_string()));
    }

    #[test]
    fn test_extract_intent_code_execution() {
        let intent = extract_intent("run cargo test");
        assert_eq!(intent.category, IntentCategory::CodeExecution);
        assert!(intent.likely_tools.contains(&"exec".to_string()));
    }

    #[test]
    fn test_extract_intent_web_url() {
        let intent = extract_intent("summarize https://example.com");
        assert_eq!(intent.category, IntentCategory::WebResearch);
    }

    #[test]
    fn test_extract_intent_web_keyword() {
        let intent = extract_intent("search for rust async");
        assert_eq!(intent.category, IntentCategory::WebResearch);
    }

    #[test]
    fn test_extract_intent_knowledge() {
        let intent = extract_intent("what is the borrow checker");
        assert_eq!(intent.category, IntentCategory::KnowledgeQuery);
        assert!(intent.likely_tools.is_empty());
    }

    #[test]
    fn test_extract_intent_memory() {
        let intent = extract_intent("what did we discuss last time");
        assert_eq!(intent.category, IntentCategory::MemoryRecall);
    }

    #[test]
    fn test_extract_intent_ambiguous() {
        let intent = extract_intent("hello");
        assert_eq!(intent.category, IntentCategory::Ambiguous);
        assert!(intent.confidence < 0.2);
    }

    #[test]
    fn test_extract_intent_communication() {
        let intent = extract_intent("send a message to the team");
        assert_eq!(intent.category, IntentCategory::Communication);
    }

    // ---- Phase 2: Search Terms ----

    #[test]
    fn test_search_terms_removes_stops() {
        let terms = extract_search_terms("what is the rust borrow checker");
        assert_eq!(terms, vec!["rust", "borrow", "checker"]);
    }

    #[test]
    fn test_search_terms_max_eight() {
        let terms = extract_search_terms(
            "one two three four five six seven eight nine ten eleven twelve",
        );
        assert!(terms.len() <= 8);
    }

    #[test]
    fn test_search_terms_empty() {
        let terms = extract_search_terms("");
        assert!(terms.is_empty());
    }

    #[test]
    fn test_search_terms_strips_punctuation() {
        let terms = extract_search_terms("rust?");
        assert_eq!(terms, vec!["rust"]);
    }

    // ---- Phase 3: Grounding Retrieval ----

    #[test]
    fn test_retrieve_no_knowledge_store() {
        let intent = IntentSignal {
            category: IntentCategory::KnowledgeQuery,
            likely_tools: vec![],
            search_terms: vec!["rust".to_string()],
            confidence: 0.4,
        };
        let payload = retrieve_grounding(&intent, None, "", 500);
        assert!(payload.knowledge_snippets.is_empty());
    }

    #[test]
    fn test_retrieve_with_hits() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_knowledge.db");
        let ks = KnowledgeStore::open(&db_path).unwrap();
        ks.ingest(
            "test_doc",
            None,
            "The Rust borrow checker ensures memory safety without garbage collection.",
            4096,
            256,
        )
        .unwrap();
        let intent = IntentSignal {
            category: IntentCategory::KnowledgeQuery,
            likely_tools: vec![],
            search_terms: vec!["borrow".to_string(), "checker".to_string()],
            confidence: 0.4,
        };
        let payload = retrieve_grounding(&intent, Some(&ks), "", 500);
        assert!(!payload.knowledge_snippets.is_empty());
        let joined = payload.knowledge_snippets.join(" ");
        assert!(joined.to_lowercase().contains("borrow"));
    }

    #[test]
    fn test_retrieve_respects_budget() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_knowledge.db");
        let ks = KnowledgeStore::open(&db_path).unwrap();
        let large_content = "Rust borrow checker ".repeat(200);
        ks.ingest("large_doc", None, &large_content, 4096, 256).unwrap();
        let intent = IntentSignal {
            category: IntentCategory::KnowledgeQuery,
            likely_tools: vec![],
            search_terms: vec!["borrow".to_string()],
            confidence: 0.4,
        };
        let payload = retrieve_grounding(&intent, Some(&ks), "", 50);
        // With budget of 50 tokens ≈ 200 chars, content should be truncated
        let total_chars: usize = payload
            .knowledge_snippets
            .iter()
            .map(|s| s.len())
            .sum::<usize>()
            + payload.tool_hints.len();
        assert!(total_chars <= 50 * 4 + 50); // some tolerance
    }

    #[test]
    fn test_retrieve_ambiguous_skips() {
        let intent = IntentSignal {
            category: IntentCategory::Ambiguous,
            likely_tools: vec![],
            search_terms: vec!["hello".to_string()],
            confidence: 0.1,
        };
        let payload = retrieve_grounding(&intent, None, "some context", 500);
        assert!(payload.knowledge_snippets.is_empty());
        assert!(payload.tool_hints.is_empty());
    }

    #[test]
    fn test_retrieve_no_terms_skips() {
        let intent = IntentSignal {
            category: IntentCategory::KnowledgeQuery,
            likely_tools: vec![],
            search_terms: vec![],
            confidence: 0.4,
        };
        let payload = retrieve_grounding(&intent, None, "some context", 500);
        assert!(payload.knowledge_snippets.is_empty());
    }

    // ---- Phase 4: Formatting ----

    #[test]
    fn test_format_with_snippets() {
        let payload = GroundingPayload {
            knowledge_snippets: vec!["Rust is a systems language.".to_string()],
            tool_hints: String::new(),
            estimated_tokens: 10,
        };
        let result = format_grounding_message(&payload);
        assert!(result.is_some());
        let text = result.unwrap();
        assert!(text.contains("[grounding]"));
        assert!(text.contains("Rust is a systems language."));
    }

    #[test]
    fn test_format_empty_returns_none() {
        let payload = GroundingPayload {
            knowledge_snippets: vec![],
            tool_hints: String::new(),
            estimated_tokens: 0,
        };
        assert!(format_grounding_message(&payload).is_none());
    }

    #[test]
    fn test_format_with_both() {
        let payload = GroundingPayload {
            knowledge_snippets: vec!["Snippet one.".to_string()],
            tool_hints: "read_file: 5/5 succeeded".to_string(),
            estimated_tokens: 20,
        };
        let result = format_grounding_message(&payload).unwrap();
        assert!(result.contains("[grounding]"));
        assert!(result.contains("Snippet one."));
        assert!(result.contains("Tool context:"));
        assert!(result.contains("read_file"));
    }

    // ---- Phase 5: E2E ----

    #[test]
    fn test_e2e_full_pipeline() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_knowledge.db");
        let ks = KnowledgeStore::open(&db_path).unwrap();
        ks.ingest(
            "rust_doc",
            None,
            "The borrow checker is Rust's key innovation for memory safety.",
            4096,
            256,
        )
        .unwrap();

        let intent = extract_intent("explain borrow checker");
        // Should be KnowledgeQuery since "explain" is a knowledge keyword
        let learning = "- read_file: 5/5 succeeded recently\n- exec: 3/4 succeeded recently";
        let payload = retrieve_grounding(&intent, Some(&ks), learning, 500);
        let result = format_grounding_message(&payload);
        assert!(result.is_some());
        let text = result.unwrap();
        assert!(text.contains("[grounding]"));
        assert!(text.to_lowercase().contains("borrow"));
    }

    #[test]
    fn test_e2e_empty_graceful() {
        let intent = extract_intent("explain borrow checker");
        let payload = retrieve_grounding(&intent, None, "", 500);
        let result = format_grounding_message(&payload);
        assert!(result.is_none());
    }
}
