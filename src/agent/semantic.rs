//! BM25-based semantic memory retrieval.
//!
//! Indexes conversation turns from the SQLite store plus memory files
//! (observations, MEMORY.md sections, daily notes) and returns relevant
//! context for a query, ranked by BM25 score.
//!
//! Zero additional dependencies beyond `rusqlite` (Phase 0).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use tracing::warn;

use crate::store::conversation::ConversationStore;

// ---------------------------------------------------------------------------
// Tokenizer (simple whitespace + punctuation split)
// ---------------------------------------------------------------------------

/// Common English stop words to filter out.
const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "between",
    "through", "during", "before", "after", "above", "below", "and", "but",
    "or", "not", "no", "if", "then", "else", "when", "up", "out", "that",
    "this", "it", "i", "you", "he", "she", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our", "their",
    "what", "which", "who", "whom", "so", "than", "too", "very",
];

/// Tokenize text: split on non-alphanumeric, lowercase, filter stop words.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .map(|w| w.to_lowercase())
        .filter(|w| w.len() > 1 && !STOP_WORDS.contains(&w.as_str()))
        .collect()
}

// ---------------------------------------------------------------------------
// BM25 Engine
// ---------------------------------------------------------------------------

/// BM25 parameters.
const K1: f64 = 1.2;
const B: f64 = 0.75;

/// A document in the index.
struct IndexedDoc {
    id: String,
    term_freqs: HashMap<String, u32>,
    doc_len: u32,
}

/// BM25 scoring engine with inverted index.
struct BM25Engine {
    docs: Vec<IndexedDoc>,
    /// term -> set of doc indices
    postings: HashMap<String, Vec<usize>>,
    /// Total documents indexed.
    total_docs: usize,
    /// Average document length.
    avg_dl: f64,
}

impl BM25Engine {
    fn new() -> Self {
        Self {
            docs: Vec::new(),
            postings: HashMap::new(),
            total_docs: 0,
            avg_dl: 0.0,
        }
    }

    /// Add a document to the index.
    fn add_doc(&mut self, id: String, text: &str) {
        let tokens = tokenize(text);
        let doc_len = tokens.len() as u32;

        let mut term_freqs: HashMap<String, u32> = HashMap::new();
        for token in &tokens {
            *term_freqs.entry(token.clone()).or_insert(0) += 1;
        }

        let doc_idx = self.docs.len();
        for term in term_freqs.keys() {
            self.postings
                .entry(term.clone())
                .or_default()
                .push(doc_idx);
        }

        self.docs.push(IndexedDoc {
            id,
            term_freqs,
            doc_len,
        });

        // Update stats.
        self.total_docs = self.docs.len();
        let total_len: u64 = self.docs.iter().map(|d| d.doc_len as u64).sum();
        self.avg_dl = if self.total_docs > 0 {
            total_len as f64 / self.total_docs as f64
        } else {
            0.0
        };
    }

    /// Query the index and return (doc_id, score) pairs sorted by score descending.
    fn query(&self, text: &str, max_results: usize) -> Vec<(String, f64)> {
        if self.total_docs == 0 {
            return Vec::new();
        }

        let query_tokens = tokenize(text);
        let mut scores: HashMap<usize, f64> = HashMap::new();

        for token in &query_tokens {
            let df = self
                .postings
                .get(token)
                .map(|p| p.len())
                .unwrap_or(0) as f64;
            if df == 0.0 {
                continue;
            }

            // IDF with smoothing.
            let idf = ((self.total_docs as f64 - df + 0.5) / (df + 0.5) + 1.0).ln();
            if idf <= 0.0 {
                continue;
            }

            if let Some(posting) = self.postings.get(token) {
                for &doc_idx in posting {
                    let doc = &self.docs[doc_idx];
                    let tf = *doc.term_freqs.get(token).unwrap_or(&0) as f64;
                    let dl = doc.doc_len as f64;

                    let tf_score = (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * dl / self.avg_dl));
                    *scores.entry(doc_idx).or_insert(0.0) += idf * tf_score;
                }
            }
        }

        let mut results: Vec<(String, f64)> = scores
            .into_iter()
            .map(|(idx, score)| (self.docs[idx].id.clone(), score))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(max_results);
        results
    }
}

// ---------------------------------------------------------------------------
// Memory Fragment
// ---------------------------------------------------------------------------

/// Source of a memory fragment.
#[derive(Debug, Clone)]
pub enum FragmentSource {
    ConversationTurn,
    Observation,
    LongTermSection,
    DailyNote,
}

/// A piece of indexed memory.
#[derive(Debug, Clone)]
pub struct MemoryFragment {
    pub id: String,
    pub source: FragmentSource,
    pub content: String,
    pub timestamp: Option<String>,
    pub token_estimate: usize,
}

/// A scored query result.
#[derive(Debug, Clone)]
pub struct ScoredFragment {
    pub fragment: MemoryFragment,
    pub score: f64,
}

// ---------------------------------------------------------------------------
// SemanticIndex
// ---------------------------------------------------------------------------

struct IndexState {
    engine: BM25Engine,
    fragments: HashMap<String, MemoryFragment>,
    dirty: bool,
}

/// Public API for semantic memory retrieval.
pub struct SemanticIndex {
    workspace: PathBuf,
    store: Arc<ConversationStore>,
    state: RwLock<IndexState>,
}

impl SemanticIndex {
    /// Create a new semantic index (lazy — does not build until first query).
    pub fn new(workspace: &Path, store: Arc<ConversationStore>) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
            store,
            state: RwLock::new(IndexState {
                engine: BM25Engine::new(),
                fragments: HashMap::new(),
                dirty: true, // Will build on first query.
            }),
        }
    }

    /// Full rebuild: load from SQLite + scan memory files.
    pub fn rebuild(&self) {
        let mut engine = BM25Engine::new();
        let mut fragments = HashMap::new();

        // 1. Index assistant turns from SQLite.
        // Iterate sessions by channel then turns (no "get all" method needed).
        let sessions_by_channel = ["cli", "telegram", "whatsapp", "voice", "feishu", "email"];
        for channel in &sessions_by_channel {
            let sessions = self.store.list_sessions(channel, 100);
            for session in &sessions {
                let turns = self.store.get_session_turns(&session.id, 1000);
                for turn in &turns {
                    if turn.role == "assistant" || turn.role == "user" {
                        if let Some(ref content) = turn.content {
                            if content.len() > 10 {
                                let doc_id = format!("turn:{}", turn.id);
                                let frag = MemoryFragment {
                                    id: doc_id.clone(),
                                    source: FragmentSource::ConversationTurn,
                                    content: content.clone(),
                                    timestamp: Some(turn.created_at.clone()),
                                    token_estimate: content.len() / 4,
                                };
                                engine.add_doc(doc_id.clone(), content);
                                fragments.insert(doc_id, frag);
                            }
                        }
                    }
                }
            }
        }

        // 2. Index observations.
        let obs_dir = self.workspace.join("memory").join("observations");
        if obs_dir.is_dir() {
            if let Ok(entries) = fs::read_dir(&obs_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) == Some("md") {
                        if let Ok(content) = fs::read_to_string(&path) {
                            let filename = path
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("unknown");
                            let doc_id = format!("obs:{}", filename);
                            let frag = MemoryFragment {
                                id: doc_id.clone(),
                                source: FragmentSource::Observation,
                                content: content.clone(),
                                timestamp: None,
                                token_estimate: content.len() / 4,
                            };
                            engine.add_doc(doc_id.clone(), &content);
                            fragments.insert(doc_id, frag);
                        }
                    }
                }
            }
        }

        // 3. Index MEMORY.md sections.
        let memory_path = self.workspace.join("MEMORY.md");
        if memory_path.is_file() {
            if let Ok(content) = fs::read_to_string(&memory_path) {
                let sections = split_by_headings(&content);
                for (heading, section_content) in &sections {
                    if section_content.len() > 10 {
                        let doc_id = format!("mem:{}", heading);
                        let frag = MemoryFragment {
                            id: doc_id.clone(),
                            source: FragmentSource::LongTermSection,
                            content: section_content.clone(),
                            timestamp: None,
                            token_estimate: section_content.len() / 4,
                        };
                        engine.add_doc(doc_id.clone(), section_content);
                        fragments.insert(doc_id, frag);
                    }
                }
            }
        }

        // 4. Index daily notes.
        let notes_dir = self.workspace.join("memory").join("notes");
        if notes_dir.is_dir() {
            if let Ok(entries) = fs::read_dir(&notes_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) == Some("md") {
                        if let Ok(content) = fs::read_to_string(&path) {
                            let filename = path
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("unknown");
                            let doc_id = format!("daily:{}", filename);
                            let frag = MemoryFragment {
                                id: doc_id.clone(),
                                source: FragmentSource::DailyNote,
                                content: content.clone(),
                                timestamp: None,
                                token_estimate: content.len() / 4,
                            };
                            engine.add_doc(doc_id.clone(), &content);
                            fragments.insert(doc_id, frag);
                        }
                    }
                }
            }
        }

        // Swap in the new state.
        let mut state = self.state.write().unwrap();
        state.engine = engine;
        state.fragments = fragments;
        state.dirty = false;
    }

    /// Incrementally add a turn after an assistant response.
    pub fn add_turn(&self, turn_id: &str, content: &str) {
        if content.len() <= 10 {
            return;
        }
        let doc_id = format!("turn:{}", turn_id);
        let frag = MemoryFragment {
            id: doc_id.clone(),
            source: FragmentSource::ConversationTurn,
            content: content.to_string(),
            timestamp: None,
            token_estimate: content.len() / 4,
        };
        let mut state = self.state.write().unwrap();
        state.engine.add_doc(doc_id.clone(), content);
        state.fragments.insert(doc_id, frag);
    }

    /// Incrementally add an observation.
    pub fn add_observation(&self, path: &Path, content: &str) {
        let filename = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let doc_id = format!("obs:{}", filename);
        let frag = MemoryFragment {
            id: doc_id.clone(),
            source: FragmentSource::Observation,
            content: content.to_string(),
            timestamp: None,
            token_estimate: content.len() / 4,
        };
        let mut state = self.state.write().unwrap();
        state.engine.add_doc(doc_id.clone(), content);
        state.fragments.insert(doc_id, frag);
    }

    /// Mark the index as dirty (will rebuild on next query).
    pub fn mark_dirty(&self) {
        self.state.write().unwrap().dirty = true;
    }

    /// Query the index and return scored fragments.
    pub fn query(&self, text: &str, max_tokens: usize) -> Vec<ScoredFragment> {
        // Rebuild if dirty.
        if self.state.read().unwrap().dirty {
            self.rebuild();
        }

        let state = self.state.read().unwrap();
        let results = state.engine.query(text, 50);

        let mut scored = Vec::new();
        let mut total_tokens = 0;

        for (doc_id, score) in results {
            if let Some(frag) = state.fragments.get(&doc_id) {
                if total_tokens + frag.token_estimate > max_tokens {
                    continue;
                }
                total_tokens += frag.token_estimate;
                scored.push(ScoredFragment {
                    fragment: frag.clone(),
                    score,
                });
            }
        }

        scored
    }

    /// Get relevant context formatted for injection into the system prompt.
    pub fn get_relevant_context(&self, query: &str, max_tokens: usize) -> String {
        let scored = self.query(query, max_tokens);
        if scored.is_empty() {
            return String::new();
        }

        let mut parts = Vec::new();
        for sf in &scored {
            let source_label = match sf.fragment.source {
                FragmentSource::ConversationTurn => "conversation",
                FragmentSource::Observation => "observation",
                FragmentSource::LongTermSection => "memory",
                FragmentSource::DailyNote => "daily note",
            };
            parts.push(format!(
                "[{}] {:.2}: {}",
                source_label,
                sf.score,
                truncate_content(&sf.fragment.content, 500)
            ));
        }

        parts.join("\n\n")
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Split markdown by `##` headings.
fn split_by_headings(text: &str) -> Vec<(String, String)> {
    let mut sections = Vec::new();
    let mut current_heading = String::from("intro");
    let mut current_content = String::new();

    for line in text.lines() {
        if let Some(heading) = line.strip_prefix("## ") {
            if !current_content.trim().is_empty() {
                sections.push((current_heading, current_content.trim().to_string()));
            }
            current_heading = heading.trim().to_string();
            current_content = String::new();
        } else {
            current_content.push_str(line);
            current_content.push('\n');
        }
    }

    if !current_content.trim().is_empty() {
        sections.push((current_heading, current_content.trim().to_string()));
    }

    sections
}

/// Truncate content to approximately `max_chars` at a word boundary.
fn truncate_content(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        return s.to_string();
    }
    let truncated = &s[..max_chars];
    if let Some(last_space) = truncated.rfind(' ') {
        format!("{}...", &truncated[..last_space])
    } else {
        format!("{}...", truncated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("Hello world, this is a test!");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // Stop words should be filtered.
        assert!(!tokens.contains(&"this".to_string()));
        assert!(!tokens.contains(&"is".to_string()));
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_tokenize_empty() {
        let tokens = tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_bm25_basic_search() {
        let mut engine = BM25Engine::new();
        engine.add_doc("doc1".to_string(), "Rust is a systems programming language");
        engine.add_doc("doc2".to_string(), "Python is great for data science");
        engine.add_doc("doc3".to_string(), "Rust and Python are both popular");

        let results = engine.query("Rust programming", 10);
        assert!(!results.is_empty());
        // doc1 should rank highest (both "rust" and "programming" match).
        assert_eq!(results[0].0, "doc1");
    }

    #[test]
    fn test_bm25_no_match() {
        let mut engine = BM25Engine::new();
        engine.add_doc("doc1".to_string(), "Hello world");

        let results = engine.query("quantum physics", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_empty_index() {
        let engine = BM25Engine::new();
        let results = engine.query("anything", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_split_by_headings() {
        let text = "# Title\nIntro text\n## Section One\nContent one\n## Section Two\nContent two\n";
        let sections = split_by_headings(text);
        assert_eq!(sections.len(), 3);
        assert_eq!(sections[0].0, "intro");
        assert_eq!(sections[1].0, "Section One");
        assert_eq!(sections[2].0, "Section Two");
    }

    #[test]
    fn test_truncate_content() {
        let short = "Hello";
        assert_eq!(truncate_content(short, 100), "Hello");

        let long = "This is a longer string that needs truncation at a word boundary";
        let truncated = truncate_content(long, 30);
        assert!(truncated.ends_with("..."));
        assert!(truncated.len() <= 35); // 30 + "..."
    }

    #[test]
    fn test_semantic_index_in_memory() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = Arc::new(ConversationStore::new(&db_path).unwrap());

        // Add some turns to the store.
        let sid = store.ensure_session("cli", "direct");
        store.add_turn(&sid, "assistant", Some("The voice pipeline uses jack-voice for audio capture"), None, None, None, None, None, None, None, false, None);
        store.add_turn(&sid, "assistant", Some("SQLite is used for the conversation store"), None, None, None, None, None, None, None, false, None);

        let workspace = dir.path().to_path_buf();
        let index = SemanticIndex::new(&workspace, store);
        index.rebuild();

        let results = index.query("voice pipeline audio", 5000);
        assert!(!results.is_empty());
        assert!(results[0].fragment.content.contains("voice pipeline"));
    }

    #[test]
    fn test_incremental_add_turn() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = Arc::new(ConversationStore::new(&db_path).unwrap());

        let workspace = dir.path().to_path_buf();
        let index = SemanticIndex::new(&workspace, store);
        index.rebuild(); // Empty build.

        // Add incrementally.
        index.add_turn("turn-123", "Docker deployment uses containerization");

        let results = index.query("docker container", 5000);
        assert!(!results.is_empty());
        assert_eq!(results[0].fragment.id, "turn:turn-123");
    }

    #[test]
    fn test_get_relevant_context() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = Arc::new(ConversationStore::new(&db_path).unwrap());

        let workspace = dir.path().to_path_buf();
        let index = SemanticIndex::new(&workspace, store);
        index.rebuild();

        index.add_turn("t1", "Memory retrieval uses BM25 scoring algorithm");

        let ctx = index.get_relevant_context("BM25 scoring", 5000);
        assert!(!ctx.is_empty());
        assert!(ctx.contains("BM25"));
    }
}
