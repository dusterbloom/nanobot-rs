#![allow(dead_code)]
use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::{Path, PathBuf};

/// Persistent knowledge store backed by SQLite + FTS5.
///
/// Stores chunked documents with full-text search (BM25 ranking).
/// Used for million-token context: ingest large documents, search by keyword.
pub struct KnowledgeStore {
    conn: Connection,
    db_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct IngestResult {
    pub source_id: i64,
    pub chunks_created: usize,
    pub total_chars: usize,
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub source_name: String,
    pub chunk_idx: i64,
    pub snippet: String,
    pub rank: f64,
}

#[derive(Debug, Clone)]
pub struct SourceInfo {
    pub id: i64,
    pub name: String,
    pub path: Option<String>,
    pub total_chunks: i64,
    pub total_chars: i64,
    pub created_at: String,
}

#[derive(Debug, Clone)]
pub struct StoreStats {
    pub total_sources: i64,
    pub total_chunks: i64,
    pub total_chars: i64,
}

impl KnowledgeStore {
    /// Open or create the knowledge database at the specified path.
    pub fn open(db_path: &Path) -> Result<Self> {
        // Ensure parent directory exists
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .context("Failed to create knowledge store directory")?;
        }

        let conn = Connection::open(db_path).context("Failed to open knowledge database")?;

        // Enable WAL mode and mmap for performance
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "mmap_size", 268435456)?; // 256MB mmap

        // Initialize schema
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                path TEXT,
                total_chunks INTEGER NOT NULL DEFAULT 0,
                total_chars INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL REFERENCES sources(id),
                chunk_idx INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                content='chunks',
                content_rowid='id'
            );

            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
                INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
            END;
            "#,
        )
        .context("Failed to initialize knowledge store schema")?;

        Ok(Self {
            conn,
            db_path: db_path.to_path_buf(),
        })
    }

    /// Open the knowledge database at the default location (~/.nanobot/knowledge.db).
    pub fn open_default() -> Result<Self> {
        let home = dirs::home_dir().context("Failed to determine home directory")?;
        let db_path = home.join(".nanobot").join("knowledge.db");
        Self::open(&db_path)
    }

    /// Ingest a document into the knowledge store.
    ///
    /// Splits the text into overlapping chunks and stores them with FTS5 indexing.
    /// If a source with the same name exists, it will be replaced.
    pub fn ingest(
        &self,
        name: &str,
        path: Option<&str>,
        text: &str,
        chunk_size: usize,
        overlap: usize,
    ) -> Result<IngestResult> {
        // Delete existing source if present
        self.delete_source(name)?;

        // Split text into chunks
        let chunks = chunk_text(text, chunk_size, overlap);
        let total_chars = text.chars().count();

        // Insert source
        self.conn.execute(
            "INSERT INTO sources (name, path, total_chunks, total_chars) VALUES (?1, ?2, ?3, ?4)",
            params![name, path, chunks.len() as i64, total_chars as i64],
        )?;

        let source_id = self.conn.last_insert_rowid();

        // Insert chunks
        let mut stmt = self
            .conn
            .prepare("INSERT INTO chunks (source_id, chunk_idx, content) VALUES (?1, ?2, ?3)")?;

        for (idx, chunk_text) in chunks.iter().enumerate() {
            stmt.execute(params![source_id, idx as i64, chunk_text])?;
        }

        Ok(IngestResult {
            source_id,
            chunks_created: chunks.len(),
            total_chars,
        })
    }

    /// Search the knowledge store using FTS5 BM25 ranking.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT
                s.name,
                c.chunk_idx,
                snippet(chunks_fts, 0, '»', '«', '...', 32) as snippet,
                bm25(chunks_fts) as rank
            FROM chunks_fts
            JOIN chunks c ON chunks_fts.rowid = c.id
            JOIN sources s ON c.source_id = s.id
            WHERE chunks_fts MATCH ?1
            ORDER BY rank
            LIMIT ?2
            "#,
        )?;

        let hits = stmt
            .query_map(params![query, limit as i64], |row| {
                Ok(SearchHit {
                    source_name: row.get(0)?,
                    chunk_idx: row.get(1)?,
                    snippet: row.get(2)?,
                    rank: row.get(3)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(hits)
    }

    /// Get a specific chunk by source name and chunk index.
    pub fn get_chunk(&self, source_name: &str, chunk_idx: usize) -> Result<Option<String>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT c.content
            FROM chunks c
            JOIN sources s ON c.source_id = s.id
            WHERE s.name = ?1 AND c.chunk_idx = ?2
            "#,
        )?;

        let content = stmt
            .query_row(params![source_name, chunk_idx as i64], |row| {
                row.get::<_, String>(0)
            })
            .optional()?;

        Ok(content)
    }

    /// Get a range of chunks by source name and chunk indices [start_idx, end_idx).
    pub fn get_chunks_range(
        &self,
        source_name: &str,
        start_idx: usize,
        end_idx: usize,
    ) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT c.content
            FROM chunks c
            JOIN sources s ON c.source_id = s.id
            WHERE s.name = ?1 AND c.chunk_idx >= ?2 AND c.chunk_idx < ?3
            ORDER BY c.chunk_idx
            "#,
        )?;

        let chunks = stmt
            .query_map(
                params![source_name, start_idx as i64, end_idx as i64],
                |row| row.get::<_, String>(0),
            )?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(chunks)
    }

    /// List all ingested sources with metadata.
    pub fn list_sources(&self) -> Result<Vec<SourceInfo>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, path, total_chunks, total_chars, created_at FROM sources ORDER BY created_at DESC"
        )?;

        let sources = stmt
            .query_map([], |row| {
                Ok(SourceInfo {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    path: row.get(2)?,
                    total_chunks: row.get(3)?,
                    total_chars: row.get(4)?,
                    created_at: row.get(5)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(sources)
    }

    /// Delete a source and all its chunks.
    ///
    /// Returns true if a source was deleted, false if not found.
    pub fn delete_source(&self, name: &str) -> Result<bool> {
        // Get source_id first
        let source_id: Option<i64> = self
            .conn
            .query_row(
                "SELECT id FROM sources WHERE name = ?1",
                params![name],
                |row| row.get(0),
            )
            .optional()?;

        if let Some(id) = source_id {
            // Delete chunks (triggers will handle FTS cleanup)
            self.conn
                .execute("DELETE FROM chunks WHERE source_id = ?1", params![id])?;

            // Delete source
            self.conn
                .execute("DELETE FROM sources WHERE id = ?1", params![id])?;

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get statistics about the knowledge store.
    pub fn stats(&self) -> Result<StoreStats> {
        let total_sources: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM sources", [], |row| row.get(0))?;

        let total_chunks: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;

        let total_chars: i64 = self.conn.query_row(
            "SELECT COALESCE(SUM(total_chars), 0) FROM sources",
            [],
            |row| row.get(0),
        )?;

        Ok(StoreStats {
            total_sources,
            total_chunks,
            total_chars,
        })
    }

    /// Get the database path.
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }
}

/// Split text into overlapping chunks, avoiding mid-line splits.
fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![];
    }

    let mut chunks = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let total_chars = chars.len();

    if total_chars <= chunk_size {
        return vec![text.to_string()];
    }

    let step = chunk_size.saturating_sub(overlap);
    if step == 0 {
        // Overlap >= chunk_size, just return single chunk
        return vec![text.to_string()];
    }

    let mut start = 0;
    while start < total_chars {
        let mut end = (start + chunk_size).min(total_chars);

        // If not at the end, try to extend to the next newline
        if end < total_chars {
            // Look ahead for newline within a reasonable distance
            let search_limit = (end + 100).min(total_chars);
            if let Some(newline_pos) = chars[end..search_limit].iter().position(|&c| c == '\n') {
                end = end + newline_pos + 1; // Include the newline
            }
        }

        let chunk: String = chars[start..end].iter().collect();
        chunks.push(chunk);

        // Move to next chunk with overlap
        start += step;

        // Prevent infinite loop if step is too small
        if start <= chunks.len() * step - step {
            break;
        }
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_open_creates_db() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let store = KnowledgeStore::open(&db_path).unwrap();
        assert!(db_path.exists());
        assert_eq!(store.db_path(), &db_path);
    }

    #[test]
    fn test_ingest_basic() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = KnowledgeStore::open(&db_path).unwrap();

        let text = "Hello, this is a test document.";
        let result = store
            .ingest("test_doc", Some("/path/to/doc"), text, 4096, 256)
            .unwrap();

        assert_eq!(result.chunks_created, 1);
        assert_eq!(result.total_chars, text.chars().count());

        // Verify we can retrieve it
        let chunk = store.get_chunk("test_doc", 0).unwrap();
        assert_eq!(chunk, Some(text.to_string()));
    }

    #[test]
    fn test_ingest_chunking() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = KnowledgeStore::open(&db_path).unwrap();

        // Create 10K char text
        let text = "a".repeat(10000);
        let result = store.ingest("large_doc", None, &text, 1000, 100).unwrap();

        // With 1000 char chunks and 100 overlap, step = 900
        // Expected chunks: ceil(10000 / 900) = 12 chunks
        assert!(result.chunks_created >= 11 && result.chunks_created <= 12);
        assert_eq!(result.total_chars, 10000);
    }

    #[test]
    fn test_ingest_overlap_line_boundary() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = KnowledgeStore::open(&db_path).unwrap();

        let text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n";
        let result = store.ingest("lines_doc", None, text, 15, 5).unwrap();

        // Should create multiple chunks
        assert!(result.chunks_created > 1);

        // Verify all chunks can be retrieved
        for i in 0..result.chunks_created {
            let chunk = store.get_chunk("lines_doc", i).unwrap();
            assert!(chunk.is_some());
        }

        // Verify we can search and find content
        let results = store.search("Line", 10).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_search_bm25() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = KnowledgeStore::open(&db_path).unwrap();

        store
            .ingest(
                "doc1",
                None,
                "The quick brown fox jumps over the lazy dog.",
                4096,
                256,
            )
            .unwrap();
        store
            .ingest(
                "doc2",
                None,
                "A completely different topic about databases.",
                4096,
                256,
            )
            .unwrap();

        let results = store.search("fox", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source_name, "doc1");
        assert!(results[0].snippet.contains("fox"));
    }

    #[test]
    fn test_search_ranking() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = KnowledgeStore::open(&db_path).unwrap();

        // Doc with term appearing once
        store
            .ingest("doc1", None, "The database system is important.", 4096, 256)
            .unwrap();

        // Doc with term appearing multiple times
        store
            .ingest(
                "doc2",
                None,
                "Database, database, database everywhere! Database systems.",
                4096,
                256,
            )
            .unwrap();

        let results = store.search("database", 10).unwrap();
        assert_eq!(results.len(), 2);

        // doc2 should rank higher (lower BM25 score = better rank)
        assert_eq!(results[0].source_name, "doc2");
    }

    #[test]
    fn test_get_chunk() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = KnowledgeStore::open(&db_path).unwrap();

        let text = "Chunk content here.";
        store.ingest("test_doc", None, text, 4096, 256).unwrap();

        let chunk = store.get_chunk("test_doc", 0).unwrap();
        assert_eq!(chunk, Some(text.to_string()));

        let no_chunk = store.get_chunk("test_doc", 999).unwrap();
        assert_eq!(no_chunk, None);

        let no_source = store.get_chunk("nonexistent", 0).unwrap();
        assert_eq!(no_source, None);
    }

    #[test]
    fn test_get_chunks_range() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = KnowledgeStore::open(&db_path).unwrap();

        // Create multi-chunk document
        let text = "a".repeat(5000);
        store.ingest("large_doc", None, &text, 1000, 100).unwrap();

        let chunks = store.get_chunks_range("large_doc", 1, 4).unwrap();
        assert_eq!(chunks.len(), 3); // Indices 1, 2, 3
    }

    #[test]
    fn test_list_sources() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = KnowledgeStore::open(&db_path).unwrap();

        store
            .ingest("doc1", Some("/path/1"), "Content 1", 4096, 256)
            .unwrap();
        store.ingest("doc2", None, "Content 2", 4096, 256).unwrap();
        store
            .ingest("doc3", Some("/path/3"), "Content 3", 4096, 256)
            .unwrap();

        let sources = store.list_sources().unwrap();
        assert_eq!(sources.len(), 3);

        // Verify all sources are present
        let names: Vec<_> = sources.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"doc1"));
        assert!(names.contains(&"doc2"));
        assert!(names.contains(&"doc3"));

        // Verify paths are correct
        let doc1 = sources.iter().find(|s| s.name == "doc1").unwrap();
        assert_eq!(doc1.path, Some("/path/1".to_string()));

        let doc2 = sources.iter().find(|s| s.name == "doc2").unwrap();
        assert_eq!(doc2.path, None);
    }

    #[test]
    fn test_delete_source() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = KnowledgeStore::open(&db_path).unwrap();

        store
            .ingest("doc1", None, "Content to be deleted", 4096, 256)
            .unwrap();

        let deleted = store.delete_source("doc1").unwrap();
        assert!(deleted);

        // Verify search returns nothing
        let results = store.search("deleted", 10).unwrap();
        assert_eq!(results.len(), 0);

        // Verify source is gone
        let sources = store.list_sources().unwrap();
        assert_eq!(sources.len(), 0);

        // Delete non-existent should return false
        let not_deleted = store.delete_source("nonexistent").unwrap();
        assert!(!not_deleted);
    }

    #[test]
    fn test_stats() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = KnowledgeStore::open(&db_path).unwrap();

        let stats = store.stats().unwrap();
        assert_eq!(stats.total_sources, 0);
        assert_eq!(stats.total_chunks, 0);
        assert_eq!(stats.total_chars, 0);

        store.ingest("doc1", None, "Short", 4096, 256).unwrap();
        store
            .ingest("doc2", None, "a".repeat(5000).as_str(), 1000, 100)
            .unwrap();

        let stats = store.stats().unwrap();
        assert_eq!(stats.total_sources, 2);
        assert!(stats.total_chunks > 1);
        assert_eq!(stats.total_chars, 5 + 5000);
    }

    #[test]
    fn test_ingest_duplicate_source_replaces() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = KnowledgeStore::open(&db_path).unwrap();

        store
            .ingest("doc", None, "Original content", 4096, 256)
            .unwrap();
        store
            .ingest("doc", None, "Replaced content", 4096, 256)
            .unwrap();

        let sources = store.list_sources().unwrap();
        assert_eq!(sources.len(), 1);

        let chunk = store.get_chunk("doc", 0).unwrap();
        assert_eq!(chunk, Some("Replaced content".to_string()));

        // Verify search only finds new content
        let results = store.search("Original", 10).unwrap();
        assert_eq!(results.len(), 0);

        let results = store.search("Replaced", 10).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_search_empty_db() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = KnowledgeStore::open(&db_path).unwrap();

        let results = store.search("anything", 10).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_chunk_text_basic() {
        let text = "Hello world";
        let chunks = chunk_text(text, 100, 10);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Hello world");
    }

    #[test]
    fn test_chunk_text_with_overlap() {
        let text = "a".repeat(1000);
        let chunks = chunk_text(&text, 100, 20);

        // step = 100 - 20 = 80
        // Expected: ceil(1000 / 80) = 13 chunks
        assert!(chunks.len() >= 12 && chunks.len() <= 13);

        // Verify overlap
        for i in 0..chunks.len() - 1 {
            let current_end = &chunks[i][chunks[i].len() - 20..];
            let next_start = &chunks[i + 1][..20];
            // Should have some overlap
            assert!(current_end == next_start || chunks[i].len() < 20);
        }
    }

    #[test]
    fn test_chunk_text_empty() {
        let chunks = chunk_text("", 100, 10);
        assert_eq!(chunks.len(), 0);
    }
}
