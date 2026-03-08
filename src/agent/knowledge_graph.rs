#![allow(dead_code)]
//! Knowledge graph: entity/relation store backed by petgraph.
//!
//! Stores entities (people, tools, concepts) and their relationships.
//! Persisted as JSON. Supports temporal queries via entity timestamps.
//! Used by the reflector to accumulate structured knowledge, and by
//! proactive recall to inject relevant context before each turn.

#[cfg(feature = "knowledge-graph")]
use petgraph::graph::{DiGraph, NodeIndex};
#[cfg(feature = "knowledge-graph")]
use petgraph::visit::EdgeRef;
#[cfg(feature = "knowledge-graph")]
use petgraph::Direction;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// An entity in the knowledge graph (person, tool, concept, project, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub name: String,
    pub kind: String, // "person", "tool", "concept", "project", etc.
    pub summary: String,
    pub created_at: String,
    pub updated_at: String,
}

/// A directed relation between two entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    pub label: String, // "uses", "prefers", "created", "knows", etc.
    pub context: String,
    pub created_at: String,
}

/// Serializable graph format for JSON persistence.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct GraphData {
    entities: Vec<Entity>,
    relations: Vec<(usize, usize, Relation)>, // (from_idx, to_idx, relation)
}

/// Knowledge graph backed by petgraph DiGraph.
///
/// When the `knowledge-graph` feature is disabled, all methods are no-ops
/// that return empty results or Ok(()).
pub struct KnowledgeGraph {
    #[cfg(feature = "knowledge-graph")]
    graph: DiGraph<Entity, Relation>,
    #[cfg(feature = "knowledge-graph")]
    name_index: HashMap<String, NodeIndex>,
    path: PathBuf,
}

impl KnowledgeGraph {
    /// Open or create a knowledge graph at the specified JSON path.
    pub fn open(path: &Path) -> Result<Self> {
        #[cfg(feature = "knowledge-graph")]
        {
            let (graph, name_index) = if path.exists() {
                let data =
                    std::fs::read_to_string(path).context("Failed to read knowledge graph")?;
                let gd: GraphData =
                    serde_json::from_str(&data).context("Failed to parse knowledge graph")?;
                Self::from_graph_data(gd)
            } else {
                (DiGraph::new(), HashMap::new())
            };
            Ok(Self {
                graph,
                name_index,
                path: path.to_path_buf(),
            })
        }

        #[cfg(not(feature = "knowledge-graph"))]
        {
            Ok(Self {
                path: path.to_path_buf(),
            })
        }
    }

    /// Open the default knowledge graph at `~/.nanobot/knowledge_graph.json`.
    pub fn open_default() -> Result<Self> {
        let home = dirs::home_dir().context("Failed to determine home directory")?;
        let path = home.join(".nanobot").join("knowledge_graph.json");
        Self::open(&path)
    }

    /// Persist the graph to disk as JSON.
    pub fn save(&self) -> Result<()> {
        #[cfg(feature = "knowledge-graph")]
        {
            if let Some(parent) = self.path.parent() {
                std::fs::create_dir_all(parent)
                    .context("Failed to create knowledge graph directory")?;
            }
            let gd = self.to_graph_data();
            let json =
                serde_json::to_string_pretty(&gd).context("Failed to serialize knowledge graph")?;
            std::fs::write(&self.path, json).context("Failed to write knowledge graph")?;
        }

        #[cfg(not(feature = "knowledge-graph"))]
        let _ = &self.path;

        Ok(())
    }

    /// Add or update an entity. Returns true if the entity was newly created.
    #[cfg(feature = "knowledge-graph")]
    pub fn upsert_entity(&mut self, name: &str, kind: &str, summary: &str) -> bool {
        let now = chrono::Utc::now().to_rfc3339();
        if let Some(&idx) = self.name_index.get(name) {
            let entity = &mut self.graph[idx];
            entity.kind = kind.to_string();
            entity.summary = summary.to_string();
            entity.updated_at = now;
            false
        } else {
            let entity = Entity {
                name: name.to_string(),
                kind: kind.to_string(),
                summary: summary.to_string(),
                created_at: now.clone(),
                updated_at: now,
            };
            let idx = self.graph.add_node(entity);
            self.name_index.insert(name.to_string(), idx);
            true
        }
    }

    #[cfg(not(feature = "knowledge-graph"))]
    pub fn upsert_entity(&mut self, _name: &str, _kind: &str, _summary: &str) -> bool {
        false
    }

    /// Add a directed relation between two entities (creates entities if missing).
    #[cfg(feature = "knowledge-graph")]
    pub fn add_relation(&mut self, from: &str, label: &str, to: &str, context: &str) {
        // Ensure both entities exist.
        if !self.name_index.contains_key(from) {
            self.upsert_entity(from, "unknown", "");
        }
        if !self.name_index.contains_key(to) {
            self.upsert_entity(to, "unknown", "");
        }

        let from_idx = self.name_index[from];
        let to_idx = self.name_index[to];
        let now = chrono::Utc::now().to_rfc3339();

        // Check for duplicate relation.
        let existing = self
            .graph
            .edges_connecting(from_idx, to_idx)
            .any(|e| e.weight().label == label);
        if !existing {
            self.graph.add_edge(
                from_idx,
                to_idx,
                Relation {
                    label: label.to_string(),
                    context: context.to_string(),
                    created_at: now,
                },
            );
        }
    }

    #[cfg(not(feature = "knowledge-graph"))]
    pub fn add_relation(&mut self, _from: &str, _label: &str, _to: &str, _context: &str) {}

    /// Get an entity by name.
    #[cfg(feature = "knowledge-graph")]
    pub fn get_entity(&self, name: &str) -> Option<&Entity> {
        self.name_index.get(name).map(|&idx| &self.graph[idx])
    }

    #[cfg(not(feature = "knowledge-graph"))]
    pub fn get_entity(&self, _name: &str) -> Option<&Entity> {
        None
    }

    /// Get all relations from a given entity (outgoing edges).
    #[cfg(feature = "knowledge-graph")]
    pub fn relations_from(&self, name: &str) -> Vec<(&str, &Relation)> {
        let Some(&idx) = self.name_index.get(name) else {
            return vec![];
        };
        self.graph
            .edges_directed(idx, Direction::Outgoing)
            .map(|e| (self.graph[e.target()].name.as_str(), e.weight()))
            .collect()
    }

    #[cfg(not(feature = "knowledge-graph"))]
    pub fn relations_from(&self, _name: &str) -> Vec<(&str, &Relation)> {
        vec![]
    }

    /// Get all relations to a given entity (incoming edges).
    #[cfg(feature = "knowledge-graph")]
    pub fn relations_to(&self, name: &str) -> Vec<(&str, &Relation)> {
        let Some(&idx) = self.name_index.get(name) else {
            return vec![];
        };
        self.graph
            .edges_directed(idx, Direction::Incoming)
            .map(|e| (self.graph[e.source()].name.as_str(), e.weight()))
            .collect()
    }

    #[cfg(not(feature = "knowledge-graph"))]
    pub fn relations_to(&self, _name: &str) -> Vec<(&str, &Relation)> {
        vec![]
    }

    /// Find entities matching a query (case-insensitive substring on name or summary).
    #[cfg(feature = "knowledge-graph")]
    pub fn search_entities(&self, query: &str) -> Vec<&Entity> {
        let lower = query.to_lowercase();
        self.graph
            .node_weights()
            .filter(|e| {
                e.name.to_lowercase().contains(&lower) || e.summary.to_lowercase().contains(&lower)
            })
            .collect()
    }

    #[cfg(not(feature = "knowledge-graph"))]
    pub fn search_entities(&self, _query: &str) -> Vec<&Entity> {
        vec![]
    }

    /// Total number of entities.
    #[cfg(feature = "knowledge-graph")]
    pub fn entity_count(&self) -> usize {
        self.graph.node_count()
    }

    #[cfg(not(feature = "knowledge-graph"))]
    pub fn entity_count(&self) -> usize {
        0
    }

    /// Total number of relations.
    #[cfg(feature = "knowledge-graph")]
    pub fn relation_count(&self) -> usize {
        self.graph.edge_count()
    }

    #[cfg(not(feature = "knowledge-graph"))]
    pub fn relation_count(&self) -> usize {
        0
    }

    /// Export a human-readable summary for LLM context injection.
    #[cfg(feature = "knowledge-graph")]
    pub fn export_context(&self, max_entities: usize) -> String {
        let mut lines = Vec::new();
        let mut count = 0;
        for entity in self.graph.node_weights() {
            if count >= max_entities {
                break;
            }
            let mut line = format!("- {} ({})", entity.name, entity.kind);
            if !entity.summary.is_empty() {
                line.push_str(&format!(": {}", entity.summary));
            }
            // Add relations.
            let rels = self.relations_from(&entity.name);
            for (target, rel) in &rels {
                line.push_str(&format!(" → {} {}", rel.label, target));
            }
            lines.push(line);
            count += 1;
        }
        lines.join("\n")
    }

    #[cfg(not(feature = "knowledge-graph"))]
    pub fn export_context(&self, _max_entities: usize) -> String {
        String::new()
    }

    // --- Internal helpers ---

    #[cfg(feature = "knowledge-graph")]
    fn from_graph_data(gd: GraphData) -> (DiGraph<Entity, Relation>, HashMap<String, NodeIndex>) {
        let mut graph = DiGraph::new();
        let mut name_index = HashMap::new();
        let mut idx_map: Vec<NodeIndex> = Vec::new();

        for entity in gd.entities {
            let name = entity.name.clone();
            let idx = graph.add_node(entity);
            name_index.insert(name, idx);
            idx_map.push(idx);
        }

        for (from, to, rel) in gd.relations {
            if from < idx_map.len() && to < idx_map.len() {
                graph.add_edge(idx_map[from], idx_map[to], rel);
            }
        }

        (graph, name_index)
    }

    #[cfg(feature = "knowledge-graph")]
    fn to_graph_data(&self) -> GraphData {
        // Build stable index mapping: NodeIndex -> sequential index.
        let mut node_to_idx: HashMap<NodeIndex, usize> = HashMap::new();
        let mut entities = Vec::new();

        for (i, idx) in self.graph.node_indices().enumerate() {
            node_to_idx.insert(idx, i);
            entities.push(self.graph[idx].clone());
        }

        let mut relations = Vec::new();
        for edge in self.graph.edge_indices() {
            let (source, target) = self.graph.edge_endpoints(edge).unwrap();
            if let (Some(&from), Some(&to)) = (node_to_idx.get(&source), node_to_idx.get(&target)) {
                relations.push((from, to, self.graph[edge].clone()));
            }
        }

        GraphData {
            entities,
            relations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_graph() -> (TempDir, KnowledgeGraph) {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test_kg.json");
        let graph = KnowledgeGraph::open(&path).unwrap();
        (tmp, graph)
    }

    #[cfg(feature = "knowledge-graph")]
    #[test]
    fn test_upsert_entity_creates_new() {
        let (_tmp, mut graph) = make_graph();
        let created = graph.upsert_entity("Rust", "language", "Systems programming language");
        assert!(created);
        assert_eq!(graph.entity_count(), 1);
        let entity = graph.get_entity("Rust").unwrap();
        assert_eq!(entity.kind, "language");
    }

    #[cfg(feature = "knowledge-graph")]
    #[test]
    fn test_upsert_entity_updates_existing() {
        let (_tmp, mut graph) = make_graph();
        graph.upsert_entity("Rust", "language", "Old summary");
        let created = graph.upsert_entity("Rust", "language", "New summary");
        assert!(!created);
        assert_eq!(graph.entity_count(), 1);
        assert_eq!(graph.get_entity("Rust").unwrap().summary, "New summary");
    }

    #[cfg(feature = "knowledge-graph")]
    #[test]
    fn test_add_relation() {
        let (_tmp, mut graph) = make_graph();
        graph.upsert_entity("User", "person", "The user");
        graph.upsert_entity("Rust", "language", "Systems lang");
        graph.add_relation("User", "prefers", "Rust", "mentioned in session");
        assert_eq!(graph.relation_count(), 1);
        let rels = graph.relations_from("User");
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].0, "Rust");
        assert_eq!(rels[0].1.label, "prefers");
    }

    #[cfg(feature = "knowledge-graph")]
    #[test]
    fn test_add_relation_creates_missing_entities() {
        let (_tmp, mut graph) = make_graph();
        graph.add_relation("Alice", "knows", "Bob", "met at conference");
        assert_eq!(graph.entity_count(), 2);
        assert_eq!(graph.relation_count(), 1);
        assert_eq!(graph.get_entity("Alice").unwrap().kind, "unknown");
    }

    #[cfg(feature = "knowledge-graph")]
    #[test]
    fn test_duplicate_relation_not_added() {
        let (_tmp, mut graph) = make_graph();
        graph.add_relation("A", "likes", "B", "context1");
        graph.add_relation("A", "likes", "B", "context2");
        assert_eq!(graph.relation_count(), 1);
    }

    #[cfg(feature = "knowledge-graph")]
    #[test]
    fn test_different_labels_both_added() {
        let (_tmp, mut graph) = make_graph();
        graph.add_relation("A", "likes", "B", "");
        graph.add_relation("A", "works-with", "B", "");
        assert_eq!(graph.relation_count(), 2);
    }

    #[cfg(feature = "knowledge-graph")]
    #[test]
    fn test_relations_to() {
        let (_tmp, mut graph) = make_graph();
        graph.add_relation("A", "manages", "B", "");
        graph.add_relation("C", "supports", "B", "");
        let incoming = graph.relations_to("B");
        assert_eq!(incoming.len(), 2);
    }

    #[cfg(feature = "knowledge-graph")]
    #[test]
    fn test_search_entities() {
        let (_tmp, mut graph) = make_graph();
        graph.upsert_entity("Rust", "language", "Systems programming");
        graph.upsert_entity("Python", "language", "Scripting language");
        graph.upsert_entity("nanobot", "project", "AI assistant in Rust");
        let results = graph.search_entities("rust");
        assert_eq!(results.len(), 2); // "Rust" entity + "nanobot" (summary contains "Rust")
    }

    #[cfg(feature = "knowledge-graph")]
    #[test]
    fn test_save_and_reload() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("persist.json");

        // Create and save.
        {
            let mut graph = KnowledgeGraph::open(&path).unwrap();
            graph.upsert_entity("Rust", "language", "Fast and safe");
            graph.upsert_entity("User", "person", "The user");
            graph.add_relation("User", "prefers", "Rust", "stated explicitly");
            graph.save().unwrap();
        }

        // Reload and verify.
        {
            let graph = KnowledgeGraph::open(&path).unwrap();
            assert_eq!(graph.entity_count(), 2);
            assert_eq!(graph.relation_count(), 1);
            let entity = graph.get_entity("Rust").unwrap();
            assert_eq!(entity.summary, "Fast and safe");
            let rels = graph.relations_from("User");
            assert_eq!(rels.len(), 1);
            assert_eq!(rels[0].1.label, "prefers");
        }
    }

    #[cfg(feature = "knowledge-graph")]
    #[test]
    fn test_export_context() {
        let (_tmp, mut graph) = make_graph();
        graph.upsert_entity("Rust", "language", "Systems lang");
        graph.upsert_entity("User", "person", "The user");
        graph.add_relation("User", "prefers", "Rust", "");
        let ctx = graph.export_context(10);
        assert!(ctx.contains("Rust"));
        assert!(ctx.contains("User"));
        assert!(ctx.contains("prefers"));
    }

    // Tests that work without the knowledge-graph feature.
    #[test]
    fn test_open_nonexistent_path() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("new_kg.json");
        let graph = KnowledgeGraph::open(&path).unwrap();
        assert_eq!(graph.entity_count(), 0);
        assert_eq!(graph.relation_count(), 0);
    }

    #[test]
    fn test_save_empty_graph() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("empty.json");
        let graph = KnowledgeGraph::open(&path).unwrap();
        graph.save().unwrap();
    }

    #[test]
    fn test_search_empty_graph() {
        let (_tmp, graph) = make_graph();
        let results = graph.search_entities("anything");
        assert!(results.is_empty());
    }

    #[test]
    fn test_relations_from_empty() {
        let (_tmp, graph) = make_graph();
        assert!(graph.relations_from("nobody").is_empty());
    }
}
