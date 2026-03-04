#![allow(dead_code)]
//! Thin wrapper around fastembed for local ONNX text embeddings.
//!
//! Single responsibility: text → f32 vector. Lazy-loads the model on first use.
//! Uses AllMiniLML6V2 (384 dimensions, ~5ms/sentence on Apple Silicon).

#[cfg(feature = "semantic")]
use anyhow::{Context, Result};
#[cfg(feature = "semantic")]
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
#[cfg(feature = "semantic")]
use std::sync::Mutex;
#[cfg(feature = "semantic")]
use tracing::{debug, info};

/// Embedding vector type alias.
pub type Embedding = Vec<f32>;

/// Embedding dimensions for AllMiniLML6V2.
pub const EMBEDDING_DIM: usize = 384;

/// Global singleton — model loads once on first embed call.
/// Uses Mutex<Option<>> because embed() requires &mut self and OnceLock::get_or_try_init is unstable.
#[cfg(feature = "semantic")]
static MODEL: once_cell::sync::Lazy<Mutex<TextEmbedding>> = once_cell::sync::Lazy::new(|| {
    info!("Loading embedding model (AllMiniLML6V2)...");
    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
    )
    .expect("Failed to initialize fastembed model");
    info!("Embedding model loaded ({}d)", EMBEDDING_DIM);
    Mutex::new(model)
});

/// Embed a single text string. Returns a vector of EMBEDDING_DIM floats.
#[cfg(feature = "semantic")]
pub fn embed_one(text: &str) -> Result<Embedding> {
    let mut model = MODEL.lock().map_err(|e| anyhow::anyhow!("Model lock poisoned: {}", e))?;
    let mut results = model
        .embed(vec![text], None)
        .context("Embedding failed")?;
    debug!("Embedded 1 text ({} chars)", text.len());
    results
        .pop()
        .ok_or_else(|| anyhow::anyhow!("Empty embedding result"))
}

/// Embed a batch of texts. Returns one vector per input.
#[cfg(feature = "semantic")]
pub fn embed_batch(texts: &[&str]) -> Result<Vec<Embedding>> {
    if texts.is_empty() {
        return Ok(vec![]);
    }
    let mut model = MODEL.lock().map_err(|e| anyhow::anyhow!("Model lock poisoned: {}", e))?;
    let results = model
        .embed(texts.to_vec(), None)
        .context("Batch embedding failed")?;
    debug!("Embedded {} texts", texts.len());
    Ok(results)
}

/// Stub when semantic feature is disabled.
#[cfg(not(feature = "semantic"))]
pub fn embed_one(_text: &str) -> anyhow::Result<Embedding> {
    Err(anyhow::anyhow!(
        "Semantic search requires the 'semantic' feature flag"
    ))
}

/// Stub when semantic feature is disabled.
#[cfg(not(feature = "semantic"))]
pub fn embed_batch(_texts: &[&str]) -> anyhow::Result<Vec<Embedding>> {
    Err(anyhow::anyhow!(
        "Semantic search requires the 'semantic' feature flag"
    ))
}

/// Cosine similarity between two vectors (pure math, no feature gate).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn test_cosine_similarity_mismatched_len() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_scaled() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 4.0, 6.0]; // same direction, 2x scale
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[cfg(feature = "semantic")]
    #[test]
    fn test_embed_one_returns_correct_dim() {
        let embedding = embed_one("Hello world").unwrap();
        assert_eq!(embedding.len(), EMBEDDING_DIM);
    }

    #[cfg(feature = "semantic")]
    #[test]
    fn test_embed_batch_returns_correct_count() {
        let texts = &["Hello", "World", "Test"];
        let embeddings = embed_batch(texts).unwrap();
        assert_eq!(embeddings.len(), 3);
        for e in &embeddings {
            assert_eq!(e.len(), EMBEDDING_DIM);
        }
    }

    #[cfg(feature = "semantic")]
    #[test]
    fn test_semantic_similarity() {
        let a = embed_one("The cat sat on the mat").unwrap();
        let b = embed_one("A feline rested on the rug").unwrap();
        let c = embed_one("Database connection pooling strategies").unwrap();
        let sim_ab = cosine_similarity(&a, &b);
        let sim_ac = cosine_similarity(&a, &c);
        // Similar sentences should have higher similarity than unrelated ones
        assert!(
            sim_ab > sim_ac,
            "Similar sentences ({:.3}) should score higher than unrelated ({:.3})",
            sim_ab,
            sim_ac
        );
    }
}
