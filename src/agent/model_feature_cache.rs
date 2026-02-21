//! Per-model feature support cache.
//!
//! Tracks which models have rejected specific API fields at runtime.
//! When a model returns "does not support X configuration", we record that
//! and omit the field on subsequent calls — avoiding repeated failures.
//!
//! Cache persists to `~/.nanobot/cache/model_feature_cache.json`.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Mutex;

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

/// In-memory cache: model_id -> set of unsupported feature names.
static CACHE: Lazy<Mutex<HashMap<String, HashSet<String>>>> =
    Lazy::new(|| Mutex::new(load_from_disk_sync()));

/// Serializable form written to disk.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct CacheFile {
    /// model_id -> list of unsupported feature names.
    unsupported: HashMap<String, Vec<String>>,
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Returns `true` if `feature` is known to be unsupported by `model`.
///
/// `feature` is a raw field name, e.g. `"reasoning"`.
pub fn is_feature_unsupported(model: &str, feature: &str) -> bool {
    let guard = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    guard
        .get(model)
        .map(|set| set.contains(feature))
        .unwrap_or(false)
}

/// Record that `model` does not support `feature`.
///
/// Idempotent. Persists the updated cache to disk (best-effort).
pub fn mark_feature_unsupported(model: &str, feature: &str) {
    {
        let mut guard = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        guard
            .entry(model.to_string())
            .or_default()
            .insert(feature.to_string());
        debug!(
            "model_feature_cache: {} does not support '{}'",
            model, feature
        );
    }
    save_to_disk_sync();
}

// ── Disk helpers ─────────────────────────────────────────────────────────────

fn cache_path() -> Option<PathBuf> {
    dirs::home_dir()
        .map(|h| h.join(".nanobot").join("cache").join("model_feature_cache.json"))
}

fn load_from_disk_sync() -> HashMap<String, HashSet<String>> {
    let Some(path) = cache_path() else {
        return HashMap::new();
    };

    let data = match std::fs::read_to_string(&path) {
        Ok(d) => d,
        Err(_) => return HashMap::new(),
    };

    let cf: CacheFile = match serde_json::from_str(&data) {
        Ok(v) => v,
        Err(e) => {
            warn!("model_feature_cache: failed to parse cache file: {}", e);
            return HashMap::new();
        }
    };

    cf.unsupported
        .into_iter()
        .map(|(k, v)| (k, v.into_iter().collect()))
        .collect()
}

fn save_to_disk_sync() {
    let Some(path) = cache_path() else {
        return;
    };

    let snapshot: HashMap<String, Vec<String>> = {
        let guard = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        guard
            .iter()
            .map(|(k, v)| {
                let mut sorted: Vec<String> = v.iter().cloned().collect();
                sorted.sort();
                (k.clone(), sorted)
            })
            .collect()
    };

    let cf = CacheFile { unsupported: snapshot };

    let json = match serde_json::to_string_pretty(&cf) {
        Ok(j) => j,
        Err(e) => {
            warn!("model_feature_cache: serialization failed: {}", e);
            return;
        }
    };

    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(e) = std::fs::write(&path, json) {
        warn!("model_feature_cache: write failed: {}", e);
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Unique suffix so parallel test runs don't share cache files.
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_cache_file() -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("nanobot_feature_cache_test_{}.json", n))
    }

    /// Build an isolated in-memory cache (does not touch the global static).
    fn make_cache() -> HashMap<String, HashSet<String>> {
        HashMap::new()
    }

    fn insert(cache: &mut HashMap<String, HashSet<String>>, model: &str, feature: &str) {
        cache
            .entry(model.to_string())
            .or_default()
            .insert(feature.to_string());
    }

    fn contains(cache: &HashMap<String, HashSet<String>>, model: &str, feature: &str) -> bool {
        cache
            .get(model)
            .map(|s| s.contains(feature))
            .unwrap_or(false)
    }

    // ── unit tests against in-memory helpers ─────────────────────────────────

    #[test]
    fn test_unsupported_after_mark() {
        let mut cache = make_cache();
        assert!(!contains(&cache, "nemotron-8b", "reasoning"));
        insert(&mut cache, "nemotron-8b", "reasoning");
        assert!(contains(&cache, "nemotron-8b", "reasoning"));
    }

    #[test]
    fn test_different_model_not_affected() {
        let mut cache = make_cache();
        insert(&mut cache, "nemotron-8b", "reasoning");
        assert!(!contains(&cache, "gpt-4o", "reasoning"));
    }

    #[test]
    fn test_different_feature_not_affected() {
        let mut cache = make_cache();
        insert(&mut cache, "nemotron-8b", "reasoning");
        assert!(!contains(&cache, "nemotron-8b", "temperature"));
    }

    #[test]
    fn test_idempotent_insert() {
        let mut cache = make_cache();
        insert(&mut cache, "nemotron-8b", "reasoning");
        insert(&mut cache, "nemotron-8b", "reasoning");
        let features = cache.get("nemotron-8b").unwrap();
        // Should still be exactly one entry.
        assert_eq!(features.len(), 1);
    }

    #[test]
    fn test_multiple_features_per_model() {
        let mut cache = make_cache();
        insert(&mut cache, "nemotron-8b", "reasoning");
        insert(&mut cache, "nemotron-8b", "stream_options");
        assert!(contains(&cache, "nemotron-8b", "reasoning"));
        assert!(contains(&cache, "nemotron-8b", "stream_options"));
    }

    // ── serialization roundtrip ───────────────────────────────────────────────

    #[test]
    fn test_serialization_roundtrip() {
        let path = temp_cache_file();

        // Write
        let mut cache: HashMap<String, HashSet<String>> = HashMap::new();
        cache
            .entry("nemotron-8b".to_string())
            .or_default()
            .insert("reasoning".to_string());

        let snapshot: HashMap<String, Vec<String>> = cache
            .iter()
            .map(|(k, v)| {
                let mut sorted: Vec<String> = v.iter().cloned().collect();
                sorted.sort();
                (k.clone(), sorted)
            })
            .collect();
        let cf = CacheFile { unsupported: snapshot };
        std::fs::write(&path, serde_json::to_string_pretty(&cf).unwrap()).unwrap();

        // Read back
        let data = std::fs::read_to_string(&path).unwrap();
        let parsed: CacheFile = serde_json::from_str(&data).unwrap();
        let reloaded: HashMap<String, HashSet<String>> = parsed
            .unsupported
            .into_iter()
            .map(|(k, v)| (k, v.into_iter().collect()))
            .collect();

        assert!(reloaded
            .get("nemotron-8b")
            .map(|s| s.contains("reasoning"))
            .unwrap_or(false));

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_empty_file_loads_as_empty() {
        let path = temp_cache_file();
        let cf = CacheFile::default();
        std::fs::write(&path, serde_json::to_string(&cf).unwrap()).unwrap();

        let data = std::fs::read_to_string(&path).unwrap();
        let parsed: CacheFile = serde_json::from_str(&data).unwrap();
        assert!(parsed.unsupported.is_empty());

        let _ = std::fs::remove_file(&path);
    }

    // ── global API smoke tests (uses real global cache) ───────────────────────

    #[test]
    fn test_global_mark_and_query() {
        // Use a model name unlikely to clash with real usage in other tests.
        let model = "test-model-xyz-unique-mark-query";
        let feature = "test_feature_xyz";

        // Should start unknown (unless a prior run left it — accept either).
        let was_unsupported = is_feature_unsupported(model, feature);

        mark_feature_unsupported(model, feature);

        // Must be unsupported now regardless.
        assert!(
            is_feature_unsupported(model, feature),
            "should be unsupported after marking; was_unsupported_before={}",
            was_unsupported
        );
    }

    #[test]
    fn test_global_unknown_model_returns_false() {
        let result = is_feature_unsupported("completely-unknown-model-99999", "reasoning");
        assert!(!result);
    }
}
