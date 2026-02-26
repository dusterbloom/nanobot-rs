//! Cluster state: peer tracking and capability management.

use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

/// Type of inference server a peer is running.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PeerType {
    /// Exo distributed inference (exo-explore/exo).
    Exo,
    /// LM Studio local server.
    LMStudio,
    /// llama.cpp server.
    LlamaCpp,
    /// JAN local inference server.
    Jan,
    /// Unknown OpenAI-compatible server.
    Unknown,
}

impl std::fmt::Display for PeerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PeerType::Exo => write!(f, "exo"),
            PeerType::LMStudio => write!(f, "lmstudio"),
            PeerType::LlamaCpp => write!(f, "llamacpp"),
            PeerType::Jan => write!(f, "jan"),
            PeerType::Unknown => write!(f, "unknown"),
        }
    }
}

/// A model available on a cluster peer.
#[derive(Debug, Clone)]
pub struct ClusterModel {
    /// Model identifier as reported by the peer's API.
    pub id: String,
    /// Context window size in tokens (0 if unknown).
    pub context_window: usize,
    /// Whether this model requires multiple GPUs / cluster sharding.
    pub requires_cluster: bool,
}

/// A discovered inference peer on the network.
#[derive(Debug, Clone)]
pub struct ClusterPeer {
    /// API endpoint URL (e.g. "http://192.168.1.50:52415/v1").
    pub endpoint: String,
    /// Type of server running at this endpoint.
    pub peer_type: PeerType,
    /// Models available on this peer.
    pub models: Vec<ClusterModel>,
    /// Total VRAM in MB (if reported).
    pub total_vram_mb: Option<u64>,
    /// When this peer was last seen healthy.
    pub last_seen: Instant,
    /// Whether the peer is currently reachable.
    pub healthy: bool,
}

/// Result of searching the cluster for a model.
#[derive(Debug, Clone)]
pub struct ModelMatch {
    /// The peer that has this model.
    pub peer: ClusterPeer,
    /// The matched model info.
    pub model: ClusterModel,
}

/// Thread-safe cluster state tracking all discovered peers.
#[derive(Clone)]
pub struct ClusterState {
    peers: Arc<RwLock<Vec<ClusterPeer>>>,
}

impl ClusterState {
    /// Create empty cluster state.
    pub fn new() -> Self {
        Self {
            peers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Update or insert a peer. Matches by endpoint URL.
    pub async fn update_peer(&self, peer: ClusterPeer) {
        let mut peers = self.peers.write().await;
        if let Some(existing) = peers.iter_mut().find(|p| p.endpoint == peer.endpoint) {
            *existing = peer;
        } else {
            tracing::info!(
                endpoint = %peer.endpoint,
                peer_type = %peer.peer_type,
                models = peer.models.len(),
                "cluster_peer_discovered"
            );
            peers.push(peer);
        }
    }

    /// Remove peers not seen within the given duration.
    pub async fn remove_stale_peers(&self, max_age: std::time::Duration) {
        let mut peers = self.peers.write().await;
        let before = peers.len();
        peers.retain(|p| p.last_seen.elapsed() < max_age);
        let removed = before - peers.len();
        if removed > 0 {
            tracing::info!(removed, remaining = peers.len(), "cluster_stale_peers_removed");
        }
    }

    /// Get all healthy peers.
    pub async fn get_healthy_peers(&self) -> Vec<ClusterPeer> {
        let peers = self.peers.read().await;
        peers.iter().filter(|p| p.healthy).cloned().collect()
    }

    /// Get all peers (including unhealthy).
    pub async fn get_all_peers(&self) -> Vec<ClusterPeer> {
        self.peers.read().await.clone()
    }

    /// Find the first healthy peer that has a model matching the given ID.
    /// Model matching is case-insensitive and supports substring match.
    pub async fn find_model(&self, model_id: &str) -> Option<ModelMatch> {
        let peers = self.peers.read().await;
        let lower = model_id.to_ascii_lowercase();
        for peer in peers.iter().filter(|p| p.healthy) {
            for model in &peer.models {
                if model.id.to_ascii_lowercase() == lower
                    || model.id.to_ascii_lowercase().contains(&lower)
                {
                    return Some(ModelMatch {
                        peer: peer.clone(),
                        model: model.clone(),
                    });
                }
            }
        }
        None
    }

    /// Number of tracked peers.
    pub async fn peer_count(&self) -> usize {
        self.peers.read().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_peer(endpoint: &str, peer_type: PeerType, models: Vec<&str>, healthy: bool) -> ClusterPeer {
        ClusterPeer {
            endpoint: endpoint.to_string(),
            peer_type,
            models: models.into_iter().map(|id| ClusterModel {
                id: id.to_string(),
                context_window: 4096,
                requires_cluster: false,
            }).collect(),
            total_vram_mb: None,
            last_seen: Instant::now(),
            healthy,
        }
    }

    #[tokio::test]
    async fn test_update_peer_insert() {
        let state = ClusterState::new();
        let peer = make_peer("http://192.168.1.10:52415/v1", PeerType::Exo, vec!["qwen-72b"], true);
        state.update_peer(peer).await;
        assert_eq!(state.peer_count().await, 1);
    }

    #[tokio::test]
    async fn test_update_peer_replace() {
        let state = ClusterState::new();
        let peer1 = make_peer("http://192.168.1.10:52415/v1", PeerType::Exo, vec!["qwen-72b"], true);
        state.update_peer(peer1).await;

        let peer2 = make_peer("http://192.168.1.10:52415/v1", PeerType::Exo, vec!["qwen-72b", "llama-70b"], true);
        state.update_peer(peer2).await;

        assert_eq!(state.peer_count().await, 1);
        let peers = state.get_all_peers().await;
        assert_eq!(peers[0].models.len(), 2);
    }

    #[tokio::test]
    async fn test_get_healthy_peers() {
        let state = ClusterState::new();
        state.update_peer(make_peer("http://a:1234/v1", PeerType::LMStudio, vec!["m1"], true)).await;
        state.update_peer(make_peer("http://b:1234/v1", PeerType::LMStudio, vec!["m2"], false)).await;

        let healthy = state.get_healthy_peers().await;
        assert_eq!(healthy.len(), 1);
        assert_eq!(healthy[0].endpoint, "http://a:1234/v1");
    }

    #[tokio::test]
    async fn test_find_model_exact() {
        let state = ClusterState::new();
        state.update_peer(make_peer("http://exo:52415/v1", PeerType::Exo, vec!["qwen3.5-72b-q4"], true)).await;

        let found = state.find_model("qwen3.5-72b-q4").await;
        assert!(found.is_some());
        assert_eq!(found.unwrap().model.id, "qwen3.5-72b-q4");
    }

    #[tokio::test]
    async fn test_find_model_case_insensitive() {
        let state = ClusterState::new();
        state.update_peer(make_peer("http://exo:52415/v1", PeerType::Exo, vec!["Qwen3.5-72B-Q4"], true)).await;

        let found = state.find_model("qwen3.5-72b-q4").await;
        assert!(found.is_some());
    }

    #[tokio::test]
    async fn test_find_model_substring() {
        let state = ClusterState::new();
        state.update_peer(make_peer("http://exo:52415/v1", PeerType::Exo, vec!["qwen3.5-72b-q4-exl2"], true)).await;

        let found = state.find_model("qwen3.5-72b").await;
        assert!(found.is_some());
    }

    #[tokio::test]
    async fn test_find_model_skips_unhealthy() {
        let state = ClusterState::new();
        state.update_peer(make_peer("http://down:52415/v1", PeerType::Exo, vec!["qwen-72b"], false)).await;

        let found = state.find_model("qwen-72b").await;
        assert!(found.is_none());
    }

    #[tokio::test]
    async fn test_remove_stale_peers() {
        let state = ClusterState::new();
        // Fresh peer
        state.update_peer(make_peer("http://fresh:1234/v1", PeerType::LMStudio, vec!["m1"], true)).await;

        // Manually insert a stale peer
        {
            let mut peers = state.peers.write().await;
            peers.push(ClusterPeer {
                endpoint: "http://stale:1234/v1".to_string(),
                peer_type: PeerType::Unknown,
                models: vec![],
                total_vram_mb: None,
                last_seen: Instant::now() - std::time::Duration::from_secs(120),
                healthy: false,
            });
        }

        assert_eq!(state.peer_count().await, 2);
        state.remove_stale_peers(std::time::Duration::from_secs(60)).await;
        assert_eq!(state.peer_count().await, 1);
    }
}
