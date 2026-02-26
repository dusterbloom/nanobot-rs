//! Smart provider routing for the cluster.
//!
//! Decides whether a model request should be routed to a cluster peer,
//! kept local (single-GPU), or forwarded to a cloud provider.

use std::sync::Arc;

use crate::cluster::state::ClusterState;
use crate::config::schema::ClusterConfig;
use crate::providers::base::LLMProvider;
use crate::providers::factory::{self, ProviderSpec};

// ---------------------------------------------------------------------------
// RoutingDecision
// ---------------------------------------------------------------------------

/// Where to route a model request.
#[derive(Debug, Clone)]
pub enum RoutingDecision {
    /// Route to a cluster peer (endpoint URL + model ID).
    Cluster { endpoint: String, model: String },
    /// Use local single-GPU provider (no change needed).
    Local,
    /// Fall back to cloud provider (no change needed).
    Cloud,
}

// ---------------------------------------------------------------------------
// ClusterRouter
// ---------------------------------------------------------------------------

/// Routes model requests between the cluster, local, and cloud providers.
pub struct ClusterRouter {
    state: ClusterState,
    config: ClusterConfig,
}

impl ClusterRouter {
    /// Create a new `ClusterRouter`.
    pub fn new(state: ClusterState, config: ClusterConfig) -> Self {
        Self { state, config }
    }

    /// Decide where to route a request for `model_id`.
    ///
    /// Routing rules (in priority order):
    /// 1. Cluster disabled → `Local` if `has_local`, else `Cloud`.
    /// 2. Model found on cluster AND `prefer_cluster` → `Cluster`.
    /// 3. Model found on cluster AND !`prefer_cluster` → prefer `Local`/`Cloud`,
    ///    fall back to `Cluster` if neither is available.
    /// 4. Model not found on cluster → `Local` if `has_local`, else `Cloud`.
    pub async fn route(&self, model_id: &str, has_local: bool, has_cloud: bool) -> RoutingDecision {
        // Rule 1: cluster feature is disabled.
        if !self.config.enabled {
            let decision = if has_local {
                RoutingDecision::Local
            } else {
                RoutingDecision::Cloud
            };
            tracing::info!(
                model = model_id,
                reason = "cluster_disabled",
                "routing_decision: {:?}",
                decision
            );
            return decision;
        }

        // Query cluster state for the model.
        let match_opt = self.state.find_model(model_id).await;

        let decision = match match_opt {
            Some(m) => {
                let endpoint = m.peer.endpoint.clone();
                let model = m.model.id.clone();

                if self.config.prefer_cluster {
                    // Rule 2: prefer_cluster is true — always use cluster when available.
                    RoutingDecision::Cluster { endpoint, model }
                } else {
                    // Rule 3: prefer local/cloud, use cluster only as last resort.
                    if has_local {
                        RoutingDecision::Local
                    } else if has_cloud {
                        RoutingDecision::Cloud
                    } else {
                        RoutingDecision::Cluster { endpoint, model }
                    }
                }
            }
            None => {
                // Rule 4: model not found on cluster.
                if has_local {
                    RoutingDecision::Local
                } else {
                    RoutingDecision::Cloud
                }
            }
        };

        tracing::info!(
            model = model_id,
            has_local,
            has_cloud,
            prefer_cluster = self.config.prefer_cluster,
            "routing_decision: {:?}",
            decision
        );

        decision
    }

    /// Create an OpenAI-compatible provider for a `Cluster` routing decision.
    ///
    /// Returns `Some((provider, model_id))` when the decision is `Cluster`,
    /// or `None` for `Local` / `Cloud` decisions (no provider change needed).
    pub async fn create_cluster_provider(
        &self,
        decision: &RoutingDecision,
    ) -> Option<(Arc<dyn LLMProvider>, String)> {
        match decision {
            RoutingDecision::Cluster { endpoint, model } => {
                let spec = ProviderSpec {
                    // Local cluster endpoints don't require real auth keys.
                    api_key: "cluster".to_string(),
                    api_base: Some(endpoint.clone()),
                    model: Some(model.clone()),
                    jit_gate: None,
                };
                let provider = factory::create_openai_compat(spec);
                tracing::info!(
                    endpoint = %endpoint,
                    model = %model,
                    "cluster_provider_created"
                );
                Some((provider, model.clone()))
            }
            RoutingDecision::Local | RoutingDecision::Cloud => None,
        }
    }

    /// Human-readable summary of the current cluster state for logging/status display.
    pub async fn summary(&self) -> String {
        let peers = self.state.get_all_peers().await;
        if peers.is_empty() {
            return format!(
                "Cluster: enabled={}, no peers discovered",
                self.config.enabled
            );
        }

        let healthy: Vec<_> = peers.iter().filter(|p| p.healthy).collect();
        let total_models: usize = healthy.iter().map(|p| p.models.len()).sum();

        let peer_lines: Vec<String> = peers
            .iter()
            .map(|p| {
                let model_names: Vec<&str> = p.models.iter().map(|m| m.id.as_str()).collect();
                format!(
                    "  {} [{}] healthy={} models=[{}]",
                    p.endpoint,
                    p.peer_type,
                    p.healthy,
                    model_names.join(", ")
                )
            })
            .collect();

        format!(
            "Cluster: enabled={} prefer_cluster={} peers={} healthy={} models={}\n{}",
            self.config.enabled,
            self.config.prefer_cluster,
            peers.len(),
            healthy.len(),
            total_models,
            peer_lines.join("\n")
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::cluster::state::{ClusterModel, ClusterPeer, PeerType};

    fn make_config(enabled: bool, prefer_cluster: bool) -> ClusterConfig {
        ClusterConfig {
            enabled,
            prefer_cluster,
            ..Default::default()
        }
    }

    fn make_peer(endpoint: &str, models: Vec<&str>, healthy: bool) -> ClusterPeer {
        ClusterPeer {
            endpoint: endpoint.to_string(),
            peer_type: PeerType::Exo,
            models: models
                .into_iter()
                .map(|id| ClusterModel {
                    id: id.to_string(),
                    context_window: 4096,
                    requires_cluster: false,
                })
                .collect(),
            total_vram_mb: None,
            last_seen: Instant::now(),
            healthy,
        }
    }

    async fn router_with_peer(
        peer: ClusterPeer,
        enabled: bool,
        prefer_cluster: bool,
    ) -> ClusterRouter {
        let state = ClusterState::new();
        state.update_peer(peer).await;
        ClusterRouter::new(state, make_config(enabled, prefer_cluster))
    }

    // --- Route to cluster when model is available and prefer_cluster is true ---

    #[tokio::test]
    async fn test_route_to_cluster_prefer_cluster_true() {
        let peer = make_peer("http://192.168.1.50:52415/v1", vec!["qwen-72b"], true);
        let router = router_with_peer(peer, true, true).await;

        let decision = router.route("qwen-72b", true, true).await;
        match decision {
            RoutingDecision::Cluster { endpoint, model } => {
                assert_eq!(endpoint, "http://192.168.1.50:52415/v1");
                assert_eq!(model, "qwen-72b");
            }
            other => panic!("Expected Cluster, got {:?}", other),
        }
    }

    // --- Route to local when cluster has no matching model ---

    #[tokio::test]
    async fn test_route_to_local_when_model_not_on_cluster() {
        let peer = make_peer("http://192.168.1.50:52415/v1", vec!["llama-70b"], true);
        let router = router_with_peer(peer, true, true).await;

        let decision = router.route("qwen-72b", true, true).await;
        assert!(
            matches!(decision, RoutingDecision::Local),
            "Expected Local when model is absent from cluster"
        );
    }

    // --- Route to cloud when neither cluster nor local available ---

    #[tokio::test]
    async fn test_route_to_cloud_when_no_local_and_no_cluster_match() {
        let state = ClusterState::new(); // empty — no peers
        let router = ClusterRouter::new(state, make_config(true, true));

        let decision = router.route("gpt-4o", false, true).await;
        assert!(
            matches!(decision, RoutingDecision::Cloud),
            "Expected Cloud when cluster is empty and no local"
        );
    }

    // --- Disabled cluster always bypasses ---

    #[tokio::test]
    async fn test_disabled_cluster_routes_local() {
        let peer = make_peer("http://192.168.1.50:52415/v1", vec!["qwen-72b"], true);
        let router = router_with_peer(peer, false, true).await;

        let decision = router.route("qwen-72b", true, true).await;
        assert!(
            matches!(decision, RoutingDecision::Local),
            "Disabled cluster should route to Local when has_local"
        );
    }

    #[tokio::test]
    async fn test_disabled_cluster_routes_cloud_when_no_local() {
        let peer = make_peer("http://192.168.1.50:52415/v1", vec!["qwen-72b"], true);
        let router = router_with_peer(peer, false, true).await;

        let decision = router.route("qwen-72b", false, true).await;
        assert!(
            matches!(decision, RoutingDecision::Cloud),
            "Disabled cluster should route to Cloud when !has_local"
        );
    }

    // --- prefer_cluster=false prefers local/cloud over cluster ---

    #[tokio::test]
    async fn test_prefer_cluster_false_uses_local_when_available() {
        let peer = make_peer("http://192.168.1.50:52415/v1", vec!["qwen-72b"], true);
        let router = router_with_peer(peer, true, false).await;

        let decision = router.route("qwen-72b", true, true).await;
        assert!(
            matches!(decision, RoutingDecision::Local),
            "prefer_cluster=false should prefer Local over Cluster"
        );
    }

    #[tokio::test]
    async fn test_prefer_cluster_false_uses_cloud_when_no_local() {
        let peer = make_peer("http://192.168.1.50:52415/v1", vec!["qwen-72b"], true);
        let router = router_with_peer(peer, true, false).await;

        let decision = router.route("qwen-72b", false, true).await;
        assert!(
            matches!(decision, RoutingDecision::Cloud),
            "prefer_cluster=false should prefer Cloud over Cluster"
        );
    }

    #[tokio::test]
    async fn test_prefer_cluster_false_falls_back_to_cluster_when_no_local_no_cloud() {
        let peer = make_peer("http://192.168.1.50:52415/v1", vec!["qwen-72b"], true);
        let router = router_with_peer(peer, true, false).await;

        let decision = router.route("qwen-72b", false, false).await;
        match decision {
            RoutingDecision::Cluster { endpoint, model } => {
                assert_eq!(endpoint, "http://192.168.1.50:52415/v1");
                assert_eq!(model, "qwen-72b");
            }
            other => panic!(
                "Expected Cluster as last resort when prefer_cluster=false, got {:?}",
                other
            ),
        }
    }

    // --- create_cluster_provider ---

    #[tokio::test]
    async fn test_create_cluster_provider_returns_provider_for_cluster_decision() {
        let state = ClusterState::new();
        let router = ClusterRouter::new(state, make_config(true, true));

        let decision = RoutingDecision::Cluster {
            endpoint: "http://192.168.1.50:52415/v1".to_string(),
            model: "qwen-72b".to_string(),
        };

        let result = router.create_cluster_provider(&decision).await;
        assert!(result.is_some(), "Should return a provider for Cluster decision");
        let (provider, model_id) = result.unwrap();
        assert_eq!(model_id, "qwen-72b");
        // The provider should report the cluster endpoint as its base.
        assert_eq!(
            provider.get_api_base().as_deref(),
            Some("http://192.168.1.50:52415/v1")
        );
    }

    #[tokio::test]
    async fn test_create_cluster_provider_returns_none_for_local() {
        let state = ClusterState::new();
        let router = ClusterRouter::new(state, make_config(true, true));

        let result = router.create_cluster_provider(&RoutingDecision::Local).await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_create_cluster_provider_returns_none_for_cloud() {
        let state = ClusterState::new();
        let router = ClusterRouter::new(state, make_config(true, true));

        let result = router.create_cluster_provider(&RoutingDecision::Cloud).await;
        assert!(result.is_none());
    }

    // --- summary ---

    #[tokio::test]
    async fn test_summary_no_peers() {
        let state = ClusterState::new();
        let router = ClusterRouter::new(state, make_config(false, true));
        let s = router.summary().await;
        assert!(s.contains("no peers discovered"));
    }

    #[tokio::test]
    async fn test_summary_with_peers() {
        let peer = make_peer("http://exo:52415/v1", vec!["qwen-72b", "llama-70b"], true);
        let router = router_with_peer(peer, true, true).await;
        let s = router.summary().await;
        assert!(s.contains("http://exo:52415/v1"));
        assert!(s.contains("qwen-72b"));
        assert!(s.contains("llama-70b"));
    }
}
