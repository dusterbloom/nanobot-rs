//! Discovery-first local inference.
//!
//! Before any local server is spawned, candidate endpoints are probed in
//! parallel (`GET /v1/models`, ≤1s each). Endpoint and model are then adopted
//! as a PAIR from a single discovery result — never resolved independently.
//! (The `:1234` bug: model came from `lmsMainModel` adopted against a healthy
//! Higgs on :8000, while the endpoint fell back to the LM Studio port because
//! `localApiBase` was empty and `localBackend` said "lmstudio".)

use std::time::Duration;

use crate::config::schema::{Config, LocalAutostart};

/// Per-endpoint probe budget. Probes run in parallel, so worst-case startup
/// cost is ~1s total, not 1s per candidate.
const PROBE_TIMEOUT_MS: u64 = 1000;

/// User-facing note shown when discovery finds nothing and autostart is off.
pub(crate) const NO_SERVER_NOTE: &str =
    "no local inference server found — start one (higgs serve / LM Studio) or set localAutostart";

/// Where a candidate endpoint came from. Variant order IS the selection
/// priority when no candidate serves the expected model:
/// configured `localApiBase` > Higgs > LM Studio > cluster peers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum EndpointSource {
    /// `agents.defaults.localApiBase` from config.
    Configured,
    /// Higgs on `agents.defaults.higgsPort`.
    Higgs,
    /// LM Studio on `agents.defaults.lmsPort`.
    LmStudio,
    /// An entry from `cluster.endpoints`.
    ClusterPeer,
}

/// A healthy endpoint (answered `GET /v1/models`) and the model ids it serves.
#[derive(Debug, Clone)]
pub(crate) struct DiscoveredEndpoint {
    /// Normalized base URL ending in `/v1`.
    pub base_url: String,
    pub source: EndpointSource,
    /// Served model ids as reported by the endpoint (may be empty).
    pub models: Vec<String>,
}

/// Endpoint + model adopted TOGETHER from one discovery result.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AdoptedLocal {
    pub base_url: String,
    pub model: String,
    pub source: EndpointSource,
}

/// What local startup should do after discovery. Spawning happens ONLY for
/// the `Spawn*` variants, which require an explicit `localAutostart` opt-in.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum StartupAction {
    /// A healthy endpoint was found — use it, spawn nothing.
    UseDiscovered(AdoptedLocal),
    /// Nothing found and `localAutostart: "higgs"` — start the Higgs sidecar.
    SpawnHiggs,
    /// Nothing found and `localAutostart: "lmstudio"` — start LM Studio.
    SpawnLmStudio,
    /// Nothing found and `localAutostart: "off"` — show [`NO_SERVER_NOTE`].
    NoServerNote,
}

/// Normalize a candidate URL to `http(s)://host:port/v1` form.
pub(crate) fn normalize_base_url(raw: &str) -> String {
    let raw = raw.trim().trim_end_matches('/');
    let with_scheme = if raw.starts_with("http://") || raw.starts_with("https://") {
        raw.to_string()
    } else {
        format!("http://{raw}")
    };
    if with_scheme.ends_with("/v1") {
        with_scheme
    } else {
        format!("{with_scheme}/v1")
    }
}

/// Build the ordered candidate list from config (deduplicated, normalized).
pub(crate) fn candidate_endpoints(config: &Config) -> Vec<(String, EndpointSource)> {
    let d = &config.agents.defaults;
    let mut raw: Vec<(String, EndpointSource)> = Vec::new();
    if !d.local_api_base.trim().is_empty() {
        raw.push((d.local_api_base.clone(), EndpointSource::Configured));
    }
    raw.push((
        format!("http://127.0.0.1:{}", d.higgs_port),
        EndpointSource::Higgs,
    ));
    raw.push((
        format!("http://127.0.0.1:{}", d.lms_port),
        EndpointSource::LmStudio,
    ));
    for ep in &config.cluster.endpoints {
        raw.push((ep.clone(), EndpointSource::ClusterPeer));
    }

    let mut out: Vec<(String, EndpointSource)> = Vec::new();
    for (url, source) in raw {
        let url = normalize_base_url(&url);
        if !out.iter().any(|(seen, _)| *seen == url) {
            out.push((url, source));
        }
    }
    out
}

/// Pure endpoint selection over healthy candidates.
///
/// Priority: (a) any endpoint serving `expected_model` (source order breaks
/// ties), else (b)–(e) pure source order. The adopted model always comes from
/// the CHOSEN endpoint's served list (falling back to `expected_model` only
/// when the endpoint reports no ids), so endpoint and model cannot diverge.
pub(crate) fn select_endpoint(
    candidates: &[DiscoveredEndpoint],
    expected_model: &str,
) -> Option<AdoptedLocal> {
    let mut ranked: Vec<&DiscoveredEndpoint> = candidates.iter().collect();
    ranked.sort_by_key(|c| c.source);

    let serves_expected = |c: &DiscoveredEndpoint| {
        !expected_model.is_empty()
            && c.models
                .iter()
                .any(|id| crate::higgs::model_id_matches(id, expected_model))
    };
    let chosen = ranked
        .iter()
        .find(|c| serves_expected(c))
        .or_else(|| ranked.first())?;

    let model = crate::higgs::adopt_served_model(expected_model, &chosen.models)
        .unwrap_or_else(|| expected_model.to_string());
    Some(AdoptedLocal {
        base_url: chosen.base_url.clone(),
        model,
        source: chosen.source,
    })
}

/// Pure startup decision: discovery result + autostart policy → action.
pub(crate) fn decide_startup(
    discovered: Option<AdoptedLocal>,
    autostart: LocalAutostart,
) -> StartupAction {
    let Some(pair) = discovered else {
        return match autostart {
            LocalAutostart::Off => StartupAction::NoServerNote,
            LocalAutostart::Higgs => StartupAction::SpawnHiggs,
            LocalAutostart::Lmstudio => StartupAction::SpawnLmStudio,
        };
    };
    StartupAction::UseDiscovered(pair)
}

/// Probe all candidates in parallel; return only the healthy ones.
/// Result order follows candidate (priority) order, not response order.
pub(crate) async fn discover_endpoints(config: &Config) -> Vec<DiscoveredEndpoint> {
    let api_key = config.agents.defaults.local_api_key.clone();
    let probes = candidate_endpoints(config).into_iter().map(|(base, source)| {
        let key = api_key.clone();
        async move {
            let models = probe_models(&base, &key).await?;
            Some(DiscoveredEndpoint {
                base_url: base,
                source,
                models,
            })
        }
    });
    futures_util::future::join_all(probes)
        .await
        .into_iter()
        .flatten()
        .collect()
}

/// `GET {base}/models` with a hard [`PROBE_TIMEOUT_MS`] budget.
/// `Some(ids)` = healthy endpoint (ids may be empty); `None` = dead/not OpenAI.
async fn probe_models(base_url: &str, api_key: &str) -> Option<Vec<String>> {
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(PROBE_TIMEOUT_MS))
        .connect_timeout(Duration::from_millis(PROBE_TIMEOUT_MS))
        .build()
        .ok()?;
    let mut req = client.get(&url);
    if !api_key.is_empty() {
        req = req.bearer_auth(api_key);
    }
    let resp = req.send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let json = resp.json::<serde_json::Value>().await.ok()?;
    let ids = json
        .get("data")?
        .as_array()?
        .iter()
        .filter_map(|m| m.get("id").and_then(|id| id.as_str()).map(String::from))
        .collect();
    Some(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ep(base: &str, source: EndpointSource, models: &[&str]) -> DiscoveredEndpoint {
        DiscoveredEndpoint {
            base_url: base.to_string(),
            source,
            models: models.iter().map(|m| m.to_string()).collect(),
        }
    }

    // -- acceptance case #1: higgs healthy on :8000, :1234 dead, stale
    //    "lmstudio" backend remnants. Discovery must adopt the higgs PAIR. --
    #[test]
    fn test_select_adopts_higgs_pair_when_lms_dead() {
        // :1234 never answered, so it is simply absent from candidates.
        let candidates = vec![ep(
            "http://127.0.0.1:8000/v1",
            EndpointSource::Higgs,
            &["Qwen3.6-35B-A3B-4bit"],
        )];
        let pair = select_endpoint(&candidates, "Qwen3.6-35B-A3B-4bit").unwrap();
        assert_eq!(pair.base_url, "http://127.0.0.1:8000/v1");
        assert_eq!(pair.model, "Qwen3.6-35B-A3B-4bit");
        assert_eq!(pair.source, EndpointSource::Higgs);
    }

    #[test]
    fn test_select_prefers_endpoint_serving_expected_model() {
        // Configured endpoint is healthy but serves something else; LM Studio
        // serves (a normalized alias of) the expected model. The alias id the
        // server actually announces is adopted — never the configured string.
        let candidates = vec![
            ep(
                "http://192.168.1.22:18100/v1",
                EndpointSource::Configured,
                &["glm-4.7-flash"],
            ),
            ep(
                "http://127.0.0.1:1234/v1",
                EndpointSource::LmStudio,
                &["qwen36-35b"],
            ),
        ];
        let pair = select_endpoint(&candidates, "Qwen3.6-35B-A3B-4bit").unwrap();
        assert_eq!(pair.base_url, "http://127.0.0.1:1234/v1");
        assert_eq!(pair.model, "qwen36-35b");
        assert_eq!(pair.source, EndpointSource::LmStudio);
    }

    #[test]
    fn test_select_configured_wins_when_both_serve_expected() {
        let candidates = vec![
            ep(
                "http://127.0.0.1:8000/v1",
                EndpointSource::Higgs,
                &["qwen36-35b"],
            ),
            ep(
                "http://192.168.1.22:18100/v1",
                EndpointSource::Configured,
                &["qwen36-35b"],
            ),
        ];
        let pair = select_endpoint(&candidates, "qwen36-35b").unwrap();
        assert_eq!(pair.source, EndpointSource::Configured);
        assert_eq!(pair.base_url, "http://192.168.1.22:18100/v1");
    }

    #[test]
    fn test_select_falls_back_to_source_priority_and_served_model() {
        // Nobody serves the expected model: higgs beats lms, and the model is
        // adopted from what the chosen endpoint ACTUALLY serves.
        let candidates = vec![
            ep(
                "http://127.0.0.1:1234/v1",
                EndpointSource::LmStudio,
                &["gemma-3n-e4b"],
            ),
            ep(
                "http://127.0.0.1:8000/v1",
                EndpointSource::Higgs,
                &["minicpm5-1b"],
            ),
        ];
        let pair = select_endpoint(&candidates, "some-model-nobody-has").unwrap();
        assert_eq!(pair.source, EndpointSource::Higgs);
        assert_eq!(pair.base_url, "http://127.0.0.1:8000/v1");
        assert_eq!(pair.model, "minicpm5-1b");
    }

    #[test]
    fn test_select_none_when_no_healthy_candidates() {
        assert_eq!(select_endpoint(&[], "anything"), None);
    }

    #[test]
    fn test_select_keeps_expected_model_when_endpoint_reports_no_ids() {
        let candidates = vec![ep("http://127.0.0.1:8000/v1", EndpointSource::Higgs, &[])];
        let pair = select_endpoint(&candidates, "qwen36-35b").unwrap();
        assert_eq!(pair.model, "qwen36-35b");
    }

    // -- decision fn: no server found must produce the note, never a spawn --
    #[test]
    fn test_decide_no_server_and_autostart_off_is_note_not_spawn() {
        let action = decide_startup(None, LocalAutostart::Off);
        assert_eq!(action, StartupAction::NoServerNote);
    }

    #[test]
    fn test_decide_no_server_spawns_only_with_explicit_autostart() {
        assert_eq!(
            decide_startup(None, LocalAutostart::Higgs),
            StartupAction::SpawnHiggs
        );
        assert_eq!(
            decide_startup(None, LocalAutostart::Lmstudio),
            StartupAction::SpawnLmStudio
        );
    }

    #[test]
    fn test_decide_discovered_endpoint_always_wins_over_autostart() {
        let pair = AdoptedLocal {
            base_url: "http://127.0.0.1:8000/v1".to_string(),
            model: "qwen36-35b".to_string(),
            source: EndpointSource::Higgs,
        };
        for autostart in [
            LocalAutostart::Off,
            LocalAutostart::Higgs,
            LocalAutostart::Lmstudio,
        ] {
            assert_eq!(
                decide_startup(Some(pair.clone()), autostart),
                StartupAction::UseDiscovered(pair.clone())
            );
        }
    }

    // -- candidate assembly --
    #[test]
    fn test_candidates_cover_configured_higgs_lms_and_cluster() {
        let mut config = Config::default();
        config.agents.defaults.local_api_base = "http://192.168.1.22:18100".to_string();
        config.agents.defaults.higgs_port = 8000;
        config.agents.defaults.lms_port = 1234;
        config.cluster.endpoints = vec!["192.168.1.50:52415".to_string()];

        let cands = candidate_endpoints(&config);
        assert_eq!(
            cands,
            vec![
                (
                    "http://192.168.1.22:18100/v1".to_string(),
                    EndpointSource::Configured
                ),
                ("http://127.0.0.1:8000/v1".to_string(), EndpointSource::Higgs),
                (
                    "http://127.0.0.1:1234/v1".to_string(),
                    EndpointSource::LmStudio
                ),
                (
                    "http://192.168.1.50:52415/v1".to_string(),
                    EndpointSource::ClusterPeer
                ),
            ]
        );
    }

    #[test]
    fn test_candidates_dedup_configured_equal_to_higgs() {
        let mut config = Config::default();
        config.agents.defaults.local_api_base = "http://127.0.0.1:8000/v1".to_string();
        config.agents.defaults.higgs_port = 8000;

        let cands = candidate_endpoints(&config);
        let higgs_url_count = cands
            .iter()
            .filter(|(url, _)| url == "http://127.0.0.1:8000/v1")
            .count();
        assert_eq!(higgs_url_count, 1, "configured==higgs must not probe twice");
        // The shared URL keeps the higher-priority Configured source.
        assert_eq!(cands[0].1, EndpointSource::Configured);
    }

    #[test]
    fn test_normalize_base_url() {
        assert_eq!(
            normalize_base_url("192.168.1.50:52415"),
            "http://192.168.1.50:52415/v1"
        );
        assert_eq!(
            normalize_base_url("http://127.0.0.1:8000/v1/"),
            "http://127.0.0.1:8000/v1"
        );
        assert_eq!(
            normalize_base_url("https://peer.local:8443"),
            "https://peer.local:8443/v1"
        );
    }
}
