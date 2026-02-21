use std::collections::HashMap;
use std::sync::RwLock;
use std::time::Instant;

use tracing::{debug, warn};

// --- Constants ---

const DEGRADED_THRESHOLD: u32 = 3;

// --- Types ---

#[derive(Debug, Clone)]
pub struct ProbeResult {
    pub healthy: bool,
    pub latency_ms: u64,
    #[allow(dead_code)]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProbeStatus {
    Healthy,
    Degraded,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct ProbeState {
    pub name: String,
    pub status: ProbeStatus,
    pub consecutive_failures: u32,
    pub last_result: Option<ProbeResult>,
    pub last_checked: Option<Instant>,
    pub last_healthy: Option<Instant>,
}

impl ProbeState {
    pub fn new(name: String) -> Self {
        Self {
            name,
            status: ProbeStatus::Unknown,
            consecutive_failures: 0,
            last_result: None,
            last_checked: None,
            last_healthy: None,
        }
    }

    /// Pure state machine: record a probe result and update status.
    /// Healthy -> Degraded after DEGRADED_THRESHOLD consecutive failures.
    /// Degraded -> Healthy on first success.
    pub fn record_result(&mut self, result: ProbeResult) {
        let now = Instant::now();
        self.last_checked = Some(now);

        if result.healthy {
            self.consecutive_failures = 0;
            self.status = ProbeStatus::Healthy;
            self.last_healthy = Some(now);
        } else {
            self.consecutive_failures += 1;
            if self.consecutive_failures >= DEGRADED_THRESHOLD {
                self.status = ProbeStatus::Degraded;
            }
        }

        self.last_result = Some(result);
    }
}

// --- Trait ---

#[async_trait::async_trait]
pub trait HealthProbe: Send + Sync {
    fn name(&self) -> &str;
    fn interval_secs(&self) -> u64;
    async fn check(&self) -> ProbeResult;
}

// --- Registry ---

pub struct HealthRegistry {
    probes: Vec<Box<dyn HealthProbe>>,
    states: RwLock<HashMap<String, ProbeState>>,
}

impl HealthRegistry {
    pub fn new() -> Self {
        Self {
            probes: Vec::new(),
            states: RwLock::new(HashMap::new()),
        }
    }

    pub fn register(&mut self, probe: Box<dyn HealthProbe>) {
        let name = probe.name().to_string();
        self.states
            .write()
            .unwrap()
            .insert(name.clone(), ProbeState::new(name));
        self.probes.push(probe);
    }

    /// Run probes whose interval has elapsed since last check.
    pub async fn run_due_probes(&self) {
        for probe in &self.probes {
            let name = probe.name().to_string();
            let interval = probe.interval_secs();

            // Check if this probe is due
            let is_due = {
                let states = self.states.read().unwrap();
                match states.get(&name) {
                    Some(state) => match state.last_checked {
                        Some(last) => last.elapsed().as_secs() >= interval,
                        None => true, // Never checked
                    },
                    None => true,
                }
            };

            if is_due {
                let result = probe.check().await;
                debug!(
                    "Health probe '{}': healthy={}, latency={}ms",
                    name, result.healthy, result.latency_ms
                );
                let mut states = self.states.write().unwrap();
                if let Some(state) = states.get_mut(&name) {
                    state.record_result(result);
                }
            }
        }
    }

    /// Returns true if the named probe is NOT degraded.
    /// Unknown probes (not registered) return true (optimistic default).
    pub fn is_healthy(&self, name: &str) -> bool {
        let states = self.states.read().unwrap();
        match states.get(name) {
            Some(state) => state.status != ProbeStatus::Degraded,
            None => true, // Optimistic: unknown = healthy
        }
    }

    /// Snapshot of all probe states for display.
    pub fn all_states(&self) -> Vec<ProbeState> {
        let states = self.states.read().unwrap();
        states.values().cloned().collect()
    }

    /// One-line summary of all probes.
    #[allow(dead_code)]
    pub fn summary_line(&self) -> String {
        let states = self.states.read().unwrap();
        if states.is_empty() {
            return "no probes".to_string();
        }
        states
            .values()
            .map(|s| {
                let status = match s.status {
                    ProbeStatus::Healthy => "ok",
                    ProbeStatus::Degraded => "DOWN",
                    ProbeStatus::Unknown => "?",
                };
                let latency = s
                    .last_result
                    .as_ref()
                    .map(|r| format!(" ({}ms)", r.latency_ms))
                    .unwrap_or_default();
                format!("{}:{}{}", s.name, status, latency)
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Number of registered probes.
    #[allow(dead_code)]
    pub fn probe_count(&self) -> usize {
        self.probes.len()
    }
}

// --- LcmCompactionProbe ---

pub struct LcmCompactionProbe {
    pub(crate) endpoint_url: String,
    client: reqwest::Client,
}

impl LcmCompactionProbe {
    pub fn new(endpoint_url: &str) -> Self {
        // Strip /v1 suffix and trailing slashes
        let base = endpoint_url
            .trim_end_matches('/')
            .trim_end_matches("/v1")
            .trim_end_matches('/');
        Self {
            endpoint_url: base.to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .unwrap_or_default(),
        }
    }
}

#[async_trait::async_trait]
impl HealthProbe for LcmCompactionProbe {
    fn name(&self) -> &str {
        "lcm_compaction"
    }

    fn interval_secs(&self) -> u64 {
        60
    }

    async fn check(&self) -> ProbeResult {
        let url = format!("{}/health", self.endpoint_url);
        let start = Instant::now();
        match self.client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => ProbeResult {
                healthy: true,
                latency_ms: start.elapsed().as_millis() as u64,
                detail: None,
            },
            Ok(resp) => ProbeResult {
                healthy: false,
                latency_ms: start.elapsed().as_millis() as u64,
                detail: Some(format!("HTTP {}", resp.status())),
            },
            Err(e) => {
                warn!("LCM compaction health check failed: {}", e);
                ProbeResult {
                    healthy: false,
                    latency_ms: start.elapsed().as_millis() as u64,
                    detail: Some(e.to_string()),
                }
            }
        }
    }
}

// --- TrioEndpointProbe ---

/// Health probe for trio router/specialist endpoints.
/// Sends a minimal chat completion request (max_tokens: 1) to verify the endpoint is reachable.
pub struct TrioEndpointProbe {
    name: String,
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl TrioEndpointProbe {
    pub fn new(name: &str, base_url: &str, model: &str) -> Self {
        let url = base_url.trim_end_matches('/').trim_end_matches("/v1").to_string();
        Self {
            name: name.to_string(),
            base_url: url,
            model: model.to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .unwrap_or_default(),
        }
    }
}

#[async_trait::async_trait]
impl HealthProbe for TrioEndpointProbe {
    fn name(&self) -> &str {
        &self.name
    }

    fn interval_secs(&self) -> u64 {
        60
    }

    async fn check(&self) -> ProbeResult {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let start = Instant::now();
        let body = serde_json::json!({
            "model": self.model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1
        });
        match self.client.post(&url).json(&body).send().await {
            Ok(resp) if resp.status().as_u16() == 200 => ProbeResult {
                healthy: true,
                latency_ms: start.elapsed().as_millis() as u64,
                detail: Some("ok".to_string()),
            },
            Ok(resp) => ProbeResult {
                healthy: false,
                latency_ms: start.elapsed().as_millis() as u64,
                detail: Some(format!("HTTP {}", resp.status())),
            },
            Err(e) => {
                warn!("Trio endpoint health check '{}' failed: {}", self.name, e);
                ProbeResult {
                    healthy: false,
                    latency_ms: start.elapsed().as_millis() as u64,
                    detail: Some(e.to_string()),
                }
            }
        }
    }
}

// --- Config-driven factory ---

pub fn build_registry(config: &crate::config::schema::Config) -> HealthRegistry {
    let mut reg = HealthRegistry::new();
    if config.lcm.enabled {
        if let Some(ref ep) = config.lcm.compaction_endpoint {
            reg.register(Box::new(LcmCompactionProbe::new(&ep.url)));
        }
    }
    if config.trio.enabled {
        if let Some(ref ep) = config.trio.router_endpoint {
            reg.register(Box::new(TrioEndpointProbe::new(
                "trio_router",
                &ep.url,
                &ep.model,
            )));
        }
        if let Some(ref ep) = config.trio.specialist_endpoint {
            reg.register(Box::new(TrioEndpointProbe::new(
                "trio_specialist",
                &ep.url,
                &ep.model,
            )));
        }
    }
    reg
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ProbeState tests (pure function) ---

    #[test]
    fn test_probe_state_starts_unknown() {
        let state = ProbeState::new("test".to_string());
        assert_eq!(state.status, ProbeStatus::Unknown);
        assert_eq!(state.consecutive_failures, 0);
        assert!(state.last_result.is_none());
        assert!(state.last_checked.is_none());
        assert!(state.last_healthy.is_none());
    }

    #[test]
    fn test_probe_state_healthy_on_success() {
        let mut state = ProbeState::new("test".to_string());
        state.record_result(ProbeResult { healthy: true, latency_ms: 10, detail: None });
        assert_eq!(state.status, ProbeStatus::Healthy);
        assert_eq!(state.consecutive_failures, 0);
        assert!(state.last_healthy.is_some());
    }

    #[test]
    fn test_probe_state_degrades_after_threshold() {
        let mut state = ProbeState::new("test".to_string());
        for _ in 0..DEGRADED_THRESHOLD {
            state.record_result(ProbeResult { healthy: false, latency_ms: 0, detail: None });
        }
        assert_eq!(state.status, ProbeStatus::Degraded);
        assert_eq!(state.consecutive_failures, DEGRADED_THRESHOLD);
    }

    #[test]
    fn test_probe_state_not_degraded_before_threshold() {
        let mut state = ProbeState::new("test".to_string());
        for _ in 0..(DEGRADED_THRESHOLD - 1) {
            state.record_result(ProbeResult { healthy: false, latency_ms: 0, detail: None });
        }
        assert_ne!(state.status, ProbeStatus::Degraded);
        assert_eq!(state.consecutive_failures, DEGRADED_THRESHOLD - 1);
    }

    #[test]
    fn test_probe_state_recovers_on_success() {
        let mut state = ProbeState::new("test".to_string());
        // Degrade it
        for _ in 0..DEGRADED_THRESHOLD {
            state.record_result(ProbeResult { healthy: false, latency_ms: 0, detail: None });
        }
        assert_eq!(state.status, ProbeStatus::Degraded);
        // One success recovers
        state.record_result(ProbeResult { healthy: true, latency_ms: 5, detail: None });
        assert_eq!(state.status, ProbeStatus::Healthy);
        assert_eq!(state.consecutive_failures, 0);
    }

    #[test]
    fn test_probe_state_resets_failures_on_success() {
        let mut state = ProbeState::new("test".to_string());
        state.record_result(ProbeResult { healthy: false, latency_ms: 0, detail: None });
        state.record_result(ProbeResult { healthy: false, latency_ms: 0, detail: None });
        assert_eq!(state.consecutive_failures, 2);
        state.record_result(ProbeResult { healthy: true, latency_ms: 5, detail: None });
        assert_eq!(state.consecutive_failures, 0);
    }

    // --- Registry tests ---

    #[test]
    fn test_registry_unknown_probe_is_healthy() {
        let reg = HealthRegistry::new();
        assert!(reg.is_healthy("nonexistent"));
    }

    #[test]
    fn test_registry_degraded_probe_not_healthy() {
        let mut reg = HealthRegistry::new();
        // Manually insert a degraded state
        {
            let mut states = reg.states.write().unwrap();
            let mut state = ProbeState::new("test_probe".to_string());
            for _ in 0..DEGRADED_THRESHOLD {
                state.record_result(ProbeResult { healthy: false, latency_ms: 0, detail: None });
            }
            states.insert("test_probe".to_string(), state);
        }
        assert!(!reg.is_healthy("test_probe"));
    }

    #[test]
    fn test_registry_healthy_probe_is_healthy() {
        let mut reg = HealthRegistry::new();
        {
            let mut states = reg.states.write().unwrap();
            let mut state = ProbeState::new("test_probe".to_string());
            state.record_result(ProbeResult { healthy: true, latency_ms: 10, detail: None });
            states.insert("test_probe".to_string(), state);
        }
        assert!(reg.is_healthy("test_probe"));
    }

    // --- Mock probe for async tests ---

    struct MockProbe {
        name: String,
        interval: u64,
        healthy: bool,
    }

    #[async_trait::async_trait]
    impl HealthProbe for MockProbe {
        fn name(&self) -> &str { &self.name }
        fn interval_secs(&self) -> u64 { self.interval }
        async fn check(&self) -> ProbeResult {
            ProbeResult { healthy: self.healthy, latency_ms: 1, detail: None }
        }
    }

    #[tokio::test]
    async fn test_registry_run_due_probes_calls_when_due() {
        let mut reg = HealthRegistry::new();
        reg.register(Box::new(MockProbe {
            name: "mock".to_string(),
            interval: 0, // always due
            healthy: true,
        }));
        reg.run_due_probes().await;
        let states = reg.states.read().unwrap();
        let state = states.get("mock").unwrap();
        assert_eq!(state.status, ProbeStatus::Healthy);
        assert!(state.last_checked.is_some());
    }

    #[tokio::test]
    async fn test_registry_run_due_probes_respects_interval() {
        let mut reg = HealthRegistry::new();
        reg.register(Box::new(MockProbe {
            name: "slow_mock".to_string(),
            interval: 3600, // 1 hour
            healthy: true,
        }));
        // First run: should check (never checked before)
        reg.run_due_probes().await;
        let first_check = {
            let states = reg.states.read().unwrap();
            states.get("slow_mock").unwrap().last_checked.unwrap()
        };
        // Second run immediately: should NOT check (interval not elapsed)
        reg.run_due_probes().await;
        let second_check = {
            let states = reg.states.read().unwrap();
            states.get("slow_mock").unwrap().last_checked.unwrap()
        };
        // last_checked should be unchanged (same instant)
        assert_eq!(first_check, second_check);
    }

    // --- LcmCompactionProbe tests ---

    #[test]
    fn test_lcm_probe_name() {
        let probe = LcmCompactionProbe::new("http://localhost:1234/v1");
        assert_eq!(probe.name(), "lcm_compaction");
    }

    #[test]
    fn test_lcm_probe_interval() {
        let probe = LcmCompactionProbe::new("http://localhost:1234/v1");
        assert_eq!(probe.interval_secs(), 60);
    }

    #[test]
    fn test_lcm_probe_strips_v1() {
        let probe = LcmCompactionProbe::new("http://localhost:1234/v1");
        assert_eq!(probe.endpoint_url, "http://localhost:1234");
    }

    #[test]
    fn test_lcm_probe_strips_trailing_slash() {
        let probe = LcmCompactionProbe::new("http://localhost:1234/v1/");
        assert_eq!(probe.endpoint_url, "http://localhost:1234");
    }

    // --- build_registry tests ---

    #[test]
    fn test_build_registry_empty_config() {
        let config = crate::config::schema::Config::default();
        let reg = build_registry(&config);
        assert_eq!(reg.probe_count(), 0);
    }

    #[test]
    fn test_build_registry_lcm_disabled() {
        let mut config = crate::config::schema::Config::default();
        config.lcm.enabled = false;
        config.lcm.compaction_endpoint = Some(crate::config::schema::ModelEndpoint {
            url: "http://localhost:1234/v1".to_string(),
            model: "qwen3-0.6b".to_string(),
        });
        let reg = build_registry(&config);
        assert_eq!(reg.probe_count(), 0);
    }

    #[test]
    fn test_build_registry_with_lcm_endpoint() {
        let mut config = crate::config::schema::Config::default();
        config.lcm.enabled = true;
        config.lcm.compaction_endpoint = Some(crate::config::schema::ModelEndpoint {
            url: "http://localhost:1234/v1".to_string(),
            model: "qwen3-0.6b".to_string(),
        });
        let reg = build_registry(&config);
        assert_eq!(reg.probe_count(), 1);
    }

    #[test]
    fn test_build_registry_lcm_no_endpoint() {
        let mut config = crate::config::schema::Config::default();
        config.lcm.enabled = true;
        // No compaction_endpoint set
        let reg = build_registry(&config);
        assert_eq!(reg.probe_count(), 0);
    }

    #[test]
    fn test_summary_line_no_probes() {
        let reg = HealthRegistry::new();
        assert_eq!(reg.summary_line(), "no probes");
    }

    // --- TrioEndpointProbe tests ---

    #[test]
    fn trio_endpoint_probe_strips_v1() {
        let probe = TrioEndpointProbe::new("trio_router", "http://localhost:1234/v1/", "qwen3");
        assert_eq!(probe.name(), "trio_router");
        assert_eq!(probe.base_url, "http://localhost:1234");
        assert_eq!(probe.model, "qwen3");
    }

    #[test]
    fn test_trio_probe_strips_v1_without_trailing_slash() {
        let probe = TrioEndpointProbe::new("trio_specialist", "http://localhost:8095/v1", "ministral");
        assert_eq!(probe.name(), "trio_specialist");
        assert_eq!(probe.base_url, "http://localhost:8095");
        assert_eq!(probe.model, "ministral");
    }

    #[test]
    fn test_trio_probe_no_v1_suffix() {
        let probe = TrioEndpointProbe::new("trio_router", "http://localhost:1234", "model-x");
        assert_eq!(probe.base_url, "http://localhost:1234");
    }

    // --- build_registry trio tests ---

    #[test]
    fn test_build_registry_trio_disabled_no_probes() {
        let mut config = crate::config::schema::Config::default();
        config.trio.enabled = false;
        config.trio.router_endpoint = Some(crate::config::schema::ModelEndpoint {
            url: "http://localhost:1234/v1".to_string(),
            model: "router".to_string(),
        });
        let reg = build_registry(&config);
        assert_eq!(reg.probe_count(), 0);
    }

    #[test]
    fn test_build_registry_trio_enabled_with_both_endpoints() {
        let mut config = crate::config::schema::Config::default();
        config.trio.enabled = true;
        config.trio.router_endpoint = Some(crate::config::schema::ModelEndpoint {
            url: "http://localhost:8094/v1".to_string(),
            model: "router-model".to_string(),
        });
        config.trio.specialist_endpoint = Some(crate::config::schema::ModelEndpoint {
            url: "http://localhost:8095/v1".to_string(),
            model: "specialist-model".to_string(),
        });
        let reg = build_registry(&config);
        assert_eq!(reg.probe_count(), 2);
    }

    #[test]
    fn test_build_registry_trio_enabled_router_only() {
        let mut config = crate::config::schema::Config::default();
        config.trio.enabled = true;
        config.trio.router_endpoint = Some(crate::config::schema::ModelEndpoint {
            url: "http://localhost:8094/v1".to_string(),
            model: "router-model".to_string(),
        });
        let reg = build_registry(&config);
        assert_eq!(reg.probe_count(), 1);
    }
}
