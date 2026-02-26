//! Cluster peer discovery: mDNS browse + HTTP subnet probe.
//!
//! Discovers Exo and other OpenAI-compatible inference servers on the LAN.
//! Two discovery paths run each cycle:
//!
//! 1. mDNS browse for `_exo._tcp.local.` (5 s window).
//! 2. HTTP probe: iterate the host's /24 subnet on configured ports.
//!
//! Manually-configured endpoints are always probed regardless of
//! `auto_discover`.

use std::net::IpAddr;
use std::time::{Duration, Instant};

use futures_util::stream::FuturesUnordered;
use futures_util::StreamExt;
use tokio::task::JoinHandle;
use tokio::time::timeout;

#[cfg(feature = "cluster")]
use mdns_sd::{ServiceDaemon, ServiceEvent};

use crate::cluster::state::{ClusterModel, ClusterPeer, ClusterState, PeerType};
use crate::config::schema::ClusterConfig;

/// Maximum concurrent HTTP probes at once (keeps scan fast without flooding).
const MAX_CONCURRENT_PROBES: usize = 20;

/// HTTP timeout for a single `/v1/models` probe.
const PROBE_TIMEOUT_SECS: u64 = 3;

/// mDNS browse window before moving on.
const MDNS_BROWSE_TIMEOUT_SECS: u64 = 5;

/// Exo mDNS service type.
const EXO_SERVICE_TYPE: &str = "_exo._tcp.local.";

// ---------------------------------------------------------------------------
// ClusterDiscovery
// ---------------------------------------------------------------------------

/// Background discovery engine that finds inference peers on the LAN.
pub struct ClusterDiscovery {
    config: ClusterConfig,
    state: ClusterState,
    client: reqwest::Client,
}

impl ClusterDiscovery {
    /// Create a new discovery engine.
    ///
    /// The HTTP client is pre-configured with a short connect/request timeout.
    pub fn new(config: ClusterConfig, state: ClusterState) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(PROBE_TIMEOUT_SECS))
            .connect_timeout(Duration::from_secs(PROBE_TIMEOUT_SECS))
            .build()
            .unwrap_or_default();

        Self { config, state, client }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Run a single full discovery cycle.
    ///
    /// Order of operations (manual endpoints are probed FIRST so they are
    /// always available quickly, before the slower subnet scan completes):
    ///
    /// 1. Probe manually-configured `endpoints` from config.
    /// 2. If `auto_discover` is enabled: mDNS browse for Exo peers.
    /// 3. If `auto_discover` is enabled: HTTP probe the /24 subnet.
    ///
    /// All three sources are deduplicated and any not-yet-probed endpoints from
    /// steps 2–3 are probed in a final pass.
    pub async fn discover_once(&self) {
        let mut probed: std::collections::HashSet<String> = std::collections::HashSet::new();

        // 1. Manually-configured endpoints — probe immediately so they are
        //    available before the (potentially slow) subnet scan runs.
        //    If an endpoint has no explicit port, expand it across all scan_ports.
        let manual: Vec<String> = self
            .config
            .endpoints
            .iter()
            .flat_map(|ep| {
                let ep = ep.trim_end_matches('/').to_string();
                if extract_port(&ep).is_some() {
                    vec![ep]
                } else {
                    self.config.scan_ports.iter()
                        .map(|p| format!("{}:{}", ep, p))
                        .collect()
                }
            })
            .collect();

        if !manual.is_empty() {
            tracing::debug!(count = manual.len(), "cluster_discovery_probing_manual_endpoints");
            self.probe_all(manual.clone()).await;
            probed.extend(manual);
        }

        // Probe localhost on all scan ports (local servers are excluded from subnet scan).
        let localhost_endpoints: Vec<String> = self.config.scan_ports
            .iter()
            .map(|p| format!("http://127.0.0.1:{}", p))
            .filter(|ep| !probed.contains(ep))
            .collect();
        if !localhost_endpoints.is_empty() {
            tracing::debug!(count = localhost_endpoints.len(), "cluster_discovery_probing_localhost");
            self.probe_all(localhost_endpoints.clone()).await;
            probed.extend(localhost_endpoints);
        }

        if self.config.auto_discover {
            // 2. mDNS browse for Exo peers.
            let mdns_endpoints = self.browse_mdns().await;
            tracing::debug!(found = mdns_endpoints.len(), "mdns_browse_complete");

            // 3. HTTP probe the /24 subnet on configured ports.
            let subnet_ips = get_local_subnet();
            if subnet_ips.is_empty() {
                tracing::debug!("cluster_discovery: no LAN IP found, skipping subnet scan");
            } else {
                tracing::debug!(
                    ips = subnet_ips.len(),
                    ports = self.config.scan_ports.len(),
                    "cluster_subnet_scan_start"
                );
                let subnet_endpoints = self.scan_subnet(&subnet_ips).await;
                // scan_subnet already updates state for live peers; collect for dedup.
                let _ = subnet_endpoints;
            }

            // Probe any mDNS endpoints not already probed.
            let new_mdns: Vec<String> = mdns_endpoints
                .into_iter()
                .filter(|ep| !probed.contains(ep))
                .collect();
            if !new_mdns.is_empty() {
                self.probe_all(new_mdns).await;
            }
        }
    }

    /// Start the background discovery loop.
    ///
    /// Returns a `JoinHandle` that runs forever (or until the runtime shuts
    /// down). The loop:
    ///   - Calls `discover_once()` every `scan_interval_secs`.
    ///   - Removes peers not seen for 3× the scan interval.
    pub fn run(self) -> JoinHandle<()> {
        tokio::spawn(async move {
            let interval = Duration::from_secs(self.config.scan_interval_secs);
            let stale_age = Duration::from_secs(self.config.scan_interval_secs * 3);

            tracing::info!(
                interval_secs = self.config.scan_interval_secs,
                auto_discover = self.config.auto_discover,
                "cluster_discovery_loop_start"
            );

            loop {
                self.discover_once().await;
                self.state.remove_stale_peers(stale_age).await;

                tokio::time::sleep(interval).await;
            }
        })
    }

    // -----------------------------------------------------------------------
    // mDNS
    // -----------------------------------------------------------------------

    /// Browse mDNS for `_exo._tcp.local.` for up to `MDNS_BROWSE_TIMEOUT_SECS`.
    ///
    /// Returns a list of base endpoint URLs like `http://192.168.1.50:52415`.
    #[cfg(feature = "cluster")]
    async fn browse_mdns(&self) -> Vec<String> {
        let mut endpoints = Vec::new();

        let daemon = match ServiceDaemon::new() {
            Ok(d) => d,
            Err(e) => {
                tracing::debug!(error = %e, "cluster_mdns_daemon_init_failed");
                return endpoints;
            }
        };

        let receiver = match daemon.browse(EXO_SERVICE_TYPE) {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!(error = %e, "cluster_mdns_browse_failed");
                return endpoints;
            }
        };

        let browse_deadline = tokio::time::sleep(Duration::from_secs(MDNS_BROWSE_TIMEOUT_SECS));
        tokio::pin!(browse_deadline);

        loop {
            tokio::select! {
                _ = &mut browse_deadline => break,
                event = tokio::task::spawn_blocking({
                    let recv = receiver.clone();
                    move || recv.recv_timeout(Duration::from_millis(200))
                }) => {
                    match event {
                        Ok(Ok(ServiceEvent::ServiceResolved(info))) => {
                            let port = info.get_port();
                            for addr in info.get_addresses() {
                                let url = format!("http://{}:{}", addr, port);
                                tracing::info!(
                                    url = %url,
                                    host = info.get_hostname(),
                                    "cluster_mdns_peer_found"
                                );
                                endpoints.push(url);
                            }
                        }
                        Ok(Ok(ServiceEvent::SearchStopped(_))) => break,
                        Ok(Ok(_)) => {} // ServiceFound, SearchStarted, etc.
                        Ok(Err(_)) | Err(_) => {} // timeout or task error — keep polling
                    }
                }
            }
        }

        // Best-effort stop.
        let _ = daemon.stop_browse(EXO_SERVICE_TYPE);

        endpoints
    }

    /// Stub when feature is disabled (dead code under non-cluster builds).
    #[cfg(not(feature = "cluster"))]
    async fn browse_mdns(&self) -> Vec<String> {
        Vec::new()
    }

    // -----------------------------------------------------------------------
    // Subnet scan
    // -----------------------------------------------------------------------

    /// Build the list of `http://{ip}:{port}` candidates for every IP in
    /// `subnet_ips` × `self.config.scan_ports`.
    ///
    /// Then probe them in batches of `MAX_CONCURRENT_PROBES`.
    async fn scan_subnet(&self, subnet_ips: &[IpAddr]) -> Vec<String> {
        // Build the full candidate list.
        let mut candidates: Vec<String> = Vec::new();
        for ip in subnet_ips {
            for &port in &self.config.scan_ports {
                candidates.push(format!("http://{}:{}", ip, port));
            }
        }

        // Probe in batches to limit concurrency.
        let mut found_endpoints = Vec::new();
        for batch in candidates.chunks(MAX_CONCURRENT_PROBES) {
            let mut futs: FuturesUnordered<_> = batch
                .iter()
                .map(|ep| {
                    let ep = ep.clone();
                    let client = self.client.clone();
                    async move {
                        let peer = probe_endpoint_with_client(&client, &ep).await;
                        (ep, peer)
                    }
                })
                .collect();

            while let Some((ep, maybe_peer)) = futs.next().await {
                if let Some(peer) = maybe_peer {
                    tracing::info!(endpoint = %peer.endpoint, peer_type = %peer.peer_type, "cluster_subnet_peer_found");
                    self.state.update_peer(peer).await;
                    found_endpoints.push(ep);
                }
            }
        }

        found_endpoints
    }

    // -----------------------------------------------------------------------
    // Batch probe
    // -----------------------------------------------------------------------

    /// Probe a list of base endpoint URLs, updating state for each live peer.
    async fn probe_all(&self, endpoints: Vec<String>) {
        if endpoints.is_empty() {
            return;
        }

        let mut futs: FuturesUnordered<_> = endpoints
            .into_iter()
            .map(|ep| {
                let client = self.client.clone();
                async move {
                    let peer = probe_endpoint_with_client(&client, &ep).await;
                    peer
                }
            })
            .collect();

        while let Some(maybe_peer) = futs.next().await {
            if let Some(peer) = maybe_peer {
                self.state.update_peer(peer).await;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Free functions (pure-ish helpers — easy to unit-test)
// ---------------------------------------------------------------------------

/// Probe a single base endpoint (e.g. `http://192.168.1.50:52415`) by
/// issuing `GET {endpoint}/v1/models`.
///
/// Returns `Some(ClusterPeer)` if the server responded with a valid model
/// list, `None` otherwise.
pub async fn probe_endpoint(endpoint: &str) -> Option<ClusterPeer> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(PROBE_TIMEOUT_SECS))
        .build()
        .unwrap_or_default();
    probe_endpoint_with_client(&client, endpoint).await
}

/// Inner probe that reuses a shared `reqwest::Client`.
async fn probe_endpoint_with_client(
    client: &reqwest::Client,
    endpoint: &str,
) -> Option<ClusterPeer> {
    let url = format!("{}/v1/models", endpoint.trim_end_matches('/'));

    // JAN (port 1337) validates the Host header and rejects requests from
    // remote IPs. Sending `Host: localhost` works around this.
    let mut req = client.get(&url);
    if extract_port(endpoint) == Some(1337) {
        req = req.header("Host", "localhost");
    }
    let result = timeout(Duration::from_secs(PROBE_TIMEOUT_SECS), req.send()).await;

    match result {
        Ok(Ok(resp)) => {
            let port = extract_port(endpoint);
            let server_header = resp
                .headers()
                .get("server")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_ascii_lowercase();

            let body: serde_json::Value = match resp.json().await {
                Ok(v) => v,
                Err(e) => {
                    tracing::debug!(endpoint = %endpoint, error = %e, "cluster_probe_parse_failed");
                    return None;
                }
            };

            let models = parse_models_response(&body);
            let peer_type = detect_peer_type(port, &server_header, &models);

            tracing::debug!(
                endpoint = %endpoint,
                peer_type = %peer_type,
                models = models.len(),
                "cluster_probe_success"
            );

            Some(ClusterPeer {
                endpoint: format!("{}/v1", endpoint.trim_end_matches('/')),
                peer_type,
                models,
                total_vram_mb: None,
                last_seen: Instant::now(),
                healthy: true,
            })
        }
        Ok(Err(e)) => {
            tracing::debug!(endpoint = %endpoint, error = %e, "cluster_probe_connect_failed");
            None
        }
        Err(_) => {
            tracing::debug!(endpoint = %endpoint, "cluster_probe_timeout");
            None
        }
    }
}

/// Parse the OpenAI `/v1/models` response into a `Vec<ClusterModel>`.
///
/// Expected shape: `{"data": [{"id": "model-name", ...}, ...]}`
pub fn parse_models_response(body: &serde_json::Value) -> Vec<ClusterModel> {
    body["data"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|entry| {
                    let id = entry["id"].as_str()?.to_string();
                    Some(ClusterModel {
                        id,
                        context_window: 0,
                        requires_cluster: false,
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Heuristic to identify the type of inference server.
///
/// - Port 52415 → Exo
/// - Port 1234  → LM Studio
/// - Port 8080  → llama.cpp
/// - `server` header containing "lm-studio" → LM Studio
/// - Model IDs with Exo-style shard names → Exo
/// - Default → Unknown
pub fn detect_peer_type(
    port: Option<u16>,
    server_header: &str,
    models: &[ClusterModel],
) -> PeerType {
    // Port-based heuristics.
    if let Some(p) = port {
        match p {
            52415 => return PeerType::Exo,
            1234 => return PeerType::LMStudio,
            8080 => return PeerType::LlamaCpp,
            1337 => return PeerType::Jan,
            _ => {}
        }
    }

    // Server header.
    if server_header.contains("lm-studio") || server_header.contains("lmstudio") {
        return PeerType::LMStudio;
    }
    if server_header.contains("llama") || server_header.contains("llama.cpp") {
        return PeerType::LlamaCpp;
    }
    if server_header.contains("jan") {
        return PeerType::Jan;
    }

    // Model ID patterns: Exo uses shard notation like "llama-3-8b-4bit-q4_k_m".
    for model in models {
        let id = model.id.to_ascii_lowercase();
        if id.contains("shard") || id.contains("exo") {
            return PeerType::Exo;
        }
    }

    PeerType::Unknown
}

/// Extract the port number from a URL like `http://192.168.1.5:52415`.
fn extract_port(url: &str) -> Option<u16> {
    // Strip scheme.
    let rest = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .unwrap_or(url);

    // host:port or host:port/path
    let host_port = rest.split('/').next()?;
    host_port.rsplit(':').next()?.parse::<u16>().ok()
}

/// Return up to 254 IPs in the host's primary /24 LAN subnet.
///
/// Iterates network interfaces looking for the first RFC 1918 address
/// (`10.x`, `172.16-31.x`, `192.168.x`). Returns all .1–.254 addresses
/// in that /24, excluding the host's own IP.
///
/// Returns an empty `Vec` when no LAN interface is found.
pub fn get_local_subnet() -> Vec<IpAddr> {
    if let Some(ip) = find_lan_ip() {
        return build_subnet_ips(ip);
    }
    Vec::new()
}

/// Find the host's RFC-1918 IPv4 address, preferring the interface that holds
/// the default route so that Docker bridge IPs (172.17.x, 172.18.x …) are
/// skipped in favour of the real LAN address.
///
/// Strategy:
/// 1. Parse `/proc/net/route` to find the default-route gateway IP.
/// 2. Collect all RFC-1918 IPs from `/proc/net/fib_trie`.
/// 3. Return the candidate on the same /20 subnet as the gateway.
/// 4. Fall back to the first RFC-1918 IP from fib_trie (original behaviour).
fn find_lan_ip() -> Option<std::net::Ipv4Addr> {
    let route_content = std::fs::read_to_string("/proc/net/route").ok();
    let fib_content = std::fs::read_to_string("/proc/net/fib_trie").ok();

    if let (Some(ref route), Some(ref fib)) = (&route_content, &fib_content) {
        // Find the default-route interface name.
        if let Some(iface) = default_route_interface(route) {
            // Find the gateway IP for that interface.
            if let Some(gw) = parse_gateway_ip(route, &iface) {
                // Collect all RFC-1918 candidates from fib_trie.
                let candidates = parse_all_lan_ips_from_fib_trie(fib);
                let gw_oct = gw.octets();
                // Prefer a candidate on the same /20 as the gateway
                // (same first two octets, same upper nibble of third octet).
                let preferred = candidates.into_iter().find(|ip| {
                    let o = ip.octets();
                    o[0] == gw_oct[0]
                        && o[1] == gw_oct[1]
                        && (o[2] >> 4) == (gw_oct[2] >> 4)
                });
                if preferred.is_some() {
                    return preferred;
                }
            }
        }
    }

    // Fallback: first RFC-1918 from fib_trie (original behaviour).
    let content = fib_content
        .or_else(|| std::fs::read_to_string("/proc/net/fib_trie").ok());

    #[cfg(target_os = "linux")]
    {
        return content.and_then(|c| parse_lan_ip_from_fib_trie(&c));
    }

    // BSD/macOS fallback: use route + ifconfig commands.
    #[cfg(not(target_os = "linux"))]
    {
        if let Some(ref c) = content {
            if let Some(ip) = parse_lan_ip_from_fib_trie(c) {
                return Some(ip);
            }
        }
        return find_lan_ip_bsd();
    }
}

/// macOS/BSD fallback: use `route` and `ifconfig` commands to find the host's
/// RFC-1918 IPv4 address on the default-route interface.
#[cfg(not(target_os = "linux"))]
fn find_lan_ip_bsd() -> Option<std::net::Ipv4Addr> {
    use std::process::Command;

    // Get default route info.
    let route_out = Command::new("route")
        .args(["-n", "get", "default"])
        .output()
        .ok()?;
    let route_str = String::from_utf8_lossy(&route_out.stdout);

    // Parse gateway IP.
    let gateway_line = route_str.lines().find(|l| l.trim().starts_with("gateway:"))?;
    let _gateway_ip: std::net::Ipv4Addr = gateway_line
        .trim()
        .strip_prefix("gateway:")?
        .trim()
        .parse()
        .ok()?;

    // Parse interface name.
    let iface_line = route_str.lines().find(|l| l.trim().starts_with("interface:"))?;
    let iface = iface_line.trim().strip_prefix("interface:")?.trim();

    // Get our IP on that interface.
    let ifconfig_out = Command::new("ifconfig").arg(iface).output().ok()?;
    let ifconfig_str = String::from_utf8_lossy(&ifconfig_out.stdout);

    // Find "inet X.X.X.X" line (skip inet6 lines).
    for line in ifconfig_str.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("inet ") && !trimmed.contains("inet6") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(ip) = parts[1].parse::<std::net::Ipv4Addr>() {
                    if is_rfc1918(ip) {
                        return Some(ip);
                    }
                }
            }
        }
    }

    None
}

/// Parse the interface name of the default route from `/proc/net/route` content.
///
/// The file format is tab-separated:
/// ```text
/// Iface   Destination  Gateway  Flags ...
/// eth0    00000000     0101A8C0 0003  ...
/// ```
/// A destination of `00000000` means the default route.
pub fn default_route_interface(route_content: &str) -> Option<String> {
    for line in route_content.lines().skip(1) {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() >= 2 && fields[1] == "00000000" {
            return Some(fields[0].trim().to_string());
        }
    }
    None
}

/// Parse the gateway IP for `iface`'s default route from `/proc/net/route`.
///
/// The Gateway column is a hex-encoded IPv4 in little-endian (native) byte
/// order on x86 Linux. `u32::from_str_radix` gives the LE value; calling
/// `to_be()` swaps it into network order, which `Ipv4Addr::from` expects.
pub fn parse_gateway_ip(route_content: &str, iface: &str) -> Option<std::net::Ipv4Addr> {
    for line in route_content.lines().skip(1) {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() >= 3 && fields[0].trim() == iface && fields[1] == "00000000" {
            let hex = fields[2].trim();
            let val = u32::from_str_radix(hex, 16).ok()?;
            return Some(std::net::Ipv4Addr::from(val.to_be()));
        }
    }
    None
}

/// Collect ALL RFC-1918 local host IPs from `/proc/net/fib_trie` content.
///
/// Like `parse_lan_ip_from_fib_trie` but returns every candidate rather than
/// stopping at the first one.
pub fn parse_all_lan_ips_from_fib_trie(content: &str) -> Vec<std::net::Ipv4Addr> {
    let lines: Vec<&str> = content.lines().collect();
    let mut result = Vec::new();
    let mut i = 0;
    while i + 1 < lines.len() {
        if let Some(token) = lines[i].split_whitespace().last() {
            if token.contains('.') && !token.contains('/') {
                if let Ok(ip) = token.parse::<std::net::Ipv4Addr>() {
                    let next = lines[i + 1];
                    if next.contains("/32 host") && next.contains("LOCAL") && is_rfc1918(ip) {
                        result.push(ip);
                    }
                }
            }
        }
        i += 1;
    }
    result
}

/// Pure function: parse the first RFC-1918 IP from `/proc/net/fib_trie` content.
///
/// The file has repeated blocks like:
/// ```text
///    |-- 192.168.1.100
///          /32 host LOCAL
/// ```
/// IP addresses appear on lines like `   |-- 192.168.1.100` or as bare `192.168.1.100`.
/// The immediately following line contains `/32 host LOCAL` for host entries.
pub fn parse_lan_ip_from_fib_trie(content: &str) -> Option<std::net::Ipv4Addr> {
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;
    while i + 1 < lines.len() {
        // Extract the last whitespace-separated token on the line.
        // This handles both bare "192.168.1.100" and "|-- 192.168.1.100" forms.
        if let Some(token) = lines[i].split_whitespace().last() {
            if token.contains('.') && !token.contains('/') {
                if let Ok(ip) = token.parse::<std::net::Ipv4Addr>() {
                    // The very next line must contain "/32 host" and "LOCAL".
                    let next = lines[i + 1];
                    if next.contains("/32 host") && next.contains("LOCAL") && is_rfc1918(ip) {
                        return Some(ip);
                    }
                }
            }
        }
        i += 1;
    }
    None
}

/// Returns true if `ip` is an RFC 1918 private address.
pub fn is_rfc1918(ip: std::net::Ipv4Addr) -> bool {
    let octets = ip.octets();
    match octets[0] {
        10 => true,
        172 => (16..=31).contains(&octets[1]),
        192 => octets[1] == 168,
        _ => false,
    }
}

/// Build the list of all .1–.254 IPs in the /24 containing `host_ip`.
pub fn build_subnet_ips(host_ip: std::net::Ipv4Addr) -> Vec<IpAddr> {
    let [a, b, c, _] = host_ip.octets();
    (1u8..=254)
        .filter(|&d| {
            // Skip the host's own address.
            std::net::Ipv4Addr::new(a, b, c, d) != host_ip
        })
        .map(|d| IpAddr::V4(std::net::Ipv4Addr::new(a, b, c, d)))
        .collect()
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- parse_models_response ---

    #[test]
    fn test_parse_models_response_normal() {
        let json = serde_json::json!({
            "data": [
                {"id": "qwen3-72b-q4", "object": "model"},
                {"id": "llama-3.1-8b", "object": "model"},
            ]
        });
        let models = parse_models_response(&json);
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "qwen3-72b-q4");
        assert_eq!(models[1].id, "llama-3.1-8b");
    }

    #[test]
    fn test_parse_models_response_empty_data() {
        let json = serde_json::json!({"data": []});
        assert_eq!(parse_models_response(&json).len(), 0);
    }

    #[test]
    fn test_parse_models_response_missing_data() {
        let json = serde_json::json!({"error": "not found"});
        assert_eq!(parse_models_response(&json).len(), 0);
    }

    #[test]
    fn test_parse_models_response_entry_missing_id() {
        let json = serde_json::json!({
            "data": [{"object": "model"}, {"id": "valid-model"}]
        });
        let models = parse_models_response(&json);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "valid-model");
    }

    // --- detect_peer_type ---

    #[test]
    fn test_detect_peer_type_exo_port() {
        assert_eq!(
            detect_peer_type(Some(52415), "", &[]),
            PeerType::Exo
        );
    }

    #[test]
    fn test_detect_peer_type_lmstudio_port() {
        assert_eq!(
            detect_peer_type(Some(1234), "", &[]),
            PeerType::LMStudio
        );
    }

    #[test]
    fn test_detect_peer_type_llamacpp_port() {
        assert_eq!(
            detect_peer_type(Some(8080), "", &[]),
            PeerType::LlamaCpp
        );
    }

    #[test]
    fn test_detect_peer_type_lmstudio_header() {
        assert_eq!(
            detect_peer_type(None, "lm-studio/1.0", &[]),
            PeerType::LMStudio
        );
    }

    #[test]
    fn test_detect_peer_type_llamacpp_header() {
        assert_eq!(
            detect_peer_type(None, "llama.cpp/b3000", &[]),
            PeerType::LlamaCpp
        );
    }

    #[test]
    fn test_detect_peer_type_exo_model_id() {
        let models = vec![ClusterModel {
            id: "llama-3.1-70b-shard-0".to_string(),
            context_window: 0,
            requires_cluster: false,
        }];
        assert_eq!(
            detect_peer_type(Some(9999), "", &models),
            PeerType::Exo
        );
    }

    #[test]
    fn test_detect_peer_type_jan_port() {
        assert_eq!(
            detect_peer_type(Some(1337), "", &[]),
            PeerType::Jan
        );
    }

    #[test]
    fn test_detect_peer_type_jan_header() {
        assert_eq!(
            detect_peer_type(None, "jan/0.5.3", &[]),
            PeerType::Jan
        );
    }

    #[test]
    fn test_detect_peer_type_unknown() {
        assert_eq!(detect_peer_type(Some(9999), "nginx", &[]), PeerType::Unknown);
    }

    #[test]
    fn test_detect_peer_type_no_port() {
        assert_eq!(detect_peer_type(None, "", &[]), PeerType::Unknown);
    }

    // --- extract_port ---

    #[test]
    fn test_extract_port_http() {
        assert_eq!(extract_port("http://192.168.1.5:52415"), Some(52415));
    }

    #[test]
    fn test_extract_port_https() {
        assert_eq!(extract_port("https://192.168.1.5:8443"), Some(8443));
    }

    #[test]
    fn test_extract_port_with_path() {
        assert_eq!(extract_port("http://192.168.1.5:1234/v1"), Some(1234));
    }

    #[test]
    fn test_extract_port_no_port() {
        // Without explicit port, no port extracted.
        assert_eq!(extract_port("http://192.168.1.5"), None);
    }

    // --- is_rfc1918 ---

    #[test]
    fn test_is_rfc1918_10_block() {
        assert!(is_rfc1918("10.0.0.1".parse().unwrap()));
        assert!(is_rfc1918("10.255.255.255".parse().unwrap()));
    }

    #[test]
    fn test_is_rfc1918_172_block() {
        assert!(is_rfc1918("172.16.0.1".parse().unwrap()));
        assert!(is_rfc1918("172.31.255.255".parse().unwrap()));
        assert!(!is_rfc1918("172.15.0.1".parse().unwrap()));
        assert!(!is_rfc1918("172.32.0.1".parse().unwrap()));
    }

    #[test]
    fn test_is_rfc1918_192_168_block() {
        assert!(is_rfc1918("192.168.0.1".parse().unwrap()));
        assert!(is_rfc1918("192.168.100.50".parse().unwrap()));
        assert!(!is_rfc1918("192.167.0.1".parse().unwrap()));
        assert!(!is_rfc1918("192.169.0.1".parse().unwrap()));
    }

    #[test]
    fn test_is_rfc1918_public() {
        assert!(!is_rfc1918("8.8.8.8".parse().unwrap()));
        assert!(!is_rfc1918("1.1.1.1".parse().unwrap()));
    }

    // --- build_subnet_ips ---

    #[test]
    fn test_build_subnet_ips_count() {
        let host: std::net::Ipv4Addr = "192.168.1.100".parse().unwrap();
        let ips = build_subnet_ips(host);
        // 1..=254 minus the host itself = 253 addresses.
        assert_eq!(ips.len(), 253);
    }

    #[test]
    fn test_build_subnet_ips_excludes_host() {
        let host: std::net::Ipv4Addr = "192.168.1.100".parse().unwrap();
        let ips = build_subnet_ips(host);
        assert!(!ips.contains(&IpAddr::V4(host)));
    }

    #[test]
    fn test_build_subnet_ips_range() {
        let host: std::net::Ipv4Addr = "10.0.0.50".parse().unwrap();
        let ips = build_subnet_ips(host);
        assert!(ips.contains(&IpAddr::V4("10.0.0.1".parse().unwrap())));
        assert!(ips.contains(&IpAddr::V4("10.0.0.254".parse().unwrap())));
        assert!(!ips.contains(&IpAddr::V4("10.0.0.255".parse().unwrap())));
    }

    // --- parse_lan_ip_from_fib_trie ---

    #[test]
    fn test_parse_lan_ip_from_fib_trie_finds_192_168() {
        // Realistic /proc/net/fib_trie snippet: IP line followed by "/32 host LOCAL" on same line.
        let content = "\
Main:\n\
  +-- 0.0.0.0/0 3 0 5\n\
     +-- 192.168.1.0/24 2 0 2\n\
        |-- 192.168.1.0\n\
              /24 link UNICAST\n\
        |-- 192.168.1.100\n\
              /32 host LOCAL\n\
     |-- 127.0.0.1\n\
           /32 host LOCAL\n";
        let ip = parse_lan_ip_from_fib_trie(content).unwrap();
        assert_eq!(ip, "192.168.1.100".parse::<std::net::Ipv4Addr>().unwrap());
    }

    #[test]
    fn test_parse_lan_ip_from_fib_trie_skips_loopback() {
        // Only loopback present — should return None.
        let content = "\
     |-- 127.0.0.1\n\
           /32 host LOCAL\n";
        assert!(parse_lan_ip_from_fib_trie(content).is_none());
    }

    #[test]
    fn test_parse_lan_ip_from_fib_trie_empty() {
        assert!(parse_lan_ip_from_fib_trie("").is_none());
    }

    #[test]
    fn test_parse_lan_ip_from_fib_trie_10_block() {
        let content = "\
     |-- 10.0.0.42\n\
           /32 host LOCAL\n";
        let ip = parse_lan_ip_from_fib_trie(content).unwrap();
        assert_eq!(ip, "10.0.0.42".parse::<std::net::Ipv4Addr>().unwrap());
    }

    // --- default_route_interface ---

    #[test]
    fn test_default_route_interface_parses_eth0() {
        // Typical /proc/net/route content with eth0 as default gateway.
        let content = "\
Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT\n\
eth0\t00000000\tC01AA8C0\t0003\t0\t0\t100\t00000000\t0\t0\t0\n\
eth0\t001AA8C0\t00000000\t0001\t0\t0\t100\tF0FFFFFF\t0\t0\t0\n\
docker0\t000011AC\t00000000\t0001\t0\t0\t0\t00FFFF00\t0\t0\t0\n";
        let iface = default_route_interface(content).unwrap();
        assert_eq!(iface, "eth0");
    }

    #[test]
    fn test_default_route_interface_returns_none_when_no_default() {
        // No 00000000 destination present.
        let content = "\
Iface\tDestination\tGateway\tFlags\n\
eth0\t001AA8C0\t00000000\t0001\n";
        assert!(default_route_interface(content).is_none());
    }

    // --- parse_gateway_ip ---

    #[test]
    fn test_parse_gateway_ip_eth0() {
        // Gateway C01AA8C0 little-endian -> 192.168.26.192 (0xC0 = 192, 0x1A = 26, 0xA8 = 168, 0xC0 = 192 reversed)
        // Actually: little-endian u32 C01AA8C0 = bytes [C0, A8, 1A, C0] -> 192.168.26.192.
        // to_be() swaps all 4 bytes: C0 A8 1A C0 -> stored as 0xC0A81AC0 -> Ipv4Addr::from gives 192.168.26.192.
        let content = "\
Iface\tDestination\tGateway\tFlags\n\
eth0\t00000000\tC01AA8C0\t0003\n\
eth0\t001AA8C0\t00000000\t0001\n";
        let gw = parse_gateway_ip(content, "eth0").unwrap();
        // 0xC01AA8C0 to_be = swap bytes of u32 C01AA8C0:
        // C0=192, 1A=26, A8=168, C0=192 → the u32 in memory is 0xC01AA8C0 (LE: C0 A8 1A C0),
        // to_be() reverses byte order: C0 A8 1A C0 → Ipv4Addr [192, 168, 26, 192]
        assert_eq!(gw, "192.168.26.192".parse::<std::net::Ipv4Addr>().unwrap());
    }

    #[test]
    fn test_parse_gateway_ip_returns_none_for_wrong_iface() {
        let content = "\
Iface\tDestination\tGateway\tFlags\n\
eth0\t00000000\tC01AA8C0\t0003\n";
        assert!(parse_gateway_ip(content, "wlan0").is_none());
    }

    // --- parse_all_lan_ips_from_fib_trie ---

    #[test]
    fn test_parse_all_lan_ips_returns_all_candidates() {
        // Docker bridge 172.17.0.1 comes before real LAN IP 172.26.28.187.
        let content = "\
     |-- 172.17.0.1\n\
           /32 host LOCAL\n\
     |-- 172.26.28.187\n\
           /32 host LOCAL\n\
     |-- 127.0.0.1\n\
           /32 host LOCAL\n";
        let ips = parse_all_lan_ips_from_fib_trie(content);
        assert_eq!(ips.len(), 2);
        assert!(ips.contains(&"172.17.0.1".parse::<std::net::Ipv4Addr>().unwrap()));
        assert!(ips.contains(&"172.26.28.187".parse::<std::net::Ipv4Addr>().unwrap()));
        // 127.0.0.1 is not RFC-1918, must be excluded.
        assert!(!ips.contains(&"127.0.0.1".parse::<std::net::Ipv4Addr>().unwrap()));
    }

    #[test]
    fn test_parse_all_lan_ips_empty_content() {
        assert!(parse_all_lan_ips_from_fib_trie("").is_empty());
    }

    // --- find_lan_ip_prefers_gateway_subnet ---

    #[test]
    fn test_find_lan_ip_prefers_gateway_subnet() {
        // Simulate: Docker bridge 172.17.0.1 and real LAN 172.26.28.187 both in fib_trie.
        // Gateway is 172.26.26.1 (same /20 as 172.26.28.187, i.e. 172.26.16.x–172.26.31.x).
        // default_route_interface returns "eth0"; parse_gateway_ip returns 172.26.26.1.

        let fib = "\
     |-- 172.17.0.1\n\
           /32 host LOCAL\n\
     |-- 172.26.28.187\n\
           /32 host LOCAL\n";

        let candidates = parse_all_lan_ips_from_fib_trie(fib);
        let gw: std::net::Ipv4Addr = "172.26.26.1".parse().unwrap();
        let gw_oct = gw.octets();

        let preferred = candidates.into_iter().find(|ip| {
            let o = ip.octets();
            o[0] == gw_oct[0]
                && o[1] == gw_oct[1]
                && (o[2] >> 4) == (gw_oct[2] >> 4)
        });

        assert_eq!(
            preferred.unwrap(),
            "172.26.28.187".parse::<std::net::Ipv4Addr>().unwrap()
        );
    }

    #[test]
    fn test_find_lan_ip_gateway_subnet_skips_docker() {
        // Gateway is 192.168.1.1 — Docker 172.17.0.1 should be skipped.
        let fib = "\
     |-- 172.17.0.1\n\
           /32 host LOCAL\n\
     |-- 192.168.1.50\n\
           /32 host LOCAL\n";

        let candidates = parse_all_lan_ips_from_fib_trie(fib);
        let gw: std::net::Ipv4Addr = "192.168.1.1".parse().unwrap();
        let gw_oct = gw.octets();

        let preferred = candidates.into_iter().find(|ip| {
            let o = ip.octets();
            o[0] == gw_oct[0]
                && o[1] == gw_oct[1]
                && (o[2] >> 4) == (gw_oct[2] >> 4)
        });

        assert_eq!(
            preferred.unwrap(),
            "192.168.1.50".parse::<std::net::Ipv4Addr>().unwrap()
        );
    }

    // --- find_lan_ip_bsd route parsing ---

    #[test]
    fn test_parse_bsd_route_output() {
        let route_output = "   route to: default\ndestination: default\n       mask: default\n    gateway: 192.168.1.1\n  interface: en0\n";

        let gateway_line = route_output
            .lines()
            .find(|l| l.trim().starts_with("gateway:"))
            .unwrap();
        let gw: std::net::Ipv4Addr = gateway_line
            .trim()
            .strip_prefix("gateway:")
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert_eq!(gw, std::net::Ipv4Addr::new(192, 168, 1, 1));

        let iface_line = route_output
            .lines()
            .find(|l| l.trim().starts_with("interface:"))
            .unwrap();
        let iface = iface_line
            .trim()
            .strip_prefix("interface:")
            .unwrap()
            .trim();
        assert_eq!(iface, "en0");
    }

    // --- integration: ClusterDiscovery wiring ---

    #[tokio::test]
    async fn test_discover_once_no_auto_discover_no_endpoints() {
        // With auto_discover=false and no endpoints, discover_once probes
        // localhost on scan_ports (which fail) and leaves state empty.
        // Use unlikely ports to avoid hitting real local servers.
        let config = crate::config::schema::ClusterConfig {
            enabled: true,
            auto_discover: false,
            endpoints: vec![],
            scan_ports: vec![59999, 59998],
            scan_interval_secs: 60,
            prefer_cluster: true,
        };
        let state = ClusterState::new();
        let discovery = ClusterDiscovery::new(config, state.clone());
        discovery.discover_once().await;
        assert_eq!(state.peer_count().await, 0);
    }

    #[test]
    fn test_manual_endpoint_without_port_expands_to_scan_ports() {
        // When a manual endpoint has no port, it should be expanded
        // across all scan_ports during discovery.
        let config = crate::config::schema::ClusterConfig {
            enabled: true,
            auto_discover: false,
            endpoints: vec!["http://192.168.1.22".to_string()],
            scan_ports: vec![1234, 1337, 8080],
            scan_interval_secs: 60,
            prefer_cluster: true,
        };
        // Verify the expansion logic directly.
        let expanded: Vec<String> = config
            .endpoints
            .iter()
            .flat_map(|ep| {
                let ep = ep.trim_end_matches('/').to_string();
                if extract_port(&ep).is_some() {
                    vec![ep]
                } else {
                    config.scan_ports.iter()
                        .map(|p| format!("{}:{}", ep, p))
                        .collect()
                }
            })
            .collect();
        assert_eq!(expanded, vec![
            "http://192.168.1.22:1234",
            "http://192.168.1.22:1337",
            "http://192.168.1.22:8080",
        ]);
    }

    #[test]
    fn test_manual_endpoint_with_port_not_expanded() {
        let config = crate::config::schema::ClusterConfig {
            enabled: true,
            auto_discover: false,
            endpoints: vec!["http://192.168.1.22:5000".to_string()],
            scan_ports: vec![1234, 1337],
            scan_interval_secs: 60,
            prefer_cluster: true,
        };
        let expanded: Vec<String> = config
            .endpoints
            .iter()
            .flat_map(|ep| {
                let ep = ep.trim_end_matches('/').to_string();
                if extract_port(&ep).is_some() {
                    vec![ep]
                } else {
                    config.scan_ports.iter()
                        .map(|p| format!("{}:{}", ep, p))
                        .collect()
                }
            })
            .collect();
        assert_eq!(expanded, vec!["http://192.168.1.22:5000"]);
    }
}
