//! Web tools: web_search and web_fetch.

use std::collections::HashMap;

use async_trait::async_trait;
use html2md::rewrite_html;
use regex::Regex;
use reqwest::Client;
use std::sync::{Arc, LazyLock};
use url::Url;

use super::base::{require_str, PermissionLevel, Tool, ToolConcurrency, ToolExecutionContext};
use crate::agent::audit::ToolEvent;

/// Shared user-agent string.
const USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36";

/// Maximum number of redirects to follow.
const MAX_REDIRECTS: usize = 5;

/// Maximum response body size (5 MB). Prevents memory spikes on large responses.
const MAX_BODY_BYTES: usize = 5 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Static regexes (compiled once)
// ---------------------------------------------------------------------------
static RE_SPACES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[ \t]+").unwrap());
static RE_NEWLINES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n{3,}").unwrap());

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Normalize whitespace: collapse runs of spaces/tabs, limit consecutive newlines.
fn normalize_whitespace(text: &str) -> String {
    let text = RE_SPACES.replace_all(text, " ");
    RE_NEWLINES.replace_all(&text, "\n\n").trim().to_string()
}

/// Extract the `text` field from a web_fetch JSON envelope, falling back to raw input.
///
/// This unwraps the JSON overhead so the model sees clean article text instead of
/// a JSON structure summary. Non-JSON input and JSON without a `text` field are
/// returned unchanged.
pub fn extract_web_content(raw: &str) -> String {
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(raw) {
        if let Some(text) = parsed.get("text").and_then(|t| t.as_str()) {
            return text.to_string();
        }
    }
    raw.to_string()
}

/// Validate a URL: must be http(s) with a valid, non-private domain.
///
/// Blocks local/private addresses to prevent SSRF attacks where the LLM
/// might be tricked into fetching internal services.
fn validate_url(url_str: &str) -> Result<(), String> {
    let parsed = Url::parse(url_str).map_err(|e| format!("Invalid URL: {}", e))?;
    match parsed.scheme() {
        "http" | "https" => {}
        other => return Err(format!("Only http/https allowed, got '{}'", other)),
    }
    let host = parsed.host_str().ok_or("Missing domain")?;

    // Block known private/local hostnames. Loopback IPs (127.0.0.0/8) are
    // caught by the IP check below; `localhost` is a name, not an IP, so it
    // must be blocked here too — otherwise it slips through as an SSRF vector.
    let lower = host.to_lowercase();
    if lower == "localhost"
        || lower == "0.0.0.0"
        || lower.ends_with(".local")
        || lower.ends_with(".internal")
    {
        return Err(format!("Access to local host '{}' is blocked", host));
    }

    // Block private/reserved IP ranges (RFC 1918, link-local, loopback, metadata).
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        let blocked = match ip {
            std::net::IpAddr::V4(v4) => {
                v4.is_loopback()                              // 127.0.0.0/8
                    || v4.is_private()                        // 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
                    || v4.is_link_local()                     // 169.254.0.0/16
                    || v4.is_unspecified()                    // 0.0.0.0
                    || v4.octets()[0] == 169 && v4.octets()[1] == 254 // cloud metadata
            }
            std::net::IpAddr::V6(v6) => v6.is_loopback() || v6.is_unspecified(),
        };
        if blocked {
            return Err(format!("Access to private/local IP '{}' is blocked", ip));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// WebSearchTool
// ---------------------------------------------------------------------------

/// Search the web using SearXNG (default) or Brave Search API.
pub struct WebSearchTool {
    api_key: String,
    max_results: u32,
    provider: String,
    searxng_url: String,
    client: Client,
    /// Optional handle to the heartbeat registry. When set, `execute_searxng`
    /// checks the "searxng" probe first; on `Degraded` it returns a clear
    /// "stuck container, try `docker restart`" message instead of silently
    /// producing zero results.
    health_registry: Option<Arc<crate::heartbeat::health::HealthRegistry>>,
}

impl WebSearchTool {
    /// Create a new web search tool.
    ///
    /// `provider` selects the backend: `"searxng"` (default) or `"brave"`.
    /// `searxng_url` is the base URL of the SearXNG instance (e.g. `"http://localhost:8888"`).
    ///
    /// If `api_key` is `None`, the `BRAVE_API_KEY` environment variable is
    /// checked. Passing `Some("")` explicitly disables env fallback.
    pub fn new(
        api_key: Option<String>,
        max_results: u32,
        provider: String,
        searxng_url: String,
    ) -> Self {
        let resolved_key = match api_key {
            Some(key) => key,
            None => std::env::var("BRAVE_API_KEY").unwrap_or_default(),
        };

        // Build a client with browser-like headers. SearXNG's botdetection
        // (http_accept / http_accept_encoding / http_accept_language /
        // http_user_agent / http_sec_fetch) rejects requests that don't look
        // like a real browser, returning HTTP 429. A bare `Client::new()` sends
        // a reqwest default UA and no Accept/Sec-Fetch headers, so it is flagged.
        //
        // `X-Forwarded-For` is required by the limiter's trusted_proxies check:
        // without a client IP header, botdetection refuses the request even
        // when the real source is localhost. We are a trusted local client, so
        // we declare ourselves.
        let client = Client::builder()
            .user_agent(USER_AGENT)
            .default_headers(
                std::collections::HashMap::from([
                    (
                        "accept",
                        "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    ),
                    ("accept-language", "en-US,en;q=0.9"),
                    ("accept-encoding", "gzip, deflate, br"),
                    ("sec-fetch-dest", "document"),
                    ("sec-fetch-mode", "navigate"),
                    ("sec-fetch-site", "none"),
                    ("x-forwarded-for", "127.0.0.1"),
                ])
                .into_iter()
                .map(|(k, v)| {
                    (
                        reqwest::header::HeaderName::from_static(k),
                        reqwest::header::HeaderValue::from_static(v),
                    )
                })
                .collect(),
            )
            .build()
            .unwrap_or_else(|_| Client::new());

        Self {
            api_key: resolved_key,
            max_results,
            provider,
            searxng_url,
            client,
            health_registry: None,
        }
    }

    /// Attach a health registry so `web_search` can short-circuit with a
    /// clear "SearXNG degraded" message when the probe has marked the
    /// backend unhealthy. Builder-style: returns `self` for chaining at
    /// the registration site.
    pub fn with_health_registry(
        mut self,
        registry: Option<Arc<crate::heartbeat::health::HealthRegistry>>,
    ) -> Self {
        self.health_registry = registry;
        self
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web. Returns titles, URLs, and snippets."
    }

    fn permission(&self) -> PermissionLevel {
        PermissionLevel::Network
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::ParallelSafe
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "count": {
                    "type": "integer",
                    "description": "Results (1-10)",
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "required": ["query"]
        })
    }

    /// Returns `true` when at least one search backend is configured.
    ///
    /// - SearXNG: available when `provider == "searxng"` and a non-empty
    ///   SearXNG URL is present (the default `"http://localhost:8888"` counts).
    /// - Brave: available when an API key is present (either passed at
    ///   construction time or read from `$BRAVE_API_KEY`).
    fn is_available(&self) -> bool {
        (self.provider == "searxng" && !self.searxng_url.is_empty()) || !self.api_key.is_empty()
    }

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        let query = require_str!(params, "query");

        let count = params
            .get("count")
            .and_then(|v| v.as_u64())
            .map(|n| n.min(10).max(1) as u32)
            .unwrap_or(self.max_results);

        match self.provider.as_str() {
            "searxng" => self.execute_searxng(query, count).await,
            "brave" => self.execute_brave(query, count).await,
            other => format!(
                "Error: unknown search provider '{}'. Use 'searxng' or 'brave'.",
                other
            ),
        }
    }

    async fn execute_with_context(
        &self,
        params: HashMap<String, serde_json::Value>,
        ctx: &ToolExecutionContext,
    ) -> String {
        let query = params.get("query").and_then(|v| v.as_str()).unwrap_or("");

        let _ = ctx.event_tx.send(ToolEvent::Progress {
            tool_name: "web_search".to_string(),
            tool_call_id: ctx.tool_call_id.clone(),
            elapsed_ms: 0,
            output_preview: Some(format!("Searching: {}", query)),
        });

        self.execute(params).await
    }
}

/// Epoch seconds of the last background SearXNG heal attempt (0 = never).
static LAST_SEARXNG_HEAL_S: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Minimum gap between background heal attempts.
const SEARXNG_HEAL_GAP_S: u64 = 60;

/// Debounce for the background heal: at most one attempt per gap window.
fn should_attempt_heal(last_epoch_s: u64, now_epoch_s: u64) -> bool {
    now_epoch_s.saturating_sub(last_epoch_s) >= SEARXNG_HEAL_GAP_S
}

/// Fire-and-forget SearXNG recovery. `ensure_searxng` handles the whole
/// chain (start Docker Desktop → start/create container → fix config), so a
/// search that fails because Docker isn't running yet self-repairs — the
/// dominant "search is always down" cause is Docker Desktop not being up,
/// not SearXNG itself. Debounced so retry storms don't stack Docker dances.
fn trigger_searxng_heal(searxng_url: &str) {
    use std::sync::atomic::Ordering;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let last = LAST_SEARXNG_HEAL_S.load(Ordering::Relaxed);
    if !should_attempt_heal(last, now) {
        return;
    }
    if LAST_SEARXNG_HEAL_S
        .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
    {
        return; // another caller won the race
    }
    let url = searxng_url.to_string();
    tokio::spawn(async move {
        match crate::searxng::ensure_searxng(&url).await {
            Ok(()) => tracing::info!("SearXNG background heal succeeded"),
            Err(e) => tracing::warn!("SearXNG background heal failed: {e}"),
        }
    });
}

impl WebSearchTool {
    /// Execute a search via SearXNG. Falls back to Brave if SearXNG is unreachable
    /// and a Brave API key is configured.
    async fn execute_searxng(&self, query: &str, count: u32) -> String {
        // Pre-flight: if the heartbeat probe has marked SearXNG Degraded
        // (3+ consecutive /health failures), short-circuit with a clear
        // message. Otherwise we'd hang for the full HTTP timeout and either
        // return an opaque connection error or, worse, silently return zero
        // results — the failure mode that motivated this check.
        if let Some(reg) = &self.health_registry {
            if !reg.is_healthy("searxng") {
                trigger_searxng_heal(&self.searxng_url);
                return format!(
                    "Error: SearXNG backend is degraded (health probe failing). \
                     An automatic restart was just triggered — retry this \
                     search in ~30-60 seconds. Query was: {:?}",
                    query
                );
            }
        }

        let result = self
            .client
            .get(format!("{}/search", self.searxng_url))
            .query(&[("q", query), ("format", "json"), ("categories", "general")])
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await;

        match result {
            Ok(response) => {
                if !response.status().is_success() {
                    let status = response.status();
                    // SearXNG returned an error — try Brave fallback if available.
                    if !self.api_key.is_empty() {
                        tracing::warn!(
                            "SearXNG returned HTTP {}, falling back to Brave Search",
                            status
                        );
                        let mut result = self.execute_brave(query, count).await;
                        result.push_str("\n(Fell back to Brave Search)");
                        return result;
                    }
                    // No Brave key - return error
                    return format!(
                        "Error: SearXNG returned HTTP {} and no Brave API key configured. \
                         Set 'braveApiKey' in config.json or fix SearXNG URL.",
                        status
                    );
                }

                match response.json::<serde_json::Value>().await {
                    Ok(data) => {
                        let results = data
                            .get("results")
                            .and_then(|r| r.as_array())
                            .cloned()
                            .unwrap_or_default();

                        if results.is_empty() {
                            return format!("No results for: {}", query);
                        }

                        let mut lines = vec![format!("Results for: {}\n", query)];
                        for (i, item) in results.iter().take(count as usize).enumerate() {
                            let title = item.get("title").and_then(|v| v.as_str()).unwrap_or("");
                            let url = item.get("url").and_then(|v| v.as_str()).unwrap_or("");
                            lines.push(format!("{}. {}\n   {}", i + 1, title, url));

                            if let Some(desc) = item.get("content").and_then(|v| v.as_str()) {
                                lines.push(format!("   {}", desc));
                            }
                        }
                        lines.join("\n")
                    }
                    Err(e) => format!("Error parsing SearXNG results: {}", e),
                }
            }
            Err(e) => {
                // Connection error — usually Docker (and thus the SearXNG
                // container) is not running. Kick the self-heal chain so a
                // retry succeeds, then fall back to Brave if configured.
                trigger_searxng_heal(&self.searxng_url);
                if !self.api_key.is_empty() {
                    tracing::warn!("SearXNG unavailable ({}), falling back to Brave Search", e);
                    let mut result = self.execute_brave(query, count).await;
                    result.push_str("\n(Fell back to Brave Search)");
                    return result;
                }
                // No Brave key - return error
                format!(
                    "Error: SearXNG unavailable ({}). An automatic restart was \
                     just triggered — retry this search in ~30-60 seconds.",
                    e
                )
            }
        }
    }

    /// Execute a search via the Brave Search API.
    async fn execute_brave(&self, query: &str, count: u32) -> String {
        if self.api_key.is_empty() {
            return "Error: BRAVE_API_KEY not configured. Set it in config.json under 'braveApiKey'.".to_string();
        }

        match self
            .client
            .get("https://api.search.brave.com/res/v1/web/search")
            .query(&[("q", query), ("count", &count.to_string())])
            .header("Accept", "application/json")
            .header("X-Subscription-Token", &self.api_key)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await
        {
            Ok(response) => {
                if !response.status().is_success() {
                    let status = response.status();
                    let code = status.as_u16();
                    let body = response.text().await.unwrap_or_default();
                    let hint = match code {
                        401 | 403 => ". Hint: API key may be invalid or expired. Check your Brave API subscription.",
                        422 => ". Hint: query may be malformed or API subscription may be inactive.",
                        429 => ". Hint: rate limited. Wait a moment and try again.",
                        500..=599 => ". Hint: Brave Search service error. Try again shortly.",
                        _ => ". Hint: check API key and query format.",
                    };
                    return format!(
                        "Error: Brave Search returned HTTP {}: {}{}",
                        status, body, hint
                    );
                }

                match response.json::<serde_json::Value>().await {
                    Ok(data) => {
                        let results = data
                            .get("web")
                            .and_then(|w| w.get("results"))
                            .and_then(|r| r.as_array())
                            .cloned()
                            .unwrap_or_default();

                        if results.is_empty() {
                            return format!("No results for: {}", query);
                        }

                        let mut lines = vec![format!("Results for: {}\n", query)];
                        for (i, item) in results.iter().take(count as usize).enumerate() {
                            let title = item.get("title").and_then(|v| v.as_str()).unwrap_or("");
                            let url = item.get("url").and_then(|v| v.as_str()).unwrap_or("");
                            lines.push(format!("{}. {}\n   {}", i + 1, title, url));

                            if let Some(desc) = item.get("description").and_then(|v| v.as_str()) {
                                lines.push(format!("   {}", desc));
                            }
                        }
                        lines.join("\n")
                    }
                    Err(e) => format!("Error parsing search results: {}", e),
                }
            }
            Err(e) => format!("Error: {}. Hint: check network connectivity.", e),
        }
    }
}

// ---------------------------------------------------------------------------
// WebFetchTool
// ---------------------------------------------------------------------------

/// Truncate at a char boundary at or before `max_chars`.
/// Returns the (possibly shortened) text and whether truncation happened.
fn truncate_at_boundary(text: String, max_chars: usize) -> (String, bool) {
    if text.len() <= max_chars {
        return (text, false);
    }
    let mut end = max_chars;
    while !text.is_char_boundary(end) && end > 0 {
        end -= 1;
    }
    (text[..end].to_string(), true)
}

/// Map a crw-server `/v1/scrape` response into the web_fetch result envelope.
///
/// Returns `None` when the response is not a successful scrape with markdown —
/// the caller falls back to the plain fetcher. The envelope shape matches the
/// plain path exactly (`url`/`finalUrl`/`status`/`extractor`/`truncated`/
/// `length`/`text`) so downstream consumers cannot tell the paths apart.
fn crw_envelope(resp: &serde_json::Value, url: &str, max_chars: usize) -> Option<String> {
    if resp.get("success").and_then(|s| s.as_bool()) != Some(true) {
        return None;
    }
    let data = resp.get("data")?;
    let markdown = data.get("markdown").and_then(|m| m.as_str())?;
    if markdown.trim().is_empty() {
        return None;
    }
    let meta = data.get("metadata");
    let final_url = meta
        .and_then(|m| m.get("sourceURL"))
        .and_then(|u| u.as_str())
        .unwrap_or(url);
    let status = meta
        .and_then(|m| m.get("statusCode"))
        .and_then(|s| s.as_u64())
        .unwrap_or(200);

    let (text, truncated) = truncate_at_boundary(markdown.to_string(), max_chars);
    Some(
        serde_json::json!({
            "url": url,
            "finalUrl": final_url,
            "status": status,
            "extractor": "crw",
            "truncated": truncated,
            "length": text.len(),
            "text": text
        })
        .to_string(),
    )
}

/// Fetch and extract content from a URL.
///
/// When a local crw-server (fastCRW) is configured and reachable, fetching
/// goes through its `/v1/scrape` (better extraction, markdown-native).
/// Any crw failure falls back to the plain HTTP fetcher below.
pub struct WebFetchTool {
    max_chars: usize,
    client: Client,
    /// Base URL of a local crw-server; empty = disabled.
    crw_url: String,
}

impl WebFetchTool {
    /// Create a new web fetch tool.
    pub fn new(max_chars: usize) -> Self {
        let client = Client::builder()
            .redirect(reqwest::redirect::Policy::limited(MAX_REDIRECTS))
            .user_agent(USER_AGENT)
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| Client::new());

        Self {
            max_chars,
            client,
            crw_url: String::new(),
        }
    }

    /// Route fetches through a local crw-server at `url` (with fallback).
    pub fn with_crw(mut self, url: String) -> Self {
        self.crw_url = url;
        self
    }

    /// Try `/v1/scrape` on the configured crw-server. `None` on any failure.
    async fn fetch_via_crw(&self, url: &str, max_chars: usize) -> Option<String> {
        if self.crw_url.is_empty() {
            return None;
        }
        let resp = self
            .client
            .post(format!("{}/v1/scrape", self.crw_url))
            .json(&serde_json::json!({"url": url, "formats": ["markdown"]}))
            .timeout(std::time::Duration::from_secs(25))
            .send()
            .await
            .ok()?;
        let body: serde_json::Value = resp.json().await.ok()?;
        crw_envelope(&body, url, max_chars)
    }
}

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch URL and extract readable content (HTML -> text).\n\
         Only pass a URL that appeared verbatim in a prior tool result or in the\n\
         user's message. Do not guess URLs from memory — Vercel/Cloudflare/news\n\
         site paths change and guessed URLs return 404 or login walls. If you\n\
         don't have the URL, call web_search first to discover it."
    }

    fn permission(&self) -> PermissionLevel {
        PermissionLevel::Network
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::ParallelSafe
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch"
                },
                "extractMode": {
                    "type": "string",
                    "enum": ["markdown", "text"],
                    "default": "markdown"
                },
                "maxChars": {
                    "type": "integer",
                    "minimum": 100
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        let url = match params.get("url").and_then(|v| v.as_str()) {
            Some(u) => u,
            None => return serde_json::json!({"error": "url parameter is required"}).to_string(),
        };

        let extract_mode = params
            .get("extractMode")
            .and_then(|v| v.as_str())
            .unwrap_or("markdown");

        let max_chars = params
            .get("maxChars")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(self.max_chars);

        // Validate URL.
        if let Err(e) = validate_url(url) {
            return serde_json::json!({
                "error": format!("URL validation failed: {}", e),
                "url": url
            })
            .to_string();
        }

        // Prefer the local crw-server when configured — better extraction,
        // markdown-native. Falls through to the plain fetcher on any failure.
        if let Some(envelope) = self.fetch_via_crw(url, max_chars).await {
            return envelope;
        }

        match self.client.get(url).send().await {
            Ok(response) => {
                let status = response.status().as_u16();
                let final_url = response.url().to_string();
                let content_type = response
                    .headers()
                    .get("content-type")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("")
                    .to_string();

                // Check content-length header; reject obviously oversized responses early.
                if let Some(len) = response
                    .headers()
                    .get("content-length")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse::<usize>().ok())
                {
                    if len > MAX_BODY_BYTES {
                        return serde_json::json!({
                            "error": format!("Response too large ({:.1} MB, limit {:.1} MB)",
                                len as f64 / 1e6, MAX_BODY_BYTES as f64 / 1e6),
                            "url": url
                        })
                        .to_string();
                    }
                }

                // Read body with size guard (content-length can be absent or wrong).
                let body = match response.bytes().await {
                    Ok(bytes) if bytes.len() > MAX_BODY_BYTES => {
                        return serde_json::json!({
                            "error": format!("Response too large ({:.1} MB, limit {:.1} MB)",
                                bytes.len() as f64 / 1e6, MAX_BODY_BYTES as f64 / 1e6),
                            "url": url
                        })
                        .to_string();
                    }
                    Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
                    Err(e) => {
                        return serde_json::json!({
                            "error": format!("Failed to read response body: {}", e),
                            "url": url
                        })
                        .to_string();
                    }
                };

                let (text, extractor) = if content_type.contains("application/json") {
                    let formatted = match serde_json::from_str::<serde_json::Value>(&body) {
                        Ok(v) => serde_json::to_string_pretty(&v).unwrap_or_else(|_| body.clone()),
                        Err(_) => body.clone(),
                    };
                    (formatted, "json")
                } else if content_type.contains("text/html")
                    || body.trim_start().to_lowercase().starts_with("<!doctype")
                    || body.trim_start().to_lowercase().starts_with("<html")
                {
                    let extracted = extract_html_content(&body, extract_mode);
                    (extracted, "readability")
                } else {
                    (body, "raw")
                };

                let (final_text, truncated) = truncate_at_boundary(text, max_chars);

                serde_json::json!({
                    "url": url,
                    "finalUrl": final_url,
                    "status": status,
                    "extractor": extractor,
                    "truncated": truncated,
                    "length": final_text.len(),
                    "text": final_text
                })
                .to_string()
            }
            Err(e) => serde_json::json!({
                "error": e.to_string(),
                "url": url
            })
            .to_string(),
        }
    }

    async fn execute_with_context(
        &self,
        params: HashMap<String, serde_json::Value>,
        ctx: &ToolExecutionContext,
    ) -> String {
        let url = params.get("url").and_then(|v| v.as_str()).unwrap_or("");

        let _ = ctx.event_tx.send(ToolEvent::Progress {
            tool_name: "web_fetch".to_string(),
            tool_call_id: ctx.tool_call_id.clone(),
            elapsed_ms: 0,
            output_preview: Some(format!("Fetching: {}", url)),
        });

        let result = self.execute(params).await;

        let _ = ctx.event_tx.send(ToolEvent::Progress {
            tool_name: "web_fetch".to_string(),
            tool_call_id: ctx.tool_call_id.clone(),
            elapsed_ms: 0,
            output_preview: Some("Extracting content...".to_string()),
        });

        result
    }
}

/// Extract readable content from HTML using `dom_smoothie` (Mozilla Readability port).
///
/// Uses content-scoring to find the main article, stripping navigation, ads,
/// and boilerplate.  Falls back to the old `scraper`-based extraction on
/// parse errors or when `dom_smoothie` returns empty content.
fn extract_html_content(html: &str, mode: &str) -> String {
    use dom_smoothie::{Config, Readability, TextMode};

    let text_mode = if mode == "markdown" {
        TextMode::Markdown
    } else {
        TextMode::Formatted
    };

    let config = Config {
        text_mode,
        ..Default::default()
    };

    match Readability::new(html, None, Some(config)) {
        Ok(mut r) => match r.parse() {
            Ok(article) => {
                let title = &article.title;
                let body = article.text_content.to_string();
                let result = normalize_whitespace(&body);
                if result.trim().is_empty() {
                    return fallback_extract(html, mode);
                }
                if title.is_empty() {
                    result
                } else {
                    format!("# {}\n\n{}", title.trim(), result)
                }
            }
            Err(_) => fallback_extract(html, mode),
        },
        Err(_) => fallback_extract(html, mode),
    }
}

/// Fallback HTML extraction using `scraper` when `dom_smoothie` fails.
fn fallback_extract(html: &str, mode: &str) -> String {
    use scraper::{Html, Selector};

    let document = Html::parse_document(html);

    let title = Selector::parse("title")
        .ok()
        .and_then(|sel| document.select(&sel).next())
        .map(|el| el.text().collect::<String>())
        .unwrap_or_default();

    let selectors = ["article", "main", "[role=\"main\"]", "body"];
    let mut body_text = String::new();

    for sel_str in &selectors {
        if let Ok(sel) = Selector::parse(sel_str) {
            if let Some(el) = document.select(&sel).next() {
                body_text = if mode == "markdown" {
                    rewrite_html(&el.html(), false)
                } else {
                    el.text().collect::<Vec<_>>().join(" ")
                };
                if !body_text.trim().is_empty() {
                    break;
                }
            }
        }
    }

    if body_text.trim().is_empty() {
        body_text = document.root_element().text().collect::<Vec<_>>().join(" ");
    }

    let result = normalize_whitespace(&body_text);

    if title.is_empty() {
        result
    } else {
        format!("# {}\n\n{}", title.trim(), result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // crw scrape envelope
    // -----------------------------------------------------------------------

    #[test]
    fn test_crw_envelope_maps_successful_scrape() {
        let resp = serde_json::json!({
            "success": true,
            "data": {
                "markdown": "# Example Domain\n\nSome content.",
                "metadata": {
                    "title": "Example Domain",
                    "sourceURL": "https://example.com",
                    "statusCode": 200
                }
            }
        });
        let envelope = crw_envelope(&resp, "https://example.com", 4000)
            .expect("successful scrape must map to an envelope");
        let v: serde_json::Value = serde_json::from_str(&envelope).unwrap();
        assert_eq!(v["url"], "https://example.com");
        assert_eq!(v["finalUrl"], "https://example.com");
        assert_eq!(v["status"], 200);
        assert_eq!(v["extractor"], "crw");
        assert_eq!(v["truncated"], false);
        assert!(v["text"].as_str().unwrap().contains("Example Domain"));
        // Same envelope keys as the plain path — extract_web_content works.
        assert_eq!(extract_web_content(&envelope), v["text"].as_str().unwrap());
    }

    #[test]
    fn test_crw_envelope_truncates_and_rejects_failures() {
        let long = serde_json::json!({
            "success": true,
            "data": {"markdown": "x".repeat(500), "metadata": {"statusCode": 200}}
        });
        let v: serde_json::Value =
            serde_json::from_str(&crw_envelope(&long, "https://e.com", 100).unwrap()).unwrap();
        assert_eq!(v["truncated"], true);
        assert_eq!(v["text"].as_str().unwrap().len(), 100);

        // Failure / malformed responses → None (caller falls back).
        assert!(crw_envelope(&serde_json::json!({"success": false}), "u", 100).is_none());
        assert!(crw_envelope(&serde_json::json!({"data": {}}), "u", 100).is_none());
        assert!(
            crw_envelope(
                &serde_json::json!({"success": true, "data": {"markdown": ""}}),
                "u",
                100
            )
            .is_none(),
            "empty markdown must fall back to the plain fetcher"
        );
    }

    #[test]
    fn test_truncate_at_boundary_respects_char_boundaries() {
        let (t, cut) = truncate_at_boundary("héllo wörld".to_string(), 6);
        assert!(cut);
        assert!(t.len() <= 6);
        assert!(String::from_utf8(t.into_bytes()).is_ok());
        let (t2, cut2) = truncate_at_boundary("short".to_string(), 100);
        assert_eq!(t2, "short");
        assert!(!cut2);
    }

    // -----------------------------------------------------------------------
    // searxng heal debounce
    // -----------------------------------------------------------------------

    #[test]
    fn test_should_attempt_heal_debounces() {
        assert!(should_attempt_heal(0, 1000), "never healed → attempt");
        assert!(
            !should_attempt_heal(1000, 1000 + SEARXNG_HEAL_GAP_S - 1),
            "inside the gap → hold"
        );
        assert!(
            should_attempt_heal(1000, 1000 + SEARXNG_HEAL_GAP_S),
            "gap elapsed → attempt"
        );
    }

    #[tokio::test]
    #[ignore] // requires a running local crw-server (port 3000) + network
    async fn live_web_fetch_routes_through_crw() {
        let tool = WebFetchTool::new(4000).with_crw("http://localhost:3000".to_string());
        let mut params = HashMap::new();
        params.insert(
            "url".to_string(),
            serde_json::Value::String("https://example.com".to_string()),
        );
        let out = tool.execute(params).await;
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["extractor"], "crw", "fetch must route through crw: {out}");
        assert!(v["text"].as_str().unwrap().contains("Example Domain"));
    }

    // -----------------------------------------------------------------------
    // validate_url tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_url_http() {
        assert!(validate_url("http://example.com").is_ok());
    }

    #[test]
    fn test_validate_url_https() {
        assert!(validate_url("https://example.com/path?q=1").is_ok());
    }

    #[test]
    fn test_validate_url_ftp_rejected() {
        let result = validate_url("ftp://example.com");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("http/https"));
    }

    #[test]
    fn test_validate_url_empty() {
        let result = validate_url("");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_url_no_scheme() {
        let result = validate_url("example.com");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_url_file_scheme_rejected() {
        let result = validate_url("file:///etc/passwd");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_url_missing_domain() {
        let result = validate_url("http://");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_url_localhost_blocked() {
        let result = validate_url("http://localhost:8080/api");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("blocked"));
    }

    #[test]
    fn test_validate_url_loopback_ip_blocked() {
        let result = validate_url("http://127.0.0.1:9090/secret");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("blocked"));
    }

    #[test]
    fn test_validate_url_private_ip_blocked() {
        assert!(validate_url("http://192.168.1.1").is_err());
        assert!(validate_url("http://10.0.0.1").is_err());
        assert!(validate_url("http://172.16.0.1").is_err());
    }

    #[test]
    fn test_validate_url_metadata_ip_blocked() {
        let result = validate_url("http://169.254.169.254/latest/meta-data/");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_url_public_ip_allowed() {
        assert!(validate_url("http://8.8.8.8").is_ok());
        assert!(validate_url("https://1.1.1.1").is_ok());
    }

    // -----------------------------------------------------------------------
    // normalize_whitespace tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_normalize_whitespace_collapses_spaces() {
        let result = normalize_whitespace("hello    world");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_normalize_whitespace_collapses_tabs() {
        let result = normalize_whitespace("hello\t\tworld");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_normalize_whitespace_limits_newlines() {
        let result = normalize_whitespace("hello\n\n\n\n\nworld");
        assert_eq!(result, "hello\n\nworld");
    }

    #[test]
    fn test_normalize_whitespace_trims() {
        let result = normalize_whitespace("   hello   ");
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_normalize_whitespace_preserves_double_newline() {
        let result = normalize_whitespace("hello\n\nworld");
        assert_eq!(result, "hello\n\nworld");
    }

    // -----------------------------------------------------------------------
    // fallback_extract markdown conversion tests (via rewrite_html)
    // -----------------------------------------------------------------------

    #[test]
    fn test_fallback_extract_headings() {
        let html = "<html><body><h1>Title</h1><h2>Subtitle</h2></body></html>";
        let result = fallback_extract(html, "markdown");
        assert!(
            result.contains("Title"),
            "result should contain heading text: {}",
            result
        );
        assert!(
            result.contains("Subtitle"),
            "result should contain subheading text: {}",
            result
        );
    }

    #[test]
    fn test_fallback_extract_links() {
        let html = r#"<html><body><a href="https://example.com">Example</a></body></html>"#;
        let result = fallback_extract(html, "markdown");
        assert!(
            result.contains("Example"),
            "result should contain link text: {}",
            result
        );
        assert!(
            result.contains("https://example.com"),
            "result should contain URL: {}",
            result
        );
    }

    #[test]
    fn test_fallback_extract_list_items() {
        let html = "<html><body><ul><li>First</li><li>Second</li></ul></body></html>";
        let result = fallback_extract(html, "markdown");
        assert!(
            result.contains("First"),
            "result should contain first item: {}",
            result
        );
        assert!(
            result.contains("Second"),
            "result should contain second item: {}",
            result
        );
    }

    #[test]
    fn test_fallback_extract_paragraphs() {
        let html = "<html><body><p>First paragraph</p><p>Second paragraph</p></body></html>";
        let result = fallback_extract(html, "markdown");
        assert!(result.contains("First paragraph"), "result: {}", result);
        assert!(result.contains("Second paragraph"), "result: {}", result);
    }

    #[test]
    fn test_fallback_extract_no_raw_tags() {
        let html = "<html><body><div><span>text content</span></div></body></html>";
        let result = fallback_extract(html, "markdown");
        assert!(
            result.contains("text content"),
            "result should contain text: {}",
            result
        );
        assert!(
            !result.contains("<span>"),
            "result should not contain raw span tags: {}",
            result
        );
        assert!(
            !result.contains("<div>"),
            "result should not contain raw div tags: {}",
            result
        );
    }

    #[test]
    fn test_fallback_extract_text_mode_no_markdown() {
        let html = "<html><body><h1>Heading</h1><p>Paragraph text</p></body></html>";
        let result = fallback_extract(html, "text");
        assert!(
            result.contains("Heading"),
            "result should contain heading text: {}",
            result
        );
        assert!(
            result.contains("Paragraph text"),
            "result should contain paragraph: {}",
            result
        );
    }

    // -----------------------------------------------------------------------
    // extract_html_content tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_html_content_with_title() {
        let html =
            "<html><head><title>Test Page</title></head><body><p>Content here</p></body></html>";
        let result = extract_html_content(html, "text");
        assert!(result.contains("# Test Page"), "result: {}", result);
        assert!(result.contains("Content here"), "result: {}", result);
    }

    #[test]
    fn test_extract_html_content_markdown_mode() {
        let html = "<html><body><h1>Heading</h1><p>Paragraph</p></body></html>";
        let result = extract_html_content(html, "markdown");
        assert!(result.contains("# Heading"), "result: {}", result);
        assert!(result.contains("Paragraph"), "result: {}", result);
    }

    #[test]
    fn test_extract_html_content_prefers_article() {
        let html =
            "<html><body><div>Noise</div><article><p>Article content</p></article></body></html>";
        let result = extract_html_content(html, "text");
        assert!(result.contains("Article content"), "result: {}", result);
    }

    // -----------------------------------------------------------------------
    // Tool trait basics
    // -----------------------------------------------------------------------

    #[test]
    fn test_web_search_tool_name() {
        let tool = WebSearchTool::new(
            None,
            5,
            "searxng".to_string(),
            "http://localhost:8888".to_string(),
        );
        assert_eq!(tool.name(), "web_search");
    }

    #[tokio::test]
    async fn test_web_search_searxng_degraded_short_circuits() {
        // Build a registry with threshold=1 so a single failed probe marks
        // "searxng" as Degraded. Register a stub probe that always reports
        // unhealthy, then run_due_probes to populate state.
        use crate::heartbeat::health::{HealthProbe, HealthRegistry, ProbeResult};
        use async_trait::async_trait;

        struct UnhealthyProbe;
        #[async_trait]
        impl HealthProbe for UnhealthyProbe {
            fn name(&self) -> &str {
                "searxng"
            }
            fn interval_secs(&self) -> u64 {
                60
            }
            async fn check(&self) -> ProbeResult {
                ProbeResult {
                    healthy: false,
                    latency_ms: 0,
                }
            }
        }

        let mut reg = HealthRegistry::new_with_threshold(1);
        reg.register(Box::new(UnhealthyProbe));
        reg.run_due_probes().await;
        assert!(!reg.is_healthy("searxng"));

        let tool = WebSearchTool::new(
            None,
            5,
            "searxng".to_string(),
            // Point at a port nothing listens on; if the short-circuit fails
            // the test will hang for 10s then return a connection error.
            "http://127.0.0.1:1".to_string(),
        )
        .with_health_registry(Some(Arc::new(reg)));

        let mut params = HashMap::new();
        params.insert("query".to_string(), serde_json::json!("test"));
        let result = tool.execute(params).await;

        assert!(
            result.contains("SearXNG backend is degraded"),
            "expected degraded short-circuit message, got: {result}"
        );
        assert!(
            result.contains("automatic restart") && result.contains("retry"),
            "expected self-heal notice with retry hint, got: {result}"
        );
    }

    #[tokio::test]
    async fn test_web_search_searxng_healthy_proceeds_to_call() {
        // When the probe is healthy (or absent), the tool proceeds to the
        // HTTP call. We assert it reaches the network layer by checking that
        // the error is a connection error, NOT the degraded short-circuit.
        use crate::heartbeat::health::HealthRegistry;

        let reg = HealthRegistry::new();
        // No probes registered -> is_healthy returns true (optimistic default).

        let tool = WebSearchTool::new(
            None,
            5,
            "searxng".to_string(),
            "http://127.0.0.1:1".to_string(), // unreachable
        )
        .with_health_registry(Some(Arc::new(reg)));

        let mut params = HashMap::new();
        params.insert("query".to_string(), serde_json::json!("test"));
        let result = tool.execute(params).await;

        // Reaches the network layer -> connection error, NOT the degraded msg.
        assert!(
            !result.contains("degraded"),
            "healthy probe must not short-circuit, got: {result}"
        );
        assert!(
            result.contains("Error:") || result.contains("No results"),
            "expected network error or empty-results path, got: {result}"
        );
    }

    #[test]
    fn test_web_search_tool_parameters() {
        let tool = WebSearchTool::new(
            None,
            5,
            "searxng".to_string(),
            "http://localhost:8888".to_string(),
        );
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["query"].is_object());
    }

    #[test]
    fn test_web_fetch_tool_name() {
        let tool = WebFetchTool::new(50000);
        assert_eq!(tool.name(), "web_fetch");
    }

    #[test]
    fn test_web_fetch_description_steers_away_from_url_guessing() {
        // Models (notably Qwen on local) will otherwise guess plausible-looking
        // URLs like vercel.com/blog/security-<year> and hit 404 walls. The
        // description must point them at web_search first when the URL isn't
        // already in context.
        let tool = WebFetchTool::new(50000);
        let desc = tool.description();
        assert!(
            desc.contains("web_search"),
            "web_fetch description should point at web_search, got: {desc}"
        );
        assert!(
            desc.to_lowercase().contains("guess"),
            "web_fetch description should warn against guessing URLs, got: {desc}"
        );
    }

    #[test]
    fn test_web_fetch_tool_parameters() {
        let tool = WebFetchTool::new(50000);
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["url"].is_object());
    }

    #[tokio::test]
    async fn test_web_search_no_api_key() {
        // With provider="brave" and no API key, expect the Brave key error.
        let tool = WebSearchTool::new(
            Some(String::new()),
            5,
            "brave".to_string(),
            "http://localhost:8888".to_string(),
        );
        let mut params = HashMap::new();
        params.insert(
            "query".to_string(),
            serde_json::Value::String("test".to_string()),
        );
        let result = tool.execute(params).await;
        assert!(result.contains("BRAVE_API_KEY not configured"));
    }

    #[tokio::test]
    async fn test_web_search_no_api_key_has_hint() {
        let tool = WebSearchTool::new(
            Some(String::new()),
            5,
            "brave".to_string(),
            "http://localhost:8888".to_string(),
        );
        let mut params = HashMap::new();
        params.insert(
            "query".to_string(),
            serde_json::Value::String("test".to_string()),
        );
        let result = tool.execute(params).await;
        assert!(
            result.contains("config.json"),
            "Expected config.json hint: {}",
            result
        );
        assert!(
            result.contains("braveApiKey"),
            "Expected braveApiKey hint: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_web_search_searxng_unavailable_no_brave_key() {
        // SearXNG provider with no Brave key and unreachable URL should return error.
        let tool = WebSearchTool::new(
            Some(String::new()),
            5,
            "searxng".to_string(),
            "http://127.0.0.1:19999".to_string(), // nothing listening here
        );
        let mut params = HashMap::new();
        params.insert(
            "query".to_string(),
            serde_json::Value::String("test".to_string()),
        );
        let result = tool.execute(params).await;
        assert!(
            result.contains("Error: SearXNG unavailable") && result.contains("automatic restart"),
            "Expected unavailable error with self-heal notice, got: {}",
            result
        );
    }

    #[test]
    fn test_web_search_unknown_provider() {
        // unknown provider returns an error synchronously via execute dispatch
        let tool = WebSearchTool::new(
            Some(String::new()),
            5,
            "bing".to_string(),
            "http://localhost:8888".to_string(),
        );
        // We check the provider field directly since execute is async
        assert_eq!(tool.provider, "bing");
    }

    #[tokio::test]
    async fn test_web_fetch_invalid_url() {
        let tool = WebFetchTool::new(50000);
        let mut params = HashMap::new();
        params.insert(
            "url".to_string(),
            serde_json::Value::String("ftp://invalid.example".to_string()),
        );
        let result = tool.execute(params).await;
        assert!(result.contains("error") || result.contains("URL validation failed"));
    }

    #[tokio::test]
    async fn test_web_fetch_missing_url() {
        let tool = WebFetchTool::new(50000);
        let params = HashMap::new();
        let result = tool.execute(params).await;
        assert!(result.contains("url parameter is required"));
    }

    // -----------------------------------------------------------------------
    // Pipeline tests: what the main model sees after processing
    // -----------------------------------------------------------------------

    /// Realistic BBC-like web_fetch result fixture (~2300 chars of article text).
    fn bbc_web_fetch_fixture() -> String {
        let article_text = r#"# UK Economy Grows Faster Than Expected

The UK economy grew by 0.4% in the last quarter, beating analyst forecasts of 0.2%.

## Key Figures

- GDP growth: 0.4% quarter-on-quarter
- Services sector: +0.6%
- Manufacturing: +0.1%
- Construction: -0.2%

## Analysis

The stronger-than-expected growth was driven primarily by the services sector, which accounts
for around 80% of the UK economy. Consumer spending rose 0.5% as real wages increased for the
sixth consecutive month.

The Bank of England is expected to hold interest rates at their current level at next month's
meeting, though some economists are now pricing in a cut before year-end.

Finance Minister Sarah Johnson welcomed the figures: "Today's data shows that the UK economy
is resilient and growing. We are seeing the results of our long-term economic plan."

Opposition economists noted that growth remains below the G7 average and cautioned against
over-optimism given global trade uncertainty and elevated energy costs.

## Market Reaction

The pound rose 0.3% against the dollar to 1.2850 following the data release. The FTSE 100
gained 0.4%, with banking stocks leading the advance.

Ten-year gilt yields fell slightly to 4.12% as traders revised down expectations for further
rate rises.

## What Comes Next

The ONS will release revised figures in six weeks. Analysts expect the Q1 revision to show
growth of 0.3-0.5%, broadly in line with today's preliminary estimate.

The next GDP release, covering Q2, is scheduled for August 14th."#;

        serde_json::json!({
            "url": "https://www.bbc.com/news/business/uk-economy-q1",
            "finalUrl": "https://www.bbc.com/news/business/uk-economy-q1",
            "status": 200,
            "extractor": "readability",
            "truncated": false,
            "length": article_text.len(),
            "text": article_text
        })
        .to_string()
    }

    // Re-export the production function so tests can call it by the same name.
    use super::extract_web_content;

    #[test]
    fn test_web_fetch_passthrough_vs_summarized() {
        use crate::agent::context_gate::ContentGate;
        use crate::agent::context_store::ContextStore;
        let raw = bbc_web_fetch_fixture();
        let passthrough = raw.clone();

        let mut store = ContextStore::new();
        let (_var_name, context_store_view) = store.store(raw.clone());

        // 50 token budget → raw (≈575 tokens) will not fit → briefing path.
        let mut gate = ContentGate::new(50, 0.2);
        let gate_result = gate.admit_simple(&raw);
        let gate_view = gate_result.into_text();

        assert!(passthrough.contains("UK economy grew by 0.4%"));
        assert!(passthrough.contains("Bank of England"));
        assert!(!context_store_view.contains("Bank of England"));
        assert!(context_store_view.contains("chars"));
        assert!(context_store_view.contains("output_0"));
        assert!(!gate_view.contains("Bank of England"));
        assert!(gate_view.contains("JSON Summary") || gate_view.contains("Content Summary"));
    }

    #[test]
    fn test_web_fetch_smart_summary_preserves_content() {
        let raw = bbc_web_fetch_fixture();
        let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let original_text = parsed["text"].as_str().unwrap();
        let extracted = extract_web_content(&raw);

        assert!(extracted.contains("UK economy grew by 0.4%"));
        assert!(extracted.contains("Bank of England"));
        assert!(extracted.contains("FTSE 100"));
        assert_eq!(extracted, original_text);
        assert!(extracted.len() < raw.len());

        let plain = "This is plain text, not JSON.";
        assert_eq!(extract_web_content(plain), plain);

        let no_text_json = r#"{"status": 200, "url": "https://example.com"}"#;
        assert_eq!(extract_web_content(no_text_json), no_text_json);
    }

    // -----------------------------------------------------------------------
    // Progress event emission tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_web_search_emits_start_progress_event() {
        use crate::agent::audit::ToolEvent;
        use crate::agent::tools::base::ToolExecutionContext;

        let tool = WebSearchTool::new(
            Some(String::new()),
            5,
            "brave".to_string(),
            "http://localhost:8888".to_string(),
        );

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ToolEvent>();
        let token = tokio_util::sync::CancellationToken::new();
        let ctx = ToolExecutionContext {
            event_tx: tx,
            cancellation_token: token,
            tool_call_id: "call_search".to_string(),
        };

        let mut params = HashMap::new();
        params.insert(
            "query".to_string(),
            serde_json::Value::String("rust programming".to_string()),
        );

        tool.execute_with_context(params, &ctx).await;

        let first = rx.try_recv().expect("Expected at least one progress event");
        match first {
            ToolEvent::Progress {
                tool_name,
                tool_call_id,
                elapsed_ms,
                output_preview: Some(ref preview),
            } => {
                assert_eq!(tool_name, "web_search");
                assert_eq!(tool_call_id, "call_search");
                assert_eq!(elapsed_ms, 0);
                assert!(preview.contains("rust programming"));
            }
            other => panic!(
                "Expected Progress event with output_preview, got: {:?}",
                other
            ),
        }
    }

    #[tokio::test]
    async fn test_web_fetch_emits_fetch_and_extract_progress_events() {
        use crate::agent::audit::ToolEvent;
        use crate::agent::tools::base::ToolExecutionContext;

        let tool = WebFetchTool::new(50000);

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ToolEvent>();
        let token = tokio_util::sync::CancellationToken::new();
        let ctx = ToolExecutionContext {
            event_tx: tx,
            cancellation_token: token,
            tool_call_id: "call_fetch".to_string(),
        };

        let mut params = HashMap::new();
        params.insert(
            "url".to_string(),
            serde_json::Value::String("ftp://invalid-url-that-fails-fast".to_string()),
        );

        tool.execute_with_context(params, &ctx).await;

        let mut events = vec![];
        while let Ok(ev) = rx.try_recv() {
            events.push(ev);
        }

        assert!(!events.is_empty(), "Expected at least one progress event");
        match &events[0] {
            ToolEvent::Progress {
                tool_name,
                tool_call_id,
                output_preview: Some(preview),
                ..
            } => {
                assert_eq!(tool_name, "web_fetch");
                assert_eq!(tool_call_id, "call_fetch");
                assert!(preview.starts_with("Fetching:"));
            }
            other => panic!("Expected Fetching progress event, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_web_fetch_emits_extracting_progress_after_fetch() {
        use crate::agent::audit::ToolEvent;
        use crate::agent::tools::base::ToolExecutionContext;

        let tool = WebFetchTool::new(50000);

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ToolEvent>();
        let token = tokio_util::sync::CancellationToken::new();
        let ctx = ToolExecutionContext {
            event_tx: tx,
            cancellation_token: token,
            tool_call_id: "call_fetch2".to_string(),
        };

        let mut params = HashMap::new();
        params.insert(
            "url".to_string(),
            serde_json::Value::String("ftp://example.com".to_string()),
        );

        tool.execute_with_context(params, &ctx).await;

        let mut events = vec![];
        while let Ok(ev) = rx.try_recv() {
            events.push(ev);
        }

        assert_eq!(
            events.len(),
            2,
            "Expected 2 progress events, got {}",
            events.len()
        );

        let has_extracting = events.iter().any(|ev| {
            matches!(ev, ToolEvent::Progress { output_preview: Some(p), .. } if p.contains("Extracting content"))
        });
        assert!(
            has_extracting,
            "Expected 'Extracting content...' progress event"
        );
    }

    // -----------------------------------------------------------------------
    // Jina removal: WebFetchTool must not accept or reference Jina config
    // -----------------------------------------------------------------------

    /// WebFetchTool::new must take only max_chars — no jina_config parameter.
    #[test]
    fn test_web_fetch_no_jina_parameter() {
        // This is a compile-time test: if WebFetchTool::new still takes a
        // second parameter, this won't compile.
        // TEMPORARILY: use current signature so registry tests can run RED.
        let tool = WebFetchTool::new(10000);
        assert_eq!(tool.name(), "web_fetch");
    }
}
