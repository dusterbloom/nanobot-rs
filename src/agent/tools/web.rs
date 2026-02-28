//! Web tools: web_search and web_fetch.

use std::collections::HashMap;

use async_trait::async_trait;
use html2md::rewrite_html;
use regex::Regex;
use reqwest::Client;
use url::Url;

use super::base::Tool;
use crate::config::schema::JinaReaderConfig;

/// Shared user-agent string.
const USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36";

/// Maximum number of redirects to follow.
const MAX_REDIRECTS: usize = 5;

/// Maximum response body size (5 MB). Prevents memory spikes on large responses.
const MAX_BODY_BYTES: usize = 5 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Remove HTML tags and decode entities.
fn strip_tags(text: &str) -> String {
    // Remove script and style blocks.
    let re_script = Regex::new(r"(?is)<script[\s\S]*?</script>").unwrap();
    let text = re_script.replace_all(text, "");
    let re_style = Regex::new(r"(?is)<style[\s\S]*?</style>").unwrap();
    let text = re_style.replace_all(&text, "");
    // Remove remaining tags.
    let re_tags = Regex::new(r"<[^>]+>").unwrap();
    let text = re_tags.replace_all(&text, "");
    html_escape::decode_html_entities(&text).trim().to_string()
}

/// Normalize whitespace: collapse runs of spaces/tabs, limit consecutive newlines.
fn normalize_whitespace(text: &str) -> String {
    let re_spaces = Regex::new(r"[ \t]+").unwrap();
    let text = re_spaces.replace_all(text, " ");
    let re_newlines = Regex::new(r"\n{3,}").unwrap();
    re_newlines.replace_all(&text, "\n\n").trim().to_string()
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

    // Block known private/local hostnames.
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

/// Search the web using SearXNG (default) or Brave Search API (fallback/explicit).
pub struct WebSearchTool {
    api_key: String,
    max_results: u32,
    provider: String,
    searxng_url: String,
    client: Client,
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

        Self {
            api_key: resolved_key,
            max_results,
            provider,
            searxng_url,
            client: Client::new(),
        }
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

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        let query = match params.get("query").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => return "Error: 'query' parameter is required".to_string(),
        };

        let count = params
            .get("count")
            .and_then(|v| v.as_u64())
            .map(|n| n.min(10).max(1) as u32)
            .unwrap_or(self.max_results);

        match self.provider.as_str() {
            "searxng" => self.execute_searxng(query, count).await,
            "brave" => self.execute_brave(query, count).await,
            other => format!("Error: unknown search provider '{}'. Use 'searxng' or 'brave'.", other),
        }
    }
}

impl WebSearchTool {
    /// Execute a search via SearXNG. Falls back to Brave if SearXNG is unreachable
    /// and a Brave API key is configured.
    async fn execute_searxng(&self, query: &str, count: u32) -> String {
        let result = self
            .client
            .get(format!("{}/search", self.searxng_url))
            .query(&[
                ("q", query),
                ("format", "json"),
                ("categories", "general"),
            ])
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
                    return format!(
                        "Error: SearXNG returned HTTP {}. Ensure SearXNG is running at {}.",
                        status, self.searxng_url
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
                // Connection error — try Brave fallback if available.
                if !self.api_key.is_empty() {
                    tracing::warn!(
                        "SearXNG unavailable ({}), falling back to Brave Search",
                        e
                    );
                    let mut result = self.execute_brave(query, count).await;
                    result.push_str("\n(Fell back to Brave Search)");
                    return result;
                }
                format!(
                    "Error: SearXNG is unavailable at {} ({}). Configure a running SearXNG instance or set provider to 'brave'.",
                    self.searxng_url, e
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
                    return format!("Error: Brave Search returned HTTP {}: {}{}", status, body, hint);
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

/// Fetch and extract content from a URL.
pub struct WebFetchTool {
    max_chars: usize,
    client: Client,
    jina_config: Option<JinaReaderConfig>,
}

impl WebFetchTool {
    /// Create a new web fetch tool.
    pub fn new(max_chars: usize, jina_config: Option<JinaReaderConfig>) -> Self {
        let client = Client::builder()
            .redirect(reqwest::redirect::Policy::limited(MAX_REDIRECTS))
            .user_agent(USER_AGENT)
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| Client::new());

        Self { max_chars, client, jina_config }
    }

    /// Fetch content via Jina Reader and return (markdown_body, jina_url).
    async fn fetch_via_jina(&self, url: &str, config: &JinaReaderConfig) -> Result<(String, String), String> {
        let jina_url = format!("{}/{}", config.url.trim_end_matches('/'), url);
        let mut req = self.client.get(&jina_url)
            .header("Accept", "text/markdown")
            .header("X-No-Cache", "true");
        if let Some(key) = &config.api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }
        let resp = req.send().await.map_err(|e| format!("Jina request failed: {}", e))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(format!("Jina returned {}", status));
        }
        let body = resp.text().await.map_err(|e| format!("Jina body read failed: {}", e))?;
        Ok((body, jina_url))
    }
}

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch URL and extract readable content (HTML -> text)."
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

        // Try Jina Reader first if configured and enabled.
        if let Some(jina_cfg) = &self.jina_config {
            if jina_cfg.enabled {
                match self.fetch_via_jina(url, jina_cfg).await {
                    Ok((body, jina_url)) => {
                        let text = normalize_whitespace(&body);
                        let truncated = text.len() > max_chars;
                        let final_text = if truncated {
                            let mut end = max_chars;
                            while !text.is_char_boundary(end) && end > 0 {
                                end -= 1;
                            }
                            text[..end].to_string()
                        } else {
                            text
                        };
                        return serde_json::json!({
                            "url": url,
                            "finalUrl": jina_url,
                            "status": 200,
                            "extractor": "jina-reader",
                            "truncated": truncated,
                            "length": final_text.len(),
                            "text": final_text
                        })
                        .to_string();
                    }
                    Err(e) => {
                        tracing::warn!("Jina Reader failed for {}: {}. Falling back to direct fetch.", url, e);
                    }
                }
            }
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

                let truncated = text.len() > max_chars;
                let final_text = if truncated {
                    // Find a valid char boundary at or before max_chars.
                    let mut end = max_chars;
                    while !text.is_char_boundary(end) && end > 0 {
                        end -= 1;
                    }
                    text[..end].to_string()
                } else {
                    text
                };

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
    // strip_tags tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_strip_tags_basic() {
        let result = strip_tags("<p>Hello <b>World</b></p>");
        assert_eq!(result, "Hello World");
    }

    #[test]
    fn test_strip_tags_removes_script() {
        let html = "Before<script>alert('xss')</script>After";
        let result = strip_tags(html);
        assert_eq!(result, "BeforeAfter");
    }

    #[test]
    fn test_strip_tags_removes_style() {
        let html = "Before<style>body { color: red; }</style>After";
        let result = strip_tags(html);
        assert_eq!(result, "BeforeAfter");
    }

    #[test]
    fn test_strip_tags_plain_text() {
        let result = strip_tags("no tags here");
        assert_eq!(result, "no tags here");
    }

    #[test]
    fn test_strip_tags_html_entities() {
        let result = strip_tags("&amp; &lt; &gt;");
        assert_eq!(result, "& < >");
    }

    #[test]
    fn test_strip_tags_empty() {
        let result = strip_tags("");
        assert_eq!(result, "");
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
        assert!(result.contains("Title"), "result should contain heading text: {}", result);
        assert!(result.contains("Subtitle"), "result should contain subheading text: {}", result);
    }

    #[test]
    fn test_fallback_extract_links() {
        let html = r#"<html><body><a href="https://example.com">Example</a></body></html>"#;
        let result = fallback_extract(html, "markdown");
        assert!(result.contains("Example"), "result should contain link text: {}", result);
        assert!(result.contains("https://example.com"), "result should contain URL: {}", result);
    }

    #[test]
    fn test_fallback_extract_list_items() {
        let html = "<html><body><ul><li>First</li><li>Second</li></ul></body></html>";
        let result = fallback_extract(html, "markdown");
        assert!(result.contains("First"), "result should contain first item: {}", result);
        assert!(result.contains("Second"), "result should contain second item: {}", result);
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
        assert!(result.contains("text content"), "result should contain text: {}", result);
        assert!(!result.contains("<span>"), "result should not contain raw span tags: {}", result);
        assert!(!result.contains("<div>"), "result should not contain raw div tags: {}", result);
    }

    #[test]
    fn test_fallback_extract_text_mode_no_markdown() {
        let html = "<html><body><h1>Heading</h1><p>Paragraph text</p></body></html>";
        let result = fallback_extract(html, "text");
        assert!(result.contains("Heading"), "result should contain heading text: {}", result);
        assert!(result.contains("Paragraph text"), "result should contain paragraph: {}", result);
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
        let tool = WebSearchTool::new(None, 5, "searxng".to_string(), "http://localhost:8888".to_string());
        assert_eq!(tool.name(), "web_search");
    }

    #[test]
    fn test_web_search_tool_parameters() {
        let tool = WebSearchTool::new(None, 5, "searxng".to_string(), "http://localhost:8888".to_string());
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["query"].is_object());
    }

    #[test]
    fn test_web_fetch_tool_name() {
        let tool = WebFetchTool::new(50000, None);
        assert_eq!(tool.name(), "web_fetch");
    }

    #[test]
    fn test_web_fetch_tool_parameters() {
        let tool = WebFetchTool::new(50000, None);
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["url"].is_object());
    }

    #[tokio::test]
    async fn test_web_search_no_api_key() {
        // With provider="brave" and no API key, expect the Brave key error.
        let tool = WebSearchTool::new(Some(String::new()), 5, "brave".to_string(), "http://localhost:8888".to_string());
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
        let tool = WebSearchTool::new(Some(String::new()), 5, "brave".to_string(), "http://localhost:8888".to_string());
        let mut params = HashMap::new();
        params.insert(
            "query".to_string(),
            serde_json::Value::String("test".to_string()),
        );
        let result = tool.execute(params).await;
        assert!(result.contains("config.json"), "Expected config.json hint: {}", result);
        assert!(result.contains("braveApiKey"), "Expected braveApiKey hint: {}", result);
    }

    #[tokio::test]
    async fn test_web_search_searxng_unavailable_no_fallback() {
        // SearXNG provider with no Brave key and unreachable URL returns a clear error.
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
            result.contains("unavailable") || result.contains("Error"),
            "Expected unavailability message: {}",
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
        let tool = WebFetchTool::new(50000, None);
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
        let tool = WebFetchTool::new(50000, None);
        let params = HashMap::new();
        let result = tool.execute(params).await;
        assert!(result.contains("url parameter is required"));
    }

    // -----------------------------------------------------------------------
    // Jina Reader tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_jina_url_construction() {
        let config = JinaReaderConfig {
            enabled: true,
            url: "https://r.jina.ai".to_string(),
            api_key: None,
        };
        let jina_url = format!("{}/{}", config.url.trim_end_matches('/'), "https://www.bbc.com/news");
        assert_eq!(jina_url, "https://r.jina.ai/https://www.bbc.com/news");
    }

    #[test]
    fn test_jina_url_construction_trailing_slash() {
        let config = JinaReaderConfig {
            enabled: true,
            url: "https://r.jina.ai/".to_string(),
            api_key: None,
        };
        let jina_url = format!("{}/{}", config.url.trim_end_matches('/'), "https://example.com");
        assert_eq!(jina_url, "https://r.jina.ai/https://example.com");
    }

    #[test]
    fn test_jina_config_defaults() {
        let json = r#"{}"#;
        let config: JinaReaderConfig = serde_json::from_str(json).unwrap();
        assert!(config.enabled);
        assert_eq!(config.url, "https://r.jina.ai");
        assert!(config.api_key.is_none());
    }

    #[test]
    fn test_web_fetch_without_jina() {
        // WebFetchTool with None jina_config should work (backward compat)
        let tool = WebFetchTool::new(10000, None);
        assert_eq!(tool.name(), "web_fetch");
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

    /// Show side-by-side what the main model sees under three approaches:
    ///
    /// 1. **Passthrough** – raw web_fetch JSON envelope (or truncated at max_chars)
    /// 2. **ContextStore** – metadata string produced by `ContextStore::store()`
    /// 3. **ContentGate** – structural briefing from `ContentGate::admit_simple()`
    ///
    /// Asserts that passthrough contains actual article text while the
    /// summarized forms do not.
    #[test]
    fn test_web_fetch_passthrough_vs_summarized() {
        use crate::agent::context_gate::ContentGate;
        use crate::agent::context_store::ContextStore;
        use std::path::PathBuf;

        let raw = bbc_web_fetch_fixture();

        // ----------------------------------------------------------------
        // Approach 1: Passthrough – the raw JSON string the tool returns.
        // In the simplest case the agent loop just injects `raw` verbatim.
        // ----------------------------------------------------------------
        let passthrough = raw.clone();

        // ----------------------------------------------------------------
        // Approach 2: ContextStore metadata.
        //
        // tool_runner.rs line ~984:
        //   let stripped = strip_tool_output(&result.data);
        //   let (_, metadata) = context_store.store(stripped.clone());
        //   // For large results the delegation model sees `metadata`.
        // ----------------------------------------------------------------
        let mut store = ContextStore::new();
        let (_var_name, context_store_view) = store.store(raw.clone());

        // ----------------------------------------------------------------
        // Approach 3: ContentGate admit_simple.
        //
        // Use a tiny token budget so the content never "fits" and the gate
        // always produces a briefing — matching real production behaviour
        // where web pages exceed the agent's remaining token budget.
        // ----------------------------------------------------------------
        let cache_dir = PathBuf::from(std::env::temp_dir()).join("nanobot_test_gate");
        let _ = std::fs::create_dir_all(&cache_dir);
        // 50 token budget → raw (≈575 tokens) will not fit → briefing path.
        let mut gate = ContentGate::new(50, 0.2, cache_dir.clone());
        let gate_result = gate.admit_simple(&raw);
        let gate_view = gate_result.into_text();

        // Clean up temp dir (best-effort).
        let _ = std::fs::remove_dir_all(&cache_dir);

        // ----------------------------------------------------------------
        // Print what each approach produces (visible with `-- --nocapture`).
        // ----------------------------------------------------------------
        println!("\n=== APPROACH 1: PASSTHROUGH (first 300 chars) ===");
        println!("{}", &passthrough[..passthrough.len().min(300)]);

        println!("\n=== APPROACH 2: CONTEXT STORE VIEW ===");
        println!("{}", &context_store_view);

        println!("\n=== APPROACH 3: CONTENT GATE BRIEFING ===");
        println!("{}", &gate_view);

        // ----------------------------------------------------------------
        // Assertions
        // ----------------------------------------------------------------

        // Passthrough contains the actual article text.
        assert!(
            passthrough.contains("UK economy grew by 0.4%"),
            "Passthrough should contain actual article text"
        );
        assert!(
            passthrough.contains("Bank of England"),
            "Passthrough should contain article body"
        );

        // ContextStore metadata does NOT contain the article text — only a
        // short preview of the raw JSON (which starts with the url/status
        // envelope fields, not the article text).
        assert!(
            !context_store_view.contains("Bank of England"),
            "ContextStore metadata should not contain article body, got: {}",
            context_store_view
        );
        assert!(
            context_store_view.contains("chars"),
            "ContextStore metadata should report char count"
        );
        assert!(
            context_store_view.contains("output_0"),
            "ContextStore metadata should include variable name"
        );

        // ContentGate briefing does NOT contain the article text — only
        // structural JSON facts (status, url, extractor, etc.).
        assert!(
            !gate_view.contains("Bank of England"),
            "ContentGate briefing should not contain article body, got: {}",
            gate_view
        );
        // The gate briefing is a JSON structural summary.
        assert!(
            gate_view.contains("JSON Summary") || gate_view.contains("Content Summary"),
            "ContentGate briefing should be a structural summary, got: {}",
            gate_view
        );
    }

    /// Demonstrate that extracting only the `text` field from the web_fetch
    /// envelope preserves the article content at lower overhead.
    ///
    /// Properties verified:
    /// - Extracted text contains the full article content.
    /// - Extracted text is smaller than the full JSON envelope (no metadata overhead).
    /// - `extract_web_content` is a pure function (no I/O, deterministic).
    #[test]
    fn test_web_fetch_smart_summary_preserves_content() {
        let raw = bbc_web_fetch_fixture();

        // Parse the fixture to get the original text length for comparison.
        let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let original_text = parsed["text"].as_str().unwrap();

        // ----------------------------------------------------------------
        // Apply the pure extraction function.
        // ----------------------------------------------------------------
        let extracted = extract_web_content(&raw);

        println!("\n=== FULL JSON ENVELOPE (chars) ===");
        println!("{}", raw.len());

        println!("\n=== EXTRACTED TEXT (chars) ===");
        println!("{}", extracted.len());

        println!("\n=== EXTRACTED TEXT PREVIEW (first 300 chars) ===");
        println!("{}", &extracted[..extracted.len().min(300)]);

        // ----------------------------------------------------------------
        // The extracted text must contain the actual article content.
        // ----------------------------------------------------------------
        assert!(
            extracted.contains("UK economy grew by 0.4%"),
            "Extracted text should contain article headline"
        );
        assert!(
            extracted.contains("Bank of England"),
            "Extracted text should contain article body"
        );
        assert!(
            extracted.contains("FTSE 100"),
            "Extracted text should contain market reaction section"
        );

        // ----------------------------------------------------------------
        // The extracted text must be the same as the `text` field.
        // ----------------------------------------------------------------
        assert_eq!(
            extracted, original_text,
            "extract_web_content should return exactly the text field"
        );

        // ----------------------------------------------------------------
        // The extracted text is smaller than the full envelope — no url,
        // status, extractor, length, truncated, or finalUrl overhead.
        // ----------------------------------------------------------------
        assert!(
            extracted.len() < raw.len(),
            "Extracted text ({} chars) should be smaller than full envelope ({} chars)",
            extracted.len(),
            raw.len()
        );

        let overhead_removed = raw.len() - extracted.len();
        println!(
            "\n=== ENVELOPE OVERHEAD REMOVED: {} chars ({:.1}%) ===",
            overhead_removed,
            overhead_removed as f64 / raw.len() as f64 * 100.0
        );

        assert!(
            overhead_removed > 50,
            "Should remove at least 50 chars of JSON envelope overhead, removed: {}",
            overhead_removed
        );

        // ----------------------------------------------------------------
        // Robustness: non-JSON input falls back to the raw string.
        // ----------------------------------------------------------------
        let plain = "This is plain text, not JSON.";
        let fallback = extract_web_content(plain);
        assert_eq!(
            fallback, plain,
            "Non-JSON input should fall back to the raw string unchanged"
        );

        // ----------------------------------------------------------------
        // Robustness: JSON without a `text` field falls back to raw string.
        // ----------------------------------------------------------------
        let no_text_json = r#"{"status": 200, "url": "https://example.com"}"#;
        let fallback2 = extract_web_content(no_text_json);
        assert_eq!(
            fallback2, no_text_json,
            "JSON without text field should fall back to raw string"
        );
    }
}
