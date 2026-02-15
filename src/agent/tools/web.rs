//! Web tools: web_search and web_fetch.

use std::collections::HashMap;

use async_trait::async_trait;
use regex::Regex;
use reqwest::Client;
use url::Url;

use super::base::Tool;

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

/// Search the web using Brave Search API.
pub struct WebSearchTool {
    api_key: String,
    max_results: u32,
    client: Client,
}

impl WebSearchTool {
    /// Create a new web search tool.
    ///
    /// If `api_key` is empty/None, the `BRAVE_API_KEY` environment variable is
    /// checked.
    pub fn new(api_key: Option<String>, max_results: u32) -> Self {
        let resolved_key = api_key
            .filter(|k| !k.is_empty())
            .or_else(|| std::env::var("BRAVE_API_KEY").ok())
            .unwrap_or_default();

        Self {
            api_key: resolved_key,
            max_results,
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

        if self.api_key.is_empty() {
            return "Error: BRAVE_API_KEY not configured".to_string();
        }

        let count = params
            .get("count")
            .and_then(|v| v.as_u64())
            .map(|n| n.min(10).max(1) as u32)
            .unwrap_or(self.max_results);

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
                    let body = response.text().await.unwrap_or_default();
                    return format!("Error: Brave Search returned HTTP {}: {}", status, body);
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
            Err(e) => format!("Error: {}", e),
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

        Self { max_chars, client }
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
    use dom_smoothie::{Readability, Config, TextMode};

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
                    html_to_markdown_simple(&el.html())
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

/// Very simple HTML-to-markdown converter.
fn html_to_markdown_simple(html: &str) -> String {
    // Convert links.
    let re_links =
        Regex::new(r#"(?is)<a\s+[^>]*href=["']([^"']+)["'][^>]*>([\s\S]*?)</a>"#).unwrap();
    let text = re_links.replace_all(html, |caps: &regex::Captures| {
        let href = &caps[1];
        let inner = strip_tags(&caps[2]);
        format!("[{}]({})", inner, href)
    });

    // Convert headings (h1-h6). We can't use backreferences (\1) in the regex
    // crate, so we match any closing </hN> and extract the level from the opening tag.
    let re_headings = Regex::new(r"(?is)<h([1-6])[^>]*>([\s\S]*?)</h[1-6]>").unwrap();
    let text = re_headings.replace_all(&text, |caps: &regex::Captures| {
        let level: usize = caps[1].parse().unwrap_or(1);
        let inner = strip_tags(&caps[2]);
        format!("\n{} {}\n", "#".repeat(level), inner)
    });

    // Convert list items.
    let re_li = Regex::new(r"(?is)<li[^>]*>([\s\S]*?)</li>").unwrap();
    let text = re_li.replace_all(&text, |caps: &regex::Captures| {
        let inner = strip_tags(&caps[1]);
        format!("\n- {}", inner)
    });

    // Convert block-end tags to newlines.
    let re_block = Regex::new(r"(?i)</(p|div|section|article)>").unwrap();
    let text = re_block.replace_all(&text, "\n\n");

    // Convert br/hr.
    let re_br = Regex::new(r"(?i)<(br|hr)\s*/?>").unwrap();
    let text = re_br.replace_all(&text, "\n");

    normalize_whitespace(&strip_tags(&text))
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
    // html_to_markdown_simple tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_html_to_markdown_headings() {
        let html = "<h1>Title</h1><h2>Subtitle</h2>";
        let result = html_to_markdown_simple(html);
        assert!(result.contains("# Title"), "result: {}", result);
        assert!(result.contains("## Subtitle"), "result: {}", result);
    }

    #[test]
    fn test_html_to_markdown_links() {
        let html = r#"<a href="https://example.com">Example</a>"#;
        let result = html_to_markdown_simple(html);
        assert!(
            result.contains("[Example](https://example.com)"),
            "result: {}",
            result
        );
    }

    #[test]
    fn test_html_to_markdown_list_items() {
        let html = "<ul><li>First</li><li>Second</li></ul>";
        let result = html_to_markdown_simple(html);
        assert!(result.contains("- First"), "result: {}", result);
        assert!(result.contains("- Second"), "result: {}", result);
    }

    #[test]
    fn test_html_to_markdown_paragraphs() {
        let html = "<p>First paragraph</p><p>Second paragraph</p>";
        let result = html_to_markdown_simple(html);
        assert!(result.contains("First paragraph"), "result: {}", result);
        assert!(result.contains("Second paragraph"), "result: {}", result);
    }

    #[test]
    fn test_html_to_markdown_br() {
        let html = "line1<br/>line2";
        let result = html_to_markdown_simple(html);
        assert!(result.contains("line1"), "result: {}", result);
        assert!(result.contains("line2"), "result: {}", result);
    }

    #[test]
    fn test_html_to_markdown_strips_remaining_tags() {
        let html = "<div><span>text</span></div>";
        let result = html_to_markdown_simple(html);
        assert!(result.contains("text"), "result: {}", result);
        assert!(!result.contains("<span>"), "result: {}", result);
        assert!(!result.contains("<div>"), "result: {}", result);
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
        let tool = WebSearchTool::new(None, 5);
        assert_eq!(tool.name(), "web_search");
    }

    #[test]
    fn test_web_search_tool_parameters() {
        let tool = WebSearchTool::new(None, 5);
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
    fn test_web_fetch_tool_parameters() {
        let tool = WebFetchTool::new(50000);
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["url"].is_object());
    }

    #[tokio::test]
    async fn test_web_search_no_api_key() {
        let tool = WebSearchTool::new(Some(String::new()), 5);
        let mut params = HashMap::new();
        params.insert(
            "query".to_string(),
            serde_json::Value::String("test".to_string()),
        );
        let result = tool.execute(params).await;
        assert!(result.contains("BRAVE_API_KEY not configured"));
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
}
