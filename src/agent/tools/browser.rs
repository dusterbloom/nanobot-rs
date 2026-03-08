//! Browser automation tool backed by the `agent-browser` CLI
//! (<https://github.com/vercel-labs/agent-browser>).
//!
//! Install: `npm install -g agent-browser && agent-browser install`

use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use tokio::process::Command;

use super::base::Tool;

/// Default command timeout in seconds.
const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Maximum output length returned to the LLM.
const MAX_OUTPUT_CHARS: usize = 16_000;

/// Cached result of checking whether `agent-browser` is on PATH.
static BINARY_AVAILABLE: OnceLock<bool> = OnceLock::new();

fn check_binary() -> bool {
    // Try `agent-browser --version` directly — works on all platforms.
    std::process::Command::new("agent-browser")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

const INSTALL_HINT: &str =
    "agent-browser is not installed. Run: npm install -g agent-browser && agent-browser install";

/// Browser automation tool using the `agent-browser` CLI.
///
/// Provides headless browser actions (open, click, type, etc.) that the LLM
/// can invoke to interact with JavaScript-rendered pages and SPAs.
pub struct BrowserTool {
    timeout: Duration,
    max_output_chars: usize,
}

impl BrowserTool {
    pub fn new(max_output_chars: usize) -> Self {
        Self {
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            max_output_chars,
        }
    }
}

#[async_trait]
impl Tool for BrowserTool {
    fn name(&self) -> &str {
        "browser"
    }

    fn description(&self) -> &str {
        "Control a headless browser. Actions: open (navigate to URL and get page snapshot), \
         snapshot (get current page accessibility tree), click (click element by ref like @e2), \
         type (type text into element), fill (clear and fill field), search (Google search), \
         scroll (up/down/left/right), back (navigate back), forward (navigate forward), \
         hover (hover element), press (press key like Enter/Tab), \
         get_text (extract element text), get_html (get element HTML), \
         screenshot (save screenshot), eval (run JavaScript), wait (wait for element/time), \
         close (end session)."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["open", "snapshot", "click", "type", "fill", "search",
                             "scroll", "back", "forward", "hover", "press",
                             "get_text", "get_html", "screenshot", "eval", "wait", "close"],
                    "description": "The browser action to perform."
                },
                "url": {
                    "type": "string",
                    "description": "URL to navigate to (for 'open' action)."
                },
                "ref": {
                    "type": "string",
                    "description": "Element reference from snapshot, e.g. '@e2' (for click/type/fill/hover/get_text/get_html)."
                },
                "text": {
                    "type": "string",
                    "description": "Text to type or fill (for 'type'/'fill' actions), or key name (for 'press', e.g. 'Enter', 'Tab')."
                },
                "query": {
                    "type": "string",
                    "description": "Search query (for 'search' action), or wait target (CSS selector or milliseconds for 'wait')."
                },
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                    "description": "Scroll direction (for 'scroll' action)."
                },
                "javascript": {
                    "type": "string",
                    "description": "JavaScript code to evaluate (for 'eval' action)."
                },
                "interactive": {
                    "type": "boolean",
                    "description": "Include interactive element refs in snapshot (default true)."
                }
            },
            "required": ["action"]
        })
    }

    fn is_available(&self) -> bool {
        *BINARY_AVAILABLE.get_or_init(check_binary)
    }

    async fn execute(&self, params: HashMap<String, Value>) -> String {
        // Fail fast if binary is missing — don't waste tokens on param validation.
        if !*BINARY_AVAILABLE.get_or_init(check_binary) {
            return format!("Error: {}", INSTALL_HINT);
        }

        let action = match params.get("action").and_then(|v| v.as_str()) {
            Some(a) => a,
            None => return "Error: 'action' parameter is required".to_string(),
        };

        let result = match action {
            "open" => self.do_open(&params).await,
            "snapshot" => self.do_snapshot(&params).await,
            "click" => self.do_click(&params).await,
            "type" => self.do_type(&params).await,
            "fill" => self.do_fill(&params).await,
            "search" => self.do_search(&params).await,
            "scroll" => self.do_scroll(&params).await,
            "back" => self.do_simple("back").await,
            "forward" => self.do_simple("forward").await,
            "hover" => self.do_ref_action(&params, "hover").await,
            "press" => self.do_press(&params).await,
            "get_text" => self.do_get(&params, "text").await,
            "get_html" => self.do_get(&params, "html").await,
            "screenshot" => self.do_screenshot().await,
            "eval" => self.do_eval(&params).await,
            "wait" => self.do_wait(&params).await,
            "close" => self.do_simple("close").await,
            other => return format!("Error: unknown action '{}'", other),
        };

        match result {
            Ok(output) => truncate_output(&output, self.max_output_chars),
            Err(e) => format!("Error: {}", e),
        }
    }
}

impl BrowserTool {
    async fn run_cmd(&self, args: &[&str]) -> Result<String, String> {
        if !*BINARY_AVAILABLE.get_or_init(check_binary) {
            return Err(INSTALL_HINT.to_string());
        }
        let output = Command::new("agent-browser").args(args).output();

        let output = tokio::time::timeout(self.timeout, output)
            .await
            .map_err(|_| format!("agent-browser timed out after {}s", self.timeout.as_secs()))?
            .map_err(|e| format!("failed to run agent-browser: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if output.status.success() {
            Ok(if stdout.is_empty() { stderr } else { stdout })
        } else {
            let msg = if stderr.is_empty() { &stdout } else { &stderr };
            Err(format!(
                "agent-browser exited with {}: {}",
                output.status,
                msg.trim()
            ))
        }
    }

    fn get_str<'a>(params: &'a HashMap<String, Value>, key: &str) -> Option<&'a str> {
        params.get(key).and_then(|v| v.as_str())
    }

    fn require_str<'a>(
        params: &'a HashMap<String, Value>,
        key: &str,
        action: &str,
    ) -> Result<&'a str, String> {
        Self::get_str(params, key)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| format!("'{}' action requires '{}' parameter", action, key))
    }

    // ── Actions ──────────────────────────────────────────────────────

    async fn do_open(&self, params: &HashMap<String, Value>) -> Result<String, String> {
        let url = Self::require_str(params, "url", "open")?;
        let nav = self.run_cmd(&["open", url]).await?;
        // Auto-snapshot after navigation for immediate context.
        let snap = self.run_cmd(&["snapshot", "-i"]).await.unwrap_or_default();
        Ok(format!("{}\n\n{}", nav.trim(), snap.trim()))
    }

    async fn do_snapshot(&self, params: &HashMap<String, Value>) -> Result<String, String> {
        let interactive = params
            .get("interactive")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        if interactive {
            self.run_cmd(&["snapshot", "-i"]).await
        } else {
            self.run_cmd(&["snapshot"]).await
        }
    }

    async fn do_click(&self, params: &HashMap<String, Value>) -> Result<String, String> {
        let r = Self::require_str(params, "ref", "click")?;
        self.run_cmd(&["click", r]).await
    }

    async fn do_type(&self, params: &HashMap<String, Value>) -> Result<String, String> {
        let r = Self::require_str(params, "ref", "type")?;
        let text = Self::require_str(params, "text", "type")?;
        self.run_cmd(&["type", r, text]).await
    }

    async fn do_fill(&self, params: &HashMap<String, Value>) -> Result<String, String> {
        let r = Self::require_str(params, "ref", "fill")?;
        let text = Self::require_str(params, "text", "fill")?;
        self.run_cmd(&["fill", r, text]).await
    }

    async fn do_search(&self, params: &HashMap<String, Value>) -> Result<String, String> {
        let query = Self::require_str(params, "query", "search")?;
        let encoded = url::form_urlencoded::byte_serialize(query.as_bytes()).collect::<String>();
        let url = format!("https://www.google.com/search?q={}", encoded);
        let nav = self.run_cmd(&["open", &url]).await?;
        let snap = self.run_cmd(&["snapshot", "-i"]).await.unwrap_or_default();
        Ok(format!("{}\n\n{}", nav.trim(), snap.trim()))
    }

    async fn do_scroll(&self, params: &HashMap<String, Value>) -> Result<String, String> {
        let dir = Self::require_str(params, "direction", "scroll")?;
        match dir {
            "up" | "down" | "left" | "right" => self.run_cmd(&["scroll", dir]).await,
            _ => Err("'direction' must be 'up', 'down', 'left', or 'right'".to_string()),
        }
    }

    /// Simple no-arg commands: back, forward, close.
    async fn do_simple(&self, cmd: &str) -> Result<String, String> {
        self.run_cmd(&[cmd]).await
    }

    /// Actions that take a single `ref` param: hover, etc.
    async fn do_ref_action(
        &self,
        params: &HashMap<String, Value>,
        action: &str,
    ) -> Result<String, String> {
        let r = Self::require_str(params, "ref", action)?;
        self.run_cmd(&[action, r]).await
    }

    async fn do_press(&self, params: &HashMap<String, Value>) -> Result<String, String> {
        let key = Self::require_str(params, "text", "press")?;
        self.run_cmd(&["press", key]).await
    }

    /// `get text @ref` / `get html @ref`
    async fn do_get(&self, params: &HashMap<String, Value>, what: &str) -> Result<String, String> {
        let r = Self::require_str(params, "ref", &format!("get_{}", what))?;
        self.run_cmd(&["get", what, r]).await
    }

    async fn do_screenshot(&self) -> Result<String, String> {
        self.run_cmd(&["screenshot"]).await
    }

    async fn do_eval(&self, params: &HashMap<String, Value>) -> Result<String, String> {
        let js = Self::require_str(params, "javascript", "eval")?;
        self.run_cmd(&["eval", js]).await
    }

    async fn do_wait(&self, params: &HashMap<String, Value>) -> Result<String, String> {
        let target = Self::require_str(params, "query", "wait")?;
        self.run_cmd(&["wait", target]).await
    }
}

/// Truncate output to a maximum character count, appending a note if truncated.
fn truncate_output(output: &str, max_chars: usize) -> String {
    if output.len() <= max_chars {
        output.to_string()
    } else {
        let truncated = &output[..max_chars];
        format!(
            "{}\n\n[Output truncated — {} chars total]",
            truncated,
            output.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_browser_tool_name() {
        let tool = BrowserTool::new(MAX_OUTPUT_CHARS);
        assert_eq!(tool.name(), "browser");
    }

    #[test]
    fn test_browser_tool_parameters_schema() {
        let tool = BrowserTool::new(MAX_OUTPUT_CHARS);
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        let action = &params["properties"]["action"];
        assert_eq!(action["type"], "string");
        let actions = action["enum"].as_array().unwrap();
        assert_eq!(actions.len(), 17);
        assert!(actions.contains(&Value::String("open".to_string())));
        assert!(actions.contains(&Value::String("click".to_string())));
        assert!(actions.contains(&Value::String("eval".to_string())));
        assert!(actions.contains(&Value::String("hover".to_string())));
        assert!(actions.contains(&Value::String("wait".to_string())));
        let required = params["required"].as_array().unwrap();
        assert_eq!(required, &[Value::String("action".to_string())]);
    }

    #[test]
    fn test_browser_tool_schema() {
        let tool = BrowserTool::new(MAX_OUTPUT_CHARS);
        let schema = tool.to_schema();
        assert_eq!(schema["type"], "function");
        assert_eq!(schema["function"]["name"], "browser");
    }

    #[tokio::test]
    async fn test_missing_action() {
        let tool = BrowserTool::new(MAX_OUTPUT_CHARS);
        let result = tool.execute(HashMap::new()).await;
        assert!(result.starts_with("Error:"));
        // If binary available, error mentions "action"; otherwise install hint.
        if tool.is_available() {
            assert!(result.contains("action"));
        }
    }

    #[tokio::test]
    async fn test_unknown_action() {
        let tool = BrowserTool::new(MAX_OUTPUT_CHARS);
        let mut params = HashMap::new();
        params.insert("action".to_string(), Value::String("fly".to_string()));
        let result = tool.execute(params).await;
        assert!(result.starts_with("Error:"));
        if tool.is_available() {
            assert!(result.contains("fly"));
        }
    }

    #[tokio::test]
    async fn test_open_requires_url() {
        let tool = BrowserTool::new(MAX_OUTPUT_CHARS);
        let mut params = HashMap::new();
        params.insert("action".to_string(), Value::String("open".to_string()));
        let result = tool.execute(params).await;
        assert!(result.starts_with("Error:"));
        if tool.is_available() {
            assert!(result.contains("url"));
        }
    }

    #[tokio::test]
    async fn test_click_requires_ref() {
        let tool = BrowserTool::new(MAX_OUTPUT_CHARS);
        let mut params = HashMap::new();
        params.insert("action".to_string(), Value::String("click".to_string()));
        let result = tool.execute(params).await;
        assert!(result.starts_with("Error:"));
        if tool.is_available() {
            assert!(result.contains("ref"));
        }
    }

    #[tokio::test]
    async fn test_type_requires_ref_and_text() {
        let tool = BrowserTool::new(MAX_OUTPUT_CHARS);
        let mut params = HashMap::new();
        params.insert("action".to_string(), Value::String("type".to_string()));
        params.insert("ref".to_string(), Value::String("@e1".to_string()));
        // Missing 'text'
        let result = tool.execute(params).await;
        assert!(result.starts_with("Error:"));
        if tool.is_available() {
            assert!(result.contains("text"));
        }
    }

    #[tokio::test]
    async fn test_search_requires_query() {
        let tool = BrowserTool::new(MAX_OUTPUT_CHARS);
        let mut params = HashMap::new();
        params.insert("action".to_string(), Value::String("search".to_string()));
        let result = tool.execute(params).await;
        assert!(result.starts_with("Error:"));
        if tool.is_available() {
            assert!(result.contains("query"));
        }
    }

    #[tokio::test]
    async fn test_scroll_invalid_direction() {
        let tool = BrowserTool::new(MAX_OUTPUT_CHARS);
        let mut params = HashMap::new();
        params.insert("action".to_string(), Value::String("scroll".to_string()));
        params.insert(
            "direction".to_string(),
            Value::String("diagonal".to_string()),
        );
        let result = tool.execute(params).await;
        assert!(result.starts_with("Error:"));
        if tool.is_available() {
            assert!(result.contains("up") || result.contains("down"));
        }
    }

    #[test]
    fn test_truncate_output_short() {
        let output = "hello world";
        assert_eq!(truncate_output(output, 100), "hello world");
    }

    #[test]
    fn test_truncate_output_long() {
        let output = "a".repeat(200);
        let result = truncate_output(&output, 50);
        assert!(result.contains("[Output truncated"));
        assert!(result.starts_with(&"a".repeat(50)));
    }
}
