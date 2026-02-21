/// Strip tool output to reduce token usage before storing in context.
///
/// Handles three content types:
/// - JSON with a `"text"` field (web_fetch envelopes): strips CSS class tokens,
///   bare long navigation URLs, pipe-separated nav bars, and collapses blank lines.
/// - Raw HTML: strips tags and collapses blank lines.
/// - Plain text: strips trailing whitespace and collapses 3+ blank lines to 2.
///
/// Safety guarantee: never produces empty output from non-empty input.
pub fn strip_tool_output(text: &str) -> String {
    use once_cell::sync::Lazy;
    use regex::Regex;

    // CSS class tokens like css-1abc234 (at least 6 trailing alphanumeric chars).
    static CSS_CLASS_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"\bcss-[0-9a-zA-Z]{6,}\b").expect("invalid css class regex")
    });

    // Bare navigation URLs: http(s) URLs that are at least 40 chars and appear alone on a line.
    static NAV_URL_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?m)^\s*https?://\S{30,}\s*$").expect("invalid nav url regex")
    });

    // Pipe-separated nav bars, e.g. "Home | News | Sport | Business".
    // Require at least two pipe-separated tokens of 2-30 word chars/spaces each.
    static NAV_BAR_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?m)^[\w][\w ]{1,29}(?:\s*\|\s*[\w][\w ]{1,29}){2,}\s*$")
            .expect("invalid nav bar regex")
    });

    // Three or more consecutive blank lines.
    static EXCESS_BLANK_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"\n{3,}").expect("invalid blank line regex")
    });

    // HTML tags.
    static HTML_TAG_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"<[^>]{1,200}>").expect("invalid html tag regex")
    });

    if text.is_empty() {
        return text.to_string();
    }

    // Detect JSON envelope with a "text" field (web_fetch results).
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(text) {
        if let Some(inner) = val.get("text").and_then(|v| v.as_str()) {
            let cleaned = strip_text_content(inner, &CSS_CLASS_RE, &NAV_URL_RE, &NAV_BAR_RE, &EXCESS_BLANK_RE);
            let result = if cleaned.is_empty() { inner.trim().to_string() } else { cleaned };
            // Produce compact JSON with stripped text.
            let mut out = val.clone();
            if let Some(obj) = out.as_object_mut() {
                obj.insert("text".to_string(), serde_json::Value::String(result));
            }
            return serde_json::to_string(&out).unwrap_or_else(|_| text.to_string());
        }
        // Other JSON: compact serialize to remove whitespace bloat.
        return serde_json::to_string(&val).unwrap_or_else(|_| text.to_string());
    }

    // Detect raw HTML (starts with common HTML markers after trimming).
    let trimmed = text.trim_start();
    if trimmed.starts_with("<!") || trimmed.starts_with("<html") || trimmed.starts_with("<HTML") {
        let no_tags = HTML_TAG_RE.replace_all(text, " ");
        let cleaned = strip_text_content(&no_tags, &CSS_CLASS_RE, &NAV_URL_RE, &NAV_BAR_RE, &EXCESS_BLANK_RE);
        let result = if cleaned.trim().is_empty() { text.trim().to_string() } else { cleaned };
        return result;
    }

    // Plain text.
    let cleaned = strip_text_content(text, &CSS_CLASS_RE, &NAV_URL_RE, &NAV_BAR_RE, &EXCESS_BLANK_RE);
    if cleaned.trim().is_empty() {
        text.trim().to_string()
    } else {
        cleaned
    }
}

/// Internal helper: apply text-level cleanup rules.
fn strip_text_content(
    text: &str,
    css_re: &regex::Regex,
    url_re: &regex::Regex,
    nav_re: &regex::Regex,
    blank_re: &regex::Regex,
) -> String {
    let s = css_re.replace_all(text, "");
    let s = url_re.replace_all(&s, "");
    let s = nav_re.replace_all(&s, "");
    // Strip trailing whitespace per line.
    let lines: Vec<String> = s.lines().map(|l| l.trim_end().to_string()).collect();
    let joined = lines.join("\n");
    // Collapse 3+ blank lines to 2.
    blank_re.replace_all(&joined, "\n\n").into_owned()
}

/// Strip reasoning model artifacts from user-visible output.
///
/// Handles:
/// - `<think>...</think>` blocks (including multiline, greedy)
/// - `RESPONSE:` markers — returns only the content after the last marker
pub fn sanitize_reasoning_output(content: &str) -> String {
    use once_cell::sync::Lazy;
    use regex::Regex;

    static THINK_RE: Lazy<Regex> = Lazy::new(|| {
        // (?s) enables dotall so `.` matches newlines (multiline think blocks).
        // .*? is lazy to avoid eating content between two separate blocks.
        Regex::new(r"(?s)<think>.*?</think>").expect("invalid think regex")
    });

    // 1. Strip <think>...</think> blocks.
    let stripped = THINK_RE.replace_all(content, "");

    // 2. If a RESPONSE: marker is present, keep only the text after the last one.
    let result = if let Some(pos) = stripped.rfind("RESPONSE:") {
        stripped[pos + "RESPONSE:".len()..].to_string()
    } else {
        stripped.into_owned()
    };

    // 3. Trim surrounding whitespace.
    let trimmed = result.trim().to_string();

    // 4. If stripping left nothing, return the original content trimmed.
    if trimmed.is_empty() {
        content.trim().to_string()
    } else {
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- strip_tool_output tests ---

    #[test]
    fn test_strip_css_class_tokens() {
        let input = "Welcome css-1abc234 to css-XYZ789ab the site.";
        let result = strip_tool_output(input);
        assert!(!result.contains("css-1abc234"), "CSS class token should be removed");
        assert!(!result.contains("css-XYZ789ab"), "CSS class token should be removed");
        assert!(result.contains("Welcome"), "surrounding text should remain");
    }

    #[test]
    fn test_strip_nav_bar() {
        let input = "Home | News | Sport | Business\nSome article content here.";
        let result = strip_tool_output(input);
        assert!(!result.contains("Home | News | Sport"), "nav bar should be stripped");
        assert!(result.contains("Some article content here."), "content should remain");
    }

    #[test]
    fn test_strip_bare_navigation_url() {
        let input = "Article title\nhttps://example.com/very/long/navigation/path/that/is/quite/long\nArticle body.";
        let result = strip_tool_output(input);
        assert!(!result.contains("https://example.com/very/long/navigation/path"), "long nav URL should be removed");
        assert!(result.contains("Article title"), "title should remain");
        assert!(result.contains("Article body."), "body should remain");
    }

    #[test]
    fn test_collapse_blank_lines() {
        let input = "Paragraph one.\n\n\n\nParagraph two.";
        let result = strip_tool_output(input);
        assert!(!result.contains("\n\n\n"), "3+ blank lines should be collapsed to 2");
        assert!(result.contains("Paragraph one."), "first paragraph should remain");
        assert!(result.contains("Paragraph two."), "second paragraph should remain");
    }

    #[test]
    fn test_json_compaction() {
        let input = "{\n  \"key\": \"value\",\n  \"num\": 42\n}";
        let result = strip_tool_output(input);
        // Should be compacted (no indentation whitespace)
        assert!(!result.contains("  "), "compact JSON should have no indentation");
        assert!(result.contains("\"key\""), "key should remain");
        assert!(result.contains("\"value\""), "value should remain");
    }

    #[test]
    fn test_web_fetch_envelope_strips_inner_text() {
        let css_noise = "css-1abc234";
        let input = format!(
            r#"{{"url":"https://example.com","text":"Welcome {} to the page.\nHome | News | Sport\nActual content here."}}"#,
            css_noise
        );
        let result = strip_tool_output(&input);
        assert!(!result.contains("css-1abc234"), "CSS class token should be stripped from inner text");
        assert!(result.contains("Actual content here."), "content should remain");
        // Result should still be valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&result)
            .expect("result should be valid JSON");
        assert!(parsed.get("url").is_some(), "url field should remain");
        assert!(parsed.get("text").is_some(), "text field should remain");
    }

    #[test]
    fn test_safety_never_empty_from_non_empty_input() {
        // Even if the content is all CSS tokens, the function must not return empty.
        let input = "css-1abc234 css-XYZ789ab css-abcdef01";
        let result = strip_tool_output(input);
        // The safety guarantee: non-empty input never produces empty output.
        // Regex leaves spaces/whitespace, but trim-based fallback covers the case.
        // Here the result may be whitespace from stripped tokens — verify it at
        // the pure-text level by checking the json/html path doesn't apply.
        // The plain text fallback: if cleaned.trim().is_empty() -> return original trimmed.
        // Since `input` is non-empty, result must be non-empty.
        assert!(!result.trim().is_empty(), "non-empty input must produce non-empty output");
    }

    #[test]
    fn test_plain_text_passthrough() {
        let input = "This is a simple plain text response with no noise.";
        let result = strip_tool_output(input);
        assert_eq!(result.trim(), input.trim(), "clean plain text should pass through unchanged");
    }

    // --- sanitize_reasoning_output tests ---

    #[test]
    fn test_strips_think_blocks() {
        let input = "<think>internal reasoning</think>Hello user!";
        assert_eq!(sanitize_reasoning_output(input), "Hello user!");
    }

    #[test]
    fn test_strips_multiline_think() {
        let input = "<think>\nstep 1\nstep 2\n</think>\nThe answer is 42.";
        assert_eq!(sanitize_reasoning_output(input), "The answer is 42.");
    }

    #[test]
    fn test_extracts_after_response_marker() {
        let input = "thinking...\nRESPONSE: Here is my answer.";
        assert_eq!(sanitize_reasoning_output(input), "Here is my answer.");
    }

    #[test]
    fn test_combined_think_and_response() {
        let input = "<think>reasoning</think>\nRESPONSE: Final answer.";
        assert_eq!(sanitize_reasoning_output(input), "Final answer.");
    }

    #[test]
    fn test_no_artifacts_passthrough() {
        let input = "Normal response without any artifacts.";
        assert_eq!(
            sanitize_reasoning_output(input),
            "Normal response without any artifacts."
        );
    }

    #[test]
    fn test_empty_after_strip_returns_original() {
        let input = "<think>only thinking</think>";
        // Stripping leaves empty — fall back to original trimmed.
        assert_eq!(sanitize_reasoning_output(input), "<think>only thinking</think>");
    }

    #[test]
    fn test_multiple_think_blocks() {
        let input = "<think>first</think>Hello <think>second</think>world";
        assert_eq!(sanitize_reasoning_output(input), "Hello world");
    }

    #[test]
    fn test_specialist_output_format_preserved() {
        // Specialist output that would be wrapped in [specialist:name] format
        let input = "<think>analyzing the query...</think>The file contains sensitive data.";
        let sanitized = sanitize_reasoning_output(input);
        assert_eq!(sanitized, "The file contains sensitive data.");
        // Verify wrapping still works after sanitization
        let wrapped = format!("[specialist:analyst] {}", sanitized);
        assert!(wrapped.starts_with("[specialist:analyst]"));
    }

    // --- Integration tests: realistic model output ---

    #[test]
    fn test_realistic_deepseek_r1_output() {
        // DeepSeek R1 produces multi-paragraph think blocks with code
        let input = r#"<think>
The user wants to know about Rust's ownership system.
Let me think about the key concepts:
1. Each value has exactly one owner
2. When the owner goes out of scope, the value is dropped
3. Ownership can be transferred (moved) or borrowed

I should explain this clearly with an example.
</think>

Rust's ownership system has three key rules:

1. **Each value has one owner** — the variable that holds it
2. **Values are dropped when the owner goes out of scope**
3. **Ownership can be moved or borrowed**

```rust
let s1 = String::from("hello");
let s2 = s1; // s1 is moved to s2
// println!("{}", s1); // ERROR: s1 no longer valid
```"#;
        let result = sanitize_reasoning_output(input);
        assert!(!result.contains("<think>"), "think block should be stripped");
        assert!(!result.contains("</think>"), "think close tag should be stripped");
        assert!(
            result.contains("Rust's ownership system"),
            "response content should remain"
        );
        assert!(result.contains("```rust"), "code blocks should be preserved");
    }

    #[test]
    fn test_realistic_qwen_qwq_with_response_marker() {
        let input = r#"<think>
I need to calculate 15 * 23.
15 * 20 = 300
15 * 3 = 45
300 + 45 = 345
</think>
RESPONSE: The answer is 345."#;
        let result = sanitize_reasoning_output(input);
        assert_eq!(result, "The answer is 345.");
    }

    #[test]
    fn test_realistic_mixed_artifacts_in_tool_summary() {
        // Subagent results sometimes contain thinking artifacts
        let input = "<think>Let me analyze the file...</think>The file contains 3 functions: main(), parse(), and validate().";
        let result = sanitize_reasoning_output(input);
        assert_eq!(
            result,
            "The file contains 3 functions: main(), parse(), and validate()."
        );
    }
}
