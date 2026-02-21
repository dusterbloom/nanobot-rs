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
