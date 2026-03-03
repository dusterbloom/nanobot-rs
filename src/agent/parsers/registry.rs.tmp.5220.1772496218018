use serde_json::Value;

/// A parsed tool call extracted from model text output
#[derive(Debug, Clone)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: Value,
}

/// Trait for model-family-specific tool call parsers
pub trait ToolCallParser: Send + Sync {
    /// Parser identifier (e.g., "hermes", "qwen", "llama", "deepseek")
    fn name(&self) -> &str;

    /// Parse tool calls from model text output
    fn parse(&self, text: &str) -> Vec<ParsedToolCall>;

    /// Strip tool call markup from text, leaving only natural language
    fn strip(&self, text: &str) -> String;
}

/// Registry of available parsers with model-name matching
pub struct ParserRegistry {
    parsers: Vec<Box<dyn ToolCallParser>>,
}

impl ParserRegistry {
    pub fn new() -> Self {
        Self {
            parsers: vec![
                Box::new(super::qwen::QwenParser),
                Box::new(super::llama::LlamaParser),
                Box::new(super::deepseek::DeepSeekParser),
                Box::new(super::hermes::HermesParser), // last = default fallback
            ],
        }
    }

    /// Select parser by config override, model name substring match, or fallback to hermes
    pub fn select_for_model(&self, model_name: &str, config_override: Option<&str>) -> &dyn ToolCallParser {
        // 1. Config override
        if let Some(override_name) = config_override {
            if let Some(p) = self.parsers.iter().find(|p| p.name() == override_name) {
                return p.as_ref();
            }
        }

        // 2. Model name substring match
        let lower = model_name.to_lowercase();
        if lower.contains("qwen") {
            return self.parsers.iter().find(|p| p.name() == "qwen").unwrap().as_ref();
        }
        if lower.contains("llama") || lower.contains("nemotron") {
            return self.parsers.iter().find(|p| p.name() == "llama").unwrap().as_ref();
        }
        if lower.contains("deepseek") {
            return self.parsers.iter().find(|p| p.name() == "deepseek").unwrap().as_ref();
        }

        // 3. Fallback to hermes (last in list)
        self.parsers.last().unwrap().as_ref()
    }
}

impl Default for ParserRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_selects_qwen_for_qwen_model() {
        let reg = ParserRegistry::new();
        let p = reg.select_for_model("Qwen2.5-72B-Instruct", None);
        assert_eq!(p.name(), "qwen");
    }

    #[test]
    fn test_registry_selects_llama_for_llama_model() {
        let reg = ParserRegistry::new();
        let p = reg.select_for_model("Meta-Llama-3.1-70B", None);
        assert_eq!(p.name(), "llama");
    }

    #[test]
    fn test_registry_selects_llama_for_nemotron_model() {
        let reg = ParserRegistry::new();
        let p = reg.select_for_model("nvidia/nemotron-mini-4b", None);
        assert_eq!(p.name(), "llama");
    }

    #[test]
    fn test_registry_selects_deepseek_for_deepseek_model() {
        let reg = ParserRegistry::new();
        let p = reg.select_for_model("deepseek-coder-v2", None);
        assert_eq!(p.name(), "deepseek");
    }

    #[test]
    fn test_registry_fallback_to_hermes() {
        let reg = ParserRegistry::new();
        let p = reg.select_for_model("unknown-model-v1", None);
        assert_eq!(p.name(), "hermes");
    }

    #[test]
    fn test_registry_config_override() {
        let reg = ParserRegistry::new();
        let p = reg.select_for_model("Qwen2.5-72B", Some("llama"));
        assert_eq!(p.name(), "llama"); // override wins
    }

    #[test]
    fn test_registry_config_override_unknown_falls_through_to_model_match() {
        // When override name doesn't match any parser, fall through to model name matching
        let reg = ParserRegistry::new();
        let p = reg.select_for_model("Qwen2.5-72B", Some("nonexistent-parser"));
        assert_eq!(p.name(), "qwen"); // model name match wins after bad override
    }
}
