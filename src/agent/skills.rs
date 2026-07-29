#![allow(dead_code)]
//! Skills loader for agent capabilities.
//!
//! Skills are markdown files (`SKILL.md`) that teach the agent how to use
//! specific tools or perform certain tasks.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;

static RE_FRONTMATTER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)^---\n(.*?)\n---").unwrap());
static RE_FRONTMATTER_STRIP: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)^---\n.*?\n---\n").unwrap());
use tracing::debug;

/// Information about a discovered skill.
#[derive(Debug, Clone)]
pub struct SkillInfo {
    pub name: String,
    pub path: String,
    pub source: String,
}

#[derive(Debug, Clone)]
struct SkillRecord {
    info: SkillInfo,
    metadata: Option<HashMap<String, String>>,
    skill_meta: HashMap<String, serde_json::Value>,
}

impl SkillRecord {
    fn description(&self) -> String {
        self.metadata
            .as_ref()
            .and_then(|meta| meta.get("description"))
            .filter(|desc| !desc.is_empty())
            .cloned()
            .unwrap_or_else(|| self.info.name.clone())
    }

    fn version(&self) -> Option<String> {
        self.metadata
            .as_ref()?
            .get("version")
            .filter(|version| !version.is_empty())
            .cloned()
    }

    fn frontmatter_value(&self, key: &str) -> Option<&str> {
        self.metadata.as_ref()?.get(key).map(|value| value.as_str())
    }
}

/// Result of validating a single skill.
#[derive(Debug, Clone)]
pub struct SkillValidationResult {
    pub name: String,
    pub path: String,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl SkillValidationResult {
    /// Returns true when there are no errors.
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Loads and manages agent skills from workspace and built-in directories.
pub struct SkillsLoader {
    workspace: PathBuf,
    workspace_skills: PathBuf,
    builtin_skills: PathBuf,
}

impl SkillsLoader {
    /// Create a new `SkillsLoader`.
    ///
    /// * `workspace`         - the agent workspace root.
    /// * `builtin_skills_dir` - optional override for the built-in skills directory.
    pub fn new(workspace: &Path, builtin_skills_dir: Option<&Path>) -> Self {
        let builtin = match builtin_skills_dir {
            Some(p) => p.to_path_buf(),
            None => workspace.join("builtin_skills"),
        };
        Self {
            workspace: workspace.to_path_buf(),
            workspace_skills: workspace.join("skills"),
            builtin_skills: builtin,
        }
    }

    /// List all available skills.
    ///
    /// When `filter_unavailable` is `true`, skills with unmet requirements are
    /// excluded from the result.
    pub fn list_skills(&self, filter_unavailable: bool) -> Vec<SkillInfo> {
        let skills = self.discover_skill_infos();
        if !filter_unavailable {
            return skills;
        }

        self.skill_records_from_infos(skills)
            .into_iter()
            .filter(|record| _check_requirements(&record.skill_meta))
            .map(|record| record.info)
            .collect()
    }

    fn discover_skill_infos(&self) -> Vec<SkillInfo> {
        let mut skills = Vec::new();
        let mut seen_names = HashSet::new();

        Self::collect_skill_infos(
            &self.workspace_skills,
            "workspace",
            &mut seen_names,
            &mut skills,
        );
        Self::collect_skill_infos(
            &self.builtin_skills,
            "builtin",
            &mut seen_names,
            &mut skills,
        );

        skills
    }

    fn collect_skill_infos(
        skills_dir: &Path,
        source: &str,
        seen_names: &mut HashSet<String>,
        skills: &mut Vec<SkillInfo>,
    ) {
        if !skills_dir.exists() {
            return;
        }

        let Ok(entries) = fs::read_dir(skills_dir) else {
            return;
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let skill_file = path.join("SKILL.md");
            if !skill_file.exists() {
                continue;
            }

            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            if seen_names.insert(name.clone()) {
                skills.push(SkillInfo {
                    name,
                    path: skill_file.to_string_lossy().to_string(),
                    source: source.to_string(),
                });
            }
        }
    }

    fn discover_skill_records(&self) -> Vec<SkillRecord> {
        self.skill_records_from_infos(self.discover_skill_infos())
    }

    fn skill_records_from_infos(&self, skills: Vec<SkillInfo>) -> Vec<SkillRecord> {
        skills
            .into_iter()
            .map(|info| {
                let metadata = _read_skill_metadata_from_path(Path::new(&info.path));
                let skill_meta = metadata
                    .as_ref()
                    .map(|meta| {
                        _parse_skill_metadata(
                            meta.get("metadata").map(|s| s.as_str()).unwrap_or(""),
                        )
                    })
                    .unwrap_or_default();

                SkillRecord {
                    info,
                    metadata,
                    skill_meta,
                }
            })
            .collect()
    }

    /// Load a skill's content by name.
    pub fn load_skill(&self, name: &str) -> Option<String> {
        // Workspace first.
        let workspace_skill = self.workspace_skills.join(name).join("SKILL.md");
        if workspace_skill.exists() {
            return fs::read_to_string(&workspace_skill).ok();
        }

        // Built-in.
        let builtin_skill = self.builtin_skills.join(name).join("SKILL.md");
        if builtin_skill.exists() {
            return fs::read_to_string(&builtin_skill).ok();
        }

        None
    }

    /// Load specific skills for inclusion in agent context.
    pub fn load_skills_for_context(&self, skill_names: &[String]) -> String {
        let mut parts: Vec<String> = Vec::new();

        for name in skill_names {
            if let Some(content) = self.load_skill(name) {
                let stripped = _strip_frontmatter(&content);
                parts.push(format!("### Skill: {}\n\n{}", name, stripped));
            }
        }

        if parts.is_empty() {
            String::new()
        } else {
            parts.join("\n\n---\n\n")
        }
    }

    /// Build an XML-formatted summary of all skills.
    pub fn build_skills_summary(&self) -> String {
        let skill_records = self.discover_skill_records();
        if skill_records.is_empty() {
            return String::new();
        }

        let mut lines: Vec<String> = vec!["<skills>".to_string()];

        for record in &skill_records {
            let name = _escape_xml(&record.info.name);
            let path = &record.info.path;
            let desc = _escape_xml(&record.description());
            let available = _check_requirements(&record.skill_meta);
            let version_attr = record
                .version()
                .map(|v| format!(" version=\"{}\"", _escape_xml(&v)))
                .unwrap_or_default();

            lines.push(format!(
                "  <skill available=\"{}\"{}>",
                if available { "true" } else { "false" },
                version_attr
            ));
            lines.push(format!("    <name>{}</name>", name));
            lines.push(format!("    <description>{}</description>", desc));
            lines.push(format!("    <location>{}</location>", path));

            if !available {
                let missing = _get_missing_requirements(&record.skill_meta);
                if !missing.is_empty() {
                    lines.push(format!(
                        "    <requires>{}</requires>",
                        _escape_xml(&missing)
                    ));
                }
            }

            lines.push("  </skill>".to_string());
        }

        lines.push("</skills>".to_string());
        lines.join("\n")
    }

    /// Build a compact one-line-per-skill index for system prompt injection.
    ///
    /// Format: "- skill_name (vX.Y): first 60 chars of description"
    /// Token cost: ~20 tokens per skill vs ~150 for XML summary.
    pub fn build_compact_index(&self) -> String {
        let skill_records = self.discover_skill_records();
        if skill_records.is_empty() {
            return String::new();
        }
        let mut lines = vec![
            "Available skills. Before reading any SKILL.md file by hand, \
             call `get_skills __list__` first to get canonical file paths — \
             editing by a guessed path will hit the wrong file. Then use \
             `get_skills <name>` for full content."
                .to_string(),
        ];
        for record in &skill_records {
            lines.push(Self::compact_line(record));
        }
        lines.join("\n")
    }

    /// Render a single skill as a compact one-line entry:
    /// `"- name (vX.Y): first 60 chars of description"`.
    ///
    /// Shared by `build_compact_index` (all skills) and `compact_lines` (a
    /// query-relevant subset) so the two cannot drift.
    fn compact_line(record: &SkillRecord) -> String {
        let desc = record.description();
        let display = if desc.len() > 60 {
            // Truncate at a char boundary to avoid breaking multibyte chars.
            let end = crate::utils::helpers::floor_char_boundary(&desc, 60);
            desc[..end].to_string()
        } else {
            desc
        };
        let version_suffix = record
            .version()
            .map(|v| format!(" (v{})", v))
            .unwrap_or_default();
        format!("- {}{}: {}", record.info.name, version_suffix, display)
    }

    /// Render compact one-line entries for a specific set of skill names.
    ///
    /// Names that don't resolve to a discovered skill are silently skipped.
    /// Output order follows `names`. Used to render the query-relevant skills
    /// selected by [`relevant`](Self::relevant) into the per-turn tail block.
    pub fn compact_lines(&self, names: &[String]) -> String {
        if names.is_empty() {
            return String::new();
        }
        let records = self.discover_skill_records();
        let mut lines: Vec<String> = Vec::new();
        for name in names {
            if let Some(record) = records.iter().find(|r| &r.info.name == name) {
                lines.push(Self::compact_line(record));
            }
        }
        lines.join("\n")
    }

    /// Return up to `k` skill names most relevant to `query`.
    ///
    /// Always-on skills (`get_always_skills`) are **excluded** — they are
    /// surfaced separately (the static prompt prefix), so the per-turn tail
    /// should not re-list them. Only requirement-met, non-always skills are
    /// ranked:
    /// - with the `semantic` feature: by cosine similarity between the embedded
    ///   query and each skill's embedded `"name: description"`, top-`k`.
    /// - without it (or on any embed error): the first `k` in discovery order.
    pub fn relevant(&self, query: &str, k: usize) -> Vec<String> {
        let always_set: HashSet<String> = self.get_always_skills().into_iter().collect();

        // Candidates: requirement-met skills that are not always-on.
        let candidates: Vec<SkillRecord> = self
            .discover_skill_records()
            .into_iter()
            .filter(|r| _check_requirements(&r.skill_meta))
            .filter(|r| !always_set.contains(&r.info.name))
            .collect();

        Self::rank_by_relevance(query, &candidates, k)
    }

    /// Rank candidate skills against `query`, returning up to `k` names.
    ///
    /// Falls back to discovery-order first-`k` when the `semantic` feature is
    /// off, the query is blank, or embedding fails — so the default build always
    /// returns something usable.
    fn rank_by_relevance(query: &str, candidates: &[SkillRecord], k: usize) -> Vec<String> {
        #[cfg(feature = "semantic")]
        {
            if !query.trim().is_empty() {
                if let Some(ranked) = Self::semantic_rank(query, candidates, k) {
                    return ranked;
                }
            }
        }
        #[cfg(not(feature = "semantic"))]
        let _ = query;

        candidates
            .iter()
            .take(k)
            .map(|r| r.info.name.clone())
            .collect()
    }

    /// Cosine-rank candidates by embedding similarity. Returns `None` (caller
    /// falls back) if any embedding call fails.
    #[cfg(feature = "semantic")]
    fn semantic_rank(query: &str, candidates: &[SkillRecord], k: usize) -> Option<Vec<String>> {
        use crate::agent::embedder;

        if candidates.is_empty() {
            return Some(Vec::new());
        }
        let query_vec = embedder::embed_one(query).ok()?;
        let texts: Vec<String> = candidates
            .iter()
            .map(|r| format!("{}: {}", r.info.name, r.description()))
            .collect();
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = embedder::embed_batch(&refs).ok()?;

        let mut scored: Vec<(f32, String)> = candidates
            .iter()
            .zip(embeddings.iter())
            .map(|(record, emb)| {
                (
                    embedder::cosine_similarity(&query_vec, emb),
                    record.info.name.clone(),
                )
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        Some(scored.into_iter().take(k).map(|(_, name)| name).collect())
    }

    /// Build a minimal skill name index for small local prompts.
    ///
    /// This is intentionally terse: names only, no descriptions. The model can
    /// call `get_skills __list__` for discovery details and `get_skills <name>`
    /// for full content.
    pub fn build_name_index(&self, max_names: usize) -> String {
        let mut names: Vec<String> = self
            .list_skills(false)
            .into_iter()
            .map(|s| s.name)
            .collect();
        if names.is_empty() {
            return String::new();
        }
        names.sort();

        let extra = names.len().saturating_sub(max_names);
        let visible: Vec<String> = names.into_iter().take(max_names).collect();
        let mut text = format!(
            "Skills available on demand via `get_skills`: {}.",
            visible.join(", ")
        );
        if extra > 0 {
            text.push_str(&format!(
                " Plus {} more. Use `get_skills __list__` for the full list.",
                extra
            ));
        } else {
            text.push_str(" Use `get_skills __list__` for descriptions.");
        }
        text
    }

    /// Get skills marked as `always=true` that also meet requirements.
    pub fn get_always_skills(&self) -> Vec<String> {
        let mut result: Vec<String> = Vec::new();

        for record in self.discover_skill_records() {
            if !_check_requirements(&record.skill_meta) {
                continue;
            }
            if record
                .skill_meta
                .get("always")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                result.push(record.info.name.clone());
                continue;
            }
            // Also check top-level "always" in frontmatter.
            if record
                .frontmatter_value("always")
                .map(|v| v == "true")
                .unwrap_or(false)
            {
                result.push(record.info.name.clone());
            }
        }

        result
    }

    /// Parse YAML-like frontmatter metadata from a skill file.
    pub fn get_skill_metadata(&self, name: &str) -> Option<HashMap<String, String>> {
        let content = self.load_skill(name)?;
        _parse_frontmatter_metadata(&content)
    }

    /// Validate a skill by name and return a `SkillValidationResult`.
    pub fn validate_skill(&self, skill: &SkillInfo) -> SkillValidationResult {
        let mut errors = vec![];
        let mut warnings = vec![];

        if skill.name.is_empty() {
            errors.push("Missing name".to_string());
        }

        let desc = self._get_skill_description(&skill.name);
        // _get_skill_description falls back to the name itself when no description found.
        if desc == skill.name || desc.is_empty() {
            errors.push("Missing description".to_string());
        }

        // Check requirements availability.
        let skill_meta = self._get_skill_meta(&skill.name);
        if !_check_requirements(&skill_meta) {
            let missing = _get_missing_requirements(&skill_meta);
            warnings.push(format!("Unmet requirement(s): {}", missing));
        }

        // Lint oversized SKILL.md files — large skills drift and are fragile
        // to patch one edit_file call at a time.
        if let Ok(content) = fs::read_to_string(&skill.path) {
            let line_count = content.lines().count();
            if line_count > 200 {
                warnings.push(format!(
                    "SKILL.md is oversized ({} lines > 200); consider splitting into focused skills",
                    line_count
                ));
            }
        }

        SkillValidationResult {
            name: skill.name.clone(),
            path: skill.path.clone(),
            errors,
            warnings,
        }
    }

    /// Validate all discoverable skills and return one result per skill.
    pub fn validate_all(&self) -> Vec<SkillValidationResult> {
        self.list_skills(false)
            .iter()
            .map(|s| self.validate_skill(s))
            .collect()
    }

    /// Get cleanup commands from all skills that declare one in frontmatter.
    pub fn get_cleanup_commands(&self) -> Vec<(String, String)> {
        let mut commands = Vec::new();
        for record in self.discover_skill_records() {
            if let Some(cmd) = record.frontmatter_value("cleanup") {
                if !cmd.is_empty() {
                    commands.push((record.info.name.clone(), cmd.to_string()));
                }
            }
        }
        commands
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn _get_skill_description(&self, name: &str) -> String {
        if let Some(meta) = self.get_skill_metadata(name) {
            if let Some(desc) = meta.get("description") {
                if !desc.is_empty() {
                    return desc.clone();
                }
            }
        }
        name.to_string()
    }

    fn _get_skill_version(&self, name: &str) -> Option<String> {
        let meta = self.get_skill_metadata(name)?;
        let v = meta.get("version")?;
        if v.is_empty() {
            None
        } else {
            Some(v.clone())
        }
    }

    fn _get_skill_meta(&self, name: &str) -> HashMap<String, serde_json::Value> {
        let meta = match self.get_skill_metadata(name) {
            Some(m) => m,
            None => return HashMap::new(),
        };
        let raw = meta.get("metadata").map(|s| s.as_str()).unwrap_or("");
        _parse_skill_metadata(raw)
    }
}

// ---------------------------------------------------------------------------
// Module-level helpers
// ---------------------------------------------------------------------------

/// Strip YAML frontmatter from markdown content.
fn _strip_frontmatter(content: &str) -> String {
    if content.starts_with("---") {
        if let Some(m) = RE_FRONTMATTER_STRIP.find(content) {
            return content[m.end()..].trim().to_string();
        }
    }
    content.to_string()
}

fn _read_skill_metadata_from_path(path: &Path) -> Option<HashMap<String, String>> {
    let content = fs::read_to_string(path).ok()?;
    _parse_frontmatter_metadata(&content)
}

fn _parse_frontmatter_metadata(content: &str) -> Option<HashMap<String, String>> {
    if !content.starts_with("---") {
        return None;
    }

    let caps = RE_FRONTMATTER.captures(content)?;
    let frontmatter = caps.get(1)?.as_str();

    let mut metadata = HashMap::new();
    for line in frontmatter.lines() {
        if let Some((key, value)) = line.split_once(':') {
            let k = key.trim().to_string();
            let v = value
                .trim()
                .trim_matches(|c| c == '"' || c == '\'')
                .to_string();
            metadata.insert(k, v);
        }
    }

    Some(metadata)
}

/// Parse skill metadata JSON from frontmatter value.
/// Looks for a "nanobot" key for backward compatibility with upstream skill files.
fn _parse_skill_metadata(raw: &str) -> HashMap<String, serde_json::Value> {
    if raw.is_empty() {
        return HashMap::new();
    }
    match serde_json::from_str::<serde_json::Value>(raw) {
        Ok(serde_json::Value::Object(map)) => {
            if let Some(serde_json::Value::Object(nanobot)) = map.get("nanobot") {
                nanobot
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            } else {
                HashMap::new()
            }
        }
        _ => HashMap::new(),
    }
}

/// Check if skill requirements are met (bins in PATH, env vars set).
fn _check_requirements(skill_meta: &HashMap<String, serde_json::Value>) -> bool {
    if let Some(requires) = skill_meta.get("requires") {
        // Check required binaries.
        if let Some(bins) = requires.get("bins").and_then(|v| v.as_array()) {
            for bin_val in bins {
                if let Some(bin_name) = bin_val.as_str() {
                    if !_command_exists(bin_name) {
                        debug!("Skill requirement not met: binary '{}' not found", bin_name);
                        return false;
                    }
                }
            }
        }
        // Check required environment variables.
        if let Some(env_vars) = requires.get("env").and_then(|v| v.as_array()) {
            for env_val in env_vars {
                if let Some(env_name) = env_val.as_str() {
                    if std::env::var(env_name).is_err() {
                        debug!("Skill requirement not met: env var '{}' not set", env_name);
                        return false;
                    }
                }
            }
        }
    }
    true
}

/// Get a description of what requirements are missing.
fn _get_missing_requirements(skill_meta: &HashMap<String, serde_json::Value>) -> String {
    let mut missing: Vec<String> = Vec::new();

    if let Some(requires) = skill_meta.get("requires") {
        if let Some(bins) = requires.get("bins").and_then(|v| v.as_array()) {
            for bin_val in bins {
                if let Some(bin_name) = bin_val.as_str() {
                    if !_command_exists(bin_name) {
                        missing.push(format!("CLI: {}", bin_name));
                    }
                }
            }
        }
        if let Some(env_vars) = requires.get("env").and_then(|v| v.as_array()) {
            for env_val in env_vars {
                if let Some(env_name) = env_val.as_str() {
                    if std::env::var(env_name).is_err() {
                        missing.push(format!("ENV: {}", env_name));
                    }
                }
            }
        }
    }

    missing.join(", ")
}

/// Check whether a binary exists on the PATH.
fn _command_exists(name: &str) -> bool {
    std::process::Command::new("which")
        .arg(name)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Escape XML special characters.
fn _escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Helper: create a workspace temp dir with a skills/ subdirectory
    /// containing one skill named `test-skill` with a SKILL.md file.
    fn make_workspace_with_skill(frontmatter: Option<&str>, body: &str) -> (TempDir, SkillsLoader) {
        let tmp = TempDir::new().unwrap();
        let skill_dir = tmp.path().join("skills").join("test-skill");
        fs::create_dir_all(&skill_dir).unwrap();

        let content = match frontmatter {
            Some(fm) => format!("---\n{}\n---\n{}", fm, body),
            None => body.to_string(),
        };
        fs::write(skill_dir.join("SKILL.md"), &content).unwrap();

        // Point builtin_skills to a non-existent dir so it is ignored.
        let loader = SkillsLoader::new(tmp.path(), Some(&tmp.path().join("no_builtin")));
        (tmp, loader)
    }

    // ----- _strip_frontmatter -----

    #[test]
    fn test_strip_frontmatter_with_frontmatter() {
        let content = "---\ntitle: Test\n---\nBody content";
        let result = _strip_frontmatter(content);
        assert_eq!(result, "Body content");
    }

    #[test]
    fn test_strip_frontmatter_without_frontmatter() {
        let content = "Just plain markdown content";
        let result = _strip_frontmatter(content);
        assert_eq!(result, "Just plain markdown content");
    }

    #[test]
    fn test_strip_frontmatter_multiline_body() {
        let content = "---\nkey: val\n---\nLine 1\nLine 2\nLine 3";
        let result = _strip_frontmatter(content);
        assert_eq!(result, "Line 1\nLine 2\nLine 3");
    }

    // ----- _escape_xml -----

    #[test]
    fn test_escape_xml_ampersand() {
        assert_eq!(_escape_xml("a & b"), "a &amp; b");
    }

    #[test]
    fn test_escape_xml_angle_brackets() {
        assert_eq!(_escape_xml("<tag>"), "&lt;tag&gt;");
    }

    #[test]
    fn test_escape_xml_combined() {
        assert_eq!(_escape_xml("x < y & y > z"), "x &lt; y &amp; y &gt; z");
    }

    #[test]
    fn test_escape_xml_no_special_chars() {
        assert_eq!(_escape_xml("hello world"), "hello world");
    }

    // ----- _parse_skill_metadata -----

    #[test]
    fn test_parse_skill_metadata_empty_string() {
        let result = _parse_skill_metadata("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_skill_metadata_valid_json() {
        let raw = r#"{"nanobot": {"always": true, "priority": 1}}"#;
        let result = _parse_skill_metadata(raw);
        assert_eq!(result.get("always").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(result.get("priority").and_then(|v| v.as_i64()), Some(1));
    }

    #[test]
    fn test_parse_skill_metadata_no_nanobot_key() {
        let raw = r#"{"other": {"key": "val"}}"#;
        let result = _parse_skill_metadata(raw);
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_skill_metadata_invalid_json() {
        let raw = "not json at all";
        let result = _parse_skill_metadata(raw);
        assert!(result.is_empty());
    }

    // ----- _check_requirements -----

    #[test]
    fn test_check_requirements_no_requires_key() {
        let meta: HashMap<String, serde_json::Value> = HashMap::new();
        assert!(_check_requirements(&meta));
    }

    #[test]
    fn test_check_requirements_with_existing_bin() {
        // "ls" should always exist on Linux/macOS.
        let mut meta: HashMap<String, serde_json::Value> = HashMap::new();
        meta.insert("requires".to_string(), serde_json::json!({"bins": ["ls"]}));
        assert!(_check_requirements(&meta));
    }

    #[test]
    fn test_check_requirements_with_missing_bin() {
        let mut meta: HashMap<String, serde_json::Value> = HashMap::new();
        meta.insert(
            "requires".to_string(),
            serde_json::json!({"bins": ["this_binary_does_not_exist_xyz_123"]}),
        );
        assert!(!_check_requirements(&meta));
    }

    #[test]
    fn test_check_requirements_with_missing_env() {
        let mut meta: HashMap<String, serde_json::Value> = HashMap::new();
        meta.insert(
            "requires".to_string(),
            serde_json::json!({"env": ["NANOBOT_TEST_NONEXISTENT_VAR_XYZ"]}),
        );
        assert!(!_check_requirements(&meta));
    }

    // ----- list_skills -----

    #[test]
    fn test_list_skills_finds_workspace_skill() {
        let (_tmp, loader) = make_workspace_with_skill(None, "# Test Skill\nDoes stuff.");
        let skills = loader.list_skills(false);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "test-skill");
        assert_eq!(skills[0].source, "workspace");
    }

    #[test]
    fn test_list_skills_empty_workspace() {
        let tmp = TempDir::new().unwrap();
        let loader = SkillsLoader::new(tmp.path(), Some(&tmp.path().join("no_builtin")));
        let skills = loader.list_skills(false);
        assert!(skills.is_empty());
    }

    // ----- load_skill -----

    #[test]
    fn test_load_skill_returns_content() {
        let (_tmp, loader) = make_workspace_with_skill(None, "# My Skill\nInstructions here.");
        let content = loader.load_skill("test-skill");
        assert!(content.is_some());
        assert!(content.unwrap().contains("Instructions here."));
    }

    #[test]
    fn test_load_skill_nonexistent_returns_none() {
        let (_tmp, loader) = make_workspace_with_skill(None, "body");
        let content = loader.load_skill("no-such-skill");
        assert!(content.is_none());
    }

    // ----- get_skill_metadata -----

    #[test]
    fn test_get_skill_metadata_with_frontmatter() {
        let frontmatter = "description: A cool skill\nauthor: tester";
        let (_tmp, loader) = make_workspace_with_skill(Some(frontmatter), "body");
        let meta = loader.get_skill_metadata("test-skill");
        assert!(meta.is_some());
        let meta = meta.unwrap();
        assert_eq!(
            meta.get("description").map(|s| s.as_str()),
            Some("A cool skill")
        );
        assert_eq!(meta.get("author").map(|s| s.as_str()), Some("tester"));
    }

    #[test]
    fn test_get_skill_metadata_without_frontmatter() {
        let (_tmp, loader) = make_workspace_with_skill(None, "plain body");
        let meta = loader.get_skill_metadata("test-skill");
        assert!(meta.is_none());
    }

    #[test]
    fn test_get_skill_metadata_strips_quotes() {
        let frontmatter = "description: \"Quoted value\"";
        let (_tmp, loader) = make_workspace_with_skill(Some(frontmatter), "body");
        let meta = loader.get_skill_metadata("test-skill").unwrap();
        assert_eq!(
            meta.get("description").map(|s| s.as_str()),
            Some("Quoted value")
        );
    }

    // ----- build_skills_summary -----

    #[test]
    fn test_build_skills_summary_xml_format() {
        let frontmatter = "description: Test description";
        let (_tmp, loader) = make_workspace_with_skill(Some(frontmatter), "body");
        let summary = loader.build_skills_summary();
        assert!(summary.starts_with("<skills>"));
        assert!(summary.ends_with("</skills>"));
        assert!(summary.contains("<name>test-skill</name>"));
        assert!(summary.contains("<description>Test description</description>"));
    }

    #[test]
    fn test_build_skills_summary_empty() {
        let tmp = TempDir::new().unwrap();
        let loader = SkillsLoader::new(tmp.path(), Some(&tmp.path().join("no_builtin")));
        let summary = loader.build_skills_summary();
        assert_eq!(summary, "");
    }

    #[test]
    fn test_build_skills_summary_escapes_xml() {
        // Create a skill whose name contains XML-special characters.
        let tmp = TempDir::new().unwrap();
        let skill_dir = tmp.path().join("skills").join("a&b");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\ndescription: x < y\n---\nbody",
        )
        .unwrap();
        let loader = SkillsLoader::new(tmp.path(), Some(&tmp.path().join("no_builtin")));
        let summary = loader.build_skills_summary();
        assert!(summary.contains("a&amp;b"));
        assert!(summary.contains("x &lt; y"));
    }

    // ----- load_skills_for_context -----

    #[test]
    fn test_load_skills_for_context_strips_frontmatter() {
        let frontmatter = "description: Test";
        let (_tmp, loader) = make_workspace_with_skill(Some(frontmatter), "The body content.\n");
        let names = vec!["test-skill".to_string()];
        let result = loader.load_skills_for_context(&names);
        assert!(result.contains("### Skill: test-skill"));
        assert!(result.contains("The body content."));
        // Frontmatter should be stripped.
        assert!(!result.contains("description: Test"));
    }

    #[test]
    fn test_load_skills_for_context_nonexistent_skill() {
        let (_tmp, loader) = make_workspace_with_skill(None, "body");
        let names = vec!["no-such-skill".to_string()];
        let result = loader.load_skills_for_context(&names);
        assert_eq!(result, "");
    }

    // ----- get_always_skills -----

    #[test]
    fn test_get_always_skills_with_always_flag() {
        let frontmatter = "always: true\ndescription: Always on";
        let (_tmp, loader) = make_workspace_with_skill(Some(frontmatter), "body");
        let always = loader.get_always_skills();
        assert!(always.contains(&"test-skill".to_string()));
    }

    #[test]
    fn test_get_always_skills_without_flag() {
        let frontmatter = "description: Normal skill";
        let (_tmp, loader) = make_workspace_with_skill(Some(frontmatter), "body");
        let always = loader.get_always_skills();
        assert!(always.is_empty());
    }

    // ----- builtin skill priority -----

    #[test]
    fn test_workspace_skill_shadows_builtin() {
        let tmp = TempDir::new().unwrap();

        // Create workspace skill.
        let ws_skill = tmp.path().join("skills").join("overlap");
        fs::create_dir_all(&ws_skill).unwrap();
        fs::write(ws_skill.join("SKILL.md"), "workspace version").unwrap();

        // Create builtin skill with same name.
        let bi_dir = tmp.path().join("builtin");
        let bi_skill = bi_dir.join("overlap");
        fs::create_dir_all(&bi_skill).unwrap();
        fs::write(bi_skill.join("SKILL.md"), "builtin version").unwrap();

        let loader = SkillsLoader::new(tmp.path(), Some(&bi_dir));
        let skills = loader.list_skills(false);

        // Should only find one, from workspace.
        let overlap_skills: Vec<&SkillInfo> =
            skills.iter().filter(|s| s.name == "overlap").collect();
        assert_eq!(overlap_skills.len(), 1);
        assert_eq!(overlap_skills[0].source, "workspace");

        // load_skill should return workspace version.
        let content = loader.load_skill("overlap").unwrap();
        assert_eq!(content, "workspace version");
    }

    #[test]
    fn test_index_uses_workspace_metadata_when_shadowing_builtin() {
        let tmp = TempDir::new().unwrap();

        let ws_skill = tmp.path().join("skills").join("overlap");
        fs::create_dir_all(&ws_skill).unwrap();
        fs::write(
            ws_skill.join("SKILL.md"),
            "---\ndescription: Workspace description\n---\nworkspace version",
        )
        .unwrap();

        let bi_dir = tmp.path().join("builtin");
        let bi_skill = bi_dir.join("overlap");
        fs::create_dir_all(&bi_skill).unwrap();
        fs::write(
            bi_skill.join("SKILL.md"),
            "---\ndescription: Builtin description\n---\nbuiltin version",
        )
        .unwrap();

        let loader = SkillsLoader::new(tmp.path(), Some(&bi_dir));
        let index = loader.build_compact_index();
        let summary = loader.build_skills_summary();

        assert!(index.contains("Workspace description"));
        assert!(!index.contains("Builtin description"));
        assert!(summary.contains("Workspace description"));
        assert!(!summary.contains("Builtin description"));
    }

    // ----- build_compact_index -----

    #[test]
    fn test_compact_index_format() {
        let frontmatter = "description: A helpful skill for coding tasks";
        let (_tmp, loader) = make_workspace_with_skill(Some(frontmatter), "body");
        let index = loader.build_compact_index();
        // Must start with the header line.
        assert!(
            index.starts_with("Available skills"),
            "index should start with header: {}",
            index
        );
        // Must contain one-line entry for the skill.
        assert!(
            index.contains("- test-skill: A helpful skill for coding tasks"),
            "index should contain skill entry: {}",
            index
        );
    }

    #[test]
    fn test_compact_index_preamble_is_imperative() {
        // The header must tell the model to call `get_skills __list__`
        // BEFORE reading any SKILL.md file by hand. A parenthetical hint
        // is not enough — small local models routinely skip hints and
        // re-read skill files from guessed paths, which is how Qwen3.6
        // ended up editing the wrong file.
        let frontmatter = "description: A skill";
        let (_tmp, loader) = make_workspace_with_skill(Some(frontmatter), "body");
        let index = loader.build_compact_index();
        let lower = index.to_lowercase();
        assert!(
            lower.contains("get_skills __list__"),
            "preamble must name the get_skills __list__ tool call: {index}"
        );
        // The preamble must frame __list__ as a prerequisite, not an option.
        assert!(
            lower.contains("before") || lower.contains("first") || lower.contains("mandatory"),
            "preamble must be imperative (before/first/mandatory), got: {index}"
        );
        // And it must warn against editing by guessed path.
        assert!(
            lower.contains("guess") || lower.contains("wrong file") || lower.contains("canonical"),
            "preamble must warn about guessed paths / wrong file: {index}"
        );
    }

    #[test]
    fn test_compact_index_empty() {
        let tmp = TempDir::new().unwrap();
        let loader = SkillsLoader::new(tmp.path(), Some(&tmp.path().join("no_builtin")));
        let index = loader.build_compact_index();
        assert_eq!(index, "", "empty skills should return empty string");
    }

    #[test]
    fn test_compact_index_truncates_long_description() {
        let long_desc = "A".repeat(80);
        let frontmatter = format!("description: {}", long_desc);
        let (_tmp, loader) = make_workspace_with_skill(Some(&frontmatter), "body");
        let index = loader.build_compact_index();
        // The description should be truncated to 60 chars.
        let entry_line = index
            .lines()
            .find(|l| l.starts_with("- test-skill:"))
            .expect("should have a test-skill entry");
        // After "- test-skill: " (15 chars), description should be <= 60 chars.
        let desc_part = entry_line
            .strip_prefix("- test-skill: ")
            .expect("entry should start with '- test-skill: '");
        assert!(
            desc_part.len() <= 60,
            "description part '{}' should be <= 60 chars, got {}",
            desc_part,
            desc_part.len()
        );
    }

    // ----- validate_skill -----

    #[test]
    fn test_validate_skill_missing_description() {
        // A skill with no frontmatter has no description.
        let (_tmp, loader) = make_workspace_with_skill(None, "# Undescribed skill\nDoes stuff.");
        let skills = loader.list_skills(false);
        assert_eq!(skills.len(), 1);
        let result = loader.validate_skill(&skills[0]);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("Missing description")),
            "expected 'Missing description' error, got: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_validate_skill_ok() {
        let frontmatter = "description: A proper description for the skill";
        let (_tmp, loader) = make_workspace_with_skill(Some(frontmatter), "body");
        let skills = loader.list_skills(false);
        assert_eq!(skills.len(), 1);
        let result = loader.validate_skill(&skills[0]);
        assert!(
            result.errors.is_empty(),
            "expected no errors, got: {:?}",
            result.errors
        );
        assert!(result.is_valid());
    }

    #[test]
    fn test_validate_skill_warns_on_oversized_file() {
        // A SKILL.md over 200 lines should emit a warning (lint rule: large
        // skill files are fragile to edit and signal a skill doing too much).
        let body = "line\n".repeat(300);
        let frontmatter = "description: A proper description";
        let (_tmp, loader) = make_workspace_with_skill(Some(frontmatter), &body);
        let skills = loader.list_skills(false);
        assert_eq!(skills.len(), 1);
        let result = loader.validate_skill(&skills[0]);
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.to_lowercase().contains("lines")
                    || w.to_lowercase().contains("oversized")),
            "expected an oversized-file warning, got: {:?}",
            result.warnings
        );
    }

    #[test]
    fn test_validate_skill_no_size_warning_when_small() {
        // A small SKILL.md must not emit a size warning.
        let body = "line\n".repeat(50);
        let frontmatter = "description: A proper description";
        let (_tmp, loader) = make_workspace_with_skill(Some(frontmatter), &body);
        let skills = loader.list_skills(false);
        let result = loader.validate_skill(&skills[0]);
        assert!(
            !result
                .warnings
                .iter()
                .any(|w| w.to_lowercase().contains("lines")
                    || w.to_lowercase().contains("oversized")),
            "small skill should not trigger size warning, got: {:?}",
            result.warnings
        );
    }

    #[test]
    fn test_validate_all_returns_one_per_skill() {
        let tmp = TempDir::new().unwrap();
        for name in &["skill-a", "skill-b"] {
            let skill_dir = tmp.path().join("skills").join(name);
            fs::create_dir_all(&skill_dir).unwrap();
            fs::write(
                skill_dir.join("SKILL.md"),
                format!("---\ndescription: Desc for {}\n---\nbody", name),
            )
            .unwrap();
        }
        let loader = SkillsLoader::new(tmp.path(), Some(&tmp.path().join("no_builtin")));
        let results = loader.validate_all();
        assert_eq!(results.len(), 2);
    }

    // ----- version support (Feature 3.5) -----

    #[test]
    fn test_skill_version_in_compact_index() {
        let frontmatter = "description: A versioned skill\nversion: 1.0";
        let (_tmp, loader) = make_workspace_with_skill(Some(frontmatter), "body");
        let index = loader.build_compact_index();
        assert!(
            index.contains("(v1.0)"),
            "compact index should contain '(v1.0)', got: {}",
            index
        );
    }

    #[test]
    fn test_skill_version_in_xml_summary() {
        let frontmatter = "description: A versioned skill\nversion: 2.3";
        let (_tmp, loader) = make_workspace_with_skill(Some(frontmatter), "body");
        let summary = loader.build_skills_summary();
        assert!(
            summary.contains("version=\"2.3\""),
            "XML summary should contain version attribute, got: {}",
            summary
        );
    }

    #[test]
    fn test_skill_no_version_omits_marker() {
        let frontmatter = "description: A skill without version";
        let (_tmp, loader) = make_workspace_with_skill(Some(frontmatter), "body");
        let index = loader.build_compact_index();
        assert!(
            !index.contains("(v)"),
            "compact index must not contain bare '(v)', got: {}",
            index
        );
        let summary = loader.build_skills_summary();
        assert!(
            !summary.contains("version="),
            "XML summary must not contain version attribute when absent, got: {}",
            summary
        );
    }

    #[test]
    fn test_compact_index_multiple_skills() {
        let tmp = TempDir::new().unwrap();
        // Create two workspace skills.
        for name in &["skill-a", "skill-b"] {
            let skill_dir = tmp.path().join("skills").join(name);
            fs::create_dir_all(&skill_dir).unwrap();
            fs::write(
                skill_dir.join("SKILL.md"),
                format!("---\ndescription: Desc for {}\n---\nbody", name),
            )
            .unwrap();
        }
        let loader = SkillsLoader::new(tmp.path(), Some(&tmp.path().join("no_builtin")));
        let index = loader.build_compact_index();
        assert!(index.contains("skill-a"), "should contain skill-a");
        assert!(index.contains("skill-b"), "should contain skill-b");
        // Should be multi-line (header + 2 skills).
        assert!(index.lines().count() >= 3);
    }

    // ----- get_cleanup_commands -----

    #[test]
    fn test_get_cleanup_commands_returns_cleanup_field() {
        let frontmatter = "description: A skill with cleanup\ncleanup: /usr/bin/stop-thing";
        let (_tmp, loader) = make_workspace_with_skill(Some(frontmatter), "body");
        let cmds = loader.get_cleanup_commands();
        assert_eq!(cmds.len(), 1);
        assert_eq!(cmds[0].0, "test-skill");
        assert_eq!(cmds[0].1, "/usr/bin/stop-thing");
    }

    #[test]
    fn test_get_cleanup_commands_skips_skills_without_cleanup() {
        let frontmatter = "description: Normal skill";
        let (_tmp, loader) = make_workspace_with_skill(Some(frontmatter), "body");
        let cmds = loader.get_cleanup_commands();
        assert!(cmds.is_empty());
    }

    // ----- name index -----

    #[test]
    fn test_name_index_is_discovery_only() {
        let frontmatter = "description: A versioned skill\nversion: 1.0";
        let (_tmp, loader) = make_workspace_with_skill(Some(frontmatter), "body");
        let index = loader.build_name_index(12);
        assert!(index.contains("test-skill"));
        assert!(index.contains("get_skills __list__"));
        assert!(!index.contains("A versioned skill"));
    }

    // ----- relevant() / compact_lines() -----

    /// Create a workspace with several named skills, each `(name, frontmatter)`.
    fn make_workspace_with_skills(skills: &[(&str, &str)]) -> (TempDir, SkillsLoader) {
        let tmp = TempDir::new().unwrap();
        for (name, frontmatter) in skills {
            let skill_dir = tmp.path().join("skills").join(name);
            fs::create_dir_all(&skill_dir).unwrap();
            fs::write(
                skill_dir.join("SKILL.md"),
                format!("---\n{}\n---\nbody", frontmatter),
            )
            .unwrap();
        }
        let loader = SkillsLoader::new(tmp.path(), Some(&tmp.path().join("no_builtin")));
        (tmp, loader)
    }

    #[test]
    fn test_relevant_excludes_always_skills() {
        // always-on skills are surfaced via the static prefix, so relevant()
        // must NOT return them — the per-turn tail shows only ranked picks.
        let (_tmp, loader) = make_workspace_with_skills(&[
            ("always-one", "description: Always loaded\nalways: true"),
            ("normal-a", "description: Normal skill A"),
            ("normal-b", "description: Normal skill B"),
            ("normal-c", "description: Normal skill C"),
        ]);

        let result = loader.relevant("anything at all", 2);

        // The always-on skill must be absent.
        assert!(
            !result.contains(&"always-one".to_string()),
            "always-on skill must be excluded, got: {:?}",
            result
        );
        // Capped at k=2 ranked (non-always) skills.
        assert_eq!(
            result.len(),
            2,
            "should return k=2 ranked skills, got: {:?}",
            result
        );
        // Every returned name is a real, non-always discovered skill.
        let known: HashSet<String> = loader
            .list_skills(false)
            .into_iter()
            .map(|s| s.name)
            .collect();
        for name in &result {
            assert!(
                known.contains(name),
                "unknown skill name returned: {}",
                name
            );
            assert_ne!(name, "always-one");
        }
    }

    #[test]
    fn test_relevant_caps_ranked_at_k() {
        // With no always-on skills, relevant() returns at most k names.
        let (_tmp, loader) = make_workspace_with_skills(&[
            ("normal-a", "description: A"),
            ("normal-b", "description: B"),
            ("normal-c", "description: C"),
            ("normal-d", "description: D"),
        ]);
        let result = loader.relevant("query", 2);
        assert_eq!(result.len(), 2, "should cap at k=2, got: {:?}", result);
    }

    #[test]
    fn test_compact_lines_renders_named_subset() {
        let (_tmp, loader) = make_workspace_with_skills(&[
            ("skill-a", "description: Description for A"),
            ("skill-b", "description: Description for B"),
            ("skill-c", "description: Description for C"),
        ]);
        let names = vec!["skill-a".to_string(), "skill-c".to_string()];
        let rendered = loader.compact_lines(&names);
        assert!(rendered.contains("- skill-a: Description for A"));
        assert!(rendered.contains("- skill-c: Description for C"));
        // Unrequested skill must not appear.
        assert!(!rendered.contains("skill-b"));
        // One line per requested name.
        assert_eq!(rendered.lines().count(), 2);
    }

    #[test]
    fn test_compact_lines_skips_unknown_names() {
        let (_tmp, loader) = make_workspace_with_skills(&[("skill-a", "description: A")]);
        let names = vec!["skill-a".to_string(), "no-such-skill".to_string()];
        let rendered = loader.compact_lines(&names);
        assert_eq!(rendered.lines().count(), 1);
        assert!(rendered.contains("skill-a"));
    }

    #[cfg(feature = "semantic")]
    #[test]
    fn test_relevant_semantic_ranks_topical_skill_first() {
        // A SQL query should rank the database skill above unrelated skills.
        let (_tmp, loader) = make_workspace_with_skills(&[
            (
                "database-skill",
                "description: Query and manage SQL databases, tables, and indexes",
            ),
            (
                "cooking-skill",
                "description: Recipes for cooking pasta and Italian food",
            ),
            (
                "music-skill",
                "description: Compose and play piano and guitar music",
            ),
        ]);
        let result = loader.relevant("how do I write a SQL SELECT statement", 3);
        assert_eq!(
            result.first().map(|s| s.as_str()),
            Some("database-skill"),
            "semantic ranking should place the SQL skill first, got: {:?}",
            result
        );
    }
}
