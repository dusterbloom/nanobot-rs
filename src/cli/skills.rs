//! Skill management -- search (skills.sh), install from GitHub, and remove.

// Interactive/app boundary (error-protocol layer 3 backlog): printing IS the
// product here (REPL/TUI/CLI), and the thin glue code keeps pragmatic
// unwraps on always-set state (rl, runtime, static regexes). The deny regime
// in Cargo.toml stays live for the core; this module lands on the regime
// when its backlog is migrated.
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::shadow_reuse,
    clippy::shadow_unrelated,
    clippy::shadow_same,
    clippy::format_push_string,
    clippy::string_add
)]
fn http_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .user_agent("nanobot")
        .build()
        .map_err(|e| format!("HTTP client error: {e}"))
}

/// Install skills from a GitHub repository.
///
/// `source` can be:
/// - `owner/repo`         -- install all skills found in the repo
/// - `owner/repo@skill`   -- install a specific skill by name
///
/// Skill locations are discovered via the git trees API, so any repo layout
/// works (`skills/{name}/SKILL.md`, `skills/.curated/{name}/SKILL.md`, root
/// `SKILL.md`, ...). Skills are saved to `{workspace}/skills/{name}/SKILL.md`.
pub(crate) async fn cmd_skill_add(
    workspace: &std::path::Path,
    source: &str,
) -> Result<Vec<String>, String> {
    let (repo, specific_skill) = if let Some((repo, skill)) = source.split_once('@') {
        (repo, Some(skill))
    } else {
        (source, None)
    };

    // Validate repo format
    let parts: Vec<&str> = repo.split('/').collect();
    if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
        return Err(format!(
            "Invalid repo format: '{}'. Expected owner/repo",
            repo
        ));
    }
    let (owner, repo_name) = (parts[0], parts[1]);

    let client = http_client()?;
    let found = find_repo_skills(&client, owner, repo_name).await?;
    if found.is_empty() {
        return Err(format!("No SKILL.md files found in {owner}/{repo_name}"));
    }

    let selected: Vec<&(String, String)> = match specific_skill {
        Some(name) => {
            let hit = found
                .iter()
                .find(|(n, _)| n == name)
                .ok_or_else(|| format!("Skill '{name}' not found in {owner}/{repo_name}"))?;
            vec![hit]
        }
        None => found.iter().collect(),
    };

    let mut installed: Vec<String> = Vec::new();
    for (name, path) in selected {
        match fetch_raw_file(&client, owner, repo_name, path).await {
            Ok(content) => {
                save_skill(workspace, name, &content)?;
                installed.push(name.clone());
            }
            Err(e) => {
                eprintln!("  Skipping {name}: {e}");
            }
        }
    }

    Ok(installed)
}

/// A skill found on the skills.sh registry.
pub(crate) struct SkillHit {
    /// GitHub `owner/repo` the skill lives in.
    pub source: String,
    /// Skill name within the repo (install as `source@skill`).
    pub skill: String,
    /// Registry install count (popularity signal).
    pub installs: u64,
}

/// Search the skills.sh registry (the agentskills.io ecosystem index).
pub(crate) async fn cmd_skill_search(query: &str) -> Result<Vec<SkillHit>, String> {
    let client = http_client()?;
    let resp = client
        .get("https://skills.sh/api/search")
        .query(&[("q", query)])
        .send()
        .await
        .map_err(|e| format!("skills.sh request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("skills.sh returned {}", resp.status()));
    }
    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse skills.sh response: {e}"))?;
    Ok(parse_skill_search(&body))
}

fn parse_skill_search(body: &serde_json::Value) -> Vec<SkillHit> {
    let Some(skills) = body.get("skills").and_then(|s| s.as_array()) else {
        return Vec::new();
    };
    skills
        .iter()
        .filter_map(|s| {
            Some(SkillHit {
                source: s.get("source")?.as_str()?.to_string(),
                skill: s.get("skillId")?.as_str()?.to_string(),
                installs: s.get("installs").and_then(|i| i.as_u64()).unwrap_or(0),
            })
        })
        .collect()
}

/// Find every skill in a repo as `(name, path-to-SKILL.md)`, via the git
/// trees API (one request, any directory layout, any default branch).
async fn find_repo_skills(
    client: &reqwest::Client,
    owner: &str,
    repo: &str,
) -> Result<Vec<(String, String)>, String> {
    let url = format!("https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1");
    let resp = client
        .get(&url)
        .header("Accept", "application/vnd.github.v3+json")
        .send()
        .await
        .map_err(|e| format!("GitHub API request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!(
            "GitHub API returned {} for {owner}/{repo}",
            resp.status()
        ));
    }
    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse GitHub API response: {e}"))?;

    let mut skills = Vec::new();
    for entry in body
        .get("tree")
        .and_then(|t| t.as_array())
        .unwrap_or(&Vec::new())
    {
        let Some(path) = entry.get("path").and_then(|p| p.as_str()) else {
            continue;
        };
        if let Some(name) = skill_name_from_path(path, repo) {
            skills.push((name.to_string(), path.to_string()));
        }
    }
    Ok(skills)
}

/// The skill name is the directory containing SKILL.md; a root-level
/// SKILL.md takes the repo name. Non-SKILL.md paths yield `None`.
fn skill_name_from_path<'a>(path: &'a str, repo: &'a str) -> Option<&'a str> {
    match path.strip_suffix("/SKILL.md") {
        Some(dir) => Some(dir.rsplit('/').next().unwrap_or(dir)),
        None if path == "SKILL.md" => Some(repo),
        None => None,
    }
}

async fn fetch_raw_file(
    client: &reqwest::Client,
    owner: &str,
    repo: &str,
    path: &str,
) -> Result<String, String> {
    let url = format!("https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path}");
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Fetch failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {} for {url}", resp.status()));
    }
    resp.text()
        .await
        .map_err(|e| format!("Failed to read response: {e}"))
}

/// Save a SKILL.md to the workspace skills directory.
fn save_skill(workspace: &std::path::Path, skill_name: &str, content: &str) -> Result<(), String> {
    let skill_dir = workspace.join("skills").join(skill_name);
    std::fs::create_dir_all(&skill_dir)
        .map_err(|e| format!("Failed to create directory {}: {e}", skill_dir.display()))?;

    let skill_file = skill_dir.join("SKILL.md");
    std::fs::write(&skill_file, content)
        .map_err(|e| format!("Failed to write {}: {e}", skill_file.display()))?;

    Ok(())
}

/// Remove an installed skill by name.
pub(crate) fn cmd_skill_remove(workspace: &std::path::Path, name: &str) -> Result<(), String> {
    let skill_dir = workspace.join("skills").join(name);
    if !skill_dir.exists() {
        return Err(format!("Skill '{}' not found", name));
    }
    std::fs::remove_dir_all(&skill_dir)
        .map_err(|e| format!("Failed to remove skill '{}': {e}", name))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_skill_search_extracts_hits() {
        let body = json!({
            "query": "pdf",
            "skills": [
                {"id": "openai/skills/pdf", "skillId": "pdf", "name": "pdf",
                 "installs": 9254, "source": "openai/skills"},
                {"skillId": "broken-no-source", "installs": 1},
            ]
        });
        let hits = parse_skill_search(&body);
        assert_eq!(hits.len(), 1, "entries missing fields are skipped");
        assert_eq!(hits[0].source, "openai/skills");
        assert_eq!(hits[0].skill, "pdf");
        assert_eq!(hits[0].installs, 9254);
        assert!(parse_skill_search(&json!({})).is_empty());
    }

    #[test]
    fn skill_name_from_path_handles_all_layouts() {
        assert_eq!(
            skill_name_from_path("skills/pdf/SKILL.md", "repo"),
            Some("pdf")
        );
        assert_eq!(
            skill_name_from_path("skills/.curated/cli-creator/SKILL.md", "repo"),
            Some("cli-creator")
        );
        assert_eq!(
            skill_name_from_path("SKILL.md", "my-skill"),
            Some("my-skill")
        );
        assert_eq!(skill_name_from_path("README.md", "repo"), None);
    }
}
