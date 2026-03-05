//! Skill management -- install from GitHub and remove.

/// Install skills from a GitHub repository.
///
/// `source` can be:
/// - `owner/repo`         -- install all skills found in the repo's `skills/` dir
/// - `owner/repo@skill`   -- install a specific skill by name
///
/// Skills are saved to `{workspace}/skills/{name}/SKILL.md`.
pub(crate) async fn cmd_skill_add(workspace: &std::path::Path, source: &str) -> Result<Vec<String>, String> {
    let (repo, specific_skill) = if let Some((repo, skill)) = source.split_once('@') {
        (repo, Some(skill))
    } else {
        (source, None)
    };

    // Validate repo format
    let parts: Vec<&str> = repo.split('/').collect();
    if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
        return Err(format!("Invalid repo format: '{}'. Expected owner/repo", repo));
    }
    let (owner, repo_name) = (parts[0], parts[1]);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .user_agent("nanobot")
        .build()
        .map_err(|e| format!("HTTP client error: {e}"))?;

    let mut installed: Vec<String> = Vec::new();

    if let Some(skill_name) = specific_skill {
        // Install a specific skill
        let content = fetch_skill_md(&client, owner, repo_name, skill_name).await?;
        save_skill(workspace, skill_name, &content)?;
        installed.push(skill_name.to_string());
    } else {
        // List skills/ directory in the repo
        let skills = list_repo_skills(&client, owner, repo_name).await?;
        if skills.is_empty() {
            return Err(format!("No skills found in {owner}/{repo_name}/skills/"));
        }
        for skill_name in &skills {
            match fetch_skill_md(&client, owner, repo_name, skill_name).await {
                Ok(content) => {
                    save_skill(workspace, skill_name, &content)?;
                    installed.push(skill_name.clone());
                }
                Err(e) => {
                    eprintln!("  Skipping {skill_name}: {e}");
                }
            }
        }
    }

    Ok(installed)
}

/// Fetch SKILL.md content for a specific skill from a GitHub repo.
async fn fetch_skill_md(
    client: &reqwest::Client,
    owner: &str,
    repo: &str,
    skill_name: &str,
) -> Result<String, String> {
    // Try raw.githubusercontent.com (no API rate limits, no auth needed)
    // Try multiple common paths: skills/{name}/SKILL.md, then root SKILL.md
    let paths = [
        format!("https://raw.githubusercontent.com/{owner}/{repo}/HEAD/skills/{skill_name}/SKILL.md"),
        format!("https://raw.githubusercontent.com/{owner}/{repo}/main/skills/{skill_name}/SKILL.md"),
        format!("https://raw.githubusercontent.com/{owner}/{repo}/master/skills/{skill_name}/SKILL.md"),
    ];

    for url in &paths {
        match client.get(url).send().await {
            Ok(resp) if resp.status().is_success() => {
                return resp.text().await.map_err(|e| format!("Failed to read response: {e}"));
            }
            _ => continue,
        }
    }

    Err(format!("SKILL.md not found for '{skill_name}' in {owner}/{repo}"))
}

/// List skill directories in a GitHub repo's skills/ folder via the GitHub API.
async fn list_repo_skills(
    client: &reqwest::Client,
    owner: &str,
    repo: &str,
) -> Result<Vec<String>, String> {
    let url = format!("https://api.github.com/repos/{owner}/{repo}/contents/skills");
    let resp = client
        .get(&url)
        .header("Accept", "application/vnd.github.v3+json")
        .send()
        .await
        .map_err(|e| format!("GitHub API request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!(
            "GitHub API returned {} for {owner}/{repo}/skills/",
            resp.status()
        ));
    }

    let entries: Vec<serde_json::Value> = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse GitHub API response: {e}"))?;

    let mut skills = Vec::new();
    for entry in &entries {
        if entry.get("type").and_then(|v| v.as_str()) == Some("dir") {
            if let Some(name) = entry.get("name").and_then(|v| v.as_str()) {
                skills.push(name.to_string());
            }
        }
    }

    Ok(skills)
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
