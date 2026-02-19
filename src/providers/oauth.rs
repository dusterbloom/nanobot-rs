//! OAuth token manager for Claude Max subscriptions.
//!
//! Reads OAuth credentials from `~/.claude/.credentials.json` (written by
//! the Claude CLI during authentication) and provides valid access tokens
//! with automatic refresh when expired.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Known OAuth client ID used by the Claude CLI.
const CLAUDE_OAUTH_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";

/// Token refresh endpoint.
const OAUTH_TOKEN_URL: &str = "https://console.anthropic.com/api/oauth/token";

/// Refresh tokens this many milliseconds before actual expiry.
const EXPIRY_BUFFER_MS: i64 = 5 * 60 * 1000; // 5 minutes

// ---------------------------------------------------------------------------
// Credential file structures
// ---------------------------------------------------------------------------

/// Top-level structure of `~/.claude/.credentials.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialsFile {
    #[serde(rename = "claudeAiOauth")]
    pub claude_ai_oauth: OAuthCredentials,
}

/// The `claudeAiOauth` section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthCredentials {
    #[serde(rename = "accessToken")]
    pub access_token: String,
    #[serde(rename = "refreshToken")]
    pub refresh_token: String,
    #[serde(rename = "expiresAt")]
    pub expires_at: i64,
    #[serde(default)]
    pub scopes: Vec<String>,
    #[serde(rename = "subscriptionType", default)]
    pub subscription_type: String,
}

/// Response from the OAuth token refresh endpoint.
#[derive(Debug, Deserialize)]
struct RefreshResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: Option<i64>,
}

// ---------------------------------------------------------------------------
// OAuthTokenManager
// ---------------------------------------------------------------------------

/// Manages OAuth tokens for Claude Max, loading from and persisting to
/// the Claude CLI credentials file.
pub struct OAuthTokenManager {
    credentials_path: std::path::PathBuf,
    credentials: OAuthCredentials,
}

impl OAuthTokenManager {
    /// Load credentials from `~/.claude/.credentials.json`.
    pub fn load() -> Result<Self> {
        let home = dirs::home_dir().context("Cannot determine home directory")?;
        let path = home.join(".claude").join(".credentials.json");
        Self::load_from(path)
    }

    /// Load credentials from a specific path (useful for testing).
    fn load_from(path: std::path::PathBuf) -> Result<Self> {
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("Cannot read credentials at {}", path.display()))?;
        let file: CredentialsFile = serde_json::from_str(&content)
            .with_context(|| format!("Cannot parse credentials at {}", path.display()))?;

        info!(
            "Loaded Claude OAuth credentials (subscription={}, expires_at={})",
            file.claude_ai_oauth.subscription_type, file.claude_ai_oauth.expires_at
        );

        Ok(Self {
            credentials_path: path,
            credentials: file.claude_ai_oauth,
        })
    }

    /// Return a valid access token, refreshing if expired or about to expire.
    pub async fn access_token(&mut self) -> Result<String> {
        if self.is_expired() {
            self.refresh().await?;
        }
        Ok(self.credentials.access_token.clone())
    }

    /// Check whether the current token is expired or within the buffer window.
    fn is_expired(&self) -> bool {
        let now_ms = chrono::Utc::now().timestamp_millis();
        now_ms >= (self.credentials.expires_at - EXPIRY_BUFFER_MS)
    }

    /// Refresh the access token using the refresh token.
    async fn refresh(&mut self) -> Result<()> {
        info!("Refreshing Claude OAuth access token...");

        let client = reqwest::Client::new();
        let resp = client
            .post(OAUTH_TOKEN_URL)
            .form(&[
                ("grant_type", "refresh_token"),
                ("refresh_token", &self.credentials.refresh_token),
                ("client_id", CLAUDE_OAUTH_CLIENT_ID),
            ])
            .send()
            .await
            .context("OAuth refresh request failed")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "OAuth token refresh failed (HTTP {}): {}",
                status,
                body.chars().take(500).collect::<String>()
            );
        }

        let data: RefreshResponse = resp.json().await.context("Cannot parse refresh response")?;

        // Update credentials.
        self.credentials.access_token = data.access_token;
        if let Some(rt) = data.refresh_token {
            self.credentials.refresh_token = rt;
        }
        // expires_in is in seconds; convert to absolute ms timestamp.
        if let Some(expires_in) = data.expires_in {
            self.credentials.expires_at =
                chrono::Utc::now().timestamp_millis() + (expires_in * 1000);
        }

        debug!(
            "OAuth token refreshed (new expires_at={})",
            self.credentials.expires_at
        );

        // Persist refreshed credentials back to disk.
        self.save().ok(); // Best-effort; don't fail the request if disk write fails.

        Ok(())
    }

    /// Write the current credentials back to disk.
    fn save(&self) -> Result<()> {
        let file = CredentialsFile {
            claude_ai_oauth: self.credentials.clone(),
        };
        let content =
            serde_json::to_string_pretty(&file).context("Cannot serialize credentials")?;

        // Atomic write: write to temp file, then rename.
        let tmp = self.credentials_path.with_extension("tmp");
        std::fs::write(&tmp, &content)
            .with_context(|| format!("Cannot write to {}", tmp.display()))?;
        std::fs::rename(&tmp, &self.credentials_path).with_context(|| {
            format!(
                "Cannot rename {} → {}",
                tmp.display(),
                self.credentials_path.display()
            )
        })?;

        debug!(
            "Saved refreshed credentials to {}",
            self.credentials_path.display()
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_credentials(expires_at: i64) -> String {
        serde_json::json!({
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-test-token",
                "refreshToken": "sk-ant-ort01-test-refresh",
                "expiresAt": expires_at,
                "scopes": ["user:inference"],
                "subscriptionType": "max"
            }
        })
        .to_string()
    }

    #[test]
    fn test_load_credentials() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("credentials.json");
        std::fs::write(&path, make_credentials(9999999999999)).unwrap();

        let mgr = OAuthTokenManager::load_from(path).unwrap();
        assert_eq!(mgr.credentials.access_token, "sk-ant-oat01-test-token");
        assert_eq!(mgr.credentials.refresh_token, "sk-ant-ort01-test-refresh");
        assert_eq!(mgr.credentials.subscription_type, "max");
        assert!(!mgr.is_expired());
    }

    #[test]
    fn test_token_expired() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("credentials.json");
        // Set expiry to the past.
        std::fs::write(&path, make_credentials(1000)).unwrap();

        let mgr = OAuthTokenManager::load_from(path).unwrap();
        assert!(mgr.is_expired());
    }

    #[test]
    fn test_token_within_buffer() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("credentials.json");
        // Set expiry to 2 minutes from now (within 5-min buffer).
        let expires_at = chrono::Utc::now().timestamp_millis() + 2 * 60 * 1000;
        std::fs::write(&path, make_credentials(expires_at)).unwrap();

        let mgr = OAuthTokenManager::load_from(path).unwrap();
        assert!(mgr.is_expired());
    }

    #[test]
    fn test_token_not_expired() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("credentials.json");
        // Set expiry to 1 hour from now.
        let expires_at = chrono::Utc::now().timestamp_millis() + 60 * 60 * 1000;
        std::fs::write(&path, make_credentials(expires_at)).unwrap();

        let mgr = OAuthTokenManager::load_from(path).unwrap();
        assert!(!mgr.is_expired());
    }

    #[test]
    fn test_load_missing_file() {
        let result = OAuthTokenManager::load_from("/nonexistent/credentials.json".into());
        assert!(result.is_err());
    }

    #[test]
    fn test_load_invalid_json() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("credentials.json");
        std::fs::write(&path, "not json").unwrap();

        let result = OAuthTokenManager::load_from(path);
        assert!(result.is_err());
    }

    #[test]
    fn test_save_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("credentials.json");
        std::fs::write(&path, make_credentials(9999999999999)).unwrap();

        let mgr = OAuthTokenManager::load_from(path.clone()).unwrap();
        mgr.save().unwrap();

        // Reload and verify.
        let mgr2 = OAuthTokenManager::load_from(path).unwrap();
        assert_eq!(mgr2.credentials.access_token, "sk-ant-oat01-test-token");
    }
}
