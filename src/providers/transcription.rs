//! Voice transcription provider using Groq's Whisper API.

use std::path::Path;

use reqwest::Client;
use tracing::{error, warn};

/// Voice transcription provider using Groq's Whisper API.
///
/// Groq offers extremely fast transcription with a generous free tier.
pub struct GroqTranscriptionProvider {
    api_key: Option<String>,
    api_url: String,
    client: Client,
}

impl GroqTranscriptionProvider {
    /// Create a new transcription provider.
    ///
    /// If `api_key` is `None`, the `GROQ_API_KEY` environment variable is
    /// checked at construction time.
    pub fn new(api_key: Option<String>) -> Self {
        let resolved_key = api_key.or_else(|| std::env::var("GROQ_API_KEY").ok());

        Self {
            api_key: resolved_key,
            api_url: "https://api.groq.com/openai/v1/audio/transcriptions".to_string(),
            client: Client::new(),
        }
    }

    /// Transcribe an audio file using Groq.
    ///
    /// Returns the transcribed text, or an empty string on error.
    pub async fn transcribe(&self, file_path: &Path) -> String {
        let api_key = match &self.api_key {
            Some(k) => k.clone(),
            None => {
                warn!("Groq API key not configured for transcription");
                return String::new();
            }
        };

        if !file_path.exists() {
            error!("Audio file not found: {}", file_path.display());
            return String::new();
        }

        // Read the file bytes.
        let file_bytes = match tokio::fs::read(file_path).await {
            Ok(b) => b,
            Err(e) => {
                error!("Failed to read audio file {}: {}", file_path.display(), e);
                return String::new();
            }
        };

        let file_name = file_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        // Build multipart form.
        let file_part = reqwest::multipart::Part::bytes(file_bytes)
            .file_name(file_name)
            .mime_str("application/octet-stream")
            .unwrap_or_else(|_| reqwest::multipart::Part::bytes(Vec::new()));

        let form = reqwest::multipart::Form::new()
            .part("file", file_part)
            .text("model", "whisper-large-v3");

        match self
            .client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", api_key))
            .multipart(form)
            .timeout(std::time::Duration::from_secs(60))
            .send()
            .await
        {
            Ok(response) => {
                if !response.status().is_success() {
                    let status = response.status();
                    let body = response.text().await.unwrap_or_default();
                    error!("Groq transcription failed (HTTP {}): {}", status, body);
                    return String::new();
                }

                match response.json::<serde_json::Value>().await {
                    Ok(data) => data
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    Err(e) => {
                        error!("Failed to parse Groq transcription response: {}", e);
                        String::new()
                    }
                }
            }
            Err(e) => {
                error!("Groq transcription error: {}", e);
                String::new()
            }
        }
    }
}
