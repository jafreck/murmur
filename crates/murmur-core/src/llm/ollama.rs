use anyhow::{Context, Result};
use log::{info, warn};

use super::{ChatMessage, LlmProvider};

/// Ollama LLM provider — communicates with a local Ollama instance via HTTP.
/// Ollama must be installed and running (`ollama serve`).
pub struct OllamaProvider {
    base_url: String,
    model: String,
}

/// Response from the Ollama `/api/tags` endpoint.
#[derive(serde::Deserialize)]
struct TagsResponse {
    models: Vec<TagModel>,
}

#[derive(serde::Deserialize)]
struct TagModel {
    name: String,
}

/// Request body for `/api/chat`.
#[derive(serde::Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    stream: bool,
}

/// Response from `/api/chat` (non-streaming).
#[derive(serde::Deserialize)]
struct ChatResponse {
    message: ChatResponseMessage,
}

#[derive(serde::Deserialize)]
struct ChatResponseMessage {
    content: String,
}

/// Request body for `/api/pull`.
#[derive(serde::Serialize)]
struct PullRequest<'a> {
    model: &'a str,
    stream: bool,
}

impl OllamaProvider {
    pub fn new(model: &str) -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            model: model.to_string(),
        }
    }

    pub fn with_url(base_url: &str, model: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        }
    }

    /// Check if Ollama is running and the configured model is available.
    pub fn check_health(&self) -> Result<bool> {
        let url = format!("{}/api/tags", self.base_url);
        let resp = reqwest::blocking::get(&url).context("failed to reach Ollama")?;
        let tags: TagsResponse = resp
            .json()
            .context("invalid response from Ollama /api/tags")?;
        let available = tags
            .models
            .iter()
            .any(|m| m.name == self.model || m.name.starts_with(&format!("{}:", self.model)));
        Ok(available)
    }

    /// Pull the configured model if it is not already available.
    pub fn ensure_model(&self) -> Result<()> {
        if self.check_health().unwrap_or(false) {
            info!("model '{}' already available", self.model);
            return Ok(());
        }

        info!("pulling model '{}'…", self.model);
        let url = format!("{}/api/pull", self.base_url);
        let client = reqwest::blocking::Client::new();
        let resp = client
            .post(&url)
            .json(&PullRequest {
                model: &self.model,
                stream: false,
            })
            .send()
            .context("failed to pull model from Ollama")?;

        if !resp.status().is_success() {
            anyhow::bail!(
                "Ollama pull failed with status {}: {}",
                resp.status(),
                resp.text().unwrap_or_default()
            );
        }

        info!("model '{}' pulled successfully", self.model);
        Ok(())
    }
}

impl LlmProvider for OllamaProvider {
    fn generate(&self, messages: &[ChatMessage]) -> Result<String> {
        let url = format!("{}/api/chat", self.base_url);
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()?;

        let resp = client
            .post(&url)
            .json(&ChatRequest {
                model: &self.model,
                messages,
                stream: false,
            })
            .send()
            .context("failed to send chat request to Ollama")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            anyhow::bail!("Ollama chat failed ({status}): {body}");
        }

        let chat: ChatResponse = resp
            .json()
            .context("invalid response from Ollama /api/chat")?;
        Ok(chat.message.content)
    }

    fn is_available(&self) -> bool {
        self.check_health().unwrap_or_else(|e| {
            warn!("Ollama health check failed: {e}");
            false
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let p = OllamaProvider::new("phi3");
        assert_eq!(p.model_name(), "phi3");
        assert_eq!(p.base_url, "http://localhost:11434");
    }

    #[test]
    fn test_provider_with_custom_url() {
        let p = OllamaProvider::with_url("http://192.168.1.10:11434/", "qwen2.5:1.5b");
        assert_eq!(p.model_name(), "qwen2.5:1.5b");
        assert_eq!(p.base_url, "http://192.168.1.10:11434");
    }
}
