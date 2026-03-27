pub mod ollama;
pub mod prompt;

use anyhow::Result;

/// A message in a conversation with the LLM.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

/// Trait for LLM providers (Ollama, llama.cpp, cloud APIs in future).
pub trait LlmProvider: Send + Sync {
    /// Generate a completion for the given messages.
    fn generate(&self, messages: &[ChatMessage]) -> Result<String>;

    /// Check if the provider is available and has a model loaded.
    fn is_available(&self) -> bool;

    /// Get the name of the current model.
    fn model_name(&self) -> &str;
}
