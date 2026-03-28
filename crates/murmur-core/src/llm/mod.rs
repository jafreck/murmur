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

/// Strip `<think>...</think>` blocks from model output.
/// Some models (e.g. Qwen3) include chain-of-thought reasoning in these
/// tags. We keep only the final answer for display.
pub fn strip_thinking(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<think>") {
        result.push_str(&rest[..start]);
        if let Some(end) = rest[start..].find("</think>") {
            rest = &rest[start + end + "</think>".len()..];
        } else {
            // Unclosed <think> — drop everything after it
            return result.trim().to_string();
        }
    }
    result.push_str(rest);
    result.trim().to_string()
}

/// Format `<think>...</think>` blocks as separate sections for display.
/// Returns the thinking text and the final answer as a tuple.
pub fn split_thinking(text: &str) -> (Option<String>, String) {
    let mut thinking = String::new();
    let mut answer = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<think>") {
        answer.push_str(&rest[..start]);
        if let Some(end) = rest[start..].find("</think>") {
            let think_content = &rest[start + "<think>".len()..start + end];
            thinking.push_str(think_content.trim());
            rest = &rest[start + end + "</think>".len()..];
        } else {
            break;
        }
    }
    answer.push_str(rest);
    let answer = answer.trim().to_string();
    if thinking.is_empty() {
        (None, answer)
    } else {
        (Some(thinking), answer)
    }
}
