use log::{info, warn};
use murmur_core::llm::{
    ollama::OllamaProvider,
    prompt::{action_items_prompt, suggestion_prompt, summary_prompt},
    ChatMessage, LlmProvider, Role,
};

/// Manages the LLM lifecycle for the copilot.
pub struct LlmManager {
    provider: Option<Box<dyn LlmProvider>>,
}

impl LlmManager {
    /// Try to connect to Ollama. If unavailable, the manager is created but
    /// suggestions/summaries will return `None`.
    pub fn try_connect(url: &str, model: &str) -> Self {
        let provider = OllamaProvider::with_url(url, model);
        if provider.is_available() {
            info!("LLM connected — model '{model}' via Ollama");
            Self {
                provider: Some(Box::new(provider)),
            }
        } else {
            warn!("Ollama not available — LLM features disabled");
            Self { provider: None }
        }
    }

    /// Whether an LLM provider is connected and ready.
    pub fn is_available(&self) -> bool {
        self.provider.is_some()
    }

    /// The name of the active model, if any.
    pub fn model_name(&self) -> Option<&str> {
        self.provider.as_ref().map(|p| p.model_name())
    }

    /// Generate contextual suggestions from recent transcript.
    pub fn suggest(&self, transcript: &str) -> Option<String> {
        let provider = self.provider.as_ref()?;
        let messages = suggestion_prompt(transcript);
        match provider.generate(&messages) {
            Ok(response) => Some(response),
            Err(e) => {
                warn!("LLM suggestion failed: {e}");
                None
            }
        }
    }

    /// Generate a post-meeting summary.
    pub fn summarize(&self, transcript: &str) -> Option<String> {
        let provider = self.provider.as_ref()?;
        let messages = summary_prompt(transcript);
        match provider.generate(&messages) {
            Ok(response) => Some(response),
            Err(e) => {
                warn!("LLM summary failed: {e}");
                None
            }
        }
    }

    /// Extract action items from a transcript.
    pub fn action_items(&self, transcript: &str) -> Option<String> {
        let provider = self.provider.as_ref()?;
        let messages = action_items_prompt(transcript);
        match provider.generate(&messages) {
            Ok(response) => Some(response),
            Err(e) => {
                warn!("LLM action items extraction failed: {e}");
                None
            }
        }
    }

    /// Answer a free-form question using recent transcript as context.
    pub fn ask(&self, question: &str, context: &str) -> Option<String> {
        let provider = self.provider.as_ref()?;
        let messages = vec![
            ChatMessage {
                role: Role::System,
                content: "You are a helpful voice assistant. Answer concisely. \
                          The user has been dictating and asked a question. \
                          Here is the recent transcript for context."
                    .to_string(),
            },
            ChatMessage {
                role: Role::User,
                content: context.to_string(),
            },
            ChatMessage {
                role: Role::User,
                content: question.to_string(),
            },
        ];
        match provider.generate(&messages) {
            Ok(response) => Some(response),
            Err(e) => {
                warn!("LLM ask failed: {e}");
                None
            }
        }
    }
}
