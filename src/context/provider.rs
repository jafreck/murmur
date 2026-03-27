use serde::{Deserialize, Serialize};

/// Dictation output mode — determines how transcribed text is formatted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DictationMode {
    #[default]
    Prose,
    Code,
    Command,
    List,
}

impl std::fmt::Display for DictationMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DictationMode::Prose => write!(f, "Prose"),
            DictationMode::Code => write!(f, "Code"),
            DictationMode::Command => write!(f, "Command"),
            DictationMode::List => write!(f, "List"),
        }
    }
}

/// Runtime context gathered from the user's environment.
/// Fields are all optional — providers fill in what they can.
#[derive(Debug, Clone, Default)]
pub struct Context {
    /// Bundle ID or process identifier of the focused application (e.g. "com.microsoft.VSCode")
    pub app_id: Option<String>,
    /// Human-readable name of the focused application (e.g. "Visual Studio Code")
    pub app_name: Option<String>,
    /// Title of the focused window
    pub window_title: Option<String>,
    /// Text surrounding the cursor in the active text field (~50 chars before cursor)
    pub surrounding_text: Option<String>,
    /// Current clipboard text content
    pub clipboard_text: Option<String>,
    /// Programming language of the current file (if detectable)
    pub file_language: Option<String>,
    /// Vocabulary terms to bias Whisper toward
    pub vocabulary_hints: Vec<String>,
    /// Suggested dictation mode based on context
    pub suggested_mode: Option<DictationMode>,
}

/// Trait for components that contribute to the runtime context.
pub trait ContextProvider: Send + Sync {
    /// Human-readable name for logging/debugging
    fn name(&self) -> &str;
    /// Gather context. Should be fast and non-blocking.
    fn get_context(&self) -> Context;
}

/// Aggregates multiple context providers and merges their results.
pub struct ContextManager {
    providers: Vec<Box<dyn ContextProvider>>,
}

impl ContextManager {
    /// Create a new empty context manager.
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }

    /// Register a context provider.
    pub fn add_provider(&mut self, provider: Box<dyn ContextProvider>) {
        log::info!("Registered context provider: {}", provider.name());
        self.providers.push(provider);
    }

    /// Call all providers and merge their results.
    ///
    /// Later providers override earlier ones for `Option` fields.
    /// `vocabulary_hints` are concatenated and deduplicated.
    pub fn gather(&self) -> Context {
        let mut merged = Context::default();

        for provider in &self.providers {
            let ctx = provider.get_context();
            log::debug!("Context from provider '{}': {:?}", provider.name(), ctx);

            if ctx.app_id.is_some() {
                merged.app_id = ctx.app_id;
            }
            if ctx.app_name.is_some() {
                merged.app_name = ctx.app_name;
            }
            if ctx.window_title.is_some() {
                merged.window_title = ctx.window_title;
            }
            if ctx.surrounding_text.is_some() {
                merged.surrounding_text = ctx.surrounding_text;
            }
            if ctx.clipboard_text.is_some() {
                merged.clipboard_text = ctx.clipboard_text;
            }
            if ctx.file_language.is_some() {
                merged.file_language = ctx.file_language;
            }
            if ctx.suggested_mode.is_some() {
                merged.suggested_mode = ctx.suggested_mode;
            }

            for hint in ctx.vocabulary_hints {
                if !merged.vocabulary_hints.contains(&hint) {
                    merged.vocabulary_hints.push(hint);
                }
            }
        }

        merged
    }
}

impl Default for ContextManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_default_has_none_fields() {
        let ctx = Context::default();
        assert!(ctx.app_id.is_none());
        assert!(ctx.app_name.is_none());
        assert!(ctx.window_title.is_none());
        assert!(ctx.surrounding_text.is_none());
        assert!(ctx.clipboard_text.is_none());
        assert!(ctx.file_language.is_none());
        assert!(ctx.suggested_mode.is_none());
        assert!(ctx.vocabulary_hints.is_empty());
    }

    #[test]
    fn context_manager_no_providers_returns_default() {
        let manager = ContextManager::new();
        let ctx = manager.gather();
        assert!(ctx.app_id.is_none());
        assert!(ctx.app_name.is_none());
        assert!(ctx.vocabulary_hints.is_empty());
    }

    struct StubProvider {
        name: &'static str,
        context: Context,
    }

    impl ContextProvider for StubProvider {
        fn name(&self) -> &str {
            self.name
        }
        fn get_context(&self) -> Context {
            self.context.clone()
        }
    }

    #[test]
    fn context_manager_merges_later_overrides_earlier() {
        let mut manager = ContextManager::new();

        manager.add_provider(Box::new(StubProvider {
            name: "first",
            context: Context {
                app_id: Some("com.first.App".to_string()),
                app_name: Some("First App".to_string()),
                window_title: Some("First Window".to_string()),
                ..Default::default()
            },
        }));

        manager.add_provider(Box::new(StubProvider {
            name: "second",
            context: Context {
                app_id: Some("com.second.App".to_string()),
                // app_name is None — should keep first provider's value
                file_language: Some("rust".to_string()),
                ..Default::default()
            },
        }));

        let ctx = manager.gather();
        assert_eq!(ctx.app_id.as_deref(), Some("com.second.App"));
        assert_eq!(ctx.app_name.as_deref(), Some("First App"));
        assert_eq!(ctx.window_title.as_deref(), Some("First Window"));
        assert_eq!(ctx.file_language.as_deref(), Some("rust"));
    }

    #[test]
    fn context_manager_deduplicates_vocabulary_hints() {
        let mut manager = ContextManager::new();

        manager.add_provider(Box::new(StubProvider {
            name: "a",
            context: Context {
                vocabulary_hints: vec!["murmur".to_string(), "whisper".to_string()],
                ..Default::default()
            },
        }));

        manager.add_provider(Box::new(StubProvider {
            name: "b",
            context: Context {
                vocabulary_hints: vec!["whisper".to_string(), "dictation".to_string()],
                ..Default::default()
            },
        }));

        let ctx = manager.gather();
        assert_eq!(ctx.vocabulary_hints, vec!["murmur", "whisper", "dictation"]);
    }

    #[test]
    fn dictation_mode_serde_roundtrip() {
        let modes = [
            DictationMode::Prose,
            DictationMode::Code,
            DictationMode::Command,
            DictationMode::List,
        ];

        for mode in &modes {
            let json = serde_json::to_string(mode).unwrap();
            let deserialized: DictationMode = serde_json::from_str(&json).unwrap();
            assert_eq!(*mode, deserialized);
        }
    }

    #[test]
    fn dictation_mode_serde_snake_case() {
        assert_eq!(serde_json::to_string(&DictationMode::Prose).unwrap(), "\"prose\"");
        assert_eq!(serde_json::to_string(&DictationMode::Code).unwrap(), "\"code\"");
        assert_eq!(serde_json::to_string(&DictationMode::Command).unwrap(), "\"command\"");
        assert_eq!(serde_json::to_string(&DictationMode::List).unwrap(), "\"list\"");
    }

    #[test]
    fn dictation_mode_display() {
        assert_eq!(DictationMode::Prose.to_string(), "Prose");
        assert_eq!(DictationMode::Code.to_string(), "Code");
        assert_eq!(DictationMode::Command.to_string(), "Command");
        assert_eq!(DictationMode::List.to_string(), "List");
    }

    #[test]
    fn dictation_mode_default_is_prose() {
        assert_eq!(DictationMode::default(), DictationMode::Prose);
    }
}
