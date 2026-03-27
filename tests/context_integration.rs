use murmur::context::{Context, ContextManager, ContextProvider, DictationMode};

struct FakeProvider {
    label: &'static str,
    context: Context,
}

impl ContextProvider for FakeProvider {
    fn name(&self) -> &str {
        self.label
    }
    fn get_context(&self) -> Context {
        self.context.clone()
    }
}

#[test]
fn integration_empty_manager() {
    let manager = ContextManager::new();
    let ctx = manager.gather();
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
fn integration_single_provider() {
    let mut manager = ContextManager::new();
    manager.add_provider(Box::new(FakeProvider {
        label: "test",
        context: Context {
            app_id: Some("com.test.App".to_string()),
            app_name: Some("Test App".to_string()),
            suggested_mode: Some(DictationMode::Code),
            vocabulary_hints: vec!["rust".to_string(), "murmur".to_string()],
            ..Default::default()
        },
    }));

    let ctx = manager.gather();
    assert_eq!(ctx.app_id.as_deref(), Some("com.test.App"));
    assert_eq!(ctx.app_name.as_deref(), Some("Test App"));
    assert_eq!(ctx.suggested_mode, Some(DictationMode::Code));
    assert_eq!(ctx.vocabulary_hints, vec!["rust", "murmur"]);
}

#[test]
fn integration_multiple_providers_merge() {
    let mut manager = ContextManager::new();

    manager.add_provider(Box::new(FakeProvider {
        label: "app-detector",
        context: Context {
            app_id: Some("com.apple.Terminal".to_string()),
            app_name: Some("Terminal".to_string()),
            vocabulary_hints: vec!["bash".to_string()],
            ..Default::default()
        },
    }));

    manager.add_provider(Box::new(FakeProvider {
        label: "editor-context",
        context: Context {
            file_language: Some("python".to_string()),
            suggested_mode: Some(DictationMode::Code),
            vocabulary_hints: vec!["bash".to_string(), "python".to_string()],
            ..Default::default()
        },
    }));

    manager.add_provider(Box::new(FakeProvider {
        label: "clipboard",
        context: Context {
            clipboard_text: Some("fn main() {}".to_string()),
            ..Default::default()
        },
    }));

    let ctx = manager.gather();
    assert_eq!(ctx.app_id.as_deref(), Some("com.apple.Terminal"));
    assert_eq!(ctx.app_name.as_deref(), Some("Terminal"));
    assert_eq!(ctx.file_language.as_deref(), Some("python"));
    assert_eq!(ctx.suggested_mode, Some(DictationMode::Code));
    assert_eq!(ctx.clipboard_text.as_deref(), Some("fn main() {}"));
    assert_eq!(ctx.vocabulary_hints, vec!["bash", "python"]);
}

#[test]
fn integration_later_provider_overrides_earlier() {
    let mut manager = ContextManager::new();

    manager.add_provider(Box::new(FakeProvider {
        label: "first",
        context: Context {
            suggested_mode: Some(DictationMode::Prose),
            ..Default::default()
        },
    }));

    manager.add_provider(Box::new(FakeProvider {
        label: "second",
        context: Context {
            suggested_mode: Some(DictationMode::Command),
            ..Default::default()
        },
    }));

    let ctx = manager.gather();
    assert_eq!(ctx.suggested_mode, Some(DictationMode::Command));
}

#[test]
fn integration_dictation_mode_serde() {
    let json = r#""code""#;
    let mode: DictationMode = serde_json::from_str(json).unwrap();
    assert_eq!(mode, DictationMode::Code);
    assert_eq!(serde_json::to_string(&mode).unwrap(), r#""code""#);
}

#[test]
fn integration_app_detector_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<murmur::context::AppDetector>();
}
