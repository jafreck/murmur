//! Integration tests for Config file I/O, round-trips, error recovery,
//! and cross-module interactions.

mod helpers;

use murmur::config::{
    is_valid_language, language_name, AppContextConfig, Config, DictationMode, InputMode,
    SUPPORTED_LANGUAGES, SUPPORTED_MODELS,
};

// ═══════════════════════════════════════════════════════════════════════
//  Save / Load round-trips
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn default_config_round_trip() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("config.json");

    let original = Config::default();
    original.save_to(&path).unwrap();

    let loaded = Config::load_from(&path).unwrap();
    assert_eq!(loaded.hotkey(), original.hotkey());
    assert_eq!(loaded.model_size(), original.model_size());
    assert_eq!(loaded.language(), original.language());
    assert_eq!(loaded.spoken_punctuation(), original.spoken_punctuation());
    assert_eq!(loaded.max_recordings(), original.max_recordings());
    assert_eq!(loaded.mode(), original.mode());
    assert_eq!(loaded.streaming(), original.streaming());
    assert_eq!(loaded.translate_to_english(), original.translate_to_english());
}

#[test]
fn custom_config_round_trip() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("config.json");

    let mut config = Config::default();
    config.set_hotkey("ctrl+shift+space".to_string());
    config.set_model_size("small.en".to_string());
    config.set_language("fr".to_string());
    config.set_spoken_punctuation(true);
    config.set_filler_word_removal(true);
    config.set_max_recordings(10);
    config.set_mode(InputMode::OpenMic);
    config.set_streaming(true);
    config.set_translate_to_english(true);
    config.set_vocabulary(vec!["test".to_string()]);
    config.save_to(&path).unwrap();

    let loaded = Config::load_from(&path).unwrap();
    assert_eq!(loaded.hotkey(), "ctrl+shift+space");
    assert_eq!(loaded.model_size(), "small.en");
    assert_eq!(loaded.language(), "fr");
    assert!(loaded.spoken_punctuation());
    assert_eq!(loaded.max_recordings(), 10);
    assert_eq!(*loaded.mode(), InputMode::OpenMic);
    assert!(loaded.streaming());
    assert!(loaded.translate_to_english());
    assert_eq!(loaded.vocabulary(), vec!["test"]);
}

#[test]
fn save_creates_parent_directories() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("nested").join("deep").join("config.json");

    Config::default().save_to(&path).unwrap();
    assert!(path.exists());

    let loaded = Config::load_from(&path).unwrap();
    assert_eq!(loaded.hotkey(), Config::default().hotkey());
}

// ═══════════════════════════════════════════════════════════════════════
//  Corrupt / malformed JSON recovery
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn corrupt_json_returns_parse_error() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("config.json");

    std::fs::write(&path, "this is not json {{{").unwrap();
    let result = Config::load_from(&path);
    assert!(result.is_err());
}

#[test]
fn empty_file_returns_parse_error() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("config.json");

    std::fs::write(&path, "").unwrap();
    let result = Config::load_from(&path);
    assert!(result.is_err());
}

#[test]
fn partial_json_returns_parse_error() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("config.json");

    // Valid JSON but missing required fields
    std::fs::write(&path, r#"{"hotkey": "f5"}"#).unwrap();
    let result = Config::load_from(&path);

    // Missing required fields is a parse error
    assert!(result.is_err());
}

#[test]
fn nonexistent_file_returns_error() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("does_not_exist.json");

    assert!(!path.exists());
    let result = Config::load_from(&path);
    assert!(result.is_err());
}

#[test]
fn html_error_page_returns_parse_error() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("config.json");

    std::fs::write(&path, "<!DOCTYPE html><html>Error 500</html>").unwrap();
    let result = Config::load_from(&path);
    assert!(result.is_err());
}

// ═══════════════════════════════════════════════════════════════════════
//  Multiple save/load cycles
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn multiple_save_load_cycles_preserve_data() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("config.json");

    let mut config = Config::default();

    for model in &["tiny.en", "base.en", "small.en", "medium.en"] {
        config.set_model_size(model.to_string());
        config.save_to(&path).unwrap();

        let loaded = Config::load_from(&path).unwrap();
        assert_eq!(loaded.model_size(), *model);
    }
}

#[test]
fn overwrite_preserves_last_written() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("config.json");

    let mut c1 = Config::default();
    c1.set_hotkey("f5".to_string());
    c1.save_to(&path).unwrap();

    let mut c2 = Config::default();
    c2.set_hotkey("f9".to_string());
    c2.save_to(&path).unwrap();

    let loaded = Config::load_from(&path).unwrap();
    assert_eq!(loaded.hotkey(), "f9");
}

// ═══════════════════════════════════════════════════════════════════════
//  Config ↔ AppState round-trip via to_config
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn config_to_state_and_back() {
    use murmur::app::AppState;

    let mut original = Config::default();
    original.set_hotkey("ctrl+space".to_string());
    original.set_model_size("medium.en".to_string());
    original.set_language("de".to_string());
    original.set_spoken_punctuation(true);
    original.set_filler_word_removal(true);
    original.set_max_recordings(5);
    original.set_mode(InputMode::OpenMic);
    original.set_streaming(true);
    original.set_translate_to_english(true);

    let state = AppState::new(&original);
    let reconstructed = state.to_config(&original);

    assert_eq!(reconstructed.model_size(), "medium.en");
    assert_eq!(reconstructed.language(), "de");
    assert!(reconstructed.spoken_punctuation());
    assert_eq!(*reconstructed.mode(), InputMode::OpenMic);
    assert!(reconstructed.streaming());
    assert!(reconstructed.translate_to_english());
    // hotkey and max_recordings come from the base config
    assert_eq!(reconstructed.hotkey(), "ctrl+space");
    assert_eq!(reconstructed.max_recordings(), 5);
}

#[test]
fn config_save_load_then_state_round_trip() {
    use murmur::app::AppState;

    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("config.json");

    let mut original = Config::default();
    original.set_hotkey("f9".to_string());
    original.set_model_size("small.en".to_string());
    original.set_language("ja".to_string());
    original.set_spoken_punctuation(false);
    original.set_filler_word_removal(true);
    original.set_max_recordings(3);
    original.set_mode(InputMode::PushToTalk);
    original.set_streaming(false);
    original.set_translate_to_english(false);

    original.save_to(&path).unwrap();
    let loaded = Config::load_from(&path).unwrap();
    let state = AppState::new(&loaded);
    let final_config = state.to_config(&loaded);

    assert_eq!(final_config.model_size(), "small.en");
    assert_eq!(final_config.language(), "ja");
    assert_eq!(*final_config.mode(), InputMode::PushToTalk);
}

// ═══════════════════════════════════════════════════════════════════════
//  Supported models & languages validation
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn all_supported_models_are_nonempty() {
    for model in SUPPORTED_MODELS {
        assert!(!model.is_empty());
    }
}

#[test]
fn all_supported_languages_have_names() {
    for (code, name) in SUPPORTED_LANGUAGES {
        assert!(!code.is_empty());
        assert!(!name.is_empty());
        assert!(is_valid_language(code));
        assert!(language_name(code).is_some());
    }
}

#[test]
fn language_name_returns_none_for_invalid() {
    assert!(language_name("zzzz").is_none());
    assert!(language_name("").is_none());
}

#[test]
fn is_valid_language_auto() {
    assert!(is_valid_language("auto"));
}

// ═══════════════════════════════════════════════════════════════════════
//  InputMode serialization
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn input_mode_serialization_round_trip() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("config.json");

    for mode in &[InputMode::PushToTalk, InputMode::OpenMic] {
        let mut config = Config::default();
        config.set_mode(mode.clone());
        config.save_to(&path).unwrap();
        let loaded = Config::load_from(&path).unwrap();
        assert_eq!(*loaded.mode(), *mode);
    }
}

#[test]
fn input_mode_display() {
    assert_eq!(InputMode::PushToTalk.to_string(), "Push to Talk");
    assert_eq!(InputMode::OpenMic.to_string(), "Open Mic");
}

// ═══════════════════════════════════════════════════════════════════════
//  JSON output is human-readable (pretty-printed)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn saved_config_is_pretty_printed() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("config.json");

    Config::default().save_to(&path).unwrap();
    let contents = std::fs::read_to_string(&path).unwrap();

    // Pretty-printed JSON has newlines and indentation
    assert!(contents.contains('\n'), "should be pretty-printed");
    assert!(
        contents.contains("  ") || contents.contains('\t'),
        "should be indented"
    );
}

#[test]
fn saved_config_is_valid_json() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("config.json");

    Config::default().save_to(&path).unwrap();
    let contents = std::fs::read_to_string(&path).unwrap();

    let parsed: serde_json::Value = serde_json::from_str(&contents).expect("should be valid JSON");
    assert!(parsed.is_object());
    assert!(parsed.get("hotkey").is_some());
    assert!(parsed.get("model_size").is_some());
}

// ═══════════════════════════════════════════════════════════════════════
//  effective_max_recordings
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn effective_max_recordings_values() {
    // 0 means unlimited (in-memory mode)
    assert_eq!(Config::effective_max_recordings(0), 0);
    // Positive values are passed through
    assert_eq!(Config::effective_max_recordings(5), 5);
    assert_eq!(Config::effective_max_recordings(1), 1);
}

// ═══════════════════════════════════════════════════════════════════════
//  Vocabulary & app context features
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn vocab_file_round_trip() {
    let tmp = tempfile::TempDir::new().unwrap();
    let vocab_content = "# Project vocabulary\nKubernetes\nkubectl\n\n# More terms\nhelm\n";
    std::fs::write(tmp.path().join(".murmur-vocab"), vocab_content).unwrap();

    let terms = Config::load_vocab_file(tmp.path());
    assert_eq!(terms, vec!["Kubernetes", "kubectl", "helm"]);
}

#[test]
fn effective_vocabulary_merges_all_sources() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::write(tmp.path().join(".murmur-vocab"), "file_term\n").unwrap();

    let mut app_contexts = std::collections::HashMap::new();
    app_contexts.insert(
        "com.terminal".to_string(),
        AppContextConfig {
            vocabulary: vec!["app_term".to_string()],
            mode: None,
        },
    );

    let mut cfg = Config::default();
    cfg.set_vocabulary(vec!["global_term".to_string()]);
    cfg.set_app_contexts(app_contexts);

    let vocab = cfg.effective_vocabulary(Some("com.terminal"), Some(tmp.path()));
    assert_eq!(vocab, vec!["global_term", "app_term", "file_term"]);
}

#[test]
fn effective_vocabulary_deduplicates() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::write(tmp.path().join(".murmur-vocab"), "shared\n").unwrap();

    let mut app_contexts = std::collections::HashMap::new();
    app_contexts.insert(
        "app".to_string(),
        AppContextConfig {
            vocabulary: vec!["shared".to_string()],
            mode: None,
        },
    );

    let mut cfg = Config::default();
    cfg.set_vocabulary(vec!["shared".to_string()]);
    cfg.set_app_contexts(app_contexts);

    let vocab = cfg.effective_vocabulary(Some("app"), Some(tmp.path()));
    assert_eq!(vocab, vec!["shared"]);
}

#[test]
fn app_exclusion_integration() {
    let mut cfg = Config::default();
    cfg.set_excluded_apps(vec!["com.1password".to_string(), "com.chase.mobile".to_string()]);

    assert!(cfg.is_app_excluded("com.1password"));
    assert!(cfg.is_app_excluded("com.chase.mobile"));
    assert!(!cfg.is_app_excluded("com.apple.notes"));
}

#[test]
fn dictation_mode_app_override_integration() {
    let mut app_contexts = std::collections::HashMap::new();
    app_contexts.insert(
        "com.terminal".to_string(),
        AppContextConfig {
            vocabulary: Vec::new(),
            mode: Some(DictationMode::Command),
        },
    );
    app_contexts.insert(
        "com.notes".to_string(),
        AppContextConfig {
            vocabulary: Vec::new(),
            mode: None,
        },
    );

    let mut cfg = Config::default();
    cfg.set_dictation_mode(DictationMode::Prose);
    cfg.set_app_contexts(app_contexts);

    assert_eq!(cfg.effective_dictation_mode(None), DictationMode::Prose);
    assert_eq!(
        cfg.effective_dictation_mode(Some("com.terminal")),
        DictationMode::Command
    );
    assert_eq!(
        cfg.effective_dictation_mode(Some("com.notes")),
        DictationMode::Prose
    );
    assert_eq!(
        cfg.effective_dictation_mode(Some("com.unknown")),
        DictationMode::Prose
    );
}

#[test]
fn full_config_with_vocabulary_save_load() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("config.json");

    let mut app_contexts = std::collections::HashMap::new();
    app_contexts.insert(
        "com.vscode".to_string(),
        AppContextConfig {
            vocabulary: vec!["rustfmt".to_string(), "clippy".to_string()],
            mode: Some(DictationMode::Code),
        },
    );

    let mut cfg = Config::default();
    cfg.set_hotkey("f10".to_string());
    cfg.set_model_size("small.en".to_string());
    cfg.set_language("en".to_string());
    cfg.set_spoken_punctuation(false);
    cfg.set_max_recordings(0);
    cfg.set_mode(InputMode::PushToTalk);
    cfg.set_streaming(false);
    cfg.set_translate_to_english(false);
    cfg.set_vocabulary(vec!["murmur".to_string(), "whisper".to_string()]);
    cfg.set_app_contexts(app_contexts);
    cfg.set_excluded_apps(vec!["com.1password".to_string()]);
    cfg.set_dictation_mode(DictationMode::Prose);
    cfg.set_noise_suppression(true);
    cfg.set_filler_word_removal(true);
    cfg.save_to(&path).unwrap();

    let loaded = Config::load_from(&path).unwrap();
    assert_eq!(loaded.vocabulary(), vec!["murmur", "whisper"]);
    assert_eq!(loaded.excluded_apps(), vec!["com.1password"]);
    assert_eq!(loaded.dictation_mode(), DictationMode::Prose);
    assert!(loaded.filler_word_removal());

    let vscode = loaded.app_contexts().get("com.vscode").unwrap();
    assert_eq!(vscode.vocabulary, vec!["rustfmt", "clippy"]);
    assert_eq!(vscode.mode, Some(DictationMode::Code));

    // Verify effective vocabulary merges correctly
    let vocab = loaded.effective_vocabulary(Some("com.vscode"), None);
    assert_eq!(vocab, vec!["murmur", "whisper", "rustfmt", "clippy"]);
}

#[test]
fn backward_compat_old_config_loads_with_defaults() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("config.json");

    // Write a config without any of the new fields
    let old_json = r#"{
        "hotkey": "f5",
        "model_size": "base.en",
        "language": "en",
        "spoken_punctuation": false,
        "max_recordings": 0,
        "mode": "push_to_talk",
        "streaming": false,
        "translate_to_english": false
    }"#;
    std::fs::write(&path, old_json).unwrap();

    let loaded = Config::load_from(&path).unwrap();
    assert_eq!(loaded.hotkey(), "f5");
    assert!(loaded.vocabulary().is_empty());
    assert!(loaded.app_contexts().is_empty());
    assert!(loaded.excluded_apps().is_empty());
    assert_eq!(loaded.dictation_mode(), DictationMode::Prose);
}
