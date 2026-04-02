mod catalog;
mod persistence;
mod policy;
mod schema;

pub use catalog::*;
pub use schema::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = Config::default();
        assert_eq!(cfg.model_size, "0.6b");
        assert_eq!(cfg.language, "en");
        assert!(!cfg.spoken_punctuation);
        assert_eq!(cfg.max_recordings, 0);
        assert_eq!(cfg.mode, InputMode::PushToTalk);
        assert!(!cfg.streaming);
        assert!(!cfg.translate_to_english);
    }

    #[test]
    fn test_effective_max_recordings() {
        assert_eq!(Config::effective_max_recordings(0), 0);
        assert_eq!(Config::effective_max_recordings(1), 1);
        assert_eq!(Config::effective_max_recordings(50), 50);
        assert_eq!(Config::effective_max_recordings(100), 100);
        assert_eq!(Config::effective_max_recordings(200), 100);
    }

    #[test]
    fn test_is_valid_language() {
        assert!(is_valid_language("en"));
        assert!(is_valid_language("auto"));
        assert!(is_valid_language("fr"));
        assert!(!is_valid_language("xx"));
        assert!(!is_valid_language(""));
    }

    #[test]
    fn test_language_name() {
        assert_eq!(language_name("en"), Some("English"));
        assert_eq!(language_name("auto"), Some("Auto-Detect"));
        assert_eq!(language_name("xx"), None);
    }

    #[test]
    fn test_config_roundtrip() {
        let cfg = Config {
            hotkey: "f9".to_string(),
            model_size: "small.en".to_string(),
            language: "fr".to_string(),
            spoken_punctuation: true,
            filler_word_removal: true,
            max_recordings: 10,
            mode: InputMode::OpenMic,
            streaming: true,
            translate_to_english: true,
            vocabulary: vec!["murmur".to_string()],
            app_contexts: std::collections::HashMap::new(),
            excluded_apps: Vec::new(),
            dictation_mode: DictationMode::Code,
            noise_suppression: true,
            ..Config::default()
        };

        let json = serde_json::to_string(&cfg).unwrap();
        let parsed: Config = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.hotkey, "f9");
        assert_eq!(parsed.model_size, "small.en");
        assert_eq!(parsed.language, "fr");
        assert!(parsed.spoken_punctuation);
        assert_eq!(parsed.max_recordings, 10);
        assert_eq!(parsed.mode, InputMode::OpenMic);
        assert!(parsed.streaming);
        assert!(parsed.translate_to_english);
        assert_eq!(parsed.vocabulary, vec!["murmur".to_string()]);
        assert!(parsed.app_contexts.is_empty());
        assert!(parsed.excluded_apps.is_empty());
        assert_eq!(parsed.dictation_mode, DictationMode::Code);
    }

    #[test]
    fn test_config_dir_and_file_path() {
        let dir = Config::dir();
        assert!(dir.to_string_lossy().contains("murmur"));
        let fp = Config::file_path();
        assert!(fp.to_string_lossy().contains("config.json"));
        assert!(fp.starts_with(&dir));
    }

    #[test]
    fn test_save_to_and_load_from() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("test_config.json");

        let cfg = Config {
            hotkey: "ctrl+shift+a".to_string(),
            model_size: "medium.en".to_string(),
            language: "de".to_string(),
            spoken_punctuation: true,
            filler_word_removal: true,
            max_recordings: 5,
            mode: InputMode::OpenMic,
            streaming: false,
            translate_to_english: false,
            vocabulary: vec!["test".to_string()],
            app_contexts: std::collections::HashMap::new(),
            excluded_apps: vec!["com.bank.app".to_string()],
            dictation_mode: DictationMode::Prose,
            noise_suppression: true,
            ..Config::default()
        };
        cfg.save_to(&path).unwrap();

        let loaded = Config::load_from(&path);
        assert_eq!(loaded.hotkey, "ctrl+shift+a");
        assert_eq!(loaded.model_size, "medium.en");
        assert_eq!(loaded.language, "de");
        assert!(loaded.spoken_punctuation);
        assert_eq!(loaded.max_recordings, 5);
        assert_eq!(loaded.mode, InputMode::OpenMic);
        assert!(!loaded.translate_to_english);
        assert_eq!(loaded.vocabulary, vec!["test".to_string()]);
        assert!(loaded.app_contexts.is_empty());
        assert_eq!(loaded.excluded_apps, vec!["com.bank.app".to_string()]);
        assert_eq!(loaded.dictation_mode, DictationMode::Prose);
    }

    #[test]
    fn test_load_from_nonexistent_creates_default() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("nonexistent.json");
        let loaded = Config::load_from(&path);
        assert_eq!(loaded.model_size, "0.6b");
        // Should have created the default config file
        assert!(path.exists());
    }

    #[test]
    fn test_parse_invalid_json_returns_default() {
        let path = std::path::Path::new("/tmp/test_invalid.json");
        let cfg = Config::parse("not valid json", path);
        assert_eq!(cfg.model_size, "0.6b");
    }

    #[test]
    fn test_parse_valid_json() {
        let json = r#"{"hotkey":"f5","model_size":"tiny","language":"ja","spoken_punctuation":false,"max_recordings":0,"mode":"push_to_talk","streaming":false,"translate_to_english":false}"#;
        let path = std::path::Path::new("/tmp/test.json");
        let cfg = Config::parse(json, path);
        assert_eq!(cfg.hotkey, "f5");
        assert_eq!(cfg.model_size, "tiny");
        assert_eq!(cfg.language, "ja");
    }

    #[test]
    fn test_serde_defaults() {
        // JSON without optional fields should use defaults
        let json = r#"{"hotkey":"f9","model_size":"base.en","language":"en"}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert!(!cfg.spoken_punctuation);
        assert_eq!(cfg.max_recordings, 0);
        assert_eq!(cfg.mode, InputMode::PushToTalk);
        assert!(!cfg.streaming);
        assert!(!cfg.translate_to_english);
    }

    #[test]
    fn test_supported_models_contains_expected() {
        assert!(SUPPORTED_MODELS.contains(&"tiny.en"));
        assert!(SUPPORTED_MODELS.contains(&"base.en"));
        assert!(SUPPORTED_MODELS.contains(&"small.en"));
        assert!(SUPPORTED_MODELS.contains(&"medium.en"));
        assert!(SUPPORTED_MODELS.contains(&"large-v3-turbo"));
        assert!(SUPPORTED_MODELS.contains(&"large"));
        assert!(SUPPORTED_MODELS.contains(&"distil-large-v3"));
        assert!(!SUPPORTED_MODELS.contains(&"nonexistent"));
    }

    #[test]
    fn test_supported_languages_coverage() {
        // Test a variety of languages
        for &(code, name) in SUPPORTED_LANGUAGES {
            assert!(is_valid_language(code));
            assert_eq!(language_name(code), Some(name));
        }
    }

    #[test]
    fn test_save_to_creates_parent_dirs() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("a").join("b").join("config.json");
        let cfg = Config::default();
        cfg.save_to(&path).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_effective_max_recordings_boundary() {
        assert_eq!(Config::effective_max_recordings(0), 0);
        assert_eq!(Config::effective_max_recordings(1), 1);
        assert_eq!(Config::effective_max_recordings(100), 100);
        assert_eq!(Config::effective_max_recordings(101), 100);
        assert_eq!(Config::effective_max_recordings(u32::MAX), 100);
    }

    #[test]
    fn test_input_mode_display() {
        assert_eq!(InputMode::PushToTalk.to_string(), "Push to Talk");
        assert_eq!(InputMode::OpenMic.to_string(), "Open Mic");
    }

    #[test]
    fn test_input_mode_default() {
        let mode: InputMode = Default::default();
        assert_eq!(mode, InputMode::PushToTalk);
    }

    #[test]
    fn test_input_mode_serde_round_trip() {
        let push = InputMode::PushToTalk;
        let json = serde_json::to_string(&push).unwrap();
        assert_eq!(json, "\"push_to_talk\"");
        let parsed: InputMode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, InputMode::PushToTalk);

        let open = InputMode::OpenMic;
        let json = serde_json::to_string(&open).unwrap();
        assert_eq!(json, "\"open_mic\"");
        let parsed: InputMode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, InputMode::OpenMic);
    }

    #[test]
    fn test_default_config_hotkey_platform() {
        let cfg = Config::default();
        #[cfg(target_os = "macos")]
        assert_eq!(cfg.hotkey, "rightoption");
        #[cfg(not(target_os = "macos"))]
        assert_eq!(cfg.hotkey, "rightalt");
    }

    #[test]
    fn test_config_all_fields_serialize() {
        let cfg = Config {
            hotkey: "f5".to_string(),
            model_size: "large".to_string(),
            language: "auto".to_string(),
            spoken_punctuation: true,
            filler_word_removal: true,
            max_recordings: 50,
            mode: InputMode::OpenMic,
            streaming: true,
            translate_to_english: true,
            vocabulary: vec!["Kubernetes".to_string()],
            app_contexts: std::collections::HashMap::new(),
            excluded_apps: Vec::new(),
            dictation_mode: DictationMode::Command,
            noise_suppression: true,
            ..Config::default()
        };
        let json = serde_json::to_string_pretty(&cfg).unwrap();
        assert!(json.contains("\"hotkey\": \"f5\""));
        assert!(json.contains("\"streaming\": true"));
        assert!(json.contains("\"translate_to_english\": true"));
        assert!(json.contains("\"mode\": \"open_mic\""));
        assert!(json.contains("\"Kubernetes\""));
        assert!(json.contains("\"dictation_mode\": \"command\""));
    }

    #[test]
    fn test_supported_languages_has_auto() {
        assert!(is_valid_language("auto"));
        assert_eq!(language_name("auto"), Some("Auto-Detect"));
    }

    #[test]
    fn test_supported_languages_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for (code, _) in SUPPORTED_LANGUAGES {
            assert!(seen.insert(*code), "duplicate language code: {code}");
        }
    }

    #[test]
    fn test_supported_models_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for model in SUPPORTED_MODELS {
            assert!(seen.insert(*model), "duplicate model: {model}");
        }
    }

    #[test]
    fn test_dictation_mode_serde_round_trip() {
        for (mode, expected_json) in [
            (DictationMode::Prose, "\"prose\""),
            (DictationMode::Code, "\"code\""),
            (DictationMode::Command, "\"command\""),
            (DictationMode::List, "\"list\""),
        ] {
            let json = serde_json::to_string(&mode).unwrap();
            assert_eq!(json, expected_json);
            let parsed: DictationMode = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, mode);
        }
    }

    #[test]
    fn test_dictation_mode_display() {
        assert_eq!(DictationMode::Prose.to_string(), "Prose");
        assert_eq!(DictationMode::Code.to_string(), "Code");
        assert_eq!(DictationMode::Command.to_string(), "Command");
        assert_eq!(DictationMode::List.to_string(), "List");
    }

    #[test]
    fn test_dictation_mode_default() {
        let mode: DictationMode = Default::default();
        assert_eq!(mode, DictationMode::Prose);
    }

    #[test]
    fn test_app_context_config_serde() {
        let ctx = AppContextConfig {
            vocabulary: vec!["kubectl".to_string(), "nginx".to_string()],
            mode: Some(DictationMode::Command),
        };
        let json = serde_json::to_string(&ctx).unwrap();
        let parsed: AppContextConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.vocabulary, vec!["kubectl", "nginx"]);
        assert_eq!(parsed.mode, Some(DictationMode::Command));
    }

    #[test]
    fn test_serde_defaults_new_fields() {
        // Old JSON without the new fields should still deserialize with defaults
        let json = r#"{"hotkey":"f9","model_size":"base.en","language":"en"}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert!(cfg.vocabulary.is_empty());
        assert!(cfg.app_contexts.is_empty());
        assert!(cfg.excluded_apps.is_empty());
        assert_eq!(cfg.dictation_mode, DictationMode::Prose);
    }

    #[test]
    fn test_effective_vocabulary_global_only() {
        let cfg = Config {
            vocabulary: vec!["alpha".to_string(), "beta".to_string()],
            ..Config::default()
        };
        let vocab = cfg.effective_vocabulary(None, None);
        assert_eq!(vocab, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_effective_vocabulary_with_app() {
        let mut app_contexts = std::collections::HashMap::new();
        app_contexts.insert(
            "com.editor.code".to_string(),
            AppContextConfig {
                vocabulary: vec!["rustfmt".to_string()],
                mode: None,
            },
        );
        let cfg = Config {
            vocabulary: vec!["global".to_string()],
            app_contexts,
            ..Config::default()
        };
        let vocab = cfg.effective_vocabulary(Some("com.editor.code"), None);
        assert_eq!(vocab, vec!["global", "rustfmt"]);
    }

    #[test]
    fn test_effective_vocabulary_dedup() {
        let mut app_contexts = std::collections::HashMap::new();
        app_contexts.insert(
            "app".to_string(),
            AppContextConfig {
                vocabulary: vec!["dup".to_string(), "unique".to_string()],
                mode: None,
            },
        );
        let cfg = Config {
            vocabulary: vec!["dup".to_string(), "other".to_string()],
            app_contexts,
            ..Config::default()
        };
        let vocab = cfg.effective_vocabulary(Some("app"), None);
        assert_eq!(vocab, vec!["dup", "other", "unique"]);
    }

    #[test]
    fn test_effective_vocabulary_with_vocab_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join(".murmur-vocab"), "file_term\nanother\n").unwrap();
        let cfg = Config {
            vocabulary: vec!["global".to_string()],
            ..Config::default()
        };
        let vocab = cfg.effective_vocabulary(None, Some(tmp.path()));
        assert_eq!(vocab, vec!["global", "file_term", "another"]);
    }

    #[test]
    fn test_load_vocab_file_missing() {
        let tmp = tempfile::TempDir::new().unwrap();
        let result = Config::load_vocab_file(tmp.path());
        assert!(result.is_empty());
    }

    #[test]
    fn test_load_vocab_file_with_comments() {
        let tmp = tempfile::TempDir::new().unwrap();
        let content = "# This is a comment\nterm1\n\n# Another comment\nterm2\n  \n";
        std::fs::write(tmp.path().join(".murmur-vocab"), content).unwrap();
        let result = Config::load_vocab_file(tmp.path());
        assert_eq!(result, vec!["term1", "term2"]);
    }

    #[test]
    fn test_is_app_excluded() {
        let cfg = Config {
            excluded_apps: vec!["com.1password".to_string(), "com.bank.app".to_string()],
            ..Config::default()
        };
        assert!(cfg.is_app_excluded("com.1password"));
        assert!(cfg.is_app_excluded("com.bank.app"));
        assert!(!cfg.is_app_excluded("com.editor.code"));
    }

    #[test]
    fn test_effective_dictation_mode_default() {
        let cfg = Config::default();
        assert_eq!(cfg.effective_dictation_mode(None), DictationMode::Prose);
    }

    #[test]
    fn test_effective_dictation_mode_app_override() {
        let mut app_contexts = std::collections::HashMap::new();
        app_contexts.insert(
            "com.terminal".to_string(),
            AppContextConfig {
                vocabulary: Vec::new(),
                mode: Some(DictationMode::Command),
            },
        );
        let cfg = Config {
            app_contexts,
            ..Config::default()
        };
        assert_eq!(
            cfg.effective_dictation_mode(Some("com.terminal")),
            DictationMode::Command
        );
    }

    #[test]
    fn test_effective_dictation_mode_app_without_mode() {
        let mut app_contexts = std::collections::HashMap::new();
        app_contexts.insert(
            "com.notes".to_string(),
            AppContextConfig {
                vocabulary: vec!["note".to_string()],
                mode: None,
            },
        );
        let cfg = Config {
            dictation_mode: DictationMode::List,
            app_contexts,
            ..Config::default()
        };
        assert_eq!(
            cfg.effective_dictation_mode(Some("com.notes")),
            DictationMode::List
        );
    }

    #[test]
    fn test_config_with_app_contexts_roundtrip() {
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

        let cfg = Config {
            vocabulary: vec!["murmur".to_string()],
            app_contexts,
            excluded_apps: vec!["com.1password".to_string()],
            dictation_mode: DictationMode::Prose,
            ..Config::default()
        };
        cfg.save_to(&path).unwrap();

        let loaded = Config::load_from(&path);
        assert_eq!(loaded.vocabulary, vec!["murmur"]);
        assert_eq!(loaded.excluded_apps, vec!["com.1password"]);
        assert_eq!(loaded.dictation_mode, DictationMode::Prose);
        let vscode_ctx = loaded.app_contexts.get("com.vscode").unwrap();
        assert_eq!(vscode_ctx.vocabulary, vec!["rustfmt", "clippy"]);
        assert_eq!(vscode_ctx.mode, Some(DictationMode::Code));
    }

    #[test]
    fn is_english_only_model_detects_en_suffix() {
        assert!(is_english_only_model("base.en"));
        assert!(is_english_only_model("tiny.en"));
        assert!(is_english_only_model("medium.en"));
    }

    #[test]
    fn is_english_only_model_detects_distil_prefix() {
        assert!(is_english_only_model("distil-large-v3"));
    }

    #[test]
    fn is_english_only_model_rejects_multilingual() {
        assert!(!is_english_only_model("base"));
        assert!(!is_english_only_model("large"));
        assert!(!is_english_only_model("large-v3-turbo"));
        assert!(!is_english_only_model("tiny"));
    }
}
