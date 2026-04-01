use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level application mode.
///
/// **Dictation** — transcription is pasted at the cursor via hotkey.
/// **Notes** — transcription is shown in an overlay and saved to note files,
/// triggered by wake word.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppMode {
    #[default]
    Dictation,
    Notes,
}

impl std::fmt::Display for AppMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppMode::Dictation => write!(f, "Dictation"),
            AppMode::Notes => write!(f, "Notes"),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputMode {
    #[default]
    PushToTalk,
    OpenMic,
}

impl std::fmt::Display for InputMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InputMode::PushToTalk => write!(f, "Push to Talk"),
            InputMode::OpenMic => write!(f, "Open Mic"),
        }
    }
}

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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AppContextConfig {
    /// Vocabulary terms to bias Whisper toward when this app is focused
    #[serde(default)]
    pub vocabulary: Vec<String>,
    /// Dictation mode to use when this app is focused
    #[serde(default)]
    pub mode: Option<DictationMode>,
}

/// ASR backend engine.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AsrBackend {
    Whisper,
    #[default]
    Qwen3Asr,
    Parakeet,
}

impl std::fmt::Display for AsrBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AsrBackend::Whisper => write!(f, "Whisper"),
            AsrBackend::Qwen3Asr => write!(f, "Qwen3-ASR"),
            AsrBackend::Parakeet => write!(f, "Parakeet"),
        }
    }
}

/// ONNX model quantization level.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AsrQuantization {
    Fp32,
    #[default]
    Int4,
    Int8,
}

impl std::fmt::Display for AsrQuantization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AsrQuantization::Fp32 => write!(f, "FP32"),
            AsrQuantization::Int4 => write!(f, "INT4"),
            AsrQuantization::Int8 => write!(f, "INT8"),
        }
    }
}

pub const WHISPER_MODELS: &[&str] = &[
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large-v3-turbo",
    "large",
    "distil-large-v3",
];

pub const QWEN3_ASR_MODELS: &[&str] = &["0.6b", "1.7b"];

pub const PARAKEET_MODELS: &[&str] = &["0.6b-v2"];

/// All supported models for a given backend.
pub fn supported_models(backend: AsrBackend) -> &'static [&'static str] {
    match backend {
        AsrBackend::Whisper => WHISPER_MODELS,
        AsrBackend::Qwen3Asr => QWEN3_ASR_MODELS,
        AsrBackend::Parakeet => PARAKEET_MODELS,
    }
}

/// Deprecated: use `WHISPER_MODELS` or `supported_models()` instead.
pub const SUPPORTED_MODELS: &[&str] = WHISPER_MODELS;

/// Returns true for models that only support English (`.en` suffix or `distil-*`).
pub fn is_english_only_model(model: &str) -> bool {
    model.ends_with(".en") || model.starts_with("distil-")
}

pub const SUPPORTED_LANGUAGES: &[(&str, &str)] = &[
    ("auto", "Auto-Detect"),
    ("en", "English"),
    ("zh", "Chinese"),
    ("de", "German"),
    ("es", "Spanish"),
    ("ru", "Russian"),
    ("ko", "Korean"),
    ("fr", "French"),
    ("ja", "Japanese"),
    ("pt", "Portuguese"),
    ("tr", "Turkish"),
    ("pl", "Polish"),
    ("nl", "Dutch"),
    ("ar", "Arabic"),
    ("sv", "Swedish"),
    ("it", "Italian"),
    ("id", "Indonesian"),
    ("hi", "Hindi"),
    ("fi", "Finnish"),
    ("vi", "Vietnamese"),
    ("he", "Hebrew"),
    ("uk", "Ukrainian"),
    ("el", "Greek"),
    ("cs", "Czech"),
    ("ro", "Romanian"),
    ("da", "Danish"),
    ("hu", "Hungarian"),
    ("no", "Norwegian"),
    ("th", "Thai"),
    ("ca", "Catalan"),
    ("sk", "Slovak"),
    ("hr", "Croatian"),
    ("bg", "Bulgarian"),
    ("lt", "Lithuanian"),
    ("sl", "Slovenian"),
    ("et", "Estonian"),
    ("lv", "Latvian"),
    ("sr", "Serbian"),
    ("mk", "Macedonian"),
    ("ta", "Tamil"),
    ("te", "Telugu"),
    ("ml", "Malayalam"),
    ("kn", "Kannada"),
    ("bn", "Bengali"),
    ("mr", "Marathi"),
    ("gu", "Gujarati"),
    ("pa", "Punjabi"),
    ("ur", "Urdu"),
    ("fa", "Persian"),
    ("sw", "Swahili"),
    ("af", "Afrikaans"),
    ("ms", "Malay"),
    ("az", "Azerbaijani"),
    ("sq", "Albanian"),
    ("hy", "Armenian"),
    ("ka", "Georgian"),
    ("ne", "Nepali"),
    ("mn", "Mongolian"),
    ("bs", "Bosnian"),
    ("kk", "Kazakh"),
    ("gl", "Galician"),
    ("eu", "Basque"),
    ("is", "Icelandic"),
    ("cy", "Welsh"),
    ("la", "Latin"),
    ("haw", "Hawaiian"),
    ("jw", "Javanese"),
];

pub fn is_valid_language(code: &str) -> bool {
    SUPPORTED_LANGUAGES.iter().any(|(c, _)| *c == code)
}

pub fn language_name(code: &str) -> Option<&str> {
    SUPPORTED_LANGUAGES
        .iter()
        .find(|(c, _)| *c == code)
        .map(|(_, name)| *name)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub hotkey: String,
    pub model_size: String,
    /// ASR backend engine (default: whisper)
    #[serde(default)]
    pub asr_backend: AsrBackend,
    /// ONNX model quantization level (default: int4, only used for ONNX backends)
    #[serde(default)]
    pub asr_quantization: AsrQuantization,
    pub language: String,
    #[serde(default)]
    pub spoken_punctuation: bool,
    #[serde(default)]
    pub filler_word_removal: bool,
    #[serde(default)]
    pub max_recordings: u32,
    #[serde(default)]
    pub mode: InputMode,
    #[serde(default)]
    pub streaming: bool,
    #[serde(default)]
    pub translate_to_english: bool,
    /// Enable noise suppression via nnnoiseless (default: true)
    #[serde(default = "default_true")]
    pub noise_suppression: bool,
    /// Global vocabulary terms to bias Whisper toward
    #[serde(default)]
    pub vocabulary: Vec<String>,
    /// Per-application context configurations, keyed by bundle ID or process name
    #[serde(default)]
    pub app_contexts: std::collections::HashMap<String, AppContextConfig>,
    /// App identifiers to exclude from context capture (password managers, banking apps)
    #[serde(default)]
    pub excluded_apps: Vec<String>,
    /// Default dictation mode
    #[serde(default)]
    pub dictation_mode: DictationMode,
    /// Application mode: `dictation` (paste at cursor) or `notes` (overlay + wake word)
    #[serde(default)]
    pub app_mode: AppMode,
    /// Phrase that triggers dictation when spoken (default: "murmur start dictation")
    #[serde(default = "default_wake_word")]
    pub wake_word: String,
    /// Phrase that stops dictation when spoken (default: "murmur stop dictation")
    #[serde(default = "default_stop_phrase")]
    pub stop_phrase: String,
    /// Directory for saving dictation notes (default: data_dir/murmur/notes)
    #[serde(default)]
    pub notes_dir: Option<std::path::PathBuf>,
    /// Input device name for system audio capture (e.g. "BlackHole 2ch").
    /// When set, meeting sessions capture both mic and system audio.
    #[serde(default)]
    pub system_audio_device: Option<String>,
    /// Hide the overlay window from screen capture and screen sharing.
    #[serde(default)]
    pub stealth_mode: bool,
    /// LLM model name for Ollama (default: "phi3")
    #[serde(default = "default_llm_model")]
    pub llm_model: String,
    /// Ollama API base URL
    #[serde(default = "default_ollama_url")]
    pub ollama_url: String,
    /// Directory for storing meeting sessions
    #[serde(default)]
    pub sessions_dir: Option<String>,
    /// Auto-generate summary when meeting ends
    #[serde(default)]
    pub auto_summary: bool,
    /// Automatically check for and apply updates on startup (default: false)
    #[serde(default)]
    pub auto_update: bool,
}

fn default_true() -> bool {
    true
}

fn default_wake_word() -> String {
    "murmur start dictation".to_string()
}

fn default_stop_phrase() -> String {
    "murmur stop dictation".to_string()
}

fn default_llm_model() -> String {
    "phi3".to_string()
}

fn default_ollama_url() -> String {
    "http://localhost:11434".to_string()
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hotkey: default_hotkey().to_string(),
            model_size: "0.6b".to_string(),
            asr_backend: AsrBackend::default(),
            asr_quantization: AsrQuantization::default(),
            language: "en".to_string(),
            spoken_punctuation: false,
            filler_word_removal: false,
            max_recordings: 0,
            mode: InputMode::PushToTalk,
            streaming: false,
            translate_to_english: false,
            noise_suppression: true,
            vocabulary: Vec::new(),
            app_contexts: std::collections::HashMap::new(),
            excluded_apps: Vec::new(),
            dictation_mode: DictationMode::default(),
            app_mode: AppMode::default(),
            wake_word: default_wake_word(),
            stop_phrase: default_stop_phrase(),
            notes_dir: None,
            system_audio_device: None,
            stealth_mode: false,
            llm_model: default_llm_model(),
            ollama_url: default_ollama_url(),
            sessions_dir: None,
            auto_summary: false,
            auto_update: false,
        }
    }
}

fn default_hotkey() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "rightoption"
    }
    #[cfg(not(target_os = "macos"))]
    {
        "rightalt"
    }
}

impl Config {
    pub fn dir() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("murmur")
    }

    pub fn file_path() -> PathBuf {
        Self::dir().join("config.json")
    }

    /// Whether the app is in Notes mode.
    pub fn is_notes_mode(&self) -> bool {
        self.app_mode == AppMode::Notes
    }

    /// Default model size for the current ASR backend.
    pub fn default_model_for_backend(&self) -> &'static str {
        match self.asr_backend {
            AsrBackend::Whisper => "base.en",
            AsrBackend::Qwen3Asr => "0.6b",
            AsrBackend::Parakeet => "0.6b-v2",
        }
    }

    /// Whether the current backend produces pre-formatted output (punctuation, capitalization).
    pub fn backend_has_native_formatting(&self) -> bool {
        matches!(self.asr_backend, AsrBackend::Parakeet)
    }

    /// Resolved notes directory, falling back to data_dir/murmur/notes.
    pub fn notes_dir(&self) -> PathBuf {
        self.notes_dir.clone().unwrap_or_else(|| {
            dirs::data_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("murmur")
                .join("notes")
        })
    }

    pub fn load() -> Self {
        Self::load_from(&Self::file_path())
    }

    pub fn load_from(path: &std::path::Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(contents) => Self::parse(&contents, path),
            Err(_) => {
                let config = Self::default();
                let _ = config.save_to(path);
                config
            }
        }
    }

    pub fn parse(contents: &str, source: &std::path::Path) -> Self {
        match serde_json::from_str::<Config>(contents) {
            Ok(config) => config,
            Err(e) => {
                eprintln!("Warning: unable to parse {}: {e}", source.display());
                Self::default()
            }
        }
    }

    pub fn save(&self) -> Result<()> {
        self.save_to(&Self::file_path())
    }

    pub fn save_to(&self, path: &std::path::Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn effective_max_recordings(value: u32) -> u32 {
        if value == 0 {
            0
        } else {
            value.clamp(1, 100)
        }
    }

    /// Load vocabulary terms from a `.murmur-vocab` file if it exists in the given directory.
    /// The file contains one term per line. Empty lines and lines starting with # are ignored.
    pub fn load_vocab_file(dir: &std::path::Path) -> Vec<String> {
        let path = dir.join(".murmur-vocab");
        match std::fs::read_to_string(&path) {
            Ok(contents) => contents
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(String::from)
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Collect all effective vocabulary: global config + app-specific + vocab file.
    pub fn effective_vocabulary(
        &self,
        app_id: Option<&str>,
        project_dir: Option<&std::path::Path>,
    ) -> Vec<String> {
        let mut vocab: Vec<String> = self.vocabulary.clone();

        if let Some(id) = app_id {
            if let Some(app_ctx) = self.app_contexts.get(id) {
                vocab.extend(app_ctx.vocabulary.iter().cloned());
            }
        }

        if let Some(dir) = project_dir {
            vocab.extend(Self::load_vocab_file(dir));
        }

        // Deduplicate while preserving order
        let mut seen = std::collections::HashSet::new();
        vocab.retain(|v| seen.insert(v.clone()));
        vocab
    }

    /// Check if an app is excluded from context capture.
    pub fn is_app_excluded(&self, app_id: &str) -> bool {
        self.excluded_apps.iter().any(|e| e == app_id)
    }

    /// Get the effective dictation mode for a given app.
    pub fn effective_dictation_mode(&self, app_id: Option<&str>) -> DictationMode {
        if let Some(id) = app_id {
            if let Some(ctx) = self.app_contexts.get(id) {
                if let Some(mode) = ctx.mode {
                    return mode;
                }
            }
        }
        self.dictation_mode
    }
}

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
