use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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

pub const SUPPORTED_MODELS: &[&str] = &[
    "tiny.en", "tiny",
    "base.en", "base",
    "small.en", "small",
    "medium.en", "medium",
    "large-v3-turbo", "large",
];

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
    pub language: String,
    #[serde(default)]
    pub spoken_punctuation: bool,
    #[serde(default)]
    pub max_recordings: u32,
    #[serde(default)]
    pub mode: InputMode,
    #[serde(default)]
    pub streaming: bool,
    #[serde(default)]
    pub translate_to_english: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hotkey: default_hotkey().to_string(),
            model_size: "base.en".to_string(),
            language: "en".to_string(),
            spoken_punctuation: false,
            max_recordings: 0,
            mode: InputMode::PushToTalk,
            streaming: false,
            translate_to_english: false,
        }
    }
}

fn default_hotkey() -> &'static str {
    #[cfg(target_os = "macos")]
    { "globe" }
    #[cfg(not(target_os = "macos"))]
    { "ctrl+shift+space" }
}

impl Config {
    pub fn dir() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("open-bark")
    }

    pub fn file_path() -> PathBuf {
        Self::dir().join("config.json")
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
        match serde_json::from_str(contents) {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = Config::default();
        assert_eq!(cfg.model_size, "base.en");
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
            max_recordings: 10,
            mode: InputMode::OpenMic,
            streaming: true,
            translate_to_english: true,
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
    }

    #[test]
    fn test_config_dir_and_file_path() {
        let dir = Config::dir();
        assert!(dir.to_string_lossy().contains("open-bark"));
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
            max_recordings: 5,
            mode: InputMode::OpenMic,
            streaming: false,
            translate_to_english: false,
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
    }

    #[test]
    fn test_load_from_nonexistent_creates_default() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("nonexistent.json");
        let loaded = Config::load_from(&path);
        assert_eq!(loaded.model_size, "base.en");
        // Should have created the default config file
        assert!(path.exists());
    }

    #[test]
    fn test_parse_invalid_json_returns_default() {
        let path = std::path::Path::new("/tmp/test_invalid.json");
        let cfg = Config::parse("not valid json", path);
        assert_eq!(cfg.model_size, "base.en");
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
}
