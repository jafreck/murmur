use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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
    pub toggle_mode: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hotkey: default_hotkey().to_string(),
            model_size: "base.en".to_string(),
            language: "en".to_string(),
            spoken_punctuation: false,
            max_recordings: 0,
            toggle_mode: false,
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
        let path = Self::file_path();
        match std::fs::read_to_string(&path) {
            Ok(contents) => match serde_json::from_str(&contents) {
                Ok(config) => config,
                Err(e) => {
                    eprintln!("Warning: unable to parse {}: {e}", path.display());
                    Self::default()
                }
            },
            Err(_) => {
                let config = Self::default();
                let _ = config.save();
                config
            }
        }
    }

    pub fn save(&self) -> Result<()> {
        let dir = Self::dir();
        std::fs::create_dir_all(&dir)?;
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(Self::file_path(), json)?;
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
        assert!(!cfg.toggle_mode);
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
            toggle_mode: true,
        };

        let json = serde_json::to_string(&cfg).unwrap();
        let parsed: Config = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.hotkey, "f9");
        assert_eq!(parsed.model_size, "small.en");
        assert_eq!(parsed.language, "fr");
        assert!(parsed.spoken_punctuation);
        assert_eq!(parsed.max_recordings, 10);
        assert!(parsed.toggle_mode);
    }
}
