use std::path::PathBuf;

use super::schema::Config;
use crate::error::ConfigError;

impl Config {
    pub fn dir() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("murmur")
    }

    pub fn file_path() -> PathBuf {
        Self::dir().join("config.json")
    }

    /// Load config from the default path, falling back to defaults on any error.
    pub fn load() -> Self {
        Self::load_from(&Self::file_path()).unwrap_or_else(|_| {
            let config = Self::default();
            if let Err(e) = config.save_to(&Self::file_path()) {
                log::warn!("Failed to save default config: {e}");
            }
            config
        })
    }

    /// Load and parse config from a specific path.
    ///
    /// Returns [`ConfigError::ParseFailed`] if the file cannot be read or
    /// contains invalid JSON.
    pub fn load_from(path: &std::path::Path) -> Result<Config, ConfigError> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| ConfigError::ParseFailed(format!("{}: {e}", path.display())))?;
        serde_json::from_str::<Config>(&contents)
            .map_err(|e| ConfigError::ParseFailed(format!("{}: {e}", path.display())))
    }

    /// Best-effort parse: returns defaults on invalid JSON (used by `load()`).
    pub fn parse(contents: &str, source: &std::path::Path) -> Self {
        match serde_json::from_str::<Config>(contents) {
            Ok(config) => config,
            Err(e) => {
                log::warn!("Unable to parse {}: {e}", source.display());
                Self::default()
            }
        }
    }

    pub fn save(&self) -> anyhow::Result<()> {
        self.save_to(&Self::file_path())?;
        Ok(())
    }

    pub fn save_to(&self, path: &std::path::Path) -> Result<(), ConfigError> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)
                .map_err(|e| ConfigError::SaveFailed(format!("{}: {e}", path.display())))?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| ConfigError::SaveFailed(e.to_string()))?;
        std::fs::write(path, json)
            .map_err(|e| ConfigError::SaveFailed(format!("{}: {e}", path.display())))?;
        Ok(())
    }
}
