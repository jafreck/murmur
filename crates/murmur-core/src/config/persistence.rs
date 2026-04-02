use anyhow::Result;
use std::path::PathBuf;

use super::schema::Config;

impl Config {
    pub fn dir() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("murmur")
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
}
