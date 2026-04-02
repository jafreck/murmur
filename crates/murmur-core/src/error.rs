use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("config parse failed: {0}")]
    ParseFailed(String),
    #[error("config save failed: {0}")]
    SaveFailed(String),
}
