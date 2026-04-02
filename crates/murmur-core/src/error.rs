use thiserror::Error;

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("audio device not found: {0}")]
    DeviceNotFound(String),
    #[error("audio stream failed: {0}")]
    StreamFailed(String),
    #[error("recording failed: {0}")]
    RecordingFailed(String),
}

#[derive(Debug, Error)]
pub enum TranscriptionError {
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("model download failed: {0}")]
    DownloadFailed(String),
    #[error("inference failed: {0}")]
    InferenceFailed(String),
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("config parse failed: {0}")]
    ParseFailed(String),
    #[error("config save failed: {0}")]
    SaveFailed(String),
}

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("LLM provider error: {0}")]
    ProviderError(String),
    #[error("LLM connection failed: {0}")]
    ConnectionFailed(String),
}

#[derive(Debug, Error)]
pub enum UpdateError {
    #[error("update check failed: {0}")]
    CheckFailed(String),
    #[error("update download failed: {0}")]
    DownloadFailed(String),
}
