pub mod app;
pub mod context;
pub mod input;
pub mod notes;
pub mod platform;
pub mod ui;

// Re-export core crate modules so existing `murmur::audio`, `murmur::config`,
// and `murmur::transcription` paths keep working.
pub use murmur_core::{audio, config, error, transcription};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
