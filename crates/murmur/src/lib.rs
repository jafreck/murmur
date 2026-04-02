pub mod app;
pub mod context;
pub mod input;
pub mod notes;
pub mod platform;
pub mod ui;

// Re-export config publicly (used by the binary CLI).
pub use murmur_core::config;

// Internal-only re-exports so `crate::audio` / `crate::transcription` paths
// continue to work inside the app crate without leaking to external consumers.
pub(crate) use murmur_core::audio;
pub(crate) use murmur_core::transcription;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
