pub mod app;
pub mod audio;
pub mod config;
pub mod hotkey;
pub mod inserter;
pub mod keycodes;
pub mod model;
pub mod permissions;
pub mod postprocess;
pub mod recordings;
pub mod streaming;
pub mod transcriber;
pub mod tray;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
