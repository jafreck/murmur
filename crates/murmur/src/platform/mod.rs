pub mod permissions;

#[cfg(target_os = "macos")]
pub(crate) mod ax;

pub use permissions::{check_accessibility, check_microphone};

#[cfg(target_os = "macos")]
pub use permissions::{open_accessibility_settings, re_exec, wait_for_accessibility};
