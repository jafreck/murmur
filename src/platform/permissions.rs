//! Platform-specific permission checks.

#[cfg(target_os = "macos")]
pub fn check_accessibility() -> bool {
    // On macOS, rdev uses CGEventTap which requires Accessibility permission.
    // The system will prompt automatically when rdev::listen is called.
    // We just log a hint for the user.
    log::info!("macOS: Accessibility permission is required for hotkey detection.");
    log::info!("If prompted, grant access in System Settings → Privacy & Security → Accessibility.");
    true
}

#[cfg(target_os = "windows")]
pub fn check_accessibility() -> bool {
    // Windows does not require special permissions for keyboard hooks.
    true
}

#[cfg(target_os = "linux")]
pub fn check_accessibility() -> bool {
    // On Linux/Wayland, rdev may need /dev/input access.
    // The user should be in the 'input' group.
    log::info!("Linux: If hotkeys don't work, ensure your user is in the 'input' group:");
    log::info!("  sudo usermod -aG input $USER");
    true
}

pub fn check_microphone() {
    // On all platforms, cpal will fail with an error if the microphone
    // is not accessible. We handle that in AudioRecorder::start().
    log::info!("Microphone access will be requested when recording starts.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_accessibility_returns_true() {
        assert!(check_accessibility());
    }

    #[test]
    fn test_check_microphone_does_not_panic() {
        check_microphone();
    }
}
