//! Platform-specific permission checks.

#[cfg(target_os = "macos")]
pub fn check_accessibility() -> bool {
    use std::ffi::c_void;

    #[link(name = "ApplicationServices", kind = "framework")]
    extern "C" {
        fn AXIsProcessTrustedWithOptions(options: *const c_void) -> bool;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        static kCFBooleanTrue: *const c_void;
        fn CFDictionaryCreate(
            allocator: *const c_void,
            keys: *const *const c_void,
            values: *const *const c_void,
            num_values: isize,
            key_callbacks: *const c_void,
            value_callbacks: *const c_void,
        ) -> *const c_void;
        fn CFRelease(cf: *const c_void);
    }

    extern "C" {
        static kAXTrustedCheckOptionPrompt: *const c_void;
    }

    // Skip the system prompt in test builds to avoid opening System Settings in CI.
    let prompt = !cfg!(test);

    let trusted = unsafe {
        if prompt {
            let keys: [*const c_void; 1] = [kAXTrustedCheckOptionPrompt];
            let values: [*const c_void; 1] = [kCFBooleanTrue];
            let options = CFDictionaryCreate(
                std::ptr::null(),
                keys.as_ptr(),
                values.as_ptr(),
                1,
                std::ptr::null(),
                std::ptr::null(),
            );
            let result = AXIsProcessTrustedWithOptions(options);
            if !options.is_null() {
                CFRelease(options);
            }
            result
        } else {
            // No prompt — just check
            AXIsProcessTrustedWithOptions(std::ptr::null())
        }
    };

    if trusted {
        log::info!("macOS: Accessibility permission granted.");
    } else {
        log::warn!("macOS: Accessibility permission NOT granted.");
        log::warn!("Grant access in System Settings → Privacy & Security → Accessibility.");
        log::warn!(
            "Also grant Input Monitoring in System Settings → Privacy & Security → Input Monitoring."
        );
        log::warn!("Hotkey detection will not work until permissions are granted.");
        log::warn!("Tip: click +, press ⌘⇧G, type /usr/local/bin, select murmur.");
    }

    trusted
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
    fn test_check_accessibility_does_not_panic() {
        // May return false in CI/test environments without Accessibility permission.
        let _trusted = check_accessibility();
    }

    #[test]
    fn test_check_microphone_does_not_panic() {
        check_microphone();
    }
}
