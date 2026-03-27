use tauri::{App, Manager};

use crate::commands::AppState;
use std::sync::Mutex;

/// Configure the overlay window and register managed state.
pub fn configure_overlay(app: &mut App) -> Result<(), Box<dyn std::error::Error>> {
    let config = murmur_core::config::Config::load();

    // Register application state so Tauri commands can access it.
    app.manage(AppState {
        session: Mutex::new(None),
        stealth_enabled: Mutex::new(config.stealth_mode),
    });

    // The overlay window is declared in tauri.conf.json with:
    //   transparent: true, decorations: false, alwaysOnTop: true
    // Retrieve it here for any additional runtime configuration.
    if let Some(window) = app.get_webview_window("overlay") {
        // Ensure the overlay stays on top even when other apps are focused.
        window.set_always_on_top(true)?;

        if config.stealth_mode {
            apply_stealth_mode(&window);
        }
    }

    Ok(())
}

/// Apply platform-specific stealth mode to hide the window from screen
/// capture, screen sharing, and screenshots.
#[allow(unused_variables)]
pub fn apply_stealth_mode(window: &tauri::WebviewWindow) {
    #[cfg(target_os = "macos")]
    {
        apply_stealth_macos(window);
    }

    #[cfg(not(target_os = "macos"))]
    {
        log::warn!("stealth mode is not yet implemented on this platform");
    }
}

/// Remove stealth mode so the window is visible to screen capture again.
#[allow(unused_variables)]
pub fn remove_stealth_mode(window: &tauri::WebviewWindow) {
    #[cfg(target_os = "macos")]
    {
        remove_stealth_macos(window);
    }

    #[cfg(not(target_os = "macos"))]
    {
        log::warn!("stealth mode is not yet implemented on this platform");
    }
}

// ── macOS implementation ─────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
fn apply_stealth_macos(window: &tauri::WebviewWindow) {
    use objc2::rc::Retained;
    use objc2_app_kit::NSWindow;
    use objc2_app_kit::NSWindowSharingType;
    use raw_window_handle::HasWindowHandle;

    let handle = match window.window_handle() {
        Ok(h) => h,
        Err(e) => {
            log::warn!("could not get raw window handle: {e}");
            return;
        }
    };

    if let raw_window_handle::RawWindowHandle::AppKit(appkit) = handle.as_raw() {
        unsafe {
            let ns_view = appkit.ns_view.as_ptr() as *mut objc2::runtime::AnyObject;
            let ns_window: Retained<NSWindow> = objc2::msg_send![ns_view, window];
            ns_window.setSharingType(NSWindowSharingType::None);
            log::info!("stealth mode enabled — window hidden from screen capture");
        }
    }
}

#[cfg(target_os = "macos")]
fn remove_stealth_macos(window: &tauri::WebviewWindow) {
    use objc2::rc::Retained;
    use objc2_app_kit::NSWindow;
    use objc2_app_kit::NSWindowSharingType;
    use raw_window_handle::HasWindowHandle;

    let handle = match window.window_handle() {
        Ok(h) => h,
        Err(e) => {
            log::warn!("could not get raw window handle: {e}");
            return;
        }
    };

    if let raw_window_handle::RawWindowHandle::AppKit(appkit) = handle.as_raw() {
        unsafe {
            let ns_view = appkit.ns_view.as_ptr() as *mut objc2::runtime::AnyObject;
            let ns_window: Retained<NSWindow> = objc2::msg_send![ns_view, window];
            ns_window.setSharingType(NSWindowSharingType::ReadOnly);
            log::info!("stealth mode disabled — window visible to screen capture");
        }
    }
}
