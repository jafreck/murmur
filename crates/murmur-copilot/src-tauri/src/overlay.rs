use tauri::{App, Manager};

/// Configure the overlay window.
pub fn configure_overlay(app: &mut App) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(window) = app.get_webview_window("overlay") {
        window.set_always_on_top(true)?;
        // Always hide from screen capture — no reason to expose the overlay
        // in recordings or screen shares.
        apply_stealth_mode(&window);
    }

    Ok(())
}

/// Hide the overlay window from screen capture, screen sharing, and screenshots.
#[allow(unused_variables)]
fn apply_stealth_mode(window: &tauri::WebviewWindow) {
    #[cfg(target_os = "macos")]
    {
        apply_stealth_macos(window);
    }

    #[cfg(not(target_os = "macos"))]
    {
        log::warn!("screen-capture hiding is not yet implemented on this platform");
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
            log::info!("window hidden from screen capture");
        }
    }
}
