use tauri::{App, Manager};

use crate::commands::AppState;
use std::sync::Mutex;

/// Configure the overlay window and register managed state.
pub fn configure_overlay(app: &mut App) -> Result<(), Box<dyn std::error::Error>> {
    // Register application state so Tauri commands can access it.
    app.manage(AppState {
        session: Mutex::new(None),
    });

    // The overlay window is declared in tauri.conf.json with:
    //   transparent: true, decorations: false, alwaysOnTop: true
    // Retrieve it here for any additional runtime configuration.
    if let Some(window) = app.get_webview_window("overlay") {
        // Ensure the overlay stays on top even when other apps are focused.
        window.set_always_on_top(true)?;

        // TODO (Phase 2): implement stealth mode — hide from screen capture,
        // taskbar, and alt-tab via platform-specific APIs.
    }

    Ok(())
}
