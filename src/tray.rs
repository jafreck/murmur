/// System tray UI.
///
/// This module uses `tray-icon` and `muda` to provide a cross-platform
/// system tray icon with a context menu. On macOS this appears in the
/// menu bar; on Windows in the taskbar; on Linux via AppIndicator.
///
/// TODO: Implement full tray UI with animated icons and state management.
/// For now, the app runs as a headless CLI daemon.

use anyhow::Result;

pub enum TrayState {
    Idle,
    Recording,
    Transcribing,
    Downloading,
    Error,
}

/// Placeholder for the system tray controller.
/// This will be implemented with `tray-icon` + `muda` crates.
pub struct TrayController {
    _state: TrayState,
}

impl TrayController {
    pub fn new() -> Result<Self> {
        // TODO: Initialize tray-icon and muda menu
        Ok(Self {
            _state: TrayState::Idle,
        })
    }

    pub fn set_state(&mut self, state: TrayState) {
        self._state = state;
        // TODO: Update tray icon based on state
    }
}
