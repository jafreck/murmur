//! System tray UI.
//!
//! Uses `tray-icon` (which re-exports `muda` as `tray_icon::menu`) to provide
//! a cross-platform system tray icon with a context menu. On macOS this appears
//! in the menu bar; on Windows in the taskbar; on Linux via AppIndicator.

use anyhow::Result;
use tray_icon::menu::{Menu, MenuEvent, MenuItem, MenuId, PredefinedMenuItem};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};

/// App states the tray can display.
#[derive(Debug, Clone, PartialEq)]
pub enum TrayState {
    Idle,
    Recording,
    Transcribing,
    #[allow(dead_code)]
    Downloading,
    Error,
}

/// Actions the tray menu can trigger.
#[derive(Debug, Clone)]
pub enum TrayAction {
    CopyLastDictation,
    Quit,
}

/// Manages the system tray icon and context menu.
pub struct TrayController {
    tray: TrayIcon,
    pub state: TrayState,
    copy_last_id: MenuId,
    quit_id: MenuId,
    idle_icon: Icon,
    recording_icon: Icon,
    transcribing_icon: Icon,
}

impl TrayController {
    /// Create the tray icon and menu. Must be called on the main thread.
    pub fn new() -> Result<Self> {
        let status_item = MenuItem::new("open-bark: Idle", false, None);
        let separator = PredefinedMenuItem::separator();
        let copy_last = MenuItem::new("Copy Last Dictation", true, None);
        let quit = MenuItem::new("Quit", true, None);

        let copy_last_id = copy_last.id().clone();
        let quit_id = quit.id().clone();

        let menu = Menu::new();
        menu.append(&status_item)?;
        menu.append(&separator)?;
        menu.append(&copy_last)?;
        menu.append(&quit)?;

        let idle_icon = make_solid_icon(100, 150, 255, 200)?; // blue
        let recording_icon = make_solid_icon(255, 60, 60, 230)?; // red
        let transcribing_icon = make_solid_icon(255, 200, 0, 220)?; // yellow

        let tray = TrayIconBuilder::new()
            .with_icon(idle_icon.clone())
            .with_tooltip("open-bark — Idle")
            .with_menu(Box::new(menu))
            .with_menu_on_left_click(true)
            .build()?;

        Ok(Self {
            tray,
            state: TrayState::Idle,
            copy_last_id,
            quit_id,
            idle_icon,
            recording_icon,
            transcribing_icon,
        })
    }

    /// Update the tray icon and tooltip to reflect the current state.
    pub fn set_state(&mut self, state: TrayState) {
        let (icon, tooltip) = match &state {
            TrayState::Idle => (&self.idle_icon, "open-bark — Idle"),
            TrayState::Recording => (&self.recording_icon, "open-bark — Recording..."),
            TrayState::Transcribing => {
                (&self.transcribing_icon, "open-bark — Transcribing...")
            }
            TrayState::Downloading => {
                (&self.transcribing_icon, "open-bark — Downloading model...")
            }
            TrayState::Error => (&self.recording_icon, "open-bark — Error"),
        };

        let _ = self.tray.set_icon(Some(icon.clone()));
        let _ = self.tray.set_tooltip(Some(tooltip));
        self.state = state;
    }

    /// Check if a menu event corresponds to a known action.
    pub fn match_menu_event(&self, event: &MenuEvent) -> Option<TrayAction> {
        if event.id() == &self.quit_id {
            Some(TrayAction::Quit)
        } else if event.id() == &self.copy_last_id {
            Some(TrayAction::CopyLastDictation)
        } else {
            None
        }
    }
}

/// Generate a simple solid-color 32×32 RGBA icon.
fn make_solid_icon(r: u8, g: u8, b: u8, a: u8) -> Result<Icon> {
    let size = 32u32;
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);
    for _ in 0..(size * size) {
        rgba.extend_from_slice(&[r, g, b, a]);
    }
    Icon::from_rgba(rgba, size, size).map_err(|e| anyhow::anyhow!("Icon error: {e}"))
}
