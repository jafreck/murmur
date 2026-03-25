//! System tray UI.
//!
//! Uses `tray-icon` (which re-exports `muda` as `tray_icon::menu`) to provide
//! a cross-platform system tray icon with a context menu. On macOS this appears
//! in the menu bar; on Windows in the taskbar; on Linux via AppIndicator.

use anyhow::Result;
use tray_icon::menu::{
    CheckMenuItem, Menu, MenuEvent, MenuItem, MenuId, PredefinedMenuItem, Submenu,
};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};

use crate::config::{self, Config};

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
    SetModel(String),
    SetLanguage(String),
    ToggleSpokenPunctuation,
    ToggleToggleMode,
    ToggleTranslate,
    Quit,
}

/// A model radio item: the MenuId and the model size string.
struct ModelEntry {
    id: MenuId,
    size: String,
    item: CheckMenuItem,
}

/// A language radio item.
struct LanguageEntry {
    id: MenuId,
    code: String,
    item: CheckMenuItem,
}

/// Manages the system tray icon and context menu.
pub struct TrayController {
    tray: TrayIcon,
    pub state: TrayState,

    // Action IDs
    copy_last_id: MenuId,
    spoken_punct_id: MenuId,
    toggle_mode_id: MenuId,
    translate_id: MenuId,
    quit_id: MenuId,

    // Radio groups
    model_entries: Vec<ModelEntry>,
    language_entries: Vec<LanguageEntry>,

    // Check items (stored so CheckMenuItem is not dropped)
    #[allow(dead_code)]
    spoken_punct_item: CheckMenuItem,
    #[allow(dead_code)]
    toggle_mode_item: CheckMenuItem,
    #[allow(dead_code)]
    translate_item: CheckMenuItem,

    // Status display
    status_item: MenuItem,
    hotkey_item: MenuItem,

    // Icons
    idle_icon: Icon,
    recording_icon: Icon,
    transcribing_icon: Icon,
}

/// Top languages shown in the menu (rest are omitted to keep it manageable).
const TOP_LANGUAGES: &[&str] = &[
    "auto", "en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja", "ko", "ar", "hi", "pl",
    "sv", "tr", "uk", "vi", "th",
];

impl TrayController {
    /// Create the tray icon and menu. Must be called on the main thread.
    pub fn new(config: &Config) -> Result<Self> {
        // ── Status ──────────────────────────────────────────────────────
        let status_item = MenuItem::new("open-bark: Idle", false, None);
        let hotkey_item = MenuItem::new(format!("Hotkey: {}", config.hotkey), false, None);

        // ── Copy last ───────────────────────────────────────────────────
        let copy_last = MenuItem::new("Copy Last Dictation", true, None);
        let copy_last_id = copy_last.id().clone();

        // ── Model submenu (radio-style) ─────────────────────────────────
        let model_submenu = Submenu::new("Model", true);
        let mut model_entries = Vec::new();

        let display_models: &[&str] = &[
            "tiny.en",
            "base.en",
            "small.en",
            "medium.en",
            "large-v3-turbo",
            "large",
        ];

        for &size in display_models {
            let checked = size == config.model_size;
            let label = if checked {
                format!("● {size}")
            } else {
                format!("○ {size}")
            };
            let item = CheckMenuItem::new(label, true, checked, None);
            let id = item.id().clone();
            model_submenu.append(&item)?;
            model_entries.push(ModelEntry {
                id,
                size: size.to_string(),
                item,
            });
        }

        // ── Language submenu (radio-style) ──────────────────────────────
        let lang_submenu = Submenu::new("Language", true);
        let mut language_entries = Vec::new();

        for &code in TOP_LANGUAGES {
            let name = config::language_name(code).unwrap_or(code);
            let checked = code == config.language;
            let label = if checked {
                format!("● {name}")
            } else {
                format!("○ {name}")
            };
            let item = CheckMenuItem::new(label, true, checked, None);
            let id = item.id().clone();
            lang_submenu.append(&item)?;
            language_entries.push(LanguageEntry {
                id,
                code: code.to_string(),
                item,
            });
        }

        // ── Toggle options ──────────────────────────────────────────────
        let spoken_punct_item =
            CheckMenuItem::new("Spoken Punctuation", true, config.spoken_punctuation, None);
        let spoken_punct_id = spoken_punct_item.id().clone();

        let toggle_mode_item =
            CheckMenuItem::new("Toggle Mode", true, config.toggle_mode, None);
        let toggle_mode_id = toggle_mode_item.id().clone();

        let translate_item =
            CheckMenuItem::new("Translate to English", true, config.translate_to_english, None);
        let translate_id = translate_item.id().clone();

        // ── Quit ────────────────────────────────────────────────────────
        let quit = MenuItem::new("Quit", true, None);
        let quit_id = quit.id().clone();

        // ── Assemble menu ───────────────────────────────────────────────
        let menu = Menu::new();
        menu.append(&status_item)?;
        menu.append(&PredefinedMenuItem::separator())?;
        menu.append(&copy_last)?;
        menu.append(&PredefinedMenuItem::separator())?;
        menu.append(&model_submenu)?;
        menu.append(&lang_submenu)?;
        menu.append(&hotkey_item)?;
        menu.append(&PredefinedMenuItem::separator())?;
        menu.append(&spoken_punct_item)?;
        menu.append(&toggle_mode_item)?;
        menu.append(&translate_item)?;
        menu.append(&PredefinedMenuItem::separator())?;
        menu.append(&quit)?;

        // ── Icons ───────────────────────────────────────────────────────
        let idle_icon = make_bark_icon(100, 150, 255, 200)?;
        let recording_icon = make_bark_icon(255, 60, 60, 230)?;
        let transcribing_icon = make_bark_icon(255, 200, 0, 220)?;

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
            spoken_punct_id,
            toggle_mode_id,
            translate_id,
            quit_id,
            model_entries,
            language_entries,
            spoken_punct_item,
            toggle_mode_item,
            translate_item,
            status_item,
            hotkey_item,
            idle_icon,
            recording_icon,
            transcribing_icon,
        })
    }

    /// Update the tray icon and tooltip to reflect the current state.
    pub fn set_state(&mut self, state: TrayState) {
        let (icon, tooltip, label) = match &state {
            TrayState::Idle => (&self.idle_icon, "open-bark — Idle", "open-bark: Idle"),
            TrayState::Recording => (
                &self.recording_icon,
                "open-bark — Recording...",
                "open-bark: Recording...",
            ),
            TrayState::Transcribing => (
                &self.transcribing_icon,
                "open-bark — Transcribing...",
                "open-bark: Transcribing...",
            ),
            TrayState::Downloading => (
                &self.transcribing_icon,
                "open-bark — Downloading model...",
                "open-bark: Downloading...",
            ),
            TrayState::Error => (
                &self.recording_icon,
                "open-bark — Error",
                "open-bark: Error",
            ),
        };

        let _ = self.tray.set_icon(Some(icon.clone()));
        let _ = self.tray.set_tooltip(Some(tooltip));
        self.status_item.set_text(label);
        self.state = state;
    }

    /// Update the model radio selection in the menu after a model change.
    pub fn set_model(&mut self, new_model: &str) {
        for entry in &self.model_entries {
            let selected = entry.size == new_model;
            entry.item.set_checked(selected);
            let label = if selected {
                format!("● {}", entry.size)
            } else {
                format!("○ {}", entry.size)
            };
            entry.item.set_text(label);
        }
    }

    /// Update the language radio selection in the menu after a language change.
    pub fn set_language(&mut self, new_code: &str) {
        for entry in &self.language_entries {
            let selected = entry.code == new_code;
            entry.item.set_checked(selected);
            let name = config::language_name(&entry.code).unwrap_or(&entry.code);
            let label = if selected {
                format!("● {name}")
            } else {
                format!("○ {name}")
            };
            entry.item.set_text(label);
        }
    }

    /// Update the hotkey display in the menu.
    #[allow(dead_code)]
    pub fn set_hotkey(&mut self, hotkey: &str) {
        self.hotkey_item.set_text(format!("Hotkey: {hotkey}"));
    }

    /// Check if a menu event corresponds to a known action.
    pub fn match_menu_event(&self, event: &MenuEvent) -> Option<TrayAction> {
        let id = event.id();

        if id == &self.quit_id {
            return Some(TrayAction::Quit);
        }
        if id == &self.copy_last_id {
            return Some(TrayAction::CopyLastDictation);
        }
        if id == &self.spoken_punct_id {
            return Some(TrayAction::ToggleSpokenPunctuation);
        }
        if id == &self.toggle_mode_id {
            return Some(TrayAction::ToggleToggleMode);
        }
        if id == &self.translate_id {
            return Some(TrayAction::ToggleTranslate);
        }

        for entry in &self.model_entries {
            if id == &entry.id {
                return Some(TrayAction::SetModel(entry.size.clone()));
            }
        }

        for entry in &self.language_entries {
            if id == &entry.id {
                return Some(TrayAction::SetLanguage(entry.code.clone()));
            }
        }

        None
    }
}

/// The bark icon PNG, embedded at compile time.
const BARK_PNG: &[u8] = include_bytes!("../assets/icons/bark.png");

/// Decode [`BARK_PNG`] and tint it to the given base colour.
///
/// Source pixels are treated as a grayscale luminance mask: brighter pixels map
/// closer to the supplied (r, g, b) colour while darker pixels stay dark.
/// Fully transparent pixels remain transparent.
fn make_bark_icon(r: u8, g: u8, b: u8, a: u8) -> Result<Icon> {
    let cursor = std::io::Cursor::new(BARK_PNG);
    let decoder = png::Decoder::new(cursor);
    let mut reader = decoder.read_info().map_err(|e| anyhow::anyhow!("PNG decode: {e}"))?;
    let mut buf = vec![0u8; reader.output_buffer_size().expect("PNG info missing")];
    let info = reader.next_frame(&mut buf).map_err(|e| anyhow::anyhow!("PNG frame: {e}"))?;
    let width = info.width;
    let height = info.height;

    let src = &buf[..info.buffer_size()];
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);

    let stride = match info.color_type {
        png::ColorType::Rgba => 4,
        png::ColorType::Rgb => 3,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Grayscale => 1,
        other => anyhow::bail!("Unsupported PNG colour type: {other:?}"),
    };

    for pixel in src.chunks_exact(stride) {
        let (lum, pa) = match stride {
            4 => (pixel[0] as u16, pixel[3]),
            3 => (pixel[0] as u16, 255),
            2 => (pixel[0] as u16, pixel[1]),
            1 => (pixel[0] as u16, 255),
            _ => unreachable!(),
        };

        if pa == 0 {
            rgba.extend_from_slice(&[0, 0, 0, 0]);
        } else {
            // Tint: use source luminance to scale the target colour.
            let tr = ((r as u16 * lum) / 255u16).min(255) as u8;
            let tg = ((g as u16 * lum) / 255u16).min(255) as u8;
            let tb = ((b as u16 * lum) / 255u16).min(255) as u8;
            let ta = ((a as u16 * pa as u16) / 255u16).min(255) as u8;
            rgba.extend_from_slice(&[tr, tg, tb, ta]);
        }
    }

    Icon::from_rgba(rgba, width, height).map_err(|e| anyhow::anyhow!("Icon error: {e}"))
}
