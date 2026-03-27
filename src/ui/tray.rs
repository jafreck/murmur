//! System tray UI.

use anyhow::Result;
use tray_icon::menu::{
    CheckMenuItem, Menu, MenuEvent, MenuId, MenuItem, PredefinedMenuItem, Submenu,
};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};

use crate::config::{self, Config, InputMode};

/// App states the tray can display.
#[derive(Debug, Clone, PartialEq)]
pub enum TrayState {
    /// Model is loading in the background; not ready for dictation yet.
    Loading,
    Idle,
    Recording,
    Transcribing,
    #[allow(dead_code)]
    Downloading,
    Error,
}

/// Actions the tray menu can trigger.
#[derive(Debug, Clone, PartialEq)]
pub enum TrayAction {
    CopyLastDictation,
    SetModel(String),
    SetLanguage(String),
    ToggleSpokenPunctuation,
    ToggleFillerWordRemoval,
    SetMode(InputMode),
    ToggleStreaming,
    ToggleTranslate,
    ToggleNoiseSuppression,
    SetHotkey,
    OpenConfig,
    ReloadConfig,
    Quit,
}

/// Top languages shown in the menu.
pub const TOP_LANGUAGES: &[&str] = &[
    "auto", "en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja", "ko", "ar", "hi", "pl",
    "sv", "tr", "uk", "vi", "th",
];

/// Compute the tooltip and status label for a given TrayState.
pub fn state_display(state: &TrayState) -> (&'static str, &'static str) {
    match state {
        TrayState::Loading => ("murmur — Loading model...", "murmur: Loading model..."),
        TrayState::Idle => ("murmur — Idle", "murmur: Idle"),
        TrayState::Recording => ("murmur — Recording...", "murmur: Recording..."),
        TrayState::Transcribing => ("murmur — Transcribing...", "murmur: Transcribing..."),
        TrayState::Downloading => ("murmur — Downloading model...", "murmur: Downloading..."),
        TrayState::Error => ("murmur — Error", "murmur: Error"),
    }
}

/// Build a radio-style label.
pub fn radio_label(name: &str, selected: bool) -> String {
    if selected {
        format!("● {name}")
    } else {
        format!("○ {name}")
    }
}

/// Grouped menu item IDs for constructing a MenuActionMap.
pub struct MenuActionIds {
    pub quit: MenuId,
    pub copy_last: MenuId,
    pub open_config: MenuId,
    pub reload_config: MenuId,
    pub spoken_punct: MenuId,
    pub filler_removal: MenuId,
    pub streaming: MenuId,
    pub translate: MenuId,
    pub noise_suppression: MenuId,
    pub set_hotkey: MenuId,
}

/// Menu event matching engine. Maps MenuId → TrayAction.
pub struct MenuActionMap {
    quit_id: MenuId,
    copy_last_id: MenuId,
    open_config_id: MenuId,
    reload_config_id: MenuId,
    spoken_punct_id: MenuId,
    filler_removal_id: MenuId,
    streaming_id: MenuId,
    translate_id: MenuId,
    noise_suppression_id: MenuId,
    set_hotkey_id: MenuId,
    model_ids: Vec<(MenuId, String)>,
    language_ids: Vec<(MenuId, String)>,
    mode_ids: Vec<(MenuId, InputMode)>,
}

impl MenuActionMap {
    pub fn new(
        ids: MenuActionIds,
        model_ids: Vec<(MenuId, String)>,
        language_ids: Vec<(MenuId, String)>,
        mode_ids: Vec<(MenuId, InputMode)>,
    ) -> Self {
        Self {
            quit_id: ids.quit,
            copy_last_id: ids.copy_last,
            open_config_id: ids.open_config,
            reload_config_id: ids.reload_config,
            spoken_punct_id: ids.spoken_punct,
            filler_removal_id: ids.filler_removal,
            streaming_id: ids.streaming,
            translate_id: ids.translate,
            noise_suppression_id: ids.noise_suppression,
            set_hotkey_id: ids.set_hotkey,
            model_ids,
            language_ids,
            mode_ids,
        }
    }

    pub fn match_event(&self, event_id: &MenuId) -> Option<TrayAction> {
        if event_id == &self.quit_id {
            return Some(TrayAction::Quit);
        }
        if event_id == &self.copy_last_id {
            return Some(TrayAction::CopyLastDictation);
        }
        if event_id == &self.open_config_id {
            return Some(TrayAction::OpenConfig);
        }
        if event_id == &self.reload_config_id {
            return Some(TrayAction::ReloadConfig);
        }
        if event_id == &self.spoken_punct_id {
            return Some(TrayAction::ToggleSpokenPunctuation);
        }
        if event_id == &self.filler_removal_id {
            return Some(TrayAction::ToggleFillerWordRemoval);
        }
        if event_id == &self.streaming_id {
            return Some(TrayAction::ToggleStreaming);
        }
        if event_id == &self.translate_id {
            return Some(TrayAction::ToggleTranslate);
        }
        if event_id == &self.noise_suppression_id {
            return Some(TrayAction::ToggleNoiseSuppression);
        }
        if event_id == &self.set_hotkey_id {
            return Some(TrayAction::SetHotkey);
        }

        for (id, size) in &self.model_ids {
            if event_id == id {
                return Some(TrayAction::SetModel(size.clone()));
            }
        }

        for (id, code) in &self.language_ids {
            if event_id == id {
                return Some(TrayAction::SetLanguage(code.clone()));
            }
        }

        for (id, mode) in &self.mode_ids {
            if event_id == id {
                return Some(TrayAction::SetMode(mode.clone()));
            }
        }

        None
    }
}

/// A radio-style menu entry that tracks its value alongside a CheckMenuItem.
struct RadioEntry<T> {
    value: T,
    item: CheckMenuItem,
}

impl<T> RadioEntry<T> {
    fn new(value: T, label: &str, checked: bool) -> Self {
        let item = CheckMenuItem::new(radio_label(label, checked), true, checked, None);
        Self { value, item }
    }
}

type RadioBuildResult<T> = Result<(Vec<RadioEntry<T>>, Vec<(MenuId, T)>)>;

/// Build a submenu of radio-style CheckMenuItems, returning the entries and
/// their (MenuId, value) pairs for the action map.
fn build_radio_submenu<T: Clone>(
    submenu: &Submenu,
    items: &[(&str, T)],
    is_selected: impl Fn(&T) -> bool,
) -> RadioBuildResult<T> {
    let mut entries = Vec::new();
    let mut ids = Vec::new();
    for (label, value) in items {
        let checked = is_selected(value);
        let entry = RadioEntry::new(value.clone(), label, checked);
        let id = entry.item.id().clone();
        submenu.append(&entry.item)?;
        ids.push((id, value.clone()));
        entries.push(entry);
    }
    Ok((entries, ids))
}

/// Update radio-style entries so exactly one is selected.
fn update_radio_entries<T: PartialEq>(
    entries: &[RadioEntry<T>],
    selected: &T,
    display_name: impl Fn(&T) -> String,
) {
    for entry in entries {
        let is_selected = entry.value == *selected;
        entry.item.set_checked(is_selected);
        entry
            .item
            .set_text(radio_label(&display_name(&entry.value), is_selected));
    }
}

/// Manages the system tray icon and context menu.
pub struct TrayController {
    tray: TrayIcon,
    pub state: TrayState,
    action_map: MenuActionMap,

    model_entries: Vec<RadioEntry<String>>,
    language_entries: Vec<RadioEntry<String>>,
    mode_entries: Vec<RadioEntry<InputMode>>,

    #[allow(dead_code)]
    spoken_punct_item: CheckMenuItem,
    #[allow(dead_code)]
    filler_removal_item: CheckMenuItem,
    #[allow(dead_code)]
    streaming_item: CheckMenuItem,
    #[allow(dead_code)]
    translate_item: CheckMenuItem,
    #[allow(dead_code)]
    noise_suppression_item: CheckMenuItem,

    status_item: MenuItem,
    hotkey_item: MenuItem,

    #[allow(dead_code)]
    idle_icon: Icon,
    #[allow(dead_code)]
    recording_icon: Icon,
    #[allow(dead_code)]
    transcribing_icon: Icon,
    #[allow(dead_code)]
    loading_icon: Icon,
}

impl TrayController {
    pub fn new(config: &Config) -> Result<Self> {
        let status_item = MenuItem::new("murmur: Idle", false, None);
        let hotkey_item = MenuItem::new(format!("Hotkey: {}", config.hotkey), false, None);

        let set_hotkey = MenuItem::new("Set Hotkey…", true, None);
        let set_hotkey_id = set_hotkey.id().clone();

        let copy_last = MenuItem::new("Copy Last Dictation", true, None);
        let copy_last_id = copy_last.id().clone();

        let model_submenu = Submenu::new("Model", true);
        let model_items: Vec<(&str, String)> = crate::config::SUPPORTED_MODELS
            .iter()
            .map(|&s| (s, s.to_string()))
            .collect();
        let (model_entries, model_ids) =
            build_radio_submenu(&model_submenu, &model_items, |s| s == &config.model_size)?;

        let lang_submenu = Submenu::new("Language", true);
        let lang_items: Vec<(&str, String)> = TOP_LANGUAGES
            .iter()
            .map(|&code| {
                let name = config::language_name(code).unwrap_or(code);
                (name, code.to_string())
            })
            .collect();
        let (language_entries, language_ids) =
            build_radio_submenu(&lang_submenu, &lang_items, |c| c == &config.language)?;

        let spoken_punct_item =
            CheckMenuItem::new("Spoken Punctuation", true, config.spoken_punctuation, None);
        let spoken_punct_id = spoken_punct_item.id().clone();

        let filler_removal_item = CheckMenuItem::new(
            "Filler Word Removal",
            true,
            config.filler_word_removal,
            None,
        );
        let filler_removal_id = filler_removal_item.id().clone();

        let mode_submenu = Submenu::new("Mode", true);
        let mode_items: Vec<(&str, InputMode)> = vec![
            ("Push to Talk", InputMode::PushToTalk),
            ("Open Mic", InputMode::OpenMic),
        ];
        let (mode_entries, mode_ids) =
            build_radio_submenu(&mode_submenu, &mode_items, |m| *m == config.mode)?;

        let streaming_item =
            CheckMenuItem::new("Live Streaming (Preview)", true, config.streaming, None);
        let streaming_id = streaming_item.id().clone();

        let translate_item = CheckMenuItem::new(
            "Translate to English",
            true,
            config.translate_to_english,
            None,
        );
        let translate_id = translate_item.id().clone();

        let noise_suppression_item =
            CheckMenuItem::new("Noise Suppression", true, config.noise_suppression, None);
        let noise_suppression_id = noise_suppression_item.id().clone();

        let open_config = MenuItem::new("Open Config…", true, None);
        let open_config_id = open_config.id().clone();
        let reload_config = MenuItem::new("Reload Config", true, None);
        let reload_config_id = reload_config.id().clone();

        let quit = MenuItem::new("Quit", true, None);
        let quit_id = quit.id().clone();

        let menu = Menu::new();
        menu.append(&status_item)?;
        menu.append(&PredefinedMenuItem::separator())?;
        menu.append(&copy_last)?;
        menu.append(&PredefinedMenuItem::separator())?;
        menu.append(&model_submenu)?;
        menu.append(&lang_submenu)?;
        menu.append(&hotkey_item)?;
        menu.append(&set_hotkey)?;
        menu.append(&PredefinedMenuItem::separator())?;
        menu.append(&spoken_punct_item)?;
        menu.append(&filler_removal_item)?;
        menu.append(&mode_submenu)?;
        menu.append(&streaming_item)?;
        menu.append(&translate_item)?;
        menu.append(&noise_suppression_item)?;
        menu.append(&PredefinedMenuItem::separator())?;
        menu.append(&open_config)?;
        menu.append(&reload_config)?;
        menu.append(&PredefinedMenuItem::separator())?;
        menu.append(&quit)?;

        let action_map = MenuActionMap::new(
            MenuActionIds {
                quit: quit_id,
                copy_last: copy_last_id,
                open_config: open_config_id,
                reload_config: reload_config_id,
                spoken_punct: spoken_punct_id,
                filler_removal: filler_removal_id,
                streaming: streaming_id,
                translate: translate_id,
                noise_suppression: noise_suppression_id,
                set_hotkey: set_hotkey_id,
            },
            model_ids,
            language_ids,
            mode_ids,
        );

        let colors = StateColors::for_current_appearance();
        let idle_icon = make_icon(colors.idle.0, colors.idle.1, colors.idle.2, colors.idle.3)?;
        let recording_icon = make_icon(
            colors.recording.0,
            colors.recording.1,
            colors.recording.2,
            colors.recording.3,
        )?;
        let transcribing_icon = make_icon(
            colors.transcribing.0,
            colors.transcribing.1,
            colors.transcribing.2,
            colors.transcribing.3,
        )?;
        let loading_icon = make_icon(
            colors.loading.0,
            colors.loading.1,
            colors.loading.2,
            colors.loading.3,
        )?;

        let tray = TrayIconBuilder::new()
            .with_icon(idle_icon.clone())
            .with_tooltip("murmur — Idle")
            .with_menu(Box::new(menu))
            .with_menu_on_left_click(true)
            .build()?;

        Ok(Self {
            tray,
            state: TrayState::Idle,
            action_map,
            model_entries,
            language_entries,
            mode_entries,
            spoken_punct_item,
            filler_removal_item,
            streaming_item,
            translate_item,
            noise_suppression_item,
            status_item,
            hotkey_item,
            idle_icon,
            recording_icon,
            transcribing_icon,
            loading_icon,
        })
    }

    pub fn set_state(&mut self, state: TrayState) {
        let (tooltip, label) = state_display(&state);

        // Re-check appearance on every state change so icons adapt when
        // the user switches between dark and light mode.
        let colors = StateColors::for_current_appearance();
        let color = match &state {
            TrayState::Loading => colors.loading,
            TrayState::Idle => colors.idle,
            TrayState::Recording | TrayState::Error => colors.recording,
            TrayState::Transcribing | TrayState::Downloading => colors.transcribing,
        };
        if let Ok(icon) = make_icon(color.0, color.1, color.2, color.3) {
            let _ = self.tray.set_icon(Some(icon));
        }

        let _ = self.tray.set_tooltip(Some(tooltip));
        self.status_item.set_text(label);
        self.state = state;
    }

    pub fn set_model(&mut self, new_model: &str) {
        let new_model = new_model.to_string();
        update_radio_entries(&self.model_entries, &new_model, |s| s.clone());
    }

    pub fn set_language(&mut self, new_code: &str) {
        let new_code = new_code.to_string();
        update_radio_entries(&self.language_entries, &new_code, |code| {
            config::language_name(code).unwrap_or(code).to_string()
        });
    }

    pub fn set_mode(&mut self, mode: &InputMode) {
        update_radio_entries(&self.mode_entries, mode, |m| m.to_string());
    }

    #[allow(dead_code)]
    pub fn set_hotkey(&mut self, hotkey: &str) {
        self.hotkey_item.set_text(format!("Hotkey: {hotkey}"));
    }

    pub fn set_status(&mut self, text: &str) {
        self.status_item.set_text(text);
    }

    pub fn match_menu_event(&self, event: &MenuEvent) -> Option<TrayAction> {
        self.action_map.match_event(event.id())
    }
}

/// The murmur icon PNG, embedded at compile time.
const ICON_PNG: &[u8] = include_bytes!("../../assets/icons/murmur.png");

fn make_icon(r: u8, g: u8, b: u8, a: u8) -> Result<Icon> {
    let (rgba, width, height) = tint_png_rgba(ICON_PNG, r, g, b, a)?;
    Icon::from_rgba(rgba, width, height).map_err(|e| anyhow::anyhow!("Icon error: {e}"))
}

/// RGBA colour for each tray state, tuned for dark and light menu bars.
struct StateColors {
    idle: (u8, u8, u8, u8),
    recording: (u8, u8, u8, u8),
    transcribing: (u8, u8, u8, u8),
    loading: (u8, u8, u8, u8),
}

impl StateColors {
    fn for_current_appearance() -> Self {
        if is_dark_mode() {
            // Light icons for dark menu bar
            Self {
                idle: (255, 255, 255, 200),
                recording: (255, 80, 80, 230),
                transcribing: (255, 210, 50, 220),
                loading: (180, 180, 180, 140),
            }
        } else {
            // Dark icons for light menu bar
            Self {
                idle: (30, 80, 200, 220),
                recording: (200, 30, 30, 230),
                transcribing: (180, 140, 0, 220),
                loading: (100, 100, 100, 160),
            }
        }
    }
}

/// Detect whether the macOS menu bar is using dark mode.
#[cfg(target_os = "macos")]
fn is_dark_mode() -> bool {
    use std::process::Command;
    Command::new("defaults")
        .args(["read", "-g", "AppleInterfaceStyle"])
        .output()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .eq_ignore_ascii_case("dark")
        })
        .unwrap_or(false)
}

/// Detect dark mode on Windows via the registry.
#[cfg(target_os = "windows")]
fn is_dark_mode() -> bool {
    use std::process::Command;
    // AppsUseLightTheme: 0 = dark, 1 = light
    Command::new("reg")
        .args([
            "query",
            r"HKCU\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
            "/v",
            "AppsUseLightTheme",
        ])
        .output()
        .map(|o| {
            let out = String::from_utf8_lossy(&o.stdout);
            out.contains("0x0")
        })
        .unwrap_or(true)
}

/// Detect dark mode on Linux via common desktop environment hints.
#[cfg(target_os = "linux")]
fn is_dark_mode() -> bool {
    use std::process::Command;
    // Try GNOME color-scheme first
    if let Ok(output) = Command::new("gsettings")
        .args(["get", "org.gnome.desktop.interface", "color-scheme"])
        .output()
    {
        let scheme = String::from_utf8_lossy(&output.stdout).to_lowercase();
        if scheme.contains("dark") {
            return true;
        }
        if scheme.contains("light") || scheme.contains("default") {
            return false;
        }
    }
    // Fall back to GTK theme name
    if let Ok(output) = Command::new("gsettings")
        .args(["get", "org.gnome.desktop.interface", "gtk-theme"])
        .output()
    {
        let theme = String::from_utf8_lossy(&output.stdout).to_lowercase();
        if theme.contains("dark") {
            return true;
        }
    }
    // Default to dark (bright icons) when detection fails
    true
}

/// Decode a PNG and tint its pixels.
pub fn tint_png_rgba(png_data: &[u8], r: u8, g: u8, b: u8, a: u8) -> Result<(Vec<u8>, u32, u32)> {
    let cursor = std::io::Cursor::new(png_data);
    let decoder = png::Decoder::new(cursor);
    let mut reader = decoder
        .read_info()
        .map_err(|e| anyhow::anyhow!("PNG decode: {e}"))?;
    let mut buf = vec![0u8; reader.output_buffer_size().expect("PNG info missing")];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| anyhow::anyhow!("PNG frame: {e}"))?;
    let width = info.width;
    let height = info.height;
    let src = &buf[..info.buffer_size()];

    let stride = match info.color_type {
        png::ColorType::Rgba => 4,
        png::ColorType::Rgb => 3,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Grayscale => 1,
        other => anyhow::bail!("Unsupported PNG colour type: {other:?}"),
    };

    let rgba = tint_pixels(src, stride, r, g, b, a);
    Ok((rgba, width, height))
}

/// Apply a colour tint to raw pixel data.
pub fn tint_pixels(src: &[u8], stride: usize, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
    let pixel_count = src.len() / stride;
    let mut rgba = Vec::with_capacity(pixel_count * 4);

    for pixel in src.chunks_exact(stride) {
        // For RGBA the icon is an alpha mask: replace RGB with the tint
        // colour and combine the source alpha with the tint alpha.
        // For greyscale formats the luminance modulates the tint intensity.
        let (lum, pa) = match stride {
            4 => (255u16, pixel[3]),
            3 => (pixel[0] as u16, 255),
            2 => (pixel[0] as u16, pixel[1]),
            1 => (pixel[0] as u16, 255),
            _ => (0u16, 0),
        };

        if pa == 0 {
            rgba.extend_from_slice(&[0, 0, 0, 0]);
        } else {
            let tr = ((r as u16 * lum) / 255u16).min(255) as u8;
            let tg = ((g as u16 * lum) / 255u16).min(255) as u8;
            let tb = ((b as u16 * lum) / 255u16).min(255) as u8;
            let ta = ((a as u16 * pa as u16) / 255u16).min(255) as u8;
            rgba.extend_from_slice(&[tr, tg, tb, ta]);
        }
    }

    rgba
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- TrayState / TrayAction --

    #[test]
    fn tray_state_equality() {
        assert_eq!(TrayState::Idle, TrayState::Idle);
        assert_ne!(TrayState::Idle, TrayState::Recording);
        assert_ne!(TrayState::Recording, TrayState::Transcribing);
        assert_ne!(TrayState::Transcribing, TrayState::Downloading);
        assert_ne!(TrayState::Downloading, TrayState::Error);
    }

    #[test]
    fn tray_state_clone() {
        let s = TrayState::Recording;
        assert_eq!(s.clone(), TrayState::Recording);
    }

    #[test]
    fn tray_state_debug() {
        assert_eq!(format!("{:?}", TrayState::Idle), "Idle");
        assert_eq!(format!("{:?}", TrayState::Error), "Error");
    }

    #[test]
    fn tray_action_debug_and_eq() {
        assert_eq!(TrayAction::Quit, TrayAction::Quit);
        assert_ne!(TrayAction::Quit, TrayAction::CopyLastDictation);
        assert!(format!("{:?}", TrayAction::SetModel("base.en".to_string())).contains("base.en"));
    }

    // -- state_display --

    #[test]
    fn state_display_all() {
        for state in [
            TrayState::Idle,
            TrayState::Recording,
            TrayState::Transcribing,
            TrayState::Downloading,
            TrayState::Error,
        ] {
            let (tooltip, label) = state_display(&state);
            assert!(!tooltip.is_empty());
            assert!(!label.is_empty());
            assert!(tooltip.contains("murmur"));
            assert!(label.contains("murmur"));
        }
    }

    // -- radio_label --

    #[test]
    fn radio_label_selected() {
        assert_eq!(radio_label("base.en", true), "● base.en");
    }

    #[test]
    fn radio_label_unselected() {
        assert_eq!(radio_label("base.en", false), "○ base.en");
    }

    // -- MenuActionMap --

    fn default_ids() -> MenuActionIds {
        MenuActionIds {
            quit: MenuId::new("q"),
            copy_last: MenuId::new("c"),
            open_config: MenuId::new("oc"),
            reload_config: MenuId::new("rc"),
            spoken_punct: MenuId::new("sp"),
            filler_removal: MenuId::new("fr"),
            streaming: MenuId::new("st"),
            translate: MenuId::new("tr"),
            noise_suppression: MenuId::new("ns"),
            set_hotkey: MenuId::new("sh"),
        }
    }

    #[test]
    fn menu_action_map_matches_quit() {
        let quit_id = MenuId::new("quit");
        let map = MenuActionMap::new(
            MenuActionIds {
                quit: quit_id.clone(),
                ..default_ids()
            },
            vec![],
            vec![],
            vec![],
        );
        assert_eq!(map.match_event(&quit_id), Some(TrayAction::Quit));
    }

    #[test]
    fn menu_action_map_matches_copy_last() {
        let id = MenuId::new("copy");
        let map = MenuActionMap::new(
            MenuActionIds {
                copy_last: id.clone(),
                ..default_ids()
            },
            vec![],
            vec![],
            vec![],
        );
        assert_eq!(map.match_event(&id), Some(TrayAction::CopyLastDictation));
    }

    #[test]
    fn menu_action_map_matches_open_config() {
        let id = MenuId::new("oc2");
        let map = MenuActionMap::new(
            MenuActionIds {
                open_config: id.clone(),
                ..default_ids()
            },
            vec![],
            vec![],
            vec![],
        );
        assert_eq!(map.match_event(&id), Some(TrayAction::OpenConfig));
    }

    #[test]
    fn menu_action_map_matches_reload_config() {
        let id = MenuId::new("rc2");
        let map = MenuActionMap::new(
            MenuActionIds {
                reload_config: id.clone(),
                ..default_ids()
            },
            vec![],
            vec![],
            vec![],
        );
        assert_eq!(map.match_event(&id), Some(TrayAction::ReloadConfig));
    }

    #[test]
    fn menu_action_map_matches_spoken_punct() {
        let id = MenuId::new("sp2");
        let map = MenuActionMap::new(
            MenuActionIds {
                spoken_punct: id.clone(),
                ..default_ids()
            },
            vec![],
            vec![],
            vec![],
        );
        assert_eq!(
            map.match_event(&id),
            Some(TrayAction::ToggleSpokenPunctuation)
        );
    }

    #[test]
    fn menu_action_map_matches_set_mode() {
        let id = MenuId::new("mode_ptt");
        let map = MenuActionMap::new(
            default_ids(),
            vec![],
            vec![],
            vec![(id.clone(), InputMode::PushToTalk)],
        );
        assert_eq!(
            map.match_event(&id),
            Some(TrayAction::SetMode(InputMode::PushToTalk))
        );
    }

    #[test]
    fn menu_action_map_matches_streaming() {
        let id = MenuId::new("st2");
        let map = MenuActionMap::new(
            MenuActionIds {
                streaming: id.clone(),
                ..default_ids()
            },
            vec![],
            vec![],
            vec![],
        );
        assert_eq!(map.match_event(&id), Some(TrayAction::ToggleStreaming));
    }

    #[test]
    fn menu_action_map_matches_translate() {
        let id = MenuId::new("tr2");
        let map = MenuActionMap::new(
            MenuActionIds {
                translate: id.clone(),
                ..default_ids()
            },
            vec![],
            vec![],
            vec![],
        );
        assert_eq!(map.match_event(&id), Some(TrayAction::ToggleTranslate));
    }

    #[test]
    fn menu_action_map_matches_set_hotkey() {
        let id = MenuId::new("sh2");
        let map = MenuActionMap::new(
            MenuActionIds {
                set_hotkey: id.clone(),
                ..default_ids()
            },
            vec![],
            vec![],
            vec![],
        );
        assert_eq!(map.match_event(&id), Some(TrayAction::SetHotkey));
    }

    #[test]
    fn menu_action_map_matches_model() {
        let model_id = MenuId::new("m1");
        let map = MenuActionMap::new(
            default_ids(),
            vec![(model_id.clone(), "base.en".to_string())],
            vec![],
            vec![],
        );
        assert_eq!(
            map.match_event(&model_id),
            Some(TrayAction::SetModel("base.en".to_string()))
        );
    }

    #[test]
    fn menu_action_map_matches_language() {
        let lang_id = MenuId::new("l1");
        let map = MenuActionMap::new(
            default_ids(),
            vec![],
            vec![(lang_id.clone(), "fr".to_string())],
            vec![],
        );
        assert_eq!(
            map.match_event(&lang_id),
            Some(TrayAction::SetLanguage("fr".to_string()))
        );
    }

    #[test]
    fn menu_action_map_unknown_id() {
        let map = MenuActionMap::new(default_ids(), vec![], vec![], vec![]);
        assert_eq!(map.match_event(&MenuId::new("unknown")), None);
    }

    #[test]
    fn menu_action_map_multiple_models() {
        let m1 = MenuId::new("m1");
        let m2 = MenuId::new("m2");
        let map = MenuActionMap::new(
            default_ids(),
            vec![
                (m1.clone(), "tiny.en".to_string()),
                (m2.clone(), "large".to_string()),
            ],
            vec![],
            vec![],
        );
        assert_eq!(
            map.match_event(&m1),
            Some(TrayAction::SetModel("tiny.en".to_string()))
        );
        assert_eq!(
            map.match_event(&m2),
            Some(TrayAction::SetModel("large".to_string()))
        );
    }

    // -- tint_pixels --

    #[test]
    fn tint_pixels_rgba_opaque() {
        let src = [255u8, 255, 255, 255];
        let result = tint_pixels(&src, 4, 255, 0, 0, 255);
        assert_eq!(result, vec![255, 0, 0, 255]);
    }

    #[test]
    fn tint_pixels_rgba_transparent() {
        let src = [255u8, 255, 255, 0];
        let result = tint_pixels(&src, 4, 255, 0, 0, 255);
        assert_eq!(result, vec![0, 0, 0, 0]);
    }

    #[test]
    fn tint_pixels_rgba_half_lum() {
        // RGBA stride treats the icon as an alpha mask: source RGB is
        // ignored and replaced by the tint colour.
        let src = [128u8, 0, 0, 255];
        let result = tint_pixels(&src, 4, 255, 255, 255, 255);
        assert_eq!(result, vec![255, 255, 255, 255]);
    }

    #[test]
    fn tint_pixels_rgb_stride3() {
        let src = [200u8, 100, 50];
        let result = tint_pixels(&src, 3, 255, 128, 64, 200);
        assert_eq!(result.len(), 4);
        assert_eq!(result[3], 200);
    }

    #[test]
    fn tint_pixels_grayscale_alpha_stride2() {
        let src = [128u8, 200];
        let result = tint_pixels(&src, 2, 100, 200, 50, 255);
        assert_eq!(result.len(), 4);
        assert_eq!(result[3], 200);
    }

    #[test]
    fn tint_pixels_grayscale_stride1() {
        let src = [255u8];
        let result = tint_pixels(&src, 1, 100, 200, 50, 128);
        assert_eq!(result, vec![100, 200, 50, 128]);
    }

    #[test]
    fn tint_pixels_unknown_stride() {
        let src = [1u8, 2, 3, 4, 5];
        let result = tint_pixels(&src, 5, 255, 255, 255, 255);
        assert_eq!(result, vec![0, 0, 0, 0]);
    }

    #[test]
    fn tint_pixels_black_input() {
        // RGBA alpha-mask: even black source pixels get replaced with the
        // tint colour (alpha is preserved).
        let src = [0u8, 0, 0, 255];
        let result = tint_pixels(&src, 4, 255, 255, 255, 255);
        assert_eq!(result[0..3], [255, 255, 255]);
    }

    #[test]
    fn tint_pixels_empty() {
        assert!(tint_pixels(&[], 4, 255, 0, 0, 255).is_empty());
    }

    // -- tint_png_rgba --

    #[test]
    fn tint_png_rgba_embedded_icon() {
        let (rgba, w, h) = tint_png_rgba(ICON_PNG, 100, 150, 255, 200).unwrap();
        assert!(w > 0 && h > 0);
        assert_eq!(rgba.len(), (w * h * 4) as usize);
    }

    #[test]
    fn tint_png_rgba_different_colors() {
        let (r1, _, _) = tint_png_rgba(ICON_PNG, 255, 0, 0, 255).unwrap();
        let (r2, _, _) = tint_png_rgba(ICON_PNG, 0, 0, 255, 255).unwrap();
        assert_ne!(r1, r2);
    }

    #[test]
    fn tint_png_rgba_invalid() {
        assert!(tint_png_rgba(b"not a png", 255, 0, 0, 255).is_err());
    }

    // -- constants --

    #[test]
    fn top_languages_valid() {
        for &code in TOP_LANGUAGES {
            assert!(crate::config::is_valid_language(code));
        }
    }

    // -- dark/light mode --

    #[test]
    fn is_dark_mode_does_not_panic() {
        let _ = is_dark_mode();
    }

    #[test]
    fn state_colors_dark_and_light_differ() {
        // Verify that dark and light palettes produce different idle colours
        let dark = StateColors {
            idle: (100, 150, 255, 200),
            recording: (255, 80, 80, 230),
            transcribing: (255, 210, 50, 220),
            loading: (180, 180, 180, 140),
        };
        let light = StateColors {
            idle: (30, 80, 200, 220),
            recording: (200, 30, 30, 230),
            transcribing: (180, 140, 0, 220),
            loading: (100, 100, 100, 160),
        };
        assert_ne!(dark.idle, light.idle);
        assert_ne!(dark.recording, light.recording);
    }

    #[test]
    fn embedded_icon_is_retina_resolution() {
        let (_, w, h) = tint_png_rgba(ICON_PNG, 0, 0, 0, 255).unwrap();
        // Menu bar icons need ≥ 36px for 2× Retina at 18pt display size
        assert!(w >= 36, "icon width {w} too small for Retina");
        assert!(h >= 36, "icon height {h} too small for Retina");
    }
}
