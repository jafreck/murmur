//! System tray UI.

use std::time::Instant;

use anyhow::Result;
use tray_icon::menu::{CheckMenuItem, Menu, MenuEvent, MenuItem, PredefinedMenuItem, Submenu};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};

use crate::config::{self, Config, InputMode};

mod icons;
mod menu;
mod state;

#[cfg(target_os = "macos")]
mod native_icon_cache;
#[cfg(target_os = "macos")]
use native_icon_cache::NativeIconCache;

#[cfg(target_os = "linux")]
mod linux_icon_cache;
#[cfg(target_os = "linux")]
use linux_icon_cache::LinuxIconCache;

// Re-export public API from submodules.
#[allow(deprecated)]
pub use icons::{
    compute_pulse_alpha, decode_icon_png, decode_png_bytes, decode_recording_frames,
    decode_transcribing_frames, should_throttle_frame, state_icon_key, tint_pixels, tint_png_rgba,
    DecodedIcon, IconKey,
};
pub use menu::{radio_label, MenuActionIds, MenuActionMap, TOP_LANGUAGES};
pub use state::state_display;

// Re-export for sibling submodules (native_icon_cache, linux_icon_cache).
use icons::StateColors;

use icons::{
    build_recording_animation_frames, build_transcribing_animation_frames, is_dark_mode,
    make_icon_from_decoded,
};
use menu::{build_radio_submenu, update_radio_entries, RadioEntry};

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
    SetBackend(config::AsrBackend),
    ToggleSpokenPunctuation,
    ToggleFillerWordRemoval,
    SetMode(InputMode),
    ToggleStreaming,
    ToggleTranslate,
    ToggleNoiseSuppression,
    SetHotkey,
    ToggleAppMode,
    CheckForUpdates,
    OpenConfig,
    ReloadConfig,
    Quit,
}

/// Manages the system tray icon and context menu.
pub struct TrayController {
    tray: TrayIcon,
    pub state: TrayState,
    action_map: MenuActionMap,

    model_entries: Vec<RadioEntry<String>>,
    language_entries: Vec<RadioEntry<String>>,
    mode_entries: Vec<RadioEntry<InputMode>>,
    backend_entries: Vec<RadioEntry<config::AsrBackend>>,

    spoken_punct_item: CheckMenuItem,
    filler_removal_item: CheckMenuItem,
    streaming_item: CheckMenuItem,
    translate_item: CheckMenuItem,
    #[allow(dead_code)]
    noise_suppression_item: CheckMenuItem,
    app_mode_item: CheckMenuItem,

    status_item: MenuItem,
    hotkey_item: MenuItem,
    update_item: MenuItem,

    idle_icon: Icon,
    recording_icon: Icon,
    transcribing_icon: Icon,
    loading_icon: Icon,

    /// Pre-rendered recording animation frames (from embedded APNG).
    recording_frames: Vec<Icon>,

    /// Pre-rendered transcribing animation frames (from embedded APNG).
    transcribing_frames: Vec<Icon>,

    /// Pre-decoded PNG pixel data kept for rebuilding icons on appearance change.
    decoded_icon: DecodedIcon,

    /// Cached dark-mode flag to avoid spawning a subprocess on every icon update.
    cached_dark_mode: bool,

    /// macOS: pre-built NSImage cache for zero-cost icon swaps.
    #[cfg(target_os = "macos")]
    native_cache: Option<NativeIconCache>,

    /// Linux: pre-written PNG file cache for AppIndicator.
    #[cfg(target_os = "linux")]
    linux_cache: Option<LinuxIconCache>,

    /// When set, the icon animates to indicate recording is active.
    animation_start: Option<Instant>,
    last_animation_frame: Option<Instant>,

    /// Deferred state: applied by `tick()` after one full animation cycle has
    /// elapsed since the deferred request, so the user always sees a smooth loop.
    deferred_state: Option<(TrayState, Instant)>,
}

impl TrayController {
    pub fn new(config: &Config) -> Result<Self> {
        let status_item = MenuItem::new("murmur: Idle", false, None);
        let hotkey_item = MenuItem::new(format!("Hotkey: {}", config.hotkey()), false, None);

        let backend_submenu = Submenu::new("Backend", true);
        let backend_choices: Vec<(&str, config::AsrBackend)> = vec![
            ("Qwen3-ASR", config::AsrBackend::Qwen3Asr),
            ("Whisper", config::AsrBackend::Whisper),
            ("Parakeet", config::AsrBackend::Parakeet),
        ];
        let (backend_entries, backend_ids) =
            build_radio_submenu(&backend_submenu, &backend_choices, |b| {
                *b == config.asr_backend()
            })?;

        let set_hotkey = MenuItem::new("Set Hotkey…", true, None);
        let set_hotkey_id = set_hotkey.id().clone();

        let copy_last = MenuItem::new("Copy Last Dictation", true, None);
        let copy_last_id = copy_last.id().clone();

        let model_submenu = Submenu::new("Model", true);
        let model_items: Vec<(&str, String)> =
            crate::config::supported_models(config.asr_backend())
                .iter()
                .map(|&s| (s, s.to_string()))
                .collect();
        let (model_entries, model_ids) =
            build_radio_submenu(&model_submenu, &model_items, |s| s == config.model_size())?;

        let lang_submenu = Submenu::new("Language", true);
        let lang_items: Vec<(&str, String)> = TOP_LANGUAGES
            .iter()
            .map(|&code| {
                let name = config::language_name(code).unwrap_or(code);
                (name, code.to_string())
            })
            .collect();
        let (language_entries, language_ids) =
            build_radio_submenu(&lang_submenu, &lang_items, |c| c == config.language())?;

        let spoken_punct_item = CheckMenuItem::new(
            "Spoken Punctuation",
            true,
            config.spoken_punctuation(),
            None,
        );
        let spoken_punct_id = spoken_punct_item.id().clone();

        let filler_removal_item = CheckMenuItem::new(
            "Filler Word Removal",
            true,
            config.filler_word_removal(),
            None,
        );
        let filler_removal_id = filler_removal_item.id().clone();

        let mode_submenu = Submenu::new("Mode", true);
        let mode_items: Vec<(&str, InputMode)> = vec![
            ("Push to Talk", InputMode::PushToTalk),
            ("Open Mic", InputMode::OpenMic),
        ];
        let (mode_entries, mode_ids) =
            build_radio_submenu(&mode_submenu, &mode_items, |m| m == config.mode())?;

        let streaming_item =
            CheckMenuItem::new("Live Streaming (Preview)", true, config.streaming(), None);
        let streaming_id = streaming_item.id().clone();

        let translate_item = CheckMenuItem::new(
            "Translate to English",
            true,
            config.translate_to_english(),
            None,
        );
        let translate_id = translate_item.id().clone();

        let noise_suppression_item =
            CheckMenuItem::new("Noise Suppression", true, config.noise_suppression(), None);
        let noise_suppression_id = noise_suppression_item.id().clone();

        let app_mode_item = CheckMenuItem::new("Notes Mode", true, config.is_notes_mode(), None);
        let app_mode_id = app_mode_item.id().clone();

        let open_config = MenuItem::new("Open Config…", true, None);
        let open_config_id = open_config.id().clone();
        let reload_config = MenuItem::new("Reload Config", true, None);
        let reload_config_id = reload_config.id().clone();
        let update_item = MenuItem::new("Check for Updates…", true, None);
        let check_for_updates_id = update_item.id().clone();

        let quit = MenuItem::new("Quit", true, None);
        let quit_id = quit.id().clone();

        let menu = Menu::new();
        menu.append(&status_item)?;
        menu.append(&PredefinedMenuItem::separator())?;
        menu.append(&copy_last)?;
        menu.append(&PredefinedMenuItem::separator())?;
        menu.append(&backend_submenu)?;
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
        menu.append(&app_mode_item)?;
        menu.append(&PredefinedMenuItem::separator())?;
        menu.append(&open_config)?;
        menu.append(&reload_config)?;
        menu.append(&update_item)?;
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
                app_mode: app_mode_id,
                check_for_updates: check_for_updates_id,
            },
            model_ids,
            language_ids,
            mode_ids,
            backend_ids,
        );

        let dark_mode = is_dark_mode();
        let colors = StateColors::for_appearance(dark_mode);
        let decoded = decode_icon_png()?;
        let idle_icon = make_icon_from_decoded(
            &decoded,
            colors.idle.0,
            colors.idle.1,
            colors.idle.2,
            colors.idle.3,
        )?;
        let recording_icon = make_icon_from_decoded(
            &decoded,
            colors.recording.0,
            colors.recording.1,
            colors.recording.2,
            colors.recording.3,
        )?;
        let transcribing_icon = make_icon_from_decoded(
            &decoded,
            colors.transcribing.0,
            colors.transcribing.1,
            colors.transcribing.2,
            colors.transcribing.3,
        )?;
        let loading_icon = make_icon_from_decoded(
            &decoded,
            colors.loading.0,
            colors.loading.1,
            colors.loading.2,
            colors.loading.3,
        )?;

        let recording_frames = build_recording_animation_frames(colors.recording);
        let transcribing_frames = build_transcribing_animation_frames(colors.transcribing);

        let tray = TrayIconBuilder::new()
            .with_icon(idle_icon.clone())
            .with_tooltip("murmur — Idle")
            .with_menu(Box::new(menu))
            .with_menu_on_left_click(true)
            .build()?;

        #[cfg(target_os = "macos")]
        let native_cache = match NativeIconCache::new(&tray, &decoded, &colors) {
            Ok(cache) => Some(cache),
            Err(e) => {
                log::warn!("Native icon cache unavailable, falling back to tray-icon: {e}");
                None
            }
        };

        #[cfg(target_os = "linux")]
        let linux_cache = match LinuxIconCache::new(&decoded, &colors) {
            Ok(cache) => Some(cache),
            Err(e) => {
                log::warn!("Linux icon cache unavailable, falling back to tray-icon: {e}");
                None
            }
        };

        let controller = Self {
            tray,
            state: TrayState::Idle,
            action_map,
            model_entries,
            language_entries,
            mode_entries,
            backend_entries,
            spoken_punct_item,
            filler_removal_item,
            streaming_item,
            translate_item,
            noise_suppression_item,
            app_mode_item,
            status_item,
            hotkey_item,
            update_item,
            idle_icon,
            recording_icon,
            transcribing_icon,
            loading_icon,
            recording_frames,
            transcribing_frames,
            decoded_icon: decoded,
            cached_dark_mode: dark_mode,
            #[cfg(target_os = "macos")]
            native_cache,
            #[cfg(target_os = "linux")]
            linux_cache,
            animation_start: None,
            last_animation_frame: None,
            deferred_state: None,
        };

        if config::is_english_only_model(config.model_size()) {
            controller.set_language_menu_enabled(false);
        }

        Ok(controller)
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

    /// Enable or disable all language menu entries.
    pub fn set_language_menu_enabled(&self, enabled: bool) {
        for entry in &self.language_entries {
            entry.item.set_enabled(enabled);
        }
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

    /// Update the tray menu to show that an update is available.
    pub fn set_update_available(&mut self, version: &str) {
        self.update_item
            .set_text(format!("Update Available (v{version})"));
    }

    pub fn match_menu_event(&self, event: &MenuEvent) -> Option<TrayAction> {
        self.action_map.match_event(event.id())
    }
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
}
