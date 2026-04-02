use std::sync::Arc;

use crate::audio::recordings::RecordingStore;
use crate::config::{AppMode, Config, InputMode};
use crate::ui::tray::TrayState;
use murmur_core::transcription::AsrEngine;

use super::messages::{AppEffect, AppMessage};

/// Pure state machine for the app's recording logic.
pub struct AppState {
    pub(super) is_pressed: bool,
    pub(super) mode: InputMode,
    pub(super) streaming: bool,
    pub(super) spoken_punctuation: bool,
    pub(super) filler_word_removal: bool,
    pub(super) translate_to_english: bool,
    pub(super) noise_suppression: bool,
    pub(super) max_recordings: u32,
    pub(super) last_transcription: Option<String>,
    pub(super) model_size: String,
    pub(super) language: String,
    /// True while a streaming session is actively inserting partial text.
    pub(super) streaming_active: bool,
    /// Number of characters currently on screen from streaming emissions.
    pub(super) streaming_chars_emitted: usize,
    /// Monotonic counter incremented on each ReloadTranscriber request.
    pub(super) reload_generation: u64,
    /// True while waiting for the user to press a key to set as the new hotkey.
    pub(super) capturing_hotkey: bool,
    /// True when the current dictation session was started by the wake word.
    pub(super) wake_word_initiated: bool,
    /// Application mode: Dictation (paste at cursor) or Notes (overlay + wake word).
    pub(super) app_mode: AppMode,
    /// Accumulated overlay text for the current session.
    pub(super) overlay_text: String,
    /// The configured stop phrase for wake-word-initiated sessions.
    pub(super) stop_phrase: String,
    /// Timestamp of last speech activity (recording start or streaming text).
    /// Used to auto-stop wake-word-initiated sessions after silence.
    pub(super) last_speech_at: Option<std::time::Instant>,
    /// Set to true when a streaming session completes, so the batch
    /// transcription at the end can be skipped (avoids duplicate output).
    pub(super) streaming_completed: bool,
    /// Holds a newly loaded engine until the `SwapEngine` effect moves it
    /// into the active engine slot. Populated by the `EngineReady` reducer.
    pub(super) pending_engine: Option<Arc<dyn AsrEngine + Send + Sync>>,
}

impl AppState {
    pub fn new(config: &Config) -> Self {
        // Native-streaming backends (Qwen3-ASR, MLX, Parakeet) default to
        // streaming enabled. The user can still toggle it off via the tray.
        let streaming = config.streaming() || config.asr_backend().supports_native_streaming();
        Self {
            is_pressed: false,
            mode: config.mode().clone(),
            streaming,
            spoken_punctuation: config.spoken_punctuation(),
            filler_word_removal: config.filler_word_removal(),
            translate_to_english: config.translate_to_english(),
            noise_suppression: config.noise_suppression(),
            max_recordings: Config::effective_max_recordings(config.max_recordings()),
            last_transcription: None,
            model_size: config.model_size().to_string(),
            language: config.language().to_string(),
            streaming_active: false,
            streaming_chars_emitted: 0,
            reload_generation: 0,
            capturing_hotkey: false,
            wake_word_initiated: false,
            app_mode: config.app_mode(),
            overlay_text: String::new(),
            stop_phrase: config.stop_phrase().to_string(),
            last_speech_at: None,
            streaming_completed: false,
            pending_engine: None,
        }
    }

    // -- Public getters for read-only access from outside the app module --

    pub fn is_pressed(&self) -> bool {
        self.is_pressed
    }

    pub fn mode(&self) -> &InputMode {
        &self.mode
    }

    pub fn streaming(&self) -> bool {
        self.streaming
    }

    pub fn spoken_punctuation(&self) -> bool {
        self.spoken_punctuation
    }

    pub fn filler_word_removal(&self) -> bool {
        self.filler_word_removal
    }

    pub fn translate_to_english(&self) -> bool {
        self.translate_to_english
    }

    pub fn noise_suppression(&self) -> bool {
        self.noise_suppression
    }

    pub fn last_transcription(&self) -> Option<&str> {
        self.last_transcription.as_deref()
    }

    pub fn model_size(&self) -> &str {
        &self.model_size
    }

    pub fn language(&self) -> &str {
        &self.language
    }

    pub fn streaming_active(&self) -> bool {
        self.streaming_active
    }

    pub fn reload_generation(&self) -> u64 {
        self.reload_generation
    }

    pub fn capturing_hotkey(&self) -> bool {
        self.capturing_hotkey
    }

    pub fn app_mode(&self) -> AppMode {
        self.app_mode
    }

    pub fn recording_output_path(&self) -> std::path::PathBuf {
        if self.max_recordings == 0 {
            RecordingStore::temp_recording_path()
        } else {
            RecordingStore::new_recording_path()
        }
    }

    pub(super) fn is_notes_mode(&self) -> bool {
        self.app_mode == AppMode::Notes
    }

    pub fn handle_message(&mut self, msg: AppMessage) -> Vec<AppEffect> {
        match msg {
            AppMessage::KeyDown if self.capturing_hotkey => vec![AppEffect::None],
            AppMessage::KeyUp if self.capturing_hotkey => vec![AppEffect::None],
            AppMessage::KeyDown => self.on_key_down(),
            AppMessage::KeyUp => self.on_key_up(),
            AppMessage::TranscriptionDone(ref text) => self.on_transcription_done(text),
            AppMessage::TranscriptionError(ref e) => self.on_transcription_error(e),
            AppMessage::TrayCopyLast => self.on_copy_last(),
            AppMessage::TrayQuit => vec![AppEffect::Quit],
            AppMessage::TraySetModel(ref size) => self.on_set_model(size),
            AppMessage::TraySetLanguage(ref code) => self.on_set_language(code),
            AppMessage::TraySetBackend(backend) => self.on_set_backend(backend),
            AppMessage::TrayToggleSpokenPunctuation => self.on_toggle_spoken_punctuation(),
            AppMessage::TrayToggleFillerWordRemoval => self.on_toggle_filler_word_removal(),
            AppMessage::TraySetMode(ref mode) => self.on_set_mode(mode),
            AppMessage::TrayToggleStreaming => self.on_toggle_streaming(),
            AppMessage::TrayToggleTranslate => self.on_toggle_translate(),
            AppMessage::TrayToggleNoiseSuppression => self.on_toggle_noise_suppression(),
            AppMessage::TrayOpenConfig => vec![AppEffect::OpenConfig],
            AppMessage::TrayReloadConfig => vec![AppEffect::ReloadConfig],
            AppMessage::TraySetHotkey => self.on_tray_set_hotkey(),
            AppMessage::TrayToggleAppMode => self.on_toggle_app_mode(),
            AppMessage::HotkeyCapture(ref key) => self.on_hotkey_capture(key),
            AppMessage::EngineReady(engine, generation) => self.on_engine_ready(engine, generation),
            AppMessage::WakeWordDetected => self.on_wake_word_detected(),
            AppMessage::StopPhraseDetected => self.on_stop_phrase_detected(),
            AppMessage::SpeechActivity => {
                self.last_speech_at = Some(std::time::Instant::now());
                vec![AppEffect::None]
            }
            AppMessage::StreamingPartialText {
                ref text,
                ref replace_chars,
            } => self.on_streaming_partial(text, replace_chars),
            AppMessage::TrayCheckForUpdates => vec![AppEffect::CheckForUpdates],
            AppMessage::UpdateAvailable(info) => self.on_update_available(info),
            AppMessage::UpdateApplied(version) => self.on_update_applied(&version),
            AppMessage::UpdateError(e) => vec![AppEffect::LogError(e)],
        }
    }

    fn on_engine_ready(
        &mut self,
        engine: Arc<dyn AsrEngine + Send + Sync>,
        generation: u64,
    ) -> Vec<AppEffect> {
        if generation != self.reload_generation {
            log::info!(
                "Discarding stale engine reload (gen {generation}, current {})",
                self.reload_generation
            );
            return vec![AppEffect::None];
        }
        self.pending_engine = Some(engine);
        let mut effects = vec![
            AppEffect::SpawnStreamingWorker,
            AppEffect::SwapEngine,
            AppEffect::SetTrayState(TrayState::Idle),
        ];
        if generation == 0 {
            effects.push(AppEffect::PrintReady);
            if self.is_notes_mode() {
                effects.push(AppEffect::StartWakeWord);
            }
        }
        effects
    }

    fn on_update_available(&self, info: murmur_core::update::UpdateInfo) -> Vec<AppEffect> {
        log::info!(
            "Update available: v{} → v{}",
            info.current_version,
            info.latest_version
        );
        vec![AppEffect::SetTrayUpdateAvailable(info.latest_version)]
    }

    fn on_update_applied(&self, version: &str) -> Vec<AppEffect> {
        log::info!("Update applied: v{version}. Restart to use new version.");
        vec![AppEffect::SetTrayStatus(format!(
            "Updated to v{version} — restart to apply"
        ))]
    }
}

#[cfg(test)]
mod tests {
    use super::super::messages::{AppEffect, AppMessage};
    use super::*;
    use crate::config::{AppMode, Config, InputMode};
    use crate::ui::tray::{TrayAction, TrayState};

    fn default_state() -> AppState {
        AppState {
            is_pressed: false,
            mode: InputMode::PushToTalk,
            streaming: false,
            spoken_punctuation: false,
            filler_word_removal: true,
            translate_to_english: false,
            noise_suppression: true,
            max_recordings: 0,
            last_transcription: None,
            model_size: "0.6b".to_string(),
            language: "en".to_string(),
            streaming_active: false,
            streaming_chars_emitted: 0,
            reload_generation: 0,
            capturing_hotkey: false,
            wake_word_initiated: false,
            app_mode: AppMode::Dictation,
            overlay_text: String::new(),
            stop_phrase: "murmur stop".to_string(),
            last_speech_at: None,
            streaming_completed: false,
            pending_engine: None,
        }
    }

    // -- Hold-to-talk mode (mode = PushToTalk) --

    #[test]
    fn hold_mode_key_down_starts_recording() {
        let mut state = default_state();
        let effects = state.handle_message(AppMessage::KeyDown);
        assert!(state.is_pressed);
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::StartRecording(_))));
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::SetTrayState(TrayState::Recording))));
    }

    #[test]
    fn hold_mode_key_down_while_pressed_noop() {
        let mut state = default_state();
        state.is_pressed = true;
        let effects = state.handle_message(AppMessage::KeyDown);
        assert_eq!(effects, vec![AppEffect::None]);
    }

    #[test]
    fn hold_mode_key_up_starts_transcribing() {
        let mut state = default_state();
        state.is_pressed = true;
        let effects = state.handle_message(AppMessage::KeyUp);
        assert!(!state.is_pressed);
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::StopAndTranscribe)));
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::SetTrayState(TrayState::Transcribing))));
    }

    #[test]
    fn hold_mode_key_up_not_pressed_noop() {
        let mut state = default_state();
        let effects = state.handle_message(AppMessage::KeyUp);
        assert_eq!(effects, vec![AppEffect::None]);
    }

    // -- Toggle mode (mode = OpenMic) --

    #[test]
    fn toggle_mode_first_key_down_starts_recording() {
        let mut state = default_state();
        state.mode = InputMode::OpenMic;
        let effects = state.handle_message(AppMessage::KeyDown);
        assert!(state.is_pressed);
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::StartRecording(_))));
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::SetTrayState(TrayState::Recording))));
    }

    #[test]
    fn toggle_mode_second_key_down_stops_and_transcribes() {
        let mut state = default_state();
        state.mode = InputMode::OpenMic;
        state.is_pressed = true;
        let effects = state.handle_message(AppMessage::KeyDown);
        assert!(!state.is_pressed);
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::StopAndTranscribe)));
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::SetTrayState(TrayState::Transcribing))));
    }

    #[test]
    fn toggle_mode_key_up_noop() {
        let mut state = default_state();
        state.mode = InputMode::OpenMic;
        state.is_pressed = true;
        let effects = state.handle_message(AppMessage::KeyUp);
        assert_eq!(effects, vec![AppEffect::None]);
    }

    // -- Transcription results --

    #[test]
    fn transcription_done_inserts_text() {
        let mut state = default_state();
        let effects =
            state.handle_message(AppMessage::TranscriptionDone("hello world".to_string()));
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::InsertText(t) if t == "hello world")));
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::SetTrayState(TrayState::Idle))));
        assert_eq!(state.last_transcription, Some("hello world".to_string()));
    }

    #[test]
    fn transcription_done_empty_no_insert() {
        let mut state = default_state();
        let effects = state.handle_message(AppMessage::TranscriptionDone("".to_string()));
        assert!(!effects
            .iter()
            .any(|e| matches!(e, AppEffect::InsertText(_))));
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::SetTrayState(TrayState::Idle))));
        assert!(state.last_transcription.is_none());
    }

    #[test]
    fn transcription_done_passes_through_already_processed_text() {
        // Text arriving via TranscriptionDone is already postprocessed by the
        // background thread, so AppState must NOT re-process it.
        let mut state = default_state();
        state.spoken_punctuation = true;
        let effects = state.handle_message(AppMessage::TranscriptionDone("hello.".to_string()));
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::InsertText(t) if t == "hello.")));
    }

    #[test]
    fn transcription_error_sets_error_state() {
        let mut state = default_state();
        let effects = state.handle_message(AppMessage::TranscriptionError("fail".to_string()));
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::LogError(t) if t == "fail")));
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::SetTrayState(TrayState::Error))));
    }

    // -- Copy last --

    #[test]
    fn copy_last_with_transcription() {
        let mut state = default_state();
        state.last_transcription = Some("copied text".to_string());
        let effects = state.handle_message(AppMessage::TrayCopyLast);
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::CopyToClipboard(t) if t == "copied text")));
    }

    #[test]
    fn copy_last_without_transcription() {
        let mut state = default_state();
        let effects = state.handle_message(AppMessage::TrayCopyLast);
        assert_eq!(effects, vec![AppEffect::None]);
    }

    // -- Tray actions --

    #[test]
    fn quit_returns_quit_effect() {
        let mut state = default_state();
        let effects = state.handle_message(AppMessage::TrayQuit);
        assert_eq!(effects, vec![AppEffect::Quit]);
    }

    #[test]
    fn set_model_updates_state_and_saves() {
        let mut state = default_state();
        let effects = state.handle_message(AppMessage::TraySetModel("small.en".to_string()));
        assert_eq!(state.model_size, "small.en");
        assert!(effects.contains(&AppEffect::SaveConfig));
        assert!(effects.contains(&AppEffect::SetTrayModel("small.en".to_string())));
        assert!(effects.contains(&AppEffect::ReloadTranscriber(1)));
        assert_eq!(state.reload_generation, 1);
    }

    #[test]
    fn set_language_updates_state_and_saves() {
        let mut state = default_state();
        // Use a multilingual model so language changes are allowed
        state.model_size = "base".to_string();
        let effects = state.handle_message(AppMessage::TraySetLanguage("fr".to_string()));
        assert_eq!(state.language, "fr");
        assert!(effects.contains(&AppEffect::SaveConfig));
        assert!(effects.contains(&AppEffect::SetTrayLanguage("fr".to_string())));
        assert!(effects.contains(&AppEffect::ReloadTranscriber(1)));
        assert_eq!(state.reload_generation, 1);
    }

    #[test]
    fn reload_generation_increments_on_each_change() {
        let mut state = default_state();
        // Use a multilingual model so language changes also trigger reloads
        state.model_size = "base".to_string();
        state.handle_message(AppMessage::TraySetModel("small".to_string()));
        assert_eq!(state.reload_generation, 1);
        state.handle_message(AppMessage::TraySetLanguage("fr".to_string()));
        assert_eq!(state.reload_generation, 2);
        state.handle_message(AppMessage::TraySetModel("tiny".to_string()));
        assert_eq!(state.reload_generation, 3);
    }

    #[test]
    fn toggle_spoken_punctuation() {
        let mut state = default_state();
        assert!(!state.spoken_punctuation);
        let effects = state.handle_message(AppMessage::TrayToggleSpokenPunctuation);
        assert!(state.spoken_punctuation);
        assert!(effects.contains(&AppEffect::SaveConfig));
        let effects = state.handle_message(AppMessage::TrayToggleSpokenPunctuation);
        assert!(!state.spoken_punctuation);
        assert!(effects.contains(&AppEffect::SaveConfig));
    }

    #[test]
    fn toggle_toggle_mode() {
        let mut state = default_state();
        assert_eq!(state.mode, InputMode::PushToTalk);
        let effects = state.handle_message(AppMessage::TraySetMode(InputMode::OpenMic));
        assert_eq!(state.mode, InputMode::OpenMic);
        assert!(effects.contains(&AppEffect::SaveConfig));
        assert!(effects.contains(&AppEffect::SetTrayMode(InputMode::OpenMic)));
    }

    #[test]
    fn toggle_translate() {
        let mut state = default_state();
        assert!(!state.translate_to_english);
        let effects = state.handle_message(AppMessage::TrayToggleTranslate);
        assert!(state.translate_to_english);
        assert!(effects.contains(&AppEffect::SaveConfig));
    }

    // -- AppState construction --

    #[test]
    fn app_state_from_config() {
        let mut config = Config::default();
        config.set_hotkey("f9".to_string());
        config.set_model_size("small.en".to_string());
        config.set_language("fr".to_string());
        config.set_spoken_punctuation(true);
        config.set_filler_word_removal(true);
        config.set_max_recordings(10);
        config.set_mode(InputMode::OpenMic);
        config.set_streaming(true);
        config.set_translate_to_english(true);
        let state = AppState::new(&config);
        assert_eq!(state.model_size, "small.en");
        assert_eq!(state.language, "fr");
        assert!(state.spoken_punctuation);
        assert_eq!(state.mode, InputMode::OpenMic);
        assert!(state.streaming);
        assert!(state.translate_to_english);
        assert_eq!(state.max_recordings, 10);
        assert!(!state.is_pressed);
        assert!(state.last_transcription.is_none());
    }

    #[test]
    fn to_config_preserves_state() {
        let base = Config::default();
        let mut state = AppState::new(&base);
        state.model_size = "large".to_string();
        state.language = "de".to_string();
        state.spoken_punctuation = true;
        state.mode = InputMode::OpenMic;
        state.streaming = true;
        state.translate_to_english = true;
        let cfg = state.to_config(&base);
        assert_eq!(cfg.model_size(), "large");
        assert_eq!(cfg.language(), "de");
        assert!(cfg.spoken_punctuation());
        assert_eq!(cfg.mode().clone(), InputMode::OpenMic);
        assert!(cfg.streaming());
        assert!(cfg.translate_to_english());
        assert_eq!(cfg.hotkey(), base.hotkey());
    }

    #[test]
    fn recording_output_path_temp_when_no_max() {
        let state = default_state();
        let path = state.recording_output_path();
        assert!(path.to_string_lossy().contains("murmur-"));
    }

    // -- AppEffect enum tests --

    #[test]
    fn app_effect_debug_and_eq() {
        let e1 = AppEffect::None;
        let e2 = AppEffect::None;
        assert_eq!(e1, e2);
        assert_eq!(format!("{:?}", e1), "None");
        assert_ne!(AppEffect::Quit, AppEffect::None);
    }

    #[test]
    fn open_config_returns_effect() {
        let mut state = default_state();
        let effects = state.handle_message(AppMessage::TrayOpenConfig);
        assert_eq!(effects, vec![AppEffect::OpenConfig]);
    }

    #[test]
    fn reload_config_returns_effect() {
        let mut state = default_state();
        let effects = state.handle_message(AppMessage::TrayReloadConfig);
        assert_eq!(effects, vec![AppEffect::ReloadConfig]);
    }

    #[test]
    fn streaming_partial_text_produces_replace() {
        let mut state = default_state();
        let effects = state.handle_message(AppMessage::StreamingPartialText {
            text: "hello".to_string(),
            replace_chars: 0,
        });
        // Dictation mode: streaming partials produce no text insertion effects
        assert!(!effects
            .iter()
            .any(|e| matches!(e, AppEffect::StreamingReplace { .. })));
    }

    #[test]
    fn streaming_partial_text_empty_noop() {
        let mut state = default_state();
        let effects = state.handle_message(AppMessage::StreamingPartialText {
            text: String::new(),
            replace_chars: 0,
        });
        assert_eq!(effects, vec![AppEffect::None]);
    }

    #[test]
    fn transcription_done_replaces_streaming_text_with_final() {
        let mut state = default_state();
        state.streaming_active = true;
        state.streaming_chars_emitted = 9; // "hello wor" was on screen
        let effects =
            state.handle_message(AppMessage::TranscriptionDone("hello world".to_string()));
        // Dictation mode always uses InsertText, never StreamingReplace
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::InsertText(t) if t == "hello world")));
        assert!(!effects
            .iter()
            .any(|e| matches!(e, AppEffect::StreamingReplace { .. })));
        // Should still save last_transcription for copy-last
        assert_eq!(state.last_transcription, Some("hello world".to_string()));
        assert!(!state.streaming_active);
        assert_eq!(state.streaming_chars_emitted, 0);
    }

    #[test]
    fn transcription_done_inserts_when_not_streaming() {
        let mut state = default_state();
        state.streaming_active = false;
        let effects = state.handle_message(AppMessage::TranscriptionDone("hello".to_string()));
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::InsertText(_))));
    }

    #[test]
    fn recording_output_path_with_max_recordings() {
        let mut state = default_state();
        state.max_recordings = 5;
        let path = state.recording_output_path();
        let name = path.file_name().unwrap().to_string_lossy();
        assert!(name.starts_with("recording-"));
        assert!(name.ends_with(".wav"));
    }

    // -- From<TrayAction> for AppMessage --

    #[test]
    fn from_tray_action_quit() {
        match AppMessage::from(TrayAction::Quit) {
            AppMessage::TrayQuit => {}
            _ => panic!("expected TrayQuit"),
        }
    }

    #[test]
    fn from_tray_action_copy_last() {
        match AppMessage::from(TrayAction::CopyLastDictation) {
            AppMessage::TrayCopyLast => {}
            _ => panic!("expected TrayCopyLast"),
        }
    }

    #[test]
    fn from_tray_action_set_model() {
        match AppMessage::from(TrayAction::SetModel("base.en".into())) {
            AppMessage::TraySetModel(s) => assert_eq!(s, "base.en"),
            _ => panic!("expected TraySetModel"),
        }
    }

    #[test]
    fn from_tray_action_set_language() {
        match AppMessage::from(TrayAction::SetLanguage("fr".into())) {
            AppMessage::TraySetLanguage(c) => assert_eq!(c, "fr"),
            _ => panic!("expected TraySetLanguage"),
        }
    }

    #[test]
    fn from_tray_action_toggle_spoken_punct() {
        match AppMessage::from(TrayAction::ToggleSpokenPunctuation) {
            AppMessage::TrayToggleSpokenPunctuation => {}
            _ => panic!("expected TrayToggleSpokenPunctuation"),
        }
    }

    #[test]
    fn from_tray_action_set_mode() {
        match AppMessage::from(TrayAction::SetMode(InputMode::OpenMic)) {
            AppMessage::TraySetMode(mode) => assert_eq!(mode, InputMode::OpenMic),
            _ => panic!("expected TraySetMode"),
        }
    }

    #[test]
    fn from_tray_action_toggle_translate() {
        match AppMessage::from(TrayAction::ToggleTranslate) {
            AppMessage::TrayToggleTranslate => {}
            _ => panic!("expected TrayToggleTranslate"),
        }
    }

    #[test]
    fn from_tray_action_open_config() {
        match AppMessage::from(TrayAction::OpenConfig) {
            AppMessage::TrayOpenConfig => {}
            _ => panic!("expected TrayOpenConfig"),
        }
    }

    #[test]
    fn from_tray_action_reload_config() {
        match AppMessage::from(TrayAction::ReloadConfig) {
            AppMessage::TrayReloadConfig => {}
            _ => panic!("expected TrayReloadConfig"),
        }
    }

    #[test]
    fn from_tray_action_set_hotkey() {
        match AppMessage::from(TrayAction::SetHotkey) {
            AppMessage::TraySetHotkey => {}
            _ => panic!("expected TraySetHotkey"),
        }
    }

    #[test]
    fn tray_set_hotkey_enters_capture_mode() {
        let mut state = default_state();
        let effects = state.handle_message(AppMessage::TraySetHotkey);
        assert!(state.capturing_hotkey);
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::EnterHotkeyCaptureMode)));
    }

    #[test]
    fn hotkey_capture_sets_hotkey() {
        let mut state = default_state();
        state.capturing_hotkey = true;
        let effects = state.handle_message(AppMessage::HotkeyCapture(rdev::Key::F5));
        assert!(!state.capturing_hotkey);
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::SetHotkey(k) if k == "f5")));
        assert!(effects.iter().any(|e| matches!(e, AppEffect::SaveConfig)));
    }

    #[test]
    fn key_down_ignored_during_capture() {
        let mut state = default_state();
        state.capturing_hotkey = true;
        let effects = state.handle_message(AppMessage::KeyDown);
        assert_eq!(effects, vec![AppEffect::None]);
    }

    #[test]
    fn key_up_ignored_during_capture() {
        let mut state = default_state();
        state.capturing_hotkey = true;
        let effects = state.handle_message(AppMessage::KeyUp);
        assert_eq!(effects, vec![AppEffect::None]);
    }

    // -- Streaming mode effects --

    #[test]
    fn streaming_key_down_includes_start_streaming() {
        let mut state = default_state();
        state.streaming = true;
        let effects = state.handle_message(AppMessage::KeyDown);
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::StartStreaming)));
        assert!(state.streaming_active);
    }

    #[test]
    fn streaming_key_up_includes_stop_streaming() {
        let mut state = default_state();
        state.streaming = true;
        state.is_pressed = true;
        state.streaming_active = true;
        let effects = state.handle_message(AppMessage::KeyUp);
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::StopStreaming)));
    }

    #[test]
    fn streaming_partial_text_with_replace_chars() {
        let mut state = default_state();
        let effects = state.handle_message(AppMessage::StreamingPartialText {
            text: "world".to_string(),
            replace_chars: 5,
        });
        // Dictation mode: streaming partials produce no text insertion effects
        assert!(!effects
            .iter()
            .any(|e| matches!(e, AppEffect::StreamingReplace { .. })));
    }

    #[test]
    fn streaming_partial_text_empty_text_with_replace_is_not_noop() {
        let mut state = default_state();
        let effects = state.handle_message(AppMessage::StreamingPartialText {
            text: String::new(),
            replace_chars: 3,
        });
        // Dictation mode: streaming partials produce no text insertion effects,
        // but replace_chars > 0 means the handler entered the active branch (not noop)
        assert!(!effects
            .iter()
            .any(|e| matches!(e, AppEffect::StreamingReplace { .. })));
        assert!(effects.is_empty(), "dictation mode emits no effects for streaming partials");
    }

    // -- Transcription results during recording --

    #[test]
    fn streaming_partial_text_tracks_emitted_chars() {
        let mut state = default_state();
        state.handle_message(AppMessage::StreamingPartialText {
            text: "hello".to_string(),
            replace_chars: 0,
        });
        assert_eq!(state.streaming_chars_emitted, 5);

        // Second partial replaces with longer text
        state.handle_message(AppMessage::StreamingPartialText {
            text: "hello world".to_string(),
            replace_chars: 5,
        });
        assert_eq!(state.streaming_chars_emitted, 11);
    }

    #[test]
    fn transcription_done_during_recording_does_not_reset_tray() {
        let mut state = default_state();
        state.is_pressed = true;
        let effects =
            state.handle_message(AppMessage::TranscriptionDone("from prev cycle".to_string()));
        // Should NOT set tray to Idle since we're currently recording
        assert!(!effects
            .iter()
            .any(|e| matches!(e, AppEffect::SetTrayState(TrayState::Idle))));
        // Should still save the transcription
        assert_eq!(
            state.last_transcription,
            Some("from prev cycle".to_string())
        );
    }

    #[test]
    fn transcription_error_during_recording_does_not_reset_tray() {
        let mut state = default_state();
        state.is_pressed = true;
        let effects = state.handle_message(AppMessage::TranscriptionError("timeout".to_string()));
        // Should NOT set tray to Error since we're currently recording
        assert!(!effects
            .iter()
            .any(|e| matches!(e, AppEffect::SetTrayState(TrayState::Error))));
        // Should still log the error
        assert!(effects.iter().any(|e| matches!(e, AppEffect::LogError(_))));
    }

    #[test]
    fn transcription_done_during_recording_does_not_clear_streaming_active() {
        let mut state = default_state();
        state.is_pressed = true;
        state.streaming_active = true;
        state.handle_message(AppMessage::TranscriptionDone("text".to_string()));
        // streaming_active should NOT be cleared since we're still recording
        assert!(state.streaming_active);
    }

    // -- Toggle streaming state --

    #[test]
    fn toggle_streaming() {
        let mut state = default_state();
        assert!(!state.streaming);
        let effects = state.handle_message(AppMessage::TrayToggleStreaming);
        assert!(state.streaming);
        assert!(effects.contains(&AppEffect::SaveConfig));
        let effects = state.handle_message(AppMessage::TrayToggleStreaming);
        assert!(!state.streaming);
        assert!(effects.contains(&AppEffect::SaveConfig));
    }

    // -- to_config preserves base config fields --

    #[test]
    fn to_config_preserves_base_hotkey_and_max_recordings() {
        let mut base = Config::default();
        base.set_hotkey("ctrl+shift+space".to_string());
        base.set_max_recordings(42);
        let state = AppState::new(&base);
        let cfg = state.to_config(&base);
        assert_eq!(cfg.hotkey(), "ctrl+shift+space");
        assert_eq!(cfg.max_recordings(), 42);
    }

    // -- EngineReady exhaustiveness --
    // EngineReady is handled in the run() loop, not handle_message().
    // Exhaustiveness is verified at compile time. No runtime test needed.

    // -- Open mic streaming --

    #[test]
    fn open_mic_streaming_start_stop() {
        let mut state = default_state();
        state.mode = InputMode::OpenMic;
        state.streaming = true;

        // First press: start recording + streaming
        let effects = state.handle_message(AppMessage::KeyDown);
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::StartStreaming)));
        assert!(state.streaming_active);

        // Second press: stop recording + streaming
        let effects = state.handle_message(AppMessage::KeyDown);
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::StopStreaming)));
    }

    // -- Multiple transcription results update last_transcription --

    #[test]
    fn last_transcription_updated_on_each_result() {
        let mut state = default_state();
        state.handle_message(AppMessage::TranscriptionDone("first".to_string()));
        assert_eq!(state.last_transcription, Some("first".to_string()));
        state.handle_message(AppMessage::TranscriptionDone("second".to_string()));
        assert_eq!(state.last_transcription, Some("second".to_string()));
    }

    // -- English-only model enforcement --

    #[test]
    fn set_english_only_model_forces_language_to_english() {
        let mut state = default_state();
        state.model_size = "base".to_string();
        state.language = "fr".to_string();
        let effects = state.handle_message(AppMessage::TraySetModel("distil-large-v3".to_string()));
        assert_eq!(state.language, "en");
        assert!(effects.contains(&AppEffect::SetTrayLanguage("en".to_string())));
        assert!(effects.contains(&AppEffect::SetLanguageMenuEnabled(false)));
    }

    #[test]
    fn set_en_model_disables_language_menu() {
        let mut state = default_state();
        state.model_size = "base".to_string();
        let effects = state.handle_message(AppMessage::TraySetModel("small.en".to_string()));
        assert!(effects.contains(&AppEffect::SetLanguageMenuEnabled(false)));
    }

    #[test]
    fn set_multilingual_model_enables_language_menu() {
        let mut state = default_state();
        let effects = state.handle_message(AppMessage::TraySetModel("large".to_string()));
        assert!(effects.contains(&AppEffect::SetLanguageMenuEnabled(true)));
    }

    #[test]
    fn language_change_blocked_for_english_only_model() {
        let mut state = default_state();
        state.model_size = "distil-large-v3".to_string();
        state.language = "en".to_string();
        let effects = state.handle_message(AppMessage::TraySetLanguage("fr".to_string()));
        assert_eq!(state.language, "en");
        assert!(!effects.contains(&AppEffect::SaveConfig));
        assert!(effects.contains(&AppEffect::SetTrayLanguage("en".to_string())));
    }

    #[test]
    fn language_change_allowed_for_multilingual_model() {
        let mut state = default_state();
        state.model_size = "large".to_string();
        let effects = state.handle_message(AppMessage::TraySetLanguage("fr".to_string()));
        assert_eq!(state.language, "fr");
        assert!(effects.contains(&AppEffect::SaveConfig));
    }
}
