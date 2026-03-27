use crate::audio::recordings::RecordingStore;
use crate::config::{Config, InputMode};
use crate::ui::tray::TrayState;
use rdev::Key;

use crate::transcription::transcriber::Transcriber;
#[cfg(test)]
use crate::ui::tray::TrayAction;
use std::sync::Arc;

pub enum AppMessage {
    KeyDown,
    KeyUp,
    TrayQuit,
    TrayCopyLast,
    TraySetModel(String),
    TraySetLanguage(String),
    TrayToggleSpokenPunctuation,
    TrayToggleFillerWordRemoval,
    TraySetMode(InputMode),
    TrayToggleStreaming,
    TrayToggleTranslate,
    TrayToggleNoiseSuppression,
    TrayOpenConfig,
    TrayReloadConfig,
    TraySetHotkey,
    TrayToggleWakeWord,
    TrayToggleOverlay,
    HotkeyCapture(Key),
    TranscriptionDone(String),
    TranscriptionError(String),
    StreamingPartialText {
        text: String,
        replace_chars: usize,
    },
    /// A new Transcriber has been loaded in the background and is ready to swap in.
    /// The u64 is the reload generation; stale reloads are discarded.
    TranscriberReady(Arc<Transcriber>, u64),
    /// The wake word was detected — start dictation.
    WakeWordDetected,
    /// The stop phrase was detected — stop dictation.
    StopPhraseDetected,
}

/// Effect returned by the app state machine in response to a message.
#[derive(Debug, Clone, PartialEq)]
pub enum AppEffect {
    None,
    StartRecording(std::path::PathBuf),
    StopAndTranscribe,
    /// Start streaming transcription (toggle mode only).
    StartStreaming,
    /// Stop the streaming transcription thread.
    StopStreaming,
    InsertText(String),
    /// Backspace `replace_chars` characters, then type `text` (streaming revisions).
    StreamingReplace {
        text: String,
        replace_chars: usize,
    },
    CopyToClipboard(String),
    SaveConfig,
    /// Open the config file in the user's default editor/file manager.
    OpenConfig,
    /// Reload config from disk and apply changes.
    ReloadConfig,
    SetTrayState(TrayState),
    SetTrayModel(String),
    SetTrayLanguage(String),
    /// Enable or disable the language submenu (disabled for English-only models).
    SetLanguageMenuEnabled(bool),
    SetTrayMode(InputMode),
    /// Download the model if needed and rebuild the Transcriber in a background thread.
    ReloadTranscriber(u64),
    /// Enter hotkey capture mode — the listener should capture the next key press.
    EnterHotkeyCaptureMode,
    /// Save the captured hotkey and update the listener.
    SetHotkey(String),
    /// Update the noise suppression state on the audio recorder.
    UpdateNoiseSuppression(bool),
    /// Show the overlay window.
    ShowOverlay,
    /// Hide the overlay window.
    HideOverlay,
    /// Update the overlay with new text.
    UpdateOverlayText(String),
    /// Save the current overlay text to a note file.
    SaveNote(String),
    /// Pause the wake word detector (during active dictation).
    PauseWakeWord,
    /// Resume the wake word detector (after dictation ends).
    ResumeWakeWord,
    /// Start the wake word detector.
    StartWakeWord,
    /// Stop the wake word detector.
    StopWakeWord,
    /// Spawn the overlay subprocess.
    SpawnOverlay,
    /// Kill the overlay subprocess.
    KillOverlay,
    Quit,
    LogError(String),
}

/// Pure state machine for the app's recording logic.
pub struct AppState {
    pub is_pressed: bool,
    pub mode: InputMode,
    pub streaming: bool,
    pub spoken_punctuation: bool,
    pub filler_word_removal: bool,
    pub translate_to_english: bool,
    pub noise_suppression: bool,
    pub max_recordings: u32,
    pub last_transcription: Option<String>,
    pub model_size: String,
    pub language: String,
    /// True while a streaming session is actively inserting partial text.
    pub streaming_active: bool,
    /// Number of characters currently on screen from streaming emissions.
    pub streaming_chars_emitted: usize,
    /// Monotonic counter incremented on each ReloadTranscriber request.
    pub reload_generation: u64,
    /// True while waiting for the user to press a key to set as the new hotkey.
    pub capturing_hotkey: bool,
    /// True when the current dictation session was started by the wake word.
    pub wake_word_initiated: bool,
    /// Wake word detection enabled.
    pub wake_word_enabled: bool,
    /// Overlay enabled.
    pub overlay_enabled: bool,
    /// Accumulated overlay text for the current session.
    pub overlay_text: String,
    /// The configured stop phrase for wake-word-initiated sessions.
    pub stop_phrase: String,
}

impl AppState {
    pub fn new(config: &Config) -> Self {
        Self {
            is_pressed: false,
            mode: config.mode.clone(),
            streaming: config.streaming,
            spoken_punctuation: config.spoken_punctuation,
            filler_word_removal: config.filler_word_removal,
            translate_to_english: config.translate_to_english,
            noise_suppression: config.noise_suppression,
            max_recordings: Config::effective_max_recordings(config.max_recordings),
            last_transcription: None,
            model_size: config.model_size.clone(),
            language: config.language.clone(),
            streaming_active: false,
            streaming_chars_emitted: 0,
            reload_generation: 0,
            capturing_hotkey: false,
            wake_word_initiated: false,
            wake_word_enabled: config.wake_word_enabled,
            overlay_enabled: config.overlay_enabled,
            overlay_text: String::new(),
            stop_phrase: config.stop_phrase.clone(),
        }
    }

    pub fn recording_output_path(&self) -> std::path::PathBuf {
        if self.max_recordings == 0 {
            RecordingStore::temp_recording_path()
        } else {
            RecordingStore::new_recording_path()
        }
    }

    pub fn handle_message(&mut self, msg: &AppMessage) -> Vec<AppEffect> {
        match msg {
            AppMessage::KeyDown if self.capturing_hotkey => vec![AppEffect::None],
            AppMessage::KeyUp if self.capturing_hotkey => vec![AppEffect::None],
            AppMessage::KeyDown => self.on_key_down(),
            AppMessage::KeyUp => self.on_key_up(),
            AppMessage::TranscriptionDone(text) => self.on_transcription_done(text),
            AppMessage::TranscriptionError(e) => self.on_transcription_error(e),
            AppMessage::TrayCopyLast => self.on_copy_last(),
            AppMessage::TrayQuit => vec![AppEffect::Quit],
            AppMessage::TraySetModel(size) => self.on_set_model(size),
            AppMessage::TraySetLanguage(code) => self.on_set_language(code),
            AppMessage::TrayToggleSpokenPunctuation => self.on_toggle_spoken_punctuation(),
            AppMessage::TrayToggleFillerWordRemoval => self.on_toggle_filler_word_removal(),
            AppMessage::TraySetMode(mode) => self.on_set_mode(mode),
            AppMessage::TrayToggleStreaming => self.on_toggle_streaming(),
            AppMessage::TrayToggleTranslate => self.on_toggle_translate(),
            AppMessage::TrayToggleNoiseSuppression => self.on_toggle_noise_suppression(),
            AppMessage::TrayOpenConfig => vec![AppEffect::OpenConfig],
            AppMessage::TrayReloadConfig => vec![AppEffect::ReloadConfig],
            AppMessage::TraySetHotkey => self.on_tray_set_hotkey(),
            AppMessage::TrayToggleWakeWord => self.on_toggle_wake_word(),
            AppMessage::TrayToggleOverlay => self.on_toggle_overlay(),
            AppMessage::HotkeyCapture(key) => self.on_hotkey_capture(key),
            // TranscriberReady is handled directly in the run() loop before
            // reaching handle_message, but we need an arm for exhaustiveness.
            AppMessage::TranscriberReady(_, _) => vec![AppEffect::None],
            AppMessage::WakeWordDetected => self.on_wake_word_detected(),
            AppMessage::StopPhraseDetected => self.on_stop_phrase_detected(),
            AppMessage::StreamingPartialText {
                text,
                replace_chars,
            } => self.on_streaming_partial(text, replace_chars),
        }
    }

    fn start_recording_effects(&mut self) -> Vec<AppEffect> {
        self.is_pressed = true;
        self.streaming_active = self.streaming;
        self.streaming_chars_emitted = 0;
        let path = self.recording_output_path();
        let mut effects = vec![AppEffect::StartRecording(path)];
        if self.streaming {
            effects.push(AppEffect::StartStreaming);
        }
        if self.overlay_enabled {
            self.overlay_text.clear();
            effects.push(AppEffect::ShowOverlay);
        }
        if self.wake_word_enabled {
            effects.push(AppEffect::PauseWakeWord);
        }
        effects.push(AppEffect::SetTrayState(TrayState::Recording));
        effects
    }

    fn stop_recording_effects(&mut self) -> Vec<AppEffect> {
        self.is_pressed = false;
        self.wake_word_initiated = false;
        let mut effects = vec![];
        if self.streaming {
            effects.push(AppEffect::StopStreaming);
        }
        effects.push(AppEffect::StopAndTranscribe);
        effects.push(AppEffect::SetTrayState(TrayState::Transcribing));
        effects
    }

    fn on_streaming_partial(&mut self, text: &str, replace_chars: &usize) -> Vec<AppEffect> {
        // Check for stop phrase in wake-word-initiated sessions
        if self.wake_word_initiated && !self.stop_phrase.is_empty() {
            if let Some(cleaned) =
                crate::input::wake_word::check_and_strip_stop_phrase(text, &self.stop_phrase)
            {
                // Stop phrase found — end dictation with cleaned text
                let char_count = cleaned.chars().count();
                let mut effects = vec![];
                if *replace_chars > 0 || !cleaned.is_empty() {
                    self.streaming_chars_emitted = char_count;
                    effects.push(AppEffect::StreamingReplace {
                        text: cleaned.clone(),
                        replace_chars: *replace_chars,
                    });
                }
                if self.overlay_enabled {
                    self.overlay_text = cleaned;
                }
                // Trigger stop
                effects.extend(self.stop_recording_effects());
                return effects;
            }
        }

        if *replace_chars > 0 || !text.is_empty() {
            self.streaming_chars_emitted = text.chars().count();
            let mut effects = vec![AppEffect::StreamingReplace {
                text: text.to_string(),
                replace_chars: *replace_chars,
            }];
            if self.overlay_enabled {
                self.overlay_text = text.to_string();
                effects.push(AppEffect::UpdateOverlayText(text.to_string()));
            }
            effects
        } else {
            vec![AppEffect::None]
        }
    }

    fn on_wake_word_detected(&mut self) -> Vec<AppEffect> {
        if self.is_pressed {
            return vec![AppEffect::None];
        }
        log::info!("Wake word detected — starting dictation");
        self.wake_word_initiated = true;
        self.start_recording_effects()
    }

    fn on_stop_phrase_detected(&mut self) -> Vec<AppEffect> {
        if !self.is_pressed {
            return vec![AppEffect::None];
        }
        log::info!("Stop phrase detected — stopping dictation");
        self.stop_recording_effects()
    }

    fn on_toggle_wake_word(&mut self) -> Vec<AppEffect> {
        self.wake_word_enabled = !self.wake_word_enabled;
        let mut effects = vec![AppEffect::SaveConfig];
        if self.wake_word_enabled {
            effects.push(AppEffect::StartWakeWord);
        } else {
            effects.push(AppEffect::StopWakeWord);
        }
        effects
    }

    fn on_toggle_overlay(&mut self) -> Vec<AppEffect> {
        self.overlay_enabled = !self.overlay_enabled;
        let mut effects = vec![AppEffect::SaveConfig];
        if self.overlay_enabled {
            effects.push(AppEffect::SpawnOverlay);
        } else {
            effects.push(AppEffect::KillOverlay);
        }
        effects
    }

    fn on_key_down(&mut self) -> Vec<AppEffect> {
        log::info!(
            "on_key_down: mode={:?} is_pressed={}",
            self.mode,
            self.is_pressed
        );
        match (&self.mode, self.is_pressed) {
            (InputMode::OpenMic, true) => self.stop_recording_effects(),
            (InputMode::OpenMic, false) => self.start_recording_effects(),
            (InputMode::PushToTalk, false) => self.start_recording_effects(),
            (InputMode::PushToTalk, true) => {
                // Key repeat while held — ignore.
                vec![AppEffect::None]
            }
        }
    }

    fn on_key_up(&mut self) -> Vec<AppEffect> {
        log::info!(
            "on_key_up: mode={:?} is_pressed={}",
            self.mode,
            self.is_pressed
        );
        if self.mode == InputMode::PushToTalk && self.is_pressed {
            self.stop_recording_effects()
        } else {
            vec![AppEffect::None]
        }
    }

    fn on_transcription_done(&mut self, text: &str) -> Vec<AppEffect> {
        let was_streaming = self.streaming_active;
        let streamed_chars = self.streaming_chars_emitted;
        // Only clear streaming state if not currently recording
        if !self.is_pressed {
            self.streaming_active = false;
            self.streaming_chars_emitted = 0;
        }

        let mut effects = vec![];
        if !text.is_empty() {
            if was_streaming && streamed_chars > 0 {
                effects.push(AppEffect::StreamingReplace {
                    text: text.to_string(),
                    replace_chars: streamed_chars,
                });
            } else if !was_streaming {
                effects.push(AppEffect::InsertText(text.to_string()));
            }
            self.last_transcription = Some(text.to_string());

            // Update overlay with final text and save note
            if self.overlay_enabled {
                self.overlay_text = text.to_string();
                effects.push(AppEffect::UpdateOverlayText(text.to_string()));
                effects.push(AppEffect::SaveNote(text.to_string()));
                effects.push(AppEffect::HideOverlay);
            }
        } else if self.overlay_enabled {
            effects.push(AppEffect::HideOverlay);
        }

        if !self.is_pressed {
            effects.push(AppEffect::SetTrayState(TrayState::Idle));
            // Resume wake word detection after dictation completes
            if self.wake_word_enabled {
                effects.push(AppEffect::ResumeWakeWord);
            }
        }
        effects
    }

    fn on_transcription_error(&mut self, error: &str) -> Vec<AppEffect> {
        // Only reset state if not currently recording — the error may be
        // from a previous cycle's transcription thread.
        let mut effects = vec![AppEffect::LogError(error.to_string())];
        if !self.is_pressed {
            self.streaming_active = false;
            effects.push(AppEffect::SetTrayState(TrayState::Error));
        }
        effects
    }

    fn on_copy_last(&self) -> Vec<AppEffect> {
        if let Some(ref text) = self.last_transcription {
            vec![AppEffect::CopyToClipboard(text.clone())]
        } else {
            vec![AppEffect::None]
        }
    }

    fn on_set_model(&mut self, size: &str) -> Vec<AppEffect> {
        self.model_size = size.to_string();
        self.reload_generation += 1;
        let mut effects = vec![
            AppEffect::SetTrayModel(size.to_string()),
            AppEffect::SetLanguageMenuEnabled(!crate::config::is_english_only_model(size)),
        ];
        if crate::config::is_english_only_model(size) && self.language != "en" {
            self.language = "en".to_string();
            effects.push(AppEffect::SetTrayLanguage("en".to_string()));
        }
        effects.push(AppEffect::SaveConfig);
        effects.push(AppEffect::ReloadTranscriber(self.reload_generation));
        effects
    }

    fn on_set_language(&mut self, code: &str) -> Vec<AppEffect> {
        if crate::config::is_english_only_model(&self.model_size) {
            // English-only model — ignore language change, reset tray to English
            return vec![AppEffect::SetTrayLanguage("en".to_string())];
        }
        self.language = code.to_string();
        self.reload_generation += 1;
        vec![
            AppEffect::SaveConfig,
            AppEffect::SetTrayLanguage(code.to_string()),
            AppEffect::ReloadTranscriber(self.reload_generation),
        ]
    }

    fn on_toggle_spoken_punctuation(&mut self) -> Vec<AppEffect> {
        self.spoken_punctuation = !self.spoken_punctuation;
        vec![AppEffect::SaveConfig]
    }

    fn on_toggle_filler_word_removal(&mut self) -> Vec<AppEffect> {
        self.filler_word_removal = !self.filler_word_removal;
        vec![AppEffect::SaveConfig]
    }

    fn on_set_mode(&mut self, mode: &InputMode) -> Vec<AppEffect> {
        self.mode = mode.clone();
        vec![AppEffect::SaveConfig, AppEffect::SetTrayMode(mode.clone())]
    }

    fn on_toggle_streaming(&mut self) -> Vec<AppEffect> {
        self.streaming = !self.streaming;
        vec![AppEffect::SaveConfig]
    }

    fn on_toggle_translate(&mut self) -> Vec<AppEffect> {
        self.translate_to_english = !self.translate_to_english;
        vec![AppEffect::SaveConfig]
    }

    fn on_toggle_noise_suppression(&mut self) -> Vec<AppEffect> {
        self.noise_suppression = !self.noise_suppression;
        vec![
            AppEffect::UpdateNoiseSuppression(self.noise_suppression),
            AppEffect::SaveConfig,
        ]
    }

    fn on_tray_set_hotkey(&mut self) -> Vec<AppEffect> {
        self.capturing_hotkey = true;
        vec![
            AppEffect::EnterHotkeyCaptureMode,
            AppEffect::SetTrayState(TrayState::Idle),
        ]
    }

    fn on_hotkey_capture(&mut self, key: &Key) -> Vec<AppEffect> {
        self.capturing_hotkey = false;
        let key_name = crate::input::keycodes::key_to_name(key);
        vec![AppEffect::SetHotkey(key_name), AppEffect::SaveConfig]
    }

    pub fn to_config(&self, base: &Config) -> Config {
        Config {
            hotkey: base.hotkey.clone(),
            model_size: self.model_size.clone(),
            language: self.language.clone(),
            spoken_punctuation: self.spoken_punctuation,
            filler_word_removal: self.filler_word_removal,
            max_recordings: base.max_recordings,
            mode: self.mode.clone(),
            streaming: self.streaming,
            translate_to_english: self.translate_to_english,
            noise_suppression: self.noise_suppression,
            vocabulary: base.vocabulary.clone(),
            app_contexts: base.app_contexts.clone(),
            excluded_apps: base.excluded_apps.clone(),
            dictation_mode: base.dictation_mode,
            wake_word_enabled: self.wake_word_enabled,
            wake_word: base.wake_word.clone(),
            stop_phrase: base.stop_phrase.clone(),
            overlay_enabled: self.overlay_enabled,
            notes_dir: base.notes_dir.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            model_size: "base.en".to_string(),
            language: "en".to_string(),
            streaming_active: false,
            streaming_chars_emitted: 0,
            reload_generation: 0,
            capturing_hotkey: false,
            wake_word_initiated: false,
            wake_word_enabled: false,
            overlay_enabled: false,
            overlay_text: String::new(),
            stop_phrase: "murmur stop".to_string(),
        }
    }

    // -- Hold-to-talk mode (mode = PushToTalk) --

    #[test]
    fn hold_mode_key_down_starts_recording() {
        let mut state = default_state();
        let effects = state.handle_message(&AppMessage::KeyDown);
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
        let effects = state.handle_message(&AppMessage::KeyDown);
        assert_eq!(effects, vec![AppEffect::None]);
    }

    #[test]
    fn hold_mode_key_up_starts_transcribing() {
        let mut state = default_state();
        state.is_pressed = true;
        let effects = state.handle_message(&AppMessage::KeyUp);
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
        let effects = state.handle_message(&AppMessage::KeyUp);
        assert_eq!(effects, vec![AppEffect::None]);
    }

    // -- Toggle mode (mode = OpenMic) --

    #[test]
    fn toggle_mode_first_key_down_starts_recording() {
        let mut state = default_state();
        state.mode = InputMode::OpenMic;
        let effects = state.handle_message(&AppMessage::KeyDown);
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
        let effects = state.handle_message(&AppMessage::KeyDown);
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
        let effects = state.handle_message(&AppMessage::KeyUp);
        assert_eq!(effects, vec![AppEffect::None]);
    }

    // -- Transcription results --

    #[test]
    fn transcription_done_inserts_text() {
        let mut state = default_state();
        let effects =
            state.handle_message(&AppMessage::TranscriptionDone("hello world".to_string()));
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
        let effects = state.handle_message(&AppMessage::TranscriptionDone("".to_string()));
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
        let effects = state.handle_message(&AppMessage::TranscriptionDone("hello.".to_string()));
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::InsertText(t) if t == "hello.")));
    }

    #[test]
    fn transcription_error_sets_error_state() {
        let mut state = default_state();
        let effects = state.handle_message(&AppMessage::TranscriptionError("fail".to_string()));
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
        let effects = state.handle_message(&AppMessage::TrayCopyLast);
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::CopyToClipboard(t) if t == "copied text")));
    }

    #[test]
    fn copy_last_without_transcription() {
        let mut state = default_state();
        let effects = state.handle_message(&AppMessage::TrayCopyLast);
        assert_eq!(effects, vec![AppEffect::None]);
    }

    // -- Tray actions --

    #[test]
    fn quit_returns_quit_effect() {
        let mut state = default_state();
        let effects = state.handle_message(&AppMessage::TrayQuit);
        assert_eq!(effects, vec![AppEffect::Quit]);
    }

    #[test]
    fn set_model_updates_state_and_saves() {
        let mut state = default_state();
        let effects = state.handle_message(&AppMessage::TraySetModel("small.en".to_string()));
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
        let effects = state.handle_message(&AppMessage::TraySetLanguage("fr".to_string()));
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
        state.handle_message(&AppMessage::TraySetModel("small".to_string()));
        assert_eq!(state.reload_generation, 1);
        state.handle_message(&AppMessage::TraySetLanguage("fr".to_string()));
        assert_eq!(state.reload_generation, 2);
        state.handle_message(&AppMessage::TraySetModel("tiny".to_string()));
        assert_eq!(state.reload_generation, 3);
    }

    #[test]
    fn toggle_spoken_punctuation() {
        let mut state = default_state();
        assert!(!state.spoken_punctuation);
        let effects = state.handle_message(&AppMessage::TrayToggleSpokenPunctuation);
        assert!(state.spoken_punctuation);
        assert!(effects.contains(&AppEffect::SaveConfig));
        let effects = state.handle_message(&AppMessage::TrayToggleSpokenPunctuation);
        assert!(!state.spoken_punctuation);
        assert!(effects.contains(&AppEffect::SaveConfig));
    }

    #[test]
    fn toggle_toggle_mode() {
        let mut state = default_state();
        assert_eq!(state.mode, InputMode::PushToTalk);
        let effects = state.handle_message(&AppMessage::TraySetMode(InputMode::OpenMic));
        assert_eq!(state.mode, InputMode::OpenMic);
        assert!(effects.contains(&AppEffect::SaveConfig));
        assert!(effects.contains(&AppEffect::SetTrayMode(InputMode::OpenMic)));
    }

    #[test]
    fn toggle_translate() {
        let mut state = default_state();
        assert!(!state.translate_to_english);
        let effects = state.handle_message(&AppMessage::TrayToggleTranslate);
        assert!(state.translate_to_english);
        assert!(effects.contains(&AppEffect::SaveConfig));
    }

    // -- AppState construction --

    #[test]
    fn app_state_from_config() {
        let config = Config {
            hotkey: "f9".to_string(),
            model_size: "small.en".to_string(),
            language: "fr".to_string(),
            spoken_punctuation: true,
            filler_word_removal: true,
            max_recordings: 10,
            mode: InputMode::OpenMic,
            streaming: true,
            translate_to_english: true,
            ..Config::default()
        };
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
        assert_eq!(cfg.model_size, "large");
        assert_eq!(cfg.language, "de");
        assert!(cfg.spoken_punctuation);
        assert_eq!(cfg.mode, InputMode::OpenMic);
        assert!(cfg.streaming);
        assert!(cfg.translate_to_english);
        assert_eq!(cfg.hotkey, base.hotkey);
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
        let effects = state.handle_message(&AppMessage::TrayOpenConfig);
        assert_eq!(effects, vec![AppEffect::OpenConfig]);
    }

    #[test]
    fn reload_config_returns_effect() {
        let mut state = default_state();
        let effects = state.handle_message(&AppMessage::TrayReloadConfig);
        assert_eq!(effects, vec![AppEffect::ReloadConfig]);
    }

    #[test]
    fn streaming_partial_text_produces_replace() {
        let mut state = default_state();
        let effects = state.handle_message(&AppMessage::StreamingPartialText {
            text: "hello".to_string(),
            replace_chars: 0,
        });
        assert!(effects.iter().any(|e| matches!(
            e, AppEffect::StreamingReplace { text, replace_chars }
            if text == "hello" && *replace_chars == 0
        )));
    }

    #[test]
    fn streaming_partial_text_empty_noop() {
        let mut state = default_state();
        let effects = state.handle_message(&AppMessage::StreamingPartialText {
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
            state.handle_message(&AppMessage::TranscriptionDone("hello world".to_string()));
        // Should replace the streaming text with the final transcription
        assert!(effects.iter().any(|e| matches!(
            e, AppEffect::StreamingReplace { text, replace_chars }
            if text == "hello world" && *replace_chars == 9
        )));
        // Should NOT contain InsertText (replaced via StreamingReplace)
        assert!(!effects
            .iter()
            .any(|e| matches!(e, AppEffect::InsertText(_))));
        // Should still save last_transcription for copy-last
        assert_eq!(state.last_transcription, Some("hello world".to_string()));
        assert!(!state.streaming_active);
        assert_eq!(state.streaming_chars_emitted, 0);
    }

    #[test]
    fn transcription_done_inserts_when_not_streaming() {
        let mut state = default_state();
        state.streaming_active = false;
        let effects = state.handle_message(&AppMessage::TranscriptionDone("hello".to_string()));
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
        let effects = state.handle_message(&AppMessage::TraySetHotkey);
        assert!(state.capturing_hotkey);
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::EnterHotkeyCaptureMode)));
    }

    #[test]
    fn hotkey_capture_sets_hotkey() {
        let mut state = default_state();
        state.capturing_hotkey = true;
        let effects = state.handle_message(&AppMessage::HotkeyCapture(rdev::Key::F5));
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
        let effects = state.handle_message(&AppMessage::KeyDown);
        assert_eq!(effects, vec![AppEffect::None]);
    }

    #[test]
    fn key_up_ignored_during_capture() {
        let mut state = default_state();
        state.capturing_hotkey = true;
        let effects = state.handle_message(&AppMessage::KeyUp);
        assert_eq!(effects, vec![AppEffect::None]);
    }

    // -- Streaming mode effects --

    #[test]
    fn streaming_key_down_includes_start_streaming() {
        let mut state = default_state();
        state.streaming = true;
        let effects = state.handle_message(&AppMessage::KeyDown);
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
        let effects = state.handle_message(&AppMessage::KeyUp);
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::StopStreaming)));
    }

    #[test]
    fn streaming_partial_text_with_replace_chars() {
        let mut state = default_state();
        let effects = state.handle_message(&AppMessage::StreamingPartialText {
            text: "world".to_string(),
            replace_chars: 5,
        });
        assert!(effects.iter().any(|e| matches!(
            e, AppEffect::StreamingReplace { text, replace_chars }
            if text == "world" && *replace_chars == 5
        )));
    }

    #[test]
    fn streaming_partial_text_empty_text_with_replace_is_not_noop() {
        let mut state = default_state();
        let effects = state.handle_message(&AppMessage::StreamingPartialText {
            text: String::new(),
            replace_chars: 3,
        });
        // replace_chars > 0, so should produce StreamingReplace even with empty text
        assert!(effects.iter().any(|e| matches!(
            e, AppEffect::StreamingReplace { replace_chars, .. } if *replace_chars == 3
        )));
    }

    // -- Transcription results during recording --

    #[test]
    fn streaming_partial_text_tracks_emitted_chars() {
        let mut state = default_state();
        state.handle_message(&AppMessage::StreamingPartialText {
            text: "hello".to_string(),
            replace_chars: 0,
        });
        assert_eq!(state.streaming_chars_emitted, 5);

        // Second partial replaces with longer text
        state.handle_message(&AppMessage::StreamingPartialText {
            text: "hello world".to_string(),
            replace_chars: 5,
        });
        assert_eq!(state.streaming_chars_emitted, 11);
    }

    #[test]
    fn transcription_done_during_recording_does_not_reset_tray() {
        let mut state = default_state();
        state.is_pressed = true;
        let effects = state.handle_message(&AppMessage::TranscriptionDone(
            "from prev cycle".to_string(),
        ));
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
        let effects = state.handle_message(&AppMessage::TranscriptionError("timeout".to_string()));
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
        state.handle_message(&AppMessage::TranscriptionDone("text".to_string()));
        // streaming_active should NOT be cleared since we're still recording
        assert!(state.streaming_active);
    }

    // -- Toggle streaming state --

    #[test]
    fn toggle_streaming() {
        let mut state = default_state();
        assert!(!state.streaming);
        let effects = state.handle_message(&AppMessage::TrayToggleStreaming);
        assert!(state.streaming);
        assert!(effects.contains(&AppEffect::SaveConfig));
        let effects = state.handle_message(&AppMessage::TrayToggleStreaming);
        assert!(!state.streaming);
        assert!(effects.contains(&AppEffect::SaveConfig));
    }

    // -- to_config preserves base config fields --

    #[test]
    fn to_config_preserves_base_hotkey_and_max_recordings() {
        let base = Config {
            hotkey: "ctrl+shift+space".to_string(),
            max_recordings: 42,
            ..Config::default()
        };
        let state = AppState::new(&base);
        let cfg = state.to_config(&base);
        assert_eq!(cfg.hotkey, "ctrl+shift+space");
        assert_eq!(cfg.max_recordings, 42);
    }

    // -- TranscriberReady exhaustiveness --
    // TranscriberReady is handled in the run() loop, not handle_message().
    // Exhaustiveness is verified at compile time. No runtime test needed.

    // -- Open mic streaming --

    #[test]
    fn open_mic_streaming_start_stop() {
        let mut state = default_state();
        state.mode = InputMode::OpenMic;
        state.streaming = true;

        // First press: start recording + streaming
        let effects = state.handle_message(&AppMessage::KeyDown);
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::StartStreaming)));
        assert!(state.streaming_active);

        // Second press: stop recording + streaming
        let effects = state.handle_message(&AppMessage::KeyDown);
        assert!(effects
            .iter()
            .any(|e| matches!(e, AppEffect::StopStreaming)));
    }

    // -- Multiple transcription results update last_transcription --

    #[test]
    fn last_transcription_updated_on_each_result() {
        let mut state = default_state();
        state.handle_message(&AppMessage::TranscriptionDone("first".to_string()));
        assert_eq!(state.last_transcription, Some("first".to_string()));
        state.handle_message(&AppMessage::TranscriptionDone("second".to_string()));
        assert_eq!(state.last_transcription, Some("second".to_string()));
    }

    // -- English-only model enforcement --

    #[test]
    fn set_english_only_model_forces_language_to_english() {
        let mut state = default_state();
        state.model_size = "base".to_string();
        state.language = "fr".to_string();
        let effects =
            state.handle_message(&AppMessage::TraySetModel("distil-large-v3".to_string()));
        assert_eq!(state.language, "en");
        assert!(effects.contains(&AppEffect::SetTrayLanguage("en".to_string())));
        assert!(effects.contains(&AppEffect::SetLanguageMenuEnabled(false)));
    }

    #[test]
    fn set_en_model_disables_language_menu() {
        let mut state = default_state();
        state.model_size = "base".to_string();
        let effects = state.handle_message(&AppMessage::TraySetModel("small.en".to_string()));
        assert!(effects.contains(&AppEffect::SetLanguageMenuEnabled(false)));
    }

    #[test]
    fn set_multilingual_model_enables_language_menu() {
        let mut state = default_state();
        let effects = state.handle_message(&AppMessage::TraySetModel("large".to_string()));
        assert!(effects.contains(&AppEffect::SetLanguageMenuEnabled(true)));
    }

    #[test]
    fn language_change_blocked_for_english_only_model() {
        let mut state = default_state();
        state.model_size = "distil-large-v3".to_string();
        state.language = "en".to_string();
        let effects = state.handle_message(&AppMessage::TraySetLanguage("fr".to_string()));
        assert_eq!(state.language, "en");
        assert!(!effects.contains(&AppEffect::SaveConfig));
        assert!(effects.contains(&AppEffect::SetTrayLanguage("en".to_string())));
    }

    #[test]
    fn language_change_allowed_for_multilingual_model() {
        let mut state = default_state();
        state.model_size = "large".to_string();
        let effects = state.handle_message(&AppMessage::TraySetLanguage("fr".to_string()));
        assert_eq!(state.language, "fr");
        assert!(effects.contains(&AppEffect::SaveConfig));
    }
}
