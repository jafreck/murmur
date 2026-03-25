use anyhow::Result;
use log::{error, info};
use std::sync::mpsc;
use std::sync::Arc;

use crate::audio::AudioRecorder;
use crate::config::Config;
use crate::hotkey::HotkeyManager;
use crate::inserter::TextInserter;
use crate::model;
use crate::postprocess;
use crate::recordings::RecordingStore;
use crate::streaming;
use crate::transcriber::Transcriber;
use crate::tray::{TrayAction, TrayController, TrayState};
use crate::VERSION;

use tray_icon::menu::MenuEvent;
use tray_icon::TrayIconEvent;

/// On macOS, initialize NSApplication with Accessory policy (no dock icon)
/// so the system tray icon renders and Cocoa events are dispatched.
#[cfg(target_os = "macos")]
fn init_macos_app() {
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy};

    let mtm = MainThreadMarker::new()
        .expect("init_macos_app must be called on the main thread");
    let app = NSApplication::sharedApplication(mtm);
    app.setActivationPolicy(NSApplicationActivationPolicy::Accessory);
}

/// Pump the macOS AppKit event loop, dispatching all pending Cocoa events
/// (tray icon clicks, menu interactions, rendering, etc.).
#[cfg(target_os = "macos")]
fn pump_event_loop() {
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSApplication, NSEventMask};
    use objc2_foundation::{NSDate, NSDefaultRunLoopMode};

    let mtm = MainThreadMarker::new().expect("must be on main thread");
    let app = NSApplication::sharedApplication(mtm);

    loop {
        let event = unsafe {
            app.nextEventMatchingMask_untilDate_inMode_dequeue(
                NSEventMask::Any,
                Some(&NSDate::distantPast()),
                NSDefaultRunLoopMode,
                true,
            )
        };
        match event {
            Some(event) => app.sendEvent(&event),
            None => break,
        }
    }

    std::thread::sleep(std::time::Duration::from_millis(16));
}

/// On non-macOS platforms, just sleep briefly.
#[cfg(not(target_os = "macos"))]
fn pump_event_loop() {
    std::thread::sleep(std::time::Duration::from_millis(16));
}

pub enum AppMessage {
    KeyDown,
    KeyUp,
    TrayQuit,
    TrayCopyLast,
    TraySetModel(String),
    TraySetLanguage(String),
    TrayToggleSpokenPunctuation,
    TrayTogglePushToTalk,
    TrayToggleStreaming,
    TrayToggleTranslate,
    TrayOpenConfig,
    TrayReloadConfig,
    TranscriptionDone(String),
    TranscriptionError(String),
    StreamingPartialText(String),
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
    CopyToClipboard(String),
    SaveConfig,
    SetTrayState(TrayStateTag),
    SetTrayModel(String),
    SetTrayLanguage(String),
    Quit,
    LogError(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrayStateTag {
    Idle,
    Recording,
    Transcribing,
    Error,
}

/// Pure state machine for the app's recording logic.
pub struct AppState {
    pub is_pressed: bool,
    pub push_to_talk: bool,
    pub streaming: bool,
    pub spoken_punctuation: bool,
    pub translate_to_english: bool,
    pub max_recordings: u32,
    pub last_transcription: Option<String>,
    pub model_size: String,
    pub language: String,
}

impl AppState {
    pub fn new(config: &Config) -> Self {
        Self {
            is_pressed: false,
            push_to_talk: config.push_to_talk,
            streaming: config.streaming,
            spoken_punctuation: config.spoken_punctuation,
            translate_to_english: config.translate_to_english,
            max_recordings: Config::effective_max_recordings(config.max_recordings),
            last_transcription: None,
            model_size: config.model_size.clone(),
            language: config.language.clone(),
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
            AppMessage::KeyDown => self.on_key_down(),
            AppMessage::KeyUp => self.on_key_up(),
            AppMessage::TranscriptionDone(text) => self.on_transcription_done(text),
            AppMessage::TranscriptionError(e) => self.on_transcription_error(e),
            AppMessage::TrayCopyLast => self.on_copy_last(),
            AppMessage::TrayQuit => vec![AppEffect::Quit],
            AppMessage::TraySetModel(size) => self.on_set_model(size),
            AppMessage::TraySetLanguage(code) => self.on_set_language(code),
            AppMessage::TrayToggleSpokenPunctuation => self.on_toggle_spoken_punctuation(),
            AppMessage::TrayTogglePushToTalk => self.on_toggle_push_to_talk(),
            AppMessage::TrayToggleStreaming => self.on_toggle_streaming(),
            AppMessage::TrayToggleTranslate => self.on_toggle_translate(),
            AppMessage::TrayOpenConfig | AppMessage::TrayReloadConfig => vec![AppEffect::None],
            AppMessage::StreamingPartialText(text) => {
                if !text.is_empty() {
                    vec![AppEffect::InsertText(text.clone())]
                } else {
                    vec![AppEffect::None]
                }
            }
        }
    }

    fn on_key_down(&mut self) -> Vec<AppEffect> {
        if !self.push_to_talk {
            if self.is_pressed {
                self.is_pressed = false;
                let mut effects = vec![];
                if self.streaming {
                    effects.push(AppEffect::StopStreaming);
                }
                effects.push(AppEffect::StopAndTranscribe);
                effects.push(AppEffect::SetTrayState(TrayStateTag::Transcribing));
                effects
            } else {
                self.is_pressed = true;
                let path = self.recording_output_path();
                let mut effects = vec![AppEffect::StartRecording(path)];
                if self.streaming {
                    effects.push(AppEffect::StartStreaming);
                }
                effects.push(AppEffect::SetTrayState(TrayStateTag::Recording));
                effects
            }
        } else if !self.is_pressed {
            self.is_pressed = true;
            let path = self.recording_output_path();
            let mut effects = vec![AppEffect::StartRecording(path)];
            if self.streaming {
                effects.push(AppEffect::StartStreaming);
            }
            effects.push(AppEffect::SetTrayState(TrayStateTag::Recording));
            effects
        } else {
            vec![AppEffect::None]
        }
    }

    fn on_key_up(&mut self) -> Vec<AppEffect> {
        if self.push_to_talk && self.is_pressed {
            self.is_pressed = false;
            let mut effects = vec![];
            if self.streaming {
                effects.push(AppEffect::StopStreaming);
            }
            effects.push(AppEffect::StopAndTranscribe);
            effects.push(AppEffect::SetTrayState(TrayStateTag::Transcribing));
            effects
        } else {
            vec![AppEffect::None]
        }
    }

    fn on_transcription_done(&mut self, text: &str) -> Vec<AppEffect> {
        let mut effects = vec![];
        if !text.is_empty() {
            let processed = if self.spoken_punctuation {
                crate::postprocess::process(text)
            } else {
                text.to_string()
            };
            effects.push(AppEffect::InsertText(processed.clone()));
            self.last_transcription = Some(processed);
        }
        effects.push(AppEffect::SetTrayState(TrayStateTag::Idle));
        effects
    }

    fn on_transcription_error(&mut self, error: &str) -> Vec<AppEffect> {
        vec![
            AppEffect::LogError(error.to_string()),
            AppEffect::SetTrayState(TrayStateTag::Error),
        ]
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
        vec![
            AppEffect::SaveConfig,
            AppEffect::SetTrayModel(size.to_string()),
        ]
    }

    fn on_set_language(&mut self, code: &str) -> Vec<AppEffect> {
        self.language = code.to_string();
        vec![
            AppEffect::SaveConfig,
            AppEffect::SetTrayLanguage(code.to_string()),
        ]
    }

    fn on_toggle_spoken_punctuation(&mut self) -> Vec<AppEffect> {
        self.spoken_punctuation = !self.spoken_punctuation;
        vec![AppEffect::SaveConfig]
    }

    fn on_toggle_push_to_talk(&mut self) -> Vec<AppEffect> {
        self.push_to_talk = !self.push_to_talk;
        vec![AppEffect::SaveConfig]
    }

    fn on_toggle_streaming(&mut self) -> Vec<AppEffect> {
        self.streaming = !self.streaming;
        vec![AppEffect::SaveConfig]
    }

    fn on_toggle_translate(&mut self) -> Vec<AppEffect> {
        self.translate_to_english = !self.translate_to_english;
        vec![AppEffect::SaveConfig]
    }

    pub fn to_config(&self, base: &Config) -> Config {
        Config {
            hotkey: base.hotkey.clone(),
            model_size: self.model_size.clone(),
            language: self.language.clone(),
            spoken_punctuation: self.spoken_punctuation,
            max_recordings: base.max_recordings,
            push_to_talk: self.push_to_talk,
            streaming: self.streaming,
            translate_to_english: self.translate_to_english,
        }
    }
}

/// Convert a TrayAction to the corresponding AppMessage.
pub fn tray_action_to_message(action: TrayAction) -> AppMessage {
    match action {
        TrayAction::Quit => AppMessage::TrayQuit,
        TrayAction::CopyLastDictation => AppMessage::TrayCopyLast,
        TrayAction::SetModel(s) => AppMessage::TraySetModel(s),
        TrayAction::SetLanguage(c) => AppMessage::TraySetLanguage(c),
        TrayAction::ToggleSpokenPunctuation => AppMessage::TrayToggleSpokenPunctuation,
        TrayAction::TogglePushToTalk => AppMessage::TrayTogglePushToTalk,
        TrayAction::ToggleStreaming => AppMessage::TrayToggleStreaming,
        TrayAction::ToggleTranslate => AppMessage::TrayToggleTranslate,
        TrayAction::OpenConfig => AppMessage::TrayOpenConfig,
        TrayAction::ReloadConfig => AppMessage::TrayReloadConfig,
    }
}

/// Convert a TrayStateTag to the corresponding TrayState.
pub fn tag_to_tray_state(tag: &TrayStateTag) -> TrayState {
    match tag {
        TrayStateTag::Idle => TrayState::Idle,
        TrayStateTag::Recording => TrayState::Recording,
        TrayStateTag::Transcribing => TrayState::Transcribing,
        TrayStateTag::Error => TrayState::Error,
    }
}

pub fn run() -> Result<()> {
    let mut config = Config::load();

    // Ensure model is available
    if !crate::transcriber::model_exists(&config.model_size) {
        info!("Downloading {} model...", config.model_size);
        let model_size = config.model_size.clone();
        model::download(&model_size, |percent| {
            eprint!("\rDownloading {model_size} model... {percent:.0}%");
        })?;
        eprintln!();
    }

    let model_path = crate::transcriber::find_model(&config.model_size)
        .ok_or_else(|| anyhow::anyhow!("Model '{}' not found after download", config.model_size))?;

    let transcriber = Arc::new(Transcriber::new(&model_path, &config.language)?);

    info!("Hotkey: {}", config.hotkey);
    info!("Model: {}", config.model_size);

    let parsed = crate::keycodes::parse(&config.hotkey)
        .ok_or_else(|| anyhow::anyhow!("Invalid hotkey: {}", config.hotkey))?;

    crate::permissions::check_accessibility();
    crate::permissions::check_microphone();

    #[cfg(target_os = "macos")]
    init_macos_app();

    let mut tray = TrayController::new(&config)?;

    let (tx, rx) = mpsc::channel::<AppMessage>();
    let tx_down = tx.clone();
    let tx_up = tx.clone();

    let hotkey_key = parsed.key;
    std::thread::spawn(move || {
        if let Err(e) = HotkeyManager::start(
            hotkey_key,
            move || { let _ = tx_down.send(AppMessage::KeyDown); },
            move || { let _ = tx_up.send(AppMessage::KeyUp); },
        ) {
            error!("Hotkey listener failed: {e}");
        }
    });

    let mut recorder = AudioRecorder::new();
    let mut state = AppState::new(&config);
    let mut streaming_stop: Option<mpsc::Sender<()>> = None;

    println!("open-bark v{VERSION}");
    println!("Hotkey: {}", config.hotkey);
    println!("Model: {}", config.model_size);
    println!("Ready.");

    loop {
        while let Ok(msg) = rx.try_recv() {
            let effects = state.handle_message(&msg);
            for effect in effects {
                apply_effect(
                    effect, &mut recorder, &transcriber,
                    &mut tray, &mut config, &state, &tx,
                    &mut streaming_stop,
                )?;
            }
        }

        while let Ok(event) = MenuEvent::receiver().try_recv() {
            if let Some(action) = tray.match_menu_event(&event) {
                let _ = tx.send(tray_action_to_message(action));
            }
        }

        while let Ok(_event) = TrayIconEvent::receiver().try_recv() {}

        pump_event_loop();
    }
}

fn apply_effect(
    effect: AppEffect,
    recorder: &mut AudioRecorder,
    transcriber: &Arc<Transcriber>,
    tray: &mut TrayController,
    config: &mut Config,
    state: &AppState,
    tx: &mpsc::Sender<AppMessage>,
    streaming_stop: &mut Option<mpsc::Sender<()>>,
) -> Result<()> {
    match effect {
        AppEffect::None => {}
        AppEffect::StartRecording(path) => {
            info!("Recording...");
            if let Err(e) = recorder.start(&path) {
                error!("Failed to start recording: {e}");
            }
        }
        AppEffect::StopAndTranscribe => {
            // Stop streaming first (if running)
            if let Some(stop) = streaming_stop.take() {
                let _ = stop.send(());
            }

            if let Some(audio_path) = recorder.stop() {
                info!("Transcribing...");
                let transcriber = Arc::clone(transcriber);
                let spoken_punctuation = state.spoken_punctuation;
                let translate_to_english = state.translate_to_english;
                let max_recordings = state.max_recordings;
                let tx = tx.clone();
                std::thread::spawn(move || {
                    let result = transcriber.transcribe(&audio_path, translate_to_english);
                    if max_recordings == 0 {
                        let _ = std::fs::remove_file(&audio_path);
                    } else {
                        RecordingStore::prune(max_recordings);
                    }
                    match result {
                        Ok(raw) => {
                            let text = if spoken_punctuation {
                                postprocess::process(&raw)
                            } else {
                                raw
                            };
                            let _ = tx.send(AppMessage::TranscriptionDone(text));
                        }
                        Err(e) => {
                            let _ = tx.send(AppMessage::TranscriptionError(e.to_string()));
                        }
                    }
                });
            }
        }
        AppEffect::StartStreaming => {
            info!("Starting streaming transcription...");
            let sample_buffer = recorder.sample_buffer();
            let transcriber = Arc::clone(transcriber);
            let tx_app = tx.clone();
            let language = state.language.clone();
            let translate = state.translate_to_english;
            let spoken_punct = state.spoken_punctuation;

            let (streaming_tx, streaming_rx) = mpsc::channel::<streaming::StreamingEvent>();

            // Forward streaming events to app messages
            std::thread::spawn(move || {
                while let Ok(event) = streaming_rx.recv() {
                    match event {
                        streaming::StreamingEvent::PartialText(text) => {
                            let _ = tx_app.send(AppMessage::StreamingPartialText(text));
                        }
                    }
                }
            });

            let stop = streaming::start_streaming(
                sample_buffer,
                transcriber,
                language,
                translate,
                spoken_punct,
                streaming_tx,
            );
            *streaming_stop = Some(stop);
        }
        AppEffect::StopStreaming => {
            if let Some(stop) = streaming_stop.take() {
                info!("Stopping streaming transcription");
                let _ = stop.send(());
            }
        }
        AppEffect::InsertText(text) => {
            info!("Transcription: {text}");
            if let Err(e) = TextInserter::insert(&text) {
                error!("Insert failed: {e}");
            }
        }
        AppEffect::CopyToClipboard(text) => {
            if let Ok(mut cb) = arboard::Clipboard::new() {
                let _ = cb.set_text(text);
                info!("Copied last dictation to clipboard");
            }
        }
        AppEffect::SaveConfig => {
            let new_config = state.to_config(config);
            *config = new_config;
            if let Err(e) = config.save() {
                error!("Failed to save config: {e}");
            }
        }
        AppEffect::SetTrayState(tag) => {
            tray.set_state(tag_to_tray_state(&tag));
        }
        AppEffect::SetTrayModel(size) => {
            tray.set_model(&size);
            info!("Model changed to: {size}");
        }
        AppEffect::SetTrayLanguage(code) => {
            let name = crate::config::language_name(&code).unwrap_or(&code);
            info!("Language changed to: {name} ({code})");
            tray.set_language(&code);
        }
        AppEffect::Quit => {
            info!("Quit requested via tray");
            std::process::exit(0);
        }
        AppEffect::LogError(e) => {
            error!("Transcription: {e}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_state() -> AppState {
        AppState {
            is_pressed: false,
            push_to_talk: true,
            streaming: false,
            spoken_punctuation: false,
            translate_to_english: false,
            max_recordings: 0,
            last_transcription: None,
            model_size: "base.en".to_string(),
            language: "en".to_string(),
        }
    }

    // -- Hold-to-talk mode (push_to_talk = true) --

    #[test]
    fn hold_mode_key_down_starts_recording() {
        let mut state = default_state();
        let effects = state.handle_message(&AppMessage::KeyDown);
        assert!(state.is_pressed);
        assert!(effects.iter().any(|e| matches!(e, AppEffect::StartRecording(_))));
        assert!(effects.iter().any(|e| matches!(e, AppEffect::SetTrayState(TrayStateTag::Recording))));
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
        assert!(effects.iter().any(|e| matches!(e, AppEffect::StopAndTranscribe)));
        assert!(effects.iter().any(|e| matches!(e, AppEffect::SetTrayState(TrayStateTag::Transcribing))));
    }

    #[test]
    fn hold_mode_key_up_not_pressed_noop() {
        let mut state = default_state();
        let effects = state.handle_message(&AppMessage::KeyUp);
        assert_eq!(effects, vec![AppEffect::None]);
    }

    // -- Toggle mode (push_to_talk = false) --

    #[test]
    fn toggle_mode_first_key_down_starts_recording() {
        let mut state = default_state();
        state.push_to_talk = false;
        let effects = state.handle_message(&AppMessage::KeyDown);
        assert!(state.is_pressed);
        assert!(effects.iter().any(|e| matches!(e, AppEffect::StartRecording(_))));
        assert!(effects.iter().any(|e| matches!(e, AppEffect::SetTrayState(TrayStateTag::Recording))));
    }

    #[test]
    fn toggle_mode_second_key_down_stops_and_transcribes() {
        let mut state = default_state();
        state.push_to_talk = false;
        state.is_pressed = true;
        let effects = state.handle_message(&AppMessage::KeyDown);
        assert!(!state.is_pressed);
        assert!(effects.iter().any(|e| matches!(e, AppEffect::StopAndTranscribe)));
        assert!(effects.iter().any(|e| matches!(e, AppEffect::SetTrayState(TrayStateTag::Transcribing))));
    }

    #[test]
    fn toggle_mode_key_up_noop() {
        let mut state = default_state();
        state.push_to_talk = false;
        state.is_pressed = true;
        let effects = state.handle_message(&AppMessage::KeyUp);
        assert_eq!(effects, vec![AppEffect::None]);
    }

    // -- Transcription results --

    #[test]
    fn transcription_done_inserts_text() {
        let mut state = default_state();
        let effects = state.handle_message(&AppMessage::TranscriptionDone("hello world".to_string()));
        assert!(effects.iter().any(|e| matches!(e, AppEffect::InsertText(t) if t == "hello world")));
        assert!(effects.iter().any(|e| matches!(e, AppEffect::SetTrayState(TrayStateTag::Idle))));
        assert_eq!(state.last_transcription, Some("hello world".to_string()));
    }

    #[test]
    fn transcription_done_empty_no_insert() {
        let mut state = default_state();
        let effects = state.handle_message(&AppMessage::TranscriptionDone("".to_string()));
        assert!(!effects.iter().any(|e| matches!(e, AppEffect::InsertText(_))));
        assert!(effects.iter().any(|e| matches!(e, AppEffect::SetTrayState(TrayStateTag::Idle))));
        assert!(state.last_transcription.is_none());
    }

    #[test]
    fn transcription_done_with_spoken_punctuation() {
        let mut state = default_state();
        state.spoken_punctuation = true;
        let effects = state.handle_message(&AppMessage::TranscriptionDone("hello period".to_string()));
        assert!(effects.iter().any(|e| matches!(e, AppEffect::InsertText(t) if t.contains('.'))));
    }

    #[test]
    fn transcription_error_sets_error_state() {
        let mut state = default_state();
        let effects = state.handle_message(&AppMessage::TranscriptionError("fail".to_string()));
        assert!(effects.iter().any(|e| matches!(e, AppEffect::LogError(t) if t == "fail")));
        assert!(effects.iter().any(|e| matches!(e, AppEffect::SetTrayState(TrayStateTag::Error))));
    }

    // -- Copy last --

    #[test]
    fn copy_last_with_transcription() {
        let mut state = default_state();
        state.last_transcription = Some("copied text".to_string());
        let effects = state.handle_message(&AppMessage::TrayCopyLast);
        assert!(effects.iter().any(|e| matches!(e, AppEffect::CopyToClipboard(t) if t == "copied text")));
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
    }

    #[test]
    fn set_language_updates_state_and_saves() {
        let mut state = default_state();
        let effects = state.handle_message(&AppMessage::TraySetLanguage("fr".to_string()));
        assert_eq!(state.language, "fr");
        assert!(effects.contains(&AppEffect::SaveConfig));
        assert!(effects.contains(&AppEffect::SetTrayLanguage("fr".to_string())));
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
        assert!(state.push_to_talk);
        let effects = state.handle_message(&AppMessage::TrayTogglePushToTalk);
        assert!(!state.push_to_talk);
        assert!(effects.contains(&AppEffect::SaveConfig));
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
            max_recordings: 10,
            push_to_talk: false,
            streaming: true,
            translate_to_english: true,
        };
        let state = AppState::new(&config);
        assert_eq!(state.model_size, "small.en");
        assert_eq!(state.language, "fr");
        assert!(state.spoken_punctuation);
        assert!(!state.push_to_talk);
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
        state.push_to_talk = false;
        state.streaming = true;
        state.translate_to_english = true;
        let cfg = state.to_config(&base);
        assert_eq!(cfg.model_size, "large");
        assert_eq!(cfg.language, "de");
        assert!(cfg.spoken_punctuation);
        assert!(!cfg.push_to_talk);
        assert!(cfg.streaming);
        assert!(cfg.translate_to_english);
        assert_eq!(cfg.hotkey, base.hotkey);
    }

    #[test]
    fn recording_output_path_temp_when_no_max() {
        let state = default_state();
        let path = state.recording_output_path();
        assert!(path.to_string_lossy().contains("open-bark-recording.wav"));
    }

    // -- AppEffect/TrayStateTag enum tests --

    #[test]
    fn app_effect_debug_and_eq() {
        let e1 = AppEffect::None;
        let e2 = AppEffect::None;
        assert_eq!(e1, e2);
        assert_eq!(format!("{:?}", e1), "None");
        assert_ne!(AppEffect::Quit, AppEffect::None);
    }

    #[test]
    fn tray_state_tag_variants() {
        assert_ne!(TrayStateTag::Idle, TrayStateTag::Recording);
        assert_ne!(TrayStateTag::Recording, TrayStateTag::Transcribing);
        assert_ne!(TrayStateTag::Transcribing, TrayStateTag::Error);
        assert_eq!(TrayStateTag::Idle.clone(), TrayStateTag::Idle);
    }

    #[test]
    fn open_config_and_reload_are_noop() {
        let mut state = default_state();
        let effects = state.handle_message(&AppMessage::TrayOpenConfig);
        assert_eq!(effects, vec![AppEffect::None]);
        let effects = state.handle_message(&AppMessage::TrayReloadConfig);
        assert_eq!(effects, vec![AppEffect::None]);
    }

    #[test]
    fn streaming_partial_text_inserts() {
        let mut state = default_state();
        let effects = state.handle_message(&AppMessage::StreamingPartialText("hello".to_string()));
        assert!(effects.iter().any(|e| matches!(e, AppEffect::InsertText(t) if t == "hello")));
    }

    #[test]
    fn streaming_partial_text_empty_noop() {
        let mut state = default_state();
        let effects = state.handle_message(&AppMessage::StreamingPartialText("".to_string()));
        assert_eq!(effects, vec![AppEffect::None]);
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

    // -- tray_action_to_message --

    #[test]
    fn tray_action_to_message_quit() {
        match tray_action_to_message(TrayAction::Quit) {
            AppMessage::TrayQuit => {}
            _ => panic!("expected TrayQuit"),
        }
    }

    #[test]
    fn tray_action_to_message_copy_last() {
        match tray_action_to_message(TrayAction::CopyLastDictation) {
            AppMessage::TrayCopyLast => {}
            _ => panic!("expected TrayCopyLast"),
        }
    }

    #[test]
    fn tray_action_to_message_set_model() {
        match tray_action_to_message(TrayAction::SetModel("base.en".into())) {
            AppMessage::TraySetModel(s) => assert_eq!(s, "base.en"),
            _ => panic!("expected TraySetModel"),
        }
    }

    #[test]
    fn tray_action_to_message_set_language() {
        match tray_action_to_message(TrayAction::SetLanguage("fr".into())) {
            AppMessage::TraySetLanguage(c) => assert_eq!(c, "fr"),
            _ => panic!("expected TraySetLanguage"),
        }
    }

    #[test]
    fn tray_action_to_message_toggle_spoken_punct() {
        match tray_action_to_message(TrayAction::ToggleSpokenPunctuation) {
            AppMessage::TrayToggleSpokenPunctuation => {}
            _ => panic!("expected TrayToggleSpokenPunctuation"),
        }
    }

    #[test]
    fn tray_action_to_message_toggle_mode() {
        match tray_action_to_message(TrayAction::TogglePushToTalk) {
            AppMessage::TrayTogglePushToTalk => {}
            _ => panic!("expected TrayTogglePushToTalk"),
        }
    }

    #[test]
    fn tray_action_to_message_toggle_translate() {
        match tray_action_to_message(TrayAction::ToggleTranslate) {
            AppMessage::TrayToggleTranslate => {}
            _ => panic!("expected TrayToggleTranslate"),
        }
    }

    #[test]
    fn tray_action_to_message_open_config() {
        match tray_action_to_message(TrayAction::OpenConfig) {
            AppMessage::TrayOpenConfig => {}
            _ => panic!("expected TrayOpenConfig"),
        }
    }

    #[test]
    fn tray_action_to_message_reload_config() {
        match tray_action_to_message(TrayAction::ReloadConfig) {
            AppMessage::TrayReloadConfig => {}
            _ => panic!("expected TrayReloadConfig"),
        }
    }

    // -- tag_to_tray_state --

    #[test]
    fn tag_to_tray_state_all() {
        assert_eq!(tag_to_tray_state(&TrayStateTag::Idle), TrayState::Idle);
        assert_eq!(tag_to_tray_state(&TrayStateTag::Recording), TrayState::Recording);
        assert_eq!(tag_to_tray_state(&TrayStateTag::Transcribing), TrayState::Transcribing);
        assert_eq!(tag_to_tray_state(&TrayStateTag::Error), TrayState::Error);
    }
}
