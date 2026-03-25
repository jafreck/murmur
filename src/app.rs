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

/// Pump the macOS run loop for `seconds`, allowing Cocoa to process events
/// (tray icon rendering, menu interactions, etc.).
#[cfg(target_os = "macos")]
fn pump_event_loop() {
    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        static kCFRunLoopDefaultMode: *const std::ffi::c_void;
        fn CFRunLoopRunInMode(
            mode: *const std::ffi::c_void,
            seconds: f64,
            return_after_source_handled: u8,
        ) -> i32;
    }
    unsafe {
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 1.0 / 60.0, 0);
    }
}

/// On non-macOS platforms, just sleep briefly.
#[cfg(not(target_os = "macos"))]
fn pump_event_loop() {
    std::thread::sleep(std::time::Duration::from_millis(16));
}

enum AppMessage {
    KeyDown,
    KeyUp,
    TrayQuit,
    TrayCopyLast,
    TraySetModel(String),
    TraySetLanguage(String),
    TrayToggleSpokenPunctuation,
    TrayToggleToggleMode,
    TrayToggleTranslate,
    TranscriptionDone(String),
    TranscriptionError(String),
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

    // Parse hotkey
    let parsed = crate::keycodes::parse(&config.hotkey)
        .ok_or_else(|| anyhow::anyhow!("Invalid hotkey: {}", config.hotkey))?;

    // Log platform-specific permission hints
    crate::permissions::check_accessibility();
    crate::permissions::check_microphone();

    // Initialize macOS NSApplication (required for tray icon rendering)
    #[cfg(target_os = "macos")]
    init_macos_app();

    // Create tray on the main thread
    let mut tray = TrayController::new(&config)?;

    // Channel for all app events
    let (tx, rx) = mpsc::channel::<AppMessage>();
    let tx_down = tx.clone();
    let tx_up = tx.clone();

    // Start hotkey listener in a background thread
    let hotkey_key = parsed.key;
    std::thread::spawn(move || {
        if let Err(e) = HotkeyManager::start(
            hotkey_key,
            move || {
                let _ = tx_down.send(AppMessage::KeyDown);
            },
            move || {
                let _ = tx_up.send(AppMessage::KeyUp);
            },
        ) {
            error!("Hotkey listener failed: {e}");
        }
    });

    // Main event loop — owns the AudioRecorder (which is !Send)
    let mut recorder = AudioRecorder::new();
    let mut is_pressed = false;
    let mut last_transcription: Option<String> = None;
    let mut toggle_mode = config.toggle_mode;
    let mut spoken_punctuation = config.spoken_punctuation;
    let mut translate_to_english = config.translate_to_english;
    let max_recordings = Config::effective_max_recordings(config.max_recordings);

    println!("open-bark v{VERSION}");
    println!("Hotkey: {}", config.hotkey);
    println!("Model: {}", config.model_size);
    println!("Ready.");

    // Unified event loop polling three receivers
    loop {
        // 1. Drain app message channel (non-blocking)
        while let Ok(msg) = rx.try_recv() {
            match msg {
                AppMessage::KeyDown => {
                    if toggle_mode {
                        if is_pressed {
                            start_transcribing(
                                &mut recorder,
                                &mut is_pressed,
                                &transcriber,
                                spoken_punctuation,
                                translate_to_english,
                                max_recordings,
                                tx.clone(),
                            );
                            tray.set_state(TrayState::Transcribing);
                        } else {
                            handle_start(&mut recorder, &mut is_pressed, max_recordings);
                            tray.set_state(TrayState::Recording);
                        }
                    } else if !is_pressed {
                        handle_start(&mut recorder, &mut is_pressed, max_recordings);
                        tray.set_state(TrayState::Recording);
                    }
                }
                AppMessage::KeyUp => {
                    if !toggle_mode && is_pressed {
                        start_transcribing(
                            &mut recorder,
                            &mut is_pressed,
                            &transcriber,
                            spoken_punctuation,
                            translate_to_english,
                            max_recordings,
                            tx.clone(),
                        );
                        tray.set_state(TrayState::Transcribing);
                    }
                }
                AppMessage::TranscriptionDone(text) => {
                    if !text.is_empty() {
                        info!("Transcription: {text}");
                        if let Err(e) = TextInserter::insert(&text) {
                            error!("Insert failed: {e}");
                        }
                        last_transcription = Some(text);
                    }
                    tray.set_state(TrayState::Idle);
                }
                AppMessage::TranscriptionError(e) => {
                    error!("Transcription: {e}");
                    tray.set_state(TrayState::Error);
                }
                AppMessage::TrayCopyLast => {
                    if let Some(ref text) = last_transcription {
                        if let Ok(mut cb) = arboard::Clipboard::new() {
                            let _ = cb.set_text(text.clone());
                            info!("Copied last dictation to clipboard");
                        }
                    }
                }
                AppMessage::TrayQuit => {
                    info!("Quit requested via tray");
                    return Ok(());
                }
                AppMessage::TraySetModel(size) => {
                    info!("Model changed to: {size}");
                    config.model_size = size.clone();
                    if let Err(e) = config.save() {
                        error!("Failed to save config: {e}");
                    }
                    tray.set_model(&size);
                    info!("Restart open-bark for model change to take effect");
                }
                AppMessage::TraySetLanguage(code) => {
                    let name = crate::config::language_name(&code).unwrap_or(&code);
                    info!("Language changed to: {name} ({code})");
                    config.language = code.clone();
                    if let Err(e) = config.save() {
                        error!("Failed to save config: {e}");
                    }
                    tray.set_language(&code);
                    info!("Restart open-bark for language change to take effect");
                }
                AppMessage::TrayToggleSpokenPunctuation => {
                    spoken_punctuation = !spoken_punctuation;
                    config.spoken_punctuation = spoken_punctuation;
                    if let Err(e) = config.save() {
                        error!("Failed to save config: {e}");
                    }
                    info!("Spoken punctuation: {}", if spoken_punctuation { "on" } else { "off" });
                }
                AppMessage::TrayToggleToggleMode => {
                    toggle_mode = !toggle_mode;
                    config.toggle_mode = toggle_mode;
                    if let Err(e) = config.save() {
                        error!("Failed to save config: {e}");
                    }
                    info!("Toggle mode: {}", if toggle_mode { "on" } else { "off" });
                }
                AppMessage::TrayToggleTranslate => {
                    translate_to_english = !translate_to_english;
                    config.translate_to_english = translate_to_english;
                    if let Err(e) = config.save() {
                        error!("Failed to save config: {e}");
                    }
                    info!("Translate to English: {}", if translate_to_english { "on" } else { "off" });
                }
            }
        }

        // 2. Drain tray menu events (non-blocking)
        while let Ok(event) = MenuEvent::receiver().try_recv() {
            if let Some(action) = tray.match_menu_event(&event) {
                match action {
                    TrayAction::Quit => {
                        let _ = tx.send(AppMessage::TrayQuit);
                    }
                    TrayAction::CopyLastDictation => {
                        let _ = tx.send(AppMessage::TrayCopyLast);
                    }
                    TrayAction::SetModel(size) => {
                        let _ = tx.send(AppMessage::TraySetModel(size));
                    }
                    TrayAction::SetLanguage(code) => {
                        let _ = tx.send(AppMessage::TraySetLanguage(code));
                    }
                    TrayAction::ToggleSpokenPunctuation => {
                        let _ = tx.send(AppMessage::TrayToggleSpokenPunctuation);
                    }
                    TrayAction::ToggleToggleMode => {
                        let _ = tx.send(AppMessage::TrayToggleToggleMode);
                    }
                    TrayAction::ToggleTranslate => {
                        let _ = tx.send(AppMessage::TrayToggleTranslate);
                    }
                }
            }
        }

        // 3. Drain tray icon events (click handling)
        while let Ok(_event) = TrayIconEvent::receiver().try_recv() {
            // Future: left-click to toggle recording, etc.
        }

        // Pump the platform event loop (~60 Hz)
        pump_event_loop();
    }
}

fn handle_start(recorder: &mut AudioRecorder, is_pressed: &mut bool, max_recordings: u32) {
    if *is_pressed {
        return;
    }
    *is_pressed = true;
    info!("Recording...");

    let output_path = if max_recordings == 0 {
        RecordingStore::temp_recording_path()
    } else {
        RecordingStore::new_recording_path()
    };

    if let Err(e) = recorder.start(&output_path) {
        error!("Failed to start recording: {e}");
        *is_pressed = false;
    }
}

fn start_transcribing(
    recorder: &mut AudioRecorder,
    is_pressed: &mut bool,
    transcriber: &Arc<Transcriber>,
    spoken_punctuation: bool,
    translate_to_english: bool,
    max_recordings: u32,
    tx: mpsc::Sender<AppMessage>,
) {
    if !*is_pressed {
        return;
    }
    *is_pressed = false;

    let Some(audio_path) = recorder.stop() else {
        return;
    };

    info!("Transcribing...");

    let transcriber = Arc::clone(transcriber);
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
