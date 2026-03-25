use anyhow::Result;
use log::{error, info};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

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

enum AppMessage {
    KeyDown,
    KeyUp,
    TrayQuit,
    TrayCopyLast,
    TranscriptionDone(String),
    TranscriptionError(String),
}

pub fn run() -> Result<()> {
    let config = Config::load();

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

    // Create tray on the main thread
    let mut tray = TrayController::new()?;

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
    let toggle_mode = config.toggle_mode;
    let spoken_punctuation = config.spoken_punctuation;
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
                }
            }
        }

        // 3. Drain tray icon events (click handling)
        while let Ok(_event) = TrayIconEvent::receiver().try_recv() {
            // Future: left-click to toggle recording, etc.
        }

        // Sleep to avoid busy-looping (~60 iterations/sec)
        std::thread::sleep(Duration::from_millis(16));
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
        let result = transcriber.transcribe(&audio_path);

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
