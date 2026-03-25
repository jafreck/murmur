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
use crate::VERSION;

#[derive(Debug, Clone, PartialEq)]
pub enum AppState {
    Idle,
    Recording,
    Transcribing,
    Downloading(String),
    Error(String),
}

enum AppMessage {
    KeyDown,
    KeyUp,
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

    println!("open-bark v{VERSION}");
    println!("Hotkey: {}", config.hotkey);
    println!("Model: {}", config.model_size);
    println!("Ready.");

    // Channel for hotkey events → main loop
    let (tx, rx) = mpsc::channel::<AppMessage>();
    let tx_down = tx.clone();
    let tx_up = tx;

    // Start hotkey listener in a background thread
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

    // Main event loop — owns the AudioRecorder (which is !Send)
    let mut recorder = AudioRecorder::new();
    let mut is_pressed = false;
    let toggle_mode = config.toggle_mode;
    let spoken_punctuation = config.spoken_punctuation;
    let max_recordings = Config::effective_max_recordings(config.max_recordings);

    for msg in rx {
        match msg {
            AppMessage::KeyDown => {
                if toggle_mode {
                    if is_pressed {
                        handle_stop(
                            &mut recorder,
                            &mut is_pressed,
                            &transcriber,
                            spoken_punctuation,
                            max_recordings,
                        );
                    } else {
                        handle_start(&mut recorder, &mut is_pressed, max_recordings);
                    }
                } else if !is_pressed {
                    handle_start(&mut recorder, &mut is_pressed, max_recordings);
                }
            }
            AppMessage::KeyUp => {
                if !toggle_mode {
                    handle_stop(
                        &mut recorder,
                        &mut is_pressed,
                        &transcriber,
                        spoken_punctuation,
                        max_recordings,
                    );
                }
            }
        }
    }

    Ok(())
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

fn handle_stop(
    recorder: &mut AudioRecorder,
    is_pressed: &mut bool,
    transcriber: &Arc<Transcriber>,
    spoken_punctuation: bool,
    max_recordings: u32,
) {
    if !*is_pressed {
        return;
    }
    *is_pressed = false;

    let Some(audio_path) = recorder.stop() else {
        return;
    };

    info!("Transcribing...");

    // Transcribe in a background thread to keep the main loop responsive
    let transcriber = Arc::clone(transcriber);
    std::thread::spawn(move || {
        let result = transcriber.transcribe(&audio_path);

        // Clean up
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

                if !text.is_empty() {
                    info!("Transcription: {text}");
                    if let Err(e) = TextInserter::insert(&text) {
                        error!("Failed to insert text: {e}");
                    }
                }
            }
            Err(e) => {
                error!("Transcription failed: {e}");
            }
        }
    });
}
