use anyhow::Result;
use log::{error, info};
use std::sync::mpsc;
use std::sync::Arc;

use crate::audio::AudioRecorder;
use crate::config::Config;
use crate::hotkey::{CaptureFlag, SharedHotkeyConfig};
use crate::inserter::TextInserter;
use crate::model;
use crate::postprocess;
use crate::recordings::RecordingStore;
use crate::streaming;
use crate::transcriber::Transcriber;
use crate::tray::{TrayController, TrayState};

use super::{AppEffect, AppMessage, AppState};

pub struct EffectContext<'a> {
    pub recorder: &'a mut AudioRecorder,
    pub transcriber: &'a mut Arc<Transcriber>,
    pub tray: &'a mut TrayController,
    pub config: &'a mut Config,
    pub state: &'a mut AppState,
    pub tx: &'a mpsc::Sender<AppMessage>,
    pub streaming_stop: &'a mut Option<mpsc::Sender<()>>,
    pub hotkey_config: &'a SharedHotkeyConfig,
    pub capture_flag: &'a CaptureFlag,
}

/// Apply a single effect, returning (should_quit, extra_effects).
pub fn apply_effect(
    effect: AppEffect,
    ctx: &mut EffectContext<'_>,
) -> Result<(bool, Vec<AppEffect>)> {
    match effect {
        AppEffect::None => {}
        AppEffect::StartRecording(path) => {
            info!("Recording...");
            let result = if ctx.state.max_recordings == 0 {
                ctx.recorder.start_in_memory()
            } else {
                ctx.recorder.start(&path)
            };
            if let Err(e) = result {
                error!("Failed to start recording: {e}");
                let _ = ctx.tx.send(AppMessage::TranscriptionError(
                    format!("Failed to start recording: {e}"),
                ));
            }
        }
        AppEffect::StopAndTranscribe => {
            stop_and_transcribe(ctx);
        }
        AppEffect::StartStreaming => {
            start_streaming(ctx);
        }
        AppEffect::StopStreaming => {
            if let Some(stop) = ctx.streaming_stop.take() {
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
            let new_config = ctx.state.to_config(ctx.config);
            *ctx.config = new_config;
            if let Err(e) = ctx.config.save() {
                error!("Failed to save config: {e}");
            }
        }
        AppEffect::OpenConfig => {
            open_config_file();
        }
        AppEffect::ReloadConfig => {
            return reload_config(ctx);
        }
        AppEffect::SetTrayState(state) => {
            ctx.tray.set_state(state);
        }
        AppEffect::SetTrayModel(size) => {
            ctx.tray.set_model(&size);
            info!("Model changed to: {size}");
        }
        AppEffect::SetTrayLanguage(code) => {
            let name = crate::config::language_name(&code).unwrap_or(&code);
            info!("Language changed to: {name} ({code})");
            ctx.tray.set_language(&code);
        }
        AppEffect::SetTrayMode(mode) => {
            ctx.tray.set_mode(&mode);
            info!("Mode changed to: {mode}");
        }
        AppEffect::ReloadTranscriber(generation) => {
            reload_transcriber(ctx, generation);
        }
        AppEffect::EnterHotkeyCaptureMode => {
            ctx.capture_flag.store(true, std::sync::atomic::Ordering::Relaxed);
            ctx.tray.set_status("Press a key...");
            info!("Hotkey capture mode: press any key");
        }
        AppEffect::SetHotkey(key_name) => {
            if let Some(parsed) = crate::keycodes::parse(&key_name) {
                if let Ok(mut hk) = ctx.hotkey_config.lock() {
                    *hk = (parsed.key, parsed.modifiers.into_iter().collect());
                }
                ctx.config.hotkey = key_name.clone();
                ctx.tray.set_hotkey(&key_name);
                info!("Hotkey set to: {key_name}");
            } else {
                error!("Invalid captured key: {key_name}");
            }
            ctx.tray.set_state(TrayState::Idle);
        }
        AppEffect::Quit => {
            info!("Quit requested via tray");
            return Ok((true, vec![]));
        }
        AppEffect::LogError(e) => {
            error!("Transcription: {e}");
        }
    }
    Ok((false, vec![]))
}

fn stop_and_transcribe(ctx: &mut EffectContext<'_>) {
    // Stop streaming first (if running)
    if let Some(stop) = ctx.streaming_stop.take() {
        let _ = stop.send(());
    }

    let transcriber = Arc::clone(ctx.transcriber);
    let spoken_punctuation = ctx.state.spoken_punctuation;
    let translate_to_english = ctx.state.translate_to_english;
    let max_recordings = ctx.state.max_recordings;
    let tx = ctx.tx.clone();

    if max_recordings == 0 {
        // In-memory path: no file I/O
        let Some(samples) = ctx.recorder.stop_samples() else {
            info!("StopAndTranscribe called but no audio captured");
            let _ = tx.send(AppMessage::TranscriptionDone(String::new()));
            return;
        };

        info!("Transcribing {} samples from memory...", samples.len());
        std::thread::spawn(move || {
            match transcriber.transcribe_samples(&samples, translate_to_english) {
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
    } else {
        // File path: write WAV for recording storage
        let Some(audio_path) = ctx.recorder.stop() else {
            info!("StopAndTranscribe called but no active recording");
            let _ = tx.send(AppMessage::TranscriptionDone(String::new()));
            return;
        };

        info!("Transcribing...");
        std::thread::spawn(move || {
            let result = transcriber.transcribe(&audio_path, translate_to_english);
            RecordingStore::prune(max_recordings);
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

fn start_streaming(ctx: &mut EffectContext<'_>) {
    info!("Starting streaming transcription...");
    let sample_buffer = ctx.recorder.sample_buffer();
    let transcriber = Arc::clone(ctx.transcriber);
    let tx_app = ctx.tx.clone();
    let translate = ctx.state.translate_to_english;
    let spoken_punct = ctx.state.spoken_punctuation;

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
        translate,
        spoken_punct,
        streaming_tx,
    );
    *ctx.streaming_stop = Some(stop);
}

fn open_config_file() {
    let config_path = Config::file_path();
    info!("Opening config: {}", config_path.display());
    #[cfg(target_os = "macos")]
    { let _ = std::process::Command::new("open").arg(&config_path).spawn(); }
    #[cfg(target_os = "linux")]
    { let _ = std::process::Command::new("xdg-open").arg(&config_path).spawn(); }
    #[cfg(target_os = "windows")]
    { let _ = std::process::Command::new("cmd").args(["/C", "start", ""]).arg(&config_path).spawn(); }
}

fn reload_config(ctx: &mut EffectContext<'_>) -> Result<(bool, Vec<AppEffect>)> {
    info!("Reloading config from disk...");
    let new_config = Config::load();
    let old_model = ctx.state.model_size.clone();
    let old_lang = ctx.state.language.clone();
    let old_generation = ctx.state.reload_generation;
    *ctx.state = AppState::new(&new_config);
    // Preserve reload_generation so in-flight reloads are correctly discarded
    ctx.state.reload_generation = old_generation;

    // Update the hotkey listener if the hotkey changed
    if new_config.hotkey != ctx.config.hotkey {
        if let Some(parsed) = crate::keycodes::parse(&new_config.hotkey) {
            if let Ok(mut hk) = ctx.hotkey_config.lock() {
                *hk = (parsed.key, parsed.modifiers.into_iter().collect());
            }
            info!("Hotkey updated to: {}", new_config.hotkey);
            ctx.tray.set_hotkey(&new_config.hotkey);
        } else {
            error!("Invalid hotkey in config: '{}', keeping previous", new_config.hotkey);
        }
    }

    *ctx.config = new_config;
    // If model or language changed, reload the transcriber
    if ctx.state.model_size != old_model || ctx.state.language != old_lang {
        ctx.state.reload_generation += 1;
        let gen = ctx.state.reload_generation;
        info!("Config reloaded");
        return Ok((false, vec![AppEffect::ReloadTranscriber(gen)]));
    }
    info!("Config reloaded");
    Ok((false, vec![]))
}

fn reload_transcriber(ctx: &mut EffectContext<'_>, generation: u64) {
    let model_size = ctx.state.model_size.clone();
    let language = ctx.state.language.clone();
    let tx = ctx.tx.clone();
    info!("Loading model '{model_size}'...");

    std::thread::spawn(move || {
        if !crate::transcriber::model_exists(&model_size) {
            info!("Downloading {model_size} model...");
            if let Err(e) = model::download(&model_size, |percent| {
                if (percent as u32).is_multiple_of(25) {
                    info!("Downloading {model_size}... {percent:.0}%");
                }
            }) {
                error!("Failed to download model '{model_size}': {e}");
                let _ = tx.send(AppMessage::TranscriptionError(
                    format!("Failed to download model '{model_size}': {e}"),
                ));
                return;
            }
        }

        let Some(model_path) = crate::transcriber::find_model(&model_size) else {
            error!("Model '{model_size}' not found after download");
            let _ = tx.send(AppMessage::TranscriptionError(
                format!("Model '{model_size}' not found after download"),
            ));
            return;
        };

        match Transcriber::new(&model_path, &language) {
            Ok(t) => {
                info!("Model '{model_size}' loaded successfully");
                let _ = tx.send(AppMessage::TranscriberReady(Arc::new(t), generation));
            }
            Err(e) => {
                error!("Failed to load model '{model_size}': {e}");
                let _ = tx.send(AppMessage::TranscriptionError(
                    format!("Failed to load model '{model_size}': {e}"),
                ));
            }
        }
    });
}
