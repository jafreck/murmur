use anyhow::Result;
use log::{error, info};
use std::sync::mpsc;
use std::sync::Arc;

use crate::audio::recordings::RecordingStore;
use crate::audio::AudioRecorder;
use crate::config::Config;
use crate::input::hotkey::{CaptureFlag, SharedHotkeyConfig};
use crate::input::inserter::TextInserter;
use crate::notes::NotesManager;
use crate::transcription::transcriber::Transcriber;
use crate::transcription::{model, postprocess, streaming};
use crate::ui::overlay::OverlayHandle;
use crate::ui::tray::{TrayController, TrayState};
use murmur_core::input::wake_word::WakeWordHandle;

use super::{AppEffect, AppMessage, AppState};

pub struct EffectContext<'a> {
    pub recorder: &'a mut AudioRecorder,
    pub transcriber: &'a mut Option<Arc<Transcriber>>,
    pub tray: &'a mut TrayController,
    pub config: &'a mut Config,
    pub state: &'a mut AppState,
    pub tx: &'a mpsc::Sender<AppMessage>,
    pub streaming_stop: &'a mut Option<streaming::StreamingHandle>,
    pub streaming_worker: &'a mut Option<murmur_core::transcription::SubprocessTranscriber>,
    pub hotkey_config: &'a SharedHotkeyConfig,
    pub capture_flag: &'a CaptureFlag,
    pub overlay: &'a mut Option<OverlayHandle>,
    pub wake_word: &'a mut Option<WakeWordHandle>,
    pub notes: &'a NotesManager,
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
                let _ = ctx.tx.send(AppMessage::TranscriptionError(format!(
                    "Failed to start recording: {e}"
                )));
            }
        }
        AppEffect::StopAndTranscribe => {
            stop_and_transcribe(ctx);
        }
        AppEffect::StartStreaming => {
            start_streaming(ctx);
        }
        AppEffect::StopStreaming => {
            if let Some(handle) = ctx.streaming_stop.take() {
                info!("Stopping streaming transcription");
                handle.stop_and_join();
                ctx.state.streaming_completed = true;
            }
        }
        AppEffect::InsertText(text) => {
            info!("Transcription: {text}");
            if let Err(e) = TextInserter::insert(&text) {
                error!("Insert failed: {e}");
            }
        }
        AppEffect::StreamingReplace {
            text,
            replace_chars,
        } => {
            if replace_chars > 0 {
                log::debug!("Streaming: replacing {replace_chars} chars with '{text}'");
            } else {
                log::debug!("Streaming: appending '{text}'");
            }
            if let Err(e) = TextInserter::replace(replace_chars, &text) {
                error!("Streaming replace failed: {e}");
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
        AppEffect::SetLanguageMenuEnabled(enabled) => {
            ctx.tray.set_language_menu_enabled(enabled);
        }
        AppEffect::SetTrayMode(mode) => {
            ctx.tray.set_mode(&mode);
            info!("Mode changed to: {mode}");
        }
        AppEffect::ReloadTranscriber(generation) => {
            *ctx.transcriber = None;
            ctx.tray.set_state(TrayState::Loading);
            reload_transcriber(ctx, generation);
        }
        AppEffect::EnterHotkeyCaptureMode => {
            ctx.capture_flag
                .store(true, std::sync::atomic::Ordering::Relaxed);
            ctx.tray.set_status("Press a key...");
            info!("Hotkey capture mode: press any key");
        }
        AppEffect::SetHotkey(key_name) => {
            if let Some(parsed) = crate::input::keycodes::parse(&key_name) {
                if let Ok(mut hk) = ctx.hotkey_config.lock() {
                    *hk = (parsed.key, parsed.modifiers.into_iter().collect());
                }
                ctx.config.hotkey = key_name.clone();
                ctx.tray.set_hotkey(&key_name);
                info!("Hotkey set to: {key_name}");
            } else {
                error!("Invalid captured key: {key_name}");
            }
            ctx.tray.set_state(if ctx.transcriber.is_some() {
                TrayState::Idle
            } else {
                TrayState::Loading
            });
        }
        AppEffect::UpdateNoiseSuppression(enabled) => {
            ctx.recorder.set_noise_suppression(enabled);
            info!(
                "Noise suppression {}",
                if enabled { "enabled" } else { "disabled" }
            );
        }
        AppEffect::ShowOverlay => {
            if let Some(ref mut overlay) = ctx.overlay {
                if let Err(e) = overlay.show() {
                    error!("Failed to show overlay: {e}");
                }
            }
        }
        AppEffect::HideOverlay => {
            if let Some(ref mut overlay) = ctx.overlay {
                if let Err(e) = overlay.done() {
                    error!("Failed to hide overlay: {e}");
                }
            }
        }
        AppEffect::UpdateOverlayText(text) => {
            if let Some(ref mut overlay) = ctx.overlay {
                if let Err(e) = overlay.set_text(&text) {
                    error!("Failed to update overlay text: {e}");
                }
            }
        }
        AppEffect::SaveNote(text) => {
            if !text.is_empty() {
                if let Err(e) = ctx.notes.save(&text) {
                    error!("Failed to save note: {e}");
                }
            }
        }
        AppEffect::PauseWakeWord => {
            if let Some(ref ww) = ctx.wake_word {
                ww.pause();
            }
        }
        AppEffect::ResumeWakeWord => {
            if let Some(ref ww) = ctx.wake_word {
                ww.resume();
            }
        }
        AppEffect::StartWakeWord => {
            start_wake_word(ctx);
        }
        AppEffect::StopWakeWord => {
            if let Some(ww) = ctx.wake_word.take() {
                ww.stop();
                info!("Wake word detector stopped");
            }
        }
        AppEffect::SpawnOverlay => {
            spawn_overlay(ctx);
        }
        AppEffect::KillOverlay => {
            if let Some(mut overlay) = ctx.overlay.take() {
                overlay.quit();
                info!("Overlay process stopped");
            }
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
    // Stop streaming first (if running) and wait for the thread to exit.
    let streaming_was_active = ctx.state.streaming_completed;
    ctx.state.streaming_completed = false;
    if let Some(handle) = ctx.streaming_stop.take() {
        info!("Waiting for streaming thread to finish before final transcription");
        handle.stop_and_join();
    }

    // Re-spawn the worker for the next recording session
    if ctx.streaming_worker.is_none() {
        if let Some(transcriber) = ctx.transcriber.as_ref() {
            match murmur_core::transcription::SubprocessTranscriber::new(
                transcriber.model_path(),
                transcriber.language(),
            ) {
                Ok(w) => *ctx.streaming_worker = Some(w),
                Err(e) => error!("Failed to re-spawn whisper worker: {e}"),
            }
        }
    }

    let Some(transcriber) = ctx.transcriber.as_ref().map(Arc::clone) else {
        info!("Model not loaded yet — ignoring transcription request");
        let _ = ctx.recorder.stop();
        let _ = ctx.recorder.stop_samples();
        let _ = ctx.tx.send(AppMessage::TranscriptionError(
            "Model is still loading, please try again".to_string(),
        ));
        return;
    };
    let spoken_punctuation = ctx.state.spoken_punctuation;
    let filler_word_removal = ctx.state.filler_word_removal;
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

        // When streaming was active, it already emitted the text
        // incrementally. Skip the batch re-transcription to avoid
        // duplicating output.
        if streaming_was_active {
            info!("Streaming was active — skipping batch re-transcription");
            let _ = tx.send(AppMessage::TranscriptionDone(String::new()));
            return;
        }

        info!("Transcribing {} samples from memory...", samples.len());
        std::thread::spawn(move || {
            match transcriber.transcribe_samples(&samples, translate_to_english) {
                Ok(raw) => {
                    let mut text = raw;
                    if filler_word_removal {
                        text = postprocess::remove_filler_words(&text);
                    }
                    if spoken_punctuation {
                        text = postprocess::process(&text);
                    }
                    text = postprocess::ensure_space_after_punctuation(&text);
                    if !text.is_empty() {
                        let _ = tx.send(AppMessage::TranscriptionDone(text));
                    }
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
                    let mut text = raw;
                    if filler_word_removal {
                        text = postprocess::remove_filler_words(&text);
                    }
                    if spoken_punctuation {
                        text = postprocess::process(&text);
                    }
                    text = postprocess::ensure_space_after_punctuation(&text);
                    if !text.is_empty() {
                        let _ = tx.send(AppMessage::TranscriptionDone(text));
                    }
                }
                Err(e) => {
                    let _ = tx.send(AppMessage::TranscriptionError(e.to_string()));
                }
            }
        });
    }
}

fn start_streaming(ctx: &mut EffectContext<'_>) {
    let Some(transcriber) = ctx.transcriber.as_ref().map(Arc::clone) else {
        info!("Model not loaded yet — cannot start streaming");
        return;
    };
    info!("Starting streaming transcription...");
    let sample_buffer = ctx.recorder.sample_buffer();
    let tx_app = ctx.tx.clone();
    let translate = ctx.state.translate_to_english;
    let filler_removal = ctx.state.filler_word_removal;

    let (streaming_tx, streaming_rx) = mpsc::channel::<streaming::StreamingEvent>();

    // Forward streaming events to app messages, coalescing any stale
    // events that queued up while the app was busy.  We keep the
    // `replace_chars` from the first (oldest) event in the batch —
    // it reflects what is actually on screen — and the `text` from
    // the last (newest) event.
    std::thread::spawn(move || {
        while let Ok(event) = streaming_rx.recv() {
            match event {
                streaming::StreamingEvent::SpeechDetected => {
                    let _ = tx_app.send(AppMessage::SpeechActivity);
                }
                streaming::StreamingEvent::PartialText {
                    mut text,
                    replace_chars,
                } => {
                    // Drain any newer events that arrived while we blocked.
                    while let Ok(newer) = streaming_rx.try_recv() {
                        match newer {
                            streaming::StreamingEvent::PartialText { text: t, .. } => {
                                text = t;
                            }
                            streaming::StreamingEvent::SpeechDetected => {
                                let _ = tx_app.send(AppMessage::SpeechActivity);
                            }
                        }
                    }

                    let _ = tx_app.send(AppMessage::StreamingPartialText {
                        text,
                        replace_chars,
                    });
                }
            }
        }
    });

    let worker = match ctx.streaming_worker.take() {
        Some(w) => w,
        None => {
            log::error!("No whisper worker available — cannot start streaming");
            return;
        }
    };

    let handle = streaming::start_streaming(
        sample_buffer,
        transcriber,
        translate,
        filler_removal,
        streaming_tx,
        worker,
    );
    *ctx.streaming_stop = Some(handle);
}

fn open_config_file() {
    let config_path = Config::file_path();
    info!("Opening config: {}", config_path.display());
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(&config_path).spawn();
    }
    #[cfg(target_os = "linux")]
    {
        let _ = std::process::Command::new("xdg-open")
            .arg(&config_path)
            .spawn();
    }
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("cmd")
            .args(["/C", "start", ""])
            .arg(&config_path)
            .spawn();
    }
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

    // Enforce English for English-only models
    let english_only = crate::config::is_english_only_model(&ctx.state.model_size);
    if english_only && ctx.state.language != "en" {
        ctx.state.language = "en".to_string();
    }
    ctx.tray.set_language_menu_enabled(!english_only);

    // Update the hotkey listener if the hotkey changed
    if new_config.hotkey != ctx.config.hotkey {
        if let Some(parsed) = crate::input::keycodes::parse(&new_config.hotkey) {
            if let Ok(mut hk) = ctx.hotkey_config.lock() {
                *hk = (parsed.key, parsed.modifiers.into_iter().collect());
            }
            info!("Hotkey updated to: {}", new_config.hotkey);
            ctx.tray.set_hotkey(&new_config.hotkey);
        } else {
            error!(
                "Invalid hotkey in config: '{}', keeping previous",
                new_config.hotkey
            );
        }
    }

    *ctx.config = new_config;

    // Sync all tray UI elements to the newly loaded config
    ctx.tray.sync_config(ctx.config);

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
        if !crate::transcription::transcriber::model_exists(&model_size) {
            info!("Downloading {model_size} model...");
            let last_milestone = std::cell::Cell::new(u32::MAX);
            if let Err(e) = model::download(&model_size, |percent| {
                let milestone = percent as u32 / 10 * 10;
                if milestone != last_milestone.get() {
                    last_milestone.set(milestone);
                    info!("Downloading {model_size}... {milestone}%");
                }
            }) {
                error!("Failed to download model '{model_size}': {e}");
                let _ = tx.send(AppMessage::TranscriptionError(format!(
                    "Failed to download model '{model_size}': {e}"
                )));
                return;
            }
        }

        let Some(model_path) = crate::transcription::transcriber::find_model(&model_size) else {
            error!("Model '{model_size}' not found after download");
            let _ = tx.send(AppMessage::TranscriptionError(format!(
                "Model '{model_size}' not found after download"
            )));
            return;
        };

        match Transcriber::new(&model_path, &language) {
            Ok(t) => {
                info!("Model '{model_size}' loaded successfully");
                let _ = tx.send(AppMessage::TranscriberReady(Arc::new(t), generation));
            }
            Err(e) => {
                error!("Failed to load model '{model_size}': {e}");
                let _ = tx.send(AppMessage::TranscriptionError(format!(
                    "Failed to load model '{model_size}': {e}"
                )));
            }
        }
    });
}

fn start_wake_word(ctx: &mut EffectContext<'_>) {
    // Stop existing detector if running
    if let Some(ww) = ctx.wake_word.take() {
        ww.stop();
    }

    let wake_phrase = ctx.config.wake_word.clone();
    let stop_phrase = ctx.config.stop_phrase.clone();
    let tx = ctx.tx.clone();

    let (event_tx, event_rx) = mpsc::channel();

    // Forward wake word events to app messages
    std::thread::spawn(move || {
        use murmur_core::input::wake_word::WakeWordEvent;
        while let Ok(event) = event_rx.recv() {
            let msg = match event {
                WakeWordEvent::WakeWordDetected => AppMessage::WakeWordDetected,
                WakeWordEvent::StopPhraseDetected => AppMessage::StopPhraseDetected,
            };
            if tx.send(msg).is_err() {
                break;
            }
        }
    });

    match murmur_core::input::wake_word::start_detector(wake_phrase, stop_phrase, event_tx) {
        Ok(handle) => {
            info!("Wake word detector started");
            *ctx.wake_word = Some(handle);
        }
        Err(e) => {
            error!("Failed to start wake word detector: {e}");
        }
    }
}

fn spawn_overlay(ctx: &mut EffectContext<'_>) {
    // Kill existing overlay if running
    if let Some(mut overlay) = ctx.overlay.take() {
        overlay.quit();
    }

    match OverlayHandle::spawn() {
        Ok(handle) => {
            info!("Overlay process started");
            *ctx.overlay = Some(handle);
        }
        Err(e) => {
            error!("Failed to spawn overlay: {e}");
        }
    }
}
