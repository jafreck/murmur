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
use crate::transcription::{model, postprocess, streaming};
use crate::ui::overlay::OverlayHandle;
use crate::ui::tray::{TrayController, TrayState};
use murmur_core::input::wake_word::WakeWordHandle;
use murmur_core::transcription::AsrEngine;

use super::{AppEffect, AppMessage, AppState};

// ---------------------------------------------------------------------------
// Pure decision functions (extracted for testability)
// ---------------------------------------------------------------------------

/// Result of comparing old and new configuration states.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ConfigDiff {
    /// The language after applying English-only model enforcement.
    pub effective_language: String,
    /// Whether the language selection menu should be enabled.
    pub language_menu_enabled: bool,
    /// Whether model or effective language differs from the old state.
    pub model_or_language_changed: bool,
    /// Whether the hotkey binding changed.
    pub hotkey_changed: bool,
}

/// Compare old runtime state with a newly loaded config to decide what changed.
pub(crate) fn compute_config_diff(
    old_model: &str,
    old_language: &str,
    old_hotkey: &str,
    new_config: &Config,
) -> ConfigDiff {
    let english_only = crate::config::is_english_only_model(&new_config.model_size);
    let effective_language = if english_only && new_config.language != "en" {
        "en".to_string()
    } else {
        new_config.language.clone()
    };

    ConfigDiff {
        model_or_language_changed: new_config.model_size != old_model
            || effective_language != old_language,
        hotkey_changed: new_config.hotkey != old_hotkey,
        language_menu_enabled: !english_only,
        effective_language,
    }
}

/// The outcome of deciding how to handle a stop-and-transcribe request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TranscribeDecision {
    /// No transcriber is loaded.
    NoTranscriber,
    /// No audio was captured.
    NoAudio,
    /// Streaming already emitted the text; skip batch re-transcription.
    SkipBatchStreamingDone,
    /// Run transcription on the captured audio.
    Transcribe,
}

/// Decide how to handle a stop-and-transcribe request.
///
/// `max_recordings == 0` means in-memory mode where the streaming-skip
/// optimisation applies.
pub(crate) fn decide_transcription(
    has_transcriber: bool,
    max_recordings: u32,
    streaming_was_active: bool,
    has_audio: bool,
) -> TranscribeDecision {
    if !has_transcriber {
        return TranscribeDecision::NoTranscriber;
    }
    if !has_audio {
        return TranscribeDecision::NoAudio;
    }
    if max_recordings == 0 && streaming_was_active {
        return TranscribeDecision::SkipBatchStreamingDone;
    }
    TranscribeDecision::Transcribe
}

/// Whether to record to a file or keep audio in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecordingMode {
    InMemory,
    File,
}

/// Choose the recording mode based on the max-recordings setting.
///
/// When `max_recordings` is 0 the user opted out of persisting WAV files,
/// so we capture into an in-memory sample buffer instead.
pub(crate) fn recording_mode(max_recordings: u32) -> RecordingMode {
    if max_recordings == 0 {
        RecordingMode::InMemory
    } else {
        RecordingMode::File
    }
}

/// Apply the post-processing pipeline to raw transcription output.
///
/// The pipeline conditionally removes filler words and converts spoken
/// punctuation, then always normalises spacing after punctuation marks.
pub(crate) fn postprocess_transcription(
    raw: &str,
    filler_word_removal: bool,
    spoken_punctuation: bool,
) -> String {
    let mut text = raw.to_string();
    if filler_word_removal {
        text = postprocess::remove_filler_words(&text);
    }
    if spoken_punctuation {
        text = postprocess::process(&text);
    }
    postprocess::ensure_space_after_punctuation(&text)
}

/// The tray state to show after exiting hotkey-capture mode.
pub(crate) fn tray_state_after_hotkey_capture(has_transcriber: bool) -> TrayState {
    if has_transcriber {
        TrayState::Idle
    } else {
        TrayState::Loading
    }
}

/// Whether the streaming pipeline has everything it needs to start.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamingReadiness {
    NoTranscriber,
    NoWorker,
    Ready,
}

/// Check whether the streaming pipeline can be started.
#[cfg(test)]
pub(crate) fn check_streaming_readiness(
    has_transcriber: bool,
    has_worker: bool,
) -> StreamingReadiness {
    if !has_transcriber {
        StreamingReadiness::NoTranscriber
    } else if !has_worker {
        StreamingReadiness::NoWorker
    } else {
        StreamingReadiness::Ready
    }
}

// ---------------------------------------------------------------------------

pub struct EffectContext<'a> {
    pub recorder: &'a mut AudioRecorder,
    pub engine: &'a mut Option<Arc<dyn AsrEngine + Send + Sync>>,
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
            let result = match recording_mode(ctx.state.max_recordings) {
                RecordingMode::InMemory => ctx.recorder.start_in_memory(),
                RecordingMode::File => ctx.recorder.start(&path),
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
        AppEffect::SetBackend(backend) => {
            ctx.config.asr_backend = backend;
            let default_model = ctx.config.default_model_for_backend().to_string();
            ctx.config.model_size = default_model.clone();
            ctx.state.model_size = default_model.clone();
            ctx.tray.set_model(&default_model);
            ctx.tray.sync_config(ctx.config);
            info!(
                "Backend changed to: {backend}; model reset to {default_model}. \
                 Restart murmur for the Model menu to update."
            );
        }
        AppEffect::ReloadTranscriber(generation) => {
            *ctx.engine = None;
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
            ctx.tray
                .set_state(tray_state_after_hotkey_capture(ctx.engine.is_some()));
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

    // Re-spawn the whisper subprocess worker for the next recording session.
    // Only applicable when the backend is Whisper.
    if ctx.streaming_worker.is_none()
        && ctx.engine.is_some()
        && matches!(
            ctx.config.asr_backend,
            murmur_core::config::AsrBackend::Whisper
        )
    {
        if let Some(model_path) = murmur_core::transcription::find_model(&ctx.state.model_size) {
            match murmur_core::transcription::SubprocessTranscriber::new(
                &model_path,
                &ctx.state.language,
            ) {
                Ok(w) => *ctx.streaming_worker = Some(w),
                Err(e) => error!("Failed to re-spawn whisper worker: {e}"),
            }
        }
    }

    let has_engine = ctx.engine.is_some();
    let max_recordings = ctx.state.max_recordings;

    // No-engine path: clean up both recorder modes and bail early.
    if matches!(
        decide_transcription(has_engine, max_recordings, streaming_was_active, true),
        TranscribeDecision::NoTranscriber
    ) {
        info!("Model not loaded yet — ignoring transcription request");
        let _ = ctx.recorder.stop();
        let _ = ctx.recorder.stop_samples();
        let _ = ctx.tx.send(AppMessage::TranscriptionError(
            "Model is still loading, please try again".to_string(),
        ));
        return;
    }

    let engine = ctx.engine.as_ref().map(Arc::clone).unwrap();
    let spoken_punctuation = ctx.state.spoken_punctuation;
    let filler_word_removal = ctx.state.filler_word_removal;
    let translate_to_english = ctx.state.translate_to_english;
    let tx = ctx.tx.clone();

    if max_recordings == 0 {
        // In-memory path: no file I/O
        let samples = ctx.recorder.stop_samples();
        let has_audio = samples.is_some();

        match decide_transcription(true, 0, streaming_was_active, has_audio) {
            TranscribeDecision::NoAudio => {
                info!("StopAndTranscribe called but no audio captured");
                let _ = tx.send(AppMessage::TranscriptionDone(String::new()));
            }
            TranscribeDecision::SkipBatchStreamingDone => {
                info!("Streaming was active — skipping batch re-transcription");
                let _ = tx.send(AppMessage::TranscriptionDone(String::new()));
            }
            TranscribeDecision::Transcribe => {
                let samples = samples.expect("has_audio was true");
                info!("Transcribing {} samples from memory...", samples.len());
                std::thread::spawn(move || {
                    match engine.transcribe(&samples, translate_to_english) {
                        Ok(result) => {
                            let text = if result.pre_formatted {
                                result.text
                            } else {
                                postprocess_transcription(
                                    &result.text,
                                    filler_word_removal,
                                    spoken_punctuation,
                                )
                            };
                            if text.is_empty() {
                                info!("Transcription produced no text (VAD likely detected no speech)");
                                let _ = tx.send(AppMessage::TranscriptionDone(String::new()));
                            } else {
                                let _ = tx.send(AppMessage::TranscriptionDone(text));
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(AppMessage::TranscriptionError(e.to_string()));
                        }
                    }
                });
            }
            TranscribeDecision::NoTranscriber => unreachable!("checked above"),
        }
    } else {
        // File path: write WAV for recording storage
        let audio_path = ctx.recorder.stop();
        let has_audio = audio_path.is_some();

        match decide_transcription(true, max_recordings, streaming_was_active, has_audio) {
            TranscribeDecision::NoAudio => {
                info!("StopAndTranscribe called but no active recording");
                let _ = tx.send(AppMessage::TranscriptionDone(String::new()));
            }
            TranscribeDecision::Transcribe => {
                let audio_path = audio_path.expect("has_audio was true");
                info!("Transcribing...");
                std::thread::spawn(move || {
                    let samples = murmur_core::transcription::read_wav_samples(&audio_path);
                    RecordingStore::prune(max_recordings);
                    match samples {
                        Ok(samples) => match engine.transcribe(&samples, translate_to_english) {
                            Ok(result) => {
                                let text = if result.pre_formatted {
                                    result.text
                                } else {
                                    postprocess_transcription(
                                        &result.text,
                                        filler_word_removal,
                                        spoken_punctuation,
                                    )
                                };
                                if text.is_empty() {
                                    info!("Transcription produced no text (VAD likely detected no speech)");
                                    let _ = tx.send(AppMessage::TranscriptionDone(String::new()));
                                } else {
                                    let _ = tx.send(AppMessage::TranscriptionDone(text));
                                }
                            }
                            Err(e) => {
                                let _ = tx.send(AppMessage::TranscriptionError(e.to_string()));
                            }
                        },
                        Err(e) => {
                            let _ = tx.send(AppMessage::TranscriptionError(e.to_string()));
                        }
                    }
                });
            }
            _ => unreachable!("only NoAudio/Transcribe possible in file mode"),
        }
    }
}

fn start_streaming(ctx: &mut EffectContext<'_>) {
    if ctx.engine.is_none() {
        info!("Model not loaded yet — cannot start streaming");
        return;
    }

    let engine = ctx.engine.as_ref().map(Arc::clone).unwrap();
    info!("Starting streaming transcription...");
    let sample_buffer = ctx.recorder.sample_buffer();
    let tx_app = ctx.tx.clone();
    let translate = ctx.state.translate_to_english;
    let is_whisper = matches!(
        ctx.config.asr_backend,
        murmur_core::config::AsrBackend::Whisper
    );

    let (streaming_tx, streaming_rx) = mpsc::channel::<streaming::StreamingEvent>();

    // Forward streaming events to app messages, coalescing stale events.
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
                    let mut final_replace = replace_chars;
                    while let Ok(newer) = streaming_rx.try_recv() {
                        match newer {
                            streaming::StreamingEvent::PartialText {
                                text: t,
                                replace_chars: r,
                            } => {
                                text = t;
                                // For native streaming, each event replaces
                                // all previous text, so use the latest replace_chars.
                                final_replace = r;
                            }
                            streaming::StreamingEvent::SpeechDetected => {
                                let _ = tx_app.send(AppMessage::SpeechActivity);
                            }
                        }
                    }

                    let _ = tx_app.send(AppMessage::StreamingPartialText {
                        text,
                        replace_chars: final_replace,
                    });
                }
            }
        }
    });

    let handle = if is_whisper {
        // Whisper: chunked overlap + word stitching via subprocess worker
        let worker = ctx.streaming_worker.take();
        let filler_removal = ctx.state.filler_word_removal;
        streaming::start_streaming(
            sample_buffer,
            engine,
            translate,
            filler_removal,
            streaming_tx,
            worker,
        )
    } else {
        // ONNX backends: native full-utterance streaming
        streaming::start_native_streaming(sample_buffer, engine, translate, streaming_tx)
    };

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

    let diff = compute_config_diff(&old_model, &old_lang, &ctx.config.hotkey, &new_config);

    *ctx.state = AppState::new(&new_config);
    // Preserve reload_generation so in-flight reloads are correctly discarded
    ctx.state.reload_generation = old_generation;

    // Apply English-only model enforcement
    ctx.state.language = diff.effective_language;
    ctx.tray
        .set_language_menu_enabled(diff.language_menu_enabled);

    // Update the hotkey listener if the hotkey changed
    if diff.hotkey_changed {
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
    if diff.model_or_language_changed {
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
    let backend = ctx.config.asr_backend;
    let quantization = ctx.config.asr_quantization;
    let tx = ctx.tx.clone();
    info!("Loading {backend} model '{model_size}'...");

    std::thread::spawn(move || {
        match create_engine_on_thread(backend, &model_size, &language, quantization) {
            Ok(engine) => {
                info!("{backend} model '{model_size}' loaded successfully");
                let _ = tx.send(AppMessage::EngineReady(Arc::from(engine), generation));
            }
            Err(e) => {
                error!("Failed to load {backend} model '{model_size}': {e}");
                let _ = tx.send(AppMessage::TranscriptionError(format!(
                    "Failed to load {backend} model '{model_size}': {e}"
                )));
            }
        }
    });
}

/// Download (if needed) and instantiate an ASR engine on a background thread.
pub(super) fn create_engine_on_thread(
    backend: murmur_core::config::AsrBackend,
    model_size: &str,
    language: &str,
    #[allow(unused_variables)] quantization: murmur_core::config::AsrQuantization,
) -> anyhow::Result<Box<dyn AsrEngine + Send + Sync>> {
    use murmur_core::config::AsrBackend;

    match backend {
        AsrBackend::Whisper => {
            if !murmur_core::transcription::model_exists(model_size) {
                info!("Downloading {model_size} model...");
                let last_milestone = std::cell::Cell::new(u32::MAX);
                model::download(model_size, |percent| {
                    let milestone = percent as u32 / 10 * 10;
                    if milestone != last_milestone.get() {
                        last_milestone.set(milestone);
                        info!("Downloading {model_size}... {milestone}%");
                    }
                })?;
            }

            let model_path = murmur_core::transcription::find_model(model_size)
                .ok_or_else(|| anyhow::anyhow!("Model not found: {model_size}"))?;
            let engine = murmur_core::transcription::WhisperEngine::new(&model_path, language)?;
            Ok(Box::new(engine))
        }
        #[cfg(feature = "onnx")]
        AsrBackend::Qwen3Asr => {
            if !murmur_core::transcription::model_exists_for_backend(
                AsrBackend::Qwen3Asr,
                model_size,
                quantization,
            ) {
                info!("Downloading Qwen3-ASR model...");
                murmur_core::transcription::download_for_backend(
                    AsrBackend::Qwen3Asr,
                    model_size,
                    quantization,
                    |p| {
                        let milestone = p as u32 / 10 * 10;
                        info!("Downloading Qwen3-ASR... {milestone}%");
                    },
                )?;
            }
            let model_dir = murmur_core::transcription::qwen3_asr_model_dir(model_size);
            let engine = murmur_core::transcription::QwenEngine::new(&model_dir, quantization)?;
            Ok(Box::new(engine))
        }
        #[cfg(feature = "onnx")]
        AsrBackend::Parakeet => {
            if !murmur_core::transcription::model_exists_for_backend(
                AsrBackend::Parakeet,
                model_size,
                quantization,
            ) {
                info!("Downloading Parakeet model...");
                murmur_core::transcription::download_for_backend(
                    AsrBackend::Parakeet,
                    model_size,
                    quantization,
                    |p| {
                        let milestone = p as u32 / 10 * 10;
                        info!("Downloading Parakeet... {milestone}%");
                    },
                )?;
            }
            let model_dir = murmur_core::transcription::parakeet_model_dir(model_size);
            let engine = murmur_core::transcription::ParakeetEngine::new(&model_dir, quantization)?;
            Ok(Box::new(engine))
        }
        #[cfg(not(feature = "onnx"))]
        AsrBackend::Qwen3Asr | AsrBackend::Parakeet => {
            anyhow::bail!(
                "ONNX backends require the 'onnx' feature. \
                 Rebuild with: cargo build --features onnx"
            );
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // compute_config_diff
    // -----------------------------------------------------------------------

    fn config_with(f: impl FnOnce(&mut Config)) -> Config {
        let mut c = Config::default();
        f(&mut c);
        c
    }

    #[test]
    fn config_diff_no_changes() {
        let config = Config::default();
        let diff = compute_config_diff(
            &config.model_size,
            &config.language,
            &config.hotkey,
            &config,
        );
        assert!(!diff.model_or_language_changed);
        assert!(!diff.hotkey_changed);
        // Default model is "0.6b" (Qwen3-ASR), so language menu is enabled
        assert!(diff.language_menu_enabled);
        assert_eq!(diff.effective_language, "en");
    }

    #[test]
    fn config_diff_model_changed() {
        let new_config = config_with(|c| c.model_size = "large".to_string());
        let diff = compute_config_diff("base", "en", &new_config.hotkey, &new_config);
        assert!(diff.model_or_language_changed);
        assert!(!diff.hotkey_changed);
    }

    #[test]
    fn config_diff_language_changed() {
        // Use a non-english-only model so language isn't forced
        let old = config_with(|c| c.model_size = "base".to_string());
        let new_config = config_with(|c| {
            c.model_size = "base".to_string();
            c.language = "fr".to_string();
        });
        let diff = compute_config_diff(&old.model_size, &old.language, &old.hotkey, &new_config);
        assert!(diff.model_or_language_changed);
        assert_eq!(diff.effective_language, "fr");
    }

    #[test]
    fn config_diff_hotkey_changed() {
        let old = Config::default();
        let new_config = config_with(|c| c.hotkey = "F12".to_string());
        let diff = compute_config_diff(&old.model_size, &old.language, &old.hotkey, &new_config);
        assert!(diff.hotkey_changed);
        assert!(!diff.model_or_language_changed);
    }

    #[test]
    fn config_diff_english_only_model_forces_language() {
        let new_config = config_with(|c| {
            c.model_size = "base.en".to_string();
            c.language = "fr".to_string();
        });
        let diff = compute_config_diff("base", "en", &new_config.hotkey, &new_config);
        assert_eq!(diff.effective_language, "en");
        assert!(!diff.language_menu_enabled);
        // Model changed from "base" to "base.en"
        assert!(diff.model_or_language_changed);
    }

    #[test]
    fn config_diff_english_only_model_already_english() {
        let new_config = config_with(|c| {
            c.model_size = "base.en".to_string();
            c.language = "en".to_string();
        });
        let diff = compute_config_diff("base.en", "en", &new_config.hotkey, &new_config);
        assert_eq!(diff.effective_language, "en");
        assert!(!diff.language_menu_enabled);
        assert!(!diff.model_or_language_changed);
    }

    #[test]
    fn config_diff_distil_model_forces_english() {
        let new_config = config_with(|c| {
            c.model_size = "distil-large".to_string();
            c.language = "de".to_string();
        });
        let diff = compute_config_diff("base", "de", &new_config.hotkey, &new_config);
        assert_eq!(diff.effective_language, "en");
        assert!(!diff.language_menu_enabled);
        // Model changed AND effective language changed (de → en)
        assert!(diff.model_or_language_changed);
    }

    #[test]
    fn config_diff_multiple_changes() {
        let new_config = config_with(|c| {
            c.model_size = "large".to_string();
            c.language = "fr".to_string();
            c.hotkey = "F12".to_string();
        });
        let diff = compute_config_diff("base", "en", "F10", &new_config);
        assert!(diff.model_or_language_changed);
        assert!(diff.hotkey_changed);
        assert_eq!(diff.effective_language, "fr");
        assert!(diff.language_menu_enabled);
    }

    #[test]
    fn config_diff_non_english_only_preserves_language() {
        let new_config = config_with(|c| {
            c.model_size = "large".to_string();
            c.language = "ja".to_string();
        });
        let diff = compute_config_diff("large", "ja", &new_config.hotkey, &new_config);
        assert_eq!(diff.effective_language, "ja");
        assert!(diff.language_menu_enabled);
        assert!(!diff.model_or_language_changed);
    }

    #[test]
    fn config_diff_english_only_language_change_masked() {
        // Config says "fr" but model is english-only → effective is "en".
        // Old language was "en" so no change detected.
        let new_config = config_with(|c| {
            c.model_size = "tiny.en".to_string();
            c.language = "fr".to_string();
        });
        let diff = compute_config_diff("tiny.en", "en", &new_config.hotkey, &new_config);
        assert_eq!(diff.effective_language, "en");
        assert!(!diff.model_or_language_changed);
    }

    // -----------------------------------------------------------------------
    // decide_transcription
    // -----------------------------------------------------------------------

    #[test]
    fn decide_no_transcriber() {
        assert_eq!(
            decide_transcription(false, 0, false, false),
            TranscribeDecision::NoTranscriber
        );
        assert_eq!(
            decide_transcription(false, 10, true, true),
            TranscribeDecision::NoTranscriber
        );
    }

    #[test]
    fn decide_no_audio_in_memory() {
        assert_eq!(
            decide_transcription(true, 0, false, false),
            TranscribeDecision::NoAudio
        );
    }

    #[test]
    fn decide_no_audio_file_mode() {
        assert_eq!(
            decide_transcription(true, 10, false, false),
            TranscribeDecision::NoAudio
        );
    }

    #[test]
    fn decide_streaming_skip_in_memory() {
        assert_eq!(
            decide_transcription(true, 0, true, true),
            TranscribeDecision::SkipBatchStreamingDone
        );
    }

    #[test]
    fn decide_streaming_not_skipped_in_file_mode() {
        // In file mode (max_recordings > 0), streaming skip doesn't apply
        assert_eq!(
            decide_transcription(true, 10, true, true),
            TranscribeDecision::Transcribe
        );
    }

    #[test]
    fn decide_transcribe_in_memory() {
        assert_eq!(
            decide_transcription(true, 0, false, true),
            TranscribeDecision::Transcribe
        );
    }

    #[test]
    fn decide_transcribe_file_mode() {
        assert_eq!(
            decide_transcription(true, 10, false, true),
            TranscribeDecision::Transcribe
        );
    }

    #[test]
    fn decide_no_transcriber_overrides_all() {
        // NoTranscriber takes priority regardless of other flags
        for &streaming in &[true, false] {
            for &has_audio in &[true, false] {
                assert_eq!(
                    decide_transcription(false, 0, streaming, has_audio),
                    TranscribeDecision::NoTranscriber
                );
            }
        }
    }

    #[test]
    fn decide_no_audio_overrides_streaming() {
        // Even if streaming was active, no audio means nothing to do
        assert_eq!(
            decide_transcription(true, 0, true, false),
            TranscribeDecision::NoAudio
        );
    }

    // -----------------------------------------------------------------------
    // recording_mode
    // -----------------------------------------------------------------------

    #[test]
    fn recording_mode_in_memory_when_zero() {
        assert_eq!(recording_mode(0), RecordingMode::InMemory);
    }

    #[test]
    fn recording_mode_file_when_nonzero() {
        assert_eq!(recording_mode(1), RecordingMode::File);
        assert_eq!(recording_mode(100), RecordingMode::File);
    }

    // -----------------------------------------------------------------------
    // postprocess_transcription
    // -----------------------------------------------------------------------

    #[test]
    fn postprocess_passthrough() {
        // With both flags off, only ensure_space_after_punctuation runs
        assert_eq!(
            postprocess_transcription("hello world", false, false),
            "hello world"
        );
    }

    #[test]
    fn postprocess_space_after_punctuation_always_applied() {
        // Even with both flags off the spacing normaliser runs
        assert_eq!(
            postprocess_transcription("hello,world", false, false),
            "hello, world"
        );
    }

    #[test]
    fn postprocess_filler_removal_only() {
        let result = postprocess_transcription("um hello um world", true, false);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn postprocess_spoken_punctuation_only() {
        let result = postprocess_transcription("hello period", false, true);
        assert!(result.contains('.'), "expected '.' in: {result}");
        assert!(
            !result.contains("period"),
            "spoken word should be replaced: {result}"
        );
    }

    #[test]
    fn postprocess_both_enabled() {
        let result = postprocess_transcription("um hello comma world period", true, true);
        assert!(!result.contains("um"), "filler should be removed: {result}");
        assert!(result.contains(','), "comma should be present: {result}");
        assert!(result.contains('.'), "period should be present: {result}");
    }

    #[test]
    fn postprocess_empty_text() {
        assert_eq!(postprocess_transcription("", false, false), "");
        assert_eq!(postprocess_transcription("", true, true), "");
    }

    #[test]
    fn postprocess_only_fillers_produces_empty() {
        let result = postprocess_transcription("um uh er", true, false);
        assert_eq!(result, "");
    }

    #[test]
    fn postprocess_filler_removal_preserves_real_words() {
        let result = postprocess_transcription("I think so", true, false);
        assert_eq!(result, "I think so");
    }

    // -----------------------------------------------------------------------
    // tray_state_after_hotkey_capture
    // -----------------------------------------------------------------------

    #[test]
    fn tray_after_hotkey_idle_when_transcriber_loaded() {
        assert_eq!(tray_state_after_hotkey_capture(true), TrayState::Idle);
    }

    #[test]
    fn tray_after_hotkey_loading_when_no_transcriber() {
        assert_eq!(tray_state_after_hotkey_capture(false), TrayState::Loading);
    }

    // -----------------------------------------------------------------------
    // check_streaming_readiness
    // -----------------------------------------------------------------------

    #[test]
    fn streaming_readiness_no_transcriber() {
        assert_eq!(
            check_streaming_readiness(false, false),
            StreamingReadiness::NoTranscriber
        );
        assert_eq!(
            check_streaming_readiness(false, true),
            StreamingReadiness::NoTranscriber
        );
    }

    #[test]
    fn streaming_readiness_no_worker() {
        assert_eq!(
            check_streaming_readiness(true, false),
            StreamingReadiness::NoWorker
        );
    }

    #[test]
    fn streaming_readiness_ready() {
        assert_eq!(
            check_streaming_readiness(true, true),
            StreamingReadiness::Ready
        );
    }

    #[test]
    fn streaming_readiness_no_transcriber_takes_priority() {
        // Even when worker is present, no transcriber wins
        assert_eq!(
            check_streaming_readiness(false, true),
            StreamingReadiness::NoTranscriber
        );
    }
}
