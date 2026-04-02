use log::{error, info};
use std::sync::mpsc;
use std::sync::Arc;

use super::EffectContext;
use crate::app::AppMessage;
use crate::audio::recordings::RecordingStore;
use crate::transcription::{postprocess, streaming};

// ---------------------------------------------------------------------------
// Pure decision functions (extracted for testability)
// ---------------------------------------------------------------------------

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
/// optimisation applies — but only for Whisper's chunked streaming.
/// Native-streaming backends (Qwen3-ASR, MLX, Parakeet) always do a final
/// batch pass because streaming was just a live preview and may have
/// missed tail audio.
pub(crate) fn decide_transcription(
    has_transcriber: bool,
    max_recordings: u32,
    streaming_was_active: bool,
    has_audio: bool,
    native_streaming: bool,
) -> TranscribeDecision {
    if !has_transcriber {
        return TranscribeDecision::NoTranscriber;
    }
    if !has_audio {
        return TranscribeDecision::NoAudio;
    }
    // Only skip batch for Whisper-style chunked streaming, where the
    // chunks + stitching already cover the full audio. Native streaming
    // backends always get a final re-transcription of the complete audio.
    if max_recordings == 0 && streaming_was_active && !native_streaming {
        return TranscribeDecision::SkipBatchStreamingDone;
    }
    TranscribeDecision::Transcribe
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

// ---------------------------------------------------------------------------
// Effect handlers
// ---------------------------------------------------------------------------

pub(super) fn handle_start_recording(ctx: &mut EffectContext<'_>, path: std::path::PathBuf) {
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

pub(super) fn stop_and_transcribe(ctx: &mut EffectContext<'_>) {
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
            ctx.config.asr_backend(),
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
    let native_streaming = ctx.config.asr_backend().supports_native_streaming();

    // No-engine path: clean up both recorder modes and bail early.
    if matches!(
        decide_transcription(
            has_engine,
            max_recordings,
            streaming_was_active,
            true,
            native_streaming
        ),
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

        match decide_transcription(true, 0, streaming_was_active, has_audio, native_streaming) {
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

        match decide_transcription(
            true,
            max_recordings,
            streaming_was_active,
            has_audio,
            native_streaming,
        ) {
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

pub(super) fn start_streaming(ctx: &mut EffectContext<'_>) {
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
        ctx.config.asr_backend(),
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
        // Native streaming: re-transcribe the full accumulated buffer each tick.
        let filler_removal = ctx.state.filler_word_removal;
        let spoken_punctuation = ctx.state.spoken_punctuation;
        streaming::start_native_streaming(
            sample_buffer,
            engine,
            translate,
            filler_removal,
            spoken_punctuation,
            streaming_tx,
        )
    };

    *ctx.streaming_stop = Some(handle);
}
