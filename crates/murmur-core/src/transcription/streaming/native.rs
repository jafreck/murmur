//! Native streaming transcription for engines with built-in streaming support.
//!
//! Unlike chunked streaming, this mode periodically reads all accumulated
//! audio and asks the engine to transcribe the full utterance, sending the
//! complete text as a replacement each time.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use crate::audio::TARGET_RATE;
use crate::transcription::engine::AsrEngine;

use super::{StreamingEvent, StreamingHandle, MIN_NEW_AUDIO_SECS, POLL_INTERVAL_MS};

// ── Public API ─────────────────────────────────────────────────────────

/// Start native streaming transcription for engines that support it
/// (Qwen3-ASR, MLX, Parakeet).
///
/// Unlike [`super::start_streaming`], this does **not** use chunked overlap or word
/// stitching. Instead it periodically reads all accumulated audio and asks
/// the engine to transcribe the full utterance, sending the complete text as
/// a replacement each time.
pub fn start_native_streaming(
    sample_buffer: Arc<Mutex<Vec<f32>>>,
    engine: Arc<dyn AsrEngine + Send + Sync>,
    translate: bool,
    filler_word_removal: bool,
    spoken_punctuation: bool,
    tx: mpsc::Sender<StreamingEvent>,
) -> StreamingHandle {
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let abort_flag = Arc::new(AtomicBool::new(false));
    let abort_flag_clone = Arc::clone(&abort_flag);

    let join_handle = std::thread::spawn(move || {
        native_streaming_loop(
            sample_buffer,
            engine,
            translate,
            filler_word_removal,
            spoken_punctuation,
            tx,
            stop_rx,
            abort_flag_clone,
        );
    });

    StreamingHandle::new(stop_tx, abort_flag, join_handle)
}

// ── Internal ───────────────────────────────────────────────────────────

/// Native streaming loop: transcribe accumulated audio on each tick.
#[allow(clippy::too_many_arguments)]
fn native_streaming_loop(
    sample_buffer: Arc<Mutex<Vec<f32>>>,
    engine: Arc<dyn AsrEngine + Send + Sync>,
    translate: bool,
    filler_word_removal: bool,
    spoken_punctuation: bool,
    tx: mpsc::Sender<StreamingEvent>,
    stop_rx: mpsc::Receiver<()>,
    abort_flag: Arc<AtomicBool>,
) {
    let min_samples = (MIN_NEW_AUDIO_SECS * TARGET_RATE as f32) as usize;
    let mut prev_len: usize = 0;
    let mut prev_text = String::new();

    loop {
        // Check for stop signal
        match stop_rx.try_recv() {
            Ok(()) | Err(mpsc::TryRecvError::Disconnected) => break,
            Err(mpsc::TryRecvError::Empty) => {}
        }

        let current_len = match sample_buffer.lock() {
            Ok(b) => b.len(),
            Err(e) => e.into_inner().len(),
        };

        // Wait for enough new audio
        let new_samples = current_len.saturating_sub(prev_len);
        if new_samples < min_samples {
            std::thread::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS));
            continue;
        }

        // Read all accumulated samples
        let samples: Vec<f32> = match sample_buffer.lock() {
            Ok(b) => b.clone(),
            Err(e) => e.into_inner().clone(),
        };

        if samples.is_empty() {
            std::thread::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS));
            continue;
        }

        // VAD check on just the new audio
        let new_start = prev_len.min(samples.len());
        if !crate::transcription::vad::contains_speech(&samples[new_start..]) {
            prev_len = current_len;
            std::thread::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS));
            continue;
        }

        let _ = tx.send(StreamingEvent::SpeechDetected);

        // Transcribe the full utterance
        match engine.transcribe(&samples, translate) {
            Ok(result) => {
                if abort_flag.load(Ordering::Relaxed) {
                    break;
                }

                // Apply postprocessing consistently with the batch path.
                let text = if result.pre_formatted {
                    result.text
                } else {
                    postprocess_text(&result.text, filler_word_removal, spoken_punctuation)
                };

                if !text.is_empty() && text != prev_text {
                    if text.starts_with(&prev_text) {
                        let new_suffix = &text[prev_text.len()..];
                        if !new_suffix.is_empty() {
                            let _ = tx.send(StreamingEvent::PartialText {
                                text: new_suffix.to_string(),
                                replace_chars: 0,
                            });
                        }
                    } else {
                        let replace_chars = prev_text.chars().count();
                        let _ = tx.send(StreamingEvent::PartialText {
                            text: text.clone(),
                            replace_chars,
                        });
                    }
                    prev_text = text;
                }
            }
            Err(e) => {
                log::error!("Native streaming transcription failed: {e}");
            }
        }

        prev_len = current_len;
    }
}

/// Apply postprocessing consistent with the batch transcription path.
fn postprocess_text(raw: &str, filler_word_removal: bool, spoken_punctuation: bool) -> String {
    let mut text = raw.to_string();
    if filler_word_removal {
        text = crate::transcription::postprocess::remove_filler_words(&text);
    }
    if spoken_punctuation {
        text = crate::transcription::postprocess::process(&text);
    }
    crate::transcription::postprocess::ensure_space_after_punctuation(&text)
}
