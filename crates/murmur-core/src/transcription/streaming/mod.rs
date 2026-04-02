//! Streaming transcription modes.
//!
//! Two paradigms are provided:
//! - **Chunked** ([`start_streaming`]): overlapping-window transcription with
//!   word-level stitching, suited to Whisper-class models.
//! - **Native** ([`start_native_streaming`]): full-utterance retranscription
//!   for engines that handle growing audio natively (Qwen3-ASR, MLX, Parakeet).

mod chunked;
mod native;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

pub use chunked::start_streaming;
pub use chunked::stitch;
pub use native::start_native_streaming;

/// Minimum new audio (seconds) before re-transcribing.
const MIN_NEW_AUDIO_SECS: f32 = 2.0;
/// Minimum interval between transcription attempts, in milliseconds.
const POLL_INTERVAL_MS: u64 = 300;

// ── Public types ───────────────────────────────────────────────────────

/// A message carrying newly-transcribed text from a streaming chunk.
pub enum StreamingEvent {
    /// Replace the last `replace_chars` characters with `text`.
    /// If `replace_chars` is 0, just append.
    PartialText { text: String, replace_chars: usize },
    /// VAD detected speech in the audio stream (heartbeat for silence timeout).
    SpeechDetected,
}

/// Handle returned by [`start_streaming`] to control the streaming thread.
///
/// Dropping the handle sends a stop signal (by disconnecting the channel),
/// but does **not** block until the thread exits. Call [`stop_and_join`]
/// when you need to guarantee the thread has exited before reusing the
/// `Transcriber` (e.g. before starting a final transcription pass).
pub struct StreamingHandle {
    stop_tx: mpsc::Sender<()>,
    abort_flag: Arc<AtomicBool>,
    join_handle: Option<std::thread::JoinHandle<()>>,
}

impl StreamingHandle {
    fn new(
        stop_tx: mpsc::Sender<()>,
        abort_flag: Arc<AtomicBool>,
        join_handle: std::thread::JoinHandle<()>,
    ) -> Self {
        Self {
            stop_tx,
            abort_flag,
            join_handle: Some(join_handle),
        }
    }

    /// Signal the streaming thread to stop and block until it exits.
    ///
    /// Sets the abort flag first so any in-progress whisper inference
    /// is cancelled immediately, then sends the channel stop signal
    /// and joins the thread.
    pub fn stop_and_join(mut self) {
        self.abort_flag.store(true, Ordering::Relaxed);
        let _ = self.stop_tx.send(());
        if let Some(handle) = self.join_handle.take() {
            if let Err(e) = handle.join() {
                log::error!("Streaming thread panicked: {e:?}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_event_partial_text() {
        let event = StreamingEvent::PartialText {
            text: "hello".to_string(),
            replace_chars: 3,
        };
        match event {
            StreamingEvent::PartialText {
                text,
                replace_chars,
            } => {
                assert_eq!(text, "hello");
                assert_eq!(replace_chars, 3);
            }
            StreamingEvent::SpeechDetected => panic!("unexpected variant"),
        }
    }

    #[test]
    fn test_vad_silence_no_speech() {
        let samples = vec![0.0f32; 16000];
        assert!(!crate::transcription::vad::contains_speech(&samples));
    }

    #[test]
    fn test_vad_low_noise_no_speech() {
        let samples = vec![0.001f32; 16000];
        assert!(!crate::transcription::vad::contains_speech(&samples));
    }

    #[test]
    fn test_vad_empty_no_speech() {
        assert!(!crate::transcription::vad::contains_speech(&[]));
    }
}
