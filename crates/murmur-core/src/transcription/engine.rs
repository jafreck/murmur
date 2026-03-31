//! Backend-agnostic ASR engine trait.
//!
//! Each ASR backend (Whisper, Qwen3-ASR, Parakeet) implements [`AsrEngine`]
//! so the rest of the app can transcribe without knowing which model is running.

use anyhow::Result;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// A single-shot transcription result.
pub struct TranscriptionResult {
    pub text: String,
    /// Whether the output is already punctuated/capitalized (e.g. Parakeet).
    pub pre_formatted: bool,
}

/// Opaque streaming state owned by the engine.
///
/// Each backend stores whatever it needs between streaming chunks
/// (e.g. a `WhisperState`, ONNX KV-cache tensors, etc.).
pub trait StreamingState: Send + std::any::Any {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Backend-agnostic speech-to-text engine.
pub trait AsrEngine: Send + Sync {
    /// Transcribe a complete audio buffer (16 kHz, mono, f32).
    fn transcribe(&self, samples: &[f32], translate: bool) -> Result<TranscriptionResult>;

    /// Create a fresh streaming state for incremental transcription.
    fn create_streaming_state(&self) -> Result<Box<dyn StreamingState>>;

    /// Transcribe a chunk of audio using an existing streaming state.
    ///
    /// `abort_flag` can be set to cancel mid-inference.
    fn streaming_transcribe(
        &self,
        state: &mut dyn StreamingState,
        samples: &[f32],
        translate: bool,
        abort_flag: &Arc<AtomicBool>,
    ) -> Result<TranscriptionResult>;

    /// Human-readable name of the engine (e.g. "Whisper base.en").
    fn engine_name(&self) -> String;

    /// Whether this engine supports translation to English.
    fn supports_translation(&self) -> bool;
}
