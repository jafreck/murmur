//! Whisper backend implementing [`AsrEngine`] via `whisper-rs`.

use super::engine::{AsrEngine, StreamingState, TranscriptionResult};
use super::transcriber::Transcriber;
use anyhow::Result;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Streaming state for the Whisper engine (wraps `whisper_rs::WhisperState`).
struct WhisperStreamingState {
    state: whisper_rs::WhisperState,
}

impl StreamingState for WhisperStreamingState {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

// SAFETY: WhisperState is internally thread-safe for single-threaded use within
// a streaming session. We only access it from one thread at a time.
unsafe impl Send for WhisperStreamingState {}

/// Whisper ASR engine backed by whisper-rs / whisper.cpp.
pub struct WhisperEngine {
    transcriber: Transcriber,
}

impl WhisperEngine {
    /// Load a Whisper model from the given path.
    pub fn new(model_path: &Path, language: &str) -> Result<Self> {
        let transcriber = Transcriber::new(model_path, language)?;
        Ok(Self { transcriber })
    }

    /// Access the inner transcriber (for backward-compatible callers).
    pub fn transcriber(&self) -> &Transcriber {
        &self.transcriber
    }
}

impl AsrEngine for WhisperEngine {
    fn transcribe(&self, samples: &[f32], translate: bool) -> Result<TranscriptionResult> {
        let text = self.transcriber.transcribe_samples(samples, translate)?;
        Ok(TranscriptionResult {
            text,
            pre_formatted: false,
        })
    }

    fn create_streaming_state(&self) -> Result<Box<dyn StreamingState>> {
        let state = self.transcriber.create_streaming_state()?;
        Ok(Box::new(WhisperStreamingState { state }))
    }

    fn streaming_transcribe(
        &self,
        state: &mut dyn StreamingState,
        samples: &[f32],
        translate: bool,
        abort_flag: &Arc<AtomicBool>,
    ) -> Result<TranscriptionResult> {
        let ws = state
            .as_any_mut()
            .downcast_mut::<WhisperStreamingState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid streaming state type for WhisperEngine"))?;
        let text =
            self.transcriber
                .streaming_transcribe(&mut ws.state, samples, translate, abort_flag)?;
        Ok(TranscriptionResult {
            text,
            pre_formatted: false,
        })
    }

    fn engine_name(&self) -> String {
        format!("Whisper ({})", self.transcriber.model_path().display())
    }

    fn supports_translation(&self) -> bool {
        true
    }
}
