//! MLX backend implementing [`AsrEngine`] using Apple's MLX framework.
//!
//! Runs Qwen3-ASR natively on Apple Silicon GPU via Metal.

#![cfg(feature = "mlx")]

use super::engine::{AsrEngine, StreamingState, TranscriptionResult};
use anyhow::Result;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub struct MlxEngine {
    model_name: String,
}

struct MlxStreamingState {
    accumulated_samples: Vec<f32>,
}

impl StreamingState for MlxStreamingState {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl MlxEngine {
    pub fn new(model_dir: &Path) -> Result<Self> {
        let model_name = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("mlx-qwen3-asr")
            .to_string();

        // TODO: Load safetensors weights and build model
        log::info!("MlxEngine stub loaded from {}", model_dir.display());

        Ok(Self { model_name })
    }
}

impl AsrEngine for MlxEngine {
    fn transcribe(&self, _samples: &[f32], _translate: bool) -> Result<TranscriptionResult> {
        anyhow::bail!("MLX backend not yet implemented — model architecture pending")
    }

    fn create_streaming_state(&self) -> Result<Box<dyn StreamingState>> {
        Ok(Box::new(MlxStreamingState {
            accumulated_samples: Vec::new(),
        }))
    }

    fn streaming_transcribe(
        &self,
        _state: &mut dyn StreamingState,
        _samples: &[f32],
        _translate: bool,
        _abort_flag: &Arc<AtomicBool>,
    ) -> Result<TranscriptionResult> {
        anyhow::bail!("MLX backend not yet implemented — model architecture pending")
    }

    fn engine_name(&self) -> String {
        format!("MLX ({})", self.model_name)
    }

    fn supports_translation(&self) -> bool {
        false
    }
}
