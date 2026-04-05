//! Parakeet-TDT backend implementing [`AsrEngine`] via ONNX Runtime.

use super::engine::{AsrEngine, StreamingState, TranscriptionResult};
use anyhow::{Context, Result};
use ort::session::Session;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

/// Parakeet-TDT ONNX engine.
pub struct ParakeetEngine {
    session: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
    model_name: String,
}

// SAFETY: All mutable ort::Session access is guarded by Mutex.
unsafe impl Send for ParakeetEngine {}
unsafe impl Sync for ParakeetEngine {}

struct ParakeetStreamingState;

impl StreamingState for ParakeetStreamingState {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl ParakeetEngine {
    /// Load Parakeet-TDT ONNX model from a directory.
    pub fn new(model_dir: &Path, quantization: crate::config::AsrQuantization) -> Result<Self> {
        use crate::config::AsrQuantization;

        let model_file = match quantization {
            AsrQuantization::Int8 => "model.int8.onnx",
            _ => "model.onnx",
        };

        let model_path = model_dir.join(model_file);
        let tokenizer_path = model_dir.join("tokenizer.json");

        let session = Session::builder()?
            .with_intra_threads(num_inference_threads())?
            .commit_from_file(&model_path)
            .context("Failed to load Parakeet ONNX model")?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        let model_name = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("parakeet")
            .to_string();

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            model_name,
        })
    }

    fn run_inference(&self, samples: &[f32]) -> Result<String> {
        // Parakeet expects raw audio as input: [1, num_samples] f32 at 16 kHz
        let num_samples = samples.len();
        let audio_tensor = ndarray::Array2::from_shape_vec((1, num_samples), samples.to_vec())?;
        let length_tensor = ndarray::Array1::from_vec(vec![num_samples as i64]);

        let audio_input = ort::value::Tensor::from_array(audio_tensor)?;
        let length_input = ort::value::Tensor::from_array(length_tensor)?;

        let mut sess = self.session.lock().unwrap();
        let outputs = sess.run(ort::inputs![audio_input, length_input])?;

        // Output is token IDs — decode with tokenizer
        let token_ids = outputs[0].try_extract_array::<i64>()?;
        let ids: Vec<u32> = token_ids.iter().map(|&id| id as u32).collect();

        let text = self
            .tokenizer
            .decode(&ids, true)
            .map_err(|e| anyhow::anyhow!("Tokenizer decode failed: {e}"))?;

        Ok(text.trim().to_string())
    }
}

impl AsrEngine for ParakeetEngine {
    fn transcribe(&self, samples: &[f32], _translate: bool) -> Result<TranscriptionResult> {
        let text = self.run_inference(samples)?;
        Ok(TranscriptionResult {
            text,
            pre_formatted: true, // Parakeet outputs punctuated, capitalized text
        })
    }

    fn create_streaming_state(&self) -> Result<Box<dyn StreamingState>> {
        // Parakeet is non-streaming (CTC/TDT) — uses overlap-chunk approach
        Ok(Box::new(ParakeetStreamingState))
    }

    fn streaming_transcribe(
        &self,
        _state: &mut dyn StreamingState,
        samples: &[f32],
        translate: bool,
        _abort_flag: &Arc<AtomicBool>,
    ) -> Result<TranscriptionResult> {
        // Non-streaming: transcribe each chunk independently
        self.transcribe(samples, translate)
    }

    fn engine_name(&self) -> String {
        format!("Parakeet-TDT ({})", self.model_name)
    }

    fn supports_translation(&self) -> bool {
        false
    }
}

fn num_inference_threads() -> usize {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    (cpus / 2).max(1)
}
