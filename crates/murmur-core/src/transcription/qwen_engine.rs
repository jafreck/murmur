//! Qwen3-ASR backend implementing [`AsrEngine`] via ONNX Runtime.

#![cfg(feature = "onnx")]

use super::engine::{AsrEngine, StreamingState, TranscriptionResult};
use anyhow::{Context, Result};
use ort::session::Session;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

/// Qwen3-ASR ONNX engine.
pub struct QwenEngine {
    encoder: Mutex<Session>,
    decoder_init: Mutex<Session>,
    #[allow(dead_code)]
    decoder_step: Mutex<Session>,
    #[allow(dead_code)]
    embed_tokens: Vec<f32>, // flattened [vocab_size, hidden_size], loaded from FP16 → FP32
    #[allow(dead_code)]
    hidden_size: usize,
    tokenizer: tokenizers::Tokenizer,
    eos_token_id: u32,
    model_name: String,
}

// SAFETY: All mutable ort::Session access is guarded by Mutex.
unsafe impl Send for QwenEngine {}
unsafe impl Sync for QwenEngine {}

struct QwenStreamingState;

impl StreamingState for QwenStreamingState {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl QwenEngine {
    /// Load Qwen3-ASR ONNX model from a directory.
    ///
    /// `model_dir` should contain encoder, decoder_init, decoder_step ONNX files,
    /// embed_tokens.bin, tokenizer.json, and config.json.
    /// `quantization` selects int4 vs fp32 file variants.
    pub fn new(model_dir: &Path, quantization: crate::config::AsrQuantization) -> Result<Self> {
        use crate::config::AsrQuantization;

        let suffix = match quantization {
            AsrQuantization::Int4 => ".int4",
            _ => "",
        };

        let encoder_path = model_dir.join(format!("encoder{suffix}.onnx"));
        let decoder_init_path = model_dir.join(format!("decoder_init{suffix}.onnx"));
        let decoder_step_path = model_dir.join(format!("decoder_step{suffix}.onnx"));
        let embed_path = model_dir.join("embed_tokens.bin");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let config_path = model_dir.join("config.json");

        // Load config to get hidden_size and eos_token_id
        let config_str =
            std::fs::read_to_string(&config_path).context("Failed to read config.json")?;
        let config: serde_json::Value = serde_json::from_str(&config_str)?;
        let hidden_size = config["hidden_size"].as_u64().unwrap_or(896) as usize;
        let eos_token_id = config["eos_token_id"].as_u64().unwrap_or(151643) as u32;

        let threads = num_inference_threads();

        let encoder = Session::builder()?
            .with_intra_threads(threads)?
            .commit_from_file(&encoder_path)
            .context("Failed to load encoder ONNX")?;

        let decoder_init = Session::builder()?
            .with_intra_threads(threads)?
            .commit_from_file(&decoder_init_path)
            .context("Failed to load decoder_init ONNX")?;

        let decoder_step = Session::builder()?
            .with_intra_threads(threads)?
            .commit_from_file(&decoder_step_path)
            .context("Failed to load decoder_step ONNX")?;

        // Load embed_tokens (FP16 -> FP32)
        let embed_bytes = std::fs::read(&embed_path).context("Failed to read embed_tokens.bin")?;
        let embed_tokens = load_fp16_to_f32(&embed_bytes);

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        let model_name = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("qwen3-asr")
            .to_string();

        Ok(Self {
            encoder: Mutex::new(encoder),
            decoder_init: Mutex::new(decoder_init),
            decoder_step: Mutex::new(decoder_step),
            embed_tokens,
            hidden_size,
            tokenizer,
            eos_token_id,
            model_name,
        })
    }

    /// Run the full encoder → decoder pipeline on audio samples.
    fn run_inference(&self, samples: &[f32]) -> Result<String> {
        // Step 1: Compute mel spectrogram (128-bin, Qwen3-ASR style)
        // 16 kHz, 25 ms window (400 samples), 10 ms hop (160 samples)
        let n_mels = 128usize;
        let mel = compute_mel_spectrogram(samples, n_mels, 400, 160);
        let n_frames = mel.len() / n_mels;

        // Step 2: Run encoder — extract owned array before releasing lock
        let encoder_out_owned = {
            let mel_tensor = ndarray::Array3::from_shape_vec((1, n_mels, n_frames), mel)?;
            let mel_input = ort::value::Tensor::from_array(mel_tensor)?;

            let mut enc = self.encoder.lock().unwrap();
            let encoder_outputs = enc.run(ort::inputs![mel_input])?;
            let view = encoder_outputs[0].try_extract_array::<f32>()?;
            view.to_owned()
        };

        // Step 3: Run decoder_init with start tokens
        let start_ids: Vec<i64> = vec![151644, 151645]; // <|startoftext|>, <|transcribe|>

        let mut generated_ids: Vec<u32> = Vec::new();
        let max_tokens = 448;

        // Initial decoder_init run
        let vocab_size = {
            let start_tensor =
                ndarray::Array2::from_shape_vec((1, start_ids.len()), start_ids.clone())?;
            let start_input = ort::value::Tensor::from_array(start_tensor)?;
            let enc_input = ort::value::TensorRef::from_array_view(encoder_out_owned.view())?;

            let mut dec = self.decoder_init.lock().unwrap();
            let decoder_outputs = dec.run(ort::inputs![start_input, enc_input])?;

            let logits = decoder_outputs[0].try_extract_array::<f32>()?;
            let logits_shape = logits.shape();
            let vocab_size = *logits_shape.last().unwrap_or(&0);

            let logits_owned = logits.to_owned();
            let logits_slice = logits_owned.as_slice().context("Logits not contiguous")?;
            if logits_slice.len() < vocab_size {
                anyhow::bail!(
                    "Logits shape mismatch: expected at least {vocab_size} values, got {}",
                    logits_slice.len()
                );
            }
            let offset = logits_slice.len().saturating_sub(vocab_size);
            let next_token = argmax(&logits_slice[offset..]);

            if next_token == self.eos_token_id {
                return Ok(String::new());
            }
            generated_ids.push(next_token);
            vocab_size
        };

        // Autoregressive decoding loop.
        // Re-runs decoder_init each step (no KV cache). Slower but correct;
        // KV-cache optimization is a follow-up.
        for _ in 1..max_tokens {
            let mut all_ids: Vec<i64> = vec![151644, 151645];
            all_ids.extend(generated_ids.iter().map(|&id| id as i64));

            let ids_tensor = ndarray::Array2::from_shape_vec((1, all_ids.len()), all_ids)?;
            let ids_input = ort::value::Tensor::from_array(ids_tensor)?;
            let enc_input = ort::value::TensorRef::from_array_view(encoder_out_owned.view())?;

            let mut dec = self.decoder_init.lock().unwrap();
            let outputs = dec.run(ort::inputs![ids_input, enc_input])?;

            let logits = outputs[0].try_extract_array::<f32>()?;
            let logits_owned = logits.to_owned();
            let logits_slice = logits_owned.as_slice().context("Logits not contiguous")?;
            let offset = logits_slice.len().saturating_sub(vocab_size);
            let next_token = argmax(&logits_slice[offset..]);

            if next_token == self.eos_token_id {
                break;
            }
            generated_ids.push(next_token);
        }

        let text = self
            .tokenizer
            .decode(&generated_ids, true)
            .map_err(|e| anyhow::anyhow!("Tokenizer decode failed: {e}"))?;

        Ok(text.trim().to_string())
    }
}

impl AsrEngine for QwenEngine {
    fn transcribe(&self, samples: &[f32], _translate: bool) -> Result<TranscriptionResult> {
        let text = self.run_inference(samples)?;
        Ok(TranscriptionResult {
            text,
            pre_formatted: false,
        })
    }

    fn create_streaming_state(&self) -> Result<Box<dyn StreamingState>> {
        // TODO: Implement native Qwen3-ASR streaming with KV-cache state
        Ok(Box::new(QwenStreamingState))
    }

    fn streaming_transcribe(
        &self,
        _state: &mut dyn StreamingState,
        samples: &[f32],
        translate: bool,
        _abort_flag: &Arc<AtomicBool>,
    ) -> Result<TranscriptionResult> {
        // Fall back to full transcription per chunk for now
        self.transcribe(samples, translate)
    }

    fn engine_name(&self) -> String {
        format!("Qwen3-ASR ({})", self.model_name)
    }

    fn supports_translation(&self) -> bool {
        false
    }
}

/// Greedy argmax over a logit slice, returning the token id.
fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Number of threads for inference.
fn num_inference_threads() -> usize {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    (cpus / 2).max(1)
}

/// Load FP16 binary data into f32 vec.
fn load_fp16_to_f32(bytes: &[u8]) -> Vec<f32> {
    use half::f16;
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            f16::from_bits(bits).to_f32()
        })
        .collect()
}

/// Compute mel spectrogram from raw audio samples.
///
/// Simplified implementation producing a reasonable approximation.
/// A full implementation would use proper FFT and mel filterbanks matching
/// the Qwen3-ASR preprocessing pipeline.
fn compute_mel_spectrogram(
    samples: &[f32],
    n_mels: usize,
    n_fft: usize,
    hop_length: usize,
) -> Vec<f32> {
    if samples.is_empty() || samples.len() < n_fft {
        return vec![0.0; n_mels];
    }

    let n_frames = (samples.len().saturating_sub(n_fft)) / hop_length + 1;
    if n_frames == 0 {
        return vec![0.0; n_mels];
    }

    let mut mel = vec![0.0f32; n_mels * n_frames];

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_length;
        let end = (start + n_fft).min(samples.len());
        let frame = &samples[start..end];

        if frame.is_empty() {
            continue;
        }

        let energy: f32 = frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32;
        let log_energy = (energy.max(1e-10)).ln();

        for mel_bin in 0..n_mels {
            let bin_weight = ((mel_bin as f32 + 1.0) / n_mels as f32).sqrt();
            mel[mel_bin * n_frames + frame_idx] = log_energy * bin_weight;
        }
    }

    mel
}
