//! Qwen3-ASR backend implementing [`AsrEngine`] via ONNX Runtime.
//!
//! Uses a three-model pipeline: encoder → decoder_init (prefill with KV cache)
//! → decoder_step (autoregressive with KV cache). Native streaming is supported
//! by persisting accumulated audio across chunks.

#![cfg(feature = "onnx")]

use super::engine::{AsrEngine, StreamingState, TranscriptionResult};
use anyhow::{Context, Result};
use ort::session::Session;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

// ── Model config parsed from config.json ────────────────────────────────

#[derive(Debug, Clone)]
struct QwenModelConfig {
    hidden_size: usize,
    vocab_size: usize,
    eos_token_id: u32,
    /// Start position of audio_pad tokens in the prefix sequence (typically 9).
    audio_offset: usize,
}

impl QwenModelConfig {
    fn from_json(path: &Path) -> Result<Self> {
        let s = std::fs::read_to_string(path).context("Failed to read config.json")?;
        let v: serde_json::Value = serde_json::from_str(&s)?;

        Ok(Self {
            hidden_size: v["hidden_size"].as_u64().unwrap_or(1024) as usize,
            vocab_size: v["vocab_size"].as_u64().unwrap_or(151936) as usize,
            eos_token_id: v["eos_token_id"].as_u64().unwrap_or(151643) as u32,
            audio_offset: v["audio_offset"].as_u64().unwrap_or(9) as usize,
        })
    }
}

// ── Engine ───────────────────────────────────────────────────────────────

/// Qwen3-ASR ONNX engine with KV-cache optimized decoding.
pub struct QwenEngine {
    encoder: Mutex<Session>,
    decoder_init: Mutex<Session>,
    decoder_step: Mutex<Session>,
    embed_tokens: Vec<f32>,
    config: QwenModelConfig,
    tokenizer: tokenizers::Tokenizer,
    model_name: String,
}

// SAFETY: All mutable ort::Session access is guarded by Mutex.
unsafe impl Send for QwenEngine {}
unsafe impl Sync for QwenEngine {}

impl QwenEngine {
    /// Load Qwen3-ASR ONNX model from a directory.
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

        let config = QwenModelConfig::from_json(&config_path)?;
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
            config,
            tokenizer,
            model_name,
        })
    }

    /// Build prefix token IDs for decoder_init.
    ///
    /// Layout: `[pad × audio_offset] + [audio_pad × audio_len] + [<|transcribe|>]`
    fn build_input_ids(&self, audio_len: usize) -> Vec<i64> {
        let audio_pad_id = 151647i64;
        let transcribe_id = 151646i64;

        let mut ids = Vec::with_capacity(self.config.audio_offset + audio_len + 1);
        // Prefix tokens up to audio_offset (the ONNX model handles embedding)
        ids.resize(self.config.audio_offset, audio_pad_id);
        // Audio pad tokens (replaced by encoder features inside decoder_init)
        ids.extend(std::iter::repeat(audio_pad_id).take(audio_len));
        // Suffix: transcribe token
        ids.push(transcribe_id);
        ids
    }

    /// Run encoder on audio samples, returning features `[1, N, dim]`.
    fn encode(&self, samples: &[f32]) -> Result<ndarray::Array3<f32>> {
        let n_mels = 128usize;
        let mel = compute_mel_spectrogram(samples, n_mels, 400, 160);
        let n_frames = mel.len() / n_mels;

        let mel_tensor = ndarray::Array3::from_shape_vec((1, n_mels, n_frames), mel)?;
        let mel_input = ort::value::Tensor::from_array(mel_tensor)?;

        let mut enc = self.encoder.lock().unwrap();
        let outputs = enc.run(ort::inputs![mel_input])?;
        let view = outputs[0].try_extract_array::<f32>()?;
        Ok(view.to_owned().into_dimensionality()?)
    }

    /// Run decoder_init (prefill) returning (logits, KV keys, KV values).
    fn prefill(
        &self,
        encoder_features: &ndarray::Array3<f32>,
    ) -> Result<(
        ndarray::ArrayD<f32>,
        ndarray::ArrayD<f32>,
        ndarray::ArrayD<f32>,
    )> {
        let audio_len = encoder_features.shape()[1];
        let input_ids = self.build_input_ids(audio_len);
        let seq_len = input_ids.len();

        let ids_tensor = ndarray::Array2::from_shape_vec((1, seq_len), input_ids)?;
        let pos_ids: Vec<i64> = (0..seq_len as i64).collect();
        let pos_tensor = ndarray::Array2::from_shape_vec((1, seq_len), pos_ids)?;
        let offset_tensor = ndarray::Array1::from_vec(vec![self.config.audio_offset as i64]);

        let ids_input = ort::value::Tensor::from_array(ids_tensor)?;
        let pos_input = ort::value::Tensor::from_array(pos_tensor)?;
        let audio_input = ort::value::TensorRef::from_array_view(encoder_features.view())?;
        let offset_input = ort::value::Tensor::from_array(offset_tensor)?;

        let mut dec = self.decoder_init.lock().unwrap();
        let outputs = dec.run(ort::inputs![
            ids_input,
            pos_input,
            audio_input,
            offset_input
        ])?;

        let logits = outputs[0].try_extract_array::<f32>()?.to_owned().into_dyn();
        let keys = outputs[1].try_extract_array::<f32>()?.to_owned().into_dyn();
        let values = outputs[2].try_extract_array::<f32>()?.to_owned().into_dyn();

        Ok((logits, keys, values))
    }

    /// Run one decoder_step with KV cache.
    fn step(
        &self,
        token_id: u32,
        position: usize,
        past_keys: &ndarray::ArrayD<f32>,
        past_values: &ndarray::ArrayD<f32>,
    ) -> Result<(
        ndarray::ArrayD<f32>,
        ndarray::ArrayD<f32>,
        ndarray::ArrayD<f32>,
    )> {
        let embed = lookup_embedding(&self.embed_tokens, token_id, self.config.hidden_size);
        let embed_tensor = ndarray::Array3::from_shape_vec((1, 1, self.config.hidden_size), embed)?;
        let pos_tensor = ndarray::Array2::from_elem((1, 1), position as i64);

        let embed_input = ort::value::Tensor::from_array(embed_tensor)?;
        let pos_input = ort::value::Tensor::from_array(pos_tensor)?;
        let keys_input = ort::value::TensorRef::from_array_view(past_keys.view())?;
        let values_input = ort::value::TensorRef::from_array_view(past_values.view())?;

        let mut dec = self.decoder_step.lock().unwrap();
        let outputs = dec.run(ort::inputs![
            embed_input,
            pos_input,
            keys_input,
            values_input
        ])?;

        let logits = outputs[0].try_extract_array::<f32>()?.to_owned().into_dyn();
        let keys = outputs[1].try_extract_array::<f32>()?.to_owned().into_dyn();
        let values = outputs[2].try_extract_array::<f32>()?.to_owned().into_dyn();

        Ok((logits, keys, values))
    }

    /// Full encoder → decoder pipeline with KV-cache optimization.
    fn run_inference(&self, samples: &[f32], abort_flag: &Arc<AtomicBool>) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        let encoder_features = self.encode(samples)?;
        let (logits, mut present_keys, mut present_values) = self.prefill(&encoder_features)?;

        // Greedy decode first token from prefill logits (last position)
        let vocab_size = self.config.vocab_size;
        let logits_slice = logits.as_slice().context("Logits not contiguous")?;
        if logits_slice.len() < vocab_size {
            anyhow::bail!(
                "Logits too small: expected >= {vocab_size}, got {}",
                logits_slice.len()
            );
        }
        let offset = logits_slice.len().saturating_sub(vocab_size);
        let mut next_token = argmax(&logits_slice[offset..]);

        if next_token == self.config.eos_token_id {
            return Ok(String::new());
        }

        let mut generated_ids: Vec<u32> = vec![next_token];
        let prefill_seq_len = present_keys.shape().get(3).copied().unwrap_or(0);
        let mut current_pos = prefill_seq_len;
        let max_tokens = 448;

        // Autoregressive decoding with KV cache via decoder_step
        for _ in 1..max_tokens {
            if abort_flag.load(Ordering::Relaxed) {
                break;
            }

            let (logits, new_keys, new_values) =
                self.step(next_token, current_pos, &present_keys, &present_values)?;

            present_keys = new_keys;
            present_values = new_values;

            let logits_slice = logits.as_slice().context("Step logits not contiguous")?;
            let step_offset = logits_slice.len().saturating_sub(vocab_size);
            next_token = argmax(&logits_slice[step_offset..]);

            if next_token == self.config.eos_token_id {
                break;
            }
            generated_ids.push(next_token);
            current_pos += 1;
        }

        let text = self
            .tokenizer
            .decode(&generated_ids, true)
            .map_err(|e| anyhow::anyhow!("Tokenizer decode failed: {e}"))?;

        Ok(text.trim().to_string())
    }
}

// ── Streaming state ─────────────────────────────────────────────────────

/// Native streaming state for Qwen3-ASR.
///
/// Accumulates audio samples across chunks. Each streaming call re-encodes
/// and re-decodes the full utterance to produce the latest transcription.
struct QwenStreamingState {
    accumulated_samples: Vec<f32>,
}

impl StreamingState for QwenStreamingState {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

// ── AsrEngine implementation ────────────────────────────────────────────

impl AsrEngine for QwenEngine {
    fn transcribe(&self, samples: &[f32], _translate: bool) -> Result<TranscriptionResult> {
        let no_abort = Arc::new(AtomicBool::new(false));
        let text = self.run_inference(samples, &no_abort)?;
        Ok(TranscriptionResult {
            text,
            pre_formatted: false,
        })
    }

    fn create_streaming_state(&self) -> Result<Box<dyn StreamingState>> {
        Ok(Box::new(QwenStreamingState {
            accumulated_samples: Vec::new(),
        }))
    }

    fn streaming_transcribe(
        &self,
        state: &mut dyn StreamingState,
        samples: &[f32],
        _translate: bool,
        abort_flag: &Arc<AtomicBool>,
    ) -> Result<TranscriptionResult> {
        let qs = state
            .as_any_mut()
            .downcast_mut::<QwenStreamingState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid streaming state for QwenEngine"))?;

        qs.accumulated_samples.extend_from_slice(samples);
        let text = self.run_inference(&qs.accumulated_samples, abort_flag)?;

        Ok(TranscriptionResult {
            text,
            pre_formatted: false,
        })
    }

    fn engine_name(&self) -> String {
        format!("Qwen3-ASR ({})", self.model_name)
    }

    fn supports_translation(&self) -> bool {
        false
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn lookup_embedding(embed_tokens: &[f32], token_id: u32, hidden_size: usize) -> Vec<f32> {
    let start = token_id as usize * hidden_size;
    let end = start + hidden_size;
    if end <= embed_tokens.len() {
        embed_tokens[start..end].to_vec()
    } else {
        vec![0.0; hidden_size]
    }
}

fn num_inference_threads() -> usize {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    (cpus / 2).max(1)
}

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
/// Simplified placeholder matching Qwen3-ASR params: 16kHz, 128 bins,
/// 25ms window (n_fft=400), 10ms hop (hop_length=160).
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
