//! Qwen3-ASR backend implementing [`AsrEngine`] via ONNX Runtime.
//!
//! Uses a three-model pipeline: encoder → decoder_init (prefill) →
//! decoder_step (autoregressive with KV cache). Native streaming is
//! supported by accumulating audio across chunks and re-decoding.

use super::engine::{AsrEngine, StreamingState, TranscriptionResult};
use super::postprocess::strip_asr_prefix;
use anyhow::{Context, Result};
use ort::session::Session;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

// ── Model configuration ─────────────────────────────────────────────────

/// Dimensions and token IDs parsed from `config.json`.
struct QwenModelConfig {
    hidden_size: usize,
    vocab_size: usize,
    eos_token_ids: Vec<u32>,
}

// ── Engine ───────────────────────────────────────────────────────────────

/// Qwen3-ASR ONNX engine with KV-cache decoding.
pub struct QwenEngine {
    encoder: Mutex<Session>,
    decoder_init: Mutex<Session>,
    decoder_step: Mutex<Session>,
    /// Raw FP16 embedding table, kept compact to save ~600MB on the 1.7b model.
    /// Converted to FP32 on-the-fly during single-token lookups.
    embed_tokens_fp16: Vec<u8>,
    config: QwenModelConfig,
    tokenizer: tokenizers::Tokenizer,
    /// Token IDs before the `<|audio_pad|>` block.
    prefix_tokens: Vec<u32>,
    /// Token IDs after the `<|audio_pad|>` block.
    suffix_tokens: Vec<u32>,
    /// `<|audio_pad|>` token ID, repeated N times between prefix and suffix.
    audio_pad_id: u32,
    model_name: String,
}

// SAFETY: All mutable ort::Session access is guarded by Mutex.
unsafe impl Send for QwenEngine {}
unsafe impl Sync for QwenEngine {}

// ── Streaming state ─────────────────────────────────────────────────────

/// Streaming state for Qwen3-ASR.
///
/// Accumulates audio across chunks. Each call re-encodes and re-decodes
/// the full utterance (KV-cache is used within each decode pass).
struct QwenStreamingState {
    accumulated_samples: Vec<f32>,
}

impl StreamingState for QwenStreamingState {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

// ── Construction ────────────────────────────────────────────────────────

impl QwenEngine {
    /// Load Qwen3-ASR ONNX model from a directory.
    ///
    /// `model_dir` should contain encoder, decoder_init, decoder_step ONNX
    /// files, `embed_tokens.bin`, `tokenizer.json`, and `config.json`.
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

        // -- Parse config.json ------------------------------------------------
        let config_str =
            std::fs::read_to_string(&config_path).context("Failed to read config.json")?;
        let cj: serde_json::Value = serde_json::from_str(&config_str)?;

        let hidden_size = cj["hidden_size"].as_u64().unwrap_or(1024) as usize;
        let vocab_size = cj["vocab_size"].as_u64().unwrap_or(151936) as usize;

        // eos_token_id may be a scalar or an array in config.json.
        // Both 151643 (<|endoftext|>) and 151645 (<|im_end|>) are EOS.
        let eos_token_ids = match &cj["eos_token_id"] {
            serde_json::Value::Array(arr) => arr
                .iter()
                .filter_map(|v| v.as_u64().map(|x| x as u32))
                .collect(),
            v => vec![v.as_u64().unwrap_or(151643) as u32],
        };
        // Ensure both standard EOS tokens are included
        let mut eos_set = eos_token_ids;
        for &id in &[151643u32, 151645] {
            if !eos_set.contains(&id) {
                eos_set.push(id);
            }
        }

        let config = QwenModelConfig {
            hidden_size,
            vocab_size,
            eos_token_ids: eos_set,
        };

        // -- Load ONNX sessions -----------------------------------------------
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

        // -- Embeddings (kept as FP16 to save memory) ----------------------------
        let embed_tokens_fp16 =
            std::fs::read(&embed_path).context("Failed to read embed_tokens.bin")?;

        // -- Tokenizer --------------------------------------------------------
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        // -- Build prompt template tokens -------------------------------------
        let (prefix_tokens, suffix_tokens, audio_pad_id) = build_prompt_template(&tokenizer);

        let model_name = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("qwen3-asr")
            .to_string();

        log::info!(
            "QwenEngine loaded: hidden={hidden_size} vocab={vocab_size} eos={:?} \
             audio_offset={}",
            config.eos_token_ids,
            prefix_tokens.len(),
        );

        Ok(Self {
            encoder: Mutex::new(encoder),
            decoder_init: Mutex::new(decoder_init),
            decoder_step: Mutex::new(decoder_step),
            embed_tokens_fp16,
            config,
            tokenizer,
            prefix_tokens,
            suffix_tokens,
            audio_pad_id,
            model_name,
        })
    }

    // ── Inference helpers ────────────────────────────────────────────────

    /// Build the full `input_ids` sequence for `decoder_init`.
    ///
    /// Layout: `prefix_tokens ‖ audio_pad × audio_len ‖ suffix_tokens`
    ///
    /// Returns `(ids, audio_offset)` where `audio_offset` is the index of
    /// the first `<|audio_pad|>` token.
    fn build_input_ids(&self, audio_len: usize) -> (Vec<i64>, usize) {
        let audio_offset = self.prefix_tokens.len();
        let total = audio_offset + audio_len + self.suffix_tokens.len();
        let mut ids = Vec::with_capacity(total);

        ids.extend(self.prefix_tokens.iter().map(|&t| t as i64));
        ids.extend(std::iter::repeat_n(self.audio_pad_id as i64, audio_len));
        ids.extend(self.suffix_tokens.iter().map(|&t| t as i64));

        (ids, audio_offset)
    }

    /// Run encoder on raw 16 kHz mono f32 audio → features `[1, N, hidden]`.
    fn run_encoder(&self, samples: &[f32]) -> Result<ndarray::ArrayD<f32>> {
        let n_mels = 128usize;
        let mel = super::mel::whisper_mel(samples);
        let n_frames = super::mel::mel_frame_count(samples.len());

        let mel_tensor = ndarray::Array3::from_shape_vec((1, n_mels, n_frames), mel)?;
        let mel_input = ort::value::Tensor::from_array(mel_tensor)?;

        let mut enc = self.encoder.lock().unwrap();
        let outputs = enc.run(ort::inputs![mel_input])?;
        let view = outputs[0].try_extract_array::<f32>()?;
        Ok(view.to_owned())
    }

    /// Run `decoder_init` (prefill) → `(logits, present_keys, present_values)`.
    fn run_prefill(
        &self,
        encoder_features: &ndarray::ArrayD<f32>,
    ) -> Result<(
        ndarray::ArrayD<f32>,
        ndarray::ArrayD<f32>,
        ndarray::ArrayD<f32>,
    )> {
        let audio_len = encoder_features.shape()[1];
        let (input_ids, audio_offset) = self.build_input_ids(audio_len);
        let seq_len = input_ids.len();

        let ids_tensor = ndarray::Array2::from_shape_vec((1, seq_len), input_ids)?;
        let pos_ids: Vec<i64> = (0..seq_len as i64).collect();
        let pos_tensor = ndarray::Array2::from_shape_vec((1, seq_len), pos_ids)?;
        let offset_tensor = ndarray::Array1::from_vec(vec![audio_offset as i64]);

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

        let logits = outputs[0].try_extract_array::<f32>()?.to_owned();
        let keys = outputs[1].try_extract_array::<f32>()?.to_owned();
        let values = outputs[2].try_extract_array::<f32>()?.to_owned();

        Ok((logits, keys, values))
    }

    /// Run one `decoder_step` → `(logits, new_keys, new_values)`.
    fn run_step(
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
        let hs = self.config.hidden_size;
        let embed = lookup_embedding_fp16(&self.embed_tokens_fp16, token_id, hs);

        let embed_tensor = ndarray::Array3::from_shape_vec((1, 1, hs), embed)?;
        let pos_tensor = ndarray::Array2::from_elem((1, 1), position as i64);

        let embed_input = ort::value::Tensor::from_array(embed_tensor)?;
        let pos_input = ort::value::Tensor::from_array(pos_tensor)?;
        let keys_input = ort::value::TensorRef::from_array_view(past_keys.view())?;
        let vals_input = ort::value::TensorRef::from_array_view(past_values.view())?;

        let mut dec = self.decoder_step.lock().unwrap();
        let outputs = dec.run(ort::inputs![embed_input, pos_input, keys_input, vals_input])?;

        let logits = outputs[0].try_extract_array::<f32>()?.to_owned();
        let keys = outputs[1].try_extract_array::<f32>()?.to_owned();
        let values = outputs[2].try_extract_array::<f32>()?.to_owned();

        Ok((logits, keys, values))
    }

    /// Full encoder → decoder pipeline with KV-cache decoding.
    fn run_inference(&self, samples: &[f32], abort_flag: &Arc<AtomicBool>) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        // ── Encode ──────────────────────────────────────────────────────
        let encoder_out = self.run_encoder(samples)?;

        // ── Prefill ─────────────────────────────────────────────────────
        let (logits, mut present_keys, mut present_values) = self.run_prefill(&encoder_out)?;

        // Argmax on the last position of the prefill logits.
        let vocab_size = self.config.vocab_size;
        let logits_slice = logits.as_slice().context("Logits not contiguous")?;
        if logits_slice.len() < vocab_size {
            anyhow::bail!(
                "Logits too small: expected >= {vocab_size}, got {}",
                logits_slice.len()
            );
        }
        let last_offset = logits_slice.len().saturating_sub(vocab_size);
        let mut next_token = argmax(&logits_slice[last_offset..]);

        if self.config.eos_token_ids.contains(&next_token) {
            return Ok(String::new());
        }

        let mut generated_ids: Vec<u32> = vec![next_token];
        // KV-cache seq_len after prefill = 4th dimension of the keys tensor.
        let prefill_len = present_keys.shape().get(3).copied().unwrap_or(0);
        let mut current_pos = prefill_len;
        let max_tokens = 448usize;

        // ── Autoregressive loop with KV cache ───────────────────────────
        for _ in 1..max_tokens {
            if abort_flag.load(Ordering::Relaxed) {
                break;
            }

            let (step_logits, new_keys, new_values) =
                self.run_step(next_token, current_pos, &present_keys, &present_values)?;

            present_keys = new_keys;
            present_values = new_values;

            let sl = step_logits
                .as_slice()
                .context("Step logits not contiguous")?;
            let off = sl.len().saturating_sub(vocab_size);
            next_token = argmax(&sl[off..]);
            current_pos += 1;

            if self.config.eos_token_ids.contains(&next_token) {
                break;
            }
            generated_ids.push(next_token);
        }

        let text = self
            .tokenizer
            .decode(&generated_ids, true)
            .map_err(|e| anyhow::anyhow!("Tokenizer decode failed: {e}"))?;

        // Strip the "language ...<asr_text>" prefix that the model emits
        let text = strip_asr_prefix(&text);

        Ok(text.trim().to_string())
    }
}

// ── AsrEngine ───────────────────────────────────────────────────────────

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
            .context("Invalid streaming state for QwenEngine")?;

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

/// Greedy argmax over a logit slice → token ID.
fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Look up a single token's embedding from the FP16 table, converting to FP32.
///
/// Each token occupies `hidden_size * 2` bytes (FP16). Only the requested
/// token's slice is converted, keeping the full table compact in memory.
fn lookup_embedding_fp16(fp16_bytes: &[u8], token_id: u32, hidden_size: usize) -> Vec<f32> {
    use half::f16;
    let bytes_per_token = hidden_size * 2;
    let start = token_id as usize * bytes_per_token;
    let end = start + bytes_per_token;
    if end <= fp16_bytes.len() {
        fp16_bytes[start..end]
            .chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()
    } else {
        vec![0.0; hidden_size]
    }
}

/// Build the prompt prefix (before `<|audio_pad|>`), suffix (after), and
/// the pad token ID from the tokenizer's vocabulary.
///
/// Prompt layout:
/// ```text
/// Build the Qwen3-ASR chat-template prompt tokens.
///
/// The exact prompt structure (from `src/prompt.py` in qwen3-asr-onnx):
/// ```text
/// <|im_start|>system\n<|im_end|>\n
/// <|im_start|>user\n<|audio_start|><|audio_pad|>...<|audio_end|><|im_end|>\n
/// <|im_start|>assistant\n
/// ```
///
/// Returns `(prefix, suffix, audio_pad_id)` where audio_pad tokens go between
/// prefix and suffix.
fn build_prompt_template(_tokenizer: &tokenizers::Tokenizer) -> (Vec<u32>, Vec<u32>, u32) {
    // Hardcoded token IDs matching qwen3-asr-onnx/src/prompt.py.
    // These are identical across all Qwen3-ASR model sizes.
    const IM_START: u32 = 151644;
    const IM_END: u32 = 151645;
    const AUDIO_START: u32 = 151669;
    const AUDIO_END: u32 = 151670;
    const AUDIO_PAD: u32 = 151676;
    const NEWLINE: u32 = 198;
    const SYSTEM: u32 = 9125; // "system"
    const USER: u32 = 882; // "user"
    const ASSISTANT: u32 = 77091; // "assistant"

    // Prefix: everything before <|audio_pad|> tokens
    let prefix = vec![
        // System turn (empty)
        IM_START,
        SYSTEM,
        NEWLINE,
        IM_END,
        NEWLINE,
        // User turn start + audio_start
        IM_START,
        USER,
        NEWLINE,
        AUDIO_START,
    ];

    // Suffix: everything after <|audio_pad|> tokens
    let suffix = vec![
        AUDIO_END, IM_END, NEWLINE, // Assistant turn
        IM_START, ASSISTANT, NEWLINE,
    ];

    (prefix, suffix, AUDIO_PAD)
}

/// Number of threads for ONNX inference.
fn num_inference_threads() -> usize {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    (cpus / 2).max(1)
}
