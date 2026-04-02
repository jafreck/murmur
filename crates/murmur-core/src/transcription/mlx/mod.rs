//! MLX backend implementing [`AsrEngine`] using Apple's MLX framework.
//!
//! Runs the **Qwen3-ASR** model natively on Apple Silicon GPU via Metal.
//! Weights are loaded from safetensors files produced by
//! `mlx-community/Qwen3-ASR-0.6B-bf16`.

mod cache;
mod config;
mod decoder;
mod encoder;
mod model;

use model::Qwen3AsrModel;

use super::engine::{AsrEngine, StreamingState, TranscriptionResult};
use anyhow::{Context, Result};
use mlx_rs::module::Param;
use mlx_rs::nn::{Conv2d, Embedding, LayerNorm, Linear, RmsNorm};
use mlx_rs::ops;
use mlx_rs::Array;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

// ─── Constants ──────────────────────────────────────────────────────

const N_MELS: usize = 128;
const MAX_GEN_TOKENS: usize = 448;

// ─── Weight-loading helpers ─────────────────────────────────────────

fn w(weights: &HashMap<String, Array>, key: &str) -> Result<Array> {
    weights
        .get(key)
        .cloned()
        .with_context(|| format!("missing weight: {key}"))
}

fn load_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    in_d: i32,
    out_d: i32,
    bias: bool,
) -> Result<Linear> {
    let mut linear = Linear::new(in_d, out_d)?;
    linear.weight = Param::new(w(weights, &format!("{prefix}.weight"))?);
    if bias {
        if let Some(b) = weights.get(&format!("{prefix}.bias")) {
            linear.bias = Param::new(Some(b.clone()));
        }
    } else {
        linear.bias = Param::new(None);
    }
    Ok(linear)
}

fn load_conv2d(
    weights: &HashMap<String, Array>,
    prefix: &str,
    in_ch: i32,
    out_ch: i32,
    kernel: i32,
    stride: i32,
    padding: i32,
) -> Result<Conv2d> {
    let mut conv = Conv2d::new(in_ch, out_ch, kernel)?;
    conv.stride = (stride, stride);
    conv.padding = (padding, padding);

    let mut w_arr = w(weights, &format!("{prefix}.weight"))?;
    // Auto-detect and transpose Conv2d weights from PyTorch OIHW to MLX OHWI.
    // MLX expects [O, kH, kW, I]; PyTorch saves [O, I, kH, kW].
    // Detect by checking: if shape[1] == in_ch and shape[3] != in_ch, it's OIHW.
    let shape = w_arr.shape().to_vec();
    if shape.len() == 4 && shape[1] == in_ch && shape[3] != in_ch {
        w_arr = w_arr.transpose_axes(&[0, 2, 3, 1])?;
    }
    conv.weight = Param::new(w_arr);

    if let Some(b) = weights.get(&format!("{prefix}.bias")) {
        conv.bias = Param::new(Some(b.clone()));
    } else {
        conv.bias = Param::new(None);
    }
    Ok(conv)
}

fn load_layer_norm(weights: &HashMap<String, Array>, prefix: &str, dims: i32) -> Result<LayerNorm> {
    let mut ln = LayerNorm::new(dims)?;
    if let Some(wt) = weights.get(&format!("{prefix}.weight")) {
        ln.weight = Param::new(Some(wt.clone()));
    }
    if let Some(b) = weights.get(&format!("{prefix}.bias")) {
        ln.bias = Param::new(Some(b.clone()));
    }
    Ok(ln)
}

fn load_rms_norm(
    weights: &HashMap<String, Array>,
    prefix: &str,
    dims: i32,
    eps: f32,
) -> Result<RmsNorm> {
    let mut rn = RmsNorm::new(dims)?;
    rn.eps = eps;
    rn.weight = Param::new(w(weights, &format!("{prefix}.weight"))?);
    Ok(rn)
}

fn load_embedding(
    weights: &HashMap<String, Array>,
    prefix: &str,
    count: i32,
    dims: i32,
) -> Result<Embedding> {
    let mut emb = Embedding::new(count, dims)?;
    emb.weight = Param::new(w(weights, &format!("{prefix}.weight"))?);
    Ok(emb)
}

// ─── Rotary position embeddings ─────────────────────────────────────

fn build_rope_freqs(head_dim: i32, max_len: i32, theta: f32) -> Result<(Array, Array)> {
    let half = head_dim / 2;
    let mut freqs = vec![0.0f32; half as usize];
    for i in 0..half as usize {
        freqs[i] = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
    }
    let freq_arr = Array::from_slice(&freqs, &[1, half]);

    let positions: Vec<f32> = (0..max_len).map(|p| p as f32).collect();
    let pos_arr = Array::from_slice(&positions, &[max_len, 1]);

    // [max_len, half] = pos @ freq
    let angles = ops::matmul(&pos_arr, &freq_arr)?;
    let cos_vals = ops::cos(&angles)?;
    let sin_vals = ops::sin(&angles)?;
    Ok((cos_vals, sin_vals))
}

fn apply_rope(x: &Array, cos: &Array, sin: &Array, offset: i32) -> Result<Array> {
    let shape = x.shape();
    let seq_len = shape[shape.len() - 2];
    let head_dim = shape[shape.len() - 1];
    let half = head_dim / 2;

    // Slice cos/sin for the current sequence range [offset .. offset+seq_len]
    let cos_slice = slice_seq(cos, offset, seq_len)?;
    let sin_slice = slice_seq(sin, offset, seq_len)?;

    // Split x into first half and second half along last dim
    // x1 = x[..., :half], x2 = x[..., half:]
    let x1 = narrow_last_dim(x, 0, half)?;
    let x2 = narrow_last_dim(x, half, half)?;

    // result = x1 * cos - x2 * sin, x2 * cos + x1 * sin (standard RoPE)
    let a1 = ops::multiply(&x1, &cos_slice)?;
    let b1 = ops::multiply(&x2, &sin_slice)?;
    let out1 = ops::subtract(&a1, &b1)?;

    let a2 = ops::multiply(&x2, &cos_slice)?;
    let b2 = ops::multiply(&x1, &sin_slice)?;
    let out2 = ops::add(&a2, &b2)?;

    ops::concatenate_axis(&[&out1, &out2], -1).map_err(Into::into)
}

/// Slice `arr` along dim-0: `arr[offset .. offset+len, :]`
fn slice_seq(arr: &Array, offset: i32, len: i32) -> Result<Array> {
    let full_len = arr.shape()[0];
    let end = (offset + len).min(full_len);
    // Use index slicing: reshape approach
    // arr shape [max_len, half] → take rows [offset..end]
    let indices: Vec<i32> = (offset..end).collect();
    let idx = Array::from_slice(&indices, &[end - offset]);
    arr.take_axis(&idx, 0).map_err(Into::into)
}

/// Extract `arr[..., start .. start+len]` along the last dimension.
fn narrow_last_dim(arr: &Array, start: i32, len: i32) -> Result<Array> {
    let ndim = arr.ndim() as i32;
    let shape = arr.shape();
    let _last = shape[shape.len() - 1];

    // Build gather indices
    let indices: Vec<i32> = (start..start + len).collect();
    let idx = Array::from_slice(&indices, &[len]);
    arr.take_axis(&idx, ndim - 1).map_err(Into::into)
}

/// Strip the `language ...<asr_text>` prefix that Qwen3-ASR emits.
fn strip_asr_prefix(text: &str) -> String {
    if let Some(pos) = text.find("<asr_text>") {
        return text[pos + "<asr_text>".len()..].to_string();
    }
    if let Some(pos) = text.find("asr_text>") {
        return text[pos + "asr_text>".len()..].to_string();
    }
    text.to_string()
}

// ─── MlxEngine (public API) ────────────────────────────────────────

pub struct MlxEngine {
    model: Mutex<Qwen3AsrModel>,
    tokenizer: tokenizers::Tokenizer,
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

// Safety: `Array` is `Send` (verified in mlx-rs 0.25.3). We guard all model
// access behind a `Mutex`, so concurrent `&self` calls are safe.
unsafe impl Sync for MlxEngine {}

impl MlxEngine {
    pub fn new(model_dir: &Path) -> Result<Self> {
        let model_name = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("mlx-qwen3-asr")
            .to_string();

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        let model = Qwen3AsrModel::load(model_dir)?;
        log::info!("MlxEngine loaded from {}", model_dir.display());

        Ok(Self {
            model: Mutex::new(model),
            tokenizer,
            model_name,
        })
    }

    fn decode_tokens(&self, token_ids: &[u32]) -> String {
        self.tokenizer.decode(token_ids, true).unwrap_or_default()
    }
}

impl AsrEngine for MlxEngine {
    fn transcribe(&self, samples: &[f32], _translate: bool) -> Result<TranscriptionResult> {
        let abort = Arc::new(AtomicBool::new(false));
        let mut model = self.model.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
        let token_ids = model.transcribe(samples, &abort)?;
        let raw_text = self.decode_tokens(&token_ids);
        log::debug!(
            "MLX raw decoded: '{raw_text}' from {} tokens",
            token_ids.len()
        );
        let text = strip_asr_prefix(&raw_text);
        Ok(TranscriptionResult {
            text,
            pre_formatted: false,
        })
    }

    fn create_streaming_state(&self) -> Result<Box<dyn StreamingState>> {
        Ok(Box::new(MlxStreamingState {
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
        let st = state
            .as_any_mut()
            .downcast_mut::<MlxStreamingState>()
            .context("invalid streaming state")?;

        st.accumulated_samples.extend_from_slice(samples);

        let mut model = self.model.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
        let token_ids = model.transcribe(&st.accumulated_samples, abort_flag)?;
        let text = strip_asr_prefix(&self.decode_tokens(&token_ids));
        Ok(TranscriptionResult {
            text,
            pre_formatted: false,
        })
    }

    fn engine_name(&self) -> String {
        format!("MLX ({})", self.model_name)
    }

    fn supports_translation(&self) -> bool {
        false
    }
}
