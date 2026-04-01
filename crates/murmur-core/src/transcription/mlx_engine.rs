//! MLX backend implementing [`AsrEngine`] using Apple's MLX framework.
//!
//! Runs the **Qwen3-ASR** model natively on Apple Silicon GPU via Metal.
//! Weights are loaded from safetensors files produced by
//! `mlx-community/Qwen3-ASR-0.6B-bf16`.

#![cfg(feature = "mlx")]

use super::engine::{AsrEngine, StreamingState, TranscriptionResult};
use super::mel;
use anyhow::{Context, Result};
use mlx_rs::module::{Module, Param};
use mlx_rs::nn::{Conv2d, Embedding, LayerNorm, Linear, RmsNorm};
use mlx_rs::ops;
use mlx_rs::Array;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

// ─── Constants ──────────────────────────────────────────────────────

const N_MELS: usize = 128;
const MAX_GEN_TOKENS: usize = 448;

// ─── Model configuration ───────────────────────────────────────────

#[derive(Debug, Clone)]
struct ModelConfig {
    // Audio encoder
    enc_d_model: i32,
    enc_num_layers: usize,
    enc_num_heads: usize,
    enc_ffn_dim: i32,
    enc_output_dim: i32,
    enc_downsample_hidden: i32,
    enc_n_window: usize,
    enc_n_window_infer: usize,

    // Text decoder
    dec_hidden_size: i32,
    dec_num_layers: usize,
    dec_num_heads: usize,
    dec_num_kv_heads: usize,
    dec_head_dim: i32,
    dec_intermediate_size: i32,
    dec_vocab_size: i32,
    dec_rope_theta: f32,
    dec_rms_norm_eps: f32,

    // Special token IDs
    eos_token_ids: Vec<u32>,
    audio_token_id: u32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            enc_d_model: 896,
            enc_num_layers: 18,
            enc_num_heads: 14,
            enc_ffn_dim: 3584,
            enc_output_dim: 1024,
            enc_downsample_hidden: 480,
            enc_n_window: 50,
            enc_n_window_infer: 800,

            dec_hidden_size: 1024,
            dec_num_layers: 28,
            dec_num_heads: 16,
            dec_num_kv_heads: 8,
            dec_head_dim: 128,
            dec_intermediate_size: 3072,
            dec_vocab_size: 151_936,
            dec_rope_theta: 1_000_000.0,
            dec_rms_norm_eps: 1e-6,

            eos_token_ids: vec![151_643, 151_645],
            audio_token_id: 151_676,
        }
    }
}

impl ModelConfig {
    fn try_from_json(path: &Path) -> Result<Self> {
        let mut cfg = Self::default();
        let text = std::fs::read_to_string(path).context("reading config.json")?;
        let v: serde_json::Value = serde_json::from_str(&text)?;

        // HuggingFace config uses nested thinker_config.{audio_config, text_config}
        let thinker = v.get("thinker_config").unwrap_or(&v);

        if let Some(enc) = thinker
            .get("audio_config")
            .or_else(|| v.get("audio_encoder"))
        {
            if let Some(x) = enc.get("d_model").and_then(|x| x.as_i64()) {
                cfg.enc_d_model = x as i32;
            }
            if let Some(x) = enc.get("encoder_layers").and_then(|x| x.as_i64()) {
                cfg.enc_num_layers = x as usize;
            }
            if let Some(x) = enc.get("encoder_attention_heads").and_then(|x| x.as_i64()) {
                cfg.enc_num_heads = x as usize;
            }
            if let Some(x) = enc.get("encoder_ffn_dim").and_then(|x| x.as_i64()) {
                cfg.enc_ffn_dim = x as i32;
            }
            if let Some(x) = enc.get("output_dim").and_then(|x| x.as_i64()) {
                cfg.enc_output_dim = x as i32;
            }
            if let Some(x) = enc.get("downsample_hidden_size").and_then(|x| x.as_i64()) {
                cfg.enc_downsample_hidden = x as i32;
            }
            if let Some(x) = enc.get("n_window").and_then(|x| x.as_i64()) {
                cfg.enc_n_window = x as usize;
            }
            if let Some(x) = enc.get("n_window_infer").and_then(|x| x.as_i64()) {
                cfg.enc_n_window_infer = x as usize;
            }
        }

        if let Some(dec) = thinker.get("text_config") {
            if let Some(x) = dec.get("hidden_size").and_then(|x| x.as_i64()) {
                cfg.dec_hidden_size = x as i32;
            }
            if let Some(x) = dec.get("num_hidden_layers").and_then(|x| x.as_i64()) {
                cfg.dec_num_layers = x as usize;
            }
            if let Some(x) = dec.get("num_attention_heads").and_then(|x| x.as_i64()) {
                cfg.dec_num_heads = x as usize;
            }
            if let Some(x) = dec.get("num_key_value_heads").and_then(|x| x.as_i64()) {
                cfg.dec_num_kv_heads = x as usize;
            }
            if let Some(x) = dec.get("head_dim").and_then(|x| x.as_i64()) {
                cfg.dec_head_dim = x as i32;
            }
            if let Some(x) = dec.get("intermediate_size").and_then(|x| x.as_i64()) {
                cfg.dec_intermediate_size = x as i32;
            }
            if let Some(x) = dec.get("vocab_size").and_then(|x| x.as_i64()) {
                cfg.dec_vocab_size = x as i32;
            }
        }

        // audio_token_id lives directly under thinker_config
        if let Some(x) = thinker.get("audio_token_id").and_then(|x| x.as_u64()) {
            cfg.audio_token_id = x as u32;
        }

        // eos_token_id: may be scalar or array; always ensure both 151643 and 151645
        match v.get("eos_token_id") {
            Some(serde_json::Value::Array(arr)) => {
                cfg.eos_token_ids = arr
                    .iter()
                    .filter_map(|x| x.as_u64().map(|v| v as u32))
                    .collect();
            }
            Some(x) => {
                if let Some(id) = x.as_u64() {
                    cfg.eos_token_ids = vec![id as u32];
                }
            }
            None => {}
        }
        for &id in &[151_643u32, 151_645] {
            if !cfg.eos_token_ids.contains(&id) {
                cfg.eos_token_ids.push(id);
            }
        }

        Ok(cfg)
    }
}

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

// ─── Sinusoidal position embeddings (fixed) ─────────────────────────

fn sinusoidal_embeddings(max_len: i32, d_model: i32) -> Result<Array> {
    let half = d_model / 2;
    // Match the official Qwen3-ASR formula (SinusoidsPositionEmbedding):
    // log_timescale_increment = log(10000) / (half_dim - 1)
    let log_timescale_inc = (10_000.0f32).ln() / (half - 1).max(1) as f32;
    let mut data = vec![0.0f32; (max_len * d_model) as usize];
    for pos in 0..max_len {
        for i in 0..half {
            let inv_timescale = (-log_timescale_inc * i as f32).exp();
            let angle = pos as f32 * inv_timescale;
            let idx_base = (pos * d_model) as usize;
            // Blocked layout: [sin(0)..sin(half-1), cos(0)..cos(half-1)]
            data[idx_base + i as usize] = angle.sin();
            data[idx_base + half as usize + i as usize] = angle.cos();
        }
    }
    // Trim if d_model is odd
    Ok(Array::from_slice(&data, &[max_len, d_model]))
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

// ─── Audio Encoder ──────────────────────────────────────────────────

/// Take columns `start .. start+len` from a `[rows, cols]` array.
fn take_cols(arr: &Array, start: i32, len: i32) -> Result<Array> {
    let indices: Vec<i32> = (start..start + len).collect();
    let idx = Array::from_slice(&indices, &[len]);
    arr.take_axis(&idx, 1).map_err(Into::into)
}

/// Create a block-diagonal additive attention mask for windowed encoder attention.
///
/// Tokens within the same window attend to each other; cross-window positions
/// get `-inf`.  Returns `[1, 1, seq_len, seq_len]`.
fn create_windowed_mask(seq_len: i32, cu_seqlens: &[i32]) -> Result<Array> {
    let n = seq_len as usize;
    let neg_inf = f32::NEG_INFINITY;
    let mut data = vec![0.0f32; n * n];

    // Assign each position to a window
    let mut window_ids = vec![0u32; n];
    for (win_idx, pair) in cu_seqlens.windows(2).enumerate() {
        for pos in pair[0]..pair[1] {
            window_ids[pos as usize] = win_idx as u32;
        }
    }

    // Mask cross-window positions
    for i in 0..n {
        for j in 0..n {
            if window_ids[i] != window_ids[j] {
                data[i * n + j] = neg_inf;
            }
        }
    }

    let mask = Array::from_slice(&data, &[seq_len, seq_len]);
    mask.expand_dims_axes(&[0, 1]).map_err(Into::into)
}

struct AudioAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl AudioAttention {
    fn load(
        weights: &HashMap<String, Array>,
        prefix: &str,
        d_model: i32,
        num_heads: usize,
    ) -> Result<Self> {
        let head_dim = d_model as usize / num_heads;
        Ok(Self {
            q_proj: load_linear(weights, &format!("{prefix}.q_proj"), d_model, d_model, true)?,
            k_proj: load_linear(weights, &format!("{prefix}.k_proj"), d_model, d_model, true)?,
            v_proj: load_linear(weights, &format!("{prefix}.v_proj"), d_model, d_model, true)?,
            out_proj: load_linear(
                weights,
                &format!("{prefix}.out_proj"),
                d_model,
                d_model,
                true,
            )?,
            num_heads,
            head_dim,
        })
    }

    fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array> {
        let shape = x.shape();
        let (b, seq, _d) = (shape[0], shape[1], shape[2]);
        let nh = self.num_heads as i32;
        let hd = self.head_dim as i32;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape [B, S, D] → [B, S, nh, hd] → [B, nh, S, hd]
        let q = ops::reshape(&q, &[b, seq, nh, hd])?.transpose_axes(&[0, 2, 1, 3])?;
        let k = ops::reshape(&k, &[b, seq, nh, hd])?.transpose_axes(&[0, 2, 1, 3])?;
        let v = ops::reshape(&v, &[b, seq, nh, hd])?.transpose_axes(&[0, 2, 1, 3])?;

        let scale = Array::from_f32(1.0 / (hd as f32).sqrt());
        let mut scores =
            ops::multiply(&ops::matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)?, &scale)?;

        if let Some(m) = mask {
            scores = ops::add(&scores, m)?;
        }

        let attn = ops::softmax_axis(&scores, -1, None)?;
        let out = ops::matmul(&attn, &v)?;

        // [B, nh, S, hd] → [B, S, nh, hd] → [B, S, D]
        let out = out.transpose_axes(&[0, 2, 1, 3])?;
        let out = ops::reshape(&out, &[b, seq, nh * hd])?;
        Ok(self.out_proj.forward(&out)?)
    }
}

struct AudioMlp {
    fc1: Linear,
    fc2: Linear,
}

impl AudioMlp {
    fn load(
        weights: &HashMap<String, Array>,
        prefix: &str,
        d_model: i32,
        ffn_dim: i32,
    ) -> Result<Self> {
        Ok(Self {
            fc1: load_linear(weights, &format!("{prefix}.fc1"), d_model, ffn_dim, true)?,
            fc2: load_linear(weights, &format!("{prefix}.fc2"), ffn_dim, d_model, true)?,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        let h = self.fc1.forward(x)?;
        let h = mlx_rs::nn::gelu(&h)?;
        Ok(self.fc2.forward(&h)?)
    }
}

struct AudioEncoderLayer {
    self_attn: AudioAttention,
    self_attn_layer_norm: LayerNorm,
    mlp: AudioMlp,
    final_layer_norm: LayerNorm,
}

impl AudioEncoderLayer {
    fn load(
        weights: &HashMap<String, Array>,
        prefix: &str,
        d_model: i32,
        num_heads: usize,
        ffn_dim: i32,
    ) -> Result<Self> {
        Ok(Self {
            self_attn: AudioAttention::load(
                weights,
                &format!("{prefix}.self_attn"),
                d_model,
                num_heads,
            )?,
            self_attn_layer_norm: load_layer_norm(
                weights,
                &format!("{prefix}.self_attn_layer_norm"),
                d_model,
            )?,
            mlp: AudioMlp::load(weights, &format!("{prefix}"), d_model, ffn_dim)?,
            final_layer_norm: load_layer_norm(
                weights,
                &format!("{prefix}.final_layer_norm"),
                d_model,
            )?,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        self.forward_impl(x, None)
    }

    fn forward_masked(&mut self, x: &Array, mask: &Array) -> Result<Array> {
        self.forward_impl(x, Some(mask))
    }

    fn forward_impl(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array> {
        // Pre-norm attention
        let residual = x.clone();
        let h = self.self_attn_layer_norm.forward(x)?;
        let h = self.self_attn.forward(&h, mask)?;
        let x = ops::add(&residual, &h)?;

        // Pre-norm FFN
        let residual = x.clone();
        let h = self.final_layer_norm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        ops::add(&residual, &h).map_err(Into::into)
    }
}

struct AudioEncoder {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv_out: Linear,
    layers: Vec<AudioEncoderLayer>,
    post_ln: LayerNorm,
    proj1: Linear,
    proj2: Linear,
    config: ModelConfig,
}

impl AudioEncoder {
    fn load(weights: &HashMap<String, Array>, cfg: &ModelConfig) -> Result<Self> {
        let p = "audio_tower";
        let dh = cfg.enc_downsample_hidden;

        let mut layers = Vec::with_capacity(cfg.enc_num_layers);
        for i in 0..cfg.enc_num_layers {
            layers.push(AudioEncoderLayer::load(
                weights,
                &format!("{p}.layers.{i}"),
                cfg.enc_d_model,
                cfg.enc_num_heads,
                cfg.enc_ffn_dim,
            )?);
        }

        // freq_after_conv = N_MELS / 8 = 16   (three stride-2 convs on mel axis)
        let freq_after_conv = N_MELS as i32 / 8;

        Ok(Self {
            conv1: load_conv2d(weights, &format!("{p}.conv2d1"), 1, dh, 3, 2, 1)?,
            conv2: load_conv2d(weights, &format!("{p}.conv2d2"), dh, dh, 3, 2, 1)?,
            conv3: load_conv2d(weights, &format!("{p}.conv2d3"), dh, dh, 3, 2, 1)?,
            conv_out: load_linear(
                weights,
                &format!("{p}.conv_out"),
                dh * freq_after_conv,
                cfg.enc_d_model,
                false,
            )?,
            layers,
            post_ln: load_layer_norm(weights, &format!("{p}.layer_norm"), cfg.enc_d_model)?,
            proj1: load_linear(
                weights,
                &format!("{p}.proj1"),
                cfg.enc_d_model,
                cfg.enc_d_model,
                false,
            )?,
            proj2: load_linear(
                weights,
                &format!("{p}.proj2"),
                cfg.enc_d_model,
                cfg.enc_output_dim,
                false,
            )?,
            config: cfg.clone(),
        })
    }

    /// Encode a mel spectrogram into audio embeddings.
    ///
    /// `mel_flat` is `[n_mels, n_frames]` row-major from [`mel::whisper_mel`].
    /// Returns `[1, audio_tokens, output_dim]`.
    fn forward(&mut self, mel_flat: &[f32], n_frames: usize) -> Result<Array> {
        let n_mels = N_MELS as i32;
        let chunk_size = (self.config.enc_n_window * 2) as i32; // e.g. 100

        // mel_flat is [n_mels, n_frames] row-major.
        let mel = Array::from_slice(mel_flat, &[n_mels, n_frames as i32]);

        // ── Per-chunk Conv2d processing ──
        // Split mel into chunks of chunk_size frames along the time axis.
        // Each chunk is processed independently through the conv stem.
        let total_frames = n_frames as i32;
        let n_full_chunks = total_frames / chunk_size;
        let tail_frames = total_frames - n_full_chunks * chunk_size;

        let mut chunk_conv_outputs: Vec<Array> = Vec::new();
        let mut chunk_token_lens: Vec<i32> = Vec::new();

        // Process full-size chunks
        for c in 0..n_full_chunks {
            let start = c * chunk_size;
            let chunk_mel = take_cols(&mel, start, chunk_size)?;
            let (conv_out, t_tokens) = self.conv_stem_single(&chunk_mel, n_mels, chunk_size)?;
            chunk_conv_outputs.push(conv_out);
            chunk_token_lens.push(t_tokens);
        }

        // Process tail chunk (if any)
        if tail_frames > 0 {
            let start = n_full_chunks * chunk_size;
            let chunk_mel = take_cols(&mel, start, tail_frames)?;
            let (conv_out, t_tokens) = self.conv_stem_single(&chunk_mel, n_mels, tail_frames)?;
            chunk_conv_outputs.push(conv_out);
            chunk_token_lens.push(t_tokens);
        }

        // Concatenate all chunks → [total_tokens, d_model]
        let refs: Vec<&Array> = chunk_conv_outputs.iter().collect();
        let mut x = if refs.len() == 1 {
            refs[0].clone()
        } else {
            ops::concatenate_axis(refs.as_slice(), 0)?
        };

        // ── Per-chunk sinusoidal position embeddings ──
        // Each chunk gets PE starting from position 0 (matching the official encoder).
        let max_chunk_tokens = *chunk_token_lens.iter().max().unwrap_or(&1);
        let pe = sinusoidal_embeddings(max_chunk_tokens, self.config.enc_d_model)?;

        let mut pe_parts: Vec<Array> = Vec::new();
        for &ct in &chunk_token_lens {
            let indices: Vec<i32> = (0..ct).collect();
            let idx = Array::from_slice(&indices, &[ct]);
            pe_parts.push(pe.take_axis(&idx, 0)?);
        }
        let pe_refs: Vec<&Array> = pe_parts.iter().collect();
        let pe_full = if pe_refs.len() == 1 {
            pe_refs[0].clone()
        } else {
            ops::concatenate_axis(pe_refs.as_slice(), 0)?
        };
        x = ops::add(&x, &pe_full)?;

        // ── Windowed attention ──
        // Build window boundaries for block-diagonal attention.
        // n_window_infer controls how many mel frames each attention window spans.
        // tokens_per_window = tokens_per_chunk * (n_window_infer / chunk_size)
        let total_tokens = x.shape()[0];
        let tokens_per_window = if !chunk_token_lens.is_empty() {
            let n_window_infer = self.config.enc_n_window_infer as i32;
            chunk_token_lens[0] * (n_window_infer / chunk_size).max(1)
        } else {
            total_tokens
        };

        let mut cu_seqlens: Vec<i32> = vec![0];
        let mut pos = 0i32;
        while pos < total_tokens {
            let end = (pos + tokens_per_window).min(total_tokens);
            cu_seqlens.push(end);
            pos = end;
        }

        // Add batch dim: [total_tokens, d_model] → [1, total_tokens, d_model]
        let mut h = x.expand_dims(0)?;

        // Apply transformer layers with windowed attention
        let num_windows = cu_seqlens.len() - 1;
        if num_windows <= 1 {
            // Single window: no mask needed
            for layer in &mut self.layers {
                h = layer.forward(&h)?;
            }
        } else {
            // Create block-diagonal mask
            let mask = create_windowed_mask(total_tokens, &cu_seqlens)?;
            for layer in &mut self.layers {
                h = layer.forward_masked(&h, &mask)?;
            }
        }

        // Remove batch dim for post-processing
        let h = ops::squeeze_axes(&h, &[0])?; // [total_tokens, d_model]
        let h = h.expand_dims(0)?; // back to [1, total_tokens, d_model]

        // ── Post-processing: LayerNorm → proj1 → GELU → proj2 ──
        let mut out = self.post_ln.forward(&h)?;
        out = self.proj1.forward(&out)?;
        out = mlx_rs::nn::gelu(&out)?;
        out = self.proj2.forward(&out)?;

        Ok(out)
    }

    /// Run the Conv2d stem on a single mel chunk.
    ///
    /// `chunk_mel`: `[n_mels, chunk_frames]`.
    /// Returns `(features, n_tokens)` where features is `[n_tokens, d_model]`.
    fn conv_stem_single(
        &mut self,
        chunk_mel: &Array,
        _n_mels: i32,
        _chunk_frames: i32,
    ) -> Result<(Array, i32)> {
        // NHWC input: [1, H=n_mels, W=chunk_frames, C=1]
        let x = chunk_mel.expand_dims(0)?.expand_dims(-1)?;

        // Conv2d stem with GELU
        let mut h = self.conv1.forward(&x)?;
        h = mlx_rs::nn::gelu(&h)?;
        h = self.conv2.forward(&h)?;
        h = mlx_rs::nn::gelu(&h)?;
        h = self.conv3.forward(&h)?;
        h = mlx_rs::nn::gelu(&h)?;

        // h: [1, F'=n_mels/8, T'=time_tokens, C=dhs]
        let sh = h.shape().to_vec();
        let (_b, f_d, t_d, c_d) = (sh[0], sh[1], sh[2], sh[3]);

        // Channel-major reshape matching PyTorch convention:
        // [1, F', T', C] → transpose(0,2,3,1) → [1, T', C, F'] → reshape → [T', C*F']
        let h = h.transpose_axes(&[0, 2, 3, 1])?; // [1, T', C, F']
        let h = ops::reshape(&h, &[t_d, c_d * f_d])?; // [T', C*F']

        // Linear projection → [T', d_model]
        let h = self.conv_out.forward(&h)?;

        Ok((h, t_d))
    }
}

// ─── Text Decoder ───────────────────────────────────────────────────

struct TextAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl TextAttention {
    fn load(weights: &HashMap<String, Array>, prefix: &str, cfg: &ModelConfig) -> Result<Self> {
        let h = cfg.dec_hidden_size;
        let qd = cfg.dec_num_heads as i32 * cfg.dec_head_dim;
        let kvd = cfg.dec_num_kv_heads as i32 * cfg.dec_head_dim;
        Ok(Self {
            q_proj: load_linear(weights, &format!("{prefix}.q_proj"), h, qd, false)?,
            k_proj: load_linear(weights, &format!("{prefix}.k_proj"), h, kvd, false)?,
            v_proj: load_linear(weights, &format!("{prefix}.v_proj"), h, kvd, false)?,
            o_proj: load_linear(weights, &format!("{prefix}.o_proj"), qd, h, false)?,
            q_norm: load_rms_norm(
                weights,
                &format!("{prefix}.q_norm"),
                cfg.dec_head_dim,
                cfg.dec_rms_norm_eps,
            )?,
            k_norm: load_rms_norm(
                weights,
                &format!("{prefix}.k_norm"),
                cfg.dec_head_dim,
                cfg.dec_rms_norm_eps,
            )?,
            num_heads: cfg.dec_num_heads,
            num_kv_heads: cfg.dec_num_kv_heads,
            head_dim: cfg.dec_head_dim as usize,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        rope_cos: &Array,
        rope_sin: &Array,
        cache: Option<&mut LayerKvCache>,
        offset: i32,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let shape = x.shape();
        let (b, seq) = (shape[0], shape[1]);
        let nh = self.num_heads as i32;
        let nkv = self.num_kv_heads as i32;
        let hd = self.head_dim as i32;
        let gqa_groups = nh / nkv;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // [B, S, nh*hd] → [B, S, nh, hd] → [B, nh, S, hd]
        let mut q = ops::reshape(&q, &[b, seq, nh, hd])?.transpose_axes(&[0, 2, 1, 3])?;
        let mut k = ops::reshape(&k, &[b, seq, nkv, hd])?.transpose_axes(&[0, 2, 1, 3])?;
        let v = ops::reshape(&v, &[b, seq, nkv, hd])?.transpose_axes(&[0, 2, 1, 3])?;

        // Per-head RMSNorm on Q and K: apply to each [B, heads, S, hd]
        // RmsNorm expects [..., dims], so reshape to apply per-head
        q = apply_rms_norm_per_head(&mut self.q_norm, &q)?;
        k = apply_rms_norm_per_head(&mut self.k_norm, &k)?;

        // RoPE
        q = apply_rope(&q, rope_cos, rope_sin, offset)?;
        k = apply_rope(&k, rope_cos, rope_sin, offset)?;

        // KV cache
        let (k, v) = if let Some(kv) = cache {
            kv.append(k, v)?
        } else {
            (k, v)
        };

        // GQA: repeat KV heads to match Q heads
        let k = if gqa_groups > 1 {
            repeat_kv(&k, gqa_groups)?
        } else {
            k
        };
        let v = if gqa_groups > 1 {
            repeat_kv(&v, gqa_groups)?
        } else {
            v
        };

        // Scaled dot-product attention
        let scale = Array::from_f32(1.0 / (hd as f32).sqrt());
        let mut scores =
            ops::multiply(&ops::matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)?, &scale)?;

        if let Some(m) = mask {
            scores = ops::add(&scores, m)?;
        }

        let attn = ops::softmax_axis(&scores, -1, None)?;
        let out = ops::matmul(&attn, &v)?;

        let out = out.transpose_axes(&[0, 2, 1, 3])?;
        let out = ops::reshape(&out, &[b, seq, nh * hd])?;
        Ok(self.o_proj.forward(&out)?)
    }
}

/// Apply RmsNorm to each head in a [B, heads, S, hd] tensor.
fn apply_rms_norm_per_head(norm: &mut RmsNorm, x: &Array) -> Result<Array> {
    let shape = x.shape().to_vec();
    let (b, nh, s, hd) = (shape[0], shape[1], shape[2], shape[3]);
    // Flatten to [B*nh*S, hd], apply norm, reshape back
    let flat = ops::reshape(x, &[b * nh * s, hd])?;
    let normed = norm.forward(&flat)?;
    ops::reshape(&normed, &shape).map_err(Into::into)
}

/// Repeat KV heads: [B, nkv, S, hd] → [B, nh, S, hd] where nh = nkv * groups.
fn repeat_kv(x: &Array, groups: i32) -> Result<Array> {
    let shape = x.shape();
    let (b, nkv, s, hd) = (shape[0], shape[1], shape[2], shape[3]);
    // [B, nkv, S, hd] → [B, nkv, 1, S, hd] → broadcast [B, nkv, groups, S, hd] → [B, nkv*groups, S, hd]
    let x = x.expand_dims(2)?; // [B, nkv, 1, S, hd]
    let expanded = ops::broadcast_to(&x, &[b, nkv, groups, s, hd])?;
    ops::reshape(&expanded, &[b, nkv * groups, s, hd]).map_err(Into::into)
}

struct TextMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl TextMlp {
    fn load(
        weights: &HashMap<String, Array>,
        prefix: &str,
        hidden: i32,
        intermediate: i32,
    ) -> Result<Self> {
        Ok(Self {
            gate_proj: load_linear(
                weights,
                &format!("{prefix}.gate_proj"),
                hidden,
                intermediate,
                false,
            )?,
            up_proj: load_linear(
                weights,
                &format!("{prefix}.up_proj"),
                hidden,
                intermediate,
                false,
            )?,
            down_proj: load_linear(
                weights,
                &format!("{prefix}.down_proj"),
                intermediate,
                hidden,
                false,
            )?,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        let gate = self.gate_proj.forward(x)?;
        let gate = mlx_rs::nn::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        let h = ops::multiply(&gate, &up)?;
        Ok(self.down_proj.forward(&h)?)
    }
}

struct TextDecoderLayer {
    self_attn: TextAttention,
    mlp: TextMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl TextDecoderLayer {
    fn load(weights: &HashMap<String, Array>, prefix: &str, cfg: &ModelConfig) -> Result<Self> {
        Ok(Self {
            self_attn: TextAttention::load(weights, &format!("{prefix}.self_attn"), cfg)?,
            mlp: TextMlp::load(
                weights,
                &format!("{prefix}.mlp"),
                cfg.dec_hidden_size,
                cfg.dec_intermediate_size,
            )?,
            input_layernorm: load_rms_norm(
                weights,
                &format!("{prefix}.input_layernorm"),
                cfg.dec_hidden_size,
                cfg.dec_rms_norm_eps,
            )?,
            post_attention_layernorm: load_rms_norm(
                weights,
                &format!("{prefix}.post_attention_layernorm"),
                cfg.dec_hidden_size,
                cfg.dec_rms_norm_eps,
            )?,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        rope_cos: &Array,
        rope_sin: &Array,
        cache: Option<&mut LayerKvCache>,
        offset: i32,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let h = self
            .self_attn
            .forward(&h, rope_cos, rope_sin, cache, offset, mask)?;
        let x = ops::add(&residual, &h)?;

        let residual = x.clone();
        let h = self.post_attention_layernorm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        ops::add(&residual, &h).map_err(Into::into)
    }
}

struct TextDecoder {
    embed_tokens: Embedding,
    layers: Vec<TextDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    #[allow(dead_code)]
    config: ModelConfig,
}

impl TextDecoder {
    fn load(weights: &HashMap<String, Array>, cfg: &ModelConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.dec_num_layers);
        for i in 0..cfg.dec_num_layers {
            layers.push(TextDecoderLayer::load(
                weights,
                &format!("model.layers.{i}"),
                cfg,
            )?);
        }

        // lm_head may be tied to embed_tokens
        let has_lm_head = weights.contains_key("lm_head.weight");
        let lm_head = if has_lm_head {
            load_linear(
                weights,
                "lm_head",
                cfg.dec_hidden_size,
                cfg.dec_vocab_size,
                false,
            )?
        } else {
            // Tie to embed_tokens
            let mut lh = Linear::new(cfg.dec_hidden_size, cfg.dec_vocab_size)?;
            lh.weight = Param::new(w(weights, "model.embed_tokens.weight")?);
            lh.bias = Param::new(None);
            lh
        };

        Ok(Self {
            embed_tokens: load_embedding(
                weights,
                "model.embed_tokens",
                cfg.dec_vocab_size,
                cfg.dec_hidden_size,
            )?,
            layers,
            norm: load_rms_norm(
                weights,
                "model.norm",
                cfg.dec_hidden_size,
                cfg.dec_rms_norm_eps,
            )?,
            lm_head,
            config: cfg.clone(),
        })
    }

    /// Run the decoder on a sequence of token IDs.
    /// Returns logits `[B, S, vocab]`.
    fn forward(
        &mut self,
        token_ids: &Array,
        rope_cos: &Array,
        rope_sin: &Array,
        cache: &mut KvCache,
        offset: i32,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let mut h = self.embed_tokens.forward(token_ids)?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let layer_cache = cache.layers.get_mut(i);
            h = layer.forward(&h, rope_cos, rope_sin, layer_cache, offset, mask)?;
        }

        h = self.norm.forward(&h)?;
        Ok(self.lm_head.forward(&h)?)
    }

    /// Run the decoder on pre-computed hidden states (e.g. audio embeddings).
    /// Returns logits `[B, S, vocab]`.
    fn forward_embeds(
        &mut self,
        embeds: &Array,
        rope_cos: &Array,
        rope_sin: &Array,
        cache: &mut KvCache,
        offset: i32,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let mut h = embeds.clone();

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let layer_cache = cache.layers.get_mut(i);
            h = layer.forward(&h, rope_cos, rope_sin, layer_cache, offset, mask)?;
        }

        h = self.norm.forward(&h)?;
        Ok(self.lm_head.forward(&h)?)
    }
}

// ─── KV Cache ───────────────────────────────────────────────────────

struct LayerKvCache {
    key: Option<Array>,
    value: Option<Array>,
}

impl LayerKvCache {
    fn new() -> Self {
        Self {
            key: None,
            value: None,
        }
    }

    fn append(&mut self, k: Array, v: Array) -> Result<(Array, Array)> {
        let new_k = match self.key.take() {
            Some(prev) => ops::concatenate_axis(&[&prev, &k], 2)?,
            None => k,
        };
        let new_v = match self.value.take() {
            Some(prev) => ops::concatenate_axis(&[&prev, &v], 2)?,
            None => v,
        };
        self.key = Some(new_k.clone());
        self.value = Some(new_v.clone());
        Ok((new_k, new_v))
    }
}

struct KvCache {
    layers: Vec<LayerKvCache>,
}

impl KvCache {
    fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| LayerKvCache::new()).collect(),
        }
    }
}

// ─── Full model ─────────────────────────────────────────────────────

struct Qwen3AsrModel {
    encoder: AudioEncoder,
    decoder: TextDecoder,
    rope_cos: Array,
    rope_sin: Array,
    config: ModelConfig,
}

impl Qwen3AsrModel {
    fn load(model_dir: &Path) -> Result<Self> {
        let cfg_path = model_dir.join("config.json");
        let config = if cfg_path.exists() {
            ModelConfig::try_from_json(&cfg_path)?
        } else {
            ModelConfig::default()
        };

        // Load all safetensors shards
        let weights = load_all_safetensors(model_dir)?;
        log::info!(
            "loaded {} weight tensors from {}",
            weights.len(),
            model_dir.display()
        );

        let encoder = AudioEncoder::load(&weights, &config)?;
        let decoder = TextDecoder::load(&weights, &config)?;

        // Pre-compute RoPE frequencies for the max context length
        let max_ctx = 2048i32;
        let (rope_cos, rope_sin) =
            build_rope_freqs(config.dec_head_dim, max_ctx, config.dec_rope_theta)?;

        Ok(Self {
            encoder,
            decoder,
            rope_cos,
            rope_sin,
            config,
        })
    }

    /// Build a causal attention mask of shape `[1, 1, seq, seq]`.
    fn causal_mask(seq_len: i32) -> Result<Array> {
        let neg_inf = f32::NEG_INFINITY;
        let mut data = vec![0.0f32; (seq_len * seq_len) as usize];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                data[(i * seq_len + j) as usize] = neg_inf;
            }
        }
        let mask = Array::from_slice(&data, &[seq_len, seq_len]);
        // [seq, seq] → [1, 1, seq, seq]
        mask.expand_dims_axes(&[0, 1]).map_err(Into::into)
    }

    /// Full transcription pipeline: encode audio → prefill → greedy decode.
    fn transcribe(&mut self, samples: &[f32], abort_flag: &Arc<AtomicBool>) -> Result<Vec<u32>> {
        // Chat-template token IDs (identical across all Qwen3-ASR sizes)
        const IM_START: i32 = 151644;
        const IM_END: i32 = 151645;
        const AUDIO_START: i32 = 151669;
        const AUDIO_END: i32 = 151670;
        const NEWLINE: i32 = 198;
        const SYSTEM: i32 = 9125;
        const USER: i32 = 882;
        const ASSISTANT: i32 = 77091;

        // 1. Mel spectrogram
        let mel_flat = mel::whisper_mel(samples);
        let n_frames = mel::mel_frame_count(samples.len());

        // 2. Encode audio
        let audio_embeds = self.encoder.forward(&mel_flat, n_frames)?;
        // audio_embeds: [1, audio_tokens, output_dim]
        log::debug!("MLX encoder output shape: {:?}", audio_embeds.shape());

        // 3. Build prefix tokens:
        //    <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
        let prefix_ids_vec = vec![
            IM_START,
            SYSTEM,
            NEWLINE,
            IM_END,
            NEWLINE,
            IM_START,
            USER,
            NEWLINE,
            AUDIO_START,
        ];
        let prefix_len_tok = prefix_ids_vec.len() as i32;
        let prefix_ids = Array::from_slice(&prefix_ids_vec, &[1, prefix_len_tok]);
        let prefix_emb = self.decoder.embed_tokens.forward(&prefix_ids)?;

        // 4. Build suffix tokens:
        //    <|audio_end|><|im_end|>\n<|im_start|>assistant\n
        let suffix_ids_vec = vec![AUDIO_END, IM_END, NEWLINE, IM_START, ASSISTANT, NEWLINE];
        let suffix_len_tok = suffix_ids_vec.len() as i32;
        let suffix_ids = Array::from_slice(&suffix_ids_vec, &[1, suffix_len_tok]);
        let suffix_emb = self.decoder.embed_tokens.forward(&suffix_ids)?;

        // 5. Concatenate: prefix_emb + audio_embeds + suffix_emb → [1, total, hidden]
        let prefix_embeds = ops::concatenate_axis(&[&prefix_emb, &audio_embeds, &suffix_emb], 1)?;
        let prefix_len = prefix_embeds.shape()[1];

        // 6. Prefill: run through decoder with causal mask
        let mut cache = KvCache::new(self.config.dec_num_layers);
        let mask = Self::causal_mask(prefix_len)?;
        let logits = self.decoder.forward_embeds(
            &prefix_embeds,
            &self.rope_cos,
            &self.rope_sin,
            &mut cache,
            0,
            Some(&mask),
        )?;

        // 7. Greedy decode from the last logits position
        let mut generated = Vec::new();
        let mut next_token = argmax_last_token(&logits)?;
        let mut offset = prefix_len;

        log::debug!(
            "MLX prefill done: prefix_len={prefix_len}, first_token={next_token}, eos={:?}",
            self.config.eos_token_ids
        );

        for step in 0..MAX_GEN_TOKENS {
            if abort_flag.load(Ordering::Relaxed) {
                break;
            }

            let tok_id = next_token;
            if self.config.eos_token_ids.contains(&tok_id) {
                log::debug!("MLX decode: EOS at step {step}, token {tok_id}");
                break;
            }
            generated.push(tok_id);

            // Feed the token through the decoder (single step, no mask needed with KV cache)
            let tok_arr = Array::from_slice(&[tok_id as i32], &[1, 1]);
            let step_logits = self.decoder.forward(
                &tok_arr,
                &self.rope_cos,
                &self.rope_sin,
                &mut cache,
                offset as i32,
                None,
            )?;
            offset += 1;

            next_token = argmax_last_token(&step_logits)?;
        }

        log::debug!(
            "MLX decode: generated {} tokens: {:?}",
            generated.len(),
            &generated[..generated.len().min(10)]
        );
        Ok(generated)
    }
}

/// Pick the argmax of the last token's logits → returns a token ID.
fn argmax_last_token(logits: &Array) -> Result<u32> {
    // logits: [B, S, vocab] → take last position → [B, vocab]
    let shape = logits.shape();
    let seq_len = shape[1];
    let idx = Array::from_slice(&[seq_len - 1], &[1]);
    let last = logits.take_axis(&idx, 1)?; // [1, 1, vocab]
    let last = ops::squeeze_axes(&last, &[1])?; // [1, vocab]
    let token = ops::indexing::argmax_axis(&last, -1, None)?; // [1]
    token.eval()?;
    let tok_id: i32 = token.try_item()?;
    Ok(tok_id as u32)
}

/// Load all `*.safetensors` files from a directory into one flat HashMap.
fn load_all_safetensors(dir: &Path) -> Result<HashMap<String, Array>> {
    let mut all = HashMap::new();
    let mut paths: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
        .collect();
    paths.sort();

    if paths.is_empty() {
        anyhow::bail!("no .safetensors files found in {}", dir.display());
    }

    for path in &paths {
        log::debug!("loading weights from {}", path.display());
        let shard = Array::load_safetensors(path)?;
        all.extend(shard);
    }
    Ok(all)
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
