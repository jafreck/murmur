use super::config::ModelConfig;
use super::{load_conv2d, load_layer_norm, load_linear, N_MELS};
use anyhow::Result;
use mlx_rs::module::Module;
use mlx_rs::nn::{Conv2d, LayerNorm, Linear};
use mlx_rs::ops;
use mlx_rs::Array;
use std::collections::HashMap;

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

// ─── Audio Encoder helpers ──────────────────────────────────────────

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

// ─── Audio Encoder structs ──────────────────────────────────────────

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

pub(crate) struct AudioEncoder {
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
    pub(crate) fn load(weights: &HashMap<String, Array>, cfg: &ModelConfig) -> Result<Self> {
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
    pub(crate) fn forward(&mut self, mel_flat: &[f32], n_frames: usize) -> Result<Array> {
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
