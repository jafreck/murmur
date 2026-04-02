use super::cache::{KvCache, LayerKvCache};
use super::config::ModelConfig;
use super::{apply_rope, load_embedding, load_linear, load_rms_norm, w};
use anyhow::Result;
use mlx_rs::module::{Module, Param};
use mlx_rs::nn::{Embedding, Linear, RmsNorm};
use mlx_rs::ops;
use mlx_rs::Array;
use std::collections::HashMap;

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

pub(crate) struct TextDecoder {
    pub(crate) embed_tokens: Embedding,
    layers: Vec<TextDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    #[allow(dead_code)]
    config: ModelConfig,
}

impl TextDecoder {
    pub(crate) fn load(weights: &HashMap<String, Array>, cfg: &ModelConfig) -> Result<Self> {
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
    pub(crate) fn forward(
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
    pub(crate) fn forward_embeds(
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
