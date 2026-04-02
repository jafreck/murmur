use anyhow::{Context, Result};
use std::path::Path;

#[derive(Debug, Clone)]
pub(crate) struct ModelConfig {
    // Audio encoder
    pub(crate) enc_d_model: i32,
    pub(crate) enc_num_layers: usize,
    pub(crate) enc_num_heads: usize,
    pub(crate) enc_ffn_dim: i32,
    pub(crate) enc_output_dim: i32,
    pub(crate) enc_downsample_hidden: i32,
    pub(crate) enc_n_window: usize,
    pub(crate) enc_n_window_infer: usize,

    // Text decoder
    pub(crate) dec_hidden_size: i32,
    pub(crate) dec_num_layers: usize,
    pub(crate) dec_num_heads: usize,
    pub(crate) dec_num_kv_heads: usize,
    pub(crate) dec_head_dim: i32,
    pub(crate) dec_intermediate_size: i32,
    pub(crate) dec_vocab_size: i32,
    pub(crate) dec_rope_theta: f32,
    pub(crate) dec_rms_norm_eps: f32,

    // Special token IDs
    pub(crate) eos_token_ids: Vec<u32>,
    pub(crate) audio_token_id: u32,
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
    pub(crate) fn try_from_json(path: &Path) -> Result<Self> {
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
