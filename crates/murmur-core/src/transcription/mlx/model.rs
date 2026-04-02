use super::cache::KvCache;
use super::config::ModelConfig;
use super::decoder::TextDecoder;
use super::encoder::AudioEncoder;
use super::{build_rope_freqs, MAX_GEN_TOKENS};
use crate::transcription::mel;
use anyhow::Result;
use mlx_rs::module::Module;
use mlx_rs::ops;
use mlx_rs::Array;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub(crate) struct Qwen3AsrModel {
    encoder: AudioEncoder,
    decoder: TextDecoder,
    rope_cos: Array,
    rope_sin: Array,
    config: ModelConfig,
}

impl Qwen3AsrModel {
    pub(crate) fn load(model_dir: &Path) -> Result<Self> {
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
    pub(crate) fn transcribe(
        &mut self,
        samples: &[f32],
        abort_flag: &Arc<AtomicBool>,
    ) -> Result<Vec<u32>> {
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
