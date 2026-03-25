use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::config::Config;

pub struct Transcriber {
    ctx: WhisperContext,
    language: String,
}

impl Transcriber {
    pub fn new(model_path: &Path, language: &str) -> Result<Self> {
        let ctx = WhisperContext::new_with_params(
            model_path.to_str().context("Invalid model path")?,
            WhisperContextParameters::default(),
        )
        .map_err(|e| anyhow::anyhow!("Failed to load Whisper model: {e}"))?;

        Ok(Self {
            ctx,
            language: language.to_string(),
        })
    }

    pub fn transcribe(&self, audio_path: &Path, translate: bool) -> Result<String> {
        // Read WAV file into f32 samples
        let reader = hound::WavReader::open(audio_path)
            .context("Failed to open audio file")?;

        let spec = reader.spec();
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .into_samples::<i32>()
                    .filter_map(|s| s.ok())
                    .map(|s| s as f32 / max_val)
                    .collect()
            }
            hound::SampleFormat::Float => {
                reader
                    .into_samples::<f32>()
                    .filter_map(|s| s.ok())
                    .collect()
            }
        };

        if samples.is_empty() {
            return Ok(String::new());
        }

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some(&self.language));
        params.set_translate(translate);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        let mut state = self.ctx.create_state()
            .map_err(|e| anyhow::anyhow!("Failed to create whisper state: {e}"))?;

        state
            .full(params, &samples)
            .map_err(|e| anyhow::anyhow!("Transcription failed: {e}"))?;

        let num_segments = state.full_n_segments();
        if num_segments < 0 {
            anyhow::bail!("Failed to get segments");
        }

        let mut text = String::new();
        for i in 0..num_segments {
            if let Some(segment) = state.get_segment(i) {
                if let Ok(segment_text) = segment.to_str_lossy() {
                    text.push_str(&segment_text);
                }
            }
        }

        Ok(text.trim().to_string())
    }
}

/// Check if a model file exists in any known location.
pub fn model_exists(model_size: &str) -> bool {
    find_model(model_size).is_some()
}

/// Find a model file in known locations.
pub fn find_model(model_size: &str) -> Option<PathBuf> {
    let model_filename = format!("ggml-{model_size}.bin");

    let candidates = vec![
        // App config directory
        Config::dir().join("models").join(&model_filename),
        // Common locations
        dirs::data_dir()
            .unwrap_or_default()
            .join("whisper-cpp")
            .join("models")
            .join(&model_filename),
        dirs::home_dir()
            .unwrap_or_default()
            .join(".cache")
            .join("whisper")
            .join(&model_filename),
    ];

    // macOS-specific Homebrew paths
    #[cfg(target_os = "macos")]
    let candidates = {
        let mut c = candidates;
        c.push(PathBuf::from(format!(
            "/opt/homebrew/share/whisper-cpp/models/{model_filename}"
        )));
        c.push(PathBuf::from(format!(
            "/usr/local/share/whisper-cpp/models/{model_filename}"
        )));
        c
    };

    candidates.into_iter().find(|p| p.exists())
}
