use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::config::Config;

pub struct Transcriber {
    ctx: WhisperContext,
    language: String,
}

/// Read a WAV file and return f32 samples normalized to [-1.0, 1.0].
pub fn read_wav_samples(audio_path: &Path) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(audio_path).context("Failed to open audio file")?;

    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1_i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("Failed to decode integer WAV samples")?
                .into_iter()
                .map(|s| s as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("Failed to decode float WAV samples")?,
    };

    Ok(samples)
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

    /// Return the language parameter for whisper. `None` means auto-detect.
    fn language_param(&self) -> Option<&str> {
        if self.language == "auto" {
            None
        } else {
            Some(&self.language)
        }
    }

    pub fn transcribe(&self, audio_path: &Path, translate: bool) -> Result<String> {
        let samples = read_wav_samples(audio_path)?;
        self.run_inference(&samples, translate)
    }

    /// Transcribe raw 16 kHz mono f32 samples directly (no file I/O).
    pub fn transcribe_samples(&self, samples: &[f32], translate: bool) -> Result<String> {
        self.run_inference(samples, translate)
    }

    fn run_inference(&self, samples: &[f32], translate: bool) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(self.language_param());
        params.set_translate(translate);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| anyhow::anyhow!("Failed to create whisper state: {e}"))?;

        state
            .full(params, samples)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_exists_nonexistent() {
        assert!(!model_exists("nonexistent_model_that_doesnt_exist_xyz"));
    }

    #[test]
    fn test_find_model_nonexistent() {
        assert!(find_model("nonexistent_model_that_doesnt_exist_xyz").is_none());
    }

    #[test]
    fn test_find_model_builds_correct_filename() {
        // We can't guarantee a model exists, but we can verify the function runs
        // without panicking and returns None for a clearly nonexistent model
        let result = find_model("test_does_not_exist");
        assert!(result.is_none());
    }

    #[test]
    fn test_find_model_checks_config_dir() {
        let models_dir = Config::dir().join("models");
        let _ = std::fs::create_dir_all(&models_dir);
        let model_path = models_dir.join("ggml-test_temp_model.bin");
        std::fs::write(&model_path, b"test model content").unwrap();

        let result = find_model("test_temp_model");
        assert!(result.is_some());
        assert_eq!(result.unwrap(), model_path);

        let _ = std::fs::remove_file(&model_path);
    }

    // -- read_wav_samples --

    #[test]
    fn test_read_wav_samples_int16() {
        use hound::{SampleFormat, WavSpec, WavWriter};
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let spec = WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(tmp.path(), spec).unwrap();
        writer.write_sample(0i16).unwrap();
        writer.write_sample(16384i16).unwrap();
        writer.write_sample(-16384i16).unwrap();
        writer.finalize().unwrap();

        let samples = read_wav_samples(tmp.path()).unwrap();
        assert_eq!(samples.len(), 3);
        assert!((samples[0] - 0.0).abs() < 0.01);
        assert!(samples[1] > 0.4 && samples[1] < 0.6);
        assert!(samples[2] < -0.4 && samples[2] > -0.6);
    }

    #[test]
    fn test_read_wav_samples_float32() {
        use hound::{SampleFormat, WavSpec, WavWriter};
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let spec = WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut writer = WavWriter::create(tmp.path(), spec).unwrap();
        writer.write_sample(0.0f32).unwrap();
        writer.write_sample(0.5f32).unwrap();
        writer.write_sample(-0.5f32).unwrap();
        writer.finalize().unwrap();

        let samples = read_wav_samples(tmp.path()).unwrap();
        assert_eq!(samples.len(), 3);
        assert!((samples[0] - 0.0).abs() < 0.001);
        assert!((samples[1] - 0.5).abs() < 0.001);
        assert!((samples[2] + 0.5).abs() < 0.001);
    }

    #[test]
    fn test_read_wav_samples_empty() {
        use hound::{SampleFormat, WavSpec, WavWriter};
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let spec = WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let writer = WavWriter::create(tmp.path(), spec).unwrap();
        writer.finalize().unwrap();

        let samples = read_wav_samples(tmp.path()).unwrap();
        assert!(samples.is_empty());
    }

    #[test]
    fn test_read_wav_samples_nonexistent() {
        let result = read_wav_samples(std::path::Path::new("/nonexistent/file.wav"));
        assert!(result.is_err());
    }

    #[test]
    fn test_read_wav_samples_24bit_int() {
        use hound::{SampleFormat, WavSpec, WavWriter};
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let spec = WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 24,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(tmp.path(), spec).unwrap();
        writer.write_sample(0i32).unwrap();
        writer.write_sample(4194304i32).unwrap(); // ~half of 2^23
        writer.write_sample(-4194304i32).unwrap();
        writer.finalize().unwrap();

        let samples = read_wav_samples(tmp.path()).unwrap();
        assert_eq!(samples.len(), 3);
        assert!((samples[0] - 0.0).abs() < 0.01);
        assert!(samples[1] > 0.4 && samples[1] < 0.6);
        assert!(samples[2] < -0.4 && samples[2] > -0.6);
    }

    #[test]
    fn test_read_wav_samples_max_values() {
        use hound::{SampleFormat, WavSpec, WavWriter};
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let spec = WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(tmp.path(), spec).unwrap();
        writer.write_sample(i16::MAX).unwrap();
        writer.write_sample(i16::MIN).unwrap();
        writer.finalize().unwrap();

        let samples = read_wav_samples(tmp.path()).unwrap();
        assert_eq!(samples.len(), 2);
        // i16::MAX / 32768.0 ≈ 1.0
        assert!(samples[0] > 0.99);
        // i16::MIN / 32768.0 = -1.0
        assert!(samples[1] < -0.99);
    }

    #[test]
    fn test_read_wav_samples_multi_sample() {
        use hound::{SampleFormat, WavSpec, WavWriter};
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let spec = WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut writer = WavWriter::create(tmp.path(), spec).unwrap();
        for i in 0..100 {
            writer.write_sample(i as f32 / 100.0).unwrap();
        }
        writer.finalize().unwrap();

        let samples = read_wav_samples(tmp.path()).unwrap();
        assert_eq!(samples.len(), 100);
        assert!((samples[50] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_find_model_returns_none_for_empty_string() {
        assert!(find_model("").is_none());
    }

    #[test]
    fn test_model_exists_consistent_with_find_model() {
        let size = "nonexistent_test_model_xyz";
        assert_eq!(model_exists(size), find_model(size).is_some());
    }
}
