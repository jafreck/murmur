use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::config::Config;

/// Minimum audio duration in samples at 16 kHz.
/// Clips shorter than this tend to produce hallucinated output.
const MIN_AUDIO_SAMPLES: usize = 8_000; // 0.5 seconds

/// RMS energy threshold below which audio is considered silence.
const SILENCE_RMS_THRESHOLD: f32 = 0.01;

/// Phrases commonly hallucinated by Whisper on silence or near-silence.
const HALLUCINATED_PHRASES: &[&str] = &[
    "the following",
    "thank you",
    "thanks for watching",
    "thank you for watching",
    "thanks for listening",
    "thank you for listening",
    "like and subscribe",
    "please subscribe",
    "subscribe",
    "goodbye",
    "bye bye",
    "bye",
    "you",
];

/// Return true if audio samples are effectively silence.
fn is_silent(samples: &[f32]) -> bool {
    if samples.is_empty() {
        return true;
    }
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    rms < SILENCE_RMS_THRESHOLD
}

/// Return true if text matches a known Whisper hallucination pattern.
fn is_hallucination(text: &str) -> bool {
    let normalized = text
        .trim()
        .trim_matches(|c: char| c.is_ascii_punctuation())
        .to_lowercase();
    if normalized.is_empty() {
        return false;
    }
    HALLUCINATED_PHRASES.iter().any(|&h| normalized == h)
}

/// Optional context to improve transcription accuracy via Whisper's initial_prompt.
#[derive(Debug, Clone, Default)]
pub struct TranscriptionContext {
    /// Vocabulary terms to bias the model toward (domain-specific words, names, etc.)
    pub vocabulary: Vec<String>,
    /// Text surrounding the cursor — provides sentence-level context for continuation
    pub surrounding_text: Option<String>,
    /// Additional prompt prefix (e.g., language-specific instructions)
    pub prompt_prefix: Option<String>,
}

/// Maximum length for the initial_prompt string to avoid degrading performance.
const MAX_PROMPT_CHARS: usize = 500;

/// Build a Whisper initial_prompt from context information.
///
/// The prompt is structured as:
/// 1. Optional prompt prefix
/// 2. Vocabulary terms (as a comma-separated list)
/// 3. Surrounding text (the most recent text before the cursor)
///
/// Whisper uses this as "prior context" to bias its decoder. The surrounding
/// text is especially powerful — it gives the model sentence-level continuity.
///
/// The total prompt is capped at [`MAX_PROMPT_CHARS`] to avoid degrading performance.
pub fn build_initial_prompt(ctx: &TranscriptionContext) -> Option<String> {
    if ctx.vocabulary.is_empty() && ctx.surrounding_text.is_none() && ctx.prompt_prefix.is_none() {
        return None;
    }

    let mut parts: Vec<String> = Vec::new();

    // Prompt prefix first (e.g. language hints)
    if let Some(prefix) = &ctx.prompt_prefix {
        let trimmed = prefix.trim();
        if !trimmed.is_empty() {
            parts.push(trimmed.to_string());
        }
    }

    // Vocabulary terms — format as natural-looking text to bias decoder
    if !ctx.vocabulary.is_empty() {
        let vocab_str = ctx.vocabulary.join(", ");
        parts.push(vocab_str);
    }

    // Surrounding text last — this is the most important signal as it gives
    // the model direct sentence-level context for continuation
    if let Some(surrounding) = &ctx.surrounding_text {
        let trimmed = surrounding.trim();
        if !trimmed.is_empty() {
            parts.push(trimmed.to_string());
        }
    }

    let prompt = parts.join(". ");
    if prompt.is_empty() {
        return None;
    }

    // Truncate from the LEFT if too long — the end (most recent context) is most valuable
    if prompt.len() > MAX_PROMPT_CHARS {
        let start = prompt.len() - MAX_PROMPT_CHARS;
        // Try to break at a word boundary
        let adjusted_start = prompt[start..].find(' ').map(|i| start + i + 1).unwrap_or(start);
        Some(prompt[adjusted_start..].to_string())
    } else {
        Some(prompt)
    }
}

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

    /// Transcribe with context biasing for improved accuracy.
    pub fn transcribe_with_context(
        &self,
        audio_path: &Path,
        translate: bool,
        context: &TranscriptionContext,
    ) -> Result<String> {
        let samples = read_wav_samples(audio_path)?;
        self.run_inference_with_context(&samples, translate, Some(context))
    }

    /// Transcribe raw samples with context biasing.
    pub fn transcribe_samples_with_context(
        &self,
        samples: &[f32],
        translate: bool,
        context: &TranscriptionContext,
    ) -> Result<String> {
        self.run_inference_with_context(samples, translate, Some(context))
    }

    fn run_inference(&self, samples: &[f32], translate: bool) -> Result<String> {
        self.run_inference_with_context(samples, translate, None)
    }

    fn run_inference_with_context(
        &self,
        samples: &[f32],
        translate: bool,
        context: Option<&TranscriptionContext>,
    ) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        if samples.len() < MIN_AUDIO_SAMPLES {
            log::debug!(
                "Audio too short ({} samples, need {}), skipping",
                samples.len(),
                MIN_AUDIO_SAMPLES
            );
            return Ok(String::new());
        }

        if is_silent(samples) {
            log::debug!("Audio is silence, skipping transcription");
            return Ok(String::new());
        }

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(self.language_param());
        params.set_translate(translate);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_nst(true);

        // Apply context-based initial_prompt
        let prompt_string;
        if let Some(ctx) = context {
            if let Some(prompt) = build_initial_prompt(ctx) {
                log::debug!(
                    "Using initial_prompt ({} chars): {}...",
                    prompt.len(),
                    &prompt[..prompt.len().min(80)]
                );
                prompt_string = prompt;
                params.set_initial_prompt(&prompt_string);
            }
        }

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

        let text = text.trim().to_string();

        if is_hallucination(&text) {
            log::debug!("Filtered hallucinated text: '{text}'");
            return Ok(String::new());
        }

        Ok(text)
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

    // -- is_silent --

    #[test]
    fn test_is_silent_zeros() {
        assert!(is_silent(&vec![0.0f32; 16000]));
    }

    #[test]
    fn test_is_silent_low_noise() {
        assert!(is_silent(&vec![0.001f32; 16000]));
    }

    #[test]
    fn test_is_silent_loud() {
        assert!(!is_silent(&vec![0.5f32; 16000]));
    }

    #[test]
    fn test_is_silent_empty() {
        assert!(is_silent(&[]));
    }

    #[test]
    fn test_is_silent_threshold_boundary() {
        let below = SILENCE_RMS_THRESHOLD * 0.9;
        assert!(is_silent(&vec![below; 1000]));
        let above = SILENCE_RMS_THRESHOLD * 1.1;
        assert!(!is_silent(&vec![above; 1000]));
    }

    // -- is_hallucination --

    #[test]
    fn test_is_hallucination_known_phrases() {
        assert!(is_hallucination("The following:"));
        assert!(is_hallucination("Thank you."));
        assert!(is_hallucination("  thanks for watching  "));
        assert!(is_hallucination("Goodbye."));
        assert!(is_hallucination("you"));
        assert!(is_hallucination("Bye."));
    }

    #[test]
    fn test_is_hallucination_real_speech() {
        assert!(!is_hallucination("Hello, my name is Jacob"));
        assert!(!is_hallucination("The following steps are important"));
        assert!(!is_hallucination("Thank you for your help with the code"));
    }

    #[test]
    fn test_is_hallucination_empty() {
        assert!(!is_hallucination(""));
        assert!(!is_hallucination("   "));
    }

    // -- constants --

    #[test]
    fn test_constants() {
        const { assert!(MIN_AUDIO_SAMPLES > 0) };
        const { assert!(SILENCE_RMS_THRESHOLD > 0.0) };
        assert!(!HALLUCINATED_PHRASES.is_empty());
    }

    // -- TranscriptionContext --

    #[test]
    fn test_transcription_context_default() {
        let ctx = TranscriptionContext::default();
        assert!(ctx.vocabulary.is_empty());
        assert!(ctx.surrounding_text.is_none());
        assert!(ctx.prompt_prefix.is_none());
    }

    // -- build_initial_prompt --

    #[test]
    fn test_build_prompt_empty_context() {
        let ctx = TranscriptionContext::default();
        assert!(build_initial_prompt(&ctx).is_none());
    }

    #[test]
    fn test_build_prompt_vocabulary_only() {
        let ctx = TranscriptionContext {
            vocabulary: vec!["useState".to_string(), "async".to_string(), "impl".to_string()],
            ..Default::default()
        };
        let prompt = build_initial_prompt(&ctx).unwrap();
        assert!(prompt.contains("useState"));
        assert!(prompt.contains("async"));
        assert!(prompt.contains("impl"));
    }

    #[test]
    fn test_build_prompt_surrounding_text_only() {
        let ctx = TranscriptionContext {
            surrounding_text: Some("The function returns a".to_string()),
            ..Default::default()
        };
        let prompt = build_initial_prompt(&ctx).unwrap();
        assert!(prompt.contains("The function returns a"));
    }

    #[test]
    fn test_build_prompt_combined() {
        let ctx = TranscriptionContext {
            vocabulary: vec!["boolean".to_string()],
            surrounding_text: Some("The function returns a".to_string()),
            prompt_prefix: None,
        };
        let prompt = build_initial_prompt(&ctx).unwrap();
        assert!(prompt.contains("boolean"));
        assert!(prompt.contains("The function returns a"));
    }

    #[test]
    fn test_build_prompt_with_prefix() {
        let ctx = TranscriptionContext {
            vocabulary: vec![],
            surrounding_text: None,
            prompt_prefix: Some("Technical programming discussion.".to_string()),
        };
        let prompt = build_initial_prompt(&ctx).unwrap();
        assert!(prompt.contains("Technical programming discussion"));
    }

    #[test]
    fn test_build_prompt_truncation() {
        let ctx = TranscriptionContext {
            vocabulary: (0..100).map(|i| format!("word{i}")).collect(),
            surrounding_text: Some("important context at the end".to_string()),
            prompt_prefix: None,
        };
        let prompt = build_initial_prompt(&ctx).unwrap();
        // Should be truncated to MAX_PROMPT_CHARS
        assert!(prompt.len() <= MAX_PROMPT_CHARS);
        // The end (surrounding text) should be preserved since we truncate from the left
        assert!(prompt.contains("important context at the end"));
    }

    #[test]
    fn test_build_prompt_whitespace_handling() {
        let ctx = TranscriptionContext {
            vocabulary: vec![],
            surrounding_text: Some("   ".to_string()),
            prompt_prefix: Some("  ".to_string()),
        };
        assert!(build_initial_prompt(&ctx).is_none());
    }

    #[test]
    fn test_build_prompt_vocabulary_ordering() {
        let ctx = TranscriptionContext {
            vocabulary: vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()],
            ..Default::default()
        };
        let prompt = build_initial_prompt(&ctx).unwrap();
        let alpha_pos = prompt.find("alpha").unwrap();
        let beta_pos = prompt.find("beta").unwrap();
        let gamma_pos = prompt.find("gamma").unwrap();
        assert!(alpha_pos < beta_pos);
        assert!(beta_pos < gamma_pos);
    }
}
