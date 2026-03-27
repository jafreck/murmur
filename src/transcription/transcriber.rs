use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::config::Config;
use crate::transcription::vad;

/// Minimum audio duration in samples at 16 kHz.
/// Clips shorter than this tend to produce hallucinated output.
const MIN_AUDIO_SAMPLES: usize = 8_000; // 0.5 seconds

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

/// Maximum tokens to allocate when tokenizing a single term for ranking.
const MAX_TOKENS_PER_TERM: usize = 32;

/// Terms producing fewer BPE tokens than this are considered "known" to Whisper
/// and are deprioritized in the prompt. Terms at or above this threshold are
/// novel/domain-specific and get priority.
const NOVELTY_TOKEN_THRESHOLD: usize = 2;

/// A vocabulary term annotated with its BPE token count for ranking.
#[derive(Debug, Clone)]
pub struct RankedTerm {
    pub term: String,
    pub token_count: usize,
}

/// Rank vocabulary terms by how "novel" they are to Whisper's tokenizer.
///
/// Terms that fragment into many BPE subwords are ones Whisper doesn't know as
/// units — these benefit most from prompt biasing. Single-token terms (like
/// "function" or "return") are already in Whisper's vocabulary and waste prompt
/// space.
///
/// Returns terms sorted by token count descending (most novel first).
/// Terms that fail to tokenize are kept and treated as maximally novel.
pub fn rank_vocabulary(ctx: &WhisperContext, terms: &[String]) -> Vec<RankedTerm> {
    let mut ranked: Vec<RankedTerm> = terms
        .iter()
        .map(|term| {
            let token_count = match ctx.tokenize(term, MAX_TOKENS_PER_TERM) {
                Ok(tokens) => tokens.len(),
                Err(_) => {
                    // If tokenization fails, assume the term is very novel
                    log::debug!("Failed to tokenize term '{term}', treating as novel");
                    MAX_TOKENS_PER_TERM
                }
            };
            RankedTerm {
                term: term.clone(),
                token_count,
            }
        })
        .collect();

    // Sort by token count descending — most novel terms first
    ranked.sort_by(|a, b| b.token_count.cmp(&a.token_count));
    ranked
}

/// Filter ranked terms to only those Whisper is likely to get wrong.
///
/// Returns terms that tokenize into [`NOVELTY_TOKEN_THRESHOLD`] or more
/// subwords, discarding single-token terms that Whisper already knows.
pub fn filter_novel_terms(ranked: &[RankedTerm]) -> Vec<&RankedTerm> {
    ranked
        .iter()
        .filter(|rt| rt.token_count >= NOVELTY_TOKEN_THRESHOLD)
        .collect()
}

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
        // Ensure we don't slice into the middle of a multi-byte UTF-8 character
        let start = snap_to_char_boundary(&prompt, start);
        // For CJK scripts (Chinese, Japanese, Korean), words aren't space-separated
        // so any character boundary is a valid break point. For space-delimited
        // scripts, find the next word boundary to avoid partial words.
        let adjusted_start = if is_cjk_heavy(&prompt[start..]) {
            start
        } else if let Some(i) = prompt[start..].find(' ') {
            start + i + 1
        } else if let Some(i) = prompt[start..].find(", ") {
            start + i + 2
        } else {
            start
        };
        Some(prompt[adjusted_start..].to_string())
    } else {
        Some(prompt)
    }
}

/// Snap a byte offset forward to the nearest UTF-8 character boundary.
fn snap_to_char_boundary(s: &str, byte_offset: usize) -> usize {
    let mut pos = byte_offset;
    while pos < s.len() && !s.is_char_boundary(pos) {
        pos += 1;
    }
    pos
}

/// Heuristic: check if text is predominantly CJK (no spaces between words).
/// Looks at the first 100 characters — if most are CJK codepoints, treat
/// the text as non-space-delimited.
fn is_cjk_heavy(text: &str) -> bool {
    let sample: String = text.chars().take(100).collect();
    if sample.is_empty() {
        return false;
    }
    let cjk_count = sample.chars().filter(|c| is_cjk_char(*c)).count();
    // If more than 30% of chars are CJK, treat as CJK text
    cjk_count * 100 / sample.chars().count() > 30
}

fn is_cjk_char(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}'   // CJK Unified Ideographs
        | '\u{3400}'..='\u{4DBF}' // CJK Extension A
        | '\u{3040}'..='\u{309F}' // Hiragana
        | '\u{30A0}'..='\u{30FF}' // Katakana
        | '\u{AC00}'..='\u{D7AF}' // Hangul Syllables
        | '\u{F900}'..='\u{FAFF}' // CJK Compatibility Ideographs
    )
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

        if !vad::contains_speech(samples) {
            log::debug!("VAD: no speech detected, skipping transcription");
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

        // Apply context-based initial_prompt with smart vocabulary ranking.
        // `prompt_string` is declared here so it outlives the `params` borrow
        // in `set_initial_prompt()` below.
        let prompt_string;
        if let Some(ctx) = context {
            // Rank vocabulary terms by novelty if any are provided
            let ranked_ctx;
            let effective_ctx = if !ctx.vocabulary.is_empty() {
                let ranked = rank_vocabulary(&self.ctx, &ctx.vocabulary);
                let novel: Vec<String> = filter_novel_terms(&ranked)
                    .into_iter()
                    .map(|rt| rt.term.clone())
                    .collect();
                let kept = novel.len();
                let dropped = ctx.vocabulary.len() - kept;
                if dropped > 0 {
                    log::debug!(
                        "Vocabulary ranking: {} terms → {} novel (dropped {} known)",
                        ctx.vocabulary.len(),
                        kept,
                        dropped,
                    );
                }
                ranked_ctx = TranscriptionContext {
                    vocabulary: novel,
                    surrounding_text: ctx.surrounding_text.clone(),
                    prompt_prefix: ctx.prompt_prefix.clone(),
                };
                &ranked_ctx
            } else {
                ctx
            };

            if let Some(prompt) = build_initial_prompt(effective_ctx) {
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

    // -- VAD speech detection --

    #[test]
    fn test_vad_silence_has_no_speech() {
        assert!(!vad::contains_speech(&vec![0.0f32; 16000]));
    }

    #[test]
    fn test_vad_low_noise_has_no_speech() {
        assert!(!vad::contains_speech(&vec![0.001f32; 16000]));
    }

    #[test]
    fn test_vad_empty_has_no_speech() {
        assert!(!vad::contains_speech(&[]));
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
            vocabulary: vec![
                "useState".to_string(),
                "async".to_string(),
                "impl".to_string(),
            ],
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

    // -- RankedTerm / vocabulary ranking --

    #[test]
    fn test_ranked_term_struct() {
        let rt = RankedTerm {
            term: "useState".to_string(),
            token_count: 3,
        };
        assert_eq!(rt.term, "useState");
        assert_eq!(rt.token_count, 3);
    }

    #[test]
    fn test_filter_novel_terms_above_threshold() {
        let ranked = vec![
            RankedTerm {
                term: "kAXValueAttribute".to_string(),
                token_count: 5,
            },
            RankedTerm {
                term: "rustfmt".to_string(),
                token_count: 3,
            },
            RankedTerm {
                term: "useState".to_string(),
                token_count: 2,
            },
            RankedTerm {
                term: "function".to_string(),
                token_count: 1,
            },
        ];
        let novel = filter_novel_terms(&ranked);
        assert_eq!(novel.len(), 3);
        assert_eq!(novel[0].term, "kAXValueAttribute");
        assert_eq!(novel[1].term, "rustfmt");
        assert_eq!(novel[2].term, "useState");
    }

    #[test]
    fn test_filter_novel_terms_all_known() {
        let ranked = vec![
            RankedTerm {
                term: "hello".to_string(),
                token_count: 1,
            },
            RankedTerm {
                term: "world".to_string(),
                token_count: 1,
            },
        ];
        let novel = filter_novel_terms(&ranked);
        assert!(novel.is_empty());
    }

    #[test]
    fn test_filter_novel_terms_empty() {
        let novel = filter_novel_terms(&[]);
        assert!(novel.is_empty());
    }

    #[test]
    fn test_novelty_threshold_constant() {
        const { assert!(NOVELTY_TOKEN_THRESHOLD >= 2) };
    }

    #[test]
    fn test_max_prompt_chars_within_whisper_limits() {
        const { assert!(MAX_PROMPT_CHARS <= 1000) };
        const { assert!(MAX_PROMPT_CHARS >= 200) };
    }

    // -- CJK / UTF-8 truncation --

    #[test]
    fn test_snap_to_char_boundary_ascii() {
        let s = "hello world";
        assert_eq!(snap_to_char_boundary(s, 0), 0);
        assert_eq!(snap_to_char_boundary(s, 5), 5);
    }

    #[test]
    fn test_snap_to_char_boundary_multibyte() {
        let s = "héllo";
        // 'é' is 2 bytes in UTF-8 — offset 1 is mid-character
        assert!(s.is_char_boundary(0));
        let snapped = snap_to_char_boundary(s, 2);
        assert!(s.is_char_boundary(snapped));
    }

    #[test]
    fn test_snap_to_char_boundary_cjk() {
        let s = "你好世界";
        // Each CJK char is 3 bytes. Offset 1 is mid-character.
        let snapped = snap_to_char_boundary(s, 1);
        assert!(s.is_char_boundary(snapped));
        assert_eq!(snapped, 3); // snaps to start of second char
    }

    #[test]
    fn test_is_cjk_heavy_chinese() {
        assert!(is_cjk_heavy("你好世界这是一个测试"));
    }

    #[test]
    fn test_is_cjk_heavy_japanese() {
        assert!(is_cjk_heavy("こんにちは世界"));
    }

    #[test]
    fn test_is_cjk_heavy_english() {
        assert!(!is_cjk_heavy("hello world this is a test"));
    }

    #[test]
    fn test_is_cjk_heavy_mixed() {
        // Mostly English with a few CJK chars — should not be CJK-heavy
        assert!(!is_cjk_heavy("hello world 你好 this is a test string"));
    }

    #[test]
    fn test_is_cjk_heavy_empty() {
        assert!(!is_cjk_heavy(""));
    }

    #[test]
    fn test_truncation_preserves_cjk_chars() {
        // Build a prompt with CJK surrounding text that exceeds MAX_PROMPT_CHARS
        let cjk_text = "你好".repeat(300); // 600 CJK chars = 1800 bytes
        let ctx = TranscriptionContext {
            surrounding_text: Some(cjk_text),
            ..Default::default()
        };
        let prompt = build_initial_prompt(&ctx).unwrap();
        // Every character in the result should be valid
        assert!(prompt.len() <= MAX_PROMPT_CHARS);
        for c in prompt.chars() {
            assert!(c.len_utf8() > 0);
        }
    }
}
