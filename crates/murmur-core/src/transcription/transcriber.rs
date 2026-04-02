use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use crate::transcription::model_discovery::{inference_thread_count, read_wav_samples};
use crate::transcription::prompt::{
    build_initial_prompt, filter_novel_terms, rank_vocabulary, TranscriptionContext,
};
use crate::transcription::vad;

/// Minimum audio duration in samples at 16 kHz.
/// Clips shorter than this tend to produce hallucinated output.
const MIN_AUDIO_SAMPLES: usize = 4_000; // 0.25 seconds

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

pub struct Transcriber {
    ctx: WhisperContext,
    language: String,
    model_path: PathBuf,
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
            model_path: model_path.to_path_buf(),
        })
    }

    /// Path to the loaded model file.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Language setting for this transcriber.
    pub fn language(&self) -> &str {
        &self.language
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

    // ── Streaming-optimised API ────────────────────────────────────────

    /// Create a reusable `WhisperState` for streaming.
    ///
    /// Creating a state involves allocating KV caches and other internal
    /// buffers. Reusing one across streaming iterations avoids that
    /// per-call overhead.
    pub fn create_streaming_state(&self) -> Result<WhisperState> {
        self.ctx
            .create_state()
            .map_err(|e| anyhow::anyhow!("Failed to create whisper state: {e}"))
    }

    /// Run a single streaming transcription pass.
    ///
    /// Unlike [`transcribe_samples`], this method:
    /// - reuses a caller-provided [`WhisperState`] (no per-call allocation),
    /// - skips VAD and min-length checks (the streaming loop handles those),
    /// - enables `no_context` (each pass re-transcribes the full window).
    ///
    /// If `abort_flag` is set to `true` mid-inference, whisper will abort
    /// early and an empty string is returned.
    pub fn streaming_transcribe(
        &self,
        state: &mut WhisperState,
        samples: &[f32],
        translate: bool,
        abort_flag: &Arc<AtomicBool>,
    ) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(inference_thread_count());
        params.set_language(self.language_param());
        params.set_translate(translate);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_nst(true);
        // NOTE: single_segment is intentionally NOT set here.
        // whisper-rs 0.16 has a bug where set_single_segment(true)
        // corrupts the WhisperState on subsequent reuse, causing
        // "failed to encode" error -6.
        params.set_no_context(true);

        let flag = Arc::clone(abort_flag);
        params.set_abort_callback_safe(move || flag.load(Ordering::Relaxed));

        state
            .full(params, samples)
            .map_err(|e| anyhow::anyhow!("Transcription failed: {e}"))?;

        if abort_flag.load(Ordering::Relaxed) {
            return Ok(String::new());
        }

        let text = self.extract_text(state);

        if is_hallucination(&text) {
            log::debug!("Filtered hallucinated text: '{text}'");
            return Ok(String::new());
        }

        Ok(text)
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
        params.set_n_threads(inference_thread_count());
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

        let text = self.extract_text(&state);

        if is_hallucination(&text) {
            log::debug!("Filtered hallucinated text: '{text}'");
            return Ok(String::new());
        }

        Ok(text)
    }

    /// Collect segment text from a completed whisper state.
    fn extract_text(&self, state: &WhisperState) -> String {
        let num_segments = state.full_n_segments();
        if num_segments < 0 {
            return String::new();
        }

        let mut text = String::new();
        for i in 0..num_segments {
            if let Some(segment) = state.get_segment(i) {
                if let Ok(segment_text) = segment.to_str_lossy() {
                    text.push_str(&segment_text);
                }
            }
        }
        text.trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        #[allow(clippy::const_is_empty)]
        {
            assert!(!HALLUCINATED_PHRASES.is_empty());
        }
    }
}
