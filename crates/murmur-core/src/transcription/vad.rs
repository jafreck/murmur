//! ML-based Voice Activity Detection using Silero VAD.
//!
//! Replaces the simple RMS energy threshold with a neural-network classifier
//! that can distinguish speech from background noise, breathing, keyboard
//! clicks, and other non-speech audio. This prevents Whisper from
//! hallucinating text on silence or ambient noise.

use std::sync::{Mutex, OnceLock};

use voice_activity_detector::VoiceActivityDetector;

/// Silero VAD processes 16 kHz audio in fixed 512-sample (32 ms) windows.
const CHUNK_SIZE: usize = 512;

/// Speech probability threshold — chunks scoring at or above this value
/// are considered speech. Silero's default recommendation is 0.5.
const SPEECH_THRESHOLD: f32 = 0.5;

/// Minimum fraction of chunks that must contain speech for the audio
/// to be considered as containing meaningful speech. This avoids
/// sending a clip where only a tiny blip triggered the detector.
const MIN_SPEECH_RATIO: f32 = 0.05;

/// RMS energy floor — audio below this level is pure silence / digital
/// zero and can be rejected without running the neural network.
const SILENCE_RMS_FLOOR: f32 = 0.005;

/// Singleton VAD instance reused across calls to avoid repeated model setup.
static VAD: OnceLock<Mutex<VoiceActivityDetector>> = OnceLock::new();

fn get_vad() -> Result<&'static Mutex<VoiceActivityDetector>, voice_activity_detector::Error> {
    if let Some(vad) = VAD.get() {
        return Ok(vad);
    }
    let vad = VoiceActivityDetector::builder()
        .sample_rate(16000)
        .chunk_size(CHUNK_SIZE)
        .build()?;
    // If another thread raced us, `get_or_init` returns the winner; ours is dropped.
    Ok(VAD.get_or_init(|| Mutex::new(vad)))
}

/// Check whether `samples` (16 kHz mono f32) contain speech using Silero VAD.
///
/// Returns `true` if meaningful speech is detected, `false` otherwise.
/// Conservatively returns `false` if the VAD model fails, skipping
/// transcription rather than hallucinating on non-speech audio.
pub fn contains_speech(samples: &[f32]) -> bool {
    if samples.is_empty() {
        return false;
    }

    let rms = audio_rms(samples);

    // Fast path: reject digital silence without loading the model.
    if rms < SILENCE_RMS_FLOOR {
        log::debug!("VAD: audio below noise floor (RMS={rms:.6}), skipping");
        return false;
    }

    log::debug!("VAD: audio RMS={rms:.4}, running speech detection");

    match detect_speech(samples) {
        Ok(has_speech) => {
            if !has_speech {
                log::info!("VAD: no speech detected (RMS={rms:.4})");
            }
            has_speech
        }
        Err(e) => {
            log::warn!("VAD inference failed, conservatively skipping transcription: {e}");
            false
        }
    }
}

/// Run Silero VAD inference over the audio.
fn detect_speech(samples: &[f32]) -> Result<bool, voice_activity_detector::Error> {
    let vad_mutex = get_vad()?;
    let mut vad = vad_mutex.lock().unwrap_or_else(|e| e.into_inner());
    vad.reset();

    let total_chunks = samples.len().div_ceil(CHUNK_SIZE);
    if total_chunks == 0 {
        return Ok(false);
    }

    // Pre-compute the threshold so we can exit early once enough
    // speech chunks are found, avoiding redundant ONNX inference.
    let required_speech_chunks = (MIN_SPEECH_RATIO * total_chunks as f32).ceil() as usize;
    let mut speech_chunks: usize = 0;

    for chunk_start in (0..samples.len()).step_by(CHUNK_SIZE) {
        let chunk_end = (chunk_start + CHUNK_SIZE).min(samples.len());
        let chunk = &samples[chunk_start..chunk_end];

        let probability = vad.predict(chunk.iter().copied());
        if probability >= SPEECH_THRESHOLD {
            speech_chunks += 1;
            if speech_chunks >= required_speech_chunks {
                log::debug!(
                    "VAD: early exit — {speech_chunks}/{total_chunks} chunks contain speech"
                );
                return Ok(true);
            }
        }
    }

    let speech_ratio = speech_chunks as f32 / total_chunks as f32;
    log::debug!(
        "VAD: {speech_chunks}/{total_chunks} chunks contain speech (ratio={speech_ratio:.2})"
    );

    Ok(speech_ratio >= MIN_SPEECH_RATIO)
}

/// Quick RMS energy check to reject near-zero audio without neural inference.
#[cfg_attr(not(test), allow(dead_code))]
fn is_below_noise_floor(samples: &[f32]) -> bool {
    if samples.is_empty() {
        return true;
    }
    audio_rms(samples) < SILENCE_RMS_FLOOR
}

/// Compute RMS (root-mean-square) energy of audio samples.
fn audio_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_audio_has_no_speech() {
        assert!(!contains_speech(&[]));
    }

    #[test]
    fn test_silence_has_no_speech() {
        let samples = vec![0.0f32; 16_000]; // 1 second of silence
        assert!(!contains_speech(&samples));
    }

    #[test]
    fn test_low_noise_has_no_speech() {
        let samples = vec![0.001f32; 16_000];
        assert!(!contains_speech(&samples));
    }

    #[test]
    fn test_noise_floor_check() {
        assert!(is_below_noise_floor(&[0.0; 1000]));
        assert!(is_below_noise_floor(&[0.001; 1000]));
        assert!(!is_below_noise_floor(&[0.1; 1000]));
        assert!(is_below_noise_floor(&[]));
    }

    #[test]
    fn test_vad_model_loads() {
        // Verify the singleton VAD initialises without error
        let result = get_vad();
        assert!(result.is_ok(), "VAD model failed to load: {result:?}");
    }

    #[test]
    fn test_detect_speech_on_silence() {
        let samples = vec![0.01f32; 16_000];
        // Low constant signal should not be classified as speech
        let result = detect_speech(&samples);
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }
}
