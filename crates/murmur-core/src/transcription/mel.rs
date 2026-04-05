//! Whisper-compatible mel spectrogram computation.
//!
//! Parameters match Whisper / Qwen3-ASR: 16 kHz, 128 mel bins, 25 ms window
//! (n_fft = 400), 10 ms hop (hop_length = 160), Hann window, 0–8 kHz.

use realfft::RealFftPlanner;

const SAMPLE_RATE: f32 = 16000.0;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 128;
const FMIN: f32 = 0.0;
const FMAX: f32 = 8000.0;

/// Compute a Whisper-compatible log-mel spectrogram.
///
/// Input: 16 kHz mono f32 samples.
/// Output: flattened `[n_mels, n_frames]` in row-major order, log-scaled.
pub fn whisper_mel(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
        return vec![0.0; N_MELS];
    }

    // Reflect-pad by n_fft/2 on each side (matches torch.stft center=True)
    let pad = N_FFT / 2;
    let padded_len = samples.len() + 2 * pad;
    let mut padded = vec![0.0f32; padded_len];
    // Reflect-pad left
    for i in 0..pad {
        padded[pad - 1 - i] = samples[(i + 1).min(samples.len() - 1)];
    }
    // Copy original
    padded[pad..pad + samples.len()].copy_from_slice(samples);
    // Reflect-pad right
    for i in 0..pad {
        let src = samples.len().saturating_sub(2 + i);
        padded[pad + samples.len() + i] = samples[src];
    }

    let n_frames = (padded_len - N_FFT) / HOP_LENGTH + 1;

    // Precompute Hann window (periodic, matching torch.hann_window)
    let window: Vec<f32> = (0..N_FFT)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / N_FFT as f32).cos()))
        .collect();

    // Precompute mel filterbank
    let filters = mel_filterbank(N_MELS, N_FFT, SAMPLE_RATE, FMIN, FMAX);

    // FFT setup
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let n_freqs = N_FFT / 2 + 1;

    let mut mel_spec = vec![0.0f32; N_MELS * n_frames];

    let mut fft_input = vec![0.0f32; N_FFT];
    let mut fft_output = fft.make_output_vec();

    for frame_idx in 0..n_frames {
        let start = frame_idx * HOP_LENGTH;

        // Fill frame with windowed samples from padded buffer
        for i in 0..N_FFT {
            fft_input[i] = padded[start + i] * window[i];
        }

        // FFT
        fft.process(&mut fft_input, &mut fft_output).ok();

        // Power spectrum
        let power: Vec<f32> = fft_output
            .iter()
            .map(|c| c.re * c.re + c.im * c.im)
            .collect();

        // Apply mel filterbank
        for mel_bin in 0..N_MELS {
            let filter_start = mel_bin * n_freqs;
            let mut energy = 0.0f32;
            for freq_bin in 0..n_freqs {
                energy += filters[filter_start + freq_bin] * power[freq_bin];
            }
            mel_spec[mel_bin * n_frames + frame_idx] = energy;
        }
    }

    // Log scale — matches WhisperFeatureExtractor
    let max_val = mel_spec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_offset = 1e-10_f32;

    for val in &mut mel_spec {
        *val = (*val).max(log_offset).log10();
    }

    // Clamp to max - 8.0
    let log_max = max_val.max(log_offset).log10();
    let clamp_min = log_max - 8.0;
    for val in &mut mel_spec {
        *val = (*val).max(clamp_min);
    }

    // Normalize: (log_spec + 4.0) / 4.0 — matches Whisper / Qwen3-ASR reference
    for val in &mut mel_spec {
        *val = (*val + 4.0) / 4.0;
    }

    // Drop last frame to match WhisperFeatureExtractor behavior
    if n_frames > 1 {
        let trimmed_frames = n_frames - 1;
        let mut trimmed = vec![0.0f32; N_MELS * trimmed_frames];
        for mel_bin in 0..N_MELS {
            let src_start = mel_bin * n_frames;
            let dst_start = mel_bin * trimmed_frames;
            trimmed[dst_start..dst_start + trimmed_frames]
                .copy_from_slice(&mel_spec[src_start..src_start + trimmed_frames]);
        }
        return trimmed;
    }

    mel_spec
}

/// Number of mel frames for a given sample count (with center padding, last frame dropped).
pub fn mel_frame_count(num_samples: usize) -> usize {
    if num_samples == 0 {
        return 1;
    }
    let padded_len = num_samples + N_FFT; // pad = N_FFT/2 on each side
    let raw_frames = (padded_len - N_FFT) / HOP_LENGTH + 1;
    // Drop last frame to match WhisperFeatureExtractor
    if raw_frames > 1 {
        raw_frames - 1
    } else {
        raw_frames
    }
}

/// Build a mel filterbank matrix: `[n_mels, n_freqs]` flattened.
fn mel_filterbank(n_mels: usize, n_fft: usize, sr: f32, fmin: f32, fmax: f32) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;

    // Mel scale conversion (Slaney / HTK)
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // n_mels + 2 boundary points
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert Hz points to FFT bin indices (fractional)
    let bin_points: Vec<f32> = hz_points.iter().map(|&hz| hz * n_fft as f32 / sr).collect();

    let mut filters = vec![0.0f32; n_mels * n_freqs];

    for m in 0..n_mels {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        for k in 0..n_freqs {
            let freq = k as f32;

            if freq >= left && freq <= center && center > left {
                filters[m * n_freqs + k] = (freq - left) / (center - left);
            } else if freq > center && freq <= right && right > center {
                filters[m * n_freqs + k] = (right - freq) / (right - center);
            }
        }

        // Slaney normalization: divide by mel band width
        let width = hz_points[m + 2] - hz_points[m];
        if width > 0.0 {
            let norm = 2.0 / width;
            for k in 0..n_freqs {
                filters[m * n_freqs + k] *= norm;
            }
        }
    }

    filters
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_output_shape() {
        let samples = vec![0.0f32; 16000]; // 1 second at 16kHz
        let mel = whisper_mel(&samples);
        let n_frames = mel_frame_count(16000);
        assert_eq!(mel.len(), N_MELS * n_frames);
    }

    #[test]
    fn mel_empty_input() {
        let mel = whisper_mel(&[]);
        assert_eq!(mel.len(), N_MELS);
    }

    #[test]
    fn mel_values_finite() {
        let samples: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.01).sin()).collect();
        let mel = whisper_mel(&samples);
        assert!(
            mel.iter().all(|v| v.is_finite()),
            "mel contains non-finite values"
        );
    }

    #[test]
    fn mel_filterbank_shape() {
        let filters = mel_filterbank(N_MELS, N_FFT, SAMPLE_RATE, FMIN, FMAX);
        assert_eq!(filters.len(), N_MELS * (N_FFT / 2 + 1));
    }

    #[test]
    fn hz_mel_roundtrip() {
        for hz in [0.0, 100.0, 1000.0, 4000.0, 8000.0] {
            let rt = mel_to_hz(hz_to_mel(hz));
            assert!((rt - hz).abs() < 0.1, "roundtrip failed for {hz}: got {rt}");
        }
    }
}
