use hound::{SampleFormat, WavSpec};

/// Target sample rate for Whisper input.
pub const TARGET_RATE: u32 = 16_000;

/// The WAV spec Whisper expects: 16kHz, 16-bit, mono PCM.
pub const WHISPER_WAV_SPEC: WavSpec = WavSpec {
    channels: 1,
    sample_rate: 16_000,
    bits_per_sample: 16,
    sample_format: SampleFormat::Int,
};

/// Mix multi-channel audio to mono by averaging channels.
pub fn mix_to_mono(data: &[f32], channels: u32) -> Vec<f32> {
    if channels <= 1 {
        return data.to_vec();
    }
    data.chunks(channels as usize)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Convert f32 sample to 16-bit PCM i16, clamping to [-1.0, 1.0].
/// Uses the standard 32768.0 scale factor for symmetric dynamic range.
pub fn f32_to_i16(sample: f32) -> i16 {
    let scaled = (sample.clamp(-1.0, 1.0) * 32768.0) as i32;
    scaled.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// Simple linear interpolation resampler.
/// Good enough for speech; use `rubato` crate for higher quality if needed.
pub(crate) fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    let mut output = Vec::new();
    resample_into(input, from_rate, to_rate, &mut output);
    output
}

/// Variant of [`resample`] that writes into a caller-supplied buffer, avoiding allocation.
pub(super) fn resample_into(input: &[f32], from_rate: u32, to_rate: u32, output: &mut Vec<f32>) {
    output.clear();
    if from_rate == to_rate {
        output.extend_from_slice(input);
        return;
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (input.len() as f64 / ratio) as usize;
    output.reserve(output_len);

    for i in 0..output_len {
        let src_idx = i as f64 * ratio;
        let idx = src_idx as usize;
        let frac = src_idx - idx as f64;

        let sample = if idx + 1 < input.len() {
            input[idx] as f64 * (1.0 - frac) + input[idx + 1] as f64 * frac
        } else if idx < input.len() {
            input[idx] as f64
        } else {
            0.0
        };

        output.push(sample as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_same_rate() {
        let input = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let output = resample(&input, 16000, 16000);
        assert_eq!(output, input);
    }

    #[test]
    fn test_resample_downsample() {
        let input: Vec<f32> = (0..48000).map(|i| i as f32 / 48000.0).collect();
        let output = resample(&input, 48000, 16000);
        assert!((output.len() as i64 - 16000).abs() <= 1);
    }

    #[test]
    fn test_resample_empty() {
        let output = resample(&[], 48000, 16000);
        assert!(output.is_empty());
    }

    #[test]
    fn test_resample_upsample() {
        let input: Vec<f32> = (0..8000).map(|i| i as f32 / 8000.0).collect();
        let output = resample(&input, 8000, 16000);
        // Should produce roughly 16000 samples from 8000
        assert!((output.len() as i64 - 16000).abs() <= 1);
    }

    #[test]
    fn test_resample_interpolates() {
        let input = vec![0.0, 1.0];
        let output = resample(&input, 2, 4);
        // Middle values should be interpolated between 0.0 and 1.0
        assert!(output.len() >= 3);
        assert!(output[1] > 0.0 && output[1] < 1.0);
    }

    #[test]
    fn test_resample_single_sample() {
        let input = vec![0.5];
        let output = resample(&input, 16000, 16000);
        assert_eq!(output, vec![0.5]);
    }

    #[test]
    fn test_resample_preserves_endpoints() {
        let input = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let output = resample(&input, 44100, 16000);
        // First sample should be close to 0.0
        assert!((output[0] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_resample_large_ratio() {
        let input: Vec<f32> = (0..96000).map(|i| (i as f32).sin()).collect();
        let output = resample(&input, 96000, 16000);
        assert!((output.len() as i64 - 16000).abs() <= 1);
    }

    #[test]
    fn test_resample_ratio_accuracy() {
        // 44.1kHz -> 16kHz: common real-world ratio
        let input: Vec<f32> = (0..44100).map(|i| (i as f32 / 44100.0).sin()).collect();
        let output = resample(&input, 44100, 16000);
        // Should produce approximately 16000 samples
        assert!((output.len() as i64 - 16000).abs() <= 1);
    }

    // -- mix_to_mono --

    #[test]
    fn test_mix_to_mono_single_channel() {
        let data = vec![0.1, 0.2, 0.3];
        let mono = mix_to_mono(&data, 1);
        assert_eq!(mono, data);
    }

    #[test]
    fn test_mix_to_mono_stereo() {
        let data = vec![0.0, 1.0, 0.5, 0.5, 1.0, 0.0];
        let mono = mix_to_mono(&data, 2);
        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 0.5).abs() < 0.001);
        assert!((mono[1] - 0.5).abs() < 0.001);
        assert!((mono[2] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_mix_to_mono_quad() {
        let data = vec![1.0, 0.0, 0.0, 0.0]; // 4 channels, 1 frame
        let mono = mix_to_mono(&data, 4);
        assert_eq!(mono.len(), 1);
        assert!((mono[0] - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_mix_to_mono_empty() {
        let mono = mix_to_mono(&[], 2);
        assert!(mono.is_empty());
    }

    #[test]
    fn test_mix_to_mono_six_channels() {
        // 6-channel surround: one frame
        let data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mono = mix_to_mono(&data, 6);
        assert_eq!(mono.len(), 1);
        assert!((mono[0] - 1.0 / 6.0).abs() < 0.001);
    }

    // -- f32_to_i16 --

    #[test]
    fn test_f32_to_i16_zero() {
        assert_eq!(f32_to_i16(0.0), 0);
    }

    #[test]
    fn test_f32_to_i16_max() {
        assert_eq!(f32_to_i16(1.0), 32767);
    }

    #[test]
    fn test_f32_to_i16_min() {
        assert_eq!(f32_to_i16(-1.0), -32768);
    }

    #[test]
    fn test_f32_to_i16_clamps_over() {
        assert_eq!(f32_to_i16(2.0), 32767);
    }

    #[test]
    fn test_f32_to_i16_clamps_under() {
        assert_eq!(f32_to_i16(-2.0), -32768);
    }

    #[test]
    fn test_f32_to_i16_half() {
        let v = f32_to_i16(0.5);
        assert!(v > 16000 && v < 17000);
    }

    #[test]
    fn test_f32_to_i16_negative_half() {
        let v = f32_to_i16(-0.5);
        assert!(v < -16000 && v > -17000);
    }

    // -- WHISPER_WAV_SPEC --

    #[test]
    fn test_whisper_wav_spec() {
        assert_eq!(WHISPER_WAV_SPEC.channels, 1);
        assert_eq!(WHISPER_WAV_SPEC.sample_rate, 16_000);
        assert_eq!(WHISPER_WAV_SPEC.bits_per_sample, 16);
        assert_eq!(WHISPER_WAV_SPEC.sample_format, SampleFormat::Int);
    }

    #[test]
    fn test_target_rate() {
        assert_eq!(TARGET_RATE, 16_000);
    }
}
