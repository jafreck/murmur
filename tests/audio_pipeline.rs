//! Integration tests for the audio pipeline: WAV generation, format
//! validation, sample conversion, and the capture module's utility functions.

mod helpers;

use hound::{SampleFormat, WavReader};
use murmur::audio::capture::{f32_to_i16, mix_to_mono, TARGET_RATE, WHISPER_WAV_SPEC};

// ═══════════════════════════════════════════════════════════════════════
//  WHISPER_WAV_SPEC correctness
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn whisper_spec_is_16khz_mono_16bit() {
    assert_eq!(WHISPER_WAV_SPEC.channels, 1);
    assert_eq!(WHISPER_WAV_SPEC.sample_rate, 16_000);
    assert_eq!(WHISPER_WAV_SPEC.bits_per_sample, 16);
    assert_eq!(WHISPER_WAV_SPEC.sample_format, SampleFormat::Int);
    assert_eq!(TARGET_RATE, 16_000);
}

// ═══════════════════════════════════════════════════════════════════════
//  WAV write → read round-trip
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn wav_round_trip_preserves_sample_count() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("test.wav");

    let samples = helpers::sine_wave(440.0, 16_000, 1.0);
    helpers::write_wav(&path, &samples);

    let reader = WavReader::open(&path).unwrap();
    let spec = reader.spec();
    assert_eq!(spec.channels, 1);
    assert_eq!(spec.sample_rate, 16_000);
    assert_eq!(spec.bits_per_sample, 16);

    let read_samples: Vec<i16> = reader.into_samples::<i16>().map(|s| s.unwrap()).collect();
    assert_eq!(read_samples.len(), samples.len());
}

#[test]
fn wav_round_trip_preserves_signal_shape() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("shape.wav");

    let samples = helpers::sine_wave(440.0, 16_000, 0.1);
    helpers::write_wav(&path, &samples);

    let reader = WavReader::open(&path).unwrap();
    let read_samples: Vec<f32> = reader
        .into_samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32767.0)
        .collect();

    // The reconstructed signal should be close to the original
    for (orig, read) in samples.iter().zip(read_samples.iter()) {
        let diff = (orig - read).abs();
        assert!(
            diff < 0.001,
            "sample mismatch: orig={orig}, read={read}, diff={diff}"
        );
    }
}

#[test]
fn silence_wav_has_near_zero_samples() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("silence.wav");

    helpers::write_silence_wav(&path, 0.5);

    let reader = WavReader::open(&path).unwrap();
    let samples: Vec<i16> = reader.into_samples::<i16>().map(|s| s.unwrap()).collect();

    assert_eq!(samples.len(), 8000); // 0.5s * 16000
    assert!(samples.iter().all(|&s| s == 0));
}

// ═══════════════════════════════════════════════════════════════════════
//  WAV file validation via hound
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn written_wav_has_correct_spec() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("spec.wav");

    helpers::write_tone_wav(&path, 2.0);

    let reader = WavReader::open(&path).unwrap();
    let spec = reader.spec();
    assert_eq!(spec.channels, 1);
    assert_eq!(spec.sample_rate, 16_000);
    assert_eq!(spec.bits_per_sample, 16);
    assert_eq!(spec.sample_format, SampleFormat::Int);
}

#[test]
fn wav_duration_is_correct() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("duration.wav");

    let duration_secs = 3.0;
    helpers::write_tone_wav(&path, duration_secs);

    let reader = WavReader::open(&path).unwrap();
    let num_samples = reader.len();
    let expected = (16_000.0 * duration_secs) as u32;
    assert_eq!(num_samples, expected);
}

// ═══════════════════════════════════════════════════════════════════════
//  f32_to_i16 conversion
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn f32_to_i16_edge_cases() {
    assert_eq!(f32_to_i16(0.0), 0);
    assert_eq!(f32_to_i16(1.0), 32767);
    assert_eq!(f32_to_i16(-1.0), -32768);
}

#[test]
fn f32_to_i16_clamps_overflow() {
    // Values beyond [-1.0, 1.0] should be clamped
    assert_eq!(f32_to_i16(2.0), 32767);
    assert_eq!(f32_to_i16(-2.0), -32768);
    assert_eq!(f32_to_i16(100.0), 32767);
    assert_eq!(f32_to_i16(-100.0), -32768);
}

#[test]
fn f32_to_i16_preserves_sign() {
    assert!(f32_to_i16(0.5) > 0);
    assert!(f32_to_i16(-0.5) < 0);
}

#[test]
fn f32_to_i16_half_range() {
    let half = f32_to_i16(0.5);
    // Should be approximately 16383
    assert!((half - 16383).abs() <= 1);
}

// ═══════════════════════════════════════════════════════════════════════
//  mix_to_mono
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn mix_to_mono_passthrough_for_mono() {
    let mono = vec![0.1, 0.2, 0.3, 0.4];
    let result = mix_to_mono(&mono, 1);
    assert_eq!(result, mono);
}

#[test]
fn mix_to_mono_averages_stereo() {
    // Stereo: L=1.0, R=0.0 -> mono=0.5
    let stereo = vec![1.0, 0.0, 0.0, 1.0];
    let result = mix_to_mono(&stereo, 2);
    assert_eq!(result.len(), 2);
    assert!((result[0] - 0.5).abs() < 1e-6);
    assert!((result[1] - 0.5).abs() < 1e-6);
}

#[test]
fn mix_to_mono_four_channels() {
    let quad = vec![1.0, 0.0, 0.0, 0.0]; // 4 channels, one frame
    let result = mix_to_mono(&quad, 4);
    assert_eq!(result.len(), 1);
    assert!((result[0] - 0.25).abs() < 1e-6);
}

#[test]
fn mix_to_mono_empty_input() {
    let empty: Vec<f32> = vec![];
    let result = mix_to_mono(&empty, 2);
    assert!(result.is_empty());
}

#[test]
fn mix_to_mono_preserves_silence() {
    let stereo_silence = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let result = mix_to_mono(&stereo_silence, 2);
    assert!(result.iter().all(|&s| s == 0.0));
}

// ═══════════════════════════════════════════════════════════════════════
//  Synthetic WAV helpers produce valid files
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn sine_wave_generates_correct_length() {
    let samples = helpers::sine_wave(440.0, 16_000, 2.0);
    assert_eq!(samples.len(), 32_000);
}

#[test]
fn sine_wave_stays_in_range() {
    let samples = helpers::sine_wave(440.0, 16_000, 1.0);
    for &s in &samples {
        assert!((-1.0..=1.0).contains(&s), "sample out of range: {s}");
    }
}

#[test]
fn silence_generates_all_zeros() {
    let samples = helpers::silence(16_000, 1.0);
    assert_eq!(samples.len(), 16_000);
    assert!(samples.iter().all(|&s| s == 0.0));
}

// ═══════════════════════════════════════════════════════════════════════
//  Multi-channel WAV → mono conversion pipeline
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stereo_to_mono_wav_pipeline() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("mono.wav");

    // Simulate stereo audio
    let left = helpers::sine_wave(440.0, 16_000, 0.5);
    let right = helpers::sine_wave(880.0, 16_000, 0.5);

    // Interleave L/R
    let stereo: Vec<f32> = left
        .iter()
        .zip(right.iter())
        .flat_map(|(&l, &r)| [l, r])
        .collect();

    // Mix to mono
    let mono = mix_to_mono(&stereo, 2);
    assert_eq!(mono.len(), left.len());

    // Write as mono WAV
    helpers::write_wav(&path, &mono);

    // Verify
    let reader = WavReader::open(&path).unwrap();
    assert_eq!(reader.spec().channels, 1);
    assert_eq!(reader.len() as usize, mono.len());
}

// ═══════════════════════════════════════════════════════════════════════
//  read_wav_samples (if model file not needed, just wav reading)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn read_wav_samples_from_written_file() {
    use murmur::transcription::transcriber::read_wav_samples;

    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("readable.wav");

    let original = helpers::sine_wave(440.0, 16_000, 0.5);
    helpers::write_wav(&path, &original);

    let read = read_wav_samples(&path).unwrap();
    assert_eq!(read.len(), original.len());

    // Samples should be close (i16 quantization introduces small error)
    for (orig, read) in original.iter().zip(read.iter()) {
        assert!(
            (orig - read).abs() < 0.001,
            "mismatch: orig={orig}, read={read}"
        );
    }
}

#[test]
fn read_wav_samples_nonexistent_file_errors() {
    use murmur::transcription::transcriber::read_wav_samples;

    let result = read_wav_samples(std::path::Path::new("/nonexistent/audio.wav"));
    assert!(result.is_err());
}
