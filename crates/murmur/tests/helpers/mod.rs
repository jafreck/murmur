//! Shared test utilities for integration and e2e tests.
#![allow(dead_code)]

use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use hound::{SampleFormat, WavSpec, WavWriter};
use murmur_core::config::Config;

/// The WAV spec Whisper expects.
pub const WHISPER_SPEC: WavSpec = WavSpec {
    channels: 1,
    sample_rate: 16_000,
    bits_per_sample: 16,
    sample_format: SampleFormat::Int,
};

// ── Synthetic WAV generation ────────────────────────────────────────────

/// Generate a sine wave of `freq_hz` at the given sample rate for `duration_secs`.
/// Returns f32 samples in [-1.0, 1.0].
pub fn sine_wave(freq_hz: f32, sample_rate: u32, duration_secs: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * freq_hz * t).sin()
        })
        .collect()
}

/// Generate silence (all zeros) for the given duration.
pub fn silence(sample_rate: u32, duration_secs: f32) -> Vec<f32> {
    vec![0.0; (sample_rate as f32 * duration_secs) as usize]
}

/// Write f32 samples to a WAV file at the given path using WHISPER_SPEC.
pub fn write_wav(path: &Path, samples: &[f32]) {
    let mut writer = WavWriter::create(path, WHISPER_SPEC).expect("create WAV");
    for &s in samples {
        let i16_sample = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer.write_sample(i16_sample).expect("write sample");
    }
    writer.finalize().expect("finalize WAV");
}

/// Write a valid WAV file with a 440Hz tone for the given duration.
pub fn write_tone_wav(path: &Path, duration_secs: f32) {
    let samples = sine_wave(440.0, 16_000, duration_secs);
    write_wav(path, &samples);
}

/// Write a WAV file containing silence.
pub fn write_silence_wav(path: &Path, duration_secs: f32) {
    let samples = silence(16_000, duration_secs);
    write_wav(path, &samples);
}

// ── Config fixtures ─────────────────────────────────────────────────────

/// Create a default Config saved to a temp directory. Returns (config, path).
pub fn config_in_dir(dir: &Path) -> (Config, PathBuf) {
    let path = dir.join("config.json");
    let config = Config::default();
    config.save_to(&path).expect("save config");
    (config, path)
}

/// Create a custom Config and save it. Returns (config, path).
pub fn custom_config_in_dir(dir: &Path, mutator: impl FnOnce(&mut Config)) -> (Config, PathBuf) {
    let path = dir.join("config.json");
    let mut config = Config::default();
    mutator(&mut config);
    config.save_to(&path).expect("save config");
    (config, path)
}

// ── GGML magic helpers ──────────────────────────────────────────────────

/// Write a fake GGML model file with valid magic bytes.
pub fn write_fake_ggml(path: &Path) {
    let mut f = std::fs::File::create(path).expect("create fake model");
    f.write_all(&0x67676d6cu32.to_le_bytes())
        .expect("write magic");
    f.write_all(&[0u8; 256]).expect("write padding");
}

/// Write an invalid model file (HTML error page, common download failure).
pub fn write_invalid_model(path: &Path) {
    std::fs::write(path, b"<!DOCTYPE html><html>404 Not Found</html>").expect("write invalid");
}
