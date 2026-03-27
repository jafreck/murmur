use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::{SampleFormat, WavSpec, WavWriter};
use std::collections::VecDeque;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Duration of the pre-roll buffer in milliseconds.
/// Captures audio *before* the user presses record so the first word isn't clipped.
const PRE_ROLL_MS: u32 = 200;

/// Number of 16 kHz samples in the pre-roll buffer.
const PRE_ROLL_SAMPLES: usize = (TARGET_RATE * PRE_ROLL_MS / 1000) as usize;

/// Shared state between the audio callback thread and the main thread.
struct SharedCaptureState {
    /// True while actively recording (as opposed to standby pre-roll capture).
    recording: AtomicBool,
    /// WAV file writer, set only during file-backed recordings.
    writer: Mutex<Option<WavWriter<BufWriter<File>>>>,
    /// In-memory buffer of 16 kHz mono f32 samples captured since `start()`.
    samples: Arc<Mutex<Vec<f32>>>,
    /// Ring buffer capturing recent audio while in standby mode.
    /// When recording starts, these samples are drained into `samples`
    /// so the beginning of speech is preserved.
    pre_roll: Mutex<VecDeque<f32>>,
    /// Count of samples dropped due to lock contention.
    dropped_samples: AtomicU64,
}

impl SharedCaptureState {
    fn new() -> Self {
        Self {
            recording: AtomicBool::new(false),
            writer: Mutex::new(None),
            samples: Arc::new(Mutex::new(Vec::new())),
            pre_roll: Mutex::new(VecDeque::with_capacity(PRE_ROLL_SAMPLES + 512)),
            dropped_samples: AtomicU64::new(0),
        }
    }
}

pub struct AudioRecorder {
    stream: Option<cpal::Stream>,
    shared: Arc<SharedCaptureState>,
    current_path: Option<PathBuf>,
}

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

/// The WAV spec Whisper expects: 16kHz, 16-bit, mono PCM.
pub const WHISPER_WAV_SPEC: WavSpec = WavSpec {
    channels: 1,
    sample_rate: 16_000,
    bits_per_sample: 16,
    sample_format: SampleFormat::Int,
};

/// Target sample rate for Whisper input.
pub const TARGET_RATE: u32 = 16_000;

impl Default for AudioRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioRecorder {
    pub fn new() -> Self {
        Self {
            stream: None,
            shared: Arc::new(SharedCaptureState::new()),
            current_path: None,
        }
    }

    /// Open the microphone and start capturing into the pre-roll buffer.
    ///
    /// Call once at app startup so recording starts instantly on hotkey press.
    /// If the stream is already running this is a no-op.
    pub fn warm(&mut self) -> Result<()> {
        if self.stream.is_some() {
            return Ok(());
        }

        let host = cpal::default_host();
        let device = host.default_input_device().context("No microphone found")?;

        let supported_config = device
            .default_input_config()
            .context("Failed to get default input config")?;

        let native_rate = supported_config.sample_rate();
        let native_channels = supported_config.channels() as u32;

        let shared = Arc::clone(&self.shared);

        let stream = device
            .build_input_stream(
                &supported_config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mono = mix_to_mono(data, native_channels);
                    let resampled = resample(&mono, native_rate, TARGET_RATE);

                    if shared.recording.load(Ordering::Acquire) {
                        // Recording mode: write to sample buffer and optional WAV
                        if let Ok(mut buf) = shared.samples.try_lock() {
                            buf.extend_from_slice(&resampled);
                        } else {
                            shared
                                .dropped_samples
                                .fetch_add(resampled.len() as u64, Ordering::Relaxed);
                        }

                        if let Ok(mut guard) = shared.writer.try_lock() {
                            if let Some(ref mut w) = *guard {
                                for &sample in &resampled {
                                    let _ = w.write_sample(f32_to_i16(sample));
                                }
                            }
                        }
                    } else {
                        // Standby mode: fill pre-roll ring buffer
                        if let Ok(mut ring) = shared.pre_roll.try_lock() {
                            for &s in &resampled {
                                if ring.len() >= PRE_ROLL_SAMPLES {
                                    ring.pop_front();
                                }
                                ring.push_back(s);
                            }
                        }
                    }
                },
                |err| {
                    log::error!("Audio stream error: {err}");
                },
                None,
            )
            .context("Failed to build input stream")?;

        stream.play().context("Failed to start audio stream")?;
        self.stream = Some(stream);
        log::info!("Microphone warmed up (pre-roll: {PRE_ROLL_MS}ms)");

        Ok(())
    }

    /// Ensure the stream is warm, warming it up if needed.
    fn ensure_warm(&mut self) -> Result<()> {
        if self.stream.is_none() {
            self.warm()?;
        }
        Ok(())
    }

    pub fn start_in_memory(&mut self) -> Result<()> {
        self.ensure_warm()?;

        self.shared.dropped_samples.store(0, Ordering::Relaxed);

        // Hold pre_roll lock across the transition to prevent audio samples
        // from going into pre_roll between the drain and recording=true.
        if let Ok(mut ring) = self.shared.pre_roll.lock() {
            if let Ok(mut samples) = self.shared.samples.lock() {
                samples.clear();
                samples.extend(ring.drain(..));
            }
            // Set recording while still holding pre_roll lock so the audio
            // callback immediately writes to samples instead of pre_roll.
            self.shared.recording.store(true, Ordering::Release);
        }

        self.current_path = None;

        Ok(())
    }

    /// Stop recording and return the captured samples.
    /// For in-memory recordings (no WAV file).
    pub fn stop_samples(&mut self) -> Option<Vec<f32>> {
        self.shared.recording.store(false, Ordering::Release);

        let dropped = self.shared.dropped_samples.load(Ordering::Relaxed);
        if dropped > 0 {
            log::warn!("Dropped {dropped} audio samples due to lock contention during recording");
        }

        let samples = self.shared.samples.lock().ok().map(|b| b.clone());
        self.current_path.take();
        samples.filter(|s| !s.is_empty())
    }

    pub fn start(&mut self, output_path: &Path) -> Result<()> {
        self.ensure_warm()?;

        let writer = WavWriter::create(output_path, WHISPER_WAV_SPEC)
            .context("Failed to create WAV file")?;

        self.current_path = Some(output_path.to_path_buf());
        self.shared.dropped_samples.store(0, Ordering::Relaxed);

        // Install the WAV writer before draining pre-roll
        if let Ok(mut guard) = self.shared.writer.lock() {
            *guard = Some(writer);
        }

        // Hold pre_roll lock across the drain and recording flag transition
        // to prevent audio samples from going into pre_roll during the gap.
        if let Ok(mut ring) = self.shared.pre_roll.lock() {
            if let Ok(mut samples) = self.shared.samples.lock() {
                samples.clear();
                let pre_roll_data: Vec<f32> = ring.drain(..).collect();
                samples.extend_from_slice(&pre_roll_data);

                if let Ok(mut guard) = self.shared.writer.lock() {
                    if let Some(ref mut w) = *guard {
                        for &sample in &pre_roll_data {
                            let _ = w.write_sample(f32_to_i16(sample));
                        }
                    }
                }
            }
            // Set recording while still holding pre_roll lock so the audio
            // callback immediately writes to samples instead of pre_roll.
            self.shared.recording.store(true, Ordering::Release);
        }

        Ok(())
    }

    /// Return a copy of samples captured since `start()`, beginning at `offset`.
    /// Samples are 16 kHz mono f32 in the range \[−1, 1\].
    #[allow(dead_code)]
    pub fn snapshot(&self, offset: usize) -> Vec<f32> {
        if let Ok(buf) = self.shared.samples.lock() {
            if offset < buf.len() {
                buf[offset..].to_vec()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    }

    /// Number of 16 kHz samples captured since `start()`.
    #[allow(dead_code)]
    pub fn sample_count(&self) -> usize {
        self.shared.samples.lock().map(|b| b.len()).unwrap_or(0)
    }

    /// A shared handle to the sample buffer for streaming access.
    pub fn sample_buffer(&self) -> Arc<Mutex<Vec<f32>>> {
        Arc::clone(&self.shared.samples)
    }

    pub fn stop(&mut self) -> Option<PathBuf> {
        // Transition back to standby (stream stays alive for next recording)
        self.shared.recording.store(false, Ordering::Release);

        let dropped = self.shared.dropped_samples.load(Ordering::Relaxed);
        if dropped > 0 {
            log::warn!("Dropped {dropped} audio samples due to lock contention during recording");
        }

        // Finalize the WAV file
        if let Ok(mut guard) = self.shared.writer.lock() {
            if let Some(writer) = guard.take() {
                let _ = writer.finalize();
            }
        }

        self.current_path.take()
    }
}

/// Simple linear interpolation resampler.
/// Good enough for speech; use `rubato` crate for higher quality if needed.
fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return input.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (input.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

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

    output
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
    fn test_new_recorder() {
        let recorder = AudioRecorder::new();
        assert!(recorder.stream.is_none());
        assert!(recorder.current_path.is_none());
    }

    #[test]
    fn test_stop_without_start() {
        let mut recorder = AudioRecorder::new();
        let path = recorder.stop();
        assert!(path.is_none());
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

    // -- snapshot / sample_count --

    #[test]
    fn test_snapshot_empty_recorder() {
        let recorder = AudioRecorder::new();
        assert!(recorder.snapshot(0).is_empty());
        assert_eq!(recorder.sample_count(), 0);
    }

    #[test]
    fn test_sample_buffer_returns_clone() {
        let recorder = AudioRecorder::new();
        let buf = recorder.sample_buffer();
        assert_eq!(buf.lock().unwrap().len(), 0);
    }

    #[test]
    fn test_stop_samples_without_start_returns_none() {
        let mut recorder = AudioRecorder::new();
        let samples = recorder.stop_samples();
        assert!(samples.is_none());
    }

    #[test]
    fn test_default_trait() {
        let recorder = AudioRecorder::default();
        assert!(recorder.stream.is_none());
        assert!(recorder.current_path.is_none());
        assert_eq!(recorder.sample_count(), 0);
    }

    #[test]
    fn test_snapshot_with_offset_beyond_len() {
        let recorder = AudioRecorder::new();
        let snap = recorder.snapshot(100);
        assert!(snap.is_empty());
    }

    #[test]
    fn test_snapshot_with_manual_samples() {
        let recorder = AudioRecorder::new();
        // Push samples via the public sample_buffer() handle
        {
            let buf = recorder.sample_buffer();
            buf.lock()
                .unwrap()
                .extend_from_slice(&[0.1, 0.2, 0.3, 0.4, 0.5]);
        }
        let snap = recorder.snapshot(0);
        assert_eq!(snap.len(), 5);
        assert!((snap[0] - 0.1).abs() < 0.001);

        let snap = recorder.snapshot(3);
        assert_eq!(snap.len(), 2);
        assert!((snap[0] - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_sample_count_with_manual_samples() {
        let recorder = AudioRecorder::new();
        assert_eq!(recorder.sample_count(), 0);
        {
            let buf = recorder.sample_buffer();
            buf.lock().unwrap().extend_from_slice(&[0.0; 100]);
        }
        assert_eq!(recorder.sample_count(), 100);
    }

    #[test]
    fn test_mix_to_mono_six_channels() {
        // 6-channel surround: one frame
        let data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mono = mix_to_mono(&data, 6);
        assert_eq!(mono.len(), 1);
        assert!((mono[0] - 1.0 / 6.0).abs() < 0.001);
    }

    #[test]
    fn test_resample_ratio_accuracy() {
        // 44.1kHz -> 16kHz: common real-world ratio
        let input: Vec<f32> = (0..44100).map(|i| (i as f32 / 44100.0).sin()).collect();
        let output = resample(&input, 44100, 16000);
        // Should produce approximately 16000 samples
        assert!((output.len() as i64 - 16000).abs() <= 1);
    }

    #[test]
    fn test_f32_to_i16_negative_half() {
        let v = f32_to_i16(-0.5);
        assert!(v < -16000 && v > -17000);
    }

    #[test]
    fn test_stop_clears_current_path() {
        let mut recorder = AudioRecorder::new();
        recorder.current_path = Some(std::path::PathBuf::from("/tmp/test.wav"));
        let path = recorder.stop();
        // stop() should return and clear the path
        assert_eq!(path, Some(std::path::PathBuf::from("/tmp/test.wav")));
        assert!(recorder.current_path.is_none());
    }

    // -- pre-roll --

    #[test]
    fn test_pre_roll_constants() {
        assert_eq!(PRE_ROLL_MS, 200);
        assert_eq!(PRE_ROLL_SAMPLES, 3200);
    }

    #[test]
    fn test_warm_is_idempotent_without_device() {
        // warm() will fail without a real audio device, but calling new() is fine
        let recorder = AudioRecorder::new();
        assert!(recorder.stream.is_none());
    }
}
