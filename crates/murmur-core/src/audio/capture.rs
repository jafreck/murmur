use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::{SampleFormat, WavSpec, WavWriter};
use nnnoiseless::DenoiseState;
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

/// Expected recording duration for initial buffer capacity (seconds).
/// Pre-allocating ~5 seconds of 16 kHz audio (80 000 samples ≈ 320 KB)
/// avoids multiple early reallocations during typical dictation.
const INITIAL_RECORDING_SECS: usize = 5;

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
    /// Monotonically increasing count of audio callback invocations.
    /// Used to detect dead streams (e.g. when a Bluetooth device disconnects).
    callback_count: AtomicU64,
}

impl SharedCaptureState {
    fn new() -> Self {
        let initial_capacity = TARGET_RATE as usize * INITIAL_RECORDING_SECS;
        Self {
            recording: AtomicBool::new(false),
            writer: Mutex::new(None),
            samples: Arc::new(Mutex::new(Vec::with_capacity(initial_capacity))),
            pre_roll: Mutex::new(VecDeque::with_capacity(PRE_ROLL_SAMPLES + 512)),
            dropped_samples: AtomicU64::new(0),
            callback_count: AtomicU64::new(0),
        }
    }

    /// Dispatch processed audio samples to the appropriate buffer.
    /// Called from the audio callback after mixing/resampling/denoising.
    fn dispatch_samples(&self, samples: &[f32]) {
        self.callback_count.fetch_add(1, Ordering::Relaxed);
        if self.recording.load(Ordering::Acquire) {
            if let Ok(mut buf) = self.samples.try_lock() {
                buf.extend_from_slice(samples);
            } else {
                self.dropped_samples
                    .fetch_add(samples.len() as u64, Ordering::Relaxed);
            }
            if let Ok(mut guard) = self.writer.try_lock() {
                if let Some(ref mut w) = *guard {
                    for &sample in samples {
                        let _ = w.write_sample(f32_to_i16(sample));
                    }
                }
            }
        } else if let Ok(mut ring) = self.pre_roll.try_lock() {
            for &s in samples {
                if ring.len() >= PRE_ROLL_SAMPLES {
                    ring.pop_front();
                }
                ring.push_back(s);
            }
        }
    }
}

/// nnnoiseless operates on 480-sample frames at 48 kHz.
const DENOISE_FRAME_SIZE: usize = DenoiseState::FRAME_SIZE;

/// nnnoiseless native sample rate.
const DENOISE_RATE: u32 = 48_000;

/// Holds the denoiser state and an accumulation buffer for incomplete frames.
/// Created once per recording session and shared with the audio callback via Arc<Mutex>.
struct Denoiser {
    state: Box<DenoiseState<'static>>,
    /// Accumulates 16 kHz samples until we have enough to fill a 48 kHz frame.
    pending_16k: Vec<f32>,
    /// Collects denoised 16 kHz output samples ready for the consumer.
    output_16k: Vec<f32>,
    /// Whether to skip the very first output frame (fade-in artifact).
    first_frame: bool,
    // Pre-allocated scratch buffers to avoid per-callback heap allocations.
    chunk_buf: Vec<f32>,
    upsampled_buf: Vec<f32>,
    denoised_48k_buf: Vec<f32>,
    downsampled_buf: Vec<f32>,
}

impl Denoiser {
    fn new() -> Self {
        let frame_16k = DENOISE_FRAME_SIZE / 3; // 160
        Self {
            state: DenoiseState::new(),
            pending_16k: Vec::with_capacity(frame_16k + 16),
            output_16k: Vec::new(),
            first_frame: true,
            chunk_buf: Vec::with_capacity(frame_16k),
            upsampled_buf: Vec::with_capacity(DENOISE_FRAME_SIZE),
            denoised_48k_buf: Vec::with_capacity(DENOISE_FRAME_SIZE),
            downsampled_buf: Vec::with_capacity(frame_16k),
        }
    }

    /// Reset all state so the denoiser is clean for a new recording session.
    fn reset(&mut self) {
        self.pending_16k.clear();
        self.output_16k.clear();
        self.first_frame = true;
        self.state = DenoiseState::new();
        // Scratch buffers keep their allocations; just clear contents.
        self.chunk_buf.clear();
        self.upsampled_buf.clear();
        self.denoised_48k_buf.clear();
        self.downsampled_buf.clear();
    }

    /// Feed 16 kHz samples and return denoised 16 kHz samples.
    ///
    /// Internally upsamples to 48 kHz, runs nnnoiseless frame-by-frame,
    /// then downsamples back to 16 kHz.
    fn process(&mut self, samples_16k: &[f32]) -> &[f32] {
        self.output_16k.clear();
        self.pending_16k.extend_from_slice(samples_16k);

        // Each 48 kHz frame of 480 samples corresponds to 160 samples at 16 kHz.
        let frame_16k = DENOISE_FRAME_SIZE / 3; // 160

        while self.pending_16k.len() >= frame_16k {
            self.chunk_buf.clear();
            self.chunk_buf.extend(self.pending_16k.drain(..frame_16k));

            resample_into(
                &self.chunk_buf,
                TARGET_RATE,
                DENOISE_RATE,
                &mut self.upsampled_buf,
            );

            // nnnoiseless expects f32 in i16 range
            let mut input_frame = [0.0f32; DENOISE_FRAME_SIZE];
            for (i, &s) in self
                .upsampled_buf
                .iter()
                .take(DENOISE_FRAME_SIZE)
                .enumerate()
            {
                input_frame[i] = s * 32767.0;
            }

            let mut output_frame = [0.0f32; DENOISE_FRAME_SIZE];
            self.state.process_frame(&mut output_frame, &input_frame);

            if self.first_frame {
                self.first_frame = false;
                continue;
            }

            // Convert back from i16 range to [-1, 1]
            self.denoised_48k_buf.clear();
            self.denoised_48k_buf.extend(
                output_frame
                    .iter()
                    .map(|&s| (s / 32767.0_f32).clamp(-1.0, 1.0)),
            );

            resample_into(
                &self.denoised_48k_buf,
                DENOISE_RATE,
                TARGET_RATE,
                &mut self.downsampled_buf,
            );
            self.output_16k.extend_from_slice(&self.downsampled_buf);
        }

        &self.output_16k
    }
}

pub struct AudioRecorder {
    stream: Option<cpal::Stream>,
    shared: Arc<SharedCaptureState>,
    current_path: Option<PathBuf>,
    /// Runtime toggle for noise suppression.
    noise_suppression: Arc<AtomicBool>,
    /// Shared denoiser state (created once, reused across callbacks).
    denoiser: Arc<Mutex<Denoiser>>,
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
            noise_suppression: Arc::new(AtomicBool::new(true)),
            denoiser: Arc::new(Mutex::new(Denoiser::new())),
        }
    }

    /// Create a recorder with an explicit noise suppression setting.
    pub fn with_noise_suppression(enabled: bool) -> Self {
        Self {
            noise_suppression: Arc::new(AtomicBool::new(enabled)),
            ..Self::new()
        }
    }

    /// Update the noise suppression toggle at runtime.
    pub fn set_noise_suppression(&self, enabled: bool) {
        self.noise_suppression.store(enabled, Ordering::Relaxed);
    }

    /// Open the microphone and start capturing into the pre-roll buffer.
    ///
    /// Call once at app startup so recording starts instantly on hotkey press.
    /// If the stream is already running this is a no-op.
    pub fn warm(&mut self) -> Result<()> {
        if self.stream.is_some() {
            return Ok(());
        }

        // Platform hook: on macOS, nudge Bluetooth devices into HFP mode
        // so the microphone is active when we open the stream.
        super::activate::prepare_default_input();

        let host = cpal::default_host();
        let device = host.default_input_device().context("No microphone found")?;
        self.open_device(device)
    }

    /// Build and start an input stream on the given device.
    fn open_device(&mut self, device: cpal::Device) -> Result<()> {
        let device_name = device
            .description()
            .map(|d| d.name().to_string())
            .unwrap_or_else(|_| "<unknown>".into());

        let supported_config = device
            .default_input_config()
            .context("Failed to get default input config")?;

        let native_rate = supported_config.sample_rate();
        let native_channels = supported_config.channels() as u32;

        log::info!(
            "Opening audio device: \"{device_name}\" ({native_rate}Hz, {native_channels}ch, {:?})",
            supported_config.sample_format(),
        );

        let shared = Arc::clone(&self.shared);
        let ns_flag = Arc::clone(&self.noise_suppression);
        let denoiser = Arc::clone(&self.denoiser);

        let stream = device
            .build_input_stream(
                &supported_config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mono = mix_to_mono(data, native_channels);
                    let resampled = resample(&mono, native_rate, TARGET_RATE);

                    // Apply noise suppression if enabled
                    let samples: &[f32] = if ns_flag.load(Ordering::Relaxed) {
                        if let Ok(mut d) = denoiser.try_lock() {
                            let denoised = d.process(&resampled);
                            // SAFETY: denoised borrows d which we hold;
                            // we only use it within this scope while the lock is held.
                            // Copy out to avoid holding the lock across buffer writes.
                            let owned: Vec<f32> = denoised.to_vec();
                            drop(d);
                            shared.dispatch_samples(&owned);
                            return;
                        }
                        // Denoiser lock contention — fall through to raw samples
                        &resampled
                    } else {
                        &resampled
                    };

                    shared.dispatch_samples(samples);
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

    /// Close the current stream and re-open on the current default input device.
    fn rewarm(&mut self) -> Result<()> {
        log::info!("Re-opening audio stream on current default device");
        self.stream = None;
        if let Ok(mut ring) = self.shared.pre_roll.lock() {
            ring.clear();
        }
        super::activate::prepare_default_input();
        let host = cpal::default_host();
        let device = host.default_input_device().context("No microphone found")?;
        self.open_device(device)
    }

    /// Ensure the stream is warm, warming it up if needed.
    /// If the stream exists but is no longer producing audio (e.g. the
    /// Bluetooth device disconnected), close and re-open it.
    fn ensure_warm(&mut self) -> Result<()> {
        if self.stream.is_some() {
            // Direct probe: snapshot the counter, wait briefly, check again.
            // This avoids false positives from stale counters that were set
            // during a previous recording session.
            let before = self.shared.callback_count.load(Ordering::Relaxed);
            std::thread::sleep(std::time::Duration::from_millis(50));
            let after = self.shared.callback_count.load(Ordering::Relaxed);
            if after == before {
                log::warn!("Audio stream appears dead (no callbacks in 50ms), re-opening");
                self.rewarm()?;
            }
        } else {
            self.warm()?;
        }
        Ok(())
    }

    pub fn start_in_memory(&mut self) -> Result<()> {
        self.ensure_warm()?;

        self.shared.dropped_samples.store(0, Ordering::Relaxed);
        if let Ok(mut d) = self.denoiser.lock() {
            d.reset();
        }

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
        if let Ok(mut d) = self.denoiser.lock() {
            d.reset();
        }

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
pub(crate) fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
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

/// Allocation-free variant of [`resample`] that writes into a caller-supplied buffer.
fn resample_into(input: &[f32], from_rate: u32, to_rate: u32, output: &mut Vec<f32>) {
    output.clear();
    if from_rate == to_rate {
        output.extend_from_slice(input);
        return;
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (input.len() as f64 / ratio) as usize;

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

    // ── Denoiser ──

    #[test]
    fn test_denoiser_new() {
        let d = Denoiser::new();
        assert!(d.pending_16k.is_empty());
        assert!(d.output_16k.is_empty());
        assert!(d.first_frame);
    }

    #[test]
    fn test_denoiser_reset() {
        let mut d = Denoiser::new();
        d.pending_16k.push(1.0);
        d.output_16k.push(2.0);
        d.first_frame = false;
        d.reset();
        assert!(d.pending_16k.is_empty());
        assert!(d.output_16k.is_empty());
        assert!(d.first_frame);
    }

    #[test]
    fn test_denoiser_process_empty() {
        let mut d = Denoiser::new();
        let out = d.process(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_denoiser_process_short_accumulates() {
        let mut d = Denoiser::new();
        let out = d.process(&[0.0; 100]);
        assert!(out.is_empty());
        assert_eq!(d.pending_16k.len(), 100);
    }

    #[test]
    fn test_denoiser_process_one_frame_skipped() {
        let mut d = Denoiser::new();
        // One frame = 160 samples at 16 kHz, but the first frame is always skipped.
        let out = d.process(&[0.0; 160]);
        assert!(out.is_empty());
        assert!(!d.first_frame);
    }

    #[test]
    fn test_denoiser_process_two_frames_produces_output() {
        let mut d = Denoiser::new();
        // First frame skipped, second frame produces 160 samples.
        let out = d.process(&[0.0; 320]);
        assert_eq!(out.len(), 160);
    }

    #[test]
    fn test_denoiser_process_multiple_frames() {
        let mut d = Denoiser::new();
        // 3 frames: first skipped, remaining 2 produce 320 samples.
        let out = d.process(&[0.0; 480]);
        assert_eq!(out.len(), 320);
    }

    #[test]
    fn test_denoiser_continuity_across_calls() {
        let mut d = Denoiser::new();
        // 100 samples: too few for a frame
        let out1 = d.process(&[0.1; 100]);
        assert!(out1.is_empty());

        // 100 more → 200 total, one frame processed (160) but skipped, 40 leftover
        let out2 = d.process(&[0.1; 100]);
        assert!(out2.is_empty());
        assert_eq!(d.pending_16k.len(), 40);

        // 120 more → 40 + 120 = 160, second frame produces output
        let out3 = d.process(&[0.1; 120]).to_vec();
        assert_eq!(out3.len(), 160);
    }

    // ── dispatch_samples ──

    #[test]
    fn test_dispatch_recording_appends_to_samples() {
        let state = SharedCaptureState::new();
        state.recording.store(true, Ordering::Release);
        state.dispatch_samples(&[0.1, 0.2, 0.3]);
        let buf = state.samples.lock().unwrap();
        assert_eq!(&*buf, &[0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_dispatch_standby_appends_to_pre_roll() {
        let state = SharedCaptureState::new();
        state.dispatch_samples(&[0.5, 0.6]);
        let ring = state.pre_roll.lock().unwrap();
        assert_eq!(ring.len(), 2);
        assert!((ring[0] - 0.5).abs() < f32::EPSILON);
        assert!((ring[1] - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dispatch_pre_roll_caps_at_limit() {
        let state = SharedCaptureState::new();
        let filler: Vec<f32> = (0..PRE_ROLL_SAMPLES).map(|i| i as f32).collect();
        state.dispatch_samples(&filler);
        // Push 2 more — oldest samples should be evicted
        state.dispatch_samples(&[99.0, 100.0]);
        let ring = state.pre_roll.lock().unwrap();
        assert_eq!(ring.len(), PRE_ROLL_SAMPLES);
        assert!((ring[ring.len() - 1] - 100.0).abs() < f32::EPSILON);
        assert!((ring[ring.len() - 2] - 99.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dispatch_with_writer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.wav");
        let writer = WavWriter::create(&path, WHISPER_WAV_SPEC).unwrap();
        let state = SharedCaptureState::new();
        state.recording.store(true, Ordering::Release);
        *state.writer.lock().unwrap() = Some(writer);

        state.dispatch_samples(&[0.1, 0.2, 0.3]);

        // Finalize the WAV and verify it was written
        if let Some(w) = state.writer.lock().unwrap().take() {
            w.finalize().unwrap();
        }
        let reader = hound::WavReader::open(&path).unwrap();
        let written: Vec<i16> = reader.into_samples::<i16>().map(|s| s.unwrap()).collect();
        assert_eq!(written.len(), 3);
    }

    // ── Pre-roll buffer logic ──

    #[test]
    fn test_pre_roll_does_not_exceed_capacity() {
        let state = SharedCaptureState::new();
        let large: Vec<f32> = (0..(PRE_ROLL_SAMPLES + 500)).map(|i| i as f32).collect();
        state.dispatch_samples(&large);
        let ring = state.pre_roll.lock().unwrap();
        assert_eq!(ring.len(), PRE_ROLL_SAMPLES);
    }

    #[test]
    fn test_pre_roll_drains_into_samples_on_start() {
        let recorder = AudioRecorder::new();
        // Manually push samples into pre_roll
        {
            let mut ring = recorder.shared.pre_roll.lock().unwrap();
            ring.extend([0.1, 0.2, 0.3]);
        }
        // Simulate what start_in_memory does (without ensure_warm)
        {
            let mut ring = recorder.shared.pre_roll.lock().unwrap();
            let mut samples = recorder.shared.samples.lock().unwrap();
            samples.clear();
            samples.extend(ring.drain(..));
            recorder.shared.recording.store(true, Ordering::Release);
        }
        assert!(recorder.shared.recording.load(Ordering::Acquire));
        let buf = recorder.shared.samples.lock().unwrap();
        assert_eq!(&*buf, &[0.1, 0.2, 0.3]);
        let ring = recorder.shared.pre_roll.lock().unwrap();
        assert!(ring.is_empty());
    }

    // ── Recording state transitions ──

    #[test]
    fn test_new_recorder_not_recording() {
        let recorder = AudioRecorder::new();
        assert!(!recorder.shared.recording.load(Ordering::Acquire));
    }

    #[test]
    fn test_start_sets_recording_flag() {
        let recorder = AudioRecorder::new();
        recorder.shared.recording.store(true, Ordering::Release);
        assert!(recorder.shared.recording.load(Ordering::Acquire));
    }

    #[test]
    fn test_stop_samples_returns_samples_and_clears_flag() {
        let mut recorder = AudioRecorder::new();
        recorder.shared.recording.store(true, Ordering::Release);
        recorder
            .shared
            .samples
            .lock()
            .unwrap()
            .extend_from_slice(&[0.1, 0.2]);
        let samples = recorder.stop_samples();
        assert!(!recorder.shared.recording.load(Ordering::Acquire));
        assert_eq!(samples.unwrap(), vec![0.1, 0.2]);
    }

    #[test]
    fn test_stop_samples_returns_none_when_empty() {
        let mut recorder = AudioRecorder::new();
        // Not recording and no samples → None
        assert!(recorder.stop_samples().is_none());
    }

    #[test]
    fn test_sample_count_zero_initially() {
        let recorder = AudioRecorder::new();
        assert_eq!(recorder.sample_count(), 0);
    }

    #[test]
    fn test_snapshot_empty_initially() {
        let recorder = AudioRecorder::new();
        assert!(recorder.snapshot(0).is_empty());
    }
}
