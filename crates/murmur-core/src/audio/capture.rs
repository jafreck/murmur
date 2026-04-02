use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::WavWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use super::capture_state::{SharedCaptureState, PRE_ROLL_MS};
use super::denoise::Denoiser;
use super::resample::{f32_to_i16, mix_to_mono, resample, TARGET_RATE, WHISPER_WAV_SPEC};

pub struct AudioRecorder {
    stream: Option<cpal::Stream>,
    shared: Arc<SharedCaptureState>,
    current_path: Option<PathBuf>,
    /// Runtime toggle for noise suppression.
    noise_suppression: Arc<AtomicBool>,
    /// Shared denoiser state (created once, reused across callbacks).
    denoiser: Arc<Mutex<Denoiser>>,
}

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
                            // Hot audio path: logging per-sample errors would be too noisy
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
    #[cfg(test)]
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
    #[cfg(test)]
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
                if let Err(e) = writer.finalize() {
                    log::warn!("Failed to finalize WAV file: {e}");
                }
            }
        }

        self.current_path.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

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
    fn test_stop_clears_current_path() {
        let mut recorder = AudioRecorder::new();
        recorder.current_path = Some(std::path::PathBuf::from("/tmp/test.wav"));
        let path = recorder.stop();
        // stop() should return and clear the path
        assert_eq!(path, Some(std::path::PathBuf::from("/tmp/test.wav")));
        assert!(recorder.current_path.is_none());
    }

    #[test]
    fn test_warm_is_idempotent_without_device() {
        // warm() will fail without a real audio device, but calling new() is fine
        let recorder = AudioRecorder::new();
        assert!(recorder.stream.is_none());
    }

    // ── Pre-roll buffer logic ──

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
