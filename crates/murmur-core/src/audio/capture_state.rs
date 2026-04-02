use hound::WavWriter;
use std::collections::VecDeque;
use std::fs::File;
use std::io::BufWriter;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::resample::{f32_to_i16, TARGET_RATE};

/// Duration of the pre-roll buffer in milliseconds.
/// Captures audio *before* the user presses record so the first word isn't clipped.
pub(super) const PRE_ROLL_MS: u32 = 200;

/// Number of 16 kHz samples in the pre-roll buffer.
pub(super) const PRE_ROLL_SAMPLES: usize = (TARGET_RATE * PRE_ROLL_MS / 1000) as usize;

/// Expected recording duration for initial buffer capacity (seconds).
/// Pre-allocating ~5 seconds of 16 kHz audio (80 000 samples ≈ 320 KB)
/// avoids multiple early reallocations during typical dictation.
const INITIAL_RECORDING_SECS: usize = 5;

/// Shared state between the audio callback thread and the main thread.
pub(super) struct SharedCaptureState {
    /// True while actively recording (as opposed to standby pre-roll capture).
    pub(super) recording: AtomicBool,
    /// WAV file writer, set only during file-backed recordings.
    pub(super) writer: Mutex<Option<WavWriter<BufWriter<File>>>>,
    /// In-memory buffer of 16 kHz mono f32 samples captured since `start()`.
    pub(super) samples: Arc<Mutex<Vec<f32>>>,
    /// Ring buffer capturing recent audio while in standby mode.
    /// When recording starts, these samples are drained into `samples`
    /// so the beginning of speech is preserved.
    pub(super) pre_roll: Mutex<VecDeque<f32>>,
    /// Count of samples dropped due to lock contention.
    pub(super) dropped_samples: AtomicU64,
    /// Monotonically increasing count of audio callback invocations.
    /// Used to detect dead streams (e.g. when a Bluetooth device disconnects).
    pub(super) callback_count: AtomicU64,
}

impl SharedCaptureState {
    pub(super) fn new() -> Self {
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
    pub(super) fn dispatch_samples(&self, samples: &[f32]) {
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
                        // Hot audio path: logging per-sample errors would be too noisy
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

#[cfg(test)]
mod tests {
    use super::*;
    use hound::WavWriter;

    use super::super::resample::WHISPER_WAV_SPEC;

    #[test]
    fn test_pre_roll_constants() {
        assert_eq!(PRE_ROLL_MS, 200);
        assert_eq!(PRE_ROLL_SAMPLES, 3200);
    }

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

    #[test]
    fn test_pre_roll_does_not_exceed_capacity() {
        let state = SharedCaptureState::new();
        let large: Vec<f32> = (0..(PRE_ROLL_SAMPLES + 500)).map(|i| i as f32).collect();
        state.dispatch_samples(&large);
        let ring = state.pre_roll.lock().unwrap();
        assert_eq!(ring.len(), PRE_ROLL_SAMPLES);
    }
}
