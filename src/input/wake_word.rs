//! Wake word detection using a dedicated Whisper tiny model.
//!
//! Continuously captures audio via a dedicated cpal stream, runs VAD to
//! detect speech, and transcribes short windows with Whisper tiny to check
//! for the configured wake/stop phrase. This keeps CPU usage low: the
//! neural network only runs when the VAD detects speech.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use crate::audio::capture::TARGET_RATE;
use crate::transcription::transcriber::Transcriber;
use crate::transcription::vad;

/// Duration of the detection window in seconds.
const WINDOW_SECS: f32 = 3.0;

/// Samples in one detection window.
const WINDOW_SAMPLES: usize = (TARGET_RATE as f32 * WINDOW_SECS) as usize;

/// How often to check for speech (milliseconds).
const POLL_INTERVAL_MS: u64 = 300;

/// Minimum silence gap between detections to avoid re-triggering.
const COOLDOWN_MS: u64 = 2000;

/// Events emitted by the wake word detector.
#[derive(Debug, Clone)]
pub enum WakeWordEvent {
    /// The wake phrase was detected — start dictation.
    WakeWordDetected,
    /// The stop phrase was detected — stop dictation.
    StopPhraseDetected,
}

/// Handle to control the wake word detector thread.
pub struct WakeWordHandle {
    stop_tx: mpsc::Sender<()>,
    join_handle: Option<std::thread::JoinHandle<()>>,
    paused: Arc<AtomicBool>,
}

impl WakeWordHandle {
    /// Pause detection (e.g., while dictation is active).
    pub fn pause(&self) {
        self.paused.store(true, Ordering::Relaxed);
        log::debug!("Wake word detection paused");
    }

    /// Resume detection.
    pub fn resume(&self) {
        self.paused.store(false, Ordering::Relaxed);
        log::debug!("Wake word detection resumed");
    }

    /// Stop and join the detector thread.
    pub fn stop(mut self) {
        let _ = self.stop_tx.send(());
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for WakeWordHandle {
    fn drop(&mut self) {
        let _ = self.stop_tx.send(());
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}

/// Start the wake word detector.
///
/// Loads Whisper tiny (downloading if needed), opens a dedicated audio
/// stream, and monitors for the wake/stop phrases. Sends events via `tx`.
pub fn start_detector(
    wake_phrase: String,
    stop_phrase: String,
    tx: mpsc::Sender<WakeWordEvent>,
) -> anyhow::Result<WakeWordHandle> {
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let paused = Arc::new(AtomicBool::new(false));
    let paused_clone = paused.clone();

    let join_handle = std::thread::spawn(move || {
        if let Err(e) = detector_thread(wake_phrase, stop_phrase, tx, stop_rx, paused_clone) {
            log::error!("Wake word detector failed: {e}");
        }
    });

    Ok(WakeWordHandle {
        stop_tx,
        join_handle: Some(join_handle),
        paused,
    })
}

fn detector_thread(
    wake_phrase: String,
    stop_phrase: String,
    tx: mpsc::Sender<WakeWordEvent>,
    stop_rx: mpsc::Receiver<()>,
    paused: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    // Ensure the tiny model is available
    let model_size = "tiny.en";
    if !crate::transcription::transcriber::model_exists(model_size) {
        log::info!("Downloading {model_size} model for wake word detection...");
        crate::transcription::model::download(model_size, |_| {})?;
    }

    let model_path = crate::transcription::transcriber::find_model(model_size)
        .ok_or_else(|| anyhow::anyhow!("Wake word model '{model_size}' not found"))?;

    let transcriber = Transcriber::new(&model_path, "en")?;
    log::info!("Wake word detector ready (phrase: \"{wake_phrase}\")");

    // Audio capture ring buffer shared with the cpal callback
    let ring_buffer: Arc<Mutex<Vec<f32>>> =
        Arc::new(Mutex::new(Vec::with_capacity(WINDOW_SAMPLES * 2)));

    // Open a dedicated cpal audio stream for wake word detection
    let ring_clone = ring_buffer.clone();
    let _stream = open_capture_stream(ring_clone)?;

    let wake_lower = wake_phrase.to_lowercase();
    let stop_lower = stop_phrase.to_lowercase();
    let mut last_detection = std::time::Instant::now()
        .checked_sub(std::time::Duration::from_millis(COOLDOWN_MS * 2))
        .unwrap_or_else(std::time::Instant::now);

    loop {
        // Check for stop signal
        match stop_rx.try_recv() {
            Ok(()) | Err(mpsc::TryRecvError::Disconnected) => break,
            Err(mpsc::TryRecvError::Empty) => {}
        }

        // Skip if paused
        if paused.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS));
            continue;
        }

        // Wait for enough audio
        let samples: Vec<f32> = {
            let buf = ring_buffer.lock().unwrap_or_else(|e| e.into_inner());
            if buf.len() < WINDOW_SAMPLES {
                drop(buf);
                std::thread::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS));
                continue;
            }
            // Take the most recent window
            let start = buf.len().saturating_sub(WINDOW_SAMPLES);
            buf[start..].to_vec()
        };

        // Trim the ring buffer to prevent unbounded growth
        {
            let mut buf = ring_buffer.lock().unwrap_or_else(|e| e.into_inner());
            if buf.len() > WINDOW_SAMPLES * 3 {
                let drain_to = buf.len() - WINDOW_SAMPLES * 2;
                buf.drain(..drain_to);
            }
        }

        // Only transcribe if VAD detects speech
        if !vad::contains_speech(&samples) {
            std::thread::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS));
            continue;
        }

        // Cooldown check
        if last_detection.elapsed() < std::time::Duration::from_millis(COOLDOWN_MS) {
            std::thread::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS));
            continue;
        }

        // Transcribe the window
        match transcriber.transcribe_samples(&samples, false) {
            Ok(text) => {
                let text_lower = text.to_lowercase();
                log::debug!("Wake word heard: \"{text}\"");

                if contains_phrase(&text_lower, &wake_lower) {
                    log::info!("Wake word detected!");
                    last_detection = std::time::Instant::now();
                    if tx.send(WakeWordEvent::WakeWordDetected).is_err() {
                        break;
                    }
                } else if contains_phrase(&text_lower, &stop_lower) {
                    log::info!("Stop phrase detected!");
                    last_detection = std::time::Instant::now();
                    if tx.send(WakeWordEvent::StopPhraseDetected).is_err() {
                        break;
                    }
                }
            }
            Err(e) => {
                log::warn!("Wake word transcription failed: {e}");
            }
        }

        std::thread::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS));
    }

    log::info!("Wake word detector stopped");
    Ok(())
}

/// Check if `text` contains the given `phrase` (fuzzy word-boundary match).
fn contains_phrase(text: &str, phrase: &str) -> bool {
    if phrase.is_empty() {
        return false;
    }

    let phrase_words: Vec<&str> = phrase.split_whitespace().collect();
    let text_words: Vec<&str> = text.split_whitespace().collect();

    if phrase_words.len() > text_words.len() {
        return false;
    }

    text_words.windows(phrase_words.len()).any(|window| {
        window.iter().zip(phrase_words.iter()).all(|(tw, pw)| {
            let tw_clean = tw.trim_matches(|c: char| c.is_ascii_punctuation());
            let pw_clean = pw.trim_matches(|c: char| c.is_ascii_punctuation());
            tw_clean == pw_clean
        })
    })
}

/// Open a cpal input stream that pushes 16 kHz mono samples into `buffer`.
fn open_capture_stream(buffer: Arc<Mutex<Vec<f32>>>) -> anyhow::Result<cpal::Stream> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No audio input device"))?;

    let supported = device.default_input_config()?;
    let sample_rate = supported.sample_rate();
    let channels = supported.channels() as usize;

    let config: cpal::StreamConfig = supported.into();

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            // Mix to mono
            let mono: Vec<f32> = if channels == 1 {
                data.to_vec()
            } else {
                data.chunks(channels)
                    .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                    .collect()
            };

            // Resample to 16 kHz if needed
            let samples_16k = if sample_rate == TARGET_RATE {
                mono
            } else {
                resample_simple(&mono, sample_rate, TARGET_RATE)
            };

            if let Ok(mut buf) = buffer.try_lock() {
                buf.extend_from_slice(&samples_16k);
            }
        },
        |err| {
            log::error!("Wake word audio error: {err}");
        },
        None,
    )?;

    stream.play()?;
    Ok(stream)
}

/// Simple linear resampling.
fn resample_simple(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || input.is_empty() {
        return input.to_vec();
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (input.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = src_pos - idx as f64;

        let sample = if idx + 1 < input.len() {
            input[idx] * (1.0 - frac as f32) + input[idx + 1] * frac as f32
        } else if idx < input.len() {
            input[idx]
        } else {
            0.0
        };
        output.push(sample);
    }

    output
}

/// Check streaming partial text for the stop phrase and return
/// the text with the stop phrase removed if found.
pub fn check_and_strip_stop_phrase(text: &str, stop_phrase: &str) -> Option<String> {
    let text_lower = text.to_lowercase();
    let stop_lower = stop_phrase.to_lowercase();

    if !contains_phrase(&text_lower, &stop_lower) {
        return None;
    }

    // Remove the stop phrase from the text
    let phrase_words: Vec<&str> = stop_phrase.split_whitespace().collect();
    let text_words: Vec<&str> = text.split_whitespace().collect();

    // Find the position of the stop phrase in the text
    let phrase_lower_words: Vec<&str> = stop_lower.split_whitespace().collect();
    let text_lower_words: Vec<String> = text_words
        .iter()
        .map(|w| {
            w.to_lowercase()
                .trim_matches(|c: char| c.is_ascii_punctuation())
                .to_string()
        })
        .collect();

    for i in 0..=text_words.len().saturating_sub(phrase_words.len()) {
        let matches = text_lower_words[i..i + phrase_lower_words.len()]
            .iter()
            .zip(phrase_lower_words.iter())
            .all(|(tw, pw)| {
                let pw_clean = pw.trim_matches(|c: char| c.is_ascii_punctuation());
                tw == pw_clean
            });

        if matches {
            let mut result_words: Vec<&str> = Vec::new();
            result_words.extend_from_slice(&text_words[..i]);
            result_words.extend_from_slice(&text_words[i + phrase_words.len()..]);
            let result = result_words.join(" ").trim().to_string();
            return Some(result);
        }
    }

    // Fallback: couldn't pinpoint location, return text as-is
    Some(text.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contains_phrase_basic() {
        assert!(contains_phrase("hello murmur start please", "murmur start"));
        assert!(contains_phrase("murmur start", "murmur start"));
        assert!(!contains_phrase("hello world", "murmur start"));
    }

    #[test]
    fn test_contains_phrase_punctuation() {
        assert!(contains_phrase("hello, murmur start.", "murmur start"));
        assert!(contains_phrase("\"murmur start\"", "murmur start"));
    }

    #[test]
    fn test_contains_phrase_empty() {
        assert!(!contains_phrase("hello", ""));
        assert!(!contains_phrase("", "murmur start"));
    }

    #[test]
    fn test_contains_phrase_partial() {
        assert!(!contains_phrase("murmur", "murmur start"));
        assert!(!contains_phrase("start", "murmur start"));
    }

    #[test]
    fn test_check_and_strip_stop_phrase() {
        let result = check_and_strip_stop_phrase("hello world murmur stop thanks", "murmur stop");
        assert_eq!(result, Some("hello world thanks".to_string()));
    }

    #[test]
    fn test_check_and_strip_stop_phrase_at_end() {
        let result = check_and_strip_stop_phrase("hello world murmur stop", "murmur stop");
        assert_eq!(result, Some("hello world".to_string()));
    }

    #[test]
    fn test_check_and_strip_stop_phrase_at_start() {
        let result = check_and_strip_stop_phrase("murmur stop hello world", "murmur stop");
        assert_eq!(result, Some("hello world".to_string()));
    }

    #[test]
    fn test_check_and_strip_stop_phrase_not_found() {
        let result = check_and_strip_stop_phrase("hello world", "murmur stop");
        assert_eq!(result, None);
    }

    #[test]
    fn test_resample_simple_same_rate() {
        let input = vec![1.0, 2.0, 3.0];
        let output = resample_simple(&input, 16000, 16000);
        assert_eq!(output, input);
    }

    #[test]
    fn test_resample_simple_downsample() {
        let input: Vec<f32> = (0..48000).map(|i| (i as f32 / 48000.0).sin()).collect();
        let output = resample_simple(&input, 48000, 16000);
        // Should be roughly 1/3 the length
        assert!((output.len() as f32 - 16000.0).abs() < 2.0);
    }

    #[test]
    fn test_resample_simple_empty() {
        let output = resample_simple(&[], 48000, 16000);
        assert!(output.is_empty());
    }
}
