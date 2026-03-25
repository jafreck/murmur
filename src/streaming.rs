//! Chunked streaming transcription.
//!
//! While the user holds the toggle key, audio is recorded continuously.
//! This module periodically grabs overlapping chunks from the in-memory
//! sample buffer, transcribes each chunk, and stitches partial results
//! together using longest-common-subsequence (LCS) deduplication on the
//! overlap region.

use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use crate::transcriber::Transcriber;

// ── Configuration ──────────────────────────────────────────────────────

/// Duration of each chunk sent to Whisper, in seconds.
const CHUNK_DURATION_SECS: f32 = 5.0;
/// How much overlap between consecutive chunks, in seconds.
const OVERLAP_DURATION_SECS: f32 = 2.0;
/// Minimum interval between chunk transcriptions, in milliseconds.
const POLL_INTERVAL_MS: u64 = 500;
/// RMS threshold below which a chunk is considered silence and skipped.
const SILENCE_RMS_THRESHOLD: f32 = 0.005;
/// Sample rate of the audio buffer (must match AudioRecorder output).
const SAMPLE_RATE: usize = 16_000;

// ── Public API ─────────────────────────────────────────────────────────

/// A message carrying newly-transcribed text from a streaming chunk.
pub enum StreamingEvent {
    /// New words to insert at the cursor (incremental).
    PartialText(String),
}

/// Start streaming transcription in a background thread.
///
/// Reads from `sample_buffer` (the AudioRecorder's shared buffer), transcribes
/// overlapping chunks, and sends incremental text via `tx`.
///
/// Returns a handle that, when dropped or sent `()`, signals the thread to stop.
pub fn start_streaming(
    sample_buffer: Arc<Mutex<Vec<f32>>>,
    transcriber: Arc<Transcriber>,
    language: String,
    translate: bool,
    spoken_punctuation: bool,
    tx: mpsc::Sender<StreamingEvent>,
) -> mpsc::Sender<()> {
    let (stop_tx, stop_rx) = mpsc::channel::<()>();

    std::thread::spawn(move || {
        streaming_loop(
            sample_buffer,
            transcriber,
            &language,
            translate,
            spoken_punctuation,
            tx,
            stop_rx,
        );
    });

    stop_tx
}

// ── Internal ───────────────────────────────────────────────────────────

fn streaming_loop(
    sample_buffer: Arc<Mutex<Vec<f32>>>,
    transcriber: Arc<Transcriber>,
    _language: &str,
    translate: bool,
    spoken_punctuation: bool,
    tx: mpsc::Sender<StreamingEvent>,
    stop_rx: mpsc::Receiver<()>,
) {
    let chunk_samples = (CHUNK_DURATION_SECS * SAMPLE_RATE as f32) as usize;
    let overlap_samples = (OVERLAP_DURATION_SECS * SAMPLE_RATE as f32) as usize;
    let step_samples = chunk_samples - overlap_samples;

    // Position in the sample buffer where the next chunk starts.
    let mut cursor: usize = 0;
    // Accumulated words from all previous chunks (the "committed" output).
    let mut committed_words: Vec<String> = Vec::new();

    loop {
        // Check for stop signal (non-blocking).
        // Breaks on both explicit message and sender being dropped.
        match stop_rx.try_recv() {
            Ok(()) | Err(mpsc::TryRecvError::Disconnected) => break,
            Err(mpsc::TryRecvError::Empty) => {}
        }

        // How many samples are available?
        let total_samples = sample_buffer
            .lock()
            .map(|b| b.len())
            .unwrap_or(0);

        // Wait until we have at least one full chunk from the current cursor.
        if total_samples < cursor + chunk_samples {
            std::thread::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS));
            continue;
        }

        // Extract the chunk.
        let chunk: Vec<f32> = {
            let buf = sample_buffer.lock().unwrap();
            buf[cursor..cursor + chunk_samples].to_vec()
        };

        // Skip silent chunks.
        if is_silent(&chunk) {
            cursor += step_samples;
            continue;
        }

        // Transcribe the chunk (this is the expensive part).
        let chunk_text = match transcriber.transcribe_samples(&chunk, translate) {
            Ok(t) => t,
            Err(e) => {
                log::error!("Streaming chunk transcription failed: {e}");
                cursor += step_samples;
                continue;
            }
        };

        if chunk_text.is_empty() {
            cursor += step_samples;
            continue;
        }

        let chunk_words = split_words(&chunk_text);

        // Stitch: find where committed output overlaps with this chunk's words,
        // then emit only the new (non-overlapping) portion.
        let new_words = stitch(&committed_words, &chunk_words);

        if !new_words.is_empty() {
            let new_text = if spoken_punctuation {
                crate::postprocess::process(&new_words.join(" "))
            } else {
                new_words.join(" ")
            };

            if !new_text.is_empty() {
                committed_words.extend(new_words);
                let _ = tx.send(StreamingEvent::PartialText(new_text));
            }
        }

        // Advance cursor by the non-overlapping step.
        cursor += step_samples;
    }
}

// ── Stitching ──────────────────────────────────────────────────────────

/// Given previously committed words and a new chunk's words, determine which
/// words from the chunk are genuinely new (i.e. not already covered by the
/// committed output).
///
/// Uses a suffix-prefix LCS match: we look for the longest suffix of
/// `committed` that matches a prefix of `chunk_words`, then return
/// everything in `chunk_words` after that matched prefix.
pub fn stitch(committed: &[String], chunk_words: &[String]) -> Vec<String> {
    if committed.is_empty() {
        return chunk_words.to_vec();
    }
    if chunk_words.is_empty() {
        return Vec::new();
    }

    // Only compare the tail of committed (at most as many words as the chunk).
    let tail_len = committed.len().min(chunk_words.len());
    let tail = &committed[committed.len() - tail_len..];

    // Find the longest prefix of chunk_words that matches a suffix of tail.
    let best_match = longest_suffix_prefix_match(tail, chunk_words);

    if best_match > 0 {
        chunk_words[best_match..].to_vec()
    } else {
        chunk_words.to_vec()
    }
}

/// Find the length of the longest suffix of `a` that equals a prefix of `b`,
/// using case-insensitive comparison.
fn longest_suffix_prefix_match(a: &[String], b: &[String]) -> usize {
    let max_len = a.len().min(b.len());
    let mut best = 0;

    for len in 1..=max_len {
        let suffix = &a[a.len() - len..];
        let prefix = &b[..len];
        if suffix
            .iter()
            .zip(prefix.iter())
            .all(|(s, p)| s.eq_ignore_ascii_case(p))
        {
            best = len;
        }
    }

    best
}

// ── Utilities ──────────────────────────────────────────────────────────

/// Check whether a chunk of audio is effectively silence.
pub fn is_silent(samples: &[f32]) -> bool {
    if samples.is_empty() {
        return true;
    }
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    rms < SILENCE_RMS_THRESHOLD
}

/// Split text into words, normalising whitespace.
fn split_words(text: &str) -> Vec<String> {
    text.split_whitespace().map(|w| w.to_string()).collect()
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stitch_no_overlap() {
        let committed: Vec<String> = vec!["hello".into(), "world".into()];
        let chunk: Vec<String> = vec!["foo".into(), "bar".into()];
        let result = stitch(&committed, &chunk);
        assert_eq!(result, vec!["foo", "bar"]);
    }

    #[test]
    fn test_stitch_full_overlap() {
        let committed: Vec<String> = vec!["the".into(), "quick".into(), "brown".into()];
        let chunk: Vec<String> = vec!["quick".into(), "brown".into(), "fox".into()];
        let result = stitch(&committed, &chunk);
        assert_eq!(result, vec!["fox"]);
    }

    #[test]
    fn test_stitch_single_word_overlap() {
        let committed: Vec<String> = vec!["hello".into(), "world".into()];
        let chunk: Vec<String> = vec!["world".into(), "today".into()];
        let result = stitch(&committed, &chunk);
        assert_eq!(result, vec!["today"]);
    }

    #[test]
    fn test_stitch_empty_committed() {
        let committed: Vec<String> = vec![];
        let chunk: Vec<String> = vec!["hello".into(), "world".into()];
        let result = stitch(&committed, &chunk);
        assert_eq!(result, vec!["hello", "world"]);
    }

    #[test]
    fn test_stitch_empty_chunk() {
        let committed: Vec<String> = vec!["hello".into()];
        let chunk: Vec<String> = vec![];
        let result = stitch(&committed, &chunk);
        assert!(result.is_empty());
    }

    #[test]
    fn test_stitch_case_insensitive() {
        let committed: Vec<String> = vec!["Hello".into(), "World".into()];
        let chunk: Vec<String> = vec!["world".into(), "today".into()];
        let result = stitch(&committed, &chunk);
        assert_eq!(result, vec!["today"]);
    }

    #[test]
    fn test_stitch_complete_duplicate() {
        let committed: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let chunk: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let result = stitch(&committed, &chunk);
        assert!(result.is_empty());
    }

    #[test]
    fn test_is_silent_zeros() {
        let samples = vec![0.0f32; 16000];
        assert!(is_silent(&samples));
    }

    #[test]
    fn test_is_silent_low_noise() {
        let samples = vec![0.001f32; 16000];
        assert!(is_silent(&samples));
    }

    #[test]
    fn test_is_silent_loud() {
        let samples = vec![0.5f32; 16000];
        assert!(!is_silent(&samples));
    }

    #[test]
    fn test_is_silent_empty() {
        assert!(is_silent(&[]));
    }

    #[test]
    fn test_longest_suffix_prefix_match_basic() {
        let a: Vec<String> = vec!["x".into(), "y".into(), "z".into()];
        let b: Vec<String> = vec!["y".into(), "z".into(), "w".into()];
        assert_eq!(longest_suffix_prefix_match(&a, &b), 2);
    }

    #[test]
    fn test_longest_suffix_prefix_match_none() {
        let a: Vec<String> = vec!["a".into(), "b".into()];
        let b: Vec<String> = vec!["c".into(), "d".into()];
        assert_eq!(longest_suffix_prefix_match(&a, &b), 0);
    }

    #[test]
    fn test_longest_suffix_prefix_match_full() {
        let a: Vec<String> = vec!["a".into(), "b".into()];
        let b: Vec<String> = vec!["a".into(), "b".into()];
        assert_eq!(longest_suffix_prefix_match(&a, &b), 2);
    }

    #[test]
    fn test_split_words_basic() {
        assert_eq!(split_words("hello world"), vec!["hello", "world"]);
    }

    #[test]
    fn test_split_words_extra_whitespace() {
        assert_eq!(split_words("  hello   world  "), vec!["hello", "world"]);
    }

    #[test]
    fn test_split_words_empty() {
        assert!(split_words("").is_empty());
        assert!(split_words("   ").is_empty());
    }

    #[test]
    fn test_split_words_single() {
        assert_eq!(split_words("hello"), vec!["hello"]);
    }

    #[test]
    fn test_constants() {
        assert!(CHUNK_DURATION_SECS > 0.0);
        assert!(OVERLAP_DURATION_SECS > 0.0);
        assert!(OVERLAP_DURATION_SECS < CHUNK_DURATION_SECS);
        assert!(POLL_INTERVAL_MS > 0);
        assert!(SILENCE_RMS_THRESHOLD > 0.0);
        assert_eq!(SAMPLE_RATE, 16_000);
    }

    #[test]
    fn test_streaming_event_partial_text() {
        let event = StreamingEvent::PartialText("hello".to_string());
        match event {
            StreamingEvent::PartialText(s) => assert_eq!(s, "hello"),
        }
    }

    #[test]
    fn test_is_silent_threshold_boundary() {
        // Just below threshold
        let val = SILENCE_RMS_THRESHOLD * 0.9;
        let samples = vec![val; 1000];
        assert!(is_silent(&samples));

        // Just above threshold
        let val = SILENCE_RMS_THRESHOLD * 1.1;
        let samples = vec![val; 1000];
        assert!(!is_silent(&samples));
    }

    #[test]
    fn test_stitch_long_committed_short_chunk() {
        let committed: Vec<String> = (0..100).map(|i| format!("w{i}")).collect();
        let chunk: Vec<String> = vec!["w99".into(), "new1".into()];
        let result = stitch(&committed, &chunk);
        assert_eq!(result, vec!["new1"]);
    }

    #[test]
    fn test_longest_suffix_prefix_match_case_insensitive() {
        let a: Vec<String> = vec!["Hello".into(), "WORLD".into()];
        let b: Vec<String> = vec!["hello".into(), "world".into(), "test".into()];
        assert_eq!(longest_suffix_prefix_match(&a, &b), 2);
    }
}
