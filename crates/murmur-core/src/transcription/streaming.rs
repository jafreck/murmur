//! Growing-window streaming transcription.
//!
//! While the user holds the hotkey, audio is recorded continuously.
//! This module periodically transcribes a growing window of audio
//! (from the start up to the current position), diffs against what
//! was already emitted, and sends only new words incrementally.
//!
//! Once the window exceeds `MAX_WINDOW_SECS` the start anchor slides
//! forward to keep each Whisper pass short and responsive.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use super::transcriber::Transcriber;
use crate::audio::capture::TARGET_RATE;

// ── Configuration ──────────────────────────────────────────────────────

/// Minimum new audio (seconds) before re-transcribing.
const MIN_NEW_AUDIO_SECS: f32 = 1.0;
/// Minimum interval between transcription attempts, in milliseconds.
const POLL_INTERVAL_MS: u64 = 300;
/// Maximum window size sent to Whisper during streaming, in seconds.
/// Smaller windows keep each inference pass fast at the cost of less
/// context. 10 s is a good balance between latency and accuracy.
const MAX_WINDOW_SECS: f32 = 10.0;

// ── Public API ─────────────────────────────────────────────────────────

/// A message carrying newly-transcribed text from a streaming chunk.
pub enum StreamingEvent {
    /// Replace the last `replace_chars` characters with `text`.
    /// If `replace_chars` is 0, just append.
    PartialText { text: String, replace_chars: usize },
    /// VAD detected speech in the audio stream (heartbeat for silence timeout).
    SpeechDetected,
}

/// Handle returned by [`start_streaming`] to control the streaming thread.
///
/// Dropping the handle sends a stop signal (by disconnecting the channel),
/// but does **not** block until the thread exits. Call [`stop_and_join`]
/// when you need to guarantee the thread has exited before reusing the
/// `Transcriber` (e.g. before starting a final transcription pass).
pub struct StreamingHandle {
    stop_tx: mpsc::Sender<()>,
    abort_flag: Arc<AtomicBool>,
    join_handle: Option<std::thread::JoinHandle<()>>,
}

impl StreamingHandle {
    /// Signal the streaming thread to stop and block until it exits.
    ///
    /// Sets the abort flag first so any in-progress whisper inference
    /// is cancelled immediately, then sends the channel stop signal
    /// and joins the thread.
    pub fn stop_and_join(mut self) {
        self.abort_flag.store(true, Ordering::Relaxed);
        let _ = self.stop_tx.send(());
        if let Some(handle) = self.join_handle.take() {
            if let Err(e) = handle.join() {
                log::error!("Streaming thread panicked: {e:?}");
            }
        }
    }
}

/// Start streaming transcription in a background thread.
///
/// Reads from `sample_buffer` (the AudioRecorder's shared buffer), transcribes
/// overlapping chunks, and sends incremental text via `tx`.
///
/// Returns a [`StreamingHandle`] that can stop and join the thread.
pub fn start_streaming(
    sample_buffer: Arc<Mutex<Vec<f32>>>,
    transcriber: Arc<Transcriber>,
    translate: bool,
    filler_word_removal: bool,
    tx: mpsc::Sender<StreamingEvent>,
) -> StreamingHandle {
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let abort_flag = Arc::new(AtomicBool::new(false));
    let abort_flag_clone = Arc::clone(&abort_flag);

    let join_handle = std::thread::spawn(move || {
        streaming_loop(
            sample_buffer,
            transcriber,
            translate,
            filler_word_removal,
            tx,
            stop_rx,
            abort_flag_clone,
        );
    });

    StreamingHandle {
        stop_tx,
        abort_flag,
        join_handle: Some(join_handle),
    }
}

// ── Internal ───────────────────────────────────────────────────────────

fn streaming_loop(
    sample_buffer: Arc<Mutex<Vec<f32>>>,
    transcriber: Arc<Transcriber>,
    translate: bool,
    filler_word_removal: bool,
    tx: mpsc::Sender<StreamingEvent>,
    stop_rx: mpsc::Receiver<()>,
    abort_flag: Arc<AtomicBool>,
) {
    let min_new_samples = (MIN_NEW_AUDIO_SECS * TARGET_RATE as f32) as usize;
    let max_window_samples = (MAX_WINDOW_SECS * TARGET_RATE as f32) as usize;

    let mut anchor: usize = 0;
    let mut last_transcribed: usize = 0;
    // The exact text currently on screen from streaming emissions.
    let mut emitted_text = String::new();

    // Pre-allocate the window buffer to avoid per-iteration heap allocations.
    let mut window: Vec<f32> = Vec::with_capacity(max_window_samples);

    // Create a single WhisperState up-front and reuse it across
    // iterations to avoid per-call KV-cache allocation overhead.
    let mut whisper_state = match transcriber.create_streaming_state() {
        Ok(s) => s,
        Err(e) => {
            log::error!("Failed to create whisper state for streaming: {e}");
            return;
        }
    };

    loop {
        match stop_rx.try_recv() {
            Ok(()) | Err(mpsc::TryRecvError::Disconnected) => break,
            Err(mpsc::TryRecvError::Empty) => {}
        }

        let total_samples = match sample_buffer.lock() {
            Ok(b) => b.len(),
            Err(e) => e.into_inner().len(),
        };

        if total_samples < last_transcribed + min_new_samples {
            std::thread::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS));
            continue;
        }

        let window_len = total_samples - anchor;
        if window_len > max_window_samples {
            anchor = total_samples - max_window_samples;
        }

        // Reuse the pre-allocated buffer instead of allocating each iteration.
        window.clear();
        {
            let buf = sample_buffer.lock().unwrap_or_else(|e| e.into_inner());
            window.extend_from_slice(&buf[anchor..total_samples]);
        }

        // Only run VAD on audio that arrived since the last transcription,
        // not the full window. This avoids redundant Silero inference on
        // already-checked audio.
        let new_offset = last_transcribed.saturating_sub(anchor);
        if !super::vad::contains_speech(&window[new_offset..]) {
            last_transcribed = total_samples;
            std::thread::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS));
            continue;
        }

        // Notify that speech is detected (keeps silence timeout alive)
        let _ = tx.send(StreamingEvent::SpeechDetected);

        let mut full_text = match transcriber.streaming_transcribe(
            &mut whisper_state,
            &window,
            translate,
            &abort_flag,
        ) {
            Ok(t) => t,
            Err(e) => {
                log::error!("Streaming transcription failed: {e}");
                last_transcribed = total_samples;
                continue;
            }
        };

        // Re-read the current buffer position so the min-new-audio threshold
        // is relative to *now*, not to when this transcription started.
        // Without this, audio that accumulated during a slow transcription
        // immediately triggers another (even larger) transcription, creating
        // a snowball effect that falls further and further behind real-time.
        last_transcribed = match sample_buffer.lock() {
            Ok(b) => b.len(),
            Err(e) => e.into_inner().len(),
        };

        if full_text.is_empty() {
            continue;
        }

        if filler_word_removal {
            full_text = super::postprocess::remove_filler_words(&full_text);
            if full_text.is_empty() {
                continue;
            }
        }

        // Spoken-punctuation regex (17 passes) is deferred to the final
        // transcription pass to keep each streaming iteration fast.
        full_text = super::postprocess::ensure_space_after_punctuation(&full_text);

        // If the transcription changed at all, replace everything.
        // Backspace the entire emitted text and retype Whisper's latest.
        if full_text != emitted_text {
            let replace_chars = emitted_text.chars().count();
            let _ = tx.send(StreamingEvent::PartialText {
                text: full_text.clone(),
                replace_chars,
            });
            emitted_text = full_text;
        }
    }
}

/// Find the byte length of the common prefix between two strings,
/// aligned to char boundaries.
#[allow(dead_code)]
fn common_prefix_len(a: &str, b: &str) -> usize {
    a.chars()
        .zip(b.chars())
        .take_while(|(ac, bc)| ac == bc)
        .map(|(c, _)| c.len_utf8())
        .sum()
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

/// Strip leading/trailing punctuation from a word for comparison purposes.
fn normalize_for_match(word: &str) -> &str {
    word.trim_matches(|c: char| c.is_ascii_punctuation())
}

/// Find the length of the longest suffix of `a` that equals a prefix of `b`,
/// using case-insensitive, punctuation-insensitive comparison.
fn longest_suffix_prefix_match(a: &[String], b: &[String]) -> usize {
    let max_len = a.len().min(b.len());
    let mut best = 0;

    for len in 1..=max_len {
        let suffix = &a[a.len() - len..];
        let prefix = &b[..len];
        if suffix.iter().zip(prefix.iter()).all(|(s, p)| {
            let s_norm = normalize_for_match(s);
            let p_norm = normalize_for_match(p);
            !s_norm.is_empty() && !p_norm.is_empty() && s_norm.eq_ignore_ascii_case(p_norm)
        }) {
            best = len;
        }
    }

    best
}

// ── Utilities ──────────────────────────────────────────────────────────

/// Split text into words, normalising whitespace.
#[allow(dead_code)]
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
    fn test_vad_silence_no_speech() {
        let samples = vec![0.0f32; 16000];
        assert!(!crate::transcription::vad::contains_speech(&samples));
    }

    #[test]
    fn test_vad_low_noise_no_speech() {
        let samples = vec![0.001f32; 16000];
        assert!(!crate::transcription::vad::contains_speech(&samples));
    }

    #[test]
    fn test_vad_empty_no_speech() {
        assert!(!crate::transcription::vad::contains_speech(&[]));
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
        const { assert!(MIN_NEW_AUDIO_SECS > 0.0) };
        const { assert!(MAX_WINDOW_SECS > 0.0) };
        const { assert!(MAX_WINDOW_SECS <= 30.0) };
        const { assert!(POLL_INTERVAL_MS > 0) };
        assert_eq!(TARGET_RATE, 16_000);
    }

    #[test]
    fn test_streaming_event_partial_text() {
        let event = StreamingEvent::PartialText {
            text: "hello".to_string(),
            replace_chars: 3,
        };
        match event {
            StreamingEvent::PartialText {
                text,
                replace_chars,
            } => {
                assert_eq!(text, "hello");
                assert_eq!(replace_chars, 3);
            }
            StreamingEvent::SpeechDetected => panic!("unexpected variant"),
        }
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

    #[test]
    fn test_longest_suffix_prefix_match_punctuation() {
        // "come." should match "come" (trailing punctuation stripped)
        let a: Vec<String> = vec!["it'll".into(), "come.".into()];
        let b: Vec<String> = vec!["come".into(), "pop".into(), "up".into()];
        assert_eq!(longest_suffix_prefix_match(&a, &b), 1);
    }

    #[test]
    fn test_longest_suffix_prefix_match_leading_punctuation() {
        let a: Vec<String> = vec!["hello".into(), "world".into()];
        let b: Vec<String> = vec!["\"world".into(), "today".into()];
        assert_eq!(longest_suffix_prefix_match(&a, &b), 1);
    }

    #[test]
    fn test_stitch_punctuation_mismatch() {
        // Real-world case: chunk N ends with "come." and chunk N+1 starts with "come"
        let committed: Vec<String> = vec![
            "keep".into(),
            "talking".into(),
            "and".into(),
            "eventually".into(),
            "it'll".into(),
            "come.".into(),
        ];
        let chunk: Vec<String> = vec![
            "eventually".into(),
            "it'll".into(),
            "come".into(),
            "pop".into(),
            "up".into(),
        ];
        let result = stitch(&committed, &chunk);
        assert_eq!(result, vec!["pop", "up"]);
    }

    #[test]
    fn test_normalize_for_match() {
        assert_eq!(normalize_for_match("hello"), "hello");
        assert_eq!(normalize_for_match("hello."), "hello");
        assert_eq!(normalize_for_match("hello,"), "hello");
        assert_eq!(normalize_for_match("\"hello\""), "hello");
        assert_eq!(normalize_for_match("..."), "");
    }

    // ── common_prefix_len tests ──────────────────────────────────────

    #[test]
    fn test_common_prefix_len_identical() {
        assert_eq!(common_prefix_len("hello world", "hello world"), 11);
    }

    #[test]
    fn test_common_prefix_len_prefix_match() {
        assert_eq!(common_prefix_len("hello world", "hello world foo"), 11);
    }

    #[test]
    fn test_common_prefix_len_diverges() {
        assert_eq!(common_prefix_len("hello world", "hello earth"), 6);
    }

    #[test]
    fn test_common_prefix_len_empty() {
        assert_eq!(common_prefix_len("", "hello"), 0);
        assert_eq!(common_prefix_len("hello", ""), 0);
    }

    #[test]
    fn test_common_prefix_len_no_common() {
        assert_eq!(common_prefix_len("abc", "xyz"), 0);
    }

    #[test]
    fn test_common_prefix_len_unicode() {
        // "café " = c(1) + a(1) + f(1) + é(2) + space(1) = 6 bytes
        assert_eq!(common_prefix_len("café latte", "café mocha"), 6);
    }

    #[test]
    fn test_common_prefix_len_revision_scenario() {
        // Simulates Whisper revising "Frack" to "Freck"
        let old = "My name is Jacob Frack and I am";
        let new_text = "My name is Jacob Freck and I am a principal";
        // "My name is Jacob F" = 18 bytes, then 'r' vs 'r' matches, 'a' vs 'e' diverges
        // "My name is Jacob Fr" = 19 bytes
        assert_eq!(common_prefix_len(old, new_text), 19);
    }
}
