//! Growing-window streaming transcription.
//!
//! While the user holds the hotkey, audio is recorded continuously.
//! This module periodically transcribes a growing window of audio
//! (from the start up to the current position), diffs against what
//! was already emitted, and sends only new words incrementally.
//!
//! Once the window exceeds MAX_WINDOW_SECS, the start anchor slides
//! forward to keep Whisper's input under its 30s native limit.

use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use crate::audio::TARGET_RATE;
use crate::transcriber::Transcriber;

// ── Configuration ──────────────────────────────────────────────────────

/// Minimum new audio (seconds) before re-transcribing.
const MIN_NEW_AUDIO_SECS: f32 = 1.0;
/// Minimum interval between transcription attempts, in milliseconds.
const POLL_INTERVAL_MS: u64 = 300;
/// Maximum window size sent to Whisper, in seconds.
/// Whisper natively handles up to 30s; leave headroom.
const MAX_WINDOW_SECS: f32 = 28.0;
/// RMS threshold below which audio is considered silence.
const SILENCE_RMS_THRESHOLD: f32 = 0.005;

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
    translate: bool,
    spoken_punctuation: bool,
    tx: mpsc::Sender<StreamingEvent>,
) -> mpsc::Sender<()> {
    let (stop_tx, stop_rx) = mpsc::channel::<()>();

    std::thread::spawn(move || {
        streaming_loop(
            sample_buffer,
            transcriber,
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
    translate: bool,
    spoken_punctuation: bool,
    tx: mpsc::Sender<StreamingEvent>,
    stop_rx: mpsc::Receiver<()>,
) {
    let min_new_samples = (MIN_NEW_AUDIO_SECS * TARGET_RATE as f32) as usize;
    let max_window_samples = (MAX_WINDOW_SECS * TARGET_RATE as f32) as usize;

    // Anchor: start of the window in the sample buffer.
    // Slides forward only when the window would exceed MAX_WINDOW_SECS.
    let mut anchor: usize = 0;
    // Number of samples that have been transcribed (relative to buffer start).
    let mut last_transcribed: usize = 0;
    // Words already emitted to the user.
    let mut emitted_words: Vec<String> = Vec::new();

    loop {
        match stop_rx.try_recv() {
            Ok(()) | Err(mpsc::TryRecvError::Disconnected) => break,
            Err(mpsc::TryRecvError::Empty) => {}
        }

        let total_samples = sample_buffer
            .lock()
            .map(|b| b.len())
            .unwrap_or(0);

        // Wait until enough new audio has accumulated.
        if total_samples < last_transcribed + min_new_samples {
            std::thread::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS));
            continue;
        }

        // Slide the anchor forward if the window is too large.
        let window_len = total_samples - anchor;
        if window_len > max_window_samples {
            anchor = total_samples - max_window_samples;
        }

        // Extract the window from anchor to current end.
        let window: Vec<f32> = {
            let buf = sample_buffer.lock().unwrap();
            buf[anchor..total_samples].to_vec()
        };

        if is_silent(&window) {
            last_transcribed = total_samples;
            std::thread::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS));
            continue;
        }

        // Transcribe the entire window.
        let full_text = match transcriber.transcribe_samples(&window, translate) {
            Ok(t) => t,
            Err(e) => {
                log::error!("Streaming transcription failed: {e}");
                last_transcribed = total_samples;
                continue;
            }
        };

        last_transcribed = total_samples;

        if full_text.is_empty() {
            continue;
        }

        let all_words = split_words(&full_text);

        // Diff: find words beyond what we've already emitted.
        let new_start = find_emit_boundary(&emitted_words, &all_words);

        if new_start < all_words.len() {
            // Hold back the last word — it's unstable because Whisper may
            // revise it as more audio arrives (e.g. adding punctuation,
            // changing word boundaries). Only emit confirmed words that
            // have at least one word after them.
            let confirmed_end = all_words.len() - 1;

            if new_start < confirmed_end {
                let new_words: Vec<String> = all_words[new_start..confirmed_end].to_vec();

                let new_text = if spoken_punctuation {
                    crate::postprocess::process(&new_words.join(" "))
                } else {
                    new_words.join(" ")
                };

                if !new_text.is_empty() {
                    // Prepend space for continuation (text is appended at cursor)
                    let spaced = if emitted_words.is_empty() {
                        new_text
                    } else {
                        format!(" {new_text}")
                    };
                    // Update emitted to include confirmed words (but not the held-back last word)
                    emitted_words = all_words[..confirmed_end].to_vec();
                    let _ = tx.send(StreamingEvent::PartialText(spaced));
                }
            }
        }
    }
}

/// Find the index in `all_words` where new (un-emitted) content starts.
///
/// Compares emitted words against the beginning of all_words using
/// punctuation-insensitive matching to handle Whisper's nondeterministic
/// punctuation. Returns the index of the first new word.
fn find_emit_boundary(emitted: &[String], all_words: &[String]) -> usize {
    if emitted.is_empty() {
        return 0;
    }

    // Walk through emitted and all_words in parallel, matching with
    // punctuation tolerance. If they diverge, the emitted prefix has
    // been revised by Whisper — anchor from the best match point.
    let mut matched = 0;
    for (e, a) in emitted.iter().zip(all_words.iter()) {
        if normalize_for_match(e).eq_ignore_ascii_case(normalize_for_match(a)) {
            matched += 1;
        } else {
            break;
        }
    }

    // If we matched all emitted words, new content starts after them.
    // If we matched fewer (Whisper revised earlier words), be conservative
    // and don't re-emit — just skip to after the emitted count to avoid
    // duplication. The revised words are already on screen.
    emitted.len().min(all_words.len()).max(matched)
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
        const { assert!(MIN_NEW_AUDIO_SECS > 0.0) };
        const { assert!(MAX_WINDOW_SECS > 0.0) };
        const { assert!(MAX_WINDOW_SECS <= 30.0) };
        const { assert!(POLL_INTERVAL_MS > 0) };
        const { assert!(SILENCE_RMS_THRESHOLD > 0.0) };
        assert_eq!(TARGET_RATE, 16_000);
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
            "keep".into(), "talking".into(), "and".into(),
            "eventually".into(), "it'll".into(), "come.".into(),
        ];
        let chunk: Vec<String> = vec![
            "eventually".into(), "it'll".into(), "come".into(),
            "pop".into(), "up".into(),
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

    // ── find_emit_boundary tests ──────────────────────────────────────

    #[test]
    fn test_find_emit_boundary_empty_emitted() {
        let all: Vec<String> = vec!["hello".into(), "world".into()];
        assert_eq!(find_emit_boundary(&[], &all), 0);
    }

    #[test]
    fn test_find_emit_boundary_exact_prefix() {
        let emitted: Vec<String> = vec!["hello".into(), "world".into()];
        let all: Vec<String> = vec!["hello".into(), "world".into(), "foo".into()];
        assert_eq!(find_emit_boundary(&emitted, &all), 2);
    }

    #[test]
    fn test_find_emit_boundary_no_new_words() {
        let emitted: Vec<String> = vec!["hello".into(), "world".into()];
        let all: Vec<String> = vec!["hello".into(), "world".into()];
        assert_eq!(find_emit_boundary(&emitted, &all), 2);
    }

    #[test]
    fn test_find_emit_boundary_punctuation_tolerance() {
        let emitted: Vec<String> = vec!["hello".into(), "world.".into()];
        let all: Vec<String> = vec!["hello".into(), "world".into(), "foo".into()];
        assert_eq!(find_emit_boundary(&emitted, &all), 2);
    }

    #[test]
    fn test_find_emit_boundary_whisper_revision() {
        // Whisper revised an earlier word — don't re-emit
        let emitted: Vec<String> = vec!["hello".into(), "world".into()];
        let all: Vec<String> = vec!["hello".into(), "word".into(), "foo".into()];
        // Diverges at index 1, but emitted has 2 words — skip to 2
        assert_eq!(find_emit_boundary(&emitted, &all), 2);
    }

    #[test]
    fn test_find_emit_boundary_growing_window() {
        // Simulate growing window: first call emits 3 words, second call has 5
        let emitted: Vec<String> = vec!["the".into(), "quick".into(), "brown".into()];
        let all: Vec<String> = vec![
            "the".into(), "quick".into(), "brown".into(),
            "fox".into(), "jumps".into(),
        ];
        assert_eq!(find_emit_boundary(&emitted, &all), 3);
    }
}
