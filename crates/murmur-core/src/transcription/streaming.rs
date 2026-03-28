//! Chunked streaming transcription with overlap stitching.
//!
//! Audio is captured continuously and transcribed in overlapping chunks
//! via a subprocess worker. Each chunk includes a configurable overlap
//! with the previous chunk, and word-level stitching deduplicates the
//! overlap region before appending new text.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use super::transcriber::Transcriber;
use crate::audio::capture::TARGET_RATE;

// ── Configuration ──────────────────────────────────────────────────────

/// Minimum new audio (seconds) before re-transcribing.
const MIN_NEW_AUDIO_SECS: f32 = 2.0;
/// Minimum interval between transcription attempts, in milliseconds.
const POLL_INTERVAL_MS: u64 = 300;
/// Maximum chunk size sent to Whisper, in seconds.
const MAX_CHUNK_SECS: f32 = 5.0;
/// Overlap between consecutive chunks (seconds). This gives whisper
/// context across chunk boundaries so words aren't lost or split.
const OVERLAP_SECS: f32 = 1.5;

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
/// `worker` is a pre-spawned subprocess transcriber. Spawning it ahead of
/// time avoids the model-loading delay on first recording.
///
/// Returns a [`StreamingHandle`] that can stop and join the thread.
pub fn start_streaming(
    sample_buffer: Arc<Mutex<Vec<f32>>>,
    transcriber: Arc<Transcriber>,
    translate: bool,
    filler_word_removal: bool,
    tx: mpsc::Sender<StreamingEvent>,
    worker: super::subprocess::SubprocessTranscriber,
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
            worker,
        );
    });

    StreamingHandle {
        stop_tx,
        abort_flag,
        join_handle: Some(join_handle),
    }
}

// ── Internal ───────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn streaming_loop(
    sample_buffer: Arc<Mutex<Vec<f32>>>,
    _transcriber: Arc<Transcriber>,
    translate: bool,
    filler_word_removal: bool,
    tx: mpsc::Sender<StreamingEvent>,
    stop_rx: mpsc::Receiver<()>,
    _abort_flag: Arc<AtomicBool>,
    mut worker: super::subprocess::SubprocessTranscriber,
) {
    let min_new_samples = (MIN_NEW_AUDIO_SECS * TARGET_RATE as f32) as usize;
    let max_chunk_samples = (MAX_CHUNK_SECS * TARGET_RATE as f32) as usize;
    let overlap_samples = (OVERLAP_SECS * TARGET_RATE as f32) as usize;

    // The boundary up to which audio has been consumed (minus overlap).
    let mut consumed_boundary: usize = 0;
    // Words emitted so far (for stitching overlap regions).
    let mut committed_words: Vec<String> = Vec::new();
    let mut chunk: Vec<f32> = Vec::with_capacity(max_chunk_samples);

    loop {
        match stop_rx.try_recv() {
            Ok(()) | Err(mpsc::TryRecvError::Disconnected) => {
                // Final chunk: transcribe remaining audio with overlap.
                let total = match sample_buffer.lock() {
                    Ok(b) => b.len(),
                    Err(e) => e.into_inner().len(),
                };
                let start = consumed_boundary.saturating_sub(overlap_samples);
                if total > start {
                    chunk.clear();
                    {
                        let buf = sample_buffer.lock().unwrap_or_else(|e| e.into_inner());
                        chunk.extend_from_slice(&buf[start..total]);
                    }
                    emit_chunk(
                        &mut worker,
                        &chunk,
                        translate,
                        filler_word_removal,
                        &mut committed_words,
                        &tx,
                    );
                }
                break;
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }

        let total_samples = match sample_buffer.lock() {
            Ok(b) => b.len(),
            Err(e) => e.into_inner().len(),
        };

        let new_samples = total_samples.saturating_sub(consumed_boundary);
        if new_samples < min_new_samples {
            std::thread::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS));
            continue;
        }

        // Start the chunk with overlap from the previous chunk for context.
        let chunk_start = consumed_boundary.saturating_sub(overlap_samples);
        let chunk_end = if total_samples - chunk_start > max_chunk_samples {
            chunk_start + max_chunk_samples
        } else {
            total_samples
        };

        chunk.clear();
        {
            let buf = sample_buffer.lock().unwrap_or_else(|e| e.into_inner());
            chunk.extend_from_slice(&buf[chunk_start..chunk_end]);
        }

        if !super::vad::contains_speech(&chunk) {
            consumed_boundary = chunk_end;
            std::thread::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS));
            continue;
        }

        let _ = tx.send(StreamingEvent::SpeechDetected);

        emit_chunk(
            &mut worker,
            &chunk,
            translate,
            filler_word_removal,
            &mut committed_words,
            &tx,
        );

        // Advance boundary (new audio will overlap by overlap_samples).
        consumed_boundary = match sample_buffer.lock() {
            Ok(b) => b.len(),
            Err(e) => e.into_inner().len(),
        };
    }
}

/// Transcribe a chunk and emit only the new (stitched) words.
fn emit_chunk(
    worker: &mut super::subprocess::SubprocessTranscriber,
    chunk: &[f32],
    translate: bool,
    filler_word_removal: bool,
    committed_words: &mut Vec<String>,
    tx: &mpsc::Sender<StreamingEvent>,
) {
    let mut text = match worker.transcribe(chunk, translate) {
        Ok(t) => t,
        Err(e) => {
            log::error!("Streaming transcription failed: {e}");
            return;
        }
    };

    if text.is_empty() {
        return;
    }

    if filler_word_removal {
        text = super::postprocess::remove_filler_words(&text);
        if text.is_empty() {
            return;
        }
    }
    text = super::postprocess::ensure_space_after_punctuation(&text);

    let chunk_words: Vec<String> = text.split_whitespace().map(String::from).collect();

    // Stitch: deduplicate the overlap region against previously committed words.
    let new_words = stitch(committed_words, &chunk_words);

    if !new_words.is_empty() {
        let new_text = new_words.join(" ");
        let _ = tx.send(StreamingEvent::PartialText {
            text: format!(" {new_text}"),
            replace_chars: 0,
        });
        committed_words.extend(new_words);
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
        const { assert!(MAX_CHUNK_SECS > 0.0) };
        const { assert!(MAX_CHUNK_SECS <= 30.0) };
        const { assert!(OVERLAP_SECS >= 0.0) };
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
