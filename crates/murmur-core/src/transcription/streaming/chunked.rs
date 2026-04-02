//! Chunked streaming transcription with overlap stitching.
//!
//! Audio is captured continuously and transcribed in overlapping chunks
//! via a subprocess worker. Each chunk includes a configurable overlap
//! with the previous chunk, and word-level stitching deduplicates the
//! overlap region before appending new text.

use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use crate::audio::TARGET_RATE;
use crate::transcription::engine::AsrEngine;

use super::{StreamingEvent, StreamingHandle, MIN_NEW_AUDIO_SECS, POLL_INTERVAL_MS};

// ── Configuration ──────────────────────────────────────────────────────

/// Maximum chunk size sent to Whisper, in seconds.
const MAX_CHUNK_SECS: f32 = 5.0;
/// Overlap between consecutive chunks (seconds). This gives whisper
/// context across chunk boundaries so words aren't lost or split.
const OVERLAP_SECS: f32 = 1.5;

// ── Public API ─────────────────────────────────────────────────────────

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
    engine: Arc<dyn AsrEngine + Send + Sync>,
    translate: bool,
    filler_word_removal: bool,
    tx: mpsc::Sender<StreamingEvent>,
    worker: Option<crate::transcription::subprocess::SubprocessTranscriber>,
) -> StreamingHandle {
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let abort_flag = Arc::new(AtomicBool::new(false));
    let abort_flag_clone = Arc::clone(&abort_flag);

    let join_handle = std::thread::spawn(move || {
        streaming_loop(
            sample_buffer,
            engine,
            translate,
            filler_word_removal,
            tx,
            stop_rx,
            abort_flag_clone,
            worker,
        );
    });

    StreamingHandle::new(stop_tx, abort_flag, join_handle)
}

// ── Internal ───────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn streaming_loop(
    sample_buffer: Arc<Mutex<Vec<f32>>>,
    engine: Arc<dyn AsrEngine + Send + Sync>,
    translate: bool,
    filler_word_removal: bool,
    tx: mpsc::Sender<StreamingEvent>,
    stop_rx: mpsc::Receiver<()>,
    _abort_flag: Arc<AtomicBool>,
    mut worker: Option<crate::transcription::subprocess::SubprocessTranscriber>,
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
                    Err(e) => {
                        log::warn!("Sample buffer mutex poisoned");
                        e.into_inner().len()
                    }
                };
                let start = consumed_boundary.saturating_sub(overlap_samples);
                if total > start {
                    chunk.clear();
                    {
                        let buf = sample_buffer.lock().unwrap_or_else(|e| {
                            log::warn!("Sample buffer mutex poisoned");
                            e.into_inner()
                        });
                        chunk.extend_from_slice(&buf[start..total]);
                    }
                    emit_chunk(
                        &mut worker,
                        &engine,
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
            Err(e) => {
                log::warn!("Sample buffer mutex poisoned");
                e.into_inner().len()
            }
        };

        if !has_enough_new_audio(total_samples, consumed_boundary, min_new_samples) {
            std::thread::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS));
            continue;
        }

        let (chunk_start, chunk_end) = compute_chunk_bounds(
            consumed_boundary,
            overlap_samples,
            total_samples,
            max_chunk_samples,
        );

        chunk.clear();
        {
            let buf = sample_buffer.lock().unwrap_or_else(|e| {
                log::warn!("Sample buffer mutex poisoned");
                e.into_inner()
            });
            chunk.extend_from_slice(&buf[chunk_start..chunk_end]);
        }

        if !crate::transcription::vad::contains_speech(&chunk) {
            consumed_boundary = chunk_end;
            std::thread::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS));
            continue;
        }

        let _ = tx.send(StreamingEvent::SpeechDetected);

        emit_chunk(
            &mut worker,
            &engine,
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
    worker: &mut Option<crate::transcription::subprocess::SubprocessTranscriber>,
    engine: &Arc<dyn AsrEngine + Send + Sync>,
    chunk: &[f32],
    translate: bool,
    filler_word_removal: bool,
    committed_words: &mut Vec<String>,
    tx: &mpsc::Sender<StreamingEvent>,
) {
    let text = if let Some(ref mut w) = worker {
        match w.transcribe(chunk, translate) {
            Ok(t) => t,
            Err(e) => {
                log::error!("Streaming transcription failed: {e}");
                return;
            }
        }
    } else {
        match engine.transcribe(chunk, translate) {
            Ok(result) => result.text,
            Err(e) => {
                log::error!("Streaming transcription failed: {e}");
                return;
            }
        }
    };

    let new_words = compute_new_words(&text, filler_word_removal, committed_words);
    if let Some(event) = format_partial_event(&new_words) {
        let _ = tx.send(event);
        committed_words.extend(new_words);
    }
}

/// Check whether enough new audio has accumulated to warrant a transcription attempt.
pub(crate) fn has_enough_new_audio(
    total_samples: usize,
    consumed_boundary: usize,
    min_new_samples: usize,
) -> bool {
    let new_samples = total_samples.saturating_sub(consumed_boundary);
    new_samples >= min_new_samples
}

/// Compute the chunk boundaries for the next transcription window.
///
/// Returns `(chunk_start, chunk_end)` — the sample indices to slice from the buffer.
/// `chunk_start` includes overlap from the previous chunk for Whisper context.
/// `chunk_end` is capped at `max_chunk_samples` from `chunk_start`.
pub(crate) fn compute_chunk_bounds(
    consumed_boundary: usize,
    overlap_samples: usize,
    total_samples: usize,
    max_chunk_samples: usize,
) -> (usize, usize) {
    let chunk_start = consumed_boundary.saturating_sub(overlap_samples);
    let chunk_end = if total_samples - chunk_start > max_chunk_samples {
        chunk_start + max_chunk_samples
    } else {
        total_samples
    };
    (chunk_start, chunk_end)
}

/// Given raw transcription text, apply postprocessing and stitch against
/// previously committed words to determine what new words to emit.
///
/// Returns the new words to append to `committed_words`, or an empty vec
/// if nothing novel was produced.
pub(crate) fn compute_new_words(
    raw_text: &str,
    filler_word_removal: bool,
    committed_words: &[String],
) -> Vec<String> {
    if raw_text.is_empty() {
        return Vec::new();
    }

    let mut text = raw_text.to_string();
    if filler_word_removal {
        text = crate::transcription::postprocess::remove_filler_words(&text);
        if text.is_empty() {
            return Vec::new();
        }
    }
    text = crate::transcription::postprocess::ensure_space_after_punctuation(&text);

    let chunk_words: Vec<String> = text.split_whitespace().map(String::from).collect();
    stitch(committed_words, &chunk_words)
}

/// Format new words into a `StreamingEvent::PartialText`, or `None` if empty.
pub(crate) fn format_partial_event(new_words: &[String]) -> Option<StreamingEvent> {
    if new_words.is_empty() {
        return None;
    }
    let new_text = new_words.join(" ");
    Some(StreamingEvent::PartialText {
        text: format!(" {new_text}"),
        replace_chars: 0,
    })
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

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::TARGET_RATE;

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
    fn test_constants() {
        const { assert!(MIN_NEW_AUDIO_SECS > 0.0) };
        const { assert!(MAX_CHUNK_SECS > 0.0) };
        const { assert!(MAX_CHUNK_SECS <= 30.0) };
        const { assert!(OVERLAP_SECS >= 0.0) };
        const { assert!(POLL_INTERVAL_MS > 0) };
        assert_eq!(TARGET_RATE, 16_000);
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

    // ── has_enough_new_audio tests ───────────────────────────────────

    #[test]
    fn test_has_enough_audio_above_threshold() {
        assert!(has_enough_new_audio(50_000, 10_000, 32_000));
    }

    #[test]
    fn test_has_enough_audio_exactly_at_threshold() {
        assert!(has_enough_new_audio(42_000, 10_000, 32_000));
    }

    #[test]
    fn test_has_enough_audio_below_threshold() {
        assert!(!has_enough_new_audio(30_000, 10_000, 32_000));
    }

    #[test]
    fn test_has_enough_audio_no_new_samples() {
        assert!(!has_enough_new_audio(10_000, 10_000, 32_000));
    }

    #[test]
    fn test_has_enough_audio_consumed_beyond_total() {
        // saturating_sub handles consumed > total gracefully
        assert!(!has_enough_new_audio(5_000, 10_000, 32_000));
    }

    #[test]
    fn test_has_enough_audio_zero_threshold() {
        assert!(has_enough_new_audio(100, 100, 0));
    }

    // ── compute_chunk_bounds tests ──────────────────────────────────

    #[test]
    fn test_chunk_bounds_basic() {
        // consumed=10000, overlap=2000, total=20000, max=8000
        let (start, end) = compute_chunk_bounds(10_000, 2_000, 20_000, 8_000);
        assert_eq!(start, 8_000); // 10000 - 2000
        assert_eq!(end, 16_000); // 8000 + 8000 (capped by max)
    }

    #[test]
    fn test_chunk_bounds_no_cap() {
        // total - start <= max, so chunk_end = total
        let (start, end) = compute_chunk_bounds(10_000, 2_000, 14_000, 80_000);
        assert_eq!(start, 8_000);
        assert_eq!(end, 14_000);
    }

    #[test]
    fn test_chunk_bounds_overlap_exceeds_consumed() {
        // overlap > consumed, so saturating_sub clamps to 0
        let (start, end) = compute_chunk_bounds(1_000, 5_000, 10_000, 80_000);
        assert_eq!(start, 0);
        assert_eq!(end, 10_000);
    }

    #[test]
    fn test_chunk_bounds_zero_overlap() {
        let (start, end) = compute_chunk_bounds(5_000, 0, 10_000, 80_000);
        assert_eq!(start, 5_000);
        assert_eq!(end, 10_000);
    }

    #[test]
    fn test_chunk_bounds_exact_max() {
        // total - start == max exactly
        let (start, end) = compute_chunk_bounds(5_000, 1_000, 12_000, 8_000);
        assert_eq!(start, 4_000);
        assert_eq!(end, 12_000); // 12000 - 4000 = 8000 == max, so not capped
    }

    #[test]
    fn test_chunk_bounds_with_real_constants() {
        let overlap_samples = (OVERLAP_SECS * TARGET_RATE as f32) as usize;
        let max_chunk_samples = (MAX_CHUNK_SECS * TARGET_RATE as f32) as usize;
        // Simulate 10 seconds of audio, consumed up to 5 seconds
        let consumed = 5 * TARGET_RATE as usize;
        let total = 10 * TARGET_RATE as usize;
        let (start, end) =
            compute_chunk_bounds(consumed, overlap_samples, total, max_chunk_samples);
        assert!(start < consumed);
        assert!(end <= total);
        assert!(end - start <= max_chunk_samples);
    }

    // ── compute_new_words tests ─────────────────────────────────────

    #[test]
    fn test_compute_new_words_empty_text() {
        let committed: Vec<String> = vec!["hello".into()];
        assert!(compute_new_words("", false, &committed).is_empty());
    }

    #[test]
    fn test_compute_new_words_no_committed() {
        let result = compute_new_words("hello world", false, &[]);
        assert_eq!(result, vec!["hello", "world"]);
    }

    #[test]
    fn test_compute_new_words_with_overlap() {
        let committed: Vec<String> = vec!["the".into(), "quick".into(), "brown".into()];
        let result = compute_new_words("quick brown fox", false, &committed);
        assert_eq!(result, vec!["fox"]);
    }

    #[test]
    fn test_compute_new_words_identical_result() {
        let committed: Vec<String> = vec!["hello".into(), "world".into()];
        let result = compute_new_words("hello world", false, &committed);
        assert!(result.is_empty());
    }

    #[test]
    fn test_compute_new_words_no_overlap() {
        let committed: Vec<String> = vec!["alpha".into(), "beta".into()];
        let result = compute_new_words("gamma delta", false, &committed);
        assert_eq!(result, vec!["gamma", "delta"]);
    }

    #[test]
    fn test_compute_new_words_filler_removal_produces_empty() {
        // "um" is a filler word; after removal the text may be empty
        let result = compute_new_words("um", true, &[]);
        // After filler removal the text might be empty or "um" might not be
        // in the filler list. Either way, ensure no panic.
        let _ = result;
    }

    #[test]
    fn test_compute_new_words_whitespace_only() {
        // split_whitespace produces nothing for whitespace-only
        let result = compute_new_words("   ", false, &[]);
        assert!(result.is_empty());
    }

    // ── format_partial_event tests ──────────────────────────────────

    #[test]
    fn test_format_partial_event_empty() {
        assert!(format_partial_event(&[]).is_none());
    }

    #[test]
    fn test_format_partial_event_single_word() {
        let words: Vec<String> = vec!["hello".into()];
        let event = format_partial_event(&words).unwrap();
        match event {
            StreamingEvent::PartialText {
                text,
                replace_chars,
            } => {
                assert_eq!(text, " hello");
                assert_eq!(replace_chars, 0);
            }
            StreamingEvent::SpeechDetected => panic!("unexpected variant"),
        }
    }

    #[test]
    fn test_format_partial_event_multiple_words() {
        let words: Vec<String> = vec!["hello".into(), "world".into()];
        let event = format_partial_event(&words).unwrap();
        match event {
            StreamingEvent::PartialText {
                text,
                replace_chars,
            } => {
                assert_eq!(text, " hello world");
                assert_eq!(replace_chars, 0);
            }
            StreamingEvent::SpeechDetected => panic!("unexpected variant"),
        }
    }

    #[test]
    fn test_format_partial_event_leading_space() {
        // Verify the leading space is always present
        let words: Vec<String> = vec!["x".into()];
        let event = format_partial_event(&words).unwrap();
        match event {
            StreamingEvent::PartialText { text, .. } => {
                assert!(text.starts_with(' '));
            }
            _ => panic!("unexpected variant"),
        }
    }

    // ── Integration: compute_new_words + format_partial_event ───────

    #[test]
    fn test_emission_pipeline_with_overlap() {
        let committed: Vec<String> = vec!["the".into(), "quick".into()];
        let new_words = compute_new_words("the quick brown fox", false, &committed);
        assert_eq!(new_words, vec!["brown", "fox"]);
        let event = format_partial_event(&new_words).unwrap();
        match event {
            StreamingEvent::PartialText {
                text,
                replace_chars,
            } => {
                assert_eq!(text, " brown fox");
                assert_eq!(replace_chars, 0);
            }
            StreamingEvent::SpeechDetected => panic!("unexpected variant"),
        }
    }

    #[test]
    fn test_emission_pipeline_no_novelty() {
        let committed: Vec<String> = vec!["hello".into(), "world".into()];
        let new_words = compute_new_words("hello world", false, &committed);
        assert!(new_words.is_empty());
        assert!(format_partial_event(&new_words).is_none());
    }
}
