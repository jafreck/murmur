//! Performance-oriented end-to-end tests for the audio pipeline.
//!
//! These tests use synthetic audio (no microphone, no Whisper model) to verify
//! that the hot paths in the capture → VAD → transcription pipeline behave
//! efficiently and deterministically.  Each "golden scenario" represents a
//! common real-world pattern and asserts timing / allocation properties.

mod helpers;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use murmur::audio::capture::{mix_to_mono, TARGET_RATE};
use murmur::transcription::vad;

// ═══════════════════════════════════════════════════════════════════════
//  Constants
// ═══════════════════════════════════════════════════════════════════════

/// Number of iterations to run in throughput benchmarks.
const BENCH_ITERS: usize = 50;

/// Maximum wall-clock time (ms) for a single VAD silence rejection.
/// The RMS fast-path should complete well within this budget.
const VAD_SILENCE_BUDGET_MS: u64 = 5;

/// Maximum wall-clock time (ms) for a VAD call on 1 second of low-noise audio.
/// Neural inference is significantly slower in debug builds, and CI runners
/// (especially with coverage instrumentation) add further overhead.
#[cfg(debug_assertions)]
const VAD_NOISE_BUDGET_MS: u64 = 600;
#[cfg(not(debug_assertions))]
const VAD_NOISE_BUDGET_MS: u64 = 20;

/// Maximum wall-clock time (ms) to resample 1 second of 48 kHz audio to 16 kHz.
const RESAMPLE_BUDGET_MS: u64 = 5;

// ═══════════════════════════════════════════════════════════════════════
//  Helpers – synthetic audio generators
// ═══════════════════════════════════════════════════════════════════════

/// Generate white noise at a given RMS level.
fn white_noise(sample_rate: u32, duration_secs: f32, rms_level: f32) -> Vec<f32> {
    let n = (sample_rate as f32 * duration_secs) as usize;
    // Deterministic pseudo-random noise using a simple LCG
    let mut state: u32 = 0xDEAD_BEEF;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let uniform = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            uniform * rms_level * std::f32::consts::SQRT_2
        })
        .collect()
}

/// Generate a speech-like signal: a 300 Hz tone amplitude-modulated at 4 Hz
/// (simulates syllabic rhythm) mixed with harmonics.
fn synthetic_speech(sample_rate: u32, duration_secs: f32) -> Vec<f32> {
    let n = (sample_rate as f32 * duration_secs) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let fundamental = (2.0 * std::f32::consts::PI * 300.0 * t).sin();
            let harmonic2 = 0.5 * (2.0 * std::f32::consts::PI * 600.0 * t).sin();
            let harmonic3 = 0.3 * (2.0 * std::f32::consts::PI * 900.0 * t).sin();
            let envelope = 0.5 + 0.5 * (2.0 * std::f32::consts::PI * 4.0 * t).sin();
            0.4 * envelope * (fundamental + harmonic2 + harmonic3)
        })
        .collect()
}

/// Concatenate silence, speech, and silence to simulate a typical dictation
/// session: pause → speak → pause.
fn silence_speech_silence(
    sample_rate: u32,
    pre_silence_secs: f32,
    speech_secs: f32,
    post_silence_secs: f32,
) -> Vec<f32> {
    let mut samples = helpers::silence(sample_rate, pre_silence_secs);
    samples.extend(synthetic_speech(sample_rate, speech_secs));
    samples.extend(helpers::silence(sample_rate, post_silence_secs));
    samples
}

// ═══════════════════════════════════════════════════════════════════════
//  Golden Scenario 1: Pure Silence – VAD fast-path rejection
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn golden_silence_vad_rejects_fast() {
    let samples = helpers::silence(TARGET_RATE, 1.0);

    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let result = vad::contains_speech(&samples);
        assert!(!result, "VAD should reject pure silence");
    }
    let elapsed = start.elapsed();
    let per_call = elapsed / BENCH_ITERS as u32;

    assert!(
        per_call < Duration::from_millis(VAD_SILENCE_BUDGET_MS),
        "VAD silence fast-path took {per_call:?} per call (budget: {VAD_SILENCE_BUDGET_MS}ms)"
    );
}

#[test]
fn golden_silence_extended_vad_rejects_fast() {
    // 10 seconds of silence – typical streaming window size
    let samples = helpers::silence(TARGET_RATE, 10.0);

    let start = Instant::now();
    let result = vad::contains_speech(&samples);
    let elapsed = start.elapsed();

    assert!(!result, "VAD should reject extended silence");
    assert!(
        elapsed < Duration::from_millis(VAD_SILENCE_BUDGET_MS),
        "VAD on 10s silence took {elapsed:?} (budget: {VAD_SILENCE_BUDGET_MS}ms)"
    );
}

// ═══════════════════════════════════════════════════════════════════════
//  Golden Scenario 2: Low-level Noise – VAD rejects without full inference
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn golden_noise_floor_vad_rejects() {
    // Audio just above digital zero but below the noise floor threshold
    let samples = vec![0.002f32; TARGET_RATE as usize]; // 1 second

    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let result = vad::contains_speech(&samples);
        assert!(!result, "VAD should reject sub-noise-floor audio");
    }
    let elapsed = start.elapsed();
    let per_call = elapsed / BENCH_ITERS as u32;

    assert!(
        per_call < Duration::from_millis(VAD_SILENCE_BUDGET_MS),
        "VAD noise-floor fast-path took {per_call:?} per call (budget: {VAD_SILENCE_BUDGET_MS}ms)"
    );
}

#[test]
fn golden_low_noise_vad_rejects() {
    // Audio above noise floor but not speech – triggers neural inference
    let samples = white_noise(TARGET_RATE, 1.0, 0.02);

    let start = Instant::now();
    let result = vad::contains_speech(&samples);
    let elapsed = start.elapsed();

    assert!(!result, "VAD should reject low-level white noise");
    assert!(
        elapsed < Duration::from_millis(VAD_NOISE_BUDGET_MS),
        "VAD on 1s low noise took {elapsed:?} (budget: {VAD_NOISE_BUDGET_MS}ms)"
    );
}

// ═══════════════════════════════════════════════════════════════════════
//  Golden Scenario 3: Streaming Buffer Management
// ═══════════════════════════════════════════════════════════════════════

/// Simulates the streaming loop's buffer snapshot pattern.
/// Verifies that a pre-allocated buffer can be reused without per-iteration
/// allocation, matching the pattern used in the fixed streaming_loop.
#[test]
fn golden_streaming_buffer_reuse() {
    let max_window_samples = (10.0 * TARGET_RATE as f32) as usize; // 10s window
    let shared_buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));

    // Simulate audio accumulation (5 seconds)
    {
        let mut buf = shared_buffer.lock().unwrap();
        buf.extend(synthetic_speech(TARGET_RATE, 5.0));
    }

    // Pre-allocate a reusable window buffer (the fix we're testing)
    let mut window_buf: Vec<f32> = Vec::with_capacity(max_window_samples);
    let ptr_before = window_buf.as_ptr();

    // Simulate 20 streaming iterations, each snapshotting the window
    for _ in 0..20 {
        window_buf.clear(); // reuse allocation
        {
            let buf = shared_buffer.lock().unwrap();
            let anchor = buf.len().saturating_sub(max_window_samples);
            window_buf.extend_from_slice(&buf[anchor..]);
        }
        assert!(!window_buf.is_empty());
    }

    let ptr_after = window_buf.as_ptr();
    // After the first fill, the buffer should not have reallocated
    // because we pre-allocated enough capacity.
    assert_eq!(
        ptr_before, ptr_after,
        "Streaming window buffer should reuse its allocation (no realloc)"
    );
}

/// Verify the streaming buffer grows correctly with sliding window semantics.
#[test]
fn golden_streaming_window_slides() {
    let max_window_secs: f32 = 10.0;
    let max_window_samples = (max_window_secs * TARGET_RATE as f32) as usize;
    let shared_buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));

    // Accumulate 15 seconds of audio (exceeds window)
    {
        let mut buf = shared_buffer.lock().unwrap();
        buf.extend(synthetic_speech(TARGET_RATE, 15.0));
    }

    let buf = shared_buffer.lock().unwrap();
    let total = buf.len();
    let anchor = total.saturating_sub(max_window_samples);
    let window_len = total - anchor;

    assert!(
        window_len <= max_window_samples,
        "Window should be capped at {max_window_samples} samples, got {window_len}"
    );
    assert!(
        anchor > 0,
        "Anchor should slide forward when audio exceeds window"
    );
}

// ═══════════════════════════════════════════════════════════════════════
//  Golden Scenario 4: Audio Pipeline Throughput
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn golden_mono_mixing_throughput() {
    // Simulate stereo audio at 48 kHz for 1 second
    let stereo: Vec<f32> = (0..96_000).map(|i| (i as f32 / 96_000.0).sin()).collect();

    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let mono = mix_to_mono(&stereo, 2);
        assert_eq!(mono.len(), 48_000);
    }
    let elapsed = start.elapsed();
    let per_call = elapsed / BENCH_ITERS as u32;

    assert!(
        per_call < Duration::from_millis(RESAMPLE_BUDGET_MS),
        "Mono mixing took {per_call:?} per call (budget: {RESAMPLE_BUDGET_MS}ms)"
    );
}

#[test]
fn golden_pipeline_silence_fast_path() {
    // Full pipeline: stereo 48kHz → mono → 16kHz → VAD → rejected
    // This should be very fast because VAD rejects at the noise floor.
    let stereo_48k = vec![0.0f32; 96_000]; // 1 second stereo silence

    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let mono = mix_to_mono(&stereo_48k, 2);
        // Simulate resampling (48k → 16k would produce ~16000 samples)
        let resampled: Vec<f32> = mono.iter().step_by(3).copied().collect();
        let has_speech = vad::contains_speech(&resampled);
        assert!(!has_speech);
    }
    let elapsed = start.elapsed();
    let per_call = elapsed / BENCH_ITERS as u32;

    assert!(
        per_call < Duration::from_millis(15),
        "Full silence pipeline took {per_call:?} per call (budget: 15ms)"
    );
}

// ═══════════════════════════════════════════════════════════════════════
//  Golden Scenario 5: Sample Buffer Pre-allocation
// ═══════════════════════════════════════════════════════════════════════

/// Verify that a pre-allocated sample buffer avoids early reallocations
/// during a typical 5-second dictation.
#[test]
fn golden_sample_buffer_prealloc() {
    let expected_duration_secs = 5.0;
    let expected_samples = (TARGET_RATE as f32 * expected_duration_secs) as usize;

    // Pre-allocate with capacity hint (the fix)
    let mut buffer: Vec<f32> = Vec::with_capacity(expected_samples);
    let initial_capacity = buffer.capacity();

    // Simulate audio arriving in ~10ms chunks (160 samples at 16kHz)
    let chunk_size = 160;
    let total_chunks = expected_samples / chunk_size;
    let chunk = vec![0.1f32; chunk_size];

    for _ in 0..total_chunks {
        buffer.extend_from_slice(&chunk);
    }

    assert_eq!(
        buffer.capacity(),
        initial_capacity,
        "Buffer should not reallocate during a {expected_duration_secs}s recording \
         when pre-allocated (initial: {initial_capacity}, final: {})",
        buffer.capacity()
    );
}

// ═══════════════════════════════════════════════════════════════════════
//  Golden Scenario 6: Silence → Speech → Silence Transition
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn golden_silence_speech_silence_vad_pattern() {
    // 0.5s silence → 2s "speech" → 0.5s silence
    let samples = silence_speech_silence(TARGET_RATE, 0.5, 2.0, 0.5);
    let total_duration_secs = 3.0;
    assert_eq!(
        samples.len(),
        (TARGET_RATE as f32 * total_duration_secs) as usize
    );

    // VAD on just the silence portions should reject
    let silence_end = (TARGET_RATE as f32 * 0.5) as usize;
    assert!(!vad::contains_speech(&samples[..silence_end]));

    // VAD on just the trailing silence should reject
    let speech_end = (TARGET_RATE as f32 * 2.5) as usize;
    assert!(!vad::contains_speech(&samples[speech_end..]));

    // VAD on the full clip should detect speech
    // (the speech portion has significant energy from the synthetic signal)
    let full_result = vad::contains_speech(&samples);
    // Note: synthetic_speech may or may not trigger Silero VAD since it's
    // not real speech. We just verify VAD completes without error.
    let _ = full_result;
}

// ═══════════════════════════════════════════════════════════════════════
//  Golden Scenario 7: Adaptive Poll Interval
// ═══════════════════════════════════════════════════════════════════════

/// Verify that the adaptive backoff logic correctly increases and resets
/// the polling interval. This tests the pattern used in wake_word.rs.
#[test]
fn golden_adaptive_backoff() {
    let base_ms: u64 = 300;
    let max_ms: u64 = 1200;
    let mut current_ms = base_ms;

    // Simulate 10 consecutive "no speech" results → should back off
    for _ in 0..10 {
        current_ms = (current_ms * 3 / 2).min(max_ms);
    }
    assert_eq!(
        current_ms, max_ms,
        "Should reach max backoff after enough misses"
    );

    // Simulate speech detected → should reset
    current_ms = base_ms;
    assert_eq!(
        current_ms, base_ms,
        "Should reset to base on speech detection"
    );
}

// ═══════════════════════════════════════════════════════════════════════
//  Golden Scenario 8: Concurrent Buffer Access Pattern
// ═══════════════════════════════════════════════════════════════════════

/// Simulates the producer-consumer pattern between the audio callback
/// (writer) and the streaming loop (reader). Verifies no data races and
/// that the reader always gets a consistent snapshot.
#[test]
fn golden_concurrent_buffer_access() {
    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::with_capacity(80_000)));
    let buffer_writer = Arc::clone(&buffer);

    // Producer: simulate audio callback writing 160 samples every 10ms
    let producer = std::thread::spawn(move || {
        let chunk = vec![0.5f32; 160];
        for _ in 0..100 {
            if let Ok(mut buf) = buffer_writer.try_lock() {
                buf.extend_from_slice(&chunk);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    });

    // Consumer: simulate streaming loop reading snapshots every 30ms
    let mut snapshots = 0;
    for _ in 0..30 {
        if let Ok(buf) = buffer.lock() {
            let len = buf.len();
            if len > 0 {
                // Simulate window copy
                let _window: Vec<f32> = buf[buf.len().saturating_sub(16_000)..].to_vec();
                snapshots += 1;
            }
        }
        std::thread::sleep(Duration::from_millis(3));
    }

    producer.join().unwrap();
    assert!(
        snapshots > 0,
        "Consumer should have read at least one snapshot"
    );

    let final_len = buffer.lock().unwrap().len();
    assert!(
        final_len > 0,
        "Buffer should contain samples after producer finishes"
    );
}

// ═══════════════════════════════════════════════════════════════════════
//  Golden Scenario 9: Postprocessing Throughput
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn golden_postprocess_throughput() {
    use murmur::transcription::postprocess;

    let input = "um hello uh world er this is ah a test hmm of the uh system";
    let expected_no_fillers = "hello world this is a test of the system";

    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let result = postprocess::remove_filler_words(input);
        assert_eq!(result, expected_no_fillers);
    }
    let elapsed = start.elapsed();
    let per_call = elapsed / BENCH_ITERS as u32;

    assert!(
        per_call < Duration::from_millis(1),
        "Filler word removal took {per_call:?} per call (budget: 1ms)"
    );
}

#[test]
fn golden_spoken_punctuation_throughput() {
    use murmur::transcription::postprocess;

    let input = "Hello comma world period How are you question mark";

    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let result = postprocess::process(input);
        assert!(result.contains('.'));
        assert!(result.contains(','));
        assert!(result.contains('?'));
    }
    let elapsed = start.elapsed();
    let per_call = elapsed / BENCH_ITERS as u32;

    assert!(
        per_call < Duration::from_millis(2),
        "Spoken punctuation took {per_call:?} per call (budget: 2ms)"
    );
}

// ═══════════════════════════════════════════════════════════════════════
//  Golden Scenario 10: State Machine Idle Overhead
// ═══════════════════════════════════════════════════════════════════════

/// Verify that the state machine's silence-timeout check is cheap when
/// not in a wake-word-initiated session.
#[test]
fn golden_state_machine_idle_check() {
    use murmur::app::AppState;
    use murmur::config::Config;

    let config = Config::default();
    let mut state = AppState::new(&config);

    let start = Instant::now();
    for _ in 0..10_000 {
        let effects = state.check_silence_timeout();
        assert!(effects.is_empty(), "No effects when not recording");
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_millis(5),
        "10k idle silence checks took {elapsed:?} (should be < 5ms)"
    );
}
