//! Integration tests exercising the full AppState message→effect pipeline.
//!
//! These tests simulate the same message processing loop as `run()`,
//! exercising real state machine code paths without requiring hardware
//! (microphone, display, model files).

use murmur::app::{AppEffect, AppMessage, AppState};
use murmur::config::{Config, InputMode};
use murmur::ui::tray::TrayState;

// ── Test Harness ──────────────────────────────────────────────────────

/// Simulates the message→effect pipeline from `run()`, collecting effects
/// for assertion. Processes effects via VecDeque like the real loop.
struct Harness {
    state: AppState,
}

impl Harness {
    fn new() -> Self {
        Self {
            state: AppState::new(&Config::default()),
        }
    }

    fn with_config(config: &Config) -> Self {
        Self {
            state: AppState::new(config),
        }
    }

    /// Send a message and return the effects produced.
    fn send(&mut self, msg: AppMessage) -> Vec<AppEffect> {
        self.state.handle_message(&msg)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────

fn has_start_recording(effects: &[AppEffect]) -> bool {
    effects
        .iter()
        .any(|e| matches!(e, AppEffect::StartRecording(_)))
}

fn has_stop_and_transcribe(effects: &[AppEffect]) -> bool {
    effects
        .iter()
        .any(|e| matches!(e, AppEffect::StopAndTranscribe))
}

fn has_insert_text(effects: &[AppEffect], expected: &str) -> bool {
    effects
        .iter()
        .any(|e| matches!(e, AppEffect::InsertText(t) if t == expected))
}

fn has_tray_state(effects: &[AppEffect], expected: TrayState) -> bool {
    effects
        .iter()
        .any(|e| matches!(e, AppEffect::SetTrayState(s) if *s == expected))
}

fn has_save_config(effects: &[AppEffect]) -> bool {
    effects.iter().any(|e| matches!(e, AppEffect::SaveConfig))
}

fn has_reload_transcriber(effects: &[AppEffect]) -> bool {
    effects
        .iter()
        .any(|e| matches!(e, AppEffect::ReloadTranscriber(_)))
}

fn has_no_insert_text(effects: &[AppEffect]) -> bool {
    !effects
        .iter()
        .any(|e| matches!(e, AppEffect::InsertText(_)))
}

// ═══════════════════════════════════════════════════════════════════════
//  Push-to-Talk scenarios
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn push_to_talk_full_cycle() {
    let mut h = Harness::new();

    // Press hotkey → start recording
    let fx = h.send(AppMessage::KeyDown);
    assert!(has_start_recording(&fx));
    assert!(has_tray_state(&fx, TrayState::Recording));
    assert!(h.state.is_pressed);

    // Release hotkey → stop and transcribe
    let fx = h.send(AppMessage::KeyUp);
    assert!(has_stop_and_transcribe(&fx));
    assert!(has_tray_state(&fx, TrayState::Transcribing));
    assert!(!h.state.is_pressed);

    // Transcription completes → insert text
    let fx = h.send(AppMessage::TranscriptionDone("hello world".into()));
    assert!(has_insert_text(&fx, "hello world"));
    assert!(has_tray_state(&fx, TrayState::Idle));
    assert_eq!(h.state.last_transcription, Some("hello world".to_string()));
}

#[test]
fn push_to_talk_empty_transcription_returns_to_idle() {
    let mut h = Harness::new();

    h.send(AppMessage::KeyDown);
    h.send(AppMessage::KeyUp);

    let fx = h.send(AppMessage::TranscriptionDone(String::new()));
    assert!(has_no_insert_text(&fx));
    assert!(has_tray_state(&fx, TrayState::Idle));
    assert!(h.state.last_transcription.is_none());
}

#[test]
fn push_to_talk_error_recovers() {
    let mut h = Harness::new();

    h.send(AppMessage::KeyDown);
    h.send(AppMessage::KeyUp);

    // Transcription fails (not recording, so error state applies)
    let fx = h.send(AppMessage::TranscriptionError("model failed".into()));
    assert!(has_tray_state(&fx, TrayState::Error));
    assert!(!h.state.is_pressed);

    // Next cycle should work normally
    let fx = h.send(AppMessage::KeyDown);
    assert!(has_start_recording(&fx));
    assert!(h.state.is_pressed);

    let fx = h.send(AppMessage::KeyUp);
    assert!(has_stop_and_transcribe(&fx));
}

#[test]
fn push_to_talk_key_repeat_ignored() {
    let mut h = Harness::new();

    // Press hotkey
    let fx = h.send(AppMessage::KeyDown);
    assert!(has_start_recording(&fx));

    // Key repeat events (modifier keys do this)
    for _ in 0..5 {
        let fx = h.send(AppMessage::KeyDown);
        assert_eq!(fx, vec![AppEffect::None], "key repeat should be ignored");
    }

    // Release should still work
    let fx = h.send(AppMessage::KeyUp);
    assert!(has_stop_and_transcribe(&fx));

    // Verify transcription completes
    let fx = h.send(AppMessage::TranscriptionDone("after repeats".into()));
    assert!(has_insert_text(&fx, "after repeats"));
}

#[test]
fn push_to_talk_release_without_press_ignored() {
    let mut h = Harness::new();

    // Release without prior press → no-op
    let fx = h.send(AppMessage::KeyUp);
    assert_eq!(fx, vec![AppEffect::None]);
    assert!(!h.state.is_pressed);
}

#[test]
fn push_to_talk_multiple_cycles() {
    let mut h = Harness::new();

    for i in 0..3 {
        let text = format!("cycle {i}");

        h.send(AppMessage::KeyDown);
        assert!(h.state.is_pressed);

        h.send(AppMessage::KeyUp);
        assert!(!h.state.is_pressed);

        let fx = h.send(AppMessage::TranscriptionDone(text.clone()));
        assert!(has_insert_text(&fx, &text));
        assert!(has_tray_state(&fx, TrayState::Idle));
    }

    assert_eq!(h.state.last_transcription, Some("cycle 2".to_string()));
}

#[test]
fn push_to_talk_with_spoken_punctuation() {
    let mut h = Harness::new();
    h.state.spoken_punctuation = true;

    h.send(AppMessage::KeyDown);
    h.send(AppMessage::KeyUp);

    // In the real app, apply_effect processes punctuation before sending
    // TranscriptionDone. So the message contains already-processed text.
    let fx = h.send(AppMessage::TranscriptionDone("hello.".into()));
    assert!(has_insert_text(&fx, "hello."));
}

// ═══════════════════════════════════════════════════════════════════════
//  Open Mic scenarios
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn open_mic_toggle_cycle() {
    let mut h = Harness::new();
    h.state.mode = InputMode::OpenMic;

    // First press → start recording
    let fx = h.send(AppMessage::KeyDown);
    assert!(has_start_recording(&fx));
    assert!(has_tray_state(&fx, TrayState::Recording));
    assert!(h.state.is_pressed);

    // KeyUp is ignored in open mic
    let fx = h.send(AppMessage::KeyUp);
    assert_eq!(fx, vec![AppEffect::None]);
    assert!(h.state.is_pressed, "OpenMic should stay pressed on KeyUp");

    // Second press → stop and transcribe
    let fx = h.send(AppMessage::KeyDown);
    assert!(has_stop_and_transcribe(&fx));
    assert!(has_tray_state(&fx, TrayState::Transcribing));
    assert!(!h.state.is_pressed);

    // Transcription completes
    let fx = h.send(AppMessage::TranscriptionDone("open mic test".into()));
    assert!(has_insert_text(&fx, "open mic test"));
}

#[test]
fn open_mic_multiple_toggle_cycles() {
    let mut h = Harness::new();
    h.state.mode = InputMode::OpenMic;

    for i in 0..3 {
        // Start
        let fx = h.send(AppMessage::KeyDown);
        assert!(has_start_recording(&fx));

        // Stop
        let fx = h.send(AppMessage::KeyDown);
        assert!(has_stop_and_transcribe(&fx));

        // Complete
        h.send(AppMessage::TranscriptionDone(format!("cycle {i}")));
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Streaming scenarios
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn streaming_push_to_talk_starts_and_stops_streaming() {
    let mut h = Harness::new();
    h.state.streaming = true;

    // Press → starts recording AND streaming
    let fx = h.send(AppMessage::KeyDown);
    assert!(has_start_recording(&fx));
    assert!(
        fx.iter().any(|e| matches!(e, AppEffect::StartStreaming)),
        "expected StartStreaming"
    );
    assert!(h.state.streaming_active);

    // Release → stops streaming AND transcribes
    let fx = h.send(AppMessage::KeyUp);
    assert!(
        fx.iter().any(|e| matches!(e, AppEffect::StopStreaming)),
        "expected StopStreaming"
    );
    assert!(has_stop_and_transcribe(&fx));
}

#[test]
fn streaming_suppresses_final_transcription() {
    let mut h = Harness::new();
    h.state.streaming = true;

    h.send(AppMessage::KeyDown);
    assert!(h.state.streaming_active);

    // Partial text during streaming produces a replace effect
    let fx = h.send(AppMessage::StreamingPartialText {
        text: "hello ".to_string(),
        replace_chars: 0,
    });
    assert!(fx.iter().any(|e| matches!(
        e, AppEffect::StreamingReplace { text, .. } if text == "hello "
    )));

    h.send(AppMessage::KeyUp);

    // Final transcription is suppressed (streaming already inserted text)
    let fx = h.send(AppMessage::TranscriptionDone("hello world".into()));
    assert!(
        has_no_insert_text(&fx),
        "final transcription should be suppressed when streaming was active"
    );
    // But last_transcription is still saved for copy-last
    assert_eq!(h.state.last_transcription, Some("hello world".to_string()));
    assert!(!h.state.streaming_active);
}

#[test]
fn streaming_partial_text_empty_ignored() {
    let mut h = Harness::new();

    let fx = h.send(AppMessage::StreamingPartialText {
        text: String::new(),
        replace_chars: 0,
    });
    assert_eq!(fx, vec![AppEffect::None]);
}

// ═══════════════════════════════════════════════════════════════════════
//  Tray menu actions
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn set_model_triggers_reload() {
    let mut h = Harness::new();

    let fx = h.send(AppMessage::TraySetModel("small.en".into()));
    assert_eq!(h.state.model_size, "small.en");
    assert!(has_save_config(&fx));
    assert!(has_reload_transcriber(&fx));
    assert_eq!(h.state.reload_generation, 1);
}

#[test]
fn set_language_triggers_reload() {
    let config = Config {
        model_size: "base".to_string(),
        ..Config::default()
    };
    let mut h = Harness::with_config(&config);

    let fx = h.send(AppMessage::TraySetLanguage("fr".into()));
    assert_eq!(h.state.language, "fr");
    assert!(has_save_config(&fx));
    assert!(has_reload_transcriber(&fx));
    assert_eq!(h.state.reload_generation, 1);
}

#[test]
fn reload_generation_increments_monotonically() {
    let config = Config {
        model_size: "base".to_string(),
        ..Config::default()
    };
    let mut h = Harness::with_config(&config);

    h.send(AppMessage::TraySetModel("small".into()));
    assert_eq!(h.state.reload_generation, 1);

    h.send(AppMessage::TraySetLanguage("fr".into()));
    assert_eq!(h.state.reload_generation, 2);

    h.send(AppMessage::TraySetModel("tiny".into()));
    assert_eq!(h.state.reload_generation, 3);
}

#[test]
fn toggle_spoken_punctuation() {
    let mut h = Harness::new();
    assert!(!h.state.spoken_punctuation);

    let fx = h.send(AppMessage::TrayToggleSpokenPunctuation);
    assert!(h.state.spoken_punctuation);
    assert!(has_save_config(&fx));

    h.send(AppMessage::TrayToggleSpokenPunctuation);
    assert!(!h.state.spoken_punctuation);
}

#[test]
fn toggle_streaming() {
    let mut h = Harness::new();
    // Default backend (Qwen3Asr) auto-enables streaming.
    assert!(h.state.streaming);

    h.send(AppMessage::TrayToggleStreaming);
    assert!(!h.state.streaming);

    h.send(AppMessage::TrayToggleStreaming);
    assert!(h.state.streaming);
}

#[test]
fn set_mode_push_to_talk_to_open_mic() {
    let mut h = Harness::new();
    assert_eq!(h.state.mode, InputMode::PushToTalk);

    let fx = h.send(AppMessage::TraySetMode(InputMode::OpenMic));
    assert_eq!(h.state.mode, InputMode::OpenMic);
    assert!(has_save_config(&fx));
}

#[test]
fn toggle_translate() {
    let mut h = Harness::new();
    assert!(!h.state.translate_to_english);

    h.send(AppMessage::TrayToggleTranslate);
    assert!(h.state.translate_to_english);
}

#[test]
fn copy_last_with_transcription() {
    let mut h = Harness::new();

    // Complete a transcription first
    h.send(AppMessage::KeyDown);
    h.send(AppMessage::KeyUp);
    h.send(AppMessage::TranscriptionDone("copy me".into()));

    let fx = h.send(AppMessage::TrayCopyLast);
    assert!(
        fx.iter()
            .any(|e| matches!(e, AppEffect::CopyToClipboard(t) if t == "copy me")),
        "expected CopyToClipboard"
    );
}

#[test]
fn copy_last_without_transcription() {
    let mut h = Harness::new();

    let fx = h.send(AppMessage::TrayCopyLast);
    assert_eq!(fx, vec![AppEffect::None]);
}

#[test]
fn quit_produces_quit_effect() {
    let mut h = Harness::new();

    let fx = h.send(AppMessage::TrayQuit);
    assert_eq!(fx, vec![AppEffect::Quit]);
}

// ═══════════════════════════════════════════════════════════════════════
//  Mode switching edge cases
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn switch_mode_while_idle() {
    let mut h = Harness::new();

    // Record in PushToTalk
    h.send(AppMessage::KeyDown);
    h.send(AppMessage::KeyUp);
    h.send(AppMessage::TranscriptionDone("ptt".into()));

    // Switch to OpenMic
    h.send(AppMessage::TraySetMode(InputMode::OpenMic));

    // Record in OpenMic (toggle)
    h.send(AppMessage::KeyDown); // start
    let fx = h.send(AppMessage::KeyDown); // stop
    assert!(has_stop_and_transcribe(&fx));
}

#[test]
fn error_during_recording_does_not_reset_pressed() {
    let mut h = Harness::new();

    h.send(AppMessage::KeyDown);
    assert!(h.state.is_pressed);

    // Error from a concurrent transcription should not interfere with active recording
    let fx = h.send(AppMessage::TranscriptionError("mic failed".into()));
    assert!(
        h.state.is_pressed,
        "error during active recording should not reset is_pressed"
    );
    // Error tray state is suppressed while recording
    assert!(!has_tray_state(&fx, TrayState::Error));

    // Release still works
    let fx = h.send(AppMessage::KeyUp);
    assert!(has_stop_and_transcribe(&fx));
}

#[test]
fn error_when_idle_sets_error_state() {
    let mut h = Harness::new();

    h.send(AppMessage::KeyDown);
    h.send(AppMessage::KeyUp);

    // Error arrives when not recording
    let fx = h.send(AppMessage::TranscriptionError("model failed".into()));
    assert!(has_tray_state(&fx, TrayState::Error));

    // Next recording works
    let fx = h.send(AppMessage::KeyDown);
    assert!(has_start_recording(&fx));
}

// ═══════════════════════════════════════════════════════════════════════
//  Config & transcriber reload edge cases
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn to_config_round_trips_state() {
    let base = Config::default();
    let mut h = Harness::with_config(&base);

    h.send(AppMessage::TraySetModel("large".into()));
    h.send(AppMessage::TraySetLanguage("de".into()));
    h.send(AppMessage::TrayToggleSpokenPunctuation);
    h.send(AppMessage::TraySetMode(InputMode::OpenMic));
    // Default backend auto-enables streaming, so toggle off then on.
    h.send(AppMessage::TrayToggleStreaming); // true → false
    h.send(AppMessage::TrayToggleStreaming); // false → true
    h.send(AppMessage::TrayToggleTranslate);

    let cfg = h.state.to_config(&base);
    assert_eq!(cfg.model_size, "large");
    assert_eq!(cfg.language, "de");
    assert!(cfg.spoken_punctuation);
    assert_eq!(cfg.mode, InputMode::OpenMic);
    assert!(cfg.streaming);
    assert!(cfg.translate_to_english);
    assert_eq!(cfg.hotkey, base.hotkey, "hotkey should come from base");
}

#[test]
fn from_tray_action_all_variants() {
    use murmur::ui::tray::TrayAction;

    type ActionCheck = (TrayAction, fn(&AppMessage) -> bool);
    let cases: Vec<ActionCheck> = vec![
        (TrayAction::Quit, |m| matches!(m, AppMessage::TrayQuit)),
        (TrayAction::CopyLastDictation, |m| {
            matches!(m, AppMessage::TrayCopyLast)
        }),
        (
            TrayAction::SetModel("base.en".into()),
            |m| matches!(m, AppMessage::TraySetModel(s) if s == "base.en"),
        ),
        (
            TrayAction::SetLanguage("fr".into()),
            |m| matches!(m, AppMessage::TraySetLanguage(c) if c == "fr"),
        ),
        (TrayAction::ToggleSpokenPunctuation, |m| {
            matches!(m, AppMessage::TrayToggleSpokenPunctuation)
        }),
        (
            TrayAction::SetMode(InputMode::OpenMic),
            |m| matches!(m, AppMessage::TraySetMode(mode) if *mode == InputMode::OpenMic),
        ),
        (TrayAction::ToggleStreaming, |m| {
            matches!(m, AppMessage::TrayToggleStreaming)
        }),
        (TrayAction::ToggleTranslate, |m| {
            matches!(m, AppMessage::TrayToggleTranslate)
        }),
        (TrayAction::OpenConfig, |m| {
            matches!(m, AppMessage::TrayOpenConfig)
        }),
        (TrayAction::ReloadConfig, |m| {
            matches!(m, AppMessage::TrayReloadConfig)
        }),
        (TrayAction::SetHotkey, |m| {
            matches!(m, AppMessage::TraySetHotkey)
        }),
    ];

    for (action, check) in cases {
        let msg: AppMessage = action.into();
        assert!(check(&msg));
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Regression tests for specific bugs
// ═══════════════════════════════════════════════════════════════════════

/// Regression: key repeat on modifier hotkeys (e.g. RightAlt) must not
/// interrupt an active recording.
#[test]
fn regression_modifier_key_repeat_does_not_stop_recording() {
    let mut h = Harness::new();

    // Start recording
    let fx = h.send(AppMessage::KeyDown);
    assert!(has_start_recording(&fx));
    assert!(h.state.is_pressed);

    // Simulate 10 key repeats (common with modifier keys)
    for _ in 0..10 {
        let fx = h.send(AppMessage::KeyDown);
        assert_eq!(
            fx,
            vec![AppEffect::None],
            "key repeat must not produce effects"
        );
        assert!(
            h.state.is_pressed,
            "is_pressed must stay true during key repeat"
        );
    }

    // Release → should transcribe the full recording
    let fx = h.send(AppMessage::KeyUp);
    assert!(has_stop_and_transcribe(&fx));
}

/// Regression: streaming_active must be cleared after TranscriptionDone
/// so the next non-streaming recording inserts text normally.
#[test]
fn regression_streaming_flag_cleared_after_transcription() {
    let mut h = Harness::new();
    h.state.streaming = true;

    // Streaming recording
    h.send(AppMessage::KeyDown);
    assert!(h.state.streaming_active);
    h.send(AppMessage::KeyUp);
    h.send(AppMessage::TranscriptionDone("streamed".into()));
    assert!(!h.state.streaming_active, "should be cleared");

    // Disable streaming
    h.send(AppMessage::TrayToggleStreaming);
    assert!(!h.state.streaming);

    // Next recording should insert text (not suppressed)
    h.send(AppMessage::KeyDown);
    assert!(!h.state.streaming_active);
    h.send(AppMessage::KeyUp);
    let fx = h.send(AppMessage::TranscriptionDone("not streamed".into()));
    assert!(
        has_insert_text(&fx, "not streamed"),
        "non-streaming transcription must be inserted"
    );
}

/// Regression: error from a previous recording must not interfere with current recording.
#[test]
fn regression_error_recovery_full_cycle() {
    let mut h = Harness::new();

    // First attempt: record, release, error arrives later
    h.send(AppMessage::KeyDown);
    h.send(AppMessage::KeyUp);
    h.send(AppMessage::TranscriptionError("mic busy".into()));
    assert!(!h.state.is_pressed);

    // Second attempt: succeeds
    let fx = h.send(AppMessage::KeyDown);
    assert!(has_start_recording(&fx));
    assert!(h.state.is_pressed);

    let fx = h.send(AppMessage::KeyUp);
    assert!(has_stop_and_transcribe(&fx));

    let fx = h.send(AppMessage::TranscriptionDone("recovered".into()));
    assert!(has_insert_text(&fx, "recovered"));
}

/// Regression: stale TranscriptionDone/Error from a previous recording must not
/// reset is_pressed or tray state when a new recording is active.
#[test]
fn regression_stale_transcription_result_during_recording() {
    let mut h = Harness::new();

    // Recording 1: start and stop
    h.send(AppMessage::KeyDown);
    h.send(AppMessage::KeyUp);
    // Transcription thread is running...

    // Recording 2: start immediately
    h.send(AppMessage::KeyDown);
    assert!(h.state.is_pressed);

    // Stale TranscriptionDone from recording 1 arrives during recording 2
    let fx = h.send(AppMessage::TranscriptionDone("recording one".into()));
    assert!(
        h.state.is_pressed,
        "stale TranscriptionDone must not reset is_pressed"
    );
    assert!(
        has_insert_text(&fx, "recording one"),
        "text from recording 1 should still be inserted"
    );
    assert!(
        !has_tray_state(&fx, TrayState::Idle),
        "tray should not go to Idle while recording"
    );

    // Recording 2 completes normally
    let fx = h.send(AppMessage::KeyUp);
    assert!(has_stop_and_transcribe(&fx));
    let fx = h.send(AppMessage::TranscriptionDone("recording two".into()));
    assert!(has_insert_text(&fx, "recording two"));
    assert!(has_tray_state(&fx, TrayState::Idle));
}

/// Regression: stale TranscriptionError during active recording must not disrupt it.
#[test]
fn regression_stale_error_during_recording() {
    let mut h = Harness::new();

    // Recording 1: start, stop
    h.send(AppMessage::KeyDown);
    h.send(AppMessage::KeyUp);

    // Recording 2: start
    h.send(AppMessage::KeyDown);
    assert!(h.state.is_pressed);

    // Stale error from recording 1
    h.send(AppMessage::TranscriptionError("thread panic".into()));
    assert!(
        h.state.is_pressed,
        "stale error must not reset is_pressed during active recording"
    );

    // Recording 2 completes normally
    let fx = h.send(AppMessage::KeyUp);
    assert!(has_stop_and_transcribe(&fx));
    let fx = h.send(AppMessage::TranscriptionDone("still works".into()));
    assert!(has_insert_text(&fx, "still works"));
}

/// Ensure spoken punctuation is applied by the effect handler (StopAndTranscribe),
/// not the state machine. This test verifies postprocess::process works correctly.
#[test]
fn regression_spoken_punctuation_processes_all_types() {
    use murmur::transcription::postprocess;

    let cases = [
        ("hello period", "."),
        ("what question mark", "?"),
        ("wow exclamation mark", "!"),
        ("note colon details", ":"),
        ("a comma b", ","),
    ];

    for (input, expected_punct) in cases {
        let result = postprocess::process(input);
        assert!(
            result.contains(expected_punct),
            "process('{input}') = '{result}', expected to contain '{expected_punct}'"
        );
    }
}

/// Multiple rapid model/language changes should produce monotonically
/// increasing generation counters.
#[test]
fn regression_rapid_model_changes_generation_counter() {
    let config = Config {
        model_size: "base".to_string(),
        ..Config::default()
    };
    let mut h = Harness::with_config(&config);

    let changes = [
        AppMessage::TraySetModel("tiny".into()),
        AppMessage::TraySetLanguage("fr".into()),
        AppMessage::TraySetModel("small".into()),
        AppMessage::TraySetLanguage("de".into()),
        AppMessage::TraySetModel("large".into()),
    ];

    let mut last_gen = 0u64;
    for msg in changes {
        h.send(msg);
        assert!(
            h.state.reload_generation > last_gen,
            "generation must increase monotonically"
        );
        last_gen = h.state.reload_generation;
    }
    assert_eq!(last_gen, 5);
}

// ═══════════════════════════════════════════════════════════════════════
//  Hotkey capture scenarios
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn hotkey_capture_full_cycle() {
    let mut h = Harness::new();

    // Enter capture mode
    let fx = h.send(AppMessage::TraySetHotkey);
    assert!(h.state.capturing_hotkey);
    assert!(fx
        .iter()
        .any(|e| matches!(e, AppEffect::EnterHotkeyCaptureMode)));

    // Key events are ignored during capture
    let fx = h.send(AppMessage::KeyDown);
    assert_eq!(fx, vec![AppEffect::None]);

    // Capture a key
    let fx = h.send(AppMessage::HotkeyCapture(rdev::Key::F5));
    assert!(!h.state.capturing_hotkey);
    assert!(fx
        .iter()
        .any(|e| matches!(e, AppEffect::SetHotkey(k) if k == "f5")));
    assert!(fx.iter().any(|e| matches!(e, AppEffect::SaveConfig)));

    // Normal operation resumes
    let fx = h.send(AppMessage::KeyDown);
    assert!(has_start_recording(&fx));
}

// ═══════════════════════════════════════════════════════════════════════
//  Error during active recording
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn error_during_recording_does_not_reset_tray() {
    let mut h = Harness::new();

    // Start recording
    h.send(AppMessage::KeyDown);
    assert!(h.state.is_pressed);

    // Error arrives from a previous transcription cycle
    let fx = h.send(AppMessage::TranscriptionError("stale error".into()));
    // Tray should NOT go to Error state since we're recording
    assert!(!has_tray_state(&fx, TrayState::Error));
    // But error should still be logged
    assert!(fx.iter().any(|e| matches!(e, AppEffect::LogError(_))));
    // Recording state should be preserved
    assert!(h.state.is_pressed);
}

#[test]
fn transcription_result_during_recording_does_not_reset_tray() {
    let mut h = Harness::new();

    // Start recording
    h.send(AppMessage::KeyDown);
    assert!(h.state.is_pressed);

    // Result from a previous cycle arrives
    let fx = h.send(AppMessage::TranscriptionDone("stale result".into()));
    // Tray should NOT reset to Idle — we're still recording
    assert!(!has_tray_state(&fx, TrayState::Idle));
    // But result should still be inserted and saved
    assert!(has_insert_text(&fx, "stale result"));
    assert_eq!(h.state.last_transcription, Some("stale result".to_string()));
}

// ═══════════════════════════════════════════════════════════════════════
//  Config round-trip via to_config
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn to_config_reflects_tray_mutations() {
    let base = Config {
        model_size: "base".to_string(),
        ..Config::default()
    };
    let mut h = Harness::with_config(&base);

    // Mutate state through messages
    h.send(AppMessage::TraySetModel("small".into()));
    h.send(AppMessage::TraySetLanguage("de".into()));
    h.send(AppMessage::TrayToggleSpokenPunctuation);
    h.send(AppMessage::TraySetMode(InputMode::OpenMic));
    // Default backend auto-enables streaming, so toggle off then on.
    h.send(AppMessage::TrayToggleStreaming); // true → false
    h.send(AppMessage::TrayToggleStreaming); // false → true
    h.send(AppMessage::TrayToggleTranslate);

    // Build config from state
    let cfg = h.state.to_config(&base);
    assert_eq!(cfg.model_size, "small");
    assert_eq!(cfg.language, "de");
    assert!(cfg.spoken_punctuation);
    assert_eq!(cfg.mode, InputMode::OpenMic);
    assert!(cfg.streaming);
    assert!(cfg.translate_to_english);
    // Hotkey and max_recordings come from base, not state
    assert_eq!(cfg.hotkey, base.hotkey);
    assert_eq!(cfg.max_recordings, base.max_recordings);
}

// ═══════════════════════════════════════════════════════════════════════
//  Streaming full cycle
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn streaming_push_to_talk_full_cycle() {
    let config = Config {
        streaming: true,
        ..Config::default()
    };
    let mut h = Harness::with_config(&config);

    // Press: start recording + streaming
    let fx = h.send(AppMessage::KeyDown);
    assert!(has_start_recording(&fx));
    assert!(fx.iter().any(|e| matches!(e, AppEffect::StartStreaming)));
    assert!(h.state.streaming_active);

    // Partial text arrives
    let fx = h.send(AppMessage::StreamingPartialText {
        text: "hel".into(),
        replace_chars: 0,
    });
    assert!(fx.iter().any(|e| matches!(
        e, AppEffect::StreamingReplace { text, replace_chars }
        if text == "hel" && *replace_chars == 0
    )));

    // Revised partial text
    let fx = h.send(AppMessage::StreamingPartialText {
        text: "hello".into(),
        replace_chars: 3,
    });
    assert!(fx.iter().any(|e| matches!(
        e, AppEffect::StreamingReplace { text, replace_chars }
        if text == "hello" && *replace_chars == 3
    )));

    // Release: stop recording + streaming
    let fx = h.send(AppMessage::KeyUp);
    assert!(has_stop_and_transcribe(&fx));
    assert!(fx.iter().any(|e| matches!(e, AppEffect::StopStreaming)));

    // Final transcription — should NOT re-insert since streaming was active
    let fx = h.send(AppMessage::TranscriptionDone("hello world".into()));
    assert!(has_no_insert_text(&fx));
    assert_eq!(h.state.last_transcription, Some("hello world".to_string()));
}

// ═══════════════════════════════════════════════════════════════════════
//  Multiple rapid cycles
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn multiple_push_to_talk_cycles() {
    let mut h = Harness::new();

    for i in 0..5 {
        let fx = h.send(AppMessage::KeyDown);
        assert!(has_start_recording(&fx));

        let fx = h.send(AppMessage::KeyUp);
        assert!(has_stop_and_transcribe(&fx));

        let text = format!("cycle {i}");
        let fx = h.send(AppMessage::TranscriptionDone(text.clone()));
        assert!(has_insert_text(&fx, &text));
        assert!(has_tray_state(&fx, TrayState::Idle));
        assert_eq!(h.state.last_transcription, Some(text));
    }
}

#[test]
fn copy_last_after_multiple_transcriptions() {
    let mut h = Harness::new();

    // First cycle
    h.send(AppMessage::KeyDown);
    h.send(AppMessage::KeyUp);
    h.send(AppMessage::TranscriptionDone("first".into()));

    // Second cycle
    h.send(AppMessage::KeyDown);
    h.send(AppMessage::KeyUp);
    h.send(AppMessage::TranscriptionDone("second".into()));

    // Copy last should give the most recent
    let fx = h.send(AppMessage::TrayCopyLast);
    assert!(fx
        .iter()
        .any(|e| matches!(e, AppEffect::CopyToClipboard(t) if t == "second")));
}
