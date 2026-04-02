//! Integration tests for the effect layer and cross-module interactions.
//!
//! Since `apply_effect` requires a `TrayController` (which needs a display),
//! these tests focus on the parts of the effect pipeline that can be tested
//! without hardware: config save/reload effects via the state machine,
//! model validation, and the AppState→Effect→Config flow.

mod helpers;

use murmur::app::{AppEffect, AppMessage, AppState};
use murmur::config::{Config, InputMode};
use murmur::transcription::model;
use murmur::transcription::streaming;
use murmur::ui::tray::TrayState;

// ── Test Harness ────────────────────────────────────────────────────────

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

    fn send(&mut self, msg: AppMessage) -> Vec<AppEffect> {
        self.state.handle_message(msg)
    }
}

fn has_effect(effects: &[AppEffect], pred: impl Fn(&AppEffect) -> bool) -> bool {
    effects.iter().any(pred)
}

// ═══════════════════════════════════════════════════════════════════════
//  Config mutation → SaveConfig effect → file round-trip
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn set_model_emits_save_and_reload() {
    let mut h = Harness::new();

    let fx = h.send(AppMessage::TraySetModel("small.en".into()));

    assert!(has_effect(&fx, |e| matches!(e, AppEffect::SaveConfig)));
    assert!(has_effect(&fx, |e| matches!(
        e,
        AppEffect::ReloadTranscriber(_)
    )));
    assert!(has_effect(&fx, |e| matches!(
        e,
        AppEffect::SetTrayModel(m) if m == "small.en"
    )));
    assert_eq!(h.state.model_size(), "small.en");
}

#[test]
fn set_language_emits_save_and_reload() {
    let mut config = Config::default();
    config.set_model_size("base".to_string()); // multilingual model allows language changes
    let mut h = Harness::with_config(&config);

    let fx = h.send(AppMessage::TraySetLanguage("fr".into()));

    assert!(has_effect(&fx, |e| matches!(e, AppEffect::SaveConfig)));
    assert!(has_effect(&fx, |e| matches!(
        e,
        AppEffect::ReloadTranscriber(_)
    )));
    assert_eq!(h.state.language(), "fr");
}

#[test]
fn toggle_spoken_punctuation_emits_save() {
    let mut h = Harness::new();
    assert!(!h.state.spoken_punctuation());

    let fx = h.send(AppMessage::TrayToggleSpokenPunctuation);
    assert!(has_effect(&fx, |e| matches!(e, AppEffect::SaveConfig)));
    assert!(h.state.spoken_punctuation());

    let fx = h.send(AppMessage::TrayToggleSpokenPunctuation);
    assert!(has_effect(&fx, |e| matches!(e, AppEffect::SaveConfig)));
    assert!(!h.state.spoken_punctuation());
}

#[test]
fn toggle_streaming_emits_save() {
    let mut h = Harness::new();
    let initial = h.state.streaming();

    let fx = h.send(AppMessage::TrayToggleStreaming);
    assert!(has_effect(&fx, |e| matches!(e, AppEffect::SaveConfig)));
    assert_ne!(h.state.streaming(), initial);
}

#[test]
fn toggle_translate_emits_save() {
    let mut h = Harness::new();
    let initial = h.state.translate_to_english();

    let fx = h.send(AppMessage::TrayToggleTranslate);
    assert!(has_effect(&fx, |e| matches!(e, AppEffect::SaveConfig)));
    assert_ne!(h.state.translate_to_english(), initial);
}

#[test]
fn set_mode_emits_save() {
    let mut h = Harness::new();

    let fx = h.send(AppMessage::TraySetMode(InputMode::OpenMic));
    assert!(has_effect(&fx, |e| matches!(e, AppEffect::SaveConfig)));
    assert_eq!(*h.state.mode(), InputMode::OpenMic);
}

// ═══════════════════════════════════════════════════════════════════════
//  Config state saved to disk matches state machine
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn state_mutations_produce_correct_config_on_disk() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("config.json");
    let base = Config::default();
    base.save_to(&path).unwrap();

    let mut h = Harness::with_config(&base);

    h.send(AppMessage::TraySetModel("medium".into())); // multilingual model
    h.send(AppMessage::TraySetLanguage("ja".into()));
    h.send(AppMessage::TrayToggleSpokenPunctuation);
    // Default backend (Qwen3Asr) auto-enables streaming, so toggle off then on.
    h.send(AppMessage::TrayToggleStreaming); // true → false
    h.send(AppMessage::TrayToggleStreaming); // false → true
    h.send(AppMessage::TraySetMode(InputMode::OpenMic));
    h.send(AppMessage::TrayToggleTranslate);

    // Simulate the SaveConfig effect by saving to_config
    let new_config = h.state.to_config(&base);
    new_config.save_to(&path).unwrap();

    // Reload and verify
    let loaded = Config::load_from(&path).unwrap();
    assert_eq!(loaded.model_size(), "medium");
    assert_eq!(loaded.language(), "ja");
    assert!(loaded.spoken_punctuation());
    assert!(loaded.streaming());
    assert_eq!(*loaded.mode(), InputMode::OpenMic);
    assert!(loaded.translate_to_english());
}

// ═══════════════════════════════════════════════════════════════════════
//  Reload config effect flow
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn reload_config_emits_correct_effect() {
    let mut h = Harness::new();
    let fx = h.send(AppMessage::TrayReloadConfig);
    assert!(has_effect(&fx, |e| matches!(e, AppEffect::ReloadConfig)));
}

#[test]
fn open_config_emits_correct_effect() {
    let mut h = Harness::new();
    let fx = h.send(AppMessage::TrayOpenConfig);
    assert!(has_effect(&fx, |e| matches!(e, AppEffect::OpenConfig)));
}

// ═══════════════════════════════════════════════════════════════════════
//  Quit effect
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn quit_message_emits_quit_effect() {
    let mut h = Harness::new();
    let fx = h.send(AppMessage::TrayQuit);
    assert!(has_effect(&fx, |e| matches!(e, AppEffect::Quit)));
}

// ═══════════════════════════════════════════════════════════════════════
//  Copy to clipboard
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn copy_last_with_no_transcription() {
    let mut h = Harness::new();
    let fx = h.send(AppMessage::TrayCopyLast);
    // No transcription yet, should not emit CopyToClipboard
    assert!(!has_effect(&fx, |e| matches!(
        e,
        AppEffect::CopyToClipboard(_)
    )));
}

#[test]
fn copy_last_after_transcription() {
    let mut h = Harness::new();

    // Do a transcription cycle
    h.send(AppMessage::KeyDown);
    h.send(AppMessage::KeyUp);
    h.send(AppMessage::TranscriptionDone("hello world".into()));

    let fx = h.send(AppMessage::TrayCopyLast);
    assert!(has_effect(&fx, |e| matches!(
        e,
        AppEffect::CopyToClipboard(t) if t == "hello world"
    )));
}

// ═══════════════════════════════════════════════════════════════════════
//  Model utilities (no hardware needed)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn model_filename_format() {
    assert_eq!(model::model_filename("base.en"), "ggml-base.en.bin");
    assert_eq!(model::model_filename("tiny"), "ggml-tiny.bin");
    assert_eq!(
        model::model_filename("large-v3-turbo"),
        "ggml-large-v3-turbo.bin"
    );
}

#[test]
fn model_url_format() {
    let url = model::model_url("base.en");
    assert!(url.starts_with("https://"));
    assert!(url.contains("huggingface"));
    assert!(url.contains("ggml-base.en.bin"));
}

#[test]
fn ggml_file_validation() {
    let tmp = tempfile::TempDir::new().unwrap();

    // Valid GGML magic
    let valid = tmp.path().join("valid.bin");
    helpers::write_fake_ggml(&valid);
    assert!(model::is_valid_ggml_file(&valid));

    // Invalid file
    let invalid = tmp.path().join("invalid.bin");
    helpers::write_invalid_model(&invalid);
    assert!(!model::is_valid_ggml_file(&invalid));

    // Nonexistent
    assert!(!model::is_valid_ggml_file(&tmp.path().join("nope.bin")));
}

// ═══════════════════════════════════════════════════════════════════════
//  Streaming utilities (no hardware needed)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn vad_detects_silence() {
    let samples = vec![0.0f32; 16_000];
    assert!(!murmur::transcription::vad::contains_speech(&samples));
}

#[test]
fn vad_detects_no_speech_in_empty() {
    assert!(!murmur::transcription::vad::contains_speech(&[]));
}

#[test]
fn vad_detects_no_speech_near_zero() {
    let samples = vec![0.001f32; 1000];
    assert!(!murmur::transcription::vad::contains_speech(&samples));
}

#[test]
fn stitch_empty_committed() {
    let result = streaming::stitch(&[], &["hello".into(), "world".into()]);
    assert_eq!(result, vec!["hello", "world"]);
}

#[test]
fn stitch_empty_chunk() {
    let committed: Vec<String> = vec!["hello".into()];
    let result = streaming::stitch(&committed, &[]);
    assert!(result.is_empty());
}

#[test]
fn stitch_no_overlap() {
    let committed: Vec<String> = vec!["hello".into()];
    let chunk: Vec<String> = vec!["world".into(), "foo".into()];
    let result = streaming::stitch(&committed, &chunk);
    // When no overlap found, should return the chunk words
    assert!(!result.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════
//  Reload generation tracking
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn reload_generation_increments() {
    let mut config = Config::default();
    config.set_model_size("base".to_string()); // multilingual model allows language changes
    let mut h = Harness::with_config(&config);
    let gen0 = h.state.reload_generation();

    h.send(AppMessage::TraySetModel("small".into()));
    assert!(h.state.reload_generation() > gen0);

    let gen1 = h.state.reload_generation();
    h.send(AppMessage::TraySetLanguage("de".into()));
    assert!(h.state.reload_generation() > gen1);
}

// ═══════════════════════════════════════════════════════════════════════
//  Error during active recording preserves state
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn error_during_recording_preserves_recording_state() {
    let mut h = Harness::new();

    h.send(AppMessage::KeyDown);
    assert!(h.state.is_pressed());

    // Stale error from previous cycle
    let fx = h.send(AppMessage::TranscriptionError("stale".into()));
    assert!(h.state.is_pressed(), "recording should continue");
    assert!(has_effect(&fx, |e| matches!(e, AppEffect::LogError(_))));
}

#[test]
fn result_during_recording_inserts_but_preserves_state() {
    let mut h = Harness::new();

    h.send(AppMessage::KeyDown);
    assert!(h.state.is_pressed());

    let fx = h.send(AppMessage::TranscriptionDone("late result".into()));
    assert!(h.state.is_pressed());
    assert!(has_effect(&fx, |e| matches!(
        e,
        AppEffect::InsertText(t) if t == "late result"
    )));
    // Tray should NOT go to idle while recording
    assert!(!has_effect(&fx, |e| matches!(
        e,
        AppEffect::SetTrayState(TrayState::Idle)
    )));
}

// ═══════════════════════════════════════════════════════════════════════
//  Full config→state→effect→config pipeline
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn full_config_mutation_pipeline() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("config.json");

    // Start with defaults
    let base = Config::default();
    base.save_to(&path).unwrap();
    let mut h = Harness::with_config(&base);

    // Simulate user actions
    let mutations = vec![
        AppMessage::TraySetModel("large-v3-turbo".into()),
        AppMessage::TraySetLanguage("zh".into()),
        AppMessage::TrayToggleSpokenPunctuation,
        // Default backend (Qwen3Asr) auto-enables streaming, so toggle off then on.
        AppMessage::TrayToggleStreaming, // true → false
        AppMessage::TrayToggleStreaming, // false → true
        AppMessage::TrayToggleTranslate,
        AppMessage::TraySetMode(InputMode::OpenMic),
    ];

    for msg in mutations {
        let fx = h.send(msg);
        // Every mutation should trigger SaveConfig
        assert!(
            has_effect(&fx, |e| matches!(e, AppEffect::SaveConfig)),
            "mutation should emit SaveConfig"
        );
    }

    // Save final state
    let final_config = h.state.to_config(&base);
    final_config.save_to(&path).unwrap();

    // Reload and verify everything persisted
    let reloaded = Config::load_from(&path).unwrap();
    assert_eq!(reloaded.model_size(), "large-v3-turbo");
    assert_eq!(reloaded.language(), "zh");
    assert!(reloaded.spoken_punctuation());
    assert!(reloaded.streaming());
    assert!(reloaded.translate_to_english());
    assert_eq!(*reloaded.mode(), InputMode::OpenMic);

    // Create new state from reloaded config — should match
    let state2 = AppState::new(&reloaded);
    assert_eq!(state2.model_size(), "large-v3-turbo");
    assert_eq!(state2.language(), "zh");
    assert!(state2.spoken_punctuation());
    assert!(state2.streaming());
    assert!(state2.translate_to_english());
    assert_eq!(*state2.mode(), InputMode::OpenMic);
}
