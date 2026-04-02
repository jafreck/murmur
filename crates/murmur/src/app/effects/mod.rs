mod config;
mod engine;
mod overlay;
mod recording;
mod wake_word;

use anyhow::Result;
use log::{error, info};
use std::sync::mpsc;
use std::sync::Arc;

use crate::audio::AudioRecorder;
use crate::config::Config;
use crate::input::hotkey::{CaptureFlag, SharedHotkeyConfig};
use crate::input::inserter::TextInserter;
use crate::notes::NotesManager;
use crate::transcription::streaming;
use crate::ui::overlay::OverlayHandle;
use crate::ui::tray::{TrayController, TrayState};
use murmur_core::input::wake_word::WakeWordHandle;
use murmur_core::transcription::{AsrEngine, DefaultEngineFactory};

use super::{AppEffect, AppMessage, AppState};

// Re-export pub(crate) items from submodules so they remain accessible
// at `crate::app::effects::*` as before the split.
#[allow(unused_imports)]
pub(crate) use config::{compute_config_diff, ConfigDiff};
#[allow(unused_imports)]
pub(crate) use recording::{
    decide_transcription, postprocess_transcription, recording_mode, RecordingMode,
    TranscribeDecision,
};

// Re-export create_engine_on_thread for use by app::mod.rs
pub(super) use engine::create_engine_on_thread;

// ---------------------------------------------------------------------------
// Pure decision functions (kept here as they're used in the dispatch)
// ---------------------------------------------------------------------------

/// The tray state to show after exiting hotkey-capture mode.
pub(crate) fn tray_state_after_hotkey_capture(has_transcriber: bool) -> TrayState {
    if has_transcriber {
        TrayState::Idle
    } else {
        TrayState::Loading
    }
}

/// Whether the streaming pipeline has everything it needs to start.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamingReadiness {
    NoTranscriber,
    NoWorker,
    Ready,
}

/// Check whether the streaming pipeline can be started.
#[cfg(test)]
pub(crate) fn check_streaming_readiness(
    has_transcriber: bool,
    has_worker: bool,
) -> StreamingReadiness {
    if !has_transcriber {
        StreamingReadiness::NoTranscriber
    } else if !has_worker {
        StreamingReadiness::NoWorker
    } else {
        StreamingReadiness::Ready
    }
}

// ---------------------------------------------------------------------------

pub struct EffectContext<'a> {
    pub recorder: &'a mut AudioRecorder,
    pub engine: &'a mut Option<Arc<dyn AsrEngine + Send + Sync>>,
    pub engine_factory: &'a Arc<DefaultEngineFactory>,
    pub tray: &'a mut TrayController,
    pub config: &'a mut Config,
    pub state: &'a mut AppState,
    pub tx: &'a mpsc::Sender<AppMessage>,
    pub streaming_stop: &'a mut Option<streaming::StreamingHandle>,
    pub streaming_worker: &'a mut Option<murmur_core::transcription::SubprocessTranscriber>,
    pub hotkey_config: &'a SharedHotkeyConfig,
    pub capture_flag: &'a CaptureFlag,
    pub overlay: &'a mut Option<OverlayHandle>,
    pub wake_word: &'a mut Option<WakeWordHandle>,
    pub notes: &'a NotesManager,
}

/// Apply a single effect, returning (should_quit, extra_effects).
pub fn apply_effect(
    effect: AppEffect,
    ctx: &mut EffectContext<'_>,
) -> Result<(bool, Vec<AppEffect>)> {
    match effect {
        AppEffect::None => {}
        AppEffect::StartRecording(path) => {
            recording::handle_start_recording(ctx, path);
        }
        AppEffect::StopAndTranscribe => {
            recording::stop_and_transcribe(ctx);
        }
        AppEffect::StartStreaming => {
            recording::start_streaming(ctx);
        }
        AppEffect::StopStreaming => {
            if let Some(handle) = ctx.streaming_stop.take() {
                info!("Stopping streaming transcription");
                handle.stop_and_join();
                ctx.state.streaming_completed = true;
            }
        }
        AppEffect::InsertText(text) => {
            info!("Transcription: {text}");
            if let Err(e) = TextInserter::insert(&text) {
                error!("Insert failed: {e}");
            }
        }
        AppEffect::StreamingReplace {
            text,
            replace_chars,
        } => {
            info!(
                "StreamingReplace: delete {} chars, type {} chars: {:?}",
                replace_chars,
                text.chars().count(),
                if text.len() > 60 { &text[..60] } else { &text }
            );
            if let Err(e) = TextInserter::replace(replace_chars, &text) {
                error!("Streaming replace failed: {e}");
            }
        }
        AppEffect::CopyToClipboard(text) => {
            if let Ok(mut cb) = arboard::Clipboard::new() {
                if let Err(e) = cb.set_text(text) {
                    log::warn!("Failed to copy to clipboard: {e}");
                }
                info!("Copied last dictation to clipboard");
            }
        }
        AppEffect::SaveConfig => {
            let new_config = ctx.state.to_config(ctx.config);
            *ctx.config = new_config;
            if let Err(e) = ctx.config.save() {
                error!("Failed to save config: {e}");
            }
        }
        AppEffect::OpenConfig => {
            config::open_config_file();
        }
        AppEffect::ReloadConfig => {
            return config::reload_config(ctx);
        }
        AppEffect::SetTrayState(state) => {
            ctx.tray.set_state(state);
        }
        AppEffect::SetTrayModel(size) => {
            ctx.tray.set_model(&size);
            info!("Model changed to: {size}");
        }
        AppEffect::SetTrayLanguage(code) => {
            let name = crate::config::language_name(&code).unwrap_or(&code);
            info!("Language changed to: {name} ({code})");
            ctx.tray.set_language(&code);
        }
        AppEffect::SetLanguageMenuEnabled(enabled) => {
            ctx.tray.set_language_menu_enabled(enabled);
        }
        AppEffect::SetTrayMode(mode) => {
            ctx.tray.set_mode(&mode);
            info!("Mode changed to: {mode}");
        }
        AppEffect::SetBackend(backend) => {
            ctx.config.asr_backend = backend;
            let default_model = ctx.config.default_model_for_backend().to_string();
            ctx.config.model_size = default_model.clone();
            ctx.state.model_size = default_model.clone();
            ctx.tray.set_model(&default_model);
            ctx.tray.sync_config(ctx.config);
            info!(
                "Backend changed to: {backend}; model reset to {default_model}. \
                 Restart murmur for the Model menu to update."
            );
        }
        AppEffect::ReloadTranscriber(generation) => {
            *ctx.engine = None;
            ctx.tray.set_state(TrayState::Loading);
            engine::reload_transcriber(ctx, generation);
        }
        AppEffect::SwapEngine => {
            if let Some(e) = ctx.state.pending_engine.take() {
                *ctx.engine = Some(e);
                info!("ASR engine ready");
            }
        }
        AppEffect::SpawnStreamingWorker => {
            if matches!(
                ctx.config.asr_backend,
                murmur_core::config::AsrBackend::Whisper
            ) {
                if let Some(model_path) =
                    murmur_core::transcription::find_model(&ctx.config.model_size)
                {
                    match murmur_core::transcription::SubprocessTranscriber::new(
                        &model_path,
                        &ctx.config.language,
                    ) {
                        Ok(w) => {
                            *ctx.streaming_worker = Some(w);
                            info!("Whisper worker subprocess ready");
                        }
                        Err(e) => {
                            error!("Failed to spawn whisper worker: {e}");
                        }
                    }
                }
            }
        }
        AppEffect::SetTrayUpdateAvailable(version) => {
            ctx.tray.set_update_available(&version);
        }
        AppEffect::SetTrayStatus(msg) => {
            ctx.tray.set_status(&msg);
        }
        AppEffect::CheckForUpdates => {
            let tx = ctx.tx.clone();
            std::thread::spawn(move || {
                match murmur_core::update::check_for_update(crate::VERSION) {
                    Ok(Some(info)) => {
                        let _ = tx.send(AppMessage::UpdateAvailable(info));
                    }
                    Ok(None) => {
                        info!("Already up to date (v{})", crate::VERSION);
                    }
                    Err(e) => {
                        error!("Update check failed: {e}");
                        let _ = tx.send(AppMessage::UpdateError(format!("{e}")));
                    }
                }
            });
        }
        AppEffect::PrintReady => {
            println!("Ready.");
        }
        AppEffect::EnterHotkeyCaptureMode => {
            ctx.capture_flag
                .store(true, std::sync::atomic::Ordering::Relaxed);
            ctx.tray.set_status("Press a key...");
            info!("Hotkey capture mode: press any key");
        }
        AppEffect::SetHotkey(key_name) => {
            if let Some(parsed) = crate::input::keycodes::parse(&key_name) {
                if let Ok(mut hk) = ctx.hotkey_config.lock() {
                    *hk = (parsed.key, parsed.modifiers.into_iter().collect());
                }
                ctx.config.hotkey = key_name.clone();
                ctx.tray.set_hotkey(&key_name);
                info!("Hotkey set to: {key_name}");
            } else {
                error!("Invalid captured key: {key_name}");
            }
            ctx.tray
                .set_state(tray_state_after_hotkey_capture(ctx.engine.is_some()));
        }
        AppEffect::UpdateNoiseSuppression(enabled) => {
            ctx.recorder.set_noise_suppression(enabled);
            info!(
                "Noise suppression {}",
                if enabled { "enabled" } else { "disabled" }
            );
        }
        AppEffect::ShowOverlay => {
            overlay::show(ctx);
        }
        AppEffect::HideOverlay => {
            overlay::hide(ctx);
        }
        AppEffect::UpdateOverlayText(text) => {
            overlay::update_text(ctx, &text);
        }
        AppEffect::SaveNote(text) => {
            if !text.is_empty() {
                if let Err(e) = ctx.notes.save(&text) {
                    error!("Failed to save note: {e}");
                }
            }
        }
        AppEffect::PauseWakeWord => {
            if let Some(ref ww) = ctx.wake_word {
                ww.pause();
            }
        }
        AppEffect::ResumeWakeWord => {
            if let Some(ref ww) = ctx.wake_word {
                ww.resume();
            }
        }
        AppEffect::StartWakeWord => {
            wake_word::start(ctx);
        }
        AppEffect::StopWakeWord => {
            wake_word::stop(ctx);
        }
        AppEffect::SpawnOverlay => {
            overlay::spawn(ctx);
        }
        AppEffect::KillOverlay => {
            overlay::kill(ctx);
        }
        AppEffect::Quit => {
            info!("Quit requested via tray");
            return Ok((true, vec![]));
        }
        AppEffect::LogError(e) => {
            error!("Transcription: {e}");
        }
    }
    Ok((false, vec![]))
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // compute_config_diff
    // -----------------------------------------------------------------------

    fn config_with(f: impl FnOnce(&mut Config)) -> Config {
        let mut c = Config::default();
        f(&mut c);
        c
    }

    #[test]
    fn config_diff_no_changes() {
        let config = Config::default();
        let diff = compute_config_diff(
            &config.model_size,
            &config.language,
            &config.hotkey,
            &config,
        );
        assert!(!diff.model_or_language_changed);
        assert!(!diff.hotkey_changed);
        // Default model is "0.6b" (Qwen3-ASR), so language menu is enabled
        assert!(diff.language_menu_enabled);
        assert_eq!(diff.effective_language, "en");
    }

    #[test]
    fn config_diff_model_changed() {
        let new_config = config_with(|c| c.model_size = "large".to_string());
        let diff = compute_config_diff("base", "en", &new_config.hotkey, &new_config);
        assert!(diff.model_or_language_changed);
        assert!(!diff.hotkey_changed);
    }

    #[test]
    fn config_diff_language_changed() {
        // Use a non-english-only model so language isn't forced
        let old = config_with(|c| c.model_size = "base".to_string());
        let new_config = config_with(|c| {
            c.model_size = "base".to_string();
            c.language = "fr".to_string();
        });
        let diff = compute_config_diff(&old.model_size, &old.language, &old.hotkey, &new_config);
        assert!(diff.model_or_language_changed);
        assert_eq!(diff.effective_language, "fr");
    }

    #[test]
    fn config_diff_hotkey_changed() {
        let old = Config::default();
        let new_config = config_with(|c| c.hotkey = "F12".to_string());
        let diff = compute_config_diff(&old.model_size, &old.language, &old.hotkey, &new_config);
        assert!(diff.hotkey_changed);
        assert!(!diff.model_or_language_changed);
    }

    #[test]
    fn config_diff_english_only_model_forces_language() {
        let new_config = config_with(|c| {
            c.model_size = "base.en".to_string();
            c.language = "fr".to_string();
        });
        let diff = compute_config_diff("base", "en", &new_config.hotkey, &new_config);
        assert_eq!(diff.effective_language, "en");
        assert!(!diff.language_menu_enabled);
        // Model changed from "base" to "base.en"
        assert!(diff.model_or_language_changed);
    }

    #[test]
    fn config_diff_english_only_model_already_english() {
        let new_config = config_with(|c| {
            c.model_size = "base.en".to_string();
            c.language = "en".to_string();
        });
        let diff = compute_config_diff("base.en", "en", &new_config.hotkey, &new_config);
        assert_eq!(diff.effective_language, "en");
        assert!(!diff.language_menu_enabled);
        assert!(!diff.model_or_language_changed);
    }

    #[test]
    fn config_diff_distil_model_forces_english() {
        let new_config = config_with(|c| {
            c.model_size = "distil-large".to_string();
            c.language = "de".to_string();
        });
        let diff = compute_config_diff("base", "de", &new_config.hotkey, &new_config);
        assert_eq!(diff.effective_language, "en");
        assert!(!diff.language_menu_enabled);
        // Model changed AND effective language changed (de → en)
        assert!(diff.model_or_language_changed);
    }

    #[test]
    fn config_diff_multiple_changes() {
        let new_config = config_with(|c| {
            c.model_size = "large".to_string();
            c.language = "fr".to_string();
            c.hotkey = "F12".to_string();
        });
        let diff = compute_config_diff("base", "en", "F10", &new_config);
        assert!(diff.model_or_language_changed);
        assert!(diff.hotkey_changed);
        assert_eq!(diff.effective_language, "fr");
        assert!(diff.language_menu_enabled);
    }

    #[test]
    fn config_diff_non_english_only_preserves_language() {
        let new_config = config_with(|c| {
            c.model_size = "large".to_string();
            c.language = "ja".to_string();
        });
        let diff = compute_config_diff("large", "ja", &new_config.hotkey, &new_config);
        assert_eq!(diff.effective_language, "ja");
        assert!(diff.language_menu_enabled);
        assert!(!diff.model_or_language_changed);
    }

    #[test]
    fn config_diff_english_only_language_change_masked() {
        // Config says "fr" but model is english-only → effective is "en".
        // Old language was "en" so no change detected.
        let new_config = config_with(|c| {
            c.model_size = "tiny.en".to_string();
            c.language = "fr".to_string();
        });
        let diff = compute_config_diff("tiny.en", "en", &new_config.hotkey, &new_config);
        assert_eq!(diff.effective_language, "en");
        assert!(!diff.model_or_language_changed);
    }

    // -----------------------------------------------------------------------
    // decide_transcription
    // -----------------------------------------------------------------------

    #[test]
    fn decide_no_transcriber() {
        assert_eq!(
            decide_transcription(false, 0, false, false, false),
            TranscribeDecision::NoTranscriber
        );
        assert_eq!(
            decide_transcription(false, 10, true, true, false),
            TranscribeDecision::NoTranscriber
        );
    }

    #[test]
    fn decide_no_audio_in_memory() {
        assert_eq!(
            decide_transcription(true, 0, false, false, false),
            TranscribeDecision::NoAudio
        );
    }

    #[test]
    fn decide_no_audio_file_mode() {
        assert_eq!(
            decide_transcription(true, 10, false, false, false),
            TranscribeDecision::NoAudio
        );
    }

    #[test]
    fn decide_streaming_skip_in_memory_whisper() {
        // Whisper chunked streaming: skip batch (chunks already cover full audio)
        assert_eq!(
            decide_transcription(true, 0, true, true, false),
            TranscribeDecision::SkipBatchStreamingDone
        );
    }

    #[test]
    fn decide_native_streaming_always_batch() {
        // Native streaming: always do final batch (streaming was just a preview)
        assert_eq!(
            decide_transcription(true, 0, true, true, true),
            TranscribeDecision::Transcribe
        );
    }

    #[test]
    fn decide_streaming_not_skipped_in_file_mode() {
        // In file mode (max_recordings > 0), streaming skip doesn't apply
        assert_eq!(
            decide_transcription(true, 10, true, true, false),
            TranscribeDecision::Transcribe
        );
    }

    #[test]
    fn decide_transcribe_in_memory() {
        assert_eq!(
            decide_transcription(true, 0, false, true, false),
            TranscribeDecision::Transcribe
        );
    }

    #[test]
    fn decide_transcribe_file_mode() {
        assert_eq!(
            decide_transcription(true, 10, false, true, false),
            TranscribeDecision::Transcribe
        );
    }

    #[test]
    fn decide_no_transcriber_overrides_all() {
        // NoTranscriber takes priority regardless of other flags
        for &streaming in &[true, false] {
            for &has_audio in &[true, false] {
                assert_eq!(
                    decide_transcription(false, 0, streaming, has_audio, false),
                    TranscribeDecision::NoTranscriber
                );
            }
        }
    }

    #[test]
    fn decide_no_audio_overrides_streaming() {
        // Even if streaming was active, no audio means nothing to do
        assert_eq!(
            decide_transcription(true, 0, true, false, false),
            TranscribeDecision::NoAudio
        );
    }

    // -----------------------------------------------------------------------
    // recording_mode
    // -----------------------------------------------------------------------

    #[test]
    fn recording_mode_in_memory_when_zero() {
        assert_eq!(recording_mode(0), RecordingMode::InMemory);
    }

    #[test]
    fn recording_mode_file_when_nonzero() {
        assert_eq!(recording_mode(1), RecordingMode::File);
        assert_eq!(recording_mode(100), RecordingMode::File);
    }

    // -----------------------------------------------------------------------
    // postprocess_transcription
    // -----------------------------------------------------------------------

    #[test]
    fn postprocess_passthrough() {
        // With both flags off, only ensure_space_after_punctuation runs
        assert_eq!(
            postprocess_transcription("hello world", false, false),
            "hello world"
        );
    }

    #[test]
    fn postprocess_space_after_punctuation_always_applied() {
        // Even with both flags off the spacing normaliser runs
        assert_eq!(
            postprocess_transcription("hello,world", false, false),
            "hello, world"
        );
    }

    #[test]
    fn postprocess_filler_removal_only() {
        let result = postprocess_transcription("um hello um world", true, false);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn postprocess_spoken_punctuation_only() {
        let result = postprocess_transcription("hello period", false, true);
        assert!(result.contains('.'), "expected '.' in: {result}");
        assert!(
            !result.contains("period"),
            "spoken word should be replaced: {result}"
        );
    }

    #[test]
    fn postprocess_both_enabled() {
        let result = postprocess_transcription("um hello comma world period", true, true);
        assert!(!result.contains("um"), "filler should be removed: {result}");
        assert!(result.contains(','), "comma should be present: {result}");
        assert!(result.contains('.'), "period should be present: {result}");
    }

    #[test]
    fn postprocess_empty_text() {
        assert_eq!(postprocess_transcription("", false, false), "");
        assert_eq!(postprocess_transcription("", true, true), "");
    }

    #[test]
    fn postprocess_only_fillers_produces_empty() {
        let result = postprocess_transcription("um uh er", true, false);
        assert_eq!(result, "");
    }

    #[test]
    fn postprocess_filler_removal_preserves_real_words() {
        let result = postprocess_transcription("I think so", true, false);
        assert_eq!(result, "I think so");
    }

    // -----------------------------------------------------------------------
    // tray_state_after_hotkey_capture
    // -----------------------------------------------------------------------

    #[test]
    fn tray_after_hotkey_idle_when_transcriber_loaded() {
        assert_eq!(tray_state_after_hotkey_capture(true), TrayState::Idle);
    }

    #[test]
    fn tray_after_hotkey_loading_when_no_transcriber() {
        assert_eq!(tray_state_after_hotkey_capture(false), TrayState::Loading);
    }

    // -----------------------------------------------------------------------
    // check_streaming_readiness
    // -----------------------------------------------------------------------

    #[test]
    fn streaming_readiness_no_transcriber() {
        assert_eq!(
            check_streaming_readiness(false, false),
            StreamingReadiness::NoTranscriber
        );
        assert_eq!(
            check_streaming_readiness(false, true),
            StreamingReadiness::NoTranscriber
        );
    }

    #[test]
    fn streaming_readiness_no_worker() {
        assert_eq!(
            check_streaming_readiness(true, false),
            StreamingReadiness::NoWorker
        );
    }

    #[test]
    fn streaming_readiness_ready() {
        assert_eq!(
            check_streaming_readiness(true, true),
            StreamingReadiness::Ready
        );
    }

    #[test]
    fn streaming_readiness_no_transcriber_takes_priority() {
        // Even when worker is present, no transcriber wins
        assert_eq!(
            check_streaming_readiness(false, true),
            StreamingReadiness::NoTranscriber
        );
    }
}
