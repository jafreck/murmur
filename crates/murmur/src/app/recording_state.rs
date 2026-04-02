use crate::config::InputMode;
use crate::ui::tray::TrayState;

use super::messages::AppEffect;
use super::state::AppState;

/// Recording, streaming, and transcription state-transition helpers.
///
/// These are private methods on `AppState` called from `handle_message`.
impl AppState {
    pub(super) fn start_recording_effects(&mut self) -> Vec<AppEffect> {
        self.is_pressed = true;
        self.streaming_active = self.streaming;
        self.streaming_chars_emitted = 0;
        self.last_speech_at = Some(std::time::Instant::now());
        let path = self.recording_output_path();
        // Set tray state first so the icon updates immediately.
        let mut effects = vec![AppEffect::SetTrayState(TrayState::Recording)];
        effects.push(AppEffect::StartRecording(path));
        if self.streaming {
            effects.push(AppEffect::StartStreaming);
        }
        if self.is_notes_mode() {
            self.overlay_text.clear();
            effects.push(AppEffect::ShowOverlay);
        }
        if self.is_notes_mode() {
            effects.push(AppEffect::PauseWakeWord);
        }
        effects
    }

    pub(super) fn stop_recording_effects(&mut self) -> Vec<AppEffect> {
        self.is_pressed = false;
        self.wake_word_initiated = false;
        self.last_speech_at = None;
        let mut effects = Vec::new();
        if self.streaming {
            effects.push(AppEffect::StopStreaming);
        }
        // Dispatch transcription first (synchronous cleanup happens here),
        // then update the tray so the dots animation starts smoothly
        // without being blocked by the cleanup work.
        effects.push(AppEffect::StopAndTranscribe);
        effects.push(AppEffect::SetTrayState(TrayState::Transcribing));
        effects
    }

    pub(super) fn on_streaming_partial(
        &mut self,
        text: &str,
        replace_chars: &usize,
    ) -> Vec<AppEffect> {
        // Check for stop phrase in wake-word-initiated sessions
        if self.wake_word_initiated && !self.stop_phrase.is_empty() {
            if let Some(cleaned) =
                murmur_core::input::wake_word::check_and_strip_stop_phrase(text, &self.stop_phrase)
            {
                // Stop phrase found — end dictation with cleaned text
                let char_count = cleaned.chars().count();
                let mut effects = vec![];
                if *replace_chars > 0 || !cleaned.is_empty() {
                    self.streaming_chars_emitted = char_count;
                    effects.push(AppEffect::StreamingReplace {
                        text: cleaned.clone(),
                        replace_chars: *replace_chars,
                    });
                }
                if self.is_notes_mode() {
                    self.overlay_text = cleaned;
                }
                // Trigger stop
                effects.extend(self.stop_recording_effects());
                return effects;
            }
        }

        if *replace_chars > 0 || !text.is_empty() {
            self.streaming_chars_emitted = text.chars().count();
            let mut effects = vec![];
            // In Dictation mode, type text at cursor; in Notes mode, only update overlay
            if !self.is_notes_mode() {
                effects.push(AppEffect::StreamingReplace {
                    text: text.to_string(),
                    replace_chars: *replace_chars,
                });
            }
            if self.is_notes_mode() {
                self.overlay_text = text.to_string();
                effects.push(AppEffect::UpdateOverlayText(text.to_string()));
            }
            effects
        } else {
            vec![AppEffect::None]
        }
    }

    pub(super) fn on_wake_word_detected(&mut self) -> Vec<AppEffect> {
        if self.is_pressed {
            return vec![AppEffect::None];
        }
        log::info!("Wake word detected — starting dictation");
        self.wake_word_initiated = true;
        self.start_recording_effects()
    }

    pub(super) fn on_stop_phrase_detected(&mut self) -> Vec<AppEffect> {
        if !self.is_pressed {
            return vec![AppEffect::None];
        }
        log::info!("Stop phrase detected — stopping dictation");
        self.stop_recording_effects()
    }

    /// Seconds of silence before auto-stopping a wake-word-initiated session.
    const WAKE_WORD_SILENCE_TIMEOUT_SECS: f32 = 5.0;

    /// Check if a wake-word-initiated session should auto-stop due to silence.
    /// Called from the main loop on each tick.
    pub fn check_silence_timeout(&mut self) -> Vec<AppEffect> {
        if !self.wake_word_initiated || !self.is_pressed {
            return vec![];
        }
        if let Some(last) = self.last_speech_at {
            if last.elapsed().as_secs_f32() >= Self::WAKE_WORD_SILENCE_TIMEOUT_SECS {
                log::info!(
                    "Silence timeout ({:.0}s) — auto-stopping wake-word dictation",
                    Self::WAKE_WORD_SILENCE_TIMEOUT_SECS
                );
                return self.stop_recording_effects();
            }
        }
        vec![]
    }

    pub(super) fn on_key_down(&mut self) -> Vec<AppEffect> {
        log::info!(
            "on_key_down: mode={:?} is_pressed={}",
            self.mode,
            self.is_pressed
        );
        match (&self.mode, self.is_pressed) {
            (InputMode::OpenMic, true) => self.stop_recording_effects(),
            (InputMode::OpenMic, false) => self.start_recording_effects(),
            (InputMode::PushToTalk, false) => self.start_recording_effects(),
            (InputMode::PushToTalk, true) => {
                // Key repeat while held — ignore.
                vec![AppEffect::None]
            }
        }
    }

    pub(super) fn on_key_up(&mut self) -> Vec<AppEffect> {
        log::info!(
            "on_key_up: mode={:?} is_pressed={}",
            self.mode,
            self.is_pressed
        );
        if self.mode == InputMode::PushToTalk && self.is_pressed {
            self.stop_recording_effects()
        } else {
            vec![AppEffect::None]
        }
    }

    pub(super) fn on_transcription_done(&mut self, text: &str) -> Vec<AppEffect> {
        let was_streaming = self.streaming_active;
        let streamed_chars = self.streaming_chars_emitted;
        // Only clear streaming state if not currently recording
        if !self.is_pressed {
            self.streaming_active = false;
            self.streaming_chars_emitted = 0;
        }

        let mut effects = vec![];
        if !text.is_empty() {
            // In Dictation mode, paste/type text at cursor
            if !self.is_notes_mode() {
                if was_streaming && streamed_chars > 0 {
                    effects.push(AppEffect::StreamingReplace {
                        text: text.to_string(),
                        replace_chars: streamed_chars,
                    });
                } else {
                    // Either streaming wasn't active, or it was but produced
                    // no output yet (e.g. short audio, VAD filtered). Insert
                    // the batch transcription directly.
                    effects.push(AppEffect::InsertText(text.to_string()));
                }
            }
            self.last_transcription = Some(text.to_string());

            // Update overlay with final text and save note
            if self.is_notes_mode() {
                self.overlay_text = text.to_string();
                effects.push(AppEffect::UpdateOverlayText(text.to_string()));
                effects.push(AppEffect::SaveNote(text.to_string()));
                effects.push(AppEffect::HideOverlay);
            }
        } else if self.is_notes_mode() {
            effects.push(AppEffect::HideOverlay);
        }

        if !self.is_pressed {
            effects.push(AppEffect::SetTrayState(TrayState::Idle));
            // Resume wake word detection after dictation completes
            if self.is_notes_mode() {
                effects.push(AppEffect::ResumeWakeWord);
            }
        }
        effects
    }

    pub(super) fn on_transcription_error(&mut self, error: &str) -> Vec<AppEffect> {
        // Only reset state if not currently recording — the error may be
        // from a previous cycle's transcription thread.
        let mut effects = vec![AppEffect::LogError(error.to_string())];
        if !self.is_pressed {
            self.streaming_active = false;
            effects.push(AppEffect::SetTrayState(TrayState::Error));
        }
        effects
    }
}
