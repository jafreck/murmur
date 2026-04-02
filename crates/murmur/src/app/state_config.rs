use crate::config::{AppMode, Config, InputMode};
use crate::ui::tray::TrayState;
use rdev::Key;

use super::messages::AppEffect;
use super::state::AppState;

/// Config reconciliation and tray-setting methods on `AppState`.
impl AppState {
    pub(super) fn on_copy_last(&self) -> Vec<AppEffect> {
        if let Some(ref text) = self.last_transcription {
            vec![AppEffect::CopyToClipboard(text.clone())]
        } else {
            vec![AppEffect::None]
        }
    }

    pub(super) fn on_set_model(&mut self, size: &str) -> Vec<AppEffect> {
        self.model_size = size.to_string();
        self.reload_generation += 1;
        let mut effects = vec![
            AppEffect::SetTrayModel(size.to_string()),
            AppEffect::SetLanguageMenuEnabled(!crate::config::is_english_only_model(size)),
        ];
        if crate::config::is_english_only_model(size) && self.language != "en" {
            self.language = "en".to_string();
            effects.push(AppEffect::SetTrayLanguage("en".to_string()));
        }
        effects.push(AppEffect::SaveConfig);
        effects.push(AppEffect::ReloadTranscriber(self.reload_generation));
        effects
    }

    pub(super) fn on_set_backend(
        &mut self,
        backend: murmur_core::config::AsrBackend,
    ) -> Vec<AppEffect> {
        self.reload_generation += 1;
        vec![
            AppEffect::SetBackend(backend),
            AppEffect::SaveConfig,
            AppEffect::ReloadTranscriber(self.reload_generation),
        ]
    }

    pub(super) fn on_set_language(&mut self, code: &str) -> Vec<AppEffect> {
        if crate::config::is_english_only_model(&self.model_size) {
            // English-only model — ignore language change, reset tray to English
            return vec![AppEffect::SetTrayLanguage("en".to_string())];
        }
        self.language = code.to_string();
        self.reload_generation += 1;
        vec![
            AppEffect::SaveConfig,
            AppEffect::SetTrayLanguage(code.to_string()),
            AppEffect::ReloadTranscriber(self.reload_generation),
        ]
    }

    pub(super) fn on_toggle_spoken_punctuation(&mut self) -> Vec<AppEffect> {
        self.spoken_punctuation = !self.spoken_punctuation;
        vec![AppEffect::SaveConfig]
    }

    pub(super) fn on_toggle_filler_word_removal(&mut self) -> Vec<AppEffect> {
        self.filler_word_removal = !self.filler_word_removal;
        vec![AppEffect::SaveConfig]
    }

    pub(super) fn on_set_mode(&mut self, mode: &InputMode) -> Vec<AppEffect> {
        self.mode = mode.clone();
        vec![AppEffect::SaveConfig, AppEffect::SetTrayMode(mode.clone())]
    }

    pub(super) fn on_toggle_streaming(&mut self) -> Vec<AppEffect> {
        self.streaming = !self.streaming;
        vec![AppEffect::SaveConfig]
    }

    pub(super) fn on_toggle_translate(&mut self) -> Vec<AppEffect> {
        self.translate_to_english = !self.translate_to_english;
        vec![AppEffect::SaveConfig]
    }

    pub(super) fn on_toggle_noise_suppression(&mut self) -> Vec<AppEffect> {
        self.noise_suppression = !self.noise_suppression;
        vec![
            AppEffect::UpdateNoiseSuppression(self.noise_suppression),
            AppEffect::SaveConfig,
        ]
    }

    pub(super) fn on_tray_set_hotkey(&mut self) -> Vec<AppEffect> {
        self.capturing_hotkey = true;
        vec![
            AppEffect::EnterHotkeyCaptureMode,
            AppEffect::SetTrayState(TrayState::Idle),
        ]
    }

    pub(super) fn on_hotkey_capture(&mut self, key: &Key) -> Vec<AppEffect> {
        self.capturing_hotkey = false;
        let key_name = crate::input::keycodes::key_to_name(key);
        vec![AppEffect::SetHotkey(key_name), AppEffect::SaveConfig]
    }

    pub(super) fn on_toggle_app_mode(&mut self) -> Vec<AppEffect> {
        let mut effects = vec![AppEffect::SaveConfig];
        match self.app_mode {
            AppMode::Dictation => {
                self.app_mode = AppMode::Notes;
                effects.push(AppEffect::SpawnOverlay);
                effects.push(AppEffect::StartWakeWord);
            }
            AppMode::Notes => {
                self.app_mode = AppMode::Dictation;
                effects.push(AppEffect::KillOverlay);
                effects.push(AppEffect::StopWakeWord);
            }
        }
        effects
    }

    pub fn to_config(&self, base: &Config) -> Config {
        let mut cfg = base.clone();
        cfg.set_model_size(self.model_size.clone());
        cfg.set_language(self.language.clone());
        cfg.set_spoken_punctuation(self.spoken_punctuation);
        cfg.set_filler_word_removal(self.filler_word_removal);
        cfg.set_mode(self.mode.clone());
        cfg.set_streaming(self.streaming);
        cfg.set_translate_to_english(self.translate_to_english);
        cfg.set_noise_suppression(self.noise_suppression);
        cfg.set_app_mode(self.app_mode);
        cfg
    }
}
