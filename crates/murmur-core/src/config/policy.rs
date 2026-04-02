use std::path::PathBuf;

use super::schema::{AppMode, AsrBackend, Config, DictationMode};

impl Config {
    pub fn effective_max_recordings(value: u32) -> u32 {
        if value == 0 {
            0
        } else {
            value.clamp(1, 100)
        }
    }

    /// Load vocabulary terms from a `.murmur-vocab` file if it exists in the given directory.
    /// The file contains one term per line. Empty lines and lines starting with # are ignored.
    pub fn load_vocab_file(dir: &std::path::Path) -> Vec<String> {
        let path = dir.join(".murmur-vocab");
        match std::fs::read_to_string(&path) {
            Ok(contents) => contents
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(String::from)
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Collect all effective vocabulary: global config + app-specific + vocab file.
    pub fn effective_vocabulary(
        &self,
        app_id: Option<&str>,
        project_dir: Option<&std::path::Path>,
    ) -> Vec<String> {
        let mut vocab: Vec<String> = self.vocabulary.clone();

        if let Some(id) = app_id {
            if let Some(app_ctx) = self.app_contexts.get(id) {
                vocab.extend(app_ctx.vocabulary.iter().cloned());
            }
        }

        if let Some(dir) = project_dir {
            vocab.extend(Self::load_vocab_file(dir));
        }

        // Deduplicate while preserving order
        let mut seen = std::collections::HashSet::new();
        vocab.retain(|v| seen.insert(v.clone()));
        vocab
    }

    /// Check if an app is excluded from context capture.
    pub fn is_app_excluded(&self, app_id: &str) -> bool {
        self.excluded_apps.iter().any(|e| e == app_id)
    }

    /// Get the effective dictation mode for a given app.
    pub fn effective_dictation_mode(&self, app_id: Option<&str>) -> DictationMode {
        if let Some(id) = app_id {
            if let Some(ctx) = self.app_contexts.get(id) {
                if let Some(mode) = ctx.mode {
                    return mode;
                }
            }
        }
        self.dictation_mode
    }

    /// Whether the current backend produces pre-formatted output (punctuation, capitalization).
    pub fn backend_has_native_formatting(&self) -> bool {
        matches!(self.asr_backend, AsrBackend::Parakeet)
    }

    /// Whether the app is in Notes mode.
    pub fn is_notes_mode(&self) -> bool {
        self.app_mode == AppMode::Notes
    }

    /// Resolved notes directory, falling back to data_dir/murmur/notes.
    pub fn notes_dir(&self) -> PathBuf {
        self.notes_dir.clone().unwrap_or_else(|| {
            dirs::data_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("murmur")
                .join("notes")
        })
    }

    /// Default model size for the current ASR backend.
    pub fn default_model_for_backend(&self) -> &'static str {
        match self.asr_backend {
            AsrBackend::Whisper => "base.en",
            AsrBackend::Qwen3Asr => "0.6b",
            AsrBackend::Parakeet => "0.6b-v2",
            AsrBackend::Mlx => "0.6b",
        }
    }
}
