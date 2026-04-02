use crate::ui::tray::TrayState;
use murmur_core::transcription::AsrEngine;
use rdev::Key;
use std::sync::Arc;

pub enum AppMessage {
    KeyDown,
    KeyUp,
    TrayQuit,
    TrayCopyLast,
    TraySetModel(String),
    TraySetLanguage(String),
    TraySetBackend(murmur_core::config::AsrBackend),
    TrayToggleSpokenPunctuation,
    TrayToggleFillerWordRemoval,
    TraySetMode(crate::config::InputMode),
    TrayToggleStreaming,
    TrayToggleTranslate,
    TrayToggleNoiseSuppression,
    TrayOpenConfig,
    TrayReloadConfig,
    TraySetHotkey,
    TrayToggleAppMode,
    TrayCheckForUpdates,
    HotkeyCapture(Key),
    TranscriptionDone(String),
    TranscriptionError(String),
    StreamingPartialText {
        text: String,
        replace_chars: usize,
    },
    /// A new ASR engine has been loaded in the background and is ready to swap in.
    /// The u64 is the reload generation; stale reloads are discarded.
    EngineReady(Arc<dyn AsrEngine + Send + Sync>, u64),
    /// The wake word was detected — start dictation.
    WakeWordDetected,
    /// The stop phrase was detected — stop dictation.
    StopPhraseDetected,
    /// VAD detected speech in the audio stream (keeps silence timeout alive).
    SpeechActivity,
    /// A background update check found a newer version.
    UpdateAvailable(murmur_core::update::UpdateInfo),
    /// An update has been applied; show a notification.
    UpdateApplied(String),
    /// Update check or apply failed.
    UpdateError(String),
}

/// Effect returned by the app state machine in response to a message.
#[derive(Debug, Clone, PartialEq)]
pub enum AppEffect {
    None,
    StartRecording(std::path::PathBuf),
    StopAndTranscribe,
    /// Start streaming transcription (toggle mode only).
    StartStreaming,
    /// Stop the streaming transcription thread.
    StopStreaming,
    InsertText(String),
    /// Backspace `replace_chars` characters, then type `text` (streaming revisions).
    StreamingReplace {
        text: String,
        replace_chars: usize,
    },
    CopyToClipboard(String),
    SaveConfig,
    /// Open the config file in the user's default editor/file manager.
    OpenConfig,
    /// Reload config from disk and apply changes.
    ReloadConfig,
    SetTrayState(TrayState),
    SetTrayModel(String),
    SetTrayLanguage(String),
    /// Enable or disable the language submenu (disabled for English-only models).
    SetLanguageMenuEnabled(bool),
    SetTrayMode(crate::config::InputMode),
    /// Update the ASR backend and reset the model to the backend's default.
    SetBackend(murmur_core::config::AsrBackend),
    /// Download the model if needed and rebuild the Transcriber in a background thread.
    ReloadTranscriber(u64),
    /// Enter hotkey capture mode — the listener should capture the next key press.
    EnterHotkeyCaptureMode,
    /// Save the captured hotkey and update the listener.
    SetHotkey(String),
    /// Update the noise suppression state on the audio recorder.
    UpdateNoiseSuppression(bool),
    /// Show the overlay window.
    ShowOverlay,
    /// Hide the overlay window.
    HideOverlay,
    /// Update the overlay with new text.
    UpdateOverlayText(String),
    /// Save the current overlay text to a note file.
    SaveNote(String),
    /// Pause the wake word detector (during active dictation).
    PauseWakeWord,
    /// Resume the wake word detector (after dictation ends).
    ResumeWakeWord,
    /// Start the wake word detector.
    StartWakeWord,
    /// Stop the wake word detector.
    StopWakeWord,
    /// Spawn the overlay subprocess.
    SpawnOverlay,
    /// Kill the overlay subprocess.
    KillOverlay,
    Quit,
    LogError(String),
}
