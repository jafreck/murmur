use serde::{Deserialize, Serialize};

/// Top-level application mode.
///
/// **Dictation** — transcription is pasted at the cursor via hotkey.
/// **Notes** — transcription is shown in an overlay and saved to note files,
/// triggered by wake word.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppMode {
    #[default]
    Dictation,
    Notes,
}

impl std::fmt::Display for AppMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppMode::Dictation => write!(f, "Dictation"),
            AppMode::Notes => write!(f, "Notes"),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputMode {
    #[default]
    PushToTalk,
    OpenMic,
}

impl std::fmt::Display for InputMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InputMode::PushToTalk => write!(f, "Push to Talk"),
            InputMode::OpenMic => write!(f, "Open Mic"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DictationMode {
    #[default]
    Prose,
    Code,
    Command,
    List,
}

impl std::fmt::Display for DictationMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DictationMode::Prose => write!(f, "Prose"),
            DictationMode::Code => write!(f, "Code"),
            DictationMode::Command => write!(f, "Command"),
            DictationMode::List => write!(f, "List"),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AppContextConfig {
    /// Vocabulary terms to bias Whisper toward when this app is focused
    #[serde(default)]
    pub vocabulary: Vec<String>,
    /// Dictation mode to use when this app is focused
    #[serde(default)]
    pub mode: Option<DictationMode>,
}

/// ASR backend engine.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AsrBackend {
    Whisper,
    #[default]
    Qwen3Asr,
    Parakeet,
    Mlx,
}

impl std::fmt::Display for AsrBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AsrBackend::Whisper => write!(f, "Whisper"),
            AsrBackend::Qwen3Asr => write!(f, "Qwen3-ASR"),
            AsrBackend::Parakeet => write!(f, "Parakeet"),
            AsrBackend::Mlx => write!(f, "MLX"),
        }
    }
}

impl AsrBackend {
    /// Whether this backend supports native streaming (re-transcribe the
    /// full accumulated buffer each tick). Backends that return `true` do
    /// not need the Whisper subprocess worker or chunked overlap stitching.
    pub fn supports_native_streaming(self) -> bool {
        !matches!(self, AsrBackend::Whisper)
    }
}

/// ONNX model quantization level.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AsrQuantization {
    Fp32,
    #[default]
    Int4,
    Int8,
}

impl std::fmt::Display for AsrQuantization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AsrQuantization::Fp32 => write!(f, "FP32"),
            AsrQuantization::Int4 => write!(f, "INT4"),
            AsrQuantization::Int8 => write!(f, "INT8"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub(crate) hotkey: String,
    pub(crate) model_size: String,
    /// ASR backend engine (default: whisper)
    #[serde(default)]
    pub(crate) asr_backend: AsrBackend,
    /// ONNX model quantization level (default: int4, only used for ONNX backends)
    #[serde(default)]
    pub(crate) asr_quantization: AsrQuantization,
    pub(crate) language: String,
    #[serde(default)]
    pub(crate) spoken_punctuation: bool,
    #[serde(default)]
    pub(crate) filler_word_removal: bool,
    #[serde(default)]
    pub(crate) max_recordings: u32,
    #[serde(default)]
    pub(crate) mode: InputMode,
    #[serde(default)]
    pub(crate) streaming: bool,
    #[serde(default)]
    pub(crate) translate_to_english: bool,
    /// Enable noise suppression via nnnoiseless (default: true)
    #[serde(default = "default_true")]
    pub(crate) noise_suppression: bool,
    /// Global vocabulary terms to bias Whisper toward
    #[serde(default)]
    pub(crate) vocabulary: Vec<String>,
    /// Per-application context configurations, keyed by bundle ID or process name
    #[serde(default)]
    pub(crate) app_contexts: std::collections::HashMap<String, AppContextConfig>,
    /// App identifiers to exclude from context capture (password managers, banking apps)
    #[serde(default)]
    pub(crate) excluded_apps: Vec<String>,
    /// Default dictation mode
    #[serde(default)]
    pub(crate) dictation_mode: DictationMode,
    /// Application mode: `dictation` (paste at cursor) or `notes` (overlay + wake word)
    #[serde(default)]
    pub(crate) app_mode: AppMode,
    /// Phrase that triggers dictation when spoken (default: "murmur start dictation")
    #[serde(default = "default_wake_word")]
    pub(crate) wake_word: String,
    /// Phrase that stops dictation when spoken (default: "murmur stop dictation")
    #[serde(default = "default_stop_phrase")]
    pub(crate) stop_phrase: String,
    /// Directory for saving dictation notes (default: data_dir/murmur/notes)
    #[serde(default)]
    pub(crate) notes_dir: Option<std::path::PathBuf>,
    /// Input device name for system audio capture (e.g. "BlackHole 2ch").
    /// When set, meeting sessions capture both mic and system audio.
    #[serde(default)]
    pub(crate) system_audio_device: Option<String>,
    /// Hide the overlay window from screen capture and screen sharing.
    #[serde(default)]
    pub(crate) stealth_mode: bool,
    /// LLM model name for Ollama (default: "phi3")
    #[serde(default = "default_llm_model")]
    pub(crate) llm_model: String,
    /// Ollama API base URL
    #[serde(default = "default_ollama_url")]
    pub(crate) ollama_url: String,
    /// Directory for storing meeting sessions
    #[serde(default)]
    pub(crate) sessions_dir: Option<String>,
    /// Auto-generate summary when meeting ends
    #[serde(default)]
    pub(crate) auto_summary: bool,
    /// Automatically check for and apply updates on startup (default: false)
    #[serde(default)]
    pub(crate) auto_update: bool,
}

// ── Accessor methods ────────────────────────────────────────────────────

impl Config {
    // Getters

    pub fn hotkey(&self) -> &str {
        &self.hotkey
    }
    pub fn model_size(&self) -> &str {
        &self.model_size
    }
    pub fn asr_backend(&self) -> AsrBackend {
        self.asr_backend
    }
    pub fn asr_quantization(&self) -> AsrQuantization {
        self.asr_quantization
    }
    pub fn language(&self) -> &str {
        &self.language
    }
    pub fn spoken_punctuation(&self) -> bool {
        self.spoken_punctuation
    }
    pub fn filler_word_removal(&self) -> bool {
        self.filler_word_removal
    }
    pub fn max_recordings(&self) -> u32 {
        self.max_recordings
    }
    pub fn mode(&self) -> &InputMode {
        &self.mode
    }
    pub fn streaming(&self) -> bool {
        self.streaming
    }
    pub fn translate_to_english(&self) -> bool {
        self.translate_to_english
    }
    pub fn noise_suppression(&self) -> bool {
        self.noise_suppression
    }
    pub fn vocabulary(&self) -> &[String] {
        &self.vocabulary
    }

    pub fn app_contexts(&self) -> &std::collections::HashMap<String, AppContextConfig> {
        &self.app_contexts
    }

    pub fn excluded_apps(&self) -> &[String] {
        &self.excluded_apps
    }
    pub fn dictation_mode(&self) -> DictationMode {
        self.dictation_mode
    }
    pub fn app_mode(&self) -> AppMode {
        self.app_mode
    }
    pub fn wake_word(&self) -> &str {
        &self.wake_word
    }
    pub fn stop_phrase(&self) -> &str {
        &self.stop_phrase
    }
    pub fn system_audio_device(&self) -> Option<&str> {
        self.system_audio_device.as_deref()
    }
    pub fn stealth_mode(&self) -> bool {
        self.stealth_mode
    }
    pub fn llm_model(&self) -> &str {
        &self.llm_model
    }
    pub fn ollama_url(&self) -> &str {
        &self.ollama_url
    }
    pub fn sessions_dir(&self) -> Option<&str> {
        self.sessions_dir.as_deref()
    }
    pub fn auto_summary(&self) -> bool {
        self.auto_summary
    }
    pub fn auto_update(&self) -> bool {
        self.auto_update
    }

    // Setters

    pub fn set_hotkey(&mut self, val: String) {
        self.hotkey = val;
    }
    pub fn set_model_size(&mut self, val: String) {
        self.model_size = val;
    }
    pub fn set_asr_backend(&mut self, val: AsrBackend) {
        self.asr_backend = val;
    }
    pub fn set_asr_quantization(&mut self, val: AsrQuantization) {
        self.asr_quantization = val;
    }
    pub fn set_language(&mut self, val: String) {
        self.language = val;
    }
    pub fn set_spoken_punctuation(&mut self, val: bool) {
        self.spoken_punctuation = val;
    }
    pub fn set_filler_word_removal(&mut self, val: bool) {
        self.filler_word_removal = val;
    }
    pub fn set_max_recordings(&mut self, val: u32) {
        self.max_recordings = val;
    }
    pub fn set_mode(&mut self, val: InputMode) {
        self.mode = val;
    }
    pub fn set_streaming(&mut self, val: bool) {
        self.streaming = val;
    }
    pub fn set_translate_to_english(&mut self, val: bool) {
        self.translate_to_english = val;
    }
    pub fn set_noise_suppression(&mut self, val: bool) {
        self.noise_suppression = val;
    }
    pub fn set_vocabulary(&mut self, val: Vec<String>) {
        self.vocabulary = val;
    }

    pub fn set_app_contexts(&mut self, val: std::collections::HashMap<String, AppContextConfig>) {
        self.app_contexts = val;
    }

    pub fn set_excluded_apps(&mut self, val: Vec<String>) {
        self.excluded_apps = val;
    }
    pub fn set_dictation_mode(&mut self, val: DictationMode) {
        self.dictation_mode = val;
    }
    pub fn set_app_mode(&mut self, val: AppMode) {
        self.app_mode = val;
    }
    pub fn set_wake_word(&mut self, val: String) {
        self.wake_word = val;
    }
    pub fn set_stop_phrase(&mut self, val: String) {
        self.stop_phrase = val;
    }

    pub fn set_notes_dir(&mut self, val: Option<std::path::PathBuf>) {
        self.notes_dir = val;
    }

    pub fn set_system_audio_device(&mut self, val: Option<String>) {
        self.system_audio_device = val;
    }

    pub fn set_stealth_mode(&mut self, val: bool) {
        self.stealth_mode = val;
    }
    pub fn set_llm_model(&mut self, val: String) {
        self.llm_model = val;
    }
    pub fn set_ollama_url(&mut self, val: String) {
        self.ollama_url = val;
    }
    pub fn set_sessions_dir(&mut self, val: Option<String>) {
        self.sessions_dir = val;
    }
    pub fn set_auto_summary(&mut self, val: bool) {
        self.auto_summary = val;
    }
    pub fn set_auto_update(&mut self, val: bool) {
        self.auto_update = val;
    }
}

fn default_true() -> bool {
    true
}

pub(crate) fn default_wake_word() -> String {
    "murmur start dictation".to_string()
}

pub(crate) fn default_stop_phrase() -> String {
    "murmur stop dictation".to_string()
}

pub(crate) fn default_llm_model() -> String {
    "phi3".to_string()
}

pub(crate) fn default_ollama_url() -> String {
    "http://localhost:11434".to_string()
}

pub(crate) fn default_hotkey() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "rightoption"
    }
    #[cfg(not(target_os = "macos"))]
    {
        "rightalt"
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hotkey: default_hotkey().to_string(),
            model_size: "0.6b".to_string(),
            asr_backend: AsrBackend::default(),
            asr_quantization: AsrQuantization::default(),
            language: "en".to_string(),
            spoken_punctuation: false,
            filler_word_removal: false,
            max_recordings: 0,
            mode: InputMode::PushToTalk,
            streaming: false,
            translate_to_english: false,
            noise_suppression: true,
            vocabulary: Vec::new(),
            app_contexts: std::collections::HashMap::new(),
            excluded_apps: Vec::new(),
            dictation_mode: DictationMode::default(),
            app_mode: AppMode::default(),
            wake_word: default_wake_word(),
            stop_phrase: default_stop_phrase(),
            notes_dir: None,
            system_audio_device: None,
            stealth_mode: false,
            llm_model: default_llm_model(),
            ollama_url: default_ollama_url(),
            sessions_dir: None,
            auto_summary: false,
            auto_update: false,
        }
    }
}
