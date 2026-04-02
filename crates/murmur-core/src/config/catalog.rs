use super::schema::AsrBackend;

pub const WHISPER_MODELS: &[&str] = &[
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large-v3-turbo",
    "large",
    "distil-large-v3",
];

pub const QWEN3_ASR_MODELS: &[&str] = &["0.6b", "1.7b"];

pub const PARAKEET_MODELS: &[&str] = &["0.6b-v2"];

pub const MLX_MODELS: &[&str] = &["0.6b", "1.7b"];

/// All supported models for a given backend.
pub fn supported_models(backend: AsrBackend) -> &'static [&'static str] {
    match backend {
        AsrBackend::Whisper => WHISPER_MODELS,
        AsrBackend::Qwen3Asr => QWEN3_ASR_MODELS,
        AsrBackend::Parakeet => PARAKEET_MODELS,
        AsrBackend::Mlx => MLX_MODELS,
    }
}

/// Deprecated: use `WHISPER_MODELS` or `supported_models()` instead.
pub const SUPPORTED_MODELS: &[&str] = WHISPER_MODELS;

/// Returns true for models that only support English (`.en` suffix or `distil-*`).
pub fn is_english_only_model(model: &str) -> bool {
    model.ends_with(".en") || model.starts_with("distil-")
}

pub const SUPPORTED_LANGUAGES: &[(&str, &str)] = &[
    ("auto", "Auto-Detect"),
    ("en", "English"),
    ("zh", "Chinese"),
    ("de", "German"),
    ("es", "Spanish"),
    ("ru", "Russian"),
    ("ko", "Korean"),
    ("fr", "French"),
    ("ja", "Japanese"),
    ("pt", "Portuguese"),
    ("tr", "Turkish"),
    ("pl", "Polish"),
    ("nl", "Dutch"),
    ("ar", "Arabic"),
    ("sv", "Swedish"),
    ("it", "Italian"),
    ("id", "Indonesian"),
    ("hi", "Hindi"),
    ("fi", "Finnish"),
    ("vi", "Vietnamese"),
    ("he", "Hebrew"),
    ("uk", "Ukrainian"),
    ("el", "Greek"),
    ("cs", "Czech"),
    ("ro", "Romanian"),
    ("da", "Danish"),
    ("hu", "Hungarian"),
    ("no", "Norwegian"),
    ("th", "Thai"),
    ("ca", "Catalan"),
    ("sk", "Slovak"),
    ("hr", "Croatian"),
    ("bg", "Bulgarian"),
    ("lt", "Lithuanian"),
    ("sl", "Slovenian"),
    ("et", "Estonian"),
    ("lv", "Latvian"),
    ("sr", "Serbian"),
    ("mk", "Macedonian"),
    ("ta", "Tamil"),
    ("te", "Telugu"),
    ("ml", "Malayalam"),
    ("kn", "Kannada"),
    ("bn", "Bengali"),
    ("mr", "Marathi"),
    ("gu", "Gujarati"),
    ("pa", "Punjabi"),
    ("ur", "Urdu"),
    ("fa", "Persian"),
    ("sw", "Swahili"),
    ("af", "Afrikaans"),
    ("ms", "Malay"),
    ("az", "Azerbaijani"),
    ("sq", "Albanian"),
    ("hy", "Armenian"),
    ("ka", "Georgian"),
    ("ne", "Nepali"),
    ("mn", "Mongolian"),
    ("bs", "Bosnian"),
    ("kk", "Kazakh"),
    ("gl", "Galician"),
    ("eu", "Basque"),
    ("is", "Icelandic"),
    ("cy", "Welsh"),
    ("la", "Latin"),
    ("haw", "Hawaiian"),
    ("jw", "Javanese"),
];

pub fn is_valid_language(code: &str) -> bool {
    SUPPORTED_LANGUAGES.iter().any(|(c, _)| *c == code)
}

pub fn language_name(code: &str) -> Option<&str> {
    SUPPORTED_LANGUAGES
        .iter()
        .find(|(c, _)| *c == code)
        .map(|(_, name)| *name)
}
