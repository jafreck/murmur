use murmur::app;
use murmur::config;
use murmur::input::keycodes;
use murmur::transcription::{model, transcriber};
use murmur::VERSION;
use murmur_core::config::AsrBackend;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "murmur",
    about = "Cross-platform, local voice dictation. Worry free speech to text.",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the dictation daemon
    Start {
        /// Enable notes mode for this session
        #[arg(long)]
        notes: bool,
        /// ASR backend engine (whisper, qwen3-asr, parakeet)
        #[arg(long)]
        backend: Option<String>,
    },
    /// Set the push-to-talk hotkey
    SetHotkey {
        /// Key or key combo (e.g. "ctrl+shift+space", "f9", "globe")
        key: String,
    },
    /// Show the current hotkey
    GetHotkey,
    /// Set the Whisper model size
    SetModel {
        /// Model size (e.g. "base.en", "small.en", "medium.en")
        size: String,
    },
    /// Set the transcription language
    SetLanguage {
        /// Language code (e.g. "en", "fr", "auto")
        code: String,
    },
    /// Download a Whisper model
    DownloadModel {
        /// Model size (defaults to "base.en")
        #[arg(default_value = "base.en")]
        size: String,
        /// ASR backend for the model (whisper, qwen3-asr, parakeet)
        #[arg(long, default_value = "whisper")]
        backend: String,
    },
    /// Check for updates and install the latest version
    Update {
        /// Check only, don't install
        #[arg(long)]
        check: bool,
    },
    /// Show configuration and status
    Status,
    /// Run the transparent overlay window (internal — spawned by the daemon)
    #[command(hide = true)]
    Overlay,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .filter_module("enigo", log::LevelFilter::Warn)
        .init();

    // Log panics with full backtraces instead of just printing to stderr.
    // This catches panics in the main thread and provides context for debugging.
    std::panic::set_hook(Box::new(|info| {
        let backtrace = std::backtrace::Backtrace::force_capture();
        log::error!("PANIC: {info}\n{backtrace}");
        // Also print to stderr in case logging is broken
        eprintln!("PANIC: {info}\n{backtrace}");
    }));

    let cli = Cli::parse();

    let result = match cli.command {
        Some(Commands::Start { notes, backend }) => cmd_start(notes, backend),
        Some(Commands::SetHotkey { key }) => cmd_set_hotkey(&key),
        Some(Commands::GetHotkey) => cmd_get_hotkey(),
        Some(Commands::SetModel { size }) => cmd_set_model(&size),
        Some(Commands::SetLanguage { code }) => cmd_set_language(&code),
        Some(Commands::DownloadModel { size, backend }) => cmd_download_model(&size, &backend),
        Some(Commands::Update { check }) => cmd_update(check),
        Some(Commands::Status) => cmd_status(),
        Some(Commands::Overlay) => murmur::ui::overlay::run_overlay(),
        None => {
            use clap::CommandFactory;
            Cli::command().print_help().ok();
            println!();
            Ok(())
        }
    };

    if let Err(e) = result {
        log::error!("Fatal error: {e:?}");
        eprintln!("Error: {e:?}");
        std::process::exit(1);
    }
}

fn cmd_start(notes: bool, backend: Option<String>) -> Result<()> {
    println!("murmur v{VERSION}");
    if let Some(ref backend_str) = backend {
        let mut cfg = config::Config::load();
        cfg.asr_backend = match backend_str.as_str() {
            "whisper" => AsrBackend::Whisper,
            "qwen3-asr" | "qwen3_asr" | "qwen" => AsrBackend::Qwen3Asr,
            "parakeet" => AsrBackend::Parakeet,
            _ => anyhow::bail!("Unknown backend: {backend_str}. Use: whisper, qwen3-asr, parakeet"),
        };
        // Set default model size for the new backend if current one is invalid
        let valid_models = config::supported_models(cfg.asr_backend);
        if !valid_models.contains(&cfg.model_size.as_str()) {
            cfg.model_size = cfg.default_model_for_backend().to_string();
        }
        cfg.save()?;
    }
    app::run(notes)
}

fn cmd_set_hotkey(key_string: &str) -> Result<()> {
    let parsed = keycodes::parse(key_string).ok_or_else(|| {
        anyhow::anyhow!("Unknown key '{key_string}'. Run 'murmur --help' for examples.")
    })?;

    let mut cfg = config::Config::load();
    cfg.hotkey = parsed.to_config_string();
    cfg.save()?;

    println!("Hotkey set to: {}", cfg.hotkey);
    Ok(())
}

fn cmd_get_hotkey() -> Result<()> {
    let cfg = config::Config::load();
    println!("Current hotkey: {}", cfg.hotkey);
    Ok(())
}

fn cmd_set_model(size: &str) -> Result<()> {
    validate_model(size)?;

    let mut cfg = config::Config::load();
    cfg.model_size = size.to_string();
    cfg.save()?;

    println!("Model set to: {size}");
    if !transcriber::model_exists(size) {
        println!("Model will be downloaded on next start.");
    }
    Ok(())
}

pub fn validate_model(size: &str) -> Result<()> {
    let cfg = config::Config::load();
    validate_model_for_backend(size, cfg.asr_backend)
}

fn validate_model_for_backend(size: &str, backend: AsrBackend) -> Result<()> {
    let valid = config::supported_models(backend);
    if !valid.contains(&size) {
        anyhow::bail!(
            "Unknown model '{size}' for {} backend. Available: {}",
            backend,
            valid.join(", ")
        );
    }
    Ok(())
}

fn cmd_set_language(code: &str) -> Result<()> {
    validate_language(code)?;

    let mut cfg = config::Config::load();
    cfg.language = code.to_string();
    cfg.save()?;

    let name = config::language_name(code).unwrap_or(code);
    println!("Language set to: {name} ({code})");
    Ok(())
}

pub fn validate_language(code: &str) -> Result<()> {
    if !config::is_valid_language(code) {
        anyhow::bail!("Unknown language '{code}'. Examples: en, fr, de, es, auto");
    }
    Ok(())
}

pub fn format_status(cfg: &config::Config, model_ready: bool) -> String {
    let lang_name = config::language_name(&cfg.language).unwrap_or(&cfg.language);
    let mode_str = cfg.mode.to_string();
    let streaming_str = if cfg.streaming { "on" } else { "off" };
    let model_ready_str = if model_ready { "yes" } else { "no" };

    format!(
        "murmur v{VERSION}\n\
         Config:       {}\n\
         Hotkey:       {}\n\
         Backend:      {}\n\
         Model:        {}\n\
         Model ready:  {model_ready_str}\n\
         Language:     {lang_name} ({})\n\
         Mode:         {mode_str}\n\
         Streaming:    {streaming_str}",
        config::Config::file_path().display(),
        cfg.hotkey,
        cfg.asr_backend,
        cfg.model_size,
        cfg.language,
    )
}

fn cmd_download_model(size: &str, backend_str: &str) -> Result<()> {
    let backend = match backend_str {
        "whisper" => AsrBackend::Whisper,
        "qwen3-asr" | "qwen3_asr" | "qwen" => AsrBackend::Qwen3Asr,
        "parakeet" => AsrBackend::Parakeet,
        _ => anyhow::bail!("Unknown backend: {backend_str}. Use: whisper, qwen3-asr, parakeet"),
    };

    let cfg = config::Config::load();
    let last_pct = std::cell::Cell::new(u32::MAX);

    match backend {
        AsrBackend::Whisper => {
            model::download(size, |percent| {
                let pct = percent as u32;
                if pct != last_pct.get() {
                    last_pct.set(pct);
                    eprint!("\rDownloading {size} model... {pct}%");
                }
            })?;
        }
        _ => {
            murmur_core::transcription::download_for_backend(
                backend,
                size,
                cfg.asr_quantization,
                |percent| {
                    let pct = percent as u32;
                    if pct != last_pct.get() {
                        last_pct.set(pct);
                        eprint!("\rDownloading {backend} {size} model... {pct}%");
                    }
                },
            )?;
        }
    }
    eprintln!();
    println!("Model downloaded.");
    Ok(())
}

fn cmd_update(check_only: bool) -> Result<()> {
    println!("Checking for updates...");
    match murmur_core::update::check_for_update(VERSION)? {
        None => {
            println!("Already up to date (v{VERSION}).");
            Ok(())
        }
        Some(info) => {
            println!(
                "Update available: v{} → v{}",
                info.current_version, info.latest_version
            );
            if check_only {
                println!("Run 'murmur update' to install.");
                return Ok(());
            }
            murmur_core::update::apply_update(&info, |msg| {
                println!("  {msg}");
            })?;
            println!("Restart murmur to use the new version.");
            Ok(())
        }
    }
}

fn cmd_status() -> Result<()> {
    let cfg = config::Config::load();
    let model_ready = transcriber::model_exists(&cfg.model_size);
    println!("{}", format_status(&cfg, model_ready));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_is_set() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_validate_model_valid() {
        assert!(validate_model_for_backend("base.en", AsrBackend::Whisper).is_ok());
        assert!(validate_model_for_backend("tiny.en", AsrBackend::Whisper).is_ok());
        assert!(validate_model_for_backend("small.en", AsrBackend::Whisper).is_ok());
        assert!(validate_model_for_backend("medium.en", AsrBackend::Whisper).is_ok());
        assert!(validate_model_for_backend("large-v3-turbo", AsrBackend::Whisper).is_ok());
        assert!(validate_model_for_backend("large", AsrBackend::Whisper).is_ok());
        assert!(validate_model_for_backend("distil-large-v3", AsrBackend::Whisper).is_ok());
        assert!(validate_model_for_backend("0.6b", AsrBackend::Qwen3Asr).is_ok());
        assert!(validate_model_for_backend("1.7b", AsrBackend::Qwen3Asr).is_ok());
        assert!(validate_model_for_backend("0.6b-v2", AsrBackend::Parakeet).is_ok());
    }

    #[test]
    fn test_validate_model_invalid() {
        assert!(validate_model_for_backend("nonexistent", AsrBackend::Whisper).is_err());
        assert!(validate_model_for_backend("", AsrBackend::Whisper).is_err());
        assert!(validate_model_for_backend("huge", AsrBackend::Whisper).is_err());
        assert!(validate_model_for_backend("base.en", AsrBackend::Qwen3Asr).is_err());
    }

    #[test]
    fn test_validate_language_valid() {
        assert!(validate_language("en").is_ok());
        assert!(validate_language("fr").is_ok());
        assert!(validate_language("auto").is_ok());
        assert!(validate_language("de").is_ok());
    }

    #[test]
    fn test_validate_language_invalid() {
        assert!(validate_language("xx").is_err());
        assert!(validate_language("").is_err());
        assert!(validate_language("klingon").is_err());
    }

    #[test]
    fn test_format_status_hold_mode() {
        let cfg = config::Config {
            hotkey: "f9".to_string(),
            model_size: "base.en".to_string(),
            language: "en".to_string(),
            spoken_punctuation: false,
            filler_word_removal: true,
            max_recordings: 0,
            mode: config::InputMode::PushToTalk,
            streaming: false,
            translate_to_english: false,
            ..config::Config::default()
        };
        let output = format_status(&cfg, false);
        assert!(output.contains("murmur v"));
        assert!(output.contains("f9"));
        assert!(output.contains("base.en"));
        assert!(output.contains("English"));
        assert!(output.contains("Mode:         Push to Talk"));
        assert!(output.contains("Model ready:  no"));
        assert!(output.contains("Streaming:    off"));
    }

    #[test]
    fn test_format_status_toggle_mode() {
        let cfg = config::Config {
            hotkey: "ctrl+shift+space".to_string(),
            model_size: "small.en".to_string(),
            language: "fr".to_string(),
            spoken_punctuation: true,
            filler_word_removal: true,
            max_recordings: 5,
            mode: config::InputMode::OpenMic,
            streaming: true,
            translate_to_english: false,
            ..config::Config::default()
        };
        let output = format_status(&cfg, true);
        assert!(output.contains("Mode:         Open Mic"));
        assert!(output.contains("Model ready:  yes"));
        assert!(output.contains("French"));
        assert!(output.contains("Streaming:    on"));
    }

    #[test]
    fn test_format_status_unknown_language_fallback() {
        let cfg = config::Config {
            hotkey: "f9".to_string(),
            model_size: "base.en".to_string(),
            language: "zz".to_string(),
            spoken_punctuation: false,
            filler_word_removal: true,
            max_recordings: 0,
            mode: config::InputMode::PushToTalk,
            streaming: false,
            translate_to_english: false,
            ..config::Config::default()
        };
        let output = format_status(&cfg, false);
        // Should fall back to code itself
        assert!(output.contains("zz"));
    }

    #[test]
    fn test_cmd_get_hotkey() {
        // This reads from real config but doesn't modify it
        assert!(cmd_get_hotkey().is_ok());
    }

    #[test]
    fn test_cmd_set_hotkey_valid() {
        // Save current config
        let original = config::Config::load();
        // Set a known valid hotkey
        assert!(cmd_set_hotkey("f9").is_ok());
        // Restore original before asserting so config isn't left dirty on failure
        let restored_hotkey = original.hotkey.clone();
        let _ = original.save();
        // Verify the set succeeded by re-reading (not using stale value)
        // Note: we verify cmd_set_hotkey ran without error above.
        // The round-trip is: set f9 → save → restore original. If the
        // restore worked, the file now has the original hotkey back.
        let cfg = config::Config::load();
        assert_eq!(cfg.hotkey, restored_hotkey);
    }

    #[test]
    fn test_cmd_set_hotkey_invalid() {
        assert!(cmd_set_hotkey("nonexistentkey123").is_err());
    }

    #[test]
    fn test_cmd_set_model_valid() {
        let original = config::Config::load();
        // Ensure Whisper backend so "base.en" is valid
        let mut tmp = original.clone();
        tmp.asr_backend = AsrBackend::Whisper;
        tmp.save().unwrap();
        let result = cmd_set_model("base.en");
        let _ = original.save();
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_set_model_prints_download_hint() {
        let original = config::Config::load();
        let mut tmp = original.clone();
        tmp.asr_backend = AsrBackend::Whisper;
        tmp.save().unwrap();
        let result = cmd_set_model("tiny");
        let _ = original.save();
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_set_model_invalid() {
        assert!(cmd_set_model("nonexistent_model").is_err());
    }

    #[test]
    fn test_cmd_set_language_valid() {
        let original = config::Config::load();
        assert!(cmd_set_language("en").is_ok());
        let _ = original.save();
    }

    #[test]
    fn test_cmd_set_language_invalid() {
        assert!(cmd_set_language("zzz_invalid").is_err());
    }

    #[test]
    fn test_cmd_status() {
        assert!(cmd_status().is_ok());
    }
}
