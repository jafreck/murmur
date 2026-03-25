mod app;
mod audio;
mod config;
mod hotkey;
mod inserter;
mod keycodes;
mod model;
mod permissions;
mod postprocess;
mod recordings;
mod streaming;
mod transcriber;
mod tray;

use anyhow::Result;
use clap::{Parser, Subcommand};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser)]
#[command(
    name = "open-bark",
    about = "Cross-platform, local voice dictation. Hold a key, speak, release — your words appear at the cursor.",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the dictation daemon
    Start,
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
    },
    /// Show configuration and status
    Status,
}

fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Start) => cmd_start(),
        Some(Commands::SetHotkey { key }) => cmd_set_hotkey(&key),
        Some(Commands::GetHotkey) => cmd_get_hotkey(),
        Some(Commands::SetModel { size }) => cmd_set_model(&size),
        Some(Commands::SetLanguage { code }) => cmd_set_language(&code),
        Some(Commands::DownloadModel { size }) => cmd_download_model(&size),
        Some(Commands::Status) => cmd_status(),
        None => {
            // No subcommand: print help
            use clap::CommandFactory;
            Cli::command().print_help()?;
            println!();
            Ok(())
        }
    }
}

fn cmd_start() -> Result<()> {
    println!("open-bark v{VERSION}");
    app::run()
}

fn cmd_set_hotkey(key_string: &str) -> Result<()> {
    let parsed = keycodes::parse(key_string)
        .ok_or_else(|| anyhow::anyhow!("Unknown key '{key_string}'. Run 'open-bark --help' for examples."))?;

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
    if !config::SUPPORTED_MODELS.contains(&size) {
        anyhow::bail!(
            "Unknown model '{size}'. Available: {}",
            config::SUPPORTED_MODELS.join(", ")
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
    let push_to_talk_str = if cfg.push_to_talk {
        "on (hold to talk)"
    } else {
        "off (press to start/stop)"
    };
    let streaming_str = if cfg.streaming { "on" } else { "off" };
    let model_ready_str = if model_ready { "yes" } else { "no" };

    format!(
        "open-bark v{VERSION}\n\
         Config:       {}\n\
         Hotkey:       {}\n\
         Model:        {}\n\
         Model ready:  {model_ready_str}\n\
         Language:     {lang_name} ({})\n\
         Push to Talk: {push_to_talk_str}\n\
         Streaming:    {streaming_str}",
        config::Config::file_path().display(),
        cfg.hotkey,
        cfg.model_size,
        cfg.language,
    )
}

fn cmd_download_model(size: &str) -> Result<()> {
    model::download(size, |percent| {
        eprint!("\rDownloading {size} model... {percent:.0}%");
    })?;
    eprintln!();
    println!("Model downloaded.");
    Ok(())
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
        assert!(validate_model("base.en").is_ok());
        assert!(validate_model("tiny.en").is_ok());
        assert!(validate_model("small.en").is_ok());
        assert!(validate_model("medium.en").is_ok());
        assert!(validate_model("large-v3-turbo").is_ok());
        assert!(validate_model("large").is_ok());
    }

    #[test]
    fn test_validate_model_invalid() {
        assert!(validate_model("nonexistent").is_err());
        assert!(validate_model("").is_err());
        assert!(validate_model("huge").is_err());
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
            max_recordings: 0,
            push_to_talk: true,
            streaming: false,
            translate_to_english: false,
        };
        let output = format_status(&cfg, false);
        assert!(output.contains("open-bark v"));
        assert!(output.contains("f9"));
        assert!(output.contains("base.en"));
        assert!(output.contains("English"));
        assert!(output.contains("on (hold to talk)"));
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
            max_recordings: 5,
            push_to_talk: false,
            streaming: true,
            translate_to_english: false,
        };
        let output = format_status(&cfg, true);
        assert!(output.contains("off (press to start/stop)"));
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
            max_recordings: 0,
            push_to_talk: true,
            streaming: false,
            translate_to_english: false,
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
        // Verify it was saved
        let cfg = config::Config::load();
        assert_eq!(cfg.hotkey, "f9");
        // Restore original
        let _ = original.save();
    }

    #[test]
    fn test_cmd_set_hotkey_invalid() {
        assert!(cmd_set_hotkey("nonexistentkey123").is_err());
    }

    #[test]
    fn test_cmd_set_model_valid() {
        let original = config::Config::load();
        assert!(cmd_set_model("base.en").is_ok());
        let _ = original.save();
    }

    #[test]
    fn test_cmd_set_model_prints_download_hint() {
        // Use a model that almost certainly doesn't exist locally
        let original = config::Config::load();
        // "tiny" is a valid model but likely not downloaded
        assert!(cmd_set_model("tiny").is_ok());
        let _ = original.save();
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
