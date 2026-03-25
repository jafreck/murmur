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
    if !config::SUPPORTED_MODELS.contains(&size) {
        anyhow::bail!(
            "Unknown model '{size}'. Available: {}",
            config::SUPPORTED_MODELS.join(", ")
        );
    }

    let mut cfg = config::Config::load();
    cfg.model_size = size.to_string();
    cfg.save()?;

    println!("Model set to: {size}");
    if !transcriber::model_exists(size) {
        println!("Model will be downloaded on next start.");
    }
    Ok(())
}

fn cmd_set_language(code: &str) -> Result<()> {
    if !config::is_valid_language(code) {
        anyhow::bail!("Unknown language '{code}'. Examples: en, fr, de, es, auto");
    }

    let mut cfg = config::Config::load();
    cfg.language = code.to_string();
    cfg.save()?;

    let name = config::language_name(code).unwrap_or(code);
    println!("Language set to: {name} ({code})");
    Ok(())
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
    let lang_name = config::language_name(&cfg.language).unwrap_or(&cfg.language);

    println!("open-bark v{VERSION}");
    println!("Config:      {}", config::Config::file_path().display());
    println!("Hotkey:      {}", cfg.hotkey);
    println!("Model:       {}", cfg.model_size);
    println!(
        "Model ready: {}",
        if transcriber::model_exists(&cfg.model_size) { "yes" } else { "no" }
    );
    println!("Language:    {lang_name} ({})", cfg.language);
    println!(
        "Toggle:      {}",
        if cfg.toggle_mode { "on (press to start/stop)" } else { "off (hold to talk)" }
    );
    Ok(())
}
