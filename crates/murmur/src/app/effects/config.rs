use anyhow::Result;
use log::{error, info};

use crate::config::Config;

use super::EffectContext;
use crate::app::{AppEffect, AppState};

// ---------------------------------------------------------------------------
// Pure decision functions (extracted for testability)
// ---------------------------------------------------------------------------

/// Result of comparing old and new configuration states.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ConfigDiff {
    /// The language after applying English-only model enforcement.
    pub effective_language: String,
    /// Whether the language selection menu should be enabled.
    pub language_menu_enabled: bool,
    /// Whether model or effective language differs from the old state.
    pub model_or_language_changed: bool,
    /// Whether the hotkey binding changed.
    pub hotkey_changed: bool,
}

/// Compare old runtime state with a newly loaded config to decide what changed.
pub(crate) fn compute_config_diff(
    old_model: &str,
    old_language: &str,
    old_hotkey: &str,
    new_config: &Config,
) -> ConfigDiff {
    let english_only = crate::config::is_english_only_model(new_config.model_size());
    let effective_language = if english_only && new_config.language() != "en" {
        "en".to_string()
    } else {
        new_config.language().to_string()
    };

    ConfigDiff {
        model_or_language_changed: new_config.model_size() != old_model
            || effective_language != old_language,
        hotkey_changed: new_config.hotkey() != old_hotkey,
        language_menu_enabled: !english_only,
        effective_language,
    }
}

// ---------------------------------------------------------------------------
// Effect handlers
// ---------------------------------------------------------------------------

pub(super) fn open_config_file() {
    let config_path = Config::file_path();
    info!("Opening config: {}", config_path.display());
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(&config_path).spawn();
    }
    #[cfg(target_os = "linux")]
    {
        let _ = std::process::Command::new("xdg-open")
            .arg(&config_path)
            .spawn();
    }
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("cmd")
            .args(["/C", "start", ""])
            .arg(&config_path)
            .spawn();
    }
}

pub(super) fn reload_config(ctx: &mut EffectContext<'_>) -> Result<(bool, Vec<AppEffect>)> {
    info!("Reloading config from disk...");
    let new_config = Config::load();
    let old_model = ctx.state.model_size.clone();
    let old_lang = ctx.state.language.clone();
    let old_generation = ctx.state.reload_generation;

    let diff = compute_config_diff(&old_model, &old_lang, ctx.config.hotkey(), &new_config);

    *ctx.state = AppState::new(&new_config);
    // Preserve reload_generation so in-flight reloads are correctly discarded
    ctx.state.reload_generation = old_generation;

    // Apply English-only model enforcement
    ctx.state.language = diff.effective_language;
    ctx.tray
        .set_language_menu_enabled(diff.language_menu_enabled);

    // Update the hotkey listener if the hotkey changed
    if diff.hotkey_changed {
        if let Some(parsed) = crate::input::keycodes::parse(new_config.hotkey()) {
            if let Ok(mut hk) = ctx.hotkey_config.lock() {
                *hk = (parsed.key, parsed.modifiers.into_iter().collect());
            }
            info!("Hotkey updated to: {}", new_config.hotkey());
            ctx.tray.set_hotkey(new_config.hotkey());
        } else {
            error!(
                "Invalid hotkey in config: '{}', keeping previous",
                new_config.hotkey()
            );
        }
    }

    *ctx.config = new_config;

    // Sync all tray UI elements to the newly loaded config
    ctx.tray.sync_config(ctx.config);

    // If model or language changed, reload the transcriber
    if diff.model_or_language_changed {
        ctx.state.reload_generation += 1;
        let gen = ctx.state.reload_generation;
        info!("Config reloaded");
        return Ok((false, vec![AppEffect::ReloadTranscriber(gen)]));
    }
    info!("Config reloaded");
    Ok((false, vec![]))
}
