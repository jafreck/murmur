mod effects;
mod state;

// Re-export the public API
pub use effects::EffectContext;
pub use state::{AppEffect, AppMessage, AppState};

use anyhow::Result;
use log::{error, info};
use std::collections::VecDeque;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::sync::Arc;

use crate::audio::AudioRecorder;
use crate::config::Config;
use crate::hotkey::{self, HotkeyManager};
use crate::tray::{TrayAction, TrayController};
use crate::transcriber::Transcriber;
use crate::VERSION;

use tray_icon::menu::MenuEvent;
use tray_icon::TrayIconEvent;

/// On macOS, initialize NSApplication with Accessory policy (no dock icon)
/// so the system tray icon renders and Cocoa events are dispatched.
#[cfg(target_os = "macos")]
fn init_macos_app() {
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy};

    let mtm = MainThreadMarker::new()
        .expect("init_macos_app must be called on the main thread");
    let app = NSApplication::sharedApplication(mtm);
    app.setActivationPolicy(NSApplicationActivationPolicy::Accessory);
}

/// Pump the macOS AppKit event loop, dispatching all pending Cocoa events
/// (tray icon clicks, menu interactions, rendering, etc.).
#[cfg(target_os = "macos")]
fn pump_event_loop() {
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSApplication, NSEventMask};
    use objc2_foundation::{NSDate, NSDefaultRunLoopMode};

    let mtm = MainThreadMarker::new().expect("must be on main thread");
    let app = NSApplication::sharedApplication(mtm);

    loop {
        let event = unsafe {
            app.nextEventMatchingMask_untilDate_inMode_dequeue(
                NSEventMask::Any,
                Some(&NSDate::distantPast()),
                NSDefaultRunLoopMode,
                true,
            )
        };
        match event {
            Some(event) => app.sendEvent(&event),
            None => break,
        }
    }

    std::thread::sleep(std::time::Duration::from_millis(16));
}

/// On non-macOS platforms, just sleep briefly.
#[cfg(not(target_os = "macos"))]
fn pump_event_loop() {
    std::thread::sleep(std::time::Duration::from_millis(16));
}

impl From<TrayAction> for AppMessage {
    fn from(action: TrayAction) -> Self {
        match action {
            TrayAction::Quit => AppMessage::TrayQuit,
            TrayAction::CopyLastDictation => AppMessage::TrayCopyLast,
            TrayAction::SetModel(s) => AppMessage::TraySetModel(s),
            TrayAction::SetLanguage(c) => AppMessage::TraySetLanguage(c),
            TrayAction::ToggleSpokenPunctuation => AppMessage::TrayToggleSpokenPunctuation,
            TrayAction::SetMode(mode) => AppMessage::TraySetMode(mode),
            TrayAction::ToggleStreaming => AppMessage::TrayToggleStreaming,
            TrayAction::ToggleTranslate => AppMessage::TrayToggleTranslate,
            TrayAction::OpenConfig => AppMessage::TrayOpenConfig,
            TrayAction::ReloadConfig => AppMessage::TrayReloadConfig,
            TrayAction::SetHotkey => AppMessage::TraySetHotkey,
        }
    }
}

pub fn run() -> Result<()> {
    let mut config = Config::load();

    info!("Hotkey: {}", config.hotkey);
    info!("Model: {}", config.model_size);

    let parsed = crate::keycodes::parse(&config.hotkey)
        .ok_or_else(|| anyhow::anyhow!("Invalid hotkey: {}", config.hotkey))?;

    crate::permissions::check_accessibility();
    crate::permissions::check_microphone();

    #[cfg(target_os = "macos")]
    init_macos_app();

    let mut tray = TrayController::new(&config)?;
    tray.set_state(crate::tray::TrayState::Loading);

    let (tx, rx) = mpsc::channel::<AppMessage>();
    let tx_down = tx.clone();
    let tx_up = tx.clone();

    let hotkey_config = hotkey::shared_hotkey(&parsed);
    let capture_flag = Arc::new(AtomicBool::new(false));

    let hotkey_config_listener = hotkey_config.clone();
    let capture_flag_listener = capture_flag.clone();
    let tx_capture = tx.clone();
    std::thread::spawn(move || {
        if let Err(e) = HotkeyManager::start(
            hotkey_config_listener,
            capture_flag_listener,
            move || { let _ = tx_down.send(AppMessage::KeyDown); },
            move || { let _ = tx_up.send(AppMessage::KeyUp); },
            move |key| { let _ = tx_capture.send(AppMessage::HotkeyCapture(key)); },
        ) {
            error!("Hotkey listener failed: {e}");
        }
    });

    // Load the model on a background thread so the tray appears immediately.
    let mut transcriber: Option<Arc<Transcriber>> = None;
    {
        let model_size = config.model_size.clone();
        let language = config.language.clone();
        let tx_load = tx.clone();
        std::thread::spawn(move || {
            if !crate::transcriber::model_exists(&model_size) {
                info!("Downloading {model_size} model...");
                if let Err(e) = crate::model::download(&model_size, |percent| {
                    eprint!("\rDownloading {model_size} model... {percent:.0}%");
                }) {
                    error!("Failed to download model '{model_size}': {e}");
                    let _ = tx_load.send(AppMessage::TranscriptionError(
                        format!("Failed to download model '{model_size}': {e}"),
                    ));
                    return;
                }
                eprintln!();
            }

            let Some(model_path) = crate::transcriber::find_model(&model_size) else {
                error!("Model '{model_size}' not found after download");
                let _ = tx_load.send(AppMessage::TranscriptionError(
                    format!("Model '{model_size}' not found after download"),
                ));
                return;
            };

            match Transcriber::new(&model_path, &language) {
                Ok(t) => {
                    info!("Model '{model_size}' loaded");
                    let _ = tx_load.send(AppMessage::TranscriberReady(Arc::new(t), 0));
                }
                Err(e) => {
                    error!("Failed to load model '{model_size}': {e}");
                    let _ = tx_load.send(AppMessage::TranscriptionError(
                        format!("Failed to load model '{model_size}': {e}"),
                    ));
                }
            }
        });
    }

    let mut recorder = AudioRecorder::new();
    let mut state = AppState::new(&config);
    let mut streaming_stop: Option<mpsc::Sender<()>> = None;

    println!("open-bark v{VERSION}");
    println!("Hotkey: {}", config.hotkey);
    println!("Model: {}", config.model_size);
    println!("Loading model in background...");

    loop {
        let mut should_quit = false;

        while let Ok(msg) = rx.try_recv() {
            // TranscriberReady carries an Arc that must be moved out,
            // so handle it directly instead of going through the state machine.
            if let AppMessage::TranscriberReady(new_t, generation) = msg {
                if generation == state.reload_generation {
                    transcriber = Some(new_t);
                    tray.set_state(crate::tray::TrayState::Idle);
                    info!("Transcriber ready");
                    if generation == 0 {
                        println!("Ready.");
                    }
                } else {
                    info!("Discarding stale transcriber reload (gen {generation}, current {})", state.reload_generation);
                }
                continue;
            }
            let mut effects = VecDeque::from(state.handle_message(&msg));
            while let Some(effect) = effects.pop_front() {
                let (quit, extra) = effects::apply_effect(effect, &mut EffectContext {
                    recorder: &mut recorder,
                    transcriber: &mut transcriber,
                    tray: &mut tray,
                    config: &mut config,
                    state: &mut state,
                    tx: &tx,
                    streaming_stop: &mut streaming_stop,
                    hotkey_config: &hotkey_config,
                    capture_flag: &capture_flag,
                })?;
                effects.extend(extra);
                if quit {
                    should_quit = true;
                }
            }
        }

        if should_quit {
            break;
        }

        while let Ok(event) = MenuEvent::receiver().try_recv() {
            if let Some(action) = tray.match_menu_event(&event) {
                let _ = tx.send(action.into());
            }
        }

        while let Ok(_event) = TrayIconEvent::receiver().try_recv() {}

        pump_event_loop();
    }

    Ok(())
}
