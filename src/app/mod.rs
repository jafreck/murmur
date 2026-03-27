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
use crate::input::hotkey::{self, HotkeyManager};
use crate::input::keycodes;
use crate::platform::permissions;
use crate::transcription::transcriber::Transcriber;
use crate::ui::tray::{TrayAction, TrayController};
use crate::VERSION;

use tray_icon::menu::MenuEvent;
use tray_icon::TrayIconEvent;

/// On macOS, initialize NSApplication with Accessory policy (no dock icon)
/// so the system tray icon renders and Cocoa events are dispatched.
#[cfg(target_os = "macos")]
fn init_macos_app() {
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy};

    let mtm = MainThreadMarker::new().expect("init_macos_app must be called on the main thread");
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
            TrayAction::ToggleFillerWordRemoval => AppMessage::TrayToggleFillerWordRemoval,
            TrayAction::SetMode(mode) => AppMessage::TraySetMode(mode),
            TrayAction::ToggleStreaming => AppMessage::TrayToggleStreaming,
            TrayAction::ToggleTranslate => AppMessage::TrayToggleTranslate,
            TrayAction::ToggleNoiseSuppression => AppMessage::TrayToggleNoiseSuppression,
            TrayAction::OpenConfig => AppMessage::TrayOpenConfig,
            TrayAction::ReloadConfig => AppMessage::TrayReloadConfig,
            TrayAction::SetHotkey => AppMessage::TraySetHotkey,
            TrayAction::ToggleWakeWord => AppMessage::TrayToggleWakeWord,
            TrayAction::ToggleOverlay => AppMessage::TrayToggleOverlay,
        }
    }
}

pub fn run() -> Result<()> {
    let mut config = Config::load();

    info!("Hotkey: {}", config.hotkey);
    info!("Model: {}", config.model_size);

    let parsed = keycodes::parse(&config.hotkey)
        .ok_or_else(|| anyhow::anyhow!("Invalid hotkey: {}", config.hotkey))?;

    permissions::check_accessibility();
    permissions::check_microphone();

    #[cfg(target_os = "macos")]
    init_macos_app();

    let mut tray = TrayController::new(&config)?;
    tray.set_state(crate::ui::tray::TrayState::Loading);

    let (tx, rx) = mpsc::channel::<AppMessage>();
    let hotkey_config = hotkey::shared_hotkey(&parsed);
    let capture_flag = Arc::new(AtomicBool::new(false));

    let hotkey_config_listener = hotkey_config.clone();
    let capture_flag_listener = capture_flag.clone();
    let tx_hotkey = tx.clone();
    std::thread::spawn(move || loop {
        let tx_down = tx_hotkey.clone();
        let tx_up = tx_hotkey.clone();
        let tx_capture = tx_hotkey.clone();
        match HotkeyManager::start(
            hotkey_config_listener.clone(),
            capture_flag_listener.clone(),
            move || {
                let _ = tx_down.send(AppMessage::KeyDown);
            },
            move || {
                let _ = tx_up.send(AppMessage::KeyUp);
            },
            move |key| {
                let _ = tx_capture.send(AppMessage::HotkeyCapture(key));
            },
        ) {
            Ok(()) => {
                info!("Hotkey listener exited");
                break;
            }
            Err(e) => {
                error!(
                    "Hotkey listener failed: {e}. Retrying in 5 seconds... \
                         On macOS, ensure Accessibility permission is granted in \
                         System Settings → Privacy & Security → Accessibility."
                );
                std::thread::sleep(std::time::Duration::from_secs(5));
            }
        }
    });

    // Load the model on a background thread so the tray appears immediately.
    let mut transcriber: Option<Arc<Transcriber>> = None;
    {
        let model_size = config.model_size.clone();
        let language = config.language.clone();
        let tx_load = tx.clone();
        std::thread::spawn(move || {
            if !crate::transcription::transcriber::model_exists(&model_size) {
                info!("Downloading {model_size} model...");
                let last_pct = std::cell::Cell::new(u32::MAX);
                if let Err(e) = crate::transcription::model::download(&model_size, |percent| {
                    let pct = percent as u32;
                    if pct != last_pct.get() {
                        last_pct.set(pct);
                        eprint!("\rDownloading {model_size} model... {pct}%");
                    }
                }) {
                    error!("Failed to download model '{model_size}': {e}");
                    let _ = tx_load.send(AppMessage::TranscriptionError(format!(
                        "Failed to download model '{model_size}': {e}"
                    )));
                    return;
                }
                eprintln!();
            }

            let Some(model_path) = crate::transcription::transcriber::find_model(&model_size)
            else {
                error!("Model '{model_size}' not found after download");
                let _ = tx_load.send(AppMessage::TranscriptionError(format!(
                    "Model '{model_size}' not found after download"
                )));
                return;
            };

            match Transcriber::new(&model_path, &language) {
                Ok(t) => {
                    info!("Model '{model_size}' loaded");
                    let _ = tx_load.send(AppMessage::TranscriberReady(Arc::new(t), 0));
                }
                Err(e) => {
                    error!("Failed to load model '{model_size}': {e}");
                    let _ = tx_load.send(AppMessage::TranscriptionError(format!(
                        "Failed to load model '{model_size}': {e}"
                    )));
                }
            }
        });
    }

    let mut recorder = AudioRecorder::with_noise_suppression(config.noise_suppression);
    if let Err(e) = recorder.warm() {
        error!("Failed to warm microphone: {e}");
    }
    let mut state = AppState::new(&config);
    let mut streaming_stop: Option<crate::transcription::streaming::StreamingHandle> = None;

    // Notes manager for the overlay
    let notes = crate::notes::NotesManager::new(config.notes_dir());

    // Overlay subprocess (if enabled)
    let mut overlay: Option<crate::ui::overlay::OverlayHandle> = if config.overlay_enabled {
        match crate::ui::overlay::OverlayHandle::spawn() {
            Ok(h) => {
                info!("Overlay started");
                Some(h)
            }
            Err(e) => {
                error!("Failed to start overlay: {e}");
                None
            }
        }
    } else {
        None
    };

    // Wake word detector — deferred until the main transcriber is ready
    let mut wake_word: Option<crate::input::wake_word::WakeWordHandle> = None;

    println!("murmur v{VERSION}");
    println!("Hotkey: {}", config.hotkey);
    println!("Model: {}", config.model_size);
    if config.wake_word_enabled {
        println!("Wake word: \"{}\"", config.wake_word);
    }
    println!("Loading model in background...");

    loop {
        let mut should_quit = false;

        while let Ok(msg) = rx.try_recv() {
            // TranscriberReady carries an Arc that must be moved out,
            // so handle it directly instead of going through the state machine.
            if let AppMessage::TranscriberReady(new_t, generation) = msg {
                if generation == state.reload_generation {
                    transcriber = Some(new_t);
                    tray.set_state(crate::ui::tray::TrayState::Idle);
                    info!("Transcriber ready");
                    if generation == 0 {
                        println!("Ready.");
                        // Start wake word detector now that the main model is loaded
                        if config.wake_word_enabled && wake_word.is_none() {
                            let wake_phrase = config.wake_word.clone();
                            let stop_phrase = config.stop_phrase.clone();
                            let ww_tx = tx.clone();
                            let (event_tx, event_rx) = mpsc::channel();
                            std::thread::spawn(move || {
                                use crate::input::wake_word::WakeWordEvent;
                                while let Ok(event) = event_rx.recv() {
                                    let msg = match event {
                                        WakeWordEvent::WakeWordDetected => {
                                            AppMessage::WakeWordDetected
                                        }
                                        WakeWordEvent::StopPhraseDetected => {
                                            AppMessage::StopPhraseDetected
                                        }
                                    };
                                    if ww_tx.send(msg).is_err() {
                                        break;
                                    }
                                }
                            });
                            match crate::input::wake_word::start_detector(
                                wake_phrase,
                                stop_phrase,
                                event_tx,
                            ) {
                                Ok(handle) => {
                                    info!("Wake word detector started");
                                    wake_word = Some(handle);
                                }
                                Err(e) => {
                                    error!("Failed to start wake word detector: {e}");
                                }
                            }
                        }
                    }
                } else {
                    info!(
                        "Discarding stale transcriber reload (gen {generation}, current {})",
                        state.reload_generation
                    );
                }
                continue;
            }
            let mut effects = VecDeque::from(state.handle_message(&msg));
            while let Some(effect) = effects.pop_front() {
                let (quit, extra) = effects::apply_effect(
                    effect,
                    &mut EffectContext {
                        recorder: &mut recorder,
                        transcriber: &mut transcriber,
                        tray: &mut tray,
                        config: &mut config,
                        state: &mut state,
                        tx: &tx,
                        streaming_stop: &mut streaming_stop,
                        hotkey_config: &hotkey_config,
                        capture_flag: &capture_flag,
                        overlay: &mut overlay,
                        wake_word: &mut wake_word,
                        notes: &notes,
                    },
                )?;
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

        tray.tick();
        pump_event_loop();
    }

    // Clean up
    if let Some(ww) = wake_word {
        ww.stop();
    }
    if let Some(mut ov) = overlay {
        ov.quit();
    }

    Ok(())
}
