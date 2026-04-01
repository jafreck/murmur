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
use crate::config::{AppMode, Config};
use crate::input::hotkey::{self, HotkeyManager};
use crate::input::keycodes;
use crate::platform::permissions;
use crate::ui::tray::{TrayAction, TrayController};
use crate::VERSION;
use murmur_core::transcription::AsrEngine;

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
///
/// Does **not** sleep — the caller controls pacing via `recv_timeout`.
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
}

/// On non-macOS platforms, this is a no-op — the caller controls pacing via
/// `recv_timeout`.
#[cfg(not(target_os = "macos"))]
fn pump_event_loop() {}

/// How often the main loop ticks when idle (no recording in progress).
/// Higher values save CPU; 100ms gives ~10 ticks/sec which is plenty for
/// tray updates and UI responsiveness.
const IDLE_TICK_MS: u64 = 100;

/// How often the main loop ticks during active recording or transcription.
/// 16ms ≈ 60fps for responsive streaming text updates.
const ACTIVE_TICK_MS: u64 = 16;

impl From<TrayAction> for AppMessage {
    fn from(action: TrayAction) -> Self {
        match action {
            TrayAction::Quit => AppMessage::TrayQuit,
            TrayAction::CopyLastDictation => AppMessage::TrayCopyLast,
            TrayAction::SetModel(s) => AppMessage::TraySetModel(s),
            TrayAction::SetLanguage(c) => AppMessage::TraySetLanguage(c),
            TrayAction::SetBackend(b) => AppMessage::TraySetBackend(b),
            TrayAction::ToggleSpokenPunctuation => AppMessage::TrayToggleSpokenPunctuation,
            TrayAction::ToggleFillerWordRemoval => AppMessage::TrayToggleFillerWordRemoval,
            TrayAction::SetMode(mode) => AppMessage::TraySetMode(mode),
            TrayAction::ToggleStreaming => AppMessage::TrayToggleStreaming,
            TrayAction::ToggleTranslate => AppMessage::TrayToggleTranslate,
            TrayAction::ToggleNoiseSuppression => AppMessage::TrayToggleNoiseSuppression,
            TrayAction::OpenConfig => AppMessage::TrayOpenConfig,
            TrayAction::ReloadConfig => AppMessage::TrayReloadConfig,
            TrayAction::SetHotkey => AppMessage::TraySetHotkey,
            TrayAction::ToggleAppMode => AppMessage::TrayToggleAppMode,
            TrayAction::CheckForUpdates => AppMessage::TrayCheckForUpdates,
        }
    }
}

pub fn run(notes_mode: bool) -> Result<()> {
    let mut config = Config::load();

    if notes_mode {
        config.app_mode = AppMode::Notes;
    }

    info!("Hotkey: {}", config.hotkey);
    info!("Model: {}", config.model_size);

    let parsed = keycodes::parse(&config.hotkey)
        .ok_or_else(|| anyhow::anyhow!("Invalid hotkey: {}", config.hotkey))?;

    let accessible = permissions::check_accessibility();
    permissions::check_microphone();

    // If accessibility isn't granted, wait for the user to grant it,
    // then re-exec so the process picks up the new permission.
    #[cfg(target_os = "macos")]
    if !accessible {
        permissions::open_accessibility_settings();
        eprintln!("⚠ Accessibility permission required for hotkey detection.");
        eprintln!("  Grant access in the System Settings window that just opened,");
        eprintln!("  then murmur will restart automatically.");
        permissions::wait_for_accessibility();
        eprintln!("✓ Permission granted — restarting...");
        permissions::re_exec();
    }

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
    let mut engine: Option<Arc<dyn AsrEngine + Send + Sync>> = None;
    let mut streaming_worker: Option<murmur_core::transcription::SubprocessTranscriber> = None;
    {
        let backend = config.asr_backend;
        let model_size = config.model_size.clone();
        let language = config.language.clone();
        let quantization = config.asr_quantization;
        let tx_load = tx.clone();

        // Warn about streaming limitations for non-Whisper backends
        if config.streaming && !matches!(backend, murmur_core::config::AsrBackend::Whisper) {
            info!(
                "Streaming with {backend} uses chunk-based fallback \
                 (subprocess streaming is Whisper-only)"
            );
        }

        std::thread::spawn(move || {
            match effects::create_engine_on_thread(backend, &model_size, &language, quantization) {
                Ok(e) => {
                    info!("{backend} model '{model_size}' loaded");
                    let _ = tx_load.send(AppMessage::EngineReady(Arc::from(e), 0));
                }
                Err(e) => {
                    error!("Failed to load {backend} model '{model_size}': {e}");
                    let _ = tx_load.send(AppMessage::TranscriptionError(format!(
                        "Failed to load {backend} model '{model_size}': {e}"
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
    let mut overlay: Option<crate::ui::overlay::OverlayHandle> = if config.is_notes_mode() {
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
    let mut wake_word: Option<murmur_core::input::wake_word::WakeWordHandle> = None;

    println!("murmur v{VERSION}");
    println!("Hotkey: {}", config.hotkey);
    println!("Model: {}", config.model_size);
    if config.is_notes_mode() {
        println!("Wake word: \"{}\"", config.wake_word);
    }
    println!("Loading model in background...");

    // Background update check
    {
        let auto_update = config.auto_update;
        let tx_update = tx.clone();
        std::thread::spawn(
            move || match murmur_core::update::check_for_update(crate::VERSION) {
                Ok(Some(info)) => {
                    info!(
                        "Update available: v{} → v{}",
                        info.current_version, info.latest_version
                    );
                    if auto_update {
                        if let Err(e) = murmur_core::update::apply_update(&info, |msg| {
                            info!("Update: {msg}");
                        }) {
                            error!("Auto-update failed: {e}");
                            let _ = tx_update.send(AppMessage::UpdateError(format!("{e}")));
                        } else {
                            let _ = tx_update
                                .send(AppMessage::UpdateApplied(info.latest_version.clone()));
                        }
                    } else {
                        let _ = tx_update.send(AppMessage::UpdateAvailable(info));
                    }
                }
                Ok(None) => {
                    info!("No updates available");
                }
                Err(e) => {
                    info!("Update check failed: {e}");
                }
            },
        );
    }

    let mut last_appearance_check = std::time::Instant::now();
    // How often to re-check system dark/light mode (seconds).
    const APPEARANCE_CHECK_INTERVAL_SECS: u64 = 5;

    loop {
        let mut should_quit = false;

        // Adaptive tick rate: fast during recording for responsive streaming,
        // slow when idle to save CPU.
        let tick_ms = if state.is_pressed {
            ACTIVE_TICK_MS
        } else {
            IDLE_TICK_MS
        };

        // Block on the message channel instead of busy-polling.
        // `recv_timeout` wakes immediately when a message arrives, or after
        // `tick_ms` so we can still pump UI events and check timeouts.
        match rx.recv_timeout(std::time::Duration::from_millis(tick_ms)) {
            Ok(msg) => {
                // Process this message, then drain any remaining pending messages.
                let _ = tx.send(msg); // re-enqueue so the drain loop below handles it
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }

        while let Ok(msg) = rx.try_recv() {
            // EngineReady carries an Arc that must be moved out,
            // so handle it directly instead of going through the state machine.
            if let AppMessage::EngineReady(new_engine, generation) = msg {
                if generation == state.reload_generation {
                    // Pre-spawn the whisper worker subprocess so streaming
                    // starts instantly without model-loading delay.
                    // Only applicable for the Whisper backend.
                    if matches!(config.asr_backend, murmur_core::config::AsrBackend::Whisper) {
                        if let Some(model_path) =
                            murmur_core::transcription::find_model(&config.model_size)
                        {
                            match murmur_core::transcription::SubprocessTranscriber::new(
                                &model_path,
                                &config.language,
                            ) {
                                Ok(w) => {
                                    streaming_worker = Some(w);
                                    info!("Whisper worker subprocess ready");
                                }
                                Err(e) => {
                                    error!("Failed to spawn whisper worker: {e}");
                                }
                            }
                        }
                    }
                    engine = Some(new_engine);
                    tray.set_state(crate::ui::tray::TrayState::Idle);
                    info!("ASR engine ready");
                    if generation == 0 {
                        println!("Ready.");
                        // Start wake word detector now that the main model is loaded
                        if config.is_notes_mode() && wake_word.is_none() {
                            let wake_phrase = config.wake_word.clone();
                            let stop_phrase = config.stop_phrase.clone();
                            let ww_tx = tx.clone();
                            let (event_tx, event_rx) = mpsc::channel();
                            std::thread::spawn(move || {
                                use murmur_core::input::wake_word::WakeWordEvent;
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
                            match murmur_core::input::wake_word::start_detector(
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
                        "Discarding stale engine reload (gen {generation}, current {})",
                        state.reload_generation
                    );
                }
                continue;
            }
            // Handle update messages directly (they carry owned data or
            // trigger side-effects that the pure state machine can't do).
            match msg {
                AppMessage::UpdateAvailable(info) => {
                    info!(
                        "Update available: v{} → v{}",
                        info.current_version, info.latest_version
                    );
                    tray.set_update_available(&info.latest_version);
                    continue;
                }
                AppMessage::UpdateApplied(version) => {
                    info!("Update applied: v{version}. Restart to use new version.");
                    tray.set_status(&format!("Updated to v{version} — restart to apply"));
                    continue;
                }
                AppMessage::UpdateError(e) => {
                    error!("Update error: {e}");
                    continue;
                }
                AppMessage::TrayCheckForUpdates => {
                    let tx_update = tx.clone();
                    std::thread::spawn(move || {
                        match murmur_core::update::check_for_update(crate::VERSION) {
                            Ok(Some(info)) => {
                                let _ = tx_update.send(AppMessage::UpdateAvailable(info));
                            }
                            Ok(None) => {
                                info!("Already up to date (v{})", crate::VERSION);
                            }
                            Err(e) => {
                                error!("Update check failed: {e}");
                                let _ = tx_update.send(AppMessage::UpdateError(format!("{e}")));
                            }
                        }
                    });
                    continue;
                }
                _ => {}
            }
            let mut effects = VecDeque::from(state.handle_message(&msg));
            while let Some(effect) = effects.pop_front() {
                // Keep the tray animation smooth between potentially-blocking
                // effects (e.g. typing, clipboard, worker respawn).
                tray.tick();

                let (quit, extra) = effects::apply_effect(
                    effect,
                    &mut EffectContext {
                        recorder: &mut recorder,
                        engine: &mut engine,
                        tray: &mut tray,
                        config: &mut config,
                        state: &mut state,
                        tx: &tx,
                        streaming_stop: &mut streaming_stop,
                        streaming_worker: &mut streaming_worker,
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

        // Periodically re-check dark/light mode so icons adapt.
        if last_appearance_check.elapsed().as_secs() >= APPEARANCE_CHECK_INTERVAL_SECS {
            tray.refresh_appearance();
            last_appearance_check = std::time::Instant::now();
        }

        // Auto-stop wake-word dictation after sustained silence
        let silence_effects = state.check_silence_timeout();
        for effect in silence_effects {
            let (quit, _) = effects::apply_effect(
                effect,
                &mut EffectContext {
                    recorder: &mut recorder,
                    engine: &mut engine,
                    tray: &mut tray,
                    config: &mut config,
                    state: &mut state,
                    tx: &tx,
                    streaming_stop: &mut streaming_stop,
                    streaming_worker: &mut streaming_worker,
                    hotkey_config: &hotkey_config,
                    capture_flag: &capture_flag,
                    overlay: &mut overlay,
                    wake_word: &mut wake_word,
                    notes: &notes,
                },
            )?;
            if quit {
                should_quit = true;
                break;
            }
        }

        if should_quit {
            break;
        }

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
