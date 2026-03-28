#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod llm;
mod meeting;
mod overlay;
mod session;

use log::info;
use murmur_core::config::Config;
use tauri::Emitter;

fn main() {
    env_logger::init();
    info!("starting murmur-copilot");

    let config = Config::load();
    let llm_manager = llm::LlmManager::try_connect(&config.ollama_url, &config.llm_model);
    let session_store = session::SessionStore::new(config.sessions_dir.as_deref());

    tauri::Builder::default()
        .manage(commands::AppState {
            session: std::sync::Mutex::new(None),
            llm: std::sync::Arc::new(std::sync::Mutex::new(llm_manager)),
            session_store: std::sync::Mutex::new(session_store),
            auto_summary: config.auto_summary,
        })
        .setup(|app| {
            overlay::configure_overlay(app)?;
            info!("overlay window configured");

            // Spawn wake word detector on a background thread (model loading is slow)
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                if let Err(e) = start_wake_word_detector(handle) {
                    log::error!("Wake word detector failed to start: {e}");
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::start_meeting,
            commands::stop_meeting,
            commands::get_status,
            commands::list_audio_devices,
            commands::set_system_audio_device,
            commands::get_suggestion,
            commands::generate_summary,
            commands::get_llm_status,
            commands::extract_action_items,
            commands::ask_question,
            commands::list_sessions,
            commands::get_session,
            commands::delete_session,
            commands::export_session,
        ])
        .run(tauri::generate_context!())
        .expect("error while running murmur-copilot");
}

/// Start the wake word detector, forwarding events as Tauri events.
///
/// This blocks while loading the model, so call from a background thread.
fn start_wake_word_detector(app: tauri::AppHandle) -> anyhow::Result<()> {
    use murmur_core::input::wake_word::{start_detector, WakeWordEvent};

    let wake_phrase = "murmur listen".to_string();
    let stop_phrase = "murmur stop".to_string();

    let (tx, rx) = std::sync::mpsc::channel();

    let _handle = start_detector(wake_phrase, stop_phrase, tx)?;
    info!("wake word detector started");

    // Forward events to the Tauri frontend — this loop keeps _handle alive
    for event in rx {
        match event {
            WakeWordEvent::WakeWordDetected => {
                info!("wake word detected — emitting copilot-wake");
                let _ = app.emit("copilot-wake", ());
            }
            WakeWordEvent::StopPhraseDetected => {
                info!("stop phrase detected — emitting copilot-sleep");
                let _ = app.emit("copilot-sleep", ());
            }
        }
    }

    Ok(())
}
