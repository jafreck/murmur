#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod llm;
mod meeting;
mod overlay;
mod session;

use log::info;
use murmur_core::config::Config;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    info!("starting murmur-copilot");

    let config = Config::load();
    let llm_manager = llm::LlmManager::try_connect(&config.ollama_url, &config.llm_model);
    let session_store = session::SessionStore::new(config.sessions_dir.as_deref());

    tauri::Builder::default()
        .manage(commands::AppState {
            session: std::sync::Mutex::new(None),
            stealth_enabled: std::sync::Mutex::new(config.stealth_mode),
            llm: std::sync::Arc::new(std::sync::Mutex::new(llm_manager)),
            session_store: std::sync::Mutex::new(session_store),
            auto_summary: config.auto_summary,
        })
        .setup(|app| {
            overlay::configure_overlay(app)?;
            info!("overlay window configured");
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::start_meeting,
            commands::stop_meeting,
            commands::get_status,
            commands::list_audio_devices,
            commands::set_system_audio_device,
            commands::toggle_stealth,
            commands::get_suggestion,
            commands::generate_summary,
            commands::get_llm_status,
            commands::extract_action_items,
            commands::list_sessions,
            commands::get_session,
            commands::delete_session,
            commands::export_session,
        ])
        .run(tauri::generate_context!())
        .expect("error while running murmur-copilot");
}
