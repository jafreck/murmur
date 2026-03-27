#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod meeting;
mod overlay;

use log::info;

fn main() {
    env_logger::init();
    info!("starting murmur-copilot");

    tauri::Builder::default()
        .setup(|app| {
            overlay::configure_overlay(app)?;
            info!("overlay window configured");
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::start_meeting,
            commands::stop_meeting,
            commands::get_status,
        ])
        .run(tauri::generate_context!())
        .expect("error while running murmur-copilot");
}
