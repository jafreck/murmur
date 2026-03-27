use std::sync::Mutex;

use log::info;
use tauri::{AppHandle, Emitter, State};

use crate::meeting::{MeetingSession, SessionState};

/// Application-level state managed by Tauri.
pub struct AppState {
    pub session: Mutex<Option<MeetingSession>>,
}

#[tauri::command]
pub fn start_meeting(state: State<'_, AppState>, app: AppHandle) -> Result<String, String> {
    let mut guard = state.session.lock().map_err(|e| e.to_string())?;

    if let Some(ref s) = *guard {
        if s.state() == SessionState::Recording {
            return Err("meeting already in progress".into());
        }
    }

    let mut session = MeetingSession::new().map_err(|e| e.to_string())?;
    let rx = session.start().map_err(|e| e.to_string())?;
    *guard = Some(session);

    // Spawn a thread that forwards transcript updates to the frontend via Tauri events.
    let handle = app.clone();
    std::thread::spawn(move || {
        for update in rx {
            if handle.emit("transcript-update", &update).is_err() {
                break;
            }
        }
        info!("transcript forwarding thread exited");
    });

    Ok("meeting started".into())
}

#[tauri::command]
pub fn stop_meeting(state: State<'_, AppState>) -> Result<String, String> {
    let mut guard = state.session.lock().map_err(|e| e.to_string())?;
    match guard.as_mut() {
        Some(session) if session.state() == SessionState::Recording => {
            let transcript = session.stop();
            *guard = None;
            Ok(transcript)
        }
        _ => Err("no active meeting to stop".into()),
    }
}

#[tauri::command]
pub fn get_status(state: State<'_, AppState>) -> Result<SessionState, String> {
    let guard = state.session.lock().map_err(|e| e.to_string())?;
    Ok(guard
        .as_ref()
        .map(|s| s.state())
        .unwrap_or(SessionState::Idle))
}
