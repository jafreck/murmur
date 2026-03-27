use std::sync::{Arc, Mutex};

use log::info;
use tauri::{AppHandle, Emitter, Manager, State};

use crate::llm::LlmManager;
use crate::meeting::{MeetingSession, SessionState, TranscriptEntry};
use crate::overlay;

/// Application-level state managed by Tauri.
pub struct AppState {
    pub session: Mutex<Option<MeetingSession>>,
    pub stealth_enabled: Mutex<bool>,
    pub llm: Arc<Mutex<LlmManager>>,
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
pub fn stop_meeting(state: State<'_, AppState>) -> Result<Vec<TranscriptEntry>, String> {
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

/// Lists available audio input devices. Virtual loopback devices (BlackHole,
/// Loopback, etc.) are flagged with `is_loopback_hint: true`.
#[tauri::command]
pub fn list_audio_devices() -> Vec<murmur_core::audio::AudioDevice> {
    murmur_core::audio::list_audio_devices()
}

/// Set (or clear) the system audio device used for capturing remote
/// participants' audio during a meeting.
#[tauri::command]
pub fn set_system_audio_device(
    state: State<'_, AppState>,
    device_name: Option<String>,
) -> Result<String, String> {
    let mut guard = state.session.lock().map_err(|e| e.to_string())?;
    if let Some(ref mut session) = *guard {
        session.set_system_audio_device(device_name.as_deref());
    }
    match &device_name {
        Some(name) => Ok(format!("system audio device set to '{name}'")),
        None => Ok("system audio device cleared".into()),
    }
}

/// Toggle stealth mode on/off. Returns the new state.
#[tauri::command]
pub fn toggle_stealth(state: State<'_, AppState>, app: AppHandle) -> Result<bool, String> {
    let mut stealth = state.stealth_enabled.lock().map_err(|e| e.to_string())?;
    let new_state = !*stealth;
    *stealth = new_state;

    if let Some(window) = app.get_webview_window("overlay") {
        if new_state {
            overlay::apply_stealth_mode(&window);
        } else {
            overlay::remove_stealth_mode(&window);
        }
    }

    info!("stealth mode toggled to {new_state}");
    Ok(new_state)
}

/// Get an AI-generated suggestion based on the current meeting transcript.
#[tauri::command]
pub async fn get_suggestion(state: State<'_, AppState>) -> Result<Option<String>, String> {
    let transcript_text = {
        let session_guard = state.session.lock().map_err(|e| e.to_string())?;
        match session_guard.as_ref() {
            Some(session) => session.transcript_text(),
            None => return Err("no active meeting".into()),
        }
    };

    if transcript_text.is_empty() {
        return Ok(None);
    }

    let llm = state.llm.clone();
    tokio::task::spawn_blocking(move || {
        let llm = llm.lock().map_err(|e| e.to_string())?;
        Ok(llm.suggest(&transcript_text))
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Generate a full meeting summary from the transcript.
#[tauri::command]
pub async fn generate_summary(state: State<'_, AppState>) -> Result<Option<String>, String> {
    let transcript_text = {
        let session_guard = state.session.lock().map_err(|e| e.to_string())?;
        match session_guard.as_ref() {
            Some(session) => session.transcript_text(),
            None => return Err("no meeting session available".into()),
        }
    };

    if transcript_text.is_empty() {
        return Ok(None);
    }

    let llm = state.llm.clone();
    tokio::task::spawn_blocking(move || {
        let llm = llm.lock().map_err(|e| e.to_string())?;
        Ok(llm.summarize(&transcript_text))
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Check if the LLM backend is available and return model info.
#[tauri::command]
pub fn get_llm_status(state: State<'_, AppState>) -> Result<LlmStatus, String> {
    let llm = state.llm.lock().map_err(|e| e.to_string())?;
    Ok(LlmStatus {
        available: llm.is_available(),
        model: llm.model_name().map(String::from),
    })
}

#[derive(serde::Serialize)]
pub struct LlmStatus {
    pub available: bool,
    pub model: Option<String>,
}

/// Extract action items from the meeting transcript.
#[tauri::command]
pub async fn extract_action_items(state: State<'_, AppState>) -> Result<Option<String>, String> {
    let transcript_text = {
        let session_guard = state.session.lock().map_err(|e| e.to_string())?;
        match session_guard.as_ref() {
            Some(session) => session.transcript_text(),
            None => return Err("no meeting session available".into()),
        }
    };

    if transcript_text.is_empty() {
        return Ok(None);
    }

    let llm = state.llm.clone();
    tokio::task::spawn_blocking(move || {
        let llm = llm.lock().map_err(|e| e.to_string())?;
        Ok(llm.action_items(&transcript_text))
    })
    .await
    .map_err(|e| e.to_string())?
}
