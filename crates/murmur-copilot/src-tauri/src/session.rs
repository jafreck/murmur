use std::path::PathBuf;

use anyhow::{Context, Result};
use log::info;
use serde::{Deserialize, Serialize};

use crate::meeting::{Speaker, TranscriptEntry};

/// Validate that a session ID is safe for use in file paths.
fn validate_session_id(id: &str) -> Result<()> {
    if id.is_empty() || id.contains(['/', '\\', '.']) {
        anyhow::bail!("invalid session ID: must not contain '/', '\\', or '.'");
    }
    Ok(())
}

/// A lightweight summary returned when listing sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub id: String,
    pub title: Option<String>,
    pub started_at: String,
    pub ended_at: String,
    pub duration_secs: u64,
    pub entry_count: usize,
}

/// A saved meeting session with metadata and content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedSession {
    pub id: String,
    pub title: Option<String>,
    pub started_at: String,
    pub ended_at: String,
    pub duration_secs: u64,
    pub transcript: Vec<TranscriptEntry>,
    pub summary: Option<String>,
    pub action_items: Option<String>,
}

impl SavedSession {
    /// Create a new session from transcript entries.
    #[allow(dead_code)]
    pub fn from_transcript(
        transcript: Vec<TranscriptEntry>,
        started_at: String,
        duration_secs: u64,
    ) -> Self {
        let id = uuid_v4();
        let ended_at = now_iso8601();
        Self {
            id,
            title: None,
            started_at,
            ended_at,
            duration_secs,
            transcript,
            summary: None,
            action_items: None,
        }
    }

    /// Export the session as markdown text.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        let title = self.title.as_deref().unwrap_or("Untitled Meeting");
        md.push_str(&format!("# {title}\n\n"));
        md.push_str(&format!("**Date:** {}\n", self.started_at));
        md.push_str(&format!(
            "**Duration:** {} min {} sec\n\n",
            self.duration_secs / 60,
            self.duration_secs % 60,
        ));

        if let Some(ref summary) = self.summary {
            md.push_str("## Summary\n\n");
            md.push_str(summary);
            md.push_str("\n\n");
        }

        if let Some(ref items) = self.action_items {
            md.push_str("## Action Items\n\n");
            md.push_str(items);
            md.push_str("\n\n");
        }

        md.push_str("## Transcript\n\n");
        for entry in &self.transcript {
            let label = match entry.speaker {
                Speaker::User => "You",
                Speaker::Remote => "Remote",
            };
            md.push_str(&format!("**{label}:** {}\n\n", entry.text));
        }

        md
    }

    fn to_summary(&self) -> SessionSummary {
        SessionSummary {
            id: self.id.clone(),
            title: self.title.clone(),
            started_at: self.started_at.clone(),
            ended_at: self.ended_at.clone(),
            duration_secs: self.duration_secs,
            entry_count: self.transcript.len(),
        }
    }
}

/// Manages persistent storage of meeting sessions as JSON files.
pub struct SessionStore {
    dir: PathBuf,
}

impl SessionStore {
    /// Create a new store, using the configured or default sessions directory.
    pub fn new(custom_dir: Option<&str>) -> Self {
        let dir = match custom_dir {
            Some(d) => PathBuf::from(d),
            None => default_sessions_dir(),
        };
        Self { dir }
    }

    /// Save a session to disk.
    #[allow(dead_code)]
    pub fn save(&self, session: &SavedSession) -> Result<()> {
        validate_session_id(&session.id)?;
        std::fs::create_dir_all(&self.dir)
            .with_context(|| format!("failed to create sessions dir: {}", self.dir.display()))?;
        let path = self.dir.join(format!("{}.json", session.id));
        let json = serde_json::to_string_pretty(session)?;
        std::fs::write(&path, json)
            .with_context(|| format!("failed to write session: {}", path.display()))?;
        info!("saved session {} to {}", session.id, path.display());
        Ok(())
    }

    /// Load a full session by ID.
    pub fn load(&self, id: &str) -> Result<SavedSession> {
        validate_session_id(id)?;
        let path = self.dir.join(format!("{id}.json"));
        let json =
            std::fs::read_to_string(&path).with_context(|| format!("session not found: {id}"))?;
        let session: SavedSession = serde_json::from_str(&json)
            .with_context(|| format!("failed to parse session: {id}"))?;
        Ok(session)
    }

    /// List all saved sessions (lightweight summaries sorted newest-first).
    pub fn list(&self) -> Result<Vec<SessionSummary>> {
        if !self.dir.exists() {
            return Ok(Vec::new());
        }

        let mut summaries = Vec::new();
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "json") {
                match std::fs::read_to_string(&path) {
                    Ok(json) => match serde_json::from_str::<SavedSession>(&json) {
                        Ok(session) => summaries.push(session.to_summary()),
                        Err(e) => {
                            log::warn!("skipping malformed session {}: {e}", path.display());
                        }
                    },
                    Err(e) => {
                        log::warn!("could not read {}: {e}", path.display());
                    }
                }
            }
        }

        // Sort newest first.
        summaries.sort_by(|a, b| b.started_at.cmp(&a.started_at));
        Ok(summaries)
    }

    /// Delete a session by ID.
    pub fn delete(&self, id: &str) -> Result<()> {
        validate_session_id(id)?;
        let path = self.dir.join(format!("{id}.json"));
        std::fs::remove_file(&path).with_context(|| format!("failed to delete session: {id}"))?;
        info!("deleted session {id}");
        Ok(())
    }
}

fn default_sessions_dir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("murmur-copilot")
        .join("sessions")
}

#[allow(dead_code)]
fn now_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Simple ISO 8601 without pulling in chrono.
    format_unix_timestamp(secs)
}

#[allow(dead_code)]
fn format_unix_timestamp(secs: u64) -> String {
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let minutes = (time_secs % 3600) / 60;
    let seconds = time_secs % 60;

    // Days since epoch to Y-M-D (simplified Gregorian).
    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

#[allow(dead_code)]
fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    // Algorithm from https://howardhinnant.github.io/date_algorithms.html
    days += 719_468;
    let era = days / 146_097;
    let doe = days - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[allow(dead_code)]
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    // Simple pseudo-UUID from timestamp + random bits via address-space noise.
    let hash = nanos
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    format!(
        "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
        (hash >> 96) as u32,
        (hash >> 80) as u16,
        (hash >> 64) as u16 & 0x0FFF,
        ((hash >> 48) as u16 & 0x3FFF) | 0x8000,
        hash as u64 & 0xFFFF_FFFF_FFFF,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uuid_v4_format() {
        let id = uuid_v4();
        assert_eq!(id.len(), 36);
        assert_eq!(id.chars().filter(|&c| c == '-').count(), 4);
    }

    #[test]
    fn test_format_unix_timestamp() {
        let ts = format_unix_timestamp(0);
        assert_eq!(ts, "1970-01-01T00:00:00Z");
    }

    #[test]
    fn test_saved_session_to_markdown() {
        let session = SavedSession {
            id: "test-id".into(),
            title: Some("Sprint Standup".into()),
            started_at: "2025-01-01T10:00:00Z".into(),
            ended_at: "2025-01-01T10:15:00Z".into(),
            duration_secs: 900,
            transcript: vec![TranscriptEntry {
                speaker: Speaker::User,
                text: "Hello team".into(),
                timestamp_ms: 0,
            }],
            summary: Some("Quick standup call.".into()),
            action_items: Some("- Review PR #42".into()),
        };
        let md = session.to_markdown();
        assert!(md.contains("# Sprint Standup"));
        assert!(md.contains("**Duration:** 15 min 0 sec"));
        assert!(md.contains("Quick standup call."));
        assert!(md.contains("Review PR #42"));
        assert!(md.contains("**You:** Hello team"));
    }

    #[test]
    fn test_session_store_roundtrip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = SessionStore::new(Some(tmp.path().to_str().unwrap()));

        let session = SavedSession {
            id: "test-123".into(),
            title: Some("Test".into()),
            started_at: "2025-01-01T00:00:00Z".into(),
            ended_at: "2025-01-01T00:30:00Z".into(),
            duration_secs: 1800,
            transcript: vec![],
            summary: None,
            action_items: None,
        };

        store.save(&session).unwrap();
        let loaded = store.load("test-123").unwrap();
        assert_eq!(loaded.id, "test-123");
        assert_eq!(loaded.title.as_deref(), Some("Test"));

        let list = store.list().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, "test-123");

        store.delete("test-123").unwrap();
        let list = store.list().unwrap();
        assert!(list.is_empty());
    }

    #[test]
    fn test_session_store_list_empty_dir() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = SessionStore::new(Some(tmp.path().to_str().unwrap()));
        let list = store.list().unwrap();
        assert!(list.is_empty());
    }

    #[test]
    fn test_session_store_list_nonexistent_dir() {
        let store = SessionStore::new(Some("/nonexistent/path/sessions"));
        let list = store.list().unwrap();
        assert!(list.is_empty());
    }
}
