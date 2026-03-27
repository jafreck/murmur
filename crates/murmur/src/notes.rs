//! Timestamped note file management.
//!
//! Each dictation session saves its transcript to a separate file under
//! the configured notes directory. Files are named with ISO 8601 timestamps
//! so they sort chronologically.

use anyhow::Result;
use std::path::{Path, PathBuf};

/// Manages note files in a directory.
pub struct NotesManager {
    dir: PathBuf,
}

impl NotesManager {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    /// Ensure the notes directory exists.
    pub fn ensure_dir(&self) -> Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        Ok(())
    }

    /// Generate a note file path for the current timestamp.
    pub fn new_note_path(&self) -> PathBuf {
        let ts = chrono::Local::now().format("%Y-%m-%dT%H-%M-%S");
        self.dir.join(format!("note-{ts}.txt"))
    }

    /// Generate a note file path for a given timestamp string.
    pub fn note_path_for(&self, timestamp: &str) -> PathBuf {
        self.dir.join(format!("note-{timestamp}.txt"))
    }

    /// Save text to a new timestamped note file. Returns the path written.
    pub fn save(&self, text: &str) -> Result<PathBuf> {
        self.ensure_dir()?;
        let path = self.new_note_path();
        std::fs::write(&path, text)?;
        log::info!("Note saved: {}", path.display());
        Ok(path)
    }

    /// Save text to a specific path within the notes directory.
    pub fn save_to(&self, path: &Path, text: &str) -> Result<()> {
        self.ensure_dir()?;
        std::fs::write(path, text)?;
        Ok(())
    }

    /// List all note files, sorted by name (chronological).
    pub fn list_notes(&self) -> Result<Vec<PathBuf>> {
        if !self.dir.exists() {
            return Ok(Vec::new());
        }
        let mut notes: Vec<PathBuf> = std::fs::read_dir(&self.dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension().is_some_and(|ext| ext == "txt")
                    && p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("note-"))
            })
            .collect();
        notes.sort();
        Ok(notes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_note_path_format() {
        let mgr = NotesManager::new(PathBuf::from("/tmp/murmur-notes"));
        let path = mgr.new_note_path();
        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(name.starts_with("note-"));
        assert!(name.ends_with(".txt"));
    }

    #[test]
    fn test_note_path_for() {
        let mgr = NotesManager::new(PathBuf::from("/tmp/murmur-notes"));
        let path = mgr.note_path_for("2026-03-27T10-00-00");
        assert_eq!(
            path,
            PathBuf::from("/tmp/murmur-notes/note-2026-03-27T10-00-00.txt")
        );
    }

    #[test]
    fn test_save_and_list() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = NotesManager::new(dir.path().to_path_buf());

        let path = mgr.save("hello world").unwrap();
        assert!(path.exists());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello world");

        let notes = mgr.list_notes().unwrap();
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0], path);
    }

    #[test]
    fn test_list_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = NotesManager::new(dir.path().to_path_buf());
        let notes = mgr.list_notes().unwrap();
        assert!(notes.is_empty());
    }

    #[test]
    fn test_list_nonexistent_dir() {
        let mgr = NotesManager::new(PathBuf::from("/nonexistent/murmur-notes"));
        let notes = mgr.list_notes().unwrap();
        assert!(notes.is_empty());
    }

    #[test]
    fn test_save_to() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = NotesManager::new(dir.path().to_path_buf());
        let path = dir.path().join("note-custom.txt");
        mgr.save_to(&path, "custom text").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "custom text");
    }
}
