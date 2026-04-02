use std::path::PathBuf;

use std::sync::atomic::{AtomicU64, Ordering};

use crate::config::Config;

const FILE_PREFIX: &str = "recording-";
const FILE_EXTENSION: &str = "wav";

pub struct RecordingStore;

impl RecordingStore {
    pub fn recordings_dir() -> PathBuf {
        Config::dir().join("recordings")
    }

    pub fn ensure_dir_at(dir: &std::path::Path) {
        if let Err(e) = std::fs::create_dir_all(dir) {
            log::warn!("Could not create recordings directory: {e}");
        }
    }

    pub fn temp_recording_path() -> PathBuf {
        let unique = uuid_short();
        std::env::temp_dir().join(format!("murmur-{unique}.wav"))
    }

    pub fn new_recording_path() -> PathBuf {
        Self::new_recording_path_in(&Self::recordings_dir())
    }

    pub fn new_recording_path_in(dir: &std::path::Path) -> PathBuf {
        Self::ensure_dir_at(dir);
        let now = chrono_timestamp();
        let unique = &uuid_short();
        let filename = format!("{FILE_PREFIX}{now}-{unique}.{FILE_EXTENSION}");
        dir.join(filename)
    }

    pub fn list_recordings_in(dir: &std::path::Path) -> Vec<(PathBuf, String)> {
        Self::ensure_dir_at(dir);
        let Ok(entries) = std::fs::read_dir(dir) else {
            return vec![];
        };

        let mut recordings: Vec<(PathBuf, String)> = entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                name.starts_with(FILE_PREFIX) && name.ends_with(&format!(".{FILE_EXTENSION}"))
            })
            .map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                (e.path(), name)
            })
            .collect();

        // Sort by name descending (newest first, since names contain timestamps)
        recordings.sort_by(|a, b| b.1.cmp(&a.1));
        recordings
    }

    pub fn prune(max_count: u32) {
        Self::prune_in(&Self::recordings_dir(), max_count);
    }

    pub fn prune_in(dir: &std::path::Path, max_count: u32) {
        let recordings = Self::list_recordings_in(dir);
        if recordings.len() <= max_count as usize {
            return;
        }

        for (path, _) in recordings.into_iter().skip(max_count as usize) {
            if let Err(e) = std::fs::remove_file(&path) {
                log::warn!("Could not remove old recording {}: {e}", path.display());
            }
        }
    }
}

fn chrono_timestamp() -> String {
    // Simple timestamp without pulling in the chrono crate
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", duration.as_secs())
}

fn uuid_short() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{:016x}", (ts as u64).wrapping_add(count))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temp_recording_path() {
        let path = RecordingStore::temp_recording_path();
        assert!(path.to_string_lossy().contains("murmur-"));
    }

    #[test]
    fn test_chrono_timestamp() {
        let ts = chrono_timestamp();
        assert!(!ts.is_empty());
        assert!(ts.parse::<u64>().is_ok());
    }

    #[test]
    fn test_uuid_short() {
        let id = uuid_short();
        assert_eq!(id.len(), 16);
    }

    #[test]
    fn test_ensure_dir_creates_directory() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path().join("test_recordings");
        assert!(!dir.exists());
        RecordingStore::ensure_dir_at(&dir);
        assert!(dir.exists());
    }

    #[test]
    fn test_new_recording_path_format() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = RecordingStore::new_recording_path_in(tmp.path());
        let name = path.file_name().unwrap().to_string_lossy();
        assert!(name.starts_with("recording-"));
        assert!(name.ends_with(".wav"));
    }

    #[test]
    fn test_list_recordings_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let recordings = RecordingStore::list_recordings_in(tmp.path());
        assert!(recordings.is_empty());
    }

    #[test]
    fn test_list_recordings_filters_non_recording_files() {
        let tmp = tempfile::TempDir::new().unwrap();
        // Create a non-recording file
        std::fs::write(tmp.path().join("other.txt"), "not a recording").unwrap();
        // Create a recording file
        std::fs::write(tmp.path().join("recording-123-abc.wav"), "").unwrap();
        let recordings = RecordingStore::list_recordings_in(tmp.path());
        assert_eq!(recordings.len(), 1);
        assert!(recordings[0].1.starts_with("recording-"));
    }

    #[test]
    fn test_list_recordings_sorted_descending() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("recording-001-aaa.wav"), "").unwrap();
        std::fs::write(tmp.path().join("recording-003-ccc.wav"), "").unwrap();
        std::fs::write(tmp.path().join("recording-002-bbb.wav"), "").unwrap();
        let recordings = RecordingStore::list_recordings_in(tmp.path());
        assert_eq!(recordings.len(), 3);
        assert!(recordings[0].1 > recordings[1].1);
        assert!(recordings[1].1 > recordings[2].1);
    }

    #[test]
    fn test_prune_removes_oldest() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("recording-001-aaa.wav"), "").unwrap();
        std::fs::write(tmp.path().join("recording-002-bbb.wav"), "").unwrap();
        std::fs::write(tmp.path().join("recording-003-ccc.wav"), "").unwrap();

        RecordingStore::prune_in(tmp.path(), 2);

        let remaining = RecordingStore::list_recordings_in(tmp.path());
        assert_eq!(remaining.len(), 2);
        // Newest should remain
        assert!(remaining.iter().any(|(_, n)| n.contains("003")));
        assert!(remaining.iter().any(|(_, n)| n.contains("002")));
    }

    #[test]
    fn test_prune_noop_when_under_limit() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("recording-001-aaa.wav"), "").unwrap();
        RecordingStore::prune_in(tmp.path(), 5);
        let remaining = RecordingStore::list_recordings_in(tmp.path());
        assert_eq!(remaining.len(), 1);
    }

    #[test]
    fn test_delete_all() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("recording-001-aaa.wav"), "").unwrap();
        std::fs::write(tmp.path().join("recording-002-bbb.wav"), "").unwrap();
        // Inline delete-all logic (the wrapper was removed as dead code)
        for (path, _) in RecordingStore::list_recordings_in(tmp.path()) {
            let _ = std::fs::remove_file(&path);
        }
        let remaining = RecordingStore::list_recordings_in(tmp.path());
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_recordings_dir_path() {
        let dir = RecordingStore::recordings_dir();
        assert!(dir.to_string_lossy().contains("recordings"));
    }

    #[test]
    fn test_new_recording_paths_are_unique() {
        let tmp = tempfile::TempDir::new().unwrap();
        let p1 = RecordingStore::new_recording_path_in(tmp.path());
        std::thread::sleep(std::time::Duration::from_millis(10));
        let p2 = RecordingStore::new_recording_path_in(tmp.path());
        assert_ne!(p1, p2);
    }

    #[test]
    fn test_prune_wrapper() {
        // Calls the real dir but safe with a large limit
        RecordingStore::prune(1000);
    }

    #[test]
    fn test_new_recording_path_wrapper() {
        let path = RecordingStore::new_recording_path();
        let name = path.file_name().unwrap().to_string_lossy();
        assert!(name.starts_with("recording-"));
        assert!(name.ends_with(".wav"));
    }
}
