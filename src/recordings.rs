use std::path::PathBuf;

use crate::config::Config;

const FILE_PREFIX: &str = "recording-";
const FILE_EXTENSION: &str = "wav";

pub struct RecordingStore;

impl RecordingStore {
    pub fn recordings_dir() -> PathBuf {
        Config::dir().join("recordings")
    }

    pub fn ensure_dir() {
        let dir = Self::recordings_dir();
        if let Err(e) = std::fs::create_dir_all(&dir) {
            eprintln!("Warning: could not create recordings directory: {e}");
        }
    }

    pub fn temp_recording_path() -> PathBuf {
        std::env::temp_dir().join("open-bark-recording.wav")
    }

    pub fn new_recording_path() -> PathBuf {
        Self::ensure_dir();
        let now = chrono_timestamp();
        let unique = &uuid_short();
        let filename = format!("{FILE_PREFIX}{now}-{unique}.{FILE_EXTENSION}");
        Self::recordings_dir().join(filename)
    }

    pub fn list_recordings() -> Vec<(PathBuf, String)> {
        Self::ensure_dir();
        let dir = Self::recordings_dir();
        let Ok(entries) = std::fs::read_dir(&dir) else {
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
        let recordings = Self::list_recordings();
        if recordings.len() <= max_count as usize {
            return;
        }

        for (path, _) in recordings.into_iter().skip(max_count as usize) {
            if let Err(e) = std::fs::remove_file(&path) {
                eprintln!(
                    "Warning: could not remove old recording {}: {e}",
                    path.display()
                );
            }
        }
    }

    #[allow(dead_code)]
    pub fn delete_all() {
        for (path, _) in Self::list_recordings() {
            let _ = std::fs::remove_file(&path);
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
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let mut hasher = DefaultHasher::new();
    SystemTime::now().hash(&mut hasher);
    std::process::id().hash(&mut hasher);
    format!("{:08x}", hasher.finish() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temp_recording_path() {
        let path = RecordingStore::temp_recording_path();
        assert!(path.to_string_lossy().contains("open-bark-recording.wav"));
    }

    #[test]
    fn test_chrono_timestamp() {
        let ts = chrono_timestamp();
        assert!(!ts.is_empty());
        // Should be a number
        assert!(ts.parse::<u64>().is_ok());
    }

    #[test]
    fn test_uuid_short() {
        let id = uuid_short();
        assert_eq!(id.len(), 8);
    }
}
