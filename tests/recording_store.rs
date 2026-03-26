//! Integration tests for RecordingStore: file creation, listing, pruning,
//! and path management with real temporary directories.

use murmur::audio::recordings::RecordingStore;
use std::fs;

// ═══════════════════════════════════════════════════════════════════════
//  Directory management
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ensure_dir_creates_nested_directories() {
    let tmp = tempfile::TempDir::new().unwrap();
    let deep = tmp.path().join("a").join("b").join("c");
    assert!(!deep.exists());

    RecordingStore::ensure_dir_at(&deep);
    assert!(deep.exists());
    assert!(deep.is_dir());
}

#[test]
fn ensure_dir_is_idempotent() {
    let tmp = tempfile::TempDir::new().unwrap();
    let dir = tmp.path().join("recordings");

    RecordingStore::ensure_dir_at(&dir);
    assert!(dir.exists());

    // Calling again should not error
    RecordingStore::ensure_dir_at(&dir);
    assert!(dir.exists());
}

// ═══════════════════════════════════════════════════════════════════════
//  Recording path generation
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn new_recording_path_creates_parent_dir() {
    let tmp = tempfile::TempDir::new().unwrap();
    let dir = tmp.path().join("new_recordings");
    assert!(!dir.exists());

    let path = RecordingStore::new_recording_path_in(&dir);
    assert!(dir.exists(), "should have created parent dir");
    assert!(path.starts_with(&dir));
}

#[test]
fn new_recording_path_has_correct_format() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = RecordingStore::new_recording_path_in(tmp.path());

    let name = path.file_name().unwrap().to_string_lossy();
    assert!(name.starts_with("recording-"), "got: {name}");
    assert!(name.ends_with(".wav"), "got: {name}");
}

#[test]
fn recording_paths_are_unique() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut paths = std::collections::HashSet::new();

    for _ in 0..20 {
        let path = RecordingStore::new_recording_path_in(tmp.path());
        assert!(paths.insert(path), "duplicate path generated");
    }
}

#[test]
fn temp_recording_path_is_in_temp_dir() {
    let path = RecordingStore::temp_recording_path();
    let temp_dir = std::env::temp_dir();
    assert!(
        path.starts_with(&temp_dir),
        "expected path in {}, got {}",
        temp_dir.display(),
        path.display()
    );
    assert!(path.to_string_lossy().contains("murmur-"));
    assert!(path.to_string_lossy().ends_with(".wav"));
}

// ═══════════════════════════════════════════════════════════════════════
//  Listing recordings
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn list_empty_dir() {
    let tmp = tempfile::TempDir::new().unwrap();
    let recordings = RecordingStore::list_recordings_in(tmp.path());
    assert!(recordings.is_empty());
}

#[test]
fn list_ignores_non_recording_files() {
    let tmp = tempfile::TempDir::new().unwrap();

    // Create non-recording files
    fs::write(tmp.path().join("notes.txt"), "").unwrap();
    fs::write(tmp.path().join("config.json"), "{}").unwrap();
    fs::write(tmp.path().join("audio.mp3"), "").unwrap();

    let recordings = RecordingStore::list_recordings_in(tmp.path());
    assert!(recordings.is_empty());
}

#[test]
fn list_finds_recording_files() {
    let tmp = tempfile::TempDir::new().unwrap();

    fs::write(tmp.path().join("recording-001-aaa.wav"), "").unwrap();
    fs::write(tmp.path().join("recording-002-bbb.wav"), "").unwrap();
    fs::write(tmp.path().join("recording-003-ccc.wav"), "").unwrap();
    fs::write(tmp.path().join("other.wav"), "").unwrap(); // Not a recording

    let recordings = RecordingStore::list_recordings_in(tmp.path());
    assert_eq!(recordings.len(), 3);
}

#[test]
fn list_sorted_newest_first() {
    let tmp = tempfile::TempDir::new().unwrap();

    fs::write(tmp.path().join("recording-100-aaa.wav"), "").unwrap();
    fs::write(tmp.path().join("recording-300-ccc.wav"), "").unwrap();
    fs::write(tmp.path().join("recording-200-bbb.wav"), "").unwrap();

    let recordings = RecordingStore::list_recordings_in(tmp.path());
    assert_eq!(recordings.len(), 3);

    // Should be sorted descending by name (newest first)
    assert!(recordings[0].1 > recordings[1].1);
    assert!(recordings[1].1 > recordings[2].1);
}

#[test]
fn list_returns_full_paths() {
    let tmp = tempfile::TempDir::new().unwrap();
    fs::write(tmp.path().join("recording-001-aaa.wav"), "data").unwrap();

    let recordings = RecordingStore::list_recordings_in(tmp.path());
    assert_eq!(recordings.len(), 1);
    assert!(recordings[0].0.is_absolute() || recordings[0].0.starts_with(tmp.path()));
    assert!(recordings[0].0.exists());
}

// ═══════════════════════════════════════════════════════════════════════
//  Pruning
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn prune_removes_oldest_files() {
    let tmp = tempfile::TempDir::new().unwrap();

    fs::write(tmp.path().join("recording-001-aaa.wav"), "old").unwrap();
    fs::write(tmp.path().join("recording-002-bbb.wav"), "mid").unwrap();
    fs::write(tmp.path().join("recording-003-ccc.wav"), "new").unwrap();

    RecordingStore::prune_in(tmp.path(), 2);

    let remaining = RecordingStore::list_recordings_in(tmp.path());
    assert_eq!(remaining.len(), 2);
    // Newest two should remain
    assert!(remaining.iter().any(|(_, n)| n.contains("003")));
    assert!(remaining.iter().any(|(_, n)| n.contains("002")));
    // Oldest should be gone
    assert!(!remaining.iter().any(|(_, n)| n.contains("001")));
}

#[test]
fn prune_noop_when_under_limit() {
    let tmp = tempfile::TempDir::new().unwrap();

    fs::write(tmp.path().join("recording-001-aaa.wav"), "").unwrap();
    fs::write(tmp.path().join("recording-002-bbb.wav"), "").unwrap();

    RecordingStore::prune_in(tmp.path(), 5);

    let remaining = RecordingStore::list_recordings_in(tmp.path());
    assert_eq!(remaining.len(), 2);
}

#[test]
fn prune_noop_when_at_limit() {
    let tmp = tempfile::TempDir::new().unwrap();

    fs::write(tmp.path().join("recording-001-aaa.wav"), "").unwrap();
    fs::write(tmp.path().join("recording-002-bbb.wav"), "").unwrap();

    RecordingStore::prune_in(tmp.path(), 2);

    let remaining = RecordingStore::list_recordings_in(tmp.path());
    assert_eq!(remaining.len(), 2);
}

#[test]
fn prune_to_zero_removes_all() {
    let tmp = tempfile::TempDir::new().unwrap();

    fs::write(tmp.path().join("recording-001-aaa.wav"), "").unwrap();
    fs::write(tmp.path().join("recording-002-bbb.wav"), "").unwrap();

    RecordingStore::prune_in(tmp.path(), 0);

    let remaining = RecordingStore::list_recordings_in(tmp.path());
    assert!(remaining.is_empty());
}

#[test]
fn prune_to_one_keeps_newest() {
    let tmp = tempfile::TempDir::new().unwrap();

    fs::write(tmp.path().join("recording-001-aaa.wav"), "").unwrap();
    fs::write(tmp.path().join("recording-002-bbb.wav"), "").unwrap();
    fs::write(tmp.path().join("recording-003-ccc.wav"), "").unwrap();

    RecordingStore::prune_in(tmp.path(), 1);

    let remaining = RecordingStore::list_recordings_in(tmp.path());
    assert_eq!(remaining.len(), 1);
    assert!(remaining[0].1.contains("003"));
}

#[test]
fn prune_does_not_affect_non_recordings() {
    let tmp = tempfile::TempDir::new().unwrap();

    fs::write(tmp.path().join("recording-001-aaa.wav"), "").unwrap();
    fs::write(tmp.path().join("recording-002-bbb.wav"), "").unwrap();
    fs::write(tmp.path().join("notes.txt"), "keep me").unwrap();
    fs::write(tmp.path().join("config.json"), "{}").unwrap();

    RecordingStore::prune_in(tmp.path(), 1);

    // Non-recording files should survive
    assert!(tmp.path().join("notes.txt").exists());
    assert!(tmp.path().join("config.json").exists());
}

#[test]
fn prune_empty_dir() {
    let tmp = tempfile::TempDir::new().unwrap();
    // Should not panic
    RecordingStore::prune_in(tmp.path(), 5);
}

// ═══════════════════════════════════════════════════════════════════════
//  End-to-end: create → list → prune cycle
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn create_list_prune_cycle() {
    let tmp = tempfile::TempDir::new().unwrap();

    // Create recordings
    let p1 = RecordingStore::new_recording_path_in(tmp.path());
    fs::write(&p1, "audio1").unwrap();
    std::thread::sleep(std::time::Duration::from_millis(10));

    let p2 = RecordingStore::new_recording_path_in(tmp.path());
    fs::write(&p2, "audio2").unwrap();
    std::thread::sleep(std::time::Duration::from_millis(10));

    let p3 = RecordingStore::new_recording_path_in(tmp.path());
    fs::write(&p3, "audio3").unwrap();

    // List should show all 3
    let list = RecordingStore::list_recordings_in(tmp.path());
    assert_eq!(list.len(), 3);

    // Prune to 2
    RecordingStore::prune_in(tmp.path(), 2);
    let remaining = RecordingStore::list_recordings_in(tmp.path());
    assert_eq!(remaining.len(), 2);
}

// ═══════════════════════════════════════════════════════════════════════
//  Large-scale pruning
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn prune_large_number_of_files() {
    let tmp = tempfile::TempDir::new().unwrap();

    for i in 0..50 {
        let name = format!("recording-{:06}-{:016x}.wav", i, i as u64);
        fs::write(tmp.path().join(name), "").unwrap();
    }

    let before = RecordingStore::list_recordings_in(tmp.path());
    assert_eq!(before.len(), 50);

    RecordingStore::prune_in(tmp.path(), 10);

    let after = RecordingStore::list_recordings_in(tmp.path());
    assert_eq!(after.len(), 10);
}

// ═══════════════════════════════════════════════════════════════════════
//  recordings_dir returns a sensible path
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn recordings_dir_contains_recordings() {
    let dir = RecordingStore::recordings_dir();
    assert!(
        dir.to_string_lossy().contains("recordings"),
        "got: {}",
        dir.display()
    );
}
