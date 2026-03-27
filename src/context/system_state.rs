use std::sync::{Arc, Mutex};

/// Maximum number of clipboard characters to capture as context.
const MAX_CLIPBOARD_CHARS: usize = 500;

/// Maximum number of characters to keep in the recent text buffer.
const DEFAULT_MAX_RECENT_CHARS: usize = 1000;

/// Maximum number of individual entries to keep.
const MAX_ENTRIES: usize = 50;

/// Provides the current clipboard text as context.
///
/// Reading the clipboard before dictation helps Whisper understand what the user
/// might be referencing — e.g., if they just copied a function name, they're likely
/// to say it during dictation.
pub struct ClipboardWatcher;

impl ClipboardWatcher {
    pub fn new() -> Self {
        ClipboardWatcher
    }

    /// Read the current clipboard text, if any.
    /// Returns None if clipboard is empty, contains non-text data, or access fails.
    pub fn get_clipboard_text(&self) -> Option<String> {
        match arboard::Clipboard::new() {
            Ok(mut cb) => match cb.get_text() {
                Ok(text) if !text.trim().is_empty() => {
                    let trimmed = text.trim();
                    if trimmed.len() > MAX_CLIPBOARD_CHARS {
                        Some(trimmed[..MAX_CLIPBOARD_CHARS].to_string())
                    } else {
                        Some(trimmed.to_string())
                    }
                }
                _ => None,
            },
            Err(e) => {
                log::debug!("Failed to access clipboard: {e}");
                None
            }
        }
    }
}

/// Tracks recently dictated text to provide continuity context.
///
/// When the user dictates multiple utterances in sequence, the earlier ones
/// provide valuable context for the later ones. This tracker maintains a
/// rolling window of recent transcription output.
pub struct RecentTextTracker {
    buffer: Arc<Mutex<RecentTextBuffer>>,
}

struct RecentTextBuffer {
    /// Recent text entries, newest last
    entries: Vec<String>,
    /// Maximum total characters to retain
    max_chars: usize,
}

impl RecentTextTracker {
    pub fn new() -> Self {
        Self {
            buffer: Arc::new(Mutex::new(RecentTextBuffer {
                entries: Vec::new(),
                max_chars: DEFAULT_MAX_RECENT_CHARS,
            })),
        }
    }

    /// Record a new transcription result.
    pub fn push(&self, text: &str) {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return;
        }
        if let Ok(mut buf) = self.buffer.lock() {
            buf.entries.push(trimmed.to_string());
            // Trim entries if too many
            while buf.entries.len() > MAX_ENTRIES {
                buf.entries.remove(0);
            }
            // Trim total character count
            let mut total_chars: usize = buf.entries.iter().map(|e| e.len()).sum();
            while total_chars > buf.max_chars && !buf.entries.is_empty() {
                total_chars -= buf.entries[0].len();
                buf.entries.remove(0);
            }
        }
    }

    /// Get the recent text as a single string (entries joined by spaces).
    /// Returns None if no recent text exists.
    pub fn get_recent_text(&self) -> Option<String> {
        if let Ok(buf) = self.buffer.lock() {
            if buf.entries.is_empty() {
                return None;
            }
            Some(buf.entries.join(" "))
        } else {
            None
        }
    }

    /// Clear all tracked text.
    pub fn clear(&self) {
        if let Ok(mut buf) = self.buffer.lock() {
            buf.entries.clear();
        }
    }

    /// Get the number of tracked entries.
    pub fn entry_count(&self) -> usize {
        self.buffer.lock().map(|buf| buf.entries.len()).unwrap_or(0)
    }

    /// Get a clone of the tracker that shares the same buffer.
    /// This is useful for sharing between the app loop and context providers.
    pub fn shared(&self) -> Self {
        Self {
            buffer: Arc::clone(&self.buffer),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- ClipboardWatcher --

    #[test]
    fn test_clipboard_watcher_new() {
        let watcher = ClipboardWatcher::new();
        let _ = watcher;
    }

    #[test]
    fn test_clipboard_watcher_returns_option() {
        let watcher = ClipboardWatcher::new();
        // In test environment, clipboard access may or may not work
        let result = watcher.get_clipboard_text();
        let _ = result; // Just verify it doesn't panic
    }

    #[test]
    fn test_max_clipboard_chars_positive() {
        assert!(MAX_CLIPBOARD_CHARS > 0);
    }

    // -- RecentTextTracker --

    #[test]
    fn test_recent_text_tracker_new_empty() {
        let tracker = RecentTextTracker::new();
        assert!(tracker.get_recent_text().is_none());
        assert_eq!(tracker.entry_count(), 0);
    }

    #[test]
    fn test_recent_text_tracker_push_and_get() {
        let tracker = RecentTextTracker::new();
        tracker.push("hello world");
        assert_eq!(tracker.entry_count(), 1);
        let text = tracker.get_recent_text().unwrap();
        assert_eq!(text, "hello world");
    }

    #[test]
    fn test_recent_text_tracker_multiple_entries() {
        let tracker = RecentTextTracker::new();
        tracker.push("first");
        tracker.push("second");
        tracker.push("third");
        assert_eq!(tracker.entry_count(), 3);
        let text = tracker.get_recent_text().unwrap();
        assert!(text.contains("first"));
        assert!(text.contains("second"));
        assert!(text.contains("third"));
    }

    #[test]
    fn test_recent_text_tracker_ignores_empty() {
        let tracker = RecentTextTracker::new();
        tracker.push("");
        tracker.push("   ");
        tracker.push("\n\t");
        assert_eq!(tracker.entry_count(), 0);
        assert!(tracker.get_recent_text().is_none());
    }

    #[test]
    fn test_recent_text_tracker_trims_whitespace() {
        let tracker = RecentTextTracker::new();
        tracker.push("  hello  ");
        let text = tracker.get_recent_text().unwrap();
        assert_eq!(text, "hello");
    }

    #[test]
    fn test_recent_text_tracker_max_entries() {
        let tracker = RecentTextTracker::new();
        for i in 0..(MAX_ENTRIES + 10) {
            tracker.push(&format!("entry {i}"));
        }
        assert!(tracker.entry_count() <= MAX_ENTRIES);
    }

    #[test]
    fn test_recent_text_tracker_max_chars() {
        let tracker = RecentTextTracker::new();
        // Push a lot of text to exceed the character limit
        for _ in 0..100 {
            tracker.push(&"a".repeat(100));
        }
        let text = tracker.get_recent_text().unwrap();
        assert!(text.len() <= DEFAULT_MAX_RECENT_CHARS + 200); // Allow some slack for joining
    }

    #[test]
    fn test_recent_text_tracker_clear() {
        let tracker = RecentTextTracker::new();
        tracker.push("hello");
        tracker.push("world");
        assert_eq!(tracker.entry_count(), 2);
        tracker.clear();
        assert_eq!(tracker.entry_count(), 0);
        assert!(tracker.get_recent_text().is_none());
    }

    #[test]
    fn test_recent_text_tracker_shared() {
        let tracker = RecentTextTracker::new();
        let shared = tracker.shared();

        tracker.push("from original");
        assert_eq!(shared.entry_count(), 1);
        assert_eq!(shared.get_recent_text().unwrap(), "from original");

        shared.push("from shared");
        assert_eq!(tracker.entry_count(), 2);
    }

    #[test]
    fn test_recent_text_tracker_preserves_order() {
        let tracker = RecentTextTracker::new();
        tracker.push("alpha");
        tracker.push("beta");
        tracker.push("gamma");
        let text = tracker.get_recent_text().unwrap();
        let alpha_pos = text.find("alpha").unwrap();
        let beta_pos = text.find("beta").unwrap();
        let gamma_pos = text.find("gamma").unwrap();
        assert!(alpha_pos < beta_pos);
        assert!(beta_pos < gamma_pos);
    }

    #[test]
    fn test_recent_text_tracker_thread_safe() {
        let tracker = RecentTextTracker::new();
        let shared = tracker.shared();

        let handle = std::thread::spawn(move || {
            for i in 0..10 {
                shared.push(&format!("thread entry {i}"));
            }
        });

        for i in 0..10 {
            tracker.push(&format!("main entry {i}"));
        }

        handle.join().unwrap();
        assert!(tracker.entry_count() > 0);
        assert!(tracker.entry_count() <= 20);
    }

    // -- Constants --

    #[test]
    fn test_constants() {
        assert!(DEFAULT_MAX_RECENT_CHARS > 0);
        assert!(MAX_ENTRIES > 0);
        assert!(MAX_CLIPBOARD_CHARS > 0);
    }
}
