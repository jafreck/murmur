//! Cursor/caret context provider using macOS Accessibility API.
//!
//! Reads the text surrounding the cursor in the focused text field to provide
//! context for Whisper transcription biasing.

/// The number of characters before the cursor to capture for context.
#[cfg(any(target_os = "macos", test))]
const CONTEXT_CHARS_BEFORE: usize = 200;

/// The number of characters after the cursor to capture for context.
#[cfg(any(target_os = "macos", test))]
const CONTEXT_CHARS_AFTER: usize = 50;

/// Text surrounding the cursor position.
#[derive(Debug, Clone)]
pub struct SurroundingText {
    /// Text before the cursor (up to CONTEXT_CHARS_BEFORE characters).
    pub before: String,
    /// Text after the cursor (up to CONTEXT_CHARS_AFTER characters).
    pub after: String,
    /// The full value of the text field (if available).
    pub full_value: Option<String>,
}

/// Provides text context around the cursor position in the focused application.
pub struct CursorContext;

impl CursorContext {
    pub fn new() -> Self {
        CursorContext
    }

    /// Attempt to read the text surrounding the cursor in the focused text field.
    /// Returns `None` if accessibility access is denied or no text field is focused.
    pub fn get_surrounding_text(&self) -> Option<SurroundingText> {
        #[cfg(target_os = "macos")]
        {
            macos::get_surrounding_text()
        }
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
    }
}

impl Default for CursorContext {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// macOS implementation using shared platform::ax helpers
// ---------------------------------------------------------------------------
#[cfg(target_os = "macos")]
mod macos {
    use super::{SurroundingText, CONTEXT_CHARS_AFTER, CONTEXT_CHARS_BEFORE};
    use crate::platform::ax;

    pub(super) fn get_surrounding_text() -> Option<SurroundingText> {
        // 1. System-wide element
        let system = ax::ax_system_wide();
        if system.is_null() {
            log::debug!("cursor context: failed to create system-wide AX element");
            return None;
        }

        // Create attribute name strings at runtime (avoids linking HIServices sub-framework)
        let attr_focused = ax::cfstring_create(b"AXFocusedUIElement\0");
        let attr_value = ax::cfstring_create(b"AXValue\0");
        let attr_range = ax::cfstring_create(b"AXSelectedTextRange\0");

        // 2. Focused UI element
        let focused = match ax::ax_copy_attr(system.0, attr_focused.0) {
            Some(f) => f,
            None => {
                log::debug!("cursor context: no focused UI element");
                return None;
            }
        };

        // 3. Text value
        let value_ref = match ax::ax_copy_attr(focused.0, attr_value.0) {
            Some(v) => v,
            None => {
                log::debug!("cursor context: focused element has no value attribute");
                return None;
            }
        };

        let full_text = match ax::cfstring_to_string(value_ref.0) {
            Some(t) => t,
            None => {
                log::debug!("cursor context: could not convert value to string");
                return None;
            }
        };

        // 4. Selected text range (cursor position)
        let range_ref = match ax::ax_copy_attr(focused.0, attr_range.0) {
            Some(r) => r,
            None => {
                log::debug!("cursor context: no selected text range attribute");
                return None;
            }
        };

        let range = match ax::ax_value_to_range(range_ref.0) {
            Some(r) => r,
            None => {
                log::debug!("cursor context: could not extract CFRange from AXValue");
                return None;
            }
        };

        // 5. Split text around the cursor
        let cursor_pos = range.location as usize;

        // Clamp to actual text length
        let chars: Vec<char> = full_text.chars().collect();
        let cursor_pos = cursor_pos.min(chars.len());

        let start = cursor_pos.saturating_sub(CONTEXT_CHARS_BEFORE);
        let end = (cursor_pos + CONTEXT_CHARS_AFTER).min(chars.len());

        let before: String = chars[start..cursor_pos].iter().collect();
        let after: String = chars[cursor_pos..end].iter().collect();

        Some(SurroundingText {
            before,
            after,
            full_value: Some(full_text),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cursor_context_new() {
        let ctx = CursorContext::new();
        let _ = ctx;
    }

    #[test]
    fn test_surrounding_text_struct() {
        let st = SurroundingText {
            before: "hello ".to_string(),
            after: "world".to_string(),
            full_value: Some("hello world".to_string()),
        };
        assert_eq!(st.before, "hello ");
        assert_eq!(st.after, "world");
        assert!(st.full_value.is_some());
    }

    #[test]
    fn test_context_chars_constants() {
        const { assert!(CONTEXT_CHARS_BEFORE > 0) };
        const { assert!(CONTEXT_CHARS_AFTER > 0) };
        const { assert!(CONTEXT_CHARS_BEFORE >= CONTEXT_CHARS_AFTER) };
    }

    #[test]
    fn test_get_surrounding_text_returns_option() {
        let ctx = CursorContext::new();
        // In a test environment without a GUI, this should return None
        let result = ctx.get_surrounding_text();
        // Verify it doesn't panic — result may be Some or None depending on environment
        let _ = result;
    }
}
