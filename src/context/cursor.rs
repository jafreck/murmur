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
// macOS implementation via raw Accessibility FFI
// ---------------------------------------------------------------------------
#[cfg(target_os = "macos")]
mod macos {
    use super::{SurroundingText, CONTEXT_CHARS_AFTER, CONTEXT_CHARS_BEFORE};
    use std::ffi::c_void;
    use std::ptr;

    // ── Core Foundation types ──────────────────────────────────────────

    type CFTypeRef = *const c_void;
    type CFStringRef = *const c_void;
    type CFIndex = isize;
    type AXUIElementRef = CFTypeRef;
    type AXValueRef = CFTypeRef;
    type AXError = i32;

    const K_AX_ERROR_SUCCESS: AXError = 0;
    const K_AX_VALUE_TYPE_CF_RANGE: u32 = 4;
    const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    struct CFRange {
        location: CFIndex,
        length: CFIndex,
    }

    // ── FFI declarations ───────────────────────────────────────────────

    #[link(name = "ApplicationServices", kind = "framework")]
    extern "C" {
        fn AXUIElementCreateSystemWide() -> AXUIElementRef;
        fn AXUIElementCopyAttributeValue(
            element: AXUIElementRef,
            attribute: CFStringRef,
            value: *mut CFTypeRef,
        ) -> AXError;
        fn AXValueGetValue(value: AXValueRef, value_type: u32, value_ptr: *mut c_void) -> bool;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFRelease(cf: CFTypeRef);
        fn CFStringGetLength(theString: CFStringRef) -> CFIndex;
        fn CFStringGetCString(
            theString: CFStringRef,
            buffer: *mut u8,
            buffer_size: CFIndex,
            encoding: u32,
        ) -> bool;
        fn CFStringCreateWithCString(
            alloc: CFTypeRef,
            c_str: *const u8,
            encoding: u32,
        ) -> CFStringRef;
    }

    // ── RAII guard for CFTypeRef ────────────────────────────────────────

    /// Releases a Core Foundation reference on drop.
    struct CfGuard(CFTypeRef);

    impl Drop for CfGuard {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { CFRelease(self.0) };
            }
        }
    }

    // ── Helpers ────────────────────────────────────────────────────────

    unsafe fn cfstring_to_string(cf_str: CFStringRef) -> Option<String> {
        if cf_str.is_null() {
            return None;
        }
        let len = CFStringGetLength(cf_str);
        // UTF-8 can use up to 4 bytes per character, plus a NUL terminator.
        let max_size = len * 4 + 1;
        let mut buffer = vec![0u8; max_size as usize];
        if CFStringGetCString(
            cf_str,
            buffer.as_mut_ptr(),
            max_size,
            K_CF_STRING_ENCODING_UTF8,
        ) {
            let nul_pos = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
            Some(String::from_utf8_lossy(&buffer[..nul_pos]).into_owned())
        } else {
            None
        }
    }

    /// Create a CFStringRef from a NUL-terminated byte literal.
    /// The caller must release the returned reference (use `CfGuard`).
    unsafe fn cfstring_create(s: &[u8]) -> CFStringRef {
        CFStringCreateWithCString(ptr::null(), s.as_ptr(), K_CF_STRING_ENCODING_UTF8)
    }

    /// Copy an accessibility attribute from `element`.
    /// The caller must release the returned reference (use `CfGuard`).
    unsafe fn ax_copy_attr(element: AXUIElementRef, attr: CFStringRef) -> Option<CFTypeRef> {
        let mut value: CFTypeRef = ptr::null();
        let err = AXUIElementCopyAttributeValue(element, attr, &mut value);
        if err == K_AX_ERROR_SUCCESS && !value.is_null() {
            Some(value)
        } else {
            None
        }
    }

    /// Extract a `CFRange` from an `AXValueRef`.
    unsafe fn ax_value_to_range(value: AXValueRef) -> Option<CFRange> {
        let mut range = CFRange {
            location: 0,
            length: 0,
        };
        let ok = AXValueGetValue(
            value,
            K_AX_VALUE_TYPE_CF_RANGE,
            &mut range as *mut CFRange as *mut c_void,
        );
        if ok {
            Some(range)
        } else {
            None
        }
    }

    // ── Public entry point ─────────────────────────────────────────────

    pub(super) fn get_surrounding_text() -> Option<SurroundingText> {
        unsafe {
            // 1. System-wide element
            let system = AXUIElementCreateSystemWide();
            if system.is_null() {
                log::debug!("cursor context: failed to create system-wide AX element");
                return None;
            }
            let _sys_guard = CfGuard(system);

            // Create attribute name strings at runtime (avoids linking HIServices sub-framework)
            let attr_focused = cfstring_create(b"AXFocusedUIElement\0");
            let _g1 = CfGuard(attr_focused);
            let attr_value = cfstring_create(b"AXValue\0");
            let _g2 = CfGuard(attr_value);
            let attr_range = cfstring_create(b"AXSelectedTextRange\0");
            let _g3 = CfGuard(attr_range);

            // 2. Focused UI element
            let focused = match ax_copy_attr(system, attr_focused) {
                Some(f) => f,
                None => {
                    log::debug!("cursor context: no focused UI element");
                    return None;
                }
            };
            let _focused_guard = CfGuard(focused);

            // 3. Text value
            let value_ref = match ax_copy_attr(focused, attr_value) {
                Some(v) => v,
                None => {
                    log::debug!("cursor context: focused element has no value attribute");
                    return None;
                }
            };
            let _value_guard = CfGuard(value_ref);

            let full_text = match cfstring_to_string(value_ref) {
                Some(t) => t,
                None => {
                    log::debug!("cursor context: could not convert value to string");
                    return None;
                }
            };

            // 4. Selected text range (cursor position)
            let range_ref = match ax_copy_attr(focused, attr_range) {
                Some(r) => r,
                None => {
                    log::debug!("cursor context: no selected text range attribute");
                    return None;
                }
            };
            let _range_guard = CfGuard(range_ref);

            let range = match ax_value_to_range(range_ref) {
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
