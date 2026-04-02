use anyhow::{Context, Result};
use arboard::{Clipboard, ImageData};
use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use std::thread;
use std::time::Duration;

/// Saved clipboard content for restore after paste.
enum SavedClipboard {
    Text(String),
    Image(ImageData<'static>),
    Empty,
}

pub struct TextInserter;

/// The modifier key used for paste: Cmd on macOS, Ctrl elsewhere.
pub fn paste_modifier() -> Key {
    #[cfg(target_os = "macos")]
    {
        Key::Meta
    }
    #[cfg(not(target_os = "macos"))]
    {
        Key::Control
    }
}

impl TextInserter {
    pub fn insert(text: &str) -> Result<()> {
        let mut clipboard = Clipboard::new().context("Failed to access clipboard")?;

        // Save current clipboard contents (try text first, then image)
        let saved = if let Ok(text) = clipboard.get_text() {
            SavedClipboard::Text(text)
        } else if let Ok(img) = clipboard.get_image() {
            SavedClipboard::Image(img.to_owned_img())
        } else {
            SavedClipboard::Empty
        };

        // Set transcription text
        clipboard
            .set_text(text.to_string())
            .context("Failed to set clipboard text")?;

        // Small delay for clipboard to settle
        thread::sleep(Duration::from_millis(50));

        // Simulate paste
        Self::simulate_paste()?;

        // Restore the previous clipboard contents in a background thread
        // so the main event loop is not blocked for 400ms.
        thread::spawn(move || {
            // Wait for the target application to consume the paste before
            // restoring. Slow apps (Electron, remote desktop) may need
            // longer than this, but there is no reliable cross-platform
            // way to detect paste completion.
            thread::sleep(Duration::from_millis(400));
            match saved {
                SavedClipboard::Text(prev) => {
                    if let Err(e) = Clipboard::new().and_then(|mut cb| cb.set_text(prev)) {
                        log::debug!("Failed to restore clipboard text: {e}");
                    }
                }
                SavedClipboard::Image(img) => {
                    if let Err(e) = Clipboard::new().and_then(|mut cb| cb.set_image(img)) {
                        log::debug!("Failed to restore clipboard image: {e}");
                    }
                }
                SavedClipboard::Empty => {}
            }
        });

        Ok(())
    }

    /// Replace the last `backspace_count` characters with `text` at the
    /// current cursor position.
    ///
    /// On macOS this uses the Accessibility API (`AXUIElement`) to
    /// atomically select and replace text in the focused field — no
    /// synthetic key events, no focus changes, works over fullscreen apps.
    /// Falls back to backspace + paste on other platforms or if AX fails.
    pub fn replace(backspace_count: usize, text: &str) -> Result<()> {
        if backspace_count == 0 && text.is_empty() {
            return Ok(());
        }

        #[cfg(target_os = "macos")]
        {
            match ax_replace(backspace_count, text) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    log::warn!("AX replace failed ({e}), falling back to key events");
                }
            }
        }

        Self::replace_via_keys(backspace_count, text)
    }

    /// Fallback: backspaces + enigo/clipboard text entry.
    fn replace_via_keys(backspace_count: usize, text: &str) -> Result<()> {
        let mut enigo =
            Enigo::new(&Settings::default()).map_err(|e| anyhow::anyhow!("Enigo init: {e}"))?;

        let mut remaining = backspace_count;
        const BATCH: usize = 8;
        while remaining > 0 {
            let batch = remaining.min(BATCH);
            for _ in 0..batch {
                enigo
                    .key(Key::Backspace, Direction::Click)
                    .map_err(|e| anyhow::anyhow!("Backspace failed: {e}"))?;
            }
            remaining -= batch;
            if remaining > 0 {
                thread::sleep(Duration::from_millis(5));
            }
        }

        if !text.is_empty() {
            enigo
                .text(text)
                .map_err(|e| anyhow::anyhow!("Text input failed: {e}"))?;
        }

        Ok(())
    }

    fn simulate_paste() -> Result<()> {
        let mut enigo =
            Enigo::new(&Settings::default()).map_err(|e| anyhow::anyhow!("Enigo init: {e}"))?;

        let modifier = paste_modifier();

        enigo
            .key(modifier, Direction::Press)
            .map_err(|e| anyhow::anyhow!("Key press failed: {e}"))?;
        enigo
            .key(Key::Unicode('v'), Direction::Click)
            .map_err(|e| anyhow::anyhow!("Key click failed: {e}"))?;
        enigo
            .key(modifier, Direction::Release)
            .map_err(|e| anyhow::anyhow!("Key release failed: {e}"))?;

        Ok(())
    }
}

// ── macOS Accessibility API text replacement ────────────────────────────

#[cfg(target_os = "macos")]
fn ax_replace(delete_chars: usize, new_text: &str) -> Result<()> {
    use std::ffi::c_void;
    use std::ptr;

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
    #[derive(Copy, Clone)]
    struct CFRange {
        location: CFIndex,
        length: CFIndex,
    }

    #[link(name = "ApplicationServices", kind = "framework")]
    extern "C" {
        fn AXUIElementCreateSystemWide() -> AXUIElementRef;
        fn AXUIElementCopyAttributeValue(
            element: AXUIElementRef,
            attribute: CFStringRef,
            value: *mut CFTypeRef,
        ) -> AXError;
        fn AXUIElementSetAttributeValue(
            element: AXUIElementRef,
            attribute: CFStringRef,
            value: CFTypeRef,
        ) -> AXError;
        fn AXValueCreate(value_type: u32, value_ptr: *const c_void) -> AXValueRef;
        fn AXValueGetValue(value: AXValueRef, value_type: u32, value_ptr: *mut c_void) -> bool;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFRelease(cf: CFTypeRef);
        fn CFStringCreateWithCString(
            alloc: CFTypeRef,
            c_str: *const u8,
            encoding: u32,
        ) -> CFStringRef;
    }

    struct CfGuard(CFTypeRef);
    impl Drop for CfGuard {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { CFRelease(self.0) };
            }
        }
    }

    unsafe fn cfstr(s: &[u8]) -> CFStringRef {
        CFStringCreateWithCString(ptr::null(), s.as_ptr(), K_CF_STRING_ENCODING_UTF8)
    }

    unsafe fn ax_copy(element: AXUIElementRef, attr: CFStringRef) -> Option<CFTypeRef> {
        let mut value: CFTypeRef = ptr::null();
        let err = AXUIElementCopyAttributeValue(element, attr, &mut value);
        if err == K_AX_ERROR_SUCCESS && !value.is_null() {
            Some(value)
        } else {
            None
        }
    }

    unsafe {
        // Get focused UI element
        let system = AXUIElementCreateSystemWide();
        let _sys_guard = CfGuard(system);

        let focused_attr = cfstr(b"AXFocusedUIElement\0");
        let _fa_guard = CfGuard(focused_attr);

        let focused =
            ax_copy(system, focused_attr).ok_or_else(|| anyhow::anyhow!("No focused element"))?;
        let _focused_guard = CfGuard(focused);

        // Get current selected text range to find cursor position
        let sel_range_attr = cfstr(b"AXSelectedTextRange\0");
        let _sra_guard = CfGuard(sel_range_attr);

        let range_val = ax_copy(focused, sel_range_attr)
            .ok_or_else(|| anyhow::anyhow!("No selected text range"))?;
        let _rv_guard = CfGuard(range_val);

        let mut cur_range = CFRange {
            location: 0,
            length: 0,
        };
        if !AXValueGetValue(
            range_val,
            K_AX_VALUE_TYPE_CF_RANGE,
            &mut cur_range as *mut CFRange as *mut c_void,
        ) {
            anyhow::bail!("Failed to read cursor range");
        }

        // Build a new range: select `delete_chars` characters before cursor
        let new_range = CFRange {
            location: (cur_range.location + cur_range.length)
                .saturating_sub(delete_chars as CFIndex),
            length: delete_chars as CFIndex,
        };

        // Set the selection to cover the text we want to replace
        let new_range_val = AXValueCreate(
            K_AX_VALUE_TYPE_CF_RANGE,
            &new_range as *const CFRange as *const c_void,
        );
        if new_range_val.is_null() {
            anyhow::bail!("Failed to create AXValue for range");
        }
        let _nrv_guard = CfGuard(new_range_val);

        let err = AXUIElementSetAttributeValue(focused, sel_range_attr, new_range_val);
        if err != K_AX_ERROR_SUCCESS {
            anyhow::bail!("Failed to set selected text range (AX error {err})");
        }

        // Replace selected text with new text
        let sel_text_attr = cfstr(b"AXSelectedText\0");
        let _sta_guard = CfGuard(sel_text_attr);

        let replacement = {
            let mut bytes = new_text.as_bytes().to_vec();
            bytes.push(0); // NUL-terminate
            cfstr(&bytes)
        };
        let _rep_guard = CfGuard(replacement);

        let err = AXUIElementSetAttributeValue(focused, sel_text_attr, replacement);
        if err != K_AX_ERROR_SUCCESS {
            anyhow::bail!("Failed to set selected text (AX error {err})");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paste_modifier_returns_expected() {
        let m = paste_modifier();
        #[cfg(target_os = "macos")]
        assert_eq!(m, Key::Meta);
        #[cfg(not(target_os = "macos"))]
        assert_eq!(m, Key::Control);
    }
}
