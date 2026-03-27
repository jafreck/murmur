//! macOS active application detection using objc2-app-kit.

use super::provider::{Context, ContextProvider, DictationMode};
use super::title_analyzer;

/// Detects the currently focused application on the host OS.
pub struct AppDetector;

impl AppDetector {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AppDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(target_os = "macos")]
impl ContextProvider for AppDetector {
    fn name(&self) -> &str {
        "AppDetector"
    }

    fn get_context(&self) -> Context {
        use objc2_app_kit::NSWorkspace;

        let workspace = NSWorkspace::sharedWorkspace();
        let front_app = workspace.frontmostApplication();

        let (app_id, app_name) = match front_app {
            Some(app) => {
                let bundle_id = app.bundleIdentifier().map(|id| id.to_string());
                let name = app.localizedName().map(|n| n.to_string());
                (bundle_id, name)
            }
            None => {
                log::debug!("AppDetector: no frontmost application found");
                (None, None)
            }
        };

        // Get window title from the accessibility API
        let window_title = get_window_title_ax();

        log::debug!(
            "AppDetector: app_id={:?}, app_name={:?}, title={:?}",
            app_id,
            app_name,
            window_title
        );

        // Auto-detect language, mode, and prompt from the window title
        let (file_language, suggested_mode, vocabulary_hints) =
            if let Some(ref title) = window_title {
                let tc = title_analyzer::analyze_title(title);
                let mut hints = Vec::new();
                if let Some(ref lang) = tc.language {
                    hints.push(lang.clone());
                }
                (tc.language, tc.suggested_mode, hints)
            } else if let Some(ref id) = app_id {
                // No window title available — infer mode from app type
                if title_analyzer::is_terminal_app(id) {
                    (None, Some(DictationMode::Code), Vec::new())
                } else {
                    (None, None, Vec::new())
                }
            } else {
                (None, None, Vec::new())
            };

        Context {
            app_id,
            app_name,
            window_title,
            file_language,
            suggested_mode,
            vocabulary_hints,
            ..Default::default()
        }
    }
}

/// Read the focused window's title via the Accessibility API.
#[cfg(target_os = "macos")]
fn get_window_title_ax() -> Option<String> {
    use std::ffi::c_void;
    use std::ptr;

    type CFTypeRef = *const c_void;
    type CFStringRef = *const c_void;
    type CFIndex = isize;
    type AXUIElementRef = CFTypeRef;
    type AXError = i32;

    const K_AX_ERROR_SUCCESS: AXError = 0;
    const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;

    #[link(name = "ApplicationServices", kind = "framework")]
    extern "C" {
        fn AXUIElementCreateApplication(pid: i32) -> AXUIElementRef;
        fn AXUIElementCopyAttributeValue(
            element: AXUIElementRef,
            attribute: CFStringRef,
            value: *mut CFTypeRef,
        ) -> AXError;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFRelease(cf: CFTypeRef);
        fn CFStringGetLength(s: CFStringRef) -> CFIndex;
        fn CFStringGetCString(s: CFStringRef, buf: *mut u8, size: CFIndex, enc: u32) -> bool;
        fn CFStringCreateWithCString(alloc: CFTypeRef, s: *const u8, enc: u32) -> CFStringRef;
    }

    unsafe {
        // Get the PID of the frontmost app
        use objc2_app_kit::NSWorkspace;
        let workspace = NSWorkspace::sharedWorkspace();
        let front_app = workspace.frontmostApplication()?;
        let pid = front_app.processIdentifier();

        // Create an AXUIElement for the app
        let app_element = AXUIElementCreateApplication(pid);
        if app_element.is_null() {
            return None;
        }

        // Get AXFocusedWindow
        let attr_focused_window = CFStringCreateWithCString(
            ptr::null(),
            c"AXFocusedWindow".as_ptr().cast(),
            K_CF_STRING_ENCODING_UTF8,
        );
        if attr_focused_window.is_null() {
            CFRelease(app_element);
            return None;
        }

        let mut window: CFTypeRef = ptr::null();
        let err = AXUIElementCopyAttributeValue(app_element, attr_focused_window, &mut window);
        CFRelease(attr_focused_window);

        if err != K_AX_ERROR_SUCCESS || window.is_null() {
            CFRelease(app_element);
            return None;
        }

        // Get AXTitle from the window
        let attr_title = CFStringCreateWithCString(
            ptr::null(),
            c"AXTitle".as_ptr().cast(),
            K_CF_STRING_ENCODING_UTF8,
        );
        if attr_title.is_null() {
            CFRelease(window);
            CFRelease(app_element);
            return None;
        }

        let mut title_ref: CFTypeRef = ptr::null();
        let err = AXUIElementCopyAttributeValue(window, attr_title, &mut title_ref);
        CFRelease(attr_title);
        CFRelease(window);
        CFRelease(app_element);

        if err != K_AX_ERROR_SUCCESS || title_ref.is_null() {
            return None;
        }

        // Convert CFString to Rust String
        let len = CFStringGetLength(title_ref);
        let max_size = len * 4 + 1;
        let mut buffer = vec![0u8; max_size as usize];
        let result = if CFStringGetCString(
            title_ref,
            buffer.as_mut_ptr(),
            max_size,
            K_CF_STRING_ENCODING_UTF8,
        ) {
            let nul_pos = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
            Some(String::from_utf8_lossy(&buffer[..nul_pos]).into_owned())
        } else {
            None
        };

        CFRelease(title_ref);
        result
    }
}

#[cfg(not(target_os = "macos"))]
impl ContextProvider for AppDetector {
    fn name(&self) -> &str {
        "AppDetector"
    }

    fn get_context(&self) -> Context {
        log::debug!("AppDetector: not on macOS, returning default context");
        Context::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_detector_name() {
        let detector = AppDetector::new();
        assert_eq!(detector.name(), "AppDetector");
    }

    #[test]
    fn app_detector_default() {
        let detector = AppDetector;
        assert_eq!(detector.name(), "AppDetector");
    }

    #[test]
    fn app_detector_returns_context() {
        let detector = AppDetector::new();
        let ctx = detector.get_context();
        // On macOS CI/test environments, we may or may not get a frontmost app.
        // On non-macOS, we get the default context.
        // Either way, the call should not panic.
        assert!(ctx.window_title.is_none() || ctx.window_title.is_some());
    }
}
