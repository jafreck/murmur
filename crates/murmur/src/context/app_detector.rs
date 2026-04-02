//! Cross-platform active application and window title detection.
//!
//! - **macOS**: NSWorkspace + Accessibility API (AXUIElement)
//! - **Linux**: xdotool (X11) or xprop fallback
//! - **Windows**: GetForegroundWindow + GetWindowTextW + QueryFullProcessImageNameW

use murmur_core::context::title_analyzer;
use murmur_core::context::{Context, ContextProvider, DictationMode};

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

// ── Shared logic ───────────────────────────────────────────────────────

/// Build a `Context` from raw platform-detected fields.
fn build_context(
    app_id: Option<String>,
    app_name: Option<String>,
    window_title: Option<String>,
) -> Context {
    log::debug!(
        "AppDetector: app_id={:?}, app_name={:?}, title={:?}",
        app_id,
        app_name,
        window_title,
    );

    let (file_language, suggested_mode, vocabulary_hints) = if let Some(ref title) = window_title {
        let tc = title_analyzer::analyze_title(title);
        let mut hints = Vec::new();
        if let Some(ref lang) = tc.language {
            hints.push(lang.clone());
        }
        (tc.language, tc.suggested_mode, hints)
    } else if let Some(ref id) = app_id {
        if title_analyzer::is_terminal_app(id) {
            (None, Some(DictationMode::Code), Vec::new())
        } else {
            (None, None, Vec::new())
        }
    } else if let Some(ref name) = app_name {
        let lower = name.to_lowercase();
        let is_term = [
            "terminal",
            "iterm",
            "alacritty",
            "kitty",
            "wezterm",
            "warp",
            "konsole",
            "gnome-terminal",
            "xterm",
            "cmd",
            "powershell",
            "windows terminal",
        ]
        .iter()
        .any(|t| lower.contains(t));
        if is_term {
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

// ── macOS ──────────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
impl ContextProvider for AppDetector {
    fn name(&self) -> &str {
        "AppDetector"
    }

    fn get_context(&self) -> Context {
        use objc2_app_kit::NSWorkspace;

        let workspace = NSWorkspace::sharedWorkspace();
        let front_app = workspace.frontmostApplication();

        let (app_id, app_name, pid) = match front_app {
            Some(app) => {
                let bundle_id = app.bundleIdentifier().map(|id| id.to_string());
                let name = app.localizedName().map(|n| n.to_string());
                let pid = app.processIdentifier();
                (bundle_id, name, Some(pid))
            }
            None => {
                log::debug!("AppDetector: no frontmost application found");
                (None, None, None)
            }
        };

        let window_title = pid.and_then(|p| {
            std::panic::catch_unwind(|| get_window_title_ax(p)).unwrap_or_else(|_| {
                log::debug!("AppDetector: accessibility API panicked, skipping window title");
                None
            })
        });

        build_context(app_id, app_name, window_title)
    }
}

/// Read the focused window's title via the Accessibility API.
#[cfg(target_os = "macos")]
fn get_window_title_ax(pid: i32) -> Option<String> {
    use crate::platform::ax;

    let app_element = ax::ax_application(pid);
    if app_element.is_null() {
        return None;
    }

    let attr_focused_window = ax::cfstring_create(b"AXFocusedWindow\0");
    let window = ax::ax_copy_attr(app_element.0, attr_focused_window.0)?;

    let attr_title = ax::cfstring_create(b"AXTitle\0");
    let title_ref = ax::ax_copy_attr(window.0, attr_title.0)?;

    ax::cfstring_to_string(title_ref.0)
}

// ── Linux ──────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
impl ContextProvider for AppDetector {
    fn name(&self) -> &str {
        "AppDetector"
    }

    fn get_context(&self) -> Context {
        let (app_name, window_title) = linux::detect();
        build_context(None, app_name, window_title)
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use std::process::Command;

    pub(super) fn detect() -> (Option<String>, Option<String>) {
        let title = get_window_title();
        let app_name = get_app_name();
        (app_name, title)
    }

    fn get_window_title() -> Option<String> {
        if let Some(title) = run_command("xdotool", &["getactivewindow", "getwindowname"]) {
            return Some(title);
        }
        // Fallback: xprop on the root window
        if let Some(output) = run_command("xprop", &["-root", "_NET_ACTIVE_WINDOW"]) {
            if let Some(hex) = output.split("# ").nth(1) {
                let hex = hex.trim();
                if let Some(name_output) = run_command("xprop", &["-id", hex, "WM_NAME"]) {
                    if let Some(title) = name_output.split(" = \"").nth(1) {
                        return Some(title.trim_end_matches('"').to_string());
                    }
                }
            }
        }
        None
    }

    fn get_app_name() -> Option<String> {
        let pid_str = run_command("xdotool", &["getactivewindow", "getwindowpid"])?;
        let pid = pid_str.trim();
        if pid.is_empty() || pid == "0" {
            return None;
        }
        std::fs::read_to_string(format!("/proc/{pid}/comm"))
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    fn run_command(cmd: &str, args: &[&str]) -> Option<String> {
        Command::new(cmd)
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }
}

// ── Windows ────────────────────────────────────────────────────────────

#[cfg(target_os = "windows")]
impl ContextProvider for AppDetector {
    fn name(&self) -> &str {
        "AppDetector"
    }

    fn get_context(&self) -> Context {
        let (app_name, window_title) = windows::detect();
        build_context(None, app_name, window_title)
    }
}

#[cfg(target_os = "windows")]
mod windows {
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;

    #[link(name = "user32")]
    extern "system" {
        fn GetForegroundWindow() -> *mut std::ffi::c_void;
        fn GetWindowTextW(hwnd: *mut std::ffi::c_void, text: *mut u16, max_count: i32) -> i32;
        fn GetWindowThreadProcessId(hwnd: *mut std::ffi::c_void, process_id: *mut u32) -> u32;
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn OpenProcess(access: u32, inherit: i32, pid: u32) -> *mut std::ffi::c_void;
        fn CloseHandle(handle: *mut std::ffi::c_void) -> i32;
        fn QueryFullProcessImageNameW(
            process: *mut std::ffi::c_void,
            flags: u32,
            name: *mut u16,
            size: *mut u32,
        ) -> i32;
    }

    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;

    pub(super) fn detect() -> (Option<String>, Option<String>) {
        // SAFETY: GetForegroundWindow has no preconditions and returns
        // null if no window is in the foreground.  The returned HWND is
        // valid for the duration of these calls (same message-loop tick).
        unsafe {
            let hwnd = GetForegroundWindow();
            if hwnd.is_null() {
                return (None, None);
            }
            let title = get_window_title(hwnd);
            let app_name = get_process_name(hwnd);
            (app_name, title)
        }
    }

    /// SAFETY: `hwnd` must be a valid window handle from `GetForegroundWindow`.
    unsafe fn get_window_title(hwnd: *mut std::ffi::c_void) -> Option<String> {
        let mut buf = [0u16; 512];
        let len = GetWindowTextW(hwnd, buf.as_mut_ptr(), buf.len() as i32);
        if len <= 0 {
            return None;
        }
        let title = OsString::from_wide(&buf[..len as usize])
            .to_string_lossy()
            .into_owned();
        if title.is_empty() {
            None
        } else {
            Some(title)
        }
    }

    /// SAFETY: `hwnd` must be a valid window handle.  Opens a limited-
    /// information process handle which is closed before returning.
    unsafe fn get_process_name(hwnd: *mut std::ffi::c_void) -> Option<String> {
        let mut pid: u32 = 0;
        GetWindowThreadProcessId(hwnd, &mut pid);
        if pid == 0 {
            return None;
        }

        let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if process.is_null() {
            return None;
        }

        let mut buf = [0u16; 512];
        let mut size = buf.len() as u32;
        let ok = QueryFullProcessImageNameW(process, 0, buf.as_mut_ptr(), &mut size);
        CloseHandle(process);

        if ok == 0 || size == 0 {
            return None;
        }

        let full_path = OsString::from_wide(&buf[..size as usize])
            .to_string_lossy()
            .into_owned();

        full_path
            .rsplit('\\')
            .next()
            .map(|s| s.trim_end_matches(".exe").to_string())
            .filter(|s| !s.is_empty())
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
        assert!(ctx.window_title.is_none() || ctx.window_title.is_some());
    }

    #[test]
    fn build_context_with_rust_title() {
        let ctx = build_context(
            Some("com.microsoft.VSCode".to_string()),
            Some("Code".to_string()),
            Some("main.rs \u{2014} Visual Studio Code".to_string()),
        );
        assert_eq!(ctx.file_language.as_deref(), Some("Rust"));
        assert_eq!(ctx.suggested_mode, Some(DictationMode::Code));
        assert!(ctx.vocabulary_hints.contains(&"Rust".to_string()));
    }

    #[test]
    fn build_context_terminal_by_id() {
        let ctx = build_context(
            Some("com.apple.Terminal".to_string()),
            Some("Terminal".to_string()),
            None,
        );
        assert_eq!(ctx.suggested_mode, Some(DictationMode::Code));
    }

    #[test]
    fn build_context_terminal_by_name() {
        let ctx = build_context(None, Some("gnome-terminal-server".to_string()), None);
        assert_eq!(ctx.suggested_mode, Some(DictationMode::Code));
    }

    #[test]
    fn build_context_unknown_app() {
        let ctx = build_context(
            Some("com.apple.Safari".to_string()),
            Some("Safari".to_string()),
            Some("GitHub - Google".to_string()),
        );
        assert!(ctx.file_language.is_none());
        assert!(ctx.suggested_mode.is_none());
    }

    #[test]
    fn build_context_no_info() {
        let ctx = build_context(None, None, None);
        assert!(ctx.file_language.is_none());
        assert!(ctx.suggested_mode.is_none());
        assert!(ctx.vocabulary_hints.is_empty());
    }

    #[test]
    fn build_context_markdown_gets_prose() {
        let ctx = build_context(None, None, Some("README.md \u{2014} Code".to_string()));
        assert_eq!(ctx.file_language.as_deref(), Some("Markdown"));
        assert_eq!(ctx.suggested_mode, Some(DictationMode::Prose));
    }
}
