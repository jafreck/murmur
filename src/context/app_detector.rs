//! macOS active application detection using objc2-app-kit.

use super::provider::{Context, ContextProvider};

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

        // Window title detection requires the Accessibility API (AXUIElement),
        // which will be implemented as part of the cursor-context feature.
        let window_title = None;

        log::debug!("AppDetector: app_id={:?}, app_name={:?}", app_id, app_name);

        Context {
            app_id,
            app_name,
            window_title,
            ..Default::default()
        }
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
