//! Transparent overlay window for live dictation display.
//!
//! Runs as a subprocess (`murmur overlay`). Reads JSON-line commands from
//! stdin to show/hide the window and update displayed text.

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Commands sent from the main process to the overlay via stdin.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum OverlayCommand {
    /// Show the overlay window.
    Show,
    /// Hide the overlay window.
    Hide,
    /// Replace the displayed text.
    Text { content: String },
    /// Clear the displayed text.
    Clear,
    /// Signal that the session is complete (overlay can save and hide).
    Done,
    /// Quit the overlay process.
    Quit,
}

/// Run the overlay eframe application. Called from `murmur overlay`.
pub fn run_overlay() -> Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_transparent(true)
            .with_always_on_top()
            .with_decorations(false)
            .with_inner_size([500.0, 200.0])
            .with_min_inner_size([200.0, 80.0]),
        ..Default::default()
    };

    eframe::run_native(
        "murmur overlay",
        native_options,
        Box::new(|_cc| Ok(Box::new(OverlayApp::new()))),
    )
    .map_err(|e| anyhow::anyhow!("Overlay error: {e}"))?;

    Ok(())
}

struct OverlayApp {
    text: String,
    visible: bool,
    cmd_rx: std::sync::mpsc::Receiver<OverlayCommand>,
    should_quit: bool,
    /// Current opacity for fade animation (0.0 = invisible, 1.0 = fully visible)
    opacity: f32,
    fade_state: FadeState,
    #[cfg(target_os = "macos")]
    window_level_set: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum FadeState {
    Hidden,
    FadingIn,
    Visible,
    FadingOut,
}

/// Fade animation speed (opacity change per second)
const FADE_IN_SPEED: f32 = 6.0;
const FADE_OUT_SPEED: f32 = 2.0;

impl OverlayApp {
    fn new() -> Self {
        let (tx, rx) = std::sync::mpsc::channel();

        // Read stdin on a background thread, parsing JSON-line commands.
        std::thread::spawn(move || {
            use std::io::BufRead;
            let stdin = std::io::stdin();
            for line in stdin.lock().lines() {
                let Ok(line) = line else { break };
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }
                match serde_json::from_str::<OverlayCommand>(&line) {
                    Ok(cmd) => {
                        if tx.send(cmd).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        log::warn!("Overlay: invalid command: {e}: {line}");
                    }
                }
            }
            // stdin closed — signal quit
            let _ = tx.send(OverlayCommand::Quit);
        });

        Self {
            text: String::new(),
            visible: false,
            cmd_rx: rx,
            should_quit: false,
            opacity: 0.0,
            fade_state: FadeState::Hidden,
            #[cfg(target_os = "macos")]
            window_level_set: false,
        }
    }

    fn process_commands(&mut self, ctx: &egui::Context) {
        while let Ok(cmd) = self.cmd_rx.try_recv() {
            match cmd {
                OverlayCommand::Show => {
                    self.visible = true;
                    self.fade_state = FadeState::FadingIn;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
                    ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
                }
                OverlayCommand::Hide => {
                    self.visible = false;
                    self.opacity = 0.0;
                    self.fade_state = FadeState::Hidden;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Visible(false));
                }
                OverlayCommand::Text { content } => {
                    self.text = content;
                }
                OverlayCommand::Clear => {
                    self.text.clear();
                }
                OverlayCommand::Done => {
                    self.fade_state = FadeState::FadingOut;
                }
                OverlayCommand::Quit => {
                    self.should_quit = true;
                }
            }
        }
    }

    fn update_fade(&mut self, ctx: &egui::Context, dt: f32) {
        match self.fade_state {
            FadeState::FadingIn => {
                self.opacity = (self.opacity + FADE_IN_SPEED * dt).min(1.0);
                if self.opacity >= 1.0 {
                    self.fade_state = FadeState::Visible;
                }
                ctx.request_repaint();
            }
            FadeState::FadingOut => {
                self.opacity = (self.opacity - FADE_OUT_SPEED * dt).max(0.0);
                if self.opacity <= 0.0 {
                    self.fade_state = FadeState::Hidden;
                    self.visible = false;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Visible(false));
                }
                ctx.request_repaint();
            }
            FadeState::Visible | FadeState::Hidden => {}
        }
    }
}

impl eframe::App for OverlayApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0, 0.0, 0.0, 0.0]
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.process_commands(ctx);

        if self.should_quit {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

        // On macOS, elevate window level to appear over fullscreen apps
        #[cfg(target_os = "macos")]
        if !self.window_level_set {
            set_macos_overlay_window_level();
            self.window_level_set = true;
        }

        let dt = ctx.input(|i| i.stable_dt);
        self.update_fade(ctx, dt);

        // Request periodic repaints to check for new commands
        ctx.request_repaint_after(std::time::Duration::from_millis(50));

        if !self.visible && self.fade_state == FadeState::Hidden {
            return;
        }

        let alpha = self.opacity;

        // Semi-transparent dark panel with fade
        let panel_frame = egui::Frame::new()
            .fill(egui::Color32::from_rgba_unmultiplied(
                20,
                20,
                20,
                (180.0 * alpha) as u8,
            ))
            .inner_margin(egui::Margin::same(16))
            .corner_radius(12.0);

        egui::CentralPanel::default()
            .frame(panel_frame)
            .show(ctx, |ui| {
                ui.visuals_mut().override_text_color = Some(egui::Color32::from_rgba_unmultiplied(
                    255,
                    255,
                    255,
                    (230.0 * alpha) as u8,
                ));

                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .show(ui, |ui| {
                        if self.text.is_empty() {
                            ui.label(
                                egui::RichText::new("Listening…")
                                    .size(18.0)
                                    .italics()
                                    .color(egui::Color32::from_rgba_unmultiplied(
                                        180,
                                        180,
                                        180,
                                        (160.0 * alpha) as u8,
                                    )),
                            );
                        } else {
                            ui.label(egui::RichText::new(&self.text).size(18.0));
                        }
                    });
            });
    }
}

/// Set the macOS window level high enough to appear over fullscreen apps.
#[cfg(target_os = "macos")]
fn set_macos_overlay_window_level() {
    use objc2_app_kit::NSApplication;
    use objc2_foundation::MainThreadMarker;

    // NSScreenSaverWindowLevel = 1000 — appears over fullscreen apps
    const SCREEN_SAVER_WINDOW_LEVEL: isize = 1000;

    let Some(mtm) = MainThreadMarker::new() else {
        log::warn!("Overlay: not on main thread, cannot set window level");
        return;
    };
    let app = NSApplication::sharedApplication(mtm);
    let windows = app.windows();

    for window in windows {
        let title = window.title();
        if title.to_string().contains("murmur overlay") {
            window.setLevel(SCREEN_SAVER_WINDOW_LEVEL);
            // NSWindowCollectionBehaviorCanJoinAllSpaces (1 << 0)
            // | NSWindowCollectionBehaviorFullScreenAuxiliary (1 << 8)
            window.setCollectionBehavior(
                objc2_app_kit::NSWindowCollectionBehavior::CanJoinAllSpaces
                    | objc2_app_kit::NSWindowCollectionBehavior::FullScreenAuxiliary,
            );
            log::debug!("Overlay: macOS window level set to screen-saver level");
            break;
        }
    }
}

/// Handle for managing the overlay subprocess from the main app.
pub struct OverlayHandle {
    child: std::process::Child,
    stdin: std::io::BufWriter<std::process::ChildStdin>,
}

impl OverlayHandle {
    /// Spawn `murmur overlay` as a subprocess.
    pub fn spawn() -> Result<Self> {
        let exe = std::env::current_exe()?;
        let mut child = std::process::Command::new(exe)
            .arg("overlay")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to capture overlay stdin"))?;

        Ok(Self {
            child,
            stdin: std::io::BufWriter::new(stdin),
        })
    }

    /// Send a command to the overlay process.
    pub fn send(&mut self, cmd: &OverlayCommand) -> Result<()> {
        use std::io::Write;
        serde_json::to_writer(&mut self.stdin, cmd)?;
        writeln!(self.stdin)?;
        self.stdin.flush()?;
        Ok(())
    }

    /// Show the overlay.
    pub fn show(&mut self) -> Result<()> {
        self.send(&OverlayCommand::Show)
    }

    /// Hide the overlay.
    pub fn hide(&mut self) -> Result<()> {
        self.send(&OverlayCommand::Hide)
    }

    /// Update the overlay text.
    pub fn set_text(&mut self, text: &str) -> Result<()> {
        self.send(&OverlayCommand::Text {
            content: text.to_string(),
        })
    }

    /// Clear overlay text.
    pub fn clear(&mut self) -> Result<()> {
        self.send(&OverlayCommand::Clear)
    }

    /// Signal session done.
    pub fn done(&mut self) -> Result<()> {
        self.send(&OverlayCommand::Done)
    }

    /// Gracefully shut down the overlay.
    pub fn quit(&mut self) {
        let _ = self.send(&OverlayCommand::Quit);
        let _ = self.child.wait();
    }

    /// Check if the overlay process is still running.
    pub fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
}

impl Drop for OverlayHandle {
    fn drop(&mut self) {
        let _ = self.send(&OverlayCommand::Quit);
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlay_command_serialize_show() {
        let cmd = OverlayCommand::Show;
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains("\"cmd\":\"show\""));
    }

    #[test]
    fn test_overlay_command_serialize_text() {
        let cmd = OverlayCommand::Text {
            content: "hello world".to_string(),
        };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains("\"cmd\":\"text\""));
        assert!(json.contains("hello world"));
    }

    #[test]
    fn test_overlay_command_deserialize_show() {
        let cmd: OverlayCommand = serde_json::from_str(r#"{"cmd":"show"}"#).unwrap();
        assert!(matches!(cmd, OverlayCommand::Show));
    }

    #[test]
    fn test_overlay_command_deserialize_text() {
        let cmd: OverlayCommand =
            serde_json::from_str(r#"{"cmd":"text","content":"hello"}"#).unwrap();
        match cmd {
            OverlayCommand::Text { content } => assert_eq!(content, "hello"),
            _ => panic!("Expected Text command"),
        }
    }

    #[test]
    fn test_overlay_command_deserialize_clear() {
        let cmd: OverlayCommand = serde_json::from_str(r#"{"cmd":"clear"}"#).unwrap();
        assert!(matches!(cmd, OverlayCommand::Clear));
    }

    #[test]
    fn test_overlay_command_deserialize_done() {
        let cmd: OverlayCommand = serde_json::from_str(r#"{"cmd":"done"}"#).unwrap();
        assert!(matches!(cmd, OverlayCommand::Done));
    }

    #[test]
    fn test_overlay_command_deserialize_quit() {
        let cmd: OverlayCommand = serde_json::from_str(r#"{"cmd":"quit"}"#).unwrap();
        assert!(matches!(cmd, OverlayCommand::Quit));
    }

    #[test]
    fn test_overlay_command_roundtrip() {
        let cmds = vec![
            OverlayCommand::Show,
            OverlayCommand::Hide,
            OverlayCommand::Text {
                content: "test".to_string(),
            },
            OverlayCommand::Clear,
            OverlayCommand::Done,
            OverlayCommand::Quit,
        ];
        for cmd in cmds {
            let json = serde_json::to_string(&cmd).unwrap();
            let parsed: OverlayCommand = serde_json::from_str(&json).unwrap();
            let json2 = serde_json::to_string(&parsed).unwrap();
            assert_eq!(json, json2);
        }
    }
}
