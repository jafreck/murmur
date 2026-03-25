# Wiring Up the System Tray GUI — Implementation Guide

This document describes how to implement the system tray GUI for open-bark
using the `tray-icon` crate (which re-exports `muda` for menus). The tray
provides visual feedback for app state and a context menu for configuration,
replacing the current headless CLI daemon mode.

## Architecture Overview

The current `app.rs` uses a channel-based event loop:

```
hotkey thread ──(KeyDown/KeyUp)──► mpsc channel ──► main loop
                                                    ├── AudioRecorder
                                                    └── Transcriber (Arc, spawned)
```

The tray integration extends this to a unified event loop that handles
**three** event sources: hotkey events, tray icon events, and menu events.

```
hotkey thread ──(KeyDown/KeyUp)──┐
                                 ▼
                          AppMessage channel ──► main event loop
                                 ▲                ├── AudioRecorder
tray menu events ────────────────┘                ├── Transcriber
tray icon events (click) ────────┘                └── TrayController
```

## Key Constraint

`tray-icon` requires the tray to be **created and driven on the main
thread**. On macOS this is mandatory (Cocoa requirement); on Windows and
Linux it's strongly recommended. This aligns with our existing architecture
since the main loop already runs on the main thread.

Both `TrayIconEvent::receiver()` and `MenuEvent::receiver()` return
channel receivers that can be polled with `try_recv()` in the same loop.

---

## Step 1: Remove the `muda` dependency from Cargo.toml

`tray-icon` re-exports all of `muda` under `tray_icon::menu`. The separate
`muda = "0.15"` line in `Cargo.toml` is unnecessary and can cause version
conflicts. Remove it:

```toml
# REMOVE this line:
# muda = "0.15"

# KEEP this:
tray-icon = "0.19"
```

---

## Step 2: Create icon assets

The tray needs icons for each state. Create simple 32×32 RGBA PNG files:

```
assets/icons/
├── idle.png           # Waveform outline (normal state)
├── recording.png      # Filled/active waveform (recording)
├── transcribing.png   # Dots or spinner (processing)
├── downloading.png    # Download arrow or progress
└── error.png          # Warning/error indicator
```

For a minimal first pass, generate a single solid-color icon procedurally
(see Step 3). Replace with designed assets later.

---

## Step 3: Implement `tray.rs`

Replace the current stub with a full implementation:

```rust
use anyhow::Result;
use tray_icon::menu::{Menu, MenuEvent, MenuItem, MenuId, PredefinedMenuItem};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder, TrayIconEvent};

/// App states the tray can display.
#[derive(Debug, Clone, PartialEq)]
pub enum TrayState {
    Idle,
    Recording,
    Transcribing,
    Downloading,
    Error,
}

/// Actions the tray menu can trigger.
#[derive(Debug, Clone)]
pub enum TrayAction {
    CopyLastDictation,
    Quit,
}

/// Manages the system tray icon and context menu.
pub struct TrayController {
    tray: TrayIcon,
    pub state: TrayState,
    copy_last_id: MenuId,
    quit_id: MenuId,
    idle_icon: Icon,
    recording_icon: Icon,
    transcribing_icon: Icon,
}

impl TrayController {
    /// Create the tray icon and menu. Must be called on the main thread.
    pub fn new() -> Result<Self> {
        let status_item = MenuItem::new("open-bark: Idle", false, None);
        let separator = PredefinedMenuItem::separator();
        let copy_last = MenuItem::new("Copy Last Dictation", true, None);
        let quit = MenuItem::new("Quit", true, None);

        let copy_last_id = copy_last.id().clone();
        let quit_id = quit.id().clone();

        let menu = Menu::new();
        menu.append(&status_item)?;
        menu.append(&separator)?;
        menu.append(&copy_last)?;
        menu.append(&quit)?;

        let idle_icon = make_solid_icon(100, 150, 255, 200)?;       // blue
        let recording_icon = make_solid_icon(255, 60, 60, 230)?;    // red
        let transcribing_icon = make_solid_icon(255, 200, 0, 220)?; // yellow

        let tray = TrayIconBuilder::new()
            .with_icon(idle_icon.clone())
            .with_tooltip("open-bark — Idle")
            .with_menu(Box::new(menu))
            .with_menu_on_left_click(true)
            .build()?;

        Ok(Self {
            tray,
            state: TrayState::Idle,
            copy_last_id,
            quit_id,
            idle_icon,
            recording_icon,
            transcribing_icon,
        })
    }

    /// Update the tray icon and tooltip to reflect the current state.
    pub fn set_state(&mut self, state: TrayState) {
        let (icon, tooltip) = match &state {
            TrayState::Idle => (&self.idle_icon, "open-bark — Idle"),
            TrayState::Recording => (
                &self.recording_icon, "open-bark — Recording..."
            ),
            TrayState::Transcribing => (
                &self.transcribing_icon, "open-bark — Transcribing..."
            ),
            TrayState::Downloading => (
                &self.transcribing_icon, "open-bark — Downloading model..."
            ),
            TrayState::Error => (&self.recording_icon, "open-bark — Error"),
        };

        let _ = self.tray.set_icon(Some(icon.clone()));
        let _ = self.tray.set_tooltip(Some(tooltip));
        self.state = state;
    }

    /// Check if a menu event corresponds to a known action.
    pub fn match_menu_event(&self, event: &MenuEvent) -> Option<TrayAction> {
        if event.id() == &self.quit_id {
            Some(TrayAction::Quit)
        } else if event.id() == &self.copy_last_id {
            Some(TrayAction::CopyLastDictation)
        } else {
            None
        }
    }
}

/// Generate a simple solid-color 32×32 RGBA icon.
fn make_solid_icon(r: u8, g: u8, b: u8, a: u8) -> Result<Icon> {
    let size = 32u32;
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);
    for _ in 0..(size * size) {
        rgba.extend_from_slice(&[r, g, b, a]);
    }
    Icon::from_rgba(rgba, size, size)
        .map_err(|e| anyhow::anyhow!("Icon error: {e}"))
}
```

### Loading real PNG icons later

Once you have icon PNGs in `assets/icons/`, add `png = "0.17"` to
`Cargo.toml` and load them at compile time:

```rust
fn load_icon(png_bytes: &[u8]) -> Result<Icon> {
    let decoder = png::Decoder::new(std::io::Cursor::new(png_bytes));
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    buf.truncate(info.buffer_size());
    Icon::from_rgba(buf, info.width, info.height)
        .map_err(|e| anyhow::anyhow!("Icon error: {e}"))
}

// Usage:
let idle_icon = load_icon(include_bytes!("../assets/icons/idle.png"))?;
```

---

## Step 4: Extend `AppMessage` in `app.rs`

Add tray-originated messages and transcription results to the channel:

```rust
enum AppMessage {
    KeyDown,
    KeyUp,
    TrayQuit,
    TrayCopyLast,
    TranscriptionDone(String),
    TranscriptionError(String),
}
```

The `TranscriptionDone` / `TranscriptionError` variants let background
transcription threads report results back to the main loop, which can
then update the tray state. Currently `handle_stop` spawns a thread that
calls `TextInserter::insert` directly and never reports back — these
variants close that gap.

---

## Step 5: Rewrite the main event loop in `app.rs`

The main loop must now poll **three** receivers in a non-blocking fashion:
the `mpsc` channel, `TrayIconEvent::receiver()`, and
`MenuEvent::receiver()`.

```rust
use crate::tray::{TrayController, TrayState, TrayAction};
use tray_icon::TrayIconEvent;
use tray_icon::menu::MenuEvent;
use std::time::Duration;

pub fn run() -> Result<()> {
    let config = Config::load();

    // ... model download, transcriber init, hotkey parsing (unchanged) ...

    // Create tray on the main thread
    let mut tray = TrayController::new()?;

    // Channel for all app events
    let (tx, rx) = mpsc::channel::<AppMessage>();

    // Clone senders for different producers
    let tx_down = tx.clone();
    let tx_up = tx.clone();

    // Hotkey listener (background thread)
    let hotkey_key = parsed.key;
    std::thread::spawn(move || {
        if let Err(e) = HotkeyManager::start(
            hotkey_key,
            move || { let _ = tx_down.send(AppMessage::KeyDown); },
            move || { let _ = tx_up.send(AppMessage::KeyUp); },
        ) {
            error!("Hotkey listener failed: {e}");
        }
    });

    let mut recorder = AudioRecorder::new();
    let mut is_pressed = false;
    let mut last_transcription: Option<String> = None;

    println!("Ready.");

    // Unified event loop
    loop {
        // 1. Drain our app message channel (non-blocking)
        while let Ok(msg) = rx.try_recv() {
            match msg {
                AppMessage::KeyDown => {
                    if !push_to_talk {
                        if is_pressed {
                            start_transcribing(
                                &mut recorder, &mut is_pressed,
                                &transcriber, spoken_punctuation,
                                max_recordings, tx.clone(),
                            );
                            tray.set_state(TrayState::Transcribing);
                        } else {
                            handle_start(
                                &mut recorder, &mut is_pressed, max_recordings
                            );
                            tray.set_state(TrayState::Recording);
                        }
                    } else if !is_pressed {
                        handle_start(
                            &mut recorder, &mut is_pressed, max_recordings
                        );
                        tray.set_state(TrayState::Recording);
                    }
                }
                AppMessage::KeyUp => {
                    if push_to_talk && is_pressed {
                        start_transcribing(
                            &mut recorder, &mut is_pressed,
                            &transcriber, spoken_punctuation,
                            max_recordings, tx.clone(),
                        );
                        tray.set_state(TrayState::Transcribing);
                    }
                }
                AppMessage::TranscriptionDone(text) => {
                    if !text.is_empty() {
                        info!("Transcription: {text}");
                        if let Err(e) = TextInserter::insert(&text) {
                            error!("Insert failed: {e}");
                        }
                        last_transcription = Some(text);
                    }
                    tray.set_state(TrayState::Idle);
                }
                AppMessage::TranscriptionError(e) => {
                    error!("Transcription: {e}");
                    tray.set_state(TrayState::Error);
                }
                AppMessage::TrayCopyLast => {
                    if let Some(ref text) = last_transcription {
                        if let Ok(mut cb) = arboard::Clipboard::new() {
                            let _ = cb.set_text(text.clone());
                            info!("Copied last dictation to clipboard");
                        }
                    }
                }
                AppMessage::TrayQuit => {
                    info!("Quit requested via tray");
                    return Ok(());
                }
            }
        }

        // 2. Drain tray menu events (non-blocking)
        while let Ok(event) = MenuEvent::receiver().try_recv() {
            if let Some(action) = tray.match_menu_event(&event) {
                match action {
                    TrayAction::Quit => {
                        let _ = tx.send(AppMessage::TrayQuit);
                    }
                    TrayAction::CopyLastDictation => {
                        let _ = tx.send(AppMessage::TrayCopyLast);
                    }
                }
            }
        }

        // 3. Drain tray icon events (optional — handle clicks)
        while let Ok(_event) = TrayIconEvent::receiver().try_recv() {
            // Future: left-click to toggle recording, etc.
        }

        // Sleep to avoid busy-looping (~60 iterations/sec)
        std::thread::sleep(Duration::from_millis(16));
    }
}
```

---

## Step 6: Update `handle_stop` → `start_transcribing`

Rename `handle_stop` to `start_transcribing` and add a `tx` parameter so
results flow back through the channel:

```rust
fn start_transcribing(
    recorder: &mut AudioRecorder,
    is_pressed: &mut bool,
    transcriber: &Arc<Transcriber>,
    spoken_punctuation: bool,
    max_recordings: u32,
    tx: mpsc::Sender<AppMessage>,
) {
    if !*is_pressed { return; }
    *is_pressed = false;

    let Some(audio_path) = recorder.stop() else { return; };

    info!("Transcribing...");

    let transcriber = Arc::clone(transcriber);
    std::thread::spawn(move || {
        let result = transcriber.transcribe(&audio_path);

        if max_recordings == 0 {
            let _ = std::fs::remove_file(&audio_path);
        } else {
            RecordingStore::prune(max_recordings);
        }

        match result {
            Ok(raw) => {
                let text = if spoken_punctuation {
                    postprocess::process(&raw)
                } else {
                    raw
                };
                let _ = tx.send(AppMessage::TranscriptionDone(text));
            }
            Err(e) => {
                let _ = tx.send(AppMessage::TranscriptionError(
                    e.to_string()
                ));
            }
        }
    });
}
```

---

## Step 7: Animated recording icon (optional enhancement)

The macOS open-wispr has a bouncing waveform animation during recording.
To replicate with `tray-icon`, swap icons on each loop iteration:

```rust
// In the main event loop, when state is Recording:
if tray.state == TrayState::Recording {
    animation_frame = (animation_frame + 1) % frames.len();
    let _ = tray.tray.set_icon(Some(frames[animation_frame].clone()));
}
```

Generate frames as slightly different RGBA icons (e.g., pulsing circle
or alternating bar heights). The 16ms sleep gives ~60fps — more than
enough for a subtle pulse.

---

## Step 8: Dynamic menu rebuilding (optional enhancement)

Update the context menu to reflect current state and recent recordings:

```rust
impl TrayController {
    pub fn rebuild_menu(
        &mut self,
        state: &TrayState,
        last_transcription: &Option<String>,
    ) {
        let status_text = match state {
            TrayState::Idle => "open-bark: Ready",
            TrayState::Recording => "open-bark: Recording...",
            TrayState::Transcribing => "open-bark: Transcribing...",
            TrayState::Downloading => "open-bark: Downloading...",
            TrayState::Error => "open-bark: Error",
        };

        let menu = Menu::new();
        let _ = menu.append(&MenuItem::new(status_text, false, None));
        let _ = menu.append(&PredefinedMenuItem::separator());

        let copy = MenuItem::new(
            "Copy Last Dictation",
            last_transcription.is_some(),
            None,
        );
        self.copy_last_id = copy.id().clone();
        let _ = menu.append(&copy);

        let _ = menu.append(&PredefinedMenuItem::separator());
        let quit = MenuItem::new("Quit", true, None);
        self.quit_id = quit.id().clone();
        let _ = menu.append(&quit);

        self.tray.set_menu(Some(Box::new(menu)));
    }
}
```

Call `rebuild_menu()` after each `set_state()` call. Note that rebuilding
creates new `MenuId` values, so the stored `copy_last_id` and `quit_id`
must be updated (as shown above).

---

## Summary of changes

| File | Action |
|---|---|
| `Cargo.toml` | Remove `muda`. Optionally add `png`. |
| `src/tray.rs` | Full rewrite with TrayController |
| `src/app.rs` | Extend AppMessage; rewrite loop to poll 3 receivers; rename handle_stop → start_transcribing with tx param |
| `src/main.rs` | No changes needed |
| `assets/icons/*.png` | Add icon assets (or use procedural icons initially) |

## Platform-specific notes

| Platform | Notes |
|---|---|
| **macOS** | Use `with_icon_as_template(true)` for dark/light mode adaptation. Tray must be on main thread. Icon appears in top menu bar. |
| **Windows** | Use `with_menu_on_left_click(true)` (Windows convention). `.ico` preferred but RGBA works. Icon in bottom-right system tray. |
| **Linux** | Requires `libayatana-appindicator3-dev` at build time and runtime. GNOME needs AppIndicator extension. Some Wayland compositors may not show tray. |

## Testing

The tray is inherently visual. Testing strategy:

1. **Manual smoke test:** `cargo run -- start` → verify icon, menu, state
   transitions, Quit.
2. **Unit-testable logic:** `TrayController::match_menu_event` action
   dispatch, `AppMessage` handling logic (extract into pure functions).
3. **CI:** Tray cannot render in headless CI. Gate behind
   `#[cfg(not(ci))]` or compile-time feature flag.
