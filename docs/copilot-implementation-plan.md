# Murmur Copilot — Implementation Plan

Detailed, phase-by-phase implementation plan for the murmur copilot.
Each phase lists the exact files to create/modify, the public APIs to expose,
the migration steps, and the acceptance criteria that must pass before moving on.

> **Reference**: [copilot-expansion-plan.md](./copilot-expansion-plan.md)

---

## Table of Contents

- [Phase 0 — Workspace Restructuring](#phase-0--workspace-restructuring)
- [Phase 1 — Copilot Skeleton](#phase-1--copilot-skeleton)
- [Phase 2 — System Audio & Stealth](#phase-2--system-audio--stealth)
- [Phase 3 — Local LLM Integration](#phase-3--local-llm-integration)
- [Phase 4 — Polish & Release](#phase-4--polish--release)
- [Dependency Map](#dependency-map)
- [Risk Register](#risk-register)

---

## Phase 0 — Workspace Restructuring

**Goal**: Convert the single-crate repo into a Cargo workspace with `murmur-core`
(shared library) and `murmur` (dictation binary). No new features — purely structural.
All existing tests must continue to pass.

### 0.1 — Create workspace manifest

**File**: `Cargo.toml` (workspace root — replaces current)

Convert the root `Cargo.toml` into a workspace manifest:

```toml
[workspace]
resolver = "2"
members = [
    "crates/murmur-core",
    "crates/murmur",
]

[workspace.package]
edition = "2021"
license = "MIT"
repository = "https://github.com/jacobfreck/murmur"

[workspace.dependencies]
# Shared deps go here — each crate references them via { workspace = true }
whisper-rs = "0.16"
cpal = "0.17"
hound = "3.5"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
reqwest = { version = "0.13", features = ["blocking"] }
dirs = "6"
regex = "1"
anyhow = "1"
voice_activity_detector = "0.2"
log = "0.4"
env_logger = "0.11"
nnnoiseless = "0.5"
png = "0.18.1"
tray-icon = "0.21"
rdev = "0.5"
arboard = "3"
enigo = "0.6"
clap = { version = "4", features = ["derive"] }
tempfile = "3"
```

### 0.2 — Extract `murmur-core` crate

**Directory**: `crates/murmur-core/`

Move these modules (no API changes — just path changes):

| Source (current)              | Destination (murmur-core)                    |
|-------------------------------|----------------------------------------------|
| `src/audio/capture.rs`        | `crates/murmur-core/src/audio/capture.rs`    |
| `src/audio/recordings.rs`     | `crates/murmur-core/src/audio/recordings.rs` |
| `src/audio/mod.rs`            | `crates/murmur-core/src/audio/mod.rs`        |
| `src/transcription/model.rs`  | `crates/murmur-core/src/transcription/model.rs` |
| `src/transcription/postprocess.rs` | `crates/murmur-core/src/transcription/postprocess.rs` |
| `src/transcription/streaming.rs` | `crates/murmur-core/src/transcription/streaming.rs` |
| `src/transcription/transcriber.rs` | `crates/murmur-core/src/transcription/transcriber.rs` |
| `src/transcription/vad.rs`    | `crates/murmur-core/src/transcription/vad.rs` |
| `src/transcription/mod.rs`    | `crates/murmur-core/src/transcription/mod.rs` |
| `src/context/provider.rs`     | `crates/murmur-core/src/context/provider.rs` |
| `src/context/title_analyzer.rs` | `crates/murmur-core/src/context/title_analyzer.rs` |
| `src/context/system_state.rs` | `crates/murmur-core/src/context/system_state.rs` |

**Config split**: `src/config.rs` must be split into two files:

1. **`crates/murmur-core/src/config.rs`** — shared types used by both products:
   - `DictationMode` enum (used by `ContextProvider`)
   - `SUPPORTED_MODELS`, `SUPPORTED_LANGUAGES`, and their helper functions
   - `is_english_only_model`, `is_valid_language`, `language_name`
   - Model directory helpers (`model_dir`, `model_path`)
   - A new `CoreConfig` trait or struct with just the fields the core needs:
     `model_size`, `language`, `noise_suppression`, `vocabulary`, `app_contexts`

2. **`crates/murmur/src/config.rs`** — dictation-specific config:
   - `Config` struct (full, with `hotkey`, `mode`, `streaming`, etc.)
   - `InputMode` enum
   - `AppContextConfig` struct
   - `Config::load()`, `Config::save()`, `Config::file_path()`
   - Implements `Deref<Target = CoreConfig>` or provides accessor methods
     so core modules can consume it

**`crates/murmur-core/src/lib.rs`**:
```rust
pub mod audio;
pub mod config;
pub mod context;
pub mod transcription;
```

**`crates/murmur-core/Cargo.toml`**:
```toml
[package]
name = "murmur-core"
version = "0.1.0"
edition.workspace = true
license.workspace = true
description = "Shared transcription engine for murmur products"

[dependencies]
whisper-rs = { workspace = true }
cpal = { workspace = true }
hound = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
reqwest = { workspace = true }
dirs = { workspace = true }
regex = { workspace = true }
anyhow = { workspace = true }
voice_activity_detector = { workspace = true }
log = { workspace = true }
nnnoiseless = { workspace = true }

[features]
default = []
cuda = ["whisper-rs/cuda"]
metal = ["whisper-rs/metal"]
vulkan = ["whisper-rs/vulkan"]

[dev-dependencies]
tempfile = { workspace = true }
```

### 0.3 — Refactor `murmur` binary crate

**Directory**: `crates/murmur/`

Move the remaining modules (unchanged):

| Source (current)              | Destination (murmur binary)            |
|-------------------------------|----------------------------------------|
| `src/main.rs`                 | `crates/murmur/src/main.rs`            |
| `src/app/`                    | `crates/murmur/src/app/`               |
| `src/ui/`                     | `crates/murmur/src/ui/`                |
| `src/input/`                  | `crates/murmur/src/input/`             |
| `src/platform/`               | `crates/murmur/src/platform/`          |
| `src/context/app_detector.rs` | `crates/murmur/src/context/app_detector.rs` |
| `src/context/cursor.rs`       | `crates/murmur/src/context/cursor.rs`  |

**`crates/murmur/src/lib.rs`**:
```rust
pub mod app;
pub mod config;
pub mod context;
pub mod input;
pub mod platform;
pub mod ui;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
```

All `use crate::audio::`, `use crate::transcription::`, and shared `use crate::config::`
imports in the binary crate change to `use murmur_core::audio::`, etc.

**`crates/murmur/Cargo.toml`**:
```toml
[package]
name = "murmur"
version = "0.1.1"
edition.workspace = true
license.workspace = true
description = "Cross-platform, local voice dictation"

[dependencies]
murmur-core = { path = "../murmur-core" }
tray-icon = { workspace = true }
rdev = { workspace = true }
arboard = { workspace = true }
enigo = { workspace = true }
clap = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
dirs = { workspace = true }
anyhow = { workspace = true }
log = { workspace = true }
env_logger = { workspace = true }
png = { workspace = true }

[target.'cfg(target_os = "macos")'.dependencies]
objc2-app-kit = { version = "0.3", features = ["NSApplication", "NSRunningApplication", "NSEvent", "NSWorkspace"] }
objc2-foundation = { version = "0.3", features = ["NSDate", "NSRunLoop"] }
objc2 = "0.6"

[features]
default = []
cuda = ["murmur-core/cuda"]
metal = ["murmur-core/metal"]
vulkan = ["murmur-core/vulkan"]

[[bin]]
name = "murmur"
path = "src/main.rs"

[dev-dependencies]
tempfile = { workspace = true }
```

### 0.4 — Migrate tests

| Test file (current)               | Destination                    | Changes needed |
|-----------------------------------|--------------------------------|----------------|
| `tests/audio_pipeline.rs`         | `crates/murmur-core/tests/`   | `use murmur_core::` |
| `tests/postprocess_integration.rs`| `crates/murmur-core/tests/`   | `use murmur_core::` |
| `tests/recording_store.rs`        | `crates/murmur-core/tests/`   | `use murmur_core::` |
| `tests/context_integration.rs`    | `crates/murmur-core/tests/`   | `use murmur_core::` |
| `tests/config_integration.rs`     | `crates/murmur/tests/`        | `use murmur::` |
| `tests/effect_integration.rs`     | `crates/murmur/tests/`        | `use murmur::` |
| `tests/cli_e2e.rs`                | `crates/murmur/tests/`        | `use murmur::` |
| `tests/integration.rs`            | `crates/murmur/tests/`        | `use murmur::` |
| `tests/helpers/`                  | Duplicate into both if shared  | — |

### 0.5 — CI & workflow updates

- **`.github/workflows/ci.yml`**: Change `cargo build/test/clippy` to workspace-level commands.
  The binary name stays `murmur`, so install scripts (`scripts/install.sh`) should still work.
- **`Cargo.lock`**: Regenerate after workspace restructuring.
- **`assets/`**: Move to workspace root (shared) or `crates/murmur/assets/` (dictation-only).

### 0.6 — Acceptance criteria

- [ ] `cargo build --workspace` succeeds
- [ ] `cargo test --workspace` — all existing tests pass
- [ ] `cargo clippy --workspace` — no new warnings
- [ ] `cargo build --release -p murmur` produces a working binary
- [ ] The `murmur` binary behaves identically to pre-restructuring
- [ ] CI pipeline passes on all three platforms (macOS, Linux, Windows)

---

## Phase 1 — Copilot Skeleton

**Goal**: Scaffold `murmur-copilot` with Tauri. Wire murmur-core's streaming
transcription to a basic overlay UI. Microphone-only capture. Prove the pipeline
works end-to-end: start copilot → begin meeting → see live transcript in overlay.

### 1.1 — Initialize Tauri project

```
crates/murmur-copilot/
├── Cargo.toml
├── tauri.conf.json
├── build.rs
├── src/                     ← Rust backend (Tauri plugin)
│   ├── main.rs              ← Tauri entry point
│   ├── lib.rs               ← re-exports
│   ├── meeting.rs           ← session lifecycle
│   └── commands.rs          ← Tauri IPC commands
├── frontend/                ← Web UI (TypeScript/React or Svelte)
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── components/
│       │   ├── Transcript.tsx
│       │   └── StatusBar.tsx
│       └── hooks/
│           └── useTranscription.ts
└── icons/                   ← App icons
```

**`crates/murmur-copilot/Cargo.toml`**:
```toml
[package]
name = "murmur-copilot"
version = "0.1.0"
edition.workspace = true
license.workspace = true
description = "Local meeting copilot powered by murmur"

[dependencies]
murmur-core = { path = "../murmur-core" }
tauri = { version = "2", features = ["tray-icon"] }
tauri-plugin-shell = "2"
serde = { workspace = true }
serde_json = { workspace = true }
anyhow = { workspace = true }
log = { workspace = true }
env_logger = { workspace = true }

[build-dependencies]
tauri-build = { version = "2", features = [] }

[features]
default = ["custom-protocol"]
custom-protocol = ["tauri/custom-protocol"]
cuda = ["murmur-core/cuda"]
metal = ["murmur-core/metal"]
```

### 1.2 — Meeting session model

**File**: `crates/murmur-copilot/src/meeting.rs`

```rust
pub enum MeetingState {
    Idle,
    Recording { started_at: Instant },
    Paused { elapsed: Duration },
}

pub struct MeetingSession {
    pub id: String,           // UUID
    pub state: MeetingState,
    pub transcript: Vec<TranscriptSegment>,
}

pub struct TranscriptSegment {
    pub timestamp: Duration,  // offset from meeting start
    pub text: String,
    pub is_final: bool,       // false = partial/streaming
}

impl MeetingSession {
    pub fn new() -> Self;
    pub fn start(&mut self);
    pub fn pause(&mut self);
    pub fn resume(&mut self);
    pub fn stop(&mut self);
    pub fn append_segment(&mut self, seg: TranscriptSegment);
    pub fn full_transcript(&self) -> String;
}
```

### 1.3 — Tauri IPC commands

**File**: `crates/murmur-copilot/src/commands.rs`

These are the Tauri commands exposed to the frontend:

```rust
#[tauri::command]
async fn start_meeting(state: State<'_, AppState>) -> Result<String, String>;

#[tauri::command]
async fn stop_meeting(state: State<'_, AppState>) -> Result<MeetingSummary, String>;

#[tauri::command]
async fn pause_meeting(state: State<'_, AppState>) -> Result<(), String>;

#[tauri::command]
async fn resume_meeting(state: State<'_, AppState>) -> Result<(), String>;

#[tauri::command]
async fn get_transcript(state: State<'_, AppState>) -> Result<Vec<TranscriptSegment>, String>;

#[tauri::command]
async fn get_meeting_state(state: State<'_, AppState>) -> Result<MeetingStateInfo, String>;
```

The `start_meeting` command:
1. Creates a new `MeetingSession`
2. Initializes `murmur_core::audio::AudioRecorder`
3. Starts `murmur_core::transcription::streaming::start_streaming()`
4. Spawns a thread that forwards `StreamingEvent` → Tauri event emitter
5. Returns the session ID

### 1.4 — Transcription bridge

**File**: `crates/murmur-copilot/src/bridge.rs`

Bridge between murmur-core streaming events and Tauri's event system:

```rust
/// Runs on a background thread. Receives StreamingEvents from murmur-core
/// and emits them as Tauri events to the frontend.
pub fn streaming_to_tauri(
    rx: mpsc::Receiver<StreamingEvent>,
    app_handle: tauri::AppHandle,
    session: Arc<Mutex<MeetingSession>>,
) {
    for event in rx {
        match event {
            StreamingEvent::PartialText { text, replace_chars } => {
                // Update session transcript
                // Emit "transcript-update" event to frontend
                app_handle.emit("transcript-update", TranscriptPayload {
                    text,
                    replace_chars,
                    timestamp_ms: /* ... */,
                }).ok();
            }
        }
    }
}
```

### 1.5 — Frontend: live transcript UI

**File**: `frontend/src/components/Transcript.tsx`

- Listens to `transcript-update` Tauri events
- Renders a scrolling transcript view with auto-scroll
- Shows partial (in-progress) text in a different style (e.g., gray/italic)
- Minimal styling — functional, not polished

**File**: `frontend/src/components/StatusBar.tsx`

- Meeting timer (00:00:00)
- Start / Pause / Stop buttons
- Connection status indicator (recording / paused / idle)

### 1.6 — Copilot config

**File**: `crates/murmur-copilot/src/config.rs`

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopilotConfig {
    pub model_size: String,         // Whisper model (shares murmur-core model list)
    pub language: String,
    pub noise_suppression: bool,
    pub overlay_opacity: f32,       // 0.0–1.0
    pub overlay_position: Position, // TopLeft, TopRight, BottomLeft, BottomRight, Custom(x, y)
    pub always_on_top: bool,
    pub stealth_mode: bool,         // Phase 2: exclude from screen capture
    pub auto_save_transcripts: bool,
    pub transcript_dir: PathBuf,    // where to save .md meeting notes
}
```

Config stored at `~/.config/murmur-copilot/config.json` (separate from dictation config).

### 1.7 — Acceptance criteria

- [ ] `cargo tauri build -p murmur-copilot` produces a working app bundle
- [ ] Launching the app shows a floating overlay window
- [ ] Clicking "Start Meeting" begins microphone recording
- [ ] Live transcript appears in the overlay with <2s latency
- [ ] Clicking "Stop Meeting" stops recording and shows final transcript
- [ ] Overlay window can be repositioned by dragging
- [ ] No interference with the `murmur` dictation binary

---

## Phase 2 — System Audio & Stealth

**Goal**: Capture both sides of the conversation (microphone + system audio).
Implement stealth mode so the overlay is invisible to screen share.

### 2.1 — System audio capture (murmur-core)

This is the hardest new technical challenge and benefits both products long-term.
Add to murmur-core since it's a reusable audio capability.

**File**: `crates/murmur-core/src/audio/system_capture.rs`

```rust
/// Platform-specific system audio capture.
pub struct SystemAudioCapture {
    // ...
}

impl SystemAudioCapture {
    /// Create a new system audio capture source.
    /// On macOS: uses ScreenCaptureKit (macOS 13+)
    /// On Linux: uses PipeWire/PulseAudio monitor source
    /// On Windows: uses WASAPI loopback capture
    pub fn new() -> Result<Self>;

    /// Start capturing system audio. Returns a stream of f32 samples
    /// resampled to 16kHz mono (matching Whisper's expected input).
    pub fn start(&mut self) -> Result<()>;

    /// Stop capturing.
    pub fn stop(&mut self);

    /// Access the shared sample buffer (same pattern as AudioRecorder).
    pub fn sample_buffer(&self) -> Arc<Mutex<Vec<f32>>>;
}
```

#### macOS implementation (`system_capture_macos.rs`)

- Use `screencapturekit-rs` or raw `SCStream` via `objc2` bindings
- `SCStreamConfiguration` with `capturesAudio = true`, `capturesVideo = false`
- Requires macOS 13+ (Ventura). Fall back gracefully on older macOS with a clear error.
- Requires Screen Recording permission in System Settings

#### Linux implementation (`system_capture_linux.rs`)

- Use PipeWire (preferred) or PulseAudio monitor source via `libpulse-binding`
- Connect to the monitor source of the default output device
- Resample to 16kHz using a simple linear resampler or `rubato` crate

#### Windows implementation (`system_capture_windows.rs`)

- Use WASAPI loopback capture via `cpal`'s loopback device support
  or direct `windows-rs` WASAPI bindings
- `IAudioClient::Initialize` with `AUDCLNT_STREAMFLAGS_LOOPBACK`

### 2.2 — Dual-stream transcription

**File**: `crates/murmur-copilot/src/dual_stream.rs`

Run two parallel transcription streams:

```rust
pub struct DualStreamTranscription {
    mic_stream: StreamingHandle,
    system_stream: StreamingHandle,
    // Merges events from both streams into a unified timeline
}

impl DualStreamTranscription {
    pub fn start(
        mic_buffer: Arc<Mutex<Vec<f32>>>,
        system_buffer: Arc<Mutex<Vec<f32>>>,
        transcriber: Arc<Transcriber>,
        tx: mpsc::Sender<MergedTranscriptEvent>,
    ) -> Self;

    pub fn stop(self);
}

pub struct MergedTranscriptEvent {
    pub source: AudioSource, // Mic or System
    pub text: String,
    pub replace_chars: usize,
    pub timestamp: Duration,
}

pub enum AudioSource {
    Microphone,  // "You" — the user
    System,      // "Them" — the other party
}
```

**Open design question**: Should both streams share one `Transcriber` (and
serialize access), or instantiate two? Two Whisper contexts doubles memory
(~100–400MB depending on model). Serialized access adds latency. Recommendation:
**share one `Transcriber`** but use a mutex with a fast-poll pattern, since
each streaming pass is only 1–2s of audio. Benchmark and revisit if latency
is unacceptable.

### 2.3 — Stealth window flags

**File**: `crates/murmur-copilot/src/overlay.rs`

```rust
/// Configure the Tauri window to be invisible to screen capture.
pub fn apply_stealth_flags(window: &tauri::WebviewWindow) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        // Set NSWindow.sharingType = .none
        // This excludes the window from screen capture, AirPlay, and screenshots
    }

    #[cfg(target_os = "linux")]
    {
        // X11: _NET_WM_BYPASS_COMPOSITOR hint
        // Wayland: layer_shell protocol or compositor-specific exclusion
        // Note: Linux stealth is compositor-dependent and may not be universally reliable
    }

    #[cfg(target_os = "windows")]
    {
        // SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)
        // Available on Windows 10 2004+
    }

    Ok(())
}

/// Additional overlay window management.
pub fn configure_overlay(window: &tauri::WebviewWindow, config: &CopilotConfig) -> Result<()> {
    window.set_always_on_top(config.always_on_top)?;
    window.set_decorations(false)?;
    // Set transparency / opacity
    // Set initial position based on config.overlay_position
    Ok(())
}
```

### 2.4 — Permissions handling

**File**: `crates/murmur-copilot/src/permissions.rs`

Extend platform permissions for copilot-specific needs:

- **macOS**: Screen Recording permission (for system audio via ScreenCaptureKit)
- **Linux**: PipeWire/PulseAudio access (usually no special permission needed)
- **Windows**: No special permissions needed for WASAPI loopback

Provide clear user-facing messages when permissions are missing, with instructions
for granting them.

### 2.5 — Frontend updates

- Add source labels to transcript ("You" / "Them" or speaker icons)
- Add stealth mode toggle in settings
- Visual indicator when stealth mode is active
- System audio permission status indicator

### 2.6 — Acceptance criteria

- [ ] System audio captures desktop audio output (test with music playback)
- [ ] Dual-stream transcription shows mic and system audio separately
- [ ] Transcript labels correctly identify "You" (mic) vs "Them" (system)
- [ ] Stealth mode hides overlay from screen share (test with OBS/Zoom share)
- [ ] Stealth mode can be toggled at runtime
- [ ] Graceful fallback when system audio capture is unavailable
- [ ] macOS Screen Recording permission is requested on first use

---

## Phase 3 — Local LLM Integration

**Goal**: Integrate a local LLM runtime for contextual suggestions and
post-meeting summaries. Zero-config start with a bundled small model.

### 3.1 — LLM runtime abstraction

**File**: `crates/murmur-copilot/src/llm/mod.rs`

```rust
pub mod local;
pub mod cloud;
pub mod prompt;

/// Trait for LLM backends (local or cloud).
#[async_trait]
pub trait LlmBackend: Send + Sync {
    /// Generate a completion for the given prompt.
    async fn complete(&self, prompt: &str, max_tokens: usize) -> Result<String>;

    /// Stream a completion token-by-token.
    async fn stream_complete(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>>;

    /// Human-readable name (e.g., "Phi-3-mini (local)" or "GPT-4o (cloud)")
    fn model_name(&self) -> &str;

    /// Whether this backend requires network access
    fn is_local(&self) -> bool;
}
```

### 3.2 — Local LLM via llama.cpp

**File**: `crates/murmur-copilot/src/llm/local.rs`

Use `llama-cpp-rs` (Rust bindings to llama.cpp) for local inference:

```rust
pub struct LocalLlm {
    model: LlamaModel,
    context: LlamaContext,
    model_name: String,
}

impl LocalLlm {
    /// Load a GGUF model from the given path.
    pub fn new(model_path: &Path) -> Result<Self>;

    /// Download a default small model if none exists.
    /// Candidates: Phi-3.5-mini-instruct (3.8B, ~2.3GB Q4),
    ///             Qwen2.5-1.5B-Instruct (~1GB Q4),
    ///             Gemma-2-2B-it (~1.5GB Q4)
    pub fn download_default(on_progress: impl Fn(f64)) -> Result<PathBuf>;
}

impl LlmBackend for LocalLlm {
    // ...
}
```

**Model storage**: `~/.local/share/murmur-copilot/models/` (same pattern as
murmur's Whisper model storage, using `dirs::data_dir()`).

**Dependency**: Add `llama-cpp-2` crate (Rust bindings with auto-build of
llama.cpp, supports Metal/CUDA acceleration).

### 3.3 — Cloud LLM opt-in

**File**: `crates/murmur-copilot/src/llm/cloud.rs`

Optional cloud backends for users who want higher-quality responses:

```rust
pub struct OpenAiBackend {
    api_key: String,
    model: String, // "gpt-4o-mini", "gpt-4o", etc.
    client: reqwest::Client,
}

pub struct AnthropicBackend {
    api_key: String,
    model: String, // "claude-sonnet-4-20250514", etc.
    client: reqwest::Client,
}

// Implement LlmBackend for each
```

API keys stored in config file or system keychain. Cloud is purely opt-in —
the app works fully without any API keys.

### 3.4 — Prompt engineering

**File**: `crates/murmur-copilot/src/llm/prompt.rs`

Pre-built prompt templates for meeting intelligence:

```rust
/// Generate a contextual suggestion based on recent transcript.
pub fn suggestion_prompt(recent_transcript: &str, full_context: &str) -> String;

/// Generate a post-meeting summary with key points and action items.
pub fn summary_prompt(full_transcript: &str) -> String;

/// Generate follow-up questions based on what was discussed.
pub fn followup_prompt(transcript: &str) -> String;

/// Generate a response suggestion when the user is asked a question.
pub fn response_prompt(transcript: &str, question_context: &str) -> String;
```

Prompts should be:
- Concise (small models have limited context windows — 4K–8K tokens)
- Structured (use clear section markers so small models don't go off-track)
- Configurable (allow users to modify system prompts via config)

### 3.5 — Suggestion engine

**File**: `crates/murmur-copilot/src/suggestions.rs`

The suggestion engine monitors the transcript and periodically generates
contextual suggestions:

```rust
pub struct SuggestionEngine {
    llm: Arc<dyn LlmBackend>,
    /// How many transcript segments to include in the prompt context window
    context_window: usize,
    /// Minimum interval between suggestion generations
    cooldown: Duration,
}

impl SuggestionEngine {
    pub fn new(llm: Arc<dyn LlmBackend>) -> Self;

    /// Called when new transcript text arrives. Decides whether to
    /// generate a new suggestion based on cooldown and content heuristics.
    pub async fn on_transcript_update(
        &mut self,
        transcript: &[TranscriptSegment],
    ) -> Option<Suggestion>;
}

pub struct Suggestion {
    pub id: String,
    pub kind: SuggestionKind,
    pub text: String,
    pub confidence: f32,
    pub generated_at: Instant,
}

pub enum SuggestionKind {
    TalkingPoint,       // "You could mention..."
    FollowUp,           // "Ask about..."
    Clarification,      // "They mentioned X, you could clarify..."
    ActionItem,         // "Action item detected: ..."
}
```

### 3.6 — Post-meeting summary

**File**: `crates/murmur-copilot/src/summary.rs`

When a meeting ends, automatically generate and save a summary:

```rust
pub struct MeetingSummary {
    pub meeting_id: String,
    pub duration: Duration,
    pub key_points: Vec<String>,
    pub action_items: Vec<ActionItem>,
    pub decisions: Vec<String>,
    pub full_transcript: String,
    pub generated_at: DateTime<Utc>,
}

pub struct ActionItem {
    pub description: String,
    pub assignee: Option<String>, // from speaker diarization (Phase 4)
    pub due_date: Option<String>, // if mentioned in meeting
}

/// Generate a meeting summary from the full transcript.
pub async fn generate_summary(
    transcript: &[TranscriptSegment],
    llm: &dyn LlmBackend,
) -> Result<MeetingSummary>;

/// Save the summary as a Markdown file.
pub fn save_summary(summary: &MeetingSummary, dir: &Path) -> Result<PathBuf>;
```

Output format: Markdown file saved to `~/.local/share/murmur-copilot/meetings/`:
```
meetings/
├── 2026-03-27-standup.md
├── 2026-03-27-1on1.md
└── ...
```

### 3.7 — Frontend: suggestions & summary UI

**File**: `frontend/src/components/Suggestions.tsx`
- Real-time suggestion cards that appear alongside the transcript
- Dismiss, pin, or copy suggestions
- Suggestion kind icons/colors

**File**: `frontend/src/components/Summary.tsx`
- Post-meeting summary view
- Key points, action items, decisions in structured layout
- Export as Markdown / copy to clipboard

**File**: `frontend/src/components/Settings.tsx`
- LLM backend selection (Local / OpenAI / Anthropic)
- API key input for cloud backends
- Model selection for local backend
- Suggestion frequency / sensitivity controls

### 3.8 — Acceptance criteria

- [ ] Local LLM loads and generates text without network access
- [ ] Suggestions appear during a meeting with reasonable quality
- [ ] Post-meeting summary is generated and saved as Markdown
- [ ] Cloud LLM backend works when API key is configured
- [ ] App works fully without any API keys (local-only)
- [ ] LLM inference doesn't block the transcription pipeline
- [ ] Model download shows progress and handles errors gracefully

---

## Phase 4 — Polish & Release

**Goal**: Speaker diarization, session management, meeting history,
cross-platform testing, packaging, and documentation.

### 4.1 — Speaker diarization

Add to murmur-core (reusable capability):

**File**: `crates/murmur-core/src/transcription/diarization.rs`

- Use `pyannote-rs` or a Rust ONNX-based speaker embedding model
- Cluster speaker embeddings to identify distinct speakers
- Label transcript segments with speaker IDs
- In dual-stream mode (Phase 2), correlate: mic = "You", system = "Them",
  then sub-diarize the system audio if multiple remote speakers

### 4.2 — Session persistence & history

**File**: `crates/murmur-copilot/src/storage.rs`

- Store meeting sessions in a local SQLite database (`rusqlite`)
- Schema: `meetings(id, title, started_at, ended_at, duration_secs, transcript_path, summary_path)`
- Full-text search over past transcripts
- Meeting history browser in the frontend

### 4.3 — Cross-platform testing & packaging

- **macOS**: `.dmg` installer via `cargo tauri build`
- **Linux**: `.AppImage` and `.deb` packages
- **Windows**: `.msi` installer (Phase 4 stretch goal)
- CI matrix: build & test on macOS (ARM + Intel), Ubuntu, Windows

### 4.4 — Documentation

- README for murmur-copilot (separate from murmur README)
- User guide: installation, first meeting, configuration
- Architecture doc: how the components fit together
- Contributing guide for the copilot crate

### 4.5 — Acceptance criteria

- [ ] Speaker diarization identifies 2+ speakers accurately
- [ ] Meeting history shows past meetings with search
- [ ] App installs cleanly on macOS and Linux
- [ ] Documentation covers installation, usage, and configuration
- [ ] No regressions in the `murmur` dictation binary

---

## Dependency Map

```
Phase 0 ─── Phase 1 ─── Phase 2 ─── Phase 4
  │                        │
  │                        └── Phase 3 ─── Phase 4
  │
  └── (murmur binary continues working throughout)
```

- **Phase 0** is a prerequisite for all other phases
- **Phase 2** and **Phase 3** can be developed in parallel after Phase 1
- **Phase 4** depends on both Phase 2 and Phase 3

Within each phase, tasks are ordered by dependency (earlier tasks first).

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| System audio capture is platform-fragile | High | Medium | Prototype on macOS first; abstract behind trait; accept Linux/Windows may lag |
| Local LLM quality too low for useful suggestions | Medium | Medium | Test multiple small models early; provide cloud opt-in as escape hatch |
| Dual Whisper contexts exceed memory budget | Medium | Low | Share one context with mutex; benchmark serialized vs. parallel |
| Tauri 2.0 API changes during development | Low | Low | Pin Tauri version; follow stable releases |
| ScreenCaptureKit requires macOS 13+ | Low | Low | Document minimum macOS version; majority of users are on 13+ |
| Config split introduces breaking changes for existing murmur users | High | Medium | Phase 0 migration: detect old config location, auto-migrate |
| llama.cpp build complexity on CI | Medium | Medium | Use pre-built binaries or `llama-cpp-2` crate with vendored build |
| Legal risk from recording system audio | Medium | Low | Require explicit user consent; document legal considerations; do not auto-start |
