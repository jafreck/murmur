# murmur-copilot

> [!WARNING]
> murmur-copilot is in preview and under active development. APIs and features may change.

A local meeting assistant built on top of [murmur](../../README.md). It provides a transparent overlay that live-transcribes meetings, generates AI-powered suggestions, and saves session history — all 100% on-device.

## Features

- **Live streaming transcription** — real-time speech-to-text displayed in a floating overlay
- **Dual-stream audio** — captures both your microphone (you) and system audio (remote participants) simultaneously
- **LLM-powered suggestions** — contextual talking points and responses via a local Ollama model
- **Meeting summaries** — auto-generate summaries and action items when a meeting ends
- **Session history** — browse, review, and export past meetings as Markdown
- **Speaker labelling** — transcript entries are tagged as "You" or "Remote"

## Prerequisites

| Dependency | Purpose | Required? |
|---|---|---|
| **Rust toolchain** | Build from source | Yes |
| **Tauri v2 CLI** | Build the desktop app | Yes |
| **Node.js ≥ 18** | Frontend build tooling | Yes |
| **Ollama** | Local LLM for suggestions/summaries | Optional |
| **Virtual audio device** | System audio capture (e.g. BlackHole on macOS) | Optional |

### Ollama setup (optional)

```bash
# Install Ollama
brew install ollama   # macOS
# or see https://ollama.ai

# Pull a model
ollama pull phi3

# Start the server (runs on http://localhost:11434)
ollama serve
```

### System audio capture (optional)

To capture remote participants' audio you need a virtual audio device that mirrors system output:

1. Install [BlackHole](https://existential.audio/blackhole/) (free) or [Loopback](https://rogueamoeba.com/loopback/) (paid)
2. Create a Multi-Output Device in Audio MIDI Setup that sends audio to both your speakers and BlackHole
3. Set the Multi-Output Device as your system output
4. In murmur-copilot, select the BlackHole device from the "System Audio" dropdown

## Build

```bash
cd crates/murmur-copilot

# Install frontend dependencies
npm install

# Development build with hot reload
npm run tauri dev

# Production build
npm run tauri build
```

## Configuration

murmur-copilot shares configuration with murmur via `~/.config/murmur/config.json`. Copilot-specific fields:

```json
{
  "system_audio_device": "BlackHole 2ch",
  "llm_model": "phi3",
  "ollama_url": "http://localhost:11434",
  "sessions_dir": null,
  "auto_summary": false
}
```

| Field | Default | Description |
|---|---|---|
| `system_audio_device` | `null` | Name of the virtual audio input device for remote audio |
| `llm_model` | `"phi3"` | Ollama model name for suggestions and summaries |
| `ollama_url` | `"http://localhost:11434"` | Ollama API base URL |
| `sessions_dir` | `null` | Custom directory for saved sessions (default: `~/.config/murmur-copilot/sessions/`) |
| `auto_summary` | `false` | Automatically generate a summary when a meeting ends |

## Usage

1. **Start a meeting** — click "Start Meeting" or use the keyboard shortcut
2. **Speak** — your words appear in the transcript in real time
3. **Get suggestions** — click "Get Suggestion" for AI-powered talking points
4. **Stop the meeting** — click "Stop Meeting" to end; a summary is auto-generated if Ollama is running
5. **Browse history** — click "📋 History" to view past meetings, export transcripts, or delete sessions

## Architecture

```
murmur-copilot
├── src/              # Frontend (TypeScript + HTML + CSS)
│   ├── App.ts        # Main app controller
│   ├── transcript.ts # Live transcript display
│   ├── history.ts    # Meeting history browser
│   └── index.html    # Overlay UI
└── src-tauri/        # Backend (Rust + Tauri)
    └── src/
        ├── main.rs      # App entry point
        ├── commands.rs  # Tauri IPC commands
        ├── meeting.rs   # Meeting session + audio pipeline
        ├── session.rs   # Persistent session storage
        ├── llm.rs       # Ollama LLM integration
        └── overlay.rs   # Window management
```

### Crate dependencies

- **murmur-core** — audio capture, ASR transcription, streaming, config, LLM abstractions
- **tauri** — desktop app framework (window management, IPC, system tray)

## Platform support

| Feature | macOS | Windows | Linux |
|---|---|---|---|
| Microphone capture | ✅ | ✅ | ✅ |
| System audio capture | ✅ (BlackHole) | ⬚ Stub | ⬚ Stub |
| LLM suggestions | ✅ | ✅ | ✅ |
| Session history | ✅ | ✅ | ✅ |

## License

MIT
