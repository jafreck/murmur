<div align="center">

# murmur

100% on-device voice dictation. Your audio never leaves your machine.

[![CI](https://github.com/jafreck/murmur/actions/workflows/ci.yml/badge.svg)](https://github.com/jafreck/murmur/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey.svg)](#platform-notes)

Worry free speech to text.
No cloud, no API keys, no data collection.

</div>

## How It Works

1. Hold the hotkey (default: `Right Option` on macOS, `Right Alt` on Windows/Linux)
2. Speak naturally
3. Release — transcribed text is pasted at your cursor

## Features

### 🤖 AI-Powered Transcription

Murmur runs [OpenAI Whisper](https://github.com/openai/whisper) — a state-of-the-art speech recognition model — entirely on your machine via [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

- **Accurate speech recognition** — Whisper was trained on 680,000 hours of multilingual audio, delivering near-human accuracy
- **90+ languages** — transcribe in your language or auto-detect it, with optional translate-to-English mode
- **Smart vocabulary biasing** — provide domain-specific terms and Murmur prioritizes them during transcription using intelligent prompt engineering that ranks novel multi-token terms higher
- **Context-aware formatting** — detects the active application and adjusts transcription formatting automatically

### ⚡ What Murmur Adds on Top of Whisper

Raw Whisper is a CLI inference tool. Murmur turns it into a seamless dictation system:

- **Push to Talk** — hold a key to record, release to transcribe and paste
- **Open Mic** — toggle recording on/off with a single keypress
- **Instant paste** — transcribed text is inserted at your cursor automatically, preserving your clipboard contents (text and images)
- **Live streaming (preview)** — see partial transcriptions appear in real time as you speak
- **Spoken punctuation** — say "period", "comma", "question mark", "new paragraph", etc. and they're converted to symbols
- **Filler word removal** — automatically strips "um", "uh", "er", "ah", and other verbal fillers
- **Noise suppression** — built-in audio denoising via [nnnoiseless](https://github.com/jneem/nnnoiseless) cleans up background noise before transcription
- **Hallucination filtering** — detects and discards phantom text that Whisper sometimes generates on silence or very short clips
- **Voice activity detection** — [Silero VAD](https://github.com/snakers4/silero-vad) accurately distinguishes speech from silence, replacing basic energy-level detection
- **System tray UI** — control everything from the menu bar: model, language, mode, hotkey, and all toggles
- **Hotkey rebinding** — set any key or combo as your trigger from the tray menu or config
- **Model management** — download and switch between Whisper models with a single command
- **Copy last dictation** — retrieve your most recent transcription from the tray menu
- **Homebrew support** — `brew install jafreck/murmur/murmur`

### 🔒 Privacy

100% local. Audio is captured, transcribed on your CPU/GPU, and discarded. No network requests are ever made except to download the Whisper model on first run.

### 🚀 Performance

Murmur is built in Rust with an optimized audio pipeline designed for low-latency dictation:

| Metric | Detail |
|---|---|
| **Pre-roll buffer** | 200 ms of audio captured before you press the hotkey — your first words are never clipped |
| **Minimum audio** | Processes recordings as short as 0.25 seconds |
| **Streaming latency** | Partial results update every ~300 ms while speaking |
| **In-memory pipeline** | Zero disk I/O by default — audio is recorded and transcribed entirely in memory |
| **GPU acceleration** | Metal (Apple Silicon), CUDA (NVIDIA), and Vulkan (cross-vendor) for faster-than-real-time inference |
| **Binary optimization** | Release builds use `opt-level = 3`, LTO, and symbol stripping for minimal overhead |
| **Distil-Whisper** | Distilled models run significantly faster while maintaining near-original accuracy |

## Install

### Quick install (recommended)

Pre-built binaries — no build tools required. macOS Apple Silicon builds include Metal GPU acceleration.

**macOS / Linux:**

```bash
curl -sSfL https://github.com/jafreck/murmur/releases/latest/download/install.sh | bash
```

**Windows (PowerShell):**

```powershell
irm https://github.com/jafreck/murmur/releases/latest/download/install.ps1 | iex
```

The installer downloads the correct binary for your platform, installs it, and registers murmur as a service that starts at login.

To install a specific version:

```bash
# macOS / Linux
MURMUR_VERSION=v0.1.0 bash <(curl -sSfL https://github.com/jafreck/murmur/releases/latest/download/install.sh)

# Windows
.\scripts\install.ps1 -Version v0.1.0
```

### Homebrew (macOS / Linux)

```bash
brew install jafreck/murmur/murmur
```

### From source

```bash
git clone https://github.com/jafreck/murmur.git
cd murmur
cargo build --release
```

The binary is at `target/release/murmur`.

#### GPU acceleration (optional)

```bash
# macOS Apple Silicon (Metal)
cargo build --release --features metal

# NVIDIA (CUDA)
cargo build --release --features cuda

# Cross-vendor (Vulkan)
cargo build --release --features vulkan
```

### From crates.io

```bash
cargo install murmur
```

## Usage

```bash
# Start the dictation daemon (system tray)
murmur start

# Download a specific model
murmur download-model base.en

# Set the hotkey
murmur set-hotkey ctrl+shift+space

# Show status
murmur status
```

## Configuration

Edit the config file:
- **macOS:** `~/Library/Application Support/murmur/config.json`
- **Windows:** `%APPDATA%\murmur\config.json`
- **Linux:** `~/.config/murmur/config.json`

```json
{
  "hotkey": "ctrl+shift+space",
  "model_size": "base.en",
  "language": "en",
  "spoken_punctuation": false,
  "filler_word_removal": false,
  "noise_suppression": true,
  "translate_to_english": false,
  "max_recordings": 0,
  "mode": "push_to_talk",
  "streaming": false,
  "vocabulary": []
}
```

### Models

murmur uses [OpenAI Whisper](https://github.com/openai/whisper) models running locally via [whisper.cpp](https://github.com/ggml-org/whisper.cpp). Models are downloaded in GGML format on first run.

| Model | Disk Size | Memory | Speed | Accuracy | Best for |
|---|---|---|---|---|---|
| `tiny.en` | 75 MB | ~273 MB | Fastest | Lower | Quick notes |
| **`base.en`** | **142 MB** | **~388 MB** | **Fast** | **Good** | **Most users (default)** |
| `small.en` | 466 MB | ~852 MB | Moderate | Better | Technical terms |
| `medium.en` | 1.5 GB | ~2.1 GB | Slower | Great | Maximum accuracy |
| `large-v3-turbo` | 1.6 GB | ~2 GB | Moderate | Great | Multilingual |
| `large` | 3 GB | ~3.9 GB | Slowest | Best | Highest accuracy |

#### Distil-Whisper (faster alternatives)

[Distil-Whisper](https://github.com/huggingface/distil-whisper) models are distilled versions that run significantly faster while maintaining near-original accuracy. English-only.

| Model | Disk Size | Memory | Speed | Accuracy | Best for |
|---|---|---|---|---|---|
| `distil-large-v3` | ~1.5 GB | ~2–3 GB | Fast | Great | Best distilled quality |

## Platform Notes

- **macOS:** Requires Accessibility and Microphone permissions
- **Windows:** May need to allow through antivirus (keyboard hook for hotkey detection)
- **Linux (X11):** Works out of the box
- **Linux (Wayland):** User must be in the `input` group for hotkey detection

## Uninstall

**macOS / Linux:**

```bash
curl -sSfL https://github.com/jafreck/murmur/releases/latest/download/uninstall.sh | bash
```

**Windows (PowerShell):**

```powershell
irm https://github.com/jafreck/murmur/releases/latest/download/uninstall.ps1 | iex
```

Removes the binary, service/startup config, and logs. Prompts before deleting user config and downloaded models.

## License

MIT

## Contributing

```bash
git config core.hooksPath hooks
```

This enables the pre-push hook which runs `cargo fmt --check` and `cargo clippy` before each push.
