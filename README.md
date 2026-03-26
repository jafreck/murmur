<div align="center">

# murmur

100% on-device voice dictation. Your audio never leaves your machine.

[![CI](https://github.com/jafreck/murmur/actions/workflows/ci.yml/badge.svg)](https://github.com/jafreck/murmur/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey.svg)](#platform-notes)

Hold a key, speak, release — your words appear at the cursor.
No cloud, no API keys, no data collection. Powered by [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

</div>

## How It Works

1. Hold the hotkey (default: `Right Option` on macOS, `Right Alt` on Windows/Linux)
2. Speak naturally
3. Release — transcribed text is pasted at your cursor

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

## Features

- **Push to Talk** — hold a key to record, release to transcribe
- **Open Mic** — toggle recording on/off with a keypress
- **Spoken Punctuation** — say "period", "comma", etc. and they're converted to symbols
- **Streaming (preview)** — see partial transcriptions as you speak. Enable via the tray menu or `"streaming": true` in config. This feature is functional but still being refined.

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
  "max_recordings": 0,
  "mode": "push_to_talk"
}
```

### Models

murmur uses [OpenAI Whisper](https://github.com/openai/whisper) models running locally via [whisper.cpp](https://github.com/ggml-org/whisper.cpp). Models are downloaded in GGML format on first run.

| Model | Size | Speed | Accuracy | Best for |
|---|---|---|---|---|
| `tiny.en` | 75 MB | Fastest | Lower | Quick notes |
| **`base.en`** | 142 MB | **Fast** | **Good** | **Most users (default)** |
| `small.en` | 466 MB | Moderate | Better | Technical terms |
| `medium.en` | 1.5 GB | Slower | Great | Maximum accuracy |
| `large-v3-turbo` | 1.6 GB | Moderate | Great | Multilingual |
| `large` | 3 GB | Slowest | Best | Highest accuracy |

## Privacy

murmur is completely local. Audio is recorded to a temp file, transcribed by whisper.cpp on your CPU/GPU, and the temp file is deleted. No network requests are made except to download the Whisper model on first run.

## Platform Notes

- **macOS:** Requires Accessibility and Microphone permissions
- **Windows:** May need to allow through antivirus (keyboard hook for hotkey detection)
- **Linux (X11):** Works out of the box
- **Linux (Wayland):** User must be in the `input` group for hotkey detection

## License

MIT
