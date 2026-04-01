<div align="center">

# murmur

100% on-device voice dictation. Your audio never leaves your machine.

[![CI](https://github.com/jafreck/murmur/actions/workflows/ci.yml/badge.svg)](https://github.com/jafreck/murmur/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey.svg)](#platform-notes)

No cloud, no API keys, no data collection.
State-of-the-art speech recognition running entirely on your hardware.

</div>

## How It Works

1. Hold the hotkey (default: `Right Option` on macOS, `Right Alt` on Windows/Linux)
2. Speak naturally
3. Release — transcribed text is pasted at your cursor

## Features

### 🎙️ Multi-Engine Speech Recognition

Murmur supports multiple ASR (Automatic Speech Recognition) backends — all running locally on your machine. Choose the engine that best fits your needs:

| Engine | Models | Languages | Streaming | Strengths |
|---|---|---|---|---|
| [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) | 0.6B, 1.7B | 52 | ✅ Native | Lowest word error rate, native streaming |
| [Whisper](https://github.com/openai/whisper) | tiny → large | 90+ | ⚠️ Chunked | Broad language coverage |
| [Parakeet-TDT](https://catalog.ngc.nvidia.com/orgs/nvidia/collections/parakeet-tdt-0.6b) | 0.6B | English | ⚠️ Chunked | Fast, pre-formatted output with punctuation |

Switch backends with a single flag:

```bash
murmur start --backend qwen3-asr    # Best accuracy + native streaming
murmur start --backend whisper      # Maximum language support
murmur start --backend parakeet     # Fast English with auto-punctuation
```

### ⚡ Real-Time Streaming

Murmur delivers live partial transcriptions as you speak. Backends with native streaming support produce smooth, append-only text updates with no flickering or re-typing.

### 🔒 Privacy First

Everything runs on your hardware. Audio is captured, transcribed, and discarded — never written to disk, never sent over the network. The only network requests murmur ever makes are to download model weights on first run.

- No cloud APIs, no telemetry, no data collection
- Audio processed entirely in memory
- Models downloaded once, run forever offline

### 🛠️ Dictation Intelligence

Raw speech recognition is just the foundation. Murmur adds layers that make dictation practical:

- **Two recording modes** — Push to Talk (hold to record) or Open Mic (toggle on/off), with instant paste at your cursor that preserves clipboard contents
- **Spoken punctuation** — say "period", "comma", "question mark", "new paragraph" and they're converted to symbols
- **Filler word removal** — automatically strips "um", "uh", "er", "ah" and other verbal fillers
- **Noise suppression** — built-in audio denoising cleans up background noise before transcription
- **Speech detection** — voice activity detection and hallucination filtering ensure only real speech is transcribed
- **Smart vocabulary biasing** — provide domain-specific terms and murmur prioritizes them during transcription
- **Context-aware formatting** — detects the active application and adjusts transcription formatting automatically
- **System tray UI** — control model, backend, language, mode, hotkey, and all settings from the menu bar

### 🚀 Performance

Murmur is built in Rust with an optimized audio pipeline designed for low-latency dictation:

| Metric | Detail |
|---|---|
| **Pre-roll buffer** | 200 ms of audio captured before you press the hotkey — first words are never clipped |
| **Minimum audio** | Processes recordings as short as 0.25 seconds |
| **Streaming latency** | Partial results update every ~300 ms while speaking |
| **In-memory pipeline** | Zero disk I/O — audio is recorded and transcribed entirely in memory |
| **GPU acceleration** | Metal (Apple Silicon), CUDA (NVIDIA), Vulkan (cross-vendor) |
| **Quantized models** | INT4/INT8 inference for faster speed with minimal accuracy loss |

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

### Homebrew (macOS / Linux)

```bash
brew install jafreck/murmur/murmur
```

### From source

```bash
git clone https://github.com/jafreck/murmur.git
cd murmur
cargo build --release --features metal     # Whisper backend (default)
cargo build --release --features metal,onnx # + Qwen3-ASR & Parakeet backends
```

#### GPU acceleration

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
# Start with default backend (Whisper)
murmur start

# Start with a specific backend
murmur start --backend qwen3-asr

# Download a model
murmur download-model base.en                       # Whisper
murmur download-model 0.6b --backend qwen3-asr      # Qwen3-ASR
murmur download-model 0.6b-v2 --backend parakeet    # Parakeet

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
  "hotkey": "rightoption",
  "asr_backend": "qwen3_asr",
  "model_size": "0.6b",
  "asr_quantization": "int4",
  "language": "en",
  "spoken_punctuation": false,
  "filler_word_removal": false,
  "noise_suppression": true,
  "translate_to_english": false,
  "mode": "push_to_talk",
  "streaming": false,
  "vocabulary": []
}
```

### Models

Models are downloaded from HuggingFace on first run and cached locally. Each backend has its own model family:

#### Qwen3-ASR (recommended)

| Model | Quantization | Disk Size | WER (LibriSpeech) | Best for |
|---|---|---|---|---|
| **`0.6b`** | **INT4** | **~2 GB** | **5.16%** | **Fast, accurate (default)** |
| `0.6b` | FP32 | ~3.3 GB | 4.42% | Maximum accuracy |
| `1.7b` | INT4 | ~4 GB | 4.20% | Lowest error rate |

#### Whisper

| Model | Disk Size | Speed | Best for |
|---|---|---|---|
| `tiny.en` | 75 MB | Fastest | Quick notes |
| **`base.en`** | **142 MB** | **Fast** | **Most users** |
| `small.en` | 466 MB | Moderate | Technical terms |
| `medium.en` | 1.5 GB | Slower | High accuracy |
| `large-v3-turbo` | 1.6 GB | Moderate | Multilingual |
| `distil-large-v3` | ~1.5 GB | Fast | Best distilled quality |

#### Parakeet-TDT

| Model | Quantization | Disk Size | Best for |
|---|---|---|---|
| **`0.6b-v2`** | **INT8** | **~500 MB** | **Fast English with auto-punctuation** |

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

## murmur-copilot

> [!WARNING]
> murmur-copilot is in preview and under active development. APIs and features may change.

murmur-copilot is a local meeting assistant built on top of murmur. It provides a transparent overlay that live-transcribes meetings, captures both your microphone and remote participants' system audio, and offers LLM-powered suggestions and summaries — all running on-device.

**Key features:** live dual-stream transcription, AI suggestions via local Ollama, meeting session history with export.

See [crates/murmur-copilot/README.md](crates/murmur-copilot/README.md) for build instructions, configuration, and usage.

## License

MIT

## Contributing

```bash
git config core.hooksPath .githooks
```

This enables the pre-push hook which runs `cargo fmt --check` and `cargo clippy` before each push.
