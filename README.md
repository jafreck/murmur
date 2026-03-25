# open-bark

Cross-platform, local voice dictation. Hold a key, speak, release — your words appear at the cursor.

Everything runs on-device. No audio or text ever leaves your machine. Powered by [whisper.cpp](https://github.com/ggml-org/whisper.cpp) via [whisper-rs](https://codeberg.org/tazz4843/whisper-rs).

## How It Works

1. Hold the hotkey (default: `Ctrl+Shift+Space` on Windows/Linux, `Globe` on macOS)
2. Speak naturally
3. Release — transcribed text is pasted at your cursor

## Install

### Quick install (recommended)

Pre-built binaries — no build tools required. macOS Apple Silicon builds include Metal GPU acceleration.

**macOS / Linux:**

```bash
curl -sSf https://raw.githubusercontent.com/jacobfreck/open-bark/main/scripts/install.sh | bash
```

**Windows (PowerShell):**

```powershell
irm https://raw.githubusercontent.com/jacobfreck/open-bark/main/scripts/install.ps1 | iex
```

The installer downloads the correct binary for your platform, installs it, and registers open-bark as a service that starts at login.

To install a specific version:

```bash
# macOS / Linux
OPEN_BARK_VERSION=v0.1.0 bash <(curl -sSf https://raw.githubusercontent.com/jacobfreck/open-bark/main/scripts/install.sh)

# Windows
.\scripts\install.ps1 -Version v0.1.0
```

### From source

```bash
git clone https://github.com/jacobfreck/open-bark.git
cd open-bark
cargo build --release
```

The binary is at `target/release/open-bark`.

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
cargo install open-bark
```

## Usage

```bash
# Start the dictation daemon (system tray)
open-bark start

# Download a specific model
open-bark download-model base.en

# Set the hotkey
open-bark set-hotkey ctrl+shift+space

# Show status
open-bark status
```

## Configuration

Edit the config file:
- **macOS:** `~/Library/Application Support/open-bark/config.json`
- **Windows:** `%APPDATA%\open-bark\config.json`
- **Linux:** `~/.config/open-bark/config.json`

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

| Model | Size | Speed | Accuracy | Best for |
|---|---|---|---|---|
| `tiny.en` | 75 MB | Fastest | Lower | Quick notes |
| **`base.en`** | 142 MB | **Fast** | **Good** | **Most users (default)** |
| `small.en` | 466 MB | Moderate | Better | Technical terms |
| `medium.en` | 1.5 GB | Slower | Great | Maximum accuracy |
| `large-v3-turbo` | 1.6 GB | Moderate | Great | Multilingual |
| `large` | 3 GB | Slowest | Best | Highest accuracy |

## Privacy

open-bark is completely local. Audio is recorded to a temp file, transcribed by whisper.cpp on your CPU/GPU, and the temp file is deleted. No network requests are made except to download the Whisper model on first run.

## Platform Notes

- **macOS:** Requires Accessibility and Microphone permissions
- **Windows:** May need to allow through antivirus (keyboard hook for hotkey detection)
- **Linux (X11):** Works out of the box
- **Linux (Wayland):** User must be in the `input` group for hotkey detection

## License

MIT
