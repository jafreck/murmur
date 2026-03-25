#!/usr/bin/env bash
#
# open-bark install script
#
# Detects OS and architecture, builds with appropriate feature flags
# (Metal on Apple Silicon macOS), installs the binary, and registers
# open-bark as a user-level service (launchd on macOS, systemd on Linux).
#
# Usage:
#   curl -sSf https://raw.githubusercontent.com/jacobfreck/open-bark/main/scripts/install.sh | bash
#   # or from a local clone:
#   ./scripts/install.sh

set -euo pipefail

REPO_URL="https://github.com/jacobfreck/open-bark.git"
INSTALL_DIR="/usr/local/bin"
APP_NAME="open-bark"

# ── Helpers ──────────────────────────────────────────────────────────────────

info()  { printf "\033[1;34m==>\033[0m %s\n" "$1"; }
warn()  { printf "\033[1;33mWarning:\033[0m %s\n" "$1"; }
error() { printf "\033[1;31mError:\033[0m %s\n" "$1" >&2; exit 1; }

require() {
    command -v "$1" >/dev/null 2>&1 || error "'$1' is required but not found. Please install it first."
}

# ── Detect platform ─────────────────────────────────────────────────────────

detect_platform() {
    OS="$(uname -s)"
    ARCH="$(uname -m)"

    case "$OS" in
        Darwin) PLATFORM="macos" ;;
        Linux)  PLATFORM="linux" ;;
        *)      error "Unsupported OS: $OS. Only macOS and Linux are supported." ;;
    esac

    case "$ARCH" in
        arm64|aarch64) ARCH_LABEL="arm64" ;;
        x86_64)        ARCH_LABEL="x86_64" ;;
        *)             error "Unsupported architecture: $ARCH" ;;
    esac

    info "Detected: $PLATFORM ($ARCH_LABEL)"
}

# ── Choose build features ───────────────────────────────────────────────────

choose_features() {
    CARGO_FEATURES=""

    if [ "$PLATFORM" = "macos" ] && [ "$ARCH_LABEL" = "arm64" ]; then
        CARGO_FEATURES="--features metal"
        info "Apple Silicon detected — building with Metal acceleration"
    elif [ "$PLATFORM" = "macos" ] && [ "$ARCH_LABEL" = "x86_64" ]; then
        info "Intel Mac detected — building without GPU acceleration"
    elif [ "$PLATFORM" = "linux" ]; then
        # Check for NVIDIA GPU / CUDA toolkit
        if command -v nvcc >/dev/null 2>&1; then
            CARGO_FEATURES="--features cuda"
            info "CUDA toolkit detected — building with CUDA acceleration"
        elif command -v vulkaninfo >/dev/null 2>&1; then
            CARGO_FEATURES="--features vulkan"
            info "Vulkan detected — building with Vulkan acceleration"
        else
            info "No GPU toolkit detected — building CPU-only"
        fi
    fi
}

# ── Install system dependencies ──────────────────────────────────────────────

install_dependencies() {
    if [ "$PLATFORM" = "linux" ]; then
        info "Installing Linux build dependencies..."
        if command -v apt-get >/dev/null 2>&1; then
            sudo apt-get update -qq
            sudo apt-get install -y -qq build-essential cmake libasound2-dev \
                libgtk-3-dev libayatana-appindicator3-dev pkg-config
        elif command -v dnf >/dev/null 2>&1; then
            sudo dnf install -y gcc gcc-c++ cmake alsa-lib-devel \
                gtk3-devel libayatana-appindicator-gtk3-devel pkg-config
        elif command -v pacman >/dev/null 2>&1; then
            sudo pacman -Sy --noconfirm base-devel cmake alsa-lib \
                gtk3 libayatana-appindicator pkg-config
        else
            warn "Unknown Linux package manager. Ensure build deps are installed:"
            warn "  cmake, alsa dev headers, gtk3 dev headers, appindicator dev headers"
        fi
    fi
    # macOS: Xcode Command Line Tools provide everything needed
}

# ── Install Rust (if needed) ────────────────────────────────────────────────

install_rust() {
    if ! command -v cargo >/dev/null 2>&1; then
        info "Rust not found — installing via rustup..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        # shellcheck source=/dev/null
        source "$HOME/.cargo/env"
    else
        info "Rust found: $(rustc --version)"
    fi
}

# ── Clone & build ───────────────────────────────────────────────────────────

build() {
    BUILD_DIR="$(mktemp -d)"
    trap 'rm -rf "$BUILD_DIR"' EXIT

    # If running from inside the repo, build in-place; otherwise clone
    if [ -f "Cargo.toml" ] && grep -q 'name = "open-bark"' Cargo.toml 2>/dev/null; then
        info "Building from local repository..."
        BUILD_DIR="$(pwd)"
        trap - EXIT  # don't delete the user's repo
    else
        info "Cloning $REPO_URL..."
        git clone --depth 1 "$REPO_URL" "$BUILD_DIR"
        cd "$BUILD_DIR"
    fi

    info "Building release binary... (this may take a few minutes)"
    # shellcheck disable=SC2086
    cargo build --release $CARGO_FEATURES
}

# ── Install binary ──────────────────────────────────────────────────────────

install_binary() {
    BINARY="target/release/$APP_NAME"
    [ -f "$BINARY" ] || error "Build artifact not found at $BINARY"

    info "Installing $APP_NAME to $INSTALL_DIR..."
    sudo install -m 755 "$BINARY" "$INSTALL_DIR/$APP_NAME"
    info "Installed: $(command -v $APP_NAME)"
}

# ── Register as service ─────────────────────────────────────────────────────

setup_service_macos() {
    PLIST_DIR="$HOME/Library/LaunchAgents"
    PLIST="$PLIST_DIR/com.jacobfreck.open-bark.plist"
    BINARY_PATH="$INSTALL_DIR/$APP_NAME"
    LOG_DIR="$HOME/Library/Logs/open-bark"

    mkdir -p "$PLIST_DIR" "$LOG_DIR"

    cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.jacobfreck.open-bark</string>

    <key>ProgramArguments</key>
    <array>
        <string>${BINARY_PATH}</string>
        <string>start</string>
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>StandardOutPath</key>
    <string>${LOG_DIR}/stdout.log</string>

    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/stderr.log</string>

    <key>ProcessType</key>
    <string>Interactive</string>
</dict>
</plist>
EOF

    # Unload first if already loaded (ignore errors)
    launchctl bootout "gui/$(id -u)/com.jacobfreck.open-bark" 2>/dev/null || true

    launchctl bootstrap "gui/$(id -u)" "$PLIST"

    info "Registered launchd service (runs at login)"
    info "  Logs: $LOG_DIR/"
    info "  Stop:    launchctl bootout gui/$(id -u)/com.jacobfreck.open-bark"
    info "  Restart: launchctl kickstart -k gui/$(id -u)/com.jacobfreck.open-bark"
}

setup_service_linux() {
    SERVICE_DIR="$HOME/.config/systemd/user"
    SERVICE="$SERVICE_DIR/open-bark.service"
    BINARY_PATH="$INSTALL_DIR/$APP_NAME"

    mkdir -p "$SERVICE_DIR"

    cat > "$SERVICE" <<EOF
[Unit]
Description=open-bark voice dictation
After=graphical-session.target

[Service]
Type=simple
ExecStart=${BINARY_PATH} start
Restart=on-failure
RestartSec=5
Environment=DISPLAY=:0

[Install]
WantedBy=default.target
EOF

    systemctl --user daemon-reload
    systemctl --user enable --now open-bark.service

    info "Registered systemd user service (runs at login)"
    info "  Status:  systemctl --user status open-bark"
    info "  Logs:    journalctl --user -u open-bark -f"
    info "  Stop:    systemctl --user stop open-bark"
    info "  Restart: systemctl --user restart open-bark"
}

setup_service() {
    info "Setting up $APP_NAME as a service..."
    if [ "$PLATFORM" = "macos" ]; then
        setup_service_macos
    else
        setup_service_linux
    fi
}

# ── Post-install notes ───────────────────────────────────────────────────────

print_post_install() {
    echo ""
    info "✅ open-bark installed and running!"
    echo ""
    echo "  The Whisper model will download automatically on first use."
    echo ""
    if [ "$PLATFORM" = "macos" ]; then
        echo "  ⚠  macOS permissions required:"
        echo "     • Accessibility: System Settings → Privacy & Security → Accessibility"
        echo "     • Microphone:    System Settings → Privacy & Security → Microphone"
        echo ""
    elif [ "$PLATFORM" = "linux" ]; then
        echo "  ⚠  Wayland users: add yourself to the 'input' group for hotkey support:"
        echo "     sudo usermod -aG input \$USER"
        echo ""
    fi
    echo "  Configure: $APP_NAME set-hotkey <key>"
    echo "  Uninstall: sudo rm $INSTALL_DIR/$APP_NAME"
    echo ""
}

# ── Main ─────────────────────────────────────────────────────────────────────

main() {
    echo ""
    echo "  ┌──────────────────────────────────┐"
    echo "  │  open-bark installer              │"
    echo "  │  Local voice dictation for all    │"
    echo "  └──────────────────────────────────┘"
    echo ""

    detect_platform
    choose_features
    install_dependencies
    install_rust
    require git
    require cmake
    build
    install_binary
    setup_service
    print_post_install
}

main "$@"
