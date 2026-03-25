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

# ── Colors & symbols ────────────────────────────────────────────────────────

BOLD="\033[1m"
DIM="\033[2m"
RESET="\033[0m"
RED="\033[1;31m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
BLUE="\033[1;34m"
CYAN="\033[1;36m"
WHITE="\033[1;37m"

SPINNER_PID=""
STEP_COUNT=0
TOTAL_STEPS=7

# ── Animated helpers ─────────────────────────────────────────────────────────

# Pulsing circle indicator
spinner_start() {
    local msg="$1"
    local frames=("○" "◎" "●" "◉" "●" "◎")
    local colors=("\033[34m" "\033[36m" "\033[1;36m" "\033[1;37m" "\033[1;36m" "\033[36m")

    (
        local i=0
        while true; do
            local idx=$((i % ${#frames[@]}))
            local frame="${frames[$idx]}"
            local color="${colors[$idx]}"
            printf "\r  ${color}${frame}${RESET} ${DIM}%s${RESET}  " "$msg"
            sleep 0.15
            i=$((i + 1))
        done
    ) &
    SPINNER_PID=$!
}

spinner_stop() {
    if [ -n "$SPINNER_PID" ]; then
        kill "$SPINNER_PID" 2>/dev/null
        wait "$SPINNER_PID" 2>/dev/null || true
        SPINNER_PID=""
        printf "\r\033[2K"
    fi
}

# Run a command with a spinner, show result
run_step() {
    local label="$1"
    shift

    STEP_COUNT=$((STEP_COUNT + 1))
    spinner_start "$label"

    local output
    local exit_code=0
    output=$("$@" 2>&1) || exit_code=$?

    spinner_stop

    if [ $exit_code -eq 0 ]; then
        printf "  ${GREEN}✔${RESET} %s\n" "$label"
    else
        printf "  ${RED}✖${RESET} %s\n" "$label"
        if [ -n "$output" ]; then
            echo ""
            printf "${DIM}%s${RESET}\n" "$output"
        fi
        exit 1
    fi
}

step_header() {
    local label="$1"
    STEP_COUNT=$((STEP_COUNT + 1))
    printf "\n  ${WHITE}[${STEP_COUNT}/${TOTAL_STEPS}]${RESET} ${BOLD}%s${RESET}\n" "$label"
}

info()  { printf "  ${BLUE}│${RESET} %s\n" "$1"; }
detail(){ printf "  ${DIM}│ %s${RESET}\n" "$1"; }
warn()  { printf "  ${YELLOW}⚠${RESET}  %s\n" "$1"; }
error() { spinner_stop; printf "\n  ${RED}✖ Error:${RESET} %s\n\n" "$1" >&2; exit 1; }

require() {
    command -v "$1" >/dev/null 2>&1 || error "'$1' is required but not found. Please install it first."
}

# Animated progress bar for long builds
progress_bar() {
    local label="$1"
    local log_file="$2"

    local bar_width=30
    local frames=("░" "▒" "▓" "█")
    local pulse_colors=("\033[34m" "\033[36m" "\033[35m" "\033[36m" "\033[34m")
    local i=0
    local crate_count=0
    local last_crate=""

    (
        while true; do
            # Count compiled crates from build log
            if [ -f "$log_file" ]; then
                crate_count=$(grep -c "Compiling\|Downloading" "$log_file" 2>/dev/null || echo "0")
                last_crate=$(grep "Compiling" "$log_file" 2>/dev/null | tail -1 | sed 's/.*Compiling //' | sed 's/ v.*//' || echo "")
            fi

            # Build animated bar
            local bar=""
            for j in $(seq 0 $((bar_width - 1))); do
                local offset=$(( (i + j) % 20 ))
                if [ $offset -lt 5 ]; then
                    local ci=$(( (i + j) / 3 % ${#pulse_colors[@]} ))
                    bar="${bar}${pulse_colors[$ci]}${frames[$((offset % 4))]}${RESET}"
                else
                    bar="${bar}${DIM}─${RESET}"
                fi
            done

            local status_text="$label"
            if [ -n "$last_crate" ]; then
                status_text="Compiling ${last_crate}"
            fi
            # Truncate status to 35 chars
            if [ ${#status_text} -gt 35 ]; then
                status_text="${status_text:0:32}..."
            fi

            printf "\r  ${CYAN}⟫${RESET} ${bar} ${DIM}%s${RESET} ${DIM}(%d crates)${RESET}    " \
                "$status_text" "$crate_count"

            sleep 0.1
            i=$((i + 1))
        done
    ) &
    SPINNER_PID=$!
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
}

# ── Choose build features ───────────────────────────────────────────────────

choose_features() {
    CARGO_FEATURES=""
    GPU_LABEL="CPU-only"

    if [ "$PLATFORM" = "macos" ] && [ "$ARCH_LABEL" = "arm64" ]; then
        CARGO_FEATURES="--features metal"
        GPU_LABEL="Metal"
    elif [ "$PLATFORM" = "linux" ]; then
        if command -v nvcc >/dev/null 2>&1; then
            CARGO_FEATURES="--features cuda"
            GPU_LABEL="CUDA"
        elif command -v vulkaninfo >/dev/null 2>&1; then
            CARGO_FEATURES="--features vulkan"
            GPU_LABEL="Vulkan"
        fi
    fi
}

# ── Install system dependencies ──────────────────────────────────────────────

install_dependencies() {
    if [ "$PLATFORM" = "linux" ]; then
        if command -v apt-get >/dev/null 2>&1; then
            run_step "Installing system libraries (apt)" \
                sudo apt-get install -y -qq build-essential cmake libasound2-dev \
                    libgtk-3-dev libayatana-appindicator3-dev pkg-config
        elif command -v dnf >/dev/null 2>&1; then
            run_step "Installing system libraries (dnf)" \
                sudo dnf install -y -q gcc gcc-c++ cmake alsa-lib-devel \
                    gtk3-devel libayatana-appindicator-gtk3-devel pkg-config
        elif command -v pacman >/dev/null 2>&1; then
            run_step "Installing system libraries (pacman)" \
                sudo pacman -Sy --noconfirm --quiet base-devel cmake alsa-lib \
                    gtk3 libayatana-appindicator pkg-config
        else
            warn "Unknown package manager — ensure build deps are installed manually"
        fi
    fi
}

# ── Install Rust (if needed) ────────────────────────────────────────────────

install_rust() {
    if ! command -v cargo >/dev/null 2>&1; then
        run_step "Installing Rust via rustup" \
            bash -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'
        # shellcheck source=/dev/null
        source "$HOME/.cargo/env"
    fi
}

# ── Clone & build ───────────────────────────────────────────────────────────

build() {
    BUILD_DIR="$(mktemp -d)"
    trap 'rm -rf "$BUILD_DIR"' EXIT

    if [ -f "Cargo.toml" ] && grep -q 'name = "open-bark"' Cargo.toml 2>/dev/null; then
        BUILD_DIR="$(pwd)"
        trap - EXIT
        detail "Building from local repository"
    else
        run_step "Cloning repository" \
            git clone --depth 1 "$REPO_URL" "$BUILD_DIR"
        cd "$BUILD_DIR"
    fi

    # Build with animated progress
    local log_file
    log_file="$(mktemp)"

    progress_bar "Building release" "$log_file"

    local exit_code=0
    # shellcheck disable=SC2086
    cargo build --release $CARGO_FEATURES >"$log_file" 2>&1 || exit_code=$?

    spinner_stop
    rm -f "$log_file"

    if [ $exit_code -eq 0 ]; then
        printf "\r  ${GREEN}✔${RESET} Build complete\n"
    else
        printf "\r  ${RED}✖${RESET} Build failed\n"
        exit 1
    fi
}

# ── Install binary ──────────────────────────────────────────────────────────

install_binary() {
    BINARY="target/release/$APP_NAME"
    [ -f "$BINARY" ] || error "Build artifact not found at $BINARY"

    run_step "Installing binary to $INSTALL_DIR" \
        sudo install -m 755 "$BINARY" "$INSTALL_DIR/$APP_NAME"
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

    launchctl bootout "gui/$(id -u)/com.jacobfreck.open-bark" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST"
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
}

setup_service() {
    if [ "$PLATFORM" = "macos" ]; then
        run_step "Registering launchd service" setup_service_macos
    else
        run_step "Registering systemd service" setup_service_linux
    fi
}

# ── Post-install ─────────────────────────────────────────────────────────────

print_banner() {
    echo ""
    printf "  ${CYAN}┌──────────────────────────────────────┐${RESET}\n"
    printf "  ${CYAN}│${RESET}  ${BOLD}🐕 open-bark installer${RESET}               ${CYAN}│${RESET}\n"
    printf "  ${CYAN}│${RESET}  ${DIM}Local voice dictation for everyone${RESET}   ${CYAN}│${RESET}\n"
    printf "  ${CYAN}└──────────────────────────────────────┘${RESET}\n"
    echo ""
}

print_summary() {
    local rust_ver
    rust_ver="$(rustc --version 2>/dev/null | cut -d' ' -f2 || echo "?")"

    echo ""
    printf "  ${CYAN}┌──────────────────────────────────────┐${RESET}\n"
    printf "  ${CYAN}│${RESET}                                      ${CYAN}│${RESET}\n"
    printf "  ${CYAN}│${RESET}  ${GREEN}${BOLD}✔ open-bark installed successfully!${RESET}  ${CYAN}│${RESET}\n"
    printf "  ${CYAN}│${RESET}                                      ${CYAN}│${RESET}\n"
    printf "  ${CYAN}└──────────────────────────────────────┘${RESET}\n"
    echo ""
    printf "  ${DIM}────────────────────────────────────────${RESET}\n"
    printf "  ${WHITE}Platform${RESET}   %s (%s)\n" "$PLATFORM" "$ARCH_LABEL"
    printf "  ${WHITE}GPU${RESET}        %s\n" "$GPU_LABEL"
    printf "  ${WHITE}Rust${RESET}       %s\n" "$rust_ver"
    printf "  ${WHITE}Binary${RESET}     %s\n" "$INSTALL_DIR/$APP_NAME"
    printf "  ${DIM}────────────────────────────────────────${RESET}\n"
    echo ""

    if [ "$PLATFORM" = "macos" ]; then
        printf "  ${YELLOW}⚠${RESET}  ${BOLD}macOS permissions required:${RESET}\n"
        printf "     ${DIM}•${RESET} Accessibility: ${DIM}System Settings → Privacy → Accessibility${RESET}\n"
        printf "     ${DIM}•${RESET} Microphone:    ${DIM}System Settings → Privacy → Microphone${RESET}\n"
        echo ""
        printf "  ${DIM}Manage service:${RESET}\n"
        printf "     ${DIM}Stop:${RESET}    launchctl bootout gui/$(id -u)/com.jacobfreck.open-bark\n"
        printf "     ${DIM}Restart:${RESET} launchctl kickstart -k gui/$(id -u)/com.jacobfreck.open-bark\n"
        printf "     ${DIM}Logs:${RESET}    ~/Library/Logs/open-bark/\n"
    else
        printf "  ${YELLOW}⚠${RESET}  ${BOLD}Wayland users:${RESET} add yourself to the 'input' group:\n"
        printf "     sudo usermod -aG input \$USER\n"
        echo ""
        printf "  ${DIM}Manage service:${RESET}\n"
        printf "     ${DIM}Status:${RESET}  systemctl --user status open-bark\n"
        printf "     ${DIM}Logs:${RESET}    journalctl --user -u open-bark -f\n"
        printf "     ${DIM}Restart:${RESET} systemctl --user restart open-bark\n"
    fi

    echo ""
    printf "  ${DIM}Configure:${RESET} open-bark set-hotkey <key>\n"
    printf "  ${DIM}Uninstall:${RESET} sudo rm %s/%s\n" "$INSTALL_DIR" "$APP_NAME"
    echo ""
}

# ── Main ─────────────────────────────────────────────────────────────────────

cleanup() {
    spinner_stop
}
trap cleanup EXIT

main() {
    print_banner

    step_header "Detecting platform"
    detect_platform
    choose_features
    info "$PLATFORM ($ARCH_LABEL) · GPU: $GPU_LABEL"

    step_header "Checking prerequisites"
    install_rust
    require git
    require cmake
    printf "  ${GREEN}✔${RESET} All prerequisites met\n"

    step_header "Installing dependencies"
    install_dependencies
    if [ "$PLATFORM" = "macos" ]; then
        printf "  ${GREEN}✔${RESET} Xcode tools provide everything needed\n"
    fi

    step_header "Building open-bark"
    build

    step_header "Installing binary"
    install_binary

    step_header "Configuring service"
    setup_service

    step_header "Done"
    print_summary
}

main "$@"
