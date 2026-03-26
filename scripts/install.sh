#!/usr/bin/env bash
#
# murmur install script
#
# Downloads a pre-built binary from GitHub releases, installs it, and
# registers murmur as a user-level service (launchd/systemd).
#
# No build tools required — just curl.
#
# Usage:
#   curl -sSf https://raw.githubusercontent.com/jacobfreck/murmur/main/scripts/install.sh | bash
#   # or from a local clone:
#   ./scripts/install.sh
#   # specific version:
#   OPEN_BARK_VERSION=v0.1.0 ./scripts/install.sh

set -euo pipefail

REPO="jacobfreck/murmur"
INSTALL_DIR="/usr/local/bin"
APP_NAME="murmur"

# ── Colors ───────────────────────────────────────────────────────────────────

BOLD="\033[1m"
DIM="\033[2m"
RESET="\033[0m"
RED="\033[1;31m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
CYAN="\033[1;36m"
WHITE="\033[1;37m"

SPINNER_PID=""
STEP_COUNT=0
TOTAL_STEPS=5

# ── Animated helpers ─────────────────────────────────────────────────────────

spinner_start() {
    local msg="$1"
    local frames=("○" "◎" "●" "◉" "●" "◎")
    local colors=("\033[34m" "\033[36m" "\033[1;36m" "\033[1;37m" "\033[1;36m" "\033[36m")

    (
        local i=0
        while true; do
            local idx=$((i % ${#frames[@]}))
            printf "\r  ${colors[$idx]}${frames[$idx]}${RESET} ${DIM}%s${RESET}  " "$msg"
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

run_step() {
    local label="$1"; shift
    spinner_start "$label"
    local output exit_code=0
    output=$("$@" 2>&1) || exit_code=$?
    spinner_stop
    if [ $exit_code -eq 0 ]; then
        printf "  ${GREEN}✔${RESET} %s\n" "$label"
    else
        printf "  ${RED}✖${RESET} %s\n" "$label"
        [ -n "$output" ] && printf "\n${DIM}%s${RESET}\n" "$output"
        exit 1
    fi
}

step_header() {
    STEP_COUNT=$((STEP_COUNT + 1))
    printf "\n  ${WHITE}[${STEP_COUNT}/${TOTAL_STEPS}]${RESET} ${BOLD}%s${RESET}\n" "$1"
}

info()  { printf "  ${CYAN}│${RESET} %s\n" "$1"; }
warn()  { printf "  ${YELLOW}⚠${RESET}  %s\n" "$1"; }
error() { spinner_stop; printf "\n  ${RED}✖ Error:${RESET} %s\n\n" "$1" >&2; exit 1; }

# ── Detect platform ─────────────────────────────────────────────────────────

detect_platform() {
    OS="$(uname -s)"
    ARCH="$(uname -m)"

    case "$OS" in
        Darwin) PLATFORM="macos" ;;
        Linux)  PLATFORM="linux" ;;
        *)      error "Unsupported OS: $OS. Use install.ps1 for Windows." ;;
    esac

    case "$ARCH" in
        arm64|aarch64) ARCH_LABEL="arm64" ;;
        x86_64)        ARCH_LABEL="x86_64" ;;
        *)             error "Unsupported architecture: $ARCH" ;;
    esac

    if [ "$PLATFORM" = "macos" ] && [ "$ARCH_LABEL" = "arm64" ]; then
        ARTIFACT="murmur-darwin-arm64"
        GPU_LABEL="Metal"
    elif [ "$PLATFORM" = "macos" ]; then
        ARTIFACT="murmur-darwin-x86_64"
        GPU_LABEL="CPU"
    else
        ARTIFACT="murmur-linux-x86_64"
        GPU_LABEL="CPU"
    fi
}

# ── Resolve version ─────────────────────────────────────────────────────────

resolve_version() {
    VERSION="${OPEN_BARK_VERSION:-latest}"

    if [ "$VERSION" = "latest" ]; then
        VERSION=$(curl -sI "https://github.com/$REPO/releases/latest" \
            | grep -i '^location:' \
            | sed 's|.*/tag/||' \
            | tr -d '\r\n')
        [ -n "$VERSION" ] || error "Could not determine latest release. Set OPEN_BARK_VERSION=v0.1.0 to specify."
    fi
}

# ── Download ─────────────────────────────────────────────────────────────────

download_binary() {
    local url="https://github.com/$REPO/releases/download/${VERSION}/${ARTIFACT}.tar.gz"
    TMP_DIR="$(mktemp -d)"

    curl -fsSL "$url" -o "$TMP_DIR/$ARTIFACT.tar.gz" \
        || error "Download failed.\n         URL: $url\n         Does release $VERSION have artifact $ARTIFACT?"

    tar xzf "$TMP_DIR/$ARTIFACT.tar.gz" -C "$TMP_DIR"
    chmod +x "$TMP_DIR/$ARTIFACT"
}

# ── Install ──────────────────────────────────────────────────────────────────

install_binary() {
    sudo install -m 755 "$TMP_DIR/$ARTIFACT" "$INSTALL_DIR/$APP_NAME"
    rm -rf "$TMP_DIR"
}

# ── Service setup ────────────────────────────────────────────────────────────

setup_service() {
    if [ "$PLATFORM" = "macos" ]; then
        setup_service_macos
    else
        setup_service_linux
    fi
}

setup_service_macos() {
    local plist_dir="$HOME/Library/LaunchAgents"
    local plist="$plist_dir/com.jacobfreck.murmur.plist"
    local bin="$INSTALL_DIR/$APP_NAME"
    local log_dir="$HOME/Library/Logs/murmur"

    mkdir -p "$plist_dir" "$log_dir"

    cat > "$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.jacobfreck.murmur</string>
    <key>ProgramArguments</key>
    <array>
        <string>${bin}</string>
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
    <string>${log_dir}/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${log_dir}/stderr.log</string>
    <key>ProcessType</key>
    <string>Interactive</string>
</dict>
</plist>
EOF

    launchctl bootout "gui/$(id -u)/com.jacobfreck.murmur" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$plist"
}

setup_service_linux() {
    local svc_dir="$HOME/.config/systemd/user"
    local svc="$svc_dir/murmur.service"
    local bin="$INSTALL_DIR/$APP_NAME"

    mkdir -p "$svc_dir"

    cat > "$svc" <<EOF
[Unit]
Description=murmur voice dictation
After=graphical-session.target

[Service]
Type=simple
ExecStart=${bin} start
Restart=on-failure
RestartSec=5
Environment=DISPLAY=:0

[Install]
WantedBy=default.target
EOF

    systemctl --user daemon-reload
    systemctl --user enable --now murmur.service
}

# ── Output ───────────────────────────────────────────────────────────────────

print_banner() {
    echo ""
    printf "  ${CYAN}┌──────────────────────────────────────┐${RESET}\n"
    printf "  ${CYAN}│${RESET}  ${BOLD}🐕 murmur installer${RESET}               ${CYAN}│${RESET}\n"
    printf "  ${CYAN}│${RESET}  ${DIM}Local voice dictation for everyone${RESET}   ${CYAN}│${RESET}\n"
    printf "  ${CYAN}└──────────────────────────────────────┘${RESET}\n"
    echo ""
}

print_summary() {
    echo ""
    printf "  ${CYAN}┌──────────────────────────────────────┐${RESET}\n"
    printf "  ${CYAN}│${RESET}                                      ${CYAN}│${RESET}\n"
    printf "  ${CYAN}│${RESET}  ${GREEN}${BOLD}✔ murmur installed successfully!${RESET}  ${CYAN}│${RESET}\n"
    printf "  ${CYAN}│${RESET}                                      ${CYAN}│${RESET}\n"
    printf "  ${CYAN}└──────────────────────────────────────┘${RESET}\n"
    echo ""
    printf "  ${DIM}────────────────────────────────────────${RESET}\n"
    printf "  ${WHITE}Platform${RESET}   %s (%s)\n" "$PLATFORM" "$ARCH_LABEL"
    printf "  ${WHITE}GPU${RESET}        %s\n" "$GPU_LABEL"
    printf "  ${WHITE}Version${RESET}    %s\n" "$VERSION"
    printf "  ${WHITE}Binary${RESET}     %s\n" "$INSTALL_DIR/$APP_NAME"
    printf "  ${DIM}────────────────────────────────────────${RESET}\n"
    echo ""

    if [ "$PLATFORM" = "macos" ]; then
        printf "  ${YELLOW}⚠${RESET}  ${BOLD}macOS permissions required:${RESET}\n"
        printf "     ${DIM}•${RESET} Accessibility: ${DIM}System Settings → Privacy → Accessibility${RESET}\n"
        printf "     ${DIM}•${RESET} Microphone:    ${DIM}System Settings → Privacy → Microphone${RESET}\n"
        echo ""
        printf "  ${DIM}Manage service:${RESET}\n"
        printf "     ${DIM}Stop:${RESET}    launchctl bootout gui/$(id -u)/com.jacobfreck.murmur\n"
        printf "     ${DIM}Restart:${RESET} launchctl kickstart -k gui/$(id -u)/com.jacobfreck.murmur\n"
        printf "     ${DIM}Logs:${RESET}    ~/Library/Logs/murmur/\n"
    else
        printf "  ${YELLOW}⚠${RESET}  ${BOLD}Wayland users:${RESET} add yourself to the 'input' group:\n"
        printf "     sudo usermod -aG input \$USER\n"
        echo ""
        printf "  ${DIM}Manage service:${RESET}\n"
        printf "     ${DIM}Status:${RESET}  systemctl --user status murmur\n"
        printf "     ${DIM}Logs:${RESET}    journalctl --user -u murmur -f\n"
        printf "     ${DIM}Restart:${RESET} systemctl --user restart murmur\n"
    fi

    echo ""
    printf "  ${DIM}Configure:${RESET} murmur set-hotkey <key>\n"
    printf "  ${DIM}Uninstall:${RESET} sudo rm %s/%s\n" "$INSTALL_DIR" "$APP_NAME"
    echo ""
}

# ── Main ─────────────────────────────────────────────────────────────────────

cleanup() { spinner_stop; }
trap cleanup EXIT

main() {
    print_banner

    step_header "Detecting platform"
    detect_platform
    info "$PLATFORM ($ARCH_LABEL) · GPU: $GPU_LABEL"

    step_header "Resolving version"
    run_step "Finding latest release" resolve_version
    info "Version: $VERSION"

    step_header "Downloading"
    run_step "Downloading $ARTIFACT ($VERSION)" download_binary

    step_header "Installing"
    run_step "Installing to $INSTALL_DIR" install_binary

    step_header "Configuring service"
    run_step "Registering service" setup_service

    print_summary
}

main "$@"
