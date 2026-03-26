#!/usr/bin/env bash
#
# murmur uninstall script
#
# Stops the service, removes the binary, service config, logs, and
# optionally the user config/models.
#
# Usage:
#   curl -sSfL https://github.com/jafreck/murmur/releases/latest/download/uninstall.sh | bash
#   # or from a local clone:
#   ./scripts/uninstall.sh

set -euo pipefail

APP_NAME="murmur"
INSTALL_DIR="/usr/local/bin"

# ── Colors ───────────────────────────────────────────────────────────────────

BOLD="\033[1m"
DIM="\033[2m"
RESET="\033[0m"
RED="\033[1;31m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
CYAN="\033[1;36m"
WHITE="\033[1;37m"

info()  { printf "  ${CYAN}│${RESET} %s\n" "$1"; }
warn()  { printf "  ${YELLOW}⚠${RESET}  %s\n" "$1"; }
ok()    { printf "  ${GREEN}✔${RESET} %s\n" "$1"; }
skip()  { printf "  ${DIM}─${RESET} %s ${DIM}(not found)${RESET}\n" "$1"; }

# ── Detect platform ─────────────────────────────────────────────────────────

OS="$(uname -s)"
case "$OS" in
    Darwin) PLATFORM="macos" ;;
    Linux)  PLATFORM="linux" ;;
    *)      printf "  ${RED}✖${RESET} Unsupported OS: %s. Use uninstall.ps1 for Windows.\n" "$OS"; exit 1 ;;
esac

# ── Banner ───────────────────────────────────────────────────────────────────

echo ""
printf "  ${CYAN}┌──────────────────────────────────────┐${RESET}\n"
printf "  ${CYAN}│${RESET}  ${BOLD}🐕 murmur uninstaller${RESET}             ${CYAN}│${RESET}\n"
printf "  ${CYAN}└──────────────────────────────────────┘${RESET}\n"
echo ""

# ── Stop & remove service ────────────────────────────────────────────────────

if [ "$PLATFORM" = "macos" ]; then
    PLIST="$HOME/Library/LaunchAgents/com.jafreck.murmur.plist"
    LOG_DIR="$HOME/Library/Logs/murmur"
    CONFIG_DIR="$HOME/Library/Application Support/murmur"

    if launchctl print "gui/$(id -u)/com.jafreck.murmur" &>/dev/null; then
        launchctl bootout "gui/$(id -u)/com.jafreck.murmur" 2>/dev/null || true
        ok "Stopped launchd service"
    else
        skip "Launchd service"
    fi

    if [ -f "$PLIST" ]; then
        rm -f "$PLIST"
        ok "Removed $PLIST"
    else
        skip "Launchd plist"
    fi

    if [ -d "$LOG_DIR" ]; then
        rm -rf "$LOG_DIR"
        ok "Removed $LOG_DIR"
    else
        skip "Log directory"
    fi
else
    SVC="$HOME/.config/systemd/user/murmur.service"
    CONFIG_DIR="$HOME/.config/murmur"

    if systemctl --user is-active murmur.service &>/dev/null; then
        systemctl --user stop murmur.service
        ok "Stopped systemd service"
    else
        skip "Systemd service (not running)"
    fi

    if systemctl --user is-enabled murmur.service &>/dev/null; then
        systemctl --user disable murmur.service 2>/dev/null || true
        ok "Disabled systemd service"
    fi

    if [ -f "$SVC" ]; then
        rm -f "$SVC"
        systemctl --user daemon-reload
        ok "Removed $SVC"
    else
        skip "Systemd unit file"
    fi
fi

# ── Remove binary ────────────────────────────────────────────────────────────

BIN="$INSTALL_DIR/$APP_NAME"
if [ -f "$BIN" ]; then
    sudo rm -f "$BIN"
    ok "Removed $BIN"
else
    skip "Binary ($BIN)"
fi

# ── User data ────────────────────────────────────────────────────────────────

echo ""
if [ -d "$CONFIG_DIR" ]; then
    printf "  ${YELLOW}?${RESET}  Config & models found at:\n"
    printf "     ${DIM}%s${RESET}\n" "$CONFIG_DIR"
    echo ""

    if [ -t 0 ]; then
        printf "  Remove config and downloaded models? [y/N] "
        read -r answer
        if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
            rm -rf "$CONFIG_DIR"
            ok "Removed $CONFIG_DIR"
        else
            info "Kept $CONFIG_DIR"
        fi
    else
        warn "Run interactively to remove config, or delete manually:"
        printf "     rm -rf \"%s\"\n" "$CONFIG_DIR"
    fi
else
    skip "Config directory"
fi

# ── Done ─────────────────────────────────────────────────────────────────────

echo ""
printf "  ${GREEN}${BOLD}✔ murmur uninstalled.${RESET}\n"
echo ""
