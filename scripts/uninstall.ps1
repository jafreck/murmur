#Requires -Version 5.1
<#
.SYNOPSIS
    murmur uninstaller for Windows.

.DESCRIPTION
    Stops murmur, removes the binary, startup shortcut, PATH entry,
    and optionally the user config and models.

.EXAMPLE
    irm https://github.com/jafreck/murmur/releases/latest/download/uninstall.ps1 | iex

.EXAMPLE
    .\scripts\uninstall.ps1
#>
param(
    [string]$InstallDir = "$env:LOCALAPPDATA\murmur"
)

$ErrorActionPreference = "Stop"
$AppName = "murmur"

function Write-Ok   { param([string]$Msg) Write-Host "  ✔ " -ForegroundColor Green -NoNewline; Write-Host $Msg }
function Write-Skip { param([string]$Msg) Write-Host "  ─ " -ForegroundColor DarkGray -NoNewline; Write-Host "$Msg (not found)" -ForegroundColor DarkGray }
function Write-Info { param([string]$Msg) Write-Host "  │ " -ForegroundColor Cyan -NoNewline; Write-Host $Msg }

# ── Banner ───────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "  ┌──────────────────────────────────────┐" -ForegroundColor Cyan
Write-Host "  │  " -ForegroundColor Cyan -NoNewline
Write-Host "🐕 murmur uninstaller" -NoNewline
Write-Host "             │" -ForegroundColor Cyan
Write-Host "  └──────────────────────────────────────┘" -ForegroundColor Cyan
Write-Host ""

# ── Stop running process ─────────────────────────────────────────────────────

$procs = Get-Process -Name $AppName -ErrorAction SilentlyContinue
if ($procs) {
    $procs | Stop-Process -Force
    Write-Ok "Stopped running murmur process"
} else {
    Write-Skip "Running process"
}

# ── Remove startup shortcut ──────────────────────────────────────────────────

$startupDir = [Environment]::GetFolderPath("Startup")
$shortcut = Join-Path $startupDir "murmur.lnk"
if (Test-Path $shortcut) {
    Remove-Item $shortcut -Force
    Write-Ok "Removed startup shortcut"
} else {
    Write-Skip "Startup shortcut"
}

# ── Remove binary & install directory ────────────────────────────────────────

if (Test-Path $InstallDir) {
    Remove-Item $InstallDir -Recurse -Force
    Write-Ok "Removed $InstallDir"
} else {
    Write-Skip "Install directory ($InstallDir)"
}

# ── Remove from PATH ────────────────────────────────────────────────────────

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -like "*$InstallDir*") {
    $newPath = ($userPath -split ";" | Where-Object { $_ -ne $InstallDir }) -join ";"
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Ok "Removed from PATH"
} else {
    Write-Skip "PATH entry"
}

# ── User data ────────────────────────────────────────────────────────────────

$configDir = Join-Path $env:APPDATA "murmur"
Write-Host ""
if (Test-Path $configDir) {
    Write-Host "  ? " -ForegroundColor Yellow -NoNewline
    Write-Host "Config & models found at:"
    Write-Host "     $configDir" -ForegroundColor DarkGray
    Write-Host ""

    $answer = Read-Host "  Remove config and downloaded models? [y/N]"
    if ($answer -eq "y" -or $answer -eq "Y") {
        Remove-Item $configDir -Recurse -Force
        Write-Ok "Removed $configDir"
    } else {
        Write-Info "Kept $configDir"
    }
} else {
    Write-Skip "Config directory"
}

# ── Done ─────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "  ✔ murmur uninstalled." -ForegroundColor Green
Write-Host ""
