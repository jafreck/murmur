#Requires -Version 5.1
<#
.SYNOPSIS
    murmur installer for Windows.

.DESCRIPTION
    Downloads a pre-built binary from GitHub releases, installs it, and
    registers murmur as a startup task. No build tools required.

.EXAMPLE
    irm https://raw.githubusercontent.com/jacobfreck/murmur/main/scripts/install.ps1 | iex

.EXAMPLE
    .\scripts\install.ps1
    .\scripts\install.ps1 -Version v0.1.0
#>
param(
    [string]$Version = "latest",
    [string]$InstallDir = "$env:LOCALAPPDATA\murmur"
)

$ErrorActionPreference = "Stop"
$Repo = "jacobfreck/murmur"
$AppName = "murmur"
$Artifact = "murmur-windows-x86_64"

# ── Colors & helpers ─────────────────────────────────────────────────────────

function Write-Banner {
    Write-Host ""
    Write-Host "  ┌──────────────────────────────────────┐" -ForegroundColor Cyan
    Write-Host "  │  " -ForegroundColor Cyan -NoNewline
    Write-Host "🐕 murmur installer" -NoNewline
    Write-Host "               │" -ForegroundColor Cyan
    Write-Host "  │  " -ForegroundColor Cyan -NoNewline
    Write-Host "Local voice dictation for everyone" -ForegroundColor DarkGray -NoNewline
    Write-Host "   │" -ForegroundColor Cyan
    Write-Host "  └──────────────────────────────────────┘" -ForegroundColor Cyan
    Write-Host ""
}

$script:StepCount = 0
$TotalSteps = 4

function Write-Step {
    param([string]$Label)
    $script:StepCount++
    Write-Host ""
    Write-Host "  [$script:StepCount/$TotalSteps] " -ForegroundColor White -NoNewline
    Write-Host "$Label" -ForegroundColor White
}

function Write-Info {
    param([string]$Message)
    Write-Host "  │ " -ForegroundColor Cyan -NoNewline
    Write-Host "$Message"
}

function Write-Success {
    param([string]$Message)
    Write-Host "  ✔ " -ForegroundColor Green -NoNewline
    Write-Host "$Message"
}

function Write-Fail {
    param([string]$Message)
    Write-Host "  ✖ " -ForegroundColor Red -NoNewline
    Write-Host "$Message"
}

function Invoke-WithSpinner {
    param(
        [string]$Label,
        [scriptblock]$Action
    )

    $frames = @("○", "◎", "●", "◉", "●", "◎")
    $colors = @("Blue", "Cyan", "Cyan", "White", "Cyan", "Blue")

    $job = Start-Job -ScriptBlock $Action

    $i = 0
    while ($job.State -eq "Running") {
        $idx = $i % $frames.Count
        Write-Host "`r  " -NoNewline
        Write-Host "$($frames[$idx]) " -ForegroundColor $colors[$idx] -NoNewline
        Write-Host "$Label  " -ForegroundColor DarkGray -NoNewline
        Start-Sleep -Milliseconds 150
        $i++
    }

    $result = Receive-Job $job -ErrorAction SilentlyContinue
    $failed = $job.State -eq "Failed"
    Remove-Job $job -Force

    Write-Host "`r                                                          `r" -NoNewline

    if ($failed) {
        Write-Fail $Label
        throw "Step failed: $Label"
    } else {
        Write-Success $Label
    }

    return $result
}

# ── Resolve version ─────────────────────────────────────────────────────────

function Resolve-LatestVersion {
    if ($Version -eq "latest") {
        $response = Invoke-WebRequest -Uri "https://github.com/$Repo/releases/latest" `
            -MaximumRedirection 0 -ErrorAction SilentlyContinue -UseBasicParsing 2>$null
        if ($response.Headers.Location) {
            $location = $response.Headers.Location
        } else {
            $response = Invoke-WebRequest -Uri "https://github.com/$Repo/releases/latest" `
                -UseBasicParsing -ErrorAction Stop
            $location = $response.BaseResponse.ResponseUri.AbsoluteUri
        }
        $script:Version = ($location -split "/tag/")[-1].Trim()
        if (-not $script:Version) {
            throw "Could not determine latest release. Use -Version v0.1.0 to specify."
        }
    }
}

# ── Download ─────────────────────────────────────────────────────────────────

function Get-Binary {
    $url = "https://github.com/$Repo/releases/download/$Version/$Artifact.zip"
    $tmpDir = Join-Path $env:TEMP "murmur-install"

    if (Test-Path $tmpDir) { Remove-Item $tmpDir -Recurse -Force }
    New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null

    Invoke-WebRequest -Uri $url -OutFile "$tmpDir\$Artifact.zip" -UseBasicParsing -ErrorAction Stop
    Expand-Archive -Path "$tmpDir\$Artifact.zip" -DestinationPath $tmpDir -Force

    $script:DownloadedBinary = "$tmpDir\$Artifact.exe"

    if (-not (Test-Path $script:DownloadedBinary)) {
        throw "Binary not found after extraction at $($script:DownloadedBinary)"
    }
}

# ── Install ──────────────────────────────────────────────────────────────────

function Install-Binary {
    if (-not (Test-Path $InstallDir)) {
        New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    }

    Copy-Item $script:DownloadedBinary "$InstallDir\$AppName.exe" -Force

    # Add to PATH if not already there
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($userPath -notlike "*$InstallDir*") {
        [Environment]::SetEnvironmentVariable("Path", "$userPath;$InstallDir", "User")
    }

    # Clean up temp files
    $tmpDir = Split-Path $script:DownloadedBinary
    Remove-Item $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
}

# ── Startup task ─────────────────────────────────────────────────────────────

function Register-StartupTask {
    $startupDir = [Environment]::GetFolderPath("Startup")
    $shortcutPath = Join-Path $startupDir "murmur.lnk"
    $targetPath = Join-Path $InstallDir "$AppName.exe"

    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $targetPath
    $shortcut.Arguments = "start"
    $shortcut.WindowStyle = 7  # Minimized
    $shortcut.Description = "murmur voice dictation"
    $shortcut.Save()
}

# ── Summary ──────────────────────────────────────────────────────────────────

function Write-Summary {
    Write-Host ""
    Write-Host "  ┌──────────────────────────────────────┐" -ForegroundColor Cyan
    Write-Host "  │                                      │" -ForegroundColor Cyan
    Write-Host "  │  " -ForegroundColor Cyan -NoNewline
    Write-Host "✔ murmur installed successfully!" -ForegroundColor Green -NoNewline
    Write-Host "  │" -ForegroundColor Cyan
    Write-Host "  │                                      │" -ForegroundColor Cyan
    Write-Host "  └──────────────────────────────────────┘" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  ────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host "  Platform   " -ForegroundColor White -NoNewline
    Write-Host "Windows (x86_64)"
    Write-Host "  Version    " -ForegroundColor White -NoNewline
    Write-Host "$Version"
    Write-Host "  Binary     " -ForegroundColor White -NoNewline
    Write-Host "$InstallDir\$AppName.exe"
    Write-Host "  ────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  ⚠ " -ForegroundColor Yellow -NoNewline
    Write-Host "Your antivirus may flag the keyboard hook — allow it." -ForegroundColor White
    Write-Host ""
    Write-Host "  Manage:" -ForegroundColor DarkGray
    Write-Host "     Start:     " -ForegroundColor DarkGray -NoNewline
    Write-Host "murmur start"
    Write-Host "     Startup:   " -ForegroundColor DarkGray -NoNewline
    Write-Host "shell:startup (remove shortcut to disable)"
    Write-Host "     Configure: " -ForegroundColor DarkGray -NoNewline
    Write-Host "murmur set-hotkey <key>"
    Write-Host "     Uninstall: " -ForegroundColor DarkGray -NoNewline
    Write-Host "Remove-Item `"$InstallDir`" -Recurse"
    Write-Host ""
}

# ── Main ─────────────────────────────────────────────────────────────────────

Write-Banner

Write-Step "Detecting platform"
Write-Info "Windows (x86_64) · CPU"

Write-Step "Resolving version"
Invoke-WithSpinner "Finding latest release" { Resolve-LatestVersion }
# Re-resolve in main scope since job runs in a child scope
Resolve-LatestVersion
Write-Info "Version: $Version"

Write-Step "Downloading & installing"
Invoke-WithSpinner "Downloading $Artifact ($Version)" {
    param()
    $url = "https://github.com/$using:Repo/releases/download/$using:Version/$using:Artifact.zip"
    $tmpDir = Join-Path $env:TEMP "murmur-install"
    if (Test-Path $tmpDir) { Remove-Item $tmpDir -Recurse -Force }
    New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null
    Invoke-WebRequest -Uri $url -OutFile "$tmpDir\$using:Artifact.zip" -UseBasicParsing
    Expand-Archive -Path "$tmpDir\$using:Artifact.zip" -DestinationPath $tmpDir -Force
}

# Install in main scope (needs filesystem access + env vars)
$tmpDir = Join-Path $env:TEMP "murmur-install"
$script:DownloadedBinary = "$tmpDir\$Artifact.exe"
Install-Binary
Write-Success "Installed to $InstallDir"

Write-Step "Configuring startup"
Invoke-WithSpinner "Registering startup shortcut" {
    $startupDir = [Environment]::GetFolderPath("Startup")
    $shortcutPath = Join-Path $startupDir "murmur.lnk"
    $targetPath = Join-Path $using:InstallDir "$using:AppName.exe"
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $targetPath
    $shortcut.Arguments = "start"
    $shortcut.WindowStyle = 7
    $shortcut.Description = "murmur voice dictation"
    $shortcut.Save()
}

Write-Summary
