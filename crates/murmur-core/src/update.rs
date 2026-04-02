//! Self-update: check GitHub Releases for newer versions, download and replace
//! the running binary in-place.
//!
//! Replacing at the same path preserves macOS TCC grants (Accessibility,
//! Input Monitoring, Microphone).

use anyhow::{bail, Context, Result};
use log::info;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

const GITHUB_REPO: &str = "jafreck/murmur";

// ── Public types ────────────────────────────────────────────────────────────

/// Information about an available update.
#[derive(Debug, Clone)]
pub struct UpdateInfo {
    pub current_version: String,
    pub latest_version: String,
    pub download_url: String,
    pub tag: String,
}

// ── GitHub API types ────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct GitHubRelease {
    tag_name: String,
    assets: Vec<GitHubAsset>,
}

#[derive(Deserialize)]
struct GitHubAsset {
    name: String,
    browser_download_url: String,
}

// ── Version comparison ──────────────────────────────────────────────────────

/// Parse a version string like "0.1.2" into (major, minor, patch).
fn parse_version(v: &str) -> Option<(u64, u64, u64)> {
    let v = v.strip_prefix('v').unwrap_or(v);
    let mut parts = v.splitn(3, '.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    Some((major, minor, patch))
}

/// Returns true if `latest` is strictly newer than `current`.
fn is_newer(current: &str, latest: &str) -> bool {
    match (parse_version(current), parse_version(latest)) {
        (Some(c), Some(l)) => l > c,
        _ => false,
    }
}

// ── Platform detection ──────────────────────────────────────────────────────

/// The artifact name (without extension) for this platform.
fn platform_artifact() -> Result<&'static str> {
    if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
        Ok("murmur-darwin-arm64")
    } else if cfg!(target_os = "macos") && cfg!(target_arch = "x86_64") {
        Ok("murmur-darwin-x86_64")
    } else if cfg!(target_os = "linux") && cfg!(target_arch = "x86_64") {
        Ok("murmur-linux-x86_64")
    } else if cfg!(target_os = "windows") && cfg!(target_arch = "x86_64") {
        Ok("murmur-windows-x86_64")
    } else {
        bail!("No pre-built binary for this platform")
    }
}

/// The archive extension for this platform.
fn archive_extension() -> &'static str {
    if cfg!(target_os = "windows") {
        "zip"
    } else {
        "tar.gz"
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Check GitHub Releases for a version newer than `current_version`.
///
/// Returns `None` when the running version is already the latest.
pub fn check_for_update(current_version: &str) -> Result<Option<UpdateInfo>> {
    let url = format!("https://api.github.com/repos/{GITHUB_REPO}/releases/latest");

    let client = reqwest::blocking::Client::builder()
        .user_agent("murmur-updater")
        .build()?;

    let resp = client
        .get(&url)
        .send()
        .context("Failed to reach GitHub Releases API")?;

    if !resp.status().is_success() {
        bail!(
            "GitHub API returned {} when checking for updates",
            resp.status()
        );
    }

    let release: GitHubRelease = resp.json().context("Failed to parse release JSON")?;
    let latest = release
        .tag_name
        .strip_prefix('v')
        .unwrap_or(&release.tag_name);

    if !is_newer(current_version, latest) {
        return Ok(None);
    }

    let artifact = platform_artifact()?;
    let ext = archive_extension();
    let asset_name = format!("{artifact}.{ext}");

    let asset = release
        .assets
        .iter()
        .find(|a| a.name == asset_name)
        .with_context(|| format!("Release {} has no asset '{asset_name}'", release.tag_name))?;

    Ok(Some(UpdateInfo {
        current_version: current_version.to_string(),
        latest_version: latest.to_string(),
        download_url: asset.browser_download_url.clone(),
        tag: release.tag_name,
    }))
}

/// Download the release artifact and replace the current binary in-place.
///
/// `progress` is called with human-readable status messages.
pub fn apply_update(info: &UpdateInfo, progress: impl Fn(&str)) -> Result<()> {
    let current_exe =
        std::env::current_exe().context("Cannot determine current executable path")?;
    let artifact = platform_artifact()?;

    // Download to a temporary directory
    let tmp_dir = tempdir().context("Failed to create temp directory")?;
    let archive_name = format!("{artifact}.{}", archive_extension());
    let archive_path = tmp_dir.join(&archive_name);

    progress("Downloading update...");
    crate::util::download_to_file(&info.download_url, &archive_path, None)?;

    // Extract
    progress("Extracting...");
    extract_archive(&archive_path, &tmp_dir)?;

    let extracted_binary = tmp_dir.join(artifact);
    if !extracted_binary.exists() {
        bail!(
            "Expected binary '{}' not found after extraction",
            extracted_binary.display()
        );
    }

    // Replace the running binary
    progress("Installing...");
    replace_binary(&extracted_binary, &current_exe)?;

    // Platform-specific post-install
    #[cfg(target_os = "macos")]
    {
        progress("Re-signing binary...");
        codesign(&current_exe);
    }

    // Clean up
    let _ = fs::remove_dir_all(&tmp_dir);

    info!(
        "Updated murmur from v{} to v{}",
        info.current_version, info.latest_version
    );
    progress(&format!(
        "Updated to v{} (was v{})",
        info.latest_version, info.current_version
    ));

    Ok(())
}

// ── Internal helpers ────────────────────────────────────────────────────────

fn tempdir() -> Result<PathBuf> {
    let dir = std::env::temp_dir().join(format!("murmur-update-{}", std::process::id()));
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn extract_archive(archive: &Path, dest: &Path) -> Result<()> {
    #[cfg(not(target_os = "windows"))]
    {
        let status = std::process::Command::new("tar")
            .args(["xzf", &archive.to_string_lossy()])
            .current_dir(dest)
            .status()
            .context("Failed to run tar")?;

        if !status.success() {
            bail!("tar extraction failed");
        }
    }

    #[cfg(target_os = "windows")]
    {
        let status = std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!(
                    "Expand-Archive -Path '{}' -DestinationPath '{}' -Force",
                    archive.to_string_lossy(),
                    dest.to_string_lossy()
                ),
            ])
            .status()
            .context("Failed to run PowerShell Expand-Archive")?;

        if !status.success() {
            bail!("Archive extraction failed");
        }
    }

    Ok(())
}

/// Replace the target binary with the new one using atomic rename where
/// possible. On failure, falls back to copy.
fn replace_binary(new_binary: &Path, target: &Path) -> Result<()> {
    // Preserve permissions from the existing binary
    #[cfg(unix)]
    let permissions = fs::metadata(target).ok().map(|m| m.permissions());

    // Remove macOS quarantine attribute
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("xattr")
            .args(["-d", "com.apple.quarantine", &new_binary.to_string_lossy()])
            .status();
    }

    // Try atomic rename first (works if same filesystem)
    if fs::rename(new_binary, target).is_err() {
        // Fall back to copy
        fs::copy(new_binary, target).context("Failed to copy new binary into place")?;
    }

    // Restore permissions
    #[cfg(unix)]
    if let Some(perms) = permissions {
        use std::os::unix::fs::PermissionsExt;
        let mut p = perms;
        p.set_mode(p.mode() | 0o111); // ensure executable
        fs::set_permissions(target, p).ok();
    }

    Ok(())
}

/// Ad-hoc codesign the binary so macOS TCC recognises it.
#[cfg(target_os = "macos")]
fn codesign(binary: &Path) {
    let _ = std::process::Command::new("codesign")
        .args(["-s", "-", "-f", &binary.to_string_lossy()])
        .status();
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_version() {
        assert_eq!(parse_version("0.1.2"), Some((0, 1, 2)));
        assert_eq!(parse_version("v1.2.3"), Some((1, 2, 3)));
        assert_eq!(parse_version("10.20.30"), Some((10, 20, 30)));
        assert_eq!(parse_version("bad"), None);
        assert_eq!(parse_version("1.2"), None);
    }

    #[test]
    fn test_is_newer() {
        assert!(is_newer("0.1.0", "0.1.1"));
        assert!(is_newer("0.1.1", "0.2.0"));
        assert!(is_newer("0.1.1", "1.0.0"));
        assert!(!is_newer("0.1.1", "0.1.1"));
        assert!(!is_newer("0.2.0", "0.1.9"));
        assert!(is_newer("0.1.1", "v0.1.2"));
    }

    #[test]
    fn test_platform_artifact() {
        // Should return a valid artifact name on any CI platform
        let result = platform_artifact();
        assert!(result.is_ok());
        let name = result.unwrap();
        assert!(name.starts_with("murmur-"));
    }
}
