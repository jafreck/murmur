//! End-to-end tests that invoke the `murmur` binary as a subprocess.
//!
//! These tests exercise the real CLI parsing, argument validation, and
//! output formatting — the same code path users hit in production.

use std::process::Command;

/// Get the path to the compiled binary via `cargo build` output dir.
fn murmur_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_murmur"))
}

// ═══════════════════════════════════════════════════════════════════════
//  Help & version
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn no_args_prints_help() {
    let output = murmur_bin().output().expect("failed to run murmur");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "exit code should be 0");
    assert!(
        stdout.contains("Usage") || stdout.contains("usage"),
        "should print usage info, got: {stdout}"
    );
}

#[test]
fn version_flag() {
    let output = murmur_bin()
        .arg("--version")
        .output()
        .expect("failed to run murmur");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    assert!(
        stdout.contains("murmur"),
        "version output should contain 'murmur', got: {stdout}"
    );
}

#[test]
fn help_flag() {
    let output = murmur_bin()
        .arg("--help")
        .output()
        .expect("failed to run murmur");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    assert!(stdout.contains("dictation") || stdout.contains("voice") || stdout.contains("speech"));
    assert!(stdout.contains("start"));
    assert!(stdout.contains("set-hotkey"));
    assert!(stdout.contains("set-model"));
    assert!(stdout.contains("set-language"));
    assert!(stdout.contains("status"));
}

// ═══════════════════════════════════════════════════════════════════════
//  Subcommand help
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn start_help() {
    let output = murmur_bin()
        .args(["start", "--help"])
        .output()
        .expect("failed to run murmur");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("daemon") || stdout.contains("dictation") || stdout.contains("Start"));
}

#[test]
fn set_hotkey_help() {
    let output = murmur_bin()
        .args(["set-hotkey", "--help"])
        .output()
        .expect("failed to run murmur");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("hotkey") || stdout.contains("key") || stdout.contains("Key"),
        "got: {stdout}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
//  Status command
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn status_shows_config_info() {
    let output = murmur_bin()
        .arg("status")
        .output()
        .expect("failed to run murmur");
    assert!(output.status.success(), "status should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should contain key fields
    assert!(stdout.contains("murmur v"), "should show version");
    assert!(stdout.contains("Hotkey:"), "should show hotkey");
    assert!(stdout.contains("Model:"), "should show model");
    assert!(stdout.contains("Language:"), "should show language");
    assert!(stdout.contains("Mode:"), "should show mode");
    assert!(stdout.contains("Streaming:"), "should show streaming");
}

// ═══════════════════════════════════════════════════════════════════════
//  set-hotkey validation
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn set_hotkey_invalid_key_fails() {
    let output = murmur_bin()
        .args(["set-hotkey", "nonexistent_key_xyz"])
        .output()
        .expect("failed to run murmur");
    assert!(
        !output.status.success(),
        "invalid hotkey should fail with non-zero exit"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Unknown key") || stderr.contains("Error"),
        "should report error, got: {stderr}"
    );
}

#[test]
fn set_hotkey_missing_arg_fails() {
    let output = murmur_bin()
        .arg("set-hotkey")
        .output()
        .expect("failed to run murmur");
    assert!(!output.status.success(), "missing arg should fail");
}

// ═══════════════════════════════════════════════════════════════════════
//  set-model validation
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn set_model_invalid_model_fails() {
    let output = murmur_bin()
        .args(["set-model", "nonexistent_model_xyz"])
        .output()
        .expect("failed to run murmur");
    assert!(!output.status.success(), "invalid model should fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Unknown model") || stderr.contains("Error"),
        "should report error, got: {stderr}"
    );
}

#[test]
fn set_model_missing_arg_fails() {
    let output = murmur_bin()
        .arg("set-model")
        .output()
        .expect("failed to run murmur");
    assert!(!output.status.success());
}

// ═══════════════════════════════════════════════════════════════════════
//  set-language validation
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn set_language_invalid_fails() {
    let output = murmur_bin()
        .args(["set-language", "zzz_invalid"])
        .output()
        .expect("failed to run murmur");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Unknown language") || stderr.contains("Error"),
        "should report error, got: {stderr}"
    );
}

#[test]
fn set_language_missing_arg_fails() {
    let output = murmur_bin()
        .arg("set-language")
        .output()
        .expect("failed to run murmur");
    assert!(!output.status.success());
}

// ═══════════════════════════════════════════════════════════════════════
//  Unknown subcommand
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn unknown_subcommand_fails() {
    let output = murmur_bin()
        .arg("frobnicate")
        .output()
        .expect("failed to run murmur");
    assert!(!output.status.success(), "unknown subcommand should fail");
}

// ═══════════════════════════════════════════════════════════════════════
//  get-hotkey reads config
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn get_hotkey_succeeds() {
    let output = murmur_bin()
        .arg("get-hotkey")
        .output()
        .expect("failed to run murmur");
    assert!(output.status.success(), "get-hotkey should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Current hotkey:"),
        "should show current hotkey, got: {stdout}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
//  download-model help (don't actually download)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn download_model_help() {
    let output = murmur_bin()
        .args(["download-model", "--help"])
        .output()
        .expect("failed to run murmur");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Download") || stdout.contains("download") || stdout.contains("model"),
        "got: {stdout}"
    );
}
