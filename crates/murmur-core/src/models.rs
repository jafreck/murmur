//! Central module for model management: download, existence checks, and
//! filesystem layout for all ASR backends.
//!
//! This is the canonical location for model-related logic. The older paths
//! (`transcription::model` and `transcription::model_discovery`) re-export
//! from here for backward compatibility.

use anyhow::{Context, Result};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::config::{AsrBackend, AsrQuantization, Config};

// ---------------------------------------------------------------------------
// ModelRegistry — unified interface
// ---------------------------------------------------------------------------

/// Central registry for all model types and their filesystem layouts.
pub struct ModelRegistry;

impl ModelRegistry {
    /// Check if a model exists for the given backend.
    pub fn exists(backend: AsrBackend, model_size: &str, quantization: AsrQuantization) -> bool {
        model_exists_for_backend(backend, model_size, quantization)
    }

    /// Download the model for the given backend. Returns the model path or
    /// directory.
    pub fn download(
        backend: AsrBackend,
        model_size: &str,
        quantization: AsrQuantization,
        on_progress: impl Fn(f64),
    ) -> Result<PathBuf> {
        download_for_backend(backend, model_size, quantization, on_progress)
    }

    /// Find a Whisper GGML model file in known locations.
    pub fn find_whisper_model(model_size: &str) -> Option<PathBuf> {
        find_model(model_size)
    }

    /// Get the model directory for a backend and model size.
    pub fn model_dir(backend: AsrBackend, model_size: &str) -> PathBuf {
        match backend {
            AsrBackend::Whisper => Config::dir().join("models"),
            AsrBackend::Qwen3Asr => qwen3_asr_model_dir(model_size),
            AsrBackend::Parakeet => parakeet_model_dir(model_size),
            AsrBackend::Mlx => mlx_model_dir(model_size),
        }
    }
}

// ---------------------------------------------------------------------------
// Whisper GGML model discovery
// ---------------------------------------------------------------------------

/// Check if a Whisper GGML model file exists in any known location.
pub fn model_exists(model_size: &str) -> bool {
    find_model(model_size).is_some()
}

/// Find a Whisper GGML model file in known locations.
///
/// Searches the app config directory, common data/cache directories, and
/// (on macOS) Homebrew installation paths.
pub fn find_model(model_size: &str) -> Option<PathBuf> {
    let model_filename = format!("ggml-{model_size}.bin");

    let candidates = vec![
        // App config directory
        Config::dir().join("models").join(&model_filename),
        // Common locations
        dirs::data_dir()
            .unwrap_or_default()
            .join("whisper-cpp")
            .join("models")
            .join(&model_filename),
        dirs::home_dir()
            .unwrap_or_default()
            .join(".cache")
            .join("whisper")
            .join(&model_filename),
    ];

    // macOS-specific Homebrew paths
    #[cfg(target_os = "macos")]
    let candidates = {
        let mut c = candidates;
        c.push(PathBuf::from(format!(
            "/opt/homebrew/share/whisper-cpp/models/{model_filename}"
        )));
        c.push(PathBuf::from(format!(
            "/usr/local/share/whisper-cpp/models/{model_filename}"
        )));
        c
    };

    candidates.into_iter().find(|p| p.exists())
}

// ---------------------------------------------------------------------------
// Whisper GGML model download
// ---------------------------------------------------------------------------

const BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

/// Build the expected model filename for a given model size.
pub fn model_filename(model_size: &str) -> String {
    format!("ggml-{model_size}.bin")
}

/// Build the download URL for a given model size.
///
/// Standard models are hosted at `ggerganov/whisper.cpp` on HuggingFace.
/// Distil-whisper models are hosted under the `distil-whisper` organisation
/// in dedicated `{model}-ggml` repos.
pub fn model_url(model_size: &str) -> String {
    let filename = model_filename(model_size);
    if model_size.starts_with("distil-") {
        format!("https://huggingface.co/distil-whisper/{model_size}-ggml/resolve/main/{filename}")
    } else {
        format!("{BASE_URL}/{filename}")
    }
}

pub fn download(model_size: &str, on_progress: impl Fn(f64)) -> Result<PathBuf> {
    let filename = model_filename(model_size);
    let models_dir = Config::dir().join("models");
    let dest_path = models_dir.join(&filename);

    if dest_path.exists() {
        if is_valid_ggml_file(&dest_path) {
            log::info!(
                "Model '{model_size}' already exists at {}",
                dest_path.display()
            );
            return Ok(dest_path);
        }
        log::warn!(
            "Existing model file at {} is invalid (possibly a partial download), re-downloading",
            dest_path.display()
        );
        if let Err(e) = std::fs::remove_file(&dest_path) {
            log::warn!(
                "Failed to remove invalid model file {}: {e}",
                dest_path.display()
            );
        }
    }

    std::fs::create_dir_all(&models_dir).context("Failed to create models directory")?;

    let url = model_url(model_size);
    log::info!("Downloading {model_size} model from {url}...");

    let response = reqwest::blocking::get(&url).context("Failed to connect to HuggingFace")?;

    if !response.status().is_success() {
        anyhow::bail!(
            "Download failed with HTTP status {}. Check your network connection.",
            response.status()
        );
    }

    let total = response.content_length().unwrap_or(0);
    let mut reader = response;

    // Write to a temporary file first, then rename atomically to avoid
    // leaving a partial file at the final path if the process is killed.
    let part_path = models_dir.join(format!("{filename}.part"));
    let mut file = File::create(&part_path).context("Failed to create temporary model file")?;

    let mut downloaded: u64 = 0;
    let mut buf = [0u8; 8192];

    loop {
        let n = reader.read(&mut buf).context("Download read error")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).context("Write error")?;
        downloaded += n as u64;
        if total > 0 {
            on_progress(downloaded as f64 / total as f64 * 100.0);
        }
    }

    drop(file);

    // Validate before promoting to the final path
    if !is_valid_ggml_file(&part_path) {
        if let Err(e) = std::fs::remove_file(&part_path) {
            log::warn!(
                "Failed to clean up partial download {}: {e}",
                part_path.display()
            );
        }
        anyhow::bail!(
            "Downloaded file is not a valid GGML model. \
             Check your network connection or try from a different network."
        );
    }

    std::fs::rename(&part_path, &dest_path)
        .context("Failed to move downloaded model to final path")?;

    log::info!("Model downloaded to {}", dest_path.display());
    Ok(dest_path)
}

/// Check GGML/GGJT/GGUF magic bytes at the start of the file.
pub fn is_valid_ggml_file(path: &Path) -> bool {
    let Ok(mut file) = File::open(path) else {
        return false;
    };
    let mut magic = [0u8; 4];
    if file.read_exact(&mut magic).is_err() {
        return false;
    }
    let magic_u32 = u32::from_le_bytes(magic);
    // GGML: 0x67676d6c, GGJT: 0x67676a74, GGUF: 0x46554747
    matches!(magic_u32, 0x67676d6c | 0x67676a74 | 0x46554747)
}

// ---------------------------------------------------------------------------
// ONNX model helpers
// ---------------------------------------------------------------------------

/// Return the directory where Qwen3-ASR ONNX models are stored.
pub fn qwen3_asr_model_dir(model_size: &str) -> PathBuf {
    Config::dir()
        .join("models")
        .join(format!("qwen3-asr-{model_size}"))
}

/// Return the directory where MLX Qwen3-ASR models are stored.
pub fn mlx_model_dir(model_size: &str) -> PathBuf {
    Config::dir()
        .join("models")
        .join(format!("mlx-qwen3-asr-{model_size}"))
}

/// Return the directory where Parakeet ONNX models are stored.
pub fn parakeet_model_dir(model_size: &str) -> PathBuf {
    Config::dir()
        .join("models")
        .join(format!("parakeet-{model_size}"))
}

fn qwen3_asr_hf_repo(model_size: &str) -> String {
    format!("andrewleech/qwen3-asr-{model_size}-onnx")
}

fn parakeet_hf_repo(_model_size: &str) -> &'static str {
    "istupakov/parakeet-tdt-0.6b-v3-onnx"
}

fn qwen3_asr_files(quantization: AsrQuantization) -> Vec<&'static str> {
    match quantization {
        AsrQuantization::Int4 => vec![
            "encoder.int4.onnx",
            "decoder_init.int4.onnx",
            "decoder_step.int4.onnx",
            "decoder_weights.int4.data",
            "embed_tokens.bin",
            "tokenizer.json",
            "config.json",
        ],
        _ => vec![
            "encoder.onnx",
            "decoder_init.onnx",
            "decoder_step.onnx",
            "decoder_weights.data",
            "embed_tokens.bin",
            "tokenizer.json",
            "config.json",
        ],
    }
}

fn parakeet_files(quantization: AsrQuantization) -> Vec<&'static str> {
    match quantization {
        AsrQuantization::Int8 => vec!["model.int8.onnx", "tokenizer.json"],
        _ => vec!["model.onnx", "tokenizer.json"],
    }
}

/// Download a single file from HuggingFace into `dest_dir`, using an atomic
/// `.part` rename.  `file_progress` is called with bytes-so-far for this file.
fn download_hf_file(
    repo: &str,
    filename: &str,
    dest_dir: &Path,
    file_progress: impl Fn(u64, u64),
) -> Result<()> {
    let dest = dest_dir.join(filename);
    if dest.exists() {
        log::info!("  {} already present, skipping", filename);
        // Report full progress for this file so the caller can advance.
        if let Ok(meta) = std::fs::metadata(&dest) {
            file_progress(meta.len(), meta.len());
        }
        return Ok(());
    }

    let url = format!("https://huggingface.co/{repo}/resolve/main/{filename}");
    log::info!("  downloading {filename} from {url}");

    let response = reqwest::blocking::get(&url).context("Failed to connect to HuggingFace")?;

    if !response.status().is_success() {
        anyhow::bail!(
            "Download of {filename} failed with HTTP status {}",
            response.status()
        );
    }

    let total = response.content_length().unwrap_or(0);
    let mut reader = response;

    let part_path = dest_dir.join(format!("{filename}.part"));
    let mut file = File::create(&part_path).context("Failed to create temporary file")?;

    let mut downloaded: u64 = 0;
    let mut buf = [0u8; 8192];

    loop {
        let n = reader.read(&mut buf).context("Download read error")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).context("Write error")?;
        downloaded += n as u64;
        file_progress(downloaded, total);
    }

    drop(file);

    std::fs::rename(&part_path, &dest).context("Failed to move downloaded file to final path")?;
    Ok(())
}

/// Download all ONNX model files for the given backend, model size, and
/// quantization level. Returns the model directory path.
pub fn download_onnx_model(
    backend: AsrBackend,
    model_size: &str,
    quantization: AsrQuantization,
    on_progress: impl Fn(f64),
) -> Result<PathBuf> {
    let (dir, repo, files): (PathBuf, String, Vec<&str>) = match backend {
        AsrBackend::Qwen3Asr => (
            qwen3_asr_model_dir(model_size),
            qwen3_asr_hf_repo(model_size),
            qwen3_asr_files(quantization),
        ),
        AsrBackend::Parakeet => (
            parakeet_model_dir(model_size),
            parakeet_hf_repo(model_size).to_owned(),
            parakeet_files(quantization),
        ),
        AsrBackend::Whisper => {
            anyhow::bail!("Use download() for Whisper models");
        }
        AsrBackend::Mlx => {
            anyhow::bail!(
                "Use download_for_backend() for MLX models"
            );
        }
    };

    std::fs::create_dir_all(&dir).context("Failed to create model directory")?;

    let file_count = files.len() as f64;
    for (idx, filename) in files.iter().enumerate() {
        let base_frac = idx as f64 / file_count;
        download_hf_file(&repo, filename, &dir, |downloaded, total| {
            let file_frac = if total > 0 {
                downloaded as f64 / total as f64
            } else {
                0.0
            };
            on_progress((base_frac + file_frac / file_count) * 100.0);
        })?;
    }

    on_progress(100.0);

    // Validate all files are present
    for filename in &files {
        let path = dir.join(filename);
        if !path.exists() {
            anyhow::bail!("Expected file {} is missing after download", filename);
        }
    }

    log::info!("ONNX model downloaded to {}", dir.display());
    Ok(dir)
}

/// Check whether all required ONNX model files are present on disk.
pub fn onnx_model_exists(
    backend: AsrBackend,
    model_size: &str,
    quantization: AsrQuantization,
) -> bool {
    let (dir, files): (PathBuf, Vec<&str>) = match backend {
        AsrBackend::Qwen3Asr => (
            qwen3_asr_model_dir(model_size),
            qwen3_asr_files(quantization),
        ),
        AsrBackend::Parakeet => (parakeet_model_dir(model_size), parakeet_files(quantization)),
        AsrBackend::Whisper => return model_exists(model_size),
        AsrBackend::Mlx => return false,
    };

    files.iter().all(|f| dir.join(f).exists())
}

// ---------------------------------------------------------------------------
// Backend-agnostic wrappers
// ---------------------------------------------------------------------------

/// Download the model for the given backend configuration.
pub fn download_for_backend(
    backend: AsrBackend,
    model_size: &str,
    quantization: AsrQuantization,
    on_progress: impl Fn(f64),
) -> Result<PathBuf> {
    match backend {
        AsrBackend::Whisper => download(model_size, on_progress),
        AsrBackend::Qwen3Asr | AsrBackend::Parakeet => {
            download_onnx_model(backend, model_size, quantization, on_progress)
        }
        AsrBackend::Mlx => {
            let dir = mlx_model_dir(model_size);
            if dir.join("model.safetensors").exists() && dir.join("config.json").exists() {
                return Ok(dir);
            }
            anyhow::bail!(
                "MLX model not found at {}. Download it manually with:\n  \
                 huggingface-cli download moona3k/mlx-qwen3-asr-{} --local-dir {}",
                dir.display(),
                model_size,
                dir.display()
            );
        }
    }
}

/// Check if the model exists for the given backend.
pub fn model_exists_for_backend(
    backend: AsrBackend,
    model_size: &str,
    quantization: AsrQuantization,
) -> bool {
    match backend {
        AsrBackend::Whisper => model_exists(model_size),
        AsrBackend::Qwen3Asr | AsrBackend::Parakeet => {
            onnx_model_exists(backend, model_size, quantization)
        }
        AsrBackend::Mlx => {
            let dir = mlx_model_dir(model_size);
            dir.join("model.safetensors").exists() && dir.join("config.json").exists()
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // -- GGML validation tests --

    #[test]
    fn test_is_valid_ggml_file_valid() {
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&0x46554747u32.to_le_bytes()).unwrap();
        tmp.write_all(&[0u8; 100]).unwrap();
        tmp.flush().unwrap();
        assert!(is_valid_ggml_file(tmp.path()));
    }

    #[test]
    fn test_is_valid_ggml_magic() {
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&0x67676d6cu32.to_le_bytes()).unwrap();
        tmp.write_all(&[0u8; 100]).unwrap();
        tmp.flush().unwrap();
        assert!(is_valid_ggml_file(tmp.path()));
    }

    #[test]
    fn test_is_valid_ggjt_magic() {
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&0x67676a74u32.to_le_bytes()).unwrap();
        tmp.write_all(&[0u8; 100]).unwrap();
        tmp.flush().unwrap();
        assert!(is_valid_ggml_file(tmp.path()));
    }

    #[test]
    fn test_is_valid_ggml_file_invalid() {
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(b"<!DOCTYPE html>").unwrap();
        tmp.flush().unwrap();
        assert!(!is_valid_ggml_file(tmp.path()));
    }

    #[test]
    fn test_is_valid_ggml_file_nonexistent() {
        assert!(!is_valid_ggml_file(std::path::Path::new(
            "/nonexistent/file"
        )));
    }

    #[test]
    fn test_is_valid_ggml_file_too_short() {
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&[0u8; 2]).unwrap();
        tmp.flush().unwrap();
        assert!(!is_valid_ggml_file(tmp.path()));
    }

    #[test]
    fn test_is_valid_ggml_file_empty() {
        let tmp = NamedTempFile::new().unwrap();
        assert!(!is_valid_ggml_file(tmp.path()));
    }

    // -- Download tests --

    #[test]
    fn test_download_already_exists() {
        let models_dir = Config::dir().join("models");
        let _ = std::fs::create_dir_all(&models_dir);
        let model_path = models_dir.join("ggml-test_download_exists.bin");
        let mut f = std::fs::File::create(&model_path).unwrap();
        f.write_all(&0x67676d6cu32.to_le_bytes()).unwrap();
        f.write_all(&[0u8; 100]).unwrap();
        drop(f);

        let result = download("test_download_exists", |_| {});
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), model_path);

        let _ = std::fs::remove_file(&model_path);
    }

    #[test]
    fn test_base_url_format() {
        assert!(BASE_URL.starts_with("https://"));
        assert!(BASE_URL.contains("huggingface"));
    }

    // -- Filename / URL tests --

    #[test]
    fn test_model_filename() {
        assert_eq!(model_filename("base.en"), "ggml-base.en.bin");
        assert_eq!(model_filename("tiny"), "ggml-tiny.bin");
        assert_eq!(model_filename("large-v3-turbo"), "ggml-large-v3-turbo.bin");
        assert_eq!(
            model_filename("distil-large-v3"),
            "ggml-distil-large-v3.bin"
        );
    }

    #[test]
    fn test_model_url() {
        let url = model_url("base.en");
        assert!(url.starts_with("https://"));
        assert!(url.contains("ggml-base.en.bin"));
        assert!(url.contains("huggingface"));
    }

    #[test]
    fn test_distil_model_url() {
        let url = model_url("distil-large-v3");
        assert_eq!(
            url,
            "https://huggingface.co/distil-whisper/distil-large-v3-ggml/resolve/main/ggml-distil-large-v3.bin"
        );
    }

    // -- Model discovery tests --

    #[test]
    fn test_model_exists_nonexistent() {
        assert!(!model_exists("nonexistent_model_that_doesnt_exist_xyz"));
    }

    #[test]
    fn test_find_model_nonexistent() {
        assert!(find_model("nonexistent_model_that_doesnt_exist_xyz").is_none());
    }

    #[test]
    fn test_find_model_builds_correct_filename() {
        let result = find_model("test_does_not_exist");
        assert!(result.is_none());
    }

    #[test]
    fn test_find_model_checks_config_dir() {
        let models_dir = Config::dir().join("models");
        let _ = std::fs::create_dir_all(&models_dir);
        let model_path = models_dir.join("ggml-test_temp_model.bin");
        std::fs::write(&model_path, b"test model content").unwrap();

        let result = find_model("test_temp_model");
        assert!(result.is_some());
        assert_eq!(result.unwrap(), model_path);

        let _ = std::fs::remove_file(&model_path);
    }

    #[test]
    fn test_find_model_returns_none_for_empty_string() {
        assert!(find_model("").is_none());
    }

    #[test]
    fn test_model_exists_consistent_with_find_model() {
        let size = "nonexistent_test_model_xyz";
        assert_eq!(model_exists(size), find_model(size).is_some());
    }

    // -- ModelRegistry tests --

    #[test]
    fn test_registry_model_dir_whisper() {
        let dir = ModelRegistry::model_dir(AsrBackend::Whisper, "base.en");
        assert!(dir.ends_with("models"));
    }

    #[test]
    fn test_registry_model_dir_qwen3() {
        let dir = ModelRegistry::model_dir(AsrBackend::Qwen3Asr, "base");
        assert!(dir.ends_with("qwen3-asr-base"));
    }

    #[test]
    fn test_registry_exists_nonexistent() {
        assert!(!ModelRegistry::exists(
            AsrBackend::Whisper,
            "nonexistent_xyz",
            AsrQuantization::default(),
        ));
    }
}
