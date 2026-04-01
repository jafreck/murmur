use anyhow::{Context, Result};
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;

use crate::config::{AsrBackend, AsrQuantization, Config};

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
        let _ = std::fs::remove_file(&dest_path);
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
        let _ = std::fs::remove_file(&part_path);
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

// ---------------------------------------------------------------------------
// ONNX model helpers
// ---------------------------------------------------------------------------

/// Return the directory where Qwen3-ASR ONNX models are stored.
pub fn qwen3_asr_model_dir(model_size: &str) -> PathBuf {
    Config::dir()
        .join("models")
        .join(format!("qwen3-asr-{model_size}"))
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
    dest_dir: &std::path::Path,
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
        AsrBackend::Whisper => return super::transcriber::model_exists(model_size),
    };

    files.iter().all(|f| dir.join(f).exists())
}

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
    }
}

/// Check if the model exists for the given backend.
pub fn model_exists_for_backend(
    backend: AsrBackend,
    model_size: &str,
    quantization: AsrQuantization,
) -> bool {
    match backend {
        AsrBackend::Whisper => super::transcriber::model_exists(model_size),
        AsrBackend::Qwen3Asr | AsrBackend::Parakeet => {
            onnx_model_exists(backend, model_size, quantization)
        }
    }
}

/// Check GGML/GGJT/GGUF magic bytes at the start of the file.
pub fn is_valid_ggml_file(path: &std::path::Path) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_is_valid_ggml_file_valid() {
        let mut tmp = NamedTempFile::new().unwrap();
        // Write GGUF magic
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

    #[test]
    fn test_download_already_exists() {
        // Create a fake model file with valid GGML magic in the models dir
        let models_dir = Config::dir().join("models");
        let _ = std::fs::create_dir_all(&models_dir);
        let model_path = models_dir.join("ggml-test_download_exists.bin");
        let mut f = std::fs::File::create(&model_path).unwrap();
        f.write_all(&0x67676d6cu32.to_le_bytes()).unwrap();
        f.write_all(&[0u8; 100]).unwrap();
        drop(f);

        // download should return early with the existing path
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
}
