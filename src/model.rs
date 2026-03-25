use anyhow::{Context, Result};
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;

use crate::config::Config;

const BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

pub fn download(model_size: &str, on_progress: impl Fn(f64)) -> Result<PathBuf> {
    let model_filename = format!("ggml-{model_size}.bin");
    let models_dir = Config::dir().join("models");
    let dest_path = models_dir.join(&model_filename);

    if dest_path.exists() {
        log::info!("Model '{model_size}' already exists at {}", dest_path.display());
        return Ok(dest_path);
    }

    std::fs::create_dir_all(&models_dir)
        .context("Failed to create models directory")?;

    let url = format!("{BASE_URL}/{model_filename}");
    log::info!("Downloading {model_size} model from {url}...");

    let response = reqwest::blocking::get(&url)
        .context("Failed to connect to HuggingFace")?;

    if !response.status().is_success() {
        anyhow::bail!(
            "Download failed with HTTP status {}. Check your network connection.",
            response.status()
        );
    }

    let total = response.content_length().unwrap_or(0);
    let mut reader = response;
    let mut file = File::create(&dest_path)
        .context("Failed to create model file")?;

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

    // Validate the downloaded file
    if !is_valid_ggml_file(&dest_path) {
        let _ = std::fs::remove_file(&dest_path);
        anyhow::bail!(
            "Downloaded file is not a valid GGML model. \
             Check your network connection or try from a different network."
        );
    }

    log::info!("Model downloaded to {}", dest_path.display());
    Ok(dest_path)
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
        assert!(!is_valid_ggml_file(std::path::Path::new("/nonexistent/file")));
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
        // Create a fake model file in the models dir
        let models_dir = Config::dir().join("models");
        let _ = std::fs::create_dir_all(&models_dir);
        let model_path = models_dir.join("ggml-test_download_exists.bin");
        std::fs::write(&model_path, b"fake model").unwrap();

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
}
