//! Shared utilities for downloading files with atomic rename semantics.

use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::Context;

/// Download `url` to `dest` atomically.
///
/// The response body is streamed to a temporary `.tmp` file next to `dest`,
/// then renamed into place on success.  If any error occurs the temporary file
/// is removed so partial downloads never appear at the final path.
///
/// The optional `progress` callback receives `(bytes_downloaded, total_bytes)`
/// after each chunk.  `total_bytes` is `0` when the server omits
/// `Content-Length`.
pub fn download_to_file(
    url: &str,
    dest: &Path,
    progress: Option<&dyn Fn(u64, u64)>,
) -> anyhow::Result<()> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).context("Failed to create destination directory")?;
    }

    let tmp_path = tmp_path_for(dest);

    match stream_to_file(url, &tmp_path, progress) {
        Ok(()) => {
            fs::rename(&tmp_path, dest).context("Failed to move downloaded file to final path")?;
            Ok(())
        }
        Err(e) => {
            let _ = fs::remove_file(&tmp_path);
            Err(e)
        }
    }
}

/// Build the temporary download path by appending `.tmp` to `dest`.
fn tmp_path_for(dest: &Path) -> PathBuf {
    let mut s = OsString::from(dest.as_os_str());
    s.push(".tmp");
    PathBuf::from(s)
}

/// Stream an HTTP GET response into `path`, invoking `progress` on each chunk.
fn stream_to_file(
    url: &str,
    path: &Path,
    progress: Option<&dyn Fn(u64, u64)>,
) -> anyhow::Result<()> {
    let response = reqwest::blocking::get(url).context("Failed to connect to server")?;

    if !response.status().is_success() {
        anyhow::bail!("Download failed with HTTP status {}", response.status());
    }

    let total = response.content_length().unwrap_or(0);
    let mut reader = response;
    let mut file = File::create(path).context("Failed to create temporary download file")?;

    let mut downloaded: u64 = 0;
    let mut buf = [0u8; 8192];

    loop {
        let n = reader.read(&mut buf).context("Download read error")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).context("Write error")?;
        downloaded += n as u64;
        if let Some(cb) = progress {
            cb(downloaded, total);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tmp_path_appends_suffix() {
        let dest = Path::new("/a/b/file.tar.gz");
        assert_eq!(tmp_path_for(dest), PathBuf::from("/a/b/file.tar.gz.tmp"));
    }

    #[test]
    fn tmp_path_no_extension() {
        let dest = Path::new("/a/b/binary");
        assert_eq!(tmp_path_for(dest), PathBuf::from("/a/b/binary.tmp"));
    }
}
