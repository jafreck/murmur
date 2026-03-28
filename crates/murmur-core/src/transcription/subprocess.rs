//! Subprocess-based whisper inference.
//!
//! Spawns `murmur-whisper-worker` as a child process and communicates via
//! stdin/stdout pipes. This isolates whisper's Accelerate/BLAS usage from
//! the parent process's Core Audio thread, avoiding "failed to encode"
//! error -6 that occurs when both run in the same process.

use anyhow::{Context, Result};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

/// A whisper transcriber that runs inference in a child process.
pub struct SubprocessTranscriber {
    child: Child,
}

impl SubprocessTranscriber {
    /// Spawn the worker process with the given model and language.
    ///
    /// The worker binary is expected at the same location as the current
    /// executable (same target directory).
    pub fn new(model_path: &Path, language: &str) -> Result<Self> {
        let worker_path = find_worker_binary()?;
        log::info!(
            "Spawning whisper worker: {} (model: {})",
            worker_path.display(),
            model_path.display()
        );

        let child = Command::new(&worker_path)
            .arg(model_path.to_str().context("Invalid model path")?)
            .arg(language)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("Failed to spawn murmur-whisper-worker")?;

        Ok(Self { child })
    }

    /// Send audio samples to the worker and get back transcribed text.
    pub fn transcribe(&mut self, samples: &[f32], translate: bool) -> Result<String> {
        let stdin = self
            .child
            .stdin
            .as_mut()
            .context("Worker stdin not available")?;
        let stdout = self
            .child
            .stdout
            .as_mut()
            .context("Worker stdout not available")?;

        // Write request: u32(sample_count) | f32[samples] | u8(translate)
        let count = (samples.len() as u32).to_le_bytes();
        stdin.write_all(&count)?;

        let sample_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(samples.as_ptr() as *const u8, samples.len() * 4) };
        stdin.write_all(sample_bytes)?;

        let flag = [if translate { 1u8 } else { 0u8 }];
        stdin.write_all(&flag)?;
        stdin.flush()?;

        // Read response: u32(text_len) | utf8[text]
        let mut len_buf = [0u8; 4];
        stdout
            .read_exact(&mut len_buf)
            .context("Failed to read response length from worker")?;
        let text_len = u32::from_le_bytes(len_buf) as usize;

        if text_len == 0 {
            return Ok(String::new());
        }

        let mut text_buf = vec![0u8; text_len];
        stdout
            .read_exact(&mut text_buf)
            .context("Failed to read response text from worker")?;

        String::from_utf8(text_buf).context("Worker returned invalid UTF-8")
    }

    /// Send shutdown signal to the worker.
    pub fn shutdown(&mut self) {
        if let Some(ref mut stdin) = self.child.stdin {
            let _ = stdin.write_all(&0u32.to_le_bytes());
            let _ = stdin.flush();
        }
        let _ = self.child.wait();
    }
}

impl Drop for SubprocessTranscriber {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Find the worker binary next to the current executable.
fn find_worker_binary() -> Result<PathBuf> {
    let current_exe = std::env::current_exe().context("Failed to get current executable path")?;
    let exe_dir = current_exe
        .parent()
        .context("Executable has no parent directory")?;

    let worker_name = if cfg!(windows) {
        "murmur-whisper-worker.exe"
    } else {
        "murmur-whisper-worker"
    };

    let worker_path = exe_dir.join(worker_name);
    anyhow::ensure!(
        worker_path.exists(),
        "murmur-whisper-worker not found at {}. Build it with: cargo build -p murmur-core --bin murmur-whisper-worker",
        worker_path.display()
    );
    Ok(worker_path)
}
