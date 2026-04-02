use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use log::{info, warn};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use super::resample::{mix_to_mono, resample};
use super::TARGET_RATE;

/// Metadata for an available audio input device.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AudioDevice {
    pub name: String,
    /// True when the device name suggests it carries system/loopback audio
    /// (e.g. BlackHole, Loopback, Soundflower).
    pub is_loopback_hint: bool,
}

/// Well-known virtual audio device names that typically carry system output.
const LOOPBACK_HINTS: &[&str] = &[
    "blackhole",
    "loopback",
    "soundflower",
    "virtual",
    "aggregate",
    "stereo mix",
    "what u hear",
];

fn looks_like_loopback(name: &str) -> bool {
    let lower = name.to_lowercase();
    LOOPBACK_HINTS.iter().any(|hint| lower.contains(hint))
}

/// Lists available audio input devices on the system.
///
/// On macOS, virtual audio devices such as BlackHole or Loopback appear as
/// regular input devices and can capture system audio output when configured
/// as the system's audio output (or via an aggregate device).
pub fn list_audio_devices() -> Vec<AudioDevice> {
    let host = cpal::default_host();
    let mut devices = Vec::new();

    match host.input_devices() {
        Ok(iter) => {
            for device in iter {
                if let Ok(desc) = device.description() {
                    let name = desc.name().to_string();
                    devices.push(AudioDevice {
                        is_loopback_hint: looks_like_loopback(&name),
                        name,
                    });
                }
            }
        }
        Err(e) => {
            warn!("failed to enumerate input devices: {e}");
        }
    }

    devices
}

/// Captures audio from a named input device, resampling to 16 kHz mono f32.
///
/// Designed for system audio capture: the caller selects a virtual audio
/// device (e.g. BlackHole 2ch) that mirrors the system output, and this
/// capturer records from it just like a microphone.
pub struct SystemAudioCapturer {
    device_name: String,
    stream: Option<cpal::Stream>,
    samples: Arc<Mutex<Vec<f32>>>,
    running: Arc<AtomicBool>,
}

impl SystemAudioCapturer {
    /// Create a new capturer targeting the input device with the given name.
    ///
    /// The device is not opened until [`start`] is called.
    pub fn new(device_name: &str) -> Result<Self> {
        // Validate that the device exists.
        let host = cpal::default_host();
        let _device = find_device_by_name(&host, device_name)
            .with_context(|| format!("audio device not found: {device_name}"))?;

        Ok(Self {
            device_name: device_name.to_string(),
            stream: None,
            samples: Arc::new(Mutex::new(Vec::new())),
            running: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Start capturing audio. Returns a handle to the shared sample buffer
    /// that accumulates 16 kHz mono f32 samples.
    pub fn start(&mut self) -> Result<Arc<Mutex<Vec<f32>>>> {
        if self.running.load(Ordering::SeqCst) {
            anyhow::bail!("system audio capture already running");
        }

        let host = cpal::default_host();
        let device =
            find_device_by_name(&host, &self.device_name).context("audio device disappeared")?;

        let supported_config = device
            .default_input_config()
            .context("no supported input config for device")?;

        let source_rate = supported_config.sample_rate();
        let source_channels = supported_config.channels() as u32;
        let config: cpal::StreamConfig = supported_config.into();

        let samples = Arc::clone(&self.samples);
        let running = Arc::clone(&self.running);

        // Clear any samples from a previous session.
        samples.lock().unwrap().clear();

        let stream = device
            .build_input_stream(
                &config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if !running.load(Ordering::Relaxed) {
                        return;
                    }
                    let mono = mix_to_mono(data, source_channels);
                    let resampled = if source_rate != TARGET_RATE {
                        resample(&mono, source_rate, TARGET_RATE)
                    } else {
                        mono
                    };
                    if let Ok(mut buf) = samples.try_lock() {
                        buf.extend_from_slice(&resampled);
                    }
                },
                |err| {
                    warn!("system audio capture error: {err}");
                },
                None,
            )
            .context("failed to build input stream for system audio")?;

        stream
            .play()
            .context("failed to start system audio stream")?;
        self.running.store(true, Ordering::SeqCst);
        self.stream = Some(stream);

        info!("system audio capture started on '{}'", self.device_name);
        Ok(Arc::clone(&self.samples))
    }

    /// Stop capturing.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(stream) = self.stream.take() {
            drop(stream);
        }
        info!("system audio capture stopped");
    }

    /// Whether the capturer is actively recording.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// The name of the target device.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }
}

impl Drop for SystemAudioCapturer {
    fn drop(&mut self) {
        self.stop();
    }
}

fn find_device_by_name(host: &cpal::Host, name: &str) -> Result<cpal::Device> {
    let devices = host
        .input_devices()
        .context("failed to enumerate input devices")?;

    for device in devices {
        if let Ok(desc) = device.description() {
            if desc.name() == name {
                return Ok(device);
            }
        }
    }

    anyhow::bail!("input device '{}' not found", name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loopback_hint_detection() {
        assert!(looks_like_loopback("BlackHole 2ch"));
        assert!(looks_like_loopback("Loopback Audio"));
        assert!(looks_like_loopback("Soundflower (2ch)"));
        assert!(looks_like_loopback("Virtual Audio Device"));
        assert!(!looks_like_loopback("Built-in Microphone"));
        assert!(!looks_like_loopback("USB Audio Headset"));
    }

    #[test]
    #[ignore] // Requires audio hardware; segfaults on headless CI (Windows)
    fn test_list_audio_devices_returns_vec() {
        let devices = list_audio_devices();
        let _ = devices;
    }

    #[test]
    #[ignore] // Requires audio hardware; segfaults on headless CI (Windows)
    fn test_capturer_rejects_nonexistent_device() {
        let result = SystemAudioCapturer::new("__nonexistent_device_12345__");
        assert!(result.is_err());
    }
}
