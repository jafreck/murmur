use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::{SampleFormat, WavSpec, WavWriter};
use std::io::BufWriter;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub struct AudioRecorder {
    stream: Option<cpal::Stream>,
    writer: Arc<Mutex<Option<WavWriter<BufWriter<File>>>>>,
    current_path: Option<PathBuf>,
}

impl AudioRecorder {
    pub fn new() -> Self {
        Self {
            stream: None,
            writer: Arc::new(Mutex::new(None)),
            current_path: None,
        }
    }

    pub fn start(&mut self, output_path: &Path) -> Result<()> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .context("No microphone found")?;

        let supported_config = device
            .default_input_config()
            .context("Failed to get default input config")?;

        let native_rate = supported_config.sample_rate();
        let native_channels = supported_config.channels() as u32;

        // Whisper expects 16kHz, 16-bit, mono PCM WAV
        let target_rate: u32 = 16_000;
        let spec = WavSpec {
            channels: 1,
            sample_rate: target_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };

        let writer = WavWriter::create(output_path, spec)
            .context("Failed to create WAV file")?;
        let writer = Arc::new(Mutex::new(Some(writer)));
        let writer_clone = Arc::clone(&writer);

        self.current_path = Some(output_path.to_path_buf());

        let stream = device.build_input_stream(
            &supported_config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if let Ok(mut guard) = writer_clone.try_lock() {
                    if let Some(ref mut w) = *guard {
                        // Mix to mono if needed, then resample to 16kHz
                        let mono: Vec<f32> = if native_channels > 1 {
                            data.chunks(native_channels as usize)
                                .map(|frame| {
                                    frame.iter().sum::<f32>() / native_channels as f32
                                })
                                .collect()
                        } else {
                            data.to_vec()
                        };

                        let resampled = resample(&mono, native_rate, target_rate);
                        for sample in resampled {
                            let clamped = sample.clamp(-1.0, 1.0);
                            let _ = w.write_sample((clamped * 32767.0) as i16);
                        }
                    }
                }
            },
            |err| {
                log::error!("Audio stream error: {err}");
            },
            None,
        ).context("Failed to build input stream")?;

        stream.play().context("Failed to start audio stream")?;
        self.stream = Some(stream);
        self.writer = writer;

        Ok(())
    }

    pub fn stop(&mut self) -> Option<PathBuf> {
        // Drop the stream first to stop callbacks
        drop(self.stream.take());

        // Finalize the WAV file
        if let Ok(mut guard) = self.writer.lock() {
            if let Some(writer) = guard.take() {
                let _ = writer.finalize();
            }
        }

        self.current_path.take()
    }
}

/// Simple linear interpolation resampler.
/// Good enough for speech; use `rubato` crate for higher quality if needed.
fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return input.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (input.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_idx = i as f64 * ratio;
        let idx = src_idx as usize;
        let frac = src_idx - idx as f64;

        let sample = if idx + 1 < input.len() {
            input[idx] as f64 * (1.0 - frac) + input[idx + 1] as f64 * frac
        } else if idx < input.len() {
            input[idx] as f64
        } else {
            0.0
        };

        output.push(sample as f32);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_same_rate() {
        let input = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let output = resample(&input, 16000, 16000);
        assert_eq!(output, input);
    }

    #[test]
    fn test_resample_downsample() {
        let input: Vec<f32> = (0..48000).map(|i| i as f32 / 48000.0).collect();
        let output = resample(&input, 48000, 16000);
        assert!((output.len() as i64 - 16000).abs() <= 1);
    }

    #[test]
    fn test_resample_empty() {
        let output = resample(&[], 48000, 16000);
        assert!(output.is_empty());
    }

    #[test]
    fn test_resample_upsample() {
        let input: Vec<f32> = (0..8000).map(|i| i as f32 / 8000.0).collect();
        let output = resample(&input, 8000, 16000);
        // Should produce roughly 16000 samples from 8000
        assert!((output.len() as i64 - 16000).abs() <= 1);
    }

    #[test]
    fn test_resample_interpolates() {
        let input = vec![0.0, 1.0];
        let output = resample(&input, 2, 4);
        // Middle values should be interpolated between 0.0 and 1.0
        assert!(output.len() >= 3);
        assert!(output[1] > 0.0 && output[1] < 1.0);
    }

    #[test]
    fn test_resample_single_sample() {
        let input = vec![0.5];
        let output = resample(&input, 16000, 16000);
        assert_eq!(output, vec![0.5]);
    }

    #[test]
    fn test_new_recorder() {
        let recorder = AudioRecorder::new();
        assert!(recorder.stream.is_none());
        assert!(recorder.current_path.is_none());
    }

    #[test]
    fn test_stop_without_start() {
        let mut recorder = AudioRecorder::new();
        let path = recorder.stop();
        assert!(path.is_none());
    }

    #[test]
    fn test_resample_preserves_endpoints() {
        let input = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let output = resample(&input, 44100, 16000);
        // First sample should be close to 0.0
        assert!((output[0] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_resample_large_ratio() {
        let input: Vec<f32> = (0..96000).map(|i| (i as f32).sin()).collect();
        let output = resample(&input, 96000, 16000);
        assert!((output.len() as i64 - 16000).abs() <= 1);
    }
}
