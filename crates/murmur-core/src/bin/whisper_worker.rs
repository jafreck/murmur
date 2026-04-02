//! Whisper inference worker process.
//!
//! Runs as a child process, reads audio samples from stdin, runs whisper
//! inference, and writes transcribed text to stdout. This isolates whisper's
//! Accelerate/BLAS usage from the parent process's Core Audio thread, avoiding
//! the "failed to encode" error -6.
//!
//! Protocol (little-endian binary):
//!   Request:  u32(sample_count) | f32[sample_count] | u8(translate)
//!   Response: u32(text_byte_len) | utf8[text_byte_len]
//!
//! A sample_count of 0 signals the worker to shut down.

use std::io::{self, Read, Write};

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        log::error!("usage: murmur-whisper-worker <model_path> <language>");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let language = &args[2];

    log::info!("Loading model: {model_path}");
    let ctx = WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
        .expect("failed to load whisper model");

    let language_param: Option<&str> = if language == "auto" {
        None
    } else {
        Some(language)
    };

    log::info!("Worker ready");

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = stdin.lock();
    let mut writer = stdout.lock();

    loop {
        // Read sample count (u32 LE)
        let mut count_buf = [0u8; 4];
        if reader.read_exact(&mut count_buf).is_err() {
            break; // Parent closed pipe
        }
        let sample_count = u32::from_le_bytes(count_buf) as usize;

        if sample_count == 0 {
            break; // Shutdown signal
        }

        // Read f32 samples
        let mut sample_bytes = vec![0u8; sample_count * 4];
        if reader.read_exact(&mut sample_bytes).is_err() {
            break;
        }
        let samples: Vec<f32> = sample_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Read translate flag
        let mut flag_buf = [0u8; 1];
        if reader.read_exact(&mut flag_buf).is_err() {
            break;
        }
        let translate = flag_buf[0] != 0;

        // Run inference
        let text = match run_inference(&ctx, &samples, language_param, translate) {
            Ok(t) => t,
            Err(e) => {
                log::error!("Inference failed: {e}");
                String::new()
            }
        };

        // Write response: u32(text_len) | utf8(text)
        let text_bytes = text.as_bytes();
        let len = (text_bytes.len() as u32).to_le_bytes();
        if writer.write_all(&len).is_err() || writer.write_all(text_bytes).is_err() {
            break;
        }
        let _ = writer.flush();
    }
}

fn run_inference(
    ctx: &WhisperContext,
    samples: &[f32],
    language: Option<&str>,
    translate: bool,
) -> anyhow::Result<String> {
    let mut state = ctx
        .create_state()
        .map_err(|e| anyhow::anyhow!("Failed to create state: {e}"))?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(murmur_core::transcription::model_discovery::inference_thread_count());
    params.set_language(language);
    params.set_translate(translate);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_suppress_nst(true);
    params.set_no_context(true);

    state
        .full(params, samples)
        .map_err(|e| anyhow::anyhow!("Transcription failed: {e}"))?;

    let num_segments = state.full_n_segments();
    let mut text = String::new();
    for i in 0..num_segments {
        if let Some(segment) = state.get_segment(i) {
            if let Ok(segment_text) = segment.to_str_lossy() {
                text.push_str(&segment_text);
            }
        }
    }

    Ok(text.trim().to_string())
}
