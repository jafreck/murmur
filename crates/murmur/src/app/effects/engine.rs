use log::{error, info};
use std::sync::Arc;

use crate::transcription::model;
use murmur_core::transcription::AsrEngine;

use super::EffectContext;
use crate::app::AppMessage;

// ---------------------------------------------------------------------------
// Effect handlers
// ---------------------------------------------------------------------------

pub(super) fn reload_transcriber(ctx: &mut EffectContext<'_>, generation: u64) {
    let model_size = ctx.state.model_size.clone();
    let language = ctx.state.language.clone();
    let backend = ctx.config.asr_backend;
    let quantization = ctx.config.asr_quantization;
    let tx = ctx.tx.clone();
    info!("Loading {backend} model '{model_size}'...");

    std::thread::spawn(move || {
        match create_engine_on_thread(backend, &model_size, &language, quantization) {
            Ok(engine) => {
                info!("{backend} model '{model_size}' loaded successfully");
                let _ = tx.send(AppMessage::EngineReady(Arc::from(engine), generation));
            }
            Err(e) => {
                error!("Failed to load {backend} model '{model_size}': {e}");
                let _ = tx.send(AppMessage::TranscriptionError(format!(
                    "Failed to load {backend} model '{model_size}': {e}"
                )));
            }
        }
    });
}

/// Download (if needed) and instantiate an ASR engine on a background thread.
pub(in crate::app) fn create_engine_on_thread(
    backend: murmur_core::config::AsrBackend,
    model_size: &str,
    language: &str,
    #[allow(unused_variables)] quantization: murmur_core::config::AsrQuantization,
) -> anyhow::Result<Box<dyn AsrEngine + Send + Sync>> {
    use murmur_core::config::AsrBackend;

    match backend {
        AsrBackend::Whisper => {
            if !murmur_core::transcription::model_exists(model_size) {
                info!("Downloading {model_size} model...");
                let last_milestone = std::cell::Cell::new(u32::MAX);
                model::download(model_size, |percent| {
                    let milestone = percent as u32 / 10 * 10;
                    if milestone != last_milestone.get() {
                        last_milestone.set(milestone);
                        info!("Downloading {model_size}... {milestone}%");
                    }
                })?;
            }

            let model_path = murmur_core::transcription::find_model(model_size)
                .ok_or_else(|| anyhow::anyhow!("Model not found: {model_size}"))?;
            let engine = murmur_core::transcription::WhisperEngine::new(&model_path, language)?;
            Ok(Box::new(engine))
        }
        #[cfg(feature = "onnx")]
        AsrBackend::Qwen3Asr => {
            if !murmur_core::transcription::model_exists_for_backend(
                AsrBackend::Qwen3Asr,
                model_size,
                quantization,
            ) {
                info!("Downloading Qwen3-ASR model...");
                murmur_core::transcription::download_for_backend(
                    AsrBackend::Qwen3Asr,
                    model_size,
                    quantization,
                    |p| {
                        let milestone = p as u32 / 10 * 10;
                        info!("Downloading Qwen3-ASR... {milestone}%");
                    },
                )?;
            }
            let model_dir = murmur_core::transcription::qwen3_asr_model_dir(model_size);
            let engine = murmur_core::transcription::QwenEngine::new(&model_dir, quantization)?;
            Ok(Box::new(engine))
        }
        #[cfg(feature = "onnx")]
        AsrBackend::Parakeet => {
            if !murmur_core::transcription::model_exists_for_backend(
                AsrBackend::Parakeet,
                model_size,
                quantization,
            ) {
                info!("Downloading Parakeet model...");
                murmur_core::transcription::download_for_backend(
                    AsrBackend::Parakeet,
                    model_size,
                    quantization,
                    |p| {
                        let milestone = p as u32 / 10 * 10;
                        info!("Downloading Parakeet... {milestone}%");
                    },
                )?;
            }
            let model_dir = murmur_core::transcription::parakeet_model_dir(model_size);
            let engine = murmur_core::transcription::ParakeetEngine::new(&model_dir, quantization)?;
            Ok(Box::new(engine))
        }
        #[cfg(not(feature = "onnx"))]
        AsrBackend::Qwen3Asr | AsrBackend::Parakeet => {
            anyhow::bail!(
                "ONNX backends require the 'onnx' feature. \
                 Rebuild with: cargo build --features onnx"
            );
        }
        #[cfg(feature = "mlx")]
        AsrBackend::Mlx => {
            let model_dir = murmur_core::transcription::mlx_model_dir(model_size);
            let engine = murmur_core::transcription::MlxEngine::new(&model_dir)?;
            Ok(Box::new(engine))
        }
        #[cfg(not(feature = "mlx"))]
        AsrBackend::Mlx => {
            anyhow::bail!(
                "MLX backend requires the 'mlx' feature. \
                 Rebuild with: cargo build --features mlx"
            );
        }
    }
}
