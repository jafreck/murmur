use log::{error, info};
use std::sync::Arc;

use murmur_core::transcription::{AsrEngine, EngineFactory};

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
    let factory = Arc::clone(ctx.engine_factory);
    info!("Loading {backend} model '{model_size}'...");

    std::thread::spawn(move || {
        match create_engine_on_thread(&*factory, backend, &model_size, &language, quantization) {
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

/// Download (if needed) and instantiate an ASR engine via the factory.
pub(in crate::app) fn create_engine_on_thread(
    factory: &dyn EngineFactory,
    backend: murmur_core::config::AsrBackend,
    model_size: &str,
    language: &str,
    quantization: murmur_core::config::AsrQuantization,
) -> anyhow::Result<Box<dyn AsrEngine + Send + Sync>> {
    if !factory.model_exists(backend, model_size, quantization) {
        info!("Downloading {backend} model...");
        let last_milestone = std::cell::Cell::new(u32::MAX);
        let backend_name = backend.to_string();
        factory.download_model(
            backend,
            model_size,
            quantization,
            Box::new(move |percent| {
                let milestone = percent as u32 / 10 * 10;
                if milestone != last_milestone.get() {
                    last_milestone.set(milestone);
                    info!("Downloading {backend_name}... {milestone}%");
                }
            }),
        )?;
    }

    factory.create_engine(backend, model_size, language, quantization)
}
