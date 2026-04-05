use log::{error, info};
#[cfg(feature = "whisper")]
use std::sync::mpsc;
#[cfg(feature = "whisper")]
use std::sync::Arc;

use super::EffectContext;
#[cfg(feature = "whisper")]
use crate::app::AppMessage;
#[cfg(feature = "whisper")]
use murmur_core::transcription::AsrEngine;

// ---------------------------------------------------------------------------
// Effect handlers
// ---------------------------------------------------------------------------

/// Create the ASR engine used for wake word detection (Whisper tiny).
///
/// Downloads the model if it is not already present on disk.
#[cfg(feature = "whisper")]
pub(in crate::app) fn create_wake_word_engine() -> anyhow::Result<Arc<dyn AsrEngine + Send + Sync>>
{
    let model_size = "tiny.en";
    if !murmur_core::transcription::model_exists(model_size) {
        log::info!("Downloading {model_size} model for wake word detection...");
        murmur_core::transcription::download(model_size, |_| {})?;
    }
    let model_path = murmur_core::transcription::find_model(model_size)
        .ok_or_else(|| anyhow::anyhow!("Wake word model '{model_size}' not found"))?;
    let engine = murmur_core::transcription::WhisperEngine::new(&model_path, "en")?;
    Ok(Arc::new(engine))
}

pub(super) fn start(ctx: &mut EffectContext<'_>) {
    #[cfg(not(feature = "whisper"))]
    {
        let _ = ctx;
        error!("Wake word detection requires the 'whisper' feature");
    }

    #[cfg(feature = "whisper")]
    {
        // Stop existing detector if running
        if let Some(ww) = ctx.wake_word.take() {
            ww.stop();
        }

        let wake_phrase = ctx.config.wake_word.clone();
        let stop_phrase = ctx.config.stop_phrase.clone();
        let tx = ctx.tx.clone();

        let (event_tx, event_rx) = mpsc::channel();

        // Forward wake word events to app messages
        std::thread::spawn(move || {
            use murmur_core::input::wake_word::WakeWordEvent;
            while let Ok(event) = event_rx.recv() {
                let msg = match event {
                    WakeWordEvent::WakeWordDetected => AppMessage::WakeWordDetected,
                    WakeWordEvent::StopPhraseDetected => AppMessage::StopPhraseDetected,
                };
                if tx.send(msg).is_err() {
                    break;
                }
            }
        });

        match murmur_core::input::wake_word::start_detector(
            wake_phrase,
            stop_phrase,
            event_tx,
            create_wake_word_engine,
        ) {
            Ok(handle) => {
                info!("Wake word detector started");
                *ctx.wake_word = Some(handle);
            }
            Err(e) => {
                error!("Failed to start wake word detector: {e}");
            }
        }
    }
}

pub(super) fn stop(ctx: &mut EffectContext<'_>) {
    if let Some(ww) = ctx.wake_word.take() {
        ww.stop();
        info!("Wake word detector stopped");
    }
}
