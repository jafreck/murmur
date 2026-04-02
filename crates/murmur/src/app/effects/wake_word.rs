use log::{error, info};
use std::sync::mpsc;

use super::EffectContext;
use crate::app::AppMessage;

// ---------------------------------------------------------------------------
// Effect handlers
// ---------------------------------------------------------------------------

pub(super) fn start(ctx: &mut EffectContext<'_>) {
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

    match murmur_core::input::wake_word::start_detector(wake_phrase, stop_phrase, event_tx) {
        Ok(handle) => {
            info!("Wake word detector started");
            *ctx.wake_word = Some(handle);
        }
        Err(e) => {
            error!("Failed to start wake word detector: {e}");
        }
    }
}

pub(super) fn stop(ctx: &mut EffectContext<'_>) {
    if let Some(ww) = ctx.wake_word.take() {
        ww.stop();
        info!("Wake word detector stopped");
    }
}
