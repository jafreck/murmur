use log::{error, info};

use crate::ui::overlay::OverlayHandle;

use super::EffectContext;

// ---------------------------------------------------------------------------
// Effect handlers
// ---------------------------------------------------------------------------

pub(super) fn show(ctx: &mut EffectContext<'_>) {
    if let Some(ref mut overlay) = ctx.overlay {
        if let Err(e) = overlay.show() {
            error!("Failed to show overlay: {e}");
        }
    }
}

pub(super) fn hide(ctx: &mut EffectContext<'_>) {
    if let Some(ref mut overlay) = ctx.overlay {
        if let Err(e) = overlay.done() {
            error!("Failed to hide overlay: {e}");
        }
    }
}

pub(super) fn update_text(ctx: &mut EffectContext<'_>, text: &str) {
    if let Some(ref mut overlay) = ctx.overlay {
        if let Err(e) = overlay.set_text(text) {
            error!("Failed to update overlay text: {e}");
        }
    }
}

pub(super) fn spawn(ctx: &mut EffectContext<'_>) {
    // Kill existing overlay if running
    if let Some(mut overlay) = ctx.overlay.take() {
        overlay.quit();
    }

    match OverlayHandle::spawn() {
        Ok(handle) => {
            info!("Overlay process started");
            *ctx.overlay = Some(handle);
        }
        Err(e) => {
            error!("Failed to spawn overlay: {e}");
        }
    }
}

pub(super) fn kill(ctx: &mut EffectContext<'_>) {
    if let Some(mut overlay) = ctx.overlay.take() {
        overlay.quit();
        info!("Overlay process stopped");
    }
}
