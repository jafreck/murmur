//! Tray state synchronization: syncing app state to tray visual state.

use std::time::Instant;

use crate::config::Config;

use super::icons::{
    build_recording_animation_frames, build_transcribing_animation_frames, is_animating_state,
    is_dark_mode, make_icon_from_decoded, StateColors, ANIMATION_FRAME_INTERVAL_MS,
    ANIMATION_PERIOD_SECS,
};
use super::menu::update_radio_entries;
use super::{TrayController, TrayState};

#[cfg(target_os = "macos")]
use super::native_icon_cache::NativeIconCache;

#[cfg(target_os = "linux")]
use super::linux_icon_cache::LinuxIconCache;

/// Compute the tooltip and status label for a given TrayState.
pub fn state_display(state: &TrayState) -> (&'static str, &'static str) {
    match state {
        TrayState::Loading => ("murmur — Loading model...", "murmur: Loading model..."),
        TrayState::Idle => ("murmur — Idle", "murmur: Idle"),
        TrayState::Recording => ("murmur — Recording...", "murmur: Recording..."),
        TrayState::Transcribing => ("murmur — Transcribing...", "murmur: Transcribing..."),
        TrayState::Error => ("murmur — Error", "murmur: Error"),
    }
}

impl TrayController {
    pub fn set_state(&mut self, state: TrayState) {
        // If we're currently animating and the new state is not the same
        // animation, defer the transition until the current cycle completes.
        if is_animating_state(&self.state) && !is_animating_state(&state) {
            if let Some(start) = self.animation_start {
                let elapsed = start.elapsed().as_secs_f64();
                if elapsed < ANIMATION_PERIOD_SECS {
                    self.deferred_state = Some((state, Instant::now()));
                    return;
                }
            }
        }

        // Clear any pending deferred state since we're applying a new one.
        self.deferred_state = None;

        let (tooltip, label) = state_display(&state);

        // Start or stop animation based on the new state.
        if is_animating_state(&state) {
            // Always reset when entering a new animating state so the
            // animation starts from frame 0.
            if self.animation_start.is_none() || self.state != state {
                self.animation_start = Some(Instant::now());
                self.last_animation_frame = None;
            }

            // Set the first animation frame immediately so there is no
            // flash of the static icon before tick() takes over.
            let first_frame_set = self.try_set_animation_frame_cached(&state, 0);
            if !first_frame_set {
                let frame = match &state {
                    TrayState::Recording => self.recording_frames.first(),
                    TrayState::Transcribing => self.transcribing_frames.first(),
                    _ => None,
                };
                if let Some(icon) = frame {
                    if let Err(e) = self.tray.set_icon(Some(icon.clone())) {
                        log::debug!("Failed to set tray icon: {e}");
                    }
                }
            }
        } else {
            self.animation_start = None;
            self.last_animation_frame = None;

            // Set the static icon for non-animating states.
            let used_cache = self.try_set_icon_cached(&state);
            if !used_cache {
                let icon = match &state {
                    TrayState::Idle => &self.idle_icon,
                    TrayState::Recording | TrayState::Error => &self.recording_icon,
                    TrayState::Transcribing => &self.transcribing_icon,
                    TrayState::Loading => &self.loading_icon,
                };
                let _ = self.tray.set_icon(Some(icon.clone()));
            }
        }

        if let Err(e) = self.tray.set_tooltip(Some(tooltip)) {
            log::debug!("Failed to set tray tooltip: {e}");
        }
        self.status_item.set_text(label);
        self.state = state;
    }

    /// Re-check the system appearance and rebuild all cached icons if it changed.
    /// Call this periodically (e.g. every few seconds) from the main loop,
    /// not on every state transition.
    pub fn refresh_appearance(&mut self) {
        let dark = is_dark_mode();
        if dark == self.cached_dark_mode {
            return;
        }
        self.cached_dark_mode = dark;
        let colors = StateColors::for_appearance(dark);
        if let Ok(i) = make_icon_from_decoded(
            &self.decoded_icon,
            colors.idle.0,
            colors.idle.1,
            colors.idle.2,
            colors.idle.3,
        ) {
            self.idle_icon = i;
        }
        if let Ok(i) = make_icon_from_decoded(
            &self.decoded_icon,
            colors.recording.0,
            colors.recording.1,
            colors.recording.2,
            colors.recording.3,
        ) {
            self.recording_icon = i;
        }
        if let Ok(i) = make_icon_from_decoded(
            &self.decoded_icon,
            colors.transcribing.0,
            colors.transcribing.1,
            colors.transcribing.2,
            colors.transcribing.3,
        ) {
            self.transcribing_icon = i;
        }
        if let Ok(i) = make_icon_from_decoded(
            &self.decoded_icon,
            colors.loading.0,
            colors.loading.1,
            colors.loading.2,
            colors.loading.3,
        ) {
            self.loading_icon = i;
        }
        self.recording_frames = build_recording_animation_frames(colors.recording);
        self.transcribing_frames = build_transcribing_animation_frames(colors.transcribing);
        #[cfg(target_os = "macos")]
        {
            match NativeIconCache::new(&self.tray, &self.decoded_icon, &colors) {
                Ok(cache) => self.native_cache = Some(cache),
                Err(e) => {
                    log::warn!("Failed to rebuild native icon cache: {e}");
                    self.native_cache = None;
                }
            }
        }
        #[cfg(target_os = "linux")]
        {
            match LinuxIconCache::new(&self.decoded_icon, &colors) {
                Ok(cache) => self.linux_cache = Some(cache),
                Err(e) => {
                    log::warn!("Failed to rebuild Linux icon cache: {e}");
                    self.linux_cache = None;
                }
            }
        }
        // Re-apply the current state with the new icons.
        self.set_state(self.state.clone());
    }

    /// Try to set the icon using a platform-native cache.
    /// Returns `true` if a cache was used, `false` to fall back to tray-icon.
    fn try_set_icon_cached(&self, _state: &TrayState) -> bool {
        #[cfg(target_os = "macos")]
        if let Some(ref cache) = self.native_cache {
            cache.set_icon_for_state(_state);
            return true;
        }
        #[cfg(target_os = "linux")]
        if let Some(ref cache) = self.linux_cache {
            // SAFETY: the app_indicator pointer is valid for the lifetime of
            // self.tray, which outlives every call to this method.
            unsafe {
                cache.set_icon_for_state(self.tray.app_indicator(), _state);
            }
            return true;
        }
        false
    }

    /// Try to set an animation frame using a platform-native cache.
    #[allow(unused_variables)]
    fn try_set_animation_frame_cached(&self, state: &TrayState, frame_idx: usize) -> bool {
        #[cfg(target_os = "macos")]
        if let Some(ref cache) = self.native_cache {
            return cache.set_animation_frame(state, frame_idx);
        }
        #[cfg(target_os = "linux")]
        if let Some(ref cache) = self.linux_cache {
            // SAFETY: same as try_set_icon_cached.
            unsafe {
                return cache.set_animation_frame(self.tray.app_indicator(), state, frame_idx);
            }
        }
        false
    }

    /// Advance the tray animation by one frame (if active).
    ///
    /// Call this every iteration of the main event loop. The method
    /// internally throttles itself to [`ANIMATION_FRAME_INTERVAL_MS`] so
    /// it is cheap to call at the full loop rate.
    pub fn tick(&mut self) {
        let Some(start) = self.animation_start else {
            return;
        };

        if !is_animating_state(&self.state) {
            return;
        }

        // If a deferred state is waiting and one full cycle has played
        // since the deferred state was set, apply it now. This guarantees
        // at least one smooth animation cycle after GPU contention ends.
        if let Some((_, deferred_at)) = &self.deferred_state {
            if deferred_at.elapsed().as_secs_f64() >= ANIMATION_PERIOD_SECS {
                let (deferred, _) = self.deferred_state.take().unwrap();
                self.set_state(deferred);
                return;
            }
        }

        if let Some(last) = self.last_animation_frame {
            if last.elapsed().as_millis() < ANIMATION_FRAME_INTERVAL_MS {
                return;
            }
        }

        let frames = match &self.state {
            TrayState::Recording => &self.recording_frames,
            TrayState::Transcribing => &self.transcribing_frames,
            _ => return,
        };

        if frames.is_empty() {
            return;
        }

        let elapsed = start.elapsed().as_secs_f64();
        let cycle_progress = (elapsed / ANIMATION_PERIOD_SECS).fract();
        let idx = (cycle_progress * frames.len() as f64) as usize % frames.len();

        if self.try_set_animation_frame_cached(&self.state.clone(), idx) {
            self.last_animation_frame = Some(Instant::now());
            return;
        }

        if let Err(e) = self.tray.set_icon(Some(frames[idx].clone())) {
            log::debug!("Failed to set tray animation frame: {e}");
        }

        self.last_animation_frame = Some(Instant::now());
    }

    /// Sync all tray UI elements to match the given config.
    /// Used after reloading config from disk.
    pub fn sync_config(&mut self, config: &Config) {
        self.set_model(&config.model_size);
        self.set_language(&config.language);
        self.set_mode(&config.mode);
        self.set_hotkey(&config.hotkey);
        update_radio_entries(&self.backend_entries, &config.asr_backend, |b| {
            b.to_string()
        });
        self.spoken_punct_item
            .set_checked(config.spoken_punctuation);
        self.filler_removal_item
            .set_checked(config.filler_word_removal);
        self.streaming_item.set_checked(config.streaming);
        self.translate_item.set_checked(config.translate_to_english);
        self.app_mode_item.set_checked(config.is_notes_mode());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- state_display --

    #[test]
    fn state_display_all() {
        for state in [
            TrayState::Idle,
            TrayState::Recording,
            TrayState::Transcribing,
            TrayState::Error,
        ] {
            let (tooltip, label) = state_display(&state);
            assert!(!tooltip.is_empty());
            assert!(!label.is_empty());
            assert!(tooltip.contains("murmur"));
            assert!(label.contains("murmur"));
        }
    }
}
