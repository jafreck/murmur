use anyhow::{Context, Result};
use log::error;
use rdev::{listen, Event, EventType, Key};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub struct ParsedHotkey {
    pub key: Key,
    pub modifiers: Vec<Key>,
}

/// Shared hotkey configuration that can be updated at runtime.
/// The listener thread reads from this on every key event.
pub type SharedHotkeyConfig = Arc<Mutex<(Key, HashSet<Key>)>>;

/// Shared flag to enter hotkey capture mode.
pub type CaptureFlag = Arc<AtomicBool>;

/// Create a shared hotkey config from a parsed hotkey.
pub fn shared_hotkey(parsed: &ParsedHotkey) -> SharedHotkeyConfig {
    Arc::new(Mutex::new((
        parsed.key,
        parsed.modifiers.iter().copied().collect(),
    )))
}

impl ParsedHotkey {
    pub fn to_config_string(&self) -> String {
        let key_name = super::keycodes::key_to_name(&self.key);
        if self.modifiers.is_empty() {
            key_name
        } else {
            let mod_names: Vec<String> = self.modifiers
                .iter()
                .map(super::keycodes::key_to_name)
                .collect();
            format!("{}+{}", mod_names.join("+"), key_name)
        }
    }
}

/// Return true if `key` is any modifier key (Shift, Ctrl, Alt, Meta).
fn is_modifier(key: &Key) -> bool {
    matches!(
        key,
        Key::ShiftLeft
            | Key::ShiftRight
            | Key::ControlLeft
            | Key::ControlRight
            | Key::Alt
            | Key::AltGr
            | Key::MetaLeft
            | Key::MetaRight
    )
}

pub struct HotkeyManager;

impl HotkeyManager {
    /// Start listening for global key events. This blocks the calling thread.
    ///
    /// The hotkey is read dynamically from `hotkey_config` on every key event,
    /// allowing it to be updated at runtime via `ReloadConfig`.
    ///
    /// When `capture_flag` is set, the next key press is captured via `on_capture`
    /// instead of being matched against the hotkey.
    pub fn start(
        hotkey_config: SharedHotkeyConfig,
        capture_flag: CaptureFlag,
        on_key_down: impl Fn() + Send + 'static,
        on_key_up: impl Fn() + Send + 'static,
        on_capture: impl Fn(Key) + Send + 'static,
    ) -> Result<()> {
        let held_modifiers: Mutex<HashSet<Key>> = Mutex::new(HashSet::new());
        // Track whether we fired on_key_down — only fire on_key_up if so.
        // This prevents phantom release events (e.g. from enigo's paste)
        // from generating spurious KeyUp messages.
        let hotkey_active = AtomicBool::new(false);

        listen(move |event: Event| {
            // Capture mode: intercept the next key press as the new hotkey
            if capture_flag.load(Ordering::Relaxed) {
                if let EventType::KeyPress(key) = event.event_type {
                    capture_flag.store(false, Ordering::Relaxed);
                    on_capture(key);
                }
                return;
            }

            // Read the current hotkey config on each event
            let (target_key, required) = match hotkey_config.lock() {
                Ok(cfg) => cfg.clone(),
                Err(_) => {
                    log::error!("Hotkey config mutex poisoned — dropping event");
                    return;
                }
            };

            let modifiers_ok = || -> bool {
                if required.is_empty() {
                    return true;
                }
                held_modifiers
                    .lock()
                    .map(|held| required.is_subset(&held))
                    .unwrap_or(false)
            };

            match event.event_type {
                // For modifier-key hotkeys, rdev translates macOS flagsChanged
                // into KeyPress/KeyRelease, but enigo's injected CGEvents can
                // desync rdev's internal state, inverting press/release polarity.
                // Use hotkey_active as the source of truth and toggle on any
                // flagsChanged event for the target modifier key.
                EventType::KeyPress(key) | EventType::KeyRelease(key)
                    if is_modifier(&key) =>
                {
                    // Update held_modifiers for combo support
                    if matches!(event.event_type, EventType::KeyPress(_)) {
                        if let Ok(mut held) = held_modifiers.lock() {
                            held.insert(key);
                        }
                    } else if let Ok(mut held) = held_modifiers.lock() {
                        held.remove(&key);
                    }

                    if key == target_key {
                        if hotkey_active.load(Ordering::Relaxed) {
                            log::info!("Hotkey release (modifier key={key:?})");
                            hotkey_active.store(false, Ordering::Relaxed);
                            on_key_up();
                        } else if modifiers_ok() {
                            log::info!("Hotkey press (modifier key={key:?})");
                            hotkey_active.store(true, Ordering::Relaxed);
                            on_key_down();
                        }
                    }
                }
                // Non-modifier keys: press/release polarity is reliable
                EventType::KeyPress(key) if key == target_key => {
                    if !hotkey_active.load(Ordering::Relaxed) && modifiers_ok() {
                        log::info!("Hotkey press (key={key:?})");
                        hotkey_active.store(true, Ordering::Relaxed);
                        on_key_down();
                    }
                }
                EventType::KeyRelease(key) if key == target_key => {
                    if hotkey_active.load(Ordering::Relaxed) {
                        log::info!("Hotkey release (key={key:?})");
                        hotkey_active.store(false, Ordering::Relaxed);
                        on_key_up();
                    }
                }
                _ => {}
            }
        })
        .map_err(|e| {
            error!("Hotkey listener error: {e:?}");
            anyhow::anyhow!("Failed to start hotkey listener: {e:?}")
        })
        .context("Hotkey listener exited unexpectedly")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_config_string_single_key() {
        let hk = ParsedHotkey {
            key: Key::F9,
            modifiers: vec![],
        };
        assert_eq!(hk.to_config_string(), "f9");
    }

    #[test]
    fn to_config_string_with_modifiers() {
        let hk = ParsedHotkey {
            key: Key::Space,
            modifiers: vec![Key::ControlLeft, Key::ShiftLeft],
        };
        let s = hk.to_config_string();
        assert_eq!(s, "ctrl+shift+space");
    }

    #[test]
    fn to_config_string_single_letter() {
        let hk = ParsedHotkey {
            key: Key::KeyA,
            modifiers: vec![],
        };
        assert_eq!(hk.to_config_string(), "a");
    }

    #[test]
    fn to_config_string_meta_modifier() {
        let hk = ParsedHotkey {
            key: Key::KeyV,
            modifiers: vec![Key::MetaLeft],
        };
        assert_eq!(hk.to_config_string(), "cmd+v");
    }

    #[test]
    fn to_config_string_round_trips() {
        // Every config string produced by to_config_string must be parseable
        let cases = vec![
            ParsedHotkey { key: Key::F9, modifiers: vec![] },
            ParsedHotkey { key: Key::Space, modifiers: vec![Key::ControlLeft] },
            ParsedHotkey { key: Key::KeyA, modifiers: vec![Key::MetaLeft, Key::ShiftLeft] },
        ];
        for hk in &cases {
            let s = hk.to_config_string();
            let parsed = crate::input::keycodes::parse(&s);
            assert!(parsed.is_some(), "Failed to parse round-tripped config: {s}");
            let parsed = parsed.unwrap();
            assert_eq!(parsed.key, hk.key, "Key mismatch for config: {s}");
        }
    }

    #[test]
    fn is_modifier_recognizes_all_modifiers() {
        assert!(is_modifier(&Key::ShiftLeft));
        assert!(is_modifier(&Key::ShiftRight));
        assert!(is_modifier(&Key::ControlLeft));
        assert!(is_modifier(&Key::ControlRight));
        assert!(is_modifier(&Key::Alt));
        assert!(is_modifier(&Key::AltGr));
        assert!(is_modifier(&Key::MetaLeft));
        assert!(is_modifier(&Key::MetaRight));
    }

    #[test]
    fn is_modifier_rejects_non_modifiers() {
        assert!(!is_modifier(&Key::Space));
        assert!(!is_modifier(&Key::KeyA));
        assert!(!is_modifier(&Key::F9));
    }
}
