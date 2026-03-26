use anyhow::{Context, Result};
use log::error;
use rdev::{listen, Event, EventType, Key};
use std::collections::HashSet;
use std::sync::Mutex;

pub struct ParsedHotkey {
    pub key: Key,
    pub modifiers: Vec<Key>,
}

impl ParsedHotkey {
    pub fn to_config_string(&self) -> String {
        let key_name = crate::keycodes::key_to_name(&self.key);
        if self.modifiers.is_empty() {
            key_name
        } else {
            let mod_names: Vec<String> = self.modifiers
                .iter()
                .map(|k| crate::keycodes::key_to_name(k))
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
    /// When `required_modifiers` is non-empty, the key-down callback only fires
    /// when all listed modifier keys are currently held.
    pub fn start(
        target_key: Key,
        required_modifiers: Vec<Key>,
        on_key_down: impl Fn() + Send + 'static,
        on_key_up: impl Fn() + Send + 'static,
    ) -> Result<()> {
        let required: HashSet<Key> = required_modifiers.into_iter().collect();
        let held_modifiers: Mutex<HashSet<Key>> = Mutex::new(HashSet::new());

        listen(move |event: Event| {
            match event.event_type {
                EventType::KeyPress(key) if is_modifier(&key) => {
                    if let Ok(mut held) = held_modifiers.lock() {
                        held.insert(key);
                    }
                }
                EventType::KeyRelease(key) if is_modifier(&key) => {
                    if let Ok(mut held) = held_modifiers.lock() {
                        held.remove(&key);
                    }
                    if key == target_key {
                        on_key_up();
                    }
                }
                EventType::KeyPress(key) if key == target_key => {
                    let mods_ok = if required.is_empty() {
                        true
                    } else if let Ok(held) = held_modifiers.lock() {
                        required.is_subset(&held)
                    } else {
                        false
                    };
                    if mods_ok {
                        on_key_down();
                    }
                }
                EventType::KeyRelease(key) if key == target_key => {
                    on_key_up();
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
            let parsed = crate::keycodes::parse(&s);
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
