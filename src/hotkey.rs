use anyhow::{Context, Result};
use log::error;
use rdev::{listen, Event, EventType, Key};

pub struct ParsedHotkey {
    pub key: Key,
    pub modifiers: Vec<Key>,
}

impl ParsedHotkey {
    pub fn to_config_string(&self) -> String {
        let key_name = format!("{:?}", self.key).to_lowercase();
        if self.modifiers.is_empty() {
            key_name
        } else {
            let mod_names: Vec<String> = self.modifiers
                .iter()
                .map(|k| format!("{:?}", k).to_lowercase())
                .collect();
            format!("{}+{}", mod_names.join("+"), key_name)
        }
    }
}

pub struct HotkeyManager;

impl HotkeyManager {
    /// Start listening for global key events. This blocks the calling thread.
    pub fn start(
        target_key: Key,
        on_key_down: impl Fn() + Send + 'static,
        on_key_up: impl Fn() + Send + 'static,
    ) -> Result<()> {
        listen(move |event: Event| {
            match event.event_type {
                EventType::KeyPress(key) if key == target_key => {
                    on_key_down();
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
        assert!(s.contains("space"));
        assert!(s.contains("+"));
        assert!(s.contains("controlleft"));
        assert!(s.contains("shiftleft"));
    }

    #[test]
    fn to_config_string_single_letter() {
        let hk = ParsedHotkey {
            key: Key::KeyA,
            modifiers: vec![],
        };
        assert_eq!(hk.to_config_string(), "keya");
    }

    #[test]
    fn to_config_string_meta_modifier() {
        let hk = ParsedHotkey {
            key: Key::KeyV,
            modifiers: vec![Key::MetaLeft],
        };
        let s = hk.to_config_string();
        assert!(s.contains("metaleft"));
        assert!(s.contains("keyv"));
    }
}
