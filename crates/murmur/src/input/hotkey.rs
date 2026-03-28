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
            let mod_names: Vec<String> = self
                .modifiers
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

/// Action returned by [`HotkeyEvaluator::evaluate`].
#[derive(Debug, PartialEq, Eq)]
pub enum HotkeyAction {
    /// The configured hotkey was pressed.
    KeyDown,
    /// The configured hotkey was released.
    KeyUp,
    /// Capture mode intercepted a key press.
    CapturedKey(Key),
}

/// Pure hotkey event evaluator — no OS callbacks, no channels.
///
/// Encapsulates the stateful matching logic so it can be unit-tested
/// without spawning a global listener.
#[derive(Default)]
pub struct HotkeyEvaluator {
    held_modifiers: HashSet<Key>,
    hotkey_active: bool,
}

impl HotkeyEvaluator {
    pub fn new() -> Self {
        Self::default()
    }

    fn modifiers_ok(&self, hotkey_modifiers: &HashSet<Key>) -> bool {
        if hotkey_modifiers.is_empty() {
            return true;
        }
        hotkey_modifiers.is_subset(&self.held_modifiers)
    }

    /// Evaluate a key event against the current hotkey configuration.
    ///
    /// Returns the action to take, if any.  The caller is responsible for
    /// acting on the result (sending messages, updating capture flags, etc.).
    pub fn evaluate(
        &mut self,
        event_type: &EventType,
        hotkey_key: Key,
        hotkey_modifiers: &HashSet<Key>,
        is_capture_mode: bool,
    ) -> Option<HotkeyAction> {
        // Capture mode: intercept the next key press as the new hotkey
        if is_capture_mode {
            if let EventType::KeyPress(key) = event_type {
                return Some(HotkeyAction::CapturedKey(*key));
            }
            return None;
        }

        match event_type {
            // For modifier-key hotkeys, rdev translates macOS flagsChanged
            // into KeyPress/KeyRelease, but enigo's injected CGEvents can
            // desync rdev's internal state, inverting press/release polarity.
            // Use hotkey_active as the source of truth and toggle on any
            // flagsChanged event for the target modifier key.
            EventType::KeyPress(key) | EventType::KeyRelease(key) if is_modifier(key) => {
                if matches!(event_type, EventType::KeyPress(_)) {
                    self.held_modifiers.insert(*key);
                } else {
                    self.held_modifiers.remove(key);
                }

                if *key == hotkey_key {
                    if self.hotkey_active {
                        self.hotkey_active = false;
                        Some(HotkeyAction::KeyUp)
                    } else if self.modifiers_ok(hotkey_modifiers) {
                        self.hotkey_active = true;
                        Some(HotkeyAction::KeyDown)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            // Non-modifier keys: press/release polarity is reliable
            EventType::KeyPress(key) if *key == hotkey_key => {
                if !self.hotkey_active && self.modifiers_ok(hotkey_modifiers) {
                    self.hotkey_active = true;
                    Some(HotkeyAction::KeyDown)
                } else {
                    None
                }
            }
            EventType::KeyRelease(key) if *key == hotkey_key => {
                if self.hotkey_active {
                    self.hotkey_active = false;
                    Some(HotkeyAction::KeyUp)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
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
        let mut evaluator = HotkeyEvaluator::new();

        listen(move |event: Event| {
            let is_capture = capture_flag.load(Ordering::Relaxed);

            // Only lock the config when not in capture mode
            let (target_key, required) = if is_capture {
                (Key::Unknown(0), HashSet::new())
            } else {
                match hotkey_config.lock() {
                    Ok(cfg) => cfg.clone(),
                    Err(_) => {
                        log::error!("Hotkey config mutex poisoned — dropping event");
                        return;
                    }
                }
            };

            match evaluator.evaluate(&event.event_type, target_key, &required, is_capture) {
                Some(HotkeyAction::KeyDown) => {
                    log::info!("Hotkey press (event={:?})", event.event_type);
                    on_key_down();
                }
                Some(HotkeyAction::KeyUp) => {
                    log::info!("Hotkey release (event={:?})", event.event_type);
                    on_key_up();
                }
                Some(HotkeyAction::CapturedKey(key)) => {
                    capture_flag.store(false, Ordering::Relaxed);
                    on_capture(key);
                }
                None => {}
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

    // ── HotkeyEvaluator tests ──────────────────────────────────────

    fn no_mods() -> HashSet<Key> {
        HashSet::new()
    }

    fn mods(keys: &[Key]) -> HashSet<Key> {
        keys.iter().copied().collect()
    }

    // --- Non-modifier key tests ---

    #[test]
    fn eval_non_modifier_press_fires_key_down() {
        let mut eval = HotkeyEvaluator::new();
        let r = eval.evaluate(&EventType::KeyPress(Key::F9), Key::F9, &no_mods(), false);
        assert_eq!(r, Some(HotkeyAction::KeyDown));
    }

    #[test]
    fn eval_non_modifier_release_after_press_fires_key_up() {
        let mut eval = HotkeyEvaluator::new();
        eval.evaluate(&EventType::KeyPress(Key::F9), Key::F9, &no_mods(), false);
        let r = eval.evaluate(&EventType::KeyRelease(Key::F9), Key::F9, &no_mods(), false);
        assert_eq!(r, Some(HotkeyAction::KeyUp));
    }

    #[test]
    fn eval_phantom_release_suppressed() {
        let mut eval = HotkeyEvaluator::new();
        let r = eval.evaluate(&EventType::KeyRelease(Key::F9), Key::F9, &no_mods(), false);
        assert_eq!(r, None);
    }

    #[test]
    fn eval_repeated_press_ignored() {
        let mut eval = HotkeyEvaluator::new();
        eval.evaluate(&EventType::KeyPress(Key::F9), Key::F9, &no_mods(), false);
        let r = eval.evaluate(&EventType::KeyPress(Key::F9), Key::F9, &no_mods(), false);
        assert_eq!(r, None);
    }

    #[test]
    fn eval_wrong_key_ignored() {
        let mut eval = HotkeyEvaluator::new();
        assert_eq!(
            eval.evaluate(&EventType::KeyPress(Key::F10), Key::F9, &no_mods(), false),
            None
        );
        assert_eq!(
            eval.evaluate(&EventType::KeyRelease(Key::F10), Key::F9, &no_mods(), false),
            None
        );
    }

    #[test]
    fn eval_full_press_release_cycle() {
        let mut eval = HotkeyEvaluator::new();
        let m = no_mods();
        assert_eq!(
            eval.evaluate(&EventType::KeyPress(Key::F9), Key::F9, &m, false),
            Some(HotkeyAction::KeyDown)
        );
        assert_eq!(
            eval.evaluate(&EventType::KeyRelease(Key::F9), Key::F9, &m, false),
            Some(HotkeyAction::KeyUp)
        );
        // Can press again after release
        assert_eq!(
            eval.evaluate(&EventType::KeyPress(Key::F9), Key::F9, &m, false),
            Some(HotkeyAction::KeyDown)
        );
        assert_eq!(
            eval.evaluate(&EventType::KeyRelease(Key::F9), Key::F9, &m, false),
            Some(HotkeyAction::KeyUp)
        );
    }

    // --- Modifier-only hotkey tests ---

    #[test]
    fn eval_modifier_press_fires_key_down() {
        let mut eval = HotkeyEvaluator::new();
        let r = eval.evaluate(
            &EventType::KeyPress(Key::ControlLeft),
            Key::ControlLeft,
            &no_mods(),
            false,
        );
        assert_eq!(r, Some(HotkeyAction::KeyDown));
    }

    #[test]
    fn eval_modifier_toggle_press_release() {
        let mut eval = HotkeyEvaluator::new();
        let m = no_mods();
        // Press → KeyDown
        assert_eq!(
            eval.evaluate(
                &EventType::KeyPress(Key::MetaLeft),
                Key::MetaLeft,
                &m,
                false
            ),
            Some(HotkeyAction::KeyDown)
        );
        // Release → KeyUp (toggle)
        assert_eq!(
            eval.evaluate(
                &EventType::KeyRelease(Key::MetaLeft),
                Key::MetaLeft,
                &m,
                false
            ),
            Some(HotkeyAction::KeyUp)
        );
    }

    #[test]
    fn eval_modifier_toggle_two_presses() {
        // Simulates desynced polarity: two presses without a release
        let mut eval = HotkeyEvaluator::new();
        let m = no_mods();
        assert_eq!(
            eval.evaluate(
                &EventType::KeyPress(Key::MetaLeft),
                Key::MetaLeft,
                &m,
                false
            ),
            Some(HotkeyAction::KeyDown)
        );
        // Second press while active → toggles to KeyUp
        assert_eq!(
            eval.evaluate(
                &EventType::KeyPress(Key::MetaLeft),
                Key::MetaLeft,
                &m,
                false
            ),
            Some(HotkeyAction::KeyUp)
        );
    }

    #[test]
    fn eval_modifier_release_when_inactive_with_no_required_mods() {
        // Desynced scenario: release arrives first → treated as activation
        let mut eval = HotkeyEvaluator::new();
        let r = eval.evaluate(
            &EventType::KeyRelease(Key::ControlLeft),
            Key::ControlLeft,
            &no_mods(),
            false,
        );
        assert_eq!(r, Some(HotkeyAction::KeyDown));
    }

    #[test]
    fn eval_non_target_modifier_no_trigger() {
        let mut eval = HotkeyEvaluator::new();
        let r = eval.evaluate(
            &EventType::KeyPress(Key::ShiftLeft),
            Key::ControlLeft,
            &no_mods(),
            false,
        );
        assert_eq!(r, None);
    }

    // --- Modifier combo tests ---

    #[test]
    fn eval_combo_without_held_modifiers_ignored() {
        let mut eval = HotkeyEvaluator::new();
        let required = mods(&[Key::ControlLeft]);
        let r = eval.evaluate(
            &EventType::KeyPress(Key::Space),
            Key::Space,
            &required,
            false,
        );
        assert_eq!(r, None);
    }

    #[test]
    fn eval_combo_with_held_modifiers_fires() {
        let mut eval = HotkeyEvaluator::new();
        let required = mods(&[Key::ControlLeft]);
        // Hold the modifier (not the target, so no action)
        eval.evaluate(
            &EventType::KeyPress(Key::ControlLeft),
            Key::Space,
            &required,
            false,
        );
        // Now press the target key
        let r = eval.evaluate(
            &EventType::KeyPress(Key::Space),
            Key::Space,
            &required,
            false,
        );
        assert_eq!(r, Some(HotkeyAction::KeyDown));
    }

    #[test]
    fn eval_combo_partial_modifiers_held() {
        let mut eval = HotkeyEvaluator::new();
        let required = mods(&[Key::ControlLeft, Key::ShiftLeft]);
        // Hold only Ctrl
        eval.evaluate(
            &EventType::KeyPress(Key::ControlLeft),
            Key::Space,
            &required,
            false,
        );
        assert_eq!(
            eval.evaluate(
                &EventType::KeyPress(Key::Space),
                Key::Space,
                &required,
                false
            ),
            None,
        );
        // Now also hold Shift
        eval.evaluate(
            &EventType::KeyPress(Key::ShiftLeft),
            Key::Space,
            &required,
            false,
        );
        assert_eq!(
            eval.evaluate(
                &EventType::KeyPress(Key::Space),
                Key::Space,
                &required,
                false
            ),
            Some(HotkeyAction::KeyDown),
        );
    }

    #[test]
    fn eval_combo_modifier_released_before_target() {
        let mut eval = HotkeyEvaluator::new();
        let required = mods(&[Key::ControlLeft]);
        // Hold then release modifier
        eval.evaluate(
            &EventType::KeyPress(Key::ControlLeft),
            Key::Space,
            &required,
            false,
        );
        eval.evaluate(
            &EventType::KeyRelease(Key::ControlLeft),
            Key::Space,
            &required,
            false,
        );
        // Target press with modifier no longer held
        let r = eval.evaluate(
            &EventType::KeyPress(Key::Space),
            Key::Space,
            &required,
            false,
        );
        assert_eq!(r, None);
    }

    #[test]
    fn eval_modifier_target_with_required_combo() {
        // Target is a modifier, but other modifiers are also required
        let mut eval = HotkeyEvaluator::new();
        let required = mods(&[Key::ShiftLeft]);
        // Press target modifier without Shift held → modifiers_ok fails
        let r = eval.evaluate(
            &EventType::KeyPress(Key::ControlLeft),
            Key::ControlLeft,
            &required,
            false,
        );
        assert_eq!(r, None);
        // Hold Shift, then press target modifier
        eval.evaluate(
            &EventType::KeyPress(Key::ShiftLeft),
            Key::ControlLeft,
            &required,
            false,
        );
        let r = eval.evaluate(
            &EventType::KeyPress(Key::ControlLeft),
            Key::ControlLeft,
            &required,
            false,
        );
        assert_eq!(r, Some(HotkeyAction::KeyDown));
    }

    // --- Capture mode tests ---

    #[test]
    fn eval_capture_mode_key_press() {
        let mut eval = HotkeyEvaluator::new();
        let r = eval.evaluate(&EventType::KeyPress(Key::KeyA), Key::F9, &no_mods(), true);
        assert_eq!(r, Some(HotkeyAction::CapturedKey(Key::KeyA)));
    }

    #[test]
    fn eval_capture_mode_modifier_press() {
        let mut eval = HotkeyEvaluator::new();
        let r = eval.evaluate(
            &EventType::KeyPress(Key::ShiftLeft),
            Key::F9,
            &no_mods(),
            true,
        );
        assert_eq!(r, Some(HotkeyAction::CapturedKey(Key::ShiftLeft)));
    }

    #[test]
    fn eval_capture_mode_release_ignored() {
        let mut eval = HotkeyEvaluator::new();
        let r = eval.evaluate(&EventType::KeyRelease(Key::KeyA), Key::F9, &no_mods(), true);
        assert_eq!(r, None);
    }

    // --- Unrelated events ---

    #[test]
    fn eval_unrelated_event_ignored() {
        let mut eval = HotkeyEvaluator::new();
        let r = eval.evaluate(
            &EventType::Wheel {
                delta_x: 0,
                delta_y: 0,
            },
            Key::F9,
            &no_mods(),
            false,
        );
        assert_eq!(r, None);
    }

    // ── ParsedHotkey / helpers tests ───────────────────────────────

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
            ParsedHotkey {
                key: Key::F9,
                modifiers: vec![],
            },
            ParsedHotkey {
                key: Key::Space,
                modifiers: vec![Key::ControlLeft],
            },
            ParsedHotkey {
                key: Key::KeyA,
                modifiers: vec![Key::MetaLeft, Key::ShiftLeft],
            },
        ];
        for hk in &cases {
            let s = hk.to_config_string();
            let parsed = crate::input::keycodes::parse(&s);
            assert!(
                parsed.is_some(),
                "Failed to parse round-tripped config: {s}"
            );
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

    #[test]
    fn shared_hotkey_creates_valid_config() {
        let hk = ParsedHotkey {
            key: Key::Space,
            modifiers: vec![Key::ControlLeft, Key::ShiftLeft],
        };
        let config = shared_hotkey(&hk);
        let guard = config.lock().unwrap();
        assert_eq!(guard.0, Key::Space);
        assert!(guard.1.contains(&Key::ControlLeft));
        assert!(guard.1.contains(&Key::ShiftLeft));
        assert_eq!(guard.1.len(), 2);
    }

    #[test]
    fn shared_hotkey_no_modifiers() {
        let hk = ParsedHotkey {
            key: Key::F9,
            modifiers: vec![],
        };
        let config = shared_hotkey(&hk);
        let guard = config.lock().unwrap();
        assert_eq!(guard.0, Key::F9);
        assert!(guard.1.is_empty());
    }

    #[test]
    fn shared_hotkey_deduplicates_modifiers() {
        let hk = ParsedHotkey {
            key: Key::Space,
            modifiers: vec![Key::ControlLeft, Key::ControlLeft],
        };
        let config = shared_hotkey(&hk);
        let guard = config.lock().unwrap();
        // HashSet deduplicates
        assert_eq!(guard.1.len(), 1);
    }

    #[test]
    fn to_config_string_alt_modifier() {
        let hk = ParsedHotkey {
            key: Key::Space,
            modifiers: vec![Key::Alt],
        };
        assert_eq!(hk.to_config_string(), "alt+space");
    }

    #[test]
    fn to_config_string_altgr_modifier() {
        let hk = ParsedHotkey {
            key: Key::KeyA,
            modifiers: vec![Key::AltGr],
        };
        assert_eq!(hk.to_config_string(), "rightalt+a");
    }
}
