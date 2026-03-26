use super::hotkey::ParsedHotkey;
use rdev::Key;

/// Single source of truth for key ↔ name mappings.
/// Each entry is `(Key, &["canonical_name", "alias1", ...])`.
/// The first string is the canonical name used for serialization.
pub const KEY_MAP: &[(Key, &[&str])] = &[
    // Letters
    (Key::KeyA, &["a"]),
    (Key::KeyB, &["b"]),
    (Key::KeyC, &["c"]),
    (Key::KeyD, &["d"]),
    (Key::KeyE, &["e"]),
    (Key::KeyF, &["f"]),
    (Key::KeyG, &["g"]),
    (Key::KeyH, &["h"]),
    (Key::KeyI, &["i"]),
    (Key::KeyJ, &["j"]),
    (Key::KeyK, &["k"]),
    (Key::KeyL, &["l"]),
    (Key::KeyM, &["m"]),
    (Key::KeyN, &["n"]),
    (Key::KeyO, &["o"]),
    (Key::KeyP, &["p"]),
    (Key::KeyQ, &["q"]),
    (Key::KeyR, &["r"]),
    (Key::KeyS, &["s"]),
    (Key::KeyT, &["t"]),
    (Key::KeyU, &["u"]),
    (Key::KeyV, &["v"]),
    (Key::KeyW, &["w"]),
    (Key::KeyX, &["x"]),
    (Key::KeyY, &["y"]),
    (Key::KeyZ, &["z"]),
    // Numbers
    (Key::Num0, &["0"]),
    (Key::Num1, &["1"]),
    (Key::Num2, &["2"]),
    (Key::Num3, &["3"]),
    (Key::Num4, &["4"]),
    (Key::Num5, &["5"]),
    (Key::Num6, &["6"]),
    (Key::Num7, &["7"]),
    (Key::Num8, &["8"]),
    (Key::Num9, &["9"]),
    // Function keys
    (Key::F1, &["f1"]),
    (Key::F2, &["f2"]),
    (Key::F3, &["f3"]),
    (Key::F4, &["f4"]),
    (Key::F5, &["f5"]),
    (Key::F6, &["f6"]),
    (Key::F7, &["f7"]),
    (Key::F8, &["f8"]),
    (Key::F9, &["f9"]),
    (Key::F10, &["f10"]),
    (Key::F11, &["f11"]),
    (Key::F12, &["f12"]),
    // Modifiers
    (Key::ControlLeft, &["ctrl", "control", "leftctrl"]),
    (Key::ControlRight, &["rightctrl", "rightcontrol"]),
    (Key::ShiftLeft, &["shift", "leftshift"]),
    (Key::ShiftRight, &["rightshift"]),
    (Key::Alt, &["alt", "option", "opt", "leftoption", "leftalt"]),
    (Key::AltGr, &["rightalt", "rightoption", "altgr"]),
    (Key::MetaLeft, &["cmd", "command", "meta", "super", "win"]),
    (Key::MetaRight, &["rightcmd", "rightmeta"]),
    // Special keys
    (Key::Space, &["space"]),
    (Key::Return, &["return", "enter"]),
    (Key::Tab, &["tab"]),
    (Key::Escape, &["escape", "esc"]),
    (Key::Backspace, &["backspace", "delete"]),
    (Key::CapsLock, &["capslock"]),
    // Punctuation
    (Key::Minus, &["minus", "-"]),
    (Key::Equal, &["equal", "="]),
    (Key::LeftBracket, &["leftbracket", "["]),
    (Key::RightBracket, &["rightbracket", "]"]),
    (Key::BackSlash, &["backslash", "\\"]),
    (Key::SemiColon, &["semicolon", ";"]),
    (Key::Quote, &["quote", "'"]),
    (Key::Comma, &["comma", ","]),
    (Key::Dot, &["dot", ".", "period"]),
    (Key::Slash, &["slash", "/"]),
    (Key::BackQuote, &["grave", "`", "backtick"]),
];

/// Parse a key string like "ctrl+shift+space" or "f9" or "globe" into an rdev Key.
pub fn parse(input: &str) -> Option<ParsedHotkey> {
    let lowered = input.to_lowercase();
    let parts: Vec<&str> = lowered.split('+').map(|s| s.trim()).collect();

    let key_name = parts.last()?;
    let key = name_to_key(key_name)?;

    let modifiers: Vec<Key> = parts[..parts.len() - 1]
        .iter()
        .filter_map(|m| {
            let result = name_to_key(m);
            if result.is_none() {
                eprintln!("Warning: unrecognized modifier '{m}' in hotkey '{input}' (ignored)");
            }
            result
        })
        .collect();

    Some(ParsedHotkey { key, modifiers })
}

fn name_to_key(name: &str) -> Option<Key> {
    let lower = name.to_lowercase();
    let lower = lower.as_str();

    // Handle platform-specific globe/fn key before the table lookup
    if lower == "globe" || lower == "fn" {
        #[cfg(target_os = "macos")]
        return Some(Key::Function);

        #[cfg(not(target_os = "macos"))]
        {
            eprintln!("Warning: Globe/fn key not available on this platform. Using F9 instead.");
            return Some(Key::F9);
        }
    }

    for (key, aliases) in KEY_MAP {
        if aliases.contains(&lower) {
            return Some(*key);
        }
    }
    None
}

/// Map an rdev Key back to its canonical name (the first alias in KEY_MAP).
/// Falls back to `format!("{:?}", key).to_lowercase()` for unknown keys.
pub fn key_to_name(key: &Key) -> String {
    for (k, aliases) in KEY_MAP {
        if k == key {
            return aliases[0].to_string();
        }
    }
    format!("{:?}", key).to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_key() {
        let parsed = parse("f9").unwrap();
        assert_eq!(parsed.key, Key::F9);
        assert!(parsed.modifiers.is_empty());
    }

    #[test]
    fn test_parse_combo() {
        let parsed = parse("ctrl+shift+space").unwrap();
        assert_eq!(parsed.key, Key::Space);
        assert_eq!(parsed.modifiers.len(), 2);
    }

    #[test]
    fn test_parse_unknown() {
        assert!(parse("nonexistentkey").is_none());
    }

    #[test]
    fn test_parse_case_insensitive() {
        let parsed = parse("CTRL+SHIFT+SPACE").unwrap();
        assert_eq!(parsed.key, Key::Space);
    }

    #[test]
    fn test_parse_all_letters() {
        for (letter, expected) in [
            ("a", Key::KeyA),
            ("b", Key::KeyB),
            ("c", Key::KeyC),
            ("d", Key::KeyD),
            ("e", Key::KeyE),
            ("f", Key::KeyF),
            ("g", Key::KeyG),
            ("h", Key::KeyH),
            ("i", Key::KeyI),
            ("j", Key::KeyJ),
            ("k", Key::KeyK),
            ("l", Key::KeyL),
            ("m", Key::KeyM),
            ("n", Key::KeyN),
            ("o", Key::KeyO),
            ("p", Key::KeyP),
            ("q", Key::KeyQ),
            ("r", Key::KeyR),
            ("s", Key::KeyS),
            ("t", Key::KeyT),
            ("u", Key::KeyU),
            ("v", Key::KeyV),
            ("w", Key::KeyW),
            ("x", Key::KeyX),
            ("y", Key::KeyY),
            ("z", Key::KeyZ),
        ] {
            let parsed = parse(letter).unwrap();
            assert_eq!(parsed.key, expected, "failed for letter '{letter}'");
        }
    }

    #[test]
    fn test_parse_all_numbers() {
        for (num, expected) in [
            ("0", Key::Num0),
            ("1", Key::Num1),
            ("2", Key::Num2),
            ("3", Key::Num3),
            ("4", Key::Num4),
            ("5", Key::Num5),
            ("6", Key::Num6),
            ("7", Key::Num7),
            ("8", Key::Num8),
            ("9", Key::Num9),
        ] {
            let parsed = parse(num).unwrap();
            assert_eq!(parsed.key, expected, "failed for number '{num}'");
        }
    }

    #[test]
    fn test_parse_function_keys() {
        for (name, expected) in [
            ("f1", Key::F1),
            ("f2", Key::F2),
            ("f3", Key::F3),
            ("f4", Key::F4),
            ("f5", Key::F5),
            ("f6", Key::F6),
            ("f7", Key::F7),
            ("f8", Key::F8),
            ("f9", Key::F9),
            ("f10", Key::F10),
            ("f11", Key::F11),
            ("f12", Key::F12),
        ] {
            let parsed = parse(name).unwrap();
            assert_eq!(parsed.key, expected, "failed for '{name}'");
        }
    }

    #[test]
    fn test_parse_modifier_aliases() {
        assert_eq!(parse("ctrl").unwrap().key, Key::ControlLeft);
        assert_eq!(parse("control").unwrap().key, Key::ControlLeft);
        assert_eq!(parse("leftctrl").unwrap().key, Key::ControlLeft);
        assert_eq!(parse("rightctrl").unwrap().key, Key::ControlRight);
        assert_eq!(parse("rightcontrol").unwrap().key, Key::ControlRight);
        assert_eq!(parse("shift").unwrap().key, Key::ShiftLeft);
        assert_eq!(parse("leftshift").unwrap().key, Key::ShiftLeft);
        assert_eq!(parse("rightshift").unwrap().key, Key::ShiftRight);
        assert_eq!(parse("alt").unwrap().key, Key::Alt);
        assert_eq!(parse("option").unwrap().key, Key::Alt);
        assert_eq!(parse("opt").unwrap().key, Key::Alt);
        assert_eq!(parse("leftoption").unwrap().key, Key::Alt);
        assert_eq!(parse("leftalt").unwrap().key, Key::Alt);
        assert_eq!(parse("rightalt").unwrap().key, Key::AltGr);
        assert_eq!(parse("rightoption").unwrap().key, Key::AltGr);
        assert_eq!(parse("altgr").unwrap().key, Key::AltGr);
        assert_eq!(parse("cmd").unwrap().key, Key::MetaLeft);
        assert_eq!(parse("command").unwrap().key, Key::MetaLeft);
        assert_eq!(parse("meta").unwrap().key, Key::MetaLeft);
        assert_eq!(parse("super").unwrap().key, Key::MetaLeft);
        assert_eq!(parse("win").unwrap().key, Key::MetaLeft);
        assert_eq!(parse("rightcmd").unwrap().key, Key::MetaRight);
        assert_eq!(parse("rightmeta").unwrap().key, Key::MetaRight);
    }

    #[test]
    fn test_parse_special_keys() {
        assert_eq!(parse("space").unwrap().key, Key::Space);
        assert_eq!(parse("return").unwrap().key, Key::Return);
        assert_eq!(parse("enter").unwrap().key, Key::Return);
        assert_eq!(parse("tab").unwrap().key, Key::Tab);
        assert_eq!(parse("escape").unwrap().key, Key::Escape);
        assert_eq!(parse("esc").unwrap().key, Key::Escape);
        assert_eq!(parse("backspace").unwrap().key, Key::Backspace);
        assert_eq!(parse("delete").unwrap().key, Key::Backspace);
        assert_eq!(parse("capslock").unwrap().key, Key::CapsLock);
    }

    #[test]
    fn test_parse_punctuation() {
        assert_eq!(parse("-").unwrap().key, Key::Minus);
        assert_eq!(parse("minus").unwrap().key, Key::Minus);
        assert_eq!(parse("=").unwrap().key, Key::Equal);
        assert_eq!(parse("equal").unwrap().key, Key::Equal);
        assert_eq!(parse("[").unwrap().key, Key::LeftBracket);
        assert_eq!(parse("leftbracket").unwrap().key, Key::LeftBracket);
        assert_eq!(parse("]").unwrap().key, Key::RightBracket);
        assert_eq!(parse("rightbracket").unwrap().key, Key::RightBracket);
        assert_eq!(parse("\\").unwrap().key, Key::BackSlash);
        assert_eq!(parse("backslash").unwrap().key, Key::BackSlash);
        assert_eq!(parse(";").unwrap().key, Key::SemiColon);
        assert_eq!(parse("semicolon").unwrap().key, Key::SemiColon);
        assert_eq!(parse("'").unwrap().key, Key::Quote);
        assert_eq!(parse("quote").unwrap().key, Key::Quote);
        assert_eq!(parse(",").unwrap().key, Key::Comma);
        assert_eq!(parse("comma").unwrap().key, Key::Comma);
        assert_eq!(parse(".").unwrap().key, Key::Dot);
        assert_eq!(parse("dot").unwrap().key, Key::Dot);
        assert_eq!(parse("period").unwrap().key, Key::Dot);
        assert_eq!(parse("/").unwrap().key, Key::Slash);
        assert_eq!(parse("slash").unwrap().key, Key::Slash);
        assert_eq!(parse("`").unwrap().key, Key::BackQuote);
        assert_eq!(parse("grave").unwrap().key, Key::BackQuote);
        assert_eq!(parse("backtick").unwrap().key, Key::BackQuote);
    }

    #[test]
    fn test_parse_globe_fn_key() {
        // On macOS this maps to Key::Function, on other platforms to F9
        let parsed = parse("globe").unwrap();
        #[cfg(target_os = "macos")]
        assert_eq!(parsed.key, Key::Function);
        #[cfg(not(target_os = "macos"))]
        assert_eq!(parsed.key, Key::F9);

        let parsed_fn = parse("fn").unwrap();
        #[cfg(target_os = "macos")]
        assert_eq!(parsed_fn.key, Key::Function);
        #[cfg(not(target_os = "macos"))]
        assert_eq!(parsed_fn.key, Key::F9);
    }

    #[test]
    fn test_parse_with_spaces_around_plus() {
        let parsed = parse("ctrl + shift + a").unwrap();
        assert_eq!(parsed.key, Key::KeyA);
        assert_eq!(parsed.modifiers.len(), 2);
    }

    #[test]
    fn test_parse_modifier_with_unknown_base_key() {
        assert!(parse("ctrl+nonexistent").is_none());
    }

    #[test]
    fn test_parse_unknown_modifier_filtered() {
        let parsed = parse("unknownmod+f9").unwrap();
        assert_eq!(parsed.key, Key::F9);
        // Unknown modifier should be filtered out
        assert!(parsed.modifiers.is_empty());
    }
}
