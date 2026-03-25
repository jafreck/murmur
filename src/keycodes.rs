use crate::hotkey::ParsedHotkey;
use rdev::Key;

/// Parse a key string like "ctrl+shift+space" or "f9" or "globe" into an rdev Key.
pub fn parse(input: &str) -> Option<ParsedHotkey> {
    let lowered = input.to_lowercase();
    let parts: Vec<&str> = lowered.split('+').map(|s| s.trim()).collect();

    let key_name = parts.last()?;
    let key = name_to_key(key_name)?;

    let modifiers: Vec<Key> = parts[..parts.len() - 1]
        .iter()
        .filter_map(|m| name_to_key(m))
        .collect();

    Some(ParsedHotkey { key, modifiers })
}

fn name_to_key(name: &str) -> Option<Key> {
    match name.to_lowercase().as_str() {
        // Letters
        "a" => Some(Key::KeyA),
        "b" => Some(Key::KeyB),
        "c" => Some(Key::KeyC),
        "d" => Some(Key::KeyD),
        "e" => Some(Key::KeyE),
        "f" => Some(Key::KeyF),
        "g" => Some(Key::KeyG),
        "h" => Some(Key::KeyH),
        "i" => Some(Key::KeyI),
        "j" => Some(Key::KeyJ),
        "k" => Some(Key::KeyK),
        "l" => Some(Key::KeyL),
        "m" => Some(Key::KeyM),
        "n" => Some(Key::KeyN),
        "o" => Some(Key::KeyO),
        "p" => Some(Key::KeyP),
        "q" => Some(Key::KeyQ),
        "r" => Some(Key::KeyR),
        "s" => Some(Key::KeyS),
        "t" => Some(Key::KeyT),
        "u" => Some(Key::KeyU),
        "v" => Some(Key::KeyV),
        "w" => Some(Key::KeyW),
        "x" => Some(Key::KeyX),
        "y" => Some(Key::KeyY),
        "z" => Some(Key::KeyZ),

        // Numbers
        "0" => Some(Key::Num0),
        "1" => Some(Key::Num1),
        "2" => Some(Key::Num2),
        "3" => Some(Key::Num3),
        "4" => Some(Key::Num4),
        "5" => Some(Key::Num5),
        "6" => Some(Key::Num6),
        "7" => Some(Key::Num7),
        "8" => Some(Key::Num8),
        "9" => Some(Key::Num9),

        // Function keys
        "f1" => Some(Key::F1),
        "f2" => Some(Key::F2),
        "f3" => Some(Key::F3),
        "f4" => Some(Key::F4),
        "f5" => Some(Key::F5),
        "f6" => Some(Key::F6),
        "f7" => Some(Key::F7),
        "f8" => Some(Key::F8),
        "f9" => Some(Key::F9),
        "f10" => Some(Key::F10),
        "f11" => Some(Key::F11),
        "f12" => Some(Key::F12),

        // Modifiers
        "ctrl" | "control" | "leftctrl" => Some(Key::ControlLeft),
        "rightctrl" | "rightcontrol" => Some(Key::ControlRight),
        "shift" | "leftshift" => Some(Key::ShiftLeft),
        "rightshift" => Some(Key::ShiftRight),
        "alt" | "option" | "opt" | "leftoption" | "leftalt" => Some(Key::Alt),
        "rightalt" | "rightoption" | "altgr" => Some(Key::AltGr),
        "cmd" | "command" | "meta" | "super" | "win" => Some(Key::MetaLeft),
        "rightcmd" | "rightmeta" => Some(Key::MetaRight),

        // Special keys
        "space" => Some(Key::Space),
        "return" | "enter" => Some(Key::Return),
        "tab" => Some(Key::Tab),
        "escape" | "esc" => Some(Key::Escape),
        "backspace" | "delete" => Some(Key::Backspace),
        "capslock" => Some(Key::CapsLock),

        // macOS-specific
        #[cfg(target_os = "macos")]
        "globe" | "fn" => Some(Key::Function),

        // Fallback for globe/fn on non-macOS
        #[cfg(not(target_os = "macos"))]
        "globe" | "fn" => {
            eprintln!("Warning: Globe/fn key not available on this platform. Using F9 instead.");
            Some(Key::F9)
        }

        // Punctuation
        "-" | "minus" => Some(Key::Minus),
        "=" | "equal" => Some(Key::Equal),
        "[" | "leftbracket" => Some(Key::LeftBracket),
        "]" | "rightbracket" => Some(Key::RightBracket),
        "\\" | "backslash" => Some(Key::BackSlash),
        ";" | "semicolon" => Some(Key::SemiColon),
        "'" | "quote" => Some(Key::Quote),
        "," | "comma" => Some(Key::Comma),
        "." | "dot" | "period" => Some(Key::Dot),
        "/" | "slash" => Some(Key::Slash),
        "`" | "grave" | "backtick" => Some(Key::BackQuote),

        _ => None,
    }
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
            ("a", Key::KeyA), ("b", Key::KeyB), ("c", Key::KeyC), ("d", Key::KeyD),
            ("e", Key::KeyE), ("f", Key::KeyF), ("g", Key::KeyG), ("h", Key::KeyH),
            ("i", Key::KeyI), ("j", Key::KeyJ), ("k", Key::KeyK), ("l", Key::KeyL),
            ("m", Key::KeyM), ("n", Key::KeyN), ("o", Key::KeyO), ("p", Key::KeyP),
            ("q", Key::KeyQ), ("r", Key::KeyR), ("s", Key::KeyS), ("t", Key::KeyT),
            ("u", Key::KeyU), ("v", Key::KeyV), ("w", Key::KeyW), ("x", Key::KeyX),
            ("y", Key::KeyY), ("z", Key::KeyZ),
        ] {
            let parsed = parse(letter).unwrap();
            assert_eq!(parsed.key, expected, "failed for letter '{letter}'");
        }
    }

    #[test]
    fn test_parse_all_numbers() {
        for (num, expected) in [
            ("0", Key::Num0), ("1", Key::Num1), ("2", Key::Num2), ("3", Key::Num3),
            ("4", Key::Num4), ("5", Key::Num5), ("6", Key::Num6), ("7", Key::Num7),
            ("8", Key::Num8), ("9", Key::Num9),
        ] {
            let parsed = parse(num).unwrap();
            assert_eq!(parsed.key, expected, "failed for number '{num}'");
        }
    }

    #[test]
    fn test_parse_function_keys() {
        for (name, expected) in [
            ("f1", Key::F1), ("f2", Key::F2), ("f3", Key::F3), ("f4", Key::F4),
            ("f5", Key::F5), ("f6", Key::F6), ("f7", Key::F7), ("f8", Key::F8),
            ("f9", Key::F9), ("f10", Key::F10), ("f11", Key::F11), ("f12", Key::F12),
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
