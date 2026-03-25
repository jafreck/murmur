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
}
