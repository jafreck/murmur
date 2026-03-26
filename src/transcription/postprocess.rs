use std::sync::OnceLock;

use regex::Regex;

/// Replacement rules for spoken punctuation.
/// Multi-word patterns must come before their single-word substrings
/// (e.g. "semi colon" before "colon") to avoid partial matches.
const REPLACEMENTS: &[(&str, &str)] = &[
    (r"\bperiod\b", "."),
    (r"\bfull stop\b", "."),
    (r"\bcomma\b", ","),
    (r"\bquestion mark\b", "?"),
    (r"\bexclamation mark\b", "!"),
    (r"\bexclamation point\b", "!"),
    (r"\bsemicolon\b", ";"),
    (r"\bsemi colon\b", ";"),
    (r"\bcolon\b", ":"),
    (r"\bellipsis\b", "..."),
    (r"\bdash\b", " —"),
    (r"\bhyphen\b", "-"),
    (r"\bopen quote\b", "\""),
    (r"\bclose quote\b", "\""),
    (r"\bopen paren\b", "("),
    (r"\bclose paren\b", ")"),
    (r"\bnew line\b", "\n"),
    (r"\bnewline\b", "\n"),
    (r"\bnew paragraph\b", "\n\n"),
];

fn compiled_replacements() -> &'static [(Regex, &'static str)] {
    static COMPILED: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    COMPILED.get_or_init(|| {
        REPLACEMENTS
            .iter()
            .filter_map(|(pattern, replacement)| {
                Regex::new(&format!("(?i){pattern}"))
                    .ok()
                    .map(|re| (re, *replacement))
            })
            .collect()
    })
}

/// Process transcribed text, replacing spoken punctuation words with symbols.
pub fn process(text: &str) -> String {
    let mut result = text.to_string();

    for (re, replacement) in compiled_replacements() {
        result = re.replace_all(&result, *replacement).to_string();
    }

    result = fix_spacing_around_punctuation(&result);
    result = ensure_space_after_punctuation(&result);
    result
}

/// Remove spaces before punctuation marks.
fn fix_spacing_around_punctuation(text: &str) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"\s+([.,?!:;])").unwrap());
    re.replace_all(text, "$1").to_string()
}

/// Ensure a space exists after punctuation when followed by a word character.
fn ensure_space_after_punctuation(text: &str) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"([.,?!:;])(\w)").unwrap());
    re.replace_all(text, "$1 $2").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_replacements() {
        assert!(process("hello period").contains('.'));
        assert!(process("hello comma world").contains(','));
        assert!(process("what question mark").contains('?'));
    }

    #[test]
    fn test_case_insensitive() {
        assert!(process("hello PERIOD").contains('.'));
        assert!(process("hello Period").contains('.'));
    }

    #[test]
    fn test_spacing_fix() {
        let result = process("hello comma world");
        assert_eq!(result, "hello, world");
    }

    #[test]
    fn test_new_line() {
        let result = process("first line new line second line");
        assert!(result.contains('\n'));
    }

    #[test]
    fn test_new_paragraph() {
        let result = process("first new paragraph second");
        assert!(result.contains("\n\n"));
    }

    #[test]
    fn test_no_replacements() {
        let input = "hello world this is a test";
        assert_eq!(process(input), input);
    }

    #[test]
    fn test_empty_string() {
        assert_eq!(process(""), "");
    }

    #[test]
    fn test_ensure_space_after_punct() {
        assert_eq!(
            ensure_space_after_punctuation("hello,world"),
            "hello, world"
        );
        assert_eq!(
            ensure_space_after_punctuation("hello, world"),
            "hello, world"
        );
    }

    #[test]
    fn test_full_stop() {
        let result = process("end full stop");
        assert!(result.contains('.'));
    }

    #[test]
    fn test_exclamation_mark() {
        let result = process("wow exclamation mark");
        assert!(result.contains('!'));
    }

    #[test]
    fn test_exclamation_point() {
        let result = process("wow exclamation point");
        assert!(result.contains('!'));
    }

    #[test]
    fn test_colon() {
        let result = process("note colon details");
        assert!(result.contains(':'));
    }

    #[test]
    fn test_semicolon() {
        let result = process("first semicolon second");
        assert!(result.contains(';'));
    }

    #[test]
    fn test_semi_colon() {
        let result = process("first semi colon second");
        assert!(result.contains(';'), "result was: {result}");
        assert!(!result.contains("semi"), "result was: {result}");
    }

    #[test]
    fn test_ellipsis() {
        let result = process("wait ellipsis");
        assert!(result.contains("..."));
    }

    #[test]
    fn test_dash() {
        let result = process("one dash two");
        assert!(result.contains('—'));
    }

    #[test]
    fn test_hyphen() {
        let result = process("well hyphen known");
        assert!(result.contains('-'));
    }

    #[test]
    fn test_open_close_quote() {
        let result = process("open quote hello close quote");
        assert_eq!(result.matches('"').count(), 2);
    }

    #[test]
    fn test_open_close_paren() {
        let result = process("open paren note close paren");
        assert!(result.contains('('));
        assert!(result.contains(')'));
    }

    #[test]
    fn test_newline_variant() {
        let result = process("line one newline line two");
        assert!(result.contains('\n'));
    }

    #[test]
    fn test_fix_spacing_multiple_spaces() {
        // fix_spacing_around_punctuation only removes spaces immediately before punct
        let result = fix_spacing_around_punctuation("hello   ,world");
        assert!(result.contains(","));
    }

    #[test]
    fn test_ensure_space_after_multiple_punct() {
        assert_eq!(
            ensure_space_after_punctuation("a.b,c!d?e:f;g"),
            "a. b, c! d? e: f; g"
        );
    }

    #[test]
    fn test_multiple_replacements_in_one_string() {
        let result = process("hello comma how are you question mark I am fine period");
        assert!(result.contains(','));
        assert!(result.contains('?'));
        assert!(result.contains('.'));
    }
}
