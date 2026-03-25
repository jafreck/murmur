use regex::Regex;

/// Replacement rules for spoken punctuation.
const REPLACEMENTS: &[(&str, &str)] = &[
    (r"\bperiod\b", "."),
    (r"\bfull stop\b", "."),
    (r"\bcomma\b", ","),
    (r"\bquestion mark\b", "?"),
    (r"\bexclamation mark\b", "!"),
    (r"\bexclamation point\b", "!"),
    (r"\bcolon\b", ":"),
    (r"\bsemicolon\b", ";"),
    (r"\bsemi colon\b", ";"),
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

/// Process transcribed text, replacing spoken punctuation words with symbols.
pub fn process(text: &str) -> String {
    let mut result = text.to_string();

    for (pattern, replacement) in REPLACEMENTS {
        if let Ok(re) = Regex::new(&format!("(?i){pattern}")) {
            result = re.replace_all(&result, *replacement).to_string();
        }
    }

    result = fix_spacing_around_punctuation(&result);
    result = ensure_space_after_punctuation(&result);
    result
}

/// Remove spaces before punctuation marks.
fn fix_spacing_around_punctuation(text: &str) -> String {
    let re = Regex::new(r"\s+([.,?!:;])").unwrap();
    re.replace_all(text, "$1").to_string()
}

/// Ensure a space exists after punctuation when followed by a word character.
fn ensure_space_after_punctuation(text: &str) -> String {
    let re = Regex::new(r"([.,?!:;])(\w)").unwrap();
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
        assert_eq!(ensure_space_after_punctuation("hello,world"), "hello, world");
        assert_eq!(ensure_space_after_punctuation("hello, world"), "hello, world");
    }
}
