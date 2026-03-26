//! Integration tests for the spoken punctuation postprocessing pipeline.
//!
//! These test realistic multi-sentence transcription inputs flowing through
//! the full postprocessing pipeline — beyond the unit-level single-replacement tests.

use murmur::transcription::postprocess;

// ═══════════════════════════════════════════════════════════════════════
//  Realistic multi-sentence dictation
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn full_sentence_with_punctuation() {
    let input = "Hello comma how are you question mark I am fine period";
    let result = postprocess::process(input);
    assert_eq!(result, "Hello, how are you? I am fine.");
}

#[test]
fn multiple_sentences() {
    let input = "The weather is nice period I went for a walk period It was great exclamation mark";
    let result = postprocess::process(input);
    assert_eq!(
        result,
        "The weather is nice. I went for a walk. It was great!"
    );
}

#[test]
fn question_and_answer() {
    let input = "What is your name question mark My name is Alice period";
    let result = postprocess::process(input);
    assert_eq!(result, "What is your name? My name is Alice.");
}

#[test]
fn comma_separated_list() {
    let input = "I need eggs comma milk comma bread comma and butter period";
    let result = postprocess::process(input);
    assert_eq!(result, "I need eggs, milk, bread, and butter.");
}

// ═══════════════════════════════════════════════════════════════════════
//  Mixed punctuation types
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn parenthetical_expression() {
    let input = "The result open paren see appendix close paren was surprising period";
    let result = postprocess::process(input);
    assert!(result.contains('('));
    assert!(result.contains(')'));
    assert!(result.ends_with('.'));
}

#[test]
fn quoted_speech() {
    let input = "She said open quote hello close quote period";
    let result = postprocess::process(input);
    assert_eq!(result.matches('"').count(), 2);
    assert!(result.ends_with('.'));
}

#[test]
fn colon_and_semicolon() {
    let input = "Note colon this is important semicolon do not forget period";
    let result = postprocess::process(input);
    assert!(result.contains(':'));
    assert!(result.contains(';'));
    assert!(result.ends_with('.'));
}

// ═══════════════════════════════════════════════════════════════════════
//  Case insensitivity
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn case_insensitive_all_caps() {
    let input = "Hello PERIOD World COMMA Goodbye EXCLAMATION MARK";
    let result = postprocess::process(input);
    assert!(result.contains('.'));
    assert!(result.contains(','));
    assert!(result.contains('!'));
}

#[test]
fn case_insensitive_mixed_case() {
    let input = "Test Period test Comma test Question Mark";
    let result = postprocess::process(input);
    assert!(result.contains('.'));
    assert!(result.contains(','));
    assert!(result.contains('?'));
}

// ═══════════════════════════════════════════════════════════════════════
//  Newlines and paragraphs
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn new_line_insertion() {
    let input = "First line new line Second line";
    let result = postprocess::process(input);
    assert!(result.contains('\n'));
    let lines: Vec<&str> = result.lines().collect();
    assert!(lines.len() >= 2);
}

#[test]
fn new_paragraph_insertion() {
    let input = "First paragraph new paragraph Second paragraph";
    let result = postprocess::process(input);
    assert!(result.contains("\n\n"));
}

#[test]
fn newline_variant() {
    let input = "Line one newline Line two";
    let result = postprocess::process(input);
    assert!(result.contains('\n'));
}

// ═══════════════════════════════════════════════════════════════════════
//  Edge cases
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn empty_input() {
    assert_eq!(postprocess::process(""), "");
}

#[test]
fn no_punctuation_words() {
    let input = "This is a normal sentence without any spoken punctuation";
    assert_eq!(postprocess::process(input), input);
}

#[test]
fn only_punctuation_words() {
    let input = "period comma question mark";
    let result = postprocess::process(input);
    // Should contain the punctuation marks
    assert!(result.contains('.'));
    assert!(result.contains(','));
    assert!(result.contains('?'));
}

#[test]
fn consecutive_punctuation() {
    let input = "Really question mark exclamation mark";
    let result = postprocess::process(input);
    assert!(result.contains('?'));
    assert!(result.contains('!'));
}

#[test]
fn whitespace_only() {
    let result = postprocess::process("   ");
    // Should not crash
    assert!(!result.is_empty() || result.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════
//  Spacing rules
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn no_space_before_punctuation() {
    let result = postprocess::process("hello comma world");
    // Should be "hello, world" not "hello , world"
    assert!(!result.contains(" ,"), "got: {result}");
}

#[test]
fn space_after_punctuation_before_word() {
    let result = postprocess::process("hello period world");
    // Should have space after period before next word
    assert!(
        result.contains(". ") || result.ends_with('.'),
        "got: {result}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
//  Multi-word replacement patterns
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn full_stop_variant() {
    let input = "That's all full stop";
    let result = postprocess::process(input);
    assert!(result.contains('.'));
    assert!(!result.contains("full stop"));
}

#[test]
fn exclamation_point_variant() {
    let input = "Amazing exclamation point";
    let result = postprocess::process(input);
    assert!(result.contains('!'));
    assert!(!result.contains("exclamation point"));
}

#[test]
fn semi_colon_two_words() {
    let input = "first semi colon second";
    let result = postprocess::process(input);
    assert!(result.contains(';'), "got: {result}");
    assert!(!result.contains("semi colon"), "got: {result}");
}

// ═══════════════════════════════════════════════════════════════════════
//  Dash and hyphen
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn dash_em_dash() {
    let result = postprocess::process("this dash is important");
    assert!(result.contains('—'));
}

#[test]
fn hyphen_regular() {
    let result = postprocess::process("well hyphen known");
    assert!(result.contains('-'));
}

// ═══════════════════════════════════════════════════════════════════════
//  Ellipsis
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ellipsis_three_dots() {
    let result = postprocess::process("wait for it ellipsis");
    assert!(result.contains("..."));
}

// ═══════════════════════════════════════════════════════════════════════
//  Realistic dictation scenarios
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn email_dictation() {
    let input = "Dear John comma new paragraph Thank you for your email period \
                 I will review the proposal and get back to you by Friday period \
                 new paragraph Best regards comma Alice period";
    let result = postprocess::process(input);
    assert!(result.contains("Dear John,"));
    assert!(result.contains("\n\n"));
    assert!(result.contains("Friday."));
}

#[test]
fn code_review_comment() {
    let input = "This function has a bug colon it doesn't handle the null case period \
                 Please add a check before line 42 period";
    let result = postprocess::process(input);
    assert!(result.contains(':'));
    assert!(result.matches('.').count() >= 2);
}

// ═══════════════════════════════════════════════════════════════════════
//  Idempotence: processing output of process should not change it
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn already_punctuated_text_unchanged() {
    let already_done = "Hello, how are you? I am fine.";
    let result = postprocess::process(already_done);
    assert_eq!(result, already_done);
}
