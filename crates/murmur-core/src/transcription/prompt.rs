#[cfg(feature = "whisper")]
use whisper_rs::WhisperContext;

/// Optional context to improve transcription accuracy via Whisper's initial_prompt.
#[derive(Debug, Clone, Default)]
pub struct TranscriptionContext {
    /// Vocabulary terms to bias the model toward (domain-specific words, names, etc.)
    pub vocabulary: Vec<String>,
    /// Text surrounding the cursor — provides sentence-level context for continuation
    pub surrounding_text: Option<String>,
    /// Additional prompt prefix (e.g., language-specific instructions)
    pub prompt_prefix: Option<String>,
}

/// Maximum length for the initial_prompt string to avoid degrading performance.
const MAX_PROMPT_CHARS: usize = 500;

/// Maximum tokens to allocate when tokenizing a single term for ranking.
#[cfg(feature = "whisper")]
const MAX_TOKENS_PER_TERM: usize = 32;

/// Terms producing fewer BPE tokens than this are considered "known" to Whisper
/// and are deprioritized in the prompt. Terms at or above this threshold are
/// novel/domain-specific and get priority.
const NOVELTY_TOKEN_THRESHOLD: usize = 2;

/// A vocabulary term annotated with its BPE token count for ranking.
#[derive(Debug, Clone)]
pub struct RankedTerm {
    pub term: String,
    pub token_count: usize,
}

/// Rank vocabulary terms by how "novel" they are to Whisper's tokenizer.
///
/// Terms that fragment into many BPE subwords are ones Whisper doesn't know as
/// units — these benefit most from prompt biasing. Single-token terms (like
/// "function" or "return") are already in Whisper's vocabulary and waste prompt
/// space.
///
/// Returns terms sorted by token count descending (most novel first).
/// Terms that fail to tokenize are kept and treated as maximally novel.
#[cfg(feature = "whisper")]
pub fn rank_vocabulary(ctx: &WhisperContext, terms: &[String]) -> Vec<RankedTerm> {
    let mut ranked: Vec<RankedTerm> = terms
        .iter()
        .map(|term| {
            let token_count = match ctx.tokenize(term, MAX_TOKENS_PER_TERM) {
                Ok(tokens) => tokens.len(),
                Err(_) => {
                    // If tokenization fails, assume the term is very novel
                    log::debug!("Failed to tokenize term '{term}', treating as novel");
                    MAX_TOKENS_PER_TERM
                }
            };
            RankedTerm {
                term: term.clone(),
                token_count,
            }
        })
        .collect();

    // Sort by token count descending — most novel terms first
    ranked.sort_by(|a, b| b.token_count.cmp(&a.token_count));
    ranked
}

/// Filter ranked terms to only those Whisper is likely to get wrong.
///
/// Returns terms that tokenize into [`NOVELTY_TOKEN_THRESHOLD`] or more
/// subwords, discarding single-token terms that Whisper already knows.
pub fn filter_novel_terms(ranked: &[RankedTerm]) -> Vec<&RankedTerm> {
    ranked
        .iter()
        .filter(|rt| rt.token_count >= NOVELTY_TOKEN_THRESHOLD)
        .collect()
}

/// Build a Whisper initial_prompt from context information.
///
/// The prompt is structured as:
/// 1. Optional prompt prefix
/// 2. Vocabulary terms (as a comma-separated list)
/// 3. Surrounding text (the most recent text before the cursor)
///
/// Whisper uses this as "prior context" to bias its decoder. The surrounding
/// text is especially powerful — it gives the model sentence-level continuity.
///
/// The total prompt is capped at [`MAX_PROMPT_CHARS`] to avoid degrading performance.
pub fn build_initial_prompt(ctx: &TranscriptionContext) -> Option<String> {
    if ctx.vocabulary.is_empty() && ctx.surrounding_text.is_none() && ctx.prompt_prefix.is_none() {
        return None;
    }

    let mut parts: Vec<String> = Vec::new();

    // Prompt prefix first (e.g. language hints)
    if let Some(prefix) = &ctx.prompt_prefix {
        let trimmed = prefix.trim();
        if !trimmed.is_empty() {
            parts.push(trimmed.to_string());
        }
    }

    // Vocabulary terms — format as natural-looking text to bias decoder
    if !ctx.vocabulary.is_empty() {
        let vocab_str = ctx.vocabulary.join(", ");
        parts.push(vocab_str);
    }

    // Surrounding text last — this is the most important signal as it gives
    // the model direct sentence-level context for continuation
    if let Some(surrounding) = &ctx.surrounding_text {
        let trimmed = surrounding.trim();
        if !trimmed.is_empty() {
            parts.push(trimmed.to_string());
        }
    }

    let prompt = parts.join(". ");
    if prompt.is_empty() {
        return None;
    }

    // Truncate from the LEFT if too long — the end (most recent context) is most valuable
    if prompt.len() > MAX_PROMPT_CHARS {
        let start = prompt.len() - MAX_PROMPT_CHARS;
        // Ensure we don't slice into the middle of a multi-byte UTF-8 character
        let start = snap_to_char_boundary(&prompt, start);
        // For CJK scripts (Chinese, Japanese, Korean), words aren't space-separated
        // so any character boundary is a valid break point. For space-delimited
        // scripts, find the next word boundary to avoid partial words.
        let adjusted_start = if is_cjk_heavy(&prompt[start..]) {
            start
        } else if let Some(i) = prompt[start..].find(' ') {
            start + i + 1
        } else if let Some(i) = prompt[start..].find(", ") {
            start + i + 2
        } else {
            start
        };
        Some(prompt[adjusted_start..].to_string())
    } else {
        Some(prompt)
    }
}

/// Snap a byte offset forward to the nearest UTF-8 character boundary.
fn snap_to_char_boundary(s: &str, byte_offset: usize) -> usize {
    let mut pos = byte_offset;
    while pos < s.len() && !s.is_char_boundary(pos) {
        pos += 1;
    }
    pos
}

/// Heuristic: check if text is predominantly CJK (no spaces between words).
/// Looks at the first 100 characters — if most are CJK codepoints, treat
/// the text as non-space-delimited.
fn is_cjk_heavy(text: &str) -> bool {
    let sample: String = text.chars().take(100).collect();
    if sample.is_empty() {
        return false;
    }
    let cjk_count = sample.chars().filter(|c| is_cjk_char(*c)).count();
    // If more than 30% of chars are CJK, treat as CJK text
    cjk_count * 100 / sample.chars().count() > 30
}

fn is_cjk_char(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}'   // CJK Unified Ideographs
        | '\u{3400}'..='\u{4DBF}' // CJK Extension A
        | '\u{3040}'..='\u{309F}' // Hiragana
        | '\u{30A0}'..='\u{30FF}' // Katakana
        | '\u{AC00}'..='\u{D7AF}' // Hangul Syllables
        | '\u{F900}'..='\u{FAFF}' // CJK Compatibility Ideographs
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- TranscriptionContext --

    #[test]
    fn test_transcription_context_default() {
        let ctx = TranscriptionContext::default();
        assert!(ctx.vocabulary.is_empty());
        assert!(ctx.surrounding_text.is_none());
        assert!(ctx.prompt_prefix.is_none());
    }

    // -- build_initial_prompt --

    #[test]
    fn test_build_prompt_empty_context() {
        let ctx = TranscriptionContext::default();
        assert!(build_initial_prompt(&ctx).is_none());
    }

    #[test]
    fn test_build_prompt_vocabulary_only() {
        let ctx = TranscriptionContext {
            vocabulary: vec![
                "useState".to_string(),
                "async".to_string(),
                "impl".to_string(),
            ],
            ..Default::default()
        };
        let prompt = build_initial_prompt(&ctx).unwrap();
        assert!(prompt.contains("useState"));
        assert!(prompt.contains("async"));
        assert!(prompt.contains("impl"));
    }

    #[test]
    fn test_build_prompt_surrounding_text_only() {
        let ctx = TranscriptionContext {
            surrounding_text: Some("The function returns a".to_string()),
            ..Default::default()
        };
        let prompt = build_initial_prompt(&ctx).unwrap();
        assert!(prompt.contains("The function returns a"));
    }

    #[test]
    fn test_build_prompt_combined() {
        let ctx = TranscriptionContext {
            vocabulary: vec!["boolean".to_string()],
            surrounding_text: Some("The function returns a".to_string()),
            prompt_prefix: None,
        };
        let prompt = build_initial_prompt(&ctx).unwrap();
        assert!(prompt.contains("boolean"));
        assert!(prompt.contains("The function returns a"));
    }

    #[test]
    fn test_build_prompt_with_prefix() {
        let ctx = TranscriptionContext {
            vocabulary: vec![],
            surrounding_text: None,
            prompt_prefix: Some("Technical programming discussion.".to_string()),
        };
        let prompt = build_initial_prompt(&ctx).unwrap();
        assert!(prompt.contains("Technical programming discussion"));
    }

    #[test]
    fn test_build_prompt_truncation() {
        let ctx = TranscriptionContext {
            vocabulary: (0..100).map(|i| format!("word{i}")).collect(),
            surrounding_text: Some("important context at the end".to_string()),
            prompt_prefix: None,
        };
        let prompt = build_initial_prompt(&ctx).unwrap();
        // Should be truncated to MAX_PROMPT_CHARS
        assert!(prompt.len() <= MAX_PROMPT_CHARS);
        // The end (surrounding text) should be preserved since we truncate from the left
        assert!(prompt.contains("important context at the end"));
    }

    #[test]
    fn test_build_prompt_whitespace_handling() {
        let ctx = TranscriptionContext {
            vocabulary: vec![],
            surrounding_text: Some("   ".to_string()),
            prompt_prefix: Some("  ".to_string()),
        };
        assert!(build_initial_prompt(&ctx).is_none());
    }

    #[test]
    fn test_build_prompt_vocabulary_ordering() {
        let ctx = TranscriptionContext {
            vocabulary: vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()],
            ..Default::default()
        };
        let prompt = build_initial_prompt(&ctx).unwrap();
        let alpha_pos = prompt.find("alpha").unwrap();
        let beta_pos = prompt.find("beta").unwrap();
        let gamma_pos = prompt.find("gamma").unwrap();
        assert!(alpha_pos < beta_pos);
        assert!(beta_pos < gamma_pos);
    }

    // -- RankedTerm / vocabulary ranking --

    #[test]
    fn test_ranked_term_struct() {
        let rt = RankedTerm {
            term: "useState".to_string(),
            token_count: 3,
        };
        assert_eq!(rt.term, "useState");
        assert_eq!(rt.token_count, 3);
    }

    #[test]
    fn test_filter_novel_terms_above_threshold() {
        let ranked = vec![
            RankedTerm {
                term: "kAXValueAttribute".to_string(),
                token_count: 5,
            },
            RankedTerm {
                term: "rustfmt".to_string(),
                token_count: 3,
            },
            RankedTerm {
                term: "useState".to_string(),
                token_count: 2,
            },
            RankedTerm {
                term: "function".to_string(),
                token_count: 1,
            },
        ];
        let novel = filter_novel_terms(&ranked);
        assert_eq!(novel.len(), 3);
        assert_eq!(novel[0].term, "kAXValueAttribute");
        assert_eq!(novel[1].term, "rustfmt");
        assert_eq!(novel[2].term, "useState");
    }

    #[test]
    fn test_filter_novel_terms_all_known() {
        let ranked = vec![
            RankedTerm {
                term: "hello".to_string(),
                token_count: 1,
            },
            RankedTerm {
                term: "world".to_string(),
                token_count: 1,
            },
        ];
        let novel = filter_novel_terms(&ranked);
        assert!(novel.is_empty());
    }

    #[test]
    fn test_filter_novel_terms_empty() {
        let novel = filter_novel_terms(&[]);
        assert!(novel.is_empty());
    }

    #[test]
    fn test_novelty_threshold_constant() {
        const { assert!(NOVELTY_TOKEN_THRESHOLD >= 2) };
    }

    #[test]
    fn test_max_prompt_chars_within_whisper_limits() {
        const { assert!(MAX_PROMPT_CHARS <= 1000) };
        const { assert!(MAX_PROMPT_CHARS >= 200) };
    }

    // -- CJK / UTF-8 truncation --

    #[test]
    fn test_snap_to_char_boundary_ascii() {
        let s = "hello world";
        assert_eq!(snap_to_char_boundary(s, 0), 0);
        assert_eq!(snap_to_char_boundary(s, 5), 5);
    }

    #[test]
    fn test_snap_to_char_boundary_multibyte() {
        let s = "héllo";
        // 'é' is 2 bytes in UTF-8 — offset 1 is mid-character
        assert!(s.is_char_boundary(0));
        let snapped = snap_to_char_boundary(s, 2);
        assert!(s.is_char_boundary(snapped));
    }

    #[test]
    fn test_snap_to_char_boundary_cjk() {
        let s = "你好世界";
        // Each CJK char is 3 bytes. Offset 1 is mid-character.
        let snapped = snap_to_char_boundary(s, 1);
        assert!(s.is_char_boundary(snapped));
        assert_eq!(snapped, 3); // snaps to start of second char
    }

    #[test]
    fn test_is_cjk_heavy_chinese() {
        assert!(is_cjk_heavy("你好世界这是一个测试"));
    }

    #[test]
    fn test_is_cjk_heavy_japanese() {
        assert!(is_cjk_heavy("こんにちは世界"));
    }

    #[test]
    fn test_is_cjk_heavy_english() {
        assert!(!is_cjk_heavy("hello world this is a test"));
    }

    #[test]
    fn test_is_cjk_heavy_mixed() {
        // Mostly English with a few CJK chars — should not be CJK-heavy
        assert!(!is_cjk_heavy("hello world 你好 this is a test string"));
    }

    #[test]
    fn test_is_cjk_heavy_empty() {
        assert!(!is_cjk_heavy(""));
    }

    #[test]
    fn test_truncation_preserves_cjk_chars() {
        // Build a prompt with CJK surrounding text that exceeds MAX_PROMPT_CHARS
        let cjk_text = "你好".repeat(300); // 600 CJK chars = 1800 bytes
        let ctx = TranscriptionContext {
            surrounding_text: Some(cjk_text),
            ..Default::default()
        };
        let prompt = build_initial_prompt(&ctx).unwrap();
        // Every character in the result should be valid
        assert!(prompt.len() <= MAX_PROMPT_CHARS);
        for c in prompt.chars() {
            assert!(c.len_utf8() > 0);
        }
    }
}
