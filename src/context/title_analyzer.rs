//! Auto-detect programming language and generate prompt context from window titles.
//!
//! Editors typically include the filename in the window title, e.g.:
//! - `"auth.rs — Visual Studio Code"`
//! - `"main.py - PyCharm"`
//! - `"App.tsx - WebStorm"`
//! - `"vim ~/.config/fish/config.fish"`
//!
//! This module extracts the filename, maps its extension to a programming language,
//! and generates a descriptive prompt prefix for Whisper biasing — all without
//! any user configuration.

use super::provider::DictationMode;

/// Result of analyzing a window title.
#[derive(Debug, Clone, PartialEq)]
pub struct TitleContext {
    /// Detected programming language (e.g. "Rust", "Python")
    pub language: Option<String>,
    /// File extension that was matched (e.g. "rs", "py")
    pub extension: Option<String>,
    /// Extracted filename from the title (e.g. "auth.rs")
    pub filename: Option<String>,
    /// Auto-generated prompt prefix for Whisper (e.g. "Rust programming.")
    pub prompt_prefix: Option<String>,
    /// Suggested dictation mode based on detected context
    pub suggested_mode: Option<DictationMode>,
}

/// Known file extension → (language name, prompt prefix, dictation mode) mappings.
const EXTENSION_MAP: &[(&str, &str, &str)] = &[
    // Systems
    ("rs", "Rust", "Rust programming"),
    ("go", "Go", "Go programming"),
    ("c", "C", "C programming"),
    ("h", "C", "C programming"),
    ("cpp", "C++", "C++ programming"),
    ("cc", "C++", "C++ programming"),
    ("cxx", "C++", "C++ programming"),
    ("hpp", "C++", "C++ programming"),
    ("zig", "Zig", "Zig programming"),
    // JVM
    ("java", "Java", "Java programming"),
    ("kt", "Kotlin", "Kotlin programming"),
    ("kts", "Kotlin", "Kotlin programming"),
    ("scala", "Scala", "Scala programming"),
    ("groovy", "Groovy", "Groovy programming"),
    // .NET
    ("cs", "C#", "C# programming"),
    ("fs", "F#", "F# programming"),
    ("vb", "Visual Basic", "Visual Basic programming"),
    // Web / JS
    ("js", "JavaScript", "JavaScript programming"),
    ("jsx", "JavaScript React", "JavaScript React programming"),
    ("ts", "TypeScript", "TypeScript programming"),
    ("tsx", "TypeScript React", "TypeScript React programming"),
    ("mjs", "JavaScript", "JavaScript programming"),
    ("cjs", "JavaScript", "JavaScript programming"),
    // Python
    ("py", "Python", "Python programming"),
    ("pyi", "Python", "Python programming"),
    ("pyx", "Cython", "Cython programming"),
    // Ruby
    ("rb", "Ruby", "Ruby programming"),
    ("erb", "Ruby", "Ruby template"),
    // PHP
    ("php", "PHP", "PHP programming"),
    // Swift / Obj-C
    ("swift", "Swift", "Swift programming"),
    ("m", "Objective-C", "Objective-C programming"),
    ("mm", "Objective-C++", "Objective-C++ programming"),
    // Shell
    ("sh", "Shell", "Shell scripting"),
    ("bash", "Bash", "Bash scripting"),
    ("zsh", "Zsh", "Zsh scripting"),
    ("fish", "Fish", "Fish shell scripting"),
    ("ps1", "PowerShell", "PowerShell scripting"),
    // Config / Data
    ("json", "JSON", "JSON configuration"),
    ("yaml", "YAML", "YAML configuration"),
    ("yml", "YAML", "YAML configuration"),
    ("toml", "TOML", "TOML configuration"),
    ("xml", "XML", "XML markup"),
    ("ini", "INI", "INI configuration"),
    // Markup / Docs
    ("md", "Markdown", "Markdown documentation"),
    ("mdx", "MDX", "MDX documentation"),
    ("rst", "reStructuredText", "reStructuredText documentation"),
    ("tex", "LaTeX", "LaTeX document"),
    ("html", "HTML", "HTML markup"),
    ("htm", "HTML", "HTML markup"),
    ("css", "CSS", "CSS styling"),
    ("scss", "SCSS", "SCSS styling"),
    ("sass", "Sass", "Sass styling"),
    ("less", "Less", "Less styling"),
    // Functional
    ("hs", "Haskell", "Haskell programming"),
    ("ml", "OCaml", "OCaml programming"),
    ("mli", "OCaml", "OCaml programming"),
    ("ex", "Elixir", "Elixir programming"),
    ("exs", "Elixir", "Elixir programming"),
    ("erl", "Erlang", "Erlang programming"),
    ("clj", "Clojure", "Clojure programming"),
    ("lisp", "Lisp", "Lisp programming"),
    ("el", "Emacs Lisp", "Emacs Lisp programming"),
    // Data / Query
    ("sql", "SQL", "SQL database queries"),
    ("graphql", "GraphQL", "GraphQL queries"),
    ("gql", "GraphQL", "GraphQL queries"),
    ("proto", "Protocol Buffers", "Protocol Buffers definition"),
    // DevOps / Infra
    ("tf", "Terraform", "Terraform infrastructure"),
    ("hcl", "HCL", "HCL configuration"),
    ("dockerfile", "Dockerfile", "Docker configuration"),
    ("nix", "Nix", "Nix configuration"),
    // Misc
    ("r", "R", "R programming"),
    ("jl", "Julia", "Julia programming"),
    ("lua", "Lua", "Lua programming"),
    ("dart", "Dart", "Dart programming"),
    ("v", "V", "V programming"),
    ("nim", "Nim", "Nim programming"),
    ("cr", "Crystal", "Crystal programming"),
];

/// Terminal app bundle IDs — suggest command mode for these.
const TERMINAL_APP_IDS: &[&str] = &[
    "com.googlecode.iterm2",
    "com.apple.Terminal",
    "org.alacritty",
    "io.warp.warpterm",
    "net.kovidgoyal.kitty",
    "com.github.wez.wezterm",
];

/// Analyze a window title to extract language and context information.
pub fn analyze_title(title: &str) -> TitleContext {
    let filename = extract_filename(title);

    if let Some(ref name) = filename {
        if let Some((ext, lang, prefix)) = lookup_extension(name) {
            let mode = if is_doc_extension(ext) {
                Some(DictationMode::Prose)
            } else {
                Some(DictationMode::Code)
            };
            return TitleContext {
                language: Some(lang.to_string()),
                extension: Some(ext.to_string()),
                filename: Some(name.clone()),
                prompt_prefix: Some(format!("{prefix}.")),
                suggested_mode: mode,
            };
        }
    }

    TitleContext {
        language: None,
        extension: None,
        filename,
        prompt_prefix: None,
        suggested_mode: None,
    }
}

/// Check if an app is a terminal emulator.
pub fn is_terminal_app(app_id: &str) -> bool {
    TERMINAL_APP_IDS.contains(&app_id)
}

/// Extract a filename from a window title.
///
/// Handles common editor title formats:
/// - `"filename.ext — App Name"` (VS Code, em dash)
/// - `"filename.ext - App Name"` (JetBrains, hyphen)
/// - `"filename.ext — Edited — App Name"` (multiple separators)
/// - `"~/path/to/filename.ext"` (terminals)
/// - `"App Name — filename.ext"` (some editors put filename last)
fn extract_filename(title: &str) -> Option<String> {
    let title = title.trim();
    if title.is_empty() {
        return None;
    }

    // Split on common title separators (em dash, en dash, hyphen with spaces)
    let segments: Vec<&str> = title
        .split(&['\u{2014}', '\u{2013}'][..]) // em dash, en dash
        .flat_map(|s| s.split(" - "))
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    // Check each segment for something that looks like a filename
    for segment in &segments {
        if let Some(name) = try_extract_filename_from_segment(segment) {
            return Some(name);
        }
    }

    // If no segment matched, try the whole title (e.g. terminal showing a path)
    try_extract_filename_from_segment(title)
}

/// Try to find a filename with a known extension in a title segment.
fn try_extract_filename_from_segment(segment: &str) -> Option<String> {
    // Handle paths: take the last path component (supports / and \)
    let candidate = segment
        .rsplit(&['/', '\\'][..])
        .next()
        .unwrap_or(segment)
        .trim();

    // Strip common prefixes/suffixes editors add
    let candidate = candidate
        .trim_start_matches("● ") // VS Code modified indicator
        .trim_start_matches("◉ ")
        .trim_start_matches("* ")
        .trim_end_matches(" [Modified]")
        .trim_end_matches(" [+]")
        .trim_end_matches(" •")
        .trim();

    // Must contain a dot for an extension
    if candidate.rfind('.').is_some() {
        // Only accept files with known programming/config extensions
        // to avoid false matches like "report.pdf" or "Mr. Smith"
        if lookup_extension(candidate).is_some() {
            return Some(candidate.to_string());
        }
    }

    // Handle extension-less known filenames
    let lower = candidate.to_lowercase();
    if matches!(
        lower.as_str(),
        "dockerfile" | "makefile" | "justfile" | "rakefile" | "gemfile" | "cmakelists.txt"
    ) {
        return Some(candidate.to_string());
    }

    None
}

/// Look up a filename's extension in the known language map.
fn lookup_extension(filename: &str) -> Option<(&'static str, &'static str, &'static str)> {
    // Handle "Dockerfile" and similar extensionless files
    let lower = filename.to_lowercase();
    if lower == "dockerfile" {
        return Some(("dockerfile", "Dockerfile", "Docker configuration"));
    }
    if lower == "makefile" || lower == "justfile" {
        return Some(("makefile", "Make", "Build system configuration"));
    }

    let ext = filename.rsplit('.').next()?.to_lowercase();
    EXTENSION_MAP
        .iter()
        .find(|(e, _, _)| *e == ext.as_str())
        .map(|&(e, l, p)| (e, l, p))
}

/// Check if an extension is for documentation/prose (should use prose mode).
fn is_doc_extension(ext: &str) -> bool {
    matches!(ext, "md" | "mdx" | "rst" | "tex" | "txt")
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- analyze_title --

    #[test]
    fn test_analyze_rust_file() {
        let ctx = analyze_title("auth.rs — Visual Studio Code");
        assert_eq!(ctx.language.as_deref(), Some("Rust"));
        assert_eq!(ctx.extension.as_deref(), Some("rs"));
        assert_eq!(ctx.filename.as_deref(), Some("auth.rs"));
        assert!(ctx.prompt_prefix.unwrap().contains("Rust"));
        assert_eq!(ctx.suggested_mode, Some(DictationMode::Code));
    }

    #[test]
    fn test_analyze_python_file() {
        let ctx = analyze_title("main.py - PyCharm");
        assert_eq!(ctx.language.as_deref(), Some("Python"));
        assert_eq!(ctx.extension.as_deref(), Some("py"));
        assert_eq!(ctx.filename.as_deref(), Some("main.py"));
        assert_eq!(ctx.suggested_mode, Some(DictationMode::Code));
    }

    #[test]
    fn test_analyze_typescript_react() {
        let ctx = analyze_title("App.tsx — WebStorm");
        assert_eq!(ctx.language.as_deref(), Some("TypeScript React"));
        assert_eq!(ctx.extension.as_deref(), Some("tsx"));
    }

    #[test]
    fn test_analyze_markdown_gets_prose_mode() {
        let ctx = analyze_title("README.md — Visual Studio Code");
        assert_eq!(ctx.language.as_deref(), Some("Markdown"));
        assert_eq!(ctx.suggested_mode, Some(DictationMode::Prose));
    }

    #[test]
    fn test_analyze_no_filename() {
        let ctx = analyze_title("Google Chrome");
        assert!(ctx.language.is_none());
        assert!(ctx.prompt_prefix.is_none());
        assert!(ctx.suggested_mode.is_none());
    }

    #[test]
    fn test_analyze_empty_title() {
        let ctx = analyze_title("");
        assert!(ctx.language.is_none());
        assert!(ctx.filename.is_none());
    }

    #[test]
    fn test_analyze_path_in_title() {
        let ctx = analyze_title("~/src/murmur/src/main.rs");
        assert_eq!(ctx.language.as_deref(), Some("Rust"));
        assert_eq!(ctx.filename.as_deref(), Some("main.rs"));
    }

    #[test]
    fn test_analyze_modified_indicator() {
        let ctx = analyze_title("● config.toml — Visual Studio Code");
        assert_eq!(ctx.language.as_deref(), Some("TOML"));
        assert_eq!(ctx.filename.as_deref(), Some("config.toml"));
    }

    #[test]
    fn test_analyze_dockerfile() {
        let ctx = analyze_title("Dockerfile — Visual Studio Code");
        assert_eq!(ctx.language.as_deref(), Some("Dockerfile"));
        assert_eq!(ctx.filename.as_deref(), Some("Dockerfile"));
    }

    #[test]
    fn test_analyze_multiple_separators() {
        let ctx = analyze_title("lib.rs — myproject — Visual Studio Code");
        assert_eq!(ctx.language.as_deref(), Some("Rust"));
        assert_eq!(ctx.filename.as_deref(), Some("lib.rs"));
    }

    #[test]
    fn test_analyze_go_file() {
        let ctx = analyze_title("handler.go — GoLand");
        assert_eq!(ctx.language.as_deref(), Some("Go"));
        assert_eq!(ctx.suggested_mode, Some(DictationMode::Code));
    }

    #[test]
    fn test_analyze_sql_file() {
        let ctx = analyze_title("schema.sql — DataGrip");
        assert_eq!(ctx.language.as_deref(), Some("SQL"));
        assert!(ctx.prompt_prefix.unwrap().contains("SQL"));
    }

    #[test]
    fn test_analyze_shell_script() {
        let ctx = analyze_title("deploy.sh — Terminal");
        assert_eq!(ctx.language.as_deref(), Some("Shell"));
    }

    // -- extract_filename --

    #[test]
    fn test_extract_filename_em_dash() {
        assert_eq!(
            extract_filename("file.rs — App"),
            Some("file.rs".to_string())
        );
    }

    #[test]
    fn test_extract_filename_hyphen() {
        assert_eq!(
            extract_filename("file.py - App"),
            Some("file.py".to_string())
        );
    }

    #[test]
    fn test_extract_filename_path() {
        assert_eq!(
            extract_filename("/Users/me/src/main.rs"),
            Some("main.rs".to_string())
        );
    }

    #[test]
    fn test_extract_filename_windows_path() {
        assert_eq!(
            extract_filename("C:\\Users\\me\\src\\main.rs"),
            Some("main.rs".to_string())
        );
    }

    #[test]
    fn test_extract_filename_none_for_no_extension() {
        assert!(extract_filename("Google Chrome").is_none());
    }

    #[test]
    fn test_extract_filename_ignores_unknown_extensions() {
        // Should not match non-programming extensions
        assert!(extract_filename("report.pdf - Preview").is_none());
        assert!(extract_filename("photo.jpg — Photos").is_none());
        assert!(extract_filename("document.docx - Word").is_none());
    }

    #[test]
    fn test_analyze_non_editor_app() {
        // Browser tabs, email, etc. should not produce false matches
        let ctx = analyze_title("GitHub - Pull Request #18 - Google Chrome");
        assert!(ctx.language.is_none());
        assert!(ctx.suggested_mode.is_none());
    }

    // -- is_terminal_app --

    #[test]
    fn test_is_terminal_app() {
        assert!(is_terminal_app("com.apple.Terminal"));
        assert!(is_terminal_app("com.googlecode.iterm2"));
        assert!(!is_terminal_app("com.microsoft.VSCode"));
    }

    // -- lookup_extension --

    #[test]
    fn test_lookup_known_extensions() {
        assert!(lookup_extension("file.rs").is_some());
        assert!(lookup_extension("file.py").is_some());
        assert!(lookup_extension("file.tsx").is_some());
        assert!(lookup_extension("file.go").is_some());
    }

    #[test]
    fn test_lookup_case_insensitive() {
        assert!(lookup_extension("FILE.RS").is_some());
        assert!(lookup_extension("Main.PY").is_some());
    }

    #[test]
    fn test_lookup_unknown_extension() {
        assert!(lookup_extension("file.xyz123").is_none());
    }

    // -- is_doc_extension --

    #[test]
    fn test_is_doc_extension() {
        assert!(is_doc_extension("md"));
        assert!(is_doc_extension("rst"));
        assert!(is_doc_extension("tex"));
        assert!(!is_doc_extension("rs"));
        assert!(!is_doc_extension("py"));
    }
}
