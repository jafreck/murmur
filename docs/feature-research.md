# Murmur Feature Research

## Current State

Murmur is a local, cross-platform voice dictation daemon (Rust + whisper.cpp). It
records audio via a global hotkey, transcribes locally with Whisper, and pastes text
into the active app via OS-level clipboard/keystrokes. Current features include
push-to-talk, open mic mode, spoken punctuation, streaming preview, model/language
selection, and hotkey remapping.

What it does **not** have: any awareness of the focused application, cursor position,
editor state, or document context.

---

## Part 1 — General STT Features Worth Implementing

### 1. Custom Vocabulary / Hotword Biasing

Whisper's `initial_prompt` parameter can be seeded with domain-specific terms to bias
the decoder toward expected words. Murmur could:

- Let users define a `vocabulary` list in config.json (project names, API names, etc.)
- Automatically inject those terms into Whisper's prompt at inference time
- Support per-project vocabulary files (e.g., `.murmur-vocab` in a repo root)

Research shows WER drops of ~30% on rare/domain words with prompt biasing alone
(arXiv 2410.18363). This is the lowest-effort, highest-impact improvement available.

### 2. Voice Commands (Beyond Dictation)

Move beyond pure transcription into programmable actions:

- **Text formatting**: "all caps", "title case", "camel case", "snake case"
- **Editing**: "select all", "undo that", "delete last word", "new line", "new paragraph"
- **Navigation**: "go to end", "go to start"
- **Punctuation mode toggle**: "punctuation on/off"
- **Corrections**: "scratch that" (delete last utterance), "replace X with Y"

Talon Voice demonstrates how powerful a command grammar can be for developers.
Murmur could start with a small built-in set and allow user-defined commands in config.

### 3. Output Formatting Modes

Different contexts need different output shapes:

- **Prose mode** (current default): capitalize first word, add period at end
- **Code mode**: no capitalization correction, preserve casing, insert as-is
- **Command mode**: interpret speech as commands rather than text
- **List mode**: each utterance becomes a bullet point

These could be toggled from the tray menu or via voice command.

### 4. Post-Processing Pipeline

Allow user-defined text transformations after transcription:

- Regex find/replace rules
- Auto-expand abbreviations
- Strip filler words ("um", "uh", "like")
- Custom scripts (pipe transcription through a shell command)

Config example:
```json
{
  "post_processing": [
    { "type": "strip_fillers" },
    { "type": "replace", "from": "my app", "to": "MyApp" },
    { "type": "shell", "command": "sed 's/foo/bar/g'" }
  ]
}
```

### 5. Multi-Destination Output

Currently murmur only pastes into the active app. Options:

- **File append**: write transcriptions to a log/journal file
- **Stdout/pipe mode**: run murmur as a CLI filter (stdin audio → stdout text)
- **Webhook/socket**: send transcription to a local server (IDE extension, etc.)
- **Named pipe / Unix socket**: for integration with other tools

### 6. Speaker Adaptation / Voice Profile

Whisper doesn't natively support speaker adaptation, but murmur could:

- Store recent transcription corrections to build a personal correction dictionary
- Use `initial_prompt` with the user's common phrasing style
- Auto-learn from "scratch that" + re-dictation pairs

### 7. Confidence Indicators

Surface Whisper's token-level log-probabilities:

- Highlight low-confidence words visually in a notification/overlay
- Offer alternatives for uncertain segments
- Auto-flag transcriptions below a confidence threshold for review

### 8. Audio Preprocessing

- Noise gate / noise suppression before sending to Whisper
- Voice activity detection (VAD) to trim silence and reduce inference time
- Echo cancellation for use during calls/meetings

### 9. Transcription History / Search

- Persist transcription text alongside audio recordings
- Provide a CLI or tray-accessible history viewer
- Full-text search across past dictations
- "Copy from history" feature

---

## Part 2 — Context-Aware Features (Leveraging Runtime State)

This is where murmur could differentiate itself from every other Whisper wrapper.

### A. Active Application Detection

**What to capture**: the bundle ID / process name / window title of the focused app.

**macOS**: `NSWorkspace.sharedWorkspace.frontmostApplication` (via objc2 or
cocoa crate). **Linux**: `xdotool getactivewindow getwindowpid` or D-Bus.
**Windows**: `GetForegroundWindow` + `GetWindowText`.

**How to use it**:

1. **Per-app vocabulary biasing**: When VS Code is focused, seed Whisper's prompt
   with programming keywords. When Slack is focused, seed with teammate names.
   Config:
   ```json
   {
     "app_contexts": {
       "com.microsoft.VSCode": {
         "vocabulary": ["useState", "async", "impl", "struct"],
         "mode": "code"
       },
       "com.tinyspeck.slackmacgap": {
         "vocabulary": ["@channel", "standup", "PR"],
         "mode": "prose"
       }
     }
   }
   ```

2. **Auto-switch output mode**: code mode in terminals/editors, prose mode in email,
   command mode in a launcher.

3. **App-specific voice commands**: "build project" only works in IDE, "send message"
   only in chat apps.

### B. Cursor / Caret Context

**What to capture**: the text surrounding the cursor position.

**macOS**: Accessibility API (`AXUIElement`) can read the focused text field's value
and selected range. **Windows**: UI Automation (`IUIAutomationTextPattern`).
**Linux**: AT-SPI2 (limited support).

**How to use it**:

1. **Contextual prompt biasing**: Read the last ~50 characters before the cursor and
   include them in Whisper's `initial_prompt`. This gives the model sentence-level
   context, dramatically improving accuracy for continuations. Example: if the cursor
   follows "The function returns a", Whisper is far more likely to correctly hear
   "boolean" vs. "bull Ian".

2. **Smart capitalization**: If the cursor follows a period or newline, capitalize the
   first word. If mid-sentence, don't.

3. **Inline completion style**: Instead of pasting as a block, insert text that flows
   naturally from what's already there.

4. **Code-aware insertion**: If the cursor is inside a string literal, don't
   auto-capitalize. If inside a comment, use prose mode. If on a blank line, use code
   mode.

### C. Editor Integration (Deep Context)

Go beyond generic accessibility APIs with editor-specific extensions:

1. **VS Code Extension**: A companion extension that exposes:
   - Current file language (Python, Rust, TypeScript, etc.)
   - Cursor position (line, column)
   - Surrounding code (function name, class, scope)
   - Open file list / project name
   - Symbols in scope (for vocabulary biasing)
   
   Communication via local WebSocket or Unix socket.

2. **Language-aware vocabulary**: Automatically extract identifiers from the current
   file's AST and feed them as vocabulary hints. When editing `auth.rs`, Whisper gets
   biased toward "authenticate", "token", "bearer", "session" — words actually in the
   file.

3. **Scope-aware mode switching**: Inside a doc-comment → prose mode. Inside a
   function body → code mode. Inside a string → verbatim mode.

### D. Window Title / Document Context

Window titles often contain the filename or document name:

- Extract the current filename from the title bar
- Look up project-level vocabulary files based on the workspace path
- Detect if the user is in a terminal (adjust for command dictation)

### E. System State Awareness

- **Clipboard contents**: If the user just copied something, it might be relevant
  context for the next dictation
- **Recently typed text**: Monitor keystrokes (already have rdev) to maintain a
  rolling context window of what was recently typed, feeding it to Whisper's prompt
- **Notification state**: Detect if in a meeting (calendar integration) to switch
  to meeting-optimized vocabulary

---

## Part 3 — Architectural Considerations

### Context Provider Trait

Design a pluggable context system:

```rust
trait ContextProvider: Send + Sync {
    fn get_context(&self) -> Context;
}

struct Context {
    app_id: Option<String>,
    window_title: Option<String>,
    surrounding_text: Option<String>,
    file_language: Option<String>,
    vocabulary_hints: Vec<String>,
    suggested_mode: Option<DictationMode>,
}
```

Platform-specific implementations provide different levels of context. The
transcription pipeline consumes whatever context is available to:
1. Build Whisper's `initial_prompt`
2. Select output mode
3. Choose post-processing rules

### Privacy Considerations

Context-awareness means reading screen content. Murmur should:
- Be transparent about what context is captured
- Never transmit context off-device (already local-only)
- Allow users to disable context features entirely
- Exclude sensitive apps (password managers, banking) from context capture

### Incremental Adoption Path

1. **v1**: Custom vocabulary in config (no OS APIs needed)
2. **v2**: Active app detection + per-app profiles
3. **v3**: Cursor context via accessibility APIs
4. **v4**: Editor extension for deep integration

---

## Key References

- Contextual Biasing for Whisper: arXiv 2410.18363
- WhisperBiasing (TCPGen): github.com/BriansIDP/WhisperBiasing
- CB-Whisper keyword spotting: aclanthology.org/2024.lrec-main.262
- Speech-Aware Long Context Pruning: arXiv 2511.11139
- Context-Aware ASR Prompting: INTERSPEECH 2024
- Talon Voice + Cursorless: talonvoice.com, cursorless.org
- Patent CA2839265A1: context-aware recognition model switching
