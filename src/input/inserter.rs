use anyhow::{Context, Result};
use arboard::{Clipboard, ImageData};
use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use std::thread;
use std::time::Duration;

/// Saved clipboard content for restore after paste.
enum SavedClipboard {
    Text(String),
    Image(ImageData<'static>),
    Empty,
}

pub struct TextInserter;

/// The modifier key used for paste: Cmd on macOS, Ctrl elsewhere.
pub fn paste_modifier() -> Key {
    #[cfg(target_os = "macos")]
    {
        Key::Meta
    }
    #[cfg(not(target_os = "macos"))]
    {
        Key::Control
    }
}

impl TextInserter {
    pub fn insert(text: &str) -> Result<()> {
        let mut clipboard = Clipboard::new().context("Failed to access clipboard")?;

        // Save current clipboard contents (try text first, then image)
        let saved = if let Ok(text) = clipboard.get_text() {
            SavedClipboard::Text(text)
        } else if let Ok(img) = clipboard.get_image() {
            SavedClipboard::Image(img.to_owned_img())
        } else {
            SavedClipboard::Empty
        };

        // Set transcription text
        clipboard
            .set_text(text.to_string())
            .context("Failed to set clipboard text")?;

        // Small delay for clipboard to settle
        thread::sleep(Duration::from_millis(50));

        // Simulate paste
        Self::simulate_paste()?;

        // Wait for the target application to consume the paste before restoring.
        // Slow apps (Electron, remote desktop) may need longer than this, but
        // there is no reliable cross-platform way to detect paste completion.
        thread::sleep(Duration::from_millis(400));
        match saved {
            SavedClipboard::Text(prev) => {
                let _ = clipboard.set_text(prev);
            }
            SavedClipboard::Image(img) => {
                let _ = clipboard.set_image(img);
            }
            SavedClipboard::Empty => {}
        }

        Ok(())
    }

    /// Delete `backspace_count` characters, then type `text`.
    /// Used for streaming revisions where Whisper changes earlier output.
    pub fn replace(backspace_count: usize, text: &str) -> Result<()> {
        let mut enigo =
            Enigo::new(&Settings::default()).map_err(|e| anyhow::anyhow!("Enigo init: {e}"))?;

        // Send backspaces to delete the revised portion
        for _ in 0..backspace_count {
            enigo
                .key(Key::Backspace, Direction::Click)
                .map_err(|e| anyhow::anyhow!("Backspace failed: {e}"))?;
        }

        if !text.is_empty() {
            // Type the replacement text directly (no clipboard involvement)
            enigo
                .text(text)
                .map_err(|e| anyhow::anyhow!("Text input failed: {e}"))?;
        }

        Ok(())
    }

    fn simulate_paste() -> Result<()> {
        let mut enigo =
            Enigo::new(&Settings::default()).map_err(|e| anyhow::anyhow!("Enigo init: {e}"))?;

        let modifier = paste_modifier();

        enigo
            .key(modifier, Direction::Press)
            .map_err(|e| anyhow::anyhow!("Key press failed: {e}"))?;
        enigo
            .key(Key::Unicode('v'), Direction::Click)
            .map_err(|e| anyhow::anyhow!("Key click failed: {e}"))?;
        enigo
            .key(modifier, Direction::Release)
            .map_err(|e| anyhow::anyhow!("Key release failed: {e}"))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paste_modifier_returns_expected() {
        let m = paste_modifier();
        #[cfg(target_os = "macos")]
        assert_eq!(m, Key::Meta);
        #[cfg(not(target_os = "macos"))]
        assert_eq!(m, Key::Control);
    }
}
