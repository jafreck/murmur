use anyhow::{Context, Result};
use arboard::Clipboard;
use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use std::thread;
use std::time::Duration;

pub struct TextInserter;

impl TextInserter {
    pub fn insert(text: &str) -> Result<()> {
        let mut clipboard = Clipboard::new().context("Failed to access clipboard")?;

        // Save current clipboard contents
        let saved = clipboard.get_text().ok();

        // Set transcription text
        clipboard
            .set_text(text.to_string())
            .context("Failed to set clipboard text")?;

        // Small delay for clipboard to settle
        thread::sleep(Duration::from_millis(50));

        // Simulate paste
        Self::simulate_paste()?;

        // Restore previous clipboard after paste completes
        thread::sleep(Duration::from_millis(150));
        if let Some(prev) = saved {
            let _ = clipboard.set_text(prev);
        }

        Ok(())
    }

    fn simulate_paste() -> Result<()> {
        let mut enigo =
            Enigo::new(&Settings::default()).map_err(|e| anyhow::anyhow!("Enigo init: {e}"))?;

        #[cfg(target_os = "macos")]
        let modifier = Key::Meta;
        #[cfg(not(target_os = "macos"))]
        let modifier = Key::Control;

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
