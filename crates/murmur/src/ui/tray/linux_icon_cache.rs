//! Linux icon cache: pre-writes PNG files to a temp directory so that tray
//! icon updates just point AppIndicator at an existing file instead of
//! re-encoding and re-writing on every `set_icon()` call.

use std::path::{Path, PathBuf};

use anyhow::Result;

use super::{tint_pixels, DecodedIcon, StateColors, TrayState, PULSE_ALPHA_MIN, PULSE_FRAME_COUNT};

fn encode_png(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    {
        let mut encoder = png::Encoder::new(std::io::Cursor::new(&mut buf), width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .map_err(|e| anyhow::anyhow!("PNG encode header: {e}"))?;
        writer
            .write_image_data(rgba)
            .map_err(|e| anyhow::anyhow!("PNG encode data: {e}"))?;
    }
    Ok(buf)
}

/// Write a tinted icon to disk and return the stem (path without `.png`).
fn write_icon(
    dir: &Path,
    name: &str,
    decoded: &DecodedIcon,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
) -> Result<String> {
    let rgba = tint_pixels(&decoded.pixels, decoded.stride, r, g, b, a);
    let png_data = encode_png(&rgba, decoded.width, decoded.height)?;
    let file_path = dir.join(format!("{name}.png"));
    std::fs::write(&file_path, &png_data)?;
    // AppIndicator wants the full path without the .png extension.
    Ok(file_path.with_extension("").to_string_lossy().into_owned())
}

/// Pre-written PNG icon cache for Linux/AppIndicator.
pub struct LinuxIconCache {
    /// The directory containing all pre-written PNGs.
    dir: PathBuf,
    /// AppIndicator theme path (the parent directory as a string).
    theme_path: String,
    idle: String,
    recording: String,
    transcribing: String,
    loading: String,
    loading_pulse: Vec<String>,
    downloading_pulse: Vec<String>,
}

impl LinuxIconCache {
    /// Build the cache: write all icon PNGs to a temp directory.
    pub fn new(decoded: &DecodedIcon, colors: &StateColors) -> Result<Self> {
        let dir = dirs::runtime_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join("murmur-tray-cache");
        std::fs::create_dir_all(&dir)?;

        let theme_path = dir.to_string_lossy().into_owned();

        let c = &colors.idle;
        let idle = write_icon(&dir, "idle", decoded, c.0, c.1, c.2, c.3)?;
        let c = &colors.recording;
        let recording = write_icon(&dir, "recording", decoded, c.0, c.1, c.2, c.3)?;
        let c = &colors.transcribing;
        let transcribing = write_icon(&dir, "transcribing", decoded, c.0, c.1, c.2, c.3)?;
        let c = &colors.loading;
        let loading = write_icon(&dir, "loading", decoded, c.0, c.1, c.2, c.3)?;

        let loading_pulse = Self::write_pulse_frames(&dir, "loading", decoded, colors.loading)?;
        let downloading_pulse =
            Self::write_pulse_frames(&dir, "downloading", decoded, colors.transcribing)?;

        Ok(Self {
            dir,
            theme_path,
            idle,
            recording,
            transcribing,
            loading,
            loading_pulse,
            downloading_pulse,
        })
    }

    fn write_pulse_frames(
        dir: &Path,
        prefix: &str,
        decoded: &DecodedIcon,
        base: (u8, u8, u8, u8),
    ) -> Result<Vec<String>> {
        (0..PULSE_FRAME_COUNT)
            .map(|i| {
                let phase_angle = (i as f64 / PULSE_FRAME_COUNT as f64) * std::f64::consts::TAU;
                let phase = phase_angle.sin();
                let scale = PULSE_ALPHA_MIN + (1.0 - PULSE_ALPHA_MIN) * (phase + 1.0) / 2.0;
                let a = (base.3 as f64 * scale).round().min(255.0) as u8;
                let name = format!("{prefix}-pulse-{i:02}");
                write_icon(dir, &name, decoded, base.0, base.1, base.2, a)
            })
            .collect()
    }

    /// Get the icon stem for a given state.
    fn icon_for_state(&self, state: &TrayState) -> &str {
        match state {
            TrayState::Idle => &self.idle,
            TrayState::Recording | TrayState::Error => &self.recording,
            TrayState::Transcribing | TrayState::Downloading => &self.transcribing,
            TrayState::Loading => &self.loading,
        }
    }

    /// Set the tray icon for a state by pointing AppIndicator at a
    /// pre-written PNG — no encoding or disk writes.
    ///
    /// # Safety
    ///
    /// The `indicator` pointer must be valid and derived from the same
    /// `TrayIcon` that is alive for the duration of this call.
    pub unsafe fn set_icon_for_state(
        &self,
        indicator: *const libappindicator::AppIndicator,
        state: &TrayState,
    ) {
        let name = self.icon_for_state(state);
        self.apply(indicator, name);
    }

    /// Set a pulse animation frame.
    ///
    /// # Safety
    ///
    /// Same as [`set_icon_for_state`].
    pub unsafe fn set_pulse_frame(
        &self,
        indicator: *const libappindicator::AppIndicator,
        state: &TrayState,
        frame_idx: usize,
    ) -> bool {
        let frames = match state {
            TrayState::Loading => &self.loading_pulse,
            TrayState::Downloading => &self.downloading_pulse,
            _ => return false,
        };
        if let Some(name) = frames.get(frame_idx) {
            self.apply(indicator, name);
            true
        } else {
            false
        }
    }

    unsafe fn apply(&self, indicator: *const libappindicator::AppIndicator, icon_name: &str) {
        let indicator = &mut *(indicator as *mut libappindicator::AppIndicator);
        indicator.set_icon_theme_path(&self.theme_path);
        indicator.set_icon_full(icon_name, "murmur tray icon");
    }
}

impl Drop for LinuxIconCache {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_png_produces_valid_png() {
        let rgba = [255, 0, 0, 255].repeat(4);
        let png_data = encode_png(&rgba, 2, 2).unwrap();
        assert_eq!(&png_data[..4], &[137, 80, 78, 71]);
    }

    #[test]
    fn encode_png_roundtrip() {
        let rgba: Vec<u8> = [100, 150, 200, 255].repeat(9);
        let png_data = encode_png(&rgba, 3, 3).unwrap();
        let decoder = png::Decoder::new(std::io::Cursor::new(&png_data));
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!(info.width, 3);
        assert_eq!(info.height, 3);
        assert_eq!(&buf[..info.buffer_size()], &rgba[..]);
    }

    #[test]
    fn write_icon_creates_file() {
        let dir = std::env::temp_dir().join("murmur-test-write-icon");
        let _ = std::fs::create_dir_all(&dir);
        let decoded = DecodedIcon {
            pixels: vec![255; 16], // 2×2 RGBA
            stride: 4,
            width: 2,
            height: 2,
        };
        let stem = write_icon(&dir, "test-icon", &decoded, 255, 0, 0, 255).unwrap();
        assert!(std::path::Path::new(&format!("{stem}.png")).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_icon_stem_has_no_extension() {
        let dir = std::env::temp_dir().join("murmur-test-stem");
        let _ = std::fs::create_dir_all(&dir);
        let decoded = DecodedIcon {
            pixels: vec![128; 16],
            stride: 4,
            width: 2,
            height: 2,
        };
        let stem = write_icon(&dir, "check", &decoded, 0, 255, 0, 200).unwrap();
        assert!(!stem.ends_with(".png"));
        assert!(stem.ends_with("check"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
