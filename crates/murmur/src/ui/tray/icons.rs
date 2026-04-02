//! Icon decoding, tinting, animation, and appearance detection for the tray UI.

use anyhow::Result;
use tray_icon::Icon;

use super::TrayState;

/// How often (in milliseconds) the recording/transcribing animation advances.
pub(super) const ANIMATION_FRAME_INTERVAL_MS: u128 = 40;

/// Number of embedded animation frames per state.
pub(super) const ANIMATION_FRAME_COUNT: usize = 28;

/// Duration of one full animation cycle in seconds.
pub(super) const ANIMATION_PERIOD_SECS: f64 =
    ANIMATION_FRAME_COUNT as f64 * ANIMATION_FRAME_INTERVAL_MS as f64 / 1000.0;

/// Whether the given state should show an animation.
pub(super) fn is_animating_state(state: &TrayState) -> bool {
    matches!(state, TrayState::Recording | TrayState::Transcribing)
}

/// Which icon variant to display for a given tray state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IconKey {
    Idle,
    Recording,
    Transcribing,
    Loading,
}

/// Map a [`TrayState`] to the icon variant that should be displayed.
pub fn state_icon_key(state: &TrayState) -> IconKey {
    match state {
        TrayState::Idle => IconKey::Idle,
        TrayState::Recording | TrayState::Error => IconKey::Recording,
        TrayState::Transcribing | TrayState::Downloading => IconKey::Transcribing,
        TrayState::Loading => IconKey::Loading,
    }
}

/// Returns true if not enough time has passed since the last animation frame.
pub fn should_throttle_frame(last_frame_elapsed_ms: u128, interval_ms: u128) -> bool {
    last_frame_elapsed_ms < interval_ms
}

/// Compute the pulse alpha scale for the current animation frame.
///
/// Given elapsed time since animation start, the pulse period, and the
/// minimum alpha, returns a scale factor in [`alpha_min`, 1.0] suitable
/// for modulating icon opacity.
#[deprecated(note = "Pulse animation has been removed; kept for API compatibility")]
pub fn compute_pulse_alpha(elapsed_secs: f64, period_secs: f64, alpha_min: f64) -> f64 {
    let phase = (elapsed_secs * std::f64::consts::TAU / period_secs).sin();
    alpha_min + (1.0 - alpha_min) * (phase + 1.0) / 2.0
}

/// The murmur icon PNG, embedded at compile time.
const ICON_PNG: &[u8] = include_bytes!("../../../assets/icons/murmur.png");

/// Pre-decoded PNG pixel data so we can tint without re-decoding.
#[derive(Clone)]
pub struct DecodedIcon {
    pub pixels: Vec<u8>,
    pub stride: usize,
    pub width: u32,
    pub height: u32,
}

/// Decode the embedded icon PNG once and return the raw pixel data.
pub fn decode_icon_png() -> Result<DecodedIcon> {
    decode_png_bytes(ICON_PNG)
}

/// Decode arbitrary PNG bytes into raw pixel data.
pub fn decode_png_bytes(data: &[u8]) -> Result<DecodedIcon> {
    let cursor = std::io::Cursor::new(data);
    let decoder = png::Decoder::new(cursor);
    let mut reader = decoder
        .read_info()
        .map_err(|e| anyhow::anyhow!("PNG decode: {e}"))?;
    let mut buf = vec![0u8; reader.output_buffer_size().expect("PNG info missing")];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| anyhow::anyhow!("PNG frame: {e}"))?;
    let stride = match info.color_type {
        png::ColorType::Rgba => 4,
        png::ColorType::Rgb => 3,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Grayscale => 1,
        other => anyhow::bail!("Unsupported PNG colour type: {other:?}"),
    };
    Ok(DecodedIcon {
        pixels: buf[..info.buffer_size()].to_vec(),
        stride,
        width: info.width,
        height: info.height,
    })
}

/// Build an Icon by tinting pre-decoded pixel data (no PNG decoding).
pub(super) fn make_icon_from_decoded(
    decoded: &DecodedIcon,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
) -> Result<Icon> {
    let rgba = tint_pixels(&decoded.pixels, decoded.stride, r, g, b, a);
    Icon::from_rgba(rgba, decoded.width, decoded.height)
        .map_err(|e| anyhow::anyhow!("Icon error: {e}"))
}

/// Embedded recording animation frame PNGs (extracted from GIF).
const RECORDING_FRAME_PNGS: &[&[u8]] = &[
    include_bytes!("../../../assets/icons/recording_00.png"),
    include_bytes!("../../../assets/icons/recording_01.png"),
    include_bytes!("../../../assets/icons/recording_02.png"),
    include_bytes!("../../../assets/icons/recording_03.png"),
    include_bytes!("../../../assets/icons/recording_04.png"),
    include_bytes!("../../../assets/icons/recording_05.png"),
    include_bytes!("../../../assets/icons/recording_06.png"),
    include_bytes!("../../../assets/icons/recording_07.png"),
    include_bytes!("../../../assets/icons/recording_08.png"),
    include_bytes!("../../../assets/icons/recording_09.png"),
    include_bytes!("../../../assets/icons/recording_10.png"),
    include_bytes!("../../../assets/icons/recording_11.png"),
    include_bytes!("../../../assets/icons/recording_12.png"),
    include_bytes!("../../../assets/icons/recording_13.png"),
    include_bytes!("../../../assets/icons/recording_14.png"),
    include_bytes!("../../../assets/icons/recording_15.png"),
    include_bytes!("../../../assets/icons/recording_16.png"),
    include_bytes!("../../../assets/icons/recording_17.png"),
    include_bytes!("../../../assets/icons/recording_18.png"),
    include_bytes!("../../../assets/icons/recording_19.png"),
    include_bytes!("../../../assets/icons/recording_20.png"),
    include_bytes!("../../../assets/icons/recording_21.png"),
    include_bytes!("../../../assets/icons/recording_22.png"),
    include_bytes!("../../../assets/icons/recording_23.png"),
    include_bytes!("../../../assets/icons/recording_24.png"),
    include_bytes!("../../../assets/icons/recording_25.png"),
    include_bytes!("../../../assets/icons/recording_26.png"),
    include_bytes!("../../../assets/icons/recording_27.png"),
];

/// Decode all embedded recording frame PNGs.
pub fn decode_recording_frames() -> Vec<DecodedIcon> {
    RECORDING_FRAME_PNGS
        .iter()
        .filter_map(|data| decode_png_bytes(data).ok())
        .collect()
}

/// Build recording animation icons from embedded multi-frame PNGs.
pub(super) fn build_recording_animation_frames(color: (u8, u8, u8, u8)) -> Vec<Icon> {
    decode_recording_frames()
        .iter()
        .filter_map(|decoded| {
            make_icon_from_decoded(decoded, color.0, color.1, color.2, color.3).ok()
        })
        .collect()
}

/// Embedded transcribing animation frame PNGs (extracted from APNG).
const TRANSCRIBING_FRAME_PNGS: &[&[u8]] = &[
    include_bytes!("../../../assets/icons/transcribing_00.png"),
    include_bytes!("../../../assets/icons/transcribing_01.png"),
    include_bytes!("../../../assets/icons/transcribing_02.png"),
    include_bytes!("../../../assets/icons/transcribing_03.png"),
    include_bytes!("../../../assets/icons/transcribing_04.png"),
    include_bytes!("../../../assets/icons/transcribing_05.png"),
    include_bytes!("../../../assets/icons/transcribing_06.png"),
    include_bytes!("../../../assets/icons/transcribing_07.png"),
    include_bytes!("../../../assets/icons/transcribing_08.png"),
    include_bytes!("../../../assets/icons/transcribing_09.png"),
    include_bytes!("../../../assets/icons/transcribing_10.png"),
    include_bytes!("../../../assets/icons/transcribing_11.png"),
    include_bytes!("../../../assets/icons/transcribing_12.png"),
    include_bytes!("../../../assets/icons/transcribing_13.png"),
    include_bytes!("../../../assets/icons/transcribing_14.png"),
    include_bytes!("../../../assets/icons/transcribing_15.png"),
    include_bytes!("../../../assets/icons/transcribing_16.png"),
    include_bytes!("../../../assets/icons/transcribing_17.png"),
    include_bytes!("../../../assets/icons/transcribing_18.png"),
    include_bytes!("../../../assets/icons/transcribing_19.png"),
    include_bytes!("../../../assets/icons/transcribing_20.png"),
    include_bytes!("../../../assets/icons/transcribing_21.png"),
    include_bytes!("../../../assets/icons/transcribing_22.png"),
    include_bytes!("../../../assets/icons/transcribing_23.png"),
    include_bytes!("../../../assets/icons/transcribing_24.png"),
    include_bytes!("../../../assets/icons/transcribing_25.png"),
    include_bytes!("../../../assets/icons/transcribing_26.png"),
    include_bytes!("../../../assets/icons/transcribing_27.png"),
];

/// Decode all embedded transcribing frame PNGs.
pub fn decode_transcribing_frames() -> Vec<DecodedIcon> {
    TRANSCRIBING_FRAME_PNGS
        .iter()
        .filter_map(|data| decode_png_bytes(data).ok())
        .collect()
}

/// Build transcribing animation icons from embedded multi-frame PNGs.
pub(super) fn build_transcribing_animation_frames(color: (u8, u8, u8, u8)) -> Vec<Icon> {
    decode_transcribing_frames()
        .iter()
        .filter_map(|decoded| {
            make_icon_from_decoded(decoded, color.0, color.1, color.2, color.3).ok()
        })
        .collect()
}

/// Build an icon by decoding the embedded PNG and tinting (used in tests).
#[cfg(test)]
fn make_icon(r: u8, g: u8, b: u8, a: u8) -> Result<Icon> {
    let (rgba, width, height) = tint_png_rgba(ICON_PNG, r, g, b, a)?;
    Icon::from_rgba(rgba, width, height).map_err(|e| anyhow::anyhow!("Icon error: {e}"))
}

/// RGBA colour for each tray state, tuned for dark and light menu bars.
pub(super) struct StateColors {
    pub(super) idle: (u8, u8, u8, u8),
    pub(super) recording: (u8, u8, u8, u8),
    pub(super) transcribing: (u8, u8, u8, u8),
    pub(super) loading: (u8, u8, u8, u8),
}

impl StateColors {
    pub(super) fn for_appearance(dark: bool) -> Self {
        if dark {
            // Light icons for dark menu bar
            Self {
                idle: (255, 255, 255, 200),
                recording: (255, 80, 80, 230),
                transcribing: (255, 210, 50, 220),
                loading: (180, 180, 180, 140),
            }
        } else {
            // Dark icons for light menu bar
            Self {
                idle: (30, 80, 200, 220),
                recording: (200, 30, 30, 230),
                transcribing: (180, 140, 0, 220),
                loading: (100, 100, 100, 160),
            }
        }
    }
}

/// Detect whether the macOS menu bar is using dark mode.
#[cfg(target_os = "macos")]
pub(super) fn is_dark_mode() -> bool {
    use std::process::Command;
    Command::new("defaults")
        .args(["read", "-g", "AppleInterfaceStyle"])
        .output()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .eq_ignore_ascii_case("dark")
        })
        .unwrap_or(false)
}

/// Detect dark mode on Windows via the registry.
#[cfg(target_os = "windows")]
pub(super) fn is_dark_mode() -> bool {
    use std::process::Command;
    // AppsUseLightTheme: 0 = dark, 1 = light
    Command::new("reg")
        .args([
            "query",
            r"HKCU\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
            "/v",
            "AppsUseLightTheme",
        ])
        .output()
        .map(|o| {
            let out = String::from_utf8_lossy(&o.stdout);
            out.contains("0x0")
        })
        .unwrap_or(true)
}

/// Detect dark mode on Linux via common desktop environment hints.
#[cfg(target_os = "linux")]
pub(super) fn is_dark_mode() -> bool {
    use std::process::Command;
    // Try GNOME color-scheme first
    if let Ok(output) = Command::new("gsettings")
        .args(["get", "org.gnome.desktop.interface", "color-scheme"])
        .output()
    {
        let scheme = String::from_utf8_lossy(&output.stdout).to_lowercase();
        if scheme.contains("dark") {
            return true;
        }
        if scheme.contains("light") || scheme.contains("default") {
            return false;
        }
    }
    // Fall back to GTK theme name
    if let Ok(output) = Command::new("gsettings")
        .args(["get", "org.gnome.desktop.interface", "gtk-theme"])
        .output()
    {
        let theme = String::from_utf8_lossy(&output.stdout).to_lowercase();
        if theme.contains("dark") {
            return true;
        }
    }
    // Default to dark (bright icons) when detection fails
    true
}

/// Decode a PNG and tint its pixels.
pub fn tint_png_rgba(png_data: &[u8], r: u8, g: u8, b: u8, a: u8) -> Result<(Vec<u8>, u32, u32)> {
    let cursor = std::io::Cursor::new(png_data);
    let decoder = png::Decoder::new(cursor);
    let mut reader = decoder
        .read_info()
        .map_err(|e| anyhow::anyhow!("PNG decode: {e}"))?;
    let mut buf = vec![0u8; reader.output_buffer_size().expect("PNG info missing")];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| anyhow::anyhow!("PNG frame: {e}"))?;
    let width = info.width;
    let height = info.height;
    let src = &buf[..info.buffer_size()];

    let stride = match info.color_type {
        png::ColorType::Rgba => 4,
        png::ColorType::Rgb => 3,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Grayscale => 1,
        other => anyhow::bail!("Unsupported PNG colour type: {other:?}"),
    };

    let rgba = tint_pixels(src, stride, r, g, b, a);
    Ok((rgba, width, height))
}

/// Apply a colour tint to raw pixel data.
pub fn tint_pixels(src: &[u8], stride: usize, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
    let pixel_count = src.len() / stride;
    let mut rgba = Vec::with_capacity(pixel_count * 4);

    for pixel in src.chunks_exact(stride) {
        // For RGBA the icon is an alpha mask: replace RGB with the tint
        // colour and combine the source alpha with the tint alpha.
        // For greyscale formats the luminance modulates the tint intensity.
        let (lum, pa) = match stride {
            4 => (255u16, pixel[3]),
            3 => (pixel[0] as u16, 255),
            2 => (pixel[0] as u16, pixel[1]),
            1 => (pixel[0] as u16, 255),
            _ => (0u16, 0),
        };

        if pa == 0 {
            rgba.extend_from_slice(&[0, 0, 0, 0]);
        } else {
            let tr = ((r as u16 * lum) / 255u16).min(255) as u8;
            let tg = ((g as u16 * lum) / 255u16).min(255) as u8;
            let tb = ((b as u16 * lum) / 255u16).min(255) as u8;
            let ta = ((a as u16 * pa as u16) / 255u16).min(255) as u8;
            rgba.extend_from_slice(&[tr, tg, tb, ta]);
        }
    }

    rgba
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- tint_pixels --

    #[test]
    fn tint_pixels_rgba_opaque() {
        let src = [255u8, 255, 255, 255];
        let result = tint_pixels(&src, 4, 255, 0, 0, 255);
        assert_eq!(result, vec![255, 0, 0, 255]);
    }

    #[test]
    fn tint_pixels_rgba_transparent() {
        let src = [255u8, 255, 255, 0];
        let result = tint_pixels(&src, 4, 255, 0, 0, 255);
        assert_eq!(result, vec![0, 0, 0, 0]);
    }

    #[test]
    fn tint_pixels_rgba_half_lum() {
        // RGBA stride treats the icon as an alpha mask: source RGB is
        // ignored and replaced by the tint colour.
        let src = [128u8, 0, 0, 255];
        let result = tint_pixels(&src, 4, 255, 255, 255, 255);
        assert_eq!(result, vec![255, 255, 255, 255]);
    }

    #[test]
    fn tint_pixels_rgb_stride3() {
        let src = [200u8, 100, 50];
        let result = tint_pixels(&src, 3, 255, 128, 64, 200);
        assert_eq!(result.len(), 4);
        assert_eq!(result[3], 200);
    }

    #[test]
    fn tint_pixels_grayscale_alpha_stride2() {
        let src = [128u8, 200];
        let result = tint_pixels(&src, 2, 100, 200, 50, 255);
        assert_eq!(result.len(), 4);
        assert_eq!(result[3], 200);
    }

    #[test]
    fn tint_pixels_grayscale_stride1() {
        let src = [255u8];
        let result = tint_pixels(&src, 1, 100, 200, 50, 128);
        assert_eq!(result, vec![100, 200, 50, 128]);
    }

    #[test]
    fn tint_pixels_unknown_stride() {
        let src = [1u8, 2, 3, 4, 5];
        let result = tint_pixels(&src, 5, 255, 255, 255, 255);
        assert_eq!(result, vec![0, 0, 0, 0]);
    }

    #[test]
    fn tint_pixels_black_input() {
        // RGBA alpha-mask: even black source pixels get replaced with the
        // tint colour (alpha is preserved).
        let src = [0u8, 0, 0, 255];
        let result = tint_pixels(&src, 4, 255, 255, 255, 255);
        assert_eq!(result[0..3], [255, 255, 255]);
    }

    #[test]
    fn tint_pixels_empty() {
        assert!(tint_pixels(&[], 4, 255, 0, 0, 255).is_empty());
    }

    // -- tint_png_rgba --

    #[test]
    fn tint_png_rgba_embedded_icon() {
        let (rgba, w, h) = tint_png_rgba(ICON_PNG, 100, 150, 255, 200).unwrap();
        assert!(w > 0 && h > 0);
        assert_eq!(rgba.len(), (w * h * 4) as usize);
    }

    #[test]
    fn tint_png_rgba_different_colors() {
        let (r1, _, _) = tint_png_rgba(ICON_PNG, 255, 0, 0, 255).unwrap();
        let (r2, _, _) = tint_png_rgba(ICON_PNG, 0, 0, 255, 255).unwrap();
        assert_ne!(r1, r2);
    }

    #[test]
    fn tint_png_rgba_invalid() {
        assert!(tint_png_rgba(b"not a png", 255, 0, 0, 255).is_err());
    }

    // -- decode_icon_png / make_icon_from_decoded --

    #[test]
    fn decode_icon_png_succeeds() {
        let decoded = decode_icon_png().unwrap();
        assert!(decoded.width > 0 && decoded.height > 0);
        assert_eq!(
            decoded.pixels.len(),
            (decoded.width as usize * decoded.height as usize) * decoded.stride
        );
    }

    #[test]
    fn make_icon_from_decoded_matches_make_icon() {
        let decoded = decode_icon_png().unwrap();
        let from_decoded = make_icon_from_decoded(&decoded, 255, 80, 80, 230);
        let from_raw = make_icon(255, 80, 80, 230);
        assert!(from_decoded.is_ok());
        assert!(from_raw.is_ok());
    }

    // -- dark/light mode --

    #[test]
    fn is_dark_mode_does_not_panic() {
        let _ = is_dark_mode();
    }

    #[test]
    fn state_colors_dark_and_light_differ() {
        // Verify that dark and light palettes produce different idle colours
        let dark = StateColors {
            idle: (100, 150, 255, 200),
            recording: (255, 80, 80, 230),
            transcribing: (255, 210, 50, 220),
            loading: (180, 180, 180, 140),
        };
        let light = StateColors {
            idle: (30, 80, 200, 220),
            recording: (200, 30, 30, 230),
            transcribing: (180, 140, 0, 220),
            loading: (100, 100, 100, 160),
        };
        assert_ne!(dark.idle, light.idle);
        assert_ne!(dark.recording, light.recording);
    }

    #[test]
    fn embedded_icon_is_retina_resolution() {
        let (_, w, h) = tint_png_rgba(ICON_PNG, 0, 0, 0, 255).unwrap();
        // Menu bar icons need ≥ 36px for 2× Retina at 18pt display size
        assert!(w >= 36, "icon width {w} too small for Retina");
        assert!(h >= 36, "icon height {h} too small for Retina");
    }
}
