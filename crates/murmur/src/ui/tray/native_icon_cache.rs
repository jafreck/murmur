//! macOS-native icon cache: pre-builds `NSImage` objects so that tray icon
//! updates are just a pointer swap (`button.setImage`) instead of the
//! encode-PNG → decode-PNG round-trip that `tray-icon`'s `set_icon()` performs
//! on every call.

use anyhow::Result;
use objc2::AnyThread;
use objc2::MainThreadMarker;
use objc2_app_kit::{NSCellImagePosition, NSImage, NSStatusItem};
use objc2_foundation::{NSData, NSSize};

use super::{tint_pixels, DecodedIcon, StateColors, TrayState, PULSE_FRAME_COUNT};

/// A single cached macOS-native icon ready for instant display.
struct CachedImage {
    nsimage: objc2::rc::Retained<NSImage>,
}

impl CachedImage {
    /// Build an `NSImage` from raw RGBA pixels.
    fn from_rgba(rgba: &[u8], width: u32, height: u32) -> Result<Self> {
        // Encode to PNG (done once at startup, not per-frame).
        let png_data = encode_png(rgba, width, height)?;
        let nsdata = NSData::from_vec(png_data);
        let nsimage = NSImage::initWithData(NSImage::alloc(), &nsdata)
            .ok_or_else(|| anyhow::anyhow!("Failed to create NSImage from PNG data"))?;
        let icon_height: f64 = 18.0;
        let icon_width: f64 = (width as f64) / (height as f64 / icon_height);
        nsimage.setSize(NSSize::new(icon_width, icon_height));
        Ok(Self { nsimage })
    }

    /// Build from decoded icon data with a tint colour.
    fn from_decoded(decoded: &DecodedIcon, r: u8, g: u8, b: u8, a: u8) -> Result<Self> {
        let rgba = tint_pixels(&decoded.pixels, decoded.stride, r, g, b, a);
        Self::from_rgba(&rgba, decoded.width, decoded.height)
    }
}

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

/// Pre-built native icon cache for all tray states and animation frames.
pub struct NativeIconCache {
    mtm: MainThreadMarker,
    ns_status_item: objc2::rc::Retained<NSStatusItem>,
    idle: CachedImage,
    recording: CachedImage,
    transcribing: CachedImage,
    loading: CachedImage,
    loading_pulse: Vec<CachedImage>,
    downloading_pulse: Vec<CachedImage>,
}

impl NativeIconCache {
    /// Build the cache from the decoded icon and current appearance colours.
    pub fn new(
        tray: &tray_icon::TrayIcon,
        decoded: &DecodedIcon,
        colors: &StateColors,
    ) -> Result<Self> {
        let mtm = MainThreadMarker::new()
            .ok_or_else(|| anyhow::anyhow!("NativeIconCache must be created on the main thread"))?;
        let ns_status_item = tray
            .ns_status_item()
            .ok_or_else(|| anyhow::anyhow!("NSStatusItem not available"))?;

        let idle = CachedImage::from_decoded(
            decoded,
            colors.idle.0,
            colors.idle.1,
            colors.idle.2,
            colors.idle.3,
        )?;
        let recording = CachedImage::from_decoded(
            decoded,
            colors.recording.0,
            colors.recording.1,
            colors.recording.2,
            colors.recording.3,
        )?;
        let transcribing = CachedImage::from_decoded(
            decoded,
            colors.transcribing.0,
            colors.transcribing.1,
            colors.transcribing.2,
            colors.transcribing.3,
        )?;
        let loading = CachedImage::from_decoded(
            decoded,
            colors.loading.0,
            colors.loading.1,
            colors.loading.2,
            colors.loading.3,
        )?;

        let loading_pulse = Self::build_pulse_cache(decoded, colors.loading)?;
        let downloading_pulse = Self::build_pulse_cache(decoded, colors.transcribing)?;

        Ok(Self {
            mtm,
            ns_status_item,
            idle,
            recording,
            transcribing,
            loading,
            loading_pulse,
            downloading_pulse,
        })
    }

    fn build_pulse_cache(
        decoded: &DecodedIcon,
        base: (u8, u8, u8, u8),
    ) -> Result<Vec<CachedImage>> {
        use super::PULSE_ALPHA_MIN;
        (0..PULSE_FRAME_COUNT)
            .map(|i| {
                let phase_angle = (i as f64 / PULSE_FRAME_COUNT as f64) * std::f64::consts::TAU;
                let phase = phase_angle.sin();
                let scale = PULSE_ALPHA_MIN + (1.0 - PULSE_ALPHA_MIN) * (phase + 1.0) / 2.0;
                let a = (base.3 as f64 * scale).round().min(255.0) as u8;
                CachedImage::from_decoded(decoded, base.0, base.1, base.2, a)
            })
            .collect()
    }

    /// Set the tray icon to the cached image for the given state.
    /// This bypasses `tray-icon`'s `set_icon()` entirely — just a pointer swap.
    pub fn set_icon_for_state(&self, state: &TrayState) {
        let image = match state {
            TrayState::Idle => &self.idle,
            TrayState::Recording | TrayState::Error => &self.recording,
            TrayState::Transcribing | TrayState::Downloading => &self.transcribing,
            TrayState::Loading => &self.loading,
        };
        self.apply_nsimage(&image.nsimage);
    }

    /// Set a pulse animation frame. Returns `false` if the frame index is out
    /// of range or the state is not a pulsing state.
    pub fn set_pulse_frame(&self, state: &TrayState, frame_idx: usize) -> bool {
        let frames = match state {
            TrayState::Loading => &self.loading_pulse,
            TrayState::Downloading => &self.downloading_pulse,
            _ => return false,
        };
        if let Some(cached) = frames.get(frame_idx) {
            self.apply_nsimage(&cached.nsimage);
            true
        } else {
            false
        }
    }

    fn apply_nsimage(&self, nsimage: &NSImage) {
        if let Some(button) = self.ns_status_item.button(self.mtm) {
            button.setImage(Some(nsimage));
            button.setImagePosition(NSCellImagePosition::ImageLeft);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- encode_png --

    #[test]
    fn encode_png_produces_valid_png() {
        // 2×2 red square
        let rgba = [255, 0, 0, 255].repeat(4);
        let png_data = encode_png(&rgba, 2, 2).unwrap();
        // PNG magic bytes
        assert_eq!(&png_data[..4], &[137, 80, 78, 71]);
    }

    #[test]
    fn encode_png_single_pixel() {
        let rgba = [0, 255, 0, 128];
        let png_data = encode_png(&rgba, 1, 1).unwrap();
        assert!(!png_data.is_empty());
        assert_eq!(&png_data[..4], &[137, 80, 78, 71]);
    }

    #[test]
    fn encode_png_roundtrip() {
        let rgba: Vec<u8> = [100, 150, 200, 255].repeat(9);
        let png_data = encode_png(&rgba, 3, 3).unwrap();
        // Decode back and verify dimensions
        let decoder = png::Decoder::new(std::io::Cursor::new(&png_data));
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!(info.width, 3);
        assert_eq!(info.height, 3);
        assert_eq!(&buf[..info.buffer_size()], &rgba[..]);
    }

    #[test]
    fn encode_png_transparent_pixels() {
        let rgba = [0, 0, 0, 0].repeat(4);
        let png_data = encode_png(&rgba, 2, 2).unwrap();
        assert!(!png_data.is_empty());
    }

    // -- CachedImage --

    #[test]
    fn cached_image_from_rgba_small() {
        let rgba = [255, 255, 255, 255].repeat(4);
        let img = CachedImage::from_rgba(&rgba, 2, 2).unwrap();
        let size = img.nsimage.size();
        // Width is scaled proportionally to 18pt height
        assert!((size.height - 18.0).abs() < 0.01);
        assert!((size.width - 18.0).abs() < 0.01); // square → equal
    }

    #[test]
    fn cached_image_from_rgba_rectangular() {
        // 4 wide × 2 tall
        let rgba = [128, 64, 32, 200].repeat(8);
        let img = CachedImage::from_rgba(&rgba, 4, 2).unwrap();
        let size = img.nsimage.size();
        assert!((size.height - 18.0).abs() < 0.01);
        // 4:2 aspect → width should be 36
        assert!((size.width - 36.0).abs() < 0.01);
    }

    #[test]
    fn cached_image_from_decoded_embedded_icon() {
        let decoded = super::super::decode_icon_png().unwrap();
        let img = CachedImage::from_decoded(&decoded, 255, 80, 80, 230).unwrap();
        let size = img.nsimage.size();
        assert!((size.height - 18.0).abs() < 0.01);
        assert!(size.width > 0.0);
    }

    #[test]
    fn cached_image_different_tints_differ() {
        let decoded = super::super::decode_icon_png().unwrap();
        let red = CachedImage::from_decoded(&decoded, 255, 0, 0, 255).unwrap();
        let blue = CachedImage::from_decoded(&decoded, 0, 0, 255, 255).unwrap();
        // Both should succeed — we can't compare NSImage contents easily,
        // but we can verify they're valid distinct objects.
        assert!(!std::ptr::eq(&*red.nsimage, &*blue.nsimage));
    }
}
