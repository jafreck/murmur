use nnnoiseless::DenoiseState;

use super::resample::{resample_into, TARGET_RATE};

/// nnnoiseless operates on 480-sample frames at 48 kHz.
const DENOISE_FRAME_SIZE: usize = DenoiseState::FRAME_SIZE;

/// nnnoiseless native sample rate.
const DENOISE_RATE: u32 = 48_000;

/// Holds the denoiser state and an accumulation buffer for incomplete frames.
/// Created once per recording session and shared with the audio callback via Arc<Mutex>.
pub(super) struct Denoiser {
    state: Box<DenoiseState<'static>>,
    /// Accumulates 16 kHz samples until we have enough to fill a 48 kHz frame.
    pending_16k: Vec<f32>,
    /// Collects denoised 16 kHz output samples ready for the consumer.
    output_16k: Vec<f32>,
    /// Whether to skip the very first output frame (fade-in artifact).
    first_frame: bool,
    // Pre-allocated scratch buffers to avoid per-callback heap allocations.
    chunk_buf: Vec<f32>,
    upsampled_buf: Vec<f32>,
    denoised_48k_buf: Vec<f32>,
    downsampled_buf: Vec<f32>,
}

impl Denoiser {
    pub(super) fn new() -> Self {
        let frame_16k = DENOISE_FRAME_SIZE / 3; // 160
        Self {
            state: DenoiseState::new(),
            pending_16k: Vec::with_capacity(frame_16k + 16),
            output_16k: Vec::new(),
            first_frame: true,
            chunk_buf: Vec::with_capacity(frame_16k),
            upsampled_buf: Vec::with_capacity(DENOISE_FRAME_SIZE),
            denoised_48k_buf: Vec::with_capacity(DENOISE_FRAME_SIZE),
            downsampled_buf: Vec::with_capacity(frame_16k),
        }
    }

    /// Reset all state so the denoiser is clean for a new recording session.
    pub(super) fn reset(&mut self) {
        self.pending_16k.clear();
        self.output_16k.clear();
        self.first_frame = true;
        self.state = DenoiseState::new();
        // Scratch buffers keep their allocations; just clear contents.
        self.chunk_buf.clear();
        self.upsampled_buf.clear();
        self.denoised_48k_buf.clear();
        self.downsampled_buf.clear();
    }

    /// Feed 16 kHz samples and return denoised 16 kHz samples.
    ///
    /// Internally upsamples to 48 kHz, runs nnnoiseless frame-by-frame,
    /// then downsamples back to 16 kHz.
    pub(super) fn process(&mut self, samples_16k: &[f32]) -> &[f32] {
        self.output_16k.clear();
        self.pending_16k.extend_from_slice(samples_16k);

        // Each 48 kHz frame of 480 samples corresponds to 160 samples at 16 kHz.
        let frame_16k = DENOISE_FRAME_SIZE / 3; // 160

        while self.pending_16k.len() >= frame_16k {
            self.chunk_buf.clear();
            self.chunk_buf.extend(self.pending_16k.drain(..frame_16k));

            resample_into(
                &self.chunk_buf,
                TARGET_RATE,
                DENOISE_RATE,
                &mut self.upsampled_buf,
            );

            // nnnoiseless expects f32 in i16 range
            let mut input_frame = [0.0f32; DENOISE_FRAME_SIZE];
            for (i, &s) in self
                .upsampled_buf
                .iter()
                .take(DENOISE_FRAME_SIZE)
                .enumerate()
            {
                input_frame[i] = s * 32767.0;
            }

            let mut output_frame = [0.0f32; DENOISE_FRAME_SIZE];
            self.state.process_frame(&mut output_frame, &input_frame);

            if self.first_frame {
                self.first_frame = false;
                continue;
            }

            // Convert back from i16 range to [-1, 1]
            self.denoised_48k_buf.clear();
            self.denoised_48k_buf.extend(
                output_frame
                    .iter()
                    .map(|&s| (s / 32767.0_f32).clamp(-1.0, 1.0)),
            );

            resample_into(
                &self.denoised_48k_buf,
                DENOISE_RATE,
                TARGET_RATE,
                &mut self.downsampled_buf,
            );
            self.output_16k.extend_from_slice(&self.downsampled_buf);
        }

        &self.output_16k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_denoiser_new() {
        let d = Denoiser::new();
        assert!(d.pending_16k.is_empty());
        assert!(d.output_16k.is_empty());
        assert!(d.first_frame);
    }

    #[test]
    fn test_denoiser_reset() {
        let mut d = Denoiser::new();
        d.pending_16k.push(1.0);
        d.output_16k.push(2.0);
        d.first_frame = false;
        d.reset();
        assert!(d.pending_16k.is_empty());
        assert!(d.output_16k.is_empty());
        assert!(d.first_frame);
    }

    #[test]
    fn test_denoiser_process_empty() {
        let mut d = Denoiser::new();
        let out = d.process(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_denoiser_process_short_accumulates() {
        let mut d = Denoiser::new();
        let out = d.process(&[0.0; 100]);
        assert!(out.is_empty());
        assert_eq!(d.pending_16k.len(), 100);
    }

    #[test]
    fn test_denoiser_process_one_frame_skipped() {
        let mut d = Denoiser::new();
        // One frame = 160 samples at 16 kHz, but the first frame is always skipped.
        let out = d.process(&[0.0; 160]);
        assert!(out.is_empty());
        assert!(!d.first_frame);
    }

    #[test]
    fn test_denoiser_process_two_frames_produces_output() {
        let mut d = Denoiser::new();
        // First frame skipped, second frame produces 160 samples.
        let out = d.process(&[0.0; 320]);
        assert_eq!(out.len(), 160);
    }

    #[test]
    fn test_denoiser_process_multiple_frames() {
        let mut d = Denoiser::new();
        // 3 frames: first skipped, remaining 2 produce 320 samples.
        let out = d.process(&[0.0; 480]);
        assert_eq!(out.len(), 320);
    }

    #[test]
    fn test_denoiser_continuity_across_calls() {
        let mut d = Denoiser::new();
        // 100 samples: too few for a frame
        let out1 = d.process(&[0.1; 100]);
        assert!(out1.is_empty());

        // 100 more → 200 total, one frame processed (160) but skipped, 40 leftover
        let out2 = d.process(&[0.1; 100]);
        assert!(out2.is_empty());
        assert_eq!(d.pending_16k.len(), 40);

        // 120 more → 40 + 120 = 160, second frame produces output
        let out3 = d.process(&[0.1; 120]).to_vec();
        assert_eq!(out3.len(), 160);
    }
}
