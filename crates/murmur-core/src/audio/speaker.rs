/// Simple speaker activity detection based on audio energy levels.
///
/// Not true diarization — uses the fact that the microphone captures the
/// local user while the system audio stream captures remote participants.
/// Compares smoothed energy levels to decide who is currently talking.
pub struct SpeakerTracker {
    /// Exponentially-smoothed energy level for the mic stream.
    mic_energy: f32,
    /// Exponentially-smoothed energy level for the system stream.
    sys_energy: f32,
    /// Smoothing factor (0..1). Higher = more responsive, noisier.
    alpha: f32,
    /// Energy threshold below which a stream is considered silent.
    silence_threshold: f32,
}

/// Which speaker (if any) is currently active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveSpeaker {
    /// Only the local user is speaking (mic energy dominant).
    User,
    /// Only remote participant(s) are speaking (system audio dominant).
    Remote,
    /// Both streams have significant energy.
    Both,
    /// Neither stream has significant energy.
    Silence,
}

impl SpeakerTracker {
    /// Create a tracker with default sensitivity settings.
    pub fn new() -> Self {
        Self {
            mic_energy: 0.0,
            sys_energy: 0.0,
            alpha: 0.3,
            silence_threshold: 0.005,
        }
    }

    /// Create a tracker with custom smoothing and silence threshold.
    pub fn with_settings(alpha: f32, silence_threshold: f32) -> Self {
        Self {
            mic_energy: 0.0,
            sys_energy: 0.0,
            alpha: alpha.clamp(0.01, 1.0),
            silence_threshold: silence_threshold.max(0.0),
        }
    }

    /// Update with new audio samples from each stream.
    ///
    /// Returns which speaker (if any) is currently active based on
    /// comparing the smoothed RMS energy of each stream against the
    /// silence threshold.
    pub fn update(&mut self, mic_samples: &[f32], sys_samples: &[f32]) -> ActiveSpeaker {
        let mic_rms = rms_energy(mic_samples);
        let sys_rms = rms_energy(sys_samples);

        self.mic_energy = self.mic_energy * (1.0 - self.alpha) + mic_rms * self.alpha;
        self.sys_energy = self.sys_energy * (1.0 - self.alpha) + sys_rms * self.alpha;

        let mic_active = self.mic_energy > self.silence_threshold;
        let sys_active = self.sys_energy > self.silence_threshold;

        match (mic_active, sys_active) {
            (true, true) => ActiveSpeaker::Both,
            (true, false) => ActiveSpeaker::User,
            (false, true) => ActiveSpeaker::Remote,
            (false, false) => ActiveSpeaker::Silence,
        }
    }

    /// Current smoothed mic energy level.
    pub fn mic_energy(&self) -> f32 {
        self.mic_energy
    }

    /// Current smoothed system audio energy level.
    pub fn sys_energy(&self) -> f32 {
        self.sys_energy
    }
}

impl Default for SpeakerTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the RMS (root mean square) energy of an audio buffer.
fn rms_energy(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples
        .iter()
        .filter(|s| s.is_finite())
        .map(|s| s * s)
        .sum();
    (sum_sq / samples.len() as f32).sqrt().min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_energy_silence() {
        assert_eq!(rms_energy(&[0.0; 100]), 0.0);
    }

    #[test]
    fn test_rms_energy_empty() {
        assert_eq!(rms_energy(&[]), 0.0);
    }

    #[test]
    fn test_rms_energy_signal() {
        let samples = vec![0.5; 100];
        let e = rms_energy(&samples);
        assert!((e - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_speaker_tracker_silence() {
        let mut tracker = SpeakerTracker::new();
        let silence = vec![0.0; 160];
        let result = tracker.update(&silence, &silence);
        assert_eq!(result, ActiveSpeaker::Silence);
    }

    #[test]
    fn test_speaker_tracker_user_speaking() {
        let mut tracker = SpeakerTracker::new();
        let loud = vec![0.5; 160];
        let silence = vec![0.0; 160];
        // Feed several frames to overcome smoothing.
        for _ in 0..10 {
            tracker.update(&loud, &silence);
        }
        let result = tracker.update(&loud, &silence);
        assert_eq!(result, ActiveSpeaker::User);
    }

    #[test]
    fn test_speaker_tracker_remote_speaking() {
        let mut tracker = SpeakerTracker::new();
        let loud = vec![0.5; 160];
        let silence = vec![0.0; 160];
        for _ in 0..10 {
            tracker.update(&silence, &loud);
        }
        let result = tracker.update(&silence, &loud);
        assert_eq!(result, ActiveSpeaker::Remote);
    }

    #[test]
    fn test_speaker_tracker_both() {
        let mut tracker = SpeakerTracker::new();
        let loud = vec![0.5; 160];
        for _ in 0..10 {
            tracker.update(&loud, &loud);
        }
        let result = tracker.update(&loud, &loud);
        assert_eq!(result, ActiveSpeaker::Both);
    }

    #[test]
    fn test_speaker_tracker_default() {
        let tracker = SpeakerTracker::default();
        assert_eq!(tracker.mic_energy(), 0.0);
        assert_eq!(tracker.sys_energy(), 0.0);
    }

    #[test]
    fn test_speaker_tracker_custom_settings() {
        let tracker = SpeakerTracker::with_settings(0.5, 0.01);
        assert_eq!(tracker.mic_energy(), 0.0);
    }
}
