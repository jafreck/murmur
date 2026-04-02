pub mod activate;
pub mod capture;
mod capture_state;
mod denoise;
pub mod recordings;
pub(crate) mod resample;
pub mod speaker;
pub mod system_capture;

pub use activate::prepare_default_input;
pub use capture::AudioRecorder;
pub use recordings::RecordingStore;
pub use resample::{f32_to_i16, mix_to_mono, TARGET_RATE, WHISPER_WAV_SPEC};
pub use speaker::{ActiveSpeaker, SpeakerTracker};
pub use system_capture::{list_audio_devices, AudioDevice, SystemAudioCapturer};
