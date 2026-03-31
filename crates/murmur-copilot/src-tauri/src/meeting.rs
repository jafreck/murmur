use std::sync::{mpsc, Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use log::{info, warn};
use murmur_core::{
    audio::{AudioRecorder, SystemAudioCapturer},
    config::Config,
    transcription::{
        start_streaming, streaming::StreamingHandle, AsrEngine, StreamingEvent, WhisperEngine,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    Idle,
    Recording,
    Stopped,
}

/// Identifies who is speaking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Speaker {
    User,
    Remote,
}

/// A single labelled transcript fragment.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TranscriptEntry {
    pub speaker: Speaker,
    pub text: String,
    pub timestamp_ms: u64,
}

/// Manages a single meeting's audio capture and streaming transcription.
///
/// Supports an optional second audio stream (system audio) so both the
/// user's microphone and the remote participants' audio are transcribed.
pub struct MeetingSession {
    state: SessionState,
    recorder: AudioRecorder,
    system_capturer: Option<SystemAudioCapturer>,
    transcriber: Arc<dyn AsrEngine + Send + Sync>,
    model_path: std::path::PathBuf,
    language: String,
    mic_streaming: Option<StreamingHandle>,
    sys_streaming: Option<StreamingHandle>,
    transcript: Arc<Mutex<Vec<TranscriptEntry>>>,
    config: Config,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

impl MeetingSession {
    /// Create a new session. Downloads / locates the Whisper model on first use.
    pub fn new() -> Result<Self> {
        let config = Config::load();

        let model_path =
            murmur_core::transcription::find_model(&config.model_size).unwrap_or_else(|| {
                info!(
                    "model not found locally — downloading {}",
                    config.model_size
                );
                murmur_core::transcription::download(&config.model_size, |p| {
                    info!("download progress: {:.0}%", p * 100.0);
                })
                .expect("failed to download whisper model")
            });

        let transcriber: Arc<dyn AsrEngine + Send + Sync> =
            Arc::new(WhisperEngine::new(&model_path, &config.language)?);
        let mut recorder = AudioRecorder::with_noise_suppression(config.noise_suppression);
        recorder.warm()?;

        // Set up system audio capturer if a device is configured.
        let system_capturer = config.system_audio_device.as_deref().and_then(|name| {
            match SystemAudioCapturer::new(name) {
                Ok(c) => Some(c),
                Err(e) => {
                    warn!("could not open system audio device '{name}': {e}");
                    None
                }
            }
        });

        Ok(Self {
            state: SessionState::Idle,
            recorder,
            system_capturer,
            transcriber,
            model_path: model_path.clone(),
            language: config.language.clone(),
            mic_streaming: None,
            sys_streaming: None,
            transcript: Arc::new(Mutex::new(Vec::new())),
            config,
        })
    }

    pub fn state(&self) -> SessionState {
        self.state
    }

    /// Returns the current transcript as a single formatted string.
    pub fn transcript_text(&self) -> String {
        let entries = self.transcript.lock().unwrap();
        entries
            .iter()
            .map(|e| {
                let label = match e.speaker {
                    Speaker::User => "You",
                    Speaker::Remote => "Remote",
                };
                format!("{label}: {}", e.text)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Configure (or clear) the system audio device at runtime.
    pub fn set_system_audio_device(&mut self, device_name: Option<&str>) {
        // Stop any existing system capture.
        if let Some(ref mut cap) = self.system_capturer {
            cap.stop();
        }

        self.system_capturer = device_name.and_then(|name| match SystemAudioCapturer::new(name) {
            Ok(c) => {
                info!("system audio device set to '{name}'");
                Some(c)
            }
            Err(e) => {
                warn!("could not open system audio device '{name}': {e}");
                None
            }
        });
    }

    /// Start microphone (and optionally system audio) capture with streaming
    /// transcription. Returns a receiver that yields transcript updates.
    pub fn start(&mut self) -> Result<mpsc::Receiver<TranscriptUpdate>> {
        if self.state == SessionState::Recording {
            anyhow::bail!("meeting session already recording");
        }

        // ── Microphone stream ────────────────────────────────────────────
        self.recorder.start_in_memory()?;
        let mic_samples = self.recorder.sample_buffer();

        let (mic_tx, mic_rx) = mpsc::channel::<StreamingEvent>();
        let mic_worker = murmur_core::transcription::SubprocessTranscriber::new(
            &self.model_path,
            &self.language,
        )?;
        let mic_handle = start_streaming(
            mic_samples,
            Arc::clone(&self.transcriber),
            self.config.translate_to_english,
            self.config.filler_word_removal,
            mic_tx,
            mic_worker,
        );
        self.mic_streaming = Some(mic_handle);

        // ── System audio stream (optional) ───────────────────────────────
        let sys_rx_opt = if let Some(ref mut capturer) = self.system_capturer {
            match capturer.start() {
                Ok(sys_samples) => {
                    match murmur_core::transcription::SubprocessTranscriber::new(
                        &self.model_path,
                        &self.language,
                    ) {
                        Ok(sys_worker) => {
                            let (sys_tx, sys_rx) = mpsc::channel::<StreamingEvent>();
                            let sys_handle = start_streaming(
                                sys_samples,
                                Arc::clone(&self.transcriber),
                                self.config.translate_to_english,
                                self.config.filler_word_removal,
                                sys_tx,
                                sys_worker,
                            );
                            self.sys_streaming = Some(sys_handle);
                            Some(sys_rx)
                        }
                        Err(e) => {
                            warn!("failed to spawn system audio worker: {e}");
                            None
                        }
                    }
                }
                Err(e) => {
                    warn!("failed to start system audio capture: {e}");
                    None
                }
            }
        } else {
            None
        };

        // ── Merge streams into labelled transcript updates ───────────────
        let (update_tx, update_rx) = mpsc::channel::<TranscriptUpdate>();
        let transcript = Arc::clone(&self.transcript);

        // Mic forwarder
        let update_tx_mic = update_tx.clone();
        let transcript_mic = Arc::clone(&transcript);
        std::thread::spawn(move || {
            for event in mic_rx {
                if let StreamingEvent::PartialText {
                    text,
                    replace_chars,
                } = event
                {
                    {
                        let mut entries = transcript_mic.lock().unwrap();
                        entries.push(TranscriptEntry {
                            speaker: Speaker::User,
                            text: text.clone(),
                            timestamp_ms: now_ms(),
                        });
                    }
                    let _ = update_tx_mic.send(TranscriptUpdate {
                        speaker: Speaker::User,
                        text,
                        replace_chars,
                    });
                }
            }
        });

        // System audio forwarder (if active)
        if let Some(sys_rx) = sys_rx_opt {
            let update_tx_sys = update_tx;
            let transcript_sys = Arc::clone(&transcript);
            std::thread::spawn(move || {
                for event in sys_rx {
                    if let StreamingEvent::PartialText {
                        text,
                        replace_chars,
                    } = event
                    {
                        {
                            let mut entries = transcript_sys.lock().unwrap();
                            entries.push(TranscriptEntry {
                                speaker: Speaker::Remote,
                                text: text.clone(),
                                timestamp_ms: now_ms(),
                            });
                        }
                        let _ = update_tx_sys.send(TranscriptUpdate {
                            speaker: Speaker::Remote,
                            text,
                            replace_chars,
                        });
                    }
                }
            });
        }

        self.state = SessionState::Recording;
        info!("meeting started — streaming transcription active");
        Ok(update_rx)
    }

    /// Stop recording and streaming. Returns the accumulated transcript.
    pub fn stop(&mut self) -> Vec<TranscriptEntry> {
        if let Some(handle) = self.mic_streaming.take() {
            handle.stop_and_join();
        }
        if let Some(handle) = self.sys_streaming.take() {
            handle.stop_and_join();
        }
        self.recorder.stop_samples();
        if let Some(ref mut capturer) = self.system_capturer {
            capturer.stop();
        }
        self.state = SessionState::Stopped;
        info!("meeting stopped");

        let entries = self.transcript.lock().unwrap();
        entries.clone()
    }
}

/// A simplified transcript update sent to the frontend.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TranscriptUpdate {
    pub speaker: Speaker,
    pub text: String,
    pub replace_chars: usize,
}
