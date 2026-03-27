use std::sync::{mpsc, Arc, Mutex};

use anyhow::Result;
use log::info;
use murmur_core::{
    audio::AudioRecorder,
    config::Config,
    transcription::{start_streaming, streaming::StreamingHandle, StreamingEvent, Transcriber},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    Idle,
    Recording,
    Stopped,
}

/// Manages a single meeting's audio capture and streaming transcription.
pub struct MeetingSession {
    state: SessionState,
    recorder: AudioRecorder,
    transcriber: Arc<Transcriber>,
    streaming_handle: Option<StreamingHandle>,
    transcript: Arc<Mutex<String>>,
    config: Config,
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

        let transcriber = Arc::new(Transcriber::new(&model_path, &config.language)?);
        let mut recorder = AudioRecorder::with_noise_suppression(config.noise_suppression);
        recorder.warm()?;

        Ok(Self {
            state: SessionState::Idle,
            recorder,
            transcriber,
            streaming_handle: None,
            transcript: Arc::new(Mutex::new(String::new())),
            config,
        })
    }

    pub fn state(&self) -> SessionState {
        self.state
    }

    /// Start microphone capture and streaming transcription.
    /// Returns a receiver that yields transcript text updates.
    pub fn start(&mut self) -> Result<mpsc::Receiver<TranscriptUpdate>> {
        self.recorder.start_in_memory()?;
        let sample_buffer = self.recorder.sample_buffer();

        let (streaming_tx, streaming_rx) = mpsc::channel::<StreamingEvent>();
        let handle = start_streaming(
            sample_buffer,
            Arc::clone(&self.transcriber),
            self.config.translate_to_english,
            self.config.filler_word_removal,
            streaming_tx,
        );
        self.streaming_handle = Some(handle);

        // Forward streaming events into simpler transcript updates while
        // also accumulating the full transcript text.
        let (update_tx, update_rx) = mpsc::channel::<TranscriptUpdate>();
        let transcript = Arc::clone(&self.transcript);
        std::thread::spawn(move || {
            for event in streaming_rx {
                match event {
                    StreamingEvent::PartialText {
                        text,
                        replace_chars,
                    } => {
                        {
                            let mut buf = transcript.lock().unwrap();
                            if replace_chars > 0 {
                                let new_len = buf.len().saturating_sub(replace_chars);
                                buf.truncate(new_len);
                            }
                            buf.push_str(&text);
                        }
                        let _ = update_tx.send(TranscriptUpdate {
                            text: text.clone(),
                            replace_chars,
                        });
                    }
                    StreamingEvent::SpeechDetected => {
                        // VAD heartbeat — no action needed for transcript display
                    }
                }
            }
        });

        self.state = SessionState::Recording;
        info!("meeting started — streaming transcription active");
        Ok(update_rx)
    }

    /// Stop recording and streaming. Returns the accumulated transcript.
    pub fn stop(&mut self) -> String {
        if let Some(handle) = self.streaming_handle.take() {
            handle.stop_and_join();
        }
        self.recorder.stop_samples();
        self.state = SessionState::Stopped;
        info!("meeting stopped");

        let transcript = self.transcript.lock().unwrap();
        transcript.clone()
    }
}

/// A simplified transcript update sent to the frontend.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TranscriptUpdate {
    pub text: String,
    pub replace_chars: usize,
}
