pub mod engine;
#[cfg(feature = "onnx")]
pub mod mel;
pub mod model;
#[cfg(feature = "onnx")]
pub mod parakeet_engine;
pub mod postprocess;
#[cfg(feature = "onnx")]
pub mod qwen_engine;
pub mod streaming;
pub mod subprocess;
pub mod transcriber;
pub mod vad;
pub mod whisper_engine;

pub use engine::{AsrEngine, StreamingState, TranscriptionResult};
pub use model::{
    download, download_for_backend, download_onnx_model, model_exists_for_backend,
    onnx_model_exists, parakeet_model_dir, qwen3_asr_model_dir,
};
#[cfg(feature = "onnx")]
pub use parakeet_engine::ParakeetEngine;
pub use postprocess::process;
#[cfg(feature = "onnx")]
pub use qwen_engine::QwenEngine;
pub use streaming::{start_native_streaming, start_streaming, StreamingEvent};
pub use subprocess::SubprocessTranscriber;
pub use transcriber::{
    build_initial_prompt, filter_novel_terms, find_model, model_exists, rank_vocabulary,
    read_wav_samples, RankedTerm, Transcriber, TranscriptionContext,
};
pub use whisper_engine::WhisperEngine;
