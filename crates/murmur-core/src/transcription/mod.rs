pub mod engine;
pub mod factory;
#[cfg(any(feature = "onnx", feature = "mlx"))]
pub mod mel;
#[cfg(feature = "mlx")]
mod mlx;
pub mod model;
pub mod model_discovery;
#[cfg(feature = "onnx")]
pub mod parakeet_engine;
pub mod postprocess;
pub mod prompt;
#[cfg(feature = "onnx")]
pub mod qwen_engine;
pub mod streaming;
pub mod subprocess;
pub mod transcriber;
pub mod vad;
pub mod whisper_engine;

pub use engine::{AsrEngine, StreamingState, TranscriptionResult};
pub use factory::DefaultEngineFactory;
#[cfg(feature = "mlx")]
pub use mlx::MlxEngine;
pub use model::{
    download, download_for_backend, download_onnx_model, mlx_model_dir, model_exists_for_backend,
    onnx_model_exists, parakeet_model_dir, qwen3_asr_model_dir,
};
pub use model_discovery::{find_model, model_exists, read_wav_samples};
#[cfg(feature = "onnx")]
pub use parakeet_engine::ParakeetEngine;
pub use postprocess::process;
pub use prompt::{
    build_initial_prompt, filter_novel_terms, rank_vocabulary, RankedTerm, TranscriptionContext,
};
#[cfg(feature = "onnx")]
pub use qwen_engine::QwenEngine;
pub use streaming::{start_native_streaming, start_streaming, StreamingEvent};
pub use subprocess::SubprocessTranscriber;
pub use transcriber::Transcriber;
pub use whisper_engine::WhisperEngine;
