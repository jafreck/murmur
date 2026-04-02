//! Backward-compatible re-exports — canonical implementations live in
//! [`crate::models`].

pub use crate::models::{
    download, download_for_backend, download_onnx_model, is_valid_ggml_file, mlx_model_dir,
    model_exists_for_backend, model_filename, model_url, onnx_model_exists, parakeet_model_dir,
    qwen3_asr_model_dir,
};
