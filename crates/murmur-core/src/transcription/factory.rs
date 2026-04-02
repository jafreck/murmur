//! Engine factory for ASR engine construction.

use anyhow::Result;

use crate::config::{AsrBackend, AsrQuantization};

use super::engine::AsrEngine;

/// Factory that delegates to the concrete engine constructors
/// and model helpers already in murmur-core.
///
/// Constructed once at app startup and shared across threads.
pub struct DefaultEngineFactory;

impl DefaultEngineFactory {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DefaultEngineFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultEngineFactory {
    pub fn create_engine(
        &self,
        backend: AsrBackend,
        model: &str,
        #[allow(unused_variables)] language: &str,
        #[allow(unused_variables)] quantization: AsrQuantization,
    ) -> Result<Box<dyn AsrEngine + Send + Sync>> {
        match backend {
            #[cfg(feature = "whisper")]
            AsrBackend::Whisper => {
                let model_path = super::model_discovery::find_model(model)
                    .ok_or_else(|| anyhow::anyhow!("Model not found: {model}"))?;
                let engine = super::whisper_engine::WhisperEngine::new(&model_path, language)?;
                Ok(Box::new(engine))
            }
            #[cfg(not(feature = "whisper"))]
            AsrBackend::Whisper => {
                anyhow::bail!(
                    "Whisper backend requires the 'whisper' feature. \
                     Rebuild with: cargo build --features whisper"
                );
            }
            #[cfg(feature = "onnx")]
            AsrBackend::Qwen3Asr => {
                let model_dir = super::model::qwen3_asr_model_dir(model);
                let engine = super::qwen_engine::QwenEngine::new(&model_dir, quantization)?;
                Ok(Box::new(engine))
            }
            #[cfg(feature = "onnx")]
            AsrBackend::Parakeet => {
                let model_dir = super::model::parakeet_model_dir(model);
                let engine = super::parakeet_engine::ParakeetEngine::new(&model_dir, quantization)?;
                Ok(Box::new(engine))
            }
            #[cfg(not(feature = "onnx"))]
            AsrBackend::Qwen3Asr | AsrBackend::Parakeet => {
                anyhow::bail!(
                    "ONNX backends require the 'onnx' feature. \
                     Rebuild with: cargo build --features onnx"
                );
            }
            #[cfg(feature = "mlx")]
            AsrBackend::Mlx => {
                let model_dir = super::model::mlx_model_dir(model);
                let engine = super::mlx::MlxEngine::new(&model_dir)?;
                Ok(Box::new(engine))
            }
            #[cfg(not(feature = "mlx"))]
            AsrBackend::Mlx => {
                anyhow::bail!(
                    "MLX backend requires the 'mlx' feature. \
                     Rebuild with: cargo build --features mlx"
                );
            }
        }
    }

    pub fn model_exists(
        &self,
        backend: AsrBackend,
        model: &str,
        quantization: AsrQuantization,
    ) -> bool {
        super::model::model_exists_for_backend(backend, model, quantization)
    }

    pub fn download_model(
        &self,
        backend: AsrBackend,
        model: &str,
        quantization: AsrQuantization,
        progress: Box<dyn Fn(f64) + Send>,
    ) -> Result<()> {
        super::model::download_for_backend(backend, model, quantization, progress)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_factory_reports_missing_whisper_model() {
        let factory = DefaultEngineFactory::new();
        // A nonsensical model name should not exist on disk.
        assert!(!factory.model_exists(
            AsrBackend::Whisper,
            "nonexistent-model-xyz",
            AsrQuantization::default()
        ));
    }

    #[cfg(feature = "whisper")]
    #[test]
    fn default_factory_create_engine_fails_for_missing_model() {
        let factory = DefaultEngineFactory::new();
        let result = factory.create_engine(
            AsrBackend::Whisper,
            "nonexistent-model-xyz",
            "en",
            AsrQuantization::default(),
        );
        assert!(result.is_err());
    }
}
