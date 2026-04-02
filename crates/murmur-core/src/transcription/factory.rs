//! Engine factory trait and default implementation.
//!
//! [`EngineFactory`] decouples engine construction from the app crate so
//! consumers depend on the abstraction rather than concrete engine types.

use anyhow::Result;

use crate::config::{AsrBackend, AsrQuantization};

use super::engine::AsrEngine;

/// Factory for creating ASR engines and managing their models.
///
/// Constructed once at app startup and shared across threads.
pub trait EngineFactory: Send + Sync {
    /// Create an engine for the given backend/model/language combination.
    fn create_engine(
        &self,
        backend: AsrBackend,
        model: &str,
        language: &str,
        quantization: AsrQuantization,
    ) -> Result<Box<dyn AsrEngine + Send + Sync>>;

    /// Check if the model exists locally for the given backend.
    fn model_exists(&self, backend: AsrBackend, model: &str, quantization: AsrQuantization)
        -> bool;

    /// Download the model for the given backend.
    fn download_model(
        &self,
        backend: AsrBackend,
        model: &str,
        quantization: AsrQuantization,
        progress: Box<dyn Fn(f64) + Send>,
    ) -> Result<()>;
}

/// Default factory that delegates to the concrete engine constructors
/// and model helpers already in murmur-core.
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

impl EngineFactory for DefaultEngineFactory {
    fn create_engine(
        &self,
        backend: AsrBackend,
        model: &str,
        language: &str,
        #[allow(unused_variables)] quantization: AsrQuantization,
    ) -> Result<Box<dyn AsrEngine + Send + Sync>> {
        match backend {
            AsrBackend::Whisper => {
                let model_path = super::model_discovery::find_model(model)
                    .ok_or_else(|| anyhow::anyhow!("Model not found: {model}"))?;
                let engine = super::whisper_engine::WhisperEngine::new(&model_path, language)?;
                Ok(Box::new(engine))
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

    fn model_exists(
        &self,
        backend: AsrBackend,
        model: &str,
        quantization: AsrQuantization,
    ) -> bool {
        super::model::model_exists_for_backend(backend, model, quantization)
    }

    fn download_model(
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
