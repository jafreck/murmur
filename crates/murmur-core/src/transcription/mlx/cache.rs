use anyhow::Result;
use mlx_rs::ops;
use mlx_rs::Array;

pub(crate) struct LayerKvCache {
    key: Option<Array>,
    value: Option<Array>,
}

impl LayerKvCache {
    pub(crate) fn new() -> Self {
        Self {
            key: None,
            value: None,
        }
    }

    pub(crate) fn append(&mut self, k: Array, v: Array) -> Result<(Array, Array)> {
        let new_k = match self.key.take() {
            Some(prev) => ops::concatenate_axis(&[&prev, &k], 2)?,
            None => k,
        };
        let new_v = match self.value.take() {
            Some(prev) => ops::concatenate_axis(&[&prev, &v], 2)?,
            None => v,
        };
        self.key = Some(new_k.clone());
        self.value = Some(new_v.clone());
        Ok((new_k, new_v))
    }
}

pub(crate) struct KvCache {
    pub(crate) layers: Vec<LayerKvCache>,
}

impl KvCache {
    pub(crate) fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| LayerKvCache::new()).collect(),
        }
    }
}
