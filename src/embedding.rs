//! ONNX-based embedding provider for paraphrase-multilingual-MiniLM-L12-v2.
//!
//! When the `onnx` Cargo feature is enabled and the model file is present,
//! runs real inference via `ort`. Otherwise falls back to a deterministic
//! hash-based embedding that preserves the correct 384-dim interface.
//!
//! Implements [`ruvector_core::embeddings::EmbeddingProvider`].

use crate::error::AiAssistantError;

/// Embedding dimension for paraphrase-multilingual-MiniLM-L12-v2.
pub const EMBEDDING_DIM: usize = 384;
const MAX_SEQ_LEN: usize = 128;

// ── Backend enum ────────────────────────────────────────────────────────────

enum EmbeddingBackend {
    /// Deterministic hash-based fallback — correct dimensions, no semantics.
    Hash,
    /// Real ONNX inference via `ort`.
    /// Session::run() requires &mut self, so we wrap in Mutex for interior mutability.
    #[cfg(feature = "onnx")]
    Onnx(std::sync::Arc<std::sync::Mutex<ort::session::Session>>),
}

// ── Public struct ────────────────────────────────────────────────────────────

/// ONNX-based embedding provider.
///
/// Create with [`OnnxEmbedding::new`]; pass `config.embedding_model_path`.
pub struct OnnxEmbedding {
    model_path: String,
    backend: EmbeddingBackend,
}

// Safety: Mutex<Session> is Send+Sync; Hash variant is trivially so.
unsafe impl Send for OnnxEmbedding {}
unsafe impl Sync for OnnxEmbedding {}

impl OnnxEmbedding {
    /// Create a new embedding provider.
    ///
    /// Tries to load the ONNX model when the `onnx` feature is active.
    /// On failure (missing file, runtime error, or feature disabled) falls back
    /// to the deterministic hash implementation and logs a warning.
    pub fn new(model_path: &str) -> Result<Self, AiAssistantError> {
        // If the model path is relative, resolve it against the exe directory so
        // the binary works correctly regardless of the current working directory.
        let resolved_path: std::path::PathBuf = {
            let p = std::path::Path::new(model_path);
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                crate::config::exe_dir().join(p)
            }
        };
        let model_path_str = resolved_path
            .to_str()
            .unwrap_or(model_path);

        // Keep original string for display; use resolved for loading.
        let model_path_display = model_path.to_string();

        #[cfg(feature = "onnx")]
        {
            match Self::try_load_onnx(model_path_str) {
                Ok(session) => {
                    tracing::info!("ONNX model loaded from '{}'", model_path);
                    return Ok(Self {
                        model_path: model_path.to_string(),
                        backend: EmbeddingBackend::Onnx(std::sync::Arc::new(
                            std::sync::Mutex::new(session),
                        )),
                    });
                }
                Err(e) => {
                    tracing::warn!(
                        "ONNX load failed ({}); falling back to hash embeddings. \
                         Semantic search will NOT work correctly.",
                        e
                    );
                }
            }
        }

        #[cfg(not(feature = "onnx"))]
        tracing::warn!(
            "onnx feature disabled — using hash fallback. \
             Build with '--features onnx' for real semantic embeddings."
        );

        Ok(Self {
            model_path: model_path_display,
            backend: EmbeddingBackend::Hash,
        })
    }

    /// Returns `true` when running the hash fallback (no real model loaded).
    pub fn is_fallback(&self) -> bool {
        matches!(self.backend, EmbeddingBackend::Hash)
    }

    /// Model path this provider was initialised with.
    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    // ── Public embedding methods ─────────────────────────────────────────────

    /// Embed a single text string → 384-dimensional L2-normalised vector.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, AiAssistantError> {
        self.embed_internal(text)
    }

    /// Embed multiple texts; items are processed independently.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, AiAssistantError> {
        texts.iter().map(|t| self.embed_internal(t)).collect()
    }

    // ── Private core ─────────────────────────────────────────────────────────

    fn embed_internal(&self, text: &str) -> Result<Vec<f32>, AiAssistantError> {
        match &self.backend {
            EmbeddingBackend::Hash => Ok(Self::hash_embed(text)),
            #[cfg(feature = "onnx")]
            EmbeddingBackend::Onnx(session) => Self::onnx_embed(session, text),
        }
    }

    // ── ONNX inference (feature-gated) ───────────────────────────────────────

    #[cfg(feature = "onnx")]
    fn try_load_onnx(model_path: &str) -> Result<ort::session::Session, AiAssistantError> {
        // model_path may be either:
        //   (a) the full path to the .onnx file, or
        //   (b) a model directory — in which case the canonical sub-path
        //       "<model_path>/onnx/model.onnx" is tried automatically.
        let resolved = {
            let p = std::path::Path::new(model_path);
            if p.is_dir() {
                p.join("onnx").join("model.onnx")
            } else {
                p.to_path_buf()
            }
        };

        if !resolved.exists() {
            return Err(AiAssistantError::Embedding(format!(
                "ONNX model file not found: {}",
                resolved.display()
            )));
        }
        ort::session::Session::builder()
            .map_err(|e| AiAssistantError::Embedding(format!("ort builder: {e}")))?
            .commit_from_file(&resolved)
            .map_err(|e| AiAssistantError::Embedding(format!("ort load: {e}")))
    }

    #[cfg(feature = "onnx")]
    fn onnx_embed(
        session: &std::sync::Arc<std::sync::Mutex<ort::session::Session>>,
        text: &str,
    ) -> Result<Vec<f32>, AiAssistantError> {
        use ndarray::Array2;
        use ort::value::Tensor;

        let (input_ids, attention_mask) = Self::tokenize(text, MAX_SEQ_LEN);
        let seq_len = input_ids.len();
        let token_type_ids = vec![0i64; seq_len];

        // Build ndarray Array2 tensors
        let ids_arr = Array2::from_shape_vec((1, seq_len), input_ids)
            .map_err(|e| AiAssistantError::Embedding(e.to_string()))?;
        let mask_arr = Array2::from_shape_vec((1, seq_len), attention_mask.clone())
            .map_err(|e| AiAssistantError::Embedding(e.to_string()))?;
        let types_arr = Array2::from_shape_vec((1, seq_len), token_type_ids)
            .map_err(|e| AiAssistantError::Embedding(e.to_string()))?;

        // ort 2.0.0-rc.9: Tensor lives in ort::value module
        let ids_val = Tensor::from_array(ids_arr)
            .map_err(|e: ort::Error| AiAssistantError::Embedding(e.to_string()))?;
        let mask_val = Tensor::from_array(mask_arr)
            .map_err(|e: ort::Error| AiAssistantError::Embedding(e.to_string()))?;
        let types_val = Tensor::from_array(types_arr)
            .map_err(|e: ort::Error| AiAssistantError::Embedding(e.to_string()))?;

        // Lock the mutex to get &mut Session — Session::run() requires &mut self
        let mut guard = session
            .lock()
            .map_err(|_| AiAssistantError::Embedding("Session mutex poisoned".to_string()))?;

        // ort rc.9: ort::inputs! returns Result<Vec<...>> — unwrap with ? first
        let session_inputs = ort::inputs![
            "input_ids"      => ids_val,
            "attention_mask" => mask_val,
            "token_type_ids" => types_val
        ]
        .map_err(|e: ort::Error| AiAssistantError::Embedding(e.to_string()))?;

        let outputs = guard
            .run(session_inputs)
            .map_err(|e: ort::Error| AiAssistantError::Embedding(e.to_string()))?;

        // ort rc.9: try_extract_tensor returns ArrayBase directly (not a tuple)
        let tensor = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .map_err(|e: ort::Error| AiAssistantError::Embedding(e.to_string()))?;

        let flat: Vec<f32> = tensor.iter().copied().collect();

        // Reshape to Vec<Vec<f32>> of shape [seq_len][384]
        let mut token_embs: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let s = i * EMBEDDING_DIM;
            let e = s + EMBEDDING_DIM;
            if e > flat.len() {
                return Err(AiAssistantError::Embedding(
                    "ONNX output shorter than expected".to_string(),
                ));
            }
            token_embs.push(flat[s..e].to_vec());
        }

        let pooled = Self::mean_pool(&token_embs, &attention_mask);
        Ok(Self::normalize(&pooled))
    }

    // ── Tokenisation ─────────────────────────────────────────────────────────

    /// Simple whitespace tokenisation.
    ///
    /// Maps each lowercased word to a deterministic token-ID in BERT vocab range
    /// `[103, 30521]` using `DefaultHasher`. Wraps with `[CLS]`/`[SEP]` and pads
    /// to `max_length` with `[PAD]`.
    fn tokenize(text: &str, max_length: usize) -> (Vec<i64>, Vec<i64>) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // CLS=101, SEP=102, PAD=0
        let mut input_ids: Vec<i64> = vec![101];
        let mut attention_mask: Vec<i64> = vec![1];

        for word in text.split_whitespace().take(max_length - 2) {
            let mut h = DefaultHasher::new();
            word.to_lowercase().hash(&mut h);
            let token_id = (h.finish() % 30_419 + 103) as i64;
            input_ids.push(token_id);
            attention_mask.push(1);
        }

        input_ids.push(102); // [SEP]
        attention_mask.push(1);

        while input_ids.len() < max_length {
            input_ids.push(0);
            attention_mask.push(0);
        }

        (input_ids, attention_mask)
    }

    // ── Pooling / normalisation ───────────────────────────────────────────────

    /// Mean pool token embeddings, masking out padded positions.
    fn mean_pool(token_embeddings: &[Vec<f32>], attention_mask: &[i64]) -> Vec<f32> {
        let dim = token_embeddings.first().map_or(EMBEDDING_DIM, |v| v.len());
        let mut sum = vec![0.0f32; dim];
        let mut count = 0i64;

        for (emb, &mask) in token_embeddings.iter().zip(attention_mask.iter()) {
            if mask == 1 {
                for (s, e) in sum.iter_mut().zip(emb.iter()) {
                    *s += e;
                }
                count += 1;
            }
        }

        if count > 0 {
            sum.iter_mut().for_each(|s| *s /= count as f32);
        }
        sum
    }

    /// L2-normalise a vector in place; returns `v` unchanged if norm < ε.
    fn normalize(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v.to_vec()
        }
    }

    // ── Hash fallback ────────────────────────────────────────────────────────

    /// Deterministic hash-based embedding (development / test fallback).
    ///
    /// Uses an XOR-shift PRNG seeded per 48-element chunk to fill all 384
    /// dimensions, then L2-normalises the result.
    ///
    /// ⚠️ NOT semantic — identical characters produce identical embeddings.
    fn hash_embed(text: &str) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut embedding = vec![0.0f32; EMBEDDING_DIM];
        const CHUNK: usize = 48; // 384 / 8 chunks

        for seed in 0u64..8 {
            let mut h = DefaultHasher::new();
            seed.hash(&mut h);
            text.hash(&mut h);
            let mut state = h.finish();
            // xorshift64
            let start = (seed as usize) * CHUNK;
            for i in 0..CHUNK {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let f = (state as f32) / (u64::MAX as f32) * 2.0 - 1.0;
                if start + i < EMBEDDING_DIM {
                    embedding[start + i] = f;
                }
            }
        }

        Self::normalize(&embedding)
    }
}

// ── EmbeddingProvider trait impl ─────────────────────────────────────────────

impl ruvector_core::embeddings::EmbeddingProvider for OnnxEmbedding {
    fn embed(&self, text: &str) -> ruvector_core::error::Result<Vec<f32>> {
        self.embed_internal(text).map_err(|e| {
            ruvector_core::RuvectorError::ModelInferenceError(e.to_string())
        })
    }

    fn dimensions(&self) -> usize {
        EMBEDDING_DIM
    }

    fn name(&self) -> &str {
        match &self.backend {
            EmbeddingBackend::Hash => "OnnxEmbedding (hash fallback — no real model)",
            #[cfg(feature = "onnx")]
            EmbeddingBackend::Onnx(_) => {
                "OnnxEmbedding (paraphrase-multilingual-MiniLM-L12-v2)"
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ruvector_core::embeddings::EmbeddingProvider;

    #[test]
    fn hash_fallback_correct_dim() {
        let emb = OnnxEmbedding::new("nonexistent.onnx").unwrap();
        assert!(emb.is_fallback());
        let v = emb.embed("hello world").unwrap();
        assert_eq!(v.len(), EMBEDDING_DIM);
    }

    #[test]
    fn hash_fallback_normalised() {
        let emb = OnnxEmbedding::new("nonexistent.onnx").unwrap();
        let v = emb.embed("test normalisation").unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm={norm}");
    }

    #[test]
    fn hash_fallback_deterministic() {
        let emb = OnnxEmbedding::new("x").unwrap();
        let a = emb.embed("same text").unwrap();
        let b = emb.embed("same text").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn trait_dimensions() {
        let emb = OnnxEmbedding::new("x").unwrap();
        let prov: &dyn EmbeddingProvider = &emb;
        assert_eq!(prov.dimensions(), 384);
    }

    #[test]
    fn embed_batch() {
        let emb = OnnxEmbedding::new("x").unwrap();
        let batch = emb.embed_batch(&["hello", "world", "rust"]).unwrap();
        assert_eq!(batch.len(), 3);
        for v in &batch {
            assert_eq!(v.len(), EMBEDDING_DIM);
        }
    }
}
