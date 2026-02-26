//! Multilingual NER — ONNX-backed BertForTokenClassification.
//!
//! Model: `models/multilingual-ner/` (mBERT, 9 BIO labels).
//! Label map (from config.json `id2label`):
//!   0=O  1=B-PER  2=I-PER  3=B-ORG  4=I-ORG  5=B-LOC  6=I-LOC  7=B-MISC  8=I-MISC
//!
//! Falls back to empty results when the `onnx` feature is disabled or the
//! model files cannot be loaded.

use crate::error::AiAssistantError;

const MAX_SEQ_LEN: usize = 512;

// ── EntityType ────────────────────────────────────────────────────────────────

/// A named entity category.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EntityType {
    Person,
    Location,
    Organization,
    Other,
}

impl EntityType {
    /// Graph node label for this entity type.
    pub fn label(&self) -> &'static str {
        match self {
            EntityType::Person => "Person",
            EntityType::Location => "Location",
            EntityType::Organization => "Organization",
            EntityType::Other => "Other",
        }
    }
}

// id2label mapping: 0=O, 1=B-PER, 2=I-PER, 3=B-ORG, 4=I-ORG,
//                   5=B-LOC, 6=I-LOC, 7=B-MISC, 8=I-MISC
fn label_id_to_entity_type(id: usize) -> Option<EntityType> {
    match id {
        1 | 2 => Some(EntityType::Person),
        3 | 4 => Some(EntityType::Organization),
        5 | 6 => Some(EntityType::Location),
        7 | 8 => Some(EntityType::Other),
        _ => None, // O label
    }
}

fn is_begin_label(id: usize) -> bool {
    matches!(id, 1 | 3 | 5 | 7)
}

fn is_inside_label(id: usize) -> bool {
    matches!(id, 2 | 4 | 6 | 8)
}

// ── Backend ───────────────────────────────────────────────────────────────────

enum NerBackend {
    /// No model — returns empty results.
    Fallback,
    /// Real ONNX inference.
    #[cfg(feature = "onnx")]
    Onnx {
        session: std::sync::Arc<std::sync::Mutex<ort::session::Session>>,
        tokenizer: std::sync::Arc<tokenizers::Tokenizer>,
    },
}

// ── Public struct ─────────────────────────────────────────────────────────────

/// Multilingual named entity recogniser.
///
/// Create with [`MultilingualNer::new`], passing the directory containing
/// `model.onnx` and `tokenizer.json`.
pub struct MultilingualNer {
    backend: NerBackend,
}

// Safety: same pattern as OnnxEmbedding — Arc<Mutex<Session>> is Send+Sync.
unsafe impl Send for MultilingualNer {}
unsafe impl Sync for MultilingualNer {}

impl MultilingualNer {
    /// Load the NER model from `model_dir`.
    ///
    /// Gracefully falls back to empty-result mode when:
    /// - `onnx` feature is not enabled, or
    /// - the model files are missing or cannot be loaded.
    pub fn new(model_dir: &str) -> Result<Self, AiAssistantError> {
        let resolved: std::path::PathBuf = {
            let p = std::path::Path::new(model_dir);
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                crate::config::exe_dir().join(p)
            }
        };

        #[cfg(feature = "onnx")]
        {
            match Self::try_load(&resolved) {
                Ok((session, tokenizer)) => {
                    tracing::info!(
                        "Multilingual NER model loaded from '{}'",
                        resolved.display()
                    );
                    return Ok(Self {
                        backend: NerBackend::Onnx {
                            session: std::sync::Arc::new(std::sync::Mutex::new(session)),
                            tokenizer: std::sync::Arc::new(tokenizer),
                        },
                    });
                }
                Err(e) => {
                    tracing::warn!(
                        "NER model load failed ({}); NER running in fallback mode (no entities)",
                        e
                    );
                }
            }
        }

        #[cfg(not(feature = "onnx"))]
        tracing::warn!("onnx feature disabled — Multilingual NER unavailable");

        Ok(Self {
            backend: NerBackend::Fallback,
        })
    }

    /// Returns `true` if running without a real ONNX model.
    pub fn is_fallback(&self) -> bool {
        matches!(self.backend, NerBackend::Fallback)
    }

    /// Extract named entities from `text`.
    ///
    /// Returns `Vec<(entity_text, EntityType)>` deduplicated by lower-case text.
    /// Returns an empty `Vec` in fallback mode or on inference errors.
    pub fn extract(&self, text: &str) -> Vec<(String, EntityType)> {
        match &self.backend {
            NerBackend::Fallback => Vec::new(),
            #[cfg(feature = "onnx")]
            NerBackend::Onnx { session, tokenizer } => {
                Self::run_ner(session, tokenizer, text).unwrap_or_else(|e| {
                    tracing::warn!("NER inference failed: {}; returning empty entity list", e);
                    Vec::new()
                })
            }
        }
    }

    // ── ONNX loading ──────────────────────────────────────────────────────────

    #[cfg(feature = "onnx")]
    fn try_load(
        model_dir: &std::path::Path,
    ) -> Result<(ort::session::Session, tokenizers::Tokenizer), AiAssistantError> {
        let onnx_path = model_dir.join("model.onnx");
        if !onnx_path.exists() {
            return Err(AiAssistantError::Embedding(format!(
                "NER model.onnx not found: {}",
                onnx_path.display()
            )));
        }

        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(AiAssistantError::Embedding(format!(
                "NER tokenizer.json not found: {}",
                tokenizer_path.display()
            )));
        }

        let session = ort::session::Session::builder()
            .map_err(|e| AiAssistantError::Embedding(format!("ort builder: {e}")))?
            .commit_from_file(&onnx_path)
            .map_err(|e| AiAssistantError::Embedding(format!("ort load: {e}")))?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| AiAssistantError::Embedding(format!("NER tokenizer load: {e}")))?;

        Ok((session, tokenizer))
    }

    // ── ONNX inference ────────────────────────────────────────────────────────

    #[cfg(feature = "onnx")]
    fn run_ner(
        session: &std::sync::Arc<std::sync::Mutex<ort::session::Session>>,
        tokenizer: &std::sync::Arc<tokenizers::Tokenizer>,
        text: &str,
    ) -> Result<Vec<(String, EntityType)>, AiAssistantError> {
        use ndarray::Array2;
        use ort::value::Tensor;

        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| AiAssistantError::Embedding(format!("NER encode: {e}")))?;

        let raw_ids = encoding.get_ids();
        let raw_mask = encoding.get_attention_mask();
        let tokens: Vec<String> = encoding.get_tokens().to_vec();
        let seq_len = raw_ids.len().min(MAX_SEQ_LEN);

        if seq_len == 0 {
            return Ok(Vec::new());
        }

        let input_ids: Vec<i64> = raw_ids[..seq_len].iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = raw_mask[..seq_len].iter().map(|&x| x as i64).collect();
        let token_type_ids = vec![0i64; seq_len];

        let ids_arr = Array2::from_shape_vec((1, seq_len), input_ids)
            .map_err(|e| AiAssistantError::Embedding(e.to_string()))?;
        let mask_arr = Array2::from_shape_vec((1, seq_len), attention_mask)
            .map_err(|e| AiAssistantError::Embedding(e.to_string()))?;
        let types_arr = Array2::from_shape_vec((1, seq_len), token_type_ids)
            .map_err(|e| AiAssistantError::Embedding(e.to_string()))?;

        let ids_val = Tensor::from_array(ids_arr)
            .map_err(|e: ort::Error| AiAssistantError::Embedding(e.to_string()))?;
        let mask_val = Tensor::from_array(mask_arr)
            .map_err(|e: ort::Error| AiAssistantError::Embedding(e.to_string()))?;
        let types_val = Tensor::from_array(types_arr)
            .map_err(|e: ort::Error| AiAssistantError::Embedding(e.to_string()))?;

        let guard = session
            .lock()
            .map_err(|_| AiAssistantError::Embedding("NER session mutex poisoned".to_string()))?;

        let session_inputs = ort::inputs![
            "input_ids"      => ids_val,
            "attention_mask" => mask_val,
            "token_type_ids" => types_val
        ]
        .map_err(|e: ort::Error| AiAssistantError::Embedding(e.to_string()))?;

        let outputs = guard
            .run(session_inputs)
            .map_err(|e: ort::Error| AiAssistantError::Embedding(e.to_string()))?;

        // logits shape: [1, seq_len, num_labels=9]
        let logits = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e: ort::Error| AiAssistantError::Embedding(e.to_string()))?;

        let flat: Vec<f32> = logits.iter().copied().collect();
        const NUM_LABELS: usize = 9;

        // Argmax per token position → label_id
        let label_ids: Vec<usize> = (0..seq_len)
            .filter_map(|i| {
                let offset = i * NUM_LABELS;
                if offset + NUM_LABELS > flat.len() {
                    return None;
                }
                let slice = &flat[offset..offset + NUM_LABELS];
                Some(
                    slice
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0),
                )
            })
            .collect();

        Ok(Self::bio_to_spans(&label_ids, &tokens, seq_len))
    }

    // ── BIO span assembly ─────────────────────────────────────────────────────

    /// Convert per-token BIO label IDs into `(entity_text, EntityType)` spans.
    ///
    /// - Token 0 is `[CLS]` and the last is `[SEP]` — both skipped.
    /// - WordPiece continuation tokens (`##foo`) are concatenated without space.
    /// - Results are deduplicated by lower-case text within the same entity type.
    #[cfg(feature = "onnx")]
    fn bio_to_spans(
        label_ids: &[usize],
        tokens: &[String],
        seq_len: usize,
    ) -> Vec<(String, EntityType)> {
        let mut entities: Vec<(String, EntityType)> = Vec::new();
        let mut current: Option<(String, EntityType)> = None;

        // i=0 is [CLS], last valid index before [SEP] is seq_len-2
        let end = seq_len.min(label_ids.len()).saturating_sub(1);

        for i in 1..end {
            let label_id = label_ids[i];
            let token = tokens.get(i).map(|s| s.as_str()).unwrap_or("");

            if token.starts_with("##") {
                // WordPiece sub-word: append directly (strip ## prefix)
                if let Some((ref mut text, _)) = current {
                    text.push_str(token.trim_start_matches('#'));
                }
                continue;
            }

            if is_begin_label(label_id) {
                // Flush previous entity before starting a new one
                if let Some(e) = current.take() {
                    if !e.0.trim().is_empty() {
                        entities.push(e);
                    }
                }
                if let Some(et) = label_id_to_entity_type(label_id) {
                    current = Some((token.to_string(), et));
                }
            } else if is_inside_label(label_id) {
                match current {
                    Some((ref mut text, _)) => {
                        text.push(' ');
                        text.push_str(token);
                    }
                    None => {
                        // I- without a preceding B- — treat as entity start
                        if let Some(et) = label_id_to_entity_type(label_id) {
                            current = Some((token.to_string(), et));
                        }
                    }
                }
            } else {
                // O label — flush current entity
                if let Some(e) = current.take() {
                    if !e.0.trim().is_empty() {
                        entities.push(e);
                    }
                }
            }
        }

        // Flush any trailing entity
        if let Some(e) = current {
            if !e.0.trim().is_empty() {
                entities.push(e);
            }
        }

        // Deduplicate by (lower-case text, entity type)
        let mut seen = std::collections::HashSet::new();
        entities.retain(|(text, et)| seen.insert((text.to_lowercase(), et.label())));

        entities
    }
}
