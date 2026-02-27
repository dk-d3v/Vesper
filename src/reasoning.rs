//! DagReasoningBank wrapper — stores high-quality response patterns and
//! retrieves similar ones to guide future responses.
//!
//! # Design
//! - Only stores patterns with `quality_score >= MIN_PATTERN_QUALITY` (0.7).
//! - Embeds text via `OnnxEmbedding` before storage/query.
//! - Maintains a local `HashMap<id → text>` side-map because
//!   `DagReasoningBank::store_pattern` does not expose mutable metadata.

use std::collections::HashMap;
use std::sync::Arc;

use ruvector_dag::{DagReasoningBank, ReasoningBankConfig};

use crate::{embedding::OnnxEmbedding, error::AiAssistantError};

/// Minimum quality score required to persist a pattern.
const MIN_PATTERN_QUALITY: f32 = 0.7;

/// Wrapper around [`DagReasoningBank`] that embeds text before storage
/// and keeps a `pattern_id → text` side-map for retrieval.
pub struct ReasoningStore {
    embedding: Arc<OnnxEmbedding>,
    bank: DagReasoningBank,
    /// Maps pattern ID → original response text.
    texts: HashMap<u64, String>,
}

impl ReasoningStore {
    /// Create a new store backed by the given embedding provider.
    pub fn new(embedding: Arc<OnnxEmbedding>) -> Self {
        let config = ReasoningBankConfig {
            similarity_threshold: MIN_PATTERN_QUALITY,
            ..ReasoningBankConfig::default()
        };
        Self {
            embedding,
            bank: DagReasoningBank::new(config),
            texts: HashMap::new(),
        }
    }

    /// Store a pattern only when `quality_score >= MIN_PATTERN_QUALITY`.
    ///
    /// Embeds `text`, stores the resulting vector in the bank, and saves
    /// the original text in the local side-map.
    ///
    /// Returns the assigned pattern ID on success, or `None` when the score
    /// is below threshold or embedding fails (graceful degradation).
    pub fn store_if_quality(&mut self, text: &str, quality_score: f32) -> Option<u64> {
        if quality_score < MIN_PATTERN_QUALITY {
            return None;
        }
        let vector = match self.embedding.embed(text) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("reasoning: embed failed during store: {}", e);
                return None;
            }
        };
        let id = self.bank.store_pattern(vector, quality_score);
        self.texts.insert(id, text.to_string());
        tracing::debug!(
            pattern_id = id,
            quality = quality_score,
            "reasoning: pattern stored"
        );
        Some(id)
    }

    /// Find the top-`k` most similar high-quality patterns for a query.
    ///
    /// Returns the original response texts from the side-map.
    /// Returns an empty [`Vec`] when the bank is empty or embedding fails
    /// (graceful degradation — never panics).
    pub fn find_similar(&self, query_text: &str, top_k: usize) -> Vec<String> {
        if self.bank.pattern_count() == 0 {
            return Vec::new();
        }
        let query_vec = match self.embedding.embed(query_text) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("reasoning: embed failed during query: {}", e);
                return Vec::new();
            }
        };
        self.bank
            .query_similar(&query_vec, top_k)
            .into_iter()
            .filter_map(|(id, _score)| self.texts.get(&id).cloned())
            .collect()
    }

    /// Total number of stored patterns.
    #[inline]
    pub fn pattern_count(&self) -> usize {
        self.bank.pattern_count()
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> ReasoningStore {
        let emb = Arc::new(OnnxEmbedding::new("nonexistent.onnx").unwrap());
        ReasoningStore::new(emb)
    }

    #[test]
    fn below_threshold_not_stored() {
        let mut store = make_store();
        let result = store.store_if_quality("great answer", 0.5);
        assert!(result.is_none(), "score < 0.7 should return None");
        assert_eq!(store.pattern_count(), 0);
    }

    #[test]
    fn exactly_at_threshold_stored() {
        let mut store = make_store();
        let result = store.store_if_quality("excellent answer", MIN_PATTERN_QUALITY);
        assert!(result.is_some(), "score == 0.7 should be stored");
        assert_eq!(store.pattern_count(), 1);
    }

    #[test]
    fn above_threshold_stored() {
        let mut store = make_store();
        let result = store.store_if_quality("perfect answer", 0.95);
        assert!(result.is_some());
        assert_eq!(store.pattern_count(), 1);
    }

    #[test]
    fn find_similar_on_empty_bank_returns_empty() {
        let store = make_store();
        let results = store.find_similar("some query", 3);
        assert!(results.is_empty(), "empty bank should return no results");
    }

    #[test]
    fn multiple_patterns_stored_independently() {
        let mut store = make_store();
        store.store_if_quality("First high quality answer.", 0.8);
        store.store_if_quality("Second excellent answer.", 0.9);
        store.store_if_quality("Third below threshold.", 0.5); // not stored
        assert_eq!(store.pattern_count(), 2);
    }

    #[test]
    fn find_similar_does_not_panic_on_query() {
        let mut store = make_store();
        store.store_if_quality("The capital of France is Paris.", 0.9);
        // With hash embeddings, just verify no panic
        let _results = store.find_similar("What is the capital of France?", 3);
    }
}
