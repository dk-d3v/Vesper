//! Tests for [`ai_assistant::embedding`]

use ai_assistant::embedding::{OnnxEmbedding, EMBEDDING_DIM};
use ruvector_core::embeddings::EmbeddingProvider;

/// Test 1: OnnxEmbedding::new("nonexistent") uses fallback (no panic).
#[test]
fn test_new_nonexistent_path_uses_fallback() {
    let emb = OnnxEmbedding::new("nonexistent_model_path.onnx")
        .expect("Should succeed with hash fallback");
    assert!(emb.is_fallback(), "Should use hash fallback when model file missing");
}

/// Test 2: embed() returns Vec with EMBEDDING_DIM=384 elements.
#[test]
fn test_embed_returns_384_dimensions() {
    let emb = OnnxEmbedding::new("nonexistent.onnx").unwrap();
    let vec = emb.embed("test text for embedding").unwrap();
    assert_eq!(
        vec.len(),
        EMBEDDING_DIM,
        "Embedding should have {EMBEDDING_DIM} dimensions, got {}",
        vec.len()
    );
    assert_eq!(EMBEDDING_DIM, 384, "EMBEDDING_DIM constant should be 384");
}

/// Test 3: embed() output is L2-normalized (length ≈ 1.0).
#[test]
fn test_embed_is_l2_normalized() {
    let emb = OnnxEmbedding::new("nonexistent.onnx").unwrap();
    let vec = emb.embed("normalize me please").unwrap();
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "Embedding should be L2-normalized; norm={norm}"
    );
}

/// Test 4: embed("same text") == embed("same text") → deterministic.
#[test]
fn test_embed_is_deterministic() {
    let emb = OnnxEmbedding::new("x").unwrap();
    let a = emb.embed("same text for determinism check").unwrap();
    let b = emb.embed("same text for determinism check").unwrap();
    assert_eq!(a, b, "Embedding should be deterministic for identical inputs");
}

/// Test 5: embed("text A") != embed("text B") → different texts produce different embeddings.
#[test]
fn test_embed_different_texts_differ() {
    let emb = OnnxEmbedding::new("x").unwrap();
    let a = emb.embed("the quick brown fox").unwrap();
    let b = emb.embed("a completely different sentence about cats").unwrap();
    assert_ne!(a, b, "Different texts should produce different embeddings");
}

/// Test 6: embed_batch() returns correct count.
#[test]
fn test_embed_batch_returns_correct_count() {
    let emb = OnnxEmbedding::new("x").unwrap();
    let texts = &["hello", "world", "rust", "testing"];
    let batch = emb.embed_batch(texts).unwrap();
    assert_eq!(batch.len(), texts.len(), "Batch should return same count as input");
    for (i, vec) in batch.iter().enumerate() {
        assert_eq!(
            vec.len(),
            EMBEDDING_DIM,
            "Batch item {i} should have {EMBEDDING_DIM} dimensions"
        );
    }
}

/// Test 7: EmbeddingProvider trait dims() == 384.
#[test]
fn test_embedding_provider_trait_dims() {
    let emb = OnnxEmbedding::new("x").unwrap();
    let provider: &dyn EmbeddingProvider = &emb;
    assert_eq!(
        provider.dimensions(),
        384,
        "EmbeddingProvider::dimensions() should return 384"
    );
}

/// Extra: embed_batch with empty slice returns empty vec.
#[test]
fn test_embed_batch_empty_input() {
    let emb = OnnxEmbedding::new("x").unwrap();
    let batch = emb.embed_batch(&[]).unwrap();
    assert!(batch.is_empty(), "Empty batch input should return empty vec");
}

/// Extra: model_path() returns the path provided to new().
#[test]
fn test_model_path_accessor() {
    let path = "some/model/path";
    let emb = OnnxEmbedding::new(path).unwrap();
    assert_eq!(emb.model_path(), path);
}

/// Extra: embed empty string works without panic.
#[test]
fn test_embed_empty_string() {
    let emb = OnnxEmbedding::new("x").unwrap();
    let vec = emb.embed("").unwrap();
    assert_eq!(vec.len(), EMBEDDING_DIM, "Empty string embedding should still have correct dims");
}
