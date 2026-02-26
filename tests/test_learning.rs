//! Tests for [`ai_assistant::learning`]

use ai_assistant::learning::LearningEngine;

/// Test 1: LearningEngine::new() succeeds.
#[test]
fn test_learning_engine_new_succeeds() {
    let engine = LearningEngine::new();
    assert!(engine.is_ok(), "LearningEngine::new() should succeed");
}

/// Test 2: record_trajectory() returns LearningResult.
#[test]
fn test_record_trajectory_returns_result() {
    let mut engine = LearningEngine::new().unwrap();
    let result = engine.record_trajectory(
        "What is Rust?",
        "Rust is a systems programming language.",
        "Context: Rust\nUser: What is Rust?",
        "Rust is a memory-safe systems programming language developed by Mozilla.",
    );
    assert!(result.is_ok(), "record_trajectory should succeed");
}

/// Test 3: quality_score is between 0.0 and 1.0.
#[test]
fn test_quality_score_in_range() {
    let mut engine = LearningEngine::new().unwrap();
    let result = engine
        .record_trajectory(
            "user question",
            "some context",
            "assembled prompt",
            "This is a reasonable response with enough length to score well.",
        )
        .unwrap();

    assert!(
        result.quality_score >= 0.0,
        "quality_score should be >= 0.0, got {}",
        result.quality_score
    );
    assert!(
        result.quality_score <= 1.0,
        "quality_score should be <= 1.0, got {}",
        result.quality_score
    );
}

/// Test 4: find_patterns() returns Ok without panic.
#[test]
fn test_find_patterns_ok_no_panic() {
    let mut engine = LearningEngine::new().unwrap();

    // Record a few trajectories first
    for i in 0..3 {
        engine
            .record_trajectory(
                &format!("question {}", i),
                "context",
                "prompt",
                "response with enough length to be meaningful here",
            )
            .unwrap();
    }

    let result = engine.find_patterns();
    assert!(result.is_ok(), "find_patterns should succeed without panic");
}

/// Test 5: Trajectory ID is non-empty.
#[test]
fn test_trajectory_id_non_empty() {
    let mut engine = LearningEngine::new().unwrap();
    let result = engine
        .record_trajectory("q", "ctx", "prompt", "response")
        .unwrap();
    assert!(
        !result.trajectory_id.is_empty(),
        "Trajectory ID should not be empty"
    );
}

/// Test 6: Short response has lower quality than long response.
#[test]
fn test_short_response_lower_quality_than_long() {
    let mut engine = LearningEngine::new().unwrap();

    let short_result = engine
        .record_trajectory("q", "c", "p", "short")
        .unwrap();

    let long_response = "This is a very detailed and comprehensive response that explains \
        the topic thoroughly with multiple sentences and paragraphs. It contains \
        substantial information about the subject matter and provides good coverage.";
    let long_result = engine
        .record_trajectory("q", "c", "p", long_response)
        .unwrap();

    assert!(
        short_result.quality_score <= long_result.quality_score,
        "Short response quality ({}) should be <= long response quality ({})",
        short_result.quality_score,
        long_result.quality_score
    );
}

/// Extra: trajectory IDs are sequential and unique.
#[test]
fn test_trajectory_ids_are_unique() {
    let mut engine = LearningEngine::new().unwrap();

    let id1 = engine.record_trajectory("q1", "c", "p", "r").unwrap().trajectory_id;
    let id2 = engine.record_trajectory("q2", "c", "p", "r").unwrap().trajectory_id;
    let id3 = engine.record_trajectory("q3", "c", "p", "r").unwrap().trajectory_id;

    assert_ne!(id1, id2, "Trajectory IDs should be unique");
    assert_ne!(id2, id3, "Trajectory IDs should be unique");
    assert_ne!(id1, id3, "Trajectory IDs should be unique");
}

/// Extra: Low-quality phrase caps quality score at 0.2.
#[test]
fn test_low_quality_phrase_caps_score() {
    let mut engine = LearningEngine::new().unwrap();

    // "i don't know" should cap at 0.2
    let result = engine
        .record_trajectory(
            "q",
            "c",
            "p",
            "I don't know the answer to this question at all.",
        )
        .unwrap();

    assert_eq!(
        result.quality_score, 0.2,
        "Response with 'i don't know' should score exactly 0.2"
    );
}

/// Extra: quality_score for quality_score = 0 when empty response.
#[test]
fn test_empty_response_has_zero_quality() {
    let mut engine = LearningEngine::new().unwrap();
    let result = engine.record_trajectory("q", "c", "p", "").unwrap();
    assert_eq!(result.quality_score, 0.0, "Empty response should have quality 0.0");
}
