//! Tests for [`ai_assistant::types`]

use ai_assistant::types::{
    CoherenceResult, ConversationTurn, Episode, EpisodeTier, FinalPrompt, Language, Session,
    UserMessage,
};
use std::time::SystemTime;

// ── Session tests ─────────────────────────────────────────────────────────────

fn make_turn(text: &str) -> ConversationTurn {
    ConversationTurn {
        user_message: UserMessage {
            text: text.to_string(),
            language: Language::English,
            timestamp: SystemTime::now(),
        },
        response_text: format!("Response to: {}", text),
        timestamp: SystemTime::now(),
    }
}

/// Test 1: Session::add_turn() appends a turn.
#[test]
fn test_session_add_turn_appends() {
    let mut session = Session::new();
    assert_eq!(session.turns.len(), 0);

    session.add_turn(make_turn("hello"));
    assert_eq!(session.turns.len(), 1);
    assert_eq!(session.turn_count, 1);

    session.add_turn(make_turn("world"));
    assert_eq!(session.turns.len(), 2);
    assert_eq!(session.turn_count, 2);
}

/// Test 2: Session::add_turn() trims to 50 turns max.
#[test]
fn test_session_add_turn_trims_to_50() {
    let mut session = Session::new();

    // Add 55 turns
    for i in 0..55 {
        session.add_turn(make_turn(&format!("turn {}", i)));
    }

    assert_eq!(session.turns.len(), 50, "Turns should be trimmed to 50");
    assert_eq!(session.turn_count, 55, "turn_count tracks total including trimmed");

    // The first turn should be turn 5 (turns 0..4 were removed)
    assert!(session.turns[0].user_message.text.contains("turn 5"));
}

// ── FinalPrompt tests ─────────────────────────────────────────────────────────

/// Test 3: FinalPrompt::full_content() with empty context returns only user_text.
#[test]
fn test_final_prompt_full_content_empty_context() {
    let prompt = FinalPrompt {
        system: "System prompt".to_string(),
        memory_context: String::new(),
        reasoning_hints: String::new(),
        context: String::new(),
        user_text: "Hello, AI!".to_string(),
        estimated_tokens: 10,
    };

    let content = prompt.full_content();
    assert_eq!(content, "Hello, AI!");
}

/// Test 4: FinalPrompt::full_content() with non-empty context includes both.
#[test]
fn test_final_prompt_full_content_with_context() {
    let prompt = FinalPrompt {
        system: "System".to_string(),
        memory_context: String::new(),
        reasoning_hints: String::new(),
        context: "Some background info".to_string(),
        user_text: "My question".to_string(),
        estimated_tokens: 20,
    };

    let content = prompt.full_content();
    assert!(content.contains("Context:"), "Should include 'Context:' header");
    assert!(content.contains("Some background info"), "Should include context text");
    assert!(content.contains("My question"), "Should include user text");
}

// ── EpisodeTier tests ──────────────────────────────────────────────────────────

/// Test 5: EpisodeTier derives Clone, PartialEq.
#[test]
fn test_episode_tier_clone_partial_eq() {
    let hot = EpisodeTier::Hot;
    let hot_clone = hot.clone();
    assert_eq!(hot, hot_clone);

    let warm = EpisodeTier::Warm;
    assert_ne!(hot, warm);

    let cold = EpisodeTier::Cold;
    assert_ne!(warm, cold);
    assert_ne!(hot, cold);
}

// ── CoherenceResult tests ──────────────────────────────────────────────────────

/// Test 6: CoherenceResult variants are correct.
#[test]
fn test_coherence_result_variants() {
    let reflex = CoherenceResult::Reflex;
    let revised = CoherenceResult::Revised("new context".to_string());
    let halt = CoherenceResult::Halt;

    // Verify pattern matching works
    match reflex {
        CoherenceResult::Reflex => {}
        _ => panic!("Expected Reflex"),
    }

    match revised {
        CoherenceResult::Revised(s) => assert_eq!(s, "new context"),
        _ => panic!("Expected Revised"),
    }

    match halt {
        CoherenceResult::Halt => {}
        _ => panic!("Expected Halt"),
    }
}

// ── Episode serialization tests ───────────────────────────────────────────────

/// Test 7: Episode serialization round-trip (serde).
#[test]
fn test_episode_serde_round_trip() {
    let episode = Episode {
        id: "test-id-123".to_string(),
        text: "Test episode text".to_string(),
        embedding: vec![0.1, 0.2, 0.3],
        timestamp: SystemTime::UNIX_EPOCH,
        tier: EpisodeTier::Hot,
        quality_score: 0.85,
    };

    let serialized = serde_json::to_string(&episode).expect("Serialization should succeed");
    let deserialized: Episode =
        serde_json::from_str(&serialized).expect("Deserialization should succeed");

    assert_eq!(deserialized.id, episode.id);
    assert_eq!(deserialized.text, episode.text);
    assert_eq!(deserialized.tier, episode.tier);
    assert!((deserialized.quality_score - episode.quality_score).abs() < 1e-6);
    assert_eq!(deserialized.embedding.len(), episode.embedding.len());
}

// ── Language tests ────────────────────────────────────────────────────────────

#[test]
fn test_language_clone_partial_eq() {
    assert_eq!(Language::Turkish, Language::Turkish);
    assert_eq!(Language::English, Language::English);
    assert_ne!(Language::Turkish, Language::English);

    let other = Language::Other("fr".to_string());
    let other_clone = other.clone();
    assert_eq!(other, other_clone);
}

#[test]
fn test_language_serde_round_trip() {
    let langs = vec![
        Language::Turkish,
        Language::English,
        Language::Other("de".to_string()),
    ];

    for lang in langs {
        let json = serde_json::to_string(&lang).expect("should serialize");
        let back: Language = serde_json::from_str(&json).expect("should deserialize");
        assert_eq!(lang, back);
    }
}
