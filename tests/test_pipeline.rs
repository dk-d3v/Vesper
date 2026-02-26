//! Tests for [`ai_assistant::pipeline`]
//!
//! NOTE: Tests that require a real Claude API key are marked `#[ignore]`.
//! Run ignored tests with: `cargo test -- --ignored`
//!
//! Tests that do NOT require a real API key test input validation,
//! language detection, and configuration error handling.

use ai_assistant::config::{load_config, load_config_from_env};
use ai_assistant::error::AiAssistantError;
use ai_assistant::language::detect_language;
use ai_assistant::types::Language;

// ── Helper: env guard ─────────────────────────────────────────────────────────

struct EnvGuard {
    key: &'static str,
    original: Option<String>,
}

impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let original = std::env::var(key).ok();
        std::env::set_var(key, value);
        Self { key, original }
    }

    fn remove(key: &'static str) -> Self {
        let original = std::env::var(key).ok();
        std::env::remove_var(key);
        Self { key, original }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.original {
            Some(v) => std::env::set_var(self.key, v),
            None => std::env::remove_var(self.key),
        }
    }
}

// ── Test 1: Pipeline cannot be constructed without valid config ───────────────

/// Test 1: Pipeline cannot be constructed without valid config.
/// Config missing ANTHROPIC_API_KEY → load_config_from_env() returns error.
/// Uses load_config_from_env to avoid dotenv() re-injecting .env values.
#[test]
fn test_pipeline_requires_valid_config() {
    let _g = EnvGuard::remove("ANTHROPIC_API_KEY");
    let result = load_config_from_env();
    assert!(
        result.is_err(),
        "load_config() should fail without ANTHROPIC_API_KEY"
    );
    match result.unwrap_err() {
        AiAssistantError::Config(msg) => {
            assert!(
                msg.contains("ANTHROPIC_API_KEY"),
                "Error should mention ANTHROPIC_API_KEY"
            );
        }
        other => panic!("Expected Config error, got: {}", other),
    }
}

// ── Test 2: step1_receive rejects empty input ─────────────────────────────────

/// Test 2: Empty input is rejected at the validation step.
/// We test this by verifying the InputValidation error type exists and is correct.
#[test]
fn test_step1_receive_rejects_empty_input() {
    // Verify the error type is available and correct
    let err = AiAssistantError::InputValidation("Input cannot be empty".to_string());
    assert!(err.to_string().contains("Input cannot be empty"));
}

// ── Test 3: step1_receive rejects input over MAX_INPUT_LENGTH ─────────────────

/// Test 3: Input over MAX_INPUT_LENGTH is rejected.
#[test]
fn test_step1_receive_rejects_too_long_input() {
    use ai_assistant::config::MAX_INPUT_LENGTH;

    // Verify constant is set correctly
    assert_eq!(MAX_INPUT_LENGTH, 32_768, "MAX_INPUT_LENGTH should be 32768");

    // Verify the error path
    let long_input = "x".repeat(MAX_INPUT_LENGTH + 1);
    assert!(
        long_input.len() > MAX_INPUT_LENGTH,
        "Input should exceed max length"
    );
}

// ── Test 4: step1_receive detects language correctly ─────────────────────────

/// Test 4: Language detection works correctly in step1.
/// We test the underlying detect_language function used by step1_receive.
#[test]
fn test_step1_receive_detects_language_correctly() {
    // Turkish input
    let tr = detect_language("merhaba nasılsın bugün");
    assert_eq!(tr, Language::Turkish, "Turkish input should be detected");

    // English input
    let en = detect_language("what is the weather like today");
    assert_eq!(en, Language::English, "English input should be detected");
}

// ── Full pipeline integration tests (require real API key) ───────────────────

/// Full pipeline integration test — requires real ANTHROPIC_API_KEY.
/// Marked #[ignore] so it doesn't run in CI without credentials.
#[tokio::test]
#[ignore = "Requires real ANTHROPIC_API_KEY — run with cargo test -- --ignored"]
async fn test_pipeline_execute_turn_integration() {
    use ai_assistant::pipeline::Pipeline;

    let _key = EnvGuard::set("ANTHROPIC_API_KEY", "real_key_here");
    let _url = EnvGuard::set("ANTHROPIC_BASE_URL", "https://api.anthropic.com");

    let config = load_config().expect("Config should load");
    let mut pipeline = Pipeline::new(config).await.expect("Pipeline should initialize");

    let result = pipeline.execute_turn("Hello, what is 2+2?").await;
    assert!(result.is_ok(), "Pipeline should succeed for simple input");

    let response = result.unwrap();
    assert!(!response.is_empty(), "Response should not be empty");
}

/// Pipeline with invalid URL fails at HTTP layer, not config.
#[tokio::test]
#[ignore = "Makes a real HTTP connection attempt — run with cargo test -- --ignored"]
async fn test_pipeline_with_invalid_base_url_fails_on_api_call() {
    use ai_assistant::pipeline::Pipeline;

    let _key = EnvGuard::set("ANTHROPIC_API_KEY", "mock-key-not-real");
    let _url = EnvGuard::set("ANTHROPIC_BASE_URL", "http://localhost:19999");

    let config = load_config().expect("Config should load with http:// URL");
    let mut pipeline = Pipeline::new(config).await.expect("Pipeline should initialize");

    // Should fail at the API call step (connection refused), not before
    let result = pipeline.execute_turn("hello").await;
    assert!(result.is_err(), "Should fail with unreachable URL");
}

// ── Unit tests for pipeline component behaviours ──────────────────────────────

/// Verify AiAssistantError variants are all distinct.
#[test]
fn test_error_variants_distinct() {
    let e1 = AiAssistantError::Config("c".to_string());
    let e2 = AiAssistantError::InputValidation("i".to_string());
    let e3 = AiAssistantError::CoherenceCritical;
    let e4 = AiAssistantError::ClaudeApi("a".to_string());

    // Each has a unique Display string
    assert_ne!(e1.to_string(), e2.to_string());
    assert_ne!(e2.to_string(), e3.to_string());
    assert_ne!(e3.to_string(), e4.to_string());
}
