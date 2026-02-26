//! Tests for [`ai_assistant::config`]
//!
//! Env-var tests use a process-wide `Mutex` to run serially even under the
//! default multi-threaded test harness (`cargo test`).

use ai_assistant::config::{
    load_config, load_config_from_env, CRITICAL_THRESHOLD, HALLUCINATION_THRESHOLD,
    MAX_INPUT_LENGTH, MAX_TOKENS, MEMORY_TOP_K, MIN_RAG_RELEVANCE, RAG_MAX_HOPS, RAG_TOP_K,
    REFLEX_THRESHOLD, SKILL_TOP_K,
};
use std::sync::{Mutex, MutexGuard};

// ── Serialiser ────────────────────────────────────────────────────────────────

static ENV_MUTEX: Mutex<()> = Mutex::new(());

fn lock_env() -> MutexGuard<'static, ()> {
    ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner())
}

// ── Helper: guard that restores env vars on drop ──────────────────────────────

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

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Test 1: load_config() fails when ANTHROPIC_API_KEY is missing.
#[test]
fn test_load_config_fails_missing_api_key() {
    let _lock = lock_env();
    let _g = EnvGuard::remove("ANTHROPIC_API_KEY");
    let _g2 = EnvGuard::set("ANTHROPIC_BASE_URL", "https://api.anthropic.com");

    // Use load_config_from_env so dotenv() doesn't re-inject .env values
    // after our EnvGuard::remove() call.
    let result = load_config_from_env();
    assert!(result.is_err(), "Expected error with missing API key");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("ANTHROPIC_API_KEY"),
        "Error should mention ANTHROPIC_API_KEY, got: {msg}"
    );
}

/// Test 2: load_config() succeeds with all required env vars set.
#[test]
fn test_load_config_succeeds_with_all_vars() {
    let _lock = lock_env();
    let _key = EnvGuard::set("ANTHROPIC_API_KEY", "test-mock-key-not-real");
    let _url = EnvGuard::set("ANTHROPIC_BASE_URL", "https://api.anthropic.com");
    let _model = EnvGuard::set("CLAUDE_MODEL", "claude-mock-test");
    let _emb = EnvGuard::set("EMBEDDING_MODEL_PATH", "./models/test");

    let result = load_config();
    assert!(result.is_ok(), "Expected Ok, got: {:?}", result.err());

    let cfg = result.unwrap();
    assert_eq!(cfg.anthropic_api_key, "test-mock-key-not-real");
    assert_eq!(cfg.anthropic_base_url, "https://api.anthropic.com");
    assert_eq!(cfg.claude_model, "claude-mock-test");
    assert_eq!(cfg.embedding_model_path, "./models/test");
}

/// Test 3: ANTHROPIC_BASE_URL must start with http:// or https://.
#[test]
fn test_load_config_invalid_base_url() {
    let _lock = lock_env();
    let _key = EnvGuard::set("ANTHROPIC_API_KEY", "mock-key");
    let _url = EnvGuard::set("ANTHROPIC_BASE_URL", "ftp://bad-url.com");

    let result = load_config();
    assert!(result.is_err(), "Expected error for ftp:// URL");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("http://") || msg.contains("https://"),
        "Error should mention http/https requirement, got: {msg}"
    );
}

/// Test 4: Empty ANTHROPIC_API_KEY returns error.
#[test]
fn test_load_config_empty_api_key_returns_error() {
    let _lock = lock_env();
    let _key = EnvGuard::set("ANTHROPIC_API_KEY", "");
    let _url = EnvGuard::set("ANTHROPIC_BASE_URL", "https://api.anthropic.com");

    let result = load_config();
    assert!(result.is_err(), "Expected error for empty API key");
}

/// Test 5: Default values are used when optional vars are missing.
#[test]
fn test_load_config_defaults_for_optional_vars() {
    let _lock = lock_env();
    let _key = EnvGuard::set("ANTHROPIC_API_KEY", "mock-key");
    let _url = EnvGuard::remove("ANTHROPIC_BASE_URL");
    let _model = EnvGuard::remove("CLAUDE_MODEL");
    let _emb = EnvGuard::remove("EMBEDDING_MODEL_PATH");

    // Use load_config_from_env so dotenv() doesn't re-inject .env values
    // after our EnvGuard::remove() calls.
    let result = load_config_from_env();
    assert!(result.is_ok(), "Expected Ok with defaults, got: {:?}", result.err());

    let cfg = result.unwrap();
    assert!(
        cfg.anthropic_base_url.starts_with("https://"),
        "Default base URL should be https://"
    );
    assert!(!cfg.claude_model.is_empty(), "Default claude_model should be non-empty");
    assert!(!cfg.embedding_model_path.is_empty(), "Default embedding path should be non-empty");
}

/// Test 6: Constants have expected values.
#[test]
fn test_constants_have_expected_values() {
    assert_eq!(REFLEX_THRESHOLD, 0.3, "REFLEX_THRESHOLD should be 0.3");
    assert_eq!(CRITICAL_THRESHOLD, 0.8, "CRITICAL_THRESHOLD should be 0.8");
    assert_eq!(HALLUCINATION_THRESHOLD, 0.7, "HALLUCINATION_THRESHOLD should be 0.7");
    assert_eq!(MAX_TOKENS, 4096, "MAX_TOKENS should be 4096");
    assert_eq!(MAX_INPUT_LENGTH, 32_768, "MAX_INPUT_LENGTH should be 32768");
    assert_eq!(MIN_RAG_RELEVANCE, 0.7, "MIN_RAG_RELEVANCE should be 0.7");
    assert_eq!(RAG_TOP_K, 5, "RAG_TOP_K should be 5");
    assert_eq!(RAG_MAX_HOPS, 3, "RAG_MAX_HOPS should be 3");
    assert_eq!(MEMORY_TOP_K, 10, "MEMORY_TOP_K should be 10");
    assert_eq!(SKILL_TOP_K, 5, "SKILL_TOP_K should be 5");
}

/// Test 7: CLAUDE_THINKING_BUDGET_TOKENS=5000 sets thinking_budget_tokens to 5000.
#[test]
fn test_thinking_budget_tokens_parsed_from_env() {
    let _lock = lock_env();
    let _key = EnvGuard::set("ANTHROPIC_API_KEY", "mock-key");
    let _url = EnvGuard::set("ANTHROPIC_BASE_URL", "https://api.anthropic.com");
    let _budget = EnvGuard::set("CLAUDE_THINKING_BUDGET_TOKENS", "5000");

    let result = load_config_from_env();
    assert!(result.is_ok(), "Expected Ok, got: {:?}", result.err());
    let cfg = result.unwrap();
    assert_eq!(cfg.thinking_budget_tokens, 5000, "thinking_budget_tokens should be 5000");
}

/// Test 8: Unset CLAUDE_THINKING_BUDGET_TOKENS defaults to 0 (disabled).
#[test]
fn test_thinking_budget_tokens_defaults_to_zero() {
    let _lock = lock_env();
    let _key = EnvGuard::set("ANTHROPIC_API_KEY", "mock-key");
    let _url = EnvGuard::set("ANTHROPIC_BASE_URL", "https://api.anthropic.com");
    let _budget = EnvGuard::remove("CLAUDE_THINKING_BUDGET_TOKENS");

    let result = load_config_from_env();
    assert!(result.is_ok(), "Expected Ok, got: {:?}", result.err());
    let cfg = result.unwrap();
    assert_eq!(cfg.thinking_budget_tokens, 0, "thinking_budget_tokens should be 0 when unset");
}

/// Test 9: Invalid CLAUDE_THINKING_BUDGET_TOKENS (non-numeric) falls back to 0.
#[test]
fn test_thinking_budget_tokens_invalid_value_defaults_to_zero() {
    let _lock = lock_env();
    let _key = EnvGuard::set("ANTHROPIC_API_KEY", "mock-key");
    let _url = EnvGuard::set("ANTHROPIC_BASE_URL", "https://api.anthropic.com");
    let _budget = EnvGuard::set("CLAUDE_THINKING_BUDGET_TOKENS", "not-a-number");

    let result = load_config_from_env();
    assert!(result.is_ok(), "Expected Ok with fallback, got: {:?}", result.err());
    let cfg = result.unwrap();
    assert_eq!(cfg.thinking_budget_tokens, 0, "Invalid value should fall back to 0");
}
