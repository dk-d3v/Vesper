//! Configuration loading from environment variables via dotenvy.
//! No values are ever hardcoded here.

use crate::error::AiAssistantError;

/// Runtime configuration loaded from the environment.
#[derive(Debug, Clone)]
pub struct Config {
    /// Anthropic API key — sourced from `ANTHROPIC_API_KEY`
    pub anthropic_api_key: String,
    /// Base URL for the Anthropic API — sourced from `ANTHROPIC_BASE_URL`
    pub anthropic_base_url: String,
    /// Claude model identifier — sourced from `CLAUDE_MODEL`
    pub claude_model: String,
    /// Path to the local ONNX embedding model — sourced from `EMBEDDING_MODEL_PATH`
    pub embedding_model_path: String,
    /// Path to the multilingual NER ONNX model directory — sourced from `NER_MODEL_PATH`
    pub ner_model_path: String,
    /// Extended thinking budget in tokens — sourced from `CLAUDE_THINKING_BUDGET_TOKENS`.
    /// `0` means extended thinking is disabled.
    pub thinking_budget_tokens: u32,
    /// Enable adaptive thinking mode — sourced from `CLAUDE_THINKING_ADAPTIVE`.
    /// When `true`, sends `thinking: {type: "adaptive"}` instead of manual budget mode.
    /// Supported on Claude Opus 4.6 and Sonnet 4.6. Takes precedence over `thinking_budget_tokens`.
    pub thinking_adaptive: bool,
    /// Adaptive thinking effort level — sourced from `CLAUDE_THINKING_EFFORT`.
    /// Valid values: `"low"` | `"medium"` | `"high"`. Invalid values silently become `None`.
    /// NOTE: `"max"` is intentionally excluded — it is Opus 4.6 only and returns an API
    /// error on Sonnet 4.6 and all other models. Users who explicitly need `max` must pass
    /// the raw API parameter themselves rather than relying on this config field.
    /// Sent as `output_config: {effort: "..."}` alongside `thinking: {type: "adaptive"}`.
    pub thinking_effort: Option<String>,
}

/// Load configuration purely from already-set environment variables.
///
/// Does **not** call `dotenvy::dotenv()` — useful in tests that need to
/// control the env precisely via [`std::env::set_var`] / [`std::env::remove_var`].
///
/// # Errors
/// Returns [`AiAssistantError::Config`] if required variables are missing or invalid.
pub fn load_config_from_env() -> Result<Config, AiAssistantError> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| AiAssistantError::Config("ANTHROPIC_API_KEY not set".to_string()))?;

    if api_key.is_empty() {
        return Err(AiAssistantError::Config(
            "ANTHROPIC_API_KEY is empty".to_string(),
        ));
    }

    let base_url = std::env::var("ANTHROPIC_BASE_URL")
        .unwrap_or_else(|_| "https://api.anthropic.com".to_string());

    if !base_url.starts_with("http://") && !base_url.starts_with("https://") {
        return Err(AiAssistantError::Config(
            "ANTHROPIC_BASE_URL must start with http:// or https://".to_string(),
        ));
    }

    // SECURITY: warn when a plaintext HTTP endpoint is configured.
    // The API key travels in the `x-api-key` header, which would be exposed
    // in cleartext on http:// connections. Only acceptable on localhost
    // for local-proxy development setups.
    if base_url.starts_with("http://") {
        eprintln!(
            "WARNING: ANTHROPIC_BASE_URL uses plaintext http://. \
             The API key will be transmitted without TLS encryption. \
             Set ANTHROPIC_BASE_URL=https://api.anthropic.com for production."
        );
    }

    let claude_model = std::env::var("CLAUDE_MODEL")
        .unwrap_or_else(|_| "claude-opus-4-6".to_string());

    let embedding_model_path = std::env::var("EMBEDDING_MODEL_PATH")
        .unwrap_or_else(|_| "./models/paraphrase-multilingual-MiniLM-L12-v2".to_string());

    let ner_model_path = std::env::var("NER_MODEL_PATH")
        .unwrap_or_else(|_| "./models/multilingual-ner".to_string());

    let thinking_budget_tokens = std::env::var("CLAUDE_THINKING_BUDGET_TOKENS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(0);

    let thinking_adaptive = std::env::var("CLAUDE_THINKING_ADAPTIVE")
        .map(|v| v.to_lowercase() == "true")
        .unwrap_or(false);

    // "max" is deliberately excluded: it is Opus 4.6 only and the API returns an error
    // when used with Sonnet 4.6 or any other model. Silently ignore it so misconfigured
    // environments degrade gracefully to the API default ("high") instead of hard-failing.
    let thinking_effort = std::env::var("CLAUDE_THINKING_EFFORT")
        .ok()
        .filter(|v| matches!(v.as_str(), "low" | "medium" | "high"));

    Ok(Config {
        anthropic_api_key: api_key,
        anthropic_base_url: base_url,
        claude_model,
        embedding_model_path,
        ner_model_path,
        thinking_budget_tokens,
        thinking_adaptive,
        thinking_effort,
    })
}

/// Load configuration from the environment (`.env` + system env vars).
///
/// Loads `.env` via `dotenvy` first (ignoring errors if the file is absent),
/// then delegates to [`load_config_from_env`].
///
/// # Errors
/// Returns [`AiAssistantError::Config`] if required variables are missing or invalid.
pub fn load_config() -> Result<Config, AiAssistantError> {
    // Load .env if present; ignore the error — variables may already be set externally.
    let _ = dotenvy::dotenv();
    load_config_from_env()
}

// ── Pipeline thresholds ────────────────────────────────────────────────────

/// Coherence energy below this value → reflex (fast) path.
pub const REFLEX_THRESHOLD: f64 = 0.3;

/// Coherence energy above this value → critical halt.
pub const CRITICAL_THRESHOLD: f64 = 0.8;

/// Similarity score below which a generated claim is flagged as a hallucination.
pub const HALLUCINATION_THRESHOLD: f64 = 0.7;

/// Maximum tokens allowed in a single Claude request.
pub const MAX_TOKENS: usize = 25000;

/// Maximum allowed length (characters) for user input.
pub const MAX_INPUT_LENGTH: usize = 32_768;

/// Minimum cosine-similarity score for a RAG chunk to be included.
pub const MIN_RAG_RELEVANCE: f32 = 0.7;

/// Number of top-k results to retrieve from the graph-RAG index.
pub const RAG_TOP_K: usize = 5;

/// Maximum graph-traversal hops during RAG expansion.
pub const RAG_MAX_HOPS: usize = 3;

/// Number of episodic-memory results to retrieve per query.
pub const MEMORY_TOP_K: usize = 10;

/// Number of skill results to retrieve per query.
pub const SKILL_TOP_K: usize = 5;

/// Maximum number of conversation turns kept in the in-memory session buffer.
pub const MAX_SESSION_TURNS: usize = 50;

// ── Exe-relative path helpers ──────────────────────────────────────────────

/// Returns the directory containing the running executable.
///
/// Falls back to `"."` if the path cannot be determined (e.g. in unit tests).
/// Use this to resolve resource paths (`.env`, `models/`, `prompts/`) so the
/// binary works correctly regardless of the current working directory.
pub fn exe_dir() -> std::path::PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."))
}
