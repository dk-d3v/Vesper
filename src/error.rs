//! Custom error types for the AI Assistant pipeline.

use thiserror::Error;

/// Unified error type propagated through every pipeline step.
#[derive(Debug, Error)]
pub enum AiAssistantError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Memory error: {0}")]
    Memory(String),

    #[error("Graph context error: {0}")]
    GraphContext(String),

    #[error("Coherence critical halt: contradiction energy too high")]
    CoherenceCritical,

    #[error("Claude API error: {0}")]
    ClaudeApi(String),

    #[error("MCP tools error: {0}")]
    McpTools(String),

    #[error("Verification error: {0}")]
    Verification(String),

    #[error("Learning error: {0}")]
    Learning(String),

    #[error("Audit error: {0}")]
    Audit(String),

    #[error("Pipeline error: {0}")]
    Pipeline(String),

    #[error("Input validation error: {0}")]
    InputValidation(String),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Environment variable error: {0}")]
    EnvVar(String),
}
