//! Claude API HTTP client using reqwest.
//!
//! Claude generates text only — all business logic remains in Rust.
//! Handles multi-turn tool-use conversations with proper error mapping
//! for 401, 429, and 5xx responses.

use crate::{config::Config, error::AiAssistantError, types::{ClaudeResponse, Tool, ToolCall}};
use serde_json::json;

/// HTTP client for the Anthropic Messages API.
pub struct ClaudeClient {
    client: reqwest::Client,
}

impl Default for ClaudeClient {
    fn default() -> Self {
        Self::new()
    }
}

impl ClaudeClient {
    /// Create a new `ClaudeClient` with default reqwest settings.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /// Send a single-turn user message to Claude and return its response.
    ///
    /// `tools` is passed only when non-empty so that Claude does not receive an
    /// empty tool list (which some model versions reject).
    pub async fn send_message(
        &self,
        config: &Config,
        system: &str,
        user_content: &str,
        tools: &[Tool],
    ) -> Result<ClaudeResponse, AiAssistantError> {
        let messages = vec![json!({
            "role": "user",
            "content": user_content
        })];

        let body = self.build_body(config, system, &messages, tools);
        let raw = self.post(config, body).await?;
        Self::parse_response(raw)
    }

    /// Send a follow-up message containing tool results (multi-turn tool use).
    ///
    /// `messages` is the full conversation history including tool result blocks.
    pub async fn send_with_tool_results(
        &self,
        config: &Config,
        system: &str,
        messages: &[serde_json::Value],
        tools: &[Tool],
    ) -> Result<ClaudeResponse, AiAssistantError> {
        let body = self.build_body(config, system, messages, tools);
        let raw = self.post(config, body).await?;
        Self::parse_response(raw)
    }

    // ── Private helpers ────────────────────────────────────────────────────

    /// Build the JSON request body.
    fn build_body(
        &self,
        config: &Config,
        system: &str,
        messages: &[serde_json::Value],
        tools: &[Tool],
    ) -> serde_json::Value {
        let mut body = json!({
            "model":      config.claude_model,
            "max_tokens": 4096,
            "system":     system,
            "messages":   messages,
        });

        if !tools.is_empty() {
            body["tools"] = serde_json::to_value(tools).unwrap_or(json!([]));
        }

        body
    }

    /// Execute the POST request and surface structured HTTP errors.
    async fn post(
        &self,
        config: &Config,
        body: serde_json::Value,
    ) -> Result<serde_json::Value, AiAssistantError> {
        let url = format!("{}/v1/messages", config.anthropic_base_url);

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &config.anthropic_api_key)
            .header("content-type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .map_err(AiAssistantError::Http)?;

        let status = response.status();

        if status.is_success() {
            return response
                .json::<serde_json::Value>()
                .await
                .map_err(AiAssistantError::Http);
        }

        // Read body for diagnostics before consuming the response.
        let error_body = response
            .text()
            .await
            .unwrap_or_else(|_| "(unreadable body)".to_string());

        Err(map_http_error(status.as_u16(), &error_body))
    }

    /// Parse the raw Anthropic API JSON into a [`ClaudeResponse`].
    fn parse_response(json: serde_json::Value) -> Result<ClaudeResponse, AiAssistantError> {
        let model = json
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let input_tokens = json
            .pointer("/usage/input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        let output_tokens = json
            .pointer("/usage/output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        let content_arr = json
            .get("content")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let mut text_parts: Vec<String> = Vec::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        for block in &content_arr {
            match block.get("type").and_then(|t| t.as_str()) {
                Some("text") => {
                    if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                        text_parts.push(t.to_string());
                    }
                }
                Some("tool_use") => {
                    let id = block
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let name = block
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let input = block
                        .get("input")
                        .cloned()
                        .unwrap_or(json!({}));

                    if id.is_empty() || name.is_empty() {
                        return Err(AiAssistantError::ClaudeApi(
                            "tool_use block missing id or name".to_string(),
                        ));
                    }

                    tool_calls.push(ToolCall { id, name, input });
                }
                _ => {} // ignore unknown block types
            }
        }

        Ok(ClaudeResponse {
            text: text_parts.join("\n"),
            tool_calls,
            model,
            input_tokens,
            output_tokens,
        })
    }
}

// ── HTTP error mapping ────────────────────────────────────────────────────────

/// Maximum number of bytes from an HTTP error body included in error messages.
/// Prevents large or potentially sensitive server responses from propagating
/// verbatim through error chains and log sinks.
const MAX_ERROR_BODY_LEN: usize = 200;

fn map_http_error(status: u16, body: &str) -> AiAssistantError {
    // Truncate raw body to avoid leaking large or sensitive API error payloads.
    // Use char-based truncation to avoid panicking at a multi-byte UTF-8 boundary.
    let safe_body = if body.chars().count() > MAX_ERROR_BODY_LEN {
        let truncated: String = body.chars().take(MAX_ERROR_BODY_LEN).collect();
        format!("{truncated}…[truncated]")
    } else {
        body.to_string()
    };

    match status {
        401 => AiAssistantError::ClaudeApi(
            "Unauthorized: check ANTHROPIC_API_KEY".to_string(),
        ),
        429 => AiAssistantError::ClaudeApi("Rate limited by Anthropic API".to_string()),
        s if s >= 500 => AiAssistantError::ClaudeApi(format!(
            "Anthropic server error {s}: {safe_body}"
        )),
        s => AiAssistantError::ClaudeApi(format!("HTTP {s}: {safe_body}")),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_text_response() {
        let json = serde_json::json!({
            "model": "claude-opus-4-6",
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let resp = ClaudeClient::parse_response(json).unwrap();
        assert_eq!(resp.text, "Hello!");
        assert!(resp.tool_calls.is_empty());
        assert_eq!(resp.input_tokens, 10);
        assert_eq!(resp.output_tokens, 5);
    }

    #[test]
    fn parse_tool_use_response() {
        let json = serde_json::json!({
            "model": "claude-opus-4-6",
            "content": [{
                "type": "tool_use",
                "id": "call_01",
                "name": "web_search",
                "input": {"query": "rust programming"}
            }],
            "usage": {"input_tokens": 20, "output_tokens": 15}
        });
        let resp = ClaudeClient::parse_response(json).unwrap();
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "web_search");
        assert_eq!(resp.tool_calls[0].id, "call_01");
    }

    #[test]
    fn map_401() {
        let err = map_http_error(401, "");
        assert!(err.to_string().contains("Unauthorized"));
    }

    #[test]
    fn map_429() {
        let err = map_http_error(429, "");
        assert!(err.to_string().contains("Rate limited"));
    }

    #[test]
    fn map_503() {
        let err = map_http_error(503, "overloaded");
        assert!(err.to_string().contains("server error"));
    }
}
