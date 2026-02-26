//! Claude API HTTP client using reqwest.
//!
//! Claude generates text only — all business logic remains in Rust.
//! Handles multi-turn tool-use conversations with proper error mapping
//! for 401, 429, and 5xx responses.
//!
//! # SSE Streaming
//! `send_message_stream()` enables token-by-token delivery via a callback.
//! It parses Anthropic's Server-Sent Events protocol and invokes the
//! callback for every `text_delta` event.  Post-stream steps still receive
//! the complete assembled text.

use futures_util::StreamExt;

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

        let body = self.build_body(config, system, &messages, tools, false);
        let raw = self.post(config, body).await?;
        Self::parse_response(raw)
    }

    /// Stream a single-turn user message to Claude token-by-token.
    ///
    /// `on_token` is called for every text delta received from the SSE stream.
    /// The method returns a complete [`ClaudeResponse`] after the stream ends.
    ///
    /// # Protocol
    /// Sets `"stream": true` in the request body and parses Anthropic's SSE
    /// format.  Events of type `content_block_delta` with `delta.type =
    /// "text_delta"` produce token callbacks; all others are silently ignored.
    pub async fn send_message_stream<F>(
        &self,
        config: &Config,
        system: &str,
        user_content: &str,
        tools: &[Tool],
        on_token: F,
    ) -> Result<ClaudeResponse, AiAssistantError>
    where
        F: Fn(&str),
    {
        let messages = vec![json!({
            "role": "user",
            "content": user_content
        })];

        let body = self.build_body(config, system, &messages, tools, true);
        self.post_stream(config, body, on_token).await
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
        let body = self.build_body(config, system, messages, tools, false);
        let raw = self.post(config, body).await?;
        Self::parse_response(raw)
    }

    // ── Private helpers ────────────────────────────────────────────────────

    /// Build the JSON request body.
    ///
    /// `stream` enables SSE streaming mode when `true`.
    ///
    /// When `config.thinking_budget_tokens > 0`, extended thinking is enabled:
    /// - `temperature` and `top_k` are omitted (incompatible with thinking API)
    /// - `max_tokens` is set to `budget_tokens + 4096` so it exceeds the budget
    fn build_body(
        &self,
        config: &Config,
        system: &str,
        messages: &[serde_json::Value],
        tools: &[Tool],
        stream: bool,
    ) -> serde_json::Value {
        let thinking_enabled = config.thinking_budget_tokens > 0;

        // When thinking is enabled, max_tokens must exceed budget_tokens.
        let max_tokens: u32 = if thinking_enabled {
            config.thinking_budget_tokens + 4096
        } else {
            4096
        };

        let mut body = json!({
            "model":      config.claude_model,
            "max_tokens": max_tokens,
            "system":     system,
            "messages":   messages,
        });

        if stream {
            body["stream"] = json!(true);
        }

        if !tools.is_empty() {
            body["tools"] = serde_json::to_value(tools).unwrap_or(json!([]));
        }

        // Extended thinking — temperature/top_k must NOT be set alongside thinking.
        if thinking_enabled {
            body["thinking"] = json!({
                "type": "enabled",
                "budget_tokens": config.thinking_budget_tokens
            });
        }

        body
    }

    /// Execute a streaming POST request.
    ///
    /// Parses Anthropic SSE events and calls `on_token` for each
    /// `text_delta`.  Accumulates the full response text and metadata
    /// (model, usage) from `message_start` and `message_delta` events.
    async fn post_stream<F>(
        &self,
        config: &Config,
        body: serde_json::Value,
        on_token: F,
    ) -> Result<ClaudeResponse, AiAssistantError>
    where
        F: Fn(&str),
    {
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
        if !status.is_success() {
            let error_body = response
                .text()
                .await
                .unwrap_or_else(|_| "(unreadable body)".to_string());
            return Err(map_http_error(status.as_u16(), &error_body));
        }

        // Accumulator state
        let mut full_text = String::new();
        let mut model = String::from("unknown");
        let mut input_tokens: u32 = 0;
        let mut output_tokens: u32 = 0;

        // Buffer for partial SSE lines across chunk boundaries
        let mut line_buf = String::new();
        let mut byte_stream = response.bytes_stream();

        while let Some(chunk) = byte_stream.next().await {
            let bytes = chunk.map_err(AiAssistantError::Http)?;
            let text = String::from_utf8_lossy(&bytes);

            // Append bytes to line buffer and process complete lines
            line_buf.push_str(&text);

            // Process all complete lines (terminated by '\n')
            while let Some(pos) = line_buf.find('\n') {
                let line = line_buf[..pos].trim_end_matches('\r').to_string();
                line_buf = line_buf[pos + 1..].to_string();

                process_sse_line(
                    &line,
                    &on_token,
                    &mut full_text,
                    &mut model,
                    &mut input_tokens,
                    &mut output_tokens,
                );
            }
        }

        // Drain any remaining partial line
        if !line_buf.trim().is_empty() {
            process_sse_line(
                line_buf.trim(),
                &on_token,
                &mut full_text,
                &mut model,
                &mut input_tokens,
                &mut output_tokens,
            );
        }

        Ok(ClaudeResponse {
            text: full_text,
            tool_calls: vec![], // Tool use not supported in streaming mode
            model,
            input_tokens,
            output_tokens,
        })
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

// ── SSE event parsing ─────────────────────────────────────────────────────────

/// Process a single SSE line, updating accumulators and calling `on_token`.
///
/// Anthropic SSE events of interest:
/// - `message_start` → carries model name and input token count
/// - `content_block_delta` with `delta.type = "text_delta"` → text token
/// - `message_delta` → carries final output token count
fn process_sse_line<F>(
    line: &str,
    on_token: &F,
    full_text: &mut String,
    model: &mut String,
    input_tokens: &mut u32,
    output_tokens: &mut u32,
) where
    F: Fn(&str),
{
    // SSE data lines start with "data: "
    let Some(data) = line.strip_prefix("data: ") else {
        return; // event:, comment, or empty line — skip
    };

    // "[DONE]" sentinel closes the stream
    if data.trim() == "[DONE]" {
        return;
    }

    let Ok(event) = serde_json::from_str::<serde_json::Value>(data) else {
        return; // malformed JSON — skip silently
    };

    match event.get("type").and_then(|t| t.as_str()) {
        // Capture model name and input token count from stream start
        Some("message_start") => {
            if let Some(m) = event.pointer("/message/model").and_then(|v| v.as_str()) {
                *model = m.to_string();
            }
            if let Some(tok) = event
                .pointer("/message/usage/input_tokens")
                .and_then(|v| v.as_u64())
            {
                *input_tokens = tok as u32;
            }
        }

        // Content block start — emit prefix markers for thinking/text blocks
        Some("content_block_start") => {
            let block_type = event
                .pointer("/content_block/type")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            if block_type == "thinking" {
                on_token("\n💭 Thinking: ");
            }
            // text blocks: the "Assistant: " label is already printed in main.rs
        }

        // Content block delta — handles text_delta, thinking_delta, signature_delta
        Some("content_block_delta") => {
            let delta_type = event
                .pointer("/delta/type")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            match delta_type {
                "text_delta" => {
                    if let Some(token) = event.pointer("/delta/text").and_then(|v| v.as_str()) {
                        on_token(token);
                        full_text.push_str(token);
                    }
                }
                "thinking_delta" => {
                    if let Some(thinking) = event
                        .pointer("/delta/thinking")
                        .and_then(|v| v.as_str())
                    {
                        on_token(thinking);
                    }
                }
                "signature_delta" => {
                    // Verification signature — not displayed to the user
                }
                _ => {}
            }
        }

        // Final output token count
        Some("message_delta") => {
            if let Some(tok) = event
                .pointer("/usage/output_tokens")
                .and_then(|v| v.as_u64())
            {
                *output_tokens = tok as u32;
            }
        }

        _ => {} // ping, error, etc. — ignore
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

    #[test]
    fn sse_text_delta_parsed() {
        use std::cell::RefCell;

        let line = r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        let mut text = String::new();
        let mut model = String::new();
        let mut input = 0u32;
        let mut output = 0u32;
        let tokens: RefCell<Vec<String>> = RefCell::new(Vec::new());

        process_sse_line(
            line,
            &|t: &str| tokens.borrow_mut().push(t.to_string()),
            &mut text,
            &mut model,
            &mut input,
            &mut output,
        );

        assert_eq!(*tokens.borrow(), vec!["Hello"]);
        assert_eq!(text, "Hello");
    }

    #[test]
    fn sse_message_start_captures_model() {
        let line = r#"data: {"type":"message_start","message":{"model":"claude-opus-4-6","usage":{"input_tokens":42}}}"#;
        let mut text = String::new();
        let mut model = String::new();
        let mut input = 0u32;
        let mut output = 0u32;

        process_sse_line(
            line,
            &|_| {},
            &mut text,
            &mut model,
            &mut input,
            &mut output,
        );

        assert_eq!(model, "claude-opus-4-6");
        assert_eq!(input, 42);
    }

    #[test]
    fn sse_non_data_lines_ignored() {
        use std::cell::Cell;

        let lines = ["event: content_block_delta", ": ping", ""];
        let mut text = String::new();
        let mut model = String::new();
        let mut input = 0u32;
        let mut output = 0u32;
        let called = Cell::new(false);

        for line in &lines {
            process_sse_line(
                line,
                &|_| called.set(true),
                &mut text,
                &mut model,
                &mut input,
                &mut output,
            );
        }

        assert!(!called.get());
        assert!(text.is_empty());
    }

    #[test]
    fn sse_thinking_delta_calls_on_token() {
        use std::cell::RefCell;

        let line = r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"I am reasoning..."}}"#;
        let mut text = String::new();
        let mut model = String::new();
        let mut input = 0u32;
        let mut output = 0u32;
        let tokens: RefCell<Vec<String>> = RefCell::new(Vec::new());

        process_sse_line(
            line,
            &|t: &str| tokens.borrow_mut().push(t.to_string()),
            &mut text,
            &mut model,
            &mut input,
            &mut output,
        );

        assert_eq!(*tokens.borrow(), vec!["I am reasoning..."]);
        // Thinking tokens do NOT contribute to full_text
        assert!(text.is_empty(), "thinking_delta should not update full_text");
    }

    #[test]
    fn sse_content_block_start_thinking_emits_prefix() {
        use std::cell::RefCell;

        let line = r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#;
        let mut text = String::new();
        let mut model = String::new();
        let mut input = 0u32;
        let mut output = 0u32;
        let tokens: RefCell<Vec<String>> = RefCell::new(Vec::new());

        process_sse_line(
            line,
            &|t: &str| tokens.borrow_mut().push(t.to_string()),
            &mut text,
            &mut model,
            &mut input,
            &mut output,
        );

        let emitted = tokens.borrow();
        assert!(!emitted.is_empty(), "Should emit thinking prefix token");
        assert!(
            emitted[0].contains("💭"),
            "Thinking prefix should contain 💭, got: {:?}",
            emitted[0]
        );
    }

    #[test]
    fn sse_signature_delta_ignored() {
        use std::cell::Cell;

        let line = r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"abc123"}}"#;
        let mut text = String::new();
        let mut model = String::new();
        let mut input = 0u32;
        let mut output = 0u32;
        let called = Cell::new(false);

        process_sse_line(
            line,
            &|_| called.set(true),
            &mut text,
            &mut model,
            &mut input,
            &mut output,
        );

        assert!(!called.get(), "signature_delta should not call on_token");
        assert!(text.is_empty());
    }

    #[test]
    fn parse_response_skips_thinking_blocks() {
        let json = serde_json::json!({
            "model": "claude-opus-4-6",
            "content": [
                {"type": "thinking", "thinking": "Internal reasoning here"},
                {"type": "text", "text": "Final answer"}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let resp = ClaudeClient::parse_response(json).unwrap();
        assert_eq!(resp.text, "Final answer", "Only text blocks should be in response text");
        assert!(resp.tool_calls.is_empty());
    }
}
