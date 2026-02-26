//! MCP (Model Context Protocol) tools manager.
//!
//! Loads server definitions from `mcp_servers.json`, exposes their tools to
//! Claude as [`Tool`] descriptors, and handles tool execution via subprocess
//! JSON-RPC calls. Claude declares intent; **Rust executes everything**.
//!
//! Returns an empty manager (no error) when `mcp_servers.json` is absent or
//! contains no servers — MCP tooling is entirely optional.
//!
//! # Security
//! - Subprocess commands are launched with [`Command::new`], **not** via a
//!   shell. Arguments are passed as discrete values, not interpolated strings,
//!   so shell injection is not possible.
//! - The command executable is checked against [`ALLOWED_COMMANDS`] before
//!   spawning, preventing a malicious `mcp_servers.json` from running
//!   arbitrary binaries.
//! - A read timeout ([`RPC_READ_TIMEOUT_SECS`]) prevents a hung subprocess
//!   from blocking the pipeline indefinitely.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::Duration;

use serde_json::json;

use crate::{
    error::AiAssistantError,
    types::{Tool, ToolCall, ToolResult},
};

// ── Security constants ────────────────────────────────────────────────────────

/// Executables permitted to be launched as MCP server processes.
///
/// A malicious or misconfigured `mcp_servers.json` could otherwise point the
/// `command` field at any binary on the host. This allowlist restricts spawning
/// to known-safe runtimes. Extend it as needed for your deployment.
const ALLOWED_COMMANDS: &[&str] = &[
    "node",
    "npx",
    "python",
    "python3",
    "uvx",
    "deno",
    "bun",
];

/// Seconds to wait for a line of output from a spawned MCP process before
/// treating the call as failed and killing the child. Prevents a hung
/// subprocess from blocking the pipeline indefinitely.
const RPC_READ_TIMEOUT_SECS: u64 = 10;

// ── MCP server configuration ──────────────────────────────────────────────────

/// Configuration for a single MCP server entry inside `mcp_servers.json`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct McpServerConfig {
    /// Executable to invoke (e.g. `"npx"`, `"node"`, `"python"`).
    pub command: String,
    /// Arguments passed after `command`.
    #[serde(default)]
    pub args: Vec<String>,
    /// Extra environment variables for the process.
    #[serde(default)]
    pub env: HashMap<String, String>,
}

/// Top-level shape of `mcp_servers.json`.
#[derive(Debug, serde::Deserialize)]
struct McpServersFile {
    #[serde(rename = "mcpServers", default)]
    mcp_servers: HashMap<String, McpServerConfig>,
}

// ── Manager ───────────────────────────────────────────────────────────────────

/// Manages MCP tools loaded from a `mcp_servers.json` configuration file.
pub struct McpToolManager {
    /// Map from server name → config.
    servers: HashMap<String, McpServerConfig>,
    /// Cached tool definitions derived from server configs.
    tools: Vec<Tool>,
}

impl McpToolManager {
    // ── Construction ──────────────────────────────────────────────────────

    /// Load MCP configuration from `path`.
    ///
    /// Returns an **empty manager** (no error) when:
    /// - The file does not exist.
    /// - The `mcpServers` object is empty.
    ///
    /// Returns [`AiAssistantError::McpTools`] only on malformed JSON.
    pub fn load(path: &str) -> Result<Self, AiAssistantError> {
        if !std::path::Path::new(path).exists() {
            tracing::info!("mcp_servers.json not found at '{}' — MCP disabled.", path);
            return Ok(Self::empty());
        }

        let raw = std::fs::read_to_string(path)
            .map_err(|e| AiAssistantError::McpTools(format!("read {path}: {e}")))?;

        let file: McpServersFile = serde_json::from_str(&raw)
            .map_err(|e| AiAssistantError::McpTools(format!("parse {path}: {e}")))?;

        if file.mcp_servers.is_empty() {
            tracing::info!("mcp_servers.json has no servers — MCP disabled.");
            return Ok(Self::empty());
        }

        let tools = build_tools(&file.mcp_servers);
        tracing::info!("Loaded {} MCP server(s), {} tool(s).", file.mcp_servers.len(), tools.len());

        Ok(Self {
            servers: file.mcp_servers,
            tools,
        })
    }

    fn empty() -> Self {
        Self {
            servers: HashMap::new(),
            tools: Vec::new(),
        }
    }

    // ── Queries ───────────────────────────────────────────────────────────

    /// All tool definitions ready to pass to Claude.
    pub fn get_tools(&self) -> &[Tool] {
        &self.tools
    }

    /// `true` if at least one MCP server is configured.
    pub fn has_tools(&self) -> bool {
        !self.tools.is_empty()
    }

    // ── Execution ─────────────────────────────────────────────────────────

    /// Execute a tool call declared by Claude.
    ///
    /// Spawns the associated MCP server process, sends a JSON-RPC request on
    /// stdin, reads the response from stdout, and returns a [`ToolResult`].
    ///
    /// **Failures are returned as a descriptive `ToolResult`, not propagated
    /// as errors**, so Claude can observe the failure and decide how to proceed.
    pub async fn execute_tool(&self, call: &ToolCall) -> Result<ToolResult, AiAssistantError> {
        let server_name = self.find_server_for_tool(&call.name);

        let Some(server_name) = server_name else {
            return Ok(ToolResult {
                tool_use_id: call.id.clone(),
                content: format!("Error: no MCP server registered for tool '{}'", call.name),
            });
        };

        let config = &self.servers[server_name];

        tracing::debug!(
            "Executing MCP tool '{}' via server '{}' ({})",
            call.name, server_name, config.command
        );

        let content = match spawn_rpc(config, &call.name, &call.input) {
            Ok(result) => result,
            Err(e) => format!("Error executing tool '{}': {}", call.name, e),
        };

        Ok(ToolResult {
            tool_use_id: call.id.clone(),
            content,
        })
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /// Find which server name owns the given tool name.
    fn find_server_for_tool<'a>(&'a self, tool_name: &'a str) -> Option<&'a str> {
        // Try to find a server whose prefix matches the tool name.
        let prefix_match = tool_name.split('/').next().and_then(|prefix| {
            if self.servers.contains_key(prefix) {
                Some(prefix)
            } else {
                None
            }
        });

        if prefix_match.is_some() {
            return prefix_match;
        }

        // If the full tool name matches a registered tool and there's only one server,
        // use that server.
        let tool_exists = self.tools.iter().any(|t| t.name == tool_name);
        if tool_exists && self.servers.len() == 1 {
            return self.servers.keys().next().map(String::as_str);
        }

        None
    }
}

// ── JSON-RPC subprocess call ──────────────────────────────────────────────────

/// Spawn the MCP server process, write a `tools/call` JSON-RPC request, and
/// read back the first response line from stdout.
///
/// # Security
/// - `config.command` is validated against [`ALLOWED_COMMANDS`] before spawning.
/// - Arguments come from `config.args` (a parsed JSON array), never from shell
///   expansion — there is no shell involved.
/// - A thread-based timeout of [`RPC_READ_TIMEOUT_SECS`] seconds kills the
///   child process if it does not respond in time.
fn spawn_rpc(
    config: &McpServerConfig,
    tool_name: &str,
    input: &serde_json::Value,
) -> Result<String, String> {
    // SECURITY: validate command against an allowlist before spawning.
    // Normalise to the final path component so both "node" and "/usr/bin/node" match.
    let cmd_base = std::path::Path::new(&config.command)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(config.command.as_str());

    if !ALLOWED_COMMANDS.contains(&cmd_base) {
        return Err(format!(
            "MCP command '{}' is not in the allowed-command list. \
             Add it to ALLOWED_COMMANDS in mcp_tools.rs after review.",
            config.command
        ));
    }

    let mut cmd = Command::new(&config.command);
    cmd.args(&config.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());

    for (k, v) in &config.env {
        cmd.env(k, v);
    }

    let mut child = cmd.spawn().map_err(|e| format!("spawn: {e}"))?;

    let request = json!({
        "jsonrpc": "2.0",
        "id":      1,
        "method":  "tools/call",
        "params": {
            "name":      tool_name,
            "arguments": input
        }
    });

    let request_str = serde_json::to_string(&request).map_err(|e| e.to_string())?;

    // Write request to stdin.
    if let Some(stdin) = child.stdin.as_mut() {
        stdin
            .write_all(request_str.as_bytes())
            .and_then(|_| stdin.write_all(b"\n"))
            .map_err(|e| format!("stdin write: {e}"))?;
    }

    // Read first line of stdout with a timeout.
    // Spawn a background thread to perform the blocking read; the main thread
    // waits for RPC_READ_TIMEOUT_SECS before killing the child.
    let stdout = child.stdout.take().ok_or("no stdout")?;
    let (tx, rx) = std::sync::mpsc::channel::<Result<String, String>>();
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        let result = reader
            .read_line(&mut line)
            .map(|_| line)
            .map_err(|e| format!("stdout read: {e}"));
        let _ = tx.send(result);
    });

    let line = match rx.recv_timeout(Duration::from_secs(RPC_READ_TIMEOUT_SECS)) {
        Ok(Ok(l)) => l,
        Ok(Err(e)) => {
            let _ = child.kill();
            return Err(e);
        }
        Err(_timeout) => {
            let _ = child.kill();
            return Err(format!(
                "MCP server did not respond within {RPC_READ_TIMEOUT_SECS}s"
            ));
        }
    };

    let _ = child.wait();

    if line.trim().is_empty() {
        return Err("MCP server returned empty response".to_string());
    }

    // Parse and extract the result content.
    let resp: serde_json::Value =
        serde_json::from_str(line.trim()).map_err(|e| format!("parse rpc response: {e}"))?;

    if let Some(error) = resp.get("error") {
        return Err(format!("JSON-RPC error: {error}"));
    }

    let content = resp
        .pointer("/result/content/0/text")
        .and_then(|v| v.as_str())
        .or_else(|| resp.pointer("/result").and_then(|v| v.as_str()))
        .map(str::to_string)
        .unwrap_or_else(|| resp["result"].to_string());

    Ok(content)
}

// ── Tool-list builder ─────────────────────────────────────────────────────────

/// Build Claude-compatible [`Tool`] descriptors from server configs.
///
/// Each server contributes one generic tool named `"{server}/run"` with an
/// open JSON schema. Real implementations would query each server's
/// `tools/list` endpoint; this stub keeps the build dependency-free.
fn build_tools(servers: &HashMap<String, McpServerConfig>) -> Vec<Tool> {
    servers
        .keys()
        .map(|name| Tool {
            name: format!("{}/run", name),
            description: format!("Execute a command on the '{}' MCP server.", name),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "The action or query to perform."
                    }
                },
                "required": ["action"]
            }),
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_missing_file_returns_empty() {
        let mgr = McpToolManager::load("/does/not/exist.json").unwrap();
        assert!(!mgr.has_tools());
        assert!(mgr.get_tools().is_empty());
    }

    #[test]
    fn load_empty_servers_returns_empty() {
        let tmp = tempfile_with_content(r#"{"mcpServers":{}}"#);
        let mgr = McpToolManager::load(&tmp).unwrap();
        assert!(!mgr.has_tools());
        drop_temp(tmp);
    }

    #[test]
    fn load_malformed_json_returns_error() {
        let tmp = tempfile_with_content("{not valid json}");
        let result = McpToolManager::load(&tmp);
        assert!(result.is_err());
        drop_temp(tmp);
    }

    #[test]
    fn load_valid_server_builds_tools() {
        let content = r#"{
            "mcpServers": {
                "my-server": {
                    "command": "node",
                    "args": ["server.js"]
                }
            }
        }"#;
        let tmp = tempfile_with_content(content);
        let mgr = McpToolManager::load(&tmp).unwrap();
        assert!(mgr.has_tools());
        assert_eq!(mgr.get_tools()[0].name, "my-server/run");
        drop_temp(tmp);
    }

    // ── helpers ──────────────────────────────────────────────────────────

    fn tempfile_with_content(content: &str) -> String {
        use std::io::Write;
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let path = format!("target/test_mcp_{}_{}.json", std::process::id(), id);
        std::fs::create_dir_all("target").ok();
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    fn drop_temp(path: String) {
        let _ = std::fs::remove_file(path);
    }
}
