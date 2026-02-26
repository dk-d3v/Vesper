//! Tests for [`ai_assistant::mcp_tools`]

use ai_assistant::mcp_tools::McpToolManager;
use std::io::Write;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Write content to a temp file and return its path.
fn tempfile_with_content(content: &str) -> String {
    let path = format!(
        "target/test_mcp_tools_{}.json",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
    );
    std::fs::create_dir_all("target").ok();
    let mut f = std::fs::File::create(&path).expect("Should create temp file");
    f.write_all(content.as_bytes()).expect("Should write content");
    path
}

fn drop_temp(path: &str) {
    let _ = std::fs::remove_file(path);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Test 1: McpToolManager::load("nonexistent.json") returns empty manager (no error).
#[test]
fn test_load_nonexistent_file_returns_empty_manager() {
    let result = McpToolManager::load("/definitely/does/not/exist/mcp.json");
    assert!(result.is_ok(), "load() on missing file should not error");

    let mgr = result.unwrap();
    assert!(!mgr.has_tools(), "Missing file should yield no tools");
    assert!(mgr.get_tools().is_empty(), "Missing file should yield empty tool list");
}

/// Test 2: McpToolManager with empty mcpServers has no tools.
#[test]
fn test_load_empty_mcp_servers_has_no_tools() {
    let path = tempfile_with_content(r#"{"mcpServers":{}}"#);
    let result = McpToolManager::load(&path);
    assert!(result.is_ok(), "Parsing empty mcpServers should not error");

    let mgr = result.unwrap();
    assert!(!mgr.has_tools(), "Empty mcpServers should yield no tools");
    drop_temp(&path);
}

/// Test 3: has_tools() returns false for empty manager.
#[test]
fn test_has_tools_false_for_empty_manager() {
    let mgr = McpToolManager::load("/nonexistent.json").unwrap();
    assert!(!mgr.has_tools(), "Empty manager should return false for has_tools()");
}

/// Test 4: McpToolManager::load("mcp_servers.json") succeeds.
///
/// This loads the real project-level mcp_servers.json.
#[test]
fn test_load_real_mcp_servers_json_succeeds() {
    // The real file is in the project root; CI also has it from the repo.
    let result = McpToolManager::load("mcp_servers.json");
    assert!(result.is_ok(), "load('mcp_servers.json') should succeed, got: {:?}", result.err());
}

/// Test 5: Invalid JSON in file returns error.
#[test]
fn test_load_invalid_json_returns_error() {
    let path = tempfile_with_content("{not valid json at all!!!");
    let result = McpToolManager::load(&path);
    assert!(result.is_err(), "Malformed JSON should return an error");
    drop_temp(&path);
}

/// Extra: A valid server config produces tools with correct naming.
#[test]
fn test_valid_server_produces_named_tool() {
    let content = r#"{
        "mcpServers": {
            "my-test-server": {
                "command": "node",
                "args": ["server.js"],
                "env": {}
            }
        }
    }"#;
    let path = tempfile_with_content(content);
    let mgr = McpToolManager::load(&path).unwrap();

    assert!(mgr.has_tools(), "Server config should produce tools");
    let tools = mgr.get_tools();
    assert_eq!(tools.len(), 1, "One server should produce one tool");
    assert_eq!(
        tools[0].name, "my-test-server/run",
        "Tool name should be 'server_name/run'"
    );
    assert!(!tools[0].description.is_empty(), "Tool description should not be empty");
    drop_temp(&path);
}

/// Extra: Multiple servers produce multiple tools.
#[test]
fn test_multiple_servers_produce_multiple_tools() {
    let content = r#"{
        "mcpServers": {
            "server-alpha": {"command": "node", "args": []},
            "server-beta":  {"command": "python", "args": []}
        }
    }"#;
    let path = tempfile_with_content(content);
    let mgr = McpToolManager::load(&path).unwrap();

    assert_eq!(mgr.get_tools().len(), 2, "Two servers should produce two tools");
    drop_temp(&path);
}
