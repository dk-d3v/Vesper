# Security Audit Report — AI Assistant

**Date:** 2026-02-26  
**Auditor:** Security Review Mode (automated static analysis)  
**Codebase Revision:** Phase 6  
**Scope:** All files under `src/`, `mcp_servers.json`, `.env.example`, `.gitignore`

---

## Executive Summary

The AI Assistant codebase demonstrates **strong baseline security hygiene**. No hardcoded secrets were found. Credentials flow exclusively from environment variables into a typed `Config` struct and are never logged or serialised. Input length enforcement, session bounding, and a coherence-based critical halt are all in place.

Three **Medium** issues were identified and fixed during this audit:

1. **HTTP error body forwarding** — raw API error responses were propagated verbatim into error messages without size limits, risking exposure of large or sensitive payloads.
2. **Plaintext HTTP without warning** — configuring `ANTHROPIC_BASE_URL=http://…` would transmit the API key in cleartext with no diagnostic.
3. **Unbounded MCP subprocess execution** — any executable could be specified in `mcp_servers.json` without a command allowlist, and there was no read timeout to prevent a hung child from blocking the pipeline.

All three were **fixed in place**. No Critical or High findings were identified.

---

## Findings

### FINDING-001: HTTP Error Body Forwarded Verbatim
**Severity:** Medium  
**File:** [`src/claude_api.rs:202`](../src/claude_api.rs:202)  
**Description:** `map_http_error()` previously passed the raw API response body directly into error messages for 4xx/5xx codes. The body could be arbitrarily large and might contain upstream diagnostic data.  
**Impact:** Potentially large error payloads propagating through logs or user-facing error channels; minor information leakage from upstream API error context.  
**Remediation:** Added `MAX_ERROR_BODY_LEN = 200` constant; body is now truncated to 200 characters before inclusion in any error variant.  
**Status:** ✅ Fixed — [`src/claude_api.rs:202`](../src/claude_api.rs:202)

---

### FINDING-002: Plaintext HTTP Endpoint Accepted Without Warning
**Severity:** Medium  
**File:** [`src/config.rs:36`](../src/config.rs:36)  
**Description:** `load_config()` accepted `http://` URLs for `ANTHROPIC_BASE_URL` without any diagnostic. The API key is transmitted in the `x-api-key` header; on a plaintext HTTP connection it would travel in cleartext.  
**Impact:** Accidental misconfiguration in a production deployment would expose the API key over the network.  
**Remediation:** Added an `eprintln!` warning when `base_url.starts_with("http://")` is detected, clearly indicating the key will be transmitted without TLS. `https://` default remains unchanged.  
**Status:** ✅ Fixed — [`src/config.rs:43`](../src/config.rs:43)

---

### FINDING-003: Unrestricted MCP Subprocess Command
**Severity:** Medium  
**File:** [`src/mcp_tools.rs`](../src/mcp_tools.rs)  
**Description:** `spawn_rpc()` called `Command::new(&config.command)` without validating `command` against any allowlist. A malicious or tampered `mcp_servers.json` could specify an arbitrary executable (e.g. `rm`, `curl`, `bash`).  
**Impact:** Local privilege of the running process could be abused to execute unintended binaries via a malicious configuration file.  
**Remediation:** Added `ALLOWED_COMMANDS` constant listing `["node", "npx", "python", "python3", "uvx", "deno", "bun"]`. `spawn_rpc()` now normalises the command to its basename and rejects any command not in the list with a descriptive error.  
**Status:** ✅ Fixed — [`src/mcp_tools.rs:57`](../src/mcp_tools.rs:57)

---

### FINDING-004: MCP Subprocess Has No Read Timeout
**Severity:** Medium  
**File:** [`src/mcp_tools.rs:208`](../src/mcp_tools.rs:208) (pre-fix)  
**Description:** `BufReader::read_line()` on the child's stdout was blocking and unbounded. A hung or misbehaving MCP server would block the Tokio executor thread indefinitely.  
**Impact:** Denial-of-service against the pipeline if a configured MCP server stalls.  
**Remediation:** The blocking read is now performed on a dedicated OS thread. The calling thread waits at most `RPC_READ_TIMEOUT_SECS` (10 seconds) via `mpsc::channel::recv_timeout`. On timeout, the child process is killed.  
**Status:** ✅ Fixed — [`src/mcp_tools.rs:243`](../src/mcp_tools.rs:243)

---

### FINDING-005: Audit Trail is In-Memory Only (Accepted Risk)
**Severity:** Low  
**File:** [`src/audit.rs:79`](../src/audit.rs:79)  
**Description:** `AuditTrail` stores its entries in a `Vec<AuditEntry>` in process memory. If the process exits (crash, SIGKILL, restart), the entire audit log is lost.  
**Impact:** Forensic trail is not durable — a post-incident investigation would find no audit data if the process was not gracefully shut down.  
**Remediation (recommended):** Persist audit entries to an append-only file or a write-ahead log (e.g. SQLite journal) on every `record()` call. The `AuditTrail` API is compatible with this change.  
**Status:** ⚠️ Accepted Risk — documented limitation. No code exists to modify after the fact (the `Vec` is append-only in practice), so integrity within a session is sound. Durability must be addressed in a future storage layer.

---

### FINDING-006: In-Memory Structures Grow Unbounded Across Long Sessions
**Severity:** Low  
**File:** [`src/audit.rs:79`](../src/audit.rs:79), [`src/learning.rs:48`](../src/learning.rs:48), [`src/memory.rs:66`](../src/memory.rs:66)  
**Description:**
- `AuditTrail.entries` — grows by one per conversation turn, forever.
- `LearningEngine.records` — one `TrajectoryRecord` added per turn, never pruned.
- `MemoryStore.causal_cache` — one entry per turn, never pruned.
- `MemoryStore.quality_map` — entries are removed on `auto_consolidate()` but only for low-quality episodes; the map can still grow large in active sessions.

`Session.turns` is correctly bounded at 50 entries.  
**Impact:** In very long-running processes the memory footprint grows monotonically, eventually causing OOM on memory-constrained hosts.  
**Remediation (recommended):**
- `AuditTrail`: export and flush entries older than N turns to a file, or cap in-memory retention.
- `LearningEngine.records`: apply a sliding window (e.g. retain last 1 000 records).
- `MemoryStore.causal_cache`: prune entries beyond a configurable ceiling.  
**Status:** ⚠️ Accepted Risk for current use-case (interactive REPL with bounded sessions). Recommend fixing before deployment in a long-running server scenario.

---

### FINDING-007: No Hardcoded Secrets — PASS
**Severity:** Info  
**File:** All `src/*.rs`  
**Description:** A full regex search for `sk-ant`, `api.key`, `Bearer`, `Authorization`, `password`, `secret`, and `hardcoded` found **no hardcoded credential values** in any source file. `ANTHROPIC_API_KEY` is read exclusively from environment variables via `std::env::var()`.  
**Status:** ✅ Pass

---

### FINDING-008: `.env` Correctly Excluded from Git — PASS
**Severity:** Info  
**File:** [`.gitignore:2`](../.gitignore:2)  
**Description:** `.env` is present on line 2 of `.gitignore`. `.env.example` contains only empty placeholders (no real values).  
**Status:** ✅ Pass

---

### FINDING-009: API Key Sent Only in Header — PASS
**Severity:** Info  
**File:** [`src/claude_api.rs:102`](../src/claude_api.rs:102)  
**Description:** The API key is set exclusively in the `x-api-key` HTTP header. It does not appear in the URL, query parameters, or the JSON body of any request.  
**Status:** ✅ Pass

---

### FINDING-010: TLS Validation via reqwest Default — PASS
**Severity:** Info  
**File:** [`src/claude_api.rs:25`](../src/claude_api.rs:25), [`Cargo.toml:27`](../Cargo.toml:27)  
**Description:** `reqwest::Client::new()` uses the platform's native TLS backend (via `reqwest`'s default feature set). Certificate validation is enabled by default and not disabled anywhere in the codebase (`danger_accept_invalid_certs` is absent).  
**Status:** ✅ Pass

---

### FINDING-011: Input Validation Enforced at Pipeline Entry — PASS
**Severity:** Info  
**File:** [`src/pipeline.rs:127`](../src/pipeline.rs:127)  
**Description:** `step1_receive()` rejects empty strings and inputs exceeding `MAX_INPUT_LENGTH` (32 768 chars, defined in [`src/config.rs:74`](../src/config.rs:74)) before any processing occurs.  
**Status:** ✅ Pass

---

### FINDING-012: Session Turn Buffer Bounded — PASS
**Severity:** Info  
**File:** [`src/types.rs:167`](../src/types.rs:167)  
**Description:** `Session::add_turn()` caps `self.turns` at 50 entries by calling `self.turns.remove(0)` when the limit is exceeded.  
**Status:** ✅ Pass

---

### FINDING-013: MCP Subprocess Not Shell-Expanded — PASS
**Severity:** Info  
**File:** [`src/mcp_tools.rs`](../src/mcp_tools.rs)  
**Description:** `Command::new(&config.command).args(&config.args)` passes arguments as a discrete `Vec<String>`, not through a shell. Shell metacharacters (`;`, `|`, `&&`, `$()`) in argument values are inert — they are passed literally to `execv()` / `CreateProcess`.  
**Status:** ✅ Pass — shell injection is not possible by construction.

---

### FINDING-014: Graph Context Uses Label-Based Filter, Not Raw Cypher — PASS
**Severity:** Info  
**File:** [`src/graph_context.rs:137`](../src/graph_context.rs:137)  
**Description:** Entity queries call `self.graph.get_nodes_by_label("Document")` and then perform in-process Rust filtering. No user-supplied string is interpolated into a Cypher template, eliminating graph injection risk.  
**Status:** ✅ Pass

---

### FINDING-015: Error Messages Don't Expose API Key — PASS
**Severity:** Info  
**File:** [`src/claude_api.rs:202`](../src/claude_api.rs:202), [`src/error.rs`](../src/error.rs)  
**Description:** The 401 handler emits `"Unauthorized: check ANTHROPIC_API_KEY"` — the name of the variable, not its value. No `AiAssistantError` variant formats `config.anthropic_api_key` into its display string.  
**Status:** ✅ Pass

---

### FINDING-016: Audit Chain Tamper Detection Functional — PASS
**Severity:** Info  
**File:** [`src/audit.rs:135`](../src/audit.rs:135)  
**Description:** `verify_chain()` validates the genesis link and every inter-entry `prev_hash` link. The test at [`src/audit.rs:236`](../src/audit.rs:236) confirms that mutating a `prev_hash` field causes verification to fail. The chain is append-only within the lifetime of the process.  
**Status:** ✅ Pass

---

## OWASP Top 10 Mapping

| OWASP Category | Status |
|---|---|
| A01 – Broken Access Control | N/A (single-user REPL, no access control layer) |
| A02 – Cryptographic Failures | ✅ TLS enforced (warning on http://), SHAKE-256 for audit chain |
| A03 – Injection | ✅ No shell injection; no raw Cypher; no SQL |
| A04 – Insecure Design | ✅ Audit trail, coherence halt, MAX_INPUT_LENGTH all by design |
| A05 – Security Misconfiguration | ✅ Fixed: http:// warning added; command allowlist added |
| A06 – Vulnerable & Outdated Components | ℹ️ `reqwest 0.11` (not audited; recommend `cargo audit`) |
| A07 – Authentication Failures | ✅ API key validated at startup; empty key rejected |
| A08 – Software & Data Integrity | ✅ SHAKE-256 hash chain; no deserialization of untrusted formats |
| A09 – Logging & Monitoring Failures | ⚠️ Audit trail in-memory only (FINDING-005) |
| A10 – Server-Side Request Forgery | N/A (no user-controlled URL fetching) |

---

## Recommendations (Non-Blocking)

1. **Run `cargo audit`** to check all dependencies (including `ruvector` sub-crates) against the RustSec advisory database.
2. **Persist the audit trail** to an append-only file or SQLite WAL before any production deployment.
3. **Add sliding-window caps** to `LearningEngine.records` and `MemoryStore.causal_cache` for long-running server deployments.
4. **Rate-limit pipeline calls** at the binary entry point (`src/main.rs`) if ever exposed over a network interface.
5. **Pin `reqwest` to 0.12+** which uses `rustls` by default and eliminates dependency on the OS TLS library.

---

## Files Modified During This Audit

| File | Change |
|---|---|
| [`src/claude_api.rs`](../src/claude_api.rs) | Truncate HTTP error body to 200 chars (FINDING-001) |
| [`src/config.rs`](../src/config.rs) | Warn on `http://` base URL (FINDING-002) |
| [`src/mcp_tools.rs`](../src/mcp_tools.rs) | Add `ALLOWED_COMMANDS` allowlist + `RPC_READ_TIMEOUT_SECS` timeout (FINDING-003, FINDING-004) |
