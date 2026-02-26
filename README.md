# AI Assistant 🤖

A Rust-based AI assistant powered by the [ruvector](https://github.com/ruvnet/ruvector) ecosystem and Claude API. Features semantic memory, graph-based RAG, coherence checking, verified responses, and continuous learning — with full audit trail.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [ONNX Embedding Model](#onnx-embedding-model)
- [MCP Tools (Optional)](#mcp-tools-optional)
- [Architecture Overview](#architecture-overview)
- [Security Notes](#security-notes)
- [Development](#development)
- [Project Structure](#project-structure)
- [License](#license)

---

## Features

- **10-step conversation pipeline** — every turn executes all steps, mandatory and in order
- **Semantic memory** with ONNX embeddings (`paraphrase-multilingual-MiniLM-L12-v2`, 384-dim, L2-normalized)
- **Graph-based RAG** context retrieval with 3-hop traversal, GNN pattern analysis, and PageRank scoring
- **Prime-Radiant coherence checking** — detects contradictions and revises or halts before API call
- **Claude API integration** using the Bridge Code Pattern — all orchestration logic lives in Rust
- **Proof-carrying verification** with SHAKE-256 witness chain per response
- **SONA trajectory learning** and pattern discovery (observation mode, no weight updates)
- **Immutable RVF audit trail** — append-only hash chain of every conversation turn
- **Optional MCP tools integration** — load external tools from `mcp_servers.json`
- **Turkish/English language detection** with multilingual fallback
- **Graceful degradation** — graph and SONA failures degrade silently; auth and embedding failures halt

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| [Rust](https://rustup.rs) | 1.75+ | Stable toolchain |
| Git | Any | Required for submodule checkout |
| Anthropic API key | — | Required at runtime |
| ONNX model files | — | Optional — hash-based fallback if absent |

---

## Quick Start

```bash
# 1. Clone with submodule
git clone --recurse-submodules <repo-url>
cd ai-assistant

# 2. Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 3. Build
cargo build --release

# 4. Run
./target/release/ai-assistant
```

> **Windows:** Use `.\target\release\ai-assistant.exe`

Once running, type any message at the `>` prompt. The assistant processes your input through all 10 pipeline steps and returns a verified response. Type `quit` or `exit` to stop.

---

## Configuration

All configuration is read from environment variables (loaded from `.env` via [`dotenvy`](Cargo.toml:32)).

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ Yes | — | Anthropic API key for Claude access |
| `ANTHROPIC_BASE_URL` | No | `https://api.anthropic.com` | API endpoint — supports `http://` for local proxies |
| `CLAUDE_MODEL` | No | `claude-opus-4-6` | Claude model identifier |
| `EMBEDDING_MODEL_PATH` | No | `./models/paraphrase-multilingual-MiniLM-L12-v2` | Path to ONNX model directory |

**`.env` example:**

```dotenv
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_BASE_URL=https://api.anthropic.com
CLAUDE_MODEL=claude-opus-4-6
EMBEDDING_MODEL_PATH=./models/paraphrase-multilingual-MiniLM-L12-v2
```

> ⚠️ Never commit your `.env` file. It is listed in [`.gitignore`](.gitignore).

---

## ONNX Embedding Model

The assistant uses `paraphrase-multilingual-MiniLM-L12-v2` for 384-dimensional semantic embeddings.
Without the ONNX model, a hash-based fallback is used (lower semantic quality).

```bash
mkdir -p models/paraphrase-multilingual-MiniLM-L12-v2

# Download from Hugging Face (ONNX format required):
# https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# The directory must contain:
#   model.onnx       — the ONNX model weights
#   tokenizer.json   — the tokenizer vocabulary
```

To enable ONNX support at compile time, build with the `onnx` feature:

```bash
cargo build --release --features onnx
```

Without `--features onnx`, the crate compiles without the `ort` and `ndarray` dependencies and
uses the hash-based embedding fallback defined in [`src/embedding.rs`](src/embedding.rs).

---

## MCP Tools (Optional)

External tools can be registered in [`mcp_servers.json`](mcp_servers.json) at the project root.
If the file is absent, the assistant runs without tool support — this is not an error.

**Format:**

```json
{
  "mcpServers": {
    "my-tool": {
      "command": "npx",
      "args": ["-y", "@mcp/some-tool"]
    }
  }
}
```

**Allowed launcher commands:** `node`, `npx`, `python`, `python3`, `uvx`, `deno`, `bun`

> Tool execution is fully managed by the Rust process in [`src/mcp_tools.rs`](src/mcp_tools.rs).
> Claude cannot invoke tools directly — it only requests them; Rust executes and returns results.

---

## Architecture Overview

Every user turn passes through exactly **10 mandatory steps** in [`src/pipeline.rs`](src/pipeline.rs):

```
User Input
    │
    ▼
Step 1: Receive & Validate        — sanitize input, detect language (Turkish/English/Other)
    │
    ▼
Step 2: Semantic Memory Search    — ONNX embeddings → AgenticDB (Hot/Warm/Cold tiers)
    │
    ▼
Step 3: Graph Context + RAG       — 3-hop graph traversal, GNN patterns, PageRank, MinCut
    │
    ▼
Step 4: Merge Context + Prompt    — priority merge (Hot→Warm→Cold), 4096 token budget
    │
    ▼
Step 5: Coherence Check           — prime-radiant energy scoring
    │                               < 0.1 → Reflex  |  ≥ 0.8 → HALT ⛔
    ▼
Step 6: Claude API Call           — Bridge Code Pattern + optional MCP tool loop
    │
    ▼
Step 7: Security Check            — hallucination detection (> 0.7) + SHAKE-256 witness chain
    │
    ▼
Step 8: SONA Learning             — trajectory recording + K-means++ pattern discovery
    │
    ▼
Step 9: AgenticDB Update + Audit  — episode store + RVF immutable hash chain
    │
    ▼
Step 10: Return Verified Response
    │
    ▼
User Output
```

**Graceful degradation:** Graph context failures (Step 3) and SONA failures (Step 8) produce empty
results and the pipeline continues. Coherence critical halts (Step 5), API failures (Step 6),
embedding failures (Step 2), and audit failures (Step 9) terminate the turn with an error.

### Ruvector Subsystems Used

| Subsystem | Crate | Role |
|---|---|---|
| AgenticDB | `ruvector-core` | Semantic episode storage |
| SONA | `ruvector-sona` | Trajectory learning |
| Graph / RAG | `ruvector-graph` | 3-hop context retrieval |
| Prime-Radiant | `prime-radiant` | Coherence & hallucination scoring |
| Verified | `ruvector-verified` | Proof-carrying response validation |
| Cognitive Container | `ruvector-cognitive-container` | Witness chain storage |
| RVF Runtime / Crypto | `rvf-runtime`, `rvf-crypto` | Immutable audit trail |

---

## Security Notes

- **Never commit `.env`** — your API key lives there and is gitignored
- **API key isolation** — read once at startup by [`src/config.rs`](src/config.rs), never logged or serialized
- **Input sanitization** — null bytes and control characters stripped; input capped at 10,000 chars
- **Token budget** — hard cap of 4,096 tokens prevents prompt injection overflow
- **MCP allowlist** — only approved launcher commands can be used in `mcp_servers.json`
- **Immutable audit trail** — every turn is SHAKE-256 hashed and chained; tamper-evident
- **Proof-carrying responses** — each response carries a cryptographic proof from `ruvector-verified`

For full security controls, see [`docs/security-audit.md`](docs/security-audit.md).

---

## Development

```bash
# Run all tests
cargo test

# Run with debug-level tracing
RUST_LOG=debug cargo run

# Run with info-level tracing (default)
RUST_LOG=info cargo run

# Build optimized release binary
cargo build --release

# Build with ONNX embedding support
cargo build --release --features onnx

# Run a specific test module
cargo test test_pipeline
```

Logging uses the [`tracing`](Cargo.toml:37) crate. Set `RUST_LOG` to `error`, `warn`, `info`, or `debug`.

---

## Project Structure

```
ai-assistant/
├── Cargo.toml                  # Package manifest; ruvector path dependencies
├── .env.example                # Environment variable template (no secrets)
├── mcp_servers.json            # Optional MCP tool server definitions
├── prompts/
│   ├── system_en.md            # English system prompt
│   └── system_tr.md            # Turkish system prompt
├── models/                     # ONNX model files (gitignored)
│   └── paraphrase-multilingual-MiniLM-L12-v2/
│       ├── model.onnx
│       └── tokenizer.json
├── ruvector/                   # Git submodule — read-only, do not modify
├── docs/
│   ├── specification.md        # Full technical specification and pseudocode
│   ├── architecture.md         # Module diagrams, interfaces, data flow
│   └── security-audit.md       # Security controls and audit findings
├── src/
│   ├── main.rs                 # Entry point and REPL loop
│   ├── config.rs               # Environment loading, Config struct
│   ├── types.rs                # All shared data types (no business logic)
│   ├── error.rs                # AiAssistantError enum
│   ├── language.rs             # Language detection (whatlang)
│   ├── embedding.rs            # ONNX embedding provider
│   ├── memory.rs               # AgenticDB wrapper, tiered episode storage
│   ├── graph_context.rs        # Graph queries, RAG, GNN, PageRank
│   ├── coherence.rs            # Prime-Radiant coherence engine
│   ├── claude_api.rs           # Claude HTTP client (Bridge Code Pattern)
│   ├── mcp_tools.rs            # MCP tool loader and executor
│   ├── verification.rs         # Proof validation, SHAKE-256 witness chain
│   ├── learning.rs             # SONA trajectory recording
│   ├── audit.rs                # RVF immutable audit trail
│   ├── pipeline.rs             # 10-step orchestrator
│   └── lib.rs                  # Library root (re-exports all modules)
└── tests/
    ├── test_config.rs
    ├── test_embedding.rs
    ├── test_coherence.rs
    ├── test_audit.rs
    └── test_graph_context.rs
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
