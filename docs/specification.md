# AI Assistant — Full Technical Specification & Pseudocode with TDD Anchors

> **Version:** 1.0.0  
> **Date:** 2026-02-26  
> **Status:** Draft  
> **Language:** Rust (pure)  
> **LLM Backend:** Claude API via reqwest  
> **Semantic Memory:** ruvector ecosystem (git submodule — read-only)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Constraints](#2-key-constraints)
3. [Module Inventory](#3-module-inventory)
4. [Data Types](#4-data-types)
5. [10-Step Conversation Pipeline](#5-10-step-conversation-pipeline)
6. [Error Handling Strategy](#6-error-handling-strategy)
7. [Configuration](#7-configuration)
8. [Dependency Map](#8-dependency-map)
9. [Performance Considerations](#9-performance-considerations)
10. [Security Requirements](#10-security-requirements)

---

## 1. Project Overview

A Rust-based AI assistant that leverages the **ruvector** ecosystem for:

- **Semantic memory** — vector-based episode storage and retrieval
- **Graph-based RAG** — Cypher queries, multi-hop retrieval, GNN analysis
- **Coherence checking** — prime-radiant contradiction energy detection
- **Verified computation** — proof-carrying responses with witness chains
- **Learning** — SONA trajectory recording and pattern discovery

**Claude API** serves as the text generation backend only. All orchestration logic,
tool execution, memory management, and verification run in Rust. Every conversation
turn executes a **mandatory 10-step pipeline**.

### Success Criteria

- Each conversation turn completes all 10 pipeline steps
- Responses are coherence-verified before delivery
- All turns produce an auditable RVF trail with SHAKE-256 hashes
- Semantic memory improves retrieval quality over time
- System degrades gracefully when optional subsystems are unavailable

---

## 2. Key Constraints

| Constraint | Detail |
|---|---|
| Language | Pure Rust — no FFI to Python/C++ except ONNX runtime |
| ruvector submodule | Git submodule at `./ruvector/` — **NO files inside ruvector/ may be modified** |
| Claude API | Called via `reqwest` HTTP POST — Claude ONLY generates text |
| Embedding model | Custom ONNX: `paraphrase-multilingual-MiniLM-L12-v2` — **NOT** default `HashEmbedding` |
| Environment | All secrets from `.env` via `dotenvy` — no hardcoded secrets |
| File size | Every source file MUST be < 500 lines |
| Token budget | Context window capped at 4096 tokens per prompt |
| MCP tools | Optional — loaded from `mcp_servers.json` at startup if present |

---

## 3. Module Inventory

Each module has a **single responsibility** and stays under 500 lines.

### 3.1 `src/main.rs` — Entry Point & REPL Loop

**Responsibility:** Bootstrap the application, initialize all subsystems, run the
interactive CLI/REPL loop, and dispatch each user message through the 10-step pipeline.

```
FUNCTION main():
    config = Config::load_from_env()
    embedding = OnnxEmbeddingProvider::new(config.embedding_model_path)
    memory = AgenticDB::new_with_embedding(embedding)
    graph = RuvectorGraph::new()
    coherence = PrimeRadiant::new()
    verifier = RuvectorVerified::new()
    container = CognitiveContainer::new()
    sona = Sona::new()
    rvf = RvfRuntime::new()
    crypto = RvfCrypto::new()
    mcp_tools = load_mcp_tools_if_present("mcp_servers.json")
    session = Session::new()

    // TEST: test all subsystems initialize without panic
    // TEST: test REPL reads from stdin and dispatches to pipeline

    LOOP:
        raw_input = read_line_from_stdin()
        IF raw_input == "exit" OR raw_input == "quit":
            BREAK
        result = pipeline::execute(raw_input, &config, &memory, &graph,
                                   &coherence, &verifier, &container,
                                   &sona, &rvf, &crypto, &mcp_tools, &mut session)
        MATCH result:
            Ok(response) => print(response)
            Err(e) => print_error(e)
```

### 3.2 `src/config.rs` — Configuration

**Responsibility:** Load all environment variables from `.env` via `dotenvy`, validate
them, and expose a typed `Config` struct.

```
STRUCT Config:
    anthropic_api_key: String      // ANTHROPIC_API_KEY — required
    anthropic_base_url: String     // ANTHROPIC_BASE_URL — required
    claude_model: String           // CLAUDE_MODEL — required
    embedding_model_path: String   // EMBEDDING_MODEL_PATH — required

FUNCTION Config::load_from_env() -> Result<Config, AiAssistantError>:
    dotenvy::dotenv().ok()  // load .env file, ignore if missing
    
    api_key = env::var("ANTHROPIC_API_KEY")
        .map_err(|_| AiAssistantError::Config("ANTHROPIC_API_KEY not set"))
    base_url = env::var("ANTHROPIC_BASE_URL")
        .map_err(|_| AiAssistantError::Config("ANTHROPIC_BASE_URL not set"))
    model = env::var("CLAUDE_MODEL")
        .map_err(|_| AiAssistantError::Config("CLAUDE_MODEL not set"))
    embed_path = env::var("EMBEDDING_MODEL_PATH")
        .map_err(|_| AiAssistantError::Config("EMBEDDING_MODEL_PATH not set"))
    
    validate_api_key_format(api_key)?
    validate_url_format(base_url)?
    validate_file_exists(embed_path)?
    
    RETURN Ok(Config { anthropic_api_key, anthropic_base_url, claude_model, embedding_model_path })
```

// TEST: test missing env var returns Config error variant  
// TEST: test valid .env loads all fields  
// TEST: test invalid URL format rejected  
// TEST: test nonexistent embedding model path rejected  

### 3.3 `src/embedding.rs` — ONNX Embedding Provider

**Responsibility:** Wrap the ONNX runtime to produce embeddings using
`paraphrase-multilingual-MiniLM-L12-v2`. Implements the ruvector `EmbeddingProvider`
trait so AgenticDB uses real semantic vectors instead of hash-based fallback.

```
STRUCT OnnxEmbeddingProvider:
    session: ort::Session   // ONNX runtime session
    tokenizer: Tokenizer    // HuggingFace tokenizer for the model

FUNCTION OnnxEmbeddingProvider::new(model_path: &str) -> Result<Self, AiAssistantError>:
    session = ort::Session::builder()
        .with_model_from_file(model_path)?
    tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", parent_dir(model_path)))?
    RETURN Ok(Self { session, tokenizer })

IMPL EmbeddingProvider FOR OnnxEmbeddingProvider:
    FUNCTION embed(&self, text: &str) -> Result<Vec<f32>, Error>:
        tokens = self.tokenizer.encode(text, add_special_tokens=true)
        input_ids = tokens.get_ids()          // Vec<u32>
        attention_mask = tokens.get_attention_mask()
        
        outputs = self.session.run(
            inputs: [input_ids, attention_mask]
        )?
        
        // Mean pooling over token dimension
        embeddings = mean_pool(outputs[0], attention_mask)
        
        // L2 normalize
        normalized = l2_normalize(embeddings)
        
        RETURN Ok(normalized)   // Vec<f32> of dim 384

FUNCTION mean_pool(token_embeddings: Tensor, mask: Tensor) -> Vec<f32>:
    // Apply attention mask, sum across token dimension, divide by mask sum
    masked = token_embeddings * mask.unsqueeze(-1)
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    RETURN summed / counts

FUNCTION l2_normalize(vec: Vec<f32>) -> Vec<f32>:
    norm = sqrt(sum(v*v for v in vec))
    IF norm == 0: RETURN vec
    RETURN vec.iter().map(|v| v / norm).collect()
```

// TEST: test embedding output dimension is 384  
// TEST: test L2 norm of output ≈ 1.0  
// TEST: test similar texts produce high cosine similarity  
// TEST: test dissimilar texts produce low cosine similarity  
// TEST: test empty string handled gracefully  
// TEST: test multilingual input produces valid embeddings  

### 3.4 `src/memory.rs` — AgenticDB Wrapper

**Responsibility:** Initialize AgenticDB with the custom ONNX embedding provider.
Expose episode storage, retrieval, skill search, causal edges, temporal tiered
compression, and auto-consolidation.

```
STRUCT MemoryStore:
    db: AgenticDB<OnnxEmbeddingProvider>
    temporal: TemporalTensor

FUNCTION MemoryStore::new(embedding: OnnxEmbeddingProvider) -> Self:
    db = AgenticDB::builder()
        .with_embedding_provider(embedding)   // MUST use custom, NOT default
        .build()
    temporal = TemporalTensor::new()
    RETURN Self { db, temporal }

FUNCTION retrieve_similar_episodes(&self, embedding: &[f32], top_k: usize) -> Vec<Episode>:
    RETURN self.db.search_episodes(embedding, top_k)

FUNCTION search_skills(&self, embedding: &[f32], top_k: usize) -> Vec<Skill>:
    RETURN self.db.search_skills(embedding, top_k)

FUNCTION store_episode(&self, episode: Episode) -> Result<EpisodeId, AiAssistantError>:
    RETURN self.db.insert_episode(episode)

FUNCTION add_causal_edge(&self, from: &UserMessage, to: &VerifiedResponse):
    self.db.add_causal_edge(from.text, to.text)

FUNCTION get_causal_edges(&self, topic: &str) -> Vec<CausalEdge>:
    RETURN self.db.query_causal_edges(topic)

FUNCTION auto_consolidate(&self):
    self.db.consolidate_low_quality_episodes()

FUNCTION apply_tiered_compression(&self, episode: &mut Episode):
    age = now() - episode.timestamp
    IF age < Duration::hours(24):
        episode.tier = Tier::Hot
        self.temporal.quantize(episode, bits=8)
    ELIF age < Duration::days(7):
        episode.tier = Tier::Warm
        self.temporal.quantize(episode, bits=6)   // 5-7 bit range
    ELSE:
        episode.tier = Tier::Cold
        self.temporal.quantize(episode, bits=3)
```

// TEST: test AgenticDB initialized with OnnxEmbeddingProvider (not HashEmbedding)  
// TEST: test episode storage and retrieval roundtrip  
// TEST: test tiered compression — episode < 24h gets 8-bit (Hot)  
// TEST: test tiered compression — episode 1-7d gets 5-7-bit (Warm)  
// TEST: test tiered compression — episode > 7d gets 3-bit (Cold)  
// TEST: test causal edge creation and query  
// TEST: test auto_consolidate removes low-quality duplicates  

### 3.5 `src/graph_context.rs` — Graph Context, RAG, Topic Analysis

**Responsibility:** Execute Cypher queries on the ruvector-graph, run multi-hop RAG
retrieval, GNN relationship analysis, PageRank importance scoring, and MinCut
topic clustering.

```
STRUCT GraphStore:
    graph: RuvectorGraph
    rag_engine: RagEngine

FUNCTION GraphStore::new() -> Self:
    graph = RuvectorGraph::new()
    rag_engine = RagEngine::new(&graph)
    RETURN Self { graph, rag_engine }

FUNCTION cypher_query(&self, query: &str, params: HashMap) -> Vec<GraphEntity>:
    RETURN self.graph.execute_cypher(query, params)

FUNCTION rag_retrieve(&self, query: &str, config: &RagConfig) -> RagContext:
    RETURN self.rag_engine.retrieve(query, config)

FUNCTION gnn_analyze(&self, entities: &[GraphEntity]) -> Vec<GnnPattern>:
    RETURN self.graph.gnn_analyze(entities)

FUNCTION get_pagerank(&self) -> HashMap<NodeId, f64>:
    RETURN ruvector_solver::pagerank(&self.graph, sqrt_iterations=true)

FUNCTION find_topic_clusters(&self) -> Vec<TopicGroup>:
    RETURN ruvector_mincut::find_clusters(&self.graph)

FUNCTION remove_irrelevant(&self, clusters: &[TopicGroup], query: &str) -> Subgraph:
    RETURN ruvector_mincut::remove_irrelevant(&self.graph, clusters, query)

FUNCTION is_empty(&self) -> bool:
    RETURN self.graph.node_count() == 0
```

// TEST: test Cypher query returns expected node/edge structure  
// TEST: test RAG retrieval respects min_relevance threshold  
// TEST: test RAG retrieval respects max_context_tokens limit  
// TEST: test GNN analyze returns patterns for connected entities  
// TEST: test PageRank with sqrt iterations converges  
// TEST: test MinCut correctly separates unrelated topics  
// TEST: test is_empty returns true for new graph  

### 3.6 `src/coherence.rs` — Prime-Radiant Coherence Engine

**Responsibility:** Compute contradiction energy on context, classify into reflex/
revised/critical lanes, revise context when needed, and detect hallucinations
in generated responses.

```
CONST REFLEX_THRESHOLD: f64 = 0.1
CONST CRITICAL_THRESHOLD: f64 = 0.8
CONST HALLUCINATION_THRESHOLD: f64 = 0.7

STRUCT CoherenceEngine:
    engine: PrimeRadiant

FUNCTION CoherenceEngine::new() -> Self:
    engine = PrimeRadiant::new()
    RETURN Self { engine }

FUNCTION compute_contradiction_energy(&self, context: &str) -> f64:
    RETURN self.engine.compute_contradiction_energy(context)

FUNCTION check_coherence(&self, prompt: &FinalPrompt) -> CoherenceResult:
    energy = self.compute_contradiction_energy(&prompt.context_as_text())
    
    IF energy < REFLEX_THRESHOLD:
        RETURN CoherenceResult::Reflex       // < 1ms, proceed immediately
    ELIF energy < CRITICAL_THRESHOLD:
        revised = self.engine.revise_context(prompt.context_as_text())
        RETURN CoherenceResult::Revised(revised)  // ~10ms retrieval lane
    ELSE:
        RETURN CoherenceResult::Critical     // halt, notify user

FUNCTION detect_hallucination(&self, response: &str, context: &str) -> f64:
    RETURN self.engine.detect_hallucination(response, context)

FUNCTION revise_for_hallucination(&self, response: &str, context: &str) -> String:
    // Attempt to strip unsupported claims
    RETURN self.engine.strip_unsupported_claims(response, context)
```

// TEST: test energy below REFLEX_THRESHOLD returns Reflex  
// TEST: test energy between REFLEX and CRITICAL returns Revised with modified context  
// TEST: test energy above CRITICAL_THRESHOLD returns Critical  
// TEST: test hallucination score > threshold triggers revision  
// TEST: test hallucination score ≤ threshold passes through  
// TEST: test threshold constants are in valid range (0.0..1.0)  

### 3.7 `src/claude_api.rs` — Claude API HTTP Client

**Responsibility:** Build and send HTTP POST requests to the Claude Messages API via
`reqwest`. Handle tool_use response blocks by executing tools in Rust and re-calling
Claude with results. Parse response into typed structs.

```
STRUCT ClaudeClient:
    http: reqwest::Client
    config: Config

FUNCTION ClaudeClient::new(config: &Config) -> Self:
    http = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
    RETURN Self { http, config: config.clone() }

FUNCTION call(&self, prompt: &FinalPrompt, tools: Option<&[Tool]>) -> Result<ClaudeResponse, AiAssistantError>:
    url = format!("{}/v1/messages", self.config.anthropic_base_url)
    
    body = ClaudeRequestBody {
        model: self.config.claude_model.clone(),
        max_tokens: 4096,
        system: prompt.system.clone(),
        messages: vec![
            Message { role: "user", content: prompt.full_content() }
        ],
        tools: tools.map(|t| t.to_vec()).unwrap_or_default()
    }
    
    response = self.http.post(&url)
        .header("x-api-key", &self.config.anthropic_api_key)
        .header("content-type", "application/json")
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .send()
        .await?
    
    IF response.status() == 401:
        RETURN Err(AiAssistantError::Auth("Invalid API key"))
    IF response.status() == 429:
        RETURN Err(AiAssistantError::RateLimit("Rate limited, retry later"))
    IF response.status() >= 500:
        RETURN Err(AiAssistantError::ApiServer("Claude API server error"))
    
    parsed = response.json::<ClaudeApiResponse>().await?
    RETURN Ok(ClaudeResponse::from(parsed))

FUNCTION call_with_tool_results(&self, prompt: &FinalPrompt, 
                                 tool_results: Vec<ToolResult>,
                                 tools: Option<&[Tool]>) -> Result<ClaudeResponse, AiAssistantError>:
    // Build messages array with tool_result blocks
    messages = vec![
        Message { role: "user", content: prompt.full_content() },
        Message { role: "assistant", content: original_tool_use_blocks },
        Message { role: "user", content: tool_results_as_content(tool_results) }
    ]
    // Re-call with same structure
    RETURN self.call_with_messages(messages, tools).await
```

// TEST: test successful API call returns parsed ClaudeResponse  
// TEST: test 401 returns Auth error  
// TEST: test 429 returns RateLimit error  
// TEST: test 500+ returns ApiServer error  
// TEST: test tool_use blocks detected in response  
// TEST: test re-call with tool results builds correct message chain  
// TEST: test request headers include x-api-key and anthropic-version  
// TEST: test request body serialization matches Claude API schema  
// TEST: test timeout after 60 seconds  

### 3.8 `src/mcp_tools.rs` — MCP Tool Loader & Manager

**Responsibility:** Parse `mcp_servers.json` at startup to discover available MCP
tools. Convert them to Claude tool-use format. Execute tool calls dispatched
from Claude responses.

```
STRUCT McpToolManager:
    tools: Vec<McpTool>
    servers: HashMap<String, McpServerConfig>

STRUCT McpTool:
    name: String
    description: String
    input_schema: serde_json::Value
    server_name: String

STRUCT McpServerConfig:
    command: String
    args: Vec<String>
    env: HashMap<String, String>

FUNCTION McpToolManager::load(path: &str) -> Result<Option<Self>, AiAssistantError>:
    IF NOT file_exists(path):
        RETURN Ok(None)   // MCP tools are optional
    
    content = read_file(path)?
    config = serde_json::from_str::<McpConfig>(&content)?
    
    tools = Vec::new()
    servers = HashMap::new()
    
    FOR (server_name, server_config) IN config.mcp_servers:
        discovered = discover_tools_from_server(server_name, server_config)?
        tools.extend(discovered)
        servers.insert(server_name, server_config)
    
    RETURN Ok(Some(Self { tools, servers }))

FUNCTION to_claude_tools(&self) -> Vec<Tool>:
    RETURN self.tools.iter().map(|t| Tool {
        name: t.name.clone(),
        description: t.description.clone(),
        input_schema: t.input_schema.clone()
    }).collect()

FUNCTION execute_tool(&self, name: &str, input: serde_json::Value) -> Result<ToolResult, AiAssistantError>:
    tool = self.tools.iter().find(|t| t.name == name)
        .ok_or(AiAssistantError::ToolNotFound(name))?
    server = self.servers.get(&tool.server_name)
        .ok_or(AiAssistantError::ServerNotFound(tool.server_name))?
    
    result = invoke_mcp_server(server, name, input)?
    RETURN Ok(ToolResult { tool_use_id, content: result })
```

// TEST: test missing mcp_servers.json returns Ok(None)  
// TEST: test valid mcp_servers.json loads tools  
// TEST: test to_claude_tools produces valid Claude tool schema  
// TEST: test execute_tool with known tool returns result  
// TEST: test execute_tool with unknown tool returns ToolNotFound error  
// TEST: test malformed JSON returns parse error  

### 3.9 `src/verification.rs` — Proof Validation & Witness Chain

**Responsibility:** Use ruvector-verified for proof-carrying validation of responses.
Maintain a cognitive container witness chain where each entry is hashed with SHAKE-256.

```
STRUCT VerificationEngine:
    verifier: RuvectorVerified
    container: CognitiveContainer
    witness_chain: Vec<WitnessEntry>

STRUCT WitnessEntry:
    response_hash: Vec<u8>     // SHAKE-256 of response text
    proof: Proof
    prev_hash: Vec<u8>         // hash of previous witness entry
    timestamp: DateTime<Utc>

FUNCTION VerificationEngine::new() -> Self:
    verifier = RuvectorVerified::new()
    container = CognitiveContainer::new()
    RETURN Self { verifier, container, witness_chain: Vec::new() }

FUNCTION validate_response(&self, response_text: &str) -> Result<Proof, AiAssistantError>:
    proof = self.verifier.validate(response_text)?
    RETURN Ok(proof)

FUNCTION record_witness(&mut self, response: &str, proof: &Proof) -> WitnessEntry:
    response_hash = shake256(response.as_bytes())
    
    prev_hash = IF self.witness_chain.is_empty():
        vec![0u8; 32]   // genesis entry
    ELSE:
        shake256(serialize(self.witness_chain.last()))
    
    entry = WitnessEntry {
        response_hash,
        proof: proof.clone(),
        prev_hash,
        timestamp: Utc::now()
    }
    
    self.container.record(entry.clone())
    self.witness_chain.push(entry.clone())
    RETURN entry

FUNCTION verify_chain_integrity(&self) -> bool:
    FOR i IN 1..self.witness_chain.len():
        expected_prev = shake256(serialize(self.witness_chain[i-1]))
        IF self.witness_chain[i].prev_hash != expected_prev:
            RETURN false
    RETURN true
```

// TEST: test validate_response returns proof for valid response  
// TEST: test first witness entry has zero prev_hash (genesis)  
// TEST: test subsequent entries chain correctly (prev_hash matches)  
// TEST: test verify_chain_integrity returns true for untampered chain  
// TEST: test verify_chain_integrity returns false if entry modified  
// TEST: test SHAKE-256 hash output is correct length  

### 3.10 `src/learning.rs` — SONA Trajectory & Pattern Discovery

**Responsibility:** Record conversation trajectories using SONA, compute quality
scores, and discover patterns via K-means++. **Does NOT apply LoRA updates.**

```
STRUCT LearningEngine:
    sona: Sona

FUNCTION LearningEngine::new() -> Self:
    sona = Sona::new()
    RETURN Self { sona }

FUNCTION record_trajectory(&self, turn: &ConversationTurn) -> Result<LearningResult, AiAssistantError>:
    trajectory = self.sona.begin_trajectory()
    
    trajectory.add_step(TrajectoryStep::UserMessage(turn.user_msg.clone()))
    trajectory.add_step(TrajectoryStep::Context(turn.semantic_context.clone()))
    trajectory.add_step(TrajectoryStep::Prompt(turn.prompt.clone()))
    trajectory.add_step(TrajectoryStep::Response(turn.response.clone()))
    
    quality_score = compute_quality(turn)
    trajectory.set_score(quality_score)
    
    trajectory_id = self.sona.end_trajectory(trajectory)?
    
    // Pattern discovery via K-means++
    patterns = self.sona.find_patterns()?
    
    // NOTE: DO NOT call apply_micro_lora() or apply_base_lora()
    // Learning is observation-only in this system
    
    RETURN Ok(LearningResult {
        trajectory_id,
        quality_score,
        new_patterns: patterns
    })

FUNCTION compute_quality(turn: &ConversationTurn) -> f64:
    // Quality heuristics:
    // 1. Coherence score (from step 5/7)
    // 2. Response relevance to query
    // 3. Context utilization ratio
    // 4. Response length appropriateness
    coherence_score = turn.coherence_energy
    relevance = cosine_similarity(turn.user_msg.embedding, turn.response.embedding)
    context_util = turn.context_tokens_used / turn.context_tokens_available
    length_score = sigmoid_length_penalty(turn.response.text.len())
    
    RETURN weighted_average([
        (coherence_score, 0.3),
        (relevance, 0.35),
        (context_util, 0.2),
        (length_score, 0.15)
    ])
```

// TEST: test trajectory records all 4 steps  
// TEST: test quality score is in range [0.0, 1.0]  
// TEST: test pattern finding returns Vec<Pattern>  
// TEST: test LoRA methods are never called (compile-time guarantee via API surface)  
// TEST: test quality weights sum to 1.0  
// TEST: test empty conversation turn handled gracefully  

### 3.11 `src/audit.rs` — RVF Audit Trail

**Responsibility:** Maintain an append-only audit trail using `rvf-runtime` and
`rvf-crypto`. Each entry is SHAKE-256 hashed and chained to the previous entry.

```
STRUCT AuditTrail:
    runtime: RvfRuntime
    crypto: RvfCrypto
    last_hash: Vec<u8>

FUNCTION AuditTrail::new() -> Self:
    runtime = RvfRuntime::new()
    crypto = RvfCrypto::new()
    last_hash = vec![0u8; 32]   // genesis hash
    RETURN Self { runtime, crypto, last_hash }

FUNCTION record(&mut self, turn: &ConversationTurn) -> Result<AuditResult, AiAssistantError>:
    serialized = serde_json::to_string(turn)?
    hash = self.crypto.shake256(serialized.as_bytes())
    
    entry = RvfEntry {
        data: serialized,
        hash: hash.clone(),
        prev_hash: self.last_hash.clone(),
        timestamp: Utc::now()
    }
    
    episode_id = self.runtime.record(entry)?
    self.last_hash = hash.clone()
    
    RETURN Ok(AuditResult { episode_id, hash })

FUNCTION verify_trail(&self) -> Result<bool, AiAssistantError>:
    entries = self.runtime.get_all_entries()?
    FOR i IN 1..entries.len():
        IF entries[i].prev_hash != entries[i-1].hash:
            RETURN Ok(false)
    RETURN Ok(true)

FUNCTION get_last_hash(&self) -> Vec<u8>:
    RETURN self.last_hash.clone()
```

// TEST: test first entry has genesis prev_hash  
// TEST: test hash chain is contiguous  
// TEST: test verify_trail detects tampering  
// TEST: test record returns valid episode_id  
// TEST: test SHAKE-256 produces consistent hashes for same input  

### 3.12 `src/pipeline.rs` — 10-Step Pipeline Orchestrator

**Responsibility:** Orchestrate the 10-step conversation pipeline. Each step
is a separate function call. Handle graceful degradation when optional subsystems
fail. This is the core coordination module.

> Detailed pseudocode in [Section 5](#5-10-step-conversation-pipeline).

### 3.13 `src/types.rs` — Shared Type Definitions

**Responsibility:** Define all shared data structures used across modules. No
business logic — only type definitions, derives, and simple constructors.

> Detailed type definitions in [Section 4](#4-data-types).

### 3.14 `src/language.rs` — Language Detection

**Responsibility:** Detect the language of user input messages using a lightweight
heuristic or crate (e.g., `whatlang`). Returns an ISO 639-1 language code.

```
FUNCTION detect_language(text: &str) -> Language:
    IF text.is_empty():
        RETURN Language::Unknown
    
    detection = whatlang::detect(text)
    MATCH detection:
        Some(info) IF info.confidence() > 0.5 => 
            RETURN Language::from_whatlang(info.lang())
        _ => 
            RETURN Language::default()   // English fallback
```

// TEST: test English text detected as "en"  
// TEST: test Turkish text detected as "tr"  
// TEST: test empty string returns Unknown  
// TEST: test very short text falls back to default  
// TEST: test confidence threshold respected  

---

## 4. Data Types

All shared types reside in `src/types.rs`.

### 4.1 Core Message Types

```
STRUCT UserMessage:
    text: String                    // Raw user input, validated non-empty
    language: Language              // Detected language (ISO 639-1)
    timestamp: DateTime<Utc>       // When message was received
    embedding: Option<Vec<f32>>    // Populated in step 2

ENUM Language:
    English,
    Turkish,
    German,
    French,
    Spanish,
    Unknown,
    Other(String)                  // ISO 639-1 code

    FUNCTION default() -> Self: Language::English
    FUNCTION code(&self) -> &str   // Returns "en", "tr", etc.
```

// TEST: test UserMessage requires non-empty text  
// TEST: test Language::code() returns correct ISO codes  

### 4.2 Context Types

```
STRUCT SemanticContext:
    episodes: Vec<Episode>         // Ranked by relevance after OT distance
    skills: Vec<Skill>             // Matched skills from AgenticDB

STRUCT GraphContext:
    entities: Vec<GraphEntity>     // From Cypher query
    rag_context: RagContext        // Multi-hop RAG retrieval result
    gnn_patterns: Vec<GnnPattern>  // GNN relationship analysis
    causal: Vec<CausalEdge>        // Causal edges from AgenticDB
    pagerank: HashMap<NodeId, f64> // Importance scores
    relevant_subgraph: Subgraph    // After MinCut pruning

STRUCT GraphEntity:
    id: NodeId
    label: String
    properties: HashMap<String, serde_json::Value>
    edges: Vec<Edge>

STRUCT RagContext:
    documents: Vec<RagDocument>
    total_tokens: usize

STRUCT RagDocument:
    content: String
    relevance_score: f64
    source: String
    hop_depth: u32                 // How many hops from query

STRUCT RagConfig:
    top_k_docs: usize              // Default: 5
    min_relevance: f64             // Default: 0.7
    max_context_tokens: usize      // Default: 4096
    max_hops: u32                  // Default: 3
```

// TEST: test SemanticContext can hold zero episodes (empty memory)  
// TEST: test RagConfig defaults are applied  
// TEST: test GraphContext fields are all independently optional for graceful degradation  

### 4.3 Prompt Types

```
STRUCT FinalPrompt:
    system: String                 // System prompt (language-aware)
    context: String                // Merged semantic + graph context
    user: String                   // Original user text
    total_tokens: usize            // Estimated total token count

    FUNCTION full_content(&self) -> String:
        RETURN format!("{}\n\n{}", self.context, self.user)

    FUNCTION context_as_text(&self) -> String:
        RETURN self.context.clone()
```

// TEST: test full_content concatenates context and user text  
// TEST: test total_tokens reflects actual prompt size  

### 4.4 Coherence Types

```
ENUM CoherenceResult:
    Reflex,                        // Energy < 0.1 — proceed immediately
    Revised(String),               // Energy 0.1..0.8 — context revised
    Critical                       // Energy ≥ 0.8 — halt pipeline
```

// TEST: test all three variants constructible  
// TEST: test Revised carries modified context string  

### 4.5 API Response Types

```
STRUCT ClaudeResponse:
    text: String                   // Generated response text
    tool_calls: Vec<ToolCall>      // Tool use blocks (may be empty)
    model: String                  // Model that generated response
    usage: Usage                   // Token usage stats

STRUCT ToolCall:
    id: String                     // Unique tool use ID
    name: String                   // Tool name
    input: serde_json::Value       // Tool input arguments

STRUCT ToolResult:
    tool_use_id: String            // Matches ToolCall.id
    content: String                // Tool execution result

STRUCT Usage:
    input_tokens: u32
    output_tokens: u32

STRUCT Tool:
    name: String
    description: String
    input_schema: serde_json::Value
```

// TEST: test ClaudeResponse deserialization from JSON  
// TEST: test has_tool_calls returns true when tool_calls non-empty  
// TEST: test has_tool_calls returns false when empty  

### 4.6 Verification Types

```
STRUCT VerifiedResponse:
    text: String                   // Final response text
    proof: Proof                   // Proof-carrying validation result
    witness: WitnessEntry          // Witness chain entry

STRUCT Proof:
    valid: bool
    confidence: f64
    details: String
```

// TEST: test VerifiedResponse always has proof and witness  

### 4.7 Episode & Memory Types

```
STRUCT Episode:
    id: Option<EpisodeId>
    text: String
    embedding: Vec<f32>
    timestamp: DateTime<Utc>
    tier: Tier

ENUM Tier:
    Hot,                           // < 24h, 8-bit quantization
    Warm,                          // 1-7 days, 5-7-bit quantization
    Cold                           // > 7 days, 3-bit quantization

STRUCT Skill:
    name: String
    description: String
    embedding: Vec<f32>

STRUCT CausalEdge:
    from: String
    to: String
    weight: f64
    timestamp: DateTime<Utc>

TYPE EpisodeId = u64
TYPE NodeId = u64
```

// TEST: test Tier classification by age  
// TEST: test Episode can be serialized/deserialized  

### 4.8 Learning Types

```
STRUCT LearningResult:
    trajectory_id: TrajectoryId
    quality_score: f64             // 0.0..1.0
    new_patterns: Vec<Pattern>

STRUCT Pattern:
    cluster_id: u32
    centroid: Vec<f32>
    member_count: usize

TYPE TrajectoryId = u64
```

// TEST: test quality_score clamped to [0.0, 1.0]  
// TEST: test patterns have valid cluster_ids  

### 4.9 Audit Types

```
STRUCT AuditResult:
    episode_id: EpisodeId
    hash: Vec<u8>                  // SHAKE-256 hash

STRUCT RvfEntry:
    data: String                   // Serialized ConversationTurn
    hash: Vec<u8>
    prev_hash: Vec<u8>
    timestamp: DateTime<Utc>
```

// TEST: test hash length is consistent (32 bytes for SHAKE-256/256)  

### 4.10 Session & Turn Types

```
STRUCT Session:
    turns: Vec<ConversationTurn>
    state: SessionState
    created_at: DateTime<Utc>

STRUCT SessionState:
    turn_count: usize
    total_tokens_used: usize
    last_coherence_energy: f64

STRUCT ConversationTurn:
    user_msg: UserMessage
    semantic_context: SemanticContext
    graph_context: GraphContext
    prompt: FinalPrompt
    response: VerifiedResponse
    coherence_energy: f64
    learning: LearningResult
    audit: AuditResult
    timestamp: DateTime<Utc>
```

// TEST: test Session starts with zero turns  
// TEST: test ConversationTurn has all fields populated after pipeline  

### 4.11 Configuration Types

```
STRUCT Config:
    anthropic_api_key: String      // From ANTHROPIC_API_KEY env var
    anthropic_base_url: String     // From ANTHROPIC_BASE_URL env var
    claude_model: String           // From CLAUDE_MODEL env var
    embedding_model_path: String   // From EMBEDDING_MODEL_PATH env var

    // No Default impl — all fields are required
```

// TEST: test Config has no Default — must be loaded from env  

---

## 5. 10-Step Conversation Pipeline

The pipeline orchestrator lives in `src/pipeline.rs`. Each step is an independent
function that takes explicit inputs and returns explicit outputs. The orchestrator
calls them in sequence, handling errors and graceful degradation.

### Pipeline Orchestrator

```
FUNCTION execute(
    raw_input: &str,
    config: &Config,
    memory: &MemoryStore,
    graph: &GraphStore,
    coherence: &CoherenceEngine,
    verifier: &VerificationEngine,
    sona: &LearningEngine,
    audit: &AuditTrail,
    mcp_tools: &Option<McpToolManager>,
    session: &mut Session
) -> Result<String, AiAssistantError>:

    // STEP 1: Receive and validate user message
    user_msg = step1_receive_message(raw_input)?
    
    // STEP 2: Semantic memory search
    semantic_ctx = step2_semantic_search(&user_msg, memory)?
    
    // STEP 3: Graph context + RAG (graceful degradation if graph empty)
    graph_ctx = step3_graph_context(&user_msg, &semantic_ctx, graph)
        .unwrap_or_else(|e| {
            log::warn("Graph context failed, using empty: {}", e)
            GraphContext::empty()
        })
    
    // STEP 4: Merge context and prepare prompt
    prompt = step4_prepare_prompt(&user_msg, &semantic_ctx, &graph_ctx, session)?
    
    // STEP 5: Pre-API coherence check
    coherence_result = step5_coherence_check(&prompt, coherence)?
    MATCH coherence_result:
        CoherenceResult::Critical =>
            RETURN Err(AiAssistantError::CoherenceCritical(
                "Context contradictions too severe to proceed"))
        CoherenceResult::Revised(new_context) =>
            prompt = prompt.with_revised_context(new_context)
        CoherenceResult::Reflex =>
            // proceed as-is
    
    // STEP 6: Call Claude API
    claude_tools = mcp_tools.as_ref().map(|m| m.to_claude_tools())
    claude_response = step6_call_claude(&prompt, config, claude_tools.as_deref(), mcp_tools)?
    
    // STEP 7: Post-response security check
    verified = step7_security_check(&claude_response, coherence, verifier)?
    
    // STEP 8: SONA learning record (graceful degradation)
    turn_draft = ConversationTurn::draft(&user_msg, &semantic_ctx, &graph_ctx, &prompt, &verified)
    learning = step8_sona_learning(&turn_draft, sona)
        .unwrap_or_else(|e| {
            log::warn("SONA learning failed, skipping: {}", e)
            LearningResult::empty()
        })
    
    // STEP 9: Update memory + RVF audit
    audit_result = step9_update_and_audit(&turn_draft, memory, audit)?
    
    // STEP 10: Return response and update session
    final_turn = turn_draft.finalize(learning, audit_result)
    response = step10_return(&verified, &final_turn, session)
    
    RETURN Ok(response)
```

// TEST: test full pipeline executes all 10 steps  
// TEST: test pipeline handles graph failure gracefully  
// TEST: test pipeline handles SONA failure gracefully  
// TEST: test pipeline halts on Critical coherence  
// TEST: test pipeline revises context on Revised coherence  
// TEST: test pipeline passes through on Reflex coherence  

---

### STEP 1 — Receive User Message

```
FUNCTION step1_receive_message(raw_input: &str) -> Result<UserMessage, AiAssistantError>:
    // Validate input
    trimmed = raw_input.trim()
    
    IF trimmed.is_empty():
        RETURN Err(AiAssistantError::Validation("Input cannot be empty"))
    
    IF trimmed.len() > MAX_INPUT_LENGTH:    // MAX_INPUT_LENGTH = 10_000
        RETURN Err(AiAssistantError::Validation("Input exceeds maximum length"))
    
    // Sanitize: remove null bytes, control characters
    sanitized = sanitize_input(trimmed)
    
    // Detect language
    language = detect_language(&sanitized)
    
    RETURN Ok(UserMessage {
        text: sanitized.to_string(),
        language,
        timestamp: Utc::now(),
        embedding: None    // populated in step 2
    })

CONST MAX_INPUT_LENGTH: usize = 10_000
```

// TEST: test empty string returns Validation error  
// TEST: test whitespace-only string returns Validation error  
// TEST: test string exceeding MAX_INPUT_LENGTH rejected  
// TEST: test null bytes removed from input  
// TEST: test valid input returns UserMessage with correct fields  
// TEST: test language field populated  
// TEST: test timestamp is approximately now  

---

### STEP 2 — Embedding + Semantic Memory Search

```
FUNCTION step2_semantic_search(
    msg: &mut UserMessage,
    memory: &MemoryStore
) -> Result<SemanticContext, AiAssistantError>:
    
    // Generate embedding for user message
    embedding = memory.db.embedding_provider().embed(&msg.text)?
    msg.embedding = Some(embedding.clone())
    
    // Retrieve similar episodes from AgenticDB
    similar_episodes = memory.retrieve_similar_episodes(&embedding, top_k=10)
    
    // Search for relevant skills
    skills = memory.search_skills(&embedding, top_k=5)
    
    // Apply temporal tiered compression to each episode
    FOR episode IN &mut similar_episodes:
        memory.apply_tiered_compression(episode)
    
    // Compute Optimal Transport distances for ranking
    episode_embeddings = similar_episodes.iter()
        .map(|ep| ep.embedding.as_slice())
        .collect::<Vec<_>>()
    
    distances = ruvector_math::optimal_transport(&embedding, &episode_embeddings)
    
    // Rank episodes by information content (lower OT distance = more relevant)
    ranked = zip(similar_episodes, distances)
        .sorted_by(|(_, d1), (_, d2)| d1.partial_cmp(d2))
        .map(|(ep, _)| ep)
        .collect::<Vec<_>>()
    
    RETURN Ok(SemanticContext {
        episodes: ranked,
        skills
    })
```

// TEST: test embedding dimensions match model output (384)  
// TEST: test top_k=10 returns at most 10 episodes  
// TEST: test tiered compression applied to all retrieved episodes  
// TEST: test OT distance ranking puts most relevant first  
// TEST: test empty memory returns empty episodes vec  
// TEST: test skills search returns at most 5 skills  
// TEST: test user message embedding field populated after step 2  

---

### STEP 3 — Graph Context + RAG + Topic Analysis

```
FUNCTION step3_graph_context(
    msg: &UserMessage,
    semantic_ctx: &SemanticContext,
    graph: &GraphStore
) -> Result<GraphContext, AiAssistantError>:
    
    // Early return if graph is empty (graceful degradation)
    IF graph.is_empty():
        RETURN Ok(GraphContext::empty())
    
    // Cypher query for entities related to user message
    entities = graph.cypher_query(
        "MATCH (n)-[r]->(m) WHERE n.text CONTAINS $topic RETURN n, r, m",
        hashmap!{ "topic" => msg.text.clone() }
    )?
    
    // Multi-hop RAG retrieval
    rag_config = RagConfig {
        top_k_docs: 5,
        min_relevance: 0.7,
        max_context_tokens: 4096,
        max_hops: 3
    }
    rag_context = graph.rag_retrieve(&msg.text, &rag_config)?
    
    // GNN relationship analysis
    gnn_patterns = graph.gnn_analyze(&entities)?
    
    // Causal edges from memory
    // NOTE: This crosses into memory module — accept as parameter or via trait
    causal = Vec::new()   // populated by caller if available
    
    // PageRank importance scoring (O(√n) via sqrt iterations)
    pagerank = graph.get_pagerank()?
    
    // MinCut topic clustering
    topic_groups = graph.find_topic_clusters()?
    relevant_subgraph = graph.remove_irrelevant(&topic_groups, &msg.text)?
    
    RETURN Ok(GraphContext {
        entities,
        rag_context,
        gnn_patterns,
        causal,
        pagerank,
        relevant_subgraph
    })
```

// TEST: test empty graph returns GraphContext::empty()  
// TEST: test Cypher query returns expected entity structure  
// TEST: test RAG config respected (top_k, min_relevance, max_tokens)  
// TEST: test multi-hop retrieval limits to max_hops=3  
// TEST: test GNN analysis returns patterns for connected entities  
// TEST: test PageRank scores are normalized  
// TEST: test MinCut removes nodes irrelevant to query  
// TEST: test graph errors propagated correctly  

---

### STEP 4 — Context Merge & Prompt Preparation

```
FUNCTION step4_prepare_prompt(
    msg: &UserMessage,
    sem: &SemanticContext,
    graph: &GraphContext,
    session: &Session
) -> Result<FinalPrompt, AiAssistantError>:
    
    // Load language-specific system prompt
    system_prompt = load_system_prompt(&msg.language)
    
    // Merge contexts with priority: Hot > Warm > Cold
    context_parts = Vec::new()
    
    // Add hot episodes first (highest priority)
    FOR ep IN sem.episodes.iter().filter(|e| e.tier == Tier::Hot):
        context_parts.push(format!("[MEMORY/HOT] {}", ep.text))
    
    // Add RAG context
    FOR doc IN &graph.rag_context.documents:
        context_parts.push(format!("[RAG/hop{}] {}", doc.hop_depth, doc.content))
    
    // Add warm episodes
    FOR ep IN sem.episodes.iter().filter(|e| e.tier == Tier::Warm):
        context_parts.push(format!("[MEMORY/WARM] {}", ep.text))
    
    // Add skills
    FOR skill IN &sem.skills:
        context_parts.push(format!("[SKILL] {}: {}", skill.name, skill.description))
    
    // Add cold episodes last (lowest priority)
    FOR ep IN sem.episodes.iter().filter(|e| e.tier == Tier::Cold):
        context_parts.push(format!("[MEMORY/COLD] {}", ep.text))
    
    // Add recent session context (last 3 turns)
    FOR turn IN session.turns.iter().rev().take(3).rev():
        context_parts.push(format!("[SESSION] User: {} | Assistant: {}", 
            turn.user_msg.text, turn.response.text))
    
    merged_context = context_parts.join("\n\n")
    
    // Build full prompt
    full_prompt = format!("{}\n\n{}\n\nUser: {}", system_prompt, merged_context, msg.text)
    
    // Token limit enforcement
    total_tokens = estimate_tokens(&full_prompt)
    IF total_tokens > 4096:
        // Trim from lowest priority (cold → warm) until under limit
        merged_context = trim_context_to_fit(context_parts, 4096 - estimate_tokens(&system_prompt) - estimate_tokens(&msg.text))
        total_tokens = estimate_tokens(&format!("{}\n\n{}\n\nUser: {}", system_prompt, merged_context, msg.text))
    
    RETURN Ok(FinalPrompt {
        system: system_prompt,
        context: merged_context,
        user: msg.text.clone(),
        total_tokens
    })

FUNCTION load_system_prompt(language: &Language) -> String:
    MATCH language:
        Language::English => include_str!("../prompts/system_en.txt")
        Language::Turkish => include_str!("../prompts/system_tr.txt")
        _ => include_str!("../prompts/system_en.txt")   // fallback

FUNCTION estimate_tokens(text: &str) -> usize:
    // Rough estimate: ~4 chars per token for English, ~3 for other languages
    RETURN text.len() / 4

FUNCTION trim_context_to_fit(parts: Vec<String>, max_tokens: usize) -> String:
    // Remove from end (lowest priority = cold) first
    result = Vec::new()
    current_tokens = 0
    FOR part IN parts:
        part_tokens = estimate_tokens(&part)
        IF current_tokens + part_tokens <= max_tokens:
            result.push(part)
            current_tokens += part_tokens
        ELSE:
            BREAK
    RETURN result.join("\n\n")
```

// TEST: test hot episodes appear before warm and cold  
// TEST: test token limit enforced at 4096  
// TEST: test cold episodes trimmed first when over limit  
// TEST: test system prompt loaded for correct language  
// TEST: test English fallback for unknown language  
// TEST: test session context includes last 3 turns  
// TEST: test empty context produces valid prompt  
// TEST: test estimate_tokens produces reasonable estimate  

---

### STEP 5 — Pre-API Coherence Check

```
FUNCTION step5_coherence_check(
    prompt: &FinalPrompt,
    coherence: &CoherenceEngine
) -> Result<CoherenceResult, AiAssistantError>:
    
    result = coherence.check_coherence(prompt)
    
    MATCH &result:
        CoherenceResult::Reflex =>
            log::debug("Coherence: Reflex lane (<1ms)")
        CoherenceResult::Revised(ctx) =>
            log::info("Coherence: Revised lane (~10ms), context modified")
        CoherenceResult::Critical =>
            log::warn("Coherence: Critical — contradictions too severe")
    
    RETURN Ok(result)
```

// TEST: test low energy (< 0.1) returns Reflex  
// TEST: test medium energy (0.1..0.8) returns Revised with modified context  
// TEST: test high energy (≥ 0.8) returns Critical  
// TEST: test Revised result contains different context than input  
// TEST: test Critical result halts pipeline in orchestrator  

---

### STEP 6 — Claude API Call

```
FUNCTION step6_call_claude(
    prompt: &FinalPrompt,
    config: &Config,
    tools: Option<&[Tool]>,
    mcp_manager: &Option<McpToolManager>
) -> Result<ClaudeResponse, AiAssistantError>:
    
    client = ClaudeClient::new(config)
    
    response = client.call(prompt, tools)?
    
    // Handle tool use blocks
    IF response.has_tool_calls():
        tool_results = Vec::new()
        
        FOR tool_call IN &response.tool_calls:
            MATCH mcp_manager:
                Some(manager) =>
                    result = manager.execute_tool(&tool_call.name, tool_call.input.clone())?
                    tool_results.push(result)
                None =>
                    RETURN Err(AiAssistantError::ToolExecution(
                        format!("Tool {} requested but no MCP manager available", tool_call.name)))
        
        // Re-call Claude with tool results
        final_response = client.call_with_tool_results(prompt, tool_results, tools)?
        
        // Prevent infinite tool-use loops
        IF final_response.has_tool_calls():
            RETURN Err(AiAssistantError::ToolExecution(
                "Claude requested tools again after results — aborting to prevent loop"))
        
        RETURN Ok(final_response)
    
    RETURN Ok(response)
```

// TEST: test successful API call returns valid ClaudeResponse  
// TEST: test tool_use blocks trigger tool execution  
// TEST: test tool results re-submitted to Claude  
// TEST: test double tool_use loop prevented  
// TEST: test missing MCP manager with tool_use returns error  
// TEST: test 401 auth error propagated  
// TEST: test 429 rate limit error propagated  
// TEST: test 500 server error propagated  
// TEST: test timeout after 60 seconds  

---

### STEP 7 — Post-Response Security Check

```
FUNCTION step7_security_check(
    response: &ClaudeResponse,
    coherence: &CoherenceEngine,
    verifier: &mut VerificationEngine
) -> Result<VerifiedResponse, AiAssistantError>:
    
    // 1. Hallucination detection via coherence engine
    hallucination_energy = coherence.detect_hallucination(&response.text, &context)
    
    final_text = IF hallucination_energy > HALLUCINATION_THRESHOLD:
        log::warn("Hallucination detected (energy={}), revising response", hallucination_energy)
        revised = coherence.revise_for_hallucination(&response.text, &context)
        format!("⚠️ Note: Some claims could not be verified.\n\n{}", revised)
    ELSE:
        response.text.clone()
    
    // 2. Proof-carrying validation
    proof = verifier.validate_response(&final_text)?
    
    // 3. Witness chain recording
    witness = verifier.record_witness(&final_text, &proof)
    
    RETURN Ok(VerifiedResponse {
        text: final_text,
        proof,
        witness
    })
```

// TEST: test hallucination below threshold passes response through unchanged  
// TEST: test hallucination above threshold adds warning prefix  
// TEST: test proof validation produces Proof struct  
// TEST: test witness entry recorded with SHAKE-256 hash  
// TEST: test witness chain links to previous entry  
// TEST: test response text preserved when no hallucination  

---

### STEP 8 — SONA Learning Record

```
FUNCTION step8_sona_learning(
    turn: &ConversationTurn,
    sona: &LearningEngine
) -> Result<LearningResult, AiAssistantError>:
    
    result = sona.record_trajectory(turn)?
    
    log::info("Learning: trajectory={}, quality={:.2}, patterns={}",
        result.trajectory_id, result.quality_score, result.new_patterns.len())
    
    RETURN Ok(result)
```

// TEST: test trajectory recorded with all 4 steps  
// TEST: test quality score computed and in [0.0, 1.0]  
// TEST: test pattern finding executes (K-means++)  
// TEST: test no LoRA methods called  
// TEST: test failure returns Err (caught by pipeline for graceful degradation)  

---

### STEP 9 — AgenticDB Update & RVF Audit

```
FUNCTION step9_update_and_audit(
    turn: &ConversationTurn,
    memory: &MemoryStore,
    audit: &mut AuditTrail
) -> Result<AuditResult, AiAssistantError>:
    
    // 1. Generate embedding for response
    embedding = memory.db.embedding_provider().embed(&turn.response.text)?
    
    // 2. Store episode in AgenticDB
    episode = Episode {
        id: None,
        text: turn.response.text.clone(),
        embedding,
        timestamp: Utc::now(),
        tier: Tier::Hot    // new episodes always start as Hot
    }
    memory.store_episode(episode)?
    
    // 3. Add causal edge between user message and response
    memory.add_causal_edge(&turn.user_msg, &turn.response)
    
    // 4. Auto-consolidate low-quality episodes
    memory.auto_consolidate()
    
    // 5. Record RVF audit trail entry
    audit_result = audit.record(turn)?
    
    log::info("Audit: episode stored, hash={:?}", audit_result.hash)
    
    RETURN Ok(audit_result)
```

// TEST: test response embedding generated and stored  
// TEST: test new episode has Tier::Hot  
// TEST: test causal edge links user message to response  
// TEST: test auto_consolidate runs without error  
// TEST: test RVF audit entry created with valid hash  
// TEST: test audit hash chains to previous entry  

---

### STEP 10 — Return Response

```
FUNCTION step10_return(
    verified: &VerifiedResponse,
    turn: &ConversationTurn,
    session: &mut Session
) -> String:
    
    // Update session state
    session.turns.push(turn.clone())
    session.state.turn_count += 1
    session.state.total_tokens_used += turn.prompt.total_tokens
    session.state.last_coherence_energy = turn.coherence_energy
    
    RETURN verified.text.clone()
```

// TEST: test session turn count incremented  
// TEST: test session total tokens updated  
// TEST: test response text returned unchanged  
// TEST: test session stores complete ConversationTurn  

---

## 6. Error Handling Strategy

### 6.1 Custom Error Enum

```
ENUM AiAssistantError:
    // Configuration errors
    Config(String)                         // Missing or invalid env vars
    
    // Input validation
    Validation(String)                     // Empty input, too long, etc.
    
    // Embedding errors
    Embedding(String)                      // ONNX runtime failures
    
    // Memory errors
    Memory(String)                         // AgenticDB operations
    
    // Graph errors
    Graph(String)                          // Cypher query, RAG, GNN failures
    
    // Coherence errors
    Coherence(String)                      // Prime-radiant engine failures
    CoherenceCritical(String)              // Contradiction energy too high
    
    // API errors
    Auth(String)                           // 401 Unauthorized
    RateLimit(String)                      // 429 Too Many Requests
    ApiServer(String)                      // 500+ Server Error
    ApiNetwork(String)                     // Network/timeout errors
    
    // Tool errors
    ToolNotFound(String)                   // MCP tool not registered
    ServerNotFound(String)                 // MCP server not registered
    ToolExecution(String)                  // Tool execution failure
    
    // Verification errors
    Verification(String)                   // Proof validation failure
    
    // Learning errors
    Learning(String)                       // SONA trajectory/pattern failure
    
    // Audit errors
    Audit(String)                          // RVF recording failure
    
    // Serialization
    Serialization(String)                  // JSON/serde failures
    
    // Generic
    Internal(String)                       // Unexpected internal errors

IMPL Display FOR AiAssistantError:
    // Human-readable error messages for each variant

IMPL std::error::Error FOR AiAssistantError

IMPL From<reqwest::Error> FOR AiAssistantError:
    RETURN AiAssistantError::ApiNetwork(e.to_string())

IMPL From<serde_json::Error> FOR AiAssistantError:
    RETURN AiAssistantError::Serialization(e.to_string())

IMPL From<ort::Error> FOR AiAssistantError:
    RETURN AiAssistantError::Embedding(e.to_string())
```

// TEST: test each error variant has Display impl  
// TEST: test From<reqwest::Error> conversion  
// TEST: test From<serde_json::Error> conversion  
// TEST: test From<ort::Error> conversion  

### 6.2 Graceful Degradation Policy

| Subsystem | On Failure | Pipeline Behavior |
|---|---|---|
| Graph (Step 3) | `GraphContext::empty()` | Skip graph context, continue with semantic only |
| SONA Learning (Step 8) | `LearningResult::empty()` | Log warning, continue without learning |
| MCP Tools | Tools unavailable | Proceed without tool_use in Claude request |
| Coherence (Critical) | `CoherenceResult::Critical` | **HALT pipeline**, return error to user |
| Claude API (Step 6) | Network/auth/rate errors | **HALT pipeline**, return error to user |
| Audit (Step 9) | Recording failure | **HALT pipeline**, data integrity required |
| Embedding (Step 2) | ONNX failure | **HALT pipeline**, core functionality |

### 6.3 Logging Strategy

```
ENUM LogLevel:
    ERROR   // Pipeline-halting failures
    WARN    // Graceful degradation events, hallucination detection
    INFO    // Step completions, audit records
    DEBUG   // Coherence scores, token counts, timing

// Use `tracing` crate for structured logging
// Each pipeline step logs entry/exit with timing
```

// TEST: test graceful degradation for graph failure  
// TEST: test graceful degradation for SONA failure  
// TEST: test pipeline halts on critical coherence  
// TEST: test pipeline halts on API auth failure  

---

## 7. Configuration

### 7.1 Environment Variables (`.env`)

| Variable | Required | Description | Example |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | Claude API authentication key | `sk-ant-api03-...` |
| `ANTHROPIC_BASE_URL` | ✅ | Claude API base URL | `https://api.anthropic.com` |
| `CLAUDE_MODEL` | ✅ | Model identifier for Claude | `claude-sonnet-4-20250514` |
| `EMBEDDING_MODEL_PATH` | ✅ | Path to ONNX model file | `./models/paraphrase-multilingual-MiniLM-L12-v2.onnx` |

### 7.2 MCP Configuration (`mcp_servers.json`)

Optional file loaded at startup. If absent, tool_use is disabled.

```
SCHEMA mcp_servers.json:
{
    "mcpServers": {
        "<server_name>": {
            "command": "<executable>",
            "args": ["<arg1>", "<arg2>"],
            "env": {
                "<ENV_VAR>": "<value>"
            }
        }
    }
}
```

// TEST: test missing mcp_servers.json → no tools loaded, no error  
// TEST: test valid mcp_servers.json → tools discovered and loaded  
// TEST: test malformed mcp_servers.json → error with clear message  

### 7.3 System Prompts

Language-specific system prompts stored as files:

| File | Language | Fallback |
|---|---|---|
| `prompts/system_en.txt` | English | Default fallback |
| `prompts/system_tr.txt` | Turkish | — |

System prompts define the assistant's persona, capabilities, and behavioral
constraints. They MUST NOT contain secrets or API keys.

// TEST: test system_en.txt exists and is non-empty  
// TEST: test system prompts contain no hardcoded secrets  

### 7.4 Constants

```
CONST MAX_INPUT_LENGTH: usize = 10_000
CONST MAX_CONTEXT_TOKENS: usize = 4096
CONST MAX_RESPONSE_TOKENS: usize = 4096
CONST EMBEDDING_DIM: usize = 384
CONST SEMANTIC_TOP_K: usize = 10
CONST SKILL_TOP_K: usize = 5
CONST RAG_TOP_K_DOCS: usize = 5
CONST RAG_MIN_RELEVANCE: f64 = 0.7
CONST RAG_MAX_HOPS: u32 = 3
CONST REFLEX_THRESHOLD: f64 = 0.1
CONST CRITICAL_THRESHOLD: f64 = 0.8
CONST HALLUCINATION_THRESHOLD: f64 = 0.7
CONST API_TIMEOUT_SECS: u64 = 60
CONST SESSION_CONTEXT_TURNS: usize = 3
```

// TEST: test all thresholds are in valid range  
// TEST: test EMBEDDING_DIM matches model output  

---

## 8. Dependency Map

### 8.1 Crate Dependencies

| Crate | Purpose | Version Constraint |
|---|---|---|
| `ruvector` | Git submodule — AgenticDB, graph, math, solver, mincut | Local path |
| `ruvector-graph` | Cypher queries, RAG engine, GNN | Via ruvector |
| `ruvector-math` | Optimal Transport, linear algebra | Via ruvector |
| `ruvector-solver` | PageRank | Via ruvector |
| `ruvector-mincut` | Topic clustering | Via ruvector |
| `ruvector-verified` | Proof-carrying validation | Via ruvector |
| `rvf-runtime` | Audit trail runtime | Via ruvector |
| `rvf-crypto` | SHAKE-256 hashing | Via ruvector |
| `prime-radiant` | Coherence engine | Via ruvector |
| `sona` | SONA learning trajectories | Via ruvector |
| `reqwest` | HTTP client for Claude API | Latest, with `json` feature |
| `tokio` | Async runtime | Latest, with `full` feature |
| `serde` | Serialization | Latest, with `derive` feature |
| `serde_json` | JSON parsing | Latest |
| `dotenvy` | .env file loading | Latest |
| `ort` | ONNX runtime | Latest |
| `tokenizers` | HuggingFace tokenizer | Latest |
| `chrono` | DateTime handling | Latest, with `serde` feature |
| `tracing` | Structured logging | Latest |
| `tracing-subscriber` | Log output | Latest |
| `whatlang` | Language detection | Latest |
| `thiserror` | Error derive macros | Latest |

### 8.2 Module Dependency Graph

```
main.rs
  ├── config.rs
  ├── pipeline.rs
  │     ├── types.rs
  │     ├── language.rs
  │     ├── embedding.rs
  │     ├── memory.rs
  │     │     └── embedding.rs
  │     ├── graph_context.rs
  │     ├── coherence.rs
  │     ├── claude_api.rs
  │     │     └── mcp_tools.rs
  │     ├── verification.rs
  │     ├── learning.rs
  │     └── audit.rs
  └── types.rs
```

---

## 9. Performance Considerations

| Concern | Strategy |
|---|---|
| Embedding latency | ONNX runtime with optimized model; cache recent embeddings |
| Memory search | AgenticDB vector index; top_k limits retrieval scope |
| Graph queries | PageRank O(√n) via sqrt iterations; MinCut prunes irrelevant nodes |
| Token estimation | Lightweight character-based estimate (~4 chars/token) |
| Coherence check | Three-lane system: Reflex (<1ms), Retrieval (~10ms), Critical (halt) |
| API latency | 60s timeout; no retry on 429 (let user retry) |
| Tiered compression | Hot/Warm/Cold quantization reduces memory footprint for old episodes |
| Session context | Limited to last 3 turns to control prompt size |
| Audit trail | Append-only with hash chaining — O(1) per write |

// TEST: test embedding cached for same input within session  
// TEST: test prompt stays within 4096 token budget  
// TEST: test reflex coherence lane completes in < 5ms  

---

## 10. Security Requirements

| Requirement | Implementation |
|---|---|
| No hardcoded secrets | All secrets from `.env` via `dotenvy` |
| API key protection | Key read once at startup, never logged or serialized |
| Input sanitization | Step 1 removes null bytes, control characters |
| Output verification | Step 7 hallucination detection + proof-carrying validation |
| Audit integrity | SHAKE-256 hash chain — tamper-evident append-only log |
| Witness chain | Cognitive container records every response with cryptographic proof |
| Tool execution | Tools managed by Rust — Claude cannot execute arbitrary code |
| Token limits | Hard cap at 4096 tokens prevents prompt injection via context overflow |

// TEST: test API key not present in any log output  
// TEST: test API key not serialized in ConversationTurn  
// TEST: test input sanitization removes dangerous characters  
// TEST: test witness chain detects tampering  

---

## Appendix A: File Size Budget

| File | Estimated Lines | Budget (max 500) |
|---|---|---|
| `src/main.rs` | ~80 | ✅ |
| `src/config.rs` | ~60 | ✅ |
| `src/embedding.rs` | ~100 | ✅ |
| `src/memory.rs` | ~120 | ✅ |
| `src/graph_context.rs` | ~130 | ✅ |
| `src/coherence.rs` | ~80 | ✅ |
| `src/claude_api.rs` | ~150 | ✅ |
| `src/mcp_tools.rs` | ~120 | ✅ |
| `src/verification.rs` | ~100 | ✅ |
| `src/learning.rs` | ~90 | ✅ |
| `src/audit.rs` | ~80 | ✅ |
| `src/pipeline.rs` | ~200 | ✅ |
| `src/types.rs` | ~150 | ✅ |
| `src/language.rs` | ~40 | ✅ |
| **Total** | **~1500** | — |

---

## Appendix B: TDD Anchor Summary

| Module | Test Count | Key Behaviors |
|---|---|---|
| `config` | 4 | Env loading, validation |
| `embedding` | 6 | Dimensions, normalization, multilingual |
| `memory` | 7 | ONNX provider, tiered compression, causal edges |
| `graph_context` | 8 | Cypher, RAG, GNN, PageRank, MinCut |
| `coherence` | 6 | Energy thresholds, hallucination detection |
| `claude_api` | 9 | HTTP calls, errors, tool handling |
| `mcp_tools` | 6 | Loading, execution, missing file |
| `verification` | 6 | Proofs, witness chain, integrity |
| `learning` | 6 | Trajectories, quality, patterns |
| `audit` | 5 | Hash chain, genesis, tampering |
| `pipeline` | 6 | Full flow, degradation, coherence halts |
| `types` | 10 | Construction, serialization, defaults |
| `language` | 5 | Detection, confidence, fallback |
| `step1` | 7 | Validation, sanitization |
| `step2` | 7 | Embedding, retrieval, ranking |
| `step3` | 8 | Graph queries, RAG, clustering |
| `step4` | 8 | Priority merge, token limits, prompts |
| `step5` | 5 | Coherence lanes |
| `step6` | 9 | API calls, tool loops |
| `step7` | 6 | Hallucination, proofs, witness |
| `step8` | 5 | Trajectories, no LoRA |
| `step9` | 6 | Episode storage, audit chain |
| `step10` | 4 | Session update, response |
| **Total** | **~141** | — |

---

*End of specification. This document serves as the authoritative reference for
implementing the AI Assistant. All modules, types, pipeline steps, and TDD anchors
defined here must be reflected in the final implementation.*
