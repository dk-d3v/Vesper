# RuVector API Notes

> Findings from Phase 4a examination of `ruvector/` submodule.  
> **Do not modify any files inside `ruvector/`.**  
> All paths are relative to the workspace root `c:/Users/admin/ai-assistant/`.

---

## Table of Contents

1. [Cargo.toml Corrections](#1-cargotoml-corrections)
2. [ruvector-core — HNSW + AgenticDB](#2-ruvector-core--hnsw--agenticdb)
3. [prime-radiant — Coherence Engine](#3-prime-radiant--coherence-engine)
4. [ruvector-graph — Graph DB + RAG](#4-ruvector-graph--graph-db--rag)
5. [sona — Self-Optimising Learning](#5-sona--self-optimising-learning)
6. [rvf-runtime — Audit Governance](#6-rvf-runtime--audit-governance)
7. [rvf-crypto — Cryptographic Witness](#7-rvf-crypto--cryptographic-witness)
8. [ruvector-verified — Formal Proofs](#8-ruvector-verified--formal-proofs)
9. [ruvector-cognitive-container — Sealed Container](#9-ruvector-cognitive-container--sealed-container)
10. [ruvector-temporal-tensor — Tiered Compression](#10-ruvector-temporal-tensor--tiered-compression)
11. [ruvector-solver — Sparse Linear Solver](#11-ruvector-solver--sparse-linear-solver)
12. [ruvector-math — Optimal Transport + Topology](#12-ruvector-math--optimal-transport--topology)
13. [ruvector-mincut — Graph Min-Cut](#13-ruvector-mincut--graph-min-cut)
14. [onnx-embeddings (example) — Local Inference](#14-onnx-embeddings-example--local-inference)
15. [OSpipe (example) — Ingestion Pipeline Architecture](#15-ospipe-example--ingestion-pipeline-architecture)
16. [Graph Examples — Usage Patterns](#16-graph-examples--usage-patterns)
17. [Implementation Mapping](#17-implementation-mapping)

---

## 1. Cargo.toml Corrections

**Bug found and fixed**: `rvf-runtime` and `rvf-crypto` are nested inside an `rvf/` subdirectory.

```toml
# WRONG (original):
rvf-runtime = { path = "./ruvector/crates/rvf-runtime" }
rvf-crypto  = { path = "./ruvector/crates/rvf-crypto" }

# CORRECT (fixed in Cargo.toml):
rvf-runtime = { path = "./ruvector/crates/rvf/rvf-runtime" }
rvf-crypto  = { path = "./ruvector/crates/rvf/rvf-crypto" }
```

**Note on `onnx-embeddings` and `ospipe`**: These are **example applications** at
`ruvector/examples/onnx-embeddings/` and `ruvector/examples/OSpipe/`, NOT library crates.
They cannot be added as `[dependencies]`. For ONNX inference, use the `ort` crate directly
or the `ruvector-core` `EmbeddingProvider` trait with `ApiEmbedding`.

**AgenticDB feature flag** required:
```toml
ruvector-core = { path = "./ruvector/crates/ruvector-core", features = ["storage"] }
```

---

## 2. ruvector-core — HNSW + AgenticDB

**Path**: `ruvector/crates/ruvector-core/`

### 2.1 EmbeddingProvider Trait

```rust
use ruvector_core::EmbeddingProvider;

pub trait EmbeddingProvider: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn dimensions(&self) -> usize;
    fn name(&self) -> &str;
}

pub type BoxedEmbeddingProvider = Arc<dyn EmbeddingProvider>;
```

**Built-in implementations**:

```rust
use ruvector_core::{HashEmbedding, ApiEmbedding};

// Hash-based placeholder (not semantic — for testing only)
let emb = HashEmbedding::new(384);

// OpenAI API embedding (real semantic embeddings)
// Dimension auto-selected: text-embedding-3-small = 1536, text-embedding-3-large = 3072, ada-002 = 1536
let emb = ApiEmbedding::openai("sk-...", "text-embedding-3-small");

// Cohere API embedding
let emb = ApiEmbedding::cohere("api-key", "embed-english-v3.0");

// Voyage AI embedding
let emb = ApiEmbedding::voyage("api-key", "voyage-2");

// Custom endpoint
let emb = ApiEmbedding::new(api_key, endpoint, model, dimensions);
```

**Candle embedding** (feature-gated, GPU-capable):
```rust
use ruvector_core::candle::CandleEmbedding;
let emb = CandleEmbedding::from_pretrained("sentence-transformers/all-MiniLM-L6-v2", false)?;
```

### 2.2 AgenticDB (requires `storage` feature)

```rust
use ruvector_core::agenticdb::{AgenticDB, DbOptions, ReflexionEpisode, Skill, CausalEdge, LearningSession, Experience};

// Construction
let db = AgenticDB::new(DbOptions::default())?;
let db = AgenticDB::with_dimensions(384)?;
let db = AgenticDB::with_embedding_provider(provider, options)?;
```

**Schema (5 tables in embedded SQLite + HNSW)**:
- `vectors_table` — base HNSW vector store
- `reflexion_episodes` — `ReflexionEpisode { id, embedding, content, outcome, timestamp, tags }`
- `skills_library` — `Skill { id, name, description, embedding, success_rate, usage_count }`
- `causal_edges` — `CausalEdge { id, cause_id, effect_id, strength, evidence_count }`
- `learning_sessions` — `LearningSession { id, start_time, end_time, experiences, avg_reward }`

**Core operations**:

```rust
// Store a reflexion episode
db.store_episode(embedding: Vec<f32>, content: &str, outcome: &str, tags: &[&str])?;

// Retrieve similar episodes (k-NN)
let episodes: Vec<ReflexionEpisode> = db.retrieve_similar_episodes(&embedding, k)?;

// Create a skill
db.create_skill(name: &str, description: &str, embedding: Vec<f32>)?;

// Search skills by description
let skills: Vec<Skill> = db.search_skills("query description", k)?;

// Auto-consolidate skills (merges duplicates, updates success rates)
db.auto_consolidate(threshold: f32)?;

// Add causal edge between two stored episodes
db.add_causal_edge(cause_id: &str, effect_id: &str, strength: f32)?;

// Utility-weighted query (combines similarity + reward signal)
db.query_with_utility(embedding: &[f32], k: usize, utility_weight: f32)?;

// Session management
let session_id = db.start_session(metadata: &str)?;
db.add_experience(session_id: &str, experience: Experience)?;
db.predict_with_confidence(session_id: &str, state: Vec<f32>)?; // -> Prediction { value, confidence }
```

**Policy Memory Store** (RL-style):
```rust
use ruvector_core::agenticdb::PolicyMemoryStore;
let store = PolicyMemoryStore::new(&db);
store.store_policy(state_embedding: Vec<f32>, action: &str, q_value: f64)?;
store.get_best_action(state_embedding: &[f32], k: usize)? // -> Option<String>
store.update_q_value(policy_id: &str, new_q_value: f64)?;
```

**Session State Index** (conversation context with TTL):
```rust
use ruvector_core::agenticdb::SessionStateIndex;
let idx = SessionStateIndex::new(&db, session_id, ttl_seconds: 3600);
idx.add_turn(turn_number: usize, role: &str, content: &str)?;
let relevant: Vec<SessionTurn> = idx.find_relevant_turns(query: &str, k: usize)?;
let context: Vec<SessionTurn> = idx.get_session_context()?;
idx.cleanup_expired()?; // prune TTL-expired turns
```

**Witness Log** (append-only audit trail inside AgenticDB):
```rust
use ruvector_core::agenticdb::WitnessLog;
let log = WitnessLog::new(&db);
log.append(agent_id: &str, action_type: &str, details: &str)?; // returns entry_id
let entries: Vec<WitnessEntry> = log.search(query: &str, k: usize)?;
log.verify_chain()?; // -> bool, checks hash chain integrity
```

---

## 3. prime-radiant — Coherence Engine

**Path**: `ruvector/crates/prime-radiant/`

### 3.1 Core Types

```rust
use prime_radiant::{
    CoherenceEngine, SheafGraph, SheafNode, SheafEdge, RestrictionMap,
    CoherenceGate, GovernancePolicy, GateDecision, ComputeLane, CoherenceEnergy,
};
```

### 3.2 SheafGraph Construction

```rust
// Create a sheaf node with a feature vector (representing a system component)
let node = SheafNode::new(vec![0.5_f32; 256]);  // 256-dim feature vector

// Create an edge with restriction maps (rho_forward, rho_backward) and weight
let edge = SheafEdge::new(node_u_id, node_v_id, rho_forward, rho_backward, weight: f32);

let mut graph = SheafGraph::new();
let node_id = graph.add_node(node);
graph.add_edge(edge);
```

### 3.3 CoherenceEngine

```rust
let engine = CoherenceEngine::new();

// Compute coherence energy of the sheaf graph
// Lower energy = more coherent (consistent) system state
let energy: CoherenceEnergy = engine.compute_energy(&graph);
// energy.total: f32 — Sheaf Laplacian energy
// energy.harmonic: f32 — harmonic component
// energy.coboundary: f32 — coboundary component
```

### 3.4 CoherenceGate — Routing Decision

```rust
use prime_radiant::{CoherenceGate, GovernancePolicy, GateDecision, ComputeLane};

let policy = GovernancePolicy::default(); // configurable thresholds
let gate = CoherenceGate::new(policy);

let decision: GateDecision = gate.evaluate(&action_description, &energy);
// decision.lane: ComputeLane
// decision.confidence: f32
// decision.explanation: String

// ComputeLane variants:
match decision.lane {
    ComputeLane::Reflex    => { /* fast path, no retrieval */ }
    ComputeLane::Retrieval => { /* RAG lookup needed */ }
    ComputeLane::Heavy     => { /* full LLM reasoning */ }
    ComputeLane::Human     => { /* escalate to human */ }
}
```

### 3.5 Additional Exports (advanced use)

```rust
use prime_radiant::{
    // Cohomology / spectral
    CohomologyGroup, SpectralGap, PersistentCohomology,
    // Governance
    GovernancePolicy, PolicyAction, PolicyEngine,
    // Hyperbolic geometry
    HyperbolicSpace, PoincareEmbedding,
    // Distributed coherence
    DistributedCoherence, NodeCoherence,
    // SIMD-accelerated operations
    SimdCoherence,
};
```

---

## 4. ruvector-graph — Graph DB + RAG

**Path**: `ruvector/crates/ruvector-graph/`

### 4.1 Exports

```rust
use ruvector_graph::{
    GraphDB,
    RagEngine, HybridIndex, GnnConfig, RagConfig,
    SemanticSearch, VectorCypherParser, EmbeddingConfig,
    Transaction, TransactionManager,
};
```

### 4.2 GraphDB

```rust
// Graph database with Cypher query support
let db = GraphDB::open("./data/graph.db")?;

// Execute Cypher queries (Neo4j-compatible subset)
// CREATE, MATCH, WHERE, RETURN, ORDER BY, LIMIT, SET, MERGE
// Relationship patterns: (a)-[:TYPE]->(b), variable-length: *1..3
// Aggregations: count(), avg(), min(), max()
// shortestPath()
```

**Cypher patterns supported** (from examples):
```cypher
-- Create node
CREATE (n:Person {name: 'Alice', age: 30}) RETURN n

-- Pattern match with filter
MATCH (p:Person) WHERE p.age > 25 RETURN p.name ORDER BY p.age DESC

-- Create relationship
MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
CREATE (a)-[r:KNOWS {since: 2023}]->(b) RETURN r

-- Variable-length traversal
MATCH (start)-[:KNOWS*1..3]->(end) RETURN end

-- Shortest path
MATCH path = shortestPath((a)-[:KNOWS*]-(b)) RETURN path

-- MERGE (upsert)
MERGE (p:Person {email: 'alice@example.com'})
ON CREATE SET p.created_at = timestamp()
ON MATCH SET p.last_seen = timestamp()
```

### 4.3 RagEngine — Hybrid Vector+Graph Search

```rust
let rag_config = RagConfig {
    top_k: 5,
    similarity_threshold: 0.7,
    // ... other options
};
let embedding_config = EmbeddingConfig { /* model, dimensions */ };
let rag = RagEngine::new(rag_config, embedding_config)?;

// HybridIndex: combined HNSW + graph index
let index = HybridIndex::new()?;

// VectorCypherParser: parse hybrid queries combining vector similarity + Cypher
let parser = VectorCypherParser::new();

// SemanticSearch: pure vector semantic search
let search = SemanticSearch::new(embedding_config)?;
```

### 4.4 Transaction Support

```rust
let tx_manager = TransactionManager::new(&db);
let tx = tx_manager.begin()?;
// ... operations on tx
tx.commit()?;
// or tx.rollback()?;
```

### 4.5 Distributed Extensions

```rust
use ruvector_graph::distributed::{/* cluster types */};
// Cluster support for sharded graph databases
```

---

## 5. sona — Self-Optimising Learning

**Path**: `ruvector/crates/sona/`

### 5.1 Exports

```rust
use sona::{
    SonaEngine, SonaConfig,
    TrajectoryBuffer, TrajectoryBuilder, TrajectoryIdGen,
    PatternConfig, ReasoningBank,
    // training exports
};
```

### 5.2 SonaEngine Construction

```rust
let config = SonaConfig {
    hidden_dim: 256,
    embedding_dim: 256,
    ..Default::default()
};
let engine = SonaEngine::new(config);
```

### 5.3 Trajectory Learning (Reflexion loop)

```rust
// Begin a trajectory with initial state embedding
let mut builder: TrajectoryBuilder = engine.begin_trajectory(vec![0.1_f32; 256]);

// Add steps: (state_embedding, actions, reward)
builder.add_step(
    vec![0.5_f32; 256],   // next state embedding
    vec![],                // action embeddings (can be empty)
    0.8_f32,               // step reward
);

// End trajectory with final reward; triggers learning
engine.end_trajectory(builder, 0.85_f32);
```

### 5.4 Micro-LoRA Adaptation

```rust
// Apply learned micro-LoRA adaptation to a vector
// Modifies output in-place based on accumulated trajectory experience
let input = vec![0.1_f32; 256];
let mut output = vec![0.0_f32; 256];
engine.apply_micro_lora(&input, &mut output);
```

### 5.5 ReasoningBank

```rust
use sona::{ReasoningBank, PatternConfig};

let config = PatternConfig {
    // pattern matching configuration
    ..Default::default()
};
let bank = ReasoningBank::new(config);
// Stores and retrieves reasoning patterns for few-shot learning
```

### 5.6 TrajectoryBuffer

```rust
use sona::TrajectoryBuffer;
// Circular buffer for trajectory replay
// Used internally by SonaEngine for EWC++ and replay
```

---

## 6. rvf-runtime — Audit Governance

**Path**: `ruvector/crates/rvf/rvf-runtime/`  
⚠️ Note nested under `rvf/` — Cargo.toml path must be `./ruvector/crates/rvf/rvf-runtime`.

### 6.1 Exports

```rust
use rvf_runtime::{
    RvfStore,
    GovernancePolicy, ParsedWitness, WitnessBuilder, WitnessError,
    CowEngine, WitnessEvent,
    // adversarial detection
    // seed crypto
};
```

### 6.2 RvfStore — Persistent Witness Store

```rust
let store = RvfStore::open("./data/audit.rvf")?;
// or in-memory:
let store = RvfStore::new_in_memory()?;
```

### 6.3 WitnessBuilder — Build Governance Records

```rust
let policy = GovernancePolicy::default();  // configurable thresholds/rules
let builder = WitnessBuilder::new(policy);

// Build a witness record for an agent action
let witness = builder
    .agent("agent-id")
    .action("tool_call")
    .data(serde_json::json!({ "tool": "search", "query": "..." }))
    .build()?;

let parsed: ParsedWitness = store.append(witness)?;
// parsed.id: String — unique witness ID
// parsed.timestamp: u64
// parsed.hash: [u8; 32]
```

### 6.4 CowEngine — Copy-on-Write Mutation Tracking

```rust
let cow = CowEngine::new(&store);
// Tracks mutations with immutable history
// WitnessEvent: emitted on each state transition
```

---

## 7. rvf-crypto — Cryptographic Witness

**Path**: `ruvector/crates/rvf/rvf-crypto/`  
⚠️ Note nested under `rvf/` — Cargo.toml path must be `./ruvector/crates/rvf/rvf-crypto`.

### 7.1 Exports

```rust
use rvf_crypto::{
    // attestation
    WitnessEntry, create_witness_chain, verify_witness_chain,
    // lineage
    lineage_witness_entry, verify_lineage_chain,
    // hash
    shake256_128, shake256_256,
};
```

### 7.2 Cryptographic Witness Chain (Ed25519)

```rust
use rvf_crypto::{WitnessEntry, create_witness_chain, verify_witness_chain};

// Create a chain of cryptographically linked witness entries
// Each entry contains: content_hash, previous_hash, Ed25519 signature
let entries: Vec<WitnessEntry> = create_witness_chain(
    events: &[(&str, &[u8])],  // (action_type, content_bytes) pairs
    signing_key: &ed25519_key,
)?;

// Verify the entire chain is intact (no tampering)
let is_valid: bool = verify_witness_chain(&entries, &verifying_key)?;
```

### 7.3 Lineage Tracking

```rust
// Track lineage of a data artifact through transformations
let entry = lineage_witness_entry(parent_hash: &[u8], transform: &str, data: &[u8])?;
let is_valid: bool = verify_lineage_chain(&entries)?;
```

### 7.4 Hash Utilities

```rust
// SHAKE-256 with 128-bit output (16 bytes)
let hash: [u8; 16] = shake256_128(data: &[u8]);

// SHAKE-256 with 256-bit output (32 bytes)
let hash: [u8; 32] = shake256_256(data: &[u8]);
```

---

## 8. ruvector-verified — Formal Proofs

**Path**: `ruvector/crates/ruvector-verified/`

### 8.1 Exports

```rust
use ruvector_verified::{
    ProofEnvironment, ProofStats,
    VerifiedOp,
    ProofAttestation,  // from proof_store
    VerifiedStage,     // from pipeline
};
```

### 8.2 ProofEnvironment

```rust
// Pre-loads RuVector type declarations and proof symbols
let mut env = ProofEnvironment::new();
// env has built-in symbols for common types

// Allocate a fresh term ID
let term_id: u32 = env.alloc_term();

// Require a named symbol (returns index if found)
let idx: usize = env.require_symbol("VectorStore")?;

// Cache lookup by content-addressed hash
let cached: Option<u32> = env.cache_lookup(hash_key: u64);

// Reset environment (clears cache but keeps builtins)
env.reset();

// Stats
let stats: &ProofStats = env.stats();
// stats.cache_hits, stats.cache_misses, stats.terms_allocated
```

### 8.3 VerifiedOp

```rust
// A value with an associated proof ID
let op: VerifiedOp<Vec<f32>> = VerifiedOp { 
    value: embedding_vec,
    proof_id: env.alloc_term(),
};
// T must implement Copy (or Clone in extended usage)
```

### 8.4 VerifiedStage (pipeline)

```rust
// Wraps a pipeline stage with formal verification gates
use ruvector_verified::VerifiedStage;
// Used to create verification-gated processing pipelines
```

---

## 9. ruvector-cognitive-container — Sealed Container

**Path**: `ruvector/crates/ruvector-cognitive-container/`

### 9.1 Exports

```rust
use ruvector_cognitive_container::{
    CognitiveContainer, ComponentMask, ContainerConfig, ContainerSnapshot, Delta, TickResult,
    ContainerEpochBudget, EpochController, Phase,
    ContainerError, Result,
    Arena, MemoryConfig, MemorySlab,
    CoherenceDecision, ContainerWitnessReceipt, VerificationResult, WitnessChain,
};
```

### 9.2 CognitiveContainer

```rust
let config = ContainerConfig {
    // component mask: bitflags for which components to enable
    components: ComponentMask::all(),
    // memory configuration
    memory: MemoryConfig::default(),
    // epoch budget
    epoch_budget: ContainerEpochBudget::default(),
};

let mut container = CognitiveContainer::new(config)?;

// Tick the container (one reasoning step)
// Processes: graph ingest → min-cut → spectral analysis → evidence accumulation
let result: TickResult = container.tick(input_delta: Delta)?;
// result.deltas: Vec<Delta> — state changes this tick
// result.witness: ContainerWitnessReceipt — cryptographic proof of tick
// result.coherence: CoherenceDecision — pass/fail/escalate decision

// Snapshot current state
let snapshot: ContainerSnapshot = container.snapshot()?;
```

### 9.3 WitnessChain

```rust
// Tamper-evident chain linking every epoch to its predecessor
let chain: WitnessChain = container.witness_chain();
let receipt: &ContainerWitnessReceipt = chain.latest();
// receipt.epoch_hash: [u8; 32]
// receipt.parent_hash: [u8; 32]
// receipt.timestamp: u64

let result: VerificationResult = chain.verify()?;
```

### 9.4 EpochController

```rust
let controller = EpochController::new(budget: ContainerEpochBudget);
// Manages tick budget: CPU time, memory, iteration limits
// phase: Phase::{ Ingest | Process | Emit | Idle }
```

### 9.5 Memory Management

```rust
// Arena allocator for zero-GC operation
let arena = Arena::new(MemoryConfig { capacity: 1024 * 1024 });
let slab: MemorySlab = arena.alloc(size: usize)?;
```

---

## 10. ruvector-temporal-tensor — Tiered Compression

**Path**: `ruvector/crates/ruvector-temporal-tensor/`

### 10.1 Exports

```rust
use ruvector_temporal_tensor::{TemporalTensorCompressor, TierPolicy};
use ruvector_temporal_tensor::segment;
```

### 10.2 Compression Tiers

| Tier | Bits | Ratio vs f32 | Access Pattern |
|------|------|-------------|----------------|
| Hot  | 8    | ~4.0×       | Frequently accessed |
| Warm | 7    | ~4.57×      | Moderately accessed |
| Warm | 5    | ~6.4×       | Aggressively compressed |
| Cold | 3    | ~10.67×     | Rarely accessed |

### 10.3 Usage

```rust
// Create compressor for n-element tensors
// tensor_id: u64 (for access-pattern tracking)
let mut comp = TemporalTensorCompressor::new(TierPolicy::default(), n_elements: 128, tensor_id: 0);

// Mark access count (determines tier: high count = hot = 8-bit)
comp.set_access(access_count: 100, tensor_id: 0);

// Push frames (compressed segments emitted at boundaries)
let frame: Vec<f32> = vec![1.0; 128];
let mut segment: Vec<u8> = Vec::new();
comp.push_frame(&frame, timestamp: 1, &mut segment);

// Force-emit current segment
comp.flush(&mut segment);

// Decode segment back to f32
let mut decoded: Vec<f32> = Vec::new();
segment::decode(&segment, &mut decoded);
assert_eq!(decoded.len(), 128);

// Random-access single frame decode
let frame_0: Option<Vec<f32>> = segment::decode_single_frame(&segment, frame_index: 0);

// Compression ratio inspection
let ratio: f32 = segment::compression_ratio(&segment); // > 1.0
```

### 10.4 Integration Modules

```rust
// AgentDB integration (store compressed tensors in AgenticDB)
use ruvector_temporal_tensor::agentdb;

// Coherence integration (compressed coherence states)
use ruvector_temporal_tensor::coherence;

// Core trait for custom tensor types
use ruvector_temporal_tensor::core_trait::TemporalTensor;
```

---

## 11. ruvector-solver — Sparse Linear Solver

**Path**: `ruvector/crates/ruvector-solver/`

### 11.1 Exports

```rust
use ruvector_solver::{
    types::{ComputeBudget, CsrMatrix},
    traits::SolverEngine,
    router,           // solver selection router
};

// Feature-gated solvers:
#[cfg(feature = "neumann")]
use ruvector_solver::neumann::NeumannSolver;

#[cfg(feature = "cg")]
use ruvector_solver::cg;              // Conjugate Gradient

#[cfg(feature = "forward-push")]
use ruvector_solver::forward_push;

#[cfg(feature = "backward-push")]
use ruvector_solver::backward_push;
```

### 11.2 NeumannSolver (primary solver)

Solves `Ax = b` using the Neumann series: `x = Σ (I-A)^k b`

```rust
use ruvector_solver::types::{ComputeBudget, CsrMatrix};
use ruvector_solver::neumann::NeumannSolver;
use ruvector_solver::traits::SolverEngine;

// Build CSR matrix from COO format (row, col, value)
let matrix = CsrMatrix::<f32>::from_coo(n_rows: 3, n_cols: 3, entries: vec![
    (0, 0, 2.0_f32), (0, 1, -0.5_f32),
    (1, 0, -0.5_f32), (1, 1, 2.0_f32), (1, 2, -0.5_f32),
    (2, 1, -0.5_f32), (2, 2, 2.0_f32),
]);

let rhs = vec![1.0_f32, 0.0, 1.0];

// tolerance: f32, max_iterations: usize
let solver = NeumannSolver::new(tolerance: 1e-6, max_iterations: 500);
let result = solver.solve(&matrix, &rhs)?;

assert!(result.residual_norm < 1e-4);
// result.solution: Vec<f32>
// result.iterations: usize
// result.converged: bool
```

### 11.3 SolverEngine Trait

```rust
pub trait SolverEngine {
    fn solve(&self, matrix: &CsrMatrix<f32>, rhs: &[f32]) -> Result<SolveResult>;
}
```

---

## 12. ruvector-math — Optimal Transport + Topology

**Path**: `ruvector/crates/ruvector-math/`

### 12.1 Prelude

```rust
use ruvector_math::prelude::*;
```

### 12.2 Key Capabilities

```rust
// Sliced Wasserstein Distance (optimal transport)
use ruvector_math::SlicedWasserstein;
let dist = SlicedWasserstein::new(n_projections: 100);
let distance: f32 = dist.compute(&distribution_a, &distribution_b)?;

// Sinkhorn Solver (entropic regularized OT)
use ruvector_math::SinkhornSolver;
let solver = SinkhornSolver::new(epsilon: 0.01, max_iter: 1000);
let transport_plan = solver.solve(&cost_matrix, &weights_a, &weights_b)?;

// Persistent Homology (topological data analysis)
use ruvector_math::PersistentHomology;
let ph = PersistentHomology::new();
let barcodes = ph.compute(&point_cloud)?;

// Product Manifolds (Riemannian geometry)
use ruvector_math::ProductManifold;
```

---

## 13. ruvector-mincut — Graph Min-Cut

**Path**: `ruvector/crates/ruvector-mincut/`

### 13.1 Prelude

```rust
use ruvector_mincut::prelude::*;
```

### 13.2 MinCutBuilder (primary API)

```rust
use ruvector_mincut::{MinCutBuilder, DynamicMinCut};

// Build a min-cut problem
let result = MinCutBuilder::new()
    .add_edge(u: usize, v: usize, weight: f32)
    .add_edge(0, 1, 2.0)
    .add_edge(1, 2, 3.0)
    .approximate(true)     // use subpolynomial-time approximation
    .build()
    .solve()?;

// result.cut_value: f32
// result.partition_a: Vec<usize>
// result.partition_b: Vec<usize>

// Dynamic min-cut (streaming updates)
let mut dynamic = DynamicMinCut::new(n_nodes: 100);
dynamic.add_edge(u, v, weight)?;
dynamic.remove_edge(u, v)?;
let current_cut = dynamic.min_cut()?;
```

### 13.3 Available Algorithms

```rust
// Subpolynomial-time (fastest for large graphs)
use ruvector_mincut::subpolynomial::SubpolynomialMinCut;

// Parallel (multi-threaded)
use ruvector_mincut::parallel::ParallelMinCut;

// Exact (small graphs)
// use ruvector_mincut::localkcut::LocalKCut;

// Spectral / SNN
use ruvector_mincut::snn::SnnMinCut;

// Canonical (deterministic)
use ruvector_mincut::canonical::CanonicalMinCut;
```

---

## 14. onnx-embeddings (example) — Local Inference

**Path**: `ruvector/examples/onnx-embeddings/`  
⚠️ This is an **example application**, not a library crate. Cannot be used as `[dependency]`.

### 14.1 Architecture

```text
EmbedderBuilder -> EmbedderConfig -> Embedder (ONNX Runtime via `ort` crate)
                                         |
                      RuVectorBuilder -> RagPipeline -> RuVectorEmbeddings
```

### 14.2 PretrainedModel Variants and Dimensions

```rust
pub enum PretrainedModel {
    AllMiniLmL6V2,      // 384 dims, "sentence-transformers/all-MiniLM-L6-v2"
    AllMiniLmL12V2,     // 384 dims, "sentence-transformers/all-MiniLM-L12-v2"
    AllMpnetBaseV2,     // 768 dims, "sentence-transformers/all-mpnet-base-v2"
    BgeSmallEnV15,      // 384 dims, "BAAI/bge-small-en-v1.5"
    BgeLargeEnV15,      // 1024 dims, "BAAI/bge-large-en-v1.5"
    // ... others
}
```

### 14.3 Embedder API (async initialization)

```rust
use onnx_embeddings::{Embedder, EmbedderBuilder, EmbeddingOutput, EmbedderConfig, ModelSource, PoolingStrategy};

// Builder pattern
let config = EmbedderBuilder::new()
    .pretrained(PretrainedModel::AllMiniLmL12V2)
    .pooling(PoolingStrategy::MeanPooling)
    .normalize(true)
    .batch_size(32)
    .max_length(512)
    .build();

// Async initialization (downloads tokenizer if needed)
let mut embedder = Embedder::new(config).await?;

// Embed single text
let vec: Vec<f32> = embedder.embed_one("Hello world")?;

// Embed batch
let output: EmbeddingOutput = embedder.embed(&["text 1", "text 2", "text 3"])?;
// output.embeddings: Vec<Vec<f32>>
// output.dimensions: usize
// output.token_counts: Vec<usize>

// Cosine similarity between two texts
let sim: f32 = embedder.similarity("text a", "text b")?;

// Most similar from candidates
let ranked: Vec<(usize, f32)> = embedder.most_similar("query", &candidates, top_k: 5)?;

// K-means clustering
let labels: Vec<usize> = embedder.cluster(&texts, n_clusters: 3)?;

// GPU check
let has_gpu: bool = embedder.has_gpu();
```

### 14.4 RagPipeline Integration

```rust
use onnx_embeddings::{RagPipeline, RuVectorBuilder, RuVectorEmbeddings};

let pipeline = RuVectorBuilder::new()
    .with_embedder(embedder)
    .build_rag()?;
// pipeline wraps Embedder + ruvector-core HNSW for document storage/search
```

### 14.5 Recommendation for ai-assistant

Since `onnx-embeddings` cannot be a direct dependency:
- **Option A**: Use `ruvector-core::ApiEmbedding::openai(...)` — no local model needed
- **Option B**: Add `ort` crate directly + copy the embedding logic from the example
- **Option C**: `ruvector-core::candle::CandleEmbedding::from_pretrained(...)` — if `candle` feature enabled

---

## 15. OSpipe (example) — Ingestion Pipeline Architecture

**Path**: `ruvector/examples/OSpipe/`  
⚠️ Example application. Reference architecture only.

### 15.1 Pipeline Architecture

```text
Screenpipe → Capture → Safety Gate → Dedup → Embed → VectorStore (HNSW)
                                                            |
                                     Search Router ←────────+
                                     (Semantic / Keyword / Hybrid)
```

### 15.2 Module Pattern

```
capture/    - CapturedFrame { ocr_text, transcription, ui_events, timestamp }
storage/    - HNSW-backed store + EmbeddingEngine
search/     - QueryRouter: semantic | keyword | hybrid
pipeline/   - IngestPipeline { dedup_threshold, batch_size }
safety/     - PiiDetector, ContentRedactor
config/     - AppConfig (from env)
error/      - unified Error type
graph/      - graph-enriched memory
learning/   - SONA integration for session learning
server/     - REST API (non-WASM only)
wasm/       - WASM-compatible API surface
```

### 15.3 Key Patterns to Adopt

```rust
// Safety gate: PII redaction before storage
let safe_content = safety::redact_pii(&raw_content);

// Deduplication before embedding (cosine threshold)
if !pipeline.is_duplicate(&embedding, threshold: 0.95) {
    storage.insert(embedding, metadata)?;
}

// Hybrid search routing
match search_router.route(&query) {
    SearchMode::Semantic  => store.semantic_search(&query_embedding, k),
    SearchMode::Keyword   => store.keyword_search(&query_text, k),
    SearchMode::Hybrid    => store.hybrid_search(&query_embedding, &query_text, k),
}
```

---

## 16. Graph Examples — Usage Patterns

**Path**: `ruvector/examples/graph/`

> Note: All graph examples are **template stubs** — the actual `ruvector-graph` public API
> for node/relationship CRUD is not yet exposed. Use `GraphDB::execute_cypher()` as primary interface.

### 16.1 Cypher-First Pattern

```rust
// All graph operations go through Cypher strings
let result = db.execute_cypher(r#"
    CREATE (n:Memory {content: $content, embedding: $embedding})
    RETURN n.id
"#, params)?;
```

### 16.2 Hybrid Query Pattern (planned API)

```rust
// Combine Cypher graph traversal with vector similarity
let results = db.hybrid_query()
    .cypher("MATCH (n:Memory)-[:RELATED]->(m:Memory) RETURN m")
    .vector_similarity(query_embedding, threshold: 0.7)
    .combine_scores(|vec, graph| 0.7 * vec + 0.3 * graph)
    .top_k(10)
    .execute()?;
```

### 16.3 RAG Pattern

```rust
// RagEngine: store documents, retrieve context for LLM
let rag = RagEngine::new(rag_config, embedding_config)?;
rag.ingest(text: &str, metadata: HashMap)?;
let context: Vec<String> = rag.retrieve(query: &str, k: 5)?;
```

---

## 17. Implementation Mapping

Maps `src/` modules to ruvector APIs they should use:

| `src/` module | Primary ruvector API | Notes |
|---------------|---------------------|-------|
| [`embedding.rs`](../src/embedding.rs) | `ruvector_core::{EmbeddingProvider, ApiEmbedding, HashEmbedding}` | Wrap `BoxedEmbeddingProvider` |
| [`memory.rs`](../src/memory.rs) | `ruvector_core::agenticdb::{AgenticDB, SessionStateIndex, WitnessLog}` | Needs `storage` feature |
| [`graph_context.rs`](../src/graph_context.rs) | `ruvector_graph::{GraphDB, RagEngine, HybridIndex}` | Cypher queries |
| [`coherence.rs`](../src/coherence.rs) | `prime_radiant::{CoherenceEngine, SheafGraph, CoherenceGate, GateDecision}` | ComputeLane routing |
| [`learning.rs`](../src/learning.rs) | `sona::{SonaEngine, TrajectoryBuilder, ReasoningBank}` | Micro-LoRA + EWC++ |
| [`audit.rs`](../src/audit.rs) | `rvf_runtime::{RvfStore, WitnessBuilder}` + `rvf_crypto::{create_witness_chain, verify_witness_chain}` | Ed25519 audit chain |
| [`verification.rs`](../src/verification.rs) | `ruvector_verified::{ProofEnvironment, VerifiedOp, VerifiedStage}` | Formal proof gating |
| [`pipeline.rs`](../src/pipeline.rs) | `ruvector_cognitive_container::{CognitiveContainer, TickResult}` | Sealed container ticks |
| [`claude_api.rs`](../src/claude_api.rs) | `reqwest` (direct HTTP) | Claude Messages API |
| [`config.rs`](../src/config.rs) | `dotenvy` | Load from `.env` |
| [`language.rs`](../src/language.rs) | (pure logic) | TR/EN detection |
| [`mcp_tools.rs`](../src/mcp_tools.rs) | `mcp_servers.json` schema | JSON-RPC tool calls |
| [`error.rs`](../src/error.rs) | (pure Rust) | Unified error type |
| [`types.rs`](../src/types.rs) | (pure Rust) | Shared data types |

### Key Feature Flags

```toml
# Cargo.toml additions needed for full functionality:
ruvector-core = { path = "...", features = ["storage"] }
ruvector-solver = { path = "...", features = ["neumann"] }
```

### Embedding Strategy Decision

For the ai-assistant, recommended embedding approach:
1. **Development**: `HashEmbedding::new(384)` — zero dependencies, instant
2. **Production**: `ApiEmbedding::openai(api_key, "text-embedding-3-small")` — 1536 dims, best quality
3. **Local/offline**: Add `ort` crate + implement `EmbeddingProvider` trait using ONNX Runtime

### Dimension Consistency

All vectors in the same `AgenticDB` instance must have the **same dimension**.  
Choose one and stick to it:
- `384` — all-MiniLM models, BgeSmall
- `1536` — OpenAI text-embedding-3-small, ada-002
- `3072` — OpenAI text-embedding-3-large
- `256` — SonaEngine default hidden/embedding dim

> **Recommendation**: Use `1536` for production (OpenAI), `384` for testing (HashEmbedding).
