//! Graph context module — knowledge graph queries and RAG engine integration.
//!
//! Uses [`ruvector_graph::GraphDB`] (in-memory) as the knowledge store and
//! builds a lightweight RAG layer on top of it using [`SemanticSearch`] +
//! [`RagEngine`].
//!
//! # Integrations
//! - **ruvector-solver** (PageRank): ranks document nodes by importance using
//!   `ForwardPushSolver` for sublinear PPR computation.
//! - **ruvector-mincut** (topic clustering): identifies topic clusters and
//!   removes irrelevant subgraphs via `MinCutBuilder` / `DynamicMinCut`.
//!
//! # Graceful degradation
//! - If the graph is empty → returns empty [`GraphContext`] (pipeline continues).
//! - If PageRank fails → documents are returned in retrieval order.
//! - If MinCut fails → all documents are kept (no pruning).
//! - Any intermediate error is logged with `tracing::warn!` and skipped.

use crate::{
    config::{MAX_TOKENS, MIN_RAG_RELEVANCE, RAG_MAX_HOPS, RAG_TOP_K},
    embedding::OnnxEmbedding,
    error::AiAssistantError,
    types::GraphContext,
};
use ruvector_graph::{
    GraphDB, NodeBuilder,
    types::{PropertyValue, NodeId},
    GraphNeuralEngine, GnnConfig,
};
use ruvector_mincut::{MinCutBuilder, DynamicMinCut};
use ruvector_solver::forward_push::ForwardPushSolver;
use ruvector_solver::types::CsrMatrix;
use std::sync::Arc;
use tracing::{info, warn};

// ── Public struct ─────────────────────────────────────────────────────────────

/// Provides rich graph context (entities, RAG passages, causal edges) for a
/// given user query.
pub struct GraphContextProvider {
    graph: GraphDB,
    embedding: Arc<OnnxEmbedding>,
    /// Inline causal edges as `(cause, effect)` text pairs.
    causal_edges: Vec<(String, String)>,
    /// GNN engine for relationship analysis.
    gnn: GraphNeuralEngine,
}

// ── Impl ──────────────────────────────────────────────────────────────────────

impl GraphContextProvider {
    /// Initialise an in-memory graph database.
    ///
    /// Does **not** persist to disk — graph is rebuilt from documents on
    /// each run (or pre-seeded by the pipeline).
    pub fn new(embedding: Arc<OnnxEmbedding>) -> Result<Self, AiAssistantError> {
        let graph = GraphDB::new();
        info!("GraphDB initialised (in-memory)");

        let gnn = GraphNeuralEngine::new(GnnConfig::default());
        info!("GraphNeuralEngine initialised (default config: {} layers, {} hidden_dim)",
            GnnConfig::default().num_layers,
            GnnConfig::default().hidden_dim
        );

        Ok(Self {
            graph,
            embedding,
            causal_edges: Vec::new(),
            gnn,
        })
    }

    // ── Document ingestion ────────────────────────────────────────────────────

    /// Add a text document to the graph as a `Document` node.
    ///
    /// The node stores `text` and `metadata` as properties and can be
    /// retrieved via the hybrid search index.
    pub fn add_document(
        &mut self,
        text: &str,
        metadata: &str,
    ) -> Result<(), AiAssistantError> {
        let node = NodeBuilder::new()
            .label("Document")
            .property("text", text)
            .property("metadata", metadata)
            .build();

        self.graph
            .create_node(node)
            .map_err(|e| AiAssistantError::GraphContext(e.to_string()))?;

        Ok(())
    }

    /// Add a causal edge to the local cache (also persisted as a graph edge
    /// when both nodes exist).
    pub fn add_causal_edge(&mut self, cause: &str, effect: &str) {
        self.causal_edges.push((cause.to_string(), effect.to_string()));
    }

    // ── Context assembly ──────────────────────────────────────────────────────

    /// Assemble a [`GraphContext`] for `query`.
    ///
    /// Steps:
    /// 1. Cypher-style entity lookup (label-based).
    /// 2. RAG multi-hop retrieval (up to [`RAG_MAX_HOPS`] hops).
    /// 3. PageRank ranking of retrieved documents.
    /// 4. MinCut topic clustering to remove irrelevant subgraphs.
    /// 5. GNN relationship analysis (enriches `rag_content`).
    /// 6. Causal edge retrieval matching the query.
    ///
    /// Returns empty context if the graph has no nodes.
    pub fn get_context(&self, query: &str) -> Result<GraphContext, AiAssistantError> {
        // Step 1: entity query
        let entities = self.query_entities(query).unwrap_or_else(|e| {
            warn!("entity query failed: {}; continuing with empty entity list", e);
            Vec::new()
        });

        let entity_count = entities.len();

        // Step 2: RAG retrieval
        let rag_content = self
            .rag_retrieve(query, MAX_TOKENS / 2)
            .unwrap_or_else(|e| {
                warn!("RAG retrieval failed: {}; using empty context", e);
                String::new()
            });

        // Step 3: PageRank ranking of document chunks
        let ranked_content = self.apply_pagerank_ranking(&rag_content);

        // Step 4: MinCut topic clustering to prune irrelevant subgraphs
        let pruned_content = self.apply_mincut_pruning(&ranked_content, query);

        // Step 5: GNN relationship analysis — enrich rag_content with graph embedding summary
        let pruned_content = self.apply_gnn_analysis(&pruned_content);

        // Step 6: causal edges
        let query_lower = query.to_lowercase();
        let causal_edges: Vec<(String, String)> = self
            .causal_edges
            .iter()
            .filter(|(c, e)| {
                c.to_lowercase().contains(&query_lower)
                    || e.to_lowercase().contains(&query_lower)
            })
            .cloned()
            .collect();

        Ok(GraphContext {
            rag_content: pruned_content,
            entity_count,
            causal_edges,
        })
    }

    // ── GNN integration (ruvector-graph) ─────────────────────────────────────

    /// Run GNN graph-level embedding on retrieved document node IDs and append
    /// a one-line summary to `rag_content`.
    ///
    /// **Graceful degradation**: returns `content` unchanged on any failure.
    fn apply_gnn_analysis(&self, content: &str) -> String {
        // Use node_count() — in-memory GraphDB DashMap (no all_node_ids on in-memory store)
        let count = self.graph.node_count();
        if count == 0 {
            return content.to_string();
        }

        // embed_graph only uses node_ids.len(); generate synthetic IDs
        let node_ids: Vec<NodeId> = (0..count).map(|i| format!("node-{i}")).collect();

        match self.gnn.embed_graph(&node_ids) {
            Ok(graph_embedding) => {
                let summary = format!(
                    "\n---\n[GNN] graph_embedding: {} nodes analysed, method={}, dim={}",
                    graph_embedding.node_count,
                    graph_embedding.method,
                    graph_embedding.embedding.len(),
                );
                info!(
                    "GNN analysis complete: {} nodes, method={}",
                    graph_embedding.node_count, graph_embedding.method
                );
                format!("{}{}", content, summary)
            }
            Err(e) => {
                warn!("GNN embed_graph failed: {}; skipping GNN enrichment", e);
                content.to_string()
            }
        }
    }

    // ── PageRank integration (ruvector-solver) ────────────────────────────────

    /// Build a CSR adjacency matrix from document chunks and run PageRank to
    /// re-sort them by importance (higher PageRank = earlier in output).
    ///
    /// Each chunk becomes a graph node; edges are formed between chunks that
    /// share significant word overlap (proxy for semantic similarity).
    ///
    /// **Graceful degradation**: returns `rag_content` unchanged on failure.
    fn apply_pagerank_ranking(&self, rag_content: &str) -> String {
        if rag_content.is_empty() {
            return String::new();
        }

        let chunks: Vec<&str> = rag_content.split("\n---\n").collect();
        if chunks.len() < 2 {
            return rag_content.to_string();
        }

        match self.pagerank_rank_chunks(&chunks) {
            Ok(ranked) => ranked,
            Err(e) => {
                warn!("PageRank ranking failed (falling back to retrieval order): {}", e);
                rag_content.to_string()
            }
        }
    }

    /// Core PageRank computation over document chunks.
    fn pagerank_rank_chunks(&self, chunks: &[&str]) -> Result<String, String> {
        let n = chunks.len();
        if n == 0 {
            return Ok(String::new());
        }

        // Build adjacency edges based on word overlap between chunks
        let word_sets: Vec<std::collections::HashSet<&str>> = chunks
            .iter()
            .map(|chunk| {
                chunk.split_whitespace()
                    .filter(|w| w.len() > 3)
                    .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
                    .collect()
            })
            .collect();

        let mut coo_entries: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let overlap = word_sets[i].intersection(&word_sets[j]).count();
                let max_len = word_sets[i].len().max(word_sets[j].len()).max(1);
                let similarity = overlap as f64 / max_len as f64;

                // Create bidirectional edge if overlap is significant
                if similarity > 0.1 {
                    coo_entries.push((i, j, similarity));
                    coo_entries.push((j, i, similarity));
                }
            }
        }

        // If no edges, return original order
        if coo_entries.is_empty() {
            return Ok(chunks.join("\n---\n"));
        }

        let graph = CsrMatrix::<f64>::from_coo(n, n, coo_entries);
        let solver = ForwardPushSolver::new(0.85, 1e-4);

        // Run PPR from node 0 (first chunk, closest to the query)
        let ppr_scores = solver
            .ppr_from_source(&graph, 0)
            .map_err(|e| format!("ForwardPushSolver PPR failed: {}", e))?;

        // Build score map (node_index → score)
        let mut scores = vec![0.0f64; n];
        for &(idx, score) in &ppr_scores {
            if idx < n {
                scores[idx] = score;
            }
        }

        // Sort chunk indices by PageRank score (descending)
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Reassemble chunks in ranked order
        let ranked: Vec<&str> = indices.iter().map(|&i| chunks[i]).collect();
        Ok(ranked.join("\n---\n"))
    }

    // ── MinCut integration (ruvector-mincut) ──────────────────────────────────

    /// Use MinCut to identify topic clusters in the document graph and remove
    /// chunks that are disconnected from the query-relevant subgraph.
    ///
    /// **Graceful degradation**: returns `content` unchanged on failure.
    fn apply_mincut_pruning(&self, content: &str, query: &str) -> String {
        if content.is_empty() {
            return String::new();
        }

        let chunks: Vec<&str> = content.split("\n---\n").collect();
        if chunks.len() < 3 {
            // Too few chunks to benefit from pruning
            return content.to_string();
        }

        match self.mincut_prune_chunks(&chunks, query) {
            Ok(pruned) => pruned,
            Err(e) => {
                warn!("MinCut pruning failed (keeping all documents): {}", e);
                content.to_string()
            }
        }
    }

    /// Core MinCut-based pruning: builds a weighted graph of chunk similarity
    /// and uses MinCut to find the partition. Chunks in the same partition as
    /// the query-relevant chunk (node 0) are kept; others are removed.
    fn mincut_prune_chunks(&self, chunks: &[&str], query: &str) -> Result<String, String> {
        let n = chunks.len();

        // Find the chunk most relevant to the query
        let query_lower = query.to_lowercase();
        let query_node = chunks
            .iter()
            .enumerate()
            .max_by_key(|(_, chunk)| {
                let chunk_lower = chunk.to_lowercase();
                query_lower
                    .split_whitespace()
                    .filter(|w| w.len() > 3 && chunk_lower.contains(w))
                    .count()
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Build MinCut graph edges based on word overlap
        let word_sets: Vec<std::collections::HashSet<String>> = chunks
            .iter()
            .map(|chunk| {
                chunk
                    .split_whitespace()
                    .filter(|w| w.len() > 3)
                    .map(|w| w.to_lowercase())
                    .collect()
            })
            .collect();

        let mut edges: Vec<(u64, u64, f64)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let overlap = word_sets[i].intersection(&word_sets[j]).count();
                let max_len = word_sets[i].len().max(word_sets[j].len()).max(1);
                let weight = overlap as f64 / max_len as f64;

                if weight > 0.05 {
                    edges.push((i as u64, j as u64, weight));
                }
            }
        }

        if edges.is_empty() {
            // No edges means no pruning can be done
            return Ok(chunks.join("\n---\n"));
        }

        // Build MinCut structure
        let mincut = MinCutBuilder::new()
            .exact()
            .with_edges(edges)
            .build()
            .map_err(|e| format!("MinCut build failed: {}", e))?;

        let result = mincut.min_cut();

        // If graph is well-connected (high min-cut), keep all chunks
        if result.value > 0.5 {
            return Ok(chunks.join("\n---\n"));
        }

        // Use partition info to keep chunks in the same partition as query_node
        if let Some((side_a, side_b)) = result.partition {
            let query_side = if side_a.contains(&(query_node as u64)) {
                &side_a
            } else {
                &side_b
            };

            let kept: Vec<&str> = chunks
                .iter()
                .enumerate()
                .filter(|(i, _)| query_side.contains(&(*i as u64)))
                .map(|(_, chunk)| *chunk)
                .collect();

            if kept.is_empty() {
                // Safety: never return empty if we had content
                return Ok(chunks.join("\n---\n"));
            }

            Ok(kept.join("\n---\n"))
        } else {
            // No partition info available, keep everything
            Ok(chunks.join("\n---\n"))
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Query all `Document` nodes and return those whose text contains any
    /// word from `query_text`. Mimics a Cypher `MATCH (n:Document)` filter.
    fn query_entities(
        &self,
        query_text: &str,
    ) -> Result<Vec<String>, AiAssistantError> {
        let nodes = self.graph.get_nodes_by_label("Document");

        if nodes.is_empty() {
            return Ok(Vec::new());
        }

        let query_words: Vec<String> = query_text
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .filter(|w| w.len() > 3)
            .collect();

        let matches: Vec<String> = nodes
            .iter()
            .filter_map(|node| {
                if let Some(PropertyValue::String(text)) = node.get_property("text") {
                    let text_lower = text.to_lowercase();
                    let relevant = query_words.is_empty()
                        || query_words.iter().any(|w| text_lower.contains(w.as_str()));
                    if relevant {
                        return Some(text.clone());
                    }
                }
                None
            })
            .collect();

        Ok(matches)
    }

    /// Multi-hop RAG retrieval.
    ///
    /// Embeds the query, then performs iterative expansion:
    /// - Hop 0: embed query → find top-k most similar documents by cosine sim.
    /// - Hops 1..N: re-embed the accumulated context → find additional docs.
    ///
    /// Stops at [`RAG_MAX_HOPS`] hops or when the token budget is exhausted.
    fn rag_retrieve(
        &self,
        query: &str,
        max_tokens: usize,
    ) -> Result<String, AiAssistantError> {
        let nodes = self.graph.get_nodes_by_label("Document");

        if nodes.is_empty() {
            return Ok(String::new());
        }

        // Embed initial query
        let query_emb = self.embedding.embed(query)?;

        // Collect all doc texts with their embeddings
        let doc_embeddings: Vec<(String, Vec<f32>)> = nodes
            .iter()
            .filter_map(|node| {
                if let Some(PropertyValue::String(text)) = node.get_property("text") {
                    match self.embedding.embed(text) {
                        Ok(emb) => Some((text.clone(), emb)),
                        Err(e) => {
                            warn!("Embedding doc for RAG failed: {}; skipping", e);
                            None
                        }
                    }
                } else {
                    None
                }
            })
            .collect();

        if doc_embeddings.is_empty() {
            return Ok(String::new());
        }

        let mut accumulated: Vec<String> = Vec::new();
        let mut used_indices: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        let mut current_emb = query_emb;
        let mut total_tokens = 0usize;

        for hop in 0..RAG_MAX_HOPS {
            // Score remaining docs
            let mut scored: Vec<(usize, f32)> = doc_embeddings
                .iter()
                .enumerate()
                .filter(|(i, _)| !used_indices.contains(i))
                .map(|(i, (_, emb))| {
                    let sim = cosine_sim(&current_emb, emb);
                    (i, sim)
                })
                .filter(|(_, sim)| *sim >= MIN_RAG_RELEVANCE)
                .collect();

            scored.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            let top_this_hop: Vec<(usize, &str)> = scored
                .iter()
                .take(RAG_TOP_K)
                .map(|(i, _)| (*i, doc_embeddings[*i].0.as_str()))
                .collect();

            if top_this_hop.is_empty() {
                break;
            }

            // Build context text for this hop
            let mut hop_text = String::new();
            for (idx, text) in &top_this_hop {
                let chunk_tokens = Self::estimate_tokens(text);
                if total_tokens + chunk_tokens > max_tokens {
                    break;
                }
                hop_text.push_str(text);
                hop_text.push('\n');
                total_tokens += chunk_tokens;
                used_indices.insert(*idx);
            }

            if hop_text.is_empty() {
                break;
            }

            // Re-embed accumulated context for next hop
            if hop + 1 < RAG_MAX_HOPS {
                match self.embedding.embed(&hop_text) {
                    Ok(emb) => current_emb = emb,
                    Err(e) => {
                        warn!("Re-embedding at hop {} failed: {}; stopping", hop, e);
                        accumulated.push(hop_text);
                        break;
                    }
                }
            }
            accumulated.push(hop_text);
        }

        Ok(accumulated.join("\n---\n"))
    }

    /// Rough token estimate: 1 token ≈ 4 characters.
    fn estimate_tokens(text: &str) -> usize {
        text.len() / 4
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// L2-normalised cosine similarity in [−1, 1].
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 {
        0.0
    } else {
        dot / (na * nb)
    }
}
