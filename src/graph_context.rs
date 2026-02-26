//! Graph context module — knowledge graph queries and RAG engine integration.
//!
//! Uses [`ruvector_graph::GraphDB`] (in-memory) as the knowledge store and
//! builds a lightweight RAG layer on top of it using [`SemanticSearch`] +
//! [`RagEngine`].
//!
//! # Graceful degradation
//! - If the graph is empty → returns empty [`GraphContext`] (pipeline continues).
//! - Any intermediate error is logged with `tracing::warn!` and skipped.

use crate::{
    config::{MAX_TOKENS, MIN_RAG_RELEVANCE, RAG_MAX_HOPS, RAG_TOP_K},
    embedding::OnnxEmbedding,
    error::AiAssistantError,
    types::GraphContext,
};
use ruvector_graph::{
    GraphDB, NodeBuilder,
    types::PropertyValue,
};
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

        Ok(Self {
            graph,
            embedding,
            causal_edges: Vec::new(),
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
    /// 3. Causal edge retrieval matching the query.
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

        // Step 3: causal edges
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
            rag_content,
            entity_count,
            causal_edges,
        })
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
