//! Memory module — episode storage and retrieval.
//!
//! Wraps [`AgenticDB`] from `ruvector-core` with [`OnnxEmbedding`] as the
//! embedding provider.  Never uses `HashEmbedding`.
//!
//! # Graceful degradation
//! If `AgenticDB` fails to initialize (e.g., missing disk path, redb error),
//! the store transparently falls back to an in-memory `Vec` with cosine
//! similarity search.
//!
//! # Temporal compression
//! Episode embeddings are compressed through `ruvector-temporal-tensor`
//! before being returned, applying tiered quantization (8/5/3-bit) based on
//! the episode's age.
//!
//! # OT ranking
//! Retrieved episodes are ranked with Sliced Wasserstein distance
//! (`ruvector-math`), falling back to cosine similarity on failure.

use crate::{
    config::{MEMORY_TOP_K, SKILL_TOP_K},
    embedding::{OnnxEmbedding, EMBEDDING_DIM},
    error::AiAssistantError,
    types::{Episode, EpisodeTier, SemanticContext},
};
use ruvector_core::{embeddings::BoxedEmbeddingProvider, AgenticDB};
use ruvector_core::types::DbOptions;
use ruvector_math::optimal_transport::{OptimalTransport, SlicedWasserstein};
use ruvector_temporal_tensor::{segment as tt_segment, TemporalTensorCompressor, TierPolicy};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tracing::{info, warn};
use uuid::Uuid;

// ── Fallback episode record ───────────────────────────────────────────────────

#[derive(Clone)]
struct InMemEpisode {
    id: String,
    text: String,
    embedding: Vec<f32>,
    quality: f32,
    timestamp: SystemTime,
}

// ── Storage backend ───────────────────────────────────────────────────────────

enum MemoryBackend {
    /// AgenticDB backed by OnnxEmbedding (primary path).
    Agentic(AgenticDB),
    /// In-memory Vec fallback when AgenticDB fails.
    InMemory(Vec<InMemEpisode>),
}

// ── Public struct ─────────────────────────────────────────────────────────────

/// Persistent / in-memory episode store with semantic retrieval.
pub struct MemoryStore {
    embedding: Arc<OnnxEmbedding>,
    backend: MemoryBackend,
    /// Local cache of (cause, effect) string pairs.
    causal_cache: Vec<(String, String)>,
    /// Maps episode_id → (quality_score, store_time) for consolidation.
    quality_map: HashMap<String, (f32, SystemTime)>,
}

// ── Conversions ───────────────────────────────────────────────────────────────

/// Convert a `ReflexionEpisode` (ruvector-core internal type) to our [`Episode`].
fn reflexion_to_episode(
    ep: &ruvector_core::agenticdb::ReflexionEpisode,
    quality: f32,
) -> Episode {
    let now_ts = chrono::Utc::now().timestamp();
    let age_secs = (now_ts - ep.timestamp).max(0) as u64;
    let tier = MemoryStore::get_tier(age_secs);
    let ts = UNIX_EPOCH + Duration::from_secs(ep.timestamp.max(0) as u64);

    Episode {
        id: ep.id.clone(),
        text: ep.critique.clone(),
        embedding: ep.embedding.clone(),
        timestamp: ts,
        tier,
        quality_score: quality,
    }
}

// ── Core impl ─────────────────────────────────────────────────────────────────

impl MemoryStore {
    /// Initialise the store with a shared `Arc<OnnxEmbedding>`.
    ///
    /// Preferred constructor when the same embedding model is also used by
    /// `GraphContextProvider` — avoids loading the ONNX model twice.
    pub fn new_with_arc(embedding_arc: Arc<OnnxEmbedding>) -> Result<Self, AiAssistantError> {
        // Coerce Arc<OnnxEmbedding> → Arc<dyn EmbeddingProvider>
        let provider: BoxedEmbeddingProvider = embedding_arc.clone();

        let options = DbOptions {
            dimensions: EMBEDDING_DIM,
            storage_path: "./data/agenticdb.db".to_string(),
            ..DbOptions::default()
        };

        let backend = match AgenticDB::with_embedding_provider(options, provider) {
            Ok(db) => {
                info!("AgenticDB initialised with OnnxEmbedding provider");
                MemoryBackend::Agentic(db)
            }
            Err(e) => {
                warn!(
                    "AgenticDB init failed ({}); falling back to in-memory store",
                    e
                );
                MemoryBackend::InMemory(Vec::new())
            }
        };

        Ok(Self {
            embedding: embedding_arc,
            backend,
            causal_cache: Vec::new(),
            quality_map: HashMap::new(),
        })
    }

    /// Initialise the store with `OnnxEmbedding` (takes ownership).
    ///
    /// Wraps the value in `Arc` internally. Prefer [`MemoryStore::new_with_arc`]
    /// when sharing the embedding model with other components.
    pub fn new(embedding: OnnxEmbedding) -> Result<Self, AiAssistantError> {
        let embedding_arc: Arc<OnnxEmbedding> = Arc::new(embedding);

        // Coerce Arc<OnnxEmbedding> → Arc<dyn EmbeddingProvider>
        let provider: BoxedEmbeddingProvider = embedding_arc.clone();

        let options = DbOptions {
            dimensions: EMBEDDING_DIM,
            storage_path: "./data/agenticdb.db".to_string(),
            ..DbOptions::default()
        };

        let backend = match AgenticDB::with_embedding_provider(options, provider) {
            Ok(db) => {
                info!("AgenticDB initialised with OnnxEmbedding provider");
                MemoryBackend::Agentic(db)
            }
            Err(e) => {
                warn!(
                    "AgenticDB init failed ({}); falling back to in-memory store",
                    e
                );
                MemoryBackend::InMemory(Vec::new())
            }
        };

        Ok(Self {
            embedding: embedding_arc,
            backend,
            causal_cache: Vec::new(),
            quality_map: HashMap::new(),
        })
    }

    /// Test-only constructor that skips AgenticDB and uses in-memory storage.
    ///
    /// AgenticDB pre-allocates ~2 GB which crashes CI under memory pressure;
    /// this path bypasses that allocation while exercising all public methods.
    /// Available in all build profiles so integration tests in `tests/` can use it.
    pub fn new_in_memory_for_test(embedding: OnnxEmbedding) -> Self {
        let embedding_arc = Arc::new(embedding);
        Self {
            embedding: embedding_arc,
            backend: MemoryBackend::InMemory(Vec::new()),
            causal_cache: Vec::new(),
            quality_map: HashMap::new(),
        }
    }

    // ── Tier helper ───────────────────────────────────────────────────────────

    /// Classify an episode by its age.
    pub fn get_tier(age_seconds: u64) -> EpisodeTier {
        const ONE_DAY: u64 = 86_400;
        const ONE_WEEK: u64 = 604_800;
        if age_seconds < ONE_DAY {
            EpisodeTier::Hot
        } else if age_seconds < ONE_WEEK {
            EpisodeTier::Warm
        } else {
            EpisodeTier::Cold
        }
    }

    // ── Store ─────────────────────────────────────────────────────────────────

    /// Persist an episode; returns its unique ID.
    pub fn store_episode(
        &mut self,
        text: &str,
        quality_score: f32,
    ) -> Result<String, AiAssistantError> {
        let now = SystemTime::now();

        // Pre-compute embedding for InMemory path before taking &mut backend.
        let precomputed_emb = if matches!(self.backend, MemoryBackend::InMemory(_)) {
            Some(self.embedding.embed(text)?)
        } else {
            None
        };

        let id = match &mut self.backend {
            MemoryBackend::Agentic(db) => db
                .store_episode(text.to_string(), vec![], vec![], text.to_string())
                .map_err(|e| AiAssistantError::Memory(e.to_string()))?,
            MemoryBackend::InMemory(episodes) => {
                let embedding = precomputed_emb.unwrap_or_default();
                let id = Uuid::new_v4().to_string();
                episodes.push(InMemEpisode {
                    id: id.clone(),
                    text: text.to_string(),
                    embedding,
                    quality: quality_score,
                    timestamp: now,
                });
                id
            }
        };

        self.quality_map.insert(id.clone(), (quality_score, now));
        Ok(id)
    }

    // ── Retrieve ──────────────────────────────────────────────────────────────

    /// Retrieve the `top_k` most semantically similar episodes.
    pub fn retrieve_similar(
        &self,
        query_text: &str,
        top_k: usize,
    ) -> Result<Vec<Episode>, AiAssistantError> {
        let query_emb = self.embedding.embed(query_text)?;

        let mut episodes: Vec<Episode> = match &self.backend {
            MemoryBackend::Agentic(db) => {
                let raw = db
                    .retrieve_similar_episodes(query_text, top_k)
                    .map_err(|e| AiAssistantError::Memory(e.to_string()))?;
                raw.iter()
                    .map(|ep| {
                        let quality = self
                            .quality_map
                            .get(&ep.id)
                            .map(|(q, _)| *q)
                            .unwrap_or(1.0);
                        reflexion_to_episode(ep, quality)
                    })
                    .collect()
            }
            MemoryBackend::InMemory(episodes) => {
                let mut scored: Vec<(f32, Episode)> = episodes
                    .iter()
                    .map(|ep| {
                        let sim = cosine_sim(&query_emb, &ep.embedding);
                        let age = ep
                            .timestamp
                            .elapsed()
                            .unwrap_or_default()
                            .as_secs();
                        let tier = Self::get_tier(age);
                        (
                            sim,
                            Episode {
                                id: ep.id.clone(),
                                text: ep.text.clone(),
                                embedding: ep.embedding.clone(),
                                timestamp: ep.timestamp,
                                tier,
                                quality_score: ep.quality,
                            },
                        )
                    })
                    .collect();
                scored.sort_by(|a, b| {
                    b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                });
                scored
                    .into_iter()
                    .take(top_k)
                    .map(|(_, ep)| ep)
                    .collect()
            }
        };

        // Apply temporal compression to embeddings based on tier.
        for ep in &mut episodes {
            ep.embedding =
                apply_temporal_compression(&ep.embedding, &ep.tier);
        }

        // Re-rank using Optimal Transport (fallback: preserve cosine order).
        episodes = rank_by_ot(&query_emb, episodes);

        Ok(episodes)
    }

    // ── Skills ────────────────────────────────────────────────────────────────

    /// Return skill names relevant to `query_text`.
    pub fn search_skills(
        &self,
        query_text: &str,
        top_k: usize,
    ) -> Result<Vec<String>, AiAssistantError> {
        match &self.backend {
            MemoryBackend::Agentic(db) => {
                let skills = db
                    .search_skills(query_text, top_k)
                    .map_err(|e| AiAssistantError::Memory(e.to_string()))?;
                Ok(skills.into_iter().map(|s| s.name).collect())
            }
            MemoryBackend::InMemory(_) => {
                warn!("search_skills: in-memory backend has no skill library");
                Ok(Vec::new())
            }
        }
    }

    // ── Causal edges ──────────────────────────────────────────────────────────

    /// Record a causal edge (cause → effect) in AgenticDB and local cache.
    pub fn add_causal_edge(
        &mut self,
        cause: &str,
        effect: &str,
    ) -> Result<(), AiAssistantError> {
        self.causal_cache.push((cause.to_string(), effect.to_string()));

        if let MemoryBackend::Agentic(db) = &self.backend {
            db.add_causal_edge(
                vec![cause.to_string()],
                vec![effect.to_string()],
                1.0,
                format!("{} → {}", cause, effect),
            )
            .map_err(|e| AiAssistantError::Memory(e.to_string()))?;
        }
        Ok(())
    }

    /// Retrieve causal edges where cause or effect contains `topic`.
    pub fn get_causal_edges(
        &self,
        topic: &str,
    ) -> Result<Vec<(String, String)>, AiAssistantError> {
        let topic_lower = topic.to_lowercase();
        let results = self
            .causal_cache
            .iter()
            .filter(|(c, e)| {
                c.to_lowercase().contains(&topic_lower)
                    || e.to_lowercase().contains(&topic_lower)
            })
            .cloned()
            .collect();
        Ok(results)
    }

    // ── Consolidation ─────────────────────────────────────────────────────────

    /// Drop episodes with `quality_score < 0.3`. Returns count removed.
    pub fn auto_consolidate(&mut self) -> Result<usize, AiAssistantError> {
        const LOW_QUALITY: f32 = 0.3;

        let low_ids: Vec<String> = self
            .quality_map
            .iter()
            .filter(|(_, (q, _))| *q < LOW_QUALITY)
            .map(|(id, _)| id.clone())
            .collect();

        let count = low_ids.len();

        for id in &low_ids {
            self.quality_map.remove(id);
            match &mut self.backend {
                MemoryBackend::Agentic(db) => {
                    // Delete the reflexion vector entry from VectorDB.
                    let vector_id = format!("reflexion_{}", id);
                    if let Err(e) = db.delete(&vector_id) {
                        warn!("auto_consolidate: delete {} failed: {}", vector_id, e);
                    }
                }
                MemoryBackend::InMemory(episodes) => {
                    episodes.retain(|ep| ep.id != *id);
                }
            }
        }

        if count > 0 {
            info!("auto_consolidate: removed {} low-quality episodes", count);
        }
        Ok(count)
    }

    // ── Semantic context ──────────────────────────────────────────────────────

    /// Combine retrieved episodes and skills into a [`SemanticContext`].
    pub fn build_semantic_context(
        &self,
        query: &str,
    ) -> Result<SemanticContext, AiAssistantError> {
        let episodes = self.retrieve_similar(query, MEMORY_TOP_K)?;
        let skill_ids = self.search_skills(query, SKILL_TOP_K)?;
        Ok(SemanticContext { episodes, skill_ids })
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Apply tiered quantization-compression to an embedding.
///
/// Hot → 8-bit, Warm → 5-bit (via TierPolicy), Cold → 3-bit.
/// Returns uncompressed original if compression fails.
fn apply_temporal_compression(embedding: &[f32], tier: &EpisodeTier) -> Vec<f32> {
    let access_count: u64 = match tier {
        EpisodeTier::Hot => 100,
        EpisodeTier::Warm => 10,
        EpisodeTier::Cold => 1,
    };

    let dim = embedding.len();
    if dim == 0 {
        return embedding.to_vec();
    }

    let mut comp = TemporalTensorCompressor::new(
        TierPolicy::default(),
        dim as u32,
        0u32,
    );
    comp.set_access(access_count as u32, 0u32);

    let mut seg: Vec<u8> = Vec::new();
    comp.push_frame(embedding, 0, &mut seg);
    comp.flush(&mut seg);

    let mut decoded: Vec<f32> = Vec::new();
    tt_segment::decode(&seg, &mut decoded);

    if decoded.len() == dim {
        decoded
    } else {
        warn!(
            "Temporal compression produced wrong dim ({} vs {}); using original",
            decoded.len(),
            dim
        );
        embedding.to_vec()
    }
}

/// Rank `episodes` by Sliced Wasserstein distance to `query_emb`.
///
/// Treats each embedding as a 1D empirical distribution (one point per
/// dimension). Lower distance = more similar → appears earlier.
/// Falls back to original order if OT returns NaN/Inf.
fn rank_by_ot(query_emb: &[f32], mut episodes: Vec<Episode>) -> Vec<Episode> {
    if episodes.is_empty() || query_emb.is_empty() {
        return episodes;
    }

    let sw = SlicedWasserstein::new(50).with_seed(42);
    // OT distance method uses f64 — convert query embedding once
    let query_pts: Vec<Vec<f64>> = query_emb.iter().map(|&v| vec![v as f64]).collect();

    let mut scored: Vec<(f64, Episode)> = episodes
        .drain(..)
        .map(|ep| {
            let ep_pts: Vec<Vec<f64>> = ep.embedding.iter().map(|&v| vec![v as f64]).collect();
            let dist = sw.distance(&query_pts, &ep_pts);
            (dist, ep)
        })
        .collect();

    // Check for NaN/Inf — if any, fall back to original (insertion) order.
    let any_bad = scored.iter().any(|(d, _): &(f64, Episode)| !d.is_finite());
    if any_bad {
        warn!("OT ranking produced non-finite distances; preserving retrieval order");
        return scored.into_iter().map(|(_, ep)| ep).collect();
    }

    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().map(|(_, ep)| ep).collect()
}

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
