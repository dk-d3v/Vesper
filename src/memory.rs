//! Memory module — episode storage and retrieval with sheaf-graph coherence.
//!
//! Uses [`AgenticDB`] + [`OnnxEmbedding`] for semantic storage and falls back
//! to an in-memory Vec when AgenticDB is unavailable. [`MemoryCoherenceLayer`]
//! (prime-radiant) tracks episodes in a SheafGraph for contradiction detection.

use crate::{
    config::{MEMORY_TOP_K, SKILL_TOP_K},
    embedding::{OnnxEmbedding, EMBEDDING_DIM},
    error::AiAssistantError,
    types::{Episode, EpisodeTier, SemanticContext},
};
use prime_radiant::ruvllm_integration::{
    MemoryCoherenceConfig, MemoryCoherenceLayer, MemoryEntry,
};
use ruvector_core::{embeddings::BoxedEmbeddingProvider, AgenticDB};
use ruvector_core::types::DbOptions;
use ruvector_math::optimal_transport::{OptimalTransport, SlicedWasserstein};
use ruvector_temporal_tensor::{segment as tt_segment, TemporalTensorCompressor, TierPolicy};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
    thread,
};
use tracing::{info, warn, error};
use uuid::Uuid;

/// Minimum cosine-similarity for episode retrieval.
///
/// Matches the consensus across ruvector reference implementations
/// (`reasoning_bank.rs`, `memory_layer.rs`, `fusion_graph.rs`): 0.7.
/// Tuned down to 0.62 for broader associative recall in practice.
const MIN_EPISODE_SIMILARITY: f32 = 0.62;

/// Sliding window size for near-duplicate detection (mirrors `FrameDeduplicator`).
const DEDUP_WINDOW_SIZE: usize = 50;

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
    /// Sliding window of recent embeddings for near-duplicate detection.
    dedup_window: VecDeque<Vec<f32>>,
    /// Sheaf-graph coherence layer — contradiction detection across episodes.
    coherence: MemoryCoherenceLayer,
}

// ── Conversions ───────────────────────────────────────────────────────────────

fn reflexion_to_episode(ep: &ruvector_core::agenticdb::ReflexionEpisode, quality: f32) -> Episode {
    let now_ts = chrono::Utc::now().timestamp();
    let age_secs = (now_ts - ep.timestamp).max(0) as u64;
    let tier = MemoryStore::get_tier(age_secs);
    let ts = UNIX_EPOCH + Duration::from_secs(ep.timestamp.max(0) as u64);

    // Build full Q+A text for the <memory> block:
    //   task       = user_query  (what was asked)
    //   observations[0] = assistant_response (what was answered)
    // critique is only the embedding key — not shown to Claude.
    let assistant_text = ep.observations.first().map(|s| s.as_str()).unwrap_or("");
    let text = if assistant_text.is_empty() {
        ep.task.clone()
    } else {
        format!("User: {}\nAssistant: {}", ep.task, assistant_text)
    };

    Episode {
        id: ep.id.clone(),
        text,
        embedding: ep.embedding.clone(),
        timestamp: ts,
        tier,
        quality_score: quality,
    }
}

// ── Module-private helpers ────────────────────────────────────────────────────

/// Try to open AgenticDB with retry on lock conflict; fall back to InMemory.
fn init_agentic_backend(provider: BoxedEmbeddingProvider, options: DbOptions) -> MemoryBackend {
    const MAX_TRIES: u8 = 3;
    let mut last_locked = false;
    for attempt in 1..=MAX_TRIES {
        match AgenticDB::with_embedding_provider(options.clone(), provider.clone()) {
            Ok(db) => {
                info!("AgenticDB initialised with OnnxEmbedding provider (attempt {})", attempt);
                return MemoryBackend::Agentic(db);
            }
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("already open") || msg.contains("Cannot acquire lock") {
                    error!(
                        "AgenticDB lock conflict ({}/{}): {} — \
                         is another instance of ai-assistant already running?",
                        attempt, MAX_TRIES, msg
                    );
                    if attempt < MAX_TRIES {
                        thread::sleep(Duration::from_millis(500));
                    }
                    last_locked = true;
                } else {
                    warn!("AgenticDB init failed ({}); falling back to in-memory store", e);
                    break;
                }
            }
        }
    }
    if last_locked {
        error!(
            "AgenticDB still locked after {} attempts — in-memory store (non-persistent). \
             Ensure no other instance is running, then restart.",
            MAX_TRIES
        );
    }
    MemoryBackend::InMemory(Vec::new())
}

/// Build a [`MemoryCoherenceLayer`] configured for the project embedding dimension.
fn new_coherence_layer() -> MemoryCoherenceLayer {
    MemoryCoherenceLayer::with_config(MemoryCoherenceConfig {
        embedding_dim: EMBEDDING_DIM,
        ..Default::default()
    })
}

// ── Core impl ─────────────────────────────────────────────────────────────────

impl MemoryStore {
    /// Initialise with a shared `Arc<OnnxEmbedding>` (preferred when sharing
    /// the embedding model with `GraphContextProvider`).
    pub fn new_with_arc(embedding_arc: Arc<OnnxEmbedding>) -> Result<Self, AiAssistantError> {
        let provider: BoxedEmbeddingProvider = embedding_arc.clone();
        let options = DbOptions {
            dimensions: EMBEDDING_DIM,
            storage_path: crate::config::exe_dir()
                .join("data")
                .join("agenticdb.db")
                .to_string_lossy()
                .into_owned(),
            ..DbOptions::default()
        };
        Ok(Self {
            embedding: embedding_arc,
            backend: init_agentic_backend(provider, options),
            causal_cache: Vec::new(),
            quality_map: HashMap::new(),
            dedup_window: VecDeque::with_capacity(DEDUP_WINDOW_SIZE),
            coherence: new_coherence_layer(),
        })
    }

    /// Initialise with `OnnxEmbedding` (takes ownership; wraps in `Arc` internally).
    pub fn new(embedding: OnnxEmbedding) -> Result<Self, AiAssistantError> {
        let embedding_arc: Arc<OnnxEmbedding> = Arc::new(embedding);
        let provider: BoxedEmbeddingProvider = embedding_arc.clone();
        let options = DbOptions {
            dimensions: EMBEDDING_DIM,
            storage_path: crate::config::exe_dir()
                .join("data")
                .join("agenticdb.db")
                .to_string_lossy()
                .into_owned(),
            ..DbOptions::default()
        };
        Ok(Self {
            embedding: embedding_arc,
            backend: init_agentic_backend(provider, options),
            causal_cache: Vec::new(),
            quality_map: HashMap::new(),
            dedup_window: VecDeque::with_capacity(DEDUP_WINDOW_SIZE),
            coherence: new_coherence_layer(),
        })
    }

    /// Test-only constructor — skips AgenticDB (avoids ~2 GB pre-alloc in CI).
    pub fn new_in_memory_for_test(embedding: OnnxEmbedding) -> Self {
        Self {
            embedding: Arc::new(embedding),
            backend: MemoryBackend::InMemory(Vec::new()),
            causal_cache: Vec::new(),
            quality_map: HashMap::new(),
            dedup_window: VecDeque::with_capacity(DEDUP_WINDOW_SIZE),
            coherence: new_coherence_layer(),
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

    /// Persist an episode and register it in the coherence layer; returns its ID.
    ///
    /// Coherence gate runs **before** the backend write — if the SheafGraph detects
    /// a contradiction the episode is rejected and `Err` is returned immediately.
    ///
    /// # Embedding strategy
    /// Only `user_query` is used to generate the embedding (critique field in
    /// AgenticDB). This mirrors the ruvector demo pattern where `critique` is a
    /// short lesson/query string — NOT the full assistant response — so that
    /// future retrievals with short queries yield high cosine similarity.
    /// `assistant_response` is stored as an observation (human-readable) but is
    /// NOT embedded.
    pub fn store_episode(
        &mut self,
        user_query: &str,
        assistant_response: &str,
        quality_score: f32,
    ) -> Result<String, AiAssistantError> {
        let now = SystemTime::now();
        // Embed only the user query — short, focused embedding avoids dilution
        // caused by mixing 500-word assistant responses into the vector.
        let embedding = self.embedding.embed(user_query)?;

        // ── Coherence gate (pre-write) ────────────────────────────────────────
        let coherence_key = Uuid::new_v4().to_string();
        let entry = MemoryEntry::episodic(coherence_key.clone(), embedding.clone(), 0);
        match self.coherence.add_with_coherence(entry) {
            Ok(result) if !result.is_coherent => {
                warn!(
                    episode_key = %coherence_key,
                    conflicts  = result.conflicting_memories.len(),
                    energy     = result.energy,
                    query_preview = %user_query.chars().take(60).collect::<String>(),
                    "coherence: contradiction detected — episode rejected (not stored)"
                );
                return Err(AiAssistantError::Memory(
                    "coherence contradiction: episode conflicts with existing memory".to_string(),
                ));
            }
            Ok(_) => {
                // Coherent — proceed to backend write.
            }
            Err(e) => {
                // Coherence layer error is non-fatal; log and continue.
                warn!("coherence layer insert failed: {e}");
            }
        }

        // ── Backend write ─────────────────────────────────────────────────────
        // AgenticDB: task = user_query (short description),
        //            observations = [assistant_response] (full detail),
        //            critique = user_query  ← ONLY THIS IS EMBEDDED for retrieval
        let combined_text = format!("User: {}\nAssistant: {}", user_query, assistant_response);
        let id = match &mut self.backend {
            MemoryBackend::Agentic(db) => db
                .store_episode(
                    user_query.to_string(),
                    vec![],
                    vec![assistant_response.to_string()],
                    user_query.to_string(), // critique — embedded field
                )
                .map_err(|e| AiAssistantError::Memory(e.to_string()))?,
            MemoryBackend::InMemory(episodes) => {
                let id = Uuid::new_v4().to_string();
                episodes.push(InMemEpisode {
                    id: id.clone(),
                    // Store the full combined text for human-readable context,
                    // but the embedding only covers user_query (see above).
                    text: combined_text,
                    embedding: embedding.clone(),
                    quality: quality_score,
                    timestamp: now,
                });
                id
            }
        };

        // Sliding dedup window (FrameDeduplicator pattern).
        if self.dedup_window.len() >= DEDUP_WINDOW_SIZE {
            self.dedup_window.pop_front();
        }
        self.dedup_window.push_back(embedding);

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

        // Log any incoherent memories detected by the sheaf graph.
        let incoherent = self.coherence.find_incoherent_memories();
        if !incoherent.is_empty() {
            info!(count = incoherent.len(), "coherence: incoherent memories detected");
        }

        let mut episodes: Vec<Episode> = match &self.backend {
            MemoryBackend::Agentic(db) => {
                // Fetch more than top_k — AgenticDB has no score cut-off.
                let fetch_k = (top_k * 3).max(top_k + 10);
                let raw = db
                    .retrieve_similar_episodes(query_text, fetch_k)
                    .map_err(|e| AiAssistantError::Memory(e.to_string()))?;

                let filtered: Vec<Episode> = raw
                    .iter()
                    .filter_map(|ep| {
                        if ep.embedding.is_empty() { return None; }
                        let sim = cosine_sim(&query_emb, &ep.embedding);
                        if sim < MIN_EPISODE_SIMILARITY {
                            info!(
                                episode_id = %ep.id,
                                similarity = format!("{:.4}", sim),
                                threshold = MIN_EPISODE_SIMILARITY,
                                task_preview = %ep.task.chars().take(50).collect::<String>(),
                                "episode rejected — below similarity threshold"
                            );
                            return None;
                        }
                        let quality = self.quality_map.get(&ep.id).map(|(q, _)| *q).unwrap_or(1.0);
                        Some(reflexion_to_episode(ep, quality))
                    })
                    .take(top_k)
                    .collect();

                let rejected = raw.len().saturating_sub(filtered.len());
                if rejected > 0 {
                    info!(
                        rejected,
                        threshold = MIN_EPISODE_SIMILARITY,
                        "Agentic: {} episode(s) below similarity threshold filtered out",
                        rejected
                    );
                }
                filtered
            }
            MemoryBackend::InMemory(episodes) => {
                let mut scored: Vec<(f32, Episode)> = episodes
                    .iter()
                    .filter_map(|ep| {
                        let sim = cosine_sim(&query_emb, &ep.embedding);
                        if sim < MIN_EPISODE_SIMILARITY {
                            info!(
                                episode_id = %ep.id,
                                similarity = format!("{:.4}", sim),
                                threshold = MIN_EPISODE_SIMILARITY,
                                text_preview = %ep.text.chars().take(50).collect::<String>(),
                                "episode rejected — below similarity threshold"
                            );
                            return None;
                        }
                        let age = ep.timestamp.elapsed().unwrap_or_default().as_secs();
                        let tier = Self::get_tier(age);
                        Some((sim, Episode {
                            id: ep.id.clone(),
                            text: ep.text.clone(),
                            embedding: ep.embedding.clone(),
                            timestamp: ep.timestamp,
                            tier,
                            quality_score: ep.quality,
                        }))
                    })
                    .collect();

                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

                let rejected = episodes.len().saturating_sub(scored.len());
                if rejected > 0 {
                    info!(
                        rejected,
                        threshold = MIN_EPISODE_SIMILARITY,
                        "InMemory: {} episode(s) below similarity threshold filtered out",
                        rejected
                    );
                }
                scored.into_iter().take(top_k).map(|(_, ep)| ep).collect()
            }
        };

        // Apply temporal compression (8/5/3-bit tiered quantization).
        for ep in &mut episodes {
            ep.embedding = apply_temporal_compression(&ep.embedding, &ep.tier);
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
        Ok(self
            .causal_cache
            .iter()
            .filter(|(c, e)| {
                c.to_lowercase().contains(&topic_lower)
                    || e.to_lowercase().contains(&topic_lower)
            })
            .cloned()
            .collect())
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

    // ── Near-duplicate check ──────────────────────────────────────────────────

    /// Return `true` when any stored episode's cosine similarity exceeds `threshold`.
    ///
    /// O(window_size) scan of `dedup_window` (≤ 50 entries) — no DB query.
    pub fn is_near_duplicate(&self, text: &str, threshold: f32) -> bool {
        if self.dedup_window.is_empty() { return false; }
        match self.embedding.embed(text) {
            Ok(query_emb) => self
                .dedup_window
                .iter()
                .any(|stored| cosine_sim(&query_emb, stored) > threshold),
            Err(_) => false,
        }
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

// ── Private functions ─────────────────────────────────────────────────────────

/// Apply tiered quantization-compression to an embedding.
///
/// Hot → 8-bit, Warm → 5-bit (via TierPolicy), Cold → 3-bit.
/// Returns uncompressed original if compression fails.
fn apply_temporal_compression(embedding: &[f32], tier: &EpisodeTier) -> Vec<f32> {
    let access_count: u32 = match tier {
        EpisodeTier::Hot => 100,
        EpisodeTier::Warm => 10,
        EpisodeTier::Cold => 1,
    };
    let dim = embedding.len();
    if dim == 0 { return embedding.to_vec(); }
    let mut comp = TemporalTensorCompressor::new(TierPolicy::default(), dim as u32, 0u32);
    comp.set_access(access_count, 0u32);
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
            decoded.len(), dim
        );
        embedding.to_vec()
    }
}

/// Rank `episodes` by Sliced Wasserstein distance to `query_emb`.
///
/// Treats each embedding as a 1D empirical distribution. Lower distance = more
/// similar → appears earlier. Falls back to original order on NaN/Inf.
fn rank_by_ot(query_emb: &[f32], mut episodes: Vec<Episode>) -> Vec<Episode> {
    if episodes.is_empty() || query_emb.is_empty() { return episodes; }
    let sw = SlicedWasserstein::new(50).with_seed(42);
    let query_pts: Vec<Vec<f64>> = query_emb.iter().map(|&v| vec![v as f64]).collect();
    let mut scored: Vec<(f64, Episode)> = episodes
        .drain(..)
        .map(|ep| {
            let ep_pts: Vec<Vec<f64>> = ep.embedding.iter().map(|&v| vec![v as f64]).collect();
            (sw.distance(&query_pts, &ep_pts), ep)
        })
        .collect();
    if scored.iter().any(|(d, _)| !d.is_finite()) {
        warn!("OT ranking produced non-finite distances; preserving retrieval order");
        return scored.into_iter().map(|(_, ep)| ep).collect();
    }
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().map(|(_, ep)| ep).collect()
}

/// L2-normalised cosine similarity in [−1, 1].
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 { 0.0 } else { dot / (na * nb) }
}
