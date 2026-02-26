//! SONA-based trajectory recording and pattern discovery.
//!
//! CRITICAL: This module NEVER calls `apply_micro_lora()` or `apply_base_lora()`.
//! Those functions modify local model weights — inapplicable when routing through
//! the Claude API. Trajectory recording and pattern extraction are the only
//! operations performed.

use ruvector_sona::{SonaEngine, TrajectoryBuilder};
use ruvector_sona::engine::SonaEngineBuilder;
use std::sync::Arc;

use crate::{embedding::OnnxEmbedding, error::AiAssistantError, types::LearningResult};

/// Minimum response length (chars) before quality scoring ramps up.
const QUALITY_MIN_LENGTH: usize = 50;

/// Divisor used to scale quality from response length.
const QUALITY_LENGTH_DIVISOR: f32 = 500.0;

/// Low-quality sentinel phrases that cap the score.
const LOW_QUALITY_PHRASES: &[&str] = &[
    "i don't know",
    "i do not know",
    "i cannot answer",
    "i have no information",
    "bilmiyorum",
    "cevap veremiyorum",
];

/// In-memory record used for graceful fallback when SONA API differs.
#[derive(Debug, Clone)]
struct TrajectoryRecord {
    trajectory_id: String,
    first_word: String,
    quality: f32,
}

/// Learning engine that records conversation trajectories via SONA.
///
/// Trajectories are submitted to the SONA reasoning bank for K-means++
/// pattern extraction. LoRA weight-update methods are deliberately never
/// called — Claude is a remote API, not a local model.
pub struct LearningEngine {
    /// SONA engine (hidden_dim = 256 for ONNX embedding vectors).
    engine: SonaEngine,
    /// ONNX embedding provider shared with the rest of the pipeline.
    embedding: Arc<OnnxEmbedding>,
    /// Trajectory counter (used to generate unique IDs).
    trajectory_counter: u64,
    /// Fallback in-memory store (populated alongside SONA).
    records: Vec<TrajectoryRecord>,
}

impl LearningEngine {
    /// Create a new learning engine.
    ///
    /// `embedding` is the shared [`OnnxEmbedding`] instance (real ONNX
    /// inference when the model is loaded, hash fallback otherwise).
    pub fn new(embedding: Arc<OnnxEmbedding>) -> Result<Self, AiAssistantError> {
        let engine = SonaEngineBuilder::new()
            .hidden_dim(256)
            .micro_lora_rank(2)
            .base_lora_rank(8)
            .quality_threshold(0.3)
            .buffer_capacity(1000)
            .build();

        Ok(Self {
            engine,
            embedding,
            trajectory_counter: 0,
            records: Vec::new(),
        })
    }

    /// Record a conversation turn as a SONA trajectory.
    ///
    /// Steps recorded:
    ///   1. `user_message`  — query embedding proxy
    ///   2. `context`       — retrieval embedding proxy
    ///   3. `prompt`        — assembled prompt embedding proxy
    ///   4. `response`      — final response embedding proxy
    ///
    /// Does NOT call `apply_micro_lora` or `apply_base_lora`.
    pub fn record_trajectory(
        &mut self,
        user_message: &str,
        context: &str,
        prompt: &str,
        response: &str,
    ) -> Result<LearningResult, AiAssistantError> {
        self.trajectory_counter += 1;
        let trajectory_id = format!("traj-{:06}", self.trajectory_counter);

        let quality = Self::compute_quality(response, user_message);

        // Build query embedding using real ONNX inference (hash fallback if unavailable)
        let query_embedding = self
            .embedding
            .embed(user_message)
            .unwrap_or_else(|_| vec![0.0; 256]);

        let mut builder: TrajectoryBuilder =
            self.engine.begin_trajectory(query_embedding);

        // Step 1 — context retrieval
        builder.add_step(
            self.embedding.embed(context).unwrap_or_else(|_| vec![0.0; 256]),
            vec![],
            quality * 0.8,
        );

        // Step 2 — prompt assembly
        builder.add_step(
            self.embedding.embed(prompt).unwrap_or_else(|_| vec![0.0; 256]),
            vec![],
            quality * 0.9,
        );

        // Step 3 — response generation (highest reward)
        builder.add_step(
            self.embedding.embed(response).unwrap_or_else(|_| vec![0.0; 256]),
            vec![],
            quality,
        );

        // Submit trajectory — intentionally NOT calling apply_micro_lora/apply_base_lora
        self.engine.end_trajectory(builder, quality);

        // Tick the background loop (non-blocking; may run pattern extraction)
        let _tick_result = self.engine.tick();

        // Fallback record for in-memory pattern counting
        let first_word = user_message
            .split_whitespace()
            .next()
            .unwrap_or("_")
            .to_lowercase();

        self.records.push(TrajectoryRecord {
            trajectory_id: trajectory_id.clone(),
            first_word,
            quality,
        });

        let pattern_count = self.count_patterns();

        Ok(LearningResult {
            trajectory_id,
            quality_score: quality,
            pattern_count,
        })
    }

    /// Trigger explicit pattern extraction from buffered trajectories.
    ///
    /// Uses the SONA engine's forced background-learning cycle and returns the
    /// number of new patterns discovered (via the SONA stats delta).
    pub fn find_patterns(&mut self) -> Result<usize, AiAssistantError> {
        let stats_before = self.engine.stats();
        let _result = self.engine.force_learn();
        let stats_after = self.engine.stats();

        // Patterns extracted = trajectories processed in this cycle
        let new_patterns = stats_after
            .trajectories_buffered
            .saturating_sub(stats_before.trajectories_buffered);

        // Fallback: count distinct first words
        let fallback_count = self.count_patterns();

        Ok(new_patterns.max(fallback_count))
    }

    /// Compute a quality score in [0.0, 1.0] for a response.
    ///
    /// Heuristic rules (in priority order):
    /// - If response contains a low-quality phrase → 0.2
    /// - If response is too short → `len / QUALITY_LENGTH_DIVISOR`
    /// - Otherwise → `min(1.0, len / QUALITY_LENGTH_DIVISOR)`
    fn compute_quality(response: &str, _user_message: &str) -> f32 {
        let lower = response.to_lowercase();

        // Penalise known low-quality answers
        for phrase in LOW_QUALITY_PHRASES {
            if lower.contains(phrase) {
                return 0.2;
            }
        }

        if response.len() < QUALITY_MIN_LENGTH {
            return (response.len() as f32 / QUALITY_LENGTH_DIVISOR).clamp(0.0, 1.0);
        }

        (response.len() as f32 / QUALITY_LENGTH_DIVISOR).clamp(0.0, 1.0)
    }

    /// Count distinct patterns using the fallback first-word heuristic.
    fn count_patterns(&self) -> usize {
        use std::collections::HashSet;
        self.records
            .iter()
            .map(|r| r.first_word.as_str())
            .collect::<HashSet<_>>()
            .len()
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Convert text bytes into a 256-dimensional f32 embedding proxy.
///
/// Splits bytes into 256 equal chunks and averages each, normalised to [0, 1].
fn text_to_embedding(text: &str, dim: usize) -> Vec<f32> {
    let mut embedding = vec![0.0f32; dim];
    let bytes: Vec<u8> = text.bytes().take(dim * 4).collect();

    if bytes.is_empty() {
        return embedding;
    }

    let chunk_size = ((bytes.len() + dim - 1) / dim).max(1);
    for (i, chunk) in bytes.chunks(chunk_size).take(dim).enumerate() {
        let avg: f32 = chunk.iter().map(|&b| b as f32).sum::<f32>() / chunk.len() as f32;
        embedding[i] = avg / 255.0;
    }
    embedding
}
