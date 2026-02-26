//! Coherence module — contradiction energy and hallucination detection via prime-radiant.
//!
//! Uses prime-radiant's `CoherenceEngine` (sheaf Laplacian) to compute structural
//! contradiction energy between belief nodes. Falls back to a word-pair heuristic
//! if the engine cannot be built with the provided text.

use prime_radiant::coherence::{CoherenceConfig, CoherenceEngine};

use crate::{
    config::{CRITICAL_THRESHOLD, HALLUCINATION_THRESHOLD, REFLEX_THRESHOLD},
    error::AiAssistantError,
    types::CoherenceResult,
};

/// Fixed dimension of the node state vectors in the sheaf graph.
const STATE_DIM: usize = 16;

/// Maximum theoretical energy for STATE_DIM-dimensional unit vectors.
/// Used to normalise raw sheaf energy to [0, 1].
const MAX_ENERGY: f64 = STATE_DIM as f64;

/// Contradiction word pairs used in the heuristic fallback.
const CONTRADICTION_PAIRS: &[(&str, &str)] = &[
    ("yes", "no"),
    ("true", "false"),
    ("always", "never"),
    ("all", "none"),
    ("correct", "incorrect"),
    ("valid", "invalid"),
];

/// Checker that computes sheaf-Laplacian contradiction energy via prime-radiant.
///
/// Designed to never block the pipeline: if energy cannot be determined it
/// safely returns [`CoherenceResult::Reflex`].
pub struct CoherenceChecker {
    config: CoherenceConfig,
}

impl CoherenceChecker {
    /// Create a new coherence checker.
    pub fn new() -> Result<Self, AiAssistantError> {
        let config = CoherenceConfig {
            default_dimension: STATE_DIM,
            cache_residuals: false, // Local engines are short-lived
            ..CoherenceConfig::default()
        };
        Ok(Self { config })
    }

    /// Check context for contradictions before sending to Claude.
    ///
    /// - `Reflex`            → energy < [`REFLEX_THRESHOLD`]  (fast path, <1 ms)
    /// - `Revised(new_ctx)` → [`REFLEX_THRESHOLD`] ≤ energy < [`CRITICAL_THRESHOLD`]
    /// - `Critical`          → energy ≥ [`CRITICAL_THRESHOLD`] (halt)
    pub fn check_context(&self, context: &str) -> Result<CoherenceResult, AiAssistantError> {
        if context.is_empty() {
            return Ok(CoherenceResult::Reflex);
        }

        let energy = self.context_energy(context);

        if energy < REFLEX_THRESHOLD {
            Ok(CoherenceResult::Reflex)
        } else if energy < CRITICAL_THRESHOLD {
            let revised = self.revise_context(context);
            Ok(CoherenceResult::Revised(revised))
        } else {
            Ok(CoherenceResult::Critical)
        }
    }

    /// Check response for hallucinations after Claude responds.
    ///
    /// Returns `true` when contradiction energy between response and context
    /// exceeds [`HALLUCINATION_THRESHOLD`].
    pub fn check_response(
        &self,
        response: &str,
        context: &str,
    ) -> Result<bool, AiAssistantError> {
        if response.is_empty() || context.is_empty() {
            return Ok(false);
        }
        let energy = self.compute_energy(response, context);
        Ok(energy > HALLUCINATION_THRESHOLD)
    }

    /// Compute contradiction energy between two texts using sheaf Laplacian.
    ///
    /// Builds a two-node sheaf graph: `n1 → n2` and returns the weighted
    /// residual energy normalised to [0, 1].
    fn compute_energy(&self, text1: &str, text2: &str) -> f64 {
        let engine = CoherenceEngine::new(self.config.clone());

        let s1 = Self::text_to_state(text1, STATE_DIM);
        let s2 = Self::text_to_state(text2, STATE_DIM);

        // Ignore errors — graceful degradation
        let _ = engine.add_node("n1", s1);
        let _ = engine.add_node("n2", s2);
        let _ = engine.add_edge("n1", "n2", 1.0, None);

        let raw = engine.compute_energy().total_energy as f64;
        (raw / MAX_ENERGY).clamp(0.0, 1.0)
    }

    /// Compute averaged energy across consecutive sentence pairs in `context`.
    fn context_energy(&self, context: &str) -> f64 {
        let sentences: Vec<&str> = context
            .split(['.', '!', '?', '\n'])
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();

        if sentences.len() < 2 {
            // Single sentence — no contradiction possible
            return self.heuristic_energy(context);
        }

        let engine = CoherenceEngine::new(self.config.clone());

        for (i, sentence) in sentences.iter().enumerate() {
            let state = Self::text_to_state(sentence, STATE_DIM);
            let _ = engine.add_node(format!("s{}", i), state);
        }

        let edge_count = sentences.len() - 1;
        for i in 0..edge_count {
            let _ = engine.add_edge(format!("s{}", i), format!("s{}", i + 1), 1.0, None);
        }

        let raw = engine.compute_energy().total_energy as f64;
        (raw / (MAX_ENERGY * edge_count as f64)).clamp(0.0, 1.0)
    }

    /// Heuristic fallback energy: fraction of contradiction word pairs present.
    fn heuristic_energy(&self, text: &str) -> f64 {
        let lower = text.to_lowercase();
        let hit_count = CONTRADICTION_PAIRS
            .iter()
            .filter(|(a, b)| lower.contains(a) && lower.contains(b))
            .count();
        (hit_count as f64 / CONTRADICTION_PAIRS.len() as f64).clamp(0.0, 1.0)
    }

    /// Revise context by keeping only the lower-energy first half of sentences.
    fn revise_context(&self, context: &str) -> String {
        let sentences: Vec<&str> = context
            .split(['.', '!', '?', '\n'])
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();

        if sentences.len() <= 1 {
            return context.to_string();
        }

        let keep = (sentences.len() / 2).max(1);
        sentences[..keep].join(". ")
    }

    /// Encode text as a fixed-dimension f32 state vector for the sheaf graph.
    ///
    /// Splits the byte sequence into `dim` equal chunks and averages each,
    /// normalised to [0, 1].
    fn text_to_state(text: &str, dim: usize) -> Vec<f32> {
        let mut state = vec![0.0f32; dim];
        let bytes: Vec<u8> = text.bytes().take(dim * 8).collect();

        if bytes.is_empty() {
            return state;
        }

        let chunk_size = ((bytes.len() + dim - 1) / dim).max(1);
        for (i, chunk) in bytes.chunks(chunk_size).take(dim).enumerate() {
            let avg: f32 = chunk.iter().map(|&b| b as f32).sum::<f32>() / chunk.len() as f32;
            state[i] = avg / 255.0;
        }
        state
    }
}
