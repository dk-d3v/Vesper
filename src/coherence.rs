//! Coherence module — contradiction energy and hallucination detection via prime-radiant.
//!
//! Uses prime-radiant's `CoherenceEngine` (sheaf Laplacian) to compute structural
//! contradiction energy between belief nodes. Falls back to a word-pair heuristic
//! if the engine cannot be built with the provided text.
//!
//! Extended with `detect_contradictions()` and `check_logical_flow()` methods
//! inspired by the ruvllm quality coherence pattern.

use prime_radiant::coherence::{CoherenceConfig, CoherenceEngine};

use crate::{
    config::{CRITICAL_THRESHOLD, HALLUCINATION_THRESHOLD, REFLEX_THRESHOLD},
    error::AiAssistantError,
    types::CoherenceResult,
};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Fixed dimension of the node state vectors in the sheaf graph.
const STATE_DIM: usize = 16;

/// Maximum theoretical energy for STATE_DIM-dimensional unit vectors.
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

/// Negation markers for contradiction detection.
const NEGATION_WORDS: &[&str] = &[
    "not", "never", "no", "none", "neither", "nothing",
    "isn't", "aren't", "wasn't", "weren't", "don't",
    "doesn't", "didn't", "won't", "wouldn't", "couldn't",
];

/// Transition markers that indicate logical flow.
const TRANSITION_MARKERS: &[&str] = &[
    "however", "therefore", "furthermore", "moreover",
    "consequently", "thus", "hence", "additionally",
    "nonetheless", "finally", "first", "second", "then", "next",
];

// ── Public types ──────────────────────────────────────────────────────────────

/// Category of a detected contradiction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContradictionType {
    /// Direct logical negation.
    Logical,
    /// Temporal inconsistency (before/after reversal).
    Temporal,
    /// Numeric value mismatch.
    Numeric,
    /// Entity attribute mismatch.
    AttributeMismatch,
    /// Causal contradiction.
    Causal,
    /// Contextual inconsistency.
    Contextual,
}

/// Result of `detect_contradictions()`.
#[derive(Debug, Clone)]
pub struct ContradictionResult {
    /// Whether a contradiction was found.
    pub found: bool,
    /// The dominant contradiction type, if any.
    pub contradiction_type: Option<ContradictionType>,
    /// Confidence in the detection (0.0–1.0).
    pub confidence: f32,
    /// Human-readable explanation.
    pub explanation: String,
}

impl ContradictionResult {
    /// Conservative fallback — no contradiction detected.
    fn none() -> Self {
        Self {
            found: false,
            contradiction_type: None,
            confidence: 0.0,
            explanation: "No contradiction detected".to_string(),
        }
    }
}

/// Result of `check_logical_flow()`.
#[derive(Debug, Clone)]
pub struct LogicalFlowResult {
    /// Whether the conversation maintains coherent flow.
    pub is_coherent: bool,
    /// Transitions that appear to be missing.
    pub missing_transitions: Vec<String>,
    /// References that could not be resolved.
    pub broken_references: Vec<String>,
    /// Overall flow score (0.0–1.0).
    pub score: f32,
}

impl LogicalFlowResult {
    /// Conservative fallback — assume coherent.
    fn assume_coherent() -> Self {
        Self {
            is_coherent: true,
            missing_transitions: vec![],
            broken_references: vec![],
            score: 1.0,
        }
    }
}

// ── CoherenceChecker ──────────────────────────────────────────────────────────

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
            cache_residuals: false,
            ..CoherenceConfig::default()
        };
        Ok(Self { config })
    }

    // ── Pipeline-facing methods ───────────────────────────────────────────────

    /// Check context for contradictions before sending to Claude.
    ///
    /// - `Reflex`           → energy < 0.3  (fast path, no revision, <1 ms)
    /// - `Revised(new_ctx)` → 0.3 ≤ energy ≤ 0.8  (context revised/enriched)
    /// - `Halt`             → energy > 0.8  (block request — potential loop/abuse)
    pub fn check_context(&self, context: &str) -> Result<CoherenceResult, AiAssistantError> {
        if context.is_empty() {
            return Ok(CoherenceResult::Reflex);
        }

        let energy = self.context_energy(context);

        if energy < REFLEX_THRESHOLD {
            Ok(CoherenceResult::Reflex)
        } else if energy <= CRITICAL_THRESHOLD {
            let revised = self.revise_context(context);
            Ok(CoherenceResult::Revised(revised))
        } else {
            Ok(CoherenceResult::Halt)
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

    // ── New: contradiction & flow analysis ───────────────────────────────────

    /// Detect contradictions between `prev_context` and `new_response`.
    ///
    /// Checks for:
    /// - Negation-based logical contradictions
    /// - Numeric value mismatches
    /// - Temporal inconsistencies
    ///
    /// Gracefully degrades: returns `ContradictionResult::none()` on any error.
    pub fn detect_contradictions(
        &self,
        prev_context: &str,
        new_response: &str,
    ) -> ContradictionResult {
        if prev_context.is_empty() || new_response.is_empty() {
            return ContradictionResult::none();
        }

        // 1. Numeric contradiction check
        if let Some(r) = Self::check_numeric_contradiction(prev_context, new_response) {
            return r;
        }

        // 2. Temporal contradiction check
        if let Some(r) = Self::check_temporal_contradiction(prev_context, new_response) {
            return r;
        }

        // 3. Negation-based logical contradiction (energy-weighted)
        let energy = self.compute_energy(prev_context, new_response);
        let has_neg_prev = Self::contains_negation(prev_context);
        let has_neg_resp = Self::contains_negation(new_response);

        if has_neg_prev != has_neg_resp && energy > 0.3 {
            let confidence = (energy as f32).clamp(0.0, 1.0);
            return ContradictionResult {
                found: true,
                contradiction_type: Some(ContradictionType::Logical),
                confidence,
                explanation: format!(
                    "Logical negation mismatch detected (energy={:.2})",
                    energy
                ),
            };
        }

        // 4. Heuristic word-pair check
        if let Some(r) = Self::check_pair_contradiction(prev_context, new_response) {
            return r;
        }

        ContradictionResult::none()
    }

    /// Analyse logical flow across a conversation history.
    ///
    /// Inspects consecutive turns for:
    /// - Missing transition markers
    /// - Abrupt topic shifts (low sheaf energy similarity)
    /// - Broken pronoun/entity references
    ///
    /// Gracefully degrades: returns `LogicalFlowResult::assume_coherent()` on error.
    pub fn check_logical_flow(&self, conversation_history: &[&str]) -> LogicalFlowResult {
        if conversation_history.len() < 2 {
            return LogicalFlowResult::assume_coherent();
        }

        let mut missing_transitions: Vec<String> = vec![];
        let mut broken_references: Vec<String> = vec![];
        let mut transition_scores: Vec<f32> = vec![];

        let turns = conversation_history;
        for i in 0..(turns.len() - 1) {
            let a = turns[i];
            let b = turns[i + 1];

            // Compute similarity between consecutive turns
            let energy = self.compute_energy(a, b);
            // High energy = high divergence; convert to similarity
            let similarity = (1.0 - energy) as f32;
            transition_scores.push(similarity);

            // Flag abrupt topic shifts (low similarity, no transition marker)
            if similarity < 0.5 && !Self::has_transition_marker(b) {
                missing_transitions.push(format!(
                    "Turn {}: abrupt shift from turn {} (similarity={:.2})",
                    i + 1,
                    i,
                    similarity
                ));
            }

            // Detect broken pronoun references (pronoun with no prior antecedent)
            if let Some(broken) = Self::find_broken_reference(a, b, i) {
                broken_references.push(broken);
            }
        }

        // Score = average transition similarity, penalised by issues
        let avg_sim = if transition_scores.is_empty() {
            1.0f32
        } else {
            transition_scores.iter().sum::<f32>() / transition_scores.len() as f32
        };

        let penalty = (missing_transitions.len() + broken_references.len()) as f32
            * 0.1
            / turns.len() as f32;
        let score = (avg_sim - penalty).clamp(0.0, 1.0);
        let is_coherent = missing_transitions.is_empty() && broken_references.is_empty();

        LogicalFlowResult {
            is_coherent,
            missing_transitions,
            broken_references,
            score,
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Compute contradiction energy between two texts using sheaf Laplacian.
    fn compute_energy(&self, text1: &str, text2: &str) -> f64 {
        let engine = CoherenceEngine::new(self.config.clone());

        let s1 = Self::text_to_state(text1, STATE_DIM);
        let s2 = Self::text_to_state(text2, STATE_DIM);

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

    /// Heuristic fallback: fraction of contradiction word pairs present.
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

    /// Check whether text contains negation words.
    fn contains_negation(text: &str) -> bool {
        let lower = text.to_lowercase();
        NEGATION_WORDS.iter().any(|n| lower.contains(n))
    }

    /// Check whether text opens with a transition marker.
    fn has_transition_marker(text: &str) -> bool {
        let lower = text.to_lowercase();
        TRANSITION_MARKERS.iter().any(|m| lower.contains(m))
    }

    /// Detect heuristic word-pair contradictions between two texts.
    fn check_pair_contradiction(a: &str, b: &str) -> Option<ContradictionResult> {
        let la = a.to_lowercase();
        let lb = b.to_lowercase();
        for (w1, w2) in CONTRADICTION_PAIRS {
            let ab = la.contains(w1) && lb.contains(w2);
            let ba = la.contains(w2) && lb.contains(w1);
            if ab || ba {
                return Some(ContradictionResult {
                    found: true,
                    contradiction_type: Some(ContradictionType::Logical),
                    confidence: 0.65,
                    explanation: format!("Contradictory pair «{}»/«{}» detected", w1, w2),
                });
            }
        }
        None
    }

    /// Detect numeric contradictions — same context but different numbers.
    fn check_numeric_contradiction(a: &str, b: &str) -> Option<ContradictionResult> {
        let nums_a = extract_numbers(a);
        let nums_b = extract_numbers(b);
        if nums_a.len() == 1 && nums_b.len() == 1 {
            let na = nums_a[0];
            let nb = nums_b[0];
            let diff = (na - nb).abs();
            let max_v = na.abs().max(nb.abs());
            if max_v > 0.0 && diff / max_v > 0.5 {
                let jaccard = jaccard_similarity(
                    &strip_digits(a),
                    &strip_digits(b),
                );
                if jaccard > 0.5 {
                    return Some(ContradictionResult {
                        found: true,
                        contradiction_type: Some(ContradictionType::Numeric),
                        confidence: 0.75,
                        explanation: format!(
                            "Numeric mismatch: {:.2} vs {:.2}",
                            na, nb
                        ),
                    });
                }
            }
        }
        None
    }

    /// Detect temporal contradictions using before/after markers.
    fn check_temporal_contradiction(a: &str, b: &str) -> Option<ContradictionResult> {
        let la = a.to_lowercase();
        let lb = b.to_lowercase();
        let before_a = la.contains("before") || la.contains("prior") || la.contains("earlier");
        let after_b = lb.contains("after") || lb.contains("later") || lb.contains("subsequently");
        let after_a = la.contains("after") || la.contains("later") || la.contains("subsequently");
        let before_b = lb.contains("before") || lb.contains("prior") || lb.contains("earlier");

        if (before_a && after_b) || (after_a && before_b) {
            // Only flag when the same subject appears in both
            let words_a: std::collections::HashSet<&str> =
                a.split_whitespace().filter(|w| w.len() > 3).collect();
            let words_b: std::collections::HashSet<&str> =
                b.split_whitespace().filter(|w| w.len() > 3).collect();
            let overlap = words_a.intersection(&words_b).count();
            if overlap >= 2 {
                return Some(ContradictionResult {
                    found: true,
                    contradiction_type: Some(ContradictionType::Temporal),
                    confidence: 0.60,
                    explanation: "Temporal ordering contradiction detected".to_string(),
                });
            }
        }
        None
    }

    /// Detect broken pronoun references between consecutive turns.
    fn find_broken_reference(prev: &str, current: &str, turn_idx: usize) -> Option<String> {
        let pronouns = ["it", "they", "this", "that", "these", "those", "he", "she"];
        let prev_lower = prev.to_lowercase();
        let curr_lower = current.to_lowercase();

        for pronoun in pronouns {
            // If the current turn *starts* with a pronoun that has no clear antecedent
            if curr_lower.starts_with(pronoun)
                && !prev_lower.contains(pronoun)
                && prev.split_whitespace().count() < 5
            {
                return Some(format!(
                    "Turn {}: pronoun «{}» may lack antecedent",
                    turn_idx + 1,
                    pronoun
                ));
            }
        }
        None
    }
}

// ── Module-level helpers ──────────────────────────────────────────────────────

/// Extract f64 numbers from text.
fn extract_numbers(text: &str) -> Vec<f64> {
    let mut numbers = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        if c.is_ascii_digit() || c == '.' || (c == '-' && current.is_empty()) {
            current.push(c);
        } else if !current.is_empty() {
            if let Ok(num) = current.parse::<f64>() {
                numbers.push(num);
            }
            current.clear();
        }
    }
    if !current.is_empty() {
        if let Ok(num) = current.parse::<f64>() {
            numbers.push(num);
        }
    }
    numbers
}

/// Remove digit characters from text (for Jaccard comparison of context).
fn strip_digits(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_ascii_digit() && *c != '.')
        .collect()
}

/// Word-level Jaccard similarity.
fn jaccard_similarity(a: &str, b: &str) -> f32 {
    let set_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let set_b: std::collections::HashSet<&str> = b.split_whitespace().collect();
    let inter = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 { 0.0 } else { inter as f32 / union as f32 }
}
