//! Shared types and data structures for the AI Assistant.

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Spoken/written language of user input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Language {
    Turkish,
    English,
    Other(String),
}

/// Episode tier for temporal compression in AgenticDB.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EpisodeTier {
    /// < 24 hours — stored at 8-bit precision
    Hot,
    /// < 7 days — stored at 5-7-bit precision
    Warm,
    /// Older — stored at 3-bit precision
    Cold,
}

/// Result of the coherence check (Step 5).
///
/// | Score       | Variant        | Action                                  |
/// |-------------|----------------|-----------------------------------------|
/// | `< 0.3`     | **Reflex**     | Pass through directly, no revision      |
/// | `0.3 – 0.8` | **Revised**    | Context revised/enriched before Claude   |
/// | `> 0.8`     | **Halt**       | Block the request entirely (loop/abuse)  |
#[derive(Debug, Clone)]
pub enum CoherenceResult {
    /// Contradiction energy < 0.3 → fast path, no revision needed (<1 ms)
    Reflex,
    /// 0.3 ≤ energy ≤ 0.8 → context revised/enriched (~10 ms)
    Revised(String),
    /// Energy > 0.8 → halt pipeline (potential loop/abuse)
    Halt,
}

/// Claude tool definition (for MCP tools passed to the API).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Tool call returned inside a Claude API response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// Tool result sent back to Claude in the next turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_use_id: String,
    pub content: String,
}

/// A validated user message ready for the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMessage {
    pub text: String,
    pub language: Language,
    pub timestamp: SystemTime,
}

/// A single memory episode stored in AgenticDB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub id: String,
    pub text: String,
    pub embedding: Vec<f32>,
    pub timestamp: SystemTime,
    pub tier: EpisodeTier,
    pub quality_score: f32,
}

/// Semantic context assembled in Step 2 (memory retrieval).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticContext {
    pub episodes: Vec<Episode>,
    pub skill_ids: Vec<String>,
}

/// Graph-RAG context assembled in Step 3.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphContext {
    pub rag_content: String,
    pub entity_count: usize,
    pub causal_edges: Vec<(String, String)>,
}

/// Fully assembled prompt ready to send to Claude (Step 4 output).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalPrompt {
    pub system: String,
    /// Cross-session memory episodes — injected into the *system* prompt
    /// under a `<memory>` tag so Claude can reference them as long-term recall.
    pub memory_context: String,
    /// High-quality past response patterns — injected under `<reasoning_hints>`
    /// to guide Claude toward proven response strategies.
    pub reasoning_hints: String,
    /// Current-session graph-RAG content — injected into the *user* message.
    pub context: String,
    pub user_text: String,
    pub estimated_tokens: usize,
}

impl FinalPrompt {
    /// Returns the system prompt with optional `<memory>` and
    /// `<reasoning_hints>` blocks appended.
    ///
    /// Both blocks are placed *after* all system instructions so that
    /// Claude reads the behavioural rules first.
    pub fn system_with_memory(&self) -> String {
        let mut out = self.system.clone();

        if !self.memory_context.is_empty() {
            out.push_str(&format!("\n\n<memory>\n{}\n</memory>", self.memory_context));
        }

        if !self.reasoning_hints.is_empty() {
            out.push_str(&format!(
                "\n\n<reasoning_hints>\n{}\n</reasoning_hints>",
                self.reasoning_hints
            ));
        }

        out
    }

    /// Returns the combined graph-RAG context + user text for the `user` role message.
    ///
    /// Memory episodes are **not** included here — they live in the system prompt.
    pub fn full_content(&self) -> String {
        if self.context.is_empty() {
            self.user_text.clone()
        } else {
            format!("Context:\n{}\n\nUser: {}", self.context, self.user_text)
        }
    }
}

/// Raw response received from the Claude API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeResponse {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub model: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Claude response after cryptographic witness verification (Step 7).
#[derive(Debug, Clone)]
pub struct VerifiedResponse {
    pub text: String,
    pub witness_hash: String,
}

/// Outcome of the online-learning update (Step 8).
#[derive(Debug, Clone)]
pub struct LearningResult {
    pub trajectory_id: String,
    pub quality_score: f32,
    pub pattern_count: usize,
}

/// Audit record written to the immutable log (Step 9).
#[derive(Debug, Clone)]
pub struct AuditResult {
    pub episode_id: String,
    pub hash: String,
}

/// All data produced by a single pipeline execution (one conversation turn).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub user_message: UserMessage,
    pub response_text: String,
    pub timestamp: SystemTime,
}

/// In-memory session state for the running process.
#[derive(Debug, Clone, Default)]
pub struct Session {
    pub turns: Vec<ConversationTurn>,
    pub turn_count: usize,
}

impl Session {
    /// Creates a new empty session.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a turn and trims the buffer to the last 50 entries.
    pub fn add_turn(&mut self, turn: ConversationTurn) {
        self.turn_count += 1;
        self.turns.push(turn);
        // Prevent unbounded memory growth
        if self.turns.len() > 50 {
            self.turns.remove(0);
        }
    }
}
