//! 10-step conversation pipeline orchestrator.
//!
//! Every conversation turn executes ALL 10 steps — none are optional.
//! Steps 3, 8, and 9 degrade gracefully (they log warnings and continue).
//!
//! # Integrations
//! - **ruvector-mincut** (ADIM 4): Smart context trimming using MinCut to
//!   remove least-connected chunks when over the token limit, instead of
//!   naive character truncation.

use std::sync::Arc;

use crate::{
    audit::{AuditTrail, AuditEntryType},
    claude_api::ClaudeClient,
    coherence::CoherenceChecker,
    config::{Config, MAX_TOKENS},
    error::AiAssistantError,
    graph_context::GraphContextProvider,
    language::{detect_language, load_system_prompt},
    learning::LearningEngine,
    mcp_tools::McpToolManager,
    memory::MemoryStore,
    types::*,
    verification::Verifier,
    embedding::OnnxEmbedding,
};

use ruvector_mincut::MinCutBuilder;

// ── Pipeline struct ───────────────────────────────────────────────────────────

/// Core pipeline holding all subsystem components.
pub struct Pipeline {
    config: Config,
    memory: MemoryStore,
    graph: GraphContextProvider,
    coherence: CoherenceChecker,
    claude: ClaudeClient,
    mcp: McpToolManager,
    verifier: Verifier,
    learning: LearningEngine,
    audit: AuditTrail,
    session: Session,
}

impl Pipeline {
    /// Initialise all components from `config`.
    ///
    /// A single `OnnxEmbedding` is loaded once and shared via `Arc` between
    /// `MemoryStore` and `GraphContextProvider`, avoiding a second ~500 MB load.
    pub async fn new(config: Config) -> Result<Self, AiAssistantError> {
        let embedding = Arc::new(OnnxEmbedding::new(&config.embedding_model_path)?);

        let memory = MemoryStore::new_with_arc(Arc::clone(&embedding))?;
        let graph = GraphContextProvider::new(Arc::clone(&embedding))?;
        let coherence = CoherenceChecker::new()?;
        let claude = ClaudeClient::new();
        let mcp = McpToolManager::load("mcp_servers.json")?;
        let verifier = Verifier::new()?;
        let learning = LearningEngine::new()?;
        let audit = AuditTrail::new();
        let session = Session::new();

        Ok(Self {
            config,
            memory,
            graph,
            coherence,
            claude,
            mcp,
            verifier,
            learning,
            audit,
            session,
        })
    }

    /// Execute a full 10-step conversation turn with SSE streaming.
    ///
    /// `on_token` is called for every text delta received from Claude.
    /// The pipeline still runs all 10 steps — steps 7-10 receive the
    /// fully assembled response text after the stream completes.
    pub async fn execute_turn_stream<F>(
        &mut self,
        raw_input: &str,
        on_token: F,
    ) -> Result<String, AiAssistantError>
    where
        F: Fn(&str) + Send + 'static,
    {
        // Step 1 — validate input and detect language
        let msg = self.step1_receive(raw_input)?;

        // Step 2 — semantic memory search
        let semantic = self.step2_semantic_search(&msg)?;

        // Step 3 — graph context (graceful degradation)
        let graph_ctx = self.step3_graph_context(&msg)?;

        // Step 4 — merge context and prepare prompt
        let prompt = self.step4_prepare_prompt(&msg, &semantic, &graph_ctx)?;

        // Step 5 — pre-API coherence check
        let coherence = self.step5_coherence_check(&prompt)?;
        let prompt = match coherence {
            CoherenceResult::Reflex => prompt,
            CoherenceResult::Revised(new_ctx) => FinalPrompt {
                context: new_ctx,
                ..prompt
            },
            CoherenceResult::Halt => {
                let _ = self
                    .audit
                    .record("COHERENCE_HALT", AuditEntryType::SecurityHalt);
                return Err(AiAssistantError::CoherenceHalt);
            }
        };

        // Step 6 — streaming Claude API call (token-by-token)
        let claude_resp = self.step6_claude_call_stream(&prompt, on_token).await?;

        // Step 7 — post-response security / hallucination check
        let verified = self.step7_security_check(claude_resp, &prompt)?;

        // Step 8 — SONA learning record (graceful degradation)
        let quality = self.step8_learning(&msg, &prompt, &verified)
            .map(|r| r.quality_score)
            .unwrap_or(0.7);

        // Step 9 — AgenticDB update + RVF audit (graceful degradation)
        let _ = self.step9_update_and_audit(&msg, &verified, quality);

        // Step 10 — return verified response, update session
        Ok(self.step10_return(msg, verified))
    }

    /// Execute a full 10-step conversation turn.
    ///
    /// Returns the final verified response text, or an error if a
    /// non-degradable step fails (e.g. CoherenceHalt, Claude API error).
    pub async fn execute_turn(&mut self, raw_input: &str) -> Result<String, AiAssistantError> {
        // Step 1 — validate input and detect language
        let msg = self.step1_receive(raw_input)?;

        // Step 2 — semantic memory search
        let semantic = self.step2_semantic_search(&msg)?;

        // Step 3 — graph context (graceful degradation)
        let graph_ctx = self.step3_graph_context(&msg)?;

        // Step 4 — merge context and prepare prompt
        let prompt = self.step4_prepare_prompt(&msg, &semantic, &graph_ctx)?;

        // Step 5 — pre-API coherence check
        let coherence = self.step5_coherence_check(&prompt)?;
        let prompt = match coherence {
            CoherenceResult::Reflex => prompt,
            CoherenceResult::Revised(new_ctx) => FinalPrompt {
                context: new_ctx,
                ..prompt
            },
            CoherenceResult::Halt => {
                let _ = self
                    .audit
                    .record("COHERENCE_HALT", AuditEntryType::SecurityHalt);
                return Err(AiAssistantError::CoherenceHalt);
            }
        };

        // Step 6 — call Claude API (with optional tool use)
        let claude_resp = self.step6_claude_call(&prompt).await?;

        // Step 7 — post-response security / hallucination check
        let verified = self.step7_security_check(claude_resp, &prompt)?;

        // Step 8 — SONA learning record (graceful degradation)
        let quality = self.step8_learning(&msg, &prompt, &verified)
            .map(|r| r.quality_score)
            .unwrap_or(0.7);

        // Step 9 — AgenticDB update + RVF audit (graceful degradation)
        let _ = self.step9_update_and_audit(&msg, &verified, quality);

        // Step 10 — return verified response, update session
        Ok(self.step10_return(msg, verified))
    }
}

// ── Private step implementations ──────────────────────────────────────────────

impl Pipeline {
    /// STEP 1 — Validate input, detect language.
    fn step1_receive(&self, raw_input: &str) -> Result<UserMessage, AiAssistantError> {
        if raw_input.trim().is_empty() {
            return Err(AiAssistantError::InputValidation(
                "Input cannot be empty".to_string(),
            ));
        }
        if raw_input.len() > crate::config::MAX_INPUT_LENGTH {
            return Err(AiAssistantError::InputValidation(
                "Input too long".to_string(),
            ));
        }
        let language = detect_language(raw_input);
        Ok(UserMessage {
            text: raw_input.trim().to_string(),
            language,
            timestamp: std::time::SystemTime::now(),
        })
    }

    /// STEP 2 — Semantic memory search.
    fn step2_semantic_search(
        &self,
        msg: &UserMessage,
    ) -> Result<SemanticContext, AiAssistantError> {
        self.memory.build_semantic_context(&msg.text)
    }

    /// STEP 3 — Graph context retrieval (graceful degradation).
    ///
    /// Returns empty [`GraphContext`] if graph has no relevant content or errors.
    fn step3_graph_context(&self, msg: &UserMessage) -> Result<GraphContext, AiAssistantError> {
        self.graph.get_context(&msg.text).or_else(|e| {
            tracing::warn!("Graph context failed (degrading gracefully): {}", e);
            Ok(GraphContext {
                rag_content: String::new(),
                entity_count: 0,
                causal_edges: vec![],
            })
        })
    }

    /// STEP 4 — Merge context and prepare prompt.
    ///
    /// Hot/warm/cold episodes appear first, then graph RAG content.
    /// Context is trimmed using MinCut-based intelligent pruning when over
    /// `MAX_TOKENS`. Falls back to simple `chars().take()` if MinCut fails.
    fn step4_prepare_prompt(
        &self,
        msg: &UserMessage,
        semantic: &SemanticContext,
        graph: &GraphContext,
    ) -> Result<FinalPrompt, AiAssistantError> {
        let system = load_system_prompt(&msg.language);

        let mut context_parts: Vec<String> = Vec::new();
        for ep in &semantic.episodes {
            context_parts.push(ep.text.clone());
        }
        if !graph.rag_content.is_empty() {
            context_parts.push(graph.rag_content.clone());
        }

        let context = context_parts.join("\n---\n");
        let estimated_tokens =
            context.len() / 4 + msg.text.len() / 4 + system.len() / 4;

        // Trim if over token limit — use MinCut-based smart trimming
        let context = if estimated_tokens > MAX_TOKENS {
            smart_context_trim(&context, MAX_TOKENS * 4)
        } else {
            context
        };

        Ok(FinalPrompt {
            system,
            context,
            user_text: msg.text.clone(),
            estimated_tokens,
        })
    }

    /// STEP 5 — Pre-API coherence check.
    fn step5_coherence_check(
        &self,
        prompt: &FinalPrompt,
    ) -> Result<CoherenceResult, AiAssistantError> {
        if prompt.context.is_empty() {
            return Ok(CoherenceResult::Reflex);
        }
        self.coherence.check_context(&prompt.context)
    }

    /// STEP 6 (streaming) — Call Claude API with SSE token callbacks.
    ///
    /// Uses `send_message_stream()` instead of `send_message()` so the REPL
    /// can display tokens as they arrive.  Tool-use is not supported in
    /// streaming mode; if the response would require tool calls the fallback
    /// `send_message()` path is taken automatically (degradation).
    async fn step6_claude_call_stream<F>(
        &self,
        prompt: &FinalPrompt,
        on_token: F,
    ) -> Result<ClaudeResponse, AiAssistantError>
    where
        F: Fn(&str) + Send + 'static,
    {
        let tools = self.mcp.get_tools();

        self.claude
            .send_message_stream(
                &self.config,
                &prompt.system,
                &prompt.full_content(),
                tools,
                on_token,
            )
            .await
    }

    /// STEP 6 — Call Claude API (with optional MCP tool use).
    ///
    /// Rust manages tool execution; Claude only generates text.
    async fn step6_claude_call(
        &self,
        prompt: &FinalPrompt,
    ) -> Result<ClaudeResponse, AiAssistantError> {
        let tools = self.mcp.get_tools();

        let mut response = self
            .claude
            .send_message(
                &self.config,
                &prompt.system,
                &prompt.full_content(),
                tools,
            )
            .await?;

        // Handle tool calls — Rust executes, Claude gets results
        if !response.tool_calls.is_empty() {
            let mut messages: Vec<serde_json::Value> = vec![serde_json::json!({
                "role": "user",
                "content": prompt.full_content()
            })];

            // Build assistant content array from tool calls (Anthropic API format)
            let assistant_content: Vec<serde_json::Value> = response
                .tool_calls
                .iter()
                .map(|tc| {
                    serde_json::json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input
                    })
                })
                .collect();
            messages.push(serde_json::json!({
                "role": "assistant",
                "content": assistant_content
            }));

            // Execute each tool call (Rust manages execution)
            for tool_call in &response.tool_calls {
                let result = self
                    .mcp
                    .execute_tool(tool_call)
                    .await
                    .unwrap_or_else(|e| ToolResult {
                        tool_use_id: tool_call.id.clone(),
                        content: format!("Tool error: {}", e),
                    });

                messages.push(serde_json::json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": result.tool_use_id,
                        "content": result.content
                    }]
                }));
            }

            // Re-call Claude with tool results
            response = self
                .claude
                .send_with_tool_results(&self.config, &prompt.system, &messages, tools)
                .await?;
        }

        Ok(response)
    }

    /// STEP 7 — Post-response security check and witness recording.
    ///
    /// Checks for hallucination (logs warning but does not halt).
    /// Records the response in the cryptographic witness chain.
    fn step7_security_check(
        &mut self,
        response: ClaudeResponse,
        prompt: &FinalPrompt,
    ) -> Result<VerifiedResponse, AiAssistantError> {
        let is_hallucination = self
            .coherence
            .check_response(&response.text, &prompt.context)
            .unwrap_or(false);

        if is_hallucination {
            tracing::warn!("Potential hallucination detected in response");
        }

        self.verifier
            .validate_and_record(&response.text, &prompt.context)
    }

    /// STEP 8 — SONA learning record (graceful degradation).
    fn step8_learning(
        &mut self,
        msg: &UserMessage,
        prompt: &FinalPrompt,
        verified: &VerifiedResponse,
    ) -> Result<LearningResult, AiAssistantError> {
        let result = self.learning.record_trajectory(
            &msg.text,
            &prompt.context,
            &prompt.full_content(),
            &verified.text,
        );

        match result {
            Ok(r) => {
                // Periodically run explicit pattern extraction
                if r.trajectory_id.ends_with('0') {
                    let _ = self.learning.find_patterns();
                }
                Ok(r)
            }
            Err(e) => {
                tracing::warn!("SONA learning failed (degrading gracefully): {}", e);
                Ok(LearningResult {
                    trajectory_id: "skipped".to_string(),
                    quality_score: 0.0,
                    pattern_count: 0,
                })
            }
        }
    }

    /// STEP 9 — AgenticDB update + RVF audit trail (graceful degradation).
    fn step9_update_and_audit(
        &mut self,
        msg: &UserMessage,
        verified: &VerifiedResponse,
        quality_score: f32,
    ) -> Result<AuditResult, AiAssistantError> {
        // Store episode in memory
        let _episode_id = self
            .memory
            .store_episode(&verified.text, quality_score)
            .unwrap_or_else(|e| {
                tracing::warn!("Episode storage failed: {}", e);
                "unknown".to_string()
            });

        // Add causal edge
        let _ = self.memory.add_causal_edge(&msg.text, &verified.text);

        // Auto-consolidate low-quality episodes
        let _ = self.memory.auto_consolidate();

        // Add conversation turn to graph context
        let doc = format!("Q: {} A: {}", msg.text, verified.text);
        let _ = self.graph.add_document(&doc, "conversation");

        // RVF audit trail
        let audit_data = format!(
            "turn|{}|{}|{}",
            msg.text, verified.text, verified.witness_hash
        );
        self.audit
            .record(&audit_data, AuditEntryType::ConversationTurn)
    }

    /// STEP 10 — Return verified response and update session.
    fn step10_return(&mut self, msg: UserMessage, verified: VerifiedResponse) -> String {
        self.session.add_turn(ConversationTurn {
            user_message: msg,
            response_text: verified.text.clone(),
            timestamp: std::time::SystemTime::now(),
        });
        verified.text
    }
}

// ── MinCut-based smart context trimming (ADIM 4) ─────────────────────────────

/// Intelligently trim context to fit within `char_limit` using MinCut.
///
/// Builds a graph where context chunks are nodes and edges represent
/// semantic similarity (word overlap). Uses MinCut to identify the
/// least-connected chunks and removes them first, preserving the most
/// coherent subgraph.
///
/// **Graceful degradation**: Falls back to `chars().take()` on failure.
fn smart_context_trim(context: &str, char_limit: usize) -> String {
    if context.len() <= char_limit {
        return context.to_string();
    }

    // Try MinCut-based trimming
    match mincut_trim(context, char_limit) {
        Ok(trimmed) => trimmed,
        Err(e) => {
            tracing::warn!(
                "MinCut smart trimming failed (falling back to char truncation): {}",
                e
            );
            context.chars().take(char_limit).collect()
        }
    }
}

/// Core MinCut trimming: splits context into chunks, builds a similarity
/// graph, and iteratively removes the least-connected chunk until the
/// total length fits within the character limit.
fn mincut_trim(context: &str, char_limit: usize) -> Result<String, String> {
    // Split into chunks (by separator or by paragraphs)
    let chunks: Vec<&str> = if context.contains("\n---\n") {
        context.split("\n---\n").collect()
    } else {
        context.split("\n\n").collect()
    };

    if chunks.len() < 2 {
        // Single chunk: just truncate
        return Ok(context.chars().take(char_limit).collect());
    }

    // Build word sets for similarity computation
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

    // Build edge list for MinCut
    let n = chunks.len();
    let mut edges: Vec<(u64, u64, f64)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let overlap = word_sets[i].intersection(&word_sets[j]).count();
            let max_len = word_sets[i].len().max(word_sets[j].len()).max(1);
            let weight = overlap as f64 / max_len as f64;
            if weight > 0.02 {
                edges.push((i as u64, j as u64, weight));
            }
        }
    }

    if edges.is_empty() {
        // No edges: keep first chunks that fit
        return Ok(greedy_char_trim(&chunks, char_limit));
    }

    // Build MinCut and get partition
    let mincut = MinCutBuilder::new()
        .exact()
        .with_edges(edges)
        .build()
        .map_err(|e| format!("MinCut build for trimming failed: {}", e))?;

    let result = mincut.min_cut();

    // Use partition to determine which side to keep (keep larger side)
    let mut kept_indices: Vec<usize> = if let Some((side_a, side_b)) = result.partition {
        // Compute total char length of each side
        let len_a: usize = side_a.iter().map(|&i| chunks.get(i as usize).map_or(0, |c| c.len())).sum();
        let len_b: usize = side_b.iter().map(|&i| chunks.get(i as usize).map_or(0, |c| c.len())).sum();

        // Prefer the side that fits within the limit; if both fit, keep larger
        let keep_side = if len_a <= char_limit && len_b <= char_limit {
            if len_a >= len_b { &side_a } else { &side_b }
        } else if len_a <= char_limit {
            &side_a
        } else if len_b <= char_limit {
            &side_b
        } else {
            // Neither side fits alone; keep the smaller one and truncate
            if len_a <= len_b { &side_a } else { &side_b }
        };

        keep_side.iter().map(|&i| i as usize).collect()
    } else {
        // No partition: keep all and truncate
        (0..n).collect()
    };

    kept_indices.sort_unstable();

    // Assemble kept chunks
    let kept: Vec<&str> = kept_indices
        .iter()
        .filter_map(|&i| chunks.get(i))
        .copied()
        .collect();

    let separator = if context.contains("\n---\n") {
        "\n---\n"
    } else {
        "\n\n"
    };
    let result_text = kept.join(separator);

    // Final safety: ensure we're within limit
    if result_text.len() <= char_limit {
        Ok(result_text)
    } else {
        Ok(result_text.chars().take(char_limit).collect())
    }
}

/// Simple greedy trim: keep first chunks that fit within char_limit.
fn greedy_char_trim(chunks: &[&str], char_limit: usize) -> String {
    let mut result = String::new();
    let mut total = 0usize;
    for chunk in chunks {
        let needed = if result.is_empty() {
            chunk.len()
        } else {
            chunk.len() + 5 // "\n---\n" separator
        };
        if total + needed > char_limit {
            break;
        }
        if !result.is_empty() {
            result.push_str("\n---\n");
        }
        result.push_str(chunk);
        total += needed;
    }
    if result.is_empty() && !chunks.is_empty() {
        // At least include truncated first chunk
        return chunks[0].chars().take(char_limit).collect();
    }
    result
}
