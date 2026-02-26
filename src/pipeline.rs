//! 10-step conversation pipeline orchestrator.
//!
//! Every conversation turn executes ALL 10 steps — none are optional.
//! Steps 3, 8, and 9 degrade gracefully (they log warnings and continue).

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
    /// Two `OnnxEmbedding` instances are created — one for `MemoryStore`
    /// (which takes ownership) and one for `GraphContextProvider` (Arc).
    pub async fn new(config: Config) -> Result<Self, AiAssistantError> {
        let embedding_memory = OnnxEmbedding::new(&config.embedding_model_path)?;
        let embedding_graph = Arc::new(OnnxEmbedding::new(&config.embedding_model_path)?);

        let memory = MemoryStore::new(embedding_memory)?;
        let graph = GraphContextProvider::new(embedding_graph)?;
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

    /// Execute a full 10-step conversation turn.
    ///
    /// Returns the final verified response text, or an error if a
    /// non-degradable step fails (e.g. CoherenceCritical, Claude API error).
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
            CoherenceResult::Critical => {
                let _ = self
                    .audit
                    .record("COHERENCE_CRITICAL_HALT", AuditEntryType::SecurityHalt);
                return Err(AiAssistantError::CoherenceCritical);
            }
        };

        // Step 6 — call Claude API (with optional tool use)
        let claude_resp = self.step6_claude_call(&prompt).await?;

        // Step 7 — post-response security / hallucination check
        let verified = self.step7_security_check(claude_resp, &prompt)?;

        // Step 8 — SONA learning record (graceful degradation)
        let _ = self.step8_learning(&msg, &prompt, &verified);

        // Step 9 — AgenticDB update + RVF audit (graceful degradation)
        let _ = self.step9_update_and_audit(&msg, &verified);

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
    /// Context is trimmed to stay under `MAX_TOKENS`.
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

        // Trim if over token limit
        let context = if estimated_tokens > MAX_TOKENS {
            let char_limit = MAX_TOKENS * 4;
            context.chars().take(char_limit).collect()
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

            // Add assistant's tool-call message
            let assistant_content = serde_json::to_string(&response)?;
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
    ) -> Result<AuditResult, AiAssistantError> {
        // Store episode in memory
        let _episode_id = self
            .memory
            .store_episode(&verified.text, 0.7)
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
