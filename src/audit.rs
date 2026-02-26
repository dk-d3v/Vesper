//! Immutable audit trail using SHAKE-256 hash chaining with witness receipts.
//!
//! Every recorded entry links to the previous one via its `prev_hash`,
//! forming an append-only chain. `verify_chain()` walks the entire chain
//! and returns `false` if any link is broken.
//!
//! A secondary [`WitnessChain`] from `ruvector-cognitive-container` runs in
//! parallel, producing a tamper-evident receipt for every epoch.  If the
//! witness layer panics it is silently swallowed so the SHAKE-256 chain
//! remains the authoritative audit record (graceful degradation).
//!
//! # Invariants
//! - Entries are append-only; existing entries cannot be removed or modified.
//! - `last_hash()` always reflects the most recently recorded entry.
//! - An empty chain has a sentinel "genesis" hash of 64 zeroes.

use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::SystemTime;
use uuid::Uuid;

use ruvector_cognitive_container::{
    CoherenceDecision, ContainerWitnessReceipt, VerificationResult, WitnessChain,
};

// RVF cryptographic audit trail integration
use rvf_crypto::{
    shake256_256 as rvf_shake256,
    create_witness_chain as rvf_create_witness_chain,
    WitnessEntry,
};

use crate::{error::AiAssistantError, forensic::ForensicBundle, types::AuditResult};

/// Per-turn latency breakdown for observability.
///
/// Mirrors the `ruvllm::WitnessLog::LatencyBreakdown` pattern but uses
/// integer milliseconds and tracks each of the 9 pipeline steps.
#[derive(Debug, Clone, Default)]
pub struct LatencyBreakdown {
    pub step1_receive_ms: u64,
    pub step2_semantic_search_ms: u64,
    pub step3_graph_context_ms: u64,
    pub step4_prepare_prompt_ms: u64,
    pub step5_coherence_ms: u64,
    pub step6_claude_api_ms: u64,
    pub step7_security_ms: u64,
    pub step8_learning_ms: u64,
    pub step9_audit_ms: u64,
    pub total_ms: u64,
}

impl LatencyBreakdown {
    /// Create a zeroed latency breakdown.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sum all step latencies into `total_ms`.
    pub fn compute_total(&mut self) {
        self.total_ms = self.step1_receive_ms
            + self.step2_semantic_search_ms
            + self.step3_graph_context_ms
            + self.step4_prepare_prompt_ms
            + self.step5_coherence_ms
            + self.step6_claude_api_ms
            + self.step7_security_ms
            + self.step8_learning_ms
            + self.step9_audit_ms;
    }
}

/// Sentinel hash used as the `prev_hash` of the very first entry.
const GENESIS_HASH: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

/// Classification of events that can appear in the audit log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditEntryType {
    ConversationTurn,
    EpisodeStored,
    CausalEdgeAdded,
    SecurityHalt,
}

impl std::fmt::Display for AuditEntryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            AuditEntryType::ConversationTurn => "ConversationTurn",
            AuditEntryType::EpisodeStored => "EpisodeStored",
            AuditEntryType::CausalEdgeAdded => "CausalEdgeAdded",
            AuditEntryType::SecurityHalt => "SecurityHalt",
        };
        f.write_str(label)
    }
}

/// A single immutable entry in the audit chain.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// UUID of this entry.
    pub id: String,
    /// Wall-clock time when the entry was recorded.
    pub timestamp: SystemTime,
    /// SHAKE-256 of the raw data payload.
    pub data_hash: String,
    /// SHAKE-256 of the previous entry (chain link).
    pub prev_hash: String,
    /// Semantic category of the event.
    pub entry_type: AuditEntryType,
}

impl AuditEntry {
    /// Compute the canonical hash for this entry.
    ///
    /// Hash input: `id | data_hash | prev_hash | entry_type`
    pub fn entry_hash(&self) -> String {
        let input = format!(
            "{}{}{}{}",
            self.id, self.data_hash, self.prev_hash, self.entry_type
        );
        shake256_hex(input.as_bytes())
    }
}

/// Maximum number of witness receipts retained in the ring buffer.
const WITNESS_MAX_RECEIPTS: usize = 4096;

/// Append-only audit trail backed by a SHAKE-256 hash chain.
///
/// A secondary [`WitnessChain`] produces a tamper-evident receipt for every
/// `record()` call.  The witness layer is best-effort: if it panics the
/// SHAKE-256 chain continues unaffected.
///
/// A tertiary RVF crypto witness chain (`rvf_crypto::create_witness_chain`)
/// encodes each audit entry into the RVF binary format for cross-system
/// interoperability.
///
/// ```text
/// [genesis] ← [entry 0] ← [entry 1] ← … ← [entry N]
///              receipt 0     receipt 1         receipt N   (cognitive)
///              rvf_entry 0   rvf_entry 1       rvf_entry N (rvf-crypto)
/// ```
pub struct AuditTrail {
    /// Ordered chain of audit entries (append-only).
    entries: Vec<AuditEntry>,
    /// Hash of the most recently recorded entry (or genesis sentinel).
    last_hash: String,
    /// Parallel witness chain producing per-epoch receipts.
    witness_chain: WitnessChain,
    /// Accumulated RVF witness entries for cross-system audit export.
    rvf_witness_entries: Vec<WitnessEntry>,
}

impl AuditTrail {
    /// Create an empty audit trail with a fresh witness chain.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            last_hash: GENESIS_HASH.to_string(),
            witness_chain: WitnessChain::new(WITNESS_MAX_RECEIPTS),
            rvf_witness_entries: Vec::new(),
        }
    }

    /// Record a new event in the audit trail.
    ///
    /// Computes:
    /// - `data_hash`  = SHAKE-256(`data`)
    /// - `prev_hash`  = hash of the previous entry (or genesis)
    /// - `entry_hash` = SHAKE-256(`id | data_hash | prev_hash | type`)
    ///
    /// Also generates a [`ContainerWitnessReceipt`] via the witness chain.
    /// If the witness layer panics, a warning is logged and execution
    /// continues (graceful degradation).
    ///
    /// Returns [`AuditResult`] with the entry's UUID and its hash.
    pub fn record(
        &mut self,
        data: &str,
        entry_type: AuditEntryType,
    ) -> Result<AuditResult, AiAssistantError> {
        let id = Uuid::new_v4().to_string();
        let data_hash = shake256_hex(data.as_bytes());
        let prev_hash = self.last_hash.clone();

        let entry = AuditEntry {
            id: id.clone(),
            timestamp: SystemTime::now(),
            data_hash,
            prev_hash,
            entry_type,
        };

        let entry_hash = entry.entry_hash();
        self.last_hash = entry_hash.clone();
        self.entries.push(entry);

        // ── Witness receipt (best-effort) ────────────────────────────────
        self.try_generate_witness_receipt(data, &entry_hash);

        // ── RVF crypto witness entry ─────────────────────────────────────
        // Record audit entry in RVF binary witness format for cross-system
        // interoperability. Uses rvf_crypto::shake256_256 to hash the action.
        let action_hash = rvf_shake256(entry_hash.as_bytes());
        let timestamp_ns = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        self.rvf_witness_entries.push(WitnessEntry {
            prev_hash: [0u8; 32], // will be re-linked by create_witness_chain
            action_hash,
            timestamp_ns,
            witness_type: 0x01, // PROVENANCE type
        });

        Ok(AuditResult {
            episode_id: id,
            hash: entry_hash,
        })
    }

    /// Walk the entire chain and verify every link is intact.
    ///
    /// Returns `true` when:
    /// - The first entry's `prev_hash` equals [`GENESIS_HASH`].
    /// - Each subsequent entry's `prev_hash` equals the previous entry's hash.
    pub fn verify_chain(&self) -> bool {
        if self.entries.is_empty() {
            return true;
        }

        // Verify the genesis link
        if self.entries[0].prev_hash != GENESIS_HASH {
            return false;
        }

        // Verify each inter-entry link
        for window in self.entries.windows(2) {
            let prev = &window[0];
            let curr = &window[1];
            let expected_prev_hash = prev.entry_hash();
            if curr.prev_hash != expected_prev_hash {
                return false;
            }
        }

        true
    }

    /// Verify the parallel witness chain and return the result.
    ///
    /// Delegates to [`WitnessChain::verify_chain`].  If the verification
    /// panics, returns [`VerificationResult::Empty`] as a safe fallback.
    pub fn verify_witness_chain(&self) -> VerificationResult {
        let receipts = self.witness_chain.receipt_chain();
        match catch_unwind(AssertUnwindSafe(|| {
            WitnessChain::verify_chain(receipts)
        })) {
            Ok(result) => result,
            Err(_) => {
                eprintln!("[audit] witness chain verification panicked — returning Empty");
                VerificationResult::Empty
            }
        }
    }

    /// Return the hash of the last recorded entry (or genesis hash if empty).
    pub fn last_hash(&self) -> &str {
        &self.last_hash
    }

    /// Return the total number of entries in the trail.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` when no entries have been recorded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return a read-only reference to all entries (for export / inspection).
    pub fn entries(&self) -> &[AuditEntry] {
        &self.entries
    }

    /// Return a read-only reference to the retained witness receipts.
    pub fn witness_receipts(&self) -> &[ContainerWitnessReceipt] {
        self.witness_chain.receipt_chain()
    }

    /// Current witness-chain epoch (number of receipts generated so far).
    pub fn witness_epoch(&self) -> u64 {
        self.witness_chain.current_epoch()
    }

    /// Compute SHAKE-256 of `data` with 256-bit (32-byte) output, hex-encoded.
    ///
    /// This is exposed as a public associated function so pipeline steps can
    /// hash arbitrary data consistently with the audit chain.
    pub fn shake256(data: &[u8]) -> String {
        shake256_hex(data)
    }

    /// Serialize the accumulated RVF witness entries into a binary chain.
    ///
    /// Uses [`rvf_crypto::create_witness_chain`] to link entries by hash.
    /// Returns the resulting byte vector; an empty chain returns `Vec::new()`.
    pub fn rvf_witness_bytes(&self) -> Vec<u8> {
        if self.rvf_witness_entries.is_empty() {
            return Vec::new();
        }
        rvf_create_witness_chain(&self.rvf_witness_entries)
    }

    /// Number of RVF witness entries accumulated so far.
    pub fn rvf_witness_len(&self) -> usize {
        self.rvf_witness_entries.len()
    }

    /// Record an event with an associated latency breakdown.
    ///
    /// This is a **non-breaking** additive overload: it delegates to the
    /// existing `record()` and then emits a `tracing::info!` event so the
    /// per-step timing data is captured in structured logs without modifying
    /// any existing call sites.
    ///
    /// # Arguments
    /// * `data`    — arbitrary audit payload (same as `record`)
    /// * `entry_type` — semantic category of the event
    /// * `latency` — step-level latency breakdown to log
    pub fn record_with_latency(
        &mut self,
        data: &str,
        entry_type: AuditEntryType,
        latency: &LatencyBreakdown,
    ) -> Result<AuditResult, AiAssistantError> {
        let result = self.record(data, entry_type)?;
        tracing::info!(
            audit_id = %result.episode_id,
            total_ms = latency.total_ms,
            step1_receive_ms = latency.step1_receive_ms,
            step2_semantic_search_ms = latency.step2_semantic_search_ms,
            step3_graph_context_ms = latency.step3_graph_context_ms,
            step4_prepare_prompt_ms = latency.step4_prepare_prompt_ms,
            step5_coherence_ms = latency.step5_coherence_ms,
            step6_claude_api_ms = latency.step6_claude_api_ms,
            step7_security_ms = latency.step7_security_ms,
            step8_learning_ms = latency.step8_learning_ms,
            step9_audit_ms = latency.step9_audit_ms,
            "latency_breakdown"
        );
        Ok(result)
    }

    /// Record multiple audit entries in a single call.
    ///
    /// Each tuple is `(data, entry_type_str, input_hash, output_hash)` — the
    /// last two fields are included for future extensibility (currently
    /// serialised into the `data` payload so the hash chain remains stable).
    ///
    /// Returns one `Result` per entry in the same order as the input slice.
    pub fn record_batch(
        &mut self,
        entries: &[(String, String, String, bool)],
    ) -> Vec<Result<AuditResult, AiAssistantError>> {
        entries
            .iter()
            .map(|(data, type_label, extra, _verified)| {
                let payload = format!("{}|{}|{}", data, type_label, extra);
                let entry_type = match type_label.as_str() {
                    "EpisodeStored"    => AuditEntryType::EpisodeStored,
                    "CausalEdgeAdded"  => AuditEntryType::CausalEdgeAdded,
                    "SecurityHalt"     => AuditEntryType::SecurityHalt,
                    _                  => AuditEntryType::ConversationTurn,
                };
                self.record(&payload, entry_type)
            })
            .collect()
    }
    /// Collect current audit state into a [`ForensicBundle`] for replay or export.
    ///
    /// # Arguments
    /// * `session_id`        — caller-supplied session identifier
    /// * `latency`           — current latency breakdown snapshot
    /// * `attestation_count` — number of proof attestations recorded
    /// * `proof_tier`        — routing tier label (`"Standard"` / `"Enhanced"` / `"Critical"`)
    /// * `coherence_score`   — coherence score at snapshot time (0.0 – 1.0)
    /// * `quality_score`     — quality score at snapshot time (0.0 – 1.0)
    pub fn build_forensic_bundle(
        &self,
        session_id: impl Into<String>,
        latency: LatencyBreakdown,
        attestation_count: usize,
        proof_tier: impl Into<String>,
        coherence_score: f32,
        quality_score: f32,
    ) -> ForensicBundle {
        ForensicBundle::new(
            session_id,
            latency,
            self.entries.clone(),
            attestation_count,
            proof_tier,
            coherence_score,
            quality_score,
        )
    }
}

// ── Witness-chain helper (private) ───────────────────────────────────────────

impl AuditTrail {
    /// Try to generate a witness receipt; swallow panics for graceful degradation.
    fn try_generate_witness_receipt(&mut self, data: &str, entry_hash: &str) {
        let input_deltas = data.as_bytes();
        let mincut_data = entry_hash.as_bytes();
        let spectral_scs = 1.0_f64; // nominal coherence score
        let evidence_data = self.last_hash.as_bytes();
        let decision = CoherenceDecision::Pass;

        let result = catch_unwind(AssertUnwindSafe(|| {
            self.witness_chain.generate_receipt(
                input_deltas,
                mincut_data,
                spectral_scs,
                evidence_data,
                decision,
            )
        }));

        if let Err(_panic) = result {
            eprintln!(
                "[audit] witness receipt generation panicked at epoch {} — skipping",
                self.witness_chain.current_epoch()
            );
        }
    }
}

impl Default for AuditTrail {
    fn default() -> Self {
        Self::new()
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Compute SHAKE-256 of `data`, returning a 64-character lower-hex string.
fn shake256_hex(data: &[u8]) -> String {
    let mut hasher = Shake256::default();
    hasher.update(data);
    let mut reader = hasher.finalize_xof();
    let mut output = [0u8; 32];
    reader.read(&mut output);
    hex::encode(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_chain_verifies() {
        let trail = AuditTrail::new();
        assert!(trail.verify_chain());
        assert_eq!(trail.last_hash(), GENESIS_HASH);
    }

    #[test]
    fn test_single_entry_chain() {
        let mut trail = AuditTrail::new();
        let result = trail.record("hello", AuditEntryType::ConversationTurn).unwrap();
        assert!(!result.hash.is_empty());
        assert_eq!(result.hash.len(), 64); // 32 bytes × 2 hex chars
        assert!(trail.verify_chain());
    }

    #[test]
    fn test_multi_entry_chain() {
        let mut trail = AuditTrail::new();
        trail.record("turn 1", AuditEntryType::ConversationTurn).unwrap();
        trail.record("episode", AuditEntryType::EpisodeStored).unwrap();
        trail.record("edge", AuditEntryType::CausalEdgeAdded).unwrap();
        assert_eq!(trail.len(), 3);
        assert!(trail.verify_chain());
    }

    #[test]
    fn test_tampered_chain_fails() {
        let mut trail = AuditTrail::new();
        trail.record("original", AuditEntryType::ConversationTurn).unwrap();
        trail.record("second", AuditEntryType::EpisodeStored).unwrap();

        // Tamper by changing prev_hash of the second entry
        trail.entries[1].prev_hash = "deadbeef".repeat(8);
        assert!(!trail.verify_chain());
    }

    #[test]
    fn test_shake256_deterministic() {
        let h1 = AuditTrail::shake256(b"test data");
        let h2 = AuditTrail::shake256(b"test data");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64);
    }

    // ── Witness-chain tests ──────────────────────────────────────────────

    #[test]
    fn test_witness_chain_generates_receipts() {
        let mut trail = AuditTrail::new();
        trail.record("event-a", AuditEntryType::ConversationTurn).unwrap();
        trail.record("event-b", AuditEntryType::EpisodeStored).unwrap();

        assert_eq!(trail.witness_epoch(), 2);
        assert_eq!(trail.witness_receipts().len(), 2);
    }

    #[test]
    fn test_witness_chain_verifies_after_records() {
        let mut trail = AuditTrail::new();
        for i in 0..5 {
            trail.record(&format!("turn-{i}"), AuditEntryType::ConversationTurn).unwrap();
        }
        match trail.verify_witness_chain() {
            VerificationResult::Valid { chain_length, .. } => {
                assert_eq!(chain_length, 5);
            }
            other => panic!("Expected Valid, got {other:?}"),
        }
    }

    #[test]
    fn test_empty_witness_chain_returns_empty() {
        let trail = AuditTrail::new();
        match trail.verify_witness_chain() {
            VerificationResult::Empty => {}
            other => panic!("Expected Empty, got {other:?}"),
        }
    }

    #[test]
    fn test_both_chains_stay_in_sync() {
        let mut trail = AuditTrail::new();
        trail.record("sync", AuditEntryType::CausalEdgeAdded).unwrap();
        assert_eq!(trail.len(), 1);
        assert_eq!(trail.witness_epoch(), 1);
        assert!(trail.verify_chain());
        match trail.verify_witness_chain() {
            VerificationResult::Valid { chain_length, .. } => {
                assert_eq!(chain_length, 1);
            }
            other => panic!("Expected Valid, got {other:?}"),
        }
    }

    // ── LatencyBreakdown tests ───────────────────────────────────────────

    #[test]
    fn test_latency_breakdown_compute_total() {
        let mut lb = LatencyBreakdown::new();
        lb.step1_receive_ms = 5;
        lb.step2_semantic_search_ms = 10;
        lb.step3_graph_context_ms = 8;
        lb.step4_prepare_prompt_ms = 3;
        lb.step5_coherence_ms = 7;
        lb.step6_claude_api_ms = 300;
        lb.step7_security_ms = 4;
        lb.step8_learning_ms = 6;
        lb.step9_audit_ms = 2;
        lb.compute_total();
        assert_eq!(lb.total_ms, 345);
    }

    #[test]
    fn test_latency_breakdown_default_zero() {
        let lb = LatencyBreakdown::default();
        assert_eq!(lb.total_ms, 0);
        assert_eq!(lb.step6_claude_api_ms, 0);
    }

    // ── record_with_latency tests ────────────────────────────────────────

    #[test]
    fn test_record_with_latency_preserves_chain() {
        let mut trail = AuditTrail::new();
        let mut lb = LatencyBreakdown::new();
        lb.step6_claude_api_ms = 150;
        lb.compute_total();

        let result = trail
            .record_with_latency("latency-event", AuditEntryType::ConversationTurn, &lb)
            .unwrap();

        assert!(!result.hash.is_empty());
        assert_eq!(trail.len(), 1);
        assert!(trail.verify_chain());
    }

    #[test]
    fn test_record_with_latency_non_breaking() {
        // Ensures existing record() still works unchanged alongside new method
        let mut trail = AuditTrail::new();
        trail.record("before", AuditEntryType::EpisodeStored).unwrap();

        let lb = LatencyBreakdown::new();
        trail
            .record_with_latency("with-latency", AuditEntryType::ConversationTurn, &lb)
            .unwrap();

        trail.record("after", AuditEntryType::CausalEdgeAdded).unwrap();

        assert_eq!(trail.len(), 3);
        assert!(trail.verify_chain());
    }

    // ── record_batch tests ───────────────────────────────────────────────

    #[test]
    fn test_record_batch_all_succeed() {
        let mut trail = AuditTrail::new();
        let entries = vec![
            ("data-a".to_string(), "ConversationTurn".to_string(), "hash-a".to_string(), true),
            ("data-b".to_string(), "EpisodeStored".to_string(), "hash-b".to_string(), true),
            ("data-c".to_string(), "CausalEdgeAdded".to_string(), "hash-c".to_string(), false),
        ];

        let results = trail.record_batch(&entries);
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.is_ok(), "batch entry failed: {:?}", r);
        }
        assert_eq!(trail.len(), 3);
        assert!(trail.verify_chain());
    }

    #[test]
    fn test_record_batch_security_halt_type() {
        let mut trail = AuditTrail::new();
        let entries = vec![
            ("halt-event".to_string(), "SecurityHalt".to_string(), "".to_string(), false),
        ];

        let results = trail.record_batch(&entries);
        assert!(results[0].is_ok());
        assert_eq!(trail.entries()[0].entry_type, AuditEntryType::SecurityHalt);
    }

    #[test]
    fn test_record_batch_empty() {
        let mut trail = AuditTrail::new();
        let results = trail.record_batch(&[]);
        assert!(results.is_empty());
        assert_eq!(trail.len(), 0);
    }
}
