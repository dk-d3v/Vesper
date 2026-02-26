//! Immutable audit trail using SHAKE-256 hash chaining.
//!
//! Every recorded entry links to the previous one via its `prev_hash`,
//! forming an append-only chain. `verify_chain()` walks the entire chain
//! and returns `false` if any link is broken.
//!
//! # Invariants
//! - Entries are append-only; existing entries cannot be removed or modified.
//! - `last_hash()` always reflects the most recently recorded entry.
//! - An empty chain has a sentinel "genesis" hash of 64 zeroes.

use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};
use std::time::SystemTime;
use uuid::Uuid;

use crate::{error::AiAssistantError, types::AuditResult};

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

/// Append-only audit trail backed by a SHAKE-256 hash chain.
///
/// ```text
/// [genesis] ← [entry 0] ← [entry 1] ← … ← [entry N]
/// ```
pub struct AuditTrail {
    /// Ordered chain of audit entries (append-only).
    entries: Vec<AuditEntry>,
    /// Hash of the most recently recorded entry (or genesis sentinel).
    last_hash: String,
}

impl AuditTrail {
    /// Create an empty audit trail.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            last_hash: GENESIS_HASH.to_string(),
        }
    }

    /// Record a new event in the audit trail.
    ///
    /// Computes:
    /// - `data_hash`  = SHAKE-256(`data`)
    /// - `prev_hash`  = hash of the previous entry (or genesis)
    /// - `entry_hash` = SHAKE-256(`id | data_hash | prev_hash | type`)
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

    /// Compute SHAKE-256 of `data` with 256-bit (32-byte) output, hex-encoded.
    ///
    /// This is exposed as a public associated function so pipeline steps can
    /// hash arbitrary data consistently with the audit chain.
    pub fn shake256(data: &[u8]) -> String {
        shake256_hex(data)
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
}
