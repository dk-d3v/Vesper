//! Proof-carrying validation and witness chain recording.
//!
//! Uses `ruvector-verified` [`ProofEnvironment`] to track proof terms, and
//! SHAKE-256 (from the `sha3` crate) to hash each (response, context) pair
//! into a 32-byte witness that is stored in a local chain.

use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};

use ruvector_verified::ProofEnvironment;

use crate::{error::AiAssistantError, types::VerifiedResponse};

/// Proof-carrying verifier with an in-memory witness chain.
///
/// Each call to [`validate_and_record`] appends a new witness hash to the
/// internal chain. [`verify_witness`] checks whether a hash is recorded.
pub struct Verifier {
    /// Formal proof environment (tracks term IDs and cache).
    env: ProofEnvironment,
    /// Ordered list of SHAKE-256 witness hashes.
    witnesses: Vec<String>,
}

impl Verifier {
    /// Create a new verifier with a fresh proof environment.
    pub fn new() -> Result<Self, AiAssistantError> {
        Ok(Self {
            env: ProofEnvironment::new(),
            witnesses: Vec::new(),
        })
    }

    /// Validate a response and record it in the witness chain.
    ///
    /// 1. Computes SHAKE-256(`response_text` + `context`) → 32-byte hex digest.
    /// 2. Allocates a proof term in the [`ProofEnvironment`].
    /// 3. Appends the hash to the local witness chain.
    /// 4. Returns a [`VerifiedResponse`] containing the hash.
    pub fn validate_and_record(
        &mut self,
        response_text: &str,
        context: &str,
    ) -> Result<VerifiedResponse, AiAssistantError> {
        // Compute witness hash
        let hash = shake256_hex(
            format!("{response_text}\x00{context}").as_bytes(),
        );

        // Allocate a proof term (monotonically increasing ID)
        let _proof_id = self.env.alloc_term();

        // Append to witness chain
        self.witnesses.push(hash.clone());

        Ok(VerifiedResponse {
            text: response_text.to_string(),
            witness_hash: hash,
        })
    }

    /// Verify that a previously recorded witness hash exists in the chain.
    ///
    /// Returns `true` when the hash is found, `false` otherwise.
    pub fn verify_witness(&self, hash: &str) -> Result<bool, AiAssistantError> {
        Ok(self.witnesses.iter().any(|h| h == hash))
    }

    /// Return the total number of witnesses recorded.
    pub fn witness_count(&self) -> usize {
        self.witnesses.len()
    }

    /// Return proof environment statistics (for diagnostics / logging).
    pub fn proof_stats(&self) -> ruvector_verified::ProofStats {
        self.env.stats().clone()
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Compute SHAKE-256 of `data`, returning a 64-character lower-hex string
/// (32 bytes × 2 hex digits each).
pub(crate) fn shake256_hex(data: &[u8]) -> String {
    let mut hasher = Shake256::default();
    hasher.update(data);
    let mut reader = hasher.finalize_xof();
    let mut output = [0u8; 32];
    reader.read(&mut output);
    hex::encode(output)
}
