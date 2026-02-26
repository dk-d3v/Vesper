//! Forensic bundle for post-incident audit replay.
//!
//! A [`ForensicBundle`] is a point-in-time snapshot of a session's audit
//! state.  It is produced by [`crate::audit::AuditTrail::build_forensic_bundle`]
//! and can be passed to [`replay_audit`] to produce a human-readable timeline.

use std::time::SystemTime;

use crate::audit::{AuditEntry, LatencyBreakdown};

// ── ForensicBundle ────────────────────────────────────────────────────────────

/// A point-in-time snapshot of session state suitable for forensic replay.
///
/// Immutable once constructed — clone the inner vectors to prevent accidental
/// mutation of live audit state.
#[derive(Debug, Clone)]
pub struct ForensicBundle {
    /// UUID-like unique session identifier.
    pub session_id: String,
    /// Unix timestamp in milliseconds at bundle creation time.
    pub timestamp_ms: u64,
    /// Per-step latency breakdown captured at snapshot time.
    pub latency: LatencyBreakdown,
    /// Clone of all audit entries from the [`AuditTrail`] at snapshot time.
    pub audit_entries: Vec<AuditEntry>,
    /// Number of cryptographic attestations recorded in the session.
    pub attestation_count: usize,
    /// Routing tier label: `"Standard"`, `"Enhanced"`, or `"Critical"`.
    pub proof_tier: String,
    /// Coherence score at snapshot time (0.0 – 1.0).
    pub coherence_score: f32,
    /// Response quality score at snapshot time (0.0 – 1.0).
    pub quality_score: f32,
}

impl ForensicBundle {
    /// Construct a [`ForensicBundle`] from explicit fields.
    ///
    /// `audit_entries` should be a clone of the live audit trail entries.
    /// `proof_tier` should be the `Display` string of a `ProofTier` value.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        session_id: impl Into<String>,
        latency: LatencyBreakdown,
        audit_entries: Vec<AuditEntry>,
        attestation_count: usize,
        proof_tier: impl Into<String>,
        coherence_score: f32,
        quality_score: f32,
    ) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            session_id: session_id.into(),
            timestamp_ms,
            latency,
            audit_entries,
            attestation_count,
            proof_tier: proof_tier.into(),
            coherence_score,
            quality_score,
        }
    }
}

// ── replay_audit ──────────────────────────────────────────────────────────────

/// Produce a human-readable replay of a [`ForensicBundle`].
///
/// Returns one line per audit entry (index, type, truncated hash) followed
/// by a single summary line.  The output is deterministic — suitable for
/// snapshot tests or logging.
///
/// # Example output
/// ```text
/// [0] ConversationTurn  hash=a3b4c5d6…
/// [1] EpisodeStored     hash=deadbeef…
/// SUMMARY session=<id> entries=2 tier=Enhanced quality=0.75 coherence=0.80 latency_ms=345
/// ```
pub fn replay_audit(bundle: &ForensicBundle) -> Vec<String> {
    let mut lines: Vec<String> = bundle
        .audit_entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let short_hash = &entry.data_hash[..8.min(entry.data_hash.len())];
            format!(
                "[{i}] {entry_type:<20} hash={short_hash}…",
                entry_type = entry.entry_type.to_string(),
            )
        })
        .collect();

    lines.push(format!(
        "SUMMARY session={session} entries={entries} tier={tier} \
         quality={quality:.2} coherence={coherence:.2} latency_ms={latency}",
        session = bundle.session_id,
        entries = bundle.audit_entries.len(),
        tier = bundle.proof_tier,
        quality = bundle.quality_score,
        coherence = bundle.coherence_score,
        latency = bundle.latency.total_ms,
    ));

    lines
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::{AuditEntryType, AuditTrail};

    fn make_bundle_with_entries(n: usize) -> ForensicBundle {
        let mut trail = AuditTrail::new();
        for i in 0..n {
            trail
                .record(&format!("event-{i}"), AuditEntryType::ConversationTurn)
                .unwrap();
        }
        let mut latency = LatencyBreakdown::new();
        latency.step6_claude_api_ms = 100;
        latency.compute_total();

        trail.build_forensic_bundle(
            "session-test-001",
            latency,
            3,
            "Enhanced",
            0.80,
            0.75,
        )
    }

    #[test]
    fn forensic_bundle_new_sets_fields() {
        let bundle = make_bundle_with_entries(2);
        assert_eq!(bundle.session_id, "session-test-001");
        assert_eq!(bundle.audit_entries.len(), 2);
        assert_eq!(bundle.attestation_count, 3);
        assert_eq!(bundle.proof_tier, "Enhanced");
        assert!((bundle.quality_score - 0.75).abs() < f32::EPSILON);
        assert!((bundle.coherence_score - 0.80).abs() < f32::EPSILON);
        assert!(bundle.timestamp_ms > 0);
    }

    #[test]
    fn replay_audit_line_count_matches_entries_plus_summary() {
        let bundle = make_bundle_with_entries(3);
        let lines = replay_audit(&bundle);
        // 3 entry lines + 1 summary line
        assert_eq!(lines.len(), 4);
    }

    #[test]
    fn replay_audit_summary_contains_key_fields() {
        let bundle = make_bundle_with_entries(1);
        let lines = replay_audit(&bundle);
        let summary = lines.last().unwrap();
        assert!(summary.starts_with("SUMMARY"), "summary line: {summary}");
        assert!(summary.contains("session-test-001"), "{summary}");
        assert!(summary.contains("Enhanced"), "{summary}");
        assert!(summary.contains("entries=1"), "{summary}");
    }

    #[test]
    fn replay_audit_empty_bundle_has_only_summary() {
        let bundle = make_bundle_with_entries(0);
        let lines = replay_audit(&bundle);
        assert_eq!(lines.len(), 1);
        let summary = &lines[0];
        assert!(summary.contains("entries=0"), "{summary}");
    }

    #[test]
    fn replay_audit_entry_lines_contain_hash_prefix() {
        let bundle = make_bundle_with_entries(2);
        let lines = replay_audit(&bundle);
        // First line should reference index 0 and have "hash="
        assert!(lines[0].contains("[0]"), "{}", lines[0]);
        assert!(lines[0].contains("hash="), "{}", lines[0]);
        assert!(lines[1].contains("[1]"), "{}", lines[1]);
    }
}
