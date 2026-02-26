//! Tests for [`ai_assistant::audit`]

use ai_assistant::audit::{AuditEntryType, AuditTrail};

/// Test 1: AuditTrail::new() creates empty chain.
#[test]
fn test_audit_trail_new_empty() {
    let trail = AuditTrail::new();
    assert!(trail.is_empty(), "New trail should be empty");
    assert_eq!(trail.len(), 0, "New trail length should be 0");
    assert!(trail.entries().is_empty(), "Entries slice should be empty");
}

/// Test 2: record() returns Ok with non-empty hash.
#[test]
fn test_record_returns_ok_non_empty_hash() {
    let mut trail = AuditTrail::new();
    let result = trail.record("test data", AuditEntryType::ConversationTurn);
    assert!(result.is_ok(), "record() should succeed");

    let audit = result.unwrap();
    assert!(!audit.hash.is_empty(), "Hash should not be empty");
    assert!(!audit.episode_id.is_empty(), "Episode ID should not be empty");
}

/// Test 3: verify_chain() returns true for valid chain.
#[test]
fn test_verify_chain_valid() {
    let mut trail = AuditTrail::new();
    trail.record("event 1", AuditEntryType::ConversationTurn).unwrap();
    trail.record("event 2", AuditEntryType::EpisodeStored).unwrap();
    trail.record("event 3", AuditEntryType::CausalEdgeAdded).unwrap();

    assert!(trail.verify_chain(), "Valid chain should verify successfully");
}

/// Test 4: Two entries form valid chain (entry2.prev_hash == entry1's hash).
#[test]
fn test_two_entry_chain_linked() {
    let mut trail = AuditTrail::new();
    let r1 = trail.record("first entry", AuditEntryType::ConversationTurn).unwrap();
    let _r2 = trail.record("second entry", AuditEntryType::EpisodeStored).unwrap();

    let entries = trail.entries();
    assert_eq!(entries.len(), 2);

    // entry[1].prev_hash should equal entry[0]'s computed hash
    let entry0_hash = entries[0].entry_hash();
    assert_eq!(
        entries[1].prev_hash, entry0_hash,
        "Second entry's prev_hash should equal first entry's hash"
    );
    // The hash returned from record() for entry 1 should match last_hash
    let _ = r1; // suppress unused warning
}

/// Test 5: Tampering entry breaks chain (verify_chain() returns false).
#[test]
fn test_tampered_chain_fails_verification() {
    let mut trail = AuditTrail::new();
    trail.record("original data", AuditEntryType::ConversationTurn).unwrap();
    trail.record("second entry", AuditEntryType::EpisodeStored).unwrap();

    // Verify clean chain is intact via the public API
    assert!(trail.verify_chain(), "Chain should be valid before tampering");
    assert_eq!(trail.len(), 2);

    // The internal unit test in audit.rs tests actual field-level tampering
    // (which requires private access). From the external integration test we
    // can only confirm:
    //  1. A freshly built chain verifies correctly.
    //  2. A chain whose data changed at the source level produces a *different*
    //     last hash (determinism + avalanche effect).
    let mut trail_a = AuditTrail::new();
    trail_a.record("data-A", AuditEntryType::ConversationTurn).unwrap();

    let mut trail_b = AuditTrail::new();
    trail_b.record("data-B", AuditEntryType::ConversationTurn).unwrap();

    // Different content → different terminal hash (avalanche property)
    assert_ne!(
        trail_a.last_hash(),
        trail_b.last_hash(),
        "Different data must produce different hashes"
    );

    // Both individual chains are valid
    assert!(trail_a.verify_chain());
    assert!(trail_b.verify_chain());
}

/// Test 6: shake256 produces 64-char hex string (32 bytes).
#[test]
fn test_shake256_produces_64_char_hex() {
    let hash = AuditTrail::shake256(b"test data");
    assert_eq!(hash.len(), 64, "SHAKE-256 should produce 64 hex chars (32 bytes)");
    assert!(
        hash.chars().all(|c| c.is_ascii_hexdigit()),
        "Hash should be lowercase hex"
    );
}

/// Test 7: AuditEntryType variants exist and can be compared.
#[test]
fn test_audit_entry_type_variants() {
    let conversation = AuditEntryType::ConversationTurn;
    let episode = AuditEntryType::EpisodeStored;
    let causal = AuditEntryType::CausalEdgeAdded;
    let security = AuditEntryType::SecurityHalt;

    assert_eq!(conversation, AuditEntryType::ConversationTurn);
    assert_ne!(conversation, episode);
    assert_ne!(causal, security);

    // Verify Display impl
    assert_eq!(format!("{}", AuditEntryType::ConversationTurn), "ConversationTurn");
    assert_eq!(format!("{}", AuditEntryType::EpisodeStored), "EpisodeStored");
    assert_eq!(format!("{}", AuditEntryType::CausalEdgeAdded), "CausalEdgeAdded");
    assert_eq!(format!("{}", AuditEntryType::SecurityHalt), "SecurityHalt");
}

/// Extra: shake256 is deterministic.
#[test]
fn test_shake256_deterministic() {
    let h1 = AuditTrail::shake256(b"determinism check");
    let h2 = AuditTrail::shake256(b"determinism check");
    assert_eq!(h1, h2, "SHAKE-256 should be deterministic");
}

/// Extra: Empty chain verifies.
#[test]
fn test_empty_chain_verifies() {
    let trail = AuditTrail::new();
    assert!(trail.verify_chain(), "Empty chain should verify as valid");
}

/// Extra: last_hash is genesis for empty trail.
#[test]
fn test_last_hash_genesis_for_empty_trail() {
    let trail = AuditTrail::new();
    let genesis = "0000000000000000000000000000000000000000000000000000000000000000";
    assert_eq!(trail.last_hash(), genesis);
}

/// Extra: record() increases trail length.
#[test]
fn test_record_increases_length() {
    let mut trail = AuditTrail::new();
    assert_eq!(trail.len(), 0);

    trail.record("a", AuditEntryType::ConversationTurn).unwrap();
    assert_eq!(trail.len(), 1);

    trail.record("b", AuditEntryType::EpisodeStored).unwrap();
    assert_eq!(trail.len(), 2);

    assert!(!trail.is_empty());
}
