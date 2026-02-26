//! Tests for [`ai_assistant::verification`]

use ai_assistant::verification::Verifier;

/// Test 1: Verifier::new() succeeds.
#[test]
fn test_verifier_new_succeeds() {
    let verifier = Verifier::new();
    assert!(verifier.is_ok(), "Verifier::new() should succeed");
}

/// Test 2: validate_and_record() returns VerifiedResponse with non-empty witness_hash.
#[test]
fn test_validate_and_record_returns_verified_response() {
    let mut verifier = Verifier::new().unwrap();
    let result = verifier.validate_and_record("test response", "test context");
    assert!(result.is_ok(), "validate_and_record should succeed");

    let verified = result.unwrap();
    assert!(!verified.witness_hash.is_empty(), "witness_hash should not be empty");
    assert_eq!(verified.text, "test response", "text should match input");
}

/// Test 3: witness_hash is 64 chars (SHAKE-256 hex).
#[test]
fn test_witness_hash_is_64_chars() {
    let mut verifier = Verifier::new().unwrap();
    let verified = verifier
        .validate_and_record("response text", "context text")
        .unwrap();
    assert_eq!(
        verified.witness_hash.len(),
        64,
        "SHAKE-256 hash should be 64 hex chars (32 bytes)"
    );
    assert!(
        verified.witness_hash.chars().all(|c| c.is_ascii_hexdigit()),
        "Hash should only contain hex digits"
    );
}

/// Test 4: verify_witness(valid_hash) returns true.
#[test]
fn test_verify_witness_valid_hash_returns_true() {
    let mut verifier = Verifier::new().unwrap();
    let verified = verifier.validate_and_record("hello", "world").unwrap();
    let hash = verified.witness_hash.clone();

    let found = verifier.verify_witness(&hash).unwrap();
    assert!(found, "A recorded witness hash should be found");
}

/// Test 5: verify_witness("invalid_hash") returns false.
#[test]
fn test_verify_witness_invalid_hash_returns_false() {
    let verifier = Verifier::new().unwrap();
    let found = verifier.verify_witness("not_a_real_hash_at_all").unwrap();
    assert!(!found, "An unrecorded hash should not be found");
}

/// Test 6: Same input produces same hash (determinism).
#[test]
fn test_same_input_produces_same_hash() {
    let mut v1 = Verifier::new().unwrap();
    let mut v2 = Verifier::new().unwrap();

    let h1 = v1.validate_and_record("response", "context").unwrap().witness_hash;
    let h2 = v2.validate_and_record("response", "context").unwrap().witness_hash;

    assert_eq!(h1, h2, "Same input should produce same hash");
}

/// Extra: witness_count() increments with each record.
#[test]
fn test_witness_count_increments() {
    let mut verifier = Verifier::new().unwrap();
    assert_eq!(verifier.witness_count(), 0);

    verifier.validate_and_record("r1", "c1").unwrap();
    assert_eq!(verifier.witness_count(), 1);

    verifier.validate_and_record("r2", "c2").unwrap();
    assert_eq!(verifier.witness_count(), 2);
}

/// Extra: verify_witness checks all recorded hashes.
#[test]
fn test_verify_witness_finds_any_recorded_hash() {
    let mut verifier = Verifier::new().unwrap();

    let h1 = verifier.validate_and_record("first", "ctx").unwrap().witness_hash;
    let h2 = verifier.validate_and_record("second", "ctx").unwrap().witness_hash;
    let h3 = verifier.validate_and_record("third", "ctx").unwrap().witness_hash;

    assert!(verifier.verify_witness(&h1).unwrap());
    assert!(verifier.verify_witness(&h2).unwrap());
    assert!(verifier.verify_witness(&h3).unwrap());
    assert!(!verifier.verify_witness("completely_wrong_hash").unwrap());
}

/// Extra: Different inputs produce different hashes.
#[test]
fn test_different_inputs_produce_different_hashes() {
    let mut verifier = Verifier::new().unwrap();
    let h1 = verifier.validate_and_record("response A", "context A").unwrap().witness_hash;
    let h2 = verifier.validate_and_record("response B", "context B").unwrap().witness_hash;
    assert_ne!(h1, h2, "Different inputs should produce different hashes");
}
