//! Proof-carrying validation and witness chain recording.
//!
//! Uses `ruvector-verified` [`ProofEnvironment`] to track proof terms, and
//! SHAKE-256 (from the `sha3` crate) to hash each (response, context) pair
//! into a 32-byte witness that is stored in a local chain.
//!
//! Also provides [`SchemaValidator`] / [`JsonSchemaValidator`] / [`TypeValidator`] /
//! [`CombinedValidator`] for validating MCP tool response JSON — a native-Rust
//! port inspired by `ruvllm::quality::validators`.

use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};

use ruvector_verified::{
    proof_store::{create_attestation, ProofAttestation},
    ProofEnvironment,
};

use crate::{error::AiAssistantError, types::VerifiedResponse};

// ── ProofTier ─────────────────────────────────────────────────────────────────

/// Proof routing tier based on quality score and security flags.
///
/// Mirrors the tiered routing pattern from `ruvector-verified::gated`,
/// adapted for AI response quality: each tier selects the depth of
/// verification and downstream handling applied to a response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofTier {
    /// Normal responses — quality score < 0.7.
    Standard,
    /// Good responses — 0.7 ≤ quality score < 0.9.
    Enhanced,
    /// High-stakes responses — quality score ≥ 0.9 **or** security flagged.
    Critical,
}

impl std::fmt::Display for ProofTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProofTier::Standard => write!(f, "Standard"),
            ProofTier::Enhanced => write!(f, "Enhanced"),
            ProofTier::Critical => write!(f, "Critical"),
        }
    }
}

/// Route a response to the cheapest sufficient proof tier.
///
/// # Rules
/// - `security_flagged = true`  → [`ProofTier::Critical`] (always)
/// - `quality_score ≥ 0.9`      → [`ProofTier::Critical`]
/// - `quality_score ≥ 0.7`      → [`ProofTier::Enhanced`]
/// - otherwise                  → [`ProofTier::Standard`]
pub fn route_proof(quality_score: f32, security_flagged: bool) -> ProofTier {
    if security_flagged || quality_score >= 0.9 {
        ProofTier::Critical
    } else if quality_score >= 0.7 {
        ProofTier::Enhanced
    } else {
        ProofTier::Standard
    }
}

// ── Verifier ──────────────────────────────────────────────────────────────────

/// Proof-carrying verifier with an in-memory witness chain.
///
/// Each call to [`validate_and_record`] appends a new witness hash to the
/// internal chain. [`verify_witness`] checks whether a hash is recorded.
pub struct Verifier {
    /// Formal proof environment (tracks term IDs and cache).
    env: ProofEnvironment,
    /// Ordered list of SHAKE-256 witness hashes.
    witnesses: Vec<String>,
    /// Cryptographic attestation chain — one entry per validated response.
    attestations: Vec<ProofAttestation>,
}

impl Verifier {
    /// Create a new verifier with a fresh proof environment.
    pub fn new() -> Result<Self, AiAssistantError> {
        Ok(Self {
            env: ProofEnvironment::new(),
            witnesses: Vec::new(),
            attestations: Vec::new(),
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

        // Allocate a proof term and create a cryptographic attestation
        let proof_id = self.env.alloc_term();
        let attestation = create_attestation(&self.env, proof_id);
        self.attestations.push(attestation);

        // Append hash to witness chain
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

    /// Return a reference to the full attestation chain.
    ///
    /// Each entry is a 82-byte-serializable [`ProofAttestation`] that was
    /// produced for the corresponding [`validate_and_record`] call.
    pub fn attestation_chain(&self) -> &[ProofAttestation] {
        &self.attestations
    }

    /// Return proof environment statistics (for diagnostics / logging).
    pub fn proof_stats(&self) -> ruvector_verified::ProofStats {
        self.env.stats().clone()
    }

    /// Validate a response, record it in the witness chain, and route to a [`ProofTier`].
    ///
    /// Combines [`validate_and_record`] with [`route_proof`].  The `quality_score`
    /// must be in `[0.0, 1.0]`; values outside that range are clamped silently.
    ///
    /// Returns `(VerifiedResponse, ProofTier)` on success.
    pub fn validate_and_route(
        &mut self,
        response_text: &str,
        context: &str,
        quality_score: f32,
    ) -> Result<(VerifiedResponse, ProofTier), AiAssistantError> {
        // Clamp to valid range
        let score = quality_score.clamp(0.0, 1.0);

        // Determine tier before recording (no state mutation yet)
        // Security flag: a simple heuristic — check for blocked keywords.
        let security_flagged = is_security_flagged(response_text);
        let tier = route_proof(score, security_flagged);

        // Record in witness chain
        let verified = self.validate_and_record(response_text, context)?;

        Ok((verified, tier))
    }

    /// Validates a JSON tool response using schema validation.
    ///
    /// Returns [`ValidationResult`] — non-blocking, never panics.
    /// Uses [`mcp_tool_response_validator`] internally.
    pub fn validate_tool_response(&self, json: &serde_json::Value) -> ValidationResult {
        let validator = mcp_tool_response_validator();
        validator.validate(json)
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

// ── ValidationResult ──────────────────────────────────────────────────────────

/// Result of a schema/type validation operation.
///
/// Inspired by `ruvllm::quality::validators::ValidationResult`.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether all checks passed.
    pub valid: bool,
    /// Human-readable error messages (one per failed check).
    pub errors: Vec<String>,
    /// Non-fatal advisory messages.
    pub warnings: Vec<String>,
    /// Number of top-level fields observed during validation.
    pub field_count: usize,
}

impl ValidationResult {
    /// Create a passing result with `field_count` fields examined.
    pub fn ok(field_count: usize) -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            field_count,
        }
    }

    /// Append an error message and mark the result as invalid.
    pub fn with_error(mut self, msg: impl Into<String>) -> Self {
        self.valid = false;
        self.errors.push(msg.into());
        self
    }

    /// Append a warning message (does not affect `valid`).
    pub fn with_warning(mut self, msg: impl Into<String>) -> Self {
        self.warnings.push(msg.into());
        self
    }

    /// Merge `other` into `self` (AND semantics):
    /// - `valid` becomes `false` if either side is invalid.
    /// - errors, warnings, and field counts are accumulated.
    pub fn merge(mut self, other: ValidationResult) -> Self {
        if !other.valid {
            self.valid = false;
        }
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
        self.field_count += other.field_count;
        self
    }
}

// ── SchemaValidator trait ─────────────────────────────────────────────────────

/// Trait for validating a [`serde_json::Value`] against a rule.
///
/// Mirrors `ruvllm::quality::validators::SchemaValidator`.
pub trait SchemaValidator: Send + Sync {
    /// Validate `value` and return a [`ValidationResult`].
    fn validate(&self, value: &serde_json::Value) -> ValidationResult;

    /// Short identifier used in error messages and logging.
    fn name(&self) -> &str;
}

// ── JsonSchemaValidator ───────────────────────────────────────────────────────

/// Validates JSON objects against required-field and forbidden-field rules,
/// with an optional maximum nesting depth guard.
///
/// Simplified port of `ruvllm::quality::validators::JsonSchemaValidator`.
pub struct JsonSchemaValidator {
    required_fields: Vec<String>,
    forbidden_fields: Vec<String>,
    max_depth: usize,
}

impl JsonSchemaValidator {
    /// Create a validator that requires `required` fields to be present at the
    /// root of the validated JSON object.
    pub fn new(required: &[&str]) -> Self {
        Self {
            required_fields: required.iter().map(|s| s.to_string()).collect(),
            forbidden_fields: Vec::new(),
            max_depth: usize::MAX,
        }
    }

    /// Declare fields that must **not** appear in the JSON object.
    pub fn with_forbidden(mut self, fields: &[&str]) -> Self {
        self.forbidden_fields = fields.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Set the maximum allowed nesting depth (default: unlimited).
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Recursively check whether `value` exceeds `max` nesting levels.
    fn check_depth(value: &serde_json::Value, current: usize, max: usize) -> bool {
        if current > max {
            return false;
        }
        match value {
            serde_json::Value::Object(map) => {
                map.values().all(|v| Self::check_depth(v, current + 1, max))
            }
            serde_json::Value::Array(arr) => {
                arr.iter().all(|v| Self::check_depth(v, current + 1, max))
            }
            _ => true,
        }
    }
}

impl SchemaValidator for JsonSchemaValidator {
    fn validate(&self, value: &serde_json::Value) -> ValidationResult {
        let obj = match value.as_object() {
            Some(o) => o,
            None => {
                return ValidationResult::ok(0)
                    .with_error("expected a JSON object at root");
            }
        };

        let field_count = obj.len();
        let mut result = ValidationResult::ok(field_count);

        // Required-field checks
        for field in &self.required_fields {
            if !obj.contains_key(field.as_str()) {
                result = result.with_error(format!("missing required field '{field}'"));
            }
        }

        // Forbidden-field checks
        for field in &self.forbidden_fields {
            if obj.contains_key(field.as_str()) {
                result = result.with_error(format!("forbidden field '{field}' is present"));
            }
        }

        // Depth check
        if !Self::check_depth(value, 0, self.max_depth) {
            result = result.with_error(format!(
                "JSON nesting exceeds maximum depth of {}",
                self.max_depth
            ));
        }

        result
    }

    fn name(&self) -> &str {
        "JsonSchemaValidator"
    }
}

// ── TypeValidator ─────────────────────────────────────────────────────────────

/// Validates that a specific field inside a JSON object has the expected type.
///
/// Supported type strings: `"string"`, `"number"`, `"bool"`, `"array"`,
/// `"object"`, `"null"`.
pub struct TypeValidator {
    field: String,
    expected_type: &'static str,
}

impl TypeValidator {
    /// Expect `field` to be a JSON string.
    pub fn string(field: &str) -> Self {
        Self { field: field.to_string(), expected_type: "string" }
    }

    /// Expect `field` to be a JSON number.
    pub fn number(field: &str) -> Self {
        Self { field: field.to_string(), expected_type: "number" }
    }

    /// Expect `field` to be a JSON boolean.
    pub fn boolean(field: &str) -> Self {
        Self { field: field.to_string(), expected_type: "bool" }
    }
}

impl SchemaValidator for TypeValidator {
    fn validate(&self, value: &serde_json::Value) -> ValidationResult {
        let field_value = match value.get(&self.field) {
            Some(v) => v,
            None => {
                // Field absence is not a type error; JsonSchemaValidator
                // handles required-field checks.
                return ValidationResult::ok(0);
            }
        };

        let type_ok = match self.expected_type {
            "string" => field_value.is_string(),
            "number" => field_value.is_number(),
            "bool"   => field_value.is_boolean(),
            "array"  => field_value.is_array(),
            "object" => field_value.is_object(),
            "null"   => field_value.is_null(),
            _        => true,
        };

        let mut result = ValidationResult::ok(1);
        if !type_ok {
            result = result.with_error(format!(
                "field '{}' must be {} but got {}",
                self.field,
                self.expected_type,
                json_type_name(field_value),
            ));
        }
        result
    }

    fn name(&self) -> &str {
        "TypeValidator"
    }
}

// ── CombinedValidator ─────────────────────────────────────────────────────────

/// Chains multiple [`SchemaValidator`]s with AND logic:
/// every validator must pass for the combined result to be valid.
pub struct CombinedValidator {
    validators: Vec<Box<dyn SchemaValidator>>,
}

impl CombinedValidator {
    /// Create an empty combined validator.
    pub fn new() -> Self {
        Self { validators: Vec::new() }
    }

    /// Append a validator to the chain (builder pattern).
    pub fn add(mut self, v: impl SchemaValidator + 'static) -> Self {
        self.validators.push(Box::new(v));
        self
    }
}

impl Default for CombinedValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl SchemaValidator for CombinedValidator {
    fn validate(&self, value: &serde_json::Value) -> ValidationResult {
        let mut combined = ValidationResult::ok(0);
        for v in &self.validators {
            combined = combined.merge(v.validate(value));
        }
        combined
    }

    fn name(&self) -> &str {
        "CombinedValidator"
    }
}

// ── MCP tool response validator factory ───────────────────────────────────────

/// Build a validator for the MCP tool response JSON format.
///
/// Expected shape: `{"type": "tool_result", "content": [...]}`
/// — the `"type"` field is required and must be a string.
pub fn mcp_tool_response_validator() -> CombinedValidator {
    CombinedValidator::new()
        .add(JsonSchemaValidator::new(&["type"]))
        .add(TypeValidator::string("type"))
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Minimal heuristic security check: returns `true` when the response text
/// contains any of the known blocked phrases.
///
/// This intentionally avoids loading external configuration so the verifier
/// has no I/O dependencies.
fn is_security_flagged(text: &str) -> bool {
    const BLOCKED: &[&str] = &[
        "SECURITY_HALT",
        "INJECTION_DETECTED",
        "UNSAFE_CONTENT",
    ];
    let lower = text.to_lowercase();
    BLOCKED.iter().any(|kw| lower.contains(&kw.to_lowercase()))
}

/// Return a human-readable type label for a [`serde_json::Value`].
fn json_type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null      => "null",
        serde_json::Value::Bool(_)   => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_)  => "array",
        serde_json::Value::Object(_) => "object",
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── ValidationResult ────────────────────────────────────────────────────

    #[test]
    fn validation_result_ok_is_valid() {
        let r = ValidationResult::ok(3);
        assert!(r.valid);
        assert_eq!(r.field_count, 3);
        assert!(r.errors.is_empty());
    }

    #[test]
    fn validation_result_with_error_is_invalid() {
        let r = ValidationResult::ok(1).with_error("bad field");
        assert!(!r.valid);
        assert_eq!(r.errors.len(), 1);
    }

    #[test]
    fn validation_result_with_warning_stays_valid() {
        let r = ValidationResult::ok(1).with_warning("heads up");
        assert!(r.valid);
        assert_eq!(r.warnings.len(), 1);
    }

    #[test]
    fn validation_result_merge_accumulates() {
        let a = ValidationResult::ok(2);
        let b = ValidationResult::ok(1).with_error("oops");
        let c = a.merge(b);
        assert!(!c.valid);
        assert_eq!(c.field_count, 3);
        assert_eq!(c.errors.len(), 1);
    }

    // ── JsonSchemaValidator ─────────────────────────────────────────────────

    #[test]
    fn json_schema_required_field_present() {
        let v = JsonSchemaValidator::new(&["type"]);
        let r = v.validate(&json!({"type": "tool_result"}));
        assert!(r.valid);
    }

    #[test]
    fn json_schema_required_field_missing() {
        let v = JsonSchemaValidator::new(&["type"]);
        let r = v.validate(&json!({"content": []}));
        assert!(!r.valid);
        assert!(r.errors[0].contains("type"));
    }

    #[test]
    fn json_schema_forbidden_field_rejected() {
        let v = JsonSchemaValidator::new(&[]).with_forbidden(&["secret"]);
        let r = v.validate(&json!({"secret": "x"}));
        assert!(!r.valid);
    }

    #[test]
    fn json_schema_non_object_rejected() {
        let v = JsonSchemaValidator::new(&[]);
        let r = v.validate(&json!("not an object"));
        assert!(!r.valid);
    }

    #[test]
    fn json_schema_max_depth_exceeded() {
        let v = JsonSchemaValidator::new(&[]).with_max_depth(1);
        let deep = json!({"a": {"b": {"c": 1}}});
        let r = v.validate(&deep);
        assert!(!r.valid);
    }

    #[test]
    fn json_schema_max_depth_within_limit() {
        let v = JsonSchemaValidator::new(&[]).with_max_depth(5);
        let shallow = json!({"a": 1});
        let r = v.validate(&shallow);
        assert!(r.valid);
    }

    // ── TypeValidator ───────────────────────────────────────────────────────

    #[test]
    fn type_validator_string_ok() {
        let v = TypeValidator::string("type");
        let r = v.validate(&json!({"type": "tool_result"}));
        assert!(r.valid);
    }

    #[test]
    fn type_validator_string_wrong_type() {
        let v = TypeValidator::string("type");
        let r = v.validate(&json!({"type": 42}));
        assert!(!r.valid);
        assert!(r.errors[0].contains("string"));
    }

    #[test]
    fn type_validator_number_ok() {
        let v = TypeValidator::number("count");
        let r = v.validate(&json!({"count": 7}));
        assert!(r.valid);
    }

    #[test]
    fn type_validator_boolean_ok() {
        let v = TypeValidator::boolean("flag");
        let r = v.validate(&json!({"flag": true}));
        assert!(r.valid);
    }

    #[test]
    fn type_validator_missing_field_passes() {
        // Absence is not a type error
        let v = TypeValidator::string("type");
        let r = v.validate(&json!({}));
        assert!(r.valid);
    }

    // ── CombinedValidator ───────────────────────────────────────────────────

    #[test]
    fn combined_all_pass() {
        let c = CombinedValidator::new()
            .add(JsonSchemaValidator::new(&["type"]))
            .add(TypeValidator::string("type"));
        let r = c.validate(&json!({"type": "tool_result"}));
        assert!(r.valid);
    }

    #[test]
    fn combined_one_fails_propagates() {
        let c = CombinedValidator::new()
            .add(JsonSchemaValidator::new(&["type"]))
            .add(TypeValidator::string("type"));
        let r = c.validate(&json!({"type": 99}));
        assert!(!r.valid);
    }

    // ── mcp_tool_response_validator ─────────────────────────────────────────

    #[test]
    fn mcp_validator_valid_response() {
        let v = mcp_tool_response_validator();
        let r = v.validate(&json!({"type": "tool_result", "content": []}));
        assert!(r.valid, "errors: {:?}", r.errors);
    }

    #[test]
    fn mcp_validator_missing_type() {
        let v = mcp_tool_response_validator();
        let r = v.validate(&json!({"content": []}));
        assert!(!r.valid);
    }

    #[test]
    fn mcp_validator_type_wrong_kind() {
        let v = mcp_tool_response_validator();
        let r = v.validate(&json!({"type": 123}));
        assert!(!r.valid);
    }

    // ── Verifier::validate_tool_response ────────────────────────────────────

    #[test]
    fn verifier_validate_tool_response_valid() {
        let v = Verifier::new().unwrap();
        let r = v.validate_tool_response(&json!({"type": "tool_result"}));
        assert!(r.valid);
    }

    #[test]
    fn verifier_validate_tool_response_invalid() {
        let v = Verifier::new().unwrap();
        let r = v.validate_tool_response(&json!({"no_type": true}));
        assert!(!r.valid);
    }

    // ── attestation_chain ───────────────────────────────────────────────────

    #[test]
    fn attestation_chain_grows_with_each_validation() {
        let mut v = Verifier::new().unwrap();
        assert_eq!(v.attestation_chain().len(), 0);

        v.validate_and_record("response one", "ctx").unwrap();
        assert_eq!(v.attestation_chain().len(), 1);

        v.validate_and_record("response two", "ctx").unwrap();
        assert_eq!(v.attestation_chain().len(), 2);
    }

    #[test]
    fn attestation_has_correct_verifier_version() {
        let mut v = Verifier::new().unwrap();
        v.validate_and_record("hello", "world").unwrap();
        let att = &v.attestation_chain()[0];
        assert_eq!(att.verifier_version, 0x00_01_00_00);
        assert!(att.verification_timestamp_ns > 0);
    }

    #[test]
    fn attestation_hashes_are_non_trivial() {
        let mut v = Verifier::new().unwrap();
        v.validate_and_record("test response", "some context").unwrap();
        let att = &v.attestation_chain()[0];
        // SEC-002: proof_term_hash must have most bytes non-zero
        let nonzero = att.proof_term_hash.iter().filter(|&&b| b != 0).count();
        assert!(nonzero >= 16, "proof_term_hash has too many zero bytes: {nonzero}/32");
    }

    // ── route_proof ─────────────────────────────────────────────────────────

    #[test]
    fn route_proof_low_quality_is_standard() {
        assert_eq!(route_proof(0.5, false), ProofTier::Standard);
        assert_eq!(route_proof(0.0, false), ProofTier::Standard);
        assert_eq!(route_proof(0.69, false), ProofTier::Standard);
    }

    #[test]
    fn route_proof_medium_quality_is_enhanced() {
        assert_eq!(route_proof(0.7, false), ProofTier::Enhanced);
        assert_eq!(route_proof(0.8, false), ProofTier::Enhanced);
        assert_eq!(route_proof(0.89, false), ProofTier::Enhanced);
    }

    #[test]
    fn route_proof_high_quality_is_critical() {
        assert_eq!(route_proof(0.9, false), ProofTier::Critical);
        assert_eq!(route_proof(1.0, false), ProofTier::Critical);
    }

    #[test]
    fn route_proof_security_flagged_always_critical() {
        // Even very low quality: if security-flagged → Critical
        assert_eq!(route_proof(0.1, true), ProofTier::Critical);
        assert_eq!(route_proof(0.7, true), ProofTier::Critical);
    }

    // ── validate_and_route ──────────────────────────────────────────────────

    #[test]
    fn validate_and_route_returns_correct_tier() {
        let mut v = Verifier::new().unwrap();
        let (_, tier) = v.validate_and_route("hello world", "ctx", 0.5).unwrap();
        assert_eq!(tier, ProofTier::Standard);

        let (_, tier) = v.validate_and_route("good answer", "ctx", 0.75).unwrap();
        assert_eq!(tier, ProofTier::Enhanced);

        let (_, tier) = v.validate_and_route("excellent", "ctx", 0.95).unwrap();
        assert_eq!(tier, ProofTier::Critical);
    }

    #[test]
    fn validate_and_route_security_flagged_forces_critical() {
        let mut v = Verifier::new().unwrap();
        let (_, tier) = v
            .validate_and_route("SECURITY_HALT detected", "ctx", 0.3)
            .unwrap();
        assert_eq!(tier, ProofTier::Critical);
    }

    #[test]
    fn validate_and_route_records_witness() {
        let mut v = Verifier::new().unwrap();
        assert_eq!(v.witness_count(), 0);
        v.validate_and_route("response", "context", 0.8).unwrap();
        assert_eq!(v.witness_count(), 1);
    }

    #[test]
    fn proof_tier_display() {
        assert_eq!(ProofTier::Standard.to_string(), "Standard");
        assert_eq!(ProofTier::Enhanced.to_string(), "Enhanced");
        assert_eq!(ProofTier::Critical.to_string(), "Critical");
    }
}
