//! Tests for [`ai_assistant::language`]

use ai_assistant::language::{detect_language, load_system_prompt};
use ai_assistant::types::Language;

/// Test 1: detect_language("merhaba nasılsın") → Turkish (Turkish chars).
#[test]
fn test_detect_turkish_by_special_char_nasil() {
    let result = detect_language("merhaba nasılsın");
    assert_eq!(result, Language::Turkish, "nasılsın contains ı and ş");
}

/// Test 2: detect_language("hello world") → English.
#[test]
fn test_detect_english_hello_world() {
    let result = detect_language("hello world");
    assert_eq!(result, Language::English);
}

/// Test 3: detect_language("") → English (default).
#[test]
fn test_detect_empty_defaults_to_english() {
    let result = detect_language("");
    assert_eq!(result, Language::English, "Empty input should default to English");
}

/// Test 4: detect_language("çok güzel") → Turkish (ç special char).
#[test]
fn test_detect_turkish_by_cedilla() {
    let result = detect_language("çok güzel");
    assert_eq!(result, Language::Turkish, "'ç' is an unambiguous Turkish character");
}

/// Test 5: detect_language("the quick brown fox") → English.
#[test]
fn test_detect_english_the_quick_brown_fox() {
    let result = detect_language("the quick brown fox");
    assert_eq!(result, Language::English, "'the' is a strong English signal");
}

/// Test 6: load_system_prompt(Turkish) returns non-empty string.
#[test]
fn test_load_system_prompt_turkish_non_empty() {
    let prompt = load_system_prompt(&Language::Turkish);
    assert!(!prompt.is_empty(), "Turkish system prompt should not be empty");
    assert!(prompt.len() > 10, "Turkish system prompt should be substantial");
}

/// Test 7: load_system_prompt(English) returns non-empty string.
#[test]
fn test_load_system_prompt_english_non_empty() {
    let prompt = load_system_prompt(&Language::English);
    assert!(!prompt.is_empty(), "English system prompt should not be empty");
    assert!(prompt.len() > 10, "English system prompt should be substantial");
}

/// Extra: Other language falls back to English prompt.
#[test]
fn test_load_system_prompt_other_returns_fallback() {
    let prompt = load_system_prompt(&Language::Other("fr".to_string()));
    assert!(!prompt.is_empty(), "Other language should fall back to a non-empty prompt");
}

/// Extra: Turkish word-frequency detection (no special chars).
#[test]
fn test_detect_turkish_by_word_frequency() {
    // "bu bir test ve daha" — common Turkish function words
    let result = detect_language("bu bir test ve daha");
    assert_eq!(result, Language::Turkish, "Turkish word frequency should win");
}

/// Extra: detect_language with only spaces → English default.
#[test]
fn test_detect_only_whitespace_defaults_to_english() {
    let result = detect_language("   ");
    assert_eq!(result, Language::English, "Whitespace-only should default to English");
}
