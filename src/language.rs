//! Heuristic language detection and system-prompt loading.

use crate::types::Language;

/// Unicode characters that appear exclusively in Turkish text.
const TURKISH_CHARS: &[char] = &[
    'ç', 'ğ', 'ı', 'İ', 'ö', 'ş', 'ü', 'Ç', 'Ğ', 'Ö', 'Ş', 'Ü',
];

/// High-frequency Turkish function words and particles.
const TURKISH_WORDS: &[&str] = &[
    "ve", "bir", "bu", "de", "da", "ile", "için", "var", "ne", "ben",
    "sen", "biz", "siz", "olan", "gibi", "kadar", "daha", "çok", "nasıl", "neden",
];

/// High-frequency English function words.
const ENGLISH_WORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should", "can",
    "what", "how", "why", "when", "where", "who", "which", "that", "this",
];

/// Detect the language of `text` using character and word heuristics.
///
/// Detection order:
/// 1. Turkish-specific Unicode characters (strong signal).
/// 2. Word-frequency scoring for Turkish vs. English.
/// 3. Default to [`Language::English`] when no signal is found.
pub fn detect_language(text: &str) -> Language {
    // Step 1 — Turkish-specific characters are an unambiguous signal.
    let turkish_char_count = text.chars().filter(|c| TURKISH_CHARS.contains(c)).count();
    if turkish_char_count > 0 {
        return Language::Turkish;
    }

    // Step 2 — Word-frequency scoring.
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();

    if words.is_empty() {
        return Language::English;
    }

    let turkish_score = words
        .iter()
        .filter(|w| TURKISH_WORDS.contains(w))
        .count();
    let english_score = words
        .iter()
        .filter(|w| ENGLISH_WORDS.contains(w))
        .count();

    if turkish_score > english_score {
        Language::Turkish
    } else {
        // Default to English when English score > 0 or no signal at all.
        Language::English
    }
}

/// Load the system prompt for `language` from the `prompts/` directory.
///
/// Falls back to a minimal inline prompt if the file cannot be read.
pub fn load_system_prompt(language: &Language) -> String {
    let filename = match language {
        Language::Turkish => "prompts/system_tr.md",
        Language::English => "prompts/system_en.md",
        Language::Other(_) => "prompts/system_en.md",
    };
    let path = crate::config::exe_dir().join(filename);

    std::fs::read_to_string(&path).unwrap_or_else(|_| {
        "You are a helpful AI assistant. Respond clearly and accurately.".to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_turkish_by_special_char() {
        assert_eq!(detect_language("Merhaba, nasılsın?"), Language::Turkish);
    }

    #[test]
    fn detects_english_by_words() {
        assert_eq!(detect_language("What is the weather like today?"), Language::English);
    }

    #[test]
    fn empty_input_defaults_to_english() {
        assert_eq!(detect_language(""), Language::English);
    }

    #[test]
    fn detects_turkish_by_word_score() {
        // No special chars but Turkish words dominate.
        assert_eq!(detect_language("bu bir test ve daha iyi bir sonuc"), Language::Turkish);
    }
}
