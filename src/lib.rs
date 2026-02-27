//! AI Assistant library — re-exports all modules for integration testing.
//!
//! The binary (`main.rs`) and integration tests (`tests/`) both import from
//! this crate root. Module declarations here mirror those in `main.rs`.

#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

pub mod audit;
pub mod forensic;
pub mod claude_api;
pub mod coherence;
pub mod config;
pub mod embedding;
pub mod error;
pub mod graph_context;
pub mod language;
pub mod learning;
pub mod memory;
pub mod mcp_tools;
pub mod ner;
pub mod pipeline;
pub mod reasoning;
pub mod types;
pub mod verification;
