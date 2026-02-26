//! AI Assistant entry point.
//!
//! Initialises all pipeline components from environment configuration and
//! runs an interactive REPL loop. Press Ctrl+C or type `/quit` to exit.
//!
//! # SSE Streaming
//! Each response is streamed token-by-token via the Anthropic SSE protocol.
//! Tokens are printed with `print!` + `stdout().flush()` as they arrive,
//! giving the user a typewriter-style display.

#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

use std::io::{self, BufRead, Write};

mod audit;
mod claude_api;
mod coherence;
mod config;
mod embedding;
mod error;
mod forensic;
mod graph_context;
mod language;
mod learning;
mod memory;
mod mcp_tools;
mod ner;
mod pipeline;
mod types;
mod verification;

use pipeline::Pipeline;

#[tokio::main]
async fn main() {
    // Initialise structured logging — suppress ort crate noise, default WARN elsewhere.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("ort=error".parse().unwrap())
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    // Determine directory containing this executable so resources are found
    // regardless of the current working directory.
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."));

    // Load .env from exe directory first; fall back to cwd.
    let env_path = exe_dir.join(".env");
    if env_path.exists() {
        dotenvy::from_path(&env_path).ok();
    } else {
        dotenvy::dotenv().ok();
    }

    // Load configuration from already-set environment variables.
    use config::load_config_from_env;
    let config = match load_config_from_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Configuration error: {}", e);
            eprintln!("Please check your .env file. See .env.example for required variables.");
            std::process::exit(1);
        }
    };

    println!("🤖 AI Assistant starting...");
    println!("   Model:    {}", config.claude_model);
    println!("   Endpoint: {}", config.anthropic_base_url);

    // Initialise the full 10-step pipeline.
    let mut pipeline = match Pipeline::new(config).await {
        Ok(p) => {
            println!("✅ All systems initialised");
            p
        }
        Err(e) => {
            eprintln!("Initialisation error: {}", e);
            std::process::exit(1);
        }
    };

    println!("💬 Type your message (Ctrl+C or /quit to exit)\n");

    // REPL loop — one `execute_turn_stream` call per user input line.
    let stdin = io::stdin();
    loop {
        print!("You: ");
        io::stdout().flush().unwrap_or_default();

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }
                if input == "/quit" || input == "/exit" {
                    break;
                }

                // Print the "Assistant: " prefix before streaming begins
                print!("\nAssistant: ");
                io::stdout().flush().unwrap_or_default();

                // Token-by-token streaming callback:
                // each delta is printed immediately without a trailing newline.
                let token_callback = || {
                    move |token: &str| {
                        print!("{}", token);
                        // Flush on every token so the terminal shows it immediately.
                        let _ = io::stdout().flush();
                    }
                };

                match pipeline.execute_turn_stream(input, token_callback()).await {
                    Ok(_response) => {
                        // Stream already printed all tokens; just add the final newline.
                        println!("\n");
                    }
                    Err(error::AiAssistantError::CoherenceHalt) => {
                        println!(
                            "\n\n⚠️  Critical contradiction detected. \
                             Please rephrase your message.\n"
                        );
                    }
                    Err(e) => {
                        eprintln!("\n\n❌ Error: {}\n", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Read error: {}", e);
                break;
            }
        }
    }

    println!("\n👋 Goodbye!");
}
