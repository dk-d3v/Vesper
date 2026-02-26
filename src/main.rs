//! AI Assistant entry point.
//!
//! Initialises all pipeline components from environment configuration and
//! runs an interactive REPL loop. Press Ctrl+C or type `/quit` to exit.

use std::io::{self, BufRead, Write};

mod audit;
mod claude_api;
mod coherence;
mod config;
mod embedding;
mod error;
mod graph_context;
mod language;
mod learning;
mod memory;
mod mcp_tools;
mod pipeline;
mod types;
mod verification;

use config::load_config;
use pipeline::Pipeline;

#[tokio::main]
async fn main() {
    // Initialise structured logging — default level WARN to keep output clean.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::WARN.into()),
        )
        .init();

    // Load configuration from .env / system environment.
    let config = match load_config() {
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

    // REPL loop — one `execute_turn` call per user input line.
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

                match pipeline.execute_turn(input).await {
                    Ok(response) => println!("\nAssistant: {}\n", response),
                    Err(error::AiAssistantError::CoherenceCritical) => {
                        println!(
                            "\n⚠️  Critical contradiction detected. \
                             Please rephrase your message.\n"
                        );
                    }
                    Err(e) => {
                        eprintln!("\n❌ Error: {}\n", e);
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
