//! `octoflow chat` — conversational AI coding assistant.
//!
//! Generates .flow code from natural language descriptions. Two backends:
//!
//! **Local mode** (default): Runs a GGUF model on the GPU.
//! **API mode** (`--provider api`): Calls any OpenAI-compatible API via curl.
//!
//! The core loop:
//! 1. User describes what they want
//! 2. LLM generates .flow code (local streaming or API call)
//! 3. Code blocks are extracted from markdown output
//! 4. Code is written to disk, parsed, and executed
//! 5. On error: auto-retry up to 3 times with error fed back to LLM
//! 6. User iterates with follow-up instructions (multi-turn history)
//!
//! Uses the existing LLM infrastructure (local mode):
//! - stdlib/llm/gguf.flow — GGUF model loading
//! - stdlib/llm/generate.flow — autoregressive generation with KV cache
//! - stdlib/llm/chat.flow — chat template building
//! - stdlib/llm/sampling.flow — top-k/top-p/temperature sampling

pub mod streaming;
pub mod context;
pub mod extraction;
pub mod model;
pub mod knowledge;
pub mod autoskill;
pub mod memory;
pub mod config;

use crate::{CliError, Overrides};
use crate::compiler;
use std::io::{self, BufRead, Write};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

/// Token count during inference (shared between inference thread and main thread).
static INFERENCE_TOKEN_COUNT: AtomicU64 = AtomicU64::new(0);
/// Wall-clock epoch (ms) when inference started generating.
static INFERENCE_START_MS: AtomicU64 = AtomicU64::new(0);
/// Wall-clock epoch (ms) when first token was emitted.
static INFERENCE_FIRST_TOKEN_MS: AtomicU64 = AtomicU64::new(0);
/// Last measured tokens/sec (exposed to .flow via `gguf_tokens_per_sec()`).
pub static INFERENCE_LAST_TOK_PER_SEC: AtomicU64 = AtomicU64::new(0);

fn current_epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Default model path — can be overridden with --model flag.
const DEFAULT_MODEL_PATH: &str = "models/qwen3-1.7b-q5_k_m.gguf";

/// Maximum auto-fix retry attempts when generated code has errors.
const MAX_FIX_ATTEMPTS: usize = 3;

/// Default API endpoint (OpenAI-compatible).
const DEFAULT_API_URL: &str = "https://api.openai.com/v1/chat/completions";

/// Default model name for API mode.
const DEFAULT_API_MODEL: &str = "gpt-4o-mini";

/// OctoFlow Contract — compressed LLM system prompt (~2500 tokens).
/// Sourced from Enhancement team's CONTRACT.md (replaces language-guide-v2.md).
#[allow(dead_code)]
const CONTRACT: &str = include_str!("../../../docs/llm/contract-v1.md");

/// Maximum tool calls per generation turn (ReAct loop budget).
const MAX_TOOL_CALLS: usize = 3;

/// Embedded GBNF grammar for constrained decoding (fallback when filesystem copy is missing).
const BUILTIN_GBNF: &str = include_str!("../../../stdlib/llm/octoflow.gbnf");

/// Build the system prompt from the contract (default: non-thinking mode).
#[cfg(test)]
fn build_system_prompt() -> String {
    build_system_prompt_with_think(false)
}

/// Build the system prompt, optionally enabling thinking mode.
/// Uses the monolithic CONTRACT — kept for tests.
#[cfg(test)]
fn build_system_prompt_with_think(think: bool) -> String {
    let think_directive = if think { "" } else { "\n/no_think\n" };
    format!(
        "You are OctoFlow Chat, an AI coding assistant. Generate .flow code that accomplishes what the user asks.\n\
         Output .flow code inside a ```flow code block. Include comments for clarity.\n\n\
         {}\n{}\n",
        CONTRACT.trim(),
        think_directive,
    )
}

/// Build the system prompt with tool-use instructions (when --allow-net is active).
/// Uses the monolithic CONTRACT — kept for tests.
#[cfg(test)]
fn build_system_prompt_with_tools(think: bool) -> String {
    let think_directive = if think { "" } else { "\n/no_think\n" };
    format!(
        "You are OctoFlow Chat, an AI coding assistant. Generate .flow code that accomplishes what the user asks.\n\
         Output .flow code inside a ```flow code block. Include comments for clarity.\n\n\
         {}\n\n\
         ## Tools\n\
         You have access to web tools. Before generating code, you can search the web or read pages to find APIs, docs, or data.\n\
         To use a tool, output EXACTLY one of these lines (nothing else on that line):\n\
         SEARCH: your search query\n\
         READ: https://example.com/page\n\n\
         After each tool call, you will receive the results. Then generate the code.\n\
         Max {} tool calls per turn. When ready, output the ```flow code block.\n{}",
        CONTRACT.trim(),
        MAX_TOOL_CALLS,
        think_directive,
    )
}

/// Build a context-aware system prompt using the knowledge tree + auto-skill engine.
/// Replaces the monolithic CONTRACT (~2500 tokens) with L0 core (~439 tokens)
/// plus dynamically-selected domain context based on the user's message.
/// Total prompt size: ~450-1600 tokens (vs ~2500 with CONTRACT).
#[allow(dead_code)]
fn build_context_prompt(user_message: &str, think: bool, tools: bool) -> String {
    build_context_prompt_full(user_message, think, tools, None, None)
}

/// Build context-aware prompt with optional persistent memory and user config.
fn build_context_prompt_full(
    user_message: &str,
    think: bool,
    tools: bool,
    mem: Option<&memory::PersistentMemory>,
    cfg: Option<&config::UserConfig>,
) -> String {
    let think_directive = if think { "" } else { "\n/no_think\n" };

    // L0 core is always included (~439 tokens)
    let l0 = knowledge::L0_CORE.trim();

    // User config (~300 tokens: preferences + project instructions)
    let config_context = cfg.map(|c| c.to_context()).unwrap_or_default();

    // Memory summary (~300 tokens if available)
    let memory_summary = mem.map(|m| m.summarize()).unwrap_or_default();

    // Scan user message for domain keywords and assemble matching context
    let scan = autoskill::scan_keywords(user_message);
    let skill_context = autoskill::assemble_skill_context(&scan);

    let tools_section = if tools {
        format!(
            "\n\n## Tools\n\
             You have access to web tools. Before generating code, you can search the web or read pages to find APIs, docs, or data.\n\
             To use a tool, output EXACTLY one of these lines (nothing else on that line):\n\
             SEARCH: your search query\n\
             READ: https://example.com/page\n\n\
             After each tool call, you will receive the results. Then generate the code.\n\
             Max {} tool calls per turn. When ready, output the ```flow code block.",
            MAX_TOOL_CALLS,
        )
    } else {
        String::new()
    };

    format!(
        "You are OctoFlow Chat, an AI coding assistant. Generate .flow code that accomplishes what the user asks.\n\
         Output .flow code inside a ```flow code block. Include comments for clarity.\n\n\
         {}\n{}{}{}{}{}\n",
        l0,
        config_context,
        memory_summary,
        skill_context,
        tools_section,
        think_directive,
    )
}

/// Detect a tool call in LLM output. Returns (tool_name, argument).
fn detect_tool_call(output: &str) -> Option<(&'static str, String)> {
    for line in output.lines() {
        let trimmed = line.trim();
        if let Some(query) = trimmed.strip_prefix("SEARCH:") {
            let query = query.trim();
            if !query.is_empty() {
                return Some(("web_search", query.to_string()));
            }
        }
        if let Some(url) = trimmed.strip_prefix("READ:") {
            let url = url.trim();
            if !url.is_empty() {
                return Some(("web_read", url.to_string()));
            }
        }
    }
    None
}

/// Format search results for injection into LLM context.
fn format_search_results(results: &[crate::Value]) -> String {
    results.iter()
        .take(10)
        .enumerate()
        .map(|(i, v)| {
            if let crate::Value::Map(m) = v {
                let title = m.get("title").map(|v| v.as_str().unwrap_or_default().to_string()).unwrap_or_default();
                let url = m.get("url").map(|v| v.as_str().unwrap_or_default().to_string()).unwrap_or_default();
                let snippet = m.get("snippet").map(|v| v.as_str().unwrap_or_default().to_string()).unwrap_or_default();
                // Truncate long fields
                let title = if title.len() > 80 { format!("{}...", &title[..77]) } else { title };
                let snippet = if snippet.len() > 150 { format!("{}...", &snippet[..147]) } else { snippet };
                format!("{}. {} ({})\n   {}", i + 1, title, url, snippet)
            } else {
                String::new()
            }
        })
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Truncate text to approximately max_chars, breaking at word boundaries.
fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }
    // Find the last space before max_chars
    let truncated = &text[..max_chars];
    if let Some(last_space) = truncated.rfind(' ') {
        format!("{}...", &text[..last_space])
    } else {
        format!("{}...", truncated)
    }
}

/// Check if web tools should be enabled.
/// Currently controlled by the `web_tools` field on ChatSession.
fn web_tools_enabled(session: &ChatSession) -> bool {
    session.web_tools
}

/// Provider backend for code generation.
#[derive(Debug, Clone, PartialEq)]
pub enum Provider {
    /// Local GGUF model inference (default).
    Local,
    /// OpenAI-compatible API via curl.
    Api,
}

/// A single message in the conversation history.
struct ChatMessage {
    role: String,    // "user" or "assistant"
    content: String, // user prompt or generated code
}

/// Persistent chat session with multi-turn history.
struct ChatSession {
    opts: ChatOpts,
    conversation: Vec<ChatMessage>,
    project_context: String,
    /// Version stack: each successful generation pushes code here.
    code_versions: Vec<String>,
    /// Whether web tools (SEARCH/READ) are enabled for this session.
    web_tools: bool,
    #[allow(dead_code)]
    model_loaded: bool,
    /// Persistent memory (cross-session, loaded from ~/.octoflow/memory.json).
    persistent_memory: memory::PersistentMemory,
    /// Session memory (in-memory, tracks loaded modules and corrections).
    session_memory: memory::SessionMemory,
    /// User config (OCTOFLOW.md + ~/.octoflow/preferences.md).
    user_config: config::UserConfig,
}

/// Parsed chat options.
pub struct ChatOpts {
    pub model_path: String,
    pub output_file: String,
    pub max_tokens: usize,
    pub provider: Provider,
    pub api_key: Option<String>,
    pub api_url: Option<String>,
    pub api_model: Option<String>,
    pub web_tools: bool,
    /// Path to GBNF grammar file for constrained decoding (default: built-in OctoFlow grammar).
    /// Set to empty string to disable grammar constraints.
    pub grammar_path: Option<String>,
    /// Enable Qwen3 thinking mode (deep reasoning). Default: off (fast conversation).
    pub think: bool,
    /// Disable all memory loading/saving for privacy.
    pub no_memory: bool,
}

impl Default for ChatOpts {
    fn default() -> Self {
        ChatOpts {
            model_path: DEFAULT_MODEL_PATH.to_string(),
            output_file: "main.flow".to_string(),
            max_tokens: 512,
            provider: Provider::Local,
            api_key: None,
            api_url: None,
            api_model: None,
            web_tools: false,
            grammar_path: None,
            think: false,
            no_memory: false,
        }
    }
}

/// Parse chat subcommand arguments.
pub fn parse_chat_args(args: &[String]) -> Result<ChatOpts, CliError> {
    let mut opts = ChatOpts::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                if i + 1 >= args.len() {
                    return Err(CliError::Compile("--model requires a path".into()));
                }
                opts.model_path = args[i + 1].clone();
                i += 2;
            }
            "--output" | "-o" => {
                if i + 1 >= args.len() {
                    return Err(CliError::Compile("--output requires a file path".into()));
                }
                opts.output_file = args[i + 1].clone();
                i += 2;
            }
            "--max-tokens" => {
                if i + 1 >= args.len() {
                    return Err(CliError::Compile("--max-tokens requires a number".into()));
                }
                opts.max_tokens = args[i + 1].parse::<usize>().map_err(|_| {
                    CliError::Compile(format!("--max-tokens: invalid number '{}'", args[i + 1]))
                })?;
                i += 2;
            }
            "--provider" => {
                if i + 1 >= args.len() {
                    return Err(CliError::Compile("--provider requires a value (local, api)".into()));
                }
                opts.provider = match args[i + 1].as_str() {
                    "local" => Provider::Local,
                    "api" => Provider::Api,
                    other => return Err(CliError::Compile(
                        format!("unknown provider '{}'. Use 'local' or 'api'", other)
                    )),
                };
                i += 2;
            }
            "--api-key" => {
                if i + 1 >= args.len() {
                    return Err(CliError::Compile("--api-key requires a value".into()));
                }
                opts.api_key = Some(args[i + 1].clone());
                i += 2;
            }
            "--api-url" => {
                if i + 1 >= args.len() {
                    return Err(CliError::Compile("--api-url requires a URL".into()));
                }
                opts.api_url = Some(args[i + 1].clone());
                i += 2;
            }
            "--api-model" => {
                if i + 1 >= args.len() {
                    return Err(CliError::Compile("--api-model requires a model name".into()));
                }
                opts.api_model = Some(args[i + 1].clone());
                i += 2;
            }
            "--web-tools" | "--allow-net" => {
                opts.web_tools = true;
                i += 1;
            }
            other if other.starts_with("--allow-net=") => {
                opts.web_tools = true;
                i += 1;
            }
            "--grammar" => {
                if i + 1 >= args.len() {
                    return Err(CliError::Compile("--grammar requires a .gbnf file path".into()));
                }
                opts.grammar_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--no-grammar" => {
                opts.grammar_path = Some(String::new()); // empty = disabled
                i += 1;
            }
            "--think" => {
                opts.think = true;
                i += 1;
            }
            "--no-memory" => {
                opts.no_memory = true;
                i += 1;
            }
            other => {
                return Err(CliError::Compile(format!("unknown chat option: '{}'", other)));
            }
        }
    }

    // Resolve API key: --api-key flag > OCTOFLOW_API_KEY > OPENAI_API_KEY
    if opts.provider == Provider::Api && opts.api_key.is_none() {
        if let Ok(key) = std::env::var("OCTOFLOW_API_KEY") {
            if !key.is_empty() {
                opts.api_key = Some(key);
            }
        }
        if opts.api_key.is_none() {
            if let Ok(key) = std::env::var("OPENAI_API_KEY") {
                if !key.is_empty() {
                    opts.api_key = Some(key);
                }
            }
        }
    }

    Ok(opts)
}

/// Run the chat REPL loop.
pub fn run_chat(args: &[String]) -> Result<(), CliError> {
    let opts = parse_chat_args(args)?;
    let no_mem = opts.no_memory;

    let session = match opts.provider {
        Provider::Api => {
            // API mode: validate key, skip model discovery
            if opts.api_key.is_none() {
                streaming::print_banner();
                eprintln!("{}error: API mode requires an API key.{}", streaming::RED, streaming::RESET);
                eprintln!();
                eprintln!("Set one of:");
                eprintln!("  --api-key <key>");
                eprintln!("  OCTOFLOW_API_KEY environment variable");
                eprintln!("  OPENAI_API_KEY environment variable");
                return Ok(());
            }

            let api_url = opts.api_url.as_deref().unwrap_or(DEFAULT_API_URL);
            let api_model = opts.api_model.as_deref().unwrap_or(DEFAULT_API_MODEL);

            // Extract host from URL for display
            let host = api_url
                .strip_prefix("https://").or_else(|| api_url.strip_prefix("http://"))
                .and_then(|s| s.split('/').next())
                .unwrap_or(api_url);

            let web_tools = opts.web_tools;
            streaming::print_banner();
            eprintln!("Provider: {}api{} ({} @ {})", streaming::CYAN, streaming::RESET, api_model, host);
            eprintln!("Output: {}", opts.output_file);
            eprintln!();

            ChatSession {
                opts,
                conversation: Vec::new(),
                project_context: context::scan_project_context("."),
                code_versions: Vec::new(),
                web_tools,
                model_loaded: false,
                persistent_memory: if no_mem { memory::PersistentMemory::new() } else { memory::PersistentMemory::load() },
                session_memory: memory::SessionMemory::new(),
                user_config: if no_mem { config::UserConfig { global_prefs: None, global_memory: None, project_instructions: None, project_memory: None } } else { config::UserConfig::load() },
            }
        }
        Provider::Local => {
            // Local mode: model discovery
            let model_path = if std::path::Path::new(&opts.model_path).exists() {
                opts.model_path.clone()
            } else if opts.model_path == DEFAULT_MODEL_PATH {
                match model::find_model() {
                    Some(p) => p.to_string_lossy().into_owned(),
                    None => {
                        streaming::print_banner();
                        model::print_download_instructions();
                        return Ok(());
                    }
                }
            } else {
                eprintln!("Model not found: {}", opts.model_path);
                eprintln!("Use --model <path> to specify a valid GGUF model.");
                return Ok(());
            };

            streaming::print_banner();
            eprintln!("Provider: {}local{} ({})", streaming::CYAN, streaming::RESET, model_path);
            if opts.think {
                eprintln!("Mode: {}thinking{} (deep reasoning)", streaming::CYAN, streaming::RESET);
            }
            eprintln!("Output: {}", opts.output_file);
            eprintln!();

            ChatSession {
                opts: ChatOpts {
                    model_path,
                    output_file: opts.output_file,
                    max_tokens: opts.max_tokens,
                    provider: Provider::Local,
                    api_key: None,
                    api_url: None,
                    api_model: None,
                    web_tools: opts.web_tools,
                    grammar_path: opts.grammar_path,
                    think: opts.think,
                    no_memory: no_mem,
                },
                conversation: Vec::new(),
                project_context: context::scan_project_context("."),
                code_versions: Vec::new(),
                web_tools: opts.web_tools,
                model_loaded: false,
                persistent_memory: if no_mem { memory::PersistentMemory::new() } else { memory::PersistentMemory::load() },
                session_memory: memory::SessionMemory::new(),
                user_config: if no_mem { config::UserConfig { global_prefs: None, global_memory: None, project_instructions: None, project_memory: None } } else { config::UserConfig::load() },
            }
        }
    };

    let mut session = session;

    let stdin = io::stdin();
    let mut reader = stdin.lock();

    loop {
        // Prompt
        eprint!("{}you> {}", streaming::BOLD, streaming::RESET);
        io::stderr().flush().ok();

        // Multiline input: backslash continuation
        let input = match read_multiline_input(&mut reader) {
            Some(s) => s,
            None => break, // EOF
        };

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // Handle special commands (prefix+argument parsing)
        if input.starts_with(':') {
            let (cmd, arg) = match input.find(' ') {
                Some(pos) => (&input[..pos], input[pos + 1..].trim()),
                None => (input, ""),
            };
            match cmd {
                ":help" | ":h" => print_chat_help(),
                ":quit" | ":q" | ":exit" => break,
                ":run" | ":r" => {
                    match run_generated_file(&session.opts.output_file) {
                        Ok(()) => {}
                        Err(e) => streaming::print_error(&format!("{}", e)),
                    }
                }
                ":show" | ":s" => show_generated_file(&session.opts.output_file),
                ":clear" | ":c" => {
                    match arg {
                        "history" => {
                            session.conversation.clear();
                            eprintln!("Cleared conversation history.");
                        }
                        "file" => {
                            if std::path::Path::new(&session.opts.output_file).exists() {
                                std::fs::remove_file(&session.opts.output_file).ok();
                                eprintln!("Cleared {}", session.opts.output_file);
                            }
                            session.code_versions.clear();
                        }
                        _ => {
                            // :clear with no arg — clear everything (backward compat)
                            if std::path::Path::new(&session.opts.output_file).exists() {
                                std::fs::remove_file(&session.opts.output_file).ok();
                                eprintln!("Cleared {}", session.opts.output_file);
                            }
                            session.conversation.clear();
                            session.code_versions.clear();
                        }
                    }
                }
                ":history" => {
                    for msg in &session.conversation {
                        let prefix = if msg.role == "user" { "you" } else { "ai" };
                        eprintln!("[{}] {}", prefix, msg.content.lines().next().unwrap_or(""));
                    }
                }
                ":undo" | ":u" => {
                    if session.code_versions.len() < 2 {
                        eprintln!("Nothing to undo.");
                    } else {
                        session.code_versions.pop();
                        let prev = session.code_versions.last().unwrap().clone();
                        std::fs::write(&session.opts.output_file, &prev).ok();
                        // Remove last conversation exchange (user + assistant)
                        if session.conversation.len() >= 2 {
                            session.conversation.pop();
                            session.conversation.pop();
                        }
                        eprintln!("Restored previous version ({} versions remain).", session.code_versions.len());
                        show_generated_file(&session.opts.output_file);
                    }
                }
                ":diff" | ":d" => {
                    if session.code_versions.len() < 2 {
                        eprintln!("Need at least 2 versions to diff.");
                    } else {
                        let old = &session.code_versions[session.code_versions.len() - 2];
                        let new = session.code_versions.last().unwrap();
                        print_diff(old, new);
                    }
                }
                ":edit" | ":e" => {
                    let editor = std::env::var("EDITOR")
                        .or_else(|_| std::env::var("VISUAL"))
                        .unwrap_or_else(|_| {
                            if cfg!(windows) { "notepad".to_string() } else { "nano".to_string() }
                        });
                    if !std::path::Path::new(&session.opts.output_file).exists() {
                        eprintln!("No generated file to edit.");
                    } else {
                        let status = Command::new(&editor)
                            .arg(&session.opts.output_file)
                            .status();
                        match status {
                            Ok(s) if s.success() => {
                                if let Ok(edited) = std::fs::read_to_string(&session.opts.output_file) {
                                    session.code_versions.push(edited);
                                    eprintln!("Saved edit. Running...");
                                    match run_generated_file(&session.opts.output_file) {
                                        Ok(()) => {}
                                        Err(e) => streaming::print_error(&format!("{}", e)),
                                    }
                                }
                            }
                            Ok(_) => eprintln!("Editor exited with error."),
                            Err(e) => eprintln!("Failed to launch {}: {}", editor, e),
                        }
                    }
                }
                _ => eprintln!("Unknown command: {}. Type :help for commands.", cmd),
            }
            continue;
        }

        // Add user message to history
        session.conversation.push(ChatMessage {
            role: "user".to_string(),
            content: input.to_string(),
        });

        // Generate code via LLM with auto-fix loop
        eprintln!();
        let result = generate_with_auto_fix(&mut session, input);

        match result {
            Ok(code) => {
                // Push to version stack
                session.code_versions.push(code.clone());
                // Save to history
                session.conversation.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: code,
                });
                // Update session memory
                session.session_memory.turn_count += 1;
            }
            Err(e) => {
                streaming::print_error(&format!("{}", e));
                session.session_memory.corrections_made += 1;
            }
        }

        eprintln!();
    }

    // Session end: update persistent memory and save
    if !session.opts.no_memory {
        session.persistent_memory.total_sessions += 1;
        for module in &session.session_memory.modules_used {
            session.persistent_memory.record_module(module);
        }
        session.persistent_memory.save();

        // Auto-save human-readable memory.md files
        if let Some(entry) = config::build_memory_entry(
            &session.session_memory.modules_used,
            session.session_memory.corrections_made,
            session.session_memory.turn_count,
        ) {
            config::save_global_memory(&entry);
            config::save_project_memory(&entry);
        }
    }

    Ok(())
}

/// Generate code with optional ReAct tool-use loop, then auto-fix errors.
fn generate_with_auto_fix(session: &mut ChatSession, user_prompt: &str) -> Result<String, CliError> {
    // Phase 1: ReAct tool-use loop (if web tools enabled)
    let enriched_prompt = if web_tools_enabled(session) {
        react_tool_loop(session, user_prompt)?
    } else {
        user_prompt.to_string()
    };

    // Phase 2: generate + auto-fix loop
    generate_and_fix(session, &enriched_prompt)
}

/// ReAct tool-use loop: let the LLM call SEARCH/READ before generating code.
fn react_tool_loop(session: &mut ChatSession, user_prompt: &str) -> Result<String, CliError> {
    let mut context = user_prompt.to_string();
    let mut tool_calls = 0;

    while tool_calls < MAX_TOOL_CALLS {
        // Generate with tool-use system prompt
        eprint!("{}Thinking...{}", streaming::DIM, streaming::RESET);
        let raw = match session.opts.provider {
            Provider::Api => generate_code_api_with_tools(&session.opts, &context, &session.project_context, &session.conversation, &session.persistent_memory, &session.user_config),
            Provider::Local => generate_code_with_tools(&session.opts, &context, &session.project_context, &session.conversation, &session.persistent_memory, &session.user_config),
        };
        let output = match raw {
            Ok(o) => { streaming::clear_line(); o }
            Err(e) => { streaming::clear_line(); return Err(e); }
        };

        // Check for tool call
        if let Some((tool, arg)) = detect_tool_call(&output) {
            match tool {
                "web_search" => {
                    eprintln!("{}  Searching: {}{}", streaming::DIM, arg, streaming::RESET);
                    match crate::io::web::web_search(&arg) {
                        Ok(results) => {
                            let formatted = format_search_results(&results);
                            context = format!(
                                "{}\n\n[Search results for \"{}\":\n{}\n]",
                                context, arg, formatted
                            );
                        }
                        Err(e) => {
                            context = format!(
                                "{}\n\n[Search failed: {}. Generate code with available information.]",
                                context, e
                            );
                        }
                    }
                }
                "web_read" => {
                    eprintln!("{}  Reading: {}{}", streaming::DIM, arg, streaming::RESET);
                    match crate::io::web::web_read(&arg) {
                        Ok(page) => {
                            let (title, text) = if let crate::Value::Map(ref m) = page {
                                (
                                    m.get("title").map(|v| v.as_str().unwrap_or_default().to_string()).unwrap_or_default(),
                                    m.get("text").map(|v| v.as_str().unwrap_or_default().to_string()).unwrap_or_default(),
                                )
                            } else {
                                (String::new(), String::new())
                            };
                            let truncated = truncate_text(&text, 2000);
                            context = format!(
                                "{}\n\n[Page content from {}:\nTitle: {}\n{}\n]",
                                context, arg, title, truncated
                            );
                        }
                        Err(e) => {
                            context = format!(
                                "{}\n\n[Read failed: {}. Generate code with available information.]",
                                context, e
                            );
                        }
                    }
                }
                _ => {}
            }
            tool_calls += 1;
        } else {
            // No tool call — LLM is ready to generate code
            break;
        }
    }

    if tool_calls >= MAX_TOOL_CALLS {
        context = format!("{}\n\n[Tool budget reached. Generate code now.]", context);
    }

    Ok(context)
}

/// Generate + auto-fix loop (phase 2 of the pipeline).
fn generate_and_fix(session: &mut ChatSession, user_prompt: &str) -> Result<String, CliError> {
    let mut last_error: Option<String> = None;
    let mut last_code: Option<String> = None;

    for attempt in 0..MAX_FIX_ATTEMPTS {
        // Build prompt: original request + failing code + error feedback if retrying
        let prompt = if let Some(ref err) = last_error {
            streaming::print_retry(attempt + 1, MAX_FIX_ATTEMPTS);
            build_fix_prompt(user_prompt, last_code.as_deref(), err)
        } else {
            user_prompt.to_string()
        };

        // Generate (branch on provider)
        eprint!("{}Generating...{}", streaming::DIM, streaming::RESET);
        let generate_result = match session.opts.provider {
            Provider::Api => generate_code_api(&session.opts, &prompt, &session.project_context, &session.conversation, &session.persistent_memory, &session.user_config),
            Provider::Local => generate_code(&session.opts, &prompt, &session.project_context, &session.conversation, &session.persistent_memory, &session.user_config),
        };
        let raw_output = match generate_result {
            Ok(output) => {
                streaming::clear_line();
                output
            }
            Err(e) => {
                streaming::clear_line();
                return Err(e);
            }
        };

        // Extract code from markdown blocks
        let code = extraction::extract_code(&raw_output);

        if code.trim().is_empty() {
            last_error = Some("LLM generated empty output".to_string());
            continue;
        }

        // Save code for retry context
        last_code = Some(code.clone());

        // Write to output file
        std::fs::write(&session.opts.output_file, &code)
            .map_err(|e| CliError::Io(format!("write {}: {}", session.opts.output_file, e)))?;

        // Show the generated code
        eprintln!("{}--- {} ---{}", streaming::DIM, session.opts.output_file, streaming::RESET);
        for (i, line) in code.lines().enumerate() {
            eprintln!("{}{:4}{} | {}", streaming::DIM, i + 1, streaming::RESET, line);
        }
        eprintln!("{}---{}", streaming::DIM, streaming::RESET);
        eprintln!();

        // Try parsing
        match octoflow_parser::parse(&code) {
            Err(e) => {
                last_error = Some(format!("Parse error: {}", e));
                streaming::print_error(&format!("Parse error: {}", e));
                continue;
            }
            Ok(program) => {
                let base_dir = std::path::Path::new(&session.opts.output_file)
                    .parent()
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_else(|| ".".to_string());

                // Pre-flight validation before execution
                let report = crate::analysis::preflight::validate(&program, &base_dir);
                if !report.passed {
                    let preflight_json = crate::format_preflight_json(&report, &code);
                    let first_err = report.checks.iter()
                        .flat_map(|c| &c.errors)
                        .next();
                    let err_msg = if let Some(e) = first_err {
                        if e.suggestion.is_empty() {
                            format!("Pre-flight: {} (line {})", e.message, e.line)
                        } else {
                            format!("Pre-flight: {} (line {}). Fix: {}", e.message, e.line, e.suggestion)
                        }
                    } else {
                        "Pre-flight validation failed".to_string()
                    };
                    last_error = Some(format!("{}\n{}", err_msg, preflight_json));
                    streaming::print_error(&err_msg);
                    continue;
                }

                // Execute with sandbox: scoped to project dir, iteration limit
                let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
                let overrides = Overrides {
                    allow_read: crate::PermScope::AllowScoped(vec![cwd.clone()]),
                    allow_write: crate::PermScope::AllowScoped(vec![cwd]),
                    allow_net: crate::PermScope::Deny,
                    max_iters: Some(1_000_000),
                    ..Default::default()
                };

                compiler::capture_output_start();
                let exec_result = compiler::execute(&program, &base_dir, &overrides);
                let captured = compiler::capture_output_take();

                match exec_result {
                    Ok(_) => {
                        if !captured.is_empty() {
                            eprintln!("{}Output:{}", streaming::DIM, streaming::RESET);
                            eprint!("{}", captured);
                        }
                        streaming::print_success(&format!("Saved to {}", session.opts.output_file));
                        return Ok(code);
                    }
                    Err(e) => {
                        if !captured.is_empty() {
                            eprint!("{}", captured);
                        }
                        // Include structured error JSON for the LLM
                        let json = crate::format_error_json_full(&e, Some(&code));
                        last_error = Some(format!("{}\n{}", e, json));
                        streaming::print_error(&format!("{}", e));
                        continue;
                    }
                }
            }
        }
    }

    Err(CliError::Compile(format!(
        "Failed after {} attempts. Last error: {}",
        MAX_FIX_ATTEMPTS,
        last_error.unwrap_or_else(|| "unknown".to_string())
    )))
}

/// Generate a short response for tool-use detection (local mode).
/// Uses the tool-aware system prompt and shorter max tokens.
fn generate_code_with_tools(
    opts: &ChatOpts,
    user_prompt: &str,
    project_context: &str,
    history: &[ChatMessage],
    mem: &memory::PersistentMemory,
    cfg: &config::UserConfig,
) -> Result<String, CliError> {
    // For local mode, run normal generation with tool-aware prompt
    // Small models may output SEARCH:/READ: or go straight to code
    let mut tool_opts = ChatOpts {
        model_path: opts.model_path.clone(),
        output_file: opts.output_file.clone(),
        max_tokens: 128, // Short — just need tool call or start of code
        provider: opts.provider.clone(),
        api_key: opts.api_key.clone(),
        api_url: opts.api_url.clone(),
        api_model: opts.api_model.clone(),
        web_tools: true,
        grammar_path: opts.grammar_path.clone(),
        think: opts.think,
        no_memory: opts.no_memory,
    };
    let _ = &mut tool_opts; // Silence unused warning
    // Use the existing generate_code but with context-aware system prompt
    let mut full_prompt = build_context_prompt_full(user_prompt, opts.think, true, Some(mem), Some(cfg));
    if !project_context.is_empty() {
        full_prompt.push_str("\n\n");
        full_prompt.push_str(project_context);
    }
    let history_start = if history.len() > 8 { history.len() - 8 } else { 0 };
    for msg in &history[history_start..] {
        let truncated = truncate_for_context(&msg.content, 10, 500);
        full_prompt.push_str(&format!(
            "\n\n{}: {}",
            if msg.role == "user" { "User" } else { "Assistant" },
            truncated
        ));
    }
    full_prompt.push_str(&format!("\n\nUser request: {}\n\nRespond with SEARCH/READ tool call or ```flow code:", user_prompt));
    // For now, return the prompt as if the model would process it
    // In local mode, this actually invokes the LLM — reuse generate_code
    generate_code(opts, user_prompt, project_context, history, mem, cfg)
}

/// Generate a short response for tool-use detection (API mode).
fn generate_code_api_with_tools(
    opts: &ChatOpts,
    user_prompt: &str,
    project_context: &str,
    history: &[ChatMessage],
    mem: &memory::PersistentMemory,
    cfg: &config::UserConfig,
) -> Result<String, CliError> {
    let api_key = opts.api_key.as_deref().unwrap_or("");
    let api_url = opts.api_url.as_deref().unwrap_or(DEFAULT_API_URL);
    let api_model = opts.api_model.as_deref().unwrap_or(DEFAULT_API_MODEL);

    let mut system_content = build_context_prompt_full(user_prompt, opts.think, true, Some(mem), Some(cfg));
    if !project_context.is_empty() {
        system_content.push_str("\n\n<project-context>\n");
        system_content.push_str(project_context);
        system_content.push_str("\n</project-context>");
    }

    let mut messages = Vec::new();
    messages.push(format!(
        r#"{{"role":"system","content":{}}}"#,
        json_escape(&system_content)
    ));

    let history_start = if history.len() > 8 { history.len() - 8 } else { 0 };
    for msg in &history[history_start..] {
        messages.push(format!(
            r#"{{"role":"{}","content":{}}}"#,
            if msg.role == "user" { "user" } else { "assistant" },
            json_escape(&msg.content)
        ));
    }

    messages.push(format!(
        r#"{{"role":"user","content":{}}}"#,
        json_escape(user_prompt)
    ));

    let messages_json = messages.join(",");
    let request_body = format!(
        r#"{{"model":"{}","messages":[{}],"max_tokens":256,"temperature":0.3}}"#,
        api_model, messages_json
    );

    let temp_request = ".octoflow_chat_tools.json";
    std::fs::write(temp_request, &request_body)
        .map_err(|e| CliError::Io(format!("write temp request: {}", e)))?;

    let output = Command::new("curl")
        .args(&[
            "-s", "-X", "POST", api_url,
            "-H", "Content-Type: application/json",
            "-H", &format!("Authorization: Bearer {}", api_key),
            "-d", &format!("@{}", temp_request),
        ])
        .output()
        .map_err(|e| CliError::Io(format!("curl failed: {}", e)))?;

    std::fs::remove_file(temp_request).ok();

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(CliError::Compile(format!("curl error: {}", stderr.trim())));
    }

    let response = String::from_utf8_lossy(&output.stdout).to_string();
    let content = extract_api_content(&response)
        .ok_or_else(|| {
            if let Some(err_msg) = extract_api_error(&response) {
                CliError::Compile(format!("API error: {}", err_msg))
            } else {
                CliError::Compile("failed to parse tool-use API response".into())
            }
        })?;

    Ok(content)
}

/// Generate .flow code by running the LLM inference pipeline.
fn generate_code(
    opts: &ChatOpts,
    user_prompt: &str,
    project_context: &str,
    history: &[ChatMessage],
    mem: &memory::PersistentMemory,
    cfg: &config::UserConfig,
) -> Result<String, CliError> {
    // Build the full prompt: L0 core + config + memory + auto-skill context + project files + history + user request
    let mut full_prompt = build_context_prompt_full(user_prompt, opts.think, false, Some(mem), Some(cfg));

    if !project_context.is_empty() {
        full_prompt.push_str("\n\n");
        full_prompt.push_str(project_context);
    }

    // Include conversation history (last 8 messages, truncated per message)
    let history_start = if history.len() > 8 { history.len() - 8 } else { 0 };
    for msg in &history[history_start..] {
        let truncated = truncate_for_context(&msg.content, 10, 500);
        full_prompt.push_str(&format!(
            "\n\n{}: {}",
            if msg.role == "user" { "User" } else { "Assistant" },
            truncated
        ));
    }

    full_prompt.push_str(&format!("\n\nUser request: {}\n\nWrite the .flow code:", user_prompt));

    // Resolve grammar path: default to built-in, --grammar overrides, --no-grammar disables
    let grammar_path = match &opts.grammar_path {
        Some(p) if p.is_empty() => None, // --no-grammar
        Some(p) => Some(p.clone()),
        None => {
            // Default: use filesystem grammar, fall back to embedded copy
            let builtin = "stdlib/llm/octoflow.gbnf";
            if std::path::Path::new(builtin).exists() {
                Some(builtin.to_string())
            } else {
                // Write embedded grammar to temp file
                let temp = std::env::temp_dir().join("octoflow_builtin.gbnf");
                if std::fs::write(&temp, BUILTIN_GBNF).is_ok() {
                    Some(temp.to_string_lossy().into_owned())
                } else {
                    None
                }
            }
        }
    };

    // Grammar setup snippet (injected into .flow script if grammar is enabled)
    let grammar_setup = if let Some(ref gpath) = grammar_path {
        let escaped = gpath.replace('\\', "/").replace('"', "\\\"");
        format!(
            r#"
let _grammar_ok = grammar_load("{}")
"#,
            escaped
        )
    } else {
        String::new()
    };

    // Grammar-constrained sampling snippet
    let grammar_sample = if grammar_path.is_some() {
        r#"
    let masked_logits = grammar_mask(logits, vocab)
    let sampled_idx = sample_top_p(masked_logits, 0.9, 0.7)
    let token_text = vocab[int(sampled_idx)]
    grammar_advance(token_text)
"#
    } else {
        r#"
    let sampled_idx = sample_top_p(logits, 0.9, 0.7)
    let token_text = vocab[int(sampled_idx)]
"#
    };

    // Create a temporary .flow script that runs generation
    let script = format!(
        r#"use "gguf"
use "generate"
use "chat"
use "sampling"
use "ops"

let model_path = "{model_path}"
let prompt = "{prompt}"
let max_tokens = {max_tokens}.0
let max_seq = {max_seq}.0

let model = gguf_load_from_file(model_path)
let vocab = gguf_load_vocab(model_path)
{grammar_setup}
let n_embd = map_get(model, "n_embd")
let n_layer = map_get(model, "n_layer")
let eps = gguf_meta_default(model, "attention.layer_norm_rms_epsilon", 0.00001)

let chat_tokens = resolve_chat_tokens(model_path, model, vocab)
let prompt_ids = build_chat_tokens(model_path, model, vocab, prompt)
let stop_tokens = get_stop_tokens(model, chat_tokens)
let n_prompt = len(prompt_ids)

let out_norm_w = gguf_load_tensor(model_path, model, "output_norm.weight")

let total_steps = n_prompt + max_tokens
let mut current_token = prompt_ids[0]
let mut generated_text = ""
let mut seq_pos = 0.0
let mut gen_count = 0.0
let mut done = 0.0

while seq_pos < total_steps
  if done == 1.0
    seq_pos = total_steps
  end
  if seq_pos >= max_seq
    seq_pos = total_steps
  end
  if seq_pos < total_steps

  let tok_emb = gguf_load_tensor(model_path, model, "token_embd.weight", current_token)
  let mut hidden = []
  let mut ei = 0.0
  while ei < n_embd
    push(hidden, tok_emb[int(ei)])
    ei = ei + 1.0
  end

  let mut layer_idx = 0.0
  while layer_idx < n_layer
    let next_layer = layer_idx + 1.0
    if next_layer < n_layer
      gguf_prefetch_layer(model_path, model, next_layer)
    end
    let layer_out = gguf_infer_layer(model_path, model, hidden, layer_idx, seq_pos, max_seq)
    if next_layer < n_layer
      gguf_prefetch_complete()
    end
    let mut hi = 0.0
    while hi < n_embd
      hidden[int(hi)] = layer_out[int(hi)]
      hi = hi + 1.0
    end
    if layer_idx >= 2.0
      let evict_idx = layer_idx - 2.0
      let _ev = gguf_evict_layer(model_path, model, evict_idx)
    end
    layer_idx = layer_idx + 1.0
  end

  let prefill_end = n_prompt - 1.0
  if seq_pos < prefill_end
    let npos = seq_pos + 1.0
    current_token = prompt_ids[int(npos)]
  else
    let normed_out = rmsnorm_cpu(hidden, out_norm_w, n_embd, eps)
    let logits = gguf_matvec(model_path, model, "token_embd.weight", normed_out)
{grammar_sample}
    chat_emit_token(token_text)
    generated_text = generated_text + token_text
    gen_count = gen_count + 1.0

    if is_stop_token(stop_tokens, sampled_idx) == 1.0
      done = 1.0
    end
    if gen_count >= max_tokens
      done = 1.0
    end
    current_token = sampled_idx
  end

  end
  seq_pos = seq_pos + 1.0
end

let decoded = bpe_decode(generated_text)
print("{{decoded}}")
"#,
        model_path = opts.model_path.replace('\\', "/").replace('"', "\\\""),
        prompt = full_prompt.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n").replace('\r', ""),
        max_tokens = opts.max_tokens,
        max_seq = opts.max_tokens + 256,
        grammar_setup = grammar_setup,
        grammar_sample = grammar_sample,
    );

    // Write temp script
    let temp_path = ".octoflow_chat_gen.flow";
    std::fs::write(temp_path, &script)
        .map_err(|e| CliError::Io(format!("write temp script: {}", e)))?;

    // Parse the temp script before spawning
    let source = std::fs::read_to_string(temp_path)
        .map_err(|e| CliError::Io(format!("{}: {}", temp_path, e)))?;

    let program = octoflow_parser::parse(&source)
        .map_err(|e| CliError::Parse(format!("{}", e)))?;

    let base_dir = "stdlib/llm".to_string();
    let overrides = Overrides {
        allow_read: crate::PermScope::AllowAll,
        allow_write: crate::PermScope::AllowAll,
        allow_net: crate::PermScope::AllowAll,
        allow_ffi: true,
        max_iters: Some(1_000_000),
        ..Default::default()
    };

    // Reset inference metrics
    INFERENCE_TOKEN_COUNT.store(0, Ordering::Relaxed);
    INFERENCE_START_MS.store(current_epoch_ms(), Ordering::Relaxed);
    INFERENCE_FIRST_TOKEN_MS.store(0, Ordering::Relaxed);

    // Run inference with 60-second wall-clock timeout.
    // Token callback and output capture are thread-local, so they must be
    // set up on the inference thread (not the main thread).
    let program_clone = program.clone();
    let base_dir_clone = base_dir.clone();
    let overrides_clone = overrides.clone();
    let (tx, rx) = std::sync::mpsc::channel::<(Result<(usize, bool), CliError>, String)>();
    std::thread::spawn(move || {
        compiler::capture_output_start();
        compiler::set_token_callback(Box::new(|token: &str| {
            use std::io::Write;
            let stderr = std::io::stderr();
            let mut out = stderr.lock();
            let _ = write!(out, "{}{}{}", "\x1b[36m", token, "\x1b[0m");
            let _ = out.flush();
            let prev = INFERENCE_TOKEN_COUNT.fetch_add(1, Ordering::Relaxed);
            if prev == 0 {
                INFERENCE_FIRST_TOKEN_MS.store(current_epoch_ms(), Ordering::Relaxed);
            }
        }));
        let result = compiler::execute(&program_clone, &base_dir_clone, &overrides_clone);
        compiler::clear_token_callback();
        let output = compiler::capture_output_take();
        let _ = tx.send((result, output));
    });

    let (exec_result, output) = match rx.recv_timeout(std::time::Duration::from_secs(60)) {
        Ok(pair) => pair,
        Err(_) => {
            std::fs::remove_file(temp_path).ok();
            return Err(CliError::Compile(
                "Generation timed out after 60s. Try a simpler prompt or use --provider api.".into()
            ));
        }
    };

    // Print inference performance summary
    let end_ms = current_epoch_ms();
    let start_ms = INFERENCE_START_MS.load(Ordering::Relaxed);
    let first_ms = INFERENCE_FIRST_TOKEN_MS.load(Ordering::Relaxed);
    let token_count = INFERENCE_TOKEN_COUNT.load(Ordering::Relaxed);
    let elapsed_ms = end_ms.saturating_sub(start_ms);
    if token_count > 0 && elapsed_ms > 0 {
        let tok_per_sec = token_count as f64 / (elapsed_ms as f64 / 1000.0);
        let first_token_ms = if first_ms > 0 { first_ms.saturating_sub(start_ms) } else { 0 };
        // Extract model name from path for display
        let model_name = std::path::Path::new(&opts.model_path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "local".into());
        eprintln!(); // newline after streaming tokens
        eprintln!(
            "{}[{}] {} tokens in {:.1}s ({:.1} tok/s, first token: {}ms){}",
            streaming::DIM, model_name, token_count,
            elapsed_ms as f64 / 1000.0, tok_per_sec, first_token_ms,
            streaming::RESET,
        );
        // Store for gguf_tokens_per_sec() builtin
        INFERENCE_LAST_TOK_PER_SEC.store(tok_per_sec.to_bits(), Ordering::Relaxed);
    } else {
        eprintln!(); // newline after streaming tokens
    }

    std::fs::remove_file(temp_path).ok();
    exec_result?;

    if output.trim().is_empty() {
        return Err(CliError::Compile("LLM generated empty output".into()));
    }

    Ok(output)
}

/// Generate .flow code by calling an OpenAI-compatible API via curl.
fn generate_code_api(
    opts: &ChatOpts,
    user_prompt: &str,
    project_context: &str,
    history: &[ChatMessage],
    mem: &memory::PersistentMemory,
    cfg: &config::UserConfig,
) -> Result<String, CliError> {
    let api_key = opts.api_key.as_deref().unwrap_or("");
    let api_url = opts.api_url.as_deref().unwrap_or(DEFAULT_API_URL);
    let api_model = opts.api_model.as_deref().unwrap_or(DEFAULT_API_MODEL);

    // Build messages array for the chat completions API — context-aware prompt
    let mut system_content = build_context_prompt_full(user_prompt, opts.think, false, Some(mem), Some(cfg));
    if !project_context.is_empty() {
        system_content.push_str("\n\n<project-context>\n");
        system_content.push_str(project_context);
        system_content.push_str("\n</project-context>");
    }

    // Build JSON messages array
    let mut messages = Vec::new();
    messages.push(format!(
        r#"{{"role":"system","content":{}}}"#,
        json_escape(&system_content)
    ));

    // Include conversation history (last 8 messages max for API)
    let history_start = if history.len() > 8 { history.len() - 8 } else { 0 };
    for msg in &history[history_start..] {
        messages.push(format!(
            r#"{{"role":"{}","content":{}}}"#,
            if msg.role == "user" { "user" } else { "assistant" },
            json_escape(&msg.content)
        ));
    }

    // Current user request
    messages.push(format!(
        r#"{{"role":"user","content":{}}}"#,
        json_escape(user_prompt)
    ));

    let messages_json = messages.join(",");
    let request_body = format!(
        r#"{{"model":"{}","messages":[{}],"max_tokens":2048,"temperature":0.3}}"#,
        api_model, messages_json
    );

    // Write request to temp file (avoids shell escaping issues)
    let temp_request = ".octoflow_chat_request.json";
    std::fs::write(temp_request, &request_body)
        .map_err(|e| CliError::Io(format!("write temp request: {}", e)))?;

    // Call curl with spinner
    let spinner_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let spinner_flag = spinner_stop.clone();
    let spinner_thread = std::thread::spawn(move || {
        let frames = ['|', '/', '-', '\\'];
        let mut i = 0;
        while !spinner_flag.load(std::sync::atomic::Ordering::Relaxed) {
            eprint!("\r{}Waiting for API response... {}{}", streaming::DIM, frames[i % 4], streaming::RESET);
            io::stderr().flush().ok();
            std::thread::sleep(std::time::Duration::from_millis(200));
            i += 1;
        }
        streaming::clear_line();
    });

    let output = Command::new("curl")
        .args(&[
            "-s", "-X", "POST", api_url,
            "-H", "Content-Type: application/json",
            "-H", &format!("Authorization: Bearer {}", api_key),
            "-d", &format!("@{}", temp_request),
        ])
        .output()
        .map_err(|e| CliError::Io(format!("curl failed: {}", e)));

    // Stop spinner
    spinner_stop.store(true, std::sync::atomic::Ordering::Relaxed);
    spinner_thread.join().ok();

    let output = output?;

    // Clean up temp file
    std::fs::remove_file(temp_request).ok();

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(CliError::Compile(format!("curl error: {}", stderr.trim())));
    }

    let response = String::from_utf8_lossy(&output.stdout).to_string();

    // Parse JSON response to extract content
    // Look for "content": "..." in the response
    let content = extract_api_content(&response)
        .ok_or_else(|| {
            // Check for API error messages
            if let Some(err_msg) = extract_api_error(&response) {
                CliError::Compile(format!("API error: {}", err_msg))
            } else {
                CliError::Compile(format!("failed to parse API response: {}", &response[..response.len().min(200)]))
            }
        })?;

    if content.trim().is_empty() {
        return Err(CliError::Compile("API returned empty content".into()));
    }

    Ok(content)
}

/// Escape a string as a JSON string value (with surrounding quotes).
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Extract the assistant's content from an OpenAI-compatible JSON response.
/// Looks for `"content": "..."` in `choices[0].message`.
fn extract_api_content(response: &str) -> Option<String> {
    // Find "choices" array, then "message", then "content"
    // Simple JSON extraction without a full parser
    let choices_idx = response.find("\"choices\"")?;
    let after_choices = &response[choices_idx..];
    let content_idx = after_choices.find("\"content\"")?;
    let after_content = &after_choices[content_idx + 9..]; // skip `"content"`

    // Skip whitespace and colon
    let after_colon = after_content.trim_start();
    let after_colon = after_colon.strip_prefix(':')?;
    let after_colon = after_colon.trim_start();

    // Parse the JSON string value
    if !after_colon.starts_with('"') {
        return None;
    }
    parse_json_string(after_colon)
}

/// Extract an error message from an API error response.
fn extract_api_error(response: &str) -> Option<String> {
    let error_idx = response.find("\"error\"")?;
    let after_error = &response[error_idx..];
    let message_idx = after_error.find("\"message\"")?;
    let after_message = &after_error[message_idx + 9..];
    let after_colon = after_message.trim_start();
    let after_colon = after_colon.strip_prefix(':')?;
    let after_colon = after_colon.trim_start();
    if !after_colon.starts_with('"') {
        return None;
    }
    parse_json_string(after_colon)
}

/// Parse a JSON string starting at `"` and return the unescaped content.
fn parse_json_string(s: &str) -> Option<String> {
    if !s.starts_with('"') {
        return None;
    }
    let mut result = String::new();
    let mut chars = s[1..].chars();
    loop {
        match chars.next()? {
            '"' => return Some(result),
            '\\' => {
                match chars.next()? {
                    '"' => result.push('"'),
                    '\\' => result.push('\\'),
                    'n' => result.push('\n'),
                    'r' => result.push('\r'),
                    't' => result.push('\t'),
                    '/' => result.push('/'),
                    'u' => {
                        let hex: String = (0..4).filter_map(|_| chars.next()).collect();
                        if let Ok(code) = u32::from_str_radix(&hex, 16) {
                            if let Some(c) = char::from_u32(code) {
                                result.push(c);
                            }
                        }
                    }
                    other => {
                        result.push('\\');
                        result.push(other);
                    }
                }
            }
            c => result.push(c),
        }
    }
}

/// Run the generated .flow file.
fn run_generated_file(path: &str) -> Result<(), CliError> {
    if !std::path::Path::new(path).exists() {
        return Err(CliError::Io(format!("no generated file: {}", path)));
    }

    let source = std::fs::read_to_string(path)
        .map_err(|e| CliError::Io(format!("{}: {}", path, e)))?;

    let program = octoflow_parser::parse(&source)
        .map_err(|e| CliError::Parse(format!("{}", e)))?;

    let base_dir = std::path::Path::new(path)
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| ".".to_string());

    let overrides = Overrides {
        allow_read: crate::PermScope::AllowAll,
        allow_write: crate::PermScope::AllowAll,
        allow_net: crate::PermScope::AllowAll,
        ..Default::default()
    };

    // Pre-flight validation
    let report = crate::analysis::preflight::validate(&program, &base_dir);
    if !report.passed {
        eprintln!("PRE-FLIGHT: {}", report);
        return Err(CliError::Compile("pre-flight check failed".into()));
    }

    streaming::print_status(&format!("Running {}...\n", path), streaming::DIM);
    let start = std::time::Instant::now();
    compiler::execute(&program, &base_dir, &overrides)?;
    let elapsed = start.elapsed();

    if elapsed.as_secs_f64() >= 1.0 {
        streaming::print_success(&format!("ok {:.1}s", elapsed.as_secs_f64()));
    } else {
        streaming::print_success(&format!("ok {}ms", elapsed.as_millis()));
    }

    Ok(())
}

/// Show the contents of the generated file.
fn show_generated_file(path: &str) {
    match std::fs::read_to_string(path) {
        Ok(content) => {
            eprintln!("{}--- {} ---{}", streaming::DIM, path, streaming::RESET);
            for (i, line) in content.lines().enumerate() {
                eprintln!("{}{:4}{} | {}", streaming::DIM, i + 1, streaming::RESET, line);
            }
            eprintln!("{}---{}", streaming::DIM, streaming::RESET);
        }
        Err(_) => {
            eprintln!("No generated file yet.");
        }
    }
}

/// Read multiline input: lines ending with `\` continue on the next line.
fn read_multiline_input(reader: &mut impl BufRead) -> Option<String> {
    let mut accumulated = String::new();
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => {
                if accumulated.is_empty() {
                    return None; // EOF with no input
                }
                return Some(accumulated);
            }
            Ok(_) => {}
            Err(_) => return None,
        }
        let trimmed = line.trim_end_matches('\n').trim_end_matches('\r');
        if trimmed.ends_with('\\') {
            accumulated.push_str(&trimmed[..trimmed.len() - 1]);
            accumulated.push(' ');
            eprint!("{}... {}", streaming::DIM, streaming::RESET);
            io::stderr().flush().ok();
        } else {
            accumulated.push_str(trimmed);
            return Some(accumulated);
        }
    }
}

/// Build a fix prompt that includes the failing code and error message.
fn build_fix_prompt(user_prompt: &str, code: Option<&str>, error: &str) -> String {
    let mut prompt = user_prompt.to_string();
    if let Some(code) = code {
        prompt.push_str("\n\nThe previous code was:\n```flow\n");
        prompt.push_str(code);
        prompt.push_str("\n```");
    }
    prompt.push_str(&format!(
        "\n\nIt produced this error:\n{}\n\nFix the error and regenerate the complete code.",
        error
    ));
    prompt
}

/// Truncate content for context window: keep up to max_lines lines, max_chars total.
fn truncate_for_context(content: &str, max_lines: usize, max_chars: usize) -> String {
    let mut result = String::new();
    let mut lines_taken = 0;
    for line in content.lines() {
        if lines_taken >= max_lines || result.len() + line.len() > max_chars {
            result.push_str("...");
            break;
        }
        if lines_taken > 0 {
            result.push('\n');
        }
        result.push_str(line);
        lines_taken += 1;
    }
    result
}

/// Print a simple line-by-line diff between two code versions.
fn print_diff(old: &str, new: &str) {
    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();

    // Simple LCS-based diff
    let lcs = compute_lcs(&old_lines, &new_lines);
    let mut oi = 0;
    let mut ni = 0;
    let mut li = 0;

    while oi < old_lines.len() || ni < new_lines.len() {
        if li < lcs.len() && oi < old_lines.len() && ni < new_lines.len()
            && old_lines[oi] == lcs[li] && new_lines[ni] == lcs[li]
        {
            // Unchanged
            eprintln!("{}  {}{}", streaming::DIM, old_lines[oi], streaming::RESET);
            oi += 1;
            ni += 1;
            li += 1;
        } else if li < lcs.len() && oi < old_lines.len() && old_lines[oi] != lcs[li] {
            // Removed
            eprintln!("{}- {}{}", streaming::RED, old_lines[oi], streaming::RESET);
            oi += 1;
        } else if oi >= old_lines.len() || (li < lcs.len() && ni < new_lines.len() && new_lines[ni] != lcs[li]) {
            // Added
            eprintln!("{}+ {}{}", streaming::GREEN, new_lines[ni], streaming::RESET);
            ni += 1;
        } else {
            // Remaining old lines are removed
            eprintln!("{}- {}{}", streaming::RED, old_lines[oi], streaming::RESET);
            oi += 1;
        }
    }
}

/// Compute the Longest Common Subsequence of two line slices.
fn compute_lcs<'a>(a: &[&'a str], b: &[&'a str]) -> Vec<&'a str> {
    let m = a.len();
    let n = b.len();
    // DP table
    let mut dp = vec![vec![0u32; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            if a[i - 1] == b[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }
    // Backtrack
    let mut result = Vec::new();
    let mut i = m;
    let mut j = n;
    while i > 0 && j > 0 {
        if a[i - 1] == b[j - 1] {
            result.push(a[i - 1]);
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] > dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    result.reverse();
    result
}

/// Print chat help.
fn print_chat_help() {
    eprintln!("{}OctoFlow Chat Commands:{}", streaming::BOLD, streaming::RESET);
    eprintln!();
    eprintln!("  Type a description to generate .flow code.");
    eprintln!("  Type a follow-up to modify the generated code.");
    eprintln!("  End a line with \\ to continue on the next line.");
    eprintln!();
    eprintln!("  :run, :r           Re-run the generated file");
    eprintln!("  :show, :s          Show the generated code");
    eprintln!("  :undo, :u          Restore previous version");
    eprintln!("  :diff, :d          Diff last two versions");
    eprintln!("  :edit, :e          Open in $EDITOR, then auto-run");
    eprintln!("  :clear, :c         Clear file + history");
    eprintln!("  :clear history     Clear conversation only");
    eprintln!("  :clear file        Clear file only");
    eprintln!("  :history           Show conversation history");
    eprintln!("  :help, :h          Show this help");
    eprintln!("  :quit, :q          Exit chat");
    eprintln!();
    eprintln!("{}General Options:{}", streaming::BOLD, streaming::RESET);
    eprintln!("  --output, -o <path>    Output .flow file (default: main.flow)");
    eprintln!("  --max-tokens <n>       Max generation tokens (default: 512)");
    eprintln!("  --web-tools            Enable web search/read during generation");
    eprintln!("  --allow-net            Enable network + auto-enable web tools");
    eprintln!("  --no-memory            Disable all memory loading/saving (privacy)");
    eprintln!();
    eprintln!("{}Local Mode (default):{}", streaming::BOLD, streaming::RESET);
    eprintln!("  --model, -m <path>     Path to GGUF model (default: {})", DEFAULT_MODEL_PATH);
    eprintln!();
    eprintln!("{}API Mode:{}", streaming::BOLD, streaming::RESET);
    eprintln!("  --provider api         Use an OpenAI-compatible API");
    eprintln!("  --api-key <key>        API key (or set OCTOFLOW_API_KEY / OPENAI_API_KEY)");
    eprintln!("  --api-url <url>        Endpoint (default: {})", DEFAULT_API_URL);
    eprintln!("  --api-model <name>     Model name (default: {})", DEFAULT_API_MODEL);
    eprintln!();
    eprintln!("{}Examples:{}", streaming::DIM, streaming::RESET);
    eprintln!("  octoflow chat                                      # local GGUF model");
    eprintln!("  octoflow chat --provider api --api-key sk-...       # OpenAI");
    eprintln!("  octoflow chat --provider api --api-url http://localhost:11434/v1/chat/completions  # Ollama");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_chat_args_defaults() {
        let args: Vec<String> = vec![];
        let opts = parse_chat_args(&args).unwrap();
        assert_eq!(opts.model_path, DEFAULT_MODEL_PATH);
        assert_eq!(opts.output_file, "main.flow");
        assert_eq!(opts.max_tokens, 512);
        assert_eq!(opts.provider, Provider::Local);
        assert!(opts.api_key.is_none());
        assert!(opts.api_url.is_none());
        assert!(opts.api_model.is_none());
    }

    #[test]
    fn test_parse_chat_args_model() {
        let args: Vec<String> = vec!["--model".into(), "my-model.gguf".into()];
        let opts = parse_chat_args(&args).unwrap();
        assert_eq!(opts.model_path, "my-model.gguf");
    }

    #[test]
    fn test_parse_chat_args_short_model() {
        let args: Vec<String> = vec!["-m".into(), "my-model.gguf".into()];
        let opts = parse_chat_args(&args).unwrap();
        assert_eq!(opts.model_path, "my-model.gguf");
    }

    #[test]
    fn test_parse_chat_args_output() {
        let args: Vec<String> = vec!["--output".into(), "app.flow".into()];
        let opts = parse_chat_args(&args).unwrap();
        assert_eq!(opts.output_file, "app.flow");
    }

    #[test]
    fn test_parse_chat_args_max_tokens() {
        let args: Vec<String> = vec!["--max-tokens".into(), "500".into()];
        let opts = parse_chat_args(&args).unwrap();
        assert_eq!(opts.max_tokens, 500);
    }

    #[test]
    fn test_parse_chat_args_unknown() {
        let args: Vec<String> = vec!["--unknown".into()];
        assert!(parse_chat_args(&args).is_err());
    }

    #[test]
    fn test_parse_chat_args_all() {
        let args: Vec<String> = vec![
            "-m".into(), "model.gguf".into(),
            "-o".into(), "output.flow".into(),
            "--max-tokens".into(), "300".into(),
        ];
        let opts = parse_chat_args(&args).unwrap();
        assert_eq!(opts.model_path, "model.gguf");
        assert_eq!(opts.output_file, "output.flow");
        assert_eq!(opts.max_tokens, 300);
    }

    // --- API provider tests ---

    #[test]
    fn test_parse_chat_args_provider_api() {
        let args: Vec<String> = vec![
            "--provider".into(), "api".into(),
            "--api-key".into(), "sk-test123".into(),
        ];
        let opts = parse_chat_args(&args).unwrap();
        assert_eq!(opts.provider, Provider::Api);
        assert_eq!(opts.api_key.as_deref(), Some("sk-test123"));
    }

    #[test]
    fn test_parse_chat_args_provider_local_explicit() {
        let args: Vec<String> = vec!["--provider".into(), "local".into()];
        let opts = parse_chat_args(&args).unwrap();
        assert_eq!(opts.provider, Provider::Local);
    }

    #[test]
    fn test_parse_chat_args_provider_invalid() {
        let args: Vec<String> = vec!["--provider".into(), "magic".into()];
        assert!(parse_chat_args(&args).is_err());
    }

    #[test]
    fn test_parse_chat_args_api_url() {
        let args: Vec<String> = vec![
            "--provider".into(), "api".into(),
            "--api-key".into(), "key".into(),
            "--api-url".into(), "http://localhost:11434/v1/chat/completions".into(),
        ];
        let opts = parse_chat_args(&args).unwrap();
        assert_eq!(opts.api_url.as_deref(), Some("http://localhost:11434/v1/chat/completions"));
    }

    #[test]
    fn test_parse_chat_args_api_model() {
        let args: Vec<String> = vec![
            "--provider".into(), "api".into(),
            "--api-key".into(), "key".into(),
            "--api-model".into(), "claude-sonnet-4-20250514".into(),
        ];
        let opts = parse_chat_args(&args).unwrap();
        assert_eq!(opts.api_model.as_deref(), Some("claude-sonnet-4-20250514"));
    }

    #[test]
    fn test_parse_chat_args_api_all_options() {
        let args: Vec<String> = vec![
            "--provider".into(), "api".into(),
            "--api-key".into(), "sk-abc".into(),
            "--api-url".into(), "https://custom.api.com/v1/chat/completions".into(),
            "--api-model".into(), "gpt-4o".into(),
            "-o".into(), "app.flow".into(),
            "--max-tokens".into(), "500".into(),
        ];
        let opts = parse_chat_args(&args).unwrap();
        assert_eq!(opts.provider, Provider::Api);
        assert_eq!(opts.api_key.as_deref(), Some("sk-abc"));
        assert_eq!(opts.api_url.as_deref(), Some("https://custom.api.com/v1/chat/completions"));
        assert_eq!(opts.api_model.as_deref(), Some("gpt-4o"));
        assert_eq!(opts.output_file, "app.flow");
        assert_eq!(opts.max_tokens, 500);
    }

    #[test]
    fn test_parse_chat_args_api_no_key_allowed_at_parse() {
        // parse_chat_args doesn't error on missing key — run_chat does
        let args: Vec<String> = vec!["--provider".into(), "api".into()];
        let opts = parse_chat_args(&args).unwrap();
        assert_eq!(opts.provider, Provider::Api);
        assert!(opts.api_key.is_none());
    }

    #[test]
    fn test_parse_chat_args_missing_provider_value() {
        let args: Vec<String> = vec!["--provider".into()];
        assert!(parse_chat_args(&args).is_err());
    }

    #[test]
    fn test_parse_chat_args_missing_api_key_value() {
        let args: Vec<String> = vec!["--api-key".into()];
        assert!(parse_chat_args(&args).is_err());
    }

    #[test]
    fn test_parse_chat_args_missing_api_url_value() {
        let args: Vec<String> = vec!["--api-url".into()];
        assert!(parse_chat_args(&args).is_err());
    }

    #[test]
    fn test_parse_chat_args_missing_api_model_value() {
        let args: Vec<String> = vec!["--api-model".into()];
        assert!(parse_chat_args(&args).is_err());
    }

    // --- JSON utility tests ---

    #[test]
    fn test_json_escape_simple() {
        assert_eq!(json_escape("hello"), "\"hello\"");
    }

    #[test]
    fn test_json_escape_special_chars() {
        assert_eq!(json_escape("a\"b\\c\nd"), "\"a\\\"b\\\\c\\nd\"");
    }

    #[test]
    fn test_json_escape_tab() {
        assert_eq!(json_escape("a\tb"), "\"a\\tb\"");
    }

    #[test]
    fn test_extract_api_content_valid() {
        let response = r#"{"choices":[{"message":{"role":"assistant","content":"```flow\nlet x = 42\n```"}}]}"#;
        let content = extract_api_content(response).unwrap();
        assert_eq!(content, "```flow\nlet x = 42\n```");
    }

    #[test]
    fn test_extract_api_content_with_escapes() {
        let response = r#"{"choices":[{"message":{"content":"line1\nline2"}}]}"#;
        let content = extract_api_content(response).unwrap();
        assert_eq!(content, "line1\nline2");
    }

    #[test]
    fn test_extract_api_content_missing() {
        let response = r#"{"error":{"message":"bad request"}}"#;
        assert!(extract_api_content(response).is_none());
    }

    #[test]
    fn test_extract_api_error_valid() {
        let response = r#"{"error":{"message":"Invalid API key","type":"auth_error"}}"#;
        let err = extract_api_error(response).unwrap();
        assert_eq!(err, "Invalid API key");
    }

    #[test]
    fn test_extract_api_error_missing() {
        let response = r#"{"choices":[{"message":{"content":"ok"}}]}"#;
        assert!(extract_api_error(response).is_none());
    }

    #[test]
    fn test_parse_json_string_simple() {
        assert_eq!(parse_json_string("\"hello\""), Some("hello".to_string()));
    }

    #[test]
    fn test_parse_json_string_escapes() {
        assert_eq!(parse_json_string("\"a\\nb\""), Some("a\nb".to_string()));
        assert_eq!(parse_json_string("\"a\\\"b\""), Some("a\"b".to_string()));
    }

    #[test]
    fn test_parse_json_string_unicode() {
        assert_eq!(parse_json_string("\"\\u0041\""), Some("A".to_string()));
    }

    #[test]
    fn test_parse_json_string_not_string() {
        assert_eq!(parse_json_string("42"), None);
    }

    // --- System prompt tests ---

    #[test]
    fn test_system_prompt_includes_contract() {
        let prompt = build_system_prompt();
        assert!(prompt.contains("GPU-native"));
        assert!(prompt.contains("## Syntax"));
        assert!(prompt.contains("## Built-in Functions"));
    }

    #[test]
    fn test_system_prompt_includes_patterns() {
        let prompt = build_system_prompt();
        assert!(prompt.contains("## Patterns"));
        assert!(prompt.contains("gpu_fill"));
        assert!(prompt.contains("http_get"));
    }

    #[test]
    fn test_system_prompt_has_preamble() {
        let prompt = build_system_prompt();
        assert!(prompt.starts_with("You are OctoFlow Chat"));
    }

    #[test]
    fn test_system_prompt_includes_critical_rules() {
        let prompt = build_system_prompt();
        assert!(prompt.contains("## Critical Rules"));
        assert!(prompt.contains("print()"));
    }

    #[test]
    fn test_system_prompt_think_mode_off() {
        let prompt = build_system_prompt_with_think(false);
        assert!(prompt.contains("/no_think"), "non-thinking mode should include /no_think directive");
    }

    #[test]
    fn test_system_prompt_think_mode_on() {
        let prompt = build_system_prompt_with_think(true);
        assert!(!prompt.contains("/no_think"), "thinking mode should NOT include /no_think directive");
    }

    #[test]
    fn test_system_prompt_with_tools_think_mode() {
        let prompt = build_system_prompt_with_tools(false);
        assert!(prompt.contains("## Tools"));
        assert!(prompt.contains("/no_think"));
        let prompt_think = build_system_prompt_with_tools(true);
        assert!(prompt_think.contains("## Tools"));
        assert!(!prompt_think.contains("/no_think"));
    }

    #[test]
    fn test_parse_chat_args_think_flag() {
        let args: Vec<String> = vec!["--think".into()];
        let opts = parse_chat_args(&args).unwrap();
        assert!(opts.think);
    }

    #[test]
    fn test_parse_chat_args_default_no_think() {
        let args: Vec<String> = vec![];
        let opts = parse_chat_args(&args).unwrap();
        assert!(!opts.think);
    }

    #[test]
    fn test_parse_chat_args_no_memory() {
        let args: Vec<String> = vec!["--no-memory".into()];
        let opts = parse_chat_args(&args).unwrap();
        assert!(opts.no_memory);
    }

    #[test]
    fn test_parse_chat_args_default_memory_on() {
        let args: Vec<String> = vec![];
        let opts = parse_chat_args(&args).unwrap();
        assert!(!opts.no_memory);
    }

    // --- Multiline input tests ---

    #[test]
    fn test_read_multiline_single_line() {
        let input = b"hello world\n" as &[u8];
        let mut cursor = io::Cursor::new(input);
        let result = read_multiline_input(&mut cursor);
        assert_eq!(result, Some("hello world".to_string()));
    }

    #[test]
    fn test_read_multiline_continuation() {
        let input = b"hello \\\nworld\n" as &[u8];
        let mut cursor = io::Cursor::new(input);
        let result = read_multiline_input(&mut cursor);
        assert_eq!(result, Some("hello  world".to_string()));
    }

    #[test]
    fn test_read_multiline_eof() {
        let input = b"" as &[u8];
        let mut cursor = io::Cursor::new(input);
        let result = read_multiline_input(&mut cursor);
        assert_eq!(result, None);
    }

    #[test]
    fn test_read_multiline_triple_continuation() {
        let input = b"a\\\nb\\\nc\n" as &[u8];
        let mut cursor = io::Cursor::new(input);
        let result = read_multiline_input(&mut cursor);
        assert_eq!(result, Some("a b c".to_string()));
    }

    // --- build_fix_prompt tests ---

    #[test]
    fn test_build_fix_prompt_with_code() {
        let prompt = build_fix_prompt("make a counter", Some("let x = 0"), "undefined variable");
        assert!(prompt.contains("make a counter"));
        assert!(prompt.contains("```flow\nlet x = 0\n```"));
        assert!(prompt.contains("undefined variable"));
    }

    #[test]
    fn test_build_fix_prompt_without_code() {
        let prompt = build_fix_prompt("make a counter", None, "empty output");
        assert!(prompt.contains("make a counter"));
        assert!(!prompt.contains("```flow"));
        assert!(prompt.contains("empty output"));
    }

    // --- truncate_for_context tests ---

    #[test]
    fn test_truncate_short() {
        let result = truncate_for_context("hello", 10, 500);
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_truncate_by_lines() {
        let input = "line1\nline2\nline3\nline4";
        let result = truncate_for_context(input, 2, 500);
        assert_eq!(result, "line1\nline2...");
    }

    #[test]
    fn test_truncate_by_chars() {
        let input = "short\nthis is a much longer line that exceeds the limit";
        let result = truncate_for_context(input, 10, 15);
        assert_eq!(result, "short...");
    }

    // --- LCS / diff tests ---

    #[test]
    fn test_compute_lcs_identical() {
        let a = vec!["a", "b", "c"];
        let b = vec!["a", "b", "c"];
        assert_eq!(compute_lcs(&a, &b), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_compute_lcs_one_changed() {
        let a = vec!["a", "b", "c"];
        let b = vec!["a", "x", "c"];
        assert_eq!(compute_lcs(&a, &b), vec!["a", "c"]);
    }

    #[test]
    fn test_compute_lcs_empty() {
        let a: Vec<&str> = vec![];
        let b = vec!["a"];
        assert_eq!(compute_lcs(&a, &b), Vec::<&str>::new());
    }

    // --- ReAct tool-use tests ---

    #[test]
    fn test_detect_tool_call_search() {
        let output = "SEARCH: rust async tutorial";
        let result = detect_tool_call(output);
        assert_eq!(result, Some(("web_search", "rust async tutorial".to_string())));
    }

    #[test]
    fn test_detect_tool_call_read() {
        let output = "READ: https://docs.rs/tokio";
        let result = detect_tool_call(output);
        assert_eq!(result, Some(("web_read", "https://docs.rs/tokio".to_string())));
    }

    #[test]
    fn test_detect_tool_call_code_block() {
        let output = "```flow\nlet x = 42\n```";
        let result = detect_tool_call(output);
        assert!(result.is_none());
    }

    #[test]
    fn test_detect_tool_call_mixed_text() {
        // If there's a SEARCH line among other text, detect it
        let output = "I need to find an API.\nSEARCH: free weather API\nLet me check...";
        let result = detect_tool_call(output);
        assert_eq!(result, Some(("web_search", "free weather API".to_string())));
    }

    #[test]
    fn test_detect_tool_call_empty_query() {
        let output = "SEARCH: ";
        let result = detect_tool_call(output);
        assert!(result.is_none());
    }

    #[test]
    fn test_format_search_results_empty() {
        let results: Vec<crate::Value> = vec![];
        assert_eq!(format_search_results(&results), "");
    }

    #[test]
    fn test_format_search_results_single() {
        use std::collections::HashMap;
        let mut m = HashMap::new();
        m.insert("title".to_string(), crate::Value::Str("Example".to_string()));
        m.insert("url".to_string(), crate::Value::Str("https://example.com".to_string()));
        m.insert("snippet".to_string(), crate::Value::Str("An example site.".to_string()));
        let results = vec![crate::Value::Map(m)];
        let formatted = format_search_results(&results);
        assert!(formatted.contains("Example"));
        assert!(formatted.contains("https://example.com"));
        assert!(formatted.contains("An example site."));
    }

    #[test]
    fn test_truncate_text_short() {
        assert_eq!(truncate_text("hello", 100), "hello");
    }

    #[test]
    fn test_truncate_text_long() {
        let text = "The quick brown fox jumps over the lazy dog and keeps running";
        let result = truncate_text(text, 30);
        assert!(result.len() <= 34); // 30 + "..."
        assert!(result.ends_with("..."));
    }

    #[test]
    fn test_parse_chat_args_web_tools() {
        let args: Vec<String> = vec!["--web-tools".into()];
        let opts = parse_chat_args(&args).unwrap();
        assert!(opts.web_tools);
    }

    #[test]
    fn test_parse_chat_args_allow_net_enables_web_tools() {
        let args: Vec<String> = vec!["--allow-net".into()];
        let opts = parse_chat_args(&args).unwrap();
        assert!(opts.web_tools);
    }

    #[test]
    fn test_system_prompt_with_tools_has_tool_section() {
        let prompt = build_system_prompt_with_tools(false);
        assert!(prompt.contains("## Tools"));
        assert!(prompt.contains("SEARCH:"));
        assert!(prompt.contains("READ:"));
    }

    // --- Context engine tests ---

    #[test]
    fn test_context_prompt_contains_l0() {
        let prompt = build_context_prompt("hello world", false, false);
        assert!(prompt.contains("OctoFlow"));
        assert!(prompt.contains("let x ="));
        assert!(prompt.contains("/no_think"));
    }

    #[test]
    fn test_context_prompt_no_contract() {
        // Context prompt should NOT include the full CONTRACT
        let prompt = build_context_prompt("hello", false, false);
        // L0 is ~439 tokens, CONTRACT is ~2500 tokens
        // Context prompt should be much shorter than CONTRACT-based prompt
        let contract_prompt = build_system_prompt_with_think(false);
        assert!(prompt.len() < contract_prompt.len());
    }

    #[test]
    fn test_context_prompt_web_skills() {
        let prompt = build_context_prompt("build an http server with api endpoints", false, false);
        // Should detect web domain and include web-related L1 content
        assert!(prompt.contains("http") || prompt.contains("HTTP"));
        assert!(prompt.contains("Relevant Modules"));
    }

    #[test]
    fn test_context_prompt_with_tools() {
        let prompt = build_context_prompt("fetch data from api", false, true);
        assert!(prompt.contains("## Tools"));
        assert!(prompt.contains("SEARCH:"));
    }

    #[test]
    fn test_context_prompt_think_mode() {
        let prompt_no_think = build_context_prompt("hello", false, false);
        assert!(prompt_no_think.contains("/no_think"));
        let prompt_think = build_context_prompt("hello", true, false);
        assert!(!prompt_think.contains("/no_think"));
    }

    #[test]
    fn test_context_prompt_generic_message() {
        // Generic message without domain keywords → L0 only, no skill context
        let prompt = build_context_prompt("print hello world", false, false);
        assert!(prompt.contains("OctoFlow"));
        assert!(!prompt.contains("Relevant Modules"));
    }

    // --- User config integration tests ---

    #[test]
    fn test_context_prompt_with_user_config() {
        let cfg = config::UserConfig {
            global_prefs: Some("Always use GPU.".into()),
            global_memory: None,
            project_instructions: Some("Stock analysis project.".into()),
            project_memory: None,
        };
        let prompt = build_context_prompt_full("hello", false, false, None, Some(&cfg));
        assert!(prompt.contains("## User Preferences"));
        assert!(prompt.contains("GPU"));
        assert!(prompt.contains("## Project Instructions"));
        assert!(prompt.contains("Stock analysis"));
    }

    #[test]
    fn test_context_prompt_with_memory_and_config() {
        let mut mem = memory::PersistentMemory::new();
        mem.record_module("data/csv");
        let cfg = config::UserConfig {
            global_prefs: Some("Use snake_case.".into()),
            global_memory: None,
            project_instructions: None,
            project_memory: None,
        };
        let prompt = build_context_prompt_full("read csv", false, false, Some(&mem), Some(&cfg));
        assert!(prompt.contains("snake_case"));
        assert!(prompt.contains("data/csv"));
    }

    #[test]
    fn test_context_prompt_no_config() {
        // With no config, should work the same as build_context_prompt
        let prompt_no_cfg = build_context_prompt_full("hello", false, false, None, None);
        let prompt_default = build_context_prompt("hello", false, false);
        assert_eq!(prompt_no_cfg, prompt_default);
    }

    #[test]
    fn test_builtin_gbnf_embedded() {
        // F-09: GBNF grammar is embedded in the binary
        assert!(!BUILTIN_GBNF.is_empty(), "embedded GBNF should not be empty");
        assert!(BUILTIN_GBNF.contains("root"), "GBNF should contain 'root' rule");
        assert!(BUILTIN_GBNF.len() > 100, "GBNF should be substantial");
    }
}
