//! OctoView — GPU-native browser that transpiles HTML/CSS/JS to OctoView Documents.
//!
//! Every page becomes typed, flat, queryable data.
//! Browsing the web and scraping it are the same operation.

mod action;
mod cache;
mod extract;
mod fetch;
mod js_engine;
mod meta;
mod ovd;
mod query;
mod selector;
mod session;
mod shell;
mod tui;
mod watch;

use clap::{Parser, Subcommand};
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "octoview",
    version,
    about = "OctoView — the web, rewoven. GPU-native browser & web transpiler."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Transpile a URL or HTML file to OctoView Document and display it
    View {
        /// URL or local file path
        source: String,

        /// Don't use cache (always re-fetch)
        #[arg(long)]
        no_cache: bool,

        /// Cache max age in seconds (default: 3600)
        #[arg(long, default_value = "3600")]
        cache_age: u64,

        /// Force headless Chrome JS rendering (for SPAs, Cloudflare)
        #[arg(long)]
        js: bool,

        /// Auto-detect if JS rendering needed, fall back automatically
        #[arg(long)]
        auto_js: bool,

        /// JS rendering wait timeout in seconds (default: 15)
        #[arg(long, default_value = "15")]
        js_wait: u64,
    },

    /// Query a URL's content with SQL-like syntax
    Query {
        /// URL or local file path
        source: String,

        /// Query string (e.g., "SELECT text, href FROM nodes WHERE type = 'link'")
        query: String,

        /// Maximum column width in output
        #[arg(long, default_value = "60")]
        max_width: usize,

        /// Output format: table (default), csv, json, jsonl
        #[arg(long, default_value = "table")]
        format: String,

        /// Write output to file instead of stdout
        #[arg(short, long)]
        output: Option<String>,

        /// Don't use cache
        #[arg(long)]
        no_cache: bool,

        /// Cache max age in seconds
        #[arg(long, default_value = "3600")]
        cache_age: u64,

        /// Force headless Chrome JS rendering
        #[arg(long)]
        js: bool,

        /// Auto-detect if JS rendering needed
        #[arg(long)]
        auto_js: bool,

        /// JS rendering wait timeout in seconds
        #[arg(long, default_value = "15")]
        js_wait: u64,
    },

    /// Interactive query shell — load pages, run queries, switch contexts
    Shell {
        /// Optional URL to load on startup
        source: Option<String>,

        /// Force headless Chrome JS rendering
        #[arg(long)]
        js: bool,

        /// JS rendering wait timeout in seconds
        #[arg(long, default_value = "15")]
        js_wait: u64,
    },

    /// Interactive TUI browser — browse pages in the terminal
    Browse {
        /// URL or local file path
        source: String,

        /// Don't use cache
        #[arg(long)]
        no_cache: bool,

        /// Cache max age in seconds
        #[arg(long, default_value = "3600")]
        cache_age: u64,

        /// Force headless Chrome JS rendering
        #[arg(long)]
        js: bool,

        /// Auto-detect if JS rendering needed
        #[arg(long)]
        auto_js: bool,

        /// JS rendering wait timeout in seconds
        #[arg(long, default_value = "15")]
        js_wait: u64,
    },

    /// Show document statistics for a URL
    Stats {
        /// URL or local file path
        source: String,

        /// Don't use cache
        #[arg(long)]
        no_cache: bool,

        /// Force headless Chrome JS rendering
        #[arg(long)]
        js: bool,

        /// JS rendering wait timeout in seconds
        #[arg(long, default_value = "15")]
        js_wait: u64,
    },

    /// Save transpiled page as .ovd file
    Save {
        /// URL or local file path
        source: String,

        /// Output .ovd file path
        #[arg(short, long)]
        output: String,

        /// Force headless Chrome JS rendering
        #[arg(long)]
        js: bool,

        /// JS rendering wait timeout in seconds
        #[arg(long, default_value = "15")]
        js_wait: u64,
    },

    /// Load and display a cached .ovd file
    Load {
        /// Path to .ovd file
        path: String,

        /// Optional query
        #[arg(short, long)]
        query: Option<String>,
    },

    /// Watch a URL for changes at intervals
    Watch {
        /// URL to monitor
        source: String,

        /// Check interval in seconds (default: 60)
        #[arg(short, long, default_value = "60")]
        interval: u64,

        /// Maximum number of checks (default: unlimited)
        #[arg(short = 'n', long)]
        max_checks: Option<usize>,

        /// Only report when changes are detected
        #[arg(long)]
        only_changes: bool,

        /// Append reports to file
        #[arg(short, long)]
        output: Option<String>,

        /// Output in JSONL format (for pipelines)
        #[arg(long)]
        jsonl: bool,

        /// Force headless Chrome JS rendering
        #[arg(long)]
        js: bool,

        /// JS rendering wait timeout in seconds
        #[arg(long, default_value = "15")]
        js_wait: u64,
    },

    /// Compare two pages or a page against its cache
    Diff {
        /// First URL or .ovd file
        source_a: String,

        /// Second URL or .ovd file (if omitted, compares against cache)
        source_b: Option<String>,

        /// Force headless Chrome JS rendering
        #[arg(long)]
        js: bool,

        /// JS rendering wait timeout in seconds
        #[arg(long, default_value = "15")]
        js_wait: u64,
    },

    /// Manage the .ovd cache
    Cache {
        #[command(subcommand)]
        action: CacheAction,
    },

    /// CSS selector query (e.g., "a.nav-link", "div.card > h2", ".price:bold")
    Select {
        /// URL or local file path
        source: String,

        /// CSS selector (e.g., "a.nav-link", "div.card > h2", ":text(price)")
        selector: String,

        /// Output format: table (default), csv, json, jsonl, feed
        #[arg(long, default_value = "table")]
        format: String,

        /// Maximum column width
        #[arg(long, default_value = "60")]
        max_width: usize,

        /// Write output to file
        #[arg(short, long)]
        output: Option<String>,

        /// Don't use cache
        #[arg(long)]
        no_cache: bool,

        /// Force headless Chrome JS rendering
        #[arg(long)]
        js: bool,

        /// JS rendering wait timeout in seconds
        #[arg(long, default_value = "15")]
        js_wait: u64,
    },

    /// LLM-optimized page feed — structured semantic summary for AI agents
    Feed {
        /// URL or local file path
        source: String,

        /// Output format: feed (default), jsonl
        #[arg(long, default_value = "feed")]
        format: String,

        /// Optional CSS selector to filter content
        #[arg(short, long)]
        selector: Option<String>,

        /// Write output to file
        #[arg(short, long)]
        output: Option<String>,

        /// Don't use cache
        #[arg(long)]
        no_cache: bool,

        /// Force headless Chrome JS rendering
        #[arg(long)]
        js: bool,

        /// Intercept API calls (fetch/XHR) — captures raw JSON data
        #[arg(long)]
        intercept: bool,

        /// JS rendering wait timeout in seconds
        #[arg(long, default_value = "15")]
        js_wait: u64,
    },

    /// Intercept API calls (fetch/XHR) from a JS-rendered page
    Intercept {
        /// URL to load and intercept API calls from
        source: String,

        /// Output format: table (default), jsonl
        #[arg(long, default_value = "table")]
        format: String,

        /// Write output to file
        #[arg(short, long)]
        output: Option<String>,

        /// JS rendering wait timeout in seconds
        #[arg(long, default_value = "15")]
        js_wait: u64,
    },

    /// Login interactively — opens visible Chrome for manual login, saves session
    Login {
        /// URL to open for login (e.g., https://accounts.google.com)
        url: String,

        /// Profile name to save session under (default: "default")
        #[arg(long, default_value = "default")]
        profile: String,

        /// Browser timeout in seconds
        #[arg(long, default_value = "300")]
        timeout: u64,
    },

    /// Manage browser sessions and profiles
    #[command(subcommand)]
    Session(SessionAction),

    /// Take a screenshot of a URL
    Screenshot {
        /// URL or local file path
        source: String,

        /// Output PNG file path
        #[arg(short, long, default_value = "screenshot.png")]
        output: String,

        /// Profile name for authenticated browsing
        #[arg(long, default_value = "default")]
        profile: String,

        /// JS rendering wait timeout in seconds
        #[arg(long, default_value = "15")]
        js_wait: u64,
    },

    /// Run an action script (.ova) — multi-step web automation
    Run {
        /// Path to .ova action script file
        script: String,

        /// Profile name for browser session (default: "default")
        #[arg(long, default_value = "default")]
        profile: String,

        /// Open headed (visible) browser
        #[arg(long)]
        headed: bool,

        /// Browser timeout in seconds
        #[arg(long, default_value = "120")]
        timeout: u64,
    },

    /// Extract structured metadata: JSON-LD, Open Graph, Twitter Cards, meta tags
    Meta {
        /// URL or local file path
        source: String,

        /// Output format: text (default), jsonl
        #[arg(long, default_value = "text")]
        format: String,

        /// Write output to file
        #[arg(short, long)]
        output: Option<String>,

        /// Don't use cache
        #[arg(long)]
        no_cache: bool,

        /// Force headless Chrome JS rendering
        #[arg(long)]
        js: bool,

        /// JS rendering wait timeout in seconds
        #[arg(long, default_value = "15")]
        js_wait: u64,
    },

    /// Crawl: follow links matching a selector, extract from each child page
    Crawl {
        /// URL to start crawling from
        source: String,

        /// CSS selector for links to follow (e.g., "a[href*=item]")
        #[arg(short, long)]
        links: String,

        /// CSS selector or SQL query to extract from each child page
        #[arg(short, long)]
        extract: Option<String>,

        /// Maximum number of links to follow (default: 10)
        #[arg(short = 'n', long, default_value = "10")]
        max_pages: usize,

        /// Output format: feed (default), jsonl, csv
        #[arg(long, default_value = "feed")]
        format: String,

        /// Write output to file
        #[arg(short, long)]
        output: Option<String>,

        /// Force headless Chrome JS rendering
        #[arg(long)]
        js: bool,

        /// JS rendering wait timeout in seconds
        #[arg(long, default_value = "15")]
        js_wait: u64,

        /// Delay between page fetches in ms (default: 500)
        #[arg(long, default_value = "500")]
        delay: u64,
    },

    /// Batch query multiple URLs
    Batch {
        /// File containing URLs (one per line)
        urls_file: String,

        /// Query to run on each page
        query: String,

        /// Maximum column width
        #[arg(long, default_value = "60")]
        max_width: usize,

        /// Output format: table (default), csv, json, jsonl
        #[arg(long, default_value = "table")]
        format: String,

        /// Write output to file instead of stdout
        #[arg(short, long)]
        output: Option<String>,

        /// Number of parallel fetches (default: 1 = sequential)
        #[arg(long, default_value = "1")]
        parallel: usize,
    },
}

#[derive(Subcommand)]
enum CacheAction {
    /// Show cache statistics
    Stats,
    /// Clear all cached pages
    Clear,
}

#[derive(Subcommand)]
enum SessionAction {
    /// List available profiles
    List,
    /// Show profile info
    Info {
        /// Profile name
        #[arg(default_value = "default")]
        profile: String,
    },
    /// Export cookies from a profile to JSON file
    Export {
        /// Profile name
        #[arg(long, default_value = "default")]
        profile: String,
        /// Output JSON file
        #[arg(short, long, default_value = "cookies.json")]
        output: String,
    },
    /// Import cookies from JSON file into a profile
    Import {
        /// Profile name
        #[arg(long, default_value = "default")]
        profile: String,
        /// Input JSON file
        #[arg(short, long)]
        input: String,
    },
    /// Browse authenticated pages using a saved profile
    Browse {
        /// URL to load
        url: String,
        /// Profile name
        #[arg(long, default_value = "default")]
        profile: String,
        /// Output format: view (default), feed, jsonl
        #[arg(long, default_value = "view")]
        format: String,
        /// Optional CSS selector
        #[arg(short, long)]
        selector: Option<String>,
        /// Write output to file
        #[arg(short, long)]
        output: Option<String>,
        /// JS rendering wait timeout in seconds
        #[arg(long, default_value = "15")]
        js_wait: u64,
    },
}

fn main() {
    // Run on a thread with larger stack (8MB) — the Commands enum is large
    // and debug builds on Windows overflow the default 1MB stack.
    let builder = std::thread::Builder::new().stack_size(8 * 1024 * 1024);
    let handler = builder.spawn(run).expect("Failed to spawn main thread");
    let result = handler.join();
    if result.is_err() {
        std::process::exit(1);
    }
}

fn run() {
    let cli = Cli::parse();

    match cli.command {
        Commands::View {
            source,
            no_cache,
            cache_age,
            js,
            auto_js,
            js_wait,
        } => {
            let js_mode = if js { JsMode::Force } else if auto_js { JsMode::Auto } else { JsMode::Off };
            cmd_view(&source, no_cache, cache_age, js_mode, js_wait);
        }

        Commands::Query {
            source,
            query: q,
            max_width,
            format,
            output,
            no_cache,
            cache_age,
            js,
            auto_js,
            js_wait,
        } => {
            let js_mode = if js { JsMode::Force } else if auto_js { JsMode::Auto } else { JsMode::Off };
            cmd_query(&source, &q, max_width, &format, output.as_deref(), no_cache, cache_age, js_mode, js_wait);
        }

        Commands::Shell {
            source,
            js,
            js_wait,
        } => {
            cmd_shell(source.as_deref(), js, js_wait);
        }

        Commands::Browse {
            source,
            no_cache,
            cache_age,
            js,
            auto_js,
            js_wait,
        } => {
            let js_mode = if js { JsMode::Force } else if auto_js { JsMode::Auto } else { JsMode::Off };
            cmd_browse(&source, no_cache, cache_age, js_mode, js_wait);
        }

        Commands::Stats { source, no_cache, js, js_wait } => {
            let js_mode = if js { JsMode::Force } else { JsMode::Off };
            cmd_stats(&source, no_cache, js_mode, js_wait);
        }

        Commands::Save { source, output, js, js_wait } => {
            let js_mode = if js { JsMode::Force } else { JsMode::Off };
            cmd_save(&source, &output, js_mode, js_wait);
        }

        Commands::Load { path, query: q } => cmd_load(&path, q.as_deref()),

        Commands::Watch {
            source,
            interval,
            max_checks,
            only_changes,
            output,
            jsonl,
            js,
            js_wait,
        } => {
            cmd_watch(&source, interval, max_checks, only_changes, output.as_deref(), jsonl, js, js_wait);
        }

        Commands::Diff {
            source_a,
            source_b,
            js,
            js_wait,
        } => {
            let js_mode = if js { JsMode::Force } else { JsMode::Off };
            cmd_diff(&source_a, source_b.as_deref(), js_mode, js_wait);
        }

        Commands::Select {
            source,
            selector: sel,
            format,
            max_width,
            output,
            no_cache,
            js,
            js_wait,
        } => {
            let js_mode = if js { JsMode::Force } else { JsMode::Off };
            cmd_select(&source, &sel, &format, max_width, output.as_deref(), no_cache, js_mode, js_wait);
        }

        Commands::Feed {
            source,
            format,
            selector: sel,
            output,
            no_cache,
            js,
            intercept,
            js_wait,
        } => {
            let js_mode = if js || intercept { JsMode::Force } else { JsMode::Auto };
            cmd_feed(&source, &format, sel.as_deref(), output.as_deref(), no_cache, js_mode, intercept, js_wait);
        }

        Commands::Intercept {
            source,
            format,
            output,
            js_wait,
        } => {
            cmd_intercept(&source, &format, output.as_deref(), js_wait);
        }

        Commands::Login { url, profile, timeout } => {
            cmd_login(&url, &profile, timeout);
        }

        Commands::Session(action) => match action {
            SessionAction::List => cmd_session_list(),
            SessionAction::Info { profile } => cmd_session_info(&profile),
            SessionAction::Export { profile, output } => cmd_session_export(&profile, &output),
            SessionAction::Import { profile, input } => cmd_session_import(&profile, &input),
            SessionAction::Browse { url, profile, format, selector, output, js_wait } => {
                cmd_session_browse(&url, &profile, &format, selector.as_deref(), output.as_deref(), js_wait);
            }
        },

        Commands::Screenshot { source, output, profile, js_wait } => {
            cmd_screenshot(&source, &output, &profile, js_wait);
        }

        Commands::Run { script, profile, headed, timeout } => {
            cmd_run(&script, &profile, headed, timeout);
        }

        Commands::Cache { action } => match action {
            CacheAction::Stats => cmd_cache_stats(),
            CacheAction::Clear => cmd_cache_clear(),
        },

        Commands::Meta { source, format, output, no_cache, js, js_wait } => {
            let js_mode = if js { JsMode::Force } else { JsMode::Off };
            cmd_meta(&source, &format, output.as_deref(), no_cache, js_mode, js_wait);
        }

        Commands::Crawl {
            source,
            links,
            extract: ext,
            max_pages,
            format,
            output,
            js,
            js_wait,
            delay,
        } => {
            let js_mode = if js { JsMode::Force } else { JsMode::Auto };
            cmd_crawl(&source, &links, ext.as_deref(), max_pages, &format, output.as_deref(), js_mode, js_wait, delay);
        }

        Commands::Batch {
            urls_file,
            query: q,
            max_width,
            format,
            output,
            parallel,
        } => cmd_batch(&urls_file, &q, max_width, &format, output.as_deref(), parallel),
    }
}

#[derive(Clone, Copy)]
enum JsMode {
    Off,
    Force,
    Auto,
}

fn cmd_view(source: &str, no_cache: bool, cache_age: u64, js_mode: JsMode, js_wait: u64) {
    let total_start = Instant::now();

    let doc = load_document(source, no_cache, cache_age, js_mode, js_wait);
    let doc = match doc {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    let total_ms = total_start.elapsed().as_millis();

    // Display summary
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║ OctoView Document                                      ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ URL:   {:<49}║", truncate_str(&doc.url, 49));
    println!("║ Title: {:<49}║", truncate_str(&doc.title, 49));
    println!("║ Nodes: {:<49}║", doc.nodes.len());
    println!("║ Time:  {:<49}║", format!("{}ms", total_ms));
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Show content by type
    let headings = doc.query_type(ovd::NodeType::Heading);
    if !headings.is_empty() {
        println!("── Headings ({}) ──", headings.len());
        for h in &headings {
            let indent = "  ".repeat(h.level.saturating_sub(1) as usize);
            println!("  {}H{}: {}", indent, h.level, truncate_str(&h.text, 70));
        }
        println!();
    }

    let links = doc.query_type(ovd::NodeType::Link);
    if !links.is_empty() {
        let visible_links: Vec<&&ovd::OvdNode> = links.iter().filter(|l| !l.text.is_empty()).collect();
        println!("── Links ({} total, {} with text) ──", links.len(), visible_links.len());
        for l in visible_links.iter().take(20) {
            println!("  {} → {}", truncate_str(&l.text, 40), truncate_str(&l.href, 50));
        }
        if visible_links.len() > 20 {
            println!("  ... and {} more", visible_links.len() - 20);
        }
        println!();
    }

    let images = doc.query_type(ovd::NodeType::Image);
    if !images.is_empty() {
        println!("── Images ({}) ──", images.len());
        for img in images.iter().take(10) {
            let desc = if !img.alt.is_empty() {
                &img.alt
            } else {
                &img.src
            };
            println!("  {}", truncate_str(desc, 70));
        }
        if images.len() > 10 {
            println!("  ... and {} more", images.len() - 10);
        }
        println!();
    }

    let stats = doc.stats();
    println!("── Stats ──");
    print!("{}", stats);
}

fn cmd_query(source: &str, q: &str, max_width: usize, format: &str, output: Option<&str>, no_cache: bool, cache_age: u64, js_mode: JsMode, js_wait: u64) {
    let total_start = Instant::now();

    let doc = match load_document(source, no_cache, cache_age, js_mode, js_wait) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    // Support css: prefix — route to selector engine
    if q.starts_with("css:") || q.starts_with("CSS:") {
        let css_sel = &q[4..].trim();
        let nodes = selector::select(&doc, css_sel);
        let total_ms = total_start.elapsed().as_millis();
        let fields = vec!["type", "text", "href", "class", "depth"];
        let result = match format {
            "csv" => query::format_csv(&nodes, &fields),
            "json" => query::format_json(&nodes, &fields),
            "jsonl" => query::format_jsonl(&nodes, &fields),
            "feed" => selector::format_feed_nodes(&nodes, css_sel),
            _ => format!(
                "-- css: {} ({} matches, {}ms) --\n{}",
                css_sel, nodes.len(), total_ms,
                query::format_results(&nodes, &fields, max_width)
            ),
        };
        write_output(&result, output);
        return;
    }

    let parsed = query::execute_query(&doc, q);
    let (nodes, fields) = parsed;
    let field_refs: Vec<&str> = fields.iter().map(|s| s.as_str()).collect();

    // Check for COUNT(*)
    if fields.len() == 1 && fields[0] == "count" {
        let total_ms = total_start.elapsed().as_millis();
        let output_str = format!("-- {} ({}ms) --\n{}", doc.url, total_ms, query::format_count(nodes.len()));
        write_output(&output_str, output);
        return;
    }

    // Check for DISTINCT
    let q_upper = q.to_uppercase();
    if q_upper.contains("SELECT DISTINCT") && !field_refs.is_empty() && field_refs[0] != "*" {
        let field = field_refs[0];
        let mut values: Vec<String> = nodes.iter().map(|n| query::node_field(n, field)).collect();
        values.sort();
        values.dedup();
        let total_ms = total_start.elapsed().as_millis();
        let output_str = format!(
            "-- {} ({}ms) --\n{}",
            doc.url,
            total_ms,
            query::format_distinct(&values, field)
        );
        write_output(&output_str, output);
        return;
    }

    let total_ms = total_start.elapsed().as_millis();

    let result = match format {
        "csv" => {
            format!("{}", query::format_csv(&nodes, &field_refs))
        }
        "json" => {
            format!("{}", query::format_json(&nodes, &field_refs))
        }
        "jsonl" => {
            format!("{}", query::format_jsonl(&nodes, &field_refs))
        }
        _ => {
            format!(
                "-- {} ({} results, {}ms) --\n{}",
                doc.url,
                nodes.len(),
                total_ms,
                query::format_results(&nodes, &field_refs, max_width)
            )
        }
    };

    write_output(&result, output);
}

fn cmd_shell(source: Option<&str>, js: bool, js_wait: u64) {
    let initial_doc = if let Some(src) = source {
        let js_mode = if js { JsMode::Force } else { JsMode::Auto };
        match load_document(src, false, 3600, js_mode, js_wait) {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("Error loading initial page: {}", e);
                None
            }
        }
    } else {
        None
    };

    let load_fn = move |url: &str, use_js: bool, wait: u64| -> Result<ovd::OvdDocument, String> {
        let js_mode = if use_js { JsMode::Force } else { JsMode::Auto };
        load_document(url, false, 3600, js_mode, wait)
    };

    shell::run_shell(initial_doc, load_fn);
}

fn cmd_browse(source: &str, no_cache: bool, cache_age: u64, js_mode: JsMode, js_wait: u64) {
    let doc = match load_document(source, no_cache, cache_age, js_mode, js_wait) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    let load_fn = move |url: &str| -> Result<ovd::OvdDocument, String> {
        load_document_static(url, false, 3600)
    };

    if let Err(e) = tui::run_tui(doc, load_fn) {
        eprintln!("TUI error: {}", e);
        std::process::exit(1);
    }
}

fn cmd_stats(source: &str, no_cache: bool, js_mode: JsMode, js_wait: u64) {
    let doc = match load_document(source, no_cache, 3600, js_mode, js_wait) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    println!("URL:   {}", doc.url);
    println!("Title: {}", doc.title);
    println!();

    let stats = doc.stats();
    print!("{}", stats);
}

fn cmd_save(source: &str, output: &str, js_mode: JsMode, js_wait: u64) {
    let doc = match load_document(source, true, 0, js_mode, js_wait) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    match ovd::write_ovd(&doc, output) {
        Ok(()) => println!("Saved {} nodes to {}", doc.nodes.len(), output),
        Err(e) => {
            eprintln!("Error writing .ovd: {}", e);
            std::process::exit(1);
        }
    }
}

fn cmd_load(path: &str, q: Option<&str>) {
    let doc = match ovd::read_ovd(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error reading .ovd: {}", e);
            std::process::exit(1);
        }
    };

    if let Some(q) = q {
        let (nodes, fields) = query::execute_query(&doc, q);
        let field_refs: Vec<&str> = fields.iter().map(|s| s.as_str()).collect();
        let table = query::format_results(&nodes, &field_refs, 60);
        println!("-- {} ({} results) --", doc.url, nodes.len());
        println!("{}", table);
    } else {
        println!("Loaded: {} ({} nodes)", doc.url, doc.nodes.len());
        println!("Title:  {}", doc.title);
        let stats = doc.stats();
        print!("{}", stats);
    }
}

fn cmd_watch(
    source: &str,
    interval: u64,
    max_checks: Option<usize>,
    only_changes: bool,
    output: Option<&str>,
    jsonl: bool,
    js: bool,
    js_wait: u64,
) {
    let use_js = js;
    let load_fn = move |url: &str| -> Result<ovd::OvdDocument, String> {
        let js_mode = if use_js { JsMode::Force } else { JsMode::Off };
        // Always bypass cache for watch mode — we want fresh data
        load_document(url, true, 0, js_mode, js_wait)
    };

    watch::run_watch(
        source,
        interval,
        max_checks,
        only_changes,
        output,
        jsonl,
        load_fn,
    );
}

fn cmd_diff(source_a: &str, source_b: Option<&str>, js_mode: JsMode, js_wait: u64) {
    // Load first document from cache (old version)
    let doc_a = if let Some(_src_b) = source_b {
        // Two explicit sources
        match load_document(source_a, false, 3600, js_mode, js_wait) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Error loading first source: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        // Compare against cache — load cached version first
        match cache::cache_lookup(source_a, u64::MAX) {
            Some(d) => d,
            None => {
                eprintln!("No cached version of '{}' found. Fetch it first with 'octoview view'.", source_a);
                std::process::exit(1);
            }
        }
    };

    // Load second document (fresh fetch)
    let doc_b_src = source_b.unwrap_or(source_a);
    let doc_b = match load_document(doc_b_src, true, 0, js_mode, js_wait) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error loading {}: {}", if source_b.is_some() { "second source" } else { "fresh version" }, e);
            std::process::exit(1);
        }
    };

    // Run diff
    print_diff(&doc_a, &doc_b);

    // Update cache with fresh version
    let _ = cache::cache_store(&doc_b);
}

fn print_diff(old: &ovd::OvdDocument, new: &ovd::OvdDocument) {
    use std::collections::HashSet;

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║ OctoView Diff                                          ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Old: {:<51}║", truncate_str(&old.url, 51));
    println!("║ New: {:<51}║", truncate_str(&new.url, 51));
    println!("║ Nodes: {} → {}  ({}{}){:<width$}║",
        old.nodes.len(),
        new.nodes.len(),
        if new.nodes.len() >= old.nodes.len() { "+" } else { "" },
        new.nodes.len() as i64 - old.nodes.len() as i64,
        "",
        width = 51usize.saturating_sub(
            format!("Nodes: {} → {}  ({}{})",
                old.nodes.len(),
                new.nodes.len(),
                if new.nodes.len() >= old.nodes.len() { "+" } else { "" },
                new.nodes.len() as i64 - old.nodes.len() as i64
            ).len()
        )
    );
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Type-level diff
    let old_stats = old.stats();
    let new_stats = new.stats();

    let type_diffs = [
        ("Headings", old_stats.headings, new_stats.headings),
        ("Paragraphs", old_stats.paragraphs, new_stats.paragraphs),
        ("Links", old_stats.links, new_stats.links),
        ("Images", old_stats.images, new_stats.images),
        ("Tables", old_stats.tables, new_stats.tables),
        ("Lists", old_stats.lists, new_stats.lists),
        ("Forms", old_stats.forms, new_stats.forms),
        ("Code blocks", old_stats.code_blocks, new_stats.code_blocks),
    ];

    let mut any_type_diff = false;
    println!("── Node Type Changes ──");
    for (name, old_count, new_count) in &type_diffs {
        if old_count != new_count {
            let delta = *new_count as i64 - *old_count as i64;
            let sign = if delta > 0 { "+" } else { "" };
            println!("  {:<15} {} → {}  ({}{})", name, old_count, new_count, sign, delta);
            any_type_diff = true;
        }
    }
    if !any_type_diff {
        println!("  (no type-level changes)");
    }
    println!();

    // Link diff
    let old_links: HashSet<(String, String)> = old
        .nodes
        .iter()
        .filter(|n| n.node_type == ovd::NodeType::Link && !n.text.is_empty())
        .map(|n| (n.text.clone(), n.href.clone()))
        .collect();
    let new_links: HashSet<(String, String)> = new
        .nodes
        .iter()
        .filter(|n| n.node_type == ovd::NodeType::Link && !n.text.is_empty())
        .map(|n| (n.text.clone(), n.href.clone()))
        .collect();

    let added_links: Vec<&(String, String)> = new_links.difference(&old_links).collect();
    let removed_links: Vec<&(String, String)> = old_links.difference(&new_links).collect();

    if !added_links.is_empty() || !removed_links.is_empty() {
        println!("── Link Changes ──");
        for (text, href) in added_links.iter().take(15) {
            println!("  + {} → {}", truncate_str(text, 40), truncate_str(href, 50));
        }
        if added_links.len() > 15 {
            println!("  ... and {} more added", added_links.len() - 15);
        }
        for (text, href) in removed_links.iter().take(15) {
            println!("  - {} → {}", truncate_str(text, 40), truncate_str(href, 50));
        }
        if removed_links.len() > 15 {
            println!("  ... and {} more removed", removed_links.len() - 15);
        }
        println!();
    }

    // Heading diff
    let old_headings: HashSet<String> = old
        .nodes
        .iter()
        .filter(|n| n.node_type == ovd::NodeType::Heading)
        .map(|n| n.text.clone())
        .collect();
    let new_headings: HashSet<String> = new
        .nodes
        .iter()
        .filter(|n| n.node_type == ovd::NodeType::Heading)
        .map(|n| n.text.clone())
        .collect();

    let added_h: Vec<&String> = new_headings.difference(&old_headings).collect();
    let removed_h: Vec<&String> = old_headings.difference(&new_headings).collect();

    if !added_h.is_empty() || !removed_h.is_empty() {
        println!("── Heading Changes ──");
        for h in &added_h {
            println!("  + {}", truncate_str(h, 70));
        }
        for h in &removed_h {
            println!("  - {}", truncate_str(h, 70));
        }
        println!();
    }

    // Text content diff (paragraphs)
    let old_paras: HashSet<String> = old
        .nodes
        .iter()
        .filter(|n| n.node_type == ovd::NodeType::Paragraph && n.text.len() > 20)
        .map(|n| n.text.clone())
        .collect();
    let new_paras: HashSet<String> = new
        .nodes
        .iter()
        .filter(|n| n.node_type == ovd::NodeType::Paragraph && n.text.len() > 20)
        .map(|n| n.text.clone())
        .collect();

    let added_p: Vec<&String> = new_paras.difference(&old_paras).collect();
    let removed_p: Vec<&String> = old_paras.difference(&new_paras).collect();

    if !added_p.is_empty() || !removed_p.is_empty() {
        println!("── Content Changes ──");
        for p in added_p.iter().take(5) {
            println!("  + {}", truncate_str(p, 80));
        }
        if added_p.len() > 5 {
            println!("  ... and {} more paragraphs added", added_p.len() - 5);
        }
        for p in removed_p.iter().take(5) {
            println!("  - {}", truncate_str(p, 80));
        }
        if removed_p.len() > 5 {
            println!("  ... and {} more paragraphs removed", removed_p.len() - 5);
        }
        println!();
    }

    // Summary
    let total_changes = added_links.len() + removed_links.len() + added_h.len() + removed_h.len() + added_p.len() + removed_p.len();
    if total_changes == 0 && !any_type_diff {
        println!("No structural differences detected.");
    } else {
        println!("── Summary ──");
        println!(
            "  {} changes: +{} links, -{} links, +{} headings, -{} headings, +{} paragraphs, -{} paragraphs",
            total_changes,
            added_links.len(),
            removed_links.len(),
            added_h.len(),
            removed_h.len(),
            added_p.len(),
            removed_p.len()
        );
    }
}

fn cmd_select(source: &str, sel: &str, format: &str, max_width: usize, output: Option<&str>, no_cache: bool, js_mode: JsMode, js_wait: u64) {
    let total_start = Instant::now();

    let doc = match load_document(source, no_cache, 3600, js_mode, js_wait) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    let nodes = selector::select(&doc, sel);
    let total_ms = total_start.elapsed().as_millis();

    let result = match format {
        "feed" => {
            let node_refs: Vec<&ovd::OvdNode> = nodes;
            selector::format_feed_nodes(&node_refs, sel)
        }
        "csv" => {
            let fields = vec!["type", "text", "href", "class", "level", "depth", "semantic"];
            query::format_csv(&nodes, &fields)
        }
        "json" => {
            let fields = vec!["type", "text", "href", "class", "level", "depth", "semantic"];
            query::format_json(&nodes, &fields)
        }
        "jsonl" => {
            let fields = vec!["type", "text", "href", "class", "level", "depth", "semantic"];
            query::format_jsonl(&nodes, &fields)
        }
        _ => {
            let fields = vec!["type", "text", "href", "class", "depth"];
            format!(
                "-- css: {} ({} matches, {}ms) --\n{}",
                sel,
                nodes.len(),
                total_ms,
                query::format_results(&nodes, &fields, max_width)
            )
        }
    };

    write_output(&result, output);
}

fn cmd_feed(source: &str, format: &str, sel: Option<&str>, output: Option<&str>, no_cache: bool, js_mode: JsMode, intercept: bool, js_wait: u64) {
    let total_start = Instant::now();

    // If intercept mode, use API interception engine
    if intercept {
        match js_engine::fetch_with_js_and_apis(source, js_wait) {
            Ok(result) => {
                let doc = extract::extract_html(&result.render.html, &result.render.url);
                if !no_cache {
                    let _ = cache::cache_store(&doc);
                }

                let total_ms = total_start.elapsed().as_millis();
                let mut feed_output = String::new();

                // Page feed
                if let Some(sel) = sel {
                    let nodes = selector::select(&doc, sel);
                    feed_output.push_str(&selector::format_feed_nodes(&nodes, sel));
                } else {
                    match format {
                        "jsonl" => feed_output.push_str(&selector::format_feed_jsonl(&doc)),
                        _ => feed_output.push_str(&selector::format_feed(&doc)),
                    }
                }

                // API data section
                if !result.api_calls.is_empty() {
                    feed_output.push_str("\n## Intercepted APIs\n");
                    feed_output.push_str(&format!("@api_calls {}\n\n", result.api_calls.len()));

                    for call in &result.api_calls {
                        if call.status == 0 || call.url.is_empty() {
                            continue;
                        }
                        feed_output.push_str(&format!(
                            "- {} {} → {} ({})\n",
                            call.method,
                            truncate_str(&call.url, 70),
                            call.status,
                            call.content_type
                        ));
                    }

                    // Show JSON responses
                    let json_apis: Vec<&js_engine::ApiCall> = result.api_calls.iter()
                        .filter(|c| c.response_body.is_some() && c.content_type.contains("json"))
                        .collect();

                    if !json_apis.is_empty() {
                        feed_output.push_str(&format!("\n### JSON Data ({} endpoints)\n", json_apis.len()));
                        for call in &json_apis {
                            if let Some(ref body) = call.response_body {
                                feed_output.push_str(&format!("\n#### {} {}\n", call.method, truncate_str(&call.url, 60)));
                                let preview = if body.len() > 2000 {
                                    format!("{}...\n({} bytes total)", &body[..2000], body.len())
                                } else {
                                    body.clone()
                                };
                                feed_output.push_str(&format!("```json\n{}\n```\n", preview));
                            }
                        }
                    }
                }

                feed_output.push_str(&format!("\n@elapsed {}ms\n", total_ms));
                write_output(&feed_output, output);
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        return;
    }

    // Standard feed (no interception)
    let doc = match load_document(source, no_cache, 3600, js_mode, js_wait) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    let result = if let Some(sel) = sel {
        let nodes = selector::select(&doc, sel);
        selector::format_feed_nodes(&nodes, sel)
    } else {
        match format {
            "jsonl" => selector::format_feed_jsonl(&doc),
            _ => selector::format_feed(&doc),
        }
    };

    write_output(&result, output);
}

fn cmd_intercept(source: &str, format: &str, output: Option<&str>, js_wait: u64) {
    eprintln!("Intercepting API calls from {}...", source);

    match js_engine::fetch_with_js_and_apis(source, js_wait) {
        Ok(result) => {
            let output_str = match format {
                "jsonl" => js_engine::format_api_calls_jsonl(&result.api_calls),
                _ => js_engine::format_api_calls(&result.api_calls),
            };
            write_output(&output_str, output);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

fn cmd_login(url: &str, profile: &str, timeout: u64) {
    match session::interactive_login(profile, url, timeout) {
        Ok(sess) => {
            match sess.get_cookies() {
                Ok(cookies) => {
                    println!("\nSession saved with {} cookies.", cookies.len());
                    println!("Use these to browse authenticated pages:");
                    println!("  octoview session browse <url> --profile {}", profile);
                    println!("  octoview feed <url> --js --profile {}", profile);
                }
                Err(e) => eprintln!("Warning: could not read cookies: {}", e),
            }
        }
        Err(e) => {
            eprintln!("Login error: {}", e);
            std::process::exit(1);
        }
    }
}

fn cmd_session_list() {
    let profiles = session::list_profiles();
    if profiles.is_empty() {
        println!("No profiles found.");
        println!("Create one with: octoview login <url> --profile <name>");
    } else {
        println!("── Browser Profiles ──");
        for name in &profiles {
            println!("  {}", name);
        }
        println!();
        println!("{} profile(s) in ~/.octoview/profiles/", profiles.len());
    }
}

fn cmd_session_info(profile: &str) {
    match session::BrowserSession::new(profile, false, 30) {
        Ok(sess) => {
            println!("{}", sess.profile_info());
            match sess.get_cookies() {
                Ok(cookies) => println!("Cookies: {}", cookies.len()),
                Err(e) => println!("Cookies: error ({})", e),
            }
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}

fn cmd_session_export(profile: &str, output: &str) {
    match session::BrowserSession::new(profile, false, 30) {
        Ok(sess) => {
            match session::export_cookies(&sess, output) {
                Ok(count) => println!("Exported {} cookies to {}", count, output),
                Err(e) => eprintln!("Error: {}", e),
            }
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}

fn cmd_session_import(profile: &str, input: &str) {
    match session::BrowserSession::new(profile, false, 30) {
        Ok(sess) => {
            match session::import_cookies(&sess, input) {
                Ok(count) => println!("Imported {} cookies into profile '{}'", count, profile),
                Err(e) => eprintln!("Error: {}", e),
            }
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}

fn cmd_session_browse(url: &str, profile: &str, format: &str, sel: Option<&str>, output: Option<&str>, js_wait: u64) {
    let total_start = Instant::now();

    let sess = match session::BrowserSession::new(profile, false, js_wait + 60) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    let doc = match sess.navigate(url) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    let total_ms = total_start.elapsed().as_millis();

    let result = if let Some(sel) = sel {
        let nodes = selector::select(&doc, sel);
        match format {
            "feed" => selector::format_feed_nodes(&nodes, sel),
            "jsonl" => {
                let fields = vec!["type", "text", "href", "class", "depth"];
                query::format_jsonl(&nodes, &fields)
            }
            _ => {
                let fields = vec!["type", "text", "href", "class", "depth"];
                format!(
                    "-- {} (profile: {}, {} matches, {}ms) --\n{}",
                    url, profile, nodes.len(), total_ms,
                    query::format_results(&nodes, &fields, 60)
                )
            }
        }
    } else {
        match format {
            "feed" => selector::format_feed(&doc),
            "jsonl" => selector::format_feed_jsonl(&doc),
            _ => {
                let mut out = String::new();
                out.push_str(&format!("╔══════════════════════════════════════════════════════════╗\n"));
                out.push_str(&format!("║ OctoView (profile: {:<36})║\n", profile));
                out.push_str(&format!("╠══════════════════════════════════════════════════════════╣\n"));
                out.push_str(&format!("║ URL:   {:<49}║\n", truncate_str(&doc.url, 49)));
                out.push_str(&format!("║ Title: {:<49}║\n", truncate_str(&doc.title, 49)));
                out.push_str(&format!("║ Nodes: {:<49}║\n", doc.nodes.len()));
                out.push_str(&format!("║ Time:  {:<49}║\n", format!("{}ms", total_ms)));
                out.push_str(&format!("╚══════════════════════════════════════════════════════════╝\n"));
                out.push_str(&format!("\n{}", doc.stats()));
                out
            }
        }
    };

    write_output(&result, output);
}

fn cmd_screenshot(source: &str, output: &str, profile: &str, js_wait: u64) {
    let sess = match session::BrowserSession::new(profile, false, js_wait + 60) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    // Navigate first
    match sess.navigate(source) {
        Ok(doc) => {
            eprintln!("Page: {} ({} nodes)", truncate_str(&doc.title, 50), doc.nodes.len());
        }
        Err(e) => {
            eprintln!("Navigation error: {}", e);
            std::process::exit(1);
        }
    }

    // Take screenshot
    match sess.screenshot_to_file(output) {
        Ok(()) => println!("Screenshot saved to {}", output),
        Err(e) => {
            eprintln!("Screenshot error: {}", e);
            std::process::exit(1);
        }
    }
}

fn cmd_cache_stats() {
    let (count, bytes) = cache::cache_stats();
    println!("OctoView Cache");
    println!("  Pages cached: {}", count);
    println!("  Total size:   {}", format_bytes(bytes));
}

fn cmd_cache_clear() {
    match cache::cache_clear() {
        Ok(count) => println!("Cleared {} cached pages", count),
        Err(e) => eprintln!("Error: {}", e),
    }
}

fn cmd_batch(urls_file: &str, q: &str, max_width: usize, format: &str, output: Option<&str>, parallel: usize) {
    let urls = match std::fs::read_to_string(urls_file) {
        Ok(content) => content
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect::<Vec<_>>(),
        Err(e) => {
            eprintln!("Error reading URLs file: {}", e);
            std::process::exit(1);
        }
    };

    eprintln!("Batch query: {} URLs (parallel: {})", urls.len(), parallel);
    eprintln!("Query: {}", q);
    eprintln!();

    let mut all_output = String::new();

    if parallel > 1 {
        // Parallel batch using threads
        use std::sync::{Arc, Mutex};
        let results: Arc<Mutex<Vec<(usize, String, Result<(Vec<ovd::OvdNode>, Vec<String>, String), String>)>>> =
            Arc::new(Mutex::new(Vec::new()));

        let indexed: Vec<(usize, String)> = urls.iter().enumerate().map(|(i, u)| (i, u.clone())).collect();
        let chunk_size = ((indexed.len() + parallel - 1) / parallel).max(1);
        let chunks: Vec<Vec<(usize, String)>> = indexed
            .chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        let mut handles = Vec::new();
        for chunk in chunks {
            let q = q.to_string();
            let results = Arc::clone(&results);
            let handle = std::thread::spawn(move || {
                for (idx, url) in chunk {
                    let result = match load_document_static(&url, false, 3600) {
                        Ok(doc) => {
                            let (nodes, fields) = query::execute_query(&doc, &q);
                            let owned_nodes: Vec<ovd::OvdNode> = nodes.into_iter().cloned().collect();
                            let url = doc.url.clone();
                            Ok((owned_nodes, fields, url))
                        }
                        Err(e) => Err(e),
                    };
                    results.lock().unwrap().push((idx, url, result));
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            let _ = handle.join();
        }

        let mut results = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
        results.sort_by_key(|(idx, _, _)| *idx);

        for (_, url, result) in &results {
            match result {
                Ok((nodes, fields, doc_url)) => {
                    let node_refs: Vec<&ovd::OvdNode> = nodes.iter().collect();
                    let field_refs: Vec<&str> = fields.iter().map(|s| s.as_str()).collect();
                    let formatted = match format {
                        "csv" => query::format_csv(&node_refs, &field_refs),
                        "json" => query::format_json(&node_refs, &field_refs),
                        "jsonl" => query::format_jsonl(&node_refs, &field_refs),
                        _ => {
                            format!(
                                "=== {} ({} results) ===\n{}",
                                truncate_str(doc_url, 60),
                                nodes.len(),
                                query::format_results(&node_refs, &field_refs, max_width)
                            )
                        }
                    };
                    all_output.push_str(&formatted);
                }
                Err(e) => {
                    all_output.push_str(&format!("=== {} (ERROR: {}) ===\n", url, e));
                }
            }
        }
    } else {
        // Sequential batch
        for url in &urls {
            match load_document(url, false, 3600, JsMode::Off, 15) {
                Ok(doc) => {
                    let (nodes, fields) = query::execute_query(&doc, q);
                    let field_refs: Vec<&str> = fields.iter().map(|s| s.as_str()).collect();
                    let formatted = match format {
                        "csv" => query::format_csv(&nodes, &field_refs),
                        "json" => query::format_json(&nodes, &field_refs),
                        "jsonl" => query::format_jsonl(&nodes, &field_refs),
                        _ => {
                            format!(
                                "=== {} ({} results) ===\n{}",
                                truncate_str(url, 60),
                                nodes.len(),
                                query::format_results(&nodes, &field_refs, max_width)
                            )
                        }
                    };
                    all_output.push_str(&formatted);
                }
                Err(e) => {
                    all_output.push_str(&format!("=== {} (ERROR: {}) ===\n", url, e));
                }
            }
        }
    }

    write_output(&all_output, output);
}

/// Simplified load_document for parallel batch (no JS engine, thread-safe).
fn load_document_static(source: &str, no_cache: bool, cache_age: u64) -> Result<ovd::OvdDocument, String> {
    if !no_cache {
        if let Some(doc) = cache::cache_lookup(source, cache_age) {
            return Ok(doc);
        }
    }

    let url = if source.starts_with("http://") || source.starts_with("https://") {
        source.to_string()
    } else if std::path::Path::new(source).exists() {
        let result = fetch::read_file(source)?;
        let doc = extract::extract_html(&result.html, &result.url);
        if !no_cache {
            let _ = cache::cache_store(&doc);
        }
        return Ok(doc);
    } else {
        format!("https://{}", source)
    };

    let result = fetch::fetch_url(&url)?;
    let doc = extract::extract_html(&result.html, &result.url);
    if !no_cache {
        let _ = cache::cache_store(&doc);
    }
    Ok(doc)
}

/// Load a document from URL, file, or cache.
fn load_document(
    source: &str,
    no_cache: bool,
    cache_age: u64,
    js_mode: JsMode,
    js_wait: u64,
) -> Result<ovd::OvdDocument, String> {
    // Check cache first
    if !no_cache {
        if let Some(doc) = cache::cache_lookup(source, cache_age) {
            return Ok(doc);
        }
    }

    // Determine URL for JS engine
    let url = if source.starts_with("http://") || source.starts_with("https://") {
        source.to_string()
    } else if std::path::Path::new(source).exists() {
        // Local file — no JS needed
        let result = fetch::read_file(source)?;
        let doc = extract::extract_html(&result.html, &result.url);
        if !no_cache {
            let _ = cache::cache_store(&doc);
        }
        return Ok(doc);
    } else {
        format!("https://{}", source)
    };

    // Force JS mode — go straight to headless Chrome
    if matches!(js_mode, JsMode::Force) {
        return load_with_js(&url, no_cache, js_wait);
    }

    // Static fetch first
    let result = fetch::fetch_url(&url)?;
    let doc = extract::extract_html(&result.html, &result.url);

    // Auto mode: check if static extraction looks incomplete
    if matches!(js_mode, JsMode::Auto) {
        if js_engine::needs_js_rendering(&result.html, doc.nodes.len()) {
            eprintln!("[auto-js] Static extraction found only {} nodes, switching to JS engine...", doc.nodes.len());
            return load_with_js(&url, no_cache, js_wait);
        }
    }

    // Cache the result
    if !no_cache {
        let _ = cache::cache_store(&doc);
    }

    Ok(doc)
}

/// Load a document using headless Chrome JS rendering.
fn load_with_js(url: &str, no_cache: bool, js_wait: u64) -> Result<ovd::OvdDocument, String> {
    let result = js_engine::fetch_with_js(url, js_wait)?;
    let doc = extract::extract_html(&result.html, &result.url);

    if !no_cache {
        let _ = cache::cache_store(&doc);
    }

    Ok(doc)
}

fn cmd_meta(source: &str, format: &str, output: Option<&str>, no_cache: bool, js_mode: JsMode, js_wait: u64) {
    // Get raw HTML — we need it for meta extraction, not the OVD
    let html = get_raw_html(source, no_cache, js_mode, js_wait);
    let html = match html {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    let page_meta = meta::extract_meta(&html);

    let content = match format {
        "jsonl" => meta::format_meta_jsonl(&page_meta),
        _ => meta::format_meta(&page_meta),
    };

    write_output(&content, output);
}

/// Get raw HTML from a source (URL, file, or cache).
fn get_raw_html(source: &str, no_cache: bool, js_mode: JsMode, js_wait: u64) -> Result<String, String> {
    if std::path::Path::new(source).exists() {
        let result = fetch::read_file(source)?;
        return Ok(result.html);
    }

    let url = if source.starts_with("http://") || source.starts_with("https://") {
        source.to_string()
    } else {
        format!("https://{}", source)
    };

    if matches!(js_mode, JsMode::Force) {
        let result = js_engine::fetch_with_js(&url, js_wait)?;
        return Ok(result.html);
    }

    let _ = no_cache; // Static fetch doesn't use cache for raw HTML
    let result = fetch::fetch_url(&url)?;
    Ok(result.html)
}

fn cmd_crawl(
    source: &str,
    link_selector: &str,
    extract_selector: Option<&str>,
    max_pages: usize,
    format: &str,
    output: Option<&str>,
    js_mode: JsMode,
    js_wait: u64,
    delay: u64,
) {
    let start = Instant::now();

    // Step 1: Load the starting page
    eprintln!("[crawl] Loading start page: {}", source);
    let doc = match load_document(source, true, 0, js_mode, js_wait) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error loading start page: {}", e);
            std::process::exit(1);
        }
    };
    eprintln!("[crawl] Got {} nodes from {}", doc.nodes.len(), truncate_str(&doc.url, 60));

    // Step 2: Find links matching the selector
    let link_nodes = selector::select(&doc, link_selector);
    let mut urls: Vec<String> = Vec::new();
    let base_url = &doc.url;

    for node in &link_nodes {
        if !node.href.is_empty() {
            let url = resolve_url(base_url, &node.href);
            if !urls.contains(&url) {
                urls.push(url);
            }
        }
    }

    eprintln!(
        "[crawl] Found {} unique links matching '{}' (will follow up to {})",
        urls.len(),
        link_selector,
        max_pages
    );

    let urls: Vec<String> = urls.into_iter().take(max_pages).collect();

    if urls.is_empty() {
        eprintln!("[crawl] No links found. Try a different selector.");
        return;
    }

    // Step 3: Visit each linked page and extract
    let mut all_output = String::new();

    for (i, url) in urls.iter().enumerate() {
        eprintln!("[crawl {}/{}] {}", i + 1, urls.len(), truncate_str(url, 60));

        if delay > 0 && i > 0 {
            std::thread::sleep(std::time::Duration::from_millis(delay));
        }

        match load_document(url, true, 0, js_mode, js_wait) {
            Ok(child_doc) => {
                let header = format!(
                    "=== [{}] {} ({} nodes) ===\n",
                    i + 1,
                    truncate_str(&child_doc.title, 60),
                    child_doc.nodes.len()
                );

                match format {
                    "feed" => {
                        if let Some(sel) = extract_selector {
                            let nodes = selector::select(&child_doc, sel);
                            let feed = selector::format_feed_nodes(&nodes, sel);
                            all_output.push_str(&header);
                            all_output.push_str(&feed);
                            all_output.push('\n');
                        } else {
                            let feed = selector::format_feed(&child_doc);
                            all_output.push_str(&header);
                            all_output.push_str(&feed);
                            all_output.push('\n');
                        }
                    }
                    "jsonl" => {
                        if let Some(sel) = extract_selector {
                            let nodes = selector::select(&child_doc, sel);
                            let jsonl = selector::format_feed_nodes(&nodes, sel);
                            all_output.push_str(&jsonl);
                        } else {
                            let jsonl = selector::format_feed_jsonl(&child_doc);
                            all_output.push_str(&jsonl);
                        }
                    }
                    "csv" => {
                        let q = if let Some(sel) = extract_selector {
                            format!("SELECT text, href FROM nodes WHERE type = 'link' AND text LIKE '%{}%'", sel)
                        } else {
                            "SELECT type, text, href FROM nodes LIMIT 50".to_string()
                        };
                        let (nodes, fields) = query::execute_query(&child_doc, &q);
                        let field_refs: Vec<&str> = fields.iter().map(|s| s.as_str()).collect();
                        all_output.push_str(&query::format_csv(&nodes, &field_refs));
                    }
                    _ => {
                        if let Some(sel) = extract_selector {
                            let nodes = selector::select(&child_doc, sel);
                            let feed = selector::format_feed_nodes(&nodes, sel);
                            all_output.push_str(&header);
                            all_output.push_str(&feed);
                            all_output.push('\n');
                        } else {
                            let feed = selector::format_feed(&child_doc);
                            all_output.push_str(&header);
                            all_output.push_str(&feed);
                            all_output.push('\n');
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("  ERROR: {}", e);
                all_output.push_str(&format!("=== [{}] ERROR: {} ===\n", i + 1, e));
            }
        }
    }

    let total_ms = start.elapsed().as_millis();
    eprintln!("[crawl] Done: {} pages in {}ms", urls.len(), total_ms);

    write_output(&all_output, output);
}

/// Resolve a possibly-relative URL against a base URL.
fn resolve_url(base: &str, href: &str) -> String {
    if href.starts_with("http://") || href.starts_with("https://") {
        return href.to_string();
    }
    if href.starts_with("//") {
        return format!("https:{}", href);
    }
    if href.starts_with('/') {
        // Absolute path — extract origin from base
        if let Some(pos) = base.find("://") {
            if let Some(slash) = base[pos + 3..].find('/') {
                return format!("{}{}", &base[..pos + 3 + slash], href);
            }
        }
        return format!("{}{}", base.trim_end_matches('/'), href);
    }
    // Relative path
    if let Some(pos) = base.rfind('/') {
        format!("{}/{}", &base[..pos], href)
    } else {
        format!("{}/{}", base, href)
    }
}

fn cmd_run(script_path: &str, profile: &str, headed: bool, timeout: u64) {
    // Parse action script
    let actions = match action::parse_action_file(script_path) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    if actions.is_empty() {
        eprintln!("No actions found in '{}'", script_path);
        return;
    }

    eprintln!(
        "[run] {} actions from '{}' (profile: {}, {})",
        actions.len(),
        script_path,
        profile,
        if headed { "headed" } else { "headless" }
    );

    // Create browser session
    let session = match session::BrowserSession::new(profile, headed, timeout) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error creating session: {}", e);
            std::process::exit(1);
        }
    };

    // Run actions
    let result = action::run_actions(&session, &actions);

    // Print collected outputs
    if !result.outputs.is_empty() {
        eprintln!();
        eprintln!("── Outputs ({}) ──", result.outputs.len());
        for output in &result.outputs {
            eprintln!("  {}", output);
        }
    }

    // Report errors
    if !result.errors.is_empty() {
        eprintln!();
        eprintln!("── Errors ({}) ──", result.errors.len());
        for err in &result.errors {
            eprintln!("  {}", err);
        }
        std::process::exit(1);
    }
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}…", &s[..max.saturating_sub(1)])
    } else {
        s.to_string()
    }
}

/// Write output to file or stdout.
fn write_output(content: &str, output_path: Option<&str>) {
    if let Some(path) = output_path {
        match std::fs::write(path, content) {
            Ok(()) => eprintln!("Output written to {}", path),
            Err(e) => eprintln!("Error writing output: {}", e),
        }
    } else {
        println!("{}", content);
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}
