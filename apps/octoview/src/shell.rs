//! Interactive query shell — load pages, run queries, switch contexts.
//!
//! The shell maintains multiple loaded documents and provides an
//! interactive prompt for SQL-like queries against OctoView Documents.
//!
//! Commands:
//!   load <url>             Load a page (static or --js)
//!   pages                  List all loaded pages
//!   use <id|name>          Switch active page
//!   <query>                Run a SQL query against active page
//!   export <format> <file> <query>  Export query results to file
//!   info                   Show active page stats
//!   fields                 List available query fields
//!   diff <url>             Load page and diff against active
//!   js <url>               Load page with JS rendering
//!   clear                  Clear screen
//!   help                   Show help
//!   quit / exit            Exit shell

use crate::action;
use crate::meta;
use crate::ovd::OvdDocument;
use crate::query;
use crate::selector;
use crate::session;
use std::io::{self, BufRead, Write};

/// A loaded page in the shell context.
struct PageContext {
    id: usize,
    doc: OvdDocument,
    alias: String,
}

/// Shell state.
pub struct Shell {
    pages: Vec<PageContext>,
    active: usize, // Index into pages
    next_id: usize,
    history: Vec<String>,
    js_mode: bool,
    js_wait: u64,
    session: Option<session::BrowserSession>,
}

impl Shell {
    pub fn new() -> Self {
        Self {
            pages: Vec::new(),
            active: 0,
            next_id: 1,
            history: Vec::new(),
            js_mode: false,
            js_wait: 15,
            session: None,
        }
    }

    fn active_doc(&self) -> Option<&OvdDocument> {
        self.pages.get(self.active).map(|p| &p.doc)
    }

    fn active_alias(&self) -> &str {
        self.pages
            .get(self.active)
            .map(|p| p.alias.as_str())
            .unwrap_or("none")
    }

    fn add_page(&mut self, doc: OvdDocument) -> usize {
        let alias = url_to_alias(&doc.url);
        let id = self.next_id;
        self.next_id += 1;

        eprintln!(
            "[{}] loaded {} nodes from {}",
            alias,
            doc.nodes.len(),
            truncate(&doc.url, 60)
        );

        self.pages.push(PageContext { id, doc, alias });
        self.active = self.pages.len() - 1;
        id
    }
}

/// Run the interactive shell.
pub fn run_shell(
    initial_doc: Option<OvdDocument>,
    load_fn: impl Fn(&str, bool, u64) -> Result<OvdDocument, String>,
) {
    let mut shell = Shell::new();

    if let Some(doc) = initial_doc {
        shell.add_page(doc);
    }

    println!("OctoView Interactive Shell");
    println!("Type 'help' for commands, or enter a SQL query.");
    println!();

    let stdin = io::stdin();
    let mut reader = stdin.lock();

    loop {
        // Print prompt
        let prompt = if shell.pages.is_empty() {
            "ov> ".to_string()
        } else {
            format!("{}> ", shell.active_alias())
        };
        print!("{}", prompt);
        io::stdout().flush().unwrap();

        // Read line
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(_) => break,
        }

        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        // Save to history
        shell.history.push(line.clone());

        // Parse command
        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        let cmd = parts[0].to_lowercase();
        let args = if parts.len() > 1 { parts[1].trim() } else { "" };

        match cmd.as_str() {
            "quit" | "exit" | "q" => break,

            "help" | "?" => print_help(),

            "clear" | "cls" => {
                print!("\x1b[2J\x1b[H"); // ANSI clear screen
                io::stdout().flush().unwrap();
            }

            "load" | "open" => {
                if args.is_empty() {
                    eprintln!("Usage: load <url>");
                    continue;
                }
                let start = std::time::Instant::now();
                match load_fn(args, false, shell.js_wait) {
                    Ok(doc) => {
                        let ms = start.elapsed().as_millis();
                        let nodes = doc.nodes.len();
                        shell.add_page(doc);
                        println!("Loaded {} nodes in {}ms", nodes, ms);
                    }
                    Err(e) => eprintln!("Error: {}", e),
                }
            }

            "js" => {
                if args.is_empty() {
                    eprintln!("Usage: js <url>");
                    continue;
                }
                let start = std::time::Instant::now();
                match load_fn(args, true, shell.js_wait) {
                    Ok(doc) => {
                        let ms = start.elapsed().as_millis();
                        let nodes = doc.nodes.len();
                        shell.add_page(doc);
                        println!("Loaded {} nodes in {}ms (JS)", nodes, ms);
                    }
                    Err(e) => eprintln!("Error: {}", e),
                }
            }

            "pages" | "ls" => {
                if shell.pages.is_empty() {
                    println!("No pages loaded. Use 'load <url>' to load one.");
                } else {
                    println!("── Loaded Pages ──");
                    for (i, page) in shell.pages.iter().enumerate() {
                        let marker = if i == shell.active { "→" } else { " " };
                        println!(
                            " {} [{}] {} ({} nodes)",
                            marker,
                            page.id,
                            truncate(&page.doc.url, 60),
                            page.doc.nodes.len()
                        );
                    }
                }
            }

            "use" | "switch" => {
                if args.is_empty() {
                    eprintln!("Usage: use <id|alias>");
                    continue;
                }
                // Try by ID
                if let Ok(id) = args.parse::<usize>() {
                    if let Some(pos) = shell.pages.iter().position(|p| p.id == id) {
                        shell.active = pos;
                        println!("Switched to [{}] {}", id, truncate(&shell.pages[pos].doc.url, 60));
                        continue;
                    }
                }
                // Try by alias prefix
                if let Some(pos) = shell
                    .pages
                    .iter()
                    .position(|p| p.alias.starts_with(args))
                {
                    shell.active = pos;
                    let page = &shell.pages[pos];
                    println!("Switched to [{}] {}", page.id, truncate(&page.doc.url, 60));
                } else {
                    eprintln!("Page not found: '{}'", args);
                }
            }

            "info" | "stats" => {
                if let Some(doc) = shell.active_doc() {
                    println!("URL:   {}", doc.url);
                    println!("Title: {}", doc.title);
                    println!();
                    let stats = doc.stats();
                    print!("{}", stats);
                } else {
                    eprintln!("No page loaded.");
                }
            }

            "fields" => {
                println!("── Available Query Fields ──");
                println!("  Node:    node_id, type, depth, parent_id, text, href, src, alt");
                println!("  Meta:    level, semantic, source_tag, confidence");
                println!("  Style:   font_size, font_weight, bold, italic, color, background, display");
                println!();
                println!("  Types:   page, heading, paragraph, link, image, table, table_row,");
                println!("           table_cell, list, list_item, form, input_field, button,");
                println!("           navigation, media, code_block, blockquote, separator,");
                println!("           section, card, modal, sidebar, footer, header, text_span, icon");
                println!();
                println!("  Roles:   content, navigation, search, sidebar, advertising,");
                println!("           tracking, interactive, media, metadata, decoration, structural");
            }

            "export" => {
                // export csv links.csv SELECT text, href FROM nodes WHERE type = 'link'
                let export_parts: Vec<&str> = args.splitn(3, ' ').collect();
                if export_parts.len() < 3 {
                    eprintln!("Usage: export <csv|json|jsonl> <file> <query>");
                    continue;
                }
                let format = export_parts[0];
                let file = export_parts[1];
                let q = export_parts[2];

                if let Some(doc) = shell.active_doc() {
                    let (nodes, fields) = query::execute_query(doc, q);
                    let field_refs: Vec<&str> = fields.iter().map(|s| s.as_str()).collect();

                    let output = match format {
                        "csv" => query::format_csv(&nodes, &field_refs),
                        "json" => query::format_json(&nodes, &field_refs),
                        "jsonl" => query::format_jsonl(&nodes, &field_refs),
                        _ => {
                            eprintln!("Unknown format: '{}'. Use csv, json, or jsonl.", format);
                            continue;
                        }
                    };

                    match std::fs::write(file, &output) {
                        Ok(()) => println!("Exported {} results to {} ({})", nodes.len(), file, format),
                        Err(e) => eprintln!("Error writing file: {}", e),
                    }
                } else {
                    eprintln!("No page loaded.");
                }
            }

            "diff" => {
                if args.is_empty() {
                    eprintln!("Usage: diff <url>");
                    continue;
                }
                let current = match shell.active_doc() {
                    Some(d) => d,
                    None => {
                        eprintln!("No page loaded to diff against.");
                        continue;
                    }
                };

                let start = std::time::Instant::now();
                match load_fn(args, shell.js_mode, shell.js_wait) {
                    Ok(new_doc) => {
                        let ms = start.elapsed().as_millis();
                        println!("Loaded {} nodes in {}ms", new_doc.nodes.len(), ms);
                        print_diff(current, &new_doc);
                        shell.add_page(new_doc);
                    }
                    Err(e) => eprintln!("Error: {}", e),
                }
            }

            "session" | "profile" => {
                if args.is_empty() || args == "info" {
                    if let Some(ref sess) = shell.session {
                        println!("{}", sess.profile_info());
                        match sess.get_cookies() {
                            Ok(cookies) => println!("Cookies: {}", cookies.len()),
                            Err(e) => println!("Cookies: error ({})", e),
                        }
                    } else {
                        println!("No active session. Use 'session open <profile>' to start one.");
                        let profiles = session::list_profiles();
                        if !profiles.is_empty() {
                            println!("Available profiles: {}", profiles.join(", "));
                        }
                    }
                } else if args.starts_with("open ") || args.starts_with("connect ") {
                    let profile = args.splitn(2, ' ').nth(1).unwrap_or("default").trim();
                    match session::BrowserSession::new(profile, false, 120) {
                        Ok(sess) => {
                            println!("Session opened (profile: {})", profile);
                            shell.session = Some(sess);
                        }
                        Err(e) => eprintln!("Error: {}", e),
                    }
                } else if args == "close" {
                    shell.session = None;
                    println!("Session closed.");
                } else if args.starts_with("export ") {
                    let path = args.splitn(2, ' ').nth(1).unwrap_or("cookies.json").trim();
                    if let Some(ref sess) = shell.session {
                        match session::export_cookies(sess, path) {
                            Ok(n) => println!("Exported {} cookies to {}", n, path),
                            Err(e) => eprintln!("Error: {}", e),
                        }
                    } else {
                        eprintln!("No active session.");
                    }
                } else if args.starts_with("import ") {
                    let path = args.splitn(2, ' ').nth(1).unwrap_or("cookies.json").trim();
                    if let Some(ref sess) = shell.session {
                        match session::import_cookies(sess, path) {
                            Ok(n) => println!("Imported {} cookies", n),
                            Err(e) => eprintln!("Error: {}", e),
                        }
                    } else {
                        eprintln!("No active session.");
                    }
                } else {
                    eprintln!("Usage: session [open <profile>|close|export <file>|import <file>|info]");
                }
            }

            "goto" | "navigate" => {
                if args.is_empty() {
                    eprintln!("Usage: goto <url>");
                    continue;
                }
                if let Some(ref sess) = shell.session {
                    let start = std::time::Instant::now();
                    match sess.navigate(args) {
                        Ok(doc) => {
                            let ms = start.elapsed().as_millis();
                            let nodes = doc.nodes.len();
                            shell.add_page(doc);
                            println!("Loaded {} nodes in {}ms (session)", nodes, ms);
                        }
                        Err(e) => eprintln!("Error: {}", e),
                    }
                } else {
                    eprintln!("No session. Use 'session open <profile>' or 'load <url>'.");
                }
            }

            "type" => {
                let parts: Vec<&str> = args.splitn(2, ' ').collect();
                if parts.len() < 2 {
                    eprintln!("Usage: type \"selector\" text to type");
                    continue;
                }
                let selector_str = parts[0].trim_matches('"').trim_matches('\'');
                let text = parts[1];
                if let Some(ref sess) = shell.session {
                    match sess.type_into(selector_str, text) {
                        Ok(()) => println!("Typed into '{}'", selector_str),
                        Err(e) => eprintln!("Error: {}", e),
                    }
                } else {
                    eprintln!("No session. Use 'session open <profile>' first.");
                }
            }

            "click" => {
                if args.is_empty() {
                    eprintln!("Usage: click <css-selector>");
                    continue;
                }
                if let Some(ref sess) = shell.session {
                    match sess.click(args) {
                        Ok(()) => println!("Clicked '{}'", args),
                        Err(e) => eprintln!("Error: {}", e),
                    }
                } else {
                    eprintln!("No session. Use 'session open <profile>' first.");
                }
            }

            "press" => {
                if args.is_empty() {
                    eprintln!("Usage: press <key>  (Enter, Tab, Escape, etc.)");
                    continue;
                }
                if let Some(ref sess) = shell.session {
                    match sess.press_key(args) {
                        Ok(()) => println!("Pressed '{}'", args),
                        Err(e) => eprintln!("Error: {}", e),
                    }
                } else {
                    eprintln!("No session. Use 'session open <profile>' first.");
                }
            }

            "screenshot" | "snap" => {
                let path = if args.is_empty() { "screenshot.png" } else { args };
                if let Some(ref sess) = shell.session {
                    match sess.screenshot_to_file(path) {
                        Ok(()) => println!("Screenshot saved to {}", path),
                        Err(e) => eprintln!("Error: {}", e),
                    }
                } else {
                    eprintln!("No session. Use 'session open <profile>' first.");
                }
            }

            "eval" => {
                if args.is_empty() {
                    eprintln!("Usage: eval <javascript>");
                    continue;
                }
                if let Some(ref sess) = shell.session {
                    match sess.eval(args) {
                        Ok(result) => println!("{}", result),
                        Err(e) => eprintln!("Error: {}", e),
                    }
                } else {
                    eprintln!("No session. Use 'session open <profile>' first.");
                }
            }

            "cookies" => {
                if let Some(ref sess) = shell.session {
                    match sess.get_cookies() {
                        Ok(cookies) => {
                            println!("── {} Cookies ──", cookies.len());
                            for c in cookies.iter().take(30) {
                                println!("  {} = {} ({})", c.name, truncate(&c.value, 30), c.domain);
                            }
                            if cookies.len() > 30 {
                                println!("  ... and {} more", cookies.len() - 30);
                            }
                        }
                        Err(e) => eprintln!("Error: {}", e),
                    }
                } else {
                    eprintln!("No session. Use 'session open <profile>' first.");
                }
            }

            "meta" | "metadata" => {
                if args.is_empty() {
                    // Try to use active page's URL
                    if let Some(doc) = shell.active_doc() {
                        if doc.url.starts_with("http") {
                            let url = doc.url.clone();
                            match crate::fetch::fetch_url(&url) {
                                Ok(result) => {
                                    let m = meta::extract_meta(&result.html);
                                    print!("{}", meta::format_meta(&m));
                                }
                                Err(e) => eprintln!("Error fetching: {}", e),
                            }
                        } else {
                            eprintln!("Usage: meta [url]  (or load a page first)");
                        }
                    } else {
                        eprintln!("Usage: meta [url]");
                    }
                } else {
                    // Fetch URL and extract metadata
                    let start = std::time::Instant::now();
                    match crate::fetch::fetch_url(args) {
                        Ok(result) => {
                            let m = meta::extract_meta(&result.html);
                            let ms = start.elapsed().as_millis();
                            println!("({}ms)", ms);
                            print!("{}", meta::format_meta(&m));
                        }
                        Err(e) => eprintln!("Error: {}", e),
                    }
                }
            }

            "links" => {
                if let Some(doc) = shell.active_doc() {
                    let nodes: Vec<&crate::ovd::OvdNode> = doc
                        .nodes
                        .iter()
                        .filter(|n| n.node_type == crate::ovd::NodeType::Link && !n.text.is_empty())
                        .collect();
                    println!("── {} Links ──", nodes.len());
                    for (i, node) in nodes.iter().enumerate() {
                        let href = if node.href.is_empty() { "-" } else { &node.href };
                        println!("  {:3}. [{}]({})", i + 1, truncate(&node.text, 50), truncate(href, 50));
                    }
                } else {
                    eprintln!("No page loaded.");
                }
            }

            "headings" => {
                if let Some(doc) = shell.active_doc() {
                    let nodes: Vec<&crate::ovd::OvdNode> = doc
                        .nodes
                        .iter()
                        .filter(|n| n.node_type == crate::ovd::NodeType::Heading)
                        .collect();
                    println!("── {} Headings ──", nodes.len());
                    for node in &nodes {
                        let indent = "  ".repeat(node.level.saturating_sub(1) as usize);
                        println!("  {}H{}: {}", indent, node.level, truncate(&node.text, 60));
                    }
                } else {
                    eprintln!("No page loaded.");
                }
            }

            "images" => {
                if let Some(doc) = shell.active_doc() {
                    let nodes: Vec<&crate::ovd::OvdNode> = doc
                        .nodes
                        .iter()
                        .filter(|n| n.node_type == crate::ovd::NodeType::Image)
                        .collect();
                    println!("── {} Images ──", nodes.len());
                    for (i, node) in nodes.iter().enumerate() {
                        let alt = if node.alt.is_empty() { "(no alt)" } else { &node.alt };
                        let src = if node.src.is_empty() { "-" } else { &node.src };
                        println!("  {:3}. {} → {}", i + 1, truncate(alt, 30), truncate(src, 50));
                    }
                } else {
                    eprintln!("No page loaded.");
                }
            }

            "run" | "script" => {
                if args.is_empty() {
                    eprintln!("Usage: run <script.ova>  (or inline: run navigate https://example.com)");
                    continue;
                }

                // Check if it's a file path or inline script
                let actions = if args.ends_with(".ova") || std::path::Path::new(args).exists() {
                    match action::parse_action_file(args) {
                        Ok(a) => a,
                        Err(e) => {
                            eprintln!("Error: {}", e);
                            continue;
                        }
                    }
                } else {
                    // Treat as single inline action
                    action::parse_actions(args)
                };

                if actions.is_empty() {
                    eprintln!("No actions parsed.");
                    continue;
                }

                // Ensure we have a session
                if shell.session.is_none() {
                    match session::BrowserSession::new("default", false, 120) {
                        Ok(sess) => {
                            eprintln!("[auto] Opened session (profile: default)");
                            shell.session = Some(sess);
                        }
                        Err(e) => {
                            eprintln!("Error creating session: {}", e);
                            continue;
                        }
                    }
                }

                if let Some(ref sess) = shell.session {
                    let result = action::run_actions(sess, &actions);

                    // Add any documents to shell pages
                    for doc in result.documents {
                        shell.add_page(doc);
                    }

                    if !result.errors.is_empty() {
                        eprintln!("({} errors)", result.errors.len());
                    }
                }
            }

            "history" | "hist" => {
                for (i, cmd) in shell.history.iter().enumerate() {
                    println!("  {:3}  {}", i + 1, cmd);
                }
            }

            "drop" | "close" => {
                if shell.pages.is_empty() {
                    eprintln!("No pages to close.");
                    continue;
                }
                if args.is_empty() {
                    // Close active page
                    let removed = shell.pages.remove(shell.active);
                    println!("Closed [{}] {}", removed.id, truncate(&removed.doc.url, 60));
                    if shell.active >= shell.pages.len() && !shell.pages.is_empty() {
                        shell.active = shell.pages.len() - 1;
                    }
                } else if let Ok(id) = args.parse::<usize>() {
                    if let Some(pos) = shell.pages.iter().position(|p| p.id == id) {
                        let removed = shell.pages.remove(pos);
                        println!("Closed [{}] {}", removed.id, truncate(&removed.doc.url, 60));
                        if shell.active >= shell.pages.len() && !shell.pages.is_empty() {
                            shell.active = shell.pages.len() - 1;
                        }
                    } else {
                        eprintln!("Page not found: {}", id);
                    }
                }
            }

            "css" | "select" => {
                if args.is_empty() {
                    eprintln!("Usage: css <selector>  (e.g., css a.nav-link, css div.card > h2)");
                    continue;
                }
                if let Some(doc) = shell.active_doc() {
                    let start = std::time::Instant::now();
                    let nodes = selector::select(doc, args);
                    let ms = start.elapsed().as_millis();
                    let fields = vec!["type", "text", "href", "class", "depth"];
                    println!("({} matches, {}ms)", nodes.len(), ms);
                    print!("{}", query::format_results(&nodes, &fields, 60));
                } else {
                    eprintln!("No page loaded.");
                }
            }

            "feed" => {
                if let Some(doc) = shell.active_doc() {
                    if args.is_empty() {
                        print!("{}", selector::format_feed(doc));
                    } else {
                        // feed <selector> — filtered feed
                        let nodes = selector::select(doc, args);
                        print!("{}", selector::format_feed_nodes(&nodes, args));
                    }
                } else {
                    eprintln!("No page loaded.");
                }
            }

            // Default: treat as SQL query or css: prefix
            _ => {
                // Handle css: prefix in shell queries
                if line.starts_with("css:") || line.starts_with("CSS:") {
                    if let Some(doc) = shell.active_doc() {
                        let sel = line[4..].trim();
                        let start = std::time::Instant::now();
                        let nodes = selector::select(doc, sel);
                        let ms = start.elapsed().as_millis();
                        let fields = vec!["type", "text", "href", "class", "depth"];
                        println!("({} matches, {}ms)", nodes.len(), ms);
                        print!("{}", query::format_results(&nodes, &fields, 60));
                    } else {
                        eprintln!("No page loaded.");
                    }
                    continue;
                }
                if shell.pages.is_empty() {
                    eprintln!("No page loaded. Use 'load <url>' first.");
                    continue;
                }

                let doc = &shell.pages[shell.active].doc;
                let q = &line;

                // Check for COUNT
                let q_upper = q.to_uppercase();
                if q_upper.contains("COUNT(*)") {
                    let (nodes, _) = query::execute_query(doc, q);
                    println!("{}", query::format_count(nodes.len()));
                    continue;
                }

                // Check for DISTINCT
                if q_upper.contains("SELECT DISTINCT") {
                    let (nodes, fields) = query::execute_query(doc, q);
                    let field_refs: Vec<&str> = fields.iter().map(|s| s.as_str()).collect();
                    if !field_refs.is_empty() && field_refs[0] != "*" {
                        let field = field_refs[0];
                        let mut values: Vec<String> =
                            nodes.iter().map(|n| query::node_field(n, field)).collect();
                        values.sort();
                        values.dedup();
                        println!("{}", query::format_distinct(&values, field));
                        continue;
                    }
                }

                let start = std::time::Instant::now();
                let (nodes, fields) = query::execute_query(doc, q);
                let field_refs: Vec<&str> = fields.iter().map(|s| s.as_str()).collect();
                let table = query::format_results(&nodes, &field_refs, 60);
                let ms = start.elapsed().as_millis();

                println!("({} results, {}ms)", nodes.len(), ms);
                print!("{}", table);
            }
        }
    }

    println!("Goodbye.");
}

/// Print differences between two OVD documents.
fn print_diff(old: &OvdDocument, new: &OvdDocument) {
    println!();
    println!("── Diff ──");
    println!("  Old: {} ({} nodes)", truncate(&old.url, 50), old.nodes.len());
    println!("  New: {} ({} nodes)", truncate(&new.url, 50), new.nodes.len());
    println!();

    // Compare by node type counts
    let old_stats = old.stats();
    let new_stats = new.stats();

    let diffs = [
        ("Headings", old_stats.headings, new_stats.headings),
        ("Paragraphs", old_stats.paragraphs, new_stats.paragraphs),
        ("Links", old_stats.links, new_stats.links),
        ("Images", old_stats.images, new_stats.images),
        ("Tables", old_stats.tables, new_stats.tables),
        ("Lists", old_stats.lists, new_stats.lists),
        ("Forms", old_stats.forms, new_stats.forms),
        ("Code blocks", old_stats.code_blocks, new_stats.code_blocks),
    ];

    let mut any_diff = false;
    for (name, old_count, new_count) in &diffs {
        if old_count != new_count {
            let delta = *new_count as i64 - *old_count as i64;
            let sign = if delta > 0 { "+" } else { "" };
            println!("  {}: {} → {} ({}{})", name, old_count, new_count, sign, delta);
            any_diff = true;
        }
    }

    if old_stats.total_nodes != new_stats.total_nodes {
        let delta = new_stats.total_nodes as i64 - old_stats.total_nodes as i64;
        let sign = if delta > 0 { "+" } else { "" };
        println!("  Total: {} → {} ({}{})", old_stats.total_nodes, new_stats.total_nodes, sign, delta);
    }

    // Find new/removed links
    let old_links: std::collections::HashSet<String> = old
        .nodes
        .iter()
        .filter(|n| n.node_type == crate::ovd::NodeType::Link && !n.text.is_empty())
        .map(|n| n.text.clone())
        .collect();

    let new_links: std::collections::HashSet<String> = new
        .nodes
        .iter()
        .filter(|n| n.node_type == crate::ovd::NodeType::Link && !n.text.is_empty())
        .map(|n| n.text.clone())
        .collect();

    let added: Vec<&String> = new_links.difference(&old_links).collect();
    let removed: Vec<&String> = old_links.difference(&new_links).collect();

    if !added.is_empty() || !removed.is_empty() {
        println!();
        if !added.is_empty() {
            println!("  + {} new links:", added.len());
            for link in added.iter().take(10) {
                println!("    + {}", truncate(link, 70));
            }
            if added.len() > 10 {
                println!("    ... and {} more", added.len() - 10);
            }
        }
        if !removed.is_empty() {
            println!("  - {} removed links:", removed.len());
            for link in removed.iter().take(10) {
                println!("    - {}", truncate(link, 70));
            }
            if removed.len() > 10 {
                println!("    ... and {} more", removed.len() - 10);
            }
        }
    } else if !any_diff {
        println!("  No structural differences detected.");
    }

    // Find new/removed headings
    let old_headings: Vec<String> = old
        .nodes
        .iter()
        .filter(|n| n.node_type == crate::ovd::NodeType::Heading)
        .map(|n| n.text.clone())
        .collect();

    let new_headings: Vec<String> = new
        .nodes
        .iter()
        .filter(|n| n.node_type == crate::ovd::NodeType::Heading)
        .map(|n| n.text.clone())
        .collect();

    let old_set: std::collections::HashSet<&String> = old_headings.iter().collect();
    let new_set: std::collections::HashSet<&String> = new_headings.iter().collect();
    let added_h: Vec<&&String> = new_set.difference(&old_set).collect();
    let removed_h: Vec<&&String> = old_set.difference(&new_set).collect();

    if !added_h.is_empty() || !removed_h.is_empty() {
        println!();
        if !added_h.is_empty() {
            println!("  + {} new headings:", added_h.len());
            for h in added_h.iter().take(5) {
                println!("    + {}", truncate(h, 70));
            }
        }
        if !removed_h.is_empty() {
            println!("  - {} removed headings:", removed_h.len());
            for h in removed_h.iter().take(5) {
                println!("    - {}", truncate(h, 70));
            }
        }
    }

    println!();
}

fn print_help() {
    println!("── OctoView Shell Commands ──");
    println!();
    println!("  Navigation:");
    println!("    load <url>            Load a page (static fetch)");
    println!("    js <url>              Load a page with JS rendering");
    println!("    pages                 List loaded pages");
    println!("    use <id|alias>        Switch active page");
    println!("    drop [id]             Close active page (or by id)");
    println!();
    println!("  Queries:");
    println!("    SELECT ...            SQL-like query against active page");
    println!("    css <selector>        CSS selector query (a.class, div > h2, :bold)");
    println!("    css:<selector>        Inline CSS query (shorthand)");
    println!("    feed [selector]       LLM-optimized page summary (or filtered)");
    println!("    meta [url]            Structured metadata (JSON-LD, OG, Twitter)");
    println!("    links                 List all links on active page");
    println!("    headings              List all headings (with hierarchy)");
    println!("    images                List all images");
    println!("    info                  Show page statistics");
    println!("    fields                List queryable fields");
    println!();
    println!("  Export:");
    println!("    export <fmt> <file> <query>   Export to csv/json/jsonl");
    println!();
    println!("  Comparison:");
    println!("    diff <url>            Load page and diff against active");
    println!();
    println!("  Session (authenticated browsing):");
    println!("    session open <profile> Connect persistent Chrome profile");
    println!("    session close          Disconnect session");
    println!("    session export <file>  Export cookies to JSON");
    println!("    session import <file>  Import cookies from JSON");
    println!("    goto <url>             Navigate within session (preserves cookies)");
    println!("    cookies                Show session cookies");
    println!();
    println!("  Interaction (requires session):");
    println!("    type \"selector\" text   Type into a form field");
    println!("    click <selector>       Click an element");
    println!("    press <key>            Press a key (Enter, Tab, Escape)");
    println!("    screenshot [file.png]  Capture page screenshot");
    println!("    eval <javascript>      Execute JavaScript in page");
    println!();
    println!("  Automation:");
    println!("    run <script.ova>      Run an action script (multi-step workflow)");
    println!("    run <action>          Run a single inline action");
    println!();
    println!("  Other:");
    println!("    history               Show command history");
    println!("    clear                 Clear screen");
    println!("    help                  Show this help");
    println!("    quit                  Exit shell");
    println!();
    println!("  Query Examples:");
    println!("    SELECT text, href FROM nodes WHERE type = 'link'");
    println!("    SELECT * FROM nodes WHERE text LIKE '%GPU%' LIMIT 10");
    println!("    SELECT DISTINCT type FROM nodes");
    println!("    SELECT COUNT(*) FROM nodes WHERE type = 'heading'");
    println!("    SELECT text, font_size FROM nodes WHERE font_size > 16 ORDER BY font_size DESC");
    println!();
    println!("  CSS Selector Examples:");
    println!("    css a.nav-link                 Links with class 'nav-link'");
    println!("    css div.card > h2              H2 headings inside cards");
    println!("    css [href*=github]             Links containing 'github'");
    println!("    css :text(price):bold           Bold nodes containing 'price'");
    println!("    css h1, h2, h3                 All major headings");
    println!("    css .featured:has-src           Featured elements with images");
    println!();
}

/// Convert a URL to a short alias for the prompt.
fn url_to_alias(url: &str) -> String {
    let url = url
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_start_matches("www.")
        .trim_start_matches("file://");

    // Take the domain (before first /)
    let domain = url.split('/').next().unwrap_or(url);

    // Take the first part of the domain (before .)
    let short = domain.split('.').next().unwrap_or(domain);

    // Limit length
    if short.len() > 12 {
        short[..12].to_string()
    } else {
        short.to_string()
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", &s[..max.saturating_sub(3)])
    } else {
        s.to_string()
    }
}
