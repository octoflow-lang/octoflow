//! Page Actions — scriptable web automation sequences.
//!
//! Actions let you define multi-step workflows:
//!   navigate → type → click → wait → screenshot → extract
//!
//! Action scripts are plain text files (.ova) with one command per line.
//! This turns OctoView from "read the web" into "use the web."
//!
//! Format:
//!   # Comment
//!   navigate <url>
//!   wait <selector> [timeout_ms]
//!   type <selector> <text>
//!   click <selector>
//!   press <key>
//!   sleep <ms>
//!   screenshot <file.png>
//!   extract [css-selector]
//!   feed
//!   eval <javascript>
//!   assert <selector>
//!   print <message>
//!   scroll [pixels]              — scroll down (default: full page)
//!   back                         — browser back
//!   forward                      — browser forward
//!   reload                       — reload page
//!   save <file>                  — save last output to file
//!   title                        — print page title
//!   url                          — print current URL
//!   source <file>                — save page HTML source to file

use crate::extract;
use crate::ovd::OvdDocument;
use crate::selector;
use crate::session::BrowserSession;

/// A single action step.
#[derive(Debug, Clone)]
pub enum Action {
    Navigate(String),
    Wait(String, u64),          // selector, timeout_ms
    Type(String, String),       // selector, text
    Click(String),              // selector
    Press(String),              // key name
    Sleep(u64),                 // milliseconds
    Screenshot(String),         // output file
    Extract(Option<String>),    // optional CSS selector
    Feed,                       // full page feed
    Eval(String),               // JavaScript code
    Assert(String),             // selector (fail if not found)
    Print(String),              // message to print
    Scroll(Option<u32>),        // pixels (None = full page)
    Back,                       // browser history back
    Forward,                    // browser history forward
    Reload,                     // reload current page
    Save(String),               // save last output to file
    Title,                      // print page title
    Url,                        // print current URL
    Source(String),             // save HTML source to file
}

/// Result from running an action sequence.
pub struct ActionResult {
    pub steps_run: usize,
    pub steps_total: usize,
    pub documents: Vec<OvdDocument>,
    pub outputs: Vec<String>,
    pub errors: Vec<String>,
}

/// Parse an action script from text.
pub fn parse_actions(script: &str) -> Vec<Action> {
    let mut actions = Vec::new();

    for line in script.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        let cmd = parts[0].to_lowercase();
        let args = if parts.len() > 1 { parts[1].trim() } else { "" };

        match cmd.as_str() {
            "navigate" | "goto" | "open" => {
                if !args.is_empty() {
                    actions.push(Action::Navigate(args.to_string()));
                }
            }
            "wait" => {
                let wait_parts: Vec<&str> = args.rsplitn(2, ' ').collect();
                if wait_parts.len() == 2 {
                    if let Ok(timeout) = wait_parts[0].parse::<u64>() {
                        actions.push(Action::Wait(wait_parts[1].to_string(), timeout));
                    } else {
                        actions.push(Action::Wait(args.to_string(), 10000));
                    }
                } else if !args.is_empty() {
                    actions.push(Action::Wait(args.to_string(), 10000));
                }
            }
            "type" => {
                // type "selector" text to type
                if let Some((sel, text)) = parse_selector_and_text(args) {
                    actions.push(Action::Type(sel, text));
                }
            }
            "click" => {
                if !args.is_empty() {
                    actions.push(Action::Click(args.to_string()));
                }
            }
            "press" => {
                if !args.is_empty() {
                    actions.push(Action::Press(args.to_string()));
                }
            }
            "sleep" | "delay" => {
                if let Ok(ms) = args.parse::<u64>() {
                    actions.push(Action::Sleep(ms));
                }
            }
            "screenshot" | "snap" => {
                let path = if args.is_empty() { "screenshot.png" } else { args };
                actions.push(Action::Screenshot(path.to_string()));
            }
            "extract" | "select" => {
                if args.is_empty() {
                    actions.push(Action::Extract(None));
                } else {
                    actions.push(Action::Extract(Some(args.to_string())));
                }
            }
            "feed" => {
                actions.push(Action::Feed);
            }
            "eval" | "js" => {
                if !args.is_empty() {
                    actions.push(Action::Eval(args.to_string()));
                }
            }
            "assert" => {
                if !args.is_empty() {
                    actions.push(Action::Assert(args.to_string()));
                }
            }
            "print" | "echo" => {
                actions.push(Action::Print(args.to_string()));
            }
            "scroll" => {
                if args.is_empty() {
                    actions.push(Action::Scroll(None));
                } else if let Ok(px) = args.parse::<u32>() {
                    actions.push(Action::Scroll(Some(px)));
                } else {
                    actions.push(Action::Scroll(None));
                }
            }
            "back" => {
                actions.push(Action::Back);
            }
            "forward" => {
                actions.push(Action::Forward);
            }
            "reload" | "refresh" => {
                actions.push(Action::Reload);
            }
            "save" | "write" => {
                if !args.is_empty() {
                    actions.push(Action::Save(args.to_string()));
                }
            }
            "title" => {
                actions.push(Action::Title);
            }
            "url" => {
                actions.push(Action::Url);
            }
            "source" | "html" => {
                let path = if args.is_empty() { "source.html" } else { args };
                actions.push(Action::Source(path.to_string()));
            }
            _ => {
                // Unknown command — skip
            }
        }
    }

    actions
}

/// Parse an action script from a file.
pub fn parse_action_file(path: &str) -> Result<Vec<Action>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read action file '{}': {}", path, e))?;
    Ok(parse_actions(&content))
}

/// Execute an action sequence against a browser session.
pub fn run_actions(session: &BrowserSession, actions: &[Action]) -> ActionResult {
    let mut result = ActionResult {
        steps_run: 0,
        steps_total: actions.len(),
        documents: Vec::new(),
        outputs: Vec::new(),
        errors: Vec::new(),
    };

    for (i, action) in actions.iter().enumerate() {
        eprintln!("[action {}/{}] {:?}", i + 1, actions.len(), action_summary(action));
        result.steps_run = i + 1;

        match action {
            Action::Navigate(url) => {
                match session.navigate(url) {
                    Ok(doc) => {
                        let msg = format!("Navigated: {} ({} nodes)", truncate(url, 50), doc.nodes.len());
                        result.outputs.push(msg.clone());
                        eprintln!("  {}", msg);
                        result.documents.push(doc);
                    }
                    Err(e) => {
                        result.errors.push(format!("Navigate error: {}", e));
                        eprintln!("  ERROR: {}", e);
                    }
                }
            }
            Action::Wait(selector, timeout_ms) => {
                match session.wait_for(selector, *timeout_ms) {
                    Ok(()) => {
                        eprintln!("  Found '{}'", selector);
                    }
                    Err(e) => {
                        result.errors.push(format!("Wait timeout: {}", e));
                        eprintln!("  TIMEOUT: {}", e);
                    }
                }
            }
            Action::Type(selector, text) => {
                match session.type_into(selector, text) {
                    Ok(()) => eprintln!("  Typed into '{}'", selector),
                    Err(e) => {
                        result.errors.push(format!("Type error: {}", e));
                        eprintln!("  ERROR: {}", e);
                    }
                }
            }
            Action::Click(selector) => {
                match session.click(selector) {
                    Ok(()) => eprintln!("  Clicked '{}'", selector),
                    Err(e) => {
                        result.errors.push(format!("Click error: {}", e));
                        eprintln!("  ERROR: {}", e);
                    }
                }
            }
            Action::Press(key) => {
                match session.press_key(key) {
                    Ok(()) => eprintln!("  Pressed '{}'", key),
                    Err(e) => {
                        result.errors.push(format!("Key error: {}", e));
                        eprintln!("  ERROR: {}", e);
                    }
                }
            }
            Action::Sleep(ms) => {
                eprintln!("  Sleeping {}ms...", ms);
                std::thread::sleep(std::time::Duration::from_millis(*ms));
            }
            Action::Screenshot(path) => {
                match session.screenshot_to_file(path) {
                    Ok(()) => {
                        let msg = format!("Screenshot: {}", path);
                        result.outputs.push(msg.clone());
                        eprintln!("  {}", msg);
                    }
                    Err(e) => {
                        result.errors.push(format!("Screenshot error: {}", e));
                        eprintln!("  ERROR: {}", e);
                    }
                }
            }
            Action::Extract(selector) => {
                match session.get_html() {
                    Ok(html) => {
                        let url = session.current_url();
                        let doc = extract::extract_html(&html, &url);

                        if let Some(sel) = selector {
                            let nodes = selector::select(&doc, sel);
                            let feed = selector::format_feed_nodes(&nodes, sel);
                            result.outputs.push(feed.clone());
                            print!("{}", feed);
                        } else {
                            let feed = selector::format_feed(&doc);
                            result.outputs.push(feed.clone());
                            print!("{}", feed);
                        }

                        result.documents.push(doc);
                    }
                    Err(e) => {
                        result.errors.push(format!("Extract error: {}", e));
                        eprintln!("  ERROR: {}", e);
                    }
                }
            }
            Action::Feed => {
                match session.get_html() {
                    Ok(html) => {
                        let url = session.current_url();
                        let doc = extract::extract_html(&html, &url);
                        let feed = selector::format_feed(&doc);
                        result.outputs.push(feed.clone());
                        print!("{}", feed);
                        result.documents.push(doc);
                    }
                    Err(e) => {
                        result.errors.push(format!("Feed error: {}", e));
                        eprintln!("  ERROR: {}", e);
                    }
                }
            }
            Action::Eval(js) => {
                match session.eval(js) {
                    Ok(val) => {
                        result.outputs.push(val.clone());
                        println!("{}", val);
                    }
                    Err(e) => {
                        result.errors.push(format!("Eval error: {}", e));
                        eprintln!("  ERROR: {}", e);
                    }
                }
            }
            Action::Assert(selector) => {
                match session.eval(&format!("document.querySelector('{}') !== null", selector)) {
                    Ok(val) if val.contains("true") => {
                        eprintln!("  PASS: '{}' found", selector);
                    }
                    _ => {
                        let msg = format!("ASSERT FAILED: '{}' not found", selector);
                        result.errors.push(msg.clone());
                        eprintln!("  {}", msg);
                        // Don't stop on assert failure — continue
                    }
                }
            }
            Action::Print(msg) => {
                println!("{}", msg);
                result.outputs.push(msg.clone());
            }
            Action::Scroll(pixels) => {
                let js = match pixels {
                    Some(px) => format!("window.scrollBy(0, {}); window.scrollY", px),
                    None => "window.scrollTo(0, document.body.scrollHeight); document.body.scrollHeight".to_string(),
                };
                match session.eval(&js) {
                    Ok(val) => eprintln!("  Scrolled (pos: {})", val),
                    Err(e) => {
                        result.errors.push(format!("Scroll error: {}", e));
                        eprintln!("  ERROR: {}", e);
                    }
                }
            }
            Action::Back => {
                match session.eval("history.back(); 'ok'") {
                    Ok(_) => {
                        std::thread::sleep(std::time::Duration::from_millis(500));
                        eprintln!("  Back → {}", session.current_url());
                    }
                    Err(e) => {
                        result.errors.push(format!("Back error: {}", e));
                        eprintln!("  ERROR: {}", e);
                    }
                }
            }
            Action::Forward => {
                match session.eval("history.forward(); 'ok'") {
                    Ok(_) => {
                        std::thread::sleep(std::time::Duration::from_millis(500));
                        eprintln!("  Forward → {}", session.current_url());
                    }
                    Err(e) => {
                        result.errors.push(format!("Forward error: {}", e));
                        eprintln!("  ERROR: {}", e);
                    }
                }
            }
            Action::Reload => {
                match session.eval("location.reload(); 'ok'") {
                    Ok(_) => {
                        std::thread::sleep(std::time::Duration::from_millis(1000));
                        eprintln!("  Reloaded");
                    }
                    Err(e) => {
                        result.errors.push(format!("Reload error: {}", e));
                        eprintln!("  ERROR: {}", e);
                    }
                }
            }
            Action::Save(path) => {
                if let Some(last) = result.outputs.last() {
                    match std::fs::write(path, last) {
                        Ok(()) => {
                            eprintln!("  Saved {} bytes to {}", last.len(), path);
                        }
                        Err(e) => {
                            result.errors.push(format!("Save error: {}", e));
                            eprintln!("  ERROR: {}", e);
                        }
                    }
                } else {
                    eprintln!("  Nothing to save (no previous output)");
                }
            }
            Action::Title => {
                match session.eval("document.title") {
                    Ok(title) => {
                        println!("{}", title);
                        result.outputs.push(title);
                    }
                    Err(e) => {
                        result.errors.push(format!("Title error: {}", e));
                        eprintln!("  ERROR: {}", e);
                    }
                }
            }
            Action::Url => {
                let url = session.current_url();
                println!("{}", url);
                result.outputs.push(url);
            }
            Action::Source(path) => {
                match session.get_html() {
                    Ok(html) => {
                        match std::fs::write(path, &html) {
                            Ok(()) => {
                                let msg = format!("Source saved: {} ({} bytes)", path, html.len());
                                result.outputs.push(msg.clone());
                                eprintln!("  {}", msg);
                            }
                            Err(e) => {
                                result.errors.push(format!("Source save error: {}", e));
                                eprintln!("  ERROR: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        result.errors.push(format!("Source error: {}", e));
                        eprintln!("  ERROR: {}", e);
                    }
                }
            }
        }
    }

    eprintln!();
    eprintln!(
        "[actions] Complete: {}/{} steps, {} errors",
        result.steps_run,
        result.steps_total,
        result.errors.len()
    );

    result
}

/// Parse "selector" text from type command args.
fn parse_selector_and_text(args: &str) -> Option<(String, String)> {
    let args = args.trim();

    // Handle quoted selector: type "input#search" hello world
    if args.starts_with('"') {
        if let Some(end) = args[1..].find('"') {
            let sel = args[1..end + 1].to_string();
            let text = args[end + 2..].trim().to_string();
            return Some((sel, text));
        }
    }
    if args.starts_with('\'') {
        if let Some(end) = args[1..].find('\'') {
            let sel = args[1..end + 1].to_string();
            let text = args[end + 2..].trim().to_string();
            return Some((sel, text));
        }
    }

    // Unquoted: type selector text
    let parts: Vec<&str> = args.splitn(2, ' ').collect();
    if parts.len() == 2 {
        Some((parts[0].to_string(), parts[1].to_string()))
    } else {
        None
    }
}

fn action_summary(action: &Action) -> String {
    match action {
        Action::Navigate(url) => format!("navigate {}", truncate(url, 50)),
        Action::Wait(sel, ms) => format!("wait {} ({}ms)", sel, ms),
        Action::Type(sel, _) => format!("type into {}", sel),
        Action::Click(sel) => format!("click {}", sel),
        Action::Press(key) => format!("press {}", key),
        Action::Sleep(ms) => format!("sleep {}ms", ms),
        Action::Screenshot(path) => format!("screenshot {}", path),
        Action::Extract(sel) => format!("extract {:?}", sel),
        Action::Feed => "feed".to_string(),
        Action::Eval(js) => format!("eval {}", truncate(js, 30)),
        Action::Assert(sel) => format!("assert {}", sel),
        Action::Print(msg) => format!("print {}", truncate(msg, 30)),
        Action::Scroll(px) => match px {
            Some(n) => format!("scroll {}px", n),
            None => "scroll (full page)".to_string(),
        },
        Action::Back => "back".to_string(),
        Action::Forward => "forward".to_string(),
        Action::Reload => "reload".to_string(),
        Action::Save(path) => format!("save {}", path),
        Action::Title => "title".to_string(),
        Action::Url => "url".to_string(),
        Action::Source(path) => format!("source {}", path),
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", &s[..max.saturating_sub(3)])
    } else {
        s.to_string()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_actions() {
        let script = r#"
# YouTube search workflow
navigate https://www.youtube.com
wait input#search
type "input#search" rust programming
press Enter
sleep 2000
extract a#video-title
screenshot results.png
"#;
        let actions = parse_actions(script);
        assert_eq!(actions.len(), 7);

        assert!(matches!(actions[0], Action::Navigate(_)));
        assert!(matches!(actions[1], Action::Wait(_, _)));
        assert!(matches!(actions[2], Action::Type(_, _)));
        assert!(matches!(actions[3], Action::Press(_)));
        assert!(matches!(actions[4], Action::Sleep(2000)));
        assert!(matches!(actions[5], Action::Extract(Some(_))));
        assert!(matches!(actions[6], Action::Screenshot(_)));
    }

    #[test]
    fn test_parse_type_quoted_selector() {
        let script = "type \"input#email\" user@example.com";
        let actions = parse_actions(script);
        assert_eq!(actions.len(), 1);
        if let Action::Type(sel, text) = &actions[0] {
            assert_eq!(sel, "input#email");
            assert_eq!(text, "user@example.com");
        } else {
            panic!("Expected Type action");
        }
    }

    #[test]
    fn test_parse_comments_and_blanks() {
        let script = r#"
# This is a comment

navigate https://example.com
# Another comment
click button.submit

"#;
        let actions = parse_actions(script);
        assert_eq!(actions.len(), 2);
    }

    #[test]
    fn test_parse_action_file_content() {
        let script = "print Hello World\nfeed\nassert body";
        let actions = parse_actions(script);
        assert_eq!(actions.len(), 3);
        assert!(matches!(actions[0], Action::Print(_)));
        assert!(matches!(actions[1], Action::Feed));
        assert!(matches!(actions[2], Action::Assert(_)));
    }

    #[test]
    fn test_parse_scroll_variants() {
        let script = "scroll\nscroll 500\nscroll full";
        let actions = parse_actions(script);
        assert_eq!(actions.len(), 3);
        assert!(matches!(actions[0], Action::Scroll(None)));
        assert!(matches!(actions[1], Action::Scroll(Some(500))));
        assert!(matches!(actions[2], Action::Scroll(None))); // "full" not a number → default
    }

    #[test]
    fn test_parse_navigation_actions() {
        let script = "back\nforward\nreload\nrefresh\ntitle\nurl";
        let actions = parse_actions(script);
        assert_eq!(actions.len(), 6);
        assert!(matches!(actions[0], Action::Back));
        assert!(matches!(actions[1], Action::Forward));
        assert!(matches!(actions[2], Action::Reload));
        assert!(matches!(actions[3], Action::Reload));
        assert!(matches!(actions[4], Action::Title));
        assert!(matches!(actions[5], Action::Url));
    }

    #[test]
    fn test_parse_save_and_source() {
        let script = "save output.txt\nsource page.html\nhtml dump.html";
        let actions = parse_actions(script);
        assert_eq!(actions.len(), 3);
        assert!(matches!(actions[0], Action::Save(_)));
        assert!(matches!(actions[1], Action::Source(_)));
        assert!(matches!(actions[2], Action::Source(_)));
        if let Action::Save(path) = &actions[0] {
            assert_eq!(path, "output.txt");
        }
    }

    #[test]
    fn test_parse_full_workflow() {
        let script = r#"
# Full Gmail-style workflow
navigate https://mail.google.com
wait input[type=email] 5000
type "input[type=email]" user@gmail.com
press Enter
sleep 2000
wait input[type=password] 5000
type "input[type=password]" mypassword
press Enter
sleep 3000
scroll 800
screenshot inbox.png
extract .mail-item
save inbox-data.txt
title
url
"#;
        let actions = parse_actions(script);
        assert_eq!(actions.len(), 15);
        assert!(matches!(actions[9], Action::Scroll(Some(800))));
        assert!(matches!(actions[12], Action::Save(_)));
        assert!(matches!(actions[14], Action::Url));
    }
}
