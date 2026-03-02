//! Watch mode — monitor URLs for changes at intervals.
//!
//! Periodically fetches a URL, compares against the previous snapshot,
//! and reports structural changes. Supports:
//!   - Configurable interval (seconds)
//!   - Optional query filter (only watch specific node types)
//!   - Export on change (csv/json/jsonl)
//!   - Change summary with link/heading/content diffs

use crate::ovd::{NodeType, OvdDocument};
use std::collections::HashSet;
use std::time::Duration;

/// Change report from a single watch cycle.
pub struct ChangeReport {
    pub timestamp: String,
    pub old_nodes: usize,
    pub new_nodes: usize,
    pub added_links: Vec<(String, String)>,
    pub removed_links: Vec<(String, String)>,
    pub added_headings: Vec<String>,
    pub removed_headings: Vec<String>,
    pub added_paragraphs: Vec<String>,
    pub removed_paragraphs: Vec<String>,
    pub has_changes: bool,
}

impl ChangeReport {
    pub fn total_changes(&self) -> usize {
        self.added_links.len()
            + self.removed_links.len()
            + self.added_headings.len()
            + self.removed_headings.len()
            + self.added_paragraphs.len()
            + self.removed_paragraphs.len()
    }
}

/// Compare two documents and produce a change report.
pub fn compute_diff(old: &OvdDocument, new: &OvdDocument) -> ChangeReport {
    let now = chrono_now();

    // Link diff
    let old_links: HashSet<(String, String)> = old
        .nodes
        .iter()
        .filter(|n| n.node_type == NodeType::Link && !n.text.is_empty())
        .map(|n| (n.text.clone(), n.href.clone()))
        .collect();
    let new_links: HashSet<(String, String)> = new
        .nodes
        .iter()
        .filter(|n| n.node_type == NodeType::Link && !n.text.is_empty())
        .map(|n| (n.text.clone(), n.href.clone()))
        .collect();

    let added_links: Vec<(String, String)> = new_links.difference(&old_links).cloned().collect();
    let removed_links: Vec<(String, String)> = old_links.difference(&new_links).cloned().collect();

    // Heading diff
    let old_headings: HashSet<String> = old
        .nodes
        .iter()
        .filter(|n| n.node_type == NodeType::Heading)
        .map(|n| n.text.clone())
        .collect();
    let new_headings: HashSet<String> = new
        .nodes
        .iter()
        .filter(|n| n.node_type == NodeType::Heading)
        .map(|n| n.text.clone())
        .collect();

    let added_headings: Vec<String> = new_headings.difference(&old_headings).cloned().collect();
    let removed_headings: Vec<String> = old_headings.difference(&new_headings).cloned().collect();

    // Paragraph diff (only substantial paragraphs)
    let old_paras: HashSet<String> = old
        .nodes
        .iter()
        .filter(|n| n.node_type == NodeType::Paragraph && n.text.len() > 20)
        .map(|n| n.text.clone())
        .collect();
    let new_paras: HashSet<String> = new
        .nodes
        .iter()
        .filter(|n| n.node_type == NodeType::Paragraph && n.text.len() > 20)
        .map(|n| n.text.clone())
        .collect();

    let added_paragraphs: Vec<String> = new_paras.difference(&old_paras).cloned().collect();
    let removed_paragraphs: Vec<String> = old_paras.difference(&new_paras).cloned().collect();

    let has_changes = !added_links.is_empty()
        || !removed_links.is_empty()
        || !added_headings.is_empty()
        || !removed_headings.is_empty()
        || !added_paragraphs.is_empty()
        || !removed_paragraphs.is_empty()
        || old.nodes.len() != new.nodes.len();

    ChangeReport {
        timestamp: now,
        old_nodes: old.nodes.len(),
        new_nodes: new.nodes.len(),
        added_links,
        removed_links,
        added_headings,
        removed_headings,
        added_paragraphs,
        removed_paragraphs,
        has_changes,
    }
}

/// Format a change report for terminal display.
pub fn format_report(report: &ChangeReport, url: &str) -> String {
    let mut out = String::new();

    if !report.has_changes {
        out.push_str(&format!(
            "[{}] {} — no changes ({} nodes)\n",
            report.timestamp,
            truncate(url, 50),
            report.new_nodes
        ));
        return out;
    }

    let node_delta = report.new_nodes as i64 - report.old_nodes as i64;
    let sign = if node_delta >= 0 { "+" } else { "" };

    out.push_str(&format!(
        "[{}] {} — {} changes (nodes: {} → {}, {}{})\n",
        report.timestamp,
        truncate(url, 50),
        report.total_changes(),
        report.old_nodes,
        report.new_nodes,
        sign,
        node_delta
    ));

    // Links
    for (text, href) in report.added_links.iter().take(10) {
        out.push_str(&format!("  + link: {} → {}\n", truncate(text, 40), truncate(href, 50)));
    }
    if report.added_links.len() > 10 {
        out.push_str(&format!("  ... +{} more links\n", report.added_links.len() - 10));
    }
    for (text, href) in report.removed_links.iter().take(5) {
        out.push_str(&format!("  - link: {} → {}\n", truncate(text, 40), truncate(href, 50)));
    }
    if report.removed_links.len() > 5 {
        out.push_str(&format!("  ... -{} more links\n", report.removed_links.len() - 5));
    }

    // Headings
    for h in report.added_headings.iter().take(5) {
        out.push_str(&format!("  + heading: {}\n", truncate(h, 70)));
    }
    for h in report.removed_headings.iter().take(5) {
        out.push_str(&format!("  - heading: {}\n", truncate(h, 70)));
    }

    // Paragraphs
    for p in report.added_paragraphs.iter().take(3) {
        out.push_str(&format!("  + content: {}\n", truncate(p, 70)));
    }
    if report.added_paragraphs.len() > 3 {
        out.push_str(&format!(
            "  ... +{} more paragraphs\n",
            report.added_paragraphs.len() - 3
        ));
    }
    for p in report.removed_paragraphs.iter().take(3) {
        out.push_str(&format!("  - content: {}\n", truncate(p, 70)));
    }

    out
}

/// Format a change report as JSONL (one JSON object per report).
pub fn format_report_jsonl(report: &ChangeReport, url: &str) -> String {
    let mut out = String::from("{");
    out.push_str(&format!("\"timestamp\":\"{}\",", report.timestamp));
    out.push_str(&format!("\"url\":\"{}\",", json_escape(url)));
    out.push_str(&format!("\"has_changes\":{},", report.has_changes));
    out.push_str(&format!("\"old_nodes\":{},", report.old_nodes));
    out.push_str(&format!("\"new_nodes\":{},", report.new_nodes));
    out.push_str(&format!("\"total_changes\":{},", report.total_changes()));
    out.push_str(&format!("\"added_links\":{},", report.added_links.len()));
    out.push_str(&format!("\"removed_links\":{},", report.removed_links.len()));
    out.push_str(&format!("\"added_headings\":{},", report.added_headings.len()));
    out.push_str(&format!("\"removed_headings\":{}", report.removed_headings.len()));
    out.push_str("}\n");
    out
}

/// Run the watch loop.
pub fn run_watch(
    url: &str,
    interval_secs: u64,
    max_checks: Option<usize>,
    only_changes: bool,
    output_file: Option<&str>,
    jsonl: bool,
    load_fn: impl Fn(&str) -> Result<OvdDocument, String>,
) {
    eprintln!(
        "Watching {} every {}s{}",
        url,
        interval_secs,
        if let Some(n) = max_checks {
            format!(" (max {} checks)", n)
        } else {
            " (Ctrl+C to stop)".to_string()
        }
    );

    // Initial fetch
    let mut prev_doc = match load_fn(url) {
        Ok(doc) => {
            eprintln!(
                "[{}] Initial: {} nodes from {}",
                chrono_now(),
                doc.nodes.len(),
                truncate(url, 50)
            );
            doc
        }
        Err(e) => {
            eprintln!("Error on initial fetch: {}", e);
            return;
        }
    };

    let interval = Duration::from_secs(interval_secs);
    let mut check_count: usize = 0;
    let mut change_count: usize = 0;

    // Open output file for appending if specified
    let mut output_writer: Option<std::fs::File> = output_file.map(|path| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap_or_else(|e| {
                eprintln!("Error opening output file '{}': {}", path, e);
                std::process::exit(1);
            })
    });

    loop {
        std::thread::sleep(interval);
        check_count += 1;

        if let Some(max) = max_checks {
            if check_count > max {
                break;
            }
        }

        // Fetch fresh version
        let new_doc = match load_fn(url) {
            Ok(doc) => doc,
            Err(e) => {
                eprintln!("[{}] Error: {}", chrono_now(), e);
                continue;
            }
        };

        // Compute diff
        let report = compute_diff(&prev_doc, &new_doc);

        if report.has_changes {
            change_count += 1;
        }

        // Output
        if !only_changes || report.has_changes {
            if jsonl {
                let line = format_report_jsonl(&report, url);
                print!("{}", line);
                if let Some(ref mut f) = output_writer {
                    use std::io::Write;
                    let _ = f.write_all(line.as_bytes());
                }
            } else {
                let text = format_report(&report, url);
                print!("{}", text);
                if let Some(ref mut f) = output_writer {
                    use std::io::Write;
                    let _ = f.write_all(text.as_bytes());
                }
            }
        }

        // Update previous snapshot
        prev_doc = new_doc;
    }

    eprintln!();
    eprintln!(
        "Watch complete: {} checks, {} changes detected",
        check_count, change_count
    );
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", &s[..max.saturating_sub(3)])
    } else {
        s.to_string()
    }
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// Get current time as HH:MM:SS string.
fn chrono_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Simple time formatting without chrono dependency
    let hours = (secs % 86400) / 3600;
    let mins = (secs % 3600) / 60;
    let s = secs % 60;
    format!("{:02}:{:02}:{:02}", hours, mins, s)
}
