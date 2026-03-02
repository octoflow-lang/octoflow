//! Query interface — SQL-like queries over OctoView Documents.
//!
//! Supports:
//!   SELECT field1, field2 FROM nodes WHERE conditions ORDER BY field LIMIT N
//!   SELECT COUNT(*) FROM nodes WHERE conditions
//!   SELECT DISTINCT field FROM nodes WHERE conditions
//!   Conditions: =, !=, <, >, <=, >=, LIKE, contains, AND, OR

use crate::ovd::{OvdDocument, OvdNode};

// ─── Query Execution ────────────────────────────────────────────────────────

/// Parsed query structure.
#[allow(dead_code)]
pub struct ParsedQuery {
    pub fields: Vec<String>,
    pub where_clause: String,
    pub order_by: Option<(String, SortDir)>,
    pub limit: Option<usize>,
    pub count: bool,
    pub distinct: bool,
}

#[derive(Clone, Copy)]
pub enum SortDir {
    Asc,
    Desc,
}

/// Parse and execute a SQL-like query against an OVD document.
pub fn execute_query<'a>(doc: &'a OvdDocument, query: &str) -> (Vec<&'a OvdNode>, Vec<String>) {
    let parsed = parse_query(query);

    // Execute WHERE clause
    let mut nodes: Vec<&OvdNode> = if parsed.where_clause.is_empty() {
        doc.nodes.iter().collect()
    } else {
        doc.query(&parsed.where_clause)
    };

    // Apply ORDER BY
    if let Some((ref field, dir)) = parsed.order_by {
        sort_nodes(&mut nodes, field, dir);
    }

    // Apply LIMIT
    if let Some(limit) = parsed.limit {
        nodes.truncate(limit);
    }

    (nodes, parsed.fields)
}

/// Parse a query string into structured components.
fn parse_query(query: &str) -> ParsedQuery {
    let query = query.trim();
    let upper = query.to_uppercase();

    let mut fields_str = String::new();
    let mut where_clause = String::new();
    let mut order_by: Option<(String, SortDir)> = None;
    let mut limit: Option<usize> = None;
    let mut count = false;
    let mut distinct = false;

    // Extract LIMIT N (from the end)
    let (query_no_limit, extracted_limit) = extract_suffix_clause(&upper, query, "LIMIT ");
    if let Some(limit_str) = extracted_limit {
        limit = limit_str.trim().parse().ok();
    }
    let query = &query_no_limit;
    let upper = query.to_uppercase();

    // Extract ORDER BY field [ASC|DESC]
    let (query_no_order, extracted_order) = extract_suffix_clause(&upper, query, "ORDER BY ");
    if let Some(order_str) = extracted_order {
        let parts: Vec<&str> = order_str.trim().split_whitespace().collect();
        if !parts.is_empty() {
            let field = parts[0].to_lowercase();
            let dir = if parts.len() > 1 && parts[1].eq_ignore_ascii_case("DESC") {
                SortDir::Desc
            } else {
                SortDir::Asc
            };
            order_by = Some((field, dir));
        }
    }
    let query = &query_no_order;
    let upper = query.to_uppercase();

    // Parse SELECT ... FROM nodes WHERE ...
    if upper.starts_with("SELECT ") {
        let rest = &query[7..]; // skip "SELECT "
        let rest_upper = rest.to_uppercase();

        // SELECT COUNT(*)
        if rest_upper.starts_with("COUNT(*)") {
            count = true;
            fields_str = "count".to_string();
            // Extract WHERE clause if present
            if let Some(where_idx) = rest_upper.find(" WHERE ") {
                where_clause = rest[where_idx + 7..].trim().to_string();
            }
        }
        // SELECT DISTINCT field
        else if rest_upper.starts_with("DISTINCT ") {
            distinct = true;
            let after_distinct = &rest[9..];
            let after_upper = after_distinct.to_uppercase();

            if let Some(from_idx) = after_upper.find(" FROM ") {
                fields_str = after_distinct[..from_idx].trim().to_string();
                let after_from = &after_distinct[from_idx + 6..];
                if let Some(where_idx) = after_from.to_uppercase().find(" WHERE ") {
                    where_clause = after_from[where_idx + 7..].trim().to_string();
                }
            } else if let Some(where_idx) = after_upper.find(" WHERE ") {
                fields_str = after_distinct[..where_idx].trim().to_string();
                where_clause = after_distinct[where_idx + 7..].trim().to_string();
            } else {
                fields_str = after_distinct.trim().to_string();
            }
        }
        // Normal SELECT
        else {
            let from_pos = rest_upper.find(" FROM ");
            let where_pos = rest_upper.find(" WHERE ");

            if let Some(from_idx) = from_pos {
                fields_str = rest[..from_idx].trim().to_string();
                let after_from = &rest[from_idx + 6..];
                if let Some(where_idx) = after_from.to_uppercase().find(" WHERE ") {
                    where_clause = after_from[where_idx + 7..].trim().to_string();
                }
            } else if let Some(where_idx) = where_pos {
                fields_str = rest[..where_idx].trim().to_string();
                where_clause = rest[where_idx + 7..].trim().to_string();
            } else {
                fields_str = rest.trim().to_string();
            }
        }
    }
    // WHERE ... (no SELECT)
    else if upper.starts_with("WHERE ") {
        fields_str = "*".to_string();
        where_clause = query[6..].trim().to_string();
    }
    // field1, field2 WHERE ...
    else if let Some(where_idx) = upper.find(" WHERE ") {
        fields_str = query[..where_idx].trim().to_string();
        where_clause = query[where_idx + 7..].trim().to_string();
    }
    // Just a predicate
    else if query.contains('=') || query.contains('<') || query.contains('>')
        || query.to_lowercase().contains("contains") || query.to_lowercase().contains("like")
    {
        fields_str = "*".to_string();
        where_clause = query.to_string();
    }

    // Parse field list
    let fields: Vec<String> = if fields_str.is_empty() || fields_str == "*" {
        vec!["*".to_string()]
    } else {
        fields_str
            .split(',')
            .map(|f| f.trim().to_string())
            .collect()
    };

    // Remove any trailing ORDER BY / LIMIT that leaked into where_clause
    let wc_upper = where_clause.to_uppercase();
    if let Some(idx) = wc_upper.find(" ORDER BY ") {
        where_clause = where_clause[..idx].trim().to_string();
    }
    if let Some(idx) = wc_upper.find(" LIMIT ") {
        where_clause = where_clause[..idx].trim().to_string();
    }

    ParsedQuery {
        fields,
        where_clause,
        order_by,
        limit,
        count,
        distinct,
    }
}

/// Extract a suffix clause (LIMIT, ORDER BY) from the query.
/// Returns (query_without_suffix, extracted_value).
fn extract_suffix_clause(upper: &str, original: &str, keyword: &str) -> (String, Option<String>) {
    if let Some(idx) = upper.rfind(keyword) {
        let value = original[idx + keyword.len()..].to_string();
        let remaining = original[..idx].trim().to_string();
        (remaining, Some(value))
    } else {
        (original.to_string(), None)
    }
}

/// Sort nodes by a field.
fn sort_nodes(nodes: &mut [&OvdNode], field: &str, dir: SortDir) {
    nodes.sort_by(|a, b| {
        let va = node_field(a, field);
        let vb = node_field(b, field);

        // Try numeric comparison first
        let cmp = match (va.parse::<f64>(), vb.parse::<f64>()) {
            (Ok(na), Ok(nb)) => na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal),
            _ => va.cmp(&vb),
        };

        match dir {
            SortDir::Asc => cmp,
            SortDir::Desc => cmp.reverse(),
        }
    });
}

// ─── Output Formatting ──────────────────────────────────────────────────────

/// Format query results as an ASCII table.
pub fn format_results(nodes: &[&OvdNode], fields: &[&str], max_width: usize) -> String {
    if nodes.is_empty() {
        return "(0 results)".to_string();
    }

    // Determine fields to display
    let fields = if fields.is_empty() || fields[0] == "*" {
        vec!["type", "text", "href", "level", "depth", "semantic"]
    } else {
        fields.to_vec()
    };

    // Compute column widths
    let mut widths: Vec<usize> = fields.iter().map(|f| f.len()).collect();
    for node in nodes {
        for (i, field) in fields.iter().enumerate() {
            let val = node_field(node, field);
            let len = val.len().min(max_width);
            if len > widths[i] {
                widths[i] = len;
            }
        }
    }

    // Cap widths
    for w in &mut widths {
        if *w > max_width {
            *w = max_width;
        }
    }

    let mut out = String::new();

    // Header
    for (i, field) in fields.iter().enumerate() {
        out.push_str(&format!(" {:width$}", field, width = widths[i]));
        if i < fields.len() - 1 {
            out.push_str(" │");
        }
    }
    out.push('\n');

    // Separator
    for (i, _) in fields.iter().enumerate() {
        out.push_str(&"─".repeat(widths[i] + 1));
        if i < fields.len() - 1 {
            out.push_str("─┼");
        }
    }
    out.push('\n');

    // Rows
    for node in nodes {
        for (i, field) in fields.iter().enumerate() {
            let val = node_field(node, field);
            let truncated = if val.len() > max_width {
                format!("{}…", &val[..max_width - 1])
            } else {
                val
            };
            out.push_str(&format!(" {:width$}", truncated, width = widths[i]));
            if i < fields.len() - 1 {
                out.push_str(" │");
            }
        }
        out.push('\n');
    }

    out
}

/// Format results as CSV.
pub fn format_csv(nodes: &[&OvdNode], fields: &[&str]) -> String {
    let fields = if fields.is_empty() || fields[0] == "*" {
        vec!["type", "text", "href", "level", "depth", "semantic"]
    } else {
        fields.to_vec()
    };

    let mut out = String::new();

    // Header row
    out.push_str(&fields.join(","));
    out.push('\n');

    // Data rows
    for node in nodes {
        let row: Vec<String> = fields
            .iter()
            .map(|f| csv_escape(&node_field(node, f)))
            .collect();
        out.push_str(&row.join(","));
        out.push('\n');
    }

    out
}

/// Format results as JSON array.
pub fn format_json(nodes: &[&OvdNode], fields: &[&str]) -> String {
    let fields = if fields.is_empty() || fields[0] == "*" {
        vec!["type", "text", "href", "level", "depth", "semantic"]
    } else {
        fields.to_vec()
    };

    let mut out = String::from("[\n");

    for (i, node) in nodes.iter().enumerate() {
        out.push_str("  {");
        for (j, field) in fields.iter().enumerate() {
            let val = node_field(node, field);
            out.push_str(&format!("\"{}\":\"{}\"", field, json_escape(&val)));
            if j < fields.len() - 1 {
                out.push(',');
            }
        }
        out.push('}');
        if i < nodes.len() - 1 {
            out.push(',');
        }
        out.push('\n');
    }

    out.push(']');
    out
}

/// Format results as JSONL (newline-delimited JSON).
pub fn format_jsonl(nodes: &[&OvdNode], fields: &[&str]) -> String {
    let fields = if fields.is_empty() || fields[0] == "*" {
        vec!["type", "text", "href", "level", "depth", "semantic"]
    } else {
        fields.to_vec()
    };

    let mut out = String::new();

    for node in nodes {
        out.push('{');
        for (j, field) in fields.iter().enumerate() {
            let val = node_field(node, field);
            out.push_str(&format!("\"{}\":\"{}\"", field, json_escape(&val)));
            if j < fields.len() - 1 {
                out.push(',');
            }
        }
        out.push('}');
        out.push('\n');
    }

    out
}

/// Format a COUNT result.
pub fn format_count(count: usize) -> String {
    format!("COUNT: {}", count)
}

/// Format a DISTINCT result.
pub fn format_distinct(values: &[String], field: &str) -> String {
    let mut out = format!("── {} distinct values for '{}' ──\n", values.len(), field);
    for val in values {
        out.push_str(&format!("  {}\n", val));
    }
    out
}

// ─── Node Field Extraction ──────────────────────────────────────────────────

/// Extract a field value from a node as string.
pub fn node_field(node: &OvdNode, field: &str) -> String {
    match field {
        "node_id" | "id" => node.node_id.to_string(),
        "type" => node.node_type.to_string(),
        "depth" => node.depth.to_string(),
        "parent_id" | "parent" => node.parent_id.to_string(),
        "text" => node.text.clone(),
        "href" => node.href.clone(),
        "src" => node.src.clone(),
        "alt" => node.alt.clone(),
        "level" => node.level.to_string(),
        "semantic" => node.semantic.to_string(),
        "source_tag" | "tag" => node.source_tag.clone(),
        "class" | "class_names" => node.class_names.clone(),
        "confidence" => format!("{:.2}", node.confidence),
        // Style fields
        "font_size" => format!("{:.0}", node.style.font_size),
        "font_weight" => node.style.font_weight.to_string(),
        "bold" => if node.style.font_weight >= 600 { "true" } else { "false" }.to_string(),
        "italic" => if node.style.font_italic { "true" } else { "false" }.to_string(),
        "color" => format!(
            "#{:02x}{:02x}{:02x}",
            node.style.color[0], node.style.color[1], node.style.color[2]
        ),
        "background" | "bg" => format!(
            "#{:02x}{:02x}{:02x}",
            node.style.background[0], node.style.background[1], node.style.background[2]
        ),
        "display" => if node.style.display_block { "block" } else { "inline" }.to_string(),
        _ => String::new(),
    }
}

// ─── Helper Functions ───────────────────────────────────────────────────────

/// CSV-escape a value (quote if contains comma, quote, or newline).
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// JSON-escape a string value.
fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ovd::{OvdDocument, OvdNode, NodeType};

    fn make_test_doc() -> OvdDocument {
        let mut doc = OvdDocument::new("http://test.local");
        doc.title = "Test".to_string();

        let page = OvdNode::new(0, NodeType::Page, -1, 0);
        doc.add_node(page);

        let mut h1 = OvdNode::new(0, NodeType::Heading, 0, 1);
        h1.text = "Welcome".to_string();
        h1.level = 1;
        h1.source_tag = "h1".to_string();
        doc.add_node(h1);

        let mut p1 = OvdNode::new(0, NodeType::Paragraph, 0, 1);
        p1.text = "First paragraph".to_string();
        p1.source_tag = "p".to_string();
        doc.add_node(p1);

        let mut link = OvdNode::new(0, NodeType::Link, 0, 1);
        link.text = "Click here".to_string();
        link.href = "/page".to_string();
        link.source_tag = "a".to_string();
        doc.add_node(link);

        let mut p2 = OvdNode::new(0, NodeType::Paragraph, 0, 1);
        p2.text = "Second paragraph".to_string();
        p2.source_tag = "p".to_string();
        doc.add_node(p2);

        doc
    }

    #[test]
    fn test_query_select_where_type() {
        let doc = make_test_doc();
        let (nodes, _fields) = execute_query(&doc, "SELECT text FROM nodes WHERE type = 'paragraph'");
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].text, "First paragraph");
        assert_eq!(nodes[1].text, "Second paragraph");
    }

    #[test]
    fn test_query_select_where_text_contains() {
        let doc = make_test_doc();
        let (nodes, _) = execute_query(&doc, "SELECT * FROM nodes WHERE text contains 'First'");
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].text, "First paragraph");
    }

    #[test]
    fn test_query_select_limit() {
        let doc = make_test_doc();
        let (nodes, _) = execute_query(&doc, "SELECT * FROM nodes LIMIT 2");
        assert_eq!(nodes.len(), 2);
    }

    #[test]
    fn test_query_count() {
        let doc = make_test_doc();
        let (nodes, _) = execute_query(&doc, "SELECT COUNT(*) FROM nodes WHERE type = 'link'");
        // COUNT queries return matching nodes (caller formats the count)
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn test_format_csv() {
        let doc = make_test_doc();
        let headings: Vec<&OvdNode> = doc.nodes.iter()
            .filter(|n| n.node_type == NodeType::Heading)
            .collect();
        let csv = format_csv(&headings, &["type", "text"]);
        assert!(csv.contains("type,text"), "CSV should have header");
        assert!(csv.contains("heading"), "CSV should contain heading type");
        assert!(csv.contains("Welcome"), "CSV should contain heading text");
    }

    #[test]
    fn test_format_json() {
        let doc = make_test_doc();
        let links: Vec<&OvdNode> = doc.nodes.iter()
            .filter(|n| n.node_type == NodeType::Link)
            .collect();
        let json = format_json(&links, &["type", "href"]);
        assert!(json.starts_with('['));
        assert!(json.contains("\"type\":\"link\""));
        assert!(json.contains("\"/page\""));
    }
}
