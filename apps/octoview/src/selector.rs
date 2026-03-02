//! CSS Selector Engine — match OVD nodes using CSS-like syntax.
//!
//! Supports standard CSS selectors adapted for OctoView's flat columnar model:
//!   tag           — match by source_tag (a, div, p, h1, etc.)
//!   .class        — match by class_names containing class
//!   tag.class     — compound tag + class
//!   .class1.class2 — multiple class match
//!   [attr]        — attribute presence (href, src, alt, text)
//!   [attr=value]  — exact attribute match
//!   [attr*=value] — attribute contains
//!   [attr^=value] — attribute starts with
//!   [attr$=value] — attribute ends with
//!   :text(word)   — text content contains word
//!   :semantic(x)  — semantic role match
//!   :type(x)      — node type match
//!   :bold         — font_weight >= 700
//!   :italic       — font_italic = true
//!   :level(n)     — heading level match
//!   :depth(<n)    — depth comparison
//!   sel1, sel2    — OR (union of matches)
//!   sel1 sel2     — descendant (sel2 is descendant of sel1)
//!   sel1 > sel2   — direct child

use crate::ovd::{OvdDocument, OvdNode, NodeType, SemanticRole};

// ─── Public API ─────────────────────────────────────────────────────────────

/// Select nodes from a document using a CSS selector string.
pub fn select<'a>(doc: &'a OvdDocument, selector: &str) -> Vec<&'a OvdNode> {
    let selector = selector.trim();
    if selector.is_empty() {
        return doc.nodes.iter().collect();
    }

    // Handle comma-separated selectors (OR union)
    if selector.contains(',') {
        let parts = split_on_commas(selector);
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for part in &parts {
            let part = part.trim();
            if !part.is_empty() {
                for node in select_single(doc, part) {
                    if seen.insert(node.node_id) {
                        result.push(node);
                    }
                }
            }
        }
        return result;
    }

    select_single(doc, selector)
}

/// Select nodes and return owned copies (for use across threads).
#[allow(dead_code)]
pub fn select_owned(doc: &OvdDocument, selector: &str) -> Vec<OvdNode> {
    select(doc, selector).into_iter().cloned().collect()
}

/// Check if a single node matches a simple selector (no combinators).
#[allow(dead_code)]
pub fn matches_selector(node: &OvdNode, selector: &str) -> bool {
    let parsed = parse_simple_selector(selector.trim());
    matches_parsed(node, &parsed)
}

// ─── Parsed Selector ────────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct SimpleSelector {
    tag: Option<String>,
    classes: Vec<String>,
    attrs: Vec<AttrSelector>,
    pseudos: Vec<PseudoSelector>,
}

#[derive(Debug)]
enum AttrSelector {
    Has(String),                    // [attr]
    Equals(String, String),         // [attr=value]
    Contains(String, String),       // [attr*=value]
    StartsWith(String, String),     // [attr^=value]
    EndsWith(String, String),       // [attr$=value]
}

#[derive(Debug)]
enum PseudoSelector {
    Text(String),                   // :text(word)
    Semantic(String),               // :semantic(content)
    Type(String),                   // :type(heading)
    Bold,                           // :bold
    Italic,                         // :italic
    Level(u8),                      // :level(2)
    DepthLt(u16),                   // :depth(<3)
    DepthGt(u16),                   // :depth(>3)
    Empty,                          // :empty (no text)
    HasHref,                        // :has-href
    HasSrc,                         // :has-src
}

#[derive(Debug)]
enum CombinedSelector {
    Simple(SimpleSelector),
    Descendant(Box<CombinedSelector>, SimpleSelector),   // ancestor descendant
    Child(Box<CombinedSelector>, SimpleSelector),         // parent > child
}

// ─── Selector Parsing ───────────────────────────────────────────────────────

/// Parse a selector that may contain combinators (space, >).
fn parse_combined_selector(input: &str) -> CombinedSelector {
    let input = input.trim();

    // Split on child combinator first (higher specificity)
    // We need to find the LAST combinator to build the tree correctly
    let tokens = tokenize_combinators(input);

    if tokens.len() == 1 {
        return CombinedSelector::Simple(parse_simple_selector(&tokens[0].0));
    }

    // Build from left to right: a > b c  →  Descendant(Child(a, b), c)
    let mut result = CombinedSelector::Simple(parse_simple_selector(&tokens[0].0));

    for i in 1..tokens.len() {
        let sel = parse_simple_selector(&tokens[i].0);
        match tokens[i].1 {
            Combinator::Child => {
                result = CombinedSelector::Child(Box::new(result), sel);
            }
            Combinator::Descendant => {
                result = CombinedSelector::Descendant(Box::new(result), sel);
            }
        }
    }

    result
}

#[derive(Debug, Clone, Copy)]
enum Combinator {
    Descendant,  // space
    Child,       // >
}

/// Tokenize a selector into (selector_part, combinator_before_it) pairs.
fn tokenize_combinators(input: &str) -> Vec<(String, Combinator)> {
    let mut tokens: Vec<(String, Combinator)> = Vec::new();
    let mut current = String::new();
    let mut in_brackets = false;
    let mut in_parens = false;
    let mut pending_combinator = Combinator::Descendant;
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];

        if ch == '[' {
            in_brackets = true;
            current.push(ch);
        } else if ch == ']' {
            in_brackets = false;
            current.push(ch);
        } else if ch == '(' {
            in_parens = true;
            current.push(ch);
        } else if ch == ')' {
            in_parens = false;
            current.push(ch);
        } else if !in_brackets && !in_parens && ch == '>' {
            // Child combinator
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                tokens.push((trimmed, pending_combinator));
            }
            current.clear();
            pending_combinator = Combinator::Child;
            // Skip whitespace after >
            i += 1;
            while i < chars.len() && chars[i] == ' ' {
                i += 1;
            }
            continue;
        } else if !in_brackets && !in_parens && ch == ' ' {
            // Possible descendant combinator — check if followed by more selector
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                tokens.push((trimmed, pending_combinator));
                current.clear();
                pending_combinator = Combinator::Descendant;
            }
            // Skip extra whitespace
            while i + 1 < chars.len() && chars[i + 1] == ' ' {
                i += 1;
            }
        } else {
            current.push(ch);
        }

        i += 1;
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        tokens.push((trimmed, pending_combinator));
    }

    tokens
}

/// Parse a simple selector (no combinators).
/// Examples: "a", ".price", "div.card", "[href]", ":bold", "h1.title:text(hello)"
fn parse_simple_selector(input: &str) -> SimpleSelector {
    let mut sel = SimpleSelector::default();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    // Parse optional tag name (starts with letter)
    if i < chars.len() && (chars[i].is_ascii_alphabetic() || chars[i] == '*') {
        let start = i;
        while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '-' || chars[i] == '_' || chars[i] == '*') {
            i += 1;
        }
        let tag = &input[start..i];
        if tag != "*" {
            sel.tag = Some(tag.to_lowercase());
        }
    }

    // Parse modifiers: .class, [attr], :pseudo
    while i < chars.len() {
        match chars[i] {
            '.' => {
                // Class selector
                i += 1;
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '-' || chars[i] == '_') {
                    i += 1;
                }
                if i > start {
                    sel.classes.push(input[start..i].to_string());
                }
            }
            '[' => {
                // Attribute selector
                i += 1;
                let start = i;
                while i < chars.len() && chars[i] != ']' {
                    i += 1;
                }
                if i > start {
                    let attr_str = &input[start..i];
                    sel.attrs.push(parse_attr_selector(attr_str));
                }
                if i < chars.len() {
                    i += 1; // skip ']'
                }
            }
            ':' => {
                // Pseudo selector
                i += 1;
                let start = i;
                while i < chars.len() && chars[i] != ':' && chars[i] != '.' && chars[i] != '[' {
                    // Handle parentheses
                    if chars[i] == '(' {
                        let mut depth = 1;
                        i += 1;
                        while i < chars.len() && depth > 0 {
                            if chars[i] == '(' { depth += 1; }
                            if chars[i] == ')' { depth -= 1; }
                            i += 1;
                        }
                    } else {
                        i += 1;
                    }
                }
                if i > start {
                    let pseudo_str = &input[start..i];
                    if let Some(ps) = parse_pseudo_selector(pseudo_str) {
                        sel.pseudos.push(ps);
                    }
                }
            }
            _ => {
                i += 1; // skip unexpected
            }
        }
    }

    sel
}

/// Parse an attribute selector string (contents between [ and ]).
fn parse_attr_selector(s: &str) -> AttrSelector {
    if let Some(idx) = s.find("*=") {
        let attr = s[..idx].trim().to_string();
        let val = s[idx + 2..].trim().trim_matches('"').trim_matches('\'').to_string();
        return AttrSelector::Contains(attr, val);
    }
    if let Some(idx) = s.find("^=") {
        let attr = s[..idx].trim().to_string();
        let val = s[idx + 2..].trim().trim_matches('"').trim_matches('\'').to_string();
        return AttrSelector::StartsWith(attr, val);
    }
    if let Some(idx) = s.find("$=") {
        let attr = s[..idx].trim().to_string();
        let val = s[idx + 2..].trim().trim_matches('"').trim_matches('\'').to_string();
        return AttrSelector::EndsWith(attr, val);
    }
    if let Some(idx) = s.find('=') {
        let attr = s[..idx].trim().to_string();
        let val = s[idx + 1..].trim().trim_matches('"').trim_matches('\'').to_string();
        return AttrSelector::Equals(attr, val);
    }
    AttrSelector::Has(s.trim().to_string())
}

/// Parse a pseudo-selector string (after the colon).
fn parse_pseudo_selector(s: &str) -> Option<PseudoSelector> {
    let s = s.trim();

    // Parameterized pseudos
    if let Some(inner) = extract_parens(s, "text") {
        return Some(PseudoSelector::Text(inner));
    }
    if let Some(inner) = extract_parens(s, "semantic") {
        return Some(PseudoSelector::Semantic(inner));
    }
    if let Some(inner) = extract_parens(s, "type") {
        return Some(PseudoSelector::Type(inner));
    }
    if let Some(inner) = extract_parens(s, "level") {
        if let Ok(n) = inner.parse::<u8>() {
            return Some(PseudoSelector::Level(n));
        }
    }
    if let Some(inner) = extract_parens(s, "depth") {
        if let Some(rest) = inner.strip_prefix('<') {
            if let Ok(n) = rest.trim().parse::<u16>() {
                return Some(PseudoSelector::DepthLt(n));
            }
        }
        if let Some(rest) = inner.strip_prefix('>') {
            if let Ok(n) = rest.trim().parse::<u16>() {
                return Some(PseudoSelector::DepthGt(n));
            }
        }
    }

    // Simple pseudos
    match s {
        "bold" => Some(PseudoSelector::Bold),
        "italic" => Some(PseudoSelector::Italic),
        "empty" => Some(PseudoSelector::Empty),
        "has-href" => Some(PseudoSelector::HasHref),
        "has-src" => Some(PseudoSelector::HasSrc),
        _ => None,
    }
}

/// Extract the content inside parentheses for a named pseudo.
/// e.g., extract_parens("text(hello)", "text") -> Some("hello")
fn extract_parens(s: &str, prefix: &str) -> Option<String> {
    if s.starts_with(prefix) {
        let rest = &s[prefix.len()..];
        if rest.starts_with('(') && rest.ends_with(')') {
            return Some(rest[1..rest.len() - 1].to_string());
        }
    }
    None
}

// ─── Selector Matching ──────────────────────────────────────────────────────

/// Match a single node against a parsed simple selector.
fn matches_parsed(node: &OvdNode, sel: &SimpleSelector) -> bool {
    // Tag match
    if let Some(ref tag) = sel.tag {
        if node.source_tag != *tag {
            // Also try matching against OVD node type name
            let type_name = node.node_type.to_string();
            if type_name != *tag {
                return false;
            }
        }
    }

    // Class matches (all must be present)
    for class in &sel.classes {
        if !has_class(&node.class_names, class) {
            return false;
        }
    }

    // Attribute matches
    for attr in &sel.attrs {
        if !matches_attr(node, attr) {
            return false;
        }
    }

    // Pseudo-selector matches
    for pseudo in &sel.pseudos {
        if !matches_pseudo(node, pseudo) {
            return false;
        }
    }

    true
}

/// Check if class_names (space-separated) contains a specific class.
fn has_class(class_names: &str, target: &str) -> bool {
    class_names.split_whitespace()
        .any(|c| c.eq_ignore_ascii_case(target))
}

/// Get an "attribute" value from a node for selector matching.
fn get_attr(node: &OvdNode, attr: &str) -> String {
    match attr {
        "href" => node.href.clone(),
        "src" => node.src.clone(),
        "alt" => node.alt.clone(),
        "text" | "content" => node.text.clone(),
        "class" | "class_names" => node.class_names.clone(),
        "tag" | "source_tag" => node.source_tag.clone(),
        "type" => node.node_type.to_string(),
        "semantic" => node.semantic.to_string(),
        "level" => node.level.to_string(),
        "depth" => node.depth.to_string(),
        "id" | "node_id" => node.node_id.to_string(),
        _ => String::new(),
    }
}

fn matches_attr(node: &OvdNode, attr: &AttrSelector) -> bool {
    match attr {
        AttrSelector::Has(name) => {
            let val = get_attr(node, name);
            !val.is_empty()
        }
        AttrSelector::Equals(name, expected) => {
            let val = get_attr(node, name);
            val == *expected
        }
        AttrSelector::Contains(name, substr) => {
            let val = get_attr(node, name).to_lowercase();
            val.contains(&substr.to_lowercase())
        }
        AttrSelector::StartsWith(name, prefix) => {
            let val = get_attr(node, name).to_lowercase();
            val.starts_with(&prefix.to_lowercase())
        }
        AttrSelector::EndsWith(name, suffix) => {
            let val = get_attr(node, name).to_lowercase();
            val.ends_with(&suffix.to_lowercase())
        }
    }
}

fn matches_pseudo(node: &OvdNode, pseudo: &PseudoSelector) -> bool {
    match pseudo {
        PseudoSelector::Text(word) => {
            node.text.to_lowercase().contains(&word.to_lowercase())
        }
        PseudoSelector::Semantic(role) => {
            node.semantic.to_string() == *role
        }
        PseudoSelector::Type(typ) => {
            node.node_type.to_string() == *typ
        }
        PseudoSelector::Bold => {
            node.style.font_weight >= 700
        }
        PseudoSelector::Italic => {
            node.style.font_italic
        }
        PseudoSelector::Level(n) => {
            node.level == *n
        }
        PseudoSelector::DepthLt(n) => {
            node.depth < *n
        }
        PseudoSelector::DepthGt(n) => {
            node.depth > *n
        }
        PseudoSelector::Empty => {
            node.text.trim().is_empty()
        }
        PseudoSelector::HasHref => {
            !node.href.is_empty()
        }
        PseudoSelector::HasSrc => {
            !node.src.is_empty()
        }
    }
}

// ─── Combined Selector Matching ─────────────────────────────────────────────

/// Select nodes matching a single selector (may have combinators).
fn select_single<'a>(doc: &'a OvdDocument, selector: &str) -> Vec<&'a OvdNode> {
    let combined = parse_combined_selector(selector);
    match_combined(doc, &combined)
}

fn match_combined<'a>(doc: &'a OvdDocument, sel: &CombinedSelector) -> Vec<&'a OvdNode> {
    match sel {
        CombinedSelector::Simple(simple) => {
            doc.nodes.iter().filter(|n| matches_parsed(n, simple)).collect()
        }
        CombinedSelector::Child(parent_sel, child_simple) => {
            // Find all nodes matching parent
            let parent_ids: std::collections::HashSet<u32> = match_combined(doc, parent_sel)
                .iter()
                .map(|n| n.node_id)
                .collect();

            // Find nodes matching child that are direct children of matched parents
            doc.nodes.iter()
                .filter(|n| {
                    matches_parsed(n, child_simple)
                    && n.parent_id >= 0
                    && parent_ids.contains(&(n.parent_id as u32))
                })
                .collect()
        }
        CombinedSelector::Descendant(ancestor_sel, desc_simple) => {
            // Find all ancestor node IDs
            let ancestor_ids: std::collections::HashSet<u32> = match_combined(doc, ancestor_sel)
                .iter()
                .map(|n| n.node_id)
                .collect();

            // For each candidate, check if any ancestor is in the matched set
            doc.nodes.iter()
                .filter(|n| {
                    matches_parsed(n, desc_simple)
                    && is_descendant_of(doc, n, &ancestor_ids)
                })
                .collect()
        }
    }
}

/// Check if a node is a descendant of any node in the given set.
fn is_descendant_of(doc: &OvdDocument, node: &OvdNode, ancestor_ids: &std::collections::HashSet<u32>) -> bool {
    let mut current_parent = node.parent_id;
    let mut depth = 0;
    while current_parent >= 0 && depth < 100 {
        if ancestor_ids.contains(&(current_parent as u32)) {
            return true;
        }
        // Walk up
        let parent_idx = current_parent as usize;
        if parent_idx < doc.nodes.len() {
            current_parent = doc.nodes[parent_idx].parent_id;
        } else {
            break;
        }
        depth += 1;
    }
    false
}

/// Split on commas but not inside brackets or parentheses.
fn split_on_commas(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth = 0;

    for ch in s.chars() {
        match ch {
            '(' | '[' => { depth += 1; current.push(ch); }
            ')' | ']' => { depth -= 1; current.push(ch); }
            ',' if depth == 0 => {
                parts.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(ch),
        }
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        parts.push(trimmed);
    }

    parts
}

// ─── LLM Feed Format ────────────────────────────────────────────────────────

/// Format a document as an LLM-optimized feed.
/// Token-efficient, structured, semantic — the "modern RSS" format.
pub fn format_feed(doc: &OvdDocument) -> String {
    let mut out = String::new();

    // Compact header
    out.push_str(&format!("# {}\n", doc.title));
    out.push_str(&format!("@source {}\n", doc.url));
    out.push_str(&format!("@nodes {}\n\n", doc.nodes.len()));

    // Headings → outline
    let headings: Vec<&OvdNode> = doc.nodes.iter()
        .filter(|n| n.node_type == NodeType::Heading && !n.text.is_empty())
        .collect();
    if !headings.is_empty() {
        out.push_str("## Outline\n");
        for h in &headings {
            let indent = "  ".repeat(h.level.saturating_sub(1) as usize);
            out.push_str(&format!("{}{}\n", indent, h.text));
        }
        out.push('\n');
    }

    // Content paragraphs (substantial ones)
    let paragraphs: Vec<&OvdNode> = doc.nodes.iter()
        .filter(|n| n.node_type == NodeType::Paragraph && n.text.len() > 20)
        .collect();
    if !paragraphs.is_empty() {
        out.push_str("## Content\n");
        for p in paragraphs.iter().take(50) {
            out.push_str(&p.text);
            out.push('\n');
        }
        if paragraphs.len() > 50 {
            out.push_str(&format!("... +{} more paragraphs\n", paragraphs.len() - 50));
        }
        out.push('\n');
    }

    // Links (with text)
    let links: Vec<&OvdNode> = doc.nodes.iter()
        .filter(|n| n.node_type == NodeType::Link && !n.text.is_empty() && !n.href.is_empty())
        .collect();
    if !links.is_empty() {
        out.push_str("## Links\n");
        // Deduplicate by text+href
        let mut seen = std::collections::HashSet::new();
        let mut count = 0;
        for l in &links {
            let key = format!("{}|{}", l.text, l.href);
            if seen.insert(key) && count < 100 {
                out.push_str(&format!("- [{}]({})\n", l.text.replace('\n', " "), l.href));
                count += 1;
            }
        }
        if links.len() > count {
            out.push_str(&format!("... +{} more links\n", links.len() - count));
        }
        out.push('\n');
    }

    // Images
    let images: Vec<&OvdNode> = doc.nodes.iter()
        .filter(|n| n.node_type == NodeType::Image && !n.src.is_empty())
        .collect();
    if !images.is_empty() {
        out.push_str("## Media\n");
        for img in images.iter().take(20) {
            let desc = if !img.alt.is_empty() { &img.alt } else { &img.src };
            out.push_str(&format!("- img: {}\n", desc));
        }
        if images.len() > 20 {
            out.push_str(&format!("... +{} more images\n", images.len() - 20));
        }
        out.push('\n');
    }

    // Navigation structure
    let nav_links: Vec<&OvdNode> = doc.nodes.iter()
        .filter(|n| n.node_type == NodeType::Link && n.semantic == SemanticRole::Navigation && !n.text.is_empty())
        .collect();
    if !nav_links.is_empty() {
        out.push_str("## Navigation\n");
        for nl in nav_links.iter().take(30) {
            out.push_str(&format!("- {} → {}\n", nl.text.replace('\n', " "), nl.href));
        }
        out.push('\n');
    }

    // Forms/Interactive
    let forms: Vec<&OvdNode> = doc.nodes.iter()
        .filter(|n| n.semantic == SemanticRole::Interactive && !n.text.is_empty())
        .collect();
    if !forms.is_empty() {
        out.push_str("## Interactive\n");
        for f in forms.iter().take(20) {
            out.push_str(&format!("- {}: {}\n", f.node_type, f.text));
        }
        out.push('\n');
    }

    out
}

/// Format selected nodes as a compact feed for LLM consumption.
pub fn format_feed_nodes(nodes: &[&OvdNode], label: &str) -> String {
    let mut out = String::new();

    if !label.is_empty() {
        out.push_str(&format!("# {} ({} results)\n\n", label, nodes.len()));
    }

    for node in nodes {
        match node.node_type {
            NodeType::Link => {
                if !node.text.is_empty() {
                    out.push_str(&format!("[{}]({})\n", node.text.replace('\n', " "), node.href));
                }
            }
            NodeType::Image => {
                let desc = if !node.alt.is_empty() { &node.alt } else { &node.src };
                out.push_str(&format!("![{}]({})\n", desc, node.src));
            }
            NodeType::Heading => {
                let prefix = "#".repeat(node.level.max(1) as usize);
                out.push_str(&format!("{} {}\n", prefix, node.text));
            }
            _ => {
                if !node.text.is_empty() {
                    out.push_str(&node.text);
                    out.push('\n');
                }
            }
        }
    }

    out
}

/// Format a document as machine-readable JSONL feed.
/// Each node type gets its own compact JSON line.
pub fn format_feed_jsonl(doc: &OvdDocument) -> String {
    let mut out = String::new();

    // Meta line
    out.push_str(&format!(
        "{{\"type\":\"meta\",\"title\":\"{}\",\"url\":\"{}\",\"nodes\":{}}}\n",
        json_escape(&doc.title),
        json_escape(&doc.url),
        doc.nodes.len()
    ));

    // Content nodes only (skip structural/decoration)
    for node in &doc.nodes {
        if node.text.is_empty()
            && node.href.is_empty()
            && node.src.is_empty()
        {
            continue;
        }

        // Skip noise
        if matches!(node.semantic, SemanticRole::Tracking | SemanticRole::Decoration) {
            continue;
        }

        out.push('{');
        out.push_str(&format!("\"t\":\"{}\"", node.node_type));

        if !node.text.is_empty() {
            out.push_str(&format!(",\"x\":\"{}\"", json_escape(&node.text)));
        }
        if !node.href.is_empty() {
            out.push_str(&format!(",\"h\":\"{}\"", json_escape(&node.href)));
        }
        if !node.src.is_empty() {
            out.push_str(&format!(",\"s\":\"{}\"", json_escape(&node.src)));
        }
        if node.level > 0 {
            out.push_str(&format!(",\"l\":{}", node.level));
        }
        if node.semantic != SemanticRole::Content {
            out.push_str(&format!(",\"r\":\"{}\"", node.semantic));
        }
        if !node.class_names.is_empty() {
            out.push_str(&format!(",\"c\":\"{}\"", json_escape(&node.class_names)));
        }

        out.push_str("}\n");
    }

    out
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ovd::{OvdNode, OvdDocument, NodeType, SemanticRole};

    fn make_doc() -> OvdDocument {
        let mut doc = OvdDocument::new("https://example.com");
        doc.title = "Test Page".to_string();

        // node 0: page
        let page = OvdNode::new(0, NodeType::Page, -1, 0);
        doc.add_node(page);

        // node 1: h1
        let mut h1 = OvdNode::new(0, NodeType::Heading, 0, 1);
        h1.text = "Welcome".to_string();
        h1.level = 1;
        h1.source_tag = "h1".to_string();
        h1.style.font_weight = 700;
        doc.add_node(h1);

        // node 2: nav
        let mut nav = OvdNode::new(0, NodeType::Navigation, 0, 1);
        nav.source_tag = "nav".to_string();
        nav.class_names = "main-nav top-bar".to_string();
        nav.semantic = SemanticRole::Navigation;
        doc.add_node(nav);

        // node 3: link inside nav
        let mut link = OvdNode::new(0, NodeType::Link, 2, 2);
        link.text = "Home".to_string();
        link.href = "/".to_string();
        link.source_tag = "a".to_string();
        link.class_names = "nav-link active".to_string();
        link.semantic = SemanticRole::Navigation;
        doc.add_node(link);

        // node 4: link inside nav
        let mut link2 = OvdNode::new(0, NodeType::Link, 2, 2);
        link2.text = "About".to_string();
        link2.href = "/about".to_string();
        link2.source_tag = "a".to_string();
        link2.class_names = "nav-link".to_string();
        link2.semantic = SemanticRole::Navigation;
        doc.add_node(link2);

        // node 5: paragraph
        let mut para = OvdNode::new(0, NodeType::Paragraph, 0, 1);
        para.text = "This is the main content of the page.".to_string();
        para.source_tag = "p".to_string();
        para.class_names = "intro bold-text".to_string();
        doc.add_node(para);

        // node 6: div.card
        let mut card = OvdNode::new(0, NodeType::Card, 0, 1);
        card.source_tag = "div".to_string();
        card.class_names = "card featured".to_string();
        doc.add_node(card);

        // node 7: h2 inside card
        let mut h2 = OvdNode::new(0, NodeType::Heading, 6, 2);
        h2.text = "Featured Article".to_string();
        h2.level = 2;
        h2.source_tag = "h2".to_string();
        h2.class_names = "card-title".to_string();
        h2.style.font_weight = 700;
        doc.add_node(h2);

        // node 8: image
        let mut img = OvdNode::new(0, NodeType::Image, 6, 2);
        img.src = "https://example.com/photo.jpg".to_string();
        img.alt = "A beautiful sunset".to_string();
        img.source_tag = "img".to_string();
        doc.add_node(img);

        doc
    }

    #[test]
    fn test_tag_selector() {
        let doc = make_doc();
        let results = select(&doc, "a");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].text, "Home");
        assert_eq!(results[1].text, "About");
    }

    #[test]
    fn test_class_selector() {
        let doc = make_doc();
        let results = select(&doc, ".nav-link");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_compound_tag_class() {
        let doc = make_doc();
        let results = select(&doc, "a.active");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "Home");
    }

    #[test]
    fn test_multi_class() {
        let doc = make_doc();
        let results = select(&doc, ".card.featured");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_attr_has() {
        let doc = make_doc();
        let results = select(&doc, "[href]");
        assert_eq!(results.len(), 2); // two links
    }

    #[test]
    fn test_attr_contains() {
        let doc = make_doc();
        let results = select(&doc, "[href*=about]");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "About");
    }

    #[test]
    fn test_attr_starts_with() {
        let doc = make_doc();
        let results = select(&doc, "[src^=https]");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_pseudo_text() {
        let doc = make_doc();
        let results = select(&doc, ":text(main content)");
        assert_eq!(results.len(), 1);
        assert!(results[0].text.contains("main content"));
    }

    #[test]
    fn test_pseudo_bold() {
        let doc = make_doc();
        let results = select(&doc, ":bold");
        assert!(results.len() >= 2); // h1 and h2 have weight 700
    }

    #[test]
    fn test_pseudo_level() {
        let doc = make_doc();
        let results = select(&doc, ":level(1)");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "Welcome");
    }

    #[test]
    fn test_pseudo_semantic() {
        let doc = make_doc();
        let results = select(&doc, ":semantic(navigation)");
        assert_eq!(results.len(), 3); // nav + 2 links
    }

    #[test]
    fn test_pseudo_type() {
        let doc = make_doc();
        let results = select(&doc, ":type(heading)");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_comma_union() {
        let doc = make_doc();
        let results = select(&doc, "h1, h2");
        // Should match by source_tag
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_child_combinator() {
        let doc = make_doc();
        let results = select(&doc, "nav > a");
        assert_eq!(results.len(), 2); // both links are direct children of nav
    }

    #[test]
    fn test_descendant_combinator() {
        let doc = make_doc();
        // card (node 6) has h2 child (node 7) and img child (node 8)
        let results = select(&doc, ".card img");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].alt, "A beautiful sunset");
    }

    #[test]
    fn test_feed_format() {
        let doc = make_doc();
        let feed = format_feed(&doc);
        assert!(feed.contains("# Test Page"));
        assert!(feed.contains("@source https://example.com"));
        assert!(feed.contains("Welcome"));
    }

    #[test]
    fn test_feed_jsonl() {
        let doc = make_doc();
        let jsonl = format_feed_jsonl(&doc);
        assert!(jsonl.starts_with("{\"type\":\"meta\""));
        assert!(jsonl.contains("\"t\":\"heading\""));
    }
}
