//! Semantic extraction engine — converts DOM tree to OctoView Document.
//!
//! Three-layer extraction:
//!   Layer 1: HTML tag → node type (deterministic)
//!   Layer 2: ARIA roles → semantic hints
//!   Layer 3: Heuristics → div-soup decoder

use html5ever::tendril::TendrilSink;
use html5ever::{parse_document, ParseOpts};
use markup5ever_rcdom::{Handle, NodeData, RcDom};

use crate::ovd::{NodeType, OvdDocument, OvdNode, SemanticRole};

/// Parse HTML string and extract an OctoView Document.
pub fn extract_html(html: &str, url: &str) -> OvdDocument {
    let dom = parse_document(RcDom::default(), ParseOpts::default())
        .from_utf8()
        .read_from(&mut html.as_bytes())
        .expect("html5ever parse failed");

    let mut doc = OvdDocument::new(url);

    // Root page node
    let root = OvdNode::new(0, NodeType::Page, -1, 0);
    doc.add_node(root);

    // Walk the DOM tree
    walk_node(&dom.document, &mut doc, 0, 1);

    // Extract title from metadata
    extract_title(&dom.document, &mut doc);

    // Post-process: heuristic layer
    apply_heuristics(&mut doc);

    doc
}

/// Recursively walk DOM nodes, extracting content.
fn walk_node(handle: &Handle, doc: &mut OvdDocument, parent_id: i32, depth: u16) {
    let node = handle;

    match &node.data {
        NodeData::Element {
            name, attrs, ..
        } => {
            let tag = name.local.as_ref();
            let attrs = attrs.borrow();

            // Get ARIA role if present
            let aria_role = attrs
                .iter()
                .find(|a| a.name.local.as_ref() == "role")
                .map(|a| a.value.to_string());

            // Get class and id for heuristics
            let class = attrs
                .iter()
                .find(|a| a.name.local.as_ref() == "class")
                .map(|a| a.value.to_string())
                .unwrap_or_default();
            let _id = attrs
                .iter()
                .find(|a| a.name.local.as_ref() == "id")
                .map(|a| a.value.to_string())
                .unwrap_or_default();

            // Skip non-content elements
            if should_skip(tag) {
                // Still walk children of <head> to find <title>
                if tag == "head" {
                    for child in node.children.borrow().iter() {
                        walk_node(child, doc, parent_id, depth);
                    }
                }
                return;
            }

            // Layer 1: HTML tag mapping
            let (node_type, semantic) = map_tag(tag, &aria_role, &class);

            // Check if this tag produces a visible node
            match node_type {
                NodeType::Unknown if is_container_tag(tag) => {
                    // Container tags (div, span, section) — walk children under same parent
                    // unless they have ARIA roles
                    if let Some(ref role) = aria_role {
                        let (nt, sem) = map_aria_role(role);
                        if nt != NodeType::Unknown {
                            let mut ovd_node = OvdNode::new(0, nt, parent_id, depth);
                            ovd_node.source_tag = tag.to_string();
                            ovd_node.semantic = sem;
                            let nid = doc.add_node(ovd_node) as i32;
                            for child in node.children.borrow().iter() {
                                walk_node(child, doc, nid, depth + 1);
                            }
                            return;
                        }
                    }
                    // Regular container — pass through
                    for child in node.children.borrow().iter() {
                        walk_node(child, doc, parent_id, depth);
                    }
                }
                NodeType::Unknown => {
                    // Unknown non-container — walk children
                    for child in node.children.borrow().iter() {
                        walk_node(child, doc, parent_id, depth);
                    }
                }
                _ => {
                    // Known content node — create OVD node
                    let mut ovd_node = OvdNode::new(0, node_type, parent_id, depth);
                    ovd_node.source_tag = tag.to_string();
                    ovd_node.class_names = class.clone();
                    ovd_node.semantic = semantic;

                    // Extract tag-specific attributes
                    match node_type {
                        NodeType::Heading => {
                            ovd_node.level = heading_level(tag);
                            ovd_node.style.font_size = heading_font_size(ovd_node.level);
                            ovd_node.style.font_weight = 700;
                        }
                        NodeType::Link => {
                            ovd_node.href = attr_val(&attrs, "href");
                            ovd_node.style.color = [0, 102, 204, 255]; // Link blue
                        }
                        NodeType::Image => {
                            ovd_node.src = attr_val(&attrs, "src");
                            ovd_node.alt = attr_val(&attrs, "alt");
                        }
                        NodeType::InputField => {
                            let input_type = attr_val(&attrs, "type");
                            let placeholder = attr_val(&attrs, "placeholder");
                            ovd_node.text = if !placeholder.is_empty() {
                                placeholder
                            } else {
                                input_type
                            };
                        }
                        NodeType::Media => {
                            ovd_node.src = attr_val(&attrs, "src");
                        }
                        NodeType::CodeBlock => {
                            ovd_node.style.font_size = 14.0;
                        }
                        NodeType::Blockquote => {
                            ovd_node.style.font_italic = true;
                        }
                        NodeType::Button => {
                            ovd_node.style.font_weight = 600;
                        }
                        _ => {}
                    }

                    // Infer style from inline formatting tags
                    match tag {
                        "strong" | "b" => {
                            ovd_node.style.font_weight = 700;
                        }
                        "em" | "i" => {
                            ovd_node.style.font_italic = true;
                        }
                        "small" => {
                            ovd_node.style.font_size = 12.0;
                        }
                        "mark" => {
                            ovd_node.style.background = [255, 255, 0, 255]; // Highlight yellow
                        }
                        "del" | "s" => {
                            ovd_node.style.color = [128, 128, 128, 255]; // Gray for strikethrough
                        }
                        "th" => {
                            ovd_node.style.font_weight = 700; // Table headers bold
                        }
                        _ => {}
                    }

                    // Parse inline style attribute
                    let style_attr = attr_val(&attrs, "style");
                    if !style_attr.is_empty() {
                        parse_inline_style(&style_attr, &mut ovd_node.style);
                    }

                    // Check aria-hidden
                    let aria_hidden = attr_val(&attrs, "aria-hidden");
                    if aria_hidden == "true" {
                        return; // Skip hidden elements
                    }

                    let nid = doc.add_node(ovd_node) as i32;

                    // Walk children to collect text and child nodes
                    for child in node.children.borrow().iter() {
                        walk_node(child, doc, nid, depth + 1);
                    }

                    // Collect direct text from children into parent
                    collect_text_into_parent(doc, nid as u32);
                }
            }
        }
        NodeData::Text { contents } => {
            let text = contents.borrow().to_string();
            let text = collapse_whitespace(&text);
            if text.is_empty() {
                return;
            }

            // Attach text to parent node if parent exists, otherwise create text span
            if parent_id >= 0 {
                let parent_idx = parent_id as usize;
                if parent_idx < doc.nodes.len() {
                    let parent_type = doc.nodes[parent_idx].node_type;
                    // For leaf nodes that should collect text directly
                    if is_text_collecting(parent_type) {
                        // Text will be collected by collect_text_into_parent
                        let mut span = OvdNode::new(0, NodeType::TextSpan, parent_id, depth);
                        span.text = text;
                        span.source_tag = "#text".to_string();
                        span.semantic = SemanticRole::Content;
                        doc.add_node(span);
                        return;
                    }
                }
            }

            // Standalone text — create paragraph
            if text.len() > 1 {
                let mut para = OvdNode::new(0, NodeType::Paragraph, parent_id, depth);
                para.text = text;
                para.source_tag = "#text".to_string();
                para.semantic = SemanticRole::Content;
                doc.add_node(para);
            }
        }
        NodeData::Document => {
            for child in node.children.borrow().iter() {
                walk_node(child, doc, parent_id, depth);
            }
        }
        _ => {}
    }
}

/// Collect text content from TextSpan children into their parent node.
fn collect_text_into_parent(doc: &mut OvdDocument, parent_id: u32) {
    let mut text_parts: Vec<String> = Vec::new();

    for node in &doc.nodes {
        if node.parent_id == parent_id as i32 && node.node_type == NodeType::TextSpan {
            if !node.text.is_empty() {
                text_parts.push(node.text.clone());
            }
        }
    }

    if !text_parts.is_empty() {
        let combined = text_parts.join(" ");
        let combined = collapse_whitespace(&combined);
        if let Some(parent) = doc.nodes.iter_mut().find(|n| n.node_id == parent_id) {
            if parent.text.is_empty() {
                parent.text = combined;
            }
        }
    }
}

/// Layer 1: Map HTML tag to node type + semantic role.
fn map_tag(tag: &str, aria_role: &Option<String>, class: &str) -> (NodeType, SemanticRole) {
    // Layer 2: ARIA role overrides tag
    if let Some(role) = aria_role {
        let (nt, sem) = map_aria_role(role);
        if nt != NodeType::Unknown {
            return (nt, sem);
        }
    }

    // Layer 1: HTML semantics
    match tag {
        "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => (NodeType::Heading, SemanticRole::Content),
        "p" => (NodeType::Paragraph, SemanticRole::Content),
        "a" => (NodeType::Link, SemanticRole::Content),
        "img" => (NodeType::Image, SemanticRole::Media),
        "picture" => (NodeType::Image, SemanticRole::Media),
        "table" => (NodeType::Table, SemanticRole::Content),
        "tr" => (NodeType::TableRow, SemanticRole::Content),
        "td" | "th" => (NodeType::TableCell, SemanticRole::Content),
        "ul" | "ol" | "dl" => (NodeType::List, SemanticRole::Content),
        "li" | "dt" | "dd" => (NodeType::ListItem, SemanticRole::Content),
        "form" => (NodeType::Form, SemanticRole::Interactive),
        "input" | "select" | "textarea" => (NodeType::InputField, SemanticRole::Interactive),
        "button" => (NodeType::Button, SemanticRole::Interactive),
        "nav" => (NodeType::Navigation, SemanticRole::Navigation),
        "video" | "audio" | "source" => (NodeType::Media, SemanticRole::Media),
        "pre" | "code" => (NodeType::CodeBlock, SemanticRole::Content),
        "blockquote" => (NodeType::Blockquote, SemanticRole::Content),
        "hr" => (NodeType::Separator, SemanticRole::Decoration),
        "article" => (NodeType::Section, SemanticRole::Content),
        "aside" => (NodeType::Sidebar, SemanticRole::Sidebar),
        "header" => (NodeType::Header, SemanticRole::Structural),
        "footer" => (NodeType::Footer, SemanticRole::Structural),
        "main" => (NodeType::Section, SemanticRole::Content),
        "figure" => (NodeType::Section, SemanticRole::Content),
        "figcaption" => (NodeType::Paragraph, SemanticRole::Content),
        "label" => (NodeType::TextSpan, SemanticRole::Interactive),
        "strong" | "b" => (NodeType::TextSpan, SemanticRole::Content),
        "em" | "i" => (NodeType::TextSpan, SemanticRole::Content),
        "mark" => (NodeType::TextSpan, SemanticRole::Content),
        "summary" => (NodeType::Button, SemanticRole::Interactive),
        "details" => (NodeType::Section, SemanticRole::Interactive),

        // Heuristic: check class names for containers
        "div" | "span" | "section" => {
            class_heuristic(class)
        }

        _ => (NodeType::Unknown, SemanticRole::Structural),
    }
}

/// Layer 2: Map ARIA role to node type.
fn map_aria_role(role: &str) -> (NodeType, SemanticRole) {
    match role {
        "button" => (NodeType::Button, SemanticRole::Interactive),
        "navigation" => (NodeType::Navigation, SemanticRole::Navigation),
        "main" => (NodeType::Section, SemanticRole::Content),
        "search" => (NodeType::Form, SemanticRole::Search),
        "tab" => (NodeType::Button, SemanticRole::Interactive),
        "tabpanel" => (NodeType::Section, SemanticRole::Content),
        "dialog" => (NodeType::Modal, SemanticRole::Interactive),
        "alert" => (NodeType::Section, SemanticRole::Interactive),
        "banner" => (NodeType::Header, SemanticRole::Structural),
        "contentinfo" => (NodeType::Footer, SemanticRole::Structural),
        "complementary" => (NodeType::Sidebar, SemanticRole::Sidebar),
        "list" => (NodeType::List, SemanticRole::Content),
        "listitem" => (NodeType::ListItem, SemanticRole::Content),
        "heading" => (NodeType::Heading, SemanticRole::Content),
        "link" => (NodeType::Link, SemanticRole::Content),
        "img" | "figure" => (NodeType::Image, SemanticRole::Media),
        "form" => (NodeType::Form, SemanticRole::Interactive),
        "textbox" => (NodeType::InputField, SemanticRole::Interactive),
        "menu" | "menubar" => (NodeType::Navigation, SemanticRole::Navigation),
        "menuitem" => (NodeType::Button, SemanticRole::Interactive),
        _ => (NodeType::Unknown, SemanticRole::Structural),
    }
}

/// Layer 3: Class-name heuristics for div/span/section.
fn class_heuristic(class: &str) -> (NodeType, SemanticRole) {
    let lower = class.to_lowercase();

    // Navigation patterns
    if contains_any(&lower, &["nav", "menu", "breadcrumb", "topbar", "toolbar"]) {
        return (NodeType::Navigation, SemanticRole::Navigation);
    }
    // Sidebar patterns
    if contains_any(&lower, &["sidebar", "aside", "side-panel", "drawer"]) {
        return (NodeType::Sidebar, SemanticRole::Sidebar);
    }
    // Header patterns
    if contains_any(&lower, &["header", "masthead", "top-bar"]) {
        return (NodeType::Header, SemanticRole::Structural);
    }
    // Footer patterns
    if contains_any(&lower, &["footer", "copyright", "bottom-bar"]) {
        return (NodeType::Footer, SemanticRole::Structural);
    }
    // Modal/dialog patterns
    if contains_any(&lower, &["modal", "dialog", "popup", "overlay", "lightbox"]) {
        return (NodeType::Modal, SemanticRole::Interactive);
    }
    // Card patterns
    if contains_any(&lower, &["card", "tile", "post", "entry", "article"]) {
        return (NodeType::Card, SemanticRole::Content);
    }
    // Button patterns
    if contains_any(&lower, &["btn", "button", "cta"]) {
        return (NodeType::Button, SemanticRole::Interactive);
    }
    // Search patterns
    if contains_any(&lower, &["search"]) {
        return (NodeType::Form, SemanticRole::Search);
    }
    // Ad patterns
    if contains_any(&lower, &["ad-", "ads-", "advert", "sponsor", "promo", "banner-ad"]) {
        return (NodeType::Section, SemanticRole::Advertising);
    }
    // Content patterns
    if contains_any(&lower, &["content", "main", "body", "article", "story"]) {
        return (NodeType::Section, SemanticRole::Content);
    }

    (NodeType::Unknown, SemanticRole::Structural)
}

/// Post-processing heuristics on the completed document.
fn apply_heuristics(doc: &mut OvdDocument) {
    // Pass 1: Collect indices of nodes whose parent has Navigation semantic
    let nav_children: Vec<usize> = doc
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| {
            node.node_type == NodeType::Link && node.parent_id >= 0 && {
                let parent_idx = node.parent_id as usize;
                parent_idx < doc.nodes.len()
                    && doc.nodes[parent_idx].semantic == SemanticRole::Navigation
            }
        })
        .map(|(i, _)| i)
        .collect();

    // Pass 2: Apply all mutations
    for (i, node) in doc.nodes.iter_mut().enumerate() {
        // Inferred nodes from class heuristics get lower confidence
        if node.source_tag == "div" || node.source_tag == "span" || node.source_tag == "section" {
            if node.node_type != NodeType::Unknown {
                node.confidence = 0.6; // Class-name inferred
            }
        }

        // Tiny images are likely icons or tracking pixels
        if node.node_type == NodeType::Image {
            if node.src.contains("1x1") || node.src.contains("pixel")
                || node.src.contains("track") || node.src.contains("beacon")
            {
                node.semantic = SemanticRole::Tracking;
                node.node_type = NodeType::Icon;
                node.confidence = 0.7;
            }
        }

        // Links in navigation get nav semantic (from pass 1)
        if nav_children.contains(&i) {
            node.semantic = SemanticRole::Navigation;
        }
    }

    // Remove empty text spans and paragraphs
    doc.nodes.retain(|n| {
        if n.node_type == NodeType::TextSpan || n.node_type == NodeType::Paragraph {
            !n.text.trim().is_empty()
        } else {
            true
        }
    });

    // Re-index node IDs after filtering
    for (i, node) in doc.nodes.iter_mut().enumerate() {
        node.node_id = i as u32;
    }
}

/// Extract <title> text from <head>.
fn extract_title(handle: &Handle, doc: &mut OvdDocument) {
    match &handle.data {
        NodeData::Element { name, .. } => {
            if name.local.as_ref() == "title" {
                let text = get_text_content(handle);
                doc.title = collapse_whitespace(&text);
                return;
            }
        }
        _ => {}
    }
    for child in handle.children.borrow().iter() {
        extract_title(child, doc);
    }
}

/// Get concatenated text content of a node.
fn get_text_content(handle: &Handle) -> String {
    let mut result = String::new();
    match &handle.data {
        NodeData::Text { contents } => {
            result.push_str(&contents.borrow());
        }
        _ => {}
    }
    for child in handle.children.borrow().iter() {
        result.push_str(&get_text_content(child));
    }
    result
}

fn should_skip(tag: &str) -> bool {
    matches!(
        tag,
        "script" | "style" | "noscript" | "template" | "svg" | "path"
        | "meta" | "link" | "head" | "iframe" | "object" | "embed"
    )
}

fn is_container_tag(tag: &str) -> bool {
    matches!(tag, "div" | "span" | "section" | "main" | "body" | "html")
}

fn is_text_collecting(nt: NodeType) -> bool {
    matches!(
        nt,
        NodeType::Heading
            | NodeType::Paragraph
            | NodeType::Link
            | NodeType::Button
            | NodeType::ListItem
            | NodeType::TableCell
            | NodeType::Blockquote
            | NodeType::CodeBlock
            | NodeType::TextSpan
            | NodeType::Card
    )
}

fn heading_level(tag: &str) -> u8 {
    match tag {
        "h1" => 1,
        "h2" => 2,
        "h3" => 3,
        "h4" => 4,
        "h5" => 5,
        "h6" => 6,
        _ => 0,
    }
}

fn heading_font_size(level: u8) -> f32 {
    match level {
        1 => 32.0,
        2 => 24.0,
        3 => 20.0,
        4 => 18.0,
        5 => 16.0,
        6 => 14.0,
        _ => 16.0,
    }
}

fn attr_val(attrs: &[html5ever::Attribute], name: &str) -> String {
    attrs
        .iter()
        .find(|a| a.name.local.as_ref() == name)
        .map(|a| a.value.to_string())
        .unwrap_or_default()
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

/// Parse inline CSS style attribute into OvdStyle.
/// Handles common properties: font-size, font-weight, color, background-color, display.
fn parse_inline_style(style: &str, ovd_style: &mut crate::ovd::OvdStyle) {
    for decl in style.split(';') {
        let decl = decl.trim();
        if decl.is_empty() {
            continue;
        }
        let parts: Vec<&str> = decl.splitn(2, ':').collect();
        if parts.len() != 2 {
            continue;
        }
        let prop = parts[0].trim().to_lowercase();
        let val = parts[1].trim().to_lowercase();

        match prop.as_str() {
            "font-size" => {
                if let Some(px) = parse_px(&val) {
                    ovd_style.font_size = px;
                }
            }
            "font-weight" => {
                match val.as_str() {
                    "bold" | "bolder" => ovd_style.font_weight = 700,
                    "normal" | "lighter" => ovd_style.font_weight = 400,
                    _ => {
                        if let Ok(w) = val.parse::<u16>() {
                            ovd_style.font_weight = w;
                        }
                    }
                }
            }
            "font-style" => {
                ovd_style.font_italic = val == "italic" || val == "oblique";
            }
            "color" => {
                if let Some(rgba) = parse_css_color(&val) {
                    ovd_style.color = rgba;
                }
            }
            "background-color" | "background" => {
                if let Some(rgba) = parse_css_color(&val) {
                    ovd_style.background = rgba;
                }
            }
            "display" => {
                ovd_style.display_block = val != "inline" && val != "none";
            }
            _ => {}
        }
    }
}

/// Parse a CSS pixel value (e.g., "16px", "1.5em", "120%").
fn parse_px(val: &str) -> Option<f32> {
    if let Some(s) = val.strip_suffix("px") {
        return s.trim().parse().ok();
    }
    if let Some(s) = val.strip_suffix("em") {
        // Approximate: 1em ≈ 16px
        return s.trim().parse::<f32>().ok().map(|v| v * 16.0);
    }
    if let Some(s) = val.strip_suffix("rem") {
        return s.trim().parse::<f32>().ok().map(|v| v * 16.0);
    }
    if let Some(s) = val.strip_suffix('%') {
        return s.trim().parse::<f32>().ok().map(|v| v / 100.0 * 16.0);
    }
    if let Some(s) = val.strip_suffix("pt") {
        return s.trim().parse::<f32>().ok().map(|v| v * 1.333);
    }
    None
}

/// Parse a CSS color value into RGBA.
/// Supports: #rgb, #rrggbb, #rrggbbaa, rgb(), rgba(), named colors.
fn parse_css_color(val: &str) -> Option<[u8; 4]> {
    let val = val.trim();

    // Hex colors
    if val.starts_with('#') {
        let hex = &val[1..];
        return match hex.len() {
            3 => {
                let r = u8::from_str_radix(&hex[0..1], 16).ok()? * 17;
                let g = u8::from_str_radix(&hex[1..2], 16).ok()? * 17;
                let b = u8::from_str_radix(&hex[2..3], 16).ok()? * 17;
                Some([r, g, b, 255])
            }
            6 => {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                Some([r, g, b, 255])
            }
            8 => {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                let a = u8::from_str_radix(&hex[6..8], 16).ok()?;
                Some([r, g, b, a])
            }
            _ => None,
        };
    }

    // rgb(r, g, b) or rgba(r, g, b, a)
    if val.starts_with("rgb") {
        let inner = val
            .trim_start_matches("rgba(")
            .trim_start_matches("rgb(")
            .trim_end_matches(')');
        let parts: Vec<&str> = inner.split(',').collect();
        if parts.len() >= 3 {
            let r = parts[0].trim().parse::<u8>().ok()?;
            let g = parts[1].trim().parse::<u8>().ok()?;
            let b = parts[2].trim().parse::<u8>().ok()?;
            let a = if parts.len() > 3 {
                (parts[3].trim().parse::<f32>().unwrap_or(1.0) * 255.0) as u8
            } else {
                255
            };
            return Some([r, g, b, a]);
        }
    }

    // Common named colors
    match val {
        "black" => Some([0, 0, 0, 255]),
        "white" => Some([255, 255, 255, 255]),
        "red" => Some([255, 0, 0, 255]),
        "green" => Some([0, 128, 0, 255]),
        "blue" => Some([0, 0, 255, 255]),
        "yellow" => Some([255, 255, 0, 255]),
        "cyan" | "aqua" => Some([0, 255, 255, 255]),
        "magenta" | "fuchsia" => Some([255, 0, 255, 255]),
        "gray" | "grey" => Some([128, 128, 128, 255]),
        "orange" => Some([255, 165, 0, 255]),
        "purple" => Some([128, 0, 128, 255]),
        "navy" => Some([0, 0, 128, 255]),
        "teal" => Some([0, 128, 128, 255]),
        "transparent" => Some([0, 0, 0, 0]),
        _ => None,
    }
}

fn collapse_whitespace(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut last_was_ws = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !last_was_ws && !result.is_empty() {
                result.push(' ');
            }
            last_was_ws = true;
        } else {
            result.push(ch);
            last_was_ws = false;
        }
    }
    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_basic_structure() {
        let html = r#"<html><head><title>Test</title></head>
            <body><h1>Hello</h1><p>World</p></body></html>"#;
        let doc = extract_html(html, "http://test.local");
        assert!(!doc.nodes.is_empty(), "should produce nodes");
        // Must have at least a page, heading, and paragraph
        let headings: Vec<_> = doc.nodes.iter().filter(|n| n.node_type == NodeType::Heading).collect();
        let paragraphs: Vec<_> = doc.nodes.iter().filter(|n| n.node_type == NodeType::Paragraph).collect();
        assert!(!headings.is_empty(), "should extract heading");
        assert!(!paragraphs.is_empty(), "should extract paragraph");
        assert_eq!(headings[0].text, "Hello");
        // Paragraph with text "World" should be among extracted paragraphs
        assert!(paragraphs.iter().any(|p| p.text == "World"),
            "should find paragraph with text 'World', got: {:?}",
            paragraphs.iter().map(|p| &p.text).collect::<Vec<_>>());
    }

    #[test]
    fn test_extract_links_and_images() {
        let html = r#"<html><body>
            <a href="/about">About Us</a>
            <img src="logo.png" alt="Logo">
            </body></html>"#;
        let doc = extract_html(html, "http://test.local");
        let links: Vec<_> = doc.nodes.iter().filter(|n| n.node_type == NodeType::Link).collect();
        let images: Vec<_> = doc.nodes.iter().filter(|n| n.node_type == NodeType::Image).collect();
        assert!(!links.is_empty(), "should extract link");
        assert_eq!(links[0].href, "/about");
        assert_eq!(links[0].text, "About Us");
        assert!(!images.is_empty(), "should extract image");
        assert_eq!(images[0].src, "logo.png");
        assert_eq!(images[0].alt, "Logo");
    }

    #[test]
    fn test_extract_nested_structure() {
        let html = r#"<html><body>
            <nav><a href="/">Home</a><a href="/blog">Blog</a></nav>
            <ul><li>Item 1</li><li>Item 2</li></ul>
            </body></html>"#;
        let doc = extract_html(html, "http://test.local");
        let nav: Vec<_> = doc.nodes.iter().filter(|n| n.node_type == NodeType::Navigation).collect();
        let items: Vec<_> = doc.nodes.iter().filter(|n| n.node_type == NodeType::ListItem).collect();
        assert!(!nav.is_empty(), "should extract nav");
        assert_eq!(items.len(), 2, "should extract 2 list items");
    }

    #[test]
    fn test_extract_title() {
        let html = r#"<html><head><title>My Page Title</title></head><body></body></html>"#;
        let doc = extract_html(html, "http://test.local");
        assert_eq!(doc.title, "My Page Title");
    }

    #[test]
    fn test_extract_stats() {
        let html = r#"<html><body>
            <h1>Title</h1><h2>Sub</h2>
            <p>Paragraph 1</p><p>Paragraph 2</p>
            <a href="/link">Link</a>
            <table><tr><td>Cell</td></tr></table>
            </body></html>"#;
        let doc = extract_html(html, "http://test.local");
        let stats = doc.stats();
        assert!(stats.headings >= 2, "expected >=2 headings, got {}", stats.headings);
        assert!(stats.paragraphs >= 2, "expected >=2 paragraphs, got {}", stats.paragraphs);
        assert!(stats.links >= 1, "expected >=1 link, got {}", stats.links);
        assert!(stats.tables >= 1, "expected >=1 table, got {}", stats.tables);
    }
}
