/// Lightweight HTML → OVD extraction for the browser.
/// Maps HTML tags to semantic OvdNode types.

use html5ever::tendril::TendrilSink;
use html5ever::{parse_document, ParseOpts};
use markup5ever_rcdom::{Handle, NodeData, RcDom};

use crate::css;
use crate::ovd::{NodeType, OvdDocument, OvdNode, OvdStyle};
use crate::selector_match::{self, Specificity};
use crate::stylesheet::{CssDeclaration, Stylesheet};

/// Parse an HTML string into an RcDom.
pub fn parse_html(html: &str) -> RcDom {
    parse_document(RcDom::default(), ParseOpts::default())
        .from_utf8()
        .read_from(&mut html.as_bytes())
        .unwrap_or_else(|_| RcDom::default())
}

/// Parse HTML string and extract to an OvdDocument (no external stylesheet).
pub fn extract_html(html: &str, url: &str) -> OvdDocument {
    let dom = parse_html(html);
    let empty_sheet = Stylesheet::default();
    extract_from_dom(&dom, url, &empty_sheet)
}

/// Parse HTML string and extract to an OvdDocument with an external stylesheet applied.
#[allow(dead_code)]
pub fn extract_html_with_stylesheet(html: &str, url: &str, stylesheet: &Stylesheet) -> OvdDocument {
    let dom = parse_html(html);
    extract_from_dom(&dom, url, stylesheet)
}

/// Extract an OvdDocument from an already-parsed DOM, applying CSS from the given stylesheet.
pub fn extract_from_dom(dom: &RcDom, url: &str, stylesheet: &Stylesheet) -> OvdDocument {
    let mut doc = OvdDocument::new(url);

    // Add root page node
    let mut page = OvdNode::new(NodeType::Page);
    page.text = url.to_string();
    doc.add_node(page);

    // Extract title
    doc.title = extract_title(&dom.document);

    // Walk DOM and extract nodes with stylesheet cascade
    let parent_style = OvdStyle::default();
    walk_node(&dom.document, &mut doc, 0, 0, stylesheet, &[], &parent_style);

    doc
}

fn extract_title(handle: &Handle) -> String {
    let node = handle;
    if let NodeData::Element { ref name, .. } = node.data {
        if name.local.as_ref() == "title" {
            return collect_text(handle);
        }
    }
    for child in node.children.borrow().iter() {
        let title = extract_title(child);
        if !title.is_empty() {
            return title;
        }
    }
    String::new()
}

/// Walk the DOM tree and collect text content of all `<style>` elements.
pub fn extract_stylesheets(handle: &Handle) -> Vec<String> {
    let mut result = Vec::new();
    extract_stylesheets_inner(handle, &mut result);
    result
}

fn extract_stylesheets_inner(handle: &Handle, result: &mut Vec<String>) {
    if let NodeData::Element { ref name, .. } = handle.data {
        if name.local.as_ref() == "style" {
            let text = collect_text(handle);
            if !text.trim().is_empty() {
                result.push(text);
            }
            return; // Don't recurse into <style> children further
        }
    }
    for child in handle.children.borrow().iter() {
        extract_stylesheets_inner(child, result);
    }
}

/// Walk the DOM tree and collect text content of all inline `<script>` elements.
/// External scripts (`<script src="...">`) with no inline body are skipped.
pub fn extract_scripts(handle: &Handle) -> Vec<String> {
    let mut scripts = Vec::new();
    extract_scripts_inner(handle, &mut scripts);
    scripts
}

fn extract_scripts_inner(handle: &Handle, scripts: &mut Vec<String>) {
    if let NodeData::Element { ref name, .. } = handle.data {
        if name.local.as_ref() == "script" {
            let text = collect_text(handle);
            if !text.trim().is_empty() {
                scripts.push(text);
            }
            return;
        }
    }
    for child in handle.children.borrow().iter() {
        extract_scripts_inner(child, scripts);
    }
}

/// Walk the DOM tree and collect absolute URLs from `<link rel="stylesheet" href="...">` elements.
pub fn extract_stylesheet_urls(handle: &Handle, base_url: &str) -> Vec<String> {
    let mut result = Vec::new();
    extract_stylesheet_urls_inner(handle, base_url, &mut result);
    result
}

fn extract_stylesheet_urls_inner(handle: &Handle, base_url: &str, result: &mut Vec<String>) {
    if let NodeData::Element { ref name, ref attrs, .. } = handle.data {
        if name.local.as_ref() == "link" {
            let attrs = attrs.borrow();
            let is_stylesheet = attrs.iter().any(|a| {
                a.name.local.as_ref() == "rel"
                    && a.value.to_string().eq_ignore_ascii_case("stylesheet")
            });
            if is_stylesheet {
                if let Some(href) = attrs
                    .iter()
                    .find(|a| a.name.local.as_ref() == "href")
                    .map(|a| a.value.to_string())
                {
                    let href = href.trim();
                    if !href.is_empty() {
                        // Resolve against base URL
                        if let Ok(base) = url::Url::parse(base_url) {
                            if let Ok(resolved) = base.join(href) {
                                result.push(resolved.to_string());
                            }
                        } else if href.starts_with("http://") || href.starts_with("https://") {
                            result.push(href.to_string());
                        }
                    }
                }
            }
        }
    }
    for child in handle.children.borrow().iter() {
        extract_stylesheet_urls_inner(child, base_url, result);
    }
}

/// Apply a single CSS declaration to an OvdStyle.
fn apply_declaration(style: &mut OvdStyle, decl: &CssDeclaration) {
    match decl.property.as_str() {
        "color" => {
            style.color = css::parse_color(&decl.value);
        }
        "background-color" | "background" => {
            style.background = css::parse_color(&decl.value);
        }
        "font-size" => {
            style.font_size_px = css::parse_font_size(&decl.value);
        }
        "font-weight" => {
            style.font_weight = css::parse_font_weight(&decl.value);
        }
        "text-align" => {
            style.text_align = css::parse_text_align(&decl.value);
        }
        "border-color" => {
            style.border_color = css::parse_color(&decl.value);
        }
        "display" => {
            if decl.value.eq_ignore_ascii_case("none") {
                style.display_none = true;
            }
            style.display = css::parse_display(&decl.value);
        }
        "margin" => {
            let vals = css::parse_box_shorthand(&decl.value);
            style.margin_top = Some(vals.0);
            style.margin_right = Some(vals.1);
            style.margin_bottom = Some(vals.2);
            style.margin_left = Some(vals.3);
        }
        "margin-top" => { style.margin_top = css::parse_length(&decl.value); }
        "margin-bottom" => { style.margin_bottom = css::parse_length(&decl.value); }
        "margin-left" => { style.margin_left = css::parse_length(&decl.value); }
        "margin-right" => { style.margin_right = css::parse_length(&decl.value); }
        "padding" => {
            let vals = css::parse_box_shorthand(&decl.value);
            style.padding_top = Some(vals.0);
            style.padding_right = Some(vals.1);
            style.padding_bottom = Some(vals.2);
            style.padding_left = Some(vals.3);
        }
        "padding-top" => { style.padding_top = css::parse_length(&decl.value); }
        "padding-bottom" => { style.padding_bottom = css::parse_length(&decl.value); }
        "padding-left" => { style.padding_left = css::parse_length(&decl.value); }
        "padding-right" => { style.padding_right = css::parse_length(&decl.value); }
        "width" => { style.width = css::parse_length(&decl.value); }
        "max-width" => { style.max_width = css::parse_length(&decl.value); }
        "text-decoration" | "text-decoration-line" => {
            if decl.value.to_lowercase().contains("underline") {
                style.text_decoration_underline = true;
            }
        }
        "line-height" => { style.line_height = css::parse_length(&decl.value); }
        "border-width" => { style.border_width = css::parse_length(&decl.value); }
        "border" => {
            for part in decl.value.split_whitespace() {
                if let Some(w) = css::parse_length(part) {
                    style.border_width = Some(w);
                }
                if let Some(c) = css::parse_color(part) {
                    style.border_color = Some(c);
                }
            }
        }
        _ => {}
    }
}

/// Match CSS rules from the stylesheet against a DOM element and return
/// the computed style (stylesheet rules cascaded by specificity, then
/// inline style on top, then inheritance from parent).
fn compute_style(
    handle: &Handle,
    attrs: &[markup5ever::Attribute],
    stylesheet: &Stylesheet,
    ancestors: &[Handle],
    parent_style: &OvdStyle,
) -> OvdStyle {
    // Collect matching rules with their specificity
    let mut matched: Vec<(Specificity, &[CssDeclaration])> = Vec::new();

    for rule in &stylesheet.rules {
        for chain in &rule.parsed_selectors {
            if selector_match::matches_element(chain, handle, ancestors) {
                let spec = Specificity::from_chain(chain);
                matched.push((spec, &rule.declarations));
                break; // One match per rule is enough
            }
        }
    }

    // Sort by specificity ascending (lower specificity first, higher overwrites)
    matched.sort_by_key(|(spec, _)| *spec);

    // Build base style from matched rules
    let mut style = OvdStyle::default();
    for (_, declarations) in &matched {
        for decl in *declarations {
            apply_declaration(&mut style, decl);
        }
    }

    // Apply inline style on top (highest specificity)
    if let Some(style_str) = attrs
        .iter()
        .find(|a| a.name.local.as_ref() == "style")
        .map(|a| a.value.to_string())
    {
        let inline = css::parse_inline_style(&style_str);
        // Merge inline over cascaded: only override if inline set something
        if inline.color.is_some() { style.color = inline.color; }
        if inline.background.is_some() { style.background = inline.background; }
        if inline.font_size_px.is_some() { style.font_size_px = inline.font_size_px; }
        if inline.font_weight.is_some() { style.font_weight = inline.font_weight; }
        if inline.text_align.is_some() { style.text_align = inline.text_align; }
        if inline.border_color.is_some() { style.border_color = inline.border_color; }
        if inline.display_none { style.display_none = true; }
        if inline.display.is_some() { style.display = inline.display; }
        if inline.margin_top.is_some() { style.margin_top = inline.margin_top; }
        if inline.margin_bottom.is_some() { style.margin_bottom = inline.margin_bottom; }
        if inline.margin_left.is_some() { style.margin_left = inline.margin_left; }
        if inline.margin_right.is_some() { style.margin_right = inline.margin_right; }
        if inline.padding_top.is_some() { style.padding_top = inline.padding_top; }
        if inline.padding_bottom.is_some() { style.padding_bottom = inline.padding_bottom; }
        if inline.padding_left.is_some() { style.padding_left = inline.padding_left; }
        if inline.padding_right.is_some() { style.padding_right = inline.padding_right; }
        if inline.width.is_some() { style.width = inline.width; }
        if inline.max_width.is_some() { style.max_width = inline.max_width; }
        if inline.text_decoration_underline { style.text_decoration_underline = true; }
        if inline.line_height.is_some() { style.line_height = inline.line_height; }
        if inline.border_width.is_some() { style.border_width = inline.border_width; }
    }

    // Legacy attributes (lowest priority, only if nothing else set them)
    if style.background.is_none() {
        if let Some(bg) = attrs
            .iter()
            .find(|a| a.name.local.as_ref() == "bgcolor")
            .and_then(|a| css::parse_color(&a.value))
        {
            style.background = Some(bg);
        }
    }
    if style.color.is_none() {
        if let Some(fg) = attrs
            .iter()
            .find(|a| a.name.local.as_ref() == "color")
            .and_then(|a| css::parse_color(&a.value))
        {
            style.color = Some(fg);
        }
    }

    // Inherit certain properties from parent if not explicitly set
    if style.color.is_none() {
        style.color = parent_style.color;
    }
    if style.font_size_px.is_none() {
        style.font_size_px = parent_style.font_size_px;
    }
    if style.font_weight.is_none() {
        style.font_weight = parent_style.font_weight;
    }
    if style.text_align.is_none() {
        style.text_align = parent_style.text_align;
    }
    // background and display_none do NOT inherit

    style
}

const MAX_DOM_DEPTH: u16 = 64;

fn walk_node(
    handle: &Handle,
    doc: &mut OvdDocument,
    depth: u16,
    parent_id: i32,
    stylesheet: &Stylesheet,
    ancestors: &[Handle],
    parent_style: &OvdStyle,
) {
    // Prevent stack overflow on deeply nested DOMs (YouTube, etc.)
    if depth > MAX_DOM_DEPTH {
        return;
    }

    match &handle.data {
        NodeData::Element { ref name, ref attrs, .. } => {
            let tag = name.local.as_ref();
            let attrs_ref = attrs.borrow();

            // Compute cascaded style for this element
            let computed = compute_style(handle, &attrs_ref, stylesheet, ancestors, parent_style);

            // Build new ancestors list with current node prepended (for children)
            // Cap at 32 ancestors for selector matching (deeper ancestors rarely matter)
            let mut child_ancestors = vec![handle.clone()];
            let ancestor_limit = ancestors.len().min(31);
            child_ancestors.extend_from_slice(&ancestors[..ancestor_limit]);

            match tag {
                "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                    let level = tag.as_bytes()[1] - b'0';
                    let text = collect_text(handle);
                    if !text.trim().is_empty() {
                        let mut node = OvdNode::new(NodeType::Heading);
                        node.text = text.trim().to_string();
                        node.level = level;
                        node.depth = depth;
                        node.parent_id = parent_id;
                        node.style = computed;
                        if !node.style.display_none {
                            doc.add_node(node);
                        }
                    }
                    return;
                }

                "p" => {
                    let text = collect_text(handle);
                    if !text.trim().is_empty() {
                        let mut node = OvdNode::new(NodeType::Paragraph);
                        node.text = text.trim().to_string();
                        node.depth = depth;
                        node.parent_id = parent_id;
                        node.style = computed;
                        if node.style.display_none {
                            return;
                        }
                        doc.add_node(node);
                    }
                    extract_links(handle, doc, depth + 1, parent_id);
                    return;
                }

                "a" => {
                    let href = attrs_ref
                        .iter()
                        .find(|a| a.name.local.as_ref() == "href")
                        .map(|a| a.value.to_string())
                        .unwrap_or_default();
                    let text = collect_text(handle);
                    if !text.trim().is_empty() && !href.is_empty() {
                        let mut node = OvdNode::new(NodeType::Link);
                        node.text = text.trim().to_string();
                        node.href = href;
                        node.depth = depth;
                        node.parent_id = parent_id;
                        node.style = computed;
                        if !node.style.display_none {
                            doc.add_node(node);
                        }
                    }
                    return;
                }

                "pre" | "code" => {
                    let text = collect_text(handle);
                    if !text.trim().is_empty() {
                        let mut node = OvdNode::new(NodeType::CodeBlock);
                        node.text = text.to_string();
                        node.depth = depth;
                        node.parent_id = parent_id;
                        node.style = computed;
                        if !node.style.display_none {
                            doc.add_node(node);
                        }
                    }
                    return;
                }

                "blockquote" => {
                    let text = collect_text(handle);
                    if !text.trim().is_empty() {
                        let mut node = OvdNode::new(NodeType::Blockquote);
                        node.text = text.trim().to_string();
                        node.depth = depth;
                        node.parent_id = parent_id;
                        node.style = computed;
                        if !node.style.display_none {
                            doc.add_node(node);
                        }
                    }
                    return;
                }

                "ul" | "ol" => {
                    let mut list_node = OvdNode::new(NodeType::List);
                    list_node.depth = depth;
                    list_node.parent_id = parent_id;
                    list_node.style = computed.clone();
                    if list_node.style.display_none {
                        return;
                    }
                    let list_id = doc.add_node(list_node) as i32;

                    for child in handle.children.borrow().iter() {
                        if let NodeData::Element { ref name, ref attrs, .. } = child.data {
                            if name.local.as_ref() == "li" {
                                let text = collect_text(child);
                                if !text.trim().is_empty() {
                                    let li_attrs = attrs.borrow();
                                    let li_style = compute_style(
                                        child,
                                        &li_attrs,
                                        stylesheet,
                                        &child_ancestors,
                                        &computed,
                                    );
                                    let mut item = OvdNode::new(NodeType::ListItem);
                                    item.text = text.trim().to_string();
                                    item.depth = depth + 1;
                                    item.parent_id = list_id;
                                    item.style = li_style;
                                    if !item.style.display_none {
                                        doc.add_node(item);
                                    }
                                }
                                extract_links(child, doc, depth + 2, list_id);
                            }
                        }
                    }
                    return;
                }

                "hr" => {
                    let mut node = OvdNode::new(NodeType::Separator);
                    node.depth = depth;
                    node.parent_id = parent_id;
                    node.style = computed;
                    if !node.style.display_none {
                        doc.add_node(node);
                    }
                    return;
                }

                "img" => {
                    let alt = attrs_ref
                        .iter()
                        .find(|a| a.name.local.as_ref() == "alt")
                        .map(|a| a.value.to_string())
                        .unwrap_or_default();
                    let src = attrs_ref
                        .iter()
                        .find(|a| a.name.local.as_ref() == "src")
                        .map(|a| a.value.to_string())
                        .unwrap_or_default();
                    let mut node = OvdNode::new(NodeType::Image);
                    node.alt = alt;
                    node.src = src;
                    node.depth = depth;
                    node.parent_id = parent_id;
                    node.style = computed;
                    if !node.style.display_none {
                        doc.add_node(node);
                    }
                    return;
                }

                "table" => {
                    let mut table = OvdNode::new(NodeType::Table);
                    table.depth = depth;
                    table.parent_id = parent_id;
                    table.style = computed.clone();
                    if table.style.display_none {
                        return;
                    }
                    let table_id = doc.add_node(table) as i32;
                    // Drop attrs borrow before calling extract_table
                    drop(attrs_ref);
                    extract_table(handle, doc, depth + 1, table_id, stylesheet, &child_ancestors, &computed);
                    return;
                }

                // Skip non-content elements but still recurse into <head>
                // so we don't miss anything (stylesheets are extracted separately)
                "script" | "style" | "noscript" | "svg" | "iframe" | "meta"
                | "link" | "head" | "template" => {
                    return;
                }

                // Container elements: recurse into children
                _ => {
                    // Drop attrs borrow before recursing
                    drop(attrs_ref);
                    for child in handle.children.borrow().iter() {
                        walk_node(child, doc, depth + 1, parent_id, stylesheet, &child_ancestors, &computed);
                    }
                    return;
                }
            }
        }

        NodeData::Text { ref contents } => {
            let text = contents.borrow().to_string();
            let text = text.trim();
            if !text.is_empty() && text.len() > 1 {
                let mut node = OvdNode::new(NodeType::TextSpan);
                node.text = text.to_string();
                node.depth = depth;
                node.parent_id = parent_id;
                // Text nodes inherit parent style
                node.style = parent_style.clone();
                // Text nodes don't have their own background
                node.style.background = None;
                doc.add_node(node);
            }
        }

        _ => {}
    }

    // Recurse into children (for Document, Comment, etc. non-element nodes)
    for child in handle.children.borrow().iter() {
        walk_node(child, doc, depth + 1, parent_id, stylesheet, ancestors, parent_style);
    }
}

/// Recursively search inside a cell's child elements (div, span, center, etc.)
/// for nested tables or links.
fn extract_cell_content(
    handle: &Handle,
    doc: &mut OvdDocument,
    depth: u16,
    parent_id: i32,
    stylesheet: &Stylesheet,
    ancestors: &[Handle],
    parent_style: &OvdStyle,
) {
    for child in handle.children.borrow().iter() {
        if let NodeData::Element { ref name, ref attrs, .. } = child.data {
            match name.local.as_ref() {
                "table" => {
                    let ga = attrs.borrow();
                    let nested_style = compute_style(child, &ga, stylesheet, ancestors, parent_style);
                    drop(ga);
                    let mut nested = OvdNode::new(NodeType::Table);
                    nested.depth = depth;
                    nested.parent_id = parent_id;
                    nested.style = nested_style.clone();
                    let nested_id = doc.add_node(nested) as i32;
                    let mut nested_ancestors = vec![child.clone()];
                    let nlimit = ancestors.len().min(31);
                    nested_ancestors.extend_from_slice(&ancestors[..nlimit]);
                    extract_table(child, doc, depth + 1, nested_id, stylesheet, &nested_ancestors, &nested_style);
                }
                "a" => {
                    let ga = attrs.borrow();
                    let href = ga
                        .iter()
                        .find(|a| a.name.local.as_ref() == "href")
                        .map(|a| a.value.to_string())
                        .unwrap_or_default();
                    drop(ga);
                    let text = collect_text(child);
                    if !text.trim().is_empty() && !href.is_empty() {
                        let mut link = OvdNode::new(NodeType::Link);
                        link.text = text.trim().to_string();
                        link.href = href;
                        link.depth = depth;
                        link.parent_id = parent_id;
                        link.style = parent_style.clone();
                        doc.add_node(link);
                    }
                }
                _ => {
                    // Recurse into containers (div, span, center, etc.)
                    extract_cell_content(child, doc, depth, parent_id, stylesheet, ancestors, parent_style);
                }
            }
        }
    }
}

/// Extract links from inside a container (e.g., inside a <p> or <li>).
fn extract_links(handle: &Handle, doc: &mut OvdDocument, depth: u16, parent_id: i32) {
    for child in handle.children.borrow().iter() {
        if let NodeData::Element { ref name, ref attrs, .. } = child.data {
            if name.local.as_ref() == "a" {
                let attrs = attrs.borrow();
                let href = attrs
                    .iter()
                    .find(|a| a.name.local.as_ref() == "href")
                    .map(|a| a.value.to_string())
                    .unwrap_or_default();
                let text = collect_text(child);
                if !text.trim().is_empty() && !href.is_empty() {
                    let mut node = OvdNode::new(NodeType::Link);
                    node.text = text.trim().to_string();
                    node.href = href;
                    node.depth = depth;
                    node.parent_id = parent_id;
                    doc.add_node(node);
                }
            }
        }
        extract_links(child, doc, depth, parent_id);
    }
}

/// Extract table structure preserving rows and links inside cells.
fn extract_table(
    handle: &Handle,
    doc: &mut OvdDocument,
    depth: u16,
    parent_id: i32,
    stylesheet: &Stylesheet,
    ancestors: &[Handle],
    parent_style: &OvdStyle,
) {
    for child in handle.children.borrow().iter() {
        if let NodeData::Element { ref name, ref attrs, .. } = child.data {
            match name.local.as_ref() {
                "tr" => {
                    let attrs_ref = attrs.borrow();
                    let row_style = compute_style(child, &attrs_ref, stylesheet, ancestors, parent_style);
                    drop(attrs_ref);

                    let mut row = OvdNode::new(NodeType::TableRow);
                    row.depth = depth;
                    row.parent_id = parent_id;
                    row.style = row_style.clone();
                    let row_id = doc.add_node(row) as i32;

                    let mut row_ancestors = vec![child.clone()];
                    let limit = ancestors.len().min(31);
                    row_ancestors.extend_from_slice(&ancestors[..limit]);

                    extract_table(child, doc, depth + 1, row_id, stylesheet, &row_ancestors, &row_style);
                }
                "td" | "th" => {
                    let attrs_ref = attrs.borrow();
                    let cell_style = compute_style(child, &attrs_ref, stylesheet, ancestors, parent_style);
                    drop(attrs_ref);

                    // Check if the cell contains nested tables — if so, recurse
                    let mut has_nested_table = false;
                    let mut has_link = false;

                    let mut cell_ancestors = vec![child.clone()];
                    let limit = ancestors.len().min(31);
                    cell_ancestors.extend_from_slice(&ancestors[..limit]);

                    for grandchild in child.children.borrow().iter() {
                        if let NodeData::Element { ref name, ref attrs, .. } = grandchild.data {
                            match name.local.as_ref() {
                                "table" => {
                                    // Nested table inside cell — recurse into it
                                    let ga = attrs.borrow();
                                    let nested_style = compute_style(grandchild, &ga, stylesheet, &cell_ancestors, &cell_style);
                                    drop(ga);
                                    let mut nested = OvdNode::new(NodeType::Table);
                                    nested.depth = depth;
                                    nested.parent_id = parent_id;
                                    nested.style = nested_style.clone();
                                    let nested_id = doc.add_node(nested) as i32;
                                    let mut nested_ancestors = vec![grandchild.clone()];
                                    let nlimit = cell_ancestors.len().min(31);
                                    nested_ancestors.extend_from_slice(&cell_ancestors[..nlimit]);
                                    extract_table(grandchild, doc, depth + 1, nested_id, stylesheet, &nested_ancestors, &nested_style);
                                    has_nested_table = true;
                                }
                                "a" => {
                                    let ga = attrs.borrow();
                                    let href = ga
                                        .iter()
                                        .find(|a| a.name.local.as_ref() == "href")
                                        .map(|a| a.value.to_string())
                                        .unwrap_or_default();
                                    drop(ga);
                                    let text = collect_text(grandchild);
                                    if !text.trim().is_empty() && !href.is_empty() {
                                        let mut link = OvdNode::new(NodeType::Link);
                                        link.text = text.trim().to_string();
                                        link.href = href;
                                        link.depth = depth;
                                        link.parent_id = parent_id;
                                        link.style = cell_style.clone();
                                        doc.add_node(link);
                                        has_link = true;
                                    }
                                }
                                // Also check for deeper containers (div, span, center, etc.)
                                // that might contain tables or links
                                _ => {
                                    extract_cell_content(grandchild, doc, depth, parent_id, stylesheet, &cell_ancestors, &cell_style);
                                }
                            }
                        }
                    }

                    // If no nested tables or links, emit as table cell with plain text
                    if !has_nested_table && !has_link {
                        let text = collect_text(child);
                        if !text.trim().is_empty() {
                            let mut cell = OvdNode::new(NodeType::TableCell);
                            cell.text = text.trim().to_string();
                            cell.depth = depth;
                            cell.parent_id = parent_id;
                            cell.style = cell_style;
                            doc.add_node(cell);
                        }
                    }
                }
                "thead" | "tbody" | "tfoot" => {
                    // Transparent containers — recurse into them
                    let mut child_ancestors = vec![child.clone()];
                    let limit = ancestors.len().min(31);
                    child_ancestors.extend_from_slice(&ancestors[..limit]);
                    extract_table(child, doc, depth, parent_id, stylesheet, &child_ancestors, parent_style);
                }
                // Nested tables
                "table" => {
                    let attrs_ref = attrs.borrow();
                    let table_style = compute_style(child, &attrs_ref, stylesheet, ancestors, parent_style);
                    drop(attrs_ref);

                    let mut nested = OvdNode::new(NodeType::Table);
                    nested.depth = depth;
                    nested.parent_id = parent_id;
                    nested.style = table_style.clone();
                    let nested_id = doc.add_node(nested) as i32;

                    let mut child_ancestors = vec![child.clone()];
                    let limit = ancestors.len().min(31);
                    child_ancestors.extend_from_slice(&ancestors[..limit]);
                    extract_table(child, doc, depth + 1, nested_id, stylesheet, &child_ancestors, &table_style);
                }
                _ => {
                    extract_table(child, doc, depth, parent_id, stylesheet, ancestors, parent_style);
                }
            }
        }
    }
}

/// Collect all text content from a DOM subtree.
fn collect_text(handle: &Handle) -> String {
    let mut result = String::new();
    collect_text_inner(handle, &mut result);
    // Normalize whitespace
    let normalized: String = result
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ");
    normalized
}

fn collect_text_inner(handle: &Handle, result: &mut String) {
    match &handle.data {
        NodeData::Text { ref contents } => {
            result.push_str(&contents.borrow());
        }
        _ => {}
    }
    for child in handle.children.borrow().iter() {
        collect_text_inner(child, result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_style_attribute() {
        let html = r#"<html><body><p style="color: red; font-size: 20px">Styled text</p></body></html>"#;
        let doc = extract_html(html, "test://");
        let p_node = doc
            .nodes
            .iter()
            .find(|n| n.node_type == NodeType::Paragraph)
            .unwrap();
        assert_eq!(p_node.style.color, Some([255, 0, 0]));
        assert_eq!(p_node.style.font_size_px, Some(20.0));
    }

    #[test]
    fn test_display_none_skipped() {
        let html = r#"<html><body><p style="display: none">Hidden</p><p>Visible</p></body></html>"#;
        let doc = extract_html(html, "test://");
        let paragraphs: Vec<_> = doc
            .nodes
            .iter()
            .filter(|n| n.node_type == NodeType::Paragraph)
            .collect();
        assert_eq!(paragraphs.len(), 1);
        assert_eq!(paragraphs[0].text, "Visible");
    }

    #[test]
    fn test_legacy_bgcolor_attribute() {
        let html = r##"<html><body><table bgcolor="#ff0000"><tr><td>Cell</td></tr></table></body></html>"##;
        let doc = extract_html(html, "test://");
        let table_node = doc
            .nodes
            .iter()
            .find(|n| n.node_type == NodeType::Table)
            .unwrap();
        assert_eq!(table_node.style.background, Some([255, 0, 0]));
    }

    #[test]
    fn test_heading_style_preserved() {
        let html = r#"<html><body><h1 style="color: blue">Title</h1></body></html>"#;
        let doc = extract_html(html, "test://");
        let h_node = doc
            .nodes
            .iter()
            .find(|n| n.node_type == NodeType::Heading)
            .unwrap();
        assert_eq!(h_node.style.color, Some([0, 0, 255]));
    }

    // --- Stylesheet extraction tests ---

    #[test]
    fn test_extract_stylesheets_single() {
        let html = r#"<html><head><style>h1 { color: red; }</style></head><body><p>Text</p></body></html>"#;
        let dom = parse_html(html);
        let sheets = extract_stylesheets(&dom.document);
        assert_eq!(sheets.len(), 1);
        assert!(sheets[0].contains("color: red"));
    }

    #[test]
    fn test_extract_stylesheets_multiple() {
        let html = r#"<html><head>
            <style>h1 { color: red; }</style>
            <style>p { font-size: 14px; }</style>
        </head><body><p>Text</p></body></html>"#;
        let dom = parse_html(html);
        let sheets = extract_stylesheets(&dom.document);
        assert_eq!(sheets.len(), 2);
        assert!(sheets[0].contains("color: red"));
        assert!(sheets[1].contains("font-size: 14px"));
    }

    #[test]
    fn test_extract_stylesheets_empty() {
        let html = r#"<html><body><p>No styles</p></body></html>"#;
        let dom = parse_html(html);
        let sheets = extract_stylesheets(&dom.document);
        assert_eq!(sheets.len(), 0);
    }

    #[test]
    fn test_extract_stylesheet_urls_basic() {
        let html = r#"<html><head>
            <link rel="stylesheet" href="/style.css">
            <link rel="stylesheet" href="theme.css">
        </head><body></body></html>"#;
        let dom = parse_html(html);
        let urls = extract_stylesheet_urls(&dom.document, "https://example.com/page.html");
        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0], "https://example.com/style.css");
        assert_eq!(urls[1], "https://example.com/theme.css");
    }

    #[test]
    fn test_extract_stylesheet_urls_absolute() {
        let html = r#"<html><head>
            <link rel="stylesheet" href="https://cdn.example.com/style.css">
        </head><body></body></html>"#;
        let dom = parse_html(html);
        let urls = extract_stylesheet_urls(&dom.document, "https://example.com/");
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0], "https://cdn.example.com/style.css");
    }

    #[test]
    fn test_extract_stylesheet_urls_ignores_non_stylesheet() {
        let html = r#"<html><head>
            <link rel="icon" href="/favicon.ico">
            <link rel="stylesheet" href="/style.css">
        </head><body></body></html>"#;
        let dom = parse_html(html);
        let urls = extract_stylesheet_urls(&dom.document, "https://example.com/");
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0], "https://example.com/style.css");
    }

    // --- Stylesheet cascade tests ---

    #[test]
    fn test_stylesheet_cascade_tag_selector() {
        let html = r#"<html><head><style>
            h1 { color: red; }
            p { font-size: 14px; }
        </style></head><body>
            <h1>Red Heading</h1>
            <p>Default paragraph</p>
        </body></html>"#;
        let doc = extract_html_with_stylesheet(
            html,
            "test://",
            &Stylesheet::parse("h1 { color: red; } p { font-size: 14px; }"),
        );
        let h1 = doc.nodes.iter().find(|n| n.node_type == NodeType::Heading).unwrap();
        assert_eq!(h1.style.color, Some([255, 0, 0]));

        let p = doc.nodes.iter().find(|n| n.node_type == NodeType::Paragraph).unwrap();
        assert_eq!(p.style.font_size_px, Some(14.0));
    }

    #[test]
    fn test_stylesheet_cascade_class_over_tag() {
        let html = r#"<html><body>
            <p class="blue">Blue text</p>
            <p>Default</p>
        </body></html>"#;
        let sheet = Stylesheet::parse("p { color: red; } .blue { color: blue; }");
        let doc = extract_html_with_stylesheet(html, "test://", &sheet);

        let paragraphs: Vec<_> = doc.nodes.iter()
            .filter(|n| n.node_type == NodeType::Paragraph)
            .collect();
        assert_eq!(paragraphs.len(), 2);
        // .blue class wins over p tag selector
        assert_eq!(paragraphs[0].style.color, Some([0, 0, 255]));
        // Plain p gets tag selector color
        assert_eq!(paragraphs[1].style.color, Some([255, 0, 0]));
    }

    #[test]
    fn test_stylesheet_cascade_id_over_class() {
        let html = r#"<html><body>
            <p id="special" class="blue">Special text</p>
        </body></html>"#;
        let sheet = Stylesheet::parse(".blue { color: blue; } #special { color: green; }");
        let doc = extract_html_with_stylesheet(html, "test://", &sheet);

        let p = doc.nodes.iter().find(|n| n.node_type == NodeType::Paragraph).unwrap();
        // #id has higher specificity than .class
        assert_eq!(p.style.color, Some([0, 128, 0]));
    }

    #[test]
    fn test_inline_style_over_id() {
        let html = r#"<html><body>
            <p id="special" style="color: orange">Inline wins</p>
        </body></html>"#;
        let sheet = Stylesheet::parse("#special { color: green; }");
        let doc = extract_html_with_stylesheet(html, "test://", &sheet);

        let p = doc.nodes.iter().find(|n| n.node_type == NodeType::Paragraph).unwrap();
        // Inline style has highest specificity
        assert_eq!(p.style.color, Some([255, 165, 0]));
    }

    #[test]
    fn test_style_inheritance_color() {
        let html = r#"<html><body>
            <div><h1>Inherited</h1></div>
        </body></html>"#;
        let sheet = Stylesheet::parse("div { color: navy; }");
        let doc = extract_html_with_stylesheet(html, "test://", &sheet);

        let h1 = doc.nodes.iter().find(|n| n.node_type == NodeType::Heading).unwrap();
        // h1 inherits color from parent div
        assert_eq!(h1.style.color, Some([0, 0, 128]));
    }

    #[test]
    fn test_background_does_not_inherit() {
        let html = r#"<html><body>
            <div><p>Child</p></div>
        </body></html>"#;
        let sheet = Stylesheet::parse("div { background: red; }");
        let doc = extract_html_with_stylesheet(html, "test://", &sheet);

        let p = doc.nodes.iter().find(|n| n.node_type == NodeType::Paragraph).unwrap();
        // background does NOT inherit
        assert_eq!(p.style.background, None);
    }

    #[test]
    fn test_stylesheet_display_none() {
        let html = r#"<html><body>
            <p>Visible</p>
            <p class="hidden">Hidden</p>
        </body></html>"#;
        let sheet = Stylesheet::parse(".hidden { display: none; }");
        let doc = extract_html_with_stylesheet(html, "test://", &sheet);

        let paragraphs: Vec<_> = doc.nodes.iter()
            .filter(|n| n.node_type == NodeType::Paragraph)
            .collect();
        assert_eq!(paragraphs.len(), 1);
        assert_eq!(paragraphs[0].text, "Visible");
    }

    #[test]
    fn test_stylesheet_font_weight_bold() {
        let html = r#"<html><body>
            <h1>Bold heading</h1>
        </body></html>"#;
        let sheet = Stylesheet::parse("h1 { font-weight: bold; }");
        let doc = extract_html_with_stylesheet(html, "test://", &sheet);

        let h1 = doc.nodes.iter().find(|n| n.node_type == NodeType::Heading).unwrap();
        assert_eq!(h1.style.font_weight, Some(700));
    }

    #[test]
    fn test_stylesheet_with_extracted_style_block() {
        // End-to-end: extract_html parses style blocks automatically
        // via extract_html (which uses empty stylesheet), but
        // extract_html_with_stylesheet applies external CSS.
        let html = r#"<html><body>
            <p class="highlight">Highlighted</p>
        </body></html>"#;
        let sheet = Stylesheet::parse(".highlight { color: yellow; font-size: 18px; }");
        let doc = extract_html_with_stylesheet(html, "test://", &sheet);

        let p = doc.nodes.iter().find(|n| n.node_type == NodeType::Paragraph).unwrap();
        assert_eq!(p.style.color, Some([255, 255, 0]));
        assert_eq!(p.style.font_size_px, Some(18.0));
    }

    #[test]
    fn test_stylesheet_box_model() {
        let html = r#"<html><body>
            <p class="spaced">Spaced text</p>
        </body></html>"#;
        let sheet = Stylesheet::parse(".spaced { margin: 20px; padding: 10px 15px; }");
        let doc = extract_html_with_stylesheet(html, "test://", &sheet);

        let p = doc.nodes.iter().find(|n| n.node_type == NodeType::Paragraph).unwrap();
        assert_eq!(p.style.margin_top, Some(20.0));
        assert_eq!(p.style.margin_bottom, Some(20.0));
        assert_eq!(p.style.padding_top, Some(10.0));
        assert_eq!(p.style.padding_left, Some(15.0));
    }

    #[test]
    fn test_stylesheet_width() {
        let html = r#"<html><body>
            <p class="narrow">Narrow</p>
        </body></html>"#;
        let sheet = Stylesheet::parse(".narrow { width: 200px; max-width: 400px; }");
        let doc = extract_html_with_stylesheet(html, "test://", &sheet);

        let p = doc.nodes.iter().find(|n| n.node_type == NodeType::Paragraph).unwrap();
        assert_eq!(p.style.width, Some(200.0));
        assert_eq!(p.style.max_width, Some(400.0));
    }

    #[test]
    fn test_stylesheet_text_decoration() {
        let html = r#"<html><body>
            <p class="underline">Underlined</p>
        </body></html>"#;
        let sheet = Stylesheet::parse(".underline { text-decoration: underline; }");
        let doc = extract_html_with_stylesheet(html, "test://", &sheet);

        let p = doc.nodes.iter().find(|n| n.node_type == NodeType::Paragraph).unwrap();
        assert!(p.style.text_decoration_underline);
    }

    #[test]
    fn test_specificity_order_tag_class_id_inline() {
        let html = r#"<html><body>
            <p id="x" class="y" style="color: white">Test</p>
        </body></html>"#;
        let sheet = Stylesheet::parse(
            "p { color: red; } .y { color: green; } #x { color: blue; }"
        );
        let doc = extract_html_with_stylesheet(html, "test://", &sheet);

        let p = doc.nodes.iter().find(|n| n.node_type == NodeType::Paragraph).unwrap();
        // Inline style wins over all
        assert_eq!(p.style.color, Some([255, 255, 255]));
    }

    // --- Script extraction tests ---

    #[test]
    fn test_extract_scripts_single() {
        let html = r#"<html><head><script>console.log("hello");</script></head><body></body></html>"#;
        let dom = parse_html(html);
        let scripts = extract_scripts(&dom.document);
        assert_eq!(scripts.len(), 1);
        assert!(scripts[0].contains("console.log"));
    }

    #[test]
    fn test_extract_scripts_multiple() {
        let html = r#"<html><head>
            <script>var x = 1;</script>
        </head><body>
            <script>var y = 2;</script>
        </body></html>"#;
        let dom = parse_html(html);
        let scripts = extract_scripts(&dom.document);
        assert_eq!(scripts.len(), 2);
        assert!(scripts[0].contains("var x"));
        assert!(scripts[1].contains("var y"));
    }

    #[test]
    fn test_extract_scripts_skips_external() {
        let html = r#"<html><head>
            <script src="https://example.com/app.js"></script>
            <script>var inline = true;</script>
        </head><body></body></html>"#;
        let dom = parse_html(html);
        let scripts = extract_scripts(&dom.document);
        // External script has no inline body, so only the inline one is extracted
        assert_eq!(scripts.len(), 1);
        assert!(scripts[0].contains("var inline"));
    }

    #[test]
    fn test_extract_scripts_empty() {
        let html = r#"<html><body><p>No scripts</p></body></html>"#;
        let dom = parse_html(html);
        let scripts = extract_scripts(&dom.document);
        assert_eq!(scripts.len(), 0);
    }
}
