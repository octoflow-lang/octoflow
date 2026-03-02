use crate::extract;
use crate::fetch;
use crate::flow_mode;
use crate::image_loader::ImageCache;
use crate::js_engine::{self, DomMutation, StoredListener, TimerEntry};
use crate::layout::{self, LayoutBox};
use crate::ovd::{NodeType, OvdDocument, OvdNode};
use crate::stylesheet::Stylesheet;
use crate::text::TextRenderer;

/// The result of loading and laying out a page.
#[allow(dead_code)]
pub struct PageState {
    pub url: String,
    pub title: String,
    pub doc: OvdDocument,
    pub layout: Vec<LayoutBox>,
    pub content_height: f32,
    pub node_count: usize,
    pub load_time_ms: u128,
    /// Event listeners registered by JS (available for future event dispatch).
    pub listeners: Vec<StoredListener>,
    /// Pending timers registered by JS (non-zero delay; delay-0 already executed).
    pub timers: Vec<TimerEntry>,
}

/// Pre-layout page data (can be produced on a background thread).
/// Contains the OVD document and metadata; layout happens on the main thread.
pub struct PageData {
    pub url: String,
    pub title: String,
    pub doc: OvdDocument,
    pub node_count: usize,
    pub fetch_time_ms: u128,
    pub is_flow: bool,
    /// Event listeners registered by JS.
    pub listeners: Vec<StoredListener>,
    /// Pending timers registered by JS.
    pub timers: Vec<TimerEntry>,
}

/// Fetch, parse, extract, and run JS — everything except layout.
/// This is `Send` and can run on a background thread.
pub fn load_page_data(url: &str) -> Result<PageData, String> {
    let start = std::time::Instant::now();

    // Fetch HTML
    let t0 = std::time::Instant::now();
    let result = fetch::fetch_url(url)?;
    eprintln!("[perf] fetch: {}ms", t0.elapsed().as_millis());

    // Parse DOM
    let t1 = std::time::Instant::now();
    let dom = extract::parse_html(&result.html);
    eprintln!("[perf] parse: {}ms", t1.elapsed().as_millis());

    // Extract <style> block content
    let style_texts = extract::extract_stylesheets(&dom.document);

    // Extract <link rel="stylesheet"> URLs
    let css_urls = extract::extract_stylesheet_urls(&dom.document, &result.url);

    // Fetch external stylesheets (cap at 3, 2s timeout)
    let t2 = std::time::Instant::now();
    let mut all_css = String::new();
    for css_url in css_urls.iter().take(3) {
        match fetch::fetch_css(css_url) {
            Ok(css_text) => {
                all_css.push_str(&css_text);
                all_css.push('\n');
            }
            Err(_) => {}
        }
    }
    eprintln!("[perf] css fetch ({}): {}ms", css_urls.len().min(3), t2.elapsed().as_millis());

    // Append inline <style> blocks
    for style_text in &style_texts {
        all_css.push_str(style_text);
        all_css.push('\n');
    }

    // Parse combined CSS
    let t3 = std::time::Instant::now();
    let stylesheet = Stylesheet::parse(&all_css);
    eprintln!("[perf] css parse ({} rules): {}ms", stylesheet.rules.len(), t3.elapsed().as_millis());

    // Extract OVD from DOM with stylesheet
    let t4 = std::time::Instant::now();
    let mut doc = extract::extract_from_dom(&dom, &result.url, &stylesheet);
    eprintln!("[perf] extract: {}ms ({} nodes)", t4.elapsed().as_millis(), doc.nodes.len());

    // Extract and execute JS
    let scripts = extract::extract_scripts(&dom.document);
    let mut listeners = Vec::new();
    let mut timers = Vec::new();
    if !scripts.is_empty() {
        let t5 = std::time::Instant::now();
        let js_result = js_engine::execute_scripts(&scripts);
        eprintln!("[perf] js ({} scripts, {} listeners, {} timers): {}ms",
            scripts.len(), js_result.listeners.len(), js_result.timers.len(),
            t5.elapsed().as_millis());
        apply_mutations(&mut doc, &js_result.mutations);
        listeners = js_result.listeners;
        timers = js_result.timers;
        for err in &js_result.errors {
            eprintln!("[JS error] {err}");
        }
    }

    let node_count = doc.nodes.len();
    let title = doc.title.clone();

    Ok(PageData {
        url: result.url,
        title,
        doc,
        node_count,
        fetch_time_ms: start.elapsed().as_millis(),
        is_flow: false,
        listeners,
        timers,
    })
}

/// Load a .flow file data (no layout). Can run on background thread.
pub fn load_flow_data(url: &str) -> Result<PageData, String> {
    let start = std::time::Instant::now();
    let path = if let Some(stripped) = url.strip_prefix("flow://") {
        stripped
    } else {
        url
    };
    let source = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path, e))?;
    let doc = flow_mode::flow_to_ovd(&source, path);
    let node_count = doc.nodes.len();
    let title = doc.title.clone();
    Ok(PageData {
        url: url.to_string(),
        title,
        doc,
        node_count,
        fetch_time_ms: start.elapsed().as_millis(),
        is_flow: true,
        listeners: Vec::new(),
        timers: Vec::new(),
    })
}

/// Finish a PageData into a PageState by running layout on the main thread.
pub fn layout_page_data(
    data: PageData,
    viewport_width: f32,
    text: &mut TextRenderer,
    image_cache: &mut ImageCache,
) -> PageState {
    let boxes = layout::layout_document(&data.doc, text, viewport_width, image_cache);
    let content_height = layout::content_height(&boxes);
    PageState {
        url: data.url,
        title: data.title,
        doc: data.doc,
        layout: boxes,
        content_height,
        node_count: data.node_count,
        load_time_ms: data.fetch_time_ms,
        listeners: data.listeners,
        timers: data.timers,
    }
}

/// Fetch HTML, extract OVD, execute JS, layout -- the full pipeline.
///
/// Pipeline steps:
/// 1. Fetch HTML
/// 2. Parse DOM with html5ever
/// 3. Extract `<style>` blocks and `<link rel="stylesheet">` URLs
/// 4. Fetch external stylesheets (error-tolerant)
/// 5. Combine all CSS into a single Stylesheet
/// 6. Extract OVD with stylesheet applied (cascade + inheritance)
/// 7. Extract inline `<script>` tags and execute via Boa JS engine
/// 8. Apply DOM mutations to OVD document
/// 9. Layout
pub fn load_page(
    url: &str,
    viewport_width: f32,
    text: &mut TextRenderer,
    image_cache: &mut ImageCache,
) -> Result<PageState, String> {
    let start = std::time::Instant::now();

    // Fetch HTML
    let t0 = std::time::Instant::now();
    let result = fetch::fetch_url(url)?;
    eprintln!("[perf] fetch: {}ms", t0.elapsed().as_millis());

    // Parse DOM
    let t1 = std::time::Instant::now();
    let dom = extract::parse_html(&result.html);
    eprintln!("[perf] parse: {}ms", t1.elapsed().as_millis());

    // Extract <style> block content
    let style_texts = extract::extract_stylesheets(&dom.document);

    // Extract <link rel="stylesheet"> URLs
    let css_urls = extract::extract_stylesheet_urls(&dom.document, &result.url);

    // Fetch external stylesheets (error-tolerant: skip failures)
    // Cap at 3 external sheets with 2s timeout to avoid stalling
    let t2 = std::time::Instant::now();
    let mut all_css = String::new();
    for css_url in css_urls.iter().take(3) {
        match fetch::fetch_css(css_url) {
            Ok(css_text) => {
                all_css.push_str(&css_text);
                all_css.push('\n');
            }
            Err(_) => {
                // Silently skip failed stylesheet fetches
            }
        }
    }
    eprintln!("[perf] css fetch ({}): {}ms", css_urls.len().min(3), t2.elapsed().as_millis());

    // Append inline <style> blocks
    for style_text in &style_texts {
        all_css.push_str(style_text);
        all_css.push('\n');
    }

    // Parse combined CSS into a Stylesheet
    let t3 = std::time::Instant::now();
    let stylesheet = Stylesheet::parse(&all_css);
    eprintln!("[perf] css parse ({} rules): {}ms", stylesheet.rules.len(), t3.elapsed().as_millis());

    // Extract OVD from DOM with stylesheet applied
    let t4 = std::time::Instant::now();
    let mut doc = extract::extract_from_dom(&dom, &result.url, &stylesheet);
    eprintln!("[perf] extract: {}ms ({} nodes)", t4.elapsed().as_millis(), doc.nodes.len());

    // Extract inline <script> tags and execute via Boa JS engine
    let scripts = extract::extract_scripts(&dom.document);
    let mut listeners = Vec::new();
    let mut timers = Vec::new();
    if !scripts.is_empty() {
        let t5 = std::time::Instant::now();
        let js_result = js_engine::execute_scripts(&scripts);
        eprintln!("[perf] js ({} scripts, {} listeners, {} timers): {}ms",
            scripts.len(), js_result.listeners.len(), js_result.timers.len(),
            t5.elapsed().as_millis());

        // Print console.log messages and apply mutations
        apply_mutations(&mut doc, &js_result.mutations);
        listeners = js_result.listeners;
        timers = js_result.timers;

        // Report JS errors to stderr
        for err in &js_result.errors {
            eprintln!("[JS error] {err}");
        }
    }

    let node_count = doc.nodes.len();
    let title = doc.title.clone();

    // Layout
    let boxes = layout::layout_document(&doc, text, viewport_width, image_cache);
    let content_height = layout::content_height(&boxes);

    let load_time_ms = start.elapsed().as_millis();

    Ok(PageState {
        url: result.url,
        title,
        doc,
        layout: boxes,
        content_height,
        node_count,
        load_time_ms,
        listeners,
        timers,
    })
}

/// Create a built-in welcome page (no network needed).
pub fn welcome_page(viewport_width: f32, text: &mut TextRenderer, image_cache: &mut ImageCache) -> PageState {
    use crate::ovd::{NodeType, OvdNode};

    let mut doc = OvdDocument::new("octoview://welcome");
    doc.title = "OctoView Browser".to_string();

    let mut page = OvdNode::new(NodeType::Page);
    page.text = "welcome".to_string();
    doc.add_node(page);

    let mut h1 = OvdNode::new(NodeType::Heading);
    h1.text = "OctoView Browser v0.1".to_string();
    h1.level = 1;
    doc.add_node(h1);

    let mut p1 = OvdNode::new(NodeType::Paragraph);
    p1.text = "The web, rewoven. GPU-native browser for developers.".to_string();
    doc.add_node(p1);

    doc.add_node(OvdNode::new(NodeType::Separator));

    let mut h2 = OvdNode::new(NodeType::Heading);
    h2.text = "Getting Started".to_string();
    h2.level = 2;
    doc.add_node(h2);

    let mut p2 = OvdNode::new(NodeType::Paragraph);
    p2.text = "Type a URL in the address bar (Ctrl+L) and press Enter to navigate.".to_string();
    doc.add_node(p2);

    let mut h3 = OvdNode::new(NodeType::Heading);
    h3.text = "Keyboard Shortcuts".to_string();
    h3.level = 2;
    doc.add_node(h3);

    let shortcuts = [
        "Ctrl+L       Focus URL bar",
        "Enter        Navigate to URL",
        "Escape       Unfocus URL bar",
        "F5           Reload page",
        "Alt+Left     Back",
        "Alt+Right    Forward",
        "Space        Scroll down",
        "PageUp/Down  Scroll page",
        "Home/End     Top/Bottom of page",
    ];
    for s in &shortcuts {
        let mut item = OvdNode::new(NodeType::ListItem);
        item.text = s.to_string();
        doc.add_node(item);
    }

    doc.add_node(OvdNode::new(NodeType::Separator));

    let mut p3 = OvdNode::new(NodeType::Paragraph);
    p3.text = "OctoView renders HTML through the OVD (OctoView Document) pipeline. Content is fetched, parsed, extracted to semantic nodes, laid out, and rendered to a GPU surface. No CSS engine, no JavaScript runtime — just structured content on pixels.".to_string();
    doc.add_node(p3);

    let mut code = OvdNode::new(NodeType::CodeBlock);
    code.text = "URL -> fetch HTML -> html5ever parse -> OVD extract -> layout -> render -> GPU".to_string();
    doc.add_node(code);

    let node_count = doc.nodes.len();
    let boxes = layout::layout_document(&doc, text, viewport_width, image_cache);
    let content_height = layout::content_height(&boxes);

    PageState {
        url: "octoview://welcome".to_string(),
        title: "OctoView Browser".to_string(),
        doc,
        layout: boxes,
        content_height,
        node_count,
        load_time_ms: 0,
        listeners: Vec::new(),
        timers: Vec::new(),
    }
}

/// Load a .flow file and render it as a syntax view in Flow Mode.
/// Returns a PageState with BrowseMode::Flow semantics.
pub fn load_flow_page(
    url: &str,
    viewport_width: f32,
    text: &mut TextRenderer,
    image_cache: &mut ImageCache,
) -> Result<PageState, String> {
    let start = std::time::Instant::now();

    // Resolve the file path from the URL
    let path = if let Some(stripped) = url.strip_prefix("flow://") {
        stripped
    } else {
        url
    };

    // Read the .flow source file
    let source = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path, e))?;

    // Create OVD document from .flow source
    let doc = flow_mode::flow_to_ovd(&source, path);
    let node_count = doc.nodes.len();
    let title = doc.title.clone();

    // Layout
    let boxes = layout::layout_document(&doc, text, viewport_width, image_cache);
    let content_height = layout::content_height(&boxes);

    let load_time_ms = start.elapsed().as_millis();

    Ok(PageState {
        url: url.to_string(),
        title,
        doc,
        layout: boxes,
        content_height,
        node_count,
        load_time_ms,
        listeners: Vec::new(),
        timers: Vec::new(),
    })
}

/// Apply JS DOM mutations to an OVD document.
///
/// Handles:
/// - `ConsoleLog` -- prints to stderr
/// - `DocumentWrite` -- parses fragment and appends nodes
/// - `CreateElement` -- adds a new node with the given tag/ID
/// - `AppendChild` -- skipped (OVD is flat, no parent-child hierarchy yet)
/// - `CreateTextNode` -- adds a paragraph node with text content
/// - `SetAttribute` -- updates matching node attributes (src, href, alt)
/// - Element-specific mutations (SetInnerHtml, etc.) need an ID-based lookup
///   (planned for a future phase with element index).
fn apply_mutations(doc: &mut OvdDocument, mutations: &[DomMutation]) {
    for mutation in mutations {
        match mutation {
            DomMutation::ConsoleLog { message } => {
                eprintln!("[JS console.log] {message}");
            }
            DomMutation::DocumentWrite { html } => {
                // Parse the HTML fragment and extract nodes, then append to doc.
                let fragment_doc = extract::extract_html(html, "");
                // Skip the first node (Page root of the fragment) to avoid duplicating it.
                for node in fragment_doc.nodes.into_iter().skip(1) {
                    doc.add_node(node);
                }
            }
            DomMutation::CreateElement { tag, id } => {
                // Map tag name to a suitable OVD NodeType.
                let node_type = tag_to_node_type(tag);
                let mut node = OvdNode::new(node_type);
                node.text = id.clone(); // Store generated ID in text for reference.
                doc.add_node(node);
            }
            DomMutation::AppendChild { parent_id: _, child_id: _ } => {
                // OVD is currently flat -- skip parent-child attachment.
                // Nodes already exist in the document from CreateElement.
            }
            DomMutation::CreateTextNode { text, id: _ } => {
                let mut node = OvdNode::new(NodeType::Paragraph);
                node.text = text.clone();
                doc.add_node(node);
            }
            DomMutation::SetAttribute { id, name, value } => {
                // Find the node by scanning for matching text (ID stored in text field).
                // This is a simple linear scan; a future phase will add an ID index.
                for node in &mut doc.nodes {
                    if node.text == *id || format!("__ov_el_{}", node.node_id) == *id {
                        match name.as_str() {
                            "src" => node.src = value.clone(),
                            "href" => node.href = value.clone(),
                            "alt" => node.alt = value.clone(),
                            _ => {} // Other attributes not yet mapped to OVD fields.
                        }
                        break;
                    }
                }
            }
            // Element-specific mutations (SetInnerHtml, SetTextContent, AddClass,
            // RemoveClass, SetStyle, Hide, Show) need an ID-based lookup.
            // Silently skip for now.
            _ => {}
        }
    }
}

/// Map an HTML tag name to an OVD NodeType.
fn tag_to_node_type(tag: &str) -> NodeType {
    match tag.to_lowercase().as_str() {
        "div" | "section" | "article" | "main" => NodeType::Section,
        "p" => NodeType::Paragraph,
        "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => NodeType::Heading,
        "a" => NodeType::Link,
        "img" => NodeType::Image,
        "span" | "em" | "strong" | "b" | "i" => NodeType::TextSpan,
        "ul" | "ol" => NodeType::List,
        "li" => NodeType::ListItem,
        "table" => NodeType::Table,
        "tr" => NodeType::TableRow,
        "td" | "th" => NodeType::TableCell,
        "form" => NodeType::Form,
        "input" | "textarea" | "select" => NodeType::InputField,
        "button" => NodeType::Button,
        "nav" => NodeType::Navigation,
        "header" => NodeType::Header,
        "footer" => NodeType::Footer,
        "code" | "pre" => NodeType::CodeBlock,
        "blockquote" => NodeType::Blockquote,
        "hr" => NodeType::Separator,
        _ => NodeType::Section, // Default to Section for unknown tags.
    }
}
