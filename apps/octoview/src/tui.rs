//! TUI browser — interactive terminal-based OctoView Document viewer.
//!
//! Features:
//!   - Scroll through pages rendered as semantic blocks
//!   - Color-coded node types (headings bold, links blue, etc.)
//!   - Navigate links with Tab/Enter
//!   - Search within page (/)
//!   - Back/forward navigation history
//!   - Status bar with page info

use crate::ovd::{NodeType, OvdDocument, SemanticRole};
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame, Terminal,
};
use std::io;

/// A rendered line of content with styling.
struct RenderLine {
    spans: Vec<StyledSpan>,
    node_idx: Option<usize>, // Index into OvdDocument.nodes if this line is a link
    is_link: bool,
}

struct StyledSpan {
    text: String,
    fg: Color,
    bg: Color,
    bold: bool,
    italic: bool,
    underline: bool,
}

/// TUI state.
pub struct TuiBrowser {
    doc: OvdDocument,
    lines: Vec<RenderLine>,
    scroll: usize,         // Current scroll offset (line)
    selected_link: usize,  // Index into link_lines
    link_lines: Vec<usize>, // Line indices that are navigable links
    search_query: String,
    search_mode: bool,
    search_results: Vec<usize>, // Line indices matching search
    search_idx: usize,     // Current search result
    status_msg: String,
    history: Vec<String>,  // URL history for back navigation
    history_idx: usize,
}

impl TuiBrowser {
    pub fn new(doc: OvdDocument) -> Self {
        let lines = render_document(&doc);
        let link_lines: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| l.is_link)
            .map(|(i, _)| i)
            .collect();

        let url = doc.url.clone();
        Self {
            doc,
            lines,
            scroll: 0,
            selected_link: 0,
            link_lines,
            search_query: String::new(),
            search_mode: false,
            search_results: Vec::new(),
            search_idx: 0,
            status_msg: String::new(),
            history: vec![url],
            history_idx: 0,
        }
    }

    /// Replace the current document (after navigation).
    pub fn load_document(&mut self, doc: OvdDocument) {
        let url = doc.url.clone();
        self.lines = render_document(&doc);
        self.link_lines = self.lines
            .iter()
            .enumerate()
            .filter(|(_, l)| l.is_link)
            .map(|(i, _)| i)
            .collect();
        self.doc = doc;
        self.scroll = 0;
        self.selected_link = 0;
        self.search_query.clear();
        self.search_results.clear();
        self.search_idx = 0;

        // Update history
        self.history_idx += 1;
        self.history.truncate(self.history_idx);
        self.history.push(url);
    }

    /// Get the href of the currently selected link, if any.
    fn selected_href(&self) -> Option<String> {
        if self.link_lines.is_empty() {
            return None;
        }
        let line_idx = self.link_lines.get(self.selected_link)?;
        let line = self.lines.get(*line_idx)?;
        let node_idx = line.node_idx?;
        let node = self.doc.nodes.get(node_idx)?;
        if node.href.is_empty() {
            None
        } else {
            Some(node.href.clone())
        }
    }

    /// Get the previous URL in history.
    fn go_back(&mut self) -> Option<String> {
        if self.history_idx > 0 {
            self.history_idx -= 1;
            Some(self.history[self.history_idx].clone())
        } else {
            None
        }
    }
}

/// Run the interactive TUI browser.
pub fn run_tui(doc: OvdDocument, load_fn: impl Fn(&str) -> Result<OvdDocument, String>) -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let mut browser = TuiBrowser::new(doc);

    loop {
        terminal.draw(|f| draw_ui(f, &browser))?;

        if let Event::Key(key) = event::read()? {
            if browser.search_mode {
                match key.code {
                    KeyCode::Esc => {
                        browser.search_mode = false;
                        browser.status_msg.clear();
                    }
                    KeyCode::Enter => {
                        browser.search_mode = false;
                        // Execute search
                        let query = browser.search_query.to_lowercase();
                        browser.search_results = browser
                            .lines
                            .iter()
                            .enumerate()
                            .filter(|(_, line)| {
                                line.spans.iter().any(|s| s.text.to_lowercase().contains(&query))
                            })
                            .map(|(i, _)| i)
                            .collect();
                        browser.search_idx = 0;
                        if !browser.search_results.is_empty() {
                            browser.scroll = browser.search_results[0];
                            browser.status_msg = format!(
                                "Found {} matches for '{}'  (n/N to navigate)",
                                browser.search_results.len(),
                                browser.search_query
                            );
                        } else {
                            browser.status_msg = format!("No matches for '{}'", browser.search_query);
                        }
                    }
                    KeyCode::Backspace => {
                        browser.search_query.pop();
                    }
                    KeyCode::Char(c) => {
                        browser.search_query.push(c);
                    }
                    _ => {}
                }
                continue;
            }

            match key.code {
                // Quit
                KeyCode::Char('q') | KeyCode::Esc => break,
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break,

                // Scroll
                KeyCode::Down | KeyCode::Char('j') => {
                    if browser.scroll < browser.lines.len().saturating_sub(1) {
                        browser.scroll += 1;
                    }
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    browser.scroll = browser.scroll.saturating_sub(1);
                }
                KeyCode::PageDown | KeyCode::Char(' ') => {
                    let page_size = terminal.size()?.height as usize - 4;
                    browser.scroll = (browser.scroll + page_size)
                        .min(browser.lines.len().saturating_sub(1));
                }
                KeyCode::PageUp => {
                    let page_size = terminal.size()?.height as usize - 4;
                    browser.scroll = browser.scroll.saturating_sub(page_size);
                }
                KeyCode::Home | KeyCode::Char('g') => {
                    browser.scroll = 0;
                }
                KeyCode::End | KeyCode::Char('G') => {
                    browser.scroll = browser.lines.len().saturating_sub(1);
                }

                // Tab navigation between links
                KeyCode::Tab => {
                    if !browser.link_lines.is_empty() {
                        browser.selected_link = (browser.selected_link + 1) % browser.link_lines.len();
                        browser.scroll = browser.link_lines[browser.selected_link];
                        let href = browser.selected_href().unwrap_or_default();
                        browser.status_msg = format!("Link: {}", href);
                    }
                }
                KeyCode::BackTab => {
                    if !browser.link_lines.is_empty() {
                        if browser.selected_link == 0 {
                            browser.selected_link = browser.link_lines.len() - 1;
                        } else {
                            browser.selected_link -= 1;
                        }
                        browser.scroll = browser.link_lines[browser.selected_link];
                        let href = browser.selected_href().unwrap_or_default();
                        browser.status_msg = format!("Link: {}", href);
                    }
                }

                // Follow link
                KeyCode::Enter => {
                    if let Some(href) = browser.selected_href() {
                        let resolved = resolve_url(&browser.doc.url, &href);
                        browser.status_msg = format!("Loading {}...", &resolved);
                        terminal.draw(|f| draw_ui(f, &browser))?;

                        match load_fn(&resolved) {
                            Ok(new_doc) => {
                                browser.load_document(new_doc);
                                browser.status_msg = format!(
                                    "Loaded: {} ({} nodes)",
                                    browser.doc.url,
                                    browser.doc.nodes.len()
                                );
                            }
                            Err(e) => {
                                browser.status_msg = format!("Error: {}", e);
                            }
                        }
                    }
                }

                // Back
                KeyCode::Backspace | KeyCode::Char('b') => {
                    if let Some(url) = browser.go_back() {
                        browser.status_msg = format!("Back to {}...", &url);
                        terminal.draw(|f| draw_ui(f, &browser))?;

                        match load_fn(&url) {
                            Ok(new_doc) => {
                                browser.lines = render_document(&new_doc);
                                browser.link_lines = browser.lines
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, l)| l.is_link)
                                    .map(|(i, _)| i)
                                    .collect();
                                browser.doc = new_doc;
                                browser.scroll = 0;
                                browser.selected_link = 0;
                                browser.status_msg = format!("Back: {}", browser.doc.url);
                            }
                            Err(e) => {
                                browser.status_msg = format!("Error: {}", e);
                                browser.history_idx += 1; // Undo the go_back
                            }
                        }
                    }
                }

                // Search
                KeyCode::Char('/') => {
                    browser.search_mode = true;
                    browser.search_query.clear();
                    browser.status_msg = "Search: ".to_string();
                }

                // Next/prev search result
                KeyCode::Char('n') => {
                    if !browser.search_results.is_empty() {
                        browser.search_idx = (browser.search_idx + 1) % browser.search_results.len();
                        browser.scroll = browser.search_results[browser.search_idx];
                        browser.status_msg = format!(
                            "Match {}/{} for '{}'",
                            browser.search_idx + 1,
                            browser.search_results.len(),
                            browser.search_query
                        );
                    }
                }
                KeyCode::Char('N') => {
                    if !browser.search_results.is_empty() {
                        if browser.search_idx == 0 {
                            browser.search_idx = browser.search_results.len() - 1;
                        } else {
                            browser.search_idx -= 1;
                        }
                        browser.scroll = browser.search_results[browser.search_idx];
                        browser.status_msg = format!(
                            "Match {}/{} for '{}'",
                            browser.search_idx + 1,
                            browser.search_results.len(),
                            browser.search_query
                        );
                    }
                }

                // Help
                KeyCode::Char('?') => {
                    browser.status_msg = "j/k=scroll  Tab=links  Enter=follow  b=back  /=search  n/N=next/prev  q=quit".to_string();
                }

                _ => {}
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
}

/// Draw the TUI frame.
fn draw_ui(f: &mut Frame, browser: &TuiBrowser) {
    let size = f.area();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Title bar
            Constraint::Min(1),   // Content
            Constraint::Length(1), // Status bar
        ])
        .split(size);

    draw_title_bar(f, chunks[0], browser);
    draw_content(f, chunks[1], browser);
    draw_status_bar(f, chunks[2], browser);
}

fn draw_title_bar(f: &mut Frame, area: Rect, browser: &TuiBrowser) {
    let title = format!(
        " OctoView │ {} │ {} nodes │ {} links",
        truncate(&browser.doc.url, area.width as usize - 40),
        browser.doc.nodes.len(),
        browser.link_lines.len()
    );
    let bar = Paragraph::new(Line::from(vec![
        Span::styled(title, Style::default().fg(Color::White).bg(Color::DarkGray)),
    ]))
    .style(Style::default().bg(Color::DarkGray));
    f.render_widget(bar, area);
}

fn draw_content(f: &mut Frame, area: Rect, browser: &TuiBrowser) {
    let visible_height = area.height as usize;
    let start = browser.scroll;
    let end = (start + visible_height).min(browser.lines.len());

    let mut text_lines: Vec<Line> = Vec::new();

    for i in start..end {
        let line = &browser.lines[i];

        // Check if this line is the selected link
        let is_selected = !browser.link_lines.is_empty()
            && browser.link_lines.get(browser.selected_link) == Some(&i);

        let spans: Vec<Span> = line
            .spans
            .iter()
            .map(|s| {
                let mut style = Style::default().fg(s.fg);
                if s.bold {
                    style = style.add_modifier(Modifier::BOLD);
                }
                if s.italic {
                    style = style.add_modifier(Modifier::ITALIC);
                }
                if s.underline {
                    style = style.add_modifier(Modifier::UNDERLINED);
                }
                if s.bg != Color::Reset {
                    style = style.bg(s.bg);
                }
                if is_selected {
                    style = style.bg(Color::DarkGray);
                }
                Span::styled(s.text.clone(), style)
            })
            .collect();

        text_lines.push(Line::from(spans));
    }

    let content = Paragraph::new(text_lines)
        .block(Block::default().borders(Borders::NONE))
        .wrap(Wrap { trim: false });

    f.render_widget(content, area);
}

fn draw_status_bar(f: &mut Frame, area: Rect, browser: &TuiBrowser) {
    let status = if browser.search_mode {
        format!("/{}", browser.search_query)
    } else if !browser.status_msg.is_empty() {
        browser.status_msg.clone()
    } else {
        let pct = if browser.lines.is_empty() {
            0
        } else {
            (browser.scroll * 100) / browser.lines.len().max(1)
        };
        format!(
            " {}  │  line {}/{}  │  {}%  │  ?=help",
            truncate(&browser.doc.title, 40),
            browser.scroll + 1,
            browser.lines.len(),
            pct
        )
    };

    let bar = Paragraph::new(Line::from(vec![
        Span::styled(status, Style::default().fg(Color::Black).bg(Color::Cyan)),
    ]))
    .style(Style::default().bg(Color::Cyan));
    f.render_widget(bar, area);
}

// ─── Document Rendering ─────────────────────────────────────────────────────

/// Convert an OVD document into styled lines for TUI display.
fn render_document(doc: &OvdDocument) -> Vec<RenderLine> {
    let mut lines: Vec<RenderLine> = Vec::new();

    // Title
    if !doc.title.is_empty() {
        lines.push(RenderLine {
            spans: vec![StyledSpan {
                text: format!("  {}", doc.title),
                fg: Color::Yellow,
                bg: Color::Reset,
                bold: true,
                italic: false,
                underline: false,
            }],
            node_idx: None,
            is_link: false,
        });
        lines.push(empty_line());
    }

    let mut prev_type = NodeType::Page;
    let mut in_list = false;
    let mut list_idx = 0;

    for (idx, node) in doc.nodes.iter().enumerate() {
        // Skip structural/empty nodes
        if node.node_type == NodeType::Page || node.node_type == NodeType::Unknown {
            continue;
        }
        if node.text.is_empty()
            && node.node_type != NodeType::Image
            && node.node_type != NodeType::Separator
            && node.node_type != NodeType::List
            && node.node_type != NodeType::Table
        {
            continue;
        }

        // Spacing between different content types
        let needs_gap = match (prev_type, node.node_type) {
            (NodeType::Heading, _) => false, // Headings already have gap
            (_, NodeType::Heading) => true,
            (NodeType::Paragraph, NodeType::Paragraph) => true,
            (NodeType::ListItem, NodeType::ListItem) => false,
            (_, NodeType::List) => true,
            (NodeType::List, _) => true,
            _ => false,
        };

        if needs_gap && !lines.is_empty() {
            lines.push(empty_line());
        }

        match node.node_type {
            NodeType::Heading => {
                if !lines.is_empty() {
                    lines.push(empty_line());
                }
                let prefix = match node.level {
                    1 => "══ ",
                    2 => "── ",
                    3 => "   ",
                    _ => "     ",
                };
                let (fg, bold) = match node.level {
                    1 => (Color::Yellow, true),
                    2 => (Color::Cyan, true),
                    3 => (Color::Green, true),
                    _ => (Color::White, true),
                };
                lines.push(RenderLine {
                    spans: vec![
                        StyledSpan {
                            text: prefix.to_string(),
                            fg,
                            bg: Color::Reset,
                            bold,
                            italic: false,
                            underline: false,
                        },
                        StyledSpan {
                            text: node.text.clone(),
                            fg,
                            bg: Color::Reset,
                            bold,
                            italic: false,
                            underline: false,
                        },
                    ],
                    node_idx: Some(idx),
                    is_link: false,
                });
            }

            NodeType::Paragraph | NodeType::TextSpan => {
                if node.semantic == SemanticRole::Advertising
                    || node.semantic == SemanticRole::Tracking
                {
                    continue; // Skip ads/trackers
                }
                // Wrap long paragraphs
                let indent = "  ";
                let text = &node.text;
                if text.len() > 200 {
                    // Multi-line wrap
                    for chunk in text_chunks(text, 100) {
                        lines.push(RenderLine {
                            spans: vec![StyledSpan {
                                text: format!("{}{}", indent, chunk),
                                fg: Color::White,
                                bg: Color::Reset,
                                bold: false,
                                italic: false,
                                underline: false,
                            }],
                            node_idx: Some(idx),
                            is_link: false,
                        });
                    }
                } else {
                    lines.push(RenderLine {
                        spans: vec![StyledSpan {
                            text: format!("{}{}", indent, text),
                            fg: Color::White,
                            bg: Color::Reset,
                            bold: false,
                            italic: false,
                            underline: false,
                        }],
                        node_idx: Some(idx),
                        is_link: false,
                    });
                }
            }

            NodeType::Link => {
                if node.text.is_empty() {
                    continue;
                }
                let href_display = if node.href.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", truncate(&node.href, 50))
                };

                lines.push(RenderLine {
                    spans: vec![
                        StyledSpan {
                            text: format!("  {}", node.text),
                            fg: Color::Blue,
                            bg: Color::Reset,
                            bold: false,
                            italic: false,
                            underline: true,
                        },
                        StyledSpan {
                            text: href_display,
                            fg: Color::DarkGray,
                            bg: Color::Reset,
                            bold: false,
                            italic: false,
                            underline: false,
                        },
                    ],
                    node_idx: Some(idx),
                    is_link: true,
                });
            }

            NodeType::Image => {
                let alt = if node.alt.is_empty() {
                    &node.src
                } else {
                    &node.alt
                };
                lines.push(RenderLine {
                    spans: vec![
                        StyledSpan {
                            text: "  [IMG] ".to_string(),
                            fg: Color::Magenta,
                            bg: Color::Reset,
                            bold: false,
                            italic: false,
                            underline: false,
                        },
                        StyledSpan {
                            text: truncate(alt, 80),
                            fg: Color::Magenta,
                            bg: Color::Reset,
                            bold: false,
                            italic: true,
                            underline: false,
                        },
                    ],
                    node_idx: Some(idx),
                    is_link: false,
                });
            }

            NodeType::List => {
                in_list = true;
                list_idx = 0;
            }

            NodeType::ListItem => {
                list_idx += 1;
                let bullet = if in_list {
                    format!("    {}. ", list_idx)
                } else {
                    "    - ".to_string()
                };
                lines.push(RenderLine {
                    spans: vec![
                        StyledSpan {
                            text: bullet,
                            fg: Color::Gray,
                            bg: Color::Reset,
                            bold: false,
                            italic: false,
                            underline: false,
                        },
                        StyledSpan {
                            text: node.text.clone(),
                            fg: Color::White,
                            bg: Color::Reset,
                            bold: false,
                            italic: false,
                            underline: false,
                        },
                    ],
                    node_idx: Some(idx),
                    is_link: false,
                });
            }

            NodeType::CodeBlock => {
                lines.push(RenderLine {
                    spans: vec![StyledSpan {
                        text: format!("  │ {}", node.text),
                        fg: Color::Green,
                        bg: Color::Reset,
                        bold: false,
                        italic: false,
                        underline: false,
                    }],
                    node_idx: Some(idx),
                    is_link: false,
                });
            }

            NodeType::Blockquote => {
                lines.push(RenderLine {
                    spans: vec![
                        StyledSpan {
                            text: "  ▎ ".to_string(),
                            fg: Color::DarkGray,
                            bg: Color::Reset,
                            bold: false,
                            italic: false,
                            underline: false,
                        },
                        StyledSpan {
                            text: node.text.clone(),
                            fg: Color::Gray,
                            bg: Color::Reset,
                            bold: false,
                            italic: true,
                            underline: false,
                        },
                    ],
                    node_idx: Some(idx),
                    is_link: false,
                });
            }

            NodeType::Separator => {
                lines.push(RenderLine {
                    spans: vec![StyledSpan {
                        text: "  ────────────────────────────────────────".to_string(),
                        fg: Color::DarkGray,
                        bg: Color::Reset,
                        bold: false,
                        italic: false,
                        underline: false,
                    }],
                    node_idx: None,
                    is_link: false,
                });
            }

            NodeType::TableCell => {
                lines.push(RenderLine {
                    spans: vec![StyledSpan {
                        text: format!("  │ {}", node.text),
                        fg: Color::White,
                        bg: Color::Reset,
                        bold: false,
                        italic: false,
                        underline: false,
                    }],
                    node_idx: Some(idx),
                    is_link: false,
                });
            }

            NodeType::Button => {
                lines.push(RenderLine {
                    spans: vec![
                        StyledSpan {
                            text: "  [".to_string(),
                            fg: Color::Gray,
                            bg: Color::Reset,
                            bold: false,
                            italic: false,
                            underline: false,
                        },
                        StyledSpan {
                            text: node.text.clone(),
                            fg: Color::Cyan,
                            bg: Color::Reset,
                            bold: true,
                            italic: false,
                            underline: false,
                        },
                        StyledSpan {
                            text: "]".to_string(),
                            fg: Color::Gray,
                            bg: Color::Reset,
                            bold: false,
                            italic: false,
                            underline: false,
                        },
                    ],
                    node_idx: Some(idx),
                    is_link: false,
                });
            }

            NodeType::InputField => {
                let placeholder = if !node.text.is_empty() {
                    &node.text
                } else {
                    "..."
                };
                lines.push(RenderLine {
                    spans: vec![StyledSpan {
                        text: format!("  [ {} ]", placeholder),
                        fg: Color::DarkGray,
                        bg: Color::Reset,
                        bold: false,
                        italic: false,
                        underline: false,
                    }],
                    node_idx: Some(idx),
                    is_link: false,
                });
            }

            // Skip structural container types
            NodeType::Navigation | NodeType::Section | NodeType::Header
            | NodeType::Footer | NodeType::Sidebar | NodeType::Card
            | NodeType::Modal | NodeType::Form | NodeType::Table
            | NodeType::TableRow | NodeType::Media | NodeType::Icon => {
                // These are containers — children render themselves
            }

            _ => {}
        }

        if node.node_type != NodeType::List {
            if in_list && node.node_type != NodeType::ListItem {
                in_list = false;
            }
        }
        prev_type = node.node_type;
    }

    // Ensure at least one line
    if lines.is_empty() {
        lines.push(RenderLine {
            spans: vec![StyledSpan {
                text: "(empty document)".to_string(),
                fg: Color::DarkGray,
                bg: Color::Reset,
                bold: false,
                italic: true,
                underline: false,
            }],
            node_idx: None,
            is_link: false,
        });
    }

    lines
}

fn empty_line() -> RenderLine {
    RenderLine {
        spans: vec![StyledSpan {
            text: String::new(),
            fg: Color::Reset,
            bg: Color::Reset,
            bold: false,
            italic: false,
            underline: false,
        }],
        node_idx: None,
        is_link: false,
    }
}

/// Split text into chunks of approximately `width` characters at word boundaries.
fn text_chunks(text: &str, width: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();

    for word in text.split_whitespace() {
        if current.len() + word.len() + 1 > width && !current.is_empty() {
            chunks.push(current);
            current = String::new();
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", &s[..max.saturating_sub(3)])
    } else {
        s.to_string()
    }
}

/// Resolve a relative URL against a base URL.
fn resolve_url(base: &str, href: &str) -> String {
    if href.starts_with("http://") || href.starts_with("https://") {
        return href.to_string();
    }
    if href.starts_with("//") {
        return format!("https:{}", href);
    }

    // Try to extract scheme + host from base
    if let Some(scheme_end) = base.find("://") {
        let after_scheme = &base[scheme_end + 3..];
        let host_end = after_scheme.find('/').unwrap_or(after_scheme.len());
        let base_origin = &base[..scheme_end + 3 + host_end];

        if href.starts_with('/') {
            return format!("{}{}", base_origin, href);
        }

        // Relative path — resolve against base directory
        let base_path = &base[..base.rfind('/').unwrap_or(base.len())];
        return format!("{}/{}", base_path, href);
    }

    href.to_string()
}
