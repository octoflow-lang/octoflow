/// OVD Inspector -- node tree panel with properties view.
/// Toggle with F12. Renders as a right-side panel taking 30% of window width.

use crate::framebuffer::Framebuffer;
use crate::layout::LayoutBox;
use crate::ovd::{NodeType, OvdDocument};
use crate::text::{FontStyle, TextRenderer};

pub const INSPECTOR_WIDTH_PCT: f32 = 0.30;
const INSPECTOR_BG: [u8; 3] = [24, 24, 38];
const NODE_TEXT_COLOR: [u8; 3] = [205, 214, 244];
const SELECTED_BG: [u8; 3] = [49, 50, 68];
const TYPE_COLOR: [u8; 3] = [137, 180, 250];
const VALUE_COLOR: [u8; 3] = [166, 227, 161];
const LINE_HEIGHT: f32 = 16.0;
const INDENT_PX: f32 = 12.0;

// Separator/border color (from chrome)
const SEPARATOR: [u8; 3] = [69, 71, 90];
// Dim text for labels
const DIM_TEXT: [u8; 3] = [130, 135, 160];
// Header text
const HEADER_COLOR: [u8; 3] = [245, 224, 220];

pub struct InspectorState {
    pub visible: bool,
    pub scroll_y: f32,
    pub selected_node: Option<usize>,
}

impl InspectorState {
    pub fn new() -> Self {
        Self {
            visible: false,
            scroll_y: 0.0,
            selected_node: None,
        }
    }

    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    /// Width of the inspector panel in pixels (0 when hidden).
    pub fn panel_width(&self, window_width: u32) -> f32 {
        if self.visible {
            (window_width as f32 * INSPECTOR_WIDTH_PCT).floor()
        } else {
            0.0
        }
    }

    /// Width available for page content (full width minus inspector).
    pub fn content_viewport_width(&self, window_width: u32) -> f32 {
        window_width as f32 - self.panel_width(window_width)
    }
}

/// Render the inspector panel on the right side of the framebuffer.
///
/// Parameters:
/// - `state`: inspector state (visibility, scroll, selection)
/// - `doc`: the OVD document with all nodes
/// - `_boxes`: layout boxes (unused currently, reserved for future box overlay)
/// - `fb`: framebuffer to draw onto
/// - `text`: text renderer
/// - `top_y`: Y offset where the content area starts (below toolbar+tabs)
/// - `window_width`: total framebuffer width
/// - `window_height`: total framebuffer height
pub fn render_inspector(
    state: &InspectorState,
    doc: &OvdDocument,
    _boxes: &[LayoutBox],
    fb: &mut Framebuffer,
    text: &mut TextRenderer,
    top_y: f32,
    window_width: u32,
    window_height: u32,
) {
    if !state.visible {
        return;
    }

    let panel_w = state.panel_width(window_width);
    let panel_x = (window_width as f32 - panel_w) as i32;
    let panel_h = window_height as f32 - top_y;

    // 1. Background
    fb.draw_rect(
        panel_x,
        top_y as i32,
        panel_w as u32,
        panel_h as u32,
        INSPECTOR_BG[0],
        INSPECTOR_BG[1],
        INSPECTOR_BG[2],
    );

    // 2. Left border separator
    fb.draw_vline(
        panel_x,
        top_y as i32,
        panel_h as u32,
        SEPARATOR[0],
        SEPARATOR[1],
        SEPARATOR[2],
    );

    // 3. Header: "OVD Inspector"
    let header_y = top_y as i32 + 4;
    text.draw_text(
        fb,
        "OVD Inspector",
        panel_x + 8,
        header_y,
        14.0,
        FontStyle::Bold,
        HEADER_COLOR[0],
        HEADER_COLOR[1],
        HEADER_COLOR[2],
    );

    // Separator below header
    let sep_y = top_y as i32 + 22;
    fb.draw_hline(
        panel_x + 1,
        sep_y,
        panel_w as u32 - 1,
        SEPARATOR[0],
        SEPARATOR[1],
        SEPARATOR[2],
    );

    // -- Node tree (upper half) --
    let tree_top = sep_y as f32 + 2.0;
    let tree_height = (panel_h - 24.0) * 0.5; // upper half of content area
    let properties_top = tree_top + tree_height;

    // Render node tree entries
    let mut y = tree_top - state.scroll_y;
    for (idx, node) in doc.nodes.iter().enumerate() {
        let row_y = y;

        // Skip if off-screen (above tree_top or below properties_top)
        if row_y + LINE_HEIGHT < tree_top || row_y > properties_top {
            y += LINE_HEIGHT;
            continue;
        }

        let indent = node.depth as f32 * INDENT_PX;
        let text_x = panel_x + 8 + indent as i32;
        let text_y = row_y as i32;

        // Selection highlight
        if state.selected_node == Some(idx) {
            fb.draw_rect(
                panel_x + 1,
                text_y,
                panel_w as u32 - 1,
                LINE_HEIGHT as u32,
                SELECTED_BG[0],
                SELECTED_BG[1],
                SELECTED_BG[2],
            );
        }

        // Type name in blue
        let type_name = node_type_name(node.node_type);
        text.draw_text(
            fb,
            type_name,
            text_x,
            text_y,
            11.0,
            FontStyle::Regular,
            TYPE_COLOR[0],
            TYPE_COLOR[1],
            TYPE_COLOR[2],
        );

        // Text preview in green (if present)
        if !node.text.is_empty() {
            let (tw, _) = text.measure_text(type_name, 11.0, FontStyle::Regular);
            let preview_x = text_x + tw as i32 + 4;
            let max_preview_w = (panel_x as f32 + panel_w - preview_x as f32 - 8.0).max(0.0);
            let preview = truncate_text(&node.text, 40);
            // Only draw if there's room
            if max_preview_w > 20.0 {
                text.draw_text(
                    fb,
                    &preview,
                    preview_x,
                    text_y,
                    11.0,
                    FontStyle::Regular,
                    VALUE_COLOR[0],
                    VALUE_COLOR[1],
                    VALUE_COLOR[2],
                );
            }
        }

        y += LINE_HEIGHT;
    }

    // -- Properties panel (lower half) --

    // Separator between tree and properties
    fb.draw_hline(
        panel_x + 1,
        properties_top as i32,
        panel_w as u32 - 1,
        SEPARATOR[0],
        SEPARATOR[1],
        SEPARATOR[2],
    );

    let prop_y_start = properties_top as i32 + 4;
    text.draw_text(
        fb,
        "Properties",
        panel_x + 8,
        prop_y_start,
        12.0,
        FontStyle::Bold,
        HEADER_COLOR[0],
        HEADER_COLOR[1],
        HEADER_COLOR[2],
    );

    if let Some(sel_idx) = state.selected_node {
        if let Some(node) = doc.nodes.get(sel_idx) {
            let mut py = prop_y_start + 20;
            let label_x = panel_x + 12;
            let value_x = panel_x + 80;

            // Type
            text.draw_text(
                fb,
                "Type:",
                label_x,
                py,
                11.0,
                FontStyle::Regular,
                DIM_TEXT[0],
                DIM_TEXT[1],
                DIM_TEXT[2],
            );
            text.draw_text(
                fb,
                node_type_name(node.node_type),
                value_x,
                py,
                11.0,
                FontStyle::Regular,
                TYPE_COLOR[0],
                TYPE_COLOR[1],
                TYPE_COLOR[2],
            );
            py += LINE_HEIGHT as i32;

            // ID
            text.draw_text(
                fb,
                "ID:",
                label_x,
                py,
                11.0,
                FontStyle::Regular,
                DIM_TEXT[0],
                DIM_TEXT[1],
                DIM_TEXT[2],
            );
            let id_str = format!("{}", node.node_id);
            text.draw_text(
                fb,
                &id_str,
                value_x,
                py,
                11.0,
                FontStyle::Regular,
                NODE_TEXT_COLOR[0],
                NODE_TEXT_COLOR[1],
                NODE_TEXT_COLOR[2],
            );
            py += LINE_HEIGHT as i32;

            // Depth
            text.draw_text(
                fb,
                "Depth:",
                label_x,
                py,
                11.0,
                FontStyle::Regular,
                DIM_TEXT[0],
                DIM_TEXT[1],
                DIM_TEXT[2],
            );
            let depth_str = format!("{}", node.depth);
            text.draw_text(
                fb,
                &depth_str,
                value_x,
                py,
                11.0,
                FontStyle::Regular,
                NODE_TEXT_COLOR[0],
                NODE_TEXT_COLOR[1],
                NODE_TEXT_COLOR[2],
            );
            py += LINE_HEIGHT as i32;

            // Parent
            text.draw_text(
                fb,
                "Parent:",
                label_x,
                py,
                11.0,
                FontStyle::Regular,
                DIM_TEXT[0],
                DIM_TEXT[1],
                DIM_TEXT[2],
            );
            let parent_str = if node.parent_id >= 0 {
                format!("{}", node.parent_id)
            } else {
                "(none)".to_string()
            };
            text.draw_text(
                fb,
                &parent_str,
                value_x,
                py,
                11.0,
                FontStyle::Regular,
                NODE_TEXT_COLOR[0],
                NODE_TEXT_COLOR[1],
                NODE_TEXT_COLOR[2],
            );
            py += LINE_HEIGHT as i32;

            // Text preview
            if !node.text.is_empty() {
                text.draw_text(
                    fb,
                    "Text:",
                    label_x,
                    py,
                    11.0,
                    FontStyle::Regular,
                    DIM_TEXT[0],
                    DIM_TEXT[1],
                    DIM_TEXT[2],
                );
                let text_preview = truncate_text(&node.text, 50);
                text.draw_text(
                    fb,
                    &text_preview,
                    value_x,
                    py,
                    11.0,
                    FontStyle::Regular,
                    VALUE_COLOR[0],
                    VALUE_COLOR[1],
                    VALUE_COLOR[2],
                );
            }
        }
    } else {
        // No selection hint
        text.draw_text(
            fb,
            "Click a node above",
            panel_x + 12,
            prop_y_start + 20,
            11.0,
            FontStyle::Regular,
            DIM_TEXT[0],
            DIM_TEXT[1],
            DIM_TEXT[2],
        );
    }
}

/// Hit test for the inspector node tree. Returns the node index if a tree row was clicked.
///
/// Parameters:
/// - `state`: inspector state
/// - `doc`: OVD document
/// - `click_x`, `click_y`: mouse click coordinates
/// - `top_y`: content area top offset
/// - `window_width`, `window_height`: framebuffer dimensions
pub fn inspector_hit_test(
    state: &InspectorState,
    doc: &OvdDocument,
    click_x: f32,
    click_y: f32,
    top_y: f32,
    window_width: u32,
    _window_height: u32,
) -> Option<usize> {
    if !state.visible {
        return None;
    }

    let panel_w = state.panel_width(window_width);
    let panel_x = window_width as f32 - panel_w;

    // Check if click is within the inspector panel horizontally
    if click_x < panel_x {
        return None;
    }

    // Tree starts below the header (top_y + 24)
    let tree_top = top_y + 24.0;
    let panel_h = _window_height as f32 - top_y;
    let tree_height = (panel_h - 24.0) * 0.5;
    let properties_top = tree_top + tree_height;

    // Check if click is in the tree area vertically
    if click_y < tree_top || click_y >= properties_top {
        return None;
    }

    // Calculate which row was clicked
    let relative_y = click_y - tree_top + state.scroll_y;
    let row_idx = (relative_y / LINE_HEIGHT) as usize;

    if row_idx < doc.nodes.len() {
        Some(row_idx)
    } else {
        None
    }
}

/// Human-readable name for each OVD node type.
pub fn node_type_name(nt: NodeType) -> &'static str {
    match nt {
        NodeType::Page => "Page",
        NodeType::Heading => "Heading",
        NodeType::Paragraph => "Paragraph",
        NodeType::Link => "Link",
        NodeType::Image => "Image",
        NodeType::Table => "Table",
        NodeType::TableRow => "TableRow",
        NodeType::TableCell => "TableCell",
        NodeType::List => "List",
        NodeType::ListItem => "ListItem",
        NodeType::Form => "Form",
        NodeType::InputField => "InputField",
        NodeType::Button => "Button",
        NodeType::Navigation => "Navigation",
        NodeType::Media => "Media",
        NodeType::CodeBlock => "CodeBlock",
        NodeType::Blockquote => "Blockquote",
        NodeType::Separator => "Separator",
        NodeType::Section => "Section",
        NodeType::Card => "Card",
        NodeType::Header => "Header",
        NodeType::Footer => "Footer",
        NodeType::TextSpan => "TextSpan",
        NodeType::Unknown => "Unknown",
    }
}

/// Truncate text to `max_chars` with ellipsis, collapsing whitespace.
pub fn truncate_text(text: &str, max_chars: usize) -> String {
    // Collapse whitespace and trim
    let collapsed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.len() <= max_chars {
        collapsed
    } else {
        let truncated: String = collapsed.chars().take(max_chars.saturating_sub(3)).collect();
        format!("{}...", truncated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inspector_toggle() {
        let mut state = InspectorState::new();
        assert!(!state.visible);
        state.toggle();
        assert!(state.visible);
        state.toggle();
        assert!(!state.visible);
    }

    #[test]
    fn test_inspector_panel_width() {
        let mut state = InspectorState::new();
        state.visible = true;
        let w = state.panel_width(1000);
        assert_eq!(w, 300.0);
    }

    #[test]
    fn test_inspector_hidden_width() {
        let state = InspectorState::new();
        let w = state.panel_width(1000);
        assert_eq!(w, 0.0);
    }
}
