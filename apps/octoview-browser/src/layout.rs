use crate::ovd::{NodeType, OvdDocument, OvdStyle};
use crate::text::{FontStyle, TextRenderer};

/// A positioned rectangle in the page coordinate space.
#[allow(dead_code)]
pub struct LayoutBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub node_idx: usize,
    pub content: LayoutContent,
    pub style: LayoutStyle,
}

#[allow(dead_code)]
pub enum LayoutContent {
    /// Container block (no direct rendering)
    Block,
    /// Wrapped text lines
    Text(Vec<TextLine>),
    /// Link: wrapped text + href
    Link(Vec<TextLine>, String),
    /// Image with decoded pixels (or alt text fallback)
    Image {
        alt: String,
        pixels: Option<Vec<u8>>,
        img_width: u32,
        img_height: u32,
    },
    /// Horizontal rule
    Separator,
    /// List item bullet/number prefix + text
    ListItem(String, Vec<TextLine>),
    /// Code block: monospace text lines
    CodeBlock(Vec<TextLine>),
    /// Blockquote: indented text lines with left border
    Blockquote(Vec<TextLine>),
}

pub struct TextLine {
    pub text: String,
    pub x_offset: f32,
    pub y_offset: f32,
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct LayoutStyle {
    pub font_size: f32,
    pub font_style: FontStyle,
    pub text_color: [u8; 3],
    pub bg_color: Option<[u8; 3]>,
    pub margin_top: f32,
    pub margin_bottom: f32,
    pub margin_left: f32,
    pub margin_right: f32,
    pub padding_left: f32,
    pub padding_top: f32,
    pub padding_bottom: f32,
    pub padding_right: f32,
    pub text_decoration_underline: bool,
    pub border_color: Option<[u8; 3]>,
    pub border_width: f32,
}

impl Default for LayoutStyle {
    fn default() -> Self {
        Self {
            font_size: 14.0,
            font_style: FontStyle::Regular,
            text_color: [205, 214, 244], // #cdd6f4
            bg_color: None,
            margin_top: 0.0,
            margin_bottom: 8.0,
            margin_left: 0.0,
            margin_right: 0.0,
            padding_left: 0.0,
            padding_top: 0.0,
            padding_bottom: 0.0,
            padding_right: 0.0,
            text_decoration_underline: false,
            border_color: None,
            border_width: 0.0,
        }
    }
}

// Default content margin
const MARGIN_LEFT: f32 = 16.0;
const MARGIN_RIGHT: f32 = 16.0;

/// Override layout style defaults with CSS inline style values from the node.
fn apply_ovd_style(style: &mut LayoutStyle, ovd_style: &OvdStyle) {
    if let Some(color) = ovd_style.color {
        style.text_color = color;
    }
    if let Some(bg) = ovd_style.background {
        style.bg_color = Some(bg);
    }
    if let Some(size) = ovd_style.font_size_px {
        style.font_size = size;
    }
    if let Some(weight) = ovd_style.font_weight {
        if weight >= 700 {
            style.font_style = FontStyle::Bold;
        }
    }
    // Box model overrides from CSS
    if let Some(mt) = ovd_style.margin_top {
        style.margin_top = mt;
    }
    if let Some(mb) = ovd_style.margin_bottom {
        style.margin_bottom = mb;
    }
    if let Some(ml) = ovd_style.margin_left {
        style.margin_left = ml;
    }
    if let Some(mr) = ovd_style.margin_right {
        style.margin_right = mr;
    }
    if let Some(pt) = ovd_style.padding_top {
        style.padding_top = pt;
    }
    if let Some(pb) = ovd_style.padding_bottom {
        style.padding_bottom = pb;
    }
    if let Some(pl) = ovd_style.padding_left {
        style.padding_left = pl;
    }
    if let Some(pr) = ovd_style.padding_right {
        style.padding_right = pr;
    }
    if ovd_style.text_decoration_underline {
        style.text_decoration_underline = true;
    }
    if let Some(bc) = ovd_style.border_color {
        style.border_color = Some(bc);
    }
    if let Some(bw) = ovd_style.border_width {
        style.border_width = bw;
    }
}

/// Compute the effective content area for a node, applying CSS margins and depth indentation.
fn effective_area(
    node: &crate::ovd::OvdNode,
    viewport_width: f32,
) -> (f32, f32) {
    let depth_indent = (node.depth as f32).min(8.0) * 4.0; // subtle depth-based indent
    let ml = node.style.margin_left.unwrap_or(0.0);
    let mr = node.style.margin_right.unwrap_or(0.0);
    let pl = node.style.padding_left.unwrap_or(0.0);
    let pr = node.style.padding_right.unwrap_or(0.0);

    let x = MARGIN_LEFT + depth_indent + ml + pl;
    let w = (viewport_width - MARGIN_LEFT - MARGIN_RIGHT - depth_indent - ml - mr - pl - pr).max(60.0);

    // Apply max-width constraint
    let w = if let Some(max_w) = node.style.max_width {
        w.min(max_w)
    } else {
        w
    };

    // Apply explicit width
    let w = if let Some(explicit_w) = node.style.width {
        explicit_w.min(w)
    } else {
        w
    };

    (x, w)
}

/// Walk OVD nodes and produce positioned layout boxes.
pub fn layout_document(
    doc: &OvdDocument,
    text: &mut TextRenderer,
    viewport_width: f32,
    image_cache: &mut crate::image_loader::ImageCache,
) -> Vec<LayoutBox> {
    let mut boxes: Vec<LayoutBox> = Vec::new();
    let mut cursor_y: f32 = 0.0;
    let _content_width = viewport_width - MARGIN_LEFT - MARGIN_RIGHT;

    // Collect table row groups for horizontal layout
    let mut table_cells: Vec<(usize, &crate::ovd::OvdNode)> = Vec::new();
    let mut in_table_parent: i32 = -1;

    for (idx, node) in doc.nodes.iter().enumerate() {
        // Flush table cells when we leave a table context
        if node.node_type != NodeType::TableCell && !table_cells.is_empty() {
            layout_table_row(
                &table_cells, text, &mut boxes, &mut cursor_y,
                viewport_width,
            );
            table_cells.clear();
            in_table_parent = -1;
        }

        // Skip display:none nodes
        if node.style.display_none {
            continue;
        }

        let (node_x, node_w) = effective_area(node, viewport_width);

        match node.node_type {
            NodeType::Page => {
                // Root container: apply background if present
                if let Some(bg) = node.style.background {
                    let mut layout_style = LayoutStyle::default();
                    layout_style.bg_color = Some(bg);
                    boxes.push(LayoutBox {
                        x: 0.0,
                        y: cursor_y,
                        width: viewport_width,
                        height: 0.0, // Will be adjusted at end
                        node_idx: idx,
                        content: LayoutContent::Block,
                        style: layout_style,
                    });
                }
            }

            NodeType::Section | NodeType::Card | NodeType::Header
            | NodeType::Footer | NodeType::Navigation | NodeType::Form => {
                // Container nodes: render background/border, add vertical spacing
                let mt = node.style.margin_top.unwrap_or(4.0);
                let mb = node.style.margin_bottom.unwrap_or(4.0);
                cursor_y += mt;

                if node.style.background.is_some() || node.style.border_color.is_some() {
                    let mut layout_style = LayoutStyle::default();
                    apply_ovd_style(&mut layout_style, &node.style);
                    layout_style.margin_top = mt;
                    layout_style.margin_bottom = mb;

                    let pt = node.style.padding_top.unwrap_or(0.0);
                    let pb = node.style.padding_bottom.unwrap_or(0.0);

                    boxes.push(LayoutBox {
                        x: node_x,
                        y: cursor_y,
                        width: node_w,
                        height: pt + pb + 2.0, // placeholder, children fill
                        node_idx: idx,
                        content: LayoutContent::Block,
                        style: layout_style,
                    });
                    cursor_y += pt;
                }
            }

            NodeType::Table => {
                // Table container: add spacing, track for cell grouping
                cursor_y += node.style.margin_top.unwrap_or(4.0);
                if let Some(bg) = node.style.background {
                    let mut layout_style = LayoutStyle::default();
                    layout_style.bg_color = Some(bg);
                    boxes.push(LayoutBox {
                        x: node_x,
                        y: cursor_y,
                        width: node_w,
                        height: 0.0,
                        node_idx: idx,
                        content: LayoutContent::Block,
                        style: layout_style,
                    });
                }
            }

            NodeType::TableRow => {
                // Start a new row context
                in_table_parent = node.node_id as i32;
                cursor_y += node.style.margin_top.unwrap_or(1.0);
            }

            NodeType::TableCell => {
                // Group with current row or start new row on parent change
                if !table_cells.is_empty() && node.parent_id != in_table_parent {
                    layout_table_row(
                        &table_cells, text, &mut boxes, &mut cursor_y,
                        viewport_width,
                    );
                    table_cells.clear();
                }
                in_table_parent = node.parent_id;
                table_cells.push((idx, node));
                continue; // Don't fall through
            }

            NodeType::Heading => {
                let (default_font_size, default_mt, default_mb) = match node.level {
                    1 => (32.0, 24.0, 16.0),
                    2 => (24.0, 20.0, 12.0),
                    3 => (20.0, 16.0, 10.0),
                    4 => (18.0, 12.0, 8.0),
                    5 => (16.0, 10.0, 6.0),
                    _ => (14.0, 8.0, 4.0),
                };

                let mut layout_style = LayoutStyle {
                    font_size: default_font_size,
                    font_style: FontStyle::Bold,
                    text_color: [245, 224, 220], // #f5e0dc warm white
                    margin_top: default_mt,
                    margin_bottom: default_mb,
                    ..Default::default()
                };
                apply_ovd_style(&mut layout_style, &node.style);

                cursor_y += layout_style.margin_top;
                let pt = layout_style.padding_top;
                let pb = layout_style.padding_bottom;
                cursor_y += pt;

                let lines = wrap_text_to_lines(
                    text, &node.text, node_w, layout_style.font_size, layout_style.font_style,
                );
                let line_h = text.line_height(layout_style.font_size);
                let text_height = lines.len() as f32 * line_h;
                let height = text_height + pt + pb;

                let mb = layout_style.margin_bottom;
                boxes.push(LayoutBox {
                    x: node_x,
                    y: cursor_y - pt,
                    width: node_w,
                    height,
                    node_idx: idx,
                    content: LayoutContent::Text(lines),
                    style: layout_style,
                });

                cursor_y += text_height + pb + mb;
            }

            NodeType::Paragraph | NodeType::TextSpan => {
                if node.text.trim().is_empty() {
                    continue;
                }
                let default_mt = if node.node_type == NodeType::Paragraph { 4.0 } else { 0.0 };
                let default_mb = if node.node_type == NodeType::Paragraph { 8.0 } else { 2.0 };

                let mut layout_style = LayoutStyle {
                    font_size: 14.0,
                    margin_top: node.style.margin_top.unwrap_or(default_mt),
                    margin_bottom: node.style.margin_bottom.unwrap_or(default_mb),
                    ..Default::default()
                };
                apply_ovd_style(&mut layout_style, &node.style);

                cursor_y += layout_style.margin_top;
                let pt = layout_style.padding_top;
                let pb = layout_style.padding_bottom;

                let lines = wrap_text_to_lines(
                    text, &node.text, node_w, layout_style.font_size, layout_style.font_style,
                );
                let line_h = text.line_height(layout_style.font_size);
                let text_height = lines.len() as f32 * line_h;
                let height = text_height + pt + pb;

                // Offset lines for padding
                let lines: Vec<TextLine> = lines.into_iter().map(|mut tl| {
                    tl.y_offset += pt;
                    tl
                }).collect();

                let mb = layout_style.margin_bottom;
                boxes.push(LayoutBox {
                    x: node_x,
                    y: cursor_y,
                    width: node_w,
                    height,
                    node_idx: idx,
                    content: LayoutContent::Text(lines),
                    style: layout_style,
                });

                cursor_y += height + mb;
            }

            NodeType::Link => {
                if node.text.trim().is_empty() {
                    continue;
                }
                let mut layout_style = LayoutStyle {
                    font_size: 14.0,
                    text_color: [137, 180, 250], // #89b4fa blue
                    text_decoration_underline: true,
                    margin_bottom: node.style.margin_bottom.unwrap_or(4.0),
                    margin_top: node.style.margin_top.unwrap_or(0.0),
                    ..Default::default()
                };
                apply_ovd_style(&mut layout_style, &node.style);

                cursor_y += layout_style.margin_top;
                let lines = wrap_text_to_lines(
                    text, &node.text, node_w, layout_style.font_size, layout_style.font_style,
                );
                let line_h = text.line_height(layout_style.font_size);
                let height = lines.len() as f32 * line_h;

                let mb = layout_style.margin_bottom;
                boxes.push(LayoutBox {
                    x: node_x,
                    y: cursor_y,
                    width: node_w,
                    height,
                    node_idx: idx,
                    content: LayoutContent::Link(lines, node.href.clone()),
                    style: layout_style,
                });

                cursor_y += height + mb;
            }

            NodeType::CodeBlock => {
                if node.text.trim().is_empty() {
                    continue;
                }
                let padding = node.style.padding_left.unwrap_or(12.0);
                let mut layout_style = LayoutStyle {
                    font_size: 13.0,
                    font_style: FontStyle::Monospace,
                    text_color: [166, 227, 161], // #a6e3a1 green
                    bg_color: Some([30, 30, 46]),  // #1e1e2e
                    padding_top: node.style.padding_top.unwrap_or(padding),
                    padding_bottom: node.style.padding_bottom.unwrap_or(padding),
                    padding_left: padding,
                    margin_top: node.style.margin_top.unwrap_or(8.0),
                    margin_bottom: node.style.margin_bottom.unwrap_or(12.0),
                    ..Default::default()
                };
                apply_ovd_style(&mut layout_style, &node.style);

                cursor_y += layout_style.margin_top;
                let code_width = node_w - layout_style.padding_left * 2.0;

                // Code blocks preserve newlines
                let mut all_lines = Vec::new();
                let mut y_off = 0.0;
                let line_h = text.line_height(layout_style.font_size);
                for raw_line in node.text.lines() {
                    let wrapped = wrap_text_to_lines(
                        text, raw_line, code_width, layout_style.font_size, FontStyle::Monospace,
                    );
                    for mut tl in wrapped {
                        tl.x_offset += layout_style.padding_left;
                        tl.y_offset = y_off + layout_style.padding_top;
                        y_off += line_h;
                        all_lines.push(tl);
                    }
                }
                let height = y_off + layout_style.padding_top + layout_style.padding_bottom;

                let mb = layout_style.margin_bottom;
                boxes.push(LayoutBox {
                    x: node_x,
                    y: cursor_y,
                    width: node_w,
                    height,
                    node_idx: idx,
                    content: LayoutContent::CodeBlock(all_lines),
                    style: layout_style,
                });

                cursor_y += height + mb;
            }

            NodeType::Blockquote => {
                if node.text.trim().is_empty() {
                    continue;
                }
                let indent = node.style.padding_left.unwrap_or(20.0);
                let mut layout_style = LayoutStyle {
                    font_size: 14.0,
                    text_color: [180, 190, 220],
                    margin_top: node.style.margin_top.unwrap_or(4.0),
                    margin_bottom: node.style.margin_bottom.unwrap_or(8.0),
                    padding_left: indent,
                    ..Default::default()
                };
                apply_ovd_style(&mut layout_style, &node.style);

                cursor_y += layout_style.margin_top;
                let quote_width = node_w - indent;
                let lines = wrap_text_to_lines(
                    text, &node.text, quote_width, layout_style.font_size, layout_style.font_style,
                );
                // Offset lines for indent
                let lines: Vec<TextLine> = lines
                    .into_iter()
                    .map(|mut tl| {
                        tl.x_offset += indent;
                        tl
                    })
                    .collect();
                let line_h = text.line_height(layout_style.font_size);
                let height = lines.len() as f32 * line_h + 8.0;

                let mb = layout_style.margin_bottom;
                boxes.push(LayoutBox {
                    x: node_x,
                    y: cursor_y,
                    width: node_w,
                    height,
                    node_idx: idx,
                    content: LayoutContent::Blockquote(lines),
                    style: layout_style,
                });

                cursor_y += height + mb;
            }

            NodeType::ListItem => {
                let indent = node.style.padding_left.unwrap_or(24.0);
                let bullet = "\u{2022} ".to_string();
                let mut layout_style = LayoutStyle {
                    font_size: 14.0,
                    margin_top: node.style.margin_top.unwrap_or(0.0),
                    margin_bottom: node.style.margin_bottom.unwrap_or(2.0),
                    padding_left: indent,
                    ..Default::default()
                };
                apply_ovd_style(&mut layout_style, &node.style);

                cursor_y += layout_style.margin_top;
                let item_width = node_w - indent;
                let lines = wrap_text_to_lines(
                    text, &node.text, item_width, layout_style.font_size, layout_style.font_style,
                );
                let lines: Vec<TextLine> = lines
                    .into_iter()
                    .map(|mut tl| {
                        tl.x_offset += indent;
                        tl
                    })
                    .collect();
                let line_h = text.line_height(layout_style.font_size);
                let height = lines.len() as f32 * line_h;

                let mb = layout_style.margin_bottom;
                boxes.push(LayoutBox {
                    x: node_x,
                    y: cursor_y,
                    width: node_w,
                    height,
                    node_idx: idx,
                    content: LayoutContent::ListItem(bullet, lines),
                    style: layout_style,
                });

                cursor_y += height + mb;
            }

            NodeType::List => {
                cursor_y += node.style.margin_top.unwrap_or(4.0);
                // Children (ListItems) are laid out individually
            }

            NodeType::Separator => {
                let mt = node.style.margin_top.unwrap_or(12.0);
                let mb = node.style.margin_bottom.unwrap_or(12.0);
                cursor_y += mt;
                let mut layout_style = LayoutStyle {
                    text_color: [69, 71, 90], // #45475a
                    margin_bottom: mb,
                    ..Default::default()
                };
                apply_ovd_style(&mut layout_style, &node.style);

                boxes.push(LayoutBox {
                    x: node_x,
                    y: cursor_y,
                    width: node_w,
                    height: 1.0,
                    node_idx: idx,
                    content: LayoutContent::Separator,
                    style: layout_style,
                });
                cursor_y += 1.0 + mb;
            }

            NodeType::Image => {
                cursor_y += node.style.margin_top.unwrap_or(4.0);
                let src = &node.src;
                let alt_text = if node.alt.is_empty() {
                    "[image]".to_string()
                } else {
                    format!("[image: {}]", node.alt)
                };

                // Try to resolve and fetch the image
                let resolved = resolve_url(&doc.url, src);
                let (img_pixels, img_w, img_h) = if let Some(ref resolved_url) = resolved {
                    if let Some(img) = image_cache.get_or_fetch(resolved_url, node_w as u32)
                    {
                        (Some(img.pixels.clone()), img.width, img.height)
                    } else {
                        (None, 0, 0)
                    }
                } else {
                    (None, 0, 0)
                };

                let height = if img_pixels.is_some() {
                    img_h as f32
                } else {
                    let font_size = 12.0;
                    text.line_height(font_size) + 8.0
                };

                let mut layout_style = LayoutStyle {
                    font_size: 12.0,
                    text_color: [150, 150, 170],
                    bg_color: if img_pixels.is_none() {
                        Some([35, 35, 50])
                    } else {
                        None
                    },
                    margin_bottom: node.style.margin_bottom.unwrap_or(8.0),
                    padding_left: if img_pixels.is_none() { 8.0 } else { 0.0 },
                    padding_top: if img_pixels.is_none() { 4.0 } else { 0.0 },
                    padding_bottom: if img_pixels.is_none() { 4.0 } else { 0.0 },
                    ..Default::default()
                };
                apply_ovd_style(&mut layout_style, &node.style);

                let width = if img_pixels.is_some() {
                    img_w as f32
                } else {
                    let (w, _) = text.measure_text(&alt_text, 12.0, FontStyle::Regular);
                    w + 16.0
                };

                let mb = layout_style.margin_bottom;
                boxes.push(LayoutBox {
                    x: node_x,
                    y: cursor_y,
                    width,
                    height,
                    node_idx: idx,
                    content: LayoutContent::Image {
                        alt: alt_text,
                        pixels: img_pixels,
                        img_width: img_w,
                        img_height: img_h,
                    },
                    style: layout_style,
                });

                cursor_y += height + mb;
            }

            NodeType::Button | NodeType::InputField => {
                if node.text.trim().is_empty() {
                    continue;
                }
                let mut layout_style = LayoutStyle {
                    font_size: 13.0,
                    bg_color: Some([49, 50, 68]), // #313244
                    margin_bottom: node.style.margin_bottom.unwrap_or(4.0),
                    margin_top: node.style.margin_top.unwrap_or(2.0),
                    padding_left: 8.0,
                    padding_top: 4.0,
                    ..Default::default()
                };
                apply_ovd_style(&mut layout_style, &node.style);

                cursor_y += layout_style.margin_top;
                let (w, _) = text.measure_text(&node.text, layout_style.font_size, FontStyle::Regular);
                let height = text.line_height(layout_style.font_size) + 8.0;

                let mb = layout_style.margin_bottom;
                boxes.push(LayoutBox {
                    x: node_x,
                    y: cursor_y,
                    width: w + 16.0,
                    height,
                    node_idx: idx,
                    content: LayoutContent::Text(vec![TextLine {
                        text: node.text.clone(),
                        x_offset: 8.0,
                        y_offset: 4.0,
                    }]),
                    style: layout_style,
                });

                cursor_y += height + mb;
            }

            _ => {
                // Unknown types with text: render as paragraph
                if !node.text.trim().is_empty() {
                    let mut layout_style = LayoutStyle {
                        font_size: 14.0,
                        margin_bottom: node.style.margin_bottom.unwrap_or(4.0),
                        margin_top: node.style.margin_top.unwrap_or(0.0),
                        ..Default::default()
                    };
                    apply_ovd_style(&mut layout_style, &node.style);

                    cursor_y += layout_style.margin_top;
                    let lines = wrap_text_to_lines(
                        text, &node.text, node_w, layout_style.font_size, layout_style.font_style,
                    );
                    let line_h = text.line_height(layout_style.font_size);
                    let height = lines.len() as f32 * line_h;

                    let mb = layout_style.margin_bottom;
                    boxes.push(LayoutBox {
                        x: node_x,
                        y: cursor_y,
                        width: node_w,
                        height,
                        node_idx: idx,
                        content: LayoutContent::Text(lines),
                        style: layout_style,
                    });

                    cursor_y += height + mb;
                }
            }
        }
    }

    // Flush any remaining table cells
    if !table_cells.is_empty() {
        layout_table_row(
            &table_cells, text, &mut boxes, &mut cursor_y,
            viewport_width,
        );
    }

    // Fix Page node height to cover all content
    if let Some(page_box) = boxes.iter_mut().find(|b| matches!(b.content, LayoutContent::Block) && b.y == 0.0) {
        page_box.height = cursor_y;
    }

    boxes
}

/// Lay out table cells horizontally across the row.
fn layout_table_row(
    cells: &[(usize, &crate::ovd::OvdNode)],
    text: &mut TextRenderer,
    boxes: &mut Vec<LayoutBox>,
    cursor_y: &mut f32,
    viewport_width: f32,
) {
    if cells.is_empty() {
        return;
    }
    let content_width = viewport_width - MARGIN_LEFT - MARGIN_RIGHT;
    let cell_count = cells.len() as f32;
    let cell_width = (content_width / cell_count).max(40.0);
    let cell_gap = 4.0;

    let mut max_height: f32 = 0.0;
    let mut cell_boxes: Vec<LayoutBox> = Vec::new();

    for (i, (idx, node)) in cells.iter().enumerate() {
        if node.text.trim().is_empty() {
            continue;
        }
        let mut layout_style = LayoutStyle {
            font_size: 13.0,
            margin_bottom: 2.0,
            padding_left: 4.0,
            padding_top: 2.0,
            padding_bottom: 2.0,
            ..Default::default()
        };
        apply_ovd_style(&mut layout_style, &node.style);

        let usable_width = cell_width - cell_gap - layout_style.padding_left;
        let lines = wrap_text_to_lines(
            text, &node.text, usable_width, layout_style.font_size, layout_style.font_style,
        );
        // Offset lines for padding
        let lines: Vec<TextLine> = lines.into_iter().map(|mut tl| {
            tl.x_offset += layout_style.padding_left;
            tl.y_offset += layout_style.padding_top;
            tl
        }).collect();

        let line_h = text.line_height(layout_style.font_size);
        let height = lines.len() as f32 * line_h + layout_style.padding_top + layout_style.padding_bottom;
        if height > max_height {
            max_height = height;
        }

        let x = MARGIN_LEFT + i as f32 * cell_width;
        cell_boxes.push(LayoutBox {
            x,
            y: *cursor_y,
            width: cell_width - cell_gap,
            height, // will be updated to max_height
            node_idx: *idx,
            content: LayoutContent::Text(lines),
            style: layout_style,
        });
    }

    // Set all cells to uniform height and push
    for mut b in cell_boxes {
        b.height = max_height;
        boxes.push(b);
    }

    *cursor_y += max_height + 2.0;
}

/// Compute total content height from layout boxes.
pub fn content_height(boxes: &[LayoutBox]) -> f32 {
    boxes
        .iter()
        .map(|b| b.y + b.height + b.style.margin_bottom)
        .fold(0.0f32, f32::max)
}

/// Resolve a relative URL against a base URL.
fn resolve_url(base_url: &str, relative: &str) -> Option<String> {
    if relative.is_empty() {
        return None;
    }
    if relative.starts_with("http://") || relative.starts_with("https://") {
        return Some(relative.to_string());
    }
    url::Url::parse(base_url)
        .ok()
        .and_then(|base| base.join(relative).ok())
        .map(|u| u.to_string())
}

/// Wrap text into positioned TextLine structs.
fn wrap_text_to_lines(
    text_renderer: &mut TextRenderer,
    text: &str,
    max_width: f32,
    font_size: f32,
    style: FontStyle,
) -> Vec<TextLine> {
    let wrapped = text_renderer.wrap_text(text, max_width, font_size, style);
    let line_h = text_renderer.line_height(font_size);
    wrapped
        .into_iter()
        .enumerate()
        .map(|(i, line_text)| TextLine {
            text: line_text,
            x_offset: 0.0,
            y_offset: i as f32 * line_h,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_loader::ImageCache;
    use crate::ovd::{OvdDocument, OvdNode, NodeType};

    #[test]
    fn test_heading_layout() {
        let mut doc = OvdDocument::new("test://");
        let mut page = OvdNode::new(NodeType::Page);
        page.depth = 0;
        doc.add_node(page);

        let mut h = OvdNode::new(NodeType::Heading);
        h.text = "Hello World".to_string();
        h.level = 1;
        h.depth = 1;
        doc.add_node(h);

        let mut tr = TextRenderer::new();
        let mut ic = ImageCache::new();
        let boxes = layout_document(&doc, &mut tr, 800.0, &mut ic);

        assert!(!boxes.is_empty());
        assert_eq!(boxes[0].style.font_size, 32.0);
        assert_eq!(boxes[0].style.font_style, FontStyle::Bold);
    }

    #[test]
    fn test_paragraph_wrapping() {
        let mut doc = OvdDocument::new("test://");
        let mut p = OvdNode::new(NodeType::Paragraph);
        p.text = "This is a long paragraph that should wrap to multiple lines when the viewport is narrow enough to require wrapping behavior.".to_string();
        doc.add_node(p);

        let mut tr = TextRenderer::new();
        let mut ic = ImageCache::new();
        let boxes = layout_document(&doc, &mut tr, 200.0, &mut ic);

        assert!(!boxes.is_empty());
        if let LayoutContent::Text(lines) = &boxes[0].content {
            assert!(lines.len() > 1, "Should wrap to multiple lines at 200px");
        } else {
            panic!("Expected Text content");
        }
    }

    #[test]
    fn test_link_layout() {
        let mut doc = OvdDocument::new("test://");
        let mut link = OvdNode::new(NodeType::Link);
        link.text = "Click here".to_string();
        link.href = "https://example.com".to_string();
        doc.add_node(link);

        let mut tr = TextRenderer::new();
        let mut ic = ImageCache::new();
        let boxes = layout_document(&doc, &mut tr, 800.0, &mut ic);

        assert!(!boxes.is_empty());
        assert_eq!(boxes[0].style.text_color, [137, 180, 250]); // blue
        if let LayoutContent::Link(_, href) = &boxes[0].content {
            assert_eq!(href, "https://example.com");
        } else {
            panic!("Expected Link content");
        }
    }

    #[test]
    fn test_content_height() {
        let mut doc = OvdDocument::new("test://");
        for i in 0..5 {
            let mut p = OvdNode::new(NodeType::Paragraph);
            p.text = format!("Paragraph {i}");
            doc.add_node(p);
        }

        let mut tr = TextRenderer::new();
        let mut ic = ImageCache::new();
        let boxes = layout_document(&doc, &mut tr, 800.0, &mut ic);
        let h = content_height(&boxes);
        assert!(h > 0.0, "Content height should be positive");
    }

    #[test]
    fn test_style_override_color() {
        let mut doc = OvdDocument::new("test://");
        let mut p = OvdNode::new(NodeType::Paragraph);
        p.text = "Red text".to_string();
        p.style.color = Some([255, 0, 0]);
        doc.add_node(p);

        let mut tr = TextRenderer::new();
        let mut ic = ImageCache::new();
        let boxes = layout_document(&doc, &mut tr, 800.0, &mut ic);
        assert!(!boxes.is_empty());
        assert_eq!(boxes[0].style.text_color, [255, 0, 0]);
    }

    #[test]
    fn test_style_override_font_size() {
        let mut doc = OvdDocument::new("test://");
        let mut p = OvdNode::new(NodeType::Paragraph);
        p.text = "Big text".to_string();
        p.style.font_size_px = Some(24.0);
        doc.add_node(p);

        let mut tr = TextRenderer::new();
        let mut ic = ImageCache::new();
        let boxes = layout_document(&doc, &mut tr, 800.0, &mut ic);
        assert!(!boxes.is_empty());
        assert_eq!(boxes[0].style.font_size, 24.0);
    }

    #[test]
    fn test_style_override_background() {
        let mut doc = OvdDocument::new("test://");
        let mut p = OvdNode::new(NodeType::Paragraph);
        p.text = "Highlighted".to_string();
        p.style.background = Some([255, 255, 0]);
        doc.add_node(p);

        let mut tr = TextRenderer::new();
        let mut ic = ImageCache::new();
        let boxes = layout_document(&doc, &mut tr, 800.0, &mut ic);
        assert!(!boxes.is_empty());
        assert_eq!(boxes[0].style.bg_color, Some([255, 255, 0]));
    }

    #[test]
    fn test_style_override_bold() {
        let mut doc = OvdDocument::new("test://");
        let mut p = OvdNode::new(NodeType::Paragraph);
        p.text = "Bold text".to_string();
        p.style.font_weight = Some(700);
        doc.add_node(p);

        let mut tr = TextRenderer::new();
        let mut ic = ImageCache::new();
        let boxes = layout_document(&doc, &mut tr, 800.0, &mut ic);
        assert!(!boxes.is_empty());
        assert_eq!(boxes[0].style.font_style, FontStyle::Bold);
    }

    #[test]
    fn test_css_margin_applied() {
        let mut doc = OvdDocument::new("test://");
        let mut p = OvdNode::new(NodeType::Paragraph);
        p.text = "Margin test".to_string();
        p.style.margin_top = Some(20.0);
        p.style.margin_bottom = Some(30.0);
        doc.add_node(p);

        let mut tr = TextRenderer::new();
        let mut ic = ImageCache::new();
        let boxes = layout_document(&doc, &mut tr, 800.0, &mut ic);
        assert!(!boxes.is_empty());
        assert_eq!(boxes[0].style.margin_top, 20.0);
        assert_eq!(boxes[0].style.margin_bottom, 30.0);
        // Y position should include margin_top
        assert!(boxes[0].y >= 20.0);
    }

    #[test]
    fn test_css_padding_applied() {
        let mut doc = OvdDocument::new("test://");
        let mut p = OvdNode::new(NodeType::Paragraph);
        p.text = "Padding test".to_string();
        p.style.padding_top = Some(10.0);
        p.style.padding_bottom = Some(10.0);
        doc.add_node(p);

        let mut tr = TextRenderer::new();
        let mut ic = ImageCache::new();
        let boxes = layout_document(&doc, &mut tr, 800.0, &mut ic);
        assert!(!boxes.is_empty());
        assert_eq!(boxes[0].style.padding_top, 10.0);
        assert_eq!(boxes[0].style.padding_bottom, 10.0);
    }

    #[test]
    fn test_link_underline() {
        let mut doc = OvdDocument::new("test://");
        let mut link = OvdNode::new(NodeType::Link);
        link.text = "Underlined link".to_string();
        link.href = "https://example.com".to_string();
        doc.add_node(link);

        let mut tr = TextRenderer::new();
        let mut ic = ImageCache::new();
        let boxes = layout_document(&doc, &mut tr, 800.0, &mut ic);
        assert!(!boxes.is_empty());
        assert!(boxes[0].style.text_decoration_underline);
    }

    #[test]
    fn test_table_cells_horizontal() {
        let mut doc = OvdDocument::new("test://");
        let mut table = OvdNode::new(NodeType::Table);
        table.depth = 0;
        let table_id = doc.add_node(table) as i32;

        let mut row = OvdNode::new(NodeType::TableRow);
        row.depth = 1;
        row.parent_id = table_id;
        let row_id = doc.add_node(row) as i32;

        let mut c1 = OvdNode::new(NodeType::TableCell);
        c1.text = "Cell A".to_string();
        c1.depth = 2;
        c1.parent_id = row_id;
        doc.add_node(c1);

        let mut c2 = OvdNode::new(NodeType::TableCell);
        c2.text = "Cell B".to_string();
        c2.depth = 2;
        c2.parent_id = row_id;
        doc.add_node(c2);

        let mut tr = TextRenderer::new();
        let mut ic = ImageCache::new();
        let boxes = layout_document(&doc, &mut tr, 800.0, &mut ic);
        let text_boxes: Vec<_> = boxes.iter()
            .filter(|b| matches!(&b.content, LayoutContent::Text(_)))
            .collect();
        assert_eq!(text_boxes.len(), 2);
        // Cell B should be to the right of Cell A (horizontal layout)
        assert!(text_boxes[1].x > text_boxes[0].x, "Cells should be side by side");
    }

    #[test]
    fn test_container_background() {
        let mut doc = OvdDocument::new("test://");
        let mut section = OvdNode::new(NodeType::Section);
        section.style.background = Some([50, 50, 80]);
        section.depth = 0;
        doc.add_node(section);

        let mut tr = TextRenderer::new();
        let mut ic = ImageCache::new();
        let boxes = layout_document(&doc, &mut tr, 800.0, &mut ic);
        let block_boxes: Vec<_> = boxes.iter()
            .filter(|b| matches!(b.content, LayoutContent::Block))
            .collect();
        assert!(!block_boxes.is_empty());
        assert_eq!(block_boxes[0].style.bg_color, Some([50, 50, 80]));
    }

    #[test]
    fn test_depth_indentation() {
        let mut doc = OvdDocument::new("test://");
        let mut p1 = OvdNode::new(NodeType::Paragraph);
        p1.text = "Shallow".to_string();
        p1.depth = 1;
        doc.add_node(p1);

        let mut p2 = OvdNode::new(NodeType::Paragraph);
        p2.text = "Deep".to_string();
        p2.depth = 5;
        doc.add_node(p2);

        let mut tr = TextRenderer::new();
        let mut ic = ImageCache::new();
        let boxes = layout_document(&doc, &mut tr, 800.0, &mut ic);
        assert_eq!(boxes.len(), 2);
        // Deeper node should be indented more
        assert!(boxes[1].x > boxes[0].x, "Deeper node should have more indent");
    }
}
