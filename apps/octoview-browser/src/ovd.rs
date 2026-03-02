/// Minimal OVD types for the browser layout engine.
/// Matches the structures in apps/octoview/src/ovd.rs.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum TextAlign {
    Left,
    Center,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Display {
    Block,
    Inline,
    InlineBlock,
    None,
    TableRow,
    TableCell,
}

#[derive(Debug, Clone)]
pub struct OvdStyle {
    pub color: Option<[u8; 3]>,
    pub background: Option<[u8; 3]>,
    pub font_weight: Option<u16>,
    pub font_size_px: Option<f32>,
    pub text_align: Option<TextAlign>,
    pub border_color: Option<[u8; 3]>,
    pub display_none: bool,
    // Box model
    pub margin_top: Option<f32>,
    pub margin_bottom: Option<f32>,
    pub margin_left: Option<f32>,
    pub margin_right: Option<f32>,
    pub padding_top: Option<f32>,
    pub padding_bottom: Option<f32>,
    pub padding_left: Option<f32>,
    pub padding_right: Option<f32>,
    pub width: Option<f32>,
    pub max_width: Option<f32>,
    // Display
    pub display: Option<Display>,
    // Text
    pub text_decoration_underline: bool,
    pub line_height: Option<f32>,
    pub border_width: Option<f32>,
}

impl Default for OvdStyle {
    fn default() -> Self {
        Self {
            color: None,
            background: None,
            font_weight: None,
            font_size_px: None,
            text_align: None,
            border_color: None,
            display_none: false,
            margin_top: None,
            margin_bottom: None,
            margin_left: None,
            margin_right: None,
            padding_top: None,
            padding_bottom: None,
            padding_left: None,
            padding_right: None,
            width: None,
            max_width: None,
            display: None,
            text_decoration_underline: false,
            line_height: None,
            border_width: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
#[allow(dead_code)]
pub enum NodeType {
    Page = 0x01,
    Heading = 0x02,
    Paragraph = 0x03,
    Link = 0x04,
    Image = 0x05,
    Table = 0x06,
    TableRow = 0x07,
    TableCell = 0x08,
    List = 0x09,
    ListItem = 0x0A,
    Form = 0x0B,
    InputField = 0x0C,
    Button = 0x0D,
    Navigation = 0x0E,
    Media = 0x0F,
    CodeBlock = 0x10,
    Blockquote = 0x11,
    Separator = 0x12,
    Section = 0x13,
    Card = 0x14,
    Header = 0x18,
    Footer = 0x17,
    TextSpan = 0x19,
    Unknown = 0xFF,
}

#[derive(Debug, Clone)]
pub struct OvdNode {
    pub node_id: u32,
    pub node_type: NodeType,
    pub depth: u16,
    pub parent_id: i32,
    pub text: String,
    pub href: String,
    pub src: String,
    pub alt: String,
    pub level: u8,
    pub style: OvdStyle,
}

impl OvdNode {
    pub fn new(node_type: NodeType) -> Self {
        Self {
            node_id: 0,
            node_type,
            depth: 0,
            parent_id: -1,
            text: String::new(),
            href: String::new(),
            src: String::new(),
            alt: String::new(),
            level: 0,
            style: OvdStyle::default(),
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct OvdDocument {
    pub url: String,
    pub title: String,
    pub nodes: Vec<OvdNode>,
}

impl OvdDocument {
    pub fn new(url: &str) -> Self {
        Self {
            url: url.to_string(),
            title: String::new(),
            nodes: Vec::new(),
        }
    }

    pub fn add_node(&mut self, mut node: OvdNode) -> u32 {
        let id = self.nodes.len() as u32;
        node.node_id = id;
        self.nodes.push(node);
        id
    }
}
