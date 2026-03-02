//! OctoView Document (OVD) — flat columnar page representation.
//!
//! Every webpage becomes a typed, flat array of nodes.
//! Simultaneously: render target, database table, semantic model.

use std::fmt;

/// Node type — what this content IS semantically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
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
    Modal = 0x15,
    Sidebar = 0x16,
    Footer = 0x17,
    Header = 0x18,
    TextSpan = 0x19,
    Icon = 0x1A,
    Unknown = 0xFF,
}

impl fmt::Display for NodeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeType::Page => write!(f, "page"),
            NodeType::Heading => write!(f, "heading"),
            NodeType::Paragraph => write!(f, "paragraph"),
            NodeType::Link => write!(f, "link"),
            NodeType::Image => write!(f, "image"),
            NodeType::Table => write!(f, "table"),
            NodeType::TableRow => write!(f, "table_row"),
            NodeType::TableCell => write!(f, "table_cell"),
            NodeType::List => write!(f, "list"),
            NodeType::ListItem => write!(f, "list_item"),
            NodeType::Form => write!(f, "form"),
            NodeType::InputField => write!(f, "input_field"),
            NodeType::Button => write!(f, "button"),
            NodeType::Navigation => write!(f, "navigation"),
            NodeType::Media => write!(f, "media"),
            NodeType::CodeBlock => write!(f, "code_block"),
            NodeType::Blockquote => write!(f, "blockquote"),
            NodeType::Separator => write!(f, "separator"),
            NodeType::Section => write!(f, "section"),
            NodeType::Card => write!(f, "card"),
            NodeType::Modal => write!(f, "modal"),
            NodeType::Sidebar => write!(f, "sidebar"),
            NodeType::Footer => write!(f, "footer"),
            NodeType::Header => write!(f, "header"),
            NodeType::TextSpan => write!(f, "text_span"),
            NodeType::Icon => write!(f, "icon"),
            NodeType::Unknown => write!(f, "unknown"),
        }
    }
}

/// Semantic role — inferred purpose of this node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SemanticRole {
    Content = 0x01,
    Navigation = 0x02,
    Search = 0x03,
    Sidebar = 0x04,
    Advertising = 0x05,
    Tracking = 0x06,
    Interactive = 0x07,
    Media = 0x08,
    Metadata = 0x09,
    Decoration = 0x0A,
    Structural = 0x0B,
}

impl fmt::Display for SemanticRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SemanticRole::Content => write!(f, "content"),
            SemanticRole::Navigation => write!(f, "navigation"),
            SemanticRole::Search => write!(f, "search"),
            SemanticRole::Sidebar => write!(f, "sidebar"),
            SemanticRole::Advertising => write!(f, "advertising"),
            SemanticRole::Tracking => write!(f, "tracking"),
            SemanticRole::Interactive => write!(f, "interactive"),
            SemanticRole::Media => write!(f, "media"),
            SemanticRole::Metadata => write!(f, "metadata"),
            SemanticRole::Decoration => write!(f, "decoration"),
            SemanticRole::Structural => write!(f, "structural"),
        }
    }
}

/// Computed style record — flat, no cascade.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct OvdStyle {
    pub font_size: f32,
    pub font_weight: u16,
    pub font_italic: bool,
    pub color: [u8; 4],
    pub background: [u8; 4],
    pub display_block: bool,
}

impl Default for OvdStyle {
    fn default() -> Self {
        Self {
            font_size: 16.0,
            font_weight: 400,
            font_italic: false,
            color: [0, 0, 0, 255],
            background: [255, 255, 255, 0],
            display_block: true,
        }
    }
}

/// A single node in the OctoView Document.
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
    pub semantic: SemanticRole,
    pub source_tag: String,
    pub class_names: String,
    pub style: OvdStyle,
    pub confidence: f32,
}

impl OvdNode {
    pub fn new(node_id: u32, node_type: NodeType, parent_id: i32, depth: u16) -> Self {
        Self {
            node_id,
            node_type,
            depth,
            parent_id,
            text: String::new(),
            href: String::new(),
            src: String::new(),
            alt: String::new(),
            level: 0,
            semantic: SemanticRole::Content,
            source_tag: String::new(),
            class_names: String::new(),
            style: OvdStyle::default(),
            confidence: 1.0,
        }
    }
}

/// The OctoView Document — a flat collection of typed nodes.
pub struct OvdDocument {
    pub url: String,
    pub title: String,
    pub nodes: Vec<OvdNode>,
    pub created: u64,
}

impl OvdDocument {
    pub fn new(url: &str) -> Self {
        Self {
            url: url.to_string(),
            title: String::new(),
            nodes: Vec::new(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Add a node, returning its assigned node_id.
    pub fn add_node(&mut self, mut node: OvdNode) -> u32 {
        let id = self.nodes.len() as u32;
        node.node_id = id;
        self.nodes.push(node);
        id
    }

    /// Query nodes by type.
    pub fn query_type(&self, node_type: NodeType) -> Vec<&OvdNode> {
        self.nodes.iter().filter(|n| n.node_type == node_type).collect()
    }

    /// Query nodes by semantic role.
    #[allow(dead_code)]
    pub fn query_semantic(&self, role: SemanticRole) -> Vec<&OvdNode> {
        self.nodes.iter().filter(|n| n.semantic == role).collect()
    }

    /// Query nodes matching a predicate string.
    /// Supports: type = 'heading', level = 1, semantic = 'content',
    /// depth < 3, confidence > 0.7, text LIKE '%word%',
    /// type != 'unknown', text != '', AND, OR
    pub fn query(&self, predicate: &str) -> Vec<&OvdNode> {
        let predicate = predicate.trim();
        if predicate.is_empty() || predicate == "*" {
            return self.nodes.iter().collect();
        }

        self.nodes
            .iter()
            .filter(|node| eval_predicate(node, predicate))
            .collect()
    }

    /// Summary statistics.
    pub fn stats(&self) -> OvdStats {
        let mut stats = OvdStats::default();
        stats.total_nodes = self.nodes.len();
        for node in &self.nodes {
            match node.node_type {
                NodeType::Heading => stats.headings += 1,
                NodeType::Paragraph => stats.paragraphs += 1,
                NodeType::Link => stats.links += 1,
                NodeType::Image => stats.images += 1,
                NodeType::Table => stats.tables += 1,
                NodeType::List => stats.lists += 1,
                NodeType::Form => stats.forms += 1,
                NodeType::CodeBlock => stats.code_blocks += 1,
                _ => {}
            }
            if node.depth > stats.max_depth {
                stats.max_depth = node.depth;
            }
        }
        stats
    }
}

/// Evaluate a compound predicate (supports AND and OR).
fn eval_predicate(node: &OvdNode, predicate: &str) -> bool {
    // Split on OR first (lower precedence than AND)
    // We need to handle " OR " carefully to not split inside quotes
    let or_groups = split_respecting_quotes(predicate, " OR ");
    if or_groups.len() > 1 {
        return or_groups.iter().any(|group| eval_and_group(node, group.trim()));
    }
    eval_and_group(node, predicate)
}

/// Evaluate an AND-connected group of conditions.
fn eval_and_group(node: &OvdNode, predicate: &str) -> bool {
    let conditions = split_respecting_quotes(predicate, " AND ");
    conditions.iter().all(|cond| eval_condition(node, cond.trim()))
}

/// Split a string on a delimiter, but don't split inside quoted strings.
fn split_respecting_quotes<'a>(s: &'a str, delim: &str) -> Vec<&'a str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut in_quote = false;
    let mut quote_char = '\'';
    let bytes = s.as_bytes();
    let delim_bytes = delim.as_bytes();

    let mut i = 0;
    while i < bytes.len() {
        let ch = bytes[i] as char;
        if !in_quote && (ch == '\'' || ch == '"') {
            in_quote = true;
            quote_char = ch;
        } else if in_quote && ch == quote_char {
            in_quote = false;
        } else if !in_quote && i + delim_bytes.len() <= bytes.len()
            && s[i..].starts_with(delim)
        {
            parts.push(&s[start..i]);
            i += delim_bytes.len();
            start = i;
            continue;
        }
        i += 1;
    }
    parts.push(&s[start..]);
    parts
}

/// Evaluate a single condition against a node.
fn eval_condition(node: &OvdNode, cond: &str) -> bool {
    let cond = cond.trim();

    // NOT prefix
    if cond.starts_with("NOT ") || cond.starts_with("not ") {
        return !eval_condition(node, &cond[4..]);
    }

    // Get the field value for string-based ops
    let get_str_field = |field: &str| -> String {
        match field {
            "type" => node.node_type.to_string(),
            "semantic" => node.semantic.to_string(),
            "source_tag" | "tag" => node.source_tag.clone(),
            "class" | "class_names" => node.class_names.clone(),
            "text" => node.text.clone(),
            "href" => node.href.clone(),
            "src" => node.src.clone(),
            "alt" => node.alt.clone(),
            "level" => node.level.to_string(),
            "depth" => node.depth.to_string(),
            "node_id" | "id" => node.node_id.to_string(),
            "parent_id" | "parent" => node.parent_id.to_string(),
            "confidence" => format!("{:.2}", node.confidence),
            // Style fields
            "font_size" => format!("{:.0}", node.style.font_size),
            "font_weight" => node.style.font_weight.to_string(),
            "bold" => if node.style.font_weight >= 600 { "true" } else { "false" }.to_string(),
            "italic" => if node.style.font_italic { "true" } else { "false" }.to_string(),
            _ => String::new(),
        }
    };

    let get_num_field = |field: &str| -> Option<f64> {
        match field {
            "level" => Some(node.level as f64),
            "depth" => Some(node.depth as f64),
            "node_id" | "id" => Some(node.node_id as f64),
            "parent_id" | "parent" => Some(node.parent_id as f64),
            "confidence" => Some(node.confidence as f64),
            "font_size" => Some(node.style.font_size as f64),
            "font_weight" => Some(node.style.font_weight as f64),
            _ => None,
        }
    };

    // field LIKE 'pattern' (SQL-style: % = any, _ = single char)
    if let Some((field, pattern)) = try_parse_op(cond, " LIKE ") {
        let val = get_str_field(field).to_lowercase();
        return sql_like_match(&val, &pattern.to_lowercase());
    }
    if let Some((field, pattern)) = try_parse_op(cond, " like ") {
        let val = get_str_field(field).to_lowercase();
        return sql_like_match(&val, &pattern.to_lowercase());
    }

    // field NOT LIKE 'pattern'
    if let Some((field, pattern)) = try_parse_op(cond, " NOT LIKE ") {
        let val = get_str_field(field).to_lowercase();
        return !sql_like_match(&val, &pattern.to_lowercase());
    }

    // field contains 'word'
    if let Some((field, val)) = try_parse_op(cond, " contains ") {
        return get_str_field(field).to_lowercase().contains(&val.to_lowercase());
    }

    // field != 'value' or field != ''
    if let Some((field, val)) = try_parse_op(cond, " != ") {
        let actual = get_str_field(field);
        // Numeric comparison if both parse
        if let (Some(a), Ok(b)) = (get_num_field(field), val.parse::<f64>()) {
            return (a - b).abs() >= 0.001;
        }
        return actual != val;
    }

    // field >= value
    if let Some((field, val)) = try_parse_op(cond, " >= ") {
        if let (Some(a), Ok(b)) = (get_num_field(field), val.parse::<f64>()) {
            return a >= b;
        }
        return get_str_field(field) >= val;
    }

    // field <= value
    if let Some((field, val)) = try_parse_op(cond, " <= ") {
        if let (Some(a), Ok(b)) = (get_num_field(field), val.parse::<f64>()) {
            return a <= b;
        }
        return get_str_field(field) <= val;
    }

    // field > value
    if let Some((field, val)) = try_parse_op(cond, " > ") {
        if let (Some(a), Ok(b)) = (get_num_field(field), val.parse::<f64>()) {
            return a > b;
        }
        return get_str_field(field) > val;
    }

    // field < value
    if let Some((field, val)) = try_parse_op(cond, " < ") {
        if let (Some(a), Ok(b)) = (get_num_field(field), val.parse::<f64>()) {
            return a < b;
        }
        return get_str_field(field) < val;
    }

    // field = 'value' (must come after >= and <=)
    if let Some((field, val)) = try_parse_op(cond, " = ") {
        let actual = get_str_field(field);
        if let (Some(a), Ok(b)) = (get_num_field(field), val.parse::<f64>()) {
            return (a - b).abs() < 0.001;
        }
        return actual == val;
    }

    // field='value' (no spaces around =, but not !=)
    if let Some(eq_idx) = cond.find('=') {
        if eq_idx > 0 && &cond[eq_idx - 1..eq_idx] != "!" && &cond[eq_idx - 1..eq_idx] != "<" && &cond[eq_idx - 1..eq_idx] != ">" {
            let field = cond[..eq_idx].trim();
            let val = cond[eq_idx + 1..].trim().trim_matches('\'').trim_matches('"');
            let actual = get_str_field(field);
            if let (Some(a), Ok(b)) = (get_num_field(field), val.parse::<f64>()) {
                return (a - b).abs() < 0.001;
            }
            return actual == val;
        }
    }

    true // Unknown condition — pass
}

/// Try to parse "field OP value" from a condition string.
/// Returns (field, value) with quotes stripped from value.
fn try_parse_op<'a>(cond: &'a str, op: &str) -> Option<(&'a str, String)> {
    if let Some(idx) = cond.find(op) {
        let field = cond[..idx].trim();
        let val = cond[idx + op.len()..].trim().trim_matches('\'').trim_matches('"');
        Some((field, val.to_string()))
    } else {
        None
    }
}

/// SQL LIKE pattern matching. % matches any sequence, _ matches single char.
fn sql_like_match(value: &str, pattern: &str) -> bool {
    like_match(value.as_bytes(), pattern.as_bytes())
}

fn like_match(value: &[u8], pattern: &[u8]) -> bool {
    let mut vi = 0;
    let mut pi = 0;
    let mut star_pi = usize::MAX;
    let mut star_vi = 0;

    while vi < value.len() {
        if pi < pattern.len() && pattern[pi] == b'_' {
            vi += 1;
            pi += 1;
        } else if pi < pattern.len() && pattern[pi] == b'%' {
            star_pi = pi;
            star_vi = vi;
            pi += 1;
        } else if pi < pattern.len() && pattern[pi] == value[vi] {
            vi += 1;
            pi += 1;
        } else if star_pi != usize::MAX {
            pi = star_pi + 1;
            star_vi += 1;
            vi = star_vi;
        } else {
            return false;
        }
    }

    while pi < pattern.len() && pattern[pi] == b'%' {
        pi += 1;
    }

    pi == pattern.len()
}

#[derive(Debug, Default)]
pub struct OvdStats {
    pub total_nodes: usize,
    pub headings: usize,
    pub paragraphs: usize,
    pub links: usize,
    pub images: usize,
    pub tables: usize,
    pub lists: usize,
    pub forms: usize,
    pub code_blocks: usize,
    pub max_depth: u16,
}

impl fmt::Display for OvdStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Total nodes: {}", self.total_nodes)?;
        writeln!(f, "  Headings:    {}", self.headings)?;
        writeln!(f, "  Paragraphs:  {}", self.paragraphs)?;
        writeln!(f, "  Links:       {}", self.links)?;
        writeln!(f, "  Images:      {}", self.images)?;
        writeln!(f, "  Tables:      {}", self.tables)?;
        writeln!(f, "  Lists:       {}", self.lists)?;
        writeln!(f, "  Forms:       {}", self.forms)?;
        writeln!(f, "  Code blocks: {}", self.code_blocks)?;
        writeln!(f, "  Max depth:   {}", self.max_depth)?;
        Ok(())
    }
}

/// Serialize OVD to binary .ovd file.
pub fn write_ovd(doc: &OvdDocument, path: &str) -> std::io::Result<()> {
    use byteorder::{LittleEndian, WriteBytesExt};
    use std::io::Write;

    let mut f = std::fs::File::create(path)?;

    // Header: magic + version + node count + timestamp
    f.write_all(b"OVDX")?;
    f.write_u16::<LittleEndian>(1)?; // version
    f.write_u16::<LittleEndian>(0)?; // flags
    f.write_u32::<LittleEndian>(doc.nodes.len() as u32)?;
    f.write_u64::<LittleEndian>(doc.created)?;

    // URL
    write_str(&mut f, &doc.url)?;

    // Title
    write_str(&mut f, &doc.title)?;

    // Nodes
    for node in &doc.nodes {
        f.write_u32::<LittleEndian>(node.node_id)?;
        f.write_u8(node.node_type as u8)?;
        f.write_u16::<LittleEndian>(node.depth)?;
        f.write_i32::<LittleEndian>(node.parent_id)?;
        f.write_u8(node.level)?;
        f.write_u8(node.semantic as u8)?;

        // Variable-length strings
        write_str(&mut f, &node.text)?;
        write_str(&mut f, &node.href)?;
        write_str(&mut f, &node.src)?;
        write_str(&mut f, &node.alt)?;
        write_str(&mut f, &node.source_tag)?;
        write_str(&mut f, &node.class_names)?;
    }

    // Footer sentinel
    f.write_all(b"OVDX")?;
    Ok(())
}

/// Read OVD from binary .ovd file.
pub fn read_ovd(path: &str) -> std::io::Result<OvdDocument> {
    use byteorder::{LittleEndian, ReadBytesExt};
    use std::io::Read;

    let mut f = std::fs::File::open(path)?;

    // Header
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != b"OVDX" {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not an OVD file"));
    }
    let _version = f.read_u16::<LittleEndian>()?;
    let _flags = f.read_u16::<LittleEndian>()?;
    let node_count = f.read_u32::<LittleEndian>()?;
    let created = f.read_u64::<LittleEndian>()?;

    let url = read_str(&mut f)?;
    let title = read_str(&mut f)?;

    let mut doc = OvdDocument::new(&url);
    doc.title = title;
    doc.created = created;

    for _ in 0..node_count {
        let node_id = f.read_u32::<LittleEndian>()?;
        let type_byte = f.read_u8()?;
        let depth = f.read_u16::<LittleEndian>()?;
        let parent_id = f.read_i32::<LittleEndian>()?;
        let level = f.read_u8()?;
        let semantic_byte = f.read_u8()?;

        let text = read_str(&mut f)?;
        let href = read_str(&mut f)?;
        let src = read_str(&mut f)?;
        let alt = read_str(&mut f)?;
        let source_tag = read_str(&mut f)?;
        let class_names = read_str(&mut f).unwrap_or_default();

        let node_type = byte_to_node_type(type_byte);
        let semantic = byte_to_semantic(semantic_byte);

        let mut node = OvdNode::new(node_id, node_type, parent_id, depth);
        node.level = level;
        node.semantic = semantic;
        node.text = text;
        node.href = href;
        node.src = src;
        node.alt = alt;
        node.source_tag = source_tag;
        node.class_names = class_names;
        doc.nodes.push(node);
    }

    Ok(doc)
}

fn write_str(f: &mut std::fs::File, s: &str) -> std::io::Result<()> {
    use byteorder::{LittleEndian, WriteBytesExt};
    use std::io::Write;
    let bytes = s.as_bytes();
    f.write_u32::<LittleEndian>(bytes.len() as u32)?;
    f.write_all(bytes)?;
    Ok(())
}

fn read_str(f: &mut std::fs::File) -> std::io::Result<String> {
    use byteorder::{LittleEndian, ReadBytesExt};
    use std::io::Read;
    let len = f.read_u32::<LittleEndian>()? as usize;
    if len > 10_000_000 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "string too long"));
    }
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf)?;
    String::from_utf8(buf)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

fn byte_to_node_type(b: u8) -> NodeType {
    match b {
        0x01 => NodeType::Page,
        0x02 => NodeType::Heading,
        0x03 => NodeType::Paragraph,
        0x04 => NodeType::Link,
        0x05 => NodeType::Image,
        0x06 => NodeType::Table,
        0x07 => NodeType::TableRow,
        0x08 => NodeType::TableCell,
        0x09 => NodeType::List,
        0x0A => NodeType::ListItem,
        0x0B => NodeType::Form,
        0x0C => NodeType::InputField,
        0x0D => NodeType::Button,
        0x0E => NodeType::Navigation,
        0x0F => NodeType::Media,
        0x10 => NodeType::CodeBlock,
        0x11 => NodeType::Blockquote,
        0x12 => NodeType::Separator,
        0x13 => NodeType::Section,
        0x14 => NodeType::Card,
        0x15 => NodeType::Modal,
        0x16 => NodeType::Sidebar,
        0x17 => NodeType::Footer,
        0x18 => NodeType::Header,
        0x19 => NodeType::TextSpan,
        0x1A => NodeType::Icon,
        _ => NodeType::Unknown,
    }
}

fn byte_to_semantic(b: u8) -> SemanticRole {
    match b {
        0x01 => SemanticRole::Content,
        0x02 => SemanticRole::Navigation,
        0x03 => SemanticRole::Search,
        0x04 => SemanticRole::Sidebar,
        0x05 => SemanticRole::Advertising,
        0x06 => SemanticRole::Tracking,
        0x07 => SemanticRole::Interactive,
        0x08 => SemanticRole::Media,
        0x09 => SemanticRole::Metadata,
        0x0A => SemanticRole::Decoration,
        _ => SemanticRole::Structural,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_doc() -> OvdDocument {
        let mut doc = OvdDocument::new("http://test.local/page");
        doc.title = "Test Page".to_string();

        let page = OvdNode::new(0, NodeType::Page, -1, 0);
        doc.add_node(page);

        let mut h1 = OvdNode::new(0, NodeType::Heading, 0, 1);
        h1.text = "Hello World".to_string();
        h1.level = 1;
        h1.source_tag = "h1".to_string();
        h1.semantic = SemanticRole::Content;
        doc.add_node(h1);

        let mut link = OvdNode::new(0, NodeType::Link, 0, 1);
        link.text = "Click me".to_string();
        link.href = "/about".to_string();
        link.source_tag = "a".to_string();
        link.semantic = SemanticRole::Navigation;
        doc.add_node(link);

        let mut p = OvdNode::new(0, NodeType::Paragraph, 0, 1);
        p.text = "Some content here.".to_string();
        p.source_tag = "p".to_string();
        doc.add_node(p);

        doc
    }

    #[test]
    fn test_ovd_write_read_roundtrip() {
        let doc = make_test_doc();
        let path = std::env::temp_dir().join("test_ovd_roundtrip.ovd");
        let path_str = path.to_str().unwrap();

        write_ovd(&doc, path_str).expect("write_ovd failed");
        let loaded = read_ovd(path_str).expect("read_ovd failed");

        assert_eq!(loaded.url, "http://test.local/page");
        assert_eq!(loaded.title, "Test Page");
        assert_eq!(loaded.nodes.len(), doc.nodes.len());

        // Verify node data survived roundtrip
        assert_eq!(loaded.nodes[1].text, "Hello World");
        assert_eq!(loaded.nodes[1].node_type, NodeType::Heading);
        assert_eq!(loaded.nodes[1].level, 1);
        assert_eq!(loaded.nodes[2].href, "/about");
        assert_eq!(loaded.nodes[2].node_type, NodeType::Link);
        assert_eq!(loaded.nodes[2].semantic, SemanticRole::Navigation);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_ovd_stats() {
        let doc = make_test_doc();
        let stats = doc.stats();
        assert_eq!(stats.total_nodes, 4);
        assert_eq!(stats.headings, 1);
        assert_eq!(stats.links, 1);
        assert_eq!(stats.paragraphs, 1);
        assert_eq!(stats.max_depth, 1);
    }

    #[test]
    fn test_ovd_query_type() {
        let doc = make_test_doc();
        let headings = doc.query_type(NodeType::Heading);
        assert_eq!(headings.len(), 1);
        assert_eq!(headings[0].text, "Hello World");
    }

    #[test]
    fn test_ovd_query_predicate() {
        let doc = make_test_doc();
        let results = doc.query("type = 'link'");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].href, "/about");

        let results = doc.query("text contains 'content'");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_type, NodeType::Paragraph);
    }

    #[test]
    fn test_read_ovd_invalid_file() {
        let path = std::env::temp_dir().join("test_ovd_invalid.ovd");
        std::fs::write(&path, b"NOT_OVD_DATA").unwrap();
        let result = read_ovd(path.to_str().unwrap());
        assert!(result.is_err(), "should fail on invalid OVD file");
        let _ = std::fs::remove_file(&path);
    }
}
