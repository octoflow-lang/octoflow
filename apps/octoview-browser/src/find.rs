use crate::layout::{LayoutBox, LayoutContent};

#[allow(dead_code)]
pub struct FindMatch {
    pub box_idx: usize,
    pub char_start: usize,
    pub char_len: usize,
    pub y_position: f32,
}

pub struct FindState {
    pub query: String,
    pub cursor: usize,
    pub matches: Vec<FindMatch>,
    pub active_match: usize,
    pub visible: bool,
}

impl FindState {
    pub fn new() -> Self {
        Self {
            query: String::new(),
            cursor: 0,
            matches: Vec::new(),
            active_match: 0,
            visible: false,
        }
    }

    pub fn toggle(&mut self) {
        self.visible = !self.visible;
        if !self.visible {
            self.query.clear();
            self.cursor = 0;
            self.matches.clear();
            self.active_match = 0;
        }
    }

    pub fn search(&mut self, boxes: &[LayoutBox]) {
        self.matches.clear();
        self.active_match = 0;
        if self.query.is_empty() {
            return;
        }
        let query_lower = self.query.to_lowercase();

        for (idx, bx) in boxes.iter().enumerate() {
            let text = extract_box_text(&bx.content);
            if text.is_empty() {
                continue;
            }
            let text_lower = text.to_lowercase();
            let mut start = 0;
            while let Some(pos) = text_lower[start..].find(&query_lower) {
                let char_start = start + pos;
                self.matches.push(FindMatch {
                    box_idx: idx,
                    char_start,
                    char_len: self.query.len(),
                    y_position: bx.y,
                });
                start = char_start + 1;
            }
        }
    }

    pub fn next_match(&mut self) {
        if !self.matches.is_empty() {
            self.active_match = (self.active_match + 1) % self.matches.len();
        }
    }

    pub fn prev_match(&mut self) {
        if !self.matches.is_empty() {
            if self.active_match == 0 {
                self.active_match = self.matches.len() - 1;
            } else {
                self.active_match -= 1;
            }
        }
    }

    pub fn active_y(&self) -> Option<f32> {
        self.matches.get(self.active_match).map(|m| m.y_position)
    }

    pub fn match_count_text(&self) -> String {
        if self.query.is_empty() {
            String::new()
        } else if self.matches.is_empty() {
            "0/0".to_string()
        } else {
            format!("{}/{}", self.active_match + 1, self.matches.len())
        }
    }

    pub fn insert(&mut self, ch: char) {
        self.query.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
    }

    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            let prev = self.query[..self.cursor]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.query.remove(prev);
            self.cursor = prev;
        }
    }
}

fn extract_box_text(content: &LayoutContent) -> String {
    match content {
        LayoutContent::Text(lines)
        | LayoutContent::CodeBlock(lines)
        | LayoutContent::Blockquote(lines) => lines
            .iter()
            .map(|l| l.text.as_str())
            .collect::<Vec<_>>()
            .join(" "),
        LayoutContent::Link(lines, _) => lines
            .iter()
            .map(|l| l.text.as_str())
            .collect::<Vec<_>>()
            .join(" "),
        LayoutContent::ListItem(prefix, lines) => {
            let text: String = lines
                .iter()
                .map(|l| l.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            format!("{}{}", prefix, text)
        }
        LayoutContent::Image { ref alt, .. } => alt.clone(),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{LayoutContent, LayoutStyle, TextLine};

    fn make_text_box(text: &str, y: f32) -> LayoutBox {
        LayoutBox {
            x: 0.0,
            y,
            width: 100.0,
            height: 20.0,
            node_idx: 0,
            content: LayoutContent::Text(vec![TextLine {
                text: text.to_string(),
                x_offset: 0.0,
                y_offset: 0.0,
            }]),
            style: LayoutStyle::default(),
        }
    }

    #[test]
    fn test_find_basic() {
        let boxes = vec![
            make_text_box("Hello world", 0.0),
            make_text_box("world peace", 20.0),
        ];
        let mut fs = FindState::new();
        fs.query = "world".to_string();
        fs.search(&boxes);
        assert_eq!(fs.matches.len(), 2);
        assert_eq!(fs.match_count_text(), "1/2");
    }

    #[test]
    fn test_find_case_insensitive() {
        let boxes = vec![make_text_box("Hello World", 0.0)];
        let mut fs = FindState::new();
        fs.query = "hello".to_string();
        fs.search(&boxes);
        assert_eq!(fs.matches.len(), 1);
    }

    #[test]
    fn test_find_navigation() {
        let boxes = vec![make_text_box("abc abc abc", 0.0)];
        let mut fs = FindState::new();
        fs.query = "abc".to_string();
        fs.search(&boxes);
        assert_eq!(fs.matches.len(), 3);
        assert_eq!(fs.active_match, 0);
        fs.next_match();
        assert_eq!(fs.active_match, 1);
        fs.next_match();
        assert_eq!(fs.active_match, 2);
        fs.next_match();
        assert_eq!(fs.active_match, 0); // wraps
        fs.prev_match();
        assert_eq!(fs.active_match, 2); // wraps back
    }

    #[test]
    fn test_find_empty_query() {
        let boxes = vec![make_text_box("Hello", 0.0)];
        let mut fs = FindState::new();
        fs.search(&boxes);
        assert_eq!(fs.matches.len(), 0);
        assert_eq!(fs.match_count_text(), "");
    }
}
