use std::collections::HashMap;

use crate::framebuffer::Framebuffer;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum FontStyle {
    Regular,
    Bold,
    Monospace,
    MonospaceBold,
}

#[derive(Hash, PartialEq, Eq, Clone)]
struct GlyphKey {
    ch: char,
    size_px: u32, // font size in pixels (integer for hash)
    style: FontStyle,
}

struct GlyphEntry {
    metrics: fontdue::Metrics,
    bitmap: Vec<u8>, // alpha coverage per pixel
}

pub struct TextRenderer {
    font_regular: fontdue::Font,
    font_bold: fontdue::Font,
    font_mono: fontdue::Font,
    font_mono_bold: fontdue::Font,
    glyph_cache: HashMap<GlyphKey, GlyphEntry>,
}

impl TextRenderer {
    pub fn new() -> Self {
        // Embed Inter from OctoUI — no runtime file dependencies
        const INTER_TTF: &[u8] = include_bytes!("../../../octoui/fonts/Inter.ttf");

        let regular_settings = fontdue::FontSettings::default();
        let font_regular =
            fontdue::Font::from_bytes(INTER_TTF as &[u8], regular_settings).expect("Inter.ttf");

        // Bold: same Inter font (fontdue doesn't do weight variants, but Inter
        // looks acceptable at all sizes — v2 can add Inter-Bold if needed)
        let bold_settings = fontdue::FontSettings::default();
        let font_bold =
            fontdue::Font::from_bytes(INTER_TTF as &[u8], bold_settings).expect("Inter.ttf bold");

        // Monospace: Consolas for code blocks (Inter is proportional)
        let mono_data =
            std::fs::read("C:\\Windows\\Fonts\\consola.ttf").expect("Consolas not found");
        let mono_bold_data =
            std::fs::read("C:\\Windows\\Fonts\\consolab.ttf").expect("Consolas Bold not found");

        Self {
            font_regular,
            font_bold,
            font_mono: fontdue::Font::from_bytes(mono_data, fontdue::FontSettings::default())
                .unwrap(),
            font_mono_bold: fontdue::Font::from_bytes(
                mono_bold_data,
                fontdue::FontSettings::default(),
            )
            .unwrap(),
            glyph_cache: HashMap::new(),
        }
    }

    fn font_for_style(&self, style: FontStyle) -> &fontdue::Font {
        match style {
            FontStyle::Regular => &self.font_regular,
            FontStyle::Bold => &self.font_bold,
            FontStyle::Monospace => &self.font_mono,
            FontStyle::MonospaceBold => &self.font_mono_bold,
        }
    }

    #[allow(dead_code)]
    fn rasterize(&mut self, ch: char, size: f32, style: FontStyle) -> &GlyphEntry {
        let key = GlyphKey {
            ch,
            size_px: size as u32,
            style,
        };
        if !self.glyph_cache.contains_key(&key) {
            let font = self.font_for_style(style);
            let (metrics, bitmap) = font.rasterize(ch, size);
            self.glyph_cache.insert(key.clone(), GlyphEntry { metrics, bitmap });
        }
        self.glyph_cache.get(&key).unwrap()
    }

    /// Measure the width and height of a text string at the given size.
    pub fn measure_text(&mut self, text: &str, size: f32, style: FontStyle) -> (f32, f32) {
        let mut width = 0.0f32;
        let font = self.font_for_style(style);
        for ch in text.chars() {
            let metrics = font.metrics(ch, size);
            width += metrics.advance_width;
        }
        (width, self.line_height(size))
    }

    /// Draw a text string onto the framebuffer at (x, y) which is the top-left baseline origin.
    pub fn draw_text(
        &mut self,
        fb: &mut Framebuffer,
        text: &str,
        x: i32,
        y: i32,
        size: f32,
        style: FontStyle,
        r: u8,
        g: u8,
        b: u8,
    ) {
        let mut cursor_x = x as f32;
        let font = self.font_for_style(style);
        // Use real font ascent from the font's metrics table
        let ascent = font
            .horizontal_line_metrics(size)
            .map(|m| m.ascent)
            .unwrap_or(size * 0.8);

        for ch in text.chars() {
            let key = GlyphKey {
                ch,
                size_px: size as u32,
                style,
            };
            if !self.glyph_cache.contains_key(&key) {
                let font = self.font_for_style(style);
                let (metrics, bitmap) = font.rasterize(ch, size);
                self.glyph_cache.insert(key.clone(), GlyphEntry { metrics, bitmap });
            }
            let entry = self.glyph_cache.get(&key).unwrap();

            let gx = cursor_x as i32 + entry.metrics.xmin;
            // Baseline is at y + ascent. Glyph top = baseline - (ymin + height).
            // So bitmap top-left Y = y + ascent - ymin - height.
            let gy = y + ascent as i32 - entry.metrics.ymin - entry.metrics.height as i32;

            for row in 0..entry.metrics.height {
                for col in 0..entry.metrics.width {
                    let coverage = entry.bitmap[row * entry.metrics.width + col];
                    if coverage > 0 {
                        fb.blend_pixel(
                            gx + col as i32,
                            gy + row as i32,
                            r,
                            g,
                            b,
                            coverage,
                        );
                    }
                }
            }

            cursor_x += entry.metrics.advance_width;
        }
    }

    /// Line height for a given font size, derived from real font metrics.
    pub fn line_height(&self, size: f32) -> f32 {
        self.font_regular
            .horizontal_line_metrics(size)
            .map(|m| (m.ascent - m.descent + m.line_gap).ceil())
            .unwrap_or((size * 1.4).ceil())
    }

    /// Wrap text into lines that fit within max_width pixels.
    pub fn wrap_text(
        &mut self,
        text: &str,
        max_width: f32,
        size: f32,
        style: FontStyle,
    ) -> Vec<String> {
        if text.is_empty() {
            return vec![String::new()];
        }

        let mut lines: Vec<String> = Vec::new();
        let mut current_line = String::new();
        let mut current_width = 0.0f32;

        for word in text.split_whitespace() {
            let (word_width, _) = self.measure_text(word, size, style);

            if current_line.is_empty() {
                // First word on line — always add it even if it overflows
                current_line.push_str(word);
                current_width = word_width;
            } else {
                let (space_width, _) = self.measure_text(" ", size, style);
                let new_width = current_width + space_width + word_width;
                if new_width > max_width {
                    // Wrap to next line
                    lines.push(current_line);
                    current_line = word.to_string();
                    current_width = word_width;
                } else {
                    current_line.push(' ');
                    current_line.push_str(word);
                    current_width = new_width;
                }
            }
        }

        if !current_line.is_empty() {
            lines.push(current_line);
        }

        if lines.is_empty() {
            lines.push(String::new());
        }

        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_renderer_creation() {
        let _tr = TextRenderer::new();
    }

    #[test]
    fn test_measure_text() {
        let mut tr = TextRenderer::new();
        let (w, h) = tr.measure_text("Hello", 16.0, FontStyle::Regular);
        assert!(w > 0.0, "Text width should be positive");
        assert!(h > 0.0, "Text height should be positive");
    }

    #[test]
    fn test_wrap_text() {
        let mut tr = TextRenderer::new();
        let lines = tr.wrap_text(
            "This is a test of text wrapping in the browser",
            100.0,
            14.0,
            FontStyle::Regular,
        );
        assert!(lines.len() > 1, "Should wrap to multiple lines");
    }

    #[test]
    fn test_draw_text() {
        let mut tr = TextRenderer::new();
        let mut fb = Framebuffer::new(200, 50);
        fb.clear(0, 0, 0);
        tr.draw_text(&mut fb, "Hi", 10, 10, 16.0, FontStyle::Regular, 255, 255, 255);
        // Check that some pixels got blended (non-zero)
        let has_nonzero = fb.pixels.iter().any(|&p| p > 0 && p < 255);
        assert!(has_nonzero, "Text should produce blended pixels");
    }
}
