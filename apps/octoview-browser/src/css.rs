/// CSS inline style parser for OctoView browser.
/// Parses `style="..."` attribute values into OvdStyle structs.

use crate::ovd::{OvdStyle, TextAlign};

/// Parse an inline style attribute string into an OvdStyle.
///
/// Handles semicolon-separated property declarations like:
///   "color: red; font-size: 16px; font-weight: bold"
pub fn parse_inline_style(style_attr: &str) -> OvdStyle {
    let mut style = OvdStyle::default();

    for decl in style_attr.split(';') {
        let decl = decl.trim();
        if decl.is_empty() {
            continue;
        }

        // Split on first colon only (values can contain colons, e.g. rgb())
        let mut parts = decl.splitn(2, ':');
        let prop = match parts.next() {
            Some(p) => p.trim().to_lowercase(),
            None => continue,
        };
        let val = match parts.next() {
            Some(v) => v.trim(),
            None => continue,
        };

        match prop.as_str() {
            "color" => {
                style.color = parse_color(val);
            }
            "background-color" | "background" => {
                style.background = parse_color(val);
            }
            "font-weight" => {
                style.font_weight = parse_font_weight(val);
            }
            "font-size" => {
                style.font_size_px = parse_font_size(val);
            }
            "text-align" => {
                style.text_align = parse_text_align(val);
            }
            "border-color" => {
                style.border_color = parse_color(val);
            }
            "display" => {
                if val.eq_ignore_ascii_case("none") {
                    style.display_none = true;
                }
                style.display = parse_display(val);
            }
            "margin" => {
                let vals = parse_box_shorthand(val);
                style.margin_top = Some(vals.0);
                style.margin_right = Some(vals.1);
                style.margin_bottom = Some(vals.2);
                style.margin_left = Some(vals.3);
            }
            "margin-top" => { style.margin_top = parse_length(val); }
            "margin-bottom" => { style.margin_bottom = parse_length(val); }
            "margin-left" => { style.margin_left = parse_length(val); }
            "margin-right" => { style.margin_right = parse_length(val); }
            "padding" => {
                let vals = parse_box_shorthand(val);
                style.padding_top = Some(vals.0);
                style.padding_right = Some(vals.1);
                style.padding_bottom = Some(vals.2);
                style.padding_left = Some(vals.3);
            }
            "padding-top" => { style.padding_top = parse_length(val); }
            "padding-bottom" => { style.padding_bottom = parse_length(val); }
            "padding-left" => { style.padding_left = parse_length(val); }
            "padding-right" => { style.padding_right = parse_length(val); }
            "width" => { style.width = parse_length(val); }
            "max-width" => { style.max_width = parse_length(val); }
            "text-decoration" | "text-decoration-line" => {
                if val.to_lowercase().contains("underline") {
                    style.text_decoration_underline = true;
                }
            }
            "line-height" => { style.line_height = parse_length(val); }
            "border-width" => { style.border_width = parse_length(val); }
            "border" => {
                // Simple border shorthand: "1px solid #color"
                for part in val.split_whitespace() {
                    if let Some(w) = parse_length(part) {
                        style.border_width = Some(w);
                    }
                    if let Some(c) = parse_color(part) {
                        style.border_color = Some(c);
                    }
                }
            }
            _ => {}
        }
    }

    style
}

/// Parse a CSS color value into an RGB triple.
///
/// Supports:
/// - `#RGB` (shorthand hex)
/// - `#RRGGBB` (full hex)
/// - `rgb(r, g, b)` (functional notation)
/// - 16 named HTML colors (black, white, red, green, blue, etc.)
pub fn parse_color(val: &str) -> Option<[u8; 3]> {
    let val = val.trim();

    // Hex colors
    if val.starts_with('#') {
        let hex = &val[1..];
        return match hex.len() {
            3 => {
                // #RGB -> #RRGGBB
                let r = u8::from_str_radix(&hex[0..1], 16).ok()?;
                let g = u8::from_str_radix(&hex[1..2], 16).ok()?;
                let b = u8::from_str_radix(&hex[2..3], 16).ok()?;
                Some([r * 17, g * 17, b * 17])
            }
            6 => {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                Some([r, g, b])
            }
            _ => None,
        };
    }

    // rgb(r, g, b) functional notation
    let lower = val.to_lowercase();
    if lower.starts_with("rgb(") && lower.ends_with(')') {
        let inner = &lower[4..lower.len() - 1];
        let components: Vec<&str> = inner.split(',').collect();
        if components.len() == 3 {
            let r = components[0].trim().parse::<u8>().ok()?;
            let g = components[1].trim().parse::<u8>().ok()?;
            let b = components[2].trim().parse::<u8>().ok()?;
            return Some([r, g, b]);
        }
        return None;
    }

    // Named colors (16 basic HTML colors)
    match lower.as_str() {
        "black" => Some([0, 0, 0]),
        "white" => Some([255, 255, 255]),
        "red" => Some([255, 0, 0]),
        "green" => Some([0, 128, 0]),
        "blue" => Some([0, 0, 255]),
        "yellow" => Some([255, 255, 0]),
        "cyan" | "aqua" => Some([0, 255, 255]),
        "magenta" | "fuchsia" => Some([255, 0, 255]),
        "gray" | "grey" => Some([128, 128, 128]),
        "silver" => Some([192, 192, 192]),
        "maroon" => Some([128, 0, 0]),
        "olive" => Some([128, 128, 0]),
        "navy" => Some([0, 0, 128]),
        "purple" => Some([128, 0, 128]),
        "teal" => Some([0, 128, 128]),
        "orange" => Some([255, 165, 0]),
        _ => None,
    }
}

/// Parse a CSS font-size value into pixels.
///
/// Supports:
/// - `16px` (pixels)
/// - `12pt` (points, converted at 1pt = 1.333px)
/// - `16` (bare number, treated as px)
pub fn parse_font_size(val: &str) -> Option<f32> {
    let val = val.trim().to_lowercase();

    if val.ends_with("px") {
        let num = &val[..val.len() - 2];
        return num.trim().parse::<f32>().ok();
    }

    if val.ends_with("pt") {
        let num = &val[..val.len() - 2];
        let pt = num.trim().parse::<f32>().ok()?;
        // 1pt = 1.333px (96/72)
        return Some(pt * (96.0 / 72.0));
    }

    // Bare number
    val.parse::<f32>().ok()
}

/// Parse a CSS font-weight value.
///
/// Supports:
/// - `bold` (700)
/// - `normal` (400)
/// - numeric values (100-900)
pub fn parse_font_weight(val: &str) -> Option<u16> {
    let val = val.trim().to_lowercase();
    match val.as_str() {
        "bold" => Some(700),
        "normal" => Some(400),
        "lighter" => Some(300),
        "bolder" => Some(800),
        _ => val.parse::<u16>().ok(),
    }
}

/// Parse a CSS text-align value.
pub fn parse_text_align(val: &str) -> Option<TextAlign> {
    match val.trim().to_lowercase().as_str() {
        "left" => Some(TextAlign::Left),
        "center" => Some(TextAlign::Center),
        "right" => Some(TextAlign::Right),
        _ => None,
    }
}

/// Parse a CSS length value (px, em, pt, or bare number) into pixels.
pub fn parse_length(val: &str) -> Option<f32> {
    let val = val.trim().to_lowercase();
    if val == "0" {
        return Some(0.0);
    }
    if val == "auto" || val == "inherit" || val == "initial" {
        return None;
    }
    if val.ends_with("px") {
        return val[..val.len() - 2].trim().parse::<f32>().ok();
    }
    if val.ends_with("pt") {
        let pt = val[..val.len() - 2].trim().parse::<f32>().ok()?;
        return Some(pt * (96.0 / 72.0));
    }
    if val.ends_with("em") || val.ends_with("rem") {
        let end = if val.ends_with("rem") { 3 } else { 2 };
        let em = val[..val.len() - end].trim().parse::<f32>().ok()?;
        return Some(em * 16.0); // 1em = 16px default
    }
    // Bare number
    val.parse::<f32>().ok()
}

/// Parse CSS box shorthand (margin/padding) into (top, right, bottom, left).
pub fn parse_box_shorthand(val: &str) -> (f32, f32, f32, f32) {
    let parts: Vec<f32> = val
        .split_whitespace()
        .filter_map(|p| parse_length(p))
        .collect();
    match parts.len() {
        1 => (parts[0], parts[0], parts[0], parts[0]),
        2 => (parts[0], parts[1], parts[0], parts[1]),
        3 => (parts[0], parts[1], parts[2], parts[1]),
        4 => (parts[0], parts[1], parts[2], parts[3]),
        _ => (0.0, 0.0, 0.0, 0.0),
    }
}

/// Parse a CSS display value.
pub fn parse_display(val: &str) -> Option<crate::ovd::Display> {
    use crate::ovd::Display;
    match val.trim().to_lowercase().as_str() {
        "block" => Some(Display::Block),
        "inline" => Some(Display::Inline),
        "inline-block" => Some(Display::InlineBlock),
        "none" => Some(Display::None),
        "table-row" => Some(Display::TableRow),
        "table-cell" => Some(Display::TableCell),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_colors() {
        // #RRGGBB
        assert_eq!(parse_color("#ff0000"), Some([255, 0, 0]));
        assert_eq!(parse_color("#00ff00"), Some([0, 255, 0]));
        assert_eq!(parse_color("#0000ff"), Some([0, 0, 255]));
        assert_eq!(parse_color("#1a2b3c"), Some([0x1a, 0x2b, 0x3c]));

        // #RGB shorthand
        assert_eq!(parse_color("#f00"), Some([255, 0, 0]));
        assert_eq!(parse_color("#0f0"), Some([0, 255, 0]));
        assert_eq!(parse_color("#abc"), Some([0xaa, 0xbb, 0xcc]));

        // Invalid
        assert_eq!(parse_color("#xyz"), None);
        assert_eq!(parse_color("#12345"), None);
    }

    #[test]
    fn test_named_colors() {
        assert_eq!(parse_color("red"), Some([255, 0, 0]));
        assert_eq!(parse_color("blue"), Some([0, 0, 255]));
        assert_eq!(parse_color("black"), Some([0, 0, 0]));
        assert_eq!(parse_color("white"), Some([255, 255, 255]));
        assert_eq!(parse_color("green"), Some([0, 128, 0]));
        assert_eq!(parse_color("orange"), Some([255, 165, 0]));
        assert_eq!(parse_color("navy"), Some([0, 0, 128]));
        // Case insensitive
        assert_eq!(parse_color("Red"), Some([255, 0, 0]));
        assert_eq!(parse_color("BLUE"), Some([0, 0, 255]));
        // Unknown
        assert_eq!(parse_color("chartreuse"), None);
    }

    #[test]
    fn test_rgb_functional() {
        assert_eq!(parse_color("rgb(255, 0, 0)"), Some([255, 0, 0]));
        assert_eq!(parse_color("rgb(0,128,0)"), Some([0, 128, 0]));
        assert_eq!(parse_color("rgb( 10 , 20 , 30 )"), Some([10, 20, 30]));
        // Invalid
        assert_eq!(parse_color("rgb(300, 0, 0)"), None); // 300 > u8
        assert_eq!(parse_color("rgb(1,2)"), None); // too few
    }

    #[test]
    fn test_inline_style_parsing() {
        let style = parse_inline_style("color: red; font-size: 16px; font-weight: bold");
        assert_eq!(style.color, Some([255, 0, 0]));
        assert_eq!(style.font_size_px, Some(16.0));
        assert_eq!(style.font_weight, Some(700));

        // Background and text-align
        let style = parse_inline_style("background-color: #003366; text-align: center");
        assert_eq!(style.background, Some([0x00, 0x33, 0x66]));
        assert_eq!(style.text_align, Some(TextAlign::Center));

        // Trailing semicolons and spaces
        let style = parse_inline_style("  color: blue ;  ;  ");
        assert_eq!(style.color, Some([0, 0, 255]));

        // font-size with pt
        let style = parse_inline_style("font-size: 12pt");
        let px = style.font_size_px.unwrap();
        assert!((px - 16.0).abs() < 0.1);

        // border-color
        let style = parse_inline_style("border-color: #ff0000");
        assert_eq!(style.border_color, Some([255, 0, 0]));
    }

    #[test]
    fn test_display_none() {
        let style = parse_inline_style("display: none");
        assert!(style.display_none);

        let style = parse_inline_style("display: block");
        assert!(!style.display_none);

        let style = parse_inline_style("display: None");
        assert!(style.display_none);
    }

    #[test]
    fn test_font_weight_variants() {
        assert_eq!(parse_font_weight("bold"), Some(700));
        assert_eq!(parse_font_weight("normal"), Some(400));
        assert_eq!(parse_font_weight("600"), Some(600));
        assert_eq!(parse_font_weight("lighter"), Some(300));
        assert_eq!(parse_font_weight("bolder"), Some(800));
        assert_eq!(parse_font_weight("invalid"), None);
    }

    #[test]
    fn test_parse_length() {
        assert_eq!(parse_length("16px"), Some(16.0));
        assert_eq!(parse_length("0"), Some(0.0));
        assert_eq!(parse_length("10"), Some(10.0));
        assert_eq!(parse_length("12pt"), Some(16.0));
        assert_eq!(parse_length("1em"), Some(16.0));
        assert_eq!(parse_length("1.5rem"), Some(24.0));
        assert_eq!(parse_length("auto"), None);
        assert_eq!(parse_length("inherit"), None);
    }

    #[test]
    fn test_parse_box_shorthand() {
        // 1 value: all sides
        assert_eq!(parse_box_shorthand("10px"), (10.0, 10.0, 10.0, 10.0));
        // 2 values: top/bottom, left/right
        assert_eq!(parse_box_shorthand("10px 20px"), (10.0, 20.0, 10.0, 20.0));
        // 3 values: top, left/right, bottom
        assert_eq!(parse_box_shorthand("10px 20px 30px"), (10.0, 20.0, 30.0, 20.0));
        // 4 values: top, right, bottom, left
        assert_eq!(parse_box_shorthand("10px 20px 30px 40px"), (10.0, 20.0, 30.0, 40.0));
    }

    #[test]
    fn test_parse_display() {
        use crate::ovd::Display;
        assert_eq!(parse_display("block"), Some(Display::Block));
        assert_eq!(parse_display("inline"), Some(Display::Inline));
        assert_eq!(parse_display("none"), Some(Display::None));
        assert_eq!(parse_display("inline-block"), Some(Display::InlineBlock));
        assert_eq!(parse_display("flex"), None); // not supported yet
    }

    #[test]
    fn test_inline_style_box_model() {
        let style = parse_inline_style("margin: 10px 20px; padding: 5px");
        assert_eq!(style.margin_top, Some(10.0));
        assert_eq!(style.margin_right, Some(20.0));
        assert_eq!(style.margin_bottom, Some(10.0));
        assert_eq!(style.margin_left, Some(20.0));
        assert_eq!(style.padding_top, Some(5.0));
        assert_eq!(style.padding_left, Some(5.0));
    }

    #[test]
    fn test_inline_style_width() {
        let style = parse_inline_style("width: 300px; max-width: 600px");
        assert_eq!(style.width, Some(300.0));
        assert_eq!(style.max_width, Some(600.0));
    }

    #[test]
    fn test_inline_style_text_decoration() {
        let style = parse_inline_style("text-decoration: underline");
        assert!(style.text_decoration_underline);

        let style = parse_inline_style("text-decoration: none");
        assert!(!style.text_decoration_underline);
    }

    #[test]
    fn test_inline_style_border() {
        let style = parse_inline_style("border: 1px solid red");
        assert_eq!(style.border_width, Some(1.0));
        assert_eq!(style.border_color, Some([255, 0, 0]));
    }
}
