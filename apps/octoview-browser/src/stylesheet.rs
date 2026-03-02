/// CSS stylesheet parser for OctoView browser.
///
/// Parses CSS stylesheet text (from `<style>` blocks or external files)
/// into a list of CssRule structs. Each rule has a raw selector string
/// and a list of property:value declarations.

/// A single CSS property declaration (e.g., `color: red`).
#[derive(Debug, Clone)]
pub struct CssDeclaration {
    pub property: String,
    pub value: String,
}

/// A CSS rule consisting of a selector string and its declarations.
/// Selectors are pre-parsed at stylesheet parse time for performance.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CssRule {
    pub selector_text: String,
    pub declarations: Vec<CssDeclaration>,
    pub parsed_selectors: Vec<crate::selector_match::SelectorChain>,
}

/// A parsed CSS stylesheet containing an ordered list of rules.
#[derive(Debug, Clone, Default)]
pub struct Stylesheet {
    pub rules: Vec<CssRule>,
}

impl Stylesheet {
    /// Parse a raw CSS string into a Stylesheet.
    ///
    /// Handles comment stripping, rule extraction via balanced-brace splitting,
    /// and declaration parsing. Skips at-rules (e.g., `@media`, `@import`).
    pub fn parse(css_text: &str) -> Self {
        let mut stylesheet = Stylesheet::default();
        let css = strip_comments(css_text);

        // Split on balanced braces to find rules
        let mut pos = 0;
        while pos < css.len() {
            // Skip whitespace
            let trimmed = css[pos..].trim_start();
            pos = css.len() - trimmed.len();

            if pos >= css.len() {
                break;
            }

            // Skip at-rules that are terminated by semicolons (no block).
            // Examples: @import url("..."); @charset "UTF-8";
            if css[pos..].starts_with('@') {
                // Check if this at-rule has a block or is semicolon-terminated
                let semi_pos = css[pos..].find(';');
                let brace_pos = find_opening_brace(&css[pos..]);
                match (semi_pos, brace_pos) {
                    (Some(s), Some(b)) if s < b => {
                        // Semicolon comes before brace: statement at-rule, skip it
                        pos += s + 1;
                        continue;
                    }
                    (Some(s), None) => {
                        // No brace at all: skip to semicolon
                        pos += s + 1;
                        continue;
                    }
                    _ => {
                        // Brace comes first (e.g., @media): fall through to brace handling
                    }
                }
            }

            // Find opening brace
            if let Some(brace_offset) = find_opening_brace(&css[pos..]) {
                let selector_text = css[pos..pos + brace_offset].trim().to_string();
                let block_start = pos + brace_offset + 1;

                // Find matching closing brace (handle nested braces for @media etc.)
                if let Some(brace_end_offset) = find_matching_close_brace(&css[block_start..]) {
                    let block_text = &css[block_start..block_start + brace_end_offset];

                    if !selector_text.is_empty() && !selector_text.starts_with('@') {
                        let declarations = parse_declarations(block_text);
                        let parsed_selectors = crate::selector_match::parse_selector_list(&selector_text);
                        stylesheet.rules.push(CssRule {
                            selector_text,
                            declarations,
                            parsed_selectors,
                        });
                    }
                    pos = block_start + brace_end_offset + 1;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        stylesheet
    }
}

/// Remove CSS comments (`/* ... */`) from a string.
fn strip_comments(css: &str) -> String {
    let mut result = String::with_capacity(css.len());
    let mut chars = css.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '/' {
            if chars.peek() == Some(&'*') {
                chars.next(); // consume *
                // Skip until */
                loop {
                    match chars.next() {
                        Some('*') if chars.peek() == Some(&'/') => {
                            chars.next();
                            break;
                        }
                        None => break,
                        _ => {}
                    }
                }
                // Insert a space to avoid joining tokens across comment boundaries
                result.push(' ');
                continue;
            }
        }
        result.push(c);
    }
    result
}

/// Find the position of the first unquoted `{` in a string.
fn find_opening_brace(s: &str) -> Option<usize> {
    let mut in_single_quote = false;
    let mut in_double_quote = false;
    for (i, c) in s.char_indices() {
        match c {
            '\'' if !in_double_quote => in_single_quote = !in_single_quote,
            '"' if !in_single_quote => in_double_quote = !in_double_quote,
            '{' if !in_single_quote && !in_double_quote => return Some(i),
            _ => {}
        }
    }
    None
}

/// Find the position of the matching `}` for an already-opened block.
/// Handles nested braces.
fn find_matching_close_brace(s: &str) -> Option<usize> {
    let mut depth = 0;
    let mut in_single_quote = false;
    let mut in_double_quote = false;
    for (i, c) in s.char_indices() {
        match c {
            '\'' if !in_double_quote => in_single_quote = !in_single_quote,
            '"' if !in_single_quote => in_double_quote = !in_double_quote,
            '{' if !in_single_quote && !in_double_quote => depth += 1,
            '}' if !in_single_quote && !in_double_quote => {
                if depth == 0 {
                    return Some(i);
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    None
}

/// Parse CSS declarations from a block body (the text between `{` and `}`).
///
/// Each declaration is `property: value` separated by semicolons.
fn parse_declarations(block: &str) -> Vec<CssDeclaration> {
    block
        .split(';')
        .filter_map(|decl| {
            let decl = decl.trim();
            if decl.is_empty() {
                return None;
            }
            let mut parts = decl.splitn(2, ':');
            let prop = parts.next()?.trim().to_lowercase();
            let val = parts.next()?.trim().to_string();
            if prop.is_empty() {
                return None;
            }
            Some(CssDeclaration {
                property: prop,
                value: val,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_rule() {
        let css = "h1 { color: red; font-size: 24px; }";
        let sheet = Stylesheet::parse(css);
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(sheet.rules[0].selector_text, "h1");
        assert_eq!(sheet.rules[0].declarations.len(), 2);
        assert_eq!(sheet.rules[0].declarations[0].property, "color");
        assert_eq!(sheet.rules[0].declarations[0].value, "red");
        assert_eq!(sheet.rules[0].declarations[1].property, "font-size");
        assert_eq!(sheet.rules[0].declarations[1].value, "24px");
    }

    #[test]
    fn test_parse_multiple_rules() {
        let css = r#"
            h1 { color: blue; }
            p { font-size: 16px; }
            a { text-decoration: underline; }
        "#;
        let sheet = Stylesheet::parse(css);
        assert_eq!(sheet.rules.len(), 3);
        assert_eq!(sheet.rules[0].selector_text, "h1");
        assert_eq!(sheet.rules[1].selector_text, "p");
        assert_eq!(sheet.rules[2].selector_text, "a");
        assert_eq!(sheet.rules[0].declarations[0].value, "blue");
        assert_eq!(sheet.rules[1].declarations[0].value, "16px");
        assert_eq!(sheet.rules[2].declarations[0].value, "underline");
    }

    #[test]
    fn test_parse_class_selector() {
        let css = ".container { margin: 0 auto; padding: 10px; }";
        let sheet = Stylesheet::parse(css);
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(sheet.rules[0].selector_text, ".container");
        assert_eq!(sheet.rules[0].declarations.len(), 2);
        assert_eq!(sheet.rules[0].declarations[0].property, "margin");
        assert_eq!(sheet.rules[0].declarations[0].value, "0 auto");
    }

    #[test]
    fn test_parse_id_selector() {
        let css = "#main { width: 960px; background: #fff; }";
        let sheet = Stylesheet::parse(css);
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(sheet.rules[0].selector_text, "#main");
        assert_eq!(sheet.rules[0].declarations.len(), 2);
        assert_eq!(sheet.rules[0].declarations[0].property, "width");
        assert_eq!(sheet.rules[0].declarations[1].property, "background");
    }

    #[test]
    fn test_parse_compound_selector() {
        let css = "div.container > h1 { color: green; }";
        let sheet = Stylesheet::parse(css);
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(sheet.rules[0].selector_text, "div.container > h1");
        assert_eq!(sheet.rules[0].declarations[0].property, "color");
        assert_eq!(sheet.rules[0].declarations[0].value, "green");
    }

    #[test]
    fn test_parse_comma_selectors() {
        let css = "h1, h2, h3 { font-weight: bold; }";
        let sheet = Stylesheet::parse(css);
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(sheet.rules[0].selector_text, "h1, h2, h3");
        assert_eq!(sheet.rules[0].declarations[0].property, "font-weight");
        assert_eq!(sheet.rules[0].declarations[0].value, "bold");
    }

    #[test]
    fn test_strip_comments() {
        let css = "/* header styles */ h1 { color: red; /* inline comment */ }";
        let sheet = Stylesheet::parse(css);
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(sheet.rules[0].selector_text, "h1");
        assert_eq!(sheet.rules[0].declarations.len(), 1);
        assert_eq!(sheet.rules[0].declarations[0].property, "color");
        assert_eq!(sheet.rules[0].declarations[0].value, "red");
    }

    #[test]
    fn test_parse_empty_declarations() {
        let css = "h1 { }";
        let sheet = Stylesheet::parse(css);
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(sheet.rules[0].selector_text, "h1");
        assert_eq!(sheet.rules[0].declarations.len(), 0);
    }

    #[test]
    fn test_parse_nested_braces_skip() {
        // @media rules should be skipped without crashing
        let css = r#"
            @media screen and (max-width: 600px) {
                h1 { color: red; }
            }
            p { font-size: 14px; }
        "#;
        let sheet = Stylesheet::parse(css);
        // The @media rule is skipped, the p rule is parsed
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(sheet.rules[0].selector_text, "p");
        assert_eq!(sheet.rules[0].declarations[0].value, "14px");
    }

    #[test]
    fn test_strip_comments_function() {
        assert_eq!(strip_comments("/* comment */"), " ");
        assert_eq!(strip_comments("a /* x */ b"), "a   b");
        assert_eq!(strip_comments("no comments"), "no comments");
        assert_eq!(strip_comments("a/* nested /* not really */b"), "a b");
        assert_eq!(strip_comments("/**/"), " ");
    }

    #[test]
    fn test_parse_declarations_trailing_semicolon() {
        let decls = parse_declarations("color: red; font-size: 16px;");
        assert_eq!(decls.len(), 2);
        assert_eq!(decls[0].property, "color");
        assert_eq!(decls[1].property, "font-size");
    }

    #[test]
    fn test_parse_declarations_no_semicolon() {
        let decls = parse_declarations("color: red");
        assert_eq!(decls.len(), 1);
        assert_eq!(decls[0].property, "color");
        assert_eq!(decls[0].value, "red");
    }

    #[test]
    fn test_property_lowercased() {
        let css = "h1 { Color: Red; FONT-SIZE: 20px; }";
        let sheet = Stylesheet::parse(css);
        assert_eq!(sheet.rules[0].declarations[0].property, "color");
        assert_eq!(sheet.rules[0].declarations[1].property, "font-size");
        // Values preserve original casing
        assert_eq!(sheet.rules[0].declarations[0].value, "Red");
    }

    #[test]
    fn test_value_with_colon() {
        // Values can contain colons (e.g., rgb() or URL)
        let css = "a { background: url(http://example.com/img.png); }";
        let sheet = Stylesheet::parse(css);
        assert_eq!(sheet.rules[0].declarations[0].property, "background");
        assert_eq!(
            sheet.rules[0].declarations[0].value,
            "url(http://example.com/img.png)"
        );
    }

    #[test]
    fn test_multiple_at_rules_skipped() {
        let css = r#"
            @import url("style.css");
            @charset "UTF-8";
            body { margin: 0; }
        "#;
        let sheet = Stylesheet::parse(css);
        // @import has no braces so parser advances past it.
        // @charset also has no braces. body rule is parsed.
        // However our parser looks for { ... }, and @import/@charset don't have blocks,
        // so the parser will find body { margin: 0 } correctly.
        assert!(sheet.rules.iter().any(|r| r.selector_text == "body"));
    }
}
