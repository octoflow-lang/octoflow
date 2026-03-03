# OctoView CSS Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add CSS stylesheet support (style blocks + external sheets + selector matching + cascade) so websites render with correct styles.

**Architecture:** Match CSS selectors against html5ever's RcDom during extraction (while DOM structure is available), cascade by specificity, merge with inline styles, store in existing OvdStyle. Layout and rendering pipelines unchanged.

**Tech Stack:** cssparser 0.36, selectors 0.35, html5ever 0.38 (existing)

---

### Task 1: Add CSS crate dependencies

**Files:**
- Modify: `apps/octoview-browser/Cargo.toml`

**Step 1: Add cssparser and selectors dependencies**

```toml
cssparser = "0.36"
selectors = "0.35"
```

**Step 2: Verify it compiles**

Run: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\OctoFlow\run_test.ps1" build -p octoview-browser`
Expected: Build succeeds (may have warnings)

**Step 3: Commit**

```
feat(browser): add cssparser + selectors crate dependencies
```

---

### Task 2: Create stylesheet types and parser module

**Files:**
- Create: `apps/octoview-browser/src/stylesheet.rs`
- Modify: `apps/octoview-browser/src/main.rs` (add mod declaration)

**Step 1: Define core types**

Create `stylesheet.rs` with:

```rust
/// CSS stylesheet parser and rule storage for OctoView browser.
/// Uses Servo's cssparser + selectors crates.

use cssparser::{Parser, ParserInput, Token, ParseError, BasicParseErrorKind};
use crate::css::parse_color;
use crate::ovd::OvdStyle;

/// A single CSS declaration (property: value).
#[derive(Debug, Clone)]
pub struct CssDeclaration {
    pub property: String,
    pub value: String,
}

/// A CSS rule: selector text + declarations.
#[derive(Debug, Clone)]
pub struct CssRule {
    pub selector_text: String,
    pub declarations: Vec<CssDeclaration>,
}

/// Collection of all CSS rules from all sources.
#[derive(Debug, Clone, Default)]
pub struct Stylesheet {
    pub rules: Vec<CssRule>,
}
```

**Step 2: Implement CSS declaration parser**

Parse declaration blocks like `{ color: red; font-size: 16px; }` using cssparser:

```rust
impl Stylesheet {
    /// Parse a CSS stylesheet string into rules.
    pub fn parse(css_text: &str) -> Self {
        let mut stylesheet = Stylesheet::default();
        let mut input = ParserInput::new(css_text);
        let mut parser = Parser::new(&mut input);

        // Parse top-level rules
        while !parser.is_exhausted() {
            if let Ok(rule) = parse_qualified_rule(&mut parser) {
                stylesheet.rules.push(rule);
            } else {
                // Skip to next rule on error
                let _ = parser.next();
            }
        }
        stylesheet
    }
}
```

**Step 3: Parse individual qualified rules (selector + block)**

```rust
fn parse_qualified_rule(parser: &mut Parser) -> Result<CssRule, ()> {
    // Collect selector tokens until '{'
    let selector_text = parse_selector_text(parser)?;

    // Parse declaration block
    let declarations = parse_declaration_block(parser)?;

    Ok(CssRule { selector_text, declarations })
}
```

**Step 4: Parse declaration blocks**

```rust
fn parse_declaration_block(parser: &mut Parser) -> Result<Vec<CssDeclaration>, ()> {
    let mut declarations = Vec::new();
    parser.parse_nested_block(|parser| {
        loop {
            let result = parser.parse_until_after(
                cssparser::Delimiter::Semicolon,
                |parser| parse_one_declaration(parser),
            );
            if let Ok(decl) = result {
                declarations.push(decl);
            }
            if parser.is_exhausted() {
                break;
            }
        }
        Ok(())
    }).map_err(|_: ParseError<'_, ()>| ())?;
    Ok(declarations)
}
```

**Step 5: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_rule() {
        let css = "h1 { color: red; font-size: 24px; }";
        let sheet = Stylesheet::parse(css);
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(sheet.rules[0].selector_text.trim(), "h1");
        assert_eq!(sheet.rules[0].declarations.len(), 2);
    }

    #[test]
    fn test_parse_multiple_rules() {
        let css = "h1 { color: red; } p { color: blue; } .highlight { background: yellow; }";
        let sheet = Stylesheet::parse(css);
        assert_eq!(sheet.rules.len(), 3);
    }

    #[test]
    fn test_parse_class_selector() {
        let css = ".container { color: #333; }";
        let sheet = Stylesheet::parse(css);
        assert_eq!(sheet.rules[0].selector_text.trim(), ".container");
    }

    #[test]
    fn test_parse_id_selector() {
        let css = "#main { font-size: 16px; }";
        let sheet = Stylesheet::parse(css);
        assert_eq!(sheet.rules[0].selector_text.trim(), "#main");
    }
}
```

**Step 6: Run tests**

Run: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\OctoFlow\run_test.ps1" test -p octoview-browser`
Expected: All tests pass

**Step 7: Commit**

```
feat(browser): CSS stylesheet parser with rule extraction
```

---

### Task 3: CSS selector matching engine

**Files:**
- Create: `apps/octoview-browser/src/selector_match.rs`
- Modify: `apps/octoview-browser/src/main.rs` (add mod)

**Step 1: Implement a simple selector matcher**

Rather than using the full `selectors` crate Element trait (complex, requires custom SelectorImpl), implement a pragmatic selector matcher that handles the most common patterns:

```rust
/// Simple CSS selector matcher for OctoView.
/// Matches selectors against DOM nodes during extraction.
///
/// Supported selector patterns:
/// - Tag: h1, p, div, a
/// - Class: .classname
/// - ID: #idname
/// - Descendant: div p (p inside div)
/// - Child: div > p (direct child)
/// - Multiple classes: .foo.bar
/// - Compound: h1.title, div#main
/// - Comma groups: h1, h2, h3

use markup5ever_rcdom::{Handle, NodeData};

#[derive(Debug, Clone)]
pub struct SimpleSelector {
    pub tag: Option<String>,
    pub id: Option<String>,
    pub classes: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum Combinator {
    Descendant,  // space
    Child,       // >
}

#[derive(Debug, Clone)]
pub struct SelectorChain {
    pub parts: Vec<(SimpleSelector, Option<Combinator>)>,
}

/// Specificity as (id_count, class_count, tag_count)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Specificity(pub u32, pub u32, pub u32);
```

**Step 2: Parse selector text into SelectorChain**

Parse selector strings like `div.container > h1.title` into structured parts.

**Step 3: Implement matching**

Match a SelectorChain against a DOM node by walking up the ancestor chain:

```rust
pub fn matches(chain: &SelectorChain, node: &Handle, ancestors: &[Handle]) -> bool
```

**Step 4: Implement specificity calculation**

```rust
impl SelectorChain {
    pub fn specificity(&self) -> Specificity {
        let mut ids = 0;
        let mut classes = 0;
        let mut tags = 0;
        for (sel, _) in &self.parts {
            if sel.id.is_some() { ids += 1; }
            classes += sel.classes.len() as u32;
            if sel.tag.is_some() { tags += 1; }
        }
        Specificity(ids, classes, tags)
    }
}
```

**Step 5: Write tests**

Tests for: tag matching, class matching, id matching, compound selectors, descendant combinators, child combinators, specificity ordering, comma-separated groups.

**Step 6: Run tests, commit**

```
feat(browser): CSS selector matching with specificity
```

---

### Task 4: Extract style blocks and external stylesheets

**Files:**
- Modify: `apps/octoview-browser/src/extract.rs`
- Modify: `apps/octoview-browser/src/page.rs`
- Modify: `apps/octoview-browser/src/fetch.rs`

**Step 1: Extract `<style>` block content during HTML walk**

In extract.rs, before the main extraction walk, do a first pass to collect all `<style>` element text content.

**Step 2: Extract `<link rel="stylesheet">` URLs**

Find all `<link rel="stylesheet" href="...">` elements and collect their URLs.

**Step 3: Fetch external stylesheets**

In page.rs, after fetching HTML, fetch each external stylesheet URL (with timeout, error tolerance).

**Step 4: Combine all stylesheets**

Merge style block rules + external sheet rules into a single Stylesheet, maintaining source order.

**Step 5: Write tests, commit**

```
feat(browser): extract CSS from style blocks and link elements
```

---

### Task 5: Apply matched styles during extraction

**Files:**
- Modify: `apps/octoview-browser/src/extract.rs`
- Modify: `apps/octoview-browser/src/page.rs`

**Step 1: Pass stylesheet + ancestor stack through extraction**

Modify `walk_node()` to accept a `&Stylesheet` and `&[Handle]` ancestor stack. Each recursive call pushes the current node.

**Step 2: Match and cascade styles for each node**

For each OVD node created during extraction:
1. Collect all matching CSS rules from the stylesheet
2. Sort by specificity (lowest first)
3. Build cascaded OvdStyle from matched declarations (later rules override earlier)
4. Merge with inline style (inline has highest specificity)
5. Store in node.style

**Step 3: Implement style inheritance**

After cascade, inherit certain properties from parent if not explicitly set:
- `color` — inherits
- `font-size` — inherits
- `font-weight` — inherits
- `font-family` — inherits (when we support it)
- `text-align` — inherits
- `background` — does NOT inherit

**Step 4: Write integration tests**

Test with HTML that has `<style>` blocks:
```html
<style>h1 { color: red; } .blue { color: blue; }</style>
<h1>Red heading</h1>
<p class="blue">Blue text</p>
```

**Step 5: Run tests, commit**

```
feat(browser): apply cascaded CSS styles during extraction
```

---

### Task 6: Extend OvdStyle for more CSS properties

**Files:**
- Modify: `apps/octoview-browser/src/ovd.rs`
- Modify: `apps/octoview-browser/src/css.rs`
- Modify: `apps/octoview-browser/src/layout.rs`

**Step 1: Add new style properties**

```rust
pub struct OvdStyle {
    // existing...
    pub margin_top: Option<f32>,
    pub margin_bottom: Option<f32>,
    pub margin_left: Option<f32>,
    pub margin_right: Option<f32>,
    pub padding_top: Option<f32>,
    pub padding_bottom: Option<f32>,
    pub padding_left: Option<f32>,
    pub padding_right: Option<f32>,
    pub border_width: Option<f32>,
    pub width: Option<f32>,
    pub max_width: Option<f32>,
    pub text_decoration_underline: bool,
    pub text_decoration_line_through: bool,
    pub font_italic: bool,
}
```

**Step 2: Parse new properties in css.rs**

Add cases to declaration parsing for margin, padding, border-width, width, max-width, text-decoration, font-style.

**Step 3: Apply new properties in layout.rs**

Map new OvdStyle fields to LayoutStyle, replacing hardcoded margins/padding with CSS values (falling back to type-specific defaults).

**Step 4: Write tests, commit**

```
feat(browser): extend CSS property support (margin, padding, text-decoration)
```

---

### Task 7: Handle @media and @import rules

**Files:**
- Modify: `apps/octoview-browser/src/stylesheet.rs`

**Step 1: Skip @-rules gracefully**

During stylesheet parsing, detect at-rules (`@media`, `@import`, `@font-face`, `@keyframes`) and handle them:
- `@import url(...)` — fetch and parse the imported stylesheet
- `@media screen` / `@media all` — parse inner rules (always applies)
- `@media print` — skip (we're screen-only)
- Others — skip gracefully

**Step 2: Handle @media viewport width**

For `@media (max-width: Npx)` and `@media (min-width: Npx)`, evaluate against viewport width and include/exclude rules.

**Step 3: Write tests, commit**

```
feat(browser): handle @import and @media rules in stylesheets
```

---

### Task 8: Visual verification and polish

**Files:**
- Various

**Step 1: Test with real websites**

Load Wikipedia, Hacker News, GitHub in the browser. Identify rendering issues.

**Step 2: Fix top issues**

Address the most visible rendering problems found during testing.

**Step 3: Commit**

```
fix(browser): CSS rendering fixes from real-world testing
```
