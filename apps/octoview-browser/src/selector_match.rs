/// CSS selector matching engine for OctoView browser.
///
/// Parses CSS selector strings and matches them against DOM nodes.
/// Supports tag, class, id, compound, descendant (space), and child (>)
/// combinators, plus specificity calculation for cascade ordering.

use markup5ever_rcdom::{Handle, NodeData};

/// A single simple selector component (e.g., `div`, `.class`, `#id`, or `div.class#id`).
#[derive(Debug, Clone, PartialEq)]
pub struct SimpleSelector {
    pub tag: Option<String>,
    pub id: Option<String>,
    pub classes: Vec<String>,
}

/// How adjacent selectors combine in a chain.
#[derive(Debug, Clone, PartialEq)]
pub enum Combinator {
    /// Space combinator: matches any ancestor.
    Descendant,
    /// `>` combinator: matches direct parent only.
    Child,
}

/// A chain of simple selectors with combinators.
///
/// Parts are stored from RIGHT to LEFT (matching order: start at element, walk up).
/// For `div.container > p.intro`, parts would be:
///   `[(p.intro, None), (div.container, Some(Child))]`
///
/// The first element's combinator is always `None` (it is the key selector).
/// Subsequent elements carry the combinator that connects them to the previous part.
#[derive(Debug, Clone)]
pub struct SelectorChain {
    pub parts: Vec<(SimpleSelector, Option<Combinator>)>,
}

/// CSS specificity: (ids, classes+attributes, tags).
///
/// Higher specificity takes precedence in the cascade.
/// Compared lexicographically: ids > classes > tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Specificity(pub u32, pub u32, pub u32);

impl Specificity {
    /// Calculate the specificity of a selector chain.
    pub fn from_chain(chain: &SelectorChain) -> Self {
        let mut ids = 0u32;
        let mut classes = 0u32;
        let mut tags = 0u32;
        for (sel, _) in &chain.parts {
            if sel.id.is_some() {
                ids += 1;
            }
            classes += sel.classes.len() as u32;
            if sel.tag.is_some() {
                tags += 1;
            }
        }
        Specificity(ids, classes, tags)
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parse a comma-separated selector list into individual chains.
///
/// Example: `"h1, h2, h3"` -> 3 SelectorChains.
pub fn parse_selector_list(text: &str) -> Vec<SelectorChain> {
    text.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| parse_selector_chain(s))
        .collect()
}

/// Parse a single selector chain (no commas).
///
/// Example: `"div.container > h1.title"` -> chain with two parts.
pub fn parse_selector_chain(text: &str) -> SelectorChain {
    let text = text.trim();
    if text.is_empty() {
        return SelectorChain { parts: Vec::new() };
    }

    // Tokenize into simple selectors and combinators.
    // We walk the string and split on whitespace and `>`.
    let mut tokens: Vec<(String, Option<Combinator>)> = Vec::new();
    let mut current = String::new();
    let mut pending_combinator: Option<Combinator> = None;

    let mut chars = text.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            '>' => {
                chars.next();
                // Flush current token if any
                if !current.trim().is_empty() {
                    tokens.push((current.trim().to_string(), None));
                    current.clear();
                }
                pending_combinator = Some(Combinator::Child);
            }
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
                // Flush current token if any
                if !current.trim().is_empty() {
                    tokens.push((current.trim().to_string(), None));
                    current.clear();
                    // Only set descendant if we don't already have a child combinator pending
                    if pending_combinator.is_none() {
                        pending_combinator = Some(Combinator::Descendant);
                    }
                }
                // Skip additional whitespace
                while chars.peek().map_or(false, |c| c.is_whitespace()) {
                    chars.next();
                }
            }
            _ => {
                // If we have a pending combinator and are starting a new token,
                // record the combinator with the PREVIOUS token
                if !current.is_empty() || pending_combinator.is_none() {
                    current.push(c);
                } else {
                    // Starting a new token after combinator
                    current.push(c);
                }
                chars.next();
            }
        }
    }

    // Flush last token
    if !current.trim().is_empty() {
        tokens.push((current.trim().to_string(), None));
    }

    // Now build the chain. Tokens are left-to-right, we need right-to-left.
    // The combinators go BETWEEN tokens: token0 [comb] token1 [comb] token2
    // We need to associate each combinator with the right-hand selector when
    // stored right-to-left.
    //
    // Build intermediate: associate combinators between tokens.
    // tokens = [("div.container", None), ("h1.title", None)]
    // With pending_combinator tracking, we need to rebuild.

    // Actually, let me rebuild this more carefully.
    // Re-parse: split text on combinators to get ordered (selector, combinator) pairs.
    let parts_lr = parse_chain_parts(text);

    // Reverse for right-to-left matching order.
    // The rightmost selector (key selector) gets combinator = None.
    // Each selector to its left gets the combinator that connected it to its right neighbor.
    let mut parts: Vec<(SimpleSelector, Option<Combinator>)> = Vec::new();
    for (i, (sel_text, comb)) in parts_lr.iter().enumerate().rev() {
        let simple = parse_simple_selector(sel_text);
        if i == parts_lr.len() - 1 {
            // Rightmost: no combinator (key selector)
            parts.push((simple, None));
        } else {
            // This selector's combinator is what connects it to the next selector to the right.
            // In the left-to-right list, comb is the combinator AFTER this selector.
            parts.push((simple, comb.clone()));
        }
    }

    SelectorChain { parts }
}

/// Parse a chain into left-to-right (selector_text, combinator_after) pairs.
fn parse_chain_parts(text: &str) -> Vec<(String, Option<Combinator>)> {
    let mut result: Vec<(String, Option<Combinator>)> = Vec::new();
    let mut current = String::new();

    let mut chars = text.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            '>' => {
                chars.next();
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    result.push((trimmed, Some(Combinator::Child)));
                    current.clear();
                } else if let Some(last) = result.last_mut() {
                    // `>` after whitespace: upgrade descendant to child
                    last.1 = Some(Combinator::Child);
                }
                // Skip trailing whitespace after >
                while chars.peek().map_or(false, |c| c.is_whitespace()) {
                    chars.next();
                }
            }
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    // Tentatively mark as descendant; may be upgraded to child if > follows
                    result.push((trimmed, Some(Combinator::Descendant)));
                    current.clear();
                }
                // Skip additional whitespace
                while chars.peek().map_or(false, |c| c.is_whitespace()) {
                    chars.next();
                }
                // Check if next char is > (in which case the descendant should become child)
                if chars.peek() == Some(&'>') {
                    // Will be handled in next iteration
                }
            }
            _ => {
                current.push(c);
                chars.next();
            }
        }
    }

    // Last token has no combinator after it
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        result.push((trimmed, None));
    }

    // Fix: if the last entry has a combinator but no following selector, remove combinator
    if let Some(last) = result.last_mut() {
        if last.1.is_some() {
            // Check if this is actually the last: only remove if nothing follows
            // This shouldn't happen with well-formed input, but be safe
        }
    }

    result
}

/// Parse a simple selector string like `div.foo#bar` or `.class` or `#id` or `*`.
pub fn parse_simple_selector(text: &str) -> SimpleSelector {
    let text = text.trim();
    if text.is_empty() || text == "*" {
        return SimpleSelector {
            tag: None,
            id: None,
            classes: Vec::new(),
        };
    }

    let mut tag: Option<String> = None;
    let mut id: Option<String> = None;
    let mut classes: Vec<String> = Vec::new();

    // Split the selector on `.` and `#` boundaries while preserving delimiters.
    let mut current = String::new();
    let mut current_type: u8 = 0; // 0=tag, 1=class, 2=id

    for c in text.chars() {
        match c {
            '.' => {
                flush_part(&current, current_type, &mut tag, &mut id, &mut classes);
                current.clear();
                current_type = 1; // class
            }
            '#' => {
                flush_part(&current, current_type, &mut tag, &mut id, &mut classes);
                current.clear();
                current_type = 2; // id
            }
            _ => {
                current.push(c);
            }
        }
    }
    flush_part(&current, current_type, &mut tag, &mut id, &mut classes);

    SimpleSelector { tag, id, classes }
}

fn flush_part(
    text: &str,
    part_type: u8,
    tag: &mut Option<String>,
    id: &mut Option<String>,
    classes: &mut Vec<String>,
) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    match part_type {
        0 => *tag = Some(text.to_lowercase()),
        1 => classes.push(text.to_string()),
        2 => *id = Some(text.to_string()),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// DOM node helpers
// ---------------------------------------------------------------------------

/// Extract the lowercase tag name from a DOM node.
pub fn get_tag_name(node: &Handle) -> Option<String> {
    if let NodeData::Element { ref name, .. } = node.data {
        Some(name.local.as_ref().to_lowercase())
    } else {
        None
    }
}

/// Extract the class list from a DOM node's `class` attribute.
pub fn get_classes(node: &Handle) -> Vec<String> {
    if let NodeData::Element { ref attrs, .. } = node.data {
        let attrs = attrs.borrow();
        for attr in attrs.iter() {
            if attr.name.local.as_ref() == "class" {
                return attr
                    .value
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect();
            }
        }
    }
    Vec::new()
}

/// Extract the id from a DOM node's `id` attribute.
pub fn get_id(node: &Handle) -> Option<String> {
    if let NodeData::Element { ref attrs, .. } = node.data {
        let attrs = attrs.borrow();
        for attr in attrs.iter() {
            if attr.name.local.as_ref() == "id" {
                let val = attr.value.trim().to_string();
                if !val.is_empty() {
                    return Some(val);
                }
            }
        }
    }
    None
}

/// Check if a single SimpleSelector matches a single DOM node.
pub fn simple_matches(sel: &SimpleSelector, node: &Handle) -> bool {
    // Check tag
    if let Some(ref tag) = sel.tag {
        match get_tag_name(node) {
            Some(ref node_tag) if node_tag == tag => {}
            _ => return false,
        }
    }

    // Check id
    if let Some(ref id) = sel.id {
        match get_id(node) {
            Some(ref node_id) if node_id == id => {}
            _ => return false,
        }
    }

    // Check classes (all must match)
    if !sel.classes.is_empty() {
        let node_classes = get_classes(node);
        for cls in &sel.classes {
            if !node_classes.contains(cls) {
                return false;
            }
        }
    }

    // Must be an element node (not text, document, etc.)
    matches!(node.data, NodeData::Element { .. })
}

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// Check if a selector chain matches a DOM node, given its ancestor chain.
///
/// `ancestors` should be ordered from the immediate parent (index 0) to the
/// root (last index). This is the "bottom-up" matching algorithm.
pub fn matches_element(chain: &SelectorChain, node: &Handle, ancestors: &[Handle]) -> bool {
    if chain.parts.is_empty() {
        return false;
    }

    // First part is the key selector (rightmost) - must match the node itself
    let (ref key_sel, _) = chain.parts[0];
    if !simple_matches(key_sel, node) {
        return false;
    }

    // Walk remaining parts against ancestors
    let mut ancestor_idx = 0;
    for part_idx in 1..chain.parts.len() {
        let (ref sel, ref comb) = chain.parts[part_idx];
        match comb {
            Some(Combinator::Child) => {
                // Must match the immediate next ancestor
                if ancestor_idx >= ancestors.len() {
                    return false;
                }
                if !simple_matches(sel, &ancestors[ancestor_idx]) {
                    return false;
                }
                ancestor_idx += 1;
            }
            Some(Combinator::Descendant) | None => {
                // Must match some ancestor (any distance)
                let mut found = false;
                while ancestor_idx < ancestors.len() {
                    if simple_matches(sel, &ancestors[ancestor_idx]) {
                        ancestor_idx += 1;
                        found = true;
                        break;
                    }
                    ancestor_idx += 1;
                }
                if !found {
                    return false;
                }
            }
        }
    }

    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use markup5ever::{Attribute, LocalName, Namespace, QualName};
    use markup5ever_rcdom::Node;
    use std::cell::RefCell;

    /// Helper: create a DOM element node with the given tag and attributes.
    fn make_element(tag: &str, attrs: Vec<(&str, &str)>) -> Handle {
        let qual = QualName::new(
            None,
            Namespace::from("http://www.w3.org/1999/xhtml"),
            LocalName::from(tag),
        );
        let attributes: Vec<Attribute> = attrs
            .iter()
            .map(|(k, v)| Attribute {
                name: QualName::new(None, Namespace::from(""), LocalName::from(*k)),
                value: markup5ever::tendril::StrTendril::from(*v),
            })
            .collect();
        Node::new(NodeData::Element {
            name: qual,
            attrs: RefCell::new(attributes),
            template_contents: RefCell::new(None),
            mathml_annotation_xml_integration_point: false,
        })
    }

    // --- Parsing tests ---

    #[test]
    fn test_parse_tag_selector() {
        let chains = parse_selector_list("h1");
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].parts.len(), 1);
        assert_eq!(chains[0].parts[0].0.tag, Some("h1".to_string()));
        assert!(chains[0].parts[0].0.id.is_none());
        assert!(chains[0].parts[0].0.classes.is_empty());
    }

    #[test]
    fn test_parse_class_selector() {
        let chains = parse_selector_list(".highlight");
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].parts.len(), 1);
        assert!(chains[0].parts[0].0.tag.is_none());
        assert_eq!(chains[0].parts[0].0.classes, vec!["highlight"]);
    }

    #[test]
    fn test_parse_id_selector() {
        let chains = parse_selector_list("#main");
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].parts.len(), 1);
        assert_eq!(chains[0].parts[0].0.id, Some("main".to_string()));
    }

    #[test]
    fn test_parse_compound() {
        let chains = parse_selector_list("div.container");
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].parts.len(), 1);
        assert_eq!(chains[0].parts[0].0.tag, Some("div".to_string()));
        assert_eq!(chains[0].parts[0].0.classes, vec!["container"]);
    }

    #[test]
    fn test_parse_descendant() {
        let chains = parse_selector_list("div p");
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].parts.len(), 2);
        // Right-to-left order: p is first (key), div is second
        assert_eq!(chains[0].parts[0].0.tag, Some("p".to_string()));
        assert_eq!(chains[0].parts[0].1, None); // key selector
        assert_eq!(chains[0].parts[1].0.tag, Some("div".to_string()));
        assert_eq!(chains[0].parts[1].1, Some(Combinator::Descendant));
    }

    #[test]
    fn test_parse_child() {
        let chains = parse_selector_list("div > p");
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].parts.len(), 2);
        assert_eq!(chains[0].parts[0].0.tag, Some("p".to_string()));
        assert_eq!(chains[0].parts[0].1, None);
        assert_eq!(chains[0].parts[1].0.tag, Some("div".to_string()));
        assert_eq!(chains[0].parts[1].1, Some(Combinator::Child));
    }

    #[test]
    fn test_parse_multi_class() {
        let chains = parse_selector_list(".foo.bar");
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].parts.len(), 1);
        let sel = &chains[0].parts[0].0;
        assert!(sel.tag.is_none());
        assert!(sel.classes.contains(&"foo".to_string()));
        assert!(sel.classes.contains(&"bar".to_string()));
    }

    #[test]
    fn test_parse_complex() {
        let chains = parse_selector_list("div.container > p.intro");
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].parts.len(), 2);
        // Key selector: p.intro
        assert_eq!(chains[0].parts[0].0.tag, Some("p".to_string()));
        assert_eq!(chains[0].parts[0].0.classes, vec!["intro"]);
        assert_eq!(chains[0].parts[0].1, None);
        // Parent: div.container with child combinator
        assert_eq!(chains[0].parts[1].0.tag, Some("div".to_string()));
        assert_eq!(chains[0].parts[1].0.classes, vec!["container"]);
        assert_eq!(chains[0].parts[1].1, Some(Combinator::Child));
    }

    #[test]
    fn test_parse_comma_separated() {
        let chains = parse_selector_list("h1, h2, h3");
        assert_eq!(chains.len(), 3);
        assert_eq!(chains[0].parts[0].0.tag, Some("h1".to_string()));
        assert_eq!(chains[1].parts[0].0.tag, Some("h2".to_string()));
        assert_eq!(chains[2].parts[0].0.tag, Some("h3".to_string()));
    }

    // --- Specificity tests ---

    #[test]
    fn test_specificity_ordering() {
        // id > class > tag
        let id_sel = parse_selector_chain("#main");
        let class_sel = parse_selector_chain(".container");
        let tag_sel = parse_selector_chain("div");

        let id_spec = Specificity::from_chain(&id_sel);
        let class_spec = Specificity::from_chain(&class_sel);
        let tag_spec = Specificity::from_chain(&tag_sel);

        assert!(id_spec > class_spec);
        assert!(class_spec > tag_spec);
        assert!(id_spec > tag_spec);
    }

    #[test]
    fn test_specificity_calculation() {
        // h1 -> (0, 0, 1)
        let s = Specificity::from_chain(&parse_selector_chain("h1"));
        assert_eq!(s, Specificity(0, 0, 1));

        // .foo -> (0, 1, 0)
        let s = Specificity::from_chain(&parse_selector_chain(".foo"));
        assert_eq!(s, Specificity(0, 1, 0));

        // #bar -> (1, 0, 0)
        let s = Specificity::from_chain(&parse_selector_chain("#bar"));
        assert_eq!(s, Specificity(1, 0, 0));

        // div.foo#bar -> (1, 1, 1)
        let s = Specificity::from_chain(&parse_selector_chain("div.foo#bar"));
        assert_eq!(s, Specificity(1, 1, 1));

        // div.foo.bar -> (0, 2, 1)
        let s = Specificity::from_chain(&parse_selector_chain("div.foo.bar"));
        assert_eq!(s, Specificity(0, 2, 1));

        // div p -> (0, 0, 2)
        let s = Specificity::from_chain(&parse_selector_chain("div p"));
        assert_eq!(s, Specificity(0, 0, 2));

        // #main .content p -> (1, 1, 1)
        let s = Specificity::from_chain(&parse_selector_chain("#main .content p"));
        assert_eq!(s, Specificity(1, 1, 1));
    }

    // --- DOM helper tests ---

    #[test]
    fn test_get_tag_name() {
        let node = make_element("div", vec![]);
        assert_eq!(get_tag_name(&node), Some("div".to_string()));
    }

    #[test]
    fn test_get_classes() {
        let node = make_element("div", vec![("class", "foo bar baz")]);
        let classes = get_classes(&node);
        assert_eq!(classes, vec!["foo", "bar", "baz"]);
    }

    #[test]
    fn test_get_id() {
        let node = make_element("div", vec![("id", "main")]);
        assert_eq!(get_id(&node), Some("main".to_string()));
    }

    // --- Simple matching tests ---

    #[test]
    fn test_simple_matches_tag() {
        let sel = parse_simple_selector("div");
        let div = make_element("div", vec![]);
        let span = make_element("span", vec![]);
        assert!(simple_matches(&sel, &div));
        assert!(!simple_matches(&sel, &span));
    }

    #[test]
    fn test_simple_matches_class() {
        let sel = parse_simple_selector(".active");
        let yes = make_element("div", vec![("class", "active highlight")]);
        let no = make_element("div", vec![("class", "inactive")]);
        assert!(simple_matches(&sel, &yes));
        assert!(!simple_matches(&sel, &no));
    }

    #[test]
    fn test_simple_matches_id() {
        let sel = parse_simple_selector("#main");
        let yes = make_element("div", vec![("id", "main")]);
        let no = make_element("div", vec![("id", "sidebar")]);
        assert!(simple_matches(&sel, &yes));
        assert!(!simple_matches(&sel, &no));
    }

    #[test]
    fn test_simple_matches_compound() {
        let sel = parse_simple_selector("div.container#main");
        let yes = make_element("div", vec![("class", "container"), ("id", "main")]);
        let wrong_tag = make_element("span", vec![("class", "container"), ("id", "main")]);
        let wrong_class = make_element("div", vec![("class", "wrapper"), ("id", "main")]);
        assert!(simple_matches(&sel, &yes));
        assert!(!simple_matches(&sel, &wrong_tag));
        assert!(!simple_matches(&sel, &wrong_class));
    }

    // --- Chain matching tests ---

    #[test]
    fn test_matches_tag_only() {
        let chain = parse_selector_chain("p");
        let node = make_element("p", vec![]);
        assert!(matches_element(&chain, &node, &[]));

        let wrong = make_element("div", vec![]);
        assert!(!matches_element(&chain, &wrong, &[]));
    }

    #[test]
    fn test_matches_descendant() {
        let chain = parse_selector_chain("div p");
        let p = make_element("p", vec![]);
        let div = make_element("div", vec![]);
        let span = make_element("span", vec![]);

        // p inside div -> match
        assert!(matches_element(&chain, &p, &[div.clone()]));

        // p inside span -> no match
        assert!(!matches_element(&chain, &p, &[span.clone()]));

        // p inside span inside div -> match (descendant can skip)
        assert!(matches_element(&chain, &p, &[span.clone(), div.clone()]));
    }

    #[test]
    fn test_matches_child() {
        let chain = parse_selector_chain("div > p");
        let p = make_element("p", vec![]);
        let div = make_element("div", vec![]);
        let span = make_element("span", vec![]);

        // p directly inside div -> match
        assert!(matches_element(&chain, &p, &[div.clone()]));

        // p inside span inside div -> no match (child requires direct parent)
        assert!(!matches_element(&chain, &p, &[span.clone(), div.clone()]));
    }

    #[test]
    fn test_matches_complex_chain() {
        // body div.container > p
        let chain = parse_selector_chain("body div.container > p");
        let p = make_element("p", vec![]);
        let div_container = make_element("div", vec![("class", "container")]);
        let body = make_element("body", vec![]);
        let html = make_element("html", vec![]);

        // p -> div.container -> body -> html: should match
        assert!(matches_element(
            &chain,
            &p,
            &[div_container.clone(), body.clone(), html.clone()]
        ));

        // p -> div.container -> html (no body): should not match
        assert!(!matches_element(
            &chain,
            &p,
            &[div_container.clone(), html.clone()]
        ));
    }

    #[test]
    fn test_matches_no_ancestors_needed() {
        let chain = parse_selector_chain(".highlight");
        let node = make_element("span", vec![("class", "highlight")]);
        assert!(matches_element(&chain, &node, &[]));
    }

    #[test]
    fn test_matches_insufficient_ancestors() {
        let chain = parse_selector_chain("html body div p");
        let p = make_element("p", vec![]);
        let div = make_element("div", vec![]);
        // Only one ancestor, need three
        assert!(!matches_element(&chain, &p, &[div]));
    }

    #[test]
    fn test_universal_selector() {
        let sel = parse_simple_selector("*");
        let div = make_element("div", vec![]);
        assert!(simple_matches(&sel, &div));
    }
}
