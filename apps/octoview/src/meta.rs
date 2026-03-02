//! Metadata extraction — structured data from HTML pages.
//!
//! Extracts:
//!   - JSON-LD (schema.org) — structured data in <script type="application/ld+json">
//!   - Open Graph (og:*) — Facebook/social preview metadata
//!   - Twitter Cards — Twitter-specific preview metadata
//!   - Standard meta tags — description, keywords, author, etc.
//!   - Canonical URL, favicon, RSS/Atom feeds
//!
//! This is what LLM agents need most: structured, machine-readable page metadata
//! without parsing the visual DOM.

/// Structured metadata extracted from a page.
#[derive(Debug, Clone)]
pub struct PageMeta {
    pub title: String,
    pub description: String,
    pub canonical_url: String,
    pub favicon: String,
    pub language: String,
    pub author: String,
    pub keywords: Vec<String>,
    pub og: Vec<(String, String)>,       // Open Graph properties
    pub twitter: Vec<(String, String)>,   // Twitter Card properties
    pub json_ld: Vec<String>,             // Raw JSON-LD blocks
    pub feeds: Vec<(String, String)>,     // (type, url) — RSS/Atom feeds
    pub meta_tags: Vec<(String, String)>, // All other meta name/content pairs
}

impl PageMeta {
    pub fn new() -> Self {
        Self {
            title: String::new(),
            description: String::new(),
            canonical_url: String::new(),
            favicon: String::new(),
            language: String::new(),
            author: String::new(),
            keywords: Vec::new(),
            og: Vec::new(),
            twitter: Vec::new(),
            json_ld: Vec::new(),
            feeds: Vec::new(),
            meta_tags: Vec::new(),
        }
    }
}

/// Extract metadata from raw HTML.
pub fn extract_meta(html: &str) -> PageMeta {
    let mut meta = PageMeta::new();

    // Extract <title>
    if let Some(title) = extract_tag_content(html, "title") {
        meta.title = title.trim().to_string();
    }

    // Extract <html lang="...">
    if let Some(lang) = extract_attr_from_tag(html, "html", "lang") {
        meta.language = lang;
    }

    // Extract all <meta> tags
    extract_meta_tags(html, &mut meta);

    // Extract JSON-LD
    extract_json_ld(html, &mut meta);

    // Extract <link> tags (canonical, favicon, feeds)
    extract_link_tags(html, &mut meta);

    meta
}

/// Extract meta tags (name/property/content patterns).
fn extract_meta_tags(html: &str, meta: &mut PageMeta) {
    let lower = html.to_lowercase();
    let mut pos = 0;

    while let Some(start) = lower[pos..].find("<meta") {
        let abs_start = pos + start;
        let tag_end = match lower[abs_start..].find('>') {
            Some(e) => abs_start + e + 1,
            None => break,
        };

        let tag = &html[abs_start..tag_end];

        // Get name or property attribute
        let name = extract_attr_value(tag, "name")
            .or_else(|| extract_attr_value(tag, "property"))
            .unwrap_or_default()
            .to_lowercase();

        let content = extract_attr_value(tag, "content").unwrap_or_default();

        if !name.is_empty() && !content.is_empty() {
            // Open Graph
            if name.starts_with("og:") {
                meta.og.push((name[3..].to_string(), content.clone()));
                // Also fill convenience fields
                match name.as_str() {
                    "og:title" if meta.title.is_empty() => meta.title = content.clone(),
                    "og:description" if meta.description.is_empty() => meta.description = content.clone(),
                    "og:url" if meta.canonical_url.is_empty() => meta.canonical_url = content.clone(),
                    _ => {}
                }
            }
            // Twitter Cards
            else if name.starts_with("twitter:") {
                meta.twitter.push((name[8..].to_string(), content.clone()));
            }
            // Standard meta tags
            else {
                match name.as_str() {
                    "description" => meta.description = content.clone(),
                    "author" => meta.author = content.clone(),
                    "keywords" => {
                        meta.keywords = content
                            .split(',')
                            .map(|k| k.trim().to_string())
                            .filter(|k| !k.is_empty())
                            .collect();
                    }
                    _ => {}
                }
                meta.meta_tags.push((name, content));
            }
        }

        pos = tag_end;
    }
}

/// Extract JSON-LD blocks.
fn extract_json_ld(html: &str, meta: &mut PageMeta) {
    let lower = html.to_lowercase();
    let pattern = "application/ld+json";
    let mut pos = 0;

    while let Some(found) = lower[pos..].find(pattern) {
        let abs = pos + found;

        // Find the closing > of the script tag
        let script_end = match lower[abs..].find('>') {
            Some(e) => abs + e + 1,
            None => break,
        };

        // Find </script>
        let close = match lower[script_end..].find("</script>") {
            Some(c) => script_end + c,
            None => break,
        };

        let json = html[script_end..close].trim().to_string();
        if !json.is_empty() {
            meta.json_ld.push(json);
        }

        pos = close + 9;
    }
}

/// Extract <link> tags for canonical, favicon, feeds.
fn extract_link_tags(html: &str, meta: &mut PageMeta) {
    let lower = html.to_lowercase();
    let mut pos = 0;

    while let Some(start) = lower[pos..].find("<link") {
        let abs_start = pos + start;
        let tag_end = match lower[abs_start..].find('>') {
            Some(e) => abs_start + e + 1,
            None => break,
        };

        let tag = &html[abs_start..tag_end];
        let rel = extract_attr_value(tag, "rel").unwrap_or_default().to_lowercase();
        let href = extract_attr_value(tag, "href").unwrap_or_default();
        let link_type = extract_attr_value(tag, "type").unwrap_or_default().to_lowercase();

        if !href.is_empty() {
            match rel.as_str() {
                "canonical" => meta.canonical_url = href,
                "icon" | "shortcut icon" | "apple-touch-icon" => {
                    if meta.favicon.is_empty() {
                        meta.favicon = href;
                    }
                }
                "alternate" => {
                    if link_type.contains("rss") || link_type.contains("atom") || link_type.contains("xml") {
                        meta.feeds.push((link_type, href));
                    }
                }
                _ => {}
            }
        }

        pos = tag_end;
    }
}

/// Extract attribute value from a tag string.
fn extract_attr_value(tag: &str, attr_name: &str) -> Option<String> {
    let lower_tag = tag.to_lowercase();
    let patterns = [
        format!("{}=\"", attr_name),
        format!("{}='", attr_name),
    ];

    for pattern in &patterns {
        if let Some(start) = lower_tag.find(pattern.as_str()) {
            let val_start = start + pattern.len();
            let quote = if pattern.ends_with('"') { '"' } else { '\'' };
            if let Some(end) = tag[val_start..].find(quote) {
                return Some(tag[val_start..val_start + end].to_string());
            }
        }
    }
    None
}

/// Extract text content between opening and closing tags.
fn extract_tag_content(html: &str, tag: &str) -> Option<String> {
    let lower = html.to_lowercase();
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);

    if let Some(start) = lower.find(&open) {
        if let Some(gt) = lower[start..].find('>') {
            let content_start = start + gt + 1;
            if let Some(end) = lower[content_start..].find(&close) {
                return Some(html[content_start..content_start + end].to_string());
            }
        }
    }
    None
}

/// Extract attribute from a specific tag.
fn extract_attr_from_tag(html: &str, tag: &str, attr: &str) -> Option<String> {
    let lower = html.to_lowercase();
    let open = format!("<{}", tag);

    if let Some(start) = lower.find(&open) {
        if let Some(gt) = lower[start..].find('>') {
            let tag_str = &html[start..start + gt + 1];
            return extract_attr_value(tag_str, attr);
        }
    }
    None
}

/// Format metadata as human-readable text.
pub fn format_meta(meta: &PageMeta) -> String {
    let mut out = String::new();

    out.push_str("── Page Metadata ──\n");

    if !meta.title.is_empty() {
        out.push_str(&format!("  Title:       {}\n", meta.title));
    }
    if !meta.description.is_empty() {
        out.push_str(&format!("  Description: {}\n", truncate(&meta.description, 80)));
    }
    if !meta.canonical_url.is_empty() {
        out.push_str(&format!("  Canonical:   {}\n", meta.canonical_url));
    }
    if !meta.author.is_empty() {
        out.push_str(&format!("  Author:      {}\n", meta.author));
    }
    if !meta.language.is_empty() {
        out.push_str(&format!("  Language:    {}\n", meta.language));
    }
    if !meta.favicon.is_empty() {
        out.push_str(&format!("  Favicon:     {}\n", meta.favicon));
    }
    if !meta.keywords.is_empty() {
        out.push_str(&format!("  Keywords:    {}\n", meta.keywords.join(", ")));
    }

    // Open Graph
    if !meta.og.is_empty() {
        out.push_str("\n── Open Graph ──\n");
        for (key, val) in &meta.og {
            out.push_str(&format!("  og:{}: {}\n", key, truncate(val, 70)));
        }
    }

    // Twitter Cards
    if !meta.twitter.is_empty() {
        out.push_str("\n── Twitter Card ──\n");
        for (key, val) in &meta.twitter {
            out.push_str(&format!("  twitter:{}: {}\n", key, truncate(val, 70)));
        }
    }

    // JSON-LD
    if !meta.json_ld.is_empty() {
        out.push_str(&format!("\n── JSON-LD ({} blocks) ──\n", meta.json_ld.len()));
        for (i, json) in meta.json_ld.iter().enumerate() {
            out.push_str(&format!("  [{}] {}\n", i + 1, truncate(json, 200)));
        }
    }

    // Feeds
    if !meta.feeds.is_empty() {
        out.push_str("\n── Feeds ──\n");
        for (ft, url) in &meta.feeds {
            out.push_str(&format!("  {} → {}\n", ft, url));
        }
    }

    out
}

/// Format metadata as JSONL.
pub fn format_meta_jsonl(meta: &PageMeta) -> String {
    let mut out = String::new();

    // Core fields as one line
    out.push_str(&format!(
        "{{\"type\":\"meta\",\"title\":\"{}\",\"description\":\"{}\",\"url\":\"{}\",\"author\":\"{}\",\"lang\":\"{}\"}}\n",
        escape_json(&meta.title),
        escape_json(&meta.description),
        escape_json(&meta.canonical_url),
        escape_json(&meta.author),
        escape_json(&meta.language),
    ));

    // OG properties
    for (key, val) in &meta.og {
        out.push_str(&format!(
            "{{\"type\":\"og\",\"key\":\"{}\",\"value\":\"{}\"}}\n",
            escape_json(key),
            escape_json(val),
        ));
    }

    // Twitter properties
    for (key, val) in &meta.twitter {
        out.push_str(&format!(
            "{{\"type\":\"twitter\",\"key\":\"{}\",\"value\":\"{}\"}}\n",
            escape_json(key),
            escape_json(val),
        ));
    }

    // JSON-LD blocks
    for json in &meta.json_ld {
        out.push_str(&format!(
            "{{\"type\":\"json-ld\",\"data\":{}}}\n",
            json.trim(),
        ));
    }

    // Feeds
    for (ft, url) in &meta.feeds {
        out.push_str(&format!(
            "{{\"type\":\"feed\",\"format\":\"{}\",\"url\":\"{}\"}}\n",
            escape_json(ft),
            escape_json(url),
        ));
    }

    out
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", &s[..max.saturating_sub(3)])
    } else {
        s.to_string()
    }
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_meta() {
        let html = r#"
<html lang="en">
<head>
    <title>Test Page</title>
    <meta name="description" content="A test description">
    <meta name="author" content="Jane Doe">
    <meta name="keywords" content="rust, gpu, compiler">
    <link rel="canonical" href="https://example.com/test">
    <link rel="icon" href="/favicon.ico">
</head>
<body>Hello</body>
</html>
"#;
        let meta = extract_meta(html);
        assert_eq!(meta.title, "Test Page");
        assert_eq!(meta.description, "A test description");
        assert_eq!(meta.author, "Jane Doe");
        assert_eq!(meta.keywords, vec!["rust", "gpu", "compiler"]);
        assert_eq!(meta.canonical_url, "https://example.com/test");
        assert_eq!(meta.favicon, "/favicon.ico");
        assert_eq!(meta.language, "en");
    }

    #[test]
    fn test_open_graph() {
        let html = r#"
<head>
    <meta property="og:title" content="OG Title">
    <meta property="og:description" content="OG description here">
    <meta property="og:image" content="https://example.com/img.png">
    <meta property="og:url" content="https://example.com/page">
    <meta property="og:type" content="article">
</head>
"#;
        let meta = extract_meta(html);
        assert_eq!(meta.og.len(), 5);
        assert_eq!(meta.og[0], ("title".to_string(), "OG Title".to_string()));
        assert_eq!(meta.og[1], ("description".to_string(), "OG description here".to_string()));
    }

    #[test]
    fn test_twitter_cards() {
        let html = r#"
<head>
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:site" content="@example">
    <meta name="twitter:title" content="Twitter Title">
</head>
"#;
        let meta = extract_meta(html);
        assert_eq!(meta.twitter.len(), 3);
        assert_eq!(meta.twitter[0], ("card".to_string(), "summary_large_image".to_string()));
    }

    #[test]
    fn test_json_ld() {
        let html = r#"
<head>
    <script type="application/ld+json">
    {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": "Test Article"
    }
    </script>
</head>
"#;
        let meta = extract_meta(html);
        assert_eq!(meta.json_ld.len(), 1);
        assert!(meta.json_ld[0].contains("Article"));
    }

    #[test]
    fn test_rss_feed() {
        let html = r#"
<head>
    <link rel="alternate" type="application/rss+xml" href="/feed.xml">
    <link rel="alternate" type="application/atom+xml" href="/atom.xml">
</head>
"#;
        let meta = extract_meta(html);
        assert_eq!(meta.feeds.len(), 2);
        assert_eq!(meta.feeds[0].1, "/feed.xml");
        assert_eq!(meta.feeds[1].1, "/atom.xml");
    }

    #[test]
    fn test_format_meta() {
        let html = r#"
<html lang="en">
<head>
    <title>Format Test</title>
    <meta name="description" content="Testing format">
    <meta property="og:title" content="OG Test">
</head>
</html>
"#;
        let meta = extract_meta(html);
        let formatted = format_meta(&meta);
        assert!(formatted.contains("Title:       Format Test"));
        assert!(formatted.contains("og:title: OG Test"));
    }

    #[test]
    fn test_format_jsonl() {
        let html = r#"
<head>
    <title>JSONL Test</title>
    <meta name="description" content="Test &quot;quoted&quot;">
</head>
"#;
        let meta = extract_meta(html);
        let jsonl = format_meta_jsonl(&meta);
        assert!(jsonl.contains("\"title\":\"JSONL Test\""));
    }
}
