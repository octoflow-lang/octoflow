/// Flow Mode -- detect and render .flow files natively.
/// v2 renders source as syntax view; full OctoUI rendering is v3.

/// Check if a URL should trigger Flow Mode rendering.
pub fn is_flow_url(url: &str) -> bool {
    let url_lower = url.to_lowercase();
    url_lower.ends_with(".flow")
        || url_lower.starts_with("flow://")
}

/// Create an OVD document from .flow source code (syntax-highlighted view).
pub fn flow_to_ovd(source: &str, file_path: &str) -> crate::ovd::OvdDocument {
    use crate::ovd::{NodeType, OvdNode, OvdDocument};

    let mut doc = OvdDocument::new(file_path);
    let title = std::path::Path::new(file_path)
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| "Flow Program".to_string());
    doc.title = format!("{} - Flow Mode", title);

    let mut page = OvdNode::new(NodeType::Page);
    page.text = file_path.to_string();
    doc.add_node(page);

    let mut h1 = OvdNode::new(NodeType::Heading);
    h1.text = title;
    h1.level = 1;
    doc.add_node(h1);

    let mut info = OvdNode::new(NodeType::Paragraph);
    info.text = "Rendered in Flow Mode -- native .flow content".to_string();
    doc.add_node(info);

    doc.add_node(OvdNode::new(NodeType::Separator));

    // Show source code in a code block
    let mut code = OvdNode::new(NodeType::CodeBlock);
    code.text = source.to_string();
    doc.add_node(code);

    doc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ovd::NodeType;

    #[test]
    fn test_is_flow_url() {
        assert!(is_flow_url("test.flow"));
        assert!(is_flow_url("flow://app"));
        assert!(is_flow_url("/path/to/file.flow"));
        assert!(is_flow_url("C:\\path\\test.FLOW"));
        assert!(!is_flow_url("https://example.com"));
        assert!(!is_flow_url("file.html"));
    }

    #[test]
    fn test_flow_to_ovd() {
        let doc = flow_to_ovd("let x = 1.0\nprint(x)", "test.flow");
        assert!(doc.title.contains("Flow Mode"));
        assert!(doc.nodes.len() >= 4); // page, heading, para, separator, code
        // Check there's a code block
        let has_code = doc.nodes.iter().any(|n| n.node_type == NodeType::CodeBlock);
        assert!(has_code);
    }

    #[test]
    fn test_flow_to_ovd_extracts_filename() {
        let doc = flow_to_ovd("let y = 2.0", "/some/path/hello.flow");
        assert!(doc.title.contains("hello.flow"));
    }

    #[test]
    fn test_is_flow_url_flow_protocol() {
        assert!(is_flow_url("flow://my-app/main"));
        assert!(is_flow_url("flow://"));
    }

    #[test]
    fn test_is_flow_url_case_insensitive() {
        assert!(is_flow_url("TEST.FLOW"));
        assert!(is_flow_url("test.Flow"));
        assert!(is_flow_url("FLOW://app"));
    }
}
