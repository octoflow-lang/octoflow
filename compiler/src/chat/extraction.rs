//! Code extraction from LLM markdown output.
//!
//! Parses LLM responses for ```flow ... ``` or ``` ... ``` code blocks
//! and extracts the code content.

/// Extract code from markdown-formatted LLM output.
///
/// 1. If ````flow ... ```` blocks found, extract and concatenate them.
/// 2. If generic ```` ... ```` blocks found, extract and concatenate them.
/// 3. Otherwise, return the entire output as raw .flow code.
pub fn extract_code(output: &str) -> String {
    // Try ```flow blocks first
    let flow_blocks = extract_fenced_blocks(output, "flow");
    if !flow_blocks.is_empty() {
        return flow_blocks.join("\n\n");
    }

    // Try generic ``` blocks
    let generic_blocks = extract_fenced_blocks(output, "");
    if !generic_blocks.is_empty() {
        return generic_blocks.join("\n\n");
    }

    // Treat entire output as raw .flow code
    output.to_string()
}

/// Extract fenced code blocks with optional language tag.
fn extract_fenced_blocks(text: &str, lang: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut in_block = false;
    let mut current_block = String::new();
    let fence_start = if lang.is_empty() { "```" } else { &format!("```{}", lang) };

    for line in text.lines() {
        let trimmed = line.trim();
        if !in_block && trimmed.starts_with(fence_start) {
            in_block = true;
            current_block.clear();
            continue;
        }
        if in_block && trimmed == "```" {
            in_block = false;
            let code = current_block.trim().to_string();
            if !code.is_empty() {
                blocks.push(code);
            }
            continue;
        }
        if in_block {
            current_block.push_str(line);
            current_block.push('\n');
        }
    }

    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_flow_block() {
        let input = "Here's the code:\n```flow\nlet x = 42\nprint(x)\n```\nDone!";
        assert_eq!(extract_code(input), "let x = 42\nprint(x)");
    }

    #[test]
    fn test_extract_generic_block() {
        let input = "Try this:\n```\nlet y = 10\n```";
        assert_eq!(extract_code(input), "let y = 10");
    }

    #[test]
    fn test_no_blocks_returns_raw() {
        let input = "let z = 5\nprint(z)";
        assert_eq!(extract_code(input), "let z = 5\nprint(z)");
    }

    #[test]
    fn test_multiple_blocks() {
        let input = "First:\n```flow\nlet a = 1\n```\nSecond:\n```flow\nlet b = 2\n```";
        assert_eq!(extract_code(input), "let a = 1\n\nlet b = 2");
    }
}
