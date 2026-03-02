//! Project context scanning for chat prompt building.
//!
//! Scans the current directory for .flow files and extracts headers,
//! imports, and function signatures to inject into the LLM system prompt.

use std::path::Path;

/// Maximum tokens of project context to inject into the prompt.
const MAX_CONTEXT_LINES: usize = 50;

/// Scan the current directory for .flow files and build a project context string.
/// Reads the first 20 lines of each file (headers, imports, function signatures).
pub fn scan_project_context(dir: &str) -> String {
    let path = Path::new(dir);
    if !path.is_dir() {
        return String::new();
    }

    let mut context = String::from("Existing project files:\n");
    let mut total_lines = 0;

    let entries: Vec<_> = match std::fs::read_dir(path) {
        Ok(rd) => rd.filter_map(|e| e.ok()).collect(),
        Err(_) => return String::new(),
    };

    for entry in &entries {
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) != Some("flow") {
            continue;
        }
        if total_lines >= MAX_CONTEXT_LINES {
            break;
        }
        let filename = p.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        // Skip our own temp files
        if filename.starts_with(".octoflow_chat") {
            continue;
        }
        if let Ok(content) = std::fs::read_to_string(&p) {
            context.push_str(&format!("\n--- {} ---\n", filename));
            for line in content.lines().take(20) {
                context.push_str(line);
                context.push('\n');
                total_lines += 1;
                if total_lines >= MAX_CONTEXT_LINES {
                    break;
                }
            }
        }
    }

    if total_lines == 0 {
        String::new()
    } else {
        context
    }
}
