//! User config loader — OCTOFLOW.md + ~/.octoflow/ preferences and memory.
//!
//! Two scopes, mirroring Claude Code's CLAUDE.md + .claude/:
//!
//! **User-level** (`~/.octoflow/`):
//! - `preferences.md` — user writes, style/defaults/preferences
//! - `memory.md` — LLM auto-updates, discovered chains/corrections/patterns
//!
//! **Project-level** (cwd):
//! - `OCTOFLOW.md` — user writes, project instructions
//! - `.octoflow/memory.md` — LLM auto-updates, modules used/patterns that worked

/// Maximum tokens for user config in the prompt (~4 chars/token).
#[allow(dead_code)]
const CONFIG_TOKEN_BUDGET: usize = 300;
/// Maximum tokens per config section.
const SECTION_TOKEN_BUDGET: usize = 150;
/// Maximum memory entries before eviction.
const MAX_MEMORY_ENTRIES: usize = 50;

/// Loaded user config from all four sources.
pub struct UserConfig {
    /// ~/.octoflow/preferences.md contents (user-written global prefs).
    pub global_prefs: Option<String>,
    /// ~/.octoflow/memory.md contents (LLM auto-updated global memory).
    pub global_memory: Option<String>,
    /// ./OCTOFLOW.md contents (user-written project instructions).
    pub project_instructions: Option<String>,
    /// ./.octoflow/memory.md contents (LLM auto-updated project memory).
    pub project_memory: Option<String>,
}

impl UserConfig {
    /// Load config from all four sources. Returns None for missing files.
    pub fn load() -> Self {
        UserConfig {
            global_prefs: read_optional(&global_prefs_path()),
            global_memory: read_optional(&global_memory_path()),
            project_instructions: read_optional(&project_instructions_path()),
            project_memory: read_optional(&project_memory_path()),
        }
    }

    /// Build context string for prompt injection, capped at ~300 tokens.
    pub fn to_context(&self) -> String {
        let mut parts = Vec::new();

        // Global preferences (user-written)
        if let Some(ref prefs) = self.global_prefs {
            let truncated = truncate_to_tokens(prefs.trim(), SECTION_TOKEN_BUDGET);
            if !truncated.is_empty() {
                parts.push(format!("## User Preferences\n{}", truncated));
            }
        }

        // Project instructions (user-written)
        if let Some(ref instructions) = self.project_instructions {
            let truncated = truncate_to_tokens(instructions.trim(), SECTION_TOKEN_BUDGET);
            if !truncated.is_empty() {
                parts.push(format!("## Project Instructions\n{}", truncated));
            }
        }

        if parts.is_empty() {
            return String::new();
        }

        format!("\n{}\n", parts.join("\n\n"))
    }

    /// Check if any config was loaded.
    pub fn has_any(&self) -> bool {
        self.global_prefs.is_some()
            || self.global_memory.is_some()
            || self.project_instructions.is_some()
            || self.project_memory.is_some()
    }
}

/// Auto-save entry to append to memory.md files on session end.
pub struct MemoryEntry {
    pub content: String,
}

/// Append an entry to ~/.octoflow/memory.md.
pub fn save_global_memory(entry: &str) {
    let path = global_memory_path();
    append_memory_entry(&path, entry);
}

/// Append an entry to ./.octoflow/memory.md.
pub fn save_project_memory(entry: &str) {
    let path = project_memory_path();
    append_memory_entry(&path, entry);
}

/// Build a memory entry from session data.
pub fn build_memory_entry(
    modules_used: &[String],
    corrections: u32,
    turns: u32,
) -> Option<String> {
    if modules_used.is_empty() && corrections == 0 {
        return None;
    }

    let date = current_date();
    let mut lines = Vec::new();
    lines.push(format!("## Session {}", date));

    if !modules_used.is_empty() {
        let mods: Vec<&str> = modules_used.iter().take(10).map(|s| s.as_str()).collect();
        lines.push(format!("- Modules used: {}", mods.join(", ")));
    }

    if corrections > 0 {
        lines.push(format!("- Corrections: {}", corrections));
    }

    lines.push(format!("- Turns: {}", turns));

    Some(lines.join("\n"))
}

// --- Path helpers ---

fn global_dir() -> Option<String> {
    if let Ok(home) = std::env::var("USERPROFILE") {
        Some(format!("{}/.octoflow", home))
    } else if let Ok(home) = std::env::var("HOME") {
        Some(format!("{}/.octoflow", home))
    } else {
        None
    }
}

fn global_prefs_path() -> Option<String> {
    global_dir().map(|d| format!("{}/preferences.md", d))
}

fn global_memory_path() -> Option<String> {
    global_dir().map(|d| format!("{}/memory.md", d))
}

fn project_instructions_path() -> Option<String> {
    // Look for OCTOFLOW.md in cwd
    let path = "OCTOFLOW.md".to_string();
    if std::path::Path::new(&path).exists() {
        Some(path)
    } else {
        None
    }
}

fn project_memory_path() -> Option<String> {
    Some(".octoflow/memory.md".to_string())
}

// --- File I/O helpers ---

fn read_optional(path: &Option<String>) -> Option<String> {
    let path = path.as_ref()?;
    match std::fs::read_to_string(path) {
        Ok(content) if !content.trim().is_empty() => Some(content),
        _ => None,
    }
}

fn append_memory_entry(path: &Option<String>, entry: &str) {
    let path = match path {
        Some(p) => p,
        None => return,
    };

    // Create parent directory if needed (only when we have data to write)
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    // Read existing content
    let existing = std::fs::read_to_string(path).unwrap_or_default();

    // Count existing entries (## headers)
    let entry_count = existing.matches("\n## ").count() + if existing.starts_with("## ") { 1 } else { 0 };

    // Evict oldest entries if over limit
    let content = if entry_count >= MAX_MEMORY_ENTRIES {
        // Keep newest entries, drop oldest
        let entries: Vec<&str> = existing.split("\n## ").collect();
        if entries.len() > 1 {
            // First element is preamble (before first ##), skip it
            let keep_start = entries.len().saturating_sub(MAX_MEMORY_ENTRIES / 2);
            let mut result = String::new();
            for (i, e) in entries.iter().enumerate() {
                if i == 0 || i >= keep_start {
                    if i > 0 {
                        result.push_str("\n## ");
                    }
                    result.push_str(e);
                }
            }
            result
        } else {
            existing
        }
    } else {
        existing
    };

    // Append new entry
    let separator = if content.is_empty() || content.ends_with('\n') { "" } else { "\n" };
    let new_content = format!("{}{}{}\n", content, separator, entry);

    let _ = std::fs::write(path, new_content);
}

/// Truncate text to approximately max_tokens (~4 chars/token).
fn truncate_to_tokens(text: &str, max_tokens: usize) -> String {
    let max_chars = max_tokens * 4;
    if text.len() <= max_chars {
        return text.to_string();
    }
    // Find the last newline before max_chars for clean truncation
    let slice = &text[..max_chars];
    if let Some(pos) = slice.rfind('\n') {
        format!("{}...", &text[..pos])
    } else {
        format!("{}...", slice)
    }
}

/// Get current date as YYYY-MM-DD.
fn current_date() -> String {
    // Use std::time to get a rough date (no chrono dep)
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Approximate date calculation (good enough for memory entries)
    let days = secs / 86400;
    let years = 1970 + (days * 400) / 146097;
    // Simplified — just use year and day-of-year for a reasonable format
    let day_of_year = days - ((years - 1970) * 365 + (years - 1969) / 4 - (years - 1901) / 100 + (years - 1601) / 400);
    let month = (day_of_year * 12 / 366) + 1;
    let day = day_of_year - (month - 1) * 30 + 1;
    format!("{}-{:02}-{:02}", years, month.min(12), day.min(31).max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_no_files() {
        // In test environment, config files likely don't exist
        let cfg = UserConfig {
            global_prefs: None,
            global_memory: None,
            project_instructions: None,
            project_memory: None,
        };
        assert!(!cfg.has_any());
        assert!(cfg.to_context().is_empty());
    }

    #[test]
    fn test_to_context_with_prefs() {
        let cfg = UserConfig {
            global_prefs: Some("Always use GPU when available.\nPrefer concise output.".into()),
            global_memory: None,
            project_instructions: None,
            project_memory: None,
        };
        assert!(cfg.has_any());
        let ctx = cfg.to_context();
        assert!(ctx.contains("## User Preferences"));
        assert!(ctx.contains("GPU"));
    }

    #[test]
    fn test_to_context_with_project() {
        let cfg = UserConfig {
            global_prefs: None,
            global_memory: None,
            project_instructions: Some("This project analyzes stock data.\nCSVs are in ./data/.".into()),
            project_memory: None,
        };
        let ctx = cfg.to_context();
        assert!(ctx.contains("## Project Instructions"));
        assert!(ctx.contains("stock data"));
    }

    #[test]
    fn test_to_context_both() {
        let cfg = UserConfig {
            global_prefs: Some("Use snake_case.".into()),
            global_memory: None,
            project_instructions: Some("Web scraping project.".into()),
            project_memory: None,
        };
        let ctx = cfg.to_context();
        assert!(ctx.contains("## User Preferences"));
        assert!(ctx.contains("## Project Instructions"));
        assert!(ctx.contains("snake_case"));
        assert!(ctx.contains("Web scraping"));
    }

    #[test]
    fn test_truncate_to_tokens() {
        let short = "hello world";
        assert_eq!(truncate_to_tokens(short, 100), short);

        let long = "a".repeat(1000);
        let truncated = truncate_to_tokens(&long, 50);
        assert!(truncated.len() < 1000);
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_build_memory_entry_empty() {
        let entry = build_memory_entry(&[], 0, 0);
        assert!(entry.is_none());
    }

    #[test]
    fn test_build_memory_entry_with_data() {
        let modules = vec!["data/csv".to_string(), "stats/timeseries".to_string()];
        let entry = build_memory_entry(&modules, 2, 5);
        assert!(entry.is_some());
        let text = entry.unwrap();
        assert!(text.contains("## Session"));
        assert!(text.contains("data/csv"));
        assert!(text.contains("Corrections: 2"));
        assert!(text.contains("Turns: 5"));
    }

    #[test]
    fn test_to_context_truncation() {
        // Create a very long preferences string
        let long_prefs = "Use GPU.\n".repeat(500); // ~4500 chars
        let cfg = UserConfig {
            global_prefs: Some(long_prefs),
            global_memory: None,
            project_instructions: None,
            project_memory: None,
        };
        let ctx = cfg.to_context();
        // Should be truncated to ~150 tokens * 4 chars = ~600 chars
        assert!(ctx.len() < 800);
    }

    #[test]
    fn test_current_date_format() {
        let date = current_date();
        assert!(date.starts_with("20")); // year 20xx
        assert_eq!(date.len(), 10); // YYYY-MM-DD
        assert_eq!(date.as_bytes()[4], b'-');
        assert_eq!(date.as_bytes()[7], b'-');
    }

    #[test]
    fn test_global_dir_exists() {
        // On this system, USERPROFILE or HOME should be set
        let dir = global_dir();
        assert!(dir.is_some());
        assert!(dir.unwrap().contains(".octoflow"));
    }
}
