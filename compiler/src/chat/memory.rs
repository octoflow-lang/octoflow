//! Memory system — persistent cross-session + ephemeral session state.
//!
//! Persistent memory lives at `~/.octoflow/memory.json` (max 32KB).
//! Session memory is in-memory only, discarded when the session ends.
//!
//! The memory summary (~300 tokens) is injected into every system prompt.

/// Maximum persistent memory file size (32 KB).
const MAX_MEMORY_SIZE: usize = 32 * 1024;

/// Maximum entries per field.
const MAX_FREQUENT_MODULES: usize = 50;
const MAX_KNOWN_CORRECTIONS: usize = 100;
const MAX_SUCCESSFUL_PATTERNS: usize = 30;

/// Persistent memory — saved to ~/.octoflow/memory.json.
#[derive(Clone)]
pub struct PersistentMemory {
    pub verbose: bool,
    pub naming_convention: String,
    pub preferred_patterns: Vec<String>,
    pub frequent_modules: Vec<ModuleUsage>,
    pub known_corrections: Vec<(String, String)>, // (wrong, fix)
    pub successful_patterns: Vec<PatternUsage>,
    pub total_sessions: u32,
}

#[derive(Clone)]
pub struct ModuleUsage {
    pub name: String,
    pub count: u32,
}

#[derive(Clone)]
pub struct PatternUsage {
    pub pattern: String,
    pub count: u32,
    pub modules: Vec<String>,
}

/// Session memory — in-memory only, tracks current conversation state.
pub struct SessionMemory {
    pub modules_used: Vec<String>,
    pub context_loaded: Vec<String>,
    pub corrections_made: u32,
    pub turn_count: u32,
}

impl PersistentMemory {
    /// Create empty persistent memory.
    pub fn new() -> Self {
        PersistentMemory {
            verbose: false,
            naming_convention: "snake_case".to_string(),
            preferred_patterns: Vec::new(),
            frequent_modules: Vec::new(),
            known_corrections: Vec::new(),
            successful_patterns: Vec::new(),
            total_sessions: 0,
        }
    }

    /// Load persistent memory from ~/.octoflow/memory.json.
    /// Returns default memory if file doesn't exist or is malformed.
    pub fn load() -> Self {
        let path = match memory_path() {
            Some(p) => p,
            None => return Self::new(),
        };

        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => return Self::new(),
        };

        Self::parse_json(&content)
    }

    /// Save persistent memory to ~/.octoflow/memory.json.
    pub fn save(&self) {
        let path = match memory_path() {
            Some(p) => p,
            None => return,
        };

        // Ensure directory exists
        if let Some(parent) = std::path::Path::new(&path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let json = self.to_json();

        // Enforce 32KB limit
        if json.len() > MAX_MEMORY_SIZE {
            // Evict entries and try again
            let mut trimmed = self.clone();
            trimmed.evict();
            let json = trimmed.to_json();
            let _ = std::fs::write(&path, json);
        } else {
            let _ = std::fs::write(&path, json);
        }
    }

    /// Record a successful module usage.
    pub fn record_module(&mut self, module: &str) {
        if let Some(entry) = self.frequent_modules.iter_mut().find(|m| m.name == module) {
            entry.count += 1;
        } else {
            self.frequent_modules.push(ModuleUsage {
                name: module.to_string(),
                count: 1,
            });
        }

        // Evict if over limit
        if self.frequent_modules.len() > MAX_FREQUENT_MODULES {
            self.frequent_modules.sort_by(|a, b| b.count.cmp(&a.count));
            self.frequent_modules.truncate(MAX_FREQUENT_MODULES);
        }
    }

    /// Record a known correction (wrong code → fix).
    pub fn record_correction(&mut self, wrong: &str, fix: &str) {
        // Check for existing
        if let Some(entry) = self.known_corrections.iter_mut().find(|(w, _)| w == wrong) {
            entry.1 = fix.to_string();
        } else {
            self.known_corrections.push((wrong.to_string(), fix.to_string()));
        }

        // Evict oldest if over limit
        if self.known_corrections.len() > MAX_KNOWN_CORRECTIONS {
            self.known_corrections.remove(0); // LRU: remove oldest
        }
    }

    /// Generate a ~300 token summary for context injection.
    pub fn summarize(&self) -> String {
        if self.is_empty() {
            return String::new();
        }

        let mut parts = Vec::new();

        // User preferences
        let verbose = if self.verbose { "yes" } else { "no" };
        parts.push(format!("User preferences: [verbose={}, naming={}]", verbose, self.naming_convention));

        // Top modules (top 5)
        if !self.frequent_modules.is_empty() {
            let top: Vec<String> = self.frequent_modules.iter()
                .take(5)
                .map(|m| format!("{} ({}x)", m.name, m.count))
                .collect();
            parts.push(format!("Top modules: {}", top.join(", ")));
        }

        // Known corrections (top 5)
        if !self.known_corrections.is_empty() {
            let fixes: Vec<String> = self.known_corrections.iter()
                .rev() // Most recent first
                .take(5)
                .map(|(w, f)| format!("{}→{}", w, f))
                .collect();
            parts.push(format!("Known fixes: {}", fixes.join(", ")));
        }

        // Successful patterns (top 3)
        if !self.successful_patterns.is_empty() {
            let patterns: Vec<String> = self.successful_patterns.iter()
                .take(3)
                .map(|p| p.pattern.clone())
                .collect();
            parts.push(format!("Recent patterns: {}", patterns.join(", ")));
        }

        if parts.is_empty() {
            return String::new();
        }

        format!("\n## Memory\n{}\n", parts.join("\n"))
    }

    fn is_empty(&self) -> bool {
        self.frequent_modules.is_empty()
            && self.known_corrections.is_empty()
            && self.successful_patterns.is_empty()
    }

    /// Evict entries to fit within size limits.
    fn evict(&mut self) {
        self.frequent_modules.sort_by(|a, b| b.count.cmp(&a.count));
        self.frequent_modules.truncate(MAX_FREQUENT_MODULES / 2);

        if self.known_corrections.len() > MAX_KNOWN_CORRECTIONS / 2 {
            let keep = MAX_KNOWN_CORRECTIONS / 2;
            let start = self.known_corrections.len() - keep;
            self.known_corrections = self.known_corrections[start..].to_vec();
        }

        self.successful_patterns.sort_by(|a, b| b.count.cmp(&a.count));
        self.successful_patterns.truncate(MAX_SUCCESSFUL_PATTERNS / 2);
    }

    /// Parse memory from JSON string. Minimal hand-written parser (zero deps).
    fn parse_json(content: &str) -> Self {
        let mut mem = Self::new();
        let trimmed = content.trim();

        // Extract simple fields
        if let Some(v) = json_get_bool(trimmed, "verbose") {
            mem.verbose = v;
        }
        if let Some(v) = json_get_string(trimmed, "naming_convention") {
            mem.naming_convention = v;
        }
        if let Some(v) = json_get_number(trimmed, "total_sessions") {
            mem.total_sessions = v as u32;
        }

        // Extract frequent_modules array
        if let Some(arr) = json_get_array(trimmed, "frequent_modules") {
            for item in arr {
                if let (Some(name), Some(count)) = (json_get_string(&item, "name"), json_get_number(&item, "count")) {
                    mem.frequent_modules.push(ModuleUsage {
                        name,
                        count: count as u32,
                    });
                }
            }
        }

        // Extract known_corrections object
        if let Some(obj_str) = json_get_object(trimmed, "known_corrections") {
            let pairs = json_object_entries(&obj_str);
            for (key, val_str) in pairs {
                if let Some(fix) = json_get_string(&val_str, "fix") {
                    mem.known_corrections.push((key, fix));
                }
            }
        }

        // Extract successful_patterns array
        if let Some(arr) = json_get_array(trimmed, "successful_patterns") {
            for item in arr {
                if let (Some(pattern), Some(count)) = (json_get_string(&item, "pattern"), json_get_number(&item, "count")) {
                    mem.successful_patterns.push(PatternUsage {
                        pattern,
                        count: count as u32,
                        modules: Vec::new(), // Simplified — modules not critical for summary
                    });
                }
            }
        }

        mem
    }

    /// Serialize to JSON.
    fn to_json(&self) -> String {
        let mut json = String::from("{\n");
        json.push_str("  \"version\": 1,\n");
        json.push_str(&format!("  \"user_style\": {{\n"));
        json.push_str(&format!("    \"verbose\": {},\n", self.verbose));
        json.push_str(&format!("    \"naming_convention\": \"{}\",\n", json_escape_str(&self.naming_convention)));

        json.push_str("    \"preferred_patterns\": [");
        let pats: Vec<String> = self.preferred_patterns.iter()
            .map(|p| format!("\"{}\"", json_escape_str(p)))
            .collect();
        json.push_str(&pats.join(", "));
        json.push_str("]\n  },\n");

        // frequent_modules
        json.push_str("  \"frequent_modules\": [\n");
        let mods: Vec<String> = self.frequent_modules.iter()
            .map(|m| format!("    {{ \"name\": \"{}\", \"count\": {} }}", json_escape_str(&m.name), m.count))
            .collect();
        json.push_str(&mods.join(",\n"));
        json.push_str("\n  ],\n");

        // known_corrections
        json.push_str("  \"known_corrections\": {\n");
        let corrs: Vec<String> = self.known_corrections.iter()
            .map(|(w, f)| format!("    \"{}\": {{ \"fix\": \"{}\" }}", json_escape_str(w), json_escape_str(f)))
            .collect();
        json.push_str(&corrs.join(",\n"));
        json.push_str("\n  },\n");

        // successful_patterns
        json.push_str("  \"successful_patterns\": [\n");
        let spats: Vec<String> = self.successful_patterns.iter()
            .map(|p| format!("    {{ \"pattern\": \"{}\", \"count\": {} }}", json_escape_str(&p.pattern), p.count))
            .collect();
        json.push_str(&spats.join(",\n"));
        json.push_str("\n  ],\n");

        json.push_str(&format!("  \"total_sessions\": {}\n", self.total_sessions));
        json.push_str("}\n");

        json
    }
}

impl SessionMemory {
    /// Create empty session memory.
    pub fn new() -> Self {
        SessionMemory {
            modules_used: Vec::new(),
            context_loaded: Vec::new(),
            corrections_made: 0,
            turn_count: 0,
        }
    }

    /// Check if a knowledge module is already loaded in this session.
    pub fn is_loaded(&self, module_key: &str) -> bool {
        self.context_loaded.iter().any(|m| m == module_key)
    }

    /// Mark a knowledge module as loaded.
    pub fn mark_loaded(&mut self, module_key: &str) {
        if !self.is_loaded(module_key) {
            self.context_loaded.push(module_key.to_string());
        }
    }

    /// Record module usage.
    pub fn record_module(&mut self, module: &str) {
        if !self.modules_used.contains(&module.to_string()) {
            self.modules_used.push(module.to_string());
        }
    }
}

/// Get the path to ~/.octoflow/memory.json.
fn memory_path() -> Option<String> {
    if let Ok(home) = std::env::var("USERPROFILE") {
        Some(format!("{}/.octoflow/memory.json", home))
    } else if let Ok(home) = std::env::var("HOME") {
        Some(format!("{}/.octoflow/memory.json", home))
    } else {
        None
    }
}

// --- Minimal JSON helpers (no external deps) ---

fn json_escape_str(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

fn json_get_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\"", key);
    let pos = json.find(&pattern)?;
    let after = &json[pos + pattern.len()..];
    // Skip whitespace and colon
    let after = after.trim_start();
    let after = after.strip_prefix(':')?;
    let after = after.trim_start();
    // Read quoted string
    let after = after.strip_prefix('"')?;
    let end = after.find('"')?;
    Some(after[..end].replace("\\\"", "\"").replace("\\\\", "\\").replace("\\n", "\n"))
}

fn json_get_bool(json: &str, key: &str) -> Option<bool> {
    let pattern = format!("\"{}\"", key);
    let pos = json.find(&pattern)?;
    let after = &json[pos + pattern.len()..];
    let after = after.trim_start();
    let after = after.strip_prefix(':')?;
    let after = after.trim_start();
    if after.starts_with("true") {
        Some(true)
    } else if after.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn json_get_number(json: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{}\"", key);
    let pos = json.find(&pattern)?;
    let after = &json[pos + pattern.len()..];
    let after = after.trim_start();
    let after = after.strip_prefix(':')?;
    let after = after.trim_start();
    // Read digits
    let end = after.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')?;
    after[..end].parse().ok()
}

fn json_get_array(json: &str, key: &str) -> Option<Vec<String>> {
    let pattern = format!("\"{}\"", key);
    let pos = json.find(&pattern)?;
    let after = &json[pos + pattern.len()..];
    let after = after.trim_start();
    let after = after.strip_prefix(':')?;
    let after = after.trim_start();
    let after = after.strip_prefix('[')?;

    // Find matching ]
    let mut depth = 1;
    let mut end = 0;
    let mut in_string = false;
    let bytes = after.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if in_string {
            if bytes[i] == b'\\' {
                if i + 1 < bytes.len() {
                    i += 2;
                } else {
                    break; // trailing backslash, malformed
                }
                continue;
            }
            if bytes[i] == b'"' {
                in_string = false;
            }
        } else {
            match bytes[i] {
                b'"' => in_string = true,
                b'[' => { depth += 1; if depth > 64 { return None; } }
                b']' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i;
                        break;
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }

    if depth != 0 {
        return None;
    }

    let arr_content = &after[..end];
    // Split by top-level commas between { }
    let items = split_json_array(arr_content);
    Some(items)
}

fn json_get_object(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\"", key);
    let pos = json.find(&pattern)?;
    let after = &json[pos + pattern.len()..];
    let after = after.trim_start();
    let after = after.strip_prefix(':')?;
    let after = after.trim_start();
    let after = after.strip_prefix('{')?;

    let mut depth = 1;
    let mut end = 0;
    let mut in_string = false;
    let bytes = after.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if in_string {
            if bytes[i] == b'\\' {
                i += 2;
                continue;
            }
            if bytes[i] == b'"' {
                in_string = false;
            }
        } else {
            match bytes[i] {
                b'"' => in_string = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i;
                        break;
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }

    if depth != 0 {
        return None;
    }

    Some(format!("{{{}}}", &after[..end]))
}

fn json_object_entries(json: &str) -> Vec<(String, String)> {
    let mut entries = Vec::new();
    let trimmed = json.trim();
    let inner = if trimmed.starts_with('{') && trimmed.ends_with('}') {
        &trimmed[1..trimmed.len() - 1]
    } else {
        trimmed
    };

    // Find "key": value pairs
    let mut pos = 0;
    let bytes = inner.as_bytes();
    while pos < bytes.len() {
        // Find key
        let key_start = match inner[pos..].find('"') {
            Some(p) => pos + p + 1,
            None => break,
        };
        let key_end = match inner[key_start..].find('"') {
            Some(p) => key_start + p,
            None => break,
        };
        let key = inner[key_start..key_end].to_string();

        // Find : then value
        pos = key_end + 1;
        let colon = match inner[pos..].find(':') {
            Some(p) => pos + p + 1,
            None => break,
        };

        // Find the value (could be string, number, object, array)
        let val_start = inner[colon..].find(|c: char| !c.is_whitespace())
            .map(|p| colon + p)
            .unwrap_or(colon);

        if val_start >= inner.len() {
            break;
        }

        let val_char = bytes[val_start];
        let val_end;

        if val_char == b'{' {
            // Object value — find matching }
            let mut depth = 1;
            let mut i = val_start + 1;
            while i < bytes.len() && depth > 0 {
                match bytes[i] {
                    b'{' => depth += 1,
                    b'}' => depth -= 1,
                    _ => {}
                }
                i += 1;
            }
            val_end = i;
        } else {
            // Simple value — find next comma or end
            val_end = inner[val_start..].find(',')
                .map(|p| val_start + p)
                .unwrap_or(inner.len());
        }

        let val = inner[val_start..val_end].trim().to_string();
        entries.push((key, val));
        pos = val_end + 1;
    }

    entries
}

fn split_json_array(content: &str) -> Vec<String> {
    let mut items = Vec::new();
    let mut depth = 0;
    let mut start = 0;
    let mut in_string = false;
    let bytes = content.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if in_string {
            if bytes[i] == b'\\' {
                i += 2;
                continue;
            }
            if bytes[i] == b'"' {
                in_string = false;
            }
        } else {
            match bytes[i] {
                b'"' => in_string = true,
                b'{' | b'[' => depth += 1,
                b'}' | b']' => depth -= 1,
                b',' if depth == 0 => {
                    let item = content[start..i].trim();
                    if !item.is_empty() {
                        items.push(item.to_string());
                    }
                    start = i + 1;
                }
                _ => {}
            }
        }
        i += 1;
    }

    let last = content[start..].trim();
    if !last.is_empty() {
        items.push(last.to_string());
    }

    items
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_memory_is_empty() {
        let mem = PersistentMemory::new();
        assert!(mem.is_empty());
        assert_eq!(mem.total_sessions, 0);
    }

    #[test]
    fn test_record_module() {
        let mut mem = PersistentMemory::new();
        mem.record_module("data/csv");
        mem.record_module("data/csv");
        mem.record_module("ml/preprocess");
        assert_eq!(mem.frequent_modules.len(), 2);
        assert_eq!(mem.frequent_modules[0].count, 2);
        assert_eq!(mem.frequent_modules[1].count, 1);
    }

    #[test]
    fn test_record_correction() {
        let mut mem = PersistentMemory::new();
        mem.record_correction("print(x)", "print(\"{x}\")");
        assert_eq!(mem.known_corrections.len(), 1);
        assert_eq!(mem.known_corrections[0].0, "print(x)");
    }

    #[test]
    fn test_summarize_empty() {
        let mem = PersistentMemory::new();
        assert!(mem.summarize().is_empty());
    }

    #[test]
    fn test_summarize_with_data() {
        let mut mem = PersistentMemory::new();
        mem.record_module("data/csv");
        mem.record_correction("true", "1.0");
        let summary = mem.summarize();
        assert!(summary.contains("## Memory"));
        assert!(summary.contains("data/csv"));
        assert!(summary.contains("true→1.0"));
    }

    #[test]
    fn test_json_roundtrip() {
        let mut mem = PersistentMemory::new();
        mem.verbose = true;
        mem.naming_convention = "camelCase".to_string();
        mem.total_sessions = 42;
        mem.record_module("data/csv");
        mem.record_module("data/csv");
        mem.record_correction("print(x)", "print(\"{x}\")");

        let json = mem.to_json();
        let loaded = PersistentMemory::parse_json(&json);

        assert_eq!(loaded.verbose, true);
        assert_eq!(loaded.naming_convention, "camelCase");
        assert_eq!(loaded.total_sessions, 42);
        assert_eq!(loaded.frequent_modules.len(), 1);
        assert_eq!(loaded.frequent_modules[0].count, 2);
        assert_eq!(loaded.known_corrections.len(), 1);
    }

    #[test]
    fn test_session_memory() {
        let mut session = SessionMemory::new();
        assert!(!session.is_loaded("L1-web"));
        session.mark_loaded("L1-web");
        assert!(session.is_loaded("L1-web"));
        // No duplicate loading
        session.mark_loaded("L1-web");
        assert_eq!(session.context_loaded.len(), 1);
    }

    #[test]
    fn test_module_eviction() {
        let mut mem = PersistentMemory::new();
        for i in 0..60 {
            mem.record_module(&format!("module_{}", i));
        }
        // Should be capped at MAX_FREQUENT_MODULES
        assert!(mem.frequent_modules.len() <= MAX_FREQUENT_MODULES);
    }

    #[test]
    fn test_to_json_format() {
        let mem = PersistentMemory::new();
        let json = mem.to_json();
        assert!(json.contains("\"version\": 1"));
        assert!(json.contains("\"total_sessions\": 0"));
    }

    #[test]
    fn test_json_get_array_trailing_backslash() {
        // A-05: trailing backslash should not panic
        let malformed = r#"{"modules":["test\"]"#;
        let result = json_get_array(malformed, "modules");
        // Should return None or empty, not panic
        assert!(result.is_none() || result.unwrap().is_empty());
    }

    #[test]
    fn test_json_get_array_deeply_nested() {
        // A-10: deeply nested JSON should not stack overflow
        let mut json = r#"{"arr":"#.to_string();
        for _ in 0..100 { json.push('['); }
        for _ in 0..100 { json.push(']'); }
        json.push('}');
        let result = json_get_array(&json, "arr");
        // Should return None (depth exceeded), not crash
        assert!(result.is_none());
    }
}
