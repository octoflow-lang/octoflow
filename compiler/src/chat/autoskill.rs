//! Auto-skill engine — keyword-based domain detection for context-aware prompting.
//!
//! Scans user messages for domain-specific keywords and assembles the right
//! knowledge context (L1 overviews + L2 module signatures) within an 800-token budget.

use super::knowledge;

/// Maximum tokens of skill context (beyond L0 + memory).
const SKILL_TOKEN_BUDGET: usize = 800;

/// Minimum keyword hits to trigger L2 loading for a domain.
const L2_THRESHOLD: usize = 2;

/// Maximum domains before falling back to L1-only mode.
const MAX_FULL_DOMAINS: usize = 3;

/// Keyword → domain mapping. Keywords that map to multiple domains are listed
/// in each domain's entry. Based on Enhancement team's auto-skills.md spec.
static KEYWORD_MAP: &[(&str, &[&str])] = &[
    // Web
    ("scrape", &["web"]), ("http", &["web"]), ("server", &["web"]),
    ("api", &["web"]), ("fetch", &["web"]), ("url", &["web"]),
    ("cors", &["web"]), ("rest", &["web"]), ("webhook", &["web"]),
    ("endpoint", &["web"]), ("download", &["web"]), ("upload", &["web"]),
    ("request", &["web"]),
    // Data
    ("csv", &["data"]), ("json", &["data", "web"]), ("file", &["data"]),
    ("read", &["data"]), ("write", &["data"]), ("parse", &["data"]),
    ("load", &["data"]), ("save", &["data"]), ("import", &["data"]),
    ("export", &["data"]), ("validate", &["data"]),
    // ML
    ("train", &["ml"]), ("predict", &["ml"]), ("classify", &["ml"]),
    ("cluster", &["ml"]), ("regression", &["ml", "stats"]),
    ("neural", &["ml", "ai"]), ("kmeans", &["ml"]), ("knn", &["ml"]),
    ("accuracy", &["ml"]), ("preprocess", &["ml"]),
    ("split", &["ml"]),
    // Stats
    ("mean", &["stats"]), ("median", &["stats"]), ("stddev", &["stats"]),
    ("correlation", &["stats"]), ("distribution", &["stats"]),
    ("hypothesis", &["stats"]), ("sharpe", &["stats"]),
    ("ema", &["stats"]), ("sma", &["stats"]), ("rsi", &["stats"]),
    ("macd", &["stats"]), ("bollinger", &["stats"]),
    ("risk", &["stats"]), ("skewness", &["stats"]), ("percentile", &["stats"]),
    // GPU
    ("gpu", &["gpu", "loom"]), ("parallel", &["gpu"]),
    ("compute", &["gpu", "loom"]), ("matrix", &["gpu", "science"]),
    ("vector", &["gpu", "science"]), ("vulkan", &["gpu"]),
    ("shader", &["gpu"]), ("million", &["gpu"]), ("scale", &["gpu"]),
    // Science
    ("derivative", &["science"]), ("integral", &["science"]),
    ("physics", &["science"]), ("optimize", &["science"]),
    ("signal", &["science"]), ("fft", &["science"]),
    ("interpolate", &["science"]), ("euler", &["science"]),
    ("runge", &["science"]), ("newton", &["science"]),
    ("wave", &["science"]), ("spring", &["science"]),
    ("gravity", &["science"]),
    // GUI
    ("window", &["gui"]), ("gui", &["gui"]), ("button", &["gui"]),
    ("plot", &["gui"]), ("chart", &["gui"]), ("canvas", &["gui"]),
    ("widget", &["gui"]), ("slider", &["gui"]), ("checkbox", &["gui"]),
    ("graph", &["gui", "collections"]), ("visualize", &["gui"]),
    ("dashboard", &["gui"]),
    // Media
    ("image", &["media"]), ("bmp", &["media"]), ("gif", &["media"]),
    ("wav", &["media"]), ("audio", &["media"]), ("video", &["media"]),
    ("pixel", &["media"]), ("filter", &["media"]),
    ("encode", &["media"]), ("decode", &["media"]),
    ("photo", &["media"]), ("picture", &["media"]),
    ("animation", &["media"]),
    // Crypto
    ("hash", &["crypto"]), ("encrypt", &["crypto"]),
    ("uuid", &["crypto"]), ("base64", &["crypto"]),
    ("hex", &["crypto"]), ("sha", &["crypto"]),
    ("token", &["crypto"]), ("checksum", &["crypto"]),
    // DB
    ("database", &["db"]), ("sql", &["db"]), ("table", &["db"]),
    ("query", &["db"]), ("insert", &["db"]), ("select", &["db"]),
    ("join", &["db"]), ("schema", &["db"]), ("column", &["db"]),
    ("row", &["db"]),
    // DevOps
    ("config", &["devops"]), ("log", &["devops"]),
    ("process", &["devops"]), ("shell", &["devops"]),
    ("deploy", &["devops"]), ("env", &["devops", "sys"]),
    ("template", &["devops"]), ("walk", &["devops"]),
    ("glob", &["devops"]), ("ini", &["devops"]), ("path", &["devops"]),
    // Collections
    ("heap", &["collections"]), ("queue", &["collections"]),
    ("stack", &["collections"]), ("set", &["collections"]),
    ("tree", &["collections"]), ("edge", &["collections"]),
    ("node", &["collections"]), ("bfs", &["collections"]),
    ("priority", &["collections"]), ("fifo", &["collections"]),
    ("lifo", &["collections"]),
    // String
    ("regex", &["string"]), ("pattern", &["string"]),
    ("format", &["string"]), ("pad", &["string"]),
    ("replace", &["string"]), ("match", &["string"]),
    ("search", &["string"]), ("trim", &["string"]),
    ("upper", &["string"]), ("lower", &["string"]),
    ("word", &["string"]),
    // Sys
    ("args", &["sys"]), ("platform", &["sys"]),
    ("timer", &["sys"]), ("benchmark", &["sys"]),
    ("memory", &["sys"]), ("system", &["sys"]),
    ("os", &["sys"]), ("cli", &["sys"]), ("flag", &["sys"]),
    // Terminal
    ("terminal", &["terminal"]), ("sixel", &["terminal"]),
    ("kitty", &["terminal"]), ("render", &["terminal"]),
    ("ascii", &["terminal"]), ("halfblock", &["terminal"]),
    ("console", &["terminal"]), ("ansi", &["terminal"]),
    // AI
    ("llm", &["ai"]), ("inference", &["ai"]),
    ("tokenize", &["ai"]), ("generate", &["ai"]),
    ("embed", &["ai"]), ("transformer", &["ai"]),
    ("gguf", &["ai"]), ("chat", &["ai"]), ("prompt", &["ai"]),
    // Loom
    ("loom", &["loom"]), ("dispatch", &["loom"]),
    ("jit", &["loom"]), ("kernel", &["loom"]),
    ("boot", &["loom"]), ("pipeline", &["data", "loom"]),
];

/// Result of scanning a user message for domain keywords.
pub struct SkillScan {
    /// Domains detected with their hit counts, sorted by count descending.
    pub domains: Vec<(String, usize)>,
}

/// Scan a user message for domain keywords. Returns matched domains with hit counts.
pub fn scan_keywords(message: &str) -> SkillScan {
    let lower = message.to_lowercase();
    let words: Vec<&str> = lower.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|w| !w.is_empty())
        .collect();

    // Count keyword hits per domain
    let mut domain_hits: Vec<(&str, usize)> = Vec::new();

    for (keyword, domains) in KEYWORD_MAP.iter() {
        // Check if keyword appears as a whole word in the message
        if words.iter().any(|w| *w == *keyword) {
            for domain in *domains {
                if let Some(entry) = domain_hits.iter_mut().find(|(d, _)| d == domain) {
                    entry.1 += 1;
                } else {
                    domain_hits.push((domain, 1));
                }
            }
        }
    }

    // Sort by hit count descending
    domain_hits.sort_by(|a, b| b.1.cmp(&a.1));

    SkillScan {
        domains: domain_hits.iter().map(|(d, c)| (d.to_string(), *c)).collect(),
    }
}

/// Assemble skill context from detected domains. Returns the text to append
/// to the system prompt after L0. Stays within SKILL_TOKEN_BUDGET.
pub fn assemble_skill_context(scan: &SkillScan) -> String {
    if scan.domains.is_empty() {
        return String::new();
    }

    let mut parts: Vec<String> = Vec::new();
    let mut token_estimate: usize = 0;

    let overflow = scan.domains.len() > MAX_FULL_DOMAINS;

    for (domain, hits) in &scan.domains {
        // Load L1 overview (always, if budget allows)
        if let Some(l1) = knowledge::get_l1(domain) {
            let l1_tokens = estimate_tokens(l1);
            if token_estimate + l1_tokens > SKILL_TOKEN_BUDGET {
                break;
            }
            parts.push(l1.to_string());
            token_estimate += l1_tokens;
        }

        // Load L2 modules only if:
        // 1. Not in overflow mode (>3 domains)
        // 2. Domain has >= L2_THRESHOLD keyword hits
        if !overflow && *hits >= L2_THRESHOLD {
            let l2_modules = knowledge::get_l2_for_domain(domain);
            // Take top 2 L2 modules per domain
            for (_, content) in l2_modules.iter().take(2) {
                let l2_tokens = estimate_tokens(content);
                if token_estimate + l2_tokens > SKILL_TOKEN_BUDGET {
                    break;
                }
                parts.push(content.to_string());
                token_estimate += l2_tokens;
            }
        }
    }

    if parts.is_empty() {
        return String::new();
    }

    format!("\n## Relevant Modules\n{}", parts.join("\n"))
}

/// Rough token estimate: ~4 chars per token (conservative).
fn estimate_tokens(text: &str) -> usize {
    (text.len() + 3) / 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_web_keywords() {
        let scan = scan_keywords("build an http server with api endpoints");
        assert!(!scan.domains.is_empty());
        let domains: Vec<&str> = scan.domains.iter().map(|(d, _)| d.as_str()).collect();
        assert!(domains.contains(&"web"));
    }

    #[test]
    fn test_scan_ml_keywords() {
        let scan = scan_keywords("train a model to classify images and predict labels");
        let domains: Vec<&str> = scan.domains.iter().map(|(d, _)| d.as_str()).collect();
        assert!(domains.contains(&"ml"));
    }

    #[test]
    fn test_scan_no_keywords() {
        let scan = scan_keywords("hello world");
        // "hello" and "world" are not keywords
        assert!(scan.domains.is_empty());
    }

    #[test]
    fn test_scan_multi_domain() {
        let scan = scan_keywords("read csv data and compute mean and stddev");
        let domains: Vec<&str> = scan.domains.iter().map(|(d, _)| d.as_str()).collect();
        assert!(domains.contains(&"data"));
        assert!(domains.contains(&"stats"));
    }

    #[test]
    fn test_assemble_empty() {
        let scan = SkillScan { domains: vec![] };
        assert!(assemble_skill_context(&scan).is_empty());
    }

    #[test]
    fn test_assemble_single_domain() {
        let scan = scan_keywords("http server api");
        let ctx = assemble_skill_context(&scan);
        assert!(!ctx.is_empty());
        assert!(ctx.contains("Relevant Modules"));
    }

    #[test]
    fn test_assemble_respects_budget() {
        // Even with many domains, should not exceed budget
        let scan = scan_keywords(
            "http server csv json train predict gpu parallel \
             image bmp hash encrypt sql database regex format \
             terminal ansi timer args"
        );
        let ctx = assemble_skill_context(&scan);
        let tokens = estimate_tokens(&ctx);
        assert!(tokens <= SKILL_TOKEN_BUDGET + 50); // small margin for header
    }

    #[test]
    fn test_l2_threshold() {
        // Single keyword hit → L1 only, no L2
        let scan = scan_keywords("train something");
        let ctx = assemble_skill_context(&scan);
        // With only 1 hit for "ml", should not include L2 details
        let ml_hits = scan.domains.iter().find(|(d, _)| d == "ml").map(|(_, c)| *c).unwrap_or(0);
        if ml_hits < L2_THRESHOLD {
            // L2 content should NOT be present (just L1 overview)
            // We can't easily distinguish L1 from L2 in the output, but
            // the budget should be small
            assert!(estimate_tokens(&ctx) < 200);
        }
    }

    #[test]
    fn test_scan_case_insensitive() {
        let scan = scan_keywords("HTTP Server API");
        let domains: Vec<&str> = scan.domains.iter().map(|(d, _)| d.as_str()).collect();
        assert!(domains.contains(&"web"));
    }

    #[test]
    fn test_overflow_mode() {
        // >3 domains should trigger L1-only mode
        let scan = scan_keywords(
            "http csv train gpu image hash regex terminal"
        );
        assert!(scan.domains.len() > 3);
        let ctx = assemble_skill_context(&scan);
        // In overflow mode, only L1 files loaded — should be compact
        assert!(estimate_tokens(&ctx) <= SKILL_TOKEN_BUDGET);
    }
}
