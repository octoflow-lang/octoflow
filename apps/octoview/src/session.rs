//! Browser Session — persistent Chrome session with profile, cookies, interaction.
//!
//! Solves the fundamental problem: every call to fetch_with_js() launches a new
//! Chrome with a temp profile, so cookies/state are lost. BrowserSession keeps
//! Chrome alive with a persistent user_data_dir.
//!
//! Features:
//!   - Persistent Chrome profiles (~/.octoview/profiles/<name>/)
//!   - Cookie export/import (storageState equivalent)
//!   - Form interaction: type, click, press_key
//!   - Screenshot capture
//!   - Multi-page navigation within one session
//!   - Interactive login flow (headed Chrome for manual login)

use headless_chrome::{Browser, LaunchOptions};
use headless_chrome::protocol::cdp::Network;
use headless_chrome::protocol::cdp::Page;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crate::extract;
use crate::js_engine::STEALTH_JS;
use crate::ovd::OvdDocument;

// ─── Profile Management ────────────────────────────────────────────────────

/// Get the profiles directory.
fn profiles_dir() -> PathBuf {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".octoview").join("profiles")
}

/// Get the path for a named profile.
fn profile_path(name: &str) -> PathBuf {
    profiles_dir().join(name)
}

/// List available profiles.
pub fn list_profiles() -> Vec<String> {
    let dir = profiles_dir();
    if !dir.exists() {
        return Vec::new();
    }
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                .filter_map(|e| e.file_name().into_string().ok())
                .collect()
        })
        .unwrap_or_default()
}

// ─── Cookie Serialization ──────────────────────────────────────────────────

/// A serializable cookie (simplified from Chrome's Cookie type).
#[derive(Debug, Clone)]
pub struct CookieData {
    pub name: String,
    pub value: String,
    pub domain: String,
    pub path: String,
    pub secure: bool,
    pub http_only: bool,
    pub expires: f64,
}

/// Export cookies from the current tab to a JSON file.
pub fn export_cookies(session: &BrowserSession, path: &str) -> Result<usize, String> {
    let cookies = session.get_cookies()?;
    let json = cookies_to_json(&cookies);
    std::fs::write(path, &json).map_err(|e| format!("Write error: {}", e))?;
    Ok(cookies.len())
}

/// Import cookies from a JSON file into the current tab.
pub fn import_cookies(session: &BrowserSession, path: &str) -> Result<usize, String> {
    let json = std::fs::read_to_string(path).map_err(|e| format!("Read error: {}", e))?;
    let cookies = json_to_cookies(&json)?;
    let count = cookies.len();
    session.set_cookies(cookies)?;
    Ok(count)
}

fn cookies_to_json(cookies: &[CookieData]) -> String {
    let mut out = String::from("[\n");
    for (i, c) in cookies.iter().enumerate() {
        out.push_str("  {\n");
        out.push_str(&format!("    \"name\": \"{}\",\n", json_esc(&c.name)));
        out.push_str(&format!("    \"value\": \"{}\",\n", json_esc(&c.value)));
        out.push_str(&format!("    \"domain\": \"{}\",\n", json_esc(&c.domain)));
        out.push_str(&format!("    \"path\": \"{}\",\n", json_esc(&c.path)));
        out.push_str(&format!("    \"secure\": {},\n", c.secure));
        out.push_str(&format!("    \"httpOnly\": {},\n", c.http_only));
        out.push_str(&format!("    \"expires\": {:.0}\n", c.expires));
        out.push_str("  }");
        if i < cookies.len() - 1 {
            out.push(',');
        }
        out.push('\n');
    }
    out.push(']');
    out
}

fn json_to_cookies(json: &str) -> Result<Vec<CookieData>, String> {
    // Simple JSON array parser for cookie objects
    let json = json.trim();
    if !json.starts_with('[') {
        return Err("Expected JSON array".to_string());
    }

    let mut cookies = Vec::new();
    let inner = &json[1..json.len().saturating_sub(1)];
    let objects = split_json_objects(inner);

    for obj in &objects {
        let name = extract_json_str(obj, "name").unwrap_or_default();
        let value = extract_json_str(obj, "value").unwrap_or_default();
        let domain = extract_json_str(obj, "domain").unwrap_or_default();
        let path = extract_json_str(obj, "path").unwrap_or_else(|| "/".to_string());
        let secure = extract_json_bool(obj, "secure").unwrap_or(false);
        let http_only = extract_json_bool(obj, "httpOnly").unwrap_or(false);
        let expires = extract_json_num(obj, "expires").unwrap_or(0.0);

        if !name.is_empty() {
            cookies.push(CookieData {
                name,
                value,
                domain,
                path,
                secure,
                http_only,
                expires,
            });
        }
    }

    Ok(cookies)
}

// ─── Browser Session ───────────────────────────────────────────────────────

/// A persistent browser session with Chrome profile, cookies, and interaction.
#[allow(dead_code)]
pub struct BrowserSession {
    browser: Browser,
    tab: Arc<headless_chrome::Tab>,
    profile_name: String,
    profile_dir: PathBuf,
    headed: bool,
}

#[allow(dead_code)]
impl BrowserSession {
    /// Create a new session with a named profile.
    /// Profile data persists in ~/.octoview/profiles/<name>/.
    pub fn new(profile_name: &str, headed: bool, timeout_secs: u64) -> Result<Self, String> {
        let profile_dir = profile_path(profile_name);

        // Create profile directory if it doesn't exist
        std::fs::create_dir_all(&profile_dir)
            .map_err(|e| format!("Cannot create profile dir: {}", e))?;

        let stealth = stealth_args(headed);
        let arg_refs: Vec<&std::ffi::OsStr> = stealth
            .iter()
            .map(|s| std::ffi::OsStr::new(s.as_str()))
            .collect();

        let options = LaunchOptions::default_builder()
            .headless(!headed) // headed = visible Chrome window for manual login
            .sandbox(false)
            .window_size(Some((1920, 1080)))
            .idle_browser_timeout(Duration::from_secs(timeout_secs))
            .user_data_dir(Some(profile_dir.clone()))
            .args(arg_refs)
            .build()
            .map_err(|e| format!("Chrome launch config error: {}", e))?;

        let mode = if headed { "headed" } else { "headless" };
        eprintln!("[session] Launching {} Chrome (profile: {})...", mode, profile_name);

        let browser = Browser::new(options)
            .map_err(|e| format!("Chrome launch error: {} (is Chrome/Edge installed?)", e))?;

        let tab = browser
            .new_tab()
            .map_err(|e| format!("Tab creation error: {}", e))?;

        // Inject stealth if headless
        if !headed {
            tab.evaluate(STEALTH_JS, false)
                .map_err(|e| format!("Stealth injection error: {}", e))?;
        }

        // Set realistic user-agent
        tab.set_user_agent(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
             (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            Some("en-US,en;q=0.9"),
            Some("Win32"),
        )
        .map_err(|e| format!("User-agent error: {}", e))?;

        eprintln!("[session] Ready (profile: {})", profile_name);

        Ok(Self {
            browser,
            tab,
            profile_name: profile_name.to_string(),
            profile_dir,
            headed,
        })
    }

    /// Navigate to a URL, wait for content, return OVD document.
    /// Cookies and session state are preserved across calls.
    pub fn navigate(&self, url: &str) -> Result<OvdDocument, String> {
        eprintln!("[session] Navigating to {}...", truncate(url, 60));

        // Re-inject stealth before navigation
        if !self.headed {
            let _ = self.tab.evaluate(STEALTH_JS, false);
        }

        self.tab.navigate_to(url)
            .map_err(|e| format!("Navigation error: {}", e))?;

        self.tab.wait_until_navigated()
            .map_err(|e| format!("Navigation wait error: {}", e))?;

        // Re-inject stealth after navigation
        if !self.headed {
            let _ = self.tab.evaluate(STEALTH_JS, false);
        }

        // Wait for dynamic content
        let _ = self.tab.evaluate(WAIT_FOR_CONTENT_JS, true);

        // Extract HTML and build OVD
        let html = self.tab.get_content()
            .map_err(|e| format!("Content extraction error: {}", e))?;
        let final_url = self.tab.get_url();

        eprintln!("[session] Got {} bytes from {}", html.len(), truncate(&final_url, 50));

        Ok(extract::extract_html(&html, &final_url))
    }

    /// Get the current page URL.
    pub fn current_url(&self) -> String {
        self.tab.get_url()
    }

    /// Get the current page title.
    pub fn current_title(&self) -> Result<String, String> {
        let result = self.tab.evaluate("document.title", false)
            .map_err(|e| format!("Eval error: {}", e))?;
        Ok(result.value
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_default())
    }

    /// Get current page as HTML string.
    pub fn get_html(&self) -> Result<String, String> {
        self.tab.get_content()
            .map_err(|e| format!("Content error: {}", e))
    }

    // ── Cookies ──

    /// Get all cookies from the current session.
    pub fn get_cookies(&self) -> Result<Vec<CookieData>, String> {
        let cookies = self.tab.get_cookies()
            .map_err(|e| format!("Cookie read error: {}", e))?;

        Ok(cookies.into_iter().map(|c| CookieData {
            name: c.name,
            value: c.value,
            domain: c.domain,
            path: c.path,
            secure: c.secure,
            http_only: c.http_only,
            expires: c.expires,
        }).collect())
    }

    /// Set cookies in the current session.
    pub fn set_cookies(&self, cookies: Vec<CookieData>) -> Result<(), String> {
        let params: Vec<Network::CookieParam> = cookies.into_iter().map(|c| {
            Network::CookieParam {
                name: c.name,
                value: c.value,
                url: None,
                domain: Some(c.domain),
                path: Some(c.path),
                secure: Some(c.secure),
                http_only: Some(c.http_only),
                same_site: None,
                expires: if c.expires > 0.0 { Some(c.expires) } else { None },
                priority: None,
                same_party: None,
                source_scheme: None,
                source_port: None,
                partition_key: None,
            }
        }).collect();

        self.tab.set_cookies(params)
            .map_err(|e| format!("Cookie set error: {}", e))?;

        Ok(())
    }

    // ── Form Interaction ──

    /// Type text into an element found by CSS selector.
    pub fn type_into(&self, selector: &str, text: &str) -> Result<(), String> {
        let element = self.tab.find_element(selector)
            .map_err(|e| format!("Element not found '{}': {}", selector, e))?;
        element.click()
            .map_err(|e| format!("Click error: {}", e))?;
        self.tab.type_str(text)
            .map_err(|e| format!("Type error: {}", e))?;
        Ok(())
    }

    /// Click an element found by CSS selector.
    pub fn click(&self, selector: &str) -> Result<(), String> {
        let element = self.tab.find_element(selector)
            .map_err(|e| format!("Element not found '{}': {}", selector, e))?;
        element.click()
            .map_err(|e| format!("Click error: {}", e))?;

        // Wait briefly for any navigation/re-render
        std::thread::sleep(Duration::from_millis(300));
        Ok(())
    }

    /// Press a keyboard key (e.g., "Enter", "Tab", "Escape").
    pub fn press_key(&self, key: &str) -> Result<(), String> {
        self.tab.press_key(key)
            .map_err(|e| format!("Key press error: {}", e))?;
        std::thread::sleep(Duration::from_millis(200));
        Ok(())
    }

    /// Wait for an element to appear (with timeout).
    pub fn wait_for(&self, selector: &str, timeout_ms: u64) -> Result<(), String> {
        self.tab.set_default_timeout(Duration::from_millis(timeout_ms));
        self.tab.wait_for_element(selector)
            .map_err(|e| format!("Wait timeout for '{}': {}", selector, e))?;
        // Reset timeout
        self.tab.set_default_timeout(Duration::from_secs(10));
        Ok(())
    }

    /// Evaluate JavaScript in the current page context.
    pub fn eval(&self, js: &str) -> Result<String, String> {
        let result = self.tab.evaluate(js, false)
            .map_err(|e| format!("Eval error: {}", e))?;
        Ok(result.value
            .map(|v| {
                if let Some(s) = v.as_str() {
                    s.to_string()
                } else {
                    v.to_string()
                }
            })
            .unwrap_or_default())
    }

    // ── Screenshots ──

    /// Capture a full-page screenshot as PNG bytes.
    pub fn screenshot(&self) -> Result<Vec<u8>, String> {
        self.tab.capture_screenshot(
            Page::CaptureScreenshotFormatOption::Png,
            None,
            None,
            true,
        )
        .map_err(|e| format!("Screenshot error: {}", e))
    }

    /// Capture a screenshot and save to file.
    pub fn screenshot_to_file(&self, path: &str) -> Result<(), String> {
        let data = self.screenshot()?;
        std::fs::write(path, &data)
            .map_err(|e| format!("Write error: {}", e))?;
        eprintln!("[session] Screenshot saved: {} ({} bytes)", path, data.len());
        Ok(())
    }

    /// Get profile info.
    pub fn profile_info(&self) -> String {
        format!(
            "Profile: {}\nPath: {}\nMode: {}\nURL: {}",
            self.profile_name,
            self.profile_dir.display(),
            if self.headed { "headed (visible)" } else { "headless" },
            self.current_url()
        )
    }

    /// Get the profile name.
    pub fn profile_name(&self) -> &str {
        &self.profile_name
    }

    /// Check if this is a headed (visible) session.
    pub fn is_headed(&self) -> bool {
        self.headed
    }

    /// Open a new tab in the same browser session.
    #[allow(dead_code)]
    pub fn new_tab(&self) -> Result<Arc<headless_chrome::Tab>, String> {
        self.browser.new_tab()
            .map_err(|e| format!("New tab error: {}", e))
    }
}

// ─── Interactive Login ─────────────────────────────────────────────────────

/// Launch a visible Chrome for manual login.
/// The user logs in manually, then we capture the session state.
pub fn interactive_login(profile_name: &str, url: &str, timeout_secs: u64) -> Result<BrowserSession, String> {
    eprintln!("┌─────────────────────────────────────────────────────┐");
    eprintln!("│ OctoView Interactive Login                         │");
    eprintln!("│                                                    │");
    eprintln!("│ A Chrome window will open. Log in manually.        │");
    eprintln!("│ When done, press ENTER here to capture session.    │");
    eprintln!("│                                                    │");
    eprintln!("│ Profile: {:<41}│", profile_name);
    eprintln!("│ URL:     {:<41}│", truncate(url, 41));
    eprintln!("└─────────────────────────────────────────────────────┘");
    eprintln!();

    // Launch headed Chrome with the profile
    let session = BrowserSession::new(profile_name, true, timeout_secs)?;

    // Navigate to the login URL
    session.tab.navigate_to(url)
        .map_err(|e| format!("Navigation error: {}", e))?;

    session.tab.wait_until_navigated()
        .map_err(|e| format!("Navigation wait error: {}", e))?;

    eprintln!("[login] Chrome opened at {}", url);
    eprintln!("[login] Complete your login, then press ENTER here...");

    // Wait for user to press Enter
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)
        .map_err(|e| format!("Input error: {}", e))?;

    // Capture session state
    let cookies = session.get_cookies()?;
    let title = session.current_title().unwrap_or_default();
    let final_url = session.current_url();

    eprintln!();
    eprintln!("[login] Session captured:");
    eprintln!("  URL:     {}", truncate(&final_url, 60));
    eprintln!("  Title:   {}", truncate(&title, 60));
    eprintln!("  Cookies: {}", cookies.len());
    eprintln!();
    eprintln!("[login] Session saved to profile '{}'. Use --profile {} for authenticated browsing.", profile_name, profile_name);

    // Save cookies to profile dir for backup
    let cookie_path = profile_path(profile_name).join("cookies.json");
    let json = cookies_to_json(&cookies);
    let _ = std::fs::write(&cookie_path, &json);

    Ok(session)
}

// ─── Launch Arguments ──────────────────────────────────────────────────────

/// Stealth launch args (adapted for session mode).
fn stealth_args(headed: bool) -> Vec<String> {
    let mut args = vec![
        "--disable-blink-features=AutomationControlled".to_string(),
        "--disable-infobars".to_string(),
        "--window-size=1920,1080".to_string(),
        "--start-maximized".to_string(),
        "--enable-webgl".to_string(),
        "--disable-extensions".to_string(),
        "--disable-component-extensions-with-background-pages".to_string(),
        "--disable-save-password-bubble".to_string(),
        "--disable-breakpad".to_string(),
        "--no-first-run".to_string(),
        "--no-default-browser-check".to_string(),
        "--lang=en-US,en".to_string(),
        "--log-level=3".to_string(),
        "--disable-dev-shm-usage".to_string(),
    ];

    // Only add --headless=new for headless mode
    if !headed {
        args.insert(0, "--headless=new".to_string());
    }

    args
}

/// DOM stability wait script.
const WAIT_FOR_CONTENT_JS: &str = r#"
new Promise((resolve) => {
    let checks = 0;
    let lastNodeCount = 0;
    let stableCount = 0;
    const check = () => {
        checks++;
        const currentNodes = document.querySelectorAll('*').length;
        if (currentNodes === lastNodeCount) {
            stableCount++;
        } else {
            stableCount = 0;
        }
        lastNodeCount = currentNodes;
        if (stableCount >= 4 || checks > 100) {
            resolve(currentNodes);
        } else {
            setTimeout(check, 100);
        }
    };
    setTimeout(check, 500);
})
"#;

// ─── Helpers ───────────────────────────────────────────────────────────────

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", &s[..max.saturating_sub(3)])
    } else {
        s.to_string()
    }
}

fn json_esc(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn split_json_objects(s: &str) -> Vec<String> {
    let mut objects = Vec::new();
    let mut depth = 0;
    let mut start = 0;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, ch) in s.char_indices() {
        if escape_next { escape_next = false; continue; }
        if ch == '\\' && in_string { escape_next = true; continue; }
        if ch == '"' { in_string = !in_string; continue; }
        if in_string { continue; }

        match ch {
            '{' => { if depth == 0 { start = i; } depth += 1; }
            '}' => { depth -= 1; if depth == 0 { objects.push(s[start..=i].to_string()); } }
            _ => {}
        }
    }
    objects
}

fn extract_json_str(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\":", key);
    let idx = json.find(&pattern)?;
    let rest = json[idx + pattern.len()..].trim_start();
    if !rest.starts_with('"') { return None; }
    let inner = &rest[1..];
    let mut end = 0;
    let mut esc = false;
    for ch in inner.chars() {
        if esc { esc = false; end += ch.len_utf8(); continue; }
        if ch == '\\' { esc = true; end += 1; continue; }
        if ch == '"' { break; }
        end += ch.len_utf8();
    }
    Some(inner[..end].replace("\\\"", "\"").replace("\\\\", "\\"))
}

fn extract_json_bool(json: &str, key: &str) -> Option<bool> {
    let pattern = format!("\"{}\":", key);
    let idx = json.find(&pattern)?;
    let rest = json[idx + pattern.len()..].trim_start();
    if rest.starts_with("true") { Some(true) }
    else if rest.starts_with("false") { Some(false) }
    else { None }
}

fn extract_json_num(json: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{}\":", key);
    let idx = json.find(&pattern)?;
    let rest = json[idx + pattern.len()..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-').unwrap_or(rest.len());
    rest[..end].parse().ok()
}
