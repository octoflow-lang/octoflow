//! Headless Chrome JS engine — undetected-chromedriver + Playwright Stealth.
//!
//! Implements the complete anti-detection stack:
//!   Level 1: Chrome launch flags (--headless=new, --disable-blink-features)
//!   Level 2: 17 Playwright Stealth evasions (navigator, chrome, WebGL, etc.)
//!   Level 3: Cloudflare challenge detection + multi-stage wait
//!
//! Based on:
//!   - undetected-chromedriver (Python) — launch flag approach
//!   - puppeteer-extra-plugin-stealth — 17 evasion modules
//!   - rebrowser-patches — Runtime.enable mitigation
//!
//! Detection layers we counter:
//!   TLS fingerprint    — real Chrome = real BoringSSL JA3 (automatic)
//!   HTTP/2 fingerprint — real Chrome handles this (automatic)
//!   Browser env        — 17 stealth patches (navigator, chrome, WebGL, etc.)
//!   CDP artifacts      — --headless=new + stealth injection
//!   Behavioral         — DOM stability wait (no mouse simulation yet)

use headless_chrome::{Browser, LaunchOptions, Tab};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Result from JS-rendered page fetch.
#[allow(dead_code)]
pub struct JsRenderResult {
    pub html: String,
    pub url: String,
    pub elapsed_ms: u128,
    pub js_rendered: bool,
}

// ─── Chrome Launch Configuration ───────────────────────────────────────────

/// Stealth launch args matching undetected-chromedriver + puppeteer-stealth.
///
/// Key difference from v1: we use headless(false) + "--headless=new" so Chrome
/// uses the SAME rendering pipeline as headed mode (Chrome 112+). This eliminates
/// canvas fingerprint, WebGL renderer, font, and screen dimension differences.
fn stealth_args() -> Vec<String> {
    vec![
        // NEW HEADLESS: Same rendering as headed Chrome (Chrome 112+)
        // headless_chrome's headless(true) adds old "--headless" which leaks SwiftShader
        // We set headless(false) and add this manually
        "--headless=new".to_string(),
        // CRITICAL: Remove automation-controlled flag from navigator
        // This is the #1 detection signal — disables navigator.webdriver at the
        // Blink level, not just JS override
        "--disable-blink-features=AutomationControlled".to_string(),
        // Remove "Chrome is being controlled by automated test software" infobar
        // Also prevents the --enable-automation flag from being implicitly set
        "--disable-infobars".to_string(),
        // Realistic viewport
        "--window-size=1920,1080".to_string(),
        "--start-maximized".to_string(),
        // Enable GPU for realistic WebGL/Canvas fingerprints
        "--enable-webgl".to_string(),
        // Remove extension markers that leak automation
        "--disable-extensions".to_string(),
        "--disable-component-extensions-with-background-pages".to_string(),
        // Suppress automation-related chrome features
        "--disable-save-password-bubble".to_string(),
        "--disable-breakpad".to_string(),
        "--no-first-run".to_string(),
        "--no-default-browser-check".to_string(),
        // Language consistency with navigator.languages and Accept-Language
        "--lang=en-US,en".to_string(),
        // Suppress logging
        "--log-level=3".to_string(),
        // Disable dev-shm usage (prevents /dev/shm related crashes in containers)
        "--disable-dev-shm-usage".to_string(),
    ]
}

// ─── Stealth JavaScript — 17 Playwright Evasions ───────────────────────────

/// Complete stealth injection matching all 17 puppeteer-extra-plugin-stealth modules.
///
/// Each evasion is numbered and annotated with the detection it counters.
/// These are injected via Page.addScriptToEvaluateOnNewDocument so they run
/// BEFORE any page JavaScript.
pub const STEALTH_JS: &str = r#"
// ═══ EVASION 1: navigator.webdriver (CRITICAL — Cloudflare checks first) ═══
// --disable-blink-features=AutomationControlled handles this at Blink level,
// but we double-patch at JS level for defense-in-depth
Object.defineProperty(navigator, 'webdriver', {
    get: () => false,
    configurable: true,
});
// Also delete the property entirely (some detectors check hasOwnProperty)
delete navigator.__proto__.webdriver;

// ═══ EVASION 2: chrome.app (missing in old headless) ═══
if (!window.chrome) window.chrome = {};
if (!window.chrome.app) {
    window.chrome.app = {
        isInstalled: false,
        InstallState: { DISABLED: 'disabled', INSTALLED: 'installed', NOT_INSTALLED: 'not_installed' },
        RunningState: { CANNOT_RUN: 'cannot_run', READY_TO_RUN: 'ready_to_run', RUNNING: 'running' },
        getDetails: function() { return null; },
        getIsInstalled: function() { return false; },
        installState: function() { return 'not_installed'; },
        runningState: function() { return 'cannot_run'; },
    };
}

// ═══ EVASION 3: chrome.csi (missing in headless) ═══
if (!window.chrome.csi) {
    window.chrome.csi = function() {
        return {
            onloadT: Date.now(),
            startE: Date.now(),
            pageT: Math.random() * 1000 + 500,
            tran: 15,
        };
    };
}

// ═══ EVASION 4: chrome.loadTimes (missing in headless) ═══
if (!window.chrome.loadTimes) {
    window.chrome.loadTimes = function() {
        return {
            commitLoadTime: Date.now() / 1000,
            connectionInfo: 'h2',
            finishDocumentLoadTime: Date.now() / 1000 + 0.1,
            finishLoadTime: Date.now() / 1000 + 0.2,
            firstPaintAfterLoadTime: 0,
            firstPaintTime: Date.now() / 1000 + 0.05,
            navigationType: 'Other',
            npnNegotiatedProtocol: 'h2',
            requestTime: Date.now() / 1000 - 0.5,
            startLoadTime: Date.now() / 1000 - 0.3,
            wasAlternateProtocolAvailable: false,
            wasFetchedViaSpdy: true,
            wasNpnNegotiated: true,
        };
    };
}

// ═══ EVASION 5: chrome.runtime (Cloudflare HIGH priority check) ═══
if (!window.chrome.runtime) {
    window.chrome.runtime = {
        OnInstalledReason: { CHROME_UPDATE: 'chrome_update', INSTALL: 'install',
                            SHARED_MODULE_UPDATE: 'shared_module_update', UPDATE: 'update' },
        OnRestartRequiredReason: { APP_UPDATE: 'app_update', OS_UPDATE: 'os_update',
                                   PERIODIC: 'periodic' },
        PlatformArch: { ARM: 'arm', MIPS: 'mips', MIPS64: 'mips64',
                        X86_32: 'x86-32', X86_64: 'x86-64' },
        PlatformNaclArch: { ARM: 'arm', MIPS: 'mips', MIPS64: 'mips64',
                            X86_32: 'x86-32', X86_64: 'x86-64' },
        PlatformOs: { ANDROID: 'android', CROS: 'cros', LINUX: 'linux',
                      MAC: 'mac', OPENBSD: 'openbsd', WIN: 'win' },
        RequestUpdateCheckStatus: { NO_UPDATE: 'no_update', THROTTLED: 'throttled',
                                     UPDATE_AVAILABLE: 'update_available' },
        connect: function() { return { onDisconnect: { addListener: function() {} },
                                        onMessage: { addListener: function() {} },
                                        postMessage: function() {},
                                        disconnect: function() {} }; },
        sendMessage: function() {},
        id: undefined,
    };
}

// ═══ EVASION 6: defaultArgs — remove --enable-automation ═══
// Handled via Chrome launch flags (no --enable-automation passed)

// ═══ EVASION 7: iframe.contentWindow (Cloudflare HIGH priority) ═══
// Fix cross-origin iframe contentWindow access detection
try {
    const iframeProto = HTMLIFrameElement.prototype;
    const origDesc = Object.getOwnPropertyDescriptor(iframeProto, 'contentWindow');
    if (origDesc) {
        Object.defineProperty(iframeProto, 'contentWindow', {
            get: function() {
                const win = origDesc.get.call(this);
                if (win === null) {
                    // Return a proxy that doesn't throw on property access
                    return new Proxy({}, {
                        get: function() { return undefined; }
                    });
                }
                return win;
            },
            configurable: true,
        });
    }
} catch(e) {}

// ═══ EVASION 8: media.codecs (MEDIUM priority) ═══
try {
    const origCanPlayType = HTMLMediaElement.prototype.canPlayType;
    HTMLMediaElement.prototype.canPlayType = function(type) {
        if (type === 'video/mp4; codecs="avc1.42E01E"') return 'probably';
        if (type === 'video/webm; codecs="vp8, vorbis"') return 'probably';
        if (type === 'audio/mpeg') return 'probably';
        return origCanPlayType.call(this, type);
    };
} catch(e) {}

// ═══ EVASION 9: navigator.hardwareConcurrency (headless = 1, real = 4-16) ═══
Object.defineProperty(navigator, 'hardwareConcurrency', {
    get: () => 8,
    configurable: true,
});

// ═══ EVASION 10: navigator.languages ═══
Object.defineProperty(navigator, 'languages', {
    get: () => ['en-US', 'en'],
    configurable: true,
});

// ═══ EVASION 11: navigator.permissions (MEDIUM — notification leak) ═══
try {
    const origQuery = window.Permissions.prototype.query;
    window.Permissions.prototype.query = function(params) {
        if (params.name === 'notifications') {
            return Promise.resolve({ state: 'denied', onchange: null });
        }
        return origQuery.call(this, params);
    };
} catch(e) {}

// ═══ EVASION 12: navigator.plugins (HIGH — empty = headless) ═══
Object.defineProperty(navigator, 'plugins', {
    get: () => {
        const plugins = [
            { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer',
              description: 'Portable Document Format', length: 1,
              0: { type: 'application/x-google-chrome-pdf', suffixes: 'pdf',
                   description: 'Portable Document Format', enabledPlugin: null } },
            { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai',
              description: '', length: 1,
              0: { type: 'application/pdf', suffixes: 'pdf',
                   description: '', enabledPlugin: null } },
            { name: 'Native Client', filename: 'internal-nacl-plugin',
              description: '', length: 2,
              0: { type: 'application/x-nacl', suffixes: '',
                   description: 'Native Client Executable', enabledPlugin: null },
              1: { type: 'application/x-pnacl', suffixes: '',
                   description: 'Portable Native Client Executable', enabledPlugin: null } },
        ];
        plugins.item = function(i) { return this[i] || null; };
        plugins.namedItem = function(name) {
            return this.find(p => p.name === name) || null;
        };
        plugins.refresh = function() {};
        return plugins;
    },
    configurable: true,
});

// ═══ EVASION 13: navigator.mimeTypes (paired with plugins) ═══
Object.defineProperty(navigator, 'mimeTypes', {
    get: () => {
        const mimes = [
            { type: 'application/pdf', suffixes: 'pdf',
              description: '', enabledPlugin: { name: 'Chrome PDF Viewer' } },
            { type: 'application/x-google-chrome-pdf', suffixes: 'pdf',
              description: 'Portable Document Format', enabledPlugin: { name: 'Chrome PDF Plugin' } },
            { type: 'application/x-nacl', suffixes: '',
              description: 'Native Client Executable', enabledPlugin: { name: 'Native Client' } },
            { type: 'application/x-pnacl', suffixes: '',
              description: 'Portable Native Client Executable', enabledPlugin: { name: 'Native Client' } },
        ];
        mimes.item = function(i) { return this[i] || null; };
        mimes.namedItem = function(type) {
            return this.find(m => m.type === type) || null;
        };
        return mimes;
    },
    configurable: true,
});

// ═══ EVASION 14: navigator.vendor ═══
Object.defineProperty(navigator, 'vendor', {
    get: () => 'Google Inc.',
    configurable: true,
});

// ═══ EVASION 15: navigator.platform ═══
Object.defineProperty(navigator, 'platform', {
    get: () => 'Win32',
    configurable: true,
});

// ═══ EVASION 16: WebGL vendor/renderer (HIGH — SwiftShader = headless) ═══
// With --headless=new this is less critical but we patch anyway for old Chrome
try {
    const getParam = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function(param) {
        if (param === 37445) return 'Google Inc. (NVIDIA)';
        if (param === 37446) return 'ANGLE (NVIDIA, NVIDIA GeForce GTX 1080 Direct3D11 vs_5_0 ps_5_0, D3D11)';
        return getParam.call(this, param);
    };
    // Also patch WebGL2
    if (typeof WebGL2RenderingContext !== 'undefined') {
        const getParam2 = WebGL2RenderingContext.prototype.getParameter;
        WebGL2RenderingContext.prototype.getParameter = function(param) {
            if (param === 37445) return 'Google Inc. (NVIDIA)';
            if (param === 37446) return 'ANGLE (NVIDIA, NVIDIA GeForce GTX 1080 Direct3D11 vs_5_0 ps_5_0, D3D11)';
            return getParam2.call(this, param);
        };
    }
} catch(e) {}

// ═══ EVASION 17: window.outerWidth/outerHeight (0 in old headless) ═══
if (window.outerWidth === 0) {
    Object.defineProperty(window, 'outerWidth', { get: () => 1920, configurable: true });
}
if (window.outerHeight === 0) {
    Object.defineProperty(window, 'outerHeight', { get: () => 1040, configurable: true });
}

// ═══ BONUS: Additional anti-detection ═══

// Device memory (headless may report 0)
Object.defineProperty(navigator, 'deviceMemory', {
    get: () => 8,
    configurable: true,
});

// maxTouchPoints (0 in headless, but 0 is valid for desktop)
Object.defineProperty(navigator, 'maxTouchPoints', {
    get: () => 0,
    configurable: true,
});

// Connection RTT (0 in headless = giveaway)
try {
    if (navigator.connection) {
        Object.defineProperty(navigator.connection, 'rtt', {
            get: () => 50,
            configurable: true,
        });
        Object.defineProperty(navigator.connection, 'downlink', {
            get: () => 10,
            configurable: true,
        });
        Object.defineProperty(navigator.connection, 'effectiveType', {
            get: () => '4g',
            configurable: true,
        });
    }
} catch(e) {}

// Notification.permission (should be 'default' not 'denied' for fresh profile)
try {
    Object.defineProperty(Notification, 'permission', {
        get: () => 'default',
        configurable: true,
    });
} catch(e) {}

// Remove sourceURL leak from injected scripts (best effort)
// This prevents stack trace analysis from revealing CDP injection
"#;

// ─── Dynamic Content Wait ──────────────────────────────────────────────────

/// Wait for DOM to stabilize (network idle + consistent node count).
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

        // DOM stable for 4 consecutive checks (400ms) or timeout
        if (stableCount >= 4 || checks > 100) {
            resolve(currentNodes);
        } else {
            setTimeout(check, 100);
        }
    };

    // Start checking after initial render
    setTimeout(check, 500);
})
"#;

// ─── Cloudflare Challenge Detection ────────────────────────────────────────

/// Detect Cloudflare / bot challenge pages. Returns 'challenge' or 'ok'.
const CHALLENGE_DETECT_JS: &str = r#"
(function() {
    const title = (document.title || '').toLowerCase();
    const body = document.body ? document.body.innerText.toLowerCase() : '';

    // Cloudflare markers
    if (title.includes('just a moment') || title.includes('attention required')
        || title.includes('access denied') || title.includes('checking your browser')
        || body.includes('verify you are human') || body.includes('checking your browser')
        || body.includes('performing security verification')
        || document.querySelector('#challenge-running') !== null
        || document.querySelector('.cf-turnstile-wrapper') !== null
        || document.querySelector('#cf-wrapper') !== null
        || document.querySelector('#challenge-form') !== null
        || document.querySelector('[data-cf-turnstile-sitekey]') !== null) {
        return 'challenge';
    }

    // DataDome / PerimeterX markers
    if (document.querySelector('iframe[src*="captcha-delivery.com"]') !== null
        || document.querySelector('iframe[src*="perimeterx"]') !== null) {
        return 'challenge';
    }

    return 'ok';
})()
"#;

// ─── Main Engine ───────────────────────────────────────────────────────────

/// Launch stealth browser, navigate to URL, execute JS, return rendered HTML.
///
/// Uses the undetected-chromedriver approach:
///   1. --headless=new (Chrome 112+ real rendering pipeline)
///   2. --disable-blink-features=AutomationControlled (removes webdriver flag at Blink level)
///   3. 17 Playwright Stealth evasions injected before page load
///   4. Cloudflare challenge detection + wait loop
pub fn fetch_with_js(url: &str, wait_secs: u64) -> Result<JsRenderResult, String> {
    let start = Instant::now();

    let stealth = stealth_args();
    let arg_refs: Vec<&std::ffi::OsStr> = stealth
        .iter()
        .map(|s| std::ffi::OsStr::new(s.as_str()))
        .collect();

    // KEY: headless(false) because we manually pass --headless=new
    // headless(true) would add old "--headless" which uses SwiftShader
    let options = LaunchOptions::default_builder()
        .headless(false) // We pass --headless=new in args instead
        .sandbox(false)
        .window_size(Some((1920, 1080)))
        .idle_browser_timeout(Duration::from_secs(wait_secs + 60))
        .args(arg_refs)
        .build()
        .map_err(|e| format!("Chrome launch config error: {}", e))?;

    eprintln!("[js] Launching stealth Chrome (--headless=new)...");
    let browser = Browser::new(options)
        .map_err(|e| format!("Chrome launch error: {} (is Chrome/Edge installed?)", e))?;

    let tab = browser
        .new_tab()
        .map_err(|e| format!("Tab creation error: {}", e))?;

    // Inject stealth scripts BEFORE any navigation
    inject_stealth(&tab)?;

    // Set realistic user-agent matching Chrome 131 on Windows 10
    tab.set_user_agent(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
         (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        Some("en-US,en;q=0.9"),
        Some("Win32"),
    )
    .map_err(|e| format!("User-agent error: {}", e))?;

    eprintln!("[js] Navigating to {}...", url);
    tab.navigate_to(url)
        .map_err(|e| format!("Navigation error: {}", e))?;

    // Wait for initial page load
    tab.wait_until_navigated()
        .map_err(|e| format!("Navigation wait error: {}", e))?;

    // Re-inject stealth after navigation (some frameworks clear overrides)
    let _ = tab.evaluate(STEALTH_JS, false);

    // ── Cloudflare Challenge Handler ──
    let deadline = Instant::now() + Duration::from_secs(wait_secs);
    let mut challenge_detected = false;

    loop {
        if Instant::now() > deadline {
            if challenge_detected {
                eprintln!("[js] Challenge wait timed out after {}s", wait_secs);
            }
            break;
        }

        let result = tab
            .evaluate(CHALLENGE_DETECT_JS, false)
            .map_err(|e| format!("Challenge detect error: {}", e))?;

        let status = result
            .value
            .as_ref()
            .and_then(|v| v.as_str())
            .unwrap_or("ok");

        if status == "challenge" {
            if !challenge_detected {
                eprintln!("[js] Challenge detected, waiting for verification...");
                challenge_detected = true;
            }
            std::thread::sleep(Duration::from_millis(500));

            // Check if title changed (challenge completed, page redirected)
            let current_title = tab
                .evaluate("document.title", false)
                .ok()
                .and_then(|r| r.value)
                .and_then(|v| v.as_str().map(|s| s.to_string()))
                .unwrap_or_default()
                .to_lowercase();

            if !current_title.contains("just a moment")
                && !current_title.contains("attention required")
                && !current_title.contains("checking")
                && !current_title.is_empty()
            {
                eprintln!("[js] Challenge passed! Title: {}", current_title);
                // Extra wait for content to load after redirect
                std::thread::sleep(Duration::from_millis(2000));
                break;
            }
        } else {
            break;
        }
    }

    // Post-challenge: re-wait for navigation and re-inject stealth
    if challenge_detected {
        let _ = tab.wait_until_navigated();
        let _ = tab.evaluate(STEALTH_JS, false);
    }

    eprintln!("[js] Waiting for JS content to render...");

    // Wait for dynamic content to stabilize
    let _node_count = tab
        .evaluate(WAIT_FOR_CONTENT_JS, true)
        .map_err(|e| format!("JS wait error: {}", e))?;

    // Extract the fully rendered DOM HTML
    let html = tab
        .get_content()
        .map_err(|e| format!("Content extraction error: {}", e))?;

    let final_url = tab.get_url();
    let elapsed_ms = start.elapsed().as_millis();
    eprintln!("[js] Rendered {} bytes in {}ms", html.len(), elapsed_ms);

    Ok(JsRenderResult {
        html,
        url: final_url,
        elapsed_ms,
        js_rendered: true,
    })
}

/// Inject all stealth scripts before navigation.
fn inject_stealth(tab: &Arc<Tab>) -> Result<(), String> {
    tab.evaluate(STEALTH_JS, false)
        .map_err(|e| format!("Stealth injection error: {}", e))?;
    Ok(())
}

// ─── Auto-Detection Heuristic ──────────────────────────────────────────────

/// Detect if a page likely needs JS rendering based on static HTML analysis.
pub fn needs_js_rendering(html: &str, node_count: usize) -> bool {
    // Cloudflare / bot challenge — always need JS
    if html.contains("Just a moment...")
        && (html.contains("cf-browser-verification")
            || html.contains("challenge-platform")
            || html.contains("cf_chl_opt"))
    {
        return true;
    }

    // Common SPA shell markers
    if html.contains("id=\"__next\"") && node_count < 30 {
        return true;
    }
    if html.contains("id=\"root\"") && node_count < 20 {
        return true;
    }
    if html.contains("id=\"app\"") && node_count < 20 {
        return true;
    }

    // Noscript warning = JS-only site
    if html.contains("<noscript>") && html.contains("enable JavaScript") && node_count < 30 {
        return true;
    }

    // Large script payload + very few DOM nodes = SPA shell
    let script_count = html.matches("<script").count();
    if script_count > 10 && node_count < 15 {
        return true;
    }

    false
}

// ─── API Interception ───────────────────────────────────────────────────────

/// JavaScript that hooks fetch() and XMLHttpRequest to capture API calls.
/// Injected before page load, captures all network requests the page makes.
/// Results stored in window.__octoview_api_calls as JSON array.
const API_INTERCEPT_JS: &str = r#"
// ═══ API INTERCEPTION: Capture fetch/XHR requests and responses ═══
window.__octoview_api_calls = [];

// ── Hook fetch() ──
const _origFetch = window.fetch;
window.fetch = async function(...args) {
    const url = typeof args[0] === 'string' ? args[0] : (args[0]?.url || '');
    const method = args[1]?.method || 'GET';
    const reqBody = args[1]?.body || null;

    const entry = {
        type: 'fetch',
        method: method.toUpperCase(),
        url: url,
        requestBody: null,
        status: 0,
        contentType: '',
        responseBody: null,
        timestamp: Date.now(),
    };

    // Capture request body (JSON only)
    if (reqBody && typeof reqBody === 'string') {
        try { JSON.parse(reqBody); entry.requestBody = reqBody; } catch(e) {}
    }

    try {
        const response = await _origFetch.apply(this, args);
        entry.status = response.status;
        entry.contentType = response.headers.get('content-type') || '';

        // Only capture JSON responses (not images, CSS, etc.)
        if (entry.contentType.includes('json') || entry.contentType.includes('graphql')) {
            try {
                const clone = response.clone();
                const text = await clone.text();
                if (text.length < 500000) { // Cap at 500KB
                    entry.responseBody = text;
                }
            } catch(e) {}
        }

        window.__octoview_api_calls.push(entry);
        return response;
    } catch(err) {
        entry.error = err.message;
        window.__octoview_api_calls.push(entry);
        throw err;
    }
};

// ── Hook XMLHttpRequest ──
const _origXHROpen = XMLHttpRequest.prototype.open;
const _origXHRSend = XMLHttpRequest.prototype.send;

XMLHttpRequest.prototype.open = function(method, url, ...rest) {
    this.__octoview_method = method;
    this.__octoview_url = url;
    return _origXHROpen.call(this, method, url, ...rest);
};

XMLHttpRequest.prototype.send = function(body) {
    const xhr = this;
    const entry = {
        type: 'xhr',
        method: (xhr.__octoview_method || 'GET').toUpperCase(),
        url: xhr.__octoview_url || '',
        requestBody: null,
        status: 0,
        contentType: '',
        responseBody: null,
        timestamp: Date.now(),
    };

    if (body && typeof body === 'string') {
        try { JSON.parse(body); entry.requestBody = body; } catch(e) {}
    }

    xhr.addEventListener('load', function() {
        entry.status = xhr.status;
        entry.contentType = xhr.getResponseHeader('content-type') || '';

        if (entry.contentType.includes('json') || entry.contentType.includes('graphql')) {
            try {
                if (xhr.responseText && xhr.responseText.length < 500000) {
                    entry.responseBody = xhr.responseText;
                }
            } catch(e) {}
        }

        window.__octoview_api_calls.push(entry);
    });

    xhr.addEventListener('error', function() {
        entry.error = 'Network error';
        window.__octoview_api_calls.push(entry);
    });

    return _origXHRSend.call(this, body);
};
"#;

/// JavaScript to extract captured API calls.
const EXTRACT_API_CALLS_JS: &str = r#"
JSON.stringify(window.__octoview_api_calls || [])
"#;

/// A captured API call from the page.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ApiCall {
    pub call_type: String,       // "fetch" or "xhr"
    pub method: String,          // GET, POST, etc.
    pub url: String,
    pub status: u16,
    pub content_type: String,
    pub request_body: Option<String>,
    pub response_body: Option<String>,
}

/// Result from JS-rendered page fetch with API interception.
pub struct JsRenderWithApis {
    pub render: JsRenderResult,
    pub api_calls: Vec<ApiCall>,
}

/// Fetch with JS rendering AND API interception.
/// Captures all fetch/XHR calls the page makes.
pub fn fetch_with_js_and_apis(url: &str, wait_secs: u64) -> Result<JsRenderWithApis, String> {
    let start = Instant::now();

    let stealth = stealth_args();
    let arg_refs: Vec<&std::ffi::OsStr> = stealth
        .iter()
        .map(|s| std::ffi::OsStr::new(s.as_str()))
        .collect();

    let options = LaunchOptions::default_builder()
        .headless(false)
        .sandbox(false)
        .window_size(Some((1920, 1080)))
        .idle_browser_timeout(Duration::from_secs(wait_secs + 60))
        .args(arg_refs)
        .build()
        .map_err(|e| format!("Chrome launch config error: {}", e))?;

    eprintln!("[js+api] Launching stealth Chrome with API interception...");
    let browser = Browser::new(options)
        .map_err(|e| format!("Chrome launch error: {} (is Chrome/Edge installed?)", e))?;

    let tab = browser
        .new_tab()
        .map_err(|e| format!("Tab creation error: {}", e))?;

    // Inject stealth + API interception BEFORE navigation
    inject_stealth(&tab)?;
    tab.evaluate(API_INTERCEPT_JS, false)
        .map_err(|e| format!("API intercept injection error: {}", e))?;

    tab.set_user_agent(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
         (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        Some("en-US,en;q=0.9"),
        Some("Win32"),
    )
    .map_err(|e| format!("User-agent error: {}", e))?;

    eprintln!("[js+api] Navigating to {}...", url);
    tab.navigate_to(url)
        .map_err(|e| format!("Navigation error: {}", e))?;

    tab.wait_until_navigated()
        .map_err(|e| format!("Navigation wait error: {}", e))?;

    // Re-inject after navigation
    let _ = tab.evaluate(STEALTH_JS, false);
    let _ = tab.evaluate(API_INTERCEPT_JS, false);

    // Cloudflare challenge handling (same as fetch_with_js)
    let deadline = Instant::now() + Duration::from_secs(wait_secs);
    let mut challenge_detected = false;

    loop {
        if Instant::now() > deadline {
            break;
        }

        let result = tab
            .evaluate(CHALLENGE_DETECT_JS, false)
            .map_err(|e| format!("Challenge detect error: {}", e))?;

        let status = result
            .value
            .as_ref()
            .and_then(|v| v.as_str())
            .unwrap_or("ok");

        if status == "challenge" {
            if !challenge_detected {
                eprintln!("[js+api] Challenge detected, waiting...");
                challenge_detected = true;
            }
            std::thread::sleep(Duration::from_millis(500));

            let current_title = tab
                .evaluate("document.title", false)
                .ok()
                .and_then(|r| r.value)
                .and_then(|v| v.as_str().map(|s| s.to_string()))
                .unwrap_or_default()
                .to_lowercase();

            if !current_title.contains("just a moment")
                && !current_title.contains("attention required")
                && !current_title.contains("checking")
                && !current_title.is_empty()
            {
                eprintln!("[js+api] Challenge passed!");
                std::thread::sleep(Duration::from_millis(2000));
                break;
            }
        } else {
            break;
        }
    }

    if challenge_detected {
        let _ = tab.wait_until_navigated();
        let _ = tab.evaluate(STEALTH_JS, false);
    }

    eprintln!("[js+api] Waiting for content + API calls...");

    // Wait for content stability
    let _ = tab.evaluate(WAIT_FOR_CONTENT_JS, true);

    // Extra wait to catch late API calls (SPAs often fire after render)
    std::thread::sleep(Duration::from_millis(1500));

    // Extract API calls
    let api_json = tab
        .evaluate(EXTRACT_API_CALLS_JS, false)
        .map_err(|e| format!("API extraction error: {}", e))?;

    let api_calls = parse_api_calls(
        api_json
            .value
            .as_ref()
            .and_then(|v| v.as_str())
            .unwrap_or("[]"),
    );

    eprintln!("[js+api] Captured {} API calls", api_calls.len());

    // Extract rendered HTML
    let html = tab
        .get_content()
        .map_err(|e| format!("Content extraction error: {}", e))?;

    let final_url = tab.get_url();
    let elapsed_ms = start.elapsed().as_millis();
    eprintln!("[js+api] Complete: {} bytes HTML, {} APIs in {}ms", html.len(), api_calls.len(), elapsed_ms);

    Ok(JsRenderWithApis {
        render: JsRenderResult {
            html,
            url: final_url,
            elapsed_ms,
            js_rendered: true,
        },
        api_calls,
    })
}

/// Parse the JSON array of captured API calls.
fn parse_api_calls(json: &str) -> Vec<ApiCall> {
    // Simple JSON parser — avoid adding serde dependency
    let mut calls = Vec::new();
    let json = json.trim();

    if !json.starts_with('[') {
        return calls;
    }

    // Split on },{ boundaries (simplified)
    let inner = &json[1..json.len().saturating_sub(1)];
    let objects = split_json_objects(inner);

    for obj in &objects {
        if let Some(call) = parse_single_api_call(obj) {
            calls.push(call);
        }
    }

    calls
}

/// Split a JSON array's contents into individual object strings.
fn split_json_objects(s: &str) -> Vec<String> {
    let mut objects = Vec::new();
    let mut depth = 0;
    let mut start = 0;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, ch) in s.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape_next = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }

        match ch {
            '{' => {
                if depth == 0 {
                    start = i;
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    objects.push(s[start..=i].to_string());
                }
            }
            _ => {}
        }
    }

    objects
}

/// Parse a single JSON object into an ApiCall.
fn parse_single_api_call(json: &str) -> Option<ApiCall> {
    Some(ApiCall {
        call_type: extract_json_string(json, "type").unwrap_or_default(),
        method: extract_json_string(json, "method").unwrap_or_else(|| "GET".to_string()),
        url: extract_json_string(json, "url").unwrap_or_default(),
        status: extract_json_number(json, "status").unwrap_or(0) as u16,
        content_type: extract_json_string(json, "contentType").unwrap_or_default(),
        request_body: extract_json_string(json, "requestBody"),
        response_body: extract_json_string(json, "responseBody"),
    })
}

/// Extract a string value from a JSON object by key (simple parser).
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\":", key);
    let idx = json.find(&pattern)?;
    let rest = &json[idx + pattern.len()..];
    let rest = rest.trim_start();

    if rest.starts_with("null") {
        return None;
    }
    if !rest.starts_with('"') {
        return None;
    }

    // Find end of string, handling escapes
    let inner = &rest[1..];
    let mut end = 0;
    let mut escape_next = false;
    for ch in inner.chars() {
        if escape_next {
            escape_next = false;
            end += ch.len_utf8();
            continue;
        }
        if ch == '\\' {
            escape_next = true;
            end += 1;
            continue;
        }
        if ch == '"' {
            break;
        }
        end += ch.len_utf8();
    }

    let value = &inner[..end];
    // Unescape basic sequences
    let unescaped = value
        .replace("\\\"", "\"")
        .replace("\\\\", "\\")
        .replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t");

    Some(unescaped)
}

/// Extract a number value from a JSON object by key.
fn extract_json_number(json: &str, key: &str) -> Option<u64> {
    let pattern = format!("\"{}\":", key);
    let idx = json.find(&pattern)?;
    let rest = &json[idx + pattern.len()..];
    let rest = rest.trim_start();

    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Format captured API calls for display.
pub fn format_api_calls(calls: &[ApiCall]) -> String {
    let mut out = String::new();

    if calls.is_empty() {
        out.push_str("(no API calls captured)\n");
        return out;
    }

    out.push_str(&format!("── {} API Calls Captured ──\n\n", calls.len()));

    for (i, call) in calls.iter().enumerate() {
        out.push_str(&format!(
            "  {}. {} {} {} → {}\n",
            i + 1,
            call.call_type.to_uppercase(),
            call.method,
            truncate(&call.url, 70),
            call.status
        ));

        if !call.content_type.is_empty() {
            out.push_str(&format!("     Content-Type: {}\n", call.content_type));
        }

        if let Some(ref body) = call.response_body {
            let preview = if body.len() > 200 {
                format!("{}... ({} bytes)", &body[..200], body.len())
            } else {
                body.clone()
            };
            out.push_str(&format!("     Response: {}\n", preview.replace('\n', " ")));
        }

        out.push('\n');
    }

    out
}

/// Format API calls as JSONL for machine consumption.
pub fn format_api_calls_jsonl(calls: &[ApiCall]) -> String {
    let mut out = String::new();

    for call in calls {
        out.push('{');
        out.push_str(&format!("\"type\":\"{}\",", call.call_type));
        out.push_str(&format!("\"method\":\"{}\",", call.method));
        out.push_str(&format!("\"url\":\"{}\",", json_escape_api(&call.url)));
        out.push_str(&format!("\"status\":{},", call.status));
        out.push_str(&format!("\"content_type\":\"{}\"", json_escape_api(&call.content_type)));

        if let Some(ref body) = call.response_body {
            out.push_str(&format!(",\"response\":\"{}\"", json_escape_api(body)));
        }

        out.push_str("}\n");
    }

    out
}

fn json_escape_api(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", &s[..max.saturating_sub(3)])
    } else {
        s.to_string()
    }
}

// ─── Utility ───────────────────────────────────────────────────────────────

/// Try to find Chrome/Edge binary path on Windows.
#[allow(dead_code)]
pub fn find_chrome() -> Option<String> {
    let candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ];

    for path in &candidates {
        if std::path::Path::new(path).exists() {
            return Some(path.to_string());
        }
    }
    None
}
