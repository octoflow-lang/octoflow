# OctoView Stealth Engine — Technical Deep Dive

## Architecture

OctoView uses a **two-tier fetch pipeline**:

```
URL ──► Static HTTP (reqwest) ──► html5ever parse ──► OVD
         ~150ms, no JS                    │
         │                                │
         └── if <20 nodes (--auto-js) ────┘
                    │
                    ▼
         Headless Chrome ──► full JS render ──► html5ever parse ──► OVD
         ~5-10s, stealth mode
```

**Tier A** (static): Amazon, BBC, Wikipedia, HN — fast path, no Chrome needed.
**Tier B** (JS hydration): Netflix, LinkedIn, react.dev — static gets most content.
**Tier C** (SPA/Cloudflare): Twitter, YouTube, Discord — requires headless Chrome.

---

## Stealth Approach: Three Levels

### Level 1: Chrome Launch Flags

Based on **undetected-chromedriver** (Python) approach:

| Flag | Purpose | Detection it counters |
|------|---------|----------------------|
| `--headless=new` | Chrome 112+ new headless — same rendering as headed | Canvas/WebGL/font fingerprint |
| `--disable-blink-features=AutomationControlled` | Removes `navigator.webdriver` at Blink level | #1 bot detection signal |
| `--disable-infobars` | No "controlled by automation" bar | Visual automation indicator |
| `--disable-extensions` | No extension markers | Extension fingerprinting |
| `--disable-component-extensions-with-background-pages` | Remove background extension traces | Process enumeration |
| `--window-size=1920,1080` | Realistic viewport | Screen dimension checks |
| `--enable-webgl` | GPU rendering for realistic fingerprints | WebGL hash comparison |
| `--lang=en-US,en` | Consistent with navigator.languages | Language mismatch detection |

**Key insight**: We set `headless(false)` in the `LaunchOptions` and pass `--headless=new` as a custom argument. The old `--headless` flag (what `headless(true)` adds) uses SwiftShader software rendering which produces a completely different canvas/WebGL fingerprint.

### Level 2: 17 Playwright Stealth Evasions

All 17 modules from `puppeteer-extra-plugin-stealth`, injected before page load:

| # | Module | Cloudflare Priority | What it does |
|---|--------|-------------------|--------------|
| 1 | `navigator.webdriver` | CRITICAL | `false` + delete from prototype |
| 2 | `chrome.app` | HIGH | Full app object with install/running state |
| 3 | `chrome.csi` | MEDIUM | Timing function with realistic values |
| 4 | `chrome.loadTimes` | MEDIUM | Network timing (h2, SPDY flags) |
| 5 | `chrome.runtime` | HIGH | Full runtime object (connect, sendMessage) |
| 6 | `defaultArgs` | HIGH | No --enable-automation (launch flags) |
| 7 | `iframe.contentWindow` | HIGH | Cross-origin iframe Proxy fix |
| 8 | `media.codecs` | MEDIUM | canPlayType returns 'probably' |
| 9 | `hardwareConcurrency` | MEDIUM | 8 (headless default is 1) |
| 10 | `languages` | LOW | ['en-US', 'en'] |
| 11 | `permissions` | MEDIUM | Notification permission normalization |
| 12 | `plugins` | HIGH | 3 Chrome plugins with full structure |
| 13 | `mimeTypes` | MEDIUM | 4 MIME types matching plugins |
| 14 | `vendor` | LOW | 'Google Inc.' |
| 15 | `platform` | LOW | 'Win32' |
| 16 | `WebGL` | HIGH | NVIDIA renderer (vs SwiftShader) |
| 17 | `outerDimensions` | MEDIUM | window.outerWidth/Height |

Plus bonus evasions: deviceMemory (8), maxTouchPoints (0), connection RTT (50ms), Notification.permission ('default').

### Level 3: Cloudflare Challenge Handler

Multi-stage detection and wait loop:

```
Navigate ──► Check for challenge markers ──► If challenge:
                                                  │
                                                  ▼
                                        Wait loop (500ms polls)
                                                  │
                                        Check title changed?
                                                  │
                                        ├── Yes: Challenge passed!
                                        │        Wait 2s for content
                                        │
                                        └── No: Continue waiting
                                                  │
                                        Timeout after --js-wait
```

Challenge markers detected:
- Title: "Just a moment...", "Attention Required", "Access Denied"
- Body: "Verify you are human", "Checking your browser"
- DOM: `#challenge-running`, `.cf-turnstile-wrapper`, `#cf-wrapper`, `[data-cf-turnstile-sitekey]`
- Third-party: `captcha-delivery.com` (DataDome), `perimeterx` (PerimeterX)

---

## How Cloudflare Detects Bots (4 Layers)

### Layer 1: TLS Fingerprinting (JA3/JA4)
- Extracts cipher suites from TLS ClientHello
- Real Chrome = real BoringSSL fingerprint (we pass automatically)
- Only catches non-browser HTTP clients (Python requests, curl)

### Layer 2: HTTP/2 Fingerprinting
- SETTINGS frame values, header order
- Real Chrome handles this automatically
- Catches custom HTTP implementations

### Layer 3: Browser Environment
- `navigator.webdriver`, `navigator.plugins`, `window.chrome`
- Canvas fingerprint hash, WebGL renderer
- Font enumeration, screen dimensions, audio context
- **Our 17 evasions target this layer**

### Layer 4: CDP/Automation Detection (hardest)
- Runtime.enable detection via Error.stack getter
- sourceURL leak from injected scripts
- $cdc_ property on document (chromedriver)
- MutationObserver watching for automation DOM changes
- Behavioral analysis (mouse movement, click timing)

**What blocks us on enterprise Turnstile**: Layer 4 (CDP artifacts from headless_chrome crate). The Rust `headless_chrome` library sends `Runtime.enable` which can be detected. Solutions would require either `rebrowser-patches` approach (isolated worlds) or using `--remote-debugging-pipe` transport.

---

## Test Results

### Fingerprint Tests

| Test Site | Result | Notes |
|-----------|--------|-------|
| bot.sannysoft.com | **0 FAILs** | All fingerprint tests pass |
| nowsecure.nl | **PASS** | nodriver test page |

### Cloudflare Sites by Protection Tier

| Site | Tier | Result | Nodes | Time |
|------|------|--------|-------|------|
| Discord | MEDIUM | **PASS** | 770 | 6s |
| Medium | MEDIUM | **PASS** | 65 | 5s |
| Canva | MEDIUM | **PASS** | 1,399 | 6s |
| Spotify | MEDIUM | **PASS** | 1,979 | 9s |
| ChatGPT | MED-HIGH | **PASS** | 111 | 7s |
| Nike | HIGH | **PASS** | 2,104 | 10s |
| Indeed | HIGH | **PASS** | 409 | 6s |
| Glassdoor | HIGH | **PASS** | 894 | 8s |
| StackOverflow | ENTERPRISE | **BLOCK** | 27 | 34s |
| DoorDash | ENTERPRISE | **BLOCK** | 27 | 24s |
| GameStop | ENTERPRISE | **BLOCK** | 27 | 24s |

**Pass rate: 10/13 (77%) across all Cloudflare sites tested, including HIGH-tier.**

### SPA Sites (non-Cloudflare)

| Site | Static | With JS | Improvement |
|------|--------|---------|-------------|
| Twitter/X | 8 | 62 | 7.8x |
| YouTube | 40 | 403 | 10x |
| Reddit | 257 | 1,573 | 6.1x |
| Instagram | 113 | 108 | ~1x (auth-gated) |
| WhatsApp | 30 | 37 | ~1x (auth-gated) |

---

## What Still Blocks Us

Three Cloudflare Turnstile enterprise sites remain blocked. The detection signals are:

1. **CDP Runtime.enable** — `headless_chrome` sends this command; Cloudflare detects it via Error.stack getter timing (partially mitigated by V8 change in May 2025, but variants exist)

2. **sourceURL leak** — Scripts injected via `Page.addScriptToEvaluateOnNewDocument` contain `//# sourceURL=` comments that appear in stack traces

3. **Behavioral analysis** — No mouse movement or scroll events during challenge wait; enterprise Cloudflare may require proof of human-like interaction

### Potential Solutions (Not Yet Implemented)

| Approach | Complexity | Effectiveness |
|----------|-----------|---------------|
| `rebrowser-patches` isolated worlds | HIGH | Defeats Runtime.enable detection |
| `--remote-debugging-pipe` transport | MEDIUM | Eliminates WebSocket artifacts |
| Chrome binary patching (remove $cdc_) | MEDIUM | Defeats document.$cdc_ scan |
| Mouse/keyboard simulation | HIGH | Defeats behavioral analysis |
| Managed browser API (Browserless.io) | LOW | Outsources entire detection problem |

---

## CLI Usage

```bash
# Static extraction (fast, no JS)
octoview view https://amazon.com

# Force JS rendering
octoview view https://x.com --js

# Auto-detect: static first, JS fallback if needed
octoview view https://x.com --auto-js

# Longer wait for slow SPAs or Cloudflare
octoview view https://nike.com --js --js-wait 30

# Query with JS rendering
octoview query https://youtube.com "SELECT text, href FROM nodes WHERE type = 'link'" --js
```

---

## References

- [undetected-chromedriver](https://github.com/ultrafunkamsterdam/undetected-chromedriver) — Chrome binary patching approach
- [puppeteer-extra-plugin-stealth](https://github.com/berstend/puppeteer-extra/tree/master/packages/puppeteer-extra-plugin-stealth) — 17 evasion modules
- [rebrowser-patches](https://github.com/rebrowser/rebrowser-patches) — Runtime.enable bypass via isolated worlds
- [rebrowser-bot-detector](https://github.com/rebrowser/rebrowser-bot-detector) — 10-point CDP detection test
- [Antoine Vastel: New headless Chrome fingerprint](https://antoinevastel.com/bot%20detection/2023/02/19/new-headless-chrome.html)
- [Cloudflare JA3/JA4 fingerprinting](https://developers.cloudflare.com/bots/additional-configurations/ja3-ja4-fingerprint/)
- [Castle.io: CDP detection evolution](https://blog.castle.io/from-puppeteer-stealth-to-nodriver-how-anti-detect-frameworks-evolved-to-evade-bot-detection/)
