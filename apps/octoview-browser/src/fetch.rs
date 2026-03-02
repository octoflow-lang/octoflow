#[allow(dead_code)]
pub struct FetchResult {
    pub html: String,
    pub url: String,
    pub status: u16,
    pub elapsed_ms: u128,
}

/// Fetch a URL and return the HTML body.
/// Adds https:// if no scheme is present.
pub fn fetch_url(url: &str) -> Result<FetchResult, String> {
    let url = normalize_url(url);
    let start = std::time::Instant::now();

    let client = reqwest::blocking::Client::builder()
        .user_agent("OctoView/0.1")
        .timeout(std::time::Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
        .map_err(|e| format!("HTTP client error: {e}"))?;

    let response = client
        .get(&url)
        .send()
        .map_err(|e| format!("Fetch failed: {e}"))?;

    let status = response.status().as_u16();
    let final_url = response.url().to_string();
    let html = response
        .text()
        .map_err(|e| format!("Failed to read response body: {e}"))?;

    Ok(FetchResult {
        html,
        url: final_url,
        status,
        elapsed_ms: start.elapsed().as_millis(),
    })
}

/// Fetch a CSS stylesheet from a URL.
/// Returns the CSS text on success, or an error string on failure.
/// Uses a 10 second timeout and does not crash on failure.
pub fn fetch_css(url: &str) -> Result<String, String> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("OctoView/0.1")
        .timeout(std::time::Duration::from_secs(2))
        .redirect(reqwest::redirect::Policy::limited(3))
        .build()
        .map_err(|e| format!("CSS fetch client error: {e}"))?;

    let response = client
        .get(url)
        .send()
        .map_err(|e| format!("CSS fetch failed for {url}: {e}"))?;

    let text = response
        .text()
        .map_err(|e| format!("CSS read body failed for {url}: {e}"))?;

    Ok(text)
}

fn normalize_url(url: &str) -> String {
    let url = url.trim();
    if url.starts_with("http://") || url.starts_with("https://") {
        url.to_string()
    } else if url.starts_with("file://") {
        url.to_string()
    } else {
        format!("https://{url}")
    }
}
