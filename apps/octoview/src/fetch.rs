//! HTTP fetch layer — retrieve web pages for transpilation.

use std::time::Instant;

#[allow(dead_code)]
pub struct FetchResult {
    pub html: String,
    pub url: String,
    pub status: u16,
    pub content_type: String,
    pub elapsed_ms: u128,
}

/// Fetch a URL and return the HTML body.
pub fn fetch_url(url: &str) -> Result<FetchResult, String> {
    let start = Instant::now();

    let client = reqwest::blocking::Client::builder()
        .user_agent("OctoView/0.1 (GPU-native browser; +https://octoflow.dev)")
        .timeout(std::time::Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| format!("HTTP client error: {}", e))?;

    let response = client
        .get(url)
        .header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
        .header("Accept-Language", "en-US,en;q=0.9")
        .send()
        .map_err(|e| format!("fetch error: {}", e))?;

    let status = response.status().as_u16();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    let final_url = response.url().to_string();

    let html = response
        .text()
        .map_err(|e| format!("body read error: {}", e))?;

    let elapsed_ms = start.elapsed().as_millis();

    Ok(FetchResult {
        html,
        url: final_url,
        status,
        content_type,
        elapsed_ms,
    })
}

/// Read HTML from a local file.
pub fn read_file(path: &str) -> Result<FetchResult, String> {
    let start = Instant::now();
    let html = std::fs::read_to_string(path)
        .map_err(|e| format!("file read error: {}", e))?;
    let elapsed_ms = start.elapsed().as_millis();

    Ok(FetchResult {
        html,
        url: format!("file://{}", path),
        status: 200,
        content_type: "text/html".to_string(),
        elapsed_ms,
    })
}
