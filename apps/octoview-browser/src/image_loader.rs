/// Image fetching, decoding, and caching for the browser.
/// Supports PNG and JPEG images. Resizes to fit viewport width.

use std::collections::HashMap;

pub struct DecodedImage {
    pub pixels: Vec<u8>, // RGBA
    pub width: u32,
    pub height: u32,
}

pub struct ImageCache {
    cache: HashMap<String, Option<DecodedImage>>,
}

impl ImageCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Fetch and decode an image URL. Returns None on failure.
    /// Caches results (including failures as None).
    pub fn get_or_fetch(&mut self, url: &str, max_width: u32) -> Option<&DecodedImage> {
        if !self.cache.contains_key(url) {
            let decoded = fetch_and_decode(url, max_width);
            self.cache.insert(url.to_string(), decoded);
        }
        self.cache.get(url).and_then(|v| v.as_ref())
    }
}

fn fetch_and_decode(url: &str, max_width: u32) -> Option<DecodedImage> {
    // Skip data: URLs and empty src
    if url.is_empty() || url.starts_with("data:") {
        return None;
    }

    let client = reqwest::blocking::Client::builder()
        .user_agent("OctoView/0.2")
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .ok()?;

    let response = client.get(url).send().ok()?;
    if !response.status().is_success() {
        return None;
    }

    let bytes = response.bytes().ok()?;
    // Cap at 5MB
    if bytes.len() > 5_000_000 {
        return None;
    }

    let img = image::load_from_memory(&bytes).ok()?;
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());

    // Resize if wider than max_width (never upscale)
    if w > max_width && max_width > 0 {
        let scale = max_width as f32 / w as f32;
        let new_h = (h as f32 * scale) as u32;
        let resized = image::imageops::resize(
            &rgba,
            max_width,
            new_h,
            image::imageops::FilterType::Triangle,
        );
        Some(DecodedImage {
            pixels: resized.into_raw(),
            width: max_width,
            height: new_h,
        })
    } else {
        Some(DecodedImage {
            pixels: rgba.into_raw(),
            width: w,
            height: h,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_cache_empty() {
        let mut cache = ImageCache::new();
        assert!(cache.get_or_fetch("", 800).is_none());
    }

    #[test]
    fn test_image_cache_data_url_skipped() {
        let mut cache = ImageCache::new();
        assert!(
            cache
                .get_or_fetch("data:image/png;base64,abc", 800)
                .is_none()
        );
    }

    #[test]
    fn test_image_cache_clear() {
        let mut cache = ImageCache::new();
        cache.get_or_fetch("", 800);
        assert_eq!(cache.cache.len(), 1);
        cache.clear();
        assert_eq!(cache.cache.len(), 0);
    }
}
