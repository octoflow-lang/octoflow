//! OVD cache — pre-transpiled pages for instant return visits.
//!
//! Cache location: ~/.octoview/cache/
//! Key: xxhash64 of URL → filename

use crate::ovd::{read_ovd, write_ovd, OvdDocument};
use xxhash_rust::xxh64;

/// Get the cache directory path.
fn cache_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    let dir = std::path::PathBuf::from(home).join(".octoview").join("cache");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Compute cache key from URL.
fn cache_key(url: &str) -> String {
    let hash = xxh64::xxh64(url.as_bytes(), 0);
    format!("{:016x}.ovd", hash)
}

/// Check if a cached .ovd exists and is fresh (within max_age_secs).
pub fn cache_lookup(url: &str, max_age_secs: u64) -> Option<OvdDocument> {
    let path = cache_dir().join(cache_key(url));
    if !path.exists() {
        return None;
    }

    // Check file age
    if let Ok(metadata) = std::fs::metadata(&path) {
        if let Ok(modified) = metadata.modified() {
            let age = std::time::SystemTime::now()
                .duration_since(modified)
                .unwrap_or_default();
            if age.as_secs() > max_age_secs {
                return None; // Stale
            }
        }
    }

    read_ovd(path.to_str()?).ok()
}

/// Store an OVD document in the cache.
pub fn cache_store(doc: &OvdDocument) -> Result<(), String> {
    let path = cache_dir().join(cache_key(&doc.url));
    write_ovd(doc, path.to_str().unwrap_or("cache.ovd"))
        .map_err(|e| format!("cache write error: {}", e))
}

/// Clear all cached .ovd files.
pub fn cache_clear() -> Result<usize, String> {
    let dir = cache_dir();
    let mut count = 0;
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            if entry.path().extension().map(|e| e == "ovd").unwrap_or(false) {
                let _ = std::fs::remove_file(entry.path());
                count += 1;
            }
        }
    }
    Ok(count)
}

/// Get cache statistics.
pub fn cache_stats() -> (usize, u64) {
    let dir = cache_dir();
    let mut count = 0;
    let mut total_bytes = 0u64;
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            if entry.path().extension().map(|e| e == "ovd").unwrap_or(false) {
                count += 1;
                total_bytes += entry.metadata().map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    (count, total_bytes)
}
