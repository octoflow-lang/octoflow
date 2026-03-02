use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookmarkEntry {
    pub url: String,
    pub title: String,
    pub added_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub url: String,
    pub title: String,
    pub visited_at: u64,
}

pub struct BookmarkStore {
    pub bookmarks: Vec<BookmarkEntry>,
    pub history: Vec<HistoryEntry>,
    data_dir: PathBuf,
}

const MAX_HISTORY: usize = 1000;

impl BookmarkStore {
    pub fn new() -> Self {
        let data_dir = dirs_path();
        std::fs::create_dir_all(&data_dir).ok();
        let mut store = Self {
            bookmarks: Vec::new(),
            history: Vec::new(),
            data_dir,
        };
        store.load();
        store
    }

    /// Create a store with a custom data directory (for testing).
    #[allow(dead_code)]
    pub fn with_dir(data_dir: PathBuf) -> Self {
        Self {
            bookmarks: Vec::new(),
            history: Vec::new(),
            data_dir,
        }
    }

    fn bookmarks_path(&self) -> PathBuf {
        self.data_dir.join("bookmarks.json")
    }

    fn history_path(&self) -> PathBuf {
        self.data_dir.join("history.json")
    }

    fn load(&mut self) {
        if let Ok(data) = std::fs::read_to_string(self.bookmarks_path()) {
            self.bookmarks = serde_json::from_str(&data).unwrap_or_default();
        }
        if let Ok(data) = std::fs::read_to_string(self.history_path()) {
            self.history = serde_json::from_str(&data).unwrap_or_default();
        }
    }

    pub fn save(&self) {
        if let Ok(json) = serde_json::to_string_pretty(&self.bookmarks) {
            std::fs::write(self.bookmarks_path(), json).ok();
        }
        if let Ok(json) = serde_json::to_string_pretty(&self.history) {
            std::fs::write(self.history_path(), json).ok();
        }
    }

    pub fn add_bookmark(&mut self, url: &str, title: &str) {
        if self.bookmarks.iter().any(|b| b.url == url) {
            return;
        }
        self.bookmarks.push(BookmarkEntry {
            url: url.to_string(),
            title: title.to_string(),
            added_at: now_unix(),
        });
        self.save();
    }

    #[allow(dead_code)]
    pub fn remove_bookmark(&mut self, idx: usize) {
        if idx < self.bookmarks.len() {
            self.bookmarks.remove(idx);
            self.save();
        }
    }

    #[allow(dead_code)]
    pub fn is_bookmarked(&self, url: &str) -> bool {
        self.bookmarks.iter().any(|b| b.url == url)
    }

    pub fn add_history(&mut self, url: &str, title: &str) {
        self.history.push(HistoryEntry {
            url: url.to_string(),
            title: title.to_string(),
            visited_at: now_unix(),
        });
        if self.history.len() > MAX_HISTORY {
            self.history.drain(0..self.history.len() - MAX_HISTORY);
        }
        self.save();
    }

    #[allow(dead_code)]
    pub fn remove_history(&mut self, idx: usize) {
        if idx < self.history.len() {
            self.history.remove(idx);
            self.save();
        }
    }
}

fn dirs_path() -> PathBuf {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".octoview")
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bookmark_add_remove() {
        let dir = std::env::temp_dir().join("octoview_test_bm");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).ok();
        let mut store = BookmarkStore::with_dir(dir);
        store.add_bookmark("https://example.com", "Example");
        assert_eq!(store.bookmarks.len(), 1);
        assert!(store.is_bookmarked("https://example.com"));
        // No duplicate
        store.add_bookmark("https://example.com", "Example");
        assert_eq!(store.bookmarks.len(), 1);
        store.remove_bookmark(0);
        assert_eq!(store.bookmarks.len(), 0);
    }

    #[test]
    fn test_history_cap() {
        let dir = std::env::temp_dir().join("octoview_test_hist");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).ok();
        let mut store = BookmarkStore::with_dir(dir);
        for i in 0..1050 {
            store.history.push(HistoryEntry {
                url: format!("https://example.com/{i}"),
                title: format!("Page {i}"),
                visited_at: i as u64,
            });
        }
        if store.history.len() > MAX_HISTORY {
            store.history.drain(0..store.history.len() - MAX_HISTORY);
        }
        assert_eq!(store.history.len(), 1000);
    }
}
