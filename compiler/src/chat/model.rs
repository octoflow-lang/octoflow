//! Model discovery and cache management for octoflow chat.
//!
//! Search order for GGUF model files:
//! 1. Explicit --model flag
//! 2. models/*.gguf in current directory
//! 3. ~/.octoflow/models/*.gguf
//! 4. Print download instructions if not found

use std::path::{Path, PathBuf};

/// Search for a GGUF model file in standard locations.
/// Returns the path to the first model found, or None.
pub fn find_model() -> Option<PathBuf> {
    // 1. Check models/ in current directory
    if let Some(p) = find_in_dir("models") {
        return Some(p);
    }

    // 2. Check ~/.octoflow/models/
    if let Some(home) = home_dir() {
        let octoflow_models = home.join(".octoflow").join("models");
        if let Some(p) = find_in_dir(octoflow_models.to_str().unwrap_or("")) {
            return Some(p);
        }
    }

    None
}

/// Search a directory for .gguf files.
fn find_in_dir(dir: &str) -> Option<PathBuf> {
    let path = Path::new(dir);
    if !path.is_dir() {
        return None;
    }
    let entries = std::fs::read_dir(path).ok()?;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) == Some("gguf") {
            return Some(p);
        }
    }
    None
}

/// Get the user's home directory.
fn home_dir() -> Option<PathBuf> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()
        .map(PathBuf::from)
}

/// Print download instructions when no model is found.
pub fn print_download_instructions() {
    eprintln!("No GGUF model found. To use octoflow chat, download a model:");
    eprintln!();
    eprintln!("  Recommended (1.26 GB, best quality/size balance):");
    eprintln!("  mkdir models");
    eprintln!("  curl -L -o models/qwen3-1.7b-q5_k_m.gguf \\");
    eprintln!("    https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q5_K_M.gguf");
    eprintln!();
    eprintln!("Then run: octoflow chat");
}
